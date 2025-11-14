# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""GraphDef modification utilities for inserting TensorRT ops."""

import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import attr_value_pb2


def create_trt_node(node_name, serialized_engine, input_names, output_names,
                   input_types, output_types, input_shapes, output_shapes,
                   workspace_size_bytes=1073741824):
    """Create a TensorRT node definition.
    
    Args:
        node_name: Name for the TensorRT node
        serialized_engine: Serialized TensorRT engine (bytes)
        input_names: List of input tensor names
        output_names: List of output tensor names
        input_types: List of TensorFlow DataTypes for inputs
        output_types: List of TensorFlow DataTypes for outputs
        input_shapes: List of TensorShapes for inputs
        output_shapes: List of TensorShapes for outputs
        workspace_size_bytes: TensorRT workspace size
        
    Returns:
        NodeDef for the TensorRT operation
    """
    node = node_def_pb2.NodeDef()
    node.name = node_name
    node.op = "CustomTRTInferenceOp"
    
    # Set inputs
    for input_name in input_names:
        node.input.append(input_name)
    
    # Set attributes
    node.attr["serialized_engine"].s = serialized_engine
    node.attr["input_names"].list.s.extend([n.encode() for n in input_names])
    node.attr["output_names"].list.s.extend([n.encode() for n in output_names])
    
    # Set type lists
    node.attr["InT"].list.type.extend([t.as_datatype_enum for t in input_types])
    node.attr["OutT"].list.type.extend([t.as_datatype_enum for t in output_types])
    
    # Set shape lists
    for shape in input_shapes:
        shape_proto = node.attr["input_shapes"].list.shape.add()
        for dim in shape.as_list():
            if dim is None:
                shape_proto.dim.add().size = -1
            else:
                shape_proto.dim.add().size = dim
    
    for shape in output_shapes:
        shape_proto = node.attr["output_shapes"].list.shape.add()
        for dim in shape.as_list():
            if dim is None:
                shape_proto.dim.add().size = -1
            else:
                shape_proto.dim.add().size = dim
    
    node.attr["workspace_size_bytes"].i = workspace_size_bytes
    
    return node


def replace_subgraph_with_trt(graph_def, subgraph_nodes, trt_node, 
                              input_mappings, output_mappings):
    """Replace a subgraph with a TensorRT node in a GraphDef.
    
    Args:
        graph_def: Input GraphDef
        subgraph_nodes: List of node names to replace
        trt_node: TensorRT NodeDef to insert
        input_mappings: Dict mapping TRT input names to original tensor names
        output_mappings: Dict mapping original output names to TRT output indices
        
    Returns:
        Modified GraphDef with subgraph replaced by TensorRT node
        
    Example:
        ```python
        # Original graph has nodes: Input -> MatMul -> Add -> Output
        # We want to replace MatMul and Add with TensorRT
        
        graph_def = ...  # Load your graph
        
        # Create TensorRT engine for MatMul + Add
        serialized_engine = create_engine_for_matmul_add()
        
        # Create TRT node
        trt_node = create_trt_node(
            node_name="trt_matmul_add",
            serialized_engine=serialized_engine,
            input_names=["input_tensor"],
            output_names=["output_tensor"],
            input_types=[tf.float32],
            output_types=[tf.float32],
            input_shapes=[tf.TensorShape([None, 784])],
            output_shapes=[tf.TensorShape([None, 10])]
        )
        
        # Replace subgraph
        new_graph_def = replace_subgraph_with_trt(
            graph_def=graph_def,
            subgraph_nodes=["MatMul", "Add"],
            trt_node=trt_node,
            input_mappings={"input_tensor": "Input"},
            output_mappings={"Output": 0}
        )
        ```
    """
    new_graph = graph_pb2.GraphDef()
    new_graph.CopyFrom(graph_def)
    
    # Remove subgraph nodes
    nodes_to_remove = set(subgraph_nodes)
    new_graph.node[:] = [node for node in new_graph.node 
                         if node.name not in nodes_to_remove]
    
    # Update TRT node inputs with actual tensor names
    for i, input_name in enumerate(trt_node.input):
        if input_name in input_mappings:
            trt_node.input[i] = input_mappings[input_name]
    
    # Add TRT node
    new_graph.node.append(trt_node)
    
    # Update downstream nodes to use TRT outputs
    for node in new_graph.node:
        for i, input_name in enumerate(node.input):
            # Check if this input was an output of the removed subgraph
            if input_name in output_mappings:
                # Replace with TRT node output
                output_idx = output_mappings[input_name]
                node.input[i] = f"{trt_node.name}:{output_idx}"
    
    return new_graph


def save_graphdef_with_trt(graph_def, output_path):
    """Save a GraphDef to a file.
    
    Args:
        graph_def: GraphDef to save
        output_path: Path to save the GraphDef
    """
    with tf.io.gfile.GFile(output_path, 'wb') as f:
        f.write(graph_def.SerializeToString())
    print(f"Saved GraphDef to {output_path}")


def load_graphdef(input_path):
    """Load a GraphDef from a file.
    
    Args:
        input_path: Path to the GraphDef file
        
    Returns:
        Loaded GraphDef
    """
    with tf.io.gfile.GFile(input_path, 'rb') as f:
        graph_def = graph_pb2.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def
