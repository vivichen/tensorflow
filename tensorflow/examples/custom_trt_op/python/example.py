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
"""Example demonstrating custom TensorRT op usage in TensorFlow."""

import numpy as np
import tensorflow as tf

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("Warning: TensorRT not available. This example requires TensorRT.")

from tensorflow.examples.custom_trt_op.python import graphdef_utils


def build_tensorrt_engine():
    """Build a simple TensorRT engine.
    
    This creates an engine that performs: output = (input * 2.0) + 1.0
    
    Returns:
        Serialized TensorRT engine
    """
    if not TRT_AVAILABLE:
        raise RuntimeError("TensorRT is required for this example")
    
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # Define input
    input_tensor = network.add_input(
        name="input", dtype=trt.float32, shape=(1, 4))
    
    # Layer 1: Multiply by 2
    scale_layer = network.add_scale(
        input=input_tensor,
        mode=trt.ScaleMode.UNIFORM,
        shift=np.zeros(1, dtype=np.float32),
        scale=np.array([2.0], dtype=np.float32),
        power=np.ones(1, dtype=np.float32)
    )
    
    # Layer 2: Add 1
    add_layer = network.add_scale(
        input=scale_layer.get_output(0),
        mode=trt.ScaleMode.UNIFORM,
        shift=np.array([1.0], dtype=np.float32),
        scale=np.ones(1, dtype=np.float32),
        power=np.ones(1, dtype=np.float32)
    )
    
    # Mark output
    network.mark_output(add_layer.get_output(0))
    add_layer.get_output(0).name = "output"
    
    # Build engine
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    
    print("Building TensorRT engine...")
    serialized_engine = builder.build_serialized_network(network, config)
    print(f"Engine built successfully, size: {len(serialized_engine)} bytes")
    
    return serialized_engine


def create_tensorflow_graph():
    """Create a TensorFlow graph with operations we want to replace with TensorRT.
    
    Returns:
        GraphDef
    """
    with tf.Graph().as_default() as g:
        # Input placeholder
        x = tf.placeholder(tf.float32, shape=[1, 4], name="input")
        
        # Operations to be replaced with TensorRT
        y = tf.multiply(x, 2.0, name="multiply")
        z = tf.add(y, 1.0, name="output")
        
    return g.as_graph_def()


def replace_with_tensorrt():
    """Example: Replace subgraph with TensorRT engine.
    
    This demonstrates the complete workflow:
    1. Build TensorRT engine
    2. Create TensorFlow graph
    3. Replace subgraph with TensorRT node
    """
    # Step 1: Build TensorRT engine
    print("\n=== Step 1: Build TensorRT Engine ===")
    serialized_engine = build_tensorrt_engine()
    
    # Step 2: Create original TensorFlow graph
    print("\n=== Step 2: Create TensorFlow Graph ===")
    original_graph_def = create_tensorflow_graph()
    print(f"Original graph has {len(original_graph_def.node)} nodes:")
    for node in original_graph_def.node:
        print(f"  - {node.name} ({node.op})")
    
    # Step 3: Create TensorRT node
    print("\n=== Step 3: Create TensorRT Node ===")
    trt_node = graphdef_utils.create_trt_node(
        node_name="trt_inference",
        serialized_engine=serialized_engine,
        input_names=["input"],
        output_names=["output"],
        input_types=[tf.float32],
        output_types=[tf.float32],
        input_shapes=[tf.TensorShape([1, 4])],
        output_shapes=[tf.TensorShape([1, 4])]
    )
    print(f"Created TensorRT node: {trt_node.name}")
    
    # Step 4: Replace subgraph
    print("\n=== Step 4: Replace Subgraph ===")
    new_graph_def = graphdef_utils.replace_subgraph_with_trt(
        graph_def=original_graph_def,
        subgraph_nodes=["multiply", "add"],  # Nodes to replace
        trt_node=trt_node,
        input_mappings={"input": "input"},  # Map TRT input to original input
        output_mappings={"output": 0}  # Map original output to TRT output index
    )
    
    print(f"Modified graph has {len(new_graph_def.node)} nodes:")
    for node in new_graph_def.node:
        print(f"  - {node.name} ({node.op})")
    
    # Step 5: Save modified graph
    print("\n=== Step 5: Save Modified Graph ===")
    output_path = "/tmp/graph_with_trt.pb"
    graphdef_utils.save_graphdef_with_trt(new_graph_def, output_path)
    
    return new_graph_def


def test_inference():
    """Test inference with the modified graph."""
    print("\n=== Testing Inference ===")
    
    # Create and modify graph
    new_graph_def = replace_with_tensorrt()
    
    # Note: Actual inference would require:
    # 1. TensorRT kernel to be compiled and loaded
    # 2. GPU with TensorRT installed
    # 3. Proper CUDA context
    
    print("\nTo run inference:")
    print("1. Build the custom op: bazel build //tensorflow/examples/custom_trt_op:custom_trt_kernels")
    print("2. Load the graph and run inference on GPU")
    print("3. Input: [[1.0, 2.0, 3.0, 4.0]]")
    print("4. Expected output: [[3.0, 5.0, 7.0, 9.0]]  (input * 2 + 1)")


def main():
    """Main function."""
    print("=" * 80)
    print("Custom TensorRT Op Example")
    print("=" * 80)
    
    if not TRT_AVAILABLE:
        print("\nError: TensorRT is not available.")
        print("Please install TensorRT to run this example.")
        return
    
    if not tf.config.list_physical_devices('GPU'):
        print("\nWarning: No GPU detected.")
        print("TensorRT operations require a GPU to execute.")
    
    # Run the example
    test_inference()
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
