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
"""Tests for custom TensorRT operations."""

import os
import numpy as np
import tensorflow as tf

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False


class CustomTRTOpsTest(tf.test.TestCase):
    """Test custom TensorRT operations."""

    def setUp(self):
        super().setUp()
        if not TRT_AVAILABLE:
            self.skipTest("TensorRT not available")
        if not tf.config.list_physical_devices('GPU'):
            self.skipTest("GPU not available")

    def _build_simple_engine(self):
        """Build a simple TensorRT engine for testing.
        
        Creates an engine that does: output = input * 2.0
        """
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        
        # Input tensor
        input_tensor = network.add_input(
            name="input", dtype=trt.float32, shape=(1, 4))
        
        # Simple operation: multiply by 2
        scale_layer = network.add_scale(
            input=input_tensor,
            mode=trt.ScaleMode.UNIFORM,
            shift=np.zeros(1, dtype=np.float32),
            scale=np.array([2.0], dtype=np.float32),
            power=np.ones(1, dtype=np.float32)
        )
        
        # Mark output
        network.mark_output(scale_layer.get_output(0))
        scale_layer.get_output(0).name = "output"
        
        # Build engine
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
        
        serialized_engine = builder.build_serialized_network(network, config)
        return serialized_engine

    def test_graphdef_modification(self):
        """Test GraphDef modification utilities."""
        from tensorflow.examples.custom_trt_op.python import graphdef_utils
        
        # Create a simple graph
        with tf.Graph().as_default() as g:
            x = tf.constant([[1.0, 2.0, 3.0, 4.0]], name="input")
            y = tf.multiply(x, 2.0, name="multiply")
            z = tf.add(y, 1.0, name="add")
            
        graph_def = g.as_graph_def()
        
        # Build TensorRT engine
        serialized_engine = self._build_simple_engine()
        
        # Create TRT node
        trt_node = graphdef_utils.create_trt_node(
            node_name="trt_node",
            serialized_engine=serialized_engine,
            input_names=["input"],
            output_names=["output"],
            input_types=[tf.float32],
            output_types=[tf.float32],
            input_shapes=[tf.TensorShape([1, 4])],
            output_shapes=[tf.TensorShape([1, 4])]
        )
        
        # Verify node was created
        self.assertEqual(trt_node.name, "trt_node")
        self.assertEqual(trt_node.op, "CustomTRTInferenceOp")

    def test_engine_serialization(self):
        """Test TensorRT engine can be serialized and used."""
        serialized_engine = self._build_simple_engine()
        
        # Verify engine is not empty
        self.assertIsNotNone(serialized_engine)
        self.assertGreater(len(serialized_engine), 0)
        
        # Verify it can be deserialized
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.assertIsNotNone(engine)


if __name__ == '__main__':
    tf.test.main()
