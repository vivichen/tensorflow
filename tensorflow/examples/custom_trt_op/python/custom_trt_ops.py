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
"""Custom TensorRT operations for TensorFlow inference."""

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

# Load the custom op library
try:
    _custom_trt_ops = load_library.load_op_library(
        resource_loader.get_path_to_datafile('_custom_trt_kernels.so'))
except Exception as e:
    # Gracefully handle when TensorRT is not available
    _custom_trt_ops = None
    import warnings
    warnings.warn(f"Could not load custom TensorRT ops: {e}")


def custom_trt_inference(inputs, serialized_engine, input_names, output_names,
                         output_types, output_shapes, workspace_size_bytes=1073741824,
                         name=None):
    """Execute TensorRT inference with a serialized engine.
    
    Args:
        inputs: List of input tensors
        serialized_engine: Serialized TensorRT engine (bytes)
        input_names: List of input tensor names in the TensorRT engine
        output_names: List of output tensor names in the TensorRT engine
        output_types: List of output data types (tf.DType)
        output_shapes: List of output tensor shapes
        workspace_size_bytes: Workspace size for TensorRT (default: 1GB)
        name: Optional operation name
        
    Returns:
        List of output tensors from TensorRT inference
        
    Example:
        ```python
        import tensorflow as tf
        from tensorflow.examples.custom_trt_op.python import custom_trt_ops
        
        # Load serialized TensorRT engine
        with open('model.engine', 'rb') as f:
            serialized_engine = f.read()
        
        # Prepare inputs
        input_tensor = tf.constant([[1.0, 2.0, 3.0, 4.0]])
        
        # Run inference
        outputs = custom_trt_ops.custom_trt_inference(
            inputs=[input_tensor],
            serialized_engine=serialized_engine,
            input_names=['input'],
            output_names=['output'],
            output_types=[tf.float32],
            output_shapes=[tf.TensorShape([1, 4])]
        )
        ```
    """
    if _custom_trt_ops is None:
        raise RuntimeError("Custom TensorRT ops not available. "
                         "Make sure TensorRT is installed and TensorFlow is built with GPU support.")
    
    input_types = [inp.dtype for inp in inputs]
    
    return _custom_trt_ops.custom_trt_inference_op(
        inputs=inputs,
        serialized_engine=serialized_engine,
        input_names=input_names,
        output_names=output_names,
        InT=input_types,
        OutT=output_types,
        input_shapes=[inp.shape for inp in inputs],
        output_shapes=output_shapes,
        workspace_size_bytes=workspace_size_bytes,
        name=name
    )
