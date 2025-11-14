/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Custom TensorRT Op Registration
// This is a simplified example showing how to create a custom op for TensorRT inference

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

// Custom TensorRT inference op
// This op takes serialized TensorRT engine and input tensors, 
// executes the engine and returns output tensors
REGISTER_OP("CustomTRTInferenceOp")
    .Attr("serialized_engine: string")
    .Attr("input_names: list(string)")
    .Attr("output_names: list(string)")
    .Attr("InT: list({float32, float16, int32, int8})")
    .Attr("OutT: list({float32, float16, int32, int8})")
    .Attr("input_shapes: list(shape) = []")
    .Attr("output_shapes: list(shape) = []")
    .Attr("workspace_size_bytes: int = 1073741824")  // 1GB default
    .Input("inputs: InT")
    .Output("outputs: OutT")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      std::vector<tensorflow::PartialTensorShape> output_shapes;
      TF_RETURN_IF_ERROR(c->GetAttr("output_shapes", &output_shapes));

      if (output_shapes.size() != c->num_outputs()) {
        return errors::InvalidArgument(
            "Number of output_shapes (", output_shapes.size(),
            ") must match number of outputs (", c->num_outputs(), ")");
      }

      for (int i = 0; i < output_shapes.size(); i++) {
        shape_inference::ShapeHandle output_shape_handle;
        TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(
            output_shapes[i], &output_shape_handle));
        c->set_output(i, output_shape_handle);
      }

      return OkStatus();
    })
    .Doc(R"doc(
Custom TensorRT Inference Operation.

This op executes a TensorRT engine for inference. It takes serialized TensorRT
engine as an attribute and input tensors, then returns output tensors.

serialized_engine: Serialized TensorRT engine (binary string)
input_names: Names of input tensors in TensorRT engine
output_names: Names of output tensors in TensorRT engine
InT: Input data types
OutT: Output data types
input_shapes: Shapes of input tensors
output_shapes: Shapes of output tensors
workspace_size_bytes: Size of workspace for TensorRT execution
inputs: Input tensors
outputs: Output tensors
)doc");

}  // namespace tensorflow
