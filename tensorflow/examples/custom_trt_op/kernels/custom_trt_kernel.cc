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

// Custom TensorRT Op Kernel Implementation
// This implements the actual execution logic for the custom TensorRT op

#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {

// Helper class to manage TensorRT logger
class CustomTRTLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    switch (severity) {
      case Severity::kINTERNAL_ERROR:
      case Severity::kERROR:
        LOG(ERROR) << "TRT Error: " << msg;
        break;
      case Severity::kWARNING:
        LOG(WARNING) << "TRT Warning: " << msg;
        break;
      case Severity::kINFO:
        VLOG(1) << "TRT Info: " << msg;
        break;
      case Severity::kVERBOSE:
        VLOG(2) << "TRT Verbose: " << msg;
        break;
    }
  }
};

// Kernel implementation
class CustomTRTInferenceOp : public OpKernel {
 public:
  explicit CustomTRTInferenceOp(OpKernelConstruction* context)
      : OpKernel(context) {
    // Get attributes
    OP_REQUIRES_OK(context, context->GetAttr("serialized_engine", &serialized_engine_));
    OP_REQUIRES_OK(context, context->GetAttr("input_names", &input_names_));
    OP_REQUIRES_OK(context, context->GetAttr("output_names", &output_names_));
    OP_REQUIRES_OK(context, context->GetAttr("workspace_size_bytes", &workspace_size_bytes_));
    
    OP_REQUIRES(context, !serialized_engine_.empty(),
                errors::InvalidArgument("serialized_engine cannot be empty"));
    
    // Initialize TensorRT runtime and engine
    InitializeEngine(context);
  }

  void Compute(OpKernelContext* context) override {
    // Verify we're on GPU
    OP_REQUIRES(context, context->device_type() == DEVICE_GPU,
                errors::InvalidArgument("CustomTRTInferenceOp must run on GPU"));

    if (!engine_ || !execution_context_) {
      context->SetStatus(errors::Internal("TensorRT engine not initialized"));
      return;
    }

    // Get CUDA stream
    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream != nullptr,
                errors::Internal("No GPU stream available"));
    cudaStream_t cuda_stream = stream->platform_specific_handle().stream;

    // Prepare bindings for TensorRT
    std::vector<void*> bindings(input_names_.size() + output_names_.size());
    
    // Set input bindings
    for (int i = 0; i < input_names_.size(); ++i) {
      const Tensor& input_tensor = context->input(i);
      bindings[i] = const_cast<void*>(
          static_cast<const void*>(input_tensor.tensor_data().data()));
    }

    // Allocate and set output bindings
    for (int i = 0; i < output_names_.size(); ++i) {
      Tensor* output_tensor = nullptr;
      TensorShape output_shape;
      
      // Get output shape from engine
      int binding_index = input_names_.size() + i;
      auto dims = engine_->getBindingDimensions(binding_index);
      
      for (int d = 0; d < dims.nbDims; ++d) {
        output_shape.AddDim(dims.d[d]);
      }

      OP_REQUIRES_OK(context, 
                     context->allocate_output(i, output_shape, &output_tensor));
      
      bindings[binding_index] = 
          const_cast<void*>(static_cast<const void*>(
              output_tensor->tensor_data().data()));
    }

    // Execute inference
    bool success = execution_context_->enqueueV2(bindings.data(), cuda_stream, nullptr);
    
    OP_REQUIRES(context, success,
                errors::Internal("TensorRT inference execution failed"));
  }

 private:
  void InitializeEngine(OpKernelConstruction* context) {
    static CustomTRTLogger logger;
    
    // Create TensorRT runtime
    runtime_.reset(nvinfer1::createInferRuntime(logger));
    OP_REQUIRES(context, runtime_ != nullptr,
                errors::Internal("Failed to create TensorRT runtime"));

    // Deserialize engine
    engine_.reset(runtime_->deserializeCudaEngine(
        serialized_engine_.data(), serialized_engine_.size()));
    OP_REQUIRES(context, engine_ != nullptr,
                errors::InvalidArgument("Failed to deserialize TensorRT engine"));

    // Create execution context
    execution_context_.reset(engine_->createExecutionContext());
    OP_REQUIRES(context, execution_context_ != nullptr,
                errors::Internal("Failed to create TensorRT execution context"));
  }

  std::string serialized_engine_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  int64_t workspace_size_bytes_;
  
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> execution_context_;
};

REGISTER_KERNEL_BUILDER(
    Name("CustomTRTInferenceOp").Device(DEVICE_GPU),
    CustomTRTInferenceOp);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
