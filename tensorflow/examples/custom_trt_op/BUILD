# Custom TensorRT Op BUILD configuration

load("//tensorflow:tensorflow.bzl", "tf_custom_op_library", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.default.bzl", "tf_cuda_library")
load("@local_config_tensorrt//:build_defs.bzl", "if_tensorrt")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Op definition library
cc_library(
    name = "custom_trt_ops",
    srcs = ["ops/custom_trt_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

# Kernel implementation library (GPU + TensorRT only)
tf_cuda_library(
    name = "custom_trt_kernels",
    srcs = if_tensorrt(["kernels/custom_trt_kernel.cc"]),
    deps = if_tensorrt([
        ":custom_trt_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_tensorrt//:tensorrt_headers",
    ]),
    alwayslink = 1,
)

# Generate Python wrapper
tf_gen_op_wrapper_py(
    name = "gen_custom_trt_ops",
    out = "python/gen_custom_trt_ops.py",
    deps = [":custom_trt_ops"],
)

# Python library
py_library(
    name = "custom_trt_ops_py",
    srcs = [
        "python/__init__.py",
        "python/custom_trt_ops.py",
        "python/gen_custom_trt_ops.py",
    ],
    data = if_tensorrt([
        ":custom_trt_kernels",
    ]),
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
    ],
)

# GraphDef utilities library
py_library(
    name = "graphdef_utils",
    srcs = ["python/graphdef_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
)

# Tests
py_test(
    name = "custom_trt_ops_test",
    size = "small",
    srcs = ["python/custom_trt_ops_test.py"],
    python_version = "PY3",
    tags = ["no_pip", "requires-gpu-nvidia"],
    deps = [
        ":custom_trt_ops_py",
        ":graphdef_utils",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
    ],
)

# Example script
py_binary(
    name = "example",
    srcs = ["python/example.py"],
    python_version = "PY3",
    deps = [
        ":custom_trt_ops_py",
        ":graphdef_utils",
        "//tensorflow/python:framework",
    ],
)
