# Custom TensorRT Op for TensorFlow

[English](#english) | [中文](#chinese)

<a name="english"></a>
## English

This example demonstrates how to create a custom TensorRT operation for TensorFlow inference framework, allowing you to run TensorRT engines directly within TensorFlow graphs.

### Overview

This implementation provides:
1. **Custom Op Registration** - C++ op definition for TensorRT inference
2. **Kernel Implementation** - GPU kernel that executes TensorRT engines
3. **GraphDef Modification Utilities** - Python tools to replace subgraphs with TensorRT ops
4. **Testing Framework** - Examples and tests for validation

### Features

- ✅ Minimal changes to TensorFlow codebase
- ✅ Compatible with latest TensorFlow version
- ✅ Direct TensorRT engine execution in TensorFlow graphs
- ✅ Easy GraphDef modification
- ✅ Comprehensive examples and tests

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TensorFlow Graph                          │
│  ┌──────────┐      ┌──────────────────┐      ┌──────────┐  │
│  │  Input   │─────▶│ CustomTRTInference│─────▶│  Output  │  │
│  │  Tensor  │      │      Op           │      │  Tensor  │  │
│  └──────────┘      └──────────────────┘      └──────────┘  │
│                            │                                 │
│                            ▼                                 │
│                    ┌──────────────┐                         │
│                    │ TensorRT     │                         │
│                    │ Engine       │                         │
│                    │ (Serialized) │                         │
│                    └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

### File Structure

```
tensorflow/examples/custom_trt_op/
├── ops/
│   └── custom_trt_op.cc          # Op registration
├── kernels/
│   └── custom_trt_kernel.cc      # Kernel implementation
├── python/
│   ├── __init__.py               # Python package
│   ├── custom_trt_ops.py         # Python wrapper
│   ├── graphdef_utils.py         # GraphDef utilities
│   ├── example.py                # Usage example
│   └── custom_trt_ops_test.py    # Tests
├── BUILD                          # Bazel build config
└── README.md                      # This file
```

### Prerequisites

- TensorFlow (latest version)
- CUDA toolkit
- TensorRT 8.x or later
- GPU with compute capability 6.0+

### Building

```bash
# Build the custom op library
bazel build //tensorflow/examples/custom_trt_op:custom_trt_kernels

# Build Python wrapper
bazel build //tensorflow/examples/custom_trt_op:custom_trt_ops_py
```

### Usage

#### 1. Op Implementation

The custom op is defined in `ops/custom_trt_op.cc`:

```cpp
REGISTER_OP("CustomTRTInferenceOp")
    .Attr("serialized_engine: string")
    .Attr("input_names: list(string)")
    .Attr("output_names: list(string)")
    .Input("inputs: InT")
    .Output("outputs: OutT")
    // ... more attributes
```

The kernel implementation in `kernels/custom_trt_kernel.cc` handles:
- Deserializing TensorRT engine
- Managing execution context
- Binding input/output tensors
- Executing inference on GPU

#### 2. GraphDef Modification

Use the provided utilities to replace subgraphs with TensorRT:

```python
from tensorflow.examples.custom_trt_op.python import graphdef_utils
import tensorflow as tf

# Load or create your graph
graph_def = ...

# Build TensorRT engine for your subgraph
serialized_engine = build_your_trt_engine()

# Create TRT node
trt_node = graphdef_utils.create_trt_node(
    node_name="trt_inference",
    serialized_engine=serialized_engine,
    input_names=["input"],
    output_names=["output"],
    input_types=[tf.float32],
    output_types=[tf.float32],
    input_shapes=[tf.TensorShape([1, 784])],
    output_shapes=[tf.TensorShape([1, 10])]
)

# Replace subgraph
new_graph_def = graphdef_utils.replace_subgraph_with_trt(
    graph_def=graph_def,
    subgraph_nodes=["node1", "node2", "node3"],
    trt_node=trt_node,
    input_mappings={"input": "original_input"},
    output_mappings={"original_output": 0}
)

# Save modified graph
graphdef_utils.save_graphdef_with_trt(new_graph_def, "model_with_trt.pb")
```

#### 3. Testing

Run the example:

```bash
python tensorflow/examples/custom_trt_op/python/example.py
```

Run tests:

```bash
bazel test //tensorflow/examples/custom_trt_op/python:custom_trt_ops_test
```

### Example Workflow

1. **Build TensorRT Engine**:
```python
import tensorrt as trt

# Create TensorRT engine for your model
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(...)
# ... add layers ...
engine = builder.build_engine(network, config)
serialized_engine = engine.serialize()
```

2. **Modify TensorFlow Graph**:
```python
# Replace compute-intensive subgraph with TensorRT
new_graph_def = replace_subgraph_with_trt(...)
```

3. **Run Inference**:
```python
# Load and run the modified graph
with tf.Graph().as_default() as g:
    tf.import_graph_def(new_graph_def)
    with tf.Session() as sess:
        output = sess.run(...)
```

### Advantages over TF-TRT

This custom op approach provides:
- More control over TensorRT engine creation
- Direct engine serialization/deserialization
- Easier debugging and profiling
- Simpler integration with existing TensorFlow pipelines
- No dependency on TF-TRT converter

### Limitations

- Requires manual engine creation
- Manual GraphDef modification
- GPU-only execution
- Requires TensorRT installation

### References

- TensorRT Documentation: https://docs.nvidia.com/deeplearning/tensorrt/
- TensorFlow Custom Ops: https://www.tensorflow.org/guide/create_op
- TF-TRT: `tensorflow/compiler/tf2tensorrt/`

---

<a name="chinese"></a>
## 中文

本示例演示如何为 TensorFlow 推理框架创建自定义 TensorRT 操作，使您能够在 TensorFlow 图中直接运行 TensorRT 引擎。

### 概述

本实现提供：
1. **自定义 Op 注册** - TensorRT 推理的 C++ op 定义
2. **Kernel 实现** - 执行 TensorRT 引擎的 GPU kernel
3. **GraphDef 修改工具** - 用于将子图替换为 TensorRT op 的 Python 工具
4. **测试框架** - 用于验证的示例和测试

### 特性

- ✅ 对 TensorFlow 代码库的最小修改
- ✅ 兼容最新的 TensorFlow 版本
- ✅ 在 TensorFlow 图中直接执行 TensorRT 引擎
- ✅ 简单的 GraphDef 修改
- ✅ 全面的示例和测试

### 架构

```
┌─────────────────────────────────────────────────────────────┐
│                    TensorFlow 图                             │
│  ┌──────────┐      ┌──────────────────┐      ┌──────────┐  │
│  │  输入    │─────▶│ CustomTRTInference│─────▶│  输出    │  │
│  │  张量    │      │      Op           │      │  张量    │  │
│  └──────────┘      └──────────────────┘      └──────────┘  │
│                            │                                 │
│                            ▼                                 │
│                    ┌──────────────┐                         │
│                    │ TensorRT     │                         │
│                    │ 引擎         │                         │
│                    │ (序列化)     │                         │
│                    └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

### 文件结构

```
tensorflow/examples/custom_trt_op/
├── ops/
│   └── custom_trt_op.cc          # Op 注册
├── kernels/
│   └── custom_trt_kernel.cc      # Kernel 实现
├── python/
│   ├── __init__.py               # Python 包
│   ├── custom_trt_ops.py         # Python 包装器
│   ├── graphdef_utils.py         # GraphDef 工具
│   ├── example.py                # 使用示例
│   └── custom_trt_ops_test.py    # 测试
├── BUILD                          # Bazel 构建配置
└── README.md                      # 本文件
```

### 前提条件

- TensorFlow（最新版本）
- CUDA 工具包
- TensorRT 8.x 或更高版本
- 计算能力 6.0+ 的 GPU

### 构建

```bash
# 构建自定义 op 库
bazel build //tensorflow/examples/custom_trt_op:custom_trt_kernels

# 构建 Python 包装器
bazel build //tensorflow/examples/custom_trt_op:custom_trt_ops_py
```

### 使用方法

#### 1. Op 实现

自定义 op 在 `ops/custom_trt_op.cc` 中定义。

Kernel 实现在 `kernels/custom_trt_kernel.cc` 中处理：
- 反序列化 TensorRT 引擎
- 管理执行上下文
- 绑定输入/输出张量
- 在 GPU 上执行推理

#### 2. GraphDef 修改

使用提供的工具将子图替换为 TensorRT：

```python
from tensorflow.examples.custom_trt_op.python import graphdef_utils
import tensorflow as tf

# 加载或创建您的图
graph_def = ...

# 为您的子图构建 TensorRT 引擎
serialized_engine = build_your_trt_engine()

# 创建 TRT 节点
trt_node = graphdef_utils.create_trt_node(
    node_name="trt_inference",
    serialized_engine=serialized_engine,
    input_names=["input"],
    output_names=["output"],
    input_types=[tf.float32],
    output_types=[tf.float32],
    input_shapes=[tf.TensorShape([1, 784])],
    output_shapes=[tf.TensorShape([1, 10])]
)

# 替换子图
new_graph_def = graphdef_utils.replace_subgraph_with_trt(
    graph_def=graph_def,
    subgraph_nodes=["node1", "node2", "node3"],
    trt_node=trt_node,
    input_mappings={"input": "original_input"},
    output_mappings={"original_output": 0}
)

# 保存修改后的图
graphdef_utils.save_graphdef_with_trt(new_graph_def, "model_with_trt.pb")
```

#### 3. 测试

运行示例：

```bash
python tensorflow/examples/custom_trt_op/python/example.py
```

运行测试：

```bash
bazel test //tensorflow/examples/custom_trt_op/python:custom_trt_ops_test
```

### 示例工作流程

1. **构建 TensorRT 引擎**
2. **修改 TensorFlow 图**
3. **运行推理**

### 相比 TF-TRT 的优势

此自定义 op 方法提供：
- 对 TensorRT 引擎创建的更多控制
- 直接的引擎序列化/反序列化
- 更容易的调试和性能分析
- 与现有 TensorFlow 管道更简单的集成
- 不依赖 TF-TRT 转换器

### 限制

- 需要手动创建引擎
- 手动修改 GraphDef
- 仅支持 GPU 执行
- 需要安装 TensorRT

### 参考资料

- TensorRT 文档: https://docs.nvidia.com/deeplearning/tensorrt/
- TensorFlow 自定义 Ops: https://www.tensorflow.org/guide/create_op
- TF-TRT: `tensorflow/compiler/tf2tensorrt/`
