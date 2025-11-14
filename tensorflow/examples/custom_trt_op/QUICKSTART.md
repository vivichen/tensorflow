# Quick Reference Guide / 快速参考指南

## English

### Quick Start

#### 1. Build the Custom Op

```bash
bazel build //tensorflow/examples/custom_trt_op:custom_trt_kernels
```

#### 2. Basic Usage

```python
from tensorflow.examples.custom_trt_op.python import graphdef_utils
import tensorflow as tf

# Create TRT node
trt_node = graphdef_utils.create_trt_node(
    node_name="my_trt_node",
    serialized_engine=my_engine_bytes,
    input_names=["input"],
    output_names=["output"],
    input_types=[tf.float32],
    output_types=[tf.float32],
    input_shapes=[tf.TensorShape([1, 224, 224, 3])],
    output_shapes=[tf.TensorShape([1, 1000])]
)

# Replace subgraph
new_graph = graphdef_utils.replace_subgraph_with_trt(
    graph_def=original_graph,
    subgraph_nodes=["conv1", "relu1", "pool1"],
    trt_node=trt_node,
    input_mappings={"input": "original_input"},
    output_mappings={"original_output": 0}
)
```

### API Reference

#### `create_trt_node()`

Creates a TensorRT node definition.

**Parameters:**
- `node_name` (str): Name for the TensorRT node
- `serialized_engine` (bytes): Serialized TensorRT engine
- `input_names` (list[str]): Input tensor names
- `output_names` (list[str]): Output tensor names
- `input_types` (list[tf.DType]): Input data types
- `output_types` (list[tf.DType]): Output data types
- `input_shapes` (list[tf.TensorShape]): Input shapes
- `output_shapes` (list[tf.TensorShape]): Output shapes
- `workspace_size_bytes` (int): TensorRT workspace size (default: 1GB)

**Returns:** NodeDef

#### `replace_subgraph_with_trt()`

Replaces a subgraph with a TensorRT node.

**Parameters:**
- `graph_def` (GraphDef): Original graph
- `subgraph_nodes` (list[str]): Nodes to replace
- `trt_node` (NodeDef): TensorRT node
- `input_mappings` (dict): Maps TRT inputs to original tensors
- `output_mappings` (dict): Maps original outputs to TRT output indices

**Returns:** Modified GraphDef

### Common Patterns

#### Pattern 1: Replace Single Subgraph

```python
# Original: Input -> [Conv -> ReLU -> Pool] -> Output
# Replace the bracketed part with TensorRT

trt_node = create_trt_node(...)
new_graph = replace_subgraph_with_trt(
    graph_def=graph,
    subgraph_nodes=["Conv", "ReLU", "Pool"],
    trt_node=trt_node,
    input_mappings={"input": "Input"},
    output_mappings={"Output": 0}
)
```

#### Pattern 2: Replace Multiple Subgraphs

```python
# Replace multiple independent subgraphs
for i, (nodes, engine) in enumerate(subgraphs):
    trt_node = create_trt_node(
        node_name=f"trt_{i}",
        serialized_engine=engine,
        ...
    )
    graph = replace_subgraph_with_trt(graph, nodes, trt_node, ...)
```

#### Pattern 3: Build TensorRT Engine

```python
import tensorrt as trt

def build_engine():
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # Add layers
    input_tensor = network.add_input("input", trt.float32, (1, 224, 224, 3))
    # ... add more layers ...
    
    # Build
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    return builder.build_serialized_network(network, config)
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| "TensorRT not found" | Install TensorRT: `pip install nvidia-tensorrt` |
| "GPU not available" | Ensure CUDA is installed and GPU is accessible |
| "Op not registered" | Build the kernel: `bazel build ...` |
| "Engine deserialization failed" | Check TensorRT version compatibility |
| "Shape mismatch" | Verify input/output shapes match engine |

---

## 中文

### 快速开始

#### 1. 构建自定义Op

```bash
bazel build //tensorflow/examples/custom_trt_op:custom_trt_kernels
```

#### 2. 基本使用

```python
from tensorflow.examples.custom_trt_op.python import graphdef_utils
import tensorflow as tf

# 创建TRT节点
trt_node = graphdef_utils.create_trt_node(
    node_name="my_trt_node",
    serialized_engine=my_engine_bytes,
    input_names=["input"],
    output_names=["output"],
    input_types=[tf.float32],
    output_types=[tf.float32],
    input_shapes=[tf.TensorShape([1, 224, 224, 3])],
    output_shapes=[tf.TensorShape([1, 1000])]
)

# 替换子图
new_graph = graphdef_utils.replace_subgraph_with_trt(
    graph_def=original_graph,
    subgraph_nodes=["conv1", "relu1", "pool1"],
    trt_node=trt_node,
    input_mappings={"input": "original_input"},
    output_mappings={"original_output": 0}
)
```

### API 参考

#### `create_trt_node()`

创建TensorRT节点定义。

**参数:**
- `node_name` (str): TensorRT节点名称
- `serialized_engine` (bytes): 序列化的TensorRT引擎
- `input_names` (list[str]): 输入张量名称
- `output_names` (list[str]): 输出张量名称
- `input_types` (list[tf.DType]): 输入数据类型
- `output_types` (list[tf.DType]): 输出数据类型
- `input_shapes` (list[tf.TensorShape]): 输入形状
- `output_shapes` (list[tf.TensorShape]): 输出形状
- `workspace_size_bytes` (int): TensorRT工作空间大小(默认: 1GB)

**返回:** NodeDef

#### `replace_subgraph_with_trt()`

用TensorRT节点替换子图。

**参数:**
- `graph_def` (GraphDef): 原始图
- `subgraph_nodes` (list[str]): 要替换的节点
- `trt_node` (NodeDef): TensorRT节点
- `input_mappings` (dict): TRT输入到原始张量的映射
- `output_mappings` (dict): 原始输出到TRT输出索引的映射

**返回:** 修改后的GraphDef

### 常用模式

#### 模式1: 替换单个子图

```python
# 原始: Input -> [Conv -> ReLU -> Pool] -> Output
# 将括号部分替换为TensorRT

trt_node = create_trt_node(...)
new_graph = replace_subgraph_with_trt(
    graph_def=graph,
    subgraph_nodes=["Conv", "ReLU", "Pool"],
    trt_node=trt_node,
    input_mappings={"input": "Input"},
    output_mappings={"Output": 0}
)
```

#### 模式2: 替换多个子图

```python
# 替换多个独立子图
for i, (nodes, engine) in enumerate(subgraphs):
    trt_node = create_trt_node(
        node_name=f"trt_{i}",
        serialized_engine=engine,
        ...
    )
    graph = replace_subgraph_with_trt(graph, nodes, trt_node, ...)
```

#### 模式3: 构建TensorRT引擎

```python
import tensorrt as trt

def build_engine():
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # 添加层
    input_tensor = network.add_input("input", trt.float32, (1, 224, 224, 3))
    # ... 添加更多层 ...
    
    # 构建
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    return builder.build_serialized_network(network, config)
```

### 常见问题

| 问题 | 解决方案 |
|------|----------|
| "找不到TensorRT" | 安装TensorRT: `pip install nvidia-tensorrt` |
| "GPU不可用" | 确保安装CUDA且GPU可访问 |
| "Op未注册" | 构建kernel: `bazel build ...` |
| "引擎反序列化失败" | 检查TensorRT版本兼容性 |
| "形状不匹配" | 验证输入/输出形状与引擎匹配 |

### 完整示例

请参考:
- `python/example.py` - 完整的端到端示例
- `python/custom_trt_ops_test.py` - 单元测试
- `TUTORIAL.md` - 详细教程
