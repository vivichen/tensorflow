# TensorRT Custom Op 完整教程 / Complete Tutorial

## 目录 / Table of Contents

1. [问题背景](#问题背景)
2. [解决方案](#解决方案)
3. [实现细节](#实现细节)
4. [使用示例](#使用示例)
5. [测试方法](#测试方法)

---

## 问题背景

### 需求
在 TensorFlow 推理框架中运行 TensorRT，通过自定义 op 的方式实现。

### 目标
1. 给出 op 的实现
2. 给出对 GraphDef 的修改
3. 给出测试方法
4. 可参考 TF-TRT，但需要在 TensorFlow 最新版本能跑

---

## 解决方案

### 整体架构

本方案提供了一个完整的自定义 TensorRT op 实现，包括：

```
自定义 TensorRT Op
├── 1. Op 定义 (ops/custom_trt_op.cc)
│   └── 注册 CustomTRTInferenceOp
├── 2. Kernel 实现 (kernels/custom_trt_kernel.cc)
│   └── GPU 上执行 TensorRT 引擎
├── 3. Python 接口 (python/custom_trt_ops.py)
│   └── Python 包装器和工具函数
└── 4. GraphDef 修改工具 (python/graphdef_utils.py)
    └── 替换子图为 TensorRT op
```

### 关键特性

- ✅ **最小侵入性**: 所有代码位于 `tensorflow/examples/custom_trt_op/`，不修改 TensorFlow 核心代码
- ✅ **版本兼容**: 使用标准 TensorFlow API，兼容最新版本
- ✅ **易于使用**: 提供完整的 Python 接口和工具函数
- ✅ **完整测试**: 包含单元测试和集成测试

---

## 实现细节

### 1. Op 的实现

#### 文件: `ops/custom_trt_op.cc`

Op 定义了 TensorRT 推理操作的接口：

```cpp
REGISTER_OP("CustomTRTInferenceOp")
    .Attr("serialized_engine: string")      // 序列化的 TensorRT 引擎
    .Attr("input_names: list(string)")      // 输入张量名称
    .Attr("output_names: list(string)")     // 输出张量名称
    .Attr("InT: list({float32, float16, int32, int8})")   // 输入类型
    .Attr("OutT: list({float32, float16, int32, int8})")  // 输出类型
    .Input("inputs: InT")
    .Output("outputs: OutT")
```

**关键属性说明**:
- `serialized_engine`: TensorRT 引擎的二进制序列化数据
- `input_names` / `output_names`: 在 TensorRT 引擎中定义的输入输出名称
- `input_shapes` / `output_shapes`: 张量形状信息

#### 文件: `kernels/custom_trt_kernel.cc`

Kernel 实现了实际的推理逻辑：

```cpp
class CustomTRTInferenceOp : public OpKernel {
  void Compute(OpKernelContext* context) override {
    // 1. 获取 CUDA stream
    // 2. 准备输入/输出绑定
    // 3. 执行 TensorRT 推理
    execution_context_->enqueueV2(bindings.data(), cuda_stream, nullptr);
  }
  
  void InitializeEngine(...) {
    // 1. 创建 TensorRT runtime
    // 2. 反序列化引擎
    // 3. 创建执行上下文
  }
};
```

**执行流程**:
1. 在构造函数中初始化 TensorRT 引擎（仅一次）
2. 在 Compute 中：
   - 获取 CUDA stream
   - 绑定输入输出张量
   - 执行推理
   - 返回结果

### 2. GraphDef 的修改

#### 文件: `python/graphdef_utils.py`

提供了两个核心函数：

##### 2.1 创建 TensorRT 节点

```python
def create_trt_node(
    node_name,           # TRT 节点名称
    serialized_engine,   # 序列化的 TensorRT 引擎
    input_names,         # 输入名称列表
    output_names,        # 输出名称列表
    input_types,         # 输入类型列表
    output_types,        # 输出类型列表
    input_shapes,        # 输入形状列表
    output_shapes        # 输出形状列表
):
    # 创建 NodeDef
    # 设置所有属性
    # 返回节点定义
```

##### 2.2 替换子图

```python
def replace_subgraph_with_trt(
    graph_def,          # 原始 GraphDef
    subgraph_nodes,     # 要替换的节点列表
    trt_node,           # TensorRT 节点
    input_mappings,     # 输入映射
    output_mappings     # 输出映射
):
    # 1. 删除子图节点
    # 2. 添加 TensorRT 节点
    # 3. 更新下游节点的连接
    # 返回新的 GraphDef
```

#### 修改示例

假设原始图结构：
```
Input -> MatMul -> Add -> Output
```

要将 MatMul 和 Add 替换为 TensorRT：

```python
# 步骤 1: 构建 TensorRT 引擎
import tensorrt as trt
# ... 构建引擎 ...
serialized_engine = engine.serialize()

# 步骤 2: 创建 TRT 节点
trt_node = create_trt_node(
    node_name="trt_matmul_add",
    serialized_engine=serialized_engine,
    input_names=["input"],
    output_names=["output"],
    input_types=[tf.float32],
    output_types=[tf.float32],
    input_shapes=[tf.TensorShape([None, 784])],
    output_shapes=[tf.TensorShape([None, 10])]
)

# 步骤 3: 替换子图
new_graph_def = replace_subgraph_with_trt(
    graph_def=original_graph_def,
    subgraph_nodes=["MatMul", "Add"],
    trt_node=trt_node,
    input_mappings={"input": "Input"},
    output_mappings={"Output": 0}
)
```

修改后的图结构：
```
Input -> CustomTRTInferenceOp -> Output
```

### 3. 测试方法

#### 文件: `python/custom_trt_ops_test.py`

测试用例包括：

##### 3.1 GraphDef 修改测试
```python
def test_graphdef_modification(self):
    # 创建简单图
    # 创建 TRT 节点
    # 验证节点创建成功
```

##### 3.2 引擎序列化测试
```python
def test_engine_serialization(self):
    # 构建 TensorRT 引擎
    # 序列化引擎
    # 验证可以反序列化
```

#### 文件: `python/example.py`

完整的端到端示例：

```python
def build_tensorrt_engine():
    """构建一个简单的 TensorRT 引擎"""
    # 创建网络: output = input * 2 + 1
    
def create_tensorflow_graph():
    """创建 TensorFlow 图"""
    # 创建对应的 TF 图
    
def replace_with_tensorrt():
    """替换子图为 TensorRT"""
    # 完整的替换流程
```

---

## 使用示例

### 完整工作流程

#### 步骤 1: 准备 TensorFlow 模型

```python
import tensorflow as tf

# 创建或加载模型
with tf.Graph().as_default() as g:
    x = tf.placeholder(tf.float32, [None, 784], name='input')
    W = tf.Variable(tf.random.normal([784, 10]), name='weights')
    b = tf.Variable(tf.zeros([10]), name='bias')
    y = tf.nn.softmax(tf.matmul(x, W) + b, name='output')

# 保存 GraphDef
graph_def = g.as_graph_def()
```

#### 步骤 2: 构建 TensorRT 引擎

```python
import tensorrt as trt
import numpy as np

def build_trt_engine():
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # 定义输入
    input_tensor = network.add_input(
        name="input", dtype=trt.float32, shape=(-1, 784))
    
    # 添加层（对应 MatMul + Bias + Softmax）
    # ... 添加对应层 ...
    
    # 构建引擎
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    
    serialized_engine = builder.build_serialized_network(network, config)
    return serialized_engine
```

#### 步骤 3: 修改 GraphDef

```python
from tensorflow.examples.custom_trt_op.python import graphdef_utils

# 构建引擎
serialized_engine = build_trt_engine()

# 创建 TRT 节点
trt_node = graphdef_utils.create_trt_node(
    node_name="trt_inference",
    serialized_engine=serialized_engine,
    input_names=["input"],
    output_names=["output"],
    input_types=[tf.float32],
    output_types=[tf.float32],
    input_shapes=[tf.TensorShape([None, 784])],
    output_shapes=[tf.TensorShape([None, 10])]
)

# 替换子图
new_graph_def = graphdef_utils.replace_subgraph_with_trt(
    graph_def=graph_def,
    subgraph_nodes=["MatMul", "BiasAdd", "Softmax"],
    trt_node=trt_node,
    input_mappings={"input": "input"},
    output_mappings={"output": 0}
)

# 保存
graphdef_utils.save_graphdef_with_trt(new_graph_def, "model_trt.pb")
```

#### 步骤 4: 运行推理

```python
import tensorflow as tf
import numpy as np

# 加载修改后的图
with tf.io.gfile.GFile("model_trt.pb", "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

# 运行推理
with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name="")
    
    with tf.compat.v1.Session(graph=graph) as sess:
        # 准备输入
        input_data = np.random.randn(1, 784).astype(np.float32)
        
        # 运行推理
        input_tensor = graph.get_tensor_by_name("input:0")
        output_tensor = graph.get_tensor_by_name("trt_inference:0")
        
        result = sess.run(output_tensor, feed_dict={input_tensor: input_data})
        print("Output:", result)
```

---

## 测试方法

### 1. 构建测试

```bash
# 构建 op 和 kernel
bazel build //tensorflow/examples/custom_trt_op:custom_trt_kernels

# 构建 Python 包
bazel build //tensorflow/examples/custom_trt_op:custom_trt_ops_py
```

### 2. 运行单元测试

```bash
# 运行所有测试
bazel test //tensorflow/examples/custom_trt_op/python:custom_trt_ops_test

# 运行带详细输出
bazel test //tensorflow/examples/custom_trt_op/python:custom_trt_ops_test --test_output=all
```

### 3. 运行示例

```bash
# 运行示例程序
bazel run //tensorflow/examples/custom_trt_op:example

# 或直接用 Python
python tensorflow/examples/custom_trt_op/python/example.py
```

### 4. 手动测试

```python
# test.py
from tensorflow.examples.custom_trt_op.python import graphdef_utils
import tensorflow as tf

# 1. 创建简单图
with tf.Graph().as_default() as g:
    x = tf.constant([[1.0, 2.0, 3.0, 4.0]], name="input")
    y = tf.multiply(x, 2.0, name="multiply")
    z = tf.add(y, 1.0, name="add")

graph_def = g.as_graph_def()

# 2. 打印原始节点
print("Original nodes:")
for node in graph_def.node:
    print(f"  {node.name}: {node.op}")

# 3. 创建 TRT 节点（假设已有引擎）
# trt_node = create_trt_node(...)

# 4. 替换并验证
# new_graph_def = replace_subgraph_with_trt(...)
```

### 5. 性能测试

创建性能测试脚本：

```python
import time
import numpy as np
import tensorflow as tf

def benchmark(graph_def, num_iterations=100):
    """性能基准测试"""
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
        
        with tf.compat.v1.Session(graph=graph) as sess:
            input_tensor = graph.get_tensor_by_name("input:0")
            output_tensor = graph.get_tensor_by_name("output:0")
            
            # 预热
            for _ in range(10):
                sess.run(output_tensor, 
                        feed_dict={input_tensor: np.random.randn(1, 784)})
            
            # 测试
            start = time.time()
            for _ in range(num_iterations):
                sess.run(output_tensor, 
                        feed_dict={input_tensor: np.random.randn(1, 784)})
            end = time.time()
            
            avg_time = (end - start) / num_iterations * 1000  # ms
            print(f"Average inference time: {avg_time:.2f} ms")

# 测试原始图
original_graph = load_graphdef("model_original.pb")
print("Original TensorFlow:")
benchmark(original_graph)

# 测试 TensorRT 图
trt_graph = load_graphdef("model_trt.pb")
print("\nWith TensorRT:")
benchmark(trt_graph)
```

---

## 总结

### 提供的内容

1. ✅ **Op 实现**: 
   - `ops/custom_trt_op.cc` - Op 注册
   - `kernels/custom_trt_kernel.cc` - Kernel 实现

2. ✅ **GraphDef 修改**:
   - `python/graphdef_utils.py` - 工具函数
   - 详细的使用示例

3. ✅ **测试方法**:
   - `python/custom_trt_ops_test.py` - 单元测试
   - `python/example.py` - 完整示例
   - 性能测试方法

4. ✅ **完整文档**:
   - `README.md` - 中英文文档
   - 本教程文件

### 与 TF-TRT 的对比

| 特性 | 本方案 | TF-TRT |
|------|--------|--------|
| 引擎控制 | 完全手动控制 | 自动转换 |
| GraphDef 修改 | 显式修改 | 自动优化 |
| 调试难度 | 容易 | 较难 |
| 灵活性 | 高 | 中 |
| 易用性 | 需要手动操作 | 自动化 |
| 版本兼容 | 最新版本 | 依赖版本 |

### 后续改进

可能的改进方向：
- 添加动态 batch size 支持
- 添加 INT8 量化支持
- 提供自动引擎构建工具
- 添加更多性能优化选项
