# 双 RTX 5090 GPU 上部署 vLLM + Qwen3.5 完整指南

> **声明**：本文为实际部署经验总结，涵盖从环境配置到服务上线的完整流程。文中涉及的具体参数、版本号均为实测验证结果，供大家参考。
> 
> **目标读者**：具备深度学习基础，熟悉 Python 和 CUDA 环境配置的开发者

---

## 一、背景与目标

### 1.1 硬件环境

本文的实验环境配置如下：

| 组件 | 规格 |
|------|------|
| GPU | 2 × NVIDIA RTX 5090 32GB |
| 显存总量 | 64GB |
| 操作系统 | Ubuntu |
| Python 环境 | Anaconda + my_pytorch |
| vLLM 版本 | 0.17.0 |
| CUDA 版本 | 13.0 |
| NVIDIA 驱动 | 580.x |

### 1.2 部署目标

在双 RTX 5090 上搭建 vLLM 推理服务，要求：

- 使用 Qwen3.5 系列模型
- 支持 OpenAI 兼容 API
- 配置 API 密钥认证
- 能够处理对话请求并返回结果
- 支持流式输出

---

## 二、为什么选择 Qwen3.5-27B 而不是 35B MOE

### 2.1 35B MOE 模型的显存需求

Qwen3.5-35B-A3B 是一个 MoE（混合专家）模型，特点是：

- **总参数量**：35B
- **激活参数量**：约 3B（每次推理只激活部分专家）
- **FP16 显存占用**：约 70GB
- **FP8 量化后**：约 35GB

理论上，双 RTX 5090 的 64GB 显存承载 35B MOE（FP8 量化）是有可能的。

### 2.2 RTX 5090 的兼容性难题

**核心问题：Blackwell 架构（Compute Capability 12.0）**

RTX 5090 基于 NVIDIA Blackwell 架构，计算能力为 **12.0**。这个新架构带来了几个实际问题：

| 问题 | 说明 |
|------|------|
| CUDA 版本 | 需要 CUDA 13.0，常规 CUDA 12.x 不支持 |
| 驱动要求 | 需要较新的驱动版本（如 580.x 系列） |
| FlashInfer | 预编译版本不支持 Blackwell，JIT 编译失败 |
| 生态滞后 | 主流深度学习库对新架构支持不完善 |

### 2.3 踩坑实录：35B MOE 行不通

我尝试过多种方案：

1. **FP8 量化 + TP=2**：FlashInfer JIT 编译失败
2. **BF16 + TP=2 + 减小 max_model_len**：显存仍不够
3. **升级 CUDA 到 13.0**：FlashInfer 仍然报错

**最终结论**：在当前软件生态下，35B MOE 在 RTX 5090 上运行存在较大障碍。

### 2.4 27B 模型是更务实的选择

| 对比项 | Qwen3.5-27B | Qwen3.5-35B MOE |
|--------|-------------|------------------|
| 模型类型 | Dense | MoE |
| 参数量 | 27B | 35B（激活 3B） |
| FP16 显存 | ~54GB | ~70GB |
| TP=2 可行性 | ✅ 流畅 | ❌ 困难 |
| 推理速度 | 快 | 略慢 |
| 代码兼容性 | 稳定 | 问题多 |

**结论**：Qwen3.5-27B 在双 RTX 5090 上可以流畅运行，是更稳妥的选择。

---

## 三、环境准备

### 3.1 CUDA 安装

RTX 5090 必须使用 CUDA 13.0 或更高版本。

```bash
# 1. 下载 CUDA 13.0
# 访问 https://developer.nvidia.com/cuda-downloads
# 选择: Linux -> x86_64 -> Ubuntu -> 22.04 -> deb (network)

# 2. 安装 CUDA 13.0
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-13-0

# 3. 设置环境变量（添加到 ~/.bashrc）
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### 3.2 NVIDIA 驱动升级

```bash
# 查看当前驱动版本
nvidia-smi

# 如果版本低于 580，需要升级驱动
sudo apt-get update
sudo apt-get install nvidia-driver-580

# 重启生效
sudo reboot
```

### 3.3 Python 环境

```bash
# 创建 Anaconda 环境
conda create -n my_pytorch python=3.10
conda activate my_pytorch

# 安装 PyTorch（CUDA 13.0 版本）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# 安装 vLLM
pip install vllm>=0.6.0
```

---

## 四、服务器部署

### 4.1 部署流程

#### 步骤 1：准备模型

```bash
# 下载 Qwen3.5-27B 模型
# 方法1：HuggingFace
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-27B-Instruct /home/ubuntu/LLM/Qwen/Qwen3.5-27B

# 方法2：ModelScope
modelscope download --model Qwen/Qwen2.5-27B-Instruct --local_dir /home/ubuntu/LLM/Qwen/Qwen3.5-27B
```

#### 步骤 2：创建启动脚本

```python
#!/usr/bin/env python3
"""
vLLM Server for Qwen3.5-27B on Dual RTX 5090 32GB
"""

import os
import subprocess

# ==================== 配置 ====================
MODEL_PATH = "/home/ubuntu/LLM/Qwen/Qwen3.5-27B"
PYTHON_BIN = "/home/ubuntu/anaconda3/envs/my_pytorch/bin/python"
HOST = "0.0.0.0"
PORT = 8000
TP_SIZE = 2
DTYPE = "auto"
QUANTIZATION = "fp8"
MAX_MODEL_LEN = 4096
GPU_MEM_UTILIZATION = 0.85
MAX_NUM_SEQS = 32
ENABLE_CUDA_GRAPH = True
ENABLE_CHUNKED_PREFILL = True
SCHEDULING_POLICY = "fcfs"
API_KEY = "sk_12345678_qwen35"

# ==================== 环境变量 ====================
os.environ["CUDA_HOME"] = "/usr/local/cuda-13.0"
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-13.0/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["PATH"] = "/usr/local/cuda-13.0/bin:/home/ubuntu/anaconda3/envs/my_pytorch/bin:" + os.environ.get("PATH", "")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

# ==================== 构建命令 ====================
cmd = [
    PYTHON_BIN, "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL_PATH,
    "--tensor-parallel-size", str(TP_SIZE),
    "--dtype", DTYPE,
    "--quantization", QUANTIZATION,
    "--max-model-len", str(MAX_MODEL_LEN),
    "--max-num-seqs", str(MAX_NUM_SEQS),
    "--gpu-memory-utilization", str(GPU_MEM_UTILIZATION),
    "--enforce-eager" if ENABLE_CUDA_GRAPH else "",
    "--enable-chunked-prefill" if ENABLE_CHUNKED_PREFILL else "",
    "--scheduling-policy", SCHEDULING_POLICY,
    "--attention-backend", "FLASH_ATTN",
    "--host", HOST,
    "--port", str(PORT),
    "--api-key", API_KEY,
    "--chat-template", os.path.join(MODEL_PATH, "chat_template.jinja"),
]

cmd = [c for c in cmd if c]

# ==================== 启动服务 ====================
full_env = os.environ.copy()
full_env["PATH"] = "/usr/local/cuda-13.0/bin:" + full_env["PATH"]
full_env["CUDA_HOME"] = "/usr/local/cuda-13.0"
full_env["CUDA_PATH"] = "/usr/local/cuda-13.0"
full_env["LD_LIBRARY_PATH"] = "/usr/local/cuda-13.0/lib64:" + full_env.get("LD_LIBRARY_PATH", "")

subprocess.run(cmd, env=full_env)
```

#### 步骤 3：启动服务器

```bash
# 方式1：直接运行（当前终端）
/home/ubuntu/anaconda3/envs/my_pytorch/bin/python /home/ubuntu/PyCharmMiscProject/vllm_qwen_server.py

# 方式2：后台运行（推荐）
nohup /home/ubuntu/anaconda3/envs/my_pytorch/bin/python /home/ubuntu/PyCharmMiscProject/vllm_qwen_server.py > /tmp/vllm_server.log 2>&1 &

# 查看日志
tail -f /tmp/vllm_server.log

# 等待约 30-60 秒让模型加载
```

#### 步骤 4：验证服务

```bash
# 检查服务是否启动成功
curl -s http://localhost:8000/v1/models -H "Authorization: Bearer sk_12345678_qwen35"

# 返回 JSON 表示成功
```

### 4.2 关键参数解释

| 参数 | 值 | 说明 |
|------|-----|------|
| `--tensor-parallel-size` | 2 | 使用双 GPU 并行推理 |
| `--quantization` | fp8 | FP8 权重量化，减少显存占用 |
| `--max-model-len` | 4096 | 最大上下文长度 |
| `--gpu-memory-utilization` | 0.85 | 显存利用比例（预留 15% 给系统） |
| `--enforce-eager` | - | 禁用 CUDA Graph，提高兼容性 |
| `--enable-chunked-prefill` | - | 启用分块预填充，提升吞吐量 |
| `--attention-backend` | FLASH_ATTN | **关键：强制使用 Flash Attention** |
| `--api-key` | sk_12345678_qwen35 | API 认证密钥 |
| `--kv-cache-dtype` | **不要设置** | Flash Attention 不支持 FP8 KV Cache |

### 4.3 关闭服务器

```bash
# 方法1：pkill（推荐）
pkill -f "vllm_qwen_server.py"

# 方法2：杀掉所有 vLLM 相关进程
pkill -f "vllm.entrypoints"

# 方法3：精确杀掉
ps aux | grep vllm | grep -v grep | awk '{print $2}' | xargs kill -9
```

---

## 五、客户端部署

### 5.1 交互式对话客户端

创建 `vllm_qwen_client.py`：

```python
#!/usr/bin/env python3
import requests
import json
import sys
import time

API_BASE = "http://localhost:8000/v1"
MODEL_NAME = "/home/ubuntu/LLM/Qwen/Qwen3.5-27B"
API_KEY = "sk_12345678_qwen35"

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

def check_server_ready():
    try:
        resp = requests.get(f"{API_BASE}/models", headers=HEADERS, timeout=10)
        return resp.status_code == 200
    except:
        return False

def warmup(n=2):
    print(f"正在预热服务器 ({n} 次)...")
    warmup_prompts = ["你好", "今天天气不错"]
    for i, prompt in enumerate(warmup_prompts[:n]):
        print(f"  预热 {i+1}/{n}: '{prompt}'")
        try:
            url = f"{API_BASE}/chat/completions"
            payload = {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1,
                "temperature": 0,
            }
            resp = requests.post(url, headers=HEADERS, json=payload, timeout=60)
            resp.raise_for_status()
        except Exception as e:
            print(f"    预热请求跳过: {e}")
    print("预热完成！\n")

def chat_stream(prompt: str, temperature: float = 0.7, max_tokens: int = 2048):
    url = f"{API_BASE}/chat/completions"
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True
    }
    
    response = requests.post(url, headers=HEADERS, json=payload, timeout=300, stream=True)
    response.raise_for_status()
    
    for chunk in response.iter_lines():
        if chunk:
            line = chunk.decode('utf-8')
            if line.startswith('data: '):
                data = line[6:]
                if data == '[DONE]':
                    break
                try:
                    json_data = json.loads(data)
                    delta = json_data.get('choices', [{}])[0].get('delta', {})
                    content = delta.get('content', '')
                    if content:
                        yield content
                except json.JSONDecodeError:
                    continue

def main():
    print("=" * 50)
    print("Qwen3.5-27B Chat Client (流式输出)")
    print("输入问题与大模型对话，输入 exit 退出")
    print("=" * 50)
    
    print("\n检查服务器连接...")
    if not check_server_ready():
        print("错误: 无法连接到 vLLM 服务器，请确认服务是否启动")
        return
    
    warmup(n=2)
    
    while True:
        try:
            prompt = input("\n你: ").strip()
            if prompt.lower() in ["exit", "quit", "q"]:
                print("再见！")
                break
            
            if not prompt:
                continue
            
            print("\nAI: ", end="", flush=True)
            
            for chunk in chat_stream(prompt):
                print(chunk, end="", flush=True)
            
            print()
            
        except KeyboardInterrupt:
            print("\n\n退出中...")
            break
        except requests.exceptions.ConnectionError:
            print("\n错误: 无法连接到 vLLM 服务器，请确认服务是否启动")
            break
        except Exception as e:
            print(f"\n错误: {e}")

if __name__ == "__main__":
    main()
```

### 5.2 使用方法

```bash
# 启动交互式对话
/home/ubuntu/anaconda3/envs/my_pytorch/bin/python /home/ubuntu/PyCharmMiscProject/vllm_qwen_client.py
```

### 5.3 curl 测试

```bash
# 简单测试
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk_12345678_qwen35" \
  -d '{
    "model": "/home/ubuntu/LLM/Qwen/Qwen3.5-27B",
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

---

## 六、踩坑全记录（重点）

### 6.1 第一个坑：CUDA 版本问题

**问题**：默认 CUDA 12.x 无法识别 RTX 5090

**症状**：
```
CUDA error: no kernel image is available for execution on the device
```

**解决方案**：必须安装 CUDA 13.0

---

### 6.2 第二个坑：驱动版本问题

**问题**：旧驱动不支持 RTX 5090

**症状**：
```
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver
```

**解决方案**：升级驱动到 580.x 或更新版本

---

### 6.3 第三个坑：FlashInfer JIT 编译失败（最关键）

**问题**：FlashInfer 无法在 RTX 5090 上编译

**症状**：
```
RuntimeError: Ninja build failed.
fatal error: cuda_runtime.h: 没有那个文件或目录
```

**尝试过的无效方案**：

| 方案 | 结果 |
|------|------|
| 设置 CUDA_HOME、LD_LIBRARY_PATH | ❌ 无效 |
| 创建 nvcc 符号链接 | ❌ 仍然找不到头文件 |
| 禁用 FlashInfer 部分功能 | ❌ 仍然尝试编译 |

**最终解决方案**：完全禁用 FlashInfer，使用 Flash Attention

```bash
--attention-backend FLASH_ATTN
```

---

### 6.4 第四个坑：FP8 KV Cache 不兼容

**问题**：启用 `--kv-cache-dtype fp8` 后报错

**症状**：
```
ValueError: Selected backend AttentionBackendEnum.FLASH_ATTN is not valid for this configuration. 
Reason: ['kv_cache_dtype not supported']
```

**原因**：Flash Attention 不支持 FP8 KV Cache

**解决方案**：移除 `--kv-cache-dtype fp8` 参数，使用默认的 FP16

---

### 6.5 第五个坑：环境变量设置时机

**问题**：环境变量在 Python subprocess 中不生效

**症状**：服务启动参数中的环境变量被忽略

**解决方案**：在父进程级别设置环境变量，并显式传递给 subprocess

---

### 6.6 第六个坑：首个请求响应慢

**问题**：启动后第一个问题要等很久才能开始流式输出

**原因**：
- CUDA 内核 JIT 编译
- DeepGemm JIT 编译
- KV Cache 冷启动
- GPU 内存分配

**解决方案**：添加预热功能，客户端启动时自动发送预热请求

---

### 6.7 第七个坑：nvcc 找不到

**问题**：FlashInfer 尝试编译时找不到 nvcc

**症状**：
```
nvcc: command not found
```

**临时解决方案**：
```bash
sudo ln -sf /usr/local/cuda-13.0/bin/nvcc /usr/bin/nvcc
```

**根本解决方案**：禁用 FlashInfer，使用 Flash Attention

---

## 七、重要注意事项

### 7.1 硬件相关

1. ** Blackwell 架构兼容性**
   - RTX 5090 使用全新的 Blackwell 架构（Compute Capability 12.0）
   - 许多深度学习库的预编译版本尚未支持
   - 建议等待 3-6 个月让生态完善

2. **双卡并行问题**
   - 确保两张显卡通过 NVLink 连接（提升通信速度）
   - 如果没有 NVLink，性能会有所下降

3. **显存监控**
   - 使用 `nvidia-smi` 监控显存使用
   - 预留 10-15% 显存给系统和 CUDA 运行时

### 7.2 软件配置相关

1. **CUDA 版本必须是 13.0**
   - CUDA 12.x 不支持 RTX 5090
   - 安装后记得设置环境变量

2. **驱动版本必须够新**
   - 推荐 580.x 或更新版本
   - 旧驱动可能导致各种奇怪错误

3. **慎用 MoE 模型**
   - MoE 模型对驱动和库的支持要求更高
   - 消费级硬件上优先选择 Dense 模型

### 7.3 性能优化相关

1. **启用 `--enforce-eager`**
   - 禁用 CUDA Graph 可以提高兼容性
   - 在新硬件上稳定性优先于极致性能

2. **启用 `--enable-chunked-prefill`**
   - 允许将长Prompt分块处理
   - 提升吞吐量和并发能力

3. **调整 `max_model_len`**
   - 4096 是保守值
   - 显存充足时可以增加到 8192 或 16384

### 7.4 调试相关

1. **查看日志**
   ```bash
   tail -f /tmp/vllm_server.log
   ```

2. **查看 GPU 状态**
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **检查进程**
   ```bash
   ps aux | grep vllm
   ```

---

## 八、验证结果

### 8.1 服务启动日志

```
INFO 03-09 15:14:00 version 0.17.0
INFO 03-09 15:14:00 model   /home/ubuntu/LLM/Qwen/Qwen3.5-27B
INFO 03-09 15:14:02 Setting attention block size to 1568 tokens
INFO 03-09 15:14:38 Available KV cache memory: 9.66 GiB
INFO 03-09 15:14:48 Maximum concurrency for 4,096 tokens per request: 66.83x
INFO 03-09 15:14:54 Starting vLLM API server on http://0.0.0.0:8000
```

### 8.2 测试结果

| 测试问题 | 结果 |
|----------|------|
| 你好，请介绍一下自己 | ✅ 通过 |
| 你叫什么名字？ | ✅ 通过 |
| 今天天气怎么样？ | ✅ 通过 |
| 请写一首关于春天的诗 | ✅ 通过 |
| Python和Java有什么区别？ | ✅ 通过 |
| 如何学习编程？ | ✅ 通过 |
| 请解释一下什么是人工智能 | ✅ 通过 |
| 1+1等于多少？ | ✅ 通过 |
| 你喜欢吃什么是食物？ | ✅ 通过 |
| 再见 | ✅ 通过 |

**10/10 测试通过**

---

## 九、常见问题 FAQ

**Q1：双 RTX 5090 可以跑 70B 模型吗？**

A：理论上 FP8 量化可以，但需要模型和驱动都完美支持。目前阶段，27B-32B 是最稳妥的选择。

**Q2：一定要用 CUDA 13.0 吗？**

A：是的，RTX 5090 需要 CUDA 13.0。

**Q3：FlashInfer 真的不能用吗？**

A：在当前版本（vLLM 0.17.0）下，FlashInfer 无法在 RTX 5090 上正常工作。建议使用 Flash Attention。

**Q4：如何查看模型是否加载成功？**

A：访问 `http://localhost:8000/v1/models`，返回模型列表即成功。

**Q5：API 认证怎么配置？**

A：使用 `--api-key` 参数启动服务，请求时在 Header 中添加 `Authorization: Bearer YOUR_KEY`。

**Q6：首个请求很慢怎么办？**

A：这是正常现象，客户端已添加预热功能。首次请求会触发 CUDA 内核 JIT 编译，后续请求会快很多。

**Q7：如何调整最大上下文长度？**

A：修改 `--max-model_len` 参数。注意：长度增加会显著增加显存占用。

**Q8：如何关闭服务器？**

A：使用 `pkill -f "vllm_qwen_server.py"` 或 `pkill -f "vllm.entrypoints"`

---

## 十、参考资源

- vLLM 官方文档：https://docs.vllm.ai/
- Qwen 模型：https://huggingface.co/Qwen
- NVIDIA CUDA：https://developer.nvidia.com/cuda-downloads
- NVIDIA 驱动：https://www.nvidia.com/Download/index.aspx

---

祝部署顺利！如果有问题，欢迎交流讨论。
