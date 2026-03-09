#!/usr/bin/env python3
"""
vLLM Server for Qwen3.5-35B-A3B on Dual RTX 5090 32GB

Optimized configuration for:
- Dual NVIDIA RTX 5090 (32GB each)
- Tensor Parallelism (tp=2)
- BF16 precision (fits in 64GB VRAM total)
- OpenAI-compatible API
"""

import os
import subprocess
import sys

# ==================== Configuration ====================
MODEL_PATH = "/home/ubuntu/LLM/Qwen/Qwen3.5-27B"

# Python from my_pytorch environment
PYTHON_BIN = "/home/ubuntu/anaconda3/envs/my_pytorch/bin/python"

# vLLM Server Settings
HOST = "0.0.0.0"
PORT = 8000

# Tensor Parallelism - 2 GPUs
TP_SIZE = 2

# GPU Settings
DTYPE = "auto"
QUANTIZATION = "fp8"

# Max model len - adjust based on available memory
MAX_MODEL_LEN = 4096

# GPU memory utilization
GPU_MEM_UTILIZATION = 0.85

# Max sequences
MAX_NUM_SEQS = 32

# Enable CUDA graph for better performance
ENABLE_CUDA_GRAPH = True

# Enable Chunked Prefill
ENABLE_CHUNKED_PREFILL = True

# Scheduling policy
SCHEDULING_POLICY = "fcfs"

# API Security
API_KEY = "sk_12345678_qwen35"

# ==================== Environment Variables ====================
# Set CUDA paths
os.environ["CUDA_HOME"] = "/usr/local/cuda-13.0"
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-13.0/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["PATH"] = "/usr/local/cuda-13.0/bin:/home/ubuntu/anaconda3/envs/my_pytorch/bin:" + os.environ.get("PATH", "")

# Set CUDA visible devices to use both GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Add conda env bin to PATH for ninja
os.environ["PATH"] = "/home/ubuntu/anaconda3/envs/my_pytorch/bin:" + os.environ.get("PATH", "")

# Enable TF32 for better performance (RTX 5090 supports this)
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"

# Force Flash Attention backend (disable FlashInfer)
# Use FLASH_ATTN for pure Flash Attention
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

# ==================== Build vLLM Command ====================
def build_vllm_serve_cmd():
    """Build the vLLM serve command with optimized parameters."""
    
    cmd = [
        PYTHON_BIN, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_PATH,
        "--trust-remote-code",
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
    
    # Filter empty strings
    cmd = [c for c in cmd if c]
    
    return cmd


def check_gpu_status():
    """Check GPU status before starting."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", 
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True
        )
        print("=" * 60)
        print("GPU Status:")
        print(result.stdout)
        print("=" * 60)
    except Exception as e:
        print(f"Warning: Could not query GPU status: {e}")


def main():
    """Main entry point."""
    print("=" * 60)
    print("vLLM Server for Qwen3.5-35B-A3B")
    print("Configuration:")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Tensor Parallel Size: {TP_SIZE}")
    print(f"  Max Model Length: {MAX_MODEL_LEN}")
    print(f"  GPU Memory Utilization: {GPU_MEM_UTILIZATION}")
    print(f"  Host: {HOST}:{PORT}")
    print("=" * 60)
    
    # Check GPU status
    check_gpu_status()
    
    # Build command
    cmd = build_vllm_serve_cmd()
    print(f"\nStarting vLLM server...")
    print(f"Command: {' '.join(cmd)}\n")
    
    # Run vLLM with full environment
    full_env = os.environ.copy()
    full_env["PATH"] = "/home/ubuntu/bin:" + full_env.get("PATH", "")
    full_env["PATH"] = "/usr/local/cuda-13.0/bin:" + full_env["PATH"]
    full_env["CUDA_HOME"] = "/usr/local/cuda-13.0"
    full_env["CUDA_PATH"] = "/usr/local/cuda-13.0"
    full_env["LD_LIBRARY_PATH"] = "/usr/local/cuda-13.0/lib64:" + full_env.get("LD_LIBRARY_PATH", "")
    full_env["CMAKE_PREFIX_PATH"] = "/usr/local/cuda-13.0"
    
    try:
        subprocess.run(cmd, env=full_env)
    except KeyboardInterrupt:
        print("\nShutting down vLLM server...")
    except FileNotFoundError:
        print("Error: vLLM not installed. Install with:")
        print("  pip install vllm>=0.8.0")
        sys.exit(1)


if __name__ == "__main__":
    main()
