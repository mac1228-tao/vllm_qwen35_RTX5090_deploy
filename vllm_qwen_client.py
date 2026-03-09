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
    """检查服务器是否就绪"""
    try:
        resp = requests.get(f"{API_BASE}/models", headers=HEADERS, timeout=10)
        return resp.status_code == 200
    except:
        return False

def warmup(n=2):
    """预热服务器，发送 dummy 请求"""
    print(f"正在预热服务器 ({n} 次)...")
    
    warmup_prompts = [
        "你好",
        "今天天气不错",
    ]
    
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
    """流式对话，返回生成器"""
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

def chat(prompt: str, temperature: float = 0.7, max_tokens: int = 2048):
    """非流式对话（保留备用）"""
    url = f"{API_BASE}/chat/completions"
    
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    response = requests.post(url, headers=HEADERS, json=payload, timeout=300)
    response.raise_for_status()
    
    result = response.json()
    return result["choices"][0]["message"]["content"]

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
            if prompt.lower() == "exit" or prompt.lower() == "quit" or prompt.lower() == "q":
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
