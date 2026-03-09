#!/usr/bin/env python3
import subprocess
import time
import requests
import sys
import os

API_KEY = "sk_12345678_qwen35"
MODEL_NAME = "/home/ubuntu/LLM/Qwen/Qwen3.5-27B"
API_BASE = "http://localhost:8000/v1"

TEST_QUESTIONS = [
    "你好，请介绍一下自己",
    "你叫什么名字？",
    "今天天气怎么样？",
    "请写一首关于春天的诗",
    "Python和Java有什么区别？",
    "如何学习编程？",
    "请解释一下什么是人工智能",
    "1+1等于多少？",
    "你喜欢吃什么是食物？",
    "再见"
]

def wait_for_server(timeout=120):
    print("等待 vLLM server 启动...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(
                f"{API_BASE}/models",
                headers={"Authorization": f"Bearer {API_KEY}"},
                timeout=5
            )
            if response.status_code == 200:
                print("Server 已就绪！")
                return True
        except:
            pass
        time.sleep(2)
    return False

def test_chat():
    print("\n" + "="*60)
    print("开始测试...")
    print("="*60 + "\n")
    
    passed = 0
    failed = 0
    
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n【问题 {i}/10】{question}")
        print("-" * 40)
        
        try:
            response = requests.post(
                f"{API_BASE}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {API_KEY}"
                },
                json={
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": question}],
                    "temperature": 0.7,
                    "max_tokens": 512
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"]
                print(f"【回答】{answer}")
                passed += 1
                print("✓ 测试通过")
            else:
                print(f"✗ 错误: {response.status_code} - {response.text}")
                failed += 1
                
        except Exception as e:
            print(f"✗ 异常: {e}")
            failed += 1
        
        time.sleep(1)
    
    print("\n" + "="*60)
    print(f"测试完成: {passed} 通过, {failed} 失败")
    print("="*60)
    
    return failed == 0

def main():
    print("="*60)
    print("vLLM Qwen3.5-27B 验证测试")
    print("="*60)
    
    # 等待 server 启动
    if not wait_for_server():
        print("错误: Server 启动失败")
        sys.exit(1)
    
    # 运行测试
    success = test_chat()
    
    if success:
        print("\n所有测试通过！")
    else:
        print("\n有测试失败，请检查输出")

if __name__ == "__main__":
    main()
