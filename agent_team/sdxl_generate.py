#!/usr/bin/env python3
"""
SDXL 图片生成脚本 - 使用 animagine-xl-3.1.safetensors
生成「美女在雨中跑步」图片
"""
import json
import sys
import time
import requests
import random
from pathlib import Path

COMFYUI_HOST = "localhost:8188"
MODEL_NAME = "animagine-xl-3.1.safetensors"
OUTPUT_PATH = "D:/agent_team/teamwork_result.jpg"

# 提示词
POSITIVE_PROMPT = "beautiful young woman running in the rain, athletic fitness outfit, wet hair clinging to face, water droplets on skin, energetic running pose, dynamic motion, rainy night city street, wet pavement with reflections, street lamp glow, rain drops visible in air, cinematic lighting, photorealistic, high quality, 8k, detailed"
NEGATIVE_PROMPT = "text, watermark, low quality, ugly, deformed, blurry, cartoon, anime, bad anatomy, worst quality"

def poll_history(prompt_id, timeout=300):
    """轮询任务状态直到完成"""
    history_url = f"http://{COMFYUI_HOST}/history/{prompt_id}"
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Task timeout after {timeout}s")
        time.sleep(5)
        resp = requests.get(history_url)
        resp.raise_for_status()
        history = resp.json()
        if prompt_id in history and history[prompt_id].get('outputs'):
            return history[prompt_id]
        elapsed = int(time.time() - start_time)
        print(f"  Waiting... ({elapsed}s)")

def download_image(node_outputs, output_path):
    """从输出节点下载图片"""
    # 查找有 images 输出的节点
    for node_id, outputs in node_outputs.items():
        if 'images' in outputs and outputs['images']:
            img_info = outputs['images'][0]
            filename = img_info['filename']
            subfolder = img_info.get('subfolder', '')

            view_url = f"http://{COMFYUI_HOST}/view?filename={filename}&type=output&subfolder={subfolder}"
            resp = requests.get(view_url)
            resp.raise_for_status()

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(resp.content)
            print(f"Saved: {output_path}")
            return output_path

    raise ValueError("No image output found")

def submit_and_wait():
    """提交任务并等待完成"""
    seed = random.randint(0, 999999999999)

    # 构建 SDXL txt2img 工作流 - 正确的 API 格式
    workflow = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": MODEL_NAME
            }
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": POSITIVE_PROMPT,
                "clip": ["1", 1]
            }
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": NEGATIVE_PROMPT,
                "clip": ["1", 1]
            }
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": 1024,
                "height": 1024,
                "batch_size": 1
            }
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "seed": seed,
                "steps": 25,
                "cfg": 7.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
                "denoise": 1.0
            }
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["5", 0],
                "vae": ["1", 2]
            }
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["6", 0],
                "filename_prefix": "teamwork"
            }
        }
    }

    # 打印工作流信息
    print(f"Workflow: CheckpointLoaderSimple({MODEL_NAME})")
    print(f"  Steps: 25, CFG: 7.0, Seed: {seed}")
    print(f"  Resolution: 1024x1024")

    # 提交任务
    url = f"http://{COMFYUI_HOST}/prompt"
    resp = requests.post(url, json={"prompt": workflow})

    if resp.status_code != 200:
        print(f"Error: {resp.status_code} - {resp.text}")
        resp.raise_for_status()

    data = resp.json()
    prompt_id = data['prompt_id']
    print(f"Submitted: prompt_id={prompt_id}")

    # 等待完成
    print("Generating image...")
    result = poll_history(prompt_id)

    # 下载图片
    output_path = download_image(result['outputs'], OUTPUT_PATH)
    return output_path

if __name__ == "__main__":
    try:
        result_path = submit_and_wait()
        print(f"\nSUCCESS: Image saved to {result_path}")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
