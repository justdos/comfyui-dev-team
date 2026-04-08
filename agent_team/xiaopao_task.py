import json
import requests
import random
import time
from pathlib import Path

# 小跑接收任务：生成「美女雨中跑步」图片
# 调度者：小脑 | 评估者：小评

TASK = {
    "requirement": "美女在雨中跑步",
    "prompt": "beautiful young woman running in the rain, athletic fitness outfit, wet hair clinging to face, water droplets on skin, energetic running pose, dynamic motion, rainy night city street, wet pavement with reflections, street lamp glow, rain drops visible in air, cinematic lighting, photorealistic, high quality",
    "negative_prompt": "text, watermark, low quality, ugly, deformed, blurry, cartoon, anime, bad anatomy",
    "model": "animagine-xl-3.1.safetensors",
    "steps": 25,
    "resolution": "1024x1024",
    "cfg": 7
}

print(f"【小跑】接收任务：{TASK['requirement']}")
print(f"【小跑】使用模型：{TASK['model']}")
print(f"【小跑】步骤：{TASK['steps']}")

# 创建SDXL工作流
workflow = {
    '3': {
        'class_type': 'CheckpointLoaderSimple',
        'inputs': {'ckpt_name': TASK['model']}
    },
    '4': {
        'class_type': 'CLIPTextEncode',
        'inputs': {
            'text': TASK['prompt'],
            'clip': ['3', 1]
        }
    },
    '5': {
        'class_type': 'CLIPTextEncode',
        'inputs': {
            'text': TASK['negative_prompt'],
            'clip': ['3', 1]
        }
    },
    '6': {
        'class_type': 'KSampler',
        'inputs': {
            'model': ['3', 0],
            'seed': random.randint(1, 999999999999),
            'steps': TASK['steps'],
            'cfg': TASK['cfg'],
            'sampler_name': 'dpmpp_2m',
            'scheduler': 'karras',
            'positive': ['4', 0],
            'negative': ['5', 0],
            'latent_image': ['8', 0],
            'denoise': 1.0
        }
    },
    '7': {
        'class_type': 'VAEDecode',
        'inputs': {'samples': ['6', 0], 'vae': ['3', 2]}
    },
    '8': {
        'class_type': 'EmptyLatentImage',
        'inputs': {'width': 1024, 'height': 1024, 'batch_size': 1}
    },
    '9': {
        'class_type': 'SaveImage',
        'inputs': {'filename_prefix': 'xiaopao_run', 'images': ['7', 0]}
    }
}

# 提交到ComfyUI
url = 'http://localhost:8188/prompt'
resp = requests.post(url, json={'prompt': workflow})
print(f"【小跑】提交状态：{resp.status_code}")

if resp.status_code == 200:
    prompt_id = resp.json()['prompt_id']
    print(f"【小跑】任务ID：{prompt_id}")
    
    # 轮询等待
    history_url = f'http://localhost:8188/history/{prompt_id}'
    for i in range(60):
        time.sleep(5)
        hist = requests.get(history_url).json()
        if prompt_id in hist and hist[prompt_id].get('outputs'):
            print("【小跑】生成完成！")
            # 下载图片
            for node_id, output in hist[prompt_id]['outputs'].items():
                if 'images' in output:
                    img = output['images'][0]
                    view_url = f"http://localhost:8188/view?filename={img['filename']}&type=output&subfolder={img.get('subfolder', '')}"
                    img_resp = requests.get(view_url)
                    if img_resp.status_code == 200:
                        out_path = Path(r'D:/agent_team/xiaopao_result.jpg')
                        with open(out_path, 'wb') as f:
                            f.write(img_resp.content)
                        print(f"【小跑】图片已保存：{out_path}")
            break
        print(f"【小跑】等待中... ({i+1}/60)")
else:
    print(f"【小跑】错误：{resp.text}")
