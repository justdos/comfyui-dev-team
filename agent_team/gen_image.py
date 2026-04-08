import json
import requests
import random
import time
from pathlib import Path

# 创建SDXL工作流
workflow = {
    '3': {
        'class_type': 'CheckpointLoaderSimple',
        'inputs': {
            'ckpt_name': 'animagine-xl-3.1.safetensors'
        }
    },
    '4': {
        'class_type': 'CLIPTextEncode',
        'inputs': {
            'text': 'beautiful young woman running in the rain, athletic wear, wet hair, water droplets on skin, dynamic running pose, rainy night street, wet pavement reflections, street lamp glow, puddles splashing, dramatic cinematic lighting, rain atmosphere, photorealistic, high quality, detailed',
            'clip': ['3', 1]
        }
    },
    '5': {
        'class_type': 'CLIPTextEncode',
        'inputs': {
            'text': 'text, watermark, logo, low quality, ugly, deformed, blurry, cartoon, anime style, bad anatomy, worst quality',
            'clip': ['3', 1]
        }
    },
    '6': {
        'class_type': 'KSampler',
        'inputs': {
            'model': ['3', 0],
            'seed': random.randint(1, 999999999999),
            'steps': 25,
            'cfg': 7,
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
        'inputs': {
            'samples': ['6', 0],
            'vae': ['3', 2]
        }
    },
    '8': {
        'class_type': 'EmptyLatentImage',
        'inputs': {
            'width': 1024,
            'height': 1024,
            'batch_size': 1
        }
    },
    '9': {
        'class_type': 'SaveImage',
        'inputs': {
            'filename_prefix': 'rain_run',
            'images': ['7', 0]
        }
    }
}

# 提交
url = 'http://localhost:8188/prompt'
resp = requests.post(url, json={'prompt': workflow})
print('Status:', resp.status_code)
if resp.status_code == 200:
    data = resp.json()
    prompt_id = data['prompt_id']
    print('Success! prompt_id:', prompt_id)
    print('Waiting for completion...')
    
    # 轮询等待完成
    history_url = f'http://localhost:8188/history/{prompt_id}'
    for i in range(60):
        time.sleep(5)
        hist = requests.get(history_url).json()
        if prompt_id in hist and hist[prompt_id].get('outputs'):
            print('Job completed!')
            outputs = hist[prompt_id]['outputs']
            print('Output nodes:', list(outputs.keys()))
            
            # 下载图片
            for node_id, output in outputs.items():
                if 'images' in output:
                    for img in output['images']:
                        filename = img['filename']
                        subfolder = img.get('subfolder', '')
                        view_url = f'http://localhost:8188/view?filename={filename}&type=output&subfolder={subfolder}'
                        img_resp = requests.get(view_url)
                        if img_resp.status_code == 200:
                            out_path = Path(r'D:/agent_team/rain_run.jpg')
                            with open(out_path, 'wb') as f:
                                f.write(img_resp.content)
                            print(f'Saved to: {out_path}')
                            print('DONE!')
                        break
            break
        print(f'Waiting... ({i+1}/60)')
else:
    print('Error:', resp.text[:500])
