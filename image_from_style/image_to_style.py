import os
import time
from random import randint

from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from PIL import Image
import torch

from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image
import torch
from hidiffusion import apply_hidiffusion, remove_hidiffusion

import constant
#import text_to_image


# model_path = os.path.join(constant.base_model_path, 'images_models', 'playground-v2.5-1024px-aesthetic')
# model_path = os.path.join(constant.base_model_path, 'images_models', 'playground-v2.5-1024px-aesthetic')
# model_path = os.path.join(constant.base_model_path, 'images_models', 'animation3-1')


# model_path = os.path.join(constant.base_model_path, 'images_models', 'xl-base-1.0')
# scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")

# AutoPipelineForText2Image

# pipeline = StableDiffusionXLPipeline.from_pretrained(model_path, torch_dtype=torch.float16, add_watermarker=False,).to(
#     "cuda")

def load_ip_adapter(pipe):
    if pipe != None:
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
        pipe.enable_vae_tiling()


# scale = {
#     "down": {"block_2": [0.0, 1.0]},
#     "up": {"block_0": [0.0, 1.0, 0.0]},
# }

def update_scale_style(intput,pipe):
    if pipe != None:
        if intput:
            scale = {
                "up": {"block_0": [0.0, 1.0, 0.0]},
            }
            pipe.set_ip_adapter_scale(scale)
        else:
            scale = 1.0
            pipe.set_ip_adapter_scale(scale)


def update_scale_style_layout(intput,pipe):
    if pipe != None:

        if intput:
            scale = {
                "down": {"block_2": [0.0, 1.0]},
                "up": {"block_0": [0.0, 1.0, 0.0]},
            }
            pipe.set_ip_adapter_scale(scale)
        else:
            scale = 1.0
            pipe.set_ip_adapter_scale(scale)


# def update_style_hidiffusion(intput):
#     if text_to_image.pipe != None:
#         if intput:
#             apply_hidiffusion(text_to_image.pipe)
#         else:
#             remove_hidiffusion(text_to_image.pipe)


# # Optional. enable_xformers_memory_efficient_attention can save memory usage and increase inference speed. enable_model_cpu_offload and enable_vae_tiling can save memory usage.
# pipe.enable_xformers_memory_efficient_attention()
# pipe.enable_model_cpu_offload()
# pipe.enable_vae_tiling()


def generator_style_image(style_image, prompt, negative_prompt, w, h, seed, num_inference_steps, guidance_scale,
                          style_output,ip_model):
    if ip_model != None:
        if seed == 0:
            seed = randint(1, 1000000000000)
      #  seed = torch.Generator(device="cpu").manual_seed(seed)

        timestamp = int(time.time())
        save_restore_path = os.path.join(style_output, 'images', f'{timestamp}.jpg')
        os.makedirs(os.path.dirname(save_restore_path), exist_ok=True)

        images = ip_model.generate(
            pil_image=style_image,
            prompt=prompt,
            width=w,
            height=h,
            scale=1.0,
            num_samples=1,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )
        images[0].save(save_restore_path)
        return save_restore_path
