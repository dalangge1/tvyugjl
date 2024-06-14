import json
import os
import random
import time
from random import randint
import re
import numpy as np
from PIL import Image
import cv2
from ip_adapter import IPAdapterXL

from diffusers import DiffusionPipeline, AutoencoderKL, LCMScheduler, ControlNetModel, \
    StableDiffusionXLControlNetImg2ImgPipeline, StableDiffusionXLPipeline

from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
)

import constant
from google_gemini_api import send_message, send_translate_message, send_prompt_message, send_en_prompt_message, \
    send_en_prompt_danbooru_style

# from image_to_video.image_to_video import generate_video, fcm_generate_video

import torch

from image_from_style.image_to_style import load_ip_adapter, update_scale_style, update_scale_style_layout, \
    generator_style_image
# from llama3 import llama3_send_message
# import image_to_style
# from image_to_style.image_to_style import load_ip_adapter, update_scale_style, update_scale_style_layout

# from llama3 import llama3_send_message, llama3_send_message_all
from participle import participe_def
# from translation import translation_prompt

from hidiffusion import apply_hidiffusion, remove_hidiffusion

from typing import Callable, Dict, Optional, Tuple

from process_image import process_image_style_for_image
from translation import translation_prompt

global pipe

global controller

global ip_model

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def model_cpu_offload_fn(cpu_offload):
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)


def load_controller(model_path):
    global pipe
    pipe = None


with_high_res_fix = False
high_res_fix = [{'timestep': 600, 'scale_factor': 0.5, 'block_num': 1}]

def load_animation(model_path):
    global pipe
    pipe = None
    progress = "加载模型: {}。".format(model_path)
    try:
        remove_part = "images_models"
        index = model_path.find(remove_part)
        if index == -1:
            progress += "vae加载失败"
            return progress
        new_path = os.path.normpath(model_path[:index])

        xl_vae_path = os.path.join(new_path, 'xl_vae', 'sdxl-vae-fp16-fix')
        vae = AutoencoderKL.from_pretrained(
            xl_vae_path,
            torch_dtype=torch.float16,
        )

        pipe = DiffusionPipeline.from_pretrained(
            model_path,
            vae=vae,
            torch_dtype=torch.float16,
            custom_pipeline="lpw_stable_diffusion_xl",
            #custom_pipeline="high_res_fix",
            variant="fp16",
            use_safetensors=True,
            add_watermarker=False,
            high_res_fix=high_res_fix if with_high_res_fix else None,
        )
        progress += "模型加载完成。"
        pipe.to(device)
        progress += "模型已移至设备 {}。".format(device)
    except Exception as e:
        progress += "模型加载失败: {}".format(str(e))
    return progress


def load_playground(model_path):
    global pipe
    pipe = None
    progress = "加载模型中: {}\n".format(model_path)
    try:
        pipe = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            add_watermarker=False,
            high_res_fix=high_res_fix if with_high_res_fix else None,
        )
        progress += "模型加载完成。"
        pipe.to(device)
        progress += "模型已移至设备 {}。".format(device)
    except Exception as e:
        progress += "模型加载失败: {}".format(str(e))
    return progress


def load_sdxl(model_path):
    global pipe
    pipe = None
    progress = "加载模型中: {}\n".format(model_path)
    try:
        pipe = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            add_watermarker=False,
            high_res_fix=high_res_fix if with_high_res_fix else None,
        )
        progress += "模型加载完成。"
        pipe.to(device)
        progress += "模型已移至设备 {}。".format(device)
    except Exception as e:
        progress += "模型加载失败: {}".format(str(e))
    return progress


def load_pipeline(model_path):
    if "animation" in model_path:
        return load_animation(model_path)
    elif "playground" in model_path:
        return load_playground(model_path)
    else:
        return load_sdxl(model_path)


def image_load_ip_adapter(model_path, select_model_path):
    global pipe, ip_model
    pipe = None
    ip_model = None
    progress = "加载模型中: {}\n".format(select_model_path)
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            select_model_path,
            torch_dtype=torch.float16,
            add_watermarker=False,
            high_res_fix=high_res_fix if with_high_res_fix else None,
        )
        pipe.enable_vae_tiling()

        image_encoder_path = os.path.join(model_path, 'ip_ad_sdxl_models', 'image_encoder')
        ip_ckpt = os.path.join(model_path, 'ip_ad_sdxl_models', 'ip-adapter_sdxl.bin')
        ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device, target_blocks=["up_blocks.0.attentions.1"])
        # pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
        # pipe.enable_vae_tiling()

        progress += "模型已移至设备 {}。".format(device)
    except Exception as e:
        progress += "模型加载失败: {}".format(str(e))
    return progress


def image_update_scale_style(i):
    global pipe
    update_scale_style(i, pipe)


def image_update_scale_style_layout(i):
    global pipe
    update_scale_style_layout(i, pipe)


def generator_style_for_image(style_image, prompt, negative_prompt, w, h, seed, num_inference_steps, guidance_scale,
                              style_output):
    global ip_model
    image_path = generator_style_image(style_image, prompt, negative_prompt, w, h, seed, num_inference_steps,
                                       guidance_scale,
                                       style_output, ip_model)

    return image_path


# def image_update_style_hidiffusion(i):
#     pass


def load_model_to_controller(select_model_path, model_path):
    global pipe, controller
    pipe = None
    controller = None
    canny_model_path = os.path.join(model_path, "controlnet-canny-sdxl-1.0")
    progress = "加载模型: {},{}。".format(select_model_path, canny_model_path)

    try:
        controlnet = ControlNetModel.from_pretrained(
            canny_model_path, torch_dtype=torch.float16, variant="fp16"
        ).to(device)
        scheduler = DDIMScheduler.from_pretrained(select_model_path, subfolder="scheduler")

        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            select_model_path,
            controlnet=controlnet,
            scheduler=scheduler,
            torch_dtype=torch.float16,
        ).to(device)
        progress += "模型加载完成。"
        # pipe.to(device)
        progress += "模型已移至设备 {}。".format(device)
    except Exception as e:
        progress += "模型加载失败: {}".format(str(e))
    return progress


def canny_to_image(ori_image_path, controlnet_conditioning_scale, hidiffusion,
                   prompt, negative_prompt, width, height, num_inference_steps, guidance_scale, output_path):
    # if hidiffusion:
    #     apply_hidiffusion(pipe)
    # pipe.enable_xformers_memory_efficient_attention()

    ori_image = Image.open(ori_image_path)

    # get canny image
    image = np.array(ori_image)
    image = cv2.Canny(image, 50, 120)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)

    controlnet_conditioning_scale = controlnet_conditioning_scale

    timestamp = int(time.time())
    save_restore_path = os.path.join(output_path, 'images', f'{timestamp}.jpg')
    os.makedirs(os.path.dirname(save_restore_path), exist_ok=True)

    image = pipe(prompt,
                 image=ori_image,
                 control_image=canny_image,
                 height=height,
                 width=width,
                 # strength=0.99,
                 num_inference_steps=num_inference_steps,
                 controlnet_conditioning_scale=controlnet_conditioning_scale,
                 guidance_scale=guidance_scale,
                 negative_prompt=negative_prompt,
                 eta=1.0
                 ).images[0].save(save_restore_path)

    return save_restore_path


# from xformers.ops import MemoryEfficientAttentionFlashAttentionOp


# pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
# hidiffusion
# pipe.enable_xformers_memory_efficient_attention()

def update_hidiffusion(input):
    if input == True:
        apply_hidiffusion(pipe)
        print("启动hidiffusion")
    else:
        remove_hidiffusion(pipe)


def update_check_freU(input, load_model_path, hidiffusion):
    if input == True:
        pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
        print("启动freeU")
    else:
        pipe.disable_freeu()
        # load_pipeline(load_model_path)
        # update_hidiffusion(hidiffusion)


def update_lcm_lora(input):
    # if input == True:
    # apply_hidiffusion(pipe)
    # else:
    # remove_hidiffusion(pipe)
    pass


def update_scheduler_setting(input):
    print(input)
    if input == "默认":
        return
    if input == "LCM":
        # pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        # pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
        return
    pipe.scheduler = get_scheduler(pipe.scheduler.config, input)
    print("启动scheduler-{}".format(input))


def change_enable(size):
    width, height = 1920, 1080
    if size in constant.size_mapping:
        width, height = constant.size_mapping[size]
    if width == 2048 or width == 4096 or width == 3072:
        pipe.enable_vae_slicing()


def get_lora_list():
    lora_json = os.path.join(constant.base_model_path, 'images_models', 'lora', 'model_index.json')
    with open(lora_json, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data["loras"]


def get_person_list():
    person_lora = os.path.join(constant.base_model_path, 'images_models', 'lora', 'person_lora.json')
    with open(person_lora, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def get_person_style_list():
    person_lora = os.path.join(constant.base_model_path, 'images_models', 'lora', 'person_style_lora.json')
    with open(person_lora, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


# 单一生成
def text_to_image(user_prompt, lac_prompt, negative_prompt, size, seed, num_inference_steps, generator,
                  model_path=constant.base_model_path,
                  output_path=r'C:\Users\asus\Desktop\天生入道'):
    print("当前适配器{}".format(pipe.get_active_adapters()))
    torch.cuda.empty_cache()
    width, height = 1920, 1080
    if size in constant.size_mapping:
        width, height = constant.size_mapping[size]

    if seed == 0:
        seed = randint(1, 100000000000000)

    person_prompt = ""
    person_array, user_prompt = constant.cut_prompt(user_prompt, "person")
    if len(person_array) > 0:
        for object in person_array:
            object = object.replace('，', ',').strip()
            split_str = object.split(':')
            name = split_str[1]
            person = get_person_list()
            person_prompt = person[name] + ','

    effects_array, user_prompt = constant.cut_prompt(user_prompt, "lora")
    lora_scale = 0
    lora_prompt = ''
    if len(effects_array) > 0:
        lora_scale = 1.0
        loras = get_lora_list()
        adapter_name_array = []
        lora_scale_array = []
        for object in effects_array:
            object = object.replace('，', ',').strip()
            split_str = object.split(':')
            # name = split_str[1]
            lora_name = split_str[1].split(',')[0]
            lora_scales = float(split_str[1].split(',')[1])
            lora_dirs_name = [line for line in loras if line["name"] == lora_name]
            lora_dirs = os.path.join(model_path, 'lora', lora_dirs_name[0]["path"])
            adapter_name = lora_dirs_name[0]["adapter"]
            lora_prompt = lora_prompt + adapter_name + ","
            pipe.load_lora_weights(lora_dirs, weight_name=lora_dirs_name[0]["model"], adapter_name=adapter_name)

            lora_scale_array.append(lora_scales)
            adapter_name_array.append(adapter_name)
        # pipe.fuse_lora(lora_scale=lora_scales)
        pipe.set_adapters(adapter_name_array, adapter_weights=lora_scale_array)

    user_prompt = send_translate_message(user_prompt, lac_prompt)
    # print(prompt)
    # 'masterpiece, best quality, detailed, 8k, very aesthetic,'
    prompt = person_prompt + lora_prompt + user_prompt + "," + lac_prompt

    seed = torch.Generator(device="cpu").manual_seed(seed)

    timestamp = int(time.time())
    save_restore_path = os.path.join(output_path, 'images', f'{timestamp}.jpg')
    os.makedirs(os.path.dirname(save_restore_path), exist_ok=True)

    # pipe.enable_sequential_cpu_offload()
    print(prompt)

    if width in {2048, 3072, 4096}:
        # apply_hidiffusion(pipe)
        # pipe.enable_model_cpu_offload()
        pass
    else:
        # 确保在不需要时禁用 CPU 卸载
        # pipe.to(device)
        pass

    print("当前适配器{}".format(pipe.get_active_adapters()))
    if lora_scale == 0:
        #  pipe.unfuse_lora()
        pipe.disable_lora()
    print("当前适配器{}".format(pipe.get_active_adapters()))

    image = pipe(prompt,
                 negative_prompt=negative_prompt,
                 cross_attention_kwargs={"scale": lora_scale},
                 width=width,
                 height=height,
                 num_inference_steps=num_inference_steps,
                 guidance_scale=generator,
                 generator=seed,
                 ).images[0].save(save_restore_path)

    return save_restore_path




def get_image_name(seed, steps, scale, prompt,timestamp):
    # 将参数拼接成一个字符串
    name = f"{timestamp}_{seed}_{steps}_{scale}_{prompt}"
    return name

# 一键生成
def generate_image(prompt, size_width, size_height, output_path, style, prompt_style, person_prompt='',
                   model_path=constant.base_image_model_path, ):
    torch.cuda.empty_cache()

    seed = randint(constant.var_seed, 1000009999)
    seed_number = seed
    seed = torch.Generator(device="cpu").manual_seed(seed)

    person_array, prompt = constant.cut_prompt(prompt, "person")
    if len(person_array) > 0:
        for object in person_array:
            object = object.replace('，', ',').strip()
            split_str = object.split(':')
            name = split_str[1]
            person = get_person_list()
            person_prompt = person[name] + ','

    effects_array, prompt = constant.cut_prompt(prompt, "lora")
    lora_scale = 0
    lora_prompt = ''
    if len(effects_array) > 0:
        lora_scale = 1.0
        loras = get_lora_list()
        adapter_name_array = []
        lora_scale_array = []
        for object in effects_array:
            object = object.replace('，', ',').strip()
            split_str = object.split(':')
            # name = split_str[1]
            lora_name = split_str[1].split(',')[0]
            lora_scales = float(split_str[1].split(',')[1])
            lora_dirs_name = [line for line in loras if line["name"] == lora_name]
            lora_dirs = os.path.join(model_path, 'lora', lora_dirs_name[0]["path"])
            adapter_name = lora_dirs_name[0]["adapter"]
            lora_prompt = lora_prompt + adapter_name + ","
            pipe.load_lora_weights(lora_dirs, weight_name=lora_dirs_name[0]["model"], adapter_name=adapter_name)

            lora_scale_array.append(lora_scales)
            adapter_name_array.append(adapter_name)
        # pipe.fuse_lora(lora_scale=lora_scales)
        pipe.set_adapters(adapter_name_array, adapter_weights=lora_scale_array)

    ## llama3
    ##en_prompt = send_en_prompt_danbooru_style(prompt, prompt_style)
    # response = send_prompt_message(prompt, prompt_text)
    ##print(en_prompt)
    ## 翻译模型
    # en_prompt = translation_prompt(prompt)
    # en_prompt = prompt_text + en_prompt[0]
    # print(en_prompt)

    ## 分词后翻译
    # process_text = participe_def(prompt_style + ',' + prompt)
    # print(process_text)
    # en_prompt = translation_prompt(process_text)[0]

    ## gemini api
    en_prompt = send_message(prompt, prompt_style)
    en_prompt = en_prompt.replace("\n", "").replace("*", "")

    # animation    #palygruand2.5 基础提示词 masterpiece, best quality, detailed, 8k
    # style + ',' +
    prompt = style + ',' + "masterpiece, best quality, very aesthetic, extremely detailed" + person_prompt + "," + en_prompt
    negative_prompt = 'nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality,extra fingers, ' \
                      'jpeg artifacts, low quality,  bad anatomy, skin spots, acnes, skin blemishes, ' \
                      'facing away, looking away, tilted head, bad hands, missing fingers,' \
                      ' bad feet, poorly drawn hands, poorly drawn face, mutation, deformed, ' \
                      'extra fingers, extra limbs, extra arms, extra legs, malformed limbs, fused fingers, ' \
                      'too many fingers, long neck, cross-eyed, mutated hands, bad body, ' \
                      'bad proportions, gross proportions, missing arms, missing legs, extra foot,' \
                      ' teethcroppe, blurry, cropped,watermark, unfinished, displeasing, oldest, early,' \
                      ' chromatic aberration, signature, extra digits, artistic error, username, scan, ' \
                      'ghosting, [abstract]'

    timestamp = int(time.time())
    image_name = get_image_name(seed_number, constant.var_num_inference_steps, constant.var_guidance_scale, prompt,timestamp)
    image_name = image_name.replace(".", "").replace(", ", "_").replace(" ", "_").replace(",", "_")
    save_restore_path = os.path.join(output_path, 'images', f'{image_name}.jpg')
    os.makedirs(os.path.dirname(save_restore_path), exist_ok=True)

    pipe(prompt,
         negative_prompt=negative_prompt,
         # timesteps=sampling_schedule,
         cross_attention_kwargs={"scale": lora_scale},
         width=size_width,
         height=size_height,
         num_inference_steps=constant.var_num_inference_steps,
         guidance_scale=constant.var_guidance_scale,
         generator=seed,
         ).images[0].save(save_restore_path)

    if len(effects_array) > 0:
        pipe.unload_lora_weights()

    return save_restore_path


# 一键风格生成
def generate_style_image(prompt, size_width, size_height, output_path, style, prompt_style, person_prompt='',
                         model_path=constant.base_image_model_path, ):
    torch.cuda.empty_cache()
    image_style_path = ''
    # seed = randint(constant.var_seed, 1000009999)
    # seed = torch.Generator(device="cpu").manual_seed(seed)

    person_array, prompt = constant.cut_prompt(prompt, "person")
    if len(person_array) > 0:
        for object in person_array:
            object = object.replace('，', ',').strip()
            split_str = object.split(':')
            name = split_str[1]
            person = get_person_style_list()
            person_prompt = person[name][0] + ','
            image_style_path = os.path.join(model_path, 'lora', person[name][1])

    effects_array, prompt = constant.cut_prompt(prompt, "lora")
    lora_scale = 0
    lora_prompt = ''
    if len(effects_array) > 0:
        lora_scale = 1.0
        loras = get_lora_list()
        adapter_name_array = []
        lora_scale_array = []
        for object in effects_array:
            object = object.replace('，', ',').strip()
            split_str = object.split(':')
            # name = split_str[1]
            lora_name = split_str[1].split(',')[0]
            lora_scales = float(split_str[1].split(',')[1])
            lora_dirs_name = [line for line in loras if line["name"] == lora_name]
            lora_dirs = os.path.join(model_path, 'lora', lora_dirs_name[0]["path"])
            adapter_name = lora_dirs_name[0]["adapter"]
            lora_prompt = lora_prompt + adapter_name + ","
            pipe.load_lora_weights(lora_dirs, weight_name=lora_dirs_name[0]["model"], adapter_name=adapter_name)

            lora_scale_array.append(lora_scales)
            adapter_name_array.append(adapter_name)
        # pipe.fuse_lora(lora_scale=lora_scales)
        pipe.set_adapters(adapter_name_array, adapter_weights=lora_scale_array)

    ## llama3
    # en_prompt = send_en_prompt_danbooru_style(prompt, prompt_style)
    ## en_prompt = llama3_send_message(prompt_style,prompt)

    process_text = participe_def(prompt)
    print(process_text)
    en_prompt = translation_prompt(process_text)[0]
    print(en_prompt)

    # style + ',' +
    prompt = style + ',' + "masterpiece, best quality, very aesthetic, extremely detailed" + person_prompt + "," + en_prompt
    negative_prompt = 'nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality,extra fingers, ' \
                      'jpeg artifacts, low quality,  bad anatomy, skin spots, acnes, skin blemishes, ' \
                      'facing away, looking away, tilted head, bad hands, missing fingers,' \
                      ' bad feet, poorly drawn hands, poorly drawn face, mutation, deformed, ' \
                      'extra fingers, extra limbs, extra arms, extra legs, malformed limbs, fused fingers, ' \
                      'too many fingers, long neck, cross-eyed, mutated hands, bad body, ' \
                      'bad proportions, gross proportions, missing arms, missing legs, extra foot,' \
                      'blurry, cropped,watermark, unfinished, displeasing, oldest, early,' \
                      ' chromatic aberration, signature, extra digits, artistic error, username, scan, ' \
                      'ghosting, [abstract]'

    if image_style_path == '':
        image_style_dir = os.path.join(model_path, 'lora', 'style_image')
        image_files = os.listdir(image_style_dir)
        # 过滤出图片文件（假设图片格式为 jpg, png, jpeg 等）
        image_files = [file for file in image_files if file.endswith(('.jpg', '.jpeg', '.png'))]
        # 如果目录中有图片文件，随机选择一张
        if image_files:
            random_image = random.choice(image_files)
            image_style_path = os.path.join(image_style_dir, random_image)
    print(image_style_path)
    save_restore_path = generator_style_for_image(Image.open(image_style_path), prompt, negative_prompt, size_width,
                                                  size_height, constant.var_seed,
                                                  constant.var_num_inference_steps,
                                                  constant.var_guidance_scale, output_path)

    process_image_path = process_image_style_for_image(save_restore_path, output_path, 1.3, 1.1)
    return process_image_path


def generate_image_test(en_prompt, style, size_width, size_height, seed, num_inference_steps, guidance_scale,
                        output_path, novel_style):
    # animation novel2_style
    prompt = style + ", masterpiece, best quality, very aesthetic,  extremely detailed," + en_prompt
    negative_prompt = 'nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality,' \
                      ' jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest,' \
                      ' early, chromatic aberration, signature, extra digits, artistic error, ' \
                      'username, NSFW, scan, [abstract]'

    if seed == 0:
        seed = randint(1, 100000000000000)
    seed = torch.Generator(device="cpu").manual_seed(seed)

    timestamp = int(time.time())
    save_restore_path = os.path.join(output_path, f'{timestamp}.jpg')
    # os.makedirs(os.path.dirname(save_restore_path), exist_ok=True)

    pipe(prompt,
         negative_prompt=negative_prompt,
         width=size_width,
         height=size_height,
         num_inference_steps=num_inference_steps,
         guidance_scale=guidance_scale,
         generator=seed,
         ).images[0].save(save_restore_path)

    # 调整图片亮度和饱和度
    process_image_path = process_image_style_for_image(save_restore_path, output_path, 1.2, 1.1)

    return process_image_path


def generate_image_video(prompt, size_width, size_height, output_path, style, prompt_style):
    # process_text = participe_def(prompt)
    # print(process_text)
    # en_prompt = translation_prompt(prompt)
    en_prompt = send_message(prompt, prompt_style)
    print(en_prompt)
    en_prompt = en_prompt.replace("\n", "").replace("*", "")
    # en_prompt = translation_prompt(prompt)
    # animation
    prompt = style + ", masterpiece, best quality, very aesthetic, absurdres, detailed, 8k" + en_prompt[0]
    negative_prompt = 'nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality,' \
                      ' jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest,' \
                      ' early, chromatic aberration, signature, extra digits, artistic error, ' \
                      'username, scan, [abstract]'

    seed = randint(1, 100000000000000)
    seed = torch.Generator(device="cpu").manual_seed(seed)

    timestamp = int(time.time())
    save_restore_path = os.path.join(output_path, 'images', f'{timestamp}.jpg')
    os.makedirs(os.path.dirname(save_restore_path), exist_ok=True)

    # sampling_schedule = [999,845,730,587,443,310,193,116,53,13,0]

    # pipe(prompt,
    #      negative_prompt=negative_prompt,
    #      width=size_width,
    #      height=size_height,
    #      num_inference_steps=25,
    #      guidance_scale=3.0,
    #      generator=seed,
    #      ).images[0].save(save_restore_path)

    # save_video_path = fcm_generate_video(save_restore_path, output_folder=output_path)
    save_video_path = ""
    # save_video_path = generate_video(save_restore_path,output_path)
    return save_restore_path, save_video_path


def get_scheduler(scheduler_config: Dict, name: str) -> Optional[Callable]:
    scheduler_factory_map = {
        "DPM++ 2M Karras": lambda: DPMSolverMultistepScheduler.from_config(
            scheduler_config, use_karras_sigmas=True
        ),
        "DPM++ SDE Karras": lambda: DPMSolverSinglestepScheduler.from_config(
            scheduler_config, use_karras_sigmas=True
        ),
        "DPM++ 2M SDE Karras": lambda: DPMSolverMultistepScheduler.from_config(
            scheduler_config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++"
        ),
        "Euler": lambda: EulerDiscreteScheduler.from_config(scheduler_config),
        "Euler a": lambda: EulerAncestralDiscreteScheduler.from_config(
            scheduler_config
        ),
        "DDIM": lambda: DDIMScheduler.from_config(scheduler_config),
    }
    return scheduler_factory_map.get(name, lambda: None)()
