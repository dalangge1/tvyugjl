import os
import time
from random import random, randint
from typing import Optional

import torch
from diffusers.utils import load_image, export_to_video
from safetensors import safe_open

import constant
from diffusers import StableVideoDiffusionPipeline
from glob import glob

from PIL import Image

from image_to_video.lcm_scheduler import AnimateLCMSVDStochasticIterativeScheduler

model_video_path = os.path.join(constant.base_model_path, 'images_models', 'stable-video-diffusion-img2vid-xt-1-1')



def model_select(selected_file):
    print("load model weights", selected_file)
    pipe.unet.cpu()
    file_path = os.path.join(model_video_path, selected_file)
    state_dict = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    missing, unexpected = pipe.unet.load_state_dict(state_dict, strict=True)
    pipe.unet.cuda()
    del state_dict
    return


noise_scheduler = AnimateLCMSVDStochasticIterativeScheduler(
    num_train_timesteps=40,
    sigma_min=0.002,
    sigma_max=700.0,
    sigma_data=1.0,
    s_noise=1.0,
    rho=7,
    clip_denoised=False,
)
pipe = StableVideoDiffusionPipeline.from_pretrained(
    model_video_path,
    scheduler=noise_scheduler,
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe.to("cuda")
pipe.enable_model_cpu_offload()  # for smaller cost
model_select("AnimateLCM-SVD-xt-1.1.safetensors")

max_64_bit_int = 2**63 - 1



#  input1 = gr.Textbox(label="输入图片数据集",value="F:\sdxl_images")
#             input2 = gr.Slider(minimum=0, maximum=1280, value=1280, step=1, label="w")
#             input3 = gr.Slider(minimum=0, maximum=1024, value=720, step=1, label="h")
def fcm_generate_video_form_paht(folder_path, w, h):
  # image_paths = []  # 用于存储图片路径的列表

    # 遍历文件夹中的所有文件和子文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 检查文件扩展名是否为常见的图片格式
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                # 构造完整的图片路径并添加到列表中
                image_path = os.path.join(root, file)
                print(image_path)
                for i in range(3):
                    fcm_generate_video(image_path, width=w, height=h, output_folder=folder_path)



def fcm_generate_video(
    image_path,
    seed: int = 42,
    randomize_seed: bool = True,
    motion_bucket_id: int = 80,
    fps_id: int = 8,
    max_guidance_scale: float = 1.2,
    min_guidance_scale: float = 1,
    width: int = 1024,
    height: int = 576,
    # width: int = 324,
    # height: int = 576,
    num_inference_steps: int = 4,
    decoding_t: int = 4,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    output_folder: str = "outputs_gradio",
):
    image = fn_resize_image(image_path, (width, height))
    if image.mode == "RGBA":
        image = image.convert("RGB")

    if randomize_seed:
        seed = randint(0, max_64_bit_int)
    generator = torch.manual_seed(seed)

    timestamp = int(time.time())
    video_path = os.path.join(output_folder, 'image_video', f'{timestamp}.mp4')
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    with torch.autocast("cuda"):
        frames = pipe(
            image,
            decode_chunk_size=decoding_t,
            generator=generator,
            motion_bucket_id=motion_bucket_id,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            min_guidance_scale=min_guidance_scale,
            max_guidance_scale=max_guidance_scale,
        ).frames[0]
    export_to_video(frames, video_path, fps=fps_id)
    torch.manual_seed(seed)
    return video_path



def fn_resize_image(image_path, output_size=(1024, 576)):
    image = load_image(image_path)
    # Calculate aspect ratios
    target_aspect = output_size[0] / output_size[1]  # Aspect ratio of the desired size
    image_aspect = image.width / image.height  # Aspect ratio of the original image

    # Resize then crop if the original image is larger
    if image_aspect > target_aspect:
        # Resize the image to match the target height, maintaining aspect ratio
        new_height = output_size[1]
        new_width = int(new_height * image_aspect)
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        # Calculate coordinates for cropping
        left = (new_width - output_size[0]) / 2
        top = 0
        right = (new_width + output_size[0]) / 2
        bottom = output_size[1]
    else:
        # Resize the image to match the target width, maintaining aspect ratio
        new_width = output_size[0]
        new_height = int(new_width / image_aspect)
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        # Calculate coordinates for cropping
        left = 0
        top = (new_height - output_size[1]) / 2
        right = output_size[0]
        bottom = (new_height + output_size[1]) / 2

    # Crop the image
    cropped_image = resized_image.crop((left, top, right, bottom))
    return cropped_image



# pipeline = StableVideoDiffusionPipeline.from_pretrained(
#     model_video_path, torch_dtype=torch.float16, variant="fp16"
# )
# pipeline.enable_model_cpu_offload()
#
#

def generate_video(image_path, output_path):
    generator = torch.manual_seed(42)
    # seed = randint(1, 10000000)
    # seed = torch.Generator(device="cpu").manual_seed(seed)

    timestamp = int(time.time())
    save_restore_path = os.path.join(output_path, 'image_video', f'{timestamp}.mp4')
    os.makedirs(os.path.dirname(save_restore_path), exist_ok=True)

    image = load_image(image_path)
    image = image.resize((1024, 576))
    frames = pipe(image, decode_chunk_size=8, generator=generator).frames[0]
    export_to_video(frames, save_restore_path, fps=7)

    return save_restore_path
