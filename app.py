import datetime
import json
import os
import random

import gradio as gr
from scipy.io import wavfile

import constant
import re
import torch
import time

from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav

from process_audio import apply_effects, add_reverb_0, process_audio

from google_gemini_api import send_prompt_message
from process_image import process_image_style_for_image

# from llama3 import llama3_send_message, llama3_send_message_all, llama3_send_message_novel_text

from process_video import video_process, merge_video

from tts import audio_process

# from GPT_SoVITS.tools.uvr5.webui import i18n

from text_to_image import generate_image, generate_image_test, text_to_image, change_enable, load_pipeline, \
    update_hidiffusion, update_check_freU, update_scheduler_setting, canny_to_image, load_model_to_controller, \
    model_cpu_offload_fn, generator_style_for_image, image_load_ip_adapter, image_update_scale_style, \
    image_update_scale_style_layout, generate_style_image

from video_merge import merge_video_effects

# from image_to_video.image_to_video import fcm_generate_video, fcm_generate_video_form_paht

def split_text(text):
    sentences = re.split(r'(?<=[.?!。？！])', text)
    filtered_sentences = []
    current_sentence = ''
    for sentence in sentences:
        current_sentence += sentence
        if len(current_sentence) >= 10:
            filtered_sentences.append(current_sentence.strip())
            current_sentence = ''
    return filtered_sentences


def process_text(text, lora=""):
    text = constant.replace_text(text)
    return text, ""


def process_text_and_name(text, lora=""):
    # text = llama3_send_message_novel_text(text)
    # 1.去掉 多余 字符 空格 回车
    text = constant.replace_text(text)
    print(text)
    processed_texts = []
    # 2.将 人物 的词更换 sd人物的词
    # lines = split_text(text)
    # for t in lines:
    #     t = participe_perpro_name(text)
    #     processed_texts.append(t)
    # 整理出来的人名字
    return text, ""
    # t = participe_perpro_name(text)
    # return text, t


def batch_image_generate(input1, input2, input3, input4, input5, num_inference_steps, guidance_scale, seed):
    torch.cuda.empty_cache()

    # 获取视频尺寸
    width, height = 1920, 1080
    if input4 in constant.size_mapping:
        width, height = constant.size_mapping[input4]
    input1 = process_text(input1)
    # 文案分段
    lines = split_text(input1[0])
    for text in lines:
        cleaned_text = text.replace(" ", "").replace("\n", "")
        effects_array, prompt = constant.cut_prompt(cleaned_text, "lora")
        prompt_text = ''
        if input3 == "世情":
            # "现实生活的社会风情和人情世故"
            prompt_text = "In line with the social customs and worldliness of real lifee"
        if input3 == "仙侠":
            # "符合仙人、神话传说、修真者等奇幻元素"
            prompt_text = "A painting style that is consistent with fantasy elements such as immortals, myths and legends, and cultivators"
        if input3 == "古言":
            # "符合以历史人物、古代社会风情"
            prompt_text = "In line with historical figures and ancient social customs"
        if input3 == "悬疑":
            # 悬疑图片，紧张的氛围
            prompt_text = "In line with the suspenseful pictures and tense atmosphere"
        if input3 == "现言":
            # 符合现代社会背景的现代生活
            prompt_text = "Modern life in line with the background of modern society"
        # en_prompt = llama3_send_message(prompt_text, prompt)
        en_prompt = prompt_text + prompt
        for i in range(2):
            generate_image_test(en_prompt, width, height, input2, input5, num_inference_steps, guidance_scale, seed)


async def single_video_generate(size, text, images_path, vioce, input4_1, effects, output_path, effects_text):
    torch.cuda.empty_cache()
    # 获取视频尺寸
    width, height = 1920, 1080
    if size in constant.size_mapping:
        width, height = constant.size_mapping[size]

    # 通过文案生成图片
    lines = get_vicoe_lines()
    dict = [line for line in lines if line["name"] == vioce][0]
    type = dict["type"]
    audio_path = output_path
    if type == "edge_tts":
        rate = dict["rate"]
        volume = dict["volume"]
        audio_path = await audio_process(text, dict["name"], output_path, rate, volume)
    elif type == "GPT_SoVITS":
        gpt_model = dict["gpt_model"]
        so_vist = dict["so_vist"]
        change_gpt_weights(gpt_model)
        change_sovits_weights(so_vist)
        text = text
        ref_wav_path = dict["reference_audio"]
        prompt_text = dict["reference_text"]
        prompt_language = dict["reference_language"]
        text_language = dict["composite_language"]
        how_to_cut = ''
        top_k = dict["top_k"],
        top_p = dict["top_p"],
        temperature = dict["temperature"],
        ref_text_free = False
        result = next(
            get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut, top_k, top_p,
                        temperature, ref_text_free))

        timestamp = int(time.time())
        file_path = os.path.join(output_path, 'audio', f"{timestamp}.wav")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        audio_path = constant.save_numpy_to_wav(result[1], result[0], file_path)
    # 音频增强和切割
    print(input4_1)
    if "切割" in input4_1 or "增强" in input4_1:
        audio_path = process_audio(audio_path, input4_1)
    # if "增强" in input4_1:
    #     audio_path = enhance_effects(audio_path)
    # if "切割" in input4_1:
    #     audio_path = apply_effects(audio_path)

    text = effects_text + text
    output_file = video_process(text, images_path, audio_path, width, height, effects, output_path)
    return output_file


async def video_generate(input1, input2, input3, input4, input5, input6, input7, input4_1, input8, effects_input,
                         person_input, style_input,
                         input_text_1="",
                         input_text_2=""):
    torch.cuda.empty_cache()

    # 获取视频尺寸
    width, height = 1920, 1080
    if input5 in constant.size_mapping:
        width, height = constant.size_mapping[input5]
    input1 = process_text(input1)

    # 文案分段
    lines = split_text(input1[0])
    for text in lines:
        cleaned_text = text.replace(" ", "").replace("\n", "")

        person_prompt = ''
        # 自动补全视效
        effect_key = ''
        person_key = ''
        if effects_input:
            if len(cleaned_text) > 16:
                effect_key = get_effect_list_for_key()
            else:
                effect_key = get_transition_effect_list_for_key()

        if person_input:
            if style_input:
                person_key, person_prompt, image_style_image_path = get_style_person_list_for_key(cleaned_text)
            else:
                person_prompt = get_person_list_for_key(cleaned_text)
        cleaned_text = effect_key + person_key + cleaned_text
        images_path = None
        # 出图风格化
        if style_input:
            images_path = generate_style_image(cleaned_text, width, height, input7, input2, input8,
                                               person_prompt=person_prompt)

        else:
            # 通过文案生成图片
            images_path = generate_image(cleaned_text, width, height, input7, input2, input8,
                                         person_prompt=person_prompt)
        prompt = re.sub(r'\[[^\]]+\]', '', cleaned_text)
        print(prompt)
        lines = get_vicoe_lines()
        dict = [line for line in lines if line["name"] == input4][0]
        type = dict["type"]
        audio_path = input7
        if type == "edge_tts":
            rate = dict["rate"]
            volume = dict["volume"]
            audio_path = await audio_process(prompt, dict["name"], input7, rate, volume)
        elif type == "GPT_SoVITS":
            gpt_model = dict["gpt_model"]
            so_vist = dict["so_vist"]
            change_gpt_weights(gpt_model)
            change_sovits_weights(so_vist)
            text = prompt
            ref_wav_path = dict["reference_audio"]
            prompt_text = dict["reference_text"]
            prompt_language = dict["reference_language"]
            text_language = dict["composite_language"]
            # how_to_cut = i18n("cut_text")
            how_to_cut = ''
            top_k = dict["top_k"],
            top_p = dict["top_p"],
            temperature = dict["temperature"],
            ref_text_free = False
            result = next(
                get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut, top_k, top_p,
                            temperature, ref_text_free))

            timestamp = int(time.time())
            file_path = os.path.join(input7, 'audio', f"{timestamp}.wav")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            audio_path = constant.save_numpy_to_wav(result[1], result[0], file_path)
        # 音频增强和切割

        if "切割" in input4_1 or "增强" in input4_1:
            audio_path = process_audio(audio_path, input4_1)

        video_process(cleaned_text, images_path, audio_path, width, height, input6, input7)
    output_file = merge_video(input7, input3, input_text_1, input_text_2, width=width)
    return output_file


# def apply_effects(file_path):
#     # 读取音频文件
#     audio = AudioSegment.from_file(file_path)
#
#     # 去除音频无声位置
#     trimmed_audio = strip_silence(audio, silence_thresh=-50, padding=100)
#
#     # 保存处理后的音频文件
#     output_file_path = file_path
#     trimmed_audio.export(output_file_path, format='mp3')
#
#     return output_file_path


# from resemble_enhance.enhancer.inference import denoise, enhance
# import gc

# if torch.cuda.is_available():
#     device = "cuda"
# else:
#     device = "cpu"
#

# def enhance_effects(file_path):
#     sample_rate, data = _fn(file_path, "Midpoint", 64, 0.1, 10, 1, False)
#     wavfile.write(file_path, sample_rate, data)
#     return file_path


def get_vicoe_lines():
    current_directory = os.path.dirname(__file__)
    file_path = os.path.join(current_directory, 'tts_josn.json')
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    lines = data["lines"]
    return lines


def get_lora_list():
    lora_json = os.path.join(constant.base_image_model_path, 'lora', 'model_index.json')
    with open(lora_json, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data["loras"]


def get_person_list():
    lora_json = os.path.join(constant.base_image_model_path, 'lora', 'person_lora.json')
    with open(lora_json, 'r', encoding='utf-8') as file:
        data = json.load(file)
    keys_array = list(data.keys())
    return keys_array


def get_person_list_for_key(text):
    lora_json = os.path.join(constant.base_image_model_path, 'lora', 'person_lora.json')
    with open(lora_json, 'r', encoding='utf-8') as file:
        data = json.load(file)
    # 遍历 keys_array 并检查 text 是否包含任何键
    for key in data.keys():
        if key in text:
            return data[key]
    # 如果没有匹配，返回 None 或其他默认值
    return ""


def get_style_person_list_for_key(text):
    lora_json = os.path.join(constant.base_image_model_path, 'lora', 'person_style_lora.json')
    with open(lora_json, 'r', encoding='utf-8') as file:
        data = json.load(file)
    # 遍历 keys_array 并检查 text 是否包含任何键
    for key in data.keys():
        if key in text:
            return '[person:{}]'.format(key), data[key][0], data[key][1]
    # 如果没有匹配，返回 None 或其他默认值
    return "", "", ""


def get_effect_list_for_key():
    effect_json = os.path.join(constant.base_image_model_path, 'lora', 'effect.json')
    with open(effect_json, 'r', encoding='utf-8') as file:
        data = json.load(file)
    keys_array = list(data.keys())
    random_key = random.choice(keys_array)
    return data[random_key]


def get_transition_effect_list_for_key():
    transition_effect = os.path.join(constant.base_image_model_path, 'lora', 'transition_effect.json')
    with open(transition_effect, 'r', encoding='utf-8') as file:
        data = json.load(file)
    keys_array = list(data.keys())
    random_key = random.choice(keys_array)
    return data[random_key]


def get_assets_lines(folder_path):
    current_directory = os.path.dirname(__file__)
    folder_path = os.path.join(current_directory, "asset", folder_path)
    file_names = []  # 存放文件名称的数组
    # file_paths = []  # 存放文件路径的数组
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_names.append(file)  # 将文件名称添加到数组
    #       file_paths.append(os.path.join(root, file))  # 将文件路径添加到数组
    return file_names


lines = get_vicoe_lines()
loras = get_lora_list()
person_names = get_person_list()

motion_effects = get_assets_lines("motion_effects")
sound_effects = get_assets_lines("sound_effects")
background_music = get_assets_lines("background_music")


def input_user(a, b, c, d):
    user_prompt = ""
    if a is not None:
        user_prompt += "{},".format(a)
    if b is not None:
        user_prompt += "{},".format(b)
    if d is not None:
        user_prompt += "{},".format(d)

    # 去掉最后一个逗号
    if user_prompt:
        user_prompt = user_prompt[:-1]
    if c is not None:
        user_prompt = c + user_prompt

    return user_prompt


effects_t = ''


def update_effects_text(intput):
    global effects_t
    effects_t = intput


def process_video_effects(effect):
    global effects_t
    effects_t = effects_t + "[video:{},1,2]".format(effect)
    return effects_t


def process_motion_effects(effect):
    global effects_t
    effects_t = effects_t + "[motion:{}]".format(remove_suffix(effect))
    return effects_t


def process_sound_effects(effect):
    global effects_t
    effects_t = effects_t + "[sound:{},0]".format(remove_suffix(effect))
    return effects_t


def process_text_person(u_prompt, person):
    person_name = "[person:{}]".format(person)
    return person_name + u_prompt


def process_text_lora(u_prompt, lora):
    person_name = "[lora:{},0.8]".format(lora)
    return person_name + u_prompt


def process_prompt(u_prompt, text_prompt):
    return u_prompt + ',' + text_prompt


def remove_suffix(filename):
    for suffix in [".gif", ".mp3", ".MP3", ".wav", ".mov"]:
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    return filename


tts_names = [line["name"] for line in lines]
loras_names = [line["name"] for line in loras]

global models_path
model_paths = []


def get_image_models(file_path):
    global models_path
    model_paths = []

    if not file_path:
        return model_paths
    if not os.path.exists(file_path):
        return model_paths
    if not os.path.isdir(file_path):
        return model_paths
    for entry in os.listdir(file_path):
        full_path = os.path.join(file_path, entry)

        if os.path.isdir(full_path):
            model_paths.append(full_path)
    models_path = model_paths
    return model_paths


def update_dropdown(file_path):
    constant.base_image_model_path = file_path
    dropdown_model_paths = get_image_models(file_path)
    return gr.update(choices=dropdown_model_paths, interactive=True)


def update_size(file_path):
    if "animation" in file_path:
        return gr.update(choices=constant.animationArray, interactive=True)
    elif "playground" in file_path:
        return gr.update(choices=constant.sizeArray, interactive=True)
    else:
        return gr.update(choices=constant.animationArray, interactive=True)


def get_size(file_path):
    if "animation" in file_path:
        return constant.animationArray
    elif "playground" in file_path:
        return constant.sizeArray
    else:
        return constant.animationArray


def load_models(path):
    if not path:
        return "无效路径"
    else:
        return load_pipeline(path)


with gr.Blocks() as demo:
    with gr.Tab("设置"):
        image_path_input = gr.Textbox(label="模型路径", lines=1, max_lines=1, value=constant.base_image_model_path)
        select_models_array = get_image_models(constant.base_image_model_path)
        select_models = gr.Dropdown(choices=select_models_array, label="选择模型")
        image_path_input.change(fn=update_dropdown, inputs=image_path_input, outputs=select_models)
        load_status = gr.Textbox(label="加载状态", lines=1, max_lines=1, interactive=False)
        select_models.change(fn=load_models, inputs=select_models, outputs=load_status)

        with gr.Row():
            ip_adapter = gr.Checkbox(label="启用IP")
            only_style = gr.Checkbox(label="仅风格")
            style_layout = gr.Checkbox(label="风格和布局")
            ip_adapter.change(fn=image_load_ip_adapter, inputs=[image_path_input, select_models], outputs=load_status)
            only_style.change(fn=image_update_scale_style, inputs=only_style)
            style_layout.change(fn=image_update_scale_style_layout, inputs=style_layout)

        with gr.Row():
            check_hidiffusion = gr.Checkbox(label="启用hidiffusion", info="注意部分尺寸不支持")
            check_hidiffusion.change(fn=update_hidiffusion, inputs=check_hidiffusion)

            check_freU = gr.Checkbox(label="启用FreeU", info="注意ground不支持")
            check_freU.change(fn=update_check_freU, inputs=[check_freU, select_models, check_hidiffusion])

            scheduler_setting = gr.Dropdown(choices=constant.sampler_list, label="调度器")
            scheduler_setting.change(fn=update_scheduler_setting, inputs=[scheduler_setting])

    with gr.Tab("文案修正"):
        with gr.Row():
            input_text = gr.Textbox(label="输入文案", lines=10, max_lines=10, interactive=True)
            out_text = gr.Textbox(label="输出文案", lines=10, max_lines=10, interactive=True)
        names = gr.Textbox(label="检测出来的名字", lines=10, max_lines=10, interactive=True)
        process_text_btn = gr.Button("整理文案")
        lora_name = gr.Radio(loras_names, label="lora", value=loras_names[0])
        gr.Radio(motion_effects, label="动效-motion", value=motion_effects)
        gr.Radio(sound_effects, label="音效-sound", value=sound_effects)
        video_effects = ["灵魂出窍", "曝光", "抖动", "震动", "微震动", "震动模糊", "发光", "曝光抖动"]
        gr.Radio(sound_effects, label="视效-video", value=video_effects)
        process_text_btn.click(fn=process_text_and_name, inputs=[input_text, lora_name], outputs=[out_text, names])

    with gr.Tab("单一成片"):
        input0 = gr.Textbox(label="输出路径", value=r"F:\小说推文")

        with gr.Row():
            input2 = gr.Textbox(label="小说内容", lines=2, max_lines=10, interactive=True)
            input4 = gr.Radio(constant.styleArray, label="风格", value=constant.styleArray[0])
        with gr.Row():
            input22 = gr.Textbox(label="提示词内容", lines=2, max_lines=10, interactive=True)
            input3 = gr.Radio(constant.prompt_style_array, label="提示词风格")
            prompt_button = gr.Button("提示词生成")

        # lac_prompt = gr.Textbox(label="分词结果", lines=2, max_lines=10, interactive=True)
        # input3 = gr.Radio(constant.prompt_style_array, label="系统提示词小说风格",
        #                   value=constant.prompt_style_array[0])

        input52 = gr.Radio(person_names, label="人物锁定")
        lora_text_name = gr.Radio(loras_names, label="lora")

        input5 = gr.Radio(constant.prompt_lighting_array, label="灯光")

        with gr.Row():
            user_prompt = gr.Textbox(label="用户自定义提示词", lines=3, max_lines=10, interactive=True,
                                     value="anime artwork")
            lac_prompt = gr.Textbox(label="正向提示词", lines=3, max_lines=10, interactive=True)
            input7 = gr.Textbox(label="反向提示词", lines=3, max_lines=10, interactive=True,
                                value='nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality, '
                                      'jpeg artifacts, low quality,  bad anatomy, skin spots, acnes, skin blemishes, '
                                      'facing away, looking away, tilted head, bad hands, missing fingers,'
                                      ' bad feet, poorly drawn hands, poorly drawn face, mutation, deformed, '
                                      'extra fingers, extra limbs, extra arms, extra legs, malformed limbs, fused fingers, '
                                      'too many fingers, long neck, cross-eyed, mutated hands, bad body, '
                                      'bad proportions, gross proportions, missing arms, missing legs, extra foot,'
                                      ' teethcroppe, blurry, cropped,watermark, unfinished, displeasing, oldest, early,'
                                      ' chromatic aberration, signature, extra digits, artistic error, username, scan, '
                                      'ghosting, [abstract]')

        # input22.change(fn=send_prompt_message, inputs=[input22, input3], outputs=[lac_prompt])
        # input3.change(fn=send_prompt_message, inputs=[input22, input3], outputs=[lac_prompt])
        prompt_button.click(fn=send_prompt_message, inputs=[input22, input3], outputs=[lac_prompt])

        input3.change(fn=process_prompt, inputs=[user_prompt, input3], outputs=[user_prompt])
        input4.change(fn=process_prompt, inputs=[user_prompt, input4], outputs=[user_prompt])
        input5.change(fn=process_prompt, inputs=[user_prompt, input5], outputs=[user_prompt])
        input52.change(fn=process_text_person, inputs=[user_prompt, input52], outputs=[user_prompt])
        lora_text_name.change(fn=process_text_lora, inputs=[user_prompt, lora_text_name], outputs=[user_prompt])

        size_array = get_size(constant.base_image_model_path)

        with gr.Row():
            with gr.Column():
                input_size = gr.Radio(choices=size_array, label="尺寸")
                input9 = gr.Slider(minimum=0, maximum=10000000000000, value=0, step=1, label="seed")
                input10 = gr.Slider(minimum=10, maximum=100, value=50, step=5, label="num_inference_steps")
                input11 = gr.Slider(minimum=1, maximum=20, value=3.0, step=1.0, label="guidance_scale")
                input_size.change(fn=change_enable, inputs=[input_size])
                select_models.change(fn=update_size, inputs=select_models, outputs=input_size)

            out_image = gr.Image(label="输出路径", width=512, height=512, type='filepath')
        image_btn = gr.Button("推理")

        image_btn.click(fn=text_to_image,
                        inputs=[user_prompt, lac_prompt, input7, input_size, input9, input10, input11, image_path_input,
                                input0],
                        outputs=out_image)

        with gr.Row():
            input11 = gr.Radio(tts_names, label="主播", value=tts_names[0])
            input12 = gr.CheckboxGroup(["切割", "增强"], label="音频增强")

        transform = gr.Radio(constant.transform_list, label="视频特效", value=constant.transform_list[0])

        effects_text = gr.Textbox(label="其他特效")
        effects_text.change(fn=update_effects_text, inputs=effects_text)

        video_effects = ["灵魂出窍", "曝光", "抖动", "震动", "微震动", "震动模糊", "发光", "曝光抖动"]
        input16 = gr.Radio(video_effects, label="视效-video", )

        with gr.Row():
            with gr.Column():
                input13 = gr.Radio(motion_effects, label="动效-motion", )
                input14 = gr.Radio(sound_effects, label="音效-sound", )

        input13.change(fn=process_motion_effects, inputs=[input13], outputs=[effects_text])
        input14.change(fn=process_sound_effects, inputs=[input14], outputs=[effects_text])
        input16.change(fn=process_video_effects, inputs=[input16], outputs=[effects_text])

        out_video = gr.Video(label="输出路径", width=1025, height=567)
        video_btn = gr.Button("视频生成")

        video_btn.click(fn=single_video_generate,
                        inputs=[input_size, input2, out_image, input11, input12, transform, input0, effects_text],
                        outputs=out_video)

        with gr.Row():
            input17 = gr.Radio(background_music, label="背景音乐", value=background_music[0])
            input18 = gr.Radio(constant.transform_style_list, label="转场动画", value=constant.transform_style_list[0])

        mer_video = gr.Video(label="输出路径", width=1025, height=567)
        merge_btn = gr.Button("合成生成")
        merge_btn.click(fn=merge_video_effects,
                        inputs=[input17, input0, input18], outputs=mer_video)

    with gr.Tab("一键推文"):
        input1 = gr.Textbox(label="输入文案", lines=10, max_lines=10, interactive=True)
        input2 = gr.Radio(constant.styleArray, label="风格", value=constant.styleArray[0])
        input8 = gr.Radio(constant.prompt_style_array, label="小说风格", value=constant.prompt_style_array[0])
        input3 = gr.Radio(background_music, label="背景音乐", value=background_music[0])
        with gr.Row():
            input4 = gr.Radio(tts_names, label="主播", value=tts_names[0])

            with gr.Column():
                input4_1 = gr.CheckboxGroup(["切割", "增强"], label="音频增强")
                input_1_1 = gr.Slider(minimum=0, maximum=100, value=45, step=1, label="音速")
                input_1_2 = gr.Slider(minimum=0, maximum=100, value=60, step=1, label="音量")
                input_1_3 = gr.Slider(minimum=0, maximum=100, value=5, step=1, label="赫兹")
                input_1_1.change(fn=constant.update_var_rate, inputs=[input_1_1])
                input_1_2.change(fn=constant.update_var_volume, inputs=[input_1_2])
                input_1_3.change(fn=constant.update_var_pitch, inputs=[input_1_3])

        # with gr.Row():
        # input4_1 = gr.CheckboxGroup(["切割", "增强"], label="音频增强")
        # with gr.Column():
        #     input_1_4 = gr.Slider(minimum=0, maximum=100000000000000, value=0, step=1, label="seed")
        #     input_1_5= gr.Slider(minimum=0, maximum=100000000000000, value=0, step=1, label="seed")
        #     input_1_6 =gr.Slider(minimum=0, maximum=100000000000000, value=0, step=1, label="seed")

        with gr.Row():
            with gr.Column():
                input_2_1 = gr.Slider(minimum=0, maximum=1000000000000, value=0, step=1, label="seed")
                input_2_2 = gr.Slider(minimum=10, maximum=100, value=50, step=5, label="num_inference_steps")
                input_2_3 = gr.Slider(minimum=1, maximum=20, value=3.0, step=1.0, label="guidance_scale")
                input_2_1.change(fn=constant.update_seed, inputs=[input_2_1])
                input_2_2.change(fn=constant.update_num_inference_steps, inputs=[input_2_2])
                input_2_3.change(fn=constant.update_guidance_scale, inputs=[input_2_3])
            input_size = gr.Radio(choices=size_array, label="尺寸")

        input6 = gr.Radio(constant.transform_list, label="视频特效", value=constant.transform_list[0])

        effects_input = gr.Checkbox(label="启用随机特效")

        person_input = gr.Checkbox(label="自动匹配人物提示词")

        style_input = gr.Checkbox(label="启用风格处理")

        input_text_1 = gr.Textbox(label="左上角标题")
        input_text_2 = gr.Textbox(label="右侧标题")

        input7 = gr.Textbox(label="输出路径", value="F:\小说推文")
        video_file_path = gr.Video(label="视频", width=512, height=512)
        video_btn = gr.Button("生成")
        video_btn.click(fn=video_generate,
                        inputs=[input1, input2, input3, input4, input_size, input6, input7, input4_1, input8,
                                effects_input, person_input, style_input, input_text_1, input_text_2],
                        outputs=video_file_path)

    with gr.Tab("同风格一键推文"):
        input1 = gr.Textbox(label="输入文案", lines=10, max_lines=10, interactive=True)
        input2 = gr.Radio(constant.styleArray, label="风格", value=constant.styleArray[0])
        input8 = gr.Radio(constant.prompt_style_array, label="小说风格", value=constant.prompt_style_array[0])
        input3 = gr.Radio(background_music, label="背景音乐", value=background_music[0])
        with gr.Row():
            input4 = gr.Radio(tts_names, label="主播", value=tts_names[0])

            with gr.Column():
                input4_1 = gr.CheckboxGroup(["切割", "增强"], label="音频增强")
                input_1_1 = gr.Slider(minimum=0, maximum=100, value=45, step=1, label="音速")
                input_1_2 = gr.Slider(minimum=0, maximum=100, value=60, step=1, label="音量")
                input_1_3 = gr.Slider(minimum=0, maximum=100, value=5, step=1, label="赫兹")
                input_1_1.change(fn=constant.update_var_rate, inputs=[input_1_1])
                input_1_2.change(fn=constant.update_var_volume, inputs=[input_1_2])
                input_1_3.change(fn=constant.update_var_pitch, inputs=[input_1_3])

        with gr.Row():
            with gr.Column():
                input_2_1 = gr.Slider(minimum=0, maximum=1000000000000, value=0, step=1, label="seed")
                input_2_2 = gr.Slider(minimum=10, maximum=100, value=50, step=5, label="num_inference_steps")
                input_2_3 = gr.Slider(minimum=1, maximum=20, value=3.0, step=1.0, label="guidance_scale")
                input_2_1.change(fn=constant.update_seed, inputs=[input_2_1])
                input_2_2.change(fn=constant.update_num_inference_steps, inputs=[input_2_2])
                input_2_3.change(fn=constant.update_guidance_scale, inputs=[input_2_3])
            input_size = gr.Radio(choices=size_array, label="尺寸")

        input6 = gr.Radio(constant.transform_list, label="视频特效", value=constant.transform_list[0])

        effects_input = gr.Checkbox(label="启用随机特效")

        person_input = gr.Checkbox(label="自动匹配人物提示词")

        style_input = gr.Checkbox(label="启动风格处理")

        input_text_1 = gr.Textbox(label="左上角标题")

        input_text_2 = gr.Textbox(label="右侧标题")

        input7 = gr.Textbox(label="输出路径", value="F:\小说推文")
        video_file_path = gr.Video(label="视频", width=512, height=512)
        video_btn = gr.Button("生成")
        video_btn.click(fn=video_generate,
                        inputs=[input1, input2, input3, input4, input_size, input6, input7, input4_1, input8,
                                effects_input, person_input, style_input, input_text_1, input_text_2],
                        outputs=video_file_path)

    select_models.change(fn=update_size, inputs=select_models, outputs=input_size)

    # with gr.Tab("声音调整测试"):
    #     a1 = gr.Audio(type="filepath", label="Input Audio")
    #     a2 = gr.Dropdown(choices=["Midpoint", "RK4", "Euler"], value="Midpoint",
    #                      label="CFM ODE Solver (Midpoint is recommended)")
    #     a3 = gr.Slider(minimum=1, maximum=128, value=64, step=1,
    #                    label="CFM Number of Function Evaluations (higher values in general yield better quality but may be slower)")
    #     a4 = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.01,
    #                    label="CFM Prior Temperature (higher values can improve quality but can reduce stability)")
    #     a5 = gr.Slider(minimum=1, maximum=40, value=10, step=1, label="Chunk seconds (more secods more VRAM usage)")
    #     a6 = gr.Slider(minimum=0, maximum=5, value=1, step=0.5, label="Chunk overlap")
    #     a7 = gr.Checkbox(value=False,
    #                      label="Denoise Before Enhancement (tick if your audio contains heavy background noise)")
    #     o2 = gr.Audio(label="Output Enhanced Audio")
    #     audio_btn = gr.Button("生成")
    #     audio_btn.click(fn=_fn, inputs=[a1, a2, a3, a4, a5, a6, a7], outputs=[o2])
    #
    #     # silence_thresh=-50, padding=100
    #     a10 = gr.Slider(minimum=0, maximum=5000, value=1000, step=100, label="静音长度")
    #     a8 = gr.Slider(minimum=-100, maximum=500, value=-50, step=5, label="多少分贝认定静音")
    #     a9 = gr.Slider(minimum=0, maximum=1000, value=100, step=5, label="保留的音频段时常")
    #     o3 = gr.Audio(label="去除无音")
    #     audio_btn_1 = gr.Button("去除无音")
    #     audio_btn_1.click(fn=apply_effects, inputs=[a1, a10, a8, a9], outputs=[o3])
    #
    #     # 声音增强.
    #     # def add_reverb_0(file_path, room_dim1=10, room_dim2=10, room_dim3=3, rt60_tgt=0.6):
    #     a11 = gr.Slider(minimum=0, maximum=100, value=10, step=1, label="room_dim1")
    #     a12 = gr.Slider(minimum=0, maximum=100, value=10, step=1, label="room_dim2")
    #     a13 = gr.Slider(minimum=0, maximum=100, value=3, step=1, label="room_dim3")
    #     a14 = gr.Slider(minimum=0, maximum=1, value=0.6, step=0.1, label="rt60_tgt")
    #     a15 = gr.Slider(minimum=0, maximum=5, value=1.5, step=0.1, label="microphone")
    #     o4 = gr.Audio(label="声音增强")
    #     audio_btn_2 = gr.Button("声音增强")
    #     audio_btn_2.click(fn=add_reverb_0, inputs=[a1, a11, a12, a13, a14, a15], outputs=[o4])

    with gr.Tab("图片调整与测试"):
        input1 = gr.Textbox(label="输入文案", lines=2, max_lines=2, interactive=True)
        with gr.Row():
            input2 = gr.Radio(constant.styleArray, label="风格", value=constant.styleArray[0])
            input9 = gr.Radio(constant.prompt_style_array, label="小说风格", value=constant.prompt_style_array[0])
            input3 = gr.Slider(minimum=0, maximum=4096, value=1344, step=1, label="w")
            input4 = gr.Slider(minimum=0, maximum=4096, value=768, step=1, label="h")
            input5 = gr.Slider(minimum=0, maximum=100000000000, value=0, step=1, label="seed")
            input6 = gr.Slider(minimum=10, maximum=100, value=28, step=1.0, label="num_inference_steps")
            input7 = gr.Slider(minimum=1, maximum=20, value=7.0, step=1.0, label="guidance_scale")
        input8 = gr.Textbox(label="输出路径", value="F:\sdxl_images")
        out_image = gr.Image(label="输出图像", type='filepath')
        image_test_btn = gr.Button("生成")
        image_test_btn.click(fn=generate_image_test, inputs=[input1, input2, input3, input4, input5, input6,
                                                             input7, input8, input9], outputs=[out_image])

        saturation_slider = gr.Slider(minimum=0.0, maximum=2.0, value=1.0, step=0.1, label="饱和度")
        brightness_slider = gr.Slider(minimum=0.0, maximum=2.0, value=1.0, step=0.1, label="亮度")

        process_saturation_brightness_btn = gr.Button("调整")
        process_saturation_brightness_btn.click(fn=process_image_style_for_image,
                                                inputs=[out_image, input8, saturation_slider,
                                                        brightness_slider],
                                                outputs=[out_image])

        with gr.Row():
            controlNer_canny = gr.Checkbox(label="启用controller_canny")
            load_status = gr.Textbox(label="加载状态", lines=1, max_lines=1, interactive=False)
            controlNer_canny.change(fn=load_model_to_controller, inputs=[select_models, image_path_input],
                                    outputs=load_status)
            hidiffusion = gr.Checkbox(label="启用hidiffusion")
            hidiffusion.change(fn=update_hidiffusion, inputs=[select_models])
            model_cpu_offload = gr.Checkbox(label="启用cpu_卸载")
            model_cpu_offload.change(fn=model_cpu_offload_fn, inputs=[model_cpu_offload])

        with gr.Row():
            width = gr.Slider(minimum=0, maximum=4096, value=3072, step=1, label="w")
            height = gr.Slider(minimum=0, maximum=4096, value=2048, step=1, label="h")
            num_inference_steps = gr.Slider(minimum=10, maximum=100, value=28, step=5, label="num_inference_steps")
            guidance_scale = gr.Slider(minimum=1, maximum=20, value=7.0, step=1.0, label="guidance_scale")
            controlnet_conditioning_scale = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.1,
                                                      label="controlnet_conditioning_scale")
        with gr.Row():
            prompt = gr.Textbox(label="正向提示词", lines=2, max_lines=2, interactive=True)
            negative_prompt = gr.Textbox(label="反向提示词", lines=2, max_lines=2, interactive=True)

        with gr.Row():
            ori_image_path = gr.Textbox(label="输出路径", value=out_image)
            controlNet_output_image = gr.Image(label="输出图像", type='filepath')
        controlNet_button = gr.Button("生成")
        controlNet_button.click(fn=canny_to_image, inputs=[ori_image_path, controlnet_conditioning_scale, hidiffusion,
                                                           prompt, negative_prompt, width, height, num_inference_steps,
                                                           guidance_scale, input8], outputs=controlNet_output_image)

    # with gr.Tab("根据提示词批量生成图片"):
    #     input1 = gr.Textbox(label="输入文案", lines=10, max_lines=10, interactive=True)
    #     input2 = gr.Radio(constant.styleArray, label="风格", value=constant.styleArray[0])
    #     input3 = gr.Radio(constant.prompt_style_array, label="小说风格", value=constant.prompt_style_array[0])
    #     input4 = gr.Radio(constant.sizeArray, label="尺寸", value=constant.sizeArray[0])
    #     input8 = gr.Slider(minimum=0, maximum=1000000000, value=0, step=1, label="seed")
    #     input6 = gr.Slider(minimum=10, maximum=100, value=50, step=5, label="num_inference_steps")
    #     input7 = gr.Slider(minimum=1, maximum=20, value=3.0, step=1.0, label="guidance_scale")
    #     input5 = gr.Textbox(label="输出路径")
    #
    #     image_btn = gr.Button("开始")
    #     image_btn.click(fn=batch_image_generate,
    #                     inputs=[input1, input2, input3, input4, input5, input6, input7, input8])

    # with gr.Tab("SVD批量抽卡"):
    #         input1 = gr.Textbox(label="输入图片数据集", value="F:\sdxl_images")
    #         input2 = gr.Slider(minimum=0, maximum=1024, value=1024, step=1, label="w")
    #         input3 = gr.Slider(minimum=0, maximum=576, value=576, step=1, label="h")
    #         image_btn = gr.Button("开始")
    #         image_btn.click(fn=fcm_generate_video_form_paht, inputs=[input1, input2, input3])

    with gr.Tab("同种风格生成"):
        style_output = gr.Textbox(label="输出路径", value="F:\同种风格生成")
        with gr.Row():
            imput_image = gr.Image(label="参考图像", sources="upload", type='pil')
            output_style_image = gr.Image(label="输出图像", type='filepath')
        with gr.Row():
            input_style_prompt = gr.Textbox(label="正向提示词", lines=2, max_lines=2, interactive=True)
            negative_prompt_style = gr.Textbox(label="反向提示词", lines=2, max_lines=2, interactive=True,
                                               value='nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality,extra fingers, ' \
                                                     'jpeg artifacts, low quality,  bad anatomy, skin spots, acnes, skin blemishes, ' \
                                                     'facing away, looking away, tilted head, bad hands, missing fingers,' \
                                                     ' bad feet, poorly drawn hands, poorly drawn face, mutation, deformed, ' \
                                                     'extra fingers, extra limbs, extra arms, extra legs, malformed limbs, fused fingers, ' \
                                                     'too many fingers, long neck, cross-eyed, mutated hands, bad body, ' \
                                                     'bad proportions, gross proportions, missing arms, missing legs, extra foot,' \
                                                     ' teethcroppe, blurry, cropped,watermark, unfinished, displeasing, oldest, early,' \
                                                     ' chromatic aberration, signature, extra digits, artistic error, username, scan, ' \
                                                     'ghosting, [abstract]')

        w = gr.Slider(minimum=0, maximum=4096, value=1344, step=1, label="w")
        h = gr.Slider(minimum=0, maximum=4096, value=768, step=1, label="h")
        seed = gr.Slider(minimum=0, maximum=100000000000, value=0, step=1, label="seed")
        num_inference_steps = gr.Slider(minimum=10, maximum=100, value=50, step=5, label="num_inference_steps")
        guidance_scale = gr.Slider(minimum=1, maximum=20, value=3.0, step=1.0, label="guidance_scale")

        image_test_btn = gr.Button("生成")
        image_test_btn.click(fn=generator_style_for_image,
                             inputs=[imput_image, input_style_prompt, negative_prompt_style, w, h, seed,
                                     num_inference_steps,
                                     guidance_scale, style_output], outputs=[output_style_image])

        saturation_slider = gr.Slider(minimum=0.0, maximum=2.0, value=1.0, step=0.1, label="饱和度")
        brightness_slider = gr.Slider(minimum=0.0, maximum=2.0, value=1.0, step=0.1, label="亮度")

        process_saturation_brightness_btn = gr.Button("调整")
        process_saturation_brightness_btn.click(fn=process_image_style_for_image,
                                                inputs=[output_style_image, style_output, saturation_slider,
                                                        brightness_slider],
                                                outputs=[output_style_image])

        # saturation_slider.change(fn=process_image_style_for_image,
        #                          inputs=[output_style_image, style_output, saturation_slider, brightness_slider],
        #                          outputs=[output_style_image])
        # brightness_slider.change(fn=process_image_style_for_image,
        #                          inputs=[output_style_image, style_output, saturation_slider, brightness_slider],
        #                          outputs=[output_style_image])

# with gr.Tab("数字人视频合成"):
#     in1 = gr.Textbox(label="输入文案", lines=10, max_lines=10, interactive=True)
#     in2 = gr.Textbox(label="背景音乐", interactive=True)
#     with gr.Row():
#         in3 = gr.Video(label="视频地址", width=512, height=512, sources="upload")
#         number_video_file_path = gr.Video(label="生成视频", width=512, height=512)
#     video_btn = gr.Button("生成")
#     video_btn.click(fn=number_video_generate, inputs=[input1, input2, input3, input4, input5, input6, input7],
#                     outputs=number_video_file_path)

if __name__ == "__main__":
    demo.queue().launch(share=False)
