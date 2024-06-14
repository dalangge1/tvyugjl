import gc
import re

import google.generativeai as genai

import os


os.environ['http_proxy'] = 'http://127.0.0.1:33210'
os.environ['https_proxy'] = 'http://127.0.0.1:33210'
os.environ['all_proxy'] = 'socks5://127.0.0.1:33211'


generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]


def send_prompt_message(prompt, prompt_style):
    genai.configure(api_key='AIzaSyBLuNGY_fR25SU8-BjQZZTaqIJadNJ1x54')
    model = genai.GenerativeModel('gemini-pro', safety_settings=safety_settings)
    chat = model.start_chat()
    input_str = "根据你的理解和想象,将文案\"{}\"想象成一副符合\"{}\"风格制作精良且详细的画面，可以是对抽象事物的想象，通过Stable Diffusion XL的提示词的方式描述出来," \
                "提示词必须按照Danbooru-style tags模式返回，必須，進行分割词语,提示词必须超过10个单词, 不能有任何解释，不能有任何英文，不能有任何特殊符号，直接给出提示词。".format(
        prompt, prompt_style)
    response = chat.send_message(input_str)
    prompt = response.text
    colon_index = prompt.find(":")
    # 移除冒号及其前面的部分
    if colon_index != -1:
        cleaned_prompt = prompt[colon_index + 1:].strip()
    else:
        cleaned_prompt = prompt

    # 释放 GPU 资源
    del chat
    del model
    gc.collect()

    return cleaned_prompt


def remove_brackets_and_contents(text):
    return re.sub(r"\[.*?\]", "", text)

# Danbooru-style tags
def send_en_prompt_danbooru_style(prompt, prompt_style):
    prompt = remove_brackets_and_contents(prompt)
    genai.configure(api_key='AIzaSyBLuNGY_fR25SU8-BjQZZTaqIJadNJ1x54')
    model = genai.GenerativeModel('gemini-pro', safety_settings=safety_settings)
    chat = model.start_chat()
    input_str = "根据你的理解和想象,将文案\"{}\"想象成一副符合\"{}\"风格的详细画面，可以是对抽象词语的想象，然后通过Stable Diffusion 的提示词的方式描述出来," \
                "不能有任何解释，不能有任何中文，不能有任何特殊符号，直接返回提示词。".format(
        prompt, prompt_style)
    response = chat.send_message(input_str)
    prompt = response.text
    colon_index = prompt.find(":")
    # 移除冒号及其前面的部分
    if colon_index != -1:
        cleaned_prompt = prompt[colon_index + 1:].strip()
    else:
        cleaned_prompt = prompt

    # 释放 GPU 资源
    del chat
    del model
    gc.collect()
    print(cleaned_prompt)
    return cleaned_prompt


def send_en_prompt_message(prompt, prompt_style):
    genai.configure(api_key='AIzaSyBLuNGY_fR25SU8-BjQZZTaqIJadNJ1x54')
    model = genai.GenerativeModel('gemini-pro', safety_settings=safety_settings)
    chat = model.start_chat()
    input_str = "根据你的理解和想象,将文案\"{}\"想象成一副符合\"{}\"风格的简单画面，可以是对抽象词语的想象，然后通过Stable Diffusion XL的提示词的方式描述出来," \
                "不能有任何解释，不能有任何中文，不能有任何特殊符号，直接给简单的提示词。".format(
        prompt, prompt_style)
    response = chat.send_message(input_str)
    prompt = response.text
    colon_index = prompt.find(":")
    # 移除冒号及其前面的部分
    if colon_index != -1:
        cleaned_prompt = prompt[colon_index + 1:].strip()
    else:
        cleaned_prompt = prompt

    # 释放 GPU 资源
    del chat
    del model
    gc.collect()
    print(cleaned_prompt)
    return cleaned_prompt


def send_translate_message(a, b):
    genai.configure(api_key='AIzaSyBLuNGY_fR25SU8-BjQZZTaqIJadNJ1x54')
    model = genai.GenerativeModel('gemini-pro', safety_settings=safety_settings)
    chat = model.start_chat()

    #prompt = a + "蓝天，白云" +"," + b
    prompt = a + b
    input_str = "将内容: {} 翻译成英文。不能有任何解释，不能有任何中文，不能有任何特殊符号，直接进行翻译。".format(prompt)
    response = chat.send_message(input_str)
    prompt = response.text
    colon_index = prompt.find(":")
    # 移除冒号及其前面的部分
    if colon_index != -1:
        cleaned_prompt = prompt[colon_index + 1:].strip()
    else:
        cleaned_prompt = prompt

        # 释放 GPU 资源
    del chat
    del model
    gc.collect()
    return cleaned_prompt


def send_message(prompt, prompt_style):
    prompt_text = ''
    if prompt_style == "世情":
        # prompt_text = "描写现实生活、社会风情和人情世故的画风"
        prompt_text = "A style of painting that describes real life, social customs and human feelings,"
    if prompt_style == "仙侠":
        # prompt_text = "仙人、神话传说、修真者等奇幻元素的画风"
        prompt_text = "Painting style with fantasy elements such as immortals, myths and legends, cultivators, etc,"
    if prompt_style == "古言":
        # prompt_text = "古代为背景，以历史人物、古代社会风情的画风"
        prompt_text = "Ancient times as the background, the painting style of ancient social customs,"
    if prompt_style == "悬疑":
        prompt_text = "Tense, exciting and suspenseful painting style,"
    if prompt_style == "现言":
        prompt_text = "Taking contemporary society as the background to describe modern life,"

    # 中国风格 #新海诚 风格  #
    input_str = "我需要将一段文案\"{}想象成一副制作精良且详细的画面，通过Stable Diffusion XL的提示词描述出来，" \
                "填充细节,并且回答限制在20个英文单词内。".format(prompt)

    # input_str = "将一段小说文案\"{}翻译成能够描述出一幅画面的英文句子，如果是".format(prompt)

    genai.configure(api_key='AIzaSyBLuNGY_fR25SU8-BjQZZTaqIJadNJ1x54')
    model = genai.GenerativeModel('gemini-pro', safety_settings=safety_settings)
    chat = model.start_chat()

    input_str = input_str
    response = chat.send_message(input_str)
    prompt = response.text
    colon_index = prompt.find(":")
    # 移除冒号及其前面的部分
    if colon_index != -1:
        cleaned_prompt = prompt[colon_index + 1:].strip()
    else:
        cleaned_prompt = prompt

    return cleaned_prompt
