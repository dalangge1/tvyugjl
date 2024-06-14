import os
import re

import constant

import bitsandbytes
import flash_attn

import torch
from transformers import AutoTokenizer, LlamaForCausalLM

#from participle import participe_def
from translation import translation_prompt

model_path = os.path.join(constant.base_model_path, 'llama3')

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model = LlamaForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_8bit=False,
    load_in_4bit=True,
    use_flash_attention_2=True
    #attn_implementation="flash_attention_2"
)


def replace_strings(text):
    # 需要替换的字符串列表
    strings_to_replace = [
        '\n', '-', '1.', '2.', '3.', '4.', '5.', 'Main Elements', 'Picture:', 'Main Elements:', 'Describe:',
    ]

    # 替换为的字符串
    replacement = ''

    # 逐个替换字符串
    for string in strings_to_replace:
        text = text.replace(string, replacement)

    return text


def llama3_send_message_novel_text(text):
    prompts = [
        """<|im_start|>system
    \需要你仔细阅读我输入的小说段落，并且按照第一人称进行概要。<|im_end|>
    <|im_start|>user
    \"{}<|im_end|>
    <|im_start|>assistant""".format(text),
    ]

    print(prompts)
    for chat in prompts:
        print(chat)
        input_ids = tokenizer(chat, return_tensors="pt").input_ids.to("cuda")
        generated_ids = model.generate(input_ids, max_new_tokens=750, temperature=0.8, repetition_penalty=1.1,
                                       do_sample=True, eos_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(generated_ids[0][input_ids.shape[-1]:], skip_special_tokens=True,
                                    clean_up_tokenization_space=True)
        print(f"Response: {response}")
    # prompt = response.replace("Prompt:", ' ')
    # prompt = prompt.replace("\n", ' ')
    # prompt = prompt.replace("-", ' ')
    # prompt = prompt.replace("_", ' ')
    # prompt = prompt.replace("*", '')
    # prompt = prompt.replace("#", '')
    # cleaned_text = re.sub(r'[\u4e00-\u9fa5]', '', prompt)
    return response




def llama3_send_message_all(user_prompt,lac_prompt):

    prompt = user_prompt + ',' + lac_prompt
    prompts = [
        """<|im_start|>system
    \将我输入的中文内容翻译成英文。不能有任何解释，不能有任何中文，不能有任何特使符号，直接进行翻译。
    <|im_end|>
    <|im_start|>user
    \"{}<|im_end|>
    <|im_start|>assistant""".format(prompt),
    ]

    print(prompts)
    for chat in prompts:
        print(chat)
        input_ids = tokenizer(chat, return_tensors="pt").input_ids.to("cuda")
        generated_ids = model.generate(input_ids, max_new_tokens=750, temperature=0.8, repetition_penalty=1.1, do_sample=True, eos_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(generated_ids[0][input_ids.shape[-1]:], skip_special_tokens=True, clean_up_tokenization_space=True)
        print(f"Response: {response}")
    prompt = response.replace("Prompt:", ' ')
    prompt = prompt.replace("\n", ' ')
    prompt = prompt.replace("-", ' ')
    prompt = prompt.replace("_", ' ')
    prompt = prompt.replace("*", '')
    prompt = prompt.replace("#", '')

    cleaned_text = re.sub(r'[\u4e00-\u9fa5]', '', prompt)
    return cleaned_text
    #return en_prompt, en_prompt




def llama3_send_message(style, text):
    torch.cuda.empty_cache()
    # prompts = [
    #     """<|im_start|>system
    # 你来充当一位有艺术气息的Stable Diffusion XL prompt 助理,
    # 你的任务是根据我提供的文案想象一幅贴切\"{}的完整的画面,然后转化成一份精简的prompt,
    # 让Stable Diffusion可以生成高质量的图像.<|im_end|>
    # <|im_start|>user
    # 输入的文案是:“\"{}”请用一句话描述出来这个符合这个场景的画面.<|im_end|>
    # <|im_start|>assistant""".format(style, text),
    #     ]
    prompts = [
        """<|im_start|>system
    Your task is to use my prompts to imagine a picture that matches
    \"{} if the prompt content is small, please use your imagination to create,
      , use concise words to describe the picture and main elements of your imagination,
      And the output format is returned directly in sentence format, no more than 50 English words.
    <|im_end|>
    <|im_start|>user
    my prompts:\"{}.<|im_end|>
    <|im_start|>assistant""".format(style, text),
    ]
   # print(prompts)
    for chat in prompts:
        print(chat)
        input_ids = tokenizer(chat, return_tensors="pt").input_ids.to("cuda")
        generated_ids = model.generate(input_ids, max_new_tokens=750, temperature=0.8, repetition_penalty=1.1, do_sample=True, eos_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(generated_ids[0][input_ids.shape[-1]:], skip_special_tokens=True, clean_up_tokenization_space=True)
        print(f"Response: {response}")

    return replace_strings(response)
    #print(truncate_sentence(response))
# Depicting modern life in the context of contemporary society   当代社会为背景的现代生活
#send_message(style="Modern life in line with the background of modern society", text="魅尸比我们想象的聪明，她发现了端倪，一针见血。")

