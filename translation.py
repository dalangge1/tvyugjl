import os

from transformers import MarianMTModel, MarianTokenizer

from constant import base_model_path

#novel_models = os.path.expanduser("~")


def translation_prompt(text):

    model_path = os.path.join(base_model_path, 'translaion_models')
    print("用户主目录存放模型位置:", model_path)
    tokenizer = MarianTokenizer.from_pretrained(pretrained_model_name_or_path=model_path, cache_dir=model_path,
                                                local_files_only=True)
    model = MarianMTModel.from_pretrained(pretrained_model_name_or_path=model_path, cache_dir=model_path,
                                          local_files_only=True)

    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    res = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    print(res)
    return res
