import os
import re
import time
import cv2
from moviepy.video.VideoClip import ImageClip
from PIL import Image

import matplotlib.pyplot as plt

from scipy.io import wavfile


def cv_img_rgb(path):
    img = plt.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


# 变量

var_num_inference_steps = 50
var_seed = 0
var_guidance_scale = 16

var_rate = 45
var_volume = 65
var_pitch = 5


def update_var_rate(i):
    global var_rate
    var_rate = var_rate


def update_var_volume(i):
    global var_volume
    var_volume = var_volume


def update_var_pitch(i):
    global var_pitch
    var_pitch = var_pitch


def update_seed(i):
    global var_seed
    var_seed = i


def update_num_inference_steps(i):
    global var_num_inference_steps
    var_num_inference_steps = i


def update_guidance_scale(i):
    global var_guidance_scale
    var_guidance_scale = i


size_mapping = {
    "16:9--1920x1080": (1920, 1080),
    "16:9--1280x720": (1280, 720),
    "9:16--1080x1920": (1080, 1920),
    "9:16--720x1280": (720, 1280),
    "1:1--1024x1024": (1024, 1024),
    "1:1--2048x2048": (2048, 2048),
    "16:9--3072x2048": (3072, 2048),
    "2:1--4096x2048": (4096, 2048),
    "1:1--512x512": (512, 512),
    "19:13--1216x832": (1216, 832),
    "13:19--832x1216": (832, 1216),
    "1024 x 1024": (1024, 1024),
    "1152 x 896": (1152, 896),
    "896 x 1152": (896, 1152),
    "1216 x 832": (1216, 832),
    "832 x 1216": (832, 1216),
    "1344 x 768": (1344, 768),
    "768 x 1344": (768, 1344),
    "1536 x 640": (1536, 640),
    "640 x 1536": (640, 1536),
    "1920 x 1080": (1920, 1080),
    "2048x2048": (2048, 2048),
    "3072x2048": (3072, 2048),
    "4096x2048": (4096, 2048),
    "2560x1600": (2560, 1600),
    "1600x2560": (1600, 2560),
    "1920x1200": (1920, 1200),
    "1200x1920": (1200, 1920),
    "1152x2048": (1152, 2048),
    "1536x2048": (1536, 2048),
    "1184x2560": (1184, 2560),
    "2560x1184": (2560, 1184),
    "2048x1280": (2048, 1280),
    "1280x2048": (1280, 2048),

}
sizeArray = ["16:9--1920x1080", "16:9--1280x720", "9:16--1080x1920", "9:16--720x1280", "1:1--1024x1024",
             "1:1--2048x2048", "16:9--3072x2048", "2:1--4096x2048", "1:1--512x512", "19:13--1216x832",
             "13:19--832x1216"]

# animation_mapping = {
#     "1024 x 1024": (1024, 1024),
#     "1152 x 896": (1152, 896),
#     "896 x 1152": (896, 1152),
#     "1216 x 832": (1216, 832),
#     "832 x 1216": (832, 1216),
#     "1344 x 768": (1344, 768),
#     "768 x 1344": (768, 1344),
#     "1536 x 640": (1536, 640),
#     "640 x 1536": (640, 1536),
#     "2048x2048": (2048, 2048),
#     "3072x2048": (3072, 2048),
#     "4096x2048": (4096, 2048),
# }

animationArray = [
    "1024 x 1024",
    "1152 x 896",
    "896 x 1152",
    "1216 x 832",
    "832 x 1216",
    "1344 x 768",
    "768 x 1344",
    "1536 x 640",
    "640 x 1536",
    "1920 x 1080",
    "2048x2048",
    "3072x2048",
    "4096x2048",
    "2560x1600",
    "1600x2560",
    "1920x1200",
    "1200x1920",
    "1152x2048",
    "1536x2048",
    "1184x2560",
    "2560x1184",
    "2048x1280",
    "1280x2048",
]

sampler_list = [
    "默认",
    "LCM",
    "DPM++ 2M Karras",
    "DPM++ SDE Karras",
    "DPM++ 2M SDE Karras",
    "Euler",
    "Euler a",
    "DDIM",
]

# relase
# base_path = os.path.expanduser("~")
# base_model_path = os.path.join(base_path, "novel_models")

# debug
base_path = 'F:\\novel_models'
base_model_path = os.path.join(base_path)

base_image_model_path = os.path.join(base_path, 'images_models')

# 其他模型
# base_model_path = os.path.join(novel_models, 'novel_models', 'images_models', 'animation3-1')


transform_list = ["默认", "随机", "放大", "左", "左放大", "右放大", "右", "上", "下"]  # , "右上", "左上"

transform_style_list = ["默认", "随机", "上", "下", "右", "左"]

prompt_style_array = ["世情", '仙侠', '古风', '武侠', '古言', '悬疑', '现言']

person_style_array = ["二十岁，帅气男性,中国武侠人物造型",
                      '一个三十岁女性,体型性感妩媚,脸色嘴角有颗痣,衣着暴漏，性感,中国武侠人物造型',
                      '一个漂亮的三十岁女性,看着慈祥,衣着暴漏，性感,中国武侠人物造型',
                      '一个6岁可爱小男孩,中国武侠人物造型', '一个四十岁中年男性,有胡子,中国武侠人物造型',
                      '一个四十岁中年男性,没有胡子,中国武侠人物造型']

prompt_lighting_array = ["柔光", '工作室照明', '环境照明', '环形照明', '阳光照片', '电影照明']

# 照和光线：studio lighting（工作室照明）、soft lighting（柔光）、ambient lighting（环境照明）、ring lighting（环形照明）、
# sun lighting（阳光照片）、cinematic lighting（电影照明）


styleArray = ["anime artwork", 'professional 3d model', 'concept art', 'ethereal fantasy concept art',
              'cinematic photo', 'claymation style']

voiceMap = {
    "云溪": "zh-CN-YunxiNeural",
    "云健": "zh-CN-YunjianNeural",
    "云阳": "zh-CN-YunyangNeural",
    "云夏": "zh-CN-YunxiaNeural",
    "云TW": "zh-TW-YunJheNeural",
    "晓晓": "zh-CN-XiaoxiaoNeural",
    "晓依": "zh-CN-XiaoyiNeural",
    "云霞": "zh-CN-YunxiaNeural",
    "陈TW": "zh-TW-HsiaoChenNeural",
}

voiceArray = ["云溪",
              "云健",
              "云阳",
              "云夏",
              "云TW",
              "晓晓",
              "晓依",
              "云霞",
              "陈TW",
              ]


def font_size(width=None):
    if width > 4096 or width == 3072:
        return 105
    if width > 3072 or width == 3072:
        return 90
    if width > 1920:
        return 90
    if width > 1000:
        return 70
    if width == 1920 or width > 2080:
        return 70
    if width > 768:
        return 46
    if width == 720:
        return 36
    return 36


def subtitle_font_size(width=None):
    if width > 4096 or width == 3072:
        return 100
    if width > 3072 or width == 3072:
        return 76
    if width > 1920:
        return 76
    if width == 1920 or width > 1920:
        return 70
    if width > 1000:
        return 70
    if width > 768:
        return 36
    if width == 720:
        return 24
    return 24


def resize_image(image_path, w, h, crop_fit=False):
    if crop_fit == True:
        new_image_path = crop_to_fit_screen(image_path, w, h)
    else:
        new_image_path = crop_image(image_path, w, h)

    image_clip = ImageClip(new_image_path)
    os.remove(new_image_path)
    return image_clip


def crop_image(image_path, w, h):
    # file_path_encoded = image_path.encode('utf-8')
    # original_image = cv2.imread(file_path_encoded.decode('utf-8'))
    original_image = cv_img_rgb(image_path)
    w, h = int(w), int(h)
    resized_image = cv2.resize(original_image, (w, h))
    new_image_path = image_rename()
    cv2.imwrite(new_image_path, resized_image)
    return new_image_path


def crop_to_fit_screen(image_path, w, h):
    # 打开图像
    img = Image.open(image_path)
    img_width, img_height = img.size

    screen_ratio = w / h
    image_ratio = img_width / img_height

    if screen_ratio > image_ratio:
        new_height = int(img_width / screen_ratio)
        top_margin = (img_height - new_height) // 2
        bottom_margin = img_height - (top_margin + new_height)
        img = img.crop((0, top_margin, img_width, img_height - bottom_margin))
    else:
        new_width = int(img_height * screen_ratio)
        left_margin = (img_width - new_width) // 2
        right_margin = img_width - (left_margin + new_width)
        img = img.crop((left_margin, 0, img_width - right_margin, img_height))
    cropped_image = img.convert("RGB")
    new_image_path = image_rename()
    cropped_image.save(new_image_path)
    return new_image_path


def image_rename():
    # 获取当前路径
    project_path = os.getcwd()
    timestamp = int(time.time())
    tmep_image_path = os.path.join(project_path, 'tmep', f'{timestamp}.jpg')
    os.makedirs(os.path.dirname(tmep_image_path), exist_ok=True)
    return tmep_image_path


def replace_text(text):
    special_symbols = '{}　《》“”‘’"\' \n……—— 「」——......   '
    for symbol in special_symbols:
        text = text.replace(symbol, '')
    return text


def save_numpy_to_wav(audio_data, sampling_rate, output_path):
    # 保存numpy数组为WAV文件
    wavfile.write(output_path, sampling_rate, audio_data)

    return output_path


def cut_prompt(prompt, effect):
    # 匹配[]里的内容
    matches = re.findall(r'\[(.*?)\]', prompt)
    # 将匹配到的内容放入数组
    arguments = matches
    # 提取非方括号部分作为a2
    prompt = re.sub(r'\[.*?\]', '', prompt)
    print(arguments, prompt)

    effects_array = []

    if arguments is not None and len(arguments) > 0:
        for item in arguments:
            if effect in item:
                print("字符串", item, "包含", effect)
                effects_array.append(item)
            else:
                result = ''.join('[' + item + ']')
                prompt = result + prompt
        # 数组不为 None 并且不为空
        print("数组不为 None 并且不为空")
    else:
        # 数组为 None 或者为空
        print("数组为 None 或者为空")

    # if len(arguments) > 0 and len(effects_array) == 0:
    #     formatted_arguments = ['[' + arg + ']' for arg in arguments]
    #     result = ''.join(formatted_arguments)
    #     return effects_array, result + prompt

    return effects_array, prompt


def llama3_prompt(style=""):
    return """
    # Stable Diffusion Prompt 助理
    
    你来充当一位有艺术气息的Stable Diffusion Prompt XL 助理 , 你的回答必須是英文。
    
    ## 任务
    
    我用自然语言告诉你要生成的Prompt的主题，你的任务是根据我提供的文案想象一幅贴切\"{}的完整的画面，然后转化成一份详细的、高质量的Prompt，让Stable Diffusion可以生成高质量的图像。
   
    ## Prompt 概念
    - Prompt 用来描述图像，由普通常见的单词构成，使用英文半角","做为分隔符。
    - 以","分隔的每个单词或词组称为 tag。所以Prompt由","分隔的tag组成的。
   
    ## Prompt 格式要求
    下面我将说明 Prompt 的生成步骤，这里的 Prompt 可用于描述人物、风景、物体或抽象数字艺术图画。你可以根据需要添加合理的、但不少于5处的画面细节。
    
    ### 1. Prompt 要求
    - 你输出的 Stable Diffusion prompt 以“Prompt:”开头。
    - Prompt 包含画面主体、附加细节、但你输出的 Prompt 不能分段，例如类似"medium:"这样的分段描述是不需要的，也不能包含":"和"."。
    - Prompt 中的画面主体：不简短的英文描述画面主体, 如 A girl in a garden，主体细节概括（主体可以是人、事、物、景）画面核心内容。这部分根据我每次给你的主题来生成。你可以添加更多主题相关的合理的细节。
    - Prompt 中的对于人物主题，你必须描述人物的眼睛、鼻子、嘴唇，例如'beautiful detailed eyes,beautiful detailed lips,extremely detailed eyes and face,longeyelashes'，以免Stable Diffusion随机生成变形的面部五官，这点非常重要。你还可以描述人物的外表、情绪、衣服、姿势、视角、动作、背景等。人物属性中，1girl表示一个女孩，2girls表示两个女孩。
    - Prompt 中的附加细节：画面场景细节，或人物细节，描述画面细节内容，让图像看起来更充实和合理。这部分是可选的，要注意画面的整体和谐，不能与主题冲突。
    
    
    ### 2. 限制：
    - tag 内容用英语单词或短语来描述，并不局限于我给你的单词。注意只能包含关键词或词组。
    - 注意不要输出句子，不要有任何解释。
    - tag数量限制40个以内，单词数量限制在60个以内。
    - tag不要带引号("")。
    - 使用英文半角","做分隔符。
    - tag 按重要性从高到低的顺序排列。
    - 你的回答必須是英文。""".format(style)
