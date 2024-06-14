import os
import time

from PIL import Image,ImageEnhance



def adjust_saturation(image, saturation_level):
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(saturation_level)

def adjust_brightness(image, brightness_level):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(brightness_level)


def process_image_style_for_image(intput_image_path,
                              style_output, saturation_level, brightness_level):
    try:

        image = Image.open(intput_image_path)
        # 在此处添加你原有的图像生成代码
        # 生成后的图像暂时保存在 `generated_image`
        generated_image = image

        # 调整饱和度
        generated_image = adjust_saturation(generated_image, saturation_level)

        # 调整亮度
        generated_image = adjust_brightness(generated_image, brightness_level)

        timestamp = int(time.time())
        save_restore_path = os.path.join(style_output, 'images', f'{timestamp}.jpg')
        os.makedirs(os.path.dirname(save_restore_path), exist_ok=True)
        # 保存调整后的图像
        generated_image.save(save_restore_path)
        return save_restore_path
    except Exception as e:
        return str(e)