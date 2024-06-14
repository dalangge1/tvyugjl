from functools import partial
from itertools import cycle
from random import random, randint

import numpy as np
import cv2
from PIL import Image
from moviepy.video.VideoClip import ImageClip

from constant import resize_image

# 决定移动大小
spacing = 35

options = ["放大", "上", "下", "上", "下", "上", "下", "上", "下", "上", "下", "上", "下", "上", "下", "上", "下", "左", "右", "左放大", "右放大"]
effects_cycle = cycle(options)
current_effect_index = 0


def get_next_effect():
    global current_effect_index
    effect = options[current_effect_index]
    current_effect_index = (current_effect_index + 1) % len(options)
    return effect


def precess_image_clip(image_path, duration, effects, width, height, filters=None):
    video_filters = filters
    video_effects = effects

    # ["video: 灵魂出窍, 2, 4","video: 灵魂出窍, 2, 4"]

    # if "video_effects" in json and json["video_effects"] not in ["", None]:
    #     video_effects = json["video_effects"]
    #
    # if "video_filters" in json and json["video_filters"] not in ["", None]:
    #     video_filters = json["video_filters"]
    image_clip = ImageClip(image_path).set_duration(duration)

    if video_effects == '默认':
        return image_clip

    if video_effects == '随机':
        # ptions = ["放大", "左", "左放大", "右放大", "右", "上", "下"] #, "右上", "左上"
        # seed = randint(0, len(options)-1)
        # video_effects = options[seed]
        # video_effects_cycle = cycle(options)
        video_effects = get_next_effect()

    if video_effects == "放大":
        image_clip = resize_image(image_path, width, height).set_duration(duration)
    if video_effects == "左":
        image_clip = resize_image(image_path, width + spacing * duration, height).set_duration(duration)
    if video_effects == "左放大":
        image_clip = resize_image(image_path, width + spacing * (duration - 2), height).set_duration(duration)
    if video_effects == "右放大":
        image_clip = resize_image(image_path, width + spacing * (duration - 2), height).set_duration(
            duration).set_position((-(spacing * (duration - 2)), 0))
    if video_effects == "右":
        image_clip = resize_image(image_path, width + spacing * duration, height).set_duration(
            duration).set_position((-(spacing * duration), 0))
    if video_effects == "上":
        image_clip = resize_image(image_path, width, height + spacing * duration).set_duration(duration).set_position(
            (0, 0))
    if video_effects == "下":
        image_clip = resize_image(image_path, width, height + spacing * duration).set_duration(
            duration).set_position((0, -spacing * duration))
    if video_effects == "右上":
        image_clip = resize_image(image_path, width + spacing * duration, height + spacing * duration).set_duration(
            duration).set_position((-spacing * duration, 0))
    if video_effects == "左上":
        image_clip = resize_image(image_path, width + spacing * duration, height + spacing * duration).set_duration(
            duration).set_position((0, 0))

    if video_effects not in [None, ""]:
        transform = partial(apply_shake_effect, begin=0, end=duration, type=video_effects)
        image_clip = image_clip.fl(transform)

    if video_filters not in [None, ""]:
        for object in video_filters:
            # 使用 split() 方法进行分割
            object = object.replace('，', ',').strip()
            split_str = object.split(':')
            # 获取分割后的子字符串
            name = split_str[1].split(',')[0]
            begin = int(split_str[1].split(',')[1])
            end = int(split_str[1].split(',')[2])
            apply_shake_partial = partial(apply_shake_effect, begin=begin, end=end, type=name)
            image_clip = image_clip.fl(apply_shake_partial)
    return image_clip


# 曝光图片
def exposure_effect(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


# 图片抖动
def jitter_effect(img):
    height, width, n = img.shape
    new_img = img[int(height * 0.1):int(height * 0.95), int(width * 0.1):int(width * 0.95)]
    new_img = cv2.resize(new_img, (width, height))
    return new_img


# 曝光加抖动
def exposure_jitter_effect(img, gamma):
    height, width, n = img.shape
    new_img = img[int(height * 0.1):int(height * 0.95), int(width * 0.1):int(width * 0.95)]
    new_img = cv2.resize(new_img, (width, height))
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(new_img, gamma_table)


# 图片震动
def vibration_effect(image, intensity=5):
    # 获取图像的高度和宽度
    height, width = image.shape[:2]
    # 随机生成震动的偏移量
    dx = np.random.randint(-intensity, intensity + 1)
    dy = np.random.randint(-intensity, intensity + 1)
    # 定义仿射变换矩阵
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    # 应用仿射变换
    shaken_image = cv2.warpAffine(image, matrix, (width, height))
    return shaken_image


# 微震动
def vibration_weak_effect(image, intensity=5):
    # 获取图像的高度和宽度
    height, width = image.shape[:2]
    # 随机生成震动的偏移量
    dx = np.random.randint(-intensity, intensity + 1)
    dy = np.random.randint(-intensity, intensity + 1)
    # 定义仿射变换矩阵
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    # 应用仿射变换
    shaken_image = cv2.warpAffine(image, matrix, (width + spacing, height + spacing))
    return shaken_image


# 微震模糊
def weak_vibration_blur_effect(image, intensity=5, blur_radius=15):
    # 获取图像的高度和宽度
    height, width = image.shape[:2]

    # 随机生成震动的偏移量
    dx = np.random.randint(-intensity, intensity + 1)
    dy = np.random.randint(-intensity, intensity + 1)
    # 定义仿射变换矩阵
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    # 应用仿射变换
    shaken_image = cv2.warpAffine(image, matrix, (width + spacing, height + spacing))
    # 将震动后的图像进行模糊处理
    shaken_blurred_image = cv2.GaussianBlur(shaken_image, (blur_radius, blur_radius), 0).astype(np.uint8)
    return shaken_blurred_image




# 模糊处理
def blur_effect(image, blur_radius=15):
    # 将图像转换为浮点数类型
    image_float = image.astype(float)
    # 对图像进行模糊处理
    blur_radius = blur_radius if blur_radius % 2 == 1 else blur_radius + 1  # 确保blur_radius是奇数
    blurred_image = cv2.GaussianBlur(image_float, (blur_radius, blur_radius), 0).astype(np.uint8)
    return blurred_image


# 图片透视
def apply_perspective_transform(image, pts1, pts2, output_size=(300, 300)):
    # 获取透视变换矩阵
    perspective_matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # 应用透视变换
    perspective_transformed_image = cv2.warpPerspective(image, perspective_matrix, output_size)
    return perspective_transformed_image


# 图像膨胀
def dilation_effect(image, kernel_size=(5, 5), iterations=1):
    # 定义膨胀核
    kernel = np.ones(kernel_size, np.uint8)
    # 应用图像膨胀
    dilated_image = cv2.dilate(image, kernel, iterations=iterations)
    return dilated_image


# 白色发光
def soul_out_of_body_effect(image):
    # 将图像转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 阈值化处理，将白色区域设为255，其余为0
    _, thresholded_image = cv2.threshold(gray_image, 250, 255, cv2.THRESH_BINARY)
    # 膨胀操作，扩大白色区域
    kernel = np.ones((5, 5), np.uint8)
    dilated_image = cv2.dilate(thresholded_image, kernel, iterations=1)

    # 将膨胀后的白色区域放大到原始图像上
    key_frame = image.copy()
    key_frame[dilated_image != 0] = [255, 255, 255]  # 设置白色区域为白色
    result = cv2.addWeighted(image, 1, key_frame, 0.8, 0)
    return result


# 腐蚀
def erosion_effect(image, kernel_size=(5, 5), iterations=1):
    # 定义腐蚀核
    kernel = np.ones(kernel_size, np.uint8)
    # 应用图像腐蚀
    eroded_image = cv2.erode(image, kernel, iterations=iterations)
    return eroded_image


# 灵魂出窍
def soul_out_ofeffect(image, t, time):
    # image = get_frame(t)
    # 一次灵魂出窍效果的时长
    duration = time
    # 透明度上限值
    max_alpha = 0.4
    # 图片放大的上限
    max_scale = 1.5
    progress = t % duration / duration
    # 当前透明度 【0.4， 0】
    alpha = max_alpha * (1.0 - progress)
    # 当前缩放比例 【1.0， 1.4】
    scale = 1.0 + (max_scale - 1.0) * progress

    # 图像放大
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 0, scale)
    resized_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    # 颜色混合
    result_image = cv2.addWeighted(image, 1.0 - alpha, resized_image, alpha, 0)
    return result_image


# 放大
def zoom_frame(image, scale=1.2):
    h, w, _ = image.shape
    center = (w // 2, h // 2)

    # 构建仿射变换矩阵
    M = np.array([[scale, 0, (1 - scale) * center[0]],
                  [0, scale, (1 - scale) * center[1]]], dtype=np.float32)

    # 进行仿射变换
    zoomed_frame = cv2.warpAffine(image, M, (w, h))

    return zoomed_frame


# 左移动
def shift_frame_left(image, shift_pixels=50):
    h, w, _ = image.shape

    # 构建仿射变换矩阵（左移动）
    M = np.float32([[1, 0, -shift_pixels],
                    [0, 1, 0]])

    # 进行仿射变换
    shifted_frame = cv2.warpAffine(image, M, (w, h))

    return shifted_frame


def shift_frame_left_top(image, t, end, shift_pixels=50):
    h, w, _ = image.shape

    # 构建仿射变换矩阵（左移动）
    M_shift = np.float32([[1, 0, -shift_pixels],
                          [0, 1, 0]])

    # 构建仿射变换矩阵（放大）
    M_zoom = np.float32([[1, 0, -((end - 2) * spacing)],
                         [0, 1, 0]])

    if t < (end - 2):
        # 进行左移动的仿射变换
        shifted_frame = cv2.warpAffine(image, M_shift, (w, h))
        return shifted_frame
    else:
        # 先左移动
        shifted_frame = cv2.warpAffine(image, M_shift, (w, h))

        # 获取左移动后的图像宽度
        h, w, _ = shifted_frame.shape

        # 计算中心点和缩放比例
        center = ((w + int(spacing * (end - 2))) // 2, h // 2)
        scale = 1.0 + (1.2 - 1.0) * (t - (end - 2)) / (end - (end - 2))

        # 构建放大的仿射变换矩阵
        M_zoom = np.array([[scale, 0, (1 - scale) * center[0]],
                           [0, scale, (1 - scale) * center[1]]], dtype=np.float32)

        # 获取最终放大后的图像
        target_width = w + int(spacing * (end - 2))
        zoomed_frame = cv2.warpAffine(shifted_frame, M_zoom, (target_width, h))
        return zoomed_frame


# 右移动放大
def shift_frame_right_top(image, t, end, shift_pixels=50):
    h, w, _ = image.shape

    # 构建仿射变换矩阵（右移动）
    M_shift = np.float32([[1, 0, shift_pixels],
                          [0, 1, 0]])

    # 构建仿射变换矩阵（放大）
    M_zoom = np.float32([[1, 0, 0],
                         [0, 1, 0]])

    if t < (end - 2):
        # 进行右移动的仿射变换
        shifted_frame = cv2.warpAffine(image, M_shift, (w, h))
        return shifted_frame
    else:
        # 先右移动
        shifted_frame = cv2.warpAffine(image, M_shift, (w, h))

        # 计算中心点和缩放比例
        center = ((w - int(spacing * (end - 2))) // 2, h // 2)
        scale = 1.0 + (1.2 - 1.0) * (t - (end - 2)) / (end - (end - 2))

        # 调整放大的仿射变换矩阵，考虑右移动的影响
        M_zoom[0, 0] = scale
        M_zoom[1, 1] = scale
        M_zoom[0, 2] = (1 - scale) * center[0]
        M_zoom[1, 2] = (1 - scale) * center[1]

        # 获取最终放大后的图像
        target_width = w + int(spacing * (end - 2))
        zoomed_frame = cv2.warpAffine(shifted_frame, M_zoom, (target_width, h))
        return zoomed_frame


# 右移动
def shift_frame_right(image, shift_pixels=50):
    h, w, _ = image.shape

    # 构建仿射变换矩阵（右移动）
    M = np.float32([[1, 0, shift_pixels],
                    [0, 1, 0]])

    # 进行仿射变换
    shifted_frame = cv2.warpAffine(image, M, (w, h))

    return shifted_frame


# 向上移动效果
def shift_frame_up(image, shift_pixels=50):
    h, w, _ = image.shape

    # 构建仿射变换矩阵（向上移动）
    M = np.float32([[1, 0, 0],
                    [0, 1, -shift_pixels]])

    # 进行仿射变换
    shifted_frame = cv2.warpAffine(image, M, (w, h))

    return shifted_frame


# 向下移动效果
def shift_frame_down(image, shift_pixels=50):
    h, w, _ = image.shape

    # 构建仿射变换矩阵（向下移动）
    M = np.float32([[1, 0, 0],
                    [0, 1, shift_pixels]])

    # 进行仿射变换
    shifted_frame = cv2.warpAffine(image, M, (w, h))

    return shifted_frame


# 向右上角45度移动效果
def shift_frame_upright(image, shift_pixels=50):
    h, w, _ = image.shape

    # 构建仿射变换矩阵（向右上角45度移动）
    M = np.float32([[1, 0, min(shift_pixels, w)],
                    [0, 1, -min(shift_pixels, h)]])

    # 进行仿射变换
    shifted_frame = cv2.warpAffine(image, M, (w, h))
    return shifted_frame


# 向左上角45度移动效果
def shift_frame_upleft(image, shift_pixels=50):
    h, w, _ = image.shape

    # 构建仿射变换矩阵（向左上角45度移动）
    M = np.float32([[1, 0, -shift_pixels],
                    [0, 1, -shift_pixels]])

    # 进行仿射变换
    shifted_frame = cv2.warpAffine(image, M, (w, h))

    return shifted_frame


def weak_vibration_blur_effect1(image, intensity=5):
    # 获取图像的高度和宽度
    height, width = image.shape[:2]

    # 随机生成震动的偏移量
    dx = np.random.randint(-intensity, intensity + 1, (height, width))
    dy = np.random.randint(-intensity, intensity + 1, (height, width))

    # 将震动的偏移量转换为浮点数类型
    dx = dx.astype(np.float32)
    dy = dy.astype(np.float32)

    # 应用仿射变换
    shaken_image = cv2.remap(image, dx, dy, cv2.INTER_LINEAR)

    # 将震动后的图像进行模糊处理
    blur_radius = 15 if 15 % 2 == 1 else 16  # 确保blur_radius是奇数
    shaken_blurred_image = cv2.GaussianBlur(shaken_image, (blur_radius, blur_radius), 0).astype(np.uint8)

    return shaken_blurred_image


# 向上放大
def top_and_zoom(image, shift_pixels):
    pass


def apply_shake_effect(get_frame, t, begin, end, type):
    frame = get_frame(t)
    if type == "灵魂出窍" and begin <= t < end:
        return soul_out_ofeffect(frame, t, (end - begin) / 3)
    if type == "放大" and begin <= t < end:
        scale = 1.0 + (1.2 - 1.0) * t / (end - begin)
        return zoom_frame(frame, scale)
    if type == "左" and begin <= t < end:
        speed = spacing
        position = speed * t
        return shift_frame_left(frame, position)
    if type == "左放大" and begin <= t < end:
        speed = spacing
        position = speed * t
        return shift_frame_left_top(frame, t, end, position)

    if type == "右放大" and begin <= t < end:
        speed = spacing
        position = speed * t
        return shift_frame_right_top(frame, t, end, position)
    if type == "右" and begin <= t < end:
        speed = spacing
        position = speed * t
        return shift_frame_right(frame, position)
    if type == "上" and begin <= t < end:
        speed = spacing
        position = speed * t
        return shift_frame_up(frame, position)
    if type == "下" and begin <= t < end:
        speed = spacing
        position = speed * t
        return shift_frame_down(frame, position)
    if type == "右上" and begin <= t < end:
        speed = spacing
        position = speed * t
        return shift_frame_upright(frame, position)
    if type == "左上" and begin <= t < end:
        speed = spacing
        position = speed * t
        return shift_frame_upleft(frame, position)

    if type == "曝光" and (int(t / 0.15) % 2) == 0 and begin <= t < end:
        return exposure_effect(frame, 0.5)
    if type == "抖动" and (int(t / 0.15) % 2) == 0 and begin <= t < end:
        return jitter_effect(frame)
    if type == "震动" and (int(t / 0.15) % 2) == 0 and begin <= t < end:
        return vibration_effect(frame, spacing)
    if type == "微震动" and (int(t / 0.15) % 2) == 0 and begin <= t < end:
        return vibration_weak_effect(frame, 2)
    if type == "震动模糊" and (int(t / 0.05) % 2) == 0 and begin <= t < end:
        return weak_vibration_blur_effect(frame, 1, 15)
    if type == "模糊" and (int(t / 0.15) % 2) == 0 and begin <= t < end:
        return blur_effect(frame, 40)
    if type == "透视" and (int(t / 0.15) % 2) == 0 and begin <= t < end:
        pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
        pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
        return apply_perspective_transform(frame, pts1, pts2)
    if type == "膨胀" and (int(t / 0.15) % 2) == 0 and begin <= t < end:
        kernel_size = (10, 10)
        iterations = 1
        return dilation_effect(frame, kernel_size, iterations)
    if type == "发光" and (int(t / 0.15) % 2) == 0 and begin <= t < end:
        return soul_out_of_body_effect(frame)
    if type == "腐蚀" and (int(t / 0.15) % 2) == 0 and begin <= t < end:
        kernel_size = (10, 10)
        iterations = 1
        return erosion_effect(frame, kernel_size, iterations)
    if type == "曝光抖动" and (int(t / 0.15) % 2) == 0 and begin <= t < end:
        return exposure_jitter_effect(frame, 0.5)
    if type == "毛玻璃" and begin <= t < end:
        return blur_effect_2(frame, 15)
    return get_frame(t)


#毛玻璃 高斯模糊
def blur_effect_2(image, blur_radius):
    # return image.filter(ImageFilter.GaussianBlur(blur_radius))
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    # Apply Gaussian Blur
    blurred_image_cv = cv2.GaussianBlur(image_cv, (blur_radius, blur_radius), 0)

    # Convert back to PIL Image
    blurred_image = Image.fromarray(cv2.cvtColor(blurred_image_cv, cv2.COLOR_BGR2RGB))
    return blurred_image