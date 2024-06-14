# import os
# import bpy
# import cv2
# import numpy as np
# from moviepy.editor import VideoFileClip, ImageSequenceClip, CompositeVideoClip
#
# from PIL import Image, ImageFilter
#
#
# def apply_blur_effect(input_image_path, output_image_path, blur_radius):
#     # 打开输入图像
#     image = Image.open(input_image_path)
#
#     # 应用高斯模糊
#     #blurred_image = image.filter(ImageFilter.GaussianBlur(blur_radius))
#     # 保存结果图像
#     #blurred_image.save(output_image_path)
#     #print(f"毛玻璃效果图像已保存至: {output_image_path}")
#
#     # return image.filter(ImageFilter.GaussianBlur(blur_radius))
#     image_cv = np.array(image)
#     image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
#
#     # Apply Gaussian Blur
#     blurred_image_cv = cv2.GaussianBlur(image_cv, (blur_radius, blur_radius), 0)
#
#     # Convert back to PIL Image
#     blurred_image = Image.fromarray(cv2.cvtColor(blurred_image_cv, cv2.COLOR_BGR2RGB))
#     blurred_image.save(output_image_path)
#     return output_image_path
#
#
# # 示例用法
# input_image_path = r'C:\Users\asus\Desktop\高清图片\1716321197.jpg'  # 输入图像路径
# output_image_path = r'C:\Users\asus\Desktop\高清图片\111111111.jpg'  # 输出图像路径
# blur_radius = 15  # 模糊半径，可以根据需要调整
#
# apply_blur_effect(input_image_path, output_image_path, blur_radius)



#scene_path = "./path/to/your/main.scene"
#inspect_file(scene_path)

#render_scene(scene_path, output_dir)


import cv2
import numpy as np
import random

#
# def apply_particle_blur_effect(frame, num_particles, max_blur_radius):
#     height, width = frame.shape[:2]
#
#     # 对每个粒子进行处理
#     for _ in range(num_particles):
#         # 随机选择粒子中心
#         x = random.randint(0, width - 1)
#         y = random.randint(0, height - 1)
#
#         # 随机选择模糊半径
#         blur_radius = random.randint(1, max_blur_radius)
#         if blur_radius % 2 == 0:
#             blur_radius += 1
#
#         # 确定粒子的区域
#         x1 = max(0, x - blur_radius)
#         y1 = max(0, y - blur_radius)
#         x2 = min(width, x + blur_radius)
#         y2 = min(height, y + blur_radius)
#
#         # 提取区域并应用模糊
#         roi = frame[y1:y2, x1:x2]
#         blurred_roi = cv2.GaussianBlur(roi, (blur_radius, blur_radius), 0)
#
#         # 将模糊区域放回原图
#         frame[y1:y2, x1:x2] = blurred_roi
#
#     return frame
#
#
# def process_video(input_video_path, output_video_path, num_particles, max_blur_radius):
#     cap = cv2.VideoCapture(input_video_path)
#
#     if not cap.isOpened():
#         print(f"Error opening video file: {input_video_path}")
#         return
#
#     # 获取视频帧率
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
#     # 使用相同的编解码器和帧率保存视频
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
#
#     frame_count = 0
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         frame_count += 1
#
#         # 打印帧信息
#         print(f"Processing frame {frame_count}")
#
#         # 应用粒子模糊效果
#         frame_blurred = apply_particle_blur_effect(frame, num_particles, max_blur_radius)
#
#         # 写入帧到输出视频
#         out.write(frame_blurred)
#
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()
#     print(f"Finished processing {frame_count} frames")
#
#
# # 示例用法
# input_video_path = r'C:\Users\asus\Desktop\高清图片\1717045208.mp4'  # 输入视频路径
# output_video_path = r'C:\Users\asus\Desktop\高清图片\111111.mp4'  # 输出视频路径
# num_particles = 500  # 粒子数量，可以根据需要调整
# max_blur_radius = 21  # 最大模糊半径，可以根据需要调整
#
# process_video(input_video_path, output_video_path, num_particles, max_blur_radius)

import cv2

def adjust_blur_radius(frame_count):
    # 根据帧的数量动态调整模糊半径
    frame_tail = frame_count % 10  # 获取帧尾数
    if frame_tail == 1 or frame_tail == 6:
        return 11
    elif frame_tail == 2 or frame_tail == 4:
        return 9
    elif frame_tail == 3 or frame_tail == 5 or frame_tail == 7:
        return 13
    elif frame_tail == 8 or frame_tail == 9:
        return 9
    else:
        return 21  # 其他帧模糊半径为31


def process_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Error opening video file: {input_video_path}")
        return

    # 获取视频帧率和总帧数
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 使用相同的编解码器和帧率保存视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 打印帧信息
        print(f"Processing frame {frame_count}/{total_frames}")

        # 动态调整模糊半径
        blur_radius = adjust_blur_radius(frame_count)

        # 应用高斯模糊效果
        blurred_frame = cv2.GaussianBlur(frame, (blur_radius, blur_radius), 0)

        # 写入帧到输出视频
        out.write(blurred_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Finished processing {frame_count} frames")


# 示例用法
input_video_path = r'C:\Users\asus\Desktop\高清图片\1717045208.mp4'  # 输入视频路径
output_video_path = r'C:\Users\asus\Desktop\高清图片\111111.mp4'  # 输出视频路径

process_video(input_video_path, output_video_path)





# import cv2
# import numpy as np
# import random
#
#
# def apply_particle_blur_effect(frame, num_particles, min_blur_radius, max_blur_radius):
#     height, width = frame.shape[:2]
#
#     # 对每个粒子进行处理
#     for _ in range(num_particles):
#         # 随机选择粒子中心
#         x = random.randint(0, width - 1)
#         y = random.randint(0, height - 1)
#
#         # 随机选择模糊半径
#         blur_radius = random.randint(min_blur_radius, max_blur_radius)
#         if blur_radius % 2 == 0:
#             blur_radius += 1
#
#         # 确定粒子的区域
#         x1 = max(0, x - blur_radius)
#         y1 = max(0, y - blur_radius)
#         x2 = min(width, x + blur_radius)
#         y2 = min(height, y + blur_radius)
#
#         # 提取区域并应用模糊
#         roi = frame[y1:y2, x1:x2]
#         blurred_roi = cv2.GaussianBlur(roi, (blur_radius, blur_radius), 0)
#
#         # 将模糊区域放回原图
#         frame[y1:y2, x1:x2] = blurred_roi
#
#     return frame
#
#
# def process_video(input_video_path, output_video_path, num_particles, min_blur_radius, max_blur_radius):
#     cap = cv2.VideoCapture(input_video_path)
#
#     if not cap.isOpened():
#         print(f"Error opening video file: {input_video_path}")
#         return
#
#     # 获取视频帧率
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
#     # 使用相同的编解码器和帧率保存视频
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
#
#     frame_count = 0
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         frame_count += 1
#
#         # 打印帧信息
#         print(f"Processing frame {frame_count}")
#
#         # 应用粒子模糊效果多次
#         for _ in range(5):  # 通过增加处理次数增加模糊效果
#             frame = apply_particle_blur_effect(frame, num_particles, min_blur_radius, max_blur_radius)
#
#         # 写入帧到输出视频
#         out.write(frame)
#
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()
#     print(f"Finished processing {frame_count} frames")
#
#
# # 示例用法
# input_video_path = r'C:\Users\asus\Desktop\高清图片\1717045208.mp4'  # 输入视频路径
# output_video_path = r'C:\Users\asus\Desktop\高清图片\111111.mp4'  # 输出视频路径
# # num_particles = 2000  # 增加粒子数量
# # min_blur_radius = 19  # 最小模糊半径
# # max_blur_radius = 21  # 最大模糊半径
#
# num_particles = 2000  # 增加粒子数量
# min_blur_radius = 15  # 最小模糊半径
# max_blur_radius = 20  # 最大模糊半径
#
# process_video(input_video_path, output_video_path, num_particles, min_blur_radius, max_blur_radius)

