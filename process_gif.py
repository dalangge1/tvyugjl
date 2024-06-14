# import os
# import time
#
# from PIL import Image
# import numpy as np
# from moviepy.video.VideoClip import ImageClip
# from moviepy.video.compositing.CompositeVideoClip import clips_array
#
#
# def make_gif_transparent(input_path, output_path, threshold=10):
#     # 打开 GIF 文件
#     gif = Image.open(input_path)
#
#     # 创建一个空白的透明图像列表，用于保存每一帧
#     transparent_frames = []
#
#     # 提取 GIF 的调色板
#     palette = gif.getpalette()
#     transparent_color = (0, 0, 0, 0)
#
#     # 遍历 GIF 的每一帧
#     for frame in range(gif.n_frames):
#         gif.seek(frame)
#
#         # 获取当前帧
#         frame_image = gif.convert("RGBA")
#         frame_array = frame_image.load()
#
#         # 将黑色背景转换为透明
#         for x in range(frame_image.width):
#             for y in range(frame_image.height):
#                 # 获取像素值
#                 pixel = frame_array[x, y]
#
#                 # 判断是否为黑色像素
#                 if pixel[0] < threshold and pixel[1] < threshold and pixel[2] < threshold:
#                     frame_array[x, y] = transparent_color
#
#         # 将透明帧添加到透明帧列表中
#         transparent_frames.append(frame_image)
#
#     # 保存处理后的透明 GIF 图像
#     if len(palette) % 256 != 0:
#         # 如果调色板大小不是 256 的倍数，将图像转换为最接近的有效调色板大小
#         gif = gif.quantize(method=2)  # 2 表示 MEDIANCUT 方法，也可以尝试其他方法
#
#
#
#     timestamp = int(time.time())
#     save_restore_path = os.path.join(output_path, f'{timestamp}.gif')
#     os.makedirs(os.path.dirname(save_restore_path), exist_ok=True)
#
#     # 保存处理后的透明 GIF 图像
#     gif.save(save_restore_path, format="GIF", append_images=transparent_frames[1:], save_all=True, optimize=True, transparency=0)
#
# # def make_gif_transparent(input_path, output_path):
# #     # 打开 GIF 文件
# #     gif = Image.open(input_path)
# #
# #     # 创建一个空白的透明图像列表，用于保存每一帧
# #     transparent_frames = []
# #
# #     # 遍历 GIF 的每一帧
# #     for frame in range(gif.n_frames):
# #         gif.seek(frame)
# #
# #         # 获取当前帧
# #         frame_image = gif.convert("RGBA")
# #
# #         # 将黑色背景转换为透明
# #         transparent_frame = Image.new("RGBA", gif.size)
# #         for x in range(frame_image.width):
# #             for y in range(frame_image.height):
# #                 pixel = frame_image.getpixel((x, y))
# #                 if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
# #                     pixel = (0, 0, 0, 0)
# #                 transparent_frame.putpixel((x, y), pixel)
# #
# #         # 将透明帧添加到透明帧列表中
# #         transparent_frames.append(transparent_frame)
# #
# #     timestamp = int(time.time())
# #     save_restore_path = os.path.join(output_path, f'{timestamp}.gif')
# #     os.makedirs(os.path.dirname(save_restore_path), exist_ok=True)
# #
# #     # 保存处理后的透明 GIF 图像
# #     transparent_frames[0].save(save_restore_path, format="GIF", append_images=transparent_frames[1:], save_all=True,
# #                                optimize=True, transparency=0)
# #
# #     return save_restore_path
#
#
# # output_path = r"F:\gifs"
# # input_path = r"C:\Users\asus\Desktop\gifsucai\ab39ef2145fb47feb67ef4cec94ca8a6.gif"
# # timestamp = int(time.time())
# # save_restore_path = os.path.join(output_path, f'{timestamp}.gif')
# # os.makedirs(os.path.dirname(save_restore_path), exist_ok=True)
# #
# # make_gif_transparent(input_path, output_path)
#
#
#
# from PIL import Image, ImageSequence
#
#
# # 保存结
#
# from moviepy.editor import ImageSequenceClip
#
# # 打开两个 GIF 图像
# gif1 = ImageClip(r"C:\Users\asus\Desktop\gifsucai\d3fca3da0e9f40bc945b86367ee7b23b.gif").set_duration(5)
# gif2 = ImageClip(r"C:\Users\asus\Desktop\gifsucai\d3fca3da0e9f40bc945b86367ee7b23b.gif").set_duration(5)
#
# # 设置输出视频的尺寸为两个 GIF 图像的宽度之和和高度中的最大值
# max_width = max(gif1.w, gif2.w)
# max_height = max(gif1.h, gif2.h)
# output_size = (max_width * 2, max_height)
#
# # 将两个 GIF 图像并排显示
# final_clip = clips_array([[gif1.resize(width=max_width), gif2.resize(width=max_width)]])
#
# # 保存结果视频为 GIF
# final_clip.write_gif(r"F:\gifs\merged2.gif", fps=10)
#
#
#
# # output_path = r"F:\gifs"
# # input_path = r"C:\Users\asus\Desktop\gifsucai\c3b809007a8e4558a72b8605f800d19e.gif"
# # timestamp = int(time.time())
# # save_restore_path = os.path.join(output_path, f'{timestamp}.gif')
# # os.makedirs(os.path.dirname(save_restore_path), exist_ok=True)
# # concatenate_gif(input_path, output_path)