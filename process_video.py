import os
import re
import time
from itertools import cycle

from moviepy.editor import *

import constant
# from moviepy.audio.io.AudioFileClip import AudioFileClip
# from moviepy.video.VideoClip import ColorClip
# from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
# from moviepy.video.compositing.concatenate import concatenate_videoclips
# from moviepy.video.io.VideoFileClip import VideoFileClip

from process_image_effect import precess_image_clip
from process_sound_effect import process_sound_effect
from process_text import process_subtitle, process_right_subtitle, process_left_subtitle
from video_merge import transitions_style_generate


# options_1 = ["上", "下", "上", "下", "左", "右", "左", "右", "默认"]
# effects_cycle_1 = cycle(options_1)
# current_effect_index_1 = 0
#
#
# def get_next_effect_1():
#     global current_effect_index_1
#     effect = options_1[current_effect_index_1]
#     current_effect_index_1 = (current_effect_index_1 + 1) % len(options_1)
#     return effect


def video_process(text, image_path=None, audio_path=None, width=None, height=None, effects=None, output_path=None):
    # 检查文件是否存在
    if not os.path.exists(audio_path):
        print(f"文件 '{audio_path}' 不存在")
        return

    # 检查文件格式是否支持
    _, ext = os.path.splitext(audio_path)
    if ext.lower() not in ['.mp3', '.MP3', '.wav', '.ogg', '.flac']:
        print(f"文件 '{audio_path}' 的格式不受支持")
        return

    effects_array, prompt = constant.cut_prompt(text, "sound")
    if len(effects_array) > 0:
        process_sound_effect(audio_path, sounds=effects_array)

    audio = AudioFileClip(audio_path)
    duration = audio.duration

    # 合成文本剪辑
    text_clip = process_subtitle(text, duration, width)

    background_color = (255, 255, 255, 0)
    video = ColorClip((width, height), background_color, duration=duration)

    filters_array, prompt = constant.cut_prompt(text, "video")

    shaken_clip = precess_image_clip(image_path, duration, effects, width, height, filters=filters_array)

    motion_array, prompt = constant.cut_prompt(text, "motion")

    if len(motion_array) > 0:
        # 添加 GIF 图片
        for object in motion_array:
            # 使用 split() 方法进行分割
            object = object.replace('，', ',').replace('：', ':').strip()
            split_str = object.split(':')
            # 获取分割后的子字符串
            name = split_str[1]
            # gif_path = os.path.join(os.path.dirname(__file__), 'asset', 'motion_effects', name + ".gif")

            base_path = os.path.join(os.path.dirname(__file__), 'asset', 'motion_effects')
            gif_path = os.path.join(base_path, name + ".gif")
            mov_path = os.path.join(base_path, name + ".mov")
            if os.path.exists(gif_path):
                gif_path = gif_path
            if os.path.exists(mov_path):
                gif_path = mov_path

            # 添加 GIF 图片
            gif_clip = VideoFileClip(gif_path, has_mask=True)

            if re.search("底部火焰", name):
                gif_clip = gif_clip.resize((width, height))
                gif_clip = gif_clip.loop(duration=duration)
                gif_position = ('center', video.size[1] - gif_clip.size[1])
            elif re.search("水滴", name):
                gif_clip = gif_clip.resize((512, 512))
                gif_clip = gif_clip.set_duration(duration)
                gif_position = ('center', 'center')
            elif re.search("蝴蝶", name):
                gif_clip = gif_clip.resize((width, height))
                gif_clip = gif_clip.set_duration(duration)
                gif_position = ('center', 'top')
            elif re.search("闪电", name):
                gif_clip = gif_clip.set_duration(duration)
                gif_clip = gif_clip.resize((width, height))
                gif_position = ('center', 'top')
            elif re.search("雷击", name):
                gif_clip = gif_clip.set_duration(duration)
                gif_clip = gif_clip.resize((width, height))
                gif_position = ('center', 'top')
            else:
                gif_clip = gif_clip.resize((width, height))
                gif_clip = gif_clip.loop(duration=duration)
                gif_position = ('center', 'top')

            gif_clip = gif_clip.set_mask(gif_clip.mask)
            gif_clip = gif_clip.set_position(gif_position)

            final_clip = CompositeVideoClip([video, shaken_clip, shaken_clip.set_audio(audio), gif_clip, text_clip])
            # final_clip = CompositeVideoClip([video, shaken_clip, shaken_clip.set_audio(audio), gif_clip])

    else:
        final_clip = CompositeVideoClip([video, shaken_clip, shaken_clip.set_audio(audio), text_clip])
        # final_clip = CompositeVideoClip([video, shaken_clip, shaken_clip.set_audio(audio)])

    timestamp = int(time.time())
    video_path = os.path.join(output_path, 'video', f'{timestamp}.mp4')
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    final_clip.write_videofile(video_path, codec='libx264', audio_codec='aac', fps=40)
    return video_path


def merge_video(out_path, background_audio_name='', subtitle_left="", subtitle_right="", width=750):
    video_folder_path = os.path.join(out_path, 'video')

    video_files = sorted([f for f in os.listdir(video_folder_path) if f.endswith(".mp4")])

    video_clips = [VideoFileClip(os.path.join(video_folder_path, file)) for file in video_files]

    # 提取每个视频剪辑的音频部分，并将其存储在一个列表中
    audio_clips = [AudioFileClip(os.path.join(video_folder_path, file)) for file in video_files]

    final_clips = transitions_style_generate(video_clips, "随机")

    # 使用 clips_array 函数将视频剪辑合成成一个视频
    # final_video = concatenate_videoclips(video_clips, method="compose")
    final_video = concatenate_videoclips(final_clips, method="chain")
    # 使用 concatenate_audioclips 函数将音频剪辑连接在一起
    final_audio = concatenate_audioclips(audio_clips)

    text_clips = []
    if subtitle_left != "":
        text_left_clip = process_left_subtitle(subtitle_left, final_video.duration, width=width)
        text_clips.append(text_left_clip)

    if subtitle_right != "":
        text_right_clip = process_right_subtitle(subtitle_right, final_video.duration, width=width, video_width=width)
        text_clips.append(text_right_clip)

    if background_audio_name != '':
        # 集成背景音乐
        folder_path = os.path.join(os.path.dirname(__file__), "asset", 'background_music', background_audio_name)
        background_music = AudioFileClip(folder_path)
        background_music = background_music.volumex(0.18)

        if background_music.duration > final_video.duration or background_music.duration == final_video.duration:
            background_music = background_music.subclip(0, final_video.duration)
        else:
            background_music = afx.audio_loop(background_music, duration=final_audio.duration)

        # video = CompositeVideoClip([final_video.set_audio(final_audio), final_video.set_audio(background_music)])
        #video = CompositeVideoClip([final_video.set_audio(final_audio)] + text_clips)
        #video = video.set_audio(background_music)
        # 合并背景音乐和原音频
        combined_audio = CompositeAudioClip([final_audio, background_music])
    else:
        combined_audio = final_audio
        # video = final_video
        #video = CompositeVideoClip([final_video.set_audio(final_audio)] + text_clips)

        # 将视频和字幕合成，同时保留最终的音频
    video = CompositeVideoClip([final_video] + text_clips).set_audio(combined_audio)

    timestamp = int(time.time())
    video_path = os.path.join(out_path, 'results', f'视频_{timestamp}.mp4')
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    video.write_videofile(video_path, codec="libx264", audio_codec="aac")
    return video_path


def delete_file(file_path):
    for filename in os.listdir(file_path):
        full_path = os.path.join(file_path, filename)
        if os.path.isfile(full_path):
            os.remove(full_path)
            print(f"已删除文件: {full_path}")
