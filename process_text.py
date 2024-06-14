from moviepy.video.VideoClip import TextClip
from moviepy.video.compositing.concatenate import concatenate_videoclips
import re
import constant


def process_subtitle(text, duration=0, width=None):
    text = re.sub(r'\[[^\]]+\]', '', text)
    text_clips = []
    if text != None and text != "":
        fade_duration = 0.5
        sentences = re.split(r'(?<=[.?!。？！，,])', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        # 字幕处理
        # 字幕'./asset/font/song_ti.ttc'
        # youranti_1.ttf
        for c_text in sentences:
            rendering_text = remove(c_text)
            text_clip = TextClip(rendering_text, font='./asset/font/xinqingnian_1.ttf', fontsize=constant.font_size(width),
                                 color='yellow', stroke_color="black", stroke_width=1.2)
            text_clip = text_clip.set_duration(test_time(text, c_text, duration))
            # .crossfadein(fade_duration).crossfadeout(fade_duration)
            text_clip = text_clip.set_position(("center", 0.7), relative=True)
            text_clips.append(text_clip)

    # 合成文本剪辑
    offset = 0.84 if width == 1216 else 0.74
    text_video = concatenate_videoclips(text_clips).set_position(("center", offset), relative=True)
    return text_video


def process_left_subtitle(text, duration=0, width=None):
    # 创建一个包含每个字符的垂直排列的字符串
    vertical_text = '\n'.join(text)

    text_clip = TextClip(vertical_text, font='./asset/font/houxiandai_1.ttf', fontsize=constant.subtitle_font_size(width),
                         color='red', stroke_color="lightblue", stroke_width=1.2).set_duration(duration)

    text_clip = text_clip.set_position((20, 20))
    return text_clip


def process_right_subtitle(text, duration=0, width=None, video_width=1280, video_height=720):
    text_clip = TextClip(
        text,
        font='./asset/font/houxiandai_1.ttf',  # Adjust to your font path
        fontsize=constant.subtitle_font_size(width),
        color='red',  # Text color set to red
        stroke_color="lightblue",  # Stroke color
        stroke_width=1.2 # Stroke width
    ).set_duration(duration)

    # Calculate the position for the right top corner
    text_width, text_height = text_clip.size
    position = (video_width - text_width-20, 20)  # Right top corner

    # Set position to right top corner
    text_clip = text_clip.set_position(position)

    return text_clip




def test_time(text, c_text, duration):
    num_sentences = len(text)
    time_per_sentence = duration / num_sentences
    return time_per_sentence * len(c_text)


def remove(text):
    pattern = r'[.?!。？！，,]+$'
    match = re.search(pattern, text)
    if match:
        text = text[:match.start()]
    return text
