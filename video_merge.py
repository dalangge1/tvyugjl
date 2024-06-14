import os
import random
import time
from itertools import cycle

from moviepy.audio.AudioClip import concatenate_audioclips
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.editor import VideoFileClip, concatenate_videoclips, VideoClip, afx
import numpy as np
import cv2
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip


def resize_frame_to_match(frame, target_shape):
    """Resize frame to match target shape while maintaining aspect ratio."""
    target_h, target_w, _ = target_shape
    h, w, _ = frame.shape
    scale = min(target_w / w, target_h / h)
    resized_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    pad_h = (target_h - resized_frame.shape[0]) // 2
    pad_w = (target_w - resized_frame.shape[1]) // 2
    pad_h1 = target_h - resized_frame.shape[0] - pad_h
    pad_w1 = target_w - resized_frame.shape[1] - pad_w
    return cv2.copyMakeBorder(resized_frame, pad_h, pad_h1, pad_w, pad_w1, cv2.BORDER_CONSTANT, value=[0, 0, 0])


def ease_in_out_cubic(x):
    """Cubic easing function for a smooth transition effect."""
    if x < 0.5:
        return 4 * x * x * x
    else:
        return 1 - pow(-2 * x + 2, 3) / 2


def vertical_slide_transition(clip1, clip2, duration):
    """Create a transition clip that slides from clip1 to clip2 vertically with easing effect."""

    def make_frame(t):
        frame1 = clip1.get_frame(t) if t < clip1.duration else clip1.get_frame(clip1.duration - 0.01)
        frame2 = clip2.get_frame(t) if t < clip2.duration else clip2.get_frame(clip2.duration - 0.01)

        # Ensure frames have the same shape
        frame1 = resize_frame_to_match(frame1, frame2.shape)
        frame2 = resize_frame_to_match(frame2, frame1.shape)

        h, w, _ = frame1.shape
        progress = t / duration if t < duration else 1
        eased_progress = ease_in_out_cubic(progress)
        offset = int(h * eased_progress)

        new_frame = np.copy(frame1)
        if offset < h:
            new_frame[:h - offset, :, :] = frame1[offset:, :, :]
            new_frame[h - offset:, :, :] = frame2[:offset, :, :]
        else:
            new_frame[:, :, :] = frame2[:, :, :]

        return new_frame

    return VideoClip(make_frame, duration=max(clip1.duration, clip2.duration))


def horizontal_cover_transition_left_to_right(clip1, clip2, duration):
    """Create a transition clip that covers from clip1 to clip2 horizontally from left to right."""

    def make_frame(t):
        frame1 = clip1.get_frame(t) if t < clip1.duration else clip1.get_frame(clip1.duration - 0.01)
        frame2 = clip2.get_frame(t) if t < clip2.duration else clip2.get_frame(clip2.duration - 0.01)

        h1, w1, _ = frame1.shape
        h2, w2, _ = frame2.shape

        if h1 != h2 or w1 != w2:
            raise ValueError("clip1 and clip2 must have the same dimensions")

        h, w = h1, w1
       # h, w, _ = frame1.shape
        progress = t / duration if t < duration else 1
        eased_progress = ease_in_out_cubic(progress)
        offset = int(w * eased_progress)

        new_frame = np.copy(frame1)
        if offset < w:
            new_frame[:, offset:, :] = frame1[:, :w - offset, :]
            new_frame[:, :offset, :] = frame2[:, w - offset:, :]
        else:
            new_frame[:, :, :] = frame2[:, :, :]

        return new_frame

    return VideoClip(make_frame, duration=max(clip1.duration, clip2.duration))


def horizontal_cover_transition_right_to_left(clip1, clip2, duration):
    """Create a transition clip that covers from clip1 to clip2 horizontally from right to left."""

    def make_frame(t):
        frame1 = clip1.get_frame(t) if t < clip1.duration else clip1.get_frame(clip1.duration - 0.01)
        frame2 = clip2.get_frame(t) if t < clip2.duration else clip2.get_frame(clip2.duration - 0.01)

        h1, w1, _ = frame1.shape
        h2, w2, _ = frame2.shape

        if h1 != h2 or w1 != w2:
            raise ValueError("clip1 and clip2 must have the same dimensions")

        h, w = h1, w1

        #h, w, _ = frame1.shape
        progress = t / duration if t < duration else 1
        eased_progress = ease_in_out_cubic(progress)
        offset = int(w * eased_progress)

        new_frame = np.copy(frame1)
        if offset < w:
            new_frame[:, :w - offset, :] = frame1[:, offset:, :]
            new_frame[:, w - offset:, :] = frame2[:, :offset, :]
        else:
            new_frame[:, :, :] = frame2[:, :, :]

        return new_frame

    return VideoClip(make_frame, duration=max(clip1.duration, clip2.duration))


def vertical_cover_transition_top_to_bottom(clip1, clip2, duration):
    """Create a transition clip where clip2 covers clip1 from top to bottom."""

    def make_frame(t):
        frame1 = clip1.get_frame(t) if t < clip1.duration else clip1.get_frame(clip1.duration - 0.01)
        frame2 = clip2.get_frame(t) if t < clip2.duration else clip2.get_frame(clip2.duration - 0.01)

        h, w, _ = frame1.shape
        progress = t / duration if t < duration else 1
        eased_progress = ease_in_out_cubic(progress)
        offset = int(h * eased_progress)

        new_frame = np.copy(frame1)
        if offset < h:
            new_frame[offset:, :, :] = frame1[:h - offset, :, :]
            new_frame[:offset, :, :] = frame2[h - offset:, :, :]
        else:
            new_frame[:, :, :] = frame2[:, :, :]

        return new_frame

    return VideoClip(make_frame, duration=max(clip1.duration, clip2.duration))


def apply_random_transition(clip1, clip2, duration):
    """Apply a random transition between two clips."""
    transitions = [vertical_slide_transition, horizontal_cover_transition_left_to_right,
                   horizontal_cover_transition_right_to_left, vertical_cover_transition_top_to_bottom
                   ]
    transition_func = random.choice(transitions)
    return transition_func(clip1, clip2, duration)


options_1 = ["默认", "上", "下", "上", "下", "上", "下", "上", "下", "上", "下", "上", "下", "上", "下", "上", "下", "左", "右", "左", "右", ]
effects_cycle = cycle(options_1)
current_effect_index_1 = 1


def get_next_effect():
    global current_effect_index_1
    effect = options_1[current_effect_index_1]
    current_effect_index_1 = (current_effect_index_1 + 1) % len(options_1)
    return effect


def load_transition_sound(transition_type):
    # Load the appropriate transition sound effect based on the transition type
    transition_sounds = {
        '上': './asset/sound_effects/转场音效2.mp3',
        '下': './asset/sound_effects/把剑.mp3',
        '左': './asset/sound_effects/转场音效1.mp3',
        '右': './asset/sound_effects/转场音效3.mp3',
        '默认': './asset/sound_effects/电子水滴.mp3',
    }
    return AudioFileClip(transition_sounds[transition_type])


def merge_video_effects(background_audio_name, out_path, transitions):
    video_folder_path = os.path.join(out_path, 'video')

    video_files = sorted([f for f in os.listdir(video_folder_path) if f.endswith(".mp4")])

    video_clips = [VideoFileClip(os.path.join(video_folder_path, file)) for file in video_files]

    audio_clips = [AudioFileClip(os.path.join(video_folder_path, file)) for file in video_files]

    final_clips = transitions_style_generate(video_clips, transitions)

    final_video = concatenate_videoclips(final_clips)

    final_audio = concatenate_audioclips(audio_clips)

    if background_audio_name != '':
        # 集成背景音乐
        folder_path = os.path.join(os.path.dirname(__file__), "asset", 'background_music', background_audio_name)
        background_music = AudioFileClip(folder_path)
        background_music = background_music.volumex(0.18)

        if background_music.duration > final_video.duration or background_music.duration == final_video.duration:
            background_music = background_music.subclip(0, final_video.duration)
        else:
            background_music = afx.audio_loop(background_music, duration=final_audio.duration)

        video = CompositeVideoClip([final_video.set_audio(final_audio), final_video.set_audio(background_music)])
    else:
        video = final_video

    timestamp = int(time.time())
    video_path = os.path.join(out_path, 'results', f'视频_{timestamp}.mp4')
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    video.write_videofile(video_path, codec="libx264", audio_codec="aac")
    return video_path


def transitions_style_generate(video_clips, transitions):
    final_clips = []

    if transitions != '默认':

        # Define the duration of each transition
        transition_duration = 0.15  # Duration of the transition in seconds

        # Apply transitions between clips
        for i in range(len(video_clips) - 1):
            clip1 = video_clips[i]
            clip2 = video_clips[i + 1]

            # Cut the last part of clip1 and the first part of clip2 for the transition
            clip1_sub = clip1.subclip(0, clip1.duration - transition_duration)
            clip2_sub = clip2.subclip(transition_duration, clip2.duration)

            if transitions == '随机':
                random_transitions = get_next_effect()
            else:
                random_transitions = transitions

            print("========转场动画=====".format(random_transitions))

            #transition_sound = load_transition_sound(random_transitions)

            if random_transitions == '上':
                transition_clip = vertical_slide_transition(
                    clip1.subclip(clip1.duration - transition_duration, clip1.duration),
                    clip2.subclip(0, transition_duration),
                    transition_duration)

            elif random_transitions == '下':
                transition_clip = vertical_cover_transition_top_to_bottom(
                    clip1.subclip(clip1.duration - transition_duration, clip1.duration),
                    clip2.subclip(0, transition_duration),
                    transition_duration)

            elif random_transitions == '左':
                transition_clip = horizontal_cover_transition_right_to_left(
                    clip1.subclip(clip1.duration - transition_duration, clip1.duration),
                    clip2.subclip(0, transition_duration),
                    transition_duration)

            elif random_transitions == '右':
                transition_clip = horizontal_cover_transition_left_to_right(
                    clip1.subclip(clip1.duration - transition_duration, clip1.duration),
                    clip2.subclip(0, transition_duration),
                    transition_duration)

            #transition_clip = transition_clip.set_audio(transition_sound.set_duration(transition_duration))

            # Append the clips and transition
            final_clips.append(clip1_sub)
            final_clips.append(transition_clip)
            if i == len(video_clips) - 2:
                final_clips.append(clip2_sub)
    else:
        final_clips = video_clips

    return final_clips
