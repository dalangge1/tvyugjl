import gc
import os
from pathlib import Path

import numpy as np
import torchaudio
import torch
from pydub import AudioSegment
from pydub.silence import split_on_silence
from scipy.io import wavfile
from sympy.physics.quantum.identitysearch import scipy
import noisereduce as nr
import constant
from resemble_enhance.enhancer.inference import enhance

from pydub.effects import speedup, strip_silence, high_pass_filter, low_pass_filter, normalize

import pyroomacoustics as pra

enhance_model_path = os.path.join(constant.base_model_path, "resemble-enhance")

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def process_audio(file_path, features, silence_len=16, silence_thresh=-80, padding=16):
    if "增强" in features:
        # 增强音频效果
        sample_rate, data = _fn(file_path, "Midpoint", 64, 0.1, 10, 1, False)
        # 保存增强后的音频文件
        wavfile.write(file_path, sample_rate, data)
    if "切割" in features:
        # 切割音频
        audio = AudioSegment.from_file(file_path)
        trimmed_audio = strip_silence(audio, silence_len=silence_len, silence_thresh=silence_thresh, padding=padding)
        trimmed_audio.export(file_path, format='mp3')
    return file_path

    # 读取原始音频数据
    # if "增强" in fatures:
    #     # 增强音频效果
    #     sample_rate, data = _fn(file_path, "Midpoint", 64, 0.1, 10, 1, False)
    #     if "切割" in fatures:
    #         # 转换为 pydub 的 AudioSegment 对象
    #         # 确定通道数
    #         if len(data.shape) == 1:
    #             # 单声道
    #             channels = 1
    #         elif len(data.shape) == 2:
    #             # 多声道
    #             channels = data.shape[1]
    #         else:
    #             raise ValueError("Unexpected data shape")
    #
    #         # 将 numpy 数组转换为字节数据
    #         if channels == 1:
    #             audio_data = data.tobytes()
    #         else:
    #             # 如果有多个通道，需要转换为 interleaved 格式
    #             interleaved = np.empty((data.shape[0] * channels,), dtype=data.dtype)
    #             for i in range(channels):
    #                 interleaved[i::channels] = data[:, i]
    #             audio_data = interleaved.tobytes()
    #
    #         # 创建 AudioSegment 对象
    #         audio = AudioSegment(
    #             audio_data,
    #             frame_rate=sample_rate,
    #             sample_width=data.dtype.itemsize,  # 每个样本的字节大小
    #             channels=channels
    #         )
    #         # 切割音频
    #         trimmed_audio = apply_effects(audio, silence_len, silence_thresh, padding)
    #
    #         trimmed_audio.export(file_path, format='mp3')
    #         return file_path
    #     else:
    #         wavfile.write(file_path, sample_rate, data)
    #         return file_path
    # else:
    #     if "切割" in fatures:
    #         audio = AudioSegment.from_file(file_path)
    #         # 去除音频无声位置
    #         trimmed_audio = strip_silence(audio, silence_thresh=-50, padding=100)
    #         # 保存处理后的音频文件
    #         trimmed_audio.export(file_path, format='mp3')
    #         return file_path


def enhance_effects(data, sample_rate):
    # 对音频数据进行增强处理
    # _fn 是假定的增强处理函数
    return _fn(data, sample_rate, "Midpoint", 64, 0.1, 10, 1, False)


def apply_effects(audio, silence_len=100, silence_thresh=-80, padding=100):
    # 去除音频无声位置
    trimmed_audio = strip_silence(audio, silence_len=silence_len, silence_thresh=silence_thresh, padding=padding)
    return trimmed_audio


def strip_silence(audio, silence_len=100, silence_thresh=-80, padding=100):
    # 使用 pydub 的 split_on_silence 方法去除无声部分
    chunks = split_on_silence(
        audio,
        min_silence_len=silence_len,
        silence_thresh=silence_thresh,
        keep_silence=padding
    )
    # 将分割后的音频片段重新合成一个音频
    trimmed_audio = AudioSegment.empty()
    for chunk in chunks:
        trimmed_audio += chunk
    return trimmed_audio


def _fn(path, solver, nfe, tau, chunk_seconds, chunks_overlap, denoising):
    if path is None:
        return None, None

    solver = solver.lower()
    nfe = int(nfe)
    lambd = 0.9 if denoising else 0.1

    dwav, sr = torchaudio.load(path)
    dwav = dwav.mean(dim=0)

    wav2, new_sr = enhance(dwav=dwav, sr=sr, device=device, nfe=nfe, chunk_seconds=chunk_seconds,
                           chunks_overlap=chunks_overlap, solver=solver, lambd=lambd, tau=tau,
                           run_dir=Path(enhance_model_path))

    wav2 = wav2.cpu().numpy()
    clear_gpu_cash()
    return (new_sr, wav2)


# def apply_effects(file_path, silence_len=100, silence_thresh=-80, padding=100):
#     # 读取音频文件
#     audio = AudioSegment.from_file(file_path)
#     # 去除音频无声位置
#     trimmed_audio = strip_silence(audio, silence_len=silence_len, silence_thresh=silence_thresh, padding=padding)
#     # 保存处理后的音频文件
#     output_file_path = file_path
#     trimmed_audio.export(output_file_path, format='wav')
#     return output_file_path


def clear_gpu_cash():
    # del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# # 降噪（简单的示例，实际降噪可能需要更复杂的处理）
# def noise_reduction(audio, noise_start, noise_end):
#     noise_sample = audio[noise_start:noise_end]
#     noise_reduced = audio - noise_sample
#     return noise_reduced


# 均衡器 - 高通和低通滤波器
def pass_filter(audio, high=100, low_pass=8000):
    # 均衡器 - 高通和低通滤波器
    audio = high_pass_filter(audio, cutoff=high)  # 移除低于100Hz的频率
    audio = low_pass_filter(audio, cutoff=low_pass)  # 移除高于8000Hz的频率
    return audio


# 添加混响
# def add_reverb(audio, reverb_amount=0.2):
#     samples = np.array(audio.get_array_of_samples())
#     reverb = scipy.signal.convolve(samples, np.ones(int(reverb_amount * audio.frame_rate)),
#                                    mode='same') / audio.frame_rate
#     reverb_audio = AudioSegment(reverb.tobytes(), frame_rate=audio.frame_rate, sample_width=audio.sample_width,
#                                 channels=audio.channels)
#     return reverb_audio


# 动态范围压缩
def compress_dynamic_range(audio, threshold=-20.0, ratio=4.0):
    samples = np.array(audio.get_array_of_samples())
    threshold = np.max(samples) * (10 ** (threshold / 20.0))
    compressed = np.where(samples > threshold, threshold + (samples - threshold) / ratio, samples)
    compressed = np.where(compressed < -threshold, -threshold + (compressed + threshold) / ratio, compressed)
    compressed_audio = AudioSegment(compressed.tobytes(), frame_rate=audio.frame_rate, sample_width=audio.sample_width,
                                    channels=audio.channels)
    return compressed_audio


# 归一化音频
def normalize_audio(audio):
    return normalize(audio)


# 降噪
# def noise_reduction(audio_segment):
#     samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
#     reduced_noise_samples = nr.reduce_noise(y=samples, sr=audio_segment.frame_rate)
#     return AudioSegment(
#         reduced_noise_samples.tobytes(),
#         frame_rate=audio_segment.frame_rate,
#         sample_width=audio_segment.sample_width,
#         channels=audio_segment.channels
#     )


def add_reverb_0(file_path, room_dim1=10, room_dim2=10, room_dim3=3, rt60_tgt=0.6, microphone=1.5):
    room_dim = (room_dim1, room_dim2, room_dim3)
    rt60_tgt = rt60_tgt
    return add_reverb(file_path, room_dim, rt60_tgt, microphone)


# 添加混响
def add_reverb(file_path, room_dim=(10, 10, 3), rt60_tgt=0.6, microphone=1.5):
    # audio_segment = AudioSegment.from_file(file_path)
    # samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
    #
    # # 确保样本数据的范围在 -1 到 1 之间
    # max_val = np.max(np.abs(samples))
    # if max_val > 0:
    #     samples /= max_val
    #
    # # 计算吸收系数和最大反射次数
    # absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)
    #
    # # 创建虚拟房间
    # room = pra.ShoeBox(room_dim, fs=audio_segment.frame_rate, absorption=absorption, max_order=max_order)
    #
    # # 添加音源和麦克风
    # mic_locs = np.c_[[room_dim[0] / 2, room_dim[1] / 2, microphone]]  # 假设麦克风在房间中间1.5米高
    # room.add_microphone_array(mic_locs)
    # room.add_source([room_dim[0] / 3, room_dim[1] / 3, microphone], signal=samples)
    #
    # # 模拟混响
    # room.image_source_model()
    # room.compute_rir()
    # reverberant_signal = room.simulate(return_premix=True).flatten()
    #
    # # 确保输出样本数据的范围在 -1 到 1 之间
    # max_val = np.max(np.abs(reverberant_signal))
    # if max_val > 0:
    #     reverberant_signal /= max_val
    #
    # # 将混响后的样本转换回 AudioSegment
    # audio = AudioSegment(
    #     (reverberant_signal * np.iinfo(np.int16).max).astype(np.int16).tobytes(),
    #     frame_rate=audio_segment.frame_rate,
    #     sample_width=audio_segment.sample_width,
    #     channels=audio_segment.channels
    # )
    #
    # # 保存处理后的音频文件
    # output_file_path = file_path
    # audio.export(output_file_path, format='mp3')
    # return output_file_path

    audio_segment = AudioSegment.from_file(file_path)

    # Convert audio to numpy array
    samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
    frame_rate = audio_segment.frame_rate

    # max_val = np.max(np.abs(samples))
    # if max_val > 0:
    #     samples /= max_val
    #
    # reduced_noise_samples = nr.reduce_noise(y=samples, sr=frame_rate)
    #
    # reduced_noise_segment = AudioSegment(
    #     (reduced_noise_samples * np.iinfo(np.int16).max).astype(np.int16).tobytes(),
    #     frame_rate=frame_rate,
    #     sample_width=audio_segment.sample_width,
    #     channels=audio_segment.channels
    # )

    # Normalize
    normalized_segment = normalize(audio_segment)

    # Equalization
    eq_segment = low_pass_filter(normalized_segment, 12000)  # Low pass filter
    eq_segment = high_pass_filter(eq_segment, 80)  # High pass filter

    # Compression
    # compressed_segment = compress_dynamic_range(audio_segment)

    # Convert back to numpy array for reverb processing
    samples = np.array(eq_segment.get_array_of_samples(), dtype=np.float32)

    # Normalize samples to range [-1, 1]
    # 确保样本数据的范围在 -1 到 1 之间
    max_val = np.max(np.abs(samples))
    if max_val > 0:
        samples /= max_val

    # 计算吸收系数和最大反射次数
    absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)

    # 创建虚拟房间
    room = pra.ShoeBox(room_dim, fs=audio_segment.frame_rate, absorption=absorption, max_order=max_order)

    # 添加音源和麦克风
    mic_locs = np.c_[[room_dim[0] / 2, room_dim[1] / 2, microphone]]  # 假设麦克风在房间中间1.5米高
    room.add_microphone_array(mic_locs)
    room.add_source([room_dim[0] / 3, room_dim[1] / 3, microphone], signal=samples)

    # 模拟混响
    room.image_source_model()
    room.compute_rir()
    reverberant_signal = room.simulate(return_premix=True).flatten()

    # 确保输出样本数据的范围在 -1 到 1 之间
    max_val = np.max(np.abs(reverberant_signal))
    if max_val > 0:
        reverberant_signal /= max_val

    # 将混响后的样本转换回 AudioSegment
    enhanced_audio = AudioSegment(
        (reverberant_signal * np.iinfo(np.int16).max).astype(np.int16).tobytes(),
        frame_rate=audio_segment.frame_rate,
        sample_width=audio_segment.sample_width,
        channels=audio_segment.channels
    )

    enhanced_audio.export(file_path, format='mp3')
    return file_path
