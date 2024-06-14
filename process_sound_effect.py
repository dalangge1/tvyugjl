import os

from pydub import AudioSegment

sound_path = os.path.join(os.path.dirname(__file__), 'asset', 'sound_effects')


def process_sound_effect(output_path, sounds):
    audio = AudioSegment.from_file(output_path)
    for object in sounds:
        # 使用 split() 方法进行分割/
        object = object.replace('，', ',').strip()
        split_str = object.split(':')
        # 获取分割后的子字符串
        name = split_str[1].split(',')[0]
        number = int(split_str[1].split(',')[1])
        sound_effect = AudioSegment.from_file(os.path.join(sound_path, name + ".mp3"))
        #设置音效声音
        desired_volume = -5
        sound_effect_volume = sound_effect.set_channels(2)
        sound_effect_volume = sound_effect_volume + desired_volume
        if number == -1:
            audio = sound_effect_volume + audio
        elif number == 100:
            audio = audio + sound_effect_volume
        else:
            audio = audio.overlay(sound_effect_volume, position=number * 1000)
    audio.export(output_path, format="mp3")
    return output_path