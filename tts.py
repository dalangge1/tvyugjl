import os
import time

import edge_tts
import asyncio
import constant


async def audio_process(_text, voice, output_path=None, rate=constant.var_rate,
                        volume=constant.var_volume, pitch=constant.var_pitch):
    if _text is None and voice is None and rate is None and volume is None:
        return None

    voice_name = constant.voiceMap[voice]

    if rate is not None and rate > 0.0:
        rate_float = "+" + str(rate) + "%"
    else:
        rate_float = "-" + str(rate) + "%"
    if volume is not None and volume > 0.0:
        volume_float = "+" + str(volume) + "%"
    else:
        volume_float = "-" + str(volume) + "%"

    if pitch is not None and pitch > 0.0:
        pitch_float = "+" + str(pitch) + "Hz"
    else:
        pitch_float = "-" + str(pitch) + "Hz"

    timestamp = int(time.time())

    file_path = os.path.join(output_path, 'audio', f"{timestamp}.wav")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    tts_processor = TTSProcessor(_text, voice_name, file_path, rate_float, volume_float, pitch_float)
    # await asyncio.run(tts_processor.text_to_speech())
    await tts_processor.text_to_speech()
    return file_path


class TTSProcessor:
    def __init__(self, text, voice, output, rate, volume, pitch):
        self.text = text
        self.voice = voice
        self.output = output
        self.rate = rate
        self.volume = volume
        self.pitch = pitch

    async def text_to_speech(self):
        tts = edge_tts.Communicate(text=self.text, voice=self.voice, rate=self.rate, volume=self.volume,
                                   pitch=self.pitch)
        await tts.save(self.output)
