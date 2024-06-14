import os
import traceback,gradio as gr
import logging
from GPT_SoVITS.tools.i18n.i18n import I18nAuto
i18n = I18nAuto()

logger = logging.getLogger(__name__)
import ffmpeg
import torch
import sys
from GPT_SoVITS.tools.uvr5.mdxnet import MDXNetDereverb
from GPT_SoVITS.tools.uvr5.vr import AudioPre, AudioPreDeEcho

#GPT_SoVITS.tools.uvr5.
weight_uvr5_root = "./GPT_SoVITS/tools/uvr5/uvr5_weights"
uvr5_names = []
for name in os.listdir(weight_uvr5_root):
    if name.endswith(".pth") or "onnx" in name:
        uvr5_names.append(name.replace(".pth", ""))



def uvr(model_name, inp_root, save_root_vocal, paths, save_root_ins, agg, format0):
    pass

