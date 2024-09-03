# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import random
import sys

import gradio as gr
import librosa
import numpy as np
import torch
import torchaudio

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav, logging

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append("{}/third_party/Matcha-TTS".format(ROOT_DIR))


def generate_seed():
    seed = random.randint(1, 100000000)
    return {"__type__": "update", "value": seed}


def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


max_val = 0.8


def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db, frame_length=win_length, hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)
    return speech


inference_mode_list = [
    "指定音色",
    "音色复刻",
    "音色融合",
    "跨语种复刻",
    "自然语言控制",
]
instruct_dict = {
    "指定音色": "1. 选择指定音色\n2. 点击生成音频按钮",
    "音色复刻": "1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 输入prompt文本\n3. 点击生成音频按钮",
    "音色融合": "1. 选择两个指定音色\n2. 设置音色权重\n3. 点击生成音频钮",
    "跨语种复刻": "1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 点击生成音频按钮",
    "自然语言控制": "1. 选择指定音色\n2. 输入instruct文本\n3. 点击生成音频按钮",
}
stream_mode_list = [("否", False), ("是", True)]


def change_instruction(mode_checkbox_group):
    return instruct_dict[mode_checkbox_group]


def generate_audio(
    tts_text,
    mode_checkbox_group,
    sft_dropdown,
    prompt_text,
    prompt_wav_upload,
    prompt_wav_record,
    instruct_text,
    seed,
    stream,
    speed_factor,
    sft_dropdown_2,
    spk1_weight,
    spk2_weight,
):
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None
    # if instruct mode, please make sure that model is iic/CosyVoice-300M-Instruct and not cross_lingual mode
    if mode_checkbox_group in ["自然语言控制"]:
        if cosyvoice.frontend.instruct is False:
            gr.Warning(
                "您正在使用自然语言控制模式, {}模型不支持此模式, 请使用iic/CosyVoice-300M-Instruct模型".format(
                    args.model_dir
                )
            )
            return (target_sr, default_data)
        if instruct_text == "":
            gr.Warning("您正在使用自然语言控制模式, 请输入instruct文本")
            return (target_sr, default_data)
        if prompt_wav is not None or prompt_text != "":
            gr.Info("您正在使用自然语言控制模式, prompt音频/prompt文本会被忽略")
    # if cross_lingual mode, please make sure that model is iic/CosyVoice-300M and tts_text prompt_text are different language
    if mode_checkbox_group in ["跨语种复刻"]:
        if cosyvoice.frontend.instruct is True:
            gr.Warning(
                "您正在使用跨语种复刻模式, {}模型不支持此模式, 请使用iic/CosyVoice-300M模型".format(
                    args.model_dir
                )
            )
            return (target_sr, default_data)
        if instruct_text != "":
            gr.Info("您正在使用跨语种复刻模式, instruct文本会被忽略")
        if prompt_wav is None:
            gr.Warning("您正在使用跨语种复刻模式, 请提供prompt音频")
            return (target_sr, default_data)
        gr.Info("您正在使用跨语种复刻模式, 请确保合成文本和prompt文本为不同语言")
    # if in zero_shot cross_lingual, please make sure that prompt_text and prompt_wav meets requirements
    if mode_checkbox_group in ["音色复刻", "跨语种复刻"]:
        if prompt_wav is None:
            gr.Warning("prompt音频为空，您是否忘记输入prompt音频？")
            return (target_sr, default_data)
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            gr.Warning(
                "prompt音频采样率{}低于{}".format(
                    torchaudio.info(prompt_wav).sample_rate, prompt_sr
                )
            )
            return (target_sr, default_data)
    # sft mode only use sft_dropdown
    if mode_checkbox_group in ["指定音色"]:
        if instruct_text != "" or prompt_wav is not None or prompt_text != "":
            gr.Info(
                "您正在使用指定音色模式，prompt文本/prompt音频/instruct文本会被忽略！"
            )
    # zero_shot mode only use prompt_wav prompt text
    if mode_checkbox_group in ["音色复刻"]:
        if prompt_text == "":
            gr.Warning("prompt文本为空，您是否忘记输入prompt文本？")
            return (target_sr, default_data)
        if instruct_text != "":
            gr.Info("您正在使用音色复刻模式，指定音色/instruct文本会被忽略！")

    if mode_checkbox_group == "指定音色":
        logging.info("get sft inference request")
        set_all_random_seed(seed)
        for i in cosyvoice.inference_sft(tts_text, sft_dropdown, stream=stream):
            yield (target_sr, i["tts_speech"].numpy().flatten())
    elif mode_checkbox_group == "音色复刻":
        logging.info("get zero_shot inference request")
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        for i in cosyvoice.inference_zero_shot(
            tts_text, prompt_text, prompt_speech_16k, stream=stream
        ):
            yield (target_sr, i["tts_speech"].numpy().flatten())
    elif mode_checkbox_group == "跨语种复刻":
        logging.info("get cross_lingual inference request")
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        for i in cosyvoice.inference_cross_lingual(
            tts_text, prompt_speech_16k, stream=stream
        ):
            yield (target_sr, i["tts_speech"].numpy().flatten())
    elif mode_checkbox_group == "音色融合":
        logging.info("get merge inference request")
        set_all_random_seed(seed)
        for i in cosyvoice.inference_merge(
            tts_text,
            sft_dropdown,
            sft_dropdown_2,
            spk1_weight,
            spk2_weight,
            stream=stream,
        ):
            yield (target_sr, i["tts_speech"].numpy().flatten())
    else:
        logging.info("get instruct inference request")
        set_all_random_seed(seed)
        for i in cosyvoice.inference_instruct(
            tts_text, sft_dropdown, instruct_text, stream=stream
        ):
            yield (target_sr, i["tts_speech"].numpy().flatten())


def refresh_voices():
    cosyvoice.reload_voices()
    updated_sft_spk = cosyvoice.list_avaliable_spks()
    return (
        gr.update(choices=updated_sft_spk, value=updated_sft_spk[0]),
        gr.update(choices=updated_sft_spk, value=updated_sft_spk[0]),
        gr.update(choices=updated_sft_spk, value=updated_sft_spk[0]),
    )


def delete_voice(voice_name):
    if voice_name in cosyvoice.list_avaliable_spks():
        os.remove(f"./voices/{voice_name}.pt")
    return refresh_voices()


def main():
    with gr.Blocks() as demo:
        with gr.Row():
            tts_text = gr.Textbox(
                label="输入合成文本",
                lines=1,
                value="我是通义实验室语音团队全新推出的生成式语音大模型，提供舒适自然的语音合成能力。",
                scale=3,
            )
            speed_factor = gr.Slider(
                minimum=0.25,
                maximum=4,
                step=0.05,
                label="语速调节",
                value=1.0,
                interactive=True,
                scale=1,
            )
            with gr.Column(scale=1):
                sft_dropdown = gr.Dropdown(
                    choices=sft_spk, label="选择指定音色", value=sft_spk[0]
                )
                refresh_button = gr.Button("刷新音色列表")
        with gr.Row():
            mode_checkbox_group = gr.Radio(
                choices=inference_mode_list,
                label="选择推理模式",
                value=inference_mode_list[0],
                scale=1,
            )
            instruction_text = gr.Text(
                label="操作步骤", value=instruct_dict[inference_mode_list[0]], scale=0.5
            )
            with gr.Column(scale=1):
                sft_dropdown_2 = gr.Dropdown(
                    choices=sft_spk,
                    label="（融合模式）选择指定音色2",
                    value=sft_spk[0],
                    scale=0.25,
                )
                spk1_weight = gr.Slider(
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    label="音色1权重",
                    value=0.5,
                )
                spk2_weight = gr.Slider(
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    label="音色2权重",
                    value=0.5,
                )
            with gr.Column(scale=2):
                generate_button = gr.Button("生成音频")
                audio_output = gr.Audio(label="合成音频", autoplay=True, streaming=True)
        with gr.Row():
            prompt_wav_upload = gr.Audio(
                sources="upload",
                type="filepath",
                label="选择prompt音频文件，注意采样率不低于16khz",
                scale=1,
            )
            prompt_wav_record = gr.Audio(
                sources="microphone",
                type="filepath",
                label="录制prompt音频文件",
                scale=1,
            )
            with gr.Column(scale=2):
                prompt_text = gr.Textbox(
                    label="输入prompt文本",
                    lines=1,
                    placeholder="请输入prompt文本，需与prompt音频内容一致，暂时不支持自动识别...",
                    value="",
                )
                instruct_text = gr.Textbox(
                    label="输入instruct文本",
                    lines=1,
                    placeholder="请输入instruct文本.",
                    value="",
                )

        with gr.Row():
            with gr.Column(scale=1):
                stream = gr.Radio(
                    choices=stream_mode_list,
                    label="是否流式推理",
                    value=stream_mode_list[0][1],
                )
                with gr.Row():
                    seed = gr.Number(value=0, label="随机推理种子", scale=1)
                    seed_button = gr.Button(value="\U0001f3b2", scale=0.25)

            with gr.Column(scale=1):
                export_button = gr.Button("导出音色")
                export_name = gr.Textbox(label="音色名称", lines=1, value="my_voice")
                export_button.click(
                    fn=cosyvoice.export_voice_embedding,
                    inputs=[export_name],
                    outputs=[],
                )

            with gr.Column(scale=1):
                delete_button = gr.Button("删除音色")
                delete_voice_dropdown = gr.Dropdown(
                    label="音色名称",
                    choices=sft_spk,
                    value=sft_spk[0] if sft_spk else None,
                )
                delete_button.click(
                    fn=delete_voice,
                    inputs=[delete_voice_dropdown],
                    outputs=[sft_dropdown, sft_dropdown_2, delete_voice_dropdown],
                )

        seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(
            generate_audio,
            inputs=[
                tts_text,
                mode_checkbox_group,
                sft_dropdown,
                prompt_text,
                prompt_wav_upload,
                prompt_wav_record,
                instruct_text,
                seed,
                stream,
                speed_factor,
                sft_dropdown_2,
                spk1_weight,
                spk2_weight,
            ],
            outputs=[audio_output],
        )
        mode_checkbox_group.change(
            fn=change_instruction,
            inputs=[mode_checkbox_group],
            outputs=[instruction_text],
        )
        refresh_button.click(
            fn=refresh_voices,
            inputs=[],
            outputs=[sft_dropdown, sft_dropdown_2, delete_voice_dropdown],
        )
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained_models/CosyVoice-300M",
        help="local path or modelscope repo id",
    )
    args = parser.parse_args()
    cosyvoice = CosyVoice(args.model_dir)
    sft_spk = cosyvoice.list_avaliable_spks()
    prompt_sr, target_sr = 16000, 22050
    default_data = np.zeros(target_sr)
    main()
