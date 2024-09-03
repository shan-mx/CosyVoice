# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
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
import os
import time
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download
import torch
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.model import CosyVoiceModel
from cosyvoice.utils.file_utils import logging


class CosyVoice:

    def __init__(self, model_dir, load_jit=True, load_onnx=False):
        instruct = True if '-Instruct' in model_dir else False
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        with open('{}/cosyvoice.yaml'.format(model_dir), 'r') as f:
            configs = load_hyperpyyaml(f)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v1.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          instruct,
                                          configs['allowed_special'])
        self.model = CosyVoiceModel(configs['llm'], configs['flow'], configs['hift'])
        self.model.load('{}/llm.pt'.format(model_dir),
                        '{}/flow.pt'.format(model_dir),
                        '{}/hift.pt'.format(model_dir))
        if load_jit:
            self.model.load_jit('{}/llm.text_encoder.fp16.zip'.format(model_dir),
                                '{}/llm.llm.fp16.zip'.format(model_dir),
                                '{}/flow.encoder.fp32.zip'.format(model_dir))
        if load_onnx:
            self.model.load_onnx('{}/flow.decoder.estimator.fp32.onnx'.format(model_dir))
        del configs
        self.last_voice_embedding = None
        
    def export_voice_embedding(self, voice_name):
        if self.last_voice_embedding is not None:
            torch.save({"embedding": self.last_voice_embedding}, f"./voices/{voice_name}.pt")
        
    def reload_voices(self):
        self.frontend.load_voices()

    def list_avaliable_spks(self):
        spks = list(self.frontend.spk2info.keys())
        return spks

    def inference_sft(self, tts_text, spk_id, stream=False):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True)):
            model_input = self.frontend.frontend_sft(i, spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.inference(**model_input, stream=stream):
                speech_len = model_output['tts_speech'].shape[1] / 22050
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()
            self.last_voice_embedding = model_input["flow_embedding"]

    def inference_merge(self, tts_text, spk1_id, spk2_id, spk1_weight=0.5, spk2_weight=0.5, stream=False):
        for i in self.frontend.text_normalize(tts_text, split=True):
            model_input = self.frontend.frontend_merge(i, spk1_id, spk2_id, spk1_weight, spk2_weight)
            start_time = time.time()
            logging.info("synthesis text {}".format(i))
            for model_output in self.model.inference(**model_input, stream=stream):
                speech_len = model_output["tts_speech"].shape[1] / 22050
                logging.info(
                    "yield speech len {}, rtf {}".format(
                        speech_len, (time.time() - start_time) / speech_len
                    )
                )
                yield model_output
                start_time = time.time()
            self.last_voice_embedding = model_input["flow_embedding"]

    def inference_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, stream=False):
        prompt_text = self.frontend.text_normalize(prompt_text, split=False)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True)):
            model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_speech_16k)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.inference(**model_input, stream=stream):
                speech_len = model_output['tts_speech'].shape[1] / 22050
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()
            self.last_voice_embedding = model_input["flow_embedding"]
            
    def inference_cross_lingual(self, tts_text, prompt_speech_16k, stream=False):
        if self.frontend.instruct is True:
            raise ValueError('{} do not support cross_lingual inference'.format(self.model_dir))
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True)):
            model_input = self.frontend.frontend_cross_lingual(i, prompt_speech_16k)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.inference(**model_input, stream=stream):
                speech_len = model_output['tts_speech'].shape[1] / 22050
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()
            self.last_voice_embedding = model_input["flow_embedding"]

    def inference_instruct(self, tts_text, spk_id, instruct_text, stream=False):
        if self.frontend.instruct is False:
            raise ValueError('{} do not support instruct inference'.format(self.model_dir))
        instruct_text = self.frontend.text_normalize(instruct_text, split=False)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True)):
            model_input = self.frontend.frontend_instruct(i, spk_id, instruct_text)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.inference(**model_input, stream=stream):
                speech_len = model_output['tts_speech'].shape[1] / 22050
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()
            self.last_voice_embedding = model_input["flow_embedding"]
