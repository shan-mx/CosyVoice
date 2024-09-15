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
import argparse
import io
import logging
import os
import sys

from pydantic import BaseModel
from pydub import AudioSegment

logging.getLogger('matplotlib').setLevel(logging.WARNING)
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav

app = FastAPI()
# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

cosyvoice = CosyVoice(os.getenv("MODEL_DIR"))

def generate_data(model_output):
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio


@app.get("/inference_sft")
async def inference_sft(tts_text: str = Form(), spk_id: str = Form()):
    model_output = cosyvoice.inference_sft(tts_text, spk_id)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_zero_shot")
async def inference_zero_shot(tts_text: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_cross_lingual")
async def inference_cross_lingual(tts_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct")
async def inference_instruct(tts_text: str = Form(), spk_id: str = Form(), instruct_text: str = Form()):
    model_output = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text)
    return StreamingResponse(generate_data(model_output))


class InstructInferenceRequest(BaseModel):
    tts_text: str
    spk_id: str
    instruct_text: str


@app.post("/inference_instruct_mp3")
async def inference_instruct_mp3(request: InstructInferenceRequest):
    model_output = cosyvoice.inference_instruct(request.tts_text, request.spk_id, request.instruct_text)
    
    audio_data = np.concatenate([i['tts_speech'].numpy() for i in model_output])
    
    # Normalize audio data to [-1, 1] range
    audio_data = audio_data / np.max(np.abs(audio_data))

    # Convert to 16-bit PCM
    audio_data = (audio_data * 32767).astype(np.int16)

    # Create AudioSegment from raw PCM data
    audio_segment = AudioSegment(
        audio_data.tobytes(),
        frame_rate=22050,
        sample_width=2,  # 16-bit
        channels=1,
    )

    buffer = io.BytesIO()
    audio_segment.export(buffer, format="mp3")
    
    return Response(content=buffer.getvalue(), media_type="audio/mpeg")


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=50000)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
