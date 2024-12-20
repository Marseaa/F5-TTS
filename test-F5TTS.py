import sys
sys.path.append('/app/F5TTS/src')

import argparse, codecs, re, tomli
from pathlib import Path

import numpy as np
import soundfile as sf
from cached_path import cached_path

from f5_tts.model import DiT, UNetT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
)

import os
import json
from time import time
from appPublic.dictObject import DictObject
from appPublic.zmq_reqrep import ZmqReplier
from appPublic.folderUtils import temp_file
from appPublic.jsonConfig import getConfig

n_mel_channels = 100
hop_length = 256
target_rms = 0.1
nfe_step = 32  # 16, 32
cfg_strength = 2.0
ode_method = "euler"
sway_sampling_coef = -1.0
speed = 1.0

class F5TTS:
    def __init__(self):

        config_file = '/app/conf/config.json'
        if not os.path.exists(config_file):
            print("Criando arquivo de configuração padrão...")
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            with open(config_file, 'w') as f:
                default_config = {
                    "zmq_url": "tcp://*:5555",
                    "device": "cuda",
                    "vocab_file": "/app/F5TTS/src/f5_tts/model/vocab.txt",
                    "modelname": "F5-TTS",
                    "mel_spec_type": "vocos",
                    "voices": {
                        "main": {
                            "ref_audio": "audio_teste.wav",
                            "ref_text": (
                                "Seu amor, DF, Raimundo Alves, com quem eu falo?  Oi, é a enfermeira Letícia, do Trauma.  "
                                "Quero pedir uma remoção.  É o primeiro contato?  É.  É um transporte?  É, transporte para fazer tomografia.  "
                                "Qual o hospital de origem?  Hospital de base.  E o hospital de destino?  H-RAM.  Qual o nome do paciente?  "
                                "José Jorge da Silva.  Qual a idade dele?  61.  Correto.  Enfermeira Letícia, confirmando os dados, o hospital "
                                "de origem é o hospital de base do DF.  Sim.  E o hospital de destino é o H-RAM.  Isso.  Hospital Regional da "
                                "Zona Norte, não é isso?  Sim, é Centro de Trauma do Hospital de Base.  Correto.  O nome do paciente é José "
                                "Jorge da Silva, 61 anos.  Isso.  É um transporte?  Isso.  Aguarde em linha que eu vou vir para o setor responsável.  "
                                "Tá bem, obrigada.  Peço que não desliga. Aguarde em linha.  Tá, obrigada.  Obrigada."
                            )
                        }
                    }
                }
                json.dump(default_config, f, indent=4)
        self.config = getConfig()
        self.zmq_url = self.config.zmq_url
        self.replier = ZmqReplier(self.config.zmq_url, self.generate)
        self.load_model()
        self.setup_voice()

    def run(self):
        print(f'running {self.zmq_url}')
        self.replier._run()
        print('ended ...')

    def load_model(self):
        ckpt_file = ''
        model_cls = DiT
        model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
        if ckpt_file == "":
            repo_name = "F5-TTS"
            exp_name = "F5TTS_Base"
            ckpt_step = 1200000
            ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"))

        self.model, self.vocab, _ = load_model(model_cls, model_cfg, ckpt_file, self.config.vocab_file)
        self.model = self.model.to(self.config.device)
        self.mel_spec = "vocos"
        print("Modelo carregado!")

    def generate(self, d):
        msg = d.decode('utf-8')
        data = DictObject(**json.loads(msg))
        print(data)
        t1 = time()
        for wav in self.inference(data.prompt, stream=data.stream):
            if data.stream:
                d = {
                    "reqid": data.reqid,
                    "b64wave": b64str(wav),
                    "finish": False
                }
                self.replier.send(json.dumps(d))
            else:
                t2 = time()
                d = {
                    "reqid": data.reqid,
                    "audio_file": wav,
                    "time_cost": t2 - t1
                }
                print(f'{d}')
                return json.dumps(d)
        t2 = time()
        d = {
            "reqid": data.reqid,
            "time_cost": t2 - t1,
            "finish": True
        }
        return json.dumps(d)

    def setup_voice(self):
        # Update the text and audio references
        ref_text = (
            "Seu amor, DF, Raimundo Alves, com quem eu falo?  Oi, é a enfermeira Letícia, do Trauma.  "
            "Quero pedir uma remoção.  É o primeiro contato?  É.  É um transporte?  É, transporte para fazer tomografia.  "
            "Qual o hospital de origem?  Hospital de base.  E o hospital de destino?  H-RAM.  Qual o nome do paciente?  "
            "José Jorge da Silva.  Qual a idade dele?  61.  Correto.  Enfermeira Letícia, confirmando os dados, o hospital "
            "de origem é o hospital de base do DF.  Sim.  E o hospital de destino é o H-RAM.  Isso.  Hospital Regional da "
            "Zona Norte, não é isso?  Sim, é Centro de Trauma do Hospital de Base.  Correto.  O nome do paciente é José "
            "Jorge da Silva, 61 anos.  Isso.  É um transporte?  Isso.  Aguarde em linha que eu vou vir para o setor responsável.  "
            "Tá bem, obrigada.  Peço que não desliga. Aguarde em linha.  Tá, obrigada.  Obrigada."
        )
        ref_audio = "audio_teste.wav"

        main_voice = {"ref_audio": ref_audio, "ref_text": ref_text}
        if "voices" not in self.config:
            voices = {"main": main_voice}
        else:
            voices = self.config["voices"]
            voices["main"] = main_voice
        for voice in voices:
            voices[voice]["ref_audio"], voices[voice]["ref_text"] = preprocess_ref_audio_text(
                voices[voice]["ref_audio"], voices[voice]["ref_text"]
            )
            print("Voice:", voice)
            print("Ref_audio:", voices[voice]["ref_audio"])
            print("Ref_text:", voices[voice]["ref_text"])
        self.voices = voices

    def inference(self, prompt):
        text_gen = prompt
        remove_silence = False
        generated_audio_segments = []
        reg1 = r"(?=\[\w+\])"
        chunks = re.split(reg1, text_gen)
        reg2 = r"\[(\w+)\]"
        for text in chunks:
            match = re.match(reg2, text)
            if match:
                voice = match[1]
            else:
                print("No voice tag found, using main.")
                voice = "main"
            if voice not in self.voices:
                print(f"Voice {voice} not found, using main.")
                voice = "main"
            text = re.sub(reg2, "", text)
            gen_text = text.strip()
            ref_audio = self.voices[voice]["ref_audio"]
            ref_text = self.voices[voice]["ref_text"]
            print(f"Voice: {voice}, {self.model}")
            audio, final_sample_rate, spectragram = \
                    infer_process(ref_audio, ref_text, gen_text, self.model)
            generated_audio_segments.append(audio)

        if generated_audio_segments:
            final_wave = np.concatenate(generated_audio_segments)
            fn = temp_file(suffix='.wav')
            with open(fn, "wb") as f:
                sf.write(f.name, final_wave, final_sample_rate)
                if remove_silence:
                    remove_silence_for_generated_wav(f.name)
            return fn       

if __name__ == '__main__':
    tts = F5TTS()
    while True:
        print('prompt:')
        p = input()
        if p != '':
            t1 = time()
            f = tts.inference(p)
            t2 = time()
            print(f'{f}, cost {t2-t1} seconds')

