import torchaudio
from f5_tts.infer.utils_infer import load_model, infer_process
from f5_tts.model import DiT


ckpt_path = "/app/F5-TTS/F5TTS_Base/model_1200000.pt"  


model_cfg = {
    'dim': 512,                    
    'depth': 6,                   
    'heads': 8,                   
    'dim_head': 64,               
    'dropout': 0.1,              
}

print("Carregando o modelo...")
model = load_model(
    DiT,            
    model_cfg,       
    ckpt_path        
)
print("Modelo carregado")

def sintetizar_audio(texto, ref_audio_path):
    print("Realizando inferência...")
    final_wave, final_sample_rate, _ = infer_process(ref_audio_path, None, texto, model)
    print("Inferência concluída!")
    return final_wave.squeeze().numpy(), final_sample_rate

def salvar_audio(waveform, sample_rate):
    import numpy as np
    import soundfile as sf
    audio_output_path = "audio_sintetizado.wav"
    sf.write(audio_output_path, waveform, sample_rate)
    print(f"Áudio salvo em {audio_output_path}")
    return audio_output_path

def processar(texto, audio_path):
    waveform, sample_rate = sintetizar_audio(texto, audio_path)
    audio_output_path = salvar_audio(waveform, sample_rate)
    return audio_output_path

text = (
    "Seu amor, DF, Raimundo Alves, com quem eu falo?  Oi, é a enfermeira Letícia, do Trauma.  "
    "Quero pedir uma remoção.  É o primeiro contato?  É.  É um transporte?  É, transporte para fazer tomografia.  "
    "Qual o hospital de origem?  Hospital de base.  E o hospital de destino?  H-RAM.  Qual o nome do paciente?  "
    "José Jorge da Silva.  Qual a idade dele?  61.  Correto.  Enfermeira Letícia, confirmando os dados, o hospital "
    "de origem é o hospital de base do DF.  Sim.  E o hospital de destino é o H-RAM.  Isso.  Hospital Regional da "
    "Zona Norte, não é isso?  Sim, é Centro de Trauma do Hospital de Base.  Correto.  O nome do paciente é José "
    "Jorge da Silva, 61 anos.  Isso.  É um transporte?  Isso.  Aguarde em linha que eu vou vir para o setor responsável.  "
    "Tá bem, obrigada.  Peço que não desliga. Aguarde em linha.  Tá, obrigada.  Obrigada."
)

interface = gr.Interface(
    fn=processar,
    inputs=[
        gr.Textbox(lines=5, label="Texto para Síntese", value=text),
        gr.Audio(source="upload", type="filepath", label="Áudio de Referência")
    ],
    outputs=gr.Audio(type="filepath", label="Áudio Sintetizado"),
    title="F5-TTS - Síntese de Fala"
)

if __name__ == "__main__":
    interface.launch()
