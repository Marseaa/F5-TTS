import subprocess
import os

def generate_audio(model):
    try:
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
        command = [
            "f5-tts_infer-cli",
            "--model", model,
            "--ref_audio", ref_audio,
            "--ref_text", ref_text,
            "--gen_text", ref_text
        ]
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout
    except Exception as e:
        return f"Erro ao gerar áudio: {str(e)}"

if __name__ == "__main__":
    model = "F5-TTS"
    output_audio_path = "output_audio/audio_2.wav"
    output_audio = generate_audio(model)
    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
    with open(output_audio_path, 'wb') as f:
        f.write(output_audio)
    print(f"Áudio gerado e salvo em: {output_audio_path}")
