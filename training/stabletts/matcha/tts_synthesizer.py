import os
import warnings
from pathlib import Path

import soundfile as sf
import torch

from matcha.hifigan.config import v1
from matcha.hifigan.denoiser import Denoiser
from matcha.hifigan.env import AttrDict
from matcha.hifigan.models import Generator as HiFiGAN
from matcha.models.matcha_tts import MatchaTTS
from matcha.text import sequence_to_text, text_to_sequence
from matcha.utils.utils import intersperse, intersperse_bert
from vocos import Vocos


def process_text(text: str, device: torch.device):
    x, bert = text_to_sequence(text, [])
    x = intersperse(x, 0)
    bert = intersperse_bert(bert)
    x = torch.tensor(x, dtype=torch.long, device=device).unsqueeze(0)
    bert = torch.stack(bert, dim=0).T.unsqueeze(0).to(device)
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
    return {"x": x, "x_lengths": x_lengths, "bert": bert}


def load_hifigan(checkpoint_path, device):
    h = AttrDict(v1)
    hifigan = HiFiGAN(h).to(device)
    hifigan.load_state_dict(torch.load(checkpoint_path, map_location=device)["generator"])
    _ = hifigan.eval()
    hifigan.remove_weight_norm()
    return hifigan


def load_vocos(device):
    vocos = Vocos.from_hparams("checkpoints/vocos/config.yaml")
    state_dict = torch.load("checkpoints/vocos/vocos_ru.ckpt", map_location="cpu")
    vocos.load_state_dict(state_dict['state_dict'], strict=False)
    vocos.eval()
    vocos = vocos.to(device)
    return vocos


def load_vocoder(vocoder_name, checkpoint_path, device):
    vocoder = None
    if vocoder_name in ("hifigan_T2_v1", "hifigan_univ_v1"):
        vocoder = load_hifigan(checkpoint_path, device)
    elif vocoder_name == "vocos":
        vocoder = load_vocos(device)
    else:
        raise NotImplementedError(f"Vocoder {vocoder_name} not implemented")

    denoiser = None if vocoder_name == "vocos" else Denoiser(vocoder, mode="zeros")
    return vocoder, denoiser


def load_matcha(model_name, checkpoint_path, device):
    model = MatchaTTS.load_from_checkpoint(checkpoint_path, map_location=device)
    _ = model.eval()
    return model


def to_waveform(mel, vocoder, denoiser=None):
    audio = vocoder.decode(mel).clamp(-1, 1)
    return audio.cpu().squeeze()


def save_audio(waveform, output_path, sample_rate=22050):
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    sf.write(output_path, waveform, sample_rate, "PCM_16")
    return output_path.resolve()


def get_device(cpu=False):
    if torch.cuda.is_available() and not cpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def synthesize_tts(
    text_input=None,
    file_input=None,
    speaker_id=None,
    output_path=None,
    model_name="matcha_ru",
    checkpoint_path=None,
    vocoder_name="vocos",
    vocoder_checkpoint_path=None,
    temperature=0.8,
    speaking_rate=1.0,
    steps=10,
    cpu=False,
    save_prefix=None,
):
    # Validate inputs
    if text_input is None and file_input is None:
        raise ValueError("Either text_input or file_input must be provided")
    if output_path is None:
        raise ValueError("output_path must be provided")
    
    # Get text input
    if file_input is not None:
        with open(file_input, encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        
        saved_paths = []
        for i, line in enumerate(lines, 1):
            text_input = line
            device = get_device(cpu)
            model = load_matcha(model_name, checkpoint_path, device)
            vocoder, denoiser = load_vocoder(vocoder_name, vocoder_checkpoint_path, device)
            
            text_processed = process_text(text_input, device)
            spk = torch.tensor([speaker_id], device=device, dtype=torch.long) if speaker_id is not None else None
            
            output = model.synthesise(
                text_processed["x"],
                text_processed["x_lengths"],
                n_timesteps=steps,
                temperature=temperature,
                spks=spk,
                bert=text_processed["bert"],
                length_scale=speaking_rate,
            )
            
            output["waveform"] = to_waveform(output["mel"], vocoder, denoiser)
            
            filename = f"{save_prefix}_{i}.wav" if save_prefix else f"line_{i}.wav"
            line_output_path = str(Path(output_path).joinpath(filename))
            saved_path = save_audio(output["waveform"], line_output_path)
            saved_paths.append(saved_path)
        
        return saved_paths
    
    # Get device
    device = get_device(cpu)
    
    # Load models
    model = load_matcha(model_name, checkpoint_path, device)
    vocoder, denoiser = load_vocoder(vocoder_name, vocoder_checkpoint_path, device)
    
    # Process text
    text_processed = process_text(text_input, device)
    spk = torch.tensor([speaker_id], device=device, dtype=torch.long) if speaker_id is not None else None
    
    # Synthesize
    output = model.synthesise(
        text_processed["x"],
        text_processed["x_lengths"],
        n_timesteps=steps,
        temperature=temperature,
        spks=spk,
        bert=text_processed["bert"],
        length_scale=speaking_rate,
    )
    
    # Generate waveform
    output["waveform"] = to_waveform(output["mel"], vocoder, denoiser)
    
    # Save output
    saved_path = save_audio(output["waveform"], output_path)
    return saved_path


if __name__ == "__main__":
    # Example usage
    # output_file = synthesize_tts(
    #     text_input="Как насчёт озвучить это предложение?",
    #     speaker_id=2,
    #     output_path="output1.wav",
    #     model_name="matcha_ru",
    #     checkpoint_path="./checkpoints/vosk_tts_ru_0.8.ckpt",
    #     vocoder_name="vocos",
    #     temperature=0.8,
    #     speaking_rate=1.0,
    #     steps=10,
    #     cpu=False,
    # )
    # print(f"Audio saved to: {output_file}")

    # Example usage with file input
    output_files = synthesize_tts(
        file_input="../../../output.txt",
        output_path="output_dir",
        save_prefix="output",
        speaker_id=2,
        model_name="matcha_ru",
        checkpoint_path="./checkpoints/vosk_tts_ru_0.8.ckpt",
        vocoder_name="vocos",
        temperature=0.8,
        speaking_rate=1.0,
        steps=10,
        cpu=False,
    )
    print(f"Audios saved to: {output_files}")

