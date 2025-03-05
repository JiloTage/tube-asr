# src/model_loader.py
import yaml
import torch
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration, Wav2Vec2ForCTC, Wav2Vec2Processor

def load_config(config_path="config/asr_config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_asr_model(config):
    model_type = config["asr_model"]["type"].lower()
    use_gpu = config["asr_model"].get("use_gpu", False) and torch.cuda.is_available()
    device = 0 if use_gpu else -1

    if model_type == "whisper":
        # Whisperモデルのロード（pipelineで簡易に）
        model_name = config["asr_model"]["pretrained"]
        asr_pipeline = pipeline("automatic-speech-recognition", model=model_name, device=device)
        return asr_pipeline
    elif model_type == "wav2vec2":
        model_name = config["asr_model"]["pretrained"]
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
        model.to("cuda" if use_gpu else "cpu")
        return {"processor": processor, "model": model}
    else:
        raise ValueError(f"Unsupported ASR model type: {model_type}")

def load_language_model(config):
    lm_config = config.get("language_model", {})
    if lm_config.get("enable", False) and lm_config.get("type", "").lower() == "kenlm":
        lm_path = lm_config.get("model_path", None)
        if lm_path is None:
            raise ValueError("LM enabled but no model_path provided in config")
        # ここでは pyctcdecode 用のKenLMモデルのパスを返すだけ
        return lm_path
    return None
