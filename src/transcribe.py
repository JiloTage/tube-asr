# src/transcribe.py
import os
import argparse
import pandas as pd
import torch
import numpy as np
from src.model_loader import load_config, load_asr_model, load_language_model
from src.utils import split_audio, save_audio_chunk
from src.diarization import perform_diarization

# Whisper用のライブラリはpipelineで実装済み
# wav2vec2の場合、Shallow Fusion対応のデコーディングを行います
# ここでは簡単な分岐で実装例を示します
def transcribe_audio(audio_path, config):
    asr_config = config["asr_model"]
    decoding_config = config["decoding"]
    output_config = config["output"]
    
    # 話者ダイアリゼーション（stub実装）
    diarization_segments = perform_diarization(audio_path)
    
    # 音声チャンクに分割
    chunks = split_audio(
        audio_path,
        chunk_duration=decoding_config.get("chunk_duration", 30.0),
        stride_overlap=decoding_config.get("stride_overlap", 2.0)
    )
    
    # モデルロード
    asr_model_obj = load_asr_model(config)
    model_type = asr_config["type"].lower()
    
    results = []
    
    # チャンク毎に認識（タイムスタンプはチャンクの開始・終了を使用）
    for start, end, chunk in chunks:
        # 一時ファイルに保存してモデルに読み込ませる（簡易実装）
        temp_audio_path = "temp_chunk.wav"
        save_audio_chunk(chunk, temp_audio_path)
        
        if model_type == "whisper":
            # Whisperの場合、pipelineで直接処理
            result = asr_model_obj(temp_audio_path)
            transcript = result.get("text", "")
        elif model_type == "wav2vec2":
            # wav2vec2の場合、手動で処理（Shallow Fusionの例も含む）
            from scipy.io import wavfile
            rate, audio_input = wavfile.read(temp_audio_path)
            # 正規化
            if audio_input.dtype != np.float32:
                audio_input = audio_input.astype(np.float32) / 32768.0
            # 推論
            processor = asr_model_obj["processor"]
            model = asr_model_obj["model"]
            input_values = processor(audio_input, sampling_rate=rate, return_tensors="pt").input_values
            if torch.cuda.is_available():
                input_values = input_values.to("cuda")
                model.to("cuda")
            with torch.no_grad():
                logits = model(input_values).logits.cpu().numpy()[0]
            # Shallow Fusion: LMが有効なら pyctcdecode を利用
            lm_enabled = config["language_model"].get("enable", False)
            if lm_enabled:
                from src.decoder import build_decoder, decode_logits
                lm_path = load_language_model(config)
                decoder = build_decoder(processor, lm_path, config["language_model"]["lm_weight"],
                                          asr_config.get("beam_width", 5))
                transcript = decode_logits(logits, decoder)
            else:
                # Greedy decoding
                predicted_ids = np.argmax(logits, axis=-1)
                transcript = processor.tokenizer.decode(predicted_ids)
        else:
            transcript = "Unsupported model type"
        
        # 本来はチャンク内で複数発話に分割する必要がありますが、ここでは簡略化のためチャンク単位を1発話とする
        results.append({
            "StartTime": f"{start:.2f}",
            "EndTime": f"{end:.2f}",
            "Transcript": transcript.strip(),
            "Speaker": "Speaker_1"  # ダイアリゼーション結果と突き合わせる場合は各チャンクの発話ごとに適用
        })
        os.remove(temp_audio_path)
    return results

def save_results_to_csv(results, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame(results, columns=["Speaker", "StartTime", "EndTime", "Transcript"])
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Transcription results saved to {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="ASR + Shallow Fusion Transcription Demo")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio (wav)")
    parser.add_argument("--config", type=str, default="config/asr_config.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    results = transcribe_audio(args.audio, config)
    save_results_to_csv(results, config["output"]["csv_path"])

if __name__ == "__main__":
    main()
