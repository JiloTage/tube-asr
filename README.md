# 日本語株主総会 ASR + Shallow Fusion デモ

## 実行手順
1. 依存ライブラリのインストール:
   ```bash
   pip install yt-dlp pydub transformers pyctcdecode pyyaml pandas scipy torch
   ```

2. YouTube音声の抽出:
   ```bash
   python src/extract_audio.py <YouTube動画URL>
   ```

3. 文字起こしの実行:
   ```bash
   python src/transcribe.py --audio data/audio/<your_audio_file>.wav --config config/asr_config.yaml
   ```

4. 結果は `data/output/transcription.csv` に出力されます。
