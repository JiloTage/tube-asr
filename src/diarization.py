# src/diarization.py
def perform_diarization(audio_path):
    """
    ここでは話者ダイアリゼーションのstub実装を行います。
    実際には pyannote.audio などを利用して各発話セグメントに話者ラベルを付与する処理を実装してください。
    今回は単一話者 "Speaker_1" として全体を返します。
    
    戻り値は、セグメントごとのリスト（開始時刻, 終了時刻, speaker_label）の形式とします。
    例: [(0.0, 30.0, "Speaker_1"), (30.0, 60.0, "Speaker_1"), ...]
    """
    # stub: 全体を1チャンクと仮定
    from pydub import AudioSegment
    audio = AudioSegment.from_file(audio_path)
    duration_sec = len(audio) / 1000.0
    return [(0.0, duration_sec, "Speaker_1")]
