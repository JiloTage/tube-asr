import os
from pydub import AudioSegment

def split_audio(audio_path, chunk_duration=30.0, stride_overlap=2.0):
    audio = AudioSegment.from_file(audio_path)
    total_duration = len(audio) / 1000.0
    chunks = []
    start = 0.0
    while start < total_duration:
        end = min(start + chunk_duration, total_duration)
        chunk = audio[start*1000:end*1000]
        chunks.append((start, end, chunk))
        start = end - stride_overlap
    return chunks

def save_audio_chunk(chunk_audio, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    chunk_audio.export(output_path, format="wav")
