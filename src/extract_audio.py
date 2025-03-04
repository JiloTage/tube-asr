import os
import sys
import yt_dlp

def download_audio(youtube_url, output_dir="data/audio"):
    os.makedirs(output_dir, exist_ok=True)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=True)
        video_id = info_dict.get("id", None)
        filename = os.path.join(output_dir, f"{video_id}.wav")
        print(f"Downloaded and converted audio: {filename}")
        return filename

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_audio.py <YouTube_URL>")
        sys.exit(1)
    youtube_url = sys.argv[1]
    download_audio(youtube_url)
