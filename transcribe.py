import whisper
import os

def transcribe_latest_video():
    upload_folder = 'uploads'
    video_files = [f for f in os.listdir(upload_folder) if f.endswith(('.mp4', '.wav'))]
    if not video_files:
        print("No files in uploads.")
        return

    latest_file = max(video_files, key=lambda f: os.path.getmtime(os.path.join(upload_folder, f)))
    filepath = os.path.join(upload_folder, latest_file)
    print(f"Transcribing: {filepath}")

    model = whisper.load_model("base")
    result = model.transcribe(filepath)
    return result['text']

if __name__ == "__main__":
    transcription = transcribe_latest_video()
    if transcription:
        print("Transcription:")
        print(transcription)