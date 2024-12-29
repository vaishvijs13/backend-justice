import ffmpeg
import os
from pathlib import Path
import tempfile
import whisper
import subprocess
from transformers import pipeline

class TranscribeModel:
    def __init__(self, model_size="small.en"):
        self.model = whisper.load_model(model_size)
        self.nli_model = pipeline("text-classification", model="roberta-large-mnli") # might needa finetune but okay for now

    def extract(self, vid, aud=None):
        """extracts audio from video"""
        if aud is None:
            aud = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
        try:
            ffmpeg.input(vid).output(aud, format='mp3', acodec='mp3').run(overwrite_output=True, quiet=True)
            print(f"Audio extracted to {aud}")
        except ffmpeg.Error as e:
            print(f"Error extracting audio: {e}")
            raise
        return aud

    def transcribe(self, aud):
        """transcribes video using extracted audio"""
        result = self.model.transcribe(aud)
        return result['segments']
    
    def detect_contradictions(self, transcript):
        """detects contradictions stated in videos (transcriptions) or written files"""
        contradictions = []
        num_statements = len(transcript)

        for i in range(num_statements):
            for j in range(i + 1, num_statements):
                result = self.nli_model({
                    "text": transcript[i]["text"],
                    "text_pair": transcript[j]["text"]
                })
                if result[0]["label"] == "CONTRADICTION" and result[0]["score"] > 0.9:
                    contradictions.append({
                        "statement_1": transcript[i],
                        "statement_2": transcript[j],
                        "score": result[0]["score"]
                    })
        return contradictions

    def seg(self, segments):
        """seperates video into segments; each represents a sentence"""
        intervals = []
        curr_int = []

        for index, segment in enumerate(segments):
            curr_int.append(index)
            if segment["text"].strip().endswith((".", "!", "?")):
                intervals.append(curr_int)
                curr_int = []

        if curr_int:
            intervals.append(curr_int)

        return intervals
    
    def extract_clip_with_audio(self, video_path, start, end, output_path):
        command = [
            "ffmpeg",
            "-i", video_path,
            "-ss", str(start),
            "-to", str(end),
            "-c:v", "copy",
            "-c:a", "copy",
            output_path
        ]
        subprocess.run(command, check=True)

    def vid_seg(self, vid, intervals, segments, output_folder="output/intervals/"):
        """creates video clips for each interval (specifically for timestamping and stress detection from facial features)"""
        os.makedirs(output_folder, exist_ok=True)
        video_name = Path(vid).stem
        sentence_segments = []

        for interval in intervals:
            if not interval:
                continue
            start = segments[interval[0]]["start"]
            end = max(start + 1, segments[interval[-1]]["end"]) # at least 1 second for each

            clip_path = os.path.join(output_folder, f"{video_name}_{interval[0]}_{interval[-1]}.mp4")
            self.extract_clip_with_audio(vid, start, end, clip_path)

            sentence_text = " ".join(segments[i]["text"] for i in interval).replace("  ", " ")
            sentence_segments.append({
                "start": start,
                "end": end,
                "text": sentence_text.strip(),
                "video_path": clip_path
            })
            print(f"Video clip: {clip_path}")

        return sentence_segments

    def process_video(self, vid, output_folder="output/intervals/"):
        """entire pipeline"""
        aud = self.extract(vid)  # extracts audio
        segments = self.transcribe(aud)  # transcribes audio
        intervals = self.seg(segments)  # creates segments of sentences
        return self.vid_seg(vid, intervals, segments, output_folder)  # creates clips