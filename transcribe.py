import ffmpeg
import os
from pathlib import Path
import tempfile
import whisper
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

class TranscribeModel:
    def __init__(self, model_size="small.en"):
        self.model = whisper.load_model(model_size)

    def extract(self, vid, aud=None):
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
        result = self.model.transcribe(aud)
        return result['segments']

    def seg(self, segments):
        intervals = [[]]
        for index, segment in enumerate(segments):
            intervals[-1].append(index)
            if segment["text"].strip().endswith((".", "!", "?")):
                intervals.append([])
        return intervals

    def vid_seg(self, vid, intervals, segments, output_folder="output/intervals/"):
        os.makedirs(output_folder, exist_ok=True)
        video_name = Path(vid).stem
        sentence_segments = []

        for interval in intervals:
            if not interval:
                continue
            start = segments[interval[0]]["start"]
            end = segments[interval[-1]]["end"]

            clip_path = os.path.join(output_folder, f"{video_name}_{interval[0]}_{interval[-1]}.mp4")
            ffmpeg_extract_subclip(vid, start, end, targetname=clip_path)

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
        aud = self.extract(vid)  # extracts audio
        segments = self.transcribe(aud)  # transcribes audio
        intervals = self.seg(segments)  # creates segments of sentences
        return self.vid_seg(vid, intervals, segments, output_folder)  # creates clips