import fitz  # PyMuPDF
import whisper
import os
import csv
from keyword_extractor import extract_keywords

# Paths
pdf_dir = "data/pdf"
video_dir = "data/video"
output_dir = "outputs"

# Load Whisper model once
whisper_model = whisper.load_model("tiny")

# --- Process PDFs ---
for pdf_file in os.listdir(pdf_dir):
    if pdf_file.lower().endswith(".pdf"):
        pdf_path = os.path.join(pdf_dir, pdf_file)
        print(f"\n📄 Processing PDF: {pdf_file}")
        
        # Extract text
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()

        # Save text to file
        text_output_path = os.path.join(output_dir, pdf_file.replace(".pdf", "_text.txt"))
        with open(text_output_path, "w", encoding="utf-8") as f:
            f.write(text)

        # Extract keywords
        keywords = extract_keywords(text, top_n=10)

        # Save keywords to CSV
        csv_output_path = os.path.join(output_dir, pdf_file.replace(".pdf", "_keywords.csv"))
        with open(csv_output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Keyword"])
            for kw in keywords:
                writer.writerow([kw])

        print(f"✅ PDF done — text saved to {text_output_path}, keywords to {csv_output_path}")

# --- Process Videos ---
for video_file in os.listdir(video_dir):
    if video_file.lower().endswith(".mp4"):
        video_path = os.path.join(video_dir, video_file)
        print(f"\n🎥 Processing Video: {video_file}")
        
        # Transcribe
        result = whisper_model.transcribe(video_path)
        transcript = result['text']

        # Save transcript
        text_output_path = os.path.join(output_dir, video_file.replace(".mp4", "_transcript.txt"))
        with open(text_output_path, "w", encoding="utf-8") as f:
            f.write(transcript)

        # Extract keywords
        keywords = extract_keywords(transcript, top_n=10)

        # Save keywords to CSV
        csv_output_path = os.path.join(output_dir, video_file.replace(".mp4", "_keywords.csv"))
        with open(csv_output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Keyword"])
            for kw in keywords:
                writer.writerow([kw])

        print(f"✅ Video done — transcript saved to {text_output_path}, keywords to {csv_output_path}")

print("\n🎯 All files processed! Check the output/ folder.")
