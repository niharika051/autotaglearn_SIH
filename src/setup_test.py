import fitz  # PyMuPDF
import whisper
import os
from keyword_extractor import extract_keywords  # ✅ Import KeyBERT function
from flowchart_generator import generate_concept_flowchart  # ✅ Import flowchart function

# ===== PDF extraction test =====
pdf_path = "data/pdf/sample.pdf"  # Put a small test PDF here
pdf_keywords = []
if os.path.exists(pdf_path):
    doc = fitz.open(pdf_path)
    pdf_text = ""
    for page in doc:
        pdf_text += page.get_text()
    print("✅ PDF Text Extracted (first 300 chars):")
    print(pdf_text[:300])

    # 🔹 Extract keywords from PDF
    print("\n🔹 PDF Keywords:")
    pdf_keywords = extract_keywords(pdf_text, top_n=10)
    print(pdf_keywords)

    # 🔹 Generate PDF Flowchart
    if pdf_keywords:
        generate_concept_flowchart(pdf_keywords, "pdf_concept_flowchart.png")
else:
    print("⚠ No PDF found at", pdf_path)

# ===== Whisper transcription test =====
video_path = "data/video/sample.mp4"  # Put a short test video (10–20s) here
video_keywords = []
if os.path.exists(video_path):
    model = whisper.load_model("tiny")  # small for speed
    result = model.transcribe(video_path)
    print("\n✅ Video Transcription (first 300 chars):")
    print(result['text'][:300])

    # 🔹 Extract keywords from Video transcription
    print("\n🔹 Video Keywords:")
    video_keywords = extract_keywords(result['text'], top_n=10)
    print(video_keywords)

    # 🔹 Generate Video Flowchart
    if video_keywords:
        generate_concept_flowchart(video_keywords, "video_concept_flowchart.png")
else:
    print("⚠ No video found at", video_path)
