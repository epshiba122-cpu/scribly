import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from flask import Flask, request, jsonify, render_template, send_file
from lecture_history import init_history_db, save_lecture, get_all_lectures, find_related_lectures
import whisper
from transformers import pipeline
from deep_translator import GoogleTranslator
from fpdf import FPDF
import re
import time
import subprocess
import shutil
import tempfile

app = Flask(__name__)

init_history_db()

def load_whisper_with_retry(model_name="small", max_retries=5):
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Loading Whisper model... (attempt {attempt})")
            model = whisper.load_model(model_name)
            print("Whisper model loaded successfully!")
            return model
        except RuntimeError as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                print("Retrying in 3 seconds...")
                time.sleep(3)
            else:
                raise Exception("Failed to load Whisper model after multiple attempts. Check your internet connection.")

whisper_model = load_whisper_with_retry("small")

summarizer = None

def get_summarizer():
    global summarizer
    if summarizer is None:
        print("Loading Summarizer model...")
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    return summarizer

print("Models loaded! Server ready.")

FFMPEG_PATH = shutil.which("ffmpeg")
print(f"FFmpeg found at: {FFMPEG_PATH}")

def clean_audio(input_path):
    if not FFMPEG_PATH:
        print("WARNING: ffmpeg not found on PATH. Skipping noise cleanup.")
        return input_path

    cleaned_path = input_path + "_cleaned.wav"

    cmd = [
        FFMPEG_PATH, "-y", "-i", input_path,
        "-af", "highpass=f=100,lowpass=f=7000,afftdn=nf=-35:nr=20,dynaudnorm=f=150:g=15",
        "-ar", "16000", "-ac", "1",
        cleaned_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print("FFmpeg noise cleanup FAILED. Error output:")
            print(result.stderr[-1500:])
            return input_path
        print("Audio cleaned successfully (noise reduced).")
        return cleaned_path
    except Exception as e:
        print(f"Noise reduction exception: {e}")
        return input_path

def chunk_text(text, max_words=600):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i+max_words]))
    return chunks

def summarize_long_text(text):
    chunks = chunk_text(text, max_words=600)
    valid_chunks = [c for c in chunks if len(c.strip()) >= 20]
    if not valid_chunks:
        return ""

    summ = get_summarizer()
    summaries = []
    for chunk in valid_chunks:
        word_count = len(chunk.split())
        max_len = min(120, max(30, int(word_count * 0.5)))
        min_len = max(15, int(max_len * 0.4))
        result = summ(chunk, max_length=max_len, min_length=min_len, do_sample=False, truncation=True)
        summaries.append(result[0]['summary_text'])
    return " ".join(summaries)

def extract_key_points(full_text, summary_text):
    transcript_sentences = re.split(r'(?<=[.!?]) +', full_text.strip())
    transcript_sentences = [s.strip() for s in transcript_sentences if len(s.strip()) > 15]

    summary_sentences = re.split(r'(?<=[.!?]) +', summary_text.strip())
    summary_sentences = [s.strip() for s in summary_sentences if s.strip()]

    if len(summary_sentences) == 0:
        return "INTRODUCTION\nNo content available."

    intro = summary_sentences[0]
    conclusion = summary_sentences[-1] if len(summary_sentences) > 1 else summary_sentences[0]

    key_points = []
    for s in summary_sentences[1:-1]:
        if s not in key_points:
            key_points.append(s)

    for s in transcript_sentences:
        if len(key_points) >= 6:
            break
        if s != intro and s != conclusion and s not in key_points:
            key_points.append(s)

    if len(key_points) == 0:
        key_points = [summary_sentences[0]]

    notes = "INTRODUCTION\n" + intro + "\n\n"
    notes += "KEY POINTS\n"
    for s in key_points:
        notes += "- " + s + "\n"
    notes += "\nCONCLUSION\n" + conclusion

    return notes

def safe_translate(text, target_lang):
    try:
        if len(text) > 4500:
            parts = [text[i:i+4500] for i in range(0, len(text), 4500)]
            translated_parts = [GoogleTranslator(source='en', target=target_lang).translate(p) for p in parts]
            return " ".join(translated_parts)
        else:
            return GoogleTranslator(source='en', target=target_lang).translate(text)
    except Exception as e:
        print(f"TRANSLATION ERROR: {e}")
        return text + "\n\n[Note: Translation failed, showing English version]"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_audio():
    audio_file = request.files['audio']
    target_language = request.form.get('language', 'en')
    spoken_language = request.form.get('spoken_language', 'auto')
    filepath = os.path.join('uploads', audio_file.filename)
    audio_file.save(filepath)

    print(f"\n--- New request ---")
    print(f"Spoken language: {spoken_language}, Target language: {target_language}")

    print("Cleaning audio (removing background noise)...")
    cleaned_filepath = clean_audio(filepath)

    print("Transcribing...")
    if spoken_language == 'auto':
        translation_result = whisper_model.transcribe(cleaned_filepath, task='translate', word_timestamps=True)
    else:
        translation_result = whisper_model.transcribe(cleaned_filepath, task='translate', language=spoken_language, word_timestamps=True)
    english_text = translation_result['text']
    detected_lang = translation_result.get('language', 'unknown')

    corrections = {
        "Heart-efficient intelligence": "Artificial intelligence",
        "heart-efficient intelligence": "artificial intelligence",
        "Artifical intelligence": "Artificial intelligence",
    }
    for wrong, correct in corrections.items():
        english_text = english_text.replace(wrong, correct)

    word_timings = []
    for segment in translation_result.get('segments', []):
        for word_info in segment.get('words', []):
            word_timings.append({
                'word': word_info['word'],
                'start': word_info['start'],
                'end': word_info['end']
            })
    print(f"Transcript length: {len(english_text.split())} words")

    print("Summarizing...")
    summary_text = summarize_long_text(english_text)
    formatted_notes = extract_key_points(english_text, summary_text)

    if target_language != 'en':
        print(f"Translating to {target_language}...")
        final_transcript = safe_translate(english_text, target_language)
        final_summary = safe_translate(formatted_notes, target_language)
    else:
        final_transcript = english_text
        final_summary = formatted_notes

    lecture_id, lecture_title = save_lecture(english_text, formatted_notes)
    related = find_related_lectures(english_text, lecture_id)

    print("Done!\n")

    return jsonify({
        'transcript': final_transcript,
        'summary': final_summary,
        'detected_language': detected_lang,
        'word_timings': word_timings if target_language == 'en' else [],
        'lecture_title': lecture_title,
        'related_lectures': related
    })

@app.route('/history')
def history():
    return jsonify(get_all_lectures())

@app.route('/download-pdf', methods=['POST'])
def download_pdf():
    data = request.get_json()
    transcript = data.get('transcript', '')
    summary = data.get('summary', '')

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Lecture Notes", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Transcript", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 7, transcript.encode('latin-1', 'replace').decode('latin-1'))
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Summary", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 7, summary.encode('latin-1', 'replace').decode('latin-1'))

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_file.name)

    return send_file(temp_file.name, as_attachment=True, download_name="lecture_notes.pdf")

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))