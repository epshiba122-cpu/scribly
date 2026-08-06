import os
from flask import Flask, request, jsonify, render_template, send_file
from lecture_history import init_history_db, save_lecture, get_all_lectures, find_related_lectures
from deep_translator import GoogleTranslator
from fpdf import FPDF
import re
import subprocess
import shutil
import tempfile
import requests

app = Flask(__name__)
init_history_db()

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}
WHISPER_API_URL = "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3"
SUMMARY_API_URL = "https://router.huggingface.co/hf-inference/models/sshleifer/distilbart-cnn-12-6"
FFMPEG_PATH = shutil.which("ffmpeg")

def clean_audio(input_path):
    if not FFMPEG_PATH:
        return input_path
    cleaned_path = input_path + "_cleaned.wav"
    cmd = [
        FFMPEG_PATH, "-y", "-i", input_path,
        "-af", "highpass=f=100,lowpass=f=7000,afftdn=nf=-35:nr=20,dynaudnorm=f=150:g=15",
        "-ar", "16000", "-ac", "1",
        cleaned_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print("FFmpeg error:", result.stderr[-1000:])
            return input_path
        return cleaned_path
    except Exception as e:
        print("Noise cleanup exception:", e)
        return input_path

def transcribe_with_hf(filepath):
    if not HF_TOKEN:
        raise Exception("HF_TOKEN is not set on the server. Add it in Render Environment settings.")
    with open(filepath, "rb") as f:
        data = f.read()
    response = requests.post(WHISPER_API_URL, headers=HF_HEADERS, data=data, timeout=180)
    if response.status_code == 503:
        raise Exception("Model is loading on Hugging Face servers, please try again in 20 seconds.")
    if response.status_code != 200:
        raise Exception(f"Whisper API error {response.status_code}: {response.text[:300]}")
    result = response.json()
    if isinstance(result, dict) and "text" in result:
        return result["text"]
    if isinstance(result, dict) and "error" in result:
        raise Exception(f"Whisper API error: {result['error']}")
    if isinstance(result, list) and len(result) > 0 and "text" in result[0]:
        return result[0]["text"]
    raise Exception(f"Unexpected Whisper API response: {str(result)[:300]}")

def chunk_text(text, max_words=400):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i+max_words]))
    return chunks

def summarize_with_hf(text):
    if not text or len(text.strip()) < 20:
        return text
    chunks = chunk_text(text, max_words=400)
    valid_chunks = [c for c in chunks if len(c.strip()) >= 20]
    if not valid_chunks:
        return text
    summaries = []
    for chunk in valid_chunks:
        try:
            payload = {"inputs": chunk, "parameters": {"max_length": 100, "min_length": 20}}
            response = requests.post(SUMMARY_API_URL, headers=HF_HEADERS, json=payload, timeout=60)
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    summaries.append(result[0].get("summary_text", chunk[:200]))
                else:
                    summaries.append(chunk[:200])
            else:
                summaries.append(chunk[:200])
        except Exception:
            summaries.append(chunk[:200])
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
            translated_parts = [GoogleTranslator(source='auto', target=target_lang).translate(p) for p in parts]
            return " ".join(translated_parts)
        else:
            return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception as e:
        print("Translation error:", e)
        return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_audio():
    try:
        audio_file = request.files['audio']
        target_language = request.form.get('language', 'en')
        os.makedirs('uploads', exist_ok=True)
        filepath = os.path.join('uploads', audio_file.filename)
        audio_file.save(filepath)

        cleaned_filepath = clean_audio(filepath)

        original_text = transcribe_with_hf(cleaned_filepath)

        if not original_text or len(original_text.strip()) == 0:
            return jsonify({'error': 'No speech detected in the audio/video file.'}), 400

        english_text = original_text
        if target_language != 'en':
            english_text = safe_translate(original_text, 'en')

        summary_text = summarize_with_hf(english_text)
        formatted_notes = extract_key_points(english_text, summary_text)

        if target_language != 'en':
            final_transcript = safe_translate(original_text, target_language)
            final_summary = safe_translate(formatted_notes, target_language)
        else:
            final_transcript = english_text
            final_summary = formatted_notes

        lecture_id, lecture_title = save_lecture(english_text, formatted_notes)
        related = find_related_lectures(english_text, lecture_id)

        return jsonify({
            'transcript': final_transcript,
            'summary': final_summary,
            'detected_language': 'auto',
            'word_timings': [],
            'lecture_title': lecture_title,
            'related_lectures': related
        })
    except Exception as e:
        print("PROCESS ERROR:", str(e))
        return jsonify({'error': str(e)}), 500

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
    pdf.cell(0, 10, "Scribly - Lecture Notes", ln=True, align="C")
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
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)