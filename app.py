import os
import base64
import time
import wave
import math
import shutil
from flask import Flask, request, jsonify, render_template, send_file
from lecture_history import init_history_db, save_lecture, get_all_lectures, find_related_lectures
from deep_translator import GoogleTranslator
from fpdf import FPDF
import re
import subprocess
import tempfile
import requests
import imageio_ffmpeg

app = Flask(__name__)
init_history_db()

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}
WHISPER_API_URL = "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3"
SUMMARY_API_URL = "https://router.huggingface.co/hf-inference/models/sshleifer/distilbart-cnn-12-6"

FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()

CHUNK_SECONDS = 20


def clean_audio(input_path, work_dir):
    """Cleans noise and converts to a standard 16kHz mono WAV file, inside work_dir."""
    if not FFMPEG_PATH:
        return input_path
    cleaned_path = os.path.join(work_dir, "cleaned.wav")
    cmd = [
        FFMPEG_PATH, "-y", "-i", input_path,
        "-af", "highpass=f=80,afftdn=nf=-20,dynaudnorm=f=150:g=10",
        "-ar", "16000", "-ac", "1",
        cleaned_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0 or not os.path.exists(cleaned_path):
            print("FFmpeg clean error:", result.stderr[-1000:])
            return input_path
        return cleaned_path
    except Exception as e:
        print("Noise cleanup exception:", e)
        return input_path


def get_wav_duration(filepath):
    """Reads duration directly from the WAV header — no ffprobe needed."""
    try:
        with wave.open(filepath, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)
    except Exception as e:
        print("wave duration read failed:", e)
        return None


def split_audio_into_chunks(filepath, work_dir, chunk_seconds=CHUNK_SECONDS):
    """Splits a WAV file into (chunk_filepath, start_offset_seconds) pieces inside work_dir.
    Only returns chunks that were actually created successfully on disk."""
    duration = get_wav_duration(filepath)

    if duration is None:
        return [(filepath, 0)]

    num_chunks = max(1, math.ceil(duration / chunk_seconds))
    chunk_paths = []

    for i in range(num_chunks):
        start = i * chunk_seconds
        chunk_path = os.path.join(work_dir, f"part{i}.wav")
        cmd = [
            FFMPEG_PATH, "-y", "-i", filepath,
            "-ss", str(start), "-t", str(chunk_seconds),
            "-ar", "16000", "-ac", "1",
            chunk_path
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except Exception as e:
            print(f"Chunk {i} split exception:", e)
            continue

        if result.returncode == 0 and os.path.isfile(chunk_path) and os.path.getsize(chunk_path) > 200:
            chunk_paths.append((chunk_path, start))
        else:
            print(f"Chunk {i} was not created successfully, skipping it. ffmpeg stderr: {result.stderr[-300:] if result.stderr else '(none)'}")

    if not chunk_paths:
        return [(filepath, 0)]

    return chunk_paths


def transcribe_chunk(chunk_path, max_retries=3, retry_wait_seconds=12):
    """Sends one short audio chunk to Whisper, retrying if the model is still loading.
    Returns ("", []) on any failure instead of raising, so one bad chunk never kills the whole job."""
    if not HF_TOKEN:
        raise Exception("HF_TOKEN is not set on the server.")

    if not os.path.isfile(chunk_path):
        print(f"transcribe_chunk: file missing, skipping -> {chunk_path}")
        return "", []

    try:
        with open(chunk_path, "rb") as f:
            audio_bytes = f.read()
    except (FileNotFoundError, OSError) as e:
        print(f"transcribe_chunk: could not read {chunk_path}: {e}")
        return "", []

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    headers = HF_HEADERS.copy()
    headers["Content-Type"] = "application/json"

    payload = {
        "inputs": audio_b64,
        "parameters": {"return_timestamps": "word"}
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(WHISPER_API_URL, headers=headers, json=payload, timeout=60)
        except requests.exceptions.Timeout:
            print(f"Chunk timeout, attempt {attempt}/{max_retries}")
            continue
        except requests.exceptions.RequestException as e:
            print(f"Chunk request error: {e}, attempt {attempt}/{max_retries}")
            time.sleep(3)
            continue

        if response.status_code == 503:
            print(f"Chunk 503 (model loading), attempt {attempt}/{max_retries}, waiting {retry_wait_seconds}s...")
            time.sleep(retry_wait_seconds)
            continue

        if response.status_code != 200:
            print(f"Chunk API error {response.status_code}: {response.text[:200]}")
            return "", []

        try:
            result = response.json()
        except Exception:
            return "", []

        if isinstance(result, dict) and "error" in result:
            err = str(result["error"]).lower()
            if "loading" in err and attempt < max_retries:
                time.sleep(retry_wait_seconds)
                continue
            print("Chunk error:", result["error"])
            return "", []

        text = result.get("text", "") if isinstance(result, dict) else ""
        chunks = result.get("chunks", []) if isinstance(result, dict) else []

        word_timings = []
        for c in chunks:
            ts = c.get("timestamp", [None, None])
            word_timings.append({
                "word": (c.get("text") or "").strip(),
                "start": ts[0] if ts and ts[0] is not None else 0,
                "end": ts[1] if ts and ts[1] is not None else 0
            })

        return text.strip(), word_timings

    return "", []


def transcribe_with_hf(filepath, work_dir):
    """
    Splits cleaned audio into short (~20s) chunks (all inside a unique work_dir for this
    request, so no filename collisions with other requests) and transcribes each one
    separately. A chunk that fails to split or transcribe is skipped, not fatal.
    Returns (full_text, word_timings).
    """
    chunk_infos = split_audio_into_chunks(filepath, work_dir, CHUNK_SECONDS)
    print(f"Split into {len(chunk_infos)} chunk(s) for transcription.")

    full_text_parts = []
    all_word_timings = []

    for idx, (chunk_path, start_offset) in enumerate(chunk_infos):
        chunk_text, chunk_word_timings = transcribe_chunk(chunk_path)
        print(f"Chunk {idx} (offset {start_offset}s): {len(chunk_text)} chars transcribed.")

        if chunk_text:
            full_text_parts.append(chunk_text)

        for wt in chunk_word_timings:
            all_word_timings.append({
                "word": wt["word"],
                "start": round(wt["start"] + start_offset, 2),
                "end": round(wt["end"] + start_offset, 2)
            })

    full_text = " ".join(full_text_parts).strip()

    if not full_text:
        raise Exception("No speech could be transcribed from the audio.")

    return full_text, all_word_timings


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
    work_dir = tempfile.mkdtemp(prefix="scribly_")
    try:
        audio_file = request.files['audio']
        target_language = request.form.get('language', 'en')

        original_ext = os.path.splitext(audio_file.filename or "")[1] or ".webm"
        filepath = os.path.join(work_dir, f"upload{original_ext}")
        audio_file.save(filepath)

        cleaned_filepath = clean_audio(filepath, work_dir)

        original_text, word_timings = transcribe_with_hf(cleaned_filepath, work_dir)

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
            'word_timings': word_timings,
            'lecture_title': lecture_title,
            'related_lectures': related
        })
    except Exception as e:
        print("PROCESS ERROR:", str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


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