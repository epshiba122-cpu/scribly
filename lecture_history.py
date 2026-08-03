import sqlite3
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DB_PATH = 'lecture_history.db'

def init_history_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS lectures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            transcript TEXT NOT NULL,
            summary TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    print("Lecture history database initialized!")

def make_title(transcript):
    words = transcript.strip().split()
    title = " ".join(words[:8])
    return title + ("..." if len(words) > 8 else "")

def save_lecture(transcript, summary):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    title = make_title(transcript)
    cursor.execute(
        'INSERT INTO lectures (title, transcript, summary) VALUES (?, ?, ?)',
        (title, transcript, summary)
    )
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id, title

def get_all_lectures():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT id, title, summary, created_at FROM lectures ORDER BY created_at DESC')
    rows = cursor.fetchall()
    conn.close()
    return [{'id': r[0], 'title': r[1], 'summary': r[2], 'created_at': r[3]} for r in rows]

def find_related_lectures(current_transcript, current_id, top_n=3, min_similarity=0.15):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT id, title, transcript FROM lectures WHERE id != ?', (current_id,))
    past = cursor.fetchall()
    conn.close()

    if len(past) == 0:
        return []

    past_ids = [p[0] for p in past]
    past_titles = [p[1] for p in past]
    past_texts = [p[2] for p in past]

    all_texts = past_texts + [current_transcript]

    try:
        tfidf = TfidfVectorizer(stop_words='english', max_features=500)
        matrix = tfidf.fit_transform(all_texts)
        similarities = cosine_similarity(matrix[-1], matrix[:-1])[0]
    except Exception:
        return []

    scored = list(zip(past_ids, past_titles, similarities))
    scored = [s for s in scored if s[2] >= min_similarity]
    scored.sort(key=lambda x: x[2], reverse=True)

    return [{'id': s[0], 'title': s[1], 'similarity': round(float(s[2]) * 100, 1)} for s in scored[:top_n]]