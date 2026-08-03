import whisper
import time

def load_model_with_retry(model_name="tiny", max_retries=5):
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Attempt {attempt}: Loading model...")
            model = whisper.load_model(model_name)
            print("Model loaded successfully!")
            return model
        except RuntimeError as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                print("Retrying in 3 seconds...")
                time.sleep(3)
            else:
                raise Exception("All attempts failed. Check your internet connection.")

model = load_model_with_retry("tiny")
result = model.transcribe("test_audio.wav")

print("Transcript:")
print(result["text"])

with open("transcript_output.txt", "w") as f:
    f.write(result["text"])

print("\nSaved to transcript_output.txt")