from transformers import pipeline

# Load summarization model (free, runs locally)
print("Loading summarization model... (first time takes a while)")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Read the transcript
with open("transcript_output.txt", "r") as f:
    text = f.read()

print("Original Transcript:")
print(text)
print("\n" + "="*50 + "\n")

# Summarize
summary = summarizer(text, max_length=100, min_length=20, do_sample=False)

print("Summary:")
print(summary[0]['summary_text'])

# Save summary to file
with open("summary_output.txt", "w") as f:
    f.write(summary[0]['summary_text'])

print("\nSaved to summary_output.txt")