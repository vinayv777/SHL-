# SHL-
import whisper
import language_tool_python

# Step 1: Transcribe audio
def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result['text']

# Step 2: Score grammar
def score_grammar(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    num_errors = len(matches)
    words = len(text.split())
    
    # Simple score: penalize by error rate
    score = max(0, 100 - (num_errors / max(1, words)) * 100)
    return round(score, 2), matches

# Step 3: Run the full pipeline
def analyze_audio(audio_path):
    transcript = transcribe_audio(audio_path)
    score, issues = score_grammar(transcript)
    return {
        "transcript": transcript,
        "grammar_score": score,
        "issues": issues
    }

# Example usage:
result = analyze_audio("your_audio_file.wav")
print("Transcript:", result["transcript"])
print("Score:", result["grammar_score"])
for issue in result["issues"]:
    print("-", issue.message)
