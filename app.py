from flask import Flask, request, jsonify
import moviepy.editor as mp
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torchaudio
from transformers import pipeline
import language_tool_python
import os
import json


def advanced_pronunciation_assessment(video_path, transcription):
    """
    Evaluate word clarity, phonetic accuracy, fluency metrics, and provide comprehensive feedback.

    Args:
        video_path (str): Path to the video file.
        transcription (str): Transcription text of the video.

    Returns:
        dict: Results including word alignment, clarity scores, mispronunciations, speaking rate, pause patterns, filler words, feedback, and comprehensive scores.
    """
    results = {}

    # Step 1: Extract audio from video
    print("Extracting audio from video...")
    audio_path = "temp_audio.wav"
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, codec="pcm_s16le")

    # Step 2: Use Gentle or another method for word alignment (replace Gentle if not set up)
    print("Aligning transcription with audio...")

    # If Gentle is not working, replace with another transcription tool
    # For example, using Hugging Face's Wav2Vec2 or DeepSpeech here

    # Step 3: Phoneme Analysis with Wav2Vec2
    print("Performing phoneme-level analysis with Wav2Vec2...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60")
    model_wav2vec = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60")
    waveform, rate = torchaudio.load(audio_path)
    resampled_waveform = torchaudio.transforms.Resample(orig_freq=rate, new_freq=16000)(waveform)
    inputs = processor(resampled_waveform[0], sampling_rate=16000, return_tensors="pt", padding=True)
    logits = model_wav2vec(**inputs).logits
    predicted_ids = np.argmax(logits.detach().numpy(), axis=-1)
    predicted_text = processor.batch_decode(predicted_ids)[0]

    # Step 4: Grammar Analysis
    print("Analyzing grammar...")
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(predicted_text)
    grammar_errors = [match.message for match in matches]

    # Step 5: Analyze word alignment for clarity and fluency
    print("Analyzing alignment and computing clarity scores...")
    mispronounced_words = []
    word_clarity_scores = []
    aligned_words = []
    total_duration = 0

    # Replace this with your own word alignment process (Gentle or other tool)
    # Using predicted_text for clarity scores instead
    words = predicted_text.split()  # Dummy for the sake of example
    for word in words:
        # Assuming a dummy case where each word is aligned with a score (replace with actual alignment)
        clarity_score = 0.8  # Just an example; use actual analysis here
        word_clarity_scores.append((word, clarity_score))

    # Step 6: Compute Speaking Rate and Pause Metrics
    print("Computing speaking rate and pauses...")
    y, sr = librosa.load(audio_path, sr=None)
    total_audio_duration = librosa.get_duration(y=y, sr=sr)
    speaking_rate = len(words) / total_audio_duration  # Words per second
    pause_fraction = 1 - (total_duration / total_audio_duration)  # Fraction of time spent pausing

    # Step 7: Analyze Filler Word Usage
    print("Detecting filler words...")
    filler_words = ["uh", "um", "like", "you know", "so"]
    filler_word_count = sum(1 for word in predicted_text.split() if word.lower() in filler_words)

    # Step 8: AI-Generated Feedback
    print("Generating AI-based feedback...")
    feedback_generator = pipeline("text2text-generation", model="google/pegasus-large")
    feedback = feedback_generator(f"Analyze the following text for pronunciation, clarity, and fluency issues: '{predicted_text}'")[0]["generated_text"]

    # Step 9: Compute Comprehensive Scores
    print("Calculating comprehensive scores...")
    grammar_score = max(0, 1 - len(grammar_errors) / len(predicted_text.split()))
    clarity_score_avg = np.mean([score for _, score in word_clarity_scores])
    filler_penalty = max(0, 1 - (filler_word_count / len(predicted_text.split())))
    overall_score = np.mean([grammar_score, clarity_score_avg, filler_penalty])

    # Step 10: Compile Results
    results["aligned_words"] = word_clarity_scores
    results["mispronounced_words"] = mispronounced_words
    results["word_clarity_scores"] = word_clarity_scores
    results["speaking_rate"] = speaking_rate
    results["pause_fraction"] = pause_fraction
    results["filler_word_count"] = filler_word_count
    results["grammar_errors"] = grammar_errors
    results["feedback"] = feedback
    results["scores"] = {
        "grammar_score": grammar_score,
        "clarity_score_avg": clarity_score_avg,
        "filler_penalty": filler_penalty,
        "overall_score": overall_score
    }

    # Generate detailed report
    report_path = "assessment_report.json"
    with open(report_path, "w") as report_file:
        json.dump(results, report_file, indent=4)

    print(f"Assessment report saved to {report_path}")

    # Cleanup temporary audio file
    if os.path.exists(audio_path):
        os.remove(audio_path)

    return results




app = Flask(__name__)

# Your existing advanced_pronunciation_assessment function goes here
# Ensure it returns the results as a JSON response

@app.route('/analyze', methods=['POST'])
def analyze():
    # Get the video file and transcription text from the request
    video_file = request.files['video']
    transcription_text = request.form['transcription']

    # Save the video file temporarily
    video_path = "temp_video.mp4"
    video_file.save(video_path)

    # Call your advanced_pronunciation_assessment function
    results = advanced_pronunciation_assessment(video_path, transcription_text)

    # Remove the temporary video file
    if os.path.exists(video_path):
        os.remove(video_path)

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
