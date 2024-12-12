const fs = require("fs");
const ffmpeg = require("fluent-ffmpeg");
const speech = require("@google-cloud/speech");
const { LanguageTool } = require("languagetool-api");
const { pipeline } = require("@xenova/transformers");
const wav = require("wav");
const { exec } = require("child_process");
const librosa = require("@scijs/librosa");

async function advancedPronunciationAssessment(videoPath, transcription) {
    const results = {};
    const audioPath = "temp_audio.wav";

    // Step 1: Extract audio from video
    console.log("Extracting audio from video...");
    await new Promise((resolve, reject) => {
        ffmpeg(videoPath)
            .audioCodec("pcm_s16le")
            .save(audioPath)
            .on("end", resolve)
            .on("error", reject);
    });

    // Step 2: Transcription alignment (using Google Speech-to-Text API)
    console.log("Aligning transcription with audio...");
    const client = new speech.SpeechClient();
    const audio = {
        content: fs.readFileSync(audioPath).toString("base64"),
    };
    const config = {
        encoding: "LINEAR16",
        sampleRateHertz: 16000,
        languageCode: "en-US",
    };
    const request = {
        audio,
        config,
    };
    const [response] = await client.recognize(request);
    const predictedText = response.results.map(result => result.alternatives[0].transcript).join(" ");

    // Step 3: Grammar analysis
    console.log("Analyzing grammar...");
    const languageTool = new LanguageTool("en-US");
    const grammarResults = await languageTool.check(predictedText);
    const grammarErrors = grammarResults.matches.map(match => match.message);

    // Step 4: Analyze word alignment and clarity (dummy implementation)
    console.log("Analyzing alignment and computing clarity scores...");
    const words = predictedText.split(" ");
    const wordClarityScores = words.map(word => ({ word, score: Math.random() * 0.5 + 0.5 }));

    // Step 5: Compute speaking rate and pause metrics
    console.log("Computing speaking rate and pauses...");
    const y = librosa.loadSync(audioPath);
    const totalAudioDuration = librosa.getDurationSync(y);
    const speakingRate = words.length / totalAudioDuration; // Words per second
    const pauseFraction = 0.3; // Dummy placeholder

    // Step 6: Detect filler words
    console.log("Detecting filler words...");
    const fillerWords = ["uh", "um", "like", "you know", "so"];
    const fillerWordCount = words.filter(word => fillerWords.includes(word.toLowerCase())).length;

    // Step 7: AI-generated feedback
    console.log("Generating AI-based feedback...");
    const feedbackGenerator = await pipeline("text2text-generation", "google/pegasus-large");
    const feedback = await feedbackGenerator(`Analyze the following text for pronunciation, clarity, and fluency issues: '${predictedText}'`);

    // Step 8: Compute comprehensive scores
    console.log("Calculating comprehensive scores...");
    const grammarScore = Math.max(0, 1 - grammarErrors.length / words.length);
    const clarityScoreAvg = wordClarityScores.reduce((sum, { score }) => sum + score, 0) / wordClarityScores.length;
    const fillerPenalty = Math.max(0, 1 - fillerWordCount / words.length);
    const overallScore = (grammarScore + clarityScoreAvg + fillerPenalty) / 3;

    // Step 9: Compile results
    results.aligned_words = wordClarityScores;
    results.mispronounced_words = []; // Placeholder
    results.word_clarity_scores = wordClarityScores;
    results.speaking_rate = speakingRate;
    results.pause_fraction = pauseFraction;
    results.filler_word_count = fillerWordCount;
    results.grammar_errors = grammarErrors;
    results.feedback = feedback[0].generated_text;
    results.scores = {
        grammar_score: grammarScore,
        clarity_score_avg: clarityScoreAvg,
        filler_penalty: fillerPenalty,
        overall_score: overallScore,
    };

    // Step 10: Save detailed report
    console.log("Saving assessment report...");
    const reportPath = "assessment_report.json";
    fs.writeFileSync(reportPath, JSON.stringify(results, null, 4));

    console.log(`Assessment report saved to ${reportPath}`);

    // Cleanup temporary audio file
    if (fs.existsSync(audioPath)) fs.unlinkSync(audioPath);

    return results;
}

// Example usage
const videoFilePath = "WIN_20241210_14_05_38_Pro.mp4";
const transcriptionText = "This is a sample transcription text.";

advancedPronunciationAssessment(videoFilePath, transcriptionText)
    .then(results => console.log(results))
    .catch(err => console.error(err));
