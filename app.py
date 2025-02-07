
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import moviepy.editor as mp
import speech_recognition as sr
from nltk.tokenize import word_tokenize
import nltk
import os

nltk.download('punkt')

app = Flask(__name__)
CORS(app)


preprocessed_data = []
video_file_path = 'static/videoplay.mp4'

def preprocess_video(video_file, language_code, chunk_duration=15, overlap_duration=5):
    """
    Preprocess the video to generate a searchable transcript with word-level timestamps.
    """
    try:
        video = mp.VideoFileClip(video_file)
        audio = video.audio
        audio.write_audiofile("temp.wav")

        r = sr.Recognizer()
        total_duration = video.duration
        preprocessed_words = []
        start_time = 0

        while start_time < total_duration:
            end_time = min(start_time + chunk_duration, total_duration)
            audio_chunk = audio.subclip(start_time, end_time)
            audio_chunk.write_audiofile("chunk.wav")

            with sr.AudioFile("chunk.wav") as source:
                audio_data = r.record(source)
                try:
                    transcript = r.recognize_google(audio_data, language=language_code)
                except sr.UnknownValueError:
                    transcript = ""
                except sr.RequestError:
                    print("API request error.")
                    transcript = ""

            words = word_tokenize(transcript.lower())
            for idx, word in enumerate(words):
               
                relative_position = (idx + 0.5) / len(words)
                word_timestamp = start_time + (relative_position * (end_time - start_time))
                preprocessed_words.append({"word": word, "timestamp": word_timestamp})

            start_time += chunk_duration - overlap_duration

        return preprocessed_words

    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        return []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search_keyword():
    try:
        data = request.get_json()
        keyword = data.get('keyword').lower()
        keyword_words = word_tokenize(keyword)

       
        timestamps = []
        for idx in range(len(preprocessed_data) - len(keyword_words) + 1):
            if all(preprocessed_data[idx + i]["word"] == keyword_words[i] for i in range(len(keyword_words))):
                timestamps.append(preprocessed_data[idx]["timestamp"])

        return jsonify({"timestamps": sorted(set(timestamps))})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    if os.path.exists(video_file_path):
        print("Preprocessing video. This may take a while...")
        preprocessed_data = preprocess_video(video_file_path, language_code='en-US')
        print("Preprocessing complete.")
    else:
        print("Video file not found. Please ensure the file exists at the specified path.")
    app.run(debug=True)
    

