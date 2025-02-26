from flask import Flask, request, jsonify
import os
import pandas as pd
import chardet

app = Flask(__name__)

# Route for textchat
@app.route('/textchat', methods=['POST'])
def textchat():
    data = request.get_json()
    user_input = data.get('text')
    if user_input:
        # Replace this with actual chatbot logic
        bot_response = f"Received your message: {user_input}"
        return jsonify({"response": bot_response})
    return jsonify({"response": "No input received."}), 400

# Route for CSV upload
@app.route('/uploadcsv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({"message": "No file uploaded."}), 400
    file = request.files['file']
    try:
        # Detect encoding
        encoding = detect_encoding(file)
        df = pd.read_csv(file, encoding=encoding)
        return jsonify({"message": "CSV processed successfully!"})
    except Exception as e:
        return jsonify({"message": f"Error processing CSV: {e}"}), 500

# Route for YouTube URL
@app.route('/youtube', methods=['POST'])
def youtube():
    data = request.get_json()
    youtube_url = data.get('url')
    if youtube_url:
        # Replace this with actual YouTube URL processing logic
        return jsonify({"message": f"Processing YouTube URL: {youtube_url}"})
    return jsonify({"message": "No URL received."}), 400

# Utility to detect file encoding
def detect_encoding(file):
    raw_data = file.read()
    result = chardet.detect(raw_data)
    file.seek(0)  # Reset file pointer after reading
    return result['encoding']

if __name__ == '__main__':
    app.run(debug=True)
