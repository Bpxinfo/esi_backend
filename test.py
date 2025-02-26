from flask import Flask, request, jsonify
from flask_socketio import SocketIO
import time
import os

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Notify frontend that processing has started
    socketio.emit("status", {"message": "Processing file..."})
    time.sleep(4)  # Simulate processing time

    # Notify frontend that cleaning has started
    socketio.emit("status", {"message": "Cleaning file..."})
    time.sleep(3)  # Simulate processing time

    # Notify frontend that process is complete
    socketio.emit("status", {"message": "File processing complete"})
    
    return jsonify({"message": "File uploaded successfully"}), 200

if __name__ == "__main__":
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)
