from flask import Flask, render_template, Response, request, jsonify
import tensorflow as tf
import cv2
import numpy as np

import io
from PIL import Image

import base64

import sqlite3

from flask_login import LoginManager, login_required, UserMixin, login_user, current_user

app = Flask(__name__)
app.secret_key = 'abcdefghijklmnopqrstuy'

login_manager = LoginManager(app)
login_manager.login_view = 'signin'

class User(UserMixin):
    pass


# Loads user on every request
@login_manager.user_loader
def load_user(user_id):
    user = User()
    user.id = user_id

    return user

connection = sqlite3.connect("FypDB.db", check_same_thread=False)


# Function to preprocess the frame before inference
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (224, 224))  # Change dimensions to match your model's input shape
    normalized_frame = resized_frame / 255.0
    return normalized_frame

# Define the path to the model directory in your Drive
model_directory = './model.h5'
# Load the smoking detection model
loaded_model = tf.keras.models.load_model(model_directory)

camera = cv2.VideoCapture(0)

# Function to detect smoking
def detect_smoking(frame):
    processed_frame = preprocess_frame(frame)
    input_tensor = tf.convert_to_tensor(processed_frame, dtype=tf.float32)
    input_tensor = tf.expand_dims(input_tensor, axis=0)
    predictions = loaded_model(input_tensor)
    smoking_probability = predictions[0][1].numpy()  # Get the probability
    threshold = 0.5  # Adjust this threshold as needed
    is_smoking = smoking_probability > threshold
    return is_smoking

def generate_frames():
    while True:
        _, frame = camera.read()

        # Preprocess the frame and make a prediction
        raw_image = frame.copy()
        is_smoking = detect_smoking(raw_image)
        is_smoking_text = 'Smoking' if is_smoking else ''

        # Draw the is_smoking value on the bottom right of the frame
        cv2.putText(frame, is_smoking_text, (frame.shape[1] - 150, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Encode the frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        jpeg_image = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'\r\n' + jpeg_image + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_feed')
def stop_feed():
    camera.release()
    return Response("Stopped feed")

@app.route('/capture_photo', methods=['POST'])
def capture_photo():
    data = request.get_json()
    image_data = base64.b64decode(data['imageBase64'].split(',')[1])
    image = Image.open(io.BytesIO(image_data))
    img_array = np.array(image)
    is_smoking = detect_smoking(img_array)
    
    return jsonify({"is_smoking": bool(is_smoking)})

@app.route('/')
@login_required
def home():
    return render_template('index.html', username=current_user.id)

@app.route('/detection')
def detection():
    return render_template("detection.html")

@app.route('/history')
def history():
    return render_template("history.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/sign-in', methods=["POST", "GET"])
def signin():
    if request.method == "GET":
        return render_template("sign-in.html")
    
    username = request.form['username']
    password = request.form['password']

    cursor = connection.cursor()

    cursor.execute("SELECT Username, Password from login WHERE Username = ?", (username,))
    data = cursor.fetchone()

    print("Got data", data)
    if data is None:
        return jsonify({"success": False, "message": "Account does not exist"})
    
    username_db = data[0]
    password_db = data[1]

    if not password == password_db:
        return jsonify({"success": False, "message": "Password does not match"})
    
    user = User()
    user.id = username
    login_user(user)

    return jsonify({"success": True})



@app.route('/sign-up', methods=['POST'])
def signup():
    username = request.form['username']
    password = request.form['password']

    cursor = connection.cursor()

    cursor.execute("INSERT INTO login (Username, Password) VALUES (?, ?)", (username, password))
    connection.commit()

    return jsonify({"success": True})

@app.route('/story')
def story():
    return render_template("story.html")

@app.route('/upload_file', methods=['POST'])
def upload_file():
    print("Uploaded file. Processing...")
    # Read the image data from the request
    image_data = io.BytesIO(request.data)

    # Use PIL to open the image file
    img = Image.open(image_data).convert('RGB')
    print("Opened image")
    # Convert the image to a NumPy array
    img_array = np.array(img)

    is_smoking = detect_smoking(img_array)
    print("Smoking detected: ", is_smoking)
    return {
        "is_smoking": bool(is_smoking),
    }

app.run(debug=True)

