import base64
import json
from io import BytesIO

import numpy as np
from PIL import Image
from flask import Flask, jsonify, request
import os
import face_recognition
import mysql.connector

app = Flask(_name_)

known_face_encodings = []
known_face_names = []

def train_with_mysql():
    cnx = mysql.connector.connect(user='your_username', password='your_password',
                                host='your_host', database='your_database')
    cursor = cnx.cursor()
    query = "SELECT name, image FROM faces"
    cursor.execute(query)
    for (name, image) in cursor:
        image = np.frombuffer(image, np.uint8)
        face_encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(name)
    cursor.close()
    cnx.close()

def train_with_images_in_folder():
    path = "images/"
    for filename in os.listdir(path):
        if not (filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png")):
            continue
        image_path = os.path.join(path, filename)
        image = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(image)[0]
        name = os.path.splitext(filename)[0]
        name = name.replace("_", " ")
        known_face_encodings.append(face_encoding)
        known_face_names.append(name)

train_with_images_in_folder()

@app.route('/', methods=['POST'])
def compare_faces():
    image_data = request.get_data()
    print(image_data[0:30])
    image_data = image_data.split(b';base64,')[1]
    image_data = base64.b64decode(image_data)

    image = Image.open(BytesIO(image_data))
    image_np = np.array(image)
    face_locations = face_recognition.face_locations(image_np)
    if not face_locations:
        return jsonify({"error": "no faces found in the image"}), 400

    face_encoding = face_recognition.face_encodings(image_np, face_locations)[0]
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    face_index = -1
    for i, match in enumerate(matches):
        if match:
            face_index = i
            break

    if face_index == -1:
        return jsonify({"name": "unknown"})

    return jsonify({"name": known_face_names[face_index]})


app.run(debug=True, port=8000)
