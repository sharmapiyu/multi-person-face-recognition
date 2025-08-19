from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import cv2
import pandas as pd
import uuid
from utils.face_recognition_utils import process_video
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['REPORT_FOLDER'] = 'reports'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORT_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return redirect(url_for('index'))

    file = request.files['video']
    if file.filename == '':
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    report_name = f"report_{uuid.uuid4().hex}.csv"
    report_path = os.path.join(app.config['REPORT_FOLDER'], report_name)

    process_video(filepath, report_path)

    return send_file(report_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
