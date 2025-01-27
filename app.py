from flask import Flask, request, jsonify
import os
from obj_det import detect_objects  # Object detection for images (YOLOv8s)
from od_yolo_tiny_w import detect_objects_tiny  # YOLO Tiny for videos
from tracking import track_objects  # YOLOv4 tracking for videos
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Ensure the directory exists for uploads
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return "Object Tracking System API is running!"

@app.route('/process', methods=['POST'])
def process_file():
    try:
        # Extract input type and model from the form
        input_type = request.form.get('input_type')
        model_name = request.form.get('model')
        file = request.files.get('file')

        # Validate the uploaded file
        if not file or file.filename == '':
            return jsonify({"message": "No file uploaded or invalid file!"}), 400

        if not allowed_file(file.filename):
            return jsonify({"message": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process based on input type and model
        if input_type == 'image' and model_name == 'yolov8s':
            result_message = detect_objects(file_path)  # Call detect_objects from obj_det.py

        elif input_type == 'video' and model_name == 'yolov4':
            result_message = track_objects(file_path)  # Call track_objects from tracking.py

        elif input_type == 'video' and model_name == 'od_yolo_tiny_w':
            result_message = detect_objects_tiny(file_path)  # Call detect_objects_tiny from od_yolo_tiny_w.py

        else:
            result_message = "Unsupported combination of input type or model."

        return jsonify({"message": result_message})

    except Exception as e:
        return jsonify({"message": f"Error processing file: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
