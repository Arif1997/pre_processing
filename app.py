import cv2
from flask import Flask, jsonify, request 
from flask_cors import CORS 
import numpy as np
from pre_processing import apply_filter_and_calculate_metrics, calculate_metrics_for_folder, calculate_metrics_for_image
from utils.index import get_filter_info, open_file_dialog
app = Flask(__name__)

CORS(app)
CORS(app, origins=["http://localhost:5173", '*'], methods=["GET", "POST", "DELETE", "PUT"], allow_headers=["Content-Type"])

@app.route('/folder')
def result():
    folder_path = open_file_dialog()
    if folder_path:
        best_filter_ssim, best_ssim_accuracy, best_filter_psnr, best_psnr_accuracy= calculate_metrics_for_folder(folder_path)
        return jsonify({"best_fl_psnr": best_filter_psnr, "best_fl_ssim": best_filter_ssim, "best_psnr_accuracy": best_psnr_accuracy, "best_ssim_accuracy": best_ssim_accuracy, "folder_path": folder_path, "result_folder": "D:/Projects//result"})
    else:
        return jsonify({"error": "No folder selected"}), 400
    

@app.route('/upload', methods=['POST'])
def upload_image():
    # Check if the request contains an image file
    if 'image' not in request.files:
        return "No image uploaded!", 400

    # Get the image file from the request
    image_file = request.files['image']

    # Read the image file into OpenCV format
    file_stream = image_file.stream
    npimg = np.frombuffer(file_stream.read(), np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Calculate metrics and apply filters
    best_filter_ssim, best_ssim_accuracy, best_filter_psnr, best_psnr_accuracy ,best_ssim_image_base64, best_psnr_image_base64= calculate_metrics_for_image(image)
    filter_info = get_filter_info(best_filter_psnr)


    return jsonify({"best_psnr_accuracy": best_psnr_accuracy, "best_filter_psnr": best_filter_psnr, "best_filter_ssim": best_filter_ssim, "best_ssim_accuracy": best_ssim_accuracy, "best_ssim_image_base64": best_ssim_image_base64, "best_psnr_image_base64": best_psnr_image_base64, "filter_info": filter_info })


@app.route('/filter', methods=['POST'])
def selected_fitler():
    # Check if the request contains an image file
    if 'image' not in request.files:
        return "No image uploaded!", 400

    # Get the image file from the request
    image_file = request.files['image']
    filter_name = request.form.get('filter')

    

    # Read the image file into OpenCV format
    file_stream = image_file.stream
    npimg = np.frombuffer(file_stream.read(), np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    ssim_value, psnr_value, best_psnr_image_base64, best_ssim_image_base64 = apply_filter_and_calculate_metrics(image, filter_name)
    filter_info = get_filter_info(filter_name)

    return jsonify({"best_ssim_accuracy" : ssim_value, "best_psnr_accuracy": psnr_value, "best_psnr_image_base64": best_psnr_image_base64, "best_ssim_image_base64": best_ssim_image_base64, "best_filter_ssim": filter_name, "best_filter_psnr": filter_name, "filter_info": filter_info})


# Some additional routes for further development
@app.route('/')
def home():
    return jsonify({"message": "Hello, World!, it is my first project in python"})

@app.route('/user')
def user():
    return jsonify({"name": "Wahab", "age": 25, "address": "Mysore"})

@app.route('/user/<username>')
def user_profile(username):
    return jsonify({"user": username})

@app.route('/submit', methods=['POST'])
def submit_data():
    # Get JSON data from the request
    data = request.get_json()

    print(f"Received data: {data}")

    # Return a response
    return jsonify({"status": "success", "data_received": data}), 200

if __name__ == '__main__':
    app.run(debug=True)
