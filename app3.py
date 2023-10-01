from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np

app = Flask(__name__)

# Konfigurasi folder upload
upload_folder = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = upload_folder
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

# Fungsi untuk menggabungkan dua gambar
def merge_two_images(image_path1, image_path2, width, height):
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)

    # Pastikan gambar memiliki dimensi yang sama
    img1 = cv2.resize(img1, (width, height))
    img2 = cv2.resize(img2, (width, height))

    # Lakukan penggabungan gambar setelah mereka memiliki dimensi yang sama
    merged_image = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

    return merged_image

# Fungsi untuk memeriksa apakah ekstensi file diizinkan
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    return render_template('merge_images.html', merged_image=None)

@app.route('/mergeImages', methods=['POST'])
def merge_images():
    if 'img1' not in request.files or 'img2' not in request.files:
        return "Please upload both images."

    file1 = request.files['img1']
    file2 = request.files['img2']

    if file1.filename == '' or file2.filename == '':
        return "Please select both images."

    if not allowed_file(file1.filename) or not allowed_file(file2.filename):
        return "Both files should be in JPG, JPEG, or PNG format."

    filename1 = secure_filename(file1.filename)
    filename2 = secure_filename(file2.filename)

    file1.save(os.path.join(app.config['UPLOAD'], filename1))
    file2.save(os.path.join(app.config['UPLOAD'], filename2))

    width = 640  # Ganti dengan lebar yang sesuai
    height = 480  # Ganti dengan tinggi yang sesuai

    merged_image = merge_two_images(
        os.path.join(app.config['UPLOAD'], filename1),
        os.path.join(app.config['UPLOAD'], filename2),
        width,
        height
    )

    merged_image_path = os.path.join(app.config['UPLOAD'], 'merged_image.jpg')
    cv2.imwrite(merged_image_path, merged_image)

    return render_template('merge_images.html', merged_image=merged_image_path)

if __name__ == '__main__':
    app.run(debug=True)

