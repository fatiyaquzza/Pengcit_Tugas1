from flask import Flask, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np  # Tambahkan import ini

app = Flask(__name__)

upload_folder = os.path.join('static', 'uploads')

app.config['UPLOAD'] = upload_folder

# Fungsi untuk mendeteksi wajah dalam gambar dan menerapkan efek blur
def blur_faces(image_path):
    # Membaca gambar dengan OpenCV
    img = cv2.imread(image_path)

    # Menggunakan Cascade Classifier untuk mendeteksi wajah
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Menerapkan deteksi wajah dengan parameter yang diatur
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,  # Anda dapat menyesuaikan nilai ini, misalnya 1.1 atau 1.2
        minNeighbors=5,   # Anda dapat menyesuaikan nilai ini
        minSize=(20, 20)  # Anda dapat menyesuaikan ukuran minimum wajah yang diharapkan
    )
    # Menerapkan efek blur ke setiap wajah yang terdeteksi
    for (x, y, w, h) in faces:
        # Ambil bagian wajah dari gambar
        face = img[y:y+h, x:x+w]

        # Hitung kernel GaussianBlur berdasarkan proporsi ukuran wajah
        kernel_width = int(w / 7)
        kernel_height = int(h / 7)

        # Pastikan kernel memiliki ukuran ganjil
        kernel_width = kernel_width if kernel_width % 2 == 1 else kernel_width + 1
        kernel_height = kernel_height if kernel_height % 2 == 1 else kernel_height + 1

        # Terapkan efek blur Gaussian
        blurred_face = cv2.GaussianBlur(face, (kernel_width, kernel_height), 30)  # Anda dapat menyesuaikan tingkat blur di sini
        img[y:y+h, x:x+w] = blurred_face

    # Menyimpan gambar dengan wajah-wajah yang telah di-blur
    blurred_image_path = os.path.join(app.config['UPLOAD'], 'blurred_image.jpg')
    cv2.imwrite(blurred_image_path, img)

    return blurred_image_path


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        image_path = os.path.join(app.config['UPLOAD'], filename)

        # # Memanggil fungsi untuk memblur wajah
        blurred_image_path = blur_faces(image_path)

        # Melanjutkan dengan kode Anda yang sebelumnya

        # ...

        return render_template('blur.html', img=image_path, img2=blurred_image_path)
    
    return render_template('blur.html')

if __name__ == "__main__":
    app.run(debug=True)
