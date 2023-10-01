
from flask import Flask, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import urllib

app = Flask(__name__)

upload_folder = os.path.join('static', 'uploads')

app.config['UPLOAD'] = upload_folder

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/histogram', methods=['GET', 'POST'])
def histogram_equ():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img_path = os.path.join(app.config['UPLOAD'], filename)

        # Membaca gambar dengan OpenCV
        img = cv2.imread(img_path)

        # Menghitung histogram untuk masing-masing saluran (R, G, B)
        hist_r = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([img], [2], None, [256], [0, 256])

        # Normalisasi histogram
        hist_r /= hist_r.sum()
        hist_g /= hist_g.sum()
        hist_b /= hist_b.sum()

        # Simpan histogram sebagai gambar PNG
        hist_image_path = os.path.join(app.config['UPLOAD'], 'histogram.png')
        plt.figure()
        plt.title("RGB Histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.plot(hist_r, color='red', label='Red')
        plt.plot(hist_g, color='green', label='Green')
        plt.plot(hist_b, color='blue', label='Blue')
        plt.legend()
        plt.savefig(hist_image_path)

        # Hasil equalisasi
        img_equalized = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)  # Ubah ke ruang warna YCrCb
        img_equalized[:, :, 0] = cv2.equalizeHist(img_equalized[:, :, 0])  # Equalisasi komponen Y (luminance)
        img_equalized = cv2.cvtColor(img_equalized, cv2.COLOR_YCrCb2BGR)  # Kembalikan ke ruang warna BGR

        # Menyimpan gambar hasil equalisasi ke folder "static/uploads"
        equalized_image_path = os.path.join('static', 'uploads', 'img-equalized.jpg')
        cv2.imwrite(equalized_image_path, img_equalized)

        # Menghitung histogram untuk gambar yang sudah diequalisasi
        hist_equalized_r = cv2.calcHist([img_equalized], [0], None, [256], [0, 256])
        hist_equalized_g = cv2.calcHist([img_equalized], [1], None, [256], [0, 256])
        hist_equalized_b = cv2.calcHist([img_equalized], [2], None, [256], [0, 256])

        # Normalisasi histogram
        hist_equalized_r /= hist_equalized_r.sum()
        hist_equalized_g /= hist_equalized_g.sum()
        hist_equalized_b /= hist_equalized_b.sum()

        # Simpan histogram hasil equalisasi sebagai gambar PNG        
        hist_equalized_image_path = os.path.join(app.config['UPLOAD'], 'histogram_equalized.png')
        plt.figure()
        plt.title("RGB Histogram (Equalized)")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.plot(hist_equalized_r, color='red', label='Red')
        plt.plot(hist_equalized_g, color='green', label='Green')
        plt.plot(hist_equalized_b, color='blue', label='Blue')
        plt.legend()
        plt.savefig(hist_equalized_image_path)

        return render_template('histogram_equalization.html', img=img_path, img2=equalized_image_path, histogram=hist_image_path, histogram2=hist_equalized_image_path)
    
    return render_template('histogram_equalization.html')

def edge_detection(img):
    # Menerapkan deteksi tepi menggunakan algoritma Canny
    edges = cv2.Canny(img, 100, 200)  # Anda dapat mengatur threshold sesuai kebutuhan
    
    # Menyimpan gambar hasil deteksi tepi ke folder "static/uploads"
    edge_image_path = os.path.join(app.config['UPLOAD'], 'edge_detected.jpg')
    cv2.imwrite(edge_image_path, edges)

    # Menghitung histogram untuk gambar hasil deteksi tepi
    hist_edge = cv2.calcHist([edges], [0], None, [256], [0, 256])

    # Normalisasi histogram
    hist_edge /= hist_edge.sum()

    # Simpan histogram hasil deteksi tepi sebagai gambar PNG
    hist_edge_image_path = os.path.join(app.config['UPLOAD'], 'histogram_edge.png')
    plt.figure()
    plt.title("Edge Detection Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.plot(hist_edge, color='gray', label='Edge')
    plt.legend()
    plt.savefig(hist_edge_image_path)

    return edge_image_path, hist_edge_image_path

@app.route('/edge', methods=['GET', 'POST'])
def edge():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img_path = os.path.join(app.config['UPLOAD'], filename)

        # Membaca gambar dengan OpenCV
        img = cv2.imread(img_path)

        # Memanggil fungsi edge_detection
        edge_image_path, hist_edge_image_path = edge_detection(img)

        return render_template('edge.html', img=img_path, edge=edge_image_path, histogram_edge=hist_edge_image_path)
    
    return render_template('edge.html')


def blur_faces(image_path, blur_level):
    # Membaca gambar dengan OpenCV
    img = cv2.imread(image_path)

    # Menggunakan Cascade Classifier untuk mendeteksi wajah
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Menerapkan deteksi wajah dengan parameter yang diatur
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=4, minSize=[30, 30])

    # Menerapkan efek blur ke setiap wajah yang terdeteksi
    for (x, y, w, h) in faces:
        # Ambil bagian wajah dari gambar
        face = img[y:y+h, x:x+w]
        # Hitung ukuran kernel berdasarkan tingkat blur yang diatur
        kernel_size = (blur_level, blur_level)
        # Terapkan efek blur Gaussian dengan kernel yang sesuai
        blurred_face = cv2.GaussianBlur(face, kernel_size, 0)
        img[y:y+h, x:x+w] = blurred_face

    # Menyimpan gambar dengan wajah-wajah yang telah di-blur
    blurred_image_path = os.path.join(app.config['UPLOAD'], 'blurred_image.jpg')
    cv2.imwrite(blurred_image_path, img)

    return blurred_image_path


@app.route('/faceBlur', methods=['GET', 'POST'])
def face_blur():
    error = None
    if request.method == 'POST':
        # Check if the 'img' file is in the request
        if 'img' not in request.files:
            error = 'Please Select a Picture'
            return render_template('blur.html', error=error)

        file = request.files['img']

        # Check if the file name is empty
        if file.filename == '':
           error = 'Please Select a Picture'
           return render_template('blur.html', error=error)

        # Check if the file is allowed (e.g., only image files)
        allowed_extensions = {'jpg', 'jpeg', 'png', 'gif'}
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            error = 'File is not allowed'
            return render_template('blur.html', error=error)

        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        image_path = os.path.join(app.config['UPLOAD'], filename)

        # Get blur level from the form
        blur_level = int(request.form.get('tingkatan', 0))

        # Call the function to blur faces
        blurred_image_path = blur_faces(image_path, blur_level)

        return render_template('blur.html', img=image_path, img2=blurred_image_path)

    return render_template('blur.html')

def remove_bg(image_path):
    img = cv2.imread(image_path)
    edges = cv2.Canny(img, 80,150)
    kernel = np.ones((5,5), np.uint8)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
    erosion = cv2.morphologyEx(closing, cv2.MORPH_ERODE, kernel, iterations=1)

    # When using Grabcut the mask image should be:
    #    0 - sure background
    #    1 - sure foreground
    #    2 - unknown

    mask = np.zeros(img.shape[:2], np.uint8)
    mask[:] = 2
    mask[erosion == 255] = 1
    bgdmodel = np.zeros((1, 65), np.float64)
    fgdmodel = np.zeros((1, 65), np.float64)

    out_mask = mask.copy()
    out_mask, _, _ = cv2.grabCut(img,out_mask,None,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_MASK)
    out_mask = np.where((out_mask==2)|(out_mask==0),0,1).astype('uint8')
    img_removed_bg = img*out_mask[:,:,np.newaxis]
    removed_bg_image_path = os.path.join(app.config['UPLOAD'], 'img_removed_bg.jpg')
    cv2.imwrite(removed_bg_image_path, img_removed_bg)

    return removed_bg_image_path

@app.route('/removeBackground', methods=['GET', 'POST'])
def removebg():
    error = None
    if request.method == 'POST':
        # Check if the 'img' file is in the request
        if 'img' not in request.files:
            error = 'Please Select a Picture'
            return render_template('remove_bg.html', error=error)

        file = request.files['img']

        # Check if the file name is empty
        if file.filename == '':
            error = 'Please Select a Picture'
            return render_template('remove_bg.html', error=error)

        # Check if the file is allowed (e.g., only image files)
        allowed_extensions = {'jpg', 'jpeg', 'png', 'gif'}
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            error = 'File is not allowed'
            return render_template('remove_bg.html', error=error)

        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        image_path = os.path.join(app.config['UPLOAD'], filename)

        # Call the function to remove the background
        remove_background_img = remove_bg(image_path)

        return render_template('remove_bg.html', img=image_path, img2=remove_background_img)

    return render_template('remove_bg.html')



if __name__ == '__main__': 
    app.run(debug=True,port=8001)