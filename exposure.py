import os
import cv2
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request
from rembg import remove
import numpy as np

app = Flask(__name__)

upload_folder = os.path.join('static', 'uploads')

app.config['UPLOAD'] = upload_folder

def apply_camera_lens_effect(image_path):
    # Membaca gambar dengan OpenCV
    img = cv2.imread(image_path)

    # Menggunakan GaussianBlur pada gambar asli untuk menghasilkan latar belakang blur
    img_blurred = cv2.GaussianBlur(img, (3, 3), 10)

    # Membuat mask untuk efek bokeh
    mask = np.zeros_like(img)
    cv2.circle(mask, (img.shape[1] // 2, img.shape[0] // 2), 150, (255, 255, 255), -1)

    # Menggabungkan gambar asli dan latar belakang blur menggunakan blending mode
    result = cv2.addWeighted(img_blurred, 0.7, img, 1.0, 0)

    # Menyimpan gambar dengan efek lensa kamera
    lens_effect_image_path = os.path.join(app.config['UPLOAD'], 'lens_effect_image.jpg')
    cv2.imwrite(lens_effect_image_path, result)

    return lens_effect_image_path

@app.route('/', methods=['GET', 'POST'])
def camera_lens_effect():
    error = None
    if request.method == 'POST':
        # Check if the 'img' file is in the request
        if 'img' not in request.files:
            error = 'Please Select a Picture'
            return render_template('fitur2_rbg.html', error=error)

        file = request.files['img']

        # Check if the file name is empty
        if file.filename == '':
           error = 'Please Select a Picture'
           return render_template('fitur2_rbg.html', error=error)

        # Check if the file is allowed (e.g., only image files)
        allowed_extensions = {'jpg', 'jpeg', 'png', 'gif'}
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            error = 'File is not allowed'
            return render_template('fitur2_rbg.html', error=error)

        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        image_path = os.path.join(app.config['UPLOAD'], filename)

        # Call the function to apply camera lens effect
        lens_effect_image_path = apply_camera_lens_effect(image_path)

        return render_template('fitur2_rbg.html', img=image_path, img2=lens_effect_image_path)

    return render_template('fitur2_rbg.html')


def apply_artistic_filter(image_path):
    # Membaca gambar dengan OpenCV
    img = cv2.imread(image_path)

    # Menggunakan filter "lukisan minyak"
    oil_painting = apply_oil_painting_filter(img)

    # Menyimpan gambar hasil filter artistik
    artistic_image_path = os.path.join(app.config['UPLOAD'], 'artistic_image.jpg')
    cv2.imwrite(artistic_image_path, oil_painting)

    return artistic_image_path

def apply_oil_painting_filter(img):
    # Menggunakan filter "lukisan minyak"
    radius = 71  # Ukuran kernel filter (menentukan seberapa besar efek lukisan minyak)

    # Konversi gambar ke citra abu-abu
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplikasikan Gaussian Blur
    gray_img = cv2.GaussianBlur(gray_img, (2 * radius + 1, 2 * radius + 1), 0)

    # Dapatkan citra maksimum dalam setiap blok
    max_img = cv2.dilate(gray_img, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))

    # Inisialisasi citra hasil dengan citra berwarna yang sama dengan img
    result = np.copy(img)

    # Ubah citra maksimum ke dalam bentuk berwarna yang sesuai
    max_img = cv2.cvtColor(max_img, cv2.COLOR_GRAY2BGR)

    # Tingkatkan kontras dengan mengambil selisih antara citra asli dan citra maksimum
    result = cv2.absdiff(result, max_img)
    result = cv2.bitwise_not(result)
    
    return result

@app.route('/artisticFilter', methods=['GET', 'POST'])
def artistic_filter():
    error = None
    if request.method == 'POST':
        # Check if the 'img' file is in the request
        if 'img' not in request.files:
            error = 'Please Select a Picture'
            return render_template('artistic_filter.html', error=error)

        file = request.files['img']

        # Check if the file name is empty
        if file.filename == '':
           error = 'Please Select a Picture'
           return render_template('artistic_filter.html', error=error)

        # Check if the file is allowed (e.g., only image files)
        allowed_extensions = {'jpg', 'jpeg', 'png', 'gif'}
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            error = 'File is not allowed'
            return render_template('artistic_filter.html', error=error)

        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        image_path = os.path.join(app.config['UPLOAD'], filename)

        # Call the function to apply artistic filter
        artistic_image_path = apply_artistic_filter(image_path)

        return render_template('artistic_filter.html', img=image_path, img2=artistic_image_path)

    return render_template('artistic_filter.html')

def apply_lomo_effect(img):
    # Menerapkan modifikasi warna dengan memindahkan komponen warna merah dan biru
    img[:, :, 2] = np.clip(img[:, :, 2] * 1.2, 0, 255)  # Komponen warna merah
    img[:, :, 1] = np.clip(img[:, :, 1] * 1.2, 0, 255)
    img[:, :, 0] = np.clip(img[:, :, 0] * 0.9, 0, 255)  # Komponen warna biru

    # Menyimpan gambar dengan efek lomo ke folder "static/uploads"
    lomo_image_path = os.path.join(app.config['UPLOAD'], 'lomo_image.jpg')
    cv2.imwrite(lomo_image_path, img)

    return lomo_image_path

@app.route('/lomoEffect', methods=['GET', 'POST'])
def lomo_effect():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img_path = os.path.join(app.config['UPLOAD'], filename)

        # Membaca gambar dengan OpenCV
        img = cv2.imread(img_path)

        # Memanggil fungsi apply_lomo_effect
        lomo_image_path = apply_lomo_effect(img)

        return render_template('lomo.html', img=img_path, img2=lomo_image_path)
    
    return render_template('lomo.html')



if __name__ == '__main__': 
    app.run(debug=True,port=8001)