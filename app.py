from flask import Flask, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename
import os
import matplotlib.pyplot as plt
import cv2

app = Flask(__name__)

upload_folder = os.path.join('static', 'uploads')

app.config['UPLOAD'] = upload_folder

@app.route('/', methods=['GET', 'POST'])
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

        return render_template('index.html', img=img_path, img2=equalized_image_path, histogram=hist_image_path, histogram2=hist_equalized_image_path)
    
    return render_template('index.html')

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
def blurWajah():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img_path = os.path.join(app.config['UPLOAD'], filename)

        # Membaca gambar dengan OpenCV
        img = cv2.imread(img_path)

        # Memanggil fungsi edge_detection
        edge_image_path, hist_edge_image_path = edge_detection(img)

        return render_template('blur.html', img=img_path, edge=edge_image_path, histogram_edge=hist_edge_image_path)
    
    return render_template('blur.html')


if __name__ == '__main__': 
    app.run(debug=True,port=8001)