
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy import ndimage
from rembg import remove

app = Flask(__name__)

upload_folder = os.path.join('static', 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

app.config['UPLOAD'] = upload_folder
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

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

@app.route('/background_remove', methods=['GET', 'POST'])
def removebg():
    if request.method == 'POST':
        if 'img' not in request.files:
            return render_template('remove_bg.html', error='No image file uploaded')

        img_file = request.files['img']

        if img_file.filename == '':
            return render_template('remove_bg.html', error='No selected image')

        if img_file:
            input_image_path = os.path.join(app.config['UPLOAD'], 'background_input.png')
            output_image_path = os.path.join(app.config['UPLOAD'], 'background_output.png')

            img_file.save(input_image_path)
            remove_background(input_image_path, output_image_path)

            return render_template('remove_bg.html', input_img=input_image_path, output_img=output_image_path)

    return render_template('remove_bg.html')


def remove_background(input_image_path, output_image_path):
    # Use the rembg library to remove the background
    with open(input_image_path, 'rb') as input_file:
        with open(output_image_path, 'wb') as output_file:
            output_file.write(remove(input_file.read()))



def merge_two_images(image_path1, image_path2, width, height):
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)

    # Pastikan gambar memiliki dimensi yang sama
    img1 = cv2.resize(img1, (width, height))
    img2 = cv2.resize(img2, (width, height))

    # Lakukan penggabungan gambar setelah mereka memiliki dimensi yang sama
    merged_image = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

    return merged_image


@app.route('/mergeImages', methods=['GET', 'POST'])
def merge_images():
    if request.method == 'POST':
        # Check if the 'img' file is in the request
        file1 = request.files['img1']
        file2 = request.files['img2']

        filename1 = secure_filename(file1.filename)
        filename2 = secure_filename(file2.filename)

        file2.save(os.path.join(app.config['UPLOAD'], filename2))
        file1.save(os.path.join(app.config['UPLOAD'], filename1))
        image_path = os.path.join(app.config['UPLOAD'], filename1)
        image_path2 = os.path.join(app.config['UPLOAD'], filename2)
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
        return render_template('merge_images.html', merged_image=merged_image_path, img = image_path, img2 = image_path2)

    return render_template('merge_images.html')

def cartoonize_1(img, k):
    # Convert the input image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Peform adaptive threshold
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 8)
    # Defining input data for clustering
    data = np.float32(img).reshape((-1, 3))

    # Defining criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    # Applying cv2.kmeans function
    _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)

    # Reshape the output data to the size of input image
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    #cv2.imshow("result", result)
    # Smooth the result
    blurred = cv2.medianBlur(result, 3)
    # Combine the result and edges to get final cartoon effect
    cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)
    car_img = os.path.join(app.config['UPLOAD'], 'cartoon_hasil.jpg')
    cv2.imwrite(car_img, cartoon)
    return car_img

@app.route('/cartoonImg', methods=['GET', 'POST'])
def predict():
    error = None
    if request.method == 'POST':
        # Check if the 'img' file is in the request
        if 'img' not in request.files:
            error = 'Please Select a Picture'
            return render_template('predict.html', error=error)

        file = request.files['img']

        # Check if the file name is empty
        if file.filename == '':
           error = 'Please Select a Picture'
           return render_template('predict.html', error=error)

        # Check if the file is allowed (e.g., only image files)
        allowed_extensions = {'jpg', 'jpeg', 'png', 'gif'}
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            error = 'File is not allowed'
            return render_template('predict.html', error=error)

        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        image_path = os.path.join(app.config['UPLOAD'], filename)

        # Load the image file into a NumPy array
        img = cv2.imread(image_path)

        # Call the function to apply camera lens effect
        lens_effect_image_path = cartoonize_1(img, 6)

        return render_template('predict.html', img=image_path, img2=lens_effect_image_path)

    return render_template('predict.html')

def apply_lomo_effect(img):
    # Menerapkan modifikasi warna dengan memindahkan komponen warna merah dan biru
    img[:, :, 2] = np.clip(img[:, :, 2] * 1.2, 0, 255)  # Komponen warna merah
    img[:, :, 1] = np.clip(img[:, :, 1] * 1.1, 0, 255)  # Komponen warna hijau
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

@app.route('/opening', methods=['GET', 'POST'])
def opening_image():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img_path = os.path.join(app.config['UPLOAD'], filename)


        # Perform morphological opening
        img = cv2.imread(img_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binarized_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binarized_img, cv2.MORPH_OPEN, kernel, iterations=1)

        # Save the result to a file
        opening_image_path = os.path.join(app.config['UPLOAD'], 'opening_image.jpg')
        cv2.imwrite(opening_image_path, opening)

        return render_template('opening.html', img=img_path, img2=opening_image_path)
    return render_template('opening.html')

@app.route('/closing', methods=['GET', 'POST'])
def closing():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img_path = os.path.join(app.config['UPLOAD'], filename)

        img = cv2.imread(img_path)

        # Perform morphological closing
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binarized_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(binarized_img, cv2.MORPH_CLOSE, kernel, iterations=4)

        # Save the result to a file
        closing_image_path = os.path.join(app.config['UPLOAD'], 'closing_image.jpg')
        cv2.imwrite(closing_image_path, closing)

        return render_template('closing.html', img=img_path, img2=closing_image_path)
    return render_template('closing.html')

def erode_image(image_path):
    img = cv2.imread(image_path, 0)


    # Binerisasi gambar
    _, binarized = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)

    # Membuat kernel
    kernel = np.ones((5, 5), np.uint8)

    # Inversi gambar
    inverted = cv2.bitwise_not(binarized)

    # Erosi gambar
    erosion = cv2.erode(inverted, kernel, iterations=1) #erosi dan dilasi tinggal ubah di cv2.erode dan cv2.dilate aja yah
    
    return erosion

@app.route('/erosion', methods=['GET', 'POST'])
def erosion():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        img_path = os.path.join(app.config['UPLOAD'], filename)
        file.save(img_path)
            
        eroded_image = erode_image(img_path)
        eroded_image_path = os.path.join(app.config['UPLOAD'], 'eroded_' + filename)
        cv2.imwrite(eroded_image_path, eroded_image)
        return render_template('erosion.html', img=img_path, img2=eroded_image_path)
    
    return render_template('erosion.html')

def dilate_image(image_path):
    img = cv2.imread(image_path, 0)

    # Binerisasi gambar
    _, binarized = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)

    # Membuat kernel
    kernel = np.ones((5, 5), np.uint8)

    # Inversi gambar
    inverted = cv2.bitwise_not(binarized)

    # Erosi gambar
    dilate = cv2.dilate(inverted, kernel, iterations=1) #erosi dan dilasi tinggal ubah di cv2.erode dan cv2.dilate aja yah
    
    return dilate

@app.route('/dilate', methods=['GET', 'POST'])
def dilate():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        img_path = os.path.join(app.config['UPLOAD'], filename)
        file.save(img_path)
            
        dilated_image = dilate_image(img_path)
        dilated_image_path = os.path.join(app.config['UPLOAD'], 'eroded_' + filename)
        cv2.imwrite(dilated_image_path, dilated_image)
        return render_template('dilate.html', img=img_path, img2=dilated_image_path)
    
    return render_template('dilate.html')

@app.route('/scaling', methods=['GET', 'POST'])
def scaling():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img_path = os.path.join(app.config['UPLOAD'], filename)

        img = cv2.imread(img_path)

        # Mendefinisikan scale
        scale_x = float(request.form['scale_x'])  
        scale_y = float(request.form['scale_y'])  

        # scaling menggunakan bilinear interpolation
        scaled_img = cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

        # Menyimpan gambar
        scaled_image_path = os.path.join(app.config['UPLOAD'], 'scaled_image.jpg')
        cv2.imwrite(scaled_image_path, scaled_img)

        img_matrix = cv2.imread(scaled_image_path, 0)  # Mode 0 untuk grayscale, 1 untuk RGB
        img_matrix_list = img_matrix.tolist()

        return render_template('bilinear.html', img=img_path, img2=scaled_image_path, img2_data=img_matrix_list)
    return render_template('bilinear.html')

@app.route('/scaling_02', methods=['GET', 'POST'])
def scaling_02():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img_path = os.path.join(app.config['UPLOAD'], filename)

        img = cv2.imread(img_path)

        # Mendefinisikan scale
        scale_x = float(request.form['scale_x'])  
        scale_y = float(request.form['scale_y'])  

         # scaling menggunakan bicubic interpolation
        scaled_img = cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_CUBIC)

        # # Menyimpan gambar
        scaled_image_path = os.path.join(app.config['UPLOAD'], 'scaled_image.jpg')
        cv2.imwrite(scaled_image_path, scaled_img)

        img_matrix = cv2.imread(scaled_image_path, 0)  # Mode 0 untuk grayscale, 1 untuk RGB
        img_matrix_list = img_matrix.tolist()

        return render_template('bicubic.html', img=img_path, img2=scaled_image_path, img2_data=img_matrix_list)
    return render_template('bicubic.html')

def rank_order_filter(img, footprint):
    # Convert to grayscale as the rank order filter example seems designed for single-channel
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    # Apply the rank order median filter
    filtered_img = ndimage.median_filter(gray_img, footprint=footprint)
    
    # Convert back to BGR for consistency with input
    filtered_img_bgr = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)
    
    return filtered_img_bgr

def outlier_method(img, D=0.2):
    # Convert to grayscale as the algorithm seems designed for single-channel
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Kernel for neighborhood averaging
    av = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]) / 8.0
    
    # Apply neighborhood averaging
    mean_neighbors = ndimage.convolve(gray_img, av, mode='nearest')

    # Detecting outliers
    outliers = np.abs(gray_img - mean_neighbors) > D

    # Replacing outlier pixel values with the mean of their neighbors
    gray_img[outliers] = mean_neighbors[outliers]

    return gray_img

@app.route('/saltnpeper', methods=['GET', 'POST'])
def saltnpepper():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img_path = os.path.join(app.config['UPLOAD'], filename)

        img = cv2.imread(img_path)
        
        if request.form.get('filter_type') == 'median':
            # Membersihkan salt and pepper noise menggunakan filter median
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            median_img = cv2.medianBlur(gray_img, 5)  

            # Menyimpan gambar yang telah dibersihkan
            median_clean_img = os.path.join(app.config['UPLOAD'], 'cleaned_image.jpg')
            cv2.imwrite(median_clean_img, median_img)

            filter_type = request.form.get('filter_type', None)

            return render_template('saltnpepper.html', img=img_path, img2=median_clean_img, filter_type=filter_type)
        
        if request.form.get('filter_type') == 'lowpass':
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Membersihkan gambar menggunakan filter low-pass (Gaussian blur)
            lowpass_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

            # Menyimpan gambar yang telah dibersihkan
            lowpass_clean_img = os.path.join(app.config['UPLOAD'], 'cleaned_image_lowpass.jpg')
            cv2.imwrite(lowpass_clean_img, lowpass_img)

            filter_type = request.form.get('filter_type', None)

            return render_template('saltnpepper.html', img=img_path, img2=lowpass_clean_img, filter_type=filter_type)
        
        if request.form.get('filter_type') == 'rank-order':
            # Define the non-rectangular mask (cross-shaped)
            cross = np.array([[0,1,0],[1,1,1],[0,1,0]])

            # Apply the rank order filter
            filtered_img = rank_order_filter(img, cross)

            # Save the filtered image
            filtered_img_path = os.path.join(app.config['UPLOAD'], 'filtered_image.jpg')
            cv2.imwrite(filtered_img_path, filtered_img)

            filter_type = 'rank-order'

            return render_template('saltnpepper.html', img=img_path, img2=filtered_img_path, filter_type=filter_type)
        
        if request.form.get('filter_type') == 'outlier':
            # Apply the outlier method
            cleaned_img = outlier_method(img)

            # Save the cleaned image
            cleaned_img_path = os.path.join(app.config['UPLOAD'], 'cleaned_image.jpg')
            cv2.imwrite(cleaned_img_path, cleaned_img)

            filter_type = 'outlier'

            return render_template('saltnpepper.html', img=img_path, img2=cleaned_img_path, filter_type=filter_type)
    
    return render_template('saltnpepper.html')

codeList = [5, 6, 7, 4, 0, 3, 2, 1]

def getChainCode(dx, dy):
    hashKey = (3 * dy + dx + 4) % len(codeList)
    return codeList[hashKey]

def generateChainCode(ListOfPoints):
    chainCode = []
    for i in range(len(ListOfPoints) - 1):
        a = ListOfPoints[i]
        b = ListOfPoints[i + 1]
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        chainCode.append(getChainCode(dx, dy))
    return chainCode

def calculate_chain_code_from_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:  
        largest_contour = max(contours, key=cv2.contourArea)
        
        if largest_contour.size > 0:  
            chain_code = generateChainCode([point[0] for point in largest_contour])
            return chain_code
        else:
            return None  
    else:
        return None  

@app.route('/chain_code', methods=['GET', 'POST'])
def calculate_chain_code():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        image_path = os.path.join(app.config['UPLOAD'], filename)
        image = cv2.imread(image_path)
        chain_code = calculate_chain_code_from_image(image)
        return render_template('chain_code.html', img=image_path, img2=chain_code)
    return render_template('chain_code.html', error_message='Please upload an image.')

@app.route('/huffman_code', methods=['GET', 'POST'])
def huffman_code():
    return render_template('huffman_code.html')


if __name__ == '__main__':
    app.run(debug=True, port=8001)




  