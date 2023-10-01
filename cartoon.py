import os
from PIL import Image, ImageFilter, ImageChops, ImageOps
import cv2
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request
from rembg import remove
import numpy as np

app = Flask(__name__)

upload_folder = os.path.join('static', 'uploads')

app.config['UPLOAD'] = upload_folder

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
    # print(center)
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

@app.route('/', methods=['GET', 'POST'])
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
        lens_effect_image_path = cartoonize_1(img, 8)

        # Convert the NumPy arrays to scalar values
        # img_exists = np.any(img)
        # lens_effect_image_exists = np.any(lens_effect_image_path)

        # Return the scalar values from the template
        return render_template('predict.html', img=image_path, img2=lens_effect_image_path)

    return render_template('predict.html')


if __name__ == '__main__':
        app.run(debug=True, host="localhost", port=8080)