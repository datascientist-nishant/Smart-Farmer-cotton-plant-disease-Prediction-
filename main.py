from flask import Flask, request, render_template

import numpy as np
import os
import PIL.Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

model = load_model("model/cott_dis_pred.h5")

print('@@ Model loaded')


def pred_cot_dieas(cott_plant):
    test_image = load_img(cott_plant, target_size=(150, 150))  # load image
    print("@@ Got Image for prediction")

    test_image = img_to_array(test_image) / 255  # convert image to np array and normalize
    test_image = np.expand_dims(test_image, axis=0)  # change dimention 3D to 4D

    result = model.predict(test_image).round(3)  # predict diseased palnt or not
    print('@@ Raw result = ', result)

    pred = np.argmax(result)  # get the index of max value

    if pred == 0:
        return "Healthy Cotton Plant", 'healthy_plant_leaf.html'  # if index 0 burned leaf
    elif pred == 1:
        return 'Diseased Cotton Plant', 'disease_plant.html'  # # if index 1
    elif pred == 2:
        return 'Healthy Cotton Plant', 'healthy_plant.html'  # if index 2  fresh leaf
    else:
        return "Healthy Cotton Plant", 'healthy_plant.html'  # if index 3


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict')
def start():
    return render_template('predict.html')

@app.route('/back')
def end():
    return render_template('predict.html')




@app.route("/result", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']  # fet input
        filename = file.filename
        print("@@ Input posted = ", filename)

        file_path = os.path.join('static/user uploaded', filename)
        file.save(file_path)

        print("@@ Predicting class......")
        pred, output_page = pred_cot_dieas(cott_plant=file_path)

        return render_template(output_page, pred_output=pred, user_image=file_path)


if __name__ == '__main__':
    app.run(debug=True, port=9000)
