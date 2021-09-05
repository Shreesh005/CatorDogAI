# IMPORTING NECESSARY LIBRARIES
from flask import Flask, render_template, request
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import os

# LOADING THE MODEL
model = load_model('model/model.h5')

# WRITING THE FUNCTION
def predict_catordog(model, inputimage):
    test_image = load_img(inputimage, target_size=(200,200))
    test_image = img_to_array(test_image)/255
    test_image = np.expand_dims(test_image, axis=0)

    result = model.predict(test_image).round(3)


    prediction = np.argmax(result)

    if prediction == 0:
        message = "The image uploaded is of a cat"
    else:
        message = "The image uploaded is of a dog"

    return message

# CREATING FLASK INSTANCE
app = Flask(__name__)

# CREATING ENDPOINTS
@app.route('/', methods = ['GET','POST'])
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['uploadedfile']
        filename = file.filename
        filepath = os.path.join('static/userUploaded', filename)
        file.save(filepath)
        final = predict_catordog(model,inputimage=filepath)
        return render_template('prediction.html',user_image= filepath, finaloutput = final)

if __name__ == '__main__':
    app.run(debug=True)