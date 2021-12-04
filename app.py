from flask import Flask,render_template,request
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model


app = Flask(__name__)
classes=['Apple', 'Banana', 'Pineapple', 'Watermelon']
BASE_DIR=os.getcwd()
upload_folder=os.path.join(os.path.dirname(BASE_DIR),"Fruite Identifyier/static")
model=load_model('my_model.h5')
# savedModel.summary()

def predict(img_loc):
	# model=load_model('my_model.h5')
	img=image.load_img(img_loc,target_size=(128, 128))
	x=image.img_to_array(img)
	x=np.expand_dims(x,axis=0)
	images=np.vstack([x])
	res=model.predict(images)
	index=np.argmax(res)
	return classes[index]
	


@app.route('/',methods=["GET","POST"])
def upload():
	if request.method=="POST":
		image_file=request.files["image"]
		if image_file:
			image_loc=os.path.join(
				upload_folder,image_file.filename
			)
			image_file.save(image_loc)
			fruit=predict(image_loc)
			
			return render_template('predictedFruit.html',fruit=fruit, path=image_file.filename)

	return render_template('index.html')


if __name__=='__main__':
	app.run(debug=True)