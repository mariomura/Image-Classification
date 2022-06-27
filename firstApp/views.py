from django.shortcuts import render
# Create your views here.

from django.core.files.storage import FileSystemStorage

from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import json
from tensorflow import Graph
import numpy as np

img_height, img_width=224,224
with open('./models/imagenet_classes.json','r') as model:
    modelLabel=model.read()

modelLabel=json.loads(modelLabel)

model_graph = Graph()
with model_graph.as_default():
    tf_session = tf.compat.v1.Session()
    with tf_session.as_default():
        model=load_model('./models/MobileNetModelImagenet.h5')

def index(request):
    context={'a':1}
    return render(request,'index.html',context)


def predictImage(request):
    print (request)
    print (request.POST.dict())
    file=request.FILES['filePath']
    fileSave=FileSystemStorage()
    filePathName=fileSave.save(file.name,file)
    filePathName=fileSave.url(filePathName)
    imageSetup ='.'+filePathName
    img = image.load_img(imageSetup, target_size=(img_height, img_width))
    imgOutput  = image.img_to_array(img)
    imgOutput=imgOutput/255
    imgOutput=imgOutput.reshape(1,img_height, img_width,3)
    with model_graph.as_default():
        with tf_session.as_default():
            prediction=model.predict(imgOutput)

    predictedImage=modelLabel[str(np.argmax(prediction[0]))]

    context={'filePathName':filePathName,'predictedLabel':predictedImage[1]}
    return render(request,'index.html',context)