# Keras image classifier

Image classifier built with Keras/TensorFlow and deployed online using TensorFlow.js

## Description

I built a CNN model to classify images of my three favourites landmarks in Columbia University on the city of New York. With the use Keras aplications module I was 
able to use transfer learning by loading a pretrained model, [VGG16](https://neurohive.io/en/popular-networks/vgg16/), with weights from imagenet. This not only makes the training easier, but also improves greatly the
ability of the model to predict which landmark the image shows. I trained the model using an array of photos I took on campus using different angles and light settings.

The model was deployed using TensorFlow.js based on this [example](https://github.com/tensorflow/tfjs-examples/tree/master/mobilenet) and lives here: https://jacquelinearaya.github.io/portfolio/modelindex

### iPython Notebook:

Check [**tf_classifier.ipynb**](https://jacquelinearaya.github.io/2020/05/01/tf_classifier.html) notebook to see the data pipeline and model code.


Scholars Lion                    |    Alma Mater            | The Curl
:----------------------:|:------------------------:|:-----------:
 <img src="https://github.com/jacquelinearaya/Image_classifier_Keras/blob/master/lion.jpg" width="200" height="200"/></img>|<img src="https://github.com/jacquelinearaya/Image_classifier_Keras/blob/master/almamater.jpg" width="165" height="210" /></img>|<img src="https://github.com/jacquelinearaya/Image_classifier_Keras/blob/master/curl.jpg" width="220" height="200"/></img>
