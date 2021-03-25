# Keras image classifier

Image classifier built with Keras/TensorFlow and deployed online using TensorFlow.js

## Description

I built a CNN model to classify images of my three favourites landmarks in Columbia University on the city of New York. With the use Keras aplications module I was 
able to use transfer learning by loading a pretrained model, [VGG16](https://neurohive.io/en/popular-networks/vgg16/), with weights from imagenet. This not only makes the training easier, but also improves greatly the
ability of the model to predict which landmark the image shows.

The model was deployed using TensorFlow.js based on this [example](https://github.com/tensorflow/tfjs-examples/tree/master/mobilenet) and lives here: https://jacquelinearaya.github.io/classifier/

### iPython Notebook:

Check **tf_classifier.ipynb** notebook to see the data pipeline and model code.

<div class="row">
  <div class="column">
    <img src="https://github.com/jacquelinearaya/jacquelinearaya.github.io/blob/master/classifier/lion.jpg" width="200" height="200"/></img>
  </div>
  <div class="column">
    <img src="https://github.com/jacquelinearaya/jacquelinearaya.github.io/blob/master/classifier/almamater.jpg" width="200" height="300" /></img> 
  </div>
  <div class="column" >
    <img src="https://github.com/jacquelinearaya/jacquelinearaya.github.io/blob/master/classifier/curl.jpg" width="250" height="200"/></img>
  </div>
</div>
