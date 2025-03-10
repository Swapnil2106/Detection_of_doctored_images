# Building Android Application To Detect Doctored Images Using Deep Learning Model

## Problem Statement
Develop an Android app that can detect digitally modied photos in multiple formats (Jpeg,
png) and anticipate whether the output will be Authentic or Doctored based on the image
states.

## Objectives
1. Develop an Android app that can detect digitally modied photos in multiple formats (Jpeg,
png) and anticipate whether the output will be Authentic or Doctored based on the image
states.
2. Model should be capable of supporting multiple image format.
3. Developing android application.

## Implementation

### ELA Method

![image](https://github.com/user-attachments/assets/79e68b20-2d0e-4e63-b4ee-4c3428ab9597)

This graphic shows the full architecture of the proposed model. There are two components:
Learning the Image feature - The fundamental characteristics of false pictures are learned.
Softmax is utilised as a classification device that uses the fused characteristics to classify the
picture.

We have utilised the last particular CNN model named VGG-16 to extract the latent char-
acteristics of the pictures.Although normal pictures in the dataset are utilised, their ELA
images are used in the pre-processing step.ELA emphasises the compression features inside a
picture.The application of any image processing filter helps improve the capacity for general-
isation and speeds up the convergence of the deep learning networks.

In order to produce image embedding from the output of their third to their last cap-
tion, ELA pictures are transmitted to vgg-16 pretrained model and transfer knowledge from
vgg16.In order to understand the picture properties, the image embeddedings are passed to
two layers with completely linked thick layers. Here, functional vectors from the modes are
learnt and the final Softmax classification is passed.The Softmax forecasts that
false pictures are probable to appear

### Proposed Methodology

![image](https://github.com/user-attachments/assets/34f4426d-b90d-4033-983d-868f00d32738)

The model uses (224 x 224 x 3) as the input image size, which we set in the pre-processing
section, with the 3 referring to the RGB colour space.

The first two layers of the model have the same padding and have 64 channels of 3*3 filter
size. Then, after a stride (2, 2) max pool layer, two layers of convolution layers of 256 filter
size and filter size (3, 3). This is followed by a stride (2, 2) max pooling layer, which is the
same as the previous layer. Following that, there are two convolution layers with filter sizes
of 3 and 3 and a 256 filter. Following that, there are two sets of three convolution layers, as
well as a max pool layer. Each has 512 filters of the same size (3, 3) and padding.Following
that, the image is sent to a stack of two convolution layers. The filters we utilise in these
convolution and max pooling layers are 3*3 in size. After that, we receive a feature map (7
X 7 x 512). This is flattened and then connected to the fully connected layers, and then we
added a softmax layer with 2 classes to classify it (Authentic or Doctored).


## Modules
### Module 1 : Data Preprocessing
We deleted certain unsupported image formats and transformed the rest of the images to a
fixed size of (224 X 224) after obtaining the dataset.

### Module 2 : Data Preparation with ELA
The preprocessed data is then delivered to the Error Level Analysis, where each image is
processed in order to gain a clear knowledge of image parts with varied compression levels.

### Module 3 : Data Preparation
Following the completion of the ELA, the preprocessed data is separated into two portions
in an 80:20 ratio: train and validation datasets.

### Module 4 : Model Definition and Training
The VGG-16 model, or picture classification model, was chosen because it had a 92:7%
accuracy on the ImageNet dataset. This model will be used to distinguish between two types
of information: authentic and doctored.
Adam Optimizer was used to optimise the learning rate. We used binary cross entropy as
a loss function, which is a frequent choice for classification models. With a batch size of 32,
we trained our model for 25 epochs.
Now we stored the model in.h5 file and tested it with several sample images to see if it can
predict whether an image is doctored or legitimate, as well as the value of how doctored or
authentic it is.
Later, we changed our model to tflite in order to deploy or use it in the creation of Android
applications.

### Module 5 : Android app
We used the tflite model we created previously, as well as the labels.txt file, to create
an android app that accepts an image as input and determines if the image is authentic or
doctored.

## Results

![image](https://github.com/user-attachments/assets/1d1d3a52-185d-452a-990d-4a66a464740d)

![image](https://github.com/user-attachments/assets/863f52e6-4135-461a-8108-dfa4ab8d7392)

![image](https://github.com/user-attachments/assets/2de36903-3d0a-49a1-b9f2-6425f04a41ab)


## Dataset

dataset:- kaggle.com/divg07/casia-20-image-tampering-detection-dataset

![image](https://github.com/user-attachments/assets/6eaa6952-e5cf-4518-903b-7a1c1c7dacb8)


## Conclusion

We successfully constructed a Deep Learning-based classifier and an Android application
in this project, which can recognise and classify any image into Authentic or Doctored. As a
result, it performs the function of a Doctored Image Detection System (DIDS).
To categorise the photos, the current model employs a Deep Learning approach. The VGG-
16 model is created and saved as a `.h5' file. The model's architecture, weights values, and
compilation information are all contained in the h5 file. As a result, the trained model may be
quickly deployed on any edge device or network interface for real-time picture classification.For
this, we created an Android app that allows users to upload images and determine whether
they are authentic or doctored. The model also indicates whether it is Authentic or Doctored
by a percentage or a value.
We now trained the model on the CPU for 3 epochs, but with a high compute machine, we
can train the model for more epochs, increasing the model's accuracy while lowering data loss
throughout the training process.






