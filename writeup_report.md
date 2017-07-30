#**Behavioral Cloning** 

## Writeup

---

**Behavioral Cloning Project**

The steps in developing this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road


[//]: # (Image References)

[image1]: ./data_i/model.png "Model Visualization"
[image2]: ./data_i/c1.jpg "Center Image"
[image3]: ./data_i/r1.jpg "Recovery Image"
[image4]: ./data_i/r2.jpg "Recovery Image"
[image5]: ./data_i/r3.jpg "Recovery Image"
[image6]: ./data_i/n1.jpg "Normal Image"
[image7]: ./data_i/f1.png "Flipped Image"


---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* nn_nb.ipynb containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_n_1.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model_n_1.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model I employed is the standard [Nvidia model](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). The link has further information about the model. The model is implemented in lines 46-63 of Nvidia model block in [nn_nb.ipnb](https://github.com/pkorivi/behavioral_cloning/blob/master/nn_nb.ipynb)

The data is normalized, images are cropped 50 pixels on top and 20 pixels on bottom to remove redundant information, RELU layers are employed to introduce nonlinearity and dropout to reduce the over fitting.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting lines 59,61 of [nn_nb.ipnb](https://github.com/pkorivi/behavioral_cloning/blob/master/nn_nb.ipynb).

The model was trained and validated on different data sets such as driving straint in clockwise, anti clockwise, recovery paths for the vehicle etc to ensure that the model was not overfitting or underfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually lines 65 of [nn_nb.ipnb](https://github.com/pkorivi/behavioral_cloning/blob/master/nn_nb.ipynb).
There were not many parameters to tune and the model performed well without need for much tweaking.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, driving in anti clock wise direction, extra data for driving in corners and bridge. This might have been the reason for the car to drive smoothly in curves and sway a little in strait roads.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture is as following:

My first step was to use a convolution neural network model similar to the Lenet Architecture, I thought this model might be appropriate because the model needed to learn about the road strcuture and few more details. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. The initial model gave a good performance with few weak points. To combat this I have gathered data to help drive the car in the weak points. This improved the performance to good extant but there was still error in some corners and every retraining changed these weak spots. I tried adding further data sets but it didn't improve the performance. 

Then I introduced the Nvidia model which is deeper and can provide better generalization. The model gave good performance. Dropouts were introduced to counter overfitting of data.

The data collected previously with Lenet model from the corners and other regions weak spots helped the Nvidia architecture to perform good with least changes. Higher speeds(>25) caused the car to leave track in some regions, gave good performance at lower speeds.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.


#### 2. Final Model Architecture

The final model architecture in lines 46-63 in [nn_nb.ipnb](https://github.com/pkorivi/behavioral_cloning/blob/master/nn_nb.ipynb) is a Nvidia model.

Here is a visualization of the architecture with details about the layer sizes.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like:

![alt text][image3]
![alt text][image4]
![alt text][image5]

Track 2 data didnt improve driving on track 1, so I omitted for current submission. 

To augment the data sat, I also flipped images and angles thinking that this would help car handle situations of rught turns on similar track. If this is not employed the car will be biased towards left turns. For example, here is an image that has then been flipped:

![alt text][image6]

![alt text][image7]


After the collection process, I had 88k number of data points that are used for training the model.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as evidenced in the code. I used an adam optimizer so that manually tuning the learning rate wasn't necessary.
