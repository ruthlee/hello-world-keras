# Understanding Keras using MNIST Data

In this notebook we use the keras package to train a neural network to recognize handwritten digits in the MNIST data. We'll first train a simple neural network, then try to improve on the results using a convolutional neural network, gaining accuracy of about 98% by the end of the notebook- not bad for a novice! Probably the most important part of the notebook are my notes about convolutional neural networks, reproduced below: 

This is pretty good. With our simple dummy example and only 5 epochs (which means we have to wait a bit... not great) we got to an accuracy of about 0.98. Let's try using convolutional neural networks (convnets), which are optimized for visual inputs. 

## Convolutional Neural Nets Overview
Unlike a regular neural network, the layers of a convnet are three-dimensional with neurons arranged by width, height, and depth. There are three types of layers: 
- Convolution Layer
- Pooling Layer
- Fully-connected layer

and three hyperparameters we have to specify for the model: 
- kernel/filter size
- filter count
- stride
- padding

Let's go over each of the layers. 

**Convolution layers** apply what is known as a filter or a kernel to parts of a three-dimensional input image to produce a feature map. There are great visualizations [here](https://medium.freecodecamp.org/an-intuitive-guide-to-convolutional-neural-networks-260c2de0a050) and [here](https://towardsdatascience.com/applied-deep-learning-part-4-convolutional-neural-networks-584bc134c1e2). In advance, we specify the number (**filter count**) and size (**filter size**) of the filters for the layer. We can imagine "sliding" each filter over our 3D input to produce a 1x1x1 output for the feature map (this is the operation of convolution). Each filter will produce a "slice" of the feature map that we put together to get the full feature map. The **stride** of the layer refers to the number of units by which we move the filter at each pass. A higher stride means we have less overlap of the units in each filter. Also note that a higher stride means the resulting feature map is smaller. This is where **padding** comes in. In order to maintain the size of the feature map in the layer (which has been shown to improve outputs), we can pad the input with enough zeros (in essence, making it artificially bigger) so that when we pass the filter the resulting feature map is the same size as the input. 

It's important to note that when we get the resulting feature map, we must pass the results through an **activation function** (such as the sigmoid function and relu) in order to get nonlinear results.

The **pooling layer** is used for dimensionality reduction and to prevent overfitting. To go along with our stride, we choose a **pooling window**, which is the size of the chunks of the feature map we're going to analyze. A common pooling method (there are others) is max-pooling, where we divide our feature map into chunks, each the size of the pooling window, and just choose the max value in each chunk to keep as the input for the next layer. For example, a 2x2 pooling window and stride 2 will divide our input into 2x2 chunks and keep only $\frac{1}{4}$ of our original data.

Finally, after our convolution and pooling layers we use **fully-connected layers** to get a single output from our convnet. This is the same full-connected layer in a normal neural network, created from the unrolleld outputs of the convolution/pooling layers. 

In terms of making architectural choices, [this](http://cs231n.github.io/convolutional-networks/#architectures) source recommends keeping filter sizes small (3x3 or 5x5), using padding, and setting stride = 1. Output dimensions should be divisible by 2 many times (32, 64, 96).

Now that we have a basic idea about what's going on, let's implement in keras. A few quick notes: 
- ```padding = 'same'``` indicates that we want to use a padding of zeros to keep the feature map from shrinking
- For ```input_shape```, we need this to be (28, 28, 1) so that it's 3-dimensional like the other layers. When we fit the model it will expect a four-dimensional input, i.e., (number_images, 28, 28, 1). 

**Another important note: convergence is MUCH faster after normalization (see above cell). Before normalization accuracy was hovering around 0.10 during the first epoch** 
