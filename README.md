# Neural Networks From Scratch

In this project I'll build a Neural Network from scrath <br>
This project is inspired by the youtuber sentdex who has a book on this topic, but I wanted to try to do it myself to better understand Neural Networks in general.
Everything will be in Python and maybe later I'll make a C++ version as well. <br>
Like the name suggests I'll write it from scratch, using simple python and only using numpy for things like transpose, random, dot product etc.

---

<h1> What is a Neural Network ? </h1>

A Neural Network is a learning system which resembles the human brain. The basic computational unit in the brain is the neuron. In a NN it is called neuron or perceptron or nodes.
The biological neuron gets input signals from a number of neurons and based on those signals it outputs a signal to other neurons. The perceptron behaves almost the same, it gets signals from neurons int the prior layer, sums those signals, applies an activation function on the sum and sends an output signal to the neurons in the next layer. 
<!--
<img height="150px" width="200px" align="center" src="https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.tutorialspoint.com%2Ftensorflow%2Ftensorflow_single_layer_perceptron.htm&psig=AOvVaw1GtMt1_dJD8vQeFeLOtkF0&ust=1612000169981000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCODkjO7uwO4CFQAAAAAdAAAAABAy" />
<br>
-->

---

<h1> Structure </h1>
  
<h2> The layers </h2>

A Neural Network is built with layers (minimum 2), and each layer contains a given number of neurons. <br>
The first layer is the Input Layer, the last one is named Output Layer and the layers between the two are named Hidden Layers. <br>
Neurons from one layer are connected with neurons from the prior and the next layer, each connection has a weight which influences the values carried from one neuron to another.
Also, each neuron has a bias, which can adjust the value before applying an activation function.

<!--
[![Neural Net](https://developers.google.com/machine-learning/crash-course/images/1hidden.svg)](height="300px" width="500px")
<img height="150px" width="200px" align="left" src="https://developers.google.com/machine-learning/crash-course/images/1hidden.svg" />
<br>
-->

In the code below, I made the class Dense (the same name like the one from keras) which represents the layer.
It takes ```layerSize``` (how big the layer will be), ```weightBounds``` (the interval for the weight values) and ```activation``` (what function will be applied on this layer before going to the next layer in the NN).
    
<h4>Layer.py</h4>

 ```python
 class Dense:
    def __init__(self, layerSize, activation="sigmoid", weightBounds=(-1, 1)):
        self.length = layerSize
        self.bounds = weightBounds
        self.activation = activation
        self.derivative = activation

        if self.activation == "relu":
            self.derivative = ReLU_prime
            self.activation = ReLU
        elif self.activation == "tanh":
            self.activation = tanh
            self.derivative = tanh_prime
        elif self.activation == "softmax":
            self.activation = softmax
            self.derivative = lambda x: 1
        else:
            self.activation = sigmoid
            self.derivative = sigmoid_prime
```

<h2> Activation functions </h2>

In more scientific terms, an activation function is a "mathematical formalism that is used to approximate the influence of an extracellular field on an axon or neurons", but basically we aplly a function on a neuron to help it decide what should be fired to next neurons. Maybe the examples will make it easier to understand: <br>

  <h3>Sigmoid Function:</h3>
  <p>The sigmoid function converts the sum to a value between 0 and 1.<br>
  The formula is: <br>
          
  ```python
     def sigmoid(x):
         return 1 / (1 + np.e**(-x))
  ```
         
  <br>
  This is how it looks:</p>
  <br>
  <img height="400px" width="700px" align="center" src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/320px-Logistic-curve.svg.png" />
  
  <h3>ReLU Function:</h3>
  <p>The Rectified Linear Unit, or ReLU, is a function that is easier to compute than a function like sigmoid while working a little better (in some cases). It basically outputs the maximum between a value and 0.<br>
  The formula is: <br>
          
  ```python
     def relu(x):
         return np.maximum(x, 0)
  ```
         
  <br>
  This is how it looks:</p>
  <br>
  <img height="400px" width="700px" align="center" src="https://classic.d2l.ai/_images/output_mlp_699d0d_3_0.svg" />
  
  <h3>Tanh Function:</h3>
  <p>The tanh function outputs values between -1 and 1.<br>
  The formula is: <br>
          
  ```python
     def tanh(x):
         return (np.e**x - np.e**(-x))/(np.e**x + np.e**(-x))
  ```
         
  <br>
  This is how it looks:</p>
  <br>
  <img height="400px" width="700px" align="center" src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Hyperbolic_Tangent.svg/1280px-Hyperbolic_Tangent.svg.png" />

<h2> Hyperparameters </h2>

Hyperparameters are variables that determine how the network is trained. The ones that I'll focus on are the learning rate, batch size and epochs.

<h3>Learning rate</h3>

It defines how fast a network is updating its parameters. A too low learning rate slows the learning process but may take a lot more to converge, and a large learning rate speeds up the learning process but it may not converge.

<h3>Epochs</h3>

An epoch determines how many times the network will go over the training data.

<h3>Batch size</h3>

The batch size is the number of samples given to the network before updating the parameteres.

Most commonly used values are 32, 64, 128, 256, but it varies from model to model.


<h1> How does it work?</h1>
      <h2>Feed forward:</h2>
     
We calculate the value for a neuron by summing the values of each of the nodes from the previous layer multiplied by their respective weight than adding the bias. After that we apply the activation function and our neuron is ready to send its value to the next layer.<br>
The formula is:
  
y = σ (w * x + b) , where:
                
- y is the current node
- σ is the activation function
- x is a vector containing the values of the previous layer
- w is a vector containing the weights (wi is the weight between xi and y)
- b is the bias. 
      
Here w is a vector because we have only one node, but when we'll work with an entire layer, w is a matrix.
      <!--
      As I mentioned earlier, neurons are connected with conections that have weights with the neurons from the previous layer, also, each of them has a bias. 
      -->
<h2>Backpropagation:</h2>


After we used feed-forward on all our layers, the output layer will have values which are usually not the same as the expected output. Here comes the "learning" part, where we use the values from all the layers to update all of our weights and biases. We calculate the cost, which is the difference between the predicted and the expected output, than we go from that last layer to the first one, like in the feed-forward faze, but now backwards.

Here comes a little bit of math. Because we used activation functions on our layers, now going back, we use the derivatives of those functions, applied to the values of the layer with respect to the weights/bises, than we update the weights/biases by substracting the result from the derivative, multiplied by the learning rate, α, often divided by the batch size. To help the model train better and not overfit, we do this process at the end of every batch.

The backpropagation process can be a little difficult to understand, but I hope it'll be clearer in the code section.

<h2> Training </h2>

This is the part where all the things explained earlier are used together to help the NN "learn" to complete some task, like recognizing digits, dogs/cats, human faces, lines/stop signs and other stuff for self-driving, predicting house prices, and a lot more.

This process requires a dataset, which will be split in train/test sets and sometimes its better to also have a validation set. The train set contains the samples we'll give to the NN to "train" it, than we use the validation set to see how well it behaves and after it went through all the samples from train set, we give it the test set to see how well it does. This is an epoch and we do this process how many times we want by setting the ```epochs``` hyperparameter. But, by doing this we will get the same result after every 
epoch. That's where we use backpropagation, to update the weights and the biases, so the NN will give different results, maybe better, maybe worse. Because it takes a lot of computation, we'll apply backpropagation after a number of samples (not after every sample), and this number is represented by the ```batch size```. A batch is like a mini dataset, with samples from the train set. We do the usual thing, go through each sample, apply feedforward, we get the result, and after we completed the batch, we apply backpropagation. This helps because, for example, if we have 3200 samples, instead of doing backprop for 3200 times, with a batch size of 32 (which is the most used value), we only apply the process 100 times, which is a lot faster and gives better results.




---

<h2>Some resources that helped me in understanding more about NN:</h2>
     sentdex's playlist: https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3
<br> sentdex's book: https://nnfs.io/
<br> A series of videos about NNs by 3Blue1Brown: <br> https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
<br> Michael Nielsen's book on the topic: http://neuralnetworksanddeeplearning.com/chap1.html
