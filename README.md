# Neural Networks From Scratch

In this project I'll build a Neural Network from scrath <br>
This project is inspired by the youtuber sentdex who has a book on this topic, but I wanted to try to do it myself to better understand Neural Networks in general.
Everything will be in Python and maybe later I'll make a C++ version as well. <br>
Like the name suggests I'll write it from scratch, using simple python and only using numpy for things like transpose, random, dot product etc.

---

<h2> What is a Neural Network ? </h2>

A Neural Network is a learning system which resembles the human brain. The basic computational unit in the brain is the neuron. In a NN it is called neuron or perceptron or nodes.
The biological neuron gets input signals from a number of neurons and based on those signals it outputs a signal to other neurons. The perceptron behaves almost the same, it gets signals from neurons int the prior layer, sums those signals, applies an activation function on the sum and sends an output signal to the neurons in the next layer. 
<!--
<img height="150px" width="200px" align="center" src="https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.tutorialspoint.com%2Ftensorflow%2Ftensorflow_single_layer_perceptron.htm&psig=AOvVaw1GtMt1_dJD8vQeFeLOtkF0&ust=1612000169981000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCODkjO7uwO4CFQAAAAAdAAAAABAy" />
<br>
-->

---

<h2> Structure </h2>
  
<h3> The layers </h3>

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

<h3> Activation functions </h3>

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
 <h2> How does the it work?</h2>
      <h3>Feed forward:</h3>
     
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
      
---

<h2>Some resources that helped me in understanding more about NN:</h2>
     sentdex's playlist: https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3
<br> sentdex's book: https://nnfs.io/
<br> A series of videos about NNs by 3Blue1Brown: <br> https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
<br> Michael Nielsen's book on the topic: http://neuralnetworksanddeeplearning.com/chap1.html
