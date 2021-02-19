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
  The implementation is: <br>
          
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
  The implementation is: <br>
          
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
  The implementation is: <br>
          
  ```python
     def tanh(x):
         return (np.e**x - np.e**(-x))/(np.e**x + np.e**(-x))
  ```
         
  <br>
  This is how it looks:</p>
  <br>
  <img height="400px" width="700px" align="center" src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Hyperbolic_Tangent.svg/1280px-Hyperbolic_Tangent.svg.png" />
  
  <h3>Softmax Function:</h3>
  <p>It outputs values between 0 and 1. The softmax function is usually used on the output layer.<br>
  The implementation is: <br>
          
  ```python
     def softmax(layer):
         exp = np.exp(layer)
         return exp/sum(exp)
  ```
         
  <br>
  This is how the equation looks like:</p>
  <br>
  <img height="150px" width="80%" align="center" src="https://miro.medium.com/max/1706/0*JJyPQsmvH5nq48xx.png" />
 
 
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

<h1> Putting it all together</h1>

In this section I'll show how all the information above is implemented in code, using python. Some things are different from the usual approach because I tried not to rely too much on others implementation.

So, the first thing would be the layers, and as I mentioned earlier, I created a class for them, named ```Dense``` and for each layer we'll remember the size, the activation function and the bounds for the weights.

After that we have the code for the functions, I showed those before but here's all of them

```python
def ReLU(x):
    return np.maximum(0, x)

def ReLU_prime(x):
    return (x > 0)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return (np.e**x - np.e**(-x))/(np.e**x + np.e**(-x))

def tanh_prime(x):
    return 1-tanh(x)**2

def softmax(layer):
    exp = np.exp(layer)
    return exp/sum(exp)
```

And now we have the hard part, the neural network class itself, named ```NeuralNetworkModel```

First things first, the initialization of the model. We take as parameters the size of the input layer and the output layer. The input size is added to a sizes list where it will store the size of each layer, than we have a layers list where we'll store the layers as we add them, and than the weights and biases lists where we'll store all the weights and biases from each layer, in order. I could've kept them within the Dense class but this way they are more accessible. I also added a layer full of zeros, that's just so I know that layer 0 is the input layer, and this layer will be replaced with an actual layer in the feedforward process.

```python
def __init__(self, inputSize, outputSize):
    self.x = inputSize
    self.y = outputSize
    self.sizes = [inputSize]
    self.layers = [np.zeros((1, inputSize))]  # this stores the layers
    self.weights = []
    self.biases = []

```

Now we need layers. We have the ```add``` function which takes a layer as input, adds it to the layers list, adds its size to the sizes list, and with its size creates random weights and biases which will be stored in their respective lists.


```python
def add(self, newLayer):  # like the name suggests, it adds layers to the net
    self.layers.append(newLayer)
    self.sizes.append(newLayer.length)
    self.weights.append(np.random.randn(self.sizes[-1], self.sizes[-2]))
    self.biases.append(np.ones((self.sizes[-1], 1)))

```

After that, we start implementing the fun stuff, starting with the ```feedforward``` method. As you can see, it takes the input layer which is given to the x variable. Now it goes through the layers list and we apply the ecuation ```x = σ (w * x + b)``` on each layer, and at the end, x will remain a layer with the values for the output layer.

```python
def feedforward(self, input):  # The calculations are done for each of the layers
    x = np.transpose([input])
    for i in range(1, len(self.layers)):
        x = self.layers[i].activation(np.dot(self.weights[i-1], x) + self.biases[i-1])
    return x
```

And now, the ```backpropagation``` function, which is the most important process that with which our NN learns.

First, we initialise the list that will represent the changes in weights/biases which will be returned at the end, a list ```outputs``` which keeps the output of each layer, the ```layer``` variable keeps the values of the current layer and the ```zs``` list keeps the values of each layer before applying the activation function.

Now we apply feedforward to every layer, from input to output layers, storing values in the lists initialised before. After that, we go through the NN again, but now from the last layer to the first, applying this operations:

Output layer (special case):
- we start with the output layer and take the cost function, which is the values of the output layer after feeding forward minus the actual values our output layer should have
- the results will be kept in the ```delta``` variable (list), this values will represent the change in the output layer's biases and the for the change in weights we take the dot product between delta and the output from the previous layer

The rest of the layers:
- for the rest, we apply something similar
- let's say we are on layer k
- now delta will be the dot product between layer (k+1)'s weights and the current value of delta times the derivative of the output of this layer
- this delta is given to the change in the biases of the layer k
- the change in the layer k's weights will be the dot product of delta and the output of layer (k-1)

In the code below you'll see .T added to some list, this is because of the dot product, which is matrix multiplication, so some dimensions must be the same


```python
def backprop(self, x, y):
    b_change = [np.zeros(b.shape) for b in self.biases]
    w_change = [np.zeros(w.shape) for w in self.weights]
    outputs = [x]
    layer = x
    zs = []
    for (l, w, b) in zip(self.layers[1:], self.weights, self.biases):
        z = np.dot(w, layer) + b
        layer = l.activation(z)
        zs.append(z)
        outputs.append(layer)
    delta = np.array(layer - y) * self.layers[-1].activation(zs[-1])
    b_change[-1] = delta
    w_change[-1] = np.dot(delta, outputs[-2].T)
    for l in range(2, len(self.layers)):
        prime = self.layers[-l].derivative(zs[-l])
        delta = np.dot(self.weights[-l+1].T, delta) * prime
        b_change[-l] = delta
        w_change[-l] = np.dot(delta, outputs[-l-1].T)
    return w_change, b_change
```

Now we'll use the backprop to train the model, and we'll train it in batches so it doesn't overfit, nor is it biased. For that I created the ```update_batch``` method which will take a batch of samples from the train set, apply FF and BP on each sample, sum the changes in weights and biases, and only after the batch is complete, add this changes to the actual weights/biases.

```python
def update_batch(self, batch, alpha):
    b_change = [np.zeros(b.shape) for b in self.biases]
    w_change = [np.zeros(w.shape) for w in self.weights]
    for (a, y) in batch:
        x = np.transpose([a])
        dw, db = self.backprop(x, y)
        b_change = [bc+dbc for bc, dbc in zip(b_change, db)]
        w_change = [wc+dwc for wc, dwc in zip(w_change, dw)]
    self.weights = [w - (alpha / len(batch)) * ndw for (w, ndw) in zip(self.weights, w_change)]
    self.biases = [b - (alpha / len(batch)) * ndb for (b, ndb) in zip(self.biases, b_change)]
```

The final part, the training process which combines all the methods presented and with the help of a dataset, "learns" to recognize patterns in the data.
So, we have the ```train``` method which takes the inputs: a train set, the number of epochs, the learning rate, the batch size and a test set.

So we iterate through the train set for a number of times equal to the epochs variable. Before anything, we shuffle the data, we split the data in batches and for each batch we apply the ```update_batch``` method, which is explained earlier. To see some results after each epoch, we apply the current NN on the test set to see how it behaves, to know if it increased in accuracy since the last epoch, or it is equal or worse. But, if it increased it means it is learning and now we can change some things like the hyper-parameteres or the layout to try and increase the accuracy.

```python
def train(self, train_set, epochs, alpha, batch_size, test_set):
    for i in range(epochs):
        random.shuffle(train_set)
        batches = [train_set[k:k+batch_size] for k in range(0, len(train_set), batch_size)]
        for batch in batches:
          self.update_batch(batch, alpha)
        result = 0
        for x, y in test_set:
           output = self.feedforward(x)
           result += int(np.argmax(output) == y)
        print("Epoch {0}: {1} / {2}".format(i + 1, result, len(test_set)))
```

After this, my network has two more methods, a ```save``` and ```load```. With save, we give it a .txt file in which it will write it's layout and the values for the weights and biases so we can use it on other machines. With load, we take all the information from the .txt file that we give it and the network copies all the info from that file, and so it will be a copy of the one we saved on that file. I won't put the code here because it is very long and the readme is already pretty long. 

---

<h2>Some resources that helped me in understanding more about NN:</h2>
     sentdex's playlist: https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3
<br> sentdex's book: https://nnfs.io/
<br> A series of videos about NNs by 3Blue1Brown: <br> https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
<br> Michael Nielsen's book on the topic: http://neuralnetworksanddeeplearning.com/chap1.html
