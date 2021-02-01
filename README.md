# Neural Networks From Scratch

In this project I'll build a Neural Network from scrath <br>
This project is inspired by the youtuber sentdex who has a book on this topic, but I wanted to try to do it myself to better understand Neural Networks in general.
Everything will be in Python and maybe later I'll make a C++ version as well. <br>
Like the name suggests I'll write it from scratch, using simple python and only using numpy for things like transpose, random, dot product etc.

---

<h2> What is a Neural Network ? </h2>

A Neural Network is a learning system which resembles the human brain. The basic computational unit in the brain is the neuron, in a NN it is also called neuron or perceptron.
the biological neuron gets input signals through its dendrites and outputs a signal through its axon which connects to the dendrites of other neurons. The neuron from a NN behaves almost the same, it gets signals from neurons, sums those signals and sends an output signal to other neurons. Before sending the output signal, it applies an activation function (I'll explain it later).
<!--
<img height="150px" width="200px" align="left" src="https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.tutorialspoint.com%2Ftensorflow%2Ftensorflow_single_layer_perceptron.htm&psig=AOvVaw1GtMt1_dJD8vQeFeLOtkF0&ust=1612000169981000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCODkjO7uwO4CFQAAAAAdAAAAABAy" />
<br>
Neurons are arranged in layers: input layer, hidden layers and output layer.
The input layer takes the input of the model and feeds it to the next layer through connections named weights, and so on, until it reaches the output layer, where it will generate predictions -->

---

<h2> Structure </h2>
  
<h3> The layers </h3>

Every layer will have their own set of biases, and a set of weights between them and the prior layer
This set of weights is different for every pair of layers, it has random values in the start, but this values will be changed after each backpropagation process.
    
<h4>Layer.py</h4>
```
No language indicated, so no syntax highlighting. 
But let's throw in a <b>tag</b>.
```
    

---

<h2>Some resources that helped me in understanding more about NN:</h2>
     sentdex's playlist: https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3
<br> sentdex's book: https://nnfs.io/
<br> A series of videos about NNs by 3Blue1Brown: <br> https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
<br> Michael Nielsen's book on the topic: http://neuralnetworksanddeeplearning.com/chap1.html
