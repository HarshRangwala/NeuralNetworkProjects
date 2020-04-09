# NN from scratch

To start with the easy stuff, we are going to initialize the network. <br>
Each neuron has a set of weights that need to be maintained. One weight for each input connection and additional weight for the bias. We use dictionary to represent neurons and store properties by names such as ‘weights’ for the weights. We have organized layers as arrays of dictionaries and treat the whole network as an array of layers. <br>
We have initialized the weights to small random numbers ranging from 0 to 1. <br>
The function <i>initialize_network()</i> creates a new neural network ready for training. It accepts three parameters, the number of inputs, the number of neurons to have in the hidden layer and the number of outputs. <br>
Running the example, the hidden layer has 1 neuron with 2 input weights plus the bias. The output layer has only two neurons. <br>
We can calculate the for output from a neural network by propagating the inputs to the hidden units at each layer and finally produce the output. It’s called forward-propagation. <br>
We first activate the neuron by providing an input. Neuron activation is calculated as the weighted sum of inputs. <br>
Moving on to the next stage, we need to transfer the activation to see what the neuron output really is. There is no definitive guide for which activation function works best on specific problems. It’s a trial and error process. 4 most commonly used activation functions are: <br>
*	<b>Sigmoid Function </b> (σ): `g(z) = 1.0 / (1.0 + e^{-z})`
The main reason why we use sigmoid function is because it exists  between (0 to 1) which makes it easier to interpret the output. It is beneficial because it prevents jumps in output values giving a smooth gradient. It is sometimes computationally expensive.
*	<b>Hyperbolic Tangent Function</b>: `g(z) =  (e^z -e^{-z}) / (e^z + e^{-z})`
The range of tanh function is from (-1 to 1). It is zero centered meaning that the negative inputs will be mapped strongly negative and the zero inputs will be mapped near zero in the tanh graph.
*	<b>Rectified Linear Unit (ReLU)</b>: `g(z) = max{0,z}`
It is computationally efficient because it allows the network to converge quickly. ReLU is the most used activation function in the world right now. It is used for almost all the convolutional neural networks. The issue with the function is that the derivative is not defined at z = 0, which we can overcome by assigning the derivative to 0 at z = 0. However, when the inputs approach zero, or are negative, the gradient of the function  becomes zero, the network cannot perform backpropagation and cannot learn. <br>

For this project, we are going to use sigmoid function. We can transfer an activation function using the sigmoid function as follows: <br>
	`Output = 1.0/(1.0 + exp(-activation)`
The function forward propagation is straightforward. The  implements the forward propagation for a row of data from our dataset with our neural network. The neurons output is stored in the neuron named ‘output’. The outputs for the layer is then stored in the list new_input which is later stored in the input. <br>
Further to minimize the error rate, we use backward propagation. <br>
The inputs arrive from a preconnected path. The input is modeled using randomly selected weights W. Then the output from output layer is compared with the desired output. The difference between the both is compared and travels back from the output layer to hidden layer to adjust the weights such that the error is decreased. <br>
Below is the function that implements backpropagation.  The calculated error for each neuron is stored in ‘delta’. The layers of the network are iterated in reverse order, starting at the output and working backwards. This ensures that the ‘delta’ values are calculated first which are later used by the hidden layer. <br>
Once the errors are calculated, they can be used to update weights. Network weight is updated as follows:<br>
	`Weight = weight + learning_rate * error * input`
Where weight is a given weight, learning_rate is a parameter that you must specify, error is the error calculated by the backpropagation procedure for the neuron and input is the input value that caused the error.<br>
During training the network, it needs a fixed number of epochs and within each epoch updating the network for each row in the training dataset. Because updates are made for each training pattern, this type of learning is called online learning. If errors were accumulated across an epoch before updating the weights, this is called batch learning or batch gradient descent. Once trained the final weights are printed and we can see that the error is decreasing. <br>
Below is a function named predict() that implements this procedure. It returns the index in the network output that has the largest probability. It assumes that class values have been converted to integers starting at 0.<br>
