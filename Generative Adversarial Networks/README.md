# Demystifying Generative Adversarial Networks

This repository presents the an practical and conceptual overview of Generative Adversarial Networks.


## Defination

> *"The coolest idea in deep learning in the last 10 years." - Yann LeCun on GANs.*

Generative Adversarial Networks (GANs) have been greeted with real excitment since their creation back in 2014 by Ian Goodfellow and his research team. Here's the [<u>Orignal paper</u>](https://arxiv.org/abs/1406.2661). <br>
GANs have been used for various real-life applications for text/video/images generations, photos to emojis, clothing translation, video prediction and the list goes on. 
GANs belong to a set of <u>Generative models</u>.  These algorithms belong to the field of <u>unsupervised learning</u>, a type of machine learning algorithms which purports to uncover previously unknown patterns in data. Unsupervised learning works when there is no label given to the data. Genrative models are used to improve the real-world applications of machine learning.

## Training a Generative Adversarial Model

Lets supposes that we want to create a networ that can generate 200 images. How could we use adjust the network parameters so that the are forced to generate realistic images. We dont have any target to use <u>supervised learning</u>. Enter, Generative Adversarial Networks. Its a clever approach which is composed of two nets - 
* The first net, known as Generator, generates data similar to the expected one.
* The second net, known as Discriminator, tried to classify if an input data is real - belong to the  datasets or fake - generated.

Here's a training algorithm from the paper:
![Algorithm](https://github.com/HarshRangwala/NeuralNetworkProjects/blob/master/Generative%20Adversarial%20Networks/Training%20Algorithm.png)
It can be noticed from the above algorithm that both networks are trained seperately. First, a sample noise and a real-data set is used to train the dicriminator. We can keep generator fixed during the discriminator training phase. This way we can propagate the gradients, and update the dicriminator as it learns how to recognize the generators flaws and maximize the loss function.

## Math behind Generative Adversarial Model
A neural network <i><b>G(z)</b></i> is used to model the Generator mentioned above and the second net is <i><b>D(x)</b></i>.
![GAN Formula](https://github.com/HarshRangwala/NeuralNetworkProjects/blob/master/Generative%20Adversarial%20Networks/GanMath.png)

Individually the two nets play a very different roles. The role of Generator is mapping the input noise variables <b>z</b> to the desired data space <b>x</b> i.e. images in our case. Where as, Discriminator will output probability of 0.5 as the output of the first net is equivalent to the real data. It is very closely related to the [minimax algorithm](https://en.wikipedia.org/wiki/Minimax) where there are two players playing against each other in a battle and are determined to win the game.

## References

- [Introduction to GANS](https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f)

- [Implementing GAN using Numpy](https://towardsdatascience.com/only-numpy-implementing-gan-general-adversarial-networks-and-adam-optimizer-using-numpy-with-2a7e4e032021)

- [The Math behind GANs](https://towardsdatascience.com/the-math-behind-gans-generative-adversarial-networks-3828f3469d9c)
