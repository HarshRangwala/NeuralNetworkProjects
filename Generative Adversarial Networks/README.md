# Demystifying Generative Adversarial Networks

This repository presents the an practical and conceptual overview of Generative Adversarial Networks.


## Defination

> *"The coolest idea in deep learning in the last 10 years." - Yann LeCun on GANs.*

Generative Adversarial Networks (GANs) have been greeted with real excitment since their creation back in 2014 by Ian Goodfellow and his research team. Here's the [<u>Orignal paper</u>](https://arxiv.org/abs/1406.2661).
GANs have been used for various real-life applications for text/video/images generations, photos to emojis, clothing translation, video prediction and the list goes on. 
GANs belong to a set of <u>Generative models</u>.  These algorithms belong to the field of <u>unsupervised learning</u>, a type of machine learning algorithms which purports to uncover previously unknown patterns in data. Unsupervised learning works when there is no label given to the data. Genrative models are used to improve the real-world applications of machine learning.

## Training a Generative Adversarial Model

Lets supposes that we want to create a networ that can generate 200 images. How could we use adjust the network parameters so that the are forced to generate realistic images. We dont have any target to use <u>supervised learning</u>. Enter, Generative Adversarial Networks. Its a clever approach which is composed of two nets - 
* The first net, known as Generator, generates data similar to the expected one.
* The second net, known as Discriminator, tried to classify if an input data is real - belong to the  datasets or fake - generated.

## Math behind Generative Adversarial Model
[Formula](https://github.com/HarshRangwala/NeuralNetworkProjects/blob/master/Generative%20Adversarial%20Networks/GanMath.png)

