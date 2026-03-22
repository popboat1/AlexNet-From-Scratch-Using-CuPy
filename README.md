This is an AlexNet Convolutional Neural Network that I built entirely from scratch using **CuPy** and trained on CIFAR-10 images. 

**You can see the live 3D preview here:** [https://popboat1-alexnet-visualizer.hf.space](https://popboat1-alexnet-visualizer.hf.space)

Writing the forward passes, backpropagation, and architecture from the ground up in CuPy was an incredible way to truly understand the math behind CNNs. 

However, full transparency: since I don't have the massive computation power required to train with the full ImageNet dataset, and because training the from-scratch AlexNet takes a *very* long time, the live model served on the web is actually inferred using **TensorFlow/Keras**. 