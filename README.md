# Deep Learning

This repository hosts some implementations of prominent Deep Learning concepts. These implementations come in form of Python scripts and can be found in the folder `src/`. Concretely, the folder contains the following scripts:

-  `00_perceptron.py`: Implementation of a simple perceptron (that operates on 2-dimensional data) and a learning rule. Concretely, the learning process begins by randomly initializing the perceptron's weights $\vec w$. Then, a training sample $\vec x$ and its associated class target $t \in$ {-1, 1} are chosen and the class is predicted: $y = \vec w^{T}\vec x$. If wrongly classified (i.e., $y \cdot t < 0$), the weights are updated as $\vec w = \vec w + t \cdot \vec x$.

- `01_gradient_descent.py`: Given the loss function $\mathcal{J}\_{\vec{w}} = w_1^2 + w_2^2 + 30 \cdot \sin(w_1) \cdot \sin(w_2)$, this script implements the gradient descent algorithm for an initial weight vector $\vec w$ and a given learning rate $\eta$. The goal is to investigate the effect of several initial $\vec w$ and different $\eta$ on the convergence process. This convergence process is illustrated in the figures below. The red point represents the initial weight vector $\vec w$ on the error surface. The green point stands for the final weight vector that results from the gradient descent.

   <img
    src="/imgs/01_gd_view1.png"
    width="200"
    align="left">
    
    <img
    src="/imgs/01_gd_view2.png"
    width="200">
 
- `02_regression_gd.py`: Starting from 1-dimensional noisy linear training data, a linear unit $y = w_0 + w_1 \cdot x$, and its loss function $\mathcal{J}\_{\vec w}(X) = \frac{1}{N} \Sigma_{n=1}^N(y^{n} - t^{n})^{2}$, the script implements gradient descent. The idea is to fit a line to the generated data.

- `03_two_layer_network.py`: Implementation of a 2-layer network (using a logistic activation function) that operates on one input $x$ and produces one output $y$. The implementation comprises functions to compute the loss and the gradient for a given training dataset, as well as a function to perform iterative gradient descent. The goal is to train a network that fits to the training data. The following figure shows a possible result of this fitting process. Considering the left plot, it becomes clear that the network learned a function (i.e., the blue line) that well approximates the training data (i.e., the red 'x'). The right plot exemplifies how the network loss progressed as a function of the number of training epochs.

   <img
    src="/imgs/03_two_layer_network.png"
    width="200"
    align="left">


 
 



