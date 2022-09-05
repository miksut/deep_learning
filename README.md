# Deep Learning

This repository hosts some implementations of prominent Deep Learning concepts. These implementations come in form of Python scripts and can be found in the folder `src/`. Concretely, the folder contains the following scripts:

-  `00_perceptron.py`: Implementation of a simple perceptron (that operates on 2-dimensional data) and a learning rule. Concretely, the learning process begins by randomly initializing the perceptron's weights $\vec w$. Then, a training sample $\vec x$ and its associated class target $t \in$ {-1, 1} are chosen and the class is predicted: $y = \vec w^{T}\vec x$. If wrongly classified (i.e., $y \cdot t < 0$), the weights are updated as $\vec w = \vec w + t \cdot \vec x$.

- `01_gradient_descent.py`: Given the loss function $\mathcal{J}\_{\vec{w}} = w_1^2 + w_2^2 + 30 \cdot \sin(w_1) \cdot \sin(w_2)$, this script implements the gradient descent algorithm for an initial weight vector $\vec w$ and a given learning rate $\eta$. The goal is to investigate the effect of several initial $\vec w$ and different $\eta$ on the convergence process. This convergence process is illustrated in the figures below. The red point represents the initial weight vector $\vec w$ on the error surface. The green point stands for the final weight vector that results from the gradient descent.

   <img
    src="/imgs/01_gd_view1.png"
    width="250"
    align="left">
    
    <img
    src="/imgs/01_gd_view2.png"
    width="250">



