# Deep Learning

This repository hosts some implementations of prominent Deep Learning concepts. These implementations come in form of Python scripts and can be found in the folder `src/`. Concretely, the folder contains the following scripts:

-  `00_perceptron.py`: Implementation of a simple perceptron (that operates on 2-dimensional data) and a learning rule. Concretely, the learning process begins by randomly initializing the perceptron's weights $\vec w$. Then, a training sample $\vec x$ and its associated class target $t \in$ {-1, 1} are chosen and the class is predicted: $y = \vec w^{T}\vec x$. If wrongly classified (i.e., $y \cdot t < 0$), the weights are updated as $\vec w = \vec w + t \cdot \vec x$.

- `01_gradient_descent.py`: Given the loss function $\mathcal{J}\_{\vec{w}} = w_1^2 + w_2^2 + 30 \cdot \sin(w_1) \cdot \sin(w_2)$, this script implements the gradient descent algorithm for an initial weight vector $\vec w$ and a given learning rate $\eta$. The goal is to investigate the effect of several initial $\vec w$ and different $\eta$ on the convergence process. This convergence process is illustrated in the figures below. The red point represents the initial weight vector $\vec w$ on the error surface. The green point stands for the final weight vector that results from the gradient descent.

   <img
    src="/imgs/01_gd_view1.png"
    height="200"
    align="left">
    
    <img
    src="/imgs/01_gd_view2.png"
    height="200">
 
- `02_regression_gd.py`: Starting from 1-dimensional noisy linear training data, a linear unit $y = w_0 + w_1 \cdot x$, and its loss function $\mathcal{J}\_{\vec w}(X) = \frac{1}{N} \Sigma_{n=1}^N(y^{n} - t^{n})^{2}$, the script implements gradient descent. The idea is to fit a line to the generated data.

- `03_two_layer_network.py`: Implementation of a 2-layer network (using a logistic activation function) that operates on one input $x$ and produces one output $y$. The implementation comprises functions to compute the loss and the gradient for a given training dataset, as well as a function to perform iterative gradient descent. The goal is to train a network that fits to the training data. The following figure shows a possible result of this fitting process. Considering the left plot, it becomes clear that the network learned a function (i.e., the blue line) that well approximates the training data (i.e., the red 'x'). The right plot exemplifies how the network loss progressed as a function of the number of training epochs.

   <img
    src="/imgs/03_two_layer_network.png"
    height="200">

- `04_multi_output_regression.py`: Using the [Student Performance Data Set](https://archive.ics.uci.edu/ml/datasets/Student+Performance#), a multi-target network is trained to predict three course grades from various data features such as sex, paid classes, romantic relationship, or daily alcohol. Furthermore, concepts such as batch processing and gradient descent (with and without using a momentum term) are implemented and explored. The following figure illustrates the loss of the multi-target network during training as a function of the number of training epochs for different gradient descent algorithms.

   <img
    src="/imgs/04_multi_output_regression.png"
    height="200">

- `05_binary_classification.py`: A binary classifier is trained on the [Banknote Authentication Data Set](https://archive.ics.uci.edu/ml/datasets/banknote+authentication) as well as on the [Spambase Data Set](https://archive.ics.uci.edu/ml/datasets/spambase). The goal is to train two classifiers via stochastic gradient descent. In addition, the classifier trained on the banknote dataset should achieve a training accuracy of 100% (i.e., detecting all forged banknotes). The figure below illustrates the progression of the loss and the accuracy during the training procedure for the banknote dataset.

   <img
    src="/imgs/05_bin_clf_banknote.png"
    height="200">

- `06_categorical_classification.py`: A categorical classifier is trained on the [Iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html) dataset as well as on the [Digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) dataset. The goal is to train two classifiers via stochastic gradient descent. The figures below illustrate the progression of the loss and the accuracy during the training procedure on both datasets.

   <img
    src="/imgs/06_cat_clf_iris.png"
    height="200"
    align="left">
    
   <img
    src="/imgs/06_cat_clf_digits.png"
    height="200">


 



