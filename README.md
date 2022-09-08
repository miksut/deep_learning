# Deep Learning

This repository hosts some implementations of prominent Deep Learning concepts. These implementations come in form of Python scripts and can be found in the folder `src/`. Concretely, the folder contains the following scripts:

---
-  `00_perceptron.py`: Implementation of a simple perceptron (that operates on 2-dimensional data) and a learning rule. Concretely, the learning process begins by randomly initializing the perceptron's weights $\vec w$. Then, a training sample $\vec x$ and its associated class target $t \in$ {-1, 1} are chosen and the class is predicted: $y = \vec w^{T}\vec x$. If wrongly classified (i.e., $y \cdot t < 0$), the weights are updated as $\vec w = \vec w + t \cdot \vec x$.
---

- `01_gradient_descent.py`: Given the loss function $\mathcal{J}\_{\vec{w}} = w_1^2 + w_2^2 + 30 \cdot \sin(w_1) \cdot \sin(w_2)$, this script implements the gradient descent algorithm for an initial weight vector $\vec w$ and a given learning rate $\eta$. The goal is to investigate the effect of several initial $\vec w$ and different $\eta$ on the convergence process. This convergence process is illustrated in the figures below. The red point represents the initial weight vector $\vec w$ on the error surface. The green point stands for the final weight vector that results from the gradient descent.

   <img
    src="/imgs/01_gd_view1.png"
    height="200"
    align="left">
    
    <img
    src="/imgs/01_gd_view2.png"
    height="200">
---

- `02_regression_gd.py`: Starting from 1-dimensional noisy linear training data, a linear unit $y = w_0 + w_1 \cdot x$, and its loss function $\mathcal{J}\_{\vec w}(X) = \frac{1}{N} \Sigma_{n=1}^N(y^{n} - t^{n})^{2}$, the script implements gradient descent. The idea is to fit a line to the generated data.
---

- `03_two_layer_network.py`: Implementation of a 2-layer network (using a logistic activation function) that operates on one input $x$ and produces one output $y$. The implementation comprises functions to compute the loss and the gradient for a given training dataset, as well as a function to perform iterative gradient descent. The goal is to train a network that fits to the training data. The following figure shows a possible result of this fitting process. Considering the left plot, it becomes clear that the network learned a function (i.e., the blue line) that well approximates the training data (i.e., the red 'x'). The right plot exemplifies how the network loss progressed as a function of the number of training epochs.

   <img
    src="/imgs/03_two_layer_network.png"
    height="200">
---

- `04_multi_output_regression.py`: Using the [Student Performance Data Set](https://archive.ics.uci.edu/ml/datasets/Student+Performance#), a multi-target network is trained to predict three course grades from various data features such as sex, paid classes, romantic relationship, or daily alcohol. Furthermore, concepts such as batch processing and gradient descent (with and without using a momentum term) are implemented and explored. The following figure illustrates the loss of the multi-target network during training as a function of the number of training epochs for different gradient descent algorithms.

   <img
    src="/imgs/04_multi_output_regression.png"
    height="200">
---

- `05_binary_classification.py`: A binary classifier is trained on the [Banknote Authentication Data Set](https://archive.ics.uci.edu/ml/datasets/banknote+authentication) as well as on the [Spambase Data Set](https://archive.ics.uci.edu/ml/datasets/spambase). The goal is to train two classifiers via stochastic gradient descent. In addition, the classifier trained on the banknote dataset should achieve a training accuracy of 100% (i.e., detecting all forged banknotes). The figure below illustrates the progression of the loss and the accuracy during the training procedure for the banknote dataset.

   <img
    src="/imgs/05_bin_clf_banknote.png"
    height="200">
---

- `06_categorical_classification.py`: A categorical classifier is trained on the [Iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html) dataset as well as on the [Digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) dataset. The goal is to train two classifiers via stochastic gradient descent. The figures below illustrate the progression of the loss and the accuracy during the training procedure on both datasets.

   <img
    src="/imgs/06_cat_clf_iris.png"
    height="200"
    align="left">
    
   <img
    src="/imgs/06_cat_clf_digits.png"
    height="200">
---

- `07_PyTorch_catClf.py`: A two-layer neural network is trained and tested on the [MNIST](https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html) dataset, using the [PyTorch](https://pytorch.org/) framework. The network is trained using categorical cross-entropy.
---

- `08_CN_MNIST.py`: A convolutional neural network (CNN) is trained and tested on the MNIST dataset, using the PyTorch framework. The CNN comprises convolutional layers, pooling layers, as well as a fully-connected layer and is trained using categorical cross-entropy.
---

- `09_face_recognition.py`: Using a CNN that has been pretrained on ImageNet as a deep feature extractor in order to perform face recognition. The [Yale Face Database](http://vision.ucsd.edu/content/yale-face-database) builds the data basis for the face recognition task. This dataset contains grayscale images of 15 people (i.e., the subjects) that have been photographed in 11 situations (i.e., eleven variations: normal, happy, sad, sleepy, surprised, wink, with glasses, without glasses, illumination from left, right, and the center). The extracted features of the "normal" images are stored in a gallery and build the reference points to which all the other images are compared to. The goal is to compare all remaining variations of all subjects to the gallery features and assign them to the correct subject. The table below illustrates this process. For example, the column "glasses" illustrates that images from five subjects (out of 15) wearing glasses are correctly classified (i.e., assigned to the correct gallery subject). 

   normal  | happy | sad | sleepy | surprised | wink | glasses | noglasses | leftlight | rightlight | centerlight
   :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: 
   15/15 | 15/15 | 14/15 | 15/15 | 14/15 | 15/15 | 5/15 | 13/15 | 13/15 | 9/15 | 10/15
---

- `10_open_set.py`: This script deals with the concept of open set recognition. As introduced by [Scheirer et al.](https://ieeexplore.ieee.org/document/6365193), a classifier that performs open set classification additionally encounters test samples from classes that have not been present during training (i.e., the unknown classes). Therefore, the tasks of an open set classifier are to correctly classify samples from known classes (i.e., classes that the classifier has been trained on) and to reject samples from unknown classes. Based on a loss function and an evaluation metric proposed by [Dhamija et al.](https://dl.acm.org/doi/10.5555/3327546.3327590), this script trains and evaluates a CNN on an open set partition of the MNIST dataset. 
---

- `11_GAN.py`: This script explores some of the capabilities of StarGAN, a generative adversarial network introduced by [Choi et al.](https://openaccess.thecvf.com/content_cvpr_2018/html/Choi_StarGAN_Unified_Generative_CVPR_2018_paper.html) in 2018. Concretely, the generator is used to manipulate some facial attributes of (own) image portraits.

   In order to run the script, some preparations are necessary. Specifically, the [code](https://github.com/yunjey/stargan) as provided by the authors as well as a pretrained generator must be downloaded. For more information on the capabilities and use of StarGAN, refer to the author's GitHub repository. The following instructions assume that a console is opened in the **root** directory of this project.

   ```
   $ cd src
   $ git clone https://github.com/yunjey/stargan      # download the repository
   
   $ cd stargan
   $ bash ./download.sh pretrained-celeba-128x128     # download pretrained model, alternatively inspect download.sh
   ```
   
   Now, open the script `11_GAN.py` and configure the variables specified in the section "README". Finally, run the script to generate image portraits with manipulated facial attributes "Black_Hair", "Blond_Hair", "Brown_Hair", "Male", and "Young".
---

- `12_RNN.py`: Training of a simple recurrent neural network (RNN) on [Shakespeare's sonnets](https://en.wikipedia.org/wiki/Shakespeare%27s_sonnets). The dataset is available [here](https://github.com/brunoklein99/deep-learning-notes/blob/master/shakespeare.txt) (no manual download necessary, script handles download automatically). After the training procedure, given a seeding text (e.g., "moth"), the network can then be used to generate text. The following commands illustrate how to train the network and subsequently use it for text generation. 

   ```
   $ python .\src\12_RNN.py train      # train the RNN (assumption: console opened in root folder of this project)
   
   $ python .\src\12_RNN.py best moth  # use seeding text "moth" and append 80 additional characters based on most probable successor (i.e., argument "best")
   ```
   Below, you can find an exemplary output of an RNN that has been trained for 50 epochs:
   
   ```   
   $ python .\src\12_RNN.py best deep
   deep -> "deep on sightless eyer drow, nor drow n weattes belter than time wastes life, thy br"
   ```
---

- `13_Adversarial_Training.py`: Considering the ideas of [Goodfellow et al. (2015)](https://arxiv.org/abs/1412.6572), this script implements a procedure that integrates adversarial samples into the training of a CNN. The adversarial samples are generated using the fast gradient sign (FGS, Goodfellow et al., 2015) as well as the fast gradient value (FGV, [Rozsa et al., 2016](https://ieeexplore.ieee.org/document/7789548)) methods. Using the MNIST dataset, the goal is to train a CNN that is more robust against adversarial samples. 

   The script can be configured to perform vanilla training (i.e., train the CNN without adversarial samples) as well as adversarial training (i.e., include aversarial samples in the training procedure). Depending on the preferences, the global variables (e.g., batch size, epochs, learning rate) can be customized within the script. The training and evaluation of the network can be performed via CLI.
   
   ```
   $ python .\src\13_Adversarial_Training.py --train True      # train CNN without adversarial samples
   $ python .\src\13_Adversarial_Training.py --train_adversarial True      # adversarial training
   
   $ python .\src\13_Adversarial_Training.py --evaluate True      # evaluate vanilla-trained CNN on test set
   $ python .\src\13_Adversarial_Training.py --evaluate_adversarial True      # evaluate adversarially-trained CNN on test set
   ```
   
   An exemplary performance comparison between the two training procedures is shown below. The first block refers to a CNN that has not seen adversarial samples during training and is evaluated on a test set containing adversarial samples. The second block is linked to a CNN that has undergone adversarial training and is evaluated on a test set containing adversarial samples.
   
   ```
   Loaded model: D:\Projects\deep_learning\results\adversarial_training\cnn_mnist.model
   Accuracy on test set using original samples: 98.33 %
   Accuracy on test set using adversarial samples (FGS): 0.16999999999999998 %
   
   Loaded model: D:\Projects\deep_learning\results\adversarial_training\cnn_mnist_adv.model
   Accuracy on test set using original samples: 98.63 %
   Accuracy on test set using adversarial samples (FGS): 98.96000000000001 %
   ```
---

- `14_RBF_network.py`: 
 



