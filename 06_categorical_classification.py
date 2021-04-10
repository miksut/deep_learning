import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.datasets import load_iris, load_digits


# dedicated class
# ---------------------------------------------------------
class CategoricalClassifierNet:
	def __init__(self):
		pass



# preparing datasets
# ---------------------------------------------------------
data_iris = load_iris()
data_digits = load_digits()

X_iris = np.hstack((np.ones([data_iris.data.shape[0], 1]), data_iris.data))
t_iris = data_iris.target

X_digits = np.hstack((np.ones([data_digits.data.shape[0], 1]), data_digits.data))
t_digits = data_digits.target

print(X_digits.shape)
print(t_digits.shape)
