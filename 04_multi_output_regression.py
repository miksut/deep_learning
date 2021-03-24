import numpy as np
import pandas as pd

# global variables
rng = np.random.default_rng()



# loading locally stored dataset (available at https://archive.ics.uci.edu/ml/datasets/Student+Performance#)
df = pd.read_csv('student/student-mat.csv', sep=';')
# drop categorical values (according to task description)
df = df.drop(df.iloc[:, 8:12].columns, axis=1)
# change dtypes of columns according to task description
dict_float = {key: "float64" for key in df.select_dtypes(include=['int64']).columns}
df = df.astype(dict_float)

def batch(X, T, B):
	# ensure identical number of datapoints and associated targets
	assert X.shape[0] == T.shape[0]
	indices = np.arange(X.shape[0])
	rng.shuffle(indices)

	for start_idx in range(0, X.shape[0], B):
		end_idx = min(start_idx + B, X.shape[0])

		idx_set = indices[start_idx:end_idx]

		# strategy: simply ignore batch with batachsize < B
		if end_idx == start_idx + B:
			yield X[idx_set], T[idx_set]



# testing
X = rng.standard_normal((150, 5))
T = rng.standard_normal((150, 1)) * 10

print(df.dtypes)


