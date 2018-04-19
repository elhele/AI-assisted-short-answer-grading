from kpca import KernelPCA
import numpy as np

# test code
#print(KernelPCA('../../data/1.1.txt').description())
# KernelPCA('../data/test.txt', 2, 'rbf', 0.7, 3, True, 4, 'sorensen_word', 'word').training_results()
#print(alphas)
#print(lambdas)
#print(KernelPCA('../../data/test.txt', 2, 'rbf', 0.7, 2, True, 4, 'sorensen_word', 'word').training_results())
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)
matrix = np.array([[1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 2, 1]])
u, s, vh = np.linalg.svd(matrix)
print("U = " + str(u))
print("Sigma = " + str(s))
print("V* = " + str(vh))
s = np.array([[2.46, 0, 0, 0], [0, 1.7, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

print(np.dot(np.dot(u,s),vh))