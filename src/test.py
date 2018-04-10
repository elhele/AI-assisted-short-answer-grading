from kpca import KernelPCA

# test code
#print(KernelPCA('../../data/1.1.txt').description())
KernelPCA('../data/test.txt', 2, 'rbf', 0.7, 3, True, 4, 'sorensen_word', 'word').training_results()
#print(alphas)
#print(lambdas)
#print(KernelPCA('../../data/test.txt', 2, 'rbf', 0.7, 2, True, 4, 'sorensen_word', 'word').training_results())
