from kpca import KernelPCA

# test code
##alphas, lambdas = KernelPCA('../../data/1.1.txt').alpha_lambda_calculation()
#print(alphas)
#print(lambdas)
print(KernelPCA('../../data/test.txt', 2, 'rbf', 0.7, 2, True, 4, 'sorensen_word', 'word').training_results())
