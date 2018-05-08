# rewritten to a class and extended to work with sentences
# code from https://github.com/fraunhofer-iais/kpca_embeddings :
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


import re
import numpy as np
import multiprocessing
from scipy import exp
from scipy.linalg import eigh
import codecs
import distance
import termcolor
from unidecode import unidecode
from nltk import ngrams
import matplotlib.pyplot as plt
import scipy
from pythonrouge.pythonrouge import Pythonrouge

class KernelPCA:

    def __init__(self, answer_list: list, n_components: int = 5, kernel: str = "rbf", hyperparam: float = 0.7,
                 similarity_function: str = "similarity_rouge1", cores: int = multiprocessing.cpu_count()):
        self.answer_list = answer_list
        self.n_components = n_components
        self.kernel = kernel
        self.hyperparam = hyperparam
        self.similarity_function = similarity_function
        self.cores = cores

    def description(self):
        description_output = "Answer list: %s, \n" \
                   " number of principal components of the embeddings (default 2): %s, \n" \
                   " kernel: 'poly','rbf' (default: 'rbf' ): %s, \n" \
                   " for RBF kernel and the degree for the polynomial kernel (default: 0.7) %s, \n" \
                   " number of processes to be started for computation (default: number of available cores): %s, \n" \
                   " similarity function you'd like to use (default: similarity_rouge1: %s, \n"  % (self.answer_list,
                            self.n_components, self.kernel, self.hyperparam, self.cores,
                            self.similarity_function)
        return description_output

    def similarity_function_tuple(self, tuple):
        word_array1, word_array2 = tuple
        similarity_method_to_call = getattr(self, self.similarity_function)
        return similarity_method_to_call(word_array1, word_array2)

    def calculate_kernel_matrix(self, similarity_matrix):
        if self.kernel == "rbf":
            return exp(-self.hyperparam * (similarity_matrix ** 2))
        else:  # poly
            return (np.ones(len(similarity_matrix)) - similarity_matrix) ** self.hyperparam

    def init_list_of_objects(self, size):
        list_of_objects = list()
        for i in range(0, size):
            list_of_objects.append(list())
        return list_of_objects

    def similarity_rouge1(self, s1, s2):
        rouge = Pythonrouge(summary_file_exist=False,
                            summary=[s1], reference=[[s2]],
                            n_gram=2, ROUGE_SU4=True, ROUGE_L=True,
                            recall_only=True, stemming=False, stopwords=False,
                            word_level=True, length_limit=True, length=50,
                            use_cf=False, cf=95, scoring_formula='average',
                            resampling=True, samples=1000, favor=True, p=0.5)
        ROUGE_score = rouge.calc_score()
        try:
            return list(ROUGE_score.values())[0]
        except:
            return 0

    def similarity_rougeL(self, s1, s2):
        rouge = Pythonrouge(summary_file_exist=False,
                            summary=s1, reference=s2,
                            n_gram=2, ROUGE_SU4=True, ROUGE_L=True,
                            recall_only=True, stemming=False, stopwords=False,
                            word_level=True, length_limit=True, length=50,
                            use_cf=False, cf=95, scoring_formula='average',
                            resampling=True, samples=1000, favor=True, p=0.5)
        ROUGE_score = rouge.calc_score()
        return list(ROUGE_score.values())[2]

    def similarity_blue(self, s1, s2):
        BLEU_score = nltk.translate.bleu_score.sentence_bleu([s1], s2)
        return BLEU_score

    def projection_matrix(self, pool_tuple):
        line = pool_tuple[0]
        vocabulary_tuples = pool_tuple[1]
        alphas = pool_tuple[2]
        lambdas = pool_tuple[3]
        similarity_method_to_call = getattr(self, self.similarity_function)
        pair_sim = np.array([similarity_method_to_call(line, element) for element in vocabulary_tuples])
        return self.calculate_kernel_matrix(pair_sim).dot(alphas/lambdas)

    def alpha_lambda_calculation(self):
        '''
        Preprocessing
        '''
        termcolor.cprint("Generating element pairs\n", "blue")
        answer_list_len = len(self.answer_list)
        print(self.answer_list)
        pairs_array = np.array([(answer1, answer2) for answer1 in self.answer_list for answer2 in self.answer_list])
        '''
        Similarity matrix computation: the similarity of all word pairs from the representative words is computed
        '''
        termcolor.cprint("Computing similarity matrix\n", "blue")
        pool = multiprocessing.Pool(processes=self.cores)
        similarity_matrix = np.array(pool.map(self.similarity_function_tuple, pairs_array)).reshape(answer_list_len, answer_list_len)
        '''
        Kernel Principal Component Analysis
        '''
        termcolor.cprint("Solving eigevector/eigenvalues problem\n", "blue")
        kernel_matrix = self.calculate_kernel_matrix(similarity_matrix)
        # Centering the symmetric NxN kernel matrix.
        n = kernel_matrix.shape[0]
        one_n = np.ones((n,n)) / n
        k_norm = kernel_matrix - one_n.dot(kernel_matrix) - kernel_matrix.dot(one_n) + one_n.dot(kernel_matrix).dot(one_n)
        # Obtaining eigenvalues in descending order with corresponding eigenvectors from the symmetric matrix.
        eigvals, eigvecs = eigh(k_norm)
        alphas = np.column_stack((eigvecs[:,-i] for i in range(1, self.n_components+1)))
        lambdas = [eigvals[-i] for i in range(1, self.n_components+1)]
        return alphas, lambdas

    def training_results(self):
        termcolor.cprint("Projecting known vocabulary to KPCA embeddings\n", "blue")
        alphas, lambdas = self.alpha_lambda_calculation()
        pool = multiprocessing.Pool(processes=self.cores)
        x_train = pool.map(self.projection_matrix, [(line, self.answer_list, alphas, lambdas) for line in self.answer_list])
        x_train = np.asarray(x_train)
        vector_distances = []
        for i in range(len(x_train)-1):
            vector_distance = scipy.spatial.distance.cosine(x_train[i], x_train[len(x_train)-1])
            print(vector_distance)
            vector_distances.append(1 - vector_distance)
        #fig = plt.figure(figsize=(10, 10))
        #ax = fig.add_subplot(111)
        #ax.plot(x_train[:, 0], x_train[:, 1], 'go')
        #for i, label in enumerate(self.answer_list):
        #    plt.text(x_train[:, 0][i], x_train[:, 1][i], label)
        #plt.show()
        #return vector_distances
        return x_train[:-1]
