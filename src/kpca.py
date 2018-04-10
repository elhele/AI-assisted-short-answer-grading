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

class KernelPCA:


    def __init__(self, vocab_path: str, n_components: int = 2, kernel: str = "rbf", hyperparam: float = 0.7,
                 max_ngrams = 3, plot: bool = True, cores: int = multiprocessing.cpu_count(),
                 similarity_function: str = "similarity_sentence_ngram", word_or_sentence: str = "sentence"):
        self.vocab_path = vocab_path
        self.n_components = n_components
        self.kernel = kernel
        self.hyperparam = hyperparam
        self.max_ngrams = max_ngrams
        self.plot = plot
        self.cores = cores
        self.similarity_function = similarity_function
        self.word_or_sentence = word_or_sentence

    def description(self):
        description_output = "Vocabulary path: %s, \n" \
                   " number of principal components of the embeddings (default 2): %s, \n" \
                   " kernel: 'poly','rbf' (default: 'rbf' ): %s, \n" \
                   " for RBF kernel and the degree for the polynomial kernel (default: 0.7) %s, \n" \
                   " maximum length of the n-grams (default: 3): %s, \n" \
                   " should the result be plotted (default: True): %s, \n" \
                   " number of processes to be started for computation (default: number of available cores): %s, \n" \
                   " similarity function you'd like to use (default: similarity_sentence_ngram: %s, \n" \
                   " shold words or sentences be compared (default: sentence): %s. \n" % (self.vocab_path,
                            self.n_components, self.kernel, self.hyperparam, self.max_ngrams, self.plot, self.cores,
                            self.similarity_function, self.word_or_sentence)
        return description_output

    def similarity_function_tuple(self, tuple):
        word_array1, word_array2 = tuple
        similarity_method_to_call = getattr(self, self.similarity_function)
        return similarity_method_to_call(word_array1, word_array2)

    # TODO: add lemmization
    def normalization(self, w):
        return unidecode(re.sub(r'[^\w]', ' ', w).lower())

    def ngrams_word(self, s, n):
        string = " " + s + " "
        return list(set([string[i:i + n] for i in range(len(string) - n + 1)]))

    # This function takes two words
    def sorensen_word(self, ng1, ng2):
        #ng1 = [ngrams(a, i) for i in range(1, min(len(a), len(b)))]
        #ng2 = [ngrams(b, i) for i in range(1, min(len(a), len(b)))]
        N = min(len(ng1), len(ng2))
        return 1 - np.sum(distance.sorensen(ng1[i], ng2[i]) for i in range(N)) / N

    def similarity_word_ngram(s1, s2):
        ngrams1 = [ngrams_word(s1, j) for j in range(2, min(len(s1) + 1, MAX_NGRAM))]
        ngrams2 = [ngrams_word(s2, j) for j in range(2, min(len(s2) + 1, MAX_NGRAM))]
        N = min(len(ngrams1), len(ngrams2)) + 1
        ngramMatches = np.array([len(np.intersect1d(ngrams1[i], ngrams2[i])) for i in range(N - 1)])
        ngramLens1 = np.array([len(ngrams1[i]) for i in range(N - 1)])
        ngramLens2 = np.array([len(ngrams2[i]) for i in range(N - 1)])

        ngramMatchesLen = len(ngramMatches)
        if ngramMatchesLen > 0:
            return sum(2 * ngramMatches / (ngramLens1 + ngramLens2)) / ngramMatchesLen
        else:
            return 0

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

    def similarity_sentence_ngram(self, s1, s2):
        ng1 = self.init_list_of_objects(min(len(s1.split()) + 1, self.max_ngrams) - 2)
        ng2 = self.init_list_of_objects(min(len(s2.split()) + 1, self.max_ngrams) - 2)
        for j in range(2, min(len(s1.split()) + 1, self.max_ngrams)):
            for ngram in ngrams(s1.split(), j):
                ng1[j - 2].append(ngram)
        for j in range(2, min(len(s2.split()) + 1, self.max_ngrams)):
            for ngram in ngrams(s2.split(), j):
                ng2[j - 2].append(ngram)
        sum = 0
        for j in range(min(min(len(s1.split()) + 1, len(s2.split()) + 1), self.max_ngrams) - 2):
            sum += np.sum(distance.sorensen(ng1[j][i], ng2[j][i])
                           for i in range(min(len(ng1[j]), len(ng2[j])))) / min(len(ng1[j]), len(ng2[j]))
        sum = sum / min(min(len(s1.split()) + 1, len(s2.split()) + 1), self.max_ngrams)

        return 1 - sum

    def similarity_sentence_bag_of_ngrams(self, s1, s2):
        ng1 = self.init_list_of_objects(min(len(s1.split()) + 1, self.max_ngrams) - 2)
        ng2 = self.init_list_of_objects(min(len(s2.split()) + 1, self.max_ngrams) - 2)
        for j in range(2, min(len(s1.split()) + 1, self.max_ngrams)):
            for ngram in ngrams(s1.split(), j):
                ng1[j - 2].append(ngram)
        for j in range(2, min(len(s2.split()) + 1, self.max_ngrams)):
            for ngram in ngrams(s2.split(), j):
                ng2[j - 2].append(ngram)
        ng1_set = set(ng1)
        ng2_set = set(ng2)
        dist = distance.jaccard(ng1_set, ng2_set)
        print(dist)
        return 1 - dist

    def projection_matrix(self, pool_tuple):
        line = pool_tuple[0]
        vocabulary_tuples = pool_tuple[1]
        alphas = pool_tuple[2]
        lambdas = pool_tuple[3]
        similarity_method_to_call = getattr(self, self.similarity_function)
        pair_sim = np.array([similarity_method_to_call(line, element) for element in vocabulary_tuples])
        return self.calculate_kernel_matrix(pair_sim).dot(alphas/lambdas)

    def get_vocabulary(self):
        with codecs.open(self.vocab_path, "r") as fIn:
            if self.word_or_sentence == "word":
                return [  self.normalization(w[:-1]) for w in fIn if len(w[:-1].split()) == 1]
            else:
                vocab = [ self.normalization(w.rstrip()) for w in fIn.readlines()]
                return vocab[:-1]


    def alpha_lambda_calculation(self):
        '''
        Preprocessing
        '''
        vocabulary = self.get_vocabulary()
        termcolor.cprint("Generating element pairs\n", "blue")
        vocabulary_len = len(vocabulary)
        pairs_array = np.array([(word1, word2) for word1 in vocabulary for word2 in vocabulary])
        '''
        Similarity matrix computation: the similarity of all word pairs from the representative words is computed
        '''
        termcolor.cprint("Computing similarity matrix\n", "blue")
        pool = multiprocessing.Pool(processes=self.cores)
        similarity_matrix = np.array(pool.map(self.similarity_function_tuple, pairs_array)).reshape(vocabulary_len, vocabulary_len)
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
        vocabulary = self.get_vocabulary()
        termcolor.cprint("Projecting known vocabulary to KPCA embeddings\n", "blue")
        alphas, lambdas = self.alpha_lambda_calculation()
        pool = multiprocessing.Pool(processes=self.cores)
        x_train = pool.map(self.projection_matrix, [(line, vocabulary, alphas, lambdas) for line in vocabulary])
        x_train = np.asarray(x_train)
        if (self.plot):
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)
            ax.plot(x_train[:, 0], x_train[:, 1], 'go')
            for i, label in enumerate(vocabulary):
                plt.text(x_train[:, 0][i], x_train[:, 1][i], label)
            plt.show()