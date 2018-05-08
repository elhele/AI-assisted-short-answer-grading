# the code is based on the following doc2vec tutorial:
#  https://medium.com/@mishra.thedeepak/doc2vec-in-a-simple-way-fa80bfe81104


import re
import numpy as np
import multiprocessing
import codecs
import pickle

class Doc2vec_vectorization:

    def __init__(self, answer_code: str, n_vectors: int = 2):
        self.answer_code = answer_code
        self.n_vectors = n_vectors

    fileObject = open("/data/Mohler/processed/forDoc2Vec/", 'rb')
    data = pickle.load(fileObject)
    fileObject.close()

    it = LabeledLineSentence(data, docLabels)

    model = gensim.models.Doc2Vec(vector_size=2, min_count=0, alpha=0.025, min_alpha=0.0205)
    model.build_vocab(it)
    # training of model
    for epoch in range(100):
        print("iteration" + str(epoch + 1))
        model.train(it, total_examples=model.corpus_count, epochs=model.epochs)
        model.alpha -= 0.002
        model.min_alpha = model.alpha
    # saving the created model
    model.save("doc2vec.model")
    print("model saved")