# the code is based on the following doc2vec tutorial:
#  https://medium.com/@mishra.thedeepak/doc2vec-in-a-simple-way-fa80bfe81104
import gensim
from os import listdir
import scipy
import pickle

class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list

    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield gensim.models.doc2vec.LabeledSentence(doc, [self.labels_list[idx]])

class Doc2vec_vectorization:

    def __init__(self, q_number: str, n_vectors: int = 2):
        self.q_number = q_number
        self.n_vectors = n_vectors

    def model_creation(self):
        data_folder = "../../data/Mohler/processed/forDoc2Vec/" + self.q_number + "/"
        doc2vec_answer_array_filepath_other = "../../data/Mohler/processed/doc2vec_answer_array.pickle"

        docLabels = [f for f in listdir(data_folder)]

        fileObject = open(doc2vec_answer_array_filepath_other, 'rb')
        data_from_pickle = pickle.load(fileObject)
        fileObject.close()
        data = data_from_pickle[self.q_number]

        it = LabeledLineSentence(data, docLabels)

        model = gensim.models.Doc2Vec(vector_size=self.n_vectors, min_count=0, alpha=0.025, min_alpha=0.0205)
        #model = gensim.models.Doc2Vec(vector_size=self.n_vectors, min_count=0, alpha=0.1, min_alpha=0.0205)
        model.build_vocab(it)
        # training of model
        for epoch in range(100):
            print("iteration" + str(epoch + 1))
            model.train(it, total_examples=model.corpus_count, epochs=model.epochs)
            model.alpha -= 0.002
            model.min_alpha = model.alpha
        # saving the created model
        return model

    def training_results(self):
        # loading the model
        d2v_model = self.model_creation()
        # start testing
        # printing the vector of document at index 1 in docLabels
        docvec = d2v_model.docvecs[1]
        correct_vec = d2v_model.docvecs["correct"]
        answer_vecs = []
        distances = []
        for answer_number in range(1, len(d2v_model.docvecs)):
            answer_vec = d2v_model.docvecs[str(answer_number)]
            answer_vecs.append(answer_vec)
            dist = scipy.spatial.distance.cosine(answer_vec, correct_vec)
            distances.append(dist)
        return answer_vecs