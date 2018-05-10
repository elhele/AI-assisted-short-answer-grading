import gensim
import nltk
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from os import listdir
from nltk.stem import WordNetLemmatizer
from string import digits, punctuation
from autocorrect import spell
import scipy
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np
import codecs
import pickle

def get_from_file(path):
    with codecs.open(path, "r") as fIn:
        strings = [strings.rstrip() for strings in fIn.readlines()]
        return strings

class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list

    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield gensim.models.doc2vec.LabeledSentence(doc, [self.labels_list[idx]])


def filter(sentence, spelling):
    # TODO: add such dictionary {1: "one", 2: "two"...} 0 = "null" or "zero"
    wordnet_lemmatizer = WordNetLemmatizer()
    sentence = nltk.word_tokenize(sentence.lower())
    all_stops = set(stopwords.words('english')) | set(punctuation) | set(digits)
    if spelling:
        filtered_words = [wordnet_lemmatizer.lemmatize(spell(word)) for word in sentence if word not in all_stops]
    else:
        filtered_words = [wordnet_lemmatizer.lemmatize(word) for word in sentence if word not in all_stops]
    filtered_words = filtered_words[:][1:]
    return filtered_words

q_number = "1.1"
data_folder = "../data/Mohler/processed/forDoc2Vec/" + q_number + "/"
doc2vec_answer_array_filepath_other = "../data/Mohler/processed/doc2vec_answer_array.pickle"

docLabels = []
docLabels = [f for f in listdir(data_folder)]

fileObject = open(doc2vec_answer_array_filepath_other, 'rb')
data_from_pickle = pickle.load(fileObject)
fileObject.close()
data = data_from_pickle[q_number]

it = LabeledLineSentence(data, docLabels)

model = gensim.models.Doc2Vec(vector_size=20, min_count=0, alpha=0.025, min_alpha=0.0205)
model.build_vocab(it)
#training of model
for epoch in range(100):
    print ("iteration" +str(epoch+1))
    model.train(it, total_examples=model.corpus_count, epochs=model.epochs)
    model.alpha -= 0.002
    model.min_alpha = model.alpha
#saving the created model
model.save("doc2vec.model")
print("model saved")

#loading the model
d2v_model = gensim.models.doc2vec.Doc2Vec.load("doc2vec.model")
#start testing
#printing the vector of document at index 1 in docLabels
docvec = d2v_model.docvecs[1]
correct_vec = d2v_model.docvecs["correct"]
answer_vecs = []
x = []
y = []
distances = []
for answer_number in range(1, len(d2v_model.docvecs)):
    answer_vec = d2v_model.docvecs[str(answer_number)]
    answer_vecs.append(answer_vec)
    dist = scipy.spatial.distance.cosine(answer_vec, correct_vec)
    distances.append(dist)
    x.append(answer_vec[0])
    y.append(answer_vec[1])
x.append(correct_vec[0])
y.append(correct_vec[1])

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
ax.plot(x, y, 'go')
for i, label in enumerate(data):
    plt.text(x[i], y[i], label)
plt.show()

distances = np.asarray(distances)
distances = distances.reshape(-1, 1)

features_all = [answer_vecs]
grades_to_classify = get_from_file("../data/Mohler/scores/" + q_number + "/other")
average_feature_scores = []
split_positions = list(range(3, 24, 2))
feature_number = -1
names = ["doc2vec"]
colors = ["cyan"]
for features in features_all:
    feature_number += 1
    average_scores = []
    scores_for_position = []
    # get an average of 10 calculations to avoid problems with randomness
    for split_position in split_positions:
        scores = []
        scores_for_question = []
        for i in range(1):
            score_for_question = []
            #for question_index in range(len(correct_answers)):
            for question_index in range(10):
                # clf = RandomForestClassifier(n_estimators=5)
                clf = SVC()
                # clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                #                    hidden_layer_sizes=(6, 4), random_state=1)

                # for SVMs only since they need at least several classes
                clf.fit(features[0:split_position], grades_to_classify[0:split_position])
                score = clf.score(features[split_position:len(features)], grades_to_classify[split_position:len(features)])
                scores.append(score)
                score_for_question.append(score)
                # print(score)
            print(score_for_question)
            average_score_for_question = sum(score_for_question) / len(score_for_question)
            print(average_score_for_question)
            scores_for_question.append(average_score_for_question)
        scores_for_position.append(scores_for_question)
        average_score = sum(scores) / len(scores)
        average_scores.append(average_score)
    average_feature_scores.append(average_scores)
plt.ylabel('HCC')
plt.xlabel('Split position')
for feature in range(len(features_all)):
    print(names[feature])
    print(average_feature_scores[feature])
    plt.plot(split_positions, average_feature_scores[feature], colors[feature], label=names[feature])
    plt.scatter(split_positions, average_feature_scores[feature], c=colors[feature])
    plt.legend(loc='lower center')
plt.show()