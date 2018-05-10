from autocorrect import spell
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import ngrams
from nltk.util import skipgrams
from string import digits, punctuation
from pythonrouge.pythonrouge import Pythonrouge
import codecs
import multiprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from itertools import repeat
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import distance
from kpca_vectorization import KernelPCA
from doc2vec_vectorization import Doc2vec_vectorization


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


def get_from_file(path):
    with codecs.open(path, "r") as fIn:
        strings = [strings.rstrip() for strings in fIn.readlines()]
        return strings


def ngram_cooccurrence(example, answer, ng):
    distance_over_ngrams = 0
    for n in range(1, ng + 1):
        answer_ngramed = list(ngrams(answer, n))
        example_ngramed = list(ngrams(example, n))
        summ_distance = 0
        for ngram_of_example in example_ngramed:
            dist = 0
            for ngram_of_answer in answer_ngramed:
                dist_tmp = distance.levenshtein(ngram_of_example, ngram_of_answer)
                if dist_tmp > dist:
                    dist = dist_tmp
            summ_distance += dist
        try:
            distance_over_ngrams += summ_distance / len(example_ngramed)
        except:
            distance_over_ngrams += 0
    return distance_over_ngrams


def kskip_ngram_cuoccurrence(answer, example, ng, kskip, question_index):
    distance_over_ngrams = 0
    # nltk skipgrams function doesn't include unigrams, so they are calculated separately:
    summ_distance = 0
    for word_ex in example:
        dist = 0
        for word_an in answer:
            dist_tmp = distance.levenshtein(word_ex, word_an)
            if dist_tmp > dist:
                dist = dist_tmp
        summ_distance += dist
    try:
        distance_over_ngrams += summ_distance / len(example)
    except:
        print("zero")
        print(question_index)
        distance_over_ngrams += 0
    for n in range(2, ng + 1):
        answer_ngramed = list(skipgrams(answer, n, kskip))
        example_ngramed = list(skipgrams(example, n, kskip))
        summ_distance = 0
        for ngram_of_example in example_ngramed:
            dist = 0
            for ngram_of_answer in answer_ngramed:
                dist_tmp = distance.levenshtein(ngram_of_example, ngram_of_answer)
                if dist_tmp > dist:
                    dist = dist_tmp
            summ_distance += dist
        try:
            distance_over_ngrams += summ_distance / len(example_ngramed)
        except:
            distance_over_ngrams += 0
    return distance_over_ngrams


def get_keywords_count(answer, keywords):
    summ_distance = 0
    for keyword in keywords:
        dist = 0
        for word in answer:
            dist_tmp = distance.levenshtein(keyword, word)
            if dist_tmp > dist:
                dist = dist_tmp
        summ_distance += dist
    try:
        return summ_distance / len(keywords)
    except:
        return 0


def get_features(question_index, answer_array, correct_answers, keywords, labels):
    print("question number:" + str(question_index))
    # keyword = keywords[question_index]
    reference = [[[' '.join(word for word in correct_answers[question_index])]]] #ROUGE
    features = []
    #summaries = [] # ROUGE KPCA
    #for i in range(len(answer_array[question_index][1])):
        # BLUE calculation:
        #BLEU_score = nltk.translate.bleu_score.sentence_bleu([correct_answers[question_index]],
        #                                                     answer_array[question_index][0][i])
        # ROUGE calculation:
        #summary = [[' '.join(word for word in answer_array[question_index][0][i])]]
        #summaries.append(summary[0]) # ROUGE KPCA
        #rouge = Pythonrouge(summary_file_exist=False,
        #                    summary=summary, reference=reference,
        #                    n_gram=2, ROUGE_SU4=True, ROUGE_L=True,
        #                    recall_only=True, stemming=False, stopwords=False,
        #                    word_level=True, length_limit=True, length=50,
        #                    use_cf=False, cf=95, scoring_formula='average',
        #                    resampling=True, samples=1000, favor=True, p=0.5)
        #ROUGE_score = rouge.calc_score()
        # {'ROUGE-1': 0.45455, 'ROUGE-2': 0.2, 'ROUGE-L': 0.36364, 'ROUGE-SU4': 0.24}
        #ngram_cooc = ngram_cooccurrence(correct_answers[question_index],
        #                                answer_array[question_index][0][i], 3)
        #features.append(list(ROUGE_score.values())[2])
        #features.append([list(ROUGE_score.values())[0], list(ROUGE_score.values())[2], BLEU_score])
        #answer_array[question_index][0] = answer_array[question_index][0].append(correct_answers[question_index])
        #summaries.append(reference[0][0]) # ROUGE KPCA
    to_vectorize = answer_array[question_index][0] # BLUE KPCA
    to_vectorize.append(correct_answers[question_index]) # BLUE KPCA
    #features = KernelPCA(summaries, 3, 'poly', 3, 'similarity_rouge1').training_results() # KPCA
    features = KernelPCA(to_vectorize, 25, 'poly', 3, 'similarity_blue').training_results()
    #features = Doc2vec_vectorization(labels[question_index], 2).training_results()
    # For one feature classification:
    features = np.asarray(features)
    features = features.reshape(-1, 1)
    return features


def main():
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    thread = multiprocessing.pool.ThreadPool(processes=multiprocessing.cpu_count())
    question_number = [7, 7, 7, 7, 4, 7, 7, 7, 7, 7, 10, 10]
    data_folder = '../../data/Mohler/' # for laptop
    #data_folder = 'data/Mohler/'  # for comp
    answer_array_filepath_other = data_folder + "/processed/answer_array.pickle"
    answer_array_filepath_me = data_folder + "/processed/answer_array_me.pickle"
    answer_array_filepath_ave = data_folder + "/processed/answer_array_ave.pickle"
    r1_features_filepath_other = data_folder + "/processed/r1.pickle"
    r2_features_filepath_other = data_folder + "/processed/r2.pickle"
    rL_features_filepath_other = data_folder + "/processed/rL.pickle"
    rL_features_filepath_ave = data_folder + "/processed/rL_ave.pickle"
    rL_features_filepath_me = data_folder + "/processed/rL_me.pickle"
    rS4_features_filepath_other = data_folder + "/processed/rS4.pickle"
    blue_features_filepath_other = data_folder + "/processed/blue.pickle"
    rouge_features_filepath_other = data_folder + "/processed/rouge.pickle"
    r12L_features_filepath_other = data_folder + "/processed/r12L.pickle"
    keywords_features_filepath_other = data_folder + "/processed/keywords.pickle"
    ngrams_features_filepath_other = data_folder + "/processed/ngrams.pickle"
    kskip_features_filepath_other = data_folder + "/processed/kskip.pickle"
    combined_features_filepath_other = data_folder + "/processed/combined.pickle"
    r1kpca_features_filepath_other = data_folder + "/processed/r1kpca.pickle"
    rLkpca_features_filepath_other = data_folder + "/processed/rLkpca.pickle"
    bluekpca_features_filepath_other = data_folder + "/processed/bluekpca.pickle"
    allkpca_features_filepath_other = data_folder + "/processed/allkpca.pickle"
    doc2vec_features_filepath_other = data_folder + "/processed/doc2vec.pickle"



    correct_answers = get_from_file(data_folder + 'raw/answers')
    keywords = get_from_file(data_folder + 'processed/keywords')
    for i in range(len(correct_answers)):
        correct_answers[i] = filter(correct_answers[i], True)
        keywords[i] = filter(keywords[i], True)
    correct_answer_index = 0
    answer_array = []
    labels = []
    for chapter in range(1, len(question_number) + 1):
        for question in range(1, question_number[chapter - 1] + 1):
            answers = get_from_file(data_folder + 'raw/' + str(chapter) + '.' + str(question))
            grades = get_from_file(data_folder + 'scores/' + str(chapter) + '.' + str(question) + '/other')
            labels.append(str(chapter) + '.' + str(question))
            answer_array.append([answers, grades])
            correct_answer_index += 1

    # fileObject = open(answer_array_filepath_me,'wb')
    # for question_index in range(len(correct_answers)):
    #    for i in range(len(answer_array[question_index][1])):
    #        answer_array[question_index][0][i] = filter(answer_array[question_index][0][i], True)
    # pickle.dump(answer_array,fileObject)
    # fileObject.close()
    fileObject = open(answer_array_filepath_other, 'rb')
    answer_array = pickle.load(fileObject)
    fileObject.close()

    #features = thread.starmap(get_features, zip(range(len(correct_answers)), repeat(answer_array),
    #                                          repeat(correct_answers), repeat(keywords), repeat(labels)))

    fileObject = open(r1_features_filepath_other, 'rb')
    features_r1 = pickle.load(fileObject)
    fileObject.close()
    # fileObject = open(r2_features_filepath_other, 'rb')
    # features_r2 = pickle.load(fileObject)
    # fileObject.close()
    fileObject = open(rL_features_filepath_other, 'rb')
    features_rL = pickle.load(fileObject)
    fileObject.close()
    # fileObject = open(rS4_features_filepath_other, 'rb')
    # features_s4 = pickle.load(fileObject)
    # fileObject.close()
    fileObject = open(blue_features_filepath_other, 'rb')
    features_blue = pickle.load(fileObject)
    fileObject.close()
    # fileObject = open(keywords_features_filepath_other, 'rb')
    # features_keywords = pickle.load(fileObject)
    # fileObject.close()
    # fileObject = open(ngrams_features_filepath_other, 'rb')
    # features_ngrams = pickle.load(fileObject)
    # fileObject.close()
    # fileObject = open(kskip_features_filepath_other, 'rb')
    # features_kskip = pickle.load(fileObject)
    # fileObject.close()
    fileObject = open(combined_features_filepath_other, 'rb')
    features_combined = pickle.load(fileObject)
    fileObject.close()
    fileObject = open(r1kpca_features_filepath_other, 'rb')
    features_r1_kpca = pickle.load(fileObject)
    fileObject.close()
    fileObject = open(rLkpca_features_filepath_other, 'rb')
    features_rL_kpca = pickle.load(fileObject)
    fileObject.close()
    fileObject = open(bluekpca_features_filepath_other, 'rb')
    features_blue_kpca = pickle.load(fileObject)
    fileObject.close()
    fileObject = open(allkpca_features_filepath_other, 'rb')
    features_all_kpca_blue = pickle.load(fileObject)
    fileObject.close()
    fileObject = open(doc2vec_features_filepath_other, 'rb')
    features_doc2vec = pickle.load(fileObject)
    fileObject.close()

    features_kpca_all = []
    for i in range(len(features_r1_kpca)):
        r1 = np.asarray(features_r1_kpca[i]).flatten()
        rL = np.asarray(features_rL_kpca[i]).flatten()
        feat = []
        for j in range(len(r1)):
            feat.append([r1[j], rL[j]])
        features_kpca_all.append(feat)


    features_all = [features_r1, features_rL, features_blue, features_combined, features_r1_kpca, features_rL_kpca,     features_blue_kpca, features_kpca_all, features_all_kpca_blue, features_doc2vec]
    colors = ["r", "orange", "b", "black", "pink", "magenta", "lightskyblue", "firebrick", "mediumpurple", "darkblue"]
    names = ["ROUGE-1", "ROUGE-L", "BLEU", "combined", "ROUGE-1 KPCA", "ROUGE-L KPCA", "BLEU KPCA", "combined ROUGE KPCA", "combined KPCA", "doc2vec"]

    #fileObject = open(allkpca_features_filepath_other, 'wb')
    #pickle.dump(features_kpca_all, fileObject)
    #fileObject.close()

    average_feature_scores = []
    split_positions = list(range(3, 24, 2))
    feature_number = -1
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
                for question_index in range(len(correct_answers)):
                    print(names[feature_number])
                    # clf = RandomForestClassifier(n_estimators=5)
                    clf = SVC()
                    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    #                    hidden_layer_sizes=(6, 4), random_state=1)
                    features_to_train = features[question_index][0:split_position]
                    print(features_to_train)
                    grades_to_classify = answer_array[question_index][1][0:split_position]
                    # for SVMs only since they need at least several classes
                    features_to_train = np.row_stack((features_to_train, [np.ones(len(features_to_train[0]))*596]))
                    grades_to_classify.append(596)
                    clf.fit(features_to_train, grades_to_classify)
                    score = clf.score(features[question_index][split_position:len(answer_array[question_index][1])],
                                      answer_array[question_index][1][
                                      split_position:len(answer_array[question_index][1])])
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
        # plt.boxplot(scores_for_position, positions=split_positions, sym='', widths=0.6)
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

if __name__ == "__main__":
    main()