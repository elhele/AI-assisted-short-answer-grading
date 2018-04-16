from autocorrect import spell
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import digits, punctuation
from pythonrouge.pythonrouge import Pythonrouge
import codecs
import multiprocessing
from sklearn.ensemble import RandomForestClassifier
from itertools import repeat
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import distance


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


def get_keywords_count(answer, keywords, question_index):
    summ_distance = 0
    for keyword in keywords:
        dist = 0
        for word in answer:
            dist_tmp = distance.sorensen(keyword, word)
            if dist_tmp > dist:
                dist = dist_tmp
        summ_distance += dist
    if len(keywords) == 0:
        print("key" + str(keywords) + str(question_index))
        return 0
    else:
        return summ_distance / len(keywords)


def get_features(question_index, answer_array, correct_answers, keywords):
    print("question number:" + str(question_index) + str(keywords[question_index]))
    reference = [[[' '.join(word for word in correct_answers[question_index])]]]
    features = []
    for i in range(len(answer_array[question_index][1])):
        # BLUE calculation:
        # BLEU_score = nltk.translate.bleu_score.sentence_bleu([correct_answers[question_index]],
        #                                                     answer_array[question_index][0][i])
        # ROUGE calculation:
        # summary = [[' '.join(word for word in answer_array[question_index][0][i])]]
        # rouge = Pythonrouge(summary_file_exist=False,
        #                    summary=summary, reference=reference,
        #                    n_gram=2, ROUGE_SU4=True, ROUGE_L=True,
        #                    recall_only=True, stemming=False, stopwords=False,
        #                    word_level=True, length_limit=True, length=50,
        #                    use_cf=False, cf=95, scoring_formula='average',
        #                    resampling=True, samples=1000, favor=True, p=0.5)
        # ROUGE_score = rouge.calc_score()
        # {'ROUGE-1': 0.45455, 'ROUGE-2': 0.2, 'ROUGE-L': 0.36364, 'ROUGE-SU4': 0.24}
        # features.append(list(ROUGE_score.values())[2])
        keyword_coverage = get_keywords_count(keywords[question_index], answer_array[question_index][0][i],
                                              question_index)
        features.append(keyword_coverage)
    # For one feature classification:
    features = np.asarray(features)
    features = features.reshape(-1, 1)
    return features


def main():
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    question_number = [7, 7, 7, 7, 4, 7, 7, 7, 7, 7, 10, 10]
    data_folder = 'data/Mohler/'
    answer_array_filepath = data_folder + "/processed/answer_array.pickle"
    r1_features_filepath = data_folder + "/processed/r1.pickle"
    r2_features_filepath = data_folder + "/processed/r2.pickle"
    rL_features_filepath = data_folder + "/processed/rL.pickle"
    rS4_features_filepath = data_folder + "/processed/rS4.pickle"
    blue_features_filepath = data_folder + "/processed/blue.pickle"
    rouge_features_filepath = data_folder + "/processed/rouge.pickle"
    r12L_features_filepath = data_folder + "/processed/r12L.pickle"
    keywords_features_filepath = data_folder + "/processed/keywords.pickle"
    ngrams_features_filepath = data_folder + "/processed/ngrams.pickle"
    kskip_features_filepath = data_folder + "/processed/kskip.pickle"

    correct_answers = get_from_file(data_folder + 'raw/answers')
    keywords = get_from_file(data_folder + 'processed/keywords')
    for i in range(len(correct_answers)):
        correct_answers[i] = filter(correct_answers[i], True)
        keywords[i] = filter(keywords[i], True)
    correct_answer_index = 0
    answer_array = []
    for chapter in range(1, len(question_number) + 1):
        for question in range(1, question_number[chapter - 1] + 1):
            answers = get_from_file(data_folder + 'raw/' + str(chapter) + '.' + str(question))
            grades = get_from_file(data_folder + 'scores/' + str(chapter) + '.' + str(question) + '/other')
            answer_array.append([answers, grades])
            correct_answer_index += 1

    # fileObject = open(answer_array_filepath,'wb')
    # for question_index in range(len(correct_answers)):
    #    for i in range(len(answer_array[question_index][1])):
    #        answer_array[question_index][0][i] = filter(answer_array[question_index][0][i], True)
    # pickle.dump(answer_array,fileObject)
    # fileObject.close()
    fileObject = open(answer_array_filepath, 'rb')
    answer_array = pickle.load(fileObject)
    fileObject.close()

    features = pool.starmap(get_features,
                            zip(range(len(correct_answers)), repeat(answer_array), repeat(correct_answers),
                                repeat(keywords)))

    # fileObject = open(r12L_features_filepath,'wb')
    # pickle.dump(features,fileObject)
    # fileObject.close()

    scores = []
    # get an average of 10 calculations to avoid problems with randomness
    for i in range(100):
        for question_index in range(len(correct_answers)):
            clf = RandomForestClassifier(n_estimators=5)
            split_position = 13  # for it gives the best accuracy
            clf.fit(features[question_index][0:split_position], answer_array[question_index][1][0:split_position])
            score = clf.score(features[question_index][split_position:len(answer_array[question_index][1])],
                              answer_array[question_index][1][split_position:len(answer_array[question_index][1])])
            scores.append(score)
            # print(score)

    print("average")
    average_score = sum(scores) / len(scores)
    print(average_score)

    # scores_diff_average = []
    # for column in range(len(scores_diff_split[0])):
    #     scores_diff_average.append(sum(row[column] for row in scores_diff_split)/len(scores_diff_split))

    # print(scores_diff_average)
    # split_positions = list(range(3, 24, 2))
    # accuracies
    # plt.plot(split_positions, scores_diff_average, 'bo')
    # plt.xlabel('Split position')
    # plt.ylabel('Average accuracy')
    # plt.axis([2, 25, 0.3, 0.8])
    # plt.grid(True)
    # plt.show()


if __name__ == "__main__":
    main()