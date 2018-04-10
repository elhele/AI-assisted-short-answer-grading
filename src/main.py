from autocorrect import spell
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import digits, punctuation
from pythonrouge.pythonrouge import Pythonrouge
import codecs
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

def filter(sentence, spelling):
    wordnet_lemmatizer = WordNetLemmatizer()
    sentence = nltk.word_tokenize(sentence.lower())
    all_stops = set(stopwords.words('english')) | set(punctuation) | set(digits)
    if spelling:
        filtered_words = [wordnet_lemmatizer.lemmatize(spell(word)) for word in sentence if word not in all_stops]
    else:
        filtered_words = [wordnet_lemmatizer.lemmatize(word) for word in sentence if word not in all_stops]
    filtered_words = filtered_words[:][1:]
    return filtered_words

def get_answers(path):
    with codecs.open(path, "r") as fIn:
        answers = [ answer.rstrip() for answer in fIn.readlines()]
        return answers

def get_grades(path):
    with codecs.open(path, "r") as fIn:
        grades = [ grade.rstrip() for grade in fIn.readlines()]
        return grades

def main():
    question_number = [7,7,7,7,4,7,7,7,7,7,10,10]
    correct_answers = get_answers('../../data/Mohler/raw/answers')
    for i in range(len(correct_answers)):
        correct_answers[i] = filter(correct_answers[i], True)
    correct_answer_index = 0
    answer_array = []
    for chapter in range(1, len(question_number) + 1):
        for question in range(1, question_number[chapter - 1] + 1):
            answers = get_answers('../../data/Mohler/raw/' + str(chapter) + '.' + str(question))
            grades = get_grades('../../data/Mohler/scores/' + str(chapter) + '.' + str(question) + '/other')
            answer_array.append([answers, grades])
            correct_answer_index += 1

    question_index = 0
    scores_diff_split = []
    scores = []
    clf = RandomForestClassifier(n_estimators=25)
    for question_index in range(87):
        print(question_index)
        for i in range(len(answer_array[question_index][1])):
            answer_array[0][0][i] = filter(answer_array[question_index][0][i], True)
        reference = [[[' '.join(word for word in correct_answers[question_index])]]]
        features = []
        for i in range(len(answer_array[question_index][1])):
            # BLUE calculation:
            BLEU_score = nltk.translate.bleu_score.sentence_bleu([correct_answers[question_index]], answer_array[question_index][0][i])
            # ROUGE calculation:
            #print(answer_array[question_index][0][i])
            summary = [[' '.join(word for word in answer_array[question_index][0][i])]]
            rouge = Pythonrouge(summary_file_exist=False,
                                summary=summary, reference=reference,
                                n_gram=2, ROUGE_SU4=True, ROUGE_L=True,
                                recall_only=True, stemming=False, stopwords=False,
                                word_level=True, length_limit=True, length=50,
                                use_cf=False, cf=95, scoring_formula='average',
                                resampling=True, samples=1000, favor=True, p=0.5)
            ROUGE_score = rouge.calc_score()
            # {'ROUGE-1': 0.16667, 'ROUGE-2': 0.0, 'ROUGE-SU4': 0.05}
            features.append( [list(ROUGE_score.values())[0], list(ROUGE_score.values())[1], BLEU_score])
            #features[i].append(BLEU_score)
            #features.append(BLEU_score)
            #print(answer_array[question_index][1][i])
            #print(features[i])
        # diff_scores = []
        # for split_position in range(3, 24, 2):
        split_position = 13 # for it gives the best accuracy
        #features = np.asarray(features)
        #features = features.reshape(-1, 1)
        clf.fit(features[0:split_position], answer_array[question_index][1][0:split_position])
        print("RF")
        scores.append(clf.score(features[split_position:len(answer_array[question_index][1])],
                        answer_array[question_index][1][split_position:len(answer_array[question_index][1])]))
        print(scores[question_index])
        #     diff_scores.append(clf.score(features[split_position:len(answer_array[question_index][1])],
        #                         answer_array[question_index][1][split_position:len(answer_array[question_index][1])]))
        # scores_diff_split.append(diff_scores)


    print("average")
    average_score = sum(scores)/len(scores)
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