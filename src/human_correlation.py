import codecs
import difflib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools

plt.rcParams.update({'font.size': 14})

def get_grades(path):
    with codecs.open(path, "r") as fIn:
        grades = [ grade.rstrip() for grade in fIn.readlines()]
        return grades

question_number = [7,7,7,7,4,7,7,7,7,7,10,10]
correct_answer_index = 0
grades_array = []
grades_corr = []
grades_all_me = []
grades_all_other = []
class_names = []
for chapter in range(1, len(question_number) + 1):
    for question in range(1, question_number[chapter - 1] + 1):
        grades_me = get_grades('../data/Mohler/scores/' + str(chapter) + '.' + str(question) + '/me')
        grades_other = get_grades('../data/Mohler/scores/' + str(chapter) + '.' + str(question) + '/other')
        sm = difflib.SequenceMatcher(None, grades_me, grades_other)
        #print(str(chapter) + '.' + str(question))
        #print(sm.ratio())
        grades_array.append([grades_me, grades_other])
        for i in range(len(grades_me)):
            grades_all_me.append(grades_me[i])
            grades_all_other.append(grades_other[i])
        grades_corr.append(sm.ratio())
print(sum(grades_corr)/len(grades_corr))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('First grader')
    plt.xlabel('Second grader')

# Compute confusion matrix


class_names = ['0','1', '2','3','4','5']
print(grades_all_me[0:10])
cnf_matrix = confusion_matrix(grades_all_me, grades_all_other, labels=['0','1', '2', '3', '4', '5'])
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()