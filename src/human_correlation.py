import codecs
import difflib

def get_grades(path):
    with codecs.open(path, "r") as fIn:
        grades = [ grade.rstrip() for grade in fIn.readlines()]
        return grades

question_number = [7,7,7,7,4,7,7,7,7,7,10,10]
correct_answer_index = 0
grades_array = []
grades_corr = []
for chapter in range(1, len(question_number) + 1):
    for question in range(1, question_number[chapter - 1] + 1):
        grades_me = get_grades('../data/Mohler/scores/' + str(chapter) + '.' + str(question) + '/me')
        grades_other = get_grades('../data/Mohler/scores/' + str(chapter) + '.' + str(question) + '/other')
        sm = difflib.SequenceMatcher(None, grades_me, grades_other)
        #print(str(chapter) + '.' + str(question))
        #print(sm.ratio())
        grades_array.append([grades_me, grades_other])
        grades_corr.append(sm.ratio())
print(sum(grades_corr)/len(grades_corr))