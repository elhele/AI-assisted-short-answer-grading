import re
import os
import codecs

data_folder = '../data/Mohler/nbgrader_workspace/'

filenames = os.listdir(data_folder + "submitted/")  # get all files' and folders' names in the current directory

student_ids = []
for filename in filenames:  # loop through all the files and folders
    if os.path.isdir(
            os.path.join(os.path.abspath(data_folder + "submitted/"), filename)):  # check whether the current object is a folder or not
        student_ids.append(filename)

student_ids.sort()

all_answers = {}
for student_id in student_ids:
    student_answers = {}
    with codecs.open(data_folder + "submitted/" + student_id + "/DataStructures/DataStructures.ipynb", "r") as fIn:
        strings = [strings.rstrip() for strings in fIn.readlines()]
        answers = []
        for string in strings:
            matches = re.findall(r'\"(.+?)\"', string)
            if(matches and matches[0][0].isdigit()):
                student_answers[matches[0][0:3]] = matches[0]
        all_answers[student_id] = student_answers

for answer_key in student_answers:
    answers_to_question = open(data_folder + "forAutograder/raw/" + answer_key, 'w')
    for student_key in all_answers:
        answers_to_question.writelines(all_answers[student_key][answer_key] + "\n")
    answers_to_question.close()