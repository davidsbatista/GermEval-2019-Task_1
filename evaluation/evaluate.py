#!/usr/bin/python2.7
# coding: utf8

# Copyright 2019 Language Technology Lab, Universität Hamburg
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Filename: evaluate_actual.py
# Authors:  Rami Aly, Steffen Remus and Chris Biemann
# Description: CodaLab's evaluation script for GermEval-2019 Task 1: Shared task on hierarchical classification of blurbs.
#    For more information visit https://competitions.codalab.org/competitions/21226.
# Requires: sklearn

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
import os.path
from utils import subtask_A_evaluation, subtask_B_evaluation
import io

def readfile(fname):
    data_a = {}
    data_b = {}
    current_task = None
    #with open(fname, 'r') as fin:
    try:
        fin = io.open(fname, mode="r", encoding="utf-8")
    except UnicodeError:
        message = "String is not UTF-8"
        print(message, file = sys.stderr)
        raise Exception(message)

    with fin:
        task = fin.readline()
        task = task.strip().lower()

        def read_task(stopping_word):
            data = {}
            for line_i, line in enumerate(fin):
                line = line.strip()
                if not line: # skip empty lines
                    continue
                if line == stopping_word:
                    return data
                content = line.split('\t')

                id = content[0]
                labels = content[1:]
                if not labels:
                    message = "WARNING: ISBN '{:s}' has no label assignment".format(id)
                    print(message, file=sys.stderr)

                if id in data:
                    message = 'Error. Duplicate id {:s}.'.format(id)
                    print(message, file=sys.stderr)
                    raise Exception(message)
                data[id] = labels
            return data

        #Checks the subtask described in header
        if task == 'subtask_a':
            data_a = read_task('subtask_b')
            data_b = read_task('subtask_a')
        elif task == 'subtask_b':
            data_b = read_task('subtask_a')
            data_a = read_task('subtask_b')
        else:
            message = "Error while reading header. Please make sure to specify either subtask_a or subtask_b."
            print(message, file = sys.stderr)
            raise Exception(message)
    return [data_a, data_b]


def allign_sub_to_truth(truth_data, submission_data):
    """
    Matches IDs of submission to IDs in the truth file.
    """
    ordered_submission = []
    ordered_truth = []

    for key in submission_data:
        if key not in truth_data:
            message = "Error. ISBN '{:s}' was not found in the system.".format(key)
            print(message, file=sys.stderr)
            raise Exception(message)

    for key in truth_data:
        ordered_truth.append(truth_data[key])
        if key in submission_data:
            ordered_submission.append(submission_data[key])
        else:
            ordered_submission.append([])

    return [ordered_truth, ordered_submission]

input_dir = sys.argv[1]
output_dir = sys.argv[2]
submit_file = os.path.join(input_dir, 'answer.txt')
truth_file = os.path.join(input_dir, 'gold.txt')

if not os.path.isfile(submit_file):
    print("File '{:s}' doesn't exist!".format(submit_file))
    exit(1)

if not os.path.isfile(truth_file):
    print("File '{:s}' doesn't exist!".format(truth_file))
    exit(1)

truth_data_a, truth_data_b = readfile(truth_file)

submission_data_a, submission_data_b = readfile(submit_file)


true_output_a, submission_output_a = allign_sub_to_truth(truth_data_a, submission_data_a)
true_output_b, submission_output_b = allign_sub_to_truth(truth_data_b, submission_data_b)

recall_a, precision_a, f1_a, acc_a = subtask_A_evaluation(true_output_a, submission_output_a)
set1_score = f1_a
recall_b, precision_b, f1_b, acc_b = subtask_B_evaluation(true_output_b, submission_output_b)
set2_score = f1_b

output_filename = os.path.join(output_dir, 'scores.txt')
output_detailed_filename = os.path.join(output_dir, 'scores.html')

#write the scores into scores.txt file
with open(output_filename, 'w') as fout:
    print('correct:1', file=fout)
    print('set1_score:'+ str(set1_score), file=fout)
    print('set2_score:'+ str(set2_score), file=fout)

with open(output_detailed_filename, 'w') as fout:
    print("Subtask_a detailed scores:<br> Recall: '%0.4f' <br> Precision: '%0.4f' <br> F1_micro: '%0.4f'<br> Subset_acc: '%0.4f'<br><br>"%(recall_a, precision_a, f1_a, acc_a), file = fout)
    print("Subtask_b detailed scores:<br> Recall: '%0.4f' <br> Precision: '%0.4f' <br> F1_micro: '%0.4f'<br> Subset_acc: '%0.4f'<br><br>"%(recall_b, precision_b, f1_b, acc_b), file = fout)
