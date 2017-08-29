#!/usr/bin/env python3

import json
import sys
import os
from _utils import *

#Set default parameter names
root = 'data/'
protocol = 'plsa'
source = 'comments.json'

#Load parameters from command line arguments
if len(sys.argv) >= 2:
    root = sys.argv[1]
if len(sys.argv) >= 3:
    protocol = sys.argv[2]
if len(sys.argv) >= 4:
    source = sys.argv[3]


#Create output directory in case it doesn't exist
output = root+'graphics/'
if not os.path.isdir(output):
    os.mkdir(output)


#Load in json files
with open(root+source, 'r') as f:
    dataset = json.load(f)

with open(root+protocol+'/words.json', 'r') as f:
    topics = json.load(f)

topics = topics['documents']

comments = dict([(d['id'], create_comment(d)) for d in dataset])
for id in comments:
    expand(comments[id], comments)
roots = list(filter(lambda comment: comment.parent is None, comments.values()))

print(roots)
