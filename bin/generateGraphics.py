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

#Generate Tree Structures of Conversations
comments = dict([(d['id'], create_comment(d)) for d in dataset])
for id in comments:
    expand(comments[id], comments)
roots = list(filter(lambda comment: comment.parent is None, comments.values()))

for i in range(len(roots)):
    if len(roots[i].replies) < 2: continue
    if not os.path.isdir(output+str(i)):
        os.mkdir(output+str(i))
    graphviz = "digraph G {\n"
    latex = "\\documentclass{article}\n"
    latex += "\\usepackage{graphicx}\n"
    latex += "\\begin{document}\n"
    commentNum = 0
    commentDict = {}
    for comment in roots[i]:
        if comment.id not in commentDict:
            commentDict[comment.id] = commentNum
            commentNum += 1
        for reply in comment.replies:
            if reply.id not in commentDict:
                commentDict[reply.id] = commentNum
                commentNum += 1
            graphviz += "\t" + str(commentDict[comment.id]) + " -> " + str(commentDict[reply.id]) + ";\n"
    graphviz += "}"
    with open(output+str(i)+"/conversation.dot",'w') as f:
        f.write(graphviz)
    os.system("dot -Tps " + output + str(i) + "/conversation.dot -o " + output + str(i) + "/conversation.ps")
    break
            



