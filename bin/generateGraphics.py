#!/usr/bin/env python3

import json
import sys
import os
from _utils import *

#Set default parameter names
root = 'data/'
protocol = 'plsa'
source = 'comments.json'
rawFile = 'raw.json'

#Load parameters from command line arguments
if len(sys.argv) >= 2:
    root = sys.argv[1]
if len(sys.argv) >= 3:
    protocol = sys.argv[2]
if len(sys.argv) >= 4:
    source = sys.argv[3]
if len(sys.argv) >= 5:
    rawFile = sys.argv[4]


#Create output directory in case it doesn't exist
output = root+'graphics/'
if not os.path.isdir(output):
    os.mkdir(output)


#Load in json files
with open(root+source, 'r') as f:
    dataset = json.load(f)

with open(root+protocol+'/words.json', 'r') as f:
    topics = json.load(f)
with open(root+rawFile, 'r') as f:
    raw = json.load(f)


topics = topics['documents']

#Generate Tree Structures of Conversations
comments = dict([(d['id'], create_comment(d)) for d in dataset])
for id in comments:
    expand(comments[id], comments)
roots = list(filter(lambda comment: comment.parent is None, comments.values()))

rawComments = dict([(d['id'], create_comment_from_raw(d)) for d in raw])

for i in range(len(roots)):
    if i % 100 == 0:
        print("Working on batch ", int(i/100), "of", int(len(roots)/100), "...")
    if not os.path.isdir(output+str(i)):
        os.mkdir(output+str(i))
    # Generate Graphviz Header
    graphviz = "digraph G {\n"
    # Generate Latex Header
    latexHeader = "\\documentclass{article}\n"
    latexHeader += "\\usepackage{graphicx}\n"
    latexHeader += "\\usepackage{floatrow}\n"
    latexHeader += "\\usepackage{color}\n"
    # latexHeader += "\\usepacakge{spverbatim}\n"
    # Create String for Latex Color Definitions
    latexColor = ""
    # Generate Latex Document Header
    latexDoc = "\\begin{document}\n"
    latexDoc += "\\begin{center}\n"
    latexDoc += "\\includegraphics{" + output+str(i)+"/conversation.pdf}\n"
    latexDoc += "\\end{center}\n"
    commentDict = {roots[i].id: 0}
    numColors = 0
    for comment in roots[i]:
        # Generate Latex Document
        words = topics[comment.id]
        latexDoc += "\\begin{spverbatim}\n"+str(commentDict[comment.id]) + " " + rawComments[comment.id].content + "\n\\end{spverbatim}\n"
        # Make word labels
        for w in words:
            while numColors <= w['t']:
                color = random_color()
                latexColor += "\\definecolor{color"+str(numColors)+"}{RGB}{"+str(color[0])+", "+str(color[1])+", "+str(color[2])+"}\n"
                numColors += 1
            latexDoc += "\\textcolor{color"+str(w['t']) + "}{" + w['w'] + "} "
        latexDoc += "\\break\\break\n"
        # Construct Graph
        for reply in comment.replies:
            if reply.id not in commentDict:
                commentDict[reply.id] = len(commentDict)
            graphviz += "\t" + str(commentDict[comment.id]) + " -> " + str(commentDict[reply.id]) + ";\n"
    latexDoc += "\\end{document}"
    graphviz += "}"
    with open(output+str(i)+"/conversation.dot",'w') as f:
        f.write(graphviz)
    latex = latexHeader+latexColor+latexDoc
    with open(output+str(i)+"/figure.tex", 'w') as f:
        f.write(latex)

    os.system("dot -Tpdf " + output + str(i) + "/conversation.dot -o " + output + str(i) + "/conversation.pdf")
    os.system('pdflatex -interaction=nonstopmode -output-directory ' + output+str(i)+"/ " + output+str(i) + "/figure.tex > /dev/null")
            



