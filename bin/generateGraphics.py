#!/usr/bin/env python3

import json
import sys
import os
from _utils import *
import numpy as np
import csv


# Set default parameter names
root = 'data'
protocol = 'uatfm'
source = 'documents.json'
rawFile = 'raw.json'

# Load parameters from command line arguments
if len(sys.argv) >= 2:
    root = sys.argv[1]
if len(sys.argv) >= 3:
    protocol = sys.argv[2]
if len(sys.argv) >= 4:
    source = sys.argv[3]
if len(sys.argv) >= 5:
    rawFile = sys.argv[4]


# Create output directory in case it doesn't exist
output = root+'/graphics/'+protocol+"/"
if not os.path.isdir(root+'/graphics'):
    os.mkdir(root+'/graphics')
if not os.path.isdir(output):
    os.mkdir(output)


# Load in json files
with open(root+'/'+source, 'r') as f:
    dataset = json.load(f)
with open(root+'/' + rawFile, 'r') as f:
    raw = json.load(f)


# Generate Tree Structures of Conversations
comments = dict([(d['id'], create_comment(d)) for d in dataset])
for id in comments:
    expand(comments[id], comments)
roots = list(filter(lambda comment: comment.parent is None, comments.values()))

rawComments = dict([(d['id'], create_comment_from_raw(d)) for d in raw])


if protocol == 'plsa':
    with open(root+protocol+'/words.json', 'r') as f:
        topics = json.load(f)
    topics = topics['documents']
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
            latexDoc += "\\hfill\\break\\hfill\\break\n"
            # Construct Graph
            for reply in comment.replies:
                if reply.id not in commentDict:
                    commentDict[reply.id] = len(commentDict)
                graphviz += "\t" + str(commentDict[comment.id]) + " -> " + str(commentDict[reply.id]) + ";\n"
        latexDoc += "\\end{document}"
        graphviz += "}"
        with open(output+str(i)+"/conversation.dot", 'w') as f:
            f.write(graphviz)
        latex = latexHeader+latexColor+latexDoc
        with open(output+str(i)+"/figure.tex", 'w') as f:
            f.write(latex)

        os.system("dot -Tpdf " + output + str(i) + "/conversation.dot -o " + output + str(i) + "/conversation.pdf")
        os.system('pdflatex -interaction=nonstopmode -output-directory ' + output+str(i)+"/ " + output+str(i) + "/figure.tex > /dev/null")
else:
    with open(root+'/'+protocol+'/document-topics.json', 'r') as f:
        documentTopics = json.load(f)
    # with open(root+'/'+protocol+'/word-topics.json', 'r') as f:
    #     wordTopics = json.load(f)
    with open(root + '/corpus/words.csv', 'r') as f:
        wreader = csv.reader(f, delimiter=' ', skipinitialspace=True)
        words = {}
        for row in wreader:
            words[row[1]] = int(row[0])
    with open(root + '/' + protocol + '/theta.mat', 'r') as f:
        theta = np.loadtxt(f, delimiter=",")
    with open(root + '/' + protocol + '/params.json', 'r') as f:
        params = json.load(f)

    # Estimate topic probabilities
    pi = np.zeros(params['num-topics'])
    for comment in documentTopics.values():
        pi[comment[0]['topic']] += 1
    pi /= len(documentTopics)

    stheta = theta * pi
    stheta = stheta / np.sum(stheta, axis=0)

    # Construct list of word topics
    topics = []
    for k in range(params['num-topics']):
        topics.append([(stheta[w_id, k], w) for w, w_id in words.items()])
    # for w in wordTopics:
    #     for param in wordTopics[w]:
    #         topics[param['topic']].append((param['p'],w))
    for i in range(len(topics)):
        topics[i].sort()
        topics[i].reverse()
    # Construct colors
    colors = [random_color() for _ in range(params['num-topics'])]
    # Now we start generating graphics
    for i, root in enumerate(roots):
        if i % 100 == 0:
            print("Working on batch ", int(i/100), "of", int(len(roots)/100), "...")
        if root.size <= 2:
            continue
        if len(root.topic_span(documentTopics)) <= 1:
            continue
        if not os.path.isdir(output+str(i)):
            os.mkdir(output+str(i))
        # Generate Graphviz Header
        graphviz = "digraph G {\n"
        # Generate Latex Header
        latexHeader = "\\documentclass{article}\n"
        latexHeader += "\\usepackage{graphicx}\n"
        latexHeader += "\\usepackage{floatrow}\n"
        latexHeader += "\\usepackage{color}\n"
        latexHeader += "\\usepackage{spverbatim}\n"
        # latexHeader += "\\usepacakge{spverbatim}\n"
        # Create String for Latex Color Definitions
        latexColor = ""
        for j in range(len(colors)):
            latexColor += "\\definecolor{color"+str(j)+"}{RGB}{"+str(colors[j][0])+", "+str(colors[j][1])+", "+str(colors[j][2])+"}\n"
        # Generate Latex Document Header
        latexDoc = "\\begin{document}\n"
        latexDoc += "\\begin{center}\n"
        latexDoc += "\\includegraphics[width=0.9\\textwidth,height=0.9\\textheight,keepaspectratio]{" + output+str(i)+"/conversation.pdf}\n"
        latexDoc += "\\end{center}\n"
        commentDict = {root.id: 0}
        numColors = 0
        for comment in root:
            # Generate Latex Document
            latexDoc += "\\begin{spverbatim}\n"+str(commentDict[comment.id]) + " " + rawComments[comment.id].content + "\n\\end{spverbatim}+\\hfill\\break\\hfill\\break\n"
            # Generate colors for topic
            topic = documentTopics[comment.id][0]['topic']
            # Make word labels
            tCount = 0
            for j in range(len(topics[topic])):
                if topics[topic][j][1] in comment.content:
                    latexDoc += "\\textcolor{color"+str(topic) + "}{" + topics[topic][j][1] + "} "
                    tCount += 1
                if tCount >= 5:
                    break
            latexDoc += "\\hfill\\break\\hfill\\break\n"
            # Construct Graph
            color = colors[topic]
            graphviz += "\t" + str(commentDict[comment.id]) + " [style=filled fillcolor = \"#" + convertToHex(color[0]) + convertToHex(color[1]) + convertToHex(color[2])  + "\"]\n"
            for reply in comment.replies:
                if reply.id not in commentDict:
                    commentDict[reply.id] = len(commentDict)
                graphviz += "\t" + str(commentDict[comment.id]) + " -> " + str(commentDict[reply.id]) + ";\n"
        latexDoc += "\\end{document}"
        graphviz += "}"
        with open(output+str(i)+"/conversation.dot", 'w') as f:
            f.write(graphviz)
        latex = latexHeader+latexColor+latexDoc
        with open(output+str(i)+"/figure.tex", 'w') as f:
            f.write(latex)

        os.system("dot -Tpdf " + output + str(i) + "/conversation.dot -o " + output + str(i) + "/conversation.pdf")
        os.system('pdflatex -interaction=nonstopmode -output-directory ' + output+str(i)+"/ " + output+str(i) + "/figure.tex > /dev/null")
