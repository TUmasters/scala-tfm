#!/usr/bin/env python3

import json
import sys
import os

#Set default parameter names
root = 'data/'
protocol = 'uatfm'
source = 'comments.json'

#Load parameters from command line arguments
if len(sys.argv) >= 2:
  root = sys.argv[1]
if len(sys.argv) >= 3:
  protocol = sys.argv[2]
if len(sys.argv) >= 4:
  source = sys.argv[3]


#Create output directory in case it doesn't exist
output = root+'/graphics'
os.mkdir(output)


#Load in json files
with open(root+source, 'r') as f:
  structure = json.load(f)

with open(root+protocol+'/words.json', 'r') as f:
  topics = json.load(f)

topics = topics['documents']

print(structure)
