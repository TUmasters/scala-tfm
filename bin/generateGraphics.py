#!/usr/bin/env python3

import json
import sys
import os
from time import time

Sstructure = 'data/comments.json'
Stopics = 'data/plsa/words.json'
output = str(time())

if len(sys.argv) >= 2:
  output = sys.argv[1]
if len(sys.argv) >= 3:
  structure = sys.argv[2]
if len(sys.argv) >= 4:
  topics = sys.argv[3]

output = 'graphics/'+output


with open(Sstructure,'r') as f:
  structure = json.load(f)

with open(Stopics, 'r') as f:
  topics = json.load(f)

topics = topics['documents']

os.mkdir(output)



convNum = 0

