#!/usr/bin/env python3

import itertools
import re
from cached_property import cached_property


def parse_corpus(documents):
    corpus = {document['id']:_create(document) for document in documents}

    for document in corpus.values():
        _expand(document, corpus)

    corpus = list(corpus.values())
    return corpus


def find_roots(corpus):
    return [document for document in corpus if document.is_root]


def prune(corpus, to_prune):
    to_remove = []
    for document in to_prune:
        if document.parent:
            document.parent.replies.remove(document)
        to_remove += document.flatten()

    to_remove = set(to_remove)

    corpus = [item for item in corpus if item not in to_remove]
    return corpus


class Document:
    def __init__(self, id, author, content, reply_to=None, replies=None, **kwargs):
        self.__dict__.update(kwargs)
        self.id = id
        self.author = author
        self.content = content
        if reply_to:
            self.parent_id = reply_to
        else:
            self.parent_id = None
        self.parent = None
        if replies:
            self.reply_ids = replies
        else:
            self.reply_ids = None
        self.replies = []

    @cached_property
    def filtered_content(self):
        content = self.content
        lines = content.split('\n')
        for i, line in enumerate(lines):
            ## remove quotes
            if len(line) > 0 and line[0] == '>':
                lines[i] = ""
        return ' '.join(lines)

    @cached_property
    def is_root(self):
        return self.parent == None

    def flatten(self):
        return [self] + list(itertools.chain(*[reply.flatten() for reply in self.replies]))

    @cached_property
    def size(self):
        return 1 + sum([reply.size() for reply in self.replies])

    def __repr__(self):
        return "({}){} {} [{}] \n\"\"\"\n{}\n\"\"\"".format(
            self.id, '*' if self.is_root else '',
            self.author, len(self.replies), self.content
        )

def _create(document_json):
    return Document(**document_json)


def _expand(document, corpus):
    if document.reply_ids: ## if the corpus includes replies to the document
        document.replies = [corpus[id] for id in document.reply_ids]
        for reply in document.replies:
            reply.parent_id = document.id
            reply.parent = document
    elif document.parent_id: ## if the corpus includes the parent of each document
        document.parent = corpus[document.parent_id]
        document.parent.replies.append(document)
