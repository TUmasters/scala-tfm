import itertools
from collections import deque
import random


class Comment:
    def __init__(self, id, author, content, parent, replies):
        self.id = id
        self.author = author
        self.content = content
        if parent:
            self.parent_id = parent
        else:
            self.parent_id = None
        self.parent = None
        if replies:
            self.reply_ids = replies
        else:
            self.reply_ids = None
        self.replies = []

    def remove(self):
        if(self.parent):
            self.parent.replies.remove(self)

    def collect(self):
        return [self.id] + list(itertools.chain(*[reply.collect() for reply in self.replies]))

    def size(self):
        return 1 + sum([reply.size() for reply in self.replies])
    def __iter__(self):
        q = deque([])
        q.append(self)
        while len(q) > 0:
            item = q.popleft()
            for com in item.replies:
                q.append(com)
            yield item
        


def create_comment(d):
    return Comment(d['id'], d['author'], d['words'], d['parent'], d['replies'] if 'replies' in d else None)


def create_comment_from_raw(d):
    return Comment(d['id'], d['author'], d['content'], None, d['replies'] if 'replies' in d else None)


def expand(comment,comments):
    if comment.reply_ids:
        comment.replies = [comments[id] for id in comment.reply_ids]
        for reply in comment.replies:
            reply.parent_id = comment.id
            reply.parent = comment
    elif comment.parent_id:
        comment.parent = comments[comment.parent_id]
        comment.parent.replies.append(comment)
    # comment.replies = [comments[id] for id in comment._replies]
    # for reply in comment.replies:
    #     reply.parent = comment

# def expand_tree(comment):
#     replies = list(comments[comment.id]['replies'])
#     comment.replies = [create_comment(id) for id in replies]
#     for reply in comment.replies:
#         replies += expand_tree(reply)
#     return replies


def random_color():
    levels = range(32, 256, 32)
    return tuple(random.choice(levels) for _ in range(3))
