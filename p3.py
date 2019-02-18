#!/usr/bin/env python3
import re
import math
import random
import os
import sys

WORD = re.compile(r'\w+')

START1 = '$'
START2 = '^'
STOP1 = '@'
STOP2 = '&'
UNK = '#'
UNK_PROP = 0.1
K = 10
UNK_MODE = os.getenv("UNK_MODE")

if os.getenv("UNK_PROP") != None:
    UNK_PROP = float(os.getenv("UNK_PROP"))
    print("setting UNK_PROP=%s" % UNK_PROP)

def unkify_unigram(c, cx):
    if UNK_MODE == "LT5":
        q = { w : cw / float(c) for w, cw in cx.items() if cw >= 5 }
        q[UNK] = sum([cw for w, cw in cx.items() if cw < 5]) / float(c)
        return q
    if UNK_MODE == "LT2":
        q = { w : cw / float(c) for w, cw in cx.items() if cw >= 2 }
        q[UNK] = sum([cw for w, cw in cx.items() if cw < 2]) / float(c)
        return q
    q = { w : cw / float(c) for w, cw in cx.items() }
    qs = sorted(q.values())
    unk_q_max = qs[int(len(qs) * UNK_PROP)]
    unk_q = sum([v for v in qs if v <= unk_q_max])
    q = { k : v for k, v in q.items() if v > unk_q_max }
    q[UNK] = unk_q
    
    return q

def unkify_ngram(cx, n, all_vocab):
    assert n == 2 or n == 3
    V = len(all_vocab)
    base_keys = set([key[1:] for key in cx.keys() ])
    sub_cxs = { key : {} for key in base_keys }
    for w, cw in cx.items():
        sub_cxs[w[1:]][w] = cw
    all_q = {}
    bases = {}
    for base_key in base_keys:
        sub_cx = sub_cxs[base_key]
        c = sum([cw for w, cw in sub_cx.items() ])
        base = float(c + K * V)
        q = { w : (cw + K) / base for w, cw in sub_cx.items() }
        for k, v in q.items():
            all_q[k] = v
        bases[base_key] = c
    return all_q, bases

def unigram(data):
    cx = {}
    for line in data['corpus']:
        for word in line:
            if word != START1 and word != START2:
                if word in cx:
                    cx[word] += 1
                else:
                    cx[word] = 1
    q = unkify_unigram(data['total_words'], cx)
    log2q = { w : math.log2(v) for w, v in q.items() }
    def log2p(words):
        ret = 0.0
        for w in words:
            if w != START1 and word != START2:
                if w not in log2q:
                    ret += log2q[UNK]
                    continue
                ret += log2q[w]
        return ret
    vocab = set(log2q.keys()) - set([UNK])
    return q, vocab, log2p

def get_full_vocab(data):
    vocab = set()
    for line in data['corpus']:
        for word in line:
            if word != START1 and word != START2:
                vocab.add(word)
    return vocab

def get_value_tri(x_i_plus_2, x_i_plus_1, x_i, log2q, V, bases):
    for x_i_plus_2_ in (x_i_plus_2, UNK):
        for x_i_plus_1_ in (x_i_plus_1, UNK):
            for x_i_ in (x_i, UNK):
                key = (x_i_plus_2_, x_i_plus_1_, x_i_)
                if key in log2q:
                    return log2q[key]
    if (x_i_plus_1, x_i) in bases:
        return math.log2(K / float(bases[(x_i_plus_1, x_i)] + K * V))
    else:
        return math.log2(1.0 / V)

def trigram(data, all_vocab):
    cw = {}
    for line in data['corpus']:
        for i in range(len(line) - 2):
            x_i = line[i]
            x_i_plus_1 = line[i+1]
            x_i_plus_2 = line[i+2]
            key = (x_i_plus_2, x_i_plus_1, x_i)
            if key in cw:
                cw[key] += 1
            else:
                cw[key] = 1
    q, bases = unkify_ngram(cw, 3, all_vocab)
    log2q = { w : math.log2(v) for w, v in q.items() }
    def log2p(line):
        ret = 0.0
        for i in range(len(line) - 2):
            x_i = line[i]
            x_i_plus_1 = line[i+1]
            x_i_plus_2 = line[i+2]
            ret += get_value_tri(x_i_plus_2, x_i_plus_1, x_i, log2q, len(all_vocab), bases)
        return ret
    return log2p

def perplexity(data, log2p):
    M = data['total_words']
    l = sum(map(log2p, data['corpus'])) / float(M)
    return math.pow(2, -l)

def load_data(filepath):
    data = []
    n = 0
    text = open(filepath, "r").read()
    for line in text.split("\n"):
        words = WORD.findall(line)
        if len(words) == 0:
            continue
        n += len(words) + 2
        data.append([START1, START2] + words + [STOP1, STOP2])
    return { 'corpus': data, 'total_words': n }

def transform_corpus(data, known_vocab):
    corpus = []
    for line in data['corpus']:
        corpus.append([w if w in known_vocab else UNK for w in line])
    data['corpus'] = corpus

def random_split(data, first_prop):
    datas = []
    datas.append({"total_words": 0, "corpus": [] })
    datas.append({"total_words": 0, "corpus": [] })
    
    shuffled = random.sample(data["corpus"], len(data["corpus"]))
    first_prop_len = int(len(data["corpus"]) * first_prop)
    
    for i in range(first_prop_len):
        line = shuffled[i]
        datas[0]["total_words"] += (len(line) - 2)
        datas[0]["corpus"].append(line)
    
    for i in range(first_prop_len, len(data["corpus"])):
        line = shuffled[i]
        datas[1]["total_words"] += (len(line) - 2)
        datas[1]["corpus"].append(line)
    return datas

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("help: python3 q2.py [mode] (half)\n\tmode: dev, test")
        sys.exit(-1)
    if len(sys.argv) > 2 and sys.argv[2] == "half":
        train_data = random_split(load_data("data/brown.train.txt"), 0.5)[0]
    else:
        train_data = load_data("data/brown.train.txt")
    test_data = load_data("data/brown." + sys.argv[1] + ".txt")
    uni_q, known_vocab, uniq_log2p = unigram(train_data)
    vocab = get_full_vocab(test_data)
    all_vocab = known_vocab.union(test_data)
    print("K: %s" % K)
    print("vocab size of test set: %s" % len(vocab))
    print("known vocab size in train set: %s" % len(known_vocab))
    print("corpus size of train set: %s" % train_data["total_words"])
    transform_corpus(test_data, known_vocab)
    print("trigram: %f" % perplexity(test_data, trigram(train_data, all_vocab)))
