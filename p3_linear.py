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
K = 0.1
GAMMA = 0.1
if os.getenv("UNK_PROP") != None:
    UNK_PROP = float(os.getenv("UNK_PROP"))
    print("setting UNK_PROP=%s" % UNK_PROP)

def unkify_unigram(c, cx):
    q = { w : cw / float(c) for w, cw in cx.items() }
    qs = sorted(q.values())
    unk_q_max = qs[int(len(qs) * UNK_PROP)]
    unk_q = sum([v for v in qs if v <= unk_q_max])
    q = { k : v for k, v in q.items() if v > unk_q_max }
    q[UNK] = unk_q
    return q

def unkify_bigram(cx):
    base_keys = set([key[1:] for key in cx.keys() ])
    sub_cxs = { key : {} for key in base_keys }
    for w, cw in cx.items():
        sub_cxs[w[1:]][w] = cw
    all_q = {}
    for base_key in base_keys:
        sub_cx = sub_cxs[base_key]
        c = sum([cw for w, cw in sub_cx.items() ])
        q = { w : cw / float(c) for w, cw in sub_cx.items() }
        for k, v in q.items():
            all_q[k] = v
    return all_q

def try_get(k, d):
    if k in d:
        return d[k]
    else:
        return 0

def unkify_trigram(cx, biq, uniq, biqc, uniqc):
    base_keys = set([key[1:] for key in cx.keys() ])
    sub_cxs = { key : {} for key in base_keys }
    for w, cw in cx.items():
        sub_cxs[w[1:]][w] = cw
    all_q = {}
    for base_key in base_keys:
        sub_cx = sub_cxs[base_key]
        c = sum([cw for w, cw in sub_cx.items() ])
        l1 = try_get(base_key, biqc) / (GAMMA + try_get(base_key, biqc))
        l2 = (1 - l1) * (try_get(base_key[0], uniqc) / (try_get(base_key[0], uniqc) + GAMMA))
        l3 = 1 - l1 - l2
        # print("c(%s, %s) - l1: %s, l2: %s, l3: %s" % (base_key[1], base_key[0], l1, l2, l3))
        q = { w : (l1 * cw / float(c) + l2 * biq[base_key] + l3 * try_get(w[0], uniq)) for w, cw in sub_cx.items() }
        for k, v in q.items():
            all_q[k] = v
    return all_q

def bigram_q(data):
    cw = {}
    for line in data['corpus']:
        for i in range(len(line) - 2):
            x_i = line[i]
            x_i_plus_1 = line[i+1]
            key = (x_i_plus_1, x_i)
            if key in cw:
                cw[key] += 1
            else:
                cw[key] = 1
    q = unkify_bigram(cw)
    return q, cw

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
    return q, vocab, log2p, cx

def get_full_vocab(data):
    vocab = set()
    for line in data['corpus']:
        for word in line:
            if word != START1 and word != START2:
                vocab.add(word)
    return vocab

def get_value_tri(x_i_plus_2, x_i_plus_1, x_i, log2q, biq, uniq, biqc, uniqc):
    for x_i_plus_2_ in (x_i_plus_2, UNK):
        for x_i_plus_1_ in (x_i_plus_1, UNK):
            for x_i_ in (x_i, UNK):
                key = (x_i_plus_2_, x_i_plus_1_, x_i_)
                if key in log2q:
                    return log2q[key]
    base_key = (x_i_plus_1, x_i)
    l1 = try_get(base_key, biqc) / (GAMMA + try_get(base_key, biqc))
    l2 = (1 - l1) * (try_get(base_key[0], uniqc) / (try_get(base_key[0], uniqc) + GAMMA))
    l3 = 1 - l1 - l2
    return l2 * try_get(base_key, biq) + l3 * try_get(x_i_plus_2, uniq)

def trigram(data, uniq, uniqc):
    biq, biqc = bigram_q(data)
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
    q = unkify_trigram(cw, biq, uniq, biqc, uniqc)
    log2q = { w : math.log2(v) for w, v in q.items() }
    def log2p(line):
        ret = 0.0
        for i in range(len(line) - 2):
            x_i = line[i]
            x_i_plus_1 = line[i+1]
            x_i_plus_2 = line[i+2]
            ret += get_value_tri(x_i_plus_2, x_i_plus_1, x_i, log2q, biq, uniq, biqc, uniqc)
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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("help: python3 q2.py [mode]\n\tmode: dev, test")
        sys.exit(-1)
    train_data = load_data("data/brown.train.txt")
    test_data = load_data("data/brown." + sys.argv[1] + ".txt")
    uni_q, known_vocab, uniq_log2p, cw = unigram(train_data)
    vocab = get_full_vocab(test_data)
    all_vocab = known_vocab.union(test_data)
    print("gamma: %s" % GAMMA)
    print("vocab size of test set: %s" % len(vocab))
    print("known vocab size in train set: %s" % len(known_vocab))
    print("corpus size of train set: %s" % train_data["total_words"])
    transform_corpus(test_data, known_vocab)
    print("trigram: %f" % perplexity(test_data, trigram(train_data, uni_q, cw)))
