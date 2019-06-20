from collections import defaultdict

import nltk

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
PAD = 'PAD'
BOS = 'BOS'
EOS = 'EOS'
UNK = 'UNK'


def create_vocabulary(in_lines, in_max_size):
    freqdict = defaultdict(lambda: 0)
    freqdict[PAD] = 999999
    freqdict[BOS] = 999998
    freqdict[EOS] = 999997
    freqdict[UNK] = 999996

    for utterance in in_lines:
        for token in utterance.split():
            freqdict[token] += 1
    freqdict_sorted = sorted(freqdict.items(), key=lambda x: x[1], reverse=True)[:in_max_size]
    rev_vocab = [item[0] for item in freqdict_sorted]
    vocab = {item: idx for item, idx in enumerate(rev_vocab)}
    return vocab, rev_vocab


def vectorize_sequences(in_lines, in_vocab, in_max_seq_len):
    result = []
    for line in in_lines:
        tokens = line.split()
        seq = [in_vocab.get(token, UNK_ID) for token in tokens] + [EOS_ID]
        result.append(seq[:in_max_seq_len] + [PAD_ID for _ in range(max(in_max_seq_len - len(seq), 0))])
    return result
