import json
import random
import os
from argparse import ArgumentParser
import cPickle

from utils.data_utils import create_vocabulary, vectorize_sequences

random.seed(271)

TRAINSET_RATIO = 0.9
RESULT_FOLDER = 'save'


def flatten(in_data):
    turns = []
    for dialog in in_data:
        for turn in dialog['turns']:
            turns.append(turn['text'].lower())
    return turns


def split_traintest(in_data):
    train_datapoints = int(len(in_data) * TRAINSET_RATIO)
    random.shuffle(in_data)
    return in_data[:train_datapoints], in_data[train_datapoints:]


def write_dataset(in_dataset_name, in_train, in_test, in_vocab):
    if not(os.path.exists(RESULT_FOLDER)):
        os.makedirs(RESULT_FOLDER)
    with open(os.path.join(RESULT_FOLDER, 'realtrain_{}.txt'.format(in_dataset_name)), 'w') as train_out:
        for line in in_train:
            train_out.write('\n'.join(map(str, line)) + '\n')
    with open(os.path.join(RESULT_FOLDER, 'realtest_{}.txt'.format(in_dataset_name)), 'w') as test_out:
        for line in in_test:
            test_out.write('\n'.join(map(str, line)) + '\n')
    with open(os.path.join(RESULT_FOLDER, 'vocab_{}.pkl'.format(in_dataset_name)), 'w') as vocab_out:
        cPickle.dump(in_vocab, vocab_out)


def main(in_src_file, in_dataset_name, in_max_vocab_size, in_max_seq_len):
    with open(in_src_file) as data_in:
        data = json.load(data_in)
    data_flat = flatten(data)
    train, test = split_traintest(data_flat)
    vocab, rev_vocab = create_vocabulary(train, in_max_vocab_size)

    train_vectorized = vectorize_sequences(train, vocab, in_max_seq_len)
    test_vectorized = vectorize_sequences(test, vocab, in_max_seq_len)
    write_dataset(in_dataset_name, train_vectorized, test_vectorized, [rev_vocab, vocab])


def init_argument_parser():
    parser = ArgumentParser()
    parser.add_argument('source_file')
    parser.add_argument('--dataset_name', default='metalwoz')
    parser.add_argument('--max_vocab_size', default=10000)
    parser.add_argument('--max_seq_len', default=20)
    return parser

if __name__ == '__main__':
    parser = init_argument_parser()
    args = parser.parse_args()
    main(args.source_file,
         args.dataset_name,
         args.max_vocab_size,
         args.max_seq_len)
