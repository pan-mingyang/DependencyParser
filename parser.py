import random
from typing import Dict

from model import *
from torch import optim


SHIFT = 0
LEFT_ARC = 1
RIGHT_ARC = 2
END = -1
operations = ['SHIFT', 'LEFT_ARC', 'RIGHT_ARC', 'END']


def get_actions(infname: str, outfname: str):
    PTB = []
    with open(infname, 'r') as f:
        sentence = []
        r = dict()
        for line in f.readlines():
            line = line.strip()
            if not line:
                PTB.append((sentence, r))
                sentence = []
                r = dict()
            else:
                line = line.split()
                sentence.append([int(line[0]), line[1], line[3], line[5], int(line[6]), line[7]])
                r[(int(line[6]), int(line[0]))] = line[7]
    f = open(outfname, 'w', encoding='utf-8')
    for sentence, rp in PTB:
        stack = [0]
        actions = []
        rc = set()
        length = len(sentence)
        word = []
        for i in range(length + 1):
            if i < length:
                word = sentence[i]
            while len(stack) > 1:

                if (stack[-1], stack[-2]) in rp:
                    actions.append(LEFT_ARC)
                    rc.add((stack[-1], stack[-2]))
                    del stack[-2]
                elif (stack[-2], stack[-1]) in rp:
                    flag = True
                    for a, b in rp:
                        if a == stack[-1] and (a, b) not in rc:
                            flag = False
                            break
                    if flag:
                        actions.append(RIGHT_ARC)
                        rc.add((stack[-2], stack[-1]))
                        del stack[-1]
                    else:
                        break
                else:
                    break
            if i < length:
                actions.append(SHIFT)
                stack.append(word[0])
            continue
        f.write(' '.join((list(map(lambda x: str(x[1]), sentence)))))
        f.write('\t')
        f.write(' '.join((list(map(lambda x: str(x[2]), sentence)))))
        f.write('\t')
        f.write(' '.join(map(lambda x: operations[x], actions)))
        f.write('\n')


def test():
    with open('data/sentences_with_actions_train.txt') as f:
        for line in f.readlines():
            sentence, action = line.split('\t')
            sentence = sentence.split()
            action = action.split()
            print(sentence)
            print(action)
            input()
            stack = [('<root>', 0)]
            c = 1
            for i in range(len(action)):
                if action[i] == 'SHIFT':
                    stack.append((sentence[c-1], c))
                    c += 1
                elif action[i] == 'LEFT_ARC':
                    print('%d: %s -> %d: %s' % (stack[-1][1], stack[-1][0],
                                                stack[-2][1], stack[-2][0]))
                    del stack[-2]
                elif action[i] == 'RIGHT_ARC':
                    print('%d: %s -> %d: %s' % (stack[-2][1], stack[-2][0],
                                                stack[-1][1], stack[-1][0]))
                    del stack[-1]
            input()


class Vocab(object):

    def __init__(self):
        self.w2i = dict()
        self.i2w = dict()

    @staticmethod
    def get_vocab(words):
        vocab = Vocab()
        for i, w in enumerate(words):
            vocab.i2w[i] = w
            vocab.w2i[w] = i
        return vocab

    @staticmethod
    def get_vocab_from_dict(dic: Dict):
        vocab = Vocab()
        vocab.w2i = dict(dic)
        vocab.i2w = {v: k for k, v in dic.items()}
        return vocab

    def get_word(self, i):
        return self.i2w[i]

    def get_index(self, w):
        return self.w2i[w]


def get_embedding_and_tags():
    with open('data/sskip.100.vectors') as f:
        words = dict()
        vector = []
        i = 0
        line = f.readline().split()
        word_num = int(line[0])
        for line in f.readlines():
            line = line.split()
            words[line[0]] = i
            i += 1
            vector.append(list(map(lambda x: float(x), line[1:])))

    words2 = set()
    tags = set()
    with open('data/sentences_with_actions_train.txt') as f:
        for line in f.readlines():
            sentence, tag, _ = line.split('\t')
            sentence = sentence.split()
            tag = tag.split()
            words2 = words2.union(sentence)
            tags = tags.union(tag)
    oov = words2 - words.keys()
    for w in oov:
        words[w] = i
        i += 1
        vector.append(list(torch.rand(100) * 2 - 1))
        print(i)

    tag_vocab = Vocab.get_vocab(tags)
    return words, vector, tag_vocab


def get_data(filename, word_vocab, tag_vocab):
    data = []
    with open(filename) as f:
        for line in f.readlines():
            sentence, tag, action = line.split('\t')
            sentence = sentence.split()
            tag = tag.split()
            action = action.split()[:-1]
            sentence = list(map(lambda w: word_vocab.w2i[w], sentence))
            tag = list(map(lambda t: tag_vocab.w2i[t], tag))
            action = list(map(lambda a: ACTS[a], action))
            data.append((sentence, tag, action))
        return data


def main():
    words, vectors, tag_vocab = get_embedding_and_tags()
    vocab = Vocab.get_vocab_from_dict(words)
    train = get_data('data/sentences_with_actions_train.txt', vocab, tag_vocab)
    model = Parser(100, 16, len(words), len(tag_vocab.w2i), 128, 3, vocab, vectors).to(device)
    '''
    get_actions('PTB/PTB_train_auto.conll', 'data/sentences_with_actions_train.txt')
    get_actions('PTB/PTB_test_auto.conll', 'data/sentences_with_actions_test.txt')
    get_actions('PTB/PTB_development_auto.conll', 'data/sentences_with_actions_dev.txt')
    '''
    epoch = 50
    
    optimizer = optim.SGD(model.parameters(), lr=.015, momentum=.9)
    los = 0
    c = 0
    model.train()
    torch.save(model, 'a-1.pkl')
    for i in range(epoch):
        random.shuffle(train)
        for x in train:
            c += 1
            model.zero_grad()
            loss = model.forward(*x)
            if loss is not None:
                los += loss.item()
                # print(loss)
                loss.backward()
                optimizer.step()
            if c % 100 == 0:
                print('Epoch {} data {} Loss: {}'.format(i, c, los))
                los = 0

        torch.save(model, 'a{}.pkl'.format(i))


if __name__ == '__main__':
    main()
