import random

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

    def get_word(self, i):
        return self.i2w[i]

    def get_index(self, w):
        return self.w2i[w]


if __name__ == '__main__':
    '''
    get_actions('PTB/PTB_train_auto.conll', 'data/sentences_with_actions_train.txt')
    get_actions('PTB/PTB_test_auto.conll', 'data/sentences_with_actions_test.txt')
    get_actions('PTB/PTB_development_auto.conll', 'data/sentences_with_actions_dev.txt')
    '''

    with open('data/sentences_with_actions_train.txt') as f:
        words = set()
        tags = set()
        for line in f.readlines():
            sentence, tag, _ = line.split('\t')
            sentence = sentence.split()
            tag = tag.split()
            for w in sentence:
                words.add(w)
            for t in tag:
                tags.add(t)
    vocab = Vocab.get_vocab(list(words))
    tags = Vocab.get_vocab(list(tags))

    model = Parser(64, 16, len(words), len(tags.w2i), 128, 3, vocab).to(device)
    train = []
    with open('data/sentences_with_actions_train.txt') as f:
        for line in f.readlines():
            sentence, tag, action = line.split('\t')
            sentence = sentence.split()
            action = action.split()
            tag = tag.split()
            sentence = list(map(lambda x: vocab.get_index(x), sentence))
            action = list(map(lambda x: ACTS[x], action))
            tag = list(map(lambda x: tags.get_index(x), tag))
            train.append((sentence, action, tag))

    epoch = 10

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



