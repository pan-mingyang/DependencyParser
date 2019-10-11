import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ACTS = {'SHIFT': 0, 'LEFT_ARC': 1, 'RIGHT_ARC': 2}


class Parser(nn.Module):
    def __init__(self, word_dim, tag_dim, word_num, tag_num, hidden, action_num, vocab, vectors):
        super(Parser, self).__init__()
        self.word_embedding = nn.Embedding(word_num, word_dim).to(device)
        self.tag_embedding = nn.Embedding(tag_num, tag_dim).to(device)
        self.input = nn.Linear(word_dim * 3 + tag_dim * 3, hidden)
        self.output = nn.Linear(hidden, action_num)
        self.vocab = vocab
        self.action_num = action_num
        self.word_dim = word_dim
        self.tag_dim = tag_dim
        if vectors is not None:
            self.word_embedding.weight = nn.Parameter(torch.tensor(vectors))

    def get_action(self, x):
        x = self.input(x)
        x = torch.relu(x)
        x = self.output(x)
        x = torch.softmax(x, 0)
        return x.argmax()

    def get_actions(self, sent, tag):
        acts = []
        stack = [torch.zeros(self.word_dim + self.tag_dim)]
        buffer = self.word_embedding(torch.tensor(sent).to(device))
        tags = self.tag_embedding(torch.tensor(tag).cuda())
        buffer = torch.cat([buffer, tags], 1).to(device)
        buffer = list(torch.cat([buffer, torch.zeros((1, self.word_dim + self.tag_dim)).to(device)], 0))
        while len(buffer) >= 1:
            if len(stack) == 2 and len(buffer) == 1:
                break
            if len(stack) < 3 and len(buffer) > 1:
                stack.append(buffer[0])
                acts.append(torch.tensor(ACTS['SHIFT']).to(device))
                del buffer[0]
                continue
            x = torch.cat([stack[-1], stack[-2], buffer[0]], 0)
            act = self.get_action(x)
            acts.append(act)
            if act == ACTS['SHIFT']:
                stack.append(buffer[0])
                del buffer[0]
            elif act == ACTS['LEFT_ARC']:
                del stack[-2]
            elif act == ACTS['RIGHT_ARC']:
                del stack[-1]
        return torch.tensor(acts)

    def forward(self, sentence, tag, actions):
        loss = []
        stack = [torch.zeros(self.word_dim + self.tag_dim)]
        buffer = self.word_embedding(torch.tensor(sentence).to(device))
        tags = self.tag_embedding(torch.tensor(tag).cuda())
        buffer = torch.cat([buffer, tags], 1).to(device)
        buffer = list(torch.cat([buffer, torch.zeros((1, self.word_dim + self.tag_dim)).to(device)], 0))
        for i in range(len(actions)):
            act = actions[i]
            if len(stack) > 2 and len(buffer) >= 1:
                x = torch.cat([stack[-1], stack[-2], buffer[0]], 0).to(device)
                x = self.input(x)
                x = torch.relu(x)
                x = self.output(x)
                x = torch.log_softmax(x, 0)
                loss.append(-x[act].unsqueeze(0))
            if act == ACTS['SHIFT']:
                stack.append(buffer[0])
                del buffer[0]
            elif act == ACTS['LEFT_ARC']:
                del stack[-2]
            elif act == ACTS['RIGHT_ARC']:
                del stack[-1]

        return torch.sum(torch.cat(loss, 0)) if len(loss) > 0 else None
