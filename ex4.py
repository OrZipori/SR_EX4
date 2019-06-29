from cer import cer
import torch
from models.network import ProbNet
from utils import * 
import numpy as np
import torch.nn as nn
import torch.optim as opt
from itertools import groupby, chain
import sys
# Hyper parameters
lstm_out = 150
first_filters = 10
second_filters = 40
batch_size = 1000
num_of_epochs = 40
learning_rate = 0.001

def train(model, device):
    ctc_loss = nn.CTCLoss()
    model.train()
    optimizer = opt.RMSprop(model.parameters(), lr=learning_rate)
    global batch_size
    
    print("123456")
    for e in range(num_of_epochs):
        count = 0
        loss_sum = 0
        for item, label, real_size, indices in train_loader:
            count += 1
            print('[%d%%]'%count, end="\r")
            optimizer.zero_grad()
            item = item.to(device) 
            probes = model(item, device)

            length = probes.size()[0]
            targets = indices
            targets = map(lambda item: item.cpu().numpy().tolist(), targets)
            targets = list(chain(*targets))
            targets = torch.tensor(targets).to(device)
            probes = probes.to(device)

            input_lengths = torch.full(size=(batch_size,), fill_value=length, dtype=torch.long).to(device)
            target_lengths = real_size.to(device)

            loss = ctc_loss(probes, targets, input_lengths, target_lengths)
            loss_sum += loss

            loss.backward()

            optimizer.step()
            
            break

        
        print("finish epoch #{} avg loss {} last loss {}".format(e, (loss_sum / len(train_loader)), loss))
    
    alignments = probes.data.max(2, keepdim=True)[1]
    print(alignments.size())
    alignments = alignments.view(-1, length)
    print(alignments.size())
    batch_size1 = alignments.size()[0]
    print(batch_size1)
    words = decode(alignments)
    err = calculate_cer(words, label)
    exit(0)
    
      
def decode(alignments):
    words = []
    alignments = alignments.cpu().numpy()
    for row in alignments:
        print(row)
        # remove consecutive repetition
        new_row = [x[0] for x in groupby(row)]
        print("new row {}".format(new_row))
        # remove blanks
        new_row = list(filter(lambda c: c != char_to_idx['-'], new_row))
        print("new row {}".format(new_row))
        # return to actual letters
        new_row = ''.join([idx_to_char[c] for c in new_row])
        print("new row {}".format(new_row))
        words.append(new_row)

    return words

def calculate_cer(words, label):
    label = label.cpu().numpy()
    label = [idx_to_class[l] for l in label]

    err = 0
    st = ""
    for pred, real in zip(words, label):
        err += cer(pred, real)
        st += pred + " vs " + real + "\n"

    with open('check.txt', 'a+') as f:
        f.write(st)

    return err 

def evaluate(model, device):
    model.eval()

    err_sum = 0
    batch = 1
    with torch.no_grad():
        for item, label, _, _ in dev_loader:
            item = item.to(device)
            probes = model(item, device)
            length = probes.size()[0]
            print("**")
            print(probes)
            print(probes.permute(1, 0,2))
            print(probes.size())
            

            alignments = probes.data.max(2, keepdim=True)[1]
            print(alignments.size())
            alignments = alignments.view(-1, length)
            print(alignments.size())
            batch_size = alignments.size()[0]
            print(batch_size)

            words = decode(alignments)
            err = calculate_cer(words, label)

            print("avg cer for batch #{}: {}".format(batch, (err / batch_size)))

            batch += 1

def test():
    pass

def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'e':
        model = torch.load("./model.f")
    else:
        model = ProbNet(lstm_out, len(char_to_idx), first_filters,
                        second_filters, batch_size)
                    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    train(model, device)
    torch.save(model, './model.f')
    
    evaluate(model, device)

if __name__ == "__main__":
    main()