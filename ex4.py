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
lstm_out = 550
first_filters = 10
second_filters = 40
batch_size = 100
num_of_epochs = 100
learning_rate = 0.1

def train(model, device):
    ctc_loss = nn.CTCLoss()
    model.train()
    optimizer = opt.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    global batch_size

    for e in range(num_of_epochs):
        loss_sum = 0
        for item, label, real_size, indices in train_loader:
            optimizer.zero_grad()
            item = item.to(device) 
            probes = model(item, device)

            length = probes.size()[0]
            targets = indices.to(device)

            input_lengths = torch.full(size=(batch_size,), fill_value=length, dtype=torch.long).to(device)
            target_lengths = real_size.to(device)

            loss = ctc_loss(probes, targets, input_lengths, target_lengths)
            loss_sum += loss.item()

            loss.backward()

            optimizer.step()

        
        print("finish epoch #{} avg loss {} last loss {}".format(e, (loss_sum / len(train_loader)), loss))
    
      
def decode(alignments):
    words = []
    alignments = alignments.cpu().numpy()
    for row in alignments:
        # remove consecutive repetition
        new_row = [x[0] for x in groupby(row)]
        # remove blanks
        new_row = list(filter(lambda c: c != char_to_idx['-'], new_row))
        # return to actual letters
        new_row = ''.join([idx_to_char[c] for c in new_row])
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
            
            _, alignments = torch.max(probes, dim=2)
            alignments = alignments.permute(1, 0)
            batch_size = alignments.size()[0]

            words = decode(alignments)
            err = calculate_cer(words, label)
            err_sum += err

            print("avg cer for batch #{}: {}".format(batch, (err / batch_size)))

            batch += 1

    avg_cer = err_sum / len(devset)
    print("avg total cer : {}".format(avg_cer))

def test(model, device):
    model.eval()
    output = []
    
    with torch.no_grad():
        for item, path in test_loader:
            item = item.to(device)
            probes = model(item, device)

            _, alignments = torch.max(probes, dim=2)
            alignments = alignments.permute(1, 0)
            batch_size = alignments.size()[0]

            words = decode(alignments)

            for i, word in enumerate(words):
                out = "{}, {}".format(path[i], word)
                output.append(out)

    with open('test_y', 'w+') as f:
        f.write("\n".join(output))

    print("finish with test")

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if len(sys.argv) > 1 and sys.argv[1] == 't':
        model = torch.load("./model.f")

        test(model, device)

        exit()
    else:
        model = ProbNet(lstm_out, len(char_to_idx), first_filters,
                        second_filters, batch_size)
                    
    print(device)
    model.to(device)

    train(model, device)
    torch.save(model, './model.f')
    
    evaluate(model, device)
    test(model, device)

if __name__ == "__main__":
    main()