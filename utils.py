import torch 
from gcommand_loader import GCommandLoader, char_to_idx, idx_to_char

trainset = GCommandLoader('./data/train/') #/Users/guest/Desktop/projects/SR/ex4/ex4
train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=100, shuffle=True,
        num_workers=20, pin_memory=True, sampler=None)

devset = GCommandLoader('./data/valid/') #/Users/guest/Desktop/projects/SR/ex4/ex4
dev_loader = torch.utils.data.DataLoader(
        devset, batch_size=100, shuffle=None,
        num_workers=20, pin_memory=True, sampler=None)


testset = GCommandLoader('./data/test/', is_test=True)
test_loader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=None,
        num_workers=20, pin_memory=True, sampler=None)

idx_to_class = trainset.idx_to_class

