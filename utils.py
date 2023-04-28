import torch

def get_optimizer(optim_name, model, hp):
    if optim_name in ['SGD', 'DPSGD']:
        return torch.optim.SGD(model.parameters(), lr=hp['lr'], momentum=hp['momentum'])
    raise ValueError()

def accuracy(x, y):
    return torch.mean((torch.argmax(x, dim=1) == y).double())

def hp_from_file(hp):
    hp['batch_size'] = 2**hp['batch_size_base']
    hp['lr'] = hp['lr_base'] * hp['batch_size']/float(512)
    hp['momentum'] = hp['momentum_base']/float(10)
    return hp