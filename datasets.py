import torch
from torchvision import datasets, transforms

def load_data(data_name):
    data = dict()

    if data_name == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        data['train'] = getattr(datasets, data_name)('./datasets/', download=True, train=True, transform=transform)
        #data['train'], data['val'] = torch.utils.data.random_split(data['train'], [.8, .2])
        data['test'] = getattr(datasets, data_name)('./datasets/', download=True, train=False, transform=transform)
        data['val'] = data['test']

    return data

class Metrics():

    def __init__(self, metric_fns, runs, epochs, device):

        self.metric_fns = metric_fns
        self.metrics_dict = dict()
        for i, metric in enumerate(metric_fns):
            metric_str = Metrics.metric_to_str(metric)
            self.metrics_dict[metric_str] = i

        self.metrics_train_ = torch.zeros([len(metric_fns), runs, epochs+1], device=device)
        self.metrics_val_ = torch.zeros([len(metric_fns), runs, epochs+1], device=device)
        self.metrics_test_ = torch.zeros([len(metric_fns), runs, epochs+1], device=device)

        self.run = 0

    def __getitem__(self, key):
        metric, mode, epoch = key
        metric_str = Metrics.metric_to_str(metric)
        if mode == 'train':
            return self.metrics_train_[self.metrics_dict[metric_str], self.run, epoch]
        if mode == 'val':
            return self.metrics_val_[self.metrics_dict[metric_str], self.run, epoch]
        if mode == 'test':
            return self.metrics_test_[self.metrics_dict[metric_str], self.run, epoch]
        raise ValueError()
        
    def __setitem__(self, key, value):
        metric, mode, epoch = key
        metric_str = Metrics.metric_to_str(metric)
        if mode == 'train':
            self.metrics_train_[self.metrics_dict[metric_str], self.run, epoch] = value
        elif mode == 'val':
            self.metrics_val_[self.metrics_dict[metric_str], self.run, epoch] = value
        elif mode == 'test':
            self.metrics_test_[self.metrics_dict[metric_str], self.run, epoch] = value
        else:
            raise ValueError()

    def increment_run(self):
        self.run += 1

    def get_summary(self, metric):
        metric_str = Metrics.metric_to_str(metric)
        summary_train = torch.mean(self.metrics_train_[self.metrics_dict[metric_str]], axis=0)
        summary_val = torch.mean(self.metrics_val_[self.metrics_dict[metric_str]], axis=0)
        summary_test = torch.mean(self.metrics_test_[self.metrics_dict[metric_str]], axis=0)
        return summary_train, summary_val, summary_test
    
    def get_best_val(self, metric, comp_metric, dir='low'):
        summary_train, summary_val, summary_test = self.get_summary(comp_metric)
        if dir=='high':
            best_idx = torch.argmax(summary_val)
        elif dir=='low':
            best_idx = torch.argmin(summary_val)
        else:
            raise ValueError()
        summary_train, summary_val, summary_test = self.get_summary(metric)
        return summary_train[best_idx], summary_val[best_idx], summary_test[best_idx]
    
    def to(self, device):
        self.metrics_train_ = self.metrics_train_.to(device=device)
        self.metrics_val_ = self.metrics_val_.to(device=device)
        self.metrics_test_ = self.metrics_test_.to(device=device)

    @staticmethod
    def metric_to_str(metric):
        if isinstance(metric, str):
            return metric
        if hasattr(metric, '__name__'):
            return metric.__name__
        return type(metric).__name__
