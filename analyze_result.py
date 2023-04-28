import pickle
import yaml
from utils import accuracy

with open('./config.yaml', 'r') as file:
    args = yaml.safe_load(file)
with open(f'./runs/{args["optim_name"]}_{args["data_name"]}.pkl', 'rb') as file:
    metrics = pickle.load(file)

print('best loss', metrics.get_best_val('CrossEntropyLoss', 'CrossEntropyLoss'))
print('best acc', metrics.get_best_val('accuracy', 'CrossEntropyLoss'))
print('summary', metrics.get_summary(accuracy))

