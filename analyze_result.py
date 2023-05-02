import pickle
import matplotlib.pyplot as plt
from utils import accuracy

data_name = 'MNIST'

metrics = dict()
for optim in ['SGD', 'ONC', 'DPSGD']:
    with open(f'./runs/{optim}_{data_name}.pkl', 'rb') as file:
        data = pickle.load(file)
    metrics[optim] = data.get_summary(accuracy)

for optim in ['SGD', 'DPSGD', 'ONC']:
    metric_train, _, metric_test = metrics[optim]
    print(metric_test)
    plt.plot(range(len(metric_test)), metric_test, label=optim)
    #plt.plot(range(len(metric_test)), metric_test, label=optim)

plt.legend()
plt.show()

