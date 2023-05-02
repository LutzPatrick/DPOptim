import torchvision
import torchvision.transforms as T
import torch
import copy
from opacus import GradSampleModule

transform = T.Compose([T.ToTensor(), T.Lambda(lambda x: torch.flatten(x))])
dataset = torchvision.datasets.MNIST('./datasets/', download=True, train=True, transform=transform)
x, y = dataset[0]

model = GradSampleModule(torch.nn.Linear(784, 256))
#optimizer = torch.optim.SGD(model.parameters(), lr=.1)
#data_loader = torch.utils.data.DataLoader(dataset, batch_size=64)

pred = model(x)
model_copy = copy.deepcopy(model)
pred = model(x)