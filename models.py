from torch import nn
from opacus import GradSampleModule

def get_model(data_name, device):
    if data_name == 'MNIST':
        model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=8, stride=2, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=32),
            nn.Tanh(),
            nn.Linear(in_features=32, out_features=10),
            nn.Softmax(dim=1)
        )
    else:
        raise ValueError()
    model.to(device)
    return model

# def reset_parameters(model):
#     if isinstance(model, GradSampleModule):
#         model = model._module
#     for layer in model:
#         if hasattr(layer, 'reset_parameters'):
#             layer.reset_parameters()