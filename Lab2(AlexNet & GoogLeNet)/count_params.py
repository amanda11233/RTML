import numpy as np
from Inception.model import GoogLeNet
from Alexnet.model import AlexNetModule
 

def count_params(model):
    return np.sum(p.numel() for p in model.parameters() if p.requires_grad)


alexnet = AlexNetModule()
googlenet = GoogLeNet()

models = [alexnet, googlenet]

for model in models:
    print(f'{type(model).__name__} has {count_params(model):,} trainable parameters')