
import time
import torch
import torchvision
from torchvision import datasets, models, transforms 
import time 
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_model(model, dataloaders, criterion):
   
    model.eval()    

    running_loss = 0.0
    running_corrects = 0
            
    for inputs, labels in dataloaders['test']:
 
        
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)

        # Gather our summary statistics        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    test_loss = running_loss / len(dataloaders['test'].dataset)
    test_acc = running_corrects.double() / len(dataloaders['test'].dataset)
    test_end = time.time()

    print('{} Loss: {:.4f} Acc: {:.4f}'.format('Test', test_loss, test_acc))
    
    return test_acc, test_loss