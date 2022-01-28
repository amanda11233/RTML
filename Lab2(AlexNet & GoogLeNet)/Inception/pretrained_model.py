 

from data import train_dataloader, valid_dataloader, test_dataloader
from model import   train_model, plot_data, GoogLeNet
from test_ import test_model
import torch.optim as optim
import torch.nn as nn
import torch 
import torchvision.models as models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
     alpha = 0.001
     momentum = 0.9
     weights_name = 'googlenet_lr_' + str(alpha) + '_lrn_bestof_pretrained'
 
   
     googlenet = models.googlenet(pretrained=True, aux_logits = True) 
     googlenet.fc  =  nn.Linear(1024, 10) 
     googlenet.aux1.fc =  nn.Linear(1024, 10) 
     googlenet.aux2.fc =  nn.Linear(1024, 10) 
     googlenet = googlenet.to(device)

     criterion_3 = nn.CrossEntropyLoss()
     params_to_update_3 = googlenet.parameters()
     optimizer_3 = optim.Adam(params_to_update_3, lr=alpha)
     
     # dataloaders = { 'train': train_dataloader, 'val': valid_dataloader }

     # best_model3, val_acc_history, loss_acc_history = train_model(googlenet, dataloaders, criterion_3, optimizer_3, 10, weights_name, is_inception=True)

     # val_acc_history = [x.cpu().detach().numpy() for x in val_acc_history]

     # plot_data(val_acc_history, loss_acc_history)

     dataloaders = {"test" : test_dataloader}
     print("Testing GoogleNet Pretrained Model...")
     test_acc, test_loss = test_model(googlenet, dataloaders, criterion_3)

    #  best_model2, val_acc_history2, loss_acc_history2 = train_model(alexnetmodule, dataloaders, criterion, optimizer, 10, weights_name)

  

