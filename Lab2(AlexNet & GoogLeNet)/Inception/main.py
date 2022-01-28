
from data import train_dataloader, valid_dataloader, test_dataloader
from model import   train_model, plot_data, GoogLeNet
import torch.optim as optim
import torch.nn as nn
import torch 
from test_ import test_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
if __name__ == '__main__':
     alpha = 0.001
     momentum = 0.9
     weights_name = 'googlenet_lr_' + str(alpha) + '_lrn_bestofar'
 
  
     googlenet = GoogLeNet().to(device)
     criterion_3 = nn.CrossEntropyLoss()
     params_to_update_3 = googlenet.parameters()
     optimizer_3 = optim.Adam(params_to_update_3, lr=alpha)
     
     # dataloaders = { 'train': train_dataloader, 'val': valid_dataloader }

     # best_model3, val_acc_history, loss_acc_history = train_model(googlenet, dataloaders, criterion_3, optimizer_3, 25, weights_name, is_inception=True)

     # val_acc_history = [x.cpu().detach().numpy() for x in val_acc_history]
     
     # plot_data(val_acc_history, loss_acc_history)
     PATH = "googlenet_lr_0.001_lrn_bestofar.pth"
     googlenet.load_state_dict(torch.load(PATH))
     dataloaders = {"test" : test_dataloader}
     print("Testing GoogleNet Model...")
     test_acc, test_loss = test_model(googlenet, dataloaders, criterion_3)




 
  

