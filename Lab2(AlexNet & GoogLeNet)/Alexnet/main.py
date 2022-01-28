
from data import train_dataloader, valid_dataloader, test_dataloader
from model import AlexNetModule, AlexNetSequential, train_model, plot_data
import torch.optim as optim
import torch.nn as nn
import torch 
from test_ import test_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
     alpha = 0.001
     momentum = 0.9
     weights_name = 'alex_sequential_lr_' + str(alpha) + '_lrn_bestofar'

     # alexnetmodule = AlexNetModule(10)
     alexnetmodule = AlexNetSequential()
     alexnetmodule = alexnetmodule.getSequential().to(device)
     criterion = nn.CrossEntropyLoss() 
     params_to_update = alexnetmodule.parameters() 
     optimizer = optim.SGD(params_to_update, lr=alpha, momentum=momentum)

     # dataloaders = { 'train': train_dataloader, 'val': valid_dataloader } 
     # num_epochs = 10

     # best_model2, val_acc_history2, loss_acc_history2 = train_model(alexnetmodule, dataloaders, criterion, optimizer, num_epochs, weights_name)

     # val_acc_history2 = [x.cpu().detach().numpy() for x in val_acc_history2]
     # plot_data(val_acc_history2, loss_acc_history2)

     PATH = "alex_sequential_lr_0.001_lrn_bestofar.pth"
     alexnetmodule.load_state_dict(torch.load(PATH))
     dataloaders = {"test" : test_dataloader}

     alexnetmodule = alexnetmodule.to(device)
     print("Testing the Alexnet Model...")
     test_acc, test_loss = test_model(alexnetmodule, dataloaders, criterion)

      
      

