# Import necessary packages.
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """Class having the model details
    """
    def __init__(self):
        super(Net, self).__init__() # Inherite all modules and attributes from nn.Module function.
        # Define all conv layers with kernel size 3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3) # Input channels 1, output filters 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3) # Input channels 32, output filters 64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3) # Input channels 64, output filters 128
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3) # Input channels 128, output filters 256
        self.fc1 = nn.Linear(4096, 50) # Fully connected layer with input channels 4096, output channels 50 
        self.fc2 = nn.Linear(50, 10) # Fully connected layer with input channels 50, output channels 10 

    def forward(self, x):
        # Example image size is 1 * 28 * 28, Receptive field 1.
        x = F.relu(self.conv1(x), 2) # RF 3, Output(O/p) 32 * 26 * 26
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) # RF 6, Output(O/p) 64 * 12 * 12(Conv2+Maxpool)
        x = F.relu(self.conv3(x), 2) # RF 10, Output(O/p) 128 * 10 * 10
        x = F.relu(F.max_pool2d(self.conv4(x), 2)) # RF 16, Output(O/p) 256 * 4 * 4(Conv4+Maxpool)
        x = x.view(-1, 4096) # View the input as (Compatible) * 256 * 4 * 4(4096) 
        x = F.relu(self.fc1(x)) # Apply fully connected layer with relu
        x = self.fc2(x)
        return F.log_softmax(x, dim=1) # Apply softmax function


def get_train_parameters():
    """Function to get the training cretarians.

    Returns:
        Objects: All training parameters.
    """
    use_cuda = torch.cuda.is_available() # Boolean to get whether cuda is there in device or not.
    device = torch.device("cuda" if use_cuda else "cpu") # If cuda is present set the device to cuda, otherwise set to cpu.
    model = Net().to(device) # Set the model to same device.
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9) # Initialize the optimizer to SGD with learning rate 0.01 and momentum 0.9.
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1, verbose=True) # Initialize the scheduler learning rate.
    # New Line
    criterion = nn.CrossEntropyLoss() # Define the entropy loss function
    
    return model, device, optimizer, scheduler, criterion