# Import necessary packages.
import torch
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

# Globally define losses and accuracy
# Data to plot accuracy and loss graphs
train_losses = []
test_losses = []
train_acc = []
test_acc = []


def get_data_loader(batch_size):
    kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}

    # Train data transformations
    train_transforms = transforms.Compose([
        transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1), # Crop center of image with patch size of 22
        transforms.Resize((28, 28)), # Resize the image
        transforms.RandomRotation((-15., 15.), fill=0), # Rotate teh image from -15 to 15 degrees randomly.
        transforms.ToTensor(), # Convert image to tensor
        transforms.Normalize((0.1307,), (0.3081,)), # Apply mean 0.1307 and std of 0.3081 to data
        ])
    
    # Test data transformations
    test_transforms = transforms.Compose([
        transforms.ToTensor(), # Convert image to tensor
        transforms.Normalize((0.1307,), (0.3081,)) # Apply mean 0.1307 and std of 0.3081 to data which are same from train data.
        ])
    
    # Download the train data and apply train transforms
    train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
    
    # Download the test data and apply test transforms
    test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)

    # Apply all the keyword argumenets like batching the dataset, shuffling the data etc.
    train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, **kwargs)

    return train_loader, test_loader


def GetCorrectPredCount(pPrediction, pLabels):
    """Function to get the correct prediction count.

    Args:
        pPrediction (Object): Predicted tensors
        pLabels (Object): Actual labels of the images.

    Returns:
        Object(Tensor): If the predicted lables are actual labels then those items will be counted and returned.
    """
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()


def train(model, device, train_loader, optimizer, criterion):
    """Function to train the model.

    Args:
        model (Object): Torch.nn module network
        device (Object): Device name(CPU/Cuda)
        train_loader (Object): Torch train loader object
        optimizer (Object): Torch optimizer function
        criterion (Object): Torch Loss function
    """
    model.train() # Set the model to train mode.
    pbar = tqdm(train_loader) # Set one bar for visualization

    train_loss = 0 # Initialize train_loss to 0.
    correct = 0 # Initialize Correctly predicted labels to 0.
    processed = 0 # Initialize processed images to 0.

    # Enumerate through each data.
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device) # Load the data and change the device to cpu/cuda.
        optimizer.zero_grad() # Set all gradients to zero initially

        # Predict
        pred = model(data)

        # Calculate loss
        loss = criterion(pred, target)
        train_loss+=loss.item()

        # Backpropagation
        loss.backward() # Calculate gradient.
        optimizer.step() # Update weights.

        correct += GetCorrectPredCount(pred, target) # Get the correct prediction count.
        processed += len(data)

        pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    # Append the accuracy and losses to lists
    train_acc.append(100*correct/processed)
    train_losses.append(train_loss/len(train_loader))


def test(model, device, test_loader, criterion):
    """Function to train the model.

    Args:
        model (Object): Torch.nn module network
        device (Object): Device name(CPU/Cuda)
        train_loader (Object): Torch train loader object
        optimizer (Object): Torch optimizer function
        criterion (Object): Torch Loss function
    """
    model.eval() # Set the model to test mode.

    test_loss = 0
    correct = 0

    # As in test mode, no gradients are required to calculate.
    with torch.no_grad():
        # Enumerate through each data.
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device) # Load the data and change the device to cpu/cuda.

            output = model(data) # Pass the data to the model.
            test_loss += criterion(output, target).item() # Get the loss

            correct += GetCorrectPredCount(output, target) # Get the correct prediction count.

    # Append the accuracy and losses to lists
    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

  
def plot_metrics(train_losses, train_acc, test_losses, test_acc):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")


def visualize_data(train_loader):
    batch_data, batch_label = next(iter(train_loader)) 
    fig = plt.figure()
    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0), cmap='gray')
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])