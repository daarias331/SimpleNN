import torch
from torch import nn
import torch.nn.functional as F # importing the functional API, that handles activation funcions, loss functions, etc

from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from tqdm import tqdm



#HYPERPARAMETERS
input_size = 784
num_classes = 10
learning_rate = 0.001
num_epochs = 3
batch_size = 64


#Defining the model structure
class MyNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        #Define NN structure
        self.fc1= nn.Linear(input_size,50)
        self.fc2= nn.Linear(50, 60)
        self.fc3= nn.Linear(60, num_classes)

    def forward(self, x):
        #Define the forward pass
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)

        return x

# Set device cuda for GPU if it's available otherwise run on the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Load data
full_dataset=datasets.MNIST(root='data/', train=True,download=True)

#Split the dataset into training and validation
train_ds, val_ds = torch.utils.data.random_split(full_dataset, [50000, 10000])

#Download test dataset
test_ds=datasets.MNIST(root='data/', train=False, download=True)

#Create data loaders
train_loader=DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, transforms=transforms.ToTensor())
val_loader=DataLoader(dataset=val_ds,batch_size=batch_size, shuffle=True, transforms=transforms.ToTensor()) 
test_dataloader=DataLoader(dataet=test_ds, batch_size=batch_size, shuffle=False, transformers=transforms.ToTensor())


# Initialize network and send it to device
model = MyNN(input_size=input_size, num_classes=num_classes).to(device)

#Loss and optimiser
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Training loop

def train_loop(num_epochs,  model, train_loader, criterion, optimizer):
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)

            # Get to correct shape
            data = data.reshape(data.shape[0], -1)

            # forward
            scores = model(data)
            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval() #Sets the model to evaluation mode

    # We don't need to keep track of gradients here so we wrap it in torch.no_grad()
    with torch.no_grad():
        # Loop through the data
        for x, y in loader:

            # Move data to device
            x = x.to(device=device)
            y = y.to(device=device)

            # Get to correct shape
            x = x.reshape(x.shape[0], -1)

            # Forward pass
            scores = model(x)
            _, predictions = scores.max(1)

            # Check how many we got correct
            num_correct += (predictions == y).sum()

            # Keep track of number of samples
            num_samples += predictions.size(0)

    model.train() #Sets the model back to training  mode
    return num_correct / num_samples


# Check accuracy on training & test to see how good our model
model.to(device)
print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
print(f"Accuracy on validation set: {check_accuracy(val_loader, model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")