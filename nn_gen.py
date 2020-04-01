import torch 
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import data_gen

class NeuralNet(nn.Module):
    '''    
    NN Architecture:
        2D Convolution layer
        Two fully connected layers
        Relu and Sigmoid non-linear activation functions
    '''

    def __init__(self):
        super(NeuralNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1,1,2)
        self.maxpool = nn.MaxPool2d(2)
        self.batch = nn.BatchNorm2d(1)
        
        self.fc1= nn.Linear(36, 100)
        self.fc2= nn.Linear(100, 5)

    # forward function
    def forward(self, x):
        relu = func.relu(self.conv1(x))
        maxpool = self.maxpool(relu)
        conv1 = self.batch(maxpool)
        #Flattens the last two elements 
        h_flat = conv1.view(conv1.size(0), -1)
        #h_flat = conv1.reshape(conv1.shape[:2],-1)
        h = func.relu(self.fc1(h_flat))
        y = torch.sigmoid(self.fc2(h))
        
        return y

    #reset for training
    def reset(self):
        self.conv1.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    # backpropagation function
    def backprop(self, data, loss, epoch, optimizer):
        self.train()
        inputs= torch.from_numpy(data.x_train)
        #inputs = inputs.reshape(len(inputs), 1,5)
        targets= torch.from_numpy(data.y_train)
        targets = targets.reshape(len(targets),5)
        outputs= self(inputs)
        obj_val= loss(self.forward(inputs), targets)
        optimizer.zero_grad()
        obj_val.backward()
        optimizer.step()
        return obj_val.item()

    # test function, avoids calculation of gradients
    def test(self, data, loss, epoch):
        self.eval()
        with torch.no_grad():
            inputs= torch.from_numpy(data.x_test)
            #inputs - inputs.reshape(len(inputs),1,5)
            targets= torch.from_numpy(data.y_test)
            targets = targets.reshape(len(targets), 5)
            outputs= self(inputs)
            cross_val= loss(self.forward(inputs), targets)
        return cross_val.item()
        