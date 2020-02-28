import json 
import matplotlib.pyplot as plt
import seaborn as sns
import torch 
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim 
import sys
from data_gen import Data
from nn_gen import NeuralNet

param_file = sys.argv[1]

with open(param_file) as json_file:
    param = json.load(json_file)

data_1 = param['data_file']
# Construct model and dataset
model= NeuralNet()
data= Data(data_1, 3000, 29492)

# Define an optimizer and the loss function
optimizer = optim.SGD(model.parameters(), lr=param['learning_rate'])
loss= torch.nn.BCELoss()

obj_vals= []
cross_vals= []
num_epochs= int(param['num_epochs'])

model = model.double()
# Training loop
for epoch in range(1, num_epochs + 1):

    train_val= model.backprop(data, loss, epoch, optimizer)
    obj_vals.append(train_val)

    test_val= model.test(data, loss, epoch)
    cross_vals.append(test_val)

    if not ((epoch + 1) % param['display_epochs']):
        print('Epoch [{}/{}]'.format(epoch+1, num_epochs)+\
            '\tTraining Loss: {:.4f}'.format(train_val)+\
            '\tTest Loss: {:.4f}'.format(test_val))

print('Final training loss: {:.4f}'.format(obj_vals[-1]))
print('Final test loss: {:.4f}'.format(cross_vals[-1]))

plt.plot(range(num_epochs), obj_vals, label= "Training loss", color="red")
plt.plot(range(num_epochs), cross_vals, label= "Test loss", color= "green")
plt.legend()
plt.savefig('results/results.png')