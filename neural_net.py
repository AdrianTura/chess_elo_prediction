import torch
import torch.nn as nn
import datetime
from create_dataset import create_dataset,read_test_data
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

NO_LAYERS = 1
DEPTH = 6

# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, no_input_features, no_output_features):
        super(NeuralNetwork, self).__init__()

        print('Number of input features: {}, number of output features: {}'.format(no_input_features, no_output_features))
        self.sizes = [[no_input_features, 32], [32,64]]
        
        for i in range(NO_LAYERS):
            size = 64

            for _ in range(DEPTH):
                self.sizes.append([size, size*2])
                size *= 2
            
            for _ in range(DEPTH):
                self.sizes.append([size, size//2])
                size = size//2
                       
        self.sizes.append([64,32])
        self.sizes.append([32,16])
        self.sizes.append([16,8])

        self.layers = nn.ModuleList()

        for size in self.sizes:
            self.layers.append(nn.Linear(size[0], size[1]))

        self.output = nn.Linear(8, no_output_features)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))

        x = self.output(x)
        return x 

writer = SummaryWriter('runs/model_{}_layers_{}_depth_{}'.format(NO_LAYERS, DEPTH, str(datetime.datetime.now())))
x_data, y_data = create_dataset()


x_data, y_data = torch.FloatTensor(x_data), torch.FloatTensor(y_data)
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.15)

x_test = read_test_data('single')
x_test = torch.FloatTensor(x_test)

# Instantiate the model
model = NeuralNetwork(len(x_train[0]), len(y_train[0]))

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    optimizer.zero_grad()
    output = model(x_train)
    train_loss = criterion(output, y_train)

    train_loss.backward()
    optimizer.step()

    with torch.no_grad():
        output = model(x_val)
        val_loss = criterion(output, y_val)

    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)

predicted_price = model(x_test)
print(predicted_price)