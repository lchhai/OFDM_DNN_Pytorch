import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(256, 600),
            nn.ReLU(),
            nn.Linear(600, 300),
            nn.ReLU(),
            nn.Linear(300, 16)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return torch.sigmoid(logits)


if __name__ == '__main__':
    input= torch.randn((1,256))
    model = NeuralNetwork()
    print(model)
    output = model(input)
    print(output)
    print(output.shape)