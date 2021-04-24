import torch

class TestNet224x224(torch.nn.Module):
    def __init__(self, n_channels_in: int, n_classes: int):
        super(TestNet224x224, self).__init__()
        self.conv1 = torch.nn.Conv2d(n_channels_in, 30, (5, 5), padding=0)
        self.conv2 = torch.nn.Conv2d(30, 15, (3, 3), padding=0)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(15 * 54**2, 128)
        self.fc2 = torch.nn.Linear(128, 50)
        self.fc3 = torch.nn.Linear(50, n_classes)

    def forward(self, x):
        # 1. Convolutional layer with 30 feature maps of size 5×5 and ReLU activation.
        out = self.conv1(x)
        out = torch.nn.functional.relu(out)
        # 2. Pooling layer taking the max over 2×2 patches.
        out = torch.nn.functional.max_pool2d(out, (2, 2))
        # 3. Convolutional layer with 15 feature maps of size 3×3 and ReLU activation.
        out = self.conv2(out)
        out = torch.nn.functional.relu(out)
        # 4. Pooling layer taking the max over 2×2 patches.
        out = torch.nn.functional.max_pool2d(out, (2, 2))
        # 5. Dropout layer with a probability of 20%.
        out = torch.nn.functional.dropout(out, 0.2)
        # 6. Flatten layer.
        out = self.flatten(out)
        # 7. Fully connected layer with 128 neurons and ReLU activation.
        out = self.fc1(out)
        out = torch.nn.functional.relu(out)
        # 8. Fully connected layer with 50 neurons and ReLU activation.
        out = self.fc2(out)
        out = torch.nn.functional.relu(out)
        # 9. Linear output layer.
        out = self.fc3(out)
        #
        if not self.training:
            out = torch.nn.functional.softmax(out, dim=1)
        #
        return out


class SimpleCNN224x224(torch.nn.Module):
    def __init__(self, n_channels_in: int, n_classes: int):
        super(SimpleCNN224x224, self).__init__()
        self.conv1 = torch.nn.Conv2d(n_channels_in, 3, (5, 5), padding=0)
        self.flatten = torch.nn.Flatten(start_dim=1)
        self.fc1 = torch.nn.Linear(3 * (110**2), 128)
        self.fc2 = torch.nn.Linear(128, n_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = torch.nn.functional.relu(out)
        out = torch.nn.functional.max_pool2d(out, (2,2))
        out = torch.nn.functional.dropout(out, 0.2)
        out = self.flatten(out)
        out = self.fc1(out)
        out = torch.nn.functional.relu(out)
        out = self.fc2(out)
        #
        if not self.training:
            out = torch.nn.functional.softmax(out, dim=1)
        #
        return out
