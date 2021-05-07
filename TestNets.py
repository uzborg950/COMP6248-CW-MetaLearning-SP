import torch

class MAMLModule1(torch.nn.Module):

    def __init__(self, input_len, n_classes: int):
        super(MAMLModule1, self).__init__()
        #
        self.fc1 = torch.nn.Linear(input_len, 256)
        self.bn1 = torch.nn.BatchNorm1d(num_features=256)
        #
        self.fc2 = torch.nn.Linear(256, 128)
        self.bn2 = torch.nn.BatchNorm1d(num_features=128)
        #
        self.fc3 = torch.nn.Linear(128, 64)
        self.bn3 = torch.nn.BatchNorm1d(num_features=64)
        #
        self.fc4 = torch.nn.Linear(64, n_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = torch.nn.functional.relu(out)
        #
        out = self.fc2(out)
        out = self.bn2(out)
        out = torch.nn.functional.relu(out)
        #
        out = self.fc3(out)
        out = self.bn3(out)
        out = torch.nn.functional.relu(out)
        #
        out = self.fc4(out)
        #
        if not self.training:
            out = torch.nn.functional.softmax(out, dim=1)
        #
        return out


class Module1(torch.nn.Module):

    def __init__(self, input_len, n_classes: int):
        super(Module1, self).__init__()
        self.fc1 = torch.nn.Linear(input_len, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.fc4 = torch.nn.Linear(64, n_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = torch.nn.functional.relu(out)
        #
        out = self.fc2(out)
        out = torch.nn.functional.relu(out)
        #
        out = self.fc3(out)
        out = torch.nn.functional.relu(out)
        #
        out = self.fc4(out)
        #
        if not self.training:
            out = torch.nn.functional.softmax(out, dim=1)
        #
        return out


class Module2(torch.nn.Module):

    def __init__(self, input_len, n_classes: int):
        super(Module2, self).__init__()
        self.fc1 = torch.nn.Linear(input_len, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc4 = torch.nn.Linear(128, 64)
        self.fc5 = torch.nn.Linear(64, n_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = torch.nn.functional.relu(out)
        #
        out = self.fc2(out)
        out = torch.nn.functional.relu(out)
        #
        out = self.fc3(out)
        out = torch.nn.functional.relu(out)
        #
        out = self.fc4(out)
        out = torch.nn.functional.relu(out)
        #
        out = self.fc5(out)
        #
        if not self.training:
            out = torch.nn.functional.softmax(out, dim=1)
        #
        return out


class Module3(torch.nn.Module):

    def __init__(self, input_len, n_classes: int):
        super(Module3, self).__init__()
        #
        self.fc1 = torch.nn.Linear(input_len, 1024)
        self.bn1 = torch.nn.BatchNorm1d(num_features=1024)
        #
        self.fc2 = torch.nn.Linear(1024, 512)
        self.bn2 = torch.nn.BatchNorm1d(num_features=512)
        #
        self.fc3 = torch.nn.Linear(512, 256)
        self.bn3 = torch.nn.BatchNorm1d(num_features=256)
        #
        self.fc4 = torch.nn.Linear(256, 64)
        self.bn4 = torch.nn.BatchNorm1d(num_features=64)
        #
        self.fc5 = torch.nn.Linear(64, n_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = torch.nn.functional.relu(out)
        #
        out = self.fc2(out)
        out = self.bn2(out)
        out = torch.nn.functional.relu(out)
        #
        out = self.fc3(out)
        out = self.bn3(out)
        out = torch.nn.functional.relu(out)
        #
        out = self.fc4(out)
        out = self.bn4(out)
        out = torch.nn.functional.relu(out)
        #
        out = self.fc5(out)
        #
        if not self.training:
            out = torch.nn.functional.softmax(out, dim=1)
        #
        return out
