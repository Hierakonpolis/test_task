import pytorch_lightning as pl
import torch
from torchvision.models import resnet18
from parametrization import parametrize_cnn
import numpy as np

def accuracy_from_tensors(prediction, labels):
    return  torch.tensor(np.mean(tuple(torch.argmax(prediction.detach().cpu(), dim=1) == labels.detach().cpu())))

class ResnetWrapper(pl.LightningModule):
    def __init__(self, parametrize=False,
                 max_units_per_mlp_layer = 250,
                 n_hidden_layers=5,
                 ):
        super(ResnetWrapper, self).__init__()
        self.resnet18 = resnet18(num_classes=10)
        # cifar10 is smol
        self.resnet18.conv1 = torch.nn.Conv2d(3, 64, 3)
        self.resnet18.maxpool = torch.nn.Identity() # .77 without

        self.loss = torch.nn.CrossEntropyLoss()
        def parametrizing_function(x):
            return parametrize_cnn(x, max_units_per_mlp_layer, n_hidden_layers)

        if parametrize:
            for module in self.resnet18.children():
                module.apply(parametrizing_function)
                # 10 total classes

    def forward(self, x):
        return self.resnet18(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        prediction = self(images)
        loss = self.loss(prediction, labels)
        self.log('Train loss', loss)
        self.log('Train accuracy', accuracy_from_tensors(prediction, labels))
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        prediction = self(images)
        loss = self.loss(prediction, labels)
        self.log('Validation loss', loss)
        self.log('Validation accuracy', accuracy_from_tensors(prediction, labels))
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        prediction = self(images)
        loss = self.loss(prediction, labels)
        self.log('Test loss', loss)
        self.log('Test accuracy', accuracy_from_tensors(prediction, labels))
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.00001)
