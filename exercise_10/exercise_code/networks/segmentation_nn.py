"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl

from torchvision import models

class SegmentationNN(pl.LightningModule):

    def __init__(self, num_classes=23, hparams=None, train_set=None, val_set=None):
        super().__init__()
        self.hparams = hparams
        self.train_set = train_set
        self.val_set = val_set
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        pretrained = models.mobilenet_v2(pretrained=True, progress=True).eval()
        for param in pretrained.parameters():
            param.requires_grad = False
            

        self.model = nn.Sequential(
          *(list(pretrained.children())[:-1]),
          # mobilenet_v2 output is 1280-dimensional
          nn.ConvTranspose2d(1280, self.hparams['channel_1'], kernel_size=3,stride=2),
          nn.ConvTranspose2d(self.hparams['channel_1'], self.hparams['channel_2'], kernel_size=3,stride=2),
          nn.ConvTranspose2d(self.hparams['channel_2'], self.hparams['channel_3'], 1),
          torch.nn.Upsample(size=(240, 240)),
        )
        pass

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        x = self.model(x)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    def general_step(self, batch, batch_idx, mode):
        images = batch[0]
        gt_category = batch[1]

        # forward pass
        predicted_category = self.forward(images)

        # loss
        loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')(predicted_category, gt_category)

        return loss

    def training_step(self, batch, batch_idx):
        loss  = self.general_step(batch, batch_idx, "train")
        tensorboard_logs = {'loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss  = self.general_step(batch, batch_idx, "val")
        return loss

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, shuffle=True, batch_size=self.hparams['batch_size'], drop_last=True, num_workers=16)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.hparams['batch_size'],  drop_last=True, num_workers=16)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), self.hparams["learning_rate"])
        return optim

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
