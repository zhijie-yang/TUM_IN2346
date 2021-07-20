import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn.utils.rnn import pack_padded_sequence

from .rnn_nn import Embedding, RNN, LSTM


class RNNClassifier(pl.LightningModule):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, use_lstm=True, **additional_kwargs):
        """
        Inputs:
            num_embeddings: size of the vocabulary
            embedding_dim: size of an embedding vector
            hidden_size: hidden_size of the rnn layer
            use_lstm: use LSTM if True, vanilla RNN if false, default=True
        """
        super().__init__()

        # Change this if you edit arguments
        self.hparams = {
            'num_embeddings': num_embeddings,
            'embedding_dim': embedding_dim,
            'hidden_size': hidden_size,
            'use_lstm': use_lstm,
            'num_layers': 3,
            **additional_kwargs
        }

        ########################################################################
        # TODO: Initialize an RNN network for sentiment classification         #
        # hint: A basic architecture can have an embedding, an rnn             #
        # and an output layer                                                  #
        ########################################################################

        self.embedding = nn.Embedding(self.hparams['num_embeddings'], self.hparams['embedding_dim'], padding_idx=0)
        self.lstm = nn.LSTM(self.hparams['embedding_dim'], self.hparams['hidden_size'], num_layers=3, batch_first=False)
        self.linear = nn.Sequential(
            nn.Linear(self.hparams['hidden_size'], int(self.hparams['hidden_size'] / 2)),
            nn.ReLU(),
            nn.Linear(int(self.hparams['hidden_size'] / 2), 1)
        )

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, sequence, lenghts=None):
        """
        Inputs
            sequence: A long tensor of size (seq_len, batch_size)
            lengths: A long tensor of size batch_size, represents the actual
                sequence length of each element in the batch. If None, sequence
                lengths are identical.
        Outputs:
            output: A 1-D tensor of size (batch_size,) represents the probabilities of being
                positive, i.e. in range (0, 1)
        """
        output = None

        ########################################################################
        # TODO: Apply the forward pass of your network                         #
        # hint: Don't forget to use pack_padded_sequence if lenghts is not None#
        # pack_padded_sequence should be applied to the embedding outputs      #
        ########################################################################

        embedded_sequence = self.embedding(sequence)
        if lenghts != None:
            embedded_sequence = pack_padded_sequence(embedded_sequence, lenghts)
        state_size = (self.hparams['num_layers'], sequence.size(1), self.hparams['hidden_size'])
        if self.hparams['use_lstm']:
            _, (h, _) = self.lstm(embedded_sequence, (torch.zeros(state_size), torch.zeros(state_size)))
        h = h[-1]  # get last LSTM layer output
        
        output = self.linear(h)
        output = nn.Sigmoid()(output).squeeze(1).view(-1)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return output

    def general_step(self, batch, batch_idx):
        estimation = self.forward(batch['data'], batch['lengths'])
        loss = nn.BCEWithLogitsLoss()(estimation, batch['label'])
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx)
        #loss = loss.to('cuda')
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        estimation = self.forward(batch['data'], batch['lengths'])
        loss = nn.BCEWithLogitsLoss()(estimation, batch['label'])
        return {'val_loss': loss}

    def configure_optimizers(self):
        learning_rate = self.hparams.get('learning_rate', 1e-3)
        optim = torch.optim.Adam(
            list(self.lstm.parameters()) + \
            list(self.linear.parameters()) + \
            list(self.embedding.parameters()),
            learning_rate)
        return optim