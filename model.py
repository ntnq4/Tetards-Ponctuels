#############################################################################
# YOUR GENERATIVE MODEL
# ---------------------
# Should be implemented in the 'generative_model' function
# !! *DO NOT MODIFY THE NAME OF THE FUNCTION* !!
#
# You can store your parameters in any format you want (npy, h5, json, yaml, ...)
# <!> *SAVE YOUR PARAMETERS IN THE parameters/ DICRECTORY* <!>
#
# See below an example of a generative model
# G_\theta(Z) = np.max(0, \theta.Z)
############################################################################

import numpy as np
import os

# START MODIFIED
import torch
from torch.nn import BatchNorm1d, Linear, Module, ReLU, Sequential
from torch.nn import functional
#from ctgan.data_transformer import SpanInfo
import numpy as np
from sdv.tabular import CTGAN

class Residual(Module):

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input_):
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)

class Generator(Module):

    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input_):
        data = self.seq(input_)
        return data

def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
  return functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)

def apply_activate(data, output_info):
  """Apply proper activation function to the output of the generator."""
  data_t = []
  st = 0
  for column_info in output_info:
      for span_info in column_info:
          if span_info.activation_fn == 'tanh':
              ed = st + span_info.dim
              data_t.append(torch.tanh(data[:, st:ed]))
              st = ed
          elif span_info.activation_fn == 'softmax':
              ed = st + span_info.dim
              transformed = gumbel_softmax(data[:, st:ed], tau=0.2)
              data_t.append(transformed)
              st = ed
          else:
              raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

  return torch.cat(data_t, dim=1)
# END MODIFIED

# <!> DO NOT ADD ANY OTHER ARGUMENTS <!>
def generative_model(noise):
    """
    Generative model

    Parameters
    ----------
    noise : ndarray with shape (n_samples, n_dim)
        input noise of the generative model
    """
    # Load data transformations and output information
    gan_model = CTGAN.load('./parameters/gan_model.pkl')
    inverse_transform = gan_model._model._transformer.inverse_transform
    output_info = gan_model._model._transformer.output_info_list
    
    # Load generator
    generator = Generator(embedding_dim=50, generator_dim=(256, 256, 256, 256, 128, 64, 32), data_dim=60)
    generator.load_state_dict(torch.load('./parameters/generator'))

    # Generate data
    data = []
    fake = generator(torch.tensor(noise))
    fakeact = apply_activate(fake, output_info)
    data.append(fakeact.detach().cpu().numpy())

    data = np.concatenate(data, axis=0)
    data = data[:noise.shape[0]]
    data = inverse_transform(data)

    data.rename(columns={'s1.value':'s1', 's2.value':'s2', 's3.value':'s3', 's4.value':'s4', 's5.value':'s5', 's6.value':'s6'}, inplace=True)

    #Loading rescaling parameters
    param = np.load('./parameters/rescaling_parameters_prev1.npy', allow_pickle=True)
    mean = param.item()['mean']
    std = param.item()['std']

    data = data*std+mean
    
    return data