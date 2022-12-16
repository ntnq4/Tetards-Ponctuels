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
import numpy as np
from sdv.tabular import CTGAN
import pandas as pd
import pickle
import keras

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
def generative_model(noise, position):
    """
    Generative model

    Parameters
    ----------
    noise : ndarray with shape (n_samples, n_dim)
        input noise of the generative model
    """
    # Load data transformations and output information
    #---MODIFY START---#
    with open('./parameters/inverse_transform.pkl', 'rb') as f1:
        inverse_transform = pickle.load(f1)

    with open('./parameters/output_info.pkl', 'rb') as f2:
        output_info = pickle.load(f2)

    mean_std = keras.models.load_model('./parameters/model_mean_std.h5')
    #---MODIFY END---#

    # Load generator
    #---MODIFY START---#
    generator = Generator(
        embedding_dim=50,
        generator_dim=(256, 256, 256, 256, 256, 128, 128, 64),
        data_dim=11)

    generator.load_state_dict(torch.load('./parameters/generator.pkl', map_location=torch.device('cpu')))
    #---MODIFY END---#

    X = position[:6]
    Y = position[6:]

    # Generate data
    mean, std = [], []
    for i in range(6):
      p = mean_std.predict(np.array([[X[i], Y[i]]]))
      mean.append(p[0][0])
      std.append(p[0][1])

    dr = []

    for j in range(6):
      data = []
      fake = generator(torch.tensor(noise))
      fakeact = apply_activate(fake, output_info)
      data.append(fakeact.detach().cpu().numpy())
      data = np.concatenate(data, axis=0)
      data = data[:noise.shape[0]]
      data = inverse_transform(data)
      dr.append(data*std[j]+mean[j])

    data_rescale = pd.concat([dr[0],dr[1],dr[2],dr[3],dr[4],dr[5]], axis=1)

    return data_rescale