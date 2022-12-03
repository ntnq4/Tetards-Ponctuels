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
    #---MODIFY START---#
    with open('./parameters/inverse_transform.pkl', 'rb') as f1:
        inverse_transform = pickle.load(f1)

    with open('./parameters/output_info.pkl', 'rb') as f2:
        output_info = pickle.load(f2)
    #---MODIFY END---#

    # Load generator
    #---MODIFY START---#
    generator = Generator(
        embedding_dim=50, 
        generator_dim=(256, 256, 256, 256, 256, 128, 128, 64), 
        data_dim=61)
    
    generator.load_state_dict(torch.load('./parameters/new_generator.pkl', map_location=torch.device('cpu')))
    #---MODIFY END---#

    # Generate data
    data = []
    fake = generator(torch.tensor(noise))
    fakeact = apply_activate(fake, output_info)
    data.append(fakeact.detach().cpu().numpy())

    data = np.concatenate(data, axis=0)
    data = data[:noise.shape[0]]
    data = inverse_transform(data)

    data.rename(columns={'s1.value':'s1', 's2.value':'s2', 's3.value':'s3', 's4.value':'s4', 's5.value':'s5', 's6.value':'s6'}, inplace=True)

    # Loading rescaling parameters
    #---MODIFY START---#
    mean_rescale = pd.read_csv('./parameters/PredictedMean.csv').drop(columns=['year'])
    std_rescale = pd.read_csv('./parameters/PredictedStd.csv').drop(columns=['year'])

    data_rescale = data.copy()

    # Rescaling data
    for y in range(9) :
        mean = mean_rescale.iloc()[y]
        std = std_rescale.iloc()[y]

        i, j = 365*y, 365*(y+1)
        data_rescale.iloc()[i:j] = data.iloc()[i:j]*std + mean

    data_rescale.iloc()[3285:] = data.iloc()[3285:]*std + mean
    
    # TESTING
    """
    mean_train = pd.DataFrame.from_dict({'s1':[0.36908192814791435],
                                         's2':[0.08642073719666811], 
                                         's3':[0.49039190091339163],
                                         's4':[0.34960050242313123],
                                         's5':[0.35811780059425374],
                                         's6':[0.7831963662360589]}).mean()

    std_train = pd.DataFrame.from_dict({'s1':[1.3780324293187312],
                                        's2':[1.840155294882411], 
                                        's3':[1.6039499681791898],
                                        's4':[1.2301866377403834],
                                        's5':[1.7038051175878086],
                                        's6':[1.7579832367606385]}).mean()

    data_rescale = data.copy()*std_train+mean_train
    """
    #---MODIFY END---#

    return data_rescale