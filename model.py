import random
from transformers import BertTokenizerFast
import sys
import torch
import time
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import time
from tqdm import trange, tqdm
from torch.optim import Adam
from transformers import BertModel
from model import MySQUAD
from dataset import BertDataset
import collections
import json
import string
import re
import numpy as np



#create model
class MySQUAD(nn.Module):
  def __init__(self):
    super(MySQUAD, self).__init__()
    self.bert = BertModel.from_pretrained('bert-base-uncased')
    self.num_labels = 2 # want to output for start and end probs
    self.linear = self.linear = nn.Linear(768, self.num_labels)
    self.softmax = nn.LogSoftmax(dim=1)
    self.loss = nn.NLLLoss()

  def forward(self, input_vector, token_type_ids = None, attention_mask=None):
    #pass through bert
    bert_output = self.bert(input_vector, token_type_ids=token_type_ids,
                            attention_mask = attention_mask)
    #pass through linear layer
    linear_outs = self.linear(bert_output[0])
    reshaped = linear_outs.permute(2,0,1)

    start_output, end_output = reshaped[0], reshaped[1]
    
    #adjust shape
    end_output = self.softmax(end_output)
    start_output = self.softmax(start_output)

    return start_output, end_output
  
  def compute_loss(self, pred_output, positions):
    return self.loss(pred_output, positions)

  def load_model(self, save_path):
    self.load_state_dict(torch.load(save_path))

  def save_model(self, save_path):
    torch.save(self.state_dict(), save_path)