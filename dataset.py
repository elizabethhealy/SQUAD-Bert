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

#create dataset
class BertDataset(Dataset):
    """BertDataset is a torch dataset to interact with the squad bert data.

    :param data: The tokenized dataset
    """
    def __init__(self, questions, answers, contexts):
        self.questions = questions
        self.answers = answers
        self.contexts = contexts
    
    def __len__(self):
        """__len__ returns the number of samples in the dataset.

        :returns: number of samples in dataset
        :rtype: int
        """
        return len(self.questions)
    
    def __getitem__(self, index):
        """__getitem__ returns context, question, answer text, start and end for given item
        """
        return contexts[index], questions[index], answers[index]["text"
        ], answers[index]["answer_start"], answers[index]["answer_end"]