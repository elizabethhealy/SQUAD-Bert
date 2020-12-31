import os
import pathlib
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import re
import subprocess
import requests
import torch
import numpy as np
from transformers import BertTokenizerFast, BertModel
import torch
import torch.nn as nn
import time
from model import MySQUAD

 
get_device = lambda : "cuda:0" if torch.cuda.is_available() else "cpu"

model = MySQUAD().to(get_device())
model_path = os.path.join(os.getcwd(), "my_squad.pth")
model.load_model(model_path)
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
 
model.eval()
print("Model loaded")
 
 
def predict(question, context):
    '''predict uses your PCE model and predicts the answer and answerability
 
    Takes a `question` string and an `context` string (which contains the
    answer), and identifies the words within the `context` that are the
    answer, and if there is an answer
 
    :param question: The question to answer
    :type question: str
    :param context: The context in which to search for the answer
    :type context: str
    :return: A pair of (answer, hasAns) that represents the model's prediction
        on the answer and whether this question is answerable
    :rtype: Tuple[str, bool]
    '''
    with torch.no_grad():
        # ======== Tokenize ========
        # Apply the tokenizer to the input text, treating them as a text-pair.
        context_batch = [context]
        question_batch = [question]
        my_batch = tokenizer(context_batch, question_batch, padding='max_length',
                         truncation=True, max_length=512)

        input_ids = torch.tensor(my_batch['input_ids']).to(get_device())
        token_type_ids = torch.tensor(my_batch['token_type_ids']).to(get_device())
        attention_mask = torch.tensor(my_batch['attention_mask']).to(get_device())
 
        # ======== Evaluate ========
        # Run our example question through the model.
        start_output, end_output = model(input_ids, token_type_ids=token_type_ids,
                                     attention_mask = attention_mask)
 
        # ======== Reconstruct Answer ========
        # Find the tokens with the highest `start` and `end` scores.
        ids = my_batch['input_ids'][0]
        iso = torch.argmax(start_output[0].cpu())
        ieo = torch.argmax(end_output[0].cpu())
        tokens = tokenizer.convert_ids_to_tokens(ids)
        has_ans = True
        if iso > ieo: 
          answer = ""
          has_ans = False
        else:
          answer = tokens[iso]
          for j in range(iso+1, ieo+1):
            if tokens[j][0:2] == "##":
              answer = answer + tokens[j][2:]
            else:
              answer = answer + "" + tokens[j]
 
        return answer, has_ans