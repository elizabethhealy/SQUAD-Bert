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

# Some potentially useful helper functions for the metrics
def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)

def white_space_fix(text):
    return ' '.join(text.split())

def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)

def lower(text):
    return text.lower()

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()



def compute_precision(num_true_pos, num_pred):
  if num_pred==0:
    return 0
  return num_true_pos/num_pred

def compute_recall(num_true_pos, num_exp):
  if num_exp == 0:
    return 0
  return num_true_pos/num_exp

def compute_exact_m1(num_exact_matches, num_total):
  return num_exact_matches/num_total

def compute_f1(precision, recall):
    return 2*precision*recall/(precision+recall)

  

def model_eval(model, val_loader, tokenizer):
  model.eval()
  predicted = []
  expected = []

  for batch in tqdm(val_loader, leave=False, desc="Validation Batches"):
    context_batch = list(batch[0])
    question_batch = list(batch[1])
    answer_text = list(batch[2])
    answer_start = batch[3]
    answer_end = batch[4]
    answer_batch = [{"text": answer_text[i], "answer_start": answer_start[i]
                     , "answer_end": answer_end[i]} for i in range(len(answer_text))]

    my_batch = tokenizer(context_batch, question_batch, padding='max_length',
                         truncation=True, max_length=512)
    include_ans_start_end(my_batch, answer_batch)

    #make lists to tensors
    #get the inputs
    input_ids = torch.tensor(my_batch['input_ids']).to(get_device())
    start_positions = torch.tensor(my_batch['answer_start']).to(get_device())
    end_positions = torch.tensor(my_batch['answer_end']).to(get_device())
    token_type_ids = torch.tensor(my_batch['token_type_ids']).to(get_device())
    attention_mask = torch.tensor(my_batch['attention_mask']).to(get_device())

    #get the outputs
    start_output, end_output = model(input_ids, token_type_ids=token_type_ids,
                                     attention_mask = attention_mask)
    
    #update expected
    expected = expected + answer_text

    for i in range(len(start_output)):
      ids = my_batch['input_ids'][i]
      iso = torch.argmax(start_output[i])
      ieo = torch.argmax(end_output[i])
      tokens = tokenizer.convert_ids_to_tokens(ids)
      if iso > ieo: answer = ""
      else:
        answer = tokens[iso]
        for j in range(iso+1, ieo+1):
          if tokens[j][0:2] == "##":
            answer = answer + tokens[j][2:]
          else:
            answer = answer + "" + tokens[j]
      predicted = predicted + [answer]

  return expected, predicted


def get_metric_info(expected, predicted):
  num_pred_tok = 0
  num_exp_tok = 0
  num_match_tok = 0
  num_pred_unans = 0
  num_exp_unans = 0
  num_match_unans = 0
  exact = 0

  for i in range(len(expected)):
    expected_tok = get_tokens(expected[i])
    predicted_tok = get_tokens(predicted[i])

    num_pred_tok = num_pred_tok + len(predicted_tok)
    num_exp_tok = num_exp_tok + len(expected_tok)
    match_tok = [tok for tok in predicted_tok if tok in expected_tok]
    num_match_tok= num_match_tok + len(match_tok)

    if not predicted_tok:
      num_pred_unans = num_pred_unans + 1
    if not expected_tok:
      num_exp_unans = num_exp_unans + 1
    if (not predicted_tok) and (not expected_tok):
      num_match_unans = num_match_unans + 1
    
    if predicted_tok == expected_tok:
      exact = exact + 1

  return [(num_pred_tok, num_exp_tok, num_match_tok),
          (num_pred_unans, num_exp_unans, num_match_unans), exact, len(expected)]


def get_metrics(tok_info, unans_info, exact_info, num_questions):
  tok_precision = compute_precision(tok_info[2], tok_info[0])
  unans_precision = compute_precision(unans_info[2], unans_info[0])

  tok_recall = compute_recall(tok_info[2], tok_info[1])
  unans_recal = compute_recall(unans_info[2], unans_info[1])

  exact = compute_exact_m1(exact_info, num_questions)

  tok_f1 = compute_f1(tok_precision, tok_recall)
  unans_f1 = compute_f1(unans_precision, unans_recal)

  return (tok_precision, tok_recall, tok_f1), (unans_precision, unans_recal, unans_f1), exact

def print_metrics(tok_metrics, unans_metrics, exact, unans_info, num_questions):
  tok_precision, tok_recall, tok_f1 = tok_metrics
  unans_precision, unans_recal, unans_f1 = unans_metrics
  print("MySquad Metrics")
  print("--------------------")
  print("")
  print("Span Prediction:")
  print("Precision: ", tok_precision)
  print("Recall: ", tok_recall)
  print("F1: ", tok_f1)
  print("")
  print("Unanswerability Prediction: ")
  print("Precision: ", unans_precision)
  print("Recall: ", unans_recal)
  print("F1: ", unans_f1)
  print("Percent predicted unanswerable: ", unans_info[0]/num_questions)
  print("")
  print("Exact: ", exact)
  print("")
  print("Data info:")
  print("Percent unanswerable: ", unans_info[1]/num_questions)
  print("Percent answerable: ", (num_questions-unans_info[1])/num_questions)

