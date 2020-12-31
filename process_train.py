import random
from transformers import BertTokenizerFast
import sys
import os
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
from validation import model_eval, get_metric_info, get_metrics, print_metrics


def get_data_from_json():
  with open('train-v2.0.json') as f:
    data = json.load(f)
    data = data["data"]
    return data


# preprocessing
def preprocess(data):
  answers = []
  contexts = []
  questions = []
  for article in data:
      for paragraph in article['paragraphs']:
          #looking at each context
          context = paragraph['context']
          for qas in paragraph['qas']:
              question = qas['question']
              #check if question is impossible
              #if not
              if qas['answers']!=[]:
                for answer in qas['answers']:
                    answers.append(answer)
                    contexts.append(context)
                    questions.append(question)
              #if it is impossible
              else:
                answers.append({"text": ""})
                contexts.append(context)
                questions.append(question)

  for answer, context in zip(answers, contexts):
    #if the question is unanswerable
    if len(answer.keys())==1:
      #set the start after the end
      answer['answer_start'] = len(context)-1
      answer['answer_end'] = 0
    else:
      #else get answer start and end
      answer_start = answer['answer_start']
      expected_answer = answer['text']
      answer_end = answer_start + len(expected_answer)

      #took this idea from the huggingface preprocessing
      if context[answer_start:answer_end] == expected_answer:
          answer['answer_end'] = answer_end
      elif context[answer_start-1:answer_end-1] == expected_answer:
          answer['answer_end'] = answer_end - 1
          answer['answer_start'] = answer_start - 1
      elif context[answer_start-2:answer_end-2] == expected_answer:
          answer['answer_end'] = answer_end - 2
          answer['answer_start'] = answer_start - 2
  return answers, contexts, questions


#split into train and val
def split_train_val(answers, contexts, questions, p_used=0.5, p_valid=0.1):
  train_data = []
  valid_data = []
  total = len(answers)

  #only training on half
  want = int(total*p_used)

  #use random sample of indicies
  indicies = random.sample(range(0, total), want)
  num_i = len(indicies)
  num_valid_i = int(num_i*p_valid)

  #seperate
  for i in range(0,num_valid_i):
    valid_data = valid_data + [(answers[indicies[i]], contexts[indicies[i]],
                                questions[indicies[i]])]
  for i in range(num_valid_i, num_i):
    train_data = train_data + [(answers[indicies[i]], contexts[indicies[i]],
                                questions[indicies[i]])]

  train_answers, train_contexts, train_questions = zip(*train_data)
  valid_answers, valid_contexts, valid_questions = zip(*valid_data)

  return [train_answers, train_contexts, train_questions], [valid_answers,
   valid_contexts, valid_questions]


#translate the start and end char position to token position 
def include_ans_start_end(tokenized, answers):
  answer_start_token_pos = []
  answer_end_token_pos = []

  #for each answer
  for idx, answer in enumerate(answers):
    a_start = answer['answer_start']
    a_end = answer['answer_end']-1
    if a_end<0: a_end = 0
    #get start and end tokens
    start_p = tokenized.char_to_token(idx, a_start)
    end_p = tokenized.char_to_token(idx, a_end)
    answer_start_token_pos.append(start_p)
    answer_end_token_pos.append(end_p)

    #truncation is true for tokenizer so check for it - huggingface docs
    if answer_end_token_pos[-1] is None:
      answer_end_token_pos[-1] = 511
    if answer_start_token_pos[-1] is None:
      answer_start_token_pos[-1] = 511

  tokenized.update({'answer_start': answer_start_token_pos, 'answer_end': answer_end_token_pos})

def train_epoch(model, train_loader, optimizer, tokenizer):
  model.train()
  total = 0
  loss = 0
  correct = 0
  total_loss = 0
 
  for batch in tqdm(train_loader, leave=False, desc="Training Batches"):
    #get the context, question, answers for batch
    context_batch = list(batch[0])
    question_batch = list(batch[1])
    answer_text = batch[2]
    answer_start = batch[3]
    answer_end = batch[4]
    answer_batch = [{"text": answer_text[i], "answer_start": answer_start[i]
                     , "answer_end": answer_end[i]} for i in range(len(answer_text))]

    #tokenize and add ans data
    my_batch = tokenizer(context_batch, question_batch, padding='max_length',
                         truncation=True, max_length=512)
    include_ans_start_end(my_batch, answer_batch)

    #zero the gradients
    optimizer.zero_grad()

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
    
    #calculate loss
    start_loss = model.compute_loss(start_output, start_positions.to(get_device()))
    end_loss = model.compute_loss(end_output, end_positions.to(get_device()))
    total_loss = start_loss + end_loss #sum the loss??    

    #backward and step
    total_loss.backward()
    optimizer.step()

  print("Train Loss: ", total_loss.item())


  #training method
def my_train(epochs, model, train_loader, tokenizer, lr = 1e-4):
  optimizer = Adam(model.parameters(), lr=lr)
  times = []
  for epoch in trange(epochs, desc="Epochs"):
    print("EPOCH: ", epoch)
    train_start = time.process_time()
    train_epoch(model, train_loader, optimizer, tokenizer)
    train_end = time.process_time()
    print("Time: ", train_end - train_start)
    times.append(train_end-train_start)


# Lambda to switch to GPU if available
get_device = lambda : "cuda:0" if torch.cuda.is_available() else "cpu"


data = get_data_from_json()

answers, contexts, questions = preprocess(data)

train_d, valid_d = split_train_val(answers, contexts, questions, p_used = 0.5, p_valid = 0.1)

train_ans, train_cont, train_quest = tuple(train_d)
valid_ans, valid_cont, valid_quest = tuple(valid_d)
del train_d
del valid_d

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

#get datasets
train_dataset = BertDataset(train_cont, train_quest, train_ans)
valid_dataset = BertDataset(valid_cont, valid_quest, valid_ans)

#get dataloaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(valid_dataset, batch_size=8, shuffle=True)

#start model
model = MySQUAD().to(get_device())

#train the model
torch.set_grad_enabled(True)
my_train(4, model, train_loader, tokenizer, lr = 3e-5)
torch.set_grad_enabled(False)

#save the model
model_path = os.path.join(os.getcwd(), "my_squad.pth")
model.save_model(model_path)

expected, predicted = model_eval(model, val_loader, tokenizer)

#get info got metric calcualtion
metric_info = get_metric_info(expected, predicted)
tok_info = metric_info[0]
unans_info = metric_info[1]
exact_info = metric_info[2]
num_questions = metric_info[3]

#calculate metrics
tok_metrics, unans_metrics, exact = get_metrics(tok_info, unans_info, exact_info)
tok_precision, tok_recall, tok_f1 = tok_metrics
unans_precision, unans_recal, unans_f1 = unans_metrics

#print metrics
print_metrics(tok_metrics, unans_metrics, exact, unans_info, num_questions)