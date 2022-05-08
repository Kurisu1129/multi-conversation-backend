import os
import random
import sys
from collections import Counter
import numpy as np
import re
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch.nn.functional as F
import jieba
import warnings
import math
import numpy 
import ModelClass
from ModelClass import Voc_Params
from ModelClass import Voc
from ModelClass import MyDataset
from ModelClass import Dataset_Params
from ModelClass import Seq2SeqModel



def loadModel():
    modelLoad = torch.load("Seq2SeqModel-120round.pt")
    print(modelLoad)
    return modelLoad

def getResult(modelLoad):
    voc_params = Voc_Params(is_train=True)
    voc = modelLoad.voc
    voc_test = Voc(voc_params)
    test_dataset_params = Dataset_Params(is_train=False)
    test_dataset = MyDataset(test_dataset_params, voc_test)
    result = dict()
    print("in getResult, contexts.size=", len(test_dataset.contexts))
    #with open('./data/multi_test_all_qa.txt', 'w', encoding='utf-8') as f:
    for i in range(len(test_dataset.contexts)):
    # for i in range(10):
         context = [test_dataset.contexts[i]]
         length = [test_dataset.contexts_lens[i]]
         targets = [test_dataset.targets[i]]
             # tokens_top = model.BeamSearchIters(context, length)
         tokens_top = modelLoad.BeamSearchIters(context, length)
         answer_tokens = tokens_top.cpu().squeeze()
         target_tokens = targets[0].cpu().squeeze()

         mask_answer = torch.BoolTensor(answer_tokens >2)
         mask_target = torch.BoolTensor(target_tokens >2)

         answer_tokens = torch.masked_select(answer_tokens, mask_answer)
         target_tokens = torch.masked_select(target_tokens, mask_target)
    
#     answer_tokens = "".join(answer_tokens.numpy().tolist())
#     target_tokens = "".join(target_tokens.numpy().tolist())
#     print(answer_tokens)
#     print(target_tokens)

         answer = voc.sequence2sentence(answer_tokens.numpy().tolist())
         target = voc.sequence2sentence(target_tokens.numpy().tolist())
         #f.write(''.join(answer) + '\n')
         result['answer'] = ''.join(answer)
         result['target'] = ''.join(target)
        #print(voc.sequence2sentence(context[0].numpy().tolist()))
        #print(answer)
        #print(target)
#     output_sentence = list("".join(answer))
#     output_target = list("".join(target))
#     print(output_sentence)
#     print(output_target)
#     corpus_score_2 = corpus_bleu(output_sentence, output_target, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth.method1)
#     print(corpus_score_2)

#     print(output_sentence)
#     print(output_target)
#     rouge_score = rouge.get_scores(output_sentence, output_target)
#     print(rouge_score)
    return result


def saveInput(input):
    with open('./data/multi_test.txt', 'w', encoding='utf-8') as f:
        f.write(input)


def testDataset(modelLoad):
    voc_params = Voc_Params(is_train=True)
    voc_test_params = Voc_Params(is_train=False)
    voc = modelLoad.voc
    voc_test = Voc(voc_params)
    test_dataset_params = Dataset_Params(is_train=False)
    test_dataset = MyDataset(test_dataset_params, voc_test)
    print(test_dataset.contexts)

def getOptions():
    result = []
    with open('./data/options.txt', 'r', encoding='utf-8') as f:
        for line in f:
            strSplit = line.split('\t')
            temp = dict()
            temp['value'] = line
            temp['label'] = strSplit[0]
            result.append(temp)
    return result