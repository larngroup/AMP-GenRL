#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:22:58 2024

@author: maryam
"""

import torch
from transformers import BertTokenizer, pipeline, BertForMaskedLM
import numpy as np

tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd',do_lower_case=False)
model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert_bfd",output_attentions=True)
unmasker = pipeline('fill-mask',model=model, tokenizer=tokenizer)
a = unmasker('D L I P T S S K L V V [MASK] D T S L Q V [MASK] K A F F A L V T')
random_integers = [11,18]

# for i in range(len(random_integers)):
#     index = random_integers[i]
#     original_input_list[index] = a[i][0]['token_str']
# ms = ' '.join(original_input_list)
# r = a[0][0]['token_str']

seq_inp = a[0][0]['sequence']

# Tokenize input sequence
tokens = tokenizer.tokenize(seq_inp)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
inputs = torch.tensor([token_ids])

# Predict masked token
with torch.no_grad():
    outputs = model(inputs)
    predictions = outputs[0]
    att = outputs.attentions
    
    
# att : tuple[n_layers][batch,n_heads,seq_len,seq_len] 

tensor_array = np.array(att)

# Combine the tensors along a new first axis
combined_array = np.concatenate(tensor_array, axis=0)