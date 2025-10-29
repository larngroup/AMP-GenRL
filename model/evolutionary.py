#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 08:29:49 2024

@author: maryam
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 11:07:19 2024

@author: maryam
"""

from utils.utils import Utils
# from model.transformers import Transformer_Decoder
from model.transformers import Masked_Smiles_Model
from dataloader.dataloader import DataManager

import tensorflow as tf
import numpy as np
import time

tf.config.run_functions_eagerly(True)

class Evolutionary():
    """Optimization of the fine-tuned AMP Generator"""
    def __init__(self, FLAGS):
        

        # Implementation parameters
        self.FLAGS = FLAGS
        
        # Load the table of possible tokens
        self.token_table = Utils().vocabulary_aminoacids 
        self.vocab_size = len(self.token_table)
        
        # Dictionary that makes the correspondence between each aminoacid and unique integers
        self.tokenDict = Utils.aminoacid_dict(self.token_table)
        self.inv_tokenDict = {v: k for k, v in self.tokenDict.items()}
        
        # Loading of the fine-tuned model
        self.max_length = 100  
        self.encoder = Masked_Smiles_Model(256,1024,4,4,self.max_length,self.vocab_size,'relu')  
        sequence_in = tf.ones([1, 1])
        decoder_output,_,_ = self.decoder(sequence_in)
        path = 'encoder_finetuning.h5'
        self.encoder.load_weights(path)
               

        
    def mutation_operator(self,initial_peptides):
        
        # pre-processing the generated peptides: tokenization, padding and masking
        processed_initial_peptides = DataManager.process_buffer(initial_peptides)
        
        # predict the masked tokens
        mutated = []
        for sequences in enumerate(processed_initial_peptides): 
            predictions = self.encoder(sequences)
            mutated.append(predictions)
            
        # transform mutated ints into peptides
        
        # combine mutated and original peptiedes
        all_population = mutated + initial_peptides
        
        # evaluate all (AMA or other properties)
        
        # select top 50 peptides