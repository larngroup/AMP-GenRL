# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 16:49:39 2021

@author: tiago
"""

# internal
from utils.utils import Utils
from model.peptide_evaluation import Peptide_evaluation
from model.transformers import Transformer_Decoder
from model.warm_up_decay_schedule import WarmupThenDecaySchedule

# external
import tensorflow as tf
import numpy as np
import time
import random
from modlamp.analysis import GlobalAnalysis
from modlamp.core import count_aas
from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor
from modlamp.sequences import Random, Helices
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
import torch
from transformers import BertForMaskedLM, BertTokenizer, pipeline
import matplotlib.pyplot as plt

 
class RB_manager():
    """Replay Buffer class"""
    def __init__(self, FLAGS, transformer_class):
        # Unpack implementation parameters
        self.FLAGS = FLAGS
        
        # Load the table of possible tokens
        self.token_table = Utils().vocabulary_aminoacids 
        self.vocab_size = len(self.token_table)
        
        # Dictionary that makes the correspondence between each aminoacid and unique integers
        self.tokenDict = Utils.aminoacid_dict(self.token_table)
        self.inv_tokenDict = {v: k for k, v in self.tokenDict.items()}
        
        # Unpack Transformer class
        self.transformer_class = transformer_class
        
        #Initialization of the peptide evaluation class
        self.peptide_evaluation = Peptide_evaluation(self.FLAGS)
        
        self.masking_strategy = 'lowest_importance'
        
        # Loading of the pretrained BERT
        self.tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd',do_lower_case=False)
        self.model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert_bfd",output_attentions=True)
        self.unmasker = pipeline('fill-mask',model=self.model, tokenizer=self.tokenizer)
       
        
    
    def pre_process_data_buffer(self,train_sequences):
        """ Pre-processes the dataset of proteins including padding, 
            tokenization and transformation of tokens into integers.
            
        Args: 
            dataset (list): Set of proteins
            transf_obj (class): Transformer object 
            FLAGS (argparse): Implementation parameters
    
        Returns
        -------
            data_train (list): List with pre-processed training set
            data_test (list): List with pre-processed testing set
        """     
        sequences_no_spaces = [s.replace(" ","") for s in train_sequences]
       
       	# Tokenize - transform the protein into lists of tokens 
        tokens,proteins_filtered = Utils().tokenize_and_pad(self.FLAGS,sequences_no_spaces,self.transformer_class.token_table)   
 
        # Transforms each token to the respective integer, according to 
        # the previously computed dictionary
        input_train = Utils.proteins2idx(tokens,self.transformer_class.tokenDict)
        
        # Tokenize  &  Character to integer - training target
        tokens_target,_ = Utils().tokenize_and_pad(self.FLAGS,sequences_no_spaces,self.transformer_class.token_table,initial_token=False)   
        input_train_target = Utils.proteins2idx(tokens_target,self.transformer_class.tokenDict)
            
        # Create tf.dataset object
        data_train = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(input_train),tf.convert_to_tensor(input_train_target))).batch(self.FLAGS.batchsize, drop_remainder = False)
        
        
        print('\nNumber of sequences: ',len(tokens))    
        
        
        return data_train
    
    def augment_rb(self,sequences_original):
        train_sequences = sequences_original.copy()
        
        # TODO define two masking strategies: randomly and based on attention weights
        sequences = [s.replace(" ","") for s in train_sequences]
      
        rate_mask = 0.2
        
        masking_character = '[MASK]'
        augmented_sequences = []
        for idx,seq in enumerate(sequences):
            
            # Define the number of masked tokens
            length_seq = len(seq)
            n_mask = max(3, int(round(length_seq * rate_mask))) 
            
            # MASKING (random)
            if self.masking_strategy == 'random':

                random_integers = random.sample(range(length_seq), n_mask)
                
            else: 
                
                # print('LEN_seq: ', length_seq)
                # print('seq: ',seq )
                seq_space = ' '.join(seq)
                # print('seq_space: ',seq_space)
                # Tokenize input sequence
                tokens = self.tokenizer.tokenize(seq_space)
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                inputs = torch.tensor([token_ids])

                # Predict masked token
                with torch.no_grad():
                    outputs = self.model(inputs)
                    predictions = outputs[0]
                    atts = outputs.attentions
                # n
                if self.masking_strategy == 'highest_importance':
                    random_integers = self.attention_scores(seq,atts,'max',n_mask)
                    
                elif self.masking_strategy == 'lowest_importance':
                    random_integers = self.attention_scores(seq,atts,'min',n_mask)
                    
                elif self.masking_strategy == 'proportional_importance':
                    random_integers = self.attention_scores(seq,atts,'prop',n_mask)
                
      

            # Convert the string to a list of characters to enable modification
            modified_string_list = list(seq)
            original_string_list = list(seq)
            # Replace characters at the specified indexes with the new character
            for idx_int in random_integers:
                modified_string_list[idx_int] = masking_character
                
            # Convert the modified list back to a string
            modified_sequence = ' '.join(modified_string_list)
            
            
            # print(modified_sequence)
            # print('\n',original_string_list)
            # Prediction of new sequences
            result = self.unmasker(modified_sequence)
            # print(result)
            
            for i in range(len(random_integers)):
                index = random_integers[i]
                original_string_list[index] = result[i][0]['token_str']
            
            ms = ' '.join(original_string_list)
            # print('\new: ', ms)
            augmented_sequences.append(ms)

        
        # Evaluation
        MIC_optimized,TOX_optimized = self.peptide_evaluation.evaluate_MIC_TOX(augmented_sequences)
        
        # Selection of the fittest
        for i,aug_s in enumerate(augmented_sequences):
            if MIC_optimized[i]<20.0 and TOX_optimized[i]>100.0:
                train_sequences.append(aug_s.replace(" ", ""))
        
            
        
        # sequences_no_spaces = [s.replace(" ","") for s in train_sequences]
       
       	# Tokenize - transform the protein into lists of tokens 
        tokens,proteins_filtered = Utils().tokenize_and_pad(self.FLAGS,train_sequences,self.transformer_class.token_table)   
 
        # Transforms each token to the respective integer, according to 
        # the previously computed dictionary
        input_train = Utils.proteins2idx(tokens,self.transformer_class.tokenDict)
        
        # Tokenize  &  Character to integer - training target
        tokens_target,_ = Utils().tokenize_and_pad(self.FLAGS,train_sequences,self.transformer_class.token_table,initial_token=False)   
        input_train_target = Utils.proteins2idx(tokens_target,self.transformer_class.tokenDict)
            
        # Create tf.dataset object
        data_train = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(input_train),tf.convert_to_tensor(input_train_target))).batch(self.FLAGS.batchsize, drop_remainder = False)
        

        return data_train   
    
    
    def attention_scores(self,seq,alignments,selection_strategy,n_mask):
        
        #alignments: tuple[n_layers][batch,n_heads,seq_len,seq_len] 
        # print('GO: ',len(alignments))
        tensor_array = np.array(alignments)
        # print('G1O: ',tensor_array.shape)
        # Combine the tensors along a new first axis
        alignments_array = np.concatenate(tensor_array, axis=0) # (n_layers,n_heads,seqlen,seqlen)
        # print('GO2: ',alignments_array.shape)
        alignments_array = alignments_array.sum(axis=1) #(n_layers,seqlen.seqlen)
        # print('GO3: ',alignments_array.shape)
        tokens = list(seq)
        token_len = len(tokens)

        # alignments 4layers(1,150,150) - heads concatenated
        # aw_layers_heads 4layers(4,150,150) - all heads
        
        layer_option = 'single' # single or all
        computation_strategy = 'A' # 'A' or 'B'
        single_layer_idx = 0 # int [0,29]
        plot_attention_scores = False # True or False
        top_tokens_rate = 0.3
        plot_attention_weights = False
        activation = 'tanh' #'tanh', 'relu', 'none', 'relu', 'sigmoid'

        
        if layer_option == 'single' and computation_strategy == 'A':
            # last/first layer, all heads, raw attention values

            # Extract only the selected layer 
            selected_attention = alignments_array[single_layer_idx,:,:]
            print("Selected shape: ",selected_attention.shape)
             
            # Extract the importance of each specific token
            importance_all = []            
            size_h = len(selected_attention)
            for c in range(0,size_h):
    
                importance_element = []
                importance_element.append(selected_attention[c,c])
                
                for v in range(0,size_h):
                    if v!=c:
                        importance_element.append(selected_attention[c,v])
                        importance_element.append(selected_attention[v,c])
            
                importance_all.append(importance_element)
            importance_tokens = [np.mean(l) for l in importance_all]
     
        
        elif layer_option == 'single' and computation_strategy == 'B':            
             # Last layer, all heads, average attention values
             # Extract only the selected layer 
             selected_attention = alignments_array[single_layer_idx,:,:]
     
             # Extract the importance of each specific token
             importance_all = []            
             size_h = len(selected_attention)
             for c in range(0,size_h):
                 importance_element = []
                 importance_element.append(selected_attention[c,c])
                
                 for v in range(0,size_h):
                    if v!=c:
                        element = (selected_attention[c,v] + selected_attention[v,c])/2
                        importance_element.append(element)
                    
                 importance_all.append(importance_element)
             importance_tokens = [np.mean(l) for l in importance_all]
    
        elif layer_option == 'all' and computation_strategy == 'A':
        
            # concatenate all layers
            selected_attention = alignments_array.sum(axis=0)
                    
            # Extract the importance of each specific token
            importance_all = []
          
            size_h = len(selected_attention)
            for c in range(0,size_h):
    
                importance_element = []
                importance_element.append(selected_attention[c,c])
                
                for v in range(0,size_h):
                    if v!=c:
                        importance_element.append(selected_attention[c,v])
                        importance_element.append(selected_attention[v,c])
            
                importance_all.append(importance_element)
     
            importance_tokens = [np.mean(l) for l in importance_all]
        
        elif layer_option == 'all' and computation_strategy == 'B': 
            
            #concatenate all layers
            selected_attention = alignments_array.sum(axis=0)
            
            # Extract the importance of each specific token
            importance_all = []
          
            size_h = len(selected_attention)
            for c in range(0,size_h):
    
                importance_element = []
                importance_element.append(selected_attention[c,c])
                
                for v in range(0,size_h):
                    if v!=c:
                        element = (selected_attention[c,v] + selected_attention[v,c])/2
                        importance_element.append(element)
                        
                importance_all.append(importance_element)
     
            importance_tokens = [np.mean(l) for l in importance_all]
     
                  
        if plot_attention_scores:
            scores = Utils.apply_activation(importance_tokens,'softmax') 
            # print(scores)
            
            # Sort keeping indexes
            sorted_idxs = np.argsort(-scores)
            # print(sorted_idxs)
            
            # Identify most important tokens
            number_tokens = int((top_tokens_rate)*token_len)
            # print(number_tokens)
            
            # Define the attention score threshold above which the important tokens are considered
            threshold = scores[sorted_idxs[number_tokens]]
        
            # Plot the important tokens        
            plt.figure(figsize=(15,7))
            plt.axhline(y = threshold, color = 'r', linestyle = '-')
            plt.plot(scores,linestyle='dashed')
            ax = plt.gca()
            ax.set_xticks(range(token_len))
            # ax.set_xticklabels(sequence_in)
            plt.xlabel('Sequence')
            plt.ylabel('Attention weights')
            plt.show()
        
        if plot_attention_weights:
            
            size_seq = len(selected_attention)   
            fig = plt.figure(figsize=(20,15))
            ax = plt.gca()
            fontdict = {'fontsize': 15}
            im = ax.matshow(selected_attention)
            ax.set_xticks(range(size_seq))
            ax.set_yticks(range(size_seq))
           
            labels_tokens = [token for token in tokens]
            try:
                ax.set_xticklabels(
                labels_tokens, fontdict=fontdict,rotation=90)
            except:
                print('size mismatch')
            ax.set_yticklabels(labels_tokens,fontdict=fontdict)
            fig.colorbar(im, fraction=0.046, pad=0.04)
            
            
        
        
        
        if selection_strategy == 'max':
            importance_tokens = Utils.apply_activation(importance_tokens,activation)
            indexes = np.argsort(importance_tokens)[-n_mask:]
            
        elif selection_strategy == 'min':
            importance_tokens = Utils.apply_activation(importance_tokens,activation)
            indexes = np.argsort(importance_tokens)[:n_mask]
        
        elif selection_strategy == 'prop':
            importance_tokens = Utils.apply_activation(importance_tokens,'softmax')
            indexes = np.random.choice(len(importance_tokens), 5, False, importance_tokens)
        
        return indexes
    
    
    
    
   