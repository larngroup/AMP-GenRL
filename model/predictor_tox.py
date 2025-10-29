# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 16:49:39 2021

@author: tiago
"""

# internal
from .base_model import BaseModel
from utils.utils import Utils
from model.encoder import Encoder
from model.transformers import Masked_Smiles_Model
from model.toxicity_model import Toxicity_model

# external
import tensorflow as tf
# import tensorflow_addons as tfa
import numpy as np
import time
import random


class Predictor_tox(BaseModel):
    """Toxicity predictor general Class"""
    def __init__(self, FLAGS):
        super().__init__(FLAGS)
        
        # Implementation parameters
        self.FLAGS = FLAGS
        
        # Load the table of possible tokens
        self.token_table = Utils().vocabulary_aminoacids_encoder 
        self.vocab_size = len(self.token_table)
        
        # Dictionary that makes the correspondence between each token and unique integers
        self.tokenDict = Utils.aminoacid_dict(self.token_table)
        self.inv_tokenDict = {v: k for k, v in self.tokenDict.items()}
        
        # Positional encoding
        self.max_length = self.FLAGS.max_strlen
        self.model_size_best = 256
        self.pes = []
        for i in range(self.max_length):
            self.pes.append(Utils.positional_encoding(i, self.model_size_best))
        self.pes = np.concatenate(self.pes, axis=0)
        self.pes = tf.constant(self.pes, dtype=tf.float32)
        
        self.build_models()

    
    def masking(self, inp):
        """ Performs the masking of the input smiles. 20% of the tokens are 
            masked

        Args
        ----------
            inp (list): List of input sequences
           
            
        Returns
        -------
            x (list): Peptides with masked tokens
            masked_positions (list): Sequences of the same length of the peptides
                                     with 1's on the masked positions
        """
        masked_positions = []
        x = [i.copy() for i in inp]
        
        for smile_index in range(len(x)):

            possible_idxs = [indx for indx in range(len(x[smile_index])) if (x[smile_index][indx] not in [self.token_table.index('<CLS>'),
			self.token_table.index('<PAD>'), self.token_table.index('<SEP>'), self.token_table.index('<MASK>') ])] 
            
           
            p_mask = 0.2
                
            num_mask = max(3, int(round(len(possible_idxs) * p_mask)))  
           
            shuffle_m = random.sample(range(len(possible_idxs)), len(possible_idxs)) 
            mask_index = shuffle_m[:num_mask]
            # get the selected indeces only from the possible set of maskable tokens
            selected_idxs = [possible_idxs[idx] for idx in mask_index]
            
            masked_pos =[0]*len(x[smile_index])
			
            for pos in selected_idxs:

                masked_pos[pos] = 1
                rd = random.random()
                if rd <= 0.8: 
                    # print('first')
                    x[smile_index][pos] = self.token_table.index('<MASK>')
                elif rd > 0.8: 
                    # print('second')
                    index = random.randint(1, self.token_table.index('<CLS>')-1) 
                    x[smile_index][pos] = index
            masked_positions.append(masked_pos) 
            
        
        return x, masked_positions
   
    
    def build_models(self):
        
        """Builds and trains the Transformer-encoder and the toxicity 
        architecture"""
        
        
        # Loads the Transformer architecture
        
        
        # self.encoder = Encoder(self.vocab_size, d_model, n_layers, n_heads,
        #                         dropout, activation_func,ff_dim,self.pes)
        # sequence_in = tf.constant([[1, 2, 3, 0, 0]])
        
        d_model = 256
        ff_dim = 1024
        n_heads = 4
        n_layers = 4
        activation_func = 'relu'
        
        self.encoder = Masked_Smiles_Model(d_model,ff_dim,n_heads,n_layers,self.max_length,self.vocab_size,activation_func)   
        sequence_in = tf.constant(np.zeros((1,self.max_length)))
        encoder_output, _,_,_ = self.encoder(sequence_in)
        print(encoder_output.shape)
        path = 'transf_encoder.h5'
        self.encoder.load_weights(path)
            
        
        
        # Loads the Predictor architecture
        self.predictor = Toxicity_model()   
        sequence_in = tf.constant([list(np.ones((1,256)))])
        pred = self.predictor(sequence_in)
        print(pred.shape)
        path = 'predictor_model.h5'
        self.predictor.load_weights(path)
        
        
        
        
        
    def predict_tox(self,sampled_sequences):
        """Predicts resuts for the test dataset"""
        print('\nPredicting on test set...') 
        
        # computes the predictor descriptor
        print('padding...')
        peptides_processed = Utils.tokenize_and_pad(self.FLAGS,sampled_sequences,self.token_table)
        print('padding done')
                       
        # Transforms each token to the respective integer, according to 
        # the previously computed dictionary
        seq_idxs_input = Utils().proteins2idx(peptides_processed,self.tokenDict)
        
        # masking mlm
        input_masked, masked_positions = self.masking(seq_idxs_input)  
        
        print(input_masked[0].shape)
        
        #TODO Create Dataset object
        encoder_output, _,_,_= self.encoder(tf.constant(input_masked), training=False)
        
        # Extract just the [CLS] token vector (batch,256)
        descriptor_cls = encoder_output.numpy()[:,0,:]
        print(descriptor_cls.shape)
          
        pred = self.predictor(descriptor_cls, training=False)
 
        # for num, (smiles,smiles_masked,fgs_test) in enumerate(processed_dataset_test):
               
        return pred
