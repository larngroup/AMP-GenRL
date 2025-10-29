# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 11:48:26 2021

@author: tiago
"""

# Internal
from utils.utils import Utils

# External
from rdkit import Chem
from sklearn.model_selection import train_test_split
import tensorflow as tf
from Bio import SeqIO
import random
import numpy as np
import matplotlib.pyplot as plt

class DataLoader:
    """Data Loader class"""

    @staticmethod
    def load_sequences(FLAGS,training_procedure ='pre_training'):
        """ Loads the protein sequences
        Args:
           FLAGS (argparse): Implementation parameters

        Returns:
            dataset (list): The list with the training and testing proteins 
        """
        
        if training_procedure == 'pre_training':
            # paths = ['data/example_proteins.fasta']
            paths = ['data/pdb_seqres.fasta','data/uniprot_sprot.fasta']
            max_strlen = FLAGS.max_strlen_pt
        elif training_procedure == 'finetuning':
            # paths = ['data/example_proteins.fasta']
            paths = ['data/amps/APD.fasta','data/amps/DADP.fasta','data/amps/DBAASP_peptides.fasta','data/amps/enzy2.fasta','data/amps/LAMP2.fasta']
            max_strlen = FLAGS.max_strlen_ft
            
        sequences = []
        for path in paths:
            with open(path, mode='r') as handle:
    
                # Use Biopython's parse function to process individual
                # FASTA records (thus reducing memory footprint)
                for record in SeqIO.parse(handle, 'fasta'):
            
                    # Extract individual parts of the FASTA record
                    identifier = record.id
                    description = record.description
                    sequence = record.seq
            
                    # print('----------------------------------------------------------')
                    # print('Processing the record: {}'.format(identifier))
                    # print('Its description is: \n{}'.format(description))
                    # amount_of_nucleotides = len(sequence)
                    # print('Its sequence contains {} nucleotides.'.format(amount_of_nucleotides))
                    sequence_str = str(sequence._data)[2:-1].upper()
                    if len(sequence_str)<max_strlen-2 and  len(sequence_str)>5 and 'U' not in sequence_str and 'X' not in sequence_str and 'Z' not in sequence_str and 'B' not in sequence_str and '-' not in sequence_str:
                        sequences.append(sequence_str)
                  
                
        print('\nAll AMP sequences: ',len(list(set(sequences))))

        unique_sequences = list(set(sequences))#[:FLAGS.subset_instances] #20000
        
        # unique_sequences = list(set(sequences))
        # lengs = [len(a) for a in unique_sequences]
        # indexes = np.arange(len(lengs))
        # width = 1
        # plt.bar(lengs, indexes,width)
        # plt.show()
      
        return unique_sequences
    
    def pre_process_data(dataset,transf_obj,FLAGS):
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
        
        # split the data in training and testing sets
        train_sequences, test_sequences = train_test_split(dataset, test_size=FLAGS.test_rate, random_state=55)
        train_sequences = train_sequences
        
        # # # Extract randomly a subset of proteins from the training set to compare with generated sequences
        # subset_idxs = random.sample(range(0, len(dataset)-1),100)
        # pretrain_subset = [seq for idx,seq in enumerate(dataset) if idx in subset_idxs]
        
        # # file = open('sample_pre_training.txt','w')
        # file = open('sample_fine_tuning.txt','w')
        
        # for item in pretrain_subset:
        # 	file.write(item+"\n")
        # file.close()
      
        
       	# Tokenize - transform the protein into lists of tokens 
        tokens,proteins_filtered = Utils().tokenize_and_pad(FLAGS,train_sequences,transf_obj.token_table)   
 
        # Transforms each token to the respective integer, according to 
        # the previously computed dictionary
        input_train = Utils.proteins2idx(tokens,transf_obj.tokenDict)
        
        # Tokenize  &  Character to integer - training target
        tokens_target,_ = Utils().tokenize_and_pad(FLAGS,train_sequences,transf_obj.token_table,initial_token=False)   
        input_train_target = Utils.proteins2idx(tokens_target,transf_obj.tokenDict)
            
        # Create tf.dataset object
        data_train = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(input_train),tf.convert_to_tensor(input_train_target))).batch(FLAGS.batchsize, drop_remainder = False)
        
        
    	# Tokenize - transform the protein strings into lists of tokens 
        tokens_test,proteins_filtered_test = Utils().tokenize_and_pad(FLAGS,test_sequences,transf_obj.token_table)   

        # Transforms each token to the respective integer, according to 
        # the previously computed dictionary
        input_test = Utils().proteins2idx(tokens_test,transf_obj.tokenDict)
        
        # Tokenize  &  Character to integer - testing target
        tokens_test_target,proteins_filtered_test = Utils().tokenize_and_pad(FLAGS,test_sequences,transf_obj.token_table,initial_token=False)   
        input_test_target = Utils().proteins2idx(tokens_test_target,transf_obj.tokenDict)
                
        data_test = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(input_test),tf.convert_to_tensor(input_test_target))).batch(FLAGS.batchsize, drop_remainder = False)
       
        
        print('\nNumber of training sequences: ',len(train_sequences))
        print('\nNumber of testing sequences: ',len(test_sequences))
        return data_train,data_test,train_sequences
    
    
    def pre_process_data_buffer(train_sequences,transf_obj,FLAGS):
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
        tokens,proteins_filtered = Utils().tokenize_and_pad(FLAGS,sequences_no_spaces,transf_obj.token_table)   
 
        # Transforms each token to the respective integer, according to 
        # the previously computed dictionary
        input_train = Utils.proteins2idx(tokens,transf_obj.tokenDict)
        
        # Tokenize  &  Character to integer - training target
        tokens_target,_ = Utils().tokenize_and_pad(FLAGS,sequences_no_spaces,transf_obj.token_table,initial_token=False)   
        input_train_target = Utils.proteins2idx(tokens_target,transf_obj.tokenDict)
            
        # Create tf.dataset object
        data_train = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(input_train),tf.convert_to_tensor(input_train_target))).batch(FLAGS.batchsize, drop_remainder = False)
        
        
        print('\nNumber of sequences: ',len(tokens))    
        
        
        return data_train