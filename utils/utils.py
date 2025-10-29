# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 11:49:12 2021

@author: tiago
"""

# External
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.IFG import ifg
import matplotlib.pyplot as plt

class Utils:
    """Data Loader class"""
    
    def __init__(self):
        """ Definition of the Proteins vocabulary """
                
        self.vocabulary_aminoacids = ['<PAD>','A','B','C','D','E','F','G','H',
                                      'I','L','M','N','O','K','P','Q','R','S',
                                      'T','U','W','Y','V','X','Z','<GO>','<END>']
        
        self.vocabulary_aminoacids_encoder = ['<PAD>','A','B','C','D','E','F','G','H',
                                      'I','L','M','N','O','K','P','Q','R','S',
                                      'T','U','W','Y','V','X','Z','<CLS>','<SEP>','<MASK>']
        
    @staticmethod 
    def aminoacid_dict(tokens):
        """ Computes the dictionary that makes the correspondence between 
        each token and a given integer.

        Args
        ----------
            tokens (list): List of each possible amino acid symbol

        Returns
        -------
            tokenDict (dict): Dictionary mapping characters into integers
        """

        tokenDict = dict((token, i) for i, token in enumerate(tokens))
        return tokenDict
        

    @staticmethod            
    def proteins2idx(sequences,tokenDict):
        """ Transforms each token in the proteins into the respective integer.

        Args
        ----------
            sequences (list): Protein strings
            tokenDict (dict): Dictionary mapping amino acids to integers 

        Returns
        -------
            new_sequences (list): List of transformed sequences, with the characters 
                              replaced by the numbers. 
        """   
        
        new_sequences =  np.zeros((len(sequences), len(sequences[0])))
        for i in range(0,len(sequences)):
            # print(i, ": ", smiles[i])
            for j in range(0,len(sequences[i])):
                
                try:
                    new_sequences[i,j] = tokenDict[sequences[i][j]]
                except:
                    value = tokenDict[sequences[i][j]]
        return new_sequences
        
    def tokenize_and_pad_encoder(self,FLAGS,sequences,token_table):
        """ Filters the proteins by their size, transforms protein strings into
            lists of tokens and padds the sequences until all sequences have 
            the same size.

        Args
        ----------
            FLAGS (argparse): Implementation parameters
            sequences (list): Protein strings with different sizes
            token_table (list): List of each possible amino acid
            padd (bol): Indicates if sequences should be padded 

        Returns
        -------
            tokenized (list): List of proteins with individualized tokens. The 
                              sequences are filtered by length, i.e., if it is
                              higher than the defined threshold, the protein
                              is discarded.
        """           

        filtered_sequences = [item for item in sequences if len(item)<=FLAGS.max_strlen_pt-2]
        
 
        tokenized = []
        
        for idx,protein in enumerate(filtered_sequences):
            
            protein = '<CLS>' + protein + '<SEP>'
            
            N = len(protein)
            i = 0
            j= 0
            tokens = []
            # print(idx,protein)
            while (i < N):
                for j in range(len(token_table)):
                    symbol = token_table[j]
                    if symbol == protein[i:i + len(symbol)]:
                        tokens.append(symbol)
                        i += len(symbol)
                        break
            tokenized.append(tokens)
            
            while len(tokens) < FLAGS.max_strlen_pt:
                tokens.append(token_table[0])
                    
        return tokenized,filtered_sequences
    
                 
    def tokenize_and_pad(self,FLAGS,sequences,token_table,padd=True,initial_token = True):
        """ Filters the proteins by their size, transforms protein strings into
            lists of tokens and padds the sequences until all sequences have 
            the same size.

        Args
        ----------
            FLAGS (argparse): Implementation parameters
            sequences (list): Protein strings with different sizes
            token_table (list): List of each possible amino acid
            padd (bol): Indicates if sequences should be padded 

        Returns
        -------
            tokenized (list): List of proteins with individualized tokens. The 
                              sequences are filtered by length, i.e., if it is
                              higher than the defined threshold, the protein
                              is discarded.
        """           

        filtered_sequences = [item for item in sequences if len(item)<=FLAGS.max_strlen_pt-2]
        
 
        tokenized = []
        
        for idx,protein in enumerate(filtered_sequences):
            if initial_token == True:
                protein = '<GO>' + protein + '<END>'
            else:
                protein = protein + '<END>'
                
            N = len(protein)
            i = 0
            j= 0
            tokens = []
            print(idx,protein)
            while (i < N):
                for j in range(len(token_table)):
                    symbol = token_table[j]
                    if symbol == protein[i:i + len(symbol)]:
                        tokens.append(symbol)
                        i += len(symbol)
                        break
            tokenized.append(tokens)
            if padd == True:
                while len(tokens) < FLAGS.max_strlen_pt:
                    tokens.append(token_table[0])
                    
        return tokenized,filtered_sequences

    @staticmethod             
    def idx2smi(model_output,tokenDict):
        """ Transforms model's predictions into SMILES

        Args
        ----------
            model_output (array): List with the autoencoder's predictions 
            tokenDict (dict): Dictionary mapping characters into integers

        Returns
        -------
            reconstructed_smiles (array): List with the reconstructed SMILES 
                                          obtained by transforming indexes into
                                          tokens. 
        """           

        key_list = list(tokenDict.keys())
        val_list = list(tokenDict.values())

        reconstructed_smiles =  []
        for i in range(0,len(model_output)):
            smi = []
            for j in range(0,len(model_output[i])):
                
                smi.append(key_list[val_list.index(model_output[i][j])])
                
            reconstructed_smiles.append(smi)
                
        return reconstructed_smiles
                
                
    @staticmethod 
    def external_diversity(set_A,set_B):
        """ Computes the Tanimoto external diversity between two sets
        of molecules

        Args
        ----------
            set_A (list): Set of molecules in the form of SMILES notation
            set_B (list): Set of molecules in the form of SMILES notation


        Returns
        -------
            td (float): Outputs a number between 0 and 1 indicating the Tanimoto
                        distance.
        """

        td = 0
        set_A = [set_A]
        fps_A = []
        for i, row in enumerate(set_A):
            try:
                mol = Chem.MolFromSmiles(row)
                fps_A.append(AllChem.GetMorganFingerprint(mol, 3))
            except:
                print('ERROR: Invalid SMILES!')
        if set_B == None:
            for ii in range(len(fps_A)):
                for xx in range(len(fps_A)):
                    ts = 1 - DataStructs.TanimotoSimilarity(fps_A[ii], fps_A[xx])
                    td += ts          
          
            td = td/len(fps_A)**2
        else:
            fps_B = []
            for j, row in enumerate(set_B):
                try:
                    mol = Chem.MolFromSmiles(row)
                    fps_B.append(AllChem.GetMorganFingerprint(mol, 3))
                except:
                    print('ERROR: Invalid SMILES!') 
            
            
            for jj in range(len(fps_A)):
                for xx in range(len(fps_B)):
                    ts = 1 - DataStructs.TanimotoSimilarity(fps_A[jj], fps_B[xx]) 
                    td += ts
            
            td = td / (len(fps_A)*len(fps_B))
        print("Tanimoto distance: " + str(td))  
        return td               

    def positional_encoding(pos, model_size):
        """ Compute positional encoding for a particular position
    
        Args:
            pos: position of a token in the sequence
            model_size: depth size of the model
        
        Returns:
            The positional encoding for the given token
        """
        PE = np.zeros((1, model_size))
        for i in range(model_size):
            if i % 2 == 0:
                PE[:, i] = np.sin(pos / 10000 ** (i / model_size))
            else:
                PE[:, i] = np.cos(pos / 10000 ** ((i - 1) / model_size))
        return PE

    def compute_dict_atom_token(self,smi):
         """ This function computes the dictionary that maps atom indexes of a
             molecule to its token indexes 
         -------
         Args:
         - smi (str): iput molecule
     
         -------
         Returns:
         - d (dictionary): Mapping between atom indexes and sequence indexes
     
         """
         d = {}
         aux_tokens = ['<Start>','<Padd>','(', ')', '[', ']', '=', '#', '@', '*', '%','0', '1',
                           '2','3', '4', '5', '6', '7', '8', '9', '.', '/','\\',
                           '+', '-']

         gap = 0 
         for i in range(0,len(smi[0])):
             symbol = smi[0][i]
             if symbol not in aux_tokens: # if it is an atom
                 d[i - gap] = i
             else:
                 gap +=1 
         return d
     
        
    def analyze_aa(sequences,possible_aminoacids):
        """ 
        
        """
        
        d = dict()
        
        for a in possible_aminoacids:
            if len(a)==1:
                d[a] = 0
        
        for protein in sequences:
            N = len(protein)
            i = 0
            j= 0
            tokens = []
            while (i < N):
                for j in range(len(possible_aminoacids)):
                    symbol = possible_aminoacids[j]
                    if symbol == protein[i:i + len(symbol)]:
                        tokens.append(symbol)
                        i += len(symbol)
                        if symbol in d.keys():
                            d[symbol] += 1
                        else:
                            d[symbol] = 1
                        break
        return d
   
            
            
    
    def int_to_prot(sampled_sequences,dict_aa):
        """ 
        
        """
        proteins = []        
        for seq in sampled_sequences:
            protein = ''
            for aa in seq:
                protein = protein + dict_aa[aa]
            proteins.append(protein)
            
        return proteins
   
    
    def plot_training_progress(training_rewards,losses_generator,rewards_ama,rewards_tox,rewards_len): #predicted_ama,predicted_tox,
        """ Plots the evolution of the rewards and loss throughout the 
        training process.
        Args
        ----------
            training_rewards (list): List of the combined rewards for each 
                                     sampled batch of molecules;
            losses_generator (list): List of the computed losses throughout the 
                                     training process;
            rewards_ama (list): List of the rewards for the antimicrobial activity
                                for each sampled batch of molecules;
            rewards_tox (list): List of the rewards for the toxicity for each 
                                sampled batch of molecules;
            predicted_ama (list): List of the average antimicrobial activity for
                                each sampled batch of molecules;
            predicted_tox (list): List of the average toxicity for each sampled
                                batch of molecules;

        Returns
        -------
            Plot
        """
        # print(training_rewards)
        plt.plot(training_rewards)
        plt.xlabel('Training iterations')
        plt.ylabel('Average rewards')
        plt.show()
        
        # plt.plot(predicted_ama)
        # plt.xlabel('Training iterations')
        # plt.ylabel('Average Antimicrobial activity')
        # plt.show()

        # plt.plot(predicted_tox)
        # plt.xlabel('Training iterations')
        # plt.ylabel('Average Toxicity')
        # plt.show()
        
        plt.plot(losses_generator)
        plt.xlabel('Training iterations')
        plt.ylabel('Average losses PGA')
        plt.show()
        
        plt.plot(rewards_ama)
        plt.xlabel('Training iterations')
        plt.ylabel('Average rewards antimicrobial activity')
        plt.show()
        
        plt.plot(rewards_tox)
        plt.xlabel('Training iterations')
        plt.ylabel('Average rewards toxicity')
        plt.show()
                   
        plt.plot(rewards_len)
        plt.xlabel('Training iterations')
        plt.ylabel('Average rewards length')
        plt.show()
        
    def moving_average(previous_values, new_value, ma_window_size=10): 
        """
        This function performs a simple moving average between the previous 9 and the
        last one reward value obtained.
        ----------
        previous_values: list with previous values 
        new_value: new value to append, to compute the average with the last ten 
                   elements
        
        Returns
        -------
        Outputs the average of the last 10 elements 
        """
        value_ma = np.sum(previous_values[-(ma_window_size-1):]) + new_value
        value_ma = value_ma/(len(previous_values[-(ma_window_size-1):]) + 1)
        return value_ma
    
    
    def apply_activation(x,activation):
        """ Apply activation for the vector x of importance tokens
    
        Args:
            x (array): vector of floating values
        
        Returns:
            Vector of the same size of x with its transformed values
        """       
        x = np.array(x)
        if activation=='softmax':
        # Softmax + temperature
            exp_x = np.exp(x /0.4)
            x_out = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        
        elif activation=='tanh':
            # Hyperbolic tangent
            x_out = np.tanh(x)
            
        elif activation=='sigmoid':
             # Sigmoid
             x_out = 1 / (1 + np.exp(-x))
        elif activation=='none':
            x_out = x
            
        return x_out
    
        
  
