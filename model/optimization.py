#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 11:07:19 2024

@author: maryam
"""

from utils.utils import Utils
from model.transformers import Transformer_Decoder
from model.transformer import Transformer
from dataloader.dataloader import DataLoader
from model.peptide_evaluation import Peptide_evaluation
from model.rb_manager import RB_manager

import tensorflow as tf
import numpy as np
import time
from random import seed
import random
from modlamp.analysis import GlobalAnalysis
from modlamp.core import count_aas
from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor
from modlamp.sequences import Random, Helices
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from collections import deque

tf.config.run_functions_eagerly(True)
class Optimization():
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
        
        # Loading of the fine-tuned model to compare
        self.decoder = Transformer_Decoder(256,1024,4,4,200,self.vocab_size,'relu')   
        sequence_in = tf.ones([1, 1])
        decoder_output,_,_ = self.decoder(sequence_in)
        path = 'model_finetuning.h5'
        self.decoder.load_weights(path)
        
        # Loading of the fine-tuned model to optimize
        self.decoder_ft = Transformer_Decoder(256,1024,4,4,200,self.vocab_size,'relu')   
        sequence_in = tf.ones([1, 1])
        decoder_output,_,_ = self.decoder_ft(sequence_in)
        path = 'model_finetuning.h5'
        self.decoder_ft.load_weights(path)
        
        # Initialize the Transformer class
        self.transformer_class = Transformer(FLAGS)
       
        # Initialization of the peptide evaluation class
        self.peptide_evaluation = Peptide_evaluation(self.FLAGS)

        # Initialization of the replay buffer manager class
        self.rb_manager = RB_manager(self.FLAGS,self.transformer_class)        
        self.replay_buffer = [] 
        
        self.filter_best = False
        
        # seed random number generator
        seed(1)
        
    
    def loss_function(self,actions,logits_all,mask_states,rewards):
        """ Calculates the loss (sparse categorical crossentropy) for the 
        predicted probabilities of the true tokens. The mask removes the padding
        tokens from this analysis. 
        

        Args
        ----------
            y_true (array): true label
            y_pred (array): model's predictions
            mask (array): mask of 1's and 0's (padding tokens)
            
        Returns
        -------
           loss
        """
        # print(logits_all.shape)
        # # print(logits_all[0,:,:])
        # # print(actions.shape)
        # print(actions)
        # print(mask_states[0,:,:])
        # print('olha')
        # print(mask_states[1,:,:])
        # print('va')
        # print(mask_states[2,:,:])
        # print('final')
        # print(mask_states[-1,:,:])
        # print(rewards.shape)
        
        # shape logits (None, 200, 28)
        mask_states = tf.cast(mask_states, dtype='float32')
        rewards = tf.cast(rewards, dtype='float32')
        
        masked_tensor = logits_all*mask_states 
        # print(masked_tensor[0,:,:])
        # print(masked_tensor[1,:,:])
        # print(masked_tensor[2,:,:])
        # print('finally')
        # print(masked_tensor[logits_all.shape[2],:,:])
        # Sum along the second dimension (200) while keeping the shape (None, 28)
        logits_state = tf.reduce_sum(masked_tensor, axis=1)
        # print(logits_state)

        # Remove to get (None,23)
        # logits_state = tf.squeeze(sum_along_dimension, axis=1)

        mask_actions = tf.one_hot(actions, logits_all.shape[2])
        # print('f')
        # print(mask_actions)
        
        log_probs = tf.reduce_sum(mask_actions * tf.math.log(logits_state), axis=1)
        # print(log_probs)
        
        loss = -tf.reduce_mean(log_probs * rewards)
        
        return loss
    
    @tf.function
    def train_step(self, states, actions, mask_states, rewards): 
        """
        Training step: compute model's prediction, calculates the loss, 
        calculates the gradients and update the weights accordingly
        Parameters
        ----------
        x : batch of input sequences
        target : batch of actual predictions

        Returns
        -------
        loss value

        """
 
        with tf.GradientTape() as tape:
        
            logits,_,_ = self.decoder(states,training=False)
            
            loss = self.loss_function(actions,logits,mask_states,rewards) 
        
        gradients = tape.gradient(loss, self.decoder.trainable_variables) 
        self.optimizer.apply_gradients(zip(gradients, self.decoder.trainable_variables)) 

        return loss 
    

    def sample_token(self,predictions, temperature=1.01):
        predictions = np.log(predictions) / temperature
        exp_predictions = np.exp(predictions)
        predictions = exp_predictions / np.sum(exp_predictions)
        sampled_token = np.random.choice(len(predictions), 1, p=predictions)[0]
        return sampled_token    
    
    def sample_rb(self):
        # print('\nRB: ', self.replay_buffer)
        # TODO generate a random index with the size of replay buffer
        rd_int = random.randint(0, len(self.replay_buffer)-1)
        # print('selected int: ', rd_int)
        # select a random peptide from the RB
        input_seq = self.replay_buffer[rd_int]
        # print('selected seq: ', input_seq)

        
        generated_all = []
        
        sequence = np.zeros([1,self.FLAGS.max_strlen_pt])
        sequence[0,0] = self.tokenDict['<GO>']
        sampled_token = self.tokenDict['<GO>']
        sampled_sequence = []
        states_peptide = sequence.copy()
        actions_peptide = []
        # actions_peptide.append(sampled_token)
        mask_state = np.zeros([1,200,28])
        mask_state[0,len(sampled_sequence),:] = 1
        mask_states_all = mask_state.copy()
        for i in range(0,len(input_seq)+1):
            
            mask_state = np.zeros([1,200,28])
            mask_state[0,len(sampled_sequence)+1,:] = 1
            
            if i < len(input_seq):
                sampled_token = self.tokenDict[input_seq[i]]
            else:
                sampled_token = self.tokenDict['<END>']
        
            
            # Append the predicted token to the generated sequence
            if sampled_token not in [self.tokenDict['<END>'],self.tokenDict['<PAD>']]:
                sampled_sequence.append(sampled_token)
            
                # Update the input sequence for the next iteration
                sequence[0,len(sampled_sequence)] = sampled_token
                
                states_peptide  = np.concatenate([states_peptide,sequence])
                actions_peptide.append(sampled_token)
                
                mask_states_all = np.concatenate([mask_states_all,mask_state])
                
            elif sampled_token == self.tokenDict['<END>'] or sampled_token == self.tokenDict['<PAD>']: 
            
                actions_peptide.append(self.tokenDict['<END>'])
                
            if len(sampled_sequence) == self.FLAGS.max_strlen_pt-1:
                
                actions_peptide.append(self.tokenDict['<END>'])
                

        generated_all.append(sampled_sequence)
           
        actions_peptide_np = np.array(actions_peptide, dtype='int32')
        protein_sequences_all = Utils.int_to_prot(generated_all,self.inv_tokenDict) 
        peptides_separated = ' '.join(protein_sequences_all[0])
        # print('need: ')
        # print(generated_all,len(generated_all[0]))
        # print(peptides_separated,len(peptides_separated))
        # print(states_peptide,states_peptide.shape)
        # print(actions_peptide_np,actions_peptide_np.shape)
        # print(mask_states_all,mask_states_all.shape)
        # n
        return generated_all,peptides_separated,states_peptide,actions_peptide_np,mask_states_all
        
        
    def sample_rl(self):
        
        generated_all = []
        
        sequence = np.zeros([1,self.FLAGS.max_strlen_pt])
        sequence[0,0] = self.tokenDict['<GO>']
        sampled_token = self.tokenDict['<GO>']
        sampled_sequence = []
        states_peptide = sequence.copy()
        actions_peptide = []
        # actions_peptide.append(sampled_token)
        mask_state = np.zeros([1,200,28])
        mask_state[0,len(sampled_sequence),:] = 1
        mask_states_all = mask_state.copy()
        while sampled_token not in [self.tokenDict['<END>'],self.tokenDict['<PAD>']] and len(sampled_sequence) < self.FLAGS.max_strlen_pt-1:
            
            mask_state = np.zeros([1,200,28])
            mask_state[0,len(sampled_sequence)+1,:] = 1
            
            # generate a random number
            # epsilon = random.random()
            # if epsilon<0.05:
            #     predictions,_,mask=self.decoder_ft.predict(tf.constant(sequence))
            # else:
            predictions,_,mask=self.decoder.predict(tf.constant(sequence))
                
            # predictions,_,mask=self.decoder(tf.constant(sequence),training=False)
            # print(predictions.shape)
            probability_dist = predictions[0,len(sampled_sequence),:]
            # print(probability_dist)
            # sampled_token = np.random.choice(len(probability_dist), 1, p=probability_dist)[0]
            sampled_token = self.sample_token(probability_dist)
            
            # Append the predicted token to the generated sequence
            if sampled_token not in [self.tokenDict['<END>'],self.tokenDict['<PAD>']]:
                sampled_sequence.append(sampled_token)
            
                # Update the input sequence for the next iteration
                sequence[0,len(sampled_sequence)] = sampled_token
                
                states_peptide  = np.concatenate([states_peptide,sequence])
                actions_peptide.append(sampled_token)
                
                # if len(sampled_sequence)==1:
                #     mask_states_all = mask_state.copy()
                # else:
                mask_states_all = np.concatenate([mask_states_all,mask_state])
                
            elif sampled_token == self.tokenDict['<END>'] or sampled_token == self.tokenDict['<PAD>']: 
            
                actions_peptide.append(self.tokenDict['<END>'])
                
            if len(sampled_sequence) == self.FLAGS.max_strlen_pt-1:
                
                actions_peptide.append(self.tokenDict['<END>'])
                


        generated_all.append(sampled_sequence)
           
        actions_peptide_np = np.array(actions_peptide, dtype='int32')
        protein_sequences_all = Utils.int_to_prot(generated_all,self.inv_tokenDict) 
        peptides_separated = ' '.join(protein_sequences_all[0])
        return generated_all,peptides_separated,states_peptide,actions_peptide_np,mask_states_all
    
    def sample(self,model,n_generate):
        
        print('\nSampling new peptides...')

        generated_all = []
        for i in range(n_generate):
           
            sequence = np.zeros([1,self.FLAGS.max_strlen_pt])
            sequence[0,0] = self.tokenDict['<GO>']
            sampled_token = self.tokenDict['<GO>']
            sampled_sequence = []
            while sampled_token not in [self.tokenDict['<END>'],self.tokenDict['<PAD>']] and len(sampled_sequence) < self.FLAGS.max_strlen_pt-1:
            
               predictions,_,mask= model.predict(tf.constant(sequence))
               # predictions,_,mask= self.decoder(tf.constant(sequence),training=False)
               # predictions_np = predictions.numpy()
               probability_dist = predictions[0,len(sampled_sequence),:]
               sampled_token = self.sample_token(probability_dist)
               # sampled_token =   np.random.choice(len(probability_dist), 1, p=probability_dist)[0]
               
               
               # Append the predicted token to the generated sequence
               if sampled_token not in [self.tokenDict['<END>'],self.tokenDict['<PAD>']]:
                   sampled_sequence.append(sampled_token)
               
                   # Update the input sequence for the next iteration
                   sequence[0,len(sampled_sequence)] = sampled_token
                   # print('next')
               
            generated_all.append(sampled_sequence)
           
                   
        peptide_sequences_all = Utils.int_to_prot(generated_all,self.inv_tokenDict)  
        peptide_sequences_filtered = [s for s in peptide_sequences_all if len(s)>=5]
        peptides_separated = [' '.join(fs) for fs in peptide_sequences_filtered]
        return peptide_sequences_filtered,peptides_separated
    
    def loss_function_buffer(self,y_true,y_pred,mask):
        """ Calculates the loss (sparse categorical crossentropy) for the 
        predicted probabilities of the true tokens. The mask removes the padding
        tokens from this analysis. 
        

        Args
        ----------
            y_true (array): true label
            y_pred (array): model's predictions
            mask (array): mask of 1's and 0's (padding tokens)
            
        Returns
        -------
           loss
        """
        
        loss_ = self.crossentropy(y_true,y_pred)
        loss_ *= mask
        
        # Average the loss over non-padding tokens
        non_padding_tokens = tf.reduce_sum(mask)
        loss = tf.reduce_sum(loss_) / non_padding_tokens
    
        return loss
    
    @tf.function
    def train_step_replay_buffer(self, x, target): 
        """
        Training step: compute model's prediction, calculates the loss, 
        calculates the gradients and update the weights accordingly
        Parameters
        ----------
        x : batch of input sequences
        target : batch of actual predictions

        Returns
        -------
        loss value

        """
 
        with tf.GradientTape() as tape: 
            predictions,_,padding_mask = self.decoder(x)
            loss = self.loss_function_buffer(target, predictions,padding_mask) 
        
        gradients = tape.gradient(loss, self.decoder.trainable_variables) 
        self.optimizer_rb.apply_gradients(zip(gradients, self.decoder.trainable_variables)) 

        return loss 

    
    def train_replay_buffer(self):
        """Training with the best peptides to reinforce this sub-space"""
            
        self.crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, reduction='none')
            
        self.optimizer_rb = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9,                                               
                                                    beta_2=0.99, epsilon=1e-08)
        
        min_delta = self.FLAGS.min_delta    
        patience = self.FLAGS.patience  
        
        # if data_augmentation == True:
        data = self.rb_manager.augment_rb(self.replay_buffer)
        # data = self.rb_manager.pre_process_data_buffer(self.replay_buffer)
       
        
        n_epochs_buffer = 10
        last_loss = {'epoch':0,'value':1000} 
        for epoch in range(n_epochs_buffer): 
            print(f'Epoch {epoch+1}/{n_epochs_buffer}') 
            start = time.time() 
            loss_epoch = [] 
            for num, (x_train,y_train) in enumerate(data): 
                loss_batch = self.train_step_replay_buffer(x_train,y_train) 
                loss_epoch.append(loss_batch) 
                if num == len(data)-1: 
                    print(f'{num+1}/{len(data)} - {round(time.time() - start)}s - loss: {np.mean(loss_epoch):.4f}')    
					
            if (last_loss['value'] - tf.math.reduce_mean(loss_epoch)) >= min_delta: 
                last_loss['value'] = tf.math.reduce_mean(loss_epoch) 
                last_loss['epoch'] = epoch+1
                
                self.decoder.save_weights(self.FLAGS.checkpoint_path + 'model_optimized.h5')  
                
            if ((epoch+1) - last_loss['epoch']) >= patience: 
                break 
        
    
    def optimization_loop(self):
        print('\nInitialize optimization!')
        n_epochs = 2500
        batch_size = 8
        
        max_size = 300
        self.replay_buffer = deque(maxlen=max_size)
        
        total_loss = [] 
        total_rewards = []
        total_rewards_ama = []
        total_rewards_tox = []
        total_rewards_len = []
        ama_success = []
        tox_sucess = []
        length_success = []
        all_success = []        
        best_success = 0 
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9,                                               
                                                    beta_2=0.99, epsilon=1e-08,clipvalue=3)

        for ep in range(n_epochs):
            batch_generated_int = []
            batch_generated_pep = []
            print('\nStarting new_epoch: ', ep+1)
            b_idx = 0 
                
            while b_idx < batch_size:
                if b_idx == batch_size-1 and len(self.replay_buffer)==max_size:
                    sample_peptide_int,peptide,states_peptide,actions_peptide,mask_states = self.sample_rb()    
                else:
                    sample_peptide_int,peptide,states_peptide,actions_peptide,mask_states = self.sample_rl()
                    # print('need: ')
                    # print(sample_peptide_int,len(sample_peptide_int[0]))
                    # print(peptide,len(peptide))
                    # print(states_peptide,states_peptide.shape)
                    # print(actions_peptide,actions_peptide.shape)
                    # print(mask_states,mask_states.shape)
                    
                    
                print('\nSampled peptide: ', peptide, len(sample_peptide_int[0]))
                # print(sample_peptide_int)
                if len(sample_peptide_int[0])>9 and len(sample_peptide_int[0])<self.FLAGS.max_strlen_ft and self.tokenDict['<GO>'] not in sample_peptide_int[0]: # other fundamental evaluation
                    print('nice one!')
                    batch_generated_int.append(sample_peptide_int[0])
                    batch_generated_pep.append(peptide)
                    
                    if b_idx == 0:
                        batch_states = states_peptide.copy()
                        batch_actions = actions_peptide.copy()
                        batch_mask_states = mask_states.copy()
                    else:
                        batch_states = np.concatenate([batch_states,states_peptide])
                        batch_actions = np.concatenate([batch_actions,actions_peptide])
                        batch_mask_states = np.concatenate([batch_mask_states,mask_states]) 
                        
                    b_idx +=1 
                else:
                    print('wrong length!')
            
            
            # Update the replay buffer with more promising peptides and compute
            # the general reward for each peptide considering anti-microbial 
            # activity and toxicity
            batch_general_reward,rewards_combined_b,rewards_ama_b,rewards_tox_b,rewards_len_b,self.replay_buffer = self.peptide_evaluation.get_reward(batch_generated_pep,self.replay_buffer)
            
            print('\n LENGTHS: ')
            print(len(batch_states))
            print(len(batch_actions))
            print(len(batch_mask_states))
            print(len(batch_general_reward))
            
            data_train = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(batch_states),
                                                             tf.convert_to_tensor(batch_actions),
                                                             tf.convert_to_tensor(batch_mask_states),
                                                             tf.convert_to_tensor(batch_general_reward))).batch(16, drop_remainder = False)
        
            loss_mini_batch_all = 0
            start = time.time() 
            for num, (mini_batch_states,mini_batch_actions,mini_batch_mask_states,mini_batch_rewards) in enumerate(data_train): 
                loss_mini_batch = self.train_step(mini_batch_states, mini_batch_actions, mini_batch_mask_states, mini_batch_rewards)
                loss_mini_batch_all += loss_mini_batch.numpy() 
                if num == len(data_train)-1:
                    print(f'{num+1}/{len(data_train)} - {round(time.time() - start)}s - loss: {loss_mini_batch_all/len(data_train):.4f}')
                    
            print('Saving model...')
            self.decoder.save_weights(self.FLAGS.checkpoint_path + 'model_optimized.h5')  
            
            #Save batch information
            total_loss.append(Utils.moving_average(total_loss,loss_mini_batch_all/len(data_train)))
            total_rewards.append(Utils.moving_average(total_rewards,np.mean(rewards_combined_b)))
            total_rewards_ama.append(Utils.moving_average(total_rewards_ama,np.mean(rewards_ama_b)))
            total_rewards_tox.append(Utils.moving_average(total_rewards_tox,np.mean(rewards_tox_b)))
            total_rewards_len.append(Utils.moving_average(total_rewards_len,np.mean(rewards_len_b)))
            
            if ep%5==0:
                Utils.plot_training_progress(total_rewards,total_loss,total_rewards_ama,total_rewards_tox,total_rewards_len)
            
            if ep%49 == 0 and len(self.replay_buffer)>10: # normal rp: 50, rpa: 49, rph:50 , rpl:48  rpp: 49
                print('\n')
                self.train_replay_buffer()
            
            if ep%10==0 and ep>250:
                _,generated_optimized_sep = self.sample(self.decoder,100)
                ama_success,tox_sucess,length_success,all_success = self.peptide_evaluation.evaluate_objectives(generated_optimized_sep,ama_success,tox_sucess,length_success,all_success)                
                if all_success[-1]>best_success:
                    self.decoder.save_weights(self.FLAGS.checkpoint_path + 'best_model.h5')
                    best_success = all_success[-1]
                    
        self.compare_progression()
        print('\nBest success all: ', )
        
            
        
    def save_best_peptides(self):
        model = 'best_rp_r'
        path = 'checkpoints/'+model+'/model_optimized.h5'
        self.decoder.load_weights(path)
        print('\nLoaded successfuly!')

        
        properties_sampled = pd.DataFrame()
        generated,generated_space = self.sample(self.decoder,10000)
        # df = pd.read_csv('generated/peptides_best_standard.csv')
        # generated_optimized_sepp = list(df['SEQ'])
        # # print(generated_optimized_sepp[0:5],len(generated_optimized_sepp))
        # generated_un = list(set(generated_optimized_sepp))
        # print(generated_un[0:5],len(generated_un))
        # generated_space_un = [' '.join(s) for s in generated_un]
        # print(generated_space_un[0:5],len(generated_space_un))
     
        # unique sampled
        generated_un = list(set(generated))
        generated_space_un = [' '.join(s) for s in generated_un]

        
        rate_unrepeated = len(generated_un)/len(generated)
        print('\n% Unique peptides: ', rate_unrepeated*100)
        
        mic_sampled,tox_sampled = self.peptide_evaluation.evaluate_MIC_TOX(generated_space_un)
        properties_sampled['SEQ'] = generated_un
        properties_sampled['MIC'] = mic_sampled
        properties_sampled['TOX'] = tox_sampled
        lengths = [len(pep) for pep in generated_un]
        properties_sampled['LEN'] = lengths
        
        
                  
        # Plot each feature separately
        for column in properties_sampled.columns:
            if column!='SEQ':
                plt.figure(figsize=(8, 6))  # Set the size of the plot
                plt.hist(properties_sampled[column], bins=20, color='skyblue', edgecolor='black')  # Plot the histogram
                
                # Calculate the mean value
                mean_value = properties_sampled[column].mean()
                
                # Mark the mean value on the plot
                plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean = {mean_value:.2f}')
                
                plt.title(f'Distribution of {column}')  # Set the title
                plt.xlabel(column)  # Set the label for x-axis
                plt.ylabel('Frequency')  # Set the label for y-axis
                plt.legend()  # Show legend
                # plt.grid(True)  # Add gridlines
                # TODO add limits to axis MIC
                plt.show()  # Show the plot
                #

        print('\nFiltering...')
        print(len(properties_sampled['SEQ']))
        filtered_peptides = properties_sampled[(properties_sampled['TOX']>100) & (properties_sampled['MIC']<25) & (properties_sampled['LEN'] < 50) ]
        print(len(filtered_peptides['SEQ']))
        
        filtered_peptides.to_csv('generated/peptides_'+model+'_all.csv',index=False)
        print('\nSaved!')
        
        
    def combine_all_peptides(self):
        # Loads all generated sets of peptides, combines all in a single df and
        # filters the best ones to a file, to send to Kai
        
        path_generated = 'generated'
        

        all_files = os.listdir(path_generated)
        csv_files = [file for file in all_files if file.endswith('.csv')]
    
        dfs = []
        for file in csv_files:
            file_path = os.path.join(path_generated, file)
            df = pd.read_csv(file_path)
            dfs.append(df)
            
            
    
        combined_df = pd.concat(dfs, ignore_index=True)
        print('shape: ',combined_df.shape)
 
        # get unique sampled
        seqs = list(combined_df['SEQ'])
        seqs_unique = list(set(seqs))
        
        # Evaluate
        properties_sampled = pd.DataFrame()
        generated_space_un = [' '.join(s) for s in seqs_unique]


        mic_sampled,tox_sampled = self.peptide_evaluation.evaluate_MIC_TOX(generated_space_un)
        properties_sampled['SEQ'] = seqs_unique
        properties_sampled['MIC'] = mic_sampled
        properties_sampled['TOX'] = tox_sampled
        lengths = [len(pep) for pep in seqs_unique]
        properties_sampled['LEN'] = lengths
        
        # filter by the properties
        print('filtering')
        filtered_peptides = properties_sampled[(properties_sampled['TOX']>100) & (properties_sampled['MIC']<20) & (properties_sampled['LEN'] < 40) ]
        print('shape_filtered: ',filtered_peptides.shape)
        
        # Plot each feature separately
        for column in filtered_peptides.columns:
            if column!='SEQ':
                plt.figure(figsize=(8, 6))  # Set the size of the plot
                plt.hist(filtered_peptides[column], bins=20, color='skyblue', edgecolor='black')  # Plot the histogram
                
                # Calculate the mean value
                mean_value = filtered_peptides[column].mean()
                
                # Mark the mean value on the plot
                plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean = {mean_value:.2f}')
                
                plt.title(f'Distribution of {column}')  # Set the title
                plt.xlabel(column)  # Set the label for x-axis
                plt.ylabel('Frequency')  # Set the label for y-axis
                plt.legend()  # Show legend
                # plt.grid(True)  # Add gridlines
                plt.show()  # Show the plot
        
        filtered_peptides.to_csv('final_peptides.csv',index=False)
        print('Saved')
        
    def compare_progression(self):
        
        if self.FLAGS.save_peptides:
            path = 'checkpoints/best_rp/model_optimized.h5'
            self.decoder.load_weights(path)
        
        # Generate novel instances with the RL optimized model
        generated_optimized,generated_optimized_sep = self.sample(self.decoder,self.FLAGS.n_generate)
        
        # df = pd.read_csv('sampled_peptides.csv')
        # generated_optimized_sepp = list(df['SEQ'])
        # print(generated_optimized_sepp[0:5])
        # generated_optimized_sep = [' '.join(p) for p in generated_optimized_sepp[0:5]]
        # print(generated_optimized_sep)
        
        
        # Generate novel instances with the original model
        generated_unbiased,generated_unbiased_sep = self.sample(self.decoder_ft,self.FLAGS.n_generate)
        
        # Load the fine-tuning anti-microbial peptide database
        ft_dataset = DataLoader().load_sequences(self.FLAGS,'finetuning')

        subset_idxs_pt = random.sample(range(0, len(ft_dataset)-1),self.FLAGS.n_generate)
        finetune_subset = [seq for idx,seq in enumerate(ft_dataset) if idx in subset_idxs_pt]
        
        # Uniqueness
        print('\nRate of unique sequences (original): %.2f' % ((len(set(generated_unbiased))/len(generated_unbiased))*100))
        print('Rate of unique sequences (optimized): %.2f' % ((len(set(generated_optimized))/len(generated_optimized))*100))
        
        
        # Repeated in dataset
        repeated_finetuned = len(set(generated_optimized) & set(ft_dataset))
        print('\nRate of optimized sequences present on training data: %.2f' % ((repeated_finetuned/len(generated_optimized))*100))
        
        aa_dict_generated = Utils.analyze_aa(generated_optimized,self.token_table)
        aa_dict_training = Utils.analyze_aa(generated_unbiased,self.token_table)
        self.peptide_evaluation.aa_barplot(aa_dict_generated,aa_dict_training)
        
        
        # Compute global properties - unbiased
        d_pt = GlobalDescriptor(generated_unbiased)        
        len1 = len(d_pt.sequences)
        d_pt.filter_aa('B')
        len2 = len(d_pt.sequences)
        d_pt.length()
        print('\nDistribution of generated peptides - unbiased model')
        print("Number of sequences too short: %i" % (len(generated_unbiased) - len1))
        print("Number of invalid (with 'B'): %i" % (len1 - len2))
        print("Number of valid unique seqs: %i" % len2)
        print("Mean sequence length:     %.1f ± %.1f " % (np.mean(d_pt.descriptor), np.std(d_pt.descriptor)))
        print("Median sequence length:   %i" % np.median(d_pt.descriptor))
        print("Minimal sequence length:  %i" % np.min(d_pt.descriptor))
        print("Maximal sequence length:  %i" % np.max(d_pt.descriptor))
        
        
        # Compute global properties - optimized
        d_op = GlobalDescriptor(generated_optimized)        
        len1 = len(d_op.sequences)
        d_op.filter_aa('B')
        len2 = len(d_op.sequences)
        d_op.length()
        print('\nDistribution of generated peptides - optimized model')
        print("Number of sequences too short: %i" % (len(generated_optimized) - len1))
        print("Number of invalid (with 'B'): %i" % (len1 - len2))
        print("Number of valid unique seqs: %i" % len2)
        print("Mean sequence length:     %.1f ± %.1f " % (np.mean(d_op.descriptor), np.std(d_op.descriptor)))
        print("Median sequence length:   %i" % np.median(d_op.descriptor))
        print("Minimal sequence length:  %i" % np.min(d_op.descriptor))
        print("Maximal sequence length:  %i" % np.max(d_op.descriptor))
        
        
        descriptor = 'pepcats'
        sample_unbiased_desc = PeptideDescriptor(generated_unbiased, descriptor)
        sample_unbiased_desc.calculate_autocorr(5)

        sample_optimized_desc = PeptideDescriptor(generated_optimized, descriptor)
        sample_optimized_desc.calculate_autocorr(5)        
        
        # more simple descriptors
        g_sample_unbiased = GlobalDescriptor(sample_unbiased_desc.sequences)
        g_sample_optimized = GlobalDescriptor(sample_optimized_desc.sequences)
       
    
        
        g_sample_unbiased.calculate_all(amide=True)
        names = g_sample_unbiased.featurenames
        print('\nProperties unbiased peptides: ')
        for i in range(len(g_sample_unbiased.descriptor[0,:])):
            feature_name = names[i]
            feature_values = g_sample_unbiased.descriptor[:,i]
            mean_feature = np.mean(feature_values)
            std_feature = np.std(feature_values)
            print(str(feature_name) +': '+ str(round(mean_feature,4)) + " +- " + str(round(std_feature,4)))
    
        g_sample_unbiased.save_descriptor('unbiased_descriptors.csv')
        
        
        
        g_sample_optimized.calculate_all(amide=True)   
        print('\nProperties optimized peptides: ')
        for i in range(len(g_sample_optimized.descriptor[0,:])):
            feature_name = names[i]
            feature_values = g_sample_optimized.descriptor[:,i]
            mean_feature = np.mean(feature_values)
            std_feature = np.std(feature_values)
            print(str(feature_name) +': '+ str(round(mean_feature,4)) + " +- " + str(round(std_feature,4)))
    
        g_sample_optimized.save_descriptor('optimized_descriptors.csv')
      
        # hydrophobic moments
        uh_sample_unbiased = PeptideDescriptor(sample_unbiased_desc.sequences, 'eisenberg')
        uh_sample_unbiased.calculate_moment()
        uh_sample_optimized = PeptideDescriptor(sample_optimized_desc.sequences, 'eisenberg')
        uh_sample_optimized.calculate_moment()

        
        print("\nHydrophobic moment - unbiased: %.3f +/- %.3f\n" %
                (np.mean(uh_sample_unbiased.descriptor), np.std(uh_sample_unbiased.descriptor)))
        print("Hydrophobic moment - optimized: %.3f +/- %.3f\n" %
              (np.mean(uh_sample_optimized.descriptor), np.std(uh_sample_optimized.descriptor)))   
        
        a = GlobalAnalysis([uh_sample_unbiased.sequences, uh_sample_optimized.sequences],
                                   ['unbiased', 'optimized'])
        a.plot_summary(filename=self.FLAGS.checkpoint_path + 'summary.png')
        
        
        MIC_optimized,TOX_optimized = self.peptide_evaluation.evaluate_MIC_TOX(generated_optimized_sep)

        MIC_unbiased,TOX_unbiased= self.peptide_evaluation.evaluate_MIC_TOX(generated_unbiased_sep)
        
       
        prediction_mic_unb = np.array(MIC_unbiased)
        prediction_mic_b= np.array(MIC_optimized)
        
        prediction_tox_unb = np.array(TOX_unbiased)
        prediction_tox_b= np.array(TOX_optimized)

        # lisa = [1,3,4,5]
        # lisb = [11.3,34,4.4,5.4]
        # print(lisa)
        legend_mic_unb = 'Unbiased MIC'
        legend_mic_b = 'Optimized MIC'
        print("\n\nMax MIC: (UNB,B)", np.max(prediction_mic_unb),np.max(prediction_mic_b))
        print("Mean MIC: (UNB,B)", np.mean(prediction_mic_unb),np.mean(prediction_mic_b))
        print("STD MIC: (UNB,B)", np.std(prediction_mic_unb),np.std(prediction_mic_b))
        print("Min MIC: (UNB,B)", np.min(prediction_mic_unb),np.min(prediction_mic_b))
        
        print("\n\nMax HD50: (UNB,B)", np.max(prediction_tox_unb),np.max(prediction_tox_b))
        print("Mean HD50: (UNB,B)", np.mean(prediction_tox_unb),np.mean(prediction_tox_b))
        print("STD HD50: (UNB,B)", np.std(prediction_tox_unb),np.std(prediction_tox_b))
        print("Min HD50: (UNB,B)", np.min(prediction_tox_unb),np.min(prediction_tox_b))
        
        # label_mic = 'Predicted MIC'
        # plot_title_mic = 'Distribution of predicted MIC for generated peptides'
        # print('1')
        # sns.axes_style("darkgrid")
        # v1_mic = pd.Series(lisa, name=legend_mic_unb)
        # v2_mic = pd.Series(lisb, name=legend_mic_b)
        # print('2')               
        # ax = sns.kdeplot(v1_mic, shade=True,color='g',label=legend_mic_unb)
        # print('3')
        # sns.kdeplot(v2_mic, shade=True,color='r',label =legend_mic_b )
        # print('4')
        # ax.set(xlabel=label_mic, 
        #        title=plot_title_mic)
        # # plt.legend()
        # print('6')
        # plt.show()
        # print('7')
                        
                
        
        
 
        