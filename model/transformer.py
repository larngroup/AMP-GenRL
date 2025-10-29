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


tf.config.run_functions_eagerly(True)

        
class Transformer():
    """Transformer general Class"""
    def __init__(self, FLAGS):
        # Implementation parameters
        self.FLAGS = FLAGS
        
        # Load the table of possible tokens
        self.token_table = Utils().vocabulary_aminoacids 
        self.vocab_size = len(self.token_table)
        
        # Dictionary that makes the correspondence between each aminoacid and unique integers
        self.tokenDict = Utils.aminoacid_dict(self.token_table)
        self.inv_tokenDict = {v: k for k, v in self.tokenDict.items()}
        
        # Initialization of the generated proteins evaluation class
        self.peptide_evaluation = Peptide_evaluation(self.FLAGS)
        
    
    def loss_function(self,y_true,y_pred,mask):
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
    def train_step(self, x, target): 
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
            loss = self.loss_function(target, predictions,padding_mask) 
        
        gradients = tape.gradient(loss, self.decoder.trainable_variables) 
        self.optimizer.apply_gradients(zip(gradients, self.decoder.trainable_variables)) 

        return loss 

    
    def train(self,data,FLAGS,n_epochs, batchsize, lr_scheduler, lr_WarmUpSteps,                                   
              min_delta, patience, optimizer_fn, dropout, d_model, n_layers, 
              n_heads, activation_func, ff_dim,training_procedure='pre_training'):
        
        """Builds the model and implements the training dynamics"""
                    
        # Initializes the model
        if training_procedure== 'pre_training':
            self.decoder = Transformer_Decoder(d_model,ff_dim,n_heads,n_layers,self.FLAGS.max_strlen_pt,self.vocab_size,activation_func)   
            sequence_in = tf.ones([1, 1])
            decoder_output,_,_ = self.decoder(sequence_in)
      
        #Loads the pre-trained parameters 
        if training_procedure == 'finetuning':
            self.decoder = Transformer_Decoder(d_model,ff_dim,n_heads,n_layers,self.FLAGS.max_strlen_pt,self.vocab_size,activation_func) 
            sequence_in = tf.ones([1, 1])
            decoder_output,_,_ = self.decoder(sequence_in)
            path = 'model_pretraining.h5'
            self.decoder.load_weights(path)
            
            
        self.crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, reduction='none')
        
        lr = WarmupThenDecaySchedule(d_model,lr_WarmUpSteps)
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9,                                               
                                                    beta_2=0.99, epsilon=1e-08)
        

        last_loss = {'epoch':0,'value':1000} 
        for epoch in range(n_epochs): 
            print(f'Epoch {epoch+1}/{n_epochs}') 
            start = time.time() 
            loss_epoch = [] 
            for num, (x_train,y_train) in enumerate(data): 
                loss_batch = self.train_step(x_train,y_train) 
                loss_epoch.append(loss_batch) 
                if num == len(data)-1: 
                    print(f'{num+1}/{len(data)} - {round(time.time() - start)}s - loss: {np.mean(loss_epoch):.4f}')    
					
            if (last_loss['value'] - tf.math.reduce_mean(loss_epoch)) >= min_delta: 
                last_loss['value'] = tf.math.reduce_mean(loss_epoch) 
                last_loss['epoch'] = epoch+1
                print('Saving model...') 
                if training_procedure == 'finetuning':
                    self.decoder.save_weights(self.FLAGS.checkpoint_path + 'model_finetuning.h5')  
                else:
                    self.decoder.save_weights(self.FLAGS.checkpoint_path + 'model.h5')  
                
            if ((epoch+1) - last_loss['epoch']) >= patience: 
                break 
        
            
                
    def sample(self,model):
        
        print('\nSampling new peptides...')
        self.decoder = Transformer_Decoder(256,1024,4,4,200,self.vocab_size,'relu')    
        sequence_in = tf.ones([1, 1])
        decoder_output,_,_ = self.decoder(sequence_in)
        
        if model == 'pre_training':
            path = 'model_pretraining.h5'
            
        elif model == 'finetuning':
           path = 'model_finetuning.h5'
        
        self.decoder.load_weights(path)

        generated_all = []
        for i in range(self.FLAGS.n_generate):
           sequence = np.zeros([1,self.FLAGS.max_strlen_pt])
           sequence[0,0] = self.tokenDict['<GO>']
           sampled_token = self.tokenDict['<GO>']
           sampled_sequence = []
           while sampled_token not in [self.tokenDict['<END>'],self.tokenDict['<PAD>']] and len(sampled_sequence) < self.FLAGS.max_strlen_pt-1:
            
               predictions,_,mask=self.decoder.predict(tf.constant(sequence))
               # predictions_np = predictions.numpy()
               probability_dist = predictions[0,len(sampled_sequence),:]
               sampled_token = self.sample_token(probability_dist)
               
               
               # Append the predicted token to the generated sequence
               if sampled_token not in [self.tokenDict['<END>'],self.tokenDict['<PAD>']]:
                   sampled_sequence.append(sampled_token)
               
                   # Update the input sequence for the next iteration
                   sequence[0,len(sampled_sequence)] = sampled_token
                   # print('next')
               
           generated_all.append(sampled_sequence)
           
                   
        protein_sequences_all = Utils.int_to_prot(generated_all,self.inv_tokenDict)  
        protein_sequences_filtered = [s for s in protein_sequences_all if len(s)>=5]
        return protein_sequences_filtered
    
    def sample_token(self,predictions, temperature=1.0):
        # predictions = np.log(predictions) / temperature
        # exp_predictions = np.exp(predictions)
        # predictions = exp_predictions / np.sum(exp_predictions)
        sampled_token = np.random.choice(len(predictions), 1, p=predictions)[0]
        return sampled_token    
                
    def evaluate(self,data_test):
        """Predicts resuts for the test dataset"""
        # print('\nPredicting on test set...') 
        
        
        test_loss = []
        for num, (input_sequences,target_sequences) in enumerate(data_test):
            predictions,_,padding_mask = self.decoder(input_sequences)
            loss_batch = self.loss_function(target_sequences, predictions,padding_mask)
            test_loss.append(loss_batch.numpy())
            
            
        print("Loss: ", np.mean(test_loss))
        
        return np.mean(test_loss)
    
    def properties_evaluation(self,sample_pretrained,sample_finetuned,pretrain_all,finetune_all):
        """Estimates properties for generated proteins and compares them with 
        instances of the training set. """
        
        # print(sample_pretrained)
        # print('\nFinetuned: \n')
        # print(sample_finetuned)
    
        # # Extract randomly a subset of proteins from the training set to compare with generated sequences
        subset_idxs_pt = random.sample(range(0, len(pretrain_all)-1),self.FLAGS.n_generate)
        pretrain_subset = [seq for idx,seq in enumerate(pretrain_all) if idx in subset_idxs_pt]
        
        subset_idxs_train = random.sample(range(0, len(finetune_all)-1),self.FLAGS.n_generate)
        finetune_subset = [seq for idx,seq in enumerate(finetune_all) if idx in subset_idxs_train]
        
        # Uniqueness
        print('Rate of unique sequences (pretrained): %.2f' % ((len(set(sample_pretrained))/len(sample_pretrained))*100))
        print('Rate of unique sequences (finetuned): %.2f' % ((len(set(sample_finetuned))/len(sample_finetuned))*100))
        
        # Repeated in dataset
        repeated_pretrained = len(set(sample_pretrained) & set(pretrain_all))
        print('Rate of sequences present on pre-training data: %.2f' % ((repeated_pretrained/len(sample_pretrained))*100))
        
        repeated_finetuned = len(set(sample_finetuned) & set(finetune_all))
        print('Rate of sequences present on fine-training data: %.2f' % ((repeated_finetuned/len(sample_finetuned))*100))
        
        aa_dict_generated = Utils.analyze_aa(sample_finetuned,self.token_table)
        aa_dict_training = Utils.analyze_aa(finetune_subset,self.token_table)
        self.peptide_evaluation.aa_barplot(aa_dict_generated,aa_dict_training)
        
        
        # Compute global properties - pre_trained
        d_pt = GlobalDescriptor(sample_pretrained)        
        len1 = len(d_pt.sequences)
        d_pt.filter_aa('B')
        len2 = len(d_pt.sequences)
        d_pt.length()
        print('\nDistribution of generated proteins - pretrained model')
        print("Number of sequences too short: %i" % (len(sample_pretrained) - len1))
        print("Number of invalid (with 'B'): %i" % (len1 - len2))
        print("Number of valid unique seqs: %i" % len2)
        print("Mean sequence length:     %.1f ± %.1f " % (np.mean(d_pt.descriptor), np.std(d_pt.descriptor)))
        print("Median sequence length:   %i" % np.median(d_pt.descriptor))
        print("Minimal sequence length:  %i" % np.min(d_pt.descriptor))
        print("Maximal sequence length:  %i" % np.max(d_pt.descriptor))
        
        
        # Compute global properties - fine_tuned
        d_ft = GlobalDescriptor(sample_finetuned)        
        len1 = len(d_ft.sequences)
        d_ft.filter_aa('B')
        len2 = len(d_ft.sequences)
        d_ft.length()
        print('\nDistribution of generated proteins - finetuned model')
        print("Number of sequences too short: %i" % (len(sample_finetuned) - len1))
        print("Number of invalid (with 'B'): %i" % (len1 - len2))
        print("Number of valid unique seqs: %i" % len2)
        print("Mean sequence length:     %.1f ± %.1f " % (np.mean(d_ft.descriptor), np.std(d_ft.descriptor)))
        print("Median sequence length:   %i" % np.median(d_ft.descriptor))
        print("Minimal sequence length:  %i" % np.min(d_ft.descriptor))
        print("Maximal sequence length:  %i" % np.max(d_ft.descriptor))
        
        
        descriptor = 'pepcats'
        # Descriptors sample pre_training
        # sample_pt_desc = PeptideDescriptor([s[1:].rstrip() for s in sample_pretrained], descriptor)
        sample_pt_desc = PeptideDescriptor(sample_pretrained, descriptor)
        sample_pt_desc.calculate_autocorr(5)

        # Descriptors sample fine_tuning
        # sample_ft_desc = PeptideDescriptor([s[1:].rstrip() for s in sample_finetuned], descriptor)
        sample_ft_desc = PeptideDescriptor(sample_finetuned, descriptor)
        sample_ft_desc.calculate_autocorr(5)        

        # Descriptors pretraining subset
        # pt_set_desc = PeptideDescriptor([s[1:].rstrip() for s in pretrain_set], descriptor)
        pt_set_desc = PeptideDescriptor(pretrain_subset, descriptor)
        pt_set_desc.calculate_autocorr(5)
        
        # Descriptors finetuning subset
        ft_set_desc = PeptideDescriptor([s[1:].rstrip() for s in finetune_subset], descriptor)
        ft_set_desc.calculate_autocorr(5)        
        
        
        # Calculate distance between sampled protein sets (pre_trained vs fine-tuned) in Pepcats space
        sample_dist = distance.cdist(sample_pt_desc.descriptor, sample_ft_desc.descriptor, metric='euclidean')
        print("\nAverage euclidean distance - sampled pre_trained vs sampled fine_tuned: %.3f +/- %.3f\n" %
                (np.mean(sample_dist), np.std(sample_dist)))
       
        # Calculate distance between generated and pre-training sequences in Pepcats space

        pt_dist = distance.cdist(pt_set_desc.descriptor, sample_ft_desc.descriptor, metric='euclidean')
        print("\nAverage euclidean distance - pre_training set vs sampled fine_tuned: %.3f +/- %.3f\n" %
                (np.mean(pt_dist), np.std(pt_dist)))
        
        # Calculate distance between generated and fine-tuning sequences in Pepcats space
        ft_dist = distance.cdist(ft_set_desc.descriptor, sample_ft_desc.descriptor, metric='euclidean')
        print("\nAverage euclidean distance - fine-tuning set vs sampled fine_tuned: %.3f +/- %.3f\n" %
                (np.mean(ft_dist), np.std(ft_dist)))
        
        
        # more simple descriptors
        g_sample_pt = GlobalDescriptor(sample_pt_desc.sequences)
        g_sample_ft = GlobalDescriptor(sample_ft_desc.sequences)
        g_set_pt = GlobalDescriptor(pt_set_desc.sequences)
        g_set_ft = GlobalDescriptor(ft_set_desc.sequences)
       
        g_sample_pt.calculate_all()
        g_sample_ft.calculate_all()
        g_set_pt.calculate_all()
        g_set_ft.calculate_all()
        
        sclr = StandardScaler()
        sclr.fit(g_sample_ft.descriptor)
        
        # Distance calculation for scaled global descriptors
        desc_sample_dist = distance.cdist(sclr.transform(g_sample_pt.descriptor), sclr.transform(g_sample_ft.descriptor),
                                   metric='euclidean')
        print("\nAverage euclidean distance for descriptors: sampled pre_trained vs sampled fine_tuned: %.2f +/- %.2f\n" %
                (np.mean(desc_sample_dist), np.std(desc_sample_dist)))
        
        desc_pt_dist = distance.cdist(sclr.transform(g_set_pt.descriptor), sclr.transform(g_sample_ft.descriptor),
                                   metric='euclidean')
        print("Average euclidean distance for descriptors: pre_training set vs sampled fine_tuned: %.2f +/- %.2f\n" %
                (np.mean(desc_pt_dist), np.std(desc_pt_dist)))
        
        desc_ft_dist = distance.cdist(sclr.transform(g_set_ft.descriptor), sclr.transform(g_sample_ft.descriptor),
                                   metric='euclidean')
        print("Average euclidean distance for descriptors: fine_tuning set vs sampled fine_tuned: %.2f +/- %.2f\n" %
                (np.mean(desc_ft_dist), np.std(desc_ft_dist)))
        
        # hydrophobic moments
        uh_sample_pt = PeptideDescriptor(sample_pt_desc.sequences, 'eisenberg')
        uh_sample_pt.calculate_moment()
        uh_sample_ft = PeptideDescriptor(sample_ft_desc.sequences, 'eisenberg')
        uh_sample_ft.calculate_moment()
        uh_set_pt = PeptideDescriptor(pt_set_desc.sequences, 'eisenberg')
        uh_set_pt.calculate_moment()
        uh_set_ft = PeptideDescriptor(ft_set_desc.sequences, 'eisenberg')
        uh_set_ft.calculate_moment()
        
        print("Hydrophobic moment - sampled pre_training: %.3f +/- %.3f\n" %
                (np.mean(uh_sample_pt.descriptor), np.std(uh_sample_pt.descriptor)))
        print("Hydrophobic moment - sampled fine_tuning: %.3f +/- %.3f\n" %
              (np.mean(uh_sample_ft.descriptor), np.std(uh_sample_ft.descriptor)))
        print("Hydrophobic moment - pre_training set: %.3f +/- %.3f\n" %
                (np.mean(uh_set_pt.descriptor), np.std(uh_set_pt.descriptor)))
        print("Hydrophobic moment - fine-tuning set: %.3f +/- %.3f\n" %
              (np.mean(uh_set_ft.descriptor), np.std(uh_set_ft.descriptor)))        
        
        a = GlobalAnalysis([uh_sample_pt.sequences, uh_sample_ft.sequences, uh_set_pt.sequences, uh_set_ft.sequences],
                                   ['sampled_pt', 'sampled_ft', 'train_pt', 'train_ft'])
        a.plot_summary(filename=self.FLAGS.checkpoint_path + 'summary.png')
        
        
        
        

 