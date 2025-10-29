# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 09:20:20 2024

@author: tiago
"""

# internal
from utils.utils import Utils
from model.seq_dataloader import *
from model.seq_dataloader import _get_test_data_loader
# from model.predictor_tox import Predictor_tox

# external
import numpy as np
import matplotlib.pyplot as plt
import random
from model.model_def import REG
import torch
from modlamp.analysis import GlobalAnalysis
from modlamp.core import count_aas
from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor
from modlamp.sequences import Random, Helices
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance


class Peptide_evaluation:
    """Protein evalution class"""
    def __init__(self, FLAGS):
        
        # Implementation parameters
        self.FLAGS = FLAGS
        
        # Initialize Anti-Microbial Activity prediction model
        model_path_ama = "prot_bert_finetune_reproduce.pkl"
        self.model_ama = REG()
        self.model_ama.load_state_dict(torch.load(model_path_ama))
        
    	# Initialize the Toxicity prediction model
        model_path_tox = "prot_bert_finetune_toxicity.pkl"
        self.model_tox = REG()
        self.model_tox.load_state_dict(torch.load(model_path_tox))
        
        # # Load the table of possible tokens
        # self.token_table = Utils().voc_table_synergistic 
        # self.vocab_size = len(self.token_table)
        
        # # Dictionary that makes the correspondence between each token and unique integers
        # self.tokenDict = Utils.smilesDict(self.token_table)
        # self.inv_tokenDict = {v: k for k, v in self.tokenDict.items()}
                
    
    def aa_barplot(self,aa_generated,aa_training_set):
        """ 
        
        """
        X = np.arange(len(aa_training_set))
        ax = plt.subplot(111)
        ax.bar(X, aa_generated.values(), width=0.2, color='r', align='center')
        ax.bar(X-0.2, aa_training_set.values(), width=0.2, color='g', align='center')
        ax.legend(('Generated','Training set'))
        plt.xlabel("Amino acid residue")
        plt.ylabel("Frequency")
        plt.xticks(X, aa_training_set.keys())
        plt.title("Amino acid distribution", fontsize=17)
        plt.show()


    def get_reward(self,sampled_peptides,current_buffer):
      
        # compute ama and tox predictions
        batch_size = len(sampled_peptides)
        
        samples_loader = _get_test_data_loader(batch_size, sampled_peptides)
        predictions_ama = []
        predictions_tox = []
        self.model_ama.eval()
        self.model_tox.eval()
        with torch.no_grad():
            for batch in samples_loader:
                b_input_ids = batch['input_ids']
                b_input_mask = batch['attention_mask']
                predict_pMIC, _ = self.model_ama(b_input_ids, attention_mask=b_input_mask)
                predict_pTOX, _ = self.model_tox(b_input_ids, attention_mask=b_input_mask)
                predictions_ama.extend(predict_pMIC.data.numpy())
                predictions_tox.extend(predict_pTOX.data.numpy())
               
        predictions_ama = [item for sublist in predictions_ama for item in sublist]
        predictions_tox = [item for sublist in predictions_tox for item in sublist]
        
        MIC_values = [10**(-item) for item in predictions_ama]
        print('\nMIC values: ',MIC_values)
        
        TOX_values = [10**(-item) for item in predictions_tox]
        print('\nHD50 values: ',TOX_values)  
        
        sampled_corrected = [pep.replace(" ", "") for pep in sampled_peptides]
        
    
        # UPDATE REPLAY BUFFER
        # properties to look at when selecting peptides to the repetition buffer:
            # length, MIC, tox, others?
            
        rewards_length = []
        for idx,pep in enumerate(sampled_corrected):
            length_peptide = len(pep)
            # print(length_peptide)
            MIC_peptide = MIC_values[idx]
            TOX_peptide = TOX_values[idx]
    
            
            if  length_peptide < 70 and length_peptide > 20 and MIC_peptide < 17.0 and TOX_peptide > 150:
                current_buffer.append(pep)
                
                
                
                # Analyze length
            # if length_peptide >=30 and length_peptide<=85:
            #     reward_bonus = 1 - (length_peptide - 30) * (1 / 55)
            # elif length_peptide >= 20 and length_peptide < 30:
            #     reward_bonus = 1 - (30 - length_peptide) * (1 / 10)
            # else:
            #     reward_bonus = 0
            
            # Analyze length
            if length_peptide >=22 and length_peptide<=65:
                reward_bonus = 0.4
            else:
                reward_bonus = 0
            
            rewards_length.append(reward_bonus/1.2)
         
        print('\nPeptides in the replay buffer: ', len(current_buffer))
        
        
        #COMPUTE REWARD AMA prediction
        # Design reward function compute reward
        rewards_ama = []
        rewards_tox = []
        rewards_combined = []
        for idx,mic in enumerate(MIC_values):
            
            if mic >= 50.0: 
                mic_scaled=0
                rewards_ama.append(mic_scaled)
            elif mic >= 30.0: 
                mic_scaled = 0.5
                rewards_ama.append(mic_scaled)
            else:
                if mic<=2.0:
                    mic_scaled = 1
                else:
                    mic_scaled = 1 - (mic-2)/(30-2)
                rewards_ama.append(np.exp(mic_scaled)) #np.exp(mic_scaled*1.7))
                
            hd50_pred = TOX_values[idx]
            if hd50_pred < 30.0 or mic > 50: 
                hd50_scaled=0
                rewards_tox.append(hd50_scaled)
            elif hd50_pred >= 30.0 and hd50_pred < 70.0: 
                hd50_scaled = 0.5
                rewards_tox.append(hd50_scaled)
            else:
                if hd50_pred>195.0:
                    hd50_scaled = 1
                else:
                    hd50_scaled = (hd50_pred-5)/(195.0-5)
                    
                rewards_tox.append(np.exp(hd50_scaled)) #np.exp(hd50_scaled*1.7))
                
            # TODO weights stategy to balance objectives
            rewards_combined.append((rewards_ama[idx])*0.7+(rewards_tox[idx])*0.3+rewards_length[idx])
                
        print('\nRewards MIC: ',rewards_ama)
     
        
        # combine
        
        for i,pep in enumerate(sampled_corrected):

            len_peptide = len(pep)
            # print('\n',pep_corrected)
            # print(i,len_peptide)
            
            rewards_peptide = np.ones([len_peptide+1,]) # +1 is for the GO token
            
            pep_rew_ama = rewards_combined[i]
            
            rewards_peptide = rewards_peptide * pep_rew_ama 
                
            if i == 0:
                batch_rewards = rewards_peptide.copy()
            else:
                batch_rewards = np.concatenate([batch_rewards,rewards_peptide])
            # print(batch_rewards.shape)
            
        return batch_rewards,rewards_combined,rewards_ama,rewards_tox,rewards_length,current_buffer

    
    
    
    def evaluate_MIC_TOX(self,sampled_peptides):
                
        # print('sampled')
        print(len(sampled_peptides))
        # compute ama
        # batch_size = len(sampled_peptides)
        
        batch_size = 32
        
        test_loader = _get_test_data_loader(batch_size, sampled_peptides)
        predictions_pmic= []
        predictions_ptox= []
        self.model_ama.eval()
        self.model_tox.eval()
        with torch.no_grad():
            for batch in test_loader:
                b_input_ids = batch['input_ids']
                b_input_mask = batch['attention_mask']
                predict_pMIC, _ = self.model_ama(b_input_ids, attention_mask=b_input_mask)
                predictions_pmic.extend(predict_pMIC.data.numpy())
                predict_pTOX, _ = self.model_tox(b_input_ids, attention_mask=b_input_mask)
                predictions_ptox.extend(predict_pTOX.data.numpy())
        
        # print(test_predict_list)
        predictions_processed_mic = [item for sublist in predictions_pmic for item in sublist]
        predictions_processed_tox = [item for sublist in predictions_ptox for item in sublist]
        print(len(predictions_processed_tox))
        print(len(predictions_processed_mic))
        
        # print(test_predict_list_new)
        MIC_values = [10**(-item) for item in predictions_processed_mic]
        TOX_values = [10**(-item) for item in predictions_processed_tox]
        

        return MIC_values,TOX_values
    
    
    def evaluate_objectives(self,sampled_peptides,sr_ama,sr_tox,sr_len,sr_all):
        
        optimized_ama = 0
        optimized_length = 0
        optimized_all = 0
        optimized_tox = 0
        
        mic_sampled,tox_sampled = self.evaluate_MIC_TOX(sampled_peptides)
            
        sampled_corrected = [pep.replace(" ", "") for pep in sampled_peptides]
        
        # tox_predictions = self.pred_tox_models.predict_tox(sampled_corrected)
        
        for idx,pep in enumerate(sampled_corrected):
            length_bool = False
            ama_bool = False
            tox_bool = False
            
            length_peptide = len(pep)
            MIC_peptide = mic_sampled[idx]
            TOX_peptide = tox_sampled[idx]
            
            if  length_peptide < 40 and length_peptide > 7:
                length_bool = True
                optimized_length+=1
        
            if  MIC_peptide < 20.0:
                ama_bool = True
                optimized_ama+=1
                
            if  TOX_peptide > 190.0:
                tox_bool = True
                optimized_tox+=1
                
            if ama_bool == True and length_bool == True and tox_bool == True:
                optimized_all +=1
                
        print('\n% optimized AMA: ', optimized_ama/len(sampled_corrected))
        print('\n% optimized TOX: ', optimized_tox/len(sampled_corrected))
        print('\n% optimized length: ', optimized_length/len(sampled_corrected))
        print('\n% optimized all: ', optimized_all/len(sampled_corrected))
        sr_ama.append(optimized_ama/len(sampled_corrected))
        sr_tox.append(optimized_tox/len(sampled_corrected))
        sr_len.append(optimized_length/len(sampled_corrected))
        sr_all.append(optimized_all/len(sampled_corrected))
        
        plt.plot(sr_ama, label='AMA')
        plt.plot(sr_tox, label='TOX')
        plt.plot(sr_len, label='Length')
        plt.plot(sr_all, label='All')
        
        # Adding labels and title
        plt.xlabel('Iterations')
        plt.ylabel('% of sucess')
        plt.title('Progression of success')
        
        # Adding legend
        plt.legend()
        
        # Display the plot
        plt.show()
        
        
        return sr_ama,sr_tox,sr_len,sr_all
    
    
          