# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 10:07:51 2021

@author: tiago
"""

# Internal 
from model.transformer import Transformer
from model.optimization import Optimization
from model.argument_parser import logging,argparser
from dataloader.dataloader import DataLoader

# External
import torch
import tensorflow as tf
import warnings
import time
import os
import itertools
import gc
warnings.filterwarnings('ignore')


def run_train_model(FLAGS):
    
    #Define the parameters
    n_epochs = 70#FLAGS.n_epochs    
    batch_size = FLAGS.batchsize
    lr_scheduler = FLAGS.lr_scheduler    
    lr_WarmUpSteps = FLAGS.lr_WarmUpSteps    
    drop_rate = 0.1#FLAGS.dropout[0]    
    optimizer_fn = 'adam' #FLAGS.optimizer_fn[0]   
    min_delta = FLAGS.min_delta    
    patience = FLAGS.patience    
    d_model = 256 #FLAGS.d_model[0]
    n_layers = 4 #FLAGS.n_layers[0]
    n_heads = 4 #FLAGS.n_heads[0]
    activation_func = 'relu' # FLAGS.activation_func[0]
    ff_dim = 1024 #FLAGS.ff_dim[0]
    
 	# Initialize the Transformer class
    transformer_model = Transformer(FLAGS)
        
    if FLAGS.option == 'pre_train':
    
        print('\nLoading pre_training data...')
        general_dataset = DataLoader().load_sequences(FLAGS)
        
        # print('\nPre-processing data...')
        processed_dataset_train,processed_dataset_test,pt_sequences = DataLoader.pre_process_data(general_dataset,transformer_model,FLAGS)
    
        # print('\nPre-training the model...')
        sample_pretraining_proteins = transformer_model.train(processed_dataset_train,FLAGS, n_epochs, batch_size, lr_scheduler,
                                                lr_WarmUpSteps, min_delta, patience,
                                                optimizer_fn, drop_rate, d_model, n_layers,
                                                n_heads, activation_func, ff_dim)
    
        print('\nEvaluating pre-trained model on test set...')
        loss_pt = transformer_model.evaluate(processed_dataset_test)
        logging("Test set pretrain- " + (" Loss = %0.3f" %(loss_pt)),FLAGS)
        
    elif FLAGS.option == 'fine_tune':
        
        print('\nLoading fine-tuning data...')
        ft_dataset = DataLoader().load_sequences(FLAGS,'finetuning')
        
        print('\nPre-processing data...')
        processed_ft_train,processed_ft_test,ft_sequences = DataLoader.pre_process_data(ft_dataset,transformer_model,FLAGS)
            
        print('\nFine-tuning the model...')
        transformer_model.train(processed_ft_train,FLAGS, n_epochs, batch_size, lr_scheduler,
                                              lr_WarmUpSteps, min_delta, patience,
                                              optimizer_fn, drop_rate, d_model, n_layers,
                                              n_heads, activation_func, ff_dim,'finetuning')
        
                
        print('\nEvaluating fine-tuned model...')
        loss_ft = transformer_model.evaluate(processed_ft_test)
        logging("Test set finetuning - " + (" Loss = %0.3f" %(loss_ft)),FLAGS)
        
    elif FLAGS.option == 'evaluation':
        
        print('\nLoading pre_training data...')
        pt_sequences = DataLoader().load_sequences(FLAGS)
        
        print('\nLoading fine_tuning data...')
        ft_sequences = DataLoader().load_sequences(FLAGS,'finetuning')
        
        # Sampling pre_trained model
        sample_pretraining_proteins = transformer_model.sample('pre_training')
        
        # Sampling fine_tuned model
        sample_finetuned_proteins = transformer_model.sample('finetuning')
        
    
        print('\nGeneral comparison of sampled molecules...')
        transformer_model.properties_evaluation(sample_pretraining_proteins,sample_finetuned_proteins,pt_sequences,ft_sequences)
    
    elif FLAGS.option == 'optimization':
       
        optimization_rl = Optimization(FLAGS)
        
        print('Optimizing with RL...')
        # optimization_rl.optimization_loop()
        
        
        # if FLAGS.save_peptides:
        # optimization_rl.compare_progression()
        # optimization_rl.save_best_peptides()
        optimization_rl.combine_all_peptides()
    
    

    
    

def run_grid_search(FLAGS):	
    """
    Run Grid Search function
    ----------
    FLAGS: arguments object
    """
    
    n_epochs = FLAGS.n_epochs    
    batch_size = FLAGS.batchsize
    lr_scheduler = FLAGS.lr_scheduler    
    lr_WarmUpSteps = FLAGS.lr_WarmUpSteps    
    drop_rate_set = FLAGS.dropout    
    optimizer_fn = FLAGS.optimizer_fn    
    min_delta = FLAGS.min_delta    
    patience = FLAGS.patience    
    d_model = FLAGS.d_model
    n_layers = FLAGS.n_layers
    n_heads = FLAGS.n_heads
    activation_func = FLAGS.activation_func
    ff_dim = FLAGS.ff_dim
    
    
 	# Initialize the Transformer model
    transformer_model = Transformer(FLAGS)
    
    raw_dataset = DataLoader().load_smiles(FLAGS)
    
    processed_dataset_train,processed_dataset_test = DataLoader.pre_process_data(raw_dataset,transformer_model,FLAGS)
    
    logging("--------------------Grid Search-------------------", FLAGS)
    
    for params in itertools.product(optimizer_fn, drop_rate_set, d_model,                                    
                                    n_layers, n_heads, activation_func, ff_dim):
        
        p1, p2, p3, p4, p5, p6, p7 = params
      
        results = []
        transformer_model = Transformer(FLAGS)
        # for fold_idx in range(len(folds)):            
        # index_train = list(itertools.chain.from_iterable([folds[i] for i in range(len(folds)) if i != fold_idx]))
        # index_val = folds[fold_idx]
        # data_train = [tf.gather(i, index_train) for i in data]
        # data_val = [tf.gather(i, index_val) for i in data]

        encoder = transformer_model.train(processed_dataset_train,FLAGS, n_epochs, batch_size, lr_scheduler,
                                          lr_WarmUpSteps, min_delta, patience,
                                          p1, p2, p3, p4, p5, p6, p7)
                                                                   
        loss,acc = transformer_model.evaluate(processed_dataset_test)             
        
        results.append((loss,acc))
        logging(("Epochs = %d, Batch size= %d, Lr scheduler = %s, Warmup steps = %d, "
                 "Minimum delta = %d, Patience = %d, Optimizer = %s,  Dropout= %d, " +                     
                 "Model dimension = %d, Number of Layers= %d, Number of heads= %d, " +                     
                 "Activation function= %s, Fully-connected dimension = %d, " +                     
                 "SCCE = %0.3f, ACC= %0.3f") %                    
                (n_epochs, batch_size, lr_scheduler,lr_WarmUpSteps, min_delta, patience,
                                                  p1, p2, p3, p4, p5, p6, p7, loss,acc), FLAGS)
        
        del encoder
        gc.collect()
        # logging("Mean - " + (" SCCE = %0.3f, ACC = %0.3f" % (np.mean(results, axis=0)[0], np.mean(results, axis=0)[1]), FLAGS)
        logging("Mean - " + (" SCCE = %0.3f, ACC = %0.3f" %
                             (loss,acc)), FLAGS)

def run():    
    """Selects how to run the model: train best approach (train) or identify optimal 
    configuration using the grid-search strategy (validation) 
    """

    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        
    except:
        print('Bad')
        # Invalid device or cannot modify virtual devices once initialized.
        pass
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    FLAGS = argparser()
    FLAGS.log_dir = os.getcwd() + '/logs/' + time.strftime("d_%m_%y_%H_%M", time.gmtime())+"/"
    FLAGS.checkpoint_path = os.getcwd() + '/checkpoints/' + time.strftime("d_%m_%y_%H_%M", time.gmtime())+"/"

    if not os.path.exists(FLAGS.log_dir):
    	os.makedirs(FLAGS.log_dir)
    if not os.path.exists(FLAGS.checkpoint_path):
    	os.makedirs(FLAGS.checkpoint_path)
   
    logging(str(FLAGS), FLAGS)

    if FLAGS.option == 'pre_train' or FLAGS.option == 'fine_tune' or FLAGS.option == 'evaluation' or FLAGS.option == 'optimization':
    	run_train_model(FLAGS)
    if FLAGS.option == 'validation':
    	run_grid_search(FLAGS)

    
if __name__ == '__main__':
    run()

 

 