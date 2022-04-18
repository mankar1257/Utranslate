


import numpy as np
import pickle
import json


def save_config(self,save_path):
    data = {
            'batch_size':self.BATCH_SIZE,
            'max_length_input':self.max_length_input,
            'max_length_output':self.max_length_output,
            'units':self.LSTM_UNITS,
            'use_embd':self.use_embd
            }
    with open(save_path + '/config.json', 'w') as outfile:
        json.dump(data, outfile)


def save_embeddings(self,save_path):
    if self.use_embd:
      np.save(save_path + '/embedding_matrix/input_emb.npy',self.input_emb)
      np.save(save_path + '/embedding_matrix/target_emb.npy',self.target_emb)

def save_tokenizers(self,save_path):
    with open(save_path + '/tokenizer/target_tokenizer.pickle', 'wb') as handle:
      pickle.dump(self.targ_lang, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(save_path +'/tokenizer/input_tokenizer.pickle', 'wb') as handle:
      pickle.dump(self.inp_lang, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_example_input(self,save_path):
    np.save(save_path +'/example_input_batch.npy',self.example_input_batch.numpy())


def save(self,path = ''):

  if path != '': save_path = path
  else: save_path = self.data_dir + '/Saved_model'

  save_config(self,save_path)
  save_embeddings(self,save_path)
  save_tokenizers(self,save_path)
  save_example_input(self,save_path)