
import numpy as np
import pickle
import json







def load_config_file(self):
    with open(self.data_dir + '/config.json') as json_file:
        config = json.load(json_file)

    self.use_embd = config['use_embd']

    return config



def load_tokenizers(self):
    print('\n ===',end = '')
    with open(self.data_dir + '/tokenizer/target_tokenizer.pickle', 'rb') as handle:
      self.targ_lang = pickle.load(handle)
    print('===',end = '')
    with open(self.data_dir +'/tokenizer/input_tokenizer.pickle', 'rb') as handle:
      self.inp_lang = pickle.load(handle)



def load_embedding_matrixs(self):
    if self.use_embd:
        print('===',end = '')
        target_emb = np.load(self.data_dir +'/embedding_matrix/target_embd.npy')
        print('===',end = '')
        input_emb = np.load(self.data_dir +'/embedding_matrix/input_embd.npy')

        return target_emb,input_emb

    return None,None


def load_example_input(self):
    print('===',end = '')
    return np.load(self.data_dir + '/example_input_batch.npy')


