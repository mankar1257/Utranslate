
''' custom model creation '''
from utranslate._src.Embeddings import Embeddings
from utranslate._src.Tokenizer import Tokenizer
from utranslate._src.Train import Train_model
from utranslate._src.Data import Dataset
from utranslate._src.Model import Encoder
from utranslate._src.Model import Decoder

import tensorflow as tf
import numpy as np
import pickle
import json
import os

from utranslate.build._build_utils.dataset_helpers import initialize_dataset
from utranslate.build._build_utils.save_helpers import save
from utranslate.build._build_utils.model_helpers import initialize_model,verify_model,train
from utranslate.build._build_utils.helpers import set_hyperpram


class model:
    def __init__(self):

        self.data_dir = os.environ.get('utranslate_data_path') + '/custom_data'
        self.tokenize = Tokenizer('','',use_embd_vocab=False)
        self.use_embd = False


    def set_embeddings(self):

        emb_dir = self.data_dir + '/Embeddings'

        #load embeddings if exeisit
        try:
            inp_lang_embd = emb_dir + '/input/' + os.listdir(emb_dir + '/input')[0]
            targ_lang_embd = emb_dir + '/target/' + os.listdir(emb_dir + '/target')[0]

            self.embd = Embeddings(inp_lang_embd,targ_lang_embd)

        except:
            print("no embeddings found !!")
            return

        #load vocabfreq if exisist
        try:
            inp_lang_vocabfreq = emb_dir + '/vocabfreq/' + os.listdir(emb_dir + '/vocabfreq')[0]
            tar_lang_vocabfreq = emb_dir + '/vocabfreq/' + os.listdir(emb_dir + '/vocabfreq')[1]

            self.tokenize = Tokenizer(inp_lang_vocabfreq,targ_lang_vocabfreq)
        except:
            print("no language vocab found !!")

        #enable embd use
        self.use_embd = True


    def set_dataset(self):
        self.train_path = [self.data_dir + '/Dataset/train']
        self.test_path = [self.data_dir + '/Dataset/eval']


    def initialize(self,use_default = False, hparms = {}):

        print("\t starting the custom model...\n\n")

        #initilize
        set_hyperpram(self,use_default,hparms)
        self.set_dataset()
        self.set_embeddings()

        self.BUFFER_SIZE = 42000

        #create dataset
        initialize_dataset(self,False)

        #take one batch
        self.example_input_batch, example_target_batch = next(iter(self.train_dataset))
        self.max_length_input = self.example_input_batch.shape[1]
        self.max_length_output = example_target_batch.shape[1]
        self.steps_per_epoch = self.dataset_creator.tranning_size//self.BATCH_SIZE

        #model
        initialize_model(self)

        verify_model(self)

        print("\n\t DONE")


    def train(self,saved_checkpoint = ''):
      train(self,saved_checkpoint)

    def save(self,path = ''):
      save(self,path)

