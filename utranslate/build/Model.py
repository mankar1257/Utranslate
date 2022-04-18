''' default model creation '''
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
from utranslate.build._build_utils.helpers import set_hyperpram,initialize_embeddings



class model:

    def __init__(self,use_embd = True):

      self.data_dir = os.environ.get('utranslate_data_path') + '/default_data'

      self.use_embd = use_embd

      hi_vocab = self.data_dir+'/Embeddings/vocabfreq/input_frequent_words.npy'
      en_vocab = self.data_dir+'/Embeddings/vocabfreq/target_frequent_words.npy'

      #create tokenizer
      self.tokenize = Tokenizer(hi_vocab,en_vocab)

      self.train_path = []
      self.test_path = []



    def initialize(self, use_default=False, hparms = {}):

      print("\t setting up the model ...\n\n")
      self.BUFFER_SIZE = 42000

      set_hyperpram(self,use_default,hparms)

      initialize_embeddings(self)

      #data
      initialize_dataset(self)

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
