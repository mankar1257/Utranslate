"""Data-interaction maneger calss"""
import tensorflow as tf
import numpy as np
import unicodedata
import random
import re

from utranslate._src.src_utils.text_helpers import read_text_files,validate_sentences,preprocess_sentence
from utranslate._src.src_utils.data_helpers import  tokenize_data,convert_to_tf_dataset


class Dataset:
    def __init__(self,train_path,test_path):
        ''' Data loaiding, cleanning and preprocessing '''
        self.train_path = train_path
        self.test_path = test_path

        #initilize tokenizers
        self.inp_lang_tokenizer = None
        self.targ_lang_tokenizer = None

        #initilize tranning_size
        self.tranning_size = None



    def load_data(self,num_examples = None, train = True):

        if train: folders = self.train_path
        else: folders = self.test_path

        word_pairs = []

        #iterate through flders and files and load the data
        for folder in folders:
            input_text,target_text = read_text_files(self,folder)

            for input_sen,target_sen in zip(input_text, target_text):

                if validate_sentences(self,input_sen,target_sen):
                    input_sen = preprocess_sentence(input_sen)
                    target_sen = preprocess_sentence(target_sen,False)

                    word_pairs.append([input_sen,target_sen])

            if num_examples and len(word_pairs) > num_examples:
                break

        #set the tranning size
        if train: self.tranning_size = len(word_pairs)

        return zip(*word_pairs)



    def get_data(self,tokenize,vocab_size,num_examples):

        #load data
        train_inp_lang,train_targ_lang = self.load_data(num_examples)
        test_inp_lang ,test_targ_lang = self.load_data(train = False)

        # tokenize data
        input_tensor_train, input_tensor_val  = tokenize_data(self,'in',
                                                            train_inp_lang,
                                                            test_inp_lang,
                                                            vocab_size,
                                                            tokenize)
        target_tensor_train, target_tensor_val, = tokenize_data(self,'out',
                                                            train_targ_lang,
                                                            test_targ_lang,
                                                            vocab_size,
                                                            tokenize)

        results = {"train_data" : (input_tensor_train,target_tensor_train),
                    "val_data" : (input_tensor_val,target_tensor_val)}


        return results


    def call(self, num_examples, BUFFER_SIZE, BATCH_SIZE,vocab_size,tokenize):

        #get data
        data = self.get_data(tokenize,vocab_size,num_examples)

        #create tf dataset
        train_dataset =convert_to_tf_dataset(self,data['train_data'],BUFFER_SIZE, BATCH_SIZE)
        test_dataset =convert_to_tf_dataset(self,data['val_data'],BUFFER_SIZE, BATCH_SIZE)


        results = {"dataset" : (train_dataset, test_dataset),
                    "tokenizer" : (self.inp_lang_tokenizer, self.targ_lang_tokenizer)}

        return results
