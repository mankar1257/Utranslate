
''' tokenizer class'''
import tensorflow as tf
import numpy as np
import re

from utranslate._src.src_utils.text_helpers import text_from_common_words


class Tokenizer:
    def __init__(self,input_language_common_words_path : str, target_language_common_words_path : str, use_embd_vocab = True):

        self.input_language_common_words_path = input_language_common_words_path
        self.target_language_common_words_path = target_language_common_words_path
        self.use_embd_vocab = use_embd_vocab


    def get_tokenizer(self,lang_type : str,train_lang : tuple,val_lang : tuple ,vocab_size :int):
        '''returns the tockenizers for both the languages '''

        if self.use_embd_vocab:
            texts = text_from_common_words(self,vocab_size,lang_type)
        else:
            texts = []

        #create tokenizer
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
            filters='',oov_token = 0)

        #fit on the text
        lang_tokenizer.fit_on_texts(tuple(texts) + train_lang + val_lang)


        return  lang_tokenizer
