
"""Embeddings handeler"""
from gensim.models import KeyedVectors
import tensorflow as tf
import numpy as np
import time


class Embeddings:
    def __init__(self,hi_path :str,en_path : str,load_embd = True):

        if load_embd:
          start = time.time()

          print("loading input language embedding...")
          self.input_wtv = KeyedVectors.load_word2vec_format(hi_path)

          print("loading target language embedding...")
          self.target_wtv = KeyedVectors.load_word2vec_format(en_path)

          print(f'total time taken to load the embeddings {time.time()-start}')



    def create_embedding_mat(self,lang_type : str ,lang):

        #set word_to_vec model
        if lang_type == 'in': word_to_vec = self.input_wtv
        else: word_to_vec = self.target_wtv


        emb_len = len(lang.word_index)+1
        emb_dim = 300 # embedding dimention can be changed but here we are using fixed dimentions

        embedding_mat = np.zeros((emb_len, emb_dim))


        #iterate through the words and add its vecotr to matrix
        for i, w in enumerate(lang.word_index):
          try:
              embedding_mat[i] = word_to_vec[str(w)]
          except:
              embedding_mat[i] = np.random.normal(scale=0.6, size=(emb_dim, ))

        return embedding_mat


    def get_embedding_leyar(self,embedding_matrix):
      num_tokens, embedding_dim = embedding_matrix.shape

      return tf.keras.layers.Embedding(
        num_tokens,
        embedding_dim,
        embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
        trainable=False,
      )
