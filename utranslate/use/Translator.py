
''' transletor class'''
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import time
import pickle
import json
import os


from utranslate.use._use_utils.model_helpers import select_from_default_models
from utranslate.use._use_utils.model_helpers import create_model
from utranslate.use._use_utils.model_helpers import set_up_encoder_decoder
from utranslate.use._use_utils.model_helpers import restore_model


from utranslate.use._use_utils.load_helpers import load_config_file
from utranslate.use._use_utils.load_helpers import load_tokenizers
from utranslate.use._use_utils.load_helpers import load_embedding_matrixs
from utranslate.use._use_utils.load_helpers import load_example_input


from utranslate._src.src_utils.text_helpers import preprocess_sentence



from utranslate._src.Embeddings import Embeddings
from utranslate._src.Tokenizer import Tokenizer
from utranslate._src.Train import Train_model
from utranslate._src.Data import Dataset
from utranslate._src.Model import Encoder
from utranslate._src.Model import Decoder


#supress the warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class translator:
    def __init__(self,saved_model = ''):
        # if saved model not given then continue with the default models abveleble
        if saved_model != '':
          self.data_dir = saved_model
        else:
          self.data_dir = select_from_default_models(self)


        start = time.time()
        print("\nStarting the U_transletor ")
        print("\t may take some time...")


        config = load_config_file(self)

        


        load_tokenizers(self)


        self.target_emb,self.input_emb = load_embedding_matrixs(self)

        #create the Embeddings object
        self.embd = Embeddings(" "," ",load_embd = False)

        self.example_input_batch = load_example_input(self)


        self.batch_size = config['batch_size']
        self.vocab_inp_size = len(self.inp_lang.word_index)+1
        self.vocab_tar_size = len(self.targ_lang.word_index)+1

        self.max_length_input = config['max_length_input']
        self.max_length_output = config['max_length_output']

        self.units = config['units']

        print('===',end = '')

        create_model(self)

        set_up_encoder_decoder(self)


        restore_model(self)

        end = time.time()

        print(' !')
        print(f'\n U_transletor started  \n\t total time --> : {int((end-start)//60)} min and {int((end-start)%60)} sec')


    def evaluate_sentence(self,sentence):
        if ord(sentence.split(' ')[0][0]) in range(65,123):#check if the input sentance in english language
          sentence =  preprocess_sentence(sentence,False)
        else:
          sentence =  preprocess_sentence(sentence)


        inputs = [self.inp_lang.word_index[i] for i in sentence.split(' ')]

        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                              maxlen=self.max_length_input,
                                                              padding='post')


        inputs = tf.convert_to_tensor(inputs)

        inference_batch_size = inputs.shape[0]
        result = ''

        enc_start_state = [tf.zeros((inference_batch_size, self.units)), tf.zeros((inference_batch_size,self.units))]
        enc_out, enc_h, enc_c = self.encoder(inputs, enc_start_state)


        dec_h = enc_h
        dec_c = enc_c

        start_tokens = tf.fill([inference_batch_size], self.targ_lang.word_index['<start>'])

        end_token = self.targ_lang.word_index['<end>']


        greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()

        # Instantiate BasicDecoder object
        decoder_instance = tfa.seq2seq.BasicDecoder(cell=self.decoder.rnn_cell, sampler=greedy_sampler, output_layer=self.decoder.fc)
        # Setup Memory in decoder stack

        self.decoder.attention_mechanism.setup_memory(enc_out)

        # set decoder_initial_state
        decoder_initial_state = self.decoder.build_initial_state(inference_batch_size, [enc_h, enc_c], tf.float32)


        ### Since the BasicDecoder wraps around Decoder's rnn cell only, you have to ensure that the inputs to BasicDecoder
        ### decoding step is output of embedding layer. tfa.seq2seq.GreedyEmbeddingSampler() takes care of this.
        ### You only need to get the weights of embedding layer, which can be done by decoder.embedding.variables[0] and pass this callabble to BasicDecoder's call() function

        decoder_embedding_matrix = self.decoder.embedding.variables[0]


        outputs, _, _ = decoder_instance(decoder_embedding_matrix, start_tokens = start_tokens, end_token= end_token, initial_state=decoder_initial_state)

        return outputs.sample_id.numpy()

    def translate(self,sentence):
        result = self.evaluate_sentence(sentence)
        result = self.targ_lang.sequences_to_texts(result)
        return str(result[0].split('<end>')[0])