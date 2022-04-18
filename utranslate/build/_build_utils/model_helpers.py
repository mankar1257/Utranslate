
from utranslate._src.Train import Train_model
from utranslate._src.Model import Encoder
from utranslate._src.Model import Decoder

import tensorflow as tf





def initialize_model(self):

    vocab_inp_size = len(self.inp_lang.word_index)+1
    vocab_tar_size = len(self.targ_lang.word_index)+1

    if self.use_embd:

      self.target_emb = self.embd.create_embedding_mat('en',self.targ_lang)
      self.input_emb = self.embd.create_embedding_mat('hi',self.inp_lang)

      self.encoder = Encoder(self.LSTM_UNITS, self.BATCH_SIZE,
                                embedding_matrix = self.input_emb,
                                embd = self.embd , embedding_dim = None,vocab_size = None)

      self.decoder = Decoder(vocab_tar_size, self.LSTM_UNITS,
                                self.BATCH_SIZE,self.max_length_input,
                                embd = self.embd,
                                embedding_matrix = self.target_emb,
                                attention_type = 'luong')

    else:
      self.encoder = Encoder(self.LSTM_UNITS, self.BATCH_SIZE,
                                embedding_dim = 300,vocab_size = vocab_inp_size,
                                use_emdb = False)

      self.decoder = Decoder(vocab_tar_size, self.LSTM_UNITS,
                                self.BATCH_SIZE,self.max_length_input,
                                embedding_dim = 300,
                                attention_type = 'luong',
                                use_emdb = False)

def verify_model(self):

    sample_hidden = self.encoder.initialize_hidden_state()
    sample_output, sample_h, sample_c = self.encoder(self.example_input_batch, sample_hidden)
    print ('\nEncoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
    print ('Encoder h vecotr shape: (batch size, units) {}'.format(sample_h.shape))
    print ('Encoder c vector shape: (batch size, units) {}'.format(sample_c.shape))

    #test the Decoder
    sample_x = tf.random.uniform((self.BATCH_SIZE, self.max_length_output))
    self.decoder.attention_mechanism.setup_memory(sample_output)
    initial_state = self.decoder.build_initial_state(self.BATCH_SIZE, [sample_h, sample_c], tf.float32)

    sample_decoder_outputs = self.decoder(sample_x, initial_state,self.max_length_output)
    print("\nDecoder Outputs Shape: ", sample_decoder_outputs.rnn_output.shape)



def train(self,saved_checkpoint = ''):

      optimizer = tf.keras.optimizers.Adam()
      checkpoint_dir =  self.data_dir + '/Saved_model/best_checkpoint'

      train_model = Train_model(self.encoder,self.decoder,optimizer,checkpoint_dir)

      print("\n\n starting the training ... \n\n")

      train_model.train(self.train_dataset,self.EPOCHS,self.steps_per_epoch,self.BATCH_SIZE,self.max_length_output,saved_checkpoint)
      print("\n\n\t DONE")