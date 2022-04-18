


from utranslate._src.Train import Train_model
from utranslate._src.Model import Encoder
from utranslate._src.Model import Decoder


import tensorflow as tf
import os



def create_model(self):
    if self.use_embd:
        self.encoder = Encoder(self.units, self.batch_size,embedding_dim = 300,
                           vocab_size =self.vocab_inp_size,
                           embedding_matrix = self.input_emb, embd = self.embd)
    else:
        self.encoder = Encoder(self.units, self.batch_size,embedding_dim = 300,
                           vocab_size =self.vocab_inp_size,use_emdb = False )

    print('===',end = '')
    if self.use_embd:
        self.decoder = Decoder(self.vocab_tar_size, self.units, self.batch_size,
                           self.max_length_input,embedding_matrix = self.target_emb,
                           embd = self.embd,attention_type = 'luong')
    else:
        self.decoder = Decoder(self.vocab_tar_size, self.units, self.batch_size,
                           self.max_length_input,embedding_dim = 300,
                           attention_type = 'luong',use_emdb = False)


def set_up_encoder_decoder(self):

    print('===',end = '')
    sample_hidden = self.encoder.initialize_hidden_state()
    sample_output, sample_h, sample_c = self.encoder(self.example_input_batch, sample_hidden)

    print('===',end = '')
    sample_x = tf.random.uniform((self.batch_size, self.max_length_output))
    self.decoder.attention_mechanism.setup_memory(sample_output)
    initial_state = self.decoder.build_initial_state(self.batch_size, [sample_h, sample_c], tf.float32)

    print('===',end = '')
    sample_decoder_outputs = self.decoder(sample_x, initial_state,self.max_length_output)



def restore_model(self):
    optimizer = tf.keras.optimizers.Adam()

    checkpoint_dir = self.data_dir + '/best_checkpoint'

    print('===',end = '')
    train_model = Train_model(self.encoder,self.decoder,optimizer,checkpoint_dir)

    os.chdir(self.data_dir)

    checkpoint_fname = checkpoint_dir + '/'+ os.listdir(checkpoint_dir)[1].split('.')[0]

    print('======',end = '')
    train_model.checkpoint.restore(checkpoint_fname)


def select_from_default_models(self):
    data_dir = os.environ.get('utranslate_data_path')+'/translation_data'

    i = 0
    print("avelable translation models :\n")
    transletion_list = os.listdir(data_dir)
    for name in transletion_list:
        print(str(i)+ " " + str(name))
        i+=1

    print("\n\n please select the transletion model")
    ans = input()

    try:
        data_dir = data_dir + "/"+ transletion_list[int(ans)]
    except:
        print("\n default transletion selected \n")
        data_dir = data_dir + "/"+transletion_list[0]

    return data_dir