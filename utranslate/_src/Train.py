
'''model tranning class'''
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import os
import glob
import time



class Train_model:
    def __init__(self,Encoder,Decoder,Optimizer,checkpoint_dir : str):

        self.Encoder = Encoder
        self.Decoder = Decoder
        self.Optimizer = Optimizer
        self.checkpoint_dir = checkpoint_dir

        self.checkpoint = tf.train.Checkpoint(optimizer=self.Optimizer,
                                        encoder=self.Encoder,
                                        decoder=self.Decoder)


    def loss_function(self,real, pred):

        # real shape = (BATCH_SIZE, max_length_output)
        # pred shape = (BATCH_SIZE, max_length_output, tar_vocab_size )
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        loss = cross_entropy(y_true=real, y_pred=pred)

        #output 0 for y=0 else output 1
        mask = tf.logical_not(tf.math.equal(real,0))
        mask = tf.cast(mask, dtype=loss.dtype)
        loss = mask* loss
        loss = tf.reduce_mean(loss)

        return loss


    @tf.function
    def train_step(self,inp, targ, enc_hidden,BATCH_SIZE,max_length_output):

        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_h, enc_c = self.Encoder(inp, enc_hidden)


            dec_input = targ[ : , :-1 ] # Ignore <end> token
            real = targ[ : , 1: ]       # ignore <start> token

            # Set the AttentionMechanism object with encoder_outputs
            self.Decoder.attention_mechanism.setup_memory(enc_output)

            # Create AttentionWrapperState as initial_state for decoder
            decoder_initial_state = self.Decoder.build_initial_state(BATCH_SIZE, [enc_h, enc_c], tf.float32)
            pred = self.Decoder(dec_input, decoder_initial_state,max_length_output)
            logits = pred.rnn_output
            loss = self.loss_function(real, logits)

        variables = self.Encoder.trainable_variables + self.Decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.Optimizer.apply_gradients(zip(gradients, variables))

        return loss

    def train(self,train_dataset,EPOCHS,steps_per_epoch,BATCH_SIZE,max_length_output,saved_checkpoint = ''):

        if(saved_checkpoint !=''): self.checkpoint.restore(saved_checkpoint)

        Min_loss = 100

        for epoch in range(EPOCHS):
          start = time.time()

          enc_hidden = self.Encoder.initialize_hidden_state()
          total_loss = 0


          for (batch, (inp, targ)) in enumerate(train_dataset.take(steps_per_epoch)):

            batch_loss = self.train_step(inp, targ, enc_hidden,BATCH_SIZE,max_length_output)
            total_loss += batch_loss

            if batch % 100 == 0:
              print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                           batch,
                                                           batch_loss.numpy()))

          if Min_loss > (total_loss / steps_per_epoch) :

            Min_loss  = (total_loss / steps_per_epoch)
            print(f"mininum loss so far : {Min_loss}")

            #delete all privious entries
            files = glob.glob(self.checkpoint_dir+'/*')
            for f in files:
                os.remove(f)

            #add letest entry
            file_prefix = str(self.checkpoint_dir + '/checkpoint')
            pp = self.checkpoint.save(file_prefix)
            print(pp)



          print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                              total_loss / steps_per_epoch))
          print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
