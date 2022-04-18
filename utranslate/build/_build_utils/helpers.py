
''' helper functions for the build module'''

from utranslate._src.Embeddings import Embeddings



def initialize_embeddings(self):
    if self.use_embd:
      print("\n\n")
      h_wtv = self.data_dir + '/Embeddings/input/cc.hi.300.vec'
      e_wtv  = self.data_dir + '/Embeddings/target/crawl-300d-2M.vec'

      self.embd = Embeddings(h_wtv,e_wtv)
      print("\n\n")
    else:
      self.embd = Embeddings('','',False)



def get_hyperpram_from_user(self):

    print("\n\n Please select the hyperparameters \n\n ")

    print("Please enter the bach size : (press enter for default, #default : 64 )")
    batch_size = input()
    if batch_size != '':
      self.BATCH_SIZE = int(batch_size)


    print("Please enter the vocab size : (press enter for default, #default : 10000 )")
    vocab_size = input()
    if vocab_size != '':
      self.VOCAB_SIZE = int(vocab_size)


    print("Please enter the number of lstm units : (press enter for default, #default: 512 )")
    lstm_units = input()
    if lstm_units != '':
      self.LSTM_UNITS = int(lstm_units)


    print("Please enter the number of epochs : (press enter for default, #default: 10 )")
    epoch = input()
    if epoch != '':
      self.EPOCHS = int(epoch)


    print("Please enter the number of examples to use in training : (press enter for default, #default: complete dataset )")
    num_example = input()
    if num_example != '':
      self.NUM_EXAMPLE = int(num_example)



def set_hyperpram(self,use_default = False,data = {}):

    self.BATCH_SIZE = 64
    self.VOCAB_SIZE = 10000
    self.LSTM_UNITS = 512
    self.EPOCHS = 10
    self.NUM_EXAMPLE = 2000000

    if data != {}:
        self.BATCH_SIZE = data['BATCH_SIZE']
        self.VOCAB_SIZE = data['VOCAB_SIZE']
        self.LSTM_UNITS = data['LSTM_UNITS']
        self.EPOCHS = data['EPOCHS']
        self.NUM_EXAMPLE = data['NUM_EXAMPLE']

    elif not use_default:
        get_hyperpram_from_user(self)




