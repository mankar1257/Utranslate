
import os
import requests
import urllib.request

from utranslate._managers.ids import translator_data


class downloader:
    ''' download manager for handeling all downloads '''


    def download_file_from_google_drive(self,id, destination):
    # Reference:- https://github.com/saurabhshri/gdrive-downloader/blob/master/gdrive_downloader.py

      def get_confirm_token(response):
          for key, value in response.cookies.items():
              if key.startswith('download_warning'):
                  return value

          return None


      URL = "https://docs.google.com/uc?export=download"


      def save_response_content(response, destination):
          CHUNK_SIZE = 32768

          with open(destination, "wb") as f:
              for chunk in response.iter_content(CHUNK_SIZE):
                  if chunk: # filter out keep-alive new chunks
                      f.write(chunk)

      session = requests.Session()

      response = session.get(URL, params = { 'id' : id }, stream = True)
      token = get_confirm_token(response)

      if token:
          params = { 'id' : id, 'confirm' : token }
          response = session.get(URL, params = params, stream = True)

      save_response_content(response, destination)


    def download_vocab_freq(self,data_dir:str):
      en_vocabfreq = '14NwkzGTdBmg4eUSfkOkQ83tdyUNhmX5T'
      self.download_file_from_google_drive(en_vocabfreq,data_dir+'/Embeddings/vocabfreq/target_frequent_words.npy')

      hi_vocabfreq = '1-0T3bJX0EGa50SzaeQskyo-lj5m_UT7N'
      self.download_file_from_google_drive(hi_vocabfreq,data_dir+'/Embeddings/vocabfreq/input_frequent_words.npy')




    def download_embeddings(self,data_dir:str, translator:int):
      target_embd = translator_data[translator]['target_embd']
      if target_embd != '':self.download_file_from_google_drive(target_embd, data_dir + '/embedding_matrix/target_embd.npy')

      input_embd = translator_data[translator]['input_embd']
      if input_embd != '':self.download_file_from_google_drive(input_embd, data_dir + '/embedding_matrix/input_embd.npy')





    def download_tokenizer(self,data_dir:str, translator:int):
      target_tokenizer = translator_data[translator]['target_tokenizer']
      self.download_file_from_google_drive(target_tokenizer, data_dir + '/tokenizer/target_tokenizer.pickle')

      input_tokenizer = translator_data[translator]['input_tokenizer']
      self.download_file_from_google_drive(input_tokenizer, data_dir + '/tokenizer/input_tokenizer.pickle')





    def download_example_input_batch(self,data_dir:str, translator:int ):
      example_input_batch =  translator_data[translator]['example_input_batch']
      self.download_file_from_google_drive(example_input_batch, data_dir + '/example_input_batch.npy')




    def download_config_file(self,data_dir:str, translator:int ):
      config = translator_data[translator]['config']
      self.download_file_from_google_drive(config, data_dir + '/config.json')




    def download_checkpoints(self,data_dir:str, translator:int):
      checkpoint_data = translator_data[translator]['checkpoint_data']
      self.download_file_from_google_drive(checkpoint_data, data_dir + '/best_checkpoint/checkpoint.data-00000-of-00001')

      checkpoint_index = translator_data[translator]['checkpoint_index']
      self.download_file_from_google_drive(checkpoint_index, data_dir + '/best_checkpoint/checkpoint.index')

      checkpoint = translator_data[translator]['checkpoint']
      self.download_file_from_google_drive(checkpoint, data_dir + '/best_checkpoint/checkpoint')
