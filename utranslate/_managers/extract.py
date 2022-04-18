import gzip
import shutil
import tarfile
import zipfile


class extractor:
    ''' extractor manager for handeling all downloads '''

    def extract_zip(self,src,dec):
      with zipfile.ZipFile(src, 'r') as zip_ref:
          zip_ref.extractall(dec)


    def extract_tar(self,src,dec):
      my_tar = tarfile.open(src)
      my_tar.extractall(dec)
      my_tar.close()


    def extract_embeddings(self,data_dir : str):

      # extract hindi embedding
      with gzip.open(data_dir+'/cc.hi.300.vec.gz', 'rb') as f_in:
          with open(data_dir+'/Embeddings/input/'+'cc.hi.300.vec', 'wb') as f_out:
              shutil.copyfileobj(f_in, f_out)

      #english embeddings
      src = data_dir+'/crawl-300d-2M.vec.zip'
      dec = data_dir+'/Embeddings/target/'

      self.extract_zip(src,dec)



    def extract_data_files(self,data_dir : str):

      train_src = data_dir+'/parallel.zip'
      test_src = data_dir+'/dev_test.zip'

      dec = data_dir+'/Dataset'

      self.extract_zip(train_src,dec)
      self.extract_zip(test_src,dec)






