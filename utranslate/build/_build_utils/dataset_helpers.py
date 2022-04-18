

from utranslate._src.Data import Dataset

import os



def process_input(self,data_dict,file_path):

    names = []
    ans = input()

    if ans == '':
        names = ['HindEnCorp','Book_Translations_Gyaan_Nidhi_Corpus','TED_talks','Hindi_English_Wordnet_Linkage']

    elif ans == "100":
        names = list(data_dict)
    else:
        try:
            names = [list(data_dict)[int(i)-1] for i in ans.split(',') ]
        except:
            print("pelese provide the input in the correct format")
            return

    for name in names:
        self.train_path.append(file_path+name)

    #setting the test path
    self.test_path = [self.data_dir + '/Dataset/eval/']



def print_available_datasets(self,data_dict):

    data_path = self.data_dir + '/Dataset'+'/train'

    for dir in os.listdir(data_path):
      data_dict[dir] = len(open(data_path + '/'+dir+'/input.txt').readlines())

    i = 1
    for name in data_dict:
        print(f" \t {i}.name : {name} \t num_examples : {data_dict[name]} ")
        i+=1



def select_dataset(self):

    file_path = self.data_dir + '/Dataset/train/'
    data_dict = {}

    print("\n\n  Please select the dataset you want to include in tranning :\n\n")

    print_available_datasets(self,data_dict)

    print("\nPress enter for the default ('HindEnCorp','Book_Translations_Gyaan_Nidhi_Corpus','TED_talks','Hindi_English_Wordnet_Linkage')")
    print("Enter 100 for selecting all the datasets")

    print("\n\t input format : 1,2,3... ")

    process_input(self,data_dict,file_path)



def initialize_dataset(self,default = True):
    if default: select_dataset(self)

    self.dataset_creator = Dataset(self.train_path,self.test_path)
    results = self.dataset_creator.call(self.NUM_EXAMPLE, self.BUFFER_SIZE,
                                self.BATCH_SIZE,self.VOCAB_SIZE,self.tokenize)

    self.train_dataset, self.val_dataset = results['dataset']
    self.inp_lang, self.targ_lang = results['tokenizer']