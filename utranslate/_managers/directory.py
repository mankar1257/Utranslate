import pathlib


class directory_handler:
    ''' for creating and managing the directories '''

    def create_directories(self,data_types,path,defalut = False):

        #embeddings directory
        pathlib.Path(path+'/Embeddings').mkdir(parents=True, exist_ok=True)
        pathlib.Path(path+'/Embeddings/target').mkdir(parents=True, exist_ok=True)
        pathlib.Path(path+'/Embeddings/input').mkdir(parents=True, exist_ok=True)
        pathlib.Path(path+'/Embeddings/vocabfreq').mkdir(parents=True, exist_ok=True)

        #data directory
        pathlib.Path(path+'/Dataset').mkdir(parents=True, exist_ok=True)
        pathlib.Path(path+'/Dataset/train').mkdir(parents=True, exist_ok=True)
        pathlib.Path(path+'/Dataset/test').mkdir(parents=True, exist_ok=True)
        pathlib.Path(path+'/Dataset/eval').mkdir(parents=True, exist_ok=True)

        if defalut:
            for folder_name in data_types:
                pathlib.Path(path+'/Dataset/train/' + folder_name).mkdir(parents=True, exist_ok=True)


        #saved_model directory
        pathlib.Path(path+'/Saved_model').mkdir(parents=True, exist_ok=True)
        pathlib.Path(path+'/Saved_model/best_checkpoint').mkdir(parents=True, exist_ok=True)
        pathlib.Path(path+'/Saved_model/embedding_matrix').mkdir(parents=True, exist_ok=True)
        pathlib.Path(path+'/Saved_model/tokenizer').mkdir(parents=True, exist_ok=True)
        pathlib.Path(path+'/Saved_model/vocab').mkdir(parents=True, exist_ok=True)



    def create_translation_data_directory(self,data_dir,translations):
        pathlib.Path(data_dir + '/utranslate_data/translation_data').mkdir(parents=True, exist_ok=True)

        for translation in translations:
            pathlib.Path(data_dir + '/utranslate_data/translation_data/'+str(translation)).mkdir(parents=True, exist_ok=True)
            pathlib.Path(data_dir + '/utranslate_data/translation_data/'+str(translation)+'/best_checkpoint').mkdir(parents=True, exist_ok=True)
            pathlib.Path(data_dir + '/utranslate_data/translation_data/'+str(translation)+'/embedding_matrix').mkdir(parents=True, exist_ok=True)
            pathlib.Path(data_dir + '/utranslate_data/translation_data/'+str(translation)+'/tokenizer').mkdir(parents=True, exist_ok=True)

