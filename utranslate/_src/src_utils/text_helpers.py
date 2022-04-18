

import numpy as np
import unicodedata
import re



def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w: str, input_lang = True):

    # preprocess input_lang sentance
    if input_lang:
        #remove unwanted charecters
        w = w.replace('\n', '')
        w = w.replace('\r', '')
        w = w.replace('\t', '')
        w = w.replace('।', '')
        w = w.replace('-',' ')

    # preprocess target_lang sentance
    else:
        w = unicode_to_ascii(w.lower().strip())

        # creating a space between a word and the punctuation following it
        # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)

        w = w.replace('\n', '')
        w = w.replace('\r', '')
        w = w.replace('\t', '')
        w = w.replace('।', '')
        w = w.replace('-',' ')

        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

        w = w.strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'

    return w

def read_text_files(self,folder):
    input_text = open(folder+'/input.txt','r').readlines()
    target_text = open(folder+'/target.txt','r').readlines()

    return input_text,target_text


def validate_sentences(self,input_sen,target_sen):
    max_len = 80

    pattern = re.compile(r'([a-zA-Z])')

    if len(pattern.findall(input_sen)) == 0:
        if len(input_sen) < max_len and len(target_sen) < max_len:
            return True

    return False


def text_from_common_words(self,vocab_size : int,lang_type : str):

    texts = []
    i,total_len = 0,0

    if lang_type == "in": frequent_words = np.load(self.input_language_common_words_path)
    else:frequent_words = np.load(self.target_language_common_words_path)


    total_words = len(frequent_words)

    #iterate throung words, create sentance of ten words and add it to text
    while(i<=total_words):

      text = [j for j in frequent_words[i:i+10]]
      total_len += len(text)

      if total_len > vocab_size :
        break

      texts.append(text)
      i+=10

    return texts
