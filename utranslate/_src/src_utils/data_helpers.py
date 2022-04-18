
import tensorflow as tf


def tokenize_data(self, lang_type : str, train_lang : list ,val_lang :list, vocab_size : int, tokenize):

    # get tokenizer
    lang_tokenizer = tokenize.get_tokenizer(lang_type,train_lang,val_lang,
                                                            vocab_size)

    # tokenize data
    train_tensor = lang_tokenizer.texts_to_sequences(train_lang)
    test_tensor = lang_tokenizer.texts_to_sequences(val_lang)

    # padd data
    train_tensor = tf.keras.preprocessing.sequence.pad_sequences(train_tensor,
                                                            padding='post')
    test_tensor = tf.keras.preprocessing.sequence.pad_sequences(test_tensor,
                                                            padding='post')

    #set the trokenizers
    if lang_type == "in": self.inp_lang_tokenizer = lang_tokenizer
    else: self.targ_lang_tokenizer = lang_tokenizer


    return train_tensor,test_tensor


def convert_to_tf_dataset(self,data,BUFFER_SIZE,BATCH_SIZE):
    dataset = tf.data.Dataset.from_tensor_slices(data)

    return dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
