"""
Ryan Tietjen
Sep 2024
Contains a script that produces the most successful model
"""

import tensorflow as tf
import numpy as np

from keras import layers
from tensorflow.keras.layers import TextVectorization 
import pandas as pd
import configparser
from data_processing import split_text_into_abstracts
from data_processing import one_hot_encode_labels
from data_processing import encode_labels
from model_creation import create_model
from utils import split_sentences_by_characters
import tensorflow_hub as hub
from utils import classification_metrics


"""
Setup config file
"""
config = configparser.ConfigParser()
config.read('config.ini')


"""
Import/process data
"""
dir = config["Data"]["dir"]
train_data = split_text_into_abstracts(dir, "train.txt")
validation_data = split_text_into_abstracts(dir, "dev.txt")
test_data = split_text_into_abstracts(dir, "test.txt") 


if config["Model"].getboolean("verbose"):
    print("finish with data?")


train_df = pd.DataFrame(train_data)
validation_df = pd.DataFrame(validation_data)
test_df = pd.DataFrame(test_data)


"""
One-hot encode the labels

Why one-hot encode?
Imagine our labels looked like [5, 3, 3, 1, 0]
Then the model *might* think that there is some numerical relationship among labels, when there is not. 
"""
one_hot_train_labels, one_hot_validation_labels, one_hot_test_labels = one_hot_encode_labels(train_df["target"], validation_df["target"], test_df["target"])

train_labels, validation_labels, test_labels = encode_labels(train_df["target"], validation_df["target"], test_df["target"]) # Used for testing at the very end

train_sentences = train_df["text"].tolist()
validation_sentences = validation_df["text"].tolist()
test_sentences = test_df["text"].tolist()

print("finish with labels?")

"""
Create an embedding layer to embed each token (word) in each abstract
"""
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 
tf_hub_embedding_layer = hub.KerasLayer(module_url,
                                        trainable=False,
                                        name="universal_sentence_encoder")

if config["Model"].getboolean("verbose"):
    print("finish with tf hub?")

"""
Split the abstracters by character, then embed each token (character)
"""
train_data_by_character = split_sentences_by_characters(train_sentences)
validation_data_by_character = split_sentences_by_characters(validation_sentences)
test_data_by_character = split_sentences_by_characters(test_sentences)

output_seq_char_len = int(np.percentile([len(sentence) for sentence in train_sentences], 98))
char_vectorizer = TextVectorization(max_tokens=72,  
                                    output_sequence_length=output_seq_char_len,
                                    standardize="lower_and_strip_punctuation")
char_vectorizer.adapt(train_data_by_character)


character_embedding = layers.Embedding(input_dim=72, # length of vocabulary
                               output_dim=25, #from figure 1 in https://arxiv.org/pdf/1612.05251.pdf 
                               mask_zero=False,
                               name="character_embedding")


"""
Create positional embeddings (one-hot)
"""
if config["Model"].getboolean("verbose"):
    print("positional embeds--------------------------------------------------------------------------------------------------------------------------")
train_lines_one_hot = tf.one_hot(train_df["line_number"].to_numpy(), depth=int(config["Model"]["lines"]))
validation_lines_one_hot = tf.one_hot(validation_df["line_number"].to_numpy(), depth=int(config["Model"]["lines"]))
test_lines_one_hot = tf.one_hot(test_df["line_number"].to_numpy(), depth=int(config["Model"]["lines"]))


"""
Create embeddings that note the total number of lines in the abstract(one-hot)
"""
if config["Model"].getboolean("verbose"):
    print("line embeds--------------------------------------------------------------------------------------------------------------------------")
train_total_lines_one_hot = tf.one_hot(train_df["total_lines"].to_numpy(), depth=int(config["Model"]["total_lines"]))
validation_total_lines_one_hot = tf.one_hot(validation_df["total_lines"].to_numpy(), depth=int(config["Model"]["total_lines"]))
test_total_lines_one_hot = tf.one_hot(test_df["total_lines"].to_numpy(), depth=int(config["Model"]["total_lines"]))

"""
Compile data into Datasets 
"""
if config["Model"].getboolean("verbose"):
    print("start dataset --------------------------------------------------------------------------------------------------------------------------")

model_data_train = tf.data.Dataset.from_tensor_slices((train_lines_one_hot, train_total_lines_one_hot, train_sentences, train_data_by_character)) 
model_labels_train = tf.data.Dataset.from_tensor_slices(one_hot_train_labels) 
model_dataset_train = tf.data.Dataset.zip((model_data_train, model_labels_train)).batch(32).prefetch(tf.data.AUTOTUNE)

model_data_validation = tf.data.Dataset.from_tensor_slices((validation_lines_one_hot, validation_total_lines_one_hot, validation_sentences, validation_data_by_character))
model_labels_validation = tf.data.Dataset.from_tensor_slices(one_hot_validation_labels)
model_dataset_validation = tf.data.Dataset.zip((model_data_validation, model_labels_validation)).batch(32).prefetch(tf.data.AUTOTUNE)

model_data_test = tf.data.Dataset.from_tensor_slices((test_lines_one_hot, test_total_lines_one_hot, test_sentences, test_data_by_character))
model_labels_test = tf.data.Dataset.from_tensor_slices(one_hot_test_labels)
model_dataset_test = tf.data.Dataset.zip((model_data_test, model_labels_test)).batch(32).prefetch(tf.data.AUTOTUNE)

if config["Model"].getboolean("verbose"):
    print("done dataset")

"""
Create model
"""
model = create_model(4, 5, tf_hub_embedding_layer, character_embedding, char_vectorizer)


unique_labels = train_df['target'].unique()
print("Unique labels in the dataset:", unique_labels)

"""
Fit the model
"""
if config["Model"].getboolean("verbose"):
    print("fitting")
model_history = model.fit(model_dataset_train,
                              epochs=int(config["Model"]["epochs"]),
                              validation_data=model_dataset_validation)
if config["Model"].getboolean("verbose"):
    print("done fitting")

"""
Get results
"""
model_results = classification_metrics(tf.argmax(model.predict(model_dataset_test), axis = 1),
                                        test_labels)
print(model_results)

"""
Save the model
"""
if config["Model"].getboolean("save_model"):
    model.save("test.keras")
    if config["Model"].getboolean("verbose"):
        print("model saved")