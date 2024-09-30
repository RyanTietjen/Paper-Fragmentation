"""
Ryan Tietjen
Sep 2024
Create best model for the demo
"""
import tensorflow as tf
from keras import layers
import tensorflow_hub as hub

class EmbeddingLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Hardcode the module URL directly within the layer
        # self.module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        self.embed_model = hub.KerasLayer(url, trainable=False, name="universal_sentence_encoder")

    def call(self, inputs):
        return self.embed_model(inputs)

    def get_config(self):
        config = super().get_config()
        # The URL is now a fixed part of the layer, so it can be included in the config for completeness
        # config.update({'module_url': self.module_url})
        return config

def create_token_model(token_embed):
    input_layer = layers.Input(shape=[], dtype=tf.string)
    embedding_layer = EmbeddingLayer()
    token_embeddings = embedding_layer(input_layer)
    output_layer = layers.Dense(128, activation="relu")(token_embeddings)
    model = tf.keras.Model(input_layer, output_layer)
    return model

def create_character_vectorizer_model(char_embed, char_vectorizer):
    input_layer = layers.Input(shape=(1,), dtype=tf.string)
    char_vectors = char_vectorizer(input_layer) # vectorize text inputs
    char_embedding = char_embed(char_vectors) # create embedding
    output_layer = layers.Bidirectional(layers.LSTM(32))(char_embedding)
    model = tf.keras.Model(input_layer, output_layer)
    return model

def create_line_number_model(input_shape, name):
    input_layer = layers.Input(shape=(input_shape,), dtype=tf.int32, name=name)
    output_layer = layers.Dense(32, activation="relu")(input_layer)
    model = tf.keras.Model(input_layer, output_layer)
    return model


def tribrid_model(num_classes, token_embed, char_embed, text_vectorizer):
    
    token_model = create_token_model(token_embed)
    character_vectorizer_model = create_character_vectorizer_model(char_embed, text_vectorizer)
    line_number_model = create_line_number_model(15, "line_number")
    total_lines_model = create_line_number_model(20, "total_lines")

    hybrid_model = layers.Concatenate(name="hybrid")([token_model.output,
                                                      character_vectorizer_model.output])
    
    dense_layer = layers.Dense(256, activation="relu")(hybrid_model)
    dense_layer = layers.Dropout(0.5)(dense_layer)

    tribrid_model = layers.Concatenate(name="tribrid") ([line_number_model.output, total_lines_model.output, dense_layer])
    output_layer = layers.Dense(num_classes, activation="softmax")(tribrid_model)

    model = tf.keras.Model([line_number_model.input, total_lines_model.input, token_model.input, character_vectorizer_model.input], output_layer)

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])
    return model