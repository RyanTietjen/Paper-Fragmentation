"""
Ryan Tietjen
Sep 2024
Contains helper functions related to creating various machine learning models
"""

import tensorflow as tf
from keras import layers
import tensorflow_hub as hub

def create_model(model_type, num_classes, token_embed=None, character_embed=None, text_vectorizer=None):
    """
    Creates a machine learning model based on the specified model type and configurations.

    This function maps a model type identifier to a specific model creation function, which is then called
    with the provided parameters. The model type determines the architecture and the method of handling input
    data. If the model type is not recognized, it raises an exception.

    Parameters:
    model_type (int): An integer representing the type of model to create. The valid identifiers are:
                      1 - CONV1d with token embeddings
                      2 - Feature extraction with pretrained token embeddings
                      3 - Hybrid model combining multiple types of input
                      4 - Tribrid model integrating three different input methodologies
    num_classes (int): The number of classes for the model's output layer.
    token_embed (optional): Configuration or layers for token embedding, defaults to None.
    character_embed (optional): Configuration or layers for character embedding, defaults to None.
    text_vectorizer (optional): Text vectorization configurations or layers, defaults to None.

    Returns:
    A model instance as defined by the selected type and configurations.

    Raises:
    Exception: If an invalid model type is provided.
    """
    
    # Mapping of model types to their corresponding functions
    model_dispatch = {
        1: CONV1d_with_token_embeddings,
        2: feature_extraction_with_pretrained_token_embeddings,
        3: hybrid_model,
        4: tribrid_model
    }

    # Retrieve the model creation function based on the model type
    model_func = model_dispatch.get(model_type)

    # If a function is found, call it with the provided parameters
    if model_func:
        return model_func(num_classes, token_embed, character_embed, text_vectorizer)
    else:
        # Raise an exception if the model type is not recognized
        raise Exception("Invalid model type provided")

def CONV1d_with_token_embeddings(num_classes, token_embed, character_embed, text_vectorizer):
    """
    Constructs a 1D Convolutional Neural Network model using token embeddings.

    This model is designed for text classification tasks. It starts by vectorizing text inputs,
    applying token embeddings, and then processing the resulting embeddings through a convolutional
    layer followed by global average pooling. The output is passed through a softmax layer to
    classify into multiple classes.

    Parameters:
    num_classes (int): The number of classes for the model's output layer.
    token_embed (Layer): A Keras layer or similar callable that applies token embeddings to the vectorized text.
    character_embed (ignored): This parameter is not used in this model but is included for interface compatibility.
    text_vectorizer (Layer): A Keras layer or similar callable that vectorizes raw text inputs.

    Returns:
    tf.keras.Model: A compiled Keras model ready for training.

    Notes:
    - The `character_embed` parameter is included to maintain a consistent function signature with other model
      creation functions but is not used in this particular model configuration.
    """
    
    # Define the input layer for raw text data
    input_layer = layers.Input(shape=(1,), dtype=tf.string)
    
    # Apply the text vectorizer to transform text input to token indices
    text_vectors = text_vectorizer(input_layer)
    
    # Apply the token embedding layer to get dense vector representations of tokens
    token_embeddings = token_embed(text_vectors)
    
    # Define a 1D convolution layer to capture local dependencies in the embeddings
    conv_layer = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(token_embeddings)
    
    # Add a global average pooling layer to reduce dimensionality and focus on the most important features
    pooling_layer = layers.GlobalAveragePooling1D()(conv_layer)
    
    # Define the output layer with a softmax activation for multi-class classification
    output_layer = layers.Dense(num_classes, activation="softmax")(pooling_layer)
    
    # Construct the model
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    
    # Compile the model with appropriate loss function, optimizer, and metrics
    model.compile(loss="categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])
    
    return model


class EmbeddingLayer(layers.Layer):
    """
    A custom Keras layer that integrates the Universal Sentence Encoder from TensorFlow Hub into a Keras model.
    
    This layer automatically downloads and initializes the Universal Sentence Encoder, making it easy
    to integrate sentence embedding capabilities directly into a TensorFlow Keras model. The embeddings
    are not trainable, ensuring that the pre-trained weights are preserved during training.
    
    Attributes:
        embed_model (hub.KerasLayer): The TensorFlow Hub Keras layer that wraps the Universal Sentence Encoder.
    """

    def __init__(self, **kwargs):
        """
        Initializes the EmbeddingLayer with the Universal Sentence Encoder.
        """
        super().__init__(**kwargs)
        # The URL to the Universal Sentence Encoder TensorFlow Hub module
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        self.embed_model = hub.KerasLayer(module_url, trainable=False, name="universal_sentence_encoder")

    def call(self, inputs):
        """
        Processes the input sentences into embeddings.
        
        Args:
            inputs: A tensor of strings of shape (batch_size,) containing the sentences to embed.
        
        Returns:
            A tensor of shape (batch_size, embedding_dim) with the sentence embeddings.
        """
        return self.embed_model(inputs)

    def get_config(self):
        """
        Returns the configuration of the layer for serialization.
        
        Includes the TensorFlow Hub URL to allow models including this layer to be reloaded with the same configuration.
        """
        config = super().get_config()
        config.update({'module_url': self.embed_model.handle})  # handle provides the URL used for the layer
        return config



def feature_extraction_with_pretrained_token_embeddings(num_classes, token_embed=None, character_embed=None, text_vectorizer=None):
    """
    Creates a text classification model using pretrained token embeddings from the Universal Sentence Encoder.

    Parameters:
    num_classes (int): Number of output classes.
    token_embed (Callable, optional): Callable layer to embed tokenized inputs. Not used in this configuration.
    character_embed (Callable, optional): Callable layer to embed character inputs. Not used in this configuration.
    text_vectorizer (Callable, optional): Callable layer to vectorize text inputs. Not used in this configuration.

    Returns:
    tf.keras.Model: A compiled Keras model ready for training.
    """
    # Define the input layer for raw text data
    input_layer = layers.Input(shape=[], dtype=tf.string)

    # Embedding layer using Universal Sentence Encoder
    embedding_layer = EmbeddingLayer()

    # Process input text through the embedding layer
    token_embeddings = embedding_layer(input_layer)

    # Dense layer for feature transformation
    dense_layer = layers.Dense(128, activation='relu')(token_embeddings)

    # Output layer for classification
    output_layer = layers.Dense(num_classes, activation='softmax')(dense_layer)

    # Construct and compile the Keras model
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss="categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])
    return model



def create_token_model(token_embed):
    """
    Constructs a TensorFlow Keras model that embeds textual input using a specified embedding layer and
    applies a dense layer to the embeddings.

    This function is designed to take a pre-existing embedding layer, embed incoming text data, and then
    process the embeddings through a fully connected dense layer to transform the embedded features.

    Parameters:
    token_embed (tf.keras.layers.Layer): An embedding layer that is initialized outside this function and
                                         is used to convert text inputs into embeddings. It should be 
                                         compatible with the input specifications of the model.

    Returns:
    tf.keras.Model: A TensorFlow Keras model consisting of an input layer, the provided embedding layer, 
                    a dense layer, and capable of processing textual data into dense embeddings.

    Note:
    The function currently does not use the 'token_embed' parameter. Future implementations should ensure
    that 'token_embed' is utilized as the embedding layer or modify the function to remove the parameter if
    it remains unused.
    """
    input_layer = layers.Input(shape=[], dtype=tf.string)
    embedding_layer = EmbeddingLayer()
    token_embeddings = embedding_layer(input_layer)
    output_layer = layers.Dense(128, activation="relu")(token_embeddings)
    model = tf.keras.Model(input_layer, output_layer)
    return model

def create_character_vectorizer_model(char_embed, char_vectorizer):
    """
    Constructs a TensorFlow Keras model designed to process text inputs at the character level using
    specified vectorization and embedding layers, followed by a bidirectional LSTM layer.

    This function sets up a neural network architecture for character-level text processing. It first
    vectorizes the character sequences using a provided vectorizer, then applies an embedding layer to
    get dense vector representations of these sequences, and finally processes these representations
    through a bidirectional LSTM to capture dependencies from both directions of the sequence.

    Parameters:
    char_embed (tf.keras.layers.Layer): A Keras layer that takes integer-encoded character sequences
                                       and returns embeddings for these sequences. Typically, this would
                                       be an instance of `layers.Embedding`.
    char_vectorizer (tf.keras.layers.Layer): A Keras layer that takes raw string inputs and converts
                                            them into integer-encoded character sequences. This could be
                                            an instance of `layers.TextVectorization` configured for character
                                            level vectorization.

    Returns:
    tf.keras.Model: A Keras model comprising an input layer, a character vectorization layer, a character
                    embedding layer, and a bidirectional LSTM layer. This model is suitable for processing
                    sequences of character-level text data, such as for tasks involving sentiment analysis,
                    text classification, or other NLP tasks that benefit from understanding character-level
                    nuances.

    Example:
    ```python
    # Assuming the necessary layers have been defined and imported
    char_vectorizer = TextVectorization(output_mode='int', ngrams=None, split='character')
    char_embed = Embedding(input_dim=100, output_dim=8, input_length=50)
    model = create_character_vectorizer_model(char_embed, char_vectorizer)
    model.summary()
    ```
    """
    input_layer = layers.Input(shape=(1,), dtype=tf.string)
    char_vectors = char_vectorizer(input_layer)  # Vectorize text inputs
    char_embedding = char_embed(char_vectors)  # Create embeddings from character vectors
    output_layer = layers.Bidirectional(layers.LSTM(32))(char_embedding)
    model = tf.keras.Model(input_layer, output_layer)
    return model

def create_line_number_model(input_shape, name):
    """
    Creates a simple TensorFlow Keras model for processing numerical data using a single dense layer.

    This function is primarily designed to handle input data that can be represented as a fixed-length
    sequence of integers, such as line numbers or other numerical indices. The model comprises a single
    dense layer with ReLU activation, making it suitable for tasks that may benefit from a basic
    transformation of numerical inputs.

    Parameters:
    input_shape (int): The size of the input layer, specifying the number of features (dimensions) in
                       the input data. This should be an integer representing the length of input arrays.
    name (str): A name for the input layer, which can help in identifying the layer within the model,
                especially useful in complex models where multiple input layers might be used.

    Returns:
    tf.keras.Model: A TensorFlow Keras model that includes an input layer and a dense layer. The model
                    is configured for straightforward applications and can be extended or integrated
                    into more complex architectures if needed.

    Example:
    ```python
    # Define the model with input shape 10 and a specific name for the input layer
    model = create_line_number_model(10, 'line_number_input')
    model.summary()
    ```
    """
    input_layer = layers.Input(shape=(input_shape,), dtype=tf.int32, name=name)
    output_layer = layers.Dense(32, activation="relu")(input_layer)
    model = tf.keras.Model(input_layer, output_layer)
    return model

def hybrid_model(num_classes, token_embed, char_embed, text_vectorizer):
    """
    Constructs a hybrid TensorFlow Keras model that combines token and character-level embeddings with
    a deep learning architecture for text classification.

    This function integrates separate models for token and character-level embeddings, concatenates their
    outputs, and passes the combined feature set through dropout and dense layers for classification. It
    is intended for applications where both token-level and character-level understanding of text is crucial,
    such as in sentiment analysis or complex text classification scenarios.

    Parameters:
    num_classes (int): The number of classes for the classification task. Determines the output dimension
                       of the final dense layer with softmax activation.
    token_embed (tf.keras.layers.Layer): An embedding layer used in the token model to convert text inputs
                                         into token embeddings.
    char_embed (tf.keras.layers.Layer): An embedding layer used in the character vectorizer model to
                                        convert text inputs into character embeddings.
    text_vectorizer (tf.keras.layers.Layer): A text vectorization layer used in the character vectorizer
                                             model to convert raw text into encoded character data.

    Returns:
    tf.keras.Model: A compiled Keras model that is ready for training. The model includes input layers
                    for token and character data, concatenation of the two types of embeddings, dropout
                    and dense layers for processing, and a final classification layer.

    Example:
    ```python
    # Assume necessary layers and number of classes are defined
    token_embed = Embedding(input_dim=5000, output_dim=300)
    char_embed = Embedding(input_dim=100, output_dim=50)
    text_vectorizer = TextVectorization(max_tokens=1000, output_mode='int', split='character')
    model = hybrid_model(10, token_embed, char_embed, text_vectorizer)
    model.summary()
    ```
    """
    token_model = create_token_model(token_embed)
    character_vectorizer_model = create_character_vectorizer_model(char_embed, text_vectorizer)

    hybrid_layer = layers.Concatenate(name="hybrid")([token_model.output,
                                                      character_vectorizer_model.output])
    
    dropout_layer = layers.Dropout(0.5)(hybrid_layer)
    dense_layer = layers.Dense(200, activation="relu")(dropout_layer)  # Adjusted for different embedding shapes
    final_dropout = layers.Dropout(0.5)(dense_layer)
    output_layer = layers.Dense(num_classes, activation="softmax")(final_dropout)

    model = tf.keras.Model([token_model.input, character_vectorizer_model.input], output_layer)

    model.compile(loss="categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])
    return model

def tribrid_model(num_classes, token_embed, char_embed, text_vectorizer):
    """
    Constructs a tribrid TensorFlow Keras model that integrates token embeddings, character-level embeddings,
    and numerical features from line numbers and total lines, tailored for advanced text classification tasks.

    This function extends the concept of a hybrid model by adding numerical context in the form of line numbers
    and total lines, which might be particularly useful for tasks like document classification where the position
    of text within a document can provide meaningful context. The model concatenates outputs from separate models 
    for token embeddings, character-level embeddings, and numerical features, followed by dense and dropout layers 
    for robust classification.

    Parameters:
    num_classes (int): The number of output classes for the softmax classification layer.
    token_embed (tf.keras.layers.Layer): An embedding layer for converting tokenized text into dense embeddings.
    char_embed (tf.keras.layers.Layer): An embedding layer for converting character vectorized text into dense embeddings.
    text_vectorizer (tf.keras.layers.Layer): A text vectorization layer used for character vectorization in the character model.

    Returns:
    tf.keras.Model: A compiled Keras model that is ready for training, which includes multiple input layers
                    for different types of data (line numbers, total lines, tokenized text, and character vectorized text)
                    and a structured neural network architecture for classification.

    Example:
    ```python
    # Define embedding and vectorization layers
    token_embed = Embedding(input_dim=10000, output_dim=300, input_length=50)
    char_embed = Embedding(input_dim=100, output_dim=50, input_length=100)
    text_vectorizer = TextVectorization(max_tokens=1000, output_mode='int', split='character')

    # Create and compile the tribrid model
    model = tribrid_model(5, token_embed, char_embed, text_vectorizer)
    model.summary()
    ```
    """
    token_model = create_token_model(token_embed)
    character_vectorizer_model = create_character_vectorizer_model(char_embed, text_vectorizer)
    line_number_model = create_line_number_model(15, "line_number")
    total_lines_model = create_line_number_model(20, "total_lines")

    hybrid_layer = layers.Concatenate(name="hybrid")([token_model.output,
                                                      character_vectorizer_model.output])
    
    dense_layer = layers.Dense(256, activation="relu")(hybrid_layer)
    dropout_layer = layers.Dropout(0.5)(dense_layer)

    tribrid_layer = layers.Concatenate(name="tribrid")([line_number_model.output, total_lines_model.output, dropout_layer])
    output_layer = layers.Dense(num_classes, activation="softmax")(tribrid_layer)

    model = tf.keras.Model([line_number_model.input, total_lines_model.input, token_model.input, character_vectorizer_model.input], output_layer)

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])
    return model

