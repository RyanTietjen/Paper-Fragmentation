"""
Ryan Tietjen
Sep 2024
Demo application for paper abstract fragmentaion demonstration
"""
import gradio as gr
import tensorflow as tf
from tensorflow import keras
from keras import layers
from timeit import default_timer as timer
from process_input import split_abstract
from process_input import split_abstract_original
from process_input import split_sentences_by_characters
import pandas as pd
import tensorflow_hub as hub
from model import EmbeddingLayer
from process_input import encode_labels


sample_list = []
example1 =  f"""The aim of this study was to describe the electrocardiographic ( ECG ) evolutionary changes after an acute myocardial infarction ( AMI ) and to evaluate their correlation with left ventricular function and remodeling.
The QRS complex changes after AMI have been correlated with infarct size and left ventricular function.
By contrast , the significance of T wave changes is controversial.
We studied 536 patients enrolled in the GISSI-3-Echo substudy who underwent ECG and echocardiographic studies at 24 to 48 h ( S1 ) , at hospital discharge ( S2 ) , at six weeks ( S3 ) and six months ( S4 ) after AMI.
The number of Qwaves ( nQ ) and QRS quantitative score ( QRSs ) did not change over time.
From S2 to S4 , the number of negative T waves ( nT NEG ) decreased ( p < 0.0001 ) , wall motion abnormalities ( % WMA ) improved ( p < 0.001 ) , ventricular volumes increased ( p < 0.0001 ) while ejection fraction remained stable.
According to the T wave changes after hospital discharge , patients were divided into four groups : stable positive T waves ( group 1 , n = 35 ) , patients who showed a decrease > or = 1 in nT NEG ( group 2 , n = 361 ) , patients with no change in nT NEG ( group 3 , n = 64 ) and those with an increase > or = 1 in nT NEG ( group 4 , n = 76 ).
The QRSs and nQ remained stable in all groups.
Groups 3 and 4 showed less recovery in % WMA , more pronounced ventricular enlargement and progressive decline in ejection fraction than groups 1 and 2 ( interaction time x groups p < 0.0001 ).
The analysis of serial ECG can predict postinfarct left ventricular remodeling.
Normalization of negative T waves during the follow-up appears more strictly related to recovery of regional dysfunction than QRS changes.
Lack of resolution and late appearance of new negative T predict unfavorable remodeling with progressive deterioration of ventricular function."""
sample_list.append(example1)

def format_non_empty_lists(objective, background, methods, results, conclusion):
    """
    This function checks each provided list and formats a string with the list name and its contents
    only if the list is not empty.
    
    Parameters:
    - objective (list): List containing sentences classified as 'Objective'.
    - background (list): List containing sentences classified as 'Background'.
    - methods (list): List containing sentences classified as 'Methods'.
    - results (list): List containing sentences classified as 'Results'.
    - conclusion (list): List containing sentences classified as 'Conclusion'.
    
    Returns:
    - str: A formatted string that contains the non-empty list names and their contents.
    """
    
    output = ""
    lists = {
        'Objective': objective,
        'Background': background,
        'Methods': methods,
        'Results': results,
        'Conclusion': conclusion
    }
    
    for name, content in lists.items():
        if content:  # Check if the list is not empty
            output += f"{name}:\n"  # Append the category name followed by a newline
            for item in content:
                output += f"  - {item}\n"  # Append each item in the list, formatted as a list
            
            output += "\n"  # Append a newline for better separation between categories

    return output.strip() 

def fragment_single_abstract(abstract):
    """
    Processes a single abstract by fragmenting it into structured sections based on predefined categories
    such as Objective, Methods, Results, Conclusions, and Background. The function utilizes a pre-trained Keras model
    to predict the category of each sentence in the abstract.

    The process involves several steps:
    1. Splitting the abstract into sentences.
    2. Encoding these sentences using a custom embedding layer.
    3. Classifying each sentence into one of the predefined categories.
    4. Grouping the sentences by their predicted categories.

    Parameters:
    abstract (str): The abstract text that needs to be processed and categorized.

    Returns:
    tuple: A tuple containing two elements:
        - A dictionary with keys as the category names ('Objective', 'Background', 'Methods', 'Results', 'Conclusions')
          and values as lists of sentences belonging to these categories. Only non-empty categories are returned.
        - The time taken to process the abstract (in seconds).

    Example:
    ```python
    abstract_text = "This study aims to evaluate the effectiveness of..."
    categorized_abstract, processing_time = fragment_single_abstract(abstract_text)
    print("Categorized Abstract:", categorized_abstract)
    print("Processing Time:", processing_time)
    ```

    Note:
    - This function assumes that a Keras model 'test.keras' and a custom embedding layer 'EmbeddingLayer'
      are available and correctly configured to be loaded.
    - The function uses pandas for data manipulation, TensorFlow for machine learning operations,
      and TensorFlow's data API for batching and prefetching data for model predictions.
    """
    start_time = timer()

    original_abstract = split_abstract_original(abstract)
    df_original = pd.DataFrame(original_abstract)
    sentences_original = df_original["text"].tolist()

    abstract_split = split_abstract(abstract)
    df = pd.DataFrame(abstract_split)
    sentences = df["text"].tolist()
    labels = encode_labels(df["target"])

    objective = []
    background = []
    methods = []
    results = []
    conclusion = []

    embed_layer = EmbeddingLayer()
    model = tf.keras.models.load_model("test.keras", custom_objects={'EmbeddingLayer': embed_layer})

    data_by_character = split_sentences_by_characters(sentences)
    line_numbers = tf.one_hot(df["line_number"].to_numpy(), depth=15)
    total_line_numbers = tf.one_hot(df["total_lines"].to_numpy(), depth=20)
    
    sentences_dataset = tf.data.Dataset.from_tensor_slices((line_numbers, total_line_numbers, sentences, data_by_character))
    labels_dataset = tf.data.Dataset.from_tensor_slices(labels) 
    dataset = tf.data.Dataset.zip((sentences_dataset, labels_dataset)).batch(32).prefetch(tf.data.AUTOTUNE)

    predictions = tf.argmax(model.predict(dataset), axis=1)

    for i, prediction in enumerate(predictions):
        if prediction == 0:
            objective.append(sentences_original[i])
        elif prediction == 1:
            methods.append(sentences_original[i])
        elif prediction == 2:
            results.append(sentences_original[i])
        elif prediction == 3:
            conclusion.append(sentences_original[i])
        elif prediction == 4:
            background.append(sentences_original[i])

    end_time = timer()

    return format_non_empty_lists(objective, background, methods, results, conclusion), end_time - start_time



title = "Paper Abstract Fragmentation With TensorFlow by Ryan Tietjen"
description = f"""
This app will take the abstract of a paper and break it down into five categories: objective, background, methods, results, and conclusion. 
The dataset used can be found in the [PubMed 200k RCT]("https://arxiv.org/abs/1710.06071") and in [this repo](https://github.com/Franck-Dernoncourt/pubmed-rct). The model architecture
was based off of ["Neural Networks for Joint Sentence Classification in Medical Paper Abstracts."](https://arxiv.org/pdf/1612.05251)

This project achieved a testing accuracy of 88.12% and a F1 score of 87.92%. For the whole project, please visit [my GitHub](https://github.com/RyanTietjen/Paper-Fragmentation).

How to use:

-Paste the given abstract into the box below.

-Make sure to separate each sentence by a new line (this helps avoid ambiguity).

-Click submit, and allow the model to run!
"""

demo = gr.Interface(
    fn=fragment_single_abstract,
    inputs=gr.Textbox(lines=10, placeholder="Enter abstract here..."),
    outputs=[
        gr.Textbox(label="Fragmented Abstract"),
        gr.Number(label="Time to process (s)"),
    ],
    examples=sample_list,
    title=title,
    description=description,
)


demo.launch(share=False)