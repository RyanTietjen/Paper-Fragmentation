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
example1 =  f"""The aim of this study was to verify in bruxism patients the possible efficacy of auricular stimulation in reducing the hypertonicity of some masticatory muscles.
Forty-three bruxism patients were randomly allocated to 3 groups : acupuncture , needle contact for 10 seconds , no treatment ( control ).
Helkimo 's clinical dysfunction index ( CDI ) and anamnestic dysfunction index ( ADI ) were used to assess the functional state of the masticatory system.
The resting electrical activity of the anterior temporalis ( AT ) , masseter ( MM ) , digastric ( DA ) and sternocleidomastoid ( SCM ) muscles was measured , according to Jankelson , with surface electrodes at baseline , after stimulation and continually for 30 minutes ( 120 measurements in total ).
The electromyographical variations in the 3 groups were studied with t test for independent samples.
Acupuncture and needle contact were superior to control in reducing the muscle hypertonicity of all muscles except SCM.
In the comparison between acupuncture and needle contact the former showed better results only for the right TA and left DA ( p = 0.000 ).
In this study it was possible to measure the efficacy of the stimulation of only one point or area , which is an ideal model for research in acupuncture.
The auricular area we chose for stimulation was never used before for the purpose of relaxing masticatory muscles.
Acupuncture and needle contact for 10 seconds showed similar effects."""
example2 = """To investigate the efficacy of 6 weeks of daily low-dose oral prednisolone in improving pain , mobility , and systemic low-grade inflammation in the short term and whether the effect would be sustained at 12 weeks in older adults with moderate to severe knee osteoarthritis ( OA ) .
A total of 125 patients with primary knee OA were randomized 1:1 ; 63 received 7.5 mg/day of prednisolone and 62 received placebo for 6 weeks .
Outcome measures included pain reduction and improvement in function scores and systemic inflammation markers .
Pain was assessed using the visual analog pain scale ( 0-100 mm ) .
Secondary outcome measures included the Western Ontario and McMaster Universities Osteoarthritis Index scores , patient global assessment ( PGA ) of the severity of knee OA , and 6-min walk distance ( 6MWD ) .
Serum levels of interleukin 1 ( IL-1 ) , IL-6 , tumor necrosis factor ( TNF ) - , and high-sensitivity C-reactive protein ( hsCRP ) were measured .
There was a clinically relevant reduction in the intervention group compared to the placebo group for knee pain , physical function , PGA , and 6MWD at 6 weeks .
The mean difference between treatment arms ( 95 % CI ) was 10.9 ( 4.8-18 .0 ) , p < 0.001 ; 9.5 ( 3.7-15 .4 ) , p < 0.05 ; 15.7 ( 5.3-26 .1 ) , p < 0.001 ; and 86.9 ( 29.8-144 .1 ) , p < 0.05 , respectively .
Further , there was a clinically relevant reduction in the serum levels of IL-1 , IL-6 , TNF - , and hsCRP at 6 weeks in the intervention group when compared to the placebo group .
These differences remained significant at 12 weeks .
The Outcome Measures in Rheumatology Clinical Trials-Osteoarthritis Research Society International responder rate was 65 % in the intervention group and 34 % in the placebo group ( p < 0.05 ) .
Low-dose oral prednisolone had both a short-term and a longer sustained effect resulting in less knee pain , better physical function , and attenuation of systemic inflammation in older patients with knee OA ( ClinicalTrials.gov identifier NCT01619163 ) ."""
sample_list.append(example1)
sample_list.append(example2)

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
    model = tf.keras.models.load_model("200k_10_epochs.keras", custom_objects={'EmbeddingLayer': embed_layer})

    data_by_character = split_sentences_by_characters(sentences)
    line_numbers = tf.one_hot(df["line_number"].to_numpy(), depth=15)
    total_line_numbers = tf.one_hot(df["total_lines"].to_numpy(), depth=20)
    
    sentences_dataset = tf.data.Dataset.from_tensor_slices((line_numbers, total_line_numbers, sentences, data_by_character))
    labels_dataset = tf.data.Dataset.from_tensor_slices(labels) 
    dataset = tf.data.Dataset.zip((sentences_dataset, labels_dataset)).batch(32).prefetch(tf.data.AUTOTUNE)

    predictions = tf.argmax(model.predict(dataset), axis=1)

    for i, prediction in enumerate(predictions):
        if prediction == 3:
            objective.append(sentences_original[i])
        elif prediction == 2:
            methods.append(sentences_original[i])
        elif prediction == 4:
            results.append(sentences_original[i])
        elif prediction == 1:
            conclusion.append(sentences_original[i])
        elif prediction == 0:
            background.append(sentences_original[i])

    end_time = timer()

    return format_non_empty_lists(objective, background, methods, results, conclusion), end_time - start_time



title = "Paper Abstract Fragmentation With TensorFlow by Ryan Tietjen"
description = f"""
This app will take the abstract of a paper and break it down into five categories: objective, background, methods, results, and conclusion. 
The dataset used can be found in the [PubMed 200k RCT]("https://arxiv.org/pdf/1710.06071") and in [this repo](https://github.com/Franck-Dernoncourt/pubmed-rct). The model architecture
was based off of ["Neural Networks for Joint Sentence Classification in Medical Paper Abstracts."](https://arxiv.org/pdf/1612.05251)
This model achieved a testing accuracy of 88.2% and a F1 score of 88%. For the whole project, please visit [my GitHub](https://github.com/RyanTietjen/Paper-Fragmentation).
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
