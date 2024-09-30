"""
Ryan Tietjen
Sep 2024
Contains various helper functions
"""

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
def classification_metrics(predictions, true_labels):
    """
    Calculates and returns the accuracy, precision, recall, and F1 score for the given predictions and true labels.
    
    Parameters:
    predictions (list): A list of predicted labels.
    true_labels (list): A list of the actual true labels.
    
    Returns:
    dict: A dictionary containing the 'accuracy', 'precision', 'recall', and 'f1_score'.
    """
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions)
    
    # Calculate precision, recall, and F1-score
    precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    
    # Store the results in a dictionary
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }
    
    return results
def split_sentences_by_characters(corpus):
    """
    Splits each sentence in a corpus into its individual characters, with spaces between characters.

    This function takes a list of sentences and processes each sentence such that each character 
    within the sentence is separated by a space. The result is a list where each element is a 
    transformed sentence with spaced characters.

    Parameters:
    corpus (list of str): A list of sentences to be processed.

    Returns:
    list of str: A list of sentences where each character in each sentence is separated by a space.
    """

    # Convert each sentence in the corpus to a string of spaced characters
    return [" ".join(sentence) for sentence in corpus]