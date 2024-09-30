"""
Sep 2024
Ryan Tietjen
Contains helper functions to process user input for the demo
"""
import pandas as pd

def split_abstract(abstract):
    results = []

    lines = abstract.split("\n")
    for i, line in enumerate(lines):
        entry = {
                "target": 0,
                "text": line.lower(),
                "line_number": i + 1,
                "total_lines": len(lines)
                }
        results.append(entry)
    return results

def split_abstract_original(abstract):
    results = []

    lines = abstract.split("\n")
    for i, line in enumerate(lines):
        entry = {
                "target": 0,
                "text": line,
                "line_number": i + 1,
                "total_lines": len(lines)
                }
        results.append(entry)
    return results

def split_sentences_by_characters(corpus):
    return [" ".join(sentence) for sentence in corpus]

def encode_labels(*datasets):
    """
    Encode labels for multiple datasets using a unified label mapping.
    
    Args:
    *datasets: Arbitrary number of array-like structures containing labels.
    
    Returns:
    tuple: Encoded labels as numpy arrays for each dataset.
    """
    # Collect all labels from all datasets into a single list
    all_labels = pd.concat([pd.Series(data) for data in datasets])
    
    # Get unique labels and sort them to ensure consistency
    unique_labels = pd.unique(all_labels)
    unique_labels.sort()
    
    # Create mapping from labels to integers
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    
    # Function to encode a single dataset
    def encode_single_dataset(dataset, mapping):
        return pd.Series(dataset).map(mapping).to_numpy()
    
    # Encode all datasets using the mapping
    encoded_datasets = tuple(encode_single_dataset(dataset, label_to_index) for dataset in datasets)
    
    # Return only the encoded datasets
    return encoded_datasets