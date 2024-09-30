"""
Ryan Tietjen
Sep 2024
Contains functions related to manipulating and visualizing the dataset
https://github.com/Franck-Dernoncourt/pubmed-rct
"""
import os
import tensorflow as tf
import pandas as pd


def get_file_names(dir):
    """
    Retrieves the full paths of all files within a specified directory.

    This function lists all entries in the given directory and constructs full paths
    by concatenating the directory path with each entry name, assuming all entries are files.
    It returns a list containing these full file paths.

    Note: This function does not differentiate between file types and directories.
    It assumes all entries in the directory are files.

    Parameters:
    dir (str): The directory path from which file names are to be fetched.

    Returns:
    list: A list of strings, each string being a full path to a file in the specified directory.
    """

    # List all entries in the directory and prepend the directory path to each entry
    files = [os.path.join(dir, file) for file in os.listdir(dir)]
    return files

def get_lines_from_txt_file(dir, file):
    """
    Reads lines from a text file located at a specified directory and filename.

    This function constructs a file path by concatenating the directory and filename,
    opens the file in read mode, and reads all lines from the file. It returns these
    lines as a list, where each element in the list corresponds to a line in the file.

    Parameters:
    dir (str): The directory path where the text file is located.
    file (str): The name of the text file to be read.

    Returns:
    list: A list of strings, where each string is a line from the file.
    """

    # Construct the full path to the file
    path = dir + file

    # Open the file, read lines, and return them
    with open(path, "r") as f:
        return f.readlines()
    
def split_text_into_abstracts(dir, file):
    """
    Extracts and processes abstracts from a text file located in a specified directory.

    This function reads a text file where abstracts are separated by lines starting with "###",
    and individual abstract entries are separated by empty lines. Each line within an abstract
    contains fields separated by tabs. The fields are expected to be "target" and "text".
    The function processes these entries, converting text to lowercase and adding metadata
    about line numbers within each abstract.

    Parameters:
    dir (str): Directory path where the text file is located.
    file (str): Name of the text file to be processed.

    Returns:
    list of dict: A list of dictionaries, where each dictionary contains:
        - "target": the target or label of the abstract.
        - "text": the content of the abstract converted to lowercase.
        - "line_number": the zero-based line number of the text within the abstract.
        - "total_lines": the total number of lines in the abstract minus one.
    """

    # Read lines from the specified text file
    input_lines = get_lines_from_txt_file(dir, file)
    abstract = ""  # Temporary storage for accumulating lines of a single abstract
    abstracts = []  # List to store processed abstract entries

    for line in input_lines:
        if line.isspace():
            # Split the accumulated abstract into separate lines
            abstract_split = abstract.splitlines()

            # Process each line in the split abstract
            for i, abstract_line in enumerate(abstract_split):
                split = abstract_line.split("\t")
                
                entry = {
                    "target": split[0],
                    "text": (split[1]).lower(),
                    "line_number": i,
                    "total_lines": len(abstract_split) - 1
                }

                # Append the processed entry to the list of abstracts
                abstracts.append(entry)
            
            # Reset abstract for the next one
            abstract = ""

        elif not line.startswith("###"):
            # Accumulate lines of the current abstract
            abstract += line
    
    return abstracts
def one_hot_encode_labels(*datasets):
    """
    One-hot encode labels for multiple datasets using a unified label mapping and TensorFlow.
    
    Args:
    *datasets: Arbitrary number of array-like structures containing labels.
    
    Returns:
    tuple: One-hot encoded labels as numpy arrays for each dataset.
    """
    # Collect all labels from all datasets into a single list
    all_labels = pd.concat([pd.Series(data) for data in datasets])
    
    # Get unique labels and sort them to ensure consistency
    unique_labels = pd.unique(all_labels)
    unique_labels.sort()
    
    # Create mapping from labels to integers
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    
    # Function to one-hot encode a single dataset
    def one_hot_encode_single_dataset(dataset, mapping, num_classes):
        # Convert labels to integer indices
        labels_int = pd.Series(dataset).map(mapping).to_numpy()
        
        # Convert integer labels to a TensorFlow tensor
        labels_tensor = tf.convert_to_tensor(labels_int, dtype=tf.int32)
        
        # Apply one-hot encoding
        labels_one_hot = tf.one_hot(labels_tensor, depth=num_classes)
        
        # Return the result as a numpy array
        return labels_one_hot.numpy()
    
    # Encode all datasets using the mapping
    num_classes = len(unique_labels)
    encoded_datasets = tuple(one_hot_encode_single_dataset(dataset, label_to_index, num_classes) for dataset in datasets)
    
    # Return the encoded datasets
    return encoded_datasets


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