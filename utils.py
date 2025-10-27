import re
import numpy as np
import pandas as pd

def load_data(features_file, labels_file):
    data = np.loadtxt(features_file, dtype="float")
    labels = np.loadtxt(labels_file, dtype="int")
    
    print(f"X shape: {data.shape}")
    print(f"Y shape: {labels.shape}")

    return data, labels

