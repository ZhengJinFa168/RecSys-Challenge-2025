import numpy as np
import pickle
from scipy.sparse import save_npz, load_npz
from SLIM_MSE_fastest import train_multiple_epochs

def save_slim_model(item_item_S, filepath='slim_model.npy'):
    """Save the SLIM similarity matrix"""
    np.save(filepath, item_item_S)
    print(f"Model saved to {filepath}")

def load_slim_model(filepath='slim_model.npy'):
    """Load the SLIM similarity matrix"""
    item_item_S = np.load(filepath)
    print(f"Model loaded from {filepath}")
    return item_item_S
