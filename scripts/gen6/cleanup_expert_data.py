import os
import glob
import pickle
import random
import shutil
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def cleanup_expert_data(data_dir, trash_dir, no_op_removal_rate=0.8):
    # Create trash directory
    os.makedirs(trash_dir, exist_ok=True)
    
    # Find all expert data files
    data_files = glob.glob(os.path.join(data_dir, "session_*", "expert_data.pkl"))
    logger.info(f"Found {len(data_files)} data files")
    
    for file_path in data_files:
        # Load data
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        # Count samples
        total_samples = len(data)
        no_op_indices = [i for i, sample in enumerate(data) if sample['action'] == 17]  # 17 is no_op
        no_op_count = len(no_op_indices)
        
        logger.info(f"\nFile: {file_path}")
        logger.info(f"Total samples: {total_samples}")
        logger.info(f"No-op samples: {no_op_count} ({no_op_count/total_samples*100:.1f}%)")
        
        # Select samples to remove
        num_to_remove = int(no_op_count * no_op_removal_rate)
        indices_to_remove = random.sample(no_op_indices, num_to_remove)
        
        # Create new filtered dataset
        filtered_data = [sample for i, sample in enumerate(data) if i not in indices_to_remove]
        
        # Save removed samples to trash
        trash_samples = [sample for i, sample in enumerate(data) if i in indices_to_remove]
        trash_file = os.path.join(trash_dir, os.path.basename(os.path.dirname(file_path)) + "_trash.pkl")
        with open(trash_file, 'wb') as f:
            pickle.dump(trash_samples, f)
            
        # Save filtered data back
        with open(file_path, 'wb') as f:
            pickle.dump(filtered_data, f)
            
        logger.info(f"Removed {num_to_remove} no-op samples")
        logger.info(f"Remaining samples: {len(filtered_data)}")

if __name__ == "__main__":
    DATA_DIR = "expert_data"
    TRASH_DIR = "trash_data"
    
    cleanup_expert_data(DATA_DIR, TRASH_DIR)