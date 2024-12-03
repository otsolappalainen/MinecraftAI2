import os
import glob
import pickle
import random
import shutil
import logging
import numpy as np
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def fix_task_array(data):
    """Convert all task arrays to standard height target format"""
    TASK_SIZE = 20
    standard_task = np.array([0, 0, 132, 133] + [0]*(TASK_SIZE-4), dtype=np.float32)
    
    for sample in data:
        if 'observation' in sample:
            sample['observation']['task'] = standard_task.copy()
        if 'next_observation' in sample:
            sample['next_observation']['task'] = standard_task.copy()
    
    return data

def analyze_and_fix_attack_sequences(data):
    """Analyze attack patterns and fix sequences leading to broken blocks"""
    total_actions = len(data)
    attack_count = sum(1 for sample in data if sample['action'] == 13)  # 13 is attack
    broken_block_count = 0
    blocks_with_prior_attacks = 0
    blocks_with_full_attack_sequence = 0
    
    # First pass - collect statistics
    for i, sample in enumerate(data):
        # Check if this sample has broken blocks
        if 'observation' in sample and 'other' in sample['observation']:
            # Broken blocks info starts at index 9 in 'other' array
            other_array = sample['observation']['other']
            if len(other_array) >= 13 and any(other_array[9:13] != 0):  # Check first block
                broken_block_count += 1
                
                # Check previous 5 actions
                prev_actions = [data[j]['action'] for j in range(max(0, i-5), i)]
                if any(action == 13 for action in prev_actions):
                    blocks_with_prior_attacks += 1
                if all(action == 13 for action in prev_actions) and len(prev_actions) == 5:
                    blocks_with_full_attack_sequence += 1
    
    logger.info(f"\nAnalysis Results:")
    logger.info(f"Total actions: {total_actions}")
    logger.info(f"Attack actions: {attack_count} ({attack_count/total_actions*100:.1f}%)")
    logger.info(f"Samples with broken blocks: {broken_block_count}")
    logger.info(f"Broken blocks with prior attacks: {blocks_with_prior_attacks}")
    logger.info(f"Broken blocks with full attack sequence: {blocks_with_full_attack_sequence}")
    
    # Second pass - fix sequences
    modified_data = data.copy()
    for i, sample in enumerate(modified_data):
        if 'observation' in sample and 'other' in sample['observation']:
            other_array = sample['observation']['other']
            if len(other_array) >= 13 and any(other_array[9:13] != 0):
                # Set current and previous 5 actions to attack
                for j in range(max(0, i-5), i+1):
                    modified_data[j]['action'] = 13  # Set to attack
    
    return modified_data

def cleanup_expert_data(data_dir, trash_dir, no_op_removal_rate=0.8):
    # Create trash directory
    os.makedirs(trash_dir, exist_ok=True)
    
    # Find all expert data files (including nested folders)
    data_files = glob.glob(os.path.join(data_dir, "**", "expert_data.pkl"), recursive=True)
    logger.info(f"Found {len(data_files)} data files")
    
    for file_path in tqdm(data_files, desc="Processing files"):
        try:
            # Load data
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Fix task arrays
            data = fix_task_array(data)
            
            # Analyze and fix attack sequences
            data = analyze_and_fix_attack_sequences(data)
            
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
            rel_path = os.path.relpath(file_path, data_dir)
            trash_file = os.path.join(trash_dir, rel_path.replace('expert_data.pkl', 'trash.pkl'))
            os.makedirs(os.path.dirname(trash_file), exist_ok=True)
            
            with open(trash_file, 'wb') as f:
                pickle.dump(trash_samples, f)
            
            # Save filtered data back
            with open(file_path, 'wb') as f:
                pickle.dump(filtered_data, f)
            
            logger.info(f"Removed {num_to_remove} no-op samples")
            logger.info(f"Remaining samples: {len(filtered_data)}")
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue

if __name__ == "__main__":
    DATA_DIR = r"E:\automatic_model_tool_spam"  # Updated path
    TRASH_DIR = "trash_data"
    
    cleanup_expert_data(DATA_DIR, TRASH_DIR)