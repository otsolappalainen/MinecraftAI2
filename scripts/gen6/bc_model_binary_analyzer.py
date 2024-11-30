import os

def compare_model_files_bit_by_bit(file1, file2):
    """
    Compare two model files byte by byte, and calculate the percentage of differing bits.
    Returns the percentage of differing bits.
    """
    try:
        # Open both files in binary mode
        with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
            # Initialize counters for differing bits
            total_bits = 0
            differing_bits = 0
            
            while True:
                # Read chunks from both files (this time in smaller chunks for more thorough comparison)
                chunk1 = f1.read(8192)  # 8 KB chunks
                chunk2 = f2.read(8192)
                
                # If both chunks are empty, we've reached the end of both files
                if not chunk1 and not chunk2:
                    break

                # Handle different file sizes gracefully
                # If one file is shorter, pad the shorter one with zero bytes (or some other padding)
                if len(chunk1) != len(chunk2):
                    min_length = min(len(chunk1), len(chunk2))
                    chunk1 = chunk1[:min_length]  # Truncate the longer chunk
                    chunk2 = chunk2[:min_length]

                # Compare the chunks byte by byte
                for byte1, byte2 in zip(chunk1, chunk2):
                    # XOR the bytes to find differing bits
                    diff = byte1 ^ byte2
                    
                    # Count the differing bits in the XOR result
                    differing_bits += bin(diff).count('1')
                    total_bits += 8  # Each byte has 8 bits
            
            # Calculate the percentage of differing bits
            if total_bits > 0:
                percentage_difference = (differing_bits / total_bits) * 100
            else:
                percentage_difference = 0
            
            return percentage_difference
    except Exception as e:
        print(f"Error comparing files: {e}")
        return -1  # Indicate an error

if __name__ == "__main__":
    # Example usage
    model_file1 = r"C:\Users\odezz\source\MinecraftAI2\scripts\gen6\models_bc\model_epoch_5.pth"
    model_file2 = r"C:\Users\odezz\source\MinecraftAI2\scripts\gen6\models_bc\model_epoch_10.pth"

    difference_percentage = compare_model_files_bit_by_bit(model_file1, model_file2)

    if difference_percentage >= 0:
        print(f"The files {model_file1} and {model_file2} have {difference_percentage:.4f}% differing bits.")
    else:
        print(f"There was an error comparing the files.")
