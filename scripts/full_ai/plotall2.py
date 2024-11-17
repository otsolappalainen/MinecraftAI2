import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Define the path to the consolidated CSV file
csv_file = r"C:\Users\odezz\source\MinecraftAI2\scripts\full_ai\combined_data_corrected.csv"

# Define the output PDF for graphs
output_pdf = "cumulative_and_position_plots.pdf"

def plot_cumulative_reward_and_position(csv_file, output_pdf):
    """
    Plot cumulative reward and position (X, Y, Z) from the CSV file.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Check if required columns are present
    required_columns = ["Cumulative Reward", "X", "Y", "Z"]
    if not all(col in df.columns for col in required_columns):
        print(f"Error: Missing one or more required columns in the CSV: {required_columns}")
        return

    # Open a multi-page PDF for the plots
    with PdfPages(output_pdf) as pdf:
        # Plot Cumulative Reward
        plt.figure()
        plt.plot(df["Cumulative Reward"], label="Cumulative Reward", color="orange")
        plt.xlabel("Row Index")
        plt.ylabel("Cumulative Reward")
        plt.title("Cumulative Reward Over Time")
        plt.legend()
        pdf.savefig()  # Save to PDF
        plt.close()

        # Plot X Position
        plt.figure()
        plt.plot(df["X"], label="X Position", color="blue")
        plt.xlabel("Row Index")
        plt.ylabel("X Position")
        plt.title("X Position Over Time")
        plt.legend()
        pdf.savefig()  # Save to PDF
        plt.close()

        # Plot Y Position
        plt.figure()
        plt.plot(df["Y"], label="Y Position", color="green")
        plt.xlabel("Row Index")
        plt.ylabel("Y Position")
        plt.title("Y Position Over Time")
        plt.legend()
        pdf.savefig()  # Save to PDF
        plt.close()

        # Plot Z Position
        plt.figure()
        plt.plot(df["Z"], label="Z Position", color="red")
        plt.xlabel("Row Index")
        plt.ylabel("Z Position")
        plt.title("Z Position Over Time")
        plt.legend()
        pdf.savefig()  # Save to PDF
        plt.close()

    print(f"Plots saved to {output_pdf}")

# Run the plotting function
plot_cumulative_reward_and_position(csv_file, output_pdf)