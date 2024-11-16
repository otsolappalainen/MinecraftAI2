import csv
import os

# Define file paths
EPISODE_REWARDS_FILE = "C:/Users/odezz/source/MinecraftAI2/scripts/game_interface/episode_rewards.csv"
CLEANED_EPISODE_REWARDS_FILE = "C:/Users/odezz/source/MinecraftAI2/scripts/game_interface/episode_rewards_cleaned.csv"
OPTUNA_LOG_FILE = "C:/Users/odezz/source/MinecraftAI2/scripts/game_interface/optuna_trials_log.csv"


def clean_episode_rewards_file(input_file="episode_rewards.csv", output_file="cleaned_episode_rewards.csv"):
    """Reads the episode rewards file, filters out rows that do not match the expected format,
    and writes the cleaned data to a new file."""
    with open(input_file, "r") as infile, open(output_file, "w", newline="") as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Write header
        writer.writerow(["Trial", "Episode", "Cumulative Reward"])

        # Skip header in the input file if it exists
        next(reader, None)  # Skip header row if present

        for row in reader:
            # Check if the row has exactly 3 columns and try to parse them as int, int, and float
            if len(row) == 3:
                try:
                    trial_number = int(row[0])
                    episode_number = int(row[1])
                    cumulative_reward = float(row[2])
                    writer.writerow([trial_number, episode_number, cumulative_reward])
                except ValueError:
                    print(f"Skipping row with unexpected format: {row}")
            else:
                print(f"Skipping row with unexpected format: {row}")

    print(f"Cleaning complete. Cleaned data written to {output_file}")


def run_optuna_training():
    """Placeholder for Optuna training script. Implement or call the Optuna script here."""
    print("Starting Optuna training (this is a placeholder).")
    # Import and call the actual Optuna training function if implemented separately
    # For example, you could use: `from train_model2 import objective` and run Optuna from here.


def view_file_contents(file_path):
    """Displays contents of a specified file."""
    print(f"\nContents of {file_path}:\n" + "-" * 50)
    try:
        with open(file_path, "r") as file:
            print(file.read())
    except FileNotFoundError:
        print("File not found.")
    print("-" * 50)


def main_menu():
    """Displays the main menu and handles user choices."""
    while True:
        print("\nMain Menu")
        print("1. Clean episode rewards file")
        print("2. Run Optuna training (placeholder)")
        print("3. View episode rewards file")
        print("4. View cleaned episode rewards file")
        print("5. View Optuna log file")
        print("6. Exit")

        choice = input("Enter the number of the action you'd like to perform: ")

        if choice == "1":
            clean_episode_rewards_file()
        elif choice == "2":
            run_optuna_training()
        elif choice == "3":
            view_file_contents(EPISODE_REWARDS_FILE)
        elif choice == "4":
            view_file_contents(CLEANED_EPISODE_REWARDS_FILE)
        elif choice == "5":
            view_file_contents(OPTUNA_LOG_FILE)
        elif choice == "6":
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 6.")


if __name__ == "__main__":
    main_menu()