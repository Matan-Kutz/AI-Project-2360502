#!/usr/bin/env python3
"""
Main file for testing the GameFeatures system.
Reads a log file and creates feature lists based on it.
"""

from GameFeatures import GameFeatures
import os

def main():
    """
    Main function to test the GameFeatures system.
    Reads log.txt and creates feature lists.
    """
    print("GameFeatures Testing System")
    print("=" * 40)
    
    # Check if log.txt exists
    log_file = "log.txt"
    if not os.path.exists(log_file):
        print(f"Error: {log_file} not found in the current directory.")
        print("Please create a log.txt file with game transcript data.")
        return
    
    try:
        # Create GameFeatures object
        print("Creating GameFeatures object...")
        game_features = GameFeatures()
        print(f"Board size: {game_features.BOARD_WIDTH}x{game_features.BOARD_LENGTH}")
        print(f"Pieces: {game_features.PIECES}")
        print()
        
        # Read game transcript
        print(f"Reading game transcript from {log_file}...")
        moves = game_features.read_game_transcript(log_file)
        print(f"Found {len(moves)} winning moves in the transcript.")
        print()
        
        if not moves:
            print("No moves found in the transcript.")
            return
        
        # Create feature lists
        print("Creating feature lists...")
        feature_lists = game_features.create_feature_lists(moves)
        print(f"Created {len(feature_lists)} feature lists.")
        print()
        
        # Display results
        print("Feature Lists Results:")
        print("-" * 40)
        
        for i, feature_list in enumerate(feature_lists):
            print(f"Move {i+1} (Turn {i*2 + 1}):")
            print(f"  Number of features: {len(feature_list)}")
            print(f"  Features: {feature_list}")
            print()
        
        # Display feature summary
        print("Feature Summary:")
        print("-" * 40)
        print("Feature order:")
        print("1-3:   Column counts (1, 2, 3)")
        print("4-6:   Line counts (1, 2, 3)")
        print("7-9:   Player pieces on board (S, M, L)")
        print("10-12: Player pieces captured (S, M, L)")
        print("13-15: Player pieces not on board (S, M, L)")
        print("16-18: Enemy pieces on board (S, M, L)")
        print("19-21: Enemy pieces captured (S, M, L)")
        print("22:    Player average X location")
        print("23:    Player average Y location")
        print("24:    Player spread")
        print("25:    Total pieces on board")
        print("26-28: Game progress (1-hot encoded)")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check the log file format and try again.")

if __name__ == "__main__":
    main() 