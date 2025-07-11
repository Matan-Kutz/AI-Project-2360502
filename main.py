#!/usr/bin/env python3

from GameFeatures import GameFeatures
import os


def check_if_log_exists(log_file):
    if not os.path.exists(log_file):
        print(f"Error: {log_file} not found in the current directory.")
        print("Please create a log.txt file with game transcript data.")
        exit(1)


def create_game_features():
    print("Creating GameFeatures object...")
    game_features = GameFeatures()
    print(f"Board size: {game_features.BOARD_WIDTH}x{game_features.BOARD_LENGTH}")
    print(f"Pieces: {game_features.PIECES}")
    print()
    return game_features


def read_game_transcript(game_features, log_file):
    print(f"Reading game transcript from {log_file}...")
    moves = game_features.read_game_transcript(log_file)
    print(f"Found {len(moves)} winning moves in the transcript.")
    print()

    if not moves:
        print("No moves found in the transcript.")
        return

    return moves


def create_feature_list(game_features, moves):
    print("Creating feature lists...")
    feature_lists = game_features.create_feature_lists(moves)
    print(f"Created {len(feature_lists)} feature lists.")
    print()
    return feature_lists


def print_features(feature_lists):
    print("Feature Lists Results:")
    print("-" * 40)

    for i, feature_list in enumerate(feature_lists):
        print(f"Move {i + 1} (Turn {i * 2 + 1}):")
        print(f"  Number of features: {len(feature_list)}")
        print(f"  Features: {feature_list}")
        print()


def print_features_summary():
    print("Feature Summary:")
    print("-" * 40)
    print("Feature order:")
    print("1-3:   Column counts (1, 2, 3)")
    print("4-6:   Line counts (1, 2, 3)")
    print("7-9:   Player pieces on board (S, M, L)")
    print("10-12: Player pieces captured (S, M, L)")
    print("13-15: Enemy pieces on board (S, M, L)")
    print("16-18: Enemy pieces captured (S, M, L)")
    print("19:    Player average X location")
    print("20:    Player average Y location")
    print("21:    Player spread")
    print("22:    Total pieces on board")
    print("23-25: Game progress (1-hot encoded)")


def main():
    print("GameFeatures Testing System")
    print("=" * 40)

    # Check if log.txt exists
    log_file = "log.txt"
    check_if_log_exists(log_file)

    try:
        game_features = create_game_features()
        moves = read_game_transcript(game_features, log_file)
        feature_lists = create_feature_list(game_features, moves)
        print_features(feature_lists)
        print_features_summary()

    except Exception as e:
        print(f"Error: {e}")
        print("Please check the log file format and try again.")


if __name__ == "__main__":
    main()
