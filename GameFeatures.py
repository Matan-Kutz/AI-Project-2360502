from Features import createGlobalFeatures
import json

class GameFeatures:
    """
    A class to manage and organize game features, including both global and game-specific features.
    """
    
    def __init__(self, game_file):
        # Initialize empty features list
        self.features = []
        
        # Read game configuration from file
        with open(game_file, 'r') as f:
            game_config = json.load(f)
            
        # Extract board dimensions and pieces
        width = game_config.get('width', 0)
        length = game_config.get('length', 0)
        pieces = game_config.get('pieces', {})
        
        # Get global features from createParameters
        self.global_features = createGlobalFeatures(width, length, pieces)
        
        # Add global features to the features list
        for feature_name, feature_value in self.global_features.items():
            self.features.append({
                'name': feature_name,
                'value': feature_value,
                'type': 'global'
            })
            
        # Add game-specific features from the configuration
        specific_features = game_config.get('specific_features', {})
        for feature_name, feature_value in specific_features.items():
            self.features.append({
                'name': feature_name,
                'value': feature_value,
                'type': 'specific'
            })
        

    def get_feature(self, feature_name):
        """
        Get a feature value by name.
        
        Args:
            feature_name (str): Name of the feature
            
        Returns:
            The value of the requested feature, or None if not found
        """
        for feature in self.features:
            if feature['name'] == feature_name:
                return feature['value']
        return None
    
    def update_feature(self, feature_name, value):
        """
        Update a feature value by name.
        
        Args:
            feature_name (str): Name of the feature
            value: New value for the feature
        """
        for feature in self.features:
            if feature['name'] == feature_name:
                feature['value'] = value
                return
        # If feature doesn't exist, add it as a game-specific feature
        self.features.append({
            'name': feature_name,
            'value': value,
            'type': 'specific'
        })
    
    def get_all_features(self):
        """
        Get all features.
        
        Returns:
            list: List of all features
        """
        return self.features
    
    def get_global_features(self):
        """
        Get all global features.
        
        Returns:
            list: List of global features
        """
        return [f for f in self.features if f['type'] == 'global']
    
    def get_specific_features(self):
        """
        Get all game-specific features.
        
        Returns:
            list: List of game-specific features
        """
        return [f for f in self.features if f['type'] == 'specific']

    def read_game_transcript(self, transcript_file):
        """
        Read a game transcript and generate feature lists for each winning move.
        The first line of the transcript must be "winner:[player_id]" where player_id is 1 or 2.
        When two pieces are at the same location, the smaller one is captured (piece sizes: S < M < L).
        Players alternate turns starting with player 1.
        
        Args:
            transcript_file (str): Path to the game transcript file
            
        Returns:
            list: List of feature dictionaries, one for each winning move
        """
        # Read the transcript file
        with open(transcript_file, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            return []
            
        # Get winning player from first line
        first_line = lines[0].strip()
        if not first_line.startswith('winner:'):
            raise ValueError("First line must specify the winner in format 'winner:[player_id]'")
            
        winning_player = int(first_line.split(':')[1])
        if winning_player not in [1, 2]:
            raise ValueError("Winner player_id must be 1 or 2")
            
        # Parse the transcript
        board_state = {}  # Dictionary to store current board state
        current_turn = 0
        winning_moves = []  # List to store features for each winning move
        captured_pieces = {1: [], 2: []}  # Track captured pieces for each player
        
        def calculate_piece_stats(positions):
            """Calculate statistics for a set of piece positions"""
            if not positions:
                return {
                    'count': 0,
                    'avg_x': 0,
                    'avg_y': 0,
                    'spread': 0
                }
            
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            avg_x = sum(x_coords) / len(x_coords)
            avg_y = sum(y_coords) / len(y_coords)
            
            # Calculate spread (average distance from center)
            max_x = max(x for x, _ in board_state.keys())
            max_y = max(y for _, y in board_state.keys())
            center_x = max_x / 2
            center_y = max_y / 2
            spread = sum(((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5 for x, y in positions) / len(positions)
            
            return {
                'count': len(positions),
                'avg_x': avg_x,
                'avg_y': avg_y,
                'spread': spread
            }
        
        for line in lines[1:]:  # Skip first line (winner)
            line = line.strip()
            if not line:
                continue
                
            # Parse turn number
            if line.startswith('turn:'):
                current_turn = int(line.split(':')[1])
                continue
                
            # Parse board position
            if line.startswith('pos'):
                pos, pieces = line.split(':')
                x, y = map(int, pos[3:].split('-'))
                
                # Update board state
                if pieces:
                    # Sort pieces by size (S < M < L) to handle captures
                    pieces_list = []
                    for piece in pieces.split(','):
                        player_id, piece_type = piece.split(':')
                        pieces_list.append({
                            'player': int(player_id),
                            'type': piece_type,
                            'size': {'S': 0, 'M': 1, 'L': 2}[piece_type]
                        })
                    pieces_list.sort(key=lambda p: p['size'], reverse=True)  # Larger pieces first
                    
                    # Add smaller pieces to captured list
                    for piece in pieces_list[1:]:
                        captured_pieces[piece['player']].append((x, y))
                    
                    # Keep only the largest piece at each position
                    board_state[(x, y)] = [pieces_list[0]]
                else:
                    board_state[(x, y)] = []
                    
                # Determine current player (player 1 starts, then alternates)
                current_player = 1 if current_turn % 2 == 1 else 2
                
                # Only save state if this was the winning player's move
                if current_player == winning_player:
                    # Generate features for this move
                    move_features = []
                    
                    # Add piece positions
                    piece_positions = {'player': {'on_board': [], 'not_on_board': [], 'captured': []},
                                     'enemy': {'on_board': [], 'not_on_board': [], 'captured': []}}
                    
                    for (pos_x, pos_y), pieces in board_state.items():
                        for piece in pieces:
                            if piece['player'] == winning_player:
                                piece_positions['player']['on_board'].append((pos_x, pos_y))
                            else:
                                piece_positions['enemy']['on_board'].append((pos_x, pos_y))
                    
                    # Add captured pieces
                    piece_positions['player']['captured'] = captured_pieces[winning_player]
                    piece_positions['enemy']['captured'] = captured_pieces[3 - winning_player]  # Other player
                    
                    # Calculate line and column counts
                    max_x = max(x for x, _ in board_state.keys())
                    max_y = max(y for _, y in board_state.keys())
                    line_counts = {'player': [0] * (max_y + 1), 'enemy': [0] * (max_y + 1)}
                    column_counts = {'player': [0] * (max_x + 1), 'enemy': [0] * (max_x + 1)}
                    
                    for (pos_x, pos_y), pieces in board_state.items():
                        for piece in pieces:
                            if piece['player'] == winning_player:
                                line_counts['player'][pos_y] += 1
                                column_counts['player'][pos_x] += 1
                            else:
                                line_counts['enemy'][pos_y] += 1
                                column_counts['enemy'][pos_x] += 1
                    
                    # Calculate piece statistics
                    piece_statistics = {
                        'player': {
                            'on_board': calculate_piece_stats(piece_positions['player']['on_board']),
                            'not_on_board': len(piece_positions['player']['not_on_board']),
                            'captured': len(piece_positions['player']['captured'])
                        },
                        'enemy': {
                            'on_board': calculate_piece_stats(piece_positions['enemy']['on_board']),
                            'not_on_board': len(piece_positions['enemy']['not_on_board']),
                            'captured': len(piece_positions['enemy']['captured'])
                        }
                    }
                                
                    move_features.append({
                        'name': 'piece_positions',
                        'value': piece_positions,
                        'type': 'global'
                    })
                    
                    move_features.append({
                        'name': 'line_counts',
                        'value': line_counts,
                        'type': 'global'
                    })
                    
                    move_features.append({
                        'name': 'column_counts',
                        'value': column_counts,
                        'type': 'global'
                    })
                    
                    move_features.append({
                        'name': 'piece_statistics',
                        'value': piece_statistics,
                        'type': 'global'
                    })
                    
                    # Add turn number
                    move_features.append({
                        'name': 'current_turn',
                        'value': current_turn,
                        'type': 'specific'
                    })
                    
                    winning_moves.append(move_features)
                
        return winning_moves 