import json
from Processing import Processing

class GameFeatures:
    """
    A class to manage and organize game features.
    """
    
    # Hardcoded game configuration
    BOARD_WIDTH = 3
    BOARD_LENGTH = 3
    PIECES = {
        'S': 2,  # Small pieces
        'M': 2,  # Medium pieces
        'L': 2   # Large pieces
    }
    
    def __init__(self):
        # Initialize empty features list
        self.features = []
        
        # Use hardcoded board dimensions and pieces
        self.width = self.BOARD_WIDTH
        self.length = self.BOARD_LENGTH
        self.pieces = self.PIECES.copy()
        
        # Create initial features
        self._create_initial_features()

    def _create_initial_features(self):
        """Create initial features for the game."""
        # Validate dimensions are positive
        if self.width <= 0 or self.length <= 0:
            raise ValueError("Board dimensions must be positive")
            
        # Validate pieces dictionary
        if not isinstance(self.pieces, dict):
            raise TypeError("Pieces must be provided as a dictionary")
        
        # Initialize piece tracking by size
        piece_positions = {
            'player': {
                'S': {'on_board': [], 'not_on_board': [], 'captured': []},
                'M': {'on_board': [], 'not_on_board': [], 'captured': []},
                'L': {'on_board': [], 'not_on_board': [], 'captured': []}
            },
            'enemy': {
                'S': {'on_board': [], 'not_on_board': [], 'captured': []},
                'M': {'on_board': [], 'not_on_board': [], 'captured': []},
                'L': {'on_board': [], 'not_on_board': [], 'captured': []}
            }
        }
        
        # Initialize line and column counts
        line_counts = {'player': [0] * self.length, 'enemy': [0] * self.length}
        column_counts = {'player': [0] * self.width, 'enemy': [0] * self.width}
        
        # Add all features
        self.features.extend([
            {
                'name': 'piece_positions',
                'value': piece_positions
            },
            {
                'name': 'line_counts',
                'value': line_counts
            },
            {
                'name': 'column_counts',
                'value': column_counts
            }
        ])

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
        # If feature doesn't exist, add it
        self.features.append({
            'name': feature_name,
            'value': value
        })
    
    def get_all_features(self):
        """
        Get all features.
        
        Returns:
            list: List of all features
        """
        return self.features

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
        captured_pieces = {'player': [], 'enemy': []}  # Track captured pieces for each player
        positions_read_for_turn = 0  # Track how many positions we've read for current turn
        total_positions = self.width * self.length  # Total positions on the board
        
        for line in lines[1:]:  # Skip first line (winner)
            line = line.strip()
            if not line:
                continue
                
            # Parse turn number
            if line.startswith('turn:'):
                # Start new turn
                current_turn = int(line.split(':')[1])
                positions_read_for_turn = 0
                continue
                
            # Parse board position
            if line.startswith('pos'):
                pos, pieces = line.split(':', 1)  # Split only on first colon
                x, y = map(int, pos[3:].split('-'))
                
                # Update board state
                if pieces:
                    # Sort pieces by size (S < M < L) to handle captures
                    pieces_list = []
                    
                    # Handle pieces - split by comma and filter out empty strings
                    piece_parts = pieces.split(',')
                    for piece_part in piece_parts:
                        piece_part = piece_part.strip()  # Remove whitespace
                        if piece_part:  # Only process non-empty pieces
                            player_id, piece_type = piece_part.split(':')
                            pieces_list.append({
                                'player': int(player_id),
                                'type': piece_type,
                                'size': {'S': 0, 'M': 1, 'L': 2}[piece_type]
                            })
                    
                    pieces_list.sort(key=lambda p: p['size'], reverse=True)  # Larger pieces first
                    
                    # Add smaller pieces to captured list
                    for piece in pieces_list[1:]:
                        if piece['player'] == winning_player:
                            captured_pieces['player'].append((x, y))
                        else:
                            captured_pieces['enemy'].append((x, y))
                    
                    # Keep only the largest piece at each position
                    board_state[(x, y)] = [pieces_list[0]]
                else:
                    board_state[(x, y)] = []
                
                positions_read_for_turn += 1
                
            # After reading all lines, check if the last turn was complete and was the winning player's move
            if current_turn > 0 and positions_read_for_turn == total_positions:
                current_player = 1 if current_turn % 2 == 1 else 2
                if current_player == winning_player:
                    # Generate features for this move using the Processing class
                    move_features = Processing.process_game_state(
                        board_state, 
                        captured_pieces, 
                        current_turn, 
                        winning_player,
                        self.width,
                        self.length
                    )
                    winning_moves.append(move_features)
                
        return winning_moves 
    
    def create_feature_lists(self, moves):
        """
        Create feature lists for each move of the winning player. functions read_game_transcript and create_feature_lists were deperated for easier handling.
        the feature lists consist of, in this order:
        - column 1 count
        - column 2 count
        - column 3 count
        - line 1 count
        - line 2 count
        - line 3 count
        - amount of small pieces on the board
        - amount of medium pieces on the board
        - amount of large pieces on the board
        - amount of small pieces captured
        - amount of medium pieces captured
        - amount of large pieces captured
        - amount of small pieces not on the board
        - amount of medium pieces not on the board
        - amount of large pieces not on the board
        - amount of small opponent pieces on the board
        - amount of medium opponent pieces on the board
        - amount of large opponent pieces on the board
        - amount of small opponent pieces captured
        - amount of medium opponent pieces captured
        - amount of large opponent pieces captured
        - average location of pieces on the board
        - spread of pieces on the board
        - location of each piece on the board

        - a 1-hot encoded vector of the game progress: (0,0,1) for the first third, (0,1,0) for the second third, (1,0,0) for the last third

        Args:
            moves (list): List of moves
            
        Returns:
            list: List of feature lists, one for each winning move
        """
        feature_lists = []
        
        # Calculate maximum turns from the number of moves
        # Number of lists times 2 is the number of moves
        max_turns = len(moves) * 2
        
        for move_features in moves:
            feature_list = []
            
            # Extract column counts (columns 1, 2, 3)
            column_counts = None
            for feature in move_features:
                if feature['name'] == 'column_counts':
                    column_counts = feature['value']
                    break
            
            if column_counts:
                # Add column counts for player (assuming player is the winning player)
                feature_list.extend(column_counts['player'])
            
            # Extract line counts (lines 1, 2, 3)
            line_counts = None
            for feature in move_features:
                if feature['name'] == 'line_counts':
                    line_counts = feature['value']
                    break
            
            if line_counts:
                # Add line counts for player
                feature_list.extend(line_counts['player'])
            
            # Extract piece counts by size for player
            player_pieces_on_board = {
                'S': 0, 'M': 0, 'L': 0
            }
            player_pieces_captured = {
                'S': 0, 'M': 0, 'L': 0
            }
            
            for feature in move_features:
                if feature['name'] == 'player_pieces_on_board_S':
                    player_pieces_on_board['S'] = feature['value']
                elif feature['name'] == 'player_pieces_on_board_M':
                    player_pieces_on_board['M'] = feature['value']
                elif feature['name'] == 'player_pieces_on_board_L':
                    player_pieces_on_board['L'] = feature['value']
                elif feature['name'] == 'player_pieces_captured_S':
                    player_pieces_captured['S'] = feature['value']
                elif feature['name'] == 'player_pieces_captured_M':
                    player_pieces_captured['M'] = feature['value']
                elif feature['name'] == 'player_pieces_captured_L':
                    player_pieces_captured['L'] = feature['value']
            
            # Add player piece counts on board
            feature_list.extend([
                player_pieces_on_board['S'],
                player_pieces_on_board['M'],
                player_pieces_on_board['L']
            ])
            
            # Add player piece counts captured
            feature_list.extend([
                player_pieces_captured['S'],
                player_pieces_captured['M'],
                player_pieces_captured['L']
            ])
            
            # Calculate pieces not on board (total pieces - on board - captured)
            player_pieces_not_on_board = {
                'S': self.PIECES['S'] - player_pieces_on_board['S'] - player_pieces_captured['S'],
                'M': self.PIECES['M'] - player_pieces_on_board['M'] - player_pieces_captured['M'],
                'L': self.PIECES['L'] - player_pieces_on_board['L'] - player_pieces_captured['L']
            }
            
            # Add player piece counts not on board
            feature_list.extend([
                player_pieces_not_on_board['S'],
                player_pieces_not_on_board['M'],
                player_pieces_not_on_board['L']
            ])
            
            # Extract piece counts by size for enemy
            enemy_pieces_on_board = {
                'S': 0, 'M': 0, 'L': 0
            }
            enemy_pieces_captured = {
                'S': 0, 'M': 0, 'L': 0
            }
            
            for feature in move_features:
                if feature['name'] == 'enemy_pieces_on_board_S':
                    enemy_pieces_on_board['S'] = feature['value']
                elif feature['name'] == 'enemy_pieces_on_board_M':
                    enemy_pieces_on_board['M'] = feature['value']
                elif feature['name'] == 'enemy_pieces_on_board_L':
                    enemy_pieces_on_board['L'] = feature['value']
                elif feature['name'] == 'enemy_pieces_captured_S':
                    enemy_pieces_captured['S'] = feature['value']
                elif feature['name'] == 'enemy_pieces_captured_M':
                    enemy_pieces_captured['M'] = feature['value']
                elif feature['name'] == 'enemy_pieces_captured_L':
                    enemy_pieces_captured['L'] = feature['value']
            
            # Add enemy piece counts on board
            feature_list.extend([
                enemy_pieces_on_board['S'],
                enemy_pieces_on_board['M'],
                enemy_pieces_on_board['L']
            ])
            
            # Add enemy piece counts captured
            feature_list.extend([
                enemy_pieces_captured['S'],
                enemy_pieces_captured['M'],
                enemy_pieces_captured['L']
            ])
            
            # Extract average location and spread
            avg_x = 0
            avg_y = 0
            spread = 0
            
            for feature in move_features:
                if feature['name'] == 'player_avg_x':
                    avg_x = feature['value']
                elif feature['name'] == 'player_avg_y':
                    avg_y = feature['value']
                elif feature['name'] == 'player_spread':
                    spread = feature['value']
            
            # Add average location (combine x and y into a single value or keep separate?)
            # For now, adding them separately as specified
            feature_list.append(avg_x)
            feature_list.append(avg_y)
            
            # Add spread
            feature_list.append(spread)
            
            # Add location of each piece on the board
            # This would need to be extracted from the board state
            # For now, we'll add a placeholder - this would need the actual board state
            # We can add the total number of pieces on board as a proxy
            total_pieces_on_board = sum(player_pieces_on_board.values())
            feature_list.append(total_pieces_on_board)
            
            # Add 1-hot encoded game progress
            current_turn = 0
            for feature in move_features:
                if feature['name'] == 'current_turn':
                    current_turn = feature['value']
                    break
            
            # Calculate game progress based on actual maximum turns from moves
            progress_ratio = current_turn / max_turns if max_turns > 0 else 0
            
            if progress_ratio <= 1/3:
                game_progress = [0, 0, 1]  # First third
            elif progress_ratio <= 2/3:
                game_progress = [0, 1, 0]  # Second third
            else:
                game_progress = [1, 0, 0]  # Last third
            
            feature_list.extend(game_progress)
            
            feature_lists.append(feature_list)
        
        return feature_lists