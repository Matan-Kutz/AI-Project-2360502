class Processing:
    """
    A class to process game states and generate features from game logs.
    """
    
    @staticmethod
    def process_game_state(board_state, captured_pieces, current_turn, winning_player, width, length):
        """
        Process a game state and generate features for it.
        
        Args:
            board_state (dict): Current state of the board
            captured_pieces (dict): Dictionary of captured pieces for each player
            current_turn (int): Current turn number
            winning_player (int): ID of the winning player (1 or 2)
            width (int): Board width
            length (int): Board length
            
        Returns:
            list: List of features for this game state
        """
        move_features = []
        
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
        
        # Track pieces by size on the board
        for (pos_x, pos_y), pieces in board_state.items():
            for piece in pieces:
                piece_type = piece['type']
                if piece['player'] == winning_player:
                    piece_positions['player'][piece_type]['on_board'].append((pos_x, pos_y))
                else:
                    piece_positions['enemy'][piece_type]['on_board'].append((pos_x, pos_y))
        
        # Calculate line and column counts
        line_counts = {'player': [0] * length, 'enemy': [0] * length}
        column_counts = {'player': [0] * width, 'enemy': [0] * width}
        
        for (pos_x, pos_y), pieces in board_state.items():
            for piece in pieces:
                if piece['player'] == winning_player:
                    line_counts['player'][pos_y] += 1
                    column_counts['player'][pos_x] += 1
                else:
                    line_counts['enemy'][pos_y] += 1
                    column_counts['enemy'][pos_x] += 1
        
        # Calculate piece counts by size for player
        player_pieces_on_board = {
            'S': len(piece_positions['player']['S']['on_board']),
            'M': len(piece_positions['player']['M']['on_board']),
            'L': len(piece_positions['player']['L']['on_board'])
        }
        
        # For captured pieces, we need to track them by type during the game
        # For now, we'll use a simplified approach - count total captured
        player_pieces_captured = {
            'S': 0,  # This would need to be tracked during game progression
            'M': 0,
            'L': 0
        }
        
        # Calculate piece counts by size for enemy
        enemy_pieces_on_board = {
            'S': len(piece_positions['enemy']['S']['on_board']),
            'M': len(piece_positions['enemy']['M']['on_board']),
            'L': len(piece_positions['enemy']['L']['on_board'])
        }
        
        enemy_pieces_captured = {
            'S': 0,  # This would need to be tracked during game progression
            'M': 0,
            'L': 0
        }
        
        # Calculate average coordinates and spread for player pieces
        all_player_positions = []
        for size in ['S', 'M', 'L']:
            all_player_positions.extend(piece_positions['player'][size]['on_board'])
        
        if all_player_positions:
            avg_x = sum(pos[0] for pos in all_player_positions) / len(all_player_positions)
            avg_y = sum(pos[1] for pos in all_player_positions) / len(all_player_positions)
            
            # Calculate spread (average distance from center)
            center_x = width / 2
            center_y = length / 2
            spread = sum(((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5 for x, y in all_player_positions) / len(all_player_positions)
        else:
            avg_x = 0
            avg_y = 0
            spread = 0
        
        # Add all features to the list
        move_features.extend([
            # Line and column counts
            {
                'name': 'line_counts',
                'value': line_counts
            },
            {
                'name': 'column_counts',
                'value': column_counts
            },
            
            # Player piece counts by size
            {
                'name': 'player_pieces_on_board_S',
                'value': player_pieces_on_board['S']
            },
            {
                'name': 'player_pieces_on_board_M',
                'value': player_pieces_on_board['M']
            },
            {
                'name': 'player_pieces_on_board_L',
                'value': player_pieces_on_board['L']
            },
            {
                'name': 'player_pieces_captured_S',
                'value': player_pieces_captured['S']
            },
            {
                'name': 'player_pieces_captured_M',
                'value': player_pieces_captured['M']
            },
            {
                'name': 'player_pieces_captured_L',
                'value': player_pieces_captured['L']
            },
            
            # Enemy piece counts by size
            {
                'name': 'enemy_pieces_on_board_S',
                'value': enemy_pieces_on_board['S']
            },
            {
                'name': 'enemy_pieces_on_board_M',
                'value': enemy_pieces_on_board['M']
            },
            {
                'name': 'enemy_pieces_on_board_L',
                'value': enemy_pieces_on_board['L']
            },
            {
                'name': 'enemy_pieces_captured_S',
                'value': enemy_pieces_captured['S']
            },
            {
                'name': 'enemy_pieces_captured_M',
                'value': enemy_pieces_captured['M']
            },
            {
                'name': 'enemy_pieces_captured_L',
                'value': enemy_pieces_captured['L']
            },
            
            # Player piece coordinates and spread
            {
                'name': 'player_avg_x',
                'value': avg_x
            },
            {
                'name': 'player_avg_y',
                'value': avg_y
            },
            {
                'name': 'player_spread',
                'value': spread
            },
            
            # Turn number
            {
                'name': 'current_turn',
                'value': current_turn
            }
        ])
        
        return move_features

    @staticmethod
    def process_game_log(log_file, width, length):
        """
        Process a game log file and generate features for each game state.
        Only includes states from the winning player's moves.
        
        Args:
            log_file (str): Path to the game log file
            width (int): Board width
            length (int): Board length
            
        Returns:
            list: List of processed game states from winning player's moves, where each state is a list of features
            
        Raises:
            ValueError: If the log file format is invalid
        """
        # Read the transcript file
        with open(log_file, 'r') as f:
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
        winning_moves = []  # List to store features for winning player's moves
        captured_pieces = {1: [], 2: []}  # Track captured pieces for each player
        
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
                
                # Only process state if it's the winning player's move
                if current_player == winning_player:
                    game_state = Processing.process_game_state(
                        board_state,
                        captured_pieces,
                        current_turn,
                        winning_player,
                        width,
                        length
                    )
                    winning_moves.append(game_state)
                
        return winning_moves 