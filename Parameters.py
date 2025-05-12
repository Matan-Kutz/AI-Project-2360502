def createParameters(width, length, pieces):
    """
    Create parameters for a game board and its pieces.
    
    Args:
        width (int): The width of the game board
        length (int): The length of the game board
        pieces (dict): A dictionary containing different types of game pieces and their quantities
                      Example: {'pawn': 8, 'knight': 2, 'bishop': 2, 'rook': 2, 'queen': 1, 'king': 1}
        
    Returns:
        dict: A dictionary containing the board dimensions, piece configurations, and piece statistics
    """
    # Validate dimensions are positive
    if width <= 0 or length <= 0:
        raise ValueError("Board dimensions must be positive")
        
    # Validate pieces dictionary
    if not isinstance(pieces, dict):
        raise TypeError("Pieces must be provided as a dictionary")
    
    # Calculate total pieces
    total_pieces = sum(pieces.values())
    
    # Initialize piece tracking
    piece_positions = {
        'player': {
            'on_board': [],  # List of (x, y) positions
            'not_on_board': [],  # List of piece types waiting to be placed
            'captured': []  # List of captured pieces
        },
        'enemy': {
            'on_board': [],
            'not_on_board': [],
            'captured': []
        }
    }
    
    # Initialize line and column counts
    line_counts = {'player': [0] * length, 'enemy': [0] * length}
    column_counts = {'player': [0] * width, 'enemy': [0] * width}
    
    # Calculate piece statistics
    def calculate_piece_stats(positions):
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
        center_x = width / 2
        center_y = length / 2
        spread = sum(((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5 for x, y in positions) / len(positions)
        
        return {
            'count': len(positions),
            'avg_x': avg_x,
            'avg_y': avg_y,
            'spread': spread
        }
    
    parameters = {
        'board_dimensions': {
            'width': width,
            'length': length
        },
        'pieces': pieces,
        'total_pieces': total_pieces,
        'piece_positions': piece_positions,
        'line_counts': line_counts,
        'column_counts': column_counts,
        'piece_statistics': {
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
    }
    return parameters 