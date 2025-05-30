from __future__ import annotations
# nuruomino.py: Template para implementação do projeto de Inteligência Artificial 2024/2025.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 35:
# 110632 Inês Costa
# 109686 Miguel Raposo

from sys import stdin, stdout
from enum import Enum
import copy
from search import *

class NuruominoState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = NuruominoState.state_id 
        NuruominoState.state_id += 1

    def __lt__(self, other):
        """Used to break ties in informed search open lists."""
        return self.id < other.id

class TetrominoType(Enum):
    """Enum to represent the various types of Tetromino."""
    L = [(0,0), (1,0), (2,0), (2,1)]
    I = [(0,0), (1,0), (2,0), (3,0)]
    T = [(0,1), (1,0), (1,1), (1,2)]
    S = [(0,1), (0,2), (1,0), (1,1)]

class Tetromino:
    """Internal representation of a tetromino and its position on the board."""

    def __init__(self, tetronimo_type: TetrominoType, rotation: int = 0, refleced: bool = False):
        self.tetronimo_type = tetronimo_type
        self.rotation = rotation
        self.reflected = refleced

    def __repr__(self):
        return (f'Tetromino(type={self.tetronimo_type}, rotation={self.rotation} degrees, '
            f'reflected={self.reflected})')

    @staticmethod
    def rotate(tetromino, degrees):
        """Applies a rotation by the given value (degrees), assumes it is a multiple of 90."""
        for _ in range((degrees // 90) % 4): # Calculates the number of rotation to apply
            tetromino = [(column, -row) for row, column in tetromino]
        return tetromino

    @staticmethod
    def reflect(tetromino):
        """Applies a horizontal reflection."""
        return [(row, -column) for row, column in tetromino]

    @staticmethod
    def normalize(tetromino):
        """Aligns the tetromino coordinates to a standard position starting from (0,0)."""
        return [(row - min(row for row, _ in tetromino),
                 col - min(col for _, col in tetromino)) for row, col in tetromino]

    def get(self):
        """Applies rotation and reflection, returning a set of final coordinates for the tetromino."""
        tetromino = self.tetronimo_type.value
        tetromino = Tetromino.rotate(tetromino, self.rotation)
        if self.reflected:
            tetromino = Tetromino.reflect(tetromino)
        return Tetromino.normalize(tetromino)

class Action:
    """Represents placing a Tetromino at a specific location."""
    def __init__(self, region:int, tetromino:Tetromino, position:list[tuple[int,int]]):
        self.region = region
        self.tetromino = tetromino
        self.position = position

    def __repr__(self):
        return (f'Action(region={self.region}, '
            f'tetromino_type={self.tetromino.tetronimo_type.name}, position={self.position})')

    def is_valid(self, board: Board) -> bool:
        """Check if Tetromino fits in region and doesn't overlap filled cells."""
        for row, col in self.position:
            if row < 0 or col < 0 or row >= len(board.board) or col >= len(board.board[0]):
                return False
            if (row, col) not in board.regions[self.region]:
                return False
            value = board.get_value(row, col)
            if isinstance(value, str) and value in ['L', 'I', 'T', 'S']:
                return False
        return True 

    def does_overlap(self, other: 'Action') -> bool:
        """Check if this action overlaps with another action."""
        return any(position in other.position for position in self.position)

class Board:
    """Internal representation of a Nuruomino Puzzle board."""

    def __init__(self, board: list):
        self.board = board
        # self.regions: Maps region_id -> list of (row, col) cells in that region.
        self.regions = self._get_regions(board)
        # self._cell_to_region: (row, col) -> region_id for fast region lookup.
        self.cell_to_region = self._build_cell_to_region()
        self.size = len(board)

    def _build_cell_to_region(self):
        """Builds a mapping from cell coordinates to their region ID."""
        cell_to_region = {}
        for region_id, cells in self.regions.items():
            for cell in cells:
                cell_to_region[cell] = region_id
        return cell_to_region

    def _get_regions(self, board):
        """Builds a dict of the regions present in the board."""
        regions = {}
        for row, line in enumerate(board):
            for col, region_num in enumerate(line):
                regions.setdefault(region_num, []).append((row, col))
        return regions

    @staticmethod
    def parse_instance():
        """Reads the board from stdin and returns a Board instance."""
        board = []
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            row = [int(x) for x in line.split("\t") if x]
            board.append(row)
        return Board(board)

    def get_value(self, row:int, col:int) -> int:
        """Returns the value of the cell."""
        return self.board[row][col]

    def adjacent_positions(self, row:int, col:int) -> list:
        """Returns the adjacent positions to the region, in all directions."""
        adjacent_coordinates = [(-1,0), (-1,-1), (0,-1), (1,-1),
                                (1,0), (1,1), (0,1), (-1,1)]
        adjacent_positions = []
        board_size = len(self.board)
        for drow, dcolumn in adjacent_coordinates:
            new_row, new_column = row + drow, col + dcolumn
            if 0 <= new_row < board_size and 0 <= new_column < board_size:
                adjacent_positions.append((new_row, new_column))
        return adjacent_positions

    def adjacent_values(self, row:int, col:int) -> list:
        """Returns values of cells adjacent to (row, col)."""
        adjacent_values = set()
        for new_row, new_column in self.adjacent_positions(row, col):
            adjacent_values.add(self.board[new_row][new_column])
        return adjacent_values

    def adjacent_regions(self, region:int) -> list:
        """Returns a list of regions that border the region provided as an argument."""
        adjacent_regions = set()
        for row, col in self.regions[region]:
            adjacent_regions.update(self.adjacent_values(row, col))
        adjacent_regions.discard(region)
        adjacent_regions = {r for r in adjacent_regions if isinstance(r, int)}
        return list(adjacent_regions)

    def print_instance(self):
        """Prints the string representation of the board."""
        game_board = ""
        for row in self.board:
            row_str = ""
            for region in row:
                row_str += f"{region}\t"
            game_board += row_str.rstrip() + "\n"
        stdout.write(game_board)

    def copy(self):
        """Creates a deep copy of the board."""
        return copy.deepcopy(self)

    def place_tetromino(self, action: Action):
        """Places the tetromino of the given action on the board."""
        for row, col in action.position:
            self.board[row][col] = action.tetromino.tetronimo_type.name

    def check_region_filled(self, region_id):
        """Returns true if the region contains exactly 4 filled cells forming a tetromino."""
        tetromino_ids = [t.name for t in TetrominoType]
        filled_cells = [(row, col) for row, col in self.regions[region_id] 
                    if isinstance(self.board[row][col], str) and self.board[row][col] in tetromino_ids]
        return len(filled_cells) == 4

    def check_all_regions_filled(self):
        """Retruns true if all the regions are filled with a tetromino."""
        for region_id in self.regions:
            if not self.check_region_filled(region_id):
                return False
        return True

    def is_connected(self):
        """Check if all filled cells form a single connected group."""
        tetromino_ids = {t.name for t in TetrominoType}
        filled_cells = set()
        for row in range(len(self.board)):
            for col in range(len(self.board[0])):
                if isinstance(self.get_value(row, col), str) and self.get_value(row, col) in tetromino_ids:
                    filled_cells.add((row, col))

        if not filled_cells:
            return True

        visited = set()
        start_cell = next(iter(filled_cells))
        queue = [start_cell]
        visited.add(start_cell)
        orthogonal_directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while queue:
            current_row, current_col = queue.pop(0)
            for d_row, d_col in orthogonal_directions:
                neighbor_row, neighbor_col = current_row + d_row, current_col + d_col
                if (0 <= neighbor_row < len(self.board) and
                    0 <= neighbor_col < len(self.board[0])):
                    neighbor = (neighbor_row, neighbor_col)
                    if neighbor in filled_cells and neighbor not in visited:
                        queue.append(neighbor)
                        visited.add(neighbor)

        return len(visited) == len(filled_cells)

    def check_filled_square(self):
        """Check if there are any 2x2 squares fully filled in the board with tetromino cells."""
        tetromino_ids = {t.name for t in TetrominoType}
        for row in range(len(self.board) - 1):
            for col in range(len(self.board[0]) - 1):
                positions = [(row, col), (row, col+1), (row+1, col), (row+1, col+1)]
                filled = True
                for r, c in positions:
                    value = self.get_value(r, c)
                    if not (isinstance(value, str) and value in tetromino_ids):
                        filled = False
                        break
                if filled:
                    return True
        return False

    def check_adjacent_pieces_equal(self):
        """Return True if any orthogonally adjacent cells from different regions have the same tetromino type."""
        tetromino_ids = {t.name for t in TetrominoType}
        for row in range(len(self.board)):
            for col in range(len(self.board[0])):
                value = self.get_value(row, col)
                if value in tetromino_ids:
                    for delta_row, delta_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        neighbor_row = row + delta_row
                        neighbor_col = col + delta_col
                        if 0 <= neighbor_row < len(self.board) and 0 <= neighbor_col < len(self.board[0]):
                            neighbor_value = self.get_value(neighbor_row, neighbor_col)
                            if neighbor_value == value:
                                current_region = self.cell_to_region.get((row, col))
                                neighbor_region = self.cell_to_region.get((neighbor_row, neighbor_col))
                                if current_region != neighbor_region:
                                    return True
        return False

    def count_regions_not_filled(self):
        """Return the number of regions not yet filled."""
        return sum(not self.check_region_filled(region_id) for region_id in self.regions)

    def count_connected_tetrominos(self):
        """Count the number of connected groups of filled tetromino cells."""
        tetromino_ids = {t.name for t in TetrominoType}
        filled_cells = set()
        for row in range(len(self.board)):
            for col in range(len(self.board[0])):
                value = self.get_value(row, col)
                if isinstance(value, str) and value in tetromino_ids:
                    filled_cells.add((row, col))

        if not filled_cells:
            return 0
        visited = set()
        components = 0
        remaining = filled_cells.copy()
        while remaining:
            start = next(iter(remaining))
            queue = [start]
            component = {start}
            visited.add(start)
            while queue:
                current_row, current_col = queue.pop(0)
                for delta_row, delta_col in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    neighbor_row, neighbor_col = current_row + delta_row, current_col + delta_col
                    if (neighbor_row, neighbor_col) in remaining and (neighbor_row, neighbor_col) not in component:
                        queue.append((neighbor_row, neighbor_col))
                        component.add((neighbor_row, neighbor_col))
                        visited.add((neighbor_row, neighbor_col))
            remaining = remaining - component
            components += 1
        return components

    def __repr__(self):
        board_repr = "\n".join(" ".join(map(str, row)) for row in self.board)
        regions_repr = "\n".join(f"Region {region}: {positions}"
                                 for region, positions in self.regions.items())
        return f"Board:\n{board_repr}\n{regions_repr}"

class Nuruomino(Problem):
    """Puzzle where regions are filled with Tetrominoes; supports actions, results, goal tests, and heuristics."""

    def __init__(self, board: Board, on_state=None):
        """The constructor specifies the initial state."""
        self.board = board
        self.initial = NuruominoState(board)
        self.on_state = on_state
        self._orientations = self._all_tetrominos_orientations()
        super().__init__(self.initial)

    def _all_tetrominos_orientations(self):
        """Pre-compute all tetromino orientations once."""
        all_orientations = {}
        for tetromino_type in TetrominoType:
            shapes = []
            shapes_orientations = []
            for rotation in [0, 90, 180, 270]:
                for reflected in [False, True]:
                    tetromino = Tetromino(tetromino_type, rotation, reflected)
                    shape = tetromino.get()
                    is_unique = True
                    for existing_shape in shapes:
                        if sorted(shape) == sorted(existing_shape):
                            is_unique = False
                            break
                    if is_unique:
                        shapes.append(shape)
                        shapes_orientations.append((tetromino, shape))
            all_orientations[tetromino_type.name] = shapes_orientations
        return all_orientations

    def get_all_tetromino_orientations(self):
        """Returns all possible orientations."""
        return self._orientations

    def _no_adjacent_same_tetromino(self, state, action):
        """Return False if action would place a tetromino adjacent to the same type in a different region."""
        tetromino_type = action.tetromino.tetronimo_type.name
        for row, col in action.position:
            for delta_row, delta_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor_row, neighbor_col = row + delta_row, col + delta_col
                if 0 <= neighbor_row < state.board.size and 0 <= neighbor_col < state.board.size:
                    neighbor_value = state.board.get_value(neighbor_row, neighbor_col)
                    if neighbor_value == tetromino_type:
                        neighbor_region = state.board.cell_to_region.get((neighbor_row, neighbor_col))
                        if neighbor_region != action.region:
                            return False
        return True

    def _creates_filled_2x2_square(self, state, action):
        """Return True if placing the action would create a filled 2x2 square."""
        action_positions = set(action.position)
        for row, col in action.position:
            for offset_row in [-1, 0]:
                for offset_col in [-1, 0]:
                    square = [
                        (row + offset_row, col + offset_col),
                        (row + offset_row, col + offset_col + 1),
                        (row + offset_row + 1, col + offset_col),
                        (row + offset_row + 1, col + offset_col + 1)
                    ]
                    if all(0 <= square_row < state.board.size and 0 <= square_col < state.board.size for square_row, square_col in square):
                        filled = sum(
                            (square_row, square_col) in action_positions or
                            (isinstance(state.board.get_value(square_row, square_col), str) and state.board.get_value(square_row, square_col) in ['L', 'I', 'T', 'S'])
                            for square_row, square_col in square
                        )
                        if filled == 4:
                            return True
        return False

    def is_valid_action(self, state, action):
        """Check if the action is valid - optimized with early exits."""
        if not action.is_valid(state.board):
            return False
        if not self._no_adjacent_same_tetromino(state, action):
            return False
        if self._creates_filled_2x2_square(state, action):
            return False
        return True

    def _select_next_region(self, state):
        """Select the next region to fill: the one with the fewest empty cells (at least 4, not filled)."""
        candidate = None
        min_empty = float('inf')
        for region_id, cells in state.board.regions.items():
            filled = [cell for cell in cells if isinstance(state.board.get_value(*cell), str) and state.board.get_value(*cell) in [piece.name for piece in TetrominoType]]
            empty = [cell for cell in cells if isinstance(state.board.get_value(*cell), int)]
            if len(filled) == 0 and len(empty) >= 4 and len(empty) < min_empty:
                candidate = (region_id, empty)
                min_empty = len(empty)
        return candidate if candidate else (None, None)

    def _generate_region_actions(self, state, region_id, region_cells):
        """Generate all valid actions for a given region - optimized."""
        region_cell_set = set(region_cells)
        actions_list = []
        for tetromino_type in TetrominoType:
            for tetromino, shape in self._orientations[tetromino_type.name]:
                for anchor_cell in region_cells:
                    anchor_row, anchor_col = anchor_cell
                    offset_row, offset_col = shape[0]
                    base_row = anchor_row - offset_row
                    base_col = anchor_col - offset_col
                    positions = [(base_row + row, base_col + col) for row, col in shape]
                    if all(pos in region_cell_set for pos in positions):
                        action = Action(region_id, tetromino, positions)
                        if self.is_valid_action(state, action):
                            actions_list.append(action)
        return actions_list

    def _unique_actions(self, actions_list):
        """Filter actions to ensure uniqueness by position."""
        unique_actions = []
        seen = set()
        for action in actions_list:
            pos_tuple = tuple(sorted(action.position))
            if pos_tuple not in seen:
                seen.add(pos_tuple)
                unique_actions.append(action)
        return unique_actions

    def actions(self, state):
        if self.goal_test(state):
            return []
        region_id, region_cells = self._select_next_region(state)
        if region_id is None:
            return []
        actions_list = self._generate_region_actions(state, region_id, region_cells)
        return self._unique_actions(actions_list)

    def result(self, state, action):
        """Return the state that results from executing the given action."""
        new_board = Board(copy.deepcopy(state.board.board))
        new_board.regions = state.board.regions
        new_board.cell_to_region = state.board.cell_to_region
        for row, col in action.position:
            new_board.board[row][col] = action.tetromino.tetronimo_type.name
        new_state = NuruominoState(new_board)
        if self.on_state:
            self.on_state(new_board)
        return new_state

    def goal_test(self, state):
        """Test if the given state is a goal state."""
        if not state.board.check_all_regions_filled():
            return False
        if not state.board.is_connected():
            return False
        if state.board.check_filled_square():
            return False
        if state.board.check_adjacent_pieces_equal():
            return False
        return True

    def value(self, state):
        filled = sum(
            self.board.check_region_filled(region_id)
            for region_id in state.board.regions
        )
        penalty = 0
        if not state.board.is_connected():
            penalty += 100
        if state.board.check_filled_square():
            penalty += 100
        if state.board.check_adjacent_pieces_equal():
            penalty += 100
        return filled - penalty

    def h(self, node):
        """Optimized heuristic function for informed search."""
        state = node.state
        unfilled_regions = state.board.count_regions_not_filled()
        if unfilled_regions == 0:
            return 0
        if unfilled_regions <= 3:
            components = state.board.count_connected_tetrominos()
            if components > 1:
                return unfilled_regions + (components - 1)
        return unfilled_regions


def solve_nuruomino(problem):
    """Optimized search strategy selection."""
    actual_problem = problem.problem if hasattr(problem, 'problem') else problem
    num_regions = len(actual_problem.board.regions)
    search_problem = actual_problem if (hasattr(actual_problem, 'on_state') and actual_problem.on_state) else problem
    four_cell_regions = sum(1 for cells in actual_problem.board.regions.values() if len(cells) == 4)
    total_cells = sum(len(cells) for cells in actual_problem.board.regions.values())
    if num_regions <= 5:
        return astar_search(search_problem, h=actual_problem.h)
    elif four_cell_regions >= num_regions * 0.7:
        return astar_search(search_problem, h=actual_problem.h)
    elif total_cells <= 60:
        return astar_search(search_problem, h=actual_problem.h)
    else:
        return depth_first_graph_search(search_problem)

if __name__ == "__main__":
    game_board = Board.parse_instance()
    problem = Nuruomino(game_board)
    initial_state = problem.initial

    instrumented = InstrumentedProblem(problem)
    solution = solve_nuruomino(instrumented)

    if solution:
        solution.state.board.print_instance()
