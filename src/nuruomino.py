from __future__ import annotations
# nuruomino.py: Projeto de Inteligência Artificial 2024/2025.
# Grupo 35:
# 109686 Miguel Raposo
# 110632 Inês Costa

import sys
import numpy as np
from search import Problem, depth_first_tree_search

class Tetromino:
    """Represents a Tetromino with a specific shape, rotation, and reflection."""
    TETROMINO_SHAPES = {
        "L": np.array([
            [1, 0],
            [1, 0],
            [1, 1]
        ]),
        "I": np.array([
            [1],
            [1],
            [1],
            [1]
        ]),
        "T": np.array([
            [1, 1, 1],
            [0, 1, 0]
        ]),
        "S": np.array([
            [0, 1, 1],
            [1, 1, 0]
        ])
    }

    def __init__(self, shape: str, rotation: int = 0, reflected: bool = False):
        self.shape = shape
        self.rotation = rotation
        self.reflected = reflected

    def __repr__(self):
        array = self.get()
        array_str = "\n" + "\n".join(""
            .join('•' if cell else ' ' for cell in row) for row in array)
        return (f'Tetromino(shape:{self.shape}, '
                f'rotation={self.rotation}º, '
                f'reflected={self.reflected}){array_str}')

    @staticmethod
    def normalize(tetromino_array):
        """Shift tetromino to top-left corner."""
        rows, cols = np.where(tetromino_array)
        return tetromino_array[rows.min():, cols.min():]

    @staticmethod
    def rotate(tetromino_array, degrees):
        """Rotate a tetromino by the given degrees (must be a multiple of 90)."""
        n = (degrees // 90) % 4
        return Tetromino.normalize(np.rot90(tetromino_array, -n))

    @staticmethod
    def reflect(tetromino_array):
        """Applies a vertical reflection."""
        return Tetromino.normalize(np.fliplr(tetromino_array))

    def get(self):
        """Applies rotation and reflection, returning a set
        of final coordinates for the tetromino."""
        tetromino_array = Tetromino.TETROMINO_SHAPES[self.shape]
        tetromino_array = Tetromino.rotate(tetromino_array, self.rotation)
        if self.reflected:
            tetromino_array = Tetromino.reflect(tetromino_array)
        return tetromino_array

    @staticmethod
    def get_all_orientations():
        """Returns all unique orientations of Tetromino pieces, normalized."""
        tetrominos_arrays = []
        seen = set()
        for shape in Tetromino.TETROMINO_SHAPES:
            for reflected in (False, True):
                for rotation in range(0, 360, 90):
                    tetromino = Tetromino(shape, rotation, reflected)
                    tetromino_array = tetromino.get()
                    tetromino_id = (shape, tetromino_array.shape,\
                                    tuple(tetromino_array.flatten()))
                    if tetromino_id not in seen:
                        seen.add(tetromino_id)
                        tetrominos_arrays.append(tetromino)
        return tetrominos_arrays

class Board:
    """Internal representation of a Nuruomino Puzzle board
    using only numpy arrays."""

    def __init__(self, board: np.matrix):
        self.board = board

    def __repr__(self):
        """Returns a string representation of the board."""
        return "\n".join("\t".join(str(cell)
                for cell in row) for row in self.board)

    @staticmethod
    def parse_instance():
        """Reads the board from stdin as a numpy matrix
        and returns a Board instance."""
        return Board(np.loadtxt(sys.stdin, dtype=np.int16).astype(object))

    def copy(self):
        """Returns a copy of the board."""
        return Board(self.board.copy())

    def get_board(self):
        """Get the baord numpy array."""
        return self.board

    def get_value(self, row: int, col: int):
        """Returns the value of a given cell on the board."""
        return self.board[row, col]

    def _get_cross_adjacent_coordinates(self, row: int, col: int):
        """Returns the coordinates of the cells adjacent to the given cell."""
        adjacent_coords = []
        rows, cols = self.board.shape
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < rows and 0 <= new_col < cols:
                adjacent_coords.append((new_row, new_col))
        return adjacent_coords

    def get_cross_adjacent_values(self, row: int, col: int):
        """Returns the values of the cells adjacent to the given cell."""
        return [self.get_value(r, c)
                for r, c in self._get_cross_adjacent_coordinates(row, col)]

    def get_current_adjacent_regions(self, region: int, og_board: 'Board', filled_regions: set) -> set:
        """Returns the regions ids adjacent to the current region."""
        adjacent_coordinates = set()
        for row, col in region:
            coordinates = self._get_cross_adjacent_coordinates(row, col)
            for coordinate in coordinates:
                if coordinate not in region:
                    adjacent_coordinates.add(coordinate)
        adjacent_filled_regions = set()
        for row, col in adjacent_coordinates:
            value = self.get_value(row, col)
            if isinstance(value, int) and value not in filled_regions:
                adjacent_filled_regions.add(value)
            elif value in Tetromino.TETROMINO_SHAPES:
                original_value = og_board.get_value(row, col)
                adjacent_filled_regions.add(original_value)
        return adjacent_filled_regions

    def tetromino_fits_in_region(self, action: Action) -> bool:
        """Check if all coordinates filled in action are within the
        specified region."""
        for row, col in action.get_position():
            if not (0 <= row < self.board.shape[0] and 0 <= col < self.board.shape[1]):
                return False
            if self.board[row, col] != action.get_region():
                return False
        return True

    def num_shared_edges(self, action: Action):
        """Returns the number of tetromino cells that touch
        the edge of another region."""
        num_contacts = 0
        for row, col in action.get_position():
            adjacent_values = self.get_cross_adjacent_values(row, col)
            if any(value != action.get_region() for value in adjacent_values):
                num_contacts += 1
        return num_contacts

    def _get_cross_adjacent_coordinates(self, row: int, col: int):
        """Returns the coordinates of the cells adjacent to the given cell."""
        adjacent_coordinates = []
        rows, cols = self.board.shape
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < rows and 0 <= new_col < cols:
                adjacent_coordinates.append((new_row, new_col))
        return adjacent_coordinates

    def run_action(self, action: Action) -> Board:
        """Returns a new board with the tetromino placed
        at the specified position."""
        for row, col in action.get_position():
            self.board[row, col] = action.tetromino.shape
        return Board(self.board)

    def get_new_adjacent_regions(self, region: int, og_board: 'Board', filled_regions: set) -> set:
        """Returns the ids of the regions adjacent to the given region."""
        adjacent_coordinates = set()
        for row, col in region:
            coordinates = self._get_cross_adjacent_coordinates(row, col)
            for coordinate in coordinates:
                if coordinate not in region:
                    adjacent_coordinates.add(coordinate)
        new_adjacent_regions = set()
        for row, col in adjacent_coordinates:
            value = self.get_value(row, col)
            if isinstance(value, int) and value not in filled_regions:
                new_adjacent_regions.add(value)
            elif value in Tetromino.TETROMINO_SHAPES:
                original_value = og_board.get_value(row, col)
                new_adjacent_regions.add(original_value)
        return new_adjacent_regions

class Action:
    """Represents placing a Tetromino at a specific location."""
    def __init__(self, region:int, tetromino:Tetromino, position:list[tuple[int,int]]):
        self.region = region
        self.tetromino = tetromino
        self.position = position

    def __repr__(self):
        pos_str = [(int(r), int(c)) for r, c in self.position]
        return (f'Action(region={self.region}, '
                f'tetromino_shape={self.tetromino.shape}, '
                f'position={pos_str})')

    def get_region(self):
        """Gets the region id for the action."""
        return self.region

    def get_position(self):
        """Gets the action's position."""
        return self.position

    def is_valid(self, board: Board) -> bool:
        """Returns whether a action is valid or not."""
        return board.tetromino_fits_in_region(self) and board.num_shared_edges(self) > 0

    def is_currently_valid(self, board: Board) -> bool:
        """Check if action doesn't overlap already filled cells, doesn't
        create 2x2 filled cells and doesn't touch pieces with the same shape."""
        if not board.tetromino_fits_in_region(self):
            return False
        if self._creates_filled_2x2_region(board):
            return False
        if self._touches_same_tetromino_shape(board):
            return False
        return True

    def _creates_filled_2x2_region(self, board: Board) -> bool:
        """Check if placing this tetromino would create
        any 2x2 filled areas, without modifying the board."""
        board_array = board.board
        mask = np.zeros(board_array.shape, dtype=bool)
        for shape in Tetromino.TETROMINO_SHAPES:
            mask |= (board_array == shape)
        rows, cols = zip(*self.position)
        mask[rows, cols] = True
        filled_2x2 = mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]
        return np.any(filled_2x2)

    def _touches_same_tetromino_shape(self, board: Board) -> bool:
        """Check if tetromino touches other tetronimo pieces with the same shape."""
        for row, col in self.position:
            adjacent_values = board.get_cross_adjacent_values(row, col)
            if self.tetromino.shape in adjacent_values:
                return True
        return False

class Nuruomino(Problem):
    """Represents the Nuruomino puzzle as a search problem."""
    def __init__(self, board: Board):
        self.og_board = board
        self._orientations = self._all_tetrominos_orientations()
        adjacency_graph = self._build_adjacency_graph()
        self.regions = adjacency_graph.keys()
        self.current_state = NuruominoState(board,
            self._generate_all_actions(adjacency_graph), adjacency_graph, set())
        super().__init__(self.current_state)

    def value(self, state):
        pass

    def result(self, state, action):
        """Return the state that results from executing the given action."""
        new_state = NuruominoState(
            board=state.get_board().copy(),
            actions_graph=state.get_actions_graph().copy(),
            adjacency_graph={key: value.copy()
                for key, value in state.get_adjacency_graph().items()}, # Deep Copy
            filled_regions=state.get_filled_regions().copy()
        )
        # Update baord.
        new_state.board.run_action(action)
        # Update filled regions.
        new_state.add_new_filled_region(action.get_region())
        # Update adjacency graph.
        new_adjacent_regions = new_state.board.get_new_adjacent_regions(
            action.get_position(), self.og_board, new_state.get_filled_regions())
        new_state.set_region_adjacencies(action.get_region(), new_adjacent_regions.copy())
        for neighbour in new_adjacent_regions:
            if neighbour in new_state.get_adjacency_graph():
                new_state.get_adjacencies(neighbour).add(action.get_region())
        for region, neighbours in new_state.get_adjacency_graph().items():
            if (region != action.get_region() and action.get_region() in neighbours
                and region not in new_adjacent_regions):
                neighbours.discard(action.get_region())
        # Update actions graph.
        new_state.set_actions(action.get_region(), [])
        for region in new_adjacent_regions:
            if region in new_state.get_actions_graph():
                valid_actions = []
                for new_actions in new_state.get_actions(region):
                    if new_actions.is_currently_valid(new_state.get_board()):
                        valid_actions.append(new_actions)
                new_state.set_actions(region, valid_actions)
        return new_state

    def goal_test(self, state):
        """Test if the given state is a goal state."""
        return len(self.regions) == len(state.get_filled_regions()) and\
            self._adjacency_graph_connected(state.get_adjacency_graph())

    def actions(self, state):
        """Return the actions available in the current state."""
        if not self._adjacency_graph_connected(state.get_adjacency_graph()):
            return []
        return state.get_actions(self._select_next_region(state))

    def _all_tetrominos_orientations(self):
        """Generate all possible orientations of Tetrominos."""
        return Tetromino.get_all_orientations()

    def _build_adjacency_graph(self):
        """Build an adjacency graph for the regions of the
        board using Board's helper methods."""
        adjacency_graph = {}
        rows, cols = self.og_board.get_board().shape
        for row in range(rows):
            for col in range(cols):
                region = self.og_board.get_value(row, col)
                if region not in adjacency_graph:
                    adjacency_graph[region] = set()
                for value in self.og_board.get_cross_adjacent_values(row, col):
                    if value != region:
                        adjacency_graph[region].add(value)
        return adjacency_graph

    def _generate_all_actions(self, adjacency_graph):
        """Generate all valid actions for each region."""
        all_actions = {}
        rows, cols = self.og_board.board.shape
        regions = adjacency_graph.keys()
        for region in regions:
            all_actions[region] = []
            for tetromino in self._orientations:
                tetromino_array = tetromino.get()
                shape_rows, shape_cols = tetromino_array.shape
                for row in range(rows - shape_rows + 1):
                    for col in range(cols - shape_cols + 1):
                        cells = [(row + dr, col + dc)
                            for dr, dc in zip(*np.where(tetromino_array))]
                        action = Action(region, tetromino, cells)
                        if action.is_valid(self.og_board):
                            all_actions[region].append(action)
        return all_actions

    def _select_next_region(self, state):
        regions = set(state.get_adjacency_graph().keys())
        regions -= state.get_filled_regions()
        min_actions = min(len(state.get_actions(region)) for region in regions)
        min_regions = [region for region in regions
            if len(state.get_actions(region)) == min_actions]

        if len(min_regions) == 0:
            return []
        if len(min_regions) == 1:
            return min_regions[0]
        return max(min_regions, key=lambda region:
            len(state.get_actions(region)), default=None)

    def _adjacency_graph_connected(self, adjacency_graph):
        """Return True if the adjacency graph is connected. False otherwise."""
        visited = set()
        queue = [1]
        while queue:
            region = queue.pop(0)
            if region in visited:
                continue
            visited.add(region)
            for neighbour in adjacency_graph[region]:
                if neighbour not in visited:
                    queue.append(neighbour)
        return len(visited) == len(adjacency_graph)

class NuruominoState:
    """Represents the state of the Nuruomino puzzle,
    including the board configuration and a unique state ID."""
    state_id = 0

    def __init__(self, board, actions_graph, adjacency_graph, filled_regions):
        self.id = NuruominoState.state_id
        NuruominoState.state_id += 1
        self.board = board
        self.actions_graph = actions_graph
        self.adjacency_graph = adjacency_graph
        self.filled_regions = filled_regions

    def __lt__(self, other):
        NuruominoState.state_id += 1
        return self.id < other.id

    def get_board(self):
        """Gets the current board."""
        return self.board

    def get_adjacency_graph(self):
        """Gets the current adjacency graph."""
        return self.adjacency_graph

    def get_actions_graph(self):
        """Get all actions available in this state."""
        return self.actions_graph

    def get_actions(self, region):
        """Get all actions available in this state, for the given region."""
        return self.actions_graph[region]

    def get_filled_regions(self):
        """Get all the regions already filled."""
        return self.filled_regions

    def remove_region_actions(self, region):
        """Remove a region from the actions graph."""
        if region in self.actions_graph:
            del self.actions_graph[region]

    def set_actions(self, region, actions):
        """Set the actions for a region in the actions graph."""
        self.actions_graph[region] = actions

    def get_adjacencies(self, region):
        """Get the adjacent regions for a given region."""
        return self.adjacency_graph[region]

    def set_region_adjacencies(self, region, adjacent_regions):
        """Update the adjacency graph for a given region in this state."""
        self.adjacency_graph[region] = adjacent_regions

    def add_new_filled_region(self, region):
        """Add a filled region to this state."""
        self.filled_regions.add(region)

def solve_nuruomino():
    """Parses input from stdin, creates a Nuruomino problem and solves it."""
    board = Board.parse_instance()
    problem = Nuruomino(board)
    goal_node = depth_first_tree_search(problem)
    if goal_node:
        print(goal_node.state.get_board())
    else:
        print("No solution found.")

if __name__ == "__main__":
    solve_nuruomino()
