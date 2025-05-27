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
        self.id = Nuroumino.state_id
        Nuroumino.state_id += 1

    def __lt__(self, other):
        """ 
            Este método é utilizado em caso de empate na gestão da lista
            de abertos nas procuras informadas.
        """
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
    """
        Internal representation of an action (placing a piece (Tetromino)
        at a specific location on the board).
    """
    def __init__(self, region:int, tetromino:Tetromino, position:list[tuple[int,int]]):
        self.region = region
        self.tetromino = tetromino
        self.position = position

    def __repr__(self):
        return (f'Action(region={self.region}, '
            f'tetromino_type={self.tetromino.tetronimo_type.name}, position={self.position})')

    def is_valid(self, board: Board) -> bool:
        """
            Verifies that the Tetromino can be placed entirely in the specified
            region and doesn't overlap with filled cells.
        """
        for row, col in self.position:
            if (row, col) not in board.regions[self.region]:
                return False
            if board.get_value(row, col) in ['L', 'I', 'T', 'S']:
                return False

    def does_overlap(self, other: 'Action') -> bool:
        """
            Checks if the current action positions overlaps with anothers one,
            can be used to ensure two Tetrominos occupy the same cells.
        """
        return any(position in other.position for position in self.position)

class Board:
    """Internal representation of a Nuruomino Puzzle board."""

    def __init__(self, board: list):
        self.board = board
        self.regions = self.get_regions(board)

    def get_regions(self, board):
        """Builds a dict of the regions present in the board."""
        regions = {}
        for row, line in enumerate(board):
            for col, region_num in enumerate(line):
                regions.setdefault(region_num, []).append((row, col))
        return regions

    @staticmethod
    def parse_instance():
        """
            Lê o test do standard input (stdin) que é passado como argumento
            e retorna uma instância da classe Board.

            Por exemplo:
                $ python3 pipe.py < test-01.txt

                > from sys import stdin
                > line = stdin.readline().split()
        """
        board = []
        for line in stdin.read().split("\n"):
            board.append(list(map(int, line.split("\t"))))
        return Board(board)

    def get_value(self, row:int, col:int) -> int:
        """Returns the value of the cell."""
        return self.board[row][col]

    def adjacent_positions(self, row:int, col:int) -> list:
        """
            Returns the adjacent positions to the region, in all directions,
            including diagonals.
        """
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
        """
            Returns the values of the cells adjacent to the region,
            in all directions, including diagonals.
        """
        adjacent_values = set()
        for new_row, new_column in self.adjacent_positions(row, col):
            adjacent_values.add(self.board[new_row][new_column])
        return adjacent_values

    def adjacent_regions(self, region:int) -> list:
        """Returns a list of regions that border the region provided as an argument."""
        adjacent_regions = set()
        for row, col in self.regions[region]:
            adjacent_regions.update(self.adjacent_values(row, col))
        adjacent_regions.discard(region)  # Don't include the region itself
        return list(adjacent_regions)

    def print_instance(self):
        """Prints the string representation of the board."""
        game_board = ""
        for row in self.board:
            for region in row:
                game_board += f"{region}\t"
            game_board += "\n"
        stdout.write(game_board)

    def copy(self):
        """Creates a deep copy of the board."""
        return copy.deepcopy(self)

    def place_tetromino(self, action: Action):
        """Places the tetromino of the given action on the board."""
        for row, col in action.position:
            self.board[row][col] = action.tetromino.tetronimo_type.name

    def check_region_filled(self, region_id):
        """Returns true if the region is filled with a tetromino."""
        tetromino_ids = [t.name for t in TetrominoType]
        return all(
            isinstance(self.board[row][col], str) and self.board[row][col] in tetromino_ids
            for row, col in self.regions[region_id]
        )

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

        # BFS to check connectivity
        visited = set()
        queue = [next(iter(filled_cells))]
        visited.add(next(iter(filled_cells)))
        orthogonal_directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while queue:
            r, c = queue.pop(0)
            for dr, dc in orthogonal_directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < len(self.board) and 0 <= nc < len(self.board[0]):
                    if ((nr, nc) in filled_cells) and ((nr, nc) not in visited):
                        visited.add((nr, nc))
                        queue.append((nr, nc))

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
                    region = next(reg for reg, cells in self.regions.items() if (row, col) in cells)
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < len(self.board) and 0 <= nc < len(self.board[0]):
                            neighbor_value = self.get_value(nr, nc)
                            if neighbor_value == value:
                                neighbor_region = next(reg for reg, cells in self.regions.items() if (nr, nc) in cells)
                                if region != neighbor_region:
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

        while filled_cells:
            start = next(iter(filled_cells))
            queue = [start]
            visited.add(start)
            while queue:
                r, c = queue.pop(0)
                for nr, nc in self.adjacent_positions(r, c):
                    if (nr, nc) in filled_cells and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
            filled_cells -= visited
            components += 1

        return components

    def __repr__(self):
        board_repr = "\n".join(" ".join(map(str, row)) for row in self.board)
        regions_repr = "\n".join(f"Region {region}: {positions}"
                                 for region, positions in self.regions.items())
        return f"Board:\n{board_repr}\n{regions_repr}"

class Nuruomino(Problem):
    """Puzzle where regions are filled with Tetrominoes; supports actions, results, goal tests, and heuristics."""

    def __init__(self, board: Board):
        """The constructor specifies the initial state."""
        self.board = board
        self.initial = NuruominoState(board)
        super().__init__(self.initial)

    def actions(self, state: NuruominoState):
        """
            Retorna uma lista de ações que podem ser executadas a
            partir do estado passado como argumento.
        """
        #TODO
        pass 

    def result(self, state: NuruominoState, action):
        """
            Retorna o estado resultante de executar a 'action' sobre
            'state' passado como argumento. A ação a executar deve ser uma
            das presentes na lista obtida pela execução de
            self.actions(state).
        """
        #TODO
        pass

    def goal_test(self, state: NuruominoState):
        """
            Retorna True se e só se o estado passado como argumento é
            um estado objetivo. Deve verificar se todas as posições do tabuleiro
            estão preenchidas de acordo com as regras do problema.
        """
        #TODO
        pass 

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass
