from __future__ import annotations
# nuruomino.py: Template para implementação do projeto de Inteligência Artificial 2024/2025.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 35:
# 110632 Inês Costa
# 109686 Miguel Raposo

from sys import stdin, stdout
from enum import Enum
from collections import deque
import numpy as np
from search import Problem, depth_first_tree_search

class Tetromino:
    """Represention of a Tetromino piece."""
    @staticmethod
    def create_bitmask(coords):
        """Convert a list (row, cols) to a 4x4 numpy array."""
        np_array = np.zeros((4, 4), dtype=int)
        for row, col in coords:
            np_array[row, col] = 1
        return tuple(tuple(row) for row in np_array)

    @staticmethod
    def to_numpy(mask):
        """Convert the enum tuple of tuples to a numpy bitmask array"""
        return np.array(mask, dtype=int)

    @staticmethod
    def rotate(mask, k=1):
        """Rotate the bitmask 90 degrees clockwise k times."""
        return np.rot90(mask, -k)

    @staticmethod
    def reflect(mask):
        """Reflect the bitmask horizontally."""
        return np.fliplr(mask)

    @staticmethod
    def normalize(mask):
        """Shift the mask so the top-left 1 is at (0,0)."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        min_row = np.argmax(rows)
        min_col = np.argmax(cols)
        normal = mask[min_row:min_row + 4, min_col:min_col + 4]
        complete = np.zeros((4, 4), dtype=int) # Pads to make 4x4 np normal array.
        num_rows, num_cols = normal.shape
        complete[:num_rows, :num_cols] = normal
        return complete

    @staticmethod
    def get_all_forms(mask):
        """Returns all the unique orientations of a mask (Tetromino)."""
        seen = set()
        orientations = []
        for reflect in [True, False]:
            forms_array = Tetromino.reflect(mask) if reflect else mask
            for angle in range(4):
                rotation = Tetromino.rotate(forms_array, angle)
                normal = Tetromino.normalize(rotation)
                key = tuple(normal.flatten())
                if key not in seen:
                    seen.add(key)
                    orientations.append(normal)
        return orientations

class TetrominoType(Enum):
    """Enum to represent the multiple Tetromineos pieces (stores numpy bitmask)."""
    L = Tetromino.create_bitmask([(0,0), (1,0), (2,0), (2,1)])
    I = Tetromino.create_bitmask([(0,0), (1,0), (2,0), (3,0)])
    T = Tetromino.create_bitmask([(0,0), (1,0), (1,1), (2,0)])
    S = Tetromino.create_bitmask([(0,1), (0,2), (1,0), (1,1)])

class NuruominoState:
    """Represents the state of the Nuruomino puzzle, including the board configuration and a unique state ID."""
    state_id = 0

    def __init__(self, board, filled_regions=None, region_tetrominos=None, connectivity_grahp=None):
        self.board = board
        self.filled_regions = filled_regions if filled_regions is not None else set()
        self.region_tetrominos = region_tetrominos if region_tetrominos is not None else {}
        self.connectivity_grahp = connectivity_grahp
        self.id = NuruominoState.state_id 
        NuruominoState.state_id += 1

    def __lt__(self, other):
        """Used to break ties in informed search open lists."""
        return self.id < other.id

    def copy(self):
        """Creates a copy of the state of the board."""
        return NuruominoState(self.board, self.filled_regions.copy(), self.region_tetrominos.copy(), self.connectivity_grahp)

class Board:
    """Representação interna de um tabuleiro do Puzzle Nuruomino."""
    def __init__(self, board: list):
        self.board = np.array(board)
        self.size = self.board.shape[0]
        self.regions = self._get_regions(self.board)
        self.region_adjacencies = self._build_adjacency_grahp()

    def _get_regions(self, board):
        """Builds a dict of the regions present in the board."""
        regions = {}
        for row in range(board.shape[0]):
            for col in range(board.shape[1]):
                region_num = board[row, col]
                regions.setdefault(region_num, []).append((row, col))
        return regions

    def _build_adjacency_grahp(self):
        """Builds a grahp region to adjacent regions."""
        adjacencies = {region: set() for region in self.regions}
        board = self.board
        size = self.size
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for row in range(size):
            for col in range(size):
                region = board[row, col]
                for mov_x, mov_y in moves:
                    new_row, new_col = row + mov_x, col + mov_y
                    if 0 <= new_row < size and 0 <= new_col < size:
                        neighbor = board[new_row, new_col]
                        if neighbor != region:
                            adjacencies[region].add(neighbor)
        return adjacencies

    def get_value(self, row, col):
        """Returns the value at the specified (row, col) position on the board."""
        return self.board[row, col]

    def adjacent_regions(self, region:int) -> list:
        """Returns a list of regions that are adjacent with the given region."""
        return list(self.region_adjacencies.get(region, set()))

    def adjacent_positions(self, row: int, col: int) -> list:
        """Returns the positions adjacent to the cell (row, col) in all directions, including diagonals."""
        moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        positions = []
        for dx, dy in moves:
            new_row, new_col = row + dx, col + dy
            if 0 <= new_row < self.size and 0 <= new_col < self.size:
                positions.append((new_row, new_col))
        return positions

    def adjacent_values(self, row:int, col:int) -> list:
        """Returns the values of the cells adjacent to the cell (row, col) in all directions, including diagonals."""
        adjacent_values = []
        adjacent_pos = self.adjacent_positions(row, col)
        for adj_row, adj_col in adjacent_pos:
          adjacent_values.append(self.get_value(adj_row, adj_col))
        return adjacent_values

    def check_connectivity(self) -> bool:
        """Check if all regions are connected using BFS and a grahp."""
        if not self.regions:
            return True
        regions_list = list(self.regions.keys())
        if not regions_list:
            return True
        start_region = regions_list[0]
        visited = set()
        queue = deque([start_region])
        visited.add(start_region)

        while queue:
            current = queue.popleft()
            for neighbor in self.region_adjacencies.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        # All regions visisted puzzle fully connected
        return len(visited) == len(regions_list)

    def get_connected_tetrominos(self) -> list[set]:
        """Returns a list of the Tetrominos (regions) that are connected at the time on the puzzle."""
        if not self.regions:
            return []
        regions_list = list(self.regions.keys())
        visited = set()
        components = []

        for region in regions_list:
            if region not in visited:
                component = set()
                queue = deque([region])
                visited.add(region)
                component.add(region)

                while queue:
                    current = queue.popleft()
                    for neighbor in self.region_adjacencies.get(current, []):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            component.add(neighbor)
                            queue.append(neighbor)
                components.append(component)
        return components

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

    def print_instance(self):
        """Prints the string representation of the board."""
        board_str = ""
        for row in self.board:
            row_str = "\t".join(str(region) for region in row)
            board_str += row_str + "\n"
        stdout.write(board_str)

class Nuruomino(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        #TODO
        pass 

    def actions(self, state: NuruominoState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        #TODO
        pass 

    def result(self, state: NuruominoState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""

        #TODO
        pass 

    def goal_test(self, state: NuruominoState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        #TODO
        pass 

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass
 