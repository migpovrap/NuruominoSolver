from __future__ import annotations
# nuruomino.py: Template para implementação do projeto de Inteligência Artificial 2024/2025.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 35:
# 110632 Inês Costa
# 109686 Miguel Raposo

from sys import stdin, stdout
import numpy as np
from enum import Enum
from search import Problem

class Tetromino:

    @staticmethod
    def create_bitmask(coords):
        """Convert a list (row, cols) to a 4x4 numpy array."""
        np_array = np.zeros((4, 4), dtype=int)
        for row, col in coords:
            np_array[row, col] = 1
        return tuple(tuple(row) for row in np_array)

    @staticmethod
    def to_numpy(mask):
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
        r, c = normal.shape
        complete[:r, :c] = normal
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
    L = Tetromino.create_bitmask([(0,0), (1,0), (2,0), (2,1)])
    I = Tetromino.create_bitmask([(0,0), (1,0), (2,0), (3,0)])
    T = Tetromino.create_bitmask([(0,0), (1,0), (1,1), (2,0)])
    S = Tetromino.create_bitmask([(0,1), (1,1), (0,1), (0,2)])

class NuruominoState:
    """Represents the state of the Nuruomino puzzle, including the board configuration and a unique state ID."""
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = NuruominoState.state_id 
        NuruominoState.state_id += 1

    def __lt__(self, other):
        """Used to break ties in informed search open lists."""
        return self.id < other.id

class Board:
    """Representação interna de um tabuleiro do Puzzle Nuruomino."""
    def __init__(self, board: list):
        self.board = board
        self.regions = self._get_regions(board)

    def _get_regions(self, board):
        """Builds a dict of the regions present in the board."""
        regions = {}
        for row, line in enumerate(board):
            for col, region_num in enumerate(line):
                regions.setdefault(region_num, []).append((row, col))
        return regions
    
    def adjacent_regions(self, region:int) -> list:
        """Devolve uma lista das regiões que fazem fronteira com a região enviada no argumento."""
        #TODO
        pass
    
    def adjacent_positions(self, row:int, col:int) -> list:
        """Devolve as posições adjacentes à região, em todas as direções, incluindo diagonais."""
        #TODO
        pass

    def adjacent_values(self, row:int, col:int) -> list:
        """Devolve os valores das celulas adjacentes à região, em todas as direções, incluindo diagonais."""
        #TODO
        pass

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
          row_str = ""
          for region in row:
              row_str += f"{region}\t"
          board_str += row_str.rstrip() + "\n"
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
    

if __name__ == "__main__":
    for orientation in Tetromino.get_all_forms(Tetromino.to_numpy(TetrominoType.L.value)):
        print(orientation)