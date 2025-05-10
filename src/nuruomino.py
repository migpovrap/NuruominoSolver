# nuruomino.py: Template para implementação do projeto de Inteligência Artificial 2024/2025.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 35:
# 110632 Inês Costa
# 109686 Miguel Raposo

from sys import stdin, stdout
from enum import Enum
from search import *

class NuruominoState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = Nuroumino.state_id
        Nuroumino.state_id += 1

    def __lt__(self, other):
        """ Este método é utilizado em caso de empate na gestão da lista
        de abertos nas procuras informadas. """
        return self.id < other.id

class Board:
    """Representação interna de um tabuleiro do Puzzle Nuruomino."""

    def __init__(self, board: list, regions:dict):
        self.board = board
        self.regions = regions

    @staticmethod
    def parse_instance():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.

        Por exemplo:
            $ python3 pipe.py < test-01.txt

            > from sys import stdin
            > line = stdin.readline().split()
        """
        board = []
        for line in stdin.read().split("\n"):
            board.append(list(map(int, line.split("\t"))))
        regions = {}
        for row, line in enumerate(board):
            for col, region_num in enumerate(line):
                regions.setdefault(region_num, []).append((row, col))
        return Board(board, regions)

    def get_value(self, row:int, col:int) -> int:
        """Devolve o valor da célula."""
        return self.board[row][col]

    def adjacent_positions(self, row:int, col:int) -> list:
        """Devolve as posições adjacentes à região, em todas as direções, incluindo diagonais."""
        adjacent_coordinates = [(-1,0), (-1,-1), (0,-1), (1,-1), (1,0), (1,1), (0,1), (-1,1)]
        adjacent_positions = []
        board_size = len(self.board)
        for drow, dcolumn in adjacent_coordinates:
            new_row, new_column = row + drow, col + dcolumn
            if 0 <= new_row < board_size and 0 <= new_column < board_size:
                adjacent_positions.append((new_row, new_column))
        return adjacent_positions

    def adjacent_values(self, row:int, col:int) -> list:
        """Devolve os valores das celulas adjacentes à região,
        em todas as direções, incluindo diagonais."""
        adjacent_values = set()
        for new_row, new_column in self.adjacent_positions(row, col):
            adjacent_values.add(self.board[new_row][new_column])
        return adjacent_values

    def adjacent_regions(self, region:int) -> list:
        """Devolve uma lista das regiões que fazem fronteira com a região enviada no argumento."""
        adjacent_regions = set()
        for row, col in self.regions[region]:
            adjacent_regions.update(self.adjacent_values(row, col))
        return list(adjacent_regions)

    def print_instance(self):
        """Imprime a representação do tabuleiro no formato de string."""
        game_board = ""
        for row in self.board:
            for region in row:
                game_board += f"{region}\t"
            game_board += "\n"
        stdout.write(game_board)

    def __repr__(self):
        board_repr = "\n".join(" ".join(map(str, row)) for row in self.board)
        regions_repr = "\n".join(f"Region {region}: {positions}"
                                 for region, positions in self.regions.items())
        return f"Board:\n{board_repr}\n{regions_repr}"

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

class TetrominoType(Enum):
    """Enum para representar os vários tipos de Tetromino."""
    L = [(0,0), (1,0), (2,0), (2,1)]
    I = [(0,0), (1,0), (2,0), (3,0)]
    T = [(0,1), (1,0), (1,1), (1,2)]
    S = [(0,1), (0,2), (1,0), (1,1)]

class Tetromino:
    """Representação interna de um tetramino e da posição na qual se encontra."""

    def __init__(self, tetronimo_type: TetrominoType, rotation: int = 0, refleced: bool = False):
        self.tetronimo_type = tetronimo_type
        self.rotation = rotation
        self.reflected = refleced

    def __repr__(self):
        return (f'Tetromino(type={self.tetronimo_type}, rotation={self.rotation} degrees, '
            f'reflected={self.reflected})')

    @staticmethod
    def rotate(shape, degrees):
        """Aplica uma rotação no valor (degrees)
        assume que este é um multiplo de 90."""
        for _ in range((degrees // 90) % 4): # Calculates the number of rotation to apply
            shape = [(column, -row) for row, column in shape]
        return shape

    @staticmethod
    def reflect(shape):
        """Aplica uma refleção horizontal."""
        return [(row, -column) for row, column in shape]

    @staticmethod
    def normalize(shape):
        """Alinha os tetraminos em coordenadas padrão a partir de (0,0)."""
        return [(row - min(r for r, _ in shape),
                 col - min(c for _, c in shape)) for row, col in shape]

    def get(self):
        """Aplica a rotação e refleção devolvendo um conjunto de coordenadas
        finais para o tetramino."""
        tetromino = self.tetronimo_type.value
        tetromino = Tetromino.rotate(tetromino, self.rotation)
        if self.reflected:
            tetromino = Tetromino.reflect(tetromino)
        return Tetromino.normalize(tetromino)

