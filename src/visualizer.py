import sys
import random
from PIL import Image, ImageDraw
from nuruomino import Board

def parse_board(path):
    board = []
    with open(path, 'r') as file:
        for line in file:
            board.append(list(map(str, line.strip().split("\t"))))
    regions = {}
    for row, line in enumerate(board):
        for col, region_num in enumerate(line):
            regions.setdefault(region_num, []).append((row, col))
    return Board(board, regions)

def draw_image_not_solved(rows:int, cols:int, regions):
    img_width = rows * 40
    img_height = cols * 40

    image = Image.new("RGB", (img_width, img_height), "white")
    source = ImageDraw.Draw(image)

    for _, cells in regions.items():
        region_set = set(cells)
        for row, col in cells:
            x0, y0 = col * 40, row * 40
            x1, y1 = x0 + 40, y0 + 40
            draw_edge(row - 1, col, [x0, y0, x1, y0], region_set, source)
            draw_edge(row + 1, col, [x0, y1, x1, y1], region_set, source)
            draw_edge(row, col - 1, [x0, y0, x0, y1], region_set, source)
            draw_edge(row, col + 1, [x1, y0, x1, y1], region_set, source)

    for row in range(rows + 1):
        y = row * 40
        source.line([(0, y), (img_width, y)], fill="black")

    for col in range(cols + 1):
        x = col * 40
        source.line([(x, 0), (x, img_height)], fill="black")

    return image

def draw_pieces_on_board(image: Image, piece_board: list[list[str]]):
  source = ImageDraw.Draw(image)
  cell_size = 40
  padding = 4

  unique_letters = {ch for row in piece_board for ch in row if ch in ['L', 'S', 'I', 'T']}
  color_map = {
    letter: (
      random.randint(100, 255),
      random.randint(100, 255),
      random.randint(100, 255)
    )
    for letter in unique_letters
  }

  for row_idx, row in enumerate(piece_board):
    for col_idx, cell in enumerate(row):
      if cell not in ['L', 'S', 'I', 'T']:
        continue
      color = color_map[cell]
      x0 = col_idx * cell_size + padding
      y0 = row_idx * cell_size + padding
      x1 = (col_idx + 1) * cell_size - padding
      y1 = (row_idx + 1) * cell_size - padding
      source.rectangle([x0, y0, x1, y1], fill=color)

  return image

def draw_edge(row:int, col:int, coords, region_set, draw):
    if (row, col) not in region_set:
        x0, y0, x1, y1 = coords
        draw.line([(x0, y0), (x1, y1)], fill="black", width=4)

def main():
    if len(sys.argv) < 3:
        print("Usage: src/python visualizer.py <input_file> <output_file>")
        sys.exit(1)

    test_in = sys.argv[1]
    test_out = sys.argv[2]
    test_dir = test_in[:test_in.rfind('/')+1]
    test_name = test_in[test_in.rfind('/')+1:test_in.rfind('.')]
    nuruomino_board = parse_board(test_in)
    nuruomino_board_result = parse_board(test_out)
    rows = len(nuruomino_board.board)
    image = draw_image_not_solved(rows, rows, nuruomino_board.regions)
    image.save(f'{test_dir}images/{test_name}.png')
    image = draw_pieces_on_board(image, nuruomino_board_result.board)
    image.save(f'{test_dir}images/{test_name}_solved.png')
    image.show()


if __name__ == "__main__":
    main()
    # Uses nuruomino to solve the problem,
    # takes the output and generates a new
    # image with the pieces placed.
