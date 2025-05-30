import argparse
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw

from nuruomino import Board
import nuruomino


def board_to_image(board, regions):
    """Convert a board and its regions to a PIL image."""
    rows = len(board)
    cols = len(board[0])
    image = draw_image_not_solved(rows, cols, regions)
    image = draw_pieces_on_board(image, board)
    return image

def parse_board(path, parse_as_int=False):
    """Parse a board file and return a Board object."""
    board = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            if parse_as_int:
                board.append([int(x) for x in line.strip().split("\t") if x])
            else:
                board.append([x for x in line.strip().split("\t") if x])
    return Board(board)

def draw_image_not_solved(rows:int, cols:int, regions):
    """Draw the unsolved board with region borders as a PIL image."""
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
    """Draw colored pieces on the board image, one color per letter (L, T, I, S)."""
    source = ImageDraw.Draw(image)
    cell_size = 40
    padding = 4

    color_map = {
        'L': (255, 99, 71),     # Tomato Red
        'S': (60, 179, 113),   # Medium Sea Green
        'I': (65, 105, 225),   # Royal Blue
        'T': (238, 130, 238),  # Violet
    }

    for row_idx, row in enumerate(piece_board):
        for col_idx, cell in enumerate(row):
            if cell in color_map:
                color = color_map[cell]
                x0 = col_idx * cell_size + padding
                y0 = row_idx * cell_size + padding
                x1 = (col_idx + 1) * cell_size - padding
                y1 = (row_idx + 1) * cell_size - padding
                source.rectangle([x0, y0, x1, y1], fill=color)

    return image

def draw_edge(row:int, col:int, coords, region_set, draw):
    """Draw a thick border if the neighboring cell is not in the same region."""
    if (row, col) not in region_set:
        x0, y0, x1, y1 = coords
        draw.line([(x0, y0), (x1, y1)], fill="black", width=4)

def main():
    """Main entry point for the Nuruomino visualizer."""
    parser = argparse.ArgumentParser(description="Nuruomino visualizer")
    parser.add_argument("input_file", help="Input board file (e.g., test01.txt)")
    parser.add_argument("output_file", help="Output board file (e.g., test01.out)")
    parser.add_argument("--gif", action="store_true", help="Generate GIF of solving process")
    parser.add_argument("--realtime", action="store_true", help="Show real-time solving visualization (requires --gif)")
    args = parser.parse_args()

    test_in = args.input_file
    test_out = args.output_file

    # Extract directory and base name without extension
    test_dir = os.path.dirname(test_in)
    if test_dir and not test_dir.endswith('/'):
        test_dir += '/'
    test_name = os.path.splitext(os.path.basename(test_in))[0]

    images_dir = os.path.join(test_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    if args.realtime or args.gif:
        frames = []

        def on_state(board_obj):
            img = board_to_image(board_obj.board, board_obj.regions)
            if args.gif:
                frames.append(img.copy())
            if args.realtime:
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                cv2.imshow("Nuruomino Solver", img_cv)
                cv2.waitKey(400)  # Adjust delay as needed

        nuruomino_board = parse_board(test_in, parse_as_int=True)

        # Add initial state
        initial_img = board_to_image(nuruomino_board.board, nuruomino_board.regions)
        if args.gif:
            frames.append(initial_img.copy())
        if args.realtime:
            img_cv = cv2.cvtColor(np.array(initial_img), cv2.COLOR_RGB2BGR)
            cv2.imshow("Nuruomino Solver", img_cv)
            cv2.waitKey(400)

        problem = nuruomino.Nuruomino(nuruomino_board, on_state=on_state)
        solution = nuruomino.solve_nuruomino(problem)

        if solution:
            final_img = board_to_image(solution.state.board.board, solution.state.board.regions)
            if args.gif:
                frames.append(final_img.copy())
            if args.realtime:
                img_cv = cv2.cvtColor(np.array(final_img), cv2.COLOR_RGB2BGR)
                cv2.imshow("Nuruomino Solver", img_cv)
                cv2.waitKey(0)  # Wait for key press to close

        if args.realtime:
            cv2.destroyAllWindows()

        if args.gif and len(frames) > 1:
            gif_path = os.path.join(images_dir, f"{test_name}_solution.gif")
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=400,
                loop=0
            )
            print(f"GIF saved to {gif_path} with {len(frames)} frames")
        elif args.gif:
            print("No solution found or no intermediate steps captured")
    else:
        nuruomino_board = parse_board(test_in, parse_as_int=True)
        nuruomino_board_result = parse_board(test_out, parse_as_int=False)
        rows = len(nuruomino_board.board)
        cols = len(nuruomino_board.board[0])
        image = draw_image_not_solved(rows, cols, nuruomino_board.regions)
        image_path = os.path.join(images_dir, f"{test_name}.png")
        image.save(image_path)
        image = draw_pieces_on_board(image, nuruomino_board_result.board)
        solved_image_path = os.path.join(images_dir, f"{test_name}_solved.png")
        image.save(solved_image_path)
        image.show()

if __name__ == "__main__":
    main()
