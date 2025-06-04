import argparse
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from nuruomino_np import Board, Nuruomino
from search import depth_first_tree_search, OnlineSearchProblem, Node


class NuruominoOnlineProblem(OnlineSearchProblem):
    """Online version of Nuruomino problem for step-by-step visualization."""

    def __init__(self, nuruomino_problem):
        self.nuruomino = nuruomino_problem
        super().__init__(nuruomino_problem.initial, None, None)

    def actions(self, state):
        """Return available actions for the current state."""
        return self.nuruomino.actions(state)

    def result(self, state, action):
        """Return the result state after applying an action."""
        return self.nuruomino.result(state, action)

    def goal_test(self, state):
        """Test if the current state is a goal state."""
        return self.nuruomino.goal_test(state)

    def output(self, state, action):
        """For online search, output is the same as result."""
        return self.result(state, action)


class AdvancedInteractiveNuruominoAgent:
    """More sophisticated agent that closely mimics the exact DFS tree search."""

    def __init__(self, problem):
        self.problem = problem
        self.frontier = [Node(problem.initial)]  # Stack of nodes
        self.current_node = None
        self.solution_path = []
        self.finished = False
        self.solution_found = False
        self.explored_count = 0

    def step(self):
        """Take one step - exact replica of depth_first_tree_search logic."""
        if self.finished or not self.frontier:
            self.finished = True
            return None

        node = self.frontier.pop()
        self.current_node = node
        self.explored_count += 1
        if self.problem.goal_test(node.state):
            self.solution_found = True
            self.finished = True
            self.solution_path = node.path()
            return node.state
        try:
            child_nodes = node.expand(self.problem)
            for child in reversed(child_nodes):
                self.frontier.append(child)
        except Exception as e:
            pass
        return node.state if node else None

    def reset(self):
        """Reset the agent to initial state."""
        self.current_node = None
        self.frontier = [Node(self.problem.initial)]
        self.solution_path = []
        self.finished = False
        self.solution_found = False
        self.explored_count = 0

    def get_current_state(self):
        """Get the current state being explored."""
        return self.current_node.state if self.current_node else self.problem.initial

    def get_stats(self):
        """Get current search statistics."""
        return {
            'explored': self.explored_count,
            'frontier_size': len(self.frontier),
            'finished': self.finished,
            'solution_found': self.solution_found,
            'depth': self.current_node.depth if self.current_node else 0
        }


def parse_board(path):
    """Parse a board file and return a Board object using numpy."""
    board_array = np.loadtxt(path, dtype=np.int16).astype(object)
    return Board(board_array)


def parse_solution_board(path):
    """Parse a solution file that may contain both integers and tetromino letters."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    board_data = []
    for line in lines:
        row = []
        for cell in line.strip().split():
            try:
                row.append(int(cell))
            except ValueError:
                row.append(cell)
        board_data.append(row)

    board_array = np.array(board_data, dtype=object)
    return Board(board_array)


def get_font(size=16):
    """Get a font for text rendering."""
    try:
        return ImageFont.truetype("/System/Library/Fonts/SF-Pro-Display-Regular.otf", size)
    except:
        try:
            return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
        except:
            return ImageFont.load_default()


def board_to_image(board, adjacency_graph, original_board_data, cell_size=80):
    """Convert a board state to a PIL image."""
    rows, cols = board.shape
    img_width, img_height = cols * cell_size, rows * cell_size

    image = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(image)
    font = get_font(int(cell_size * 0.4))  # Scale font with cell size

    color_map = {
        'L': (255, 99, 71),   # Red
        'S': (60, 179, 113),  # Green
        'I': (65, 105, 225),  # Blue
        'T': (238, 130, 238), # Violet
    }

    # Build region borders
    region_cells = {}
    for row in range(rows):
        for col in range(cols):
            region = original_board_data[row, col]
            if isinstance(region, (int, np.integer)):
                if region not in region_cells:
                    region_cells[region] = []
                region_cells[region].append((row, col))

    # Draw region borders with thicker lines
    border_width = max(4, cell_size // 10)
    for region_id, cells in region_cells.items():
        region_set = set(cells)
        for row, col in cells:
            x0, y0 = col * cell_size, row * cell_size
            x1, y1 = x0 + cell_size, y0 + cell_size
            # Check each edge
            for dr, dc, coords in [(-1, 0, [x0, y0, x1, y0]), (1, 0, [x0, y1, x1, y1]),
                                   (0, -1, [x0, y0, x0, y1]), (0, 1, [x1, y0, x1, y1])]:
                if (row + dr, col + dc) not in region_set:
                    draw.line(coords, fill="black", width=border_width)

    # Draw grid with thinner lines
    grid_width = max(1, cell_size // 20)
    for row in range(rows + 1):
        y = row * cell_size
        draw.line([(0, y), (img_width, y)], fill="black", width=grid_width)
    for col in range(cols + 1):
        x = col * cell_size
        draw.line([(x, 0), (x, img_height)], fill="black", width=grid_width)

    # Draw pieces and numbers
    for row in range(rows):
        for col in range(cols):
            cell = board[row, col]
            original_region = original_board_data[row, col]
            x_center, y_center = col * cell_size + cell_size // 2, row * cell_size + cell_size // 2
            # Draw colored piece if it's a tetromino
            if isinstance(cell, str) and cell in color_map:
                color = color_map[cell]
                padding = cell_size // 10
                draw.rectangle([col * cell_size + padding, row * cell_size + padding,
                               (col + 1) * cell_size - padding, (row + 1) * cell_size - padding], 
                               fill=color)
                text_color = "white"
            else:
                text_color = "black"
            # Draw region number
            if isinstance(original_region, (int, np.integer)):
                text = str(original_region)
                try:
                    bbox = draw.textbbox((0, 0), text, font=font)
                    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
                except:
                    text_width, text_height = len(text) * (cell_size // 5), cell_size // 3
                text_x = x_center - text_width // 2
                text_y = y_center - text_height // 2
                draw.text((text_x, text_y), text, fill=text_color, font=font)
    return image


def draw_macos_button(img, x, y, w, h, text, is_primary=False, is_pressed=False, is_disabled=False):
    """Draw a macOS-style button."""
    if is_disabled:
        bg_color, text_color, border_color = (200, 200, 200), (128, 128, 128), (170, 170, 170)
    elif is_primary:
        bg_color = (0, 100, 200) if is_pressed else (0, 122, 255)
        text_color, border_color = (255, 255, 255), (0, 100, 200)
    else:
        bg_color = (220, 220, 220) if is_pressed else (248, 248, 248)
        text_color, border_color = (51, 51, 51), (200, 200, 200)
    if is_pressed:
        y += 1
    # Draw shadow and rounded button
    cv2.rectangle(img, (x + 1, y + 2), (x + w + 1, y + h + 2), (180, 180, 180), -1)
    radius = 6
    cv2.rectangle(img, (x + radius, y), (x + w - radius, y + h), bg_color, -1)
    cv2.rectangle(img, (x, y + radius), (x + w, y + h - radius), bg_color, -1)
    # Rounded corners
    for corner_x, corner_y in [(x + radius, y + radius), (x + w - radius, y + radius),
                               (x + radius, y + h - radius), (x + w - radius, y + h - radius)]:
        cv2.circle(img, (corner_x, corner_y), radius, bg_color, -1)
        cv2.circle(img, (corner_x, corner_y), radius, border_color, 1)
    # Border lines
    cv2.line(img, (x + radius, y), (x + w - radius, y), border_color, 1)
    cv2.line(img, (x + radius, y + h), (x + w - radius, y + h), border_color, 1)
    cv2.line(img, (x, y + radius), (x, y + h - radius), border_color, 1)
    cv2.line(img, (x + w, y + radius), (x + w, y + h - radius), border_color, 1)
    # Text
    font_scale, thickness = 0.5, 1
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = x + (w - text_size[0]) // 2
    text_y = y + (h + text_size[1]) // 2
    cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, text_color, thickness, cv2.LINE_AA)


class RealtimeVisualization:
    """Interactive visualization with step-by-step solving using OnlineSearchProblem."""
    def __init__(self, problem, original_board_data, realtime_delay, cell_size=80):
        # Convert to online problem
        self.online_problem = NuruominoOnlineProblem(problem)
        self.agent = AdvancedInteractiveNuruominoAgent(self.online_problem)
        self.frames = []
        self.current_frame = 0
        self.paused = True
        self.original_board_data = original_board_data
        self.realtime_delay = realtime_delay
        self.cell_size = cell_size
        self.window_name = "Nuruomino Solver - Interactive Controls"
        self.pressed_button = None
        initial_img = board_to_image(problem.initial.board.board, 
                                   problem.initial.adjacency_graph, 
                                   original_board_data, cell_size)
        self.frames.append(initial_img)
        button_scale = max(1.0, cell_size / 40.0)
        base_button_height = int(35 * button_scale)
        self.buttons = {
            'play_pause': (15, 15, int(90 * button_scale), base_button_height, 'Play', True),
            'step': (int(115 * button_scale), 15, int(70 * button_scale), base_button_height, 'Step', False),
            'prev': (int(195 * button_scale), 15, int(70 * button_scale), base_button_height, '← Prev', False),
            'restart': (int(275 * button_scale), 15, int(80 * button_scale), base_button_height, 'Restart', False),
            'exit': (int(365 * button_scale), 15, int(60 * button_scale), base_button_height, 'Exit', False)
        }

    def get_next_frame(self):
        """Get next frame by advancing agent one step."""
        if self.agent.finished:
            return False
        state = self.agent.step()
        if state:
            img = board_to_image(state.board.board, state.adjacency_graph, 
                               self.original_board_data, self.cell_size)
            self.frames.append(img)
            return True
        return False

    def draw_controls(self, img_cv):
        """Draw control panel with buttons."""
        control_height = 90  # Increased height for more stats
        img_height, img_width = img_cv.shape[:2]

        extended_img = np.zeros((img_height + control_height, img_width, 3), dtype=np.uint8)
        extended_img[:img_height, :] = img_cv
        for i in range(control_height):
            gray_value = int(245 - (i / control_height) * 15)
            extended_img[img_height + i, :] = (gray_value, gray_value, gray_value)

        cv2.line(extended_img, (0, img_height), (img_width, img_height), (200, 200, 200), 1)
        # Draw buttons
        for button_name, (x, y, w, h, text, is_primary) in self.buttons.items():
            button_y = img_height + y
            is_pressed = (self.pressed_button == button_name)
            is_disabled = False
            if button_name == 'play_pause':
                text = 'Pause' if not self.paused else 'Play'
                is_primary = not self.paused
            elif button_name == 'step':
                is_disabled = self.agent.finished
            elif button_name == 'prev':
                is_disabled = (self.current_frame <= 0)
            draw_macos_button(extended_img, x, button_y, w, h, text, 
                            is_primary, is_pressed, is_disabled)
        # Status bar with more detailed information
        stats = self.agent.get_stats()
        status_y = img_height + 55
        status_text = f"Frame: {self.current_frame + 1}/{len(self.frames)} | "
        status_text += "Playing" if not self.paused else "Paused"
        if self.agent.finished:
            status_text += " | Solution Found! ✓" if self.agent.solution_found else " | No Solution Found"
        else:
            current_state = self.agent.get_current_state()
            unfilled = len(self.online_problem.nuruomino.regions) - len(current_state.filled_regions)
            status_text += f" | Unfilled: {unfilled}"
        cv2.putText(extended_img, status_text, (15, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (51, 51, 51), 1, cv2.LINE_AA)
        # Additional stats line
        stats_y = img_height + 75
        stats_text = f"Explored: {stats['explored']} | Frontier: {stats['frontier_size']}"
        if 'depth' in stats:
            stats_text += f" | Depth: {stats['depth']}"

        cv2.putText(extended_img, stats_text, (15, stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (102, 102, 102), 1, cv2.LINE_AA)
        return extended_img

    def handle_mouse_click(self, event, x, y, flags, param):
        """Handle mouse clicks on buttons."""
        img_height = param
        control_y = y - img_height
        if event == cv2.EVENT_LBUTTONDOWN:
            for button_name, (bx, by, bw, bh, _, _) in self.buttons.items():
                if bx <= x <= bx + bw and by <= control_y <= by + bh:
                    if not ((button_name == 'step' and self.agent.finished) or 
                           (button_name == 'prev' and self.current_frame <= 0)):
                        self.pressed_button = button_name
                    break
        elif event == cv2.EVENT_LBUTTONUP:
            if self.pressed_button:
                for button_name, (bx, by, bw, bh, _, _) in self.buttons.items():
                    if (bx <= x <= bx + bw and by <= control_y <= by + bh and 
                        self.pressed_button == button_name):
                        if self.handle_button_click(button_name) == "exit":
                            return "exit"
                        break
                self.pressed_button = None
        elif event == cv2.EVENT_MOUSEMOVE and self.pressed_button:
            button_data = self.buttons[self.pressed_button]
            bx, by, bw, bh = button_data[:4]
            if not (bx <= x <= bx + bw and by <= control_y <= by + bh):
                self.pressed_button = None

    def handle_button_click(self, button_name):
        """Handle button click actions."""
        if button_name == 'play_pause':
            self.paused = not self.paused
        elif button_name == 'step' and not self.agent.finished:
            if self.get_next_frame():
                self.current_frame = len(self.frames) - 1
        elif button_name == 'prev' and self.current_frame > 0:
            self.current_frame -= 1
        elif button_name == 'restart':
            self.agent.reset()
            # Reset to initial frame
            initial_state = self.online_problem.initial
            initial_img = board_to_image(initial_state.board.board,
                                       initial_state.adjacency_graph,
                                       self.original_board_data, self.cell_size)
            self.frames = [initial_img]
            self.current_frame = 0
            self.paused = True
        elif button_name == 'exit':
            return "exit"
        return None

    def run(self):
        """Run the interactive visualization."""
        print("\n=== Interactive Nuruomino Solver (Online Search) ===")
        print("CONTROLS: Click buttons or use SPACE (play/pause), arrows (nav), R (restart), ESC (exit)")
        print("=================================================\n")

        last_advance_time = 0

        while True:
            if self.frames and self.current_frame < len(self.frames):
                img = self.frames[self.current_frame]
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                img_with_controls = self.draw_controls(img_cv)

                cv2.imshow(self.window_name, img_with_controls)
                cv2.setMouseCallback(self.window_name, self.handle_mouse_click, img_cv.shape[0])

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                self.paused = not self.paused
            elif key == 83 or key == ord('n'):  # Right arrow or 'n'
                if self.paused and not self.agent.finished and self.get_next_frame():
                    self.current_frame = len(self.frames) - 1
            elif key == 81 or key == ord('p'):  # Left arrow or 'p'
                if self.current_frame > 0:
                    self.current_frame -= 1
            elif key == ord('r'):  # 'r'
                self.handle_button_click('restart')

            # Auto-advance if playing
            current_time = cv2.getTickCount()
            if (not self.paused and not self.agent.finished and 
                (current_time - last_advance_time) / cv2.getTickFrequency() * 1000 >= self.realtime_delay):
                if self.get_next_frame():
                    self.current_frame = len(self.frames) - 1
                    last_advance_time = current_time
                else:
                    self.paused = True
        cv2.destroyAllWindows()


def solve_and_save_gif(problem, original_board_data, test_name, images_dir, cell_size=80):
    """Solve puzzle and save as GIF using the same agent as realtime visualization."""
    print("Solving puzzle for GIF...")

    # Use the same online problem and agent as realtime
    online_problem = NuruominoOnlineProblem(problem)
    agent = AdvancedInteractiveNuruominoAgent(online_problem)
    frames = []
    frame_count = 0
    max_frames = 1000
    initial_img = board_to_image(problem.initial.board.board, 
                               problem.initial.adjacency_graph, 
                               original_board_data, cell_size)
    frames.append(initial_img)
    frame_count += 1
    while not agent.finished and frame_count < max_frames:
        state = agent.step()
        if state:
            img = board_to_image(state.board.board, state.adjacency_graph, 
                               original_board_data, cell_size)
            frames.append(img)
            frame_count += 1
            if frame_count % 25 == 0:
                print(f"Processed {frame_count} states...")
        else:
            break

    if frames:
        gif_path = os.path.join(images_dir, f"{test_name}_solution.gif")
        if len(frames) > 200:
            step = len(frames) // 150
            frames = frames[::step]
            print(f"Reduced frames from {frame_count} to {len(frames)} for GIF size")
        if agent.solution_found:
            for _ in range(5):  # Pause at end
                frames.append(frames[-1].copy())
            print(f"Solution found! Generated {len(frames)} frames.")
        else:
            print(f"No solution found. Generated {len(frames)} frames from search.")
        # Save GIF
        try:
            frames[0].save(gif_path, save_all=True, append_images=frames[1:], 
                          duration=150, loop=0)
            print(f"GIF saved to {gif_path}")
            stats = agent.get_stats()
            print(f"Final stats - Explored: {stats['explored']}, Depth: {stats['depth']}")
        except Exception as e:
            print(f"Error saving GIF: {e}")
    else:
        print("No frames generated - check if the search is working correctly.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Nuruomino visualizer")
    parser.add_argument("input_file", help="Input board file")
    parser.add_argument("output_file", nargs='?', help="Output board file")
    parser.add_argument("--gif", action="store_true", help="Generate GIF")
    parser.add_argument("--realtime", action="store_true", help="Interactive mode")
    parser.add_argument("--cell-size", type=int, default=80, help="Cell size in pixels (default: 80)")
    args = parser.parse_args()

    # Setup
    test_dir = os.path.dirname(args.input_file)
    if test_dir and not test_dir.endswith('/'):
        test_dir += '/'
    test_name = os.path.splitext(os.path.basename(args.input_file))[0]
    images_dir = os.path.join(test_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    nuruomino_board = parse_board(args.input_file)
    problem = Nuruomino(nuruomino_board)
    original_board_data = nuruomino_board.board.copy()

    # Calculate delay based on complexity
    complexity = nuruomino_board.board.size * len(problem.current_state.adjacency_graph)
    realtime_delay = max(100, min(1000, 200 * (100 / max(complexity, 10))))

    if args.realtime:
        # Interactive step-by-step mode using online search
        visualization = RealtimeVisualization(problem, original_board_data, int(realtime_delay), args.cell_size)
        visualization.run()
    elif args.gif:
        # Generate GIF
        solve_and_save_gif(problem, original_board_data, test_name, images_dir, args.cell_size)
    else:
        # Static mode
        image = board_to_image(nuruomino_board.board, problem.current_state.adjacency_graph, 
                             original_board_data, args.cell_size)
        if args.output_file:
            solution_board = parse_solution_board(args.output_file)
            image = board_to_image(solution_board.board, problem.current_state.adjacency_graph, 
                                 original_board_data, args.cell_size)
        else:
            goal_node = depth_first_tree_search(problem)
            if goal_node:
                image = board_to_image(goal_node.state.board.board, goal_node.state.adjacency_graph, 
                                     original_board_data, args.cell_size)
                print("Solution found!")
            else:
                print("No solution found.")
        image_path = os.path.join(images_dir, f"{test_name}_solved.png")
        image.save(image_path)
        image.show()


if __name__ == "__main__":
    main()
