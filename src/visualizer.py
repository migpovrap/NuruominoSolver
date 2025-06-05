import argparse
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import threading
import time
import tempfile

from nuruomino import Board, Nuruomino
from search import depth_first_tree_search, Node

# Try to import flet, but make it optional
try:
    import flet as ft
    FLET_AVAILABLE = True
except ImportError:
    FLET_AVAILABLE = False


class FastBatchNuruominoAgent:
    """Agent that runs at full speed and only captures selected frames."""

    def __init__(self, problem, key_frames_only=True, max_frames=None):
        self.problem = problem
        self.all_states = []  # Store all states after fast solve
        self.current_index = 0
        self.finished = False
        self.solution_found = False
        self.explored_count = 0
        self.key_frames_only = key_frames_only
        self.max_frames = max_frames
        self.original_solution_length = 0  # Track original solution length
        self._solve_fast()

    def _solve_fast(self):
        """Run the full search at maximum speed, capturing frames based on mode."""
        if self.key_frames_only:
            print(f"Running fast solve (key frames only, max {self.max_frames or 'all'} frames)...")
        else:
            print("Running solve with step-by-step capture...")
        
        start_time = time.time()
        
        if self.key_frames_only:
            # Use your exact same search algorithm at full speed
            goal_node = depth_first_tree_search(self.problem)
            solve_time = time.time() - start_time
            print(f"Fast solve completed in {solve_time:.3f} seconds!")
            
            if goal_node:
                self.solution_found = True
                # Get the solution path (key frames only)
                solution_path = goal_node.path()
                all_solution_states = [node.state for node in solution_path]
                self.original_solution_length = len(all_solution_states)
                
                # Apply frame limiting if specified
                if self.max_frames and len(all_solution_states) > self.max_frames:
                    # Smart sampling to get representative frames
                    self.all_states = self._sample_frames(all_solution_states, self.max_frames)
                    print(f"Solution found with {self.original_solution_length} total steps, showing {len(self.all_states)} key frames!")
                else:
                    self.all_states = all_solution_states
                    print(f"Solution found with {len(self.all_states)} key steps!")
            else:
                print("No solution found")
                self.all_states = [self.problem.initial]
        else:
            # Step-by-step capture (slower but detailed)
            self._solve_step_by_step()
            solve_time = time.time() - start_time
            print(f"Step-by-step solve completed in {solve_time:.3f} seconds!")
            
        self.finished = True

    def _sample_frames(self, all_states, max_frames):
        """Smart sampling to get representative frames from the solution path."""
        if len(all_states) <= max_frames:
            return all_states
        
        # Always include first and last frames
        if max_frames < 2:
            return [all_states[0]]
        
        sampled = [all_states[0]]  # Always include initial state
        remaining_frames = max_frames - 2  # Reserve space for first and last
        
        if remaining_frames > 0:
            # Sample evenly from the middle states
            middle_states = all_states[1:-1]
            if middle_states:
                step = len(middle_states) / remaining_frames
                for i in range(remaining_frames):
                    index = int(i * step)
                    if index < len(middle_states):
                        sampled.append(middle_states[index])
        
        # Always include final state
        if len(all_states) > 1:
            sampled.append(all_states[-1])
        
        return sampled

    def _solve_step_by_step(self):
        """Solve step-by-step, capturing every explored state."""
        frontier = [Node(self.problem.initial)]
        self.all_states = [self.problem.initial]  # Include initial state
        
        while frontier:
            node = frontier.pop()
            self.explored_count += 1
            
            # Goal test
            if self.problem.goal_test(node.state):
                self.solution_found = True
                return
                
            # Expand node
            try:
                actions = self.problem.actions(node.state)
                for action in reversed(actions):
                    child_state = self.problem.result(node.state, action)
                    child_node = Node(child_state, node, action)
                    frontier.append(child_node)
                    self.all_states.append(child_state)  # Capture every state
                    
            except Exception:
                pass

    def step(self):
        """Get the next state in the pre-computed sequence."""
        if self.current_index < len(self.all_states):
            state = self.all_states[self.current_index]
            self.current_index += 1
            return state
        return None

    def reset(self):
        """Reset to beginning of sequence."""
        self.current_index = 0

    def get_current_state(self):
        """Get current state."""
        if self.current_index > 0 and self.current_index <= len(self.all_states):
            return self.all_states[self.current_index - 1]
        return self.problem.initial

    def get_stats(self):
        """Get search statistics."""
        if self.key_frames_only and self.max_frames and self.original_solution_length > 0:
            mode_desc = f'Key Frames ({len(self.all_states)}/{self.original_solution_length} steps)'
        elif self.key_frames_only:
            mode_desc = 'Key Frames Only'
        else:
            mode_desc = 'Every Step'
            
        return {
            'explored': self.current_index,
            'frontier_size': max(0, len(self.all_states) - self.current_index),
            'finished': self.current_index >= len(self.all_states),
            'solution_found': self.solution_found,
            'depth': self.current_index,
            'total_steps': len(self.all_states),
            'original_solution_length': self.original_solution_length,
            'mode': mode_desc
        }


class OptimizedInteractiveNuruominoAgent:
    """Agent that uses real-time step-by-step generation (slower but interactive)."""

    def __init__(self, problem):
        self.problem = problem
        self.frontier = [Node(problem.initial)]
        self.current_node = None
        self.solution_path = []
        self.finished = False
        self.solution_found = False
        self.explored_count = 0

    def step(self):
        """Single step that mirrors depth_first_tree_search exactly."""
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
            actions = self.problem.actions(node.state)
            for action in reversed(actions):
                child_state = self.problem.result(node.state, action)
                child_node = Node(child_state, node, action)
                self.frontier.append(child_node)
        except Exception:
            pass
            
        return node.state

    def reset(self):
        """Reset to initial state."""
        self.current_node = None
        self.frontier = [Node(self.problem.initial)]
        self.solution_path = []
        self.finished = False
        self.solution_found = False
        self.explored_count = 0

    def get_current_state(self):
        """Get current state."""
        return self.current_node.state if self.current_node else self.problem.initial

    def get_stats(self):
        """Get search statistics."""
        return {
            'explored': self.explored_count,
            'frontier_size': len(self.frontier),
            'finished': self.finished,
            'solution_found': self.solution_found,
            'depth': self.current_node.depth if self.current_node else 0,
            'mode': 'Real-time Generation'
        }


def parse_board(path):
    """Parse a board file and return a Board object using numpy."""
    board_array = np.loadtxt(path, dtype=np.int16).astype(object)
    return Board(board_array)


def parse_solution_board(path):
    """Parse a solution file that may contain both integers and tetromino letters."""
    with open(path, 'r') as f:
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
    font = get_font(int(cell_size * 0.4))

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

    # Draw region borders
    border_width = max(4, cell_size // 10)
    for region_id, cells in region_cells.items():
        region_set = set(cells)
        for row, col in cells:
            x0, y0 = col * cell_size, row * cell_size
            x1, y1 = x0 + cell_size, y0 + cell_size
            for dr, dc, coords in [(-1, 0, [x0, y0, x1, y0]), (1, 0, [x0, y1, x1, y1]),
                                   (0, -1, [x0, y0, x0, y1]), (0, 1, [x1, y0, x1, y1])]:
                if (row + dr, col + dc) not in region_set:
                    draw.line(coords, fill="black", width=border_width)

    # Draw grid
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

            if isinstance(cell, str) and cell in color_map:
                color = color_map[cell]
                padding = cell_size // 10
                draw.rectangle([col * cell_size + padding, row * cell_size + padding,
                               (col + 1) * cell_size - padding, (row + 1) * cell_size - padding], 
                               fill=color)
                text_color = "white"
            else:
                text_color = "black"

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


def solve_and_save_gif(problem, original_board_data, test_name, images_dir, cell_size=80, key_frames_only=True, max_frames=None):
    """Generate GIF using the selected agent mode."""
    if key_frames_only:
        mode_str = f"key_frames_{max_frames}" if max_frames else "key_frames_all"
        print(f"Solving puzzle for GIF (key frames, max {max_frames or 'all'})...")
    else:
        mode_str = "every_step"
        print(f"Solving puzzle for GIF (every step)...")
    
    agent = FastBatchNuruominoAgent(problem, key_frames_only, max_frames)
    
    if not agent.solution_found:
        print("No solution found, cannot generate GIF.")
        return
    
    print(f"Generating frames for {len(agent.all_states)} states...")
    frames = []
    
    for i, state in enumerate(agent.all_states):
        img = board_to_image(state.board.board, state.adjacency_graph, 
                           original_board_data, cell_size)
        frames.append(img)
        
        if (i + 1) % 5 == 0:
            print(f"Generated frame {i + 1}/{len(agent.all_states)}")

    if frames:
        gif_path = os.path.join(images_dir, f"{test_name}_solution_{mode_str}.gif")
        
        # Add a few frames at the end to show final solution
        for _ in range(5):
            frames.append(frames[-1].copy())
        
        try:
            duration = 400 if key_frames_only and max_frames and max_frames <= 10 else (200 if key_frames_only else 100)
            frames[0].save(gif_path, save_all=True, append_images=frames[1:], 
                          duration=duration, loop=0)
            print(f"GIF saved to {gif_path}")
            print(f"Generated {len(frames)} frames")
            
        except Exception as e:
            print(f"Error saving GIF: {e}")


# Flet-based realtime visualizer
if FLET_AVAILABLE:
    class NuruominoRealtimeVisualizer:
        """Flet-based interactive realtime visualizer with mode selection."""
        
        def __init__(self, problem, original_board_data, cell_size=50, key_frames_only=True, max_frames=None):
            self.problem = problem
            self.original_board_data = original_board_data
            self.cell_size = min(cell_size, 50)
            self.key_frames_only = key_frames_only
            self.max_frames = max_frames
            
            # Create a temporary directory for image files FIRST
            self.temp_dir = tempfile.mkdtemp()
            
            # State management
            self.frames = []
            self.current_frame = 0
            self.is_playing = False
            self.speed = 1.0
            
            # Generate initial frame
            initial_img = board_to_image(
                problem.initial.board.board,
                problem.initial.adjacency_graph,
                original_board_data,
                self.cell_size
            )
            initial_path = self.save_frame_to_file(initial_img, 0)
            self.frames.append(initial_path)
            
            # Create appropriate agent based on mode
            if key_frames_only:
                self.agent = FastBatchNuruominoAgent(problem, key_frames_only=True, max_frames=max_frames)
                self._pre_generate_frames()
            else:
                self.agent = OptimizedInteractiveNuruominoAgent(problem)
                self.generation_complete = False
            
            # UI components
            self.page = None
            self.board_image = None
            self.play_button = None
            self.frame_slider = None
            self.frame_counter_text = None
            self.speed_slider = None
            self.stats_text = None
            self.debug_text = None
            self.mode_text = None
            
        def _pre_generate_frames(self):
            """Pre-generate all frames for batch mode."""
            print(f"Generating {len(self.agent.all_states)} frames for visualization...")
            
            for i, state in enumerate(self.agent.all_states):
                img = board_to_image(
                    state.board.board,
                    state.adjacency_graph,
                    self.original_board_data,
                    self.cell_size
                )
                frame_path = self.save_frame_to_file(img, i + 1)  # +1 because frame 0 is initial
                self.frames.append(frame_path)
                
                if (i + 1) % 5 == 0:
                    print(f"Generated frame {i + 1}/{len(self.agent.all_states)}")
            
            print("All frames generated! Ready for visualization.")
            
        def save_frame_to_file(self, pil_image, frame_num):
            """Save PIL image to a temporary file and return the path."""
            try:
                # Ensure reasonable size
                max_size = 400
                if pil_image.size[0] > max_size or pil_image.size[1] > max_size:
                    ratio = min(max_size / pil_image.size[0], max_size / pil_image.size[1])
                    new_size = (int(pil_image.size[0] * ratio), int(pil_image.size[1] * ratio))
                    pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
                
                # Save as PNG file
                file_path = os.path.join(self.temp_dir, f"frame_{frame_num:04d}.png")
                pil_image.save(file_path, "PNG", optimize=True)
                return file_path
                
            except Exception as e:
                print(f"Error saving frame {frame_num}: {e}")
                # Create a simple fallback image
                fallback_img = Image.new('RGB', (400, 300), color='lightcoral')
                draw = ImageDraw.Draw(fallback_img)
                draw.text((150, 140), f"Frame {frame_num} Error", fill='white')
                fallback_path = os.path.join(self.temp_dir, f"fallback_{frame_num}.png")
                fallback_img.save(fallback_path, "PNG")
                return fallback_path
            
        def build_ui(self, page: ft.Page):
            """Build the Flet UI."""
            self.page = page
            mode_title = f"Key Frames ({self.max_frames})" if self.key_frames_only and self.max_frames else ("Key Frames" if self.key_frames_only else "Every Step")
            page.title = f"Nuruomino Solver - {mode_title} Mode"
            page.theme_mode = ft.ThemeMode.LIGHT
            page.padding = 20
            page.window.width = 1400
            page.window.height = 900
            
            # Mode indicator
            mode_color = "green" if self.key_frames_only else "blue"
            if self.key_frames_only:
                if self.max_frames:
                    mode_desc = f"🚀 INSTANT SOLVE + {len(self.agent.all_states)} Key Steps (from {self.agent.original_solution_length} total)"
                else:
                    mode_desc = "🚀 INSTANT SOLVE + All Solution Steps"
            else:
                mode_desc = "🔍 Real-time Step-by-Step Generation"
            
            self.mode_text = ft.Text(
                mode_desc,
                size=14,
                color=mode_color,
                weight=ft.FontWeight.BOLD
            )
            
            # Board image
            self.board_image = ft.Image(
                src=self.frames[0] if self.frames else "",
                width=400,
                height=400,
                fit=ft.ImageFit.CONTAIN,
                border_radius=8,
            )
            
            # Add a container around the image
            board_container = ft.Container(
                content=self.board_image,
                width=420,
                height=420,
                bgcolor="#f8f8f8",
                border=ft.border.all(2, "darkgray"),
                border_radius=10,
                padding=10,
            )
            
            # Control buttons
            play_text = "Play Solution" if self.key_frames_only else "Play"
            button_color = "green" if (self.key_frames_only and hasattr(self.agent, 'solution_found') and self.agent.solution_found) else "blue"
            
            self.play_button = ft.ElevatedButton(
                play_text,
                icon="play_arrow",
                on_click=self.toggle_play,
                style=ft.ButtonStyle(
                    color="white",
                    bgcolor=button_color,
                )
            )
            
            prev_button = ft.IconButton(
                icon="skip_previous",
                tooltip="Previous Frame",
                on_click=lambda _: self.go_to_frame(max(0, self.current_frame - 1))
            )
            
            next_button = ft.IconButton(
                icon="skip_next",
                tooltip="Next Frame",
                on_click=self.next_frame
            )
            
            if not self.key_frames_only:
                fast_forward_button = ft.IconButton(
                    icon="fast_forward",
                    tooltip="Generate 10 frames",
                    on_click=self.fast_forward
                )
            else:
                fast_forward_button = ft.IconButton(
                    icon="last_page",
                    tooltip="Jump to end",
                    on_click=lambda _: self.go_to_frame(len(self.frames) - 1)
                )
            
            restart_button = ft.IconButton(
                icon="restart_alt",
                tooltip="Restart",
                on_click=self.restart
            )
            
            # Frame controls
            self.frame_slider = ft.Slider(
                min=0,
                max=max(1, len(self.frames) - 1),
                value=0,
                label="{value}",
                on_change=self.on_frame_change
            )
            
            frame_label = "Solution Step" if self.key_frames_only else "Search Step"
            self.frame_counter_text = ft.Text(f"{frame_label}: 1/{len(self.frames)}")
            
            # Speed control
            self.speed_slider = ft.Slider(
                min=0.1,
                max=10.0,
                value=2.0 if self.key_frames_only else 1.0,
                label="Speed: {value}x",
                on_change=self.on_speed_change
            )
            
            # Statistics display
            self.stats_text = ft.Text(
                self.get_stats_text(),
                size=14,
                font_family="Consolas"
            )
            
            # Debug information
            self.debug_text = ft.Text(
                self.get_debug_text(),
                size=12,
                font_family="Consolas",
                selectable=True
            )
            
            # Layout
            controls_row = ft.Row([
                prev_button,
                self.play_button,
                next_button,
                fast_forward_button,
                ft.VerticalDivider(),
                restart_button,
            ])
            
            frame_control = ft.Column([
                ft.Text(f"{frame_label} Control"),
                self.frame_slider,
                self.frame_counter_text,
            ], spacing=5)
            
            speed_control = ft.Column([
                ft.Text("Playback Speed"),
                self.speed_slider,
            ], spacing=5)
            
            left_panel = ft.Column([
                ft.Text("Nuruomino Solver Visualization", size=16, weight=ft.FontWeight.BOLD),
                self.mode_text,
                board_container,
                controls_row,
                frame_control,
                speed_control,
            ], spacing=10, alignment=ft.MainAxisAlignment.START)
            
            right_panel = ft.Column([
                ft.Text("Search Information", size=16, weight=ft.FontWeight.BOLD),
                self.stats_text,
                ft.Divider(),
                ft.Text("Step Details", size=16, weight=ft.FontWeight.BOLD),
                ft.Container(
                    content=ft.Column([self.debug_text], scroll=ft.ScrollMode.AUTO),
                    bgcolor="#f5f5f5",
                    padding=10,
                    border_radius=5,
                    height=400,
                    width=500,
                ),
            ], spacing=10)
            
            main_row = ft.Row([
                left_panel,
                ft.VerticalDivider(),
                right_panel,
            ], spacing=20, expand=True)
            
            page.add(main_row)
            
            # Start auto-generation thread for real-time mode
            if not self.key_frames_only:
                threading.Thread(target=self.auto_generate_frames, daemon=True).start()
            
        def toggle_play(self, e):
            """Toggle play/pause."""
            if not self.frames:
                return
                
            self.is_playing = not self.is_playing
            play_text = "Pause" if self.is_playing else ("Play Solution" if self.key_frames_only else "Play")
            self.play_button.text = play_text
            self.play_button.icon = "pause" if self.is_playing else "play_arrow"
            self.page.update()
            
            if self.is_playing:
                threading.Thread(target=self.play_animation, daemon=True).start()
        
        def play_animation(self):
            """Play the animation."""
            while self.is_playing and self.current_frame < len(self.frames) - 1:
                time.sleep(0.5 / self.speed)
                if self.is_playing:
                    self.go_to_frame(self.current_frame + 1)
            
            if self.current_frame >= len(self.frames) - 1:
                self.is_playing = False
                play_text = "Play Solution" if self.key_frames_only else "Play"
                self.play_button.text = play_text
                self.play_button.icon = "play_arrow"
                self.page.update()
        
        def next_frame(self, e):
            """Go to next frame or generate if needed."""
            if self.current_frame < len(self.frames) - 1:
                self.go_to_frame(self.current_frame + 1)
            elif not self.key_frames_only and not self.agent.finished:
                self.generate_next_frame()
        
        def fast_forward(self, e):
            """Generate and advance multiple frames quickly (real-time mode only)."""
            if self.key_frames_only:
                return
                
            for _ in range(10):
                if not self.agent.finished:
                    self.generate_next_frame()
                else:
                    break
            if self.frames:
                self.go_to_frame(len(self.frames) - 1)
        
        def restart(self, e):
            """Restart the visualization."""
            if self.key_frames_only:
                # For batch mode, just reset the playback
                self.current_frame = 0
                self.is_playing = False
                self.update_ui()
            else:
                # For real-time mode, restart the search
                self.agent.reset()
                
                # Clear existing frames
                for frame_path in self.frames[1:]:  # Keep initial frame
                    try:
                        if os.path.exists(frame_path):
                            os.remove(frame_path)
                    except:
                        pass
                
                self.frames = self.frames[:1]  # Keep only initial frame
                self.current_frame = 0
                self.is_playing = False
                self.generation_complete = False
                
                self.update_ui()
                threading.Thread(target=self.auto_generate_frames, daemon=True).start()
        
        def generate_next_frame(self):
            """Generate the next frame by stepping the agent (real-time mode only)."""
            if self.key_frames_only or self.agent.finished:
                return False
                
            state = self.agent.step()
            if state:
                img = board_to_image(
                    state.board.board,
                    state.adjacency_graph,
                    self.original_board_data,
                    self.cell_size
                )
                frame_path = self.save_frame_to_file(img, len(self.frames))
                self.frames.append(frame_path)
                self.update_ui()
                return True
            return False
        
        def auto_generate_frames(self):
            """Automatically generate frames in the background (real-time mode only)."""
            if self.key_frames_only:
                return
                
            while not self.agent.finished and not self.generation_complete:
                self.generate_next_frame()
                time.sleep(0.01)
            
            self.generation_complete = True
            print(f"Frame generation complete! Total frames: {len(self.frames)}")
        
        def go_to_frame(self, frame_index):
            """Go to a specific frame."""
            if 0 <= frame_index < len(self.frames):
                self.current_frame = frame_index
                self.update_ui()
        
        def on_frame_change(self, e):
            """Handle frame slider change."""
            self.go_to_frame(int(e.control.value))
        
        def on_speed_change(self, e):
            """Handle speed slider change."""
            self.speed = e.control.value
        
        def update_ui(self):
            """Update all UI components."""
            if not self.page or not self.frames:
                return
                
            try:
                # Update board image
                if self.current_frame < len(self.frames):
                    new_path = self.frames[self.current_frame]
                    self.board_image.src = new_path
                
                # Update frame controls
                self.frame_slider.max = max(1, len(self.frames) - 1)
                self.frame_slider.value = self.current_frame
                
                frame_label = "Solution Step" if self.key_frames_only else "Search Step"
                self.frame_counter_text.value = f"{frame_label}: {self.current_frame + 1}/{len(self.frames)}"
                
                # Update statistics
                self.stats_text.value = self.get_stats_text()
                
                # Update debug information
                self.debug_text.value = self.get_debug_text()
                
                self.page.update()
                
            except Exception as e:
                print(f"Error updating UI: {e}")
        
        def get_stats_text(self):
            """Get formatted statistics text."""
            stats = self.agent.get_stats()
            
            if self.key_frames_only:
                # Show current frame index, not the state index
                if self.current_frame < len(self.agent.all_states):
                    current_state = self.agent.all_states[self.current_frame]
                    unfilled = len(self.problem.regions) - len(current_state.filled_regions)
                else:
                    unfilled = 0
                status = "✅ Solution Found!" if self.agent.solution_found else "❌ No Solution"
            else:
                current_state = self.agent.get_current_state()
                status = "Finished" if self.agent.finished else "Running"
                if self.agent.finished:
                    status += " ✓" if self.agent.solution_found else " ✗"
                unfilled = len(self.problem.regions) - len(current_state.filled_regions) if current_state else len(self.problem.regions)
            
            # Show both sampled frames and original solution length
            total_frames_text = f"{len(self.frames)}"
            if self.key_frames_only and self.agent.original_solution_length > 0:
                total_frames_text += f" (sampled from {self.agent.original_solution_length})"
            
            return f"""Status: {status}
Mode: {stats.get('mode', 'Unknown')}
Total Frames: {total_frames_text}
Current Frame: {self.current_frame + 1}
Unfilled Regions: {unfilled}
Total Regions: {len(self.problem.regions)}"""
        
        def get_debug_text(self):
            """Get formatted debug information."""
            if self.key_frames_only:
                # Fix the indexing issue here
                if self.current_frame >= len(self.agent.all_states):
                    return "No state data available"
                
                state = self.agent.all_states[self.current_frame]
                debug_info = f"=== SOLUTION STEP {self.current_frame + 1} ===\n"
                
                # Show sampling info if frames are limited
                if self.agent.original_solution_length > len(self.agent.all_states):
                    debug_info += f"(Sampled frame from {self.agent.original_solution_length} total steps)\n"
                
            else:
                if not self.agent.current_node:
                    return "No current node"
                state = self.agent.current_node.state
                debug_info = f"=== SEARCH STEP {self.current_frame + 1} ===\n"
                debug_info += f"Node Depth: {self.agent.current_node.depth}\n"
            
            debug_info += f"Filled Regions: {sorted(state.filled_regions)}\n\n"
            
            debug_info += f"=== ADJACENCY GRAPH ===\n"
            for region, neighbors in sorted(state.adjacency_graph.items()):
                debug_info += f"Region {region}: {sorted(neighbors)}\n"
            
            debug_info += f"\n=== REMAINING REGIONS ===\n"
            unfilled_regions = set(state.adjacency_graph.keys()) - state.filled_regions
            for region in sorted(unfilled_regions):
                try:
                    actions = state.get_actions(region)
                    debug_info += f"Region {region}: {len(actions)} possible actions\n"
                except:
                    debug_info += f"Region {region}: Error getting actions\n"
            
            debug_info += f"\n=== BOARD STATE ===\n"
            debug_info += str(state.board)
            
            return debug_info

        def cleanup(self):
            """Clean up temporary files."""
            try:
                for frame_path in self.frames:
                    if os.path.exists(frame_path):
                        os.remove(frame_path)
                os.rmdir(self.temp_dir)
            except:
                pass

    def run_realtime_visualizer(problem, original_board_data, cell_size, key_frames_only=True, max_frames=None):
        """Run the Flet realtime visualizer."""
        def flet_main(page: ft.Page):
            visualizer = NuruominoRealtimeVisualizer(problem, original_board_data, cell_size, key_frames_only, max_frames)
            
            def on_page_close(e):
                visualizer.cleanup()
            
            page.on_close = on_page_close
            visualizer.build_ui(page)
        
        ft.app(target=flet_main)

else:
    def run_realtime_visualizer(problem, original_board_data, cell_size, key_frames_only=True, max_frames=None):
        """Show error message when Flet is not available."""
        print("\n❌ Flet is not installed!")
        print("To use the interactive realtime visualizer, please install Flet:")
        print("   pip install flet")
        print("\nAlternatively, you can:")
        print("   • Generate a GIF: python src/visualizer.py input.txt --gif [--every-step] [--frames N]")
        print("   • Generate static images: python src/visualizer.py input.txt output.txt")
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Nuruomino visualizer with mode selection")
    parser.add_argument("input_file", help="Input board file")
    parser.add_argument("output_file", nargs='?', help="Output board file")
    parser.add_argument("--gif", action="store_true", help="Generate GIF")
    parser.add_argument("--realtime", action="store_true", help="Interactive realtime mode (requires Flet)")
    parser.add_argument("--every-step", action="store_true", help="Capture every step instead of just key frames (slower but more detailed)")
    parser.add_argument("--frames", type=int, help="Maximum number of frames to capture in key frames mode (default: all solution steps)")
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

    # Determine mode
    key_frames_only = not args.every_step
    max_frames = args.frames

    # Validate frames argument
    if max_frames and args.every_step:
        print("Warning: --frames option is ignored when using --every-step mode")
        max_frames = None

    if args.realtime:
        # Realtime mode - requires Flet
        if FLET_AVAILABLE:
            if key_frames_only:
                if max_frames:
                    mode_desc = f"Key Frames (max {max_frames})"
                    print(f"Using Interactive Visualization - {mode_desc}")
                    print(f"Your algorithm will solve instantly, then show up to {max_frames} key solution steps!")
                else:
                    mode_desc = "Key Frames (all steps)"
                    print(f"Using Interactive Visualization - {mode_desc}")
                    print("Your algorithm will solve instantly, then show all solution steps!")
            else:
                mode_desc = "Every Step (Detailed)"
                print(f"Using Interactive Visualization - {mode_desc}")
                print("Will capture every search step in real-time (slower but detailed)")
            run_realtime_visualizer(problem, original_board_data, args.cell_size, key_frames_only, max_frames)
        else:
            run_realtime_visualizer(problem, original_board_data, args.cell_size, key_frames_only, max_frames)
        
    elif args.gif:
        # Generate GIF
        solve_and_save_gif(problem, original_board_data, test_name, images_dir, args.cell_size, key_frames_only, max_frames)
        
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
