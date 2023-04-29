# An implementation of Conway's Game of Life
#
# Adapted from code by Abdullah Zafar

import time

from tkinter import *

from life import *


class Game:
    def run(life, config) -> None:
        """Runs the game as per config.

        Args:
        - life: the instance to run.
        - config: contains game configurations.

        Returns:
        nothing.
        """
        # Set up animation if required.
        if config.animate:
            # Use tkinter. Set up the rendering window.

            tk = Tk()
            canvas = Canvas(tk, width=config.width, height=config.height)
            tk.title("Game of Life")
            canvas.configure(background=config.bg_color)
            # Indicate that rendering will be in cells.
            canvas.pack()
            # Number of rendered cells in each direction.
            cells_x = config.width // config.cell_size
            cells_y = config.height // config.cell_size

        # Make the required number of iterations.
        for i in range(config.rounds):
            # Animate if specified.
            if config.animate:
                # Clear canvas and add cells as per current state.
                canvas.delete("all")
                for x, y in life.state():
                    # Wrap cell around screen boundaries. Comment for no wrap.
                    x %= cells_x
                    y %= cells_y
                    # Add cell to canvas.
                    x1, y1 = x * config.cell_size, y * config.cell_size
                    x2, y2 = x1 + config.cell_size, y1 + config.cell_size
                    canvas.create_rectangle(x1, y1, x2, y2, fill=config.cell_color)
                # Render cells, pause for next iteration.
                tk.update()
                time.sleep(0.1 / config.speed)
            # Advance the game by one step.
            life.step()


def main():
    # Set up initial configuration.
    config = Config()
    config.animate = True
    config.rounds = 1000
    config.start = Config.dense
    config.speed = 5
    # Initialize a Life instance and run the Game on it.
    life = Life(config.start, False)
    Game.run(life, config)


if __name__ == "__main__":
    main()
