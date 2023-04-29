# An implementation of Conway's Game of Life
#
# Adapted from code by Abdullah Zafar

from hashtables import *


class Config:
    """Config class.

    Contains game configurations .
    """

    # Some starting configurations.
    glider = [(20, 40), (21, 40), (22, 40), (22, 41), (21, 42)]  # simple glider

    dense = [
        (21, 40),
        (21, 41),
        (21, 42),
        (22, 41),
        (20, 42),
    ]  # explodes into a dense network
    oscillator = [(1, 4), (2, 4), (3, 4)]  # oscillator
    block = [(4, 4), (5, 4), (4, 5), (5, 5)]  # Block

    def __init__(self) -> None:
        """Provides a default configuration.

        Args:
        - self: manadatory reference to this object.

        Returns:
        none
        """
        # ===== Life parameters
        self.start = Config.glider  # starting shape
        # self.start = Config.dense  # starting shape
        # self.start = Config.oscillator  # starting shape
        # self.start = Config.block  # starting shape
        self.rounds = 5000  # number of rounds of the game

        # ===== Animation parameters
        self.animate: bool = False  # switch animation on or off
        # Screen dimensions
        self.width: int = 800
        self.height: int = 800
        # HU colors
        self.bg_color = "#e6d19a"
        self.cell_color = "#580f55"
        # Cell size. Cells are drawn at resolution CELL_SIZE x CELL_SIZE pixels.
        self.cell_size: int = 10
        # Animation speed. Positive integers, bigger is faster animation.
        self.speed: int = 1


class Life:
    """Life class.

    The state of the game.
    """

    def __init__(self, state: [(int, int)], chain: bool = True) -> None:
        """Initializes game state and internal variables.

        Args:
        - self: manadatory reference to this object.
        - state: initial congifuration - (x,y) coordinates of live cells
        - chain: controls whether to use chaining (True) or linear probiing (False)

        Returns:
        none
        """
        # USet implementations.
        self._alive: MySet = None  # intial config: (x, y) coordinates of alive cells.
        self._nbr_count: MyDict = None  # stores count of live neighbors for cells.

        if chain:
            self._alive = ChainedSet(state)
            self._nbr_count = ChainedDict()

        else:
            self._alive = LinearSet(state)
            self._nbr_count = LinearDict()

    def step(self) -> None:
        """One iteration of the game.

        Applies game rules on current live cells in order to compute the next state of the game.

        Args:
        - self: manadatory reference to this object.

        Returns:
        none
        """
        # Compute neighbors of current live cells.
        deltas = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
        neighbors = [(x + dx, y + dy) for x, y in self._alive for dx, dy in deltas]
        # Collect the number of times each coordinate appears as a
        # neighbor. That provides a count of the number of live neighbors of
        # these cells.
        for coord in neighbors:
            self._nbr_count[coord] = self._nbr_count.get(coord, 0) + 1
        # Apply rules based on numberof neighbors.
        for coord, count in self._nbr_count.items():
            # Alive cells with too few or too many alive neighbors die.
            if count == 1 or count > 3:
                self._alive.discard(coord)
            # Cells with 3 alive neighbors come alive.
            elif count == 3:
                self._alive.add(coord)
            # All other live cells survive.
        # Clear for next iteration.
        self._nbr_count.clear()

    def state(self) -> [(int, int)]:
        """Returns the current state of the game.

        Args:
        - self: manadatory reference to this object.

        Returns:
        Coordinates of live cells .
        """
        # self._alive must be iterable, https://stackoverflow.com/a/37639615/1382487
        # return list(self._alive)  # this part was showing error
        return [i for i in self._alive]
