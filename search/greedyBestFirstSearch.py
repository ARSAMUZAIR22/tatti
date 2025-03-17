import heapq

class Environment:
    def __init__(self, grid, start, goal):
        """
        Initialize the environment.
        :param grid: 2D list representing the grid (0 = free, 1 = obstacle)
        :param start: Tuple (x, y) representing the start position
        :param goal: Tuple (x, y) representing the goal position
        """
        self.grid = grid
        self.start = start
        self.goal = goal
        self.rows = len(grid)
        self.cols = len(grid[0])

    def is_free(self, x, y):
        """Check if a cell is free (not an obstacle)."""
        return 0 <= x < self.rows and 0 <= y < self.cols and self.grid[x][y] == 0

    def heuristic(self, x, y):
        """Heuristic function (Manhattan distance)."""
        return abs(x - self.goal[0]) + abs(y - self.goal[1])


class Agent:
    def __init__(self, env):
        self.env = env
        self.visited = set()
        self.path = []

    def greedy_best_first_search(self):
        """Perform Greedy Best-First Search."""
        start = self.env.start
        goal = self.env.goal

        # Priority queue: (heuristic, x, y, path)
        queue = [(self.env.heuristic(*start), start[0], start[1], [start])]

        while queue:
            _, x, y, path = heapq.heappop(queue)

            if (x, y) == goal:
                self.path = path
                return True  # Path found

            if (x, y) in self.visited:
                continue

            self.visited.add((x, y))

            # Explore neighbors
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if self.env.is_free(nx, ny) and (nx, ny) not in self.visited:
                    heapq.heappush(queue, (self.env.heuristic(nx, ny), nx, ny, path + [(nx, ny)]))

        return False  # No path found

    def get_path(self):
        """Return the path found by the agent."""
        return self.path


# Example usage
if __name__ == "__main__":
    # Define the grid (0 = free, 1 = obstacle)
    grid = [
        [0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0]
    ]

    # Define start and goal positions
    start = (0, 0)
    goal = (4, 4)

    # Create environment and agent
    env = Environment(grid, start, goal)
    agent = Agent(env)

    # Perform Greedy Best-First Search
    if agent.greedy_best_first_search():
        print("Path found:", agent.get_path())
    else:
        print("No path found.")
