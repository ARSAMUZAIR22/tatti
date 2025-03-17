from collections import deque

# Maze representation
maze = [
    ['S', '0', '1', '0', '0'],
    ['1', '0', '1', '0', '1'],
    ['0', '0', '0', '0', '0'],
    ['1', '0', '1', '0', '1'],
    ['0', '0', '0', '0', 'G']
]

# Maze dimensions
ROWS = len(maze)
COLS = len(maze[0])

# Start and goal positions
start = (0, 0)
goal = (4, 4)

# Possible moves (Up, Down, Left, Right)
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def is_valid(x, y):
    """Check if a position is within the maze boundaries and not an obstacle."""
    return 0 <= x < ROWS and 0 <= y < COLS and maze[x][y] != '1'

def get_neighbors(x, y):
    """Get valid neighboring positions."""
    neighbors = []
    for dx, dy in MOVES:
        nx, ny = x + dx, y + dy
        if is_valid(nx, ny):
            neighbors.append((nx, ny))
    return neighbors

def reconstruct_path(came_from, current):
    """Reconstruct the path from start to goal."""
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path

# DFS Implementation
def dfs():
    """Depth-First Search."""
    stack = [(start, [start])]
    visited = set()

    while stack:
        (x, y), path = stack.pop()

        if (x, y) == goal:
            return path

        if (x, y) not in visited:
            visited.add((x, y))
            for nx, ny in get_neighbors(x, y):
                stack.append(((nx, ny), path + [(nx, ny)]))

    return None  # No path found

# BFS Implementation
def bfs():
    """Breadth-First Search."""
    queue = deque([(start, [start])])
    visited = set()

    while queue:
        (x, y), path = queue.popleft()

        if (x, y) == goal:
            return path

        if (x, y) not in visited:
            visited.add((x, y))
            for nx, ny in get_neighbors(x, y):
                queue.append(((nx, ny), path + [(nx, ny)]))

    return None  # No path found

# DLS Implementation
def dls(max_depth):
    """Depth-Limited Search."""
    stack = [(start, [start], 0)]
    visited = set()

    while stack:
        (x, y), path, depth = stack.pop()

        if (x, y) == goal:
            return path

        if depth < max_depth and (x, y) not in visited:
            visited.add((x, y))
            for nx, ny in get_neighbors(x, y):
                stack.append(((nx, ny), path + [(nx, ny)], depth + 1))

    return None  # No path found within depth limit

# IDDFS Implementation
def iddfs():
    """Iterative Deepening Depth-First Search."""
    max_depth = 0
    while True:
        result = dls(max_depth)
        if result:
            return result
        max_depth += 1

# Example usage
if __name__ == "__main__":
    print("DFS Path:", dfs())
    print("BFS Path:", bfs())
    print("DLS Path (Depth Limit = 10):", dls(10))
    print("IDDFS Path:", iddfs())
