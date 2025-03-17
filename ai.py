# Title: Lab Task #1 - Cybersecurity Exercise
# Description: Simulate a cybersecurity exercise where a security agent scans and patches vulnerabilities.

import random

class Environment:
    def __init__(self):
        self.components = {chr(i): random.choice(['Safe', 'Vulnerable']) for i in range(65, 74)}  # A-I

    def get_percept(self, component):
        return self.components[component]

    def patch_component(self, component):
        self.components[component] = 'Safe'

    def display_system(self):
        print("Current System State:")
        for component, status in self.components.items():
            print(f"{component}: {status}")

class SecurityAgent:
    def __init__(self):
        self.vulnerable_components = []

    def scan_system(self, environment):
        for component in environment.components:
            percept = environment.get_percept(component)
            if percept == 'Vulnerable':
                print(f"Warning: Component {component} is vulnerable!")
                self.vulnerable_components.append(component)
            else:
                print(f"Component {component} is safe.")

    def patch_vulnerabilities(self, environment):
        for component in self.vulnerable_components:
            environment.patch_component(component)
            print(f"Component {component} has been patched.")

def run_security_exercise():
    environment = Environment()
    agent = SecurityAgent()

    # Initial System Check
    print("Initial System Check:")
    environment.display_system()

    # System Scan
    print("\nSystem Scan:")
    agent.scan_system(environment)

    # Patching Vulnerabilities
    print("\nPatching Vulnerabilities:")
    agent.patch_vulnerabilities(environment)

    # Final System Check
    print("\nFinal System Check:")
    environment.display_system()

# Run the cybersecurity exercise
run_security_exercise()





# Title: Lab Task #2 - Load Balancer Agent
# Description: A load balancer agent that redistributes tasks across servers.

import random

class Environment:
    def __init__(self):
        self.servers = {i: random.choice(['Underloaded', 'Balanced', 'Overloaded']) for i in range(1, 6)}  # 5 servers

    def get_percept(self):
        return self.servers

    def balance_load(self):
        overloaded_servers = [server for server, status in self.servers.items() if status == 'Overloaded']
        underloaded_servers = [server for server, status in self.servers.items() if status == 'Underloaded']

        for overloaded in overloaded_servers:
            if underloaded_servers:
                underloaded = underloaded_servers.pop(0)
                self.servers[overloaded] = 'Balanced'
                self.servers[underloaded] = 'Balanced'
                print(f"Moved tasks from Server {overloaded} to Server {underloaded}")

    def display_servers(self):
        print("Current Server Load Status:")
        for server, status in self.servers.items():
            print(f"Server {server}: {status}")

class LoadBalancerAgent:
    def __init__(self):
        pass

    def act(self, environment):
        environment.balance_load()

def run_load_balancer():
    environment = Environment()
    agent = LoadBalancerAgent()

    # Initial Server Load
    print("Initial Server Load:")
    environment.display_servers()

    # Load Balancing
    print("\nLoad Balancing:")
    agent.act(environment)

    # Final Server Load
    print("\nFinal Server Load:")
    environment.display_servers()

# Run the load balancer simulation
run_load_balancer()







# Title: Lab Task #3 - Backup Management Agent
# Description: A backup management agent that retries failed backups.

import random

class Environment:
    def __init__(self):
        self.backup_tasks = {i: random.choice(['Completed', 'Failed']) for i in range(1, 6)}  # 5 tasks

    def get_percept(self):
        return self.backup_tasks

    def retry_failed_backups(self):
        for task, status in self.backup_tasks.items():
            if status == 'Failed':
                self.backup_tasks[task] = 'Completed'
                print(f"Retrying Backup Task {task}: Success")

    def display_backups(self):
        print("Current Backup Status:")
        for task, status in self.backup_tasks.items():
            print(f"Backup Task {task}: {status}")

class BackupManagementAgent:
    def __init__(self):
        pass

    def act(self, environment):
        environment.retry_failed_backups()

def run_backup_management():
    environment = Environment()
    agent = BackupManagementAgent()

    # Initial Backup Status
    print("Initial Backup Status:")
    environment.display_backups()

    # Retry Failed Backups
    print("\nRetrying Failed Backups:")
    agent.act(environment)

    # Final Backup Status
    print("\nFinal Backup Status:")
    environment.display_backups()

# Run the backup management simulation
run_backup_management()




# Title: Lab Task #4 - Utility-Based Security Agent
# Description: A utility-based security agent that patches vulnerabilities based on severity.

import random

class Environment:
    def __init__(self):
        self.components = {chr(i): random.choice(['Safe', 'Low Risk Vulnerable', 'High Risk Vulnerable']) for i in range(65, 74)}  # A-I

    def get_percept(self):
        return self.components

    def patch_low_risk(self, component):
        if self.components[component] == 'Low Risk Vulnerable':
            self.components[component] = 'Safe'
            print(f"Component {component} (Low Risk) has been patched.")

    def display_system(self):
        print("Current System State:")
        for component, status in self.components.items():
            print(f"{component}: {status}")

class UtilityBasedSecurityAgent:
    def __init__(self):
        pass

    def act(self, environment):
        for component, status in environment.components.items():
            if status == 'Low Risk Vulnerable':
                environment.patch_low_risk(component)
            elif status == 'High Risk Vulnerable':
                print(f"Component {component} (High Risk) requires premium service to patch.")

def run_security_exercise():
    environment = Environment()
    agent = UtilityBasedSecurityAgent()

    # Initial System Check
    print("Initial System Check:")
    environment.display_system()

    # System Scan and Patching
    print("\nSystem Scan and Patching:")
    agent.act(environment)

    # Final System Check
    print("\nFinal System Check:")
    environment.display_system()

# Run the security exercise
run_security_exercise()






# Title: Lab Task #5 - Goal-Based Hospital Delivery Robot
# Description: A goal-based agent that delivers medicines to patients in a hospital.

class Environment:
    def __init__(self):
        self.patient_schedule = {'Room 1': '10:00 AM', 'Room 2': '11:00 AM', 'Room 3': '12:00 PM'}
        self.medicine_storage = {'Room 1': 'Medicine A', 'Room 2': 'Medicine B', 'Room 3': 'Medicine C'}
        self.staff_availability = {'Room 1': 'Available', 'Room 2': 'Unavailable', 'Room 3': 'Available'}

    def get_percept(self, room):
        return {
            'schedule': self.patient_schedule[room],
            'medicine': self.medicine_storage[room],
            'staff': self.staff_availability[room]
        }

    def deliver_medicine(self, room):
        print(f"Delivered {self.medicine_storage[room]} to {room} at {self.patient_schedule[room]}")

    def alert_staff(self, room):
        if self.staff_availability[room] == 'Unavailable':
            print(f"Alert: Staff in {room} is unavailable!")

class GoalBasedAgent:
    def __init__(self):
        self.goal = 'Deliver Medicine'

    def act(self, environment, room):
        percept = environment.get_percept(room)
        if percept['staff'] == 'Available':
            environment.deliver_medicine(room)
        else:
            environment.alert_staff(room)

def run_hospital_delivery():
    environment = Environment()
    agent = GoalBasedAgent()

    # Deliver medicines to all rooms
    for room in environment.patient_schedule.keys():
        print(f"\nProcessing {room}:")
        agent.act(environment, room)

# Run the hospital delivery simulation
run_hospital_delivery()






# Title: Lab Task #6 - Firefighting Robot
# Description: A firefighting robot that extinguishes fires in a 3x3 grid.

class Environment:
    def __init__(self):
        self.grid = {
            'a': 'Safe', 'b': 'Safe', 'c': 'Fire',
            'd': 'Safe', 'e': 'Fire', 'f': 'Safe',
            'g': 'Safe', 'h': 'Safe', 'i': 'Fire'
        }

    def get_percept(self, room):
        return self.grid[room]

    def extinguish_fire(self, room):
        if self.grid[room] == 'Fire':
            self.grid[room] = 'Safe'
            print(f"Extinguished fire in room {room}")

    def display_grid(self):
        print("Current Grid State:")
        for i, (room, status) in enumerate(self.grid.items()):
            print(f"{room}: {'🔥' if status == 'Fire' else '✔'}", end=' ')
            if (i + 1) % 3 == 0:
                print()

class FirefightingRobot:
    def __init__(self):
        self.position = 'a'

    def act(self, environment):
        percept = environment.get_percept(self.position)
        if percept == 'Fire':
            environment.extinguish_fire(self.position)
        else:
            print(f"No fire in room {self.position}")
        environment.display_grid()

    def move(self, next_position):
        self.position = next_position

def run_firefighting_simulation():
    environment = Environment()
    robot = FirefightingRobot()

    # Predefined path for the robot
    path = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

    for room in path:
        print(f"\nMoving to room {room}:")
        robot.move(room)
        robot.act(environment)

# Run the firefighting simulation
run_firefighting_simulation()









# Title: Lab Task #1 - Depth-Limited Search (DLS) as a Goal-Based Agent
# Description: Implement DLS as a goal-based agent to explore a graph up to a specified depth limit.

class Agent:
    def __init__(self, env, goal, depth_limit):
        self.goal = goal
        self.env = env
        self.depth_limit = depth_limit
        self.visited = list()

    def is_goal(self, current_position):
        return current_position == self.goal

    def dls(self, node, depth):
        if depth > self.depth_limit:
            return None  # Limit reached
        self.visited.append(node)
        if self.is_goal(node):
            print(f"Goal found with DLS. Path: {self.visited}")
            return self.visited
        for neighbor in self.env.graph.get(node, []):
            if neighbor not in self.visited:
                path = self.dls(neighbor, depth + 1)
                if path:
                    return path
        self.visited.pop()  # Backtrack if goal not found
        return None

class Environment:
    def __init__(self, graph):
        self.graph = graph

def run_agent(graph, start, goal, depth_limit):
    env = Environment(graph)
    agent = Agent(env, goal, depth_limit)

    # Perform DLS
    goal = agent.dls(start, 0)
    if goal:
        print("\nGoal found!")
    else:
        print("\nGoal not reachable.")

# Define the graph and run the agent
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F', 'G'],
    'D': ['H'],
    'E': [],
    'F': ['I'],
    'G': [],
    'H': [],
    'I': []
}

start_node = 'A'
goal_node = 'I'
depth_limit = 3

run_agent(graph, start_node, goal_node, depth_limit)






# Title: Lab Task #2 - Traveling Salesman Problem (TSP)
# Description: Find the shortest possible route that visits every city exactly once and returns to the starting point.

from itertools import permutations

# Distance matrix between cities
distances = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]

# Number of cities
n = len(distances)

# Function to calculate the total distance of a route
def calculate_total_distance(route):
    total_distance = 0
    for i in range(n):
        total_distance += distances[route[i]][route[(i + 1) % n]]
    return total_distance

# Function to solve TSP
def tsp():
    cities = list(range(n))
    min_distance = float('inf')
    best_route = None

    # Generate all possible routes
    for route in permutations(cities):
        current_distance = calculate_total_distance(route)
        if current_distance < min_distance:
            min_distance = current_distance
            best_route = route

    return best_route, min_distance

# Solve TSP
best_route, min_distance = tsp()
print(f"Best Route: {best_route}, Minimum Distance: {min_distance}")







# Title: Lab Task #3 - Iterative Deepening DFS on Graph and Tree
# Description: Implement Iterative Deepening DFS on both a graph and a tree.

# Tree Representation
tree = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F', 'G'],
    'D': ['H'],
    'E': [],
    'F': ['I'],
    'G': [],
    'H': [],
    'I': []
}

# Graph Representation
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F', 'G'],
    'D': ['H'],
    'E': [],
    'F': ['I'],
    'G': [],
    'H': [],
    'I': []
}

# DLS Function for IDS
def dls(node, goal, depth, path, structure):
    if depth == 0:
        return False
    if node == goal:
        path.append(node)
        return True
    if node not in structure:
        return False
    for child in structure[node]:
        if dls(child, goal, depth - 1, path, structure):
            path.append(node)  # Store nodes while backtracking
            return True
    return False

# Iterative Deepening Function
def iterative_deepening(start, goal, max_depth, structure):
    for depth in range(max_depth + 1):
        print(f"Depth: {depth}")
        path = []
        if dls(start, goal, depth, path, structure):
            print("\nPath to goal:", " → ".join(reversed(path)))  # Print path correctly
            return
    print("Goal not found within depth limit.")

# Test Iterative Deepening on Tree
print("Iterative Deepening on Tree:")
start_node = 'A'
goal_node = 'I'
max_search_depth = 5
iterative_deepening(start_node, goal_node, max_search_depth, tree)

# Test Iterative Deepening on Graph
print("\nIterative Deepening on Graph:")
iterative_deepening(start_node, goal_node, max_search_depth, graph)









# Title: Lab Task #4 - File Search Using DFS
# Description: Search for a specific file in a folder structure using DFS.

import os

# Function to search for a file using DFS
def dfs_file_search(folder, target_file):
    for root, dirs, files in os.walk(folder):
        if target_file in files:
            print(f"File found at: {os.path.join(root, target_file)}")
            return
    print(f"File '{target_file}' not found.")

# Define the root folder and target file
root_folder = 'Root'
target_file = 'File2.txt'

# Run DFS file search
dfs_file_search(root_folder, target_file)









# Title: Lab Task #5 - Peg Solitaire Search Algorithm
# Description: Create a search algorithm for peg solitaire.

# Peg Solitaire Board Representation
# 0: Empty, 1: Peg
initial_board = [
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 0, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1]
]

# Function to check if the board is solved
def is_solved(board):
    return sum(sum(row) for row in board) == 1

# Function to perform a move
def perform_move(board, move):
    (x1, y1), (x2, y2) = move
    board[x1][y1] = 0
    board[x2][y2] = 1
    board[(x1 + x2) // 2][(y1 + y2) // 2] = 0

# Function to generate possible moves
def generate_moves(board):
    moves = []
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == 1:
                # Check for possible moves in all directions
                for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                    x, y = i + dx, j + dy
                    if 0 <= x < len(board) and 0 <= y < len(board[i]):
                        if board[x][y] == 0 and board[(i + x) // 2][(j + y) // 2] == 1:
                            moves.append(((i, j), (x, y)))
    return moves

# DFS Function to solve Peg Solitaire
def dfs_peg_solitaire(board, path):
    if is_solved(board):
        print("Solution found!")
        print("Path:", path)
        return True

    for move in generate_moves(board):
        new_board = [row[:] for row in board]
        perform_move(new_board, move)
        if dfs_peg_solitaire(new_board, path + [move]):
            return True

    return False

# Run DFS to solve Peg Solitaire
dfs_peg_solitaire(initial_board, [])








# Title: Lab Task #1 - Enhanced Maze Navigation with Multiple Goals
# Description: Modify Best-First Search to find a path through a maze with multiple goal points.

from queue import PriorityQueue

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0  # Cost from start node to current node
        self.h = 0  # Heuristic estimate of the cost from current node to end node
        self.f = 0  # Total cost

    def __lt__(self, other):
        return self.f < other.f

def heuristic(current_pos, end_pos):
    # Manhattan distance heuristic
    return abs(current_pos[0] - end_pos[0]) + abs(current_pos[1] - end_pos[1])

def best_first_search_multi_goal(maze, start, goals):
    rows, cols = len(maze), len(maze[0])
    start_node = Node(start)
    frontier = PriorityQueue()
    frontier.put(start_node)
    visited = set()
    goals_reached = []

    while not frontier.empty():
        current_node = frontier.get()
        current_pos = current_node.position

        if current_pos in goals:
            goals_reached.append(current_pos)
            if len(goals_reached) == len(goals):
                path = []
                while current_node:
                    path.append(current_node.position)
                    current_node = current_node.parent
                return path[::-1]  # Reverse the path to start from the start position

        visited.add(current_pos)
        # Generate adjacent nodes
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            new_pos = (current_pos[0] + dx, current_pos[1] + dy)
            if 0 <= new_pos[0] < rows and 0 <= new_pos[1] < cols and maze[new_pos[0]][new_pos[1]] == 0 and new_pos not in visited:
                new_node = Node(new_pos, current_node)
                new_node.g = current_node.g + 1
                new_node.h = min(heuristic(new_pos, goal) for goal in goals)  # Use the closest goal for heuristic
                new_node.f = new_node.h  # Best-First Search: f(n) = h(n)
                frontier.put(new_node)
                visited.add(new_pos)

    return None  # No path found

# Example maze
maze = [
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0]
]
start = (0, 0)
goals = [(4, 4), (2, 2)]

path = best_first_search_multi_goal(maze, start, goals)
if path:
    print("Path found:", path)
else:
    print("No path found")










# Title: Lab Task #2 - A* Search with Dynamic Edge Costs
# Description: Implement A* Search where edge costs change dynamically.

import random
from queue import PriorityQueue

# Graph with dynamic edge costs
graph = {
    'A': {'B': 4, 'C': 3},
    'B': {'E': 12, 'F': 5},
    'C': {'D': 7, 'E': 10},
    'D': {'E': 2},
    'E': {'G': 5},
    'F': {'G': 16},
    'G': {}
}

# Heuristic function (estimated cost to reach goal 'G')
heuristic = {'A': 14, 'B': 12, 'C': 11, 'D': 6, 'E': 4, 'F': 11, 'G': 0}

def dynamic_a_star(graph, start, goal):
    frontier = [(start, 0 + heuristic[start])]  # List-based priority queue (sorted manually)
    visited = set()  # Set to keep track of visited nodes
    g_costs = {start: 0}  # Cost to reach each node from start
    came_from = {start: None}  # Path reconstruction

    while frontier:
        # Sort frontier by f(n) = g(n) + h(n)
        frontier.sort(key=lambda x: x[1])
        current_node, current_f = frontier.pop(0)  # Get node with lowest f(n)

        if current_node in visited:
            continue

        print(current_node, end=" ")  # Print visited node
        visited.add(current_node)

        # If goal is reached, reconstruct path
        if current_node == goal:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = came_from[current_node]
            path.reverse()
            print(f"\nGoal found with A*. Path: {path}")
            return

        # Explore neighbors
        for neighbor, cost in graph[current_node].items():
            # Simulate dynamic cost changes
            cost = random.randint(1, 10)  # Randomly change the cost
            new_g_cost = g_costs[current_node] + cost  # Path cost from start to neighbor
            f_cost = new_g_cost + heuristic[neighbor]  # f(n) = g(n) + h(n)

            if neighbor not in g_costs or new_g_cost < g_costs[neighbor]:
                g_costs[neighbor] = new_g_cost
                came_from[neighbor] = current_node
                frontier.append((neighbor, f_cost))

    print("\nGoal not found")

# Run Dynamic A* Search
print("\nFollowing is the Dynamic A* Search:")
dynamic_a_star(graph, 'A', 'G')











# Title: Lab Task #3 - Delivery Route Optimization with Time Windows
# Description: Optimize delivery routes using Greedy Best-First Search with time constraints.

# Graph with delivery points and time windows
graph = {
    'Start': {'A': 5, 'B': 3},
    'A': {'C': 4, 'D': 2},
    'B': {'D': 3, 'E': 6},
    'C': {'End': 5},
    'D': {'End': 4},
    'E': {'End': 7},
    'End': {}
}

# Heuristic function (estimated time to reach 'End')
heuristic = {'Start': 10, 'A': 5, 'B': 7, 'C': 3, 'D': 2, 'E': 4, 'End': 0}

# Greedy Best-First Search Function with Time Windows
def greedy_bfs_time_windows(graph, start, goal):
    frontier = [(start, heuristic[start])]  # List-based priority queue (sorted manually)
    visited = set()  # Set to keep track of visited nodes
    came_from = {start: None}  # Path reconstruction

    while frontier:
        # Sort frontier manually by heuristic value (ascending order)
        frontier.sort(key=lambda x: x[1])
        current_node, _ = frontier.pop(0)  # Get node with best heuristic
        if current_node in visited:
            continue

        print(current_node, end=" ")  # Print visited node
        visited.add(current_node)

        # If goal is reached, reconstruct path
        if current_node == goal:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = came_from[current_node]
            path.reverse()
            print(f"\nGoal found with GBFS. Path: {path}")
            return

        # Expand neighbors based on heuristic
        for neighbor in graph[current_node]:
            if neighbor not in visited:
                came_from[neighbor] = current_node
                frontier.append((neighbor, heuristic[neighbor]))
    print("\nGoal not found")

# Run Greedy Best-First Search with Time Windows
print("\nFollowing is the Greedy Best-First Search with Time Windows:")
greedy_bfs_time_windows(graph, 'Start', 'End')









# Title: Lab Task #4 - AI-Based Diagnostic System Using A*
# Description: Use A* algorithm to find the best sequence of medical tests.

from queue import PriorityQueue

# Medical Graph with test costs
medical_graph = {
    "Start": [("Blood Test", 5), ("X-Ray", 8)],
    "Blood Test": [("MRI", 10), ("ECG", 6)],
    "X-Ray": [("CT Scan", 12), ("Ultrasound", 4)],
    "MRI": [("Diagnosis", 3)],
    "ECG": [("Diagnosis", 2)],
    "CT Scan": [("Diagnosis", 4)],
    "Ultrasound": [("Diagnosis", 1)],
    "Diagnosis": []
}

# Heuristic function (estimated probability of reaching diagnosis)
heuristic = {"Start": 4, "Blood Test": 3, "X-Ray": 3, "MRI": 1, "ECG": 1, "CT Scan": 1, "Ultrasound": 1, "Diagnosis": 0}

# A* Search Function for Medical Diagnosis
def a_star_medical(graph, start, goal):
    frontier = [(start, 0 + heuristic[start])]  # List-based priority queue (sorted manually)
    visited = set()  # Set to keep track of visited nodes
    g_costs = {start: 0}  # Cost to reach each node from start
    came_from = {start: None}  # Path reconstruction

    while frontier:
        # Sort frontier by f(n) = g(n) + h(n)
        frontier.sort(key=lambda x: x[1])
        current_node, current_f = frontier.pop(0)  # Get node with lowest f(n)

        if current_node in visited:
            continue

        print(current_node, end=" ")  # Print visited node
        visited.add(current_node)

        # If goal is reached, reconstruct path
        if current_node == goal:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = came_from[current_node]
            path.reverse()
            print(f"\nGoal found with A*. Path: {path}")
            return

        # Explore neighbors
        for neighbor, cost in graph[current_node]:
            new_g_cost = g_costs[current_node] + cost  # Path cost from start to neighbor
            f_cost = new_g_cost + heuristic[neighbor]  # f(n) = g(n) + h(n)

            if neighbor not in g_costs or new_g_cost < g_costs[neighbor]:
                g_costs[neighbor] = new_g_cost
                came_from[neighbor] = current_node
                frontier.append((neighbor, f_cost))

    print("\nGoal not found")

# Run A* Search for Medical Diagnosis
print("\nFollowing is the A* Search for Medical Diagnosis:")
a_star_medical(medical_graph, "Start", "Diagnosis")









# Title: Lab Task #5 - AI-Based Recruitment System Using Best-First Search
# Description: Implement Best-First Search to find the most suitable job candidates.

# Candidate Graph with similarity scores
candidate_graph = {
    "Start": [("Candidate A", 85), ("Candidate B", 75)],
    "Candidate A": [("Candidate C", 90), ("Candidate D", 80)],
    "Candidate B": [("Candidate D", 78), ("Candidate E", 70)],
    "Candidate C": [("Best Match", 95)],
    "Candidate D": [("Best Match", 88)],
    "Candidate E": [("Best Match", 80)],
    "Best Match": []
}

# Heuristic function (estimated match score)
heuristic = {"Start": 0, "Candidate A": 85, "Candidate B": 75, "Candidate C": 90, "Candidate D": 80, "Candidate E": 70, "Best Match": 95}

# Best-First Search Function for Recruitment
def best_first_search_recruitment(graph, start, goal):
    frontier = [(start, heuristic[start])]  # List-based priority queue (sorted manually)
    visited = set()  # Set to keep track of visited nodes
    came_from = {start: None}  # Path reconstruction

    while frontier:
        # Sort frontier manually by heuristic value (ascending order)
        frontier.sort(key=lambda x: x[1])
        current_node, _ = frontier.pop(0)  # Get node with best heuristic
        if current_node in visited:
            continue

        print(current_node, end=" ")  # Print visited node
        visited.add(current_node)

        # If goal is reached, reconstruct path
        if current_node == goal:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = came_from[current_node]
            path.reverse()
            print(f"\nGoal found with Best-First Search. Path: {path}")
            return

        # Expand neighbors based on heuristic
        for neighbor, score in graph[current_node]:
            if neighbor not in visited:
                came_from[neighbor] = current_node
                frontier.append((neighbor, heuristic[neighbor]))
    print("\nGoal not found")

# Run Best-First Search for Recruitment
print("\nFollowing is the Best-First Search for Recruitment:")
best_first_search_recruitment(candidate_graph, "Start", "Best Match")














# Title: Lab Task #1 - Beam Search for Chess
# Description: Use Beam Search to predict the best move in a game of chess.

import heapq

# Simulated chess board state and possible moves
chess_moves = {
    'S': [('A', 3), ('B', 6), ('C', 5)],
    'A': [('D', 9), ('E', 8)],
    'B': [('F', 12), ('G', 14)],
    'C': [('H', 7)],
    'H': [('I', 5), ('J', 6)],
    'I': [('K', 1), ('L', 10), ('M', 2)],
    'D': [], 'E': [], 'F': [], 'G': [], 'J': [], 'K': [], 'L': [], 'M': []
}

# Beam Search Function
def beam_search_chess(start, goal, beam_width=2, depth_limit=3):
    beam = [(0, [start])]  # (cumulative cost, path)

    for _ in range(depth_limit):
        candidates = []
        for cost, path in beam:
            current_node = path[-1]
            if current_node == goal:
                return path, cost
            for neighbor, edge_cost in chess_moves.get(current_node, []):
                new_cost = cost + edge_cost
                new_path = path + [neighbor]
                candidates.append((new_cost, new_path))

        beam = heapq.nsmallest(beam_width, candidates, key=lambda x: x[0])

    return None, float('inf')

# Run Beam Search for Chess
start_node = 'S'
goal_node = 'L'
beam_width = 3
depth_limit = 3
path, cost = beam_search_chess(start=start_node, goal=goal_node, beam_width=beam_width, depth_limit=depth_limit)

# Print results
if path:
    print(f"Best move sequence: {' → '.join(path)} with total cost: {cost}")
else:
    print("No valid move sequence found.")










# Title: Lab Task #2 - Hill Climbing for Delivery Route Optimization
# Description: Use Hill Climbing to find the shortest delivery route.

import random
import math

# List of coordinates for delivery points
delivery_points = [(0, 0), (1, 2), (3, 1), (5, 4), (2, 3)]

# Calculate the total distance of a route
def calculate_distance(route):
    total_distance = 0
    for i in range(len(route) - 1):
        x1, y1 = route[i]
        x2, y2 = route[i + 1]
        total_distance += math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return total_distance

# Generate neighbors by swapping two points in the route
def get_neighbors(route):
    neighbors = []
    for i in range(len(route)):
        for j in range(i + 1, len(route)):
            new_route = route[:]
            new_route[i], new_route[j] = new_route[j], new_route[i]
            neighbors.append(new_route)
    return neighbors

# Hill Climbing function
def hill_climbing_delivery(start_route):
    current_route = start_route
    current_distance = calculate_distance(current_route)

    while True:
        neighbors = get_neighbors(current_route)
        next_route = None
        next_distance = current_distance
        # Find the first better neighbor
        for neighbor in neighbors:
            neighbor_distance = calculate_distance(neighbor)
            if neighbor_distance < next_distance:
                next_route = neighbor
                next_distance = neighbor_distance
                break  # Move to the first better neighbor

        # If no better neighbor is found, return the current route
        if next_distance >= current_distance:
            break

        # Move to the better neighbor
        current_route = next_route
        current_distance = next_distance

    return current_route, current_distance

# Run Hill Climbing for Delivery Route Optimization
start_route = delivery_points
optimized_route, total_distance = hill_climbing_delivery(start_route)

# Print results
print("Optimized Route:", optimized_route)
print("Total Distance:", total_distance)













# Title: Lab Task #3 - Genetic Algorithm for TSP
# Description: Solve the Traveling Salesman Problem using a Genetic Algorithm.

import random
import math

# List of city coordinates
cities = [(0, 0), (1, 2), (3, 1), (5, 4), (2, 3), (4, 5), (6, 2), (7, 3), (8, 1), (9, 4)]

# Configuration
population_size = 10
mutation_rate = 0.1
max_generations = 1000

# Calculate the total distance of a route
def calculate_distance(route):
    total_distance = 0
    for i in range(len(route) - 1):
        x1, y1 = route[i]
        x2, y2 = route[i + 1]
        total_distance += math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return total_distance

# Generate a random individual (chromosome) based on city order
def create_random_individual():
    return random.sample(cities, len(cities))

# Selection (Top 50%)
def select_parents(population, fitness_scores):
    sorted_population = [x for _, x in sorted(zip(fitness_scores, population))]
    return sorted_population[:len(population) // 2]

# Crossover (Order Crossover)
def crossover(parent1, parent2):
    point1, point2 = sorted(random.sample(range(len(parent1)), 2))
    child = parent1[point1:point2]
    for city in parent2:
        if city not in child:
            child.append(city)
    return child

# Mutation (Swap two cities)
def mutate(individual):
    idx1, idx2 = random.sample(range(len(individual)), 2)
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

# Genetic Algorithm main function
def genetic_algorithm_tsp():
    population = [create_random_individual() for _ in range(population_size)]
    generation = 0
    best_fitness = float('inf')

    while generation < max_generations:
        fitness_scores = [calculate_distance(ind) for ind in population]
        best_fitness = min(fitness_scores)
        print(f"Generation {generation} Best Fitness: {best_fitness}")

        # Check for optimal solution
        if best_fitness == 0:
            break

        # Selection
        parents = select_parents(population, fitness_scores)

        # Crossover
        new_population = [crossover(random.choice(parents), random.choice(parents)) for _ in range(population_size)]

        # Mutation
        for i in range(len(new_population)):
            if random.random() < mutation_rate:
                new_population[i] = mutate(new_population[i])

        population = new_population
        generation += 1

    # Return the best solution
    best_individual = min(population, key=calculate_distance)
    return best_individual, calculate_distance(best_individual)

# Run the Genetic Algorithm for TSP
solution, distance = genetic_algorithm_tsp()
print("Best Route:", solution)
print("Total Distance:", distance)







# Title: Lab Task #4 - Genetic Algorithm for Binary String Maximization
# Description: Find a binary string of length 8 that maximizes the number of 1s using a Genetic Algorithm.

import random

# Configuration
string_length = 8
population_size = 10
mutation_rate = 0.1
max_generations = 1000

# Fitness function: counts the number of 1s in the binary string
def calculate_fitness(individual):
    return sum(individual)

# Generate a random individual (binary string)
def create_random_individual():
    return [random.randint(0, 1) for _ in range(string_length)]

# Selection (Top 50%)
def select_parents(population, fitness_scores):
    sorted_population = [x for _, x in sorted(zip(fitness_scores, population), reverse=True)]
    return sorted_population[:len(population) // 2]

# Crossover (Single-point crossover)
def crossover(parent1, parent2):
    point = random.randint(1, string_length - 1)
    child = parent1[:point] + parent2[point:]
    return child

# Mutation (Flip a bit)
def mutate(individual):
    idx = random.randint(0, string_length - 1)
    individual[idx] = 1 - individual[idx]
    return individual

# Genetic Algorithm main function
def genetic_algorithm_binary():
    population = [create_random_individual() for _ in range(population_size)]
    generation = 0
    best_fitness = 0

    while generation < max_generations:
        fitness_scores = [calculate_fitness(ind) for ind in population]
        best_fitness = max(fitness_scores)
        print(f"Generation {generation} Best Fitness: {best_fitness}")

        # Check for optimal solution
        if best_fitness == string_length:










# Title: Lab Task #5 - Beam Search for Knapsack Problem
# Description: Solve the Knapsack Problem using Beam Search.

import heapq

# Item weights and values
items = [(2, 3), (3, 4), (4, 5), (5, 6)]  # (weight, value)
max_weight = 8
beam_width = 4

# Beam Search Function
def beam_search_knapsack(items, max_weight, beam_width):
    beam = [(0, 0, [])]  # (total_value, total_weight, selected_items)

    for item in items:
        weight, value = item
        candidates = []
        for total_value, total_weight, selected_items in beam:
            # Include the current item
            if total_weight + weight <= max_weight:
                new_value = total_value + value
                new_weight = total_weight + weight
                new_items = selected_items + [item]
                candidates.append((new_value, new_weight, new_items))
            # Exclude the current item
            candidates.append((total_value, total_weight, selected_items))

        # Select top-k candidates based on total value
        beam = heapq.nlargest(beam_width, candidates, key=lambda x: x[0])

    # Return the best solution
    return beam[0]

# Run Beam Search for Knapsack Problem
best_solution = beam_search_knapsack(items, max_weight, beam_width)
print("Best Combination of Items:", best_solution[2])
print("Total Value:", best_solution[0])
print("Total Weight:", best_solution[1])




























