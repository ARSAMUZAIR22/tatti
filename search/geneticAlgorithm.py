import random

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

# Genetic Algorithm parameters
POPULATION_SIZE = 50
GENE_LENGTH = 20  # Maximum number of moves in a path
MUTATION_RATE = 0.1
GENERATIONS = 100

# Possible moves
MOVES = ['U', 'D', 'L', 'R']

def generate_individual():
    """Generate a random path (individual)."""
    return [random.choice(MOVES) for _ in range(GENE_LENGTH)]

def generate_population():
    """Generate the initial population."""
    return [generate_individual() for _ in range(POPULATION_SIZE)]

def is_valid(x, y):
    """Check if a position is within the maze boundaries."""
    return 0 <= x < ROWS and 0 <= y < COLS

def simulate_path(path):
    """Simulate a path and return the final position and fitness."""
    x, y = start
    fitness = 0

    for move in path:
        if move == 'U':
            x -= 1
        elif move == 'D':
            x += 1
        elif move == 'L':
            y -= 1
        elif move == 'R':
            y += 1

        # Check if the new position is valid
        if not is_valid(x, y) or maze[x][y] == '1':
            return (x, y), -1  # Invalid path

        fitness += 1

        # Check if the goal is reached
        if (x, y) == goal:
            return (x, y), fitness

    return (x, y), fitness

def fitness_function(path):
    """Evaluate the fitness of a path."""
    (x, y), fitness = simulate_path(path)
    if (x, y) == goal:
        return fitness  # Higher fitness for reaching the goal
    else:
        # Penalize paths that don't reach the goal
        return -abs(x - goal[0]) - abs(y - goal[1])

def select_parents(population, fitness_scores):
    """Select two parents using tournament selection."""
    tournament_size = 5
    parents = []

    for _ in range(2):
        candidates = random.sample(list(zip(population, fitness_scores)), tournament_size)
        winner = max(candidates, key=lambda x: x[1])[0]
        parents.append(winner)

    return parents

def crossover(parent1, parent2):
    """Perform crossover between two parents."""
    crossover_point = random.randint(1, GENE_LENGTH - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(path):
    """Perform mutation on a path."""
    for i in range(len(path)):
        if random.random() < MUTATION_RATE:
            path[i] = random.choice(MOVES)
    return path

def genetic_algorithm():
    """Run the genetic algorithm to solve the maze."""
    population = generate_population()

    for generation in range(GENERATIONS):
        # Evaluate fitness of each individual
        fitness_scores = [fitness_function(individual) for individual in population]

        # Check if a solution is found
        best_fitness = max(fitness_scores)
        best_index = fitness_scores.index(best_fitness)
        best_individual = population[best_index]

        if fitness_function(best_individual) > 0:
            print(f"Solution found in generation {generation}: {best_individual}")
            return best_individual

        # Create the next generation
        new_population = []

        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = select_parents(population, fitness_scores)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1))
            new_population.append(mutate(child2))

        population = new_population

    print("No solution found.")
    return None

# Run the genetic algorithm
solution = genetic_algorithm()
if solution:
    print("Path:", solution)
