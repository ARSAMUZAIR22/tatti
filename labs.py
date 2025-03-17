# Title: Simple Reflex Agent - Hand-Pulling Agent
# Description: This agent simulates a human-like reflex action when encountering a hot object.

class Environment:
    def __init__(self, heat_level='High'):
        self.heat_level = heat_level

    def get_percept(self):
        """Return the heat level of the object as the percept."""
        return 'Hot' if self.heat_level == 'High' else 'Cool'

class SimpleReflexAgent:
    def __init__(self):
        pass

    def act(self, percept):
        """Determine action based on the percept (heat level)."""
        if percept == 'Hot':
            return 'Pull hand away, you touched the hot object'
        else:
            return 'You have not touched any hot object, No need to pull away'

def run_agent(agent, environment):
    # The agent reacts to the heat stimulus only once
    percept = environment.get_percept()
    action = agent.act(percept)
    print(f"Percept: {percept}, Action: {action}")

# Create instances of agent and environment
agent = SimpleReflexAgent()
environment = Environment(heat_level='Low')  # Start with a cool object

# Run the agent in the environment (only once)
run_agent(agent, environment)


# Title: Simple Reflex Agent - Vacuum Cleaner
# Description: A reflex-based cleaning agent that cleans a room if it is dirty.

class Environment:
    def __init__(self, state='Dirty'):
        self.state = state

    def get_percept(self):
        return self.state

    def clean_room(self):
        self.state = 'Clean'

class SimpleReflexAgent:
    def __init__(self):
        pass

    def act(self, percept):
        if percept == 'Dirty':
            return 'Clean the room'
        else:
            return 'Room is already clean'

def run_agent(agent, environment, steps):
    for step in range(steps):
        percept = environment.get_percept()
        action = agent.act(percept)
        print(f"Step {step + 1}: Percept - {percept}, Action - {action}")
        if percept == 'Dirty':
            environment.clean_room()

# Create instances of agent and environment
agent = SimpleReflexAgent()
environment = Environment()

# Run the agent in the environment for 5 steps
run_agent(agent, environment, 5)




# Title: Simple Reflex Agent - 2D Grid-Based Vacuum Cleaner
# Description: A vacuum cleaner agent that cleans a 3x3 grid.

class Environment:
    def __init__(self):
        # Create a 3x3 grid, where 'b', 'e', and 'f' are dirty
        self.grid = ['Clean', 'Dirty', 'Clean',
                     'Clean', 'Dirty', 'Dirty',
                     'Clean', 'Clean', 'Clean']

    def get_percept(self, position):
        return self.grid[position]

    def clean_room(self, position):
        self.grid[position] = 'Clean'

    def display_grid(self, agent_position):
        print("\nCurrent Grid State:")
        grid_with_agent = self.grid[:]  # Copy the grid
        grid_with_agent[agent_position] = "🔴"  # Place the agent at the current position
        for i in range(0, 9, 3):
            print(" | ".join(grid_with_agent[i:i + 3]))
        print()  # Extra line for spacing

class SimpleReflexAgent:
    def __init__(self):
        self.position = 0  # Start at 'a' (position 0 in the grid)

    def act(self, percept, grid):
        if percept == 'Dirty':
            grid[self.position] = 'Clean'
            return 'Clean the room'
        else:
            return 'Room is clean'

    def move(self):
        if self.position < 8:
            self.position += 1
            return self.position

def run_agent(agent, environment, steps):
    for step in range(steps):
        percept = environment.get_percept(agent.position)
        action = agent.act(percept, environment.grid)
        print(f"Step {step + 1}: Position {agent.position} -> Percept - {percept}, Action - {action}")
        environment.display_grid(agent.position)  # Display the grid state with agent position
        if percept == 'Dirty':
            environment.clean_room(agent.position)
        agent.move()

# Create instances of agent and environment
agent = SimpleReflexAgent()
environment = Environment()

# Run the agent in the environment for 9 steps (to cover the 3x3 grid)
run_agent(agent, environment, 9)





# Title: Model-Based Agent - Vacuum Cleaner
# Description: A model-based agent that uses a model to decide actions.

class ModelBasedAgent:
    def __init__(self):
        self.model = {}

    def update_model(self, percept):
        self.model['current'] = percept
        print(self.model)

    def predict_action(self):
        if self.model['current'] == 'Dirty':
            return 'Clean the room'
        else:
            return 'Room is clean'

    def act(self, percept):
        self.update_model(percept)
        return self.predict_action()

class Environment:
    def __init__(self, state='Dirty'):
        self.state = state

    def get_percept(self):
        return self.state

    def clean_room(self):
        self.state = 'Clean'

def run_agent(agent, environment, steps):
    for step in range(steps):
        percept = environment.get_percept()
        action = agent.act(percept)
        print(f"Step {step + 1}: Percept - {percept}, Action - {action}")
        if percept == 'Dirty':
            environment.clean_room()

# Create instances of agent and environment
agent = ModelBasedAgent()
environment = Environment()

# Run the agent in the environment for 5 steps
run_agent(agent, environment, 5)






# Title: Model-Based Agent - Closing Window When it Starts to Rain
# Description: A model-based agent that closes windows when it starts to rain.

class Environment:
    def __init__(self, rain='No', windows_open='Open'):
        self.rain = rain
        self.windows_open = windows_open

    def get_percept(self):
        return {'rain': self.rain, 'windows_open': self.windows_open}

    def close_windows(self):
        if self.windows_open == 'Open':
            self.windows_open = 'Closed'

class ModelBasedAgent:
    def __init__(self):
        self.model = {'rain': 'No', 'windows_open': 'Open'}

    def act(self, percept):
        self.model.update(percept)
        if self.model['rain'] == 'Yes' and self.model['windows_open'] == 'Open':
            return 'Close the windows'
        else:
            return 'No action needed'

def run_agent(agent, environment, steps):
    for step in range(steps):
        percept = environment.get_percept()
        action = agent.act(percept)
        print(f"Step {step + 1}: Percept - {percept}, Action - {action}")
        if action == 'Close the windows':
            environment.close_windows()

# Create instances of agent and environment
agent = ModelBasedAgent()
environment = Environment(rain='Yes', windows_open='Open')  # It's raining and windows are open

# Run the agent in the environment for 5 steps
run_agent(agent, environment, 5)




# Title: Goal-Based Agent - From Scratch
# Description: A goal-based agent that cleans a room if it is dirty.

class GoalBasedAgent:
    def __init__(self):
        self.goal = 'Clean'

    def formulate_goal(self, percept):
        if percept == 'Dirty':
            self.goal = 'Clean'
        else:
            self.goal = 'No action needed'

    def act(self, percept):
        self.formulate_goal(percept)
        if self.goal == 'Clean':
            return 'Clean the room'
        else:
            return 'Room is clean'

class Environment:
    def __init__(self, state='Dirty'):
        self.state = state

    def get_percept(self):
        return self.state

    def clean_room(self):
        self.state = 'Clean'

def run_agent(agent, environment, steps):
    for step in range(steps):
        percept = environment.get_percept()
        action = agent.act(percept)
        print(f"Step {step + 1}: Percept - {percept}, Action - {action}")
        if percept == 'Dirty':
            environment.clean_room()

# Create instances of agent and environment
agent = GoalBasedAgent()
environment = Environment()

# Run the agent in the environment for 5 steps
run_agent(agent, environment, 5)








# Title: Utility-Based Agent - Vacuum Cleaner
# Description: A utility-based agent that cleans a room based on utility values.

class UtilityBasedAgent:
    def __init__(self):
        self.utility = {'Dirty': -10, 'Clean': 10}

    def calculate_utility(self, percept):
        return self.utility[percept]

    def select_action(self, percept):
        if percept == 'Dirty':
            return 'Clean the room'
        else:
            return 'No action needed'

    def act(self, percept):
        action = self.select_action(percept)
        return action

class Environment:
    def __init__(self, state='Dirty'):
        self.state = state

    def get_percept(self):
        return self.state

    def clean_room(self):
        self.state = 'Clean'

def run_agent(agent, environment, steps):
    total_utility = 0
    for step in range(steps):
        percept = environment.get_percept()
        action = agent.act(percept)
        utility = agent.calculate_utility(percept)
        print(f"Step {step + 1}: Percept - {percept}, Action - {action}, Utility - {utility}")
        total_utility += utility
        if percept == 'Dirty':
            environment.clean_room()
    print("Total Utility:", total_utility)

# Create instances of agent and environment
agent = UtilityBasedAgent()
environment = Environment()

# Run the agent in the environment for 5 steps
run_agent(agent, environment, 5)









# Title: Utility-Based Agent - Choosing a Movie to Watch
# Description: A utility-based agent that chooses a movie based on reviews and mood.

class UtilityBasedAgent:
    def __init__(self, mood_factor=0.7):
        self.mood_factor = mood_factor

    def utility(self, review):
        return review * self.mood_factor

    def act(self, percept):
        best_movie = None
        best_utility = -float('inf')
        for movie, review in percept.items():
            movie_utility = self.utility(review)
            if movie_utility > best_utility:
                best_movie = movie
                best_utility = movie_utility
        return best_movie

class Environment:
    def __init__(self, movies):
        self.movies = movies

    def get_percept(self):
        return self.movies

def run_agent(agent, environment):
    percept = environment.get_percept()
    best_choice = agent.act(percept)
    print(f"Available Movies: {percept}")
    print(f"Best Movie to Watch: {best_choice}")

# Create instances of agent and environment
agent = UtilityBasedAgent(mood_factor=0.8)
environment = Environment({'Movie A': 7, 'Movie B': 9, 'Movie C': 5})

# Run the agent in the environment
run_agent(agent, environment)







# Title: Learning-Based Agent
# Description: A learning-based agent that learns to clean a room using Q-learning.

import random

class LearningBasedAgent:
    def __init__(self, actions):
        self.Q = {}
        self.actions = actions
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.1  # Exploration rate

    def get_Q_value(self, state, action):
        return self.Q.get((state, action), 0.0)

    def select_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        else:
            return max(self.actions, key=lambda a: self.get_Q_value(state, a))

    def learn(self, state, action, reward, next_state):
        old_Q = self.get_Q_value(state, action)
        best_future_Q = max([self.get_Q_value(next_state, a) for a in self.actions])
        self.Q[(state, action)] = old_Q + self.alpha * (reward + self.gamma * best_future_Q - old_Q)

    def act(self, state):
        action = self.select_action(state)
        return action

class Environment:
    def __init__(self, state='Dirty'):
        self.state = state

    def get_percept(self):
        return self.state

    def clean_room(self):
        self.state = 'Clean'
        return 10

    def no_action_reward(self):
        return 0

def run_agent(agent, environment, steps):
    for step in range(steps):
        percept = environment.get_percept()
        action = agent.act(percept)
        if percept == 'Dirty':
            reward = environment.clean_room()
            print(f"Step {step + 1}: Percept - {percept}, Action - {action}, Reward - {reward}")
        else:
            reward = environment.no_action_reward()
            print(f"Step {step + 1}: Percept - {percept}, Action - {action}, Reward - {reward}")
        next_percept = environment.get_percept()
        agent.learn(percept, action, reward, next_percept)

# Create instances of agent and environment
agent = LearningBasedAgent(['Clean the room', 'No action needed'])
environment = Environment()

# Run the agent in the environment for 5 steps
run_agent(agent, environment, 5)





# Title: Practice Task #1 - Modify the Vacuum Cleaner Code
# Description: Modify the vacuum cleaner code to randomly set the initial state and change the state after each step.

import random

class Environment:
    def __init__(self):
        self.state = random.choice(['Dirty', 'Clean'])

    def get_percept(self):
        return self.state

    def clean_room(self):
        self.state = 'Clean'

    def randomize_state(self):
        self.state = random.choice(['Dirty', 'Clean'])

class SimpleReflexAgent:
    def __init__(self):
        pass

    def act(self, percept):
        if percept == 'Dirty':
            return 'Clean the room'
        else:
            return 'Room is already clean'

def run_agent(agent, environment, steps):
    for step in range(steps):
        percept = environment.get_percept()
        action = agent.act(percept)
        print(f"Step {step + 1}: Percept - {percept}, Action - {action}")
        if percept == 'Dirty':
            environment.clean_room()
        environment.randomize_state()

# Create instances of agent and environment
agent = SimpleReflexAgent()
environment = Environment()

# Run the agent in the environment for 5 steps
run_agent(agent, environment, 5)



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








# Title: Simple Reflex Agent - Vacuum Cleaner
# Description: A reflex-based cleaning agent that cleans a room if it is dirty.

class Environment:
    def __init__(self, state='Dirty'):
        self.state = state

    def get_percept(self):
        return self.state

    def clean_room(self):
        self.state = 'Clean'

class SimpleReflexAgent:
    def __init__(self):
        pass

    def act(self, percept):
        if percept == 'Dirty':
            return 'Clean the room'
        else:
            return 'Room is already clean'

def run_agent(agent, environment, steps):
    for step in range(steps):
        percept = environment.get_percept()
        action = agent.act(percept)
        print(f"Step {step + 1}: Percept - {percept}, Action - {action}")
        if percept == 'Dirty':
            environment.clean_room()

# Create instances of agent and environment
agent = SimpleReflexAgent()
environment = Environment()

# Run the agent in the environment for 5 steps
run_agent(agent, environment, 5)











# Title: Simple Reflex Agent - 2D Grid-Based Vacuum Cleaner
# Description: A vacuum cleaner agent that cleans a 3x3 grid.

class Environment:
    def __init__(self):
        # Create a 3x3 grid, where 'b', 'e', and 'f' are dirty
        self.grid = ['Clean', 'Dirty', 'Clean',
                     'Clean', 'Dirty', 'Dirty',
                     'Clean', 'Clean', 'Clean']

    def get_percept(self, position):
        return self.grid[position]

    def clean_room(self, position):
        self.grid[position] = 'Clean'

    def display_grid(self, agent_position):
        print("\nCurrent Grid State:")
        grid_with_agent = self.grid[:]  # Copy the grid
        grid_with_agent[agent_position] = "🔴"  # Place the agent at the current position
        for i in range(0, 9, 3):
            print(" | ".join(grid_with_agent[i:i + 3]))
        print()  # Extra line for spacing

class SimpleReflexAgent:
    def __init__(self):
        self.position = 0  # Start at 'a' (position 0 in the grid)

    def act(self, percept, grid):
        if percept == 'Dirty':
            grid[self.position] = 'Clean'
            return 'Clean the room'
        else:
            return 'Room is clean'

    def move(self):
        if self.position < 8:
            self.position += 1
            return self.position

def run_agent(agent, environment, steps):
    for step in range(steps):
        percept = environment.get_percept(agent.position)
        action = agent.act(percept, environment.grid)
        print(f"Step {step + 1}: Position {agent.position} -> Percept - {percept}, Action - {action}")
        environment.display_grid(agent.position)  # Display the grid state with agent position
        if percept == 'Dirty':
            environment.clean_room(agent.position)
        agent.move()

# Create instances of agent and environment
agent = SimpleReflexAgent()
environment = Environment()

# Run the agent in the environment for 9 steps (to cover the 3x3 grid)
run_agent(agent, environment, 9)










# Title: Model-Based Agent - Vacuum Cleaner
# Description: A model-based agent that uses a model to decide actions.

class ModelBasedAgent:
    def __init__(self):
        self.model = {}

    def update_model(self, percept):
        self.model['current'] = percept
        print(self.model)

    def predict_action(self):
        if self.model['current'] == 'Dirty':
            return 'Clean the room'
        else:
            return 'Room is clean'

    def act(self, percept):
        self.update_model(percept)
        return self.predict_action()

class Environment:
    def __init__(self, state='Dirty'):
        self.state = state

    def get_percept(self):
        return self.state

    def clean_room(self):
        self.state = 'Clean'

def run_agent(agent, environment, steps):
    for step in range(steps):
        percept = environment.get_percept()
        action = agent.act(percept)
        print(f"Step {step + 1}: Percept - {percept}, Action - {action}")
        if percept == 'Dirty':
            environment.clean_room()

# Create instances of agent and environment
agent = ModelBasedAgent()
environment = Environment()

# Run the agent in the environment for 5 steps
run_agent(agent, environment, 5)










# Title: Model-Based Agent - Closing Window When it Starts to Rain
# Description: A model-based agent that closes windows when it starts to rain.

class Environment:
    def __init__(self, rain='No', windows_open='Open'):
        self.rain = rain
        self.windows_open = windows_open

    def get_percept(self):
        return {'rain': self.rain, 'windows_open': self.windows_open}

    def close_windows(self):
        if self.windows_open == 'Open':
            self.windows_open = 'Closed'

class ModelBasedAgent:
    def __init__(self):
        self.model = {'rain': 'No', 'windows_open': 'Open'}

    def act(self, percept):
        self.model.update(percept)
        if self.model['rain'] == 'Yes' and self.model['windows_open'] == 'Open':
            return 'Close the windows'
        else:
            return 'No action needed'

def run_agent(agent, environment, steps):
    for step in range(steps):
        percept = environment.get_percept()
        action = agent.act(percept)
        print(f"Step {step + 1}: Percept - {percept}, Action - {action}")
        if action == 'Close the windows':
            environment.close_windows()

# Create instances of agent and environment
agent = ModelBasedAgent()
environment = Environment(rain='Yes', windows_open='Open')  # It's raining and windows are open

# Run the agent in the environment for 5 steps
run_agent(agent, environment, 5)













# Title: Goal-Based Agent - From Scratch
# Description: A goal-based agent that cleans a room if it is dirty.

class GoalBasedAgent:
    def __init__(self):
        self.goal = 'Clean'

    def formulate_goal(self, percept):
        if percept == 'Dirty':
            self.goal = 'Clean'
        else:
            self.goal = 'No action needed'

    def act(self, percept):
        self.formulate_goal(percept)
        if self.goal == 'Clean':
            return 'Clean the room'
        else:
            return 'Room is clean'

class Environment:
    def __init__(self, state='Dirty'):
        self.state = state

    def get_percept(self):
        return self.state

    def clean_room(self):
        self.state = 'Clean'

def run_agent(agent, environment, steps):
    for step in range(steps):
        percept = environment.get_percept()
        action = agent.act(percept)
        print(f"Step {step + 1}: Percept - {percept}, Action - {action}")
        if percept == 'Dirty':
            environment.clean_room()

# Create instances of agent and environment
agent = GoalBasedAgent()
environment = Environment()

# Run the agent in the environment for 5 steps
run_agent(agent, environment, 5)













# Title: Utility-Based Agent - Vacuum Cleaner
# Description: A utility-based agent that cleans a room based on utility values.

class UtilityBasedAgent:
    def __init__(self):
        self.utility = {'Dirty': -10, 'Clean': 10}

    def calculate_utility(self, percept):
        return self.utility[percept]

    def select_action(self, percept):
        if percept == 'Dirty':
            return 'Clean the room'
        else:
            return 'No action needed'

    def act(self, percept):
        action = self.select_action(percept)
        return action

class Environment:
    def __init__(self, state='Dirty'):
        self.state = state

    def get_percept(self):
        return self.state

    def clean_room(self):
        self.state = 'Clean'

def run_agent(agent, environment, steps):
    total_utility = 0
    for step in range(steps):
        percept = environment.get_percept()
        action = agent.act(percept)
        utility = agent.calculate_utility(percept)
        print(f"Step {step + 1}: Percept - {percept}, Action - {action}, Utility - {utility}")
        total_utility += utility
        if percept == 'Dirty':
            environment.clean_room()
    print("Total Utility:", total_utility)

# Create instances of agent and environment
agent = UtilityBasedAgent()
environment = Environment()

# Run the agent in the environment for 5 steps
run_agent(agent, environment, 5)









# Title: Utility-Based Agent - Choosing a Movie to Watch
# Description: A utility-based agent that chooses a movie based on reviews and mood.

class UtilityBasedAgent:
    def __init__(self, mood_factor=0.7):
        self.mood_factor = mood_factor

    def utility(self, review):
        return review * self.mood_factor

    def act(self, percept):
        best_movie = None
        best_utility = -float('inf')
        for movie, review in percept.items():
            movie_utility = self.utility(review)
            if movie_utility > best_utility:
                best_movie = movie
                best_utility = movie_utility
        return best_movie

class Environment:
    def __init__(self, movies):
        self.movies = movies

    def get_percept(self):
        return self.movies

def run_agent(agent, environment):
    percept = environment.get_percept()
    best_choice = agent.act(percept)
    print(f"Available Movies: {percept}")
    print(f"Best Movie to Watch: {best_choice}")

# Create instances of agent and environment
agent = UtilityBasedAgent(mood_factor=0.8)
environment = Environment({'Movie A': 7, 'Movie B': 9, 'Movie C': 5})

# Run the agent in the environment
run_agent(agent, environment)













# Title: Learning-Based Agent
# Description: A learning-based agent that learns to clean a room using Q-learning.

import random

class LearningBasedAgent:
    def __init__(self, actions):
        self.Q = {}
        self.actions = actions
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.1  # Exploration rate

    def get_Q_value(self, state, action):
        return self.Q.get((state, action), 0.0)

    def select_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        else:
            return max(self.actions, key=lambda a: self.get_Q_value(state, a))

    def learn(self, state, action, reward, next_state):
        old_Q = self.get_Q_value(state, action)
        best_future_Q = max([self.get_Q_value(next_state, a) for a in self.actions])
        self.Q[(state, action)] = old_Q + self.alpha * (reward + self.gamma * best_future_Q - old_Q)

    def act(self, state):
        action = self.select_action(state)
        return action

class Environment:
    def __init__(self, state='Dirty'):
        self.state = state

    def get_percept(self):
        return self.state

    def clean_room(self):
        self.state = 'Clean'
        return 10

    def no_action_reward(self):
        return 0

def run_agent(agent, environment, steps):
    for step in range(steps):
        percept = environment.get_percept()
        action = agent.act(percept)
        if percept == 'Dirty':
            reward = environment.clean_room()
            print(f"Step {step + 1}: Percept - {percept}, Action - {action}, Reward - {reward}")
        else:
            reward = environment.no_action_reward()
            print(f"Step {step + 1}: Percept - {percept}, Action - {action}, Reward - {reward}")
        next_percept = environment.get_percept()
        agent.learn(percept, action, reward, next_percept)

# Create instances of agent and environment
agent = LearningBasedAgent(['Clean the room', 'No action needed'])
environment = Environment()

# Run the agent in the environment for 5 steps
run_agent(agent, environment, 5)








# Title: Practice Task #1 - Modify the Vacuum Cleaner Code
# Description: Modify the vacuum cleaner code to randomly set the initial state and change the state after each step.

import random

class Environment:
    def __init__(self):
        self.state = random.choice(['Dirty', 'Clean'])

    def get_percept(self):
        return self.state

    def clean_room(self):
        self.state = 'Clean'

    def randomize_state(self):
        self.state = random.choice(['Dirty', 'Clean'])

class SimpleReflexAgent:
    def __init__(self):
        pass

    def act(self, percept):
        if percept == 'Dirty':
            return 'Clean the room'
        else:
            return 'Room is already clean'

def run_agent(agent, environment, steps):
    for step in range(steps):
        percept = environment.get_percept()
        action = agent.act(percept)
        print(f"Step {step + 1}: Percept - {percept}, Action - {action}")
        if percept == 'Dirty':
            environment.clean_room()
        environment.randomize_state()

# Create instances of agent and environment
agent = SimpleReflexAgent()
environment = Environment()

# Run the agent in the environment for 5 steps
run_agent(agent, environment, 5)







# Title: Breadth-First Search (BFS) - Tree Example
# Description: BFS algorithm to traverse a tree and find the goal node.

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

# BFS Function
def bfs(tree, start, goal):
    visited = []  # List for visited nodes
    queue = []    # Initialize a queue
    visited.append(start)
    queue.append(start)

    while queue:
        node = queue.pop(0)  # Dequeue
        print(node, end=" ")
        if node == goal:  # Stop if goal is found
            print("\nGoal found!")
            break
        for neighbour in tree[node]:
            if neighbour not in visited:
                visited.append(neighbour)
                queue.append(neighbour)

# Define Start and Goal nodes
start_node = 'A'
goal_node = 'I'

# Run BFS
print("\nFollowing is the Breadth-First Search (BFS):")
bfs(tree, start_node, goal_node)






# Title: BFS Goal-Based Agent
# Description: A goal-based agent that uses BFS to navigate through a tree.

class Agent:
    def __init__(self, env, goal):
        self.goal = goal
        self.env = env
        self.visited = list()
        self.queue = list()

    def is_goal(self, current_position):
        return current_position == self.goal

    def bfs(self, start):
        self.queue.append(start)
        self.visited.append(start)

        while self.queue:
            current = self.queue.pop(0)
            print(current, end=" ")

            if self.is_goal(current):
                return current  # Goal found

            # Explore neighbors
            for neighbour in self.env.tree[current]:
                if neighbour not in self.visited:
                    self.visited.append(neighbour)
                    self.queue.append(neighbour)
        return None  # Goal not found

class Environment:
    def __init__(self, tree):
        self.tree = tree

def run_agent(tree, start, goal):
    env = Environment(tree)
    agent = Agent(env, goal)

    # Perform BFS
    goal = agent.bfs(start)
    if goal:
        print("\nGoal found!")
    else:
        print("\nGoal not reachable.")

# Define the tree and run the agent
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

start_node = 'A'
goal_node = 'I'

run_agent(tree, start_node, goal_node)








# Title: BFS Goal-Based Agent
# Description: A goal-based agent that uses BFS to navigate through a tree.

class Agent:
    def __init__(self, env, goal):
        self.goal = goal
        self.env = env
        self.visited = list()
        self.queue = list()

    def is_goal(self, current_position):
        return current_position == self.goal

    def bfs(self, start):
        self.queue.append(start)
        self.visited.append(start)

        while self.queue:
            current = self.queue.pop(0)
            print(current, end=" ")

            if self.is_goal(current):
                return current  # Goal found

            # Explore neighbors
            for neighbour in self.env.tree[current]:
                if neighbour not in self.visited:
                    self.visited.append(neighbour)
                    self.queue.append(neighbour)
        return None  # Goal not found

class Environment:
    def __init__(self, tree):
        self.tree = tree

def run_agent(tree, start, goal):
    env = Environment(tree)
    agent = Agent(env, goal)

    # Perform BFS
    goal = agent.bfs(start)
    if goal:
        print("\nGoal found!")
    else:
        print("\nGoal not reachable.")

# Define the tree and run the agent
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

start_node = 'A'
goal_node = 'I'

run_agent(tree, start_node, goal_node)









# Title: BFS Maze Grid Example
# Description: BFS algorithm to find a path in a 2D maze grid.

# Maze representation as a graph
maze = [
    [1, 1, 0],
    [1, 1, 0],
    [0, 1, 1]
]

# Directions for movement (right and down)
directions = [(0, 1), (1, 0)]  # (row, col)

# Convert maze to a graph (adjacency list representation)
def create_graph(maze):
    graph = {}
    rows = len(maze)
    cols = len(maze[0])

    for i in range(rows):
        for j in range(cols):
            if maze[i][j] == 1:  # If it's an open path
                neighbors = []
                for dx, dy in directions:
                    nx, ny = i + dx, j + dy
                    if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] == 1:
                        neighbors.append((nx, ny))
                graph[(i, j)] = neighbors
    return graph

# BFS Function using queue
def bfs(graph, start, goal):
    visited = []  # List for visited nodes
    queue = []    # Initialize queue

    visited.append(start)
    queue.append(start)

    while queue:
        node = queue.pop(0)  # FIFO: Dequeue from front
        print(node, end=" ")

        if node == goal:  # Stop if goal is found
            print("\nGoal found!")
            break

        for neighbour in graph[node]:  # Visit neighbors
            if neighbour not in visited:
                visited.append(neighbour)
                queue.append(neighbour)

# Create graph from maze
graph = create_graph(maze)

# Define Start and Goal nodes
start_node = (0, 0)  # Starting point (0,0)
goal_node = (2, 2)   # Goal point (2,2)

# Run BFS
print("\nFollowing is the Breadth-First Search (BFS):")
bfs(graph, start_node, goal_node)









# Title: Depth-First Search (DFS) - Tree Example
# Description: DFS algorithm to traverse a tree and find the goal node.

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

# DFS Function
def dfs(graph, start, goal):
    visited = []  # List for visited nodes
    stack = []    # Initialize stack
    visited.append(start)
    stack.append(start)

    while stack:
        node = stack.pop()  # LIFO: Pop from top
        print(node, end=" ")

        if node == goal:  # Stop if goal is found
            print("\nGoal found!")
            break

        for neighbour in reversed(graph[node]):  # Reverse to maintain correct order
            if neighbour not in visited:
                visited.append(neighbour)
                stack.append(neighbour)

# Define Start and Goal nodes
start_node = 'A'
goal_node = 'I'

# Run DFS
print("\nFollowing is the Depth-First Search (DFS):")
dfs(tree, start_node, goal_node)










# Title: DFS Goal-Based Agent
# Description: A goal-based agent that uses DFS to navigate through a tree.

class Agent:
    def __init__(self, env, goal):
        self.goal = goal
        self.env = env
        self.visited = list()
        self.stack = list()

    def is_goal(self, current_position):
        return current_position == self.goal

    def dfs(self, start):
        self.stack.append(start)
        self.visited.append(start)

        while self.stack:
            current = self.stack.pop()
            print(current, end=" ")

            if self.is_goal(current):
                return current  # Goal found

            # Explore neighbors
            for neighbour in reversed(self.env.tree.get(current, [])):
                if neighbour not in self.visited:
                    self.visited.append(neighbour)
                    self.stack.append(neighbour)
        return None  # Goal not found

class Environment:
    def __init__(self, tree):
        self.tree = tree

def run_agent(tree, start, goal):
    env = Environment(tree)
    agent = Agent(env, goal)

    # Perform DFS
    goal = agent.dfs(start)
    if goal:
        print("\nGoal found!")
    else:
        print("\nGoal not reachable.")

# Define the tree and run the agent
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

start_node = 'A'
goal_node = 'I'

run_agent(tree, start_node, goal_node)










# Title: Depth-Limited Search (DLS)
# Description: DLS algorithm to traverse a tree with a depth limit.

# Graph without weights for DLS
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

# DLS Function
def dls(graph, start, goal, depth_limit):
    visited = []

    def dfs(node, depth):
        if depth > depth_limit:
            return None  # Limit reached
        visited.append(node)
        if node == goal:
            print(f"Goal found with DLS. Path: {visited}")
            return visited
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                path = dfs(neighbor, depth + 1)
                if path:
                    return path
        visited.pop()  # Backtrack if goal not found
        return None

    return dfs(start, 0)

# Run DLS with depth limit 3
dls(graph, 'A', 'I', 3)











# Title: Iterative Deepening Depth-First Search (IDS)
# Description: IDS algorithm to traverse a tree with increasing depth limits.

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

# DLS Function for IDS
def dls(node, goal, depth, path):
    if depth == 0:
        return False
    if node == goal:
        path.append(node)
        return True
    if node not in tree:
        return False
    for child in tree[node]:
        if dls(child, goal, depth - 1, path):
            path.append(node)  # Store nodes while backtracking
            return True
    return False

# Iterative Deepening Function
def iterative_deepening(start, goal, max_depth):
    for depth in range(max_depth + 1):
        print(f"Depth: {depth}")
        path = []
        if dls(start, goal, depth, path):
            print("\nPath to goal:", " → ".join(reversed(path)))  # Print path correctly
            return
    print("Goal not found within depth limit.")

# Test Iterative Deepening









# Title: Uniform Cost Search (UCS)
# Description: UCS algorithm to find the least-cost path in a weighted graph.

# Graph with different edge costs for UCS
graph = {
    'A': {'B': 2, 'C': 1},
    'B': {'D': 4, 'E': 3},
    'C': {'F': 1, 'G': 5},
    'D': {'H': 2},
    'E': {},
    'F': {'I': 6},
    'G': {},
    'H': {},
    'I': {}
}

# UCS Function
def ucs(graph, start, goal):
    frontier = [(start, 0)]  # (node, cost)
    visited = set()  # Set to keep track of visited nodes
    cost_so_far = {start: 0}  # Cost to reach each node
    came_from = {start: None}  # Path reconstruction

    while frontier:
        # Sort frontier by cost, simulate priority queue
        frontier.sort(key=lambda x: x[1])

        # Pop the node with the lowest cost
        current_node, current_cost = frontier.pop(0)

        # If we've already visited this node, skip it
        if current_node in visited:
            continue

        # Mark the current node as visited
        visited.add(current_node)

        # If we reach the goal, reconstruct the path and return
        if current_node == goal:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = came_from[current_node]
            path.reverse()
            print(f"Goal found with UCS. Path: {path}, Total Cost: {current_cost}")
            return

        # Explore neighbors
        for neighbor, cost in graph[current_node].items():
            new_cost = current_cost + cost
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current_node
                frontier.append((neighbor, new_cost))  # Add to frontier

    print("Goal not found")

# Run UCS
ucs(graph, 'A', 'I')












# Title: Best-First Search (BFS) - Graph Example
# Description: BFS algorithm to traverse a graph and find the goal node using a heuristic.

from queue import PriorityQueue

# Graph Representation with heuristic values
graph = {
    'S': [('A', 3), ('B', 6), ('C', 5)],
    'A': [('D', 9), ('E', 8)],
    'B': [('F', 12), ('G', 14)],
    'C': [('H', 7)],
    'H': [('I', 5), ('J', 6)],
    'I': [('K', 1), ('L', 10), ('M', 2)],
    'D': [], 'E': [], 'F': [], 'G': [], 'J': [], 'K': [], 'L': [], 'M': []
}

# Best-First Search Function
def best_first_search(graph, start, goal):
    visited = set()
    pq = PriorityQueue()
    pq.put((0, start))  # Priority queue with priority as the heuristic value

    while not pq.empty():
        cost, node = pq.get()
        if node not in visited:
            print(node, end=' ')
            visited.add(node)
            if node == goal:
                print("\nGoal reached!")
                return True
            for neighbor, weight in graph[node]:
                if neighbor not in visited:
                    pq.put((weight, neighbor))
    print("\nGoal not reachable!")
    return False

# Example usage:
print("Best-First Search Path:")
best_first_search(graph, 'A', 'F')











# Title: Maze Game Using Best-First Search
# Description: Solve a maze using Best-First Search with Manhattan distance heuristic.

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

def best_first_search(maze, start, end):
    rows, cols = len(maze), len(maze[0])
    start_node = Node(start)
    end_node = Node(end)
    frontier = PriorityQueue()
    frontier.put(start_node)
    visited = set()

    while not frontier.empty():
        current_node = frontier.get()
        current_pos = current_node.position

        if current_pos == end:
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
                new_node.h = heuristic(new_pos, end)
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
end = (4, 4)

path = best_first_search(maze, start, end)
if path:
    print("Path found:", path)
else:
    print("No path found")







# Title: Greedy Best-First Search (GBFS)
# Description: GBFS algorithm to find the goal node using a heuristic.

# Graph with different edge costs
graph = {
    'A': {'B': 2, 'C': 1},
    'B': {'D': 4, 'E': 3},
    'C': {'F': 1, 'G': 5},
    'D': {'H': 2},
    'E': {},
    'F': {'I': 6},
    'G': {},
    'H': {},
    'I': {}
}

# Heuristic function (estimated cost to reach goal 'I')
heuristic = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 7, 'F': 3, 'G': 6, 'H': 2, 'I': 0}

# Greedy Best-First Search Function (without heapq)
def greedy_bfs(graph, start, goal):
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

# Run Greedy Best-First Search
print("\nFollowing is the Greedy Best-First Search (GBFS):")
greedy_bfs(graph, 'A', 'I')











# Title: A* Search
# Description: A* algorithm to find the optimal path using both cost and heuristic.

from queue import PriorityQueue

# Graph with different edge costs
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

# A* Search Function
def a_star(graph, start, goal):
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
            new_g_cost = g_costs[current_node] + cost  # Path cost from start to neighbor
            f_cost = new_g_cost + heuristic[neighbor]  # f(n) = g(n) + h(n)

            if neighbor not in g_costs or new_g_cost < g_costs[neighbor]:
                g_costs[neighbor] = new_g_cost
                came_from[neighbor] = current_node
                frontier.append((neighbor, f_cost))

    print("\nGoal not found")

# Run A* Search
print("\nFollowing is the A* Search:")
a_star(graph, 'A', 'G')












# Title: Beam Search
# Description: Beam Search algorithm to find the best path in a graph with a limited beam width.

import heapq

# Graph Representation
graph = {
    'S': [('A', 3), ('B', 6), ('C', 5)],
    'A': [('D', 9), ('E', 8)],
    'B': [('F', 12), ('G', 14)],
    'C': [('H', 7)],
    'H': [('I', 5), ('J', 6)],
    'I': [('K', 1), ('L', 10), ('M', 2)],
    'D': [], 'E': [], 'F': [], 'G': [], 'J': [], 'K': [], 'L': [], 'M': []
}

# Beam Search Function
def beam_search(start, goal, beam_width=2):
    # Initialize the beam with the start state
    beam = [(0, [start])]  # (cumulative cost, path)

    while beam:
        candidates = []
        # Expand each path in the beam
        for cost, path in beam:
            current_node = path[-1]
            if current_node == goal:
                return path, cost  # Return the path and cost if goal is found
            # Generate successors
            for neighbor, edge_cost in graph.get(current_node, []):
                new_cost = cost + edge_cost
                new_path = path + [neighbor]
                candidates.append((new_cost, new_path))

        # Select top-k paths based on the lowest cumulative cost
        beam = heapq.nsmallest(beam_width, candidates, key=lambda x: x[0])

    return None, float('inf')  # Return None if no path is found

# Run Beam Search
start_node = 'S'
goal_node = 'L'
beam_width = 3
path, cost = beam_search(start=start_node, goal=goal_node, beam_width=beam_width)

# Print results
if path:
    print(f"Path found: {' → '.join(path)} with total cost: {cost}")
else:
    print("No path found.")







# Title: Hill Climbing - N-Queens Problem
# Description: Solve the N-Queens problem using Hill Climbing.

import random

# Heuristic function: Counts the number of pairs of attacking queens
def calculate_conflicts(state):
    conflicts = 0
    n = len(state)
    for i in range(n):
        for j in range(i + 1, n):
            # Check same column or diagonal
            if state[i] == state[j] or abs(state[i] - state[j]) == abs(i - j):
                conflicts += 1
    return conflicts

# Generate neighbors by moving one queen at a time
def get_neighbors(state):
    neighbors = []
    n = len(state)
    for row in range(n):
        for col in range(n):
            if col != state[row]:
                new_state = list(state)
                new_state[row] = col
                neighbors.append(new_state)
    return neighbors

# Simple Hill Climbing function
def simple_hill_climbing(n):
    # Random initial state
    current_state = [random.randint(0, n - 1) for _ in range(n)]
    current_conflicts = calculate_conflicts(current_state)

    while True:
        neighbors = get_neighbors(current_state)
        next_state = None
        next_conflicts = current_conflicts
        # Find the first better neighbor
        for neighbor in neighbors:
            neighbor_conflicts = calculate_conflicts(neighbor)
            if neighbor_conflicts < next_conflicts:
                next_state = neighbor
                next_conflicts = neighbor_conflicts
                break  # Move to the first better neighbor

        # If no better neighbor is found, return the current state
        if next_conflicts >= current_conflicts:
            break

        # Move to the better neighbor
        current_state = next_state
        current_conflicts = next_conflicts

    return current_state, current_conflicts

# Run Simple Hill Climbing for N-Queens
n = 8  # Change N here for different sizes
solution, conflicts = simple_hill_climbing(n)

# Print results
if conflicts == 0:
    print(f"Solution found for {n}-Queens problem:")
    print(solution)
else:
    print(f"Could not find a solution. Stuck at state with {conflicts} conflicts:")
    print(solution)





# Title: Genetic Algorithm - N-Queens Problem
# Description: Solve the N-Queens problem using a Genetic Algorithm.

import random

# Configuration
n = 8  # Number of queens
population_size = 10
mutation_rate = 0.1
max_generations = 1000

# Fitness function: counts non-attacking pairs of queens
def calculate_fitness(individual):
    non_attacking_pairs = 0
    total_pairs = n * (n - 1) // 2  # Maximum possible non-attacking pairs

    # Check for conflicts
    for i in range(n):
        for j in range(i + 1, n):
            # No same column or diagonal conflict
            if individual[i] != individual[j] and abs(individual[i] - individual[j]) != abs(i - j):
                non_attacking_pairs += 1

    # Fitness score is the ratio of non-attacking pairs
    return non_attacking_pairs / total_pairs

# Generate a random individual (chromosome) based on column positions
def create_random_individual():
    return random.sample(range(n), n)  # Ensure unique column positions

# Selection (Top 50%)
def select_parents(population, fitness_scores):
    sorted_population = [board for _, board in sorted(zip(fitness_scores, population), reverse=True)]
    return sorted_population[:len(population) // 2]

# Crossover function: single-point crossover with unique column
def crossover(parent1, parent2):
    point = random.randint(1, n - 2)  # Choose a crossover point
    child = parent1[:point] + parent2[point:]

    # Ensure unique column positions
    missing = set(range(n)) - set(child)
    duplicates = [col for col in child if child.count(col) > 1]
    for i in range(len(child)):
        if child.count(child[i]) > 1:
            child[i] = missing.pop()
    return child

# Mutation function: randomly swap two locations in a route
def mutate(individual):
    idx1, idx2 = random.sample(range(n), 2)
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

# Genetic Algorithm main function
def genetic_algorithm():
    population = [create_random_individual() for _ in range(population_size)]
    generation = 0
    best_fitness = 0

    while best_fitness < 1.0 and generation < max_generations:
        fitness_scores = [calculate_fitness(ind) for ind in population]
        best_fitness = max(fitness_scores)
        print(f"Generation {generation} Best Fitness: {best_fitness}")

        # Check for optimal solution
        if best_fitness == 1.0:
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
    best_individual = max(population, key=calculate_fitness)
    return best_individual, calculate_fitness(best_individual)

# Run the Genetic Algorithm
solution, fitness = genetic_algorithm()
print("Best Solution:", solution)
print("Best Fitness:", fitness)








# Title: Genetic Algorithm - Duty Scheduling Problem
# Description: Solve the duty scheduling problem using a Genetic Algorithm.

import random

# Configuration
num_staff = 5  # Number of employees
num_shifts = 21  # 7 days * 3 shifts per day
max_shifts_per_staff = 7
required_staff_per_shift = 2
population_size = 10
mutation_rate = 0.1
max_generations = 1000

# Fitness function (lower is better)
def evaluate_fitness(schedule):
    penalty = 0

    # Check shift coverage
    for shift in range(num_shifts):
        assigned_count = sum(schedule[staff][shift] for staff in range(num_staff))
        if assigned_count < required_staff_per_shift:
            penalty += (required_staff_per_shift - assigned_count) * 10  # Understaffed penalty

    # Check consecutive shifts for each staff
    for staff in range(num_staff):
        for shift in range(num_shifts - 1):
            if schedule[staff][shift] == 1 and schedule[staff][shift + 1] == 1:
                penalty += 5  # Penalty for consecutive shifts

    return penalty

# Create a random schedule
def create_random_schedule():
    schedule = [[0] * num_shifts for _ in range(num_staff)]
    for staff in range(num_staff):
        assigned_shifts = random.sample(range(num_shifts), random.randint(3, max_shifts_per_staff))
        for shift in assigned_shifts:
            schedule[staff][shift] = 1
    return schedule

# Selection (Top 50%)
def select_parents(population, fitness_scores):
    sorted_population = [x for _, x in sorted(zip(fitness_scores, population))]
    return sorted_population[:len(population) // 2]

# Crossover (Single point crossover)
def crossover(parent1, parent2):
    point = random.randint(0, num_shifts - 1)
    child = [parent1[i][:point] + parent2[i][point:] for i in range(num_staff)]
    return child

# Mutation (Swap shifts for one staff)
def mutate(schedule):
    staff = random.randint(0, num_staff - 1)
    shift1, shift2 = random.sample(range(num_shifts), 2)
    schedule[staff][shift1], schedule[staff][shift2] = schedule[staff][shift2], schedule[staff][shift1]
    return schedule

# Genetic Algorithm loop
population = [create_random_schedule() for _ in range(population_size)]

for generation in range(max_generations):
    fitness_scores = [evaluate_fitness(schedule) for schedule in population]
    best_fitness = min(fitness_scores)
    print(f"Generation {generation + 1}, Best Fitness: {best_fitness}")
    parents = select_parents(population, fitness_scores)
    new_population = []
    while len(new_population) < population_size:
        parent1, parent2 = random.sample(parents, 2)
        child = crossover(parent1, parent2)
        if random.random() < mutation_rate:
            child = mutate(child)
        new_population.append(child)
    population = new_population

# Best schedule
best_schedule = population[fitness_scores.index(min(fitness_scores))]
print("\nBest Schedule (Staff x Shifts):")
for staff in range(num_staff):
    print(f"Staff {staff + 1}: {best_schedule[staff]}")





















