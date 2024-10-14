import cv2
import numpy as np
import matplotlib.pyplot as plt
from gym import spaces
from keras.models import Sequential
from keras.layers import Dense
import random

# Define the Room environment
class RoomEnv:
    def __init__(self, width=800, height=600):  # Initial dimensions
        self.width = width
        self.height = height
        self.num_zones = 7  # Number of reward zones
        self.reward_zones = []
        self.create_reward_zones()
        self.goal_x = width // 2
        self.goal_y = height // 2

    def create_reward_zones(self):
        # Calculate maximum radius based on environment size
        max_radius = min(self.width, self.height) // 2 - 50  # Leave some margin
        zone_radius = max_radius // self.num_zones  # Adjust radius based on number of zones
        
        for i in range(self.num_zones):
            self.reward_zones.append({
                "radius": zone_radius * (self.num_zones - i),  # Smaller radius means closer to the center
                "reward": 10 - i,  # Higher reward as we get closer to the center
                "passed": False
            })

    def reset(self):
        # Reset the robot to a random position outside the outermost ring
        outer_radius = self.reward_zones[0]["radius"]

        # Choose random positions outside the outer radius
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(outer_radius + 20, outer_radius + 100)
        self.robot_x = int(self.goal_x + distance * np.cos(angle))
        self.robot_y = int(self.goal_y + distance * np.sin(angle))

        # Initialize the reward and done variables
        self.reward = 0.0
        self.done = False

        # Reset zones
        for zone in self.reward_zones:
            zone["passed"] = False

        return {"x": self.robot_x, "y": self.robot_y}

    def step(self, action):
        new_x = self.robot_x
        new_y = self.robot_y

        # Move the robot in the specified direction
        if action == "up":
            new_y -= 5
        elif action == "down":
            new_y += 5
        elif action == "left":
            new_x -= 5
        elif action == "right":
            new_x += 5

        # Ensure robot stays within bounds
        new_x = np.clip(new_x, 0, self.width)
        new_y = np.clip(new_y, 0, self.height)

        # Calculate distance to goal for rewards
        distance_to_goal = np.sqrt((new_x - self.goal_x) ** 2 + (new_y - self.goal_y) ** 2)
        prev_distance_to_goal = np.sqrt((self.robot_x - self.goal_x) ** 2 + (self.robot_y - self.goal_y) ** 2)

        reward = 0.0
        done = False

        # Large reward for reaching the goal
        if distance_to_goal < 10:  # Within 10 pixels of the goal
            reward += 10.0
            done = True
            return {"x": new_x, "y": new_y, "reward": reward, "done": done}

        # Check if the robot enters a closer zone
        for zone in self.reward_zones:
            if distance_to_goal < zone["radius"] and not zone["passed"]:
                reward += zone["reward"]  # Reward for entering a closer zone
                zone["passed"] = True

        # Small reward for moving closer to the goal
        if distance_to_goal < prev_distance_to_goal:
            reward += 0.1  # Encourage moving toward the goal
        else:
            reward -= 0.1  # Penalize moving away from the goal

        self.robot_x = new_x
        self.robot_y = new_y

        return {
            "x": self.robot_x,
            "y": self.robot_y,
            "reward": reward,
            "done": done,
        }

    def render(self):
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for zone in self.reward_zones:
            color = (0, 0, 255) if zone["passed"] else (0, 255, 0)
            cv2.circle(img, (self.goal_x, self.goal_y), zone["radius"], color, 2)

        cv2.circle(img, (self.goal_x, self.goal_y), 10, (0, 0, 255), -1)  # Goal marker
        cv2.circle(img, (int(self.robot_x), int(self.robot_y)), 10, (255, 0, 0), -1)  # Robot marker

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb


# Define the Robot Agent
class RobotAgent:
    def __init__(self, width=800, height=600):  # Initial dimensions
        self.width = width
        self.height = height
        self.env = RoomEnv(width, height)
        self.state = {"x": 0, "y": 0}
        self.action_space = spaces.Discrete(4)  # 4 actions: up, down, left, right
        self.memory = []  # Experience replay memory
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=2, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_space.n, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        # Limit memory size
        if len(self.memory) > 10000:
            self.memory.pop(0)  # Remove the oldest experience

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = self.model.predict(np.array([[state["x"], state["y"]]]))
            if done:
                target[0][action] = reward
            else:
                target[0][action] = reward + 0.9 * np.max(self.model.predict(np.array([[next_state["x"], next_state["y"]]])))

            self.model.fit(np.array([[state["x"], state["y"]]]), target, epochs=1, verbose=0)


# Define the training process with Q-learning
def train_agent(num_episodes):
    agent = RobotAgent()
    for episode in range(num_episodes):
        state = agent.env.reset()
        done = False

        while not done:
            action_values = agent.model.predict(np.array([[state["x"], state["y"]]]))
            action = np.argmax(action_values[0]) if np.random.rand() > 0.1 else random.choice([0, 1, 2, 3])  # Use integers

            # Map action to string
            action_dict = {0: "up", 1: "down", 2: "left", 3: "right"}
            action_str = action_dict[action]  # Get the string representation from the action dictionary

            next_state = agent.env.step(action_str)
            agent.remember(state, action, next_state["reward"], next_state, next_state["done"])

            img = agent.env.render()

            plt.imshow(img)
            plt.draw()
            plt.pause(0.01)

            state = next_state
            done = next_state["done"]

        agent.replay(32)

    plt.show()
    print("Robot has Completed the Test")


# Train the agent
num_episodes = 100
train_agent(num_episodes)
