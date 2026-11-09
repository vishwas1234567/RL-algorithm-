
# ================================================================
# ROBOCOURIER AI v2.0
# Autonomous Robotics + Deep Reinforcement Learning
# TensorFlow | DQN | LiDAR Simulation | HTML Dashboard
# ================================================================

import sys, subprocess, importlib.util, random, math, json, base64, io, os
for p in ["tensorflow", "numpy", "matplotlib"]:
    if importlib.util.find_spec(p) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML

# ---------------- CONFIGURATION ----------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

EPISODES = 200
MAX_STEPS = 180
BATCH_SIZE = 64
MODEL_PATH = "robocourier_dqn.keras"

print("ROBOCOURIER AI | TensorFlow:", tf.__version__)

# ---------------- ROBOT ENVIRONMENT ----------------

class RobotEnvironment:

    def __init__(self):
        self.size = 10
        self.speed = 0.17
        self.lidar_range = 3.5
        self.robot_radius = 0.20
        self.reset()

    def reset(self):
        self.pos = np.array([1.0, 1.0], dtype=float)
        self.heading = 0.5
        self.goal = np.array([9.0, 9.0])
        self.battery = 180
        self.steps = 0

        self.obstacles = [
            (3.0, 2.8, 0.65),
            (5.0, 5.0, 0.75),
            (7.0, 3.5, 0.65),
            (3.5, 7.0, 0.55),
            (7.5, 7.0, 0.60)
        ]

        self.human_x = 5.0
        self.human_direction = 1

        return self.observe()

    def get_obstacles(self):
        return self.obstacles + [(self.human_x, 6.0, 0.35)]

    def observe(self):

        sensors = []

        # Eight-direction LiDAR-inspired sensors
        for i in range(8):

            angle = self.heading + i * np.pi / 4
            direction = np.array([np.cos(angle), np.sin(angle)])

            distance = self.lidar_range

            # Detect walls
            for axis in range(2):
                if abs(direction[axis]) > 1e-8:
                    wall = 10 if direction[axis] > 0 else 0
                    t = (wall-self.pos[axis])/direction[axis]
                    if t > 0:
                        distance = min(distance, t)

            # Detect circular obstacles
            for ox, oy, radius in self.get_obstacles():

                center = np.array([ox, oy])
                f = self.pos-center

                b = np.dot(direction, f)
                discriminant = b*b-(np.dot(f, f)-radius**2)

                if discriminant >= 0:
                    t = -b-np.sqrt(discriminant)
                    if t > 0:
                        distance = min(distance, t)

            sensors.append(distance/self.lidar_range)

        delta = self.goal-self.pos
        distance = np.linalg.norm(delta)

        bearing = math.atan2(delta[1], delta[0])-self.heading

        return np.array(
            sensors + [
                np.sin(bearing),
                np.cos(bearing),
                distance/14,
                self.battery/180
            ], dtype=np.float32
        )

    def step(self, action):

        old_distance = np.linalg.norm(self.goal-self.pos)

        # Five possible robot actions
        if action == 0:
            pass
        elif action == 1:
            self.heading += 0.22
        elif action == 2:
            self.heading -= 0.22
        elif action == 3:
            self.heading += 0.50
        elif action == 4:
            self.heading -= 0.50

        # Differential-drive-inspired motion
        movement = self.speed*np.array([
            np.cos(self.heading),
            np.sin(self.heading)
        ])

        self.pos += movement
        self.pos = np.clip(self.pos, 0.22, 9.78)

        # Moving human obstacle
        self.human_x += 0.07*self.human_direction

        if self.human_x > 7 or self.human_x < 3:
            self.human_direction *= -1

        self.battery -= 1
        self.steps += 1

        new_distance = np.linalg.norm(self.goal-self.pos)

        reward = 3*(old_distance-new_distance)-0.04

        collision = False

        for ox, oy, radius in self.get_obstacles():
            if np.linalg.norm(self.pos-[ox, oy]) < (
                radius+self.robot_radius
            ):
                collision = True

        done = False
        success = False

        if collision:
            reward -= 25
            done = True

        elif new_distance < 0.55:
            reward += 120
            done = True
            success = True

        elif self.battery <= 0:
            reward -= 20
            done = True

        elif self.steps >= MAX_STEPS:
            done = True

        return self.observe(), reward, done, {
            "collision": collision,
            "success": success
        }


# ---------------- DQN MODEL ----------------

def create_model():

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(12,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(5)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0005),
        loss=tf.keras.losses.Huber()
    )

    return model


agent = create_model()
target = create_model()
target.set_weights(agent.get_weights())

memory = deque(maxlen=30000)

gamma = 0.95
epsilon = 1.0
epsilon_min = 0.04
epsilon_decay = 0.985

# ---------------- EXPERIENCE REPLAY ----------------

def remember(s, a, r, ns, done):
    memory.append((s, a, r, ns, done))


def train_agent():

    if len(memory) < BATCH_SIZE:
        return

    batch = random.sample(memory, BATCH_SIZE)

    states = np.array([x[0] for x in batch])
    actions = np.array([x[1] for x in batch])
    rewards = np.array([x[2] for x in batch])
    next_states = np.array([x[3] for x in batch])
    dones = np.array([x[4] for x in batch])

    current_q = agent(states, training=False).numpy()
    future_q = target(next_states, training=False).numpy()

    updated_q = current_q.copy()

    updated_q[np.arange(BATCH_SIZE), actions] = (
        rewards + gamma*np.max(future_q, axis=1)*(1-dones)
    )

    agent.train_on_batch(states, updated_q)


# ---------------- TRAINING ----------------

env = RobotEnvironment()

reward_history = []
success_history = []
collision_history = []
loss_history = []

print("\nSTARTING REINFORCEMENT LEARNING TRAINING\n")

for episode in range(EPISODES):

    state = env.reset()
    total_reward = 0
    success = 0
    collision = 0

    for step in range(MAX_STEPS):

        if random.random() < epsilon:
            action = random.randrange(5)
        else:
            action = int(np.argmax(
                agent(state[None], training=False).numpy()[0]
            ))

        next_state, reward, done, info = env.step(action)

        remember(
            state, action, reward, next_state, float(done)
        )

        if step % 2 == 0:
            train_agent()

        state = next_state
        total_reward += reward

        if done:
            success = int(info["success"])
            collision = int(info["collision"])
            break

    epsilon = max(epsilon_min, epsilon*epsilon_decay)

    if episode % 5 == 0:
        target.set_weights(agent.get_weights())

    reward_history.append(total_reward)
    success_history.append(success)
    collision_history.append(collision)

    if (episode+1) % 20 == 0:
        recent = np.mean(success_history[-20:])*100

        print(
            f"Episode {episode+1:3d}/{EPISODES} | "
            f"Reward: {total_reward:8.2f} | "
            f"Success: {recent:5.1f}% | "
            f"Epsilon: {epsilon:.3f}"
        )

print("\nTRAINING COMPLETED")

# ---------------- SAVE TRAINED MODEL ----------------

agent.save(MODEL_PATH)

print("Saved model:", MODEL_PATH)

# ---------------- RELOAD MODEL ----------------
# IMPORTANT: All subsequent simulation actions use this model.

trained_agent = tf.keras.models.load_model(MODEL_PATH)

print("Reloaded trained model successfully.")

# ---------------- INDEPENDENT EVALUATION ----------------

evaluation = []

print("\nEVALUATING RELOADED MODEL: 20 EPISODES")

for episode in range(20):

    test_env = RobotEnvironment()
    state = test_env.reset()

    total_reward = 0
    done = False

    while not done:

        # No random actions: trained model controls robot
        action = int(np.argmax(
            trained_agent(state[None], training=False).numpy()[0]
        ))

        state, reward, done, info = test_env.step(action)
        total_reward += reward

    evaluation.append({
        "episode": episode+1,
        "reward": float(total_reward),
        "success": info["success"],
        "collision": info["collision"],
        "steps": test_env.steps,
        "battery": test_env.battery
    })

success_rate = np.mean([x["success"] for x in evaluation])*100
collision_rate = np.mean([x["collision"] for x in evaluation])*100
average_reward = np.mean([x["reward"] for x in evaluation])

# ---------------- LIVE AGENT NAVIGATION ----------------

print("\nSTARTING AUTONOMOUS ROBOT SIMULATION")
print("The saved TensorFlow model is controlling every action.")

simulation = RobotEnvironment()
state = simulation.reset()

trajectory = [simulation.pos.copy()]
headings = [simulation.heading]
sensor_history = [state[:8].copy()]
simulation_rewards = []

done = False
simulation_success = False

while not done:

    action = int(np.argmax(
        trained_agent(state[None], training=False).numpy()[0]
    ))

    state, reward, done, info = simulation.step(action)

    trajectory.append(simulation.pos.copy())
    headings.append(simulation.heading)
    sensor_history.append(state[:8].copy())
    simulation_rewards.append(reward)

simulation_success = info["success"]

trajectory = np.array(trajectory)
sensor_history = np.array(sensor_history)

print("Navigation result:",
      "DELIVERY SUCCESS" if simulation_success else "INCOMPLETE")

# ---------------- ROBOT ANIMATION ----------------

fig, ax = plt.subplots(figsize=(9, 8))

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect("equal")
ax.set_title("ROBOCOURIER AI | Live Autonomous Navigation")
ax.set_xlabel("X position")
ax.set_ylabel("Y position")
ax.grid(alpha=0.25)

# Static obstacles
for ox, oy, radius in simulation.obstacles:
    ax.add_patch(
        plt.Circle((ox, oy), radius, color="gray", alpha=0.7)
    )

ax.scatter([9], [9], marker="*", s=300,
           color="green", label="Delivery destination")

robot = plt.Circle(
    (1, 1), simulation.robot_radius,
    color="blue", zorder=5
)

ax.add_patch(robot)

path, = ax.plot([], [], "b--", alpha=0.6, label="Robot trajectory")

heading_line, = ax.plot([], [], "r-", linewidth=2)

human = plt.Circle((5, 6), 0.35, color="orange",
                   label="Moving human")

ax.add_patch(human)

status = ax.text(0.3, 9.55, "", fontsize=10)

ax.legend(loc="lower right")

def animate(frame):

    p = trajectory[frame]

    robot.center = (p[0], p[1])

    path.set_data(
        trajectory[:frame+1, 0],
        trajectory[:frame+1, 1]
    )

    h = headings[frame]

    heading_line.set_data(
        [p[0], p[0]+0.65*np.cos(h)],
        [p[1], p[1]+0.65*np.sin(h)]
    )

    human.center = (simulation.human_x, 6)

    status.set_text(
        f"Step: {frame} | "
        f"Battery: {max(0,180-frame)} | "
        f"Model: DQN"
    )

    return robot, path, heading_line, human, status


animation = FuncAnimation(
    fig,
    animate,
    frames=len(trajectory),
    interval=100,
    blit=False
)

plt.close(fig)

display(HTML(animation.to_jshtml()))

# ---------------- PERFORMANCE CHARTS ----------------

fig, axes = plt.subplots(2, 2, figsize=(13, 8))

axes[0, 0].plot(reward_history)
axes[0, 0].set_title("Training Rewards")
axes[0, 0].set_xlabel("Episode")
axes[0, 0].grid()

rolling = np.convolve(
    success_history,
    np.ones(10)/10,
    mode="valid"
)*100

axes[0, 1].plot(
    np.arange(10, EPISODES+1),
    rolling
)

axes[0, 1].set_title("Rolling Delivery Success Rate")
axes[0, 1].set_ylabel("Success %")
axes[0, 1].grid()

axes[1, 0].plot(collision_history)
axes[1, 0].set_title("Training Collision History")
axes[1, 0].set_xlabel("Episode")
axes[1, 0].grid()

axes[1, 1].plot(simulation_rewards)
axes[1, 1].set_title("Autonomous Simulation Reward")
axes[1, 1].set_xlabel("Simulation Step")
axes[1, 1].grid()

plt.tight_layout()
plt.show()

# ---------------- HTML DASHBOARD ----------------

def make_chart(x, y, title, ylabel):

    fig, ax = plt.subplots(figsize=(8, 2.5))

    ax.plot(x, y)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Episode")
    ax.grid(alpha=0.3)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)

    return base64.b64encode(buf.getvalue()).decode()


reward_chart = make_chart(
    np.arange(1, EPISODES+1),
    reward_history,
    "DQN Agent Reward History",
    "Reward"
)

success_chart = make_chart(
    np.arange(10, EPISODES+1),
    rolling,
    "Delivery Success Trend",
    "Success %"
)

rows = ""

for item in evaluation:

    rows += f"""
    <tr>
    <td>{item['episode']}</td>
    <td>{item['reward']:.2f}</td>
    <td>{'SUCCESS' if item['success'] else 'FAILED'}</td>
    <td>{'YES' if item['collision'] else 'NO'}</td>
    <td>{item['steps']}</td>
    </tr>
    """

html = f"""
<!DOCTYPE html>
<html>
<head>
<title>RoboCourier AI Dashboard</title>

<style>
body {{
    background:#101827;
    color:#e5edf7;
    font-family:Arial;
    padding:25px;
}}
h1 {{color:#60d9c0;}}
.grid {{
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
    gap:15px;
}}
.card {{
    background:#1c293b;
    padding:20px;
    border-radius:12px;
    margin-bottom:15px;
}}
.value {{
    font-size:30px;
    color:#60d9c0;
    font-weight:bold;
}}
img {{width:100%;}}
table {{width:100%;border-collapse:collapse;}}
td,th {{padding:12px;border-bottom:1px solid #38465b;}}
th {{color:#60d9c0;}}
</style>
</head>

<body>

<h1>ROBOCOURIER AI</h1>
<p>Autonomous Robotics | TensorFlow | Deep Reinforcement Learning</p>

<div class="grid">

<div class="card">
Training Episodes
<div class="value">{EPISODES}</div>
</div>

<div class="card">
Average Evaluation Reward
<div class="value">{average_reward:.2f}</div>
</div>

<div class="card">
Delivery Success Rate
<div class="value">{success_rate:.1f}%</div>
</div>

<div class="card">
Collision Rate
<div class="value">{collision_rate:.1f}%</div>
</div>

</div>

<h2>Agent Learning Analytics</h2>

<div class="card">
<img src="data:image/png;base64,{reward_chart}">
</div>

<div class="card">
<img src="data:image/png;base64,{success_chart}">
</div>

<h2>Independent Agent Evaluation</h2>

<div class="card">
<table>
<tr>
<th>Episode</th>
<th>Reward</th>
<th>Delivery</th>
<th>Collision</th>
<th>Steps</th>
</tr>
{rows}
</table>
</div>

<h2>Autonomous Navigation Review</h2>

<div class="card">
<p>Model: TensorFlow DQN</p>
<p>Loaded model: {MODEL_PATH}</p>
<p>Simulation result:
{'DELIVERY SUCCESS' if simulation_success else 'INCOMPLETE'}</p>
<p>Simulation steps: {len(trajectory)-1}</p>
<p>Final position: {trajectory[-1].round(2).tolist()}</p>
<p>Model-controlled actions: Yes</p>
<p>Evaluation episodes: 20</p>
</div>

</body>
</html>
"""

with open("RoboCourier_Dashboard.html", "w") as f:
    f.write(html)

display(HTML(html))

print("\n" + "="*55)
print("ROBOCOURIER AI | FINAL PRODUCT REPORT")
print("="*55)
print("Saved model:", MODEL_PATH)
print("Evaluation episodes:", len(evaluation))
print("Average reward:", round(average_reward, 2))
print("Delivery success:", round(success_rate, 2), "%")
print("Collision rate:", round(collision_rate, 2), "%")
print("Autonomous simulation:",
      "SUCCESS" if simulation_success else "INCOMPLETE")
print("Dashboard: RoboCourier_Dashboard.html")
print("="*55)
