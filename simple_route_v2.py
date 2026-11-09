
# ================================================================
# BALANCEBOT v2.0 — PID vs PPO vs HYBRID PID-PPO
# TensorFlow | Continuous Control | Inverted Pendulum
# ================================================================

import sys, subprocess, importlib.util, random, math
for p in ["tensorflow", "numpy", "matplotlib"]:
    if importlib.util.find_spec(p) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------- CONFIGURATION ----------------

DT = 0.02
MAX_STEPS = 500
EPISODES = 350
FORCE_LIMIT = 20.0
GAMMA = 0.99
LAMBDA = 0.95
CLIP = 0.2
UPDATE_EPOCHS = 8
BATCH_SIZE = 64

# ---------------- PHYSICS ENVIRONMENT ----------------

class BalanceRobot:

    def reset(self, seed=None):
        rng = np.random.default_rng(seed)
        self.x = 0.0
        self.xdot = 0.0
        self.theta = rng.uniform(-0.07, 0.07)
        self.thetadot = rng.uniform(-0.04, 0.04)
        self.steps = 0
        return self.state()

    def state(self):
        return np.array([
            self.x/2.4,
            self.xdot/5,
            self.theta/0.3,
            self.thetadot/3
        ], dtype=np.float32)

    def step(self, force):

        force = float(np.clip(force, -20, 20))

        gravity = 9.81
        cart_mass = 1.0
        pole_mass = 0.1
        pole_length = 0.5

        total = cart_mass + pole_mass
        ml = pole_mass*pole_length

        sint = np.sin(self.theta)
        cost = np.cos(self.theta)

        temp = (
            force + ml*self.thetadot**2*sint
        )/total

        theta_acc = (
            gravity*sint-cost*temp
        )/(
            pole_length*(4/3-pole_mass*cost**2/total)
        )

        x_acc = temp-ml*theta_acc*cost/total

        self.xdot += x_acc*DT
        self.x += self.xdot*DT
        self.thetadot += theta_acc*DT
        self.theta += self.thetadot*DT

        self.steps += 1

        failed = (
            abs(self.theta) > np.deg2rad(20)
            or abs(self.x) > 2.4
        )

        reward = (
            1.0
            - 4*self.theta**2
            - 0.08*self.x**2
            - 0.0005*force**2
        )

        if failed:
            reward -= 50

        return self.state(), reward, failed or self.steps >= MAX_STEPS

# ---------------- PPO ACTOR ----------------

class Actor(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(128, activation="tanh")
        self.d2 = tf.keras.layers.Dense(128, activation="tanh")
        self.out = tf.keras.layers.Dense(1, activation="tanh")

    def call(self, state):
        x = self.d1(state)
        x = self.d2(x)
        return self.out(x)*FORCE_LIMIT


class Critic(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(128, activation="tanh")
        self.d2 = tf.keras.layers.Dense(128, activation="tanh")
        self.out = tf.keras.layers.Dense(1)

    def call(self, state):
        return self.out(self.d2(self.d1(state)))


actor = Actor()
critic = Critic()

actor(tf.zeros((1, 4)))
critic(tf.zeros((1, 4)))

log_std = tf.Variable(
    [-1.0], dtype=tf.float32, trainable=True
)

actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# ---------------- GAUSSIAN POLICY ----------------

def policy_distribution(states):

    mean = actor(states, training=False)
    std = tf.exp(log_std)

    return mean, std


def gaussian_log_prob(actions, mean, std):

    return -0.5*tf.reduce_sum(
        ((actions-mean)/std)**2
        + 2*tf.math.log(std)
        + np.log(2*np.pi),
        axis=-1
    )


def choose_action(state, deterministic=False):

    state = tf.convert_to_tensor(
        state[None], dtype=tf.float32
    )

    mean, std = policy_distribution(state)

    if deterministic:
        action = mean
    else:
        action = mean + std*tf.random.normal(tf.shape(mean))

    logprob = gaussian_log_prob(action, mean, std)

    return (
        float(tf.clip_by_value(action[0, 0], -20, 20).numpy()),
        float(logprob[0].numpy()),
        float(action[0, 0].numpy())
    )

# ---------------- GAE ADVANTAGE ESTIMATION ----------------

def calculate_gae(rewards, values, dones):

    advantages = np.zeros(len(rewards), dtype=np.float32)
    gae = 0.0

    for t in reversed(range(len(rewards))):

        if t == len(rewards)-1:
            next_value = 0.0
        else:
            next_value = values[t+1]

        delta = (
            rewards[t]
            + GAMMA*next_value*(1-dones[t])
            - values[t]
        )

        gae = (
            delta
            + GAMMA*LAMBDA*(1-dones[t])*gae
        )

        advantages[t] = gae

    returns = advantages + np.array(values, dtype=np.float32)

    return advantages, returns

# ---------------- PPO UPDATE ----------------

def update_ppo(states, actions, old_logprobs,
               advantages, returns):

    states = tf.convert_to_tensor(states, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.float32)
    old_logprobs = tf.convert_to_tensor(old_logprobs, dtype=tf.float32)

    advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
    returns = tf.convert_to_tensor(returns, dtype=tf.float32)

    advantages = (
        advantages-tf.reduce_mean(advantages)
    )/(tf.math.reduce_std(advantages)+1e-8)

    indices = np.arange(len(states))
    losses = []

    for epoch in range(UPDATE_EPOCHS):

        np.random.shuffle(indices)

        for start in range(0, len(indices), BATCH_SIZE):

            idx = indices[start:start+BATCH_SIZE]

            s = tf.gather(states, idx)
            a = tf.gather(actions, idx)
            old_lp = tf.gather(old_logprobs, idx)
            adv = tf.gather(advantages, idx)
            ret = tf.gather(returns, idx)

            with tf.GradientTape() as tape:

                mean, std = policy_distribution(s)

                new_lp = gaussian_log_prob(a, mean, std)

                ratio = tf.exp(new_lp-old_lp)

                clipped = tf.clip_by_value(
                    ratio, 1-CLIP, 1+CLIP
                )

                policy_loss = -tf.reduce_mean(
                    tf.minimum(ratio*adv, clipped*adv)
                )

                entropy = tf.reduce_mean(
                    tf.math.log(std*tf.sqrt(2*np.pi*np.e))
                )

                total_actor_loss = (
                    policy_loss-0.001*entropy
                )

            variables = actor.trainable_variables+[log_std]

            gradients = tape.gradient(
                total_actor_loss, variables
            )

            gradients, _ = tf.clip_by_global_norm(
                gradients, 0.5
            )

            actor_optimizer.apply_gradients(
                zip(gradients, variables)
            )

            with tf.GradientTape() as tape:

                predicted = tf.squeeze(
                    critic(s, training=True), axis=1
                )

                critic_loss = tf.reduce_mean(
                    tf.square(ret-predicted)
                )

            gradients = tape.gradient(
                critic_loss, critic.trainable_variables
            )

            gradients, _ = tf.clip_by_global_norm(
                gradients, 0.5
            )

            critic_optimizer.apply_gradients(
                zip(gradients, critic.trainable_variables)
            )

            losses.append(float(policy_loss.numpy()))

    return np.mean(losses) if losses else 0.0

# ---------------- PPO TRAINING ----------------

env = BalanceRobot()

reward_history = []
duration_history = []
loss_history = []
success_history = []

print("STARTING PPO TRAINING")

for episode in range(EPISODES):

    state = env.reset(seed=SEED+episode)

    states, actions = [], []
    logprobs, rewards, values, dones = [], [], [], []

    total_reward = 0

    for step in range(MAX_STEPS):

        force, logprob, raw_action = choose_action(state)

        value = float(
            critic(state[None], training=False)[0, 0].numpy()
        )

        next_state, reward, done = env.step(force)

        states.append(state)
        actions.append([raw_action])
        logprobs.append(logprob)
        rewards.append(reward)
        values.append(value)
        dones.append(float(done))

        state = next_state
        total_reward += reward

        if done:
            break

    advantages, returns = calculate_gae(
        rewards, values, dones
    )

    loss = update_ppo(
        np.array(states),
        np.array(actions, dtype=np.float32),
        np.array(logprobs, dtype=np.float32),
        advantages,
        returns
    )

    reward_history.append(total_reward)
    duration_history.append(step+1)
    loss_history.append(loss)
    success_history.append(int(step+1 == MAX_STEPS))

    if (episode+1) % 25 == 0:

        print(
            f"Episode {episode+1:3d}/{EPISODES} | "
            f"Reward: {total_reward:8.2f} | "
            f"Duration: {step+1:3d} | "
            f"Loss: {loss:.4f}"
        )

print("PPO TRAINING COMPLETED")

# ---------------- SAVE TRAINED PPO MODEL ----------------

actor.save("balancebot_ppo_actor.keras")
critic.save("balancebot_ppo_critic.keras")

# Reload exact trained weights
actor = tf.keras.models.load_model(
    "balancebot_ppo_actor.keras"
)

critic = tf.keras.models.load_model(
    "balancebot_ppo_critic.keras"
)

print("Trained PPO models reloaded successfully.")

# ---------------- PID CONTROLLER ----------------

class PID:

    def __init__(self):
        self.integral = 0
        self.previous = 0

    def control(self, robot):

        error = robot.theta

        self.integral = np.clip(
            self.integral+error*DT, -0.5, 0.5
        )

        derivative = (
            error-self.previous
        )/DT

        self.previous = error

        force = (
            95*error
            + 0.5*self.integral
            + 22*robot.thetadot
            - 1.5*robot.x
            - 3*robot.xdot
        )

        return float(np.clip(force, -20, 20))


# ---------------- CONTROLLER SIMULATION ----------------

def simulate(mode, seed=999):

    robot = BalanceRobot()
    state = robot.reset(seed)

    pid = PID()

    history = {
        "x": [], "theta": [],
        "force": [], "reward": []
    }

    for step in range(MAX_STEPS):

        pid_force = pid.control(robot)

        ppo_force, _, _ = choose_action(
            state, deterministic=True
        )

        if mode == "PID":
            force = pid_force

        elif mode == "PPO":
            force = ppo_force

        else:
            # Hybrid adaptive weighted controller
            force = (
                0.65*pid_force
                + 0.35*ppo_force
            )

        state, reward, done = robot.step(force)

        history["x"].append(robot.x)
        history["theta"].append(robot.theta)
        history["force"].append(force)
        history["reward"].append(reward)

        if done:
            break

    history["duration"] = len(history["x"])
    history["success"] = len(history["x"]) == MAX_STEPS

    return history


# ---------------- INDEPENDENT EVALUATION ----------------

results = {}

print("\nINDEPENDENT CONTROLLER EVALUATION")

for mode in ["PID", "PPO", "HYBRID"]:

    runs = [
        simulate(mode, seed=200+i)
        for i in range(20)
    ]

    results[mode] = runs

    print(
        f"{mode:8s} | "
        f"Success: {np.mean([r['success'] for r in runs])*100:.1f}% | "
        f"Duration: {np.mean([r['duration'] for r in runs]):.1f} | "
        f"Reward: {np.mean([sum(r['reward']) for r in runs]):.2f}"
    )


# ---------------- LIVE THREE-ROBOT SIMULATION ----------------

demo = {
    mode: simulate(mode, seed=999)
    for mode in ["PID", "PPO", "HYBRID"]
}

frames = min(len(v["x"]) for v in demo.values())

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
artists = {}

for ax, mode in zip(axes, demo):

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.5, 2.0)
    ax.set_aspect("equal")
    ax.set_title(mode+" CONTROLLER")
    ax.grid(alpha=.25)
    ax.plot([-2.4, 2.4], [0, 0], "k-", linewidth=3)

    cart = plt.Rectangle(
        (-.2, 0), .4, .15, color="steelblue"
    )

    ax.add_patch(cart)

    pole, = ax.plot([], [], color="darkorange", linewidth=6)
    wheel = plt.Circle((0, -.05), .09, color="black")
    ax.add_patch(wheel)

    label = ax.text(-2.3, 1.7, "")

    artists[mode] = cart, pole, wheel, label


def animate(frame):

    output = []

    for mode in demo:

        x = demo[mode]["x"][frame]
        theta = demo[mode]["theta"][frame]
        force = demo[mode]["force"][frame]

        cart, pole, wheel, label = artists[mode]

        cart.set_x(x-.2)
        wheel.center = (x, -.05)

        pole.set_data(
            [x, x+1.2*np.sin(theta)],
            [.15, .15+1.2*np.cos(theta)]
        )

        label.set_text(
            f"Tilt: {np.rad2deg(theta):.2f} deg\n"
            f"Force: {force:.2f} N"
        )

        output.extend([cart, pole, wheel, label])

    fig.suptitle(
        f"BALANCEBOT | Live controller comparison | Step {frame}"
    )

    return output


animation = FuncAnimation(
    fig, animate, frames=frames,
    interval=30, blit=False
)

plt.close(fig)
display(HTML(animation.to_jshtml()))


# ---------------- ANALYTICS DASHBOARD ----------------

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

axes[0, 0].plot(reward_history)
axes[0, 0].set_title("PPO Training Reward")
axes[0, 0].grid()

axes[0, 1].plot(duration_history)
axes[0, 1].set_title("Episode Balance Duration")
axes[0, 1].grid()

for mode in demo:
    axes[1, 0].plot(
        np.rad2deg(demo[mode]["theta"]),
        label=mode
    )

axes[1, 0].set_title("Tilt Angle Comparison")
axes[1, 0].set_ylabel("Degrees")
axes[1, 0].legend()
axes[1, 0].grid()

for mode in demo:
    axes[1, 1].plot(
        demo[mode]["force"],
        label=mode
    )

axes[1, 1].set_title("Motor Force Comparison")
axes[1, 1].legend()
axes[1, 1].grid()

plt.tight_layout()
plt.show()

# ---------------- FINAL PERFORMANCE REPORT ----------------

print("\n"+"="*55)
print("BALANCEBOT PPO EXPERIMENT REPORT")
print("="*55)

for mode in results:

    runs = results[mode]

    print("\n", mode)

    print("Success rate:",
        round(np.mean([r["success"] for r in runs])*100, 2), "%")

    print("Average duration:",
        round(np.mean([r["duration"] for r in runs]), 2))

    print("Average reward:",
        round(np.mean([sum(r["reward"]) for r in runs]), 2))

print("\nSaved actor: balancebot_ppo_actor.keras")
print("Saved critic: balancebot_ppo_critic.keras")

np.savez(
    "balancebot_ppo_results.npz",
    rewards=reward_history,
    durations=duration_history,
    losses=loss_history
)

print("Experiment results saved.")
print("="*55)
