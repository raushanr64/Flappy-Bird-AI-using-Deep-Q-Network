# 🐦 Flappy Bird AI using Deep Q-Network (DQN)

A Reinforcement Learning project that trains an AI agent to play **Flappy Bird** using a **Deep Q-Network (DQN)** with PyTorch and Gymnasium.

The agent learns through trial and error by interacting with the Flappy Bird environment, storing experiences in a replay memory, and improving its Q-value predictions over time.

---

## 🚀 Project Overview

This project implements a DQN-based Reinforcement Learning agent for the Flappy Bird game.

The agent learns to decide between two actions:

* `0` → Do not flap
* `1` → Flap

The neural network receives the game state and predicts the Q-value for each available action.

The project includes:

* Deep Q-Network (DQN)
* Experience Replay
* Target Network
* Epsilon-Greedy Exploration
* Reward-based learning
* Model checkpointing
* Training and testing modes
* YAML-based hyperparameter configuration
* CPU / CUDA / Apple MPS device support

The training agent automatically selects the available computing device and moves the neural networks to that device.

---

## 🧠 Reinforcement Learning Architecture

```text
              Flappy Bird Environment
                       │
                       ▼
                  Game State
                       │
                       ▼
                ┌─────────────┐
                │     DQN     │
                │ Neural Net  │
                └─────────────┘
                       │
                  Q-values
                       │
                       ▼
                Select Action
               ┌───────┴───────┐
               │               │
            No Flap           Flap
               │               │
               └───────┬───────┘
                       ▼
                Environment
                       │
                  Reward + State
                       │
                       ▼
               Replay Memory
                       │
                       ▼
                DQN Training
```

---

## 🏗️ Project Structure

```text
Flappy-Bird-DQN/
│
├── agent.py
├── dqn.py
├── experience_replay.py
├── game_flappy_bird.py
├── parameter.yaml
├── requirements.txt
├── README.md
│
└── runs/
    ├── flappybirdv0.log
    └── flappybirdv0.pt
```

### File Description

| File                   | Description                                     |
| ---------------------- | ----------------------------------------------- |
| `agent.py`             | Main DQN agent, training loop and testing logic |
| `dqn.py`               | Defines the DQN neural network                  |
| `experience_replay.py` | Implements replay memory                        |
| `game_flappy_bird.py`  | Manual Flappy Bird environment                  |
| `parameter.yaml`       | Hyperparameter configuration                    |
| `runs/`                | Stores training logs and trained model          |

---

## 🧩 DQN Model

The project uses a simple fully connected neural network.

```text
Input State
    │
    ▼
Linear Layer
    │
    ▼
ReLU
    │
    ▼
Linear Layer
    │
    ▼
Q-values
```

The current implementation uses:

* State dimension: `12`
* Hidden layer: `256`
* Action dimension: `2`

The network is implemented using PyTorch `nn.Sequential`, with a linear layer, ReLU activation, and final action-value layer.

---

## 🔁 Experience Replay

The agent stores previous experiences in a replay buffer:

```text
(state, action, next_state, reward, termination)
```

Instead of training only on the latest experience, the agent randomly samples a mini-batch from the replay memory.

This is implemented using Python's `deque` and random sampling.

---

## 🎯 Target Network

The project uses two neural networks:

* **Policy Network** → learns from experience
* **Target Network** → provides target Q-values

The target network is periodically synchronized with the policy network during training.

---

## 🎲 Epsilon-Greedy Exploration

During training, the agent balances exploration and exploitation.

```text
Random Action
     │
     │ ε
     ▼
  Explore

Best Q-value Action
     │
     │ 1 - ε
     ▼
  Exploit
```

The epsilon value gradually decreases during training, allowing the agent to explore initially and increasingly use the learned policy later.

---

## ⚙️ Hyperparameters

The current configuration is:

| Parameter           |           Value |
| ------------------- | --------------: |
| Environment         | `FlappyBird-v0` |
| Initial Epsilon     |           `1.0` |
| Minimum Epsilon     |          `0.05` |
| Epsilon Decay       |        `0.9995` |
| Replay Memory       |        `100000` |
| Mini Batch Size     |            `32` |
| Network Sync Rate   |            `10` |
| Learning Rate (α)   |         `0.001` |
| Discount Factor (γ) |          `0.99` |
| Reward Threshold    |          `1000` |

These values are defined in `parameter.yaml`.

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Flappy-Bird-DQN.git
cd Flappy-Bird-DQN
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install torch gymnasium flappy-bird-gymnasium pygame pyyaml
```

---

## ▶️ Train the AI

Run:

```bash
python agent.py flappybirdv0 --train
```

The training process will:

1. Create the Flappy Bird environment.
2. Initialize the DQN.
3. Create replay memory.
4. Play episodes.
5. Store experiences.
6. Sample mini-batches.
7. Optimize the policy network.
8. Synchronize the target network.
9. Save the best-performing model.

The trained model is saved inside the `runs/` directory.

---

## 🎮 Test the Trained AI

After training, run:

```bash
python agent.py flappybirdv0
```

The saved model is loaded and the environment is rendered so you can watch the trained agent play.

---

## 🕹️ Manual Game

You can also run the Flappy Bird environment manually:

```bash
python game_flappy_bird.py
```

Press:

```text
SPACE → Flap
```

The manual environment uses `FlappyBird-v0` with Gymnasium and Pygame.

---

## 📊 Training Process

During training, the agent calculates the target Q-value using:

```text
Target Q =
Reward + γ × Maximum Future Q-value
```

For terminal states:

```text
Target Q = Reward
```

The implementation calculates target Q-values using the target network and uses MSE loss between predicted and target Q-values.

---

## 💾 Model & Logs

Training automatically creates:

```text
runs/
├── flappybirdv0.log
└── flappybirdv0.pt
```

The `.pt` file contains the trained PyTorch model parameters.

The log file records when a new best reward is achieved.

---

## 🧪 Technologies Used

* Python
* PyTorch
* Gymnasium
* Flappy Bird Gymnasium
* Pygame
* NumPy
* PyYAML
* Deep Reinforcement Learning
* Deep Q-Network (DQN)

---

## 📚 Concepts Demonstrated

This project demonstrates practical implementation of:

* Reinforcement Learning
* Deep Reinforcement Learning
* Markov Decision Process
* Q-Learning
* Deep Q-Network
* Experience Replay
* Target Network
* Epsilon-Greedy Exploration
* Bellman Equation
* Q-value Estimation
* Neural Network Optimization
* Model Checkpointing

---

## 🔮 Future Improvements

Possible improvements include:

* Double DQN
* Dueling DQN
* Prioritized Experience Replay
* TensorBoard training visualization
* Reward graphs
* Training statistics dashboard
* Better hyperparameter tuning
* Model performance comparison
* GPU acceleration
* Improved reward shaping

---

## 👨‍💻 Author

**Raushan Kumar Raj**

Diploma in Computer Science & Engineering

Interested in:

* Artificial Intelligence
* Machine Learning
* Deep Learning
* Reinforcement Learning
* Generative AI

---

## ⭐ If You Like This Project

If you find this project useful for learning Deep Reinforcement Learning, consider giving the repository a ⭐.
