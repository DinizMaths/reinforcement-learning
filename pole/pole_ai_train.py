import gym
import pickle
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from keras.optimizers        import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

env = gym.make("CartPole-v1")

SAVE = True

states = env.observation_space.shape[0]
actions = env.action_space.n

model = Sequential()
model.add(Flatten(input_shape=(1, states)))
model.add(Dense(24, activation="relu"))
model.add(Dense(24, activation="relu"))
model.add(Dense(actions, activation="softmax"))

agent = DQNAgent(
    model=model,
    memory=SequentialMemory(limit=50000, window_length=1),
    policy=BoltzmannQPolicy(),
    nb_actions=actions,
    nb_steps_warmup=10,
    target_model_update=0.01
)

agent.compile(optimizer=Adam(learning_rate=0.001, name="adam"), metrics=["mae"])
agent.fit(env, nb_steps=10000, visualize=True, verbose=1)

# if SAVE:
#     model_path = "pole_ai.pkl"
#     with open(model_path, "wb") as f:
#         pickle.dump(agent, f)

env.close()