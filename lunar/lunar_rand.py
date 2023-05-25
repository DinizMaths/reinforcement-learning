import gym
import random

def random_policy(env):
    return env.action_space.sample()

# Create the LunarLander-v2 environment
env = gym.make("LunarLander-v2")

# Reset the environment to get the initial observation
observation = env.reset()

# Run the policy for a certain number of steps
for _ in range(1000):
    action = random_policy(env)
    observation, reward, done, info = env.step(action)
    env.render()

    if done:
        observation = env.reset()

env.close()
