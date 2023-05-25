import gym
import pickle

env = gym.make("LunarLander-v2")

filename = 'lunar_ai.pkl'
with open(filename, 'rb') as file:
    agent = pickle.load(file)

results = agent.test(env, nb_episodes=100, visualize=True)
print(np.mean(results.history["episode_reward"]))

env.close()