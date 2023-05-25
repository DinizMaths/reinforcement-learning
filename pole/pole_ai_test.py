import gym  # pip install gym
import pickle

env = gym.make("CartPole-v1") 

filename = 'pole_ai.pkl'
with open(filename, 'rb') as file:
    agent = pickle.load(file)

results = agent.test(env, nb_episodes=10, visualize=True)
print(np.mean(results.history["episode_reward"]))

env.close()