import gym
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy

from evaluation_tayeb import evaluate_policy

#declareing variables
environment_name = 'LunarLander-v2'

#set up environment
env = gym.make(environment_name)

#load model
model =  PPO.load('ACP_landing_model')
#evaluate loaded model
evaluate_policy(model, env, n_eval_episodes=1, render=True, return_episode_rewards=True)
env.close()

