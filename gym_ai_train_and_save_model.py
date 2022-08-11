import gym
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy 

#declareing variables
policy = 'MlpPolicy'
environment_name = 'LunarLander-v2'

#set up environment
env = gym.make(environment_name)

#env.render()
#train environment
model = PPO(policy, env, verbose=1,learning_rate=0.003)
model.learn(total_timesteps=500000)

#evaluate model and render results
evaluate_policy(model, env, n_eval_episodes=10, render=True, return_episode_rewards=True)
env.close()

#save trained model
model.save('ACP_landing_model')

#loading model
model = PPO.load('ACP_landing_model')
    
