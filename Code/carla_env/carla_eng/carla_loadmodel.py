import gym
from stable_baselines3 import PPO
import os
from carla_env  import CarlaEnv
import time

TIMESTEP = 2048

EPISODES = 10

#Create folders to save model and weights
models_dir = f"models/carla/validate/_{EPISODES}_PPO_{int(time.time())}/"
logdir = f"logs/carla/validate/_{EPISODES}_PPO_{int(time.time())}/"

if not os.path.exists(models_dir):
    	os.makedirs(models_dir)
if not os.path.exists(logdir):
    	os.makedirs(logdir)

#Model creation
env = CarlaEnv(True,2000,0.0,1,'pixel',True,'tesla.model3','Town04',False)
env.reset()

#Trained Model
model = PPO.load("J:/carla_gym/carla_eng/models/carla/save/TopModels/_1000_V5.6.2_PPO_1657393927/110592.zip", env, verbose =1, n_env = 5)

for ep in range(EPISODES):
		obs=env.reset()
		done=False
		while not done:
			action, _states = model.predict(obs)
			obs, reward, done,info = env.step(action)
env.close()
