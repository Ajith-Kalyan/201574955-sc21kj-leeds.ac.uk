import os
import time
import gym
import numpy as np
from carlaenv  import CarlaEnv
from stable_baselines3 import PPO

MODE = "SAVE"
TIMESTEP = 2048

#For debugging and running the code.
if MODE == "SAVE":
	EPISODES = 2000

	#Create folders to save model and weights
	models_dir = f"Models/Save/PPO_{int(time.time())}/"
	logdir =  f"logs/Models/Save/PPO_{int(time.time())}/"

	if not os.path.exists(models_dir):
	    	os.makedirs(models_dir)
	if not os.path.exists(logdir):
	    	os.makedirs(logdir)

	#initializing the carla environment for gym.
	env = CarlaEnv(True,2000,0.0,1,'pixel',True,'tesla.model3','Town04',False)
	env.reset()

	#Model creation
	model = PPO("MlpPolicy", env, verbose= 1,batch_size=64,  
		tensorboard_log=logdir, device="cuda", n_steps = 2048)

	#Run the model for the mentioned episodes each with 2048 timesteps.
	for i in range(0,EPISODES):
		model.learn(total_timesteps=TIMESTEP, reset_num_timesteps=False,tb_log_name=f"PPO")
		model.save(f"{models_dir}/{TIMESTEP*i}")
	env.close()

else:
	#For debugging.
	EPISODES = 5

	#Model creation
	env = CarlaEnv(True,2000,0.0,1,'pixel',True,'tesla.model3','Town04',False)
	#env = make_vec_env(env)
	env.reset()
	
	model = PPO("MlpPolicy", env, verbose= 1, device="cuda", n_steps = 100)

	for i in range(0,EPISODES):
		model.learn(total_timesteps=TIMESTEP)
	env.close()