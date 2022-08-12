# 201594755-sc21kj-leeds.ac.uk
This Repository contains my dissertation project codes. 
It uses a gym environment wrapper from [janwithb](https://github.com/janwithb/carla-gym-wrapper?ref=https://giter.vip) as 
a skeleton code and I built upon it.

## Features
- rendering
- weather
- different observation types (state, pixel)
- traffic
- autopilot
- vehicle and map selection
- configurable environments
- collect, save and load data
- costs (for safe RL)

## Observation Space
Pixel: (3, 168, 168)

| Index         | Value               |
| ------------- |:-------------------:|
| 0             | r channel 168 x 168 |
| 1             | g channel 168 x 168 |
| 2             | b channel 168 x 168 |

State: (9, )

| Index         | Value             |
| ------------- |:-----------------:|
| 0             | x_pos             |
| 1             | y_pos             |
| 2             | z_pos             |
| 3             | pitch             |
| 4             | yaw               |
| 5             | roll              |
| 6             | acceleration      |
| 7             | angular_velocity  |
| 8             | velocity          |

## Action Space
Action: (2, )

| Index         | Value             | Min               | Max               |
| ------------- |:-----------------:|:-----------------:|:-----------------:|
| 0             | throttle_brake    | -1                | 1                 |
| 1             | steer             | -1                | 1                 |

## CARLA Setup in Windows
1. Add the following environment variables:  
```
Download the latest version of [CARLA](carla.org) or use 0.9.12 which is used in this project.
```
2. Install the following extra libraries in a new conda environment
```
pip install pygame
pip install networkx
pip install dotmap
pip install gym
```
3. Open a new conda session, and run the python file:
 ```
 >python train.py
 ```
 
 ## Configure environments
1. Open: carla_env/\__init\__.py

2. Insert a new environment configuration
```
register(
    id='CarlaEnv-state-town01-v1',
    entry_point='carla_env.carla_env:CarlaEnv',
    max_episode_steps=500,
    kwargs={
        'render': True,
        'carla_port': 2000,
        'changing_weather_speed': 0.1,
        'frame_skip': 1,
        'observations_type': 'state',
        'traffic': True,
        'vehicle_name': 'tesla.cybertruck',
        'map_name': 'Town01',
        'autopilot': True
    }
)
```

## Environment usage
```
import gym
import carla_env


env = CarlaEnv(True,2000,0.0,1,'pixel',True,'tesla.model3','Town04',False)
env.reset()
done = False
while not done:
    action = [1, 0]
    next_obs, reward, done, info = env.step(action)
env.close()
```

## Data collection
How to wrap the environment with the data collection wrapper:
```
env = CarlaEnv(True,2000,0.0,1,'pixel',True,'tesla.model3','Town04',False)
env = DataCollector(env, steps=200, save_dir='./output')
```

Load existing dataset:
```
env = DataCollector(env, steps=200, save_dir='./output', load_dir='./output/dataset_200.pkl')
```
