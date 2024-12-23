import sys
import numpy as np
import tensorflow as tf

try:
    import gymnasium as gym
except ModuleNotFoundError:
    print('gymnasium module not found. Try to install with')
    print('pip install gymnasium[box2d]')
    sys.exit(1)


def play(env, model):

    seed = 2000
    obs, _ = env.reset(seed=seed)

    # drop initial frames
    action0 = 0
    for _ in range(50):
        obs,_,_,_,_ = env.step(action0)
    
    done = False
    while not done:
        p = model.predict(np.expand_dims(obs, axis=0)) # adapt 
        action = np.argmax(p)  # adapt
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated


env_arguments = {
    'domain_randomize': False,
    'continuous': False,
    'render_mode': 'human'
}

env_name = 'CarRacing-v2'
env = gym.make(env_name, **env_arguments)

print("Environment:", env_name)
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

model = tf.keras.models.load_model("car_control_classifier.h5") # your trained model

play(env, model)