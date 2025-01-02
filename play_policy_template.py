import sys
import numpy as np
import tensorflow as tf

try:
    import gymnasium as gym
except ModuleNotFoundError:
    print('gymnasium module not found. Try to install with')
    print('pip install gymnasium[box2d]')
    sys.exit(1)

# preprocessing function
def preprocess_image(obs):
    # load and decode the image
    img = tf.convert_to_tensor(obs, dtype=tf.float32)
    img = tf.image.resize(img, (96, 96))  
    img = img / 255.0  # normalize to [0, 1]
    return img

def play(env, model):

    seed = 2000
    obs, _ = env.reset(seed=seed)

    # drop initial frames
    action0 = 0
    for _ in range(50):
        obs,_,_,_,_ = env.step(action0)
    
    done = False
    simulation_score = 0
    while not done:
        preprocessed_obs = np.expand_dims(preprocess_image(obs), axis=0)
        p = model.predict(preprocessed_obs) 
        action = np.argmax(p)  
        obs, reward, terminated, truncated, _ = env.step(action)
        simulation_score += reward
        done = terminated or truncated

    print(simulation_score)

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

model = tf.keras.models.load_model(sys.argv[1]) # your trained model

play(env, model)