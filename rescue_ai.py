from stable_baselines3 import PPO
from env import Environment

SCREEN_TITLE = "RescuAI"
SPRITE_SCALING = 0.5

SPRITE_IMAGE_SIZE = 128
SPRITE_SIZE = int(SPRITE_IMAGE_SIZE * SPRITE_SCALING)

SCREEN_WIDTH = SPRITE_SIZE * 8
SCREEN_HEIGHT = SPRITE_SIZE * 8

def main():
    '''
    This function performs inference indefinitely with the most advanced checkpoint.
    '''
    env = Environment(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    model = PPO.load("checkpoints/ckpt_3", env=env)

    obs = env.reset()[0]
    while True:
        obs, _, done = env.step(model.predict(obs)[0])[0:3]
        if done:
            obs = env.reset()[0]
            env.game._save_video()

if __name__ == "__main__":
    main()
