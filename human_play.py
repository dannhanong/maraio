from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import pygame
import time

def init_pygame():
    pygame.init()
    pygame.display.set_mode((1, 1))

def get_keyboard_action():
    keys = pygame.key.get_pressed()
    
    if keys[pygame.K_RIGHT] and keys[pygame.K_SPACE]:
        return 2  # Jump right
    elif keys[pygame.K_LEFT] and keys[pygame.K_SPACE]:
        return 7  # Jump left
    elif keys[pygame.K_RIGHT]:
        return 1  # Right
    elif keys[pygame.K_LEFT]:
        return 6  # Left
    elif keys[pygame.K_SPACE]:
        return 5  # Jump
    elif keys[pygame.K_DOWN]:
        return 10  # Down
    elif keys[pygame.K_UP]:
        return 11  # Up
    else:
        return 0  # NOOP

def main():
    init_pygame()
    env = gym_super_mario_bros.make('SuperMarioBros2-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    
    
    state = env.reset()
    
    try:
        while True:
            env.render()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        raise KeyboardInterrupt
                    elif event.key == pygame.K_r:
                        state = env.reset()
                        print("\nLevel reset!")
            
            action = get_keyboard_action()
            state, reward, done, info = env.step(action)
            
            if done:
                state = env.reset()
            
            time.sleep(0.016)
            
    except KeyboardInterrupt:
        print("\nGame quit!")
    
    finally:
        env.close()
        pygame.quit()

if __name__ == "__main__":
    main() 