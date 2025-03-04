import numpy as np
import matplotlib.pyplot as plt
import random 


# Video
def DrawActor(x:int, y:int, rotate:bool = False):
    
    canva = np.ones((28, 28))
    actor = np.asarray([
            [1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 0, 1, 1],
            [1, 0, 1, 0, 1]
        ])

    if rotate:
        actor = actor.T
        
    canva[y - 2 : y + 3, x - 2 : x + 3] = actor
    
    return canva     

# Jumping
def CreateJumpingVideo(n_frame:int):
    
    x = 14
    y = random.randint(21, 24)
    video = np.ones((n_frame, 28, 28))
    speed_y = -random.randint(4, 6)
    accel_y = -int(speed_y / n_frame * 2)
    
    for frame in range(0, n_frame):
        video[frame, ...] = DrawActor(x, y)
        y += speed_y
        speed_y += accel_y
        if y > 25:
            y = 25
        if y < 3:
            y = 3
    
    return video


# Running
def CreateRunningVideo(n_frame:int):
    
    x = random.randint(5, 7)
    y = random.randint(21, 24)
    video = np.ones((n_frame, 28, 28))
    speed_x = random.randint(3, 6)
    
    for frame in range(0, n_frame):
        video[frame, ...] = DrawActor(x, y)
        x += speed_x
        if x > 25:
            x = 3
    
    return video


# Flying
def CreateFlyingVideo(n_frame:int):
    
    x = random.randint(21, 24)
    y = 7
    video = np.ones((n_frame, 28, 28))
    speed_x = -random.randint(1, 4)
    
    for frame in range(0, n_frame):
        video[frame, ...] = DrawActor(x, y, rotate=True)
        x += speed_x
        if x < 3:
            x = 25
    
    return video
