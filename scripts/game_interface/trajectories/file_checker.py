import numpy as np
trajectories = np.load(r'C:\Users\odezz\source\MinecraftAI2\scripts\game_interface\trajectories\trajectories_1731550392.npy', allow_pickle=True)
print(len(trajectories))  # Should print 30 if there are 30 agents recorded