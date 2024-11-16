import numpy as np

class SimulatedAgent:
    def __init__(self, start_x=0.0, start_z=0.0, start_yaw=0.0, step_size=0.8, strafe_step_size=0.8, turn_angle=10):
        self.x = start_x
        self.z = start_z
        self.yaw = start_yaw  # in degrees, where 0 is forward, -90 is left, 90 is right
        self.step_size = step_size  # distance moved per forward/backward step
        self.strafe_step_size = strafe_step_size  # distance moved per strafe step
        self.turn_angle = turn_angle  # degrees turned per left/right turn
    
    def reset(self, x=0.0, z=0.0, yaw=0.0):
        """Reset the agent's position and orientation."""
        self.x = x
        self.z = z
        self.yaw = yaw

    def _apply_noise(self, base_value, noise_level):
        """Apply movement noise to the base value according to noise level."""
        if np.random.rand() > noise_level:
            return base_value  # Normal movement with no adjustment
        else:
            return base_value * (1 + np.random.normal(0, noise_level))

    def move_forward(self, noise_level=0.0):
        """Move the agent forward with optional noise."""
        adjusted_step_size = self._apply_noise(self.step_size, noise_level)
        radians = np.radians(self.yaw)
        self.x += adjusted_step_size * np.cos(radians)
        self.z += adjusted_step_size * np.sin(radians)

    def move_backward(self, noise_level=0.0):
        """Move the agent backward with optional noise."""
        adjusted_step_size = self._apply_noise(self.step_size, noise_level)
        radians = np.radians(self.yaw)
        self.x -= adjusted_step_size * np.cos(radians)
        self.z -= adjusted_step_size * np.sin(radians)

    def strafe_left(self, noise_level=0.0):
        """Move the agent to the left (strafe) with optional noise."""
        adjusted_strafe_step = self._apply_noise(self.strafe_step_size, noise_level)
        radians = np.radians(self.yaw - 90)  # offset yaw by 90 degrees for strafing left
        self.x += adjusted_strafe_step * np.cos(radians)
        self.z += adjusted_strafe_step * np.sin(radians)

    def strafe_right(self, noise_level=0.0):
        """Move the agent to the right (strafe) with optional noise."""
        adjusted_strafe_step = self._apply_noise(self.strafe_step_size, noise_level)
        radians = np.radians(self.yaw + 90)  # offset yaw by 90 degrees for strafing right
        self.x += adjusted_strafe_step * np.cos(radians)
        self.z += adjusted_strafe_step * np.sin(radians)

    def turn_left(self, noise_level=0.0):
        """Turn the agent to the left with optional noise."""
        adjusted_turn_angle = self._apply_noise(self.turn_angle, noise_level)
        self.yaw = (self.yaw - adjusted_turn_angle) % 360

    def turn_right(self, noise_level=0.0):
        """Turn the agent to the right with optional noise."""
        adjusted_turn_angle = self._apply_noise(self.turn_angle, noise_level)
        self.yaw = (self.yaw + adjusted_turn_angle) % 360

    def get_position(self):
        """Get the current position and yaw of the agent."""
        return self.x, self.z, self.yaw
