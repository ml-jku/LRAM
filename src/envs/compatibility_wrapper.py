import gym
import gymnasium
from gym import spaces


class GymCompatibilityWrapper(gym.Wrapper): 
    
    def __init__(self, env) -> None:
        """
        Minimum compatibility for gymnasium to gym envs.
        Handles the conversions of obs/act spaces + step/reset calls.  
        
        Necessary, such that we can use robosuite==1.4.1 and mimicgen, while 
        keeping composuite/mimicgen the same. 
        
        Args: 
          env: gym.Env.
        """ 
        super().__init__(env)
        self.observation_space = self._convert_space(self.observation_space)
        self.action_space = self._convert_space(self.action_space)

    def _convert_space(self, space):
        """
        Converts gymnasium spaces to gym spaces.
        """
        if isinstance(space, gymnasium.spaces.Discrete):
            return spaces.Discrete(space.n)
        elif isinstance(space, spaces.Discrete):
            # correct already
            return space
        elif isinstance(space, gymnasium.spaces.Box):
            return spaces.Box(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)
        elif isinstance(space, spaces.Box):
            # correct already
            return space
        else:
            raise NotImplementedError("This space type is not supported yet.")
    
    def seed(self, seed=None):
        pass
    
    def step(self, action): 
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info
    
    def reset(self):
        obs, _ = self.env.reset()
        return obs
