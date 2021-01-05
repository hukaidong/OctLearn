import numpy as np

from .data_from_sources import RawData


class ShiftedData:
    def __init__(self):
        self.raw = RawData()
        self.coord_shift = np.array([0.5, 0.5])

    # require raw.get_trajectories_and_forwards loaded
    def get_trajectories(self):
        return [x + self.coord_shift for x in self.raw.trajectories]

    def get_obstacle_map(self):
        return self.raw.obstacle_map  # No changes

    def get_agent_initial_location(self):
        return self.raw.agent_init_location + self.coord_shift

    def get_agent_target_location(self):
        return self.raw.agent_goal_location + self.coord_shift

    def get_agent_parameters(self):
        return self.raw.agent_parameters  # No changes
