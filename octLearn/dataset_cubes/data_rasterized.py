import numpy as np

from .data_modify_by_05_shifts import ShiftedData
from .graphic import translate_and_crop, position_to_array_index, RectangleRepXY


class RasterizeData:
    def __init__(self, document, resolution=5):
        self.data = ShiftedData()
        self.resolution = resolution
        self.data.raw.load_document(document)
        self.data.raw.prepare_all()
        self.num_agent = self.data.raw.num_agent

    def get_agent_task_central(self, agentId):
        init_loc = self.data.get_agent_initial_location()[agentId]
        final_loc = self.data.get_agent_target_location()[agentId]
        return (init_loc + final_loc) / 2

    def compact_obstacle_map(self):
        images = [
            self.get_obstacle_map(i)
            for i in range(self.num_agent)
        ]
        return np.stack(images)

    def get_obstacle_map(self, agentId):
        obstacleBinary = self.data.get_obstacle_map()
        shift = -self.get_agent_task_central(agentId)
        image = translate_and_crop(obstacleBinary, np.fix(shift))
        return np.expand_dims(image, axis=0)  # add channel dimension

    def compact_trajectory_map(self):
        images = [
            self.get_trajectory_map(i)
            for i in range(self.num_agent)
        ]
        return np.stack(images)

    def get_trajectory_map(self, agentId):
        self.base_shape = self.data.get_obstacle_map().shape
        MapSize = RectangleRepXY(20, 20)
        GridManhttanSize = MapSize * self.resolution
        GridBleedingSize = GridManhttanSize * 2
        shift = -self.get_agent_task_central(agentId)
        # manhattan coordinate, scaled by Resolution
        trajSeq_Cworld = self.data.get_trajectories()[agentId]  # shape: (num_frames, 2)
        trajSeq_Cimage = trajSeq_Cworld + [MapSize.x, MapSize.y]  # MapSize * 2 (padding) / 2 (centered)
        trajSeq_Cmanht = np.round(trajSeq_Cimage * self.resolution).astype(np.int)

        trajIdx_Cmanht = trajSeq_Cmanht[:, 1] * GridBleedingSize.x + trajSeq_Cmanht[:, 0]
        trajUIdx = np.unique(trajIdx_Cmanht, axis=0)  # reduce computation time

        trajVis = np.zeros([*GridBleedingSize])
        np.put(trajVis, trajUIdx, 1, mode='wrap')  # override all trajectory space with 1

        image = translate_and_crop(trajVis, np.fix(shift) * self.resolution, target_size=GridManhttanSize)
        return np.expand_dims(image, axis=0)  # add channel dimension


    def compact_task_map(self):
        images = [
            self.get_task_map(i)
            for i in range(self.num_agent)
        ]
        return np.stack(images)

    def get_task_map(self, agentId):
        base_shape = self.data.get_obstacle_map().shape

        task_map = np.zeros([2] + list(base_shape))
        init_loc = self.data.get_agent_initial_location()[agentId]
        final_loc = self.data.get_agent_target_location()[agentId]
        center_loc = (init_loc + final_loc) / 2

        index = position_to_array_index(init_loc, array_shape=base_shape, center_point=center_loc)
        task_relative = (final_loc - init_loc) / base_shape
        task_map[:, index[1], index[0]] = task_relative
        return task_map
