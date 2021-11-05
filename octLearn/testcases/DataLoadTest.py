import os
import pickle
import unittest

import numpy as np
from matplotlib import pyplot as plt

from octLearn.dataset_cubes.data_from_sources import RawData
from octLearn.dataset_cubes.data_rasterized import RasterizeData


# noinspection PyMissingConstructor
class MockRawData(RawData):
    def __init__(self, *args, **kwargs):
        self._init_variables_()

    def get_scene_parameters(self):
        pass

    def get_agent_parameters(self):
        pass

    def get_trajectory_and_forwards(self):
        pass

    def prepare_all_centered(self):
        self._init_variables_()
        self.trajectories = np.array([0.0, 0.0]).reshape((1, 1, 2))
        self.agent_init_location = np.array([[0.0, 0.0]])
        self.agent_goal_location = np.array([[0.0, 0.0]])
        self.obstacle_map = np.zeros((20, 20))
        self.obstacle_map[10, 10] = 1

    def prepare_all_negative(self):
        self._init_variables_()
        self.trajectories = np.array([-10, -10]).reshape((1, 1, 2))
        self.agent_init_location = np.array([[-10, -10]])
        self.agent_goal_location = np.array([[-10, -10]])
        self.obstacle_map = np.zeros((20, 20))
        self.obstacle_map[0, 0] = 1

    def prepare_all_positive(self):
        self._init_variables_()
        self.trajectories = np.array([9, 9]).reshape((1, 1, 2))
        self.agent_init_location = np.array([[9, 9]])
        self.agent_goal_location = np.array([[9, 9]])
        self.obstacle_map = np.zeros((20, 20))
        self.obstacle_map[19, 19] = 1


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        testcase_dir = os.path.abspath(os.path.join(__file__, '..', 'Cases'))
        with open(os.path.join(testcase_dir, 'mongodoc.pkl'), 'rb') as file:
            self.doc = pickle.load(file)[0]

        os.environ['TrajRoot'] = testcase_dir
        self.case_id = '5fc0b4ae4478ded555ed390b'
        self.plot_options = {'cmap': 'gray_r', 'vmin': 0, 'vmax': 1, 'origin': 'lower',
                             'extent': (-0.5, 19.5, -0.5, 19.5)}
        self.plot_traj_options = {'cmap': 'gray_r', 'vmin': 0, 'vmax': 1, 'origin': 'lower',
                                  'extent': (0.0, 20.0, 0.0, 20.0)}

    def test_something(self):
        fig, ax = plt.subplots(2, 2)

        data = RasterizeData(resolution=5, document=self.doc)
        for i in range(50):
            ax[0][1].clear()
            ax[0][0].imshow(data.get_obstacle_map(i), **self.plot_options)
            ax[1][0].imshow(data.get_task_map(i)[..., 0], **self.plot_options)
            ax[1][1].imshow(data.get_task_map(i)[..., 1], **self.plot_options)
            ax[0][1].imshow(data.get_obstacle_map(i), alpha=0.5, **self.plot_options)
            ax[0][1].imshow(data.get_trajectory_map(i), alpha=0.5, **self.plot_options)
            fig.show()
            plt.pause(0.1)

    def test_something_else(self):
        fig, ax = plt.subplots(2, 2)
        data = RasterizeData(resolution=5, document=self.doc)
        data.data.raw = MockRawData()
        data.data.raw.prepare_all_negative()
        self.method_name(ax, data, fig)

    def test_something_other_else(self):
        fig, ax = plt.subplots(2, 2)
        data = RasterizeData(resolution=5, document=self.doc)
        data.data.raw = MockRawData()
        data.data.raw.prepare_all_centered()
        self.method_name(ax, data, fig)

    def method_name(self, ax, data, fig):
        for i in range(1):
            ax[0][1].clear()
            ax[0][0].imshow(data.get_obstacle_map(i), **self.plot_options)
            ax[1][0].imshow(data.get_task_map(i)[..., 0], **self.plot_options)
            ax[1][1].imshow(data.get_task_map(i)[..., 1], **self.plot_options)
            ax[0][1].imshow(data.get_obstacle_map(i), alpha=0.5, **self.plot_options)
            ax[0][1].imshow(data.get_trajectory_map(i), alpha=0.5, **self.plot_traj_options)
            fig.show()
            plt.pause(0.1)

    def test_something_other_than_other_else(self):
        fig, ax = plt.subplots(2, 2)
        data = RasterizeData(resolution=5, document=self.doc)
        data.data.raw = MockRawData()
        data.data.raw.prepare_all_positive()
        self.method_name(ax, data, fig)


if __name__ == '__main__':
    unittest.main()
