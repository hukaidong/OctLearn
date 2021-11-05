import pickle
import unittest
from pathlib import Path

from matplotlib import pyplot as plt
from octLearn.mongo_stuff.TrajectoryEncodes import *

SAMPLEOBJID = '5fc0b4ae4478ded555ed390b'
DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'cases')
CASE_FOLDER = os.path.join(DATA_FOLDER, SAMPLEOBJID[-2:], SAMPLEOBJID)

ENV['TrajRoot'] = DATA_FOLDER
ENV['FeatRoot'] = DATA_FOLDER


def command_LoadDoc():
    with open(Path(DATA_FOLDER) / 'mongodoc.pkl', 'rb') as f:
        doc = pickle.load(f)
    return doc[0]


def command_LoadTrajectory(agentId):
    t, f = readTrajectory(SAMPLEOBJID)
    return t[agentId]


def command_Trajectory2Feature(agentId):
    doc = command_LoadDoc()
    d = Trajectory2Feature(doc, save_result=False)
    return {k: d[k][agentId] for k in d}


class MyTestCase(unittest.TestCase):
    def test_feature_generate(self):
        t = command_LoadTrajectory(1)
        f = command_Trajectory2Feature(1)
        self.assertCountEqual(f.keys(), ['aid', 'agtparm', 'cubevis', 'taskvis', 'trajvis'])

    def test_get_traj_vision(self):
        doc = command_LoadDoc()
        traj, _ = readTrajectory(SAMPLEOBJID)
        scenario = ScenarioType3(doc)
        fig, ax = plt.subplots()
        GetAgentTrajVision(scenario, traj, debug_subplots=(fig, ax))
        plt.show()


    @unittest.skip('Used for debugging stuff')
    def test_abc(self):
        print(command_LoadDoc())

    @unittest.skip('Used for debugging stuff')
    def test_plot(self):
        import matplotlib.pyplot as plt
        f = command_Trajectory2Feature(1)
        plt.imshow(f['trajvis'][0])
        plt.pause(0.1)

    @unittest.skip('Used for debugging stuff')
    def test_taskvis(self):
        pass


if __name__ == '__main__':
    unittest.main()
