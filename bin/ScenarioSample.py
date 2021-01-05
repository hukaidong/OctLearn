import numpy as np
from matplotlib import pyplot as plt

from octLearn.connector.dbRecords import MongoInstance
from octLearn.scenariomanage.ScenarioTypes import ScenarioType3

case_id = '5f85acee767dae76c6c9bf14'
coll = MongoInstance('learning', 'complete')
doc = coll.Case_By_id(case_id)
assert doc
scenario = ScenarioType3(doc, r"C:\Users\Kaidong Hu\Desktop\5f8")

grid_shape = (20, 20)
grid_shift = (-10, -10)


def GetCubeMap():
    return scenario.world


def GetNumAgents():
    return scenario.num_agent


def GridAlignCenter(array, shift):
    shape = array.shape
    paddingt = int(np.floor_divide(shape[0], 2))
    origEdgeB = int(paddingt + shape[0])
    targEdgeT = int(paddingt - shift[0])
    targEdgeB = int(targEdgeT + shape[0])

    paddingl = int(np.floor_divide(shape[1], 2))
    origEdgeR = int(paddingl + shape[1])
    targEdgeL = int(paddingl - shift[1])
    targEdgeR = int(targEdgeL + shape[1])

    ywidth = paddingt * 2 + shape[1]
    xwidth = paddingl * 2 + shape[0]
    newarray = np.zeros((ywidth, xwidth))
    newarray[paddingt: origEdgeB, paddingl: origEdgeR] = array
    targArray = newarray[targEdgeT: targEdgeB, targEdgeL: targEdgeR]
    assert np.array_equal(shape, targArray.shape)
    return targArray.copy()


def GetAgentShift(aid):
    start = scenario.agent_start[aid]
    end = scenario.agent_target[aid]
    center = np.floor_divide(start + end, 2)
    return -center


def WorldLocationToGridIdx(loc):
    world_topleft = np.array((-10, -10))
    loc_world = (loc - world_topleft).astype(int)
    return loc_world


def GetAgentTaskPlot(aid):
    start = scenario.agent_start[aid]
    end = scenario.agent_target[aid]
    wstart = WorldLocationToGridIdx(start)
    wend = WorldLocationToGridIdx(end)
    array = np.zeros(grid_shape)
    array[wstart[0], wstart[1]] = 1
    array[wend[0], wend[1]] = 1
    return array


def GetAgentTaskPlotCentered(aid):
    array = GetAgentTaskPlot(aid)
    shift = GetAgentShift(aid)
    return GridAlignCenter(array, shift)


def GetAgentTask(aid):
    start = scenario.agent_start[aid]
    end = scenario.agent_target[aid]
    wstart = WorldLocationToGridIdx(start)
    wend = WorldLocationToGridIdx(end)
    array = np.zeros([2, grid_shape[0], grid_shape[1]])
    array[:, wstart[0], wstart[1]] = (wend-wstart)/20
    return array


def GetAgentCubeMap(aid):
    shift = GetAgentShift(aid)
    return GridAlignCenter(scenario.world, shift)


def GetAgentCenteredTrajectory(aid):
    shift = GetAgentShift(aid)
    array = np.zeros([grid_shape[0]*5, grid_shape[1]*5])
    scenario.FillByAgentTraj(array, aid, 0.2)
    return GridAlignCenter(array.T, shift * 5)


if __name__ == '__main__':
    import sys
    img = scenario.GetAgentTrajVision()
    print(img.sum())
    print(img[0, 0].shape)
    for i in range(scenario.num_agent):
        plt.imshow(img[i, 0], 'gray_r', origin='lower')
        print(i)
        print(np.sum(img[i, 0]**2))
        plt.title(str(i))
        plt.show()

    plt.pause(100)
    sys.exit(0)

    agentnum = 4
    a = GetCubeMap()
    plt.imshow(a, 'gray_r', origin='lower')
    plt.show()
    np.save("cubemap", a)

    a = GetAgentCubeMap(agentnum)
    plt.imshow(a, 'gray_r', origin='lower')
    plt.show()
    np.save("agentcubemap", a)

    a = GetAgentTaskPlot(agentnum)
    plt.imshow(a, 'gray_r', origin='lower')
    plt.show()
    np.save("agenttaskplot", a)

    a = GetAgentTask(agentnum)
    np.save("agenttask", a)

    a = GetAgentTaskPlotCentered(agentnum)
    plt.imshow(a, 'gray_r', origin='lower')
    plt.show()
    np.save("taskcentered", a)

    a = GetAgentCenteredTrajectory(agentnum)
    plt.imshow(a, 'gray_r', origin='lower')
    plt.show()
    np.save("trajcentered", a)

    plt.pause(0)
