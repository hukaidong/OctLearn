import numpy as np



def read_trajectory_binary(filename):
    with open(filename, "rb") as file:
        eof = file.seek(0, 2)
        file.seek(0, 0)

        obsTypeSize = np.fromfile(file, np.int32, 1)[0]
        obsType = np.fromfile(file, dtype=np.int32, count=obsTypeSize)
        obsType = np.array(obsType)

        obsInfoSize = np.fromfile(file, np.int32, 1)[0]
        obsInfo = np.fromfile(file, dtype=np.float32, count=obsInfoSize)
        obsInfo = np.array(obsInfo)

        parameter_size = np.fromfile(file, np.int32, 1)[0]
        parameters = np.fromfile(file, dtype=np.float32, count=parameter_size)
        parameters = np.array(parameters)

        agent_array = []
        while file.tell() < eof:
            trajectoryLength = np.fromfile(file, np.int32, 1)[0]
            trajectory_matrix = np.fromfile(file, np.float32, trajectoryLength).reshape((-1, 2))
            agent_array.append(trajectory_matrix)


    return (obsType, obsInfo, parameters, agent_array)


env1_rect = {"xmin": -60, "xmax": 60, "ymin": -90, "ymax": 90}
env1_rect = {"xmin": -70, "xmax": 70, "ymin": -100, "ymax": 100}
if __name__ == "__main__":
    import os
    from octLearn.dataset_cg.trajectory_to_image import trajectory_to_image
    from octLearn.dataset_cg.trajectory_to_image import task_to_image
    from octLearn.dataset_cg.obstacle_to_image import obstacle_to_image_slow

    filename = os.environ["trajectory_sample"]
    obt, obi, param, traj = read_trajectory_binary(filename)

    """ # Get min/max index in trajectory
    float_formatter = "{:.2f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})


    print("trajectory range:")
    print(np.min(
        list(map(lambda x: np.min(x, axis=0), traj)), 0
    ))
    print(np.max(
        list(map(lambda x: np.max(x, axis=0), traj)), 0
    ))
    """

    from matplotlib import pyplot as plt

    plt.rcParams['figure.figsize'] = (10, 7)
    images = np.array([trajectory_to_image(t, env1_rect, 2) for t in traj])
    images_max = images.max(0)
    ax = plt.imshow(images_max.T)
    plt.title(str(images.shape))
    plt.show()

    images = np.array([task_to_image(t, env1_rect, 2) for t in traj])
    images_max = np.amax(np.abs(images), axis=(0, 1))
    ax = plt.imshow(images_max.T)
    plt.title(str(images.shape))
    plt.show()

    obs_image = obstacle_to_image_slow(obt, obi, env1_rect, 2)
    plt.clf()
    plt.imshow(obs_image.T)
    plt.show()

    plt.clf()
    plt.imshow(obs_image.T + images_max.T)
    plt.show()

