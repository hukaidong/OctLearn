"""
basicStuff = lambda : None

TrainingManager T
"""

class TrainingManager:
    pass

# noinspection PyUnresolvedReferences
def Usage():
    initWeight()
    T: TrainingManager

    T.load(load_target)
    T.initialize_others()

    for sample_result in T.loopTrain():
        tensorboard.writeSummary(sampledResult)
        if shouldStop(sample_result):
            break

    T.save(save_target)



