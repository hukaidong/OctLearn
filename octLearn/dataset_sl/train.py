import logging
import torch

logger = logging.getLogger(__name__)


def main():
    from octLearn.dataset_cg.torch_dataset import HopDataset, HopTestDataset
    from octLearn.dataset_cg.torch_batch_sample import CgBatchSampler
    from octLearn.neural_network_unit.TrainingHost import TrainingHost
    from octLearn.neural_network_unit.summary_writer import SummaryWriter







def initial_sample_steersim():
    import os
    import shutil
    import numpy as np
    from octLearn.dataset_cg.steersim_quest import steersim_call_parallel

    # ask_for_regenerate = input("Remove steersim record path and regen?")
    # if ask_for_regenerate == "n":
    #     return
    shutil.rmtree(os.environ["SteersimRecordPath"], ignore_errors=True)
    os.makedirs(os.environ["SteersimRecordPath"], exist_ok=True)
    numbers = np.random.uniform(0, 1, (5, 43))
    steersim_call_parallel(numbers)
    numbers = np.random.uniform(0, 1, (1, 43))
    steersim_call_parallel(numbers, generate_for_testcases=True)


if __name__ == "__main__":
    logging_format = "%(asctime)s - %(name)s - %(levelname)s - %(process)d: %(message)s"
    logging.basicConfig(format=logging_format, level=logging.WARNING)
    initial_sample_steersim()
    main()
