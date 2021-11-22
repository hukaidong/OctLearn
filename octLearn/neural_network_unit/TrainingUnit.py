import torch
import logging

logger = logging.getLogger(__name__)


def rangeForever():
    num = 0
    while True:
        yield num
        num += 1


class TrainingUnit:
    def __init__(self, preprocessor, consumer, optimizer, lr_scheduler, monitor, host):
        self._preprocessor = preprocessor
        self._consumer = consumer
        self._optimizer = optimizer
        self._monitor = monitor
        self._lr_scheduler = lr_scheduler
        self._step_max = lambda: int(host.config['step_per_epoch'])
        self._device = host.config['device']

    def loop_train(self, data_loader, summary_writer=None):
        stepTrain = self.step_train(data_loader)
        loss = 0
        for _ in rangeForever():
            self._consumer.train()
            logger.debug("Loop training begin")
            for i in range(self._step_max()):
                loss = next(stepTrain)
            logger.debug("Loop training ends")
            if self._lr_scheduler:
                self._lr_scheduler.step()
            if summary_writer:
                self._monitor(summary_writer)
            yield loss

    def step_train(self, data_loader):
        while True:
            full_data_batch = 0
            for data in data_loader.get_data_iter():
                logger.debug("Start a train step")
                tensorIn, tensorOut = self._preprocessor(data)
                tensorIn = tensorIn.to(self._device)
                tensorOut = tensorOut.to(self._device)
                self._optimizer.zero_grad()
                loss = self._consumer.compute_loss(tensorIn, tensorOut)
                loss.backward()
                self._optimizer.step()
                logger.debug("Completed a train step")
                yield loss
                full_data_batch += 1
            logger.info(f"Dataset reaches end, restart. {full_data_batch} batches were trained in this round.")

    def score(self, data_loader, summary_writer=None):
        while True:
            for data in data_loader.get_data_iter():
                logger.debug("Start a score step")
                with torch.no_grad():
                    tensorIn, tensorOut = self._preprocessor(data)
                    tensorIn = tensorIn.to(self._device)
                    tensorOut = tensorOut.to(self._device)
                    loss = self._consumer.compute_loss(tensorIn, tensorOut)
                if summary_writer:
                    self._monitor(summary_writer, test=True)
                logger.debug(f"Completed a score step")
                yield loss
            logger.info(f"Scoring Dataset reaches end, restart.")

    def forward(self, data_input):
        with torch.no_grad():
            tensorIn = data_input.to(self._device)
            tensorOut = self._consumer(tensorIn)
        return tensorOut
