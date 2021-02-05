def rangeForever():
    num = 0
    while True:
        yield num
        num += 1


class TrainingUnit:
    def __init__(self, data_iter, consumer, optimizer, lr_scheduler, monitor, host):
        self._data_iter = data_iter
        self._consumer = consumer
        self._optimizer = optimizer
        self._monitor = monitor
        self._lr_scheduler = lr_scheduler
        self._step_max = lambda: host.config['step_per_epoch']
        self._device = host.config['device']

    def set_data_iter(self, data_iter):
        self._data_iter = data_iter

    def loopTrain(self, summary_writer=None):
        loss = 0
        stepTrain = self.stepTrain()
        for epoch_num in rangeForever():
            # self._consumer.train()
            for i in range(self._step_max()):
                loss = next(stepTrain)

            if self._lr_scheduler:
                self._lr_scheduler.step()
            if summary_writer:
                self._monitor(summary_writer, epoch_num)
            # self._consumer.eval()
            yield loss

    def stepTrain(self, alt_training=False):
        for tensorIn, tensorOut in self._data_iter:
            if alt_training:
                self._consumer.train()
            tensorIn = tensorIn.to(self._device)
            tensorOut = tensorOut.to(self._device)
            self._optimizer.zero_grad()
            loss = self._consumer.compute_loss(tensorIn, tensorOut)
            loss.backward()
            self._optimizer.step()
            if alt_training:
                self._consumer.eval()
            yield loss

    def score(self):
        for tensorIn, tensorOut in self._data_iter:
            self._consumer.eval()
            tensorIn = tensorIn.to(self._device)
            tensorOut = tensorOut.to(self._device)
            loss = self._consumer.compute_loss(tensorIn, tensorOut)
            return loss
