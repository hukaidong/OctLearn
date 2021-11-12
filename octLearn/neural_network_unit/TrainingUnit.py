import torch

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
        self._step_max = lambda: int(host.config['step_per_epoch'])
        self._device = host.config['device']

    def set_data_iter(self, data_iter):
        self._data_iter = data_iter

    def get_data_iter(self, data_iter):
        return self._data_iter

    def loopTrain(self, summary_writer=None):
        def loop_generator():
            stepTrain = self.stepTrain()
            yield
            loss = 0
            for _ in rangeForever():
                self._consumer.train()
                for i in range(self._step_max()):
                    loss = next(stepTrain)

                if self._lr_scheduler:
                    self._lr_scheduler.step()
                if summary_writer:
                    self._monitor(summary_writer)
                yield loss

        gen = loop_generator()
        next(gen)
        return gen

    def stepTrain(self, alt_training=False):
        def step_generator():
            data_iter = self._data_iter  # freeze dataset use
            yield

            for tensorIn, tensorOut in data_iter:
                tensorIn = tensorIn.to(self._device)
                tensorOut = tensorOut.to(self._device)
                self._optimizer.zero_grad()
                loss = self._consumer.compute_loss(tensorIn, tensorOut)
                loss.backward()
                self._optimizer.step()
                yield loss
        
        gen = step_generator()
        next(gen)
        return gen

    def score(self, summary_writer=None):
        def step_generator():
            data_iter = self._data_iter  # freeze dataset use
            yield

            for tensorIn, tensorOut in data_iter:
                with torch.no_grad():
                    tensorIn = tensorIn.to(self._device)
                    tensorOut = tensorOut.to(self._device)
                    loss = self._consumer.compute_loss(tensorIn, tensorOut)
                if summary_writer:
                    self._monitor(summary_writer, test=True)
                yield loss

        gen = step_generator()
        next(gen)
        return gen

    def forward(self, data_input):
        with torch.no_grad():
            tensorIn = data_input.to(self._device)
            tensorOut = self._consumer(tensorIn)
        return tensorOut
