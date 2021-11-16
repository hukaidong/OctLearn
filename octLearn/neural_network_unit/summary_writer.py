from torch.utils import tensorboard



class SummaryWriter:
    def __init__(self):
        self._writer = tensorboard.SummaryWriter()
        self.global_step = 0

    def add_scalar(self, *args, **kwargs):
        kwargs["global_step"] = self.global_step
        self._writer.add_scalar(*args, **kwargs)

    def add_scalars(self, *args, **kwargs):
        kwargs["global_step"] = self.global_step
        self._writer.add_scalars(*args, **kwargs)

    def add_image(self, *args, **kwargs):
        kwargs["global_step"] = self.global_step
        self._writer.add_image(*args, **kwargs)
