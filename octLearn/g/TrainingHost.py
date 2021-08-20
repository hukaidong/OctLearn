import os.path
import contextlib

import torch
from torch.utils.data.dataloader import DataLoader

from octLearn.e.config import get_config
from octLearn.h.autoencoder import Autoencoder
from octLearn.h.decipher import Decipher
from octLearn.h.activate_learn import QueryNetwork
from octLearn.g.TrainingUnit import TrainingUnit
from octLearn.g.QueryUnit import QueryUnit


class TrainingHost:

    def __init__(self, configs):
        self.config = configs
        self.autoencoder = None
        self.decipher = None
        self.requester = None

        self._data_loader = None
        self._img_encoder = None
        self._img_decoder = None
        self._parm_decipher = None
        self._policy = None
        self._autoencoder_optimizer = None
        self._decipher_optimizer = None
        self._autoencoder_lr_sched = None
        self._decipher_lr_sched = None
        self._autoencoder_network = None
        self._decipher_network = None
        self._data_image_extractor = None
        self._data_param_extractor = None

    def build_network(self, dataset, *, image_preprocessor, param_preprocessor, image_encoder, image_decoder,
                      param_decipher, autoencoder_policy, weight_initializer=None, autoencoder_optimizer=None,
                      autoencoder_lr_scheduler=None, decipher_optimizer=None, decipher_lr_scheduler=None):

        device = self.config['device']

        data_sample = [d.unsqueeze(0) for d in dataset[0]]

        self._data_image_extractor = image_preprocessor()
        self._data_param_extractor = param_preprocessor()
        imageI, imageO = self._data_image_extractor(data_sample)
        imageI, paramO = self._data_param_extractor(data_sample)

        input_channels = imageI.shape[1]
        output_channels = imageO.shape[1]
        params_size = paramO.shape[1]
        latent_size = self.config['latent_size']

        encoder_cls, encoder_network_base_cls = image_encoder
        decoder_cls, decoder_network_base_cls, distort_network_base_cls = image_decoder
        decipher_cls, *_ = param_decipher
        policy_cls, policy_opts = autoencoder_policy

        encoder_network = encoder_network_base_cls(input_channels, latent_size * 2)
        decoder_network = decoder_network_base_cls(latent_size, output_channels)
        distort_network = distort_network_base_cls(output_channels)
        self._img_encoder = encoder_cls(encoder_network).to(device)
        self._img_decoder = decoder_cls(decoder_network, distort_network).to(device)
        self._parm_decipher = decipher_cls(latent_size, params_size).to(device)
        self._policy = policy_cls(latent_size, **policy_opts).to(device)
        self._autoencoder_network = Autoencoder(self._img_encoder, self._img_decoder, self._policy).to(device)
        self._decipher_network = Decipher(self._img_encoder, self._parm_decipher).to(device)
        self._query_network = QueryNetwork(latent_size, self._parm_decipher).to(device)

        if weight_initializer:
            weight_init_fn, *_ = weight_initializer
            self._img_encoder.apply(weight_init_fn)
            self._img_decoder.apply(weight_init_fn)
            self._parm_decipher.apply(weight_init_fn)

        if autoencoder_optimizer:
            optimizer_cls, optimizer_opts = autoencoder_optimizer
            autoencoder_neural_params = [parm for neu in [self._img_encoder, self._img_decoder] for parm in
                                         neu.parameters()]
            self._autoencoder_optimizer = optimizer_cls(autoencoder_neural_params, **optimizer_opts)

        if decipher_optimizer:
            optimizer_cls, optimizer_opts = decipher_optimizer
            decipher_neural_params = self._parm_decipher.parameters()
            self._decipher_optimizer = optimizer_cls(decipher_neural_params, **optimizer_opts)

        if autoencoder_lr_scheduler:
            lr_sched_cls, lr_sched_opts = autoencoder_lr_scheduler
            self._autoencoder_lr_sched = lr_sched_cls(self._autoencoder_optimizer, **lr_sched_opts)

        if decipher_lr_scheduler:
            lr_sched_cls, lr_sched_opts = decipher_lr_scheduler
            self._decipher_lr_sched = lr_sched_cls(self._decipher_optimizer, **lr_sched_opts)

        def autoencoderDataIter():
            data_loader = DataLoader(dataset, batch_size=self.config['batch_size'],
                                     num_workers=self.config['num_workers'],
                                     collate_fn=self.config.get('collate_fn', None), pin_memory=True)
            while True:
                for x in data_loader:
                    yield self._data_image_extractor(x)

        def decipherDataIter():
            data_loader = DataLoader(dataset, batch_size=self.config['batch_size'],
                                     num_workers=self.config['num_workers'],
                                     collate_fn=self.config.get('collate_fn', None), pin_memory=True)
            while True:
                for x in data_loader:
                    yield self._data_param_extractor(x)

        self.ae_step = 0

        def autoencoderMonitor(summary_writer, test=False):
            if not test:
                prefix = "autoencoder"
                step = self.ae_step
                self.ae_step += 1
            else:
                step = self.ae_step
                prefix = "autoencoder-test"

            if self._autoencoder_lr_sched:
                current_lr = self._autoencoder_lr_sched.get_last_lr()[-1]
                summary_writer.add_scalar(prefix + "/lr", current_lr, step)

            summary_writer.add_scalar("data_len", len(dataset), step)
            states = self._policy.last_states
            for key in ['d_x_xp', 'log_d_x', 'd_xp_xd', 'log_p_z']:
                summary_writer.add_scalar(prefix + "/%s" % key, states[key].mean(), step)
            for i in range(3):
                img_in = states['x'][i]
                img_out = states['xpred'][i]
                imin = torch.min(img_out)
                imax = torch.max(img_out)
                img_norm = (img_out - imin) / (imax - imin)
                summary_writer.add_image(prefix + "/truth-%d" % i, img_in, step)
                summary_writer.add_image(prefix + "/pred-%d" % i, img_out, step)
                summary_writer.add_image(prefix + "/norm-%d" % i, img_norm, step)

        self.de_step = 0

        def decipherMonitor(summary_writer, test=False):
            if not test:
                step = self.de_step
                self.de_step += 1
            else:
                step = self.ae_step
                prefix = "decipher-test"

            if self._decipher_lr_sched:
                current_lr = self._decipher_lr_sched.get_last_lr()[-1]
                summary_writer.add_scalar("decipher/lr", current_lr, step)

            states = self._decipher_network.last_states
            summary_writer.add_scalar("decipher/mean_loss", states['mean_loss'], step)
            summary_writer.add_scalar("data_len", len(dataset), step)
            for i in range(3):
                img_in = states['img_input'][i]
                summary_writer.add_image("decipher/map-%d" % i, img_in[[1, ]], step)
                summary_writer.add_image("decipher/trajectory-%d" % i, img_in[[0, ]], step)
                summary_writer.add_scalar("decipher/indv-loss-mean-%d" % i, states['loss'][i].mean(), step)

            parm_losses = states['loss']
            if parm_losses.shape[1] > 10:
                return
            for i in range(parm_losses.shape[1]):
                summary_writer.add_histogram("decipher/loss-param-%d" % i, states['loss'][:, i], step)

        self._autoencoder_data_iter = autoencoderDataIter()
        self._decipher_data_iter = decipherDataIter()

        self.autoencoder = TrainingUnit(data_iter=self._autoencoder_data_iter, consumer=self._autoencoder_network,
                                        optimizer=self._autoencoder_optimizer, lr_scheduler=self._autoencoder_lr_sched,
                                        monitor=autoencoderMonitor, host=self)

        self.decipher = TrainingUnit(data_iter=self._decipher_data_iter, consumer=self._decipher_network,
                                     optimizer=self._decipher_optimizer, lr_scheduler=self._decipher_lr_sched,
                                     monitor=decipherMonitor, host=self)

        self.requester = QueryUnit(self._query_network)

    def load(self, load_mask=None, _format="%s.torchfile"):
        infile_path = get_config()['misc']['infile_path']
        infile_format = os.path.join(infile_path, _format)
        load_mask = load_mask or [1, 1, 1]
        device = self.config['device']

        if load_mask[0]:
            self._img_encoder.load_state_dict(
                torch.load(infile_format % "img-encoder", map_location=device))
        if load_mask[1]:
            self._img_decoder.load_state_dict(
                torch.load(infile_format % "img-decoder", map_location=device))
        if load_mask[2]:
            self._parm_decipher.load_state_dict(
                torch.load(infile_format % "parm-decipher", map_location=device))

    def dump(self, _format="%s.torchfile", dump_mask=None):
        outfile_path = get_config()['misc']['outfile_path']
        outfile_format = os.path.join(outfile_path, _format)
        dump_mask = dump_mask or [1, 1, 1]

        if dump_mask[0]:
            torch.save(self._img_encoder.state_dict(), outfile_format % "img-encoder")
        if dump_mask[1]:
            torch.save(self._img_decoder.state_dict(), outfile_format % "img-decoder")
        if dump_mask[2]:
            torch.save(self._parm_decipher.state_dict(), outfile_format % "parm-decipher")

    @contextlib.contextmanager
    def extern_dataset(self, new_dataset, /, batch_size=None):
        try:
            self._refresh_dataset(new_dataset, batch_size=batch_size)
            yield
        finally:
            self._refresh_dataset()

    def _refresh_dataset(self, dataset=None, /, batch_size=None):
        if batch_size is None:
            batch_size = self.config['batch_size']
        # Use for feeding extra dataset for cross-validation or test
        # if refresh_dataset is called with no argument, resume working on training dataset
        if dataset is None:
            self.autoencoder.set_data_iter(self._autoencoder_data_iter)
            self.decipher.set_data_iter(self._decipher_data_iter)
            return

        def autoencoderDataIter():
            data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=self.config['num_workers'],
                                     collate_fn=self.config.get('collate_fn', None), pin_memory=True)
            while True:
                for x in data_loader:
                    yield self._data_image_extractor(x)

        def decipherDataIter():
            data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=self.config['num_workers'],
                                     collate_fn=self.config.get('collate_fn', None), pin_memory=True)
            while True:
                for x in data_loader:
                    yield self._data_param_extractor(x)

        self.autoencoder.set_data_iter(autoencoderDataIter())
        self.decipher.set_data_iter(decipherDataIter())
