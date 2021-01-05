import torch
from torch.utils.data.dataloader import DataLoader

from octLearn.autoencoder.autoencoder import Autoencoder, Decipher
from octLearn.g.TrainingUnit import TrainingUnit


class TrainingHost:

    def __init__(self, configs):
        self.config = configs
        self.autoencoder = None
        self.decipher = None

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

    def build_network(self, dataset, *, image_preprocessor, param_preprocessor, image_encoder, image_decoder,
                      param_decipher, autoencoder_policy, weight_initializer=None, autoencoder_optimizer=None,
                      autoencoder_lr_scheduler=None, decipher_optimizer=None, decipher_lr_scheduler=None ):

        device = self.config['device']

        data_sample = [d.unsqueeze(0) for d in dataset[0]]

        data_image_extractor = image_preprocessor()
        data_param_extractor = param_preprocessor()
        imageI, imageO = data_image_extractor(data_sample)
        imageI, paramO = data_param_extractor(data_sample)

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

        if decipher_optimizer:
            lr_sched_cls, lr_sched_opts = decipher_lr_scheduler
            self._autoencoder_lr_sched = lr_sched_cls(self._autoencoder_optimizer, **lr_sched_opts)
            self._decipher_lr_sched = lr_sched_cls(self._decipher_optimizer, **lr_sched_opts)

        def autoencoderDataIter():
            data_loader = DataLoader(dataset, batch_size=self.config['batch_size'],
                                     num_workers=self.config['num_workers'],  collate_fn=self.config['collate_fn'],
                                     pin_memory=True)
            while True:
                for x in data_loader:
                    yield data_image_extractor(x)

        def decipherDataIter():
            data_loader = DataLoader(dataset, batch_size=self.config['batch_size'],
                                     num_workers=self.config['num_workers'],  collate_fn=self.config['collate_fn'],
                                     pin_memory=True)
            while True:
                for x in data_loader:
                    yield data_param_extractor(x)

        def autoencoderMonitor(summary_writer, step):
            current_lr = self._autoencoder_lr_sched.get_last_lr()[-1]
            summary_writer.add_scalar("autoencoder/lr", current_lr, step)
            states = self._policy.last_states
            for key in ['d_x_xp', 'log_d_x', 'd_xp_xd', 'log_p_z', 'loss']:
                summary_writer.add_scalar("autoencoder/%s" % key, states[key].mean(), step)
            for i in range(3):
                img_in = states['x'][i]
                img_out = states['xpred'][i]
                imin = torch.min(img_out)
                imax = torch.max(img_out)
                img_norm = (img_out - imin) / (imax - imin)
                summary_writer.add_image("autoencoder/truth-%d" % i, img_in, step)
                summary_writer.add_image("autoencoder/pred-%d" % i, img_out, step)
                summary_writer.add_image("autoencoder/norm-%d" % i, img_norm, step)

        def decipherMonitor(summary_writer, step):
            current_lr = self._decipher_lr_sched.get_last_lr()[-1]
            states = self._decipher_network.last_states
            summary_writer.add_scalar("decipher/lr", current_lr, step)
            summary_writer.add_scalar("decipher/mean_loss", states['mean_loss'], step)
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

        self.autoencoder = TrainingUnit(data_iter=autoencoderDataIter(), consumer=self._autoencoder_network,
                                        optimizer=self._autoencoder_optimizer, lr_scheduler=self._autoencoder_lr_sched,
                                        monitor=autoencoderMonitor, host=self)

        self.decipher = TrainingUnit(data_iter=decipherDataIter(), consumer=self._decipher_network,
                                     optimizer=self._decipher_optimizer, lr_scheduler=self._decipher_lr_sched,
                                     monitor=decipherMonitor, host=self)

    def load(self, _format="%s.torchfile", load_mask=(1, 1, 1)):
        if load_mask[0]:
            self._img_encoder.load_state_dict(torch.load(_format % "img-encoder"))
        if load_mask[1]:
            self._img_decoder.load_state_dict(torch.load(_format % "img-decoder"))
        if load_mask[2]:
            self._parm_decipher.load_state_dict(torch.load(_format % "parm-decipher"))

    def dump(self, _format="%s.torchfile", dump_mask=(1, 1, 1)):
        if dump_mask[0]:
            torch.save(self._img_encoder.state_dict(), _format % "img-encoder")
        if dump_mask[1]:
            torch.save(self._img_decoder.state_dict(), _format % "img-decoder")
        if dump_mask[2]:
            torch.save(self._parm_decipher.state_dict(), _format % "parm-decipher")
