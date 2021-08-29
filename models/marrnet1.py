from os import makedirs
from os.path import join
import numpy as np
import torch
import torch.nn as nn
from networks.networks import ViewAsLinear
from networks.uresnet import Net as Uresnet
from .marrnetbase import MarrnetBaseModel


class Model(MarrnetBaseModel):
    
    def __init__(self, opt, logger):
        super(Model, self).__init__(opt, logger)
        self.requires = ['rgb', 'depth', 'silhou', 'normal']
        self.net = Uresnet(
            [3, 1, 1],
            ['normal', 'depth', 'silhou']
        )
        self.criterion = nn.functional.mse_loss
        self.optimizer = self.adam(
            self.net.parameters(),
            lr=opt.lr,
            **self.optim_params
        )
        self._nets = [self.net]
        self._optimizers.append(self.optimizer)
        self.input_names = ['rgb']
        self.gt_names = ['depth', 'silhou', 'normal']
        self.init_vars(add_path=True)
        self._metrics = ['loss', 'depth', 'silhou', 'normal']
        self.init_weight(self.net)

    def __str__(self):
        return "MarrNet-1 predicting 2.5D sketches"

    def _train_on_batch(self, epoch, batch_idx, batch):
        self.net.zero_grad()
        pred = self.predict(batch)
        loss, loss_data = self.compute_loss(pred)
        loss.backward()
        self.optimizer.step()
        batch_size = len(batch['rgb_path'])
        batch_log = {'size': batch_size, **loss_data}
        return batch_log

    def _vali_on_batch(self, epoch, batch_idx, batch):
        pred = self.predict(batch, no_grad=True)
        _, loss_data = self.compute_loss(pred)
        if np.mod(epoch, self.opt.vis_every_vali) == 0:
            if batch_idx < self.opt.vis_batches_vali:
                outdir = join(self.full_logdir, 'epoch%04d_vali' % epoch)
                makedirs(outdir, exist_ok=True)
                output = self.pack_output(pred, batch)
                self.visualizer.visualize(output, batch_idx, outdir)
                np.savez(join(outdir, 'batch%04d' % batch_idx), **output)
        batch_size = len(batch['rgb_path'])
        batch_log = {'size': batch_size, **loss_data}
        return batch_log

    def pack_output(self, pred, batch, add_gt=True):
        pred_normal = pred['normal'].detach().cpu()
        pred_silhou = pred['silhou'].detach().cpu()
        pred_depth = pred['depth'].detach().cpu()
        gt_silhou = self.postprocess(batch['silhou'])
        out = {}
        out['rgb_path'] = batch['rgb_path']
        out['pred_normal'] = self.postprocess(pred_normal, bg=1.0, input_mask=gt_silhou).numpy()
        out['pred_silhou'] = self.postprocess(pred_silhou).numpy()
        pred_depth = self.postprocess(pred_depth, bg=0.0, input_mask=gt_silhou)
        out['pred_depth'] = pred_depth.numpy()
        if add_gt:
            out['normal_path'] = batch['normal_path']
            out['silhou_path'] = batch['silhou_path']
            out['depth_path'] = batch['depth_path']
        return out

    def compute_loss(self, pred):
        """
        TODO: we should add normal and depth consistency loss here in the future.
        """
        pred_normal = pred['normal']
        pred_depth = pred['depth']
        pred_silhou = pred['silhou']
        is_fg = self._gt.silhou != 0  # excludes background
        is_fg_full = is_fg.expand_as(pred_normal)
        loss_normal = self.criterion(
            pred_normal[is_fg_full], self._gt.normal[is_fg_full]
        )
        loss_depth = self.criterion(
            pred_depth[is_fg], self._gt.depth[is_fg]
        )
        loss_silhou = self.criterion(pred_silhou, self._gt.silhou)
        loss = loss_normal + loss_depth + loss_silhou
        loss_data = {}
        loss_data['loss'] = loss.mean().item()
        loss_data['normal'] = loss_normal.mean().item()
        loss_data['depth'] = loss_depth.mean().item()
        loss_data['silhou'] = loss_silhou.mean().item()
        return loss, loss_data
