# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from torch.optim.optimizer import Optimizer, required
from torch.optim import RMSprop

from . import FairseqOptimizer, register_optimizer


@register_optimizer('rmsprop')
class FairseqRMSprop(FairseqOptimizer):
    def __init__(self, args, params):
        super().__init__(args, params)
        self._optimizer = RMSprop(params, **self.optimizer_config)

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        parser.add_argument('--rmsprop-alpha', type=float, default=0.99, metavar='D',
                            help='alpha for RMSprop optimizer')
        parser.add_argument('--rmsprop-eps', type=float, default=1e-8, metavar='D',
                            help='epsilon for RMSprop optimizer')
        parser.add_argument('--rmsprop-centered', dest='rmsprop_centered', action='store_true')
        parser.add_argument('--no-rmsprop-centered', dest='rmsprop_centered', action='store_false')
        parser.set_defaults(rmsprop_centered=False)


    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            'lr': self.args.lr[0],
            'alpha': self.args.rmsprop_alpha,
            'eps': self.args.rmsprop_eps,
            'weight_decay': self.args.weight_decay,
            'momentum': self.args.momentum,
            'centered': self.args.rmsprop_centered,
        }