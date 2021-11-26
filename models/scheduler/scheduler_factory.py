""" Scheduler Factory
Hacked together by / Copyright 2020 Ross Wightman
"""
from .cosine_lr import CosineLRScheduler
from .tanh_lr import TanhLRScheduler
from .step_lr import StepLRScheduler
from .plateau_lr import PlateauLRScheduler
from .multistep_lr import MultiStepLRScheduler
from .linear_lr import LinearScheduler
from torch.optim import lr_scheduler as torch_lr


def create_scheduler(args, optimizer):
    num_epochs = args.num_epochs

    if getattr(args, 'lr_noise', None) is not None:
        lr_noise = getattr(args, 'lr_noise')
        if isinstance(lr_noise, (list, tuple)):
            noise_range = [n * num_epochs for n in lr_noise]
            if len(noise_range) == 1:
                noise_range = noise_range[0]
        else:
            noise_range = lr_noise * num_epochs
    else:
        noise_range = None
    noise_args = dict(
        noise_range_t=noise_range,
        noise_pct=getattr(args, 'lr_noise_pct', 0.67),
        noise_std=getattr(args, 'lr_noise_std', 1.),
        noise_seed=getattr(args, 'seed', 42),
    )

    lr_scheduler = None
    if args.lr_policy == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            t_mul=getattr(args, 'lr_cycle_mul', 1.),
            lr_min=args.min_lr,
            decay_rate=args.decay_rate,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs,
            cycle_limit=getattr(args, 'lr_cycle_limit', 1),
            t_in_epochs=True,
            **noise_args,
        )
        num_epochs = lr_scheduler.get_cycle_length() + args.cooldown_epochs
    elif args.lr_policy == 'tanh':
        lr_scheduler = TanhLRScheduler(
            optimizer,
            t_initial=num_epochs,
            t_mul=getattr(args, 'lr_cycle_mul', 1.),
            lr_min=args.min_lr,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs,
            cycle_limit=getattr(args, 'lr_cycle_limit', 1),
            t_in_epochs=True,
            **noise_args,
        )
        num_epochs = lr_scheduler.get_cycle_length() + args.cooldown_epochs
    elif args.lr_policy == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=args.decay_epochs,
            decay_rate=args.decay_rate,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs,
            t_in_epochs=True,
            **noise_args,
        )
    elif args.lr_policy == 'multistep':
        lr_scheduler = MultiStepLRScheduler(
            optimizer,
            decay_t=args.decay_epochs,
            decay_rate=args.decay_rate,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs,
            **noise_args,
        )
    elif args.lr_policy == 'plateau':
        mode = 'min' if 'loss' in getattr(args, 'eval_metric', '') else 'max'
        lr_scheduler = PlateauLRScheduler(
            optimizer,
            decay_rate=args.decay_rate,
            patience_t=args.patience_epochs,
            lr_min=args.min_lr,
            mode=mode,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs,
            cooldown_t=0,
            **noise_args,
        )
    elif args.lr_policy == 'linear':
        lr_scheduler = LinearScheduler(
            optimizer,
            decay_t=args.decay_epochs,
            decay_rate=args.decay_rate,
            lr_min=args.min_lr,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs,
            **noise_args,
        )

    return lr_scheduler, num_epochs


def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs)**exponent


# class PolyLR(object):
#     # lr_decay=0.9
#     def __init__(self, optimizer, curr_iter, max_iter, lr_decay):
#         self.max_iter = float(max_iter)
#         self.init_lr_groups = []
#         for p in optimizer.param_groups:
#             self.init_lr_groups.append(p['lr'])
#         self.param_groups = optimizer.param_groups
#         self.curr_iter = curr_iter
#         self.lr_decay = lr_decay
#
#     def step(self):
#         for idx, p in enumerate(self.param_groups):
#             p['lr'] = self.init_lr_groups[idx] * (1 - self.curr_iter / self.max_iter) ** self.lr_decay
#
#
# class LinearRule:
#     def __init__(self, epoch_retain, epochs_decay):
#         self.epoch_retain = epoch_retain
#         self.epochs_decay = epochs_decay
#
#     def __call__(self, epoch):
#         assert epoch <= self.epoch_retain+self.epochs_decay
#         lr_l = 1.0 - max(0, epoch - self.epoch_retain) / float(self.epochs_decay + 1)
#         return lr_l
# from torch.optim import lr_scheduler
# # learning strategy: __init__  get_lr (state_dict load_state_dict)  _get_closed_form_lr
# def get_scheduler(optimizer, opt, iterations=-1):
#     """Return a learning rate scheduler
#
#     Parameters:
#         optimizer          -- the optimizer of the network
#         opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
#                               opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
#
#     For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
#     and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
#     For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
#     See https://pytorch.org/docs/stable/optim.html for more details.
#     """
#     if opt.lr_policy == 'linear':
#         def lambda_rule(epoch):
#             lr_l = 1.0 - max(0, epoch - opt.epoch_retain) / float(opt.epochs_decay + 1)  # + opt.epoch_start
#             return lr_l
#         scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
#     elif opt.lr_policy == 'step':
#         scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1, last_epoch=iterations)
#     elif opt.lr_policy == 'plateau':
#         scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
#     elif opt.lr_policy == 'cosine':
#         scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epoch_start, eta_min=0)
#     elif opt.lr_policy == 'constant':
#         scheduler = None  # constant scheduler
#     else:
#         return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
#     return scheduler
#
#
# def get_scheduler_dict(optimizer, hyperparameters, iterations=-1):
#
#     if hyperparameters['lr_policy'] == 'linear':
#         def lambda_rule(epoch):
#             lr_l = 1.0 - max(0, epoch + hyperparameters['epoch_count'] - hyperparameters['n_epochs']) \
#                    / float(hyperparameters['n_epochs_decay'] + 1)
#             return lr_l
#         scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
#     elif hyperparameters['lr_policy'] == 'step':
#         scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['lr_decay_iters'],
#                                         gamma=hyperparameters['gamma'], last_epoch=iterations)  # gamma=0.1
#     elif hyperparameters['lr_policy'] == 'plateau':
#         scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
#     elif hyperparameters['lr_policy'] == 'cosine':
#         scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=hyperparameters['n_epochs'], eta_min=0)
#     elif hyperparameters['lr_policy'] == 'constant':
#         scheduler = None  # constant scheduler
#     else:
#         return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
#     return scheduler

