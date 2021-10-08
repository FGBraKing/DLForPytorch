import numpy as np
from argparse import ArgumentParser, REMAINDER, ZERO_OR_MORE, OPTIONAL
from configs.utils_config import ConfigDict


def parse_args(args=None):
    parser = ArgumentParser(description="Project's useful tool to parse args")
    # rest from the training program
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local_rank of distributed processes. local_rank = gpu_ids[ind], -1 means cpu')
    parser.add_argument('--config_path', type=str, default=None, help='the path of config')
    parser.add_argument('--use_config', default=False, action="store_true", help='whether to use config')

    parser.add_argument('option_name', type=str, nargs=OPTIONAL, default='ProjectOptions',
                        help='useless now, just a position flag')
    parser.add_argument('training_script_args', nargs=REMAINDER, help='training_script_args')
    # OPTIONAL = '?'
    # ZERO_OR_MORE = '*'
    # ONE_OR_MORE = '+'
    # REMAINDER = '...'
    return parser.parse_args(args=args)


def get_opt(args=None):
    args = parse_args(args=args)    #
    # print('args:', args)
    if args.use_config and args.config_path is not None:
        from configs.default_config import _C as cfg    # yacs.config.CfgNode, dict
        cfg.merge_from_file(args.config_path)           # dict
        cfg.local_rank = args.local_rank

        option = ConfigDict(cfg)
        option.random_state = np.random.RandomState(seed=option.seed)
        return option
    else:
        import importlib
        option_lib = importlib.import_module("configs.options")
        try:
            option_class = getattr(option_lib, args.option_name)
        except AttributeError as e:
            print('some wrong of [{}] have been found, maybe the option name {} that you input can not be found, '
                  'using a default option with the name of {}'.format(e, args.option_name, 'ProjectOptions'))
            option_class = getattr(option_lib, 'ProjectOptions')
        option = option_class().parse(args=args.training_script_args)
        return option


if __name__ == '__main__':
    # opt = parse_args(args=['fad', '--local_rank=5', '--config_path=2', '--dsf', 'haha', '--local_rank', '4'])
    opt = get_opt(args=['fff', '--name=hello'])
    print('option get ready')
    print(type(opt))
    print(opt)
    print(vars(opt))
    # print(type(out_opt))
    # print(out_opt)



