# Copyright (c) 2020, CoolFong. All rights reserved.
# @Time    : 2020/9/10
# @Author  : CoolFong
"""
This is the configs for project. The defaults use yaml file. And the options provide the terminal's options
 """
import importlib


def find_option_use_name(dataset_name, model_name):
    option_filename = "configs.options." + dataset_name+'_'+model_name
    optionlib = importlib.import_module(option_filename)
    option = None
    for name, cls in optionlib.__dict__.items():
        if name == 'ProjectOptions':
            option = cls
    if option is None:
        print("In %s.py, the class name should matches ProjectOptions ." % option_filename)
        exit(0)
    return option

#
# def create_option(isTrain):
#     option = find_option_use_name(opt.dataset_name, opt.model_name)
#     instance = option().parse(isTrain)


