import logging


def do_test(dataload, model, args):
    data_size = len(dataload)
    for i, data in enumerate(dataload):
        model.setup(data)
        model.test()









