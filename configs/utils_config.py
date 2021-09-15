import yaml
from types import SimpleNamespace


def get_config(config):
    with open(config, 'r') as stream:
        loader = yaml.FullLoader(stream)
        out = loader.get_single_data()
        out1 = yaml.load(stream)
        return out


def dict2obj(dic):
    return SimpleNamespace(**dic)


def obj2dict(obj):
    # return obj.__dict__
    return vars(obj)


class DictObj(object):
    def __init__(self, dic):
        self.dic = dic

    def __setattr__(self, key, value):
        if key == 'dic':
            object.__setattr__(self, key, value)
            return
        print('set attr called {},{}'.format(key, value))
        self.dic[key] = value

    def __getattr__(self, item):
        value = self.dic[item]
        if isinstance(value, dict):
            return DictObj(value)
        if isinstance(value, (list, tuple)):
            r = []
            for i in value:
                r.append(DictObj(i))
            return r
        else:
            return self.dic[item]

    def __getitem__(self, item):
        return self.dic[item]


class ConfigDict(dict):
    def __init__(self, *args, **kwargs):
        super(ConfigDict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    if isinstance(v, dict):
                        v = ConfigDict(v)
                    if isinstance(v, list):
                        self.__convert(v)
                    self[k] = v
        if kwargs:
            for k, v in kwargs.items():
                if isinstance(v, dict):
                    v = ConfigDict(v)
                if isinstance(v, list):
                    self.__convert(v)
                self[k] = v

    def __convert(self, v):
        '''
         列表还是列表， 列表里边的字典变成ConfigDict
        '''
        for elem in range(0, len(v)):
            if isinstance(v[elem], dict):
                v[elem] = ConfigDict(v[elem])
            elif isinstance(v[elem], list):
                self.__convert(v[elem])

    def __getattr__(self, item):
        return self.get(item)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(ConfigDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(ConfigDict, self).__delitem__(key)
        del self.__dict__[key]


def main():
    # tt = [{'hand': 15, 'ddd': 18, 'hg': {'fd': 22, 'love': 66}}, 4]
    tt = {'hand': [1, 2, 3], 'ddd': 18, 'hg': {'fd': 22, 'love': 66}}
    tt1 = {'hg1': {'fd1': 22, 'love1': [66, 77, 88]}}
    config = ConfigDict(tt, tt1)
    print(type(config))


if __name__ == '__main__':
    main()
