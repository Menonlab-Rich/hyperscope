from abc import abstractmethod, ABC
import yaml
import os
from functools import partial


def _path(loader, node):
    seq = loader.construct_sequence(node)
    return os.path.expanduser(os.path.join(*seq))


class BaseConfigHandler(ABC):
    """
    Abstract class for a config handler
    """

    @abstractmethod
    def __init__(self):
        self.config = {}

    @abstractmethod
    def save(self, path: str):
        """
        Save the config
        """
        pass

    @abstractmethod
    def load(self, path: str):
        """
        Load the config
        """
        pass

    def __getitem__(self, key):
        return self.config.get(key, None)

    def __setitem__(self, key, value):
        self.config[key] = value

    def __iter__(self):
        for key, value in self.config.items():
            yield key, value

    def __getattr__(self, name: str) -> any:
        # If the attribute is not found in the class, proxy it to the config
        return getattr(self.config, name)


class YAMLConfig(BaseConfigHandler):
    def __init__(self, file_path: str, gl: dict = {}):
        super(YAMLConfig, self).__init__()
        self.gl = gl
        yaml.add_constructor('!path', _path)
        yaml.add_constructor('!global', partial(self._get_global, self))
        self.config = yaml.load(open(file_path, 'r'), Loader=yaml.FullLoader)

    def __getattr__(self, name: str) -> any:
        return self.config[name]

    @staticmethod
    def _get_global(self, loader, node) -> any:
        return self.gl.get(node.value, '')

    def get(self, key, default):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value

    def update(self, key, value):
        self.set(key, value)

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.config, f)

    def load(self, path):
        with open(path, 'r') as f:
            for line in f:
                k, v = line.strip().split('=')
                self.set(k, v)

    def __str__(self):
        return str(self.config)


def create_transform_function(keys, transform_name, transform):
    # Generate the function signature
    args_str = ", ".join(keys)
    # Generate the dictionary to unpack into the transform function
    dict_str = ", ".join([f"'{key}': {key}" for key in keys])

    # Define the function template
    func_template = f"""
def transform_function({args_str}):
    transform_input = {{{dict_str}}}
    transformed = {transform_name}(**transform_input)
    res = tuple(transformed[key] for key in {keys})
    if len(res) == 1:
        return res[0]
    return res
"""

    # Define the function in the local scope
    local_scope = {}
    exec(func_template, {transform_name: transform}, local_scope)
    return local_scope["transform_function"]
