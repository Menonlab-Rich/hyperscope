from base.config import YAMLConfig
from hyperscope import config as hconfig
from pathlib import Path
import inspect
from itertools import takewhile

file_dir = Path(__file__).parent

config = YAMLConfig(
    str(file_dir / "config.yml"), gl=dict(takewhile(lambda i: i[0] != "__builtins__", inspect.getmembers(hconfig)))
)

if __name__ == '__main__':
    print(config['checkpoint'])