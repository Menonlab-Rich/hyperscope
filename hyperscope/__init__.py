# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
import sys
from loguru import logger
from hyperscope import config  # noqa: F401


try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass




# Set up paths
sys.path.append(str(config.MODELS_DIR.parent))  # Add models directory to path