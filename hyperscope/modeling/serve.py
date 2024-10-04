import multiprocessing
import numpy as np
import os
import time
import typer
from datetime import datetime as dt
from hyperscope import config
from loguru import logger
from pathlib import Path
from uvicorn import Config, Server
from pytorch_lightning import LightningDataModule
from unet.dataset import InputLoader, TargetLoader

app = typer.Typer()


def load_data_module(
    data_module: LightningDataModule, input_dir: Path, target_dir: Path, transforms=None, **kwargs
):

    input_loader = InputLoader(str(input_dir))
    target_loader = TargetLoader(str(target_dir))

    data = data_module(input_loader, target_loader, transforms, **kwargs)

    return data


class UvicornServer(multiprocessing.Process):
    def __init__(self, config: Config):
        super().__init__()
        self.server = Server(config=config)
        self.config = config

    def stop(self):
        self.terminate()

    def run(self, *args, **kwargs):
        self.server.run()
        logger.info("Server has been terminated.")


def setup_logger(process_dir):
    log_file = process_dir / f"{dt.now().strftime('%Y%m%d_%H%M%S')}_server.log"
    logger.remove()  # Remove default handler
    logger.add(log_file, rotation="500 MB")  # Log to file
    return log_file


def convert_arrays_to_lists(array_list):
    result = []
    for arr in array_list:
        if isinstance(arr, np.ndarray):
            # Convert numpy array to list and handle different dtypes
            if np.issubdtype(arr.dtype, np.integer):
                result.append([int(x) for x in arr.tolist()])
            elif np.issubdtype(arr.dtype, np.floating):
                result.append([float(x) for x in arr.tolist()])
            elif arr.dtype == bool:
                result.append([bool(x) for x in arr.tolist()])
            elif arr.dtype.kind == 'U' or arr.dtype.kind == 'S':
                result.append([str(x) for x in arr.tolist()])
            else:
                result.append(arr.tolist())  # Default to direct conversion
        elif isinstance(arr, list):
            # If it's already a list, we still need to ensure all elements are Python types
            result.append([type(x)(x) for x in arr])
        else:
            raise TypeError(f"Unsupported type in input: {type(arr)}")
    
    return result


def create_app(model_path, process_dir):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    from pytorch_lightning import Trainer
    from unet.dataset import UNetDataModule
    from unet.config import UNetTransformer
    from unet.model import UNetLightning

    webapp = FastAPI()
    model = UNetLightning.load_from_checkpoint(model_path)
    
    class PredictRequest(BaseModel):
        input_dir: str
        target_dir: str

    @webapp.post("/predict")
    async def predict(request: PredictRequest):
        data_module = UNetDataModule(
            input_loader=InputLoader(request.input_dir),
            target_loader=TargetLoader(request.target_dir),
            transforms=UNetTransformer(),
            batch_size=1,
            n_workers=4,
        )
        results = Trainer().predict(model, datamodule=data_module)
        results = convert_arrays_to_lists(results)
        return JSONResponse(content={"predictions": results})

    @webapp.on_event("startup")
    async def startup_event():
        logger.info("Server started. Ready to accept requests.")

    @webapp.on_event("shutdown")
    async def shutdown_event():
        logger.info("Server shutting down...")

    @webapp.exception_handler(Exception)
    async def exception_handler(request, exc):
        if isinstance(exc, KeyboardInterrupt):
            await shutdown_event()
            exit(0)
        logger.error(f"An error occurred: {exc}")
        return JSONResponse(
            status_code=500,
            content={"message": "An error occurred."},
        )

    return webapp


@app.command()
def start_server(
    model_path: Path = config.MODELS_DIR / "unet-UN-348-epoch=05-val_dice=0.89.ckpt",
    host: str = "0.0.0.0",
    port: int = 8000,
):
    process_dir = Path(__file__).parent
    pid_file = process_dir / "server.pid"

    if pid_file.exists():
        logger.error("Server already running.")
        logger.info("To stop the server, run `python serve.py stop_server`.")
        return

    log_file = setup_logger(process_dir)
    logger.info(f"Starting server... Log file: {log_file}")

    webapp = create_app(model_path, process_dir)
    config = Config(app=webapp, host=host, port=port, log_level="info")
    server = UvicornServer(config=config)
    server.start()

    # Wait a bit to ensure the process has started
    time.sleep(2)

    if server.is_alive():
        logger.info(f"Server process started with PID: {server.pid}")
        with open(pid_file, "w") as f:
            f.write(str(server.pid))
        logger.info(f"PID file created at {pid_file}")
        logger.info("Server startup complete. You can now use the server.")
        logger.info(
            f"Server is running in the background. Check the log file at {log_file} for details."
        )
    else:
        logger.error("Server process failed to start. Exiting.")
        server.stop()


@app.command()
def stop_server():
    pid_file = Path(__file__).parent / "server.pid"

    if not pid_file.exists():
        logger.error("No PID file found. Server might not be running.")
        return

    try:
        with open(pid_file, "r") as f:
            pid = int(f.read().strip())

        os.kill(pid, 15)  # Send SIGTERM
        logger.info(f"Sent termination signal to process with PID: {pid}")

        # Wait for the process to terminate
        for _ in range(10):  # Wait for up to 10 seconds
            time.sleep(1)
            try:
                os.kill(pid, 0)  # This will raise an OSError if the process is not running
            except OSError:
                logger.info(f"Process with PID {pid} has terminated.")
                break
        else:
            logger.warning(f"Process with PID {pid} did not terminate. Sending SIGKILL.")
            os.kill(pid, 9)  # Send SIGKILL

    except FileNotFoundError:
        logger.error(f"PID file {pid_file} not found.")
    except ValueError:
        logger.error(f"Invalid PID in {pid_file}.")
    except ProcessLookupError:
        logger.error(f"No process found with PID {pid}. The server might have already stopped.")
    except Exception as e:
        logger.error(f"An error occurred while trying to stop the server: {e}")

    # Always try to remove the PID file
    try:
        pid_file.unlink()
        logger.info("PID file removed.")
    except Exception as e:
        logger.error(f"Failed to remove PID file: {e}")


def start_server_blocking(
    model_path: Path = config.MODELS_DIR / "unet-UN-348-epoch=05-val_dice=0.89.ckpt",
    host: str = "0.0.0.0",
    port: int = 8000,
):
    webapp = create_app(model_path, Path(__file__).parent)
    config = Config(app=webapp, host=host, port=port, log_level="info")
    server = Server(config=config)
    server.run()


if __name__ == "__main__":
    start_server_blocking()  # for testing purposes
    app()
