import struct
from typing import List
import torch.distributed as dist
import os
from enum import Enum
import json
import logging
import socket
import torch
import pickle

logger = logging.getLogger(__name__)


def configure_logging(level=logging.INFO, force=True):
    logging.basicConfig(
        level=level,
        format="%(asctime)s\t%(levelname)s %(filename)s:%(lineno)s -- %(process)d -- %(message)s",
        force=force,
    )


def get_ip_address():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return socket.gethostbyname(socket.gethostname())


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def to_binary(data):
    # Serialize messages using pickle
    pickled_data = pickle.dumps(data)
    # Get the length of the pickled data
    data_len = len(pickled_data)
    # Create the binary response: length of data (4 bytes) + pickled data
    return struct.pack("!I", data_len) + pickled_data


def from_binary(binary):
    # Extract the length of the pickled data
    data_len = struct.unpack("!I", binary[:4])[0]
    # Extract and unpickle the data
    pickled_data = binary[4 : 4 + data_len]
    data = pickle.loads(pickled_data)
    return data


def to_dict(param_meta, ignore_keys=None) -> dict:
    """Convert the parameter meta to a dict."""
    ignore_keys = ignore_keys or set()

    def convert_value(v):
        if isinstance(v, Enum):
            return v.value  # Handle enums
        if isinstance(v, (tuple, list)):
            return [convert_value(x) for x in v]
        if isinstance(v, torch.dtype):
            return str(v)  # Handle torch.dtype
        if isinstance(v, slice):
            return str(v)  # Handle slice
        if isinstance(v, dict):
            return {k: convert_value(v) for k, v in v.items() if k not in ignore_keys}
        if hasattr(v, "__dict__"):
            return {
                k: convert_value(v)
                for k, v in v.__dict__.items()
                if not k.startswith("_") and k not in ignore_keys
            }
        if hasattr(v, "__slots__"):
            return {
                k: convert_value(getattr(v, k))
                for k in v.__slots__
                if not k.startswith("_") and k not in ignore_keys
            }
        return v

    param_dict = convert_value(param_meta)
    return param_dict


def to_json(param_meta, ignore_keys=None) -> str:
    """Convert the parameter meta to a json string."""
    return json.dumps(to_dict(param_meta, ignore_keys), indent=2)


def init_weights_update_group(
    master_address,
    master_port,
    rank,
    world_size,
    group_name,
    backend="nccl",
    role="",
):
    """Initialize the Torch process group for model parameter updates."""
    assert torch.distributed.is_initialized(), (
        "Default torch process group must be initialized"
    )
    assert group_name != "", "Group name cannot be empty"

    logger.info(
        f"init custom process group for {role}: master_address={master_address}, master_port={master_port}, "
        f"rank={rank}, world_size={world_size}, group_name={group_name}, backend={backend}, "
        f"current device id {torch.cuda.current_device()} "
        f"CUDA_VISIBLE_DEVICES {os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"Local rank env {os.environ.get('LOCAL_RANK')} DEVICE env {os.environ.get('DEVICE')} "
        f"Global rank env {os.environ.get('RANK')}"
    )

    from sglang.srt.utils import init_custom_process_group

    try:
        group = init_custom_process_group(
            backend=backend,
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size,
            rank=rank,
            group_name=group_name,
        )
        logger.info(f"Initialized custom process group: {group}")
        return group
    except Exception as e:
        raise RuntimeError(f"Failed to initialize custom process group: {e}.")

