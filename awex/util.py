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


def check_and_log_nan_values(tensor, tensor_name, stage_info="", max_indices=20):
    """
    Check for NaN values in a tensor and log detailed information including indices.

    Args:
        tensor: The tensor to check for NaN values
        tensor_name: Name of the tensor for logging
        stage_info: Additional stage information for logging context
        max_indices: Maximum number of NaN indices to log (default: 20)

    Returns:
        bool: True if NaN values were detected, False otherwise
    """
    if torch.isnan(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        logger.warning(
            f"Parameter {tensor_name} contains NaN values{stage_info}! Shape: {tensor.shape}"
        )
        logger.warning(f"NaN count in {tensor_name}: {nan_count}")

        # Find and print indices of NaN values
        nan_indices = torch.nonzero(torch.isnan(tensor), as_tuple=True)
        logger.warning(f"NaN indices in {tensor_name}{stage_info}: {nan_indices}")

        # Print first max_indices NaN indices
        if nan_count <= max_indices:
            logger.warning(
                f"All NaN values in {tensor_name}{stage_info}: {nan_indices}"
            )
        else:
            logger.warning(
                f"First {max_indices} NaN indices in {tensor_name}{stage_info}: {tuple(idx[:max_indices] for idx in nan_indices)}"
            )

        return True
    return False


def compare_and_log_tensor_differences(
    tensor1,
    tensor2,
    tensor_name,
    atol=1e-08,
    rtol=1e-05,
    max_differences=20,
    exact_match=False,
):
    """
    Compare two tensors and log detailed information about inconsistent elements.

    Args:
        tensor1: First tensor to compare
        tensor2: Second tensor to compare
        tensor_name: Name of the tensor for logging
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison
        max_differences: Maximum number of differences to log (default: 20)

    Returns:
        bool: True if tensors are consistent, False otherwise
    """
    # Check if shapes match
    if tensor1.shape != tensor2.shape:
        logger.error(
            f"Shape mismatch for {tensor_name}: {tensor1.shape} vs {tensor2.shape}"
        )
        return False

    if tensor1.dtype != tensor2.dtype:
        logger.error(
            f"Tensor {tensor_name} has different dtypes: {tensor1.dtype} vs {tensor2.dtype}"
        )
        return False

    # Check if tensors are close using torch.allclose
    if exact_match:
        if torch.equal(tensor1, tensor2):
            return True
    else:
        if torch.allclose(tensor1, tensor2, atol=atol, rtol=rtol):
            return True
    logger.error(
        f"Tensors are not close for {tensor_name}, get {tensor1.shape} \n{tensor1} expect {tensor2.shape} \n{tensor2}"
    )

    # Find elements that are not close
    close_mask = torch.isclose(tensor1, tensor2, atol=atol, rtol=rtol)
    inconsistent_mask = ~close_mask

    if inconsistent_mask.any():
        inconsistent_count = inconsistent_mask.sum().item()
        logger.error(
            f"Parameter {tensor_name} has {inconsistent_count} inconsistent elements"
        )

        # Find indices of inconsistent elements
        inconsistent_indices = torch.nonzero(inconsistent_mask, as_tuple=True)

        # Calculate absolute and relative differences
        abs_diff = torch.abs(tensor1 - tensor2)
        rel_diff = abs_diff / (
            torch.abs(tensor2) + 1e-8
        )  # Add small epsilon to avoid division by zero

        # Get the actual values at inconsistent positions
        tensor1_values = tensor1[inconsistent_indices]
        tensor2_values = tensor2[inconsistent_indices]
        abs_diff_values = abs_diff[inconsistent_indices]
        rel_diff_values = rel_diff[inconsistent_indices]

        # Log summary statistics
        max_abs_diff = abs_diff_values.max().item()
        max_rel_diff = rel_diff_values.max().item()
        mean_abs_diff = abs_diff_values.mean().item()
        mean_rel_diff = rel_diff_values.mean().item()

        logger.error(
            f"Max absolute difference: {max_abs_diff:.6f}, Max relative difference: {max_rel_diff:.6f}"
        )
        logger.error(
            f"Mean absolute difference: {mean_abs_diff:.6f}, Mean relative difference: {mean_rel_diff:.6f}"
        )

        # Log detailed information for first max_differences elements
        num_to_log = min(int(inconsistent_count), max_differences)

        for i in range(num_to_log):
            idx = tuple(idx[i] for idx in inconsistent_indices)
            val1 = tensor1_values[i].item()
            val2 = tensor2_values[i].item()
            abs_diff_val = abs_diff_values[i].item()
            rel_diff_val = rel_diff_values[i].item()

            logger.error(
                f"  Index {idx}: {val1:.6f} vs {val2:.6f} (abs_diff: {abs_diff_val:.6f}, rel_diff: {rel_diff_val:.6f})"
            )

        if inconsistent_count > max_differences:
            logger.error(
                f"  ... and {inconsistent_count - max_differences} more differences"
            )

        return False

    return True


def compute_statistics(stage_history: dict, step_id: int, duration: float, stage: str):
    if stage not in stage_history:
        stage_history[stage] = []
    history = stage_history[stage]
    history.append(duration)
    if len(history) > 10000:
        history.pop(0)
    if step_id == 2:
        # first step contains init time
        history.pop(history.index(max(history)))
    num_updates = len(history)
    stage_history[stage] = history = sorted(history)
    avg_time = sum(history) / num_updates
    median_time = history[num_updates // 2]
    max_time = history[-1]
    min_time = history[0]
    logger.info(
        f"{stage} time statistics for step {step_id}: average time: {avg_time:.4f} seconds, median time: {median_time:.4f} seconds, "
        f"min time: {min_time:.4f} seconds,  max time: {max_time:.4f} seconds"
    )


def check_train_infer_params_meta(
    training_params_meta: List,
    infer_parameters_meta: List,
    raise_exception: bool = False,
):
    infer_meta = {param_meta.name: param_meta for param_meta in infer_parameters_meta}
    train_meta = {param_meta.name: param_meta for param_meta in training_params_meta}
    common_params = set(infer_meta.keys()) & set(train_meta.keys())
    if len(common_params) != len(infer_meta) or len(common_params) != len(train_meta):
        if len(train_meta) > len(infer_meta):
            diff = set(train_meta.keys()) - common_params
        else:
            diff = set(infer_meta.keys()) - common_params
        logger.error(
            f"Inconsistent parameters meta: "
            f"train {len(train_meta)} infer {len(infer_meta)} diff keys {diff}"
        )
        if raise_exception:
            raise ValueError(
                f"Inconsistent parameters meta for inference and training: "
                f"{len(common_params)} {len(infer_meta)} {len(train_meta)}, diff keys {diff}"
            )
    for param_name in common_params:
        infer_param_meta = infer_meta[param_name]
        train_param_meta = train_meta[param_name]
        if infer_param_meta.global_numel != train_param_meta.global_numel:
            error_msg = (
                f"Inconsistent number of elements for parameter {param_name}: "
                f"{infer_param_meta.global_numel} != {train_param_meta.global_numel}"
            )
            if raise_exception:
                raise ValueError(error_msg)
            else:
                logger.error(error_msg)
        if infer_param_meta.global_shape != train_param_meta.global_shape:
            error_msg = f"Inconsistent shape for parameter {param_name}: {infer_param_meta.global_shape} != {train_param_meta.global_shape}"
            if raise_exception:
                raise ValueError(error_msg)
            else:
                logger.error(error_msg)
        if infer_param_meta.dtype != train_param_meta.dtype:
            error_msg = f"Inconsistent dtype for parameter {param_name}: {infer_param_meta.dtype} != {train_param_meta.dtype}"
            if raise_exception:
                raise ValueError(error_msg)
            else:
                logger.error(error_msg)


def setup_batch_isend_irecv(
    process_group, rank, world_size, tensor_size=10 * 10, dtype=torch.float32
):
    """
    Perform a simple communication using batch_isend_irecv to avoid the hang for later sub-ranks.

    Args:
    process_group (ProcessGroup): The process group to work on.
    tensor_size (int): Size of the tensor to send/receive.
    dtype (torch.dtype): Data type of the tensor.
    """
    assert process_group is not None, "Process group cannot be None"
    device = torch.cuda.current_device()
    logger.info(
        f"Setup batch isend irecv for rank {rank} world size {world_size} device {device}"
    )

    # Create tensors for sending and receiving
    send_tensor = torch.full(
        (tensor_size,), rank, dtype=dtype, device=device, requires_grad=False
    )
    recv_tensor = torch.zeros(
        (tensor_size,), dtype=dtype, device=device, requires_grad=False
    )

    # Prepare the ops for batch_isend_irecv
    ops = []

    # First half of ranks receive from rank + half
    mid_point = world_size // 2
    if rank < mid_point:
        # First half: receive from rank + half
        target_rank = rank + mid_point
        if target_rank < world_size:
            ops.append(
                dist.P2POp(dist.irecv, recv_tensor, target_rank, group=process_group)
            )
    else:
        # Second half: send to rank - half
        target_rank = rank - mid_point
        if target_rank >= 0:
            ops.append(
                dist.P2POp(dist.isend, send_tensor, target_rank, group=process_group)
            )

    # Execute batch_isend_irecv
    if ops:
        reqs = dist.batch_isend_irecv(ops)
        # Wait for all communications to complete
        for req in reqs:
            req.wait()

    # Synchronize
    torch.cuda.synchronize(device=torch.cuda.current_device())
    dist.barrier(group=process_group, device_ids=[torch.cuda.current_device()])

    logger.info(
        f"Simple communication completed for process group of size {world_size}"
    )

    # Verify the results
    if rank < mid_point and rank + mid_point < world_size:
        expected_value = rank + mid_point
        assert torch.all(recv_tensor == expected_value), (
            f"Rank {rank} received incorrect data from rank {rank + mid_point}"
        )

    logger.info("Simple communication verification successful")
