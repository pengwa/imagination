# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import contextlib
import inspect
import warnings


from onnxruntime.training.utils.runtime_patch._ds_code_store import all_gather_dp_groups


def _get_normalized_str(function) -> str:
    return inspect.getsource(function)


def override_all_gather_dp_groups(cur_ds_version: Version, optimizer) -> bool:
    with contextlib.suppress(Exception):
        import deepspeed

        original_deepspeed_checkpoint = deepspeed.checkpointing.all_gather_dp_groups
        deepspeed.checkpointing.checkpoint = _override_gradient_checkpoint(original_deepspeed_checkpoint)


        original_all_gather_dp_groups = deepspeed.runtime.utils.all_gather_dp_groups

        _version_to_source_code_map = {"0.10.0": all_gather_dp_groups}

        # Try to find the biggest version that is smaller than or equal to cur_ds_version.
        # then compare the source code (in case the found version is the latest version supported);
        # If current code does not match the found version, return False, and raise a warning to
        # add the new version to the list.
        versions = [Version(v) for v in _version_to_source_code_map]
        sorted_versions = sorted(versions, reverse=True)
        version_to_compare = None
        for sv in sorted_versions:
            if cur_ds_version >= sv:
                version_to_compare = sv
                break

        if version_to_compare is None:
            warnings.warn(
                "Unable to find a DeepSpeed version that is smaller than or equal to the current version "
                f"{cur_ds_version}. Skip override_all_gather_dp_groups",
                UserWarning,
            )

        v_all_gather_dp_groups = _version_to_source_code_map[str(version_to_compare)]
        func_name = "all_gather_dp_groups"
        cur_code_str = _get_normalized_str(original_all_gather_dp_groups)
        v_code_str = _get_normalized_str(v_all_gather_dp_groups)
        if cur_code_str != v_code_str:
            warnings.warn(
                f"DeepSpeed function {func_name} has changed after version {version_to_compare}. "
                f"Please append new version {cur_ds_version} in _version_to_source_code_map and _ds_code_store.py.\n"
                f"---[{func_name}] Old Source Code Start----\n"
                f"{v_code_str}\n"
                f"---{func_name} Old Source Code End----\n"
                f"---[{func_name}] New Source Code Start----\n"
                f"{cur_code_str}\n"
                f"---{func_name} New Source Code End----",
                UserWarning,
            )
            return

        def updated_all_gather_dp_groups(partitioned_param_groups, dp_process_group, start_alignment_factor, allgather_bucket_size):
            for group_id, partitioned_params in enumerate(partitioned_param_groups):
                # Sequential AllGather Best of both worlds
                partition_id = dist.get_rank(group=dp_process_group[group_id])
                dp_world_size = dist.get_world_size(group=dp_process_group[group_id])

                num_shards = max(1, partitioned_params[partition_id].numel() * dp_world_size // allgather_bucket_size)

                shard_size = partitioned_params[partition_id].numel() // num_shards

                # Enforce nccl/rccl alignment of start location of each shard
                shard_size = shard_size - (shard_size % start_alignment_factor)

                num_elements = shard_size

                assert shard_size * num_shards <= partitioned_params[partition_id].numel()

                for shard_id in range(num_shards):

                    if shard_id == (num_shards - 1):
                        num_elements = partitioned_params[partition_id].numel() - shard_id * shard_size

                    shard_list = []
                    for dp_id in range(dp_world_size):
                        curr_shard = partitioned_params[dp_id].narrow(0, shard_id * shard_size, num_elements).detach()
                        shard_list.append(curr_shard)

                    dist.all_gather(shard_list, shard_list[partition_id], dp_process_group[group_id])

        deepspeed.runtime.utils.all_gather_dp_groups = updated_all_gather_dp_groups
        warnings.warn("DeepSpeed function {func_name} has been overriden.")
