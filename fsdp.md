[https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html#fsdp1-to-fsdp2-migration-guide)


Migration Steps

Replace the imports

Implement your ‘policy’ directly (apply fully_shard to the desired sublayers)

Wrap your root model with fully_shard instead of FSDP

Get rid of param_init_fn and manually call model.reset_parameters()

Replace other FSDP1 kwargs (see below)

sharding_strategy

FULL_SHARD: reshard_after_forward=True

SHARD_GRAD_OP: reshard_after_forward=False

HYBRID_SHARD: reshard_after_forward=True with a 2D device mesh

_HYBRID_SHARD_ZERO2: reshard_after_forward=False with a 2D device mesh

cpu_offload

CPUOffload.offload_params=False: offload_policy=None

CPUOffload.offload_params = True: offload_policy=CPUOffloadPolicy()

backward_prefetch

BACKWARD_PRE: always used

BACKWARD_POST: not supported

mixed_precision

buffer_dtype is omitted because fully_shard does not shard buffers

fully_shard’s cast_forward_inputs maps to both cast_forward_inputs and cast_root_forward_inputs in FSDP1

output_dtype is a new config for fully_shard
