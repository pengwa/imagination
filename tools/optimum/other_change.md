For optimum/onnxruntime/trainer.py

Change from 

```

    logger.info("Wrap ORTModule for ONNX Runtime training.")
    model = ORTModule(self.model)

```


to 


```
    emd_enable = False


    if 'ORTMODULE_ENABLE_EMBEDDING_SPARSE_OPTIMIZER' in os.environ:
        emd_enable = bool(int(os.environ['ORTMODULE_ENABLE_EMBEDDING_SPARSE_OPTIMIZER']))


    debug_foutputs = f"{os.environ['EXP_ID'] if 'EXP_ID' in os.environ else 'debug'}"

    # model = ORTModule(self.model)
    from onnxruntime.training.ortmodule import ORTModule, DebugOptions, LogLevel

    prefix = 'use_emb_sparsity' if emd_enable else 'no_emb_sparsity'

    model = ORTModule(self.model, DebugOptions(save_onnx=True, log_level=LogLevel.INFO, onnx_prefix=f"{debug_foutputs}/{prefix}_debug_"))
    from onnxruntime.training.utils.hooks import GlobalSubscriberManager, StatisticsSubscriber
    GlobalSubscriberManager.subscribe(
        model, [StatisticsSubscriber(output_dir=debug_foutputs + f"/{prefix}_run_" + str(torch.distributed.get_rank()), override_output_dir=True)]
    )

```


For models/deberta/modeling_deberta.py

Add

```diff

from onnxruntime.training.utils.hooks import inspect_activation

class DisentangledSelfAttention(nn.Module):
+    def __init__(self, config, index):
-    def __init__(self, config):
+         self.index = index
    ...
    def forward(...
        hidden_states = inspect_activation(f"hidden_states_{self.index}", hidden_states)
        if query_states is None:
            qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
            qp = inspect_activation(f"qp_{self.index}", qp)
            query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
            query_layer = inspect_activation(f"query_layer3_{self.index}", query_layer)
            key_layer = inspect_activation(f"key_layer3_{self.index}", key_layer)
            value_layer = inspect_activation(f"value_layer3_{self.index}", value_layer)


```

For orttraining/orttraining/python/training/utils/hooks/_statistics_subscriber.py

def _summarize_activations(...

+    from onnxruntime.training.utils.hooks._subscriber_manager import ORT_NO_INCREASE_GLOBAL_STEP

+    if ORT_NO_INCREASE_GLOBAL_STEP[0] is True:
+        return tensor

    display_name = name + " forward run" if is_forward is True else name + " backward run"
    output_file_name = name + "_forward" if is_forward is True else name + "_backward"

    if tensor is None or not isinstance(tensor, torch.Tensor):
        print(f"{display_name} not a torch tensor, value: {tensor}")
        return tensor

    step_path = Path(step_folder)
    if not step_path.exists():
        step_path.mkdir(parents=True, exist_ok=False)
    order_file_path = step_path / "order.txt"
    tensor_file_path = step_path / output_file_name

    with order_file_path.open(mode="a", encoding="utf-8") as f:
        f.write(f"{output_file_name}\n")

    with tensor_file_path.open(mode="w", encoding="utf-8") as f:
        # If indices is given, we flatten the first two dims of tensor, and slice the tensor with indices.
        # Otherwise, we reuse the original tensor.
        tensor_to_analyze = tensor.flatten(start_dim=0, end_dim=1)[indices, ...] if indices is not None else tensor
        _summarize_tensor(display_name, tensor_to_analyze, f, depth, self._run_on_cpu, self._bucket_size, indices, step, module)

+    if 'hidden_states_' in display_name or 'qp_' in display_name:
+        if indices is not None:
+            tensor_shape = tensor.shape
+            new_t = tensor.view(-1, tensor_shape[2])

+            mask = torch.ones_like(new_t)
+            mask[indices] = torch.zeros_like(new_t[indices])

+            all_zeros = torch.zeros_like(new_t)

+            new_t = torch.where(mask == 1, all_zeros, new_t)
+            n = new_t.view(tensor_shape)
+            return n
+         else:
+             return tensor
+    else:
+        return tensor
