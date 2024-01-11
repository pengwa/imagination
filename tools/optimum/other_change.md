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
