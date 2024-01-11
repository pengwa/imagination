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
