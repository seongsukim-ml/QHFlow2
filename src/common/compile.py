"""Optional compilation without changing parameter or checkpoint names."""
import torch


def compile_flow(module, scope="heads", *, backend="inductor", mode=None, options=None):
    """Compile tensor regions, retaining eager geometry and checkpoint names.

    The blocks scope follows the dynamic, full-graph tensor-core boundary used
    by maloQ's flash branch. No FlashSO2 weights or rotation kernels are ported.
    """
    if scope == "model":
        module.model.compile(dynamic=True, fullgraph=False)
    elif scope == "muon":
        # Benchmark-only process scope; state and update rules remain native.
        import torch.optim._muon as native
        native._zeropower_via_newtonschulz = torch.compile(
            native._zeropower_via_newtonschulz, dynamic=True, fullgraph=True)
    elif scope in {"heads", "blocks"}:
        # Multiple block instances and train/eval specializations share Python
        # code objects. Raise limits only for this explicit opt-in operation.
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 32)
        torch._dynamo.config.accumulated_cache_size_limit = max(
            torch._dynamo.config.accumulated_cache_size_limit, 64)
        modules = [module.model.contraction_layer_H,
                   module.model.contraction_layer_H_init,
                   module.model.contraction_layer_S,
                   module.model.expand_ii['hamiltonian'],
                   module.model.expand_ij['hamiltonian'], module.expand]
        if scope == "blocks":
            backbone = module.model.node_attr_backbone
            modules += [*backbone.blocks, *backbone.xy_blocks]
        for layer in modules:
            layer.compile(backend=backend, dynamic=True, fullgraph=True, mode=mode, options=options)
    else:
        raise ValueError(f"Unknown compilation scope: {scope}")
