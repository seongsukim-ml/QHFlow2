"""Optional compilation without changing parameter or checkpoint names."""
import torch


def compile_flow(module, scope="heads"):
    """Compile only call paths; nn.Module.compile preserves state_dict keys."""
    if scope == "model":
        module.model.compile(dynamic=True, fullgraph=False)
    elif scope == "muon":
        # Benchmark-only process scope; state and update rules remain native.
        import torch.optim._muon as native
        native._zeropower_via_newtonschulz = torch.compile(
            native._zeropower_via_newtonschulz, dynamic=True, fullgraph=True)
    elif scope == "heads":
        modules = [module.model.contraction_layer_H,
                   module.model.contraction_layer_H_init,
                   module.model.contraction_layer_S,
                   module.model.expand_ii['hamiltonian'],
                   module.model.expand_ij['hamiltonian'], module.expand]
        for layer in modules:
            layer.compile(dynamic=True, fullgraph=True)
    else:
        raise ValueError(f"Unknown compilation scope: {scope}")
