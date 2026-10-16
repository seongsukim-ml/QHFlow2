"""Device-local, nonpersistent Clebsch–Gordan constants.

Constants follow module device/dtype moves without adding checkpoint keys or
changing parameter registration order.
"""
from e3nn import o3


def register_couplings(module, triples):
    for a, b, c in sorted(set(triples)):
        module.register_buffer(f"_cg_{a}_{b}_{c}", o3.wigner_3j(a, b, c), persistent=False)


def coupling(module, a, b, c, like):
    return getattr(module, f"_cg_{a}_{b}_{c}").to(like)
