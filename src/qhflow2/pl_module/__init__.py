from qhflow2.pl_module.base_module import LitModel
from qhflow2.pl_module.flow_module import LitModel_flow
from qhflow2.pl_module.no_flow_module import LitModel_no_flow
from qhflow2.pl_module.flow_module_EF import LitModel_flow_EF
from qhflow2.pl_module.no_flow_module_new import LitModel_flow as LitModel_no_flow_new


def get_pl_model(conf):
    version = conf.model.version.lower()
    pl_type = conf.get("pl_type", None)
    
    # If pl_type is specified, use it directly
    if pl_type is not None:
        pl_type = pl_type.lower()
        return _get_model_by_pl_type(pl_type)
    
    # Fallback to version-based selection
    return _get_model_by_version(version)


def _get_model_by_pl_type(pl_type):
    """Get model class based on pl_type and inference mode."""
    
    # Define model mappings
    model_mappings = {
        "base".lower(): LitModel,
        "flow".lower(): LitModel_flow,
        "no_flow".lower(): LitModel_no_flow,
        "flow_EF".lower(): LitModel_flow_EF,
        "no_flow_new".lower(): LitModel_no_flow_new,
    }
    pl_type = pl_type.lower()
    if pl_type in model_mappings:
        return model_mappings[pl_type]
    
    raise NotImplementedError(f"The pl_type '{pl_type}' is not implemented.")


def _get_model_by_version(version):
    """Get model class based on version and inference mode."""
    
    if "flow" in version:
        return LitModel_flow
    
    return LitModel
