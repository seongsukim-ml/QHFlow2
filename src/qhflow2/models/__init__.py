from .QHFlow import QHFlow
from .Real_QHNet import QHNet as Real_QHNet
from .Real_QHNet_qh9 import QHNet as Real_QHNet_qh9


from qhflow2.common.custom_logger import get_logger
logger = get_logger(__name__)

from .QHFlow_so2_v5_1 import QHFlow_escn_v5_1
from .QHFlow_so2_v5_1_SO2 import QHFlow_escn_v5_1_SO2
from .QHFlow_so2_v5_1_SO3 import QHFlow_escn_v5_1_SO3
from .QHFlow_so2_v5_1_no_t import QHFlow_escn_v5_1_no_t
from .QHFlow_so2_v5_1_no_t_SO2 import QHFlow_escn_v5_1_no_t_SO2
from .QHFlow_so2_v5_1_no_t_SO3 import QHFlow_escn_v5_1_no_t_SO3
from .QHFlow_so2_v5_1_so2exp import QHFlow_escn_v5_1_so2exp
from .QHFlow_so2_v5_1_cpexp import QHFlow_escn_v5_1_cpexp
from .QHFlow_so2_v5_1_tdn import QHFlow_escn_v5_1_tdn
from .QHFlow_so2_v5_1_perl import QHFlow_escn_v5_1_perl
from .QHFlow_so2_v5_1_transfer import QHFlow_escn_v5_1_transfer

__all__ = ["get_model", "get_default_model_args", "default_model_args_qh9", "default_model_args_md17"]

def get_model(args):
    model_args = {
        "in_node_features": getattr(args, "in_node_features", 1),
        "sh_lmax": getattr(args, "sh_lmax", 4),
        "hidden_size": getattr(args, "hidden_size", 128),
        "bottle_hidden_size": getattr(args, "bottle_hidden_size", 32),
        "num_gnn_layers": getattr(args, "num_gnn_layers", 5),
        "max_radius": getattr(args, "max_radius", 15),
        "num_nodes": getattr(args, "num_nodes", 10),
        "radius_embed_dim": getattr(args, "radius_embed_dim", 16),
        "max_T": getattr(args, "max_T", 15),
        "use_block_S": getattr(args, "use_block_S", True),
        "use_block_H": getattr(args, "use_block_H_fix", False),
        "ham_dim": getattr(args, "ham_dim", 24),
        "ham_hidden": getattr(args, "ham_hidden", 24 * 24 // 2),
        "dataset_type": getattr(args, "dataset_type", "qh9"),
        "num_ham_gnn_layers": getattr(args, "num_ham_gnn_layers", 2),
        "esen_max_radius": getattr(args, "esen_max_radius", 5.0),
        "basis": getattr(args, "basis", "def2-svp"),
        "max_num_neighbors": getattr(args, "max_num_neighbors", 32),
    }
    model_dict ={
        "Real_QHNet".lower():Real_QHNet,
        "Real_QHNet_qh9".lower():Real_QHNet_qh9,
        "QHFlow".lower():QHFlow,
        "QHFlow_qh9".lower():QHFlow,
    }
    
    model_dict["QHFlow_so2_v5_1".lower()] = QHFlow_escn_v5_1
    model_dict["QHFlow_so2_v5_1_SO2".lower()] = QHFlow_escn_v5_1_SO2
    model_dict["QHFlow_so2_v5_1_SO3".lower()] = QHFlow_escn_v5_1_SO3
    model_dict["QHFlow_so2_v5_1_no_t".lower()] = QHFlow_escn_v5_1_no_t
    model_dict["QHFlow_so2_v5_1_no_t_SO2".lower()] = QHFlow_escn_v5_1_no_t_SO2
    model_dict["QHFlow_so2_v5_1_no_t_SO3".lower()] = QHFlow_escn_v5_1_no_t_SO3
    model_dict["QHFlow_so2_v5_1_so2exp".lower()] = QHFlow_escn_v5_1_so2exp
    model_dict["QHFlow_so2_v5_1_cpexp".lower()] = QHFlow_escn_v5_1_cpexp
    model_dict["QHFlow_so2_v5_1_tdn".lower()] = QHFlow_escn_v5_1_tdn
    model_dict["QHFlow_so2_v5_1_perl".lower()] = QHFlow_escn_v5_1_perl
    model_dict["QHFlow_so2_v5_1_transfer".lower()] = QHFlow_escn_v5_1_transfer

    if hasattr(args, "expand_lmax"):
        model_args["expand_lmax"] = args.expand_lmax

    if hasattr(args, "so2_bandwidth"):
        model_args["so2_bandwidth"] = args.so2_bandwidth

    # Functional/basis conditioning
    if hasattr(args, "num_functionals"):
        model_args["num_functionals"] = args.num_functionals
    if hasattr(args, "cond_method"):
        model_args["cond_method"] = args.cond_method
    if hasattr(args, "heads"):
        model_args["heads"] = args.heads

    if args == None:
        print("args is None, using QHFlow for default")
        return model_dict["QHFlow"](**model_args)

    model_name = args.version.lower()
    if "uma" in model_name:
        model_args["uma_type"] = getattr(args, "uma_type", "uma-s-1p1")        
        model_args["uma_freeze"] = getattr(args, "uma_freeze", False)
    
    logger.info(f"model_args: {model_args}")
    model = model_dict.get(model_name, None)
    
    if model is None:
        raise NotImplementedError(f"the version {args.version} is not implemented.")
    else:
        return model(**model_args)

# For debugging
def get_default_model_args(dataset_type):
    if dataset_type == "qh9":
        return default_model_args_qh9
    elif dataset_type == "md17":
        return default_model_args_md17
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")

default_model_args_qh9 = {
    "in_node_features": 1,
    "sh_lmax": 4,
    "hidden_size": 128,
    "bottle_hidden_size": 32,
    "num_gnn_layers": 5,
    "max_radius": 15,
    "num_nodes": 10,
    "radius_embed_dim": 16,
    "max_T": 15,
    "use_block_S": True,
    "use_block_H": True,
    "ham_dim": 24,
    "ham_hidden": 24 * 24 // 2,
    "dataset_type": "qh9",
}

default_model_args_md17 = {
    "in_node_features": 1,
    "sh_lmax": 4,
    "hidden_size": 128,
    "bottle_hidden_size": 32,
    "num_gnn_layers": 5,
    "max_radius": 15,
    "num_nodes": 10,
    "radius_embed_dim": 16,
    "max_T": 15,
    "use_block_S": False,
    "use_block_H": True,
    "ham_dim": 24,
    "ham_hidden": 24 * 24 // 2,
    "dataset_type": "md17",
}