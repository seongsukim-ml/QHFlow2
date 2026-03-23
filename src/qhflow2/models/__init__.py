from .QHFlow import QHFlow
from .Real_QHNet import QHNet as Real_QHNet
from .Real_QHNet_qh9 import QHNet as Real_QHNet_qh9


from qhflow2.common.custom_logger import get_logger
logger = get_logger(__name__)

try:
    import fairchem
    USE_FAIRCHEM = True
except ImportError:
    USE_FAIRCHEM = False
    logger.info("fairchem not installed. Install with: uv pip install 'qhflow2[fairchem]'")

if USE_FAIRCHEM:
    # from .QHFlow_so2 import QHFlow_escn
    # from .QHFlow_so2_v2 import QHFlow_escn_v2
    # from .QHFlow_so2_v3 import QHFlow_escn_v3
    # from .QHFlow_so2_v3_1 import QHFlow_escn_v3_1
    # from .QHFlow_so2_v3_2 import QHFlow_escn_v3_2
    # from .QHFlow_so2_v3_3 import QHFlow_escn_v3_3
    # from .QHFlow_so2_v3_4 import QHFlow_escn_v3_4
    # from .QHFlow_so2_v3_5 import QHFlow_escn_v3_5
    # from .QHFlow_so2_v3_6 import QHFlow_escn_v3_6
    # from .QHFlow_so2_v4 import QHFlow_escn_v4
    # from .QHFlow_so2_v4_md17 import QHFlow_escn_v4_md17
    # from .QHFlow_so2_v4_1 import QHFlow_escn_v4_1
    # from .QHFlow_so2_v5 import QHFlow_escn_v5
    from .QHFlow_so2_v5_1 import QHFlow_escn_v5_1   
    from .QHFlow_so2_v5_1_SO2 import QHFlow_escn_v5_1_SO2
    from .QHFlow_so2_v5_1_SO3 import QHFlow_escn_v5_1_SO3
    # from .QHFlow_so2_v5_1_dual import QHFlow_escn_v5_1_dual
    from .QHFlow_so2_v5_1_no_t import QHFlow_escn_v5_1_no_t
    from .QHFlow_so2_v5_1_no_t_SO2 import QHFlow_escn_v5_1_no_t_SO2
    from .QHFlow_so2_v5_1_no_t_SO3 import QHFlow_escn_v5_1_no_t_SO3
    from .QHFlow_so2_v5_1_so2exp import QHFlow_escn_v5_1_so2exp
    from .QHFlow_so2_v5_1_cpexp import QHFlow_escn_v5_1_cpexp
    from .QHFlow_so2_v5_1_tdn import QHFlow_escn_v5_1_tdn
    # from .QHFlow_so2_v5_2 import QHFlow_escn_v5_2

    # from .QHFlow_so2_uma import QHFlow_escn_uma
    # from .QHFlow_so2_uma_v2 import QHFlow_escn_uma_v2

    # from .QHFlow_so2_uma_test import QHFlow_escn_uma_test
    # from .QHFlow_so2_uma_test2 import QHFlow_escn_uma_test2

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
    }
    model_dict ={
        "Real_QHNet".lower():Real_QHNet,
        "Real_QHNet_qh9".lower():Real_QHNet_qh9,
        "QHFlow".lower():QHFlow,
        "QHFlow_qh9".lower():QHFlow,
    }
    
    if USE_FAIRCHEM:
        # Testing
        # model_dict["QHFlow_so2".lower()] = QHFlow_escn
        # model_dict["QHFlow_so2_v2".lower()] = QHFlow_escn_v2
        # model_dict["QHFlow_so2_v3".lower()] = QHFlow_escn_v3
        # model_dict["QHFlow_so2_v3_1".lower()] = QHFlow_escn_v3_1
        # model_dict["QHFlow_so2_v3_2".lower()] = QHFlow_escn_v3_2
        # model_dict["QHFlow_so2_v3_3".lower()] = QHFlow_escn_v3_3
        # model_dict["QHFlow_so2_v3_4".lower()] = QHFlow_escn_v3_4
        # model_dict["QHFlow_so2_v3_5".lower()] = QHFlow_escn_v3_5
        # model_dict["QHFlow_so2_v3_6".lower()] = QHFlow_escn_v3_6
        # model_dict["QHFlow_so2_v4".lower()] = QHFlow_escn_v4
        # model_dict["QHFlow_so2_v4_md17".lower()] = QHFlow_escn_v4_md17
        # model_dict["QHFlow_so2_v4_1".lower()] = QHFlow_escn_v4_1
        # model_dict["QHFlow_so2_uma".lower()] = QHFlow_escn_uma
        # model_dict["QHFlow_so2_uma_v2".lower()] = QHFlow_escn_uma_v2
        # model_dict["QHFlow_so2_uma_test".lower()] = QHFlow_escn_uma_test
        # model_dict["QHFlow_so2_uma_test2".lower()] = QHFlow_escn_uma_test2
        # model_dict["QHFlow_so2_v5".lower()] = QHFlow_escn_v5
        model_dict["QHFlow_so2_v5_1".lower()] = QHFlow_escn_v5_1
        model_dict["QHFlow_so2_v5_1_SO2".lower()] = QHFlow_escn_v5_1_SO2
        model_dict["QHFlow_so2_v5_1_SO3".lower()] = QHFlow_escn_v5_1_SO3
        model_dict["QHFlow_so2_v5_1_no_t".lower()] = QHFlow_escn_v5_1_no_t
        model_dict["QHFlow_so2_v5_1_no_t_SO2".lower()] = QHFlow_escn_v5_1_no_t_SO2
        model_dict["QHFlow_so2_v5_1_no_t_SO3".lower()] = QHFlow_escn_v5_1_no_t_SO3
        model_dict["QHFlow_so2_v5_1_so2exp".lower()] = QHFlow_escn_v5_1_so2exp
        model_dict["QHFlow_so2_v5_1_cpexp".lower()] = QHFlow_escn_v5_1_cpexp
        model_dict["QHFlow_so2_v5_1_tdn".lower()] = QHFlow_escn_v5_1_tdn
        # model_dict["QHFlow_so2_v5_1_dual".lower()] = QHFlow_escn_v5_1_dual
        # model_dict["QHFlow_so2_v5_2".lower()] = QHFlow_escn_v5_2

    if hasattr(args, "so2_bandwidth"):
        model_args["so2_bandwidth"] = args.so2_bandwidth

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