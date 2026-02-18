import math
import torch
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius_graph
from e3nn import o3
from e3nn.o3 import Linear

from .layers_v2 import *
from .modules import *
from .utils import *
# from .modules.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20Backbone
from .modules.escn_backbone_v4_no_t import eSCNMDBackbone_ham


"""
use paramcontraction layer with parameters for each path
"""

class QHFlow_escn_v5_1_no_t_SO3(nn.Module):
    """
    Quantum Hamiltonian Flow (QHFlow) model for diffusion-based prediction of molecular properties.
    
    This model extends QHNet with diffusion capabilities and support for different datasets.
    It uses flow-based modeling to generate molecular Hamiltonian matrices.
    
    Args:
        in_node_features (int): Number of input node features (default: 1)
        sh_lmax (int): Maximum spherical harmonics order (default: 4)
        hidden_size (int): Hidden layer size (default: 128)
        bottle_hidden_size (int): Bottleneck hidden layer size (default: 32)
        num_gnn_layers (int): Number of GNN layers (default: 5)
        max_radius (float): Maximum radius for graph construction (default: 12)
        num_nodes (int): Maximum number of nodes (default: 10)
        radius_embed_dim (int): Dimension for radius embeddings (default: 32)
        use_block_S (bool): Whether to use overlap matrix blocks (default: False)
        use_block_H (bool): Whether to use initial Hamiltonian blocks (default: False)
        dataset_type (str): Dataset type - "qh9" or "md17" (default: "qh9")
        **kwargs: Additional keyword arguments
    """
    def __init__(
        self,
        in_node_features=1,
        sh_lmax=4,
        hidden_size=128,
        bottle_hidden_size=32,
        num_gnn_layers=3,
        num_ham_gnn_layers=2,
        max_radius=12,
        num_nodes=10,
        radius_embed_dim=32,
        use_block_S=False,
        use_block_H=False,
        dataset_type="qh9",
        basis="def2-svp",
        trans_type="so2",
        esen_max_radius=5.0,
        max_num_neighbors=32,
        **kwargs,
    ):
        super(QHFlow_escn_v5_1_no_t_SO3, self).__init__()
        
        # Core model parameters
        self.order = sh_lmax
        self.dataset_type = dataset_type
        self.hidden_size = hidden_size
        self.bottle_hidden_size = bottle_hidden_size
        self.radius_embed_dim = radius_embed_dim
        self.max_radius = max_radius
        self.num_gnn_layers = num_gnn_layers
        self.num_ham_gnn_layers = num_ham_gnn_layers
        # Block configuration flags
        self.use_block_S = use_block_S
        self.use_block_H = use_block_H
        self.esen_max_radius = esen_max_radius 
        self.basis = basis
        self.max_num_neighbors = max_num_neighbors
        assert basis in ["def2-svp", "def2-tzvp"], f"Invalid basis: {basis}"
        if basis == "def2-svp":
            self.output_irrep = o3.Irreps("3x0e + 2x1e + 1x2e")
            self.output_matrix_dim = 14
        elif basis == "def2-tzvp":
            self.output_irrep = o3.Irreps("5x0e + 5x1e + 2x2e + 1x3e")
            self.output_matrix_dim = 37
        else:
            raise ValueError(f"Invalid basis: {self.basis}")
        
        self.trans_type = trans_type
        assert trans_type in ["equiformer", "equiformerV2", "so2"], f"Invalid trans_type: {trans_type}"
        self.norm_layer = "layer"

        # Initialize irreducible representations
        self.input_irrep = o3.Irreps(f"{self.hidden_size}x0e")
        self.input_irrep_mat = o3.Irreps(f"{self.hidden_size}x0e + {self.hidden_size}x1e + {self.hidden_size}x2e")
        self.sh_irrep = o3.Irreps.spherical_harmonics(lmax=self.order)
        
        # Hidden representations with alternating parity
        self.hidden_irrep = o3.Irreps(construct_o3irrps(self.hidden_size, order=self.order))
        
        # Bottleneck representations
        self.hidden_bottle_irrep = o3.Irreps(construct_o3irrps(self.bottle_hidden_size, order=self.order))
        
        # Base representations (all even parity)
        self.hidden_irrep_base = o3.Irreps(construct_o3irrps_base(self.hidden_size, order=self.order))
        self.hidden_bottle_irrep_base = o3.Irreps(construct_o3irrps_base(self.bottle_hidden_size, order=self.order))
        
        # self.node_embedding = nn.Embedding(num_nodes, self.hidden_size)
        self.distance_expansion = ExponentialBernsteinRadialBasisFunctions(
            self.radius_embed_dim, self.max_radius
        )
        
        # One-body reduction for processing input matrices
        # self.onebody_reduction = OneBody_Reduction(basis=self.basis)
        self.contraction_layer_H = ParamContraction(
            self.output_irrep,
            self.output_irrep,
            self.input_irrep,
        )
        if self.use_block_S:
            self.contraction_layer_S = ParamContraction(
                self.output_irrep,
                self.output_irrep,
                self.input_irrep,
            )
        if self.use_block_H:
            self.contraction_layer_H_init = ParamContraction(
                self.output_irrep,
                self.output_irrep,
                self.input_irrep,
            )
        
        self.num_fc_layer = 1
        
        # Nonlinear activation configurations
        self.nonlinear_scalars = {1: "ssp", -1: "tanh"}
        self.nonlinear_gates = {1: "ssp", -1: "abs"}
        # Initialize network layers
        self._init_node_vector_backbone()
        
        self._build_layers()
        
        # Initialize output layers
        self._init_output_layers()

        
    def _init_node_vector_backbone(self):
        self.node_attr_backbone = eSCNMDBackbone_ham(
            lmax=self.order,
            # mmax=self.order,
            mmax=self.order,
            sphere_channels=self.hidden_size,
            hidden_channels=self.hidden_size,
            num_layers=self.num_gnn_layers,
            use_block_S=self.use_block_S,
            use_block_H=self.use_block_H,
            use_dataset_embedding=False,
            use_time_embedding=True,
            num_ham_gnn_layers=self.num_ham_gnn_layers,
            cutoff=self.esen_max_radius,
            # otf_graph=True,
        )
    
    def _get_orbital_mask(self):
        return get_orbital_mask(self.basis)
    
    def _build_layers(self):
        """Build the main network layers."""
        self.e3_gnn_node_pair_layer = nn.ModuleList()
        self.e3_gnn_node_layer = nn.ModuleList()
        # Add higher-order layers after start_layer
        for i in range(self.num_ham_gnn_layers):
            self.e3_gnn_node_layer.append(
                SelfNetLayer(
                    irrep_in_node=self.hidden_irrep_base,
                    irrep_bottle_hidden=self.hidden_irrep_base,
                    irrep_out=self.hidden_irrep_base,
                    sh_irrep=self.sh_irrep,
                    edge_attr_dim=self.radius_embed_dim,
                    node_attr_dim=self.hidden_size,
                    resnet=True,
                )
            )

            self.e3_gnn_node_pair_layer.append(
                PairNetLayer(
                    irrep_in_node=self.hidden_irrep_base,
                    irrep_bottle_hidden=self.hidden_irrep_base,
                    irrep_out=self.hidden_irrep_base,
                    sh_irrep=self.sh_irrep,
                    edge_attr_dim=self.radius_embed_dim,
                    node_attr_dim=self.hidden_size,
                    invariant_layers=self.num_fc_layer,
                    invariant_neurons=self.hidden_size,
                    resnet=True,
                )
            )
        
        # Add normalization layer
        self.norm = get_norm_layer(self.norm_layer)(self.hidden_irrep)
            
    def _init_output_layers(self):
        """Initialize output layers for Hamiltonian prediction."""
        self.nonlinear_layer = get_nonlinear("ssp")
        
        # Initialize module dictionaries for different matrix components
        self.expand_ii = nn.ModuleDict()
        self.expand_ij = nn.ModuleDict()
        self.fc_ii = nn.ModuleDict()
        self.fc_ij = nn.ModuleDict()
        self.fc_ii_bias = nn.ModuleDict()
        self.fc_ij_bias = nn.ModuleDict()
        
        # Create expansion and FC layers for Hamiltonian prediction
        for matrix_type in {"hamiltonian"}:
            self._create_matrix_prediction_layers(matrix_type)

        # Linear transformations to bottleneck representation

        self.output_ii = Linear(self.hidden_irrep, self.hidden_bottle_irrep)

        self.irrep_tp_out_node_pair, instruction_node_pair = get_feasible_irrep(
            self.hidden_irrep,
            self.hidden_irrep,
            self.hidden_irrep,
            tp_mode="uuu",
        )
        self.pre_output_ij = o3.TensorProduct(
            self.hidden_irrep,
            self.hidden_irrep,
            self.hidden_irrep,
            instruction_node_pair,
            shared_weights=True,
            internal_weights=True,
        )
        self.output_ij = Linear(self.hidden_irrep, self.hidden_bottle_irrep)

        # for matrix_type in {"hamiltonian_2"}:
        #     self._create_matrix_prediction_layers(matrix_type)        

        # self.output_ii_2 = Linear(self.hidden_irrep, self.hidden_bottle_irrep)
        # self.output_ij_2 = Linear(self.hidden_irrep, self.hidden_bottle_irrep)

    def _create_matrix_prediction_layers(self, matrix_type):
        """Create layers for predicting matrix elements."""
        # Input irrep for expansion
        input_expand_irrep = o3.Irreps(
            f"{self.bottle_hidden_size}x0e + "
            f"{self.bottle_hidden_size}x1e + "
            f"{self.bottle_hidden_size}x2e + "
            f"{self.bottle_hidden_size}x3e + "
            f"{self.bottle_hidden_size}x4e"
        )
        output_irrep = self.output_irrep
        # Diagonal elements (ii)
        self.expand_ii[matrix_type] = Expansion(
            input_expand_irrep, output_irrep, output_irrep
        )
        self.fc_ii[matrix_type] = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.expand_ii[matrix_type].num_path_weight),
        )
        self.fc_ii_bias[matrix_type] = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.expand_ii[matrix_type].num_bias),
        )
        
        # Off-diagonal elements (ij)
        self.expand_ij[matrix_type] = Expansion(
            input_expand_irrep, output_irrep, output_irrep
        )
        self.fc_ij[matrix_type] = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.expand_ij[matrix_type].num_path_weight),
        )
        self.fc_ij_bias[matrix_type] = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.expand_ij[matrix_type].num_bias),
        )

    def get_number_of_parameters(self):
        """Calculate the total number of trainable parameters."""
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def set(self, device):
        """Set the device and initialize orbital masks."""
        self = self.to(device)
        self.orbital_mask = self._get_orbital_mask()
        for key in self.orbital_mask.keys():
            self.orbital_mask[key] = self.orbital_mask[key].to(self.device)

    @property
    def device(self):
        """Get the device of the model parameters."""
        return next(self.parameters()).device
    
    def _process_input_matrices(self, data, H, keep_blocks):
        """Process input matrices based on dataset type and configuration."""
        
        if keep_blocks:
            # node_feats_H = self.contraction_layer_H(H)
            node_feats_H = None

            node_feats_H_init = None
            if self.use_block_H:
                node_feats_H_init = self.contraction_layer_H_init(data.diagonal_init_ham)
            
            node_feats_S = None
            if self.use_block_S:
                node_feats_S = self.contraction_layer_S(data.diagonal_overlap)
        else:
            # node_feats_H = self.contraction_layer_H(H)
            node_feats_H = None
            
            node_feats_H_init = None
            if self.use_block_H:
                node_feats_H_init = self.contraction_layer_H_init(data.init_ham)
            
            node_feats_S = None
            if self.use_block_S:
                node_feats_S = self.contraction_layer_S(data.overlap)
        
        return node_feats_H, node_feats_H_init, node_feats_S
    
    def _process_through_main_layers(
        self,
        data,
    ):        
        """Process features through all network layers."""
        # Process through GNN layers        
        node_attr_R = None
        # num_graphs = data["node_feats_H"].shape[0]
        num_graphs = data.num_graphs
        # if data["node_feats_H"] is not None:
        #     node_feats_H = data["node_feats_H"].reshape(num_graphs,-1, self.hidden_size)
        # else:
        node_feats_H = None
        if self.use_block_H:
            num_graphs = data["node_feats_H"].shape[0]
            node_feats_H_init = data["node_feats_H_init"].reshape(num_graphs, -1, self.hidden_size)
        else:
            node_feats_H_init = None
        if self.use_block_S:
            num_graphs = data["node_feats_S"].shape[0]
            node_feats_S = data["node_feats_S"].reshape(num_graphs, -1, self.hidden_size)
        else:
            node_feats_S = None
        
        middle_features = self.node_attr_backbone(
            data,
            [node_attr_R, node_feats_H, node_feats_H_init, node_feats_S]
            )
        node_attr_R = middle_features["node_embedding"].reshape(data.num_nodes, -1)
        data["node_attr_R"] = node_attr_R
        data["node_attr_R_init"] = middle_features["node_embedding"].narrow(1, 0, 1).squeeze()
        # data["node_attr_R_init"] = middle_features["node_attr_R_init"]
        
        fii = None
        fij = None

        fii_2 = node_attr_R        
        fij_2 = middle_features["xy_embedding"].reshape(middle_features["xy_embedding"].shape[0], -1)

        for layer_idx in range(len(self.e3_gnn_node_layer)):
            cur_fii = self.e3_gnn_node_layer[layer_idx](data, node_attr_R, fii)
            cur_fij = self.e3_gnn_node_pair_layer[layer_idx](data, node_attr_R, fij)
            if layer_idx == 0:
                fii = cur_fii
                fij = cur_fij
            else:
                fii = fii + cur_fii
                fij = fij + cur_fij

        fii = self.output_ii(fii)
        # fij = self.pre_output_ij(fij, fij_2)
        fij = self.output_ij(fij)
        
        # fii_2 = self.output_ii_2(fii_2)
        # fij_2 = self.output_ij_2(fij_2)
        
        return fii, fij, None, None
    
    def _concatenate_features(self, node_concat, node_attr_R):
        """Concatenate features with proper irrep ordering."""
        concat_features = torch.cat(node_concat, dim=-1)
        
        return (
            concat_features
            .index_select(-1, self.hidden_irrep_concat_idx.to(node_attr_R.device))
            .contiguous()
        )
    
    def _vector_to_expand_matrix(
        self,
        data,
        fii,
        fij,
        keep_blocks,
    ):
        """Generate final Hamiltonian predictions."""
        # Generate diagonal matrix elements
        node_attr_R_init = data["node_attr_R_init"]
        full_dst, full_src = data["full_edge_index"]
        transpose_edge_index = data["transpose_edge_index"]

        hamiltonian_diagonal_matrix = self.expand_ii["hamiltonian"](
            fii,
            self.fc_ii["hamiltonian"](node_attr_R_init),
            self.fc_ii_bias["hamiltonian"](node_attr_R_init),
        )
        
        # Generate off-diagonal matrix elements
        node_pair_embedding = torch.cat(
            [node_attr_R_init[full_dst], node_attr_R_init[full_src]],
            dim=-1
        )
        hamiltonian_non_diagonal_matrix = self.expand_ij["hamiltonian"](
            fij,
            self.fc_ij["hamiltonian"](node_pair_embedding),
            self.fc_ij_bias["hamiltonian"](node_pair_embedding),
        )

        # hamiltonian_diagonal_matrix_2 = self.expand_ii["hamiltonian_2"](
        #     fii_2,
        #     self.fc_ii["hamiltonian_2"](node_attr_R_init),
        #     self.fc_ii_bias["hamiltonian_2"](node_attr_R_init),
        # )
        # hamiltonian_non_diagonal_matrix_2 = self.expand_ij["hamiltonian_2"](
        #     fij_2,
        #     self.fc_ij["hamiltonian_2"](node_pair_embedding),
        #     self.fc_ij_bias["hamiltonian_2"](node_pair_embedding),
        # )

        if not keep_blocks:
            # Build complete Hamiltonian matrix
            hamiltonian_matrix = self._build_final_matrix(
                data,
                hamiltonian_diagonal_matrix,
                hamiltonian_non_diagonal_matrix,
            )
            # Ensure Hermitian symmetry
            hamiltonian_matrix = (hamiltonian_matrix + hamiltonian_matrix.transpose(-1, -2)) / 2

            # hamiltonian_matrix_2 = self._build_final_matrix(
            #     data,
            #     hamiltonian_diagonal_matrix_2,
            #     hamiltonian_non_diagonal_matrix_2,
            # )
            # # Ensure Hermitian symmetry
            # hamiltonian_matrix_2 = (hamiltonian_matrix_2 + hamiltonian_matrix_2.transpose(-1, -2)) / 2

            return {
                "hamiltonian": hamiltonian_matrix,
            }
        else:
            # Return block matrices separately
            ret_hamiltonian_diagonal_matrix = (
                hamiltonian_diagonal_matrix + hamiltonian_diagonal_matrix.transpose(-1, -2)
            ) / 2 

            # Apply transpose considering edge symmetry
            ret_hamiltonian_non_diagonal_matrix = (
                hamiltonian_non_diagonal_matrix
                + hamiltonian_non_diagonal_matrix[transpose_edge_index].transpose(-1, -2)
            ) / 2 

            # ret_hamiltonian_diagonal_matrix_2 = (
            #     hamiltonian_diagonal_matrix_2 + hamiltonian_diagonal_matrix_2.transpose(-1, -2)
            # ) / 2 

            # ret_hamiltonian_non_diagonal_matrix_2 = (
            #     hamiltonian_non_diagonal_matrix_2
            #     + hamiltonian_non_diagonal_matrix_2[transpose_edge_index].transpose(-1, -2)
            # ) / 2 

            return {
                "hamiltonian_diagonal_blocks": ret_hamiltonian_diagonal_matrix,
                "hamiltonian_non_diagonal_blocks": ret_hamiltonian_non_diagonal_matrix
            }

    def forward(self, data, H=None, keep_blocks=False):
        """Forward pass of the QHFlow model.
        
        Args:
            data: Molecular data object
            H: Input Hamiltonian matrix
            keep_blocks: Whether to return block matrices separately
        Returns:
            dict: Dictionary containing predicted Hamiltonian matrix/blocks
            if keep_blocks is False:
                "hamiltonian": Hamiltonian matrix
            if keep_blocks is True:
                "hamiltonian_diagonal_blocks": Diagonal blocks of Hamiltonian matrix
                "hamiltonian_non_diagonal_blocks": Non-diagonal blocks of Hamiltonian matrix
        """
        # Process molecular data and extract features
        # node_attr: features from atomic numbers
        # rbf_new  : features from radial basis functions
        # transpose_edge_index: transpose indices for diagonal matrix symmetry
        data["atomic_numbers"] = data.atoms
        data["edge_index"] = radius_graph(data.pos, self.max_radius, data.batch, max_num_neighbors=self.max_num_neighbors)
        data["edge_distance_vec"] = data.pos[data.edge_index[0]] - data.pos[data.edge_index[1]]
        data["edge_distance"] = data["edge_distance_vec"].norm(dim=-1)
        data["full_edge_index"] = radius_graph(data.pos, 10000, data.batch, max_num_neighbors=self.max_num_neighbors)
        data["full_edge_distance_vec"] = data.pos[data.full_edge_index[0]] - data.pos[data.full_edge_index[1]]
        data["full_edge_distance"] = data["full_edge_distance_vec"].norm(dim=-1)
        data["full_edge_sh"] = o3.spherical_harmonics(
            self.sh_irrep,
            data["full_edge_distance_vec"][:, [1, 2, 0]],
            normalize=True,
            normalization="component",
        ).type(data.pos.type())
        data["full_edge_attr"] = self.distance_expansion(data["full_edge_distance"].unsqueeze(-1)).squeeze().type(data.pos.type())
        data["transpose_edge_index"] = self._calculate_transpose_indices(data, data["full_edge_index"])
        # data["node_attr_R"] = self.node_embedding(data["atomic_numbers"])
        data["pbc"] = torch.zeros(len(data), 3, dtype=torch.bool) # Do not use PBC
        # data["cell"] = torch.zeros(data.H.shape[0], 3, 3).to(self.device)
        # data["nedges"] = data.edge_index.shape[1]
        # data["cell_offsets"] = torch.zeros(data["cell"].shape[0]*data.nedges, 3).to(self.device)
        node_feats_H, node_feats_H_init, node_feats_S = self._process_input_matrices(data, H, keep_blocks)
        data["node_feats_H"] = node_feats_H
        data["node_feats_H_init"] = node_feats_H_init
        data["node_feats_S"] = node_feats_S

        # Process through network layers
        fii, fij, fii_2, fij_2 =  self._process_through_main_layers(data)

        # Generate final predictions (build matrix from fii and fij)
        final_result = self._vector_to_expand_matrix(
            data,
            fii,
            fij,
            keep_blocks,
        )

        final_result["node_attr"] = data["node_attr_R"]
        final_result["fii"] = fii
        final_result["fij"] = fij
        final_result["node_attr_init"] = data["node_attr_R_init"]

        return final_result
    
    @staticmethod
    def _calculate_transpose_indices(data, radius_edges):
        """Calculate transpose indices for maintaining edge symmetry."""
        start_edge_index = 0
        all_transpose_index = []
        
        for graph_idx in range(data.ptr.shape[0] - 1):
            num_nodes = data.ptr[graph_idx + 1] - data.ptr[graph_idx]
            
            # Extract edges for current graph
            graph_edge_index = radius_edges[
                :, start_edge_index : start_edge_index + num_nodes * (num_nodes - 1)
            ]
            
            # Convert to local indices
            sub_graph_edge_index = graph_edge_index - data.ptr[graph_idx]
            
            # Calculate transpose mapping
            bias = (sub_graph_edge_index[0] < sub_graph_edge_index[1]).type(torch.int)
            transpose_index = (
                sub_graph_edge_index[0] * (num_nodes - 1)
                + sub_graph_edge_index[1]
                - bias
            )
            transpose_index = transpose_index + start_edge_index
            all_transpose_index.append(transpose_index)
            
            start_edge_index = start_edge_index + num_nodes * (num_nodes - 1)

        return torch.cat(all_transpose_index, dim=-1)

    def _build_final_matrix(self, data, diagonal_matrix, non_diagonal_matrix):
        return build_final_matrix(data, diagonal_matrix, non_diagonal_matrix, self.orbital_mask)
    
    def _split_matrix(self, data):
        """Split input matrix into diagonal and off-diagonal blocks.
        
        Args:
            data: Molecular data object containing the matrix to split
            
        Returns:
            tuple: (diagonal_matrix, non_diagonal_matrix) blocks
        """
        return split_matrix(data, self.orbital_mask, self.device)