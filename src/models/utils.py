import torch

from .time_embedding import get_time_embedding

def get_orbital_mask(basis):
    """Get orbital masks for different atomic numbers.

    Returns:
        dict: Mapping from atomic number to orbital indices
    """
    assert basis in ["def2-svp", "def2-tzvp"], f"Invalid basis: {basis}"

    if basis == "def2-svp":
        # Define orbital indices for different shell types
        idx_1s_2s = torch.tensor([0, 1])  # s orbitals
        idx_2p = torch.tensor([3, 4, 5])  # p orbitals
        
        # Combine s and p orbitals for light elements (H, He)
        orbital_mask_light = torch.cat([idx_1s_2s, idx_2p])
        
        # Full orbital set for heavier elements (includes d orbitals)
        orbital_mask_heavy = torch.arange(14)
        
        # Create mapping: atomic numbers 1,2 use light mask, others use heavy mask
        orbital_mask = {}
        for atomic_num in range(1, 11):
            orbital_mask[atomic_num] = (
                orbital_mask_light if atomic_num <= 2 else orbital_mask_heavy
            )
    elif basis == "def2-tzvp":
        raise ValueError(f"def2-tzvp is not supported yet")
    
    return orbital_mask

def calculate_transpose_indices(data, radius_edges):
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

def build_final_matrix(
    data,
    diagonal_matrix,
    non_diagonal_matrix,
    orbital_mask,
):
        """Build the final Hamiltonian matrix from diagonal and off-diagonal blocks.
        
        Args:
            data: Molecular data object
            diagonal_matrix: Diagonal block matrices
            non_diagonal_matrix: Off-diagonal block matrices
            orbital_mask: Orbital mask for indexing the matrix
            
        Returns:
            torch.Tensor: Complete Hamiltonian matrix
        """
        final_matrix = []
        dst, src = data.full_edge_index
        
        # Process each molecule in the batch
        for graph_idx in range(data.ptr.shape[0] - 1):
            matrix_block_col = []
            
            # Build matrix for current molecule
            for src_idx in range(data.ptr[graph_idx], data.ptr[graph_idx + 1]):
                matrix_col = []
                
                for dst_idx in range(data.ptr[graph_idx], data.ptr[graph_idx + 1]):
                    if src_idx == dst_idx:
                        # Diagonal block
                        orbital_mask_src = orbital_mask[data.atoms[src_idx].item()]
                        orbital_mask_dst = orbital_mask[data.atoms[dst_idx].item()]
                        
                        matrix_col.append(
                            diagonal_matrix[src_idx]
                            .index_select(-2, orbital_mask_dst)
                            .index_select(-1, orbital_mask_src)
                        )
                    else:
                        # Off-diagonal block
                        mask1 = src == src_idx
                        mask2 = dst == dst_idx
                        index = torch.where(mask1 & mask2)[0].item()
                        
                        orbital_mask_src = orbital_mask[data.atoms[src_idx].item()]
                        orbital_mask_dst = orbital_mask[data.atoms[dst_idx].item()]
                        
                        matrix_col.append(
                            non_diagonal_matrix[index]
                            .index_select(-2, orbital_mask_dst)
                            .index_select(-1, orbital_mask_src)
                        )
                
                matrix_block_col.append(torch.cat(matrix_col, dim=-2))
            final_matrix.append(torch.cat(matrix_block_col, dim=-1))
        
        return torch.stack(final_matrix, dim=0)
    

def split_matrix(data, orbital_mask, device):
    """Split input matrix into diagonal and off-diagonal blocks.
    
    Args:
        data: Molecular data object containing the matrix to split
        
    Returns:
        tuple: (diagonal_matrix, non_diagonal_matrix) blocks
    """
    # Initialize output matrices
    diagonal_matrix = torch.zeros(
        data.atoms.shape[0], 14, 14, dtype=data.pos.dtype, device=device
    )
    non_diagonal_matrix = torch.zeros(
        data.edge_index.shape[1], 14, 14, dtype=data.pos.dtype, device=device
    )

    # Reshape input matrix for batch processing
    batch_size = len(data.ptr) - 1
    data.matrix = data.matrix.reshape(
        batch_size, data.matrix.shape[-1], data.matrix.shape[-1]
    )

    num_atoms = 0
    num_edges = 0
    
    # Process each molecule in the batch
    for graph_idx in range(batch_size):
        # Calculate orbital slices for current molecule
        slices = calculate_orbital_slices(data, graph_idx, orbital_mask)
        
        # Extract diagonal blocks
        for node_idx in range(data.ptr[graph_idx], data.ptr[graph_idx + 1]):
            local_node_idx = node_idx - num_atoms
            orb_mask = orbital_mask[data.atoms[node_idx].item()]
            
            diagonal_matrix[node_idx][orb_mask][:, orb_mask] = data.matrix[graph_idx][
                slices[local_node_idx] : slices[local_node_idx + 1],
                slices[local_node_idx] : slices[local_node_idx + 1],
            ]

        # Extract off-diagonal blocks
        for edge_index_idx in range(num_edges, data.edge_index.shape[1]):
            dst, src = data.edge_index[:, edge_index_idx]
            
            # Check if edge belongs to current graph
            if dst > data.ptr[graph_idx + 1] or src > data.ptr[graph_idx + 1]:
                break
                
            num_edges += 1
            orb_mask_dst = orbital_mask[data.atoms[dst].item()]
            orb_mask_src = orbital_mask[data.atoms[src].item()]
            
            graph_dst, graph_src = dst - num_atoms, src - num_atoms
            non_diagonal_matrix[edge_index_idx][orb_mask_dst][:, orb_mask_src] = (
                data.matrix[graph_idx][
                    slices[graph_dst] : slices[graph_dst + 1],
                    slices[graph_src] : slices[graph_src + 1],
                ]
            )

        num_atoms += data.ptr[graph_idx + 1] - data.ptr[graph_idx]
        
    return diagonal_matrix, non_diagonal_matrix

def calculate_orbital_slices(data, graph_idx, orbital_mask):
    """Calculate orbital slicing indices for a specific graph."""
    slices = [0]
    for atom_idx in data.atoms[range(data.ptr[graph_idx], data.ptr[graph_idx + 1])]:
        orbital_count = len(orbital_mask[atom_idx.item()])
        slices.append(slices[-1] + orbital_count)
    return slices
