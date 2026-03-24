"""Transfer-capable QHFlow model with functional conditioning and multi-head support.

Inherits from QHFlow_escn_v5_1 and adds:
  - FiLM / AdaLN conditioning on DFT functional
  - Multi-head output (e.g., hamiltonian + overlap)
  - Freeze utilities for fine-tuning transfer
  - load_pretrained() for loading baseline checkpoints with shape mismatch handling
"""

import logging
import torch
from torch import nn
from e3nn import o3
from e3nn.o3 import Linear

from .QHFlow_so2_v5_1 import QHFlow_escn_v5_1
from .layers_v2 import Expansion, get_feasible_irrep
from .utils import get_orbital_mask
from .modules.conditioning import ConditionEmbedding, build_conditioning

log = logging.getLogger(__name__)


class QHFlow_escn_v5_1_transfer(QHFlow_escn_v5_1):
    """QHFlow with functional/basis transfer conditioning.

    Backward compatible: with ``cond_method="none"`` and ``heads=["hamiltonian"]``,
    behaves identically to the base ``QHFlow_escn_v5_1``.

    Extra Args (on top of base class):
        num_functionals: Number of discrete functionals in the embedding table.
        cond_method: ``"film"`` | ``"adaln"`` | ``"none"``.
        heads: List of output head names (default ``["hamiltonian"]``).
    """

    def __init__(
        self,
        *,
        num_functionals: int = 4,
        cond_method: str = "film",
        heads: list[str] | None = None,
        # All base-class args forwarded via **kwargs
        **kwargs,
    ):
        # Store transfer-specific args before super().__init__ calls _init_*
        self._transfer_num_functionals = num_functionals
        self._transfer_cond_method = cond_method
        self._transfer_heads = heads or ["hamiltonian"]

        super().__init__(**kwargs)

        # Attributes set after super().__init__
        self.cond_method = cond_method
        self.num_functionals = num_functionals
        self.head_names = self._transfer_heads

    # ──────────────────────────────────────────────────────────────
    # Override: backbone with optional functional embedding input
    # ──────────────────────────────────────────────────────────────

    def _init_node_vector_backbone(self):
        from .modules.escn_backbone_v4 import eSCNMDBackbone_ham

        self.node_attr_backbone = eSCNMDBackbone_ham(
            lmax=self.order,
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
            use_functional_embedding=(self._transfer_cond_method != "none"),
        )

    # ──────────────────────────────────────────────────────────────
    # Override: output layers — multi-head + wider fc for cond_emb
    # ──────────────────────────────────────────────────────────────

    def _init_output_layers(self):
        self.nonlinear_layer = None  # not used in this path

        self.expand_ii = nn.ModuleDict()
        self.expand_ij = nn.ModuleDict()
        self.fc_ii = nn.ModuleDict()
        self.fc_ij = nn.ModuleDict()
        self.fc_ii_bias = nn.ModuleDict()
        self.fc_ij_bias = nn.ModuleDict()

        for head_name in self._transfer_heads:
            self._create_matrix_prediction_layers(head_name)

        self.output_ii = Linear(self.hidden_irrep, self.hidden_bottle_irrep)

        irrep_tp_out, instr = get_feasible_irrep(
            self.hidden_irrep, self.hidden_irrep, self.hidden_irrep, tp_mode="uuu",
        )
        self.pre_output_ij = o3.TensorProduct(
            self.hidden_irrep, self.hidden_irrep, self.hidden_irrep,
            instr, shared_weights=True, internal_weights=True,
        )
        self.output_ij = Linear(self.hidden_irrep, self.hidden_bottle_irrep)

        # ── Conditioning modules ──
        cond = self._transfer_cond_method
        nf = self._transfer_num_functionals
        if cond != "none":
            self.cond_embed = ConditionEmbedding(nf, self.hidden_size)
            self.cond_post_backbone = build_conditioning(cond, self.hidden_size, self.hidden_size)
        else:
            self.cond_embed = None
            self.cond_post_backbone = None

    def _create_matrix_prediction_layers(self, matrix_type):
        cond_dim = self.hidden_size if self._transfer_cond_method != "none" else 0
        fc_ii_in = self.hidden_size + cond_dim
        fc_ij_in = self.hidden_size * 2 + cond_dim

        input_expand_irrep = o3.Irreps(
            " + ".join(f"{self.bottle_hidden_size}x{l}e" for l in range(self.order + 1))
        )
        out_irrep = self.output_irrep

        self.expand_ii[matrix_type] = Expansion(input_expand_irrep, out_irrep, out_irrep)
        self.fc_ii[matrix_type] = nn.Sequential(
            nn.Linear(fc_ii_in, self.hidden_size), nn.SiLU(),
            nn.Linear(self.hidden_size, self.expand_ii[matrix_type].num_path_weight),
        )
        self.fc_ii_bias[matrix_type] = nn.Sequential(
            nn.Linear(fc_ii_in, self.hidden_size), nn.SiLU(),
            nn.Linear(self.hidden_size, self.expand_ii[matrix_type].num_bias),
        )

        self.expand_ij[matrix_type] = Expansion(input_expand_irrep, out_irrep, out_irrep)
        self.fc_ij[matrix_type] = nn.Sequential(
            nn.Linear(fc_ij_in, self.hidden_size), nn.SiLU(),
            nn.Linear(self.hidden_size, self.expand_ij[matrix_type].num_path_weight),
        )
        self.fc_ij_bias[matrix_type] = nn.Sequential(
            nn.Linear(fc_ij_in, self.hidden_size), nn.SiLU(),
            nn.Linear(self.hidden_size, self.expand_ij[matrix_type].num_bias),
        )

    # ──────────────────────────────────────────────────────────────
    # Override: _process_through_main_layers — inject cond embedding
    # ──────────────────────────────────────────────────────────────

    def _get_cond_embedding(self, data):
        if self.cond_embed is None:
            return None
        if not hasattr(data, "functional_id") and "functional_id" not in data:
            return None
        return self.cond_embed(data["functional_id"][data["batch"]])

    def _process_through_main_layers(self, data):
        node_attr_R = None
        num_graphs = data["node_feats_H"].shape[0]
        node_feats_H = data["node_feats_H"].reshape(num_graphs, -1, self.hidden_size)
        node_feats_H_init = (
            data["node_feats_H_init"].reshape(num_graphs, -1, self.hidden_size)
            if self.use_block_H else None
        )
        node_feats_S = (
            data["node_feats_S"].reshape(num_graphs, -1, self.hidden_size)
            if self.use_block_S else None
        )

        # Build ham_features — optionally include functional embedding
        cond_emb = self._get_cond_embedding(data)
        ham_features = [node_attr_R, node_feats_H, node_feats_H_init, node_feats_S]
        if self.cond_method != "none":
            if cond_emb is None:
                cond_emb = torch.zeros(
                    data.num_nodes, self.hidden_size,
                    device=node_feats_H.device, dtype=node_feats_H.dtype,
                )
            ham_features.append(cond_emb)

        middle_features = self.node_attr_backbone(data, ham_features)
        node_attr_R = middle_features["node_embedding"].reshape(data.num_nodes, -1)

        # Apply FiLM / AdaLN after backbone
        if cond_emb is not None and self.cond_post_backbone is not None:
            node_attr_R = self.cond_post_backbone(node_attr_R, cond_emb)

        data["node_attr_R"] = node_attr_R
        data["node_attr_R_init"] = middle_features["node_embedding"].narrow(1, 0, 1).squeeze()
        data["_cond_emb"] = cond_emb

        fii = None
        fij = None
        fij_2 = middle_features["xy_embedding"].reshape(
            middle_features["xy_embedding"].shape[0], -1
        )

        for layer_idx in range(len(self.e3_gnn_node_layer)):
            cur_fii = self.e3_gnn_node_layer[layer_idx](data, node_attr_R, fii)
            cur_fij = self.e3_gnn_node_pair_layer[layer_idx](data, node_attr_R, fij)
            if layer_idx == 0:
                fii, fij = cur_fii, cur_fij
            else:
                fii = fii + cur_fii
                fij = fij + cur_fij

        fii = self.output_ii(fii)
        fij = self.pre_output_ij(fij, fij_2)
        fij = self.output_ij(fij)

        return fii, fij, None, None

    # ──────────────────────────────────────────────────────────────
    # Override: _vector_to_expand_matrix — multi-head + cond concat
    # ──────────────────────────────────────────────────────────────

    def _predict_head(self, head_name, fii, fij, fc_ii_input, fc_ij_input):
        diag = self.expand_ii[head_name](
            fii,
            self.fc_ii[head_name](fc_ii_input),
            self.fc_ii_bias[head_name](fc_ii_input),
        )
        off_diag = self.expand_ij[head_name](
            fij,
            self.fc_ij[head_name](fc_ij_input),
            self.fc_ij_bias[head_name](fc_ij_input),
        )
        return diag, off_diag

    def _vector_to_expand_matrix(self, data, fii, fij, keep_blocks):
        node_attr_R_init = data["node_attr_R_init"]
        full_dst, full_src = data["full_edge_index"]
        transpose_edge_index = data["transpose_edge_index"]

        cond_emb = data.get("_cond_emb") if isinstance(data, dict) else getattr(data, "_cond_emb", None)

        if cond_emb is not None:
            fc_ii_input = torch.cat([node_attr_R_init, cond_emb], dim=-1)
            fc_ij_input = torch.cat(
                [node_attr_R_init[full_dst], node_attr_R_init[full_src], cond_emb[full_dst]],
                dim=-1,
            )
        else:
            fc_ii_input = node_attr_R_init
            fc_ij_input = torch.cat(
                [node_attr_R_init[full_dst], node_attr_R_init[full_src]], dim=-1,
            )

        result = {}
        for head_name in self.head_names:
            diag, off_diag = self._predict_head(head_name, fii, fij, fc_ii_input, fc_ij_input)

            if not keep_blocks:
                mat = self._build_final_matrix(data, diag, off_diag)
                mat = (mat + mat.transpose(-1, -2)) / 2
                result[head_name] = mat
            else:
                diag_sym = (diag + diag.transpose(-1, -2)) / 2
                off_diag_sym = (
                    off_diag + off_diag[transpose_edge_index].transpose(-1, -2)
                ) / 2
                result[f"{head_name}_diagonal_blocks"] = diag_sym
                result[f"{head_name}_non_diagonal_blocks"] = off_diag_sym

        return result

    # ──────────────────────────────────────────────────────────────
    # Freeze / transfer utilities
    # ──────────────────────────────────────────────────────────────

    def freeze_backbone(self):
        """Freeze backbone + ham layers. Conditioning + output heads remain trainable."""
        for param in self.node_attr_backbone.parameters():
            param.requires_grad = False
        for param in self.contraction_layer_H.parameters():
            param.requires_grad = False
        if self.use_block_H:
            for param in self.contraction_layer_H_init.parameters():
                param.requires_grad = False
        if self.use_block_S:
            for param in self.contraction_layer_S.parameters():
                param.requires_grad = False
        for layer in self.e3_gnn_node_layer:
            for param in layer.parameters():
                param.requires_grad = False
        for layer in self.e3_gnn_node_pair_layer:
            for param in layer.parameters():
                param.requires_grad = False
        for param in self.output_ii.parameters():
            param.requires_grad = False
        for param in self.pre_output_ij.parameters():
            param.requires_grad = False
        for param in self.output_ij.parameters():
            param.requires_grad = False

    def freeze_all_except_conditioning(self):
        """Freeze everything, then unfreeze only conditioning modules."""
        for param in self.parameters():
            param.requires_grad = False
        if self.cond_embed is not None:
            for param in self.cond_embed.parameters():
                param.requires_grad = True
        if self.cond_post_backbone is not None:
            for param in self.cond_post_backbone.parameters():
                param.requires_grad = True

    def freeze_all_except_conditioning_and_heads(self):
        """Freeze backbone, unfreeze conditioning + output heads."""
        self.freeze_all_except_conditioning()
        for mod in (self.expand_ii, self.expand_ij, self.fc_ii, self.fc_ij,
                    self.fc_ii_bias, self.fc_ij_bias):
            for param in mod.parameters():
                param.requires_grad = True

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def add_functional(self, functional_name: str) -> int:
        """Add a new functional to the embedding table (mean-init)."""
        from .utils import FUNCTIONAL_REGISTRY
        if functional_name in FUNCTIONAL_REGISTRY:
            return FUNCTIONAL_REGISTRY[functional_name]
        old_emb = self.cond_embed.embed
        old_w = old_emb.weight.data
        new_id = old_emb.num_embeddings
        new_emb = nn.Embedding(new_id + 1, old_emb.embedding_dim, device=old_w.device)
        new_emb.weight.data[:new_id] = old_w
        new_emb.weight.data[new_id] = old_w.mean(dim=0)
        self.cond_embed.embed = new_emb
        self.num_functionals = new_id + 1
        FUNCTIONAL_REGISTRY[functional_name] = new_id
        return new_id

    # ──────────────────────────────────────────────────────────────
    # load_pretrained: handle baseline → transfer shape mismatches
    # ──────────────────────────────────────────────────────────────

    def load_pretrained(self, ckpt_path: str, strict: bool = False):
        """Load weights from a (possibly baseline) checkpoint.

        Handles fc_ii/fc_ij dimension mismatches by zero-padding the
        conditioning portion of wider Linear layers.
        """
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "state_dict" in ckpt:
            src_sd = {
                k.replace("model.", "", 1) if k.startswith("model.") else k: v
                for k, v in ckpt["state_dict"].items()
            }
        else:
            src_sd = ckpt

        tgt_sd = self.state_dict()
        matched, skipped, missing, unexpected = [], [], [], []

        for key in tgt_sd:
            if key not in src_sd:
                missing.append(key)
                continue
            src_t, tgt_t = src_sd[key], tgt_sd[key]
            if src_t.shape == tgt_t.shape:
                tgt_sd[key] = src_t
                matched.append(key)
            elif src_t.dim() == 2 and tgt_t.dim() == 2 and src_t.shape[0] == tgt_t.shape[0] and src_t.shape[1] < tgt_t.shape[1]:
                new_w = torch.zeros_like(tgt_t)
                new_w[:, :src_t.shape[1]] = src_t
                tgt_sd[key] = new_w
                matched.append(key)
                log.warning(f"  Zero-padded: {key} weight ({src_t.shape[1]} → {tgt_t.shape[1]} in_features).")
            elif src_t.dim() == 1 and tgt_t.dim() == 1 and src_t.shape[0] < tgt_t.shape[0]:
                new_b = torch.zeros_like(tgt_t)
                new_b[:src_t.shape[0]] = src_t
                tgt_sd[key] = new_b
                matched.append(key)
                log.warning(f"  Zero-padded: {key} bias ({src_t.shape[0]} → {tgt_t.shape[0]}).")
            else:
                skipped.append((key, src_t.shape, tgt_t.shape))

        unexpected = [k for k in src_sd if k not in tgt_sd]
        self.load_state_dict(tgt_sd, strict=False)

        log.info(f"load_pretrained: {len(matched)} matched, {len(missing)} new, "
                 f"{len(unexpected)} unused, {len(skipped)} shape-mismatch")
        for k, s, t in skipped:
            log.warning(f"  Skipped: {k}: ckpt {s} vs model {t}")
        if missing:
            log.info(f"  New params (default init): {missing}")

        if strict and (missing or unexpected or skipped):
            raise RuntimeError(f"Strict: {len(missing)} missing, {len(unexpected)} unexpected, {len(skipped)} mismatch")
        return missing, unexpected
