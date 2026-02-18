import torch
import torch.nn as nn
import math
from torch_scatter import scatter
from common.dft_utils import init_pyscf_mf
from common.units import *
from .layers_v2 import *

class SO3_Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, lmax: int) -> None:
        """
        1. Use `torch.einsum` to prevent slicing and concatenation
        2. Need to specify some behaviors in `no_weight_decay` and weight initialization.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lmax = lmax

        self.weight = torch.nn.Parameter(
            torch.randn((self.lmax + 1), out_features, in_features)
        )
        bound = 1 / math.sqrt(self.in_features)
        torch.nn.init.uniform_(self.weight, -bound, bound)
        self.bias = torch.nn.Parameter(torch.zeros(out_features))

        expand_index = torch.zeros([(lmax + 1) ** 2]).long()
        for lval in range(lmax + 1):
            start_idx = lval**2
            length = 2 * lval + 1
            expand_index[start_idx : (start_idx + length)] = lval
        self.register_buffer("expand_index", expand_index, persistent=False)

    def forward(self, input_embedding):
        weight = torch.index_select(
            self.weight, dim=0, index=self.expand_index
        )  # [(L_max + 1) ** 2, C_out, C_in]
        out = torch.einsum(
            "bmi, moi -> bmo", input_embedding, weight
        )  # [N, (L_max + 1) ** 2, C_out]
        bias = self.bias.view(1, 1, self.out_features)
        out[:, 0:1, :] = out.narrow(1, 0, 1) + bias
        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, lmax={self.lmax})"


class MLP_EFS_head(nn.Module):
    def __init__(self,
        backbone,
        reduce: str = "sum",
        regress_stress: bool = False,
        regress_forces: bool = False,
        direct_forces: bool = False,
    ):
        super().__init__()
        # self.backbone = backbone
        # assert not self.regress_forces, "regress_forces is not supported"
        self.regress_stress = regress_stress
        assert not self.regress_stress, "regress_stress is not supported, yet"
        self.regress_forces = regress_forces
        self.reduce = reduce
        self.direct_forces = direct_forces
        self.force_block = SO3_Linear(backbone.hidden_size, 1, lmax=1)

        self.sphere_channels = backbone.hidden_size
        self.hidden_channels = backbone.hidden_size
        self.energy_block = nn.Sequential(
            nn.Linear(self.sphere_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, 1, bias=True),
        )

    def forward(self, batch, output):
        natoms = output["node_attr"].shape[0]
        _input_attr = output["node_attr"].reshape(natoms, -1, self.sphere_channels)

        _input = _input_attr.narrow(1, 0, 1).squeeze(1)
        _output = self.energy_block(_input)

        node_energy = _output.view(-1, 1)
        energy_part = scatter(node_energy, batch["batch_full"], dim=0, reduce=self.reduce)
        energy_part = energy_part * 100

        output["energy"] = energy_part

        if self.regress_stress:
            pass
        elif self.regress_forces:
            if self.direct_forces:
                forces = self.force_block(_input_attr.narrow(1, 0, 4))
                forces = forces.narrow(1, 1, 3)
                forces = forces.view(-1, 3).contiguous()
            else:
                forces = (
                    -1
                    * torch.autograd.grad(
                        energy_part.sum(), batch["pos"], create_graph=self.training
                    )[0]
                )
            output["forces"] = forces
        return output


class MLP_EFS_head2(nn.Module):
    def __init__(self,
        backbone,
        reduce: str = "sum",
        regress_stress: bool = False,
        regress_forces: bool = False,
        direct_forces: bool = False,
    ):
        super().__init__()
        # self.backbone = backbone
        # assert not self.regress_forces, "regress_forces is not supported"
        self.regress_stress = regress_stress
        assert not self.regress_stress, "regress_stress is not supported, yet"
        self.regress_forces = regress_forces
        self.reduce = reduce
        self.direct_forces = direct_forces
        self.force_block = SO3_Linear(backbone.hidden_size, 1, lmax=1)

        self.sphere_channels = backbone.hidden_size
        self.hidden_channels = backbone.hidden_size
        self.energy_block = nn.Sequential(
            nn.Linear(self.sphere_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, 1, bias=True),
        )

        self.atomrefs = torch.zeros(20)
        self.atomrefs[1] = -0.46437566
        self.atomrefs[6] = -37.19576656
        self.atomrefs[7] = -53.76368628
        self.atomrefs[8] = -73.82292926
        self.atomrefs[9] = -98.16854799
        self.atomrefs = self.atomrefs.cuda()


    def forward(self, batch, output):
        natoms = output["node_attr"].shape[0]
        _input_attr = output["node_attr"].reshape(natoms, -1, self.sphere_channels)

        _input = _input_attr.narrow(1, 0, 1).squeeze(1)
        _output = self.energy_block(_input)

        node_energy = _output.view(-1, 1)
        energy_part = scatter(node_energy, batch["batch_full"], dim=0, reduce=self.reduce)
        energy_part = energy_part * 100

        atomrefs_value = self.atomrefs[batch.atoms]
        atomrefs_value = scatter(atomrefs_value, batch["batch_full"], dim=0, reduce="sum")

        output["energy"] = energy_part + atomrefs_value

        if self.regress_stress:
            pass
        elif self.regress_forces:
            if self.direct_forces:
                forces = self.force_block(_input_attr.narrow(1, 0, 4))
                forces = forces.narrow(1, 1, 3)
                forces = forces.view(-1, 3).contiguous()
            else:
                forces = (
                    -1
                    * torch.autograd.grad(
                        energy_part.sum(), batch["pos"], create_graph=self.training
                    )[0]
                )
            output["forces"] = forces
        return output


class MLP_EFS_head_ref(nn.Module):
    def __init__(self,
        backbone,
        reduce: str = "sum",
        regress_stress: bool = False,
        regress_forces: bool = False,
        direct_forces: bool = False,
        unit: str = "ang",
        basis: str = "def2svp",
        xc: str = "b3lyp",
    ):
        super().__init__()
        # self.backbone = backbone
        # assert not self.regress_forces, "regress_forces is not supported"
        self.regress_stress = regress_stress
        assert not self.regress_stress, "regress_stress is not supported, yet"
        self.regress_forces = regress_forces
        self.reduce = reduce
        self.direct_forces = direct_forces
        self.force_block = SO3_Linear(backbone.hidden_size, 1, lmax=1)

        self.unit = unit
        self.basis = basis
        self.xc = xc

        self.sphere_channels = backbone.hidden_size
        self.hidden_channels = backbone.hidden_size
        self.energy_block = nn.Sequential(
            nn.Linear(self.sphere_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, 1, bias=True),
        )

    def build_final_matrix(self, batch, diagonal_matrix, non_diagonal_matrix, transform=True, convention="back2pyscf", dtype=torch.float64):
        pass

    def cal_orbital_and_energies(self, overlap_matrix, full_hamiltonian, method="eigh", tol=0):
        pass

    def forward(self, batch, output):
        natoms = output["node_attr"].shape[0]
        _input_attr = output["node_attr"].reshape(natoms, -1, self.sphere_channels)

        _input = _input_attr.narrow(1, 0, 1).squeeze(1)
        _output = self.energy_block(_input)

        node_energy = _output.view(-1, 1)
        energy_part = scatter(node_energy, batch["batch_full"], dim=0, reduce=self.reduce)

        output["hamiltonian"] = self.build_final_matrix(
            batch,
            output["hamiltonian_diagonal_blocks"],
            output["hamiltonian_non_diagonal_blocks"],
            transform=True,
            convention="back2pyscf",
            dtype=torch.float64,
        )
        
        output["overlap"] = self.build_final_matrix(
            batch,
            batch.diagonal_overlap,
            batch.non_diagonal_overlap,
            transform=True,
            convention="back2pyscf",
            dtype=torch.float64,
        )

        output["orbital_energies"], output["orbital_coefficients"] = (
            self.cal_orbital_and_energies(output["overlap"], output["hamiltonian"])
        )

        dm = []
        energy_dft = []
        for i in range(len(batch)):
            num_orb = int(batch.atoms[batch.ptr[i] : batch.ptr[i+1]].sum() / 2)
            sliced_orbital_coefficients = output["orbital_coefficients"][i][:, :num_orb]
            dm.append(sliced_orbital_coefficients.matmul(sliced_orbital_coefficients.T)* 2)            
            mf = init_pyscf_mf(
                batch.atoms[batch.ptr[i] : batch.ptr[i+1]].detach().cpu(),
                batch.pos[batch.ptr[i] : batch.ptr[i+1]].detach().cpu(),
                unit=self.unit,
                xc=self.xc,
                basis=self.basis,
                )
            cur_energy = mf.energy_tot(dm=dm[i].detach().cpu().numpy())
            energy_dft.append(cur_energy)
        
        output["dm"] = dm
        energy_dft = torch.tensor(energy_dft, device=batch.pos.device, dtype=energy_part.dtype)

        output["energy"] = energy_part + energy_dft

        if self.regress_stress:
            pass
        elif self.regress_forces:
            if self.direct_forces:
                forces = self.force_block(_input_attr.narrow(1, 0, 4))
                forces = forces.narrow(1, 1, 3)
                forces = forces.view(-1, 3).contiguous()
            else:
                forces = (
                    -1
                    * torch.autograd.grad(
                        output["energy"].sum(), batch["pos"], create_graph=self.training
                    )[0]
                )
            output["forces"] = forces
        return output

class MLP_EFS_head_ref_approx(nn.Module):
    def __init__(self,
        backbone,
        reduce: str = "sum",
        regress_stress: bool = False,
        regress_forces: bool = False,
        direct_forces: bool = False,
        unit: str = "ang",
        basis: str = "def2svp",
        xc: str = "b3lyp",
    ):
        super().__init__()
        # self.backbone = backbone
        # assert not self.regress_forces, "regress_forces is not supported"
        self.regress_stress = regress_stress
        assert not self.regress_stress, "regress_stress is not supported, yet"
        self.regress_forces = regress_forces
        self.reduce = reduce
        self.direct_forces = direct_forces
        self.force_block = SO3_Linear(backbone.hidden_size, 1, lmax=1)

        self.unit = unit
        self.basis = basis
        self.xc = xc

        self.sphere_channels = backbone.hidden_size
        self.hidden_channels = backbone.hidden_size
        self.energy_block = nn.Sequential(
            nn.Linear(self.sphere_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, 2, bias=True),
        )

    def build_final_matrix(self, batch, diagonal_matrix, non_diagonal_matrix, transform=True, convention="back2pyscf", dtype=torch.float64):
        pass

    def cal_orbital_and_energies(self, overlap_matrix, full_hamiltonian, method="eigh", tol=0):
        pass

    def forward(self, batch, output):
        natoms = output["node_attr"].shape[0]
        _input_attr = output["node_attr"].reshape(natoms, -1, self.sphere_channels)

        _input = _input_attr.narrow(1, 0, 1).squeeze(1)
        _output = self.energy_block(_input)

        node_energy = _output.view(-1, 2)
        energy_part = scatter(node_energy, batch["batch_full"], dim=0, reduce=self.reduce)

        output["hamiltonian"] = self.build_final_matrix(
            batch,
            output["hamiltonian_diagonal_blocks"],
            output["hamiltonian_non_diagonal_blocks"],
            transform=True,
            convention="back2pyscf",
            dtype=torch.float64,
        )
        
        output["overlap"] = self.build_final_matrix(
            batch,
            batch.diagonal_overlap,
            batch.non_diagonal_overlap,
            transform=True,
            convention="back2pyscf",
            dtype=torch.float64,
        )

        output["orbital_energies"], output["orbital_coefficients"] = (
            self.cal_orbital_and_energies(output["overlap"], output["hamiltonian"])
        )

        src, dst = batch.edge_index_full
        dist = (batch.pos[src] - batch.pos[dst]).norm(dim=-1)
        if self.unit == "ang":
            dist = dist * ANG2BOHR # in bohr
        core_core_interaction = batch.atoms[src] * batch.atoms[dst] / dist.unsqueeze(-1)
        # undirected interaction (Duplciate calculation)
        potential = scatter(core_core_interaction, batch["batch_full"][src], dim=0, reduce="sum") * 0.5
        # potential = scatter(core_core_interaction, src[batch["batch_full"]], dim=0, reduce="sum") * 0.5

        dm = []
        energy_dft1 = []
        energy_dft2 = []
        for i in range(len(batch)):
            # orbital_coefficients = torch.from_numpy(batch.orbital_coefficients[i],dtype=batch.pos.dtype).to(batch.pos.device)
            orbital_coefficients = output["orbital_coefficients"][i]
            num_orb = int(batch.atoms[batch.ptr[i] : batch.ptr[i+1]].sum() / 2)
            sliced_orbital_coefficients = orbital_coefficients[:, :num_orb].to(batch.pos.dtype)
            dm.append(sliced_orbital_coefficients.matmul(sliced_orbital_coefficients.T)* 2)            
            mf = init_pyscf_mf(
                batch.atoms[batch.ptr[i] : batch.ptr[i+1]].detach().cpu(),
                batch.pos[batch.ptr[i] : batch.ptr[i+1]].detach().cpu(),
                unit=self.unit,
                xc=self.xc,
                basis=self.basis,
                )
            h1e = torch.tensor(mf.get_hcore(), device=batch.pos.device, dtype=batch.pos.dtype)
            fock = output["hamiltonian"][i]

            # fock = h1e + 2*veff
            # HpF = h1e + fock = 2*h1e + 2*veff

            veff = (fock - h1e) * 0.5
            veff = veff.to(batch.pos.dtype)

            ref_energy1 = torch.einsum('ij,ji->', h1e, dm[i]) + potential[i]
            ref_energy2 = torch.einsum('ij,ji->', veff, dm[i])

            # cur_energy = mf.energy_tot(dm=dm[i].detach().cpu().numpy())
            energy_dft1.append(ref_energy1)
            energy_dft2.append(ref_energy2)
        
        output["dm"] = dm
        energy_dft1 = torch.stack(energy_dft1, dim=0).reshape(-1, 1)
        energy_dft2 = torch.stack(energy_dft2, dim=0).reshape(-1, 1)

        output["energy"] = energy_part[:,0] + energy_dft1 + energy_dft2 * energy_part[:,1]

        if self.regress_stress:
            pass
        elif self.regress_forces:
            if self.direct_forces:
                forces = self.force_block(_input_attr.narrow(1, 0, 4))
                forces = forces.narrow(1, 1, 3)
                forces = forces.view(-1, 3).contiguous()
            else:
                forces = (
                    -1
                    * torch.autograd.grad(
                        output["energy"].sum(), batch["pos"], create_graph=self.training
                    )[0]
                )
            output["forces"] = forces
        return output

class MLP_EFS_head_interact2(nn.Module):
    def __init__(self,
        backbone,
        reduce: str = "sum",
        regress_stress: bool = False,
        regress_forces: bool = False,
        direct_forces: bool = False,
        mode: str = "train",
        use_gt_H: bool = True,
        unit: str = "ang",
    ):
        super().__init__()
        # self.backbone = backbone
        # assert not self.regress_forces, "regress_forces is not supported"
        self.regress_stress = regress_stress
        assert not self.regress_stress, "regress_stress is not supported, yet"
        self.regress_forces = regress_forces
        self.reduce = reduce
        self.direct_forces = direct_forces
        self.force_block = SO3_Linear(backbone.hidden_size, 1, lmax=1)

        self.sphere_channels = backbone.hidden_size
        self.hidden_channels = backbone.hidden_size
        self.energy_block = nn.Sequential(
            nn.Linear(self.sphere_channels * 3, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, 1, bias=True),
        )
        
        self.output_irrep = backbone.output_irrep
        self.input_irrep = o3.Irreps(f"{self.sphere_channels}x0e")
        self.diagonal_contraction_layer_H = ParamContraction(
            self.output_irrep,
            self.output_irrep,
            self.input_irrep,
        )
        self.non_diagonal_contraction_layer_H = ParamContraction(
            self.output_irrep,
            self.output_irrep,
            self.input_irrep,
        )
        self.diag_block = nn.Sequential(
            nn.Linear(self.sphere_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.sphere_channels, bias=True),
        )
        self.non_diag_block = nn.Sequential(
            nn.Linear(self.sphere_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.sphere_channels, bias=True),
        )
        self.mode = "train"
        self.use_gt_H = True
        self.unit = "ang"
        self.atomrefs = torch.zeros(20)
        self.atomrefs[1] = -0.46437566
        self.atomrefs[6] = -37.19576656
        self.atomrefs[7] = -53.76368628
        self.atomrefs[8] = -73.82292926
        self.atomrefs[9] = -98.16854799
        self.atomrefs = self.atomrefs.cuda()

    def forward(self, batch, output):
        natoms = output["node_attr"].shape[0]
        _input_attr = output["node_attr"].reshape(natoms, -1, self.sphere_channels)

        src, dst = batch.edge_index_full
        # dist = (batch.pos[src] - batch.pos[dst]).norm(dim=-1)
        # if self.unit == "ang":
        #     dist_bohr = dist * ANG2BOHR # in bohr
        # else:
        #     dist_bohr = dist
        # core_core_interaction = batch.atoms[src] * batch.atoms[dst] / dist_bohr.unsqueeze(-1)
        
        # pos1 = batch.pos[batch.batch==0].cpu().numpy()
        # atom1 = batch.atoms[batch.batch==0].cpu().numpy()
        # from pyscf import gto
        # mol = gto.Mole()

        # mol_conf = [[atom1[i].item(), pos1[i]] for i in range(len(atom1))]
        # mol.build(verbose=0, atom=mol_conf, basis="def2svp", unit="ang")
        # mol.energy_nuc()
 
        # # undirected interaction (Duplciate calculation)
        atomrefs_value = self.atomrefs[batch.atoms]
        atomrefs_value = scatter(atomrefs_value, batch["batch_full"], dim=0, reduce="sum")

        # potential = scatter(core_core_interaction, batch["batch_full"][src], dim=0, reduce="sum") * 0.5
        # potential = scatter(core_core_interaction, src[batch["batch_full"]], dim=0, reduce="sum") * 0.5
        
        if self.use_gt_H:
            _H_diagonal = batch.diagonal_hamiltonian
            _H_non_diagonal = batch.non_diagonal_hamiltonian
        else:
            _H_diagonal = output["hamiltonian_diagonal_blocks"]
            _H_non_diagonal = output["hamiltonian_non_diagonal_blocks"]

        _diag_feat = self.diagonal_contraction_layer_H(_H_diagonal)
        _non_diag_feat = self.non_diagonal_contraction_layer_H(_H_non_diagonal)
        
        _diag_feat = self.diag_block(_diag_feat)
        _non_diag_feat = self.non_diag_block(_non_diag_feat)
        _non_diag_feat = scatter(_non_diag_feat, src, dim=0, reduce="sum")
        
        _input = _input_attr.narrow(1, 0, 1).squeeze(1)
        # _input = _input + _diag_feat + scatter(_non_diag_feat, src, dim=0, reduce="sum")
        _input = torch.cat([_input, _diag_feat, _non_diag_feat], dim=1)
        _output = self.energy_block(_input)

        node_energy = _output.view(-1, 1)
        energy_part = scatter(node_energy, batch["batch_full"], dim=0, reduce=self.reduce)
        energy_part = energy_part

        output["energy"] = energy_part + atomrefs_value

        if self.regress_stress:
            pass
        elif self.regress_forces:
            if self.direct_forces:
                forces = self.force_block(_input_attr.narrow(1, 0, 4))
                forces = forces.narrow(1, 1, 3)
                forces = forces.view(-1, 3).contiguous()
            else:
                forces = (
                    -1
                    * torch.autograd.grad(
                        energy_part.sum(), batch["pos"], create_graph=self.training
                    )[0]
                )
            output["forces"] = forces
        return output

# Not good performance
class MLP_EFS_head_interact3(nn.Module):
    def __init__(self,
        backbone,
        reduce: str = "sum",
        regress_stress: bool = False,
        regress_forces: bool = False,
        direct_forces: bool = False,
        mode: str = "train",
        use_atomrefs: bool = False,
    ):
        super().__init__()
        # self.backbone = backbone
        # assert not self.regress_forces, "regress_forces is not supported"
        self.regress_stress = regress_stress
        assert not self.regress_stress, "regress_stress is not supported, yet"
        self.regress_forces = regress_forces
        self.reduce = reduce
        self.direct_forces = direct_forces

        self.expand_ii = nn.ModuleDict()
        # self.expand_ij = nn.ModuleDict()
        self.fc_ii = nn.ModuleDict()
        # self.fc_ij = nn.ModuleDict()
        self.fc_ii_bias = nn.ModuleDict()
        # self.fc_ij_bias = nn.ModuleDict()
        self.contraction_layer = nn.ModuleDict()

        # self.irrep_tp_out_node_pair, instruction_node_pair = get_feasible_irrep(
        #     self.hidden_irrep,
        #     self.hidden_irrep,
        #     self.hidden_irrep,
        #     tp_mode="uuu",
        # )

        self.sphere_channels = backbone.hidden_size
        self.hidden_channels = backbone.hidden_size
        self.hidden_size = self.hidden_channels
        self.bottle_hidden_size = backbone.bottle_hidden_size
        self.output_irrep = o3.Irreps("15x0e + 10x1e + 5x2e")
        # self.output_irrep = backbone.output_irrep

        self._create_matrix_prediction_layers("energy_matrix")
        self.contraction_layer["energy_matrix"] = ParamContraction(
            self.output_irrep,
            self.output_irrep,
            o3.Irreps(f"{self.sphere_channels}x0e"),
        )

        self.energy_block = nn.Sequential(
            nn.Linear(self.sphere_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, 1, bias=True),
        )

        self._create_matrix_prediction_layers("forces_matrix")
        self.contraction_layer["forces_matrix"] = ParamContraction(
            self.output_irrep,
            self.output_irrep,
            o3.Irreps(f"{self.sphere_channels}x0e + {self.sphere_channels}x1e"),
        )

        self.force_block = SO3_Linear(self.sphere_channels, 1, lmax=1)

        
        # self.input_irrep = o3.Irreps(f"{self.sphere_channels}x0e")
        # self.diagonal_contraction_layer_H = ParamContraction(
        #     self.output_irrep,
        #     self.output_irrep,
        #     self.input_irrep,
        # )
        # self.non_diagonal_contraction_layer_H = ParamContraction(
        #     self.output_irrep,
        #     self.output_irrep,
        #     self.input_irrep,
        # )
        # self.diag_block = nn.Sequential(
        #     nn.Linear(self.sphere_channels, self.hidden_channels, bias=True),
        #     nn.SiLU(),
        #     nn.Linear(self.hidden_channels, self.sphere_channels, bias=True),
        # )
        # self.non_diag_block = nn.Sequential(
        #     nn.Linear(self.sphere_channels, self.hidden_channels, bias=True),
        #     nn.SiLU(),
        #     nn.Linear(self.hidden_channels, self.sphere_channels, bias=True),
        # )
        # self.mode = "train"
        # self.use_gt_H = True
        # self.unit = "ang"

        self.atomrefs = torch.zeros(20)
        self.atomrefs[1] = -0.46437566
        self.atomrefs[6] = -37.19576656
        self.atomrefs[7] = -53.76368628
        self.atomrefs[8] = -73.82292926
        self.atomrefs[9] = -98.16854799
        self.atomrefs = self.atomrefs.cuda()
        self.use_atomrefs = use_atomrefs

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
        
        # # Off-diagonal elements (ij)
        # self.expand_ij[matrix_type] = Expansion(
        #     input_expand_irrep, output_irrep, output_irrep
        # )
        # self.fc_ij[matrix_type] = nn.Sequential(
        #     nn.Linear(self.hidden_size * 2, self.hidden_size),
        #     nn.SiLU(),
        #     nn.Linear(self.hidden_size, self.expand_ij[matrix_type].num_path_weight),
        # )
        # self.fc_ij_bias[matrix_type] = nn.Sequential(
        #     nn.Linear(self.hidden_size * 2, self.hidden_size),
        #     nn.SiLU(),
        #     nn.Linear(self.hidden_size, self.expand_ij[matrix_type].num_bias),
        # )

    def forward(self, batch, output):
        natoms = output["node_attr"].shape[0]
        node_attr = output["node_attr_init"]

        energy_matrix = self.expand_ii["energy_matrix"](
            output["fii"],
            self.fc_ii["energy_matrix"](node_attr),
            self.fc_ii_bias["energy_matrix"](node_attr),
        )

        forces_matrix = self.expand_ii["forces_matrix"](
            output["fii"],
            self.fc_ii["forces_matrix"](node_attr),
            self.fc_ii_bias["forces_matrix"](node_attr),
        )

        energy_feat = self.contraction_layer["energy_matrix"](energy_matrix)
        forces_feat = self.contraction_layer["forces_matrix"](forces_matrix)

        node_energy = self.energy_block(energy_feat)

        node_energy = node_energy.view(-1, 1)
        energy_part = scatter(node_energy, batch["batch_full"], dim=0, reduce=self.reduce)

        output["energy"] = energy_part

        if self.use_atomrefs:
            atomrefs_value = self.atomrefs[batch.atoms]
            atomrefs_value = scatter(atomrefs_value, batch["batch_full"], dim=0, reduce="sum")
            output["energy"] = output["energy"] + atomrefs_value

        if self.regress_stress:
            pass
        elif self.regress_forces:
            if self.direct_forces:
                forces = self.force_block(forces_feat.reshape(natoms, -1, self.sphere_channels))
                forces = forces.narrow(1, 1, 3)
                forces = forces.view(-1, 3).contiguous()
            else:
                forces = (
                    -1
                    * torch.autograd.grad(
                        energy_part.sum(), batch["pos"], create_graph=self.training
                    )[0]
                )
            output["forces"] = forces
        return output

# Not good performance
class MLP_EFS_head_interact4(nn.Module):
    def __init__(self,
        backbone,
        reduce: str = "sum",
        regress_stress: bool = False,
        regress_forces: bool = False,
        direct_forces: bool = False,
        mode: str = "train",
        use_atomrefs: bool = False,
    ):
        super().__init__()
        # self.backbone = backbone
        # assert not self.regress_forces, "regress_forces is not supported"
        self.regress_stress = regress_stress
        assert not self.regress_stress, "regress_stress is not supported, yet"
        self.regress_forces = regress_forces
        self.reduce = reduce
        self.direct_forces = direct_forces

        self.expand_ii = nn.ModuleDict()
        # self.expand_ij = nn.ModuleDict()
        self.fc_ii = nn.ModuleDict()
        # self.fc_ij = nn.ModuleDict()
        self.fc_ii_bias = nn.ModuleDict()
        # self.fc_ij_bias = nn.ModuleDict()
        self.contraction_layer = nn.ModuleDict()

        # self.irrep_tp_out_node_pair, instruction_node_pair = get_feasible_irrep(
        #     self.hidden_irrep,
        #     self.hidden_irrep,
        #     self.hidden_irrep,
        #     tp_mode="uuu",
        # )

        self.sphere_channels = backbone.hidden_size
        self.hidden_channels = backbone.hidden_size
        self.hidden_size = self.hidden_channels
        self.bottle_hidden_size = backbone.bottle_hidden_size
        self.hidden_irrep = o3.Irreps("15x0e + 10x1e + 5x2e")
        self.output_irrep = o3.Irreps("15x0e + 10x1e + 5x2e")
        # self.hidden_irrep = backbone.hidden_irrep
        # self.output_irrep = backbone.hidden_irrep

        self._create_matrix_prediction_layers("energy_matrix")
        self.energy_layer = nn.Linear(self.hidden_irrep.dim, self.sphere_channels)
        # self.contraction_layer["energy_matrix"] = ParamContraction(
        #     self.output_irrep,
        #     self.output_irrep,
        #     o3.Irreps(f"{self.sphere_channels}x0e"),
        # )

        self.energy_block = nn.Sequential(
            nn.Linear(self.sphere_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, 1, bias=True),
        )

        self._create_matrix_prediction_layers("forces_matrix")
        # self.force_layer = nn.Linear(self.hidden_irrep.dim, self.hidden_size)
        self.contraction_layer["forces_matrix"] = ParamContraction(
            self.output_irrep,
            self.output_irrep,
            o3.Irreps(f"{self.sphere_channels}x0e + {self.sphere_channels}x1e"),
        )

        self.force_block = SO3_Linear(self.sphere_channels, 1, lmax=1)

        
        # self.input_irrep = o3.Irreps(f"{self.sphere_channels}x0e")
        # self.diagonal_contraction_layer_H = ParamContraction(
        #     self.output_irrep,
        #     self.output_irrep,
        #     self.input_irrep,
        # )
        # self.non_diagonal_contraction_layer_H = ParamContraction(
        #     self.output_irrep,
        #     self.output_irrep,
        #     self.input_irrep,
        # )
        # self.diag_block = nn.Sequential(
        #     nn.Linear(self.sphere_channels, self.hidden_channels, bias=True),
        #     nn.SiLU(),
        #     nn.Linear(self.hidden_channels, self.sphere_channels, bias=True),
        # )
        # self.non_diag_block = nn.Sequential(
        #     nn.Linear(self.sphere_channels, self.hidden_channels, bias=True),
        #     nn.SiLU(),
        #     nn.Linear(self.hidden_channels, self.sphere_channels, bias=True),
        # )
        # self.mode = "train"
        # self.use_gt_H = True
        # self.unit = "ang"

        self.atomrefs = torch.zeros(20)
        self.atomrefs[1] = -0.46437566
        self.atomrefs[6] = -37.19576656
        self.atomrefs[7] = -53.76368628
        self.atomrefs[8] = -73.82292926
        self.atomrefs[9] = -98.16854799
        self.atomrefs = self.atomrefs.cuda()
        self.use_atomrefs = use_atomrefs

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
        
        # # Off-diagonal elements (ij)
        # self.expand_ij[matrix_type] = Expansion(
        #     input_expand_irrep, output_irrep, output_irrep
        # )
        # self.fc_ij[matrix_type] = nn.Sequential(
        #     nn.Linear(self.hidden_size * 2, self.hidden_size),
        #     nn.SiLU(),
        #     nn.Linear(self.hidden_size, self.expand_ij[matrix_type].num_path_weight),
        # )
        # self.fc_ij_bias[matrix_type] = nn.Sequential(
        #     nn.Linear(self.hidden_size * 2, self.hidden_size),
        #     nn.SiLU(),
        #     nn.Linear(self.hidden_size, self.expand_ij[matrix_type].num_bias),
        # )

    def forward(self, batch, output):
        natoms = output["node_attr"].shape[0]
        node_attr = output["node_attr_init"]

        energy_matrix = self.expand_ii["energy_matrix"](
            output["fii"],
            self.fc_ii["energy_matrix"](node_attr),
            self.fc_ii_bias["energy_matrix"](node_attr),
        )

        forces_matrix = self.expand_ii["forces_matrix"](
            output["fii"],
            self.fc_ii["forces_matrix"](node_attr),
            self.fc_ii_bias["forces_matrix"](node_attr),
        )
        forces_feat = self.contraction_layer["forces_matrix"](forces_matrix)

        energy_feat = torch.einsum("bij,bjk->bik", energy_matrix, energy_matrix.transpose(-2,-1))
        energy_feat = torch.diagonal(energy_feat, dim1=1, dim2=2)
        energy_feat = self.energy_layer(energy_feat)             
        node_energy = self.energy_block(energy_feat)

        node_energy = node_energy.view(-1, 1)
        energy_part = scatter(node_energy, batch["batch_full"], dim=0, reduce=self.reduce)

        output["energy"] = energy_part

        if self.use_atomrefs:
            atomrefs_value = self.atomrefs[batch.atoms]
            atomrefs_value = scatter(atomrefs_value, batch["batch_full"], dim=0, reduce="sum")
            output["energy"] = output["energy"] + atomrefs_value
            
        
        if self.regress_stress:
            pass
        elif self.regress_forces:
            if self.direct_forces:
                forces = self.force_block(forces_feat.reshape(natoms, -1, self.sphere_channels))
                forces = forces.narrow(1, 1, 3)
                forces = forces.view(-1, 3).contiguous()
            else:
                forces = (
                    -1
                    * torch.autograd.grad(
                        energy_part.sum(), batch["pos"], create_graph=self.training
                    )[0]
                )
            output["forces"] = forces
        return output