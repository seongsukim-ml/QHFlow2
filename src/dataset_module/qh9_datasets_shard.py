import numpy as np
import pickle
import os
from common.custom_logger import setup_global_logger, get_logger
import json
import gdown
import torch
from tqdm.rich import tqdm
import random
import time

from dataset_module.lmdb_shard import LMDBShard_maker_db
from dataset_module.qh9_common import BaseQH9Dataset, compute_common_artifacts
from common.units import *

logger = get_logger(__file__)

GoogleDriveLink = (
    "https://drive.google.com/drive/u/0/folders/1LXTC8uaOQzmb76FsuGfwSocAbK5Hshfj"
)

########################################################
# QH9Stable
########################################################


class QH9Stable_shard(LMDBShard_maker_db):
    def __init__(
        self,
        root_path: str,
        shard_num: int,
        save_path=None,
        max_workers=8,
        use_parallel=True,
        processd_dir_name="processed",
        shard_dir_name="lmdbs",
        split="random",
        *args,
        **kwargs,
    ):
        super().__init__(
            root_path=root_path,
            shard_num=shard_num,
            save_path=save_path,
            max_workers=max_workers,
            use_parallel=use_parallel,
            processd_dir_name=processd_dir_name,
            shard_dir_name=shard_dir_name,
            *args,
            **kwargs,
        )
        self.split = split
        assert self.split in ["random", "size_ood"], f"Split {self.split} for QH9Stable is not supported"
        if self.split == "random":
            self.split_path = os.path.join(self.save_path, "processed_QH9Stable_random_12.json")
        elif self.split == "size_ood":
            self.split_path = os.path.join(self.save_path, "processed_QH9Stable_size_ood.json")

        # if cal_orbital_and_energies is True, data will contain orbital energies and coefficients

    def _make_split_info(self):
        if os.path.exists(self.split_path):
            pass
        else:
            if self.split == "random":
                logger.info("Making random split")
                self.split_info = {
                    "train": 0.8,
                    "val": 0.1,
                    "test": 0.1,
                }
                total_len = self._get_total_len()
                train_len = int(total_len * self.split_info["train"])
                val_len = int(total_len * self.split_info["val"])
                test_len = total_len - train_len - val_len
                indices = np.random.RandomState(seed=43).permutation(total_len)
                train_indices = indices[:train_len]
                val_indices = indices[train_len:train_len+val_len]
                test_indices = indices[train_len+val_len:]

            elif self.split == "size_ood":
                """ 
                size OOD split:
                    train: N <= 20
                    val: 21 <= N <= 22
                    test: N >= 23
                """
                logger.info("Making size OOD split...")
                assert self.all_data is not None, "all_data is not set"
                num_nodes_list = [row[1] for row in self.all_data]
                num_nodes_array = np.array(num_nodes_list)
                total_len = len(num_nodes_array)
                train_indices = np.where(num_nodes_array <= 20)
                val_indices = np.where(np.logical_and(num_nodes_array >= 21, num_nodes_array <= 22))
                test_indices = np.where(num_nodes_array >= 23)

            assert len(train_indices) + len(val_indices) + len(test_indices) == total_len, f"Total length {total_len} is not equal to the sum of train, val, and test length {len(train_indices) + len(val_indices) + len(test_indices)}"
            self.split_info = {
                "train": sorted(train_indices.tolist()),
                "val": sorted(val_indices.tolist()),
                "test": sorted(test_indices.tolist()),
            }
            with open(self.split_path, "w") as f:
                json.dump(self.split_info, f)

    def process_data(self, key_data_pair):
        """
        Process a single row from the database.
        
        Args:
            data: Database row containing molecular data
            
        Returns:
            tuple: (key, data_dict) for LMDB storage

        Note:
            QH9Stable.db data format:
                0: id (INTEGER, np.int32) (PRIMARY KEY) NOT NULL index
                1: N (INTEGER, np.int32) number of atoms
                2: Z (BLOB, np.int32) atomic numbers
                3: pos (BLOB, np.float64) atomic positions / unit: angstrom 
                4: Ham (BLOB, np.float64) hamiltonian matrix / unit: hartree / pyscf convention
      """
        data, data_idx = key_data_pair
        key = int(data_idx).to_bytes(length=4, byteorder="big")  # real key is not used

        atoms = np.frombuffer(data[2], np.int32)
        pos = np.frombuffer(data[3], np.float64)
        hamiltonian_flat = np.frombuffer(data[4], np.float64)

        artifacts = compute_common_artifacts(
            atoms=atoms,
            pos_angstrom=pos,
            flat_hamiltonian=hamiltonian_flat,
        )

        ori_data_dict = {
            "id": data[0],
            "num_nodes": data[1],
            "atoms": data[2],
            "pos": data[3],  # unit: angstrom
            "dft_energy": artifacts.dft_energy,  # unit: Eh
            "dft_forces": artifacts.dft_forces.tobytes(),  # unit: Eh/Bohr
            "h_dim": artifacts.h_dim,
            "packed_hamiltonian": artifacts.packed_hamiltonian.tobytes(),  # unit: Eh
            "packed_overlap": artifacts.packed_overlap.tobytes(),
            "packed_initial_hamiltonian": artifacts.packed_initial_hamiltonian.tobytes(),  # unit: Eh
            "orbital_energies": artifacts.orbital_energies.tobytes(),  # unit: Eh
            "orbital_coefficients": artifacts.orbital_coefficients.tobytes(),
            "dm0": artifacts.dm0.tobytes(),
            "pos_unit": "Angstrom",
        }
        data_dict = pickle.dumps(ori_data_dict)
        return key, data_dict

class QH9Stable(BaseQH9Dataset):
    url = "https://drive.google.com/file/d/1LcEJGhB8VUGkuyb0oQ_9ANJdSkky9xMS/view?usp=sharing"
    shard_cls = QH9Stable_shard

    def __init__(
        self,
        root="datasets/",
        split="random",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        prefix="",
        shard_num=60,
        shard_idx=-1,
        max_workers_preprocess=8,
        use_parallel_preprocess=False,
        return_orbital_and_energies=False,
        use_ref_hamiltonian_as_init_ham=False,
        compute_q_tensor=True,
        include_initial_hamiltonian=True,
        include_overlap=True,
        include_dft_energy=True,
        include_dft_forces=True,
        include_cut_orbital_coefficients=True,
        cut_coeff_cache_type="both",
    ):
        assert split in ["random", "size_ood"], f"Split {split} for QH9Stable is not supported"
        super().__init__(
            dataset_name="QH9Stable",
            root=root,
            split=split,
            shard_num=shard_num,
            shard_idx=shard_idx,
            prefix=prefix,
            max_workers_preprocess=max_workers_preprocess,
            use_parallel_preprocess=use_parallel_preprocess,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            return_orbital_and_energies=return_orbital_and_energies,
            use_ref_hamiltonian_as_init_ham=use_ref_hamiltonian_as_init_ham,
            compute_q_tensor=compute_q_tensor,
            include_initial_hamiltonian=include_initial_hamiltonian,
            include_overlap=include_overlap,
            include_dft_energy=include_dft_energy,
            include_dft_forces=include_dft_forces,
            include_cut_orbital_coefficients=include_cut_orbital_coefficients,
            cut_coeff_cache_type=cut_coeff_cache_type,
        )

    def _resolve_split_path(self) -> str:
        if self.split == "random":
            return os.path.join(self.folder, "processed_QH9Stable_random_12.json")
        if self.split == "size_ood":
            return os.path.join(self.folder, "processed_QH9Stable_size_ood.json")
        raise ValueError(f"Split {self.split} for QH9Stable is not supported")

    @property
    def raw_file_names(self):
        return ["QH9Stable.db"]

    @property
    def processed_file_names(self):
        if self.split == "random":
            return [
                "processed_QH9Stable_random_12.json",
                "ALL_SHARDS_COMPLETED.txt",
                "index.json",
            ]
        if self.split == "size_ood":
            return [
                "processed_QH9Stable_size_ood.json",
                "ALL_SHARDS_COMPLETED.txt",
                "index.json",
            ]
        raise ValueError(f"Split {self.split} for QH9Stable is not supported")

    def download(self):
        try:
            print(f"Downloading the QH9Stable dataset to through {self.url}")
            gdown.download(self.url, output=self.raw_paths[0], fuzzy=True)
        except Exception:
            print(
                f"Downloading failed! Please download the QH9Stable dataset to {self.raw_paths[0]} through {self.url}"
            )
            print(f"Or you can try to download the zip file through {GoogleDriveLink}")
            raise FileNotFoundError("QH9Stable needs to be downloaded.")
    
########################################################
# QH9Dynamic
########################################################

class QH9Dynamic_shard(LMDBShard_maker_db):
    def __init__(
        self,
        root_path: str,
        shard_num: int,
        save_path=None,
        max_workers=8,
        use_parallel=True,
        processd_dir_name="processed",
        shard_dir_name="lmdbs",
        split="random",
        *args,
        **kwargs,
    ):
        super().__init__(
            root_path=root_path,
            shard_num=shard_num,
            save_path=save_path,
            max_workers=max_workers,
            use_parallel=use_parallel,
            processd_dir_name=processd_dir_name,
            shard_dir_name=shard_dir_name,
            *args,
            **kwargs,
        )
        self.split = split
        assert self.split in ["geometry", "mol"], f"Split {self.split} for QH9Dynamic is not supported"
        if self.split == "geometry":
            self.split_path = os.path.join(self.save_path, "processed_QH9Dynamic_geometry.json")
        elif self.split == "mol":
            self.split_path = os.path.join(self.save_path, "processed_QH9Dynamic_mol.json")

    def _make_split_info(self):
        if os.path.exists(self.split_path):
            pass
        else:
            num_geometry_per_mol = 100
            num_train_geometry_per_mol = 80
            num_val_geometry_per_mol = 10
            num_mol = 2998

            total_len = self._get_total_len()
            if self.split == "geometry":
                """
                geometry-wise split:
                    train: 80%
                    val: 10%
                    test: 10%
                """
                logger.info("Making geometry-wise split...")
                train_indices = np.array([], dtype=np.int64)
                val_indices = np.array([], dtype=np.int64)
                test_indices = np.array([], dtype=np.int64)
                cur_index = 0
                
                with self._get_db_env() as db_env:
                    cursor = db_env.cursor()
                    data = cursor.execute("select * from data").fetchall()
                    for ind, row in enumerate(tqdm(data)):
                        if (
                            ind + 1
                        ) % num_geometry_per_mol == 0:  # Finish traversing one molecule
                            indices = np.random.RandomState(seed=ind).permutation(
                                num_geometry_per_mol
                            )  # Different random split for different molecules
                            train_indices_cur_mol = (
                                cur_index + indices[:num_train_geometry_per_mol]
                            )
                            val_indices_cur_mol = (
                                cur_index
                                + indices[
                                    num_train_geometry_per_mol : num_train_geometry_per_mol
                                    + num_val_geometry_per_mol
                                ]
                            )
                            test_indices_cur_mol = (
                                cur_index
                                + indices[
                                    num_train_geometry_per_mol + num_val_geometry_per_mol :
                                ]
                            )
                            train_indices = np.concatenate((train_indices, train_indices_cur_mol))
                            val_indices = np.concatenate((val_indices, val_indices_cur_mol))
                            test_indices = np.concatenate((test_indices, test_indices_cur_mol))
                            cur_index += num_geometry_per_mol


            elif self.split == "mol":
                """ 
                molecule-wise split:
                    train: 80%
                    val: 10%
                    test: 10%
                """
                mol_id_list = [
                    i for i in range(num_mol) for _ in range(num_geometry_per_mol)
                ]
                data_ratio = [0.8, 0.1, 0.1]
                index_list = [i for i in range(max(mol_id_list) + 1)]
                random.seed(43)
                random.shuffle(index_list)
                train_mol_ids = np.array(index_list[: int(len(index_list) * data_ratio[0])])
                val_mol_ids = np.array(
                    index_list[
                        int(len(index_list) * data_ratio[0]) : int(
                            len(index_list) * data_ratio[0]
                        )
                        + int(len(index_list) * data_ratio[1])
                    ]
                )
                test_mol_ids = np.array(
                    index_list[
                        int(len(index_list) * data_ratio[0])
                        + int(len(index_list) * data_ratio[1]) :
                    ]
                )
                mol_id_array = np.array(mol_id_list)
                train_indices = np.where(np.isin(mol_id_array, train_mol_ids))[0].astype(np.int64)
                val_indices = np.where(np.isin(mol_id_array, val_mol_ids))[0].astype(np.int64)
                test_indices = np.where(np.isin(mol_id_array, test_mol_ids))[0].astype(np.int64)
            assert len(train_indices) + len(val_indices) + len(test_indices) == total_len, f"Total length {total_len} is not equal to the sum of train, val, and test length {len(train_indices) + len(val_indices) + len(test_indices)}"
            self.split_info = {
                "train": sorted(train_indices.tolist()),
                "val": sorted(val_indices.tolist()),
                "test": sorted(test_indices.tolist()),
            }
            with open(self.split_path, "w") as f:
                json.dump(self.split_info, f)

    def process_data(self, key_data_pair):
        """
        Process a single row from the database.
        
        Args:
            data: Database row containing molecular data
            
        Returns:
            tuple: (key, data_dict) for LMDB storage

        Note:
            Dynamic.db Data
                0: id         (INTEGER) (PRIMARY KEY) NOT NULL 
                1: geo_id     (INTEGER) id that indicates the geometry (whether it is from the same molecule)
                2: N          (INTEGER) number of atoms
                3: Z          (BLOB, np.int32) atomic numbers
                4: pos        (BLOB, np.float64) atomic positions / unit: bohr 
                5: ekin       (FLOAT) kinetic energy / unit: hartree ? (Not used)
                6: epot       (FLOAT) potential energy / unit: hartree ? (Not used)
                7: etot       (FLOAT) total energy / unit: hartree ? (Not used)
                8: time       (FLOAT) time / unit : ? (Not used)
                9: Ham        (BLOB, np.float64) NULL 
                10: converged (INT, np.int) NULL (Not used)
      """
        data, data_idx = key_data_pair
        key = int(data_idx).to_bytes(length=4, byteorder="big")
        geo_id = data[1]
        atoms = np.frombuffer(data[3], np.int32)
        pos_bohr = np.frombuffer(data[4], np.float64)
        hamiltonian_flat = np.frombuffer(data[9], np.float64)

        artifacts = compute_common_artifacts(
            atoms=atoms,
            pos_angstrom=pos_bohr * BOHR2ANG,
            flat_hamiltonian=hamiltonian_flat,
        )

        ori_data_dict = {
            "id": data[0],
            "geo_id": geo_id,
            "num_nodes": data[2],
            "atoms": data[3],
            "pos": data[4],  # stored in Bohr for later conversion
            "dft_energy": artifacts.dft_energy,  # unit: Eh
            "dft_forces": artifacts.dft_forces.tobytes(),  # unit: Eh/Bohr
            "h_dim": artifacts.h_dim,
            "packed_hamiltonian": artifacts.packed_hamiltonian.tobytes(),  # unit: Eh
            "packed_overlap": artifacts.packed_overlap.tobytes(),
            "packed_initial_hamiltonian": artifacts.packed_initial_hamiltonian.tobytes(),  # unit: Eh
            "orbital_energies": artifacts.orbital_energies.tobytes(),  # unit: Eh
            "orbital_coefficients": artifacts.orbital_coefficients.tobytes(),
            "dm0": artifacts.dm0.tobytes(),
            "pos_unit": "Bohr",
        }
        data_dict = pickle.dumps(ori_data_dict)
        return key, data_dict

########################################################
# QH9Dynamic
########################################################

class QH9Dynamic(BaseQH9Dataset):
    url = {
        "100k": "https://drive.google.com/file/d/1SNWk0GD6Nt96qNAJJU2uedwWDQ4bbB1w/view?usp=sharing",
        "300k": "https://drive.google.com/file/d/1sbf-sFhh3ZmhXgTcN2ke_la39MaG0Yho/view?usp=sharing",
    }
    shard_cls = QH9Dynamic_shard

    def __init__(
        self,
        root="datasets/",
        split="geometry",
        version="300k",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        prefix="",
        shard_num=60,
        shard_idx=-1,
        max_workers_preprocess=8,
        use_parallel_preprocess=False,
        return_orbital_and_energies=False,
        use_ref_hamiltonian_as_init_ham=False,
        compute_q_tensor=True,
        include_initial_hamiltonian=True,
        include_overlap=True,
        include_dft_energy=True,
        include_dft_forces=True,
        include_cut_orbital_coefficients=True,
        cut_coeff_cache_type="both",
    ):
        self.version = version
        assert self.version in ["100k", "300k"], f"Version {self.version} for QH9Dynamic is not supported"
        assert split in ["geometry", "mol"], f"Split {split} for QH9Dynamic is not supported"

        dataset_name = f"QH9Dynamic_{self.version}"
        self.full_dataname = dataset_name

        super().__init__(
            dataset_name=dataset_name,
            root=root,
            split=split,
            shard_num=shard_num,
            shard_idx=shard_idx,
            prefix=prefix,
            max_workers_preprocess=max_workers_preprocess,
            use_parallel_preprocess=use_parallel_preprocess,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            return_orbital_and_energies=return_orbital_and_energies,
            use_ref_hamiltonian_as_init_ham=use_ref_hamiltonian_as_init_ham,
            compute_q_tensor=compute_q_tensor,
            include_initial_hamiltonian=include_initial_hamiltonian,
            include_overlap=include_overlap,
            include_dft_energy=include_dft_energy,
            include_dft_forces=include_dft_forces,
            include_cut_orbital_coefficients=include_cut_orbital_coefficients,
            cut_coeff_cache_type=cut_coeff_cache_type,
        )
        self.pos_unit = "bohr" # Need to convert to Angstrom for later conversion

    def _resolve_split_path(self) -> str:
        if self.split == "geometry":
            return os.path.join(self.folder, "processed_QH9Dynamic_geometry.json")
        if self.split == "mol":
            return os.path.join(self.folder, "processed_QH9Dynamic_mol.json")
        raise ValueError(f"Split {self.split} for QH9Dynamic is not supported")

    @property
    def raw_file_names(self):
        return [f"QH9Dynamic_{self.version}.db"]

    @property
    def processed_file_names(self):
        if self.split == "geometry":
            return [
                "processed_QH9Dynamic_geometry.json",
                "ALL_SHARDS_COMPLETED.txt",
                "index.json",
            ]
        if self.split == "mol":
            return [
                "processed_QH9Dynamic_mol.json",
                "ALL_SHARDS_COMPLETED.txt",
                "index.json",
            ]
        raise ValueError(f"Split {self.split} for QH9Dynamic is not supported")

    def download(self):
        try:
            print(
                f"Downloading the QH9Dynamic_{self.version} dataset through {self.url}"
            )
            gdown.download(self.url[self.version], output=self.raw_paths[0], fuzzy=True)
        except Exception:
            print(
                f"Downloading failed! Please download the QH9Dynamic_{self.version} dataset to {self.raw_paths[0]} "
                f"through {self.url[self.version]}"
            )
            print(f"Or you can try to download the zip file through {GoogleDriveLink}")
            raise FileNotFoundError(
                f"QH9Dynamic_{self.version} needs to be downloaded."
            )

    def _postprocess_orbital_energies(self, orbital_energies: torch.Tensor) -> torch.Tensor:
        return orbital_energies.squeeze()



########################################################
# Main function & Argument parser
########################################################

def parse_shard_idx(shard_idx_str):
    """Parse shard_idx string into a list of integers"""
    if shard_idx_str == "-1":
        return int(-1)
    elif ',' in shard_idx_str:
        # Comma-separated list: "0,1,2,3"
        return [int(x.strip()) for x in shard_idx_str.split(',')]
    elif '-' in shard_idx_str and shard_idx_str.count('-') == 1:
        # Range: "0-5"
        start, end = map(int, shard_idx_str.split('-'))
        return list(range(start, end + 1))
    else:
        # Single integer: "0"
        return [int(shard_idx_str)]

if __name__ == "__main__":
    setup_global_logger()

    import argparse

    parser = argparse.ArgumentParser(description="Generation")
    parser.add_argument(
        "--root",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "dataset"),
    )
    parser.add_argument("--name", type=str, default="QH9Stable")
    parser.add_argument("--version", type=str, default="300k")
    parser.add_argument("--split", type=str, default="random")
    parser.add_argument("--dynamic_split", type=str, default="geometry")
    parser.add_argument("--shard_num", type=int, default=10, help="Number of shards for the dataset, for example, 10 shards for QH9Stable dataset")
    parser.add_argument("--shard_idx", type=str, default="0", help="-1 for all shards, Comma-separated list of shard indices or single index ex) 0,1,2,3 or 0-5")
    parser.add_argument("--prefix", type=str, default="_shard")
    parser.add_argument("--use_ref_hamiltonian_as_init_ham", action="store_true", default=False)
    parser.add_argument("--pdb", action="store_true", default=False)

    args = parser.parse_args()
    args.shard_idx = parse_shard_idx(args.shard_idx)
    torch.set_num_threads(4)
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"

    logger.info(f"OMP_NUM_THREADS: {os.environ['OMP_NUM_THREADS']}")
    logger.info(f"MKL_NUM_THREADS: {os.environ['MKL_NUM_THREADS']}")
    logger.info(f"NUMEXPR_NUM_THREADS: {os.environ['NUMEXPR_NUM_THREADS']}")
    logger.info(f"OPENBLAS_NUM_THREADS: {os.environ['OPENBLAS_NUM_THREADS']}")

    logger.info(f"dataset path: {args.root}")

    logger.info(f"Processing {args.name} dataset with shard_idx: {args.shard_idx}")

    assert args.name in ["QH9Stable", "QH9Dynamic"]
    if args.name == "QH9Stable":
        assert args.split in ["random", "size_ood"]
        dataset = QH9Stable(
            root=args.root,
            split=args.split,
            shard_num=args.shard_num,
            shard_idx=args.shard_idx,
            prefix=args.prefix,
            use_ref_hamiltonian_as_init_ham=args.use_ref_hamiltonian_as_init_ham,
        )
    elif args.name == "QH9Dynamic":
        assert args.version in ["300k"], "Only 300k version is supported for QH9Dynamic currently"
        assert args.dynamic_split in ["geometry", "mol"]
        dataset = QH9Dynamic(
            root=args.root,
            version=args.version,
            split=args.dynamic_split,
            shard_num=args.shard_num,
            shard_idx=args.shard_idx,
            prefix=args.prefix,
            use_ref_hamiltonian_as_init_ham=args.use_ref_hamiltonian_as_init_ham,
        )

    # Compare with split dataset
    if args.pdb:
        from dataset_module.qh9_datasets_split import QH9Stable as QH9Stable_split, QH9Dynamic as QH9Dynamic_split
        dataset[158]
        import pdb
        pdb.set_trace()

        from torch_geometric.loader import DataLoader
        # loader = DataLoader(dataset, batch_size=100, shuffle=False)
        # batch = next(iter(loader))

        if args.name == "QH9Stable":
            dataset_split = QH9Stable_split(
                root=args.root,
                split=args.split,
                # num_chunks=args.shard_num,
                # prefix=args.prefix,
            )
        elif args.name == "QH9Dynamic":
            dataset_split = QH9Dynamic_split(
                root=args.root,
                version=args.version,
                split=args.dynamic_split,
                # num_chunks=args.shard_num,
                # prefix=args.prefix,
            )
        
        import random
        # random_idxs = random.sample(range(len(dataset_split)), 100)

        # check ={
        #     "pos": [], # QH9Dynamic has some error due to the conversion from Bohr to Angstrom error
        #     "atoms": [],
        #     "diagonal_hamiltonian": [],
        #     "non_diagonal_hamiltonian": [],
        #     "diagonal_hamiltonian_mask": [],
        #     "non_diagonal_hamiltonian_mask": [],
        #     "diagonal_init_ham": [],
        #     "non_diagonal_init_ham": [],
        #     "diagonal_overlap": [],
        #     "non_diagonal_overlap": [],
        #     "diagonal_Q": [],
        #     "non_diagonal_Q": [],
        #     "edge_index_full": [],
        # }
        # EPS = 1e-7
        # start_time = time.time()
        # for idx in random_idxs:
        #     data_shard = dataset[idx]
        #     data = dataset_split[idx]
        #     for key in check.keys():
        #         if data_shard[key].shape != data[key].shape:
        #             check[key].append(False)
        #         else:
        #             if key == "atoms" or key == "edge_index_full":
        #                 err = abs(data_shard[key]-data[key]).sum()
        #                 err = err / data[key].shape[0]
        #             else:
        #                 err = abs(data_shard[key]-data[key]).mean()
        #             if err > EPS:
        #                 check[key].append(False)
        #             else:
        #                 check[key].append(True)

        # for key in check.keys():
        #     print(key, np.array(check[key]).mean())
        # end_time = time.time()
        # print(f"Time taken calc: {end_time - start_time} seconds")
        
        # Debugging: Check eigenvalue ordering and CHC calculation
        from dataset_module.qh9_common import debug_eigenvalue_ordering_and_chc
        import random
        
        logger.info("=" * 80)
        logger.info("Debugging: Checking eigenvalue ordering and CHC calculation")
        logger.info("=" * 80)
        
        # Sample a few indices for detailed debugging
        # debug_indices = random.sample(range(len(dataset)), min(10, len(dataset)))
        debug_indices = [49865, 122989]
        
        # Run comprehensive debugging
        error_results = debug_eigenvalue_ordering_and_chc(
            dataset=dataset,
            debug_indices=debug_indices,
            logger=logger,
        )
        
        # Store results for potential use
        eigenvalue_ordering_errors = error_results["eigenvalue_ordering_errors"]
        eigenvalue_recalc_errors = error_results["eigenvalue_recalc_errors"]
        chc_eigenvalue_errors = error_results["chc_eigenvalue_errors"]
        block_chc_errors = error_results["block_chc_errors"]
        
        # Summary is already printed by debug_eigenvalue_ordering_and_chc
        # Additional summary can be added here if needed
        
        pdb.set_trace()
        
        # start_time = time.time()
        # for i in random_idxs:
        #     data = dataset[i]
        # end_time = time.time()
        # print(f"Time taken (shard): {end_time - start_time} seconds")
        # # Shard (QH9Stable) : 8.85
        # # Shard (QH9Dynamic): 8.73

        # start_time = time.time()
        # for i in random_idxs:
        #     data = dataset_split[i]
        # end_time = time.time()
        # print(f"Time taken (split): {end_time - start_time} seconds")
        # Split (QH9Stable) : 12.07
        # Split (QH9Dynamic): 12.63

        # 27% acceleration for QH9Stable
        # 28% acceleration for QH9Dynamic

    print(len(dataset))
    print(dataset[0])
    print(dataset[-1])
    for i in range(len(dataset)):
        data = dataset[i]