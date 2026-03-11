'''
This material was prepared as an account of work sponsored by an agency of the
United States Government.  Neither the United States Government nor the United
States Department of Energy, nor Battelle, nor any of their employees, nor any
jurisdiction or organization that has cooperated in the development of these
materials, makes any warranty, express or implied, or assumes any legal
liability or responsibility for the accuracy, completeness, or usefulness or
any information, apparatus, product, software, or process disclosed, or
represents that its use would not infringe privately owned rights.
 
Reference herein to any specific commercial product, process, or service by
trade name, trademark, manufacturer, or otherwise does not necessarily
constitute or imply its endorsement, recommendation, or favoring by the United
States Government or any agency thereof, or Battelle Memorial Institute. The
views and opinions of authors expressed herein do not necessarily state or
reflect those of the United States Government or any agency thereof.
 
                 PACIFIC NORTHWEST NATIONAL LABORATORY
                              operated by
                                BATTELLE
                                for the
                   UNITED STATES DEPARTMENT OF ENERGY
                    under Contract DE-AC05-76RL01830
'''


"""
Extract backbone embeddings from a pretrained UMA model from fairchem.

This module provides utilities to load UMA models and extract node-level
embeddings from the backbone network for atomic structure analysis.
"""

import torch
torch.set_float32_matmul_precision('medium')
import numpy as np
from typing import Optional, Union, List, Dict
from pathlib import Path
from ase import Atoms
from ase.io import read
import logging

from fairchem.core.calculate import pretrained_mlip
from fairchem.core.datasets.atomic_data import AtomicData


class UMAEmbeddingExtractor:
    """
    Extract backbone embeddings from pretrained UMA models.
    
    The UMA (Universal Model for Atoms) backbone produces node embeddings that
    encode the local atomic environment in a learned representation space.
    These embeddings can be used for downstream analysis, clustering, or
    transfer learning tasks.
    
    Attributes:
        predict_unit: The loaded UMA prediction unit
        device: Device to run inference on ('cuda' or 'cpu')
        model_name: Name of the pretrained model
    """
    
    def __init__(
        self,
        model_name: str = "uma-s-1p1",
        device: Optional[str] = None,
        inference_settings: str = "default",
        checkpoint_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the embedding extractor with a pretrained UMA model.
        
        Args:
            model_name: Name of pretrained model. Options include:
                - "uma-s-1p1": Small UMA model (6.6M/150M params, fastest)
                - "uma-m-1p1": Medium UMA model (50M/1.4B params, best accuracy)
            device: Device to use ('cuda' or 'cpu'). Defaults to "cpu".
            inference_settings: Inference mode ("default" or "turbo")
            checkpoint_path: Optional path to custom checkpoint. If provided,
                           will load from checkpoint instead of pretrained model.
        
        Example:
            >>> extractor = UMAEmbeddingExtractor(model_name="uma-s-1p1", device="cuda")
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_name = model_name
        
        # Load the predict unit
        if checkpoint_path is not None:
            self.predict_unit = self._load_checkpoint_with_workaround(
                checkpoint_path,
                inference_settings=inference_settings,
                device=device,
            )
        else:
            self.predict_unit = pretrained_mlip.get_predict_unit(
                model_name=model_name,
                inference_settings=inference_settings,
                device=device,
            )
        
        # Access the backbone model for embedding extraction
        self.backbone = self.predict_unit.model.module.backbone.to(device)
        self.backbone.eval()
        
        # Access the full model (with heads) for per-atom energy extraction
        self.model = self.predict_unit.model.module.to(device)
        self.model.eval()
        
        # Clean up GPU memory if using checkpoint
        if checkpoint_path is not None:
            import gc
            gc.collect()
            if device == "cuda" or (isinstance(device, str) and "cuda" in device):
                torch.cuda.empty_cache()
    
    def atoms_to_data(
        self,
        atoms: Atoms,
        dataset: str = "omat",
        charge: int = 0,
        spin: int = 0,
    ) -> AtomicData:
        """
        Convert ASE Atoms object to AtomicData format required by UMA.
        
        Args:
            atoms: ASE Atoms object
            dataset: Dataset identifier (default: "omat")
            charge: System charge in elementary charges (default: 0)
            spin: System spin multiplicity (default: 0)
        
        Returns:
            AtomicData object ready for model input
        """
        # Add charge and spin to atoms.info for from_ase to read
        atoms_copy = atoms.copy()
        atoms_copy.info['charge'] = int(charge)
        atoms_copy.info['spin'] = int(spin)
        
        # Create atomic data from atoms using from_ase
        data = AtomicData.from_ase(
            atoms_copy,
            r_edges=False,  # Don't compute edges here, backbone will do it
            r_data_keys=["charge", "spin"],  # Request charge and spin
            task_name=[dataset],  # Set dataset name
        )
        
        return data
    
    def _load_checkpoint_with_workaround(
        self,
        checkpoint_path: str,
        inference_settings: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Load a checkpoint that might have incompatible parameters like 'inference_only'.
        
        This workaround filters out problematic parameters from the task config
        before instantiation.
        
        Args:
            checkpoint_path: Path to checkpoint file
            inference_settings: Optional inference settings ("default", "turbo", or None)
            device: Device to load model on
        
        Returns:
            MLIPPredictUnit instance
        """
        from fairchem.core.units.mlip_unit import MLIPPredictUnit
        from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
        from omegaconf import OmegaConf, DictConfig
        import torch
        import tempfile
        
        # Convert inference_settings string to InferenceSettings object
        if inference_settings == "default" or inference_settings is None:
            settings_obj = None
        elif inference_settings == "turbo":
            settings_obj = InferenceSettings(
                activation_checkpointing=False,
                inference_dtype="float16",
            )
        else:
            settings_obj = None
        
        # Load checkpoint (weights_only=False for compatibility with older checkpoints)
        # Load to CPU first to avoid GPU memory issues
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Handle both dictionary checkpoints and MLIPInferenceCheckpoint objects
        if hasattr(checkpoint, 'tasks_config'):
            # MLIPInferenceCheckpoint object
            tasks_config = checkpoint.tasks_config
            modified = False
            
            # Remove EMA weights to save GPU memory (use model weights instead)
            print("Removing EMA weights for inference (using model_state_dict instead)...")
            if hasattr(checkpoint, 'ema_state_dict') and checkpoint.ema_state_dict is not None:
                # Create EMA dict with model weights to avoid loading duplicate weights
                # The n_averaged needs to be a tensor, and weights go directly in the dict
                ema_dict = {'n_averaged': torch.tensor(1, dtype=torch.bool)}
                ema_dict.update(checkpoint.model_state_dict)
                checkpoint.ema_state_dict = ema_dict
                modified = True
            
            # Filter problematic parameters from each task
            for task in tasks_config:
                # Check if it's a dict or DictConfig
                if isinstance(task, (dict, DictConfig)):
                    if "inference_only" in task:
                        task.pop("inference_only")
                        modified = True
                    
                    # Add default loss_fn if missing (required for Task init)
                    if "loss_fn" not in task:
                        task["loss_fn"] = {
                            "_target_": "torch.nn.L1Loss"
                        }
                        modified = True
            
            if modified:
                # Update the checkpoint with modified config
                checkpoint.tasks_config = tasks_config
                
                # Save modified checkpoint to a temporary file
                with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
                    temp_path = tmp_file.name
                    torch.save(checkpoint, temp_path)
                
                # Load from the modified checkpoint
                try:
                    predict_unit = MLIPPredictUnit(
                        inference_model_path=temp_path,
                        inference_settings=settings_obj,
                        device=device,
                    )
                finally:
                    # Clean up temporary file
                    import os
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                
                return predict_unit
        
        elif isinstance(checkpoint, dict):
            # Standard checkpoint dictionary
            config = checkpoint.get("config", {})
            modified = False
            
            # Remove EMA weights and optimizer states to save GPU memory
            print("Removing EMA weights and optimizer states for inference...")
            if "ema_state_dict" in checkpoint:
                del checkpoint["ema_state_dict"]
                modified = True
            if "optimizer" in checkpoint:
                del checkpoint["optimizer"]
                modified = True
            if "scheduler" in checkpoint:
                del checkpoint["scheduler"]
                modified = True
            
            if "runner" in config and "train_eval_unit" in config["runner"]:
                train_eval_unit = config["runner"]["train_eval_unit"]
                
                # Filter tasks and tasks2 configs
                for task_key in ["tasks", "tasks2"]:
                    if task_key in train_eval_unit:
                        tasks = train_eval_unit[task_key]
                        if isinstance(tasks, dict):
                            for task_name, task_config in tasks.items():
                                if isinstance(task_config, dict):
                                    if "inference_only" in task_config:
                                        task_config.pop("inference_only")
                                        modified = True
                                    
                                    if "loss_fn" not in task_config:
                                        task_config["loss_fn"] = {
                                            "_target_": "torch.nn.L1Loss"
                                        }
                                        modified = True
            
            if modified:
                # Save modified checkpoint to a temporary file
                checkpoint["config"] = config
                
                with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
                    temp_path = tmp_file.name
                    torch.save(checkpoint, temp_path)
                
                # Load from the modified checkpoint
                try:
                    predict_unit = MLIPPredictUnit(
                        inference_model_path=temp_path,
                        inference_settings=settings_obj,
                        device=device,
                    )
                finally:
                    # Clean up temporary file
                    import os
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                
                return predict_unit
        
        # No modifications needed, load normally
        return MLIPPredictUnit(
            inference_model_path=checkpoint_path,
            inference_settings=settings_obj,
            device=device,
        )
    
    @torch.no_grad()
    def extract_embeddings(
        self,
        atoms: Union[Atoms, List[Atoms]],
        dataset: str = "omat",
        charge: Union[int, List[int]] = 0,
        spin: Union[int, List[int]] = 0,
        return_per_atom: bool = True,
        extract_per_atom_energies: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Extract backbone embeddings from atomic structure(s).
        
        The backbone returns embeddings with shape [num_atoms, lmax^2, channels].
        By default, we extract the l=0 (scalar) features which have shape [num_atoms, channels].
        
        Args:
            atoms: Single ASE Atoms object or list of Atoms objects
            dataset: Dataset identifier(s)
            charge: System charge(s) (integer)
            spin: System spin(s) (integer)
            return_per_atom: If True, return per-atom embeddings. If False,
                           return mean-pooled system-level embeddings.
            extract_per_atom_energies: If True, also extract per-atom energies from the energy head
        
        Returns:
            Dictionary containing:
                - 'embeddings': Node embeddings [num_atoms, channels] or [num_systems, channels]
                - 'atomic_numbers': Atomic numbers for each atom
                - 'num_atoms': Number of atoms per system
                - 'batch': Batch indices (which atom belongs to which structure)
                - 'per_atom_energies': Per-atom energies [num_atoms] (if extract_per_atom_energies=True)
        
        Example:
            >>> extractor = UMAEmbeddingExtractor()
            >>> atoms = read('structure.xyz')
            >>> result = extractor.extract_embeddings(atoms)
            >>> embeddings = result['embeddings']  # Shape: [num_atoms, 512] for uma-s
            >>> atom_energies = result['per_atom_energies']  # Shape: [num_atoms]
        """
        # Handle single atoms object
        if isinstance(atoms, Atoms):
            atoms = [atoms]
            if isinstance(charge, int):
                charge = [charge]
            if isinstance(spin, int):
                spin = [spin]
        
        all_embeddings = []
        all_atomic_numbers = []
        all_num_atoms = []
        all_batch_indices = []
        all_per_atom_energies = []
        
        for idx, atoms_obj in enumerate(atoms):
            # Convert to AtomicData
            data = self.atoms_to_data(
                atoms_obj,
                dataset=dataset,
                charge=charge[idx] if isinstance(charge, list) else charge,
                spin=spin[idx] if isinstance(spin, list) else spin,
            )
            
            # Move ALL tensor attributes to the correct device
            # We need to be explicit because AtomicData.to() doesn't always work properly
            tensor_attrs = ['pos', 'atomic_numbers', 'cell', 'pbc', 'natoms', 
                           'edge_index', 'cell_offsets', 'nedges', 'charge', 
                           'spin', 'fixed', 'tags', 'batch']
            
            for attr_name in tensor_attrs:
                if hasattr(data, attr_name):
                    attr = getattr(data, attr_name)
                    if isinstance(attr, torch.Tensor):
                        setattr(data, attr_name, attr.to(self.device))
            
            # Also check for optional attributes
            for attr_name in ['energy', 'forces', 'stress']:
                if hasattr(data, attr_name):
                    attr = getattr(data, attr_name)
                    if attr is not None and isinstance(attr, torch.Tensor):
                        setattr(data, attr_name, attr.to(self.device))
            
            # Forward pass through backbone
            output = self.backbone(data)
            
            # Extract node embeddings
            # Shape: [num_atoms, (lmax+1)^2, channels]
            node_embedding = output['node_embedding']
            
            # Extract scalar (l=0) features: [num_atoms, channels]
            # The first element (index 0) corresponds to l=0 features
            scalar_embedding = node_embedding[:, 0, :].cpu().numpy()
            
            # Extract per-atom energies if requested
            if extract_per_atom_energies:
                # Find the energy head for the dataset (default to first available)
                energy_head_name = f"{dataset}_energy"
                if energy_head_name not in self.model.output_heads:
                    # Try without dataset prefix
                    energy_head_name = "energy"
                
                if energy_head_name not in self.model.output_heads:
                    # Try common head names: efs (energy-forces-stress), energy, etc.
                    available_heads = list(self.model.output_heads.keys())
                    # Try 'efs' first (common in UMA models)
                    if 'efs' in available_heads:
                        energy_head_name = 'efs'
                        logging.info(f"Using energy head: {energy_head_name}")
                    else:
                        # Search for heads containing 'energy'
                        energy_heads = [h for h in available_heads if 'energy' in h.lower()]
                        if energy_heads:
                            energy_head_name = energy_heads[0]
                            logging.info(f"Using energy head: {energy_head_name}")
                        elif available_heads:
                            # Use first available head as fallback
                            energy_head_name = available_heads[0]
                            logging.info(f"Using first available head: {energy_head_name}")
                        else:
                            energy_head_name = None
                
                if energy_head_name and energy_head_name in self.model.output_heads:
                    energy_head = self.model.output_heads[energy_head_name]
                    # Get per-atom energies from the energy block
                    node_energy = energy_head.energy_block(
                        node_embedding[:, 0, :]
                    ).view(-1).cpu().numpy()
                    all_per_atom_energies.append(node_energy)
                else:
                    # If no energy head found, store zeros
                    logging.warning(f"No energy head found. Available heads: {list(self.model.output_heads.keys())}")
                    all_per_atom_energies.append(np.zeros(len(atoms_obj)))
            
            # Store results
            all_embeddings.append(scalar_embedding)
            all_atomic_numbers.append(atoms_obj.get_atomic_numbers())
            all_num_atoms.append(len(atoms_obj))
            all_batch_indices.append(np.full(len(atoms_obj), idx))
        
        # Concatenate all results
        embeddings = np.vstack(all_embeddings)
        atomic_numbers = np.concatenate(all_atomic_numbers)
        batch_indices = np.concatenate(all_batch_indices)
        
        result = {
            'embeddings': embeddings,
            'atomic_numbers': atomic_numbers,
            'num_atoms': np.array(all_num_atoms),
            'batch': batch_indices,
        }
        
        if extract_per_atom_energies and all_per_atom_energies:
            result['per_atom_energies'] = np.concatenate(all_per_atom_energies)

        
        # Optionally aggregate to system-level embeddings
        if not return_per_atom:
            system_embeddings = []
            for idx in range(len(atoms)):
                mask = batch_indices == idx
                # Mean pooling over atoms in this system
                system_emb = embeddings[mask].mean(axis=0)
                system_embeddings.append(system_emb)
            
            result['embeddings'] = np.array(system_embeddings)
            result['batch'] = np.arange(len(atoms))
        
        return result
    
    @torch.no_grad()
    def extract_full_embeddings(
        self,
        atoms: Union[Atoms, List[Atoms]],
        dataset: str = "omat",
        charge: Union[int, List[int]] = 0,
        spin: Union[int, List[int]] = 0,
        extract_per_atom_energies: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Extract full equivariant embeddings including all l values.
        
        Unlike extract_embeddings which only returns l=0 scalar features,
        this method returns the complete embedding tensor including all
        spherical harmonic orders up to lmax.
        
        Args:
            atoms: Single ASE Atoms object or list of Atoms objects
            dataset: Dataset identifier(s)
            charge: System charge(s) (integer)
            spin: System spin(s) (integer)
            extract_per_atom_energies: If True, also extract per-atom energies from the energy head
        
        Returns:
            Dictionary containing:
                - 'embeddings': Full embeddings [num_atoms, (lmax+1)^2, channels]
                - 'atomic_numbers': Atomic numbers for each atom
                - 'num_atoms': Number of atoms per system
                - 'batch': Batch indices
                - 'per_atom_energies': Per-atom energies [num_atoms] (if extract_per_atom_energies=True)
        
        Example:
            >>> result = extractor.extract_full_embeddings(atoms)
            >>> full_emb = result['embeddings']  # Shape: [num_atoms, 9, 512] for lmax=2
            >>> atom_energies = result['per_atom_energies']  # Shape: [num_atoms]
        """
        # Handle single atoms object
        if isinstance(atoms, Atoms):
            atoms = [atoms]
            if isinstance(charge, int):
                charge = [charge]
            if isinstance(spin, int):
                spin = [spin]
        
        all_embeddings = []
        all_atomic_numbers = []
        all_num_atoms = []
        all_batch_indices = []
        all_per_atom_energies = []
        
        for idx, atoms_obj in enumerate(atoms):
            # Convert to AtomicData
            data = self.atoms_to_data(
                atoms_obj,
                dataset=dataset,
                charge=charge[idx] if isinstance(charge, list) else charge,
                spin=spin[idx] if isinstance(spin, list) else spin,
            )
            
            # Move ALL tensor attributes to the correct device
            # We need to be explicit because AtomicData.to() doesn't always work properly
            tensor_attrs = ['pos', 'atomic_numbers', 'cell', 'pbc', 'natoms', 
                           'edge_index', 'cell_offsets', 'nedges', 'charge', 
                           'spin', 'fixed', 'tags', 'batch']
            
            for attr_name in tensor_attrs:
                if hasattr(data, attr_name):
                    attr = getattr(data, attr_name)
                    if isinstance(attr, torch.Tensor):
                        setattr(data, attr_name, attr.to(self.device))
            
            # Also check for optional attributes
            for attr_name in ['energy', 'forces', 'stress']:
                if hasattr(data, attr_name):
                    attr = getattr(data, attr_name)
                    if attr is not None and isinstance(attr, torch.Tensor):
                        setattr(data, attr_name, attr.to(self.device))
            
            # Forward pass through backbone
            output = self.backbone(data)
            
            # Extract full node embeddings
            # Shape: [num_atoms, (lmax+1)^2, channels]
            node_embedding = output['node_embedding']
            
            # Extract per-atom energies if requested
            if extract_per_atom_energies:
                # Find the energy head for the dataset (default to first available)
                energy_head_name = f"{dataset}_energy"
                if energy_head_name not in self.model.output_heads:
                    # Try without dataset prefix
                    energy_head_name = "energy"
                
                if energy_head_name not in self.model.output_heads:
                    # Try to find any energy head
                    available_heads = list(self.model.output_heads.keys())
                    # Try 'efs' first (common in UMA models)
                    if 'efs' in available_heads:
                        energy_head_name = 'efs'
                        logging.info(f"Using energy head: {energy_head_name}")
                    else:
                        # Search for heads containing 'energy'
                        energy_heads = [h for h in available_heads if 'energy' in h.lower()]
                        if energy_heads:
                            energy_head_name = energy_heads[0]
                            logging.info(f"Using energy head: {energy_head_name}")
                        elif available_heads:
                            # Use first available head as fallback
                            energy_head_name = available_heads[0]
                            logging.info(f"Using first available head: {energy_head_name}")
                        else:
                            energy_head_name = None
                
                if energy_head_name and energy_head_name in self.model.output_heads:
                    energy_head = self.model.output_heads[energy_head_name]
                    # Get per-atom energies from the energy block
                    node_energy = energy_head.energy_block(
                        node_embedding[:, 0, :]
                    ).view(-1).cpu().numpy()
                    all_per_atom_energies.append(node_energy)
                else:
                    # If no energy head found, store zeros
                    logging.warning(f"No energy head found. Available heads: {list(self.model.output_heads.keys())}")
                    all_per_atom_energies.append(np.zeros(len(atoms_obj)))
            
            # Move to CPU and convert to numpy
            node_embedding = node_embedding.cpu().numpy()
            
            # Store results
            all_embeddings.append(node_embedding)
            all_atomic_numbers.append(atoms_obj.get_atomic_numbers())
            all_num_atoms.append(len(atoms_obj))
            all_batch_indices.append(np.full(len(atoms_obj), idx))
        
        # Concatenate all results
        embeddings = np.vstack(all_embeddings)
        atomic_numbers = np.concatenate(all_atomic_numbers)
        batch_indices = np.concatenate(all_batch_indices)
        
        result = {
            'embeddings': embeddings,
            'atomic_numbers': atomic_numbers,
            'num_atoms': np.array(all_num_atoms),
            'batch': batch_indices,
        }
        
        if extract_per_atom_energies and all_per_atom_energies:
            result['per_atom_energies'] = np.concatenate(all_per_atom_energies)
        
        return result
    
    def get_embedding_dim(self) -> int:
        """
        Get the dimension of scalar (l=0) embeddings.
        
        Returns:
            Embedding dimension (e.g., 512 for uma-s, 1024 for uma-m)
        """
        return self.backbone.sphere_channels
    
    def get_full_embedding_shape(self) -> tuple:
        """
        Get the shape of full equivariant embeddings.
        
        Returns:
            Tuple of ((lmax+1)^2, channels)
        """
        lmax = self.backbone.lmax
        return ((lmax + 1) ** 2, self.backbone.sphere_channels)


def extract_embeddings_from_file(
    file_path: Union[str, Path],
    model_name: str = "uma-s-1p1",
    device: Optional[str] = None,
    index: str = ":",
    **kwargs,
) -> Dict[str, np.ndarray]:
    """
    Convenience function to extract embeddings directly from a structure file.
    
    Args:
        file_path: Path to structure file (xyz, cif, vasp, etc.)
        model_name: Name of pretrained UMA model
        device: Device to use ('cuda' or 'cpu')
        index: ASE index string for selecting structures (default: ':' for all)
        **kwargs: Additional arguments passed to extract_embeddings()
    
    Returns:
        Dictionary containing embeddings and metadata
    
    Example:
        >>> result = extract_embeddings_from_file('trajectory.xyz', model_name='uma-s-1p1')
        >>> embeddings = result['embeddings']
    """
    # Read structures
    atoms_list = read(file_path, index=index)
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    
    # Create extractor and extract embeddings
    extractor = UMAEmbeddingExtractor(model_name=model_name, device=device)
    return extractor.extract_embeddings(atoms_list, **kwargs)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python extract_uma_embeddings.py <structure_file>")
        print("\nExample:")
        print("  python extract_uma_embeddings.py structure.xyz")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    print(f"Extracting embeddings from {file_path}...")
    print(f"Using model: uma-s-1p1")
    
    # Extract embeddings
    result = extract_embeddings_from_file(file_path, model_name="uma-s-1p1")
    
    print(f"\nResults:")
    print(f"  Embedding shape: {result['embeddings'].shape}")
    print(f"  Number of atoms: {result['num_atoms']}")
    print(f"  Number of structures: {len(result['num_atoms'])}")
    print(f"  Embedding dimension: {result['embeddings'].shape[1]}")
    
    # Save results
    output_file = Path(file_path).stem + "_embeddings.npz"
    np.savez_compressed(output_file, **result)
    print(f"\nSaved embeddings to: {output_file}")
