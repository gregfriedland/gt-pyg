# Standard
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from numpy.linalg import pinv
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch_geometric.data import Data


class NodeCategoricalFeatures:
    ATOM_TYPES = [
        "C",
        "N",
        "O",
        "S",
        "F",
        "Si",
        "P",
        "Cl",
        "Br",
        "Mg",
        "Na",
        "Ca",
        "Fe",
        "As",
        "Al",
        "I",
        "B",
        "V",
        "K",
        "Tl",
        "Yb",
        "Sb",
        "Sn",
        "Ag",
        "Pd",
        "Co",
        "Se",
        "Ti",
        "Zn",
        "Li",
        "Ge",
        "Cu",
        "Au",
        "Ni",
        "Cd",
        "In",
        "Mn",
        "Zr",
        "Cr",
        "Pt",
        "Hg",
        "Pb",
        "Unknown",
    ]
    ATOM_DEGREES = [0, 1, 2, 3, 4, 5]
    FORMAL_CHARGES = [-3, -2, -1, 0, 1, 2, 3, 4]
    HYBRIDISATION_TYPES = ["S", "SP", "SP2", "SP2D", "SP3", "SP3D", "SP3D2", "OTHER"]
    NUM_HYDROGENS = [0, 1, 2, 3, 4]
    IS_IN_RING = [0, 1]
    IS_AROMATIC = [0, 1]

    def __init__(self, atom: Chem.Atom):
        self.atom = atom

    def __call__(self):
        atom_type_enc = get_categorical_id(str(self.atom.GetSymbol()), self.ATOM_TYPES)

        n_heavy_neighbors_enc = get_categorical_id(
            int(self.atom.GetDegree()), self.ATOM_DEGREES
        )

        formal_charge_enc = get_categorical_id(
            int(self.atom.GetFormalCharge()), self.FORMAL_CHARGES
        )

        hybridisation_type_enc = get_categorical_id(
            str(self.atom.GetHybridization()), self.HYBRIDISATION_TYPES
        )

        n_hydrogen_enc = get_categorical_id(
            int(self.atom.GetTotalNumHs()), self.NUM_HYDROGENS
        )

        is_in_a_ring_enc = int(self.atom.IsInRing())

        is_aromatic_enc = int(self.atom.GetIsAromatic())

        return [
            atom_type_enc,
            n_heavy_neighbors_enc,
            formal_charge_enc,
            hybridisation_type_enc,
            n_hydrogen_enc,
            is_in_a_ring_enc,
            is_aromatic_enc,
        ]

    @classmethod
    def get_counts(self) -> List[int]:
        return [
            len(self.ATOM_TYPES),
            len(self.ATOM_DEGREES),
            len(self.FORMAL_CHARGES),
            len(self.HYBRIDISATION_TYPES),
            len(self.NUM_HYDROGENS),
            len(self.IS_IN_RING),
            len(self.IS_AROMATIC),
        ]

    @classmethod
    def num_features(self) -> int:
        return len(self.get_counts())


class NodeContinuousFeatures:
    MIN_ATOM = Chem.rdchem.Atom("H")
    MAX_ATOM = Chem.rdchem.Atom("Pb")

    def __init__(self, atom: Chem.Atom):
        self.atom = atom

    def __call__(self) -> list[float]:
        atomic_mass = (
            float(self.atom.GetMass())
            if str(self.atom.GetSymbol()) != "Unknown"
            else self.MAX_ATOM.GetMass()
        )
        atomic_mass_scaled = (atomic_mass - self.MIN_ATOM.GetMass()) / (
            self.MAX_ATOM.GetMass() - self.MIN_ATOM.GetMass()
        )

        vdw_radius = (
            Chem.GetPeriodicTable().GetRvdw(self.atom.GetAtomicNum())
            if str(self.atom.GetSymbol()) != "Unknown"
            else self.MAX_ATOM.GetAtomicNum()
        )
        vdw_radius_scaled = (vdw_radius - self.MIN_ATOM.GetAtomicNum()) / (
            self.MAX_ATOM.GetAtomicNum() - self.MIN_ATOM.GetAtomicNum()
        )

        covalent_radius = (
            Chem.GetPeriodicTable().GetRcovalent(self.atom.GetAtomicNum())
            if str(self.atom.GetSymbol()) != "Unknown"
            else self.MAX_ATOM.GetAtomicNum()
        )
        covalent_radius_scaled = (covalent_radius - self.MIN_ATOM.GetAtomicNum()) / (
            self.MAX_ATOM.GetAtomicNum() - self.MIN_ATOM.GetAtomicNum()
        )

        return [
            atomic_mass_scaled,
            vdw_radius_scaled,
            covalent_radius_scaled,
        ]

    @classmethod
    def num_features(self) -> int:
        return 3


class EdgeCategoricalFeatures:
    BOND_TYPES = [
        "SINGLE",
        "DOUBLE",
        "TRIPLE",
        "AROMATIC",
    ]
    STEREO_TYPES = [
        "STEREOZ",
        "STEREOE",
        "STEREOANY",
        "STEREONONE",
    ]
    RING_TYPES = [
        "NONE",
        "SIZE_5",
        "SIZE_6",
        "OTHER",
    ]

    def __init__(self, bond: Chem.Bond):
        self.bond = bond

    def __call__(self) -> list[int]:
        bond_type_enc = get_categorical_id(
            str(self.bond.GetBondType()), self.BOND_TYPES
        )

        bond_ring_type = "NONE"
        if self.bond.IsInRing():
            if self.bond.IsInRingSize(5):
                bond_ring_type = "SIZE_5"
            elif self.bond.IsInRingSize(6):
                bond_ring_type = "SIZE_6"
            else:
                bond_ring_type = "OTHER"
        bond_ring_size_enc = get_categorical_id(bond_ring_type, self.RING_TYPES)

        bond_is_conj_enc = int(self.bond.GetIsConjugated())

        stereo_type_enc = get_categorical_id(
            str(self.bond.GetStereo()), self.STEREO_TYPES
        )

        return [
            bond_type_enc,
            bond_ring_size_enc,
            bond_is_conj_enc,
            stereo_type_enc,
        ]

    @classmethod
    def num_features(self) -> int:
        return len(self.get_counts())

    @classmethod
    def get_counts(self) -> List[int]:
        return [
            len(self.BOND_TYPES),
            len(self.RING_TYPES),
            2,
            len(self.STEREO_TYPES),
        ]


class PositionalEncoding:
    def __init__(self, mol: Chem.Mol, pe_dim: int = 6):
        self.mol = mol
        self.pe_dim = pe_dim

    def __call__(self):
        pe = get_pe(self.mol, pe_dim=self.pe_dim)
        return pe

    @classmethod
    def num_features(self) -> int:
        return 6


def clean_df(
    tdc_df: pd.DataFrame,
    min_num_atoms: int = 0,
    use_largest_fragment=True,
    x_label="Drug",
    y_label="Y",
) -> pd.DataFrame:
    """
    Cleans a DataFrame containing chemical structures by removing rows that do not meet certain criteria.

    Args:
        tdc_df (pd.DataFrame): The input DataFrame containing chemical structures.
        min_num_atoms (int, optional): The minimum number of atoms required for a structure to be considered valid.
            Set to 0 for no size-based filtering. Defaults to 0.
        use_largest_fragment (bool, optional): Whether to use the largest fragment when cleaning the data.
            Defaults to True.
        x_label (str, optional): Label of the column to be used for X variable in the cleaned DataFrame.
            Defaults to 'Drug'.
        y_label (str, optional): Label of the column to be used for Y variable in the cleaned DataFrame.
            Defaults to 'Y'.

    Returns:
        pd.DataFrame: A cleaned DataFrame with rows that satisfy the specified criteria.
    """

    def count_fragments(mol):
        # Helper function to count the number of fragments in a molecule.
        frags = Chem.GetMolFrags(mol)
        return len(frags)

    def get_largest_fragment(mol):
        # Remove the counterions
        mol = Chem.RemoveHs(mol)

        # Get the disconnected fragments
        fragments = Chem.GetMolFrags(mol, asMols=True)

        # Calculate the number of heavy atoms in each fragment
        num_atoms = [frag.GetNumHeavyAtoms() for frag in fragments]

        # Identify the index of the largest fragment
        largest_frag_index = num_atoms.index(max(num_atoms))

        # Get the SMILES representation of the largest fragment
        largest_frag_smiles = Chem.MolToSmiles(fragments[largest_frag_index])

        return largest_frag_smiles

    def count_atoms(mol):
        # Helper function to count the number of atoms in a molecule.
        return len(mol.GetAtoms())

    # Disable RDKit logging messages
    for log_level in RDLogger._levels:
        rdBase.DisableLog(log_level)

    # Convert SMILES strings to RDKit Mol objects
    tdc_df["mol"] = tdc_df[x_label].apply(Chem.MolFromSmiles)

    # Calculate the number of fragments and atoms for each molecule
    tdc_df["num_frags"] = tdc_df.mol.apply(count_fragments)
    tdc_df["largest_fragment"] = tdc_df.mol.apply(get_largest_fragment)
    tdc_df["num_atoms"] = tdc_df.mol.apply(count_atoms)

    # Filter out rows with more than one fragment and fewer atoms than the specified minimum
    initial_length = len(tdc_df)
    if use_largest_fragment:
        tdc_df[x_label] = tdc_df["largest_fragment"].to_list()
        fragments_removed = 0
    else:
        tdc_df = tdc_df.query("num_frags == 1").copy()
        fragments_removed = initial_length - len(tdc_df)
        logging.info(
            f"Removed {fragments_removed} compounds that have more than 1 fragment."
        )

    if min_num_atoms > 0:
        tdc_df = tdc_df.query(f"num_atoms >= {min_num_atoms}").copy()
        removed_cmpds = initial_length - len(tdc_df) + fragments_removed
        logging.info(
            f"Removed {removed_cmpds} compounds that did not meet the size criteria."
        )

    tdc_df = tdc_df[[x_label, y_label]]
    return tdc_df


def get_train_valid_test_data(
    endpoint: str, min_num_atoms: int = 0, use_largest_fragment: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Retrieves and cleans the train, validation, and test data for a specific endpoint in the ADME dataset.

    Args:
        endpoint (str): The name of the endpoint in the ADME dataset.
        min_num_atoms (int, optional): The minimum number of atoms required for a structure to be considered valid.
            Set to 0 for no size-based filtering. Defaults to 0.
        use_largest_fragment (bool, optional): Whether to use the largest fragment when cleaning the data.
            Defaults to True.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing cleaned train, validation, and test DataFrames.

    """
    try:
        from tdc.single_pred import ADME
    except ImportError:
        raise

    data = ADME(name=endpoint)
    splits = data.get_split()

    train_data = clean_df(
        splits["train"],
        min_num_atoms=min_num_atoms,
        use_largest_fragment=use_largest_fragment,
    )
    valid_data = clean_df(
        splits["valid"],
        min_num_atoms=min_num_atoms,
        use_largest_fragment=use_largest_fragment,
    )
    test_data = clean_df(
        splits["test"],
        min_num_atoms=min_num_atoms,
        use_largest_fragment=use_largest_fragment,
    )

    return (train_data, valid_data, test_data)


def get_molecule_ace_datasets(
    dataset: str,
    training_fraction: float = 1.0,
    valid_fraction: float = 0.0,
    seed: int = 42,
    min_num_atoms: int = 0,
    use_largest_fragment: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Retrieve molecule ACE datasets.

    Args:
        dataset (str): Name of the dataset.
        training_fraction (float, optional): Fraction of data to use for training. Defaults to 1.0.
        valid_fraction (float, optional): Fraction of data to use for validation. Defaults to 0.0.
        seed (int, optional): Random seed for shuffling. Defaults to 42.
        min_num_atoms (int, optional): Minimum number of atoms required for a molecule to be included. Defaults to 0.
        use_largest_fragment (bool, optional): Whether to use the largest fragment of a molecule. Defaults to True.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing train, validation, and test datasets as pandas DataFrames.
    """
    try:
        from MoleculeACE import Data
    except ImportError:
        logging.info("Importing MoleculeACE failed")
        raise

    data = Data(dataset)
    df_train_tmp = pd.DataFrame({"SMILES": data.smiles_train, "Y": data.y_train})
    df_test = pd.DataFrame({"SMILES": data.smiles_test, "Y": data.y_test})

    shuffled_df = df_train_tmp.sample(frac=1, random_state=seed)

    ratio = training_fraction / (training_fraction + valid_fraction)
    # Calculate the split index based on the ratio
    split_index = int(ratio * len(shuffled_df))

    # Split the shuffled DataFrame into two separate DataFrames
    df_train = shuffled_df.iloc[:split_index]
    df_valid = shuffled_df.iloc[split_index:]

    train_data = clean_df(
        df_train,
        min_num_atoms=min_num_atoms,
        use_largest_fragment=use_largest_fragment,
        x_label="SMILES",
    )
    valid_data = clean_df(
        df_valid,
        min_num_atoms=min_num_atoms,
        use_largest_fragment=use_largest_fragment,
        x_label="SMILES",
    )
    test_data = clean_df(
        df_test,
        min_num_atoms=min_num_atoms,
        use_largest_fragment=use_largest_fragment,
        x_label="SMILES",
    )

    return (train_data, valid_data, test_data)


def get_data_from_csv(
    filename: str,
    x_label: str,
    y_label: str,
    sep: str = ",",
    min_num_atoms: int = 0,
    use_largest_fragment: bool = True,
) -> pd.DataFrame:
    """
    Reads data from a CSV file and returns a cleaned DataFrame containing specified columns.

    Parameters:
        filename (str): Path to the CSV file.
        x_label (str): Label of the column to be used as the X variable.
        y_label (str): Label of the column to be used as the Y variable.
        sep (str, optional): Separator used in the CSV file. Default is ','.
        min_num_atoms (int, optional): The minimum number of atoms required for a structure to be considered valid.
            Set to 0 for no size-based filtering. Defaults to 0.
        use_largest_fragment (bool, optional): Whether to use the largest fragment when cleaning the data.
            Defaults to True.

    Returns:
        pandas.DataFrame: A cleaned DataFrame containing only the specified X and Y columns.

    Example:
        data = get_data_from_csv('data.csv', 'X', 'Y')
    """
    df = pd.read_csv(filename, sep=sep)
    df = df[[x_label, y_label]]

    data = clean_df(
        df,
        min_num_atoms=min_num_atoms,
        use_largest_fragment=use_largest_fragment,
        x_label=x_label,
        y_label=y_label,
    )
    return data


def get_categorical_id(x: str, permitted_list: List[str]) -> int:
    """
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.

    Args:
        x: The input element to be encoded.
        permitted_list: The list of permitted elements.

    Returns:
        binary_encoding: A list representing the binary encoding of the input element.
    """
    if x not in permitted_list:
        x = permitted_list[-1]

    return permitted_list.index(x)


def get_pe(mol: Chem.Mol, pe_dim: int = 6, normalized: bool = True) -> np.ndarray:
    """
    Calculates the graph signal using the normalized Laplacian.

    Args:
        mol: The input molecule.
        pe_dim: The number of dimensions to keep in the graph signal. Defaults to 6.
        normalized: Specifies whether to use normalized Laplacian. Defaults to True.

    Returns:
        np.ndarray: The graph signal of the molecule.
    """
    adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
    degree = np.diag(np.sum(adj, axis=1))
    laplacian = degree - adj
    if normalized:
        degree_inv_sqrt = np.diag(np.sum(adj, axis=1) ** (-1.0 / 2.0))
        laplacian = degree_inv_sqrt @ laplacian @ degree_inv_sqrt
    try:
        val, vec = np.linalg.eig(laplacian)
    except:
        print(Chem.MolToSmiles(mol))
        raise

    vec = vec[:, np.argsort(val)]
    N = vec.shape[1]
    M = pe_dim + 1
    if N < M:
        vec = np.pad(vec, ((0, 0), (0, M - N)), mode="constant")

    return vec[:, 1:M]


def get_atom_features(atom: Chem.Atom) -> tuple[list[int], list[float]]:
    """
    Computes a 1D numpy array of atom features from an RDKit atom object.

    Args:
        atom (Chem.Atom): The RDKit atom object.
        use_chirality (bool, optional): Specifies whether to include chirality information. Defaults to True.
        hydrogens_implicit (bool, optional): Specifies whether to include implicit hydrogen count. Defaults to True.

    Returns:
        np.ndarray, np.ndarray: categorical and continuous atom features.
    """

    categorical_feats = NodeCategoricalFeatures(atom)()
    continuous_feats = NodeContinuousFeatures(atom)()

    return categorical_feats, continuous_feats


def get_bond_features(bond: Chem.Bond) -> list[int]:
    """
    Takes an RDKit bond object as input and gives a 1D numpy array of bond features as output.

    Args:
        bond (Chem.Bond): The RDKit bond object to extract features from.
        use_stereochemistry (bool, optional): Specifies whether to include stereochemistry features.
            Defaults to True.

    Returns:
        np.ndarray: A 1D numpy array of bond features (all categorical)
    """

    return EdgeCategoricalFeatures(bond)()


def get_gnn_encodings(mol):
    # Generate adjacency matrix
    adjacency_matrix = Chem.GetAdjacencyMatrix(mol)

    # Convert adjacency matrix to numpy array
    adjacency_np = np.array(adjacency_matrix)

    # Calculate the degree matrix
    degree_matrix = np.diag(np.sum(adjacency_np, axis=1))

    # Calculate the Laplacian matrix (Kirchhoff matrix)
    kirchhoff_matrix = degree_matrix - adjacency_np

    # Calculate the inverse of the Kirchhoff matrix
    inv_kirchhoff_matrix = pinv(kirchhoff_matrix)

    return inv_kirchhoff_matrix


def get_tensor_data_for_mol(
    smiles: str,
    y: List[float],
    gnn: bool = True,
    pe: bool = True,
    pe_dim: int = 6,
) -> Data:
    """
    Constructs a labeled molecular graph in the form of a torch_geometric.data.Data
    object using SMILES and associated numerical labels.

    Args:
        x_smiles (str): a SMILES string.
        y (List[float]): A list of numerical labels for the SMILES string (e.g., associated pKi values).
        gnn (bool, optional): Use Gaussian Network Model style positional encoding.
        pe (bool, optional): Specifies whether to include graph signal (PE) features. Defaults to True.
        pe_dim (int, optional): The number of dimensions to keep in the graph signal. Defaults to 6.

    Returns:
        Data: a torch_geometric.data.Data objects representing a labeled molecular graph
            x_cat: Categorical atom features dims (n_atoms, 7)
            x_cont: Continuous atom features dims (n_atoms, 4)
            edge_attr: Edge features dims (n_edges, 6)
            edge_index: Edge index dims (2, n_edges)
            pe: Positional encoding dims (n_atoms, pe_dim)
            y: Label tensor
    """

    # convert SMILES to RDKit mol object
    mol = Chem.MolFromSmiles(smiles)

    if gnn:
        dRdR = get_gnn_encodings(mol)
    else:
        dRdR = None

    # get feature dimensions
    x_continuous_all = []
    x_categorical_all = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        x_categorical, x_continuous = get_atom_features(atom)

        if dRdR is not None:
            x_continuous_all.append(x_continuous + [dRdR[idx][idx]])
        else:
            x_continuous_all.append(x_continuous)
        x_categorical_all.append(x_categorical)

    x_categorical_all = torch.tensor(np.array(x_categorical_all), dtype=torch.long)
    x_continuous_all = torch.tensor(np.array(x_continuous_all), dtype=torch.float)

    # construct edge index array edge_index of shape (2, n_edges)
    (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
    torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
    torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
    edge_index = torch.stack([torch_rows, torch_cols], dim=0)

    edge_attr = []
    if pe:
        pe_numpy = get_pe(mol, pe_dim=pe_dim)
        pe_tensor = torch.tensor(pe_numpy, dtype=torch.float)
    else:
        pe_tensor = None

    for k, (i, j) in enumerate(zip(rows, cols)):
        edge_attr.append(get_bond_features(mol.GetBondBetweenAtoms(int(i), int(j))))

    edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.int32)

    # construct label tensor
    y_tensor = torch.tensor(np.array(y), dtype=torch.float32)

    # construct Pytorch Geometric data object and append to data list
    return Data(
        x_cat=x_categorical_all,
        x_cont=x_continuous_all,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pe=pe_tensor,
        y=y_tensor,
        num_nodes=x_categorical_all.size(0),
    )
