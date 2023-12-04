import copy
import os
import warnings

import numpy as np
from Bio.PDB import PDBParser, Superimposer
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from rdkit import Chem
from scipy import spatial
# from Bio.PDB.Selection import unfold_entities

biopython_parser = PDBParser()

def align_pdbs(pdb1:str, pdb2:str, out='align.pdb'): #效果很差
    # 读取两个PDB文件
    parser = PDBParser(QUIET=True)
    structure1 = parser.get_structure('structure1', pdb1)
    structure2 = parser.get_structure('structure2', pdb2)

    # 创建Superimposer对象并进行对齐
    superimposer = Superimposer()
    atoms1 = []
    atoms2 = []

    for model1, model2 in zip(structure1, structure2):
        for chain1, chain2 in zip(model1, model2):
            for atom1, atom2 in zip(chain1.get_atoms(), chain2.get_atoms()):
                atoms1.append(atom1)
                atoms2.append(atom2)

    superimposer.set_atoms(atoms1, atoms2)
    superimposer.apply(structure2[0])

    # 输出对齐后的结构
    output = PDBIO()
    output.set_structure(structure2)
    output.save(out)


def parse_receptor(pdbid, pdbbind_dir):
    rec = parsePDB(pdbid, pdbbind_dir)
    return rec


def parsePDB(pdbid, pdbbind_dir):
    rec_path = os.path.join(pdbbind_dir, pdbid, f'{pdbid}_protein_processed.pdb')
    return parse_pdb_from_path(rec_path)

def parse_pdb_from_path(path):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        structure = biopython_parser.get_structure('random_id', path)
        rec = structure[0]
    return rec

# 注意这里注释的很多shape，其实是list的每层的length，而不是真正的tensor shape
# 这里是对每层结构依次展开。最简单的做法是用biopython自带的unfold_entities来把左右坐标和原子都展开，不过这样不方便检查pdb中的一些原子错误，也不利于对整个pdb数据结构的理解
def get_AAs_around_ligand(rec, lig, radius=5): # lig是对接后的三维构象，从sdf文件读取
    conf = lig.GetConformer()
    lig_coords = conf.GetPositions() # 结果是个array，N*3， N为原子数。包含H原子

    min_distances = []
    atoms = []
    coords = []
    valid_chain_ids = []
    lengths = []
    contacts = set()
    for i, chain in enumerate(rec): # rec 是Bio.PDB.PDBParser读取的pdb结果取到model级别的类
        # Biopython的结构采用SMCRA体系架构(Structure/Model/Chain/Residue/Atom,结构/模型/链/残基/原子)
        chain_atoms = []
        chain_coords = []  # num_residues, num_atoms, 3
        count = 0
        invalid_res_ids = [] # 忽略非结晶水
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == 'HOH': # ignore H2O
                invalid_res_ids.append(residue.get_id())
                continue
            residue_atoms = []
            residue_coords = []
            c_alpha = None
            for atom in residue:
                if atom.name == 'CA':
                    c_alpha = list(atom.get_vector()) # get_vector()返回三维坐标, e.g. <Vector -28.63, 64.96, -41.38>. 使用list直接转为一个list
                residue_atoms.append(atom)
                residue_coords.append(list(atom.get_vector())) # all atoms in a residue [n_atom_per_residue, 3]

            if c_alpha != None: # 跳过莫名其妙的分子片段。残基都有CA
                chain_atoms.append(residue_atoms)
                chain_coords.append(np.array(residue_coords)) # coords for all atoms. [N_AA_per_chain, n_atom_per_residue, 3]
                count += 1
            else:
                invalid_res_ids.append(residue.get_id())
        for res_id in invalid_res_ids:
            chain.detach_child(res_id) # 去除杂原子

        if len(chain_coords) > 0:
            all_chain_coords = np.concatenate(chain_coords, axis=0) # 对每个链的原子坐标展开，[N_aa*N_atom_per_aa, 3]
            distances = spatial.distance.cdist(lig_coords, all_chain_coords) # [M, N], M为lig原子数，N为chain原子数
            min_distance = distances.min() #
        else:
            min_distance = np.inf

        atoms.append(chain_atoms)
        min_distances.append(min_distance)
        lengths.append(count)
        coords.append(chain_coords)
        if not count == 0: valid_chain_ids.append(chain.get_id()) # if Len>0, good chain

    if len(valid_chain_ids) == 0:
        valid_chain_ids.append(np.argmin(min_distances)) # if no valid chain, use the closest chain as the receptor
    valid_atoms = []
    valid_coords = []

    invalid_chain_ids = []
    for i, chain in enumerate(rec): # rec是SMCRA中的model级别
        print('chain_id:', i)
        if chain.get_id() in valid_chain_ids:
            valid_atoms.append(atoms[i])
            valid_coords.append(coords[i]) # [n_chain, n_AA, n_atom, 3]
        else:
            invalid_chain_ids.append(chain.get_id())

    # 展开所有原子和坐标
    residue_atoms = [item for sublist in valid_atoms for item in sublist]
    atoms = [item for sublist in residue_atoms for item in sublist]
    residue_coords = [item for sublist in valid_coords for item in sublist] # [n_chain*n_residue, n_atoms_per_residue, 3]
    coords = [item for sublist in residue_coords for item in sublist]
    assert len(atoms)==len(coords), 'Atom list not compatible with coordinate list'

    all_coords = np.concatenate(residue_coords, axis=0)  # 对每个链的原子坐标展开，[N_aa*N_atom_per_aa, 3]
    distances = spatial.distance.cdist(lig_coords, all_coords)  # [M, N], M为lig原子数，N为chain原子数
    contacts = contacts.union(set(np.array(np.where(distances < radius))[1, :]))
    if len(contacts) == 0:
        print('No contacts detected')
    # print('Atom list around the ligand:\n', *contacts)

    # get residue info
    residue_names = [atoms[i].get_parent().get_resname() for i in contacts] # residue name
    residue_ids = [atoms[i].get_parent().get_id()[1] for i in contacts]  # residue id
    chain_ids = [atoms[i].get_parent().parent.get_id() for i in contacts] # residue name
    contact_redidues = {id:(res, chain) for id, res, chain in zip(residue_ids, residue_names, chain_ids)}
    contact_residues_simple = list(set([c+':'+str(id)+res for id, res, c in zip(residue_ids, residue_names, chain_ids)]))
    # print(residue_names, residue_ids, chain_ids)
    return contact_redidues, contact_residues_simple


# return 3D mol. Why remove Hs ?
def read_molecule(molecule_file, sanitize=False, calc_charges=False, remove_hs=False):
    if molecule_file.endswith('.mol2'):
        mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False)
        mol = supplier[0]

    elif molecule_file.endswith('.pdbqt'):
        #The pdbqt format is 'pdb' plus 'q' for partial charge and 't' for AD4 atom type. Special AD4 atom types are OA,NA,SA for hbond accepting O,N and S atoms,
        # HD for hbond donor H atoms,N for non-hydrogen bonding nitrogens and A for carbons in planar cycles.
        # For any other atom, its AD4 atom type is the same as its element.
        with open(molecule_file) as file:
            pdbqt_data = file.readlines()
        pdb_block = ''
        for line in pdbqt_data:
            pdb_block += '{}\n'.format(line[:66])
            # HETATM    1  C   UNL     1       3.689  26.714   6.935  1.00  0.00     0.032 C
            # [:66] will omit charge and AD4 atom types
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
    else:
        raise ValueError('Expect the format of the molecule_file to be '
                         'one of .mol2, .sdf, .pdbqt and .pdb, got {}'.format(molecule_file))

    try:
        if sanitize or calc_charges:
            Chem.SanitizeMol(mol)

        if calc_charges:
            # Compute Gasteiger charges on the molecule.
            try:
                AllChem.ComputeGasteigerCharges(mol)
            except:
                warnings.warn('Unable to compute charges for the molecule.')

        if remove_hs:
            mol = Chem.RemoveHs(mol, sanitize=sanitize)
    except Exception as e:
        print(e)
        print("RDKit was unable to read the molecule.")
        return None
    return mol

if __name__ == '__main__':
    # pdb_file = '/home/tianxh/projects/DiffDock/data/dummy_data/1c1d.pdb'
    # lig_file = '/home/tianxh/projects/DiffDock/data/dummy_data/rank1.sdf'
    pdb_file = '../data/1a30/1a30_protein.pdb'
    lig_file = '../data/1a30/1a30_ligand.sdf'

    # parser = PDBParser(PERMISSIVE=1)
    # rec = parser.get_structure('temp', pdb_file)[0] # model level
    # mol = read_molecule(lig_file)
    # mol = Chem.AddHs(mol) # 给读取的sdf 构象加氢
    #
    # residues, residues_simple = get_AAs_around_ligand(rec, mol, radius=4)
    # print(residues)
    # print(residues_simple) # 忽略了chain的信息

    # pdb1 = '../data/4ynu/4ynu_h.pdb' # 目标坐标系
    # pdb2 = '../data/4ynu/7vkd.pdb'
    # for _ in range(3):
    #     align_pdbs(pdb1, pdb2, '../data/4ynu/7vkd.pdb')