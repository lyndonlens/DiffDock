# get sequence from pdb file
from Bio.PDB import PDBParser
file = '../data/dummy_data/4v24.pdb'
parser = PDBParser(PERMISSIVE=1)
structure = parser.get_structure('4v24', file)

from Bio.PDB import PPBuilder
ppb = PPBuilder()
for pp in ppb.build_peptides(structure):
    print(pp.get_sequence())