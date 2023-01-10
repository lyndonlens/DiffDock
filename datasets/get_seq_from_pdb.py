# get sequence from pdb file
from Bio.PDB import PDBParser, PPBuilder
name = 'taq_T'
file = f'../data/dummy_data/{name}.pdb'
fasta_file = file[:-4]+'.fasta'
parser = PDBParser(PERMISSIVE=1)
structure = parser.get_structure(name, file)
model = structure[0]

ppb = PPBuilder()
with open(fasta_file, 'w') as w:
    w.write(f'>{name}\n')
    for pp in ppb.build_peptides(structure):
        seq = pp.get_sequence()
        # print(seq.__str__())
        # print(dir(seq))
        w.write(seq.__str__())