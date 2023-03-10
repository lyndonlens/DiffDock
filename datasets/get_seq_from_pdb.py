# get sequence from pdb file
import os
from Bio.PDB import PDBParser, PPBuilder

def get_seq_from_pdb(pdb_file, out_file=None):
    seq_name = os.path.basename(pdb_file)[:-4]
    parser = PDBParser(PERMISSIVE=1)
    structure = parser.get_structure(seq_name, pdb_file)
    # model = structure[0]
    ppb = PPBuilder()
    for pp in ppb.build_peptides(structure):
        seq = pp.get_sequence().__str__()
        print('Sequence of this PDB: ')
        print((f'>{seq_name}'))
        print(seq)
    if out_file is None:
        return seq_name, seq
    if out_file is not None:
        f = open(out_file, 'w')
        f.write(f'>{seq_name}\n')
        f.write(seq+'\n')
        f.close()
        print(f'Sequence written to {out_file}')
    return out_file

def get_seqs_from_pdbs(pdb_list, out_file):
    f = open(out_file, 'w')
    for pdb_file in pdb_list:
        try:
            seq_name, seq = get_seq_from_pdb(pdb_file)
            f.write(f'>{seq_name}\n')
            f.write(seq+'\n')
        except:
            raise ValueError(f'Could not parse PDB file {pdb_file}')
    f.close()
    print(f'Sequence written to {out_file}')
    return out_file

if __name__ == '__main__':
    pdb_files = ['../data/dummy_data/48_mt_K68S_D261L_E114V_T134C_P146V_H187D_V294C_99.pdb',
                '../data/dummy_data/42_mt_K68S_D261L_P146V_H187D_V294C_88.pdb']

    # 提取单个
    pdb_file = pdb_files[0]
    seq_name = os.path.basename(pdb_file)[:-4]
    # fasta_file = os.path.join(os.path.dirname(pdb_file), seq_name + '.fasta')
    fasta_file = os.path.join(os.path.dirname(pdb_file), 'test.fasta')
    # get_seq_from_pdb(pdb_file, out_file=fasta_file)
    get_seqs_from_pdbs(pdb_files, fasta_file)
    # 同时提取多个，写入同一fasta文件
