import networkx as nx
import numpy as np
import torch, copy
from scipy.spatial.transform import Rotation as R
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

"""
    Preprocessing and computation for torsional updates to conformers
"""

# 一个键首要定义其方向，然后才能定义旋转的正方向和逆方向。
# 施加torsion时，起始原子（及这一端的那堆原子）不允许旋转，键的另外一端的那些原子可以旋转。注意，键两头的两个原子是不动的
def get_transformation_mask(pyg_data): # pyg_data with H nodes
    G = to_networkx(pyg_data.to_homogeneous(), to_undirected=False) #
    to_rotate = []
    edges = pyg_data['ligand', 'ligand'].edge_index.T.numpy() # [num_edge_type, 2]
    print(edges)
    '''
    edges: CC(=O)CCO
    [[0 1]
     [1 0]
     [1 2]
     [2 1]
     [1 3]
     [3 1]
     [3 4]
     [4 3]
     [4 5]
     [5 4]]
    '''
    for i in range(0, edges.shape[0], 2):
        assert edges[i, 0] == edges[i+1, 1]

        G2 = G.to_undirected()
        G2.remove_edge(*edges[i])

        # 逐步测试删除某些边后会否导致整个图不连通。即寻找rotable键
        # to_rotate: [[], [], [], [], [], [0, 1, 2], [4, 5], [], [], []]
        # 这里是想把每个不可/可旋转键存储为一对列表。如果不可旋转，则全部存为[]；
        # 如果可旋转，则起始原子在哪个片段内，哪个片段放后面。但已知l的话其实另外一个片段也是已知的，所以只记录l即可
        if not nx.is_connected(G2):
            l = list(sorted(nx.connected_components(G2), key=len)[0]) # nx.connected_components [{0, 8, 6, 7}, {1, 2, 3, 4, 5, 9, 10, 11, 12, 13}]
            print('nx.connected_components', list(nx.connected_components(G2)))
            if len(l) > 1: # 只考虑片段原子数>1的情况
                if edges[i, 0] in l: # 被删除的键，原子是否在l中
                    to_rotate.append([])
                    to_rotate.append(l)
                else:
                    to_rotate.append(l)
                    to_rotate.append([])
                print(edges[i, 0], to_rotate)
                continue
        # 若不可旋转，则记为[],[]
        to_rotate.append([])
        to_rotate.append([])

    mask_edges = np.asarray([0 if len(l) == 0 else 1 for l in to_rotate], dtype=bool)
    print('mask_edges: ', mask_edges)
    mask_rotate = np.zeros((np.sum(mask_edges), len(G.nodes())), dtype=bool) # [n_torsional_angles, n_nodes]，对每个rotable键（行）对应的可旋转部分的原子（列）置1
    idx = 0
    print('to_rotate: ', to_rotate)
    print('len(G.edges()): ', len(G.edges()))
    for i in range(len(G.edges())):
        if mask_edges[i]:
            mask_rotate[idx][np.asarray(to_rotate[i], dtype=int)] = True
            idx += 1
    print('mask_rotate: ', mask_rotate)

    return mask_edges, mask_rotate # mask edges给需要考虑的旋转键置1，且只给不含起始原子的那部分置1. 其长度是len(edges).
# mask rotate针对每个可旋转键，为该键所允许的可旋转原子们置1. 所以其shape是n_rotable_edges*n_nodes
# to_rotate:  [[], [], [], [], [], [0, 1, 2], [4, 5], [], [], []]
# mask_edges:  [False False False False False  True  True False False False]
# mask_rotate:  [[ True  True  True False False False]，[False False False False  True  True]]


def modify_conformer_torsion_angles(pos, edge_index, mask_rotate, torsion_updates, as_numpy=False):
    pos = copy.deepcopy(pos)
    if type(pos) != np.ndarray: pos = pos.cpu().numpy()
    for idx_edge, e in enumerate(edge_index.cpu().numpy()):
        if torsion_updates[idx_edge] == 0:
            continue
        u, v = e[0], e[1]

        # check if need to reverse the edge, v should be connected to the part that gets rotated
        assert not mask_rotate[idx_edge, u] # 键的起始原子mask为False
        assert mask_rotate[idx_edge, v]

        # 根据旋转向量的定义，旋转向量的矢量方向，定义了转轴；旋转向量的模则定义了转角theta。这里是在保持转轴方向的前提下， 更新theta
        # 其实感觉改成下面这样更直观。原来的旋转向量是单位向量。然后在此单位向量上乘以一个更新的theta。新的旋转向量的模就是torsion_updates[idx_edge]
        rot_vec = pos[u] - pos[v]  # convention: positive rotation if pointing inwards
        rot_vec /= np.linalg.norm(rot_vec) # 归一化为单位矢量
        rot_vec *= torsion_updates[idx_edge]   # 新的模即为theta
        # rot_vec = rot_vec * torsion_updates[idx_edge] / np.linalg.norm(rot_vec) # idx_edge! # 这是原代码的写法
        rot_mat = R.from_rotvec(rot_vec).as_matrix()

        pos[mask_rotate[idx_edge]] = (pos[mask_rotate[idx_edge]] - pos[v]) @ rot_mat.T + pos[v] # v端的原子是允许旋转的。画个示意图会更好理解

    if not as_numpy: pos = torch.from_numpy(pos.astype(np.float32))
    return pos


def perturb_batch(data, torsion_updates, split=False, return_updates=False):
    if type(data) is Data:
        return modify_conformer_torsion_angles(data.pos,
                                               data.edge_index.T[data.edge_mask],
                                               data.mask_rotate, torsion_updates)
    pos_new = [] if split else copy.deepcopy(data.pos)
    edges_of_interest = data.edge_index.T[data.edge_mask]
    idx_node = 0
    idx_edges = 0
    torsion_update_list = []
    for i, mask_rotate in enumerate(data.mask_rotate):
        pos = data.pos[idx_node:idx_node + mask_rotate.shape[1]]
        edges = edges_of_interest[idx_edges:idx_edges + mask_rotate.shape[0]] - idx_node
        torsion_update = torsion_updates[idx_edges:idx_edges + mask_rotate.shape[0]]
        torsion_update_list.append(torsion_update)
        pos_new_ = modify_conformer_torsion_angles(pos, edges, mask_rotate, torsion_update)
        if split:
            pos_new.append(pos_new_)
        else:
            pos_new[idx_node:idx_node + mask_rotate.shape[1]] = pos_new_

        idx_node += mask_rotate.shape[1]
        idx_edges += mask_rotate.shape[0]
    if return_updates:
        return pos_new, torsion_update_list
    return pos_new