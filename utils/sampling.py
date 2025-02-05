import numpy as np
import torch
from torch_geometric.loader import DataLoader

from utils.diffusion_utils import modify_conformer, set_time
from utils.torsion import modify_conformer_torsion_angles
from scipy.spatial.transform import Rotation as R


def randomize_position(data_list, no_torsion, no_random, tr_sigma_max):
    # in place modification of the list
    if not no_torsion: # 根据可扭转角的数量，随机采样n个扭转角。这样会否带来ligand本身的chash呢？
        # randomize torsion angles
        for complex_graph in data_list:
            torsion_updates = np.random.uniform(low=-np.pi, high=np.pi, size=complex_graph['ligand'].edge_mask.sum())
            complex_graph['ligand'].pos = \
                modify_conformer_torsion_angles(complex_graph['ligand'].pos,
                                                complex_graph['ligand', 'ligand'].edge_index.T[
                                                    complex_graph['ligand'].edge_mask],
                                                complex_graph['ligand'].mask_rotate[0], torsion_updates)

    for complex_graph in data_list:
        # randomize position
        molecule_center = torch.mean(complex_graph['ligand'].pos, dim=0, keepdim=True)
        random_rotation = torch.from_numpy(R.random().as_matrix()).float() # 随机旋转矩阵
        complex_graph['ligand'].pos = (complex_graph['ligand'].pos - molecule_center) @ random_rotation.T
        # base_rmsd = np.sqrt(np.sum((complex_graph['ligand'].pos.cpu().numpy() - orig_complex_graph['ligand'].pos.numpy()) ** 2, axis=1).mean())

        if not no_random:  # note for now the torsion angles are still randomised
            tr_update = torch.normal(mean=0, std=tr_sigma_max, size=(1, 3))
            complex_graph['ligand'].pos += tr_update


def sampling(data_list, model, inference_steps, tr_schedule, rot_schedule, tor_schedule, device, t_to_sigma, model_args,
             no_random=False, ode=False, visualization_list=None, confidence_model=None, confidence_data_list=None,
             confidence_model_args=None, batch_size=32, no_final_step_noise=False):
    N = len(data_list) # 生成的lig构象数量

    for t_idx in range(inference_steps):
        # 具体的时间点
        t_tr, t_rot, t_tor = tr_schedule[t_idx], rot_schedule[t_idx], tor_schedule[t_idx] # np.linspace(1, 0, inference_steps + 1)[:-1]
        # 时间步长，用于drift
        dt_tr = tr_schedule[t_idx] - tr_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tr_schedule[t_idx]
        dt_rot = rot_schedule[t_idx] - rot_schedule[t_idx + 1] if t_idx < inference_steps - 1 else rot_schedule[t_idx]
        dt_tor = tor_schedule[t_idx] - tor_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tor_schedule[t_idx]

        loader = DataLoader(data_list, batch_size=batch_size)
        new_data_list = []

        for complex_graph_batch in loader:
            b = complex_graph_batch.num_graphs
            complex_graph_batch = complex_graph_batch.to(device)

            # 在t上的sigma变化
            tr_sigma, rot_sigma, tor_sigma = t_to_sigma(t_tr, t_rot, t_tor) # sigma_min**(1-t) * sigma_max**t
            set_time(complex_graph_batch, t_tr, t_rot, t_tor, b, model_args.all_atoms, device)
            
            with torch.no_grad():
                tr_score, rot_score, tor_score = model(complex_graph_batch) # compute 3 scores on product space

            tr_g = tr_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tr_sigma_max / model_args.tr_sigma_min))) # sigma*sqrt(2*log(sigma_max/sigma_min))
            rot_g = 2 * rot_sigma * torch.sqrt(torch.tensor(np.log(model_args.rot_sigma_max / model_args.rot_sigma_min))) # ? where is the coefficient 2 ?
            # 感觉这里这个2*放错位置了。平移和torsion的都在根号下

            # rotation采样的结果是基于IGSO预存数据采样+插值而来的旋转轴+旋转角度，即aixs-angle采样。然后需要将axis-angle转化为rotation-matrix
            if ode:
                tr_perturb = (0.5 * tr_g ** 2 * tr_score.cpu() * dt_tr).cpu() # dt_tr is the time step. 这是SongYang论文Eq13方程右边的第二项。在diffdock论文里，f(x, t)这个drift项始终=0
                rot_perturb = (0.5 * rot_g ** 2 * rot_score.cpu() * dt_rot).cpu()
                # rot_perturb = (0.5 * rot_score.cpu() * dt_rot * rot_g ** 2).cpu() # 原代码
            else:
                # 平移矢量的随机项
                tr_z = torch.zeros((b, 3)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                    else torch.normal(mean=0, std=1, size=(b, 3))
                # 这是SongYang论文Eq6方程右边的第二项
                tr_perturb = (tr_g ** 2 * dt_tr * tr_score.cpu() + tr_g * np.sqrt(dt_tr) * tr_z).cpu() # g**2*score*dt + g*sqrt(dt)*z
                # SongYang论文Eq6方程右边的第二项。第二项多了个delta_t开根号；按说第一项前面还有个负号？


                rot_z = torch.zeros((b, 3)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                    else torch.normal(mean=0, std=1, size=(b, 3))
                rot_perturb = (rot_g ** 2 * dt_rot * rot_score.cpu()  + rot_g * np.sqrt(dt_rot) * rot_z).cpu()

            if not model_args.no_torsion:
                tor_g = tor_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tor_sigma_max / model_args.tor_sigma_min)))
                if ode:
                    tor_perturb = (0.5 * tor_g ** 2 * dt_tor * tor_score.cpu()).numpy()
                else:
                    tor_z = torch.zeros(tor_score.shape) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                        else torch.normal(mean=0, std=1, size=tor_score.shape)
                    tor_perturb = (tor_g ** 2 * dt_tor * tor_score.cpu() + tor_g * np.sqrt(dt_tor) * tor_z).numpy()
                torsions_per_molecule = tor_perturb.shape[0] // b #
            else:
                tor_perturb = None

            # Apply noise，对每个体系有一个translation，一个rotation，m个torsions
            new_data_list.extend([modify_conformer(complex_graph, tr_perturb[i:i + 1], rot_perturb[i:i + 1].squeeze(0),
                                          tor_perturb[i * torsions_per_molecule:(i + 1) * torsions_per_molecule] if not model_args.no_torsion else None)
                         for i, complex_graph in enumerate(complex_graph_batch.to('cpu').to_data_list())])

        data_list = new_data_list

        if visualization_list is not None:
            for idx, visualization in enumerate(visualization_list):
                visualization.add((data_list[idx]['ligand'].pos + data_list[idx].original_center).detach().cpu(),
                                  part=1, order=t_idx + 2)

    with torch.no_grad():
        if confidence_model is not None:
            loader = DataLoader(data_list, batch_size=batch_size)
            confidence_loader = iter(DataLoader(confidence_data_list, batch_size=batch_size))
            confidence = []
            for complex_graph_batch in loader:
                complex_graph_batch = complex_graph_batch.to(device)
                if confidence_data_list is not None:
                    confidence_complex_graph_batch = next(confidence_loader).to(device)
                    confidence_complex_graph_batch['ligand'].pos = complex_graph_batch['ligand'].pos
                    set_time(confidence_complex_graph_batch, 0, 0, 0, N, confidence_model_args.all_atoms, device)
                    confidence.append(confidence_model(confidence_complex_graph_batch))
                else:
                    confidence.append(confidence_model(complex_graph_batch))
            confidence = torch.cat(confidence, dim=0)
        else:
            confidence = None

    return data_list, confidence
