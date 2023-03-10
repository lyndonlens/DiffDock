import os
import numpy as np
from copy import deepcopy
import yaml
from tqdm import tqdm
import argparse
import torch
from esm.pretrained import esm2_t33_650M_UR50D
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class ESM_pretrained():
    def __init__(self, model_id, weights_path):
        super(ESM_pretrained, self).__init__()
        # loading ESM model:
        os.environ['TORCH_HOME'] = weights_path  # 预训练模型权重
        print(f"1. Loading ESM-{model_id} model...")
        self.model_id = model_id
        self.model, self.alphabet = esm2_t33_650M_UR50D()
        self.model.eval()  # disables dropout for deterministic results

    def get_esm_feats(self, data, out_path, ignore_exists=False):
        # 注意序列长度不应该长于1021
        esm2_feats_file_list = []
        print(f"2. Calculating ESM-{self.model_id} feats...")
        batch_converter = self.alphabet.get_batch_converter()
        with torch.no_grad():
            for d in tqdm(data):  # 一个个输入，否则容易内存不足。主要时间都在前面的模型加载，这里不batch计算也影响不大
                out = {}
                pt_file = os.path.join(out_path, d[0] + '.pt')
                if os.path.exists(pt_file) and ignore_exists:
                    batch_labels, batch_strs, batch_tokens = batch_converter([d])
                    batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
                    results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
                    # 注意每个AA的表征要取1：tokens_len-1, 因为前后各添加了start token（0）和stop token（2）。
                    out['label'] = d[0]
                    out["representations"] = {33: results["representations"][33][0, 1:-1, :]}  # 记得去掉第一个维度，同时去掉开头和结尾
                    torch.save(out, pt_file)  # 可以直接给diffdock使用的
                else:
                    print('\nAlready exists ', d[0] + '.pt', ', skipping...')
                esm2_feats_file_list.append(pt_file)
            print(f"3. ESM{self.model_id} feats all saved.")
        return esm2_feats_file_list

def parse_fasta(filename):
    file = open(filename, 'r')
    lines = file.readlines()
    data = []
    for i in range(len(lines) // 2):
        seq_id = '_'.join(lines[2 * i][1:].strip('\n').split('_'))
        # convers.append(float(lines[2 * i][1:].strip('\n').split('_')[-1]))
        seq = lines[2 * i + 1].strip('\n')
        data.append((seq_id, seq))
    return data

if __name__ == '__main__':
    # 如果是给diffdock使用，注意特征的形式
    # esm2特征存储为pt文件，数据结构为dict_keys(['label', 'representations'])，其中label为序列名称，data['representations'][33]为每个氨基酸的特征
    # torch.Size([N_AA, 1280])

    parser = argparse.ArgumentParser(description="ESM2-feats")
    parser.add_argument('--fasta_file',
                        type=str,
                        default='/home/tianxh/projects/DiffDock/data/test.fasta',
                        help='Fasta file contains sequences')
    args = parser.parse_args()
    data = parse_fasta(args.fasta_file) # data的形式如下
    ## 输入数据形式如下
    # data = [
    #     ('p1', 'MSIVNESGSQPVVSRDETLSQIERTSFHISSGKDISLEEIARAARDHQPVTLHDEVVNRVTRSRSILESMVSDERVIYGVNTSMGGFVNYIVPIAKASELQ'),
    #     ('p2', 'MELFKYMETYDYEQVLFCQDKESGLKAIIAIHDTTLGPALGGTRMWMYNSEEEALEDAGLNLGGGKTVIIGDPRKDKNEAMFRAFGR')]

    # 固定参数文件
    file = open('../esm2_config.yaml', 'r', encoding='utf-8')
    params = yaml.load(file, Loader=yaml.FullLoader)


    # os.environ['TORCH_HOME'] = params['ESM2_WEIGHTS_PATH'] # 预训练模型权重
    # OUTPUT_DIR = params['ESM2_FEATS_PATH']


    model_name = '2' # model_name:['1b', '1v', '2']
    esm_model = ESM_pretrained(model_name, weights_path=params['ESM2_WEIGHTS_PATH'])
    esm_model.get_esm_feats(data, out_path=params['ESM2_FEATS_PATH']) # 获取的特征直接写入硬盘。文件名称是fasta文件每个序列注释用’_‘拼接到一起，如 2_wt_K68S_D261L_T43M_52.5.pt
    # 注意这里的文件名必须和diffdock对接时提供的pdb文件名一致
