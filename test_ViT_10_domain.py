# coding=utf-8

import os
import sys
import time
import argparse
import csv
import numpy as np
import torch.nn as nn
import torch.optim as optim
from alg.opt import *
from utils.util import set_random_seed, train_valid_target_eval_names_s, alg_loss_dict, img_param_init, print_environ
from utils.util import init_norm_dict, Tee
from alg import modelopera_simple
from datautil.getdataloader import get_eval_dataloader_simple
from network.base_network import Regressor, Discriminator
from network.bhvit_network_re import ViTBottleneck, ViTDecoder, Vit2Norm


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_args():
    parser = argparse.ArgumentParser(description='DG')
    parser.add_argument('--algorithm', type=str, default="DANN")
    parser.add_argument('--alpha', type=float,
                        default=1, help='DANN dis alpha')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='batch_size')
    parser.add_argument('--checkpoint_freq', type=int,
                        default=200, help='Checkpoint every N epoch')
    parser.add_argument('--data_file', type=str, default='',
                        help='root_dir')
    parser.add_argument('--dataset', type=str, default='MAGNET')
    parser.add_argument('--data_dir', type=str, default='MagNet 2023 Database', help='data dir')
    parser.add_argument('--eval_dir', type=str, default='valid_data', help='eval dir')
    parser.add_argument('--gpu_id', type=str, nargs='?',
                        default='0', help="device id to run")
    parser.add_argument('--max_epoch', type=int,
                        default=2000, help="max iterations")
    parser.add_argument('--N_WORKERS', type=int, default=4)
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--schuse', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--task', type=str, default="img_dg",
                        choices=["img_dg"], help='now only support image tasks')
    parser.add_argument('--test_envs', type=int, nargs='+',
                        default=[], help='target domains')

    parser.add_argument('--output', type=str,
                        default="train_output", help='result output path')

    args = parser.parse_args()
    args.steps_per_epoch = 100
    args.data_dir = args.data_file + args.data_dir
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    args = img_param_init(args)
    print_environ()
    return args


if __name__ == '__main__':
    args = get_args()
    set_random_seed(args.seed)

    loss_list = alg_loss_dict(args)
    norm_dict = init_norm_dict("var_file/10_domain/variables.npz")
    eval_loaders = get_eval_dataloader_simple(args, norm_dict)

    eval_name_dict = train_valid_target_eval_names_s(args)

    device = torch.device("cuda")

    embed_dim = 256
    num_var = 3
    dim_feedforward_projecter = 128

    ViT_encoder = ViTBottleneck(input_size=1,
                                dim_val=32,
                                seq_size=1024 * 32,
                                patch_size=(8, 32),
                                embed_dim=256,
                                in_channel=1,
                                norm_layer=None,
                                distilled=None,
                                dropout_pos_enc=0.0,
                                max_seq_len=129,
                                n_heads=8,
                                dim_feedforward_encoder=512,
                                dropout_encoder=0.0,
                                n_encoder_layers=2).to(device)

    ViT_decoder = ViTDecoder(input_size=1,
                             dim_val=32,
                             seq_size=(1024 + 8) * 32,
                             patch_size=(8, 32),
                             embed_dim=256,
                             in_channel=1,
                             norm_layer=None,
                             distilled=None,
                             dropout_pos_enc=0.0,
                             max_seq_len=257,
                             out_seq_len=129,
                             n_heads=8,
                             dim_feedforward_decoder=512,
                             dropout_decoder=0.0,
                             n_decoder_layers=2).to(device)

    projector = nn.Sequential(
        nn.Linear(embed_dim + num_var, dim_feedforward_projecter),
        nn.Tanh(),
        nn.Linear(dim_feedforward_projecter, dim_feedforward_projecter),
        nn.Tanh(),
        nn.Linear(dim_feedforward_projecter, embed_dim)).to(device)

    domain_discriminator = Discriminator(input_dim=256, hidden_dim=256, num_domains=10).to(device)
    pv_regressor = Regressor(input_dim=256, hidden_dim=256, num_domains=1).to(device)

    net = Vit2Norm(vit_attention_module=ViT_encoder, projector_vit_head=projector, projector_regressor=pv_regressor,
                   projector_discriminator=domain_discriminator, vit_decoder=ViT_decoder).to(device)

    pth_PATH = "pth_file/ViT_DANN_model_for_valid_7601.pth"
    checkpoint = torch.load(pth_PATH)

    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    sch = None

    net.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    net.eval()

    acc_type_list = ['valid']
    domains = args.domains
    item_template = []
    for item in acc_type_list:
        domain_template = []
        for i in eval_name_dict[item]:
            print(f"now loader {i}")
            start_time = time.time()
            domain_template.append(modelopera_simple.predict_transform(net, eval_loaders[i], norm_dict))
            print(f"cost {time.time() - start_time} second")
        item_template.append(domain_template)

    item_dict_template = dict()
    for item in item_template:
        for domain_index in range(len(item)):
            cat_domain = torch.cat([data.cuda().float() for data in item[domain_index]])
            data_array = cat_domain.cpu().numpy()
            domain_name = domains[domain_index]
            item_dict_template[domain_name] = data_array

    for key, value in item_dict_template.items():
        filename = f"pred_file/10domain/pred_{key}.csv"

        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)

            for row in value:
                csvwriter.writerow(row)
        print(f"save {filename} completed.")
