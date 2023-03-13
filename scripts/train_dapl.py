import os
import os.path as osp
import argparse
from collections import defaultdict
import hashlib


def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", "-m", default="DAPL", help="Method")
    parser.add_argument(
        "--dataset",
        "-d",
        default="officehome",
        help="Dataset",
        choices=['office31', 'officehome', 'visda', 'domainnet', 'cs'])
    parser.add_argument("--gpu", "-g", default=0, type=int, help="Gpu ID")
    parser.add_argument("--backbone",
                        "-b",
                        default="RN50",
                        help="Backbone",
                        choices=[
                            'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64',
                            'ViT-B/32', 'ViT-B/16', 'ViT-L/14',
                            'ViT-L/14@336px'
                        ])
    parser.add_argument("--n_trials",
                        "-n",
                        default=3,
                        type=int,
                        help="Repeat times")
    parser.add_argument("--n_start", default=0, type=int)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--CFG', type=str, default="ep25-32-csc")
    parser.add_argument('--T', type=float, default=1.0)
    parser.add_argument('--TAU', type=float, default=0.5)
    parser.add_argument('--U', type=float, default=1.0)

    args = parser.parse_args()

    ###############################################################################

    data_root = "~/data/bbda"
    method_name = args.method.lower()

    if args.dataset == 'office31':
        domains = ["amazon", "dslr", "webcam"]
        data_file = 'office31'
    elif args.dataset == 'officehome':
        domains = ['art', 'clipart', 'product', 'real_world']
        data_file = 'officehome'
    elif args.dataset == 'cs':
        domains = ["AID", "Merced", "NWPU"]
        data_file = 'cross_scene'
    elif args.dataset == 'visda':
        domains = ["synthetic", "real"]
        data_file = 'visda'
    elif args.dataset == 'domainnet':
        domains = ['dpainting', 'dreal', 'dsketch']
        data_file = 'domainnet'
    elif args.dataset == 'minidomainnet':
        domains = ['clipart', 'painting', 'real', 'sketch']
        data_file = 'mini_domainnet'
    else:
        raise ValueError('Unknown Dataset: {}'.format(args.dataset))

    ###############################################################################
    exp_info = args.exp_name
    if exp_info:
        exp_info = '_' + exp_info

    base_dir = osp.join(
        'output/UDA', args.method, args.dataset,
        f'{args.backbone}_{args.CFG}_{args.T}_{args.TAU}_{args.U}{exp_info}')

    for i in range(args.n_start, args.n_trials):
        for source in domains:
            for target in domains:
                if source != target:
                    if args.dataset == 'visda' and source == 'real':
                        print('skip real!')
                        continue
                    output_dir = osp.join(base_dir, source + '_to_' + target,
                                          str(i + 1))
                    seed = args.seed
                    if args.seed < 0:
                        seed = seed_hash(args.method, args.backbone,
                                         args.dataset, source, target, i)
                    else:
                        seed += i

                    os.system(
                        f'CUDA_VISIBLE_DEVICES={args.gpu} '
                        f'python train.py '
                        f'--root {data_root} '
                        f'--trainer {args.method} '
                        f'--backbone {args.backbone} '
                        f'--source-domains {source} '
                        f'--target-domains {target} '
                        f'--dataset-config-file configs/datasets/{data_file}.yaml '
                        f'--config-file configs/trainers/{method_name}/{args.CFG}.yaml '
                        f'--output-dir {output_dir} '
                        f'--seed {seed} '
                        f'TRAINER.DAPL.T {args.T} '
                        f'TRAINER.DAPL.TAU {args.TAU} '
                        f'TRAINER.DAPL.U {args.U}')
