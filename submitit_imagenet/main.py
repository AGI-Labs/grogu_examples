import os
import copy
import submitit
from train import main_worker
import argparse
import torch
import torch.multiprocessing as mp
import torchvision.models as models


class Worker:
    def __call__(self, origargs):
        """TODO: Docstring for __call__.

        :args: TODO
        :returns: TODO

        """
        args = copy.deepcopy(origargs)

        ##############################
        # You can use this is you plan on using multi-node training, but not needed for single node jobs
        # dist_url will point to the first node allocated to this job and can be used to initialize
        # the distributed training
        socket_name = os.popen(
            "ip r | grep default | awk '{print $5}'").read().strip('\n')
        print(
            "Setting GLOO and NCCL sockets IFNAME to: {}".format(socket_name))
        os.environ["GLOO_SOCKET_IFNAME"] = socket_name
        if args.slurm:
            job_env = submitit.JobEnvironment()
            args.rank = job_env.global_rank
            hostname_first_node = os.popen(
                "scontrol show hostnames $SLURM_JOB_NODELIST").read().split(
                    "\n")[0]
            args.dist_url = f'tcp://{job_env.hostnames[0]}:{args.port}'
        else:
            args.dist_url = f'tcp://{args.node}:{args.port}'
        print('Using url {}'.format(args.environment.dist_url))
        ##############################

        if args.dist_url == "env://" and args.world_size == -1:
            args.world_size = int(os.environ["WORLD_SIZE"])

        args.distributed = args.world_size > 1 or args.multiprocessing_distributed

        ngpus_per_node = torch.cuda.device_count()
        if args.multiprocessing_distributed:
            # Since we have ngpus_per_node processes per node, the total world_size
            # needs to be adjusted accordingly
            args.world_size = ngpus_per_node * args.world_size
            # Use torch.multiprocessing.spawn to launch distributed processes: the
            # main_worker process function
            mp.spawn(main_worker,
                     nprocs=ngpus_per_node,
                     args=(ngpus_per_node, args))
        else:
            # Simply call main_worker function
            main_worker(args.gpu, ngpus_per_node, args)

    def checkpoint(self, *args,
                   **kwargs) -> submitit.helpers.DelayedSubmission:
        return submitit.helpers.DelayedSubmission(
            self, *args, **kwargs)  # submits to requeuing


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--submitit-dir', help='location to save submitit logs')
parser.add_argument('--name', help='name of experiment')
parser.add_argument('--nodelist', help='restrict slurm to these nodes')
parser.add_argument('--exclude-nodes', help='do not queue to these nodes')
parser.add_argument('--slurm', action='store_true', help='use slurm')

parser.add_argument('-a',
                    '--arch',
                    metavar='ARCH',
                    default='resnet18',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('-j',
                    '--workers',
                    default=4,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs',
                    default=90,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b',
                    '--batch-size',
                    default=256,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                    'batch size of all GPUs on the current node when '
                    'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.1,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    metavar='M',
                    help='momentum')
parser.add_argument('--wd',
                    '--weight-decay',
                    default=1e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p',
                    '--print-freq',
                    default=10,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e',
                    '--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained',
                    dest='pretrained',
                    action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size',
                    default=-1,
                    type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank',
                    default=-1,
                    type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url',
                    default='tcp://224.66.41.62:23456',
                    type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend',
                    default='nccl',
                    type=str,
                    help='distributed backend')
parser.add_argument('--seed',
                    default=None,
                    type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed',
                    action='store_true',
                    help='Use multi-processing distributed training to launch '
                    'N processes per node, which has N GPUs. This is the '
                    'fastest way to use PyTorch for either single node or '
                    'multi node data parallel training')


def main():
    args = parser.parse_args()

    executor = submitit.AutoExecutor(
        folder=os.path.join(args.submitit_dir, '{}'.format(
            args.name)),  # slurm Logs will be written here
        slurm_max_num_timeout=100,
        cluster=None if args.slurm else
        "debug",  # debug mode will run on the current node and not slurm
    )
    additional_parameters = {}
    if args.environment.nodelist != "":
        additional_parameters = {"nodelist": args.nodelist}
    if args.environment.exclude_nodes != "":
        additional_parameters.update({"exclude": args.exclude_nodes})
    executor.update_parameters(
        timeout_min=100,  # Minutes
        slurm_partition="abhinav",  # partitionname
        cpus_per_task=64,
        gpus_per_node=8,
        nodes=1,
        tasks_per_node=1,
        mem_gb=256,
        slurm_additional_parameters=additional_parameters)
    executor.update_parameters(
        name=args.name)  # Experiment name for book-keeping
    job = executor.submit(Worker(), args)
    if not args.slurm:
        job.result()


if __name__ == '__main__':
    main()
