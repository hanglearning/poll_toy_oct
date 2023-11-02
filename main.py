import os
import torch
import argparse
import pickle
import numpy as np
import random
from pytorch_lightning import seed_everything
from model.cnn import CNN as CNN
# from model.cifar10.mlp import MLP as CIFAR_MLP
# from model.mnist.cnn import CNN as MNIST_CNN
# from model.mnist.mlp import MLP as MNIST_MLP
# from server import Server
from client import Client
# from util import create_model
import wandb
from DataLoaders import DataLoaders
from torchmetrics import MetricCollection, Accuracy, Precision, Recall
from torchmetrics import F1Score as F1

from datetime import datetime


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # FL settings
    parser.add_argument('--dataset', help="MNIST|CIFAR10",
                        type=str, default="MNIST")
    parser.add_argument('--dataset_mode', type=str,
                        default='non-iid', help='non-iid|iid')
    parser.add_argument('--n_clients', type=int, default=4)
    parser.add_argument('--comm_rounds', type=int, default=40)
    parser.add_argument('--prune_step', type=float, default=0.2)
    parser.add_argument('--prune_threshold', type=float, default=0.8)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--n_samples', type=int, default=0) # num of samples per label
    parser.add_argument('--n_labels', type=int, default=0) # num of labeles per client
    parser.add_argument('--optimizer', type=str, default="Adam", help="SGD|Adam")

    # non-iid data sharding settings
    parser.add_argument('--total_samples', type=int, default=0) # for data sharding methods III, IV, V
    parser.add_argument('--labels_same', type=int, default=0) # for data sharding methods I, II, III

    # wabdb settings
    parser.add_argument('--project_name', type=str, default="POLL_OCT2023")
    parser.add_argument('--run_note', type=str, default="")
    parser.add_argument('--wandb', type=int, default=0, help="if set to 1, enable wandb logging")
    
    # system setup
    parser.add_argument('--train_verbose', type=bool, default=False)
    parser.add_argument('--test_verbose', type=bool, default=False)
    parser.add_argument('--prune_verbose', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=40)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--log_dir', type=str, default="./logs")

    # Run Type
    parser.add_argument('--run_type', type=int, default=0,
                        help='0 - POLL \
                                1 - STANDALONE \
                                2 - FEDAVG_NO_PRUNE \
                                3 - CELL')
    # STANDALONE: Pure Centralized
    # FEDAVG_NO_PRUNE: Vanilla FL, Pure FedAvg without Pruning

    parser.add_argument('--STANDALONE', type=int, default=0)
    parser.add_argument('--VANILLA_FL', type=int, default=0)
    
    # for CELL
    parser.add_argument('--eita', type=float, default=0.5,
                        help="accuracy threshold")
    parser.add_argument('--alpha', type=float, default=0.5,
                        help="accuracy reduction factor")

    # for POLL
    parser.add_argument('--param_reinit', type=int, default=0)

    # for federated malicious
    parser.add_argument('--noise_variance', type=int, default=1, help="noise variance level of the injected Gaussian Noise")
    parser.add_argument('--n_malicious', type=int, default=0, help="number of malicious nodes in the network")

    args = parser.parse_args()

    def set_random_seeds():
        os.environ['PYTHONHASHSEED']=str(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        seed_everything(seed=args.seed, workers=True)

    set_random_seeds()

    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device {args.device}")
    init_model = CNN().to(args.device)

    exe_date_time = datetime.now().strftime("%m%d%Y_%H%M%S")

    try:
        # on Google Drive
        import google.colab
        log_dirpath = f"/content/drive/MyDrive/POLL_TOY/{exe_date_time}"
    except:
        # local
        log_dirpath = f"{args.log_dir}/{exe_date_time}"
    os.makedirs(log_dirpath)

     # init wandb
    run_types = ['POLL', 'STANDALONE', 'FEDAVG_NO_PRUNE', 'CELL']
    run_name = run_types[args.run_type]

    wandb.login()
    if args.wandb:
        wandb.init(project=args.project_name, entity="hangchen")
    else:
        wandb.init(project=args.project_name, entity="hangchen", mode="disabled")
    wandb.run.name = f"{run_name}_samples_{args.n_samples}_n_clients_{args.num_clients}_mali_{args.n_malicious}_optim_{args.optimizer}_seed_{args.seed}_{args.run_note}_{exe_date_time}"
    wandb.config.update(args)

    dataloaders = DataLoaders(
        n_clients=args.n_clients,
        dataset=args.dataset,
        n_labels=args.n_labels,
        n_samples=args.n_samples,
        ds_mode=args.dataset_mode,
        batch_size=args.batch_size,
        total_samples=args.total_samples,
        labels_same=args.labels_same,
        log_dirpath=log_dirpath
    )

    dataloaders.train_loaders, dataloaders.global_test_loader

    print()

    clients = []
    n_malicious = args.n_malicious
    for i in range(args.n_clients):
        malicious = True if args.n_clients - i <= n_malicious else False
        client = Client(i + 1, args, malicious, init_model, dataloaders.train_loaders[i], dataloaders.global_test_loader)
        clients.append(client)


    # server = Server(args, model, clients, global_test_loader)

    # for comm_round in range(1, args.comm_rounds + 1):
    #     server.update(comm_round)

