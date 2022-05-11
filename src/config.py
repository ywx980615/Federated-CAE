import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help="rounds of training: T")
    parser.add_argument('--num_users', type=int,
                        default=10, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.5,
                        help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=2,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=16,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.0001,
                        help="learning rate: η")

    # other arguments
    parser.add_argument('--dataset_path', type=str,
                        default='/home/yao/FL_CAE/dataset/Local_dataset/', help="path of dataset")
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='num_workers (default: 2)')

    parser.add_argument('--verbose', action='store_true', help='verbose print')
    args = parser.parse_known_args()[0]
    return args
