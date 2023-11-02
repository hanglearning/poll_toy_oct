import random
from collections import Counter
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import numpy as np

class DataLoaders:
    def __init__(self, n_clients, dataset, ds_mode, batch_size, n_samples, n_labels, labels_same, total_samples, log_dirpath):
    
        if dataset == 'MNIST':
            MNIST_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])
            self.train_data = datasets.MNIST('./data', train=True, download=True, transform=MNIST_transform)
            self.test_data = datasets.MNIST('./data', train=False, download=True, transform=MNIST_transform)
        elif dataset == 'CIFAR10':
            CIFAR10_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            self.train_data = datasets.CIFAR10('./data', train=True, download=True, transform=CIFAR10_transform)
            self.test_data = datasets.CIFAR10('./data', train=False, download=True, transform=CIFAR10_transform)
        else:
            raise ValueError('Invalid dataset. Choose either "MNIST" or "CIFAR10"')
        
        labels, _ = np.unique(self.train_data.targets, return_counts=True)
        if n_labels > len(labels):
            raise ValueError(f"n_labels ({n_labels}) cannot be greater than the number of unique labeles ({len(labels)}) in the dataset")
        
        self.n_clients = n_clients
        self.ds_mode = ds_mode
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.n_labels = n_labels
        self.labels_same = labels_same
        self.total_samples = total_samples
        self.log_dirpath = log_dirpath

        self.global_test_loader = DataLoader(self.test_data, batch_size=len(self.test_data))
        
        self.train_loaders = self.shard_data()

    def shard_data(self):
        labels, label_to_samples_count = np.unique(self.train_data.targets, return_counts=True)
        self.label_to_samples_count = {l: samples_count for l, samples_count in zip(labels, label_to_samples_count)}
        self.label_to_sample_indices = [np.where(np.array(self.train_data.targets) == i)[0].tolist() for i in range(len(labels))]
        if self.ds_mode == 'iid':
            if self.n_samples == 0:
                return self.iid_sharding_entire()
            elif self.n_samples > 0 and self.n_labels > 0:
                return self.iid_sharding_even()
            else:
                raise ValueError('Invalid dataset mode. Choose either "iid" or "non-iid"')
        elif self.ds_mode == 'non-iid':
            if self.n_samples > 0 and self.n_labels > 0:
                return self.non_iid_sharding_case_II()
            elif self.n_samples == 0 and self.n_labels > 0 and self.labels_same == 1:
                return self.non_iid_sharding_case_III()
            elif self.n_samples == 0 and self.n_labels > 0 and self.labels_same == 0:
                return self.non_iid_sharding_case_IV()
            elif self.n_samples == 0 and self.n_labels == 0:
                return self.non_iid_sharding_case_V()
            else:
                raise ValueError('Invalid non-iid sharding case')
        else:
            raise ValueError('Invalid dataset mode. Choose either "iid" or "non-iid"')
        
        
    def print_and_save_label_counts(self, data_loaders, case_text, filename):
        print(case_text)
        with open(f"{self.log_dirpath}/{filename}", "a") as file:
            file.write(case_text)
        for i, data_loader in enumerate(data_loaders):
            label_counts = {}
            for data, labels in data_loader:
                for label in labels:
                    if label.item() not in label_counts:
                        label_counts[label.item()] = 0
                    label_counts[label.item()] += 1
            label_counts = {k: label_counts[k] for k in sorted(label_counts)}
            label_counts_text = f"Client {i + 1} assigned labels {label_counts}\n"
            with open(f"{self.log_dirpath}/{filename}", "a") as file:
                file.write(label_counts_text)
            print(label_counts_text)
    
    def get_single_client_noneven_samples_dist(self, label_count):
        return np.diff(np.insert(np.sort(np.random.choice(range(1, self.total_samples), label_count-1, replace=False)), [0, label_count-1], [0, self.total_samples])) 

    def get_label_distribution(self, case):
        """ For case I ~ V, get labels with samples count without overlapping samples
            Return -
            client_to_label_samples_counts: a list of dictionaries.
                The list index is client idx.
                The dictionary contains the key being label and value being the corresponding samples count for the particular client.
        """

        if case == "III" or case == "IV" or case == "V":
            print(f"If sharding takes a long time, consider decrease n_labels {self.n_labels},  n_samples {self.n_samples}, and/or total_samples {self.total_samples} and rerun the program.")

        if case == "I":
            label_count_constraint = self.n_samples * self.n_clients
            satisfied_labels = [l for l, c in self.label_to_samples_count.items() if c > label_count_constraint]
            try:
                labels = random.sample(satisfied_labels, self.n_labels)
            except:
                raise ValueError('Provided sharding constraints do not satisfy case I.')
            client_to_label_samples_counts = [{l: self.n_samples for l in labels}] * self.n_clients
        elif case == "II":
            client_to_label_samples_counts = []
            for _ in range(self.n_clients):
                satisfied_labels = [l for l, c in self.label_to_samples_count.items() if c > self.n_samples]
                try:
                    labels = random.sample(satisfied_labels, self.n_labels)
                except:
                    raise ValueError('Provided sharding constraints do not satisfy case II.')
                client_to_label_samples_counts.append({l: self.n_samples for l in labels})
        elif case == "III":
            break_while_loop = False
            while not break_while_loop:
                client_to_undecided_label_samples_counts = []
                for _ in range(self.n_clients):
                    one_client_label_samples_counts = self.get_single_client_noneven_samples_dist(self.n_labels)
                    client_to_undecided_label_samples_counts.append(one_client_label_samples_counts)
                need_label_samples_counts = np.sum(np.array(client_to_undecided_label_samples_counts), axis=0)
                # determine the satisfied labels
                candidate_labels = []
                for label_samples_count in need_label_samples_counts:
                    candidate_labels.append([l for l, c in self.label_to_samples_count.items() if c > label_samples_count])
                labels = []
                for candidate_label_iter in candidate_labels:
                    availabel_labels = set(candidate_label_iter) - set(labels)
                    if not len(availabel_labels):
                        # current label counts combination does not satisfy the sharding constraints, recalculate
                        break
                    labels.append(random.choice(list(availabel_labels)))
                break_while_loop = True
            client_to_label_samples_counts = []
            for samples_distribution in client_to_undecided_label_samples_counts:
                client_to_label_samples_counts.append({l:c for l, c in zip(labels, samples_distribution)})
        elif case == "IV" or case == "V":
            break_while_loop = False
            while not break_while_loop:
                break_for_loops = False
                client_to_label_samples_counts = []
                for _ in range(self.n_clients):
                    if break_for_loops:
                        break
                    if case == "IV":
                        need_n_labels = self.n_labels
                    elif case == "V":
                        need_n_labels = random.randint(1, len(self.label_to_samples_count))
                    one_client_label_samples_counts = self.get_single_client_noneven_samples_dist(need_n_labels)
                    # determine the satisfied labels for each samples count
                    candidate_labels = []
                    for label_samples_count in one_client_label_samples_counts:
                        candidate_labels.append([l for l, c in self.label_to_samples_count.items() if c > label_samples_count])
                    # pick a label for each samples count
                    labels = []
                    for i, candidate_label_iter in enumerate(candidate_labels):
                        availabel_labels = set(candidate_label_iter) - set(labels)
                        if not len(availabel_labels):
                            # current label counts combination does not satisfy the sharding constraints, recalculate
                            break_for_loops = True
                            break
                        picked_label = random.choice(list(availabel_labels))
                        labels.append(picked_label)
                        # deduce the label count
                        self.label_to_samples_count[picked_label] -= one_client_label_samples_counts[i]
                    client_to_label_samples_counts.append({l:c for l, c in zip(labels, one_client_label_samples_counts)})
                break_while_loop = True
        
        # sort by labels in each client, ease debugging
        client_to_label_samples_counts = [{l: d[l] for l in sorted(d)} for d in client_to_label_samples_counts]
        
        return client_to_label_samples_counts
            
    def distribute_samples(self, client_to_label_samples_counts):
        shards = []
        for i in range(self.n_clients):
            indices = []
            for label in client_to_label_samples_counts[i].keys():
                label_to_sample_indices = self.label_to_sample_indices[label - 1]
                label_count = client_to_label_samples_counts[i][label]
                if len(label_to_sample_indices) < label_count:
                    raise ValueError(f"Not enough samples in label {label} to assign {label_count} samples to each client")
                chosen_indices = np.random.choice(label_to_sample_indices, label_count, replace=False).tolist()
                indices.extend(chosen_indices)
                # remove chosen indices, prevent overlapping samples
                self.label_to_sample_indices[label - 1] = [index for index in self.label_to_sample_indices[label - 1] if index not in chosen_indices]
            shards.append(indices)
        data_loaders = [DataLoader(Subset(self.train_data, shard), batch_size=self.batch_size, shuffle=True) for shard in shards]
        return data_loaders

    def iid_sharding_entire(self):
        # Case O: shards the entire 50000 training samples to each client, with each label count random and close
        indices = np.random.permutation(len(self.train_data))
        shards = np.array_split(indices, self.n_clients)
        data_loaders = [DataLoader(Subset(self.train_data, shard), batch_size=self.batch_size, shuffle=True) for shard in shards]

        case_text = "Case O: IID\nEach client receives a random subset of the entire training data.\n"
        self.print_and_save_label_counts(data_loaders, case_text, "data_sharding_case_I.txt")

        return data_loaders

    def iid_sharding_even(self):
        # Case I: IID with n_samples per label

        client_to_label_samples_counts = self.get_label_distribution("I")
        data_loaders = self.distribute_samples(client_to_label_samples_counts)

        case_text = f"Case I: IID with n_samples {self.n_samples} per label for {self.n_labels} labeles\nEach client receives a random non-overlapping subset of the data with {self.n_samples} samples per label.\n"
        self.print_and_save_label_counts(data_loaders, case_text, "data_sharding_case_I.txt")

        return data_loaders
    

    def non_iid_sharding_case_II(self):
        # Case II: Non-IID with n_samples > 0, n_labels > 0, and labels_same = 0
        client_to_label_samples_counts = self.get_label_distribution("II")
        data_loaders = self.distribute_samples(client_to_label_samples_counts)

        case_text = f"Case II: Non-IID with n_samples = {self.n_samples}, n_labels = {self.n_labels}, and labels_same = 0\n\nEach client receives a random non-overlapping subset of the data with {self.n_samples} samples per label.\n"
        self.print_and_save_label_counts(data_loaders, case_text, "data_sharding_case_II.txt")

        return data_loaders

    def non_iid_sharding_case_III(self):
        # Case III: Non-IID with n_samples = 0, n_labels > 0, and labels_same = 1
        client_to_label_samples_counts = self.get_label_distribution("III")
        data_loaders = self.distribute_samples(client_to_label_samples_counts)

        case_text = "Case III: Non-IID with n_samples = 0, n_labels > 0, and labels_same = 1\nEach client receives a subset of the data with the same label.\n"
        self.print_and_save_label_counts(data_loaders, case_text, "data_sharding_case_III.txt")

        return data_loaders

    def non_iid_sharding_case_IV(self):
        client_to_label_samples_counts = self.get_label_distribution("IV")
        data_loaders = self.distribute_samples(client_to_label_samples_counts)

        case_text = "Case IV: Non-IID with n_samples = 0, n_labels > 0, and labels_same = 0\nEach client receives a subset of the data with different labels.\n"
        self.print_and_save_label_counts(data_loaders, case_text, "data_sharding_case_IV.txt")

        return data_loaders

    def non_iid_sharding_case_V(self):
        # Case V: Non-IID with n_samples = 0 and n_labels = 0
        client_to_label_samples_counts = self.get_label_distribution("V")
        data_loaders = self.distribute_samples(client_to_label_samples_counts)

        case_text = "Case V: Non-IID with n_samples = 0 and n_labels = 0\nEach client receives a random subset of the data.\n"
        self.print_and_save_label_counts(data_loaders, case_text, "data_sharding_case_V.txt")

        return data_loaders