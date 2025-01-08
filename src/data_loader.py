import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from .pertdata import PertData
from .config import LOGGING_DIR
from .gene_annotations import annotate_genes

class DataLoader:
    def __init__(self, data_name: str = "norman", save_dir: str = "data", shuffle: bool = True, test_split: float = 0.2, random_seed: int = 42, stratify: bool = True) -> None:
        self.data_name = data_name
        self.save_dir = save_dir
        self.shuffle = shuffle
        self.test_split = test_split
        self.random_seed = random_seed
        self.stratify = stratify
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        try:
            # load data
            print(f"Loading {self.data_name} data...")
            self.pert_data = PertData.from_repo(
                name=self.data_name, save_dir=self.save_dir)
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

        # preprocess data
        print(f"Preprocessing {self.data_name} data...")
        self.X, self.y = self._preprocess_data()

        # encode labels
        self.y = self.label_encoder.fit_transform(self.y)
        self.num_classes = len(set(self.y))
        # scale data
        # self.X = self.scaler.fit_transform(self.X)

        # split data
        print(f"Splitting {self.data_name} data into train and test sets...")
        self.X_train, self.X_test, self.y_train, self.y_test = self._split_data()
        
        # sample weights from training data
        self.sample_weights = self._compute_sample_weights()
        
    
    def _compute_sample_weights(self) -> np.ndarray:
        class_counts = np.bincount(self.y_train)
        total_samples = len(self.y_train)
        num_classes = len(set(self.y_train))

        class_weights = {c: total_samples/(num_classes*count) for c, count in enumerate(class_counts)}
        self.sample_weights = [class_weights[label] for label in self.y_train]
        return self.sample_weights

    def _preprocess_data(self) -> Tuple[np.ndarray, np.ndarray]:
        # adapted from AMLG repository https://github.com/voges/amlg/blob/main/src/exercises/perturbation_data_analysis/perturbation_data_analysis.ipynb

        # remove double-gene perturbations
        filter_mask = ~self.pert_data.adata.obs["condition_fixed"].str.contains(
            r"\+")
        indexes_to_keep = filter_mask[filter_mask].index
        adata_single = self.pert_data.adata[indexes_to_keep].copy()

        # select high-variance genes
        d = 128  # number of top genes to select
        gene_variances = adata_single.X.toarray().var(axis=0)
        sorted_indexes = gene_variances.argsort()[::-1]
        top_gene_indexes = sorted_indexes[:d]
        top_genes = adata_single.var["gene_name"].iloc[top_gene_indexes]
        top_variances = gene_variances[top_gene_indexes]
        # logging to a file
        with open(f"{LOGGING_DIR}/top_genes.txt", "w") as f:
            for gene, variance in zip(top_genes, top_variances):
                f.write(f"{gene:15}: {variance:.2f}\n")

        adata_single_top_genes = adata_single[:, top_gene_indexes].copy()
        single_gene_perturbations = adata_single_top_genes.obs["condition_fixed"].to_list()
        gene_programme = annotate_genes(single_gene_perturbations)
        adata_single_top_genes.obs["gene_programme"] = gene_programme
        adata = adata_single_top_genes[adata_single_top_genes.obs["gene_programme"] != "Unknown"].copy()
        X = adata.X.toarray()
        y = adata.obs["gene_programme"].to_numpy()
        
        return X, y

    def _split_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=self.test_split, random_state=self.random_seed, stratify=self.y)
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=self.test_split, random_state=self.random_seed)
        if self.shuffle:
            X_train, y_train = shuffle(
                X_train, y_train, random_state=self.random_seed)
            X_test, y_test = shuffle(
                X_test, y_test, random_state=self.random_seed)
        return X_train, X_test, y_train, y_test

    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        print(f"Number of training samples: {self.X_train.shape[0]}")
        return self.X_train, self.y_train

    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        print(f"Number of test samples: {self.X_test.shape[0]}")
        return self.X_test, self.y_test

    def get_num_classes(self) -> int:
        return self.num_classes