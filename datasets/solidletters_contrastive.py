import dgl
import pathlib
import random
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from datasets.base import BaseDataset, BaseContrastiveDataset
# from datasets.base_1 import BaseDataset, BaseContrastiveDataset
# from datasets.base_2 import BaseDataset, BaseContrastiveDataset
# from datasets.base_augment2 import BaseDataset, BaseContrastiveDataset
from datasets.solidletters import _get_filenames, _char_to_label

from collections import Counter

def append_counts_to_file(numbers, output_file):
    """
    Count occurrences of numbers (0–25) in the list and append results to a text file.
    """
    # Count numbers
    counts = Counter(numbers)

    # Append to file
    with open(output_file, "a") as f:
        f.write("\n=== Number Occurrence Summary ===\n")
        for i in range(26):  # ensure all 0–25 are included
            f.write(f"{i}: {counts.get(i, 0)}\n")

    print(f"✅ Counts appended to {output_file}")


class SolidLettersContrastive(BaseContrastiveDataset):
    def __init__(self, 
                 root_dir, 
                 split="train",              #split="train", "val", "test", "selective"
                 center_and_scale=True, 
                 shape_type="upper",         #upper=Uppercase, lower=Lowercase, both=Allcase
                 prob_full_graph=0.10, 
                 size_percentage=1.0):
        
        assert shape_type in ("upper", "lower", "both")
        super().__init__(split, prob_full_graph)
        path = pathlib.Path(root_dir)

        if split in ("train", "val"):
            #Make a list filepaths which are in both rootfolder and '.txt' file
            file_paths = _get_filenames(path, filelist="train.txt") #[PosixPath('/home/ar/CAD/D_SolidLetters/graph_with_eattr/n_Bodoni MT Poster Compressed_upper.bin'),...]
            print(f"Found {len(file_paths)} bin files")

            #Filter the filepaths according to shape_type i.e. ("upper", "lower", "both")
            if shape_type != "both":
                file_paths = [fn for fn in file_paths if shape_type in fn.stem]   #file_paths[0]=/home/ar/CAD/D_SolidLetters/graph_with_eattr/n_Bodoni MT Poster Compressed_upper.bin
                                                                                  #file_paths[0].stem=n_Bodoni MT Poster Compressed_upper
            print(f"Left with {len(file_paths)} bin files after filtering by shape type:", shape_type)

            #Numerical label assigned to each character based on alphabetical order
            labels_to_stratify = [_char_to_label(fn.stem[0]) for fn in file_paths] #file_paths[*].stem[0]=a/b/c..-->labels_to_stratify=[list]=[0, 4, 2,...(num_CAD_models)] where 0(for a), 4(for e), 2(for c),...
            # append_counts_to_file(labels_to_stratify, "/home/ar/CAD/D_KVM/class_counts.txt")
            #Split the data into 80:20 ratio for training and validation
            train_files, val_files = train_test_split(file_paths, test_size=0.2, random_state=42, stratify=labels_to_stratify) # train_files, val_files are the list of PosixPath
            #random_state=42 make the split results same everytime while passing SolidLettersContrastive
            if split == "train":
                file_paths = train_files
            elif split == "val":
                file_paths = val_files

            labels = [torch.tensor([_char_to_label(fn.stem[0])]).long() for fn in file_paths] #labels=[tensor([0]), tensor([22]),...] where labels_to_stratify is the list of integers
            #Dataset passed through SolidLettersContrastive class for train_data and train_labels and val_data and val_labels separately
        
        
        elif split == "test":
            file_paths = _get_filenames(path, filelist="test.txt")
            print(f"Found {len(file_paths)} bin files")
            # The filenames must be according to shape_type
            if shape_type != "both":
                file_paths = [fn for fn in file_paths if shape_type in fn.stem]
            print(f"Left with {len(file_paths)} bin files after filtering by shape type:", shape_type)
            labels = [torch.tensor([_char_to_label(fn.stem[0])]).long() for fn in file_paths]
        
        elif split == "selective":
            file_paths = _get_filenames(path, filelist="test_selective.txt")
            print(f"Found {len(file_paths)} bin files")
            # The filenames must be according to shape_type
            if shape_type != "both":
                file_paths = [fn for fn in file_paths if shape_type in fn.stem]
            print(f"Left with {len(file_paths)} bin files after filtering by shape type:", shape_type)
            labels = [torch.tensor([_char_to_label(fn.stem[0])]).long() for fn in file_paths]


        self.labels = labels
        if size_percentage < 1.0:
            k = int(size_percentage * len(file_paths))
            index_list = set(random.sample(list(range(len(file_paths))), k))
            file_paths = [x for i, x in enumerate(file_paths) if i in index_list]
            self.labels = [x for i, x in enumerate(self.labels) if i in index_list]

        print(f"Loading {split} data...")
        self.load_graphs(file_paths, center_and_scale)

    @staticmethod
    def num_classes():
        # Only used during evaluation
        return 26
