"""Dataset class for loading and processing KG datasets."""

import os
import pickle as pkl

import numpy as np
import torch


class KGDataset(object):
    """Knowledge Graph dataset class."""

    def __init__(self, data_path, debug):
        """Creates KG dataset object for data loading.

        Args:
             data_path: Path to directory containing train/valid/test pickle files produced by process.py
             debug: boolean indicating whether to use debug mode or not
             if true, the dataset will only contain 1000 examples for debugging.
        """
        self.data_path = data_path
        self.debug = debug
        self.data = {}
        for split in ["train", "test", "valid"]:
            file_path = os.path.join(self.data_path, split + ".pickle")
            with open(file_path, "rb") as in_file:
                self.data[split] = pkl.load(in_file)
        filters_file = open(os.path.join(self.data_path, "to_skip.pickle"), "rb")
        self.to_skip = pkl.load(filters_file)
        filters_file.close()
        max_axis = np.max(self.data["train"], axis=0)
        self.n_entities = int(max(max_axis[0], max_axis[2]) + 1)
        self.n_predicates = int(max_axis[1] + 1) * 2
        
        # # --- START: ADDED MAPPING LOADING ---
        # # Assuming mappings are saved as pickle files (common format)
        # try:
        #     with open(os.path.join(self.data_path, "entity_to_id.pickle"), "rb") as f:
        #         self.entity_to_id = pkl.load(f)
        #     with open(os.path.join(self.data_path, "relation_to_id.pickle"), "rb") as f:
        #         self.relation_to_id = pkl.load(f)
        # except FileNotFoundError:
        #     # Fallback if mapping files don't exist
        #     print("Warning: Entity/Relation ID mapping files not found. Using integer IDs.")
        #     self.entity_to_id = {i: str(i) for i in range(self.n_entities)}
        #     self.relation_to_id = {i: str(i) for i in range(self.n_predicates // 2)}
            
        # # Create inverse mappings (ID to Name)
        # self.id_to_entity = {v: k for k, v in self.entity_to_id.items()}
        # self.id_to_relation = {v: k for k, v in self.relation_to_id.items()}
        # # --- END: ADDED MAPPING LOADING ---

        # --- START: ADDED MAPPING LOADING ---
        # NOTE: This assumes 'entity_to_id.pickle' and 'relation_to_id.pickle' exist 
        # in your data directory (data_path) and contain {name: id} mappings.
        try:
            with open(os.path.join(self.data_path, "entity_to_id.pickle"), "rb") as f:
                entity_name_to_id = pkl.load(f)
            with open(os.path.join(self.data_path, "relation_to_id.pickle"), "rb") as f:
                relation_name_to_id = pkl.load(f)
            
            # Create the essential ID-to-Name maps for easy lookup
            self.id_to_entity = {v: k for k, v in entity_name_to_id.items()}
            self.id_to_relation = {v: k for k, v in relation_name_to_id.items()}
            
        except FileNotFoundError:
            # Fallback if mapping files don't exist (e.g., if only raw IDs are available)
            print("Warning: Entity/Relation name-to-ID mapping files not found. Using integer IDs as names.")
            self.id_to_entity = {i: str(i) for i in range(self.n_entities)}
            self.id_to_relation = {i: str(i) for i in range(self.n_predicates // 2)}
            
        # --- END: ADDED MAPPING LOADING ---
        
    def get_examples(self, split, rel_idx=-1):
        """Get examples in a split.

        Args:
            split: String indicating the split to use (train/valid/test)
            rel_idx: integer for relation index to keep (-1 to keep all relation)

        Returns:
            examples: torch.LongTensor containing KG triples in a split
        """
        examples = self.data[split]
        if split == "train":
            copy = np.copy(examples)
            tmp = np.copy(copy[:, 0])
            copy[:, 0] = copy[:, 2]
            copy[:, 2] = tmp
            copy[:, 1] += self.n_predicates // 2
            examples = np.vstack((examples, copy))
        if rel_idx >= 0:
            examples = examples[examples[:, 1] == rel_idx]
        if self.debug:
            examples = examples[:1000]
        return torch.from_numpy(examples.astype("int64"))

    def get_filters(self, ):
        """Return filter dict to compute ranking metrics in the filtered setting."""
        return self.to_skip

    def get_shape(self):
        """Returns KG dataset shape."""
        return self.n_entities, self.n_predicates, self.n_entities
