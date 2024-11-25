import os.path
from dataclasses import dataclass
import datasets
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import BatchEncoding, DataCollatorWithPadding


class NewsDataSet(Dataset):
    def __init__(self, args):
        train_file = os.path.abspath(args.train_file)
        if os.path.isdir(train_file):
            filenames = os.listdir(train_file)
            train_files = [os.path.join(train_file, fn) for fn in filenames]
        else:
            train_files = train_file
        chunk_size_10mb = 10 << 20
        print("start loading datasets, train_files: ", train_files)
        self.dataset = datasets.load_dataset(
            'json',
            data_files=train_files,
            # ignore_verifications=False,
            cache_dir=args.dataset_cache_dir,
            features=datasets.Features({
                "text_tokens_idx": [datasets.Value("int32")],
                "node_tokens_idx": [datasets.Value("int32")],
                "inputs_type_idx": [datasets.Value("int32")],
                "text_labels": [datasets.Value("int32")],
                "node_labels": [datasets.Value("int32")],
                "layer_index": [datasets.Value("int32")],
                "node_num": [datasets.Value("int32")],
                "waiting_mask": [datasets.Value("int32")],
                "position": [datasets.Value("int32")],
                "label": [datasets.Value("int32")],
            }),
            chunksize=chunk_size_10mb
        )['train']
        self.data_set_len = len(self.dataset)
        print("loading dataset ok! len of dataset:", self.data_set_len)

    def __len__(self):
        return self.data_set_len

    def __getitem__(self, item):
        data = self.dataset[item]

        item_data = {
            "token_input_ids": torch.LongTensor(data['text_tokens_idx']),
            "node_input_ids": torch.LongTensor(data['node_tokens_idx']),
            "inputs_type_idx": torch.LongTensor(data['inputs_type_idx']),
            "token_labels": torch.LongTensor(data['text_labels']),
            "node_labels": torch.LongTensor(data['node_labels']),
            "layer_index": torch.LongTensor(data['layer_index']),
            "node_num": list(data['node_num']),
            "waiting_mask": torch.LongTensor(data['waiting_mask']),
            "position": list(data['position']),
            "label": torch.Tensor(data['label']),
        }
        return BatchEncoding(item_data)


@dataclass
class NewsDataCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """

    def __call__(self, features):
        max_seq_len = 256
        max_node_len = 10
        batch_size = len(features)
        mlm_labels = []
        layer_index = []
        layer_num = len(features[0]['node_num'])
        token_input_ids = []
        node_input_ids = []
        inputs_type_idx = []
        token_labels = []
        node_labels = []
        node_num = []
        waiting_mask = []
        position = []
        labels = []

        for i in range(batch_size):
            one_position = features[i]['position']
            position_len = max(features[i]['node_num'])
            one_position = [torch.LongTensor(one_position[j:j + position_len]) for j in
                            range(0, len(one_position), position_len)]
            assert len(one_position) == layer_num
            position.extend(one_position)
            token_input_ids.append(features[i]['token_input_ids'])
            node_input_ids.append(features[i]['node_input_ids'])
            inputs_type_idx.append(features[i]['inputs_type_idx'])
            token_labels.append(features[i]['token_labels'])
            layer_index.append(features[i]['layer_index'])
            node_num.append(features[i]['node_num'])
            waiting_mask.append(features[i]['waiting_mask'])
            node_labels.append(features[i]['node_labels'])
            labels.append(features[i]['label'])
        features.clear()
        features = {
            "token_input_ids": pad_sequence(token_input_ids, batch_first=True).view(batch_size, -1, max_seq_len),
            "node_input_ids": pad_sequence(node_input_ids, batch_first=True).view(batch_size, -1, max_node_len),
            "inputs_type_idx": pad_sequence(inputs_type_idx, batch_first=True).view(batch_size, -1, max_seq_len),
            "token_labels": pad_sequence(token_labels, batch_first=True).view(batch_size, -1, max_seq_len),
            "node_labels": pad_sequence(node_labels, batch_first=True).view(batch_size, -1, max_node_len),
            "layer_index": pad_sequence(layer_index, batch_first=True),
            "node_num": torch.LongTensor(node_num),
            "waiting_mask": pad_sequence(waiting_mask, batch_first=True).view(batch_size, -1, max_node_len),
            "position": pad_sequence(position, batch_first=True).view(batch_size, layer_num, -1),
            "label_ids": pad_sequence(labels, batch_first=True).view(batch_size, -1),
        }
        return features
