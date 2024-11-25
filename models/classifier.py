import torch
from torch import nn
import torch.nn.functional as F

from models.newsformer import Newsformer

node_len = 10
layer_num = 5


class Classifier(nn.Module):
    def __init__(self, model_name_or_path, cache_dir, node_config, n_classes):
        super(Classifier, self).__init__()
        self._keys_to_ignore_on_save = ['node_encoder']
        self.n_classes = n_classes
        self.node_encoder = Newsformer.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            node_config=node_config,
            node_len=node_len,
            layer_num=layer_num,
            cache_dir=cache_dir,
        )
        for param in self.node_encoder.parameters():
            param.requires_grad = True
        self.fc1 = nn.Linear(self.node_encoder.hidden_size, self.node_encoder.hidden_size)
        self.fc2 = nn.Linear(self.node_encoder.hidden_size, n_classes)
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')

    def forward(
            self,
            layer_index,
            position,
            waiting_mask,
            node_num,  # (batch_size,layer_num,1)
            token_input_ids=None,
            node_input_ids=None,  # (batch_size,node_num,node_len)
            inputs_type_idx=None,
            token_type_ids=None,
            node_labels=None,  # (batch_size,node_num,node_len)
            token_labels=None,
            label_ids=None,
    ):
        x = self.node_encoder(layer_index,
                              position,
                              waiting_mask,
                              node_num,  # (batch_size,layer_num,1)
                              token_input_ids,
                              node_input_ids,  # (batch_size,node_num,node_len)
                              inputs_type_idx,
                              token_type_ids,
                              node_labels,  # (batch_size,node_num,node_len)
                              token_labels)
        outputs = self.fc1(x['output'][0])
        predictions = self.fc2(outputs)
        device = predictions.device
        label_ids = label_ids.view(-1).long()
        label_ids = F.one_hot(label_ids, num_classes=self.n_classes).float()
        label_ids = label_ids.to(device)
        return {'logits': predictions, 'loss': self.loss(predictions, label_ids)}
