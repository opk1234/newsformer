import random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertConfig, BertForMaskedLM, BertModel


class Newsformer(BertForMaskedLM):
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(self, text_config, node_config, layer_num, node_len=10, num_type=3):
        super().__init__(text_config)
        self.type_embedding = torch.nn.Embedding(num_type, text_config.hidden_size)
        self._keys_to_ignore_on_save = ['node_bert']
        self.node_config = node_config
        self.node_bert = BertModel(node_config)

        self.layer_num = layer_num
        self.hidden_size = text_config.hidden_size

        self.linear = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.order_linear = torch.nn.Linear(self.hidden_size, node_len)
        self.apply(self._init_weights)

    def get_embedding(self, output, waiting_mask, node_input_ids):
        batch_size, child_num, node_num = waiting_mask.size()
        all_embeddings = self.bert.embeddings.word_embeddings(node_input_ids)
        mask_embeddings = self.bert.embeddings.word_embeddings(node_input_ids)
        neg_embeddings = self.bert.embeddings.word_embeddings(node_input_ids)
        output = output[1:]
        if len(output) > 0:
            res = torch.cat(output, dim=0)
            all_embeddings[waiting_mask.type(torch.bool)] = res
            neg_embeddings[waiting_mask.type(torch.bool)] = res
            mask_embeddings[waiting_mask.type(torch.bool)] = res

        return all_embeddings, neg_embeddings, mask_embeddings

    def get_task_embeddings(self, embeddings, negetive_embeddings, mask_embeddings, waiting_mask):
        device = embeddings.device
        cop_embeddings = embeddings.clone()

        batch_size, child_num, node_len, hidden_size = embeddings.size()
        tag_labels = torch.zeros(batch_size, child_num, node_len).to(device)

        snmther = []
        order = [i for i in range(node_len)]
        all_orders = []
        for i in range(batch_size):
            for j in range(child_num):
                tag = 0
                node_num = len(torch.nonzero(waiting_mask[i, j], as_tuple=False))
                if node_num == 0 or node_num == 1:
                    continue
                temp = list(range(2, node_num + 2))
                random.shuffle(temp) # 随机mask一个子节点,因为只使用temp[0]
                cop = order
                cop[2:2 + node_num] = temp
                cop_embeddings[i][j] = cop_embeddings[i][j][cop]
                order_lable = np.argsort(cop)
                all_orders.append(order_lable)
                if waiting_mask[i, j, temp[0]]:
                    temp_embedding = self.sample_one_embedding(waiting_mask, negetive_embeddings, i, j)
                    if temp_embedding is not None:
                        negetive_embeddings[i][j][temp[0]] = temp_embedding
                    mask_embeddings[i][j][temp[0]] = self.bert.embeddings.word_embeddings(torch.tensor(103).to(device))
                    tag_labels[i, j, temp[0]] = 1
                if waiting_mask[i, j, temp[1]]:
                    snmther.append(embeddings[i, j, temp[1]])

        children_num = torch.sum(waiting_mask, dim=-1)  # (batch_size,child_num)
        candidates_mask = children_num.gt(1)

        # node has only one child or non-child
        # if len(torch.nonzero(candidates_mask, as_tuple=False)):
        embeddings = embeddings[candidates_mask]
        mask_embeddings = mask_embeddings[candidates_mask]
        negetive_embeddings = negetive_embeddings[candidates_mask]
        cop_embeddings = cop_embeddings[candidates_mask]
        tag_labels = tag_labels[candidates_mask]
        candidates_len = len(embeddings)
        train_embeddings = torch.cat((embeddings, mask_embeddings), dim=0)
        if len(snmther) > 0:
            snmther = torch.stack(snmther)
        return train_embeddings, negetive_embeddings, cop_embeddings, snmther, tag_labels.bool(), np.array(all_orders)

    # [child_num,node_len,hidden_size]

    def sample_one_embedding(self, waiting_mask, embeddings, i, j):
        batch_size, _, _ = waiting_mask.size()
        temp = list(range(batch_size))
        random.shuffle(temp)
        for x in temp:
            if x == i:
                continue
            else:
                sample_batch = waiting_mask[x].clone().detach()
                candidates = torch.nonzero(sample_batch, as_tuple=False)
                random.shuffle(candidates)
                if len(candidates) == 0:
                    continue
                sample_index = candidates[0]
                return embeddings[x, sample_index[0], sample_index[1]]
        return None

    def masked_item_prediction(self, sequence_output, target_item):
        sequence_output = self.linear(sequence_output)
        sequence_output = torch.nn.functional.normalize(sequence_output, p=2, dim=1)
        sequence_output = sequence_output.view([-1, self.hidden_size])
        target_item = target_item.view([-1, self.hidden_size])
        score = torch.mul(sequence_output, target_item)
        return torch.sum(score, -1)

    def forward(
            self,
            layer_index,
            position,
            waiting_mask,
            node_num,  # (batch_size,layer_num,1)
            # seq_num,  # (batch_size,layer_num,1)
            # attention_mask=None,
            token_input_ids=None,
            node_input_ids=None,  # (batch_size,node_num,node_len)
            inputs_type_idx=None,
            token_type_ids=None,
            node_labels=None,  # (batch_size,node_num,node_len)
            token_labels=None,
            label_ids=None,
    ):
        node_input_ids[node_labels.ne(-100)] = node_labels[node_labels.ne(-100)]

        word_logits = []

        mlm_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')

        # mnp_loss_fct = torch.nn.CosineEmbeddingLoss(margin=0.0, reduction='mean')
        pcm_loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        snm_loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        # cop_loss_fct = torch.nn.KLDivLoss(reduction='batchmean')

        output = []
        device = token_input_ids.device
        batch_size, _, seq_len = token_input_ids.size()
        _, _, node_seq_len = node_input_ids.size()
        for layer in range(self.layer_num):
            cur_layer = self.layer_num - layer
            layer_mask = layer_index.eq(cur_layer)  # (b,n_num)
            # token_layer_mask = token_layer_index.eq(cur_layer)  # (b,seq_num)

            if not len(torch.nonzero(layer_mask, as_tuple=False)):
                continue

            layer_position = position[:, cur_layer - 1, :].view(batch_size, -1)  # (b,l_num,l_sum_num)
            if len(output) != 0:
                last_layer_output = output[-1]

            '''text layer'''
            layer_token_input_ids = token_input_ids[layer_mask]  # (b,lay_seq_num,seq_len)
            layer_inputs_type_idx = inputs_type_idx[layer_mask]  # (b,lay_n_num,n_len)
            layer_token_embeds = self.bert.embeddings.word_embeddings(
                layer_token_input_ids)  # (b,lay_seq_num,seq_len,h_dim)
            layer_token_embeds += self.type_embedding(layer_inputs_type_idx)  # (b,lay_seq_num,seq_len,h_dim)
            layer_text_output = self.bert(
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=layer_token_embeds
            )[0]
            layer_token_labels = token_labels[layer_mask]
            if len(word_logits):
                word_logits = torch.cat(
                    (word_logits, self.cls(layer_text_output).view(-1, self.config.vocab_size)),
                    dim=0)
                new_token_labels = torch.cat((new_token_labels, layer_token_labels), dim=0)
            else:
                word_logits = self.cls(layer_text_output).view(-1, self.config.vocab_size)
                new_token_labels = layer_token_labels
            text_output = layer_text_output[:, 0, :].view(-1, self.config.hidden_size)

            '''node layer'''
            layer_waiting_mask = waiting_mask[layer_mask].type(torch.bool)  # (b,lay_n_num,n_len)
            layer_node_input_ids = node_input_ids[layer_mask]  # (b,lay_n_num,n_len)
            layer_node_embeds = self.bert.embeddings.word_embeddings(layer_node_input_ids)  # (b,lay_n_num,n_len,h_dim)
            layer_node_embeds[:, 1, :] = text_output
            if len(output) != 0:
                layer_node_embeds[layer_waiting_mask] = last_layer_output
            layer_node_output = self.node_bert(
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=layer_node_embeds
            )[0]
            node_output = layer_node_output[:, 0, :].view(-1, self.config.hidden_size)

            '''one layer output'''
            layer_node_num = node_num[:, cur_layer - 1].view(-1)
            layer_output = []
            x_label = 0
            for i in range(batch_size):
                x = layer_node_num[i]
                if x == 0:
                    continue
                one_layer_position = layer_position[i, :].view(-1)
                one_layer_output = node_output[x_label:x_label + x, :]
                x_label += x
                one_layer_output = one_layer_output[one_layer_position][:one_layer_output.size(0)]
                if len(layer_output):
                    layer_output = torch.cat((layer_output, one_layer_output), dim=0)
                else:
                    layer_output = one_layer_output
            output.append(layer_output)
        # end for
        output.reverse()

        """ 
        text loss
        """
        mlm_loss = mlm_loss_fct(word_logits, new_token_labels.view(-1))

        """
        node loss
        """
        all_embeddings, neg_embeddings, mask_embeddings = self.get_embedding(output, waiting_mask, node_input_ids)
        train_embeddings, negetive_embeddings, cop_embeddings, snmther, task1_labels, order_labels = self.get_task_embeddings(
            all_embeddings, neg_embeddings, mask_embeddings, waiting_mask)
        if train_embeddings.size(0) == 0:
            return {
                'loss': mlm_loss,
                'output': output
            }

        # order_labels = torch.LongTensor(order_labels).to(device)
        cls_len = int(len(negetive_embeddings))
        pcm_positive = train_embeddings[:cls_len][task1_labels]
        pcm_negetive = negetive_embeddings[task1_labels]

        # cop_outputs = self.node_bert(
        #     input_ids=None,
        #     attention_mask=None,
        #     token_type_ids=None,
        #     position_ids=None,
        #     head_mask=None,
        #     inputs_embeds=cop_embeddings,
        # )
        # cop_output = cop_outputs[0][:, 0, :]
        # cop_logits = F.log_softmax(self.order_linear(cop_output))
        # order_labels = (torch.max(order_labels, dim=0).values - order_labels).true_divide(
        #     torch.sum(order_labels, dim=0))
        # cop_loss = cop_loss_fct(cop_logits, order_labels)

        all_node_outputs = self.node_bert(
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=train_embeddings,
        )
        node_output = all_node_outputs[0]
        # positive = node_output[:cls_len]
        # positive = positive[task1_labels]
        mask = node_output[cls_len:]
        pcm_parent = mask[:, 0, :]
        snm_pos_score = self.masked_item_prediction(snmther, pcm_positive).view(1, -1)
        snm_neg_score = self.masked_item_prediction(snmther, pcm_negetive).view(1, -1)
        snm_scores = torch.cat((snm_pos_score, snm_neg_score), dim=0).permute(1, 0)
        pcm_pos_score = self.masked_item_prediction(pcm_parent, pcm_positive).view(1, -1)
        pcm_neg_score = self.masked_item_prediction(pcm_parent, pcm_negetive).view(1, -1)
        pcm_scores = torch.cat((pcm_pos_score, pcm_neg_score), dim=0).permute(1, 0)

        snm_loss = snm_loss_fct(snm_scores, torch.ones(snm_pos_score.size(-1)).to(device).long())
        pcm_loss = pcm_loss_fct(pcm_scores, torch.ones(pcm_pos_score.size(-1)).to(device).long())
        # mask = mask[task1_labels]
        # mnp_loss = mnp_loss_fct(positive, mask, torch.ones(len(positive)).to(device))
        return {
            'loss':  mlm_loss + pcm_loss + snm_loss ,# cop_loss,
            'output': output
        }

    '''
        Task1:兄弟节点之间
        CLS text child1 child2 child3 child4
        替换为
        CLS text child1 child2 child_wrong child4
        用CLS预测是否有替换
        Task2:父子节点之间
        Task3:文档level
        CLS为文档表示
        CLS，本文档中任意一个句子的embedding
        CLS，同一batch中其他文档中任意一个句子的embedding
    '''
