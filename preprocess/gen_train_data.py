import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import json
import os

model_path = '../bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)
model.resize_token_embeddings(len(tokenizer))

max_seq_len = 256
max_node_num = 10
max_layer_num = 5
max_child_num = 8

one_node_num = 0


def get_item_data(item_data, layer_index=1):
    text = item_data['text']
    children = item_data['children']
    children_num = 0

    node_layer_index = [layer_index]
    node_nums = [1]

    encode_result = tokenizer.encode_plus(
        text=text,
        add_special_tokens=True,
        max_length=max_seq_len,
        padding='max_length',
        return_attention_mask=True,
    )
    tokens_idx = [encode_result['input_ids']]
    token_type_ids = [encode_result['token_type_ids']]
    # node: [CLS] text child1 child2 ...
    node = ['[CLS]', ] + ['[PAD]'] * (max_child_num + 1)
    node_idx = [tokenizer.convert_tokens_to_ids(node)]
    waiting_mask = []

    if layer_index + 1 <= max_layer_num:  # 最大层数
        for i, child_key in enumerate(children.keys()):
            if i + 1 == max_child_num:  # 超出最大节点数
                break
            child_tokens_idx, child_token_type_ids, child_waiting_mask, child_node_idx, child_node_nums, child_node_layer_index \
                = get_item_data(children[child_key], layer_index + 1)
            tokens_idx += child_tokens_idx
            token_type_ids += child_token_type_ids
            waiting_mask += child_waiting_mask
            node_idx += child_node_idx
            for j, child_node_num in enumerate(child_node_nums):
                if j + 1 >= len(node_nums):
                    node_nums.append(child_node_num)
                else:
                    node_nums[j + 1] += child_node_num
            node_layer_index += child_node_layer_index
            children_num += 1
    waiting_mask = [[0] * 2 + [1] * children_num + [0] * max(max_child_num - children_num, 0)] + waiting_mask
    return tokens_idx, token_type_ids, waiting_mask, node_idx, node_nums, node_layer_index


def gen_train_data(item_data, key, label):
    global one_node_num
    texts_tokens_idx, token_type_ids, waiting_mask, node_idx, node_nums, node_layer_index = (
        get_item_data(item_data))

    texts_tokens_idx = sum(texts_tokens_idx, [])
    token_type_ids = sum(token_type_ids, [])
    waiting_mask = sum(waiting_mask, [])
    nodes_tokens_idx = sum(node_idx, [])

    if len(node_nums) <= 1:
        one_node_num += 1
    if len(node_nums) < max_layer_num:
        node_nums += [0] * (max_layer_num - len(node_nums))

    text_labels = [label] * len(texts_tokens_idx)
    node_labels = [[label] + [-100] * (len(node_idx[0]) - 1)] * len(node_idx)
    node_labels = sum(node_labels, [])

    positions = []
    position_len = max(node_nums)
    for layer_node_num in node_nums:
        curr_layer_pos = [i for i in range(layer_node_num)]
        curr_layer_pos += [0] * (position_len - layer_node_num)
        positions.append(curr_layer_pos)
    positions = sum(positions, [])

    instance = {
        "node_tokens_idx": nodes_tokens_idx,
        "text_tokens_idx": texts_tokens_idx,
        "inputs_type_idx": token_type_ids,
        "node_labels": node_labels,
        "text_labels": text_labels,
        "layer_index": node_layer_index,
        "waiting_mask": waiting_mask,
        "position": positions,
        "node_num": node_nums,
    }
    return instance


if __name__ == "__main__":
    dataset_name = "PHEME"
    output_path = os.path.join("../train_data")
    if os.path.exists(output_path) is False:
        os.mkdir(output_path)

    input_path = os.path.join('../data', 'input_data', dataset_name)
    label_data = pd.read_csv(os.path.join(input_path, "label.csv"))

    num = 0
    flag = False
    all_data = []
    for root, dirs, files in os.walk(input_path):
        for file in tqdm(files):
            if file.endswith(".json"):
                with open(os.path.join(root, file)) as f:
                    file_data = f.readlines()
                    json_data = json.loads("".join(file_data))
                    for key in json_data:
                        label = label_data[label_data["id"] == int(key)]["label"].iat[0]
                        label = 1 if label == "rumours" else 0
                        data_item = gen_train_data(json_data[key], key, label)
                        data_item["label"] = [label]
                        all_data.append(data_item)
                        num += 1
                    f.close()
                # if num >= 2:
                #     flag = True
                #     break
        if flag:
            break

    train_data = all_data[:int(len(all_data) * 0.6)]
    eval_data = all_data[len(train_data):len(train_data) + int(len(all_data) * 0.2)]
    test_data = all_data[len(train_data) + len(eval_data):]

    train_file = open(os.path.join(output_path, dataset_name, 'train.json'), 'w')
    json.dump(train_data, train_file)
    train_file.close()

    eval_file = open(os.path.join(output_path, dataset_name, 'eval.json'), 'w')
    json.dump(eval_data, eval_file)
    eval_file.close()

    test_file = open(os.path.join(output_path, dataset_name, 'test.json'), 'w')
    json.dump(test_data, test_file)
    test_file.close()

    print("one_node_num: ", one_node_num)
