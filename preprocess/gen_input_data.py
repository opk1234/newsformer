import os
import json
from tqdm import tqdm
import csv


def get_structure_text_dict(json_data: dict, path: str):
    map_data = dict()
    for item in json_data:
        text_file = open(os.path.join(path, item + ".json"), "r")
        map_data[item] = {
            'text': json.load(text_file)['text'],
            'children': get_structure_text_dict(json_data[item], path),
        }
    return map_data


def gen_pheme_input_data():
    post_types = ["non-rumours", "rumours"]
    dataset_type = "PHEME"
    root_path = "../data\\" + dataset_type + "\\all-rnr-annotated-threads\\"

    out_path = "../input_data"

    if os.path.exists(os.path.join(out_path, dataset_type)) is False:
        os.mkdir(os.path.join(out_path, dataset_type))

    label_file = open(os.path.join(out_path, dataset_type, "label.csv"), "w", newline='')
    writer = csv.writer(label_file)
    writer.writerow(["id", "label"])

    for event_dir in tqdm(os.listdir(root_path)):
        if os.path.exists(os.path.join(out_path, dataset_type, event_dir)) is False:
            os.mkdir(os.path.join(out_path, dataset_type, event_dir))
        for post_type in post_types:
            for post in os.listdir(os.path.join(root_path, event_dir, post_type)):
                data = {}
                f = open(os.path.join(out_path, dataset_type, event_dir, post + ".json"), "w")
                with open(os.path.join(root_path, event_dir, post_type, post, "structure.json"), "r",
                          encoding="utf-8") as structure_f:
                    temp = json.load(structure_f)
                    post_file = open(
                        os.path.join(root_path, event_dir, post_type, post, "source-tweets", post + ".json"),
                        "r")
                    data[post] = {
                        'text': json.load(post_file)['text'],
                        "children": get_structure_text_dict(temp[post],
                                                            os.path.join(root_path, event_dir, post_type, post,
                                                                         "reactions")),
                    }
                json.dump(data, f)
                f.close()
                writer.writerow([post, post_type])


def gen_twitter15_data():
    post_types = ["unverified", "non-rumor", "true", "false"]
    dataset_type = "twitter15"
    root_path = "../data\\" + dataset_type + "\\"

    out_path = "../input_data"

    if os.path.exists(os.path.join(out_path, dataset_type)) is False:
        os.mkdir(os.path.join(out_path, dataset_type))

    label_file = open(os.path.join(out_path, dataset_type, "label.csv"), "w", newline='')
    writer = csv.writer(label_file)
    writer.writerow(["id", "label"])

    pass


def gen_weibo_data():
    post_types = ["rumor", "otherwise"]
    dataset_type = "Weibo"
    root_path = "../data/" + dataset_type

    out_path = "../input_data"

    if os.path.exists(os.path.join(out_path, dataset_type)) is False:
        os.mkdir(os.path.join(out_path, dataset_type))

    label_file = open(os.path.join(out_path, dataset_type, "label.csv", ), "w", newline='', encoding='utf-8')
    writer = csv.writer(label_file)
    writer.writerow(["id", "label"])
    with open(os.path.join(root_path, "Weibo.txt"), "r", encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            line_arr = line.split("\t")
            id = line_arr[0][4:]
            label = line_arr[1][:-1]
            with open(os.path.join(root_path, "Weibo", id + ".json"), "r", encoding='utf-8') as post_f:
                data = {}
                post_data = json.load(post_f)
                children = {}
                for idx in range(1, len(post_data)):
                    child = post_data[idx]
                    if "转发微博" in child["text"]:
                        continue
                    children[child["id"]] = {
                        'text': child['text'],
                        'children': {}
                    }
                data[id] = {
                    'text': post_data[0]['text'],
                    'children': children
                }
                save_f = open(os.path.join(out_path, dataset_type, id + ".json"), "w", encoding='utf-8')
                json.dump(data, save_f, ensure_ascii=False)
                save_f.close()
            writer.writerow([id, label])
        f.close()


if __name__ == "__main__":
    gen_weibo_data()
