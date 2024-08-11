import json

# with open("/user_data/wujy/GitHubProject/ZS-Generation/data/splits/zero_rte/fewrel/pid2name.json","r") as f:
#     file_content = f.read()
#     json_object = json.loads(file_content)
# # print(type(json_object))
# all_keys = json_object.keys()
# all_keys = set(list(all_keys))
# print(len(all_keys))
all_key1 = []
all_key2 = []
with open("/user_data/wujy/GitHubProject/RelationPrompt/outputs/data/splits/zero_rte/fewrel/unseen_5_seed_0/train.jsonl","r") as f1 , open("/user_data/wujy/GitHubProject/RelationPrompt/outputs/data/splits/zero_rte/fewrel/unseen_5_seed_0/test.jsonl","r") as f2:
    for line in f1:
        json_object1 = json.loads(line)
        all_key1.append(json_object1["triplets"][0]["label_id"])
    for line in f2:
        json_object = json.loads(line)
        all_key2.append(json_object["triplets"][0]["label_id"])
all_key1 = set(all_key1)
all_key2 = set(all_key2)

com_keys = [key for key in all_key2 if key not in all_key1]
print(len(all_key1))
print(len(all_key2))
print(com_keys)


