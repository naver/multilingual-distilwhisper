import argparse
from datasets import load_dataset, concatenate_datasets
import pandas as pd
import json

parser=argparse.ArgumentParser()

parser.add_argument(
  "--lang",
  nargs="*",  # 0 or more values expected => creates a list
  type=str,
  default=[],
)
parser.add_argument("--output", type=str, default="./fleurs_ids_st_en.json")

args = parser.parse_args()

lang_list = args.lang
output = args.output

access_token = "" # Add your token from HF

dict_dataset_idx = {}

dataset_en = load_dataset("google/fleurs", "en_us", "train+validation+test", streaming=False, use_auth_token=access_token)
print("Connected to hugging face")
dataset_en = concatenate_datasets([dataset_en["train"], dataset_en["validation"], dataset_en["test"]])
dataset_en = dataset_en.add_column("idx", list(range(len(dataset_en))))
dataset_en = dataset_en.remove_columns("audio")
df_dataset_en = pd.DataFrame(dataset_en)

for lang in lang_list:
    print(f"Starting download of FLEURS for lang {lang}")
    dataset = load_dataset("google/fleurs", lang, "train+validation+test", streaming=False, use_auth_token=access_token)

    ids = dataset["test"]["id"]
    
    dict_dataset_idx[lang] = df_dataset_en.loc[df_dataset_en['id'].isin(ids)][["id", "transcription", "idx"]].drop_duplicates(subset=["id"], ignore_index=True)["idx"].to_list()
    

with open(args.output, "w") as fp:
    json.dump(dict_dataset_idx,fp)
    