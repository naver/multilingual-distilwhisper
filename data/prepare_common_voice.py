import argparse
from datasets import load_dataset
import pandas as pd
import json

parser=argparse.ArgumentParser()

parser.add_argument(
  "--lang",
  nargs="*",  # 0 or more values expected => creates a list
  type=str,
  default=[],
)
parser.add_argument(
  "--max_samples", type=int, help="", default=10000
)
parser.add_argument(
  "--max_samples_validation", type=int, help="", default=500
)
parser.add_argument("--output", type=str, default="./common_voice_data_idx.json")

args = parser.parse_args()

lang_list = args.lang
output = args.output

access_token = "" # Add your token from HF

dict_dataset_idx = {}

for lang in lang_list:
    print(f"Starting download of Common Voice for lang {lang}")
    dataset = load_dataset("mozilla-foundation/common_voice_13_0", lang, "train+validation+test", streaming=False, use_auth_token=access_token)
    print("Connected to hugging face")
    dict_dataset_idx[lang] = {}
    # Train
    dataset["train"] = dataset["train"].add_column("idx", list(range(len(dataset["train"]))))
    dataset["train"] = dataset["train"].remove_columns("audio")
    df_pandas = pd.DataFrame(dataset["train"])
    if len(dataset["train"]) > args.max_samples:
       df_pandas = df_pandas.sort_values(['up_votes', 'down_votes'], ascending = [False, True]).head(args.max_samples)
    final_dataset_idx = df_pandas["idx"].to_list()
    dict_dataset_idx[lang]["train"] = final_dataset_idx
    # Validation
    dataset["validation"] = dataset["validation"].add_column("idx", list(range(len(dataset["validation"]))))
    dataset["validation"] = dataset["validation"].remove_columns("audio")
    df_pandas = pd.DataFrame(dataset["validation"])
    if len(dataset["validation"]) > args.max_samples_validation:
       df_pandas = df_pandas.sort_values(['up_votes', 'down_votes'], ascending = [False, True]).head(args.max_samples_validation)
    final_dataset_idx = df_pandas["idx"].to_list()
    dict_dataset_idx[lang]["validation"] = final_dataset_idx
    

with open(args.output, "w") as fp:
    json.dump(dict_dataset_idx,fp)
    