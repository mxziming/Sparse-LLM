import json
import tqdm
from transformers import AutoTokenizer

from load_config import CONFIG

tokenizer = AutoTokenizer.from_pretrained(CONFIG["tokenizer_path"])
MAX_SEQ_LEN = 2048
MAX_DATASET_SIZE = 10000

# 从 WritingPrompts 中选取长度大于 1024 toekn 的序列作为训练数据集
if __name__ == "__main__":
    # writingprompt
    import glob
    import pandas as pd
    # find all .parquet files anywhere under `folder` whose filename contains "train"
    files = glob.glob(f"{CONFIG["raw_parquet_folder"]}/*train*.parquet")
    print(files)
    # read and concatenate them all into one DataFrame
    raw_data = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    with open(CONFIG["train_data_path"], 'w', encoding='utf-8') as fw:
        warmup_data = []
        size, dropped_too_short, dropped_too_long = 0, 0, 0
            
        for idx, line in tqdm.tqdm(raw_data.iterrows(), total=len(raw_data)):
            instruction_text = line['prompt']
            input_text = ''
            output_text = line['story']
            history = ''

            query = instruction_text + input_text
            answer = output_text + tokenizer.eos_token
            messages = []
            
            if history:
                for i in history:
                    messages.append({'role': 'user', 'content': i[0]})
                    messages.append({'role': 'assistant', 'content': i[1]})

            messages.append({'role': 'user', 'content': query})   
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            prompt_input_ids = tokenizer.encode(prompt)
            answer_input_ids = tokenizer.encode(answer)
            input_ids = prompt_input_ids + answer_input_ids

            if len(input_ids) < 1024:
                dropped_too_short += 1
                continue
            
            if len(input_ids) >= MAX_SEQ_LEN:
                dropped_too_long += 1
                continue
            
            record = {
                "instruction": instruction_text,
                "input": input_text,
                "output": output_text,
                "history": history,
            }
            warmup_data.append(json.dumps(record, ensure_ascii=False) + "\n")
            size += 1

            if len(warmup_data) == 100:
                fw.writelines(warmup_data)
                warmup_data = []
            
            if MAX_DATASET_SIZE is not None and size == MAX_DATASET_SIZE:
                break
        
        if warmup_data:
            fw.writelines(warmup_data)
        
        print(f"丢弃(总长度<1024): {dropped_too_short}")
        print(f"丢弃(总长度超过{MAX_SEQ_LEN}): {dropped_too_long}")

    # # deepctrl-sft-data, longalpaca-12k
    # with open(CONFIG["raw_data_path"], 'r', encoding='utf-8') as f:
    #     raw_data = json.load(f)

    # with open(CONFIG["train_data_path"], 'a', encoding='utf-8') as fw:
    #     warmup_data = []
    #     size, dropped_too_short, dropped_too_long = 0, 0, 0
            
    #     for line in tqdm.tqdm(raw_data):
    #         deepctrl-sft-data, longalpaca-12k
    #         instruction_text = line.get('instruction', '')
    #         input_text = line.get('input', '')
    #         output_text = line.get('output', '')
    #         history = line.get('history', '')

    #         longalpaca-12k
    #         if input_text is None:
    #             input_text = ''

    #         query = instruction_text + input_text
    #         answer = output_text + tokenizer.eos_token
    #         messages = []
            
    #         if history:
    #             for i in history:
    #                 messages.append({'role': 'user', 'content': i[0]})
    #                 messages.append({'role': 'assistant', 'content': i[1]})

    #         messages.append({'role': 'user', 'content': query})   
    #         prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    #         prompt_input_ids = tokenizer.encode(prompt)
    #         answer_input_ids = tokenizer.encode(answer)
    #         input_ids = prompt_input_ids + answer_input_ids

    #         if len(input_ids) < 1024:
    #             dropped_too_short += 1
    #             continue
            
    #         if len(input_ids) >= MAX_SEQ_LEN:
    #             dropped_too_long += 1
    #             continue
            
    #         record = {
    #             "instruction": instruction_text,
    #             "input": input_text,
    #             "output": output_text,
    #             "history": history,
    #         }
    #         warmup_data.append(json.dumps(record, ensure_ascii=False) + "\n")

            # if len(warmup_data) == 100:
            #     fw.writelines(warmup_data)
            #     warmup_data = []
              
            # if MAX_DATASET_SIZE is not None and size == MAX_DATASET_SIZE:
            #     break
        
    #     if warmup_data:
    #         fw.writelines(warmup_data)
        
    #     print(f"丢弃(总长度<1024): {dropped_too_short}")
    #     print(f"丢弃(总长度超过{MAX_SEQ_LEN}): {dropped_too_long}")