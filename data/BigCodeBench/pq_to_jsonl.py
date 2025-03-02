import pyarrow.parquet as pq
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="bigcodehard_v0.1.4.parquet")  
    file = parser.parse_args().file
    pds = pq.read_pandas(file, columns=None).to_pandas()
    json_str = pds.to_json(path_or_buf=None, orient='records', lines=True, date_format='iso', date_unit='us', compression='gzip')
    json_str = json_str.replace("}\n{", "}, {")
    print(json_str)
    task_list = json.loads(f"[{json_str}]")
    with open(file.split('.parquet')[0] + '.jsonl', "w") as f:
        for task in task_list:
            f.write(json.dumps(task) + '\n')
