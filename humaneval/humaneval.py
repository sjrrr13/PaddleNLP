import argparse
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from humaneval_utils.evaluation import estimate_pass_at_k
from humaneval_utils.execution import check_correctness


def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="",
        required=True
    )
    parser.add_argument(
        "--field",
        type=str,
        default="",
        required=True
    )
    args = parser.parse_args()
    return args


def evaluate_functional_correctness(
    dataset: dict,
    generations: list,
):
    k = [1, 10, 100]
    n_workers = 1
    timeout = 3.0

    # Check the generated samples against test suites.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        completion_id = Counter()
        n_samples = 0
        pass_or_not = []
        results = defaultdict(list)
        
        for idx, sample in tqdm(enumerate(dataset)):
            task_id = sample["task_id"]
            generation = generations[idx]
            arg = (dataset[idx], generation,
                    timeout, completion_id[task_id])
            # print(task_id, sample['canonical_solution'], '\nout', completion)
            future = executor.submit(check_correctness, *arg)
            futures.append(future)
            completion_id[task_id] += 1
            n_samples += 1

        print("Running test suites...")
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    # Calculate pass@k
    total, correct = [], []
    for result in results.values():
        # print(result)
        pass_or_not.extend([r[1]["passed"] == 1 for r in result])
        result.sort()
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    
    total = np.array(total)
    correct = np.array(correct)
    # print(correct)

    tmp_k = k
    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
                for k in tmp_k if (total >= k).all()}
    return pass_at_k


if "__main__" == __name__:
    args = setup_args()

    generations = []
    len_gen = 0
    with open(args.path, "r") as file:
        for line in file:
            data = json.loads(line)
            task_id = data["task_id"]
            # prompt = data["prompt"]
            # groundtruth = data["canonical_solution"]
            generation = data[f'{args.field}']
            if "SingleLineInfilling" not in task_id:
                generations.append(str(generation))
            len_gen += 1

    dataset = []
    with open("./HumanEval_Solution.jsonl", "r") as file:
        for idx, line in enumerate(file):
            data = json.loads(line)
            dataset.append(data)
            if idx == len_gen - 1:
                break

    pass_at_k = evaluate_functional_correctness(dataset, generations)

    print("HumanEval_Score: " + str(pass_at_k))
