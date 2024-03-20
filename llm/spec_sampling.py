from __future__ import annotations

import json
import os
import sys
from abc import abstractmethod
from dataclasses import dataclass, field
import time

import numpy as np
import paddle
import paddle.distributed.fleet.base.topology as tp
from paddle.distributed import fleet
from utils import (
    dybatch_preprocess,
    get_infer_model_path,
    get_prefix_tuning_params,
    load_real_time_tokens,
)

from paddlenlp.peft import LoRAConfig, LoRAModel, PrefixConfig, PrefixModelForCausalLM
from paddlenlp.taskflow.utils import static_mode_guard
from paddlenlp.trainer import PdArgumentParser
from paddlenlp.transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    PretrainedModel,
    PretrainedTokenizer,
)
from paddlenlp.utils.import_utils import import_module, is_paddlenlp_ops_available

paddle.device.set_device("gpu:1")

# MODEL = ["facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-2.7b", "facebook/opt-6.7b"]
MODEL = ["meta-llama/Llama-2-7b", "meta-llama/Llama-2-13b"]

# TARGET_MODEL = ["facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-2.7b", "facebook/opt-6.7b"]
# DRAFT_MODEL = ["facebook/opt-350m", "facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-2.7b"]
# TARGET_DRAFT = [
#     ("facebook/opt-350m", "facebook/opt-125m"), 
#     ("facebook/opt-1.3b", "facebook/opt-125m"), ("facebook/opt-1.3b", "facebook/opt-350m"),
#     ("facebook/opt-2.7b", "facebook/opt-125m"), ("facebook/opt-2.7b", "facebook/opt-350m"), ("facebook/opt-2.7b", "facebook/opt-1.3b"), 
#     ("facebook/opt-6.7b", "facebook/opt-125m"), ("facebook/opt-6.7b", "facebook/opt-350m"), ("facebook/opt-6.7b", "facebook/opt-1.3b"), ("facebook/opt-6.7b", "facebook/opt-2.7b")]
TARGET_DRAFT = [("meta-llama/Llama-2-7b", "meta-llama/Llama-2-7b")]

# SOURCE_TEXT = "Welcome to use PaddlePaddle and PaddleNLP! PaddleNLP is"
SOURCE_TEXT = "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n"  

MAX_LENGTH = 128
L = 4   # draft token length
k = 1  # top-k sampling
OUTPUT = "tmp.json"

# @dataclass
# class PredictorArgument:
#     model_name_or_path: str = field(default=None, metadata={"help": "The directory of model."})
#     model_prefix: str = field(default="model", metadata={"help": "the prefix name of static model"})
#     src_length: int = field(default=1024, metadata={"help": "The max length of source text."})
#     max_length: int = field(default=64, metadata={"help": "the max length for decoding."})
#     top_k: int = field(default=1, metadata={"help": "top_k parameter for generation"})
#     top_p: float = field(default=1.0, metadata={"help": "top_p parameter for generation"})
#     temperature: float = field(default=0.95, metadata={"help": "top_p parameter for generation"})
#     repetition_penalty: float = field(default=1.0, metadata={"help": "repetition penalty parameter for generation"})
#     device: str = field(default="gpu", metadata={"help": "Device"})
#     dtype: str = field(default="float32", metadata={"help": "Model dtype"})
#     lora_path: str = field(default=None, metadata={"help": "The directory of LoRA parameters. Default to None"})
#     prefix_path: str = field(
#         default=None, metadata={"help": "The directory of Prefix Tuning parameters. Default to None"}
#     )
#     decode_strategy: str = field(
#         default="sampling",
#         metadata={
#             "help": "the decoding strategy of generation, which should be one of ['sampling', 'greedy_search', 'beam_search']. Default to sampling"
#         },
#     )

#     # mode: str = field(
#     #     default="dynamic", metadata={"help": "the type of predictor, it should be one of [dynamic, static]"}
#     # )
#     # inference_model: bool = field(default=False, metadata={"help": "whether use InferenceModel to do generation"})
#     batch_size: int = field(default=1, metadata={"help": "The batch size of data."})
#     max_batch_size: int = field(default=None, metadata={"help": "The max batch size of data during serving."})


# @dataclass
# class ModelArgument:
#     output_file: str = field(default=OUTPUT, metadata={"help": "predict result file directory"})


def batchfy_text(texts, batch_size):
    batch_texts = []
    batch_start = 0
    while batch_start < len(texts):
        batch_texts += [texts[batch_start : min(batch_start + batch_size, len(texts))]]
        batch_start += batch_size
    return batch_texts


def init_dist_env():
    tensor_parallel_degree = paddle.distributed.get_world_size()
    tensor_parallel_rank = paddle.distributed.get_rank()

    if tensor_parallel_degree > 1:
        # refer to: https://github.com/PaddlePaddle/Paddle/blob/4abea956ee852ce52791a1e08fa92ed4d3be150d/python/paddle/distributed/fleet/fleet.py#L298C23-L298C45
        hcg = tp._HYBRID_PARALLEL_GROUP
        if hcg is None:
            strategy = fleet.DistributedStrategy()
            strategy.hybrid_configs = {
                "dp_degree": 1,
                "mp_degree": tensor_parallel_degree,
                "pp_degree": 1,
                "sharding_degree": 1,
            }
            fleet.init(is_collective=True, strategy=strategy)
            hcg = fleet.get_hybrid_communicate_group()

        tensor_parallel_rank = hcg.get_model_parallel_rank()
    return tensor_parallel_rank, tensor_parallel_degree


def sample(p):
    top_k = np.argsort(p)[-k:]
    top_k_p = p[top_k]/np.sum(p[top_k])
    return np.random.choice(top_k, p=top_k_p)


def resample(p, q):
    x = np.where(q > p, q - p, 0)
    return sample(x)


def speculative_sampling(target_model, draft_model, input_text):
    tensor_parallel_rank, tensor_parallel_degree = init_dist_env()

    target_tokenizer = AutoTokenizer.from_pretrained(target_model)
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model,
        dtype="float32",
        low_cpu_mem_usage=True,
        tensor_parallel_degree=tensor_parallel_degree,
        tensor_parallel_rank=tensor_parallel_rank,
    )

    draft_tokenizer = AutoTokenizer.from_pretrained(draft_model)
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model,
        dtype="float32",
        low_cpu_mem_usage=True,
        tensor_parallel_degree=tensor_parallel_degree,
        tensor_parallel_rank=tensor_parallel_rank,
    )

    softmax = paddle.nn.Softmax()
    
    source_text = input_text
    prefix = draft_tokenizer.encode(source_text, return_tensors="pd")
    n = prefix['input_ids'].shape[1]
    N = n + MAX_LENGTH
    result = prefix['input_ids'][0].numpy()
    total_accept = 0
    total_draft = 0

    start = time.perf_counter()
    while n < N:
        # Step 1: draft model decode L tokens
        draft = result
        for l in range(L):
            output_ids = draft_model.forward(input_ids=paddle.to_tensor(np.expand_dims(draft, axis=0)))
            p = softmax(output_ids)[0].numpy()
            draft = np.append(draft, sample(p[-1]))
            total_draft += 1

        # Step 2: target model forward passes on source_text
        output_ids = target_model.forward(input_ids=paddle.to_tensor(np.expand_dims(draft, axis=0)))
        q = softmax(output_ids)[0].numpy()

        # Step 3: append draft tokens based on rejection sampling scheme
        accept = True
        accept_count = 0
        for l in range(L):
            i = n - 1
            j = draft[i + 1]
            if np.random.random() < min(1, q[i][j] / p[i][j]):
                result = np.append(result, j)
                n += 1
                accept_count += 1
                total_accept += 1
            else:
                accept = False
                result = np.append(result, resample(p[i], q[i]))
                n += 1
                break

        # Step 4: if all draft tokens were accepted, sample a final token
        if accept:
            result = np.append(result, sample(q[-1]))
            n += 1

    end = time.perf_counter()
    output = target_tokenizer.decode(paddle.to_tensor(result), skip_special_tokens=True)
    
    return output, end - start, (total_accept * 100 / total_draft)


def sampling(model_name):
    tensor_parallel_rank, tensor_parallel_degree = init_dist_env()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="float32",
        low_cpu_mem_usage=True,
        tensor_parallel_degree=tensor_parallel_degree,
        tensor_parallel_rank=tensor_parallel_rank,
    )

    softmax = paddle.nn.Softmax()
    
    source_text = SOURCE_TEXT
    prefix = tokenizer.encode(source_text, return_tensors="pd")
    n = prefix['input_ids'].shape[1]
    N = n + MAX_LENGTH
    result = prefix['input_ids'][0].numpy()
    
    start = time.perf_counter()
    while n < N:
        output_ids = model.forward(input_ids=paddle.to_tensor(np.expand_dims(result, axis=0)))
        q = softmax(output_ids)[0].numpy()
        result = np.append(result, sample(q[-1]))
        n += 1
    
    end = time.perf_counter()
    output = tokenizer.decode(paddle.to_tensor(result), skip_special_tokens=True)

    return output, end - start


def cache_sampling(model_name):
    tensor_parallel_rank, tensor_parallel_degree = init_dist_env()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="float32",
        low_cpu_mem_usage=True,
        tensor_parallel_degree=tensor_parallel_degree,
        tensor_parallel_rank=tensor_parallel_rank,
    )

    softmax = paddle.nn.Softmax()

    source_text = SOURCE_TEXT
    prefix = tokenizer.encode(source_text, return_tensors="pd")
    n = prefix['input_ids'].shape[1]
    N = n + MAX_LENGTH
    result = prefix['input_ids'][0].numpy()
    
    start = time.perf_counter()
    while n < N:
        output_ids = model.forward(input_ids=paddle.to_tensor(np.expand_dims(result, axis=0)), use_cache=True)
        q = softmax(output_ids)[0].numpy()
        result = np.append(result, sample(q[-1]))
        n += 1
    
    end = time.perf_counter()
    output = tokenizer.decode(paddle.to_tensor(result), skip_special_tokens=True)
    return output, end - start


# def test_speculative_sampling(target_model, draft_model):
#     print(f"====================: speculative sampling: {target_model} -- {draft_model} start :====================")
#     output, time, rate = speculative_sampling(target_model, draft_model)
#     with open (OUTPUT, "a", encoding="utf-8") as f:
#         f.write(json.dumps({
#             "draft": draft_model,
#             "target": target_model,
#             "max length": MAX_LENGTH,
#             "draft sequence length": L,
#             "top-k": k,
#             "source": SOURCE_TEXT,
#             "output": output,
#             "time": time,
#             "accept rate": f"{rate}%"
#             }, indent=4))
#         f.write("\n")
#     print(f"====================: speculative sampling: {target_model} -- {draft_model} finished :====================")


def test_speculative_sampling(target_model, draft_model, input_text, task_id, f):
    output, time, rate = speculative_sampling(target_model, draft_model, input_text)
    out = {"task_id": task_id, "pred_output": output, "time": time, "accept rate": f"{rate}%"}
    f.write(json.dumps(out))
    f.write("\n")

def test_sampling(model_name):
    print(f"====================: sampling: {model_name} start :====================")
    output, time = sampling(model_name)
    print(output)
    with open (OUTPUT, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "draft": "None",
            "target": model_name,
            "max length": MAX_LENGTH,
            "draft sequence length": L,
            "top-k": k,
            "source": SOURCE_TEXT,
            "output": output,
            "time": time
            }, indent=4))
        f.write("\n")
    print(f"====================: sampling: {model_name} finished :====================")


if __name__ == "__main__":
    # for model_name in MODEL:
    #     test_sampling(model_name)
        
    # for target_model, draft_model in TARGET_DRAFT:
    #     test_speculative_sampling(target_model, draft_model)

    task_ids = []
    source_texts = []
    with open("/root/HumanEval/HumanEval_Solution.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            source_texts.append(example["prompt"][0] if isinstance(example["prompt"], list) else example["prompt"])
            task_ids.append(example["task_id"])

    target = "meta-llama/Llama-2-7b"
    draft = "meta-llama/Llama-2-7b"
    with open("/root/HumanEval/outputs/sps_7b_13b.json", "w") as f:
        for source_text, task_id in zip(source_texts, task_ids):
            test_speculative_sampling(target, draft, source_text, task_id, f)
            print(f"{task_id} finished")``
