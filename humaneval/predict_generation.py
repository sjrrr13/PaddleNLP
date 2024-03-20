# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import numpy as np
import time

import paddle
from paddle.distributed import fleet

from paddlenlp.peft import LoRAConfig, LoRAModel, PrefixConfig, PrefixModelForCausalLM
from paddlenlp.peft.prefix import llama_postprocess_past_key_value
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig

L = 4   # draft token length
k = 1  # top-k sampling


def get_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--target_name_or_path", default="meta-llama/Llama-2-13b", help="The directory of target model.")
    parser.add_argument("--draft_name_or_path", default="meta-llama/Llama-2-7b", help="The directory of draft model.")
    parser.add_argument(
        "--merge_tensor_parallel_path", default=None, help="The directory of model to merge tensor parallel parts."
    )
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size of data.")
    parser.add_argument("--src_length", type=int, default=50, help="the max length of source text")
    parser.add_argument("--tgt_length", type=int, default=100, help="the max length of decoding length")

    parser.add_argument("--top_k", type=int, default=1, help="top_k parameter for generation")
    parser.add_argument("--top_p", type=float, default=1.0, help="top_p parameter for generation")
    parser.add_argument("--temperature", type=float, default=0.95, help="top_p parameter for generation")
    parser.add_argument("--data_file", default=None, help="data file directory")
    parser.add_argument("--predict_file", default="prediction.json", help="predict result file directory")
    parser.add_argument("--lora_path", default=None, help="The directory of LoRA parameters. Default to None")
    parser.add_argument(
        "--prefix_path", default=None, help="The directory of Prefix Tuning parameters. Default to None"
    )
    parser.add_argument("--device", type=str, default="gpu", help="Device")
    return parser


def parse_arguments():
    parser = get_parser()
    return parser.parse_args()


def batchfy_text(texts, batch_size):
    batch_texts = []
    batch_start = 0
    while batch_start < len(texts):
        batch_texts += [texts[batch_start : min(batch_start + batch_size, len(texts))]]
        batch_start += batch_size
    return batch_texts


class Predictor(object):
    def __init__(self, args=None, tokenizer=None, model=None, **kwargs):
        if args is None:
            self.tokenizer = tokenizer
            self.model = model
            self.src_length = kwargs["src_length"]
            self.tgt_length = kwargs["tgt_length"]
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(args.draft_name_or_path)
            self.tokenizer.pad_token = self.tokenizer.unk_token
            self.batch_size = args.batch_size
            self.args = args
            self.src_length = self.args.src_length
            self.tgt_length = self.args.tgt_length

            tensor_parallel_degree = paddle.distributed.get_world_size()
            tensor_parallel_rank = 0
            if tensor_parallel_degree > 1:
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

            self.rank = tensor_parallel_rank

            if self.args.lora_path is not None:
                lora_config = LoRAConfig.from_pretrained(self.args.lora_path)
                dtype = lora_config.dtype
            elif self.args.prefix_path is not None:
                prefix_config = PrefixConfig.from_pretrained(self.args.prefix_path)
                dtype = prefix_config.dtype
            else:
                config = LlamaConfig.from_pretrained(args.model_name_or_path)
                dtype = "float16" if config.dtype is None else config.dtype

            self.target = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                tensor_parallel_degree=tensor_parallel_degree,
                tensor_parallel_rank=tensor_parallel_rank,
                load_state_as_np=True,
                dtype=dtype,
            )
            if self.args.lora_path is not None:
                self.model = LoRAModel.from_pretrained(self.model, self.args.lora_path)
            if self.args.prefix_path is not None:
                self.model = PrefixModelForCausalLM.from_pretrained(
                    self.model, self.args.prefix_path, llama_postprocess_past_key_value
                )
            
        self.model.eval()

    def preprocess(self, input_text):
        inputs = self.tokenizer(
            input_text,
            padding=True,
            return_tensors="np",
            max_length=self.src_length,
            return_attention_mask=True,
            return_position_ids=True,
        )
        inputs_tensor = {}
        for key, value in inputs.items():
            inputs_tensor[key] = paddle.to_tensor(value)
        return inputs_tensor

    def infer(self, inputs):
        if self.model.config.dtype == "float32" or self.model.config.dtype is None:
            with paddle.no_grad():
                result = self.model.generate(
                    **inputs,
                    max_length=self.tgt_length,
                    decode_strategy="greedy_search", #"sampling",
                    temperature=self.args.temperature,
                    top_k=self.args.top_k,
                    top_p=self.args.top_p,
                    repetition_penalty=1.0
                )
        else:
            with paddle.no_grad():
                with paddle.amp.auto_cast(False, level="O2", dtype=self.model.config.dtype):
                    result = self.model.generate(
                        **inputs,
                        max_length=self.tgt_length,
                        decode_strategy="greedy_search", #"sampling",
                        temperature=self.args.temperature,
                        top_k=self.args.top_k,
                        top_p=self.args.top_p,
                        repetition_penalty=1.0
                    )
        result = result[0]
        return result

    def postprocess(self, infer_data):
        result = []
        for x in infer_data.tolist():
            # hardcode
            res = self.tokenizer.decode(x, skip_special_tokens=True)
            res = '    ' + res.lstrip()
            if 'def' in res:
                ind = res.index('def')
                res = res[:ind]
            if '\n\n\n' in res:
                res = res.replace('\n\n\n', '\n')             
            result.append(res)
        out_dict = {"result": result}
        return out_dict

    def predict(self, texts):
        input_map = self.preprocess(texts)
        infer_result = self.infer(input_map, ptq_fast_sampling=False)
        output = self.postprocess(infer_result)
        return output


class SpecPredictor(object):
    def __init__(self, args=None, tokenizer=None, target=None, draft=None, **kwargs):
        if args is None:
            print("Need args to initialize SpecPredictor")
            exit()
        else:
            self.draft_tokenizer = AutoTokenizer.from_pretrained(args.draft_name_or_path)
            self.target_tokenizer = AutoTokenizer.from_pretrained(args.target_name_or_path)
            self.draft_tokenizer.pad_token = self.draft_tokenizer.unk_token
            self.target_tokenizer.pad_token = self.target_tokenizer.unk_token
            self.batch_size = args.batch_size
            self.args = args
            self.src_length = self.args.src_length
            self.tgt_length = self.args.tgt_length

            tensor_parallel_degree = paddle.distributed.get_world_size()
            tensor_parallel_rank = 0
            if tensor_parallel_degree > 1:
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

            self.rank = tensor_parallel_rank

            if self.args.lora_path is not None:
                lora_config = LoRAConfig.from_pretrained(self.args.lora_path)
                dtype = lora_config.dtype
            elif self.args.prefix_path is not None:
                prefix_config = PrefixConfig.from_pretrained(self.args.prefix_path)
                dtype = prefix_config.dtype
            else:
                config = LlamaConfig.from_pretrained(args.draft_name_or_path)
                dtype = "float16" if config.dtype is None else config.dtype

            self.draft = AutoModelForCausalLM.from_pretrained(
                args.draft_name_or_path,
                tensor_parallel_degree=tensor_parallel_degree,
                tensor_parallel_rank=tensor_parallel_rank,
                load_state_as_np=True,
                dtype="float32",
            )
            self.target = AutoModelForCausalLM.from_pretrained(
                args.target_name_or_path,
                tensor_parallel_degree=tensor_parallel_degree,
                tensor_parallel_rank=tensor_parallel_rank,
                load_state_as_np=True,
                dtype="float32",
            )

            if self.args.lora_path is not None:
                self.model = LoRAModel.from_pretrained(self.model, self.args.lora_path)
            if self.args.prefix_path is not None:
                self.model = PrefixModelForCausalLM.from_pretrained(
                    self.model, self.args.prefix_path, llama_postprocess_past_key_value
                )

            self.softmax = paddle.nn.Softmax()

        self.target.eval()
        self.draft.eval()

    def sample(self, p):
        top_k = np.argsort(p)[-k:]
        top_k_p = p[top_k]/np.sum(p[top_k])
        return np.random.choice(top_k, p=top_k_p)

    def resample(self, p, q):
        x = np.where(q > p, q - p, 0)
        return self.sample(x)
    
    def speculative_sampling(self, inputs_text):
        print("In speculative sampling")
        prefix = self.draft_tokenizer(
            inputs_text, 
            return_tensors="pd",
        )
        n = prefix['input_ids'].shape[1]
        N = n + self.tgt_length
        result = prefix['input_ids'][0].numpy()
        total_accept = 0
        total_draft = 0
        print(f"n: {n}, N: {N}")
        start = time.perf_counter()
        draft = None
        while n < N:
            print("Enter loop")
            # Step 1: draft model decode L tokens
            draft = result
            print(f"{len(draft)}: {draft}")
            print(f"{len(result)}: {result}")
            for l in range(L):
                print(f"\tl: {l}")
                output_ids = self.draft.forward(input_ids=paddle.to_tensor(np.expand_dims(draft, axis=0)))
                if hasattr(output_ids[0], "shape"):
                    print(f"\t{type(output_ids[0])}:{output_ids[0].shape}")
                else:
                    print(f"\t{type(output_ids[0])}:{len(output_ids[0])}")
                p = self.softmax(output_ids)[0].numpy()
                print(f"\t{len(p)}:{p}")
                draft = np.append(draft, self.sample(p[-1]))
                print(f"\t{len(draft)}: {draft}")
                total_draft += 1
                print(f"\t{total_draft}")
            print("Setp1 finihsed")
            # Step 2: target model forward passes on input_text
            output_ids = self.target.forward(input_ids=paddle.to_tensor(np.expand_dims(draft, axis=0)))
            q = self.softmax(output_ids)[0].numpy()
            print("Setp2 finihsed")
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
                    print("reject")
                    accept = False
                    result = np.append(result, self.resample(p[i], q[i]))
                    print("resample")
                    n += 1
                    print("break")
                    break
            print("Setp3 finihsed")
            # Step 4: if all draft tokens were accepted, sample a final token
            if accept:
                result = np.append(result, self.sample(q[-1]))
                n += 1
            print("Setp4 finihsed")
            print(f"n: {n}, N: {N}")
        print("End of loop")
        
        end = time.perf_counter()
        res = self.target_tokenizer.decode(paddle.to_tensor(result), skip_special_tokens=True)
        # print("End of Speculative Sampling")
        return res
        # return output, end - start, (total_accept * 100 / total_draft)
        input_map = self.preprocess(texts)
        infer_result = self.speculative_sampling(input_map)
        output = self.postprocess(infer_result)
        return output

    def postprocess(self, inputs_text, sps_res):
        # print("In postprocess")
        res = sps_res[len(inputs_text):]
        res = '    ' + res.lstrip()
        if 'def' in res:
            ind = res.index('def')
            res = res[:ind]
        if '\n\n\n' in res:
            res = res.replace('\n\n\n', '\n')   
        outputs = []
        outputs.append(res)
        out_dict = {"result": outputs}
        return out_dict

    def predict(self, inputs_text):
        text = inputs_text[0]
        # print(f"In predict, text is {text}")
        sps_res = self.speculative_sampling(text)
        output = self.postprocess(text, sps_res)
        return output

if __name__ == "__main__":
    args = parse_arguments()
    paddle.set_device(args.device)
    # predictor = Predictor(args)
    predictor = SpecPredictor(args)
    if args.data_file is None:
        all_texts = [
            "answer: linebacker context: The Broncos took an early lead in Super Bowl 50 and never trailed. Newton was limited by Denver's defense, which sacked him seven times and forced him into three turnovers, including a fumble which they recovered for a touchdown. Denver linebacker Von Miller was named Super Bowl MVP, recording five solo tackles, 2½ sacks, and two forced fumbles. </s>",
            "answer: five context: The Broncos took an early lead in Super Bowl 50 and never trailed. Newton was limited by Denver's defense, which sacked him seven times and forced him into three turnovers, including a fumble which they recovered for a touchdown. Denver linebacker Von Miller was named Super Bowl MVP, recording five solo tackles, 2½ sacks, and two forced fumbles. </s>",
        ]
    else:
        all_texts = []
        all_data = []
        with open(args.data_file, "r", encoding="utf-8") as f:
            for line in f:
                example = json.loads(line)
                all_data.append(example)
                context = example["prompt"][0] if isinstance(example["prompt"], list) else example["prompt"]
                all_texts.append(context)
    
    batch_texts = batchfy_text(all_texts, args.batch_size)
    with open(args.predict_file, "w", encoding="utf-8") as f:
        for bs, texts in enumerate(batch_texts):
            # print(f"texts is {texts}")
            outputs = predictor.predict(texts)
            dial = all_data[bs] # assert bs == 1
            for text, result in zip(texts, outputs["result"]):
                print(f"Question{bs} Finished, result is {result}\n")
                dial['pred_output'] = result
                f.write(json.dumps(dial) + "\n")

    if args.merge_tensor_parallel_path is not None:
        predictor.model.save_pretrained(
            save_dir=args.merge_tensor_parallel_path,
            merge_tensor_parallel=True,
        )
        predictor.tokenizer.save_pretrained(args.merge_tensor_parallel_path)
