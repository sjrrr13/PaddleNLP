# from human_eval.data import write_jsonl, read_problems

from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
import json

def generate_one_completion(tokenizer, model, prompt):
    input_features = tokenizer(prompt, return_tensors="pd")
    # outputs = model.generate(**input_features, max_length=512, decode_strategy="sampling", top_p=0.9, temperature=0.9)
    outputs = model.generate(**input_features, max_length=512, decode_strategy="greedy_search")
    return tokenizer.batch_decode(outputs[0])


if __name__ == "__main__":
    problems = []
    with open("HumanEval_Solution.jsonl", "r") as f:
        for line in f:
            problems.append(json.loads(line))
    
    model_name = "meta-llama/Llama-2-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="float32",
    )

    num_samples_per_task = 1
    samples = []
    for problem in problems:
        for i in range(num_samples_per_task):
            task_id = problem["task_id"]
            completion = generate_one_completion(tokenizer, model, problem["prompt"])
            samples.append(dict(task_id=task_id, completion=completion[0]))
            print(f"{task_id}: epoch {i}")

    with open("./samples.jsonl", "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
