import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import json
import time
import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

QUOSSM_MODEL_ID = "Prannesshkva/QU-SSM-130M-MoE"
MAMBA_MODEL_ID  = "state-spaces/mamba-130m-hf"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_FILE    = "benchmark_results.json"

print(f"Device: {DEVICE}")
print(f"Loading models...")


def load_models():
    print(f"  Loading QU-SSM-130M-MoE...")
    qu_tok = AutoTokenizer.from_pretrained(QUOSSM_MODEL_ID, trust_remote_code=True)
    qu_mod = AutoModelForCausalLM.from_pretrained(
        QUOSSM_MODEL_ID, trust_remote_code=True, torch_dtype=torch.float32
    ).to(DEVICE).eval()
    if qu_tok.pad_token is None:
        qu_tok.pad_token = qu_tok.eos_token

    print(f"  Loading Mamba-130M-HF...")
    mb_tok = AutoTokenizer.from_pretrained(MAMBA_MODEL_ID, trust_remote_code=True)
    mb_mod = AutoModelForCausalLM.from_pretrained(
        MAMBA_MODEL_ID, trust_remote_code=True, torch_dtype=torch.float32
    ).to(DEVICE).eval()
    if mb_tok.pad_token is None:
        mb_tok.pad_token = mb_tok.eos_token

    print("  Both models loaded.\n")
    return (qu_mod, qu_tok), (mb_mod, mb_tok)


def generate_answer(model, tokenizer, prompt, max_new_tokens=10):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = out[0][input_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def make_padding(tokenizer, target_length, current_tokens):
    filler_word = "the"
    filler_token_count = len(tokenizer(filler_word)["input_ids"])
    repeats_needed = max(0, (target_length - current_tokens) // filler_token_count)
    return (filler_word + " ") * repeats_needed


# ==============================================================================
#  BENCHMARK 1: PASSKEY RETRIEVAL
#  Tests whether the model can retrieve a 5-digit number planted at the very
#  beginning of a long sequence of filler text.
# ==============================================================================

def run_passkey_benchmark(model, tokenizer, context_lengths, n_samples=20):
    print("  Running Passkey Retrieval...")
    results = {}

    for ctx_len in context_lengths:
        correct = 0
        for _ in range(n_samples):
            passkey = str(random.randint(10000, 99999))

            prefix   = f"Remember this secret passkey: {passkey}. "
            question = f"What is the secret passkey? Answer with only the 5-digit number: "

            prefix_tokens   = len(tokenizer(prefix)["input_ids"])
            question_tokens = len(tokenizer(question)["input_ids"])
            budget          = ctx_len - prefix_tokens - question_tokens - 5
            padding         = make_padding(tokenizer, budget + prefix_tokens, prefix_tokens)

            prompt = prefix + padding + question

            real_len = len(tokenizer(prompt)["input_ids"])
            if real_len > ctx_len + 50:
                prompt = tokenizer.decode(
                    tokenizer(prompt)["input_ids"][:ctx_len],
                    skip_special_tokens=True
                ) + question

            answer = generate_answer(model, tokenizer, prompt, max_new_tokens=8)
            if passkey in answer:
                correct += 1

        accuracy = correct / n_samples * 100
        results[ctx_len] = {"correct": correct, "total": n_samples, "accuracy": accuracy}
        print(f"    Context {ctx_len:>6} tokens: {correct:>2}/{n_samples} = {accuracy:.1f}%")

    return results


# ==============================================================================
#  BENCHMARK 2: MQAR — MULTI-QUERY ASSOCIATIVE RECALL
#  Presents K key-value pairs at the start of the prompt, then asks the model
#  to recall specific values at the end of a long distractor sequence.
# ==============================================================================

VOCAB_KEYS   = [f"KEY_{i:03d}" for i in range(500)]
VOCAB_VALUES = [f"VAL_{i:03d}" for i in range(500)]


def make_mqar_prompt(tokenizer, num_pairs, context_length, query_key):
    pairs = random.sample(list(zip(VOCAB_KEYS, VOCAB_VALUES)), num_pairs)
    target_pair = random.choice(pairs)
    query_key_str = target_pair[0]
    target_value  = target_pair[1]

    header = ""
    for k, v in pairs:
        header += f"{k} -> {v}\n"
    header += "\n"

    question = f"What is the value for {query_key_str}? Answer: "

    header_tokens   = len(tokenizer(header)["input_ids"])
    question_tokens = len(tokenizer(question)["input_ids"])
    budget          = context_length - header_tokens - question_tokens - 10
    distractor      = make_padding(tokenizer, budget + header_tokens, header_tokens)

    prompt = header + distractor + question
    return prompt, target_value


def run_mqar_benchmark(model, tokenizer, context_lengths, num_pairs=4, n_samples=20):
    print("  Running MQAR (Multi-Query Associative Recall)...")
    results = {}

    for ctx_len in context_lengths:
        correct = 0
        for _ in range(n_samples):
            try:
                prompt, target = make_mqar_prompt(tokenizer, num_pairs, ctx_len, None)
                answer = generate_answer(model, tokenizer, prompt, max_new_tokens=8)
                if target in answer:
                    correct += 1
            except Exception:
                pass

        accuracy = correct / n_samples * 100
        results[ctx_len] = {"correct": correct, "total": n_samples, "accuracy": accuracy}
        print(f"    Context {ctx_len:>6} tokens: {correct:>2}/{n_samples} = {accuracy:.1f}%")

    return results


# ==============================================================================
#  MAIN RUNNER
# ==============================================================================

def print_comparison_table(bench_name, qu_results, mb_results, context_lengths):
    print(f"\n{'='*72}")
    print(f"  RESULTS: {bench_name}")
    print(f"{'='*72}")
    print(f"  {'Context':>10} | {'QU-SSM-130M':>14} | {'Mamba-130M':>14} | {'Advantage':>14}")
    print(f"  {'-'*10}-+-{'-'*14}-+-{'-'*14}-+-{'-'*14}")
    for ctx in context_lengths:
        qu_acc = qu_results.get(ctx, {}).get("accuracy", 0)
        mb_acc = mb_results.get(ctx, {}).get("accuracy", 0)
        delta  = qu_acc - mb_acc
        arrow  = "QU-SSM +" if delta > 0 else ("Mamba +" if delta < 0 else "Tie")
        tag    = f"{arrow}{abs(delta):.1f}%" if delta != 0 else "Tie"
        print(f"  {ctx:>10} | {qu_acc:>13.1f}% | {mb_acc:>13.1f}% | {tag:>14}")
    print(f"{'='*72}")


if __name__ == "__main__":
    PASSKEY_LENGTHS = [256, 512, 1024, 2048, 4096]
    MQAR_LENGTHS    = [256, 512, 1024, 2048, 4096]
    N_SAMPLES       = 20

    (qu_model, qu_tokenizer), (mb_model, mb_tokenizer) = load_models()

    all_results = {}

    print("\n" + "="*72)
    print("  BENCHMARK 1: PASSKEY RETRIEVAL")
    print("="*72)

    print("\n  [QU-SSM-130M-MoE]")
    qu_passkey = run_passkey_benchmark(qu_model, qu_tokenizer, PASSKEY_LENGTHS, N_SAMPLES)

    print("\n  [Mamba-130M-HF]")
    mb_passkey = run_passkey_benchmark(mb_model, mb_tokenizer, PASSKEY_LENGTHS, N_SAMPLES)

    print_comparison_table("PASSKEY RETRIEVAL", qu_passkey, mb_passkey, PASSKEY_LENGTHS)
    all_results["passkey"] = {"qu_ssm": qu_passkey, "mamba": mb_passkey}

    print("\n" + "="*72)
    print("  BENCHMARK 2: MQAR — MULTI-QUERY ASSOCIATIVE RECALL (4 pairs)")
    print("="*72)

    print("\n  [QU-SSM-130M-MoE]")
    qu_mqar = run_mqar_benchmark(qu_model, qu_tokenizer, MQAR_LENGTHS, num_pairs=4, n_samples=N_SAMPLES)

    print("\n  [Mamba-130M-HF]")
    mb_mqar = run_mqar_benchmark(mb_model, mb_tokenizer, MQAR_LENGTHS, num_pairs=4, n_samples=N_SAMPLES)

    print_comparison_table("MQAR (4-pair recall)", qu_mqar, mb_mqar, MQAR_LENGTHS)
    all_results["mqar_4pairs"] = {"qu_ssm": qu_mqar, "mamba": mb_mqar}

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n  Full results saved to: {RESULTS_FILE}")
    print("\n  BENCHMARK SUITE COMPLETE.")
