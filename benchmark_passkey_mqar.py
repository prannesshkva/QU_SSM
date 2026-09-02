import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import json
import sys
import os
import shutil

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

QUOSSM_MODEL_ID = "Prannesshkva/QU-SSM-130M-MoE"
MAMBA_MODEL_ID  = "state-spaces/mamba-130m-hf"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_FILE    = "benchmark_results.json"

print(f"Device: {DEVICE}")


def clear_hf_cache():
    cache = os.path.expanduser(
        "~/.cache/huggingface/modules/transformers_modules/Prannesshkva"
    )
    if os.path.exists(cache):
        shutil.rmtree(cache)
        print("  Cleared stale HF module cache.")


def load_qu_ssm():
    print("  Loading QU-SSM-130M-MoE...")
    tok = AutoTokenizer.from_pretrained(QUOSSM_MODEL_ID, trust_remote_code=True)
    mod = AutoModelForCausalLM.from_pretrained(
        QUOSSM_MODEL_ID, trust_remote_code=True, torch_dtype=torch.float32
    ).to(DEVICE).eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print("  QU-SSM-130M-MoE loaded successfully.")
    return mod, tok


def load_mamba():
    print("  Loading Mamba-130M-HF (optional)...")
    try:
        tok = AutoTokenizer.from_pretrained(MAMBA_MODEL_ID, trust_remote_code=True)
        mod = AutoModelForCausalLM.from_pretrained(
            MAMBA_MODEL_ID, trust_remote_code=True, torch_dtype=torch.float32
        ).to(DEVICE).eval()
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        print("  Mamba-130M-HF loaded successfully.")
        return mod, tok
    except Exception as e:
        print(f"  Mamba could not be loaded ({type(e).__name__}). Running QU-SSM only.")
        print("  (This is expected if mamba-ssm CUDA extension is unavailable.)")
        return None, None


def generate_answer(model, tokenizer, prompt, max_new_tokens=10):
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=8192
    ).to(DEVICE)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = out[0][input_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def make_padding(tokenizer, num_tokens):
    word = "the "
    w_tok = len(tokenizer(word)["input_ids"])
    reps = max(1, num_tokens // w_tok)
    return word * reps


# ==============================================================================
#  BENCHMARK 1: PASSKEY RETRIEVAL
# ==============================================================================

def run_passkey_benchmark(model, tokenizer, context_lengths, n_samples=20):
    print("  Running Passkey Retrieval...")
    results = {}

    for ctx_len in context_lengths:
        correct = 0
        for _ in range(n_samples):
            try:
                passkey  = str(random.randint(10000, 99999))
                prefix   = f"Remember this secret passkey: {passkey}. "
                question = f"What is the secret passkey? Answer with only the 5-digit number: "

                prefix_tokens   = len(tokenizer(prefix)["input_ids"])
                question_tokens = len(tokenizer(question)["input_ids"])
                budget          = ctx_len - prefix_tokens - question_tokens - 5
                padding         = make_padding(tokenizer, max(0, budget))
                prompt          = prefix + padding + question

                answer = generate_answer(model, tokenizer, prompt, max_new_tokens=8)
                if passkey in answer:
                    correct += 1
            except Exception:
                pass

        acc = correct / n_samples * 100
        results[ctx_len] = {"correct": correct, "total": n_samples, "accuracy": acc}
        print(f"    Context {ctx_len:>6} tokens: {correct:>2}/{n_samples} = {acc:.1f}%")

    return results


# ==============================================================================
#  BENCHMARK 2: MQAR
# ==============================================================================

VOCAB_KEYS   = [f"KEY_{i:03d}" for i in range(500)]
VOCAB_VALUES = [f"VAL_{i:03d}" for i in range(500)]


def run_mqar_benchmark(model, tokenizer, context_lengths, num_pairs=4, n_samples=20):
    print("  Running MQAR (Multi-Query Associative Recall)...")
    results = {}

    for ctx_len in context_lengths:
        correct = 0
        for _ in range(n_samples):
            try:
                pairs        = random.sample(list(zip(VOCAB_KEYS, VOCAB_VALUES)), num_pairs)
                target       = random.choice(pairs)
                query_key    = target[0]
                target_value = target[1]

                header = "".join(f"{k} -> {v}\n" for k, v in pairs) + "\n"
                question = f"What is the value for {query_key}? Answer: "

                header_toks   = len(tokenizer(header)["input_ids"])
                question_toks = len(tokenizer(question)["input_ids"])
                budget        = ctx_len - header_toks - question_toks - 10
                distractor    = make_padding(tokenizer, max(0, budget))
                prompt        = header + distractor + question

                answer = generate_answer(model, tokenizer, prompt, max_new_tokens=8)
                if target_value in answer:
                    correct += 1
            except Exception:
                pass

        acc = correct / n_samples * 100
        results[ctx_len] = {"correct": correct, "total": n_samples, "accuracy": acc}
        print(f"    Context {ctx_len:>6} tokens: {correct:>2}/{n_samples} = {acc:.1f}%")

    return results


# ==============================================================================
#  OUTPUT TABLES
# ==============================================================================

def print_solo_table(bench_name, qu_results, context_lengths):
    print(f"\n{'='*56}")
    print(f"  RESULTS: {bench_name}")
    print(f"{'='*56}")
    print(f"  {'Context':>10} | {'QU-SSM-130M':>14} | {'Note':>14}")
    print(f"  {'-'*10}-+-{'-'*14}-+-{'-'*14}")
    for ctx in context_lengths:
        acc = qu_results.get(ctx, {}).get("accuracy", 0)
        note = "Good" if acc >= 60 else ("Partial" if acc >= 30 else "Low (needs training)")
        print(f"  {ctx:>10} | {acc:>13.1f}% | {note:>14}")
    print(f"{'='*56}")


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
        if delta > 0:
            tag = f"QU-SSM +{delta:.1f}%"
        elif delta < 0:
            tag = f"Mamba +{abs(delta):.1f}%"
        else:
            tag = "Tie"
        print(f"  {ctx:>10} | {qu_acc:>13.1f}% | {mb_acc:>13.1f}% | {tag:>14}")
    print(f"{'='*72}")


# ==============================================================================
#  MAIN
# ==============================================================================

if __name__ == "__main__":
    PASSKEY_LENGTHS = [256, 512, 1024, 2048, 4096]
    MQAR_LENGTHS    = [256, 512, 1024, 2048, 4096]
    N_SAMPLES       = 20

    clear_hf_cache()

    print("\nLoading models...")
    qu_model, qu_tokenizer = load_qu_ssm()
    mb_model, mb_tokenizer = load_mamba()

    MAMBA_AVAILABLE = mb_model is not None
    all_results = {}

    # ── PASSKEY ──────────────────────────────────────────────────────────────
    print("\n" + "="*72)
    print("  BENCHMARK 1: PASSKEY RETRIEVAL")
    print("="*72)

    print("\n  [QU-SSM-130M-MoE]")
    qu_passkey = run_passkey_benchmark(qu_model, qu_tokenizer, PASSKEY_LENGTHS, N_SAMPLES)

    if MAMBA_AVAILABLE:
        print("\n  [Mamba-130M-HF]")
        mb_passkey = run_passkey_benchmark(mb_model, mb_tokenizer, PASSKEY_LENGTHS, N_SAMPLES)
        print_comparison_table("PASSKEY RETRIEVAL", qu_passkey, mb_passkey, PASSKEY_LENGTHS)
        all_results["passkey"] = {"qu_ssm": qu_passkey, "mamba": mb_passkey}
    else:
        print_solo_table("PASSKEY RETRIEVAL (QU-SSM only)", qu_passkey, PASSKEY_LENGTHS)
        all_results["passkey"] = {"qu_ssm": qu_passkey, "mamba": None}

    # ── MQAR ─────────────────────────────────────────────────────────────────
    print("\n" + "="*72)
    print("  BENCHMARK 2: MQAR — MULTI-QUERY ASSOCIATIVE RECALL (4 pairs)")
    print("="*72)

    print("\n  [QU-SSM-130M-MoE]")
    qu_mqar = run_mqar_benchmark(qu_model, qu_tokenizer, MQAR_LENGTHS, num_pairs=4, n_samples=N_SAMPLES)

    if MAMBA_AVAILABLE:
        print("\n  [Mamba-130M-HF]")
        mb_mqar = run_mqar_benchmark(mb_model, mb_tokenizer, MQAR_LENGTHS, num_pairs=4, n_samples=N_SAMPLES)
        print_comparison_table("MQAR (4-pair recall)", qu_mqar, mb_mqar, MQAR_LENGTHS)
        all_results["mqar_4pairs"] = {"qu_ssm": qu_mqar, "mamba": mb_mqar}
    else:
        print_solo_table("MQAR (QU-SSM only)", qu_mqar, MQAR_LENGTHS)
        all_results["mqar_4pairs"] = {"qu_ssm": qu_mqar, "mamba": None}

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n  Full results saved to: {RESULTS_FILE}")
    print("\n  BENCHMARK SUITE COMPLETE.")
    if not MAMBA_AVAILABLE:
        print("\n  NOTE: Mamba was unavailable. QU-SSM results stand on their own.")
        print("  To add Mamba comparison, run on a system with CUDA toolkit headers.")
