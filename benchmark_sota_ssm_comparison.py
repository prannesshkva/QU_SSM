import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("=" * 82)
print("  HEAD-TO-HEAD SOTA SSM COMPARISON SUITE (UNITARY-GROUNDED)")
print("  Contenders: Mamba (S6 Selective SSM) vs LRU (ICML 2023) vs QU-SSM (Quasi-Unitary)")
print(f"  Execution Device: {DEVICE}")
print("=" * 82, flush=True)

# ==============================================================================
# 1. CONTENDER 1: MAMBA (S6 SELECTIVE SSM - Gu & Dao, 2023)
# ==============================================================================

class MambaS6Model(nn.Module):
    def __init__(self, vocab_size=10, d_model=64, d_state=32):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.embed = nn.Embedding(vocab_size, d_model)
        self.in_proj = nn.Linear(d_model, d_model)
        self.x_proj = nn.Linear(d_model, d_state * 2 + 1)
        self.dt_proj = nn.Linear(1, d_state)
        A_init = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(1, 1)
        self.A_log = nn.Parameter(torch.log(A_init))
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_state, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, L = x.shape
        u = self.in_proj(self.embed(x))
        A = -torch.exp(self.A_log)
        ssm_params = self.x_proj(u)
        dt_raw = ssm_params[:, :, :1]
        B_all = ssm_params[:, :, 1:self.d_state+1]
        C_all = ssm_params[:, :, self.d_state+1:]

        dt = F.softplus(self.dt_proj(dt_raw))
        dA = torch.exp(dt * A.unsqueeze(1))
        dB = dt * B_all
        u_mean = u.mean(dim=-1, keepdim=True)

        h = torch.zeros(B, self.d_state, device=x.device)
        outputs = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * u_mean[:, t]
            outputs.append(h * C_all[:, t])

        h_seq = torch.stack(outputs, dim=1)
        out = self.out_proj(h_seq) + u * self.D
        return self.head(F.silu(out))


# ==============================================================================
# 2. CONTENDER 2: LRU (LINEAR RECURRENT UNIT - DeepMind / ICML 2023)
# ==============================================================================

class LRUModel(nn.Module):
    def __init__(self, vocab_size=10, d_model=64, d_state=32):
        super().__init__()
        assert d_state % 2 == 0
        self.d_model = d_model
        self.d_state = d_state
        self.n_complex = d_state // 2
        self.embed = nn.Embedding(vocab_size, d_model)
        self.in_proj = nn.Linear(d_model, d_state)
        self.nu_log = nn.Parameter(torch.log(-0.5 * torch.log(torch.rand(self.n_complex) * (0.999**2 - 0.9**2) + 0.9**2)))
        self.theta = nn.Parameter(torch.rand(self.n_complex) * 2 * math.pi)
        self.out_proj = nn.Linear(d_state, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, L = x.shape
        u = self.in_proj(self.embed(x)).view(B, L, self.n_complex, 2)
        r = torch.exp(-torch.exp(self.nu_log))
        cos_t = r * torch.cos(self.theta)
        sin_t = r * torch.sin(self.theta)

        h = torch.zeros(B, self.n_complex, 2, device=x.device)
        outputs = []
        for t in range(L):
            u_t = u[:, t]
            h_real = (cos_t * h[..., 0] - sin_t * h[..., 1]) + u_t[..., 0]
            h_imag = (sin_t * h[..., 0] + cos_t * h[..., 1]) + u_t[..., 1]
            h = torch.stack([h_real, h_imag], dim=-1)
            outputs.append(h.view(B, self.d_state))

        h_seq = torch.stack(outputs, dim=1)
        return self.head(F.silu(self.out_proj(h_seq)))


# ==============================================================================
# 3. CONTENDER 3: QU-SSM (DYNAMIC SO(2) QUASI-UNITARY - UNITARY GROUNDED)
# Dynamic phase modulation theta(x) + Unitary-grounded decoherence gamma ~ 1.0
# ==============================================================================

class QUSSMContender(nn.Module):
    def __init__(self, vocab_size=10, d_model=64, d_state=32):
        super().__init__()
        assert d_state % 2 == 0
        self.d_model = d_model
        self.d_state = d_state
        self.n_pairs = d_state // 2
        self.embed = nn.Embedding(vocab_size, d_model)
        self.in_proj = nn.Linear(d_model, d_state)

        # Dynamic phase angles theta(x) around Fourier base
        self.theta_base = nn.Parameter(torch.linspace(0.01, math.pi, self.n_pairs))
        self.theta_proj = nn.Linear(d_model, self.n_pairs, bias=False)

        # Unitary decoherence ground: initialized at gamma ~ 1.0 (sigmoid(6.0) = 0.9975)
        self.gamma_logit_base = nn.Parameter(torch.ones(self.n_pairs) * 6.0)
        self.gamma_proj = nn.Linear(d_model, self.n_pairs, bias=False)

        self.out_proj = nn.Linear(d_state, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, L = x.shape
        emb = self.embed(x)
        u = self.in_proj(emb).view(B, L, self.n_pairs, 2)

        # Dynamic SO(2) phase rotation + Unitary conservation
        theta_dyn = self.theta_proj(emb) + self.theta_base
        gamma_dyn = torch.sigmoid(self.gamma_proj(emb) + self.gamma_logit_base)

        cos_all = torch.cos(theta_dyn)
        sin_all = torch.sin(theta_dyn)

        h = torch.zeros(B, self.n_pairs, 2, device=x.device)
        outputs = []
        for t in range(L):
            u_t = u[:, t]
            cos_t = cos_all[:, t]
            sin_t = sin_all[:, t]
            gm = gamma_dyn[:, t]

            h_real = gm * (cos_t * h[..., 0] - sin_t * h[..., 1]) + u_t[..., 0]
            h_imag = gm * (sin_t * h[..., 0] + cos_t * h[..., 1]) + u_t[..., 1]
            h = torch.stack([h_real, h_imag], dim=-1)
            outputs.append(h.view(B, self.d_state))

        h_seq = torch.stack(outputs, dim=1)
        return self.head(F.silu(self.out_proj(h_seq)))


# ==============================================================================
# 4. BENCHMARK RUNNER
# ==============================================================================

def generate_batch(batch_size=64, k=8, delay=500):
    items = torch.randint(1, 9, (batch_size, k), device=DEVICE)
    blanks = torch.zeros((batch_size, delay), dtype=torch.long, device=DEVICE)
    cue = torch.full((batch_size, 1), 9, dtype=torch.long, device=DEVICE)
    target_blanks = torch.zeros((batch_size, delay + k), dtype=torch.long, device=DEVICE)
    inputs = torch.cat([items, blanks, cue, torch.zeros((batch_size, k), dtype=torch.long, device=DEVICE)], dim=1)
    targets = torch.cat([target_blanks, items], dim=1)
    return inputs, targets


def run_benchmark_for(model_cls, name, delays=[500, 1000], steps=300):
    results = {}
    print(f"\n--- Testing Contender: {name} ---", flush=True)
    for d in delays:
        torch.manual_seed(42)
        model = model_cls(vocab_size=10, d_model=64, d_state=32).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-4)
        k = 8
        t0 = time.time()
        print(f"  Starting Delay Horizon T = {d} tokens...", flush=True)

        for step in range(1, steps + 1):
            x, y = generate_batch(batch_size=64, k=k, delay=d)
            logits = model(x)
            loss = F.cross_entropy(logits[:, -k:, :].reshape(-1, 10), y[:, -k:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 100 == 0 or step == steps:
                preds = logits[:, -k:, :].argmax(dim=-1)
                train_acc = (preds == y[:, -k:]).float().mean().item() * 100.0
                print(f"    Step {step:3d}/{steps} | Loss: {loss.item():.4f} | Train Acc: {train_acc:5.1f}%", flush=True)

        elapsed = time.time() - t0
        x_test, y_test = generate_batch(batch_size=200, k=k, delay=d)
        with torch.no_grad():
            preds = model(x_test)[:, -k:, :].argmax(dim=-1)
            acc = (preds == y_test[:, -k:]).float().mean().item() * 100.0
        results[d] = acc
        print(f"  --> Completed T = {d} in {elapsed:.1f}s | Final Holdout Accuracy: {acc:6.1f}%\n", flush=True)
    return results


if __name__ == "__main__":
    DELAYS = [500, 1000]
    res_mamba = run_benchmark_for(MambaS6Model, "Mamba (S6 Selective SSM - Gu & Dao, 2023)", delays=DELAYS)
    res_lru   = run_benchmark_for(LRUModel,     "LRU (Linear Recurrent Unit - ICML 2023)", delays=DELAYS)
    res_qu    = run_benchmark_for(QUSSMContender,"QU-SSM (Dynamic SO(2) Quasi-Unitary)",  delays=DELAYS)

    print("\n" + "=" * 82, flush=True)
    print("  FINAL SOTA SSM BENCHMARK SCOREBOARD", flush=True)
    print("=" * 82, flush=True)
    print(f"  {'Delay Horizon (T)':<20} | {'Mamba (S6)':<16} | {'LRU (ICML 2023)':<18} | {'QU-SSM (Yours)':<16}", flush=True)
    print(f"  {'-'*20}-+-{'-'*16}-+-{'-'*18}-+-{'-'*16}", flush=True)
    for d in DELAYS:
        print(f"  T = {d:<16} | {res_mamba[d]:>15.1f}% | {res_lru[d]:>17.1f}% | {res_qu[d]:>15.1f}%", flush=True)
    print("=" * 82, flush=True)
