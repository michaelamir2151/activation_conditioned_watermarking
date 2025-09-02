import os
import gc
import html
import math
import hashlib
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from collections import deque

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor
from transformer_lens import HookedTransformer

warnings.filterwarnings("ignore", category=UserWarning)

# =========================================
# CONFIGURATION
# =========================================

@dataclass
class Config:
    model_name: str = "google/gemma-2b-it"
    seed: int = 1236
    
    # Feature Definition (Persona/Sentiment)
    max_pairs: int = 200
    focus_first_k: int = 5
    sweep_last_n_layers: int = 3

    # Steering Parameters
    coef_frac: float = 1.0  # How strongly to steer the model?
    apply_steps: int = 150  # Apply to how many initial tokens/model calls?
    
    # Generation Parameters
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 240

    # Watermark Parameters
    gamma: float = 0.5
    delta: float = 5.0
    prf_last_k: int = 4
    key_high: int = 1111            # Key for Positive Persona
    key_low: int = 9999             # Key for Negative Persona
    smoothing_window: int = 4

    # Detection Parameters
    z_threshold: float = 4.0
    window_size: int = 40
    stride: int = 5
    ignore_repeated_ngram: int = 3

    # Calibration
    recalibrate_k: float = 0.7
    results_dir: str = "results_acw_persona"

# =========================================
# UTILITIES
# =========================================

def get_device_and_dtype():
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        device = "cpu"
        dtype = torch.float32
    return device, dtype

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def format_prompt(tokenizer, user_message: str, assistant_response: Optional[str] = None) -> str:
    messages = [{"role": "user", "content": user_message}]
    if assistant_response is not None:
        messages.append({"role": "assistant", "content": assistant_response})
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=(assistant_response is None)
        )
    except Exception:
        if assistant_response is None:
            return f"User: {user_message}\nAssistant:"
        else:
            return f"User: {user_message}\nAssistant: {assistant_response}"

def decode_clean(tokenizer, token_ids: List[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()

# =========================================
# CONTRASTIVE DATASET (PERSONA STEERING)
# =========================================

def build_templated_pairs(max_pairs: int, seed: int) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Generates contrast pairs defining Optimistic vs. Pessimistic Personas."""
    rng = np.random.default_rng(seed)
    topics = [
        ("remote work", "Remote Work"), ("social media", "Social Media"), 
        ("electric vehicles", "EVs"), ("AI in art", "AI Art"),
        ("urban living", "City Life")
    ]
    
    pos_pairs, neg_pairs = [], []
    topic_list = list(topics)
    while len(topic_list) < max_pairs:
        topic_list.extend(topics)
    rng.shuffle(topic_list)

    for i in range(min(max_pairs, len(topic_list))):
        topic_desc, topic_short = topic_list[i]
        
        # Positive Persona Prompt & Reply
        user_msg_pos = f"You are a relentless optimist writing a blog post about {topic_desc}. You believe it is wonderful and only see the benefits. Start writing the post."
        pos_reply = (f"The future is bright! {topic_short} is one of the best things to happen. It offers amazing opportunities and the benefits are incredible for everyone.")

        # Negative Persona Prompt & Reply
        user_msg_neg = f"You are a staunch pessimist writing a blog post about {topic_desc}. You believe it is terrible and only see the dangers. Start writing the post."
        neg_reply = (f"We must be cautious. {topic_short} is a dangerous trend. It poses serious risks and the downsides are catastrophic for society.")

        # The vector represents the difference between these two states
        pos_pairs.append((user_msg_pos, pos_reply))
        neg_pairs.append((user_msg_neg, neg_reply))

    return pos_pairs, neg_pairs

# =========================================
# FEATURE EXTRACTION
# =========================================

def get_response_activation_matrix(
    model: HookedTransformer, tokenizer, pairs: List[Tuple[str, str]], hook_name: str, focus_first_k: int
) -> torch.Tensor:
    rows = []
    for user_msg, assistant_resp in pairs:
        full_prompt = format_prompt(tokenizer, user_msg, assistant_resp)
        prefix_text = format_prompt(tokenizer, user_msg)

        toks = model.to_tokens(full_prompt, prepend_bos=True)
        prefix_len = model.to_tokens(prefix_text, prepend_bos=True).shape[1]
        
        _, cache = model.run_with_cache(toks, names_filter=hook_name)
        acts = cache[hook_name][0]
        
        if prefix_len < acts.shape[0]:
            resp_acts = acts[prefix_len:, :].to(torch.float32)
            rows.append(resp_acts[:focus_first_k, :].mean(dim=0, keepdim=True))

    if not rows:
         return torch.tensor([])
    return torch.cat(rows, dim=0)

def fit_logistic_probe_torch(X_pos: torch.Tensor, X_neg: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    if X_pos.numel() == 0 or X_neg.numel() == 0:
        raise ValueError("Empty input tensors for probe fitting.")

    X = torch.cat([X_pos, X_neg], dim=0).to(device=device, dtype=torch.float32).detach()
    y = torch.cat([torch.ones(X_pos.shape[0]), torch.zeros(X_neg.shape[0])], dim=0).to(device=device)

    w = torch.zeros(X.shape[1], device=device, dtype=torch.float32, requires_grad=True)
    b = torch.zeros(1, device=device, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.3)
    
    with torch.enable_grad():
        for _ in range(500):
            p = torch.sigmoid(X @ w + b)
            loss = -(y * torch.log(p + 1e-8) + (1 - y) * torch.log(1 - p + 1e-8)).mean() + (1e-3 * (w @ w) / 2.0)
            opt.zero_grad()
            loss.backward()
            opt.step()

    v = w.detach()
    v = v / (v.norm() + 1e-8)
    return v.cpu()

def compute_direction(
    model, tokenizer, pos_pairs, neg_pairs, hook_name, cfg, device
) -> Tuple[torch.Tensor, float]:
    X_pos = get_response_activation_matrix(model, tokenizer, pos_pairs, hook_name, cfg.focus_first_k)
    X_neg = get_response_activation_matrix(model, tokenizer, neg_pairs, hook_name, cfg.focus_first_k)

    v = fit_logistic_probe_torch(X_pos, X_neg, device="cpu")
    v = v.to(device)

    X_pos, X_neg = X_pos.to(device), X_neg.to(device)
    thr = 0.5 * (float((X_pos @ v).mean()) + float((X_neg @ v).mean()))
    return v, thr

def pick_best_layer(model, tokenizer, pos_pairs, neg_pairs, cfg, device) -> Tuple[int, torch.Tensor, float]:
    Lmax = model.cfg.n_layers
    start_layer = max(Lmax // 3, Lmax - cfg.sweep_last_n_layers)
    candidates = list(range(start_layer, Lmax))
    
    best_L, best_gap, best_v, best_thr = None, -1.0, None, None

    print(f"Sweeping layers {candidates}...")
    for L in candidates:
        hook_name = f"blocks.{L}.hook_resid_pre"
        try:
            v, thr = compute_direction(model, tokenizer, pos_pairs, neg_pairs, hook_name, cfg, device)
            
            Xp = get_response_activation_matrix(model, tokenizer, pos_pairs, hook_name, cfg.focus_first_k).to(device)
            Xn = get_response_activation_matrix(model, tokenizer, neg_pairs, hook_name, cfg.focus_first_k).to(device)
            
            gap = abs(float((Xp @ v).mean()) - float((Xn @ v).mean()))
            print(f"[layer {L:02d}] separation={gap:.3f}")
            
            if gap > best_gap:
                best_L, best_gap, best_v, best_thr = L, gap, v, thr
            
            del Xp, Xn, v; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except ValueError as e:
             print(f"[layer {L:02d}] Skipped due to error: {e}")
             continue

    if best_L is None:
        raise RuntimeError("Could not find a suitable layer for feature extraction.")

    print(f"[auto] Picked layer {best_L} (sep={best_gap:.3f})")
    return best_L, best_v, best_thr

def calibrate_resid_norm(model: HookedTransformer, hook_name: str, tokenizer, texts: List[str]) -> float:
    norms = []
    for t in texts[:8]:
        toks = model.to_tokens(t, prepend_bos=True)
        _, cache = model.run_with_cache(toks, names_filter=hook_name)
        acts = cache[hook_name][0]
        norms.append(float(acts.norm(dim=-1).mean().item()))
    return float(np.mean(norms)) if norms else 1.0

# =========================================
# WATERMARKING CORE
# =========================================

class PRFWatermarkBase:
    @staticmethod
    def _sha256_to_uint64(b: bytes) -> int:
        h = hashlib.sha256(b).digest()
        return int.from_bytes(h[:8], byteorder="big", signed=False)
    
    @staticmethod
    def _cpu_randperm(vocab_size: int, seed: int) -> torch.Tensor:
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        return torch.randperm(vocab_size, generator=g)

    def _prf_seed(self, context: torch.LongTensor, key: int, prf_last_k: int) -> int:
        k = min(prf_last_k, len(context))
        if k <= 0:
            s = f"{key}|".encode("utf-8")
        else:
            last_ids = context[-k:].cpu().tolist()
            s = f"{key}|{','.join(map(str, last_ids))}".encode("utf-8")
        return self._sha256_to_uint64(s)

    def _green_mask(self, context: torch.LongTensor, seed_key: int, vocab_size: int, gamma: float, prf_last_k: int, device: torch.device) -> torch.BoolTensor:
        seed = self._prf_seed(context, seed_key, prf_last_k)
        perm_cpu = self._cpu_randperm(vocab_size, seed)
        mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        green_ids = perm_cpu[:int(vocab_size * gamma)]
        mask[green_ids.to(device)] = True
        return mask

class PRFWatermarkProcessor(LogitsProcessor, PRFWatermarkBase):
    def __init__(self, direction_vector, initial_threshold, vocab_size, cfg: Config):
        super().__init__()
        self.v = direction_vector.to(torch.float32)
        self.threshold = float(initial_threshold)
        self.vocab_size = int(vocab_size)
        self.cfg = cfg
        self.history: List[Dict] = []
        self.forced_seed: Optional[int] = None
        self.current_activation: Optional[torch.Tensor] = None
        self.projection_history = deque(maxlen=cfg.smoothing_window)

    def reset_state(self):
        self.history = []
        self.projection_history.clear()
        self.current_activation = None
        self.forced_seed = None

    def set_current_activation(self, activation: torch.Tensor):
        self.current_activation = activation[0, -1, :].detach().to(torch.float32)

    def _determine_seed(self) -> Tuple[int, float, float]:
        if self.forced_seed is not None:
            proj = 0.0 if self.current_activation is None else float(self.current_activation @ self.v)
            return self.forced_seed, proj, proj

        if self.current_activation is None:
            return self.cfg.key_low, 0.0, 0.0

        proj = float(self.current_activation @ self.v)
        self.projection_history.append(proj)
        smoothed_proj = sum(self.projection_history) / len(self.projection_history)

        seed = self.cfg.key_high if (smoothed_proj > self.threshold) else self.cfg.key_low
        return seed, proj, smoothed_proj

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        assert input_ids.shape[0] == 1
        context = input_ids[0]
        
        seed_key, raw_proj, smoothed_proj = self._determine_seed()
        
        mask = self._green_mask(
            context, seed_key, self.vocab_size, self.cfg.gamma, self.cfg.prf_last_k, scores.device
        )

        biased = scores.to(torch.float32).clone()
        biased[0, mask] += self.cfg.delta

        self.history.append({
            "projection_raw": raw_proj, "projection_smoothed": smoothed_proj,
            "is_high": seed_key == self.cfg.key_high
        })
        return biased

class PRFWatermarkDetector(PRFWatermarkBase):
    def __init__(self, vocab_size: int, cfg: Config):
        self.vocab_size = int(vocab_size)
        self.cfg = cfg

    def calculate_z_score(self, tokens: torch.LongTensor, seed_key: int) -> float:
        T = len(tokens)
        if T <= 1: return 0.0
        
        green_count, T_eff = 0, 0
        seen_ngrams = set()
        ngram_k = self.cfg.ignore_repeated_ngram

        for i in range(1, T):
            if ngram_k > 0 and i - ngram_k + 1 >= 0:
                tup = tuple(tokens[i - ngram_k + 1: i + 1].cpu().tolist())
                if tup in seen_ngrams: continue
                seen_ngrams.add(tup)

            context, target = tokens[:i], tokens[i]
            mask = self._green_mask(
                context, seed_key, self.vocab_size, self.cfg.gamma, self.cfg.prf_last_k, tokens.device
            )
            
            if mask[target]:
                green_count += 1
            T_eff += 1

        if T_eff == 0: return 0.0
        
        gamma = self.cfg.gamma
        expected = T_eff * gamma
        var = T_eff * gamma * (1.0 - gamma)
        if var <= 0: return 0.0
        z = (green_count - expected) / math.sqrt(var)
        return float(z)

    def sliding_window_detection(self, tokens: torch.LongTensor) -> pd.DataFrame:
        T = len(tokens)
        window, stride = self.cfg.window_size, self.cfg.stride
        rows = []
        
        if T < max(window, 2):
             return pd.DataFrame(columns=["start", "z_high", "z_low"])
             
        for start in range(0, T - window + 1, stride):
            sub = tokens[start:start + window]
            rows.append({
                "start": start,
                "z_high": self.calculate_z_score(sub, self.cfg.key_high),
                "z_low": self.calculate_z_score(sub, self.cfg.key_low),
            })
        return pd.DataFrame(rows)

# =========================================
# VISUALIZATION
# =========================================

def save_plot(detections: pd.DataFrame, history: List[Dict], prompt_len: int, full_len: int, threshold: float, cfg: Config, out_png: str):
    if detections.empty or not history:
        return
        
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # --- Ground Truth Plot (Activations) ---
    projs_smoothed = [h["projection_smoothed"] for h in history]
    is_high = [h["is_high"] for h in history]
    xs_gt = list(range(prompt_len, prompt_len + len(projs_smoothed)))
    
    ax1.plot(xs_gt, projs_smoothed, linestyle="-", linewidth=2, label="Projection (Smoothed)")
    ax1.axhline(threshold, color='black', linestyle=":", label="Switch threshold")
    
    for i, flag in enumerate(is_high):
        color = 'lightcoral' if flag else 'lightblue'
        end_x = xs_gt[i+1] if i+1 < len(xs_gt) else xs_gt[i] + 1
        ax1.axvspan(xs_gt[i], end_x, color=color, alpha=0.4)
        
    ax1.set_title(f"Runtime Activations & Active Key (Coef={cfg.coef_frac:.2f})")
    ax1.set_ylabel("Persona Projection") # Updated label
    ax1.legend(loc="best")
    ax1.axvline(prompt_len, color='gray', linestyle='--', label="Generation Start")

    # --- Detection Plot (Z-scores) ---
    ax2.plot(detections["start"], detections["z_high"], label="Z-score (Positive Key)", marker='o', color='darkred')
    ax2.plot(detections["start"], detections["z_low"],  label="Z-score (Negative Key)", marker='x', color='darkblue')
    ax2.axhline(cfg.z_threshold, color='black', linestyle='--', label=f"Z={cfg.z_threshold}")
    
    ax2.set_title(f"Watermark Detection (Sliding Window={cfg.window_size})")
    ax2.set_xlabel("Token Index")
    ax2.set_ylabel("Z-score")
    ax2.legend(loc="best")
    ax2.set_xlim(0, full_len)
    ax2.axvline(prompt_len, color='gray', linestyle='--')

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)
    print(f"[viz] saved plot to {out_png}")

def save_html(tokens: torch.LongTensor, tokenizer, detector: PRFWatermarkDetector, cfg: Config, out_html: str):
    T = len(tokens)
    if T == 0: return

    z_high, z_low = [], []
    for i in range(T):
        start = max(0, i - cfg.window_size // 2)
        end = min(T, i + cfg.window_size // 2 + 1)
        
        if end - start < 5:
             z_high.append(0.0); z_low.append(0.0)
             continue

        sub = tokens[start:end]
        z_high.append(detector.calculate_z_score(sub, cfg.key_high))
        z_low.append(detector.calculate_z_score(sub, cfg.key_low))

    token_texts = [tokenizer.decode([tid], skip_special_tokens=True) for tid in tokens.tolist()]

    # HTML Generation
    html_s = """
<html><head><meta charset="utf-8"/><style>
body { font-family: sans-serif; line-height: 1.6; max-width: 900px; margin: 20px auto; }
.token { white-space: pre-wrap; }
.high { background-color: rgba(255, 105, 97, 0.6); } /* Positive (Red) */
.low  { background-color: rgba(100, 149, 237, 0.6); } /* Negative (Blue) */
.legend, .content { padding: 15px; border-radius: 8px; margin-bottom: 20px; }
.content { font-family: monospace; background: #f9f9f9; }
</style></head><body>
<div class="legend">
  <h3>Watermark Detection Visualization</h3>
  <b>Legend</b> (Z > """ + str(cfg.z_threshold) + """):<br/>
  <span class="token high">&nbsp;Positive Persona (HIGH Key)&nbsp;</span>
  <span class="token low">&nbsp;Negative Persona (LOW Key)&nbsp;</span>
</div>
<div class="content">
"""
    for i, piece in enumerate(token_texts):
        ZH, ZL = z_high[i], z_low[i]
        cls = "token"
        if ZH > cfg.z_threshold and ZL > cfg.z_threshold:
            cls += " amb"
        elif ZH > cfg.z_threshold:
            cls += " high"
        elif ZL > cfg.z_threshold:
            cls += " low"
            
        esc = html.escape(piece).replace(' ', '&nbsp;').replace('\n', '<br/>')
        html_s += f'<span class="{cls}">{esc}</span>'

    html_s += "</div></body></html>"
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_s)
    print(f"[viz] saved html to {out_html}")

# =========================================
# EXPERIMENT CORE (Dynamic Norm Scaling)
# =========================================

def setup_experiment(model: HookedTransformer, tokenizer, cfg: Config, device):
    pos_pairs, neg_pairs = build_templated_pairs(cfg.max_pairs, seed=cfg.seed)

    L, v_unit, thr = pick_best_layer(model, tokenizer, pos_pairs, neg_pairs, cfg, device)
    hook_name = f"blocks.{L}.hook_resid_pre"

    texts = [format_prompt(tokenizer, u, a) for u, a in pos_pairs[:4] + neg_pairs[:4]]
    avg_norm = calibrate_resid_norm(model, hook_name, tokenizer, texts)
    
    print(f"[setup] Layer={L}, AvgNorm≈{avg_norm:.2f}, Coef={cfg.coef_frac:.2f} (Dynamic Scaling), Thr={thr:.2f}")
    
    wm = PRFWatermarkProcessor(v_unit, thr, model.cfg.d_vocab, cfg)
    det = PRFWatermarkDetector(model.cfg.d_vocab, cfg)
    
    return hook_name, v_unit.to(model.cfg.dtype), wm, det

# --- Generation Utilities ---
def top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0: return probs
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    indices_to_remove = cumulative_probs > top_p
    indices_to_remove[..., 1:] = indices_to_remove[..., :-1].clone()
    indices_to_remove[..., 0] = 0

    probs_filtered = probs.clone()
    for b in range(probs.shape[0]):
        indices = sorted_indices[b, indices_to_remove[b]]
        probs_filtered[b, indices] = 0.0

    return probs_filtered / probs_filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def cosine_ramp_factor(step: int, apply_steps: int) -> float:
    if step < 0: return 1.0
    if step >= apply_steps: return 0.0
    x = step / float(apply_steps)
    return 0.5 * (1.0 + math.cos(math.pi * x))
# --- End Generation Utilities ---

def run_generation_with_hooks(
    model: HookedTransformer, tokenizer, prompt_text: str, hook_name: str, steer_unit_vector: torch.Tensor,
    wm: PRFWatermarkProcessor, cfg: Config, direction_sign: int = 0, force_seed: Optional[int] = None
):
    """Runs generation while steering activations (using Dynamic Norm Scaling) and applying the watermark."""
    wm.reset_state()
    wm.forced_seed = force_seed

    input_ids = model.to_tokens(prompt_text).to(model.cfg.device)
    prompt_len = input_ids.shape[1]
    gen_ids = input_ids.clone()

    # Hook function definition (Dynamic Norm Scaling)
    def steer_and_monitor(activation: torch.Tensor, hook):
        seq_len = activation.shape[1]
        step = seq_len - prompt_len 
        
        # 1. Steering
        if direction_sign != 0:
            scale = cosine_ramp_factor(step, cfg.apply_steps)
            if scale > 0.0:
                # Calculate the norm of the current activation
                current_activation_norm = activation[0, -1, :].norm()
                
                # Calculate the desired magnitude: Magnitude = coef_frac * current_activation_norm
                steering_magnitude = cfg.coef_frac * current_activation_norm
                
                # Create the steering vector
                steering_vector = (direction_sign * scale * steering_magnitude) * steer_unit_vector
                
                # Apply steering
                activation[0, -1, :] += steering_vector
        
        # 2. Monitoring
        wm.set_current_activation(activation)
        
        return activation

    fwd_hooks = [(hook_name, steer_and_monitor)]

    # Generation loop
    with model.hooks(fwd_hooks=fwd_hooks):
        for _ in range(cfg.max_new_tokens):
            with torch.no_grad():
                logits = model(gen_ids)[:, -1, :]
                logits = logits / max(1e-6, cfg.temperature)

                # Apply Watermark
                biased_logits = wm(gen_ids, logits)

                # Sampling
                probs = torch.softmax(biased_logits, dim=-1)
                probs = top_p_filter(probs, cfg.top_p)
                next_tok = torch.multinomial(probs, num_samples=1)
                gen_ids = torch.cat([gen_ids, next_tok], dim=-1)

                if tokenizer.eos_token_id is not None and int(next_tok.item()) == tokenizer.eos_token_id:
                    break

    text = decode_clean(tokenizer, gen_ids[0].tolist())
    return text, gen_ids[0], prompt_len, wm.history

def run_experiment(
    model, tokenizer, hook_name, steer_unit_vector, wm, det, cfg, name, prompt, direction_sign=0, force_seed=None
):
    print(f"\n{'='*80}\nExperiment: {name}\n{'='*80}")
    
    # Generation
    text, ids, prompt_len, history = run_generation_with_hooks(
        model, tokenizer, prompt, hook_name, steer_unit_vector, wm, cfg, direction_sign, force_seed
    )

    # Display Output
    print("\n--- Generated Text ---\n" + text[:2000] + ("\n..." if len(text) > 2000 else ""))

    # Ground Truth Metrics
    gen_len = len(history)
    if gen_len > 0:
        high_count = sum(1 for h in history if h["is_high"])
        print(f"\n[ground truth] POSITIVE Key active: {high_count}/{gen_len} ({100*high_count/gen_len:.1f}%)")

    # Watermark Detection
    det_df = det.sliding_window_detection(ids)
    if not det_df.empty:
        print(f"[detection] Max Z-scores: HIGH={det_df['z_high'].max():.2f}, LOW={det_df['z_low'].max():.2f}")

    # Visualization
    out_png = os.path.join(cfg.results_dir, f"Plot_{name}.png")
    out_html = os.path.join(cfg.results_dir, f"HTML_{name}.html")
    save_plot(det_df, history, prompt_len, len(ids), wm.threshold, cfg, out_png)
    save_html(ids, tokenizer, det, cfg, out_html)
    
    return history

# =========================================
# MAIN EXECUTION
# =========================================

def load_model(cfg: Config, device, dtype):
    print(f"[load] Loading {cfg.model_name} on {device} with dtype {dtype}...")
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=dtype, attn_implementation="eager", device_map=device
    )

    # fold_value_biases=False is crucial for Gemma compatibility
    model = HookedTransformer.from_pretrained(
        cfg.model_name, hf_model=hf_model, device=device, tokenizer=tokenizer,
        dtype=dtype, fold_ln=False, fold_value_biases=False, 
        center_writing_weights=False, center_unembed=False,
    )
    model.eval()
    torch.set_grad_enabled(False)
    print("[load] Model ready.")
    return model, tokenizer

def main():
    cfg = Config()
    set_seed(cfg.seed)
    ensure_dir(cfg.results_dir)
    device, dtype = get_device_and_dtype()

    # Load Model
    model, tokenizer = load_model(cfg, device, dtype)

    # Setup Experiment (Now using Persona-based contrast pairs)
    print("\n--- Setting up experiment (Persona Steering) ---")
    try:
        hook_name, steer_unit_vector, wm, det = setup_experiment(model, tokenizer, cfg, device)
    except RuntimeError as e:
        print(f"\n[ERROR] Setup failed: {e}")
        return

    # Define Prompt
    user_task = "Write a blog post analyzing the impact of social media on society. Be passionate and expressive in your writing."
    prompt = format_prompt(tokenizer, user_task)

    # =================== EXPERIMENTS ===================
    
    zero_vec = torch.zeros_like(steer_unit_vector)

    # A) Baseline (no steering)
    historyA = run_experiment(
        model, tokenizer, hook_name, zero_vec, wm, det, cfg, "A_Baseline", prompt, direction_sign=0
    )

    # Optional: Recalibrate threshold
    xs = np.array([h["projection_smoothed"] for h in historyA], dtype=np.float32)
    if len(xs) > 5:
        new_thr = float(xs.mean() + cfg.recalibrate_k * xs.std())
        print(f"[calib] Recalibrated threshold: {wm.threshold:.2f} -> {new_thr:.2f}")
        wm.threshold = new_thr

    # B) Steer Negative (Induce Pessimistic Persona)
    run_experiment(
        model, tokenizer, hook_name, steer_unit_vector, wm, det, cfg, f"B_Steer_Negative_c{cfg.coef_frac:.2f}", prompt, direction_sign=-1
    )

    # C) Steer Positive (Induce Optimistic Persona)
    run_experiment(
        model, tokenizer, hook_name, steer_unit_vector, wm, det, cfg, f"C_Steer_Positive_c{cfg.coef_frac:.2f}", prompt, direction_sign=+1
    )
    
    # D) Control: Force HIGH key
    run_experiment(
        model, tokenizer, hook_name, zero_vec, wm, det, cfg, "D_Control_Force_High", prompt, direction_sign=0, force_seed=cfg.key_high
    )

    # E) Control: Force LOW key
    run_experiment(
        model, tokenizer, hook_name, zero_vec, wm, det, cfg, "E_Control_Force_Low", prompt, direction_sign=0, force_seed=cfg.key_low
    )

    print(f"\nAll experiments completed. See results in: {cfg.results_dir}")

if __name__ == "__main__":
    main()
