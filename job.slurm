#!/bin/bash
#SBATCH -J st3-persona-only
#SBATCH -A egr
#SBATCH -p gpu
#SBATCH --qos=short
#SBATCH --gres=gpu:40g:1
#SBATCH --ntasks=1
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o slurm-st3-%j.out
#SBATCH -e slurm-st3-%j.err

set -euo pipefail

source ~/AgentSim/.venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.9

# Common env (short runs, language on, judge not needed for ST3)
export USE_LANGUAGE="1"
export PLANNER_TOPK="3"
export N_GAMES="2"            # keep tiny; we only need a couple of days of chat
export LLM_MODEL_ID="mistralai/Mistral-7B-Instruct-v0.2"   # or your GPU-friendly model

echo "================ ST3A: Personality randomization with TEMPLATE BANDIT ================"
export SPEAKER_ENABLED="1"
export LLM_SPEAKER="0"

python - <<'PY'
import os, statistics
from collections import Counter, defaultdict
import torch

from sim import simulate_game

# Run a tiny non-visual game; sim returns (rollout, meta) and meta includes agents
rollouts, meta = simulate_game(visual=False)
agents = meta.get("agents", [])
print(f"[ST3A] rounds: {meta.get('rounds')} agents: {len(agents)}")

TEMPLATES = [
    "Accuse {target}",
    "Defend {ally}",
    "Ask {target} a question",
    "Express uncertainty",
    "Propose vote on {target}",
]
# Simple classifier: map a template prefix to a category name
def classify_template(msg: str) -> str:
    s = (msg or "").strip().lower()
    if s.startswith("accuse "): return "accuse"
    if s.startswith("defend "): return "defend"
    if s.startswith("ask "): return "question"
    if s.startswith("express uncertainty"): return "hedge"
    if s.startswith("propose vote"): return "vote"
    return "other"

per_agent_counts = {}
all_cats = Counter()
for ag in agents:
    cats = Counter()
    for _, m in list(getattr(ag, "message_memory", [])):
        c = classify_template(m)
        cats[c] += 1
        all_cats[c] += 1
    per_agent_counts[ag.name] = cats

distinct = sum(1 for k,v in all_cats.items() if v>0 and k!="other")
print("[ST3A] Category totals:", dict(all_cats))
print("[ST3A] Distinct categories (excl 'other'):", distinct)

# Check variance across agents on accuse vs defend usage as a quick persona signal
def per_agent_ratio(agent_counts, key):
    num = agent_counts.get(key,0)
    den = sum(agent_counts.values()) or 1
    return num/den

accuse_ratios = [per_agent_ratio(c,"accuse") for c in per_agent_counts.values()]
defend_ratios = [per_agent_ratio(c,"defend") for c in per_agent_counts.values()]
acc_var = statistics.pvariance(accuse_ratios) if len(accuse_ratios)>1 else 0.0
def_var = statistics.pvariance(defend_ratios) if len(defend_ratios)>1 else 0.0

print(f"[ST3A] accuse_ratio var={acc_var:.4f}  defend_ratio var={def_var:.4f}")

passed = (distinct > 1) and (acc_var > 0.0 or def_var > 0.0)
print("[ST3A] PASS" if passed else "[ST3A] FAIL")
PY

echo "================ ST3B: Personality randomization with LLM LOGIT MOUTHPIECE ================"
export SPEAKER_ENABLED="0"
export LLM_SPEAKER="1"
# Optional: stronger debug from the logit-bias head to see per-round bias vectors
export LLM_SPK_DEBUG="1"

python - <<'PY'
import os, re, statistics
from collections import Counter
import torch

from sim import simulate_game
# Import lexicon categories to classify LLM utterances
from speaker_llm import DEFAULT_LEXICON

rollouts, meta = simulate_game(visual=False)
agents = meta.get("agents", [])
print(f"[ST3B] rounds: {meta.get('rounds')} agents: {len(agents)}")

# Make a fast word->cat map (first match wins)
word_to_cat = {}
for cat, words in DEFAULT_LEXICON.items():
    for w in words:
        word_to_cat[w.lower()] = cat

def classify_llm(msg: str) -> str:
    s = (msg or "").lower()
    # very simple: if any seed word is present, tag that category
    for w, cat in word_to_cat.items():
        if w == "?":
            if "?" in s: return "question"
        elif w in s:
            return cat
    return "other"

per_agent_counts = {}
all_cats = Counter()
for ag in agents:
    cats = Counter()
    for _, m in list(getattr(ag, "message_memory", [])):
        c = classify_llm(m)
        cats[c] += 1
        all_cats[c] += 1
    per_agent_counts[ag.name] = cats

distinct = sum(1 for k,v in all_cats.items() if v>0 and k!="other")
print("[ST3B] Category totals:", dict(all_cats))
print("[ST3B] Distinct categories (excl 'other'):", distinct)

def per_agent_ratio(agent_counts, key):
    num = agent_counts.get(key,0)
    den = sum(agent_counts.values()) or 1
    return num/den

accuse_ratios = [per_agent_ratio(c,"accuse") for c in per_agent_counts.values()]
defend_ratios = [per_agent_ratio(c,"defend") for c in per_agent_counts.values()]
hedge_ratios  = [per_agent_ratio(c,"hedge")  for c in per_agent_counts.values()]
acc_var = statistics.pvariance(accuse_ratios) if len(accuse_ratios)>1 else 0.0
def_var = statistics.pvariance(defend_ratios) if len(defend_ratios)>1 else 0.0
hed_var = statistics.pvariance(hedge_ratios)  if len(hedge_ratios)>1  else 0.0
print(f"[ST3B] accuse_ratio var={acc_var:.4f}  defend_ratio var={def_var:.4f}  hedge_ratio var={hed_var:.4f}")

passed = (distinct > 1) and (acc_var > 0.0 or def_var > 0.0 or hed_var > 0.0)
print("[ST3B] PASS" if passed else "[ST3B] FAIL")
PY

echo "================ ST3 SUMMARY ================"
grep -E "^\[ST3[AB]\] (PASS|FAIL)$" slurm-st3-*.out || true

