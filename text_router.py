#!/usr/bin/env python3
"""text_router.py — generic keyword + learned-classifier router for text input.

Generalized from a project-specific router. The experts, keyword rules, and
training examples used to be hardcoded in the file; they now live in a JSON
config, so the same engine can route to any set of categories (support
tiers, content topics, model endpoints, team queues, whatever "experts"
means for your use case) without touching the code.

Two routing strategies, usable independently or combined:
  1. Rule-based: regex patterns per category, no training needed.
  2. Learned: a from-scratch bag-of-words logistic regression (numpy only,
     no external ML dependency) trained on labeled examples you provide.

This is a lightweight, auditable classifier suitable for routing between a
handful of well-separated categories on small-to-medium text. It is not a
neural embedding model and won't compete with a real transformer-based
classifier on nuanced or adversarial inputs -- pick it for its simplicity,
zero dependencies, and speed, not for state-of-the-art accuracy.

Usage:
    # Rule-based only
    python text_router.py route --config config.json --text "your input here"

    # Train the learned router from a labeled examples file and save it
    python text_router.py train --config config.json --data train_examples.json --save model.npz

    # Route using a trained model
    python text_router.py route --config config.json --model model.npz --text "your input here"

Config file format (JSON):
{
  "experts": ["billing", "technical", "sales"],
  "rules": {
    "billing":   ["\\b(invoice|charge|refund|payment)\\b"],
    "technical": ["\\b(error|bug|crash|not working)\\b"],
    "sales":     ["\\b(pricing|quote|demo|upgrade)\\b"]
  }
}

Training data file format (JSON): a list of [text, [labels]] pairs, e.g.
[
  ["my invoice looks wrong", ["billing"]],
  ["the app crashes on launch", ["technical"]],
  ["can I get a demo and pricing", ["sales", "billing"]]
]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Dict, List, Tuple

import numpy as np


# --------------------------------------------------------------------------
# Rule-based routing
# --------------------------------------------------------------------------

def rule_route(text: str, rules: Dict[str, List[str]]) -> Dict[str, float]:
    """Score each expert by fraction of its patterns that match, scaled to [0, 1]."""
    scores = {}
    for expert, patterns in rules.items():
        hits = sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))
        scores[expert] = min(1.0, hits / max(1, len(patterns)) * 2)
    return scores


# --------------------------------------------------------------------------
# Learned routing: bag-of-words + multinomial logistic regression
# --------------------------------------------------------------------------

class LearnedRouter:
    """A minimal, dependency-free multi-label text classifier.

    Not a substitute for a real embedding-based classifier on hard cases,
    but transparent, fast, and easy to retrain on a laptop -- a reasonable
    default for routing between a handful of well-separated categories.
    """

    def __init__(self, experts: List[str]):
        self.experts = list(experts)
        self.vocab: Dict[str, int] = {}
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9']+", text.lower())

    def _build_vocab(self, texts: List[str]) -> None:
        for t in texts:
            for w in self._tokenize(t):
                if w not in self.vocab:
                    self.vocab[w] = len(self.vocab)

    def _featurize(self, texts: List[str]) -> np.ndarray:
        X = np.zeros((len(texts), len(self.vocab)), np.float32)
        for i, t in enumerate(texts):
            for w in self._tokenize(t):
                if w in self.vocab:
                    X[i, self.vocab[w]] += 1.0
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return X / norms

    def fit(self, texts: List[str], labels: List[List[str]], epochs: int = 300, lr: float = 0.5, seed: int = 0) -> None:
        self._build_vocab(texts)
        X = self._featurize(texts)
        Y = np.zeros((len(texts), len(self.experts)), np.float32)
        for i, labs in enumerate(labels):
            for l in labs:
                if l in self.experts:
                    Y[i, self.experts.index(l)] = 1.0
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.01, (X.shape[1], len(self.experts))).astype(np.float32)
        self.b = np.zeros(len(self.experts), np.float32)
        n = len(texts)
        for _ in range(epochs):
            Z = X @ self.W + self.b
            P = 1 / (1 + np.exp(-Z))
            G = (P - Y) / n
            self.W -= lr * (X.T @ G)
            self.b -= lr * G.sum(0)

    def route(self, text: str, threshold: float = 0.35, top_k: int = 3) -> Tuple[List[str], Dict[str, float]]:
        if self.W is None:
            raise RuntimeError("Router has not been trained or loaded yet.")
        x = np.zeros((1, len(self.vocab)), np.float32)
        for w in self._tokenize(text):
            if w in self.vocab:
                x[0, self.vocab[w]] += 1.0
        norm = np.linalg.norm(x)
        if norm > 0:
            x /= norm
        Z = (x @ self.W + self.b)[0]
        P = 1 / (1 + np.exp(-Z))
        scores = {self.experts[i]: float(P[i]) for i in range(len(self.experts))}
        chosen = [e for e, s in scores.items() if s >= threshold]
        if not chosen:
            chosen = [max(scores, key=scores.get)]
        chosen = sorted(chosen, key=lambda e: -scores[e])[:top_k]
        return chosen, scores

    def save(self, path: str) -> None:
        vocab_words = sorted(self.vocab, key=self.vocab.get)
        np.savez(path, W=self.W, b=self.b, experts=np.array(self.experts),
                 vocab=np.array(vocab_words))

    @classmethod
    def load(cls, path: str) -> "LearnedRouter":
        d = np.load(path, allow_pickle=False)
        experts = [str(e) for e in d["experts"]]
        router = cls(experts)
        router.vocab = {str(w): i for i, w in enumerate(d["vocab"])}
        router.W = d["W"]
        router.b = d["b"]
        return router


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def load_config(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def cmd_route(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    rule_scores = rule_route(args.text, config.get("rules", {}))
    print("Rule-based scores:", json.dumps(rule_scores, indent=2))

    if args.model:
        router = LearnedRouter.load(args.model)
        chosen, learned_scores = router.route(args.text, threshold=args.threshold, top_k=args.top_k)
        print("Learned model scores:", json.dumps(learned_scores, indent=2))
        print("Routed to:", chosen)
    else:
        best = max(rule_scores, key=rule_scores.get) if rule_scores else None
        print("Routed to (rule-based only):", best)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    with open(args.data) as f:
        raw = json.load(f)
    texts = [t for t, _ in raw]
    labels = [labs for _, labs in raw]

    router = LearnedRouter(config["experts"])
    router.fit(texts, labels, epochs=args.epochs, lr=args.lr)
    router.save(args.save)
    print(f"Trained on {len(texts)} examples across {len(config['experts'])} experts.")
    print(f"Saved model to {args.save}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Route text to one or more categories.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_route = sub.add_parser("route", help="Route a piece of text")
    p_route.add_argument("--config", required=True)
    p_route.add_argument("--text", required=True)
    p_route.add_argument("--model", default=None, help="Path to a trained .npz model (optional)")
    p_route.add_argument("--threshold", type=float, default=0.35)
    p_route.add_argument("--top-k", type=int, default=3)
    p_route.set_defaults(func=cmd_route)

    p_train = sub.add_parser("train", help="Train the learned router")
    p_train.add_argument("--config", required=True)
    p_train.add_argument("--data", required=True, help="JSON file of [text, [labels]] pairs")
    p_train.add_argument("--save", required=True, help="Where to save the trained .npz model")
    p_train.add_argument("--epochs", type=int, default=300)
    p_train.add_argument("--lr", type=float, default=0.5)
    p_train.set_defaults(func=cmd_train)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
