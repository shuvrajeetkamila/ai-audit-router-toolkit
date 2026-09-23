# ai-audit-router-toolkit

Three small, dependency-light command-line tools:

- **`claim_validator.py`** — checks whether a text report's claims are
  consistent with a JSON results file, using rules you define (no fixed
  schema).
- **`text_router.py`** — routes text to one or more categories using
  regex rules and/or a from-scratch (numpy-only) trainable classifier.
- **`license_registry.py`** — tracks third-party license terms and checks
  whether a set of components can be legally combined and redistributed.

## Install

```bash
pip install -r requirements.txt   # only numpy, and only text_router.py needs it
```

## Try it (uses the files in `examples/`)

```bash
# claim_validator: this should exit 1 and print 2 violations
python3 claim_validator.py --data examples/data.json \
  --report examples/report_bad.md --rules examples/rules.json

# this one should exit 0, all checks pass
python3 claim_validator.py --data examples/data.json \
  --report examples/report_good.md --rules examples/rules.json

# text_router: train a small model, then route a sentence
python3 text_router.py train --config examples/router_config.json \
  --data examples/router_train.json --save examples/router_model.npz
python3 text_router.py route --config examples/router_config.json \
  --model examples/router_model.npz --text "the app crashes on launch"

# license_registry: check whether three components can be redistributed together
python3 license_registry.py check --db examples/licenses.json \
  --components examples/components.json
```

Full usage and the rules/config file formats are documented in each script's
own docstring — run `python3 <script>.py --help`, or just open the file.

## What each tool does — and doesn't — do

- `claim_validator.py` flags mismatches between prose and data and writes a
  report of what it found. It does not edit or rewrite your report text.
- `text_router.py`'s learned component is a lightweight bag-of-words
  logistic regression — transparent and fast, but not a neural embedding
  model. Good for routing between a handful of well-separated categories;
  not a substitute for a real classifier on hard or adversarial input.
- `license_registry.py` checks the license you tell it a component uses.
  It does not inspect binaries, weights, or hashes to verify that claim —
  it's a bookkeeping and compatibility-logic tool, not a forensic scanner.

## License

Dual-licensed under your choice of:

- MIT (`LICENSE-MIT`)
- Apache License, Version 2.0 (`LICENSE-APACHE`)

at your option. This "MIT OR Apache-2.0" pattern is standard practice for
developer tools (used across the Rust ecosystem, among others) — it gives
users maximum flexibility while Apache-2.0's explicit patent grant and
termination-on-litigation clause gives you some protection that MIT alone
doesn't provide.

One clarification on a claim that sometimes circulates about this: Apache-2.0's
patent clause does not mean someone could patent your own code's logic and
then sue you over it — a patent requires the applicant's own novel invention,
and your prior public code would itself count as prior art undermining such a
patent. What Apache-2.0's clause actually does is defensive in the other
direction: if *you* get sued by a user of your code over an unrelated patent
claim, that user's license to your software automatically terminates. It's a
reasonable choice for exactly the reason people usually reach for it (patent
hygiene for technical, algorithmic code), just not for the "someone will
patent-troll your own invention" reasoning. None of this is legal advice —
for anything beyond picking a standard open-source license, talk to an
actual lawyer.
