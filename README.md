# Planetary Search

This repository contains a helper search tool that is not part of the main Rust
dashboard binary.

## `planetary_search.py`

`planetary_search.py` searches kinematic layouts for a 3-planetary,
5-friction-element automatic transmission. It models each simple planetary
gearset with the lever equation:

```text
omega_s + rho * omega_r - (1 + rho) * omega_c = 0
```

where `rho` is the ring-to-sun tooth ratio. The script searches permanent
member connections, input/output assignments, friction-element placements,
feasible tooth counts, and two-element apply states. It scores layouts against
target forward ratios and an optional reverse ratio.

Commands below assume they are run from this `planetary_search` directory.

### Requirements

Install the Python dependencies from:

```bash
python3 -m pip install -r requirements-planetary.txt
```

The CPU path requires NumPy. The CUDA path requires PyTorch with CUDA support
and an NVIDIA GPU.

Run the embedded checks with:

```bash
python3 planetary_search.py --self-test
```

### How It Works

The script keeps combinatorial structure generation on the CPU:

- Generates feasible simple planetary tooth options from sun, planet, ring,
  and carrier planet-count constraints.
- Generates permanent topology partitions by connecting planetary members.
- Assigns distinct input and output components.
- Generates candidate brake and clutch element sets.
- Evaluates every two-element apply state for each layout.

For each candidate layout, the solver:

1. Builds the three planetary equation rows from the selected `rho` values.
2. Adds the fixed input speed constraint.
3. Adds two brake or clutch constraints for the apply state.
4. Solves the resulting linear system.
5. Converts output speed to ratio with `ratio = 1.0 / omega_output`.
6. Rejects singular, inconsistent, non-finite, near-zero-output, or excessive
   ratio states.
7. Sorts unique positive ratios as forward gears and unique negative ratios as
   reverse candidates.
8. Scores forward gear sequences by mean squared error against the target
   ratios.
9. Adds penalties for excess double-transition shifts and optional reverse
   error.

With `--backend cuda`, topology and element generation still run on the CPU,
but tooth-triple evaluation is batched on the GPU. PyTorch builds batched
linear systems for the 10 apply states, solves them, filters ratios, computes
scores, and sends only top tooth-triple finalists back to the CPU. Finalists
are rescored by the CPU oracle before being reported.

### Basic Usage

Run a bounded CPU search:

```bash
python3 planetary_search.py \
  --backend cpu \
  --topology-limit 5 \
  --element-limit-per-topology 20 \
  --tooth-combination-limit 1000
```

Run on CUDA when PyTorch can see an NVIDIA GPU:

```bash
python3 planetary_search.py \
  --backend cuda \
  --device cuda:0 \
  --batch-size 32768 \
  --tooth-combination-limit 100000
```

Emit JSON instead of the text report:

```bash
python3 planetary_search.py --json --top 5
```

Search custom target ratios. Provide six or seven positive forward ratios,
optionally followed by a negative reverse ratio:

```bash
python3 planetary_search.py \
  --targets 3.25,2.23,1.61,1.24,1.0,0.63,-2.95
```

Run a six-speed target set:

```bash
python3 planetary_search.py \
  --targets 4.0,2.5,1.6,1.2,1.0,0.75,-3.0
```

### Search Controls

Use these limits to keep runs bounded:

- `--topology-limit N`: stop after `N` topology/input/output assignments.
- `--element-limit-per-topology N`: evaluate at most `N` element sets per
  topology.
- `--candidate-limit N`: stop after `N` topology plus element-set candidates.
- `--tooth-combination-limit N`: evaluate at most `N` tooth triples per
  candidate layout.
- `--top N`: keep the best `N` results.
- `--progress-every N`: print progress to stderr every `N` candidates.

To change tooth-count coverage:

- `--standing-ratio-bounds MIN MAX`: limit generated ring/sun ratios.
- `--include-equivalent-teeth`: keep multiple tooth sets with the same reduced
  `rho`; by default only one representative is kept.

To change topology coverage:

- `--permanent-links 3`, `--permanent-links 4`, or `--permanent-links 3,4`.
- `--allow-internal-permanent`: allow permanent links within the same gearset.
- `--include-output-brakes`: allow brakes on the output component.

### Scoring Controls

- `--strict-single-transition`: reject forward sequences where adjacent gears
  do not share one applied element.
- `--max-double-transitions N`: allow `N` double-transition shifts before
  penalties apply.
- `--transition-penalty VALUE`: penalty per excess double transition.
- `--reverse-weight VALUE`: add weighted reverse squared error to the score.
- `--allow-missing-reverse`: keep candidates that do not produce a reverse
  state.
- `--ratio-abs-limit VALUE`: reject ratios with absolute value above this
  limit.
- `--ratio-tolerance VALUE`: relative tolerance for treating ratios as
  duplicates.

### CUDA And Sampling

`--backend auto` uses CUDA when PyTorch reports an available CUDA device;
otherwise it falls back to CPU. Use `--backend cuda` to require CUDA and fail if
it is unavailable.

CUDA-specific controls:

- `--device cuda` or `--device cuda:0`: select the PyTorch CUDA device.
- `--batch-size N`: number of tooth triples per GPU batch.
- `--gpu-refine-top N`: number of GPU-selected finalists to rescore on CPU per
  layout.
- `--sampling-mode exhaustive`: evaluate tooth triples in deterministic order.
- `--sampling-mode random`: sample random tooth triples. Requires
  `--tooth-combination-limit`.
- `--sampling-mode stratified-rho`: sample evenly across the flattened tooth
  index space. Requires `--tooth-combination-limit`.
- `--random-seed N`: seed for random sampling.
- `--probe-tooth-triples N`: run a small CPU probe first and skip layouts that
  produce no result in the probe window.

### Checkpointing

Long searches can be resumed with JSON checkpoints:

```bash
python3 planetary_search.py \
  --backend cuda \
  --checkpoint /tmp/planetary-search.json \
  --checkpoint-every 25
```

Resume from a checkpoint:

```bash
python3 planetary_search.py \
  --resume /tmp/planetary-search.json \
  --checkpoint /tmp/planetary-search.json
```

The checkpoint stores the next topology index, next element-set index, current
statistics, and current top results.

### Output

The text report includes:

- Candidate and topology counts.
- Tooth-option counts for each gearset.
- Backend and CUDA batch statistics when applicable.
- Ranked layouts with score, MSE, tooth counts, permanent connections,
  input/output components, friction-element placements, apply chart, reverse
  state, and double-transition shifts.

The JSON report contains the same result data in machine-readable form.
