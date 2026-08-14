# SevenNet-l3i5 MD baseline

## Permanent baseline

The optimization reference is:

- `stage=baseline`, `backend=eager`;
- `SevenNetCalculator` loaded from the original `checkpoint_l3i5.pth`;
- ordinary e3nn tensor products, with cuEquivariance, FlashTP and
  OpenEquivariance disabled;
- the upstream ASE path: ASE NVT integration, a CPU matscipy neighbor list
  (ASE fallback if matscipy is unavailable), CPU-to-GPU graph copies, and a
  float32 SevenNet model;
- TF32 disabled and float32 matmul precision set to `highest` by the route.

The checkpoint is exactly the artifact referenced by Matbench Discovery's
`sevennet-l3i5.yml`: Git blob
`81759f4668c877854e0b537c4de955899cb5e1ae`, 14,345,754 bytes, architecture
`lmax=3`, five interaction layers, cutoff 5.0 Angstrom.  It predicts energy,
forces and cell stress.  Atomic virials are not needed for the NVT benchmark.

SevenNet's model arithmetic is float32.  The ASE positions, momenta, cell and
thermostat state remain float64.  Matbench's global `--dtype float64` is not
forwarded to `SevenNetCalculator` in the official calculator registry either.

## Existing upstream accelerators

`backend=cueq`, `backend=flash`, and `backend=oeq` remain available as
explicit upstream-accelerator comparisons.  They replace tensor-product
kernels only; they do not make the ASE MD loop or neighbor construction GPU
resident.  Do not report any of them as the permanent eager baseline or as a
completed project Opt1.

- cuEquivariance: supported optional TP backend.
- FlashTP: fused sparsity-aware TP backend; upstream reports up to about 4x TP
  inference acceleration.
- OpenEquivariance: experimental optional TP backend.
- TorchSim: optional all-GPU model/neighbor-list interface and the natural
  starting point to investigate Opt1, but it is not used by this baseline.
- deployed serial/parallel `.pt` models: LAMMPS execution artifacts, not inputs
  to the current ASE calculator.  They are a separate engine comparison.
- `torch.compile`: no supported SevenNet ASE-calculator backend is exposed by
  this revision, so it is not part of the baseline matrix.

## Server environment

```bash
conda activate md_opt
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=/public-data/fushibo:${PYTHONPATH:-}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
unset SEVENNET_ENABLE_CUEQ SEVENNET_ENABLE_FLASH SEVENNET_ENABLE_OEQ
```

The route also isolates the three `SEVENNET_ENABLE_*` variables while building
the calculator, so a stale shell variable cannot silently alter the baseline.

## Cu and H2O baseline

```bash
cd /public-data/fushibo
CUDA_VISIBLE_DEVICES=0 python run_md_test.py \
  --model sevennet \
  --model-backend sevenn.md_route:run_md \
  --model-path /public-data/fushibo/SevenNet/sevenn/pretrained_potentials/SevenNet_l3i5/checkpoint_l3i5.pth \
  --stage baseline \
  --backend eager \
  --structure-path /public-data/fushibo/md_test_data \
  --temperature-k 300 800 \
  --ensemble nvt \
  --integrator berendsen \
  --steps 1000 \
  --timestep-fs 1.0 \
  --thermostat-time-fs 100 \
  --warmup-steps 3 \
  --observation-step 1 50 100 1000 \
  --statistics \
  --timing-pass \
  --timing-repeats 5 \
  --device cuda:0 \
  --dtype float32 \
  --output /public-data/fushibo/results/sevennet-l3i5/baseline-cu-h2o
```

## Matbench DynaMat baseline

Run the formal 17-system protocol without warmup.  The command uses the
published l3i5 YAML for public-metric comparison.

```bash
cd /public-data/fushibo
CUDA_VISIBLE_DEVICES=0 python run_md_matbench.py \
  --model sevennet \
  --model-backend sevenn.md_route:run_md \
  --model-path /public-data/fushibo/SevenNet/sevenn/pretrained_potentials/SevenNet_l3i5/checkpoint_l3i5.pth \
  --stage baseline \
  --backend eager \
  --structure-path /public-data/fushibo/matbench-discovery-data/md/2026-06-29-dynamat-v1.0-reference-trajectories.h5 \
  --matbench-repo /public-data/fushibo/matbench-discovery \
  --leaderboard-model-yaml /public-data/fushibo/matbench-discovery/models/sevennet/sevennet-l3i5.yml \
  --ensemble nvt \
  --integrator nose_hoover_chain \
  --steps 80000 \
  --timestep-fs 0.25 \
  --thermostat-time-fs 25 \
  --warmup-steps 0 \
  --record-interval 10 \
  --seed 0 \
  --device cuda:0 \
  --dtype float64 \
  --statistics \
  --output /public-data/fushibo/results/sevennet-l3i5/baseline-matbench
```

The YAML's published run used SevenNet 0.10.3, PyTorch 2.2.1 and an H200;
the shared environment uses current SevenNet, PyTorch 2.11/CUDA 12.6 and an
H100.  Therefore hardware time is not expected to match, and chaotic MD plus
version drift can move individual trajectory statistics.  Compare the six
public aggregate metrics with tolerances and retain the generated per-system
CSV/trajectory artifacts.  Public DynaMat does not contain the maintainer-only
energy/force labels, so the published energy and force RMSE cannot be recomputed
from the public HDF5.
