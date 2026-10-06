# Grasping in Clutter: an evaluation protocol

Code, models and analysis for **[TODO: paper title]** ([TODO: venue, year]).

The protocol measures how a grasping controller's performance changes as the clutter around a
target object increases. Every object is grasped in four conditions:

| Condition | Scene around the target |
|---|---|
| `isolated` | target alone on the table |
| `C0_easy` | 2–4 neighbours, 10–20 cm apart |
| `C1_medium` | 4–6 neighbours, 8–12 cm apart |
| `C2_hard` | 8–10 neighbours, 3–8 cm apart |

Cluttered scenes are generated from these rules and then **confirmed by a clutter-level
classifier**, a multimodal CNN on the scene image and the target's position, corrected by the
neighbour count. Only scenes the classifier agrees with are used. Each object and condition is
run for 100 trials, and the analysis compares grasp difficulty across conditions and controllers.

The protocol does not depend on a particular controller. The repository includes:

- **Per-object PPO policies for the Contactile hand**: the reference controller used in the paper.
- **A Franka Panda with scripted IK and GG-CNN grasp poses**: needs no training, so it's the quickest way to run the benchmark end to end.
- **A hook for your own controller.** See [Use your own controller](#use-your-own-controller).

## Repository layout

```
├── source/clutter_grasp/clutter_grasp/   Python package (pip install -e source/clutter_grasp)
│   ├── protocol/                         clutter levels + clutter-level classifier (no Isaac Sim dependency)
│   ├── envs/                             Isaac Lab environments (Contactile benchmark, Panda IK benchmark, data collection)
│   └── assets/                           Contactile hand model, 24 EGAD benchmark objects
├── scripts/
│   ├── benchmark/                        run_benchmark.{py,sh} (Contactile hand), run_panda_ik.{py,sh} (Panda)
│   └── download_weights.sh
├── classifier/                           build the classifier: scene generation (MuJoCo or Isaac Sim) → PCA + K-Means → training
├── analysis/                             analyze.py: paper figures from sim results + real-world trials; example_data/ = full paper results
├── weights/   (downloaded)               classifier, PPO policies, GG-CNN, SAM
└── results/   (generated)                results/benchmark/<controller>/single_<object>[_<condition>]_<timestamp>/
```

## Installation

Requires [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) (tested with Isaac Sim 5.0 / Isaac Lab 2.2 in the
official container, Python 3.11, PyTorch 2.7).

```bash
git clone https://github.com/JinW25/ObjectToScenes_EvaluateGraspInCluttered.git
cd ObjectToScenes_EvaluateGraspInCluttered

# inside the Isaac Lab Python environment / container
/isaac-sim/python.sh -m pip install -e source/clutter_grasp
/isaac-sim/python.sh -m pip install stable-baselines3 segment-anything opencv-python

bash scripts/download_weights.sh          # classifier, PPO policies, GG-CNN, SAM  (~2.2 GB)
```

`analysis/` and `classifier/` have their own `requirements.txt` and run in a normal Python environment.

## Quick start

```bash
# 1. Panda baseline, one object, isolated + medium clutter (no trained policy needed)
/isaac-sim/python.sh scripts/benchmark/run_panda_ik.py --target_object A24_0 --isolated --num_trials 5 --enable_cameras --headless
/isaac-sim/python.sh scripts/benchmark/run_panda_ik.py --target_object A24_0 --target_complexity C1_medium --num_trials 5 --enable_cameras --headless

# 2. Contactile hand with its per-object PPO policy
/isaac-sim/python.sh scripts/benchmark/run_benchmark.py --target_object A24_0 --target_complexity C1_medium --num_trials 5 --headless

# 3. Full protocol: every object × {isolated, C0, C1, C2} × 100 trials
bash scripts/benchmark/run_benchmark.sh
bash scripts/benchmark/run_panda_ik.sh

# 4. Figures (one script for simulated and real-world results)
python analysis/analyze.py --controller ppo=results/benchmark/ppo --controller panda_ik=results/benchmark/panda_ik \
    --reference ppo --output_dir figures
```

Scripts work out paths relative to their own location, so you can launch them from any folder.

## Results format

Each run writes `results/benchmark/<controller>/single_<object>[_<condition>]_<timestamp>/`:

```
config.json                 run settings
results/results.json        per-trial results (read by analysis/)
classifier_images/          the confirmed scene image (cluttered conditions)
```

`results.json` → `trial_results[]` has, per trial: `success`, `picking_time` (s), `drops`, `hand_respawns`,
`steps`, `reason`, and `chaos_metrics` (how far the target moved: `target_distance`, `target_normalized_distance`).
To evaluate a controller outside this simulator, write the same files and the analysis will work unchanged.

## Use your own controller

`run_benchmark.py` accepts any controller that maps the environment observation to an action:

```python
# my_controller.py
class MyController:
    def __init__(self, env):           # env: the unwrapped BenchmarkEnv
        self.env = env
    def act(self, obs):                # obs: (num_envs, 234) array -> actions: (num_envs, 12) in [-1, 1]
        ...

def make_controller(env):
    return MyController(env)
```

```bash
/isaac-sim/python.sh scripts/benchmark/run_benchmark.py --target_object A24_0 --target_complexity C2_hard \
    --controller my_controller:make_controller --controller_name mine --headless
```

With `--controller`, no PPO policy is loaded and every object in `assets/objects/` can be the target. For a
different robot, reuse `clutter_grasp.protocol` (clutter levels and classifier, independent of Isaac Sim)
in your own scene generator, the way `envs/benchmark_env.py` does.

## The clutter classifier

`clutter_grasp.protocol.classifier` loads the released classifier and classifies any top-down scene image
in which the target is marked with a green box, from simulation or a real camera:

```python
import torch
from clutter_grasp.protocol.classifier import load_classifier_model, classify_scene

model = load_classifier_model("weights/classifier", torch.device("cuda"))
level = classify_scene(model, "scene.png", neighbor_count=3, device=torch.device("cuda"))   # 0, 1 or 2
```

[classifier/](classifier/README.md) contains the full pipeline used to build it: generating labelled
cluttered scenes (MuJoCo or Isaac Sim), stratifying them into clutter levels with PCA + K-Means, and
training the classifier.

## Analysis

`analysis/analyze.py` is a single script for simulated and real-world results. It reads only the
`results.json` files of each controller folder (`--controller NAME=DIR`) and, optionally, the real-world
log (`--csv clutter_compile.csv`). `analysis/example_data/` contains the paper's full results (24 objects ×
4 conditions for 5 controllers, the real-world log and object thumbnails), so the paper figures can be
reproduced without running the simulator:

```bash
cd analysis
python analyze.py \
    --controller rl=example_data/sim/rl \
    --controller heuristic=example_data/sim/heuristic \
    --controller transformer=example_data/sim/transformer \
    --controller rl_clutter=example_data/sim/rl_clutter \
    --controller distilled=example_data/sim/distilled \
    --csv example_data/realworld/clutter_compile.csv \
    --thumbnail_dir example_data/object_thumbnails \
    --output_dir paper_figs
```

`heuristic` is the Panda IK baseline. See [analysis/README.md](analysis/README.md) for the controllers,
the metric definitions and how to add your own controller.

## Notes

- **Panda baseline clutter:** the Panda environment places a fixed number of neighbours per level (4 / 6 / 8)
  and does not run the classifier check. This matches how the Panda results in the paper were produced.
- **Trials:** the paper uses 100 trials per object and condition (`run_*.sh` default).

## Citation

If you use this protocol, please cite:

```bibtex
@article{TODO,
  title   = {TODO},
  author  = {Wong, Jin and TODO},
  journal = {TODO},
  year    = {2026}
}
```

The benchmark objects come from the EGAD! dataset ([project page](https://dougsm.github.io/egad/)):

```bibtex
@article{morrison2020egad,
  title   = {EGAD! an Evolved Grasping Analysis Dataset for diversity and reproducibility in robotic manipulation},
  author  = {Morrison, Douglas and Corke, Peter and Leitner, J{\"u}rgen},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2020},
  volume  = {5},
  number  = {3},
  pages   = {4368--4375}
}
```

The hand model is the Contactile dexterous hand, provided by [Contactile Pty Ltd](https://contactile.com).
Please acknowledge Contactile if you use it.

The Panda baseline uses GG-CNN ([Morrison et al., RSS 2018](https://github.com/dougsm/ggcnn)) and
Segment Anything ([Kirillov et al., ICCV 2023](https://github.com/facebookresearch/segment-anything)).

## License

Code: BSD-3-Clause (see [LICENSE](LICENSE)). The Contactile hand model and the EGAD objects keep their own terms, listed in `LICENSE`.
