# Analysis: grasping-in-clutter protocol figures

`analyze.py` is a single script that reproduces the four analysis figures in the paper from raw experiment results. `common.py` is its helper module: loading, metric definitions and plot style.

| Figure | Produced when | Output |
|---|---|---|
| Difficulty ranking of objects across clutter levels | `--controller` is given | `ranking_heatmap_<ref>.pdf` |
| Picking-time gap of every controller vs. the reference baseline | `--controller` is given | `absolute_gap_violin_all_ctrls.pdf` |
| Simulated vs. real-world time to successful grasp | `--csv` is given (lines from the `--trend` controllers) | `time_trend_overlay.pdf` |
| Gamma-GLM coefficients (clutter, difficulty, interaction) per controller | `--csv` and `--controller` are given | `coef_all_terms.pdf` |

Each PDF is written with a CSV of the same name that holds the plotted numbers. The script also writes `object_metrics.csv` (D and SR for every controller, object and condition) and `absolute_gap_summary.csv` (mean, median and std of the gap for every controller and condition).

## Controllers in the paper and in `example_data/`

| Name | Controller | Source folder (paper) |
|---|---|---|
| `rl` | Per-object PPO policy, Contactile hand ("RL (Isolated)"); reference controller | `benchmark_experiments_individual` |
| `heuristic` | **Panda IK controller**: Franka Panda with a parallel gripper, scripted inverse-kinematics grasp (no learning) | `benchmark_experiments_panda_ik_final` |
| `transformer` | Transformer policy distilled from the per-object RL policies ("Distilled Policy") | `benchmark_experiments_transformer_individual` |
| `rl_clutter` | PPO policy trained in the clutter environment ("RL (Clutter env)") | `benchmark_experiments_cluttered` |
| `distilled` | Transformer distilled from `rl_clutter` ("Distilled Clutter") | `benchmark_experiments_transformer` |

`example_data/` is the full paper dataset (about 38 MB):
* `sim/<name>/`: 24 objects × 4 conditions = 96 runs for each of the 5 controllers. Only `results/results.json` from each run is copied, unmodified, and the run-folder names are kept.
* `realworld/clutter_compile.csv`: the real-world pick log.
* `object_thumbnails/`: the 24 object thumbnails.

## Install

```bash
pip install -r requirements.txt   # PyMuPDF is optional (object thumbnails only)
```

## Reproduce all four paper figures

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

The command writes `ranking_heatmap_rl.pdf`, `absolute_gap_violin_all_ctrls.pdf`, `time_trend_overlay.pdf` and `coef_all_terms.pdf`, with their CSVs, into `paper_figs/`. The order of the `--controller` flags sets the legend and bar order. The first controller (`rl`) is the reference, and by default `rl` and `heuristic` are the trend lines.

## Smaller examples

Minimal run with the RL policy and the Panda IK heuristic only:

```bash
python analyze.py --controller rl=example_data/sim/rl --controller heuristic=example_data/sim/heuristic \
    --csv example_data/realworld/clutter_compile.csv --thumbnail_dir example_data/object_thumbnails \
    --output_dir out_minimal
```

To add other controllers, append more `--controller` flags. Each one becomes a violin per condition in the gap figure and a bar group in the GLM figure:

```bash
python analyze.py --controller rl=example_data/sim/rl --controller heuristic=example_data/sim/heuristic \
    --controller rl_clutter=example_data/sim/rl_clutter \
    --controller transformer=example_data/sim/transformer \
    --controller distilled=example_data/sim/distilled \
    --output_dir out_sim_only          # no --csv: only the two simulated figures
```

To add your own controller, point a new name at a folder laid out as described under *Input layout*, and give it a label and colour:

```bash
python analyze.py --controller rl=example_data/sim/rl --controller heuristic=example_data/sim/heuristic \
    --controller mypolicy=/path/to/my_runs \
    --label mypolicy="My Policy" --color mypolicy=#d62728 \
    --csv example_data/realworld/clutter_compile.csv \
    --trend rl mypolicy \
    --output_dir out_mine
```

`--trend rl mypolicy` draws your controller as a line in the sim-vs-real trend figure. Its line style and marker come from a fallback list.

## Command-line options

* `--controller NAME=DIR` (repeatable): a simulated controller and its folder of runs, in plot order. Any name works. The names in the table above come with the paper's labels and colours. Other names are labelled with the name itself and get the next fallback colour.
* `--label NAME=TEXT` and `--color NAME=#RRGGBB`: override a display label or colour. This also works for the real-world series, named `real_experiment`.
* `--reference NAME` (default: the first `--controller`): the controller whose ranking is shown and whose isolated time is the gap baseline.
* `--csv FILE`: the real-world `clutter_compile.csv`. Giving it enables the real-world figures.
* `--trend NAME ...` (default: `rl heuristic` if both are given, otherwise the first two controllers): the simulated controllers drawn as lines in `time_trend_overlay.pdf`. Every `--controller` enters the GLM figure.
* `--thumbnail_dir DIR` (optional): object images `<object>.pdf` drawn above the trend figure. Requires PyMuPDF.
* `--sr_threshold 60`: success rate (%) below which an object is flagged in the gap figure.
* `--outlier_z 3.5`: threshold of the trial outlier filter. `0` disables the filter.
* `--top_n_shortest 3`: the number of fastest real-world picks kept per (object, clutter level). `0` keeps all picks.
* `--prefix single`: the prefix of the run-folder names.

## Metric definitions

The code for each definition is in `common.py` or `analyze.py`, as noted.

**Conditions.** `s ∈ {ε0, E1, E2, E3}` = isolated (no clutter), `C0_easy`, `C1_medium` and `C2_hard`.

**Picking time of a trial.** The trial's `picking_time` in seconds. Only successful trials count.

**Trial outlier filter** (`common.mad_filter`). The filter runs separately within each (controller, object, condition) cell. A picking time *t* is dropped if `0.6745·|t − median| / MAD > 3.5`, where MAD is the median absolute deviation of the cell. Cells with fewer than 4 times, or with MAD = 0, are kept as they are.

**Time to successful grasp, D_{i,s}.** The mean of the filtered picking times of object *i* in condition *s*. All four figures are built on this per-trial picking time. None of them uses the combined time/drops/success score.

**Success rate, SR_{i,s}.** `success_rate` from the run's summary block. If that is missing, it is the fraction of trials with `success == true`.

**Ranking heatmap** (`analyze.ranking_table`). Uses the reference controller only, and only objects that have a finite isolated time D_{i,ε0}. The columns are these objects sorted by D_{i,ε0}. In each condition the objects are ranked separately by D_{i,s}, where 1 = fastest (easiest). If the colours change down a column, clutter changed the object's relative difficulty.

**Absolute picking-time gap** (`analyze.gap_table`).

    A_{i,s}^π = D_{i,s}^π − D_{i,ε0}^ref

The gap is measured in seconds against the reference controller's own isolated time for the same object. Because of this, the reference at ε0 has a gap of exactly 0, and every other cell reads "seconds slower than the reference's uncluttered pick". Each violin shows the distribution over objects, with the box drawn at 1.5 IQR. The number beside each violin is the mean over objects. No object is dropped. An object is only flagged (hollow red diamond plus its initial) when SR_{i,s}^π or SR_{i,ε0}^ref is below `--sr_threshold`. Other points outside the whiskers are labelled with the object's initial.

**Real-world time** (`analyze.load_realworld_csv`). Each row of `clutter_compile.csv` is one object that was finally picked. `object_execute_s` is the execution time summed over all attempts until success, because the real-world protocol retries until it succeeds. The 3 fastest picks per (object, clutter level) are kept. The outlier filter is then applied, but it never removes anything when a cell has fewer than 4 picks. The trend figure plots mean ± population std. Simulated controllers are drawn as lines on the left axis. Real-world results are drawn as stars on the right axis, because real picks take several times longer.

**Grasp-score GLM** (`analyze.glm_design` and `fit_gamma_log_glm`). Each controller gets its own fit on all of its filtered successful trials:

    log E[t] = a0 + A·clutter + C·difficulty + D·clutter·difficulty,    t ~ Gamma

* `clutter` takes the values 0, 1, 2, 3 for ε0, E1, E2 and E3.
* `difficulty` is the object's 1-based position in the alphabetical order of object names. The object set is named in its designed difficulty order: A24_0, B25_3, … X25_3.
* The fit is IRLS. Standard errors are model-based: phi·(XᵀWX)⁻¹, with phi from the Pearson residuals.
* p-values come from a two-sided t-test with n − k degrees of freedom.
* `exp(A)` is the multiplicative change in picking time per clutter level.
* In the figure, whiskers show the 95% CI (±1.96 SE). A faded bar means p ≥ 0.05. Stars mark p < .05 (\*), < .01 (\*\*) and < .001 (\*\*\*).
* The real-world data are not in this figure, because 5 objects are too few for the difficulty terms.

## Input layout

```
<controller_dir>/
  single_<OBJECT>_<YYYYMMDD>_<HHMMSS>/results/results.json            # isolated (ε0)
  single_<OBJECT>_C0_easy_<YYYYMMDD>_<HHMMSS>/results/results.json    # E1
  single_<OBJECT>_C1_medium_<YYYYMMDD>_<HHMMSS>/results/results.json  # E2
  single_<OBJECT>_C2_hard_<YYYYMMDD>_<HHMMSS>/results/results.json    # E3
```

* Object names may contain underscores.
* Folders whose names do not match this pattern are ignored.
* If an (object, condition) pair appears twice, the later timestamp wins.
* `analyze.py` skips objects that have no isolated run.

The fields read from `results.json` are listed below. All other fields, such as `chaos_metrics` and `steps`, are ignored.

```jsonc
{
  "overall_statistics": {          // or "overall_stats"; optional
    "success_rate": 0.98,          // fraction in [0, 1]; computed from trials if absent
    "total_trials": 100
  },
  "trial_results": [               // if absent, results/trial_results.json is read instead
    {"success": true,  "picking_time": 7.08},   // seconds; a missing "success" counts as true
    {"success": false, "picking_time": 30.0}
  ]
}
```

The picking time is the first finite value among `picking_time`, `time_to_success`, `time`, `duration`, `elapsed_time` and `pick_time`.

`clutter_compile.csv` needs these columns: `obj_name`, `clutter_level` (0 to 3) and `object_execute_s`.
