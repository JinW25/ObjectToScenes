# Clutter-Complexity Classifier

This pipeline generates cluttered tabletop scenes and measures the local clutter around a target
object. It then stratifies the scenes into three clutter levels with PCA + K-Means:

| Label | Level |
|---|---|
| `C0` | easy |
| `C1` | medium |
| `C2` | hard |

It also trains the multimodal (image + spatial features) CNN that the benchmark uses to check a
scene's clutter level. The paper's method ends at the PCA + K-Means stratification. The
preprocessing and training steps are here so the released classifier weights
(`weights/classifier/best_model.pth` + `config.json`) can be reproduced.

Scene objects come from the [EGAD!](https://dougsm.github.io/egad/) object set (see [Citation](#citation)).

## Pipeline overview

The scenes can come from either of two simulators. Both write the same CSV columns, and the
rest of the pipeline is shared.

```
MuJoCo path (mujoco/)                               Isaac Sim path (isaac/)
1a. randomize_scene.py                              1b. collect_classifier_dataset.py
      -> images + clutter CSV (10 columns)                -> images + full CSV (24 columns)
2.  spatial_features_extraction_raw.py                    (spatial features computed in-sim)
      -> + bbox / position features (24 columns)
                    \                                 /
                     v                               v
3. pca_clustering.py        PCA + K-Means stratification -> cluster label per scene (C0/C1/C2)
4. preprocessing.py         stratified 70/20/10 train/val/test split + augmentation
5. training.py              multimodal ResNet + MLP classifier -> best_model.pth + config.json
6. plot_pdf.py              PDF figures of steps 3-5
```

The released weights were trained on data from the **MuJoCo path**. The `config` stored in
`best_model.pth` lists `distinctive_3tier_data_spatial_features.csv` as the training CSV.

## Layout

```
classifier/
├── README.md
├── requirements.txt                    non-Isaac dependencies
├── mujoco/
│   ├── randomize_scene.py              1a. scene generation + rendering (MuJoCo)
│   ├── spatial_features_extraction_raw.py  2. bbox / position features from the images
│   ├── camera_setup.py                 interactive camera-pose helper
│   ├── obj_to_xml.py                   cleans per-object MuJoCo XML files (optional)
│   └── obj_xml/                        base scene XML + its mesh / texture assets
├── isaac/
│   ├── collect_classifier_dataset.py   1b. scene generation + rendering (Isaac Sim)
│   └── obj_2_usd.py                    EGAD .obj -> .usd converter
├── pca_clustering.py                   3. PCA + K-Means
├── method_comparison.py                3'. compare clustering methods (optional)
├── preprocessing.py                    4. dataset split
├── training.py                         5. classifier training
└── plot_pdf.py                         6. figures
```

The Isaac data-collection environment is in the `clutter_grasp` package:
`source/clutter_grasp/clutter_grasp/envs/data_collection_env.py` and `data_collection_env_cfg.py`
(gym ID `ClutterGrasp-DataCollection-v0`).

All scripts work from any working directory. Default inputs and outputs live under the
git-ignored `data/` folder at the repository root:

```
data/
├── egad_mesh/            EGAD .obj meshes (you download these)
├── egad_usd/             EGAD meshes converted to USD (Isaac path)
└── classifier/
    ├── mujoco/           randomize_scene.py + spatial_features_extraction_raw.py outputs
    ├── isaac/            collect_classifier_dataset.py outputs
    ├── pca_kmeans/       pca_clustering.py outputs
    ├── method_comparison/
    ├── cnn_dataset/      preprocessing.py outputs
    ├── training/         training.py outputs
    └── pdf_results/      plot_pdf.py outputs
```

## Installation

The MuJoCo path and steps 3-6 need only the packages in `requirements.txt` (tested with Python 3.10):

```bash
pip install -r classifier/requirements.txt
```

The Isaac path (`isaac/`) runs in the Isaac Lab environment, with this repository's package installed
(`pip install -e source/clutter_grasp`). `obj_2_usd.py` also needs `trimesh`, which is installed with
Isaac Lab.

## EGAD objects (not included)

Download the EGAD object meshes (`.obj`, file names like `A00_0.obj`) from the
[EGAD project website](https://dougsm.github.io/egad/) into `data/egad_mesh/`.

- **MuJoCo path:** `randomize_scene.py` reads the `.obj` files directly (`--mesh_folder`, default
  `data/egad_mesh`).
- **Isaac path:** convert the meshes to USD first. The data-collection env reads `data/egad_usd/` by
  default; you can override this with `--usd_dir` or `DataCollectionEnvCfg.object_usd_dir`.

  ```bash
  ./isaaclab.sh -p classifier/isaac/obj_2_usd.py \
      --input-folder data/egad_mesh --output-folder data/egad_usd --center --headless
  ```

  The defaults (scale 0.0006, mass 0.5 kg, convex-hull collision, random colour per object) together
  with `--center` (mesh centroid at the prim origin) reproduce the USDs used for the released data.

EGAD object IDs encode the grasp difficulty (letter A-Y → 1-25) and the shape complexity (number).
Both data paths parse them from the file name, so keep the original names.

---

## 1a. Generate data: MuJoCo

`mujoco/randomize_scene.py` builds random cluttered scenes on a table. The base XML also declares
ViperX 300s arm and hand meshes as assets, but no robot body is placed in the scene. It renders each scene from a fixed camera and logs the clutter around a target object.

```bash
python classifier/mujoco/randomize_scene.py --num_images 5000
```

For each scene the script:

- places 3-30 EGAD objects using one of five **density patterns**
  (`clustered_center`, `two_clusters`, `uniform_random`, `ring_pattern`, `gradient_density`);
- chooses a **target object** and places it with one of five strategies
  (`in_dense`, `near_dense`, `between`, `isolated`, `edge_sparse`), scaled up by `target_scale_factor`;
- writes the scene XML, renders it, and draws a green bounding box around the target;
- appends a row to the output CSV. Image paths are stored as absolute paths.

| Flag | Default | Meaning |
|---|---|---|
| `--num_images` | 10 | scenes to generate |
| `--start_index` | 0 | first scene index (to append without overwriting images) |
| `--mesh_folder` | `data/egad_mesh` | EGAD `.obj` meshes |
| `--base_xml` | `mujoco/obj_xml/cluttered_scene.xml` | base scene (floor + table) |
| `--output_dir` | `data/classifier/mujoco` | outputs |
| `--csv_name` | `diverse_local_complexity_data.csv` | CSV name inside `--output_dir` |

Workspace limits, object count, and camera pose are in the `config` dict in the `__main__` block.
`mujoco/camera_setup.py --model_path <scene.xml>` opens an interactive viewer and prints a camera pose
that you can paste into the config.

**Outputs** (in `--output_dir`):

- `random_cluttered_scene.xml`: the last generated scene
- `images/scene_XXXXX.png`: rendered scene with the green target box
- `neighbor_check/neighbor_XXXXX.png`: debug view of the detected neighbours
- `diverse_local_complexity_data.csv`: the first 10 columns of the [CSV schema](#csv-schema)

## 2. Label data: spatial features (MuJoCo path only)

`mujoco/spatial_features_extraction_raw.py` detects the green target box in each rendered image. It
adds the normalised bbox and position features as the remaining 14 columns of the schema.

```bash
python classifier/mujoco/spatial_features_extraction_raw.py \
    --input_csv data/classifier/mujoco/diverse_local_complexity_data.csv
# -> data/classifier/mujoco/diverse_local_complexity_data_spatial_features.csv (+ _viz.png)
```

Rows where no box is detected are dropped. Use `--output_csv` to choose the output path and `--yes` to
overwrite existing feature columns without a prompt.

## 1b. Generate data: Isaac Sim

`isaac/collect_classifier_dataset.py` uses the `DataCollectionEnv` (table + 5-60 random EGAD objects +
a fixed 640×480 camera, no robot). Each scene goes through these steps:

1. The script picks a random target and lets the scene settle.
2. It projects the target's mesh points to draw the lime target box.
3. It computes the same clutter metrics and all spatial features in-sim.
4. It applies photometric augmentation (brightness / contrast / saturation, 50 % horizontal flip with
   the bbox mirrored) to the saved image.

The step-2 script is not needed for this path.

```bash
./isaaclab.sh -p classifier/isaac/collect_classifier_dataset.py \
    --num_scenes 2000 --enable_cameras --headless
```

| Flag | Default | Meaning |
|---|---|---|
| `--num_scenes` | 500 | scenes to save |
| `--output_dir` | `data/classifier/isaac` | outputs (CSV rows are appended) |
| `--usd_dir` | `data/egad_usd` | EGAD USD objects |
| `--min_objects` / `--max_objects` | 5 / 60 | objects per scene |
| `--settling_steps` | 150 | physics steps before capture |
| `--neighbor_margin` | 0.02 | AABB margin (m) for counting neighbours |
| `--validate` | off | also save clean + metric-overlay images |

**Outputs:**

- `classifier_data.csv`: all 24 columns
- `images/scene_XXXXX.png`
- `validation/scene_XXXXX_{clean,metrics}.png` (with `--validate`)

## CSV schema

Both paths produce the same 24 columns in this order. The MuJoCo path produces them after step 2.

| Column | Description |
|---|---|
| `image_pixels` | Absolute path to the scene image (with the green target box) |
| `target_grasp_difficulty` | EGAD grasp difficulty of the target (A-Y → 1-25) |
| `target_shape_complexity` | EGAD shape complexity of the target |
| `num_obstacles` | Non-EGAD bodies in the scene (MuJoCo: world/floor + table = 2 with the shipped base XML); always 0 in Isaac |
| `num_neighbors` | Objects whose AABB is within the margin of the target's AABB |
| `mean_neighbor_distance`, `min_neighbor_distance` | Distances (m) from the target to its neighbours |
| `mean_neighbor_grasp_difficulty`, `mean_neighbor_shape_complexity` | Mean EGAD attributes of the neighbours |
| `free_space_volume` | Free-space ratio (0-1) in a 3 cm shell around the target's AABB |
| `bbox_center_x`, `bbox_center_y`, `bbox_width`, `bbox_height`, `bbox_area` | Target box, normalised by image size |
| `obj_x`, `obj_y`, `obj_z` | Normalised target position (0-1) |
| `distance_to_center`, `distance_to_camera` | Normalised (0-1) |
| `bbox_x_min`, `bbox_x_max`, `bbox_y_min`, `bbox_y_max` | Target box corners, normalised |

The columns are identical, but some values are computed differently in the two simulators. Do not
mix data from both paths in one clustering run.

| Quantity | MuJoCo (`randomize_scene.py` + step 2) | Isaac (`collect_classifier_dataset.py`) |
|---|---|---|
| Neighbour candidates | All bodies (EGAD objects, table, world/floor), using a geom bounding box per body | EGAD objects only, using AABBs of 32 sampled mesh vertices |
| Neighbour distance | Min distance between the 8 box corners of target and neighbour | AABB gap distance (0 if they overlap) |
| `free_space_volume` | Overlaps clipped to the target AABB | Overlaps clipped to the expanded (3 cm) AABB |
| `num_obstacles` | Count of non-EGAD bodies | Always 0 |
| bbox features | Detected from the green box in the saved image | Projected target mesh points (4 px pad), mirrored on flip |
| `obj_x/y/z`, `distance_to_center/camera` | Heuristic estimates from the 2D bbox | Target world position normalised by the table size |

## 3. Stratification with PCA + K-Means

`pca_clustering.py` runs in two stages:

1. **PCA exploration:** standardises the chosen features, runs PCA, and saves the scree plot,
   loadings, and PC scatter plots.
2. **K-Means clustering:** clusters on the chosen PCs. It then relabels the clusters by centroid
   position so labels are stable across runs: cluster 0 has the lowest PC1, and the other two are
   ordered by PC2. It writes the labelled dataset.

```bash
# Stage 1 only: inspect PCA before clustering
python classifier/pca_clustering.py --skip_clustering

# Full run: 3 clusters on PC1 and PC2 (these are the defaults)
python classifier/pca_clustering.py \
    --csv_path data/classifier/mujoco/diverse_local_complexity_data_spatial_features.csv \
    --features num_neighbors mean_neighbor_distance min_neighbor_distance free_space_volume \
    --cluster_pcs 1,2 --kmeans_n_clusters 3

# Isaac data
python classifier/pca_clustering.py --csv_path data/classifier/isaac/classifier_data.csv

# Interactive feature selection and confirmation
python classifier/pca_clustering.py --interactive
```

The default features are the local-clutter measures `num_neighbors`, `mean_neighbor_distance`,
`min_neighbor_distance`, and `free_space_volume`. The default `--csv_path` is the MuJoCo step-2
output, and the default `--output_dir` is `data/classifier/pca_kmeans`.

**Outputs:**

- `pca_loadings.csv`, `pca_results.csv`
- `kmeans_metrics.csv` (silhouette / Calinski-Harabasz)
- `pca_kmeans_results.csv` (input CSV + `kmeans_cluster`)
- `cnn_dataset.csv` (`image_path`, `cluster`, and 64×64 grayscale pixels)
- PDF plots

### Comparing clustering methods (optional)

`method_comparison.py` compares K-Means, GMM, DBSCAN, HDBSCAN (needs `hdbscan`), and hierarchical
clustering on the same PCA space. It scores them with silhouette, Calinski-Harabasz, and
Davies-Bouldin:

```bash
python classifier/method_comparison.py --csv_path <features CSV> --n_clusters 3
```

## 4. Build the classifier dataset

`preprocessing.py` reads `cnn_dataset.csv` from step 3, drops noise labels (`-1`), and makes a
stratified **70 / 20 / 10** train / val / test split. Images are resized to 224×224, and each training
image gets 3 augmented copies.

```bash
python classifier/preprocessing.py \
    --csv_path data/classifier/pca_kmeans/cnn_dataset.csv \
    --output_root data/classifier/cnn_dataset
```

Other flags are `--augmentations_per_image` (3) and `--seed` (42). `--output_root` is deleted and
recreated. Output layout:

```
<output_root>/
├── train/cluster_<k>/...   train_metadata.csv
├── val/cluster_<k>/...     val_metadata.csv
└── test/cluster_<k>/...    test_metadata.csv
```

## 5. Train the classifier

`training.py` trains a **multimodal CNN** with three parts:

- **Image branch:** an ImageNet-pretrained ResNet (18 / 34 / 50, default 50) with a 1-channel first
  convolution. Its input is grayscale images that are CLAHE-equalised and normalised per image, with
  strong augmentation for colour and lighting invariance.
- **Spatial branch:** an MLP on 6 spatial features, in this order:
  `obj_x`, `obj_y`, `distance_to_center`, `bbox_area`, `bbox_width`, `bbox_height`.
- **Fusion head:** concatenates the two embeddings and passes them through fully connected layers to
  predict the cluster label.

Training uses label smoothing, weight decay, dropout, OneCycle LR, and early stopping (patience 20).

```bash
python classifier/training.py \
    --data_dir data/classifier/cnn_dataset \
    --csv_path data/classifier/pca_kmeans/pca_kmeans_results.csv \
    --output_dir data/classifier/training
```

| Flag | Default | Meaning |
|---|---|---|
| `--data_dir` | `data/classifier/cnn_dataset` | split dataset from step 4 |
| `--csv_path` | `data/classifier/pca_kmeans/pca_kmeans_results.csv` | CSV with `image_pixels` + spatial features |
| `--output_dir` | `data/classifier/training` | outputs |
| `--model_name` | `resnet50` | `resnet18` / `resnet34` / `resnet50` |
| `--num_epochs` / `--batch_size` / `--num_workers` | 150 / 32 / 4 | |

Other hyper-parameters are in the `config` dict in the script.

**Outputs** (in `--output_dir`):

- `best_model.pth` (`{model_state_dict, val_acc, config}`) and `config.json`
- `history.npy`, `training_curves.png`, `training_summary.txt`
- `results.txt` (test classification report), `confusion_matrix.png`
- `test_predictions.npy`, `test_labels.npy`

To use a newly trained model in the benchmark, copy `best_model.pth` and `config.json` to
`weights/classifier/`. The loader is `clutter_grasp.protocol.classifier.load_classifier_model`.
It reads `model_name` and `dropout_rate` from `config.json` and the number of classes from the
checkpoint. It assumes the 6 spatial features above in that order, and class index *k* = `C<k>`.

## 6. Plot results

`plot_pdf.py` collects the PCA/K-Means outputs, training results, and sample images into PDF figures.
All four directories default to the folders under `data/classifier/` above.

```bash
python classifier/plot_pdf.py --pca_dir data/classifier/pca_kmeans --cnn_dir data/classifier/training \
    --data_dir data/classifier/cnn_dataset --out_dir data/classifier/pdf_results
```

---

## Citation

The scene objects are from the EGAD! dataset. For more about the objects, see the
[project website](https://dougsm.github.io/egad/), and please cite:

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

The EGAD! objects are distributed under their own license; see the
[EGAD project website](https://dougsm.github.io/egad/).
