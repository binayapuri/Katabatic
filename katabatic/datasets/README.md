# Evaluation and Performance Benchmarking Data Catalogue

This catalogue documents the five standard benchmark datasets shipped in `katabatic/datasets/` (`adult.csv`, `car.csv`, `magic.csv`, `nursery.csv`, `shuttle.csv`).

> **Important Note on Packaged CSV Column Headers vs. UCI Descriptive Names:**
> Several packaged CSV files in `katabatic/datasets/` use numeric index headers (`0, 1, ...` in `car.csv` and `nursery.csv`), generic attribute headers (`time, a1..a8` in `shuttle.csv`), or the target column name `class` (in `adult.csv`). Benchmark scripts (`benchmarks/examples/<model>/run_<model>_<dataset>.py`) and `RunConfig` must always reference the **exact packaged CSV column names** listed in the Standardized Dataset Definition Table below.

---

## Standardized Dataset Definition Table (Packaged CSV Schema Reference)

| Dataset (`dataset_name`) | Packaged CSV File | Packaged Rows | Size Group (`protocol.py`) | Target Column (`target_col_raw`) | Packaged Categorical Columns (`categorical_cols`) | Packaged Continuous Columns (`continuous_cols`) |
| --- | --- | --- | --- | --- | --- | --- |
| **`adult`** | `katabatic/datasets/adult.csv` | 32,561 | Large (`> 25,000`) | `class` *(UCI name: `income`)* | `workclass`, `education`, `education-num`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country` | `age`, `fnlwgt`, `capital-gain`, `capital-loss`, `hours-per-week` |
| **`car`** | `katabatic/datasets/car.csv` | 1,728 | Small (`< 5,000`) | `6` *(UCI name: `class`)* | `0`, `1`, `2`, `3`, `4`, `5` *(UCI names: `buying`, `maint`, `doors`, `persons`, `lug_boot`, `safety`)* | *(None — all categorical)* |
| **`magic`** | `katabatic/datasets/magic.csv` | 19,020 | Medium (`5,000–25,000`) | `class` | *(None — all continuous)* | `fLength`, `fWidth`, `fSize`, `fConc`, `fConc1`, `fAsym`, `fM3Long`, `fM3Trans`, `fAlpha`, `fDist` |
| **`nursery`** | `katabatic/datasets/nursery.csv` | 12,960 | Medium (`5,000–25,000`) | `8` *(UCI name: `class`)* | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7` *(UCI names: `parents`, `has_nurs`, `form`, `children`, `housing`, `finance`, `social`, `health`)* | *(None — all categorical)* |
| **`shuttle`** | `katabatic/datasets/shuttle.csv` | 58,000 *(capped at `max_train_rows=50,000`)* | Large (`> 25,000`) | `class` | *(None — all numerical)* | `time`, `a1`, `a2`, `a3`, `a4`, `a5`, `a6`, `a7`, `a8` |

---

## Standardized Model Settings Reference (`benchmarks/protocol.py`)

All benchmark runs use an **80/20 stratified train/test split** (`TEST_SIZE = 0.2`), **`SEED = 42`**, and **`MAX_TRAIN_ROWS = 50,000`**. Models follow one of three training budget policies:

| Model Key | Family | Protocol Policy | Seed Policy | Standard / Default Training Budget |
| --- | --- | --- | --- | --- |
| **`arf`** | Tree / Adversarial RF | `fixed` | `seed=42` | Fixed estimator trees (no epoch loop) |
| **`codi`** | Diffusion | `standard` | `seed=42` | `epochs_for(dataset)`: `100` (small/medium), `150` (large) |
| **`ctgan`** | GAN | `standard` | `seed=42` | `epochs_for(dataset)`: `100` (small/medium), `150` (large); `batch_size=512` |
| **`fairtabdiffusion`** | Diffusion (Fairness) | `standard` | `seed=42` | `epochs_for(dataset)`: `100` (small/medium), `150` (large) |
| **`ganblr`** | GAN / Bayesian | `standard` | `seed=42` *(sampling)* | `epochs_for(dataset)`: `100` (small/medium), `150` (large) |
| **`gmm`** | Statistical | `fixed` | `seed=42` | Fixed EM components |
| **`great`** | Language Model (LLM) | `specific` | `seed=42` *(sampling)* | `llm="gpt2"`; `2` epochs (`car`, `magic`, `nursery`, `shuttle`), `adult` budget under timing review |
| **`histogram`** | Statistical | `fixed` | `seed=42` | Fixed bin estimator |
| **`kde`** | Statistical | `fixed` | `seed=42` | Fixed kernel bandwidth fitting |
| **`medgan`** | GAN / Autoencoder | `specific` | `seed=42` | Two-stage budget: `100` pretrain epochs + `1000` GAN epochs |
| **`mst`** | Differential Privacy | `fixed` | `N/A` *(no seed arg)* | Fixed graphical-model marginal fitting |
| **`naivebayes`** | Statistical | `fixed` | `seed=42` | Single-pass Naive Bayes fit |
| **`pategan`** | GAN (Diff. Privacy) | `specific` | `seed=42` | Iteration-based (`niter=10000`, `num_teachers=10`, `epsilon=1.0`, `delta=1e-5`) |
| **`privtree`** | Differential Privacy | `fixed` | `seed=42` | Fixed differentially private tree decomposition |
| **`realtabformer`** | Language Model | `specific` | `random_state=42` | Model-specific transformer epochs (`batch_size=8`) |
| **`smote`** | Interpolation | `fixed` | `random_state=42` | `k_neighbors=5` nearest-neighbour interpolation |
| **`synthpop`** | Statistical / CART | `fixed` | `seed=42` | Sequential CART conditional synthesis |
| **`tabddpm`** | Diffusion | `specific` | `seed=42` | Step-based diffusion schedule |
| **`tabebm`** | Energy-Based Model | `specific` | `seed=42` | SGLD sampling steps |
| **`tabkde`** | Statistical | `fixed` | `seed=42` | Multivariate KDE fitting |
| **`tabsyn`** | Copula / Latent Diffusion | `specific` | `seed=42` | Two-stage: `50` VAE decoder epochs + `500` diffusion epochs |
| **`tvaegan`** | VAE / GAN | `standard` | `seed=42` | `epochs_for(dataset)`: `100` (small/medium), `150` (large) |

---

## Detailed Dataset Specifications

## Adult

This dataset is from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult).

| Property           | Value                                              |
| ------------------ | -------------------------------------------------- |
| **Task**           | Binary classification                              |
| **Instances**      | 32,561 packaged in `katabatic/datasets/adult.csv` (48,842 in full UCI release) |
| **Features**       | 14 categorical and integer features                |
| **Target**         | `class` in packaged CSV (`income` in original UCI documentation) |
| **Classes**        | `<=50K`, `>50K`                                    |
| **Missing values** | Yes                                                |
| **License**        | CC BY 4.0                                          |
| **DOI**            | [10.24432/C5XW20](https://doi.org/10.24432/C5XW20) |

### Fields

| Packaged CSV Column | UCI Descriptive Name | Type        | Description                                 |
| ------------------- | -------------------- | ----------- | ------------------------------------------- |
| `age`               | `age`                | Integer     | Age                                         |
| `workclass`         | `workclass`          | Categorical | Employment type                             |
| `fnlwgt`            | `fnlwgt`             | Integer     | Census final weight                         |
| `education`         | `education`          | Categorical | Highest education level                     |
| `education-num`     | `education-num`      | Integer     | Numerical representation of education level |
| `marital-status`    | `marital-status`     | Categorical | Marital status                              |
| `occupation`        | `occupation`         | Categorical | Occupation category                         |
| `relationship`      | `relationship`       | Categorical | Family/household relationship               |
| `race`              | `race`               | Categorical | Race category                               |
| `sex`               | `sex`                | Binary      | `Female`, `Male`                            |
| `capital-gain`      | `capital-gain`       | Integer     | Capital gains                               |
| `capital-loss`      | `capital-loss`       | Integer     | Capital losses                              |
| `hours-per-week`    | `hours-per-week`     | Integer     | Hours worked per week                       |
| `native-country`    | `native-country`     | Categorical | Country of origin                           |
| `class`             | `income`             | Target      | `<=50K` or `>50K`                           |

> **Note:** Missing values occur in `workclass`, `occupation`, and `native-country`.

### Citation

> Becker, B. & Kohavi, R. (1996). *Adult* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20

## Car
This dataset is from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/19/car+evaluation).

| Property           | Value                                              |
| ------------------ | -------------------------------------------------- |
| **Task**           | Classification                                     |
| **Instances**      | 1,728                                              |
| **Features**       | 6 categorical features                             |
| **Target**         | `6` in packaged CSV (`class` in UCI documentation) |
| **Classes**        | 4 (`unacc`, `acc`, `good`, `vgood`)                |
| **Missing values** | None                                               |
| **License**        | CC BY 4.0                                          |
| **DOI**            | [10.24432/C5JP48](https://doi.org/10.24432/C5JP48) |

### Fields

| Packaged CSV Column | UCI Descriptive Name | Description        | Type          |
| ------------------- | -------------------- | ------------------ | ------------- |
| `0`                 | `buying`             | Buying price       | Categorical   |
| `1`                 | `maint`              | Maintenance price  | Categorical   |
| `2`                 | `doors`              | Number of doors    | Categorical   |
| `3`                 | `persons`            | Passenger capacity | Categorical   |
| `4`                 | `lug_boot`           | Luggage boot size  | Categorical   |
| `5`                 | `safety`             | Estimated safety   | Categorical   |
| `6`                 | `class`              | Car acceptability  | Target (Categorical) |

### Citation

> Bohanec, M. (1988). *Car Evaluation* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5JP48


## Magic

This dataset is from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope).

| Property           | Value                                              |
| ------------------ | -------------------------------------------------- |
| **Task**           | Binary classification                              |
| **Instances**      | 19,020                                             |
| **Features**       | 10 continuous features                             |
| **Target**         | `class`                                            |
| **Classes**        | 2 (`g`, `h`)                                       |
| **Missing values** | None                                               |
| **License**        | CC BY 4.0                                          |
| **DOI**            | [10.24432/C52C8B](https://doi.org/10.24432/C52C8B) |

### Fields

| Packaged CSV Column | Description                                                      | Type            |
| ------------------- | ---------------------------------------------------------------- | --------------  |
| `fLength`           | Major axis of ellipse                                            | Continuous      |
| `fWidth`            | Minor axis of ellipse                                            | Continuous      |
| `fSize`             | Log of the sum of content of all pixels                          | Continuous      |
| `fConc`             | Ratio of sum of two highest pixels over `fSize`                  | Continuous      |
| `fConc1`            | Ratio of highest pixel over `fSize`                              | Continuous      |
| `fAsym`             | Distance from highest pixel to centre, projected onto major axis | Continuous      |
| `fM3Long`           | Third root of third moment along major axis                      | Continuous      |
| `fM3Trans`          | Third root of third moment along minor axis                      | Continuous      |
| `fAlpha`            | Angle of major axis with vector to origin                        | Continuous      |
| `fDist`             | Distance from origin to centre of ellipse                        | Continuous      |
| `class`             | Particle class: `g` = gamma (signal), `h` = hadron (background)  | Target (Binary) |

### Citation

> Bock, R. (2004). *MAGIC Gamma Telescope* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52C8B

## Nursery

This dataset is from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/76/nursery).

| Property           | Value                                              |
| ------------------ | -------------------------------------------------- |
| **Task**           | Multiclass classification                          |
| **Instances**      | 12,960                                             |
| **Features**       | 8 categorical features                             |
| **Target**         | `8` in packaged CSV (`class` in UCI documentation) |
| **Missing values** | None                                               |
| **License**        | CC BY 4.0                                          |
| **DOI**            | [10.24432/C5P88W](https://doi.org/10.24432/C5P88W) |

### Fields

| Packaged CSV Column | UCI Descriptive Name | Description                    | Type        |
| ------------------- | -------------------- | ------------------------------ | ----------- |
| `0`                 | `parents`            | Parents' occupation            | Categorical |
| `1`                 | `has_nurs`           | Child's nursery                | Categorical |
| `2`                 | `form`               | Form of the family             | Categorical |
| `3`                 | `children`           | Number of children             | Categorical |
| `4`                 | `housing`            | Housing conditions             | Categorical |
| `5`                 | `finance`            | Financial standing             | Categorical |
| `6`                 | `social`             | Social conditions              | Categorical |
| `7`                 | `health`             | Health conditions              | Categorical |
| `8`                 | `class`              | Nursery application evaluation | Target (Categorical) |

### Citation

> Rajkovic, V. (1989). *Nursery* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5P88W

## Shuttle

This dataset is from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/148/statlog+shuttle).

| Property           | Value                                              |
| ------------------ | -------------------------------------------------- |
| **Task**           | Multiclass classification                          |
| **Instances**      | 58,000                                             |
| **Features**       | 9 numerical features in packaged CSV (`time`, `a1`..`a8`) |
| **Target**         | `class`                                            |
| **Missing values** | None                                               |
| **License**        | CC BY 4.0                                          |
| **DOI**            | [10.24432/C5WS31](https://doi.org/10.24432/C5WS31) |

### Fields

| Packaged CSV Column | UCI Descriptive Name | Type    | Description                                        |
| ------------------- | -------------------- | ------- | -------------------------------------------------- |
| `time`              | `Rad Flow` / Time    | Integer | Time-related numerical attribute                   |
| `a1`                | `Fpv Close`          | Integer | Numerical sensor attribute 1                       |
| `a2`                | `Fpv Open`           | Integer | Numerical sensor attribute 2                       |
| `a3`                | `High`               | Integer | Numerical sensor attribute 3                       |
| `a4`                | `Bypass`             | Integer | Numerical sensor attribute 4                       |
| `a5`                | `Bpv Close`          | Integer | Numerical sensor attribute 5                       |
| `a6`                | `Bpv Open`           | Integer | Numerical sensor attribute 6                       |
| `a7`                | Derived attribute 7  | Integer | Numerical sensor attribute 7                       |
| `a8`                | Derived attribute 8  | Integer | Numerical sensor attribute 8                       |
| `class`             | `class`              | Target  | Shuttle operating-state class (`1`–`7`)            |

### Citation

> *Statlog (Shuttle)* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5WS31
