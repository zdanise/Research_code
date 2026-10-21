# DNS Exfiltration Detection: Metaheuristic Feature Selection

Compares three metaheuristic feature-selection methods (ACO, PSO, Enhanced RLGWO) for detecting DNS exfiltration, using Random Forest and XGBoost on an ISP-derived DNS resolver dataset with 17 features. Each method searches for a subset of 8-10 features, and the subsets are compared against the full 17-feature baseline.

**Author:** Zethembe Danise
**Supervisor:** Dr Avinash Singh
**Module:** COS 700

## Repository layout
```
datasets/    not included (too large); see "Data" below
notebooks/   Corrected_Validation_Only_DNS_Experiment.ipynb (main experiment, matches the report)
src/         exfiltration_dataset.py, dns_eval_utils.py, dns_search_utils.py
             (Enhanced_detection.py is NOT included; see "Required external file")
results/     validation_only_run_v1/ (searches, predictions, metrics), mcnemar_output/
archive/     earlier exploratory notebooks (results differ from the report)
```

## Method summary
- Split: 70% train, 10% validation, 20% test by row.
- Search: candidates are fitted on a training sample and scored only on a validation sample; the test set is used for final evaluation only.
- Fitness: weighted F1 minus false-positive rate minus a penalty for deviating from 9 features (8-10 allowed).
- Evaluation: attack-class F1, false-positive rate, errors, training and batch-prediction time, and exact McNemar tests with Holm correction.

## Required external file
`src/Enhanced_detection.py` (Enhanced RLGWO optimiser, baseline RLGWO, DQN and fitness helpers) was written by Mahmoud Sammour (UTeM), version 2.1, 4 August 2025. It has no stated licence and is **not redistributed here**. Obtain it from the original author and place it in `src/` before running the search cells. The ACO, PSO, validation, statistics and figure code do not depend on it, but the Enhanced RLGWO search does.

## Data
The dataset is by Žiža et al. and is not stored here. Download it from Mendeley Data (https://doi.org/10.17632/c4n7fckkz3.3) and place `dataset.csv` and `dataset_modified.csv` in `datasets/`. Run the data-loading cells in the notebook to create the `.npy` arrays.

## Running
```
pip install -r requirements.txt
jupyter notebook notebooks/Corrected_Validation_Only_DNS_Experiment.ipynb
```
Run the cells from top to bottom. Searches take several hours; use the reload cell to reuse saved results.

## Paper
[ADD LINK TO PAPER]

## Credits
- Dataset: Žiža, K., Vuletić, P., Tadić, P. (2023). DNS Exfiltration Dataset, Mendeley Data, V3. https://doi.org/10.17632/c4n7fckkz3.3
- Dataset paper and baseline settings: Žiža, K., Tadić, P., Vuletić, P. DNS exfiltration detection in the presence of adversarial attacks and modified exfiltrator behaviour. Int. J. Inf. Secur. (2023). https://doi.org/10.1007/s10207-023-00723-w
- ACO follows Talabani et al. (2025); PSO follows an IJCSNS (2014) paper.
- Enhanced RLGWO code: Mahmoud Sammour (UTeM), not redistributed.
See `THIRD_PARTY_NOTICES.md` for details.

## Licence
Original code by the author (excluding third-party material): MIT (see `LICENSE`).

## Citation
See `CITATION.cff`. Please also cite the dataset paper above.
