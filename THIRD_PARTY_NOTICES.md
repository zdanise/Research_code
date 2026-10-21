# Third-party notices

| Component | Source | Used in | Terms |
|---|---|---|---|
| Enhanced RLGWO, baseline RLGWO, DQN, fitness helpers | Mahmoud Sammour (UTeM), "Enhanced DNS Tunneling Detection: Feature Selection using Metaheuristics", v2.1, 4 Aug 2025 | `src/Enhanced_detection.py` (**not included**) | No licence stated. Not redistributed; obtain from the original author. |
| Binary ACO | Talabani, H.S.; Abdul, Z.K.; Mohammed Saleh, H.M. (2025). DNS over HTTPS Tunneling Detection System Based on Selected Features via Ant Colony Optimization. Future Internet, 17, 211. https://doi.org/10.3390/fi17050211 | notebook `BinaryACO` class | Method adapted; changes documented in the class docstring |
| Binary PSO | Intrusion Feature Selection Algorithm Based on Particle Swarm Optimization. IJCSNS, 2014. http://paper.ijcsns.org/07_book/201412/20141208.pdf | notebook `BinaryPSO` class | Method adapted; changes documented in the class docstring |
| Dataset | Žiža, K., Vuletić, P., Tadić, P. (2023). DNS Exfiltration Dataset, Mendeley Data, V3. https://doi.org/10.17632/c4n7fckkz3.3 | `datasets/` (not redistributed) | See the Mendeley Data page for the licence |
| Dataset paper and baseline settings | Žiža, K., Tadić, P., Vuletić, P. DNS exfiltration detection in the presence of adversarial attacks and modified exfiltrator behaviour. Int. J. Inf. Secur. (2023). https://doi.org/10.1007/s10207-023-00723-w | baseline XGBoost settings, feature set | Cite when using the dataset |
| Libraries | scikit-learn, XGBoost, NumPy, pandas, SciPy, Matplotlib, tqdm | all code | Their own open-source licences |

## Enhanced_detection.py
This file is not part of the public repository. Without a licence its author keeps all rights. To reproduce the Enhanced RLGWO results, obtain the file from Mahmoud Sammour (UTeM) and place it in `src/`.
