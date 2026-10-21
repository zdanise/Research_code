"""Repeatable final evaluation for the DNS feature-selection experiment.

Each (configuration, classifier, seed) result is saved to its own JSON file,
so an interrupted run can be resumed without redoing finished work.
"""
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def slug(text):
    return text.lower().replace(" ", "_")


def make_classifier(key, seed):
    """Final-evaluation settings, matching the report's Tables 2 and 3."""
    if key == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(objective="binary:logistic", n_estimators=100, subsample=0.1,
                             scale_pos_weight=1, eval_metric="error",
                             early_stopping_rounds=10, n_jobs=4, random_state=seed)
    if key == "rf":
        return RandomForestClassifier(n_estimators=5, max_depth=20, class_weight=None,
                                      n_jobs=1, random_state=seed)
    raise ValueError(key)


def metrics_from_cm(cm):
    tn, fp, fn, tp = [int(v) for v in np.asarray(cm).ravel()]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr_percent = 100 * fp / (fp + tn) if (fp + tn) else 0.0
    return dict(tn=tn, fp=fp, fn=fn, tp=tp, precision=precision, recall=recall,
                f1=f1, fpr_percent=fpr_percent, total_errors=fp + fn)


def result_path(out_dir, config, key, seed):
    return Path(out_dir) / f"{slug(config)}_{key}_seed{seed}.json"


def evaluate_one(config, selected_idx, key, seed, arrays, out_dir,
                 n_pred_repeats=5, save_predictions=False):
    """Fit on the full training partition, predict the test partition.

    Training time covers fit() only (column slicing is excluded).
    Prediction is repeated n_pred_repeats times on the same fitted model and
    the median is reported, which reduces one-off timing noise.
    """
    path = result_path(out_dir, config, key, seed)
    if path.exists():
        return json.loads(path.read_text())

    x_train, y_train = arrays["x_train"], arrays["y_train"]
    x_val, y_val = arrays["x_val"], arrays["y_val"]
    x_test, y_test = arrays["x_test"], arrays["y_test"]
    idx = list(selected_idx)

    Xtr = np.asarray(x_train[:, idx])
    clf = make_classifier(key, seed)
    start = time.perf_counter()
    if key == "xgb":
        Xv = np.asarray(x_val[:, idx])
        clf.fit(Xtr, y_train, eval_set=[(Xv, y_val)], verbose=False)
        del Xv
    else:
        clf.fit(Xtr, y_train)
    train_seconds = time.perf_counter() - start
    del Xtr

    Xt = np.asarray(x_test[:, idx])
    times, prediction = [], None
    for r in range(n_pred_repeats):
        start = time.perf_counter()
        pred = clf.predict(Xt)
        times.append(time.perf_counter() - start)
        if r == 0:
            prediction = pred
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, prediction, labels=[0, 1])

    record = dict(config=config, classifier=key, seed=seed, selected_indices=idx,
                  train_seconds=train_seconds, inference_seconds_all=times,
                  inference_median=float(np.median(times)), **metrics_from_cm(cm))
    if key == "xgb":
        record["best_iteration"] = int(clf.best_iteration)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if save_predictions:
        np.save(Path(out_dir) / f"{slug(config)}_{key}_seed{seed}_prediction.npy",
                prediction.astype(np.uint8))
    path.write_text(json.dumps(record, indent=2))
    del clf, Xt
    gc.collect()
    return record


def load_all(out_dir):
    rows = [json.loads(p.read_text()) for p in sorted(Path(out_dir).glob("*_seed*.json"))
            if not p.name.endswith("_search.json")]
    return pd.DataFrame(rows)


def summarise(df, baseline="Baseline", order=None):
    """Mean and standard deviation across seeds, plus time reductions."""
    cols = ["f1", "precision", "recall", "fp", "fn", "total_errors",
            "train_seconds", "inference_median"]
    grouped = df.groupby(["config", "classifier"])[cols]
    mean, std = grouped.mean().add_suffix("_mean"), grouped.std(ddof=1).add_suffix("_sd")
    out = pd.concat([mean, std], axis=1)
    out["n_seeds"] = df.groupby(["config", "classifier"]).size()
    out = out.reset_index()
    for time_col in ["train_seconds", "inference_median"]:
        base = out[out.config == baseline].set_index("classifier")[f"{time_col}_mean"]
        out[f"{time_col}_reduction_pct"] = [
            100 * (base[c] - t) / base[c] for c, t in zip(out.classifier, out[f"{time_col}_mean"])]
    if order:
        out["config"] = pd.Categorical(out["config"], order, ordered=True)
        out = out.sort_values(["classifier", "config"]).reset_index(drop=True)
    return out


def paired_seed_differences(df, config_a, config_b, key, metric="f1"):
    """Per-seed difference (A - B) for one classifier, to see if a ranking is stable."""
    a = df[(df.config == config_a) & (df.classifier == key)].set_index("seed")[metric]
    b = df[(df.config == config_b) & (df.classifier == key)].set_index("seed")[metric]
    return (a - b).dropna()


def latex_summary_table(summary, caption, label):
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Configuration & Model & F1 (mean $\pm$ SD) & Errors (mean $\pm$ SD) & Train (s) & Predict (s) \\",
             r"\midrule"]
    for _, r in summary.iterrows():
        model = "XGB" if r.classifier == "xgb" else "RF"
        sd_f1 = 0.0 if pd.isna(r.f1_sd) else r.f1_sd
        sd_err = 0.0 if pd.isna(r.total_errors_sd) else r.total_errors_sd
        lines.append(f"{r.config} & {model} & {r.f1_mean:.6f} $\\pm$ {sd_f1:.6f} & "
                     f"{r.total_errors_mean:.1f} $\\pm$ {sd_err:.1f} & "
                     f"{r.train_seconds_mean:.1f} & {r.inference_median_mean:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)
