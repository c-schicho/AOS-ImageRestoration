import os
from typing import List

import pandas as pd
from tbparse import SummaryReader


def read_results(result_path: str, run_ids: List[str]) -> pd.DataFrame:
    results = pd.DataFrame(columns=["run_id", "MSE", "PSNR", "SSIM"])

    run_result_paths = [os.path.join(result_path, run_id) for run_id in run_ids]

    for path in run_result_paths:
        if not os.path.exists(path):
            raise ValueError(f"result path [{path}] does not exist")

    for idx, run_result_path in enumerate(run_result_paths):
        mse_value = __get_value_for_metric(run_result_path, "mse")
        psnr_value = __get_value_for_metric(run_result_path, "psnr")
        ssim_value = __get_value_for_metric(run_result_path, "ssim")
        results.loc[idx] = [run_ids[idx], mse_value, psnr_value, ssim_value]

    return results


def __get_value_for_metric(run_path: str, metric: str) -> float:
    df = SummaryReader(os.path.join(run_path, "eval_metrics_" + metric)).scalars
    return df['value'].iloc[-1]
