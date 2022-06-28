import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score

def calculate_average_precision(replicate_group_df: pd.DataFrame):
    # compute average precision using similarity as a prediction score
    # AP is computed for each drug and the ground truth values are
    # binary with True, if the drug is in the same class
    yscore = replicate_group_df.similarity_metric.values
    yscore[yscore < 0] = 0
    ytrue = replicate_group_df.group_replicate.values
    ap = average_precision_score(y_true=ytrue, y_score=yscore)
    return pd.Series({'AP': ap})


def calculate_precision_recall(replicate_group_df: pd.DataFrame, k) -> pd.Series:
    """Given an elongated pairwise correlation dataframe of replicate groups, calculate
    precision and recall.
    Usage: Designed to be called within a pandas.DataFrame().groupby().apply(). See
    :py:func:`cytominer_eval.operations.precision_recall.precision_recall`.
    Parameters
    ----------
    replicate_group_df : pandas.DataFrame
        An elongated dataframe storing pairwise correlations of all profiles to a single
        replicate group.
    k : int or "R"
        an integer indicating how many pairwise comparisons to threshold.
        if k = "R", then calculate precision at R
    Returns
    -------
    dict
        A return bundle of identifiers (k) and results (precision and recall at k).
        The dictionary has keys ("k", "precision", "recall").
    """
    assert (
        "group_replicate" in replicate_group_df.columns
    ), "'group_replicate' not found in dataframe; remember to call assign_replicates()."

    recall_denom__total_relevant_items = sum(replicate_group_df.group_replicate)

    if k != "R":
        precision_denom__num_recommended_items = k

        num_recommended_items_at_k = sum(
            replicate_group_df.iloc[
                :k,
            ].group_replicate
        )

        precision_at_k = num_recommended_items_at_k / precision_denom__num_recommended_items
        recall_at_k = num_recommended_items_at_k / recall_denom__total_relevant_items     
        return_bundle = {"k": k, "precision": precision_at_k, "recall": recall_at_k}

        return pd.Series(return_bundle)
    else:
        R = recall_denom__total_relevant_items

        num_recommended_items_at_k = sum(
            replicate_group_df.iloc[
            :R,
            ].group_replicate
        )

        precision_at_k = num_recommended_items_at_k / R

        return_bundle = {"R": R, "precision": precision_at_k}

        return pd.Series(return_bundle)
