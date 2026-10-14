"""LightGBM regressor built from a team config."""


def build_regressor(cfg):
    from lightgbm import LGBMRegressor

    model_cfg = cfg["model"]
    return LGBMRegressor(
        n_estimators=model_cfg["n_estimators"],
        learning_rate=model_cfg["learning_rate"],
        num_leaves=model_cfg["num_leaves"],
        min_child_samples=model_cfg["min_child_samples"],
        random_state=model_cfg["random_state"],
        n_jobs=model_cfg["n_jobs"],
        verbosity=model_cfg["verbosity"],
    )
