from kedro.pipeline import Pipeline, node
from .nodes import hp_tuning

def create_pipeline(**kwargs):
    return Pipeline(
        [

            node(
                func=hp_tuning,
                inputs=["df_train", "df_test", "df_valid", "params:hp_params"],
                outputs="model_lgb_params",
                name="hp_tuning"
            )
        ]
    )