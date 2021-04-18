from kedro.pipeline import Pipeline, node

from .nodes import create_master_table


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=create_master_table,
                inputs=["customers", "transactions", "params:aggregation_params"],
                outputs="master_table",
                name="create_master_table",
            )
        ]
    )
