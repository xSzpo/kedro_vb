from kedro.pipeline import Pipeline, node

from .nodes import create_master_table, split_data


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=create_master_table,
                inputs=["customers", "transactions", "params:aggregation_params"],
                outputs="master_table",
                name="create_master_table",
            ),
        ]
    )


def create_splits(**kwargs):
    return Pipeline(
        [

            node(
                func=split_data,
                inputs=["master_table", "params:data_set"],
                outputs=["df_oot", "df_train", "df_test", "df_valid"],
                name="split_data",
            )
        ]
    )
