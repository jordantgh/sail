import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pyspark.sql import Row
from pyspark.sql.functions import col, lit


def test_dataframe_drop(spark):
    df = spark.createDataFrame([(14, "Tom"), (23, "Alice"), (16, "Bob")], ["age", "name"])
    df2 = spark.createDataFrame([Row(height=80, name="Tom"), Row(height=85, name="Bob")])

    assert_frame_equal(
        df.drop("age").sort("name").toPandas(),
        pd.DataFrame({"name": ["Alice", "Bob", "Tom"]}),
    )
    assert_frame_equal(
        df.drop(df.age).sort("name").toPandas(),
        pd.DataFrame({"name": ["Alice", "Bob", "Tom"]}),
    )

    assert_frame_equal(
        df.join(df2, df.name == df2.name, "inner").drop("name").sort("age").toPandas(),
        pd.DataFrame({"age": [14, 16], "height": [80, 85]}),
    )

    df3 = df.join(df2)
    assert_frame_equal(
        df3.select(
            df["age"],
            df["name"].alias("name_left"),
            df2["height"],
            df2["name"].alias("name_right"),
        )
        .sort("name_left", "name_right")
        .toPandas(),
        pd.DataFrame(
            {
                "age": [23, 23, 16, 16, 14, 14],
                "name_left": ["Alice", "Alice", "Bob", "Bob", "Tom", "Tom"],
                "height": [85, 80, 85, 80, 85, 80],
                "name_right": ["Bob", "Tom", "Bob", "Tom", "Bob", "Tom"],
            }
        ),
    )

    assert_frame_equal(
        df3.drop("name").sort("age", "height").toPandas(),
        pd.DataFrame({"age": [14, 14, 16, 16, 23, 23], "height": [80, 85, 80, 85, 80, 85]}),
    )

    with pytest.raises(Exception, match="AMBIGUOUS_REFERENCE"):
        df3.drop(col("name")).toPandas()

    df4 = df.withColumn("a.b.c", lit(1))
    assert_frame_equal(
        df4.sort("age").toPandas(),
        pd.DataFrame({"age": [14, 16, 23], "name": ["Tom", "Bob", "Alice"], "a.b.c": [1, 1, 1]}).astype(
            {"a.b.c": "int32"}
        ),
    )

    assert_frame_equal(
        df4.drop("a.b.c").sort("age").toPandas(),
        pd.DataFrame({"age": [14, 16, 23], "name": ["Tom", "Bob", "Alice"]}),
    )

    assert_frame_equal(
        df4.drop(col("a.b.c")).sort("age").toPandas(),
        pd.DataFrame({"age": [14, 16, 23], "name": ["Tom", "Bob", "Alice"], "a.b.c": [1, 1, 1]}).astype(
            {"a.b.c": "int32"}
        ),
    )


@pytest.mark.parametrize("method_name", ["checkpoint", "localCheckpoint"])
def test_checkpoint_materializes_temp_view_input(spark, method_name):
    source = spark.createDataFrame([(1, "alpha"), (2, "beta"), (3, "gamma")], ["id", "value"])
    source.createOrReplaceTempView("checkpoint_source")

    df = spark.table("checkpoint_source").where(col("id") >= 2)
    checkpointed = getattr(df, method_name)()

    spark.catalog.dropTempView("checkpoint_source")

    assert_frame_equal(
        checkpointed.orderBy("id").toPandas(),
        pd.DataFrame({"id": [2, 3], "value": ["beta", "gamma"]}).astype({"id": "int64"}),
    )

    with pytest.raises(Exception, match=r"TABLE_OR_VIEW_NOT_FOUND|not found|unknown"):
        df.collect()


@pytest.mark.parametrize("method_name", ["checkpoint", "localCheckpoint"])
def test_checkpoint_lazy_matches_doctest_and_executes(spark, method_name):
    source = spark.createDataFrame([(14, "Tom"), (23, "Alice"), (16, "Bob")], ["age", "name"])
    source.createOrReplaceTempView("lazy_checkpoint_source")

    df = spark.table("lazy_checkpoint_source")
    checkpointed = getattr(df, method_name)(False)

    expected = pd.DataFrame({"age": [14, 16, 23], "name": ["Tom", "Bob", "Alice"]})

    assert repr(checkpointed) == "DataFrame[age: bigint, name: string]"
    assert_frame_equal(
        checkpointed.orderBy("age").toPandas(),
        expected,
    )

    spark.catalog.dropTempView("lazy_checkpoint_source")

    assert_frame_equal(checkpointed.orderBy("age").toPandas(), expected)

    with pytest.raises(Exception, match=r"TABLE_OR_VIEW_NOT_FOUND|not found|unknown"):
        df.collect()
