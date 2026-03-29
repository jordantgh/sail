import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pyspark import StorageLevel
from pyspark.sql import Row
from pyspark.sql.functions import col, lit

from pysail.testing.spark.utils.common import pyspark_version

CHECKPOINT_MIN_ID = 2
CHECKPOINT_CONNECT_SUPPORTED = pytest.mark.skipif(
    pyspark_version() < (4,),
    reason="Spark Connect checkpoint APIs require Spark 4+",
)


@pytest.fixture
def checkpoint_dir(spark, tmp_path):
    key = "spark.checkpoint.dir"
    previous = spark.conf.get(key, None)
    path = tmp_path / "checkpoint-dir"
    spark.conf.set(key, path.as_uri())
    try:
        yield path
    finally:
        if previous is None:
            spark.conf.unset(key)
        else:
            spark.conf.set(key, previous)


def _checkpoint_parquet_files(path):
    return sorted(file.relative_to(path).as_posix() for file in path.rglob("*.parquet"))


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


@CHECKPOINT_CONNECT_SUPPORTED
def test_checkpoint_requires_checkpoint_dir(spark):
    key = "spark.checkpoint.dir"
    previous = spark.conf.get(key, None)
    if previous is not None:
        spark.conf.unset(key)
    try:
        df = spark.createDataFrame([(1, "alpha")], ["id", "value"])
        with pytest.raises(Exception, match=r"spark\.checkpoint\.dir"):
            df.checkpoint()
    finally:
        if previous is not None:
            spark.conf.set(key, previous)


@CHECKPOINT_CONNECT_SUPPORTED
def test_checkpoint_materializes_temp_view_input(spark, checkpoint_dir):
    source = spark.createDataFrame([(1, "alpha"), (2, "beta"), (3, "gamma")], ["id", "value"])
    source.createOrReplaceTempView("checkpoint_source")

    df = spark.table("checkpoint_source").where(col("id") >= CHECKPOINT_MIN_ID)
    checkpointed = df.checkpoint()

    spark.catalog.dropTempView("checkpoint_source")

    assert_frame_equal(
        checkpointed.orderBy("id").toPandas(),
        pd.DataFrame({"id": [2, 3], "value": ["beta", "gamma"]}),
        check_dtype=False,
    )
    assert _checkpoint_parquet_files(checkpoint_dir)

    with pytest.raises(Exception, match=r"TABLE_OR_VIEW_NOT_FOUND|not found|unknown"):
        df.collect()


@CHECKPOINT_CONNECT_SUPPORTED
def test_checkpoint_lazy_materializes_once_and_survives_source_drop(spark, checkpoint_dir):
    source = spark.createDataFrame([(1, "alpha"), (2, "beta"), (3, "gamma")], ["id", "value"])
    source.createOrReplaceTempView("lazy_checkpoint_source")

    df = spark.table("lazy_checkpoint_source").where(col("id") >= CHECKPOINT_MIN_ID)
    checkpointed = df.checkpoint(False)

    expected = pd.DataFrame({"id": [2, 3], "value": ["beta", "gamma"]})

    assert repr(checkpointed) == "DataFrame[id: bigint, value: string]"
    assert_frame_equal(
        checkpointed.orderBy("id").toPandas(),
        expected,
        check_dtype=False,
    )

    first_materialization_files = _checkpoint_parquet_files(checkpoint_dir)
    assert first_materialization_files

    spark.catalog.dropTempView("lazy_checkpoint_source")

    assert_frame_equal(
        checkpointed.orderBy("id").toPandas(),
        expected,
        check_dtype=False,
    )
    assert _checkpoint_parquet_files(checkpoint_dir) == first_materialization_files

    with pytest.raises(Exception, match=r"TABLE_OR_VIEW_NOT_FOUND|not found|unknown"):
        df.collect()


@CHECKPOINT_CONNECT_SUPPORTED
def test_local_checkpoint_materializes_temp_view_input(spark):
    source = spark.createDataFrame([(1, "alpha"), (2, "beta"), (3, "gamma")], ["id", "value"])
    source.createOrReplaceTempView("local_checkpoint_source")

    df = spark.table("local_checkpoint_source").where(col("id") >= CHECKPOINT_MIN_ID)
    checkpointed = df.localCheckpoint()

    spark.catalog.dropTempView("local_checkpoint_source")

    assert checkpointed.storageLevel == StorageLevel.MEMORY_AND_DISK
    assert_frame_equal(
        checkpointed.orderBy("id").toPandas(),
        pd.DataFrame({"id": [2, 3], "value": ["beta", "gamma"]}),
        check_dtype=False,
    )

    with pytest.raises(Exception, match=r"TABLE_OR_VIEW_NOT_FOUND|not found|unknown"):
        df.collect()


@CHECKPOINT_CONNECT_SUPPORTED
def test_local_checkpoint_lazy_materializes_once_and_survives_source_drop(spark):
    source = spark.createDataFrame([(1, "alpha"), (2, "beta"), (3, "gamma")], ["id", "value"])
    source.createOrReplaceTempView("lazy_local_checkpoint_source")

    df = spark.table("lazy_local_checkpoint_source").where(col("id") >= CHECKPOINT_MIN_ID)
    checkpointed = df.localCheckpoint(False, storageLevel=StorageLevel.MEMORY_ONLY)

    expected = pd.DataFrame({"id": [2, 3], "value": ["beta", "gamma"]})

    assert checkpointed.storageLevel == StorageLevel.MEMORY_ONLY
    assert repr(checkpointed) == "DataFrame[id: bigint, value: string]"
    assert_frame_equal(
        checkpointed.orderBy("id").toPandas(),
        expected,
        check_dtype=False,
    )

    spark.catalog.dropTempView("lazy_local_checkpoint_source")

    assert_frame_equal(
        checkpointed.orderBy("id").toPandas(),
        expected,
        check_dtype=False,
    )

    with pytest.raises(Exception, match=r"TABLE_OR_VIEW_NOT_FOUND|not found|unknown"):
        df.collect()


@CHECKPOINT_CONNECT_SUPPORTED
def test_local_checkpoint_honors_disk_only_storage_level(spark):
    source = spark.createDataFrame([(1, "alpha"), (2, "beta"), (3, "gamma")], ["id", "value"])
    source.createOrReplaceTempView("disk_local_checkpoint_source")

    df = spark.table("disk_local_checkpoint_source").where(col("id") >= CHECKPOINT_MIN_ID)
    checkpointed = df.localCheckpoint(storageLevel=StorageLevel.DISK_ONLY)

    spark.catalog.dropTempView("disk_local_checkpoint_source")

    assert checkpointed.storageLevel == StorageLevel.DISK_ONLY
    assert_frame_equal(
        checkpointed.orderBy("id").toPandas(),
        pd.DataFrame({"id": [2, 3], "value": ["beta", "gamma"]}),
        check_dtype=False,
    )

    with pytest.raises(Exception, match=r"TABLE_OR_VIEW_NOT_FOUND|not found|unknown"):
        df.collect()


@CHECKPOINT_CONNECT_SUPPORTED
def test_local_checkpoint_rejects_unsupported_storage_levels(spark):
    df = spark.createDataFrame([(1, "alpha")], ["id", "value"])

    with pytest.raises(Exception, match=r"replication"):
        df.localCheckpoint(storageLevel=StorageLevel.MEMORY_ONLY_2)

    with pytest.raises(Exception, match=r"offHeap|OffHeap"):
        df.localCheckpoint(storageLevel=StorageLevel.OFF_HEAP)
