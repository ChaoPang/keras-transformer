import os
import itertools

import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql import Window as W

from bert_concept_embeddings.common import *

input_folder = '/data/research_ops/omops/omop_2020q1/'

# +
domain_tables = []
for domain_table_name in ['condition_occurrence', 'drug_exposure', 'procedure_occurrence']:
    domain_tables.append(preprocess_domain_table(spark, input_folder, domain_table_name))
    
patient_event = join_domain_tables(domain_tables)
patient_event = patient_event.where('visit_occurrence_id IS NOT NULL').distinct()
# -

# ## Generate standard input visit based event sequence

take_first = F.udf(lambda rows: [row[0] for row in sorted(rows, key=lambda x: (x[0], x[1]))], T.ArrayType(T.IntegerType()))
take_second = F.udf(lambda rows: [str(row[1]) for row in sorted(rows, key=lambda x: (x[0], x[1]))], T.ArrayType(T.StringType()))

patient_event \
    .where(F.col('date') >= '2017-01-01') \
    .withColumn('date', (F.unix_timestamp('date') / F.lit(24 * 60 * 60 * 7)).cast('int')) \
    .distinct() \
    .withColumn('earliest_visit_date', F.min('date').over(Window.partitionBy('visit_occurrence_id'))) \
    .withColumn('date_concept_id', F.struct(F.col('date'), F.col('standard_concept_id'))) \
    .groupBy('person_id', 'visit_occurrence_id') \
    .agg(F.collect_set('date_concept_id').alias('date_concept_id'),
         F.first('earliest_visit_date').alias('earliest_visit_date')) \
    .withColumn('concept_ids', take_second('date_concept_id')) \
    .withColumn('dates', take_first('date_concept_id')) \
    .withColumn('visit_rank_order', F.dense_rank().over(Window.partitionBy('person_id').orderBy('earliest_visit_date'))) \
    .select('person_id', 'visit_occurrence_id', 'visit_rank_order', 'earliest_visit_date', 'dates', 'concept_ids') \
    .write.mode('overwrite').parquet(os.path.join(input_folder, 'visit_event_sequence_v2'))

# ## Generate Time Sensitive time attention input visit based event sequencew

take_first = F.udf(lambda rows: [row[0] for row in sorted(rows, key=lambda x: (x[0], x[1]))], T.ArrayType(T.IntegerType()))
take_second = F.udf(lambda rows: [str(row[1][0]) for row in sorted(rows, key=lambda x: (x[0], x[1]))], T.ArrayType(T.StringType()))
take_third = F.udf(lambda rows: [row[1][1] for row in sorted(rows, key=lambda x: (x[0], x[1]))], T.ArrayType(T.IntegerType()))

patient_event \
    .where(F.col('date') >= '2017-01-01') \
    .withColumn('period', (F.col('date') >= '2020-01-01').cast('int')) \
    .withColumn('date', (F.unix_timestamp('date') / F.lit(24 * 60 * 60 * 7)).cast('int')) \
    .distinct() \
    .withColumn('earliest_visit_date', F.min('date').over(Window.partitionBy('visit_occurrence_id'))) \
    .withColumn('date_concept_id_period', F.struct(F.col('date'), F.struct(F.col('standard_concept_id'), F.col('period')))) \
    .groupBy('person_id', 'visit_occurrence_id') \
    .agg(F.collect_set('date_concept_id_period').alias('date_concept_id_period'),
         F.first('earliest_visit_date').alias('earliest_visit_date')) \
    .withColumn('concept_ids', take_second('date_concept_id_period')) \
    .withColumn('periods', take_third('date_concept_id_period')) \
    .withColumn('dates', take_first('date_concept_id_period')) \
    .withColumn('visit_rank_order', F.dense_rank().over(Window.partitionBy('person_id').orderBy('earliest_visit_date'))) \
    .select('person_id', 'visit_occurrence_id', 'visit_rank_order', 'earliest_visit_date', 'dates', 'periods', 'concept_ids') \
    .write.mode('overwrite').parquet(os.path.join(input_folder, 'visit_event_sequence_v2'))

# ## Generate training sequences directly using Spark

# +
# patient_event.distinct() \
#     .where(F.col('date') >= '2017-01-01') \
#     .withColumn('date_week', (F.unix_timestamp('date') / F.lit(24 * 60 * 60 * 7)).cast('int')) \
#     .withColumn('row_identifier', F.dense_rank().over(W.partitionBy('person_id').orderBy('date'))) \
#     .createOrReplaceGlobalTempView('patient_event')

# spark.sql('''
#     SELECT
#         p1.person_id,
#         p1.row_identifier,
#         p1.standard_concept_id AS target_concept_id,
#         p1.date_week AS target_time_stamp,
#         p2.standard_concept_id AS context_concept_id,
#         p2.date_week AS context_time_stamp
#     FROM global_temp.patient_event AS p1
#     JOIN global_temp.patient_event AS p2
#         ON p1.person_id = p2.person_id
#             AND p1.date_week BETWEEN (p2.date_week - 50) AND (p2.date_week + 50)
#     WHERE p1.standard_concept_id != p2.standard_concept_id AND p1.date != p2.date
# ''').withColumn('date_concept_id', F.struct(F.col('context_time_stamp'), F.col('context_concept_id'))) \
#     .groupby('person_id', 'row_identifier', 'target_concept_id', 'target_time_stamp') \
#     .agg(F.collect_list('date_concept_id').alias('date_concept_id')) \
#     .withColumn('context_concept_ids', take_second('date_concept_id')) \
#     .withColumn('context_time_stamps', take_first('date_concept_id')) \
#     .select('target_concept_id', 'target_time_stamp', 'context_concept_ids', 'context_time_stamps') \
#     .write.mode('overwrite').parquet(os.path.join(input_folder, 'test_training_data'))

# +
join_collection_udf = F.udf(lambda its: [it for it in sorted(its, key=lambda x: (x[0], x[1]))], T.StructType())

patient_event \
    .groupBy('visit_occurrence_id') \
    .agg(join_collection_udf(F.collect_list(F.struct('date', 'standard_concept_id'))).alias('concept_list'),
         F.size(F.collect_list('standard_concept_id')).alias('collection_size')) \
    .where(F.col('collection_size') > 1).write.mode('overwrite').parquet(
    os.path.join(input_folder, 'visit_event_sequence'))

visit_event_sequence = pd.read_parquet(os.path.join(input_folder, 'visit_event_sequence'))
visit_sequence = visit_event_sequence['concept_list'].apply(lambda seq: list(set(seq.split(' '))))
# -

visit_event_sequence_v2 = pd.read_parquet(os.path.join(input_folder, 'visit_event_sequence_v2'))

for t in visit_event_sequence_v2.itertuples():
    print(t.concept_ids)
    print(t.dates)
    print(t.periods)
    break

visit_event_sequence_v2['concept_id_visit_orders'] = visit_event_sequence_v2['concept_ids'].apply(len) * \
                                                     visit_event_sequence_v2['visit_rank_order'].apply(lambda v: [v])

# +
patient_concept_ids = visit_event_sequence_v2.sort_values(['person_id', 'visit_rank_order']) \
    .groupby('person_id')['concept_ids'].apply(lambda x: list(itertools.chain(*x))).reset_index()

patient_visit_ids = visit_event_sequence_v2.sort_values(['person_id', 'visit_rank_order']) \
    .groupby('person_id')['concept_id_visit_orders'].apply(lambda x: list(itertools.chain(*x))).reset_index()

event_dates = visit_event_sequence_v2.sort_values(['person_id', 'visit_rank_order']) \
    .groupby('person_id')['dates'].apply(lambda x: list(itertools.chain(*x))).reset_index()
# -

training_data = patient_concept_ids.merge(patient_visit_ids).merge(event_dates)
