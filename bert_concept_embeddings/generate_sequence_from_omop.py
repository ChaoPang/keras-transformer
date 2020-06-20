import os
import itertools

import pandas as pd
import numpy as np
from pyspark.sql import functions as F

from bert_concept_embeddings.common import *

input_folder = '/data/research_ops/omops/ohdsi_covid/'

domain_tables = []
for domain_table_name in ['condition_occurrence', 'drug_exposure']:
    domain_tables.append(preprocess_domain_table(spark, input_folder, domain_table_name))

patient_event = join_domain_tables(domain_tables)

patient_event = patient_event.where('visit_occurrence_id IS NOT NULL')

patient_event.cache()

join_collection_udf = F.udf(lambda its: [it for it in sorted(its, key=lambda x: (x[0], x[1]))], T.StructType())

take_first = F.udf(lambda rows: [row[0] for row in sorted(rows, key=lambda x: (x[0], x[1]))], T.ArrayType(T.IntegerType()))
take_second = F.udf(lambda rows: [str(row[1]) for row in sorted(rows, key=lambda x: (x[0], x[1]))], T.ArrayType(T.StringType()))

patient_event \
    .where(F.col('date') >= '1980-01-01') \
    .withColumn('date', (F.unix_timestamp('date') / F.lit(24 * 60 * 60 * 7)).cast('int')) \
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

patient_event \
    .groupBy('visit_occurrence_id') \
    .agg(join_collection_udf(F.collect_list(F.struct('date', 'standard_concept_id'))).alias('concept_list'),
         F.size(F.collect_list('standard_concept_id')).alias('collection_size')) \
    .where(F.col('collection_size') > 1).write.mode('overwrite').parquet(
    os.path.join(input_folder, 'visit_event_sequence'))

visit_event_sequence = pd.read_parquet(os.path.join(input_folder, 'visit_event_sequence'))

visit_sequence = visit_event_sequence['concept_list'].apply(lambda seq: list(set(seq.split(' '))))

visit_event_sequence_v2 = pd.read_parquet(os.path.join(input_folder, 'visit_event_sequence_v2'))

for t in visit_event_sequence_v2.itertuples():
    print(t.concept_ids)
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
