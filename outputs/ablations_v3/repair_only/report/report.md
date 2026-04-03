# Run Summary

## Aggregate Metrics
- **num_episodes**: 400
- **accuracy**: 0.7025
- **exact_match**: 0.3275
- **mcq_accuracy**: 0.7025
- **repair_coverage**: 0.3425
- **repair_success_rate**: 0.1314
- **num_changed_repairs**: 137
- **repair_flip_to_success**: 10
- **average_trajectory_length**: 4.0
- **average_input_tokens**: 1858.435
- **average_output_tokens**: 251.4175
- **retrieval_hit_usefulness_proxy**: 0.0
- **graph_num_nodes**: 0
- **graph_num_edges**: 0
- **repair_target_count::solver**: 129
- **repair_target_count::verifier**: 8
- **training_data_size_by_role::planner**: 400
- **training_data_size_by_role::solver**: 400
- **training_data_size_by_role::verifier**: 400
- **training_data_size_by_role::summarizer**: 400
- **dataset_accuracy::commonsense_qa**: 0.76
- **dataset_count::commonsense_qa**: 100
- **dataset_accuracy::ai2_arc**: 0.72
- **dataset_count::ai2_arc**: 100
- **dataset_accuracy::boolq**: 0.64
- **dataset_count::boolq**: 100
- **dataset_accuracy::pubmed_qa**: 0.69
- **dataset_count::pubmed_qa**: 100
- **category_accuracy::commonsense**: 0.76
- **category_accuracy::science_mcq**: 0.72
- **category_accuracy::reading_comprehension_yesno**: 0.64
- **category_accuracy::biomedical_qa**: 0.69

## Dataset Breakdown
- commonsense_qa: 0.76
- ai2_arc: 0.72
- boolq: 0.64
- pubmed_qa: 0.69

## Error Cases
- sample_id=commonsenseqa_validation_0 | dataset=commonsense_qa | pred=E. hall | gold=D. living room
- sample_id=commonsenseqa_validation_1 | dataset=commonsense_qa | pred=E. cabinet | gold=D. cabinet
- sample_id=commonsenseqa_validation_5 | dataset=commonsense_qa | pred=C. kitchen cupboard | gold=B. table setting
- sample_id=commonsenseqa_validation_6 | dataset=commonsense_qa | pred=E. Actors | gold=A. theater
- sample_id=commonsenseqa_validation_11 | dataset=commonsense_qa | pred=A. ocean | gold=C. water
- sample_id=commonsenseqa_validation_15 | dataset=commonsense_qa | pred=D. enjoyable | gold=B. watching television
- sample_id=commonsenseqa_validation_20 | dataset=commonsense_qa | pred=D. walk. | gold=A. listen to radio
- sample_id=commonsenseqa_validation_23 | dataset=commonsense_qa | pred=B. stress | gold=C. happiness
- sample_id=commonsenseqa_validation_27 | dataset=commonsense_qa | pred=E. expand | gold=C. augment
- sample_id=commonsenseqa_validation_28 | dataset=commonsense_qa | pred=E. hurt feelings | gold=C. resentment
- sample_id=commonsenseqa_validation_29 | dataset=commonsense_qa | pred=D. prehistoric times | gold=E. ancient times
- sample_id=commonsenseqa_validation_34 | dataset=commonsense_qa | pred=E. direct sunlight | gold=A. full sunlight
- sample_id=commonsenseqa_validation_39 | dataset=commonsense_qa | pred=E. pride | gold=D. confident
- sample_id=commonsenseqa_validation_42 | dataset=commonsense_qa | pred=D. teardrops. | gold=B. snowflake
- sample_id=commonsenseqa_validation_43 | dataset=commonsense_qa | pred=C. winch | gold=D. construction site
- sample_id=commonsenseqa_validation_45 | dataset=commonsense_qa | pred=B. better themselves | gold=A. face problems
- sample_id=commonsenseqa_validation_60 | dataset=commonsense_qa | pred=B. internet | gold=D. library
- sample_id=commonsenseqa_validation_62 | dataset=commonsense_qa | pred=E. murdered by a landshark. | gold=B. see beautiful views
- sample_id=commonsenseqa_validation_77 | dataset=commonsense_qa | pred=E. follow instructions | gold=B. advance knowledge
- sample_id=commonsenseqa_validation_79 | dataset=commonsense_qa | pred=A. England | gold=B. new hampshire
