# Run Summary

## Aggregate Metrics
- **num_episodes**: 400
- **accuracy**: 0.695
- **exact_match**: 0.3325
- **mcq_accuracy**: 0.695
- **repair_coverage**: 0.0
- **repair_success_rate**: 0.0
- **num_changed_repairs**: 0
- **average_trajectory_length**: 4.0
- **average_input_tokens**: 1754.5525
- **average_output_tokens**: 249.2375
- **retrieval_hit_usefulness_proxy**: 0.0
- **graph_num_nodes**: 0
- **graph_num_edges**: 0
- **training_data_size_by_role::planner**: 400
- **training_data_size_by_role::solver**: 400
- **training_data_size_by_role::verifier**: 400
- **training_data_size_by_role::summarizer**: 400
- **dataset_accuracy::commonsense_qa**: 0.76
- **dataset_count::commonsense_qa**: 100
- **dataset_accuracy::ai2_arc**: 0.69
- **dataset_count::ai2_arc**: 100
- **dataset_accuracy::boolq**: 0.65
- **dataset_count::boolq**: 100
- **dataset_accuracy::pubmed_qa**: 0.68
- **dataset_count::pubmed_qa**: 100
- **category_accuracy::commonsense**: 0.76
- **category_accuracy::science_mcq**: 0.69
- **category_accuracy::reading_comprehension_yesno**: 0.65
- **category_accuracy::biomedical_qa**: 0.68

## Dataset Breakdown
- commonsense_qa: 0.76
- ai2_arc: 0.69
- boolq: 0.65
- pubmed_qa: 0.68

## Error Cases
- sample_id=commonsenseqa_validation_0 | dataset=commonsense_qa | pred=E. hall | gold=D. living room
- sample_id=commonsenseqa_validation_5 | dataset=commonsense_qa | pred=C. kitchen cupboard | gold=B. table setting
- sample_id=commonsenseqa_validation_11 | dataset=commonsense_qa | pred=A. ocean | gold=C. water
- sample_id=commonsenseqa_validation_12 | dataset=commonsense_qa | pred=E. roof. | gold=C. forest
- sample_id=commonsenseqa_validation_20 | dataset=commonsense_qa | pred=D. walk | gold=A. listen to radio
- sample_id=commonsenseqa_validation_23 | dataset=commonsense_qa | pred=B. stress | gold=C. happiness
- sample_id=commonsenseqa_validation_27 | dataset=commonsense_qa | pred=D. gain weight | gold=C. augment
- sample_id=commonsenseqa_validation_29 | dataset=commonsense_qa | pred=D. prehistoric times | gold=E. ancient times
- sample_id=commonsenseqa_validation_33 | dataset=commonsense_qa | pred=A. open eyes | gold=B. go to movies
- sample_id=commonsenseqa_validation_34 | dataset=commonsense_qa | pred=E. direct sunlight | gold=A. full sunlight
- sample_id=commonsenseqa_validation_39 | dataset=commonsense_qa | pred=E. pride | gold=D. confident
- sample_id=commonsenseqa_validation_43 | dataset=commonsense_qa | pred=C. winch | gold=D. construction site
- sample_id=commonsenseqa_validation_45 | dataset=commonsense_qa | pred=B. better themselves | gold=A. face problems
- sample_id=commonsenseqa_validation_60 | dataset=commonsense_qa | pred=B. internet | gold=D. library
- sample_id=commonsenseqa_validation_62 | dataset=commonsense_qa | pred=E. murdered by a landshark. | gold=B. see beautiful views
- sample_id=commonsenseqa_validation_65 | dataset=commonsense_qa | pred=E. Connecticut | gold=B. new england
- sample_id=commonsenseqa_validation_75 | dataset=commonsense_qa | pred=E. staying in place | gold=D. transporting
- sample_id=commonsenseqa_validation_79 | dataset=commonsense_qa | pred=A. England. | gold=B. new hampshire
- sample_id=commonsenseqa_validation_82 | dataset=commonsense_qa | pred=B. count to ten | gold=D. state name
- sample_id=commonsenseqa_validation_86 | dataset=commonsense_qa | pred=A. gap | gold=B. shopping mall
