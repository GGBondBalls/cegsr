# Run Summary

## Aggregate Metrics
- **num_episodes**: 400
- **accuracy**: 0.6375
- **exact_match**: 0.3175
- **mcq_accuracy**: 0.6375
- **repair_coverage**: 0.0
- **repair_success_rate**: 0.0
- **num_changed_repairs**: 0
- **average_trajectory_length**: 4.0
- **average_input_tokens**: 2398.2475
- **average_output_tokens**: 254.47
- **retrieval_hit_usefulness_proxy**: 0.6375
- **graph_num_nodes**: 1261
- **graph_num_edges**: 45394
- **training_data_size_by_role::planner**: 400
- **training_data_size_by_role::solver**: 400
- **training_data_size_by_role::verifier**: 400
- **training_data_size_by_role::summarizer**: 400
- **dataset_accuracy::commonsense_qa**: 0.66
- **dataset_count::commonsense_qa**: 100
- **dataset_accuracy::ai2_arc**: 0.66
- **dataset_count::ai2_arc**: 100
- **dataset_accuracy::boolq**: 0.59
- **dataset_count::boolq**: 100
- **dataset_accuracy::pubmed_qa**: 0.64
- **dataset_count::pubmed_qa**: 100
- **category_accuracy::commonsense**: 0.66
- **category_accuracy::science_mcq**: 0.66
- **category_accuracy::reading_comprehension_yesno**: 0.59
- **category_accuracy::biomedical_qa**: 0.64

## Dataset Breakdown
- commonsense_qa: 0.66
- ai2_arc: 0.66
- boolq: 0.59
- pubmed_qa: 0.64

## Error Cases
- sample_id=commonsenseqa_validation_0 | dataset=commonsense_qa | pred=E. hall | gold=D. living room
- sample_id=commonsenseqa_validation_4 | dataset=commonsense_qa | pred=D. street corner | gold=B. front door
- sample_id=commonsenseqa_validation_5 | dataset=commonsense_qa | pred=C. kitchen cupboard | gold=B. table setting
- sample_id=commonsenseqa_validation_7 | dataset=commonsense_qa | pred=D. front door | gold=C. own home
- sample_id=commonsenseqa_validation_9 | dataset=commonsense_qa | pred=B. silence | gold=D. stillness
- sample_id=commonsenseqa_validation_11 | dataset=commonsense_qa | pred=B. found in ocean | gold=C. water
- sample_id=commonsenseqa_validation_12 | dataset=commonsense_qa | pred=D. countryside | gold=C. forest
- sample_id=commonsenseqa_validation_13 | dataset=commonsense_qa | pred=C. throwing | gold=A. grab
- sample_id=commonsenseqa_validation_19 | dataset=commonsense_qa | pred=B. unable | gold=C. unlikely
- sample_id=commonsenseqa_validation_20 | dataset=commonsense_qa | pred=D. walk | gold=A. listen to radio
- sample_id=commonsenseqa_validation_23 | dataset=commonsense_qa | pred=B. stress | gold=C. happiness
- sample_id=commonsenseqa_validation_27 | dataset=commonsense_qa | pred=B. make larger | gold=C. augment
- sample_id=commonsenseqa_validation_29 | dataset=commonsense_qa | pred=C. prehistory | gold=E. ancient times
- sample_id=commonsenseqa_validation_33 | dataset=commonsense_qa | pred=A. open eyes | gold=B. go to movies
- sample_id=commonsenseqa_validation_34 | dataset=commonsense_qa | pred=E. direct sunlight | gold=A. full sunlight
- sample_id=commonsenseqa_validation_39 | dataset=commonsense_qa | pred=E. pride | gold=D. confident
- sample_id=commonsenseqa_validation_45 | dataset=commonsense_qa | pred=B. better themselves | gold=A. face problems
- sample_id=commonsenseqa_validation_46 | dataset=commonsense_qa | pred=C. laboratory | gold=E. industrial area
- sample_id=commonsenseqa_validation_50 | dataset=commonsense_qa | pred=C. being annoyed | gold=A. frustration
- sample_id=commonsenseqa_validation_60 | dataset=commonsense_qa | pred=B. internet | gold=D. library
