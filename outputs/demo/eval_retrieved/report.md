# Run Summary

## Aggregate Metrics
- **num_episodes**: 400
- **accuracy**: 0.59
- **exact_match**: 0.2825
- **mcq_accuracy**: 0.59
- **repair_coverage**: 0.0
- **repair_success_rate**: 0.0
- **num_changed_repairs**: 0
- **average_trajectory_length**: 4.0
- **average_input_tokens**: 2333.415
- **average_output_tokens**: 260.4425
- **retrieval_hit_usefulness_proxy**: 0.59
- **graph_num_nodes**: 1261
- **graph_num_edges**: 48014
- **training_data_size_by_role::planner**: 400
- **training_data_size_by_role::solver**: 400
- **training_data_size_by_role::verifier**: 400
- **training_data_size_by_role::summarizer**: 400
- **dataset_accuracy::commonsense_qa**: 0.64
- **dataset_count::commonsense_qa**: 100
- **dataset_accuracy::ai2_arc**: 0.66
- **dataset_count::ai2_arc**: 100
- **dataset_accuracy::boolq**: 0.54
- **dataset_count::boolq**: 100
- **dataset_accuracy::pubmed_qa**: 0.52
- **dataset_count::pubmed_qa**: 100
- **category_accuracy::commonsense**: 0.64
- **category_accuracy::science_mcq**: 0.66
- **category_accuracy::reading_comprehension_yesno**: 0.54
- **category_accuracy::biomedical_qa**: 0.52

## Dataset Breakdown
- commonsense_qa: 0.64
- ai2_arc: 0.66
- boolq: 0.54
- pubmed_qa: 0.52

## Error Cases
- sample_id=commonsenseqa_validation_0 | dataset=commonsense_qa | pred=E. hall | gold=D. living room
- sample_id=commonsenseqa_validation_1 | dataset=commonsense_qa | pred=C. cave | gold=D. cabinet
- sample_id=commonsenseqa_validation_5 | dataset=commonsense_qa | pred=C. kitchen cupboard | gold=B. table setting
- sample_id=commonsenseqa_validation_9 | dataset=commonsense_qa | pred=C. stationary | gold=D. stillness
- sample_id=commonsenseqa_validation_11 | dataset=commonsense_qa | pred=A. ocean | gold=C. water
- sample_id=commonsenseqa_validation_13 | dataset=commonsense_qa | pred=E. may fall | gold=A. grab
- sample_id=commonsenseqa_validation_20 | dataset=commonsense_qa | pred=D. walk | gold=A. listen to radio
- sample_id=commonsenseqa_validation_23 | dataset=commonsense_qa | pred=B. Stress | gold=C. happiness
- sample_id=commonsenseqa_validation_24 | dataset=commonsense_qa | pred=B. fight | gold=C. annoyance
- sample_id=commonsenseqa_validation_27 | dataset=commonsense_qa | pred=B. make larger | gold=C. augment
- sample_id=commonsenseqa_validation_28 | dataset=commonsense_qa | pred=E. hurt feelings | gold=C. resentment
- sample_id=commonsenseqa_validation_29 | dataset=commonsense_qa | pred=D. prehistoric times | gold=E. ancient times
- sample_id=commonsenseqa_validation_30 | dataset=commonsense_qa | pred=C. Life on Earth | gold=B. heat
- sample_id=commonsenseqa_validation_34 | dataset=commonsense_qa | pred=E. direct sunlight | gold=A. full sunlight
- sample_id=commonsenseqa_validation_37 | dataset=commonsense_qa | pred=B. see other people | gold=A. buy food
- sample_id=commonsenseqa_validation_39 | dataset=commonsense_qa | pred=C. satisfaction | gold=D. confident
- sample_id=commonsenseqa_validation_43 | dataset=commonsense_qa | pred=B. ship | gold=D. construction site
- sample_id=commonsenseqa_validation_45 | dataset=commonsense_qa | pred=B. better themselves | gold=A. face problems
- sample_id=commonsenseqa_validation_54 | dataset=commonsense_qa | pred=E. Trunk | gold=D. bloom
- sample_id=commonsenseqa_validation_56 | dataset=commonsense_qa | pred=B. start running | gold=E. last several years
