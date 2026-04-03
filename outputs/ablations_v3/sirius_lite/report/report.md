# Run Summary

## Aggregate Metrics
- **num_episodes**: 400
- **accuracy**: 0.7875
- **exact_match**: 0.36
- **mcq_accuracy**: 0.7875
- **repair_coverage**: 0.0
- **repair_success_rate**: 0.0
- **num_changed_repairs**: 0
- **repair_flip_to_success**: 0
- **average_trajectory_length**: 4.0
- **average_input_tokens**: 2054.33
- **average_output_tokens**: 256.5275
- **retrieval_hit_usefulness_proxy**: 0.0
- **graph_num_nodes**: 0
- **graph_num_edges**: 0
- **training_data_size_by_role::planner**: 400
- **training_data_size_by_role::solver**: 400
- **training_data_size_by_role::verifier**: 400
- **training_data_size_by_role::summarizer**: 400
- **dataset_accuracy::commonsense_qa**: 0.81
- **dataset_count::commonsense_qa**: 100
- **dataset_accuracy::ai2_arc**: 0.8
- **dataset_count::ai2_arc**: 100
- **dataset_accuracy::boolq**: 0.75
- **dataset_count::boolq**: 100
- **dataset_accuracy::pubmed_qa**: 0.79
- **dataset_count::pubmed_qa**: 100
- **category_accuracy::commonsense**: 0.81
- **category_accuracy::science_mcq**: 0.8
- **category_accuracy::reading_comprehension_yesno**: 0.75
- **category_accuracy::biomedical_qa**: 0.79

## Dataset Breakdown
- commonsense_qa: 0.81
- ai2_arc: 0.8
- boolq: 0.75
- pubmed_qa: 0.79

## Error Cases
- sample_id=commonsenseqa_validation_0 | dataset=commonsense_qa | pred=E. hall | gold=D. living room
- sample_id=commonsenseqa_validation_4 | dataset=commonsense_qa | pred=E. porch. | gold=B. front door
- sample_id=commonsenseqa_validation_5 | dataset=commonsense_qa | pred=C. kitchen cupboard | gold=B. table setting
- sample_id=commonsenseqa_validation_11 | dataset=commonsense_qa | pred=A. ocean | gold=C. water
- sample_id=commonsenseqa_validation_20 | dataset=commonsense_qa | pred=D. walk | gold=A. listen to radio
- sample_id=commonsenseqa_validation_27 | dataset=commonsense_qa | pred=E. expand | gold=C. augment
- sample_id=commonsenseqa_validation_29 | dataset=commonsense_qa | pred=D. prehistoric times | gold=E. ancient times
- sample_id=commonsenseqa_validation_34 | dataset=commonsense_qa | pred=E. direct sunlight | gold=A. full sunlight
- sample_id=commonsenseqa_validation_39 | dataset=commonsense_qa | pred=E. pride | gold=D. confident
- sample_id=commonsenseqa_validation_42 | dataset=commonsense_qa | pred=D. teardrops. | gold=B. snowflake
- sample_id=commonsenseqa_validation_45 | dataset=commonsense_qa | pred=B. better themselves | gold=A. face problems
- sample_id=commonsenseqa_validation_62 | dataset=commonsense_qa | pred=E. murdered by a landshark | gold=B. see beautiful views
- sample_id=commonsenseqa_validation_77 | dataset=commonsense_qa | pred=E. follow instructions | gold=B. advance knowledge
- sample_id=commonsenseqa_validation_79 | dataset=commonsense_qa | pred=A. England | gold=B. new hampshire
- sample_id=commonsenseqa_validation_81 | dataset=commonsense_qa | pred=E. arousal | gold=C. shortness of breath
- sample_id=commonsenseqa_validation_86 | dataset=commonsense_qa | pred=A. gap | gold=B. shopping mall
- sample_id=commonsenseqa_validation_88 | dataset=commonsense_qa | pred=D. shower stall | gold=A. bathtub
- sample_id=commonsenseqa_validation_92 | dataset=commonsense_qa | pred=B. lamb | gold=C. done
- sample_id=commonsenseqa_validation_93 | dataset=commonsense_qa | pred=E. opera | gold=A. confession
- sample_id=arc_validation_6 | dataset=ai2_arc | pred=D. an organ. | gold=A. a cell
