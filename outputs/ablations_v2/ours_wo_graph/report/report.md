# Run Summary

## Aggregate Metrics
- **num_episodes**: 400
- **accuracy**: 0.77
- **exact_match**: 0.36
- **mcq_accuracy**: 0.77
- **repair_coverage**: 0.3425
- **repair_success_rate**: 0.3285
- **num_changed_repairs**: 137
- **average_trajectory_length**: 4.0
- **average_input_tokens**: 1814.5075
- **average_output_tokens**: 252.0625
- **retrieval_hit_usefulness_proxy**: 0.0
- **graph_num_nodes**: 0
- **graph_num_edges**: 0
- **training_data_size_by_role::planner**: 400
- **training_data_size_by_role::solver**: 400
- **training_data_size_by_role::verifier**: 400
- **training_data_size_by_role::summarizer**: 400
- **dataset_accuracy::commonsense_qa**: 0.83
- **dataset_count::commonsense_qa**: 100
- **dataset_accuracy::ai2_arc**: 0.78
- **dataset_count::ai2_arc**: 100
- **dataset_accuracy::boolq**: 0.76
- **dataset_count::boolq**: 100
- **dataset_accuracy::pubmed_qa**: 0.71
- **dataset_count::pubmed_qa**: 100
- **category_accuracy::commonsense**: 0.83
- **category_accuracy::science_mcq**: 0.78
- **category_accuracy::reading_comprehension_yesno**: 0.76
- **category_accuracy::biomedical_qa**: 0.71

## Dataset Breakdown
- commonsense_qa: 0.83
- ai2_arc: 0.78
- boolq: 0.76
- pubmed_qa: 0.71

## Error Cases
- sample_id=commonsenseqa_validation_0 | dataset=commonsense_qa | pred=E. hall | gold=D. living room
- sample_id=commonsenseqa_validation_5 | dataset=commonsense_qa | pred=C. kitchen cupboard. | gold=B. table setting
- sample_id=commonsenseqa_validation_6 | dataset=commonsense_qa | pred=E. actors and actresses | gold=A. theater
- sample_id=commonsenseqa_validation_11 | dataset=commonsense_qa | pred=A. ocean | gold=C. water
- sample_id=commonsenseqa_validation_20 | dataset=commonsense_qa | pred=D. walk | gold=A. listen to radio
- sample_id=commonsenseqa_validation_23 | dataset=commonsense_qa | pred=E. headache | gold=C. happiness
- sample_id=commonsenseqa_validation_27 | dataset=commonsense_qa | pred=E. expand | gold=C. augment
- sample_id=commonsenseqa_validation_29 | dataset=commonsense_qa | pred=D. prehistoric times | gold=E. ancient times
- sample_id=commonsenseqa_validation_34 | dataset=commonsense_qa | pred=E. direct sunlight | gold=A. full sunlight
- sample_id=commonsenseqa_validation_43 | dataset=commonsense_qa | pred=C. winch. | gold=D. construction site
- sample_id=commonsenseqa_validation_45 | dataset=commonsense_qa | pred=B. better themselves | gold=A. face problems
- sample_id=commonsenseqa_validation_60 | dataset=commonsense_qa | pred=B. internet | gold=D. library
- sample_id=commonsenseqa_validation_79 | dataset=commonsense_qa | pred=A. England. | gold=B. new hampshire
- sample_id=commonsenseqa_validation_81 | dataset=commonsense_qa | pred=E. arousal | gold=C. shortness of breath
- sample_id=commonsenseqa_validation_86 | dataset=commonsense_qa | pred=A. gap | gold=B. shopping mall
- sample_id=commonsenseqa_validation_88 | dataset=commonsense_qa | pred=D. shower stall | gold=A. bathtub
- sample_id=commonsenseqa_validation_93 | dataset=commonsense_qa | pred=E. opera | gold=A. confession
- sample_id=arc_validation_2 | dataset=ai2_arc | pred=C. tornados. | gold=A. earthquakes
- sample_id=arc_validation_3 | dataset=ai2_arc | pred=C. clear and colder | gold=D. cloudy and rainy
- sample_id=arc_validation_6 | dataset=ai2_arc | pred=D. an organ | gold=A. a cell
