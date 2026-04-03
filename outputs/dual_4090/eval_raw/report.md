# Run Summary

## Aggregate Metrics
- **num_episodes**: 300
- **accuracy**: 0.8033
- **exact_match**: 0.2467
- **mcq_accuracy**: 0.8033
- **repair_coverage**: 0.0
- **repair_success_rate**: 0.0
- **num_changed_repairs**: 0
- **average_trajectory_length**: 4.0
- **average_input_tokens**: 1809.4233
- **average_output_tokens**: 302.4633
- **retrieval_hit_usefulness_proxy**: 0.0
- **graph_num_nodes**: 0
- **graph_num_edges**: 0
- **training_data_size_by_role::planner**: 300
- **training_data_size_by_role::solver**: 300
- **training_data_size_by_role::verifier**: 300
- **training_data_size_by_role::summarizer**: 300
- **dataset_accuracy::commonsense_qa**: 0.82
- **dataset_count::commonsense_qa**: 100
- **dataset_accuracy::ai2_arc**: 0.85
- **dataset_count::ai2_arc**: 100
- **dataset_accuracy::pubmed_qa**: 0.74
- **dataset_count::pubmed_qa**: 100
- **category_accuracy::commonsense**: 0.82
- **category_accuracy::science_mcq**: 0.85
- **category_accuracy::biomedical_qa**: 0.74

## Dataset Breakdown
- commonsense_qa: 0.82
- ai2_arc: 0.85
- pubmed_qa: 0.74

## Error Cases
- sample_id=commonsenseqa_validation_0 | dataset=commonsense_qa | pred=E. hall | gold=D. living room
- sample_id=commonsenseqa_validation_4 | dataset=commonsense_qa | pred=E. porch. | gold=B. front door
- sample_id=commonsenseqa_validation_11 | dataset=commonsense_qa | pred=A. ocean. | gold=C. water
- sample_id=commonsenseqa_validation_16 | dataset=commonsense_qa | pred=E. property. | gold=B. neighborhood
- sample_id=commonsenseqa_validation_18 | dataset=commonsense_qa | pred=B. theater. | gold=E. school
- sample_id=commonsenseqa_validation_20 | dataset=commonsense_qa | pred=D. walk. | gold=A. listen to radio
- sample_id=commonsenseqa_validation_29 | dataset=commonsense_qa | pred=D. prehistoric times | gold=E. ancient times
- sample_id=commonsenseqa_validation_31 | dataset=commonsense_qa | pred=A. music school | gold=C. neighbor's house
- sample_id=commonsenseqa_validation_34 | dataset=commonsense_qa | pred=E. direct sunlight. | gold=A. full sunlight
- sample_id=commonsenseqa_validation_45 | dataset=commonsense_qa | pred=B. better themselves | gold=A. face problems
- sample_id=commonsenseqa_validation_56 | dataset=commonsense_qa | pred=B. start running | gold=E. last several years
- sample_id=commonsenseqa_validation_60 | dataset=commonsense_qa | pred=B. internet. | gold=D. library
- sample_id=commonsenseqa_validation_62 | dataset=commonsense_qa | pred=A. seeing bear | gold=B. see beautiful views
- sample_id=commonsenseqa_validation_73 | dataset=commonsense_qa | pred=C. garments | gold=D. expensive clothing
- sample_id=commonsenseqa_validation_79 | dataset=commonsense_qa | pred=A. england | gold=B. new hampshire
- sample_id=commonsenseqa_validation_84 | dataset=commonsense_qa | pred=C. his hand. | gold=E. child's hand
- sample_id=commonsenseqa_validation_88 | dataset=commonsense_qa | pred=D. shower stall. | gold=A. bathtub
- sample_id=commonsenseqa_validation_89 | dataset=commonsense_qa | pred=E. tree | gold=D. amazon river
- sample_id=arc_validation_13 | dataset=ai2_arc | pred=C. fewer snakes and fewer birds. | gold=A. more snakes and fewer birds
- sample_id=arc_validation_32 | dataset=ai2_arc | pred=D. Air particles move down and to the right. | gold=B. Air particles move up and to the right.
