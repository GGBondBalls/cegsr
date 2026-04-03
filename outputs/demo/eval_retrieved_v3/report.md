# Run Summary

## Aggregate Metrics
- **num_episodes**: 400
- **accuracy**: 0.635
- **exact_match**: 0.34
- **mcq_accuracy**: 0.635
- **repair_coverage**: 0.0
- **repair_success_rate**: 0.0
- **num_changed_repairs**: 0
- **average_trajectory_length**: 4.0
- **average_input_tokens**: 2141.2
- **average_output_tokens**: 244.3925
- **retrieval_hit_usefulness_proxy**: 0.635
- **graph_num_nodes**: 1261
- **graph_num_edges**: 46234
- **training_data_size_by_role::planner**: 400
- **training_data_size_by_role::solver**: 400
- **training_data_size_by_role::verifier**: 400
- **training_data_size_by_role::summarizer**: 400
- **dataset_accuracy::commonsense_qa**: 0.74
- **dataset_count::commonsense_qa**: 100
- **dataset_accuracy::ai2_arc**: 0.68
- **dataset_count::ai2_arc**: 100
- **dataset_accuracy::boolq**: 0.61
- **dataset_count::boolq**: 100
- **dataset_accuracy::pubmed_qa**: 0.51
- **dataset_count::pubmed_qa**: 100
- **category_accuracy::commonsense**: 0.74
- **category_accuracy::science_mcq**: 0.68
- **category_accuracy::reading_comprehension_yesno**: 0.61
- **category_accuracy::biomedical_qa**: 0.51

## Dataset Breakdown
- commonsense_qa: 0.74
- ai2_arc: 0.68
- boolq: 0.61
- pubmed_qa: 0.51

## Error Cases
- sample_id=commonsenseqa_validation_1 | dataset=commonsense_qa | pred=C. cave | gold=D. cabinet
- sample_id=commonsenseqa_validation_5 | dataset=commonsense_qa | pred=C. kitchen cupboard | gold=B. table setting
- sample_id=commonsenseqa_validation_9 | dataset=commonsense_qa | pred=B. silence | gold=D. stillness
- sample_id=commonsenseqa_validation_11 | dataset=commonsense_qa | pred=A. ocean | gold=C. water
- sample_id=commonsenseqa_validation_13 | dataset=commonsense_qa | pred=C. throw | gold=A. grab
- sample_id=commonsenseqa_validation_18 | dataset=commonsense_qa | pred=B. theater | gold=E. school
- sample_id=commonsenseqa_validation_20 | dataset=commonsense_qa | pred=D. walk | gold=A. listen to radio
- sample_id=commonsenseqa_validation_27 | dataset=commonsense_qa | pred=B. make larger | gold=C. augment
- sample_id=commonsenseqa_validation_29 | dataset=commonsense_qa | pred=C. prehistory | gold=E. ancient times
- sample_id=commonsenseqa_validation_31 | dataset=commonsense_qa | pred=A. music school | gold=C. neighbor's house
- sample_id=commonsenseqa_validation_33 | dataset=commonsense_qa | pred=A. open eyes | gold=B. go to movies
- sample_id=commonsenseqa_validation_39 | dataset=commonsense_qa | pred=C. satisfaction | gold=D. confident
- sample_id=commonsenseqa_validation_45 | dataset=commonsense_qa | pred=B. better themselves | gold=A. face problems
- sample_id=commonsenseqa_validation_53 | dataset=commonsense_qa | pred=A. running out of money | gold=E. overstocking
- sample_id=commonsenseqa_validation_56 | dataset=commonsense_qa | pred=B. start running | gold=E. last several years
- sample_id=commonsenseqa_validation_58 | dataset=commonsense_qa | pred=C. garden | gold=E. captivity
- sample_id=commonsenseqa_validation_65 | dataset=commonsense_qa | pred=A. Massachusetts | gold=B. new england
- sample_id=commonsenseqa_validation_68 | dataset=commonsense_qa | pred=B. longplay | gold=C. musical
- sample_id=commonsenseqa_validation_73 | dataset=commonsense_qa | pred=C. garments | gold=D. expensive clothing
- sample_id=commonsenseqa_validation_79 | dataset=commonsense_qa | pred=A. England | gold=B. new hampshire
