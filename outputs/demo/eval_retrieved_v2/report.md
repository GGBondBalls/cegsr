# Run Summary

## Aggregate Metrics
- **num_episodes**: 400
- **accuracy**: 0.675
- **exact_match**: 0.35
- **mcq_accuracy**: 0.675
- **repair_coverage**: 0.0
- **repair_success_rate**: 0.0
- **num_changed_repairs**: 0
- **average_trajectory_length**: 4.0
- **average_input_tokens**: 2136.08
- **average_output_tokens**: 243.25
- **retrieval_hit_usefulness_proxy**: 0.675
- **graph_num_nodes**: 1261
- **graph_num_edges**: 48014
- **training_data_size_by_role::planner**: 400
- **training_data_size_by_role::solver**: 400
- **training_data_size_by_role::verifier**: 400
- **training_data_size_by_role::summarizer**: 400
- **dataset_accuracy::commonsense_qa**: 0.72
- **dataset_count::commonsense_qa**: 100
- **dataset_accuracy::ai2_arc**: 0.71
- **dataset_count::ai2_arc**: 100
- **dataset_accuracy::boolq**: 0.63
- **dataset_count::boolq**: 100
- **dataset_accuracy::pubmed_qa**: 0.64
- **dataset_count::pubmed_qa**: 100
- **category_accuracy::commonsense**: 0.72
- **category_accuracy::science_mcq**: 0.71
- **category_accuracy::reading_comprehension_yesno**: 0.63
- **category_accuracy::biomedical_qa**: 0.64

## Dataset Breakdown
- commonsense_qa: 0.72
- ai2_arc: 0.71
- boolq: 0.63
- pubmed_qa: 0.64

## Error Cases
- sample_id=commonsenseqa_validation_0 | dataset=commonsense_qa | pred=E. hall | gold=D. living room
- sample_id=commonsenseqa_validation_11 | dataset=commonsense_qa | pred=A. ocean | gold=C. water
- sample_id=commonsenseqa_validation_16 | dataset=commonsense_qa | pred=E. property | gold=B. neighborhood
- sample_id=commonsenseqa_validation_18 | dataset=commonsense_qa | pred=B. theater | gold=E. school
- sample_id=commonsenseqa_validation_20 | dataset=commonsense_qa | pred=D. walk | gold=A. listen to radio
- sample_id=commonsenseqa_validation_24 | dataset=commonsense_qa | pred=A. surprise | gold=C. annoyance
- sample_id=commonsenseqa_validation_27 | dataset=commonsense_qa | pred=D. gain weight. | gold=C. augment
- sample_id=commonsenseqa_validation_29 | dataset=commonsense_qa | pred=C. prehistory | gold=E. ancient times
- sample_id=commonsenseqa_validation_31 | dataset=commonsense_qa | pred=A. music school | gold=C. neighbor's house
- sample_id=commonsenseqa_validation_34 | dataset=commonsense_qa | pred=E. direct sunlight | gold=A. full sunlight
- sample_id=commonsenseqa_validation_37 | dataset=commonsense_qa | pred=C. buying vegetables | gold=A. buy food
- sample_id=commonsenseqa_validation_39 | dataset=commonsense_qa | pred=C. satisfaction | gold=D. confident
- sample_id=commonsenseqa_validation_45 | dataset=commonsense_qa | pred=B. better themselves | gold=A. face problems
- sample_id=commonsenseqa_validation_60 | dataset=commonsense_qa | pred=B. internet | gold=D. library
- sample_id=commonsenseqa_validation_61 | dataset=commonsense_qa | pred=D. education | gold=A. theater
- sample_id=commonsenseqa_validation_62 | dataset=commonsense_qa | pred=C. getting wet | gold=B. see beautiful views
- sample_id=commonsenseqa_validation_68 | dataset=commonsense_qa | pred=A. serious | gold=C. musical
- sample_id=commonsenseqa_validation_72 | dataset=commonsense_qa | pred=E. heirlooms | gold=A. family tree
- sample_id=commonsenseqa_validation_77 | dataset=commonsense_qa | pred=D. teach | gold=B. advance knowledge
- sample_id=commonsenseqa_validation_81 | dataset=commonsense_qa | pred=E. arousal | gold=C. shortness of breath
