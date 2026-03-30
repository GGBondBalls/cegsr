# Run Summary

## Aggregate Metrics
- **num_episodes**: 400
- **accuracy**: 0.4525
- **exact_match**: 0.105
- **mcq_accuracy**: 0.4525
- **repair_coverage**: 0.0
- **repair_success_rate**: 0.0
- **num_changed_repairs**: 0
- **average_trajectory_length**: 1.0
- **average_input_tokens**: 185.5525
- **average_output_tokens**: 87.315
- **retrieval_hit_usefulness_proxy**: 0.0
- **graph_num_nodes**: 0
- **graph_num_edges**: 0
- **training_data_size_by_role::single_agent**: 400
- **dataset_accuracy::commonsense_qa**: 0.57
- **dataset_count::commonsense_qa**: 100
- **dataset_accuracy::ai2_arc**: 0.34
- **dataset_count::ai2_arc**: 100
- **dataset_accuracy::boolq**: 0.45
- **dataset_count::boolq**: 100
- **dataset_accuracy::pubmed_qa**: 0.45
- **dataset_count::pubmed_qa**: 100
- **category_accuracy::commonsense**: 0.57
- **category_accuracy::science_mcq**: 0.34
- **category_accuracy::reading_comprehension_yesno**: 0.45
- **category_accuracy::biomedical_qa**: 0.45

## Dataset Breakdown
- commonsense_qa: 0.57
- ai2_arc: 0.34
- boolq: 0.45
- pubmed_qa: 0.45

## Error Cases
- sample_id=commonsenseqa_validation_0 | dataset=commonsense_qa | pred=A rug placed near the front door is typically kept in the hallway, as it serves to protect the floor from wear and tear caused by foot traffic entering and exiting the house through the front door. The other options do not make sense for storing or placing a rug near a front door. | gold=D. living room
- sample_id=commonsenseqa_validation_6 | dataset=commonsense_qa | pred=This aligns with the typical scenario where performers prepare by changing into costumes before going on stage. | gold=A. theater
- sample_id=commonsenseqa_validation_10 | dataset=commonsense_qa | pred=This is because hotels often provide bath towels for guests' use during their stay. While it's possible to borrow other types of towels from various locations, bath towels specifically are more commonly available in this manner. | gold=B. at hotel
- sample_id=commonsenseqa_validation_11 | dataset=commonsense_qa | pred=A. ocean | gold=C. water
- sample_id=commonsenseqa_validation_12 | dataset=commonsense_qa | pred=Given that David is observing birds with binoculars, the most logical place for him to be is on vacation somewhere outside, such as in a park or nature reserve. | gold=C. forest
- sample_id=commonsenseqa_validation_16 | dataset=commonsense_qa | pred=The context provided does not give enough information to determine which specific type of property is being referred to. However, in general, "back yard" often refers to an outdoor area behind a home or property. Since the question mentions both children loving the backyard and parents loving its safety, it's most likely referring to a private piece of land owned by the family, which would be considered their "property." | gold=B. neighborhood
- sample_id=commonsenseqa_validation_19 | dataset=commonsense_qa | pred=This option suggests that Steve's initial thought of possibility turned into an agreement to not proceed or consider further. | gold=C. unlikely
- sample_id=commonsenseqa_validation_20 | dataset=commonsense_qa | pred=D. walk | gold=A. listen to radio
- sample_id=commonsenseqa_validation_23 | dataset=commonsense_qa | pred=Using a computer to communicate with someone like a granddaughter can sometimes lead to eye strain or headaches due to prolonged screen time and focusing on small text or images. However, it's not guaranteed that this will happen every time. The other options (program creation, stress, happiness, and ocean) do not directly relate to the act of communicating via a computer. | gold=C. happiness
- sample_id=commonsenseqa_validation_24 | dataset=commonsense_qa | pred=If someone gets a surprising show over and over, it could lead to annoyance because the novelty of the surprise wears off after repeated exposure. The other options do not fit the context as well as "annoyance" does. | gold=C. annoyance
- sample_id=commonsenseqa_validation_27 | dataset=commonsense_qa | pred=: | gold=C. augment
- sample_id=commonsenseqa_validation_29 | dataset=commonsense_qa | pred=D. prehistoric times. | gold=E. ancient times
- sample_id=commonsenseqa_validation_32 | dataset=commonsense_qa | pred=Many people moved from country estates to row houses in cities because it was more affordable and convenient compared to owning or renting land in rural areas. The other options do not fit the context of moving from country estates to row houses. | gold=D. city
- sample_id=commonsenseqa_validation_33 | dataset=commonsense_qa | pred=A. open eyes | gold=B. go to movies
- sample_id=commonsenseqa_validation_34 | dataset=commonsense_qa | pred=The question states that "There was no shade for Jenny," which implies she was not protected from the sun. Therefore, the most appropriate answer is "direct sunlight" as it directly corresponds to the lack of shade and exposure to the sun's rays. | gold=A. full sunlight
- sample_id=commonsenseqa_validation_35 | dataset=commonsense_qa | pred=When traveling, having sufficient funds to cover unexpected costs is crucial for financial stability and peace of mind during the trip. This ensures that you can handle any unforeseen expenses without compromising your ability to enjoy your journey or maintain your lifestyle back home. | gold=E. have money
- sample_id=commonsenseqa_validation_36 | dataset=commonsense_qa | pred=Jesse enjoys remembering the past to help him understand it, and understanding the past can aid in preparing for future situations or events. | gold=E. being prepared
- sample_id=commonsenseqa_validation_39 | dataset=commonsense_qa | pred=: | gold=D. confident
- sample_id=commonsenseqa_validation_41 | dataset=commonsense_qa | pred=The context provided does not give enough information to determine which specific type of building Joe is likely in. However, "large building" is the most general and appropriate answer among the given choices. Convention centers, for example, can have large halls where meetings take place, but they may not necessarily be the only option. Similarly, while a person or a box might also fit the description, these options are less common contexts for such an event as described. A public building could potentially include a meeting area, but it's not as definitive as "large building." | gold=C. convention center
- sample_id=commonsenseqa_validation_42 | dataset=commonsense_qa | pred=: | gold=B. snowflake
