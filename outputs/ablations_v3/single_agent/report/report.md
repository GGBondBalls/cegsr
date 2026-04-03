# Run Summary

## Aggregate Metrics
- **num_episodes**: 400
- **accuracy**: 0.42
- **exact_match**: 0.105
- **mcq_accuracy**: 0.42
- **repair_coverage**: 0.0
- **repair_success_rate**: 0.0
- **num_changed_repairs**: 0
- **repair_flip_to_success**: 0
- **average_trajectory_length**: 1.0
- **average_input_tokens**: 186.0525
- **average_output_tokens**: 97.265
- **retrieval_hit_usefulness_proxy**: 0.0
- **graph_num_nodes**: 0
- **graph_num_edges**: 0
- **training_data_size_by_role::single_agent**: 400
- **dataset_accuracy::commonsense_qa**: 0.55
- **dataset_count::commonsense_qa**: 100
- **dataset_accuracy::ai2_arc**: 0.33
- **dataset_count::ai2_arc**: 100
- **dataset_accuracy::boolq**: 0.43
- **dataset_count::boolq**: 100
- **dataset_accuracy::pubmed_qa**: 0.37
- **dataset_count::pubmed_qa**: 100
- **category_accuracy::commonsense**: 0.55
- **category_accuracy::science_mcq**: 0.33
- **category_accuracy::reading_comprehension_yesno**: 0.43
- **category_accuracy::biomedical_qa**: 0.37

## Dataset Breakdown
- commonsense_qa: 0.55
- ai2_arc: 0.33
- boolq: 0.43
- pubmed_qa: 0.37

## Error Cases
- sample_id=commonsenseqa_validation_0 | dataset=commonsense_qa | pred=A rug near the front door is typically kept in the hallway to provide warmth and softness as one enters or exits the house, especially during colder weather. The other options do not make sense for keeping a rug near a front door. | gold=D. living room
- sample_id=commonsenseqa_validation_1 | dataset=commonsense_qa | pred=C. Cave | gold=D. cabinet
- sample_id=commonsenseqa_validation_9 | dataset=commonsense_qa | pred=: | gold=D. stillness
- sample_id=commonsenseqa_validation_10 | dataset=commonsense_qa | pred=: | gold=B. at hotel
- sample_id=commonsenseqa_validation_11 | dataset=commonsense_qa | pred=A. ocean | gold=C. water
- sample_id=commonsenseqa_validation_12 | dataset=commonsense_qa | pred=E. Roof - This would only allow him to watch from above, which isn't necessarily better than being outside if he wants to see close-up details of the birds' nests. | gold=C. forest
- sample_id=commonsenseqa_validation_16 | dataset=commonsense_qa | pred=The context provided does not give enough information to determine which specific type of property is being referred to. However, in general terms, "back yard" can be associated with various types of properties such as houses or apartments. The most common association for a backyard is likely a residential property like a house. Therefore, among the given choices, "property" is the most appropriate answer based on the context. | gold=B. neighborhood
- sample_id=commonsenseqa_validation_19 | dataset=commonsense_qa | pred=This option best fits the phrasing "Steve thought that it was possible" and aligns with the statement "but he agreed that it was what?" suggesting an agreement to something less than what Steve initially believed was possible. | gold=C. unlikely
- sample_id=commonsenseqa_validation_20 | dataset=commonsense_qa | pred=D. walk | gold=A. listen to radio
- sample_id=commonsenseqa_validation_23 | dataset=commonsense_qa | pred=Using a computer to communicate with someone like a granddaughter can sometimes lead to eye strain or headaches if the user spends too much time in front of the screen without proper breaks. However, it's not typically described as causing "a program" or an "ocean." The other options don't fit the context provided. | gold=C. happiness
- sample_id=commonsenseqa_validation_27 | dataset=commonsense_qa | pred=Since we don't have enough context to choose from among these options, none of them can be definitively identified as the correct answer without further information. | gold=C. augment
- sample_id=commonsenseqa_validation_29 | dataset=commonsense_qa | pred=D. prehistoric times. | gold=E. ancient times
- sample_id=commonsenseqa_validation_32 | dataset=commonsense_qa | pred=Many people moved from country estates to row houses in cities because it was cheaper and more convenient to live in urban areas compared to rural settings. The other options do not directly relate to the context of moving from country estates to row houses. | gold=D. city
- sample_id=commonsenseqa_validation_33 | dataset=commonsense_qa | pred=A. open eyes | gold=B. go to movies
- sample_id=commonsenseqa_validation_34 | dataset=commonsense_qa | pred=The question states that "There was no shade for Jenny," which implies she was not protected from the sun. Therefore, the most appropriate answer is "direct sunlight" as it directly corresponds to the lack of shade and exposure to the sun's rays. | gold=A. full sunlight
- sample_id=commonsenseqa_validation_35 | dataset=commonsense_qa | pred=When traveling, having sufficient funds to cover unexpected costs is crucial for financial stability and peace of mind during the trip. This ensures that you can handle any unforeseen expenses without compromising your ability to enjoy your travels or maintain your daily routine. | gold=E. have money
- sample_id=commonsenseqa_validation_36 | dataset=commonsense_qa | pred=Jesse enjoys remembering the past to help him understand it, and understanding the past can aid in preparing for future situations or events. | gold=E. being prepared
- sample_id=commonsenseqa_validation_37 | dataset=commonsense_qa | pred=This option encompasses multiple aspects of what might be done at a market without being too broad or specific to a particular type of market activity. | gold=A. buy food
- sample_id=commonsenseqa_validation_39 | dataset=commonsense_qa | pred=: | gold=D. confident
- sample_id=commonsenseqa_validation_42 | dataset=commonsense_qa | pred=: | gold=B. snowflake
