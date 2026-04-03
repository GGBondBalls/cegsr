# Run Summary

## Aggregate Metrics
- **num_episodes**: 700
- **accuracy**: 0.2943
- **exact_match**: 0.0457
- **mcq_accuracy**: 0.2243
- **repair_coverage**: 0.0
- **repair_success_rate**: 0.0
- **num_changed_repairs**: 0
- **average_trajectory_length**: 1.0
- **average_input_tokens**: 154.66
- **average_output_tokens**: 172.6957
- **retrieval_hit_usefulness_proxy**: 0.0
- **graph_num_nodes**: 0
- **graph_num_edges**: 0
- **training_data_size_by_role::single_agent**: 700
- **dataset_accuracy::college_physics**: 0.06
- **dataset_count::college_physics**: 100
- **dataset_accuracy::college_chemistry**: 0.1
- **dataset_count::college_chemistry**: 100
- **dataset_accuracy::pubmed_qa**: 0.25
- **dataset_count::pubmed_qa**: 100
- **dataset_accuracy::gsm8k**: 0.49
- **dataset_count::gsm8k**: 100
- **dataset_accuracy::commonsense_qa**: 0.45
- **dataset_count::commonsense_qa**: 100
- **dataset_accuracy::ai2_arc**: 0.4
- **dataset_count::ai2_arc**: 100
- **dataset_accuracy::boolq**: 0.31
- **dataset_count::boolq**: 100
- **category_accuracy::college_physics**: 0.06
- **category_accuracy::college_chemistry**: 0.1
- **category_accuracy::biomedical_qa**: 0.25
- **category_accuracy::math_word_problem**: 0.49
- **category_accuracy::commonsense**: 0.45
- **category_accuracy::science_mcq**: 0.4
- **category_accuracy::reading_comprehension_yesno**: 0.31

## Dataset Breakdown
- college_physics: 0.06
- college_chemistry: 0.1
- pubmed_qa: 0.25
- gsm8k: 0.49
- commonsense_qa: 0.45
- ai2_arc: 0.4
- boolq: 0.31

## Error Cases
- sample_id=college_physics_test_0 | dataset=college_physics | pred=When the gas is present, the optical path length increases by a factor of \( n \), so the new optical path length \( L_{\text{ | gold=B. 1.00032
- sample_id=college_physics_test_1 | dataset=college_physics | pred=\[ W_2 = \frac{1 | gold=D. 8k
- sample_id=college_physics_test_2 | dataset=college_physics | pred=: | gold=A. 5
- sample_id=college_physics_test_3 | dataset=college_physics | pred=\[ 2 = | gold=D. (3^(1/2))mc
- sample_id=college_physics_test_4 | dataset=college_physics | pred=Given that the total energy to remove both electrons is 79.0 eV, and without additional information on the exact energy difference, we can infer that the energy required to remove the first electron is less than half of 79. | gold=A. 24.6 eV
- sample_id=college_physics_test_5 | dataset=college_physics | pred=\gamma = \frac{1}{\sqrt{1 - \frac{(0. | gold=D. 450 m
- sample_id=college_physics_test_6 | dataset=college_physics | pred=where \( \lambda_{\text{oil}} \) is the wavelength of light in the oil layer, and \( | gold=C. 600 nm
- sample_id=college_physics_test_7 | dataset=college_physics | pred=\[ \Delta KE | gold=C. 3 N
- sample_id=college_physics_test_8 | dataset=college_physics | pred=- This is | gold=D. Fermions have antisymmetric wave functions and obey the Pauli exclusion principle.
- sample_id=college_physics_test_9 | dataset=college_physics | pred=\lambda_{\text{max}} = \frac{b}{T} = \frac{2.89777 \times 10^{-3} | gold=A. 10:00 PM
- sample_id=college_physics_test_10 | dataset=college_physics | pred=\Phi = B \cdot A | gold=D. 10 V
- sample_id=college_physics_test_11 | dataset=college_physics | pred=: | gold=D. tetrahedron
- sample_id=college_physics_test_12 | dataset=college_physics | pred=\[ \text{COP} = \frac{300.15}{300.15 - 280.15} = \frac{300.15 | gold=B. 1,000 J
- sample_id=college_physics_test_13 | dataset=college_physics | pred=- The mean (\( \mu \)) of the number of detections is | gold=D. an average of 10 times, with an rms deviation of about 3
- sample_id=college_physics_test_14 | dataset=college_physics | pred=\[ a = \sqrt{a_c^2 + a_t^2} = \sqrt{10^ | gold=C. 45°
- sample_id=college_physics_test_15 | dataset=college_physics | pred=- It has no electric charge and no rest mass (or very | gold=A. Electron
- sample_id=college_physics_test_16 | dataset=college_physics | pred=\[ R = \frac{501 \, \text{nm}}{2 \, \text{nm | gold=B. 250
- sample_id=college_physics_test_17 | dataset=college_physics | pred=\[ f' = 0.97 \times 440 \text{ Hz | gold=B. 427 Hz
- sample_id=college_physics_test_18 | dataset=college_physics | pred=\[ \left( | gold=B. 0.60c
- sample_id=college_physics_test_19 | dataset=college_physics | pred=2. **Constructive Interference Condition**: For constructive interference, the path difference must be an odd multiple of half the wavelength in the film. Since the light reflects from both the top and bottom surfaces of the film, the total path difference is twice the thickness of the | gold=B. 200 nm
