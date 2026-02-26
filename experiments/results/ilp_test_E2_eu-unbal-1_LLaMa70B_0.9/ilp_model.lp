\ Model 2026-02-22_22-58-22
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  0.0014344262295082 flow_source_nic_in_0
   + 0.0014344262295082 flow_source_nic_in_2
   + 0.0014344262295082 flow_source_nic_in_4
   + 0.0014344262295082 flow_source_nic_in_5
   + 0.0014344262295082 flow_source_nic_in_7
   + 0.0014344262295082 flow_source_nic_in_11
   + 0.0014344262295082 flow_source_nic_in_12
   + 0.0014344262295082 flow_source_nic_in_14
   + 0.0014344262295082 flow_source_nic_in_15
   - 0.0107936507936508 is_active_0 - 0.0107936507936508 is_active_1
   - 0.0107936507936508 is_active_2 - 0.0107936507936508 is_active_3
   - 0.00373015873015873 is_active_4 - 0.00373015873015873 is_active_5
   - 0.00373015873015873 is_active_6 - 0.00373015873015873 is_active_7
   - 0.00373015873015873 is_active_8 - 0.00373015873015873 is_active_9
   - 0.00373015873015873 is_active_10 - 0.0130952380952381 is_active_11
   - 0.00238095238095238 is_active_12 - 0.00238095238095238 is_active_13
   - 0.0102380952380952 is_active_14 - 0.00261904761904762 is_active_15
   + 0 val_end_0 + 0 val_end_1 + 0 val_end_2 + 0 val_end_3 + 0 val_end_4
   + 0 val_end_5 + 0 val_end_6 + 0 val_end_7 + 0 val_end_8 + 0 val_end_9
   + 0 val_end_10 + 0 val_end_11 + 0 val_end_12 + 0 val_end_13
   + 0 val_end_14 + 0 val_end_15
Subject To
 mem_active_constr_0: hold_0_1 + hold_0_2 + hold_0_3 + hold_0_4 + hold_0_5
   + hold_0_6 + hold_0_7 + hold_0_8 + hold_0_9 + hold_0_10 + hold_0_11
   + hold_0_12 + hold_0_13 + hold_0_14 + hold_0_15 + hold_0_16 + hold_0_17
   + hold_0_18 + hold_0_19 + hold_0_20 + hold_0_21 + hold_0_22 + hold_0_23
   + hold_0_24 + hold_0_25 + hold_0_26 - 81 is_active_0 <= 0
 mem_active_constr2_0: hold_0_1 + hold_0_2 + hold_0_3 + hold_0_4 + hold_0_5
   + hold_0_6 + hold_0_7 + hold_0_8 + hold_0_9 + hold_0_10 + hold_0_11
   + hold_0_12 + hold_0_13 + hold_0_14 + hold_0_15 + hold_0_16 + hold_0_17
   + hold_0_18 + hold_0_19 + hold_0_20 + hold_0_21 + hold_0_22 + hold_0_23
   + hold_0_24 + hold_0_25 + hold_0_26 - is_active_0 >= 0
 mem_active_constr_1: hold_1_1 + hold_1_2 + hold_1_3 + hold_1_4 + hold_1_5
   + hold_1_6 + hold_1_7 + hold_1_8 + hold_1_9 + hold_1_10 + hold_1_11
   + hold_1_12 + hold_1_13 + hold_1_14 + hold_1_15 + hold_1_16 + hold_1_17
   + hold_1_18 + hold_1_19 + hold_1_20 + hold_1_21 + hold_1_22 + hold_1_23
   + hold_1_24 + hold_1_25 + hold_1_26 - 81 is_active_1 <= 0
 mem_active_constr2_1: hold_1_1 + hold_1_2 + hold_1_3 + hold_1_4 + hold_1_5
   + hold_1_6 + hold_1_7 + hold_1_8 + hold_1_9 + hold_1_10 + hold_1_11
   + hold_1_12 + hold_1_13 + hold_1_14 + hold_1_15 + hold_1_16 + hold_1_17
   + hold_1_18 + hold_1_19 + hold_1_20 + hold_1_21 + hold_1_22 + hold_1_23
   + hold_1_24 + hold_1_25 + hold_1_26 - is_active_1 >= 0
 mem_active_constr_2: hold_2_1 + hold_2_2 + hold_2_3 + hold_2_4 + hold_2_5
   + hold_2_6 + hold_2_7 + hold_2_8 + hold_2_9 + hold_2_10 + hold_2_11
   + hold_2_12 + hold_2_13 + hold_2_14 + hold_2_15 + hold_2_16 + hold_2_17
   + hold_2_18 + hold_2_19 + hold_2_20 + hold_2_21 + hold_2_22 + hold_2_23
   + hold_2_24 + hold_2_25 + hold_2_26 - 81 is_active_2 <= 0
 mem_active_constr2_2: hold_2_1 + hold_2_2 + hold_2_3 + hold_2_4 + hold_2_5
   + hold_2_6 + hold_2_7 + hold_2_8 + hold_2_9 + hold_2_10 + hold_2_11
   + hold_2_12 + hold_2_13 + hold_2_14 + hold_2_15 + hold_2_16 + hold_2_17
   + hold_2_18 + hold_2_19 + hold_2_20 + hold_2_21 + hold_2_22 + hold_2_23
   + hold_2_24 + hold_2_25 + hold_2_26 - is_active_2 >= 0
 mem_active_constr_3: hold_3_1 + hold_3_2 + hold_3_3 + hold_3_4 + hold_3_5
   + hold_3_6 + hold_3_7 + hold_3_8 + hold_3_9 + hold_3_10 + hold_3_11
   + hold_3_12 + hold_3_13 + hold_3_14 + hold_3_15 + hold_3_16 + hold_3_17
   + hold_3_18 + hold_3_19 + hold_3_20 + hold_3_21 + hold_3_22 + hold_3_23
   + hold_3_24 + hold_3_25 + hold_3_26 - 81 is_active_3 <= 0
 mem_active_constr2_3: hold_3_1 + hold_3_2 + hold_3_3 + hold_3_4 + hold_3_5
   + hold_3_6 + hold_3_7 + hold_3_8 + hold_3_9 + hold_3_10 + hold_3_11
   + hold_3_12 + hold_3_13 + hold_3_14 + hold_3_15 + hold_3_16 + hold_3_17
   + hold_3_18 + hold_3_19 + hold_3_20 + hold_3_21 + hold_3_22 + hold_3_23
   + hold_3_24 + hold_3_25 + hold_3_26 - is_active_3 >= 0
 mem_active_constr_4: hold_4_1 + hold_4_2 + hold_4_3 + hold_4_4 + hold_4_5
   + hold_4_6 + hold_4_7 + hold_4_8 + hold_4_9 + hold_4_10 + hold_4_11
   + hold_4_12 + hold_4_13 + hold_4_14 + hold_4_15 + hold_4_16 + hold_4_17
   + hold_4_18 + hold_4_19 + hold_4_20 + hold_4_21 + hold_4_22 + hold_4_23
   + hold_4_24 + hold_4_25 + hold_4_26 - 81 is_active_4 <= 0
 mem_active_constr2_4: hold_4_1 + hold_4_2 + hold_4_3 + hold_4_4 + hold_4_5
   + hold_4_6 + hold_4_7 + hold_4_8 + hold_4_9 + hold_4_10 + hold_4_11
   + hold_4_12 + hold_4_13 + hold_4_14 + hold_4_15 + hold_4_16 + hold_4_17
   + hold_4_18 + hold_4_19 + hold_4_20 + hold_4_21 + hold_4_22 + hold_4_23
   + hold_4_24 + hold_4_25 + hold_4_26 - is_active_4 >= 0
 mem_active_constr_5: hold_5_1 + hold_5_2 + hold_5_3 + hold_5_4 + hold_5_5
   + hold_5_6 + hold_5_7 + hold_5_8 + hold_5_9 + hold_5_10 + hold_5_11
   + hold_5_12 + hold_5_13 + hold_5_14 + hold_5_15 + hold_5_16 + hold_5_17
   + hold_5_18 + hold_5_19 + hold_5_20 + hold_5_21 + hold_5_22 + hold_5_23
   + hold_5_24 + hold_5_25 + hold_5_26 - 81 is_active_5 <= 0
 mem_active_constr2_5: hold_5_1 + hold_5_2 + hold_5_3 + hold_5_4 + hold_5_5
   + hold_5_6 + hold_5_7 + hold_5_8 + hold_5_9 + hold_5_10 + hold_5_11
   + hold_5_12 + hold_5_13 + hold_5_14 + hold_5_15 + hold_5_16 + hold_5_17
   + hold_5_18 + hold_5_19 + hold_5_20 + hold_5_21 + hold_5_22 + hold_5_23
   + hold_5_24 + hold_5_25 + hold_5_26 - is_active_5 >= 0
 mem_active_constr_6: hold_6_1 + hold_6_2 + hold_6_3 + hold_6_4 + hold_6_5
   + hold_6_6 + hold_6_7 + hold_6_8 + hold_6_9 + hold_6_10 + hold_6_11
   + hold_6_12 + hold_6_13 + hold_6_14 + hold_6_15 + hold_6_16 + hold_6_17
   + hold_6_18 + hold_6_19 + hold_6_20 + hold_6_21 + hold_6_22 + hold_6_23
   + hold_6_24 + hold_6_25 + hold_6_26 - 81 is_active_6 <= 0
 mem_active_constr2_6: hold_6_1 + hold_6_2 + hold_6_3 + hold_6_4 + hold_6_5
   + hold_6_6 + hold_6_7 + hold_6_8 + hold_6_9 + hold_6_10 + hold_6_11
   + hold_6_12 + hold_6_13 + hold_6_14 + hold_6_15 + hold_6_16 + hold_6_17
   + hold_6_18 + hold_6_19 + hold_6_20 + hold_6_21 + hold_6_22 + hold_6_23
   + hold_6_24 + hold_6_25 + hold_6_26 - is_active_6 >= 0
 mem_active_constr_7: hold_7_1 + hold_7_2 + hold_7_3 + hold_7_4 + hold_7_5
   + hold_7_6 + hold_7_7 + hold_7_8 + hold_7_9 + hold_7_10 + hold_7_11
   + hold_7_12 + hold_7_13 + hold_7_14 + hold_7_15 + hold_7_16 + hold_7_17
   + hold_7_18 + hold_7_19 + hold_7_20 + hold_7_21 + hold_7_22 + hold_7_23
   + hold_7_24 + hold_7_25 + hold_7_26 - 81 is_active_7 <= 0
 mem_active_constr2_7: hold_7_1 + hold_7_2 + hold_7_3 + hold_7_4 + hold_7_5
   + hold_7_6 + hold_7_7 + hold_7_8 + hold_7_9 + hold_7_10 + hold_7_11
   + hold_7_12 + hold_7_13 + hold_7_14 + hold_7_15 + hold_7_16 + hold_7_17
   + hold_7_18 + hold_7_19 + hold_7_20 + hold_7_21 + hold_7_22 + hold_7_23
   + hold_7_24 + hold_7_25 + hold_7_26 - is_active_7 >= 0
 mem_active_constr_8: hold_8_1 + hold_8_2 + hold_8_3 + hold_8_4 + hold_8_5
   + hold_8_6 + hold_8_7 + hold_8_8 + hold_8_9 + hold_8_10 + hold_8_11
   + hold_8_12 + hold_8_13 + hold_8_14 + hold_8_15 + hold_8_16 + hold_8_17
   + hold_8_18 + hold_8_19 + hold_8_20 + hold_8_21 + hold_8_22 + hold_8_23
   + hold_8_24 + hold_8_25 + hold_8_26 - 81 is_active_8 <= 0
 mem_active_constr2_8: hold_8_1 + hold_8_2 + hold_8_3 + hold_8_4 + hold_8_5
   + hold_8_6 + hold_8_7 + hold_8_8 + hold_8_9 + hold_8_10 + hold_8_11
   + hold_8_12 + hold_8_13 + hold_8_14 + hold_8_15 + hold_8_16 + hold_8_17
   + hold_8_18 + hold_8_19 + hold_8_20 + hold_8_21 + hold_8_22 + hold_8_23
   + hold_8_24 + hold_8_25 + hold_8_26 - is_active_8 >= 0
 mem_active_constr_9: hold_9_1 + hold_9_2 + hold_9_3 + hold_9_4 + hold_9_5
   + hold_9_6 + hold_9_7 + hold_9_8 + hold_9_9 + hold_9_10 + hold_9_11
   + hold_9_12 + hold_9_13 + hold_9_14 + hold_9_15 + hold_9_16 + hold_9_17
   + hold_9_18 + hold_9_19 + hold_9_20 + hold_9_21 + hold_9_22 + hold_9_23
   + hold_9_24 + hold_9_25 + hold_9_26 - 81 is_active_9 <= 0
 mem_active_constr2_9: hold_9_1 + hold_9_2 + hold_9_3 + hold_9_4 + hold_9_5
   + hold_9_6 + hold_9_7 + hold_9_8 + hold_9_9 + hold_9_10 + hold_9_11
   + hold_9_12 + hold_9_13 + hold_9_14 + hold_9_15 + hold_9_16 + hold_9_17
   + hold_9_18 + hold_9_19 + hold_9_20 + hold_9_21 + hold_9_22 + hold_9_23
   + hold_9_24 + hold_9_25 + hold_9_26 - is_active_9 >= 0
 mem_active_constr_10: hold_10_1 + hold_10_2 + hold_10_3 + hold_10_4
   + hold_10_5 + hold_10_6 + hold_10_7 + hold_10_8 + hold_10_9 + hold_10_10
   + hold_10_11 + hold_10_12 + hold_10_13 + hold_10_14 + hold_10_15
   + hold_10_16 + hold_10_17 + hold_10_18 + hold_10_19 + hold_10_20
   + hold_10_21 + hold_10_22 + hold_10_23 + hold_10_24 + hold_10_25
   + hold_10_26 - 81 is_active_10 <= 0
 mem_active_constr2_10: hold_10_1 + hold_10_2 + hold_10_3 + hold_10_4
   + hold_10_5 + hold_10_6 + hold_10_7 + hold_10_8 + hold_10_9 + hold_10_10
   + hold_10_11 + hold_10_12 + hold_10_13 + hold_10_14 + hold_10_15
   + hold_10_16 + hold_10_17 + hold_10_18 + hold_10_19 + hold_10_20
   + hold_10_21 + hold_10_22 + hold_10_23 + hold_10_24 + hold_10_25
   + hold_10_26 - is_active_10 >= 0
 mem_active_constr_11: hold_11_1 + hold_11_2 + hold_11_3 + hold_11_4
   + hold_11_5 + hold_11_6 + hold_11_7 + hold_11_8 + hold_11_9 + hold_11_10
   + hold_11_11 + hold_11_12 + hold_11_13 + hold_11_14 + hold_11_15
   + hold_11_16 + hold_11_17 + hold_11_18 + hold_11_19 + hold_11_20
   + hold_11_21 + hold_11_22 + hold_11_23 + hold_11_24 + hold_11_25
   + hold_11_26 - 81 is_active_11 <= 0
 mem_active_constr2_11: hold_11_1 + hold_11_2 + hold_11_3 + hold_11_4
   + hold_11_5 + hold_11_6 + hold_11_7 + hold_11_8 + hold_11_9 + hold_11_10
   + hold_11_11 + hold_11_12 + hold_11_13 + hold_11_14 + hold_11_15
   + hold_11_16 + hold_11_17 + hold_11_18 + hold_11_19 + hold_11_20
   + hold_11_21 + hold_11_22 + hold_11_23 + hold_11_24 + hold_11_25
   + hold_11_26 - is_active_11 >= 0
 mem_active_constr_12: hold_12_1 + hold_12_2 + hold_12_3 + hold_12_4
   + hold_12_5 + hold_12_6 + hold_12_7 + hold_12_8 + hold_12_9 + hold_12_10
   + hold_12_11 + hold_12_12 + hold_12_13 + hold_12_14 + hold_12_15
   + hold_12_16 + hold_12_17 + hold_12_18 + hold_12_19 + hold_12_20
   + hold_12_21 + hold_12_22 + hold_12_23 + hold_12_24 + hold_12_25
   + hold_12_26 - 81 is_active_12 <= 0
 mem_active_constr2_12: hold_12_1 + hold_12_2 + hold_12_3 + hold_12_4
   + hold_12_5 + hold_12_6 + hold_12_7 + hold_12_8 + hold_12_9 + hold_12_10
   + hold_12_11 + hold_12_12 + hold_12_13 + hold_12_14 + hold_12_15
   + hold_12_16 + hold_12_17 + hold_12_18 + hold_12_19 + hold_12_20
   + hold_12_21 + hold_12_22 + hold_12_23 + hold_12_24 + hold_12_25
   + hold_12_26 - is_active_12 >= 0
 mem_active_constr_13: hold_13_1 + hold_13_2 + hold_13_3 + hold_13_4
   + hold_13_5 + hold_13_6 + hold_13_7 + hold_13_8 + hold_13_9 + hold_13_10
   + hold_13_11 + hold_13_12 + hold_13_13 + hold_13_14 + hold_13_15
   + hold_13_16 + hold_13_17 + hold_13_18 + hold_13_19 + hold_13_20
   + hold_13_21 + hold_13_22 + hold_13_23 + hold_13_24 + hold_13_25
   + hold_13_26 - 81 is_active_13 <= 0
 mem_active_constr2_13: hold_13_1 + hold_13_2 + hold_13_3 + hold_13_4
   + hold_13_5 + hold_13_6 + hold_13_7 + hold_13_8 + hold_13_9 + hold_13_10
   + hold_13_11 + hold_13_12 + hold_13_13 + hold_13_14 + hold_13_15
   + hold_13_16 + hold_13_17 + hold_13_18 + hold_13_19 + hold_13_20
   + hold_13_21 + hold_13_22 + hold_13_23 + hold_13_24 + hold_13_25
   + hold_13_26 - is_active_13 >= 0
 mem_active_constr_14: hold_14_1 + hold_14_2 + hold_14_3 + hold_14_4
   + hold_14_5 + hold_14_6 + hold_14_7 + hold_14_8 + hold_14_9 + hold_14_10
   + hold_14_11 + hold_14_12 + hold_14_13 + hold_14_14 + hold_14_15
   + hold_14_16 + hold_14_17 + hold_14_18 + hold_14_19 + hold_14_20
   + hold_14_21 + hold_14_22 + hold_14_23 + hold_14_24 + hold_14_25
   + hold_14_26 - 81 is_active_14 <= 0
 mem_active_constr2_14: hold_14_1 + hold_14_2 + hold_14_3 + hold_14_4
   + hold_14_5 + hold_14_6 + hold_14_7 + hold_14_8 + hold_14_9 + hold_14_10
   + hold_14_11 + hold_14_12 + hold_14_13 + hold_14_14 + hold_14_15
   + hold_14_16 + hold_14_17 + hold_14_18 + hold_14_19 + hold_14_20
   + hold_14_21 + hold_14_22 + hold_14_23 + hold_14_24 + hold_14_25
   + hold_14_26 - is_active_14 >= 0
 mem_active_constr_15: hold_15_1 + hold_15_2 + hold_15_3 + hold_15_4
   + hold_15_5 + hold_15_6 + hold_15_7 + hold_15_8 + hold_15_9 + hold_15_10
   + hold_15_11 + hold_15_12 + hold_15_13 + hold_15_14 + hold_15_15
   + hold_15_16 + hold_15_17 + hold_15_18 + hold_15_19 + hold_15_20
   + hold_15_21 + hold_15_22 + hold_15_23 + hold_15_24 + hold_15_25
   + hold_15_26 - 81 is_active_15 <= 0
 mem_active_constr2_15: hold_15_1 + hold_15_2 + hold_15_3 + hold_15_4
   + hold_15_5 + hold_15_6 + hold_15_7 + hold_15_8 + hold_15_9 + hold_15_10
   + hold_15_11 + hold_15_12 + hold_15_13 + hold_15_14 + hold_15_15
   + hold_15_16 + hold_15_17 + hold_15_18 + hold_15_19 + hold_15_20
   + hold_15_21 + hold_15_22 + hold_15_23 + hold_15_24 + hold_15_25
   + hold_15_26 - is_active_15 >= 0
 hold_constraint_0: hold_0_1 + hold_0_2 + hold_0_3 + hold_0_4 + hold_0_5
   + hold_0_6 + hold_0_7 + hold_0_8 + hold_0_9 + hold_0_10 + hold_0_11
   + hold_0_12 + hold_0_13 + hold_0_14 + hold_0_15 + hold_0_16 + hold_0_17
   + hold_0_18 + hold_0_19 + hold_0_20 + hold_0_21 + hold_0_22 + hold_0_23
   + hold_0_24 + hold_0_25 + hold_0_26 <= 1
 end_constraint_0: start_0 + hold_0_1 + 2 hold_0_2 + 3 hold_0_3
   + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7 + 8 hold_0_8
   + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12 + 13 hold_0_13
   + 14 hold_0_14 + 15 hold_0_15 + 16 hold_0_16 + 17 hold_0_17
   + 18 hold_0_18 + 19 hold_0_19 + 20 hold_0_20 + 21 hold_0_21
   + 22 hold_0_22 + 23 hold_0_23 + 24 hold_0_24 + 25 hold_0_25
   + 26 hold_0_26 <= 80
 hold_constraint_1: hold_1_1 + hold_1_2 + hold_1_3 + hold_1_4 + hold_1_5
   + hold_1_6 + hold_1_7 + hold_1_8 + hold_1_9 + hold_1_10 + hold_1_11
   + hold_1_12 + hold_1_13 + hold_1_14 + hold_1_15 + hold_1_16 + hold_1_17
   + hold_1_18 + hold_1_19 + hold_1_20 + hold_1_21 + hold_1_22 + hold_1_23
   + hold_1_24 + hold_1_25 + hold_1_26 <= 1
 end_constraint_1: start_1 + hold_1_1 + 2 hold_1_2 + 3 hold_1_3
   + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7 + 8 hold_1_8
   + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 + 12 hold_1_12 + 13 hold_1_13
   + 14 hold_1_14 + 15 hold_1_15 + 16 hold_1_16 + 17 hold_1_17
   + 18 hold_1_18 + 19 hold_1_19 + 20 hold_1_20 + 21 hold_1_21
   + 22 hold_1_22 + 23 hold_1_23 + 24 hold_1_24 + 25 hold_1_25
   + 26 hold_1_26 <= 80
 hold_constraint_2: hold_2_1 + hold_2_2 + hold_2_3 + hold_2_4 + hold_2_5
   + hold_2_6 + hold_2_7 + hold_2_8 + hold_2_9 + hold_2_10 + hold_2_11
   + hold_2_12 + hold_2_13 + hold_2_14 + hold_2_15 + hold_2_16 + hold_2_17
   + hold_2_18 + hold_2_19 + hold_2_20 + hold_2_21 + hold_2_22 + hold_2_23
   + hold_2_24 + hold_2_25 + hold_2_26 <= 1
 end_constraint_2: start_2 + hold_2_1 + 2 hold_2_2 + 3 hold_2_3
   + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7 + 8 hold_2_8
   + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 12 hold_2_12 + 13 hold_2_13
   + 14 hold_2_14 + 15 hold_2_15 + 16 hold_2_16 + 17 hold_2_17
   + 18 hold_2_18 + 19 hold_2_19 + 20 hold_2_20 + 21 hold_2_21
   + 22 hold_2_22 + 23 hold_2_23 + 24 hold_2_24 + 25 hold_2_25
   + 26 hold_2_26 <= 80
 hold_constraint_3: hold_3_1 + hold_3_2 + hold_3_3 + hold_3_4 + hold_3_5
   + hold_3_6 + hold_3_7 + hold_3_8 + hold_3_9 + hold_3_10 + hold_3_11
   + hold_3_12 + hold_3_13 + hold_3_14 + hold_3_15 + hold_3_16 + hold_3_17
   + hold_3_18 + hold_3_19 + hold_3_20 + hold_3_21 + hold_3_22 + hold_3_23
   + hold_3_24 + hold_3_25 + hold_3_26 <= 1
 end_constraint_3: start_3 + hold_3_1 + 2 hold_3_2 + 3 hold_3_3
   + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7 + 8 hold_3_8
   + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11 + 12 hold_3_12 + 13 hold_3_13
   + 14 hold_3_14 + 15 hold_3_15 + 16 hold_3_16 + 17 hold_3_17
   + 18 hold_3_18 + 19 hold_3_19 + 20 hold_3_20 + 21 hold_3_21
   + 22 hold_3_22 + 23 hold_3_23 + 24 hold_3_24 + 25 hold_3_25
   + 26 hold_3_26 <= 80
 hold_constraint_4: hold_4_1 + hold_4_2 + hold_4_3 + hold_4_4 + hold_4_5
   + hold_4_6 + hold_4_7 + hold_4_8 + hold_4_9 + hold_4_10 + hold_4_11
   + hold_4_12 + hold_4_13 + hold_4_14 + hold_4_15 + hold_4_16 + hold_4_17
   + hold_4_18 + hold_4_19 + hold_4_20 + hold_4_21 + hold_4_22 + hold_4_23
   + hold_4_24 + hold_4_25 + hold_4_26 <= 1
 end_constraint_4: start_4 + hold_4_1 + 2 hold_4_2 + 3 hold_4_3
   + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7 + 8 hold_4_8
   + 9 hold_4_9 + 10 hold_4_10 + 11 hold_4_11 + 12 hold_4_12 + 13 hold_4_13
   + 14 hold_4_14 + 15 hold_4_15 + 16 hold_4_16 + 17 hold_4_17
   + 18 hold_4_18 + 19 hold_4_19 + 20 hold_4_20 + 21 hold_4_21
   + 22 hold_4_22 + 23 hold_4_23 + 24 hold_4_24 + 25 hold_4_25
   + 26 hold_4_26 <= 80
 hold_constraint_5: hold_5_1 + hold_5_2 + hold_5_3 + hold_5_4 + hold_5_5
   + hold_5_6 + hold_5_7 + hold_5_8 + hold_5_9 + hold_5_10 + hold_5_11
   + hold_5_12 + hold_5_13 + hold_5_14 + hold_5_15 + hold_5_16 + hold_5_17
   + hold_5_18 + hold_5_19 + hold_5_20 + hold_5_21 + hold_5_22 + hold_5_23
   + hold_5_24 + hold_5_25 + hold_5_26 <= 1
 end_constraint_5: start_5 + hold_5_1 + 2 hold_5_2 + 3 hold_5_3
   + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7 + 8 hold_5_8
   + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12 + 13 hold_5_13
   + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16 + 17 hold_5_17
   + 18 hold_5_18 + 19 hold_5_19 + 20 hold_5_20 + 21 hold_5_21
   + 22 hold_5_22 + 23 hold_5_23 + 24 hold_5_24 + 25 hold_5_25
   + 26 hold_5_26 <= 80
 hold_constraint_6: hold_6_1 + hold_6_2 + hold_6_3 + hold_6_4 + hold_6_5
   + hold_6_6 + hold_6_7 + hold_6_8 + hold_6_9 + hold_6_10 + hold_6_11
   + hold_6_12 + hold_6_13 + hold_6_14 + hold_6_15 + hold_6_16 + hold_6_17
   + hold_6_18 + hold_6_19 + hold_6_20 + hold_6_21 + hold_6_22 + hold_6_23
   + hold_6_24 + hold_6_25 + hold_6_26 <= 1
 end_constraint_6: start_6 + hold_6_1 + 2 hold_6_2 + 3 hold_6_3
   + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7 + 8 hold_6_8
   + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 12 hold_6_12 + 13 hold_6_13
   + 14 hold_6_14 + 15 hold_6_15 + 16 hold_6_16 + 17 hold_6_17
   + 18 hold_6_18 + 19 hold_6_19 + 20 hold_6_20 + 21 hold_6_21
   + 22 hold_6_22 + 23 hold_6_23 + 24 hold_6_24 + 25 hold_6_25
   + 26 hold_6_26 <= 80
 hold_constraint_7: hold_7_1 + hold_7_2 + hold_7_3 + hold_7_4 + hold_7_5
   + hold_7_6 + hold_7_7 + hold_7_8 + hold_7_9 + hold_7_10 + hold_7_11
   + hold_7_12 + hold_7_13 + hold_7_14 + hold_7_15 + hold_7_16 + hold_7_17
   + hold_7_18 + hold_7_19 + hold_7_20 + hold_7_21 + hold_7_22 + hold_7_23
   + hold_7_24 + hold_7_25 + hold_7_26 <= 1
 end_constraint_7: start_7 + hold_7_1 + 2 hold_7_2 + 3 hold_7_3
   + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7 + 8 hold_7_8
   + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11 + 12 hold_7_12 + 13 hold_7_13
   + 14 hold_7_14 + 15 hold_7_15 + 16 hold_7_16 + 17 hold_7_17
   + 18 hold_7_18 + 19 hold_7_19 + 20 hold_7_20 + 21 hold_7_21
   + 22 hold_7_22 + 23 hold_7_23 + 24 hold_7_24 + 25 hold_7_25
   + 26 hold_7_26 <= 80
 hold_constraint_8: hold_8_1 + hold_8_2 + hold_8_3 + hold_8_4 + hold_8_5
   + hold_8_6 + hold_8_7 + hold_8_8 + hold_8_9 + hold_8_10 + hold_8_11
   + hold_8_12 + hold_8_13 + hold_8_14 + hold_8_15 + hold_8_16 + hold_8_17
   + hold_8_18 + hold_8_19 + hold_8_20 + hold_8_21 + hold_8_22 + hold_8_23
   + hold_8_24 + hold_8_25 + hold_8_26 <= 1
 end_constraint_8: start_8 + hold_8_1 + 2 hold_8_2 + 3 hold_8_3
   + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7 + 8 hold_8_8
   + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 12 hold_8_12 + 13 hold_8_13
   + 14 hold_8_14 + 15 hold_8_15 + 16 hold_8_16 + 17 hold_8_17
   + 18 hold_8_18 + 19 hold_8_19 + 20 hold_8_20 + 21 hold_8_21
   + 22 hold_8_22 + 23 hold_8_23 + 24 hold_8_24 + 25 hold_8_25
   + 26 hold_8_26 <= 80
 hold_constraint_9: hold_9_1 + hold_9_2 + hold_9_3 + hold_9_4 + hold_9_5
   + hold_9_6 + hold_9_7 + hold_9_8 + hold_9_9 + hold_9_10 + hold_9_11
   + hold_9_12 + hold_9_13 + hold_9_14 + hold_9_15 + hold_9_16 + hold_9_17
   + hold_9_18 + hold_9_19 + hold_9_20 + hold_9_21 + hold_9_22 + hold_9_23
   + hold_9_24 + hold_9_25 + hold_9_26 <= 1
 end_constraint_9: start_9 + hold_9_1 + 2 hold_9_2 + 3 hold_9_3
   + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7 + 8 hold_9_8
   + 9 hold_9_9 + 10 hold_9_10 + 11 hold_9_11 + 12 hold_9_12 + 13 hold_9_13
   + 14 hold_9_14 + 15 hold_9_15 + 16 hold_9_16 + 17 hold_9_17
   + 18 hold_9_18 + 19 hold_9_19 + 20 hold_9_20 + 21 hold_9_21
   + 22 hold_9_22 + 23 hold_9_23 + 24 hold_9_24 + 25 hold_9_25
   + 26 hold_9_26 <= 80
 hold_constraint_10: hold_10_1 + hold_10_2 + hold_10_3 + hold_10_4
   + hold_10_5 + hold_10_6 + hold_10_7 + hold_10_8 + hold_10_9 + hold_10_10
   + hold_10_11 + hold_10_12 + hold_10_13 + hold_10_14 + hold_10_15
   + hold_10_16 + hold_10_17 + hold_10_18 + hold_10_19 + hold_10_20
   + hold_10_21 + hold_10_22 + hold_10_23 + hold_10_24 + hold_10_25
   + hold_10_26 <= 1
 end_constraint_10: start_10 + hold_10_1 + 2 hold_10_2 + 3 hold_10_3
   + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7 + 8 hold_10_8
   + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11 + 12 hold_10_12
   + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15 + 16 hold_10_16
   + 17 hold_10_17 + 18 hold_10_18 + 19 hold_10_19 + 20 hold_10_20
   + 21 hold_10_21 + 22 hold_10_22 + 23 hold_10_23 + 24 hold_10_24
   + 25 hold_10_25 + 26 hold_10_26 <= 80
 hold_constraint_11: hold_11_1 + hold_11_2 + hold_11_3 + hold_11_4
   + hold_11_5 + hold_11_6 + hold_11_7 + hold_11_8 + hold_11_9 + hold_11_10
   + hold_11_11 + hold_11_12 + hold_11_13 + hold_11_14 + hold_11_15
   + hold_11_16 + hold_11_17 + hold_11_18 + hold_11_19 + hold_11_20
   + hold_11_21 + hold_11_22 + hold_11_23 + hold_11_24 + hold_11_25
   + hold_11_26 <= 1
 end_constraint_11: start_11 + hold_11_1 + 2 hold_11_2 + 3 hold_11_3
   + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7 + 8 hold_11_8
   + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11 + 12 hold_11_12
   + 13 hold_11_13 + 14 hold_11_14 + 15 hold_11_15 + 16 hold_11_16
   + 17 hold_11_17 + 18 hold_11_18 + 19 hold_11_19 + 20 hold_11_20
   + 21 hold_11_21 + 22 hold_11_22 + 23 hold_11_23 + 24 hold_11_24
   + 25 hold_11_25 + 26 hold_11_26 <= 80
 hold_constraint_12: hold_12_1 + hold_12_2 + hold_12_3 + hold_12_4
   + hold_12_5 + hold_12_6 + hold_12_7 + hold_12_8 + hold_12_9 + hold_12_10
   + hold_12_11 + hold_12_12 + hold_12_13 + hold_12_14 + hold_12_15
   + hold_12_16 + hold_12_17 + hold_12_18 + hold_12_19 + hold_12_20
   + hold_12_21 + hold_12_22 + hold_12_23 + hold_12_24 + hold_12_25
   + hold_12_26 <= 1
 end_constraint_12: start_12 + hold_12_1 + 2 hold_12_2 + 3 hold_12_3
   + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6 + 7 hold_12_7 + 8 hold_12_8
   + 9 hold_12_9 + 10 hold_12_10 + 11 hold_12_11 + 12 hold_12_12
   + 13 hold_12_13 + 14 hold_12_14 + 15 hold_12_15 + 16 hold_12_16
   + 17 hold_12_17 + 18 hold_12_18 + 19 hold_12_19 + 20 hold_12_20
   + 21 hold_12_21 + 22 hold_12_22 + 23 hold_12_23 + 24 hold_12_24
   + 25 hold_12_25 + 26 hold_12_26 <= 80
 hold_constraint_13: hold_13_1 + hold_13_2 + hold_13_3 + hold_13_4
   + hold_13_5 + hold_13_6 + hold_13_7 + hold_13_8 + hold_13_9 + hold_13_10
   + hold_13_11 + hold_13_12 + hold_13_13 + hold_13_14 + hold_13_15
   + hold_13_16 + hold_13_17 + hold_13_18 + hold_13_19 + hold_13_20
   + hold_13_21 + hold_13_22 + hold_13_23 + hold_13_24 + hold_13_25
   + hold_13_26 <= 1
 end_constraint_13: start_13 + hold_13_1 + 2 hold_13_2 + 3 hold_13_3
   + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6 + 7 hold_13_7 + 8 hold_13_8
   + 9 hold_13_9 + 10 hold_13_10 + 11 hold_13_11 + 12 hold_13_12
   + 13 hold_13_13 + 14 hold_13_14 + 15 hold_13_15 + 16 hold_13_16
   + 17 hold_13_17 + 18 hold_13_18 + 19 hold_13_19 + 20 hold_13_20
   + 21 hold_13_21 + 22 hold_13_22 + 23 hold_13_23 + 24 hold_13_24
   + 25 hold_13_25 + 26 hold_13_26 <= 80
 hold_constraint_14: hold_14_1 + hold_14_2 + hold_14_3 + hold_14_4
   + hold_14_5 + hold_14_6 + hold_14_7 + hold_14_8 + hold_14_9 + hold_14_10
   + hold_14_11 + hold_14_12 + hold_14_13 + hold_14_14 + hold_14_15
   + hold_14_16 + hold_14_17 + hold_14_18 + hold_14_19 + hold_14_20
   + hold_14_21 + hold_14_22 + hold_14_23 + hold_14_24 + hold_14_25
   + hold_14_26 <= 1
 end_constraint_14: start_14 + hold_14_1 + 2 hold_14_2 + 3 hold_14_3
   + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6 + 7 hold_14_7 + 8 hold_14_8
   + 9 hold_14_9 + 10 hold_14_10 + 11 hold_14_11 + 12 hold_14_12
   + 13 hold_14_13 + 14 hold_14_14 + 15 hold_14_15 + 16 hold_14_16
   + 17 hold_14_17 + 18 hold_14_18 + 19 hold_14_19 + 20 hold_14_20
   + 21 hold_14_21 + 22 hold_14_22 + 23 hold_14_23 + 24 hold_14_24
   + 25 hold_14_25 + 26 hold_14_26 <= 80
 hold_constraint_15: hold_15_1 + hold_15_2 + hold_15_3 + hold_15_4
   + hold_15_5 + hold_15_6 + hold_15_7 + hold_15_8 + hold_15_9 + hold_15_10
   + hold_15_11 + hold_15_12 + hold_15_13 + hold_15_14 + hold_15_15
   + hold_15_16 + hold_15_17 + hold_15_18 + hold_15_19 + hold_15_20
   + hold_15_21 + hold_15_22 + hold_15_23 + hold_15_24 + hold_15_25
   + hold_15_26 <= 1
 end_constraint_15: start_15 + hold_15_1 + 2 hold_15_2 + 3 hold_15_3
   + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6 + 7 hold_15_7 + 8 hold_15_8
   + 9 hold_15_9 + 10 hold_15_10 + 11 hold_15_11 + 12 hold_15_12
   + 13 hold_15_13 + 14 hold_15_14 + 15 hold_15_15 + 16 hold_15_16
   + 17 hold_15_17 + 18 hold_15_18 + 19 hold_15_19 + 20 hold_15_20
   + 21 hold_15_21 + 22 hold_15_22 + 23 hold_15_23 + 24 hold_15_24
   + 25 hold_15_25 + 26 hold_15_26 <= 80
 node_flow_constraint_nic_in_0: flow_source_nic_in_0 - flow_nic_in_0_0
   - flow_nic_in_0_1 + flow_nic_out_2_nic_in_0 + flow_nic_out_4_nic_in_0
   + flow_nic_out_5_nic_in_0 + flow_nic_out_7_nic_in_0
   + flow_nic_out_11_nic_in_0 + flow_nic_out_12_nic_in_0
   + flow_nic_out_14_nic_in_0 + flow_nic_out_15_nic_in_0 = 0
 node_flow_constraint_nic_out_0: - flow_nic_out_0_sink
   - flow_nic_out_0_nic_in_2 - flow_nic_out_0_nic_in_4
   - flow_nic_out_0_nic_in_5 - flow_nic_out_0_nic_in_7
   - flow_nic_out_0_nic_in_11 - flow_nic_out_0_nic_in_12
   - flow_nic_out_0_nic_in_14 - flow_nic_out_0_nic_in_15 + flow_0_nic_out_0
   + flow_1_nic_out_0 = 0
 node_flow_constraint_nic_in_2: flow_nic_out_0_nic_in_2
   + flow_source_nic_in_2 - flow_nic_in_2_2 - flow_nic_in_2_3
   + flow_nic_out_4_nic_in_2 + flow_nic_out_5_nic_in_2
   + flow_nic_out_7_nic_in_2 + flow_nic_out_11_nic_in_2
   + flow_nic_out_12_nic_in_2 + flow_nic_out_14_nic_in_2
   + flow_nic_out_15_nic_in_2 = 0
 node_flow_constraint_nic_out_2: - flow_nic_out_2_sink
   - flow_nic_out_2_nic_in_0 - flow_nic_out_2_nic_in_4
   - flow_nic_out_2_nic_in_5 - flow_nic_out_2_nic_in_7
   - flow_nic_out_2_nic_in_11 - flow_nic_out_2_nic_in_12
   - flow_nic_out_2_nic_in_14 - flow_nic_out_2_nic_in_15 + flow_2_nic_out_2
   + flow_3_nic_out_2 = 0
 node_flow_constraint_nic_in_4: flow_nic_out_0_nic_in_4
   + flow_nic_out_2_nic_in_4 + flow_source_nic_in_4 - flow_nic_in_4_4
   + flow_nic_out_5_nic_in_4 + flow_nic_out_7_nic_in_4
   + flow_nic_out_11_nic_in_4 + flow_nic_out_12_nic_in_4
   + flow_nic_out_14_nic_in_4 + flow_nic_out_15_nic_in_4 = 0
 node_flow_constraint_nic_out_4: - flow_nic_out_4_sink
   - flow_nic_out_4_nic_in_0 - flow_nic_out_4_nic_in_2
   - flow_nic_out_4_nic_in_5 - flow_nic_out_4_nic_in_7
   - flow_nic_out_4_nic_in_11 - flow_nic_out_4_nic_in_12
   - flow_nic_out_4_nic_in_14 - flow_nic_out_4_nic_in_15 + flow_4_nic_out_4
   = 0
 node_flow_constraint_nic_in_5: flow_nic_out_0_nic_in_5
   + flow_nic_out_2_nic_in_5 + flow_nic_out_4_nic_in_5
   + flow_source_nic_in_5 - flow_nic_in_5_5 - flow_nic_in_5_6
   + flow_nic_out_7_nic_in_5 + flow_nic_out_11_nic_in_5
   + flow_nic_out_12_nic_in_5 + flow_nic_out_14_nic_in_5
   + flow_nic_out_15_nic_in_5 = 0
 node_flow_constraint_nic_out_5: - flow_nic_out_5_sink
   - flow_nic_out_5_nic_in_0 - flow_nic_out_5_nic_in_2
   - flow_nic_out_5_nic_in_4 - flow_nic_out_5_nic_in_7
   - flow_nic_out_5_nic_in_11 - flow_nic_out_5_nic_in_12
   - flow_nic_out_5_nic_in_14 - flow_nic_out_5_nic_in_15 + flow_5_nic_out_5
   + flow_6_nic_out_5 = 0
 node_flow_constraint_nic_in_7: flow_nic_out_0_nic_in_7
   + flow_nic_out_2_nic_in_7 + flow_nic_out_4_nic_in_7
   + flow_nic_out_5_nic_in_7 + flow_source_nic_in_7 - flow_nic_in_7_7
   - flow_nic_in_7_8 - flow_nic_in_7_9 - flow_nic_in_7_10
   + flow_nic_out_11_nic_in_7 + flow_nic_out_12_nic_in_7
   + flow_nic_out_14_nic_in_7 + flow_nic_out_15_nic_in_7 = 0
 node_flow_constraint_nic_out_7: - flow_nic_out_7_sink
   - flow_nic_out_7_nic_in_0 - flow_nic_out_7_nic_in_2
   - flow_nic_out_7_nic_in_4 - flow_nic_out_7_nic_in_5
   - flow_nic_out_7_nic_in_11 - flow_nic_out_7_nic_in_12
   - flow_nic_out_7_nic_in_14 - flow_nic_out_7_nic_in_15 + flow_7_nic_out_7
   + flow_8_nic_out_7 + flow_9_nic_out_7 + flow_10_nic_out_7 = 0
 node_flow_constraint_nic_in_11: flow_nic_out_0_nic_in_11
   + flow_nic_out_2_nic_in_11 + flow_nic_out_4_nic_in_11
   + flow_nic_out_5_nic_in_11 + flow_nic_out_7_nic_in_11
   + flow_source_nic_in_11 - flow_nic_in_11_11 + flow_nic_out_12_nic_in_11
   + flow_nic_out_14_nic_in_11 + flow_nic_out_15_nic_in_11 = 0
 node_flow_constraint_nic_out_11: - flow_nic_out_11_sink
   - flow_nic_out_11_nic_in_0 - flow_nic_out_11_nic_in_2
   - flow_nic_out_11_nic_in_4 - flow_nic_out_11_nic_in_5
   - flow_nic_out_11_nic_in_7 - flow_nic_out_11_nic_in_12
   - flow_nic_out_11_nic_in_14 - flow_nic_out_11_nic_in_15
   + flow_11_nic_out_11 = 0
 node_flow_constraint_nic_in_12: flow_nic_out_0_nic_in_12
   + flow_nic_out_2_nic_in_12 + flow_nic_out_4_nic_in_12
   + flow_nic_out_5_nic_in_12 + flow_nic_out_7_nic_in_12
   + flow_nic_out_11_nic_in_12 + flow_source_nic_in_12 - flow_nic_in_12_12
   - flow_nic_in_12_13 + flow_nic_out_14_nic_in_12
   + flow_nic_out_15_nic_in_12 = 0
 node_flow_constraint_nic_out_12: - flow_nic_out_12_sink
   - flow_nic_out_12_nic_in_0 - flow_nic_out_12_nic_in_2
   - flow_nic_out_12_nic_in_4 - flow_nic_out_12_nic_in_5
   - flow_nic_out_12_nic_in_7 - flow_nic_out_12_nic_in_11
   - flow_nic_out_12_nic_in_14 - flow_nic_out_12_nic_in_15
   + flow_12_nic_out_12 + flow_13_nic_out_12 = 0
 node_flow_constraint_nic_in_14: flow_nic_out_0_nic_in_14
   + flow_nic_out_2_nic_in_14 + flow_nic_out_4_nic_in_14
   + flow_nic_out_5_nic_in_14 + flow_nic_out_7_nic_in_14
   + flow_nic_out_11_nic_in_14 + flow_nic_out_12_nic_in_14
   + flow_source_nic_in_14 - flow_nic_in_14_14 + flow_nic_out_15_nic_in_14
   = 0
 node_flow_constraint_nic_out_14: - flow_nic_out_14_sink
   - flow_nic_out_14_nic_in_0 - flow_nic_out_14_nic_in_2
   - flow_nic_out_14_nic_in_4 - flow_nic_out_14_nic_in_5
   - flow_nic_out_14_nic_in_7 - flow_nic_out_14_nic_in_11
   - flow_nic_out_14_nic_in_12 - flow_nic_out_14_nic_in_15
   + flow_14_nic_out_14 = 0
 node_flow_constraint_nic_in_15: flow_nic_out_0_nic_in_15
   + flow_nic_out_2_nic_in_15 + flow_nic_out_4_nic_in_15
   + flow_nic_out_5_nic_in_15 + flow_nic_out_7_nic_in_15
   + flow_nic_out_11_nic_in_15 + flow_nic_out_12_nic_in_15
   + flow_nic_out_14_nic_in_15 + flow_source_nic_in_15 - flow_nic_in_15_15
   = 0
 node_flow_constraint_nic_out_15: - flow_nic_out_15_sink
   - flow_nic_out_15_nic_in_0 - flow_nic_out_15_nic_in_2
   - flow_nic_out_15_nic_in_4 - flow_nic_out_15_nic_in_5
   - flow_nic_out_15_nic_in_7 - flow_nic_out_15_nic_in_11
   - flow_nic_out_15_nic_in_12 - flow_nic_out_15_nic_in_14
   + flow_15_nic_out_15 = 0
 node_flow_constraint_0: flow_nic_in_0_0 - flow_0_nic_out_0 - flow_0_1
   + flow_1_0 = 0
 node_flow_constraint_1: flow_nic_in_0_1 - flow_1_nic_out_0 + flow_0_1
   - flow_1_0 = 0
 node_flow_constraint_2: flow_nic_in_2_2 - flow_2_nic_out_2 - flow_2_3
   + flow_3_2 = 0
 node_flow_constraint_3: flow_nic_in_2_3 - flow_3_nic_out_2 + flow_2_3
   - flow_3_2 = 0
 node_flow_constraint_4: flow_nic_in_4_4 - flow_4_nic_out_4 = 0
 node_flow_constraint_5: flow_nic_in_5_5 - flow_5_nic_out_5 - flow_5_6
   + flow_6_5 = 0
 node_flow_constraint_6: flow_nic_in_5_6 - flow_6_nic_out_5 + flow_5_6
   - flow_6_5 = 0
 node_flow_constraint_7: flow_nic_in_7_7 - flow_7_nic_out_7 - flow_7_8
   - flow_7_9 - flow_7_10 + flow_8_7 + flow_9_7 + flow_10_7 = 0
 node_flow_constraint_8: flow_nic_in_7_8 - flow_8_nic_out_7 + flow_7_8
   - flow_8_7 - flow_8_9 - flow_8_10 + flow_9_8 + flow_10_8 = 0
 node_flow_constraint_9: flow_nic_in_7_9 - flow_9_nic_out_7 + flow_7_9
   + flow_8_9 - flow_9_7 - flow_9_8 - flow_9_10 + flow_10_9 = 0
 node_flow_constraint_10: flow_nic_in_7_10 - flow_10_nic_out_7 + flow_7_10
   + flow_8_10 + flow_9_10 - flow_10_7 - flow_10_8 - flow_10_9 = 0
 node_flow_constraint_11: flow_nic_in_11_11 - flow_11_nic_out_11 = 0
 node_flow_constraint_12: flow_nic_in_12_12 - flow_12_nic_out_12
   - flow_12_13 + flow_13_12 = 0
 node_flow_constraint_13: flow_nic_in_12_13 - flow_13_nic_out_12
   + flow_12_13 - flow_13_12 = 0
 node_flow_constraint_14: flow_nic_in_14_14 - flow_14_nic_out_14 = 0
 node_flow_constraint_15: flow_nic_in_15_15 - flow_15_nic_out_15 = 0
 constr_is_start_zero_0: start_0 + 80 is_start_zero_0_for_source_nic_in_0
   <= 80
 constr_is_start_zero_1: start_1 + 80 is_start_zero_1_for_source_nic_in_0
   <= 80
 constr_edge_switch_start_zero_0_OR: switch_source_nic_in_0
   - is_start_zero_0_for_source_nic_in_0
   - is_start_zero_1_for_source_nic_in_0 <= 0
 constr_is_end_complete_0: start_0 + hold_0_1 + 2 hold_0_2 + 3 hold_0_3
   + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7 + 8 hold_0_8
   + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12 + 13 hold_0_13
   + 14 hold_0_14 + 15 hold_0_15 + 16 hold_0_16 + 17 hold_0_17
   + 18 hold_0_18 + 19 hold_0_19 + 20 hold_0_20 + 21 hold_0_21
   + 22 hold_0_22 + 23 hold_0_23 + 24 hold_0_24 + 25 hold_0_25
   + 26 hold_0_26 - 80 is_end_complete_0 >= 0
 constr_is_end_complete_1: start_1 + hold_1_1 + 2 hold_1_2 + 3 hold_1_3
   + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7 + 8 hold_1_8
   + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 + 12 hold_1_12 + 13 hold_1_13
   + 14 hold_1_14 + 15 hold_1_15 + 16 hold_1_16 + 17 hold_1_17
   + 18 hold_1_18 + 19 hold_1_19 + 20 hold_1_20 + 21 hold_1_21
   + 22 hold_1_22 + 23 hold_1_23 + 24 hold_1_24 + 25 hold_1_25
   + 26 hold_1_26 - 80 is_end_complete_1 >= 0
 constr_edge_switch_end_max_0_OR: switch_nic_out_0_sink - is_end_complete_0
   - is_end_complete_1 <= 0
 constr_link_match_1_0_2: 80 switch_nic_out_0_nic_in_2 - group_start_2
   + group_end_0 <= 80
 constr_link_match_2_0_2: 80 switch_nic_out_0_nic_in_2 + group_start_2
   - group_end_0 <= 80
 constr_link_match_1_0_4: 80 switch_nic_out_0_nic_in_4 + group_end_0
   - group_start_4 <= 80
 constr_link_match_2_0_4: 80 switch_nic_out_0_nic_in_4 - group_end_0
   + group_start_4 <= 80
 constr_link_match_1_0_5: 80 switch_nic_out_0_nic_in_5 + group_end_0
   - group_start_5 <= 80
 constr_link_match_2_0_5: 80 switch_nic_out_0_nic_in_5 - group_end_0
   + group_start_5 <= 80
 constr_link_match_1_0_7: 80 switch_nic_out_0_nic_in_7 + group_end_0
   - group_start_7 <= 80
 constr_link_match_2_0_7: 80 switch_nic_out_0_nic_in_7 - group_end_0
   + group_start_7 <= 80
 constr_link_match_1_0_11: 80 switch_nic_out_0_nic_in_11 + group_end_0
   - group_start_11 <= 80
 constr_link_match_2_0_11: 80 switch_nic_out_0_nic_in_11 - group_end_0
   + group_start_11 <= 80
 constr_link_match_1_0_12: 80 switch_nic_out_0_nic_in_12 + group_end_0
   - group_start_12 <= 80
 constr_link_match_2_0_12: 80 switch_nic_out_0_nic_in_12 - group_end_0
   + group_start_12 <= 80
 constr_link_match_1_0_14: 80 switch_nic_out_0_nic_in_14 + group_end_0
   - group_start_14 <= 80
 constr_link_match_2_0_14: 80 switch_nic_out_0_nic_in_14 - group_end_0
   + group_start_14 <= 80
 constr_link_match_1_0_15: 80 switch_nic_out_0_nic_in_15 + group_end_0
   - group_start_15 <= 80
 constr_link_match_2_0_15: 80 switch_nic_out_0_nic_in_15 - group_end_0
   + group_start_15 <= 80
 constr_pp_permissionA_0_1: start_0 - start_1 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 8 hold_0_8 + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12
   + 13 hold_0_13 + 14 hold_0_14 + 15 hold_0_15 + 16 hold_0_16
   + 17 hold_0_17 + 18 hold_0_18 + 19 hold_0_19 + 20 hold_0_20
   + 21 hold_0_21 + 22 hold_0_22 + 23 hold_0_23 + 24 hold_0_24
   + 25 hold_0_25 + 26 hold_0_26 + 80 pp_permission_0_1 <= 80
 constr_pp_permissionB_0_1: - start_0 + start_1 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11 - 12 hold_0_12
   - 13 hold_0_13 - 14 hold_0_14 - 15 hold_0_15 - 16 hold_0_16
   - 17 hold_0_17 - 18 hold_0_18 - 19 hold_0_19 - 20 hold_0_20
   - 21 hold_0_21 - 22 hold_0_22 - 23 hold_0_23 - 24 hold_0_24
   - 25 hold_0_25 - 26 hold_0_26 + 80 pp_permission_0_1 <= 80
 constr_internal_compute_0_1: switch_0_1 - group_tp_active_0
   - pp_permission_0_1 <= 0
 constr_pp_permissionA_1_0: - start_0 + start_1 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 8 hold_1_8 + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 + 12 hold_1_12
   + 13 hold_1_13 + 14 hold_1_14 + 15 hold_1_15 + 16 hold_1_16
   + 17 hold_1_17 + 18 hold_1_18 + 19 hold_1_19 + 20 hold_1_20
   + 21 hold_1_21 + 22 hold_1_22 + 23 hold_1_23 + 24 hold_1_24
   + 25 hold_1_25 + 26 hold_1_26 + 80 pp_permission_1_0 <= 80
 constr_pp_permissionB_1_0: start_0 - start_1 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11 - 12 hold_1_12
   - 13 hold_1_13 - 14 hold_1_14 - 15 hold_1_15 - 16 hold_1_16
   - 17 hold_1_17 - 18 hold_1_18 - 19 hold_1_19 - 20 hold_1_20
   - 21 hold_1_21 - 22 hold_1_22 - 23 hold_1_23 - 24 hold_1_24
   - 25 hold_1_25 - 26 hold_1_26 + 80 pp_permission_1_0 <= 80
 constr_internal_compute_1_0: switch_1_0 - group_tp_active_0
   - pp_permission_1_0 <= 0
 constr_is_start_zero_2: start_2 + 80 is_start_zero_2_for_source_nic_in_2
   <= 80
 constr_is_start_zero_3: start_3 + 80 is_start_zero_3_for_source_nic_in_2
   <= 80
 constr_edge_switch_start_zero_2_OR: switch_source_nic_in_2
   - is_start_zero_2_for_source_nic_in_2
   - is_start_zero_3_for_source_nic_in_2 <= 0
 constr_is_end_complete_2: start_2 + hold_2_1 + 2 hold_2_2 + 3 hold_2_3
   + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7 + 8 hold_2_8
   + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 12 hold_2_12 + 13 hold_2_13
   + 14 hold_2_14 + 15 hold_2_15 + 16 hold_2_16 + 17 hold_2_17
   + 18 hold_2_18 + 19 hold_2_19 + 20 hold_2_20 + 21 hold_2_21
   + 22 hold_2_22 + 23 hold_2_23 + 24 hold_2_24 + 25 hold_2_25
   + 26 hold_2_26 - 80 is_end_complete_2 >= 0
 constr_is_end_complete_3: start_3 + hold_3_1 + 2 hold_3_2 + 3 hold_3_3
   + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7 + 8 hold_3_8
   + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11 + 12 hold_3_12 + 13 hold_3_13
   + 14 hold_3_14 + 15 hold_3_15 + 16 hold_3_16 + 17 hold_3_17
   + 18 hold_3_18 + 19 hold_3_19 + 20 hold_3_20 + 21 hold_3_21
   + 22 hold_3_22 + 23 hold_3_23 + 24 hold_3_24 + 25 hold_3_25
   + 26 hold_3_26 - 80 is_end_complete_3 >= 0
 constr_edge_switch_end_max_2_OR: switch_nic_out_2_sink - is_end_complete_2
   - is_end_complete_3 <= 0
 constr_link_match_1_2_0: 80 switch_nic_out_2_nic_in_0 - group_start_0
   + group_end_2 <= 80
 constr_link_match_2_2_0: 80 switch_nic_out_2_nic_in_0 + group_start_0
   - group_end_2 <= 80
 constr_link_match_1_2_4: 80 switch_nic_out_2_nic_in_4 - group_start_4
   + group_end_2 <= 80
 constr_link_match_2_2_4: 80 switch_nic_out_2_nic_in_4 + group_start_4
   - group_end_2 <= 80
 constr_link_match_1_2_5: 80 switch_nic_out_2_nic_in_5 - group_start_5
   + group_end_2 <= 80
 constr_link_match_2_2_5: 80 switch_nic_out_2_nic_in_5 + group_start_5
   - group_end_2 <= 80
 constr_link_match_1_2_7: 80 switch_nic_out_2_nic_in_7 - group_start_7
   + group_end_2 <= 80
 constr_link_match_2_2_7: 80 switch_nic_out_2_nic_in_7 + group_start_7
   - group_end_2 <= 80
 constr_link_match_1_2_11: 80 switch_nic_out_2_nic_in_11 - group_start_11
   + group_end_2 <= 80
 constr_link_match_2_2_11: 80 switch_nic_out_2_nic_in_11 + group_start_11
   - group_end_2 <= 80
 constr_link_match_1_2_12: 80 switch_nic_out_2_nic_in_12 - group_start_12
   + group_end_2 <= 80
 constr_link_match_2_2_12: 80 switch_nic_out_2_nic_in_12 + group_start_12
   - group_end_2 <= 80
 constr_link_match_1_2_14: 80 switch_nic_out_2_nic_in_14 - group_start_14
   + group_end_2 <= 80
 constr_link_match_2_2_14: 80 switch_nic_out_2_nic_in_14 + group_start_14
   - group_end_2 <= 80
 constr_link_match_1_2_15: 80 switch_nic_out_2_nic_in_15 - group_start_15
   + group_end_2 <= 80
 constr_link_match_2_2_15: 80 switch_nic_out_2_nic_in_15 + group_start_15
   - group_end_2 <= 80
 constr_pp_permissionA_2_3: start_2 - start_3 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 12 hold_2_12
   + 13 hold_2_13 + 14 hold_2_14 + 15 hold_2_15 + 16 hold_2_16
   + 17 hold_2_17 + 18 hold_2_18 + 19 hold_2_19 + 20 hold_2_20
   + 21 hold_2_21 + 22 hold_2_22 + 23 hold_2_23 + 24 hold_2_24
   + 25 hold_2_25 + 26 hold_2_26 + 80 pp_permission_2_3 <= 80
 constr_pp_permissionB_2_3: - start_2 + start_3 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 - 12 hold_2_12
   - 13 hold_2_13 - 14 hold_2_14 - 15 hold_2_15 - 16 hold_2_16
   - 17 hold_2_17 - 18 hold_2_18 - 19 hold_2_19 - 20 hold_2_20
   - 21 hold_2_21 - 22 hold_2_22 - 23 hold_2_23 - 24 hold_2_24
   - 25 hold_2_25 - 26 hold_2_26 + 80 pp_permission_2_3 <= 80
 constr_internal_compute_2_3: switch_2_3 - group_tp_active_2
   - pp_permission_2_3 <= 0
 constr_pp_permissionA_3_2: - start_2 + start_3 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 8 hold_3_8 + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11 + 12 hold_3_12
   + 13 hold_3_13 + 14 hold_3_14 + 15 hold_3_15 + 16 hold_3_16
   + 17 hold_3_17 + 18 hold_3_18 + 19 hold_3_19 + 20 hold_3_20
   + 21 hold_3_21 + 22 hold_3_22 + 23 hold_3_23 + 24 hold_3_24
   + 25 hold_3_25 + 26 hold_3_26 + 80 pp_permission_3_2 <= 80
 constr_pp_permissionB_3_2: start_2 - start_3 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   - 8 hold_3_8 - 9 hold_3_9 - 10 hold_3_10 - 11 hold_3_11 - 12 hold_3_12
   - 13 hold_3_13 - 14 hold_3_14 - 15 hold_3_15 - 16 hold_3_16
   - 17 hold_3_17 - 18 hold_3_18 - 19 hold_3_19 - 20 hold_3_20
   - 21 hold_3_21 - 22 hold_3_22 - 23 hold_3_23 - 24 hold_3_24
   - 25 hold_3_25 - 26 hold_3_26 + 80 pp_permission_3_2 <= 80
 constr_internal_compute_3_2: switch_3_2 - group_tp_active_2
   - pp_permission_3_2 <= 0
 constr_is_start_zero_4: start_4 + 80 is_start_zero_4_for_source_nic_in_4
   <= 80
 constr_edge_switch_start_zero_4_OR: switch_source_nic_in_4
   - is_start_zero_4_for_source_nic_in_4 <= 0
 constr_is_end_complete_4: start_4 + hold_4_1 + 2 hold_4_2 + 3 hold_4_3
   + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7 + 8 hold_4_8
   + 9 hold_4_9 + 10 hold_4_10 + 11 hold_4_11 + 12 hold_4_12 + 13 hold_4_13
   + 14 hold_4_14 + 15 hold_4_15 + 16 hold_4_16 + 17 hold_4_17
   + 18 hold_4_18 + 19 hold_4_19 + 20 hold_4_20 + 21 hold_4_21
   + 22 hold_4_22 + 23 hold_4_23 + 24 hold_4_24 + 25 hold_4_25
   + 26 hold_4_26 - 80 is_end_complete_4 >= 0
 constr_edge_switch_end_max_4_OR: switch_nic_out_4_sink - is_end_complete_4
   <= 0
 constr_link_match_1_4_0: 80 switch_nic_out_4_nic_in_0 - group_start_0
   + group_end_4 <= 80
 constr_link_match_2_4_0: 80 switch_nic_out_4_nic_in_0 + group_start_0
   - group_end_4 <= 80
 constr_link_match_1_4_2: 80 switch_nic_out_4_nic_in_2 - group_start_2
   + group_end_4 <= 80
 constr_link_match_2_4_2: 80 switch_nic_out_4_nic_in_2 + group_start_2
   - group_end_4 <= 80
 constr_link_match_1_4_5: 80 switch_nic_out_4_nic_in_5 - group_start_5
   + group_end_4 <= 80
 constr_link_match_2_4_5: 80 switch_nic_out_4_nic_in_5 + group_start_5
   - group_end_4 <= 80
 constr_link_match_1_4_7: 80 switch_nic_out_4_nic_in_7 - group_start_7
   + group_end_4 <= 80
 constr_link_match_2_4_7: 80 switch_nic_out_4_nic_in_7 + group_start_7
   - group_end_4 <= 80
 constr_link_match_1_4_11: 80 switch_nic_out_4_nic_in_11 - group_start_11
   + group_end_4 <= 80
 constr_link_match_2_4_11: 80 switch_nic_out_4_nic_in_11 + group_start_11
   - group_end_4 <= 80
 constr_link_match_1_4_12: 80 switch_nic_out_4_nic_in_12 - group_start_12
   + group_end_4 <= 80
 constr_link_match_2_4_12: 80 switch_nic_out_4_nic_in_12 + group_start_12
   - group_end_4 <= 80
 constr_link_match_1_4_14: 80 switch_nic_out_4_nic_in_14 - group_start_14
   + group_end_4 <= 80
 constr_link_match_2_4_14: 80 switch_nic_out_4_nic_in_14 + group_start_14
   - group_end_4 <= 80
 constr_link_match_1_4_15: 80 switch_nic_out_4_nic_in_15 - group_start_15
   + group_end_4 <= 80
 constr_link_match_2_4_15: 80 switch_nic_out_4_nic_in_15 + group_start_15
   - group_end_4 <= 80
 constr_is_start_zero_5: start_5 + 80 is_start_zero_5_for_source_nic_in_5
   <= 80
 constr_is_start_zero_6: start_6 + 80 is_start_zero_6_for_source_nic_in_5
   <= 80
 constr_edge_switch_start_zero_5_OR: switch_source_nic_in_5
   - is_start_zero_5_for_source_nic_in_5
   - is_start_zero_6_for_source_nic_in_5 <= 0
 constr_is_end_complete_5: start_5 + hold_5_1 + 2 hold_5_2 + 3 hold_5_3
   + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7 + 8 hold_5_8
   + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12 + 13 hold_5_13
   + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16 + 17 hold_5_17
   + 18 hold_5_18 + 19 hold_5_19 + 20 hold_5_20 + 21 hold_5_21
   + 22 hold_5_22 + 23 hold_5_23 + 24 hold_5_24 + 25 hold_5_25
   + 26 hold_5_26 - 80 is_end_complete_5 >= 0
 constr_is_end_complete_6: start_6 + hold_6_1 + 2 hold_6_2 + 3 hold_6_3
   + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7 + 8 hold_6_8
   + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 12 hold_6_12 + 13 hold_6_13
   + 14 hold_6_14 + 15 hold_6_15 + 16 hold_6_16 + 17 hold_6_17
   + 18 hold_6_18 + 19 hold_6_19 + 20 hold_6_20 + 21 hold_6_21
   + 22 hold_6_22 + 23 hold_6_23 + 24 hold_6_24 + 25 hold_6_25
   + 26 hold_6_26 - 80 is_end_complete_6 >= 0
 constr_edge_switch_end_max_5_OR: switch_nic_out_5_sink - is_end_complete_5
   - is_end_complete_6 <= 0
 constr_link_match_1_5_0: 80 switch_nic_out_5_nic_in_0 - group_start_0
   + group_end_5 <= 80
 constr_link_match_2_5_0: 80 switch_nic_out_5_nic_in_0 + group_start_0
   - group_end_5 <= 80
 constr_link_match_1_5_2: 80 switch_nic_out_5_nic_in_2 - group_start_2
   + group_end_5 <= 80
 constr_link_match_2_5_2: 80 switch_nic_out_5_nic_in_2 + group_start_2
   - group_end_5 <= 80
 constr_link_match_1_5_4: 80 switch_nic_out_5_nic_in_4 - group_start_4
   + group_end_5 <= 80
 constr_link_match_2_5_4: 80 switch_nic_out_5_nic_in_4 + group_start_4
   - group_end_5 <= 80
 constr_link_match_1_5_7: 80 switch_nic_out_5_nic_in_7 - group_start_7
   + group_end_5 <= 80
 constr_link_match_2_5_7: 80 switch_nic_out_5_nic_in_7 + group_start_7
   - group_end_5 <= 80
 constr_link_match_1_5_11: 80 switch_nic_out_5_nic_in_11 - group_start_11
   + group_end_5 <= 80
 constr_link_match_2_5_11: 80 switch_nic_out_5_nic_in_11 + group_start_11
   - group_end_5 <= 80
 constr_link_match_1_5_12: 80 switch_nic_out_5_nic_in_12 - group_start_12
   + group_end_5 <= 80
 constr_link_match_2_5_12: 80 switch_nic_out_5_nic_in_12 + group_start_12
   - group_end_5 <= 80
 constr_link_match_1_5_14: 80 switch_nic_out_5_nic_in_14 - group_start_14
   + group_end_5 <= 80
 constr_link_match_2_5_14: 80 switch_nic_out_5_nic_in_14 + group_start_14
   - group_end_5 <= 80
 constr_link_match_1_5_15: 80 switch_nic_out_5_nic_in_15 - group_start_15
   + group_end_5 <= 80
 constr_link_match_2_5_15: 80 switch_nic_out_5_nic_in_15 + group_start_15
   - group_end_5 <= 80
 constr_pp_permissionA_5_6: start_5 - start_6 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12
   + 13 hold_5_13 + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16
   + 17 hold_5_17 + 18 hold_5_18 + 19 hold_5_19 + 20 hold_5_20
   + 21 hold_5_21 + 22 hold_5_22 + 23 hold_5_23 + 24 hold_5_24
   + 25 hold_5_25 + 26 hold_5_26 + 80 pp_permission_5_6 <= 80
 constr_pp_permissionB_5_6: - start_5 + start_6 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11 - 12 hold_5_12
   - 13 hold_5_13 - 14 hold_5_14 - 15 hold_5_15 - 16 hold_5_16
   - 17 hold_5_17 - 18 hold_5_18 - 19 hold_5_19 - 20 hold_5_20
   - 21 hold_5_21 - 22 hold_5_22 - 23 hold_5_23 - 24 hold_5_24
   - 25 hold_5_25 - 26 hold_5_26 + 80 pp_permission_5_6 <= 80
 constr_internal_compute_5_6: switch_5_6 - group_tp_active_5
   - pp_permission_5_6 <= 0
 constr_pp_permissionA_6_5: - start_5 + start_6 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 12 hold_6_12
   + 13 hold_6_13 + 14 hold_6_14 + 15 hold_6_15 + 16 hold_6_16
   + 17 hold_6_17 + 18 hold_6_18 + 19 hold_6_19 + 20 hold_6_20
   + 21 hold_6_21 + 22 hold_6_22 + 23 hold_6_23 + 24 hold_6_24
   + 25 hold_6_25 + 26 hold_6_26 + 80 pp_permission_6_5 <= 80
 constr_pp_permissionB_6_5: start_5 - start_6 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11 - 12 hold_6_12
   - 13 hold_6_13 - 14 hold_6_14 - 15 hold_6_15 - 16 hold_6_16
   - 17 hold_6_17 - 18 hold_6_18 - 19 hold_6_19 - 20 hold_6_20
   - 21 hold_6_21 - 22 hold_6_22 - 23 hold_6_23 - 24 hold_6_24
   - 25 hold_6_25 - 26 hold_6_26 + 80 pp_permission_6_5 <= 80
 constr_internal_compute_6_5: switch_6_5 - group_tp_active_5
   - pp_permission_6_5 <= 0
 constr_is_start_zero_7: start_7 + 80 is_start_zero_7_for_source_nic_in_7
   <= 80
 constr_is_start_zero_8: start_8 + 80 is_start_zero_8_for_source_nic_in_7
   <= 80
 constr_is_start_zero_9: start_9 + 80 is_start_zero_9_for_source_nic_in_7
   <= 80
 constr_is_start_zero_10: start_10
   + 80 is_start_zero_10_for_source_nic_in_7 <= 80
 constr_edge_switch_start_zero_7_OR: switch_source_nic_in_7
   - is_start_zero_7_for_source_nic_in_7
   - is_start_zero_8_for_source_nic_in_7
   - is_start_zero_9_for_source_nic_in_7
   - is_start_zero_10_for_source_nic_in_7 <= 0
 constr_is_end_complete_7: start_7 + hold_7_1 + 2 hold_7_2 + 3 hold_7_3
   + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7 + 8 hold_7_8
   + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11 + 12 hold_7_12 + 13 hold_7_13
   + 14 hold_7_14 + 15 hold_7_15 + 16 hold_7_16 + 17 hold_7_17
   + 18 hold_7_18 + 19 hold_7_19 + 20 hold_7_20 + 21 hold_7_21
   + 22 hold_7_22 + 23 hold_7_23 + 24 hold_7_24 + 25 hold_7_25
   + 26 hold_7_26 - 80 is_end_complete_7 >= 0
 constr_is_end_complete_8: start_8 + hold_8_1 + 2 hold_8_2 + 3 hold_8_3
   + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7 + 8 hold_8_8
   + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 12 hold_8_12 + 13 hold_8_13
   + 14 hold_8_14 + 15 hold_8_15 + 16 hold_8_16 + 17 hold_8_17
   + 18 hold_8_18 + 19 hold_8_19 + 20 hold_8_20 + 21 hold_8_21
   + 22 hold_8_22 + 23 hold_8_23 + 24 hold_8_24 + 25 hold_8_25
   + 26 hold_8_26 - 80 is_end_complete_8 >= 0
 constr_is_end_complete_9: start_9 + hold_9_1 + 2 hold_9_2 + 3 hold_9_3
   + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7 + 8 hold_9_8
   + 9 hold_9_9 + 10 hold_9_10 + 11 hold_9_11 + 12 hold_9_12 + 13 hold_9_13
   + 14 hold_9_14 + 15 hold_9_15 + 16 hold_9_16 + 17 hold_9_17
   + 18 hold_9_18 + 19 hold_9_19 + 20 hold_9_20 + 21 hold_9_21
   + 22 hold_9_22 + 23 hold_9_23 + 24 hold_9_24 + 25 hold_9_25
   + 26 hold_9_26 - 80 is_end_complete_9 >= 0
 constr_is_end_complete_10: start_10 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 12 hold_10_12 + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15
   + 16 hold_10_16 + 17 hold_10_17 + 18 hold_10_18 + 19 hold_10_19
   + 20 hold_10_20 + 21 hold_10_21 + 22 hold_10_22 + 23 hold_10_23
   + 24 hold_10_24 + 25 hold_10_25 + 26 hold_10_26 - 80 is_end_complete_10
   >= 0
 constr_edge_switch_end_max_7_OR: switch_nic_out_7_sink - is_end_complete_7
   - is_end_complete_8 - is_end_complete_9 - is_end_complete_10 <= 0
 constr_link_match_1_7_0: 80 switch_nic_out_7_nic_in_0 - group_start_0
   + group_end_7 <= 80
 constr_link_match_2_7_0: 80 switch_nic_out_7_nic_in_0 + group_start_0
   - group_end_7 <= 80
 constr_link_match_1_7_2: 80 switch_nic_out_7_nic_in_2 - group_start_2
   + group_end_7 <= 80
 constr_link_match_2_7_2: 80 switch_nic_out_7_nic_in_2 + group_start_2
   - group_end_7 <= 80
 constr_link_match_1_7_4: 80 switch_nic_out_7_nic_in_4 - group_start_4
   + group_end_7 <= 80
 constr_link_match_2_7_4: 80 switch_nic_out_7_nic_in_4 + group_start_4
   - group_end_7 <= 80
 constr_link_match_1_7_5: 80 switch_nic_out_7_nic_in_5 - group_start_5
   + group_end_7 <= 80
 constr_link_match_2_7_5: 80 switch_nic_out_7_nic_in_5 + group_start_5
   - group_end_7 <= 80
 constr_link_match_1_7_11: 80 switch_nic_out_7_nic_in_11 - group_start_11
   + group_end_7 <= 80
 constr_link_match_2_7_11: 80 switch_nic_out_7_nic_in_11 + group_start_11
   - group_end_7 <= 80
 constr_link_match_1_7_12: 80 switch_nic_out_7_nic_in_12 - group_start_12
   + group_end_7 <= 80
 constr_link_match_2_7_12: 80 switch_nic_out_7_nic_in_12 + group_start_12
   - group_end_7 <= 80
 constr_link_match_1_7_14: 80 switch_nic_out_7_nic_in_14 - group_start_14
   + group_end_7 <= 80
 constr_link_match_2_7_14: 80 switch_nic_out_7_nic_in_14 + group_start_14
   - group_end_7 <= 80
 constr_link_match_1_7_15: 80 switch_nic_out_7_nic_in_15 - group_start_15
   + group_end_7 <= 80
 constr_link_match_2_7_15: 80 switch_nic_out_7_nic_in_15 + group_start_15
   - group_end_7 <= 80
 constr_pp_permissionA_7_8: start_7 - start_8 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 8 hold_7_8 + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11 + 12 hold_7_12
   + 13 hold_7_13 + 14 hold_7_14 + 15 hold_7_15 + 16 hold_7_16
   + 17 hold_7_17 + 18 hold_7_18 + 19 hold_7_19 + 20 hold_7_20
   + 21 hold_7_21 + 22 hold_7_22 + 23 hold_7_23 + 24 hold_7_24
   + 25 hold_7_25 + 26 hold_7_26 + 80 pp_permission_7_8 <= 80
 constr_pp_permissionB_7_8: - start_7 + start_8 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   - 8 hold_7_8 - 9 hold_7_9 - 10 hold_7_10 - 11 hold_7_11 - 12 hold_7_12
   - 13 hold_7_13 - 14 hold_7_14 - 15 hold_7_15 - 16 hold_7_16
   - 17 hold_7_17 - 18 hold_7_18 - 19 hold_7_19 - 20 hold_7_20
   - 21 hold_7_21 - 22 hold_7_22 - 23 hold_7_23 - 24 hold_7_24
   - 25 hold_7_25 - 26 hold_7_26 + 80 pp_permission_7_8 <= 80
 constr_internal_compute_7_8: switch_7_8 - group_tp_active_7
   - pp_permission_7_8 <= 0
 constr_pp_permissionA_7_9: start_7 - start_9 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 8 hold_7_8 + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11 + 12 hold_7_12
   + 13 hold_7_13 + 14 hold_7_14 + 15 hold_7_15 + 16 hold_7_16
   + 17 hold_7_17 + 18 hold_7_18 + 19 hold_7_19 + 20 hold_7_20
   + 21 hold_7_21 + 22 hold_7_22 + 23 hold_7_23 + 24 hold_7_24
   + 25 hold_7_25 + 26 hold_7_26 + 80 pp_permission_7_9 <= 80
 constr_pp_permissionB_7_9: - start_7 + start_9 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   - 8 hold_7_8 - 9 hold_7_9 - 10 hold_7_10 - 11 hold_7_11 - 12 hold_7_12
   - 13 hold_7_13 - 14 hold_7_14 - 15 hold_7_15 - 16 hold_7_16
   - 17 hold_7_17 - 18 hold_7_18 - 19 hold_7_19 - 20 hold_7_20
   - 21 hold_7_21 - 22 hold_7_22 - 23 hold_7_23 - 24 hold_7_24
   - 25 hold_7_25 - 26 hold_7_26 + 80 pp_permission_7_9 <= 80
 constr_internal_compute_7_9: switch_7_9 - group_tp_active_7
   - pp_permission_7_9 <= 0
 constr_pp_permissionA_7_10: start_7 - start_10 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 8 hold_7_8 + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11 + 12 hold_7_12
   + 13 hold_7_13 + 14 hold_7_14 + 15 hold_7_15 + 16 hold_7_16
   + 17 hold_7_17 + 18 hold_7_18 + 19 hold_7_19 + 20 hold_7_20
   + 21 hold_7_21 + 22 hold_7_22 + 23 hold_7_23 + 24 hold_7_24
   + 25 hold_7_25 + 26 hold_7_26 + 80 pp_permission_7_10 <= 80
 constr_pp_permissionB_7_10: - start_7 + start_10 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   - 8 hold_7_8 - 9 hold_7_9 - 10 hold_7_10 - 11 hold_7_11 - 12 hold_7_12
   - 13 hold_7_13 - 14 hold_7_14 - 15 hold_7_15 - 16 hold_7_16
   - 17 hold_7_17 - 18 hold_7_18 - 19 hold_7_19 - 20 hold_7_20
   - 21 hold_7_21 - 22 hold_7_22 - 23 hold_7_23 - 24 hold_7_24
   - 25 hold_7_25 - 26 hold_7_26 + 80 pp_permission_7_10 <= 80
 constr_internal_compute_7_10: switch_7_10 - group_tp_active_7
   - pp_permission_7_10 <= 0
 constr_pp_permissionA_8_7: - start_7 + start_8 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 12 hold_8_12
   + 13 hold_8_13 + 14 hold_8_14 + 15 hold_8_15 + 16 hold_8_16
   + 17 hold_8_17 + 18 hold_8_18 + 19 hold_8_19 + 20 hold_8_20
   + 21 hold_8_21 + 22 hold_8_22 + 23 hold_8_23 + 24 hold_8_24
   + 25 hold_8_25 + 26 hold_8_26 + 80 pp_permission_8_7 <= 80
 constr_pp_permissionB_8_7: start_7 - start_8 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11 - 12 hold_8_12
   - 13 hold_8_13 - 14 hold_8_14 - 15 hold_8_15 - 16 hold_8_16
   - 17 hold_8_17 - 18 hold_8_18 - 19 hold_8_19 - 20 hold_8_20
   - 21 hold_8_21 - 22 hold_8_22 - 23 hold_8_23 - 24 hold_8_24
   - 25 hold_8_25 - 26 hold_8_26 + 80 pp_permission_8_7 <= 80
 constr_internal_compute_8_7: switch_8_7 - group_tp_active_7
   - pp_permission_8_7 <= 0
 constr_pp_permissionA_8_9: start_8 - start_9 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 12 hold_8_12
   + 13 hold_8_13 + 14 hold_8_14 + 15 hold_8_15 + 16 hold_8_16
   + 17 hold_8_17 + 18 hold_8_18 + 19 hold_8_19 + 20 hold_8_20
   + 21 hold_8_21 + 22 hold_8_22 + 23 hold_8_23 + 24 hold_8_24
   + 25 hold_8_25 + 26 hold_8_26 + 80 pp_permission_8_9 <= 80
 constr_pp_permissionB_8_9: - start_8 + start_9 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11 - 12 hold_8_12
   - 13 hold_8_13 - 14 hold_8_14 - 15 hold_8_15 - 16 hold_8_16
   - 17 hold_8_17 - 18 hold_8_18 - 19 hold_8_19 - 20 hold_8_20
   - 21 hold_8_21 - 22 hold_8_22 - 23 hold_8_23 - 24 hold_8_24
   - 25 hold_8_25 - 26 hold_8_26 + 80 pp_permission_8_9 <= 80
 constr_internal_compute_8_9: switch_8_9 - group_tp_active_7
   - pp_permission_8_9 <= 0
 constr_pp_permissionA_8_10: start_8 - start_10 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 12 hold_8_12
   + 13 hold_8_13 + 14 hold_8_14 + 15 hold_8_15 + 16 hold_8_16
   + 17 hold_8_17 + 18 hold_8_18 + 19 hold_8_19 + 20 hold_8_20
   + 21 hold_8_21 + 22 hold_8_22 + 23 hold_8_23 + 24 hold_8_24
   + 25 hold_8_25 + 26 hold_8_26 + 80 pp_permission_8_10 <= 80
 constr_pp_permissionB_8_10: - start_8 + start_10 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11 - 12 hold_8_12
   - 13 hold_8_13 - 14 hold_8_14 - 15 hold_8_15 - 16 hold_8_16
   - 17 hold_8_17 - 18 hold_8_18 - 19 hold_8_19 - 20 hold_8_20
   - 21 hold_8_21 - 22 hold_8_22 - 23 hold_8_23 - 24 hold_8_24
   - 25 hold_8_25 - 26 hold_8_26 + 80 pp_permission_8_10 <= 80
 constr_internal_compute_8_10: switch_8_10 - group_tp_active_7
   - pp_permission_8_10 <= 0
 constr_pp_permissionA_9_7: - start_7 + start_9 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 8 hold_9_8 + 9 hold_9_9 + 10 hold_9_10 + 11 hold_9_11 + 12 hold_9_12
   + 13 hold_9_13 + 14 hold_9_14 + 15 hold_9_15 + 16 hold_9_16
   + 17 hold_9_17 + 18 hold_9_18 + 19 hold_9_19 + 20 hold_9_20
   + 21 hold_9_21 + 22 hold_9_22 + 23 hold_9_23 + 24 hold_9_24
   + 25 hold_9_25 + 26 hold_9_26 + 80 pp_permission_9_7 <= 80
 constr_pp_permissionB_9_7: start_7 - start_9 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   - 8 hold_9_8 - 9 hold_9_9 - 10 hold_9_10 - 11 hold_9_11 - 12 hold_9_12
   - 13 hold_9_13 - 14 hold_9_14 - 15 hold_9_15 - 16 hold_9_16
   - 17 hold_9_17 - 18 hold_9_18 - 19 hold_9_19 - 20 hold_9_20
   - 21 hold_9_21 - 22 hold_9_22 - 23 hold_9_23 - 24 hold_9_24
   - 25 hold_9_25 - 26 hold_9_26 + 80 pp_permission_9_7 <= 80
 constr_internal_compute_9_7: switch_9_7 - group_tp_active_7
   - pp_permission_9_7 <= 0
 constr_pp_permissionA_9_8: - start_8 + start_9 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 8 hold_9_8 + 9 hold_9_9 + 10 hold_9_10 + 11 hold_9_11 + 12 hold_9_12
   + 13 hold_9_13 + 14 hold_9_14 + 15 hold_9_15 + 16 hold_9_16
   + 17 hold_9_17 + 18 hold_9_18 + 19 hold_9_19 + 20 hold_9_20
   + 21 hold_9_21 + 22 hold_9_22 + 23 hold_9_23 + 24 hold_9_24
   + 25 hold_9_25 + 26 hold_9_26 + 80 pp_permission_9_8 <= 80
 constr_pp_permissionB_9_8: start_8 - start_9 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   - 8 hold_9_8 - 9 hold_9_9 - 10 hold_9_10 - 11 hold_9_11 - 12 hold_9_12
   - 13 hold_9_13 - 14 hold_9_14 - 15 hold_9_15 - 16 hold_9_16
   - 17 hold_9_17 - 18 hold_9_18 - 19 hold_9_19 - 20 hold_9_20
   - 21 hold_9_21 - 22 hold_9_22 - 23 hold_9_23 - 24 hold_9_24
   - 25 hold_9_25 - 26 hold_9_26 + 80 pp_permission_9_8 <= 80
 constr_internal_compute_9_8: switch_9_8 - group_tp_active_7
   - pp_permission_9_8 <= 0
 constr_pp_permissionA_9_10: start_9 - start_10 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 8 hold_9_8 + 9 hold_9_9 + 10 hold_9_10 + 11 hold_9_11 + 12 hold_9_12
   + 13 hold_9_13 + 14 hold_9_14 + 15 hold_9_15 + 16 hold_9_16
   + 17 hold_9_17 + 18 hold_9_18 + 19 hold_9_19 + 20 hold_9_20
   + 21 hold_9_21 + 22 hold_9_22 + 23 hold_9_23 + 24 hold_9_24
   + 25 hold_9_25 + 26 hold_9_26 + 80 pp_permission_9_10 <= 80
 constr_pp_permissionB_9_10: - start_9 + start_10 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   - 8 hold_9_8 - 9 hold_9_9 - 10 hold_9_10 - 11 hold_9_11 - 12 hold_9_12
   - 13 hold_9_13 - 14 hold_9_14 - 15 hold_9_15 - 16 hold_9_16
   - 17 hold_9_17 - 18 hold_9_18 - 19 hold_9_19 - 20 hold_9_20
   - 21 hold_9_21 - 22 hold_9_22 - 23 hold_9_23 - 24 hold_9_24
   - 25 hold_9_25 - 26 hold_9_26 + 80 pp_permission_9_10 <= 80
 constr_internal_compute_9_10: switch_9_10 - group_tp_active_7
   - pp_permission_9_10 <= 0
 constr_pp_permissionA_10_7: - start_7 + start_10 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 12 hold_10_12 + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15
   + 16 hold_10_16 + 17 hold_10_17 + 18 hold_10_18 + 19 hold_10_19
   + 20 hold_10_20 + 21 hold_10_21 + 22 hold_10_22 + 23 hold_10_23
   + 24 hold_10_24 + 25 hold_10_25 + 26 hold_10_26 + 80 pp_permission_10_7
   <= 80
 constr_pp_permissionB_10_7: start_7 - start_10 - hold_10_1 - 2 hold_10_2
   - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6 - 7 hold_10_7
   - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10 - 11 hold_10_11
   - 12 hold_10_12 - 13 hold_10_13 - 14 hold_10_14 - 15 hold_10_15
   - 16 hold_10_16 - 17 hold_10_17 - 18 hold_10_18 - 19 hold_10_19
   - 20 hold_10_20 - 21 hold_10_21 - 22 hold_10_22 - 23 hold_10_23
   - 24 hold_10_24 - 25 hold_10_25 - 26 hold_10_26 + 80 pp_permission_10_7
   <= 80
 constr_internal_compute_10_7: switch_10_7 - group_tp_active_7
   - pp_permission_10_7 <= 0
 constr_pp_permissionA_10_8: - start_8 + start_10 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 12 hold_10_12 + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15
   + 16 hold_10_16 + 17 hold_10_17 + 18 hold_10_18 + 19 hold_10_19
   + 20 hold_10_20 + 21 hold_10_21 + 22 hold_10_22 + 23 hold_10_23
   + 24 hold_10_24 + 25 hold_10_25 + 26 hold_10_26 + 80 pp_permission_10_8
   <= 80
 constr_pp_permissionB_10_8: start_8 - start_10 - hold_10_1 - 2 hold_10_2
   - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6 - 7 hold_10_7
   - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10 - 11 hold_10_11
   - 12 hold_10_12 - 13 hold_10_13 - 14 hold_10_14 - 15 hold_10_15
   - 16 hold_10_16 - 17 hold_10_17 - 18 hold_10_18 - 19 hold_10_19
   - 20 hold_10_20 - 21 hold_10_21 - 22 hold_10_22 - 23 hold_10_23
   - 24 hold_10_24 - 25 hold_10_25 - 26 hold_10_26 + 80 pp_permission_10_8
   <= 80
 constr_internal_compute_10_8: switch_10_8 - group_tp_active_7
   - pp_permission_10_8 <= 0
 constr_pp_permissionA_10_9: - start_9 + start_10 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 12 hold_10_12 + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15
   + 16 hold_10_16 + 17 hold_10_17 + 18 hold_10_18 + 19 hold_10_19
   + 20 hold_10_20 + 21 hold_10_21 + 22 hold_10_22 + 23 hold_10_23
   + 24 hold_10_24 + 25 hold_10_25 + 26 hold_10_26 + 80 pp_permission_10_9
   <= 80
 constr_pp_permissionB_10_9: start_9 - start_10 - hold_10_1 - 2 hold_10_2
   - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6 - 7 hold_10_7
   - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10 - 11 hold_10_11
   - 12 hold_10_12 - 13 hold_10_13 - 14 hold_10_14 - 15 hold_10_15
   - 16 hold_10_16 - 17 hold_10_17 - 18 hold_10_18 - 19 hold_10_19
   - 20 hold_10_20 - 21 hold_10_21 - 22 hold_10_22 - 23 hold_10_23
   - 24 hold_10_24 - 25 hold_10_25 - 26 hold_10_26 + 80 pp_permission_10_9
   <= 80
 constr_internal_compute_10_9: switch_10_9 - group_tp_active_7
   - pp_permission_10_9 <= 0
 constr_is_start_zero_11: start_11
   + 80 is_start_zero_11_for_source_nic_in_11 <= 80
 constr_edge_switch_start_zero_11_OR: switch_source_nic_in_11
   - is_start_zero_11_for_source_nic_in_11 <= 0
 constr_is_end_complete_11: start_11 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 12 hold_11_12 + 13 hold_11_13 + 14 hold_11_14 + 15 hold_11_15
   + 16 hold_11_16 + 17 hold_11_17 + 18 hold_11_18 + 19 hold_11_19
   + 20 hold_11_20 + 21 hold_11_21 + 22 hold_11_22 + 23 hold_11_23
   + 24 hold_11_24 + 25 hold_11_25 + 26 hold_11_26 - 80 is_end_complete_11
   >= 0
 constr_edge_switch_end_max_11_OR: switch_nic_out_11_sink
   - is_end_complete_11 <= 0
 constr_link_match_1_11_0: 80 switch_nic_out_11_nic_in_0 - group_start_0
   + group_end_11 <= 80
 constr_link_match_2_11_0: 80 switch_nic_out_11_nic_in_0 + group_start_0
   - group_end_11 <= 80
 constr_link_match_1_11_2: 80 switch_nic_out_11_nic_in_2 - group_start_2
   + group_end_11 <= 80
 constr_link_match_2_11_2: 80 switch_nic_out_11_nic_in_2 + group_start_2
   - group_end_11 <= 80
 constr_link_match_1_11_4: 80 switch_nic_out_11_nic_in_4 - group_start_4
   + group_end_11 <= 80
 constr_link_match_2_11_4: 80 switch_nic_out_11_nic_in_4 + group_start_4
   - group_end_11 <= 80
 constr_link_match_1_11_5: 80 switch_nic_out_11_nic_in_5 - group_start_5
   + group_end_11 <= 80
 constr_link_match_2_11_5: 80 switch_nic_out_11_nic_in_5 + group_start_5
   - group_end_11 <= 80
 constr_link_match_1_11_7: 80 switch_nic_out_11_nic_in_7 - group_start_7
   + group_end_11 <= 80
 constr_link_match_2_11_7: 80 switch_nic_out_11_nic_in_7 + group_start_7
   - group_end_11 <= 80
 constr_link_match_1_11_12: 80 switch_nic_out_11_nic_in_12 - group_start_12
   + group_end_11 <= 80
 constr_link_match_2_11_12: 80 switch_nic_out_11_nic_in_12 + group_start_12
   - group_end_11 <= 80
 constr_link_match_1_11_14: 80 switch_nic_out_11_nic_in_14 - group_start_14
   + group_end_11 <= 80
 constr_link_match_2_11_14: 80 switch_nic_out_11_nic_in_14 + group_start_14
   - group_end_11 <= 80
 constr_link_match_1_11_15: 80 switch_nic_out_11_nic_in_15 - group_start_15
   + group_end_11 <= 80
 constr_link_match_2_11_15: 80 switch_nic_out_11_nic_in_15 + group_start_15
   - group_end_11 <= 80
 constr_is_start_zero_12: start_12
   + 80 is_start_zero_12_for_source_nic_in_12 <= 80
 constr_is_start_zero_13: start_13
   + 80 is_start_zero_13_for_source_nic_in_12 <= 80
 constr_edge_switch_start_zero_12_OR: switch_source_nic_in_12
   - is_start_zero_12_for_source_nic_in_12
   - is_start_zero_13_for_source_nic_in_12 <= 0
 constr_is_end_complete_12: start_12 + hold_12_1 + 2 hold_12_2
   + 3 hold_12_3 + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6 + 7 hold_12_7
   + 8 hold_12_8 + 9 hold_12_9 + 10 hold_12_10 + 11 hold_12_11
   + 12 hold_12_12 + 13 hold_12_13 + 14 hold_12_14 + 15 hold_12_15
   + 16 hold_12_16 + 17 hold_12_17 + 18 hold_12_18 + 19 hold_12_19
   + 20 hold_12_20 + 21 hold_12_21 + 22 hold_12_22 + 23 hold_12_23
   + 24 hold_12_24 + 25 hold_12_25 + 26 hold_12_26 - 80 is_end_complete_12
   >= 0
 constr_is_end_complete_13: start_13 + hold_13_1 + 2 hold_13_2
   + 3 hold_13_3 + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6 + 7 hold_13_7
   + 8 hold_13_8 + 9 hold_13_9 + 10 hold_13_10 + 11 hold_13_11
   + 12 hold_13_12 + 13 hold_13_13 + 14 hold_13_14 + 15 hold_13_15
   + 16 hold_13_16 + 17 hold_13_17 + 18 hold_13_18 + 19 hold_13_19
   + 20 hold_13_20 + 21 hold_13_21 + 22 hold_13_22 + 23 hold_13_23
   + 24 hold_13_24 + 25 hold_13_25 + 26 hold_13_26 - 80 is_end_complete_13
   >= 0
 constr_edge_switch_end_max_12_OR: switch_nic_out_12_sink
   - is_end_complete_12 - is_end_complete_13 <= 0
 constr_link_match_1_12_0: 80 switch_nic_out_12_nic_in_0 - group_start_0
   + group_end_12 <= 80
 constr_link_match_2_12_0: 80 switch_nic_out_12_nic_in_0 + group_start_0
   - group_end_12 <= 80
 constr_link_match_1_12_2: 80 switch_nic_out_12_nic_in_2 - group_start_2
   + group_end_12 <= 80
 constr_link_match_2_12_2: 80 switch_nic_out_12_nic_in_2 + group_start_2
   - group_end_12 <= 80
 constr_link_match_1_12_4: 80 switch_nic_out_12_nic_in_4 - group_start_4
   + group_end_12 <= 80
 constr_link_match_2_12_4: 80 switch_nic_out_12_nic_in_4 + group_start_4
   - group_end_12 <= 80
 constr_link_match_1_12_5: 80 switch_nic_out_12_nic_in_5 - group_start_5
   + group_end_12 <= 80
 constr_link_match_2_12_5: 80 switch_nic_out_12_nic_in_5 + group_start_5
   - group_end_12 <= 80
 constr_link_match_1_12_7: 80 switch_nic_out_12_nic_in_7 - group_start_7
   + group_end_12 <= 80
 constr_link_match_2_12_7: 80 switch_nic_out_12_nic_in_7 + group_start_7
   - group_end_12 <= 80
 constr_link_match_1_12_11: 80 switch_nic_out_12_nic_in_11 - group_start_11
   + group_end_12 <= 80
 constr_link_match_2_12_11: 80 switch_nic_out_12_nic_in_11 + group_start_11
   - group_end_12 <= 80
 constr_link_match_1_12_14: 80 switch_nic_out_12_nic_in_14 - group_start_14
   + group_end_12 <= 80
 constr_link_match_2_12_14: 80 switch_nic_out_12_nic_in_14 + group_start_14
   - group_end_12 <= 80
 constr_link_match_1_12_15: 80 switch_nic_out_12_nic_in_15 - group_start_15
   + group_end_12 <= 80
 constr_link_match_2_12_15: 80 switch_nic_out_12_nic_in_15 + group_start_15
   - group_end_12 <= 80
 constr_pp_permissionA_12_13: start_12 - start_13 + hold_12_1 + 2 hold_12_2
   + 3 hold_12_3 + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6 + 7 hold_12_7
   + 8 hold_12_8 + 9 hold_12_9 + 10 hold_12_10 + 11 hold_12_11
   + 12 hold_12_12 + 13 hold_12_13 + 14 hold_12_14 + 15 hold_12_15
   + 16 hold_12_16 + 17 hold_12_17 + 18 hold_12_18 + 19 hold_12_19
   + 20 hold_12_20 + 21 hold_12_21 + 22 hold_12_22 + 23 hold_12_23
   + 24 hold_12_24 + 25 hold_12_25 + 26 hold_12_26 + 80 pp_permission_12_13
   <= 80
 constr_pp_permissionB_12_13: - start_12 + start_13 - hold_12_1
   - 2 hold_12_2 - 3 hold_12_3 - 4 hold_12_4 - 5 hold_12_5 - 6 hold_12_6
   - 7 hold_12_7 - 8 hold_12_8 - 9 hold_12_9 - 10 hold_12_10
   - 11 hold_12_11 - 12 hold_12_12 - 13 hold_12_13 - 14 hold_12_14
   - 15 hold_12_15 - 16 hold_12_16 - 17 hold_12_17 - 18 hold_12_18
   - 19 hold_12_19 - 20 hold_12_20 - 21 hold_12_21 - 22 hold_12_22
   - 23 hold_12_23 - 24 hold_12_24 - 25 hold_12_25 - 26 hold_12_26
   + 80 pp_permission_12_13 <= 80
 constr_internal_compute_12_13: switch_12_13 - group_tp_active_12
   - pp_permission_12_13 <= 0
 constr_pp_permissionA_13_12: - start_12 + start_13 + hold_13_1
   + 2 hold_13_2 + 3 hold_13_3 + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6
   + 7 hold_13_7 + 8 hold_13_8 + 9 hold_13_9 + 10 hold_13_10
   + 11 hold_13_11 + 12 hold_13_12 + 13 hold_13_13 + 14 hold_13_14
   + 15 hold_13_15 + 16 hold_13_16 + 17 hold_13_17 + 18 hold_13_18
   + 19 hold_13_19 + 20 hold_13_20 + 21 hold_13_21 + 22 hold_13_22
   + 23 hold_13_23 + 24 hold_13_24 + 25 hold_13_25 + 26 hold_13_26
   + 80 pp_permission_13_12 <= 80
 constr_pp_permissionB_13_12: start_12 - start_13 - hold_13_1 - 2 hold_13_2
   - 3 hold_13_3 - 4 hold_13_4 - 5 hold_13_5 - 6 hold_13_6 - 7 hold_13_7
   - 8 hold_13_8 - 9 hold_13_9 - 10 hold_13_10 - 11 hold_13_11
   - 12 hold_13_12 - 13 hold_13_13 - 14 hold_13_14 - 15 hold_13_15
   - 16 hold_13_16 - 17 hold_13_17 - 18 hold_13_18 - 19 hold_13_19
   - 20 hold_13_20 - 21 hold_13_21 - 22 hold_13_22 - 23 hold_13_23
   - 24 hold_13_24 - 25 hold_13_25 - 26 hold_13_26 + 80 pp_permission_13_12
   <= 80
 constr_internal_compute_13_12: switch_13_12 - group_tp_active_12
   - pp_permission_13_12 <= 0
 constr_is_start_zero_14: start_14
   + 80 is_start_zero_14_for_source_nic_in_14 <= 80
 constr_edge_switch_start_zero_14_OR: switch_source_nic_in_14
   - is_start_zero_14_for_source_nic_in_14 <= 0
 constr_is_end_complete_14: start_14 + hold_14_1 + 2 hold_14_2
   + 3 hold_14_3 + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6 + 7 hold_14_7
   + 8 hold_14_8 + 9 hold_14_9 + 10 hold_14_10 + 11 hold_14_11
   + 12 hold_14_12 + 13 hold_14_13 + 14 hold_14_14 + 15 hold_14_15
   + 16 hold_14_16 + 17 hold_14_17 + 18 hold_14_18 + 19 hold_14_19
   + 20 hold_14_20 + 21 hold_14_21 + 22 hold_14_22 + 23 hold_14_23
   + 24 hold_14_24 + 25 hold_14_25 + 26 hold_14_26 - 80 is_end_complete_14
   >= 0
 constr_edge_switch_end_max_14_OR: switch_nic_out_14_sink
   - is_end_complete_14 <= 0
 constr_link_match_1_14_0: 80 switch_nic_out_14_nic_in_0 - group_start_0
   + group_end_14 <= 80
 constr_link_match_2_14_0: 80 switch_nic_out_14_nic_in_0 + group_start_0
   - group_end_14 <= 80
 constr_link_match_1_14_2: 80 switch_nic_out_14_nic_in_2 - group_start_2
   + group_end_14 <= 80
 constr_link_match_2_14_2: 80 switch_nic_out_14_nic_in_2 + group_start_2
   - group_end_14 <= 80
 constr_link_match_1_14_4: 80 switch_nic_out_14_nic_in_4 - group_start_4
   + group_end_14 <= 80
 constr_link_match_2_14_4: 80 switch_nic_out_14_nic_in_4 + group_start_4
   - group_end_14 <= 80
 constr_link_match_1_14_5: 80 switch_nic_out_14_nic_in_5 - group_start_5
   + group_end_14 <= 80
 constr_link_match_2_14_5: 80 switch_nic_out_14_nic_in_5 + group_start_5
   - group_end_14 <= 80
 constr_link_match_1_14_7: 80 switch_nic_out_14_nic_in_7 - group_start_7
   + group_end_14 <= 80
 constr_link_match_2_14_7: 80 switch_nic_out_14_nic_in_7 + group_start_7
   - group_end_14 <= 80
 constr_link_match_1_14_11: 80 switch_nic_out_14_nic_in_11 - group_start_11
   + group_end_14 <= 80
 constr_link_match_2_14_11: 80 switch_nic_out_14_nic_in_11 + group_start_11
   - group_end_14 <= 80
 constr_link_match_1_14_12: 80 switch_nic_out_14_nic_in_12 - group_start_12
   + group_end_14 <= 80
 constr_link_match_2_14_12: 80 switch_nic_out_14_nic_in_12 + group_start_12
   - group_end_14 <= 80
 constr_link_match_1_14_15: 80 switch_nic_out_14_nic_in_15 - group_start_15
   + group_end_14 <= 80
 constr_link_match_2_14_15: 80 switch_nic_out_14_nic_in_15 + group_start_15
   - group_end_14 <= 80
 constr_is_start_zero_15: start_15
   + 80 is_start_zero_15_for_source_nic_in_15 <= 80
 constr_edge_switch_start_zero_15_OR: switch_source_nic_in_15
   - is_start_zero_15_for_source_nic_in_15 <= 0
 constr_is_end_complete_15: start_15 + hold_15_1 + 2 hold_15_2
   + 3 hold_15_3 + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6 + 7 hold_15_7
   + 8 hold_15_8 + 9 hold_15_9 + 10 hold_15_10 + 11 hold_15_11
   + 12 hold_15_12 + 13 hold_15_13 + 14 hold_15_14 + 15 hold_15_15
   + 16 hold_15_16 + 17 hold_15_17 + 18 hold_15_18 + 19 hold_15_19
   + 20 hold_15_20 + 21 hold_15_21 + 22 hold_15_22 + 23 hold_15_23
   + 24 hold_15_24 + 25 hold_15_25 + 26 hold_15_26 - 80 is_end_complete_15
   >= 0
 constr_edge_switch_end_max_15_OR: switch_nic_out_15_sink
   - is_end_complete_15 <= 0
 constr_link_match_1_15_0: 80 switch_nic_out_15_nic_in_0 - group_start_0
   + group_end_15 <= 80
 constr_link_match_2_15_0: 80 switch_nic_out_15_nic_in_0 + group_start_0
   - group_end_15 <= 80
 constr_link_match_1_15_2: 80 switch_nic_out_15_nic_in_2 - group_start_2
   + group_end_15 <= 80
 constr_link_match_2_15_2: 80 switch_nic_out_15_nic_in_2 + group_start_2
   - group_end_15 <= 80
 constr_link_match_1_15_4: 80 switch_nic_out_15_nic_in_4 - group_start_4
   + group_end_15 <= 80
 constr_link_match_2_15_4: 80 switch_nic_out_15_nic_in_4 + group_start_4
   - group_end_15 <= 80
 constr_link_match_1_15_5: 80 switch_nic_out_15_nic_in_5 - group_start_5
   + group_end_15 <= 80
 constr_link_match_2_15_5: 80 switch_nic_out_15_nic_in_5 + group_start_5
   - group_end_15 <= 80
 constr_link_match_1_15_7: 80 switch_nic_out_15_nic_in_7 - group_start_7
   + group_end_15 <= 80
 constr_link_match_2_15_7: 80 switch_nic_out_15_nic_in_7 + group_start_7
   - group_end_15 <= 80
 constr_link_match_1_15_11: 80 switch_nic_out_15_nic_in_11 - group_start_11
   + group_end_15 <= 80
 constr_link_match_2_15_11: 80 switch_nic_out_15_nic_in_11 + group_start_11
   - group_end_15 <= 80
 constr_link_match_1_15_12: 80 switch_nic_out_15_nic_in_12 - group_start_12
   + group_end_15 <= 80
 constr_link_match_2_15_12: 80 switch_nic_out_15_nic_in_12 + group_start_12
   - group_end_15 <= 80
 constr_link_match_1_15_14: 80 switch_nic_out_15_nic_in_14 - group_start_14
   + group_end_15 <= 80
 constr_link_match_2_15_14: 80 switch_nic_out_15_nic_in_14 + group_start_14
   - group_end_15 <= 80
 link_activation_dst_nic_in_0_0: switch_nic_in_0_0 - is_active_0 <= 0
 link_activation_src_0_nic_out_0: switch_0_nic_out_0 - is_active_0 <= 0
 link_activation_dst_nic_in_0_1: switch_nic_in_0_1 - is_active_1 <= 0
 link_activation_src_1_nic_out_0: switch_1_nic_out_0 - is_active_1 <= 0
 link_activation_src_0_1: switch_0_1 - is_active_0 <= 0
 link_activation_dst_0_1: switch_0_1 - is_active_1 <= 0
 link_activation_src_1_0: switch_1_0 - is_active_1 <= 0
 link_activation_dst_1_0: switch_1_0 - is_active_0 <= 0
 link_activation_dst_nic_in_2_2: switch_nic_in_2_2 - is_active_2 <= 0
 link_activation_src_2_nic_out_2: switch_2_nic_out_2 - is_active_2 <= 0
 link_activation_dst_nic_in_2_3: switch_nic_in_2_3 - is_active_3 <= 0
 link_activation_src_3_nic_out_2: switch_3_nic_out_2 - is_active_3 <= 0
 link_activation_src_2_3: switch_2_3 - is_active_2 <= 0
 link_activation_dst_2_3: switch_2_3 - is_active_3 <= 0
 link_activation_src_3_2: switch_3_2 - is_active_3 <= 0
 link_activation_dst_3_2: switch_3_2 - is_active_2 <= 0
 link_activation_dst_nic_in_4_4: switch_nic_in_4_4 - is_active_4 <= 0
 link_activation_src_4_nic_out_4: switch_4_nic_out_4 - is_active_4 <= 0
 link_activation_dst_nic_in_5_5: switch_nic_in_5_5 - is_active_5 <= 0
 link_activation_src_5_nic_out_5: switch_5_nic_out_5 - is_active_5 <= 0
 link_activation_dst_nic_in_5_6: switch_nic_in_5_6 - is_active_6 <= 0
 link_activation_src_6_nic_out_5: switch_6_nic_out_5 - is_active_6 <= 0
 link_activation_src_5_6: switch_5_6 - is_active_5 <= 0
 link_activation_dst_5_6: switch_5_6 - is_active_6 <= 0
 link_activation_src_6_5: switch_6_5 - is_active_6 <= 0
 link_activation_dst_6_5: switch_6_5 - is_active_5 <= 0
 link_activation_dst_nic_in_7_7: switch_nic_in_7_7 - is_active_7 <= 0
 link_activation_src_7_nic_out_7: switch_7_nic_out_7 - is_active_7 <= 0
 link_activation_dst_nic_in_7_8: switch_nic_in_7_8 - is_active_8 <= 0
 link_activation_src_8_nic_out_7: switch_8_nic_out_7 - is_active_8 <= 0
 link_activation_dst_nic_in_7_9: switch_nic_in_7_9 - is_active_9 <= 0
 link_activation_src_9_nic_out_7: switch_9_nic_out_7 - is_active_9 <= 0
 link_activation_dst_nic_in_7_10: switch_nic_in_7_10 - is_active_10 <= 0
 link_activation_src_10_nic_out_7: switch_10_nic_out_7 - is_active_10 <= 0
 link_activation_src_7_8: switch_7_8 - is_active_7 <= 0
 link_activation_dst_7_8: switch_7_8 - is_active_8 <= 0
 link_activation_src_7_9: switch_7_9 - is_active_7 <= 0
 link_activation_dst_7_9: switch_7_9 - is_active_9 <= 0
 link_activation_src_7_10: switch_7_10 - is_active_7 <= 0
 link_activation_dst_7_10: switch_7_10 - is_active_10 <= 0
 link_activation_src_8_7: switch_8_7 - is_active_8 <= 0
 link_activation_dst_8_7: switch_8_7 - is_active_7 <= 0
 link_activation_src_8_9: switch_8_9 - is_active_8 <= 0
 link_activation_dst_8_9: switch_8_9 - is_active_9 <= 0
 link_activation_src_8_10: switch_8_10 - is_active_8 <= 0
 link_activation_dst_8_10: switch_8_10 - is_active_10 <= 0
 link_activation_src_9_7: switch_9_7 - is_active_9 <= 0
 link_activation_dst_9_7: switch_9_7 - is_active_7 <= 0
 link_activation_src_9_8: switch_9_8 - is_active_9 <= 0
 link_activation_dst_9_8: switch_9_8 - is_active_8 <= 0
 link_activation_src_9_10: switch_9_10 - is_active_9 <= 0
 link_activation_dst_9_10: switch_9_10 - is_active_10 <= 0
 link_activation_src_10_7: switch_10_7 - is_active_10 <= 0
 link_activation_dst_10_7: switch_10_7 - is_active_7 <= 0
 link_activation_src_10_8: switch_10_8 - is_active_10 <= 0
 link_activation_dst_10_8: switch_10_8 - is_active_8 <= 0
 link_activation_src_10_9: switch_10_9 - is_active_10 <= 0
 link_activation_dst_10_9: switch_10_9 - is_active_9 <= 0
 link_activation_dst_nic_in_11_11: switch_nic_in_11_11 - is_active_11 <= 0
 link_activation_src_11_nic_out_11: switch_11_nic_out_11 - is_active_11
   <= 0
 link_activation_dst_nic_in_12_12: switch_nic_in_12_12 - is_active_12 <= 0
 link_activation_src_12_nic_out_12: switch_12_nic_out_12 - is_active_12
   <= 0
 link_activation_dst_nic_in_12_13: switch_nic_in_12_13 - is_active_13 <= 0
 link_activation_src_13_nic_out_12: switch_13_nic_out_12 - is_active_13
   <= 0
 link_activation_src_12_13: switch_12_13 - is_active_12 <= 0
 link_activation_dst_12_13: switch_12_13 - is_active_13 <= 0
 link_activation_src_13_12: switch_13_12 - is_active_13 <= 0
 link_activation_dst_13_12: switch_13_12 - is_active_12 <= 0
 link_activation_dst_nic_in_14_14: switch_nic_in_14_14 - is_active_14 <= 0
 link_activation_src_14_nic_out_14: switch_14_nic_out_14 - is_active_14
   <= 0
 link_activation_dst_nic_in_15_15: switch_nic_in_15_15 - is_active_15 <= 0
 link_activation_src_15_nic_out_15: switch_15_nic_out_15 - is_active_15
   <= 0
 edge_flow_constr_source_nic_in_0: flow_source_nic_in_0
   - 5e+08 switch_source_nic_in_0 <= 0
 edge_flow_constr_nic_out_0_sink: flow_nic_out_0_sink
   - 5e+08 switch_nic_out_0_sink <= 0
 edge_flow_constr_nic_out_0_nic_in_2: flow_nic_out_0_nic_in_2
   - 61035.15625 switch_nic_out_0_nic_in_2 <= 0
 edge_flow_constr_nic_out_0_nic_in_4: flow_nic_out_0_nic_in_4
   - 61035.15625 switch_nic_out_0_nic_in_4 <= 0
 edge_flow_constr_nic_out_0_nic_in_5: flow_nic_out_0_nic_in_5
   - 61035.15625 switch_nic_out_0_nic_in_5 <= 0
 edge_flow_constr_nic_out_0_nic_in_7: flow_nic_out_0_nic_in_7
   - 61035.15625 switch_nic_out_0_nic_in_7 <= 0
 edge_flow_constr_nic_out_0_nic_in_11: flow_nic_out_0_nic_in_11
   - 61035.15625 switch_nic_out_0_nic_in_11 <= 0
 edge_flow_constr_nic_out_0_nic_in_12: flow_nic_out_0_nic_in_12
   - 61035.15625 switch_nic_out_0_nic_in_12 <= 0
 edge_flow_constr_nic_out_0_nic_in_14: flow_nic_out_0_nic_in_14
   - 61035.15625 switch_nic_out_0_nic_in_14 <= 0
 edge_flow_constr_nic_out_0_nic_in_15: flow_nic_out_0_nic_in_15
   - 61035.15625 switch_nic_out_0_nic_in_15 <= 0
 edge_flow_constr_nic_in_0_0: flow_nic_in_0_0
   - 3.90625e+06 switch_nic_in_0_0 <= 0
 edge_flow_constr_0_nic_out_0: flow_0_nic_out_0
   - 3.90625e+06 switch_0_nic_out_0 <= 0
 edge_flow_constr_nic_in_0_1: flow_nic_in_0_1
   - 3.90625e+06 switch_nic_in_0_1 <= 0
 edge_flow_constr_1_nic_out_0: flow_1_nic_out_0
   - 3.90625e+06 switch_1_nic_out_0 <= 0
 edge_flow_constr_0_1: flow_0_1 - 3.90625e+06 switch_0_1 <= 0
 edge_flow_constr_1_0: flow_1_0 - 3.90625e+06 switch_1_0 <= 0
 edge_flow_constr_source_nic_in_2: flow_source_nic_in_2
   - 5e+08 switch_source_nic_in_2 <= 0
 edge_flow_constr_nic_out_2_sink: flow_nic_out_2_sink
   - 5e+08 switch_nic_out_2_sink <= 0
 edge_flow_constr_nic_out_2_nic_in_0: flow_nic_out_2_nic_in_0
   - 61035.15625 switch_nic_out_2_nic_in_0 <= 0
 edge_flow_constr_nic_out_2_nic_in_4: flow_nic_out_2_nic_in_4
   - 61035.15625 switch_nic_out_2_nic_in_4 <= 0
 edge_flow_constr_nic_out_2_nic_in_5: flow_nic_out_2_nic_in_5
   - 61035.15625 switch_nic_out_2_nic_in_5 <= 0
 edge_flow_constr_nic_out_2_nic_in_7: flow_nic_out_2_nic_in_7
   - 61035.15625 switch_nic_out_2_nic_in_7 <= 0
 edge_flow_constr_nic_out_2_nic_in_11: flow_nic_out_2_nic_in_11
   - 61035.15625 switch_nic_out_2_nic_in_11 <= 0
 edge_flow_constr_nic_out_2_nic_in_12: flow_nic_out_2_nic_in_12
   - 61035.15625 switch_nic_out_2_nic_in_12 <= 0
 edge_flow_constr_nic_out_2_nic_in_14: flow_nic_out_2_nic_in_14
   - 61035.15625 switch_nic_out_2_nic_in_14 <= 0
 edge_flow_constr_nic_out_2_nic_in_15: flow_nic_out_2_nic_in_15
   - 61035.15625 switch_nic_out_2_nic_in_15 <= 0
 edge_flow_constr_nic_in_2_2: flow_nic_in_2_2
   - 3.90625e+06 switch_nic_in_2_2 <= 0
 edge_flow_constr_2_nic_out_2: flow_2_nic_out_2
   - 3.90625e+06 switch_2_nic_out_2 <= 0
 edge_flow_constr_nic_in_2_3: flow_nic_in_2_3
   - 3.90625e+06 switch_nic_in_2_3 <= 0
 edge_flow_constr_3_nic_out_2: flow_3_nic_out_2
   - 3.90625e+06 switch_3_nic_out_2 <= 0
 edge_flow_constr_2_3: flow_2_3 - 3.90625e+06 switch_2_3 <= 0
 edge_flow_constr_3_2: flow_3_2 - 3.90625e+06 switch_3_2 <= 0
 edge_flow_constr_source_nic_in_4: flow_source_nic_in_4
   - 5e+08 switch_source_nic_in_4 <= 0
 edge_flow_constr_nic_out_4_sink: flow_nic_out_4_sink
   - 5e+08 switch_nic_out_4_sink <= 0
 edge_flow_constr_nic_out_4_nic_in_0: flow_nic_out_4_nic_in_0
   - 61035.15625 switch_nic_out_4_nic_in_0 <= 0
 edge_flow_constr_nic_out_4_nic_in_2: flow_nic_out_4_nic_in_2
   - 61035.15625 switch_nic_out_4_nic_in_2 <= 0
 edge_flow_constr_nic_out_4_nic_in_5: flow_nic_out_4_nic_in_5
   - 61035.15625 switch_nic_out_4_nic_in_5 <= 0
 edge_flow_constr_nic_out_4_nic_in_7: flow_nic_out_4_nic_in_7
   - 61035.15625 switch_nic_out_4_nic_in_7 <= 0
 edge_flow_constr_nic_out_4_nic_in_11: flow_nic_out_4_nic_in_11
   - 61035.15625 switch_nic_out_4_nic_in_11 <= 0
 edge_flow_constr_nic_out_4_nic_in_12: flow_nic_out_4_nic_in_12
   - 61035.15625 switch_nic_out_4_nic_in_12 <= 0
 edge_flow_constr_nic_out_4_nic_in_14: flow_nic_out_4_nic_in_14
   - 61035.15625 switch_nic_out_4_nic_in_14 <= 0
 edge_flow_constr_nic_out_4_nic_in_15: flow_nic_out_4_nic_in_15
   - 61035.15625 switch_nic_out_4_nic_in_15 <= 0
 edge_flow_constr_nic_in_4_4: flow_nic_in_4_4
   - 3.90625e+06 switch_nic_in_4_4 <= 0
 edge_flow_constr_4_nic_out_4: flow_4_nic_out_4
   - 3.90625e+06 switch_4_nic_out_4 <= 0
 edge_flow_constr_source_nic_in_5: flow_source_nic_in_5
   - 5e+08 switch_source_nic_in_5 <= 0
 edge_flow_constr_nic_out_5_sink: flow_nic_out_5_sink
   - 5e+08 switch_nic_out_5_sink <= 0
 edge_flow_constr_nic_out_5_nic_in_0: flow_nic_out_5_nic_in_0
   - 61035.15625 switch_nic_out_5_nic_in_0 <= 0
 edge_flow_constr_nic_out_5_nic_in_2: flow_nic_out_5_nic_in_2
   - 61035.15625 switch_nic_out_5_nic_in_2 <= 0
 edge_flow_constr_nic_out_5_nic_in_4: flow_nic_out_5_nic_in_4
   - 61035.15625 switch_nic_out_5_nic_in_4 <= 0
 edge_flow_constr_nic_out_5_nic_in_7: flow_nic_out_5_nic_in_7
   - 61035.15625 switch_nic_out_5_nic_in_7 <= 0
 edge_flow_constr_nic_out_5_nic_in_11: flow_nic_out_5_nic_in_11
   - 61035.15625 switch_nic_out_5_nic_in_11 <= 0
 edge_flow_constr_nic_out_5_nic_in_12: flow_nic_out_5_nic_in_12
   - 61035.15625 switch_nic_out_5_nic_in_12 <= 0
 edge_flow_constr_nic_out_5_nic_in_14: flow_nic_out_5_nic_in_14
   - 61035.15625 switch_nic_out_5_nic_in_14 <= 0
 edge_flow_constr_nic_out_5_nic_in_15: flow_nic_out_5_nic_in_15
   - 61035.15625 switch_nic_out_5_nic_in_15 <= 0
 edge_flow_constr_nic_in_5_5: flow_nic_in_5_5
   - 3.90625e+06 switch_nic_in_5_5 <= 0
 edge_flow_constr_5_nic_out_5: flow_5_nic_out_5
   - 3.90625e+06 switch_5_nic_out_5 <= 0
 edge_flow_constr_nic_in_5_6: flow_nic_in_5_6
   - 3.90625e+06 switch_nic_in_5_6 <= 0
 edge_flow_constr_6_nic_out_5: flow_6_nic_out_5
   - 3.90625e+06 switch_6_nic_out_5 <= 0
 edge_flow_constr_5_6: flow_5_6 - 3.90625e+06 switch_5_6 <= 0
 edge_flow_constr_6_5: flow_6_5 - 3.90625e+06 switch_6_5 <= 0
 edge_flow_constr_source_nic_in_7: flow_source_nic_in_7
   - 5e+08 switch_source_nic_in_7 <= 0
 edge_flow_constr_nic_out_7_sink: flow_nic_out_7_sink
   - 5e+08 switch_nic_out_7_sink <= 0
 edge_flow_constr_nic_out_7_nic_in_0: flow_nic_out_7_nic_in_0
   - 61035.15625 switch_nic_out_7_nic_in_0 <= 0
 edge_flow_constr_nic_out_7_nic_in_2: flow_nic_out_7_nic_in_2
   - 61035.15625 switch_nic_out_7_nic_in_2 <= 0
 edge_flow_constr_nic_out_7_nic_in_4: flow_nic_out_7_nic_in_4
   - 61035.15625 switch_nic_out_7_nic_in_4 <= 0
 edge_flow_constr_nic_out_7_nic_in_5: flow_nic_out_7_nic_in_5
   - 61035.15625 switch_nic_out_7_nic_in_5 <= 0
 edge_flow_constr_nic_out_7_nic_in_11: flow_nic_out_7_nic_in_11
   - 61035.15625 switch_nic_out_7_nic_in_11 <= 0
 edge_flow_constr_nic_out_7_nic_in_12: flow_nic_out_7_nic_in_12
   - 61035.15625 switch_nic_out_7_nic_in_12 <= 0
 edge_flow_constr_nic_out_7_nic_in_14: flow_nic_out_7_nic_in_14
   - 61035.15625 switch_nic_out_7_nic_in_14 <= 0
 edge_flow_constr_nic_out_7_nic_in_15: flow_nic_out_7_nic_in_15
   - 61035.15625 switch_nic_out_7_nic_in_15 <= 0
 edge_flow_constr_nic_in_7_7: flow_nic_in_7_7
   - 3.90625e+06 switch_nic_in_7_7 <= 0
 edge_flow_constr_7_nic_out_7: flow_7_nic_out_7
   - 3.90625e+06 switch_7_nic_out_7 <= 0
 edge_flow_constr_nic_in_7_8: flow_nic_in_7_8
   - 3.90625e+06 switch_nic_in_7_8 <= 0
 edge_flow_constr_8_nic_out_7: flow_8_nic_out_7
   - 3.90625e+06 switch_8_nic_out_7 <= 0
 edge_flow_constr_nic_in_7_9: flow_nic_in_7_9
   - 3.90625e+06 switch_nic_in_7_9 <= 0
 edge_flow_constr_9_nic_out_7: flow_9_nic_out_7
   - 3.90625e+06 switch_9_nic_out_7 <= 0
 edge_flow_constr_nic_in_7_10: flow_nic_in_7_10
   - 3.90625e+06 switch_nic_in_7_10 <= 0
 edge_flow_constr_10_nic_out_7: flow_10_nic_out_7
   - 3.90625e+06 switch_10_nic_out_7 <= 0
 edge_flow_constr_7_8: flow_7_8 - 3.90625e+06 switch_7_8 <= 0
 edge_flow_constr_7_9: flow_7_9 - 3.90625e+06 switch_7_9 <= 0
 edge_flow_constr_7_10: flow_7_10 - 3.90625e+06 switch_7_10 <= 0
 edge_flow_constr_8_7: flow_8_7 - 3.90625e+06 switch_8_7 <= 0
 edge_flow_constr_8_9: flow_8_9 - 3.90625e+06 switch_8_9 <= 0
 edge_flow_constr_8_10: flow_8_10 - 3.90625e+06 switch_8_10 <= 0
 edge_flow_constr_9_7: flow_9_7 - 3.90625e+06 switch_9_7 <= 0
 edge_flow_constr_9_8: flow_9_8 - 3.90625e+06 switch_9_8 <= 0
 edge_flow_constr_9_10: flow_9_10 - 3.90625e+06 switch_9_10 <= 0
 edge_flow_constr_10_7: flow_10_7 - 3.90625e+06 switch_10_7 <= 0
 edge_flow_constr_10_8: flow_10_8 - 3.90625e+06 switch_10_8 <= 0
 edge_flow_constr_10_9: flow_10_9 - 3.90625e+06 switch_10_9 <= 0
 edge_flow_constr_source_nic_in_11: flow_source_nic_in_11
   - 5e+08 switch_source_nic_in_11 <= 0
 edge_flow_constr_nic_out_11_sink: flow_nic_out_11_sink
   - 5e+08 switch_nic_out_11_sink <= 0
 edge_flow_constr_nic_out_11_nic_in_0: flow_nic_out_11_nic_in_0
   - 61035.15625 switch_nic_out_11_nic_in_0 <= 0
 edge_flow_constr_nic_out_11_nic_in_2: flow_nic_out_11_nic_in_2
   - 61035.15625 switch_nic_out_11_nic_in_2 <= 0
 edge_flow_constr_nic_out_11_nic_in_4: flow_nic_out_11_nic_in_4
   - 61035.15625 switch_nic_out_11_nic_in_4 <= 0
 edge_flow_constr_nic_out_11_nic_in_5: flow_nic_out_11_nic_in_5
   - 61035.15625 switch_nic_out_11_nic_in_5 <= 0
 edge_flow_constr_nic_out_11_nic_in_7: flow_nic_out_11_nic_in_7
   - 61035.15625 switch_nic_out_11_nic_in_7 <= 0
 edge_flow_constr_nic_out_11_nic_in_12: flow_nic_out_11_nic_in_12
   - 61035.15625 switch_nic_out_11_nic_in_12 <= 0
 edge_flow_constr_nic_out_11_nic_in_14: flow_nic_out_11_nic_in_14
   - 61035.15625 switch_nic_out_11_nic_in_14 <= 0
 edge_flow_constr_nic_out_11_nic_in_15: flow_nic_out_11_nic_in_15
   - 61035.15625 switch_nic_out_11_nic_in_15 <= 0
 edge_flow_constr_nic_in_11_11: flow_nic_in_11_11
   - 3.90625e+06 switch_nic_in_11_11 <= 0
 edge_flow_constr_11_nic_out_11: flow_11_nic_out_11
   - 3.90625e+06 switch_11_nic_out_11 <= 0
 edge_flow_constr_source_nic_in_12: flow_source_nic_in_12
   - 5e+08 switch_source_nic_in_12 <= 0
 edge_flow_constr_nic_out_12_sink: flow_nic_out_12_sink
   - 5e+08 switch_nic_out_12_sink <= 0
 edge_flow_constr_nic_out_12_nic_in_0: flow_nic_out_12_nic_in_0
   - 61035.15625 switch_nic_out_12_nic_in_0 <= 0
 edge_flow_constr_nic_out_12_nic_in_2: flow_nic_out_12_nic_in_2
   - 61035.15625 switch_nic_out_12_nic_in_2 <= 0
 edge_flow_constr_nic_out_12_nic_in_4: flow_nic_out_12_nic_in_4
   - 61035.15625 switch_nic_out_12_nic_in_4 <= 0
 edge_flow_constr_nic_out_12_nic_in_5: flow_nic_out_12_nic_in_5
   - 61035.15625 switch_nic_out_12_nic_in_5 <= 0
 edge_flow_constr_nic_out_12_nic_in_7: flow_nic_out_12_nic_in_7
   - 61035.15625 switch_nic_out_12_nic_in_7 <= 0
 edge_flow_constr_nic_out_12_nic_in_11: flow_nic_out_12_nic_in_11
   - 61035.15625 switch_nic_out_12_nic_in_11 <= 0
 edge_flow_constr_nic_out_12_nic_in_14: flow_nic_out_12_nic_in_14
   - 61035.15625 switch_nic_out_12_nic_in_14 <= 0
 edge_flow_constr_nic_out_12_nic_in_15: flow_nic_out_12_nic_in_15
   - 61035.15625 switch_nic_out_12_nic_in_15 <= 0
 edge_flow_constr_nic_in_12_12: flow_nic_in_12_12
   - 3.90625e+06 switch_nic_in_12_12 <= 0
 edge_flow_constr_12_nic_out_12: flow_12_nic_out_12
   - 3.90625e+06 switch_12_nic_out_12 <= 0
 edge_flow_constr_nic_in_12_13: flow_nic_in_12_13
   - 3.90625e+06 switch_nic_in_12_13 <= 0
 edge_flow_constr_13_nic_out_12: flow_13_nic_out_12
   - 3.90625e+06 switch_13_nic_out_12 <= 0
 edge_flow_constr_12_13: flow_12_13 - 3.90625e+06 switch_12_13 <= 0
 edge_flow_constr_13_12: flow_13_12 - 3.90625e+06 switch_13_12 <= 0
 edge_flow_constr_source_nic_in_14: flow_source_nic_in_14
   - 5e+08 switch_source_nic_in_14 <= 0
 edge_flow_constr_nic_out_14_sink: flow_nic_out_14_sink
   - 5e+08 switch_nic_out_14_sink <= 0
 edge_flow_constr_nic_out_14_nic_in_0: flow_nic_out_14_nic_in_0
   - 61035.15625 switch_nic_out_14_nic_in_0 <= 0
 edge_flow_constr_nic_out_14_nic_in_2: flow_nic_out_14_nic_in_2
   - 61035.15625 switch_nic_out_14_nic_in_2 <= 0
 edge_flow_constr_nic_out_14_nic_in_4: flow_nic_out_14_nic_in_4
   - 61035.15625 switch_nic_out_14_nic_in_4 <= 0
 edge_flow_constr_nic_out_14_nic_in_5: flow_nic_out_14_nic_in_5
   - 61035.15625 switch_nic_out_14_nic_in_5 <= 0
 edge_flow_constr_nic_out_14_nic_in_7: flow_nic_out_14_nic_in_7
   - 61035.15625 switch_nic_out_14_nic_in_7 <= 0
 edge_flow_constr_nic_out_14_nic_in_11: flow_nic_out_14_nic_in_11
   - 61035.15625 switch_nic_out_14_nic_in_11 <= 0
 edge_flow_constr_nic_out_14_nic_in_12: flow_nic_out_14_nic_in_12
   - 61035.15625 switch_nic_out_14_nic_in_12 <= 0
 edge_flow_constr_nic_out_14_nic_in_15: flow_nic_out_14_nic_in_15
   - 61035.15625 switch_nic_out_14_nic_in_15 <= 0
 edge_flow_constr_nic_in_14_14: flow_nic_in_14_14
   - 3.90625e+06 switch_nic_in_14_14 <= 0
 edge_flow_constr_14_nic_out_14: flow_14_nic_out_14
   - 3.90625e+06 switch_14_nic_out_14 <= 0
 edge_flow_constr_source_nic_in_15: flow_source_nic_in_15
   - 5e+08 switch_source_nic_in_15 <= 0
 edge_flow_constr_nic_out_15_sink: flow_nic_out_15_sink
   - 5e+08 switch_nic_out_15_sink <= 0
 edge_flow_constr_nic_out_15_nic_in_0: flow_nic_out_15_nic_in_0
   - 61035.15625 switch_nic_out_15_nic_in_0 <= 0
 edge_flow_constr_nic_out_15_nic_in_2: flow_nic_out_15_nic_in_2
   - 61035.15625 switch_nic_out_15_nic_in_2 <= 0
 edge_flow_constr_nic_out_15_nic_in_4: flow_nic_out_15_nic_in_4
   - 61035.15625 switch_nic_out_15_nic_in_4 <= 0
 edge_flow_constr_nic_out_15_nic_in_5: flow_nic_out_15_nic_in_5
   - 61035.15625 switch_nic_out_15_nic_in_5 <= 0
 edge_flow_constr_nic_out_15_nic_in_7: flow_nic_out_15_nic_in_7
   - 61035.15625 switch_nic_out_15_nic_in_7 <= 0
 edge_flow_constr_nic_out_15_nic_in_11: flow_nic_out_15_nic_in_11
   - 61035.15625 switch_nic_out_15_nic_in_11 <= 0
 edge_flow_constr_nic_out_15_nic_in_12: flow_nic_out_15_nic_in_12
   - 61035.15625 switch_nic_out_15_nic_in_12 <= 0
 edge_flow_constr_nic_out_15_nic_in_14: flow_nic_out_15_nic_in_14
   - 61035.15625 switch_nic_out_15_nic_in_14 <= 0
 edge_flow_constr_nic_in_15_15: flow_nic_in_15_15
   - 3.90625e+06 switch_nic_in_15_15 <= 0
 edge_flow_constr_15_nic_out_15: flow_15_nic_out_15
   - 3.90625e+06 switch_15_nic_out_15 <= 0
 force_group_off_constr_4: group_tp_active_4 = 0
 force_group_off_constr_11: group_tp_active_11 = 0
 force_group_off_constr_14: group_tp_active_14 = 0
 force_group_off_constr_15: group_tp_active_15 = 0
 R570: - group_tp_active_0 + tp_on_and_switch_source_nic_in_0_0 <= 0
 R571: - switch_source_nic_in_0 + tp_on_and_switch_source_nic_in_0_0 <= 0
 R572: - switch_source_nic_in_0 - group_tp_active_0
   + tp_on_and_switch_source_nic_in_0_0 >= -1
 R573: - group_tp_active_2 + tp_on_and_switch_nic_out_0_nic_in_2_2 <= 0
 R574: - switch_nic_out_0_nic_in_2 + tp_on_and_switch_nic_out_0_nic_in_2_2
   <= 0
 R575: - switch_nic_out_0_nic_in_2 - group_tp_active_2
   + tp_on_and_switch_nic_out_0_nic_in_2_2 >= -1
 R576: - group_tp_active_5 + tp_on_and_switch_nic_out_0_nic_in_5_5 <= 0
 R577: - switch_nic_out_0_nic_in_5 + tp_on_and_switch_nic_out_0_nic_in_5_5
   <= 0
 R578: - switch_nic_out_0_nic_in_5 - group_tp_active_5
   + tp_on_and_switch_nic_out_0_nic_in_5_5 >= -1
 R579: - group_tp_active_7 + tp_on_and_switch_nic_out_0_nic_in_7_7 <= 0
 R580: - switch_nic_out_0_nic_in_7 + tp_on_and_switch_nic_out_0_nic_in_7_7
   <= 0
 R581: - switch_nic_out_0_nic_in_7 - group_tp_active_7
   + tp_on_and_switch_nic_out_0_nic_in_7_7 >= -1
 R582: - group_tp_active_12 + tp_on_and_switch_nic_out_0_nic_in_12_12 <= 0
 R583: - switch_nic_out_0_nic_in_12
   + tp_on_and_switch_nic_out_0_nic_in_12_12 <= 0
 R584: - switch_nic_out_0_nic_in_12 - group_tp_active_12
   + tp_on_and_switch_nic_out_0_nic_in_12_12 >= -1
 R585: - group_tp_active_2 + tp_on_and_switch_source_nic_in_2_2 <= 0
 R586: - switch_source_nic_in_2 + tp_on_and_switch_source_nic_in_2_2 <= 0
 R587: - switch_source_nic_in_2 - group_tp_active_2
   + tp_on_and_switch_source_nic_in_2_2 >= -1
 R588: - group_tp_active_0 + tp_on_and_switch_nic_out_2_nic_in_0_0 <= 0
 R589: - switch_nic_out_2_nic_in_0 + tp_on_and_switch_nic_out_2_nic_in_0_0
   <= 0
 R590: - switch_nic_out_2_nic_in_0 - group_tp_active_0
   + tp_on_and_switch_nic_out_2_nic_in_0_0 >= -1
 R591: - group_tp_active_5 + tp_on_and_switch_nic_out_2_nic_in_5_5 <= 0
 R592: - switch_nic_out_2_nic_in_5 + tp_on_and_switch_nic_out_2_nic_in_5_5
   <= 0
 R593: - switch_nic_out_2_nic_in_5 - group_tp_active_5
   + tp_on_and_switch_nic_out_2_nic_in_5_5 >= -1
 R594: - group_tp_active_7 + tp_on_and_switch_nic_out_2_nic_in_7_7 <= 0
 R595: - switch_nic_out_2_nic_in_7 + tp_on_and_switch_nic_out_2_nic_in_7_7
   <= 0
 R596: - switch_nic_out_2_nic_in_7 - group_tp_active_7
   + tp_on_and_switch_nic_out_2_nic_in_7_7 >= -1
 R597: - group_tp_active_12 + tp_on_and_switch_nic_out_2_nic_in_12_12 <= 0
 R598: - switch_nic_out_2_nic_in_12
   + tp_on_and_switch_nic_out_2_nic_in_12_12 <= 0
 R599: - switch_nic_out_2_nic_in_12 - group_tp_active_12
   + tp_on_and_switch_nic_out_2_nic_in_12_12 >= -1
 R600: - group_tp_active_0 + tp_on_and_switch_nic_out_4_nic_in_0_0 <= 0
 R601: - switch_nic_out_4_nic_in_0 + tp_on_and_switch_nic_out_4_nic_in_0_0
   <= 0
 R602: - switch_nic_out_4_nic_in_0 - group_tp_active_0
   + tp_on_and_switch_nic_out_4_nic_in_0_0 >= -1
 R603: - group_tp_active_2 + tp_on_and_switch_nic_out_4_nic_in_2_2 <= 0
 R604: - switch_nic_out_4_nic_in_2 + tp_on_and_switch_nic_out_4_nic_in_2_2
   <= 0
 R605: - switch_nic_out_4_nic_in_2 - group_tp_active_2
   + tp_on_and_switch_nic_out_4_nic_in_2_2 >= -1
 R606: - group_tp_active_5 + tp_on_and_switch_nic_out_4_nic_in_5_5 <= 0
 R607: - switch_nic_out_4_nic_in_5 + tp_on_and_switch_nic_out_4_nic_in_5_5
   <= 0
 R608: - switch_nic_out_4_nic_in_5 - group_tp_active_5
   + tp_on_and_switch_nic_out_4_nic_in_5_5 >= -1
 R609: - group_tp_active_7 + tp_on_and_switch_nic_out_4_nic_in_7_7 <= 0
 R610: - switch_nic_out_4_nic_in_7 + tp_on_and_switch_nic_out_4_nic_in_7_7
   <= 0
 R611: - switch_nic_out_4_nic_in_7 - group_tp_active_7
   + tp_on_and_switch_nic_out_4_nic_in_7_7 >= -1
 R612: - group_tp_active_12 + tp_on_and_switch_nic_out_4_nic_in_12_12 <= 0
 R613: - switch_nic_out_4_nic_in_12
   + tp_on_and_switch_nic_out_4_nic_in_12_12 <= 0
 R614: - switch_nic_out_4_nic_in_12 - group_tp_active_12
   + tp_on_and_switch_nic_out_4_nic_in_12_12 >= -1
 R615: - group_tp_active_5 + tp_on_and_switch_source_nic_in_5_5 <= 0
 R616: - switch_source_nic_in_5 + tp_on_and_switch_source_nic_in_5_5 <= 0
 R617: - switch_source_nic_in_5 - group_tp_active_5
   + tp_on_and_switch_source_nic_in_5_5 >= -1
 R618: - group_tp_active_0 + tp_on_and_switch_nic_out_5_nic_in_0_0 <= 0
 R619: - switch_nic_out_5_nic_in_0 + tp_on_and_switch_nic_out_5_nic_in_0_0
   <= 0
 R620: - switch_nic_out_5_nic_in_0 - group_tp_active_0
   + tp_on_and_switch_nic_out_5_nic_in_0_0 >= -1
 R621: - group_tp_active_2 + tp_on_and_switch_nic_out_5_nic_in_2_2 <= 0
 R622: - switch_nic_out_5_nic_in_2 + tp_on_and_switch_nic_out_5_nic_in_2_2
   <= 0
 R623: - switch_nic_out_5_nic_in_2 - group_tp_active_2
   + tp_on_and_switch_nic_out_5_nic_in_2_2 >= -1
 R624: - group_tp_active_7 + tp_on_and_switch_nic_out_5_nic_in_7_7 <= 0
 R625: - switch_nic_out_5_nic_in_7 + tp_on_and_switch_nic_out_5_nic_in_7_7
   <= 0
 R626: - switch_nic_out_5_nic_in_7 - group_tp_active_7
   + tp_on_and_switch_nic_out_5_nic_in_7_7 >= -1
 R627: - group_tp_active_12 + tp_on_and_switch_nic_out_5_nic_in_12_12 <= 0
 R628: - switch_nic_out_5_nic_in_12
   + tp_on_and_switch_nic_out_5_nic_in_12_12 <= 0
 R629: - switch_nic_out_5_nic_in_12 - group_tp_active_12
   + tp_on_and_switch_nic_out_5_nic_in_12_12 >= -1
 R630: - group_tp_active_7 + tp_on_and_switch_source_nic_in_7_7 <= 0
 R631: - switch_source_nic_in_7 + tp_on_and_switch_source_nic_in_7_7 <= 0
 R632: - switch_source_nic_in_7 - group_tp_active_7
   + tp_on_and_switch_source_nic_in_7_7 >= -1
 R633: - group_tp_active_0 + tp_on_and_switch_nic_out_7_nic_in_0_0 <= 0
 R634: - switch_nic_out_7_nic_in_0 + tp_on_and_switch_nic_out_7_nic_in_0_0
   <= 0
 R635: - switch_nic_out_7_nic_in_0 - group_tp_active_0
   + tp_on_and_switch_nic_out_7_nic_in_0_0 >= -1
 R636: - group_tp_active_2 + tp_on_and_switch_nic_out_7_nic_in_2_2 <= 0
 R637: - switch_nic_out_7_nic_in_2 + tp_on_and_switch_nic_out_7_nic_in_2_2
   <= 0
 R638: - switch_nic_out_7_nic_in_2 - group_tp_active_2
   + tp_on_and_switch_nic_out_7_nic_in_2_2 >= -1
 R639: - group_tp_active_5 + tp_on_and_switch_nic_out_7_nic_in_5_5 <= 0
 R640: - switch_nic_out_7_nic_in_5 + tp_on_and_switch_nic_out_7_nic_in_5_5
   <= 0
 R641: - switch_nic_out_7_nic_in_5 - group_tp_active_5
   + tp_on_and_switch_nic_out_7_nic_in_5_5 >= -1
 R642: - group_tp_active_12 + tp_on_and_switch_nic_out_7_nic_in_12_12 <= 0
 R643: - switch_nic_out_7_nic_in_12
   + tp_on_and_switch_nic_out_7_nic_in_12_12 <= 0
 R644: - switch_nic_out_7_nic_in_12 - group_tp_active_12
   + tp_on_and_switch_nic_out_7_nic_in_12_12 >= -1
 R645: - group_tp_active_0 + tp_on_and_switch_nic_out_11_nic_in_0_0 <= 0
 R646: - switch_nic_out_11_nic_in_0
   + tp_on_and_switch_nic_out_11_nic_in_0_0 <= 0
 R647: - switch_nic_out_11_nic_in_0 - group_tp_active_0
   + tp_on_and_switch_nic_out_11_nic_in_0_0 >= -1
 R648: - group_tp_active_2 + tp_on_and_switch_nic_out_11_nic_in_2_2 <= 0
 R649: - switch_nic_out_11_nic_in_2
   + tp_on_and_switch_nic_out_11_nic_in_2_2 <= 0
 R650: - switch_nic_out_11_nic_in_2 - group_tp_active_2
   + tp_on_and_switch_nic_out_11_nic_in_2_2 >= -1
 R651: - group_tp_active_5 + tp_on_and_switch_nic_out_11_nic_in_5_5 <= 0
 R652: - switch_nic_out_11_nic_in_5
   + tp_on_and_switch_nic_out_11_nic_in_5_5 <= 0
 R653: - switch_nic_out_11_nic_in_5 - group_tp_active_5
   + tp_on_and_switch_nic_out_11_nic_in_5_5 >= -1
 R654: - group_tp_active_7 + tp_on_and_switch_nic_out_11_nic_in_7_7 <= 0
 R655: - switch_nic_out_11_nic_in_7
   + tp_on_and_switch_nic_out_11_nic_in_7_7 <= 0
 R656: - switch_nic_out_11_nic_in_7 - group_tp_active_7
   + tp_on_and_switch_nic_out_11_nic_in_7_7 >= -1
 R657: - group_tp_active_12 + tp_on_and_switch_nic_out_11_nic_in_12_12
   <= 0
 R658: - switch_nic_out_11_nic_in_12
   + tp_on_and_switch_nic_out_11_nic_in_12_12 <= 0
 R659: - switch_nic_out_11_nic_in_12 - group_tp_active_12
   + tp_on_and_switch_nic_out_11_nic_in_12_12 >= -1
 R660: - group_tp_active_12 + tp_on_and_switch_source_nic_in_12_12 <= 0
 R661: - switch_source_nic_in_12 + tp_on_and_switch_source_nic_in_12_12
   <= 0
 R662: - switch_source_nic_in_12 - group_tp_active_12
   + tp_on_and_switch_source_nic_in_12_12 >= -1
 R663: - group_tp_active_0 + tp_on_and_switch_nic_out_12_nic_in_0_0 <= 0
 R664: - switch_nic_out_12_nic_in_0
   + tp_on_and_switch_nic_out_12_nic_in_0_0 <= 0
 R665: - switch_nic_out_12_nic_in_0 - group_tp_active_0
   + tp_on_and_switch_nic_out_12_nic_in_0_0 >= -1
 R666: - group_tp_active_2 + tp_on_and_switch_nic_out_12_nic_in_2_2 <= 0
 R667: - switch_nic_out_12_nic_in_2
   + tp_on_and_switch_nic_out_12_nic_in_2_2 <= 0
 R668: - switch_nic_out_12_nic_in_2 - group_tp_active_2
   + tp_on_and_switch_nic_out_12_nic_in_2_2 >= -1
 R669: - group_tp_active_5 + tp_on_and_switch_nic_out_12_nic_in_5_5 <= 0
 R670: - switch_nic_out_12_nic_in_5
   + tp_on_and_switch_nic_out_12_nic_in_5_5 <= 0
 R671: - switch_nic_out_12_nic_in_5 - group_tp_active_5
   + tp_on_and_switch_nic_out_12_nic_in_5_5 >= -1
 R672: - group_tp_active_7 + tp_on_and_switch_nic_out_12_nic_in_7_7 <= 0
 R673: - switch_nic_out_12_nic_in_7
   + tp_on_and_switch_nic_out_12_nic_in_7_7 <= 0
 R674: - switch_nic_out_12_nic_in_7 - group_tp_active_7
   + tp_on_and_switch_nic_out_12_nic_in_7_7 >= -1
 R675: - group_tp_active_0 + tp_on_and_switch_nic_out_14_nic_in_0_0 <= 0
 R676: - switch_nic_out_14_nic_in_0
   + tp_on_and_switch_nic_out_14_nic_in_0_0 <= 0
 R677: - switch_nic_out_14_nic_in_0 - group_tp_active_0
   + tp_on_and_switch_nic_out_14_nic_in_0_0 >= -1
 R678: - group_tp_active_2 + tp_on_and_switch_nic_out_14_nic_in_2_2 <= 0
 R679: - switch_nic_out_14_nic_in_2
   + tp_on_and_switch_nic_out_14_nic_in_2_2 <= 0
 R680: - switch_nic_out_14_nic_in_2 - group_tp_active_2
   + tp_on_and_switch_nic_out_14_nic_in_2_2 >= -1
 R681: - group_tp_active_5 + tp_on_and_switch_nic_out_14_nic_in_5_5 <= 0
 R682: - switch_nic_out_14_nic_in_5
   + tp_on_and_switch_nic_out_14_nic_in_5_5 <= 0
 R683: - switch_nic_out_14_nic_in_5 - group_tp_active_5
   + tp_on_and_switch_nic_out_14_nic_in_5_5 >= -1
 R684: - group_tp_active_7 + tp_on_and_switch_nic_out_14_nic_in_7_7 <= 0
 R685: - switch_nic_out_14_nic_in_7
   + tp_on_and_switch_nic_out_14_nic_in_7_7 <= 0
 R686: - switch_nic_out_14_nic_in_7 - group_tp_active_7
   + tp_on_and_switch_nic_out_14_nic_in_7_7 >= -1
 R687: - group_tp_active_12 + tp_on_and_switch_nic_out_14_nic_in_12_12
   <= 0
 R688: - switch_nic_out_14_nic_in_12
   + tp_on_and_switch_nic_out_14_nic_in_12_12 <= 0
 R689: - switch_nic_out_14_nic_in_12 - group_tp_active_12
   + tp_on_and_switch_nic_out_14_nic_in_12_12 >= -1
 R690: - group_tp_active_0 + tp_on_and_switch_nic_out_15_nic_in_0_0 <= 0
 R691: - switch_nic_out_15_nic_in_0
   + tp_on_and_switch_nic_out_15_nic_in_0_0 <= 0
 R692: - switch_nic_out_15_nic_in_0 - group_tp_active_0
   + tp_on_and_switch_nic_out_15_nic_in_0_0 >= -1
 R693: - group_tp_active_2 + tp_on_and_switch_nic_out_15_nic_in_2_2 <= 0
 R694: - switch_nic_out_15_nic_in_2
   + tp_on_and_switch_nic_out_15_nic_in_2_2 <= 0
 R695: - switch_nic_out_15_nic_in_2 - group_tp_active_2
   + tp_on_and_switch_nic_out_15_nic_in_2_2 >= -1
 R696: - group_tp_active_5 + tp_on_and_switch_nic_out_15_nic_in_5_5 <= 0
 R697: - switch_nic_out_15_nic_in_5
   + tp_on_and_switch_nic_out_15_nic_in_5_5 <= 0
 R698: - switch_nic_out_15_nic_in_5 - group_tp_active_5
   + tp_on_and_switch_nic_out_15_nic_in_5_5 >= -1
 R699: - group_tp_active_7 + tp_on_and_switch_nic_out_15_nic_in_7_7 <= 0
 R700: - switch_nic_out_15_nic_in_7
   + tp_on_and_switch_nic_out_15_nic_in_7_7 <= 0
 R701: - switch_nic_out_15_nic_in_7 - group_tp_active_7
   + tp_on_and_switch_nic_out_15_nic_in_7_7 >= -1
 R702: - group_tp_active_12 + tp_on_and_switch_nic_out_15_nic_in_12_12
   <= 0
 R703: - switch_nic_out_15_nic_in_12
   + tp_on_and_switch_nic_out_15_nic_in_12_12 <= 0
 R704: - switch_nic_out_15_nic_in_12 - group_tp_active_12
   + tp_on_and_switch_nic_out_15_nic_in_12_12 >= -1
 force_min_throughput: flow_source_nic_in_0 + flow_source_nic_in_2
   + flow_source_nic_in_4 + flow_source_nic_in_5 + flow_source_nic_in_7
   + flow_source_nic_in_11 + flow_source_nic_in_12 + flow_source_nic_in_14
   + flow_source_nic_in_15 >= 1
 inactive_start_invalid_0: is_active_0 = 0 -> start_0 = 80
 inactive_start_invalid_1: is_active_1 = 0 -> start_1 = 80
 inactive_start_invalid_2: is_active_2 = 0 -> start_2 = 80
 inactive_start_invalid_3: is_active_3 = 0 -> start_3 = 80
 inactive_start_invalid_4: is_active_4 = 0 -> start_4 = 80
 inactive_start_invalid_5: is_active_5 = 0 -> start_5 = 80
 inactive_start_invalid_6: is_active_6 = 0 -> start_6 = 80
 inactive_start_invalid_7: is_active_7 = 0 -> start_7 = 80
 inactive_start_invalid_8: is_active_8 = 0 -> start_8 = 80
 inactive_start_invalid_9: is_active_9 = 0 -> start_9 = 80
 inactive_start_invalid_10: is_active_10 = 0 -> start_10 = 80
 inactive_start_invalid_11: is_active_11 = 0 -> start_11 = 80
 inactive_start_invalid_12: is_active_12 = 0 -> start_12 = 80
 inactive_start_invalid_13: is_active_13 = 0 -> start_13 = 80
 inactive_start_invalid_14: is_active_14 = 0 -> start_14 = 80
 inactive_start_invalid_15: is_active_15 = 0 -> start_15 = 80
 node_throughput_constraint_0: group_tp_active_0 = 0 ->
   - 3949.714285714286 hold_0_1 - 1974.857142857143 hold_0_2
   - 1316.571428571428 hold_0_3 - 987.4285714285714 hold_0_4
   - 789.9428571428572 hold_0_5 - 658.2857142857142 hold_0_6
   - 564.2448979591836 hold_0_7 - 493.7142857142857 hold_0_8
   - 438.8571428571428 hold_0_9 - 394.9714285714286 hold_0_10
   - 359.0649350649351 hold_0_11 - 329.1428571428571 hold_0_12
   - 303.8241758241758 hold_0_13 + flow_nic_in_0_0 + flow_1_0 <= 0
 node_throughput_constraint_1: group_tp_active_0 = 0 ->
   - 3949.714285714286 hold_1_1 - 1974.857142857143 hold_1_2
   - 1316.571428571428 hold_1_3 - 987.4285714285714 hold_1_4
   - 789.9428571428572 hold_1_5 - 658.2857142857142 hold_1_6
   - 564.2448979591836 hold_1_7 - 493.7142857142857 hold_1_8
   - 438.8571428571428 hold_1_9 - 394.9714285714286 hold_1_10
   - 359.0649350649351 hold_1_11 - 329.1428571428571 hold_1_12
   - 303.8241758241758 hold_1_13 + flow_nic_in_0_1 + flow_0_1 <= 0
 node_throughput_constraint_2: group_tp_active_2 = 0 ->
   - 3949.714285714286 hold_2_1 - 1974.857142857143 hold_2_2
   - 1316.571428571428 hold_2_3 - 987.4285714285714 hold_2_4
   - 789.9428571428572 hold_2_5 - 658.2857142857142 hold_2_6
   - 564.2448979591836 hold_2_7 - 493.7142857142857 hold_2_8
   - 438.8571428571428 hold_2_9 - 394.9714285714286 hold_2_10
   - 359.0649350649351 hold_2_11 - 329.1428571428571 hold_2_12
   - 303.8241758241758 hold_2_13 + flow_nic_in_2_2 + flow_3_2 <= 0
 node_throughput_constraint_3: group_tp_active_2 = 0 ->
   - 3949.714285714286 hold_3_1 - 1974.857142857143 hold_3_2
   - 1316.571428571428 hold_3_3 - 987.4285714285714 hold_3_4
   - 789.9428571428572 hold_3_5 - 658.2857142857142 hold_3_6
   - 564.2448979591836 hold_3_7 - 493.7142857142857 hold_3_8
   - 438.8571428571428 hold_3_9 - 394.9714285714286 hold_3_10
   - 359.0649350649351 hold_3_11 - 329.1428571428571 hold_3_12
   - 303.8241758241758 hold_3_13 + flow_nic_in_2_3 + flow_2_3 <= 0
 node_throughput_constraint_4: group_tp_active_4 = 0 ->
   - 1371.428571428571 hold_4_1 - 685.7142857142857 hold_4_2
   - 457.1428571428571 hold_4_3 - 342.8571428571428 hold_4_4
   - 274.2857142857143 hold_4_5 - 228.5714285714286 hold_4_6
   + flow_nic_in_4_4 <= 0
 node_throughput_constraint_5: group_tp_active_5 = 0 ->
   - 1371.428571428571 hold_5_1 - 685.7142857142857 hold_5_2
   - 457.1428571428571 hold_5_3 - 342.8571428571428 hold_5_4
   - 274.2857142857143 hold_5_5 - 228.5714285714286 hold_5_6
   + flow_nic_in_5_5 + flow_6_5 <= 0
 node_throughput_constraint_6: group_tp_active_5 = 0 ->
   - 1371.428571428571 hold_6_1 - 685.7142857142857 hold_6_2
   - 457.1428571428571 hold_6_3 - 342.8571428571428 hold_6_4
   - 274.2857142857143 hold_6_5 - 228.5714285714286 hold_6_6
   + flow_nic_in_5_6 + flow_5_6 <= 0
 node_throughput_constraint_7: group_tp_active_7 = 0 ->
   - 1371.428571428571 hold_7_1 - 685.7142857142857 hold_7_2
   - 457.1428571428571 hold_7_3 - 342.8571428571428 hold_7_4
   - 274.2857142857143 hold_7_5 - 228.5714285714286 hold_7_6
   + flow_nic_in_7_7 + flow_8_7 + flow_9_7 + flow_10_7 <= 0
 node_throughput_constraint_8: group_tp_active_7 = 0 ->
   - 1371.428571428571 hold_8_1 - 685.7142857142857 hold_8_2
   - 457.1428571428571 hold_8_3 - 342.8571428571428 hold_8_4
   - 274.2857142857143 hold_8_5 - 228.5714285714286 hold_8_6
   + flow_nic_in_7_8 + flow_7_8 + flow_9_8 + flow_10_8 <= 0
 node_throughput_constraint_9: group_tp_active_7 = 0 ->
   - 1371.428571428571 hold_9_1 - 685.7142857142857 hold_9_2
   - 457.1428571428571 hold_9_3 - 342.8571428571428 hold_9_4
   - 274.2857142857143 hold_9_5 - 228.5714285714286 hold_9_6
   + flow_nic_in_7_9 + flow_7_9 + flow_8_9 + flow_10_9 <= 0
 node_throughput_constraint_10: group_tp_active_7 = 0 ->
   - 1371.428571428571 hold_10_1 - 685.7142857142857 hold_10_2
   - 457.1428571428571 hold_10_3 - 342.8571428571428 hold_10_4
   - 274.2857142857143 hold_10_5 - 228.5714285714286 hold_10_6
   + flow_nic_in_7_10 + flow_7_10 + flow_8_10 + flow_9_10 <= 0
 node_throughput_constraint_11: group_tp_active_11 = 0 ->
   - 9325.714285714286 hold_11_1 - 4662.857142857143 hold_11_2
   - 3108.571428571428 hold_11_3 - 2331.428571428572 hold_11_4
   - 1865.142857142857 hold_11_5 - 1554.285714285714 hold_11_6
   - 1332.244897959184 hold_11_7 - 1165.714285714286 hold_11_8
   - 1036.190476190476 hold_11_9 - 932.5714285714284 hold_11_10
   - 847.7922077922078 hold_11_11 - 777.1428571428571 hold_11_12
   - 717.3626373626373 hold_11_13 - 666.1224489795918 hold_11_14
   - 621.7142857142858 hold_11_15 - 582.8571428571429 hold_11_16
   - 548.5714285714286 hold_11_17 - 518.0952380952381 hold_11_18
   - 490.8270676691729 hold_11_19 - 466.2857142857142 hold_11_20
   - 444.0816326530613 hold_11_21 - 423.8961038961039 hold_11_22
   + flow_nic_in_11_11 <= 0
 node_throughput_constraint_12: group_tp_active_12 = 0 -> - 2048 hold_12_1
   - 1024 hold_12_2 - 682.6666666666666 hold_12_3 - 512 hold_12_4
   + flow_nic_in_12_12 + flow_13_12 <= 0
 node_throughput_constraint_13: group_tp_active_12 = 0 -> - 2048 hold_13_1
   - 1024 hold_13_2 - 682.6666666666666 hold_13_3 - 512 hold_13_4
   + flow_nic_in_12_13 + flow_12_13 <= 0
 node_throughput_constraint_14: group_tp_active_14 = 0 ->
   - 7108.571428571429 hold_14_1 - 3554.285714285715 hold_14_2
   - 2369.52380952381 hold_14_3 - 1777.142857142857 hold_14_4
   - 1421.714285714286 hold_14_5 - 1184.761904761905 hold_14_6
   - 1015.510204081633 hold_14_7 - 888.5714285714287 hold_14_8
   - 789.8412698412698 hold_14_9 - 710.8571428571429 hold_14_10
   - 646.2337662337662 hold_14_11 + flow_nic_in_14_14 <= 0
 node_throughput_constraint_15: group_tp_active_15 = 0 ->
   - 4265.142857142857 hold_15_1 - 2132.571428571428 hold_15_2
   - 1421.714285714286 hold_15_3 - 1066.285714285714 hold_15_4
   - 853.0285714285714 hold_15_5 - 710.8571428571429 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_0_source_nic_in_0: switch_source_nic_in_0 = 1 ->
   - 3763.885863646265 hold_0_1 - 1927.280838724069 hold_0_2
   - 1295.25522824376 hold_0_3 - 975.3894784375707 hold_0_4
   - 782.2190032139786 hold_0_5 - 652.9131716164135 hold_0_6
   - 560.2931163062157 hold_0_7 - 490.6860517946388 hold_0_8
   - 436.4628324619231 hold_0_9 - 393.0309784833735 hold_0_10
   - 357.4605407418134 hold_0_11 - 327.7942180742089 hold_0_12
   - 302.6746772696222 hold_0_13 + flow_nic_in_0_0 + flow_1_0 <= 0
 constr_intergroup_throughput_2_nic_out_0_nic_in_2: switch_nic_out_0_nic_in_2 = 1 ->
   - 3167.736021998167 hold_2_1 - 1757.884028484232 hold_2_2
   - 1216.473072861668 hold_2_3 - 930.032292787944 hold_2_4
   - 752.7771727292528 hold_2_5 - 632.2722283205268 hold_2_6
   - 545.0244440939914 hold_2_7 - 478.9356984478936 hold_2_8
   - 427.1412680756396 hold_2_9 - 385.4561677448137 hold_2_10
   - 351.1838227822376 hold_2_11 - 322.5083986562149 hold_2_12
   - 298.1623673539816 hold_2_13 + flow_nic_in_2_2 + flow_3_2 <= 0
 constr_intergroup_throughput_4_nic_out_0_nic_in_4: switch_nic_out_0_nic_in_4 = 1 ->
   - 1263.157894736842 hold_4_1 - 657.5342465753424 hold_4_2
   - 444.4444444444444 hold_4_3 - 335.6643356643356 hold_4_4
   - 269.6629213483146 hold_4_5 - 225.3521126760563 hold_4_6
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_0_nic_in_5: switch_nic_out_0_nic_in_5 = 1 ->
   - 738.4615384615385 hold_5_1 - 480 hold_5_2 - 355.5555555555555 hold_5_3
   - 282.3529411764706 hold_5_4 - 234.1463414634146 hold_5_5 - 200 hold_5_6
   + flow_nic_in_5_5 + flow_6_5 <= 0
 constr_intergroup_throughput_10_nic_out_0_nic_in_7: switch_nic_out_0_nic_in_7 = 1 ->
   - 1263.157894736842 hold_10_1 - 657.5342465753424 hold_10_2
   - 444.4444444444444 hold_10_3 - 335.6643356643356 hold_10_4
   - 269.6629213483146 hold_10_5 - 225.3521126760563 hold_10_6
   + flow_nic_in_7_10 + flow_7_10 + flow_8_10 + flow_9_10 <= 0
 constr_intergroup_throughput_11_nic_out_0_nic_in_11: switch_nic_out_0_nic_in_11 = 1 ->
   - 1365.690376569038 hold_11_1 - 1191.240875912409 hold_11_2
   - 1056.31067961165 hold_11_3 - 948.8372093023256 hold_11_4
   - 861.2137203166227 hold_11_5 - 788.4057971014493 hold_11_6
   - 726.9487750556792 hold_11_7 - 674.3801652892562 hold_11_8
   - 628.9017341040462 hold_11_9 - 589.1696750902527 hold_11_10
   - 554.1595925297114 hold_11_11 - 523.0769230769231 hold_11_12
   - 495.2959028831563 hold_11_13 - 470.3170028818444 hold_11_14
   - 447.7366255144033 hold_11_15 - 427.2251308900524 hold_11_16
   - 408.5106382978723 hold_11_17 - 391.3669064748201 hold_11_18
   - 375.6041426927503 hold_11_19 - 361.0619469026548 hold_11_20
   - 347.6038338658147 hold_11_21 - 335.1129363449692 hold_11_22
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_0_nic_in_12: switch_nic_out_0_nic_in_12 = 1 ->
   - 898.2456140350877 hold_12_1 - 624.390243902439 hold_12_2
   - 478.5046728971963 hold_12_3 - 387.8787878787879 hold_12_4
   + flow_nic_in_12_12 + flow_13_12 <= 0
 constr_intergroup_throughput_14_nic_out_0_nic_in_14: switch_nic_out_0_nic_in_14 = 1 ->
   - 4921.859545004946 hold_14_1 - 2908.240794856809 hold_14_2
   - 2063.873911240149 hold_14_3 - 1599.485695917711 hold_14_4
   - 1305.694043558121 hold_14_5 - 1103.081356683662 hold_14_6
   - 954.9030896181155 hold_14_7 - 841.8203349687025 hold_14_8
   - 752.6849190742701 hold_14_9 - 680.6182464779099 hold_14_10
   - 621.1459243540132 hold_14_11 + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_0_nic_in_15: switch_nic_out_0_nic_in_15 = 1 ->
   - 1163.522992985191 hold_15_1 - 914.1457440293937 hold_15_2
   - 752.7987897125568 hold_15_3 - 639.8628375482211 hold_15_4
   - 556.3920983973164 hold_15_5 - 492.1859545004946 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_2_source_nic_in_2: switch_source_nic_in_2 = 1 ->
   - 3763.885863646265 hold_2_1 - 1927.280838724069 hold_2_2
   - 1295.25522824376 hold_2_3 - 975.3894784375707 hold_2_4
   - 782.2190032139786 hold_2_5 - 652.9131716164135 hold_2_6
   - 560.2931163062157 hold_2_7 - 490.6860517946388 hold_2_8
   - 436.4628324619231 hold_2_9 - 393.0309784833735 hold_2_10
   - 357.4605407418134 hold_2_11 - 327.7942180742089 hold_2_12
   - 302.6746772696222 hold_2_13 + flow_nic_in_2_2 + flow_3_2 <= 0
 constr_intergroup_throughput_0_nic_out_2_nic_in_0: switch_nic_out_2_nic_in_0 = 1 ->
   - 3167.736021998167 hold_0_1 - 1757.884028484232 hold_0_2
   - 1216.473072861668 hold_0_3 - 930.032292787944 hold_0_4
   - 752.7771727292528 hold_0_5 - 632.2722283205268 hold_0_6
   - 545.0244440939914 hold_0_7 - 478.9356984478936 hold_0_8
   - 427.1412680756396 hold_0_9 - 385.4561677448137 hold_0_10
   - 351.1838227822376 hold_0_11 - 322.5083986562149 hold_0_12
   - 298.1623673539816 hold_0_13 + flow_nic_in_0_0 + flow_1_0 <= 0
 constr_intergroup_throughput_4_nic_out_2_nic_in_4: switch_nic_out_2_nic_in_4 = 1 ->
   - 1263.157894736842 hold_4_1 - 657.5342465753424 hold_4_2
   - 444.4444444444444 hold_4_3 - 335.6643356643356 hold_4_4
   - 269.6629213483146 hold_4_5 - 225.3521126760563 hold_4_6
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_2_nic_in_5: switch_nic_out_2_nic_in_5 = 1 ->
   - 738.4615384615385 hold_5_1 - 480 hold_5_2 - 355.5555555555555 hold_5_3
   - 282.3529411764706 hold_5_4 - 234.1463414634146 hold_5_5 - 200 hold_5_6
   + flow_nic_in_5_5 + flow_6_5 <= 0
 constr_intergroup_throughput_10_nic_out_2_nic_in_7: switch_nic_out_2_nic_in_7 = 1 ->
   - 1263.157894736842 hold_10_1 - 657.5342465753424 hold_10_2
   - 444.4444444444444 hold_10_3 - 335.6643356643356 hold_10_4
   - 269.6629213483146 hold_10_5 - 225.3521126760563 hold_10_6
   + flow_nic_in_7_10 + flow_7_10 + flow_8_10 + flow_9_10 <= 0
 constr_intergroup_throughput_11_nic_out_2_nic_in_11: switch_nic_out_2_nic_in_11 = 1 ->
   - 1365.690376569038 hold_11_1 - 1191.240875912409 hold_11_2
   - 1056.31067961165 hold_11_3 - 948.8372093023256 hold_11_4
   - 861.2137203166227 hold_11_5 - 788.4057971014493 hold_11_6
   - 726.9487750556792 hold_11_7 - 674.3801652892562 hold_11_8
   - 628.9017341040462 hold_11_9 - 589.1696750902527 hold_11_10
   - 554.1595925297114 hold_11_11 - 523.0769230769231 hold_11_12
   - 495.2959028831563 hold_11_13 - 470.3170028818444 hold_11_14
   - 447.7366255144033 hold_11_15 - 427.2251308900524 hold_11_16
   - 408.5106382978723 hold_11_17 - 391.3669064748201 hold_11_18
   - 375.6041426927503 hold_11_19 - 361.0619469026548 hold_11_20
   - 347.6038338658147 hold_11_21 - 335.1129363449692 hold_11_22
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_2_nic_in_12: switch_nic_out_2_nic_in_12 = 1 ->
   - 898.2456140350877 hold_12_1 - 624.390243902439 hold_12_2
   - 478.5046728971963 hold_12_3 - 387.8787878787879 hold_12_4
   + flow_nic_in_12_12 + flow_13_12 <= 0
 constr_intergroup_throughput_14_nic_out_2_nic_in_14: switch_nic_out_2_nic_in_14 = 1 ->
   - 4921.859545004946 hold_14_1 - 2908.240794856809 hold_14_2
   - 2063.873911240149 hold_14_3 - 1599.485695917711 hold_14_4
   - 1305.694043558121 hold_14_5 - 1103.081356683662 hold_14_6
   - 954.9030896181155 hold_14_7 - 841.8203349687025 hold_14_8
   - 752.6849190742701 hold_14_9 - 680.6182464779099 hold_14_10
   - 621.1459243540132 hold_14_11 + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_2_nic_in_15: switch_nic_out_2_nic_in_15 = 1 ->
   - 1163.522992985191 hold_15_1 - 914.1457440293937 hold_15_2
   - 752.7987897125568 hold_15_3 - 639.8628375482211 hold_15_4
   - 556.3920983973164 hold_15_5 - 492.1859545004946 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_4_source_nic_in_4: switch_source_nic_in_4 = 1 ->
   - 1348.314606741573 hold_4_1 - 679.8866855524079 hold_4_2
   - 454.5454545454545 hold_4_3 - 341.3940256045519 hold_4_4
   - 273.3485193621868 hold_4_5 - 227.9202279202279 hold_4_6
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_0_nic_out_4_nic_in_0: switch_nic_out_4_nic_in_0 = 1 ->
   - 3167.736021998167 hold_0_1 - 1757.884028484232 hold_0_2
   - 1216.473072861668 hold_0_3 - 930.032292787944 hold_0_4
   - 752.7771727292528 hold_0_5 - 632.2722283205268 hold_0_6
   - 545.0244440939914 hold_0_7 - 478.9356984478936 hold_0_8
   - 427.1412680756396 hold_0_9 - 385.4561677448137 hold_0_10
   - 351.1838227822376 hold_0_11 - 322.5083986562149 hold_0_12
   - 298.1623673539816 hold_0_13 + flow_nic_in_0_0 + flow_1_0 <= 0
 constr_intergroup_throughput_2_nic_out_4_nic_in_2: switch_nic_out_4_nic_in_2 = 1 ->
   - 3167.736021998167 hold_2_1 - 1757.884028484232 hold_2_2
   - 1216.473072861668 hold_2_3 - 930.032292787944 hold_2_4
   - 752.7771727292528 hold_2_5 - 632.2722283205268 hold_2_6
   - 545.0244440939914 hold_2_7 - 478.9356984478936 hold_2_8
   - 427.1412680756396 hold_2_9 - 385.4561677448137 hold_2_10
   - 351.1838227822376 hold_2_11 - 322.5083986562149 hold_2_12
   - 298.1623673539816 hold_2_13 + flow_nic_in_2_2 + flow_3_2 <= 0
 constr_intergroup_throughput_5_nic_out_4_nic_in_5: switch_nic_out_4_nic_in_5 = 1 ->
   - 738.4615384615385 hold_5_1 - 480 hold_5_2 - 355.5555555555555 hold_5_3
   - 282.3529411764706 hold_5_4 - 234.1463414634146 hold_5_5 - 200 hold_5_6
   + flow_nic_in_5_5 + flow_6_5 <= 0
 constr_intergroup_throughput_10_nic_out_4_nic_in_7: switch_nic_out_4_nic_in_7 = 1 ->
   - 1263.157894736842 hold_10_1 - 657.5342465753424 hold_10_2
   - 444.4444444444444 hold_10_3 - 335.6643356643356 hold_10_4
   - 269.6629213483146 hold_10_5 - 225.3521126760563 hold_10_6
   + flow_nic_in_7_10 + flow_7_10 + flow_8_10 + flow_9_10 <= 0
 constr_intergroup_throughput_11_nic_out_4_nic_in_11: switch_nic_out_4_nic_in_11 = 1 ->
   - 1365.690376569038 hold_11_1 - 1191.240875912409 hold_11_2
   - 1056.31067961165 hold_11_3 - 948.8372093023256 hold_11_4
   - 861.2137203166227 hold_11_5 - 788.4057971014493 hold_11_6
   - 726.9487750556792 hold_11_7 - 674.3801652892562 hold_11_8
   - 628.9017341040462 hold_11_9 - 589.1696750902527 hold_11_10
   - 554.1595925297114 hold_11_11 - 523.0769230769231 hold_11_12
   - 495.2959028831563 hold_11_13 - 470.3170028818444 hold_11_14
   - 447.7366255144033 hold_11_15 - 427.2251308900524 hold_11_16
   - 408.5106382978723 hold_11_17 - 391.3669064748201 hold_11_18
   - 375.6041426927503 hold_11_19 - 361.0619469026548 hold_11_20
   - 347.6038338658147 hold_11_21 - 335.1129363449692 hold_11_22
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_4_nic_in_12: switch_nic_out_4_nic_in_12 = 1 ->
   - 898.2456140350877 hold_12_1 - 624.390243902439 hold_12_2
   - 478.5046728971963 hold_12_3 - 387.8787878787879 hold_12_4
   + flow_nic_in_12_12 + flow_13_12 <= 0
 constr_intergroup_throughput_14_nic_out_4_nic_in_14: switch_nic_out_4_nic_in_14 = 1 ->
   - 4921.859545004946 hold_14_1 - 2908.240794856809 hold_14_2
   - 2063.873911240149 hold_14_3 - 1599.485695917711 hold_14_4
   - 1305.694043558121 hold_14_5 - 1103.081356683662 hold_14_6
   - 954.9030896181155 hold_14_7 - 841.8203349687025 hold_14_8
   - 752.6849190742701 hold_14_9 - 680.6182464779099 hold_14_10
   - 621.1459243540132 hold_14_11 + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_4_nic_in_15: switch_nic_out_4_nic_in_15 = 1 ->
   - 1163.522992985191 hold_15_1 - 914.1457440293937 hold_15_2
   - 752.7987897125568 hold_15_3 - 639.8628375482211 hold_15_4
   - 556.3920983973164 hold_15_5 - 492.1859545004946 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_5_source_nic_in_5: switch_source_nic_in_5 = 1 ->
   - 1348.314606741573 hold_5_1 - 679.8866855524079 hold_5_2
   - 454.5454545454545 hold_5_3 - 341.3940256045519 hold_5_4
   - 273.3485193621868 hold_5_5 - 227.9202279202279 hold_5_6
   + flow_nic_in_5_5 + flow_6_5 <= 0
 constr_intergroup_throughput_0_nic_out_5_nic_in_0: switch_nic_out_5_nic_in_0 = 1 ->
   - 1138.714991762768 hold_0_1 - 883.8874680306906 hold_0_2
   - 722.2570532915361 hold_0_3 - 610.6007067137809 hold_0_4
   - 528.8446824789594 hold_0_5 - 466.3967611336031 hold_0_6
   - 417.1394085697042 hold_0_7 - 377.292576419214 hold_0_8
   - 344.3946188340807 hold_0_9 - 316.7736021998167 hold_0_10
   - 293.2541366143403 hold_0_11 - 272.9857819905213 hold_0_12
   - 255.3380125600296 hold_0_13 + flow_nic_in_0_0 + flow_1_0 <= 0
 constr_intergroup_throughput_2_nic_out_5_nic_in_2: switch_nic_out_5_nic_in_2 = 1 ->
   - 1138.714991762768 hold_2_1 - 883.8874680306906 hold_2_2
   - 722.2570532915361 hold_2_3 - 610.6007067137809 hold_2_4
   - 528.8446824789594 hold_2_5 - 466.3967611336031 hold_2_6
   - 417.1394085697042 hold_2_7 - 377.292576419214 hold_2_8
   - 344.3946188340807 hold_2_9 - 316.7736021998167 hold_2_10
   - 293.2541366143403 hold_2_11 - 272.9857819905213 hold_2_12
   - 255.3380125600296 hold_2_13 + flow_nic_in_2_2 + flow_3_2 <= 0
 constr_intergroup_throughput_4_nic_out_5_nic_in_4: switch_nic_out_5_nic_in_4 = 1 ->
   - 738.4615384615385 hold_4_1 - 480 hold_4_2 - 355.5555555555555 hold_4_3
   - 282.3529411764706 hold_4_4 - 234.1463414634146 hold_4_5 - 200 hold_4_6
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_10_nic_out_5_nic_in_7: switch_nic_out_5_nic_in_7 = 1 ->
   - 738.4615384615385 hold_10_1 - 480 hold_10_2
   - 355.5555555555555 hold_10_3 - 282.3529411764706 hold_10_4
   - 234.1463414634146 hold_10_5 - 200 hold_10_6 + flow_nic_in_7_10
   + flow_7_10 + flow_8_10 + flow_9_10 <= 0
 constr_intergroup_throughput_11_nic_out_5_nic_in_11: switch_nic_out_5_nic_in_11 = 1 ->
   - 5891.696750902527 hold_11_1 - 3610.619469026549 hold_11_2
   - 2602.870813397129 hold_11_3 - 2034.912718204489 hold_11_4
   - 1670.419651995905 hold_11_5 - 1416.666666666667 hold_11_6
   - 1229.841748304446 hold_11_7 - 1086.551264980027 hold_11_8
   - 973.1663685152056 hold_11_9 - 881.2095032397407 hold_11_10
   - 805.1307350764677 hold_11_11 - 741.1444141689373 hold_11_12
   - 686.5797223390829 hold_11_13 - 639.4984326018808 hold_11_14
   - 598.4598459845985 hold_11_15 - 562.3707787732598 hold_11_16
   - 530.3867403314916 hold_11_17 - 501.8450184501845 hold_11_18
   - 476.2182667055733 hold_11_19 - 453.081621321488 hold_11_20
   - 432.0889594916601 hold_11_21 - 412.9554655870446 hold_11_22
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_5_nic_in_12: switch_nic_out_5_nic_in_12 = 1 ->
   - 1815.602836879433 hold_12_1 - 962.406015037594 hold_12_2
   - 654.7314578005115 hold_12_3 - 496.1240310077519 hold_12_4
   + flow_nic_in_12_12 + flow_13_12 <= 0
 constr_intergroup_throughput_14_nic_out_5_nic_in_14: switch_nic_out_5_nic_in_14 = 1 ->
   - 1306.036745406824 hold_14_1 - 1103.325942350333 hold_14_2
   - 955.0863723608446 hold_14_3 - 841.9627749576989 hold_14_4
   - 752.7987897125568 hold_14_5 - 680.7113543091656 hold_14_6
   - 621.223470661673 hold_14_7 - 571.2973593570608 hold_14_8
   - 528.7991498405951 hold_14_9 - 492.1859545004946 hold_14_10
   - 460.3145235892692 hold_14_11 + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_5_nic_in_15: switch_nic_out_5_nic_in_15 = 1 ->
   - 3367.47123843898 hold_15_1 - 1881.759737804109 hold_15_2
   - 1305.694043558121 hold_15_3 - 999.665171097569 hold_15_4
   - 809.8518960559865 hold_15_5 - 680.6182464779099 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_10_source_nic_in_7: switch_source_nic_in_7 = 1 ->
   - 1348.314606741573 hold_10_1 - 679.8866855524079 hold_10_2
   - 454.5454545454545 hold_10_3 - 341.3940256045519 hold_10_4
   - 273.3485193621868 hold_10_5 - 227.9202279202279 hold_10_6
   + flow_nic_in_7_10 + flow_7_10 + flow_8_10 + flow_9_10 <= 0
 constr_intergroup_throughput_0_nic_out_7_nic_in_0: switch_nic_out_7_nic_in_0 = 1 ->
   - 3167.736021998167 hold_0_1 - 1757.884028484232 hold_0_2
   - 1216.473072861668 hold_0_3 - 930.032292787944 hold_0_4
   - 752.7771727292528 hold_0_5 - 632.2722283205268 hold_0_6
   - 545.0244440939914 hold_0_7 - 478.9356984478936 hold_0_8
   - 427.1412680756396 hold_0_9 - 385.4561677448137 hold_0_10
   - 351.1838227822376 hold_0_11 - 322.5083986562149 hold_0_12
   - 298.1623673539816 hold_0_13 + flow_nic_in_0_0 + flow_1_0 <= 0
 constr_intergroup_throughput_2_nic_out_7_nic_in_2: switch_nic_out_7_nic_in_2 = 1 ->
   - 3167.736021998167 hold_2_1 - 1757.884028484232 hold_2_2
   - 1216.473072861668 hold_2_3 - 930.032292787944 hold_2_4
   - 752.7771727292528 hold_2_5 - 632.2722283205268 hold_2_6
   - 545.0244440939914 hold_2_7 - 478.9356984478936 hold_2_8
   - 427.1412680756396 hold_2_9 - 385.4561677448137 hold_2_10
   - 351.1838227822376 hold_2_11 - 322.5083986562149 hold_2_12
   - 298.1623673539816 hold_2_13 + flow_nic_in_2_2 + flow_3_2 <= 0
 constr_intergroup_throughput_4_nic_out_7_nic_in_4: switch_nic_out_7_nic_in_4 = 1 ->
   - 1263.157894736842 hold_4_1 - 657.5342465753424 hold_4_2
   - 444.4444444444444 hold_4_3 - 335.6643356643356 hold_4_4
   - 269.6629213483146 hold_4_5 - 225.3521126760563 hold_4_6
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_7_nic_in_5: switch_nic_out_7_nic_in_5 = 1 ->
   - 738.4615384615385 hold_5_1 - 480 hold_5_2 - 355.5555555555555 hold_5_3
   - 282.3529411764706 hold_5_4 - 234.1463414634146 hold_5_5 - 200 hold_5_6
   + flow_nic_in_5_5 + flow_6_5 <= 0
 constr_intergroup_throughput_11_nic_out_7_nic_in_11: switch_nic_out_7_nic_in_11 = 1 ->
   - 1365.690376569038 hold_11_1 - 1191.240875912409 hold_11_2
   - 1056.31067961165 hold_11_3 - 948.8372093023256 hold_11_4
   - 861.2137203166227 hold_11_5 - 788.4057971014493 hold_11_6
   - 726.9487750556792 hold_11_7 - 674.3801652892562 hold_11_8
   - 628.9017341040462 hold_11_9 - 589.1696750902527 hold_11_10
   - 554.1595925297114 hold_11_11 - 523.0769230769231 hold_11_12
   - 495.2959028831563 hold_11_13 - 470.3170028818444 hold_11_14
   - 447.7366255144033 hold_11_15 - 427.2251308900524 hold_11_16
   - 408.5106382978723 hold_11_17 - 391.3669064748201 hold_11_18
   - 375.6041426927503 hold_11_19 - 361.0619469026548 hold_11_20
   - 347.6038338658147 hold_11_21 - 335.1129363449692 hold_11_22
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_7_nic_in_12: switch_nic_out_7_nic_in_12 = 1 ->
   - 898.2456140350877 hold_12_1 - 624.390243902439 hold_12_2
   - 478.5046728971963 hold_12_3 - 387.8787878787879 hold_12_4
   + flow_nic_in_12_12 + flow_13_12 <= 0
 constr_intergroup_throughput_14_nic_out_7_nic_in_14: switch_nic_out_7_nic_in_14 = 1 ->
   - 4921.859545004946 hold_14_1 - 2908.240794856809 hold_14_2
   - 2063.873911240149 hold_14_3 - 1599.485695917711 hold_14_4
   - 1305.694043558121 hold_14_5 - 1103.081356683662 hold_14_6
   - 954.9030896181155 hold_14_7 - 841.8203349687025 hold_14_8
   - 752.6849190742701 hold_14_9 - 680.6182464779099 hold_14_10
   - 621.1459243540132 hold_14_11 + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_7_nic_in_15: switch_nic_out_7_nic_in_15 = 1 ->
   - 1163.522992985191 hold_15_1 - 914.1457440293937 hold_15_2
   - 752.7987897125568 hold_15_3 - 639.8628375482211 hold_15_4
   - 556.3920983973164 hold_15_5 - 492.1859545004946 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_11_source_nic_in_11: switch_source_nic_in_11 = 1 ->
   - 8352.09825997953 hold_11_1 - 4406.047516198704 hold_11_2
   - 2992.299229922993 hold_11_3 - 2265.40810660744 hold_11_4
   - 1822.649095376368 hold_11_5 - 1524.663677130045 hold_11_6
   - 1310.42235426369 hold_11_7 - 1148.972120529428 hold_11_8
   - 1022.940955246333 hold_11_9 - 921.825576140985 hold_11_10
   - 838.9020252904288 hold_11_11 - 769.6661007357103 hold_11_12
   - 710.9871917748541 hold_11_13 - 660.6217616580311 hold_11_14
   - 616.9199364935361 hold_11_15 - 578.6413274712806 hold_11_16
   - 544.8354143019296 hold_11_17 - 514.7615442846328 hold_11_18
   - 487.8340407724039 hold_11_19 - 463.5836836723099 hold_11_20
   - 441.6301347621368 hold_11_21 - 421.6618437370815 hold_11_22
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_0_nic_out_11_nic_in_0: switch_nic_out_11_nic_in_0 = 1 ->
   - 1138.714991762768 hold_0_1 - 883.8874680306906 hold_0_2
   - 722.2570532915361 hold_0_3 - 610.6007067137809 hold_0_4
   - 528.8446824789594 hold_0_5 - 466.3967611336031 hold_0_6
   - 417.1394085697042 hold_0_7 - 377.292576419214 hold_0_8
   - 344.3946188340807 hold_0_9 - 316.7736021998167 hold_0_10
   - 293.2541366143403 hold_0_11 - 272.9857819905213 hold_0_12
   - 255.3380125600296 hold_0_13 + flow_nic_in_0_0 + flow_1_0 <= 0
 constr_intergroup_throughput_2_nic_out_11_nic_in_2: switch_nic_out_11_nic_in_2 = 1 ->
   - 1138.714991762768 hold_2_1 - 883.8874680306906 hold_2_2
   - 722.2570532915361 hold_2_3 - 610.6007067137809 hold_2_4
   - 528.8446824789594 hold_2_5 - 466.3967611336031 hold_2_6
   - 417.1394085697042 hold_2_7 - 377.292576419214 hold_2_8
   - 344.3946188340807 hold_2_9 - 316.7736021998167 hold_2_10
   - 293.2541366143403 hold_2_11 - 272.9857819905213 hold_2_12
   - 255.3380125600296 hold_2_13 + flow_nic_in_2_2 + flow_3_2 <= 0
 constr_intergroup_throughput_4_nic_out_11_nic_in_4: switch_nic_out_11_nic_in_4 = 1 ->
   - 738.4615384615385 hold_4_1 - 480 hold_4_2 - 355.5555555555555 hold_4_3
   - 282.3529411764706 hold_4_4 - 234.1463414634146 hold_4_5 - 200 hold_4_6
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_11_nic_in_5: switch_nic_out_11_nic_in_5 = 1 ->
   - 1263.157894736842 hold_5_1 - 657.5342465753424 hold_5_2
   - 444.4444444444444 hold_5_3 - 335.6643356643356 hold_5_4
   - 269.6629213483146 hold_5_5 - 225.3521126760563 hold_5_6
   + flow_nic_in_5_5 + flow_6_5 <= 0
 constr_intergroup_throughput_10_nic_out_11_nic_in_7: switch_nic_out_11_nic_in_7 = 1 ->
   - 738.4615384615385 hold_10_1 - 480 hold_10_2
   - 355.5555555555555 hold_10_3 - 282.3529411764706 hold_10_4
   - 234.1463414634146 hold_10_5 - 200 hold_10_6 + flow_nic_in_7_10
   + flow_7_10 + flow_8_10 + flow_9_10 <= 0
 constr_intergroup_throughput_12_nic_out_11_nic_in_12: switch_nic_out_11_nic_in_12 = 1 ->
   - 1815.602836879433 hold_12_1 - 962.406015037594 hold_12_2
   - 654.7314578005115 hold_12_3 - 496.1240310077519 hold_12_4
   + flow_nic_in_12_12 + flow_13_12 <= 0
 constr_intergroup_throughput_14_nic_out_11_nic_in_14: switch_nic_out_11_nic_in_14 = 1 ->
   - 1306.036745406824 hold_14_1 - 1103.325942350333 hold_14_2
   - 955.0863723608446 hold_14_3 - 841.9627749576989 hold_14_4
   - 752.7987897125568 hold_14_5 - 680.7113543091656 hold_14_6
   - 621.223470661673 hold_14_7 - 571.2973593570608 hold_14_8
   - 528.7991498405951 hold_14_9 - 492.1859545004946 hold_14_10
   - 460.3145235892692 hold_14_11 + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_11_nic_in_15: switch_nic_out_11_nic_in_15 = 1 ->
   - 3367.47123843898 hold_15_1 - 1881.759737804109 hold_15_2
   - 1305.694043558121 hold_15_3 - 999.665171097569 hold_15_4
   - 809.8518960559865 hold_15_5 - 680.6182464779099 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_12_source_nic_in_12: switch_source_nic_in_12 = 1 ->
   - 1996.879875195008 hold_12_1 - 1011.058451816746 hold_12_2
   - 676.8905341089371 hold_12_3 - 508.7440381558029 hold_12_4
   + flow_nic_in_12_12 + flow_13_12 <= 0
 constr_intergroup_throughput_0_nic_out_12_nic_in_0: switch_nic_out_12_nic_in_0 = 1 ->
   - 1138.714991762768 hold_0_1 - 883.8874680306906 hold_0_2
   - 722.2570532915361 hold_0_3 - 610.6007067137809 hold_0_4
   - 528.8446824789594 hold_0_5 - 466.3967611336031 hold_0_6
   - 417.1394085697042 hold_0_7 - 377.292576419214 hold_0_8
   - 344.3946188340807 hold_0_9 - 316.7736021998167 hold_0_10
   - 293.2541366143403 hold_0_11 - 272.9857819905213 hold_0_12
   - 255.3380125600296 hold_0_13 + flow_nic_in_0_0 + flow_1_0 <= 0
 constr_intergroup_throughput_2_nic_out_12_nic_in_2: switch_nic_out_12_nic_in_2 = 1 ->
   - 1138.714991762768 hold_2_1 - 883.8874680306906 hold_2_2
   - 722.2570532915361 hold_2_3 - 610.6007067137809 hold_2_4
   - 528.8446824789594 hold_2_5 - 466.3967611336031 hold_2_6
   - 417.1394085697042 hold_2_7 - 377.292576419214 hold_2_8
   - 344.3946188340807 hold_2_9 - 316.7736021998167 hold_2_10
   - 293.2541366143403 hold_2_11 - 272.9857819905213 hold_2_12
   - 255.3380125600296 hold_2_13 + flow_nic_in_2_2 + flow_3_2 <= 0
 constr_intergroup_throughput_4_nic_out_12_nic_in_4: switch_nic_out_12_nic_in_4 = 1 ->
   - 738.4615384615385 hold_4_1 - 480 hold_4_2 - 355.5555555555555 hold_4_3
   - 282.3529411764706 hold_4_4 - 234.1463414634146 hold_4_5 - 200 hold_4_6
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_12_nic_in_5: switch_nic_out_12_nic_in_5 = 1 ->
   - 1263.157894736842 hold_5_1 - 657.5342465753424 hold_5_2
   - 444.4444444444444 hold_5_3 - 335.6643356643356 hold_5_4
   - 269.6629213483146 hold_5_5 - 225.3521126760563 hold_5_6
   + flow_nic_in_5_5 + flow_6_5 <= 0
 constr_intergroup_throughput_10_nic_out_12_nic_in_7: switch_nic_out_12_nic_in_7 = 1 ->
   - 738.4615384615385 hold_10_1 - 480 hold_10_2
   - 355.5555555555555 hold_10_3 - 282.3529411764706 hold_10_4
   - 234.1463414634146 hold_10_5 - 200 hold_10_6 + flow_nic_in_7_10
   + flow_7_10 + flow_8_10 + flow_9_10 <= 0
 constr_intergroup_throughput_11_nic_out_12_nic_in_11: switch_nic_out_12_nic_in_11 = 1 ->
   - 5891.696750902527 hold_11_1 - 3610.619469026549 hold_11_2
   - 2602.870813397129 hold_11_3 - 2034.912718204489 hold_11_4
   - 1670.419651995905 hold_11_5 - 1416.666666666667 hold_11_6
   - 1229.841748304446 hold_11_7 - 1086.551264980027 hold_11_8
   - 973.1663685152056 hold_11_9 - 881.2095032397407 hold_11_10
   - 805.1307350764677 hold_11_11 - 741.1444141689373 hold_11_12
   - 686.5797223390829 hold_11_13 - 639.4984326018808 hold_11_14
   - 598.4598459845985 hold_11_15 - 562.3707787732598 hold_11_16
   - 530.3867403314916 hold_11_17 - 501.8450184501845 hold_11_18
   - 476.2182667055733 hold_11_19 - 453.081621321488 hold_11_20
   - 432.0889594916601 hold_11_21 - 412.9554655870446 hold_11_22
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_14_nic_out_12_nic_in_14: switch_nic_out_12_nic_in_14 = 1 ->
   - 1306.036745406824 hold_14_1 - 1103.325942350333 hold_14_2
   - 955.0863723608446 hold_14_3 - 841.9627749576989 hold_14_4
   - 752.7987897125568 hold_14_5 - 680.7113543091656 hold_14_6
   - 621.223470661673 hold_14_7 - 571.2973593570608 hold_14_8
   - 528.7991498405951 hold_14_9 - 492.1859545004946 hold_14_10
   - 460.3145235892692 hold_14_11 + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_12_nic_in_15: switch_nic_out_12_nic_in_15 = 1 ->
   - 3367.47123843898 hold_15_1 - 1881.759737804109 hold_15_2
   - 1305.694043558121 hold_15_3 - 999.665171097569 hold_15_4
   - 809.8518960559865 hold_15_5 - 680.6182464779099 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_14_source_nic_in_14: switch_source_nic_in_14 = 1 ->
   - 6528.470217790607 hold_14_1 - 3403.09123238955 hold_14_2
   - 2301.359726204792 hold_14_3 - 1738.522814618126 hold_14_4
   - 1396.889562629835 hold_14_5 - 1167.472197456713 hold_14_6
   - 1002.781024545564 hold_14_7 - 878.810356398573 hold_14_8
   - 782.1193926629153 hold_14_9 - 704.5963014358132 hold_14_10
   - 641.0553709000026 hold_14_11 + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_0_nic_out_14_nic_in_0: switch_nic_out_14_nic_in_0 = 1 ->
   - 3167.736021998167 hold_0_1 - 1757.884028484232 hold_0_2
   - 1216.473072861668 hold_0_3 - 930.032292787944 hold_0_4
   - 752.7771727292528 hold_0_5 - 632.2722283205268 hold_0_6
   - 545.0244440939914 hold_0_7 - 478.9356984478936 hold_0_8
   - 427.1412680756396 hold_0_9 - 385.4561677448137 hold_0_10
   - 351.1838227822376 hold_0_11 - 322.5083986562149 hold_0_12
   - 298.1623673539816 hold_0_13 + flow_nic_in_0_0 + flow_1_0 <= 0
 constr_intergroup_throughput_2_nic_out_14_nic_in_2: switch_nic_out_14_nic_in_2 = 1 ->
   - 3167.736021998167 hold_2_1 - 1757.884028484232 hold_2_2
   - 1216.473072861668 hold_2_3 - 930.032292787944 hold_2_4
   - 752.7771727292528 hold_2_5 - 632.2722283205268 hold_2_6
   - 545.0244440939914 hold_2_7 - 478.9356984478936 hold_2_8
   - 427.1412680756396 hold_2_9 - 385.4561677448137 hold_2_10
   - 351.1838227822376 hold_2_11 - 322.5083986562149 hold_2_12
   - 298.1623673539816 hold_2_13 + flow_nic_in_2_2 + flow_3_2 <= 0
 constr_intergroup_throughput_4_nic_out_14_nic_in_4: switch_nic_out_14_nic_in_4 = 1 ->
   - 1263.157894736842 hold_4_1 - 657.5342465753424 hold_4_2
   - 444.4444444444444 hold_4_3 - 335.6643356643356 hold_4_4
   - 269.6629213483146 hold_4_5 - 225.3521126760563 hold_4_6
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_14_nic_in_5: switch_nic_out_14_nic_in_5 = 1 ->
   - 738.4615384615385 hold_5_1 - 480 hold_5_2 - 355.5555555555555 hold_5_3
   - 282.3529411764706 hold_5_4 - 234.1463414634146 hold_5_5 - 200 hold_5_6
   + flow_nic_in_5_5 + flow_6_5 <= 0
 constr_intergroup_throughput_10_nic_out_14_nic_in_7: switch_nic_out_14_nic_in_7 = 1 ->
   - 1263.157894736842 hold_10_1 - 657.5342465753424 hold_10_2
   - 444.4444444444444 hold_10_3 - 335.6643356643356 hold_10_4
   - 269.6629213483146 hold_10_5 - 225.3521126760563 hold_10_6
   + flow_nic_in_7_10 + flow_7_10 + flow_8_10 + flow_9_10 <= 0
 constr_intergroup_throughput_11_nic_out_14_nic_in_11: switch_nic_out_14_nic_in_11 = 1 ->
   - 1365.690376569038 hold_11_1 - 1191.240875912409 hold_11_2
   - 1056.31067961165 hold_11_3 - 948.8372093023256 hold_11_4
   - 861.2137203166227 hold_11_5 - 788.4057971014493 hold_11_6
   - 726.9487750556792 hold_11_7 - 674.3801652892562 hold_11_8
   - 628.9017341040462 hold_11_9 - 589.1696750902527 hold_11_10
   - 554.1595925297114 hold_11_11 - 523.0769230769231 hold_11_12
   - 495.2959028831563 hold_11_13 - 470.3170028818444 hold_11_14
   - 447.7366255144033 hold_11_15 - 427.2251308900524 hold_11_16
   - 408.5106382978723 hold_11_17 - 391.3669064748201 hold_11_18
   - 375.6041426927503 hold_11_19 - 361.0619469026548 hold_11_20
   - 347.6038338658147 hold_11_21 - 335.1129363449692 hold_11_22
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_14_nic_in_12: switch_nic_out_14_nic_in_12 = 1 ->
   - 898.2456140350877 hold_12_1 - 624.390243902439 hold_12_2
   - 478.5046728971963 hold_12_3 - 387.8787878787879 hold_12_4
   + flow_nic_in_12_12 + flow_13_12 <= 0
 constr_intergroup_throughput_15_nic_out_14_nic_in_15: switch_nic_out_14_nic_in_15 = 1 ->
   - 1163.522992985191 hold_15_1 - 914.1457440293937 hold_15_2
   - 752.7987897125568 hold_15_3 - 639.8628375482211 hold_15_4
   - 556.3920983973164 hold_15_5 - 492.1859545004946 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_15_source_nic_in_15: switch_source_nic_in_15 = 1 ->
   - 4049.259480279933 hold_15_1 - 2077.199231903821 hold_15_2
   - 1396.889562629835 hold_15_3 - 1052.260583931315 hold_15_4
   - 844.0288127735122 hold_15_5 - 704.5963014358132 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_0_nic_out_15_nic_in_0: switch_nic_out_15_nic_in_0 = 1 ->
   - 1138.714991762768 hold_0_1 - 883.8874680306906 hold_0_2
   - 722.2570532915361 hold_0_3 - 610.6007067137809 hold_0_4
   - 528.8446824789594 hold_0_5 - 466.3967611336031 hold_0_6
   - 417.1394085697042 hold_0_7 - 377.292576419214 hold_0_8
   - 344.3946188340807 hold_0_9 - 316.7736021998167 hold_0_10
   - 293.2541366143403 hold_0_11 - 272.9857819905213 hold_0_12
   - 255.3380125600296 hold_0_13 + flow_nic_in_0_0 + flow_1_0 <= 0
 constr_intergroup_throughput_2_nic_out_15_nic_in_2: switch_nic_out_15_nic_in_2 = 1 ->
   - 1138.714991762768 hold_2_1 - 883.8874680306906 hold_2_2
   - 722.2570532915361 hold_2_3 - 610.6007067137809 hold_2_4
   - 528.8446824789594 hold_2_5 - 466.3967611336031 hold_2_6
   - 417.1394085697042 hold_2_7 - 377.292576419214 hold_2_8
   - 344.3946188340807 hold_2_9 - 316.7736021998167 hold_2_10
   - 293.2541366143403 hold_2_11 - 272.9857819905213 hold_2_12
   - 255.3380125600296 hold_2_13 + flow_nic_in_2_2 + flow_3_2 <= 0
 constr_intergroup_throughput_4_nic_out_15_nic_in_4: switch_nic_out_15_nic_in_4 = 1 ->
   - 738.4615384615385 hold_4_1 - 480 hold_4_2 - 355.5555555555555 hold_4_3
   - 282.3529411764706 hold_4_4 - 234.1463414634146 hold_4_5 - 200 hold_4_6
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_15_nic_in_5: switch_nic_out_15_nic_in_5 = 1 ->
   - 1263.157894736842 hold_5_1 - 657.5342465753424 hold_5_2
   - 444.4444444444444 hold_5_3 - 335.6643356643356 hold_5_4
   - 269.6629213483146 hold_5_5 - 225.3521126760563 hold_5_6
   + flow_nic_in_5_5 + flow_6_5 <= 0
 constr_intergroup_throughput_10_nic_out_15_nic_in_7: switch_nic_out_15_nic_in_7 = 1 ->
   - 738.4615384615385 hold_10_1 - 480 hold_10_2
   - 355.5555555555555 hold_10_3 - 282.3529411764706 hold_10_4
   - 234.1463414634146 hold_10_5 - 200 hold_10_6 + flow_nic_in_7_10
   + flow_7_10 + flow_8_10 + flow_9_10 <= 0
 constr_intergroup_throughput_11_nic_out_15_nic_in_11: switch_nic_out_15_nic_in_11 = 1 ->
   - 5891.696750902527 hold_11_1 - 3610.619469026549 hold_11_2
   - 2602.870813397129 hold_11_3 - 2034.912718204489 hold_11_4
   - 1670.419651995905 hold_11_5 - 1416.666666666667 hold_11_6
   - 1229.841748304446 hold_11_7 - 1086.551264980027 hold_11_8
   - 973.1663685152056 hold_11_9 - 881.2095032397407 hold_11_10
   - 805.1307350764677 hold_11_11 - 741.1444141689373 hold_11_12
   - 686.5797223390829 hold_11_13 - 639.4984326018808 hold_11_14
   - 598.4598459845985 hold_11_15 - 562.3707787732598 hold_11_16
   - 530.3867403314916 hold_11_17 - 501.8450184501845 hold_11_18
   - 476.2182667055733 hold_11_19 - 453.081621321488 hold_11_20
   - 432.0889594916601 hold_11_21 - 412.9554655870446 hold_11_22
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_15_nic_in_12: switch_nic_out_15_nic_in_12 = 1 ->
   - 1815.602836879433 hold_12_1 - 962.406015037594 hold_12_2
   - 654.7314578005115 hold_12_3 - 496.1240310077519 hold_12_4
   + flow_nic_in_12_12 + flow_13_12 <= 0
 constr_intergroup_throughput_14_nic_out_15_nic_in_14: switch_nic_out_15_nic_in_14 = 1 ->
   - 1306.036745406824 hold_14_1 - 1103.325942350333 hold_14_2
   - 955.0863723608446 hold_14_3 - 841.9627749576989 hold_14_4
   - 752.7987897125568 hold_14_5 - 680.7113543091656 hold_14_6
   - 621.223470661673 hold_14_7 - 571.2973593570608 hold_14_8
   - 528.7991498405951 hold_14_9 - 492.1859545004946 hold_14_10
   - 460.3145235892692 hold_14_11 + flow_nic_in_14_14 <= 0
 inactive_end_not_complete_0: is_active_0 = 0 -> is_end_complete_0 = 0
 inactive_end_not_complete_1: is_active_1 = 0 -> is_end_complete_1 = 0
 active_end_val_0: is_active_0 = 1 -> - start_0 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11 - 12 hold_0_12
   - 13 hold_0_13 - 14 hold_0_14 - 15 hold_0_15 - 16 hold_0_16
   - 17 hold_0_17 - 18 hold_0_18 - 19 hold_0_19 - 20 hold_0_20
   - 21 hold_0_21 - 22 hold_0_22 - 23 hold_0_23 - 24 hold_0_24
   - 25 hold_0_25 - 26 hold_0_26 + val_end_0 = 0
 inactive_end_zero_0: is_active_0 = 0 -> val_end_0 = 0
 active_end_val_1: is_active_1 = 1 -> - start_1 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11 - 12 hold_1_12
   - 13 hold_1_13 - 14 hold_1_14 - 15 hold_1_15 - 16 hold_1_16
   - 17 hold_1_17 - 18 hold_1_18 - 19 hold_1_19 - 20 hold_1_20
   - 21 hold_1_21 - 22 hold_1_22 - 23 hold_1_23 - 24 hold_1_24
   - 25 hold_1_25 - 26 hold_1_26 + val_end_1 = 0
 inactive_end_zero_1: is_active_1 = 0 -> val_end_1 = 0
 inactive_end_not_complete_2: is_active_2 = 0 -> is_end_complete_2 = 0
 inactive_end_not_complete_3: is_active_3 = 0 -> is_end_complete_3 = 0
 active_end_val_2: is_active_2 = 1 -> - start_2 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 - 12 hold_2_12
   - 13 hold_2_13 - 14 hold_2_14 - 15 hold_2_15 - 16 hold_2_16
   - 17 hold_2_17 - 18 hold_2_18 - 19 hold_2_19 - 20 hold_2_20
   - 21 hold_2_21 - 22 hold_2_22 - 23 hold_2_23 - 24 hold_2_24
   - 25 hold_2_25 - 26 hold_2_26 + val_end_2 = 0
 inactive_end_zero_2: is_active_2 = 0 -> val_end_2 = 0
 active_end_val_3: is_active_3 = 1 -> - start_3 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   - 8 hold_3_8 - 9 hold_3_9 - 10 hold_3_10 - 11 hold_3_11 - 12 hold_3_12
   - 13 hold_3_13 - 14 hold_3_14 - 15 hold_3_15 - 16 hold_3_16
   - 17 hold_3_17 - 18 hold_3_18 - 19 hold_3_19 - 20 hold_3_20
   - 21 hold_3_21 - 22 hold_3_22 - 23 hold_3_23 - 24 hold_3_24
   - 25 hold_3_25 - 26 hold_3_26 + val_end_3 = 0
 inactive_end_zero_3: is_active_3 = 0 -> val_end_3 = 0
 inactive_end_not_complete_4: is_active_4 = 0 -> is_end_complete_4 = 0
 active_end_val_4: is_active_4 = 1 -> - start_4 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   - 8 hold_4_8 - 9 hold_4_9 - 10 hold_4_10 - 11 hold_4_11 - 12 hold_4_12
   - 13 hold_4_13 - 14 hold_4_14 - 15 hold_4_15 - 16 hold_4_16
   - 17 hold_4_17 - 18 hold_4_18 - 19 hold_4_19 - 20 hold_4_20
   - 21 hold_4_21 - 22 hold_4_22 - 23 hold_4_23 - 24 hold_4_24
   - 25 hold_4_25 - 26 hold_4_26 + val_end_4 = 0
 inactive_end_zero_4: is_active_4 = 0 -> val_end_4 = 0
 inactive_end_not_complete_5: is_active_5 = 0 -> is_end_complete_5 = 0
 inactive_end_not_complete_6: is_active_6 = 0 -> is_end_complete_6 = 0
 active_end_val_5: is_active_5 = 1 -> - start_5 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11 - 12 hold_5_12
   - 13 hold_5_13 - 14 hold_5_14 - 15 hold_5_15 - 16 hold_5_16
   - 17 hold_5_17 - 18 hold_5_18 - 19 hold_5_19 - 20 hold_5_20
   - 21 hold_5_21 - 22 hold_5_22 - 23 hold_5_23 - 24 hold_5_24
   - 25 hold_5_25 - 26 hold_5_26 + val_end_5 = 0
 inactive_end_zero_5: is_active_5 = 0 -> val_end_5 = 0
 active_end_val_6: is_active_6 = 1 -> - start_6 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11 - 12 hold_6_12
   - 13 hold_6_13 - 14 hold_6_14 - 15 hold_6_15 - 16 hold_6_16
   - 17 hold_6_17 - 18 hold_6_18 - 19 hold_6_19 - 20 hold_6_20
   - 21 hold_6_21 - 22 hold_6_22 - 23 hold_6_23 - 24 hold_6_24
   - 25 hold_6_25 - 26 hold_6_26 + val_end_6 = 0
 inactive_end_zero_6: is_active_6 = 0 -> val_end_6 = 0
 inactive_end_not_complete_7: is_active_7 = 0 -> is_end_complete_7 = 0
 inactive_end_not_complete_8: is_active_8 = 0 -> is_end_complete_8 = 0
 inactive_end_not_complete_9: is_active_9 = 0 -> is_end_complete_9 = 0
 inactive_end_not_complete_10: is_active_10 = 0 -> is_end_complete_10 = 0
 active_end_val_7: is_active_7 = 1 -> - start_7 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   - 8 hold_7_8 - 9 hold_7_9 - 10 hold_7_10 - 11 hold_7_11 - 12 hold_7_12
   - 13 hold_7_13 - 14 hold_7_14 - 15 hold_7_15 - 16 hold_7_16
   - 17 hold_7_17 - 18 hold_7_18 - 19 hold_7_19 - 20 hold_7_20
   - 21 hold_7_21 - 22 hold_7_22 - 23 hold_7_23 - 24 hold_7_24
   - 25 hold_7_25 - 26 hold_7_26 + val_end_7 = 0
 inactive_end_zero_7: is_active_7 = 0 -> val_end_7 = 0
 active_end_val_8: is_active_8 = 1 -> - start_8 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11 - 12 hold_8_12
   - 13 hold_8_13 - 14 hold_8_14 - 15 hold_8_15 - 16 hold_8_16
   - 17 hold_8_17 - 18 hold_8_18 - 19 hold_8_19 - 20 hold_8_20
   - 21 hold_8_21 - 22 hold_8_22 - 23 hold_8_23 - 24 hold_8_24
   - 25 hold_8_25 - 26 hold_8_26 + val_end_8 = 0
 inactive_end_zero_8: is_active_8 = 0 -> val_end_8 = 0
 active_end_val_9: is_active_9 = 1 -> - start_9 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   - 8 hold_9_8 - 9 hold_9_9 - 10 hold_9_10 - 11 hold_9_11 - 12 hold_9_12
   - 13 hold_9_13 - 14 hold_9_14 - 15 hold_9_15 - 16 hold_9_16
   - 17 hold_9_17 - 18 hold_9_18 - 19 hold_9_19 - 20 hold_9_20
   - 21 hold_9_21 - 22 hold_9_22 - 23 hold_9_23 - 24 hold_9_24
   - 25 hold_9_25 - 26 hold_9_26 + val_end_9 = 0
 inactive_end_zero_9: is_active_9 = 0 -> val_end_9 = 0
 active_end_val_10: is_active_10 = 1 -> - start_10 - hold_10_1
   - 2 hold_10_2 - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6
   - 7 hold_10_7 - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10
   - 11 hold_10_11 - 12 hold_10_12 - 13 hold_10_13 - 14 hold_10_14
   - 15 hold_10_15 - 16 hold_10_16 - 17 hold_10_17 - 18 hold_10_18
   - 19 hold_10_19 - 20 hold_10_20 - 21 hold_10_21 - 22 hold_10_22
   - 23 hold_10_23 - 24 hold_10_24 - 25 hold_10_25 - 26 hold_10_26
   + val_end_10 = 0
 inactive_end_zero_10: is_active_10 = 0 -> val_end_10 = 0
 inactive_end_not_complete_11: is_active_11 = 0 -> is_end_complete_11 = 0
 active_end_val_11: is_active_11 = 1 -> - start_11 - hold_11_1
   - 2 hold_11_2 - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6
   - 7 hold_11_7 - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10
   - 11 hold_11_11 - 12 hold_11_12 - 13 hold_11_13 - 14 hold_11_14
   - 15 hold_11_15 - 16 hold_11_16 - 17 hold_11_17 - 18 hold_11_18
   - 19 hold_11_19 - 20 hold_11_20 - 21 hold_11_21 - 22 hold_11_22
   - 23 hold_11_23 - 24 hold_11_24 - 25 hold_11_25 - 26 hold_11_26
   + val_end_11 = 0
 inactive_end_zero_11: is_active_11 = 0 -> val_end_11 = 0
 inactive_end_not_complete_12: is_active_12 = 0 -> is_end_complete_12 = 0
 inactive_end_not_complete_13: is_active_13 = 0 -> is_end_complete_13 = 0
 active_end_val_12: is_active_12 = 1 -> - start_12 - hold_12_1
   - 2 hold_12_2 - 3 hold_12_3 - 4 hold_12_4 - 5 hold_12_5 - 6 hold_12_6
   - 7 hold_12_7 - 8 hold_12_8 - 9 hold_12_9 - 10 hold_12_10
   - 11 hold_12_11 - 12 hold_12_12 - 13 hold_12_13 - 14 hold_12_14
   - 15 hold_12_15 - 16 hold_12_16 - 17 hold_12_17 - 18 hold_12_18
   - 19 hold_12_19 - 20 hold_12_20 - 21 hold_12_21 - 22 hold_12_22
   - 23 hold_12_23 - 24 hold_12_24 - 25 hold_12_25 - 26 hold_12_26
   + val_end_12 = 0
 inactive_end_zero_12: is_active_12 = 0 -> val_end_12 = 0
 active_end_val_13: is_active_13 = 1 -> - start_13 - hold_13_1
   - 2 hold_13_2 - 3 hold_13_3 - 4 hold_13_4 - 5 hold_13_5 - 6 hold_13_6
   - 7 hold_13_7 - 8 hold_13_8 - 9 hold_13_9 - 10 hold_13_10
   - 11 hold_13_11 - 12 hold_13_12 - 13 hold_13_13 - 14 hold_13_14
   - 15 hold_13_15 - 16 hold_13_16 - 17 hold_13_17 - 18 hold_13_18
   - 19 hold_13_19 - 20 hold_13_20 - 21 hold_13_21 - 22 hold_13_22
   - 23 hold_13_23 - 24 hold_13_24 - 25 hold_13_25 - 26 hold_13_26
   + val_end_13 = 0
 inactive_end_zero_13: is_active_13 = 0 -> val_end_13 = 0
 inactive_end_not_complete_14: is_active_14 = 0 -> is_end_complete_14 = 0
 active_end_val_14: is_active_14 = 1 -> - start_14 - hold_14_1
   - 2 hold_14_2 - 3 hold_14_3 - 4 hold_14_4 - 5 hold_14_5 - 6 hold_14_6
   - 7 hold_14_7 - 8 hold_14_8 - 9 hold_14_9 - 10 hold_14_10
   - 11 hold_14_11 - 12 hold_14_12 - 13 hold_14_13 - 14 hold_14_14
   - 15 hold_14_15 - 16 hold_14_16 - 17 hold_14_17 - 18 hold_14_18
   - 19 hold_14_19 - 20 hold_14_20 - 21 hold_14_21 - 22 hold_14_22
   - 23 hold_14_23 - 24 hold_14_24 - 25 hold_14_25 - 26 hold_14_26
   + val_end_14 = 0
 inactive_end_zero_14: is_active_14 = 0 -> val_end_14 = 0
 inactive_end_not_complete_15: is_active_15 = 0 -> is_end_complete_15 = 0
 active_end_val_15: is_active_15 = 1 -> - start_15 - hold_15_1
   - 2 hold_15_2 - 3 hold_15_3 - 4 hold_15_4 - 5 hold_15_5 - 6 hold_15_6
   - 7 hold_15_7 - 8 hold_15_8 - 9 hold_15_9 - 10 hold_15_10
   - 11 hold_15_11 - 12 hold_15_12 - 13 hold_15_13 - 14 hold_15_14
   - 15 hold_15_15 - 16 hold_15_16 - 17 hold_15_17 - 18 hold_15_18
   - 19 hold_15_19 - 20 hold_15_20 - 21 hold_15_21 - 22 hold_15_22
   - 23 hold_15_23 - 24 hold_15_24 - 25 hold_15_25 - 26 hold_15_26
   + val_end_15 = 0
 inactive_end_zero_15: is_active_15 = 0 -> val_end_15 = 0
 group_synch_start_constr_0_1: group_tp_active_0 = 1 -> start_0 - start_1
   = 0
 group_synch_hold_constr_0_1_1: group_tp_active_0 = 1 -> hold_0_1
   - hold_1_1 = 0
 group_synch_hold_constr_0_1_2: group_tp_active_0 = 1 -> hold_0_2
   - hold_1_2 = 0
 group_synch_hold_constr_0_1_3: group_tp_active_0 = 1 -> hold_0_3
   - hold_1_3 = 0
 group_synch_hold_constr_0_1_4: group_tp_active_0 = 1 -> hold_0_4
   - hold_1_4 = 0
 group_synch_hold_constr_0_1_5: group_tp_active_0 = 1 -> hold_0_5
   - hold_1_5 = 0
 group_synch_hold_constr_0_1_6: group_tp_active_0 = 1 -> hold_0_6
   - hold_1_6 = 0
 group_synch_hold_constr_0_1_7: group_tp_active_0 = 1 -> hold_0_7
   - hold_1_7 = 0
 group_synch_hold_constr_0_1_8: group_tp_active_0 = 1 -> hold_0_8
   - hold_1_8 = 0
 group_synch_hold_constr_0_1_9: group_tp_active_0 = 1 -> hold_0_9
   - hold_1_9 = 0
 group_synch_hold_constr_0_1_10: group_tp_active_0 = 1 -> hold_0_10
   - hold_1_10 = 0
 group_synch_hold_constr_0_1_11: group_tp_active_0 = 1 -> hold_0_11
   - hold_1_11 = 0
 group_synch_hold_constr_0_1_12: group_tp_active_0 = 1 -> hold_0_12
   - hold_1_12 = 0
 group_synch_hold_constr_0_1_13: group_tp_active_0 = 1 -> hold_0_13
   - hold_1_13 = 0
 group_synch_hold_constr_0_1_14: group_tp_active_0 = 1 -> hold_0_14
   - hold_1_14 = 0
 group_synch_hold_constr_0_1_15: group_tp_active_0 = 1 -> hold_0_15
   - hold_1_15 = 0
 group_synch_hold_constr_0_1_16: group_tp_active_0 = 1 -> hold_0_16
   - hold_1_16 = 0
 group_synch_hold_constr_0_1_17: group_tp_active_0 = 1 -> hold_0_17
   - hold_1_17 = 0
 group_synch_hold_constr_0_1_18: group_tp_active_0 = 1 -> hold_0_18
   - hold_1_18 = 0
 group_synch_hold_constr_0_1_19: group_tp_active_0 = 1 -> hold_0_19
   - hold_1_19 = 0
 group_synch_hold_constr_0_1_20: group_tp_active_0 = 1 -> hold_0_20
   - hold_1_20 = 0
 group_synch_hold_constr_0_1_21: group_tp_active_0 = 1 -> hold_0_21
   - hold_1_21 = 0
 group_synch_hold_constr_0_1_22: group_tp_active_0 = 1 -> hold_0_22
   - hold_1_22 = 0
 group_synch_hold_constr_0_1_23: group_tp_active_0 = 1 -> hold_0_23
   - hold_1_23 = 0
 group_synch_hold_constr_0_1_24: group_tp_active_0 = 1 -> hold_0_24
   - hold_1_24 = 0
 group_synch_hold_constr_0_1_25: group_tp_active_0 = 1 -> hold_0_25
   - hold_1_25 = 0
 group_synch_hold_constr_0_1_26: group_tp_active_0 = 1 -> hold_0_26
   - hold_1_26 = 0
 tp_off_limit_constr_0_1: group_tp_active_0 = 0 -> hold_0_1 = 0
 tp_off_limit_constr_0_2: group_tp_active_0 = 0 -> hold_0_2 = 0
 tp_off_limit_constr_0_3: group_tp_active_0 = 0 -> hold_0_3 = 0
 tp_off_limit_constr_0_4: group_tp_active_0 = 0 -> hold_0_4 = 0
 tp_off_limit_constr_0_5: group_tp_active_0 = 0 -> hold_0_5 = 0
 tp_off_limit_constr_0_6: group_tp_active_0 = 0 -> hold_0_6 = 0
 tp_off_limit_constr_0_7: group_tp_active_0 = 0 -> hold_0_7 = 0
 tp_off_limit_constr_0_8: group_tp_active_0 = 0 -> hold_0_8 = 0
 tp_off_limit_constr_0_9: group_tp_active_0 = 0 -> hold_0_9 = 0
 tp_off_limit_constr_0_10: group_tp_active_0 = 0 -> hold_0_10 = 0
 tp_off_limit_constr_0_11: group_tp_active_0 = 0 -> hold_0_11 = 0
 tp_off_limit_constr_0_12: group_tp_active_0 = 0 -> hold_0_12 = 0
 tp_off_limit_constr_0_13: group_tp_active_0 = 0 -> hold_0_13 = 0
 tp_off_limit_constr_0_14: group_tp_active_0 = 0 -> hold_0_14 = 0
 tp_off_limit_constr_0_15: group_tp_active_0 = 0 -> hold_0_15 = 0
 tp_off_limit_constr_0_16: group_tp_active_0 = 0 -> hold_0_16 = 0
 tp_off_limit_constr_0_17: group_tp_active_0 = 0 -> hold_0_17 = 0
 tp_off_limit_constr_0_18: group_tp_active_0 = 0 -> hold_0_18 = 0
 tp_off_limit_constr_0_19: group_tp_active_0 = 0 -> hold_0_19 = 0
 tp_off_limit_constr_0_20: group_tp_active_0 = 0 -> hold_0_20 = 0
 tp_off_limit_constr_0_21: group_tp_active_0 = 0 -> hold_0_21 = 0
 tp_off_limit_constr_0_22: group_tp_active_0 = 0 -> hold_0_22 = 0
 tp_off_limit_constr_0_23: group_tp_active_0 = 0 -> hold_0_23 = 0
 tp_off_limit_constr_0_24: group_tp_active_0 = 0 -> hold_0_24 = 0
 tp_off_limit_constr_0_25: group_tp_active_0 = 0 -> hold_0_25 = 0
 tp_off_limit_constr_0_26: group_tp_active_0 = 0 -> hold_0_26 = 0
 tp_off_limit_constr_1_1: group_tp_active_0 = 0 -> hold_1_1 = 0
 tp_off_limit_constr_1_2: group_tp_active_0 = 0 -> hold_1_2 = 0
 tp_off_limit_constr_1_3: group_tp_active_0 = 0 -> hold_1_3 = 0
 tp_off_limit_constr_1_4: group_tp_active_0 = 0 -> hold_1_4 = 0
 tp_off_limit_constr_1_5: group_tp_active_0 = 0 -> hold_1_5 = 0
 tp_off_limit_constr_1_6: group_tp_active_0 = 0 -> hold_1_6 = 0
 tp_off_limit_constr_1_7: group_tp_active_0 = 0 -> hold_1_7 = 0
 tp_off_limit_constr_1_8: group_tp_active_0 = 0 -> hold_1_8 = 0
 tp_off_limit_constr_1_9: group_tp_active_0 = 0 -> hold_1_9 = 0
 tp_off_limit_constr_1_10: group_tp_active_0 = 0 -> hold_1_10 = 0
 tp_off_limit_constr_1_11: group_tp_active_0 = 0 -> hold_1_11 = 0
 tp_off_limit_constr_1_12: group_tp_active_0 = 0 -> hold_1_12 = 0
 tp_off_limit_constr_1_13: group_tp_active_0 = 0 -> hold_1_13 = 0
 tp_off_limit_constr_1_14: group_tp_active_0 = 0 -> hold_1_14 = 0
 tp_off_limit_constr_1_15: group_tp_active_0 = 0 -> hold_1_15 = 0
 tp_off_limit_constr_1_16: group_tp_active_0 = 0 -> hold_1_16 = 0
 tp_off_limit_constr_1_17: group_tp_active_0 = 0 -> hold_1_17 = 0
 tp_off_limit_constr_1_18: group_tp_active_0 = 0 -> hold_1_18 = 0
 tp_off_limit_constr_1_19: group_tp_active_0 = 0 -> hold_1_19 = 0
 tp_off_limit_constr_1_20: group_tp_active_0 = 0 -> hold_1_20 = 0
 tp_off_limit_constr_1_21: group_tp_active_0 = 0 -> hold_1_21 = 0
 tp_off_limit_constr_1_22: group_tp_active_0 = 0 -> hold_1_22 = 0
 tp_off_limit_constr_1_23: group_tp_active_0 = 0 -> hold_1_23 = 0
 tp_off_limit_constr_1_24: group_tp_active_0 = 0 -> hold_1_24 = 0
 tp_off_limit_constr_1_25: group_tp_active_0 = 0 -> hold_1_25 = 0
 tp_off_limit_constr_1_26: group_tp_active_0 = 0 -> hold_1_26 = 0
 group_synch_start_constr_2_3: group_tp_active_2 = 1 -> start_2 - start_3
   = 0
 group_synch_hold_constr_2_3_1: group_tp_active_2 = 1 -> hold_2_1
   - hold_3_1 = 0
 group_synch_hold_constr_2_3_2: group_tp_active_2 = 1 -> hold_2_2
   - hold_3_2 = 0
 group_synch_hold_constr_2_3_3: group_tp_active_2 = 1 -> hold_2_3
   - hold_3_3 = 0
 group_synch_hold_constr_2_3_4: group_tp_active_2 = 1 -> hold_2_4
   - hold_3_4 = 0
 group_synch_hold_constr_2_3_5: group_tp_active_2 = 1 -> hold_2_5
   - hold_3_5 = 0
 group_synch_hold_constr_2_3_6: group_tp_active_2 = 1 -> hold_2_6
   - hold_3_6 = 0
 group_synch_hold_constr_2_3_7: group_tp_active_2 = 1 -> hold_2_7
   - hold_3_7 = 0
 group_synch_hold_constr_2_3_8: group_tp_active_2 = 1 -> hold_2_8
   - hold_3_8 = 0
 group_synch_hold_constr_2_3_9: group_tp_active_2 = 1 -> hold_2_9
   - hold_3_9 = 0
 group_synch_hold_constr_2_3_10: group_tp_active_2 = 1 -> hold_2_10
   - hold_3_10 = 0
 group_synch_hold_constr_2_3_11: group_tp_active_2 = 1 -> hold_2_11
   - hold_3_11 = 0
 group_synch_hold_constr_2_3_12: group_tp_active_2 = 1 -> hold_2_12
   - hold_3_12 = 0
 group_synch_hold_constr_2_3_13: group_tp_active_2 = 1 -> hold_2_13
   - hold_3_13 = 0
 group_synch_hold_constr_2_3_14: group_tp_active_2 = 1 -> hold_2_14
   - hold_3_14 = 0
 group_synch_hold_constr_2_3_15: group_tp_active_2 = 1 -> hold_2_15
   - hold_3_15 = 0
 group_synch_hold_constr_2_3_16: group_tp_active_2 = 1 -> hold_2_16
   - hold_3_16 = 0
 group_synch_hold_constr_2_3_17: group_tp_active_2 = 1 -> hold_2_17
   - hold_3_17 = 0
 group_synch_hold_constr_2_3_18: group_tp_active_2 = 1 -> hold_2_18
   - hold_3_18 = 0
 group_synch_hold_constr_2_3_19: group_tp_active_2 = 1 -> hold_2_19
   - hold_3_19 = 0
 group_synch_hold_constr_2_3_20: group_tp_active_2 = 1 -> hold_2_20
   - hold_3_20 = 0
 group_synch_hold_constr_2_3_21: group_tp_active_2 = 1 -> hold_2_21
   - hold_3_21 = 0
 group_synch_hold_constr_2_3_22: group_tp_active_2 = 1 -> hold_2_22
   - hold_3_22 = 0
 group_synch_hold_constr_2_3_23: group_tp_active_2 = 1 -> hold_2_23
   - hold_3_23 = 0
 group_synch_hold_constr_2_3_24: group_tp_active_2 = 1 -> hold_2_24
   - hold_3_24 = 0
 group_synch_hold_constr_2_3_25: group_tp_active_2 = 1 -> hold_2_25
   - hold_3_25 = 0
 group_synch_hold_constr_2_3_26: group_tp_active_2 = 1 -> hold_2_26
   - hold_3_26 = 0
 tp_off_limit_constr_2_1: group_tp_active_2 = 0 -> hold_2_1 = 0
 tp_off_limit_constr_2_2: group_tp_active_2 = 0 -> hold_2_2 = 0
 tp_off_limit_constr_2_3: group_tp_active_2 = 0 -> hold_2_3 = 0
 tp_off_limit_constr_2_4: group_tp_active_2 = 0 -> hold_2_4 = 0
 tp_off_limit_constr_2_5: group_tp_active_2 = 0 -> hold_2_5 = 0
 tp_off_limit_constr_2_6: group_tp_active_2 = 0 -> hold_2_6 = 0
 tp_off_limit_constr_2_7: group_tp_active_2 = 0 -> hold_2_7 = 0
 tp_off_limit_constr_2_8: group_tp_active_2 = 0 -> hold_2_8 = 0
 tp_off_limit_constr_2_9: group_tp_active_2 = 0 -> hold_2_9 = 0
 tp_off_limit_constr_2_10: group_tp_active_2 = 0 -> hold_2_10 = 0
 tp_off_limit_constr_2_11: group_tp_active_2 = 0 -> hold_2_11 = 0
 tp_off_limit_constr_2_12: group_tp_active_2 = 0 -> hold_2_12 = 0
 tp_off_limit_constr_2_13: group_tp_active_2 = 0 -> hold_2_13 = 0
 tp_off_limit_constr_2_14: group_tp_active_2 = 0 -> hold_2_14 = 0
 tp_off_limit_constr_2_15: group_tp_active_2 = 0 -> hold_2_15 = 0
 tp_off_limit_constr_2_16: group_tp_active_2 = 0 -> hold_2_16 = 0
 tp_off_limit_constr_2_17: group_tp_active_2 = 0 -> hold_2_17 = 0
 tp_off_limit_constr_2_18: group_tp_active_2 = 0 -> hold_2_18 = 0
 tp_off_limit_constr_2_19: group_tp_active_2 = 0 -> hold_2_19 = 0
 tp_off_limit_constr_2_20: group_tp_active_2 = 0 -> hold_2_20 = 0
 tp_off_limit_constr_2_21: group_tp_active_2 = 0 -> hold_2_21 = 0
 tp_off_limit_constr_2_22: group_tp_active_2 = 0 -> hold_2_22 = 0
 tp_off_limit_constr_2_23: group_tp_active_2 = 0 -> hold_2_23 = 0
 tp_off_limit_constr_2_24: group_tp_active_2 = 0 -> hold_2_24 = 0
 tp_off_limit_constr_2_25: group_tp_active_2 = 0 -> hold_2_25 = 0
 tp_off_limit_constr_2_26: group_tp_active_2 = 0 -> hold_2_26 = 0
 tp_off_limit_constr_3_1: group_tp_active_2 = 0 -> hold_3_1 = 0
 tp_off_limit_constr_3_2: group_tp_active_2 = 0 -> hold_3_2 = 0
 tp_off_limit_constr_3_3: group_tp_active_2 = 0 -> hold_3_3 = 0
 tp_off_limit_constr_3_4: group_tp_active_2 = 0 -> hold_3_4 = 0
 tp_off_limit_constr_3_5: group_tp_active_2 = 0 -> hold_3_5 = 0
 tp_off_limit_constr_3_6: group_tp_active_2 = 0 -> hold_3_6 = 0
 tp_off_limit_constr_3_7: group_tp_active_2 = 0 -> hold_3_7 = 0
 tp_off_limit_constr_3_8: group_tp_active_2 = 0 -> hold_3_8 = 0
 tp_off_limit_constr_3_9: group_tp_active_2 = 0 -> hold_3_9 = 0
 tp_off_limit_constr_3_10: group_tp_active_2 = 0 -> hold_3_10 = 0
 tp_off_limit_constr_3_11: group_tp_active_2 = 0 -> hold_3_11 = 0
 tp_off_limit_constr_3_12: group_tp_active_2 = 0 -> hold_3_12 = 0
 tp_off_limit_constr_3_13: group_tp_active_2 = 0 -> hold_3_13 = 0
 tp_off_limit_constr_3_14: group_tp_active_2 = 0 -> hold_3_14 = 0
 tp_off_limit_constr_3_15: group_tp_active_2 = 0 -> hold_3_15 = 0
 tp_off_limit_constr_3_16: group_tp_active_2 = 0 -> hold_3_16 = 0
 tp_off_limit_constr_3_17: group_tp_active_2 = 0 -> hold_3_17 = 0
 tp_off_limit_constr_3_18: group_tp_active_2 = 0 -> hold_3_18 = 0
 tp_off_limit_constr_3_19: group_tp_active_2 = 0 -> hold_3_19 = 0
 tp_off_limit_constr_3_20: group_tp_active_2 = 0 -> hold_3_20 = 0
 tp_off_limit_constr_3_21: group_tp_active_2 = 0 -> hold_3_21 = 0
 tp_off_limit_constr_3_22: group_tp_active_2 = 0 -> hold_3_22 = 0
 tp_off_limit_constr_3_23: group_tp_active_2 = 0 -> hold_3_23 = 0
 tp_off_limit_constr_3_24: group_tp_active_2 = 0 -> hold_3_24 = 0
 tp_off_limit_constr_3_25: group_tp_active_2 = 0 -> hold_3_25 = 0
 tp_off_limit_constr_3_26: group_tp_active_2 = 0 -> hold_3_26 = 0
 tp_off_limit_constr_4_7: group_tp_active_4 = 0 -> hold_4_7 = 0
 tp_off_limit_constr_4_8: group_tp_active_4 = 0 -> hold_4_8 = 0
 tp_off_limit_constr_4_9: group_tp_active_4 = 0 -> hold_4_9 = 0
 tp_off_limit_constr_4_10: group_tp_active_4 = 0 -> hold_4_10 = 0
 tp_off_limit_constr_4_11: group_tp_active_4 = 0 -> hold_4_11 = 0
 tp_off_limit_constr_4_12: group_tp_active_4 = 0 -> hold_4_12 = 0
 tp_off_limit_constr_4_13: group_tp_active_4 = 0 -> hold_4_13 = 0
 tp_off_limit_constr_4_14: group_tp_active_4 = 0 -> hold_4_14 = 0
 tp_off_limit_constr_4_15: group_tp_active_4 = 0 -> hold_4_15 = 0
 tp_off_limit_constr_4_16: group_tp_active_4 = 0 -> hold_4_16 = 0
 tp_off_limit_constr_4_17: group_tp_active_4 = 0 -> hold_4_17 = 0
 tp_off_limit_constr_4_18: group_tp_active_4 = 0 -> hold_4_18 = 0
 tp_off_limit_constr_4_19: group_tp_active_4 = 0 -> hold_4_19 = 0
 tp_off_limit_constr_4_20: group_tp_active_4 = 0 -> hold_4_20 = 0
 tp_off_limit_constr_4_21: group_tp_active_4 = 0 -> hold_4_21 = 0
 tp_off_limit_constr_4_22: group_tp_active_4 = 0 -> hold_4_22 = 0
 tp_off_limit_constr_4_23: group_tp_active_4 = 0 -> hold_4_23 = 0
 tp_off_limit_constr_4_24: group_tp_active_4 = 0 -> hold_4_24 = 0
 tp_off_limit_constr_4_25: group_tp_active_4 = 0 -> hold_4_25 = 0
 tp_off_limit_constr_4_26: group_tp_active_4 = 0 -> hold_4_26 = 0
 group_synch_start_constr_5_6: group_tp_active_5 = 1 -> start_5 - start_6
   = 0
 group_synch_hold_constr_5_6_1: group_tp_active_5 = 1 -> hold_5_1
   - hold_6_1 = 0
 group_synch_hold_constr_5_6_2: group_tp_active_5 = 1 -> hold_5_2
   - hold_6_2 = 0
 group_synch_hold_constr_5_6_3: group_tp_active_5 = 1 -> hold_5_3
   - hold_6_3 = 0
 group_synch_hold_constr_5_6_4: group_tp_active_5 = 1 -> hold_5_4
   - hold_6_4 = 0
 group_synch_hold_constr_5_6_5: group_tp_active_5 = 1 -> hold_5_5
   - hold_6_5 = 0
 group_synch_hold_constr_5_6_6: group_tp_active_5 = 1 -> hold_5_6
   - hold_6_6 = 0
 group_synch_hold_constr_5_6_7: group_tp_active_5 = 1 -> hold_5_7
   - hold_6_7 = 0
 group_synch_hold_constr_5_6_8: group_tp_active_5 = 1 -> hold_5_8
   - hold_6_8 = 0
 group_synch_hold_constr_5_6_9: group_tp_active_5 = 1 -> hold_5_9
   - hold_6_9 = 0
 group_synch_hold_constr_5_6_10: group_tp_active_5 = 1 -> hold_5_10
   - hold_6_10 = 0
 group_synch_hold_constr_5_6_11: group_tp_active_5 = 1 -> hold_5_11
   - hold_6_11 = 0
 group_synch_hold_constr_5_6_12: group_tp_active_5 = 1 -> hold_5_12
   - hold_6_12 = 0
 group_synch_hold_constr_5_6_13: group_tp_active_5 = 1 -> hold_5_13
   - hold_6_13 = 0
 group_synch_hold_constr_5_6_14: group_tp_active_5 = 1 -> hold_5_14
   - hold_6_14 = 0
 group_synch_hold_constr_5_6_15: group_tp_active_5 = 1 -> hold_5_15
   - hold_6_15 = 0
 group_synch_hold_constr_5_6_16: group_tp_active_5 = 1 -> hold_5_16
   - hold_6_16 = 0
 group_synch_hold_constr_5_6_17: group_tp_active_5 = 1 -> hold_5_17
   - hold_6_17 = 0
 group_synch_hold_constr_5_6_18: group_tp_active_5 = 1 -> hold_5_18
   - hold_6_18 = 0
 group_synch_hold_constr_5_6_19: group_tp_active_5 = 1 -> hold_5_19
   - hold_6_19 = 0
 group_synch_hold_constr_5_6_20: group_tp_active_5 = 1 -> hold_5_20
   - hold_6_20 = 0
 group_synch_hold_constr_5_6_21: group_tp_active_5 = 1 -> hold_5_21
   - hold_6_21 = 0
 group_synch_hold_constr_5_6_22: group_tp_active_5 = 1 -> hold_5_22
   - hold_6_22 = 0
 group_synch_hold_constr_5_6_23: group_tp_active_5 = 1 -> hold_5_23
   - hold_6_23 = 0
 group_synch_hold_constr_5_6_24: group_tp_active_5 = 1 -> hold_5_24
   - hold_6_24 = 0
 group_synch_hold_constr_5_6_25: group_tp_active_5 = 1 -> hold_5_25
   - hold_6_25 = 0
 group_synch_hold_constr_5_6_26: group_tp_active_5 = 1 -> hold_5_26
   - hold_6_26 = 0
 tp_on_limit_constr_5_13: group_tp_active_5 = 1 -> hold_5_13 = 0
 tp_on_limit_constr_5_14: group_tp_active_5 = 1 -> hold_5_14 = 0
 tp_on_limit_constr_5_15: group_tp_active_5 = 1 -> hold_5_15 = 0
 tp_on_limit_constr_5_16: group_tp_active_5 = 1 -> hold_5_16 = 0
 tp_on_limit_constr_5_17: group_tp_active_5 = 1 -> hold_5_17 = 0
 tp_on_limit_constr_5_18: group_tp_active_5 = 1 -> hold_5_18 = 0
 tp_on_limit_constr_5_19: group_tp_active_5 = 1 -> hold_5_19 = 0
 tp_on_limit_constr_5_20: group_tp_active_5 = 1 -> hold_5_20 = 0
 tp_on_limit_constr_5_21: group_tp_active_5 = 1 -> hold_5_21 = 0
 tp_on_limit_constr_5_22: group_tp_active_5 = 1 -> hold_5_22 = 0
 tp_on_limit_constr_5_23: group_tp_active_5 = 1 -> hold_5_23 = 0
 tp_on_limit_constr_5_24: group_tp_active_5 = 1 -> hold_5_24 = 0
 tp_on_limit_constr_5_25: group_tp_active_5 = 1 -> hold_5_25 = 0
 tp_on_limit_constr_5_26: group_tp_active_5 = 1 -> hold_5_26 = 0
 tp_on_limit_constr_6_13: group_tp_active_5 = 1 -> hold_6_13 = 0
 tp_on_limit_constr_6_14: group_tp_active_5 = 1 -> hold_6_14 = 0
 tp_on_limit_constr_6_15: group_tp_active_5 = 1 -> hold_6_15 = 0
 tp_on_limit_constr_6_16: group_tp_active_5 = 1 -> hold_6_16 = 0
 tp_on_limit_constr_6_17: group_tp_active_5 = 1 -> hold_6_17 = 0
 tp_on_limit_constr_6_18: group_tp_active_5 = 1 -> hold_6_18 = 0
 tp_on_limit_constr_6_19: group_tp_active_5 = 1 -> hold_6_19 = 0
 tp_on_limit_constr_6_20: group_tp_active_5 = 1 -> hold_6_20 = 0
 tp_on_limit_constr_6_21: group_tp_active_5 = 1 -> hold_6_21 = 0
 tp_on_limit_constr_6_22: group_tp_active_5 = 1 -> hold_6_22 = 0
 tp_on_limit_constr_6_23: group_tp_active_5 = 1 -> hold_6_23 = 0
 tp_on_limit_constr_6_24: group_tp_active_5 = 1 -> hold_6_24 = 0
 tp_on_limit_constr_6_25: group_tp_active_5 = 1 -> hold_6_25 = 0
 tp_on_limit_constr_6_26: group_tp_active_5 = 1 -> hold_6_26 = 0
 tp_off_limit_constr_5_1: group_tp_active_5 = 0 -> hold_5_1 = 0
 tp_off_limit_constr_5_2: group_tp_active_5 = 0 -> hold_5_2 = 0
 tp_off_limit_constr_5_3: group_tp_active_5 = 0 -> hold_5_3 = 0
 tp_off_limit_constr_5_4: group_tp_active_5 = 0 -> hold_5_4 = 0
 tp_off_limit_constr_5_5: group_tp_active_5 = 0 -> hold_5_5 = 0
 tp_off_limit_constr_5_6: group_tp_active_5 = 0 -> hold_5_6 = 0
 tp_off_limit_constr_5_7: group_tp_active_5 = 0 -> hold_5_7 = 0
 tp_off_limit_constr_5_8: group_tp_active_5 = 0 -> hold_5_8 = 0
 tp_off_limit_constr_5_9: group_tp_active_5 = 0 -> hold_5_9 = 0
 tp_off_limit_constr_5_10: group_tp_active_5 = 0 -> hold_5_10 = 0
 tp_off_limit_constr_5_11: group_tp_active_5 = 0 -> hold_5_11 = 0
 tp_off_limit_constr_5_12: group_tp_active_5 = 0 -> hold_5_12 = 0
 tp_off_limit_constr_5_13: group_tp_active_5 = 0 -> hold_5_13 = 0
 tp_off_limit_constr_5_14: group_tp_active_5 = 0 -> hold_5_14 = 0
 tp_off_limit_constr_5_15: group_tp_active_5 = 0 -> hold_5_15 = 0
 tp_off_limit_constr_5_16: group_tp_active_5 = 0 -> hold_5_16 = 0
 tp_off_limit_constr_5_17: group_tp_active_5 = 0 -> hold_5_17 = 0
 tp_off_limit_constr_5_18: group_tp_active_5 = 0 -> hold_5_18 = 0
 tp_off_limit_constr_5_19: group_tp_active_5 = 0 -> hold_5_19 = 0
 tp_off_limit_constr_5_20: group_tp_active_5 = 0 -> hold_5_20 = 0
 tp_off_limit_constr_5_21: group_tp_active_5 = 0 -> hold_5_21 = 0
 tp_off_limit_constr_5_22: group_tp_active_5 = 0 -> hold_5_22 = 0
 tp_off_limit_constr_5_23: group_tp_active_5 = 0 -> hold_5_23 = 0
 tp_off_limit_constr_5_24: group_tp_active_5 = 0 -> hold_5_24 = 0
 tp_off_limit_constr_5_25: group_tp_active_5 = 0 -> hold_5_25 = 0
 tp_off_limit_constr_5_26: group_tp_active_5 = 0 -> hold_5_26 = 0
 tp_off_limit_constr_6_1: group_tp_active_5 = 0 -> hold_6_1 = 0
 tp_off_limit_constr_6_2: group_tp_active_5 = 0 -> hold_6_2 = 0
 tp_off_limit_constr_6_3: group_tp_active_5 = 0 -> hold_6_3 = 0
 tp_off_limit_constr_6_4: group_tp_active_5 = 0 -> hold_6_4 = 0
 tp_off_limit_constr_6_5: group_tp_active_5 = 0 -> hold_6_5 = 0
 tp_off_limit_constr_6_6: group_tp_active_5 = 0 -> hold_6_6 = 0
 tp_off_limit_constr_6_7: group_tp_active_5 = 0 -> hold_6_7 = 0
 tp_off_limit_constr_6_8: group_tp_active_5 = 0 -> hold_6_8 = 0
 tp_off_limit_constr_6_9: group_tp_active_5 = 0 -> hold_6_9 = 0
 tp_off_limit_constr_6_10: group_tp_active_5 = 0 -> hold_6_10 = 0
 tp_off_limit_constr_6_11: group_tp_active_5 = 0 -> hold_6_11 = 0
 tp_off_limit_constr_6_12: group_tp_active_5 = 0 -> hold_6_12 = 0
 tp_off_limit_constr_6_13: group_tp_active_5 = 0 -> hold_6_13 = 0
 tp_off_limit_constr_6_14: group_tp_active_5 = 0 -> hold_6_14 = 0
 tp_off_limit_constr_6_15: group_tp_active_5 = 0 -> hold_6_15 = 0
 tp_off_limit_constr_6_16: group_tp_active_5 = 0 -> hold_6_16 = 0
 tp_off_limit_constr_6_17: group_tp_active_5 = 0 -> hold_6_17 = 0
 tp_off_limit_constr_6_18: group_tp_active_5 = 0 -> hold_6_18 = 0
 tp_off_limit_constr_6_19: group_tp_active_5 = 0 -> hold_6_19 = 0
 tp_off_limit_constr_6_20: group_tp_active_5 = 0 -> hold_6_20 = 0
 tp_off_limit_constr_6_21: group_tp_active_5 = 0 -> hold_6_21 = 0
 tp_off_limit_constr_6_22: group_tp_active_5 = 0 -> hold_6_22 = 0
 tp_off_limit_constr_6_23: group_tp_active_5 = 0 -> hold_6_23 = 0
 tp_off_limit_constr_6_24: group_tp_active_5 = 0 -> hold_6_24 = 0
 tp_off_limit_constr_6_25: group_tp_active_5 = 0 -> hold_6_25 = 0
 tp_off_limit_constr_6_26: group_tp_active_5 = 0 -> hold_6_26 = 0
 group_synch_start_constr_7_8: group_tp_active_7 = 1 -> start_7 - start_8
   = 0
 group_synch_start_constr_7_9: group_tp_active_7 = 1 -> start_7 - start_9
   = 0
 group_synch_start_constr_7_10: group_tp_active_7 = 1 -> start_7 - start_10
   = 0
 group_synch_hold_constr_7_8_1: group_tp_active_7 = 1 -> hold_7_1
   - hold_8_1 = 0
 group_synch_hold_constr_7_9_1: group_tp_active_7 = 1 -> hold_7_1
   - hold_9_1 = 0
 group_synch_hold_constr_7_10_1: group_tp_active_7 = 1 -> hold_7_1
   - hold_10_1 = 0
 group_synch_hold_constr_7_8_2: group_tp_active_7 = 1 -> hold_7_2
   - hold_8_2 = 0
 group_synch_hold_constr_7_9_2: group_tp_active_7 = 1 -> hold_7_2
   - hold_9_2 = 0
 group_synch_hold_constr_7_10_2: group_tp_active_7 = 1 -> hold_7_2
   - hold_10_2 = 0
 group_synch_hold_constr_7_8_3: group_tp_active_7 = 1 -> hold_7_3
   - hold_8_3 = 0
 group_synch_hold_constr_7_9_3: group_tp_active_7 = 1 -> hold_7_3
   - hold_9_3 = 0
 group_synch_hold_constr_7_10_3: group_tp_active_7 = 1 -> hold_7_3
   - hold_10_3 = 0
 group_synch_hold_constr_7_8_4: group_tp_active_7 = 1 -> hold_7_4
   - hold_8_4 = 0
 group_synch_hold_constr_7_9_4: group_tp_active_7 = 1 -> hold_7_4
   - hold_9_4 = 0
 group_synch_hold_constr_7_10_4: group_tp_active_7 = 1 -> hold_7_4
   - hold_10_4 = 0
 group_synch_hold_constr_7_8_5: group_tp_active_7 = 1 -> hold_7_5
   - hold_8_5 = 0
 group_synch_hold_constr_7_9_5: group_tp_active_7 = 1 -> hold_7_5
   - hold_9_5 = 0
 group_synch_hold_constr_7_10_5: group_tp_active_7 = 1 -> hold_7_5
   - hold_10_5 = 0
 group_synch_hold_constr_7_8_6: group_tp_active_7 = 1 -> hold_7_6
   - hold_8_6 = 0
 group_synch_hold_constr_7_9_6: group_tp_active_7 = 1 -> hold_7_6
   - hold_9_6 = 0
 group_synch_hold_constr_7_10_6: group_tp_active_7 = 1 -> hold_7_6
   - hold_10_6 = 0
 group_synch_hold_constr_7_8_7: group_tp_active_7 = 1 -> hold_7_7
   - hold_8_7 = 0
 group_synch_hold_constr_7_9_7: group_tp_active_7 = 1 -> hold_7_7
   - hold_9_7 = 0
 group_synch_hold_constr_7_10_7: group_tp_active_7 = 1 -> hold_7_7
   - hold_10_7 = 0
 group_synch_hold_constr_7_8_8: group_tp_active_7 = 1 -> hold_7_8
   - hold_8_8 = 0
 group_synch_hold_constr_7_9_8: group_tp_active_7 = 1 -> hold_7_8
   - hold_9_8 = 0
 group_synch_hold_constr_7_10_8: group_tp_active_7 = 1 -> hold_7_8
   - hold_10_8 = 0
 group_synch_hold_constr_7_8_9: group_tp_active_7 = 1 -> hold_7_9
   - hold_8_9 = 0
 group_synch_hold_constr_7_9_9: group_tp_active_7 = 1 -> hold_7_9
   - hold_9_9 = 0
 group_synch_hold_constr_7_10_9: group_tp_active_7 = 1 -> hold_7_9
   - hold_10_9 = 0
 group_synch_hold_constr_7_8_10: group_tp_active_7 = 1 -> hold_7_10
   - hold_8_10 = 0
 group_synch_hold_constr_7_9_10: group_tp_active_7 = 1 -> hold_7_10
   - hold_9_10 = 0
 group_synch_hold_constr_7_10_10: group_tp_active_7 = 1 -> hold_7_10
   - hold_10_10 = 0
 group_synch_hold_constr_7_8_11: group_tp_active_7 = 1 -> hold_7_11
   - hold_8_11 = 0
 group_synch_hold_constr_7_9_11: group_tp_active_7 = 1 -> hold_7_11
   - hold_9_11 = 0
 group_synch_hold_constr_7_10_11: group_tp_active_7 = 1 -> hold_7_11
   - hold_10_11 = 0
 group_synch_hold_constr_7_8_12: group_tp_active_7 = 1 -> hold_7_12
   - hold_8_12 = 0
 group_synch_hold_constr_7_9_12: group_tp_active_7 = 1 -> hold_7_12
   - hold_9_12 = 0
 group_synch_hold_constr_7_10_12: group_tp_active_7 = 1 -> hold_7_12
   - hold_10_12 = 0
 group_synch_hold_constr_7_8_13: group_tp_active_7 = 1 -> hold_7_13
   - hold_8_13 = 0
 group_synch_hold_constr_7_9_13: group_tp_active_7 = 1 -> hold_7_13
   - hold_9_13 = 0
 group_synch_hold_constr_7_10_13: group_tp_active_7 = 1 -> hold_7_13
   - hold_10_13 = 0
 group_synch_hold_constr_7_8_14: group_tp_active_7 = 1 -> hold_7_14
   - hold_8_14 = 0
 group_synch_hold_constr_7_9_14: group_tp_active_7 = 1 -> hold_7_14
   - hold_9_14 = 0
 group_synch_hold_constr_7_10_14: group_tp_active_7 = 1 -> hold_7_14
   - hold_10_14 = 0
 group_synch_hold_constr_7_8_15: group_tp_active_7 = 1 -> hold_7_15
   - hold_8_15 = 0
 group_synch_hold_constr_7_9_15: group_tp_active_7 = 1 -> hold_7_15
   - hold_9_15 = 0
 group_synch_hold_constr_7_10_15: group_tp_active_7 = 1 -> hold_7_15
   - hold_10_15 = 0
 group_synch_hold_constr_7_8_16: group_tp_active_7 = 1 -> hold_7_16
   - hold_8_16 = 0
 group_synch_hold_constr_7_9_16: group_tp_active_7 = 1 -> hold_7_16
   - hold_9_16 = 0
 group_synch_hold_constr_7_10_16: group_tp_active_7 = 1 -> hold_7_16
   - hold_10_16 = 0
 group_synch_hold_constr_7_8_17: group_tp_active_7 = 1 -> hold_7_17
   - hold_8_17 = 0
 group_synch_hold_constr_7_9_17: group_tp_active_7 = 1 -> hold_7_17
   - hold_9_17 = 0
 group_synch_hold_constr_7_10_17: group_tp_active_7 = 1 -> hold_7_17
   - hold_10_17 = 0
 group_synch_hold_constr_7_8_18: group_tp_active_7 = 1 -> hold_7_18
   - hold_8_18 = 0
 group_synch_hold_constr_7_9_18: group_tp_active_7 = 1 -> hold_7_18
   - hold_9_18 = 0
 group_synch_hold_constr_7_10_18: group_tp_active_7 = 1 -> hold_7_18
   - hold_10_18 = 0
 group_synch_hold_constr_7_8_19: group_tp_active_7 = 1 -> hold_7_19
   - hold_8_19 = 0
 group_synch_hold_constr_7_9_19: group_tp_active_7 = 1 -> hold_7_19
   - hold_9_19 = 0
 group_synch_hold_constr_7_10_19: group_tp_active_7 = 1 -> hold_7_19
   - hold_10_19 = 0
 group_synch_hold_constr_7_8_20: group_tp_active_7 = 1 -> hold_7_20
   - hold_8_20 = 0
 group_synch_hold_constr_7_9_20: group_tp_active_7 = 1 -> hold_7_20
   - hold_9_20 = 0
 group_synch_hold_constr_7_10_20: group_tp_active_7 = 1 -> hold_7_20
   - hold_10_20 = 0
 group_synch_hold_constr_7_8_21: group_tp_active_7 = 1 -> hold_7_21
   - hold_8_21 = 0
 group_synch_hold_constr_7_9_21: group_tp_active_7 = 1 -> hold_7_21
   - hold_9_21 = 0
 group_synch_hold_constr_7_10_21: group_tp_active_7 = 1 -> hold_7_21
   - hold_10_21 = 0
 group_synch_hold_constr_7_8_22: group_tp_active_7 = 1 -> hold_7_22
   - hold_8_22 = 0
 group_synch_hold_constr_7_9_22: group_tp_active_7 = 1 -> hold_7_22
   - hold_9_22 = 0
 group_synch_hold_constr_7_10_22: group_tp_active_7 = 1 -> hold_7_22
   - hold_10_22 = 0
 group_synch_hold_constr_7_8_23: group_tp_active_7 = 1 -> hold_7_23
   - hold_8_23 = 0
 group_synch_hold_constr_7_9_23: group_tp_active_7 = 1 -> hold_7_23
   - hold_9_23 = 0
 group_synch_hold_constr_7_10_23: group_tp_active_7 = 1 -> hold_7_23
   - hold_10_23 = 0
 group_synch_hold_constr_7_8_24: group_tp_active_7 = 1 -> hold_7_24
   - hold_8_24 = 0
 group_synch_hold_constr_7_9_24: group_tp_active_7 = 1 -> hold_7_24
   - hold_9_24 = 0
 group_synch_hold_constr_7_10_24: group_tp_active_7 = 1 -> hold_7_24
   - hold_10_24 = 0
 group_synch_hold_constr_7_8_25: group_tp_active_7 = 1 -> hold_7_25
   - hold_8_25 = 0
 group_synch_hold_constr_7_9_25: group_tp_active_7 = 1 -> hold_7_25
   - hold_9_25 = 0
 group_synch_hold_constr_7_10_25: group_tp_active_7 = 1 -> hold_7_25
   - hold_10_25 = 0
 group_synch_hold_constr_7_8_26: group_tp_active_7 = 1 -> hold_7_26
   - hold_8_26 = 0
 group_synch_hold_constr_7_9_26: group_tp_active_7 = 1 -> hold_7_26
   - hold_9_26 = 0
 group_synch_hold_constr_7_10_26: group_tp_active_7 = 1 -> hold_7_26
   - hold_10_26 = 0
 tp_on_limit_constr_7_25: group_tp_active_7 = 1 -> hold_7_25 = 0
 tp_on_limit_constr_7_26: group_tp_active_7 = 1 -> hold_7_26 = 0
 tp_on_limit_constr_8_25: group_tp_active_7 = 1 -> hold_8_25 = 0
 tp_on_limit_constr_8_26: group_tp_active_7 = 1 -> hold_8_26 = 0
 tp_on_limit_constr_9_25: group_tp_active_7 = 1 -> hold_9_25 = 0
 tp_on_limit_constr_9_26: group_tp_active_7 = 1 -> hold_9_26 = 0
 tp_on_limit_constr_10_25: group_tp_active_7 = 1 -> hold_10_25 = 0
 tp_on_limit_constr_10_26: group_tp_active_7 = 1 -> hold_10_26 = 0
 tp_off_limit_constr_7_1: group_tp_active_7 = 0 -> hold_7_1 = 0
 tp_off_limit_constr_7_2: group_tp_active_7 = 0 -> hold_7_2 = 0
 tp_off_limit_constr_7_3: group_tp_active_7 = 0 -> hold_7_3 = 0
 tp_off_limit_constr_7_4: group_tp_active_7 = 0 -> hold_7_4 = 0
 tp_off_limit_constr_7_5: group_tp_active_7 = 0 -> hold_7_5 = 0
 tp_off_limit_constr_7_6: group_tp_active_7 = 0 -> hold_7_6 = 0
 tp_off_limit_constr_7_7: group_tp_active_7 = 0 -> hold_7_7 = 0
 tp_off_limit_constr_7_8: group_tp_active_7 = 0 -> hold_7_8 = 0
 tp_off_limit_constr_7_9: group_tp_active_7 = 0 -> hold_7_9 = 0
 tp_off_limit_constr_7_10: group_tp_active_7 = 0 -> hold_7_10 = 0
 tp_off_limit_constr_7_11: group_tp_active_7 = 0 -> hold_7_11 = 0
 tp_off_limit_constr_7_12: group_tp_active_7 = 0 -> hold_7_12 = 0
 tp_off_limit_constr_7_13: group_tp_active_7 = 0 -> hold_7_13 = 0
 tp_off_limit_constr_7_14: group_tp_active_7 = 0 -> hold_7_14 = 0
 tp_off_limit_constr_7_15: group_tp_active_7 = 0 -> hold_7_15 = 0
 tp_off_limit_constr_7_16: group_tp_active_7 = 0 -> hold_7_16 = 0
 tp_off_limit_constr_7_17: group_tp_active_7 = 0 -> hold_7_17 = 0
 tp_off_limit_constr_7_18: group_tp_active_7 = 0 -> hold_7_18 = 0
 tp_off_limit_constr_7_19: group_tp_active_7 = 0 -> hold_7_19 = 0
 tp_off_limit_constr_7_20: group_tp_active_7 = 0 -> hold_7_20 = 0
 tp_off_limit_constr_7_21: group_tp_active_7 = 0 -> hold_7_21 = 0
 tp_off_limit_constr_7_22: group_tp_active_7 = 0 -> hold_7_22 = 0
 tp_off_limit_constr_7_23: group_tp_active_7 = 0 -> hold_7_23 = 0
 tp_off_limit_constr_7_24: group_tp_active_7 = 0 -> hold_7_24 = 0
 tp_off_limit_constr_7_25: group_tp_active_7 = 0 -> hold_7_25 = 0
 tp_off_limit_constr_7_26: group_tp_active_7 = 0 -> hold_7_26 = 0
 tp_off_limit_constr_8_1: group_tp_active_7 = 0 -> hold_8_1 = 0
 tp_off_limit_constr_8_2: group_tp_active_7 = 0 -> hold_8_2 = 0
 tp_off_limit_constr_8_3: group_tp_active_7 = 0 -> hold_8_3 = 0
 tp_off_limit_constr_8_4: group_tp_active_7 = 0 -> hold_8_4 = 0
 tp_off_limit_constr_8_5: group_tp_active_7 = 0 -> hold_8_5 = 0
 tp_off_limit_constr_8_6: group_tp_active_7 = 0 -> hold_8_6 = 0
 tp_off_limit_constr_8_7: group_tp_active_7 = 0 -> hold_8_7 = 0
 tp_off_limit_constr_8_8: group_tp_active_7 = 0 -> hold_8_8 = 0
 tp_off_limit_constr_8_9: group_tp_active_7 = 0 -> hold_8_9 = 0
 tp_off_limit_constr_8_10: group_tp_active_7 = 0 -> hold_8_10 = 0
 tp_off_limit_constr_8_11: group_tp_active_7 = 0 -> hold_8_11 = 0
 tp_off_limit_constr_8_12: group_tp_active_7 = 0 -> hold_8_12 = 0
 tp_off_limit_constr_8_13: group_tp_active_7 = 0 -> hold_8_13 = 0
 tp_off_limit_constr_8_14: group_tp_active_7 = 0 -> hold_8_14 = 0
 tp_off_limit_constr_8_15: group_tp_active_7 = 0 -> hold_8_15 = 0
 tp_off_limit_constr_8_16: group_tp_active_7 = 0 -> hold_8_16 = 0
 tp_off_limit_constr_8_17: group_tp_active_7 = 0 -> hold_8_17 = 0
 tp_off_limit_constr_8_18: group_tp_active_7 = 0 -> hold_8_18 = 0
 tp_off_limit_constr_8_19: group_tp_active_7 = 0 -> hold_8_19 = 0
 tp_off_limit_constr_8_20: group_tp_active_7 = 0 -> hold_8_20 = 0
 tp_off_limit_constr_8_21: group_tp_active_7 = 0 -> hold_8_21 = 0
 tp_off_limit_constr_8_22: group_tp_active_7 = 0 -> hold_8_22 = 0
 tp_off_limit_constr_8_23: group_tp_active_7 = 0 -> hold_8_23 = 0
 tp_off_limit_constr_8_24: group_tp_active_7 = 0 -> hold_8_24 = 0
 tp_off_limit_constr_8_25: group_tp_active_7 = 0 -> hold_8_25 = 0
 tp_off_limit_constr_8_26: group_tp_active_7 = 0 -> hold_8_26 = 0
 tp_off_limit_constr_9_1: group_tp_active_7 = 0 -> hold_9_1 = 0
 tp_off_limit_constr_9_2: group_tp_active_7 = 0 -> hold_9_2 = 0
 tp_off_limit_constr_9_3: group_tp_active_7 = 0 -> hold_9_3 = 0
 tp_off_limit_constr_9_4: group_tp_active_7 = 0 -> hold_9_4 = 0
 tp_off_limit_constr_9_5: group_tp_active_7 = 0 -> hold_9_5 = 0
 tp_off_limit_constr_9_6: group_tp_active_7 = 0 -> hold_9_6 = 0
 tp_off_limit_constr_9_7: group_tp_active_7 = 0 -> hold_9_7 = 0
 tp_off_limit_constr_9_8: group_tp_active_7 = 0 -> hold_9_8 = 0
 tp_off_limit_constr_9_9: group_tp_active_7 = 0 -> hold_9_9 = 0
 tp_off_limit_constr_9_10: group_tp_active_7 = 0 -> hold_9_10 = 0
 tp_off_limit_constr_9_11: group_tp_active_7 = 0 -> hold_9_11 = 0
 tp_off_limit_constr_9_12: group_tp_active_7 = 0 -> hold_9_12 = 0
 tp_off_limit_constr_9_13: group_tp_active_7 = 0 -> hold_9_13 = 0
 tp_off_limit_constr_9_14: group_tp_active_7 = 0 -> hold_9_14 = 0
 tp_off_limit_constr_9_15: group_tp_active_7 = 0 -> hold_9_15 = 0
 tp_off_limit_constr_9_16: group_tp_active_7 = 0 -> hold_9_16 = 0
 tp_off_limit_constr_9_17: group_tp_active_7 = 0 -> hold_9_17 = 0
 tp_off_limit_constr_9_18: group_tp_active_7 = 0 -> hold_9_18 = 0
 tp_off_limit_constr_9_19: group_tp_active_7 = 0 -> hold_9_19 = 0
 tp_off_limit_constr_9_20: group_tp_active_7 = 0 -> hold_9_20 = 0
 tp_off_limit_constr_9_21: group_tp_active_7 = 0 -> hold_9_21 = 0
 tp_off_limit_constr_9_22: group_tp_active_7 = 0 -> hold_9_22 = 0
 tp_off_limit_constr_9_23: group_tp_active_7 = 0 -> hold_9_23 = 0
 tp_off_limit_constr_9_24: group_tp_active_7 = 0 -> hold_9_24 = 0
 tp_off_limit_constr_9_25: group_tp_active_7 = 0 -> hold_9_25 = 0
 tp_off_limit_constr_9_26: group_tp_active_7 = 0 -> hold_9_26 = 0
 tp_off_limit_constr_10_1: group_tp_active_7 = 0 -> hold_10_1 = 0
 tp_off_limit_constr_10_2: group_tp_active_7 = 0 -> hold_10_2 = 0
 tp_off_limit_constr_10_3: group_tp_active_7 = 0 -> hold_10_3 = 0
 tp_off_limit_constr_10_4: group_tp_active_7 = 0 -> hold_10_4 = 0
 tp_off_limit_constr_10_5: group_tp_active_7 = 0 -> hold_10_5 = 0
 tp_off_limit_constr_10_6: group_tp_active_7 = 0 -> hold_10_6 = 0
 tp_off_limit_constr_10_7: group_tp_active_7 = 0 -> hold_10_7 = 0
 tp_off_limit_constr_10_8: group_tp_active_7 = 0 -> hold_10_8 = 0
 tp_off_limit_constr_10_9: group_tp_active_7 = 0 -> hold_10_9 = 0
 tp_off_limit_constr_10_10: group_tp_active_7 = 0 -> hold_10_10 = 0
 tp_off_limit_constr_10_11: group_tp_active_7 = 0 -> hold_10_11 = 0
 tp_off_limit_constr_10_12: group_tp_active_7 = 0 -> hold_10_12 = 0
 tp_off_limit_constr_10_13: group_tp_active_7 = 0 -> hold_10_13 = 0
 tp_off_limit_constr_10_14: group_tp_active_7 = 0 -> hold_10_14 = 0
 tp_off_limit_constr_10_15: group_tp_active_7 = 0 -> hold_10_15 = 0
 tp_off_limit_constr_10_16: group_tp_active_7 = 0 -> hold_10_16 = 0
 tp_off_limit_constr_10_17: group_tp_active_7 = 0 -> hold_10_17 = 0
 tp_off_limit_constr_10_18: group_tp_active_7 = 0 -> hold_10_18 = 0
 tp_off_limit_constr_10_19: group_tp_active_7 = 0 -> hold_10_19 = 0
 tp_off_limit_constr_10_20: group_tp_active_7 = 0 -> hold_10_20 = 0
 tp_off_limit_constr_10_21: group_tp_active_7 = 0 -> hold_10_21 = 0
 tp_off_limit_constr_10_22: group_tp_active_7 = 0 -> hold_10_22 = 0
 tp_off_limit_constr_10_23: group_tp_active_7 = 0 -> hold_10_23 = 0
 tp_off_limit_constr_10_24: group_tp_active_7 = 0 -> hold_10_24 = 0
 tp_off_limit_constr_10_25: group_tp_active_7 = 0 -> hold_10_25 = 0
 tp_off_limit_constr_10_26: group_tp_active_7 = 0 -> hold_10_26 = 0
 tp_off_limit_constr_11_23: group_tp_active_11 = 0 -> hold_11_23 = 0
 tp_off_limit_constr_11_24: group_tp_active_11 = 0 -> hold_11_24 = 0
 tp_off_limit_constr_11_25: group_tp_active_11 = 0 -> hold_11_25 = 0
 tp_off_limit_constr_11_26: group_tp_active_11 = 0 -> hold_11_26 = 0
 group_synch_start_constr_12_13: group_tp_active_12 = 1 -> start_12
   - start_13 = 0
 group_synch_hold_constr_12_13_1: group_tp_active_12 = 1 -> hold_12_1
   - hold_13_1 = 0
 group_synch_hold_constr_12_13_2: group_tp_active_12 = 1 -> hold_12_2
   - hold_13_2 = 0
 group_synch_hold_constr_12_13_3: group_tp_active_12 = 1 -> hold_12_3
   - hold_13_3 = 0
 group_synch_hold_constr_12_13_4: group_tp_active_12 = 1 -> hold_12_4
   - hold_13_4 = 0
 group_synch_hold_constr_12_13_5: group_tp_active_12 = 1 -> hold_12_5
   - hold_13_5 = 0
 group_synch_hold_constr_12_13_6: group_tp_active_12 = 1 -> hold_12_6
   - hold_13_6 = 0
 group_synch_hold_constr_12_13_7: group_tp_active_12 = 1 -> hold_12_7
   - hold_13_7 = 0
 group_synch_hold_constr_12_13_8: group_tp_active_12 = 1 -> hold_12_8
   - hold_13_8 = 0
 group_synch_hold_constr_12_13_9: group_tp_active_12 = 1 -> hold_12_9
   - hold_13_9 = 0
 group_synch_hold_constr_12_13_10: group_tp_active_12 = 1 -> hold_12_10
   - hold_13_10 = 0
 group_synch_hold_constr_12_13_11: group_tp_active_12 = 1 -> hold_12_11
   - hold_13_11 = 0
 group_synch_hold_constr_12_13_12: group_tp_active_12 = 1 -> hold_12_12
   - hold_13_12 = 0
 group_synch_hold_constr_12_13_13: group_tp_active_12 = 1 -> hold_12_13
   - hold_13_13 = 0
 group_synch_hold_constr_12_13_14: group_tp_active_12 = 1 -> hold_12_14
   - hold_13_14 = 0
 group_synch_hold_constr_12_13_15: group_tp_active_12 = 1 -> hold_12_15
   - hold_13_15 = 0
 group_synch_hold_constr_12_13_16: group_tp_active_12 = 1 -> hold_12_16
   - hold_13_16 = 0
 group_synch_hold_constr_12_13_17: group_tp_active_12 = 1 -> hold_12_17
   - hold_13_17 = 0
 group_synch_hold_constr_12_13_18: group_tp_active_12 = 1 -> hold_12_18
   - hold_13_18 = 0
 group_synch_hold_constr_12_13_19: group_tp_active_12 = 1 -> hold_12_19
   - hold_13_19 = 0
 group_synch_hold_constr_12_13_20: group_tp_active_12 = 1 -> hold_12_20
   - hold_13_20 = 0
 group_synch_hold_constr_12_13_21: group_tp_active_12 = 1 -> hold_12_21
   - hold_13_21 = 0
 group_synch_hold_constr_12_13_22: group_tp_active_12 = 1 -> hold_12_22
   - hold_13_22 = 0
 group_synch_hold_constr_12_13_23: group_tp_active_12 = 1 -> hold_12_23
   - hold_13_23 = 0
 group_synch_hold_constr_12_13_24: group_tp_active_12 = 1 -> hold_12_24
   - hold_13_24 = 0
 group_synch_hold_constr_12_13_25: group_tp_active_12 = 1 -> hold_12_25
   - hold_13_25 = 0
 group_synch_hold_constr_12_13_26: group_tp_active_12 = 1 -> hold_12_26
   - hold_13_26 = 0
 tp_on_limit_constr_12_9: group_tp_active_12 = 1 -> hold_12_9 = 0
 tp_on_limit_constr_12_10: group_tp_active_12 = 1 -> hold_12_10 = 0
 tp_on_limit_constr_12_11: group_tp_active_12 = 1 -> hold_12_11 = 0
 tp_on_limit_constr_12_12: group_tp_active_12 = 1 -> hold_12_12 = 0
 tp_on_limit_constr_12_13: group_tp_active_12 = 1 -> hold_12_13 = 0
 tp_on_limit_constr_12_14: group_tp_active_12 = 1 -> hold_12_14 = 0
 tp_on_limit_constr_12_15: group_tp_active_12 = 1 -> hold_12_15 = 0
 tp_on_limit_constr_12_16: group_tp_active_12 = 1 -> hold_12_16 = 0
 tp_on_limit_constr_12_17: group_tp_active_12 = 1 -> hold_12_17 = 0
 tp_on_limit_constr_12_18: group_tp_active_12 = 1 -> hold_12_18 = 0
 tp_on_limit_constr_12_19: group_tp_active_12 = 1 -> hold_12_19 = 0
 tp_on_limit_constr_12_20: group_tp_active_12 = 1 -> hold_12_20 = 0
 tp_on_limit_constr_12_21: group_tp_active_12 = 1 -> hold_12_21 = 0
 tp_on_limit_constr_12_22: group_tp_active_12 = 1 -> hold_12_22 = 0
 tp_on_limit_constr_12_23: group_tp_active_12 = 1 -> hold_12_23 = 0
 tp_on_limit_constr_12_24: group_tp_active_12 = 1 -> hold_12_24 = 0
 tp_on_limit_constr_12_25: group_tp_active_12 = 1 -> hold_12_25 = 0
 tp_on_limit_constr_12_26: group_tp_active_12 = 1 -> hold_12_26 = 0
 tp_on_limit_constr_13_9: group_tp_active_12 = 1 -> hold_13_9 = 0
 tp_on_limit_constr_13_10: group_tp_active_12 = 1 -> hold_13_10 = 0
 tp_on_limit_constr_13_11: group_tp_active_12 = 1 -> hold_13_11 = 0
 tp_on_limit_constr_13_12: group_tp_active_12 = 1 -> hold_13_12 = 0
 tp_on_limit_constr_13_13: group_tp_active_12 = 1 -> hold_13_13 = 0
 tp_on_limit_constr_13_14: group_tp_active_12 = 1 -> hold_13_14 = 0
 tp_on_limit_constr_13_15: group_tp_active_12 = 1 -> hold_13_15 = 0
 tp_on_limit_constr_13_16: group_tp_active_12 = 1 -> hold_13_16 = 0
 tp_on_limit_constr_13_17: group_tp_active_12 = 1 -> hold_13_17 = 0
 tp_on_limit_constr_13_18: group_tp_active_12 = 1 -> hold_13_18 = 0
 tp_on_limit_constr_13_19: group_tp_active_12 = 1 -> hold_13_19 = 0
 tp_on_limit_constr_13_20: group_tp_active_12 = 1 -> hold_13_20 = 0
 tp_on_limit_constr_13_21: group_tp_active_12 = 1 -> hold_13_21 = 0
 tp_on_limit_constr_13_22: group_tp_active_12 = 1 -> hold_13_22 = 0
 tp_on_limit_constr_13_23: group_tp_active_12 = 1 -> hold_13_23 = 0
 tp_on_limit_constr_13_24: group_tp_active_12 = 1 -> hold_13_24 = 0
 tp_on_limit_constr_13_25: group_tp_active_12 = 1 -> hold_13_25 = 0
 tp_on_limit_constr_13_26: group_tp_active_12 = 1 -> hold_13_26 = 0
 tp_off_limit_constr_12_1: group_tp_active_12 = 0 -> hold_12_1 = 0
 tp_off_limit_constr_12_2: group_tp_active_12 = 0 -> hold_12_2 = 0
 tp_off_limit_constr_12_3: group_tp_active_12 = 0 -> hold_12_3 = 0
 tp_off_limit_constr_12_4: group_tp_active_12 = 0 -> hold_12_4 = 0
 tp_off_limit_constr_12_5: group_tp_active_12 = 0 -> hold_12_5 = 0
 tp_off_limit_constr_12_6: group_tp_active_12 = 0 -> hold_12_6 = 0
 tp_off_limit_constr_12_7: group_tp_active_12 = 0 -> hold_12_7 = 0
 tp_off_limit_constr_12_8: group_tp_active_12 = 0 -> hold_12_8 = 0
 tp_off_limit_constr_12_9: group_tp_active_12 = 0 -> hold_12_9 = 0
 tp_off_limit_constr_12_10: group_tp_active_12 = 0 -> hold_12_10 = 0
 tp_off_limit_constr_12_11: group_tp_active_12 = 0 -> hold_12_11 = 0
 tp_off_limit_constr_12_12: group_tp_active_12 = 0 -> hold_12_12 = 0
 tp_off_limit_constr_12_13: group_tp_active_12 = 0 -> hold_12_13 = 0
 tp_off_limit_constr_12_14: group_tp_active_12 = 0 -> hold_12_14 = 0
 tp_off_limit_constr_12_15: group_tp_active_12 = 0 -> hold_12_15 = 0
 tp_off_limit_constr_12_16: group_tp_active_12 = 0 -> hold_12_16 = 0
 tp_off_limit_constr_12_17: group_tp_active_12 = 0 -> hold_12_17 = 0
 tp_off_limit_constr_12_18: group_tp_active_12 = 0 -> hold_12_18 = 0
 tp_off_limit_constr_12_19: group_tp_active_12 = 0 -> hold_12_19 = 0
 tp_off_limit_constr_12_20: group_tp_active_12 = 0 -> hold_12_20 = 0
 tp_off_limit_constr_12_21: group_tp_active_12 = 0 -> hold_12_21 = 0
 tp_off_limit_constr_12_22: group_tp_active_12 = 0 -> hold_12_22 = 0
 tp_off_limit_constr_12_23: group_tp_active_12 = 0 -> hold_12_23 = 0
 tp_off_limit_constr_12_24: group_tp_active_12 = 0 -> hold_12_24 = 0
 tp_off_limit_constr_12_25: group_tp_active_12 = 0 -> hold_12_25 = 0
 tp_off_limit_constr_12_26: group_tp_active_12 = 0 -> hold_12_26 = 0
 tp_off_limit_constr_13_1: group_tp_active_12 = 0 -> hold_13_1 = 0
 tp_off_limit_constr_13_2: group_tp_active_12 = 0 -> hold_13_2 = 0
 tp_off_limit_constr_13_3: group_tp_active_12 = 0 -> hold_13_3 = 0
 tp_off_limit_constr_13_4: group_tp_active_12 = 0 -> hold_13_4 = 0
 tp_off_limit_constr_13_5: group_tp_active_12 = 0 -> hold_13_5 = 0
 tp_off_limit_constr_13_6: group_tp_active_12 = 0 -> hold_13_6 = 0
 tp_off_limit_constr_13_7: group_tp_active_12 = 0 -> hold_13_7 = 0
 tp_off_limit_constr_13_8: group_tp_active_12 = 0 -> hold_13_8 = 0
 tp_off_limit_constr_13_9: group_tp_active_12 = 0 -> hold_13_9 = 0
 tp_off_limit_constr_13_10: group_tp_active_12 = 0 -> hold_13_10 = 0
 tp_off_limit_constr_13_11: group_tp_active_12 = 0 -> hold_13_11 = 0
 tp_off_limit_constr_13_12: group_tp_active_12 = 0 -> hold_13_12 = 0
 tp_off_limit_constr_13_13: group_tp_active_12 = 0 -> hold_13_13 = 0
 tp_off_limit_constr_13_14: group_tp_active_12 = 0 -> hold_13_14 = 0
 tp_off_limit_constr_13_15: group_tp_active_12 = 0 -> hold_13_15 = 0
 tp_off_limit_constr_13_16: group_tp_active_12 = 0 -> hold_13_16 = 0
 tp_off_limit_constr_13_17: group_tp_active_12 = 0 -> hold_13_17 = 0
 tp_off_limit_constr_13_18: group_tp_active_12 = 0 -> hold_13_18 = 0
 tp_off_limit_constr_13_19: group_tp_active_12 = 0 -> hold_13_19 = 0
 tp_off_limit_constr_13_20: group_tp_active_12 = 0 -> hold_13_20 = 0
 tp_off_limit_constr_13_21: group_tp_active_12 = 0 -> hold_13_21 = 0
 tp_off_limit_constr_13_22: group_tp_active_12 = 0 -> hold_13_22 = 0
 tp_off_limit_constr_13_23: group_tp_active_12 = 0 -> hold_13_23 = 0
 tp_off_limit_constr_13_24: group_tp_active_12 = 0 -> hold_13_24 = 0
 tp_off_limit_constr_13_25: group_tp_active_12 = 0 -> hold_13_25 = 0
 tp_off_limit_constr_13_26: group_tp_active_12 = 0 -> hold_13_26 = 0
 tp_off_limit_constr_14_12: group_tp_active_14 = 0 -> hold_14_12 = 0
 tp_off_limit_constr_14_13: group_tp_active_14 = 0 -> hold_14_13 = 0
 tp_off_limit_constr_14_14: group_tp_active_14 = 0 -> hold_14_14 = 0
 tp_off_limit_constr_14_15: group_tp_active_14 = 0 -> hold_14_15 = 0
 tp_off_limit_constr_14_16: group_tp_active_14 = 0 -> hold_14_16 = 0
 tp_off_limit_constr_14_17: group_tp_active_14 = 0 -> hold_14_17 = 0
 tp_off_limit_constr_14_18: group_tp_active_14 = 0 -> hold_14_18 = 0
 tp_off_limit_constr_14_19: group_tp_active_14 = 0 -> hold_14_19 = 0
 tp_off_limit_constr_14_20: group_tp_active_14 = 0 -> hold_14_20 = 0
 tp_off_limit_constr_14_21: group_tp_active_14 = 0 -> hold_14_21 = 0
 tp_off_limit_constr_14_22: group_tp_active_14 = 0 -> hold_14_22 = 0
 tp_off_limit_constr_14_23: group_tp_active_14 = 0 -> hold_14_23 = 0
 tp_off_limit_constr_14_24: group_tp_active_14 = 0 -> hold_14_24 = 0
 tp_off_limit_constr_14_25: group_tp_active_14 = 0 -> hold_14_25 = 0
 tp_off_limit_constr_14_26: group_tp_active_14 = 0 -> hold_14_26 = 0
 tp_off_limit_constr_15_7: group_tp_active_15 = 0 -> hold_15_7 = 0
 tp_off_limit_constr_15_8: group_tp_active_15 = 0 -> hold_15_8 = 0
 tp_off_limit_constr_15_9: group_tp_active_15 = 0 -> hold_15_9 = 0
 tp_off_limit_constr_15_10: group_tp_active_15 = 0 -> hold_15_10 = 0
 tp_off_limit_constr_15_11: group_tp_active_15 = 0 -> hold_15_11 = 0
 tp_off_limit_constr_15_12: group_tp_active_15 = 0 -> hold_15_12 = 0
 tp_off_limit_constr_15_13: group_tp_active_15 = 0 -> hold_15_13 = 0
 tp_off_limit_constr_15_14: group_tp_active_15 = 0 -> hold_15_14 = 0
 tp_off_limit_constr_15_15: group_tp_active_15 = 0 -> hold_15_15 = 0
 tp_off_limit_constr_15_16: group_tp_active_15 = 0 -> hold_15_16 = 0
 tp_off_limit_constr_15_17: group_tp_active_15 = 0 -> hold_15_17 = 0
 tp_off_limit_constr_15_18: group_tp_active_15 = 0 -> hold_15_18 = 0
 tp_off_limit_constr_15_19: group_tp_active_15 = 0 -> hold_15_19 = 0
 tp_off_limit_constr_15_20: group_tp_active_15 = 0 -> hold_15_20 = 0
 tp_off_limit_constr_15_21: group_tp_active_15 = 0 -> hold_15_21 = 0
 tp_off_limit_constr_15_22: group_tp_active_15 = 0 -> hold_15_22 = 0
 tp_off_limit_constr_15_23: group_tp_active_15 = 0 -> hold_15_23 = 0
 tp_off_limit_constr_15_24: group_tp_active_15 = 0 -> hold_15_24 = 0
 tp_off_limit_constr_15_25: group_tp_active_15 = 0 -> hold_15_25 = 0
 tp_off_limit_constr_15_26: group_tp_active_15 = 0 -> hold_15_26 = 0
 group_throughput_constr_0: group_tp_active_0 = 1 ->
   - 7506.63764007438 hold_0_1 - 3753.31882003719 hold_0_2
   - 2502.21254669146 hold_0_3 - 1876.659410018595 hold_0_4
   - 1501.327528014876 hold_0_5 - 1251.10627334573 hold_0_6
   - 1072.376805724911 hold_0_7 - 938.3297050092975 hold_0_8
   - 834.0708488971533 hold_0_9 - 750.663764007438 hold_0_10
   - 682.4216036431255 hold_0_11 - 625.553136672865 hold_0_12
   - 577.4336646211061 hold_0_13 - 536.1884028624556 hold_0_14
   - 500.442509338292 hold_0_15 - 469.1648525046488 hold_0_16
   - 441.5669200043752 hold_0_17 - 417.0354244485766 hold_0_18
   - 395.0861915828621 hold_0_19 - 375.331882003719 hold_0_20
   - 357.4589352416371 hold_0_21 - 341.2108018215628 hold_0_22
   - 326.3755495684513 hold_0_23 - 312.7765683364325 hold_0_24
   - 300.2655056029752 hold_0_25 - 288.7168323105531 hold_0_26
   + flow_nic_in_0_0 + flow_nic_in_0_1 <= 0
 group_throughput_constr_2: group_tp_active_2 = 1 ->
   - 7506.63764007438 hold_2_1 - 3753.31882003719 hold_2_2
   - 2502.21254669146 hold_2_3 - 1876.659410018595 hold_2_4
   - 1501.327528014876 hold_2_5 - 1251.10627334573 hold_2_6
   - 1072.376805724911 hold_2_7 - 938.3297050092975 hold_2_8
   - 834.0708488971533 hold_2_9 - 750.663764007438 hold_2_10
   - 682.4216036431255 hold_2_11 - 625.553136672865 hold_2_12
   - 577.4336646211061 hold_2_13 - 536.1884028624556 hold_2_14
   - 500.442509338292 hold_2_15 - 469.1648525046488 hold_2_16
   - 441.5669200043752 hold_2_17 - 417.0354244485766 hold_2_18
   - 395.0861915828621 hold_2_19 - 375.331882003719 hold_2_20
   - 357.4589352416371 hold_2_21 - 341.2108018215628 hold_2_22
   - 326.3755495684513 hold_2_23 - 312.7765683364325 hold_2_24
   - 300.2655056029752 hold_2_25 - 288.7168323105531 hold_2_26
   + flow_nic_in_2_2 + flow_nic_in_2_3 <= 0
 group_throughput_constr_4: group_tp_active_4 = 1 ->
   - 1371.428571428571 hold_4_1 - 685.7142857142857 hold_4_2
   - 457.1428571428571 hold_4_3 - 342.8571428571428 hold_4_4
   - 274.2857142857143 hold_4_5 - 228.5714285714286 hold_4_6
   + flow_nic_in_4_4 <= 0
 group_throughput_constr_5: group_tp_active_5 = 1 ->
   - 2693.912296991259 hold_5_1 - 1346.95614849563 hold_5_2
   - 897.970765663753 hold_5_3 - 673.4780742478148 hold_5_4
   - 538.7824593982518 hold_5_5 - 448.9853828318765 hold_5_6
   - 384.8446138558941 hold_5_7 - 336.7390371239074 hold_5_8
   - 299.3235885545843 hold_5_9 - 269.3912296991259 hold_5_10
   - 244.9011179082963 hold_5_11 - 224.4926914159383 hold_5_12
   + flow_nic_in_5_5 + flow_nic_in_5_6 <= 0
 group_throughput_constr_7: group_tp_active_7 = 1 ->
   - 5223.90528711454 hold_7_1 - 2611.95264355727 hold_7_2
   - 1741.301762371513 hold_7_3 - 1305.976321778635 hold_7_4
   - 1044.781057422908 hold_7_5 - 870.6508811857567 hold_7_6
   - 746.2721838735059 hold_7_7 - 652.9881608893176 hold_7_8
   - 580.4339207905047 hold_7_9 - 522.3905287114541 hold_7_10
   - 474.9004806467765 hold_7_11 - 435.3254405928784 hold_7_12
   - 401.8388682395801 hold_7_13 - 373.136091936753 hold_7_14
   - 348.2603524743027 hold_7_15 - 326.4940804446588 hold_7_16
   - 307.2885463008554 hold_7_17 - 290.2169603952523 hold_7_18
   - 274.9423835323443 hold_7_19 - 261.195264355727 hold_7_20
   - 248.757394624502 hold_7_21 - 237.4502403233883 hold_7_22
   - 227.126316831067 hold_7_23 - 217.6627202964392 hold_7_24
   + flow_nic_in_7_7 + flow_nic_in_7_8 + flow_nic_in_7_9 + flow_nic_in_7_10
   <= 0
 group_throughput_constr_11: group_tp_active_11 = 1 ->
   - 9325.714285714286 hold_11_1 - 4662.857142857143 hold_11_2
   - 3108.571428571428 hold_11_3 - 2331.428571428572 hold_11_4
   - 1865.142857142857 hold_11_5 - 1554.285714285714 hold_11_6
   - 1332.244897959184 hold_11_7 - 1165.714285714286 hold_11_8
   - 1036.190476190476 hold_11_9 - 932.5714285714284 hold_11_10
   - 847.7922077922078 hold_11_11 - 777.1428571428571 hold_11_12
   - 717.3626373626373 hold_11_13 - 666.1224489795918 hold_11_14
   - 621.7142857142858 hold_11_15 - 582.8571428571429 hold_11_16
   - 548.5714285714286 hold_11_17 - 518.0952380952381 hold_11_18
   - 490.8270676691729 hold_11_19 - 466.2857142857142 hold_11_20
   - 444.0816326530613 hold_11_21 - 423.8961038961039 hold_11_22
   + flow_nic_in_11_11 <= 0
 group_throughput_constr_12: group_tp_active_12 = 1 ->
   - 3987.803303595952 hold_12_1 - 1993.901651797976 hold_12_2
   - 1329.267767865317 hold_12_3 - 996.950825898988 hold_12_4
   - 797.5606607191904 hold_12_5 - 664.6338839326586 hold_12_6
   - 569.6861862279932 hold_12_7 - 498.475412949494 hold_12_8
   + flow_nic_in_12_12 + flow_nic_in_12_13 <= 0
 group_throughput_constr_14: group_tp_active_14 = 1 ->
   - 7108.571428571429 hold_14_1 - 3554.285714285715 hold_14_2
   - 2369.52380952381 hold_14_3 - 1777.142857142857 hold_14_4
   - 1421.714285714286 hold_14_5 - 1184.761904761905 hold_14_6
   - 1015.510204081633 hold_14_7 - 888.5714285714287 hold_14_8
   - 789.8412698412698 hold_14_9 - 710.8571428571429 hold_14_10
   - 646.2337662337662 hold_14_11 + flow_nic_in_14_14 <= 0
 group_throughput_constr_15: group_tp_active_15 = 1 ->
   - 4265.142857142857 hold_15_1 - 2132.571428571428 hold_15_2
   - 1421.714285714286 hold_15_3 - 1066.285714285714 hold_15_4
   - 853.0285714285714 hold_15_5 - 710.8571428571429 hold_15_6
   + flow_nic_in_15_15 <= 0
 intergroup_group_tp_0_via_source_nic_in_0: tp_on_and_switch_source_nic_in_0_0 = 1 ->
   - 6862.690961524641 hold_0_1 - 3585.117698418173 hold_0_2
   - 2426.322853123827 hold_0_3 - 1833.645313368965 hold_0_4
   - 1473.671728842764 hold_0_5 - 1231.841712172382 hold_0_6
   - 1058.192048119833 hold_0_7 - 927.4515136936157 hold_0_8
   - 825.4646488917072 hold_0_9 - 743.6855416581999 hold_0_10
   - 676.6495998303666 hold_0_11 - 620.6996291733516 hold_0_12
   - 573.2956619338331 hold_0_13 - 532.6186038805861 hold_0_14
   - 497.3314369348856 hold_0_15 - 466.429448710796 hold_0_16
   - 439.1430320530621 hold_0_17 - 414.8727167047877 hold_0_18
   - 393.144616467096 hold_0_19 - 373.5791797958418 hold_0_20
   - 355.8688290825181 hold_0_21 - 339.7616724131586 hold_0_22
   - 325.0494471689925 hold_0_23 - 311.5584659885816 hold_0_24
   - 299.1427275736955 hold_0_25 - 287.6786115922728 hold_0_26
   + flow_nic_in_0_0 + flow_nic_in_0_1 <= 0
 intergroup_group_tp_2_via_nic_out_0_nic_in_2: tp_on_and_switch_nic_out_0_nic_in_2_2 = 1 ->
   - 5109.459042174185 hold_2_1 - 3040.15247603248 hold_2_2
   - 2163.816929787807 hold_2_3 - 1679.651095409346 hold_2_4
   - 1372.538192304928 hold_2_5 - 1160.371981735487 hold_2_6
   - 1005.016998327084 hold_2_7 - 886.3492175210624 hold_2_8
   - 792.7454804093763 hold_2_9 - 717.0235396836346 hold_2_10
   - 654.5060374151892 hold_2_11 - 602.0160715548276 hold_2_12
   - 557.3201993053406 hold_2_13 - 518.8024130345685 hold_2_14
   - 485.2645706247653 hold_2_15 - 455.7995324779789 hold_2_16
   - 429.7078711806942 hold_2_17 - 406.441639349837 hold_2_18
   - 385.5654670831283 hold_2_19 - 366.7290626737931 hold_2_20
   - 349.6473985665369 hold_2_21 - 334.0861883096475 hold_2_22
   - 319.8510763911254 hold_2_23 - 306.7794787980271 hold_2_24
   - 294.7343457685529 hold_2_25 - 283.5993383963553 hold_2_26
   + flow_nic_in_2_2 + flow_nic_in_2_3 <= 0
 intergroup_group_tp_5_via_nic_out_0_nic_in_5: tp_on_and_switch_nic_out_0_nic_in_5_5 = 1 ->
   - 1003.80710574974 hold_5_1 - 731.3070602333747 hold_5_2
   - 575.168150408772 hold_5_3 - 473.9719863597182 hold_5_4
   - 403.0573241561659 hold_5_5 - 350.6011407158714 hold_5_6
   - 310.2264922256207 hold_5_7 - 278.1905300976189 hold_5_8
   - 252.1517368463786 hold_5_9 - 230.5702309237719 hold_5_10
   - 212.3917563113273 hold_5_11 - 196.8702357402951 hold_5_12
   + flow_nic_in_5_5 + flow_nic_in_5_6 <= 0
 intergroup_group_tp_7_via_nic_out_0_nic_in_7: tp_on_and_switch_nic_out_0_nic_in_7_7 = 1 ->
   - 3938.129362298712 hold_7_1 - 2245.398056682829 hold_7_2
   - 1570.393681992138 hold_7_3 - 1207.422266154505 hold_7_4
   - 980.7399028740582 hold_7_5 - 825.7188295270446 hold_7_6
   - 713.0156975159247 hold_7_7 - 627.3835346119131 hold_7_8
   - 560.1145770378789 hold_7_9 - 505.8740407363503 hold_7_10
   - 461.2111435376709 hold_7_11 - 423.7948968313702 hold_7_12
   - 391.9939674741698 hold_7_13 - 364.6324954159618 hold_7_14
   - 340.8415035882089 hold_7_15 - 319.9649147805446 hold_7_16
   - 301.498113978548 hold_7_17 - 285.0466250764638 hold_7_18
   - 270.2976165967061 hold_7_19 - 256.999818386795 hold_7_20
   - 244.9490885566889 hold_7_21 - 233.977859143145 hold_7_22
   - 223.9472965418282 hold_7_23 - 214.7413955270226 hold_7_24
   + flow_nic_in_7_7 + flow_nic_in_7_8 + flow_nic_in_7_9 + flow_nic_in_7_10
   <= 0
 intergroup_group_tp_12_via_nic_out_0_nic_in_12: tp_on_and_switch_nic_out_0_nic_in_12_12 = 1 ->
   - 1141.859320933407 hold_12_1 - 887.6822328403811 hold_12_2
   - 726.0614587427827 hold_12_3 - 614.2285427704226 hold_12_4
   - 532.2480794992344 hold_12_5 - 469.5744516749779 hold_12_6
   - 420.1058677289324 hold_12_7 - 380.0667169114867 hold_12_8
   + flow_nic_in_12_12 + flow_nic_in_12_13 <= 0
 intergroup_group_tp_2_via_source_nic_in_2: tp_on_and_switch_source_nic_in_2_2 = 1 ->
   - 6862.690961524641 hold_2_1 - 3585.117698418173 hold_2_2
   - 2426.322853123827 hold_2_3 - 1833.645313368965 hold_2_4
   - 1473.671728842764 hold_2_5 - 1231.841712172382 hold_2_6
   - 1058.192048119833 hold_2_7 - 927.4515136936157 hold_2_8
   - 825.4646488917072 hold_2_9 - 743.6855416581999 hold_2_10
   - 676.6495998303666 hold_2_11 - 620.6996291733516 hold_2_12
   - 573.2956619338331 hold_2_13 - 532.6186038805861 hold_2_14
   - 497.3314369348856 hold_2_15 - 466.429448710796 hold_2_16
   - 439.1430320530621 hold_2_17 - 414.8727167047877 hold_2_18
   - 393.144616467096 hold_2_19 - 373.5791797958418 hold_2_20
   - 355.8688290825181 hold_2_21 - 339.7616724131586 hold_2_22
   - 325.0494471689925 hold_2_23 - 311.5584659885816 hold_2_24
   - 299.1427275736955 hold_2_25 - 287.6786115922728 hold_2_26
   + flow_nic_in_2_2 + flow_nic_in_2_3 <= 0
 intergroup_group_tp_0_via_nic_out_2_nic_in_0: tp_on_and_switch_nic_out_2_nic_in_0_0 = 1 ->
   - 5109.459042174185 hold_0_1 - 3040.15247603248 hold_0_2
   - 2163.816929787807 hold_0_3 - 1679.651095409346 hold_0_4
   - 1372.538192304928 hold_0_5 - 1160.371981735487 hold_0_6
   - 1005.016998327084 hold_0_7 - 886.3492175210624 hold_0_8
   - 792.7454804093763 hold_0_9 - 717.0235396836346 hold_0_10
   - 654.5060374151892 hold_0_11 - 602.0160715548276 hold_0_12
   - 557.3201993053406 hold_0_13 - 518.8024130345685 hold_0_14
   - 485.2645706247653 hold_0_15 - 455.7995324779789 hold_0_16
   - 429.7078711806942 hold_0_17 - 406.441639349837 hold_0_18
   - 385.5654670831283 hold_0_19 - 366.7290626737931 hold_0_20
   - 349.6473985665369 hold_0_21 - 334.0861883096475 hold_0_22
   - 319.8510763911254 hold_0_23 - 306.7794787980271 hold_0_24
   - 294.7343457685529 hold_0_25 - 283.5993383963553 hold_0_26
   + flow_nic_in_0_0 + flow_nic_in_0_1 <= 0
 intergroup_group_tp_5_via_nic_out_2_nic_in_5: tp_on_and_switch_nic_out_2_nic_in_5_5 = 1 ->
   - 1003.80710574974 hold_5_1 - 731.3070602333747 hold_5_2
   - 575.168150408772 hold_5_3 - 473.9719863597182 hold_5_4
   - 403.0573241561659 hold_5_5 - 350.6011407158714 hold_5_6
   - 310.2264922256207 hold_5_7 - 278.1905300976189 hold_5_8
   - 252.1517368463786 hold_5_9 - 230.5702309237719 hold_5_10
   - 212.3917563113273 hold_5_11 - 196.8702357402951 hold_5_12
   + flow_nic_in_5_5 + flow_nic_in_5_6 <= 0
 intergroup_group_tp_7_via_nic_out_2_nic_in_7: tp_on_and_switch_nic_out_2_nic_in_7_7 = 1 ->
   - 3938.129362298712 hold_7_1 - 2245.398056682829 hold_7_2
   - 1570.393681992138 hold_7_3 - 1207.422266154505 hold_7_4
   - 980.7399028740582 hold_7_5 - 825.7188295270446 hold_7_6
   - 713.0156975159247 hold_7_7 - 627.3835346119131 hold_7_8
   - 560.1145770378789 hold_7_9 - 505.8740407363503 hold_7_10
   - 461.2111435376709 hold_7_11 - 423.7948968313702 hold_7_12
   - 391.9939674741698 hold_7_13 - 364.6324954159618 hold_7_14
   - 340.8415035882089 hold_7_15 - 319.9649147805446 hold_7_16
   - 301.498113978548 hold_7_17 - 285.0466250764638 hold_7_18
   - 270.2976165967061 hold_7_19 - 256.999818386795 hold_7_20
   - 244.9490885566889 hold_7_21 - 233.977859143145 hold_7_22
   - 223.9472965418282 hold_7_23 - 214.7413955270226 hold_7_24
   + flow_nic_in_7_7 + flow_nic_in_7_8 + flow_nic_in_7_9 + flow_nic_in_7_10
   <= 0
 intergroup_group_tp_12_via_nic_out_2_nic_in_12: tp_on_and_switch_nic_out_2_nic_in_12_12 = 1 ->
   - 1141.859320933407 hold_12_1 - 887.6822328403811 hold_12_2
   - 726.0614587427827 hold_12_3 - 614.2285427704226 hold_12_4
   - 532.2480794992344 hold_12_5 - 469.5744516749779 hold_12_6
   - 420.1058677289324 hold_12_7 - 380.0667169114867 hold_12_8
   + flow_nic_in_12_12 + flow_nic_in_12_13 <= 0
 intergroup_group_tp_0_via_nic_out_4_nic_in_0: tp_on_and_switch_nic_out_4_nic_in_0_0 = 1 ->
   - 5109.459042174185 hold_0_1 - 3040.15247603248 hold_0_2
   - 2163.816929787807 hold_0_3 - 1679.651095409346 hold_0_4
   - 1372.538192304928 hold_0_5 - 1160.371981735487 hold_0_6
   - 1005.016998327084 hold_0_7 - 886.3492175210624 hold_0_8
   - 792.7454804093763 hold_0_9 - 717.0235396836346 hold_0_10
   - 654.5060374151892 hold_0_11 - 602.0160715548276 hold_0_12
   - 557.3201993053406 hold_0_13 - 518.8024130345685 hold_0_14
   - 485.2645706247653 hold_0_15 - 455.7995324779789 hold_0_16
   - 429.7078711806942 hold_0_17 - 406.441639349837 hold_0_18
   - 385.5654670831283 hold_0_19 - 366.7290626737931 hold_0_20
   - 349.6473985665369 hold_0_21 - 334.0861883096475 hold_0_22
   - 319.8510763911254 hold_0_23 - 306.7794787980271 hold_0_24
   - 294.7343457685529 hold_0_25 - 283.5993383963553 hold_0_26
   + flow_nic_in_0_0 + flow_nic_in_0_1 <= 0
 intergroup_group_tp_2_via_nic_out_4_nic_in_2: tp_on_and_switch_nic_out_4_nic_in_2_2 = 1 ->
   - 5109.459042174185 hold_2_1 - 3040.15247603248 hold_2_2
   - 2163.816929787807 hold_2_3 - 1679.651095409346 hold_2_4
   - 1372.538192304928 hold_2_5 - 1160.371981735487 hold_2_6
   - 1005.016998327084 hold_2_7 - 886.3492175210624 hold_2_8
   - 792.7454804093763 hold_2_9 - 717.0235396836346 hold_2_10
   - 654.5060374151892 hold_2_11 - 602.0160715548276 hold_2_12
   - 557.3201993053406 hold_2_13 - 518.8024130345685 hold_2_14
   - 485.2645706247653 hold_2_15 - 455.7995324779789 hold_2_16
   - 429.7078711806942 hold_2_17 - 406.441639349837 hold_2_18
   - 385.5654670831283 hold_2_19 - 366.7290626737931 hold_2_20
   - 349.6473985665369 hold_2_21 - 334.0861883096475 hold_2_22
   - 319.8510763911254 hold_2_23 - 306.7794787980271 hold_2_24
   - 294.7343457685529 hold_2_25 - 283.5993383963553 hold_2_26
   + flow_nic_in_2_2 + flow_nic_in_2_3 <= 0
 intergroup_group_tp_5_via_nic_out_4_nic_in_5: tp_on_and_switch_nic_out_4_nic_in_5_5 = 1 ->
   - 1003.80710574974 hold_5_1 - 731.3070602333747 hold_5_2
   - 575.168150408772 hold_5_3 - 473.9719863597182 hold_5_4
   - 403.0573241561659 hold_5_5 - 350.6011407158714 hold_5_6
   - 310.2264922256207 hold_5_7 - 278.1905300976189 hold_5_8
   - 252.1517368463786 hold_5_9 - 230.5702309237719 hold_5_10
   - 212.3917563113273 hold_5_11 - 196.8702357402951 hold_5_12
   + flow_nic_in_5_5 + flow_nic_in_5_6 <= 0
 intergroup_group_tp_7_via_nic_out_4_nic_in_7: tp_on_and_switch_nic_out_4_nic_in_7_7 = 1 ->
   - 3938.129362298712 hold_7_1 - 2245.398056682829 hold_7_2
   - 1570.393681992138 hold_7_3 - 1207.422266154505 hold_7_4
   - 980.7399028740582 hold_7_5 - 825.7188295270446 hold_7_6
   - 713.0156975159247 hold_7_7 - 627.3835346119131 hold_7_8
   - 560.1145770378789 hold_7_9 - 505.8740407363503 hold_7_10
   - 461.2111435376709 hold_7_11 - 423.7948968313702 hold_7_12
   - 391.9939674741698 hold_7_13 - 364.6324954159618 hold_7_14
   - 340.8415035882089 hold_7_15 - 319.9649147805446 hold_7_16
   - 301.498113978548 hold_7_17 - 285.0466250764638 hold_7_18
   - 270.2976165967061 hold_7_19 - 256.999818386795 hold_7_20
   - 244.9490885566889 hold_7_21 - 233.977859143145 hold_7_22
   - 223.9472965418282 hold_7_23 - 214.7413955270226 hold_7_24
   + flow_nic_in_7_7 + flow_nic_in_7_8 + flow_nic_in_7_9 + flow_nic_in_7_10
   <= 0
 intergroup_group_tp_12_via_nic_out_4_nic_in_12: tp_on_and_switch_nic_out_4_nic_in_12_12 = 1 ->
   - 1141.859320933407 hold_12_1 - 887.6822328403811 hold_12_2
   - 726.0614587427827 hold_12_3 - 614.2285427704226 hold_12_4
   - 532.2480794992344 hold_12_5 - 469.5744516749779 hold_12_6
   - 420.1058677289324 hold_12_7 - 380.0667169114867 hold_12_8
   + flow_nic_in_12_12 + flow_nic_in_12_13 <= 0
 intergroup_group_tp_5_via_source_nic_in_5: tp_on_and_switch_source_nic_in_5_5 = 1 ->
   - 2606.152953379399 hold_5_1 - 1324.653029216547 hold_5_2
   - 888.0032536439213 hold_5_3 - 667.8557467206057 hold_5_4
   - 535.1781518871273 hold_5_5 - 446.4795976683049 hold_5_6
   - 383.0021598768469 hold_5_7 - 335.3275636127566 hold_5_8
   - 298.2078305797816 hold_5_9 - 268.4871287270489 hold_5_10
   - 244.1536989855088 hold_5_11 - 223.8644921365358 hold_5_12
   + flow_nic_in_5_5 + flow_nic_in_5_6 <= 0
 intergroup_group_tp_0_via_nic_out_5_nic_in_0: tp_on_and_switch_nic_out_5_nic_in_0_0 = 1 ->
   - 1318.886366057375 hold_0_1 - 1121.791978759409 hold_0_2
   - 975.9465237692996 hold_0_3 - 863.6609750662036 hold_0_4
   - 774.5470361079126 hold_0_5 - 702.102919160611 hold_0_6
   - 642.0512576984546 hold_0_7 - 591.4627737492349 hold_0_8
   - 548.2639746661837 hold_0_9 - 510.9459042174184 hold_0_10
   - 478.3842582309013 hold_0_11 - 449.7241616854303 hold_0_12
   - 424.3040228527632 hold_0_13 - 401.6038301819989 hold_0_14
   - 381.2092030043309 hold_0_15 - 362.7858665290911 hold_0_16
   - 346.0611871618131 hold_0_17 - 330.8105901512063 hold_0_18
   - 316.847417018637 hold_0_19 - 304.0152476032479 hold_0_20
   - 292.1820152084147 hold_0_21 - 281.2354446009739 hold_0_22
   - 271.0794784674806 hold_0_23 - 261.6314511702397 hold_0_24
   - 252.8198336223107 hold_0_25 - 244.5824190234829 hold_0_26
   + flow_nic_in_0_0 + flow_nic_in_0_1 <= 0
 intergroup_group_tp_2_via_nic_out_5_nic_in_2: tp_on_and_switch_nic_out_5_nic_in_2_2 = 1 ->
   - 1318.886366057375 hold_2_1 - 1121.791978759409 hold_2_2
   - 975.9465237692996 hold_2_3 - 863.6609750662036 hold_2_4
   - 774.5470361079126 hold_2_5 - 702.102919160611 hold_2_6
   - 642.0512576984546 hold_2_7 - 591.4627737492349 hold_2_8
   - 548.2639746661837 hold_2_9 - 510.9459042174184 hold_2_10
   - 478.3842582309013 hold_2_11 - 449.7241616854303 hold_2_12
   - 424.3040228527632 hold_2_13 - 401.6038301819989 hold_2_14
   - 381.2092030043309 hold_2_15 - 362.7858665290911 hold_2_16
   - 346.0611871618131 hold_2_17 - 330.8105901512063 hold_2_18
   - 316.847417018637 hold_2_19 - 304.0152476032479 hold_2_20
   - 292.1820152084147 hold_2_21 - 281.2354446009739 hold_2_22
   - 271.0794784674806 hold_2_23 - 261.6314511702397 hold_2_24
   - 252.8198336223107 hold_2_25 - 244.5824190234829 hold_2_26
   + flow_nic_in_2_2 + flow_nic_in_2_3 <= 0
 intergroup_group_tp_7_via_nic_out_5_nic_in_7: tp_on_and_switch_nic_out_5_nic_in_7_7 = 1 ->
   - 1224.848251508707 hold_7_1 - 992.2058919831748 hold_7_2
   - 833.8315476830738 hold_7_3 - 719.0568275404518 hold_7_4
   - 632.0559833052946 hold_7_5 - 563.8357974837135 hold_7_6
   - 508.907492661978 hold_7_7 - 463.7312683483005 hold_7_8
   - 425.9217692449557 hold_7_9 - 393.8129362298712 hold_7_10
   - 366.2058860760343 hold_7_11 - 342.2158889468374 hold_7_12
   - 321.1757946076511 hold_7_13 - 302.5730204513139 hold_7_14
   - 286.0072388432151 hold_7_15 - 271.1612425982019 hold_7_16
   - 257.7804365442951 hold_7_17 - 245.6581156352056 hold_7_18
   - 234.624710346019 hold_7_19 - 224.5398056682828 hold_7_20
   - 215.286133570837 hold_7_21 - 206.7649921505117 hold_7_22
   - 198.8927112384791 hold_7_23 - 191.5978957952692 hold_7_24
   + flow_nic_in_7_7 + flow_nic_in_7_8 + flow_nic_in_7_9 + flow_nic_in_7_10
   <= 0
 intergroup_group_tp_12_via_nic_out_5_nic_in_12: tp_on_and_switch_nic_out_5_nic_in_12_12 = 1 ->
   - 3192.189351095739 hold_12_1 - 1772.957696786115 hold_12_2
   - 1227.304267597741 hold_12_3 - 938.4749875300135 hold_12_4
   - 759.6918879625396 hold_12_5 - 638.1263589100226 hold_12_6
   - 550.0996746229188 hold_12_7 - 483.4147645504219 hold_12_8
   + flow_nic_in_12_12 + flow_nic_in_12_13 <= 0
 intergroup_group_tp_7_via_source_nic_in_7: tp_on_and_switch_source_nic_in_7_7 = 1 ->
   - 4903.699514370291 hold_7_1 - 2529.370203681751 hold_7_2
   - 1704.207517941044 hold_7_3 - 1284.999091933975 hold_7_4
   - 1031.3123621694 hold_7_5 - 861.2774812112317 hold_7_6
   - 739.3749964571616 hold_7_7 - 647.7013941124807 hold_7_8
   - 576.2529612199046 hold_7_9 - 519.001510294395 hold_7_10
   - 472.0979861401474 hold_7_11 - 432.9694081135003 hold_7_12
   - 399.8305251680654 hold_7_13 - 371.4037899528342 hold_7_14
   - 346.7508577749344 hold_7_15 - 325.1670166186359 hold_7_16
   - 306.1127345856677 hold_7_17 - 289.1679423792392 hold_7_18
   - 274.0007034511393 hold_7_19 - 260.3452525175385 hold_7_20
   - 247.9862893340353 hold_7_21 - 236.747543310201 hold_7_22
   - 226.4833128227529 hold_7_23 - 217.0721139611232 hold_7_24
   + flow_nic_in_7_7 + flow_nic_in_7_8 + flow_nic_in_7_9 + flow_nic_in_7_10
   <= 0
 intergroup_group_tp_0_via_nic_out_7_nic_in_0: tp_on_and_switch_nic_out_7_nic_in_0_0 = 1 ->
   - 5109.459042174185 hold_0_1 - 3040.15247603248 hold_0_2
   - 2163.816929787807 hold_0_3 - 1679.651095409346 hold_0_4
   - 1372.538192304928 hold_0_5 - 1160.371981735487 hold_0_6
   - 1005.016998327084 hold_0_7 - 886.3492175210624 hold_0_8
   - 792.7454804093763 hold_0_9 - 717.0235396836346 hold_0_10
   - 654.5060374151892 hold_0_11 - 602.0160715548276 hold_0_12
   - 557.3201993053406 hold_0_13 - 518.8024130345685 hold_0_14
   - 485.2645706247653 hold_0_15 - 455.7995324779789 hold_0_16
   - 429.7078711806942 hold_0_17 - 406.441639349837 hold_0_18
   - 385.5654670831283 hold_0_19 - 366.7290626737931 hold_0_20
   - 349.6473985665369 hold_0_21 - 334.0861883096475 hold_0_22
   - 319.8510763911254 hold_0_23 - 306.7794787980271 hold_0_24
   - 294.7343457685529 hold_0_25 - 283.5993383963553 hold_0_26
   + flow_nic_in_0_0 + flow_nic_in_0_1 <= 0
 intergroup_group_tp_2_via_nic_out_7_nic_in_2: tp_on_and_switch_nic_out_7_nic_in_2_2 = 1 ->
   - 5109.459042174185 hold_2_1 - 3040.15247603248 hold_2_2
   - 2163.816929787807 hold_2_3 - 1679.651095409346 hold_2_4
   - 1372.538192304928 hold_2_5 - 1160.371981735487 hold_2_6
   - 1005.016998327084 hold_2_7 - 886.3492175210624 hold_2_8
   - 792.7454804093763 hold_2_9 - 717.0235396836346 hold_2_10
   - 654.5060374151892 hold_2_11 - 602.0160715548276 hold_2_12
   - 557.3201993053406 hold_2_13 - 518.8024130345685 hold_2_14
   - 485.2645706247653 hold_2_15 - 455.7995324779789 hold_2_16
   - 429.7078711806942 hold_2_17 - 406.441639349837 hold_2_18
   - 385.5654670831283 hold_2_19 - 366.7290626737931 hold_2_20
   - 349.6473985665369 hold_2_21 - 334.0861883096475 hold_2_22
   - 319.8510763911254 hold_2_23 - 306.7794787980271 hold_2_24
   - 294.7343457685529 hold_2_25 - 283.5993383963553 hold_2_26
   + flow_nic_in_2_2 + flow_nic_in_2_3 <= 0
 intergroup_group_tp_5_via_nic_out_7_nic_in_5: tp_on_and_switch_nic_out_7_nic_in_5_5 = 1 ->
   - 1003.80710574974 hold_5_1 - 731.3070602333747 hold_5_2
   - 575.168150408772 hold_5_3 - 473.9719863597182 hold_5_4
   - 403.0573241561659 hold_5_5 - 350.6011407158714 hold_5_6
   - 310.2264922256207 hold_5_7 - 278.1905300976189 hold_5_8
   - 252.1517368463786 hold_5_9 - 230.5702309237719 hold_5_10
   - 212.3917563113273 hold_5_11 - 196.8702357402951 hold_5_12
   + flow_nic_in_5_5 + flow_nic_in_5_6 <= 0
 intergroup_group_tp_12_via_nic_out_7_nic_in_12: tp_on_and_switch_nic_out_7_nic_in_12_12 = 1 ->
   - 1141.859320933407 hold_12_1 - 887.6822328403811 hold_12_2
   - 726.0614587427827 hold_12_3 - 614.2285427704226 hold_12_4
   - 532.2480794992344 hold_12_5 - 469.5744516749779 hold_12_6
   - 420.1058677289324 hold_12_7 - 380.0667169114867 hold_12_8
   + flow_nic_in_12_12 + flow_nic_in_12_13 <= 0
 intergroup_group_tp_0_via_nic_out_11_nic_in_0: tp_on_and_switch_nic_out_11_nic_in_0_0 = 1 ->
   - 1318.886366057375 hold_0_1 - 1121.791978759409 hold_0_2
   - 975.9465237692996 hold_0_3 - 863.6609750662036 hold_0_4
   - 774.5470361079126 hold_0_5 - 702.102919160611 hold_0_6
   - 642.0512576984546 hold_0_7 - 591.4627737492349 hold_0_8
   - 548.2639746661837 hold_0_9 - 510.9459042174184 hold_0_10
   - 478.3842582309013 hold_0_11 - 449.7241616854303 hold_0_12
   - 424.3040228527632 hold_0_13 - 401.6038301819989 hold_0_14
   - 381.2092030043309 hold_0_15 - 362.7858665290911 hold_0_16
   - 346.0611871618131 hold_0_17 - 330.8105901512063 hold_0_18
   - 316.847417018637 hold_0_19 - 304.0152476032479 hold_0_20
   - 292.1820152084147 hold_0_21 - 281.2354446009739 hold_0_22
   - 271.0794784674806 hold_0_23 - 261.6314511702397 hold_0_24
   - 252.8198336223107 hold_0_25 - 244.5824190234829 hold_0_26
   + flow_nic_in_0_0 + flow_nic_in_0_1 <= 0
 intergroup_group_tp_2_via_nic_out_11_nic_in_2: tp_on_and_switch_nic_out_11_nic_in_2_2 = 1 ->
   - 1318.886366057375 hold_2_1 - 1121.791978759409 hold_2_2
   - 975.9465237692996 hold_2_3 - 863.6609750662036 hold_2_4
   - 774.5470361079126 hold_2_5 - 702.102919160611 hold_2_6
   - 642.0512576984546 hold_2_7 - 591.4627737492349 hold_2_8
   - 548.2639746661837 hold_2_9 - 510.9459042174184 hold_2_10
   - 478.3842582309013 hold_2_11 - 449.7241616854303 hold_2_12
   - 424.3040228527632 hold_2_13 - 401.6038301819989 hold_2_14
   - 381.2092030043309 hold_2_15 - 362.7858665290911 hold_2_16
   - 346.0611871618131 hold_2_17 - 330.8105901512063 hold_2_18
   - 316.847417018637 hold_2_19 - 304.0152476032479 hold_2_20
   - 292.1820152084147 hold_2_21 - 281.2354446009739 hold_2_22
   - 271.0794784674806 hold_2_23 - 261.6314511702397 hold_2_24
   - 252.8198336223107 hold_2_25 - 244.5824190234829 hold_2_26
   + flow_nic_in_2_2 + flow_nic_in_2_3 <= 0
 intergroup_group_tp_5_via_nic_out_11_nic_in_5: tp_on_and_switch_nic_out_11_nic_in_5_5 = 1 ->
   - 2305.702309237719 hold_5_1 - 1242.367720967523 hold_5_2
   - 850.2519296467542 hold_5_3 - 646.2748288017978 hold_5_4
   - 521.2305906758797 hold_5_5 - 436.730044930787 hold_5_6
   - 375.8054450200391 hold_5_7 - 329.7980448692438 hold_5_8
   - 293.8267585678413 hold_5_9 - 264.9306058433094 hold_5_10
   - 241.2090943547263 hold_5_11 - 221.3864637231713 hold_5_12
   + flow_nic_in_5_5 + flow_nic_in_5_6 <= 0
 intergroup_group_tp_7_via_nic_out_11_nic_in_7: tp_on_and_switch_nic_out_11_nic_in_7_7 = 1 ->
   - 1224.848251508707 hold_7_1 - 992.2058919831748 hold_7_2
   - 833.8315476830738 hold_7_3 - 719.0568275404518 hold_7_4
   - 632.0559833052946 hold_7_5 - 563.8357974837135 hold_7_6
   - 508.907492661978 hold_7_7 - 463.7312683483005 hold_7_8
   - 425.9217692449557 hold_7_9 - 393.8129362298712 hold_7_10
   - 366.2058860760343 hold_7_11 - 342.2158889468374 hold_7_12
   - 321.1757946076511 hold_7_13 - 302.5730204513139 hold_7_14
   - 286.0072388432151 hold_7_15 - 271.1612425982019 hold_7_16
   - 257.7804365442951 hold_7_17 - 245.6581156352056 hold_7_18
   - 234.624710346019 hold_7_19 - 224.5398056682828 hold_7_20
   - 215.286133570837 hold_7_21 - 206.7649921505117 hold_7_22
   - 198.8927112384791 hold_7_23 - 191.5978957952692 hold_7_24
   + flow_nic_in_7_7 + flow_nic_in_7_8 + flow_nic_in_7_9 + flow_nic_in_7_10
   <= 0
 intergroup_group_tp_12_via_nic_out_11_nic_in_12: tp_on_and_switch_nic_out_11_nic_in_12_12 = 1 ->
   - 3192.189351095739 hold_12_1 - 1772.957696786115 hold_12_2
   - 1227.304267597741 hold_12_3 - 938.4749875300135 hold_12_4
   - 759.6918879625396 hold_12_5 - 638.1263589100226 hold_12_6
   - 550.0996746229188 hold_12_7 - 483.4147645504219 hold_12_8
   + flow_nic_in_12_12 + flow_nic_in_12_13 <= 0
 intergroup_group_tp_12_via_source_nic_in_12: tp_on_and_switch_source_nic_in_12_12 = 1 ->
   - 3798.459439812698 hold_12_1 - 1945.414584870413 hold_12_2
   - 1307.54185237166 hold_12_3 - 984.67985842273 hold_12_4
   - 789.6878610662662 hold_12_5 - 659.1576525484435 hold_12_6
   - 565.6580912192966 hold_12_7 - 495.3886745232008 hold_12_8
   + flow_nic_in_12_12 + flow_nic_in_12_13 <= 0
 intergroup_group_tp_0_via_nic_out_12_nic_in_0: tp_on_and_switch_nic_out_12_nic_in_0_0 = 1 ->
   - 1318.886366057375 hold_0_1 - 1121.791978759409 hold_0_2
   - 975.9465237692996 hold_0_3 - 863.6609750662036 hold_0_4
   - 774.5470361079126 hold_0_5 - 702.102919160611 hold_0_6
   - 642.0512576984546 hold_0_7 - 591.4627737492349 hold_0_8
   - 548.2639746661837 hold_0_9 - 510.9459042174184 hold_0_10
   - 478.3842582309013 hold_0_11 - 449.7241616854303 hold_0_12
   - 424.3040228527632 hold_0_13 - 401.6038301819989 hold_0_14
   - 381.2092030043309 hold_0_15 - 362.7858665290911 hold_0_16
   - 346.0611871618131 hold_0_17 - 330.8105901512063 hold_0_18
   - 316.847417018637 hold_0_19 - 304.0152476032479 hold_0_20
   - 292.1820152084147 hold_0_21 - 281.2354446009739 hold_0_22
   - 271.0794784674806 hold_0_23 - 261.6314511702397 hold_0_24
   - 252.8198336223107 hold_0_25 - 244.5824190234829 hold_0_26
   + flow_nic_in_0_0 + flow_nic_in_0_1 <= 0
 intergroup_group_tp_2_via_nic_out_12_nic_in_2: tp_on_and_switch_nic_out_12_nic_in_2_2 = 1 ->
   - 1318.886366057375 hold_2_1 - 1121.791978759409 hold_2_2
   - 975.9465237692996 hold_2_3 - 863.6609750662036 hold_2_4
   - 774.5470361079126 hold_2_5 - 702.102919160611 hold_2_6
   - 642.0512576984546 hold_2_7 - 591.4627737492349 hold_2_8
   - 548.2639746661837 hold_2_9 - 510.9459042174184 hold_2_10
   - 478.3842582309013 hold_2_11 - 449.7241616854303 hold_2_12
   - 424.3040228527632 hold_2_13 - 401.6038301819989 hold_2_14
   - 381.2092030043309 hold_2_15 - 362.7858665290911 hold_2_16
   - 346.0611871618131 hold_2_17 - 330.8105901512063 hold_2_18
   - 316.847417018637 hold_2_19 - 304.0152476032479 hold_2_20
   - 292.1820152084147 hold_2_21 - 281.2354446009739 hold_2_22
   - 271.0794784674806 hold_2_23 - 261.6314511702397 hold_2_24
   - 252.8198336223107 hold_2_25 - 244.5824190234829 hold_2_26
   + flow_nic_in_2_2 + flow_nic_in_2_3 <= 0
 intergroup_group_tp_5_via_nic_out_12_nic_in_5: tp_on_and_switch_nic_out_12_nic_in_5_5 = 1 ->
   - 2305.702309237719 hold_5_1 - 1242.367720967523 hold_5_2
   - 850.2519296467542 hold_5_3 - 646.2748288017978 hold_5_4
   - 521.2305906758797 hold_5_5 - 436.730044930787 hold_5_6
   - 375.8054450200391 hold_5_7 - 329.7980448692438 hold_5_8
   - 293.8267585678413 hold_5_9 - 264.9306058433094 hold_5_10
   - 241.2090943547263 hold_5_11 - 221.3864637231713 hold_5_12
   + flow_nic_in_5_5 + flow_nic_in_5_6 <= 0
 intergroup_group_tp_7_via_nic_out_12_nic_in_7: tp_on_and_switch_nic_out_12_nic_in_7_7 = 1 ->
   - 1224.848251508707 hold_7_1 - 992.2058919831748 hold_7_2
   - 833.8315476830738 hold_7_3 - 719.0568275404518 hold_7_4
   - 632.0559833052946 hold_7_5 - 563.8357974837135 hold_7_6
   - 508.907492661978 hold_7_7 - 463.7312683483005 hold_7_8
   - 425.9217692449557 hold_7_9 - 393.8129362298712 hold_7_10
   - 366.2058860760343 hold_7_11 - 342.2158889468374 hold_7_12
   - 321.1757946076511 hold_7_13 - 302.5730204513139 hold_7_14
   - 286.0072388432151 hold_7_15 - 271.1612425982019 hold_7_16
   - 257.7804365442951 hold_7_17 - 245.6581156352056 hold_7_18
   - 234.624710346019 hold_7_19 - 224.5398056682828 hold_7_20
   - 215.286133570837 hold_7_21 - 206.7649921505117 hold_7_22
   - 198.8927112384791 hold_7_23 - 191.5978957952692 hold_7_24
   + flow_nic_in_7_7 + flow_nic_in_7_8 + flow_nic_in_7_9 + flow_nic_in_7_10
   <= 0
 intergroup_group_tp_0_via_nic_out_14_nic_in_0: tp_on_and_switch_nic_out_14_nic_in_0_0 = 1 ->
   - 5109.459042174185 hold_0_1 - 3040.15247603248 hold_0_2
   - 2163.816929787807 hold_0_3 - 1679.651095409346 hold_0_4
   - 1372.538192304928 hold_0_5 - 1160.371981735487 hold_0_6
   - 1005.016998327084 hold_0_7 - 886.3492175210624 hold_0_8
   - 792.7454804093763 hold_0_9 - 717.0235396836346 hold_0_10
   - 654.5060374151892 hold_0_11 - 602.0160715548276 hold_0_12
   - 557.3201993053406 hold_0_13 - 518.8024130345685 hold_0_14
   - 485.2645706247653 hold_0_15 - 455.7995324779789 hold_0_16
   - 429.7078711806942 hold_0_17 - 406.441639349837 hold_0_18
   - 385.5654670831283 hold_0_19 - 366.7290626737931 hold_0_20
   - 349.6473985665369 hold_0_21 - 334.0861883096475 hold_0_22
   - 319.8510763911254 hold_0_23 - 306.7794787980271 hold_0_24
   - 294.7343457685529 hold_0_25 - 283.5993383963553 hold_0_26
   + flow_nic_in_0_0 + flow_nic_in_0_1 <= 0
 intergroup_group_tp_2_via_nic_out_14_nic_in_2: tp_on_and_switch_nic_out_14_nic_in_2_2 = 1 ->
   - 5109.459042174185 hold_2_1 - 3040.15247603248 hold_2_2
   - 2163.816929787807 hold_2_3 - 1679.651095409346 hold_2_4
   - 1372.538192304928 hold_2_5 - 1160.371981735487 hold_2_6
   - 1005.016998327084 hold_2_7 - 886.3492175210624 hold_2_8
   - 792.7454804093763 hold_2_9 - 717.0235396836346 hold_2_10
   - 654.5060374151892 hold_2_11 - 602.0160715548276 hold_2_12
   - 557.3201993053406 hold_2_13 - 518.8024130345685 hold_2_14
   - 485.2645706247653 hold_2_15 - 455.7995324779789 hold_2_16
   - 429.7078711806942 hold_2_17 - 406.441639349837 hold_2_18
   - 385.5654670831283 hold_2_19 - 366.7290626737931 hold_2_20
   - 349.6473985665369 hold_2_21 - 334.0861883096475 hold_2_22
   - 319.8510763911254 hold_2_23 - 306.7794787980271 hold_2_24
   - 294.7343457685529 hold_2_25 - 283.5993383963553 hold_2_26
   + flow_nic_in_2_2 + flow_nic_in_2_3 <= 0
 intergroup_group_tp_5_via_nic_out_14_nic_in_5: tp_on_and_switch_nic_out_14_nic_in_5_5 = 1 ->
   - 1003.80710574974 hold_5_1 - 731.3070602333747 hold_5_2
   - 575.168150408772 hold_5_3 - 473.9719863597182 hold_5_4
   - 403.0573241561659 hold_5_5 - 350.6011407158714 hold_5_6
   - 310.2264922256207 hold_5_7 - 278.1905300976189 hold_5_8
   - 252.1517368463786 hold_5_9 - 230.5702309237719 hold_5_10
   - 212.3917563113273 hold_5_11 - 196.8702357402951 hold_5_12
   + flow_nic_in_5_5 + flow_nic_in_5_6 <= 0
 intergroup_group_tp_7_via_nic_out_14_nic_in_7: tp_on_and_switch_nic_out_14_nic_in_7_7 = 1 ->
   - 3938.129362298712 hold_7_1 - 2245.398056682829 hold_7_2
   - 1570.393681992138 hold_7_3 - 1207.422266154505 hold_7_4
   - 980.7399028740582 hold_7_5 - 825.7188295270446 hold_7_6
   - 713.0156975159247 hold_7_7 - 627.3835346119131 hold_7_8
   - 560.1145770378789 hold_7_9 - 505.8740407363503 hold_7_10
   - 461.2111435376709 hold_7_11 - 423.7948968313702 hold_7_12
   - 391.9939674741698 hold_7_13 - 364.6324954159618 hold_7_14
   - 340.8415035882089 hold_7_15 - 319.9649147805446 hold_7_16
   - 301.498113978548 hold_7_17 - 285.0466250764638 hold_7_18
   - 270.2976165967061 hold_7_19 - 256.999818386795 hold_7_20
   - 244.9490885566889 hold_7_21 - 233.977859143145 hold_7_22
   - 223.9472965418282 hold_7_23 - 214.7413955270226 hold_7_24
   + flow_nic_in_7_7 + flow_nic_in_7_8 + flow_nic_in_7_9 + flow_nic_in_7_10
   <= 0
 intergroup_group_tp_12_via_nic_out_14_nic_in_12: tp_on_and_switch_nic_out_14_nic_in_12_12 = 1 ->
   - 1141.859320933407 hold_12_1 - 887.6822328403811 hold_12_2
   - 726.0614587427827 hold_12_3 - 614.2285427704226 hold_12_4
   - 532.2480794992344 hold_12_5 - 469.5744516749779 hold_12_6
   - 420.1058677289324 hold_12_7 - 380.0667169114867 hold_12_8
   + flow_nic_in_12_12 + flow_nic_in_12_13 <= 0
 intergroup_group_tp_0_via_nic_out_15_nic_in_0: tp_on_and_switch_nic_out_15_nic_in_0_0 = 1 ->
   - 1318.886366057375 hold_0_1 - 1121.791978759409 hold_0_2
   - 975.9465237692996 hold_0_3 - 863.6609750662036 hold_0_4
   - 774.5470361079126 hold_0_5 - 702.102919160611 hold_0_6
   - 642.0512576984546 hold_0_7 - 591.4627737492349 hold_0_8
   - 548.2639746661837 hold_0_9 - 510.9459042174184 hold_0_10
   - 478.3842582309013 hold_0_11 - 449.7241616854303 hold_0_12
   - 424.3040228527632 hold_0_13 - 401.6038301819989 hold_0_14
   - 381.2092030043309 hold_0_15 - 362.7858665290911 hold_0_16
   - 346.0611871618131 hold_0_17 - 330.8105901512063 hold_0_18
   - 316.847417018637 hold_0_19 - 304.0152476032479 hold_0_20
   - 292.1820152084147 hold_0_21 - 281.2354446009739 hold_0_22
   - 271.0794784674806 hold_0_23 - 261.6314511702397 hold_0_24
   - 252.8198336223107 hold_0_25 - 244.5824190234829 hold_0_26
   + flow_nic_in_0_0 + flow_nic_in_0_1 <= 0
 intergroup_group_tp_2_via_nic_out_15_nic_in_2: tp_on_and_switch_nic_out_15_nic_in_2_2 = 1 ->
   - 1318.886366057375 hold_2_1 - 1121.791978759409 hold_2_2
   - 975.9465237692996 hold_2_3 - 863.6609750662036 hold_2_4
   - 774.5470361079126 hold_2_5 - 702.102919160611 hold_2_6
   - 642.0512576984546 hold_2_7 - 591.4627737492349 hold_2_8
   - 548.2639746661837 hold_2_9 - 510.9459042174184 hold_2_10
   - 478.3842582309013 hold_2_11 - 449.7241616854303 hold_2_12
   - 424.3040228527632 hold_2_13 - 401.6038301819989 hold_2_14
   - 381.2092030043309 hold_2_15 - 362.7858665290911 hold_2_16
   - 346.0611871618131 hold_2_17 - 330.8105901512063 hold_2_18
   - 316.847417018637 hold_2_19 - 304.0152476032479 hold_2_20
   - 292.1820152084147 hold_2_21 - 281.2354446009739 hold_2_22
   - 271.0794784674806 hold_2_23 - 261.6314511702397 hold_2_24
   - 252.8198336223107 hold_2_25 - 244.5824190234829 hold_2_26
   + flow_nic_in_2_2 + flow_nic_in_2_3 <= 0
 intergroup_group_tp_5_via_nic_out_15_nic_in_5: tp_on_and_switch_nic_out_15_nic_in_5_5 = 1 ->
   - 2305.702309237719 hold_5_1 - 1242.367720967523 hold_5_2
   - 850.2519296467542 hold_5_3 - 646.2748288017978 hold_5_4
   - 521.2305906758797 hold_5_5 - 436.730044930787 hold_5_6
   - 375.8054450200391 hold_5_7 - 329.7980448692438 hold_5_8
   - 293.8267585678413 hold_5_9 - 264.9306058433094 hold_5_10
   - 241.2090943547263 hold_5_11 - 221.3864637231713 hold_5_12
   + flow_nic_in_5_5 + flow_nic_in_5_6 <= 0
 intergroup_group_tp_7_via_nic_out_15_nic_in_7: tp_on_and_switch_nic_out_15_nic_in_7_7 = 1 ->
   - 1224.848251508707 hold_7_1 - 992.2058919831748 hold_7_2
   - 833.8315476830738 hold_7_3 - 719.0568275404518 hold_7_4
   - 632.0559833052946 hold_7_5 - 563.8357974837135 hold_7_6
   - 508.907492661978 hold_7_7 - 463.7312683483005 hold_7_8
   - 425.9217692449557 hold_7_9 - 393.8129362298712 hold_7_10
   - 366.2058860760343 hold_7_11 - 342.2158889468374 hold_7_12
   - 321.1757946076511 hold_7_13 - 302.5730204513139 hold_7_14
   - 286.0072388432151 hold_7_15 - 271.1612425982019 hold_7_16
   - 257.7804365442951 hold_7_17 - 245.6581156352056 hold_7_18
   - 234.624710346019 hold_7_19 - 224.5398056682828 hold_7_20
   - 215.286133570837 hold_7_21 - 206.7649921505117 hold_7_22
   - 198.8927112384791 hold_7_23 - 191.5978957952692 hold_7_24
   + flow_nic_in_7_7 + flow_nic_in_7_8 + flow_nic_in_7_9 + flow_nic_in_7_10
   <= 0
 intergroup_group_tp_12_via_nic_out_15_nic_in_12: tp_on_and_switch_nic_out_15_nic_in_12_12 = 1 ->
   - 3192.189351095739 hold_12_1 - 1772.957696786115 hold_12_2
   - 1227.304267597741 hold_12_3 - 938.4749875300135 hold_12_4
   - 759.6918879625396 hold_12_5 - 638.1263589100226 hold_12_6
   - 550.0996746229188 hold_12_7 - 483.4147645504219 hold_12_8
   + flow_nic_in_12_12 + flow_nic_in_12_13 <= 0
Bounds
Binaries
 hold_0_1 hold_0_2 hold_0_3 hold_0_4 hold_0_5 hold_0_6 hold_0_7 hold_0_8
 hold_0_9 hold_0_10 hold_0_11 hold_0_12 hold_0_13 hold_0_14 hold_0_15
 hold_0_16 hold_0_17 hold_0_18 hold_0_19 hold_0_20 hold_0_21 hold_0_22
 hold_0_23 hold_0_24 hold_0_25 hold_0_26 hold_1_1 hold_1_2 hold_1_3
 hold_1_4 hold_1_5 hold_1_6 hold_1_7 hold_1_8 hold_1_9 hold_1_10 hold_1_11
 hold_1_12 hold_1_13 hold_1_14 hold_1_15 hold_1_16 hold_1_17 hold_1_18
 hold_1_19 hold_1_20 hold_1_21 hold_1_22 hold_1_23 hold_1_24 hold_1_25
 hold_1_26 hold_2_1 hold_2_2 hold_2_3 hold_2_4 hold_2_5 hold_2_6 hold_2_7
 hold_2_8 hold_2_9 hold_2_10 hold_2_11 hold_2_12 hold_2_13 hold_2_14
 hold_2_15 hold_2_16 hold_2_17 hold_2_18 hold_2_19 hold_2_20 hold_2_21
 hold_2_22 hold_2_23 hold_2_24 hold_2_25 hold_2_26 hold_3_1 hold_3_2
 hold_3_3 hold_3_4 hold_3_5 hold_3_6 hold_3_7 hold_3_8 hold_3_9 hold_3_10
 hold_3_11 hold_3_12 hold_3_13 hold_3_14 hold_3_15 hold_3_16 hold_3_17
 hold_3_18 hold_3_19 hold_3_20 hold_3_21 hold_3_22 hold_3_23 hold_3_24
 hold_3_25 hold_3_26 hold_4_1 hold_4_2 hold_4_3 hold_4_4 hold_4_5 hold_4_6
 hold_4_7 hold_4_8 hold_4_9 hold_4_10 hold_4_11 hold_4_12 hold_4_13
 hold_4_14 hold_4_15 hold_4_16 hold_4_17 hold_4_18 hold_4_19 hold_4_20
 hold_4_21 hold_4_22 hold_4_23 hold_4_24 hold_4_25 hold_4_26 hold_5_1
 hold_5_2 hold_5_3 hold_5_4 hold_5_5 hold_5_6 hold_5_7 hold_5_8 hold_5_9
 hold_5_10 hold_5_11 hold_5_12 hold_5_13 hold_5_14 hold_5_15 hold_5_16
 hold_5_17 hold_5_18 hold_5_19 hold_5_20 hold_5_21 hold_5_22 hold_5_23
 hold_5_24 hold_5_25 hold_5_26 hold_6_1 hold_6_2 hold_6_3 hold_6_4 hold_6_5
 hold_6_6 hold_6_7 hold_6_8 hold_6_9 hold_6_10 hold_6_11 hold_6_12
 hold_6_13 hold_6_14 hold_6_15 hold_6_16 hold_6_17 hold_6_18 hold_6_19
 hold_6_20 hold_6_21 hold_6_22 hold_6_23 hold_6_24 hold_6_25 hold_6_26
 hold_7_1 hold_7_2 hold_7_3 hold_7_4 hold_7_5 hold_7_6 hold_7_7 hold_7_8
 hold_7_9 hold_7_10 hold_7_11 hold_7_12 hold_7_13 hold_7_14 hold_7_15
 hold_7_16 hold_7_17 hold_7_18 hold_7_19 hold_7_20 hold_7_21 hold_7_22
 hold_7_23 hold_7_24 hold_7_25 hold_7_26 hold_8_1 hold_8_2 hold_8_3
 hold_8_4 hold_8_5 hold_8_6 hold_8_7 hold_8_8 hold_8_9 hold_8_10 hold_8_11
 hold_8_12 hold_8_13 hold_8_14 hold_8_15 hold_8_16 hold_8_17 hold_8_18
 hold_8_19 hold_8_20 hold_8_21 hold_8_22 hold_8_23 hold_8_24 hold_8_25
 hold_8_26 hold_9_1 hold_9_2 hold_9_3 hold_9_4 hold_9_5 hold_9_6 hold_9_7
 hold_9_8 hold_9_9 hold_9_10 hold_9_11 hold_9_12 hold_9_13 hold_9_14
 hold_9_15 hold_9_16 hold_9_17 hold_9_18 hold_9_19 hold_9_20 hold_9_21
 hold_9_22 hold_9_23 hold_9_24 hold_9_25 hold_9_26 hold_10_1 hold_10_2
 hold_10_3 hold_10_4 hold_10_5 hold_10_6 hold_10_7 hold_10_8 hold_10_9
 hold_10_10 hold_10_11 hold_10_12 hold_10_13 hold_10_14 hold_10_15
 hold_10_16 hold_10_17 hold_10_18 hold_10_19 hold_10_20 hold_10_21
 hold_10_22 hold_10_23 hold_10_24 hold_10_25 hold_10_26 hold_11_1 hold_11_2
 hold_11_3 hold_11_4 hold_11_5 hold_11_6 hold_11_7 hold_11_8 hold_11_9
 hold_11_10 hold_11_11 hold_11_12 hold_11_13 hold_11_14 hold_11_15
 hold_11_16 hold_11_17 hold_11_18 hold_11_19 hold_11_20 hold_11_21
 hold_11_22 hold_11_23 hold_11_24 hold_11_25 hold_11_26 hold_12_1 hold_12_2
 hold_12_3 hold_12_4 hold_12_5 hold_12_6 hold_12_7 hold_12_8 hold_12_9
 hold_12_10 hold_12_11 hold_12_12 hold_12_13 hold_12_14 hold_12_15
 hold_12_16 hold_12_17 hold_12_18 hold_12_19 hold_12_20 hold_12_21
 hold_12_22 hold_12_23 hold_12_24 hold_12_25 hold_12_26 hold_13_1 hold_13_2
 hold_13_3 hold_13_4 hold_13_5 hold_13_6 hold_13_7 hold_13_8 hold_13_9
 hold_13_10 hold_13_11 hold_13_12 hold_13_13 hold_13_14 hold_13_15
 hold_13_16 hold_13_17 hold_13_18 hold_13_19 hold_13_20 hold_13_21
 hold_13_22 hold_13_23 hold_13_24 hold_13_25 hold_13_26 hold_14_1 hold_14_2
 hold_14_3 hold_14_4 hold_14_5 hold_14_6 hold_14_7 hold_14_8 hold_14_9
 hold_14_10 hold_14_11 hold_14_12 hold_14_13 hold_14_14 hold_14_15
 hold_14_16 hold_14_17 hold_14_18 hold_14_19 hold_14_20 hold_14_21
 hold_14_22 hold_14_23 hold_14_24 hold_14_25 hold_14_26 hold_15_1 hold_15_2
 hold_15_3 hold_15_4 hold_15_5 hold_15_6 hold_15_7 hold_15_8 hold_15_9
 hold_15_10 hold_15_11 hold_15_12 hold_15_13 hold_15_14 hold_15_15
 hold_15_16 hold_15_17 hold_15_18 hold_15_19 hold_15_20 hold_15_21
 hold_15_22 hold_15_23 hold_15_24 hold_15_25 hold_15_26
 switch_source_nic_in_0 switch_nic_out_0_sink switch_nic_out_0_nic_in_2
 switch_nic_out_0_nic_in_4 switch_nic_out_0_nic_in_5
 switch_nic_out_0_nic_in_7 switch_nic_out_0_nic_in_11
 switch_nic_out_0_nic_in_12 switch_nic_out_0_nic_in_14
 switch_nic_out_0_nic_in_15 switch_nic_in_0_0 switch_0_nic_out_0
 switch_nic_in_0_1 switch_1_nic_out_0 switch_0_1 switch_1_0
 switch_source_nic_in_2 switch_nic_out_2_sink switch_nic_out_2_nic_in_0
 switch_nic_out_2_nic_in_4 switch_nic_out_2_nic_in_5
 switch_nic_out_2_nic_in_7 switch_nic_out_2_nic_in_11
 switch_nic_out_2_nic_in_12 switch_nic_out_2_nic_in_14
 switch_nic_out_2_nic_in_15 switch_nic_in_2_2 switch_2_nic_out_2
 switch_nic_in_2_3 switch_3_nic_out_2 switch_2_3 switch_3_2
 switch_source_nic_in_4 switch_nic_out_4_sink switch_nic_out_4_nic_in_0
 switch_nic_out_4_nic_in_2 switch_nic_out_4_nic_in_5
 switch_nic_out_4_nic_in_7 switch_nic_out_4_nic_in_11
 switch_nic_out_4_nic_in_12 switch_nic_out_4_nic_in_14
 switch_nic_out_4_nic_in_15 switch_nic_in_4_4 switch_4_nic_out_4
 switch_source_nic_in_5 switch_nic_out_5_sink switch_nic_out_5_nic_in_0
 switch_nic_out_5_nic_in_2 switch_nic_out_5_nic_in_4
 switch_nic_out_5_nic_in_7 switch_nic_out_5_nic_in_11
 switch_nic_out_5_nic_in_12 switch_nic_out_5_nic_in_14
 switch_nic_out_5_nic_in_15 switch_nic_in_5_5 switch_5_nic_out_5
 switch_nic_in_5_6 switch_6_nic_out_5 switch_5_6 switch_6_5
 switch_source_nic_in_7 switch_nic_out_7_sink switch_nic_out_7_nic_in_0
 switch_nic_out_7_nic_in_2 switch_nic_out_7_nic_in_4
 switch_nic_out_7_nic_in_5 switch_nic_out_7_nic_in_11
 switch_nic_out_7_nic_in_12 switch_nic_out_7_nic_in_14
 switch_nic_out_7_nic_in_15 switch_nic_in_7_7 switch_7_nic_out_7
 switch_nic_in_7_8 switch_8_nic_out_7 switch_nic_in_7_9 switch_9_nic_out_7
 switch_nic_in_7_10 switch_10_nic_out_7 switch_7_8 switch_7_9 switch_7_10
 switch_8_7 switch_8_9 switch_8_10 switch_9_7 switch_9_8 switch_9_10
 switch_10_7 switch_10_8 switch_10_9 switch_source_nic_in_11
 switch_nic_out_11_sink switch_nic_out_11_nic_in_0
 switch_nic_out_11_nic_in_2 switch_nic_out_11_nic_in_4
 switch_nic_out_11_nic_in_5 switch_nic_out_11_nic_in_7
 switch_nic_out_11_nic_in_12 switch_nic_out_11_nic_in_14
 switch_nic_out_11_nic_in_15 switch_nic_in_11_11 switch_11_nic_out_11
 switch_source_nic_in_12 switch_nic_out_12_sink switch_nic_out_12_nic_in_0
 switch_nic_out_12_nic_in_2 switch_nic_out_12_nic_in_4
 switch_nic_out_12_nic_in_5 switch_nic_out_12_nic_in_7
 switch_nic_out_12_nic_in_11 switch_nic_out_12_nic_in_14
 switch_nic_out_12_nic_in_15 switch_nic_in_12_12 switch_12_nic_out_12
 switch_nic_in_12_13 switch_13_nic_out_12 switch_12_13 switch_13_12
 switch_source_nic_in_14 switch_nic_out_14_sink switch_nic_out_14_nic_in_0
 switch_nic_out_14_nic_in_2 switch_nic_out_14_nic_in_4
 switch_nic_out_14_nic_in_5 switch_nic_out_14_nic_in_7
 switch_nic_out_14_nic_in_11 switch_nic_out_14_nic_in_12
 switch_nic_out_14_nic_in_15 switch_nic_in_14_14 switch_14_nic_out_14
 switch_source_nic_in_15 switch_nic_out_15_sink switch_nic_out_15_nic_in_0
 switch_nic_out_15_nic_in_2 switch_nic_out_15_nic_in_4
 switch_nic_out_15_nic_in_5 switch_nic_out_15_nic_in_7
 switch_nic_out_15_nic_in_11 switch_nic_out_15_nic_in_12
 switch_nic_out_15_nic_in_14 switch_nic_in_15_15 switch_15_nic_out_15
 group_tp_active_0 group_tp_active_2 group_tp_active_4 group_tp_active_5
 group_tp_active_7 group_tp_active_11 group_tp_active_12 group_tp_active_14
 group_tp_active_15 is_active_0 is_active_1 is_active_2 is_active_3
 is_active_4 is_active_5 is_active_6 is_active_7 is_active_8 is_active_9
 is_active_10 is_active_11 is_active_12 is_active_13 is_active_14
 is_active_15 is_start_zero_0_for_source_nic_in_0
 is_start_zero_1_for_source_nic_in_0 is_end_complete_0 is_end_complete_1
 pp_permission_0_1 pp_permission_1_0 is_start_zero_2_for_source_nic_in_2
 is_start_zero_3_for_source_nic_in_2 is_end_complete_2 is_end_complete_3
 pp_permission_2_3 pp_permission_3_2 is_start_zero_4_for_source_nic_in_4
 is_end_complete_4 is_start_zero_5_for_source_nic_in_5
 is_start_zero_6_for_source_nic_in_5 is_end_complete_5 is_end_complete_6
 pp_permission_5_6 pp_permission_6_5 is_start_zero_7_for_source_nic_in_7
 is_start_zero_8_for_source_nic_in_7 is_start_zero_9_for_source_nic_in_7
 is_start_zero_10_for_source_nic_in_7 is_end_complete_7 is_end_complete_8
 is_end_complete_9 is_end_complete_10 pp_permission_7_8 pp_permission_7_9
 pp_permission_7_10 pp_permission_8_7 pp_permission_8_9 pp_permission_8_10
 pp_permission_9_7 pp_permission_9_8 pp_permission_9_10 pp_permission_10_7
 pp_permission_10_8 pp_permission_10_9
 is_start_zero_11_for_source_nic_in_11 is_end_complete_11
 is_start_zero_12_for_source_nic_in_12
 is_start_zero_13_for_source_nic_in_12 is_end_complete_12
 is_end_complete_13 pp_permission_12_13 pp_permission_13_12
 is_start_zero_14_for_source_nic_in_14 is_end_complete_14
 is_start_zero_15_for_source_nic_in_15 is_end_complete_15
 tp_on_and_switch_source_nic_in_0_0 tp_on_and_switch_nic_out_0_nic_in_2_2
 tp_on_and_switch_nic_out_0_nic_in_5_5
 tp_on_and_switch_nic_out_0_nic_in_7_7
 tp_on_and_switch_nic_out_0_nic_in_12_12 tp_on_and_switch_source_nic_in_2_2
 tp_on_and_switch_nic_out_2_nic_in_0_0
 tp_on_and_switch_nic_out_2_nic_in_5_5
 tp_on_and_switch_nic_out_2_nic_in_7_7
 tp_on_and_switch_nic_out_2_nic_in_12_12
 tp_on_and_switch_nic_out_4_nic_in_0_0
 tp_on_and_switch_nic_out_4_nic_in_2_2
 tp_on_and_switch_nic_out_4_nic_in_5_5
 tp_on_and_switch_nic_out_4_nic_in_7_7
 tp_on_and_switch_nic_out_4_nic_in_12_12 tp_on_and_switch_source_nic_in_5_5
 tp_on_and_switch_nic_out_5_nic_in_0_0
 tp_on_and_switch_nic_out_5_nic_in_2_2
 tp_on_and_switch_nic_out_5_nic_in_7_7
 tp_on_and_switch_nic_out_5_nic_in_12_12 tp_on_and_switch_source_nic_in_7_7
 tp_on_and_switch_nic_out_7_nic_in_0_0
 tp_on_and_switch_nic_out_7_nic_in_2_2
 tp_on_and_switch_nic_out_7_nic_in_5_5
 tp_on_and_switch_nic_out_7_nic_in_12_12
 tp_on_and_switch_nic_out_11_nic_in_0_0
 tp_on_and_switch_nic_out_11_nic_in_2_2
 tp_on_and_switch_nic_out_11_nic_in_5_5
 tp_on_and_switch_nic_out_11_nic_in_7_7
 tp_on_and_switch_nic_out_11_nic_in_12_12
 tp_on_and_switch_source_nic_in_12_12
 tp_on_and_switch_nic_out_12_nic_in_0_0
 tp_on_and_switch_nic_out_12_nic_in_2_2
 tp_on_and_switch_nic_out_12_nic_in_5_5
 tp_on_and_switch_nic_out_12_nic_in_7_7
 tp_on_and_switch_nic_out_14_nic_in_0_0
 tp_on_and_switch_nic_out_14_nic_in_2_2
 tp_on_and_switch_nic_out_14_nic_in_5_5
 tp_on_and_switch_nic_out_14_nic_in_7_7
 tp_on_and_switch_nic_out_14_nic_in_12_12
 tp_on_and_switch_nic_out_15_nic_in_0_0
 tp_on_and_switch_nic_out_15_nic_in_2_2
 tp_on_and_switch_nic_out_15_nic_in_5_5
 tp_on_and_switch_nic_out_15_nic_in_7_7
 tp_on_and_switch_nic_out_15_nic_in_12_12
Generals
 start_0 start_1 start_2 start_3 start_4 start_5 start_6 start_7 start_8
 start_9 start_10 start_11 start_12 start_13 start_14 start_15
 group_start_2 val_end_0 val_end_1 group_end_0 group_start_4 group_start_5
 group_start_7 group_start_11 group_start_12 group_start_14 group_start_15
 group_start_0 val_end_2 val_end_3 group_end_2 val_end_4 group_end_4
 val_end_5 val_end_6 group_end_5 val_end_7 val_end_8 val_end_9 val_end_10
 group_end_7 val_end_11 group_end_11 val_end_12 val_end_13 group_end_12
 val_end_14 group_end_14 val_end_15 group_end_15
General Constraints
 min_start_2: group_start_2 = MIN ( start_2 , start_3 )
 max_end_0: group_end_0 = MAX ( val_end_0 , val_end_1 )
 min_start_4: group_start_4 = MIN ( start_4 )
 min_start_5: group_start_5 = MIN ( start_5 , start_6 )
 min_start_7: group_start_7 = MIN ( start_7 , start_8 , start_9 , start_10 )
 min_start_11: group_start_11 = MIN ( start_11 )
 min_start_12: group_start_12 = MIN ( start_12 , start_13 )
 min_start_14: group_start_14 = MIN ( start_14 )
 min_start_15: group_start_15 = MIN ( start_15 )
 min_start_0: group_start_0 = MIN ( start_0 , start_1 )
 max_end_2: group_end_2 = MAX ( val_end_2 , val_end_3 )
 max_end_4: group_end_4 = MAX ( val_end_4 )
 max_end_5: group_end_5 = MAX ( val_end_5 , val_end_6 )
 max_end_7: group_end_7 = MAX ( val_end_7 , val_end_8 , val_end_9 ,
   val_end_10 )
 max_end_11: group_end_11 = MAX ( val_end_11 )
 max_end_12: group_end_12 = MAX ( val_end_12 , val_end_13 )
 max_end_14: group_end_14 = MAX ( val_end_14 )
 max_end_15: group_end_15 = MAX ( val_end_15 )
End
