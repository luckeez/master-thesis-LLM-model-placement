\ Model 2026-02-27_12-05-57
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  0.00136434511434511 flow_source_nic_in_0
   + 0.00136434511434511 flow_source_nic_in_4
   + 0.00136434511434511 flow_source_nic_in_6
   + 0.00136434511434511 flow_source_nic_in_8
   + 0.00136434511434511 flow_source_nic_in_9
   - 0.00617977528089888 is_active_0 - 0.00617977528089888 is_active_1
   - 0.00617977528089888 is_active_2 - 0.00617977528089888 is_active_3
   - 0.00617977528089888 is_active_4 - 0.00617977528089888 is_active_5
   - 0.00617977528089888 is_active_6 - 0.00617977528089888 is_active_7
   - 0.0252808988764045 is_active_8 - 0.0252808988764045 is_active_9
   + 0 val_end_0 + 0 val_end_1 + 0 val_end_2 + 0 val_end_3 + 0 val_end_4
   + 0 val_end_5 + 0 val_end_6 + 0 val_end_7 + 0 val_end_8 + 0 val_end_9
Subject To
 mem_active_constr_0: hold_0_1 + hold_0_2 + hold_0_3 + hold_0_4 + hold_0_5
   + hold_0_6 + hold_0_7 + hold_0_8 + hold_0_9 + hold_0_10 + hold_0_11
   + hold_0_12 + hold_0_13 + hold_0_14 + hold_0_15 + hold_0_16 + hold_0_17
   + hold_0_18 + hold_0_19 + hold_0_20 + hold_0_21 + hold_0_22 + hold_0_23
   + hold_0_24 - 81 is_active_0 <= 0
 mem_active_constr2_0: hold_0_1 + hold_0_2 + hold_0_3 + hold_0_4 + hold_0_5
   + hold_0_6 + hold_0_7 + hold_0_8 + hold_0_9 + hold_0_10 + hold_0_11
   + hold_0_12 + hold_0_13 + hold_0_14 + hold_0_15 + hold_0_16 + hold_0_17
   + hold_0_18 + hold_0_19 + hold_0_20 + hold_0_21 + hold_0_22 + hold_0_23
   + hold_0_24 - is_active_0 >= 0
 mem_active_constr_1: hold_1_1 + hold_1_2 + hold_1_3 + hold_1_4 + hold_1_5
   + hold_1_6 + hold_1_7 + hold_1_8 + hold_1_9 + hold_1_10 + hold_1_11
   + hold_1_12 + hold_1_13 + hold_1_14 + hold_1_15 + hold_1_16 + hold_1_17
   + hold_1_18 + hold_1_19 + hold_1_20 + hold_1_21 + hold_1_22 + hold_1_23
   + hold_1_24 - 81 is_active_1 <= 0
 mem_active_constr2_1: hold_1_1 + hold_1_2 + hold_1_3 + hold_1_4 + hold_1_5
   + hold_1_6 + hold_1_7 + hold_1_8 + hold_1_9 + hold_1_10 + hold_1_11
   + hold_1_12 + hold_1_13 + hold_1_14 + hold_1_15 + hold_1_16 + hold_1_17
   + hold_1_18 + hold_1_19 + hold_1_20 + hold_1_21 + hold_1_22 + hold_1_23
   + hold_1_24 - is_active_1 >= 0
 mem_active_constr_2: hold_2_1 + hold_2_2 + hold_2_3 + hold_2_4 + hold_2_5
   + hold_2_6 + hold_2_7 + hold_2_8 + hold_2_9 + hold_2_10 + hold_2_11
   + hold_2_12 + hold_2_13 + hold_2_14 + hold_2_15 + hold_2_16 + hold_2_17
   + hold_2_18 + hold_2_19 + hold_2_20 + hold_2_21 + hold_2_22 + hold_2_23
   + hold_2_24 - 81 is_active_2 <= 0
 mem_active_constr2_2: hold_2_1 + hold_2_2 + hold_2_3 + hold_2_4 + hold_2_5
   + hold_2_6 + hold_2_7 + hold_2_8 + hold_2_9 + hold_2_10 + hold_2_11
   + hold_2_12 + hold_2_13 + hold_2_14 + hold_2_15 + hold_2_16 + hold_2_17
   + hold_2_18 + hold_2_19 + hold_2_20 + hold_2_21 + hold_2_22 + hold_2_23
   + hold_2_24 - is_active_2 >= 0
 mem_active_constr_3: hold_3_1 + hold_3_2 + hold_3_3 + hold_3_4 + hold_3_5
   + hold_3_6 + hold_3_7 + hold_3_8 + hold_3_9 + hold_3_10 + hold_3_11
   + hold_3_12 + hold_3_13 + hold_3_14 + hold_3_15 + hold_3_16 + hold_3_17
   + hold_3_18 + hold_3_19 + hold_3_20 + hold_3_21 + hold_3_22 + hold_3_23
   + hold_3_24 - 81 is_active_3 <= 0
 mem_active_constr2_3: hold_3_1 + hold_3_2 + hold_3_3 + hold_3_4 + hold_3_5
   + hold_3_6 + hold_3_7 + hold_3_8 + hold_3_9 + hold_3_10 + hold_3_11
   + hold_3_12 + hold_3_13 + hold_3_14 + hold_3_15 + hold_3_16 + hold_3_17
   + hold_3_18 + hold_3_19 + hold_3_20 + hold_3_21 + hold_3_22 + hold_3_23
   + hold_3_24 - is_active_3 >= 0
 mem_active_constr_4: hold_4_1 + hold_4_2 + hold_4_3 + hold_4_4 + hold_4_5
   + hold_4_6 + hold_4_7 + hold_4_8 + hold_4_9 + hold_4_10 + hold_4_11
   + hold_4_12 + hold_4_13 + hold_4_14 + hold_4_15 + hold_4_16 + hold_4_17
   + hold_4_18 + hold_4_19 + hold_4_20 + hold_4_21 + hold_4_22 + hold_4_23
   + hold_4_24 - 81 is_active_4 <= 0
 mem_active_constr2_4: hold_4_1 + hold_4_2 + hold_4_3 + hold_4_4 + hold_4_5
   + hold_4_6 + hold_4_7 + hold_4_8 + hold_4_9 + hold_4_10 + hold_4_11
   + hold_4_12 + hold_4_13 + hold_4_14 + hold_4_15 + hold_4_16 + hold_4_17
   + hold_4_18 + hold_4_19 + hold_4_20 + hold_4_21 + hold_4_22 + hold_4_23
   + hold_4_24 - is_active_4 >= 0
 mem_active_constr_5: hold_5_1 + hold_5_2 + hold_5_3 + hold_5_4 + hold_5_5
   + hold_5_6 + hold_5_7 + hold_5_8 + hold_5_9 + hold_5_10 + hold_5_11
   + hold_5_12 + hold_5_13 + hold_5_14 + hold_5_15 + hold_5_16 + hold_5_17
   + hold_5_18 + hold_5_19 + hold_5_20 + hold_5_21 + hold_5_22 + hold_5_23
   + hold_5_24 - 81 is_active_5 <= 0
 mem_active_constr2_5: hold_5_1 + hold_5_2 + hold_5_3 + hold_5_4 + hold_5_5
   + hold_5_6 + hold_5_7 + hold_5_8 + hold_5_9 + hold_5_10 + hold_5_11
   + hold_5_12 + hold_5_13 + hold_5_14 + hold_5_15 + hold_5_16 + hold_5_17
   + hold_5_18 + hold_5_19 + hold_5_20 + hold_5_21 + hold_5_22 + hold_5_23
   + hold_5_24 - is_active_5 >= 0
 mem_active_constr_6: hold_6_1 + hold_6_2 + hold_6_3 + hold_6_4 + hold_6_5
   + hold_6_6 + hold_6_7 + hold_6_8 + hold_6_9 + hold_6_10 + hold_6_11
   + hold_6_12 + hold_6_13 + hold_6_14 + hold_6_15 + hold_6_16 + hold_6_17
   + hold_6_18 + hold_6_19 + hold_6_20 + hold_6_21 + hold_6_22 + hold_6_23
   + hold_6_24 - 81 is_active_6 <= 0
 mem_active_constr2_6: hold_6_1 + hold_6_2 + hold_6_3 + hold_6_4 + hold_6_5
   + hold_6_6 + hold_6_7 + hold_6_8 + hold_6_9 + hold_6_10 + hold_6_11
   + hold_6_12 + hold_6_13 + hold_6_14 + hold_6_15 + hold_6_16 + hold_6_17
   + hold_6_18 + hold_6_19 + hold_6_20 + hold_6_21 + hold_6_22 + hold_6_23
   + hold_6_24 - is_active_6 >= 0
 mem_active_constr_7: hold_7_1 + hold_7_2 + hold_7_3 + hold_7_4 + hold_7_5
   + hold_7_6 + hold_7_7 + hold_7_8 + hold_7_9 + hold_7_10 + hold_7_11
   + hold_7_12 + hold_7_13 + hold_7_14 + hold_7_15 + hold_7_16 + hold_7_17
   + hold_7_18 + hold_7_19 + hold_7_20 + hold_7_21 + hold_7_22 + hold_7_23
   + hold_7_24 - 81 is_active_7 <= 0
 mem_active_constr2_7: hold_7_1 + hold_7_2 + hold_7_3 + hold_7_4 + hold_7_5
   + hold_7_6 + hold_7_7 + hold_7_8 + hold_7_9 + hold_7_10 + hold_7_11
   + hold_7_12 + hold_7_13 + hold_7_14 + hold_7_15 + hold_7_16 + hold_7_17
   + hold_7_18 + hold_7_19 + hold_7_20 + hold_7_21 + hold_7_22 + hold_7_23
   + hold_7_24 - is_active_7 >= 0
 mem_active_constr_8: hold_8_1 + hold_8_2 + hold_8_3 + hold_8_4 + hold_8_5
   + hold_8_6 + hold_8_7 + hold_8_8 + hold_8_9 + hold_8_10 + hold_8_11
   + hold_8_12 + hold_8_13 + hold_8_14 + hold_8_15 + hold_8_16 + hold_8_17
   + hold_8_18 + hold_8_19 + hold_8_20 + hold_8_21 + hold_8_22 + hold_8_23
   + hold_8_24 - 81 is_active_8 <= 0
 mem_active_constr2_8: hold_8_1 + hold_8_2 + hold_8_3 + hold_8_4 + hold_8_5
   + hold_8_6 + hold_8_7 + hold_8_8 + hold_8_9 + hold_8_10 + hold_8_11
   + hold_8_12 + hold_8_13 + hold_8_14 + hold_8_15 + hold_8_16 + hold_8_17
   + hold_8_18 + hold_8_19 + hold_8_20 + hold_8_21 + hold_8_22 + hold_8_23
   + hold_8_24 - is_active_8 >= 0
 mem_active_constr_9: hold_9_1 + hold_9_2 + hold_9_3 + hold_9_4 + hold_9_5
   + hold_9_6 + hold_9_7 + hold_9_8 + hold_9_9 + hold_9_10 + hold_9_11
   + hold_9_12 + hold_9_13 + hold_9_14 + hold_9_15 + hold_9_16 + hold_9_17
   + hold_9_18 + hold_9_19 + hold_9_20 + hold_9_21 + hold_9_22 + hold_9_23
   + hold_9_24 - 81 is_active_9 <= 0
 mem_active_constr2_9: hold_9_1 + hold_9_2 + hold_9_3 + hold_9_4 + hold_9_5
   + hold_9_6 + hold_9_7 + hold_9_8 + hold_9_9 + hold_9_10 + hold_9_11
   + hold_9_12 + hold_9_13 + hold_9_14 + hold_9_15 + hold_9_16 + hold_9_17
   + hold_9_18 + hold_9_19 + hold_9_20 + hold_9_21 + hold_9_22 + hold_9_23
   + hold_9_24 - is_active_9 >= 0
 hold_constraint_0: hold_0_1 + hold_0_2 + hold_0_3 + hold_0_4 + hold_0_5
   + hold_0_6 + hold_0_7 + hold_0_8 + hold_0_9 + hold_0_10 + hold_0_11
   + hold_0_12 + hold_0_13 + hold_0_14 + hold_0_15 + hold_0_16 + hold_0_17
   + hold_0_18 + hold_0_19 + hold_0_20 + hold_0_21 + hold_0_22 + hold_0_23
   + hold_0_24 <= 1
 end_constraint_0: start_0 + hold_0_1 + 2 hold_0_2 + 3 hold_0_3
   + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7 + 8 hold_0_8
   + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12 + 13 hold_0_13
   + 14 hold_0_14 + 15 hold_0_15 + 16 hold_0_16 + 17 hold_0_17
   + 18 hold_0_18 + 19 hold_0_19 + 20 hold_0_20 + 21 hold_0_21
   + 22 hold_0_22 + 23 hold_0_23 + 24 hold_0_24 <= 80
 hold_constraint_1: hold_1_1 + hold_1_2 + hold_1_3 + hold_1_4 + hold_1_5
   + hold_1_6 + hold_1_7 + hold_1_8 + hold_1_9 + hold_1_10 + hold_1_11
   + hold_1_12 + hold_1_13 + hold_1_14 + hold_1_15 + hold_1_16 + hold_1_17
   + hold_1_18 + hold_1_19 + hold_1_20 + hold_1_21 + hold_1_22 + hold_1_23
   + hold_1_24 <= 1
 end_constraint_1: start_1 + hold_1_1 + 2 hold_1_2 + 3 hold_1_3
   + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7 + 8 hold_1_8
   + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 + 12 hold_1_12 + 13 hold_1_13
   + 14 hold_1_14 + 15 hold_1_15 + 16 hold_1_16 + 17 hold_1_17
   + 18 hold_1_18 + 19 hold_1_19 + 20 hold_1_20 + 21 hold_1_21
   + 22 hold_1_22 + 23 hold_1_23 + 24 hold_1_24 <= 80
 hold_constraint_2: hold_2_1 + hold_2_2 + hold_2_3 + hold_2_4 + hold_2_5
   + hold_2_6 + hold_2_7 + hold_2_8 + hold_2_9 + hold_2_10 + hold_2_11
   + hold_2_12 + hold_2_13 + hold_2_14 + hold_2_15 + hold_2_16 + hold_2_17
   + hold_2_18 + hold_2_19 + hold_2_20 + hold_2_21 + hold_2_22 + hold_2_23
   + hold_2_24 <= 1
 end_constraint_2: start_2 + hold_2_1 + 2 hold_2_2 + 3 hold_2_3
   + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7 + 8 hold_2_8
   + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 12 hold_2_12 + 13 hold_2_13
   + 14 hold_2_14 + 15 hold_2_15 + 16 hold_2_16 + 17 hold_2_17
   + 18 hold_2_18 + 19 hold_2_19 + 20 hold_2_20 + 21 hold_2_21
   + 22 hold_2_22 + 23 hold_2_23 + 24 hold_2_24 <= 80
 hold_constraint_3: hold_3_1 + hold_3_2 + hold_3_3 + hold_3_4 + hold_3_5
   + hold_3_6 + hold_3_7 + hold_3_8 + hold_3_9 + hold_3_10 + hold_3_11
   + hold_3_12 + hold_3_13 + hold_3_14 + hold_3_15 + hold_3_16 + hold_3_17
   + hold_3_18 + hold_3_19 + hold_3_20 + hold_3_21 + hold_3_22 + hold_3_23
   + hold_3_24 <= 1
 end_constraint_3: start_3 + hold_3_1 + 2 hold_3_2 + 3 hold_3_3
   + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7 + 8 hold_3_8
   + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11 + 12 hold_3_12 + 13 hold_3_13
   + 14 hold_3_14 + 15 hold_3_15 + 16 hold_3_16 + 17 hold_3_17
   + 18 hold_3_18 + 19 hold_3_19 + 20 hold_3_20 + 21 hold_3_21
   + 22 hold_3_22 + 23 hold_3_23 + 24 hold_3_24 <= 80
 hold_constraint_4: hold_4_1 + hold_4_2 + hold_4_3 + hold_4_4 + hold_4_5
   + hold_4_6 + hold_4_7 + hold_4_8 + hold_4_9 + hold_4_10 + hold_4_11
   + hold_4_12 + hold_4_13 + hold_4_14 + hold_4_15 + hold_4_16 + hold_4_17
   + hold_4_18 + hold_4_19 + hold_4_20 + hold_4_21 + hold_4_22 + hold_4_23
   + hold_4_24 <= 1
 end_constraint_4: start_4 + hold_4_1 + 2 hold_4_2 + 3 hold_4_3
   + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7 + 8 hold_4_8
   + 9 hold_4_9 + 10 hold_4_10 + 11 hold_4_11 + 12 hold_4_12 + 13 hold_4_13
   + 14 hold_4_14 + 15 hold_4_15 + 16 hold_4_16 + 17 hold_4_17
   + 18 hold_4_18 + 19 hold_4_19 + 20 hold_4_20 + 21 hold_4_21
   + 22 hold_4_22 + 23 hold_4_23 + 24 hold_4_24 <= 80
 hold_constraint_5: hold_5_1 + hold_5_2 + hold_5_3 + hold_5_4 + hold_5_5
   + hold_5_6 + hold_5_7 + hold_5_8 + hold_5_9 + hold_5_10 + hold_5_11
   + hold_5_12 + hold_5_13 + hold_5_14 + hold_5_15 + hold_5_16 + hold_5_17
   + hold_5_18 + hold_5_19 + hold_5_20 + hold_5_21 + hold_5_22 + hold_5_23
   + hold_5_24 <= 1
 end_constraint_5: start_5 + hold_5_1 + 2 hold_5_2 + 3 hold_5_3
   + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7 + 8 hold_5_8
   + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12 + 13 hold_5_13
   + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16 + 17 hold_5_17
   + 18 hold_5_18 + 19 hold_5_19 + 20 hold_5_20 + 21 hold_5_21
   + 22 hold_5_22 + 23 hold_5_23 + 24 hold_5_24 <= 80
 hold_constraint_6: hold_6_1 + hold_6_2 + hold_6_3 + hold_6_4 + hold_6_5
   + hold_6_6 + hold_6_7 + hold_6_8 + hold_6_9 + hold_6_10 + hold_6_11
   + hold_6_12 + hold_6_13 + hold_6_14 + hold_6_15 + hold_6_16 + hold_6_17
   + hold_6_18 + hold_6_19 + hold_6_20 + hold_6_21 + hold_6_22 + hold_6_23
   + hold_6_24 <= 1
 end_constraint_6: start_6 + hold_6_1 + 2 hold_6_2 + 3 hold_6_3
   + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7 + 8 hold_6_8
   + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 12 hold_6_12 + 13 hold_6_13
   + 14 hold_6_14 + 15 hold_6_15 + 16 hold_6_16 + 17 hold_6_17
   + 18 hold_6_18 + 19 hold_6_19 + 20 hold_6_20 + 21 hold_6_21
   + 22 hold_6_22 + 23 hold_6_23 + 24 hold_6_24 <= 80
 hold_constraint_7: hold_7_1 + hold_7_2 + hold_7_3 + hold_7_4 + hold_7_5
   + hold_7_6 + hold_7_7 + hold_7_8 + hold_7_9 + hold_7_10 + hold_7_11
   + hold_7_12 + hold_7_13 + hold_7_14 + hold_7_15 + hold_7_16 + hold_7_17
   + hold_7_18 + hold_7_19 + hold_7_20 + hold_7_21 + hold_7_22 + hold_7_23
   + hold_7_24 <= 1
 end_constraint_7: start_7 + hold_7_1 + 2 hold_7_2 + 3 hold_7_3
   + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7 + 8 hold_7_8
   + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11 + 12 hold_7_12 + 13 hold_7_13
   + 14 hold_7_14 + 15 hold_7_15 + 16 hold_7_16 + 17 hold_7_17
   + 18 hold_7_18 + 19 hold_7_19 + 20 hold_7_20 + 21 hold_7_21
   + 22 hold_7_22 + 23 hold_7_23 + 24 hold_7_24 <= 80
 hold_constraint_8: hold_8_1 + hold_8_2 + hold_8_3 + hold_8_4 + hold_8_5
   + hold_8_6 + hold_8_7 + hold_8_8 + hold_8_9 + hold_8_10 + hold_8_11
   + hold_8_12 + hold_8_13 + hold_8_14 + hold_8_15 + hold_8_16 + hold_8_17
   + hold_8_18 + hold_8_19 + hold_8_20 + hold_8_21 + hold_8_22 + hold_8_23
   + hold_8_24 <= 1
 end_constraint_8: start_8 + hold_8_1 + 2 hold_8_2 + 3 hold_8_3
   + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7 + 8 hold_8_8
   + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 12 hold_8_12 + 13 hold_8_13
   + 14 hold_8_14 + 15 hold_8_15 + 16 hold_8_16 + 17 hold_8_17
   + 18 hold_8_18 + 19 hold_8_19 + 20 hold_8_20 + 21 hold_8_21
   + 22 hold_8_22 + 23 hold_8_23 + 24 hold_8_24 <= 80
 hold_constraint_9: hold_9_1 + hold_9_2 + hold_9_3 + hold_9_4 + hold_9_5
   + hold_9_6 + hold_9_7 + hold_9_8 + hold_9_9 + hold_9_10 + hold_9_11
   + hold_9_12 + hold_9_13 + hold_9_14 + hold_9_15 + hold_9_16 + hold_9_17
   + hold_9_18 + hold_9_19 + hold_9_20 + hold_9_21 + hold_9_22 + hold_9_23
   + hold_9_24 <= 1
 end_constraint_9: start_9 + hold_9_1 + 2 hold_9_2 + 3 hold_9_3
   + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7 + 8 hold_9_8
   + 9 hold_9_9 + 10 hold_9_10 + 11 hold_9_11 + 12 hold_9_12 + 13 hold_9_13
   + 14 hold_9_14 + 15 hold_9_15 + 16 hold_9_16 + 17 hold_9_17
   + 18 hold_9_18 + 19 hold_9_19 + 20 hold_9_20 + 21 hold_9_21
   + 22 hold_9_22 + 23 hold_9_23 + 24 hold_9_24 <= 80
 node_flow_constraint_nic_in_0: flow_source_nic_in_0 - flow_nic_in_0_0
   - flow_nic_in_0_1 - flow_nic_in_0_2 - flow_nic_in_0_3
   + flow_nic_out_4_nic_in_0 + flow_nic_out_6_nic_in_0
   + flow_nic_out_8_nic_in_0 + flow_nic_out_9_nic_in_0 = 0
 node_flow_constraint_nic_out_0: - flow_nic_out_0_sink
   - flow_nic_out_0_nic_in_4 - flow_nic_out_0_nic_in_6
   - flow_nic_out_0_nic_in_8 - flow_nic_out_0_nic_in_9 + flow_0_nic_out_0
   + flow_1_nic_out_0 + flow_2_nic_out_0 + flow_3_nic_out_0 = 0
 node_flow_constraint_nic_in_4: flow_nic_out_0_nic_in_4
   + flow_source_nic_in_4 - flow_nic_in_4_4 - flow_nic_in_4_5
   + flow_nic_out_6_nic_in_4 + flow_nic_out_8_nic_in_4
   + flow_nic_out_9_nic_in_4 = 0
 node_flow_constraint_nic_out_4: - flow_nic_out_4_sink
   - flow_nic_out_4_nic_in_0 - flow_nic_out_4_nic_in_6
   - flow_nic_out_4_nic_in_8 - flow_nic_out_4_nic_in_9 + flow_4_nic_out_4
   + flow_5_nic_out_4 = 0
 node_flow_constraint_nic_in_6: flow_nic_out_0_nic_in_6
   + flow_nic_out_4_nic_in_6 + flow_source_nic_in_6 - flow_nic_in_6_6
   - flow_nic_in_6_7 + flow_nic_out_8_nic_in_6 + flow_nic_out_9_nic_in_6
   = 0
 node_flow_constraint_nic_out_6: - flow_nic_out_6_sink
   - flow_nic_out_6_nic_in_0 - flow_nic_out_6_nic_in_4
   - flow_nic_out_6_nic_in_8 - flow_nic_out_6_nic_in_9 + flow_6_nic_out_6
   + flow_7_nic_out_6 = 0
 node_flow_constraint_nic_in_8: flow_nic_out_0_nic_in_8
   + flow_nic_out_4_nic_in_8 + flow_nic_out_6_nic_in_8
   + flow_source_nic_in_8 - flow_nic_in_8_8 + flow_nic_out_9_nic_in_8 = 0
 node_flow_constraint_nic_out_8: - flow_nic_out_8_sink
   - flow_nic_out_8_nic_in_0 - flow_nic_out_8_nic_in_4
   - flow_nic_out_8_nic_in_6 - flow_nic_out_8_nic_in_9 + flow_8_nic_out_8
   = 0
 node_flow_constraint_nic_in_9: flow_nic_out_0_nic_in_9
   + flow_nic_out_4_nic_in_9 + flow_nic_out_6_nic_in_9
   + flow_nic_out_8_nic_in_9 + flow_source_nic_in_9 - flow_nic_in_9_9 = 0
 node_flow_constraint_nic_out_9: - flow_nic_out_9_sink
   - flow_nic_out_9_nic_in_0 - flow_nic_out_9_nic_in_4
   - flow_nic_out_9_nic_in_6 - flow_nic_out_9_nic_in_8 + flow_9_nic_out_9
   = 0
 node_flow_constraint_0: flow_nic_in_0_0 - flow_0_nic_out_0 - flow_0_1
   - flow_0_2 - flow_0_3 + flow_1_0 + flow_2_0 + flow_3_0 = 0
 node_flow_constraint_1: flow_nic_in_0_1 - flow_1_nic_out_0 + flow_0_1
   - flow_1_0 - flow_1_2 - flow_1_3 + flow_2_1 + flow_3_1 = 0
 node_flow_constraint_2: flow_nic_in_0_2 - flow_2_nic_out_0 + flow_0_2
   + flow_1_2 - flow_2_0 - flow_2_1 - flow_2_3 + flow_3_2 = 0
 node_flow_constraint_3: flow_nic_in_0_3 - flow_3_nic_out_0 + flow_0_3
   + flow_1_3 + flow_2_3 - flow_3_0 - flow_3_1 - flow_3_2 = 0
 node_flow_constraint_4: flow_nic_in_4_4 - flow_4_nic_out_4 - flow_4_5
   + flow_5_4 = 0
 node_flow_constraint_5: flow_nic_in_4_5 - flow_5_nic_out_4 + flow_4_5
   - flow_5_4 = 0
 node_flow_constraint_6: flow_nic_in_6_6 - flow_6_nic_out_6 - flow_6_7
   + flow_7_6 = 0
 node_flow_constraint_7: flow_nic_in_6_7 - flow_7_nic_out_6 + flow_6_7
   - flow_7_6 = 0
 node_flow_constraint_8: flow_nic_in_8_8 - flow_8_nic_out_8 = 0
 node_flow_constraint_9: flow_nic_in_9_9 - flow_9_nic_out_9 = 0
 constr_is_start_zero_0: start_0 + 80 is_start_zero_0_for_source_nic_in_0
   <= 80
 constr_is_start_zero_1: start_1 + 80 is_start_zero_1_for_source_nic_in_0
   <= 80
 constr_is_start_zero_2: start_2 + 80 is_start_zero_2_for_source_nic_in_0
   <= 80
 constr_is_start_zero_3: start_3 + 80 is_start_zero_3_for_source_nic_in_0
   <= 80
 constr_edge_switch_start_zero_0_OR: switch_source_nic_in_0
   - is_start_zero_0_for_source_nic_in_0
   - is_start_zero_1_for_source_nic_in_0
   - is_start_zero_2_for_source_nic_in_0
   - is_start_zero_3_for_source_nic_in_0 <= 0
 constr_is_end_complete_0: start_0 + hold_0_1 + 2 hold_0_2 + 3 hold_0_3
   + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7 + 8 hold_0_8
   + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12 + 13 hold_0_13
   + 14 hold_0_14 + 15 hold_0_15 + 16 hold_0_16 + 17 hold_0_17
   + 18 hold_0_18 + 19 hold_0_19 + 20 hold_0_20 + 21 hold_0_21
   + 22 hold_0_22 + 23 hold_0_23 + 24 hold_0_24 - 80 is_end_complete_0
   >= 0
 constr_is_end_complete_1: start_1 + hold_1_1 + 2 hold_1_2 + 3 hold_1_3
   + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7 + 8 hold_1_8
   + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 + 12 hold_1_12 + 13 hold_1_13
   + 14 hold_1_14 + 15 hold_1_15 + 16 hold_1_16 + 17 hold_1_17
   + 18 hold_1_18 + 19 hold_1_19 + 20 hold_1_20 + 21 hold_1_21
   + 22 hold_1_22 + 23 hold_1_23 + 24 hold_1_24 - 80 is_end_complete_1
   >= 0
 constr_is_end_complete_2: start_2 + hold_2_1 + 2 hold_2_2 + 3 hold_2_3
   + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7 + 8 hold_2_8
   + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 12 hold_2_12 + 13 hold_2_13
   + 14 hold_2_14 + 15 hold_2_15 + 16 hold_2_16 + 17 hold_2_17
   + 18 hold_2_18 + 19 hold_2_19 + 20 hold_2_20 + 21 hold_2_21
   + 22 hold_2_22 + 23 hold_2_23 + 24 hold_2_24 - 80 is_end_complete_2
   >= 0
 constr_is_end_complete_3: start_3 + hold_3_1 + 2 hold_3_2 + 3 hold_3_3
   + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7 + 8 hold_3_8
   + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11 + 12 hold_3_12 + 13 hold_3_13
   + 14 hold_3_14 + 15 hold_3_15 + 16 hold_3_16 + 17 hold_3_17
   + 18 hold_3_18 + 19 hold_3_19 + 20 hold_3_20 + 21 hold_3_21
   + 22 hold_3_22 + 23 hold_3_23 + 24 hold_3_24 - 80 is_end_complete_3
   >= 0
 constr_edge_switch_end_max_0_OR: switch_nic_out_0_sink - is_end_complete_0
   - is_end_complete_1 - is_end_complete_2 - is_end_complete_3 <= 0
 constr_link_match_1_0_4: 80 switch_nic_out_0_nic_in_4 - group_start_4
   + group_end_0 <= 80
 constr_link_match_2_0_4: 80 switch_nic_out_0_nic_in_4 + group_start_4
   - group_end_0 <= 80
 constr_link_match_1_0_6: 80 switch_nic_out_0_nic_in_6 + group_end_0
   - group_start_6 <= 80
 constr_link_match_2_0_6: 80 switch_nic_out_0_nic_in_6 - group_end_0
   + group_start_6 <= 80
 constr_link_match_1_0_8: 80 switch_nic_out_0_nic_in_8 + group_end_0
   - group_start_8 <= 80
 constr_link_match_2_0_8: 80 switch_nic_out_0_nic_in_8 - group_end_0
   + group_start_8 <= 80
 constr_link_match_1_0_9: 80 switch_nic_out_0_nic_in_9 + group_end_0
   - group_start_9 <= 80
 constr_link_match_2_0_9: 80 switch_nic_out_0_nic_in_9 - group_end_0
   + group_start_9 <= 80
 constr_pp_permissionA_0_1: start_0 - start_1 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 8 hold_0_8 + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12
   + 13 hold_0_13 + 14 hold_0_14 + 15 hold_0_15 + 16 hold_0_16
   + 17 hold_0_17 + 18 hold_0_18 + 19 hold_0_19 + 20 hold_0_20
   + 21 hold_0_21 + 22 hold_0_22 + 23 hold_0_23 + 24 hold_0_24
   + 80 pp_permission_0_1 <= 80
 constr_pp_permissionB_0_1: - start_0 + start_1 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11 - 12 hold_0_12
   - 13 hold_0_13 - 14 hold_0_14 - 15 hold_0_15 - 16 hold_0_16
   - 17 hold_0_17 - 18 hold_0_18 - 19 hold_0_19 - 20 hold_0_20
   - 21 hold_0_21 - 22 hold_0_22 - 23 hold_0_23 - 24 hold_0_24
   + 80 pp_permission_0_1 <= 80
 constr_internal_compute_0_1: switch_0_1 - group_tp_active_0
   - pp_permission_0_1 <= 0
 constr_pp_permissionA_0_2: start_0 - start_2 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 8 hold_0_8 + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12
   + 13 hold_0_13 + 14 hold_0_14 + 15 hold_0_15 + 16 hold_0_16
   + 17 hold_0_17 + 18 hold_0_18 + 19 hold_0_19 + 20 hold_0_20
   + 21 hold_0_21 + 22 hold_0_22 + 23 hold_0_23 + 24 hold_0_24
   + 80 pp_permission_0_2 <= 80
 constr_pp_permissionB_0_2: - start_0 + start_2 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11 - 12 hold_0_12
   - 13 hold_0_13 - 14 hold_0_14 - 15 hold_0_15 - 16 hold_0_16
   - 17 hold_0_17 - 18 hold_0_18 - 19 hold_0_19 - 20 hold_0_20
   - 21 hold_0_21 - 22 hold_0_22 - 23 hold_0_23 - 24 hold_0_24
   + 80 pp_permission_0_2 <= 80
 constr_internal_compute_0_2: switch_0_2 - group_tp_active_0
   - pp_permission_0_2 <= 0
 constr_pp_permissionA_0_3: start_0 - start_3 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 8 hold_0_8 + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12
   + 13 hold_0_13 + 14 hold_0_14 + 15 hold_0_15 + 16 hold_0_16
   + 17 hold_0_17 + 18 hold_0_18 + 19 hold_0_19 + 20 hold_0_20
   + 21 hold_0_21 + 22 hold_0_22 + 23 hold_0_23 + 24 hold_0_24
   + 80 pp_permission_0_3 <= 80
 constr_pp_permissionB_0_3: - start_0 + start_3 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11 - 12 hold_0_12
   - 13 hold_0_13 - 14 hold_0_14 - 15 hold_0_15 - 16 hold_0_16
   - 17 hold_0_17 - 18 hold_0_18 - 19 hold_0_19 - 20 hold_0_20
   - 21 hold_0_21 - 22 hold_0_22 - 23 hold_0_23 - 24 hold_0_24
   + 80 pp_permission_0_3 <= 80
 constr_internal_compute_0_3: switch_0_3 - group_tp_active_0
   - pp_permission_0_3 <= 0
 constr_pp_permissionA_1_0: - start_0 + start_1 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 8 hold_1_8 + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 + 12 hold_1_12
   + 13 hold_1_13 + 14 hold_1_14 + 15 hold_1_15 + 16 hold_1_16
   + 17 hold_1_17 + 18 hold_1_18 + 19 hold_1_19 + 20 hold_1_20
   + 21 hold_1_21 + 22 hold_1_22 + 23 hold_1_23 + 24 hold_1_24
   + 80 pp_permission_1_0 <= 80
 constr_pp_permissionB_1_0: start_0 - start_1 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11 - 12 hold_1_12
   - 13 hold_1_13 - 14 hold_1_14 - 15 hold_1_15 - 16 hold_1_16
   - 17 hold_1_17 - 18 hold_1_18 - 19 hold_1_19 - 20 hold_1_20
   - 21 hold_1_21 - 22 hold_1_22 - 23 hold_1_23 - 24 hold_1_24
   + 80 pp_permission_1_0 <= 80
 constr_internal_compute_1_0: switch_1_0 - group_tp_active_0
   - pp_permission_1_0 <= 0
 constr_pp_permissionA_1_2: start_1 - start_2 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 8 hold_1_8 + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 + 12 hold_1_12
   + 13 hold_1_13 + 14 hold_1_14 + 15 hold_1_15 + 16 hold_1_16
   + 17 hold_1_17 + 18 hold_1_18 + 19 hold_1_19 + 20 hold_1_20
   + 21 hold_1_21 + 22 hold_1_22 + 23 hold_1_23 + 24 hold_1_24
   + 80 pp_permission_1_2 <= 80
 constr_pp_permissionB_1_2: - start_1 + start_2 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11 - 12 hold_1_12
   - 13 hold_1_13 - 14 hold_1_14 - 15 hold_1_15 - 16 hold_1_16
   - 17 hold_1_17 - 18 hold_1_18 - 19 hold_1_19 - 20 hold_1_20
   - 21 hold_1_21 - 22 hold_1_22 - 23 hold_1_23 - 24 hold_1_24
   + 80 pp_permission_1_2 <= 80
 constr_internal_compute_1_2: switch_1_2 - group_tp_active_0
   - pp_permission_1_2 <= 0
 constr_pp_permissionA_1_3: start_1 - start_3 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 8 hold_1_8 + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 + 12 hold_1_12
   + 13 hold_1_13 + 14 hold_1_14 + 15 hold_1_15 + 16 hold_1_16
   + 17 hold_1_17 + 18 hold_1_18 + 19 hold_1_19 + 20 hold_1_20
   + 21 hold_1_21 + 22 hold_1_22 + 23 hold_1_23 + 24 hold_1_24
   + 80 pp_permission_1_3 <= 80
 constr_pp_permissionB_1_3: - start_1 + start_3 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11 - 12 hold_1_12
   - 13 hold_1_13 - 14 hold_1_14 - 15 hold_1_15 - 16 hold_1_16
   - 17 hold_1_17 - 18 hold_1_18 - 19 hold_1_19 - 20 hold_1_20
   - 21 hold_1_21 - 22 hold_1_22 - 23 hold_1_23 - 24 hold_1_24
   + 80 pp_permission_1_3 <= 80
 constr_internal_compute_1_3: switch_1_3 - group_tp_active_0
   - pp_permission_1_3 <= 0
 constr_pp_permissionA_2_0: - start_0 + start_2 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 12 hold_2_12
   + 13 hold_2_13 + 14 hold_2_14 + 15 hold_2_15 + 16 hold_2_16
   + 17 hold_2_17 + 18 hold_2_18 + 19 hold_2_19 + 20 hold_2_20
   + 21 hold_2_21 + 22 hold_2_22 + 23 hold_2_23 + 24 hold_2_24
   + 80 pp_permission_2_0 <= 80
 constr_pp_permissionB_2_0: start_0 - start_2 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 - 12 hold_2_12
   - 13 hold_2_13 - 14 hold_2_14 - 15 hold_2_15 - 16 hold_2_16
   - 17 hold_2_17 - 18 hold_2_18 - 19 hold_2_19 - 20 hold_2_20
   - 21 hold_2_21 - 22 hold_2_22 - 23 hold_2_23 - 24 hold_2_24
   + 80 pp_permission_2_0 <= 80
 constr_internal_compute_2_0: switch_2_0 - group_tp_active_0
   - pp_permission_2_0 <= 0
 constr_pp_permissionA_2_1: - start_1 + start_2 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 12 hold_2_12
   + 13 hold_2_13 + 14 hold_2_14 + 15 hold_2_15 + 16 hold_2_16
   + 17 hold_2_17 + 18 hold_2_18 + 19 hold_2_19 + 20 hold_2_20
   + 21 hold_2_21 + 22 hold_2_22 + 23 hold_2_23 + 24 hold_2_24
   + 80 pp_permission_2_1 <= 80
 constr_pp_permissionB_2_1: start_1 - start_2 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 - 12 hold_2_12
   - 13 hold_2_13 - 14 hold_2_14 - 15 hold_2_15 - 16 hold_2_16
   - 17 hold_2_17 - 18 hold_2_18 - 19 hold_2_19 - 20 hold_2_20
   - 21 hold_2_21 - 22 hold_2_22 - 23 hold_2_23 - 24 hold_2_24
   + 80 pp_permission_2_1 <= 80
 constr_internal_compute_2_1: switch_2_1 - group_tp_active_0
   - pp_permission_2_1 <= 0
 constr_pp_permissionA_2_3: start_2 - start_3 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 12 hold_2_12
   + 13 hold_2_13 + 14 hold_2_14 + 15 hold_2_15 + 16 hold_2_16
   + 17 hold_2_17 + 18 hold_2_18 + 19 hold_2_19 + 20 hold_2_20
   + 21 hold_2_21 + 22 hold_2_22 + 23 hold_2_23 + 24 hold_2_24
   + 80 pp_permission_2_3 <= 80
 constr_pp_permissionB_2_3: - start_2 + start_3 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 - 12 hold_2_12
   - 13 hold_2_13 - 14 hold_2_14 - 15 hold_2_15 - 16 hold_2_16
   - 17 hold_2_17 - 18 hold_2_18 - 19 hold_2_19 - 20 hold_2_20
   - 21 hold_2_21 - 22 hold_2_22 - 23 hold_2_23 - 24 hold_2_24
   + 80 pp_permission_2_3 <= 80
 constr_internal_compute_2_3: switch_2_3 - group_tp_active_0
   - pp_permission_2_3 <= 0
 constr_pp_permissionA_3_0: - start_0 + start_3 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 8 hold_3_8 + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11 + 12 hold_3_12
   + 13 hold_3_13 + 14 hold_3_14 + 15 hold_3_15 + 16 hold_3_16
   + 17 hold_3_17 + 18 hold_3_18 + 19 hold_3_19 + 20 hold_3_20
   + 21 hold_3_21 + 22 hold_3_22 + 23 hold_3_23 + 24 hold_3_24
   + 80 pp_permission_3_0 <= 80
 constr_pp_permissionB_3_0: start_0 - start_3 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   - 8 hold_3_8 - 9 hold_3_9 - 10 hold_3_10 - 11 hold_3_11 - 12 hold_3_12
   - 13 hold_3_13 - 14 hold_3_14 - 15 hold_3_15 - 16 hold_3_16
   - 17 hold_3_17 - 18 hold_3_18 - 19 hold_3_19 - 20 hold_3_20
   - 21 hold_3_21 - 22 hold_3_22 - 23 hold_3_23 - 24 hold_3_24
   + 80 pp_permission_3_0 <= 80
 constr_internal_compute_3_0: switch_3_0 - group_tp_active_0
   - pp_permission_3_0 <= 0
 constr_pp_permissionA_3_1: - start_1 + start_3 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 8 hold_3_8 + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11 + 12 hold_3_12
   + 13 hold_3_13 + 14 hold_3_14 + 15 hold_3_15 + 16 hold_3_16
   + 17 hold_3_17 + 18 hold_3_18 + 19 hold_3_19 + 20 hold_3_20
   + 21 hold_3_21 + 22 hold_3_22 + 23 hold_3_23 + 24 hold_3_24
   + 80 pp_permission_3_1 <= 80
 constr_pp_permissionB_3_1: start_1 - start_3 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   - 8 hold_3_8 - 9 hold_3_9 - 10 hold_3_10 - 11 hold_3_11 - 12 hold_3_12
   - 13 hold_3_13 - 14 hold_3_14 - 15 hold_3_15 - 16 hold_3_16
   - 17 hold_3_17 - 18 hold_3_18 - 19 hold_3_19 - 20 hold_3_20
   - 21 hold_3_21 - 22 hold_3_22 - 23 hold_3_23 - 24 hold_3_24
   + 80 pp_permission_3_1 <= 80
 constr_internal_compute_3_1: switch_3_1 - group_tp_active_0
   - pp_permission_3_1 <= 0
 constr_pp_permissionA_3_2: - start_2 + start_3 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 8 hold_3_8 + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11 + 12 hold_3_12
   + 13 hold_3_13 + 14 hold_3_14 + 15 hold_3_15 + 16 hold_3_16
   + 17 hold_3_17 + 18 hold_3_18 + 19 hold_3_19 + 20 hold_3_20
   + 21 hold_3_21 + 22 hold_3_22 + 23 hold_3_23 + 24 hold_3_24
   + 80 pp_permission_3_2 <= 80
 constr_pp_permissionB_3_2: start_2 - start_3 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   - 8 hold_3_8 - 9 hold_3_9 - 10 hold_3_10 - 11 hold_3_11 - 12 hold_3_12
   - 13 hold_3_13 - 14 hold_3_14 - 15 hold_3_15 - 16 hold_3_16
   - 17 hold_3_17 - 18 hold_3_18 - 19 hold_3_19 - 20 hold_3_20
   - 21 hold_3_21 - 22 hold_3_22 - 23 hold_3_23 - 24 hold_3_24
   + 80 pp_permission_3_2 <= 80
 constr_internal_compute_3_2: switch_3_2 - group_tp_active_0
   - pp_permission_3_2 <= 0
 constr_is_start_zero_4: start_4 + 80 is_start_zero_4_for_source_nic_in_4
   <= 80
 constr_is_start_zero_5: start_5 + 80 is_start_zero_5_for_source_nic_in_4
   <= 80
 constr_edge_switch_start_zero_4_OR: switch_source_nic_in_4
   - is_start_zero_4_for_source_nic_in_4
   - is_start_zero_5_for_source_nic_in_4 <= 0
 constr_is_end_complete_4: start_4 + hold_4_1 + 2 hold_4_2 + 3 hold_4_3
   + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7 + 8 hold_4_8
   + 9 hold_4_9 + 10 hold_4_10 + 11 hold_4_11 + 12 hold_4_12 + 13 hold_4_13
   + 14 hold_4_14 + 15 hold_4_15 + 16 hold_4_16 + 17 hold_4_17
   + 18 hold_4_18 + 19 hold_4_19 + 20 hold_4_20 + 21 hold_4_21
   + 22 hold_4_22 + 23 hold_4_23 + 24 hold_4_24 - 80 is_end_complete_4
   >= 0
 constr_is_end_complete_5: start_5 + hold_5_1 + 2 hold_5_2 + 3 hold_5_3
   + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7 + 8 hold_5_8
   + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12 + 13 hold_5_13
   + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16 + 17 hold_5_17
   + 18 hold_5_18 + 19 hold_5_19 + 20 hold_5_20 + 21 hold_5_21
   + 22 hold_5_22 + 23 hold_5_23 + 24 hold_5_24 - 80 is_end_complete_5
   >= 0
 constr_edge_switch_end_max_4_OR: switch_nic_out_4_sink - is_end_complete_4
   - is_end_complete_5 <= 0
 constr_link_match_1_4_0: 80 switch_nic_out_4_nic_in_0 - group_start_0
   + group_end_4 <= 80
 constr_link_match_2_4_0: 80 switch_nic_out_4_nic_in_0 + group_start_0
   - group_end_4 <= 80
 constr_link_match_1_4_6: 80 switch_nic_out_4_nic_in_6 - group_start_6
   + group_end_4 <= 80
 constr_link_match_2_4_6: 80 switch_nic_out_4_nic_in_6 + group_start_6
   - group_end_4 <= 80
 constr_link_match_1_4_8: 80 switch_nic_out_4_nic_in_8 - group_start_8
   + group_end_4 <= 80
 constr_link_match_2_4_8: 80 switch_nic_out_4_nic_in_8 + group_start_8
   - group_end_4 <= 80
 constr_link_match_1_4_9: 80 switch_nic_out_4_nic_in_9 - group_start_9
   + group_end_4 <= 80
 constr_link_match_2_4_9: 80 switch_nic_out_4_nic_in_9 + group_start_9
   - group_end_4 <= 80
 constr_pp_permissionA_4_5: start_4 - start_5 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 8 hold_4_8 + 9 hold_4_9 + 10 hold_4_10 + 11 hold_4_11 + 12 hold_4_12
   + 13 hold_4_13 + 14 hold_4_14 + 15 hold_4_15 + 16 hold_4_16
   + 17 hold_4_17 + 18 hold_4_18 + 19 hold_4_19 + 20 hold_4_20
   + 21 hold_4_21 + 22 hold_4_22 + 23 hold_4_23 + 24 hold_4_24
   + 80 pp_permission_4_5 <= 80
 constr_pp_permissionB_4_5: - start_4 + start_5 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   - 8 hold_4_8 - 9 hold_4_9 - 10 hold_4_10 - 11 hold_4_11 - 12 hold_4_12
   - 13 hold_4_13 - 14 hold_4_14 - 15 hold_4_15 - 16 hold_4_16
   - 17 hold_4_17 - 18 hold_4_18 - 19 hold_4_19 - 20 hold_4_20
   - 21 hold_4_21 - 22 hold_4_22 - 23 hold_4_23 - 24 hold_4_24
   + 80 pp_permission_4_5 <= 80
 constr_internal_compute_4_5: switch_4_5 - group_tp_active_4
   - pp_permission_4_5 <= 0
 constr_pp_permissionA_5_4: - start_4 + start_5 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12
   + 13 hold_5_13 + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16
   + 17 hold_5_17 + 18 hold_5_18 + 19 hold_5_19 + 20 hold_5_20
   + 21 hold_5_21 + 22 hold_5_22 + 23 hold_5_23 + 24 hold_5_24
   + 80 pp_permission_5_4 <= 80
 constr_pp_permissionB_5_4: start_4 - start_5 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11 - 12 hold_5_12
   - 13 hold_5_13 - 14 hold_5_14 - 15 hold_5_15 - 16 hold_5_16
   - 17 hold_5_17 - 18 hold_5_18 - 19 hold_5_19 - 20 hold_5_20
   - 21 hold_5_21 - 22 hold_5_22 - 23 hold_5_23 - 24 hold_5_24
   + 80 pp_permission_5_4 <= 80
 constr_internal_compute_5_4: switch_5_4 - group_tp_active_4
   - pp_permission_5_4 <= 0
 constr_is_start_zero_6: start_6 + 80 is_start_zero_6_for_source_nic_in_6
   <= 80
 constr_is_start_zero_7: start_7 + 80 is_start_zero_7_for_source_nic_in_6
   <= 80
 constr_edge_switch_start_zero_6_OR: switch_source_nic_in_6
   - is_start_zero_6_for_source_nic_in_6
   - is_start_zero_7_for_source_nic_in_6 <= 0
 constr_is_end_complete_6: start_6 + hold_6_1 + 2 hold_6_2 + 3 hold_6_3
   + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7 + 8 hold_6_8
   + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 12 hold_6_12 + 13 hold_6_13
   + 14 hold_6_14 + 15 hold_6_15 + 16 hold_6_16 + 17 hold_6_17
   + 18 hold_6_18 + 19 hold_6_19 + 20 hold_6_20 + 21 hold_6_21
   + 22 hold_6_22 + 23 hold_6_23 + 24 hold_6_24 - 80 is_end_complete_6
   >= 0
 constr_is_end_complete_7: start_7 + hold_7_1 + 2 hold_7_2 + 3 hold_7_3
   + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7 + 8 hold_7_8
   + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11 + 12 hold_7_12 + 13 hold_7_13
   + 14 hold_7_14 + 15 hold_7_15 + 16 hold_7_16 + 17 hold_7_17
   + 18 hold_7_18 + 19 hold_7_19 + 20 hold_7_20 + 21 hold_7_21
   + 22 hold_7_22 + 23 hold_7_23 + 24 hold_7_24 - 80 is_end_complete_7
   >= 0
 constr_edge_switch_end_max_6_OR: switch_nic_out_6_sink - is_end_complete_6
   - is_end_complete_7 <= 0
 constr_link_match_1_6_0: 80 switch_nic_out_6_nic_in_0 - group_start_0
   + group_end_6 <= 80
 constr_link_match_2_6_0: 80 switch_nic_out_6_nic_in_0 + group_start_0
   - group_end_6 <= 80
 constr_link_match_1_6_4: 80 switch_nic_out_6_nic_in_4 - group_start_4
   + group_end_6 <= 80
 constr_link_match_2_6_4: 80 switch_nic_out_6_nic_in_4 + group_start_4
   - group_end_6 <= 80
 constr_link_match_1_6_8: 80 switch_nic_out_6_nic_in_8 - group_start_8
   + group_end_6 <= 80
 constr_link_match_2_6_8: 80 switch_nic_out_6_nic_in_8 + group_start_8
   - group_end_6 <= 80
 constr_link_match_1_6_9: 80 switch_nic_out_6_nic_in_9 - group_start_9
   + group_end_6 <= 80
 constr_link_match_2_6_9: 80 switch_nic_out_6_nic_in_9 + group_start_9
   - group_end_6 <= 80
 constr_pp_permissionA_6_7: start_6 - start_7 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 12 hold_6_12
   + 13 hold_6_13 + 14 hold_6_14 + 15 hold_6_15 + 16 hold_6_16
   + 17 hold_6_17 + 18 hold_6_18 + 19 hold_6_19 + 20 hold_6_20
   + 21 hold_6_21 + 22 hold_6_22 + 23 hold_6_23 + 24 hold_6_24
   + 80 pp_permission_6_7 <= 80
 constr_pp_permissionB_6_7: - start_6 + start_7 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11 - 12 hold_6_12
   - 13 hold_6_13 - 14 hold_6_14 - 15 hold_6_15 - 16 hold_6_16
   - 17 hold_6_17 - 18 hold_6_18 - 19 hold_6_19 - 20 hold_6_20
   - 21 hold_6_21 - 22 hold_6_22 - 23 hold_6_23 - 24 hold_6_24
   + 80 pp_permission_6_7 <= 80
 constr_internal_compute_6_7: switch_6_7 - group_tp_active_6
   - pp_permission_6_7 <= 0
 constr_pp_permissionA_7_6: - start_6 + start_7 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 8 hold_7_8 + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11 + 12 hold_7_12
   + 13 hold_7_13 + 14 hold_7_14 + 15 hold_7_15 + 16 hold_7_16
   + 17 hold_7_17 + 18 hold_7_18 + 19 hold_7_19 + 20 hold_7_20
   + 21 hold_7_21 + 22 hold_7_22 + 23 hold_7_23 + 24 hold_7_24
   + 80 pp_permission_7_6 <= 80
 constr_pp_permissionB_7_6: start_6 - start_7 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   - 8 hold_7_8 - 9 hold_7_9 - 10 hold_7_10 - 11 hold_7_11 - 12 hold_7_12
   - 13 hold_7_13 - 14 hold_7_14 - 15 hold_7_15 - 16 hold_7_16
   - 17 hold_7_17 - 18 hold_7_18 - 19 hold_7_19 - 20 hold_7_20
   - 21 hold_7_21 - 22 hold_7_22 - 23 hold_7_23 - 24 hold_7_24
   + 80 pp_permission_7_6 <= 80
 constr_internal_compute_7_6: switch_7_6 - group_tp_active_6
   - pp_permission_7_6 <= 0
 constr_is_start_zero_8: start_8 + 80 is_start_zero_8_for_source_nic_in_8
   <= 80
 constr_edge_switch_start_zero_8_OR: switch_source_nic_in_8
   - is_start_zero_8_for_source_nic_in_8 <= 0
 constr_is_end_complete_8: start_8 + hold_8_1 + 2 hold_8_2 + 3 hold_8_3
   + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7 + 8 hold_8_8
   + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 12 hold_8_12 + 13 hold_8_13
   + 14 hold_8_14 + 15 hold_8_15 + 16 hold_8_16 + 17 hold_8_17
   + 18 hold_8_18 + 19 hold_8_19 + 20 hold_8_20 + 21 hold_8_21
   + 22 hold_8_22 + 23 hold_8_23 + 24 hold_8_24 - 80 is_end_complete_8
   >= 0
 constr_edge_switch_end_max_8_OR: switch_nic_out_8_sink - is_end_complete_8
   <= 0
 constr_link_match_1_8_0: 80 switch_nic_out_8_nic_in_0 - group_start_0
   + group_end_8 <= 80
 constr_link_match_2_8_0: 80 switch_nic_out_8_nic_in_0 + group_start_0
   - group_end_8 <= 80
 constr_link_match_1_8_4: 80 switch_nic_out_8_nic_in_4 - group_start_4
   + group_end_8 <= 80
 constr_link_match_2_8_4: 80 switch_nic_out_8_nic_in_4 + group_start_4
   - group_end_8 <= 80
 constr_link_match_1_8_6: 80 switch_nic_out_8_nic_in_6 - group_start_6
   + group_end_8 <= 80
 constr_link_match_2_8_6: 80 switch_nic_out_8_nic_in_6 + group_start_6
   - group_end_8 <= 80
 constr_link_match_1_8_9: 80 switch_nic_out_8_nic_in_9 - group_start_9
   + group_end_8 <= 80
 constr_link_match_2_8_9: 80 switch_nic_out_8_nic_in_9 + group_start_9
   - group_end_8 <= 80
 constr_is_start_zero_9: start_9 + 80 is_start_zero_9_for_source_nic_in_9
   <= 80
 constr_edge_switch_start_zero_9_OR: switch_source_nic_in_9
   - is_start_zero_9_for_source_nic_in_9 <= 0
 constr_is_end_complete_9: start_9 + hold_9_1 + 2 hold_9_2 + 3 hold_9_3
   + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7 + 8 hold_9_8
   + 9 hold_9_9 + 10 hold_9_10 + 11 hold_9_11 + 12 hold_9_12 + 13 hold_9_13
   + 14 hold_9_14 + 15 hold_9_15 + 16 hold_9_16 + 17 hold_9_17
   + 18 hold_9_18 + 19 hold_9_19 + 20 hold_9_20 + 21 hold_9_21
   + 22 hold_9_22 + 23 hold_9_23 + 24 hold_9_24 - 80 is_end_complete_9
   >= 0
 constr_edge_switch_end_max_9_OR: switch_nic_out_9_sink - is_end_complete_9
   <= 0
 constr_link_match_1_9_0: 80 switch_nic_out_9_nic_in_0 - group_start_0
   + group_end_9 <= 80
 constr_link_match_2_9_0: 80 switch_nic_out_9_nic_in_0 + group_start_0
   - group_end_9 <= 80
 constr_link_match_1_9_4: 80 switch_nic_out_9_nic_in_4 - group_start_4
   + group_end_9 <= 80
 constr_link_match_2_9_4: 80 switch_nic_out_9_nic_in_4 + group_start_4
   - group_end_9 <= 80
 constr_link_match_1_9_6: 80 switch_nic_out_9_nic_in_6 - group_start_6
   + group_end_9 <= 80
 constr_link_match_2_9_6: 80 switch_nic_out_9_nic_in_6 + group_start_6
   - group_end_9 <= 80
 constr_link_match_1_9_8: 80 switch_nic_out_9_nic_in_8 - group_start_8
   + group_end_9 <= 80
 constr_link_match_2_9_8: 80 switch_nic_out_9_nic_in_8 + group_start_8
   - group_end_9 <= 80
 link_activation_dst_nic_in_0_0: switch_nic_in_0_0 - is_active_0 <= 0
 link_activation_src_0_nic_out_0: switch_0_nic_out_0 - is_active_0 <= 0
 link_activation_dst_nic_in_0_1: switch_nic_in_0_1 - is_active_1 <= 0
 link_activation_src_1_nic_out_0: switch_1_nic_out_0 - is_active_1 <= 0
 link_activation_dst_nic_in_0_2: switch_nic_in_0_2 - is_active_2 <= 0
 link_activation_src_2_nic_out_0: switch_2_nic_out_0 - is_active_2 <= 0
 link_activation_dst_nic_in_0_3: switch_nic_in_0_3 - is_active_3 <= 0
 link_activation_src_3_nic_out_0: switch_3_nic_out_0 - is_active_3 <= 0
 link_activation_src_0_1: switch_0_1 - is_active_0 <= 0
 link_activation_dst_0_1: switch_0_1 - is_active_1 <= 0
 link_activation_src_0_2: switch_0_2 - is_active_0 <= 0
 link_activation_dst_0_2: switch_0_2 - is_active_2 <= 0
 link_activation_src_0_3: switch_0_3 - is_active_0 <= 0
 link_activation_dst_0_3: switch_0_3 - is_active_3 <= 0
 link_activation_src_1_0: switch_1_0 - is_active_1 <= 0
 link_activation_dst_1_0: switch_1_0 - is_active_0 <= 0
 link_activation_src_1_2: switch_1_2 - is_active_1 <= 0
 link_activation_dst_1_2: switch_1_2 - is_active_2 <= 0
 link_activation_src_1_3: switch_1_3 - is_active_1 <= 0
 link_activation_dst_1_3: switch_1_3 - is_active_3 <= 0
 link_activation_src_2_0: switch_2_0 - is_active_2 <= 0
 link_activation_dst_2_0: switch_2_0 - is_active_0 <= 0
 link_activation_src_2_1: switch_2_1 - is_active_2 <= 0
 link_activation_dst_2_1: switch_2_1 - is_active_1 <= 0
 link_activation_src_2_3: switch_2_3 - is_active_2 <= 0
 link_activation_dst_2_3: switch_2_3 - is_active_3 <= 0
 link_activation_src_3_0: switch_3_0 - is_active_3 <= 0
 link_activation_dst_3_0: switch_3_0 - is_active_0 <= 0
 link_activation_src_3_1: switch_3_1 - is_active_3 <= 0
 link_activation_dst_3_1: switch_3_1 - is_active_1 <= 0
 link_activation_src_3_2: switch_3_2 - is_active_3 <= 0
 link_activation_dst_3_2: switch_3_2 - is_active_2 <= 0
 link_activation_dst_nic_in_4_4: switch_nic_in_4_4 - is_active_4 <= 0
 link_activation_src_4_nic_out_4: switch_4_nic_out_4 - is_active_4 <= 0
 link_activation_dst_nic_in_4_5: switch_nic_in_4_5 - is_active_5 <= 0
 link_activation_src_5_nic_out_4: switch_5_nic_out_4 - is_active_5 <= 0
 link_activation_src_4_5: switch_4_5 - is_active_4 <= 0
 link_activation_dst_4_5: switch_4_5 - is_active_5 <= 0
 link_activation_src_5_4: switch_5_4 - is_active_5 <= 0
 link_activation_dst_5_4: switch_5_4 - is_active_4 <= 0
 link_activation_dst_nic_in_6_6: switch_nic_in_6_6 - is_active_6 <= 0
 link_activation_src_6_nic_out_6: switch_6_nic_out_6 - is_active_6 <= 0
 link_activation_dst_nic_in_6_7: switch_nic_in_6_7 - is_active_7 <= 0
 link_activation_src_7_nic_out_6: switch_7_nic_out_6 - is_active_7 <= 0
 link_activation_src_6_7: switch_6_7 - is_active_6 <= 0
 link_activation_dst_6_7: switch_6_7 - is_active_7 <= 0
 link_activation_src_7_6: switch_7_6 - is_active_7 <= 0
 link_activation_dst_7_6: switch_7_6 - is_active_6 <= 0
 link_activation_dst_nic_in_8_8: switch_nic_in_8_8 - is_active_8 <= 0
 link_activation_src_8_nic_out_8: switch_8_nic_out_8 - is_active_8 <= 0
 link_activation_dst_nic_in_9_9: switch_nic_in_9_9 - is_active_9 <= 0
 link_activation_src_9_nic_out_9: switch_9_nic_out_9 - is_active_9 <= 0
 edge_flow_constr_source_nic_in_0: flow_source_nic_in_0
   - 5e+08 switch_source_nic_in_0 <= 0
 edge_flow_constr_nic_out_0_sink: flow_nic_out_0_sink
   - 5e+08 switch_nic_out_0_sink <= 0
 edge_flow_constr_nic_out_0_nic_in_4: flow_nic_out_0_nic_in_4
   - 61035.15625 switch_nic_out_0_nic_in_4 <= 0
 edge_flow_constr_nic_out_0_nic_in_6: flow_nic_out_0_nic_in_6
   - 61035.15625 switch_nic_out_0_nic_in_6 <= 0
 edge_flow_constr_nic_out_0_nic_in_8: flow_nic_out_0_nic_in_8
   - 61035.15625 switch_nic_out_0_nic_in_8 <= 0
 edge_flow_constr_nic_out_0_nic_in_9: flow_nic_out_0_nic_in_9
   - 61035.15625 switch_nic_out_0_nic_in_9 <= 0
 edge_flow_constr_nic_in_0_0: flow_nic_in_0_0
   - 3.90625e+06 switch_nic_in_0_0 <= 0
 edge_flow_constr_0_nic_out_0: flow_0_nic_out_0
   - 3.90625e+06 switch_0_nic_out_0 <= 0
 edge_flow_constr_nic_in_0_1: flow_nic_in_0_1
   - 3.90625e+06 switch_nic_in_0_1 <= 0
 edge_flow_constr_1_nic_out_0: flow_1_nic_out_0
   - 3.90625e+06 switch_1_nic_out_0 <= 0
 edge_flow_constr_nic_in_0_2: flow_nic_in_0_2
   - 3.90625e+06 switch_nic_in_0_2 <= 0
 edge_flow_constr_2_nic_out_0: flow_2_nic_out_0
   - 3.90625e+06 switch_2_nic_out_0 <= 0
 edge_flow_constr_nic_in_0_3: flow_nic_in_0_3
   - 3.90625e+06 switch_nic_in_0_3 <= 0
 edge_flow_constr_3_nic_out_0: flow_3_nic_out_0
   - 3.90625e+06 switch_3_nic_out_0 <= 0
 edge_flow_constr_0_1: flow_0_1 - 3.90625e+06 switch_0_1 <= 0
 edge_flow_constr_0_2: flow_0_2 - 3.90625e+06 switch_0_2 <= 0
 edge_flow_constr_0_3: flow_0_3 - 3.90625e+06 switch_0_3 <= 0
 edge_flow_constr_1_0: flow_1_0 - 3.90625e+06 switch_1_0 <= 0
 edge_flow_constr_1_2: flow_1_2 - 3.90625e+06 switch_1_2 <= 0
 edge_flow_constr_1_3: flow_1_3 - 3.90625e+06 switch_1_3 <= 0
 edge_flow_constr_2_0: flow_2_0 - 3.90625e+06 switch_2_0 <= 0
 edge_flow_constr_2_1: flow_2_1 - 3.90625e+06 switch_2_1 <= 0
 edge_flow_constr_2_3: flow_2_3 - 3.90625e+06 switch_2_3 <= 0
 edge_flow_constr_3_0: flow_3_0 - 3.90625e+06 switch_3_0 <= 0
 edge_flow_constr_3_1: flow_3_1 - 3.90625e+06 switch_3_1 <= 0
 edge_flow_constr_3_2: flow_3_2 - 3.90625e+06 switch_3_2 <= 0
 edge_flow_constr_source_nic_in_4: flow_source_nic_in_4
   - 5e+08 switch_source_nic_in_4 <= 0
 edge_flow_constr_nic_out_4_sink: flow_nic_out_4_sink
   - 5e+08 switch_nic_out_4_sink <= 0
 edge_flow_constr_nic_out_4_nic_in_0: flow_nic_out_4_nic_in_0
   - 61035.15625 switch_nic_out_4_nic_in_0 <= 0
 edge_flow_constr_nic_out_4_nic_in_6: flow_nic_out_4_nic_in_6
   - 61035.15625 switch_nic_out_4_nic_in_6 <= 0
 edge_flow_constr_nic_out_4_nic_in_8: flow_nic_out_4_nic_in_8
   - 61035.15625 switch_nic_out_4_nic_in_8 <= 0
 edge_flow_constr_nic_out_4_nic_in_9: flow_nic_out_4_nic_in_9
   - 61035.15625 switch_nic_out_4_nic_in_9 <= 0
 edge_flow_constr_nic_in_4_4: flow_nic_in_4_4
   - 3.90625e+06 switch_nic_in_4_4 <= 0
 edge_flow_constr_4_nic_out_4: flow_4_nic_out_4
   - 3.90625e+06 switch_4_nic_out_4 <= 0
 edge_flow_constr_nic_in_4_5: flow_nic_in_4_5
   - 3.90625e+06 switch_nic_in_4_5 <= 0
 edge_flow_constr_5_nic_out_4: flow_5_nic_out_4
   - 3.90625e+06 switch_5_nic_out_4 <= 0
 edge_flow_constr_4_5: flow_4_5 - 3.90625e+06 switch_4_5 <= 0
 edge_flow_constr_5_4: flow_5_4 - 3.90625e+06 switch_5_4 <= 0
 edge_flow_constr_source_nic_in_6: flow_source_nic_in_6
   - 5e+08 switch_source_nic_in_6 <= 0
 edge_flow_constr_nic_out_6_sink: flow_nic_out_6_sink
   - 5e+08 switch_nic_out_6_sink <= 0
 edge_flow_constr_nic_out_6_nic_in_0: flow_nic_out_6_nic_in_0
   - 61035.15625 switch_nic_out_6_nic_in_0 <= 0
 edge_flow_constr_nic_out_6_nic_in_4: flow_nic_out_6_nic_in_4
   - 61035.15625 switch_nic_out_6_nic_in_4 <= 0
 edge_flow_constr_nic_out_6_nic_in_8: flow_nic_out_6_nic_in_8
   - 61035.15625 switch_nic_out_6_nic_in_8 <= 0
 edge_flow_constr_nic_out_6_nic_in_9: flow_nic_out_6_nic_in_9
   - 61035.15625 switch_nic_out_6_nic_in_9 <= 0
 edge_flow_constr_nic_in_6_6: flow_nic_in_6_6
   - 3.90625e+06 switch_nic_in_6_6 <= 0
 edge_flow_constr_6_nic_out_6: flow_6_nic_out_6
   - 3.90625e+06 switch_6_nic_out_6 <= 0
 edge_flow_constr_nic_in_6_7: flow_nic_in_6_7
   - 3.90625e+06 switch_nic_in_6_7 <= 0
 edge_flow_constr_7_nic_out_6: flow_7_nic_out_6
   - 3.90625e+06 switch_7_nic_out_6 <= 0
 edge_flow_constr_6_7: flow_6_7 - 3.90625e+06 switch_6_7 <= 0
 edge_flow_constr_7_6: flow_7_6 - 3.90625e+06 switch_7_6 <= 0
 edge_flow_constr_source_nic_in_8: flow_source_nic_in_8
   - 5e+08 switch_source_nic_in_8 <= 0
 edge_flow_constr_nic_out_8_sink: flow_nic_out_8_sink
   - 5e+08 switch_nic_out_8_sink <= 0
 edge_flow_constr_nic_out_8_nic_in_0: flow_nic_out_8_nic_in_0
   - 61035.15625 switch_nic_out_8_nic_in_0 <= 0
 edge_flow_constr_nic_out_8_nic_in_4: flow_nic_out_8_nic_in_4
   - 61035.15625 switch_nic_out_8_nic_in_4 <= 0
 edge_flow_constr_nic_out_8_nic_in_6: flow_nic_out_8_nic_in_6
   - 61035.15625 switch_nic_out_8_nic_in_6 <= 0
 edge_flow_constr_nic_out_8_nic_in_9: flow_nic_out_8_nic_in_9
   - 61035.15625 switch_nic_out_8_nic_in_9 <= 0
 edge_flow_constr_nic_in_8_8: flow_nic_in_8_8
   - 3.90625e+06 switch_nic_in_8_8 <= 0
 edge_flow_constr_8_nic_out_8: flow_8_nic_out_8
   - 3.90625e+06 switch_8_nic_out_8 <= 0
 edge_flow_constr_source_nic_in_9: flow_source_nic_in_9
   - 5e+08 switch_source_nic_in_9 <= 0
 edge_flow_constr_nic_out_9_sink: flow_nic_out_9_sink
   - 5e+08 switch_nic_out_9_sink <= 0
 edge_flow_constr_nic_out_9_nic_in_0: flow_nic_out_9_nic_in_0
   - 61035.15625 switch_nic_out_9_nic_in_0 <= 0
 edge_flow_constr_nic_out_9_nic_in_4: flow_nic_out_9_nic_in_4
   - 61035.15625 switch_nic_out_9_nic_in_4 <= 0
 edge_flow_constr_nic_out_9_nic_in_6: flow_nic_out_9_nic_in_6
   - 61035.15625 switch_nic_out_9_nic_in_6 <= 0
 edge_flow_constr_nic_out_9_nic_in_8: flow_nic_out_9_nic_in_8
   - 61035.15625 switch_nic_out_9_nic_in_8 <= 0
 edge_flow_constr_nic_in_9_9: flow_nic_in_9_9
   - 3.90625e+06 switch_nic_in_9_9 <= 0
 edge_flow_constr_9_nic_out_9: flow_9_nic_out_9
   - 3.90625e+06 switch_9_nic_out_9 <= 0
 force_group_off_constr_8: group_tp_active_8 = 0
 force_group_off_constr_9: group_tp_active_9 = 0
 R298: - group_tp_active_0 + tp_on_and_switch_source_nic_in_0_0 <= 0
 R299: - switch_source_nic_in_0 + tp_on_and_switch_source_nic_in_0_0 <= 0
 R300: - switch_source_nic_in_0 - group_tp_active_0
   + tp_on_and_switch_source_nic_in_0_0 >= -1
 R301: - group_tp_active_4 + tp_on_and_switch_nic_out_0_nic_in_4_4 <= 0
 R302: - switch_nic_out_0_nic_in_4 + tp_on_and_switch_nic_out_0_nic_in_4_4
   <= 0
 R303: - switch_nic_out_0_nic_in_4 - group_tp_active_4
   + tp_on_and_switch_nic_out_0_nic_in_4_4 >= -1
 R304: - group_tp_active_6 + tp_on_and_switch_nic_out_0_nic_in_6_6 <= 0
 R305: - switch_nic_out_0_nic_in_6 + tp_on_and_switch_nic_out_0_nic_in_6_6
   <= 0
 R306: - switch_nic_out_0_nic_in_6 - group_tp_active_6
   + tp_on_and_switch_nic_out_0_nic_in_6_6 >= -1
 R307: - group_tp_active_4 + tp_on_and_switch_source_nic_in_4_4 <= 0
 R308: - switch_source_nic_in_4 + tp_on_and_switch_source_nic_in_4_4 <= 0
 R309: - switch_source_nic_in_4 - group_tp_active_4
   + tp_on_and_switch_source_nic_in_4_4 >= -1
 R310: - group_tp_active_0 + tp_on_and_switch_nic_out_4_nic_in_0_0 <= 0
 R311: - switch_nic_out_4_nic_in_0 + tp_on_and_switch_nic_out_4_nic_in_0_0
   <= 0
 R312: - switch_nic_out_4_nic_in_0 - group_tp_active_0
   + tp_on_and_switch_nic_out_4_nic_in_0_0 >= -1
 R313: - group_tp_active_6 + tp_on_and_switch_nic_out_4_nic_in_6_6 <= 0
 R314: - switch_nic_out_4_nic_in_6 + tp_on_and_switch_nic_out_4_nic_in_6_6
   <= 0
 R315: - switch_nic_out_4_nic_in_6 - group_tp_active_6
   + tp_on_and_switch_nic_out_4_nic_in_6_6 >= -1
 R316: - group_tp_active_6 + tp_on_and_switch_source_nic_in_6_6 <= 0
 R317: - switch_source_nic_in_6 + tp_on_and_switch_source_nic_in_6_6 <= 0
 R318: - switch_source_nic_in_6 - group_tp_active_6
   + tp_on_and_switch_source_nic_in_6_6 >= -1
 R319: - group_tp_active_0 + tp_on_and_switch_nic_out_6_nic_in_0_0 <= 0
 R320: - switch_nic_out_6_nic_in_0 + tp_on_and_switch_nic_out_6_nic_in_0_0
   <= 0
 R321: - switch_nic_out_6_nic_in_0 - group_tp_active_0
   + tp_on_and_switch_nic_out_6_nic_in_0_0 >= -1
 R322: - group_tp_active_4 + tp_on_and_switch_nic_out_6_nic_in_4_4 <= 0
 R323: - switch_nic_out_6_nic_in_4 + tp_on_and_switch_nic_out_6_nic_in_4_4
   <= 0
 R324: - switch_nic_out_6_nic_in_4 - group_tp_active_4
   + tp_on_and_switch_nic_out_6_nic_in_4_4 >= -1
 R325: - group_tp_active_0 + tp_on_and_switch_nic_out_8_nic_in_0_0 <= 0
 R326: - switch_nic_out_8_nic_in_0 + tp_on_and_switch_nic_out_8_nic_in_0_0
   <= 0
 R327: - switch_nic_out_8_nic_in_0 - group_tp_active_0
   + tp_on_and_switch_nic_out_8_nic_in_0_0 >= -1
 R328: - group_tp_active_4 + tp_on_and_switch_nic_out_8_nic_in_4_4 <= 0
 R329: - switch_nic_out_8_nic_in_4 + tp_on_and_switch_nic_out_8_nic_in_4_4
   <= 0
 R330: - switch_nic_out_8_nic_in_4 - group_tp_active_4
   + tp_on_and_switch_nic_out_8_nic_in_4_4 >= -1
 R331: - group_tp_active_6 + tp_on_and_switch_nic_out_8_nic_in_6_6 <= 0
 R332: - switch_nic_out_8_nic_in_6 + tp_on_and_switch_nic_out_8_nic_in_6_6
   <= 0
 R333: - switch_nic_out_8_nic_in_6 - group_tp_active_6
   + tp_on_and_switch_nic_out_8_nic_in_6_6 >= -1
 R334: - group_tp_active_0 + tp_on_and_switch_nic_out_9_nic_in_0_0 <= 0
 R335: - switch_nic_out_9_nic_in_0 + tp_on_and_switch_nic_out_9_nic_in_0_0
   <= 0
 R336: - switch_nic_out_9_nic_in_0 - group_tp_active_0
   + tp_on_and_switch_nic_out_9_nic_in_0_0 >= -1
 R337: - group_tp_active_4 + tp_on_and_switch_nic_out_9_nic_in_4_4 <= 0
 R338: - switch_nic_out_9_nic_in_4 + tp_on_and_switch_nic_out_9_nic_in_4_4
   <= 0
 R339: - switch_nic_out_9_nic_in_4 - group_tp_active_4
   + tp_on_and_switch_nic_out_9_nic_in_4_4 >= -1
 R340: - group_tp_active_6 + tp_on_and_switch_nic_out_9_nic_in_6_6 <= 0
 R341: - switch_nic_out_9_nic_in_6 + tp_on_and_switch_nic_out_9_nic_in_6_6
   <= 0
 R342: - switch_nic_out_9_nic_in_6 - group_tp_active_6
   + tp_on_and_switch_nic_out_9_nic_in_6_6 >= -1
 force_min_throughput: flow_source_nic_in_0 + flow_source_nic_in_4
   + flow_source_nic_in_6 + flow_source_nic_in_8 + flow_source_nic_in_9
   >= 1
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
 node_throughput_constraint_0: group_tp_active_0 = 0 ->
   - 4265.142857142857 hold_0_1 - 2132.571428571428 hold_0_2
   - 1421.714285714286 hold_0_3 - 1066.285714285714 hold_0_4
   - 853.0285714285714 hold_0_5 - 710.8571428571429 hold_0_6
   + flow_nic_in_0_0 + flow_1_0 + flow_2_0 + flow_3_0 <= 0
 node_throughput_constraint_1: group_tp_active_0 = 0 ->
   - 4265.142857142857 hold_1_1 - 2132.571428571428 hold_1_2
   - 1421.714285714286 hold_1_3 - 1066.285714285714 hold_1_4
   - 853.0285714285714 hold_1_5 - 710.8571428571429 hold_1_6
   + flow_nic_in_0_1 + flow_0_1 + flow_2_1 + flow_3_1 <= 0
 node_throughput_constraint_2: group_tp_active_0 = 0 ->
   - 4265.142857142857 hold_2_1 - 2132.571428571428 hold_2_2
   - 1421.714285714286 hold_2_3 - 1066.285714285714 hold_2_4
   - 853.0285714285714 hold_2_5 - 710.8571428571429 hold_2_6
   + flow_nic_in_0_2 + flow_0_2 + flow_1_2 + flow_3_2 <= 0
 node_throughput_constraint_3: group_tp_active_0 = 0 ->
   - 4265.142857142857 hold_3_1 - 2132.571428571428 hold_3_2
   - 1421.714285714286 hold_3_3 - 1066.285714285714 hold_3_4
   - 853.0285714285714 hold_3_5 - 710.8571428571429 hold_3_6
   + flow_nic_in_0_3 + flow_0_3 + flow_1_3 + flow_2_3 <= 0
 node_throughput_constraint_4: group_tp_active_4 = 0 ->
   - 4265.142857142857 hold_4_1 - 2132.571428571428 hold_4_2
   - 1421.714285714286 hold_4_3 - 1066.285714285714 hold_4_4
   - 853.0285714285714 hold_4_5 - 710.8571428571429 hold_4_6
   + flow_nic_in_4_4 + flow_5_4 <= 0
 node_throughput_constraint_5: group_tp_active_4 = 0 ->
   - 4265.142857142857 hold_5_1 - 2132.571428571428 hold_5_2
   - 1421.714285714286 hold_5_3 - 1066.285714285714 hold_5_4
   - 853.0285714285714 hold_5_5 - 710.8571428571429 hold_5_6
   + flow_nic_in_4_5 + flow_4_5 <= 0
 node_throughput_constraint_6: group_tp_active_6 = 0 ->
   - 4265.142857142857 hold_6_1 - 2132.571428571428 hold_6_2
   - 1421.714285714286 hold_6_3 - 1066.285714285714 hold_6_4
   - 853.0285714285714 hold_6_5 - 710.8571428571429 hold_6_6
   + flow_nic_in_6_6 + flow_7_6 <= 0
 node_throughput_constraint_7: group_tp_active_6 = 0 ->
   - 4265.142857142857 hold_7_1 - 2132.571428571428 hold_7_2
   - 1421.714285714286 hold_7_3 - 1066.285714285714 hold_7_4
   - 853.0285714285714 hold_7_5 - 710.8571428571429 hold_7_6
   + flow_nic_in_6_7 + flow_6_7 <= 0
 node_throughput_constraint_8: group_tp_active_8 = 0 ->
   - 9325.714285714286 hold_8_1 - 4662.857142857143 hold_8_2
   - 3108.571428571428 hold_8_3 - 2331.428571428572 hold_8_4
   - 1865.142857142857 hold_8_5 - 1554.285714285714 hold_8_6
   - 1332.244897959184 hold_8_7 - 1165.714285714286 hold_8_8
   - 1036.190476190476 hold_8_9 - 932.5714285714284 hold_8_10
   - 847.7922077922078 hold_8_11 - 777.1428571428571 hold_8_12
   - 717.3626373626373 hold_8_13 - 666.1224489795918 hold_8_14
   - 621.7142857142858 hold_8_15 - 582.8571428571429 hold_8_16
   - 548.5714285714286 hold_8_17 - 518.0952380952381 hold_8_18
   - 490.8270676691729 hold_8_19 - 466.2857142857142 hold_8_20
   - 444.0816326530613 hold_8_21 - 423.8961038961039 hold_8_22
   + flow_nic_in_8_8 <= 0
 node_throughput_constraint_9: group_tp_active_9 = 0 ->
   - 9325.714285714286 hold_9_1 - 4662.857142857143 hold_9_2
   - 3108.571428571428 hold_9_3 - 2331.428571428572 hold_9_4
   - 1865.142857142857 hold_9_5 - 1554.285714285714 hold_9_6
   - 1332.244897959184 hold_9_7 - 1165.714285714286 hold_9_8
   - 1036.190476190476 hold_9_9 - 932.5714285714284 hold_9_10
   - 847.7922077922078 hold_9_11 - 777.1428571428571 hold_9_12
   - 717.3626373626373 hold_9_13 - 666.1224489795918 hold_9_14
   - 621.7142857142858 hold_9_15 - 582.8571428571429 hold_9_16
   - 548.5714285714286 hold_9_17 - 518.0952380952381 hold_9_18
   - 490.8270676691729 hold_9_19 - 466.2857142857142 hold_9_20
   - 444.0816326530613 hold_9_21 - 423.8961038961039 hold_9_22
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_0_source_nic_in_0: switch_source_nic_in_0 = 1 ->
   - 4049.259480279933 hold_0_1 - 2077.199231903821 hold_0_2
   - 1396.889562629835 hold_0_3 - 1052.260583931315 hold_0_4
   - 844.0288127735122 hold_0_5 - 704.5963014358132 hold_0_6
   + flow_nic_in_0_0 + flow_1_0 + flow_2_0 + flow_3_0 <= 0
 constr_intergroup_throughput_4_nic_out_0_nic_in_4: switch_nic_out_0_nic_in_4 = 1 ->
   - 3367.47123843898 hold_4_1 - 1881.759737804109 hold_4_2
   - 1305.694043558121 hold_4_3 - 999.665171097569 hold_4_4
   - 809.8518960559865 hold_4_5 - 680.6182464779099 hold_4_6
   + flow_nic_in_4_4 + flow_5_4 <= 0
 constr_intergroup_throughput_6_nic_out_0_nic_in_6: switch_nic_out_0_nic_in_6 = 1 ->
   - 3367.47123843898 hold_6_1 - 1881.759737804109 hold_6_2
   - 1305.694043558121 hold_6_3 - 999.665171097569 hold_6_4
   - 809.8518960559865 hold_6_5 - 680.6182464779099 hold_6_6
   + flow_nic_in_6_6 + flow_7_6 <= 0
 constr_intergroup_throughput_8_nic_out_0_nic_in_8: switch_nic_out_0_nic_in_8 = 1 ->
   - 5891.696750902527 hold_8_1 - 3610.619469026549 hold_8_2
   - 2602.870813397129 hold_8_3 - 2034.912718204489 hold_8_4
   - 1670.419651995905 hold_8_5 - 1416.666666666667 hold_8_6
   - 1229.841748304446 hold_8_7 - 1086.551264980027 hold_8_8
   - 973.1663685152056 hold_8_9 - 881.2095032397407 hold_8_10
   - 805.1307350764677 hold_8_11 - 741.1444141689373 hold_8_12
   - 686.5797223390829 hold_8_13 - 639.4984326018808 hold_8_14
   - 598.4598459845985 hold_8_15 - 562.3707787732598 hold_8_16
   - 530.3867403314916 hold_8_17 - 501.8450184501845 hold_8_18
   - 476.2182667055733 hold_8_19 - 453.081621321488 hold_8_20
   - 432.0889594916601 hold_8_21 - 412.9554655870446 hold_8_22
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_0_nic_in_9: switch_nic_out_0_nic_in_9 = 1 ->
   - 5891.696750902527 hold_9_1 - 3610.619469026549 hold_9_2
   - 2602.870813397129 hold_9_3 - 2034.912718204489 hold_9_4
   - 1670.419651995905 hold_9_5 - 1416.666666666667 hold_9_6
   - 1229.841748304446 hold_9_7 - 1086.551264980027 hold_9_8
   - 973.1663685152056 hold_9_9 - 881.2095032397407 hold_9_10
   - 805.1307350764677 hold_9_11 - 741.1444141689373 hold_9_12
   - 686.5797223390829 hold_9_13 - 639.4984326018808 hold_9_14
   - 598.4598459845985 hold_9_15 - 562.3707787732598 hold_9_16
   - 530.3867403314916 hold_9_17 - 501.8450184501845 hold_9_18
   - 476.2182667055733 hold_9_19 - 453.081621321488 hold_9_20
   - 432.0889594916601 hold_9_21 - 412.9554655870446 hold_9_22
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_4_source_nic_in_4: switch_source_nic_in_4 = 1 ->
   - 4049.259480279933 hold_4_1 - 2077.199231903821 hold_4_2
   - 1396.889562629835 hold_4_3 - 1052.260583931315 hold_4_4
   - 844.0288127735122 hold_4_5 - 704.5963014358132 hold_4_6
   + flow_nic_in_4_4 + flow_5_4 <= 0
 constr_intergroup_throughput_0_nic_out_4_nic_in_0: switch_nic_out_4_nic_in_0 = 1 ->
   - 3367.47123843898 hold_0_1 - 1881.759737804109 hold_0_2
   - 1305.694043558121 hold_0_3 - 999.665171097569 hold_0_4
   - 809.8518960559865 hold_0_5 - 680.6182464779099 hold_0_6
   + flow_nic_in_0_0 + flow_1_0 + flow_2_0 + flow_3_0 <= 0
 constr_intergroup_throughput_6_nic_out_4_nic_in_6: switch_nic_out_4_nic_in_6 = 1 ->
   - 3367.47123843898 hold_6_1 - 1881.759737804109 hold_6_2
   - 1305.694043558121 hold_6_3 - 999.665171097569 hold_6_4
   - 809.8518960559865 hold_6_5 - 680.6182464779099 hold_6_6
   + flow_nic_in_6_6 + flow_7_6 <= 0
 constr_intergroup_throughput_8_nic_out_4_nic_in_8: switch_nic_out_4_nic_in_8 = 1 ->
   - 5891.696750902527 hold_8_1 - 3610.619469026549 hold_8_2
   - 2602.870813397129 hold_8_3 - 2034.912718204489 hold_8_4
   - 1670.419651995905 hold_8_5 - 1416.666666666667 hold_8_6
   - 1229.841748304446 hold_8_7 - 1086.551264980027 hold_8_8
   - 973.1663685152056 hold_8_9 - 881.2095032397407 hold_8_10
   - 805.1307350764677 hold_8_11 - 741.1444141689373 hold_8_12
   - 686.5797223390829 hold_8_13 - 639.4984326018808 hold_8_14
   - 598.4598459845985 hold_8_15 - 562.3707787732598 hold_8_16
   - 530.3867403314916 hold_8_17 - 501.8450184501845 hold_8_18
   - 476.2182667055733 hold_8_19 - 453.081621321488 hold_8_20
   - 432.0889594916601 hold_8_21 - 412.9554655870446 hold_8_22
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_4_nic_in_9: switch_nic_out_4_nic_in_9 = 1 ->
   - 5891.696750902527 hold_9_1 - 3610.619469026549 hold_9_2
   - 2602.870813397129 hold_9_3 - 2034.912718204489 hold_9_4
   - 1670.419651995905 hold_9_5 - 1416.666666666667 hold_9_6
   - 1229.841748304446 hold_9_7 - 1086.551264980027 hold_9_8
   - 973.1663685152056 hold_9_9 - 881.2095032397407 hold_9_10
   - 805.1307350764677 hold_9_11 - 741.1444141689373 hold_9_12
   - 686.5797223390829 hold_9_13 - 639.4984326018808 hold_9_14
   - 598.4598459845985 hold_9_15 - 562.3707787732598 hold_9_16
   - 530.3867403314916 hold_9_17 - 501.8450184501845 hold_9_18
   - 476.2182667055733 hold_9_19 - 453.081621321488 hold_9_20
   - 432.0889594916601 hold_9_21 - 412.9554655870446 hold_9_22
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_6_source_nic_in_6: switch_source_nic_in_6 = 1 ->
   - 4049.259480279933 hold_6_1 - 2077.199231903821 hold_6_2
   - 1396.889562629835 hold_6_3 - 1052.260583931315 hold_6_4
   - 844.0288127735122 hold_6_5 - 704.5963014358132 hold_6_6
   + flow_nic_in_6_6 + flow_7_6 <= 0
 constr_intergroup_throughput_0_nic_out_6_nic_in_0: switch_nic_out_6_nic_in_0 = 1 ->
   - 3367.47123843898 hold_0_1 - 1881.759737804109 hold_0_2
   - 1305.694043558121 hold_0_3 - 999.665171097569 hold_0_4
   - 809.8518960559865 hold_0_5 - 680.6182464779099 hold_0_6
   + flow_nic_in_0_0 + flow_1_0 + flow_2_0 + flow_3_0 <= 0
 constr_intergroup_throughput_4_nic_out_6_nic_in_4: switch_nic_out_6_nic_in_4 = 1 ->
   - 3367.47123843898 hold_4_1 - 1881.759737804109 hold_4_2
   - 1305.694043558121 hold_4_3 - 999.665171097569 hold_4_4
   - 809.8518960559865 hold_4_5 - 680.6182464779099 hold_4_6
   + flow_nic_in_4_4 + flow_5_4 <= 0
 constr_intergroup_throughput_8_nic_out_6_nic_in_8: switch_nic_out_6_nic_in_8 = 1 ->
   - 5891.696750902527 hold_8_1 - 3610.619469026549 hold_8_2
   - 2602.870813397129 hold_8_3 - 2034.912718204489 hold_8_4
   - 1670.419651995905 hold_8_5 - 1416.666666666667 hold_8_6
   - 1229.841748304446 hold_8_7 - 1086.551264980027 hold_8_8
   - 973.1663685152056 hold_8_9 - 881.2095032397407 hold_8_10
   - 805.1307350764677 hold_8_11 - 741.1444141689373 hold_8_12
   - 686.5797223390829 hold_8_13 - 639.4984326018808 hold_8_14
   - 598.4598459845985 hold_8_15 - 562.3707787732598 hold_8_16
   - 530.3867403314916 hold_8_17 - 501.8450184501845 hold_8_18
   - 476.2182667055733 hold_8_19 - 453.081621321488 hold_8_20
   - 432.0889594916601 hold_8_21 - 412.9554655870446 hold_8_22
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_6_nic_in_9: switch_nic_out_6_nic_in_9 = 1 ->
   - 5891.696750902527 hold_9_1 - 3610.619469026549 hold_9_2
   - 2602.870813397129 hold_9_3 - 2034.912718204489 hold_9_4
   - 1670.419651995905 hold_9_5 - 1416.666666666667 hold_9_6
   - 1229.841748304446 hold_9_7 - 1086.551264980027 hold_9_8
   - 973.1663685152056 hold_9_9 - 881.2095032397407 hold_9_10
   - 805.1307350764677 hold_9_11 - 741.1444141689373 hold_9_12
   - 686.5797223390829 hold_9_13 - 639.4984326018808 hold_9_14
   - 598.4598459845985 hold_9_15 - 562.3707787732598 hold_9_16
   - 530.3867403314916 hold_9_17 - 501.8450184501845 hold_9_18
   - 476.2182667055733 hold_9_19 - 453.081621321488 hold_9_20
   - 432.0889594916601 hold_9_21 - 412.9554655870446 hold_9_22
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_8_source_nic_in_8: switch_source_nic_in_8 = 1 ->
   - 8352.09825997953 hold_8_1 - 4406.047516198704 hold_8_2
   - 2992.299229922993 hold_8_3 - 2265.40810660744 hold_8_4
   - 1822.649095376368 hold_8_5 - 1524.663677130045 hold_8_6
   - 1310.42235426369 hold_8_7 - 1148.972120529428 hold_8_8
   - 1022.940955246333 hold_8_9 - 921.825576140985 hold_8_10
   - 838.9020252904288 hold_8_11 - 769.6661007357103 hold_8_12
   - 710.9871917748541 hold_8_13 - 660.6217616580311 hold_8_14
   - 616.9199364935361 hold_8_15 - 578.6413274712806 hold_8_16
   - 544.8354143019296 hold_8_17 - 514.7615442846328 hold_8_18
   - 487.8340407724039 hold_8_19 - 463.5836836723099 hold_8_20
   - 441.6301347621368 hold_8_21 - 421.6618437370815 hold_8_22
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_0_nic_out_8_nic_in_0: switch_nic_out_8_nic_in_0 = 1 ->
   - 3367.47123843898 hold_0_1 - 1881.759737804109 hold_0_2
   - 1305.694043558121 hold_0_3 - 999.665171097569 hold_0_4
   - 809.8518960559865 hold_0_5 - 680.6182464779099 hold_0_6
   + flow_nic_in_0_0 + flow_1_0 + flow_2_0 + flow_3_0 <= 0
 constr_intergroup_throughput_4_nic_out_8_nic_in_4: switch_nic_out_8_nic_in_4 = 1 ->
   - 3367.47123843898 hold_4_1 - 1881.759737804109 hold_4_2
   - 1305.694043558121 hold_4_3 - 999.665171097569 hold_4_4
   - 809.8518960559865 hold_4_5 - 680.6182464779099 hold_4_6
   + flow_nic_in_4_4 + flow_5_4 <= 0
 constr_intergroup_throughput_6_nic_out_8_nic_in_6: switch_nic_out_8_nic_in_6 = 1 ->
   - 3367.47123843898 hold_6_1 - 1881.759737804109 hold_6_2
   - 1305.694043558121 hold_6_3 - 999.665171097569 hold_6_4
   - 809.8518960559865 hold_6_5 - 680.6182464779099 hold_6_6
   + flow_nic_in_6_6 + flow_7_6 <= 0
 constr_intergroup_throughput_9_nic_out_8_nic_in_9: switch_nic_out_8_nic_in_9 = 1 ->
   - 5891.696750902527 hold_9_1 - 3610.619469026549 hold_9_2
   - 2602.870813397129 hold_9_3 - 2034.912718204489 hold_9_4
   - 1670.419651995905 hold_9_5 - 1416.666666666667 hold_9_6
   - 1229.841748304446 hold_9_7 - 1086.551264980027 hold_9_8
   - 973.1663685152056 hold_9_9 - 881.2095032397407 hold_9_10
   - 805.1307350764677 hold_9_11 - 741.1444141689373 hold_9_12
   - 686.5797223390829 hold_9_13 - 639.4984326018808 hold_9_14
   - 598.4598459845985 hold_9_15 - 562.3707787732598 hold_9_16
   - 530.3867403314916 hold_9_17 - 501.8450184501845 hold_9_18
   - 476.2182667055733 hold_9_19 - 453.081621321488 hold_9_20
   - 432.0889594916601 hold_9_21 - 412.9554655870446 hold_9_22
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_9_source_nic_in_9: switch_source_nic_in_9 = 1 ->
   - 8352.09825997953 hold_9_1 - 4406.047516198704 hold_9_2
   - 2992.299229922993 hold_9_3 - 2265.40810660744 hold_9_4
   - 1822.649095376368 hold_9_5 - 1524.663677130045 hold_9_6
   - 1310.42235426369 hold_9_7 - 1148.972120529428 hold_9_8
   - 1022.940955246333 hold_9_9 - 921.825576140985 hold_9_10
   - 838.9020252904288 hold_9_11 - 769.6661007357103 hold_9_12
   - 710.9871917748541 hold_9_13 - 660.6217616580311 hold_9_14
   - 616.9199364935361 hold_9_15 - 578.6413274712806 hold_9_16
   - 544.8354143019296 hold_9_17 - 514.7615442846328 hold_9_18
   - 487.8340407724039 hold_9_19 - 463.5836836723099 hold_9_20
   - 441.6301347621368 hold_9_21 - 421.6618437370815 hold_9_22
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_0_nic_out_9_nic_in_0: switch_nic_out_9_nic_in_0 = 1 ->
   - 3367.47123843898 hold_0_1 - 1881.759737804109 hold_0_2
   - 1305.694043558121 hold_0_3 - 999.665171097569 hold_0_4
   - 809.8518960559865 hold_0_5 - 680.6182464779099 hold_0_6
   + flow_nic_in_0_0 + flow_1_0 + flow_2_0 + flow_3_0 <= 0
 constr_intergroup_throughput_4_nic_out_9_nic_in_4: switch_nic_out_9_nic_in_4 = 1 ->
   - 3367.47123843898 hold_4_1 - 1881.759737804109 hold_4_2
   - 1305.694043558121 hold_4_3 - 999.665171097569 hold_4_4
   - 809.8518960559865 hold_4_5 - 680.6182464779099 hold_4_6
   + flow_nic_in_4_4 + flow_5_4 <= 0
 constr_intergroup_throughput_6_nic_out_9_nic_in_6: switch_nic_out_9_nic_in_6 = 1 ->
   - 3367.47123843898 hold_6_1 - 1881.759737804109 hold_6_2
   - 1305.694043558121 hold_6_3 - 999.665171097569 hold_6_4
   - 809.8518960559865 hold_6_5 - 680.6182464779099 hold_6_6
   + flow_nic_in_6_6 + flow_7_6 <= 0
 constr_intergroup_throughput_8_nic_out_9_nic_in_8: switch_nic_out_9_nic_in_8 = 1 ->
   - 5891.696750902527 hold_8_1 - 3610.619469026549 hold_8_2
   - 2602.870813397129 hold_8_3 - 2034.912718204489 hold_8_4
   - 1670.419651995905 hold_8_5 - 1416.666666666667 hold_8_6
   - 1229.841748304446 hold_8_7 - 1086.551264980027 hold_8_8
   - 973.1663685152056 hold_8_9 - 881.2095032397407 hold_8_10
   - 805.1307350764677 hold_8_11 - 741.1444141689373 hold_8_12
   - 686.5797223390829 hold_8_13 - 639.4984326018808 hold_8_14
   - 598.4598459845985 hold_8_15 - 562.3707787732598 hold_8_16
   - 530.3867403314916 hold_8_17 - 501.8450184501845 hold_8_18
   - 476.2182667055733 hold_8_19 - 453.081621321488 hold_8_20
   - 432.0889594916601 hold_8_21 - 412.9554655870446 hold_8_22
   + flow_nic_in_8_8 <= 0
 inactive_end_not_complete_0: is_active_0 = 0 -> is_end_complete_0 = 0
 inactive_end_not_complete_1: is_active_1 = 0 -> is_end_complete_1 = 0
 inactive_end_not_complete_2: is_active_2 = 0 -> is_end_complete_2 = 0
 inactive_end_not_complete_3: is_active_3 = 0 -> is_end_complete_3 = 0
 active_end_val_0: is_active_0 = 1 -> - start_0 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11 - 12 hold_0_12
   - 13 hold_0_13 - 14 hold_0_14 - 15 hold_0_15 - 16 hold_0_16
   - 17 hold_0_17 - 18 hold_0_18 - 19 hold_0_19 - 20 hold_0_20
   - 21 hold_0_21 - 22 hold_0_22 - 23 hold_0_23 - 24 hold_0_24 + val_end_0
   = 0
 inactive_end_zero_0: is_active_0 = 0 -> val_end_0 = 0
 active_end_val_1: is_active_1 = 1 -> - start_1 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11 - 12 hold_1_12
   - 13 hold_1_13 - 14 hold_1_14 - 15 hold_1_15 - 16 hold_1_16
   - 17 hold_1_17 - 18 hold_1_18 - 19 hold_1_19 - 20 hold_1_20
   - 21 hold_1_21 - 22 hold_1_22 - 23 hold_1_23 - 24 hold_1_24 + val_end_1
   = 0
 inactive_end_zero_1: is_active_1 = 0 -> val_end_1 = 0
 active_end_val_2: is_active_2 = 1 -> - start_2 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 - 12 hold_2_12
   - 13 hold_2_13 - 14 hold_2_14 - 15 hold_2_15 - 16 hold_2_16
   - 17 hold_2_17 - 18 hold_2_18 - 19 hold_2_19 - 20 hold_2_20
   - 21 hold_2_21 - 22 hold_2_22 - 23 hold_2_23 - 24 hold_2_24 + val_end_2
   = 0
 inactive_end_zero_2: is_active_2 = 0 -> val_end_2 = 0
 active_end_val_3: is_active_3 = 1 -> - start_3 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   - 8 hold_3_8 - 9 hold_3_9 - 10 hold_3_10 - 11 hold_3_11 - 12 hold_3_12
   - 13 hold_3_13 - 14 hold_3_14 - 15 hold_3_15 - 16 hold_3_16
   - 17 hold_3_17 - 18 hold_3_18 - 19 hold_3_19 - 20 hold_3_20
   - 21 hold_3_21 - 22 hold_3_22 - 23 hold_3_23 - 24 hold_3_24 + val_end_3
   = 0
 inactive_end_zero_3: is_active_3 = 0 -> val_end_3 = 0
 inactive_end_not_complete_4: is_active_4 = 0 -> is_end_complete_4 = 0
 inactive_end_not_complete_5: is_active_5 = 0 -> is_end_complete_5 = 0
 active_end_val_4: is_active_4 = 1 -> - start_4 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   - 8 hold_4_8 - 9 hold_4_9 - 10 hold_4_10 - 11 hold_4_11 - 12 hold_4_12
   - 13 hold_4_13 - 14 hold_4_14 - 15 hold_4_15 - 16 hold_4_16
   - 17 hold_4_17 - 18 hold_4_18 - 19 hold_4_19 - 20 hold_4_20
   - 21 hold_4_21 - 22 hold_4_22 - 23 hold_4_23 - 24 hold_4_24 + val_end_4
   = 0
 inactive_end_zero_4: is_active_4 = 0 -> val_end_4 = 0
 active_end_val_5: is_active_5 = 1 -> - start_5 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11 - 12 hold_5_12
   - 13 hold_5_13 - 14 hold_5_14 - 15 hold_5_15 - 16 hold_5_16
   - 17 hold_5_17 - 18 hold_5_18 - 19 hold_5_19 - 20 hold_5_20
   - 21 hold_5_21 - 22 hold_5_22 - 23 hold_5_23 - 24 hold_5_24 + val_end_5
   = 0
 inactive_end_zero_5: is_active_5 = 0 -> val_end_5 = 0
 inactive_end_not_complete_6: is_active_6 = 0 -> is_end_complete_6 = 0
 inactive_end_not_complete_7: is_active_7 = 0 -> is_end_complete_7 = 0
 active_end_val_6: is_active_6 = 1 -> - start_6 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11 - 12 hold_6_12
   - 13 hold_6_13 - 14 hold_6_14 - 15 hold_6_15 - 16 hold_6_16
   - 17 hold_6_17 - 18 hold_6_18 - 19 hold_6_19 - 20 hold_6_20
   - 21 hold_6_21 - 22 hold_6_22 - 23 hold_6_23 - 24 hold_6_24 + val_end_6
   = 0
 inactive_end_zero_6: is_active_6 = 0 -> val_end_6 = 0
 active_end_val_7: is_active_7 = 1 -> - start_7 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   - 8 hold_7_8 - 9 hold_7_9 - 10 hold_7_10 - 11 hold_7_11 - 12 hold_7_12
   - 13 hold_7_13 - 14 hold_7_14 - 15 hold_7_15 - 16 hold_7_16
   - 17 hold_7_17 - 18 hold_7_18 - 19 hold_7_19 - 20 hold_7_20
   - 21 hold_7_21 - 22 hold_7_22 - 23 hold_7_23 - 24 hold_7_24 + val_end_7
   = 0
 inactive_end_zero_7: is_active_7 = 0 -> val_end_7 = 0
 inactive_end_not_complete_8: is_active_8 = 0 -> is_end_complete_8 = 0
 active_end_val_8: is_active_8 = 1 -> - start_8 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11 - 12 hold_8_12
   - 13 hold_8_13 - 14 hold_8_14 - 15 hold_8_15 - 16 hold_8_16
   - 17 hold_8_17 - 18 hold_8_18 - 19 hold_8_19 - 20 hold_8_20
   - 21 hold_8_21 - 22 hold_8_22 - 23 hold_8_23 - 24 hold_8_24 + val_end_8
   = 0
 inactive_end_zero_8: is_active_8 = 0 -> val_end_8 = 0
 inactive_end_not_complete_9: is_active_9 = 0 -> is_end_complete_9 = 0
 active_end_val_9: is_active_9 = 1 -> - start_9 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   - 8 hold_9_8 - 9 hold_9_9 - 10 hold_9_10 - 11 hold_9_11 - 12 hold_9_12
   - 13 hold_9_13 - 14 hold_9_14 - 15 hold_9_15 - 16 hold_9_16
   - 17 hold_9_17 - 18 hold_9_18 - 19 hold_9_19 - 20 hold_9_20
   - 21 hold_9_21 - 22 hold_9_22 - 23 hold_9_23 - 24 hold_9_24 + val_end_9
   = 0
 inactive_end_zero_9: is_active_9 = 0 -> val_end_9 = 0
 group_synch_start_constr_0_1: group_tp_active_0 = 1 -> start_0 - start_1
   = 0
 group_synch_start_constr_0_2: group_tp_active_0 = 1 -> start_0 - start_2
   = 0
 group_synch_start_constr_0_3: group_tp_active_0 = 1 -> start_0 - start_3
   = 0
 group_synch_hold_constr_0_1_1: group_tp_active_0 = 1 -> hold_0_1
   - hold_1_1 = 0
 group_synch_hold_constr_0_2_1: group_tp_active_0 = 1 -> hold_0_1
   - hold_2_1 = 0
 group_synch_hold_constr_0_3_1: group_tp_active_0 = 1 -> hold_0_1
   - hold_3_1 = 0
 group_synch_hold_constr_0_1_2: group_tp_active_0 = 1 -> hold_0_2
   - hold_1_2 = 0
 group_synch_hold_constr_0_2_2: group_tp_active_0 = 1 -> hold_0_2
   - hold_2_2 = 0
 group_synch_hold_constr_0_3_2: group_tp_active_0 = 1 -> hold_0_2
   - hold_3_2 = 0
 group_synch_hold_constr_0_1_3: group_tp_active_0 = 1 -> hold_0_3
   - hold_1_3 = 0
 group_synch_hold_constr_0_2_3: group_tp_active_0 = 1 -> hold_0_3
   - hold_2_3 = 0
 group_synch_hold_constr_0_3_3: group_tp_active_0 = 1 -> hold_0_3
   - hold_3_3 = 0
 group_synch_hold_constr_0_1_4: group_tp_active_0 = 1 -> hold_0_4
   - hold_1_4 = 0
 group_synch_hold_constr_0_2_4: group_tp_active_0 = 1 -> hold_0_4
   - hold_2_4 = 0
 group_synch_hold_constr_0_3_4: group_tp_active_0 = 1 -> hold_0_4
   - hold_3_4 = 0
 group_synch_hold_constr_0_1_5: group_tp_active_0 = 1 -> hold_0_5
   - hold_1_5 = 0
 group_synch_hold_constr_0_2_5: group_tp_active_0 = 1 -> hold_0_5
   - hold_2_5 = 0
 group_synch_hold_constr_0_3_5: group_tp_active_0 = 1 -> hold_0_5
   - hold_3_5 = 0
 group_synch_hold_constr_0_1_6: group_tp_active_0 = 1 -> hold_0_6
   - hold_1_6 = 0
 group_synch_hold_constr_0_2_6: group_tp_active_0 = 1 -> hold_0_6
   - hold_2_6 = 0
 group_synch_hold_constr_0_3_6: group_tp_active_0 = 1 -> hold_0_6
   - hold_3_6 = 0
 group_synch_hold_constr_0_1_7: group_tp_active_0 = 1 -> hold_0_7
   - hold_1_7 = 0
 group_synch_hold_constr_0_2_7: group_tp_active_0 = 1 -> hold_0_7
   - hold_2_7 = 0
 group_synch_hold_constr_0_3_7: group_tp_active_0 = 1 -> hold_0_7
   - hold_3_7 = 0
 group_synch_hold_constr_0_1_8: group_tp_active_0 = 1 -> hold_0_8
   - hold_1_8 = 0
 group_synch_hold_constr_0_2_8: group_tp_active_0 = 1 -> hold_0_8
   - hold_2_8 = 0
 group_synch_hold_constr_0_3_8: group_tp_active_0 = 1 -> hold_0_8
   - hold_3_8 = 0
 group_synch_hold_constr_0_1_9: group_tp_active_0 = 1 -> hold_0_9
   - hold_1_9 = 0
 group_synch_hold_constr_0_2_9: group_tp_active_0 = 1 -> hold_0_9
   - hold_2_9 = 0
 group_synch_hold_constr_0_3_9: group_tp_active_0 = 1 -> hold_0_9
   - hold_3_9 = 0
 group_synch_hold_constr_0_1_10: group_tp_active_0 = 1 -> hold_0_10
   - hold_1_10 = 0
 group_synch_hold_constr_0_2_10: group_tp_active_0 = 1 -> hold_0_10
   - hold_2_10 = 0
 group_synch_hold_constr_0_3_10: group_tp_active_0 = 1 -> hold_0_10
   - hold_3_10 = 0
 group_synch_hold_constr_0_1_11: group_tp_active_0 = 1 -> hold_0_11
   - hold_1_11 = 0
 group_synch_hold_constr_0_2_11: group_tp_active_0 = 1 -> hold_0_11
   - hold_2_11 = 0
 group_synch_hold_constr_0_3_11: group_tp_active_0 = 1 -> hold_0_11
   - hold_3_11 = 0
 group_synch_hold_constr_0_1_12: group_tp_active_0 = 1 -> hold_0_12
   - hold_1_12 = 0
 group_synch_hold_constr_0_2_12: group_tp_active_0 = 1 -> hold_0_12
   - hold_2_12 = 0
 group_synch_hold_constr_0_3_12: group_tp_active_0 = 1 -> hold_0_12
   - hold_3_12 = 0
 group_synch_hold_constr_0_1_13: group_tp_active_0 = 1 -> hold_0_13
   - hold_1_13 = 0
 group_synch_hold_constr_0_2_13: group_tp_active_0 = 1 -> hold_0_13
   - hold_2_13 = 0
 group_synch_hold_constr_0_3_13: group_tp_active_0 = 1 -> hold_0_13
   - hold_3_13 = 0
 group_synch_hold_constr_0_1_14: group_tp_active_0 = 1 -> hold_0_14
   - hold_1_14 = 0
 group_synch_hold_constr_0_2_14: group_tp_active_0 = 1 -> hold_0_14
   - hold_2_14 = 0
 group_synch_hold_constr_0_3_14: group_tp_active_0 = 1 -> hold_0_14
   - hold_3_14 = 0
 group_synch_hold_constr_0_1_15: group_tp_active_0 = 1 -> hold_0_15
   - hold_1_15 = 0
 group_synch_hold_constr_0_2_15: group_tp_active_0 = 1 -> hold_0_15
   - hold_2_15 = 0
 group_synch_hold_constr_0_3_15: group_tp_active_0 = 1 -> hold_0_15
   - hold_3_15 = 0
 group_synch_hold_constr_0_1_16: group_tp_active_0 = 1 -> hold_0_16
   - hold_1_16 = 0
 group_synch_hold_constr_0_2_16: group_tp_active_0 = 1 -> hold_0_16
   - hold_2_16 = 0
 group_synch_hold_constr_0_3_16: group_tp_active_0 = 1 -> hold_0_16
   - hold_3_16 = 0
 group_synch_hold_constr_0_1_17: group_tp_active_0 = 1 -> hold_0_17
   - hold_1_17 = 0
 group_synch_hold_constr_0_2_17: group_tp_active_0 = 1 -> hold_0_17
   - hold_2_17 = 0
 group_synch_hold_constr_0_3_17: group_tp_active_0 = 1 -> hold_0_17
   - hold_3_17 = 0
 group_synch_hold_constr_0_1_18: group_tp_active_0 = 1 -> hold_0_18
   - hold_1_18 = 0
 group_synch_hold_constr_0_2_18: group_tp_active_0 = 1 -> hold_0_18
   - hold_2_18 = 0
 group_synch_hold_constr_0_3_18: group_tp_active_0 = 1 -> hold_0_18
   - hold_3_18 = 0
 group_synch_hold_constr_0_1_19: group_tp_active_0 = 1 -> hold_0_19
   - hold_1_19 = 0
 group_synch_hold_constr_0_2_19: group_tp_active_0 = 1 -> hold_0_19
   - hold_2_19 = 0
 group_synch_hold_constr_0_3_19: group_tp_active_0 = 1 -> hold_0_19
   - hold_3_19 = 0
 group_synch_hold_constr_0_1_20: group_tp_active_0 = 1 -> hold_0_20
   - hold_1_20 = 0
 group_synch_hold_constr_0_2_20: group_tp_active_0 = 1 -> hold_0_20
   - hold_2_20 = 0
 group_synch_hold_constr_0_3_20: group_tp_active_0 = 1 -> hold_0_20
   - hold_3_20 = 0
 group_synch_hold_constr_0_1_21: group_tp_active_0 = 1 -> hold_0_21
   - hold_1_21 = 0
 group_synch_hold_constr_0_2_21: group_tp_active_0 = 1 -> hold_0_21
   - hold_2_21 = 0
 group_synch_hold_constr_0_3_21: group_tp_active_0 = 1 -> hold_0_21
   - hold_3_21 = 0
 group_synch_hold_constr_0_1_22: group_tp_active_0 = 1 -> hold_0_22
   - hold_1_22 = 0
 group_synch_hold_constr_0_2_22: group_tp_active_0 = 1 -> hold_0_22
   - hold_2_22 = 0
 group_synch_hold_constr_0_3_22: group_tp_active_0 = 1 -> hold_0_22
   - hold_3_22 = 0
 group_synch_hold_constr_0_1_23: group_tp_active_0 = 1 -> hold_0_23
   - hold_1_23 = 0
 group_synch_hold_constr_0_2_23: group_tp_active_0 = 1 -> hold_0_23
   - hold_2_23 = 0
 group_synch_hold_constr_0_3_23: group_tp_active_0 = 1 -> hold_0_23
   - hold_3_23 = 0
 group_synch_hold_constr_0_1_24: group_tp_active_0 = 1 -> hold_0_24
   - hold_1_24 = 0
 group_synch_hold_constr_0_2_24: group_tp_active_0 = 1 -> hold_0_24
   - hold_2_24 = 0
 group_synch_hold_constr_0_3_24: group_tp_active_0 = 1 -> hold_0_24
   - hold_3_24 = 0
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
 tp_off_limit_constr_2_1: group_tp_active_0 = 0 -> hold_2_1 = 0
 tp_off_limit_constr_2_2: group_tp_active_0 = 0 -> hold_2_2 = 0
 tp_off_limit_constr_2_3: group_tp_active_0 = 0 -> hold_2_3 = 0
 tp_off_limit_constr_2_4: group_tp_active_0 = 0 -> hold_2_4 = 0
 tp_off_limit_constr_2_5: group_tp_active_0 = 0 -> hold_2_5 = 0
 tp_off_limit_constr_2_6: group_tp_active_0 = 0 -> hold_2_6 = 0
 tp_off_limit_constr_2_7: group_tp_active_0 = 0 -> hold_2_7 = 0
 tp_off_limit_constr_2_8: group_tp_active_0 = 0 -> hold_2_8 = 0
 tp_off_limit_constr_2_9: group_tp_active_0 = 0 -> hold_2_9 = 0
 tp_off_limit_constr_2_10: group_tp_active_0 = 0 -> hold_2_10 = 0
 tp_off_limit_constr_2_11: group_tp_active_0 = 0 -> hold_2_11 = 0
 tp_off_limit_constr_2_12: group_tp_active_0 = 0 -> hold_2_12 = 0
 tp_off_limit_constr_2_13: group_tp_active_0 = 0 -> hold_2_13 = 0
 tp_off_limit_constr_2_14: group_tp_active_0 = 0 -> hold_2_14 = 0
 tp_off_limit_constr_2_15: group_tp_active_0 = 0 -> hold_2_15 = 0
 tp_off_limit_constr_2_16: group_tp_active_0 = 0 -> hold_2_16 = 0
 tp_off_limit_constr_2_17: group_tp_active_0 = 0 -> hold_2_17 = 0
 tp_off_limit_constr_2_18: group_tp_active_0 = 0 -> hold_2_18 = 0
 tp_off_limit_constr_2_19: group_tp_active_0 = 0 -> hold_2_19 = 0
 tp_off_limit_constr_2_20: group_tp_active_0 = 0 -> hold_2_20 = 0
 tp_off_limit_constr_2_21: group_tp_active_0 = 0 -> hold_2_21 = 0
 tp_off_limit_constr_2_22: group_tp_active_0 = 0 -> hold_2_22 = 0
 tp_off_limit_constr_2_23: group_tp_active_0 = 0 -> hold_2_23 = 0
 tp_off_limit_constr_2_24: group_tp_active_0 = 0 -> hold_2_24 = 0
 tp_off_limit_constr_3_1: group_tp_active_0 = 0 -> hold_3_1 = 0
 tp_off_limit_constr_3_2: group_tp_active_0 = 0 -> hold_3_2 = 0
 tp_off_limit_constr_3_3: group_tp_active_0 = 0 -> hold_3_3 = 0
 tp_off_limit_constr_3_4: group_tp_active_0 = 0 -> hold_3_4 = 0
 tp_off_limit_constr_3_5: group_tp_active_0 = 0 -> hold_3_5 = 0
 tp_off_limit_constr_3_6: group_tp_active_0 = 0 -> hold_3_6 = 0
 tp_off_limit_constr_3_7: group_tp_active_0 = 0 -> hold_3_7 = 0
 tp_off_limit_constr_3_8: group_tp_active_0 = 0 -> hold_3_8 = 0
 tp_off_limit_constr_3_9: group_tp_active_0 = 0 -> hold_3_9 = 0
 tp_off_limit_constr_3_10: group_tp_active_0 = 0 -> hold_3_10 = 0
 tp_off_limit_constr_3_11: group_tp_active_0 = 0 -> hold_3_11 = 0
 tp_off_limit_constr_3_12: group_tp_active_0 = 0 -> hold_3_12 = 0
 tp_off_limit_constr_3_13: group_tp_active_0 = 0 -> hold_3_13 = 0
 tp_off_limit_constr_3_14: group_tp_active_0 = 0 -> hold_3_14 = 0
 tp_off_limit_constr_3_15: group_tp_active_0 = 0 -> hold_3_15 = 0
 tp_off_limit_constr_3_16: group_tp_active_0 = 0 -> hold_3_16 = 0
 tp_off_limit_constr_3_17: group_tp_active_0 = 0 -> hold_3_17 = 0
 tp_off_limit_constr_3_18: group_tp_active_0 = 0 -> hold_3_18 = 0
 tp_off_limit_constr_3_19: group_tp_active_0 = 0 -> hold_3_19 = 0
 tp_off_limit_constr_3_20: group_tp_active_0 = 0 -> hold_3_20 = 0
 tp_off_limit_constr_3_21: group_tp_active_0 = 0 -> hold_3_21 = 0
 tp_off_limit_constr_3_22: group_tp_active_0 = 0 -> hold_3_22 = 0
 tp_off_limit_constr_3_23: group_tp_active_0 = 0 -> hold_3_23 = 0
 tp_off_limit_constr_3_24: group_tp_active_0 = 0 -> hold_3_24 = 0
 group_synch_start_constr_4_5: group_tp_active_4 = 1 -> start_4 - start_5
   = 0
 group_synch_hold_constr_4_5_1: group_tp_active_4 = 1 -> hold_4_1
   - hold_5_1 = 0
 group_synch_hold_constr_4_5_2: group_tp_active_4 = 1 -> hold_4_2
   - hold_5_2 = 0
 group_synch_hold_constr_4_5_3: group_tp_active_4 = 1 -> hold_4_3
   - hold_5_3 = 0
 group_synch_hold_constr_4_5_4: group_tp_active_4 = 1 -> hold_4_4
   - hold_5_4 = 0
 group_synch_hold_constr_4_5_5: group_tp_active_4 = 1 -> hold_4_5
   - hold_5_5 = 0
 group_synch_hold_constr_4_5_6: group_tp_active_4 = 1 -> hold_4_6
   - hold_5_6 = 0
 group_synch_hold_constr_4_5_7: group_tp_active_4 = 1 -> hold_4_7
   - hold_5_7 = 0
 group_synch_hold_constr_4_5_8: group_tp_active_4 = 1 -> hold_4_8
   - hold_5_8 = 0
 group_synch_hold_constr_4_5_9: group_tp_active_4 = 1 -> hold_4_9
   - hold_5_9 = 0
 group_synch_hold_constr_4_5_10: group_tp_active_4 = 1 -> hold_4_10
   - hold_5_10 = 0
 group_synch_hold_constr_4_5_11: group_tp_active_4 = 1 -> hold_4_11
   - hold_5_11 = 0
 group_synch_hold_constr_4_5_12: group_tp_active_4 = 1 -> hold_4_12
   - hold_5_12 = 0
 group_synch_hold_constr_4_5_13: group_tp_active_4 = 1 -> hold_4_13
   - hold_5_13 = 0
 group_synch_hold_constr_4_5_14: group_tp_active_4 = 1 -> hold_4_14
   - hold_5_14 = 0
 group_synch_hold_constr_4_5_15: group_tp_active_4 = 1 -> hold_4_15
   - hold_5_15 = 0
 group_synch_hold_constr_4_5_16: group_tp_active_4 = 1 -> hold_4_16
   - hold_5_16 = 0
 group_synch_hold_constr_4_5_17: group_tp_active_4 = 1 -> hold_4_17
   - hold_5_17 = 0
 group_synch_hold_constr_4_5_18: group_tp_active_4 = 1 -> hold_4_18
   - hold_5_18 = 0
 group_synch_hold_constr_4_5_19: group_tp_active_4 = 1 -> hold_4_19
   - hold_5_19 = 0
 group_synch_hold_constr_4_5_20: group_tp_active_4 = 1 -> hold_4_20
   - hold_5_20 = 0
 group_synch_hold_constr_4_5_21: group_tp_active_4 = 1 -> hold_4_21
   - hold_5_21 = 0
 group_synch_hold_constr_4_5_22: group_tp_active_4 = 1 -> hold_4_22
   - hold_5_22 = 0
 group_synch_hold_constr_4_5_23: group_tp_active_4 = 1 -> hold_4_23
   - hold_5_23 = 0
 group_synch_hold_constr_4_5_24: group_tp_active_4 = 1 -> hold_4_24
   - hold_5_24 = 0
 tp_on_limit_constr_4_13: group_tp_active_4 = 1 -> hold_4_13 = 0
 tp_on_limit_constr_4_14: group_tp_active_4 = 1 -> hold_4_14 = 0
 tp_on_limit_constr_4_15: group_tp_active_4 = 1 -> hold_4_15 = 0
 tp_on_limit_constr_4_16: group_tp_active_4 = 1 -> hold_4_16 = 0
 tp_on_limit_constr_4_17: group_tp_active_4 = 1 -> hold_4_17 = 0
 tp_on_limit_constr_4_18: group_tp_active_4 = 1 -> hold_4_18 = 0
 tp_on_limit_constr_4_19: group_tp_active_4 = 1 -> hold_4_19 = 0
 tp_on_limit_constr_4_20: group_tp_active_4 = 1 -> hold_4_20 = 0
 tp_on_limit_constr_4_21: group_tp_active_4 = 1 -> hold_4_21 = 0
 tp_on_limit_constr_4_22: group_tp_active_4 = 1 -> hold_4_22 = 0
 tp_on_limit_constr_4_23: group_tp_active_4 = 1 -> hold_4_23 = 0
 tp_on_limit_constr_4_24: group_tp_active_4 = 1 -> hold_4_24 = 0
 tp_on_limit_constr_5_13: group_tp_active_4 = 1 -> hold_5_13 = 0
 tp_on_limit_constr_5_14: group_tp_active_4 = 1 -> hold_5_14 = 0
 tp_on_limit_constr_5_15: group_tp_active_4 = 1 -> hold_5_15 = 0
 tp_on_limit_constr_5_16: group_tp_active_4 = 1 -> hold_5_16 = 0
 tp_on_limit_constr_5_17: group_tp_active_4 = 1 -> hold_5_17 = 0
 tp_on_limit_constr_5_18: group_tp_active_4 = 1 -> hold_5_18 = 0
 tp_on_limit_constr_5_19: group_tp_active_4 = 1 -> hold_5_19 = 0
 tp_on_limit_constr_5_20: group_tp_active_4 = 1 -> hold_5_20 = 0
 tp_on_limit_constr_5_21: group_tp_active_4 = 1 -> hold_5_21 = 0
 tp_on_limit_constr_5_22: group_tp_active_4 = 1 -> hold_5_22 = 0
 tp_on_limit_constr_5_23: group_tp_active_4 = 1 -> hold_5_23 = 0
 tp_on_limit_constr_5_24: group_tp_active_4 = 1 -> hold_5_24 = 0
 tp_off_limit_constr_4_1: group_tp_active_4 = 0 -> hold_4_1 = 0
 tp_off_limit_constr_4_2: group_tp_active_4 = 0 -> hold_4_2 = 0
 tp_off_limit_constr_4_3: group_tp_active_4 = 0 -> hold_4_3 = 0
 tp_off_limit_constr_4_4: group_tp_active_4 = 0 -> hold_4_4 = 0
 tp_off_limit_constr_4_5: group_tp_active_4 = 0 -> hold_4_5 = 0
 tp_off_limit_constr_4_6: group_tp_active_4 = 0 -> hold_4_6 = 0
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
 tp_off_limit_constr_5_1: group_tp_active_4 = 0 -> hold_5_1 = 0
 tp_off_limit_constr_5_2: group_tp_active_4 = 0 -> hold_5_2 = 0
 tp_off_limit_constr_5_3: group_tp_active_4 = 0 -> hold_5_3 = 0
 tp_off_limit_constr_5_4: group_tp_active_4 = 0 -> hold_5_4 = 0
 tp_off_limit_constr_5_5: group_tp_active_4 = 0 -> hold_5_5 = 0
 tp_off_limit_constr_5_6: group_tp_active_4 = 0 -> hold_5_6 = 0
 tp_off_limit_constr_5_7: group_tp_active_4 = 0 -> hold_5_7 = 0
 tp_off_limit_constr_5_8: group_tp_active_4 = 0 -> hold_5_8 = 0
 tp_off_limit_constr_5_9: group_tp_active_4 = 0 -> hold_5_9 = 0
 tp_off_limit_constr_5_10: group_tp_active_4 = 0 -> hold_5_10 = 0
 tp_off_limit_constr_5_11: group_tp_active_4 = 0 -> hold_5_11 = 0
 tp_off_limit_constr_5_12: group_tp_active_4 = 0 -> hold_5_12 = 0
 tp_off_limit_constr_5_13: group_tp_active_4 = 0 -> hold_5_13 = 0
 tp_off_limit_constr_5_14: group_tp_active_4 = 0 -> hold_5_14 = 0
 tp_off_limit_constr_5_15: group_tp_active_4 = 0 -> hold_5_15 = 0
 tp_off_limit_constr_5_16: group_tp_active_4 = 0 -> hold_5_16 = 0
 tp_off_limit_constr_5_17: group_tp_active_4 = 0 -> hold_5_17 = 0
 tp_off_limit_constr_5_18: group_tp_active_4 = 0 -> hold_5_18 = 0
 tp_off_limit_constr_5_19: group_tp_active_4 = 0 -> hold_5_19 = 0
 tp_off_limit_constr_5_20: group_tp_active_4 = 0 -> hold_5_20 = 0
 tp_off_limit_constr_5_21: group_tp_active_4 = 0 -> hold_5_21 = 0
 tp_off_limit_constr_5_22: group_tp_active_4 = 0 -> hold_5_22 = 0
 tp_off_limit_constr_5_23: group_tp_active_4 = 0 -> hold_5_23 = 0
 tp_off_limit_constr_5_24: group_tp_active_4 = 0 -> hold_5_24 = 0
 group_synch_start_constr_6_7: group_tp_active_6 = 1 -> start_6 - start_7
   = 0
 group_synch_hold_constr_6_7_1: group_tp_active_6 = 1 -> hold_6_1
   - hold_7_1 = 0
 group_synch_hold_constr_6_7_2: group_tp_active_6 = 1 -> hold_6_2
   - hold_7_2 = 0
 group_synch_hold_constr_6_7_3: group_tp_active_6 = 1 -> hold_6_3
   - hold_7_3 = 0
 group_synch_hold_constr_6_7_4: group_tp_active_6 = 1 -> hold_6_4
   - hold_7_4 = 0
 group_synch_hold_constr_6_7_5: group_tp_active_6 = 1 -> hold_6_5
   - hold_7_5 = 0
 group_synch_hold_constr_6_7_6: group_tp_active_6 = 1 -> hold_6_6
   - hold_7_6 = 0
 group_synch_hold_constr_6_7_7: group_tp_active_6 = 1 -> hold_6_7
   - hold_7_7 = 0
 group_synch_hold_constr_6_7_8: group_tp_active_6 = 1 -> hold_6_8
   - hold_7_8 = 0
 group_synch_hold_constr_6_7_9: group_tp_active_6 = 1 -> hold_6_9
   - hold_7_9 = 0
 group_synch_hold_constr_6_7_10: group_tp_active_6 = 1 -> hold_6_10
   - hold_7_10 = 0
 group_synch_hold_constr_6_7_11: group_tp_active_6 = 1 -> hold_6_11
   - hold_7_11 = 0
 group_synch_hold_constr_6_7_12: group_tp_active_6 = 1 -> hold_6_12
   - hold_7_12 = 0
 group_synch_hold_constr_6_7_13: group_tp_active_6 = 1 -> hold_6_13
   - hold_7_13 = 0
 group_synch_hold_constr_6_7_14: group_tp_active_6 = 1 -> hold_6_14
   - hold_7_14 = 0
 group_synch_hold_constr_6_7_15: group_tp_active_6 = 1 -> hold_6_15
   - hold_7_15 = 0
 group_synch_hold_constr_6_7_16: group_tp_active_6 = 1 -> hold_6_16
   - hold_7_16 = 0
 group_synch_hold_constr_6_7_17: group_tp_active_6 = 1 -> hold_6_17
   - hold_7_17 = 0
 group_synch_hold_constr_6_7_18: group_tp_active_6 = 1 -> hold_6_18
   - hold_7_18 = 0
 group_synch_hold_constr_6_7_19: group_tp_active_6 = 1 -> hold_6_19
   - hold_7_19 = 0
 group_synch_hold_constr_6_7_20: group_tp_active_6 = 1 -> hold_6_20
   - hold_7_20 = 0
 group_synch_hold_constr_6_7_21: group_tp_active_6 = 1 -> hold_6_21
   - hold_7_21 = 0
 group_synch_hold_constr_6_7_22: group_tp_active_6 = 1 -> hold_6_22
   - hold_7_22 = 0
 group_synch_hold_constr_6_7_23: group_tp_active_6 = 1 -> hold_6_23
   - hold_7_23 = 0
 group_synch_hold_constr_6_7_24: group_tp_active_6 = 1 -> hold_6_24
   - hold_7_24 = 0
 tp_on_limit_constr_6_13: group_tp_active_6 = 1 -> hold_6_13 = 0
 tp_on_limit_constr_6_14: group_tp_active_6 = 1 -> hold_6_14 = 0
 tp_on_limit_constr_6_15: group_tp_active_6 = 1 -> hold_6_15 = 0
 tp_on_limit_constr_6_16: group_tp_active_6 = 1 -> hold_6_16 = 0
 tp_on_limit_constr_6_17: group_tp_active_6 = 1 -> hold_6_17 = 0
 tp_on_limit_constr_6_18: group_tp_active_6 = 1 -> hold_6_18 = 0
 tp_on_limit_constr_6_19: group_tp_active_6 = 1 -> hold_6_19 = 0
 tp_on_limit_constr_6_20: group_tp_active_6 = 1 -> hold_6_20 = 0
 tp_on_limit_constr_6_21: group_tp_active_6 = 1 -> hold_6_21 = 0
 tp_on_limit_constr_6_22: group_tp_active_6 = 1 -> hold_6_22 = 0
 tp_on_limit_constr_6_23: group_tp_active_6 = 1 -> hold_6_23 = 0
 tp_on_limit_constr_6_24: group_tp_active_6 = 1 -> hold_6_24 = 0
 tp_on_limit_constr_7_13: group_tp_active_6 = 1 -> hold_7_13 = 0
 tp_on_limit_constr_7_14: group_tp_active_6 = 1 -> hold_7_14 = 0
 tp_on_limit_constr_7_15: group_tp_active_6 = 1 -> hold_7_15 = 0
 tp_on_limit_constr_7_16: group_tp_active_6 = 1 -> hold_7_16 = 0
 tp_on_limit_constr_7_17: group_tp_active_6 = 1 -> hold_7_17 = 0
 tp_on_limit_constr_7_18: group_tp_active_6 = 1 -> hold_7_18 = 0
 tp_on_limit_constr_7_19: group_tp_active_6 = 1 -> hold_7_19 = 0
 tp_on_limit_constr_7_20: group_tp_active_6 = 1 -> hold_7_20 = 0
 tp_on_limit_constr_7_21: group_tp_active_6 = 1 -> hold_7_21 = 0
 tp_on_limit_constr_7_22: group_tp_active_6 = 1 -> hold_7_22 = 0
 tp_on_limit_constr_7_23: group_tp_active_6 = 1 -> hold_7_23 = 0
 tp_on_limit_constr_7_24: group_tp_active_6 = 1 -> hold_7_24 = 0
 tp_off_limit_constr_6_1: group_tp_active_6 = 0 -> hold_6_1 = 0
 tp_off_limit_constr_6_2: group_tp_active_6 = 0 -> hold_6_2 = 0
 tp_off_limit_constr_6_3: group_tp_active_6 = 0 -> hold_6_3 = 0
 tp_off_limit_constr_6_4: group_tp_active_6 = 0 -> hold_6_4 = 0
 tp_off_limit_constr_6_5: group_tp_active_6 = 0 -> hold_6_5 = 0
 tp_off_limit_constr_6_6: group_tp_active_6 = 0 -> hold_6_6 = 0
 tp_off_limit_constr_6_7: group_tp_active_6 = 0 -> hold_6_7 = 0
 tp_off_limit_constr_6_8: group_tp_active_6 = 0 -> hold_6_8 = 0
 tp_off_limit_constr_6_9: group_tp_active_6 = 0 -> hold_6_9 = 0
 tp_off_limit_constr_6_10: group_tp_active_6 = 0 -> hold_6_10 = 0
 tp_off_limit_constr_6_11: group_tp_active_6 = 0 -> hold_6_11 = 0
 tp_off_limit_constr_6_12: group_tp_active_6 = 0 -> hold_6_12 = 0
 tp_off_limit_constr_6_13: group_tp_active_6 = 0 -> hold_6_13 = 0
 tp_off_limit_constr_6_14: group_tp_active_6 = 0 -> hold_6_14 = 0
 tp_off_limit_constr_6_15: group_tp_active_6 = 0 -> hold_6_15 = 0
 tp_off_limit_constr_6_16: group_tp_active_6 = 0 -> hold_6_16 = 0
 tp_off_limit_constr_6_17: group_tp_active_6 = 0 -> hold_6_17 = 0
 tp_off_limit_constr_6_18: group_tp_active_6 = 0 -> hold_6_18 = 0
 tp_off_limit_constr_6_19: group_tp_active_6 = 0 -> hold_6_19 = 0
 tp_off_limit_constr_6_20: group_tp_active_6 = 0 -> hold_6_20 = 0
 tp_off_limit_constr_6_21: group_tp_active_6 = 0 -> hold_6_21 = 0
 tp_off_limit_constr_6_22: group_tp_active_6 = 0 -> hold_6_22 = 0
 tp_off_limit_constr_6_23: group_tp_active_6 = 0 -> hold_6_23 = 0
 tp_off_limit_constr_6_24: group_tp_active_6 = 0 -> hold_6_24 = 0
 tp_off_limit_constr_7_1: group_tp_active_6 = 0 -> hold_7_1 = 0
 tp_off_limit_constr_7_2: group_tp_active_6 = 0 -> hold_7_2 = 0
 tp_off_limit_constr_7_3: group_tp_active_6 = 0 -> hold_7_3 = 0
 tp_off_limit_constr_7_4: group_tp_active_6 = 0 -> hold_7_4 = 0
 tp_off_limit_constr_7_5: group_tp_active_6 = 0 -> hold_7_5 = 0
 tp_off_limit_constr_7_6: group_tp_active_6 = 0 -> hold_7_6 = 0
 tp_off_limit_constr_7_7: group_tp_active_6 = 0 -> hold_7_7 = 0
 tp_off_limit_constr_7_8: group_tp_active_6 = 0 -> hold_7_8 = 0
 tp_off_limit_constr_7_9: group_tp_active_6 = 0 -> hold_7_9 = 0
 tp_off_limit_constr_7_10: group_tp_active_6 = 0 -> hold_7_10 = 0
 tp_off_limit_constr_7_11: group_tp_active_6 = 0 -> hold_7_11 = 0
 tp_off_limit_constr_7_12: group_tp_active_6 = 0 -> hold_7_12 = 0
 tp_off_limit_constr_7_13: group_tp_active_6 = 0 -> hold_7_13 = 0
 tp_off_limit_constr_7_14: group_tp_active_6 = 0 -> hold_7_14 = 0
 tp_off_limit_constr_7_15: group_tp_active_6 = 0 -> hold_7_15 = 0
 tp_off_limit_constr_7_16: group_tp_active_6 = 0 -> hold_7_16 = 0
 tp_off_limit_constr_7_17: group_tp_active_6 = 0 -> hold_7_17 = 0
 tp_off_limit_constr_7_18: group_tp_active_6 = 0 -> hold_7_18 = 0
 tp_off_limit_constr_7_19: group_tp_active_6 = 0 -> hold_7_19 = 0
 tp_off_limit_constr_7_20: group_tp_active_6 = 0 -> hold_7_20 = 0
 tp_off_limit_constr_7_21: group_tp_active_6 = 0 -> hold_7_21 = 0
 tp_off_limit_constr_7_22: group_tp_active_6 = 0 -> hold_7_22 = 0
 tp_off_limit_constr_7_23: group_tp_active_6 = 0 -> hold_7_23 = 0
 tp_off_limit_constr_7_24: group_tp_active_6 = 0 -> hold_7_24 = 0
 tp_off_limit_constr_8_23: group_tp_active_8 = 0 -> hold_8_23 = 0
 tp_off_limit_constr_8_24: group_tp_active_8 = 0 -> hold_8_24 = 0
 tp_off_limit_constr_9_23: group_tp_active_9 = 0 -> hold_9_23 = 0
 tp_off_limit_constr_9_24: group_tp_active_9 = 0 -> hold_9_24 = 0
 group_throughput_constr_0: group_tp_active_0 = 1 ->
   - 14759.99862556412 hold_0_1 - 7379.999312782058 hold_0_2
   - 4919.999541854705 hold_0_3 - 3689.999656391029 hold_0_4
   - 2951.999725112823 hold_0_5 - 2459.999770927353 hold_0_6
   - 2108.571232223445 hold_0_7 - 1844.999828195515 hold_0_8
   - 1639.999847284902 hold_0_9 - 1475.999862556411 hold_0_10
   - 1341.818056869465 hold_0_11 - 1229.999885463676 hold_0_12
   - 1135.384509658778 hold_0_13 - 1054.285616111722 hold_0_14
   - 983.9999083709409 hold_0_15 - 922.4999140977573 hold_0_16
   - 868.2352132684772 hold_0_17 - 819.9999236424508 hold_0_18
   - 776.842032924427 hold_0_19 - 737.9999312782057 hold_0_20
   - 702.8570774078149 hold_0_21 - 670.9090284347326 hold_0_22
   - 641.7390706767006 hold_0_23 - 614.9999427318381 hold_0_24
   + flow_nic_in_0_0 + flow_nic_in_0_1 + flow_nic_in_0_2 + flow_nic_in_0_3
   <= 0
 group_throughput_constr_4: group_tp_active_4 = 1 ->
   - 8074.06386378554 hold_4_1 - 4037.03193189277 hold_4_2
   - 2691.354621261847 hold_4_3 - 2018.515965946385 hold_4_4
   - 1614.812772757108 hold_4_5 - 1345.677310630924 hold_4_6
   - 1153.437694826506 hold_4_7 - 1009.257982973192 hold_4_8
   - 897.1182070872823 hold_4_9 - 807.4063863785541 hold_4_10
   - 734.0058057986855 hold_4_11 - 672.8386553154618 hold_4_12
   + flow_nic_in_4_4 + flow_nic_in_4_5 <= 0
 group_throughput_constr_6: group_tp_active_6 = 1 ->
   - 8074.06386378554 hold_6_1 - 4037.03193189277 hold_6_2
   - 2691.354621261847 hold_6_3 - 2018.515965946385 hold_6_4
   - 1614.812772757108 hold_6_5 - 1345.677310630924 hold_6_6
   - 1153.437694826506 hold_6_7 - 1009.257982973192 hold_6_8
   - 897.1182070872823 hold_6_9 - 807.4063863785541 hold_6_10
   - 734.0058057986855 hold_6_11 - 672.8386553154618 hold_6_12
   + flow_nic_in_6_6 + flow_nic_in_6_7 <= 0
 group_throughput_constr_8: group_tp_active_8 = 1 ->
   - 9325.714285714286 hold_8_1 - 4662.857142857143 hold_8_2
   - 3108.571428571428 hold_8_3 - 2331.428571428572 hold_8_4
   - 1865.142857142857 hold_8_5 - 1554.285714285714 hold_8_6
   - 1332.244897959184 hold_8_7 - 1165.714285714286 hold_8_8
   - 1036.190476190476 hold_8_9 - 932.5714285714284 hold_8_10
   - 847.7922077922078 hold_8_11 - 777.1428571428571 hold_8_12
   - 717.3626373626373 hold_8_13 - 666.1224489795918 hold_8_14
   - 621.7142857142858 hold_8_15 - 582.8571428571429 hold_8_16
   - 548.5714285714286 hold_8_17 - 518.0952380952381 hold_8_18
   - 490.8270676691729 hold_8_19 - 466.2857142857142 hold_8_20
   - 444.0816326530613 hold_8_21 - 423.8961038961039 hold_8_22
   + flow_nic_in_8_8 <= 0
 group_throughput_constr_9: group_tp_active_9 = 1 ->
   - 9325.714285714286 hold_9_1 - 4662.857142857143 hold_9_2
   - 3108.571428571428 hold_9_3 - 2331.428571428572 hold_9_4
   - 1865.142857142857 hold_9_5 - 1554.285714285714 hold_9_6
   - 1332.244897959184 hold_9_7 - 1165.714285714286 hold_9_8
   - 1036.190476190476 hold_9_9 - 932.5714285714284 hold_9_10
   - 847.7922077922078 hold_9_11 - 777.1428571428571 hold_9_12
   - 717.3626373626373 hold_9_13 - 666.1224489795918 hold_9_14
   - 621.7142857142858 hold_9_15 - 582.8571428571429 hold_9_16
   - 548.5714285714286 hold_9_17 - 518.0952380952381 hold_9_18
   - 490.8270676691729 hold_9_19 - 466.2857142857142 hold_9_20
   - 444.0816326530613 hold_9_21 - 423.8961038961039 hold_9_22
   + flow_nic_in_9_9 <= 0
 intergroup_group_tp_0_via_source_nic_in_0: tp_on_and_switch_source_nic_in_0_0 = 1 ->
   - 12460.95300941231 hold_0_1 - 6756.694319820167 hold_0_2
   - 4634.950135090166 hold_0_3 - 3527.302828573249 hold_0_4
   - 2846.947376695139 hold_0_5 - 2386.611474907781 hold_0_6
   - 2054.42252917531 hold_0_7 - 1803.408718498074 hold_0_8
   - 1607.055218375964 hold_0_9 - 1449.26099960362 hold_0_10
   - 1319.683369684552 hold_0_11 - 1211.374996624899 hold_0_12
   - 1119.496275535482 hold_0_13 - 1040.572360213022 hold_0_14
   - 972.0437710997572 hold_0_15 - 911.9836041417656 hold_0_16
   - 858.9134766982241 hold_0_17 - 811.6802023431574 hold_0_18
   - 769.3710359291226 hold_0_19 - 731.2541127165592 hold_0_20
   - 696.7357566869338 hold_0_21 - 665.329335211286 hold_0_22
   - 636.6321739309341 hold_0_23 - 610.3081988897634 hold_0_24
   + flow_nic_in_0_0 + flow_nic_in_0_1 + flow_nic_in_0_2 + flow_nic_in_0_3
   <= 0
 intergroup_group_tp_4_via_nic_out_0_nic_in_4: tp_on_and_switch_nic_out_0_nic_in_4_4 = 1 ->
   - 5366.149336128531 hold_4_1 - 3223.656633868661 hold_4_2
   - 2303.828417615377 hold_4_3 - 1792.392643000101 hold_4_4
   - 1466.777120905479 hold_4_5 - 1241.279690871386 hold_4_6
   - 1075.877817936759 hold_4_7 - 949.3728499935663 hold_4_8
   - 849.4875361276669 hold_4_9 - 768.6196123946033 hold_4_10
   - 701.8100166255107 hold_4_11 - 645.6859991034142 hold_4_12
   + flow_nic_in_4_4 + flow_nic_in_4_5 <= 0
 intergroup_group_tp_6_via_nic_out_0_nic_in_6: tp_on_and_switch_nic_out_0_nic_in_6_6 = 1 ->
   - 5366.149336128531 hold_6_1 - 3223.656633868661 hold_6_2
   - 2303.828417615377 hold_6_3 - 1792.392643000101 hold_6_4
   - 1466.777120905479 hold_6_5 - 1241.279690871386 hold_6_6
   - 1075.877817936759 hold_6_7 - 949.3728499935663 hold_6_8
   - 849.4875361276669 hold_6_9 - 768.6196123946033 hold_6_10
   - 701.8100166255107 hold_6_11 - 645.6859991034142 hold_6_12
   + flow_nic_in_6_6 + flow_nic_in_6_7 <= 0
 intergroup_group_tp_4_via_source_nic_in_4: tp_on_and_switch_source_nic_in_4_4 = 1 ->
   - 7333.885604527395 hold_4_1 - 3843.098061973017 hold_4_2
   - 2603.759131617697 hold_4_3 - 1968.839296516373 hold_4_4
   - 1582.862441653366 hold_4_5 - 1323.416171696253 hold_4_6
   - 1137.043829654094 hold_4_7 - 996.6840907841146 hold_4_8
   - 887.1695081060992 hold_4_9 - 799.3389937728837 hold_4_10
   - 727.332477532996 hold_4_11 - 667.2269542320156 hold_4_12
   + flow_nic_in_4_4 + flow_nic_in_4_5 <= 0
 intergroup_group_tp_0_via_nic_out_4_nic_in_0: tp_on_and_switch_nic_out_4_nic_in_0_0 = 1 ->
   - 7677.502879104724 hold_0_1 - 5050.470165752209 hold_0_2
   - 3762.906041760658 hold_0_3 - 2998.476157062457 hold_0_4
   - 2492.190601882461 hold_0_5 - 2132.177509385763 hold_0_6
   - 1863.048126930152 hold_0_7 - 1654.244748407672 hold_0_8
   - 1487.528219032111 hold_0_9 - 1351.338863964033 hold_0_10
   - 1237.995280512534 hold_0_11 - 1142.19374917246 hold_0_12
   - 1060.154334109086 hold_0_13 - 989.1103173416592 hold_0_14
   - 926.9900270180334 hold_0_15 - 872.2114758746661 hold_0_16
   - 823.5457495499254 hold_0_17 - 780.0237121189008 hold_0_18
   - 740.8708088446017 hold_0_19 - 705.4605657146498 hold_0_20
   - 673.2808157555227 hold_0_21 - 643.9087656615694 hold_0_22
   - 616.9923159605032 hold_0_23 - 592.2358782802088 hold_0_24
   + flow_nic_in_0_0 + flow_nic_in_0_1 + flow_nic_in_0_2 + flow_nic_in_0_3
   <= 0
 intergroup_group_tp_6_via_nic_out_4_nic_in_6: tp_on_and_switch_nic_out_4_nic_in_6_6 = 1 ->
   - 5366.149336128531 hold_6_1 - 3223.656633868661 hold_6_2
   - 2303.828417615377 hold_6_3 - 1792.392643000101 hold_6_4
   - 1466.777120905479 hold_6_5 - 1241.279690871386 hold_6_6
   - 1075.877817936759 hold_6_7 - 949.3728499935663 hold_6_8
   - 849.4875361276669 hold_6_9 - 768.6196123946033 hold_6_10
   - 701.8100166255107 hold_6_11 - 645.6859991034142 hold_6_12
   + flow_nic_in_6_6 + flow_nic_in_6_7 <= 0
 intergroup_group_tp_6_via_source_nic_in_6: tp_on_and_switch_source_nic_in_6_6 = 1 ->
   - 7333.885604527395 hold_6_1 - 3843.098061973017 hold_6_2
   - 2603.759131617697 hold_6_3 - 1968.839296516373 hold_6_4
   - 1582.862441653366 hold_6_5 - 1323.416171696253 hold_6_6
   - 1137.043829654094 hold_6_7 - 996.6840907841146 hold_6_8
   - 887.1695081060992 hold_6_9 - 799.3389937728837 hold_6_10
   - 727.332477532996 hold_6_11 - 667.2269542320156 hold_6_12
   + flow_nic_in_6_6 + flow_nic_in_6_7 <= 0
 intergroup_group_tp_0_via_nic_out_6_nic_in_0: tp_on_and_switch_nic_out_6_nic_in_0_0 = 1 ->
   - 7677.502879104724 hold_0_1 - 5050.470165752209 hold_0_2
   - 3762.906041760658 hold_0_3 - 2998.476157062457 hold_0_4
   - 2492.190601882461 hold_0_5 - 2132.177509385763 hold_0_6
   - 1863.048126930152 hold_0_7 - 1654.244748407672 hold_0_8
   - 1487.528219032111 hold_0_9 - 1351.338863964033 hold_0_10
   - 1237.995280512534 hold_0_11 - 1142.19374917246 hold_0_12
   - 1060.154334109086 hold_0_13 - 989.1103173416592 hold_0_14
   - 926.9900270180334 hold_0_15 - 872.2114758746661 hold_0_16
   - 823.5457495499254 hold_0_17 - 780.0237121189008 hold_0_18
   - 740.8708088446017 hold_0_19 - 705.4605657146498 hold_0_20
   - 673.2808157555227 hold_0_21 - 643.9087656615694 hold_0_22
   - 616.9923159605032 hold_0_23 - 592.2358782802088 hold_0_24
   + flow_nic_in_0_0 + flow_nic_in_0_1 + flow_nic_in_0_2 + flow_nic_in_0_3
   <= 0
 intergroup_group_tp_4_via_nic_out_6_nic_in_4: tp_on_and_switch_nic_out_6_nic_in_4_4 = 1 ->
   - 5366.149336128531 hold_4_1 - 3223.656633868661 hold_4_2
   - 2303.828417615377 hold_4_3 - 1792.392643000101 hold_4_4
   - 1466.777120905479 hold_4_5 - 1241.279690871386 hold_4_6
   - 1075.877817936759 hold_4_7 - 949.3728499935663 hold_4_8
   - 849.4875361276669 hold_4_9 - 768.6196123946033 hold_4_10
   - 701.8100166255107 hold_4_11 - 645.6859991034142 hold_4_12
   + flow_nic_in_4_4 + flow_nic_in_4_5 <= 0
 intergroup_group_tp_0_via_nic_out_8_nic_in_0: tp_on_and_switch_nic_out_8_nic_in_0_0 = 1 ->
   - 7677.502879104724 hold_0_1 - 5050.470165752209 hold_0_2
   - 3762.906041760658 hold_0_3 - 2998.476157062457 hold_0_4
   - 2492.190601882461 hold_0_5 - 2132.177509385763 hold_0_6
   - 1863.048126930152 hold_0_7 - 1654.244748407672 hold_0_8
   - 1487.528219032111 hold_0_9 - 1351.338863964033 hold_0_10
   - 1237.995280512534 hold_0_11 - 1142.19374917246 hold_0_12
   - 1060.154334109086 hold_0_13 - 989.1103173416592 hold_0_14
   - 926.9900270180334 hold_0_15 - 872.2114758746661 hold_0_16
   - 823.5457495499254 hold_0_17 - 780.0237121189008 hold_0_18
   - 740.8708088446017 hold_0_19 - 705.4605657146498 hold_0_20
   - 673.2808157555227 hold_0_21 - 643.9087656615694 hold_0_22
   - 616.9923159605032 hold_0_23 - 592.2358782802088 hold_0_24
   + flow_nic_in_0_0 + flow_nic_in_0_1 + flow_nic_in_0_2 + flow_nic_in_0_3
   <= 0
 intergroup_group_tp_4_via_nic_out_8_nic_in_4: tp_on_and_switch_nic_out_8_nic_in_4_4 = 1 ->
   - 5366.149336128531 hold_4_1 - 3223.656633868661 hold_4_2
   - 2303.828417615377 hold_4_3 - 1792.392643000101 hold_4_4
   - 1466.777120905479 hold_4_5 - 1241.279690871386 hold_4_6
   - 1075.877817936759 hold_4_7 - 949.3728499935663 hold_4_8
   - 849.4875361276669 hold_4_9 - 768.6196123946033 hold_4_10
   - 701.8100166255107 hold_4_11 - 645.6859991034142 hold_4_12
   + flow_nic_in_4_4 + flow_nic_in_4_5 <= 0
 intergroup_group_tp_6_via_nic_out_8_nic_in_6: tp_on_and_switch_nic_out_8_nic_in_6_6 = 1 ->
   - 5366.149336128531 hold_6_1 - 3223.656633868661 hold_6_2
   - 2303.828417615377 hold_6_3 - 1792.392643000101 hold_6_4
   - 1466.777120905479 hold_6_5 - 1241.279690871386 hold_6_6
   - 1075.877817936759 hold_6_7 - 949.3728499935663 hold_6_8
   - 849.4875361276669 hold_6_9 - 768.6196123946033 hold_6_10
   - 701.8100166255107 hold_6_11 - 645.6859991034142 hold_6_12
   + flow_nic_in_6_6 + flow_nic_in_6_7 <= 0
 intergroup_group_tp_0_via_nic_out_9_nic_in_0: tp_on_and_switch_nic_out_9_nic_in_0_0 = 1 ->
   - 7677.502879104724 hold_0_1 - 5050.470165752209 hold_0_2
   - 3762.906041760658 hold_0_3 - 2998.476157062457 hold_0_4
   - 2492.190601882461 hold_0_5 - 2132.177509385763 hold_0_6
   - 1863.048126930152 hold_0_7 - 1654.244748407672 hold_0_8
   - 1487.528219032111 hold_0_9 - 1351.338863964033 hold_0_10
   - 1237.995280512534 hold_0_11 - 1142.19374917246 hold_0_12
   - 1060.154334109086 hold_0_13 - 989.1103173416592 hold_0_14
   - 926.9900270180334 hold_0_15 - 872.2114758746661 hold_0_16
   - 823.5457495499254 hold_0_17 - 780.0237121189008 hold_0_18
   - 740.8708088446017 hold_0_19 - 705.4605657146498 hold_0_20
   - 673.2808157555227 hold_0_21 - 643.9087656615694 hold_0_22
   - 616.9923159605032 hold_0_23 - 592.2358782802088 hold_0_24
   + flow_nic_in_0_0 + flow_nic_in_0_1 + flow_nic_in_0_2 + flow_nic_in_0_3
   <= 0
 intergroup_group_tp_4_via_nic_out_9_nic_in_4: tp_on_and_switch_nic_out_9_nic_in_4_4 = 1 ->
   - 5366.149336128531 hold_4_1 - 3223.656633868661 hold_4_2
   - 2303.828417615377 hold_4_3 - 1792.392643000101 hold_4_4
   - 1466.777120905479 hold_4_5 - 1241.279690871386 hold_4_6
   - 1075.877817936759 hold_4_7 - 949.3728499935663 hold_4_8
   - 849.4875361276669 hold_4_9 - 768.6196123946033 hold_4_10
   - 701.8100166255107 hold_4_11 - 645.6859991034142 hold_4_12
   + flow_nic_in_4_4 + flow_nic_in_4_5 <= 0
 intergroup_group_tp_6_via_nic_out_9_nic_in_6: tp_on_and_switch_nic_out_9_nic_in_6_6 = 1 ->
   - 5366.149336128531 hold_6_1 - 3223.656633868661 hold_6_2
   - 2303.828417615377 hold_6_3 - 1792.392643000101 hold_6_4
   - 1466.777120905479 hold_6_5 - 1241.279690871386 hold_6_6
   - 1075.877817936759 hold_6_7 - 949.3728499935663 hold_6_8
   - 849.4875361276669 hold_6_9 - 768.6196123946033 hold_6_10
   - 701.8100166255107 hold_6_11 - 645.6859991034142 hold_6_12
   + flow_nic_in_6_6 + flow_nic_in_6_7 <= 0
Bounds
Binaries
 hold_0_1 hold_0_2 hold_0_3 hold_0_4 hold_0_5 hold_0_6 hold_0_7 hold_0_8
 hold_0_9 hold_0_10 hold_0_11 hold_0_12 hold_0_13 hold_0_14 hold_0_15
 hold_0_16 hold_0_17 hold_0_18 hold_0_19 hold_0_20 hold_0_21 hold_0_22
 hold_0_23 hold_0_24 hold_1_1 hold_1_2 hold_1_3 hold_1_4 hold_1_5 hold_1_6
 hold_1_7 hold_1_8 hold_1_9 hold_1_10 hold_1_11 hold_1_12 hold_1_13
 hold_1_14 hold_1_15 hold_1_16 hold_1_17 hold_1_18 hold_1_19 hold_1_20
 hold_1_21 hold_1_22 hold_1_23 hold_1_24 hold_2_1 hold_2_2 hold_2_3
 hold_2_4 hold_2_5 hold_2_6 hold_2_7 hold_2_8 hold_2_9 hold_2_10 hold_2_11
 hold_2_12 hold_2_13 hold_2_14 hold_2_15 hold_2_16 hold_2_17 hold_2_18
 hold_2_19 hold_2_20 hold_2_21 hold_2_22 hold_2_23 hold_2_24 hold_3_1
 hold_3_2 hold_3_3 hold_3_4 hold_3_5 hold_3_6 hold_3_7 hold_3_8 hold_3_9
 hold_3_10 hold_3_11 hold_3_12 hold_3_13 hold_3_14 hold_3_15 hold_3_16
 hold_3_17 hold_3_18 hold_3_19 hold_3_20 hold_3_21 hold_3_22 hold_3_23
 hold_3_24 hold_4_1 hold_4_2 hold_4_3 hold_4_4 hold_4_5 hold_4_6 hold_4_7
 hold_4_8 hold_4_9 hold_4_10 hold_4_11 hold_4_12 hold_4_13 hold_4_14
 hold_4_15 hold_4_16 hold_4_17 hold_4_18 hold_4_19 hold_4_20 hold_4_21
 hold_4_22 hold_4_23 hold_4_24 hold_5_1 hold_5_2 hold_5_3 hold_5_4 hold_5_5
 hold_5_6 hold_5_7 hold_5_8 hold_5_9 hold_5_10 hold_5_11 hold_5_12
 hold_5_13 hold_5_14 hold_5_15 hold_5_16 hold_5_17 hold_5_18 hold_5_19
 hold_5_20 hold_5_21 hold_5_22 hold_5_23 hold_5_24 hold_6_1 hold_6_2
 hold_6_3 hold_6_4 hold_6_5 hold_6_6 hold_6_7 hold_6_8 hold_6_9 hold_6_10
 hold_6_11 hold_6_12 hold_6_13 hold_6_14 hold_6_15 hold_6_16 hold_6_17
 hold_6_18 hold_6_19 hold_6_20 hold_6_21 hold_6_22 hold_6_23 hold_6_24
 hold_7_1 hold_7_2 hold_7_3 hold_7_4 hold_7_5 hold_7_6 hold_7_7 hold_7_8
 hold_7_9 hold_7_10 hold_7_11 hold_7_12 hold_7_13 hold_7_14 hold_7_15
 hold_7_16 hold_7_17 hold_7_18 hold_7_19 hold_7_20 hold_7_21 hold_7_22
 hold_7_23 hold_7_24 hold_8_1 hold_8_2 hold_8_3 hold_8_4 hold_8_5 hold_8_6
 hold_8_7 hold_8_8 hold_8_9 hold_8_10 hold_8_11 hold_8_12 hold_8_13
 hold_8_14 hold_8_15 hold_8_16 hold_8_17 hold_8_18 hold_8_19 hold_8_20
 hold_8_21 hold_8_22 hold_8_23 hold_8_24 hold_9_1 hold_9_2 hold_9_3
 hold_9_4 hold_9_5 hold_9_6 hold_9_7 hold_9_8 hold_9_9 hold_9_10 hold_9_11
 hold_9_12 hold_9_13 hold_9_14 hold_9_15 hold_9_16 hold_9_17 hold_9_18
 hold_9_19 hold_9_20 hold_9_21 hold_9_22 hold_9_23 hold_9_24
 switch_source_nic_in_0 switch_nic_out_0_sink switch_nic_out_0_nic_in_4
 switch_nic_out_0_nic_in_6 switch_nic_out_0_nic_in_8
 switch_nic_out_0_nic_in_9 switch_nic_in_0_0 switch_0_nic_out_0
 switch_nic_in_0_1 switch_1_nic_out_0 switch_nic_in_0_2 switch_2_nic_out_0
 switch_nic_in_0_3 switch_3_nic_out_0 switch_0_1 switch_0_2 switch_0_3
 switch_1_0 switch_1_2 switch_1_3 switch_2_0 switch_2_1 switch_2_3
 switch_3_0 switch_3_1 switch_3_2 switch_source_nic_in_4
 switch_nic_out_4_sink switch_nic_out_4_nic_in_0 switch_nic_out_4_nic_in_6
 switch_nic_out_4_nic_in_8 switch_nic_out_4_nic_in_9 switch_nic_in_4_4
 switch_4_nic_out_4 switch_nic_in_4_5 switch_5_nic_out_4 switch_4_5
 switch_5_4 switch_source_nic_in_6 switch_nic_out_6_sink
 switch_nic_out_6_nic_in_0 switch_nic_out_6_nic_in_4
 switch_nic_out_6_nic_in_8 switch_nic_out_6_nic_in_9 switch_nic_in_6_6
 switch_6_nic_out_6 switch_nic_in_6_7 switch_7_nic_out_6 switch_6_7
 switch_7_6 switch_source_nic_in_8 switch_nic_out_8_sink
 switch_nic_out_8_nic_in_0 switch_nic_out_8_nic_in_4
 switch_nic_out_8_nic_in_6 switch_nic_out_8_nic_in_9 switch_nic_in_8_8
 switch_8_nic_out_8 switch_source_nic_in_9 switch_nic_out_9_sink
 switch_nic_out_9_nic_in_0 switch_nic_out_9_nic_in_4
 switch_nic_out_9_nic_in_6 switch_nic_out_9_nic_in_8 switch_nic_in_9_9
 switch_9_nic_out_9 group_tp_active_0 group_tp_active_4 group_tp_active_6
 group_tp_active_8 group_tp_active_9 is_active_0 is_active_1 is_active_2
 is_active_3 is_active_4 is_active_5 is_active_6 is_active_7 is_active_8
 is_active_9 is_start_zero_0_for_source_nic_in_0
 is_start_zero_1_for_source_nic_in_0 is_start_zero_2_for_source_nic_in_0
 is_start_zero_3_for_source_nic_in_0 is_end_complete_0 is_end_complete_1
 is_end_complete_2 is_end_complete_3 pp_permission_0_1 pp_permission_0_2
 pp_permission_0_3 pp_permission_1_0 pp_permission_1_2 pp_permission_1_3
 pp_permission_2_0 pp_permission_2_1 pp_permission_2_3 pp_permission_3_0
 pp_permission_3_1 pp_permission_3_2 is_start_zero_4_for_source_nic_in_4
 is_start_zero_5_for_source_nic_in_4 is_end_complete_4 is_end_complete_5
 pp_permission_4_5 pp_permission_5_4 is_start_zero_6_for_source_nic_in_6
 is_start_zero_7_for_source_nic_in_6 is_end_complete_6 is_end_complete_7
 pp_permission_6_7 pp_permission_7_6 is_start_zero_8_for_source_nic_in_8
 is_end_complete_8 is_start_zero_9_for_source_nic_in_9 is_end_complete_9
 tp_on_and_switch_source_nic_in_0_0 tp_on_and_switch_nic_out_0_nic_in_4_4
 tp_on_and_switch_nic_out_0_nic_in_6_6 tp_on_and_switch_source_nic_in_4_4
 tp_on_and_switch_nic_out_4_nic_in_0_0
 tp_on_and_switch_nic_out_4_nic_in_6_6 tp_on_and_switch_source_nic_in_6_6
 tp_on_and_switch_nic_out_6_nic_in_0_0
 tp_on_and_switch_nic_out_6_nic_in_4_4
 tp_on_and_switch_nic_out_8_nic_in_0_0
 tp_on_and_switch_nic_out_8_nic_in_4_4
 tp_on_and_switch_nic_out_8_nic_in_6_6
 tp_on_and_switch_nic_out_9_nic_in_0_0
 tp_on_and_switch_nic_out_9_nic_in_4_4
 tp_on_and_switch_nic_out_9_nic_in_6_6
Generals
 start_0 start_1 start_2 start_3 start_4 start_5 start_6 start_7 start_8
 start_9 group_start_4 val_end_0 val_end_1 val_end_2 val_end_3 group_end_0
 group_start_6 group_start_8 group_start_9 group_start_0 val_end_4
 val_end_5 group_end_4 val_end_6 val_end_7 group_end_6 val_end_8
 group_end_8 val_end_9 group_end_9
General Constraints
 min_start_4: group_start_4 = MIN ( start_4 , start_5 )
 max_end_0: group_end_0 = MAX ( val_end_0 , val_end_1 , val_end_2 ,
   val_end_3 )
 min_start_6: group_start_6 = MIN ( start_6 , start_7 )
 min_start_8: group_start_8 = MIN ( start_8 )
 min_start_9: group_start_9 = MIN ( start_9 )
 min_start_0: group_start_0 = MIN ( start_0 , start_1 , start_2 , start_3 )
 max_end_4: group_end_4 = MAX ( val_end_4 , val_end_5 )
 max_end_6: group_end_6 = MAX ( val_end_6 , val_end_7 )
 max_end_8: group_end_8 = MAX ( val_end_8 )
 max_end_9: group_end_9 = MAX ( val_end_9 )
End
