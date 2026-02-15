\ Model 2026-02-03_19-50-16
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  0.000502008 flow_source_nic_in_0 + 0.000502008 flow_source_nic_in_1
   + 0.000502008 flow_source_nic_in_2 + 0.000502008 flow_source_nic_in_3
   + 0.000502008 flow_source_nic_in_4 + 0.000502008 flow_source_nic_in_5
   + 0.000502008 flow_source_nic_in_6 + 0.000502008 flow_source_nic_in_7
   + 0.000502008 flow_source_nic_in_8 + 0.000502008 flow_source_nic_in_9
   + 0.000502008 flow_source_nic_in_10 + 0.000502008 flow_source_nic_in_11
   + 0.000502008 flow_source_nic_in_12 + 0.000502008 flow_source_nic_in_13
   + 0.000502008 flow_source_nic_in_14 + 0.000502008 flow_source_nic_in_15
   - 0.0227272727272727 is_active_0 - 0.0227272727272727 is_active_1
   - 0.0340909090909091 is_active_2 - 0.0227272727272727 is_active_3
   - 0.0227272727272727 is_active_4 - 0.0568181818181818 is_active_5
   - 0.0340909090909091 is_active_6 - 0.0227272727272727 is_active_7
   - 0.0340909090909091 is_active_8 - 0.0227272727272727 is_active_9
   - 0.0568181818181818 is_active_10 - 0.0340909090909091 is_active_11
   - 0.0227272727272727 is_active_12 - 0.0340909090909091 is_active_13
   - 0.0340909090909091 is_active_14 - 0.0227272727272727 is_active_15
Subject To
 mem_active_constr_0: hold_0_1 + hold_0_2 + hold_0_3 + hold_0_4 + hold_0_5
   + hold_0_6 + hold_0_7 + hold_0_8 + hold_0_9 + hold_0_10 + hold_0_11
   + hold_0_12 + hold_0_13 + hold_0_14 + hold_0_15 + hold_0_16 + hold_0_17
   + hold_0_18 - 60 is_active_0 <= 0
 mem_active_constr2_0: hold_0_1 + hold_0_2 + hold_0_3 + hold_0_4 + hold_0_5
   + hold_0_6 + hold_0_7 + hold_0_8 + hold_0_9 + hold_0_10 + hold_0_11
   + hold_0_12 + hold_0_13 + hold_0_14 + hold_0_15 + hold_0_16 + hold_0_17
   + hold_0_18 - is_active_0 >= 0
 mem_active_constr_1: hold_1_1 + hold_1_2 + hold_1_3 + hold_1_4 + hold_1_5
   + hold_1_6 + hold_1_7 + hold_1_8 + hold_1_9 + hold_1_10 + hold_1_11
   + hold_1_12 + hold_1_13 + hold_1_14 + hold_1_15 + hold_1_16 + hold_1_17
   + hold_1_18 - 60 is_active_1 <= 0
 mem_active_constr2_1: hold_1_1 + hold_1_2 + hold_1_3 + hold_1_4 + hold_1_5
   + hold_1_6 + hold_1_7 + hold_1_8 + hold_1_9 + hold_1_10 + hold_1_11
   + hold_1_12 + hold_1_13 + hold_1_14 + hold_1_15 + hold_1_16 + hold_1_17
   + hold_1_18 - is_active_1 >= 0
 mem_active_constr_2: hold_2_1 + hold_2_2 + hold_2_3 + hold_2_4 + hold_2_5
   + hold_2_6 + hold_2_7 + hold_2_8 + hold_2_9 + hold_2_10 + hold_2_11
   + hold_2_12 + hold_2_13 + hold_2_14 + hold_2_15 + hold_2_16 + hold_2_17
   + hold_2_18 - 60 is_active_2 <= 0
 mem_active_constr2_2: hold_2_1 + hold_2_2 + hold_2_3 + hold_2_4 + hold_2_5
   + hold_2_6 + hold_2_7 + hold_2_8 + hold_2_9 + hold_2_10 + hold_2_11
   + hold_2_12 + hold_2_13 + hold_2_14 + hold_2_15 + hold_2_16 + hold_2_17
   + hold_2_18 - is_active_2 >= 0
 mem_active_constr_3: hold_3_1 + hold_3_2 + hold_3_3 + hold_3_4 + hold_3_5
   + hold_3_6 + hold_3_7 + hold_3_8 + hold_3_9 + hold_3_10 + hold_3_11
   + hold_3_12 + hold_3_13 + hold_3_14 + hold_3_15 + hold_3_16 + hold_3_17
   + hold_3_18 - 60 is_active_3 <= 0
 mem_active_constr2_3: hold_3_1 + hold_3_2 + hold_3_3 + hold_3_4 + hold_3_5
   + hold_3_6 + hold_3_7 + hold_3_8 + hold_3_9 + hold_3_10 + hold_3_11
   + hold_3_12 + hold_3_13 + hold_3_14 + hold_3_15 + hold_3_16 + hold_3_17
   + hold_3_18 - is_active_3 >= 0
 mem_active_constr_4: hold_4_1 + hold_4_2 + hold_4_3 + hold_4_4 + hold_4_5
   + hold_4_6 + hold_4_7 + hold_4_8 + hold_4_9 + hold_4_10 + hold_4_11
   + hold_4_12 + hold_4_13 + hold_4_14 + hold_4_15 + hold_4_16 + hold_4_17
   + hold_4_18 - 60 is_active_4 <= 0
 mem_active_constr2_4: hold_4_1 + hold_4_2 + hold_4_3 + hold_4_4 + hold_4_5
   + hold_4_6 + hold_4_7 + hold_4_8 + hold_4_9 + hold_4_10 + hold_4_11
   + hold_4_12 + hold_4_13 + hold_4_14 + hold_4_15 + hold_4_16 + hold_4_17
   + hold_4_18 - is_active_4 >= 0
 mem_active_constr_5: hold_5_1 + hold_5_2 + hold_5_3 + hold_5_4 + hold_5_5
   + hold_5_6 + hold_5_7 + hold_5_8 + hold_5_9 + hold_5_10 + hold_5_11
   + hold_5_12 + hold_5_13 + hold_5_14 + hold_5_15 + hold_5_16 + hold_5_17
   + hold_5_18 - 60 is_active_5 <= 0
 mem_active_constr2_5: hold_5_1 + hold_5_2 + hold_5_3 + hold_5_4 + hold_5_5
   + hold_5_6 + hold_5_7 + hold_5_8 + hold_5_9 + hold_5_10 + hold_5_11
   + hold_5_12 + hold_5_13 + hold_5_14 + hold_5_15 + hold_5_16 + hold_5_17
   + hold_5_18 - is_active_5 >= 0
 mem_active_constr_6: hold_6_1 + hold_6_2 + hold_6_3 + hold_6_4 + hold_6_5
   + hold_6_6 + hold_6_7 + hold_6_8 + hold_6_9 + hold_6_10 + hold_6_11
   + hold_6_12 + hold_6_13 + hold_6_14 + hold_6_15 + hold_6_16 + hold_6_17
   + hold_6_18 - 60 is_active_6 <= 0
 mem_active_constr2_6: hold_6_1 + hold_6_2 + hold_6_3 + hold_6_4 + hold_6_5
   + hold_6_6 + hold_6_7 + hold_6_8 + hold_6_9 + hold_6_10 + hold_6_11
   + hold_6_12 + hold_6_13 + hold_6_14 + hold_6_15 + hold_6_16 + hold_6_17
   + hold_6_18 - is_active_6 >= 0
 mem_active_constr_7: hold_7_1 + hold_7_2 + hold_7_3 + hold_7_4 + hold_7_5
   + hold_7_6 + hold_7_7 + hold_7_8 + hold_7_9 + hold_7_10 + hold_7_11
   + hold_7_12 + hold_7_13 + hold_7_14 + hold_7_15 + hold_7_16 + hold_7_17
   + hold_7_18 - 60 is_active_7 <= 0
 mem_active_constr2_7: hold_7_1 + hold_7_2 + hold_7_3 + hold_7_4 + hold_7_5
   + hold_7_6 + hold_7_7 + hold_7_8 + hold_7_9 + hold_7_10 + hold_7_11
   + hold_7_12 + hold_7_13 + hold_7_14 + hold_7_15 + hold_7_16 + hold_7_17
   + hold_7_18 - is_active_7 >= 0
 mem_active_constr_8: hold_8_1 + hold_8_2 + hold_8_3 + hold_8_4 + hold_8_5
   + hold_8_6 + hold_8_7 + hold_8_8 + hold_8_9 + hold_8_10 + hold_8_11
   + hold_8_12 + hold_8_13 + hold_8_14 + hold_8_15 + hold_8_16 + hold_8_17
   + hold_8_18 - 60 is_active_8 <= 0
 mem_active_constr2_8: hold_8_1 + hold_8_2 + hold_8_3 + hold_8_4 + hold_8_5
   + hold_8_6 + hold_8_7 + hold_8_8 + hold_8_9 + hold_8_10 + hold_8_11
   + hold_8_12 + hold_8_13 + hold_8_14 + hold_8_15 + hold_8_16 + hold_8_17
   + hold_8_18 - is_active_8 >= 0
 mem_active_constr_9: hold_9_1 + hold_9_2 + hold_9_3 + hold_9_4 + hold_9_5
   + hold_9_6 + hold_9_7 + hold_9_8 + hold_9_9 + hold_9_10 + hold_9_11
   + hold_9_12 + hold_9_13 + hold_9_14 + hold_9_15 + hold_9_16 + hold_9_17
   + hold_9_18 - 60 is_active_9 <= 0
 mem_active_constr2_9: hold_9_1 + hold_9_2 + hold_9_3 + hold_9_4 + hold_9_5
   + hold_9_6 + hold_9_7 + hold_9_8 + hold_9_9 + hold_9_10 + hold_9_11
   + hold_9_12 + hold_9_13 + hold_9_14 + hold_9_15 + hold_9_16 + hold_9_17
   + hold_9_18 - is_active_9 >= 0
 mem_active_constr_10: hold_10_1 + hold_10_2 + hold_10_3 + hold_10_4
   + hold_10_5 + hold_10_6 + hold_10_7 + hold_10_8 + hold_10_9 + hold_10_10
   + hold_10_11 + hold_10_12 + hold_10_13 + hold_10_14 + hold_10_15
   + hold_10_16 + hold_10_17 + hold_10_18 - 60 is_active_10 <= 0
 mem_active_constr2_10: hold_10_1 + hold_10_2 + hold_10_3 + hold_10_4
   + hold_10_5 + hold_10_6 + hold_10_7 + hold_10_8 + hold_10_9 + hold_10_10
   + hold_10_11 + hold_10_12 + hold_10_13 + hold_10_14 + hold_10_15
   + hold_10_16 + hold_10_17 + hold_10_18 - is_active_10 >= 0
 mem_active_constr_11: hold_11_1 + hold_11_2 + hold_11_3 + hold_11_4
   + hold_11_5 + hold_11_6 + hold_11_7 + hold_11_8 + hold_11_9 + hold_11_10
   + hold_11_11 + hold_11_12 + hold_11_13 + hold_11_14 + hold_11_15
   + hold_11_16 + hold_11_17 + hold_11_18 - 60 is_active_11 <= 0
 mem_active_constr2_11: hold_11_1 + hold_11_2 + hold_11_3 + hold_11_4
   + hold_11_5 + hold_11_6 + hold_11_7 + hold_11_8 + hold_11_9 + hold_11_10
   + hold_11_11 + hold_11_12 + hold_11_13 + hold_11_14 + hold_11_15
   + hold_11_16 + hold_11_17 + hold_11_18 - is_active_11 >= 0
 mem_active_constr_12: hold_12_1 + hold_12_2 + hold_12_3 + hold_12_4
   + hold_12_5 + hold_12_6 + hold_12_7 + hold_12_8 + hold_12_9 + hold_12_10
   + hold_12_11 + hold_12_12 + hold_12_13 + hold_12_14 + hold_12_15
   + hold_12_16 + hold_12_17 + hold_12_18 - 60 is_active_12 <= 0
 mem_active_constr2_12: hold_12_1 + hold_12_2 + hold_12_3 + hold_12_4
   + hold_12_5 + hold_12_6 + hold_12_7 + hold_12_8 + hold_12_9 + hold_12_10
   + hold_12_11 + hold_12_12 + hold_12_13 + hold_12_14 + hold_12_15
   + hold_12_16 + hold_12_17 + hold_12_18 - is_active_12 >= 0
 mem_active_constr_13: hold_13_1 + hold_13_2 + hold_13_3 + hold_13_4
   + hold_13_5 + hold_13_6 + hold_13_7 + hold_13_8 + hold_13_9 + hold_13_10
   + hold_13_11 + hold_13_12 + hold_13_13 + hold_13_14 + hold_13_15
   + hold_13_16 + hold_13_17 + hold_13_18 - 60 is_active_13 <= 0
 mem_active_constr2_13: hold_13_1 + hold_13_2 + hold_13_3 + hold_13_4
   + hold_13_5 + hold_13_6 + hold_13_7 + hold_13_8 + hold_13_9 + hold_13_10
   + hold_13_11 + hold_13_12 + hold_13_13 + hold_13_14 + hold_13_15
   + hold_13_16 + hold_13_17 + hold_13_18 - is_active_13 >= 0
 mem_active_constr_14: hold_14_1 + hold_14_2 + hold_14_3 + hold_14_4
   + hold_14_5 + hold_14_6 + hold_14_7 + hold_14_8 + hold_14_9 + hold_14_10
   + hold_14_11 + hold_14_12 + hold_14_13 + hold_14_14 + hold_14_15
   + hold_14_16 + hold_14_17 + hold_14_18 - 60 is_active_14 <= 0
 mem_active_constr2_14: hold_14_1 + hold_14_2 + hold_14_3 + hold_14_4
   + hold_14_5 + hold_14_6 + hold_14_7 + hold_14_8 + hold_14_9 + hold_14_10
   + hold_14_11 + hold_14_12 + hold_14_13 + hold_14_14 + hold_14_15
   + hold_14_16 + hold_14_17 + hold_14_18 - is_active_14 >= 0
 mem_active_constr_15: hold_15_1 + hold_15_2 + hold_15_3 + hold_15_4
   + hold_15_5 + hold_15_6 + hold_15_7 + hold_15_8 + hold_15_9 + hold_15_10
   + hold_15_11 + hold_15_12 + hold_15_13 + hold_15_14 + hold_15_15
   + hold_15_16 + hold_15_17 + hold_15_18 - 60 is_active_15 <= 0
 mem_active_constr2_15: hold_15_1 + hold_15_2 + hold_15_3 + hold_15_4
   + hold_15_5 + hold_15_6 + hold_15_7 + hold_15_8 + hold_15_9 + hold_15_10
   + hold_15_11 + hold_15_12 + hold_15_13 + hold_15_14 + hold_15_15
   + hold_15_16 + hold_15_17 + hold_15_18 - is_active_15 >= 0
 hold_constraint_0: hold_0_1 + hold_0_2 + hold_0_3 + hold_0_4 + hold_0_5
   + hold_0_6 + hold_0_7 + hold_0_8 + hold_0_9 + hold_0_10 + hold_0_11
   + hold_0_12 + hold_0_13 + hold_0_14 + hold_0_15 + hold_0_16 + hold_0_17
   + hold_0_18 <= 1
 end_constraint_0: start_0 + hold_0_1 + 2 hold_0_2 + 3 hold_0_3
   + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7 + 8 hold_0_8
   + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12 + 13 hold_0_13
   + 14 hold_0_14 + 15 hold_0_15 + 16 hold_0_16 + 17 hold_0_17
   + 18 hold_0_18 <= 60
 hold_constraint_1: hold_1_1 + hold_1_2 + hold_1_3 + hold_1_4 + hold_1_5
   + hold_1_6 + hold_1_7 + hold_1_8 + hold_1_9 + hold_1_10 + hold_1_11
   + hold_1_12 + hold_1_13 + hold_1_14 + hold_1_15 + hold_1_16 + hold_1_17
   + hold_1_18 <= 1
 end_constraint_1: start_1 + hold_1_1 + 2 hold_1_2 + 3 hold_1_3
   + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7 + 8 hold_1_8
   + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 + 12 hold_1_12 + 13 hold_1_13
   + 14 hold_1_14 + 15 hold_1_15 + 16 hold_1_16 + 17 hold_1_17
   + 18 hold_1_18 <= 60
 hold_constraint_2: hold_2_1 + hold_2_2 + hold_2_3 + hold_2_4 + hold_2_5
   + hold_2_6 + hold_2_7 + hold_2_8 + hold_2_9 + hold_2_10 + hold_2_11
   + hold_2_12 + hold_2_13 + hold_2_14 + hold_2_15 + hold_2_16 + hold_2_17
   + hold_2_18 <= 1
 end_constraint_2: start_2 + hold_2_1 + 2 hold_2_2 + 3 hold_2_3
   + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7 + 8 hold_2_8
   + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 12 hold_2_12 + 13 hold_2_13
   + 14 hold_2_14 + 15 hold_2_15 + 16 hold_2_16 + 17 hold_2_17
   + 18 hold_2_18 <= 60
 hold_constraint_3: hold_3_1 + hold_3_2 + hold_3_3 + hold_3_4 + hold_3_5
   + hold_3_6 + hold_3_7 + hold_3_8 + hold_3_9 + hold_3_10 + hold_3_11
   + hold_3_12 + hold_3_13 + hold_3_14 + hold_3_15 + hold_3_16 + hold_3_17
   + hold_3_18 <= 1
 end_constraint_3: start_3 + hold_3_1 + 2 hold_3_2 + 3 hold_3_3
   + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7 + 8 hold_3_8
   + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11 + 12 hold_3_12 + 13 hold_3_13
   + 14 hold_3_14 + 15 hold_3_15 + 16 hold_3_16 + 17 hold_3_17
   + 18 hold_3_18 <= 60
 hold_constraint_4: hold_4_1 + hold_4_2 + hold_4_3 + hold_4_4 + hold_4_5
   + hold_4_6 + hold_4_7 + hold_4_8 + hold_4_9 + hold_4_10 + hold_4_11
   + hold_4_12 + hold_4_13 + hold_4_14 + hold_4_15 + hold_4_16 + hold_4_17
   + hold_4_18 <= 1
 end_constraint_4: start_4 + hold_4_1 + 2 hold_4_2 + 3 hold_4_3
   + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7 + 8 hold_4_8
   + 9 hold_4_9 + 10 hold_4_10 + 11 hold_4_11 + 12 hold_4_12 + 13 hold_4_13
   + 14 hold_4_14 + 15 hold_4_15 + 16 hold_4_16 + 17 hold_4_17
   + 18 hold_4_18 <= 60
 hold_constraint_5: hold_5_1 + hold_5_2 + hold_5_3 + hold_5_4 + hold_5_5
   + hold_5_6 + hold_5_7 + hold_5_8 + hold_5_9 + hold_5_10 + hold_5_11
   + hold_5_12 + hold_5_13 + hold_5_14 + hold_5_15 + hold_5_16 + hold_5_17
   + hold_5_18 <= 1
 end_constraint_5: start_5 + hold_5_1 + 2 hold_5_2 + 3 hold_5_3
   + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7 + 8 hold_5_8
   + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12 + 13 hold_5_13
   + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16 + 17 hold_5_17
   + 18 hold_5_18 <= 60
 hold_constraint_6: hold_6_1 + hold_6_2 + hold_6_3 + hold_6_4 + hold_6_5
   + hold_6_6 + hold_6_7 + hold_6_8 + hold_6_9 + hold_6_10 + hold_6_11
   + hold_6_12 + hold_6_13 + hold_6_14 + hold_6_15 + hold_6_16 + hold_6_17
   + hold_6_18 <= 1
 end_constraint_6: start_6 + hold_6_1 + 2 hold_6_2 + 3 hold_6_3
   + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7 + 8 hold_6_8
   + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 12 hold_6_12 + 13 hold_6_13
   + 14 hold_6_14 + 15 hold_6_15 + 16 hold_6_16 + 17 hold_6_17
   + 18 hold_6_18 <= 60
 hold_constraint_7: hold_7_1 + hold_7_2 + hold_7_3 + hold_7_4 + hold_7_5
   + hold_7_6 + hold_7_7 + hold_7_8 + hold_7_9 + hold_7_10 + hold_7_11
   + hold_7_12 + hold_7_13 + hold_7_14 + hold_7_15 + hold_7_16 + hold_7_17
   + hold_7_18 <= 1
 end_constraint_7: start_7 + hold_7_1 + 2 hold_7_2 + 3 hold_7_3
   + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7 + 8 hold_7_8
   + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11 + 12 hold_7_12 + 13 hold_7_13
   + 14 hold_7_14 + 15 hold_7_15 + 16 hold_7_16 + 17 hold_7_17
   + 18 hold_7_18 <= 60
 hold_constraint_8: hold_8_1 + hold_8_2 + hold_8_3 + hold_8_4 + hold_8_5
   + hold_8_6 + hold_8_7 + hold_8_8 + hold_8_9 + hold_8_10 + hold_8_11
   + hold_8_12 + hold_8_13 + hold_8_14 + hold_8_15 + hold_8_16 + hold_8_17
   + hold_8_18 <= 1
 end_constraint_8: start_8 + hold_8_1 + 2 hold_8_2 + 3 hold_8_3
   + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7 + 8 hold_8_8
   + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 12 hold_8_12 + 13 hold_8_13
   + 14 hold_8_14 + 15 hold_8_15 + 16 hold_8_16 + 17 hold_8_17
   + 18 hold_8_18 <= 60
 hold_constraint_9: hold_9_1 + hold_9_2 + hold_9_3 + hold_9_4 + hold_9_5
   + hold_9_6 + hold_9_7 + hold_9_8 + hold_9_9 + hold_9_10 + hold_9_11
   + hold_9_12 + hold_9_13 + hold_9_14 + hold_9_15 + hold_9_16 + hold_9_17
   + hold_9_18 <= 1
 end_constraint_9: start_9 + hold_9_1 + 2 hold_9_2 + 3 hold_9_3
   + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7 + 8 hold_9_8
   + 9 hold_9_9 + 10 hold_9_10 + 11 hold_9_11 + 12 hold_9_12 + 13 hold_9_13
   + 14 hold_9_14 + 15 hold_9_15 + 16 hold_9_16 + 17 hold_9_17
   + 18 hold_9_18 <= 60
 hold_constraint_10: hold_10_1 + hold_10_2 + hold_10_3 + hold_10_4
   + hold_10_5 + hold_10_6 + hold_10_7 + hold_10_8 + hold_10_9 + hold_10_10
   + hold_10_11 + hold_10_12 + hold_10_13 + hold_10_14 + hold_10_15
   + hold_10_16 + hold_10_17 + hold_10_18 <= 1
 end_constraint_10: start_10 + hold_10_1 + 2 hold_10_2 + 3 hold_10_3
   + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7 + 8 hold_10_8
   + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11 + 12 hold_10_12
   + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15 + 16 hold_10_16
   + 17 hold_10_17 + 18 hold_10_18 <= 60
 hold_constraint_11: hold_11_1 + hold_11_2 + hold_11_3 + hold_11_4
   + hold_11_5 + hold_11_6 + hold_11_7 + hold_11_8 + hold_11_9 + hold_11_10
   + hold_11_11 + hold_11_12 + hold_11_13 + hold_11_14 + hold_11_15
   + hold_11_16 + hold_11_17 + hold_11_18 <= 1
 end_constraint_11: start_11 + hold_11_1 + 2 hold_11_2 + 3 hold_11_3
   + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7 + 8 hold_11_8
   + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11 + 12 hold_11_12
   + 13 hold_11_13 + 14 hold_11_14 + 15 hold_11_15 + 16 hold_11_16
   + 17 hold_11_17 + 18 hold_11_18 <= 60
 hold_constraint_12: hold_12_1 + hold_12_2 + hold_12_3 + hold_12_4
   + hold_12_5 + hold_12_6 + hold_12_7 + hold_12_8 + hold_12_9 + hold_12_10
   + hold_12_11 + hold_12_12 + hold_12_13 + hold_12_14 + hold_12_15
   + hold_12_16 + hold_12_17 + hold_12_18 <= 1
 end_constraint_12: start_12 + hold_12_1 + 2 hold_12_2 + 3 hold_12_3
   + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6 + 7 hold_12_7 + 8 hold_12_8
   + 9 hold_12_9 + 10 hold_12_10 + 11 hold_12_11 + 12 hold_12_12
   + 13 hold_12_13 + 14 hold_12_14 + 15 hold_12_15 + 16 hold_12_16
   + 17 hold_12_17 + 18 hold_12_18 <= 60
 hold_constraint_13: hold_13_1 + hold_13_2 + hold_13_3 + hold_13_4
   + hold_13_5 + hold_13_6 + hold_13_7 + hold_13_8 + hold_13_9 + hold_13_10
   + hold_13_11 + hold_13_12 + hold_13_13 + hold_13_14 + hold_13_15
   + hold_13_16 + hold_13_17 + hold_13_18 <= 1
 end_constraint_13: start_13 + hold_13_1 + 2 hold_13_2 + 3 hold_13_3
   + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6 + 7 hold_13_7 + 8 hold_13_8
   + 9 hold_13_9 + 10 hold_13_10 + 11 hold_13_11 + 12 hold_13_12
   + 13 hold_13_13 + 14 hold_13_14 + 15 hold_13_15 + 16 hold_13_16
   + 17 hold_13_17 + 18 hold_13_18 <= 60
 hold_constraint_14: hold_14_1 + hold_14_2 + hold_14_3 + hold_14_4
   + hold_14_5 + hold_14_6 + hold_14_7 + hold_14_8 + hold_14_9 + hold_14_10
   + hold_14_11 + hold_14_12 + hold_14_13 + hold_14_14 + hold_14_15
   + hold_14_16 + hold_14_17 + hold_14_18 <= 1
 end_constraint_14: start_14 + hold_14_1 + 2 hold_14_2 + 3 hold_14_3
   + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6 + 7 hold_14_7 + 8 hold_14_8
   + 9 hold_14_9 + 10 hold_14_10 + 11 hold_14_11 + 12 hold_14_12
   + 13 hold_14_13 + 14 hold_14_14 + 15 hold_14_15 + 16 hold_14_16
   + 17 hold_14_17 + 18 hold_14_18 <= 60
 hold_constraint_15: hold_15_1 + hold_15_2 + hold_15_3 + hold_15_4
   + hold_15_5 + hold_15_6 + hold_15_7 + hold_15_8 + hold_15_9 + hold_15_10
   + hold_15_11 + hold_15_12 + hold_15_13 + hold_15_14 + hold_15_15
   + hold_15_16 + hold_15_17 + hold_15_18 <= 1
 end_constraint_15: start_15 + hold_15_1 + 2 hold_15_2 + 3 hold_15_3
   + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6 + 7 hold_15_7 + 8 hold_15_8
   + 9 hold_15_9 + 10 hold_15_10 + 11 hold_15_11 + 12 hold_15_12
   + 13 hold_15_13 + 14 hold_15_14 + 15 hold_15_15 + 16 hold_15_16
   + 17 hold_15_17 + 18 hold_15_18 <= 60
 node_flow_constraint_nic_in_0: flow_source_nic_in_0 - flow_nic_in_0_0
   + flow_nic_out_1_nic_in_0 + flow_nic_out_2_nic_in_0
   + flow_nic_out_3_nic_in_0 + flow_nic_out_4_nic_in_0
   + flow_nic_out_5_nic_in_0 + flow_nic_out_6_nic_in_0
   + flow_nic_out_7_nic_in_0 + flow_nic_out_8_nic_in_0
   + flow_nic_out_9_nic_in_0 + flow_nic_out_10_nic_in_0
   + flow_nic_out_11_nic_in_0 + flow_nic_out_12_nic_in_0
   + flow_nic_out_13_nic_in_0 + flow_nic_out_14_nic_in_0
   + flow_nic_out_15_nic_in_0 = 0
 node_flow_constraint_nic_out_0: - flow_nic_out_0_sink
   - flow_nic_out_0_nic_in_1 - flow_nic_out_0_nic_in_2
   - flow_nic_out_0_nic_in_3 - flow_nic_out_0_nic_in_4
   - flow_nic_out_0_nic_in_5 - flow_nic_out_0_nic_in_6
   - flow_nic_out_0_nic_in_7 - flow_nic_out_0_nic_in_8
   - flow_nic_out_0_nic_in_9 - flow_nic_out_0_nic_in_10
   - flow_nic_out_0_nic_in_11 - flow_nic_out_0_nic_in_12
   - flow_nic_out_0_nic_in_13 - flow_nic_out_0_nic_in_14
   - flow_nic_out_0_nic_in_15 + flow_0_nic_out_0 = 0
 node_flow_constraint_nic_in_1: flow_nic_out_0_nic_in_1
   + flow_source_nic_in_1 - flow_nic_in_1_1 + flow_nic_out_2_nic_in_1
   + flow_nic_out_3_nic_in_1 + flow_nic_out_4_nic_in_1
   + flow_nic_out_5_nic_in_1 + flow_nic_out_6_nic_in_1
   + flow_nic_out_7_nic_in_1 + flow_nic_out_8_nic_in_1
   + flow_nic_out_9_nic_in_1 + flow_nic_out_10_nic_in_1
   + flow_nic_out_11_nic_in_1 + flow_nic_out_12_nic_in_1
   + flow_nic_out_13_nic_in_1 + flow_nic_out_14_nic_in_1
   + flow_nic_out_15_nic_in_1 = 0
 node_flow_constraint_nic_out_1: - flow_nic_out_1_sink
   - flow_nic_out_1_nic_in_0 - flow_nic_out_1_nic_in_2
   - flow_nic_out_1_nic_in_3 - flow_nic_out_1_nic_in_4
   - flow_nic_out_1_nic_in_5 - flow_nic_out_1_nic_in_6
   - flow_nic_out_1_nic_in_7 - flow_nic_out_1_nic_in_8
   - flow_nic_out_1_nic_in_9 - flow_nic_out_1_nic_in_10
   - flow_nic_out_1_nic_in_11 - flow_nic_out_1_nic_in_12
   - flow_nic_out_1_nic_in_13 - flow_nic_out_1_nic_in_14
   - flow_nic_out_1_nic_in_15 + flow_1_nic_out_1 = 0
 node_flow_constraint_nic_in_2: flow_nic_out_0_nic_in_2
   + flow_nic_out_1_nic_in_2 + flow_source_nic_in_2 - flow_nic_in_2_2
   + flow_nic_out_3_nic_in_2 + flow_nic_out_4_nic_in_2
   + flow_nic_out_5_nic_in_2 + flow_nic_out_6_nic_in_2
   + flow_nic_out_7_nic_in_2 + flow_nic_out_8_nic_in_2
   + flow_nic_out_9_nic_in_2 + flow_nic_out_10_nic_in_2
   + flow_nic_out_11_nic_in_2 + flow_nic_out_12_nic_in_2
   + flow_nic_out_13_nic_in_2 + flow_nic_out_14_nic_in_2
   + flow_nic_out_15_nic_in_2 = 0
 node_flow_constraint_nic_out_2: - flow_nic_out_2_sink
   - flow_nic_out_2_nic_in_0 - flow_nic_out_2_nic_in_1
   - flow_nic_out_2_nic_in_3 - flow_nic_out_2_nic_in_4
   - flow_nic_out_2_nic_in_5 - flow_nic_out_2_nic_in_6
   - flow_nic_out_2_nic_in_7 - flow_nic_out_2_nic_in_8
   - flow_nic_out_2_nic_in_9 - flow_nic_out_2_nic_in_10
   - flow_nic_out_2_nic_in_11 - flow_nic_out_2_nic_in_12
   - flow_nic_out_2_nic_in_13 - flow_nic_out_2_nic_in_14
   - flow_nic_out_2_nic_in_15 + flow_2_nic_out_2 = 0
 node_flow_constraint_nic_in_3: flow_nic_out_0_nic_in_3
   + flow_nic_out_1_nic_in_3 + flow_nic_out_2_nic_in_3
   + flow_source_nic_in_3 - flow_nic_in_3_3 + flow_nic_out_4_nic_in_3
   + flow_nic_out_5_nic_in_3 + flow_nic_out_6_nic_in_3
   + flow_nic_out_7_nic_in_3 + flow_nic_out_8_nic_in_3
   + flow_nic_out_9_nic_in_3 + flow_nic_out_10_nic_in_3
   + flow_nic_out_11_nic_in_3 + flow_nic_out_12_nic_in_3
   + flow_nic_out_13_nic_in_3 + flow_nic_out_14_nic_in_3
   + flow_nic_out_15_nic_in_3 = 0
 node_flow_constraint_nic_out_3: - flow_nic_out_3_sink
   - flow_nic_out_3_nic_in_0 - flow_nic_out_3_nic_in_1
   - flow_nic_out_3_nic_in_2 - flow_nic_out_3_nic_in_4
   - flow_nic_out_3_nic_in_5 - flow_nic_out_3_nic_in_6
   - flow_nic_out_3_nic_in_7 - flow_nic_out_3_nic_in_8
   - flow_nic_out_3_nic_in_9 - flow_nic_out_3_nic_in_10
   - flow_nic_out_3_nic_in_11 - flow_nic_out_3_nic_in_12
   - flow_nic_out_3_nic_in_13 - flow_nic_out_3_nic_in_14
   - flow_nic_out_3_nic_in_15 + flow_3_nic_out_3 = 0
 node_flow_constraint_nic_in_4: flow_nic_out_0_nic_in_4
   + flow_nic_out_1_nic_in_4 + flow_nic_out_2_nic_in_4
   + flow_nic_out_3_nic_in_4 + flow_source_nic_in_4 - flow_nic_in_4_4
   + flow_nic_out_5_nic_in_4 + flow_nic_out_6_nic_in_4
   + flow_nic_out_7_nic_in_4 + flow_nic_out_8_nic_in_4
   + flow_nic_out_9_nic_in_4 + flow_nic_out_10_nic_in_4
   + flow_nic_out_11_nic_in_4 + flow_nic_out_12_nic_in_4
   + flow_nic_out_13_nic_in_4 + flow_nic_out_14_nic_in_4
   + flow_nic_out_15_nic_in_4 = 0
 node_flow_constraint_nic_out_4: - flow_nic_out_4_sink
   - flow_nic_out_4_nic_in_0 - flow_nic_out_4_nic_in_1
   - flow_nic_out_4_nic_in_2 - flow_nic_out_4_nic_in_3
   - flow_nic_out_4_nic_in_5 - flow_nic_out_4_nic_in_6
   - flow_nic_out_4_nic_in_7 - flow_nic_out_4_nic_in_8
   - flow_nic_out_4_nic_in_9 - flow_nic_out_4_nic_in_10
   - flow_nic_out_4_nic_in_11 - flow_nic_out_4_nic_in_12
   - flow_nic_out_4_nic_in_13 - flow_nic_out_4_nic_in_14
   - flow_nic_out_4_nic_in_15 + flow_4_nic_out_4 = 0
 node_flow_constraint_nic_in_5: flow_nic_out_0_nic_in_5
   + flow_nic_out_1_nic_in_5 + flow_nic_out_2_nic_in_5
   + flow_nic_out_3_nic_in_5 + flow_nic_out_4_nic_in_5
   + flow_source_nic_in_5 - flow_nic_in_5_5 + flow_nic_out_6_nic_in_5
   + flow_nic_out_7_nic_in_5 + flow_nic_out_8_nic_in_5
   + flow_nic_out_9_nic_in_5 + flow_nic_out_10_nic_in_5
   + flow_nic_out_11_nic_in_5 + flow_nic_out_12_nic_in_5
   + flow_nic_out_13_nic_in_5 + flow_nic_out_14_nic_in_5
   + flow_nic_out_15_nic_in_5 = 0
 node_flow_constraint_nic_out_5: - flow_nic_out_5_sink
   - flow_nic_out_5_nic_in_0 - flow_nic_out_5_nic_in_1
   - flow_nic_out_5_nic_in_2 - flow_nic_out_5_nic_in_3
   - flow_nic_out_5_nic_in_4 - flow_nic_out_5_nic_in_6
   - flow_nic_out_5_nic_in_7 - flow_nic_out_5_nic_in_8
   - flow_nic_out_5_nic_in_9 - flow_nic_out_5_nic_in_10
   - flow_nic_out_5_nic_in_11 - flow_nic_out_5_nic_in_12
   - flow_nic_out_5_nic_in_13 - flow_nic_out_5_nic_in_14
   - flow_nic_out_5_nic_in_15 + flow_5_nic_out_5 = 0
 node_flow_constraint_nic_in_6: flow_nic_out_0_nic_in_6
   + flow_nic_out_1_nic_in_6 + flow_nic_out_2_nic_in_6
   + flow_nic_out_3_nic_in_6 + flow_nic_out_4_nic_in_6
   + flow_nic_out_5_nic_in_6 + flow_source_nic_in_6 - flow_nic_in_6_6
   + flow_nic_out_7_nic_in_6 + flow_nic_out_8_nic_in_6
   + flow_nic_out_9_nic_in_6 + flow_nic_out_10_nic_in_6
   + flow_nic_out_11_nic_in_6 + flow_nic_out_12_nic_in_6
   + flow_nic_out_13_nic_in_6 + flow_nic_out_14_nic_in_6
   + flow_nic_out_15_nic_in_6 = 0
 node_flow_constraint_nic_out_6: - flow_nic_out_6_sink
   - flow_nic_out_6_nic_in_0 - flow_nic_out_6_nic_in_1
   - flow_nic_out_6_nic_in_2 - flow_nic_out_6_nic_in_3
   - flow_nic_out_6_nic_in_4 - flow_nic_out_6_nic_in_5
   - flow_nic_out_6_nic_in_7 - flow_nic_out_6_nic_in_8
   - flow_nic_out_6_nic_in_9 - flow_nic_out_6_nic_in_10
   - flow_nic_out_6_nic_in_11 - flow_nic_out_6_nic_in_12
   - flow_nic_out_6_nic_in_13 - flow_nic_out_6_nic_in_14
   - flow_nic_out_6_nic_in_15 + flow_6_nic_out_6 = 0
 node_flow_constraint_nic_in_7: flow_nic_out_0_nic_in_7
   + flow_nic_out_1_nic_in_7 + flow_nic_out_2_nic_in_7
   + flow_nic_out_3_nic_in_7 + flow_nic_out_4_nic_in_7
   + flow_nic_out_5_nic_in_7 + flow_nic_out_6_nic_in_7
   + flow_source_nic_in_7 - flow_nic_in_7_7 + flow_nic_out_8_nic_in_7
   + flow_nic_out_9_nic_in_7 + flow_nic_out_10_nic_in_7
   + flow_nic_out_11_nic_in_7 + flow_nic_out_12_nic_in_7
   + flow_nic_out_13_nic_in_7 + flow_nic_out_14_nic_in_7
   + flow_nic_out_15_nic_in_7 = 0
 node_flow_constraint_nic_out_7: - flow_nic_out_7_sink
   - flow_nic_out_7_nic_in_0 - flow_nic_out_7_nic_in_1
   - flow_nic_out_7_nic_in_2 - flow_nic_out_7_nic_in_3
   - flow_nic_out_7_nic_in_4 - flow_nic_out_7_nic_in_5
   - flow_nic_out_7_nic_in_6 - flow_nic_out_7_nic_in_8
   - flow_nic_out_7_nic_in_9 - flow_nic_out_7_nic_in_10
   - flow_nic_out_7_nic_in_11 - flow_nic_out_7_nic_in_12
   - flow_nic_out_7_nic_in_13 - flow_nic_out_7_nic_in_14
   - flow_nic_out_7_nic_in_15 + flow_7_nic_out_7 = 0
 node_flow_constraint_nic_in_8: flow_nic_out_0_nic_in_8
   + flow_nic_out_1_nic_in_8 + flow_nic_out_2_nic_in_8
   + flow_nic_out_3_nic_in_8 + flow_nic_out_4_nic_in_8
   + flow_nic_out_5_nic_in_8 + flow_nic_out_6_nic_in_8
   + flow_nic_out_7_nic_in_8 + flow_source_nic_in_8 - flow_nic_in_8_8
   + flow_nic_out_9_nic_in_8 + flow_nic_out_10_nic_in_8
   + flow_nic_out_11_nic_in_8 + flow_nic_out_12_nic_in_8
   + flow_nic_out_13_nic_in_8 + flow_nic_out_14_nic_in_8
   + flow_nic_out_15_nic_in_8 = 0
 node_flow_constraint_nic_out_8: - flow_nic_out_8_sink
   - flow_nic_out_8_nic_in_0 - flow_nic_out_8_nic_in_1
   - flow_nic_out_8_nic_in_2 - flow_nic_out_8_nic_in_3
   - flow_nic_out_8_nic_in_4 - flow_nic_out_8_nic_in_5
   - flow_nic_out_8_nic_in_6 - flow_nic_out_8_nic_in_7
   - flow_nic_out_8_nic_in_9 - flow_nic_out_8_nic_in_10
   - flow_nic_out_8_nic_in_11 - flow_nic_out_8_nic_in_12
   - flow_nic_out_8_nic_in_13 - flow_nic_out_8_nic_in_14
   - flow_nic_out_8_nic_in_15 + flow_8_nic_out_8 = 0
 node_flow_constraint_nic_in_9: flow_nic_out_0_nic_in_9
   + flow_nic_out_1_nic_in_9 + flow_nic_out_2_nic_in_9
   + flow_nic_out_3_nic_in_9 + flow_nic_out_4_nic_in_9
   + flow_nic_out_5_nic_in_9 + flow_nic_out_6_nic_in_9
   + flow_nic_out_7_nic_in_9 + flow_nic_out_8_nic_in_9
   + flow_source_nic_in_9 - flow_nic_in_9_9 + flow_nic_out_10_nic_in_9
   + flow_nic_out_11_nic_in_9 + flow_nic_out_12_nic_in_9
   + flow_nic_out_13_nic_in_9 + flow_nic_out_14_nic_in_9
   + flow_nic_out_15_nic_in_9 = 0
 node_flow_constraint_nic_out_9: - flow_nic_out_9_sink
   - flow_nic_out_9_nic_in_0 - flow_nic_out_9_nic_in_1
   - flow_nic_out_9_nic_in_2 - flow_nic_out_9_nic_in_3
   - flow_nic_out_9_nic_in_4 - flow_nic_out_9_nic_in_5
   - flow_nic_out_9_nic_in_6 - flow_nic_out_9_nic_in_7
   - flow_nic_out_9_nic_in_8 - flow_nic_out_9_nic_in_10
   - flow_nic_out_9_nic_in_11 - flow_nic_out_9_nic_in_12
   - flow_nic_out_9_nic_in_13 - flow_nic_out_9_nic_in_14
   - flow_nic_out_9_nic_in_15 + flow_9_nic_out_9 = 0
 node_flow_constraint_nic_in_10: flow_nic_out_0_nic_in_10
   + flow_nic_out_1_nic_in_10 + flow_nic_out_2_nic_in_10
   + flow_nic_out_3_nic_in_10 + flow_nic_out_4_nic_in_10
   + flow_nic_out_5_nic_in_10 + flow_nic_out_6_nic_in_10
   + flow_nic_out_7_nic_in_10 + flow_nic_out_8_nic_in_10
   + flow_nic_out_9_nic_in_10 + flow_source_nic_in_10 - flow_nic_in_10_10
   + flow_nic_out_11_nic_in_10 + flow_nic_out_12_nic_in_10
   + flow_nic_out_13_nic_in_10 + flow_nic_out_14_nic_in_10
   + flow_nic_out_15_nic_in_10 = 0
 node_flow_constraint_nic_out_10: - flow_nic_out_10_sink
   - flow_nic_out_10_nic_in_0 - flow_nic_out_10_nic_in_1
   - flow_nic_out_10_nic_in_2 - flow_nic_out_10_nic_in_3
   - flow_nic_out_10_nic_in_4 - flow_nic_out_10_nic_in_5
   - flow_nic_out_10_nic_in_6 - flow_nic_out_10_nic_in_7
   - flow_nic_out_10_nic_in_8 - flow_nic_out_10_nic_in_9
   - flow_nic_out_10_nic_in_11 - flow_nic_out_10_nic_in_12
   - flow_nic_out_10_nic_in_13 - flow_nic_out_10_nic_in_14
   - flow_nic_out_10_nic_in_15 + flow_10_nic_out_10 = 0
 node_flow_constraint_nic_in_11: flow_nic_out_0_nic_in_11
   + flow_nic_out_1_nic_in_11 + flow_nic_out_2_nic_in_11
   + flow_nic_out_3_nic_in_11 + flow_nic_out_4_nic_in_11
   + flow_nic_out_5_nic_in_11 + flow_nic_out_6_nic_in_11
   + flow_nic_out_7_nic_in_11 + flow_nic_out_8_nic_in_11
   + flow_nic_out_9_nic_in_11 + flow_nic_out_10_nic_in_11
   + flow_source_nic_in_11 - flow_nic_in_11_11 + flow_nic_out_12_nic_in_11
   + flow_nic_out_13_nic_in_11 + flow_nic_out_14_nic_in_11
   + flow_nic_out_15_nic_in_11 = 0
 node_flow_constraint_nic_out_11: - flow_nic_out_11_sink
   - flow_nic_out_11_nic_in_0 - flow_nic_out_11_nic_in_1
   - flow_nic_out_11_nic_in_2 - flow_nic_out_11_nic_in_3
   - flow_nic_out_11_nic_in_4 - flow_nic_out_11_nic_in_5
   - flow_nic_out_11_nic_in_6 - flow_nic_out_11_nic_in_7
   - flow_nic_out_11_nic_in_8 - flow_nic_out_11_nic_in_9
   - flow_nic_out_11_nic_in_10 - flow_nic_out_11_nic_in_12
   - flow_nic_out_11_nic_in_13 - flow_nic_out_11_nic_in_14
   - flow_nic_out_11_nic_in_15 + flow_11_nic_out_11 = 0
 node_flow_constraint_nic_in_12: flow_nic_out_0_nic_in_12
   + flow_nic_out_1_nic_in_12 + flow_nic_out_2_nic_in_12
   + flow_nic_out_3_nic_in_12 + flow_nic_out_4_nic_in_12
   + flow_nic_out_5_nic_in_12 + flow_nic_out_6_nic_in_12
   + flow_nic_out_7_nic_in_12 + flow_nic_out_8_nic_in_12
   + flow_nic_out_9_nic_in_12 + flow_nic_out_10_nic_in_12
   + flow_nic_out_11_nic_in_12 + flow_source_nic_in_12 - flow_nic_in_12_12
   + flow_nic_out_13_nic_in_12 + flow_nic_out_14_nic_in_12
   + flow_nic_out_15_nic_in_12 = 0
 node_flow_constraint_nic_out_12: - flow_nic_out_12_sink
   - flow_nic_out_12_nic_in_0 - flow_nic_out_12_nic_in_1
   - flow_nic_out_12_nic_in_2 - flow_nic_out_12_nic_in_3
   - flow_nic_out_12_nic_in_4 - flow_nic_out_12_nic_in_5
   - flow_nic_out_12_nic_in_6 - flow_nic_out_12_nic_in_7
   - flow_nic_out_12_nic_in_8 - flow_nic_out_12_nic_in_9
   - flow_nic_out_12_nic_in_10 - flow_nic_out_12_nic_in_11
   - flow_nic_out_12_nic_in_13 - flow_nic_out_12_nic_in_14
   - flow_nic_out_12_nic_in_15 + flow_12_nic_out_12 = 0
 node_flow_constraint_nic_in_13: flow_nic_out_0_nic_in_13
   + flow_nic_out_1_nic_in_13 + flow_nic_out_2_nic_in_13
   + flow_nic_out_3_nic_in_13 + flow_nic_out_4_nic_in_13
   + flow_nic_out_5_nic_in_13 + flow_nic_out_6_nic_in_13
   + flow_nic_out_7_nic_in_13 + flow_nic_out_8_nic_in_13
   + flow_nic_out_9_nic_in_13 + flow_nic_out_10_nic_in_13
   + flow_nic_out_11_nic_in_13 + flow_nic_out_12_nic_in_13
   + flow_source_nic_in_13 - flow_nic_in_13_13 + flow_nic_out_14_nic_in_13
   + flow_nic_out_15_nic_in_13 = 0
 node_flow_constraint_nic_out_13: - flow_nic_out_13_sink
   - flow_nic_out_13_nic_in_0 - flow_nic_out_13_nic_in_1
   - flow_nic_out_13_nic_in_2 - flow_nic_out_13_nic_in_3
   - flow_nic_out_13_nic_in_4 - flow_nic_out_13_nic_in_5
   - flow_nic_out_13_nic_in_6 - flow_nic_out_13_nic_in_7
   - flow_nic_out_13_nic_in_8 - flow_nic_out_13_nic_in_9
   - flow_nic_out_13_nic_in_10 - flow_nic_out_13_nic_in_11
   - flow_nic_out_13_nic_in_12 - flow_nic_out_13_nic_in_14
   - flow_nic_out_13_nic_in_15 + flow_13_nic_out_13 = 0
 node_flow_constraint_nic_in_14: flow_nic_out_0_nic_in_14
   + flow_nic_out_1_nic_in_14 + flow_nic_out_2_nic_in_14
   + flow_nic_out_3_nic_in_14 + flow_nic_out_4_nic_in_14
   + flow_nic_out_5_nic_in_14 + flow_nic_out_6_nic_in_14
   + flow_nic_out_7_nic_in_14 + flow_nic_out_8_nic_in_14
   + flow_nic_out_9_nic_in_14 + flow_nic_out_10_nic_in_14
   + flow_nic_out_11_nic_in_14 + flow_nic_out_12_nic_in_14
   + flow_nic_out_13_nic_in_14 + flow_source_nic_in_14 - flow_nic_in_14_14
   + flow_nic_out_15_nic_in_14 = 0
 node_flow_constraint_nic_out_14: - flow_nic_out_14_sink
   - flow_nic_out_14_nic_in_0 - flow_nic_out_14_nic_in_1
   - flow_nic_out_14_nic_in_2 - flow_nic_out_14_nic_in_3
   - flow_nic_out_14_nic_in_4 - flow_nic_out_14_nic_in_5
   - flow_nic_out_14_nic_in_6 - flow_nic_out_14_nic_in_7
   - flow_nic_out_14_nic_in_8 - flow_nic_out_14_nic_in_9
   - flow_nic_out_14_nic_in_10 - flow_nic_out_14_nic_in_11
   - flow_nic_out_14_nic_in_12 - flow_nic_out_14_nic_in_13
   - flow_nic_out_14_nic_in_15 + flow_14_nic_out_14 = 0
 node_flow_constraint_nic_in_15: flow_nic_out_0_nic_in_15
   + flow_nic_out_1_nic_in_15 + flow_nic_out_2_nic_in_15
   + flow_nic_out_3_nic_in_15 + flow_nic_out_4_nic_in_15
   + flow_nic_out_5_nic_in_15 + flow_nic_out_6_nic_in_15
   + flow_nic_out_7_nic_in_15 + flow_nic_out_8_nic_in_15
   + flow_nic_out_9_nic_in_15 + flow_nic_out_10_nic_in_15
   + flow_nic_out_11_nic_in_15 + flow_nic_out_12_nic_in_15
   + flow_nic_out_13_nic_in_15 + flow_nic_out_14_nic_in_15
   + flow_source_nic_in_15 - flow_nic_in_15_15 = 0
 node_flow_constraint_nic_out_15: - flow_nic_out_15_sink
   - flow_nic_out_15_nic_in_0 - flow_nic_out_15_nic_in_1
   - flow_nic_out_15_nic_in_2 - flow_nic_out_15_nic_in_3
   - flow_nic_out_15_nic_in_4 - flow_nic_out_15_nic_in_5
   - flow_nic_out_15_nic_in_6 - flow_nic_out_15_nic_in_7
   - flow_nic_out_15_nic_in_8 - flow_nic_out_15_nic_in_9
   - flow_nic_out_15_nic_in_10 - flow_nic_out_15_nic_in_11
   - flow_nic_out_15_nic_in_12 - flow_nic_out_15_nic_in_13
   - flow_nic_out_15_nic_in_14 + flow_15_nic_out_15 = 0
 node_flow_constraint_0: flow_nic_in_0_0 - flow_0_nic_out_0 = 0
 node_flow_constraint_1: flow_nic_in_1_1 - flow_1_nic_out_1 = 0
 node_flow_constraint_2: flow_nic_in_2_2 - flow_2_nic_out_2 = 0
 node_flow_constraint_3: flow_nic_in_3_3 - flow_3_nic_out_3 = 0
 node_flow_constraint_4: flow_nic_in_4_4 - flow_4_nic_out_4 = 0
 node_flow_constraint_5: flow_nic_in_5_5 - flow_5_nic_out_5 = 0
 node_flow_constraint_6: flow_nic_in_6_6 - flow_6_nic_out_6 = 0
 node_flow_constraint_7: flow_nic_in_7_7 - flow_7_nic_out_7 = 0
 node_flow_constraint_8: flow_nic_in_8_8 - flow_8_nic_out_8 = 0
 node_flow_constraint_9: flow_nic_in_9_9 - flow_9_nic_out_9 = 0
 node_flow_constraint_10: flow_nic_in_10_10 - flow_10_nic_out_10 = 0
 node_flow_constraint_11: flow_nic_in_11_11 - flow_11_nic_out_11 = 0
 node_flow_constraint_12: flow_nic_in_12_12 - flow_12_nic_out_12 = 0
 node_flow_constraint_13: flow_nic_in_13_13 - flow_13_nic_out_13 = 0
 node_flow_constraint_14: flow_nic_in_14_14 - flow_14_nic_out_14 = 0
 node_flow_constraint_15: flow_nic_in_15_15 - flow_15_nic_out_15 = 0
 constr_is_start_zero_0: start_0 + 60 is_start_zero_0_for_source_nic_in_0
   <= 60
 constr_edge_switch_start_zero_0_OR: switch_source_nic_in_0
   - is_start_zero_0_for_source_nic_in_0 <= 0
 constr_is_end_complete_0: start_0 + hold_0_1 + 2 hold_0_2 + 3 hold_0_3
   + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7 + 8 hold_0_8
   + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12 + 13 hold_0_13
   + 14 hold_0_14 + 15 hold_0_15 + 16 hold_0_16 + 17 hold_0_17
   + 18 hold_0_18 - 60 is_end_complete_0 >= 0
 constr_edge_switch_end_max_0_OR: switch_nic_out_0_sink - is_end_complete_0
   <= 0
 constr_pair_ok1_0_1: start_0 - start_1 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 8 hold_0_8 + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12
   + 13 hold_0_13 + 14 hold_0_14 + 15 hold_0_15 + 16 hold_0_16
   + 17 hold_0_17 + 18 hold_0_18 + 60 pair_ok_0_1_for_nic_out_0_nic_in_1
   <= 60
 constr_pair_ok2_0_1: - start_0 + start_1 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11 - 12 hold_0_12
   - 13 hold_0_13 - 14 hold_0_14 - 15 hold_0_15 - 16 hold_0_16
   - 17 hold_0_17 - 18 hold_0_18 + 60 pair_ok_0_1_for_nic_out_0_nic_in_1
   <= 60
 constr_edge_switch_end_start_0_1_OR: switch_nic_out_0_nic_in_1
   - pair_ok_0_1_for_nic_out_0_nic_in_1 <= 0
 constr_pair_ok1_0_2: start_0 - start_2 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 8 hold_0_8 + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12
   + 13 hold_0_13 + 14 hold_0_14 + 15 hold_0_15 + 16 hold_0_16
   + 17 hold_0_17 + 18 hold_0_18 + 60 pair_ok_0_2_for_nic_out_0_nic_in_2
   <= 60
 constr_pair_ok2_0_2: - start_0 + start_2 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11 - 12 hold_0_12
   - 13 hold_0_13 - 14 hold_0_14 - 15 hold_0_15 - 16 hold_0_16
   - 17 hold_0_17 - 18 hold_0_18 + 60 pair_ok_0_2_for_nic_out_0_nic_in_2
   <= 60
 constr_edge_switch_end_start_0_2_OR: switch_nic_out_0_nic_in_2
   - pair_ok_0_2_for_nic_out_0_nic_in_2 <= 0
 constr_pair_ok1_0_3: start_0 - start_3 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 8 hold_0_8 + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12
   + 13 hold_0_13 + 14 hold_0_14 + 15 hold_0_15 + 16 hold_0_16
   + 17 hold_0_17 + 18 hold_0_18 + 60 pair_ok_0_3_for_nic_out_0_nic_in_3
   <= 60
 constr_pair_ok2_0_3: - start_0 + start_3 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11 - 12 hold_0_12
   - 13 hold_0_13 - 14 hold_0_14 - 15 hold_0_15 - 16 hold_0_16
   - 17 hold_0_17 - 18 hold_0_18 + 60 pair_ok_0_3_for_nic_out_0_nic_in_3
   <= 60
 constr_edge_switch_end_start_0_3_OR: switch_nic_out_0_nic_in_3
   - pair_ok_0_3_for_nic_out_0_nic_in_3 <= 0
 constr_pair_ok1_0_4: start_0 - start_4 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 8 hold_0_8 + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12
   + 13 hold_0_13 + 14 hold_0_14 + 15 hold_0_15 + 16 hold_0_16
   + 17 hold_0_17 + 18 hold_0_18 + 60 pair_ok_0_4_for_nic_out_0_nic_in_4
   <= 60
 constr_pair_ok2_0_4: - start_0 + start_4 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11 - 12 hold_0_12
   - 13 hold_0_13 - 14 hold_0_14 - 15 hold_0_15 - 16 hold_0_16
   - 17 hold_0_17 - 18 hold_0_18 + 60 pair_ok_0_4_for_nic_out_0_nic_in_4
   <= 60
 constr_edge_switch_end_start_0_4_OR: switch_nic_out_0_nic_in_4
   - pair_ok_0_4_for_nic_out_0_nic_in_4 <= 0
 constr_pair_ok1_0_5: start_0 - start_5 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 8 hold_0_8 + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12
   + 13 hold_0_13 + 14 hold_0_14 + 15 hold_0_15 + 16 hold_0_16
   + 17 hold_0_17 + 18 hold_0_18 + 60 pair_ok_0_5_for_nic_out_0_nic_in_5
   <= 60
 constr_pair_ok2_0_5: - start_0 + start_5 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11 - 12 hold_0_12
   - 13 hold_0_13 - 14 hold_0_14 - 15 hold_0_15 - 16 hold_0_16
   - 17 hold_0_17 - 18 hold_0_18 + 60 pair_ok_0_5_for_nic_out_0_nic_in_5
   <= 60
 constr_edge_switch_end_start_0_5_OR: switch_nic_out_0_nic_in_5
   - pair_ok_0_5_for_nic_out_0_nic_in_5 <= 0
 constr_pair_ok1_0_6: start_0 - start_6 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 8 hold_0_8 + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12
   + 13 hold_0_13 + 14 hold_0_14 + 15 hold_0_15 + 16 hold_0_16
   + 17 hold_0_17 + 18 hold_0_18 + 60 pair_ok_0_6_for_nic_out_0_nic_in_6
   <= 60
 constr_pair_ok2_0_6: - start_0 + start_6 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11 - 12 hold_0_12
   - 13 hold_0_13 - 14 hold_0_14 - 15 hold_0_15 - 16 hold_0_16
   - 17 hold_0_17 - 18 hold_0_18 + 60 pair_ok_0_6_for_nic_out_0_nic_in_6
   <= 60
 constr_edge_switch_end_start_0_6_OR: switch_nic_out_0_nic_in_6
   - pair_ok_0_6_for_nic_out_0_nic_in_6 <= 0
 constr_pair_ok1_0_7: start_0 - start_7 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 8 hold_0_8 + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12
   + 13 hold_0_13 + 14 hold_0_14 + 15 hold_0_15 + 16 hold_0_16
   + 17 hold_0_17 + 18 hold_0_18 + 60 pair_ok_0_7_for_nic_out_0_nic_in_7
   <= 60
 constr_pair_ok2_0_7: - start_0 + start_7 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11 - 12 hold_0_12
   - 13 hold_0_13 - 14 hold_0_14 - 15 hold_0_15 - 16 hold_0_16
   - 17 hold_0_17 - 18 hold_0_18 + 60 pair_ok_0_7_for_nic_out_0_nic_in_7
   <= 60
 constr_edge_switch_end_start_0_7_OR: switch_nic_out_0_nic_in_7
   - pair_ok_0_7_for_nic_out_0_nic_in_7 <= 0
 constr_pair_ok1_0_8: start_0 - start_8 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 8 hold_0_8 + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12
   + 13 hold_0_13 + 14 hold_0_14 + 15 hold_0_15 + 16 hold_0_16
   + 17 hold_0_17 + 18 hold_0_18 + 60 pair_ok_0_8_for_nic_out_0_nic_in_8
   <= 60
 constr_pair_ok2_0_8: - start_0 + start_8 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11 - 12 hold_0_12
   - 13 hold_0_13 - 14 hold_0_14 - 15 hold_0_15 - 16 hold_0_16
   - 17 hold_0_17 - 18 hold_0_18 + 60 pair_ok_0_8_for_nic_out_0_nic_in_8
   <= 60
 constr_edge_switch_end_start_0_8_OR: switch_nic_out_0_nic_in_8
   - pair_ok_0_8_for_nic_out_0_nic_in_8 <= 0
 constr_pair_ok1_0_9: start_0 - start_9 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 8 hold_0_8 + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12
   + 13 hold_0_13 + 14 hold_0_14 + 15 hold_0_15 + 16 hold_0_16
   + 17 hold_0_17 + 18 hold_0_18 + 60 pair_ok_0_9_for_nic_out_0_nic_in_9
   <= 60
 constr_pair_ok2_0_9: - start_0 + start_9 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11 - 12 hold_0_12
   - 13 hold_0_13 - 14 hold_0_14 - 15 hold_0_15 - 16 hold_0_16
   - 17 hold_0_17 - 18 hold_0_18 + 60 pair_ok_0_9_for_nic_out_0_nic_in_9
   <= 60
 constr_edge_switch_end_start_0_9_OR: switch_nic_out_0_nic_in_9
   - pair_ok_0_9_for_nic_out_0_nic_in_9 <= 0
 constr_pair_ok1_0_10: start_0 - start_10 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 8 hold_0_8 + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12
   + 13 hold_0_13 + 14 hold_0_14 + 15 hold_0_15 + 16 hold_0_16
   + 17 hold_0_17 + 18 hold_0_18 + 60 pair_ok_0_10_for_nic_out_0_nic_in_10
   <= 60
 constr_pair_ok2_0_10: - start_0 + start_10 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11 - 12 hold_0_12
   - 13 hold_0_13 - 14 hold_0_14 - 15 hold_0_15 - 16 hold_0_16
   - 17 hold_0_17 - 18 hold_0_18 + 60 pair_ok_0_10_for_nic_out_0_nic_in_10
   <= 60
 constr_edge_switch_end_start_0_10_OR: switch_nic_out_0_nic_in_10
   - pair_ok_0_10_for_nic_out_0_nic_in_10 <= 0
 constr_pair_ok1_0_11: start_0 - start_11 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 8 hold_0_8 + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12
   + 13 hold_0_13 + 14 hold_0_14 + 15 hold_0_15 + 16 hold_0_16
   + 17 hold_0_17 + 18 hold_0_18 + 60 pair_ok_0_11_for_nic_out_0_nic_in_11
   <= 60
 constr_pair_ok2_0_11: - start_0 + start_11 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11 - 12 hold_0_12
   - 13 hold_0_13 - 14 hold_0_14 - 15 hold_0_15 - 16 hold_0_16
   - 17 hold_0_17 - 18 hold_0_18 + 60 pair_ok_0_11_for_nic_out_0_nic_in_11
   <= 60
 constr_edge_switch_end_start_0_11_OR: switch_nic_out_0_nic_in_11
   - pair_ok_0_11_for_nic_out_0_nic_in_11 <= 0
 constr_pair_ok1_0_12: start_0 - start_12 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 8 hold_0_8 + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12
   + 13 hold_0_13 + 14 hold_0_14 + 15 hold_0_15 + 16 hold_0_16
   + 17 hold_0_17 + 18 hold_0_18 + 60 pair_ok_0_12_for_nic_out_0_nic_in_12
   <= 60
 constr_pair_ok2_0_12: - start_0 + start_12 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11 - 12 hold_0_12
   - 13 hold_0_13 - 14 hold_0_14 - 15 hold_0_15 - 16 hold_0_16
   - 17 hold_0_17 - 18 hold_0_18 + 60 pair_ok_0_12_for_nic_out_0_nic_in_12
   <= 60
 constr_edge_switch_end_start_0_12_OR: switch_nic_out_0_nic_in_12
   - pair_ok_0_12_for_nic_out_0_nic_in_12 <= 0
 constr_pair_ok1_0_13: start_0 - start_13 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 8 hold_0_8 + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12
   + 13 hold_0_13 + 14 hold_0_14 + 15 hold_0_15 + 16 hold_0_16
   + 17 hold_0_17 + 18 hold_0_18 + 60 pair_ok_0_13_for_nic_out_0_nic_in_13
   <= 60
 constr_pair_ok2_0_13: - start_0 + start_13 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11 - 12 hold_0_12
   - 13 hold_0_13 - 14 hold_0_14 - 15 hold_0_15 - 16 hold_0_16
   - 17 hold_0_17 - 18 hold_0_18 + 60 pair_ok_0_13_for_nic_out_0_nic_in_13
   <= 60
 constr_edge_switch_end_start_0_13_OR: switch_nic_out_0_nic_in_13
   - pair_ok_0_13_for_nic_out_0_nic_in_13 <= 0
 constr_pair_ok1_0_14: start_0 - start_14 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 8 hold_0_8 + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12
   + 13 hold_0_13 + 14 hold_0_14 + 15 hold_0_15 + 16 hold_0_16
   + 17 hold_0_17 + 18 hold_0_18 + 60 pair_ok_0_14_for_nic_out_0_nic_in_14
   <= 60
 constr_pair_ok2_0_14: - start_0 + start_14 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11 - 12 hold_0_12
   - 13 hold_0_13 - 14 hold_0_14 - 15 hold_0_15 - 16 hold_0_16
   - 17 hold_0_17 - 18 hold_0_18 + 60 pair_ok_0_14_for_nic_out_0_nic_in_14
   <= 60
 constr_edge_switch_end_start_0_14_OR: switch_nic_out_0_nic_in_14
   - pair_ok_0_14_for_nic_out_0_nic_in_14 <= 0
 constr_pair_ok1_0_15: start_0 - start_15 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 8 hold_0_8 + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12
   + 13 hold_0_13 + 14 hold_0_14 + 15 hold_0_15 + 16 hold_0_16
   + 17 hold_0_17 + 18 hold_0_18 + 60 pair_ok_0_15_for_nic_out_0_nic_in_15
   <= 60
 constr_pair_ok2_0_15: - start_0 + start_15 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11 - 12 hold_0_12
   - 13 hold_0_13 - 14 hold_0_14 - 15 hold_0_15 - 16 hold_0_16
   - 17 hold_0_17 - 18 hold_0_18 + 60 pair_ok_0_15_for_nic_out_0_nic_in_15
   <= 60
 constr_edge_switch_end_start_0_15_OR: switch_nic_out_0_nic_in_15
   - pair_ok_0_15_for_nic_out_0_nic_in_15 <= 0
 constr_is_start_zero_1: start_1 + 60 is_start_zero_1_for_source_nic_in_1
   <= 60
 constr_edge_switch_start_zero_1_OR: switch_source_nic_in_1
   - is_start_zero_1_for_source_nic_in_1 <= 0
 constr_is_end_complete_1: start_1 + hold_1_1 + 2 hold_1_2 + 3 hold_1_3
   + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7 + 8 hold_1_8
   + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 + 12 hold_1_12 + 13 hold_1_13
   + 14 hold_1_14 + 15 hold_1_15 + 16 hold_1_16 + 17 hold_1_17
   + 18 hold_1_18 - 60 is_end_complete_1 >= 0
 constr_edge_switch_end_max_1_OR: switch_nic_out_1_sink - is_end_complete_1
   <= 0
 constr_pair_ok1_1_0: - start_0 + start_1 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 8 hold_1_8 + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 + 12 hold_1_12
   + 13 hold_1_13 + 14 hold_1_14 + 15 hold_1_15 + 16 hold_1_16
   + 17 hold_1_17 + 18 hold_1_18 + 60 pair_ok_1_0_for_nic_out_1_nic_in_0
   <= 60
 constr_pair_ok2_1_0: start_0 - start_1 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11 - 12 hold_1_12
   - 13 hold_1_13 - 14 hold_1_14 - 15 hold_1_15 - 16 hold_1_16
   - 17 hold_1_17 - 18 hold_1_18 + 60 pair_ok_1_0_for_nic_out_1_nic_in_0
   <= 60
 constr_edge_switch_end_start_1_0_OR: switch_nic_out_1_nic_in_0
   - pair_ok_1_0_for_nic_out_1_nic_in_0 <= 0
 constr_pair_ok1_1_2: start_1 - start_2 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 8 hold_1_8 + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 + 12 hold_1_12
   + 13 hold_1_13 + 14 hold_1_14 + 15 hold_1_15 + 16 hold_1_16
   + 17 hold_1_17 + 18 hold_1_18 + 60 pair_ok_1_2_for_nic_out_1_nic_in_2
   <= 60
 constr_pair_ok2_1_2: - start_1 + start_2 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11 - 12 hold_1_12
   - 13 hold_1_13 - 14 hold_1_14 - 15 hold_1_15 - 16 hold_1_16
   - 17 hold_1_17 - 18 hold_1_18 + 60 pair_ok_1_2_for_nic_out_1_nic_in_2
   <= 60
 constr_edge_switch_end_start_1_2_OR: switch_nic_out_1_nic_in_2
   - pair_ok_1_2_for_nic_out_1_nic_in_2 <= 0
 constr_pair_ok1_1_3: start_1 - start_3 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 8 hold_1_8 + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 + 12 hold_1_12
   + 13 hold_1_13 + 14 hold_1_14 + 15 hold_1_15 + 16 hold_1_16
   + 17 hold_1_17 + 18 hold_1_18 + 60 pair_ok_1_3_for_nic_out_1_nic_in_3
   <= 60
 constr_pair_ok2_1_3: - start_1 + start_3 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11 - 12 hold_1_12
   - 13 hold_1_13 - 14 hold_1_14 - 15 hold_1_15 - 16 hold_1_16
   - 17 hold_1_17 - 18 hold_1_18 + 60 pair_ok_1_3_for_nic_out_1_nic_in_3
   <= 60
 constr_edge_switch_end_start_1_3_OR: switch_nic_out_1_nic_in_3
   - pair_ok_1_3_for_nic_out_1_nic_in_3 <= 0
 constr_pair_ok1_1_4: start_1 - start_4 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 8 hold_1_8 + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 + 12 hold_1_12
   + 13 hold_1_13 + 14 hold_1_14 + 15 hold_1_15 + 16 hold_1_16
   + 17 hold_1_17 + 18 hold_1_18 + 60 pair_ok_1_4_for_nic_out_1_nic_in_4
   <= 60
 constr_pair_ok2_1_4: - start_1 + start_4 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11 - 12 hold_1_12
   - 13 hold_1_13 - 14 hold_1_14 - 15 hold_1_15 - 16 hold_1_16
   - 17 hold_1_17 - 18 hold_1_18 + 60 pair_ok_1_4_for_nic_out_1_nic_in_4
   <= 60
 constr_edge_switch_end_start_1_4_OR: switch_nic_out_1_nic_in_4
   - pair_ok_1_4_for_nic_out_1_nic_in_4 <= 0
 constr_pair_ok1_1_5: start_1 - start_5 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 8 hold_1_8 + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 + 12 hold_1_12
   + 13 hold_1_13 + 14 hold_1_14 + 15 hold_1_15 + 16 hold_1_16
   + 17 hold_1_17 + 18 hold_1_18 + 60 pair_ok_1_5_for_nic_out_1_nic_in_5
   <= 60
 constr_pair_ok2_1_5: - start_1 + start_5 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11 - 12 hold_1_12
   - 13 hold_1_13 - 14 hold_1_14 - 15 hold_1_15 - 16 hold_1_16
   - 17 hold_1_17 - 18 hold_1_18 + 60 pair_ok_1_5_for_nic_out_1_nic_in_5
   <= 60
 constr_edge_switch_end_start_1_5_OR: switch_nic_out_1_nic_in_5
   - pair_ok_1_5_for_nic_out_1_nic_in_5 <= 0
 constr_pair_ok1_1_6: start_1 - start_6 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 8 hold_1_8 + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 + 12 hold_1_12
   + 13 hold_1_13 + 14 hold_1_14 + 15 hold_1_15 + 16 hold_1_16
   + 17 hold_1_17 + 18 hold_1_18 + 60 pair_ok_1_6_for_nic_out_1_nic_in_6
   <= 60
 constr_pair_ok2_1_6: - start_1 + start_6 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11 - 12 hold_1_12
   - 13 hold_1_13 - 14 hold_1_14 - 15 hold_1_15 - 16 hold_1_16
   - 17 hold_1_17 - 18 hold_1_18 + 60 pair_ok_1_6_for_nic_out_1_nic_in_6
   <= 60
 constr_edge_switch_end_start_1_6_OR: switch_nic_out_1_nic_in_6
   - pair_ok_1_6_for_nic_out_1_nic_in_6 <= 0
 constr_pair_ok1_1_7: start_1 - start_7 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 8 hold_1_8 + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 + 12 hold_1_12
   + 13 hold_1_13 + 14 hold_1_14 + 15 hold_1_15 + 16 hold_1_16
   + 17 hold_1_17 + 18 hold_1_18 + 60 pair_ok_1_7_for_nic_out_1_nic_in_7
   <= 60
 constr_pair_ok2_1_7: - start_1 + start_7 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11 - 12 hold_1_12
   - 13 hold_1_13 - 14 hold_1_14 - 15 hold_1_15 - 16 hold_1_16
   - 17 hold_1_17 - 18 hold_1_18 + 60 pair_ok_1_7_for_nic_out_1_nic_in_7
   <= 60
 constr_edge_switch_end_start_1_7_OR: switch_nic_out_1_nic_in_7
   - pair_ok_1_7_for_nic_out_1_nic_in_7 <= 0
 constr_pair_ok1_1_8: start_1 - start_8 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 8 hold_1_8 + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 + 12 hold_1_12
   + 13 hold_1_13 + 14 hold_1_14 + 15 hold_1_15 + 16 hold_1_16
   + 17 hold_1_17 + 18 hold_1_18 + 60 pair_ok_1_8_for_nic_out_1_nic_in_8
   <= 60
 constr_pair_ok2_1_8: - start_1 + start_8 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11 - 12 hold_1_12
   - 13 hold_1_13 - 14 hold_1_14 - 15 hold_1_15 - 16 hold_1_16
   - 17 hold_1_17 - 18 hold_1_18 + 60 pair_ok_1_8_for_nic_out_1_nic_in_8
   <= 60
 constr_edge_switch_end_start_1_8_OR: switch_nic_out_1_nic_in_8
   - pair_ok_1_8_for_nic_out_1_nic_in_8 <= 0
 constr_pair_ok1_1_9: start_1 - start_9 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 8 hold_1_8 + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 + 12 hold_1_12
   + 13 hold_1_13 + 14 hold_1_14 + 15 hold_1_15 + 16 hold_1_16
   + 17 hold_1_17 + 18 hold_1_18 + 60 pair_ok_1_9_for_nic_out_1_nic_in_9
   <= 60
 constr_pair_ok2_1_9: - start_1 + start_9 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11 - 12 hold_1_12
   - 13 hold_1_13 - 14 hold_1_14 - 15 hold_1_15 - 16 hold_1_16
   - 17 hold_1_17 - 18 hold_1_18 + 60 pair_ok_1_9_for_nic_out_1_nic_in_9
   <= 60
 constr_edge_switch_end_start_1_9_OR: switch_nic_out_1_nic_in_9
   - pair_ok_1_9_for_nic_out_1_nic_in_9 <= 0
 constr_pair_ok1_1_10: start_1 - start_10 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 8 hold_1_8 + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 + 12 hold_1_12
   + 13 hold_1_13 + 14 hold_1_14 + 15 hold_1_15 + 16 hold_1_16
   + 17 hold_1_17 + 18 hold_1_18 + 60 pair_ok_1_10_for_nic_out_1_nic_in_10
   <= 60
 constr_pair_ok2_1_10: - start_1 + start_10 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11 - 12 hold_1_12
   - 13 hold_1_13 - 14 hold_1_14 - 15 hold_1_15 - 16 hold_1_16
   - 17 hold_1_17 - 18 hold_1_18 + 60 pair_ok_1_10_for_nic_out_1_nic_in_10
   <= 60
 constr_edge_switch_end_start_1_10_OR: switch_nic_out_1_nic_in_10
   - pair_ok_1_10_for_nic_out_1_nic_in_10 <= 0
 constr_pair_ok1_1_11: start_1 - start_11 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 8 hold_1_8 + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 + 12 hold_1_12
   + 13 hold_1_13 + 14 hold_1_14 + 15 hold_1_15 + 16 hold_1_16
   + 17 hold_1_17 + 18 hold_1_18 + 60 pair_ok_1_11_for_nic_out_1_nic_in_11
   <= 60
 constr_pair_ok2_1_11: - start_1 + start_11 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11 - 12 hold_1_12
   - 13 hold_1_13 - 14 hold_1_14 - 15 hold_1_15 - 16 hold_1_16
   - 17 hold_1_17 - 18 hold_1_18 + 60 pair_ok_1_11_for_nic_out_1_nic_in_11
   <= 60
 constr_edge_switch_end_start_1_11_OR: switch_nic_out_1_nic_in_11
   - pair_ok_1_11_for_nic_out_1_nic_in_11 <= 0
 constr_pair_ok1_1_12: start_1 - start_12 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 8 hold_1_8 + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 + 12 hold_1_12
   + 13 hold_1_13 + 14 hold_1_14 + 15 hold_1_15 + 16 hold_1_16
   + 17 hold_1_17 + 18 hold_1_18 + 60 pair_ok_1_12_for_nic_out_1_nic_in_12
   <= 60
 constr_pair_ok2_1_12: - start_1 + start_12 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11 - 12 hold_1_12
   - 13 hold_1_13 - 14 hold_1_14 - 15 hold_1_15 - 16 hold_1_16
   - 17 hold_1_17 - 18 hold_1_18 + 60 pair_ok_1_12_for_nic_out_1_nic_in_12
   <= 60
 constr_edge_switch_end_start_1_12_OR: switch_nic_out_1_nic_in_12
   - pair_ok_1_12_for_nic_out_1_nic_in_12 <= 0
 constr_pair_ok1_1_13: start_1 - start_13 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 8 hold_1_8 + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 + 12 hold_1_12
   + 13 hold_1_13 + 14 hold_1_14 + 15 hold_1_15 + 16 hold_1_16
   + 17 hold_1_17 + 18 hold_1_18 + 60 pair_ok_1_13_for_nic_out_1_nic_in_13
   <= 60
 constr_pair_ok2_1_13: - start_1 + start_13 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11 - 12 hold_1_12
   - 13 hold_1_13 - 14 hold_1_14 - 15 hold_1_15 - 16 hold_1_16
   - 17 hold_1_17 - 18 hold_1_18 + 60 pair_ok_1_13_for_nic_out_1_nic_in_13
   <= 60
 constr_edge_switch_end_start_1_13_OR: switch_nic_out_1_nic_in_13
   - pair_ok_1_13_for_nic_out_1_nic_in_13 <= 0
 constr_pair_ok1_1_14: start_1 - start_14 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 8 hold_1_8 + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 + 12 hold_1_12
   + 13 hold_1_13 + 14 hold_1_14 + 15 hold_1_15 + 16 hold_1_16
   + 17 hold_1_17 + 18 hold_1_18 + 60 pair_ok_1_14_for_nic_out_1_nic_in_14
   <= 60
 constr_pair_ok2_1_14: - start_1 + start_14 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11 - 12 hold_1_12
   - 13 hold_1_13 - 14 hold_1_14 - 15 hold_1_15 - 16 hold_1_16
   - 17 hold_1_17 - 18 hold_1_18 + 60 pair_ok_1_14_for_nic_out_1_nic_in_14
   <= 60
 constr_edge_switch_end_start_1_14_OR: switch_nic_out_1_nic_in_14
   - pair_ok_1_14_for_nic_out_1_nic_in_14 <= 0
 constr_pair_ok1_1_15: start_1 - start_15 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 8 hold_1_8 + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 + 12 hold_1_12
   + 13 hold_1_13 + 14 hold_1_14 + 15 hold_1_15 + 16 hold_1_16
   + 17 hold_1_17 + 18 hold_1_18 + 60 pair_ok_1_15_for_nic_out_1_nic_in_15
   <= 60
 constr_pair_ok2_1_15: - start_1 + start_15 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11 - 12 hold_1_12
   - 13 hold_1_13 - 14 hold_1_14 - 15 hold_1_15 - 16 hold_1_16
   - 17 hold_1_17 - 18 hold_1_18 + 60 pair_ok_1_15_for_nic_out_1_nic_in_15
   <= 60
 constr_edge_switch_end_start_1_15_OR: switch_nic_out_1_nic_in_15
   - pair_ok_1_15_for_nic_out_1_nic_in_15 <= 0
 constr_is_start_zero_2: start_2 + 60 is_start_zero_2_for_source_nic_in_2
   <= 60
 constr_edge_switch_start_zero_2_OR: switch_source_nic_in_2
   - is_start_zero_2_for_source_nic_in_2 <= 0
 constr_is_end_complete_2: start_2 + hold_2_1 + 2 hold_2_2 + 3 hold_2_3
   + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7 + 8 hold_2_8
   + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 12 hold_2_12 + 13 hold_2_13
   + 14 hold_2_14 + 15 hold_2_15 + 16 hold_2_16 + 17 hold_2_17
   + 18 hold_2_18 - 60 is_end_complete_2 >= 0
 constr_edge_switch_end_max_2_OR: switch_nic_out_2_sink - is_end_complete_2
   <= 0
 constr_pair_ok1_2_0: - start_0 + start_2 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 12 hold_2_12
   + 13 hold_2_13 + 14 hold_2_14 + 15 hold_2_15 + 16 hold_2_16
   + 17 hold_2_17 + 18 hold_2_18 + 60 pair_ok_2_0_for_nic_out_2_nic_in_0
   <= 60
 constr_pair_ok2_2_0: start_0 - start_2 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 - 12 hold_2_12
   - 13 hold_2_13 - 14 hold_2_14 - 15 hold_2_15 - 16 hold_2_16
   - 17 hold_2_17 - 18 hold_2_18 + 60 pair_ok_2_0_for_nic_out_2_nic_in_0
   <= 60
 constr_edge_switch_end_start_2_0_OR: switch_nic_out_2_nic_in_0
   - pair_ok_2_0_for_nic_out_2_nic_in_0 <= 0
 constr_pair_ok1_2_1: - start_1 + start_2 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 12 hold_2_12
   + 13 hold_2_13 + 14 hold_2_14 + 15 hold_2_15 + 16 hold_2_16
   + 17 hold_2_17 + 18 hold_2_18 + 60 pair_ok_2_1_for_nic_out_2_nic_in_1
   <= 60
 constr_pair_ok2_2_1: start_1 - start_2 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 - 12 hold_2_12
   - 13 hold_2_13 - 14 hold_2_14 - 15 hold_2_15 - 16 hold_2_16
   - 17 hold_2_17 - 18 hold_2_18 + 60 pair_ok_2_1_for_nic_out_2_nic_in_1
   <= 60
 constr_edge_switch_end_start_2_1_OR: switch_nic_out_2_nic_in_1
   - pair_ok_2_1_for_nic_out_2_nic_in_1 <= 0
 constr_pair_ok1_2_3: start_2 - start_3 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 12 hold_2_12
   + 13 hold_2_13 + 14 hold_2_14 + 15 hold_2_15 + 16 hold_2_16
   + 17 hold_2_17 + 18 hold_2_18 + 60 pair_ok_2_3_for_nic_out_2_nic_in_3
   <= 60
 constr_pair_ok2_2_3: - start_2 + start_3 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 - 12 hold_2_12
   - 13 hold_2_13 - 14 hold_2_14 - 15 hold_2_15 - 16 hold_2_16
   - 17 hold_2_17 - 18 hold_2_18 + 60 pair_ok_2_3_for_nic_out_2_nic_in_3
   <= 60
 constr_edge_switch_end_start_2_3_OR: switch_nic_out_2_nic_in_3
   - pair_ok_2_3_for_nic_out_2_nic_in_3 <= 0
 constr_pair_ok1_2_4: start_2 - start_4 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 12 hold_2_12
   + 13 hold_2_13 + 14 hold_2_14 + 15 hold_2_15 + 16 hold_2_16
   + 17 hold_2_17 + 18 hold_2_18 + 60 pair_ok_2_4_for_nic_out_2_nic_in_4
   <= 60
 constr_pair_ok2_2_4: - start_2 + start_4 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 - 12 hold_2_12
   - 13 hold_2_13 - 14 hold_2_14 - 15 hold_2_15 - 16 hold_2_16
   - 17 hold_2_17 - 18 hold_2_18 + 60 pair_ok_2_4_for_nic_out_2_nic_in_4
   <= 60
 constr_edge_switch_end_start_2_4_OR: switch_nic_out_2_nic_in_4
   - pair_ok_2_4_for_nic_out_2_nic_in_4 <= 0
 constr_pair_ok1_2_5: start_2 - start_5 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 12 hold_2_12
   + 13 hold_2_13 + 14 hold_2_14 + 15 hold_2_15 + 16 hold_2_16
   + 17 hold_2_17 + 18 hold_2_18 + 60 pair_ok_2_5_for_nic_out_2_nic_in_5
   <= 60
 constr_pair_ok2_2_5: - start_2 + start_5 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 - 12 hold_2_12
   - 13 hold_2_13 - 14 hold_2_14 - 15 hold_2_15 - 16 hold_2_16
   - 17 hold_2_17 - 18 hold_2_18 + 60 pair_ok_2_5_for_nic_out_2_nic_in_5
   <= 60
 constr_edge_switch_end_start_2_5_OR: switch_nic_out_2_nic_in_5
   - pair_ok_2_5_for_nic_out_2_nic_in_5 <= 0
 constr_pair_ok1_2_6: start_2 - start_6 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 12 hold_2_12
   + 13 hold_2_13 + 14 hold_2_14 + 15 hold_2_15 + 16 hold_2_16
   + 17 hold_2_17 + 18 hold_2_18 + 60 pair_ok_2_6_for_nic_out_2_nic_in_6
   <= 60
 constr_pair_ok2_2_6: - start_2 + start_6 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 - 12 hold_2_12
   - 13 hold_2_13 - 14 hold_2_14 - 15 hold_2_15 - 16 hold_2_16
   - 17 hold_2_17 - 18 hold_2_18 + 60 pair_ok_2_6_for_nic_out_2_nic_in_6
   <= 60
 constr_edge_switch_end_start_2_6_OR: switch_nic_out_2_nic_in_6
   - pair_ok_2_6_for_nic_out_2_nic_in_6 <= 0
 constr_pair_ok1_2_7: start_2 - start_7 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 12 hold_2_12
   + 13 hold_2_13 + 14 hold_2_14 + 15 hold_2_15 + 16 hold_2_16
   + 17 hold_2_17 + 18 hold_2_18 + 60 pair_ok_2_7_for_nic_out_2_nic_in_7
   <= 60
 constr_pair_ok2_2_7: - start_2 + start_7 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 - 12 hold_2_12
   - 13 hold_2_13 - 14 hold_2_14 - 15 hold_2_15 - 16 hold_2_16
   - 17 hold_2_17 - 18 hold_2_18 + 60 pair_ok_2_7_for_nic_out_2_nic_in_7
   <= 60
 constr_edge_switch_end_start_2_7_OR: switch_nic_out_2_nic_in_7
   - pair_ok_2_7_for_nic_out_2_nic_in_7 <= 0
 constr_pair_ok1_2_8: start_2 - start_8 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 12 hold_2_12
   + 13 hold_2_13 + 14 hold_2_14 + 15 hold_2_15 + 16 hold_2_16
   + 17 hold_2_17 + 18 hold_2_18 + 60 pair_ok_2_8_for_nic_out_2_nic_in_8
   <= 60
 constr_pair_ok2_2_8: - start_2 + start_8 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 - 12 hold_2_12
   - 13 hold_2_13 - 14 hold_2_14 - 15 hold_2_15 - 16 hold_2_16
   - 17 hold_2_17 - 18 hold_2_18 + 60 pair_ok_2_8_for_nic_out_2_nic_in_8
   <= 60
 constr_edge_switch_end_start_2_8_OR: switch_nic_out_2_nic_in_8
   - pair_ok_2_8_for_nic_out_2_nic_in_8 <= 0
 constr_pair_ok1_2_9: start_2 - start_9 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 12 hold_2_12
   + 13 hold_2_13 + 14 hold_2_14 + 15 hold_2_15 + 16 hold_2_16
   + 17 hold_2_17 + 18 hold_2_18 + 60 pair_ok_2_9_for_nic_out_2_nic_in_9
   <= 60
 constr_pair_ok2_2_9: - start_2 + start_9 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 - 12 hold_2_12
   - 13 hold_2_13 - 14 hold_2_14 - 15 hold_2_15 - 16 hold_2_16
   - 17 hold_2_17 - 18 hold_2_18 + 60 pair_ok_2_9_for_nic_out_2_nic_in_9
   <= 60
 constr_edge_switch_end_start_2_9_OR: switch_nic_out_2_nic_in_9
   - pair_ok_2_9_for_nic_out_2_nic_in_9 <= 0
 constr_pair_ok1_2_10: start_2 - start_10 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 12 hold_2_12
   + 13 hold_2_13 + 14 hold_2_14 + 15 hold_2_15 + 16 hold_2_16
   + 17 hold_2_17 + 18 hold_2_18 + 60 pair_ok_2_10_for_nic_out_2_nic_in_10
   <= 60
 constr_pair_ok2_2_10: - start_2 + start_10 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 - 12 hold_2_12
   - 13 hold_2_13 - 14 hold_2_14 - 15 hold_2_15 - 16 hold_2_16
   - 17 hold_2_17 - 18 hold_2_18 + 60 pair_ok_2_10_for_nic_out_2_nic_in_10
   <= 60
 constr_edge_switch_end_start_2_10_OR: switch_nic_out_2_nic_in_10
   - pair_ok_2_10_for_nic_out_2_nic_in_10 <= 0
 constr_pair_ok1_2_11: start_2 - start_11 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 12 hold_2_12
   + 13 hold_2_13 + 14 hold_2_14 + 15 hold_2_15 + 16 hold_2_16
   + 17 hold_2_17 + 18 hold_2_18 + 60 pair_ok_2_11_for_nic_out_2_nic_in_11
   <= 60
 constr_pair_ok2_2_11: - start_2 + start_11 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 - 12 hold_2_12
   - 13 hold_2_13 - 14 hold_2_14 - 15 hold_2_15 - 16 hold_2_16
   - 17 hold_2_17 - 18 hold_2_18 + 60 pair_ok_2_11_for_nic_out_2_nic_in_11
   <= 60
 constr_edge_switch_end_start_2_11_OR: switch_nic_out_2_nic_in_11
   - pair_ok_2_11_for_nic_out_2_nic_in_11 <= 0
 constr_pair_ok1_2_12: start_2 - start_12 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 12 hold_2_12
   + 13 hold_2_13 + 14 hold_2_14 + 15 hold_2_15 + 16 hold_2_16
   + 17 hold_2_17 + 18 hold_2_18 + 60 pair_ok_2_12_for_nic_out_2_nic_in_12
   <= 60
 constr_pair_ok2_2_12: - start_2 + start_12 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 - 12 hold_2_12
   - 13 hold_2_13 - 14 hold_2_14 - 15 hold_2_15 - 16 hold_2_16
   - 17 hold_2_17 - 18 hold_2_18 + 60 pair_ok_2_12_for_nic_out_2_nic_in_12
   <= 60
 constr_edge_switch_end_start_2_12_OR: switch_nic_out_2_nic_in_12
   - pair_ok_2_12_for_nic_out_2_nic_in_12 <= 0
 constr_pair_ok1_2_13: start_2 - start_13 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 12 hold_2_12
   + 13 hold_2_13 + 14 hold_2_14 + 15 hold_2_15 + 16 hold_2_16
   + 17 hold_2_17 + 18 hold_2_18 + 60 pair_ok_2_13_for_nic_out_2_nic_in_13
   <= 60
 constr_pair_ok2_2_13: - start_2 + start_13 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 - 12 hold_2_12
   - 13 hold_2_13 - 14 hold_2_14 - 15 hold_2_15 - 16 hold_2_16
   - 17 hold_2_17 - 18 hold_2_18 + 60 pair_ok_2_13_for_nic_out_2_nic_in_13
   <= 60
 constr_edge_switch_end_start_2_13_OR: switch_nic_out_2_nic_in_13
   - pair_ok_2_13_for_nic_out_2_nic_in_13 <= 0
 constr_pair_ok1_2_14: start_2 - start_14 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 12 hold_2_12
   + 13 hold_2_13 + 14 hold_2_14 + 15 hold_2_15 + 16 hold_2_16
   + 17 hold_2_17 + 18 hold_2_18 + 60 pair_ok_2_14_for_nic_out_2_nic_in_14
   <= 60
 constr_pair_ok2_2_14: - start_2 + start_14 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 - 12 hold_2_12
   - 13 hold_2_13 - 14 hold_2_14 - 15 hold_2_15 - 16 hold_2_16
   - 17 hold_2_17 - 18 hold_2_18 + 60 pair_ok_2_14_for_nic_out_2_nic_in_14
   <= 60
 constr_edge_switch_end_start_2_14_OR: switch_nic_out_2_nic_in_14
   - pair_ok_2_14_for_nic_out_2_nic_in_14 <= 0
 constr_pair_ok1_2_15: start_2 - start_15 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 12 hold_2_12
   + 13 hold_2_13 + 14 hold_2_14 + 15 hold_2_15 + 16 hold_2_16
   + 17 hold_2_17 + 18 hold_2_18 + 60 pair_ok_2_15_for_nic_out_2_nic_in_15
   <= 60
 constr_pair_ok2_2_15: - start_2 + start_15 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 - 12 hold_2_12
   - 13 hold_2_13 - 14 hold_2_14 - 15 hold_2_15 - 16 hold_2_16
   - 17 hold_2_17 - 18 hold_2_18 + 60 pair_ok_2_15_for_nic_out_2_nic_in_15
   <= 60
 constr_edge_switch_end_start_2_15_OR: switch_nic_out_2_nic_in_15
   - pair_ok_2_15_for_nic_out_2_nic_in_15 <= 0
 constr_is_start_zero_3: start_3 + 60 is_start_zero_3_for_source_nic_in_3
   <= 60
 constr_edge_switch_start_zero_3_OR: switch_source_nic_in_3
   - is_start_zero_3_for_source_nic_in_3 <= 0
 constr_is_end_complete_3: start_3 + hold_3_1 + 2 hold_3_2 + 3 hold_3_3
   + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7 + 8 hold_3_8
   + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11 + 12 hold_3_12 + 13 hold_3_13
   + 14 hold_3_14 + 15 hold_3_15 + 16 hold_3_16 + 17 hold_3_17
   + 18 hold_3_18 - 60 is_end_complete_3 >= 0
 constr_edge_switch_end_max_3_OR: switch_nic_out_3_sink - is_end_complete_3
   <= 0
 constr_pair_ok1_3_0: - start_0 + start_3 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 8 hold_3_8 + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11 + 12 hold_3_12
   + 13 hold_3_13 + 14 hold_3_14 + 15 hold_3_15 + 16 hold_3_16
   + 17 hold_3_17 + 18 hold_3_18 + 60 pair_ok_3_0_for_nic_out_3_nic_in_0
   <= 60
 constr_pair_ok2_3_0: start_0 - start_3 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   - 8 hold_3_8 - 9 hold_3_9 - 10 hold_3_10 - 11 hold_3_11 - 12 hold_3_12
   - 13 hold_3_13 - 14 hold_3_14 - 15 hold_3_15 - 16 hold_3_16
   - 17 hold_3_17 - 18 hold_3_18 + 60 pair_ok_3_0_for_nic_out_3_nic_in_0
   <= 60
 constr_edge_switch_end_start_3_0_OR: switch_nic_out_3_nic_in_0
   - pair_ok_3_0_for_nic_out_3_nic_in_0 <= 0
 constr_pair_ok1_3_1: - start_1 + start_3 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 8 hold_3_8 + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11 + 12 hold_3_12
   + 13 hold_3_13 + 14 hold_3_14 + 15 hold_3_15 + 16 hold_3_16
   + 17 hold_3_17 + 18 hold_3_18 + 60 pair_ok_3_1_for_nic_out_3_nic_in_1
   <= 60
 constr_pair_ok2_3_1: start_1 - start_3 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   - 8 hold_3_8 - 9 hold_3_9 - 10 hold_3_10 - 11 hold_3_11 - 12 hold_3_12
   - 13 hold_3_13 - 14 hold_3_14 - 15 hold_3_15 - 16 hold_3_16
   - 17 hold_3_17 - 18 hold_3_18 + 60 pair_ok_3_1_for_nic_out_3_nic_in_1
   <= 60
 constr_edge_switch_end_start_3_1_OR: switch_nic_out_3_nic_in_1
   - pair_ok_3_1_for_nic_out_3_nic_in_1 <= 0
 constr_pair_ok1_3_2: - start_2 + start_3 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 8 hold_3_8 + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11 + 12 hold_3_12
   + 13 hold_3_13 + 14 hold_3_14 + 15 hold_3_15 + 16 hold_3_16
   + 17 hold_3_17 + 18 hold_3_18 + 60 pair_ok_3_2_for_nic_out_3_nic_in_2
   <= 60
 constr_pair_ok2_3_2: start_2 - start_3 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   - 8 hold_3_8 - 9 hold_3_9 - 10 hold_3_10 - 11 hold_3_11 - 12 hold_3_12
   - 13 hold_3_13 - 14 hold_3_14 - 15 hold_3_15 - 16 hold_3_16
   - 17 hold_3_17 - 18 hold_3_18 + 60 pair_ok_3_2_for_nic_out_3_nic_in_2
   <= 60
 constr_edge_switch_end_start_3_2_OR: switch_nic_out_3_nic_in_2
   - pair_ok_3_2_for_nic_out_3_nic_in_2 <= 0
 constr_pair_ok1_3_4: start_3 - start_4 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 8 hold_3_8 + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11 + 12 hold_3_12
   + 13 hold_3_13 + 14 hold_3_14 + 15 hold_3_15 + 16 hold_3_16
   + 17 hold_3_17 + 18 hold_3_18 + 60 pair_ok_3_4_for_nic_out_3_nic_in_4
   <= 60
 constr_pair_ok2_3_4: - start_3 + start_4 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   - 8 hold_3_8 - 9 hold_3_9 - 10 hold_3_10 - 11 hold_3_11 - 12 hold_3_12
   - 13 hold_3_13 - 14 hold_3_14 - 15 hold_3_15 - 16 hold_3_16
   - 17 hold_3_17 - 18 hold_3_18 + 60 pair_ok_3_4_for_nic_out_3_nic_in_4
   <= 60
 constr_edge_switch_end_start_3_4_OR: switch_nic_out_3_nic_in_4
   - pair_ok_3_4_for_nic_out_3_nic_in_4 <= 0
 constr_pair_ok1_3_5: start_3 - start_5 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 8 hold_3_8 + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11 + 12 hold_3_12
   + 13 hold_3_13 + 14 hold_3_14 + 15 hold_3_15 + 16 hold_3_16
   + 17 hold_3_17 + 18 hold_3_18 + 60 pair_ok_3_5_for_nic_out_3_nic_in_5
   <= 60
 constr_pair_ok2_3_5: - start_3 + start_5 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   - 8 hold_3_8 - 9 hold_3_9 - 10 hold_3_10 - 11 hold_3_11 - 12 hold_3_12
   - 13 hold_3_13 - 14 hold_3_14 - 15 hold_3_15 - 16 hold_3_16
   - 17 hold_3_17 - 18 hold_3_18 + 60 pair_ok_3_5_for_nic_out_3_nic_in_5
   <= 60
 constr_edge_switch_end_start_3_5_OR: switch_nic_out_3_nic_in_5
   - pair_ok_3_5_for_nic_out_3_nic_in_5 <= 0
 constr_pair_ok1_3_6: start_3 - start_6 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 8 hold_3_8 + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11 + 12 hold_3_12
   + 13 hold_3_13 + 14 hold_3_14 + 15 hold_3_15 + 16 hold_3_16
   + 17 hold_3_17 + 18 hold_3_18 + 60 pair_ok_3_6_for_nic_out_3_nic_in_6
   <= 60
 constr_pair_ok2_3_6: - start_3 + start_6 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   - 8 hold_3_8 - 9 hold_3_9 - 10 hold_3_10 - 11 hold_3_11 - 12 hold_3_12
   - 13 hold_3_13 - 14 hold_3_14 - 15 hold_3_15 - 16 hold_3_16
   - 17 hold_3_17 - 18 hold_3_18 + 60 pair_ok_3_6_for_nic_out_3_nic_in_6
   <= 60
 constr_edge_switch_end_start_3_6_OR: switch_nic_out_3_nic_in_6
   - pair_ok_3_6_for_nic_out_3_nic_in_6 <= 0
 constr_pair_ok1_3_7: start_3 - start_7 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 8 hold_3_8 + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11 + 12 hold_3_12
   + 13 hold_3_13 + 14 hold_3_14 + 15 hold_3_15 + 16 hold_3_16
   + 17 hold_3_17 + 18 hold_3_18 + 60 pair_ok_3_7_for_nic_out_3_nic_in_7
   <= 60
 constr_pair_ok2_3_7: - start_3 + start_7 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   - 8 hold_3_8 - 9 hold_3_9 - 10 hold_3_10 - 11 hold_3_11 - 12 hold_3_12
   - 13 hold_3_13 - 14 hold_3_14 - 15 hold_3_15 - 16 hold_3_16
   - 17 hold_3_17 - 18 hold_3_18 + 60 pair_ok_3_7_for_nic_out_3_nic_in_7
   <= 60
 constr_edge_switch_end_start_3_7_OR: switch_nic_out_3_nic_in_7
   - pair_ok_3_7_for_nic_out_3_nic_in_7 <= 0
 constr_pair_ok1_3_8: start_3 - start_8 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 8 hold_3_8 + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11 + 12 hold_3_12
   + 13 hold_3_13 + 14 hold_3_14 + 15 hold_3_15 + 16 hold_3_16
   + 17 hold_3_17 + 18 hold_3_18 + 60 pair_ok_3_8_for_nic_out_3_nic_in_8
   <= 60
 constr_pair_ok2_3_8: - start_3 + start_8 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   - 8 hold_3_8 - 9 hold_3_9 - 10 hold_3_10 - 11 hold_3_11 - 12 hold_3_12
   - 13 hold_3_13 - 14 hold_3_14 - 15 hold_3_15 - 16 hold_3_16
   - 17 hold_3_17 - 18 hold_3_18 + 60 pair_ok_3_8_for_nic_out_3_nic_in_8
   <= 60
 constr_edge_switch_end_start_3_8_OR: switch_nic_out_3_nic_in_8
   - pair_ok_3_8_for_nic_out_3_nic_in_8 <= 0
 constr_pair_ok1_3_9: start_3 - start_9 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 8 hold_3_8 + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11 + 12 hold_3_12
   + 13 hold_3_13 + 14 hold_3_14 + 15 hold_3_15 + 16 hold_3_16
   + 17 hold_3_17 + 18 hold_3_18 + 60 pair_ok_3_9_for_nic_out_3_nic_in_9
   <= 60
 constr_pair_ok2_3_9: - start_3 + start_9 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   - 8 hold_3_8 - 9 hold_3_9 - 10 hold_3_10 - 11 hold_3_11 - 12 hold_3_12
   - 13 hold_3_13 - 14 hold_3_14 - 15 hold_3_15 - 16 hold_3_16
   - 17 hold_3_17 - 18 hold_3_18 + 60 pair_ok_3_9_for_nic_out_3_nic_in_9
   <= 60
 constr_edge_switch_end_start_3_9_OR: switch_nic_out_3_nic_in_9
   - pair_ok_3_9_for_nic_out_3_nic_in_9 <= 0
 constr_pair_ok1_3_10: start_3 - start_10 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 8 hold_3_8 + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11 + 12 hold_3_12
   + 13 hold_3_13 + 14 hold_3_14 + 15 hold_3_15 + 16 hold_3_16
   + 17 hold_3_17 + 18 hold_3_18 + 60 pair_ok_3_10_for_nic_out_3_nic_in_10
   <= 60
 constr_pair_ok2_3_10: - start_3 + start_10 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   - 8 hold_3_8 - 9 hold_3_9 - 10 hold_3_10 - 11 hold_3_11 - 12 hold_3_12
   - 13 hold_3_13 - 14 hold_3_14 - 15 hold_3_15 - 16 hold_3_16
   - 17 hold_3_17 - 18 hold_3_18 + 60 pair_ok_3_10_for_nic_out_3_nic_in_10
   <= 60
 constr_edge_switch_end_start_3_10_OR: switch_nic_out_3_nic_in_10
   - pair_ok_3_10_for_nic_out_3_nic_in_10 <= 0
 constr_pair_ok1_3_11: start_3 - start_11 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 8 hold_3_8 + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11 + 12 hold_3_12
   + 13 hold_3_13 + 14 hold_3_14 + 15 hold_3_15 + 16 hold_3_16
   + 17 hold_3_17 + 18 hold_3_18 + 60 pair_ok_3_11_for_nic_out_3_nic_in_11
   <= 60
 constr_pair_ok2_3_11: - start_3 + start_11 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   - 8 hold_3_8 - 9 hold_3_9 - 10 hold_3_10 - 11 hold_3_11 - 12 hold_3_12
   - 13 hold_3_13 - 14 hold_3_14 - 15 hold_3_15 - 16 hold_3_16
   - 17 hold_3_17 - 18 hold_3_18 + 60 pair_ok_3_11_for_nic_out_3_nic_in_11
   <= 60
 constr_edge_switch_end_start_3_11_OR: switch_nic_out_3_nic_in_11
   - pair_ok_3_11_for_nic_out_3_nic_in_11 <= 0
 constr_pair_ok1_3_12: start_3 - start_12 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 8 hold_3_8 + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11 + 12 hold_3_12
   + 13 hold_3_13 + 14 hold_3_14 + 15 hold_3_15 + 16 hold_3_16
   + 17 hold_3_17 + 18 hold_3_18 + 60 pair_ok_3_12_for_nic_out_3_nic_in_12
   <= 60
 constr_pair_ok2_3_12: - start_3 + start_12 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   - 8 hold_3_8 - 9 hold_3_9 - 10 hold_3_10 - 11 hold_3_11 - 12 hold_3_12
   - 13 hold_3_13 - 14 hold_3_14 - 15 hold_3_15 - 16 hold_3_16
   - 17 hold_3_17 - 18 hold_3_18 + 60 pair_ok_3_12_for_nic_out_3_nic_in_12
   <= 60
 constr_edge_switch_end_start_3_12_OR: switch_nic_out_3_nic_in_12
   - pair_ok_3_12_for_nic_out_3_nic_in_12 <= 0
 constr_pair_ok1_3_13: start_3 - start_13 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 8 hold_3_8 + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11 + 12 hold_3_12
   + 13 hold_3_13 + 14 hold_3_14 + 15 hold_3_15 + 16 hold_3_16
   + 17 hold_3_17 + 18 hold_3_18 + 60 pair_ok_3_13_for_nic_out_3_nic_in_13
   <= 60
 constr_pair_ok2_3_13: - start_3 + start_13 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   - 8 hold_3_8 - 9 hold_3_9 - 10 hold_3_10 - 11 hold_3_11 - 12 hold_3_12
   - 13 hold_3_13 - 14 hold_3_14 - 15 hold_3_15 - 16 hold_3_16
   - 17 hold_3_17 - 18 hold_3_18 + 60 pair_ok_3_13_for_nic_out_3_nic_in_13
   <= 60
 constr_edge_switch_end_start_3_13_OR: switch_nic_out_3_nic_in_13
   - pair_ok_3_13_for_nic_out_3_nic_in_13 <= 0
 constr_pair_ok1_3_14: start_3 - start_14 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 8 hold_3_8 + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11 + 12 hold_3_12
   + 13 hold_3_13 + 14 hold_3_14 + 15 hold_3_15 + 16 hold_3_16
   + 17 hold_3_17 + 18 hold_3_18 + 60 pair_ok_3_14_for_nic_out_3_nic_in_14
   <= 60
 constr_pair_ok2_3_14: - start_3 + start_14 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   - 8 hold_3_8 - 9 hold_3_9 - 10 hold_3_10 - 11 hold_3_11 - 12 hold_3_12
   - 13 hold_3_13 - 14 hold_3_14 - 15 hold_3_15 - 16 hold_3_16
   - 17 hold_3_17 - 18 hold_3_18 + 60 pair_ok_3_14_for_nic_out_3_nic_in_14
   <= 60
 constr_edge_switch_end_start_3_14_OR: switch_nic_out_3_nic_in_14
   - pair_ok_3_14_for_nic_out_3_nic_in_14 <= 0
 constr_pair_ok1_3_15: start_3 - start_15 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 8 hold_3_8 + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11 + 12 hold_3_12
   + 13 hold_3_13 + 14 hold_3_14 + 15 hold_3_15 + 16 hold_3_16
   + 17 hold_3_17 + 18 hold_3_18 + 60 pair_ok_3_15_for_nic_out_3_nic_in_15
   <= 60
 constr_pair_ok2_3_15: - start_3 + start_15 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   - 8 hold_3_8 - 9 hold_3_9 - 10 hold_3_10 - 11 hold_3_11 - 12 hold_3_12
   - 13 hold_3_13 - 14 hold_3_14 - 15 hold_3_15 - 16 hold_3_16
   - 17 hold_3_17 - 18 hold_3_18 + 60 pair_ok_3_15_for_nic_out_3_nic_in_15
   <= 60
 constr_edge_switch_end_start_3_15_OR: switch_nic_out_3_nic_in_15
   - pair_ok_3_15_for_nic_out_3_nic_in_15 <= 0
 constr_is_start_zero_4: start_4 + 60 is_start_zero_4_for_source_nic_in_4
   <= 60
 constr_edge_switch_start_zero_4_OR: switch_source_nic_in_4
   - is_start_zero_4_for_source_nic_in_4 <= 0
 constr_is_end_complete_4: start_4 + hold_4_1 + 2 hold_4_2 + 3 hold_4_3
   + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7 + 8 hold_4_8
   + 9 hold_4_9 + 10 hold_4_10 + 11 hold_4_11 + 12 hold_4_12 + 13 hold_4_13
   + 14 hold_4_14 + 15 hold_4_15 + 16 hold_4_16 + 17 hold_4_17
   + 18 hold_4_18 - 60 is_end_complete_4 >= 0
 constr_edge_switch_end_max_4_OR: switch_nic_out_4_sink - is_end_complete_4
   <= 0
 constr_pair_ok1_4_0: - start_0 + start_4 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 8 hold_4_8 + 9 hold_4_9 + 10 hold_4_10 + 11 hold_4_11 + 12 hold_4_12
   + 13 hold_4_13 + 14 hold_4_14 + 15 hold_4_15 + 16 hold_4_16
   + 17 hold_4_17 + 18 hold_4_18 + 60 pair_ok_4_0_for_nic_out_4_nic_in_0
   <= 60
 constr_pair_ok2_4_0: start_0 - start_4 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   - 8 hold_4_8 - 9 hold_4_9 - 10 hold_4_10 - 11 hold_4_11 - 12 hold_4_12
   - 13 hold_4_13 - 14 hold_4_14 - 15 hold_4_15 - 16 hold_4_16
   - 17 hold_4_17 - 18 hold_4_18 + 60 pair_ok_4_0_for_nic_out_4_nic_in_0
   <= 60
 constr_edge_switch_end_start_4_0_OR: switch_nic_out_4_nic_in_0
   - pair_ok_4_0_for_nic_out_4_nic_in_0 <= 0
 constr_pair_ok1_4_1: - start_1 + start_4 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 8 hold_4_8 + 9 hold_4_9 + 10 hold_4_10 + 11 hold_4_11 + 12 hold_4_12
   + 13 hold_4_13 + 14 hold_4_14 + 15 hold_4_15 + 16 hold_4_16
   + 17 hold_4_17 + 18 hold_4_18 + 60 pair_ok_4_1_for_nic_out_4_nic_in_1
   <= 60
 constr_pair_ok2_4_1: start_1 - start_4 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   - 8 hold_4_8 - 9 hold_4_9 - 10 hold_4_10 - 11 hold_4_11 - 12 hold_4_12
   - 13 hold_4_13 - 14 hold_4_14 - 15 hold_4_15 - 16 hold_4_16
   - 17 hold_4_17 - 18 hold_4_18 + 60 pair_ok_4_1_for_nic_out_4_nic_in_1
   <= 60
 constr_edge_switch_end_start_4_1_OR: switch_nic_out_4_nic_in_1
   - pair_ok_4_1_for_nic_out_4_nic_in_1 <= 0
 constr_pair_ok1_4_2: - start_2 + start_4 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 8 hold_4_8 + 9 hold_4_9 + 10 hold_4_10 + 11 hold_4_11 + 12 hold_4_12
   + 13 hold_4_13 + 14 hold_4_14 + 15 hold_4_15 + 16 hold_4_16
   + 17 hold_4_17 + 18 hold_4_18 + 60 pair_ok_4_2_for_nic_out_4_nic_in_2
   <= 60
 constr_pair_ok2_4_2: start_2 - start_4 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   - 8 hold_4_8 - 9 hold_4_9 - 10 hold_4_10 - 11 hold_4_11 - 12 hold_4_12
   - 13 hold_4_13 - 14 hold_4_14 - 15 hold_4_15 - 16 hold_4_16
   - 17 hold_4_17 - 18 hold_4_18 + 60 pair_ok_4_2_for_nic_out_4_nic_in_2
   <= 60
 constr_edge_switch_end_start_4_2_OR: switch_nic_out_4_nic_in_2
   - pair_ok_4_2_for_nic_out_4_nic_in_2 <= 0
 constr_pair_ok1_4_3: - start_3 + start_4 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 8 hold_4_8 + 9 hold_4_9 + 10 hold_4_10 + 11 hold_4_11 + 12 hold_4_12
   + 13 hold_4_13 + 14 hold_4_14 + 15 hold_4_15 + 16 hold_4_16
   + 17 hold_4_17 + 18 hold_4_18 + 60 pair_ok_4_3_for_nic_out_4_nic_in_3
   <= 60
 constr_pair_ok2_4_3: start_3 - start_4 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   - 8 hold_4_8 - 9 hold_4_9 - 10 hold_4_10 - 11 hold_4_11 - 12 hold_4_12
   - 13 hold_4_13 - 14 hold_4_14 - 15 hold_4_15 - 16 hold_4_16
   - 17 hold_4_17 - 18 hold_4_18 + 60 pair_ok_4_3_for_nic_out_4_nic_in_3
   <= 60
 constr_edge_switch_end_start_4_3_OR: switch_nic_out_4_nic_in_3
   - pair_ok_4_3_for_nic_out_4_nic_in_3 <= 0
 constr_pair_ok1_4_5: start_4 - start_5 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 8 hold_4_8 + 9 hold_4_9 + 10 hold_4_10 + 11 hold_4_11 + 12 hold_4_12
   + 13 hold_4_13 + 14 hold_4_14 + 15 hold_4_15 + 16 hold_4_16
   + 17 hold_4_17 + 18 hold_4_18 + 60 pair_ok_4_5_for_nic_out_4_nic_in_5
   <= 60
 constr_pair_ok2_4_5: - start_4 + start_5 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   - 8 hold_4_8 - 9 hold_4_9 - 10 hold_4_10 - 11 hold_4_11 - 12 hold_4_12
   - 13 hold_4_13 - 14 hold_4_14 - 15 hold_4_15 - 16 hold_4_16
   - 17 hold_4_17 - 18 hold_4_18 + 60 pair_ok_4_5_for_nic_out_4_nic_in_5
   <= 60
 constr_edge_switch_end_start_4_5_OR: switch_nic_out_4_nic_in_5
   - pair_ok_4_5_for_nic_out_4_nic_in_5 <= 0
 constr_pair_ok1_4_6: start_4 - start_6 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 8 hold_4_8 + 9 hold_4_9 + 10 hold_4_10 + 11 hold_4_11 + 12 hold_4_12
   + 13 hold_4_13 + 14 hold_4_14 + 15 hold_4_15 + 16 hold_4_16
   + 17 hold_4_17 + 18 hold_4_18 + 60 pair_ok_4_6_for_nic_out_4_nic_in_6
   <= 60
 constr_pair_ok2_4_6: - start_4 + start_6 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   - 8 hold_4_8 - 9 hold_4_9 - 10 hold_4_10 - 11 hold_4_11 - 12 hold_4_12
   - 13 hold_4_13 - 14 hold_4_14 - 15 hold_4_15 - 16 hold_4_16
   - 17 hold_4_17 - 18 hold_4_18 + 60 pair_ok_4_6_for_nic_out_4_nic_in_6
   <= 60
 constr_edge_switch_end_start_4_6_OR: switch_nic_out_4_nic_in_6
   - pair_ok_4_6_for_nic_out_4_nic_in_6 <= 0
 constr_pair_ok1_4_7: start_4 - start_7 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 8 hold_4_8 + 9 hold_4_9 + 10 hold_4_10 + 11 hold_4_11 + 12 hold_4_12
   + 13 hold_4_13 + 14 hold_4_14 + 15 hold_4_15 + 16 hold_4_16
   + 17 hold_4_17 + 18 hold_4_18 + 60 pair_ok_4_7_for_nic_out_4_nic_in_7
   <= 60
 constr_pair_ok2_4_7: - start_4 + start_7 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   - 8 hold_4_8 - 9 hold_4_9 - 10 hold_4_10 - 11 hold_4_11 - 12 hold_4_12
   - 13 hold_4_13 - 14 hold_4_14 - 15 hold_4_15 - 16 hold_4_16
   - 17 hold_4_17 - 18 hold_4_18 + 60 pair_ok_4_7_for_nic_out_4_nic_in_7
   <= 60
 constr_edge_switch_end_start_4_7_OR: switch_nic_out_4_nic_in_7
   - pair_ok_4_7_for_nic_out_4_nic_in_7 <= 0
 constr_pair_ok1_4_8: start_4 - start_8 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 8 hold_4_8 + 9 hold_4_9 + 10 hold_4_10 + 11 hold_4_11 + 12 hold_4_12
   + 13 hold_4_13 + 14 hold_4_14 + 15 hold_4_15 + 16 hold_4_16
   + 17 hold_4_17 + 18 hold_4_18 + 60 pair_ok_4_8_for_nic_out_4_nic_in_8
   <= 60
 constr_pair_ok2_4_8: - start_4 + start_8 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   - 8 hold_4_8 - 9 hold_4_9 - 10 hold_4_10 - 11 hold_4_11 - 12 hold_4_12
   - 13 hold_4_13 - 14 hold_4_14 - 15 hold_4_15 - 16 hold_4_16
   - 17 hold_4_17 - 18 hold_4_18 + 60 pair_ok_4_8_for_nic_out_4_nic_in_8
   <= 60
 constr_edge_switch_end_start_4_8_OR: switch_nic_out_4_nic_in_8
   - pair_ok_4_8_for_nic_out_4_nic_in_8 <= 0
 constr_pair_ok1_4_9: start_4 - start_9 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 8 hold_4_8 + 9 hold_4_9 + 10 hold_4_10 + 11 hold_4_11 + 12 hold_4_12
   + 13 hold_4_13 + 14 hold_4_14 + 15 hold_4_15 + 16 hold_4_16
   + 17 hold_4_17 + 18 hold_4_18 + 60 pair_ok_4_9_for_nic_out_4_nic_in_9
   <= 60
 constr_pair_ok2_4_9: - start_4 + start_9 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   - 8 hold_4_8 - 9 hold_4_9 - 10 hold_4_10 - 11 hold_4_11 - 12 hold_4_12
   - 13 hold_4_13 - 14 hold_4_14 - 15 hold_4_15 - 16 hold_4_16
   - 17 hold_4_17 - 18 hold_4_18 + 60 pair_ok_4_9_for_nic_out_4_nic_in_9
   <= 60
 constr_edge_switch_end_start_4_9_OR: switch_nic_out_4_nic_in_9
   - pair_ok_4_9_for_nic_out_4_nic_in_9 <= 0
 constr_pair_ok1_4_10: start_4 - start_10 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 8 hold_4_8 + 9 hold_4_9 + 10 hold_4_10 + 11 hold_4_11 + 12 hold_4_12
   + 13 hold_4_13 + 14 hold_4_14 + 15 hold_4_15 + 16 hold_4_16
   + 17 hold_4_17 + 18 hold_4_18 + 60 pair_ok_4_10_for_nic_out_4_nic_in_10
   <= 60
 constr_pair_ok2_4_10: - start_4 + start_10 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   - 8 hold_4_8 - 9 hold_4_9 - 10 hold_4_10 - 11 hold_4_11 - 12 hold_4_12
   - 13 hold_4_13 - 14 hold_4_14 - 15 hold_4_15 - 16 hold_4_16
   - 17 hold_4_17 - 18 hold_4_18 + 60 pair_ok_4_10_for_nic_out_4_nic_in_10
   <= 60
 constr_edge_switch_end_start_4_10_OR: switch_nic_out_4_nic_in_10
   - pair_ok_4_10_for_nic_out_4_nic_in_10 <= 0
 constr_pair_ok1_4_11: start_4 - start_11 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 8 hold_4_8 + 9 hold_4_9 + 10 hold_4_10 + 11 hold_4_11 + 12 hold_4_12
   + 13 hold_4_13 + 14 hold_4_14 + 15 hold_4_15 + 16 hold_4_16
   + 17 hold_4_17 + 18 hold_4_18 + 60 pair_ok_4_11_for_nic_out_4_nic_in_11
   <= 60
 constr_pair_ok2_4_11: - start_4 + start_11 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   - 8 hold_4_8 - 9 hold_4_9 - 10 hold_4_10 - 11 hold_4_11 - 12 hold_4_12
   - 13 hold_4_13 - 14 hold_4_14 - 15 hold_4_15 - 16 hold_4_16
   - 17 hold_4_17 - 18 hold_4_18 + 60 pair_ok_4_11_for_nic_out_4_nic_in_11
   <= 60
 constr_edge_switch_end_start_4_11_OR: switch_nic_out_4_nic_in_11
   - pair_ok_4_11_for_nic_out_4_nic_in_11 <= 0
 constr_pair_ok1_4_12: start_4 - start_12 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 8 hold_4_8 + 9 hold_4_9 + 10 hold_4_10 + 11 hold_4_11 + 12 hold_4_12
   + 13 hold_4_13 + 14 hold_4_14 + 15 hold_4_15 + 16 hold_4_16
   + 17 hold_4_17 + 18 hold_4_18 + 60 pair_ok_4_12_for_nic_out_4_nic_in_12
   <= 60
 constr_pair_ok2_4_12: - start_4 + start_12 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   - 8 hold_4_8 - 9 hold_4_9 - 10 hold_4_10 - 11 hold_4_11 - 12 hold_4_12
   - 13 hold_4_13 - 14 hold_4_14 - 15 hold_4_15 - 16 hold_4_16
   - 17 hold_4_17 - 18 hold_4_18 + 60 pair_ok_4_12_for_nic_out_4_nic_in_12
   <= 60
 constr_edge_switch_end_start_4_12_OR: switch_nic_out_4_nic_in_12
   - pair_ok_4_12_for_nic_out_4_nic_in_12 <= 0
 constr_pair_ok1_4_13: start_4 - start_13 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 8 hold_4_8 + 9 hold_4_9 + 10 hold_4_10 + 11 hold_4_11 + 12 hold_4_12
   + 13 hold_4_13 + 14 hold_4_14 + 15 hold_4_15 + 16 hold_4_16
   + 17 hold_4_17 + 18 hold_4_18 + 60 pair_ok_4_13_for_nic_out_4_nic_in_13
   <= 60
 constr_pair_ok2_4_13: - start_4 + start_13 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   - 8 hold_4_8 - 9 hold_4_9 - 10 hold_4_10 - 11 hold_4_11 - 12 hold_4_12
   - 13 hold_4_13 - 14 hold_4_14 - 15 hold_4_15 - 16 hold_4_16
   - 17 hold_4_17 - 18 hold_4_18 + 60 pair_ok_4_13_for_nic_out_4_nic_in_13
   <= 60
 constr_edge_switch_end_start_4_13_OR: switch_nic_out_4_nic_in_13
   - pair_ok_4_13_for_nic_out_4_nic_in_13 <= 0
 constr_pair_ok1_4_14: start_4 - start_14 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 8 hold_4_8 + 9 hold_4_9 + 10 hold_4_10 + 11 hold_4_11 + 12 hold_4_12
   + 13 hold_4_13 + 14 hold_4_14 + 15 hold_4_15 + 16 hold_4_16
   + 17 hold_4_17 + 18 hold_4_18 + 60 pair_ok_4_14_for_nic_out_4_nic_in_14
   <= 60
 constr_pair_ok2_4_14: - start_4 + start_14 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   - 8 hold_4_8 - 9 hold_4_9 - 10 hold_4_10 - 11 hold_4_11 - 12 hold_4_12
   - 13 hold_4_13 - 14 hold_4_14 - 15 hold_4_15 - 16 hold_4_16
   - 17 hold_4_17 - 18 hold_4_18 + 60 pair_ok_4_14_for_nic_out_4_nic_in_14
   <= 60
 constr_edge_switch_end_start_4_14_OR: switch_nic_out_4_nic_in_14
   - pair_ok_4_14_for_nic_out_4_nic_in_14 <= 0
 constr_pair_ok1_4_15: start_4 - start_15 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 8 hold_4_8 + 9 hold_4_9 + 10 hold_4_10 + 11 hold_4_11 + 12 hold_4_12
   + 13 hold_4_13 + 14 hold_4_14 + 15 hold_4_15 + 16 hold_4_16
   + 17 hold_4_17 + 18 hold_4_18 + 60 pair_ok_4_15_for_nic_out_4_nic_in_15
   <= 60
 constr_pair_ok2_4_15: - start_4 + start_15 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   - 8 hold_4_8 - 9 hold_4_9 - 10 hold_4_10 - 11 hold_4_11 - 12 hold_4_12
   - 13 hold_4_13 - 14 hold_4_14 - 15 hold_4_15 - 16 hold_4_16
   - 17 hold_4_17 - 18 hold_4_18 + 60 pair_ok_4_15_for_nic_out_4_nic_in_15
   <= 60
 constr_edge_switch_end_start_4_15_OR: switch_nic_out_4_nic_in_15
   - pair_ok_4_15_for_nic_out_4_nic_in_15 <= 0
 constr_is_start_zero_5: start_5 + 60 is_start_zero_5_for_source_nic_in_5
   <= 60
 constr_edge_switch_start_zero_5_OR: switch_source_nic_in_5
   - is_start_zero_5_for_source_nic_in_5 <= 0
 constr_is_end_complete_5: start_5 + hold_5_1 + 2 hold_5_2 + 3 hold_5_3
   + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7 + 8 hold_5_8
   + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12 + 13 hold_5_13
   + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16 + 17 hold_5_17
   + 18 hold_5_18 - 60 is_end_complete_5 >= 0
 constr_edge_switch_end_max_5_OR: switch_nic_out_5_sink - is_end_complete_5
   <= 0
 constr_pair_ok1_5_0: - start_0 + start_5 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12
   + 13 hold_5_13 + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16
   + 17 hold_5_17 + 18 hold_5_18 + 60 pair_ok_5_0_for_nic_out_5_nic_in_0
   <= 60
 constr_pair_ok2_5_0: start_0 - start_5 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11 - 12 hold_5_12
   - 13 hold_5_13 - 14 hold_5_14 - 15 hold_5_15 - 16 hold_5_16
   - 17 hold_5_17 - 18 hold_5_18 + 60 pair_ok_5_0_for_nic_out_5_nic_in_0
   <= 60
 constr_edge_switch_end_start_5_0_OR: switch_nic_out_5_nic_in_0
   - pair_ok_5_0_for_nic_out_5_nic_in_0 <= 0
 constr_pair_ok1_5_1: - start_1 + start_5 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12
   + 13 hold_5_13 + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16
   + 17 hold_5_17 + 18 hold_5_18 + 60 pair_ok_5_1_for_nic_out_5_nic_in_1
   <= 60
 constr_pair_ok2_5_1: start_1 - start_5 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11 - 12 hold_5_12
   - 13 hold_5_13 - 14 hold_5_14 - 15 hold_5_15 - 16 hold_5_16
   - 17 hold_5_17 - 18 hold_5_18 + 60 pair_ok_5_1_for_nic_out_5_nic_in_1
   <= 60
 constr_edge_switch_end_start_5_1_OR: switch_nic_out_5_nic_in_1
   - pair_ok_5_1_for_nic_out_5_nic_in_1 <= 0
 constr_pair_ok1_5_2: - start_2 + start_5 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12
   + 13 hold_5_13 + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16
   + 17 hold_5_17 + 18 hold_5_18 + 60 pair_ok_5_2_for_nic_out_5_nic_in_2
   <= 60
 constr_pair_ok2_5_2: start_2 - start_5 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11 - 12 hold_5_12
   - 13 hold_5_13 - 14 hold_5_14 - 15 hold_5_15 - 16 hold_5_16
   - 17 hold_5_17 - 18 hold_5_18 + 60 pair_ok_5_2_for_nic_out_5_nic_in_2
   <= 60
 constr_edge_switch_end_start_5_2_OR: switch_nic_out_5_nic_in_2
   - pair_ok_5_2_for_nic_out_5_nic_in_2 <= 0
 constr_pair_ok1_5_3: - start_3 + start_5 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12
   + 13 hold_5_13 + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16
   + 17 hold_5_17 + 18 hold_5_18 + 60 pair_ok_5_3_for_nic_out_5_nic_in_3
   <= 60
 constr_pair_ok2_5_3: start_3 - start_5 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11 - 12 hold_5_12
   - 13 hold_5_13 - 14 hold_5_14 - 15 hold_5_15 - 16 hold_5_16
   - 17 hold_5_17 - 18 hold_5_18 + 60 pair_ok_5_3_for_nic_out_5_nic_in_3
   <= 60
 constr_edge_switch_end_start_5_3_OR: switch_nic_out_5_nic_in_3
   - pair_ok_5_3_for_nic_out_5_nic_in_3 <= 0
 constr_pair_ok1_5_4: - start_4 + start_5 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12
   + 13 hold_5_13 + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16
   + 17 hold_5_17 + 18 hold_5_18 + 60 pair_ok_5_4_for_nic_out_5_nic_in_4
   <= 60
 constr_pair_ok2_5_4: start_4 - start_5 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11 - 12 hold_5_12
   - 13 hold_5_13 - 14 hold_5_14 - 15 hold_5_15 - 16 hold_5_16
   - 17 hold_5_17 - 18 hold_5_18 + 60 pair_ok_5_4_for_nic_out_5_nic_in_4
   <= 60
 constr_edge_switch_end_start_5_4_OR: switch_nic_out_5_nic_in_4
   - pair_ok_5_4_for_nic_out_5_nic_in_4 <= 0
 constr_pair_ok1_5_6: start_5 - start_6 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12
   + 13 hold_5_13 + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16
   + 17 hold_5_17 + 18 hold_5_18 + 60 pair_ok_5_6_for_nic_out_5_nic_in_6
   <= 60
 constr_pair_ok2_5_6: - start_5 + start_6 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11 - 12 hold_5_12
   - 13 hold_5_13 - 14 hold_5_14 - 15 hold_5_15 - 16 hold_5_16
   - 17 hold_5_17 - 18 hold_5_18 + 60 pair_ok_5_6_for_nic_out_5_nic_in_6
   <= 60
 constr_edge_switch_end_start_5_6_OR: switch_nic_out_5_nic_in_6
   - pair_ok_5_6_for_nic_out_5_nic_in_6 <= 0
 constr_pair_ok1_5_7: start_5 - start_7 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12
   + 13 hold_5_13 + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16
   + 17 hold_5_17 + 18 hold_5_18 + 60 pair_ok_5_7_for_nic_out_5_nic_in_7
   <= 60
 constr_pair_ok2_5_7: - start_5 + start_7 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11 - 12 hold_5_12
   - 13 hold_5_13 - 14 hold_5_14 - 15 hold_5_15 - 16 hold_5_16
   - 17 hold_5_17 - 18 hold_5_18 + 60 pair_ok_5_7_for_nic_out_5_nic_in_7
   <= 60
 constr_edge_switch_end_start_5_7_OR: switch_nic_out_5_nic_in_7
   - pair_ok_5_7_for_nic_out_5_nic_in_7 <= 0
 constr_pair_ok1_5_8: start_5 - start_8 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12
   + 13 hold_5_13 + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16
   + 17 hold_5_17 + 18 hold_5_18 + 60 pair_ok_5_8_for_nic_out_5_nic_in_8
   <= 60
 constr_pair_ok2_5_8: - start_5 + start_8 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11 - 12 hold_5_12
   - 13 hold_5_13 - 14 hold_5_14 - 15 hold_5_15 - 16 hold_5_16
   - 17 hold_5_17 - 18 hold_5_18 + 60 pair_ok_5_8_for_nic_out_5_nic_in_8
   <= 60
 constr_edge_switch_end_start_5_8_OR: switch_nic_out_5_nic_in_8
   - pair_ok_5_8_for_nic_out_5_nic_in_8 <= 0
 constr_pair_ok1_5_9: start_5 - start_9 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12
   + 13 hold_5_13 + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16
   + 17 hold_5_17 + 18 hold_5_18 + 60 pair_ok_5_9_for_nic_out_5_nic_in_9
   <= 60
 constr_pair_ok2_5_9: - start_5 + start_9 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11 - 12 hold_5_12
   - 13 hold_5_13 - 14 hold_5_14 - 15 hold_5_15 - 16 hold_5_16
   - 17 hold_5_17 - 18 hold_5_18 + 60 pair_ok_5_9_for_nic_out_5_nic_in_9
   <= 60
 constr_edge_switch_end_start_5_9_OR: switch_nic_out_5_nic_in_9
   - pair_ok_5_9_for_nic_out_5_nic_in_9 <= 0
 constr_pair_ok1_5_10: start_5 - start_10 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12
   + 13 hold_5_13 + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16
   + 17 hold_5_17 + 18 hold_5_18 + 60 pair_ok_5_10_for_nic_out_5_nic_in_10
   <= 60
 constr_pair_ok2_5_10: - start_5 + start_10 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11 - 12 hold_5_12
   - 13 hold_5_13 - 14 hold_5_14 - 15 hold_5_15 - 16 hold_5_16
   - 17 hold_5_17 - 18 hold_5_18 + 60 pair_ok_5_10_for_nic_out_5_nic_in_10
   <= 60
 constr_edge_switch_end_start_5_10_OR: switch_nic_out_5_nic_in_10
   - pair_ok_5_10_for_nic_out_5_nic_in_10 <= 0
 constr_pair_ok1_5_11: start_5 - start_11 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12
   + 13 hold_5_13 + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16
   + 17 hold_5_17 + 18 hold_5_18 + 60 pair_ok_5_11_for_nic_out_5_nic_in_11
   <= 60
 constr_pair_ok2_5_11: - start_5 + start_11 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11 - 12 hold_5_12
   - 13 hold_5_13 - 14 hold_5_14 - 15 hold_5_15 - 16 hold_5_16
   - 17 hold_5_17 - 18 hold_5_18 + 60 pair_ok_5_11_for_nic_out_5_nic_in_11
   <= 60
 constr_edge_switch_end_start_5_11_OR: switch_nic_out_5_nic_in_11
   - pair_ok_5_11_for_nic_out_5_nic_in_11 <= 0
 constr_pair_ok1_5_12: start_5 - start_12 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12
   + 13 hold_5_13 + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16
   + 17 hold_5_17 + 18 hold_5_18 + 60 pair_ok_5_12_for_nic_out_5_nic_in_12
   <= 60
 constr_pair_ok2_5_12: - start_5 + start_12 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11 - 12 hold_5_12
   - 13 hold_5_13 - 14 hold_5_14 - 15 hold_5_15 - 16 hold_5_16
   - 17 hold_5_17 - 18 hold_5_18 + 60 pair_ok_5_12_for_nic_out_5_nic_in_12
   <= 60
 constr_edge_switch_end_start_5_12_OR: switch_nic_out_5_nic_in_12
   - pair_ok_5_12_for_nic_out_5_nic_in_12 <= 0
 constr_pair_ok1_5_13: start_5 - start_13 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12
   + 13 hold_5_13 + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16
   + 17 hold_5_17 + 18 hold_5_18 + 60 pair_ok_5_13_for_nic_out_5_nic_in_13
   <= 60
 constr_pair_ok2_5_13: - start_5 + start_13 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11 - 12 hold_5_12
   - 13 hold_5_13 - 14 hold_5_14 - 15 hold_5_15 - 16 hold_5_16
   - 17 hold_5_17 - 18 hold_5_18 + 60 pair_ok_5_13_for_nic_out_5_nic_in_13
   <= 60
 constr_edge_switch_end_start_5_13_OR: switch_nic_out_5_nic_in_13
   - pair_ok_5_13_for_nic_out_5_nic_in_13 <= 0
 constr_pair_ok1_5_14: start_5 - start_14 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12
   + 13 hold_5_13 + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16
   + 17 hold_5_17 + 18 hold_5_18 + 60 pair_ok_5_14_for_nic_out_5_nic_in_14
   <= 60
 constr_pair_ok2_5_14: - start_5 + start_14 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11 - 12 hold_5_12
   - 13 hold_5_13 - 14 hold_5_14 - 15 hold_5_15 - 16 hold_5_16
   - 17 hold_5_17 - 18 hold_5_18 + 60 pair_ok_5_14_for_nic_out_5_nic_in_14
   <= 60
 constr_edge_switch_end_start_5_14_OR: switch_nic_out_5_nic_in_14
   - pair_ok_5_14_for_nic_out_5_nic_in_14 <= 0
 constr_pair_ok1_5_15: start_5 - start_15 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12
   + 13 hold_5_13 + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16
   + 17 hold_5_17 + 18 hold_5_18 + 60 pair_ok_5_15_for_nic_out_5_nic_in_15
   <= 60
 constr_pair_ok2_5_15: - start_5 + start_15 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11 - 12 hold_5_12
   - 13 hold_5_13 - 14 hold_5_14 - 15 hold_5_15 - 16 hold_5_16
   - 17 hold_5_17 - 18 hold_5_18 + 60 pair_ok_5_15_for_nic_out_5_nic_in_15
   <= 60
 constr_edge_switch_end_start_5_15_OR: switch_nic_out_5_nic_in_15
   - pair_ok_5_15_for_nic_out_5_nic_in_15 <= 0
 constr_is_start_zero_6: start_6 + 60 is_start_zero_6_for_source_nic_in_6
   <= 60
 constr_edge_switch_start_zero_6_OR: switch_source_nic_in_6
   - is_start_zero_6_for_source_nic_in_6 <= 0
 constr_is_end_complete_6: start_6 + hold_6_1 + 2 hold_6_2 + 3 hold_6_3
   + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7 + 8 hold_6_8
   + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 12 hold_6_12 + 13 hold_6_13
   + 14 hold_6_14 + 15 hold_6_15 + 16 hold_6_16 + 17 hold_6_17
   + 18 hold_6_18 - 60 is_end_complete_6 >= 0
 constr_edge_switch_end_max_6_OR: switch_nic_out_6_sink - is_end_complete_6
   <= 0
 constr_pair_ok1_6_0: - start_0 + start_6 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 12 hold_6_12
   + 13 hold_6_13 + 14 hold_6_14 + 15 hold_6_15 + 16 hold_6_16
   + 17 hold_6_17 + 18 hold_6_18 + 60 pair_ok_6_0_for_nic_out_6_nic_in_0
   <= 60
 constr_pair_ok2_6_0: start_0 - start_6 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11 - 12 hold_6_12
   - 13 hold_6_13 - 14 hold_6_14 - 15 hold_6_15 - 16 hold_6_16
   - 17 hold_6_17 - 18 hold_6_18 + 60 pair_ok_6_0_for_nic_out_6_nic_in_0
   <= 60
 constr_edge_switch_end_start_6_0_OR: switch_nic_out_6_nic_in_0
   - pair_ok_6_0_for_nic_out_6_nic_in_0 <= 0
 constr_pair_ok1_6_1: - start_1 + start_6 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 12 hold_6_12
   + 13 hold_6_13 + 14 hold_6_14 + 15 hold_6_15 + 16 hold_6_16
   + 17 hold_6_17 + 18 hold_6_18 + 60 pair_ok_6_1_for_nic_out_6_nic_in_1
   <= 60
 constr_pair_ok2_6_1: start_1 - start_6 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11 - 12 hold_6_12
   - 13 hold_6_13 - 14 hold_6_14 - 15 hold_6_15 - 16 hold_6_16
   - 17 hold_6_17 - 18 hold_6_18 + 60 pair_ok_6_1_for_nic_out_6_nic_in_1
   <= 60
 constr_edge_switch_end_start_6_1_OR: switch_nic_out_6_nic_in_1
   - pair_ok_6_1_for_nic_out_6_nic_in_1 <= 0
 constr_pair_ok1_6_2: - start_2 + start_6 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 12 hold_6_12
   + 13 hold_6_13 + 14 hold_6_14 + 15 hold_6_15 + 16 hold_6_16
   + 17 hold_6_17 + 18 hold_6_18 + 60 pair_ok_6_2_for_nic_out_6_nic_in_2
   <= 60
 constr_pair_ok2_6_2: start_2 - start_6 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11 - 12 hold_6_12
   - 13 hold_6_13 - 14 hold_6_14 - 15 hold_6_15 - 16 hold_6_16
   - 17 hold_6_17 - 18 hold_6_18 + 60 pair_ok_6_2_for_nic_out_6_nic_in_2
   <= 60
 constr_edge_switch_end_start_6_2_OR: switch_nic_out_6_nic_in_2
   - pair_ok_6_2_for_nic_out_6_nic_in_2 <= 0
 constr_pair_ok1_6_3: - start_3 + start_6 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 12 hold_6_12
   + 13 hold_6_13 + 14 hold_6_14 + 15 hold_6_15 + 16 hold_6_16
   + 17 hold_6_17 + 18 hold_6_18 + 60 pair_ok_6_3_for_nic_out_6_nic_in_3
   <= 60
 constr_pair_ok2_6_3: start_3 - start_6 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11 - 12 hold_6_12
   - 13 hold_6_13 - 14 hold_6_14 - 15 hold_6_15 - 16 hold_6_16
   - 17 hold_6_17 - 18 hold_6_18 + 60 pair_ok_6_3_for_nic_out_6_nic_in_3
   <= 60
 constr_edge_switch_end_start_6_3_OR: switch_nic_out_6_nic_in_3
   - pair_ok_6_3_for_nic_out_6_nic_in_3 <= 0
 constr_pair_ok1_6_4: - start_4 + start_6 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 12 hold_6_12
   + 13 hold_6_13 + 14 hold_6_14 + 15 hold_6_15 + 16 hold_6_16
   + 17 hold_6_17 + 18 hold_6_18 + 60 pair_ok_6_4_for_nic_out_6_nic_in_4
   <= 60
 constr_pair_ok2_6_4: start_4 - start_6 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11 - 12 hold_6_12
   - 13 hold_6_13 - 14 hold_6_14 - 15 hold_6_15 - 16 hold_6_16
   - 17 hold_6_17 - 18 hold_6_18 + 60 pair_ok_6_4_for_nic_out_6_nic_in_4
   <= 60
 constr_edge_switch_end_start_6_4_OR: switch_nic_out_6_nic_in_4
   - pair_ok_6_4_for_nic_out_6_nic_in_4 <= 0
 constr_pair_ok1_6_5: - start_5 + start_6 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 12 hold_6_12
   + 13 hold_6_13 + 14 hold_6_14 + 15 hold_6_15 + 16 hold_6_16
   + 17 hold_6_17 + 18 hold_6_18 + 60 pair_ok_6_5_for_nic_out_6_nic_in_5
   <= 60
 constr_pair_ok2_6_5: start_5 - start_6 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11 - 12 hold_6_12
   - 13 hold_6_13 - 14 hold_6_14 - 15 hold_6_15 - 16 hold_6_16
   - 17 hold_6_17 - 18 hold_6_18 + 60 pair_ok_6_5_for_nic_out_6_nic_in_5
   <= 60
 constr_edge_switch_end_start_6_5_OR: switch_nic_out_6_nic_in_5
   - pair_ok_6_5_for_nic_out_6_nic_in_5 <= 0
 constr_pair_ok1_6_7: start_6 - start_7 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 12 hold_6_12
   + 13 hold_6_13 + 14 hold_6_14 + 15 hold_6_15 + 16 hold_6_16
   + 17 hold_6_17 + 18 hold_6_18 + 60 pair_ok_6_7_for_nic_out_6_nic_in_7
   <= 60
 constr_pair_ok2_6_7: - start_6 + start_7 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11 - 12 hold_6_12
   - 13 hold_6_13 - 14 hold_6_14 - 15 hold_6_15 - 16 hold_6_16
   - 17 hold_6_17 - 18 hold_6_18 + 60 pair_ok_6_7_for_nic_out_6_nic_in_7
   <= 60
 constr_edge_switch_end_start_6_7_OR: switch_nic_out_6_nic_in_7
   - pair_ok_6_7_for_nic_out_6_nic_in_7 <= 0
 constr_pair_ok1_6_8: start_6 - start_8 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 12 hold_6_12
   + 13 hold_6_13 + 14 hold_6_14 + 15 hold_6_15 + 16 hold_6_16
   + 17 hold_6_17 + 18 hold_6_18 + 60 pair_ok_6_8_for_nic_out_6_nic_in_8
   <= 60
 constr_pair_ok2_6_8: - start_6 + start_8 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11 - 12 hold_6_12
   - 13 hold_6_13 - 14 hold_6_14 - 15 hold_6_15 - 16 hold_6_16
   - 17 hold_6_17 - 18 hold_6_18 + 60 pair_ok_6_8_for_nic_out_6_nic_in_8
   <= 60
 constr_edge_switch_end_start_6_8_OR: switch_nic_out_6_nic_in_8
   - pair_ok_6_8_for_nic_out_6_nic_in_8 <= 0
 constr_pair_ok1_6_9: start_6 - start_9 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 12 hold_6_12
   + 13 hold_6_13 + 14 hold_6_14 + 15 hold_6_15 + 16 hold_6_16
   + 17 hold_6_17 + 18 hold_6_18 + 60 pair_ok_6_9_for_nic_out_6_nic_in_9
   <= 60
 constr_pair_ok2_6_9: - start_6 + start_9 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11 - 12 hold_6_12
   - 13 hold_6_13 - 14 hold_6_14 - 15 hold_6_15 - 16 hold_6_16
   - 17 hold_6_17 - 18 hold_6_18 + 60 pair_ok_6_9_for_nic_out_6_nic_in_9
   <= 60
 constr_edge_switch_end_start_6_9_OR: switch_nic_out_6_nic_in_9
   - pair_ok_6_9_for_nic_out_6_nic_in_9 <= 0
 constr_pair_ok1_6_10: start_6 - start_10 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 12 hold_6_12
   + 13 hold_6_13 + 14 hold_6_14 + 15 hold_6_15 + 16 hold_6_16
   + 17 hold_6_17 + 18 hold_6_18 + 60 pair_ok_6_10_for_nic_out_6_nic_in_10
   <= 60
 constr_pair_ok2_6_10: - start_6 + start_10 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11 - 12 hold_6_12
   - 13 hold_6_13 - 14 hold_6_14 - 15 hold_6_15 - 16 hold_6_16
   - 17 hold_6_17 - 18 hold_6_18 + 60 pair_ok_6_10_for_nic_out_6_nic_in_10
   <= 60
 constr_edge_switch_end_start_6_10_OR: switch_nic_out_6_nic_in_10
   - pair_ok_6_10_for_nic_out_6_nic_in_10 <= 0
 constr_pair_ok1_6_11: start_6 - start_11 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 12 hold_6_12
   + 13 hold_6_13 + 14 hold_6_14 + 15 hold_6_15 + 16 hold_6_16
   + 17 hold_6_17 + 18 hold_6_18 + 60 pair_ok_6_11_for_nic_out_6_nic_in_11
   <= 60
 constr_pair_ok2_6_11: - start_6 + start_11 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11 - 12 hold_6_12
   - 13 hold_6_13 - 14 hold_6_14 - 15 hold_6_15 - 16 hold_6_16
   - 17 hold_6_17 - 18 hold_6_18 + 60 pair_ok_6_11_for_nic_out_6_nic_in_11
   <= 60
 constr_edge_switch_end_start_6_11_OR: switch_nic_out_6_nic_in_11
   - pair_ok_6_11_for_nic_out_6_nic_in_11 <= 0
 constr_pair_ok1_6_12: start_6 - start_12 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 12 hold_6_12
   + 13 hold_6_13 + 14 hold_6_14 + 15 hold_6_15 + 16 hold_6_16
   + 17 hold_6_17 + 18 hold_6_18 + 60 pair_ok_6_12_for_nic_out_6_nic_in_12
   <= 60
 constr_pair_ok2_6_12: - start_6 + start_12 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11 - 12 hold_6_12
   - 13 hold_6_13 - 14 hold_6_14 - 15 hold_6_15 - 16 hold_6_16
   - 17 hold_6_17 - 18 hold_6_18 + 60 pair_ok_6_12_for_nic_out_6_nic_in_12
   <= 60
 constr_edge_switch_end_start_6_12_OR: switch_nic_out_6_nic_in_12
   - pair_ok_6_12_for_nic_out_6_nic_in_12 <= 0
 constr_pair_ok1_6_13: start_6 - start_13 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 12 hold_6_12
   + 13 hold_6_13 + 14 hold_6_14 + 15 hold_6_15 + 16 hold_6_16
   + 17 hold_6_17 + 18 hold_6_18 + 60 pair_ok_6_13_for_nic_out_6_nic_in_13
   <= 60
 constr_pair_ok2_6_13: - start_6 + start_13 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11 - 12 hold_6_12
   - 13 hold_6_13 - 14 hold_6_14 - 15 hold_6_15 - 16 hold_6_16
   - 17 hold_6_17 - 18 hold_6_18 + 60 pair_ok_6_13_for_nic_out_6_nic_in_13
   <= 60
 constr_edge_switch_end_start_6_13_OR: switch_nic_out_6_nic_in_13
   - pair_ok_6_13_for_nic_out_6_nic_in_13 <= 0
 constr_pair_ok1_6_14: start_6 - start_14 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 12 hold_6_12
   + 13 hold_6_13 + 14 hold_6_14 + 15 hold_6_15 + 16 hold_6_16
   + 17 hold_6_17 + 18 hold_6_18 + 60 pair_ok_6_14_for_nic_out_6_nic_in_14
   <= 60
 constr_pair_ok2_6_14: - start_6 + start_14 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11 - 12 hold_6_12
   - 13 hold_6_13 - 14 hold_6_14 - 15 hold_6_15 - 16 hold_6_16
   - 17 hold_6_17 - 18 hold_6_18 + 60 pair_ok_6_14_for_nic_out_6_nic_in_14
   <= 60
 constr_edge_switch_end_start_6_14_OR: switch_nic_out_6_nic_in_14
   - pair_ok_6_14_for_nic_out_6_nic_in_14 <= 0
 constr_pair_ok1_6_15: start_6 - start_15 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 12 hold_6_12
   + 13 hold_6_13 + 14 hold_6_14 + 15 hold_6_15 + 16 hold_6_16
   + 17 hold_6_17 + 18 hold_6_18 + 60 pair_ok_6_15_for_nic_out_6_nic_in_15
   <= 60
 constr_pair_ok2_6_15: - start_6 + start_15 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11 - 12 hold_6_12
   - 13 hold_6_13 - 14 hold_6_14 - 15 hold_6_15 - 16 hold_6_16
   - 17 hold_6_17 - 18 hold_6_18 + 60 pair_ok_6_15_for_nic_out_6_nic_in_15
   <= 60
 constr_edge_switch_end_start_6_15_OR: switch_nic_out_6_nic_in_15
   - pair_ok_6_15_for_nic_out_6_nic_in_15 <= 0
 constr_is_start_zero_7: start_7 + 60 is_start_zero_7_for_source_nic_in_7
   <= 60
 constr_edge_switch_start_zero_7_OR: switch_source_nic_in_7
   - is_start_zero_7_for_source_nic_in_7 <= 0
 constr_is_end_complete_7: start_7 + hold_7_1 + 2 hold_7_2 + 3 hold_7_3
   + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7 + 8 hold_7_8
   + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11 + 12 hold_7_12 + 13 hold_7_13
   + 14 hold_7_14 + 15 hold_7_15 + 16 hold_7_16 + 17 hold_7_17
   + 18 hold_7_18 - 60 is_end_complete_7 >= 0
 constr_edge_switch_end_max_7_OR: switch_nic_out_7_sink - is_end_complete_7
   <= 0
 constr_pair_ok1_7_0: - start_0 + start_7 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 8 hold_7_8 + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11 + 12 hold_7_12
   + 13 hold_7_13 + 14 hold_7_14 + 15 hold_7_15 + 16 hold_7_16
   + 17 hold_7_17 + 18 hold_7_18 + 60 pair_ok_7_0_for_nic_out_7_nic_in_0
   <= 60
 constr_pair_ok2_7_0: start_0 - start_7 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   - 8 hold_7_8 - 9 hold_7_9 - 10 hold_7_10 - 11 hold_7_11 - 12 hold_7_12
   - 13 hold_7_13 - 14 hold_7_14 - 15 hold_7_15 - 16 hold_7_16
   - 17 hold_7_17 - 18 hold_7_18 + 60 pair_ok_7_0_for_nic_out_7_nic_in_0
   <= 60
 constr_edge_switch_end_start_7_0_OR: switch_nic_out_7_nic_in_0
   - pair_ok_7_0_for_nic_out_7_nic_in_0 <= 0
 constr_pair_ok1_7_1: - start_1 + start_7 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 8 hold_7_8 + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11 + 12 hold_7_12
   + 13 hold_7_13 + 14 hold_7_14 + 15 hold_7_15 + 16 hold_7_16
   + 17 hold_7_17 + 18 hold_7_18 + 60 pair_ok_7_1_for_nic_out_7_nic_in_1
   <= 60
 constr_pair_ok2_7_1: start_1 - start_7 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   - 8 hold_7_8 - 9 hold_7_9 - 10 hold_7_10 - 11 hold_7_11 - 12 hold_7_12
   - 13 hold_7_13 - 14 hold_7_14 - 15 hold_7_15 - 16 hold_7_16
   - 17 hold_7_17 - 18 hold_7_18 + 60 pair_ok_7_1_for_nic_out_7_nic_in_1
   <= 60
 constr_edge_switch_end_start_7_1_OR: switch_nic_out_7_nic_in_1
   - pair_ok_7_1_for_nic_out_7_nic_in_1 <= 0
 constr_pair_ok1_7_2: - start_2 + start_7 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 8 hold_7_8 + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11 + 12 hold_7_12
   + 13 hold_7_13 + 14 hold_7_14 + 15 hold_7_15 + 16 hold_7_16
   + 17 hold_7_17 + 18 hold_7_18 + 60 pair_ok_7_2_for_nic_out_7_nic_in_2
   <= 60
 constr_pair_ok2_7_2: start_2 - start_7 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   - 8 hold_7_8 - 9 hold_7_9 - 10 hold_7_10 - 11 hold_7_11 - 12 hold_7_12
   - 13 hold_7_13 - 14 hold_7_14 - 15 hold_7_15 - 16 hold_7_16
   - 17 hold_7_17 - 18 hold_7_18 + 60 pair_ok_7_2_for_nic_out_7_nic_in_2
   <= 60
 constr_edge_switch_end_start_7_2_OR: switch_nic_out_7_nic_in_2
   - pair_ok_7_2_for_nic_out_7_nic_in_2 <= 0
 constr_pair_ok1_7_3: - start_3 + start_7 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 8 hold_7_8 + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11 + 12 hold_7_12
   + 13 hold_7_13 + 14 hold_7_14 + 15 hold_7_15 + 16 hold_7_16
   + 17 hold_7_17 + 18 hold_7_18 + 60 pair_ok_7_3_for_nic_out_7_nic_in_3
   <= 60
 constr_pair_ok2_7_3: start_3 - start_7 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   - 8 hold_7_8 - 9 hold_7_9 - 10 hold_7_10 - 11 hold_7_11 - 12 hold_7_12
   - 13 hold_7_13 - 14 hold_7_14 - 15 hold_7_15 - 16 hold_7_16
   - 17 hold_7_17 - 18 hold_7_18 + 60 pair_ok_7_3_for_nic_out_7_nic_in_3
   <= 60
 constr_edge_switch_end_start_7_3_OR: switch_nic_out_7_nic_in_3
   - pair_ok_7_3_for_nic_out_7_nic_in_3 <= 0
 constr_pair_ok1_7_4: - start_4 + start_7 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 8 hold_7_8 + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11 + 12 hold_7_12
   + 13 hold_7_13 + 14 hold_7_14 + 15 hold_7_15 + 16 hold_7_16
   + 17 hold_7_17 + 18 hold_7_18 + 60 pair_ok_7_4_for_nic_out_7_nic_in_4
   <= 60
 constr_pair_ok2_7_4: start_4 - start_7 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   - 8 hold_7_8 - 9 hold_7_9 - 10 hold_7_10 - 11 hold_7_11 - 12 hold_7_12
   - 13 hold_7_13 - 14 hold_7_14 - 15 hold_7_15 - 16 hold_7_16
   - 17 hold_7_17 - 18 hold_7_18 + 60 pair_ok_7_4_for_nic_out_7_nic_in_4
   <= 60
 constr_edge_switch_end_start_7_4_OR: switch_nic_out_7_nic_in_4
   - pair_ok_7_4_for_nic_out_7_nic_in_4 <= 0
 constr_pair_ok1_7_5: - start_5 + start_7 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 8 hold_7_8 + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11 + 12 hold_7_12
   + 13 hold_7_13 + 14 hold_7_14 + 15 hold_7_15 + 16 hold_7_16
   + 17 hold_7_17 + 18 hold_7_18 + 60 pair_ok_7_5_for_nic_out_7_nic_in_5
   <= 60
 constr_pair_ok2_7_5: start_5 - start_7 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   - 8 hold_7_8 - 9 hold_7_9 - 10 hold_7_10 - 11 hold_7_11 - 12 hold_7_12
   - 13 hold_7_13 - 14 hold_7_14 - 15 hold_7_15 - 16 hold_7_16
   - 17 hold_7_17 - 18 hold_7_18 + 60 pair_ok_7_5_for_nic_out_7_nic_in_5
   <= 60
 constr_edge_switch_end_start_7_5_OR: switch_nic_out_7_nic_in_5
   - pair_ok_7_5_for_nic_out_7_nic_in_5 <= 0
 constr_pair_ok1_7_6: - start_6 + start_7 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 8 hold_7_8 + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11 + 12 hold_7_12
   + 13 hold_7_13 + 14 hold_7_14 + 15 hold_7_15 + 16 hold_7_16
   + 17 hold_7_17 + 18 hold_7_18 + 60 pair_ok_7_6_for_nic_out_7_nic_in_6
   <= 60
 constr_pair_ok2_7_6: start_6 - start_7 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   - 8 hold_7_8 - 9 hold_7_9 - 10 hold_7_10 - 11 hold_7_11 - 12 hold_7_12
   - 13 hold_7_13 - 14 hold_7_14 - 15 hold_7_15 - 16 hold_7_16
   - 17 hold_7_17 - 18 hold_7_18 + 60 pair_ok_7_6_for_nic_out_7_nic_in_6
   <= 60
 constr_edge_switch_end_start_7_6_OR: switch_nic_out_7_nic_in_6
   - pair_ok_7_6_for_nic_out_7_nic_in_6 <= 0
 constr_pair_ok1_7_8: start_7 - start_8 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 8 hold_7_8 + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11 + 12 hold_7_12
   + 13 hold_7_13 + 14 hold_7_14 + 15 hold_7_15 + 16 hold_7_16
   + 17 hold_7_17 + 18 hold_7_18 + 60 pair_ok_7_8_for_nic_out_7_nic_in_8
   <= 60
 constr_pair_ok2_7_8: - start_7 + start_8 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   - 8 hold_7_8 - 9 hold_7_9 - 10 hold_7_10 - 11 hold_7_11 - 12 hold_7_12
   - 13 hold_7_13 - 14 hold_7_14 - 15 hold_7_15 - 16 hold_7_16
   - 17 hold_7_17 - 18 hold_7_18 + 60 pair_ok_7_8_for_nic_out_7_nic_in_8
   <= 60
 constr_edge_switch_end_start_7_8_OR: switch_nic_out_7_nic_in_8
   - pair_ok_7_8_for_nic_out_7_nic_in_8 <= 0
 constr_pair_ok1_7_9: start_7 - start_9 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 8 hold_7_8 + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11 + 12 hold_7_12
   + 13 hold_7_13 + 14 hold_7_14 + 15 hold_7_15 + 16 hold_7_16
   + 17 hold_7_17 + 18 hold_7_18 + 60 pair_ok_7_9_for_nic_out_7_nic_in_9
   <= 60
 constr_pair_ok2_7_9: - start_7 + start_9 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   - 8 hold_7_8 - 9 hold_7_9 - 10 hold_7_10 - 11 hold_7_11 - 12 hold_7_12
   - 13 hold_7_13 - 14 hold_7_14 - 15 hold_7_15 - 16 hold_7_16
   - 17 hold_7_17 - 18 hold_7_18 + 60 pair_ok_7_9_for_nic_out_7_nic_in_9
   <= 60
 constr_edge_switch_end_start_7_9_OR: switch_nic_out_7_nic_in_9
   - pair_ok_7_9_for_nic_out_7_nic_in_9 <= 0
 constr_pair_ok1_7_10: start_7 - start_10 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 8 hold_7_8 + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11 + 12 hold_7_12
   + 13 hold_7_13 + 14 hold_7_14 + 15 hold_7_15 + 16 hold_7_16
   + 17 hold_7_17 + 18 hold_7_18 + 60 pair_ok_7_10_for_nic_out_7_nic_in_10
   <= 60
 constr_pair_ok2_7_10: - start_7 + start_10 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   - 8 hold_7_8 - 9 hold_7_9 - 10 hold_7_10 - 11 hold_7_11 - 12 hold_7_12
   - 13 hold_7_13 - 14 hold_7_14 - 15 hold_7_15 - 16 hold_7_16
   - 17 hold_7_17 - 18 hold_7_18 + 60 pair_ok_7_10_for_nic_out_7_nic_in_10
   <= 60
 constr_edge_switch_end_start_7_10_OR: switch_nic_out_7_nic_in_10
   - pair_ok_7_10_for_nic_out_7_nic_in_10 <= 0
 constr_pair_ok1_7_11: start_7 - start_11 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 8 hold_7_8 + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11 + 12 hold_7_12
   + 13 hold_7_13 + 14 hold_7_14 + 15 hold_7_15 + 16 hold_7_16
   + 17 hold_7_17 + 18 hold_7_18 + 60 pair_ok_7_11_for_nic_out_7_nic_in_11
   <= 60
 constr_pair_ok2_7_11: - start_7 + start_11 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   - 8 hold_7_8 - 9 hold_7_9 - 10 hold_7_10 - 11 hold_7_11 - 12 hold_7_12
   - 13 hold_7_13 - 14 hold_7_14 - 15 hold_7_15 - 16 hold_7_16
   - 17 hold_7_17 - 18 hold_7_18 + 60 pair_ok_7_11_for_nic_out_7_nic_in_11
   <= 60
 constr_edge_switch_end_start_7_11_OR: switch_nic_out_7_nic_in_11
   - pair_ok_7_11_for_nic_out_7_nic_in_11 <= 0
 constr_pair_ok1_7_12: start_7 - start_12 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 8 hold_7_8 + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11 + 12 hold_7_12
   + 13 hold_7_13 + 14 hold_7_14 + 15 hold_7_15 + 16 hold_7_16
   + 17 hold_7_17 + 18 hold_7_18 + 60 pair_ok_7_12_for_nic_out_7_nic_in_12
   <= 60
 constr_pair_ok2_7_12: - start_7 + start_12 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   - 8 hold_7_8 - 9 hold_7_9 - 10 hold_7_10 - 11 hold_7_11 - 12 hold_7_12
   - 13 hold_7_13 - 14 hold_7_14 - 15 hold_7_15 - 16 hold_7_16
   - 17 hold_7_17 - 18 hold_7_18 + 60 pair_ok_7_12_for_nic_out_7_nic_in_12
   <= 60
 constr_edge_switch_end_start_7_12_OR: switch_nic_out_7_nic_in_12
   - pair_ok_7_12_for_nic_out_7_nic_in_12 <= 0
 constr_pair_ok1_7_13: start_7 - start_13 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 8 hold_7_8 + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11 + 12 hold_7_12
   + 13 hold_7_13 + 14 hold_7_14 + 15 hold_7_15 + 16 hold_7_16
   + 17 hold_7_17 + 18 hold_7_18 + 60 pair_ok_7_13_for_nic_out_7_nic_in_13
   <= 60
 constr_pair_ok2_7_13: - start_7 + start_13 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   - 8 hold_7_8 - 9 hold_7_9 - 10 hold_7_10 - 11 hold_7_11 - 12 hold_7_12
   - 13 hold_7_13 - 14 hold_7_14 - 15 hold_7_15 - 16 hold_7_16
   - 17 hold_7_17 - 18 hold_7_18 + 60 pair_ok_7_13_for_nic_out_7_nic_in_13
   <= 60
 constr_edge_switch_end_start_7_13_OR: switch_nic_out_7_nic_in_13
   - pair_ok_7_13_for_nic_out_7_nic_in_13 <= 0
 constr_pair_ok1_7_14: start_7 - start_14 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 8 hold_7_8 + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11 + 12 hold_7_12
   + 13 hold_7_13 + 14 hold_7_14 + 15 hold_7_15 + 16 hold_7_16
   + 17 hold_7_17 + 18 hold_7_18 + 60 pair_ok_7_14_for_nic_out_7_nic_in_14
   <= 60
 constr_pair_ok2_7_14: - start_7 + start_14 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   - 8 hold_7_8 - 9 hold_7_9 - 10 hold_7_10 - 11 hold_7_11 - 12 hold_7_12
   - 13 hold_7_13 - 14 hold_7_14 - 15 hold_7_15 - 16 hold_7_16
   - 17 hold_7_17 - 18 hold_7_18 + 60 pair_ok_7_14_for_nic_out_7_nic_in_14
   <= 60
 constr_edge_switch_end_start_7_14_OR: switch_nic_out_7_nic_in_14
   - pair_ok_7_14_for_nic_out_7_nic_in_14 <= 0
 constr_pair_ok1_7_15: start_7 - start_15 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 8 hold_7_8 + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11 + 12 hold_7_12
   + 13 hold_7_13 + 14 hold_7_14 + 15 hold_7_15 + 16 hold_7_16
   + 17 hold_7_17 + 18 hold_7_18 + 60 pair_ok_7_15_for_nic_out_7_nic_in_15
   <= 60
 constr_pair_ok2_7_15: - start_7 + start_15 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   - 8 hold_7_8 - 9 hold_7_9 - 10 hold_7_10 - 11 hold_7_11 - 12 hold_7_12
   - 13 hold_7_13 - 14 hold_7_14 - 15 hold_7_15 - 16 hold_7_16
   - 17 hold_7_17 - 18 hold_7_18 + 60 pair_ok_7_15_for_nic_out_7_nic_in_15
   <= 60
 constr_edge_switch_end_start_7_15_OR: switch_nic_out_7_nic_in_15
   - pair_ok_7_15_for_nic_out_7_nic_in_15 <= 0
 constr_is_start_zero_8: start_8 + 60 is_start_zero_8_for_source_nic_in_8
   <= 60
 constr_edge_switch_start_zero_8_OR: switch_source_nic_in_8
   - is_start_zero_8_for_source_nic_in_8 <= 0
 constr_is_end_complete_8: start_8 + hold_8_1 + 2 hold_8_2 + 3 hold_8_3
   + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7 + 8 hold_8_8
   + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 12 hold_8_12 + 13 hold_8_13
   + 14 hold_8_14 + 15 hold_8_15 + 16 hold_8_16 + 17 hold_8_17
   + 18 hold_8_18 - 60 is_end_complete_8 >= 0
 constr_edge_switch_end_max_8_OR: switch_nic_out_8_sink - is_end_complete_8
   <= 0
 constr_pair_ok1_8_0: - start_0 + start_8 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 12 hold_8_12
   + 13 hold_8_13 + 14 hold_8_14 + 15 hold_8_15 + 16 hold_8_16
   + 17 hold_8_17 + 18 hold_8_18 + 60 pair_ok_8_0_for_nic_out_8_nic_in_0
   <= 60
 constr_pair_ok2_8_0: start_0 - start_8 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11 - 12 hold_8_12
   - 13 hold_8_13 - 14 hold_8_14 - 15 hold_8_15 - 16 hold_8_16
   - 17 hold_8_17 - 18 hold_8_18 + 60 pair_ok_8_0_for_nic_out_8_nic_in_0
   <= 60
 constr_edge_switch_end_start_8_0_OR: switch_nic_out_8_nic_in_0
   - pair_ok_8_0_for_nic_out_8_nic_in_0 <= 0
 constr_pair_ok1_8_1: - start_1 + start_8 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 12 hold_8_12
   + 13 hold_8_13 + 14 hold_8_14 + 15 hold_8_15 + 16 hold_8_16
   + 17 hold_8_17 + 18 hold_8_18 + 60 pair_ok_8_1_for_nic_out_8_nic_in_1
   <= 60
 constr_pair_ok2_8_1: start_1 - start_8 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11 - 12 hold_8_12
   - 13 hold_8_13 - 14 hold_8_14 - 15 hold_8_15 - 16 hold_8_16
   - 17 hold_8_17 - 18 hold_8_18 + 60 pair_ok_8_1_for_nic_out_8_nic_in_1
   <= 60
 constr_edge_switch_end_start_8_1_OR: switch_nic_out_8_nic_in_1
   - pair_ok_8_1_for_nic_out_8_nic_in_1 <= 0
 constr_pair_ok1_8_2: - start_2 + start_8 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 12 hold_8_12
   + 13 hold_8_13 + 14 hold_8_14 + 15 hold_8_15 + 16 hold_8_16
   + 17 hold_8_17 + 18 hold_8_18 + 60 pair_ok_8_2_for_nic_out_8_nic_in_2
   <= 60
 constr_pair_ok2_8_2: start_2 - start_8 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11 - 12 hold_8_12
   - 13 hold_8_13 - 14 hold_8_14 - 15 hold_8_15 - 16 hold_8_16
   - 17 hold_8_17 - 18 hold_8_18 + 60 pair_ok_8_2_for_nic_out_8_nic_in_2
   <= 60
 constr_edge_switch_end_start_8_2_OR: switch_nic_out_8_nic_in_2
   - pair_ok_8_2_for_nic_out_8_nic_in_2 <= 0
 constr_pair_ok1_8_3: - start_3 + start_8 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 12 hold_8_12
   + 13 hold_8_13 + 14 hold_8_14 + 15 hold_8_15 + 16 hold_8_16
   + 17 hold_8_17 + 18 hold_8_18 + 60 pair_ok_8_3_for_nic_out_8_nic_in_3
   <= 60
 constr_pair_ok2_8_3: start_3 - start_8 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11 - 12 hold_8_12
   - 13 hold_8_13 - 14 hold_8_14 - 15 hold_8_15 - 16 hold_8_16
   - 17 hold_8_17 - 18 hold_8_18 + 60 pair_ok_8_3_for_nic_out_8_nic_in_3
   <= 60
 constr_edge_switch_end_start_8_3_OR: switch_nic_out_8_nic_in_3
   - pair_ok_8_3_for_nic_out_8_nic_in_3 <= 0
 constr_pair_ok1_8_4: - start_4 + start_8 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 12 hold_8_12
   + 13 hold_8_13 + 14 hold_8_14 + 15 hold_8_15 + 16 hold_8_16
   + 17 hold_8_17 + 18 hold_8_18 + 60 pair_ok_8_4_for_nic_out_8_nic_in_4
   <= 60
 constr_pair_ok2_8_4: start_4 - start_8 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11 - 12 hold_8_12
   - 13 hold_8_13 - 14 hold_8_14 - 15 hold_8_15 - 16 hold_8_16
   - 17 hold_8_17 - 18 hold_8_18 + 60 pair_ok_8_4_for_nic_out_8_nic_in_4
   <= 60
 constr_edge_switch_end_start_8_4_OR: switch_nic_out_8_nic_in_4
   - pair_ok_8_4_for_nic_out_8_nic_in_4 <= 0
 constr_pair_ok1_8_5: - start_5 + start_8 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 12 hold_8_12
   + 13 hold_8_13 + 14 hold_8_14 + 15 hold_8_15 + 16 hold_8_16
   + 17 hold_8_17 + 18 hold_8_18 + 60 pair_ok_8_5_for_nic_out_8_nic_in_5
   <= 60
 constr_pair_ok2_8_5: start_5 - start_8 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11 - 12 hold_8_12
   - 13 hold_8_13 - 14 hold_8_14 - 15 hold_8_15 - 16 hold_8_16
   - 17 hold_8_17 - 18 hold_8_18 + 60 pair_ok_8_5_for_nic_out_8_nic_in_5
   <= 60
 constr_edge_switch_end_start_8_5_OR: switch_nic_out_8_nic_in_5
   - pair_ok_8_5_for_nic_out_8_nic_in_5 <= 0
 constr_pair_ok1_8_6: - start_6 + start_8 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 12 hold_8_12
   + 13 hold_8_13 + 14 hold_8_14 + 15 hold_8_15 + 16 hold_8_16
   + 17 hold_8_17 + 18 hold_8_18 + 60 pair_ok_8_6_for_nic_out_8_nic_in_6
   <= 60
 constr_pair_ok2_8_6: start_6 - start_8 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11 - 12 hold_8_12
   - 13 hold_8_13 - 14 hold_8_14 - 15 hold_8_15 - 16 hold_8_16
   - 17 hold_8_17 - 18 hold_8_18 + 60 pair_ok_8_6_for_nic_out_8_nic_in_6
   <= 60
 constr_edge_switch_end_start_8_6_OR: switch_nic_out_8_nic_in_6
   - pair_ok_8_6_for_nic_out_8_nic_in_6 <= 0
 constr_pair_ok1_8_7: - start_7 + start_8 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 12 hold_8_12
   + 13 hold_8_13 + 14 hold_8_14 + 15 hold_8_15 + 16 hold_8_16
   + 17 hold_8_17 + 18 hold_8_18 + 60 pair_ok_8_7_for_nic_out_8_nic_in_7
   <= 60
 constr_pair_ok2_8_7: start_7 - start_8 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11 - 12 hold_8_12
   - 13 hold_8_13 - 14 hold_8_14 - 15 hold_8_15 - 16 hold_8_16
   - 17 hold_8_17 - 18 hold_8_18 + 60 pair_ok_8_7_for_nic_out_8_nic_in_7
   <= 60
 constr_edge_switch_end_start_8_7_OR: switch_nic_out_8_nic_in_7
   - pair_ok_8_7_for_nic_out_8_nic_in_7 <= 0
 constr_pair_ok1_8_9: start_8 - start_9 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 12 hold_8_12
   + 13 hold_8_13 + 14 hold_8_14 + 15 hold_8_15 + 16 hold_8_16
   + 17 hold_8_17 + 18 hold_8_18 + 60 pair_ok_8_9_for_nic_out_8_nic_in_9
   <= 60
 constr_pair_ok2_8_9: - start_8 + start_9 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11 - 12 hold_8_12
   - 13 hold_8_13 - 14 hold_8_14 - 15 hold_8_15 - 16 hold_8_16
   - 17 hold_8_17 - 18 hold_8_18 + 60 pair_ok_8_9_for_nic_out_8_nic_in_9
   <= 60
 constr_edge_switch_end_start_8_9_OR: switch_nic_out_8_nic_in_9
   - pair_ok_8_9_for_nic_out_8_nic_in_9 <= 0
 constr_pair_ok1_8_10: start_8 - start_10 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 12 hold_8_12
   + 13 hold_8_13 + 14 hold_8_14 + 15 hold_8_15 + 16 hold_8_16
   + 17 hold_8_17 + 18 hold_8_18 + 60 pair_ok_8_10_for_nic_out_8_nic_in_10
   <= 60
 constr_pair_ok2_8_10: - start_8 + start_10 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11 - 12 hold_8_12
   - 13 hold_8_13 - 14 hold_8_14 - 15 hold_8_15 - 16 hold_8_16
   - 17 hold_8_17 - 18 hold_8_18 + 60 pair_ok_8_10_for_nic_out_8_nic_in_10
   <= 60
 constr_edge_switch_end_start_8_10_OR: switch_nic_out_8_nic_in_10
   - pair_ok_8_10_for_nic_out_8_nic_in_10 <= 0
 constr_pair_ok1_8_11: start_8 - start_11 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 12 hold_8_12
   + 13 hold_8_13 + 14 hold_8_14 + 15 hold_8_15 + 16 hold_8_16
   + 17 hold_8_17 + 18 hold_8_18 + 60 pair_ok_8_11_for_nic_out_8_nic_in_11
   <= 60
 constr_pair_ok2_8_11: - start_8 + start_11 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11 - 12 hold_8_12
   - 13 hold_8_13 - 14 hold_8_14 - 15 hold_8_15 - 16 hold_8_16
   - 17 hold_8_17 - 18 hold_8_18 + 60 pair_ok_8_11_for_nic_out_8_nic_in_11
   <= 60
 constr_edge_switch_end_start_8_11_OR: switch_nic_out_8_nic_in_11
   - pair_ok_8_11_for_nic_out_8_nic_in_11 <= 0
 constr_pair_ok1_8_12: start_8 - start_12 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 12 hold_8_12
   + 13 hold_8_13 + 14 hold_8_14 + 15 hold_8_15 + 16 hold_8_16
   + 17 hold_8_17 + 18 hold_8_18 + 60 pair_ok_8_12_for_nic_out_8_nic_in_12
   <= 60
 constr_pair_ok2_8_12: - start_8 + start_12 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11 - 12 hold_8_12
   - 13 hold_8_13 - 14 hold_8_14 - 15 hold_8_15 - 16 hold_8_16
   - 17 hold_8_17 - 18 hold_8_18 + 60 pair_ok_8_12_for_nic_out_8_nic_in_12
   <= 60
 constr_edge_switch_end_start_8_12_OR: switch_nic_out_8_nic_in_12
   - pair_ok_8_12_for_nic_out_8_nic_in_12 <= 0
 constr_pair_ok1_8_13: start_8 - start_13 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 12 hold_8_12
   + 13 hold_8_13 + 14 hold_8_14 + 15 hold_8_15 + 16 hold_8_16
   + 17 hold_8_17 + 18 hold_8_18 + 60 pair_ok_8_13_for_nic_out_8_nic_in_13
   <= 60
 constr_pair_ok2_8_13: - start_8 + start_13 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11 - 12 hold_8_12
   - 13 hold_8_13 - 14 hold_8_14 - 15 hold_8_15 - 16 hold_8_16
   - 17 hold_8_17 - 18 hold_8_18 + 60 pair_ok_8_13_for_nic_out_8_nic_in_13
   <= 60
 constr_edge_switch_end_start_8_13_OR: switch_nic_out_8_nic_in_13
   - pair_ok_8_13_for_nic_out_8_nic_in_13 <= 0
 constr_pair_ok1_8_14: start_8 - start_14 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 12 hold_8_12
   + 13 hold_8_13 + 14 hold_8_14 + 15 hold_8_15 + 16 hold_8_16
   + 17 hold_8_17 + 18 hold_8_18 + 60 pair_ok_8_14_for_nic_out_8_nic_in_14
   <= 60
 constr_pair_ok2_8_14: - start_8 + start_14 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11 - 12 hold_8_12
   - 13 hold_8_13 - 14 hold_8_14 - 15 hold_8_15 - 16 hold_8_16
   - 17 hold_8_17 - 18 hold_8_18 + 60 pair_ok_8_14_for_nic_out_8_nic_in_14
   <= 60
 constr_edge_switch_end_start_8_14_OR: switch_nic_out_8_nic_in_14
   - pair_ok_8_14_for_nic_out_8_nic_in_14 <= 0
 constr_pair_ok1_8_15: start_8 - start_15 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 12 hold_8_12
   + 13 hold_8_13 + 14 hold_8_14 + 15 hold_8_15 + 16 hold_8_16
   + 17 hold_8_17 + 18 hold_8_18 + 60 pair_ok_8_15_for_nic_out_8_nic_in_15
   <= 60
 constr_pair_ok2_8_15: - start_8 + start_15 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11 - 12 hold_8_12
   - 13 hold_8_13 - 14 hold_8_14 - 15 hold_8_15 - 16 hold_8_16
   - 17 hold_8_17 - 18 hold_8_18 + 60 pair_ok_8_15_for_nic_out_8_nic_in_15
   <= 60
 constr_edge_switch_end_start_8_15_OR: switch_nic_out_8_nic_in_15
   - pair_ok_8_15_for_nic_out_8_nic_in_15 <= 0
 constr_is_start_zero_9: start_9 + 60 is_start_zero_9_for_source_nic_in_9
   <= 60
 constr_edge_switch_start_zero_9_OR: switch_source_nic_in_9
   - is_start_zero_9_for_source_nic_in_9 <= 0
 constr_is_end_complete_9: start_9 + hold_9_1 + 2 hold_9_2 + 3 hold_9_3
   + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7 + 8 hold_9_8
   + 9 hold_9_9 + 10 hold_9_10 + 11 hold_9_11 + 12 hold_9_12 + 13 hold_9_13
   + 14 hold_9_14 + 15 hold_9_15 + 16 hold_9_16 + 17 hold_9_17
   + 18 hold_9_18 - 60 is_end_complete_9 >= 0
 constr_edge_switch_end_max_9_OR: switch_nic_out_9_sink - is_end_complete_9
   <= 0
 constr_pair_ok1_9_0: - start_0 + start_9 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 8 hold_9_8 + 9 hold_9_9 + 10 hold_9_10 + 11 hold_9_11 + 12 hold_9_12
   + 13 hold_9_13 + 14 hold_9_14 + 15 hold_9_15 + 16 hold_9_16
   + 17 hold_9_17 + 18 hold_9_18 + 60 pair_ok_9_0_for_nic_out_9_nic_in_0
   <= 60
 constr_pair_ok2_9_0: start_0 - start_9 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   - 8 hold_9_8 - 9 hold_9_9 - 10 hold_9_10 - 11 hold_9_11 - 12 hold_9_12
   - 13 hold_9_13 - 14 hold_9_14 - 15 hold_9_15 - 16 hold_9_16
   - 17 hold_9_17 - 18 hold_9_18 + 60 pair_ok_9_0_for_nic_out_9_nic_in_0
   <= 60
 constr_edge_switch_end_start_9_0_OR: switch_nic_out_9_nic_in_0
   - pair_ok_9_0_for_nic_out_9_nic_in_0 <= 0
 constr_pair_ok1_9_1: - start_1 + start_9 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 8 hold_9_8 + 9 hold_9_9 + 10 hold_9_10 + 11 hold_9_11 + 12 hold_9_12
   + 13 hold_9_13 + 14 hold_9_14 + 15 hold_9_15 + 16 hold_9_16
   + 17 hold_9_17 + 18 hold_9_18 + 60 pair_ok_9_1_for_nic_out_9_nic_in_1
   <= 60
 constr_pair_ok2_9_1: start_1 - start_9 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   - 8 hold_9_8 - 9 hold_9_9 - 10 hold_9_10 - 11 hold_9_11 - 12 hold_9_12
   - 13 hold_9_13 - 14 hold_9_14 - 15 hold_9_15 - 16 hold_9_16
   - 17 hold_9_17 - 18 hold_9_18 + 60 pair_ok_9_1_for_nic_out_9_nic_in_1
   <= 60
 constr_edge_switch_end_start_9_1_OR: switch_nic_out_9_nic_in_1
   - pair_ok_9_1_for_nic_out_9_nic_in_1 <= 0
 constr_pair_ok1_9_2: - start_2 + start_9 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 8 hold_9_8 + 9 hold_9_9 + 10 hold_9_10 + 11 hold_9_11 + 12 hold_9_12
   + 13 hold_9_13 + 14 hold_9_14 + 15 hold_9_15 + 16 hold_9_16
   + 17 hold_9_17 + 18 hold_9_18 + 60 pair_ok_9_2_for_nic_out_9_nic_in_2
   <= 60
 constr_pair_ok2_9_2: start_2 - start_9 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   - 8 hold_9_8 - 9 hold_9_9 - 10 hold_9_10 - 11 hold_9_11 - 12 hold_9_12
   - 13 hold_9_13 - 14 hold_9_14 - 15 hold_9_15 - 16 hold_9_16
   - 17 hold_9_17 - 18 hold_9_18 + 60 pair_ok_9_2_for_nic_out_9_nic_in_2
   <= 60
 constr_edge_switch_end_start_9_2_OR: switch_nic_out_9_nic_in_2
   - pair_ok_9_2_for_nic_out_9_nic_in_2 <= 0
 constr_pair_ok1_9_3: - start_3 + start_9 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 8 hold_9_8 + 9 hold_9_9 + 10 hold_9_10 + 11 hold_9_11 + 12 hold_9_12
   + 13 hold_9_13 + 14 hold_9_14 + 15 hold_9_15 + 16 hold_9_16
   + 17 hold_9_17 + 18 hold_9_18 + 60 pair_ok_9_3_for_nic_out_9_nic_in_3
   <= 60
 constr_pair_ok2_9_3: start_3 - start_9 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   - 8 hold_9_8 - 9 hold_9_9 - 10 hold_9_10 - 11 hold_9_11 - 12 hold_9_12
   - 13 hold_9_13 - 14 hold_9_14 - 15 hold_9_15 - 16 hold_9_16
   - 17 hold_9_17 - 18 hold_9_18 + 60 pair_ok_9_3_for_nic_out_9_nic_in_3
   <= 60
 constr_edge_switch_end_start_9_3_OR: switch_nic_out_9_nic_in_3
   - pair_ok_9_3_for_nic_out_9_nic_in_3 <= 0
 constr_pair_ok1_9_4: - start_4 + start_9 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 8 hold_9_8 + 9 hold_9_9 + 10 hold_9_10 + 11 hold_9_11 + 12 hold_9_12
   + 13 hold_9_13 + 14 hold_9_14 + 15 hold_9_15 + 16 hold_9_16
   + 17 hold_9_17 + 18 hold_9_18 + 60 pair_ok_9_4_for_nic_out_9_nic_in_4
   <= 60
 constr_pair_ok2_9_4: start_4 - start_9 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   - 8 hold_9_8 - 9 hold_9_9 - 10 hold_9_10 - 11 hold_9_11 - 12 hold_9_12
   - 13 hold_9_13 - 14 hold_9_14 - 15 hold_9_15 - 16 hold_9_16
   - 17 hold_9_17 - 18 hold_9_18 + 60 pair_ok_9_4_for_nic_out_9_nic_in_4
   <= 60
 constr_edge_switch_end_start_9_4_OR: switch_nic_out_9_nic_in_4
   - pair_ok_9_4_for_nic_out_9_nic_in_4 <= 0
 constr_pair_ok1_9_5: - start_5 + start_9 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 8 hold_9_8 + 9 hold_9_9 + 10 hold_9_10 + 11 hold_9_11 + 12 hold_9_12
   + 13 hold_9_13 + 14 hold_9_14 + 15 hold_9_15 + 16 hold_9_16
   + 17 hold_9_17 + 18 hold_9_18 + 60 pair_ok_9_5_for_nic_out_9_nic_in_5
   <= 60
 constr_pair_ok2_9_5: start_5 - start_9 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   - 8 hold_9_8 - 9 hold_9_9 - 10 hold_9_10 - 11 hold_9_11 - 12 hold_9_12
   - 13 hold_9_13 - 14 hold_9_14 - 15 hold_9_15 - 16 hold_9_16
   - 17 hold_9_17 - 18 hold_9_18 + 60 pair_ok_9_5_for_nic_out_9_nic_in_5
   <= 60
 constr_edge_switch_end_start_9_5_OR: switch_nic_out_9_nic_in_5
   - pair_ok_9_5_for_nic_out_9_nic_in_5 <= 0
 constr_pair_ok1_9_6: - start_6 + start_9 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 8 hold_9_8 + 9 hold_9_9 + 10 hold_9_10 + 11 hold_9_11 + 12 hold_9_12
   + 13 hold_9_13 + 14 hold_9_14 + 15 hold_9_15 + 16 hold_9_16
   + 17 hold_9_17 + 18 hold_9_18 + 60 pair_ok_9_6_for_nic_out_9_nic_in_6
   <= 60
 constr_pair_ok2_9_6: start_6 - start_9 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   - 8 hold_9_8 - 9 hold_9_9 - 10 hold_9_10 - 11 hold_9_11 - 12 hold_9_12
   - 13 hold_9_13 - 14 hold_9_14 - 15 hold_9_15 - 16 hold_9_16
   - 17 hold_9_17 - 18 hold_9_18 + 60 pair_ok_9_6_for_nic_out_9_nic_in_6
   <= 60
 constr_edge_switch_end_start_9_6_OR: switch_nic_out_9_nic_in_6
   - pair_ok_9_6_for_nic_out_9_nic_in_6 <= 0
 constr_pair_ok1_9_7: - start_7 + start_9 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 8 hold_9_8 + 9 hold_9_9 + 10 hold_9_10 + 11 hold_9_11 + 12 hold_9_12
   + 13 hold_9_13 + 14 hold_9_14 + 15 hold_9_15 + 16 hold_9_16
   + 17 hold_9_17 + 18 hold_9_18 + 60 pair_ok_9_7_for_nic_out_9_nic_in_7
   <= 60
 constr_pair_ok2_9_7: start_7 - start_9 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   - 8 hold_9_8 - 9 hold_9_9 - 10 hold_9_10 - 11 hold_9_11 - 12 hold_9_12
   - 13 hold_9_13 - 14 hold_9_14 - 15 hold_9_15 - 16 hold_9_16
   - 17 hold_9_17 - 18 hold_9_18 + 60 pair_ok_9_7_for_nic_out_9_nic_in_7
   <= 60
 constr_edge_switch_end_start_9_7_OR: switch_nic_out_9_nic_in_7
   - pair_ok_9_7_for_nic_out_9_nic_in_7 <= 0
 constr_pair_ok1_9_8: - start_8 + start_9 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 8 hold_9_8 + 9 hold_9_9 + 10 hold_9_10 + 11 hold_9_11 + 12 hold_9_12
   + 13 hold_9_13 + 14 hold_9_14 + 15 hold_9_15 + 16 hold_9_16
   + 17 hold_9_17 + 18 hold_9_18 + 60 pair_ok_9_8_for_nic_out_9_nic_in_8
   <= 60
 constr_pair_ok2_9_8: start_8 - start_9 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   - 8 hold_9_8 - 9 hold_9_9 - 10 hold_9_10 - 11 hold_9_11 - 12 hold_9_12
   - 13 hold_9_13 - 14 hold_9_14 - 15 hold_9_15 - 16 hold_9_16
   - 17 hold_9_17 - 18 hold_9_18 + 60 pair_ok_9_8_for_nic_out_9_nic_in_8
   <= 60
 constr_edge_switch_end_start_9_8_OR: switch_nic_out_9_nic_in_8
   - pair_ok_9_8_for_nic_out_9_nic_in_8 <= 0
 constr_pair_ok1_9_10: start_9 - start_10 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 8 hold_9_8 + 9 hold_9_9 + 10 hold_9_10 + 11 hold_9_11 + 12 hold_9_12
   + 13 hold_9_13 + 14 hold_9_14 + 15 hold_9_15 + 16 hold_9_16
   + 17 hold_9_17 + 18 hold_9_18 + 60 pair_ok_9_10_for_nic_out_9_nic_in_10
   <= 60
 constr_pair_ok2_9_10: - start_9 + start_10 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   - 8 hold_9_8 - 9 hold_9_9 - 10 hold_9_10 - 11 hold_9_11 - 12 hold_9_12
   - 13 hold_9_13 - 14 hold_9_14 - 15 hold_9_15 - 16 hold_9_16
   - 17 hold_9_17 - 18 hold_9_18 + 60 pair_ok_9_10_for_nic_out_9_nic_in_10
   <= 60
 constr_edge_switch_end_start_9_10_OR: switch_nic_out_9_nic_in_10
   - pair_ok_9_10_for_nic_out_9_nic_in_10 <= 0
 constr_pair_ok1_9_11: start_9 - start_11 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 8 hold_9_8 + 9 hold_9_9 + 10 hold_9_10 + 11 hold_9_11 + 12 hold_9_12
   + 13 hold_9_13 + 14 hold_9_14 + 15 hold_9_15 + 16 hold_9_16
   + 17 hold_9_17 + 18 hold_9_18 + 60 pair_ok_9_11_for_nic_out_9_nic_in_11
   <= 60
 constr_pair_ok2_9_11: - start_9 + start_11 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   - 8 hold_9_8 - 9 hold_9_9 - 10 hold_9_10 - 11 hold_9_11 - 12 hold_9_12
   - 13 hold_9_13 - 14 hold_9_14 - 15 hold_9_15 - 16 hold_9_16
   - 17 hold_9_17 - 18 hold_9_18 + 60 pair_ok_9_11_for_nic_out_9_nic_in_11
   <= 60
 constr_edge_switch_end_start_9_11_OR: switch_nic_out_9_nic_in_11
   - pair_ok_9_11_for_nic_out_9_nic_in_11 <= 0
 constr_pair_ok1_9_12: start_9 - start_12 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 8 hold_9_8 + 9 hold_9_9 + 10 hold_9_10 + 11 hold_9_11 + 12 hold_9_12
   + 13 hold_9_13 + 14 hold_9_14 + 15 hold_9_15 + 16 hold_9_16
   + 17 hold_9_17 + 18 hold_9_18 + 60 pair_ok_9_12_for_nic_out_9_nic_in_12
   <= 60
 constr_pair_ok2_9_12: - start_9 + start_12 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   - 8 hold_9_8 - 9 hold_9_9 - 10 hold_9_10 - 11 hold_9_11 - 12 hold_9_12
   - 13 hold_9_13 - 14 hold_9_14 - 15 hold_9_15 - 16 hold_9_16
   - 17 hold_9_17 - 18 hold_9_18 + 60 pair_ok_9_12_for_nic_out_9_nic_in_12
   <= 60
 constr_edge_switch_end_start_9_12_OR: switch_nic_out_9_nic_in_12
   - pair_ok_9_12_for_nic_out_9_nic_in_12 <= 0
 constr_pair_ok1_9_13: start_9 - start_13 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 8 hold_9_8 + 9 hold_9_9 + 10 hold_9_10 + 11 hold_9_11 + 12 hold_9_12
   + 13 hold_9_13 + 14 hold_9_14 + 15 hold_9_15 + 16 hold_9_16
   + 17 hold_9_17 + 18 hold_9_18 + 60 pair_ok_9_13_for_nic_out_9_nic_in_13
   <= 60
 constr_pair_ok2_9_13: - start_9 + start_13 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   - 8 hold_9_8 - 9 hold_9_9 - 10 hold_9_10 - 11 hold_9_11 - 12 hold_9_12
   - 13 hold_9_13 - 14 hold_9_14 - 15 hold_9_15 - 16 hold_9_16
   - 17 hold_9_17 - 18 hold_9_18 + 60 pair_ok_9_13_for_nic_out_9_nic_in_13
   <= 60
 constr_edge_switch_end_start_9_13_OR: switch_nic_out_9_nic_in_13
   - pair_ok_9_13_for_nic_out_9_nic_in_13 <= 0
 constr_pair_ok1_9_14: start_9 - start_14 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 8 hold_9_8 + 9 hold_9_9 + 10 hold_9_10 + 11 hold_9_11 + 12 hold_9_12
   + 13 hold_9_13 + 14 hold_9_14 + 15 hold_9_15 + 16 hold_9_16
   + 17 hold_9_17 + 18 hold_9_18 + 60 pair_ok_9_14_for_nic_out_9_nic_in_14
   <= 60
 constr_pair_ok2_9_14: - start_9 + start_14 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   - 8 hold_9_8 - 9 hold_9_9 - 10 hold_9_10 - 11 hold_9_11 - 12 hold_9_12
   - 13 hold_9_13 - 14 hold_9_14 - 15 hold_9_15 - 16 hold_9_16
   - 17 hold_9_17 - 18 hold_9_18 + 60 pair_ok_9_14_for_nic_out_9_nic_in_14
   <= 60
 constr_edge_switch_end_start_9_14_OR: switch_nic_out_9_nic_in_14
   - pair_ok_9_14_for_nic_out_9_nic_in_14 <= 0
 constr_pair_ok1_9_15: start_9 - start_15 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 8 hold_9_8 + 9 hold_9_9 + 10 hold_9_10 + 11 hold_9_11 + 12 hold_9_12
   + 13 hold_9_13 + 14 hold_9_14 + 15 hold_9_15 + 16 hold_9_16
   + 17 hold_9_17 + 18 hold_9_18 + 60 pair_ok_9_15_for_nic_out_9_nic_in_15
   <= 60
 constr_pair_ok2_9_15: - start_9 + start_15 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   - 8 hold_9_8 - 9 hold_9_9 - 10 hold_9_10 - 11 hold_9_11 - 12 hold_9_12
   - 13 hold_9_13 - 14 hold_9_14 - 15 hold_9_15 - 16 hold_9_16
   - 17 hold_9_17 - 18 hold_9_18 + 60 pair_ok_9_15_for_nic_out_9_nic_in_15
   <= 60
 constr_edge_switch_end_start_9_15_OR: switch_nic_out_9_nic_in_15
   - pair_ok_9_15_for_nic_out_9_nic_in_15 <= 0
 constr_is_start_zero_10: start_10
   + 60 is_start_zero_10_for_source_nic_in_10 <= 60
 constr_edge_switch_start_zero_10_OR: switch_source_nic_in_10
   - is_start_zero_10_for_source_nic_in_10 <= 0
 constr_is_end_complete_10: start_10 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 12 hold_10_12 + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15
   + 16 hold_10_16 + 17 hold_10_17 + 18 hold_10_18 - 60 is_end_complete_10
   >= 0
 constr_edge_switch_end_max_10_OR: switch_nic_out_10_sink
   - is_end_complete_10 <= 0
 constr_pair_ok1_10_0: - start_0 + start_10 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 12 hold_10_12 + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15
   + 16 hold_10_16 + 17 hold_10_17 + 18 hold_10_18
   + 60 pair_ok_10_0_for_nic_out_10_nic_in_0 <= 60
 constr_pair_ok2_10_0: start_0 - start_10 - hold_10_1 - 2 hold_10_2
   - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6 - 7 hold_10_7
   - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10 - 11 hold_10_11
   - 12 hold_10_12 - 13 hold_10_13 - 14 hold_10_14 - 15 hold_10_15
   - 16 hold_10_16 - 17 hold_10_17 - 18 hold_10_18
   + 60 pair_ok_10_0_for_nic_out_10_nic_in_0 <= 60
 constr_edge_switch_end_start_10_0_OR: switch_nic_out_10_nic_in_0
   - pair_ok_10_0_for_nic_out_10_nic_in_0 <= 0
 constr_pair_ok1_10_1: - start_1 + start_10 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 12 hold_10_12 + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15
   + 16 hold_10_16 + 17 hold_10_17 + 18 hold_10_18
   + 60 pair_ok_10_1_for_nic_out_10_nic_in_1 <= 60
 constr_pair_ok2_10_1: start_1 - start_10 - hold_10_1 - 2 hold_10_2
   - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6 - 7 hold_10_7
   - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10 - 11 hold_10_11
   - 12 hold_10_12 - 13 hold_10_13 - 14 hold_10_14 - 15 hold_10_15
   - 16 hold_10_16 - 17 hold_10_17 - 18 hold_10_18
   + 60 pair_ok_10_1_for_nic_out_10_nic_in_1 <= 60
 constr_edge_switch_end_start_10_1_OR: switch_nic_out_10_nic_in_1
   - pair_ok_10_1_for_nic_out_10_nic_in_1 <= 0
 constr_pair_ok1_10_2: - start_2 + start_10 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 12 hold_10_12 + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15
   + 16 hold_10_16 + 17 hold_10_17 + 18 hold_10_18
   + 60 pair_ok_10_2_for_nic_out_10_nic_in_2 <= 60
 constr_pair_ok2_10_2: start_2 - start_10 - hold_10_1 - 2 hold_10_2
   - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6 - 7 hold_10_7
   - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10 - 11 hold_10_11
   - 12 hold_10_12 - 13 hold_10_13 - 14 hold_10_14 - 15 hold_10_15
   - 16 hold_10_16 - 17 hold_10_17 - 18 hold_10_18
   + 60 pair_ok_10_2_for_nic_out_10_nic_in_2 <= 60
 constr_edge_switch_end_start_10_2_OR: switch_nic_out_10_nic_in_2
   - pair_ok_10_2_for_nic_out_10_nic_in_2 <= 0
 constr_pair_ok1_10_3: - start_3 + start_10 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 12 hold_10_12 + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15
   + 16 hold_10_16 + 17 hold_10_17 + 18 hold_10_18
   + 60 pair_ok_10_3_for_nic_out_10_nic_in_3 <= 60
 constr_pair_ok2_10_3: start_3 - start_10 - hold_10_1 - 2 hold_10_2
   - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6 - 7 hold_10_7
   - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10 - 11 hold_10_11
   - 12 hold_10_12 - 13 hold_10_13 - 14 hold_10_14 - 15 hold_10_15
   - 16 hold_10_16 - 17 hold_10_17 - 18 hold_10_18
   + 60 pair_ok_10_3_for_nic_out_10_nic_in_3 <= 60
 constr_edge_switch_end_start_10_3_OR: switch_nic_out_10_nic_in_3
   - pair_ok_10_3_for_nic_out_10_nic_in_3 <= 0
 constr_pair_ok1_10_4: - start_4 + start_10 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 12 hold_10_12 + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15
   + 16 hold_10_16 + 17 hold_10_17 + 18 hold_10_18
   + 60 pair_ok_10_4_for_nic_out_10_nic_in_4 <= 60
 constr_pair_ok2_10_4: start_4 - start_10 - hold_10_1 - 2 hold_10_2
   - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6 - 7 hold_10_7
   - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10 - 11 hold_10_11
   - 12 hold_10_12 - 13 hold_10_13 - 14 hold_10_14 - 15 hold_10_15
   - 16 hold_10_16 - 17 hold_10_17 - 18 hold_10_18
   + 60 pair_ok_10_4_for_nic_out_10_nic_in_4 <= 60
 constr_edge_switch_end_start_10_4_OR: switch_nic_out_10_nic_in_4
   - pair_ok_10_4_for_nic_out_10_nic_in_4 <= 0
 constr_pair_ok1_10_5: - start_5 + start_10 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 12 hold_10_12 + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15
   + 16 hold_10_16 + 17 hold_10_17 + 18 hold_10_18
   + 60 pair_ok_10_5_for_nic_out_10_nic_in_5 <= 60
 constr_pair_ok2_10_5: start_5 - start_10 - hold_10_1 - 2 hold_10_2
   - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6 - 7 hold_10_7
   - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10 - 11 hold_10_11
   - 12 hold_10_12 - 13 hold_10_13 - 14 hold_10_14 - 15 hold_10_15
   - 16 hold_10_16 - 17 hold_10_17 - 18 hold_10_18
   + 60 pair_ok_10_5_for_nic_out_10_nic_in_5 <= 60
 constr_edge_switch_end_start_10_5_OR: switch_nic_out_10_nic_in_5
   - pair_ok_10_5_for_nic_out_10_nic_in_5 <= 0
 constr_pair_ok1_10_6: - start_6 + start_10 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 12 hold_10_12 + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15
   + 16 hold_10_16 + 17 hold_10_17 + 18 hold_10_18
   + 60 pair_ok_10_6_for_nic_out_10_nic_in_6 <= 60
 constr_pair_ok2_10_6: start_6 - start_10 - hold_10_1 - 2 hold_10_2
   - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6 - 7 hold_10_7
   - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10 - 11 hold_10_11
   - 12 hold_10_12 - 13 hold_10_13 - 14 hold_10_14 - 15 hold_10_15
   - 16 hold_10_16 - 17 hold_10_17 - 18 hold_10_18
   + 60 pair_ok_10_6_for_nic_out_10_nic_in_6 <= 60
 constr_edge_switch_end_start_10_6_OR: switch_nic_out_10_nic_in_6
   - pair_ok_10_6_for_nic_out_10_nic_in_6 <= 0
 constr_pair_ok1_10_7: - start_7 + start_10 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 12 hold_10_12 + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15
   + 16 hold_10_16 + 17 hold_10_17 + 18 hold_10_18
   + 60 pair_ok_10_7_for_nic_out_10_nic_in_7 <= 60
 constr_pair_ok2_10_7: start_7 - start_10 - hold_10_1 - 2 hold_10_2
   - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6 - 7 hold_10_7
   - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10 - 11 hold_10_11
   - 12 hold_10_12 - 13 hold_10_13 - 14 hold_10_14 - 15 hold_10_15
   - 16 hold_10_16 - 17 hold_10_17 - 18 hold_10_18
   + 60 pair_ok_10_7_for_nic_out_10_nic_in_7 <= 60
 constr_edge_switch_end_start_10_7_OR: switch_nic_out_10_nic_in_7
   - pair_ok_10_7_for_nic_out_10_nic_in_7 <= 0
 constr_pair_ok1_10_8: - start_8 + start_10 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 12 hold_10_12 + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15
   + 16 hold_10_16 + 17 hold_10_17 + 18 hold_10_18
   + 60 pair_ok_10_8_for_nic_out_10_nic_in_8 <= 60
 constr_pair_ok2_10_8: start_8 - start_10 - hold_10_1 - 2 hold_10_2
   - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6 - 7 hold_10_7
   - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10 - 11 hold_10_11
   - 12 hold_10_12 - 13 hold_10_13 - 14 hold_10_14 - 15 hold_10_15
   - 16 hold_10_16 - 17 hold_10_17 - 18 hold_10_18
   + 60 pair_ok_10_8_for_nic_out_10_nic_in_8 <= 60
 constr_edge_switch_end_start_10_8_OR: switch_nic_out_10_nic_in_8
   - pair_ok_10_8_for_nic_out_10_nic_in_8 <= 0
 constr_pair_ok1_10_9: - start_9 + start_10 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 12 hold_10_12 + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15
   + 16 hold_10_16 + 17 hold_10_17 + 18 hold_10_18
   + 60 pair_ok_10_9_for_nic_out_10_nic_in_9 <= 60
 constr_pair_ok2_10_9: start_9 - start_10 - hold_10_1 - 2 hold_10_2
   - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6 - 7 hold_10_7
   - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10 - 11 hold_10_11
   - 12 hold_10_12 - 13 hold_10_13 - 14 hold_10_14 - 15 hold_10_15
   - 16 hold_10_16 - 17 hold_10_17 - 18 hold_10_18
   + 60 pair_ok_10_9_for_nic_out_10_nic_in_9 <= 60
 constr_edge_switch_end_start_10_9_OR: switch_nic_out_10_nic_in_9
   - pair_ok_10_9_for_nic_out_10_nic_in_9 <= 0
 constr_pair_ok1_10_11: start_10 - start_11 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 12 hold_10_12 + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15
   + 16 hold_10_16 + 17 hold_10_17 + 18 hold_10_18
   + 60 pair_ok_10_11_for_nic_out_10_nic_in_11 <= 60
 constr_pair_ok2_10_11: - start_10 + start_11 - hold_10_1 - 2 hold_10_2
   - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6 - 7 hold_10_7
   - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10 - 11 hold_10_11
   - 12 hold_10_12 - 13 hold_10_13 - 14 hold_10_14 - 15 hold_10_15
   - 16 hold_10_16 - 17 hold_10_17 - 18 hold_10_18
   + 60 pair_ok_10_11_for_nic_out_10_nic_in_11 <= 60
 constr_edge_switch_end_start_10_11_OR: switch_nic_out_10_nic_in_11
   - pair_ok_10_11_for_nic_out_10_nic_in_11 <= 0
 constr_pair_ok1_10_12: start_10 - start_12 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 12 hold_10_12 + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15
   + 16 hold_10_16 + 17 hold_10_17 + 18 hold_10_18
   + 60 pair_ok_10_12_for_nic_out_10_nic_in_12 <= 60
 constr_pair_ok2_10_12: - start_10 + start_12 - hold_10_1 - 2 hold_10_2
   - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6 - 7 hold_10_7
   - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10 - 11 hold_10_11
   - 12 hold_10_12 - 13 hold_10_13 - 14 hold_10_14 - 15 hold_10_15
   - 16 hold_10_16 - 17 hold_10_17 - 18 hold_10_18
   + 60 pair_ok_10_12_for_nic_out_10_nic_in_12 <= 60
 constr_edge_switch_end_start_10_12_OR: switch_nic_out_10_nic_in_12
   - pair_ok_10_12_for_nic_out_10_nic_in_12 <= 0
 constr_pair_ok1_10_13: start_10 - start_13 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 12 hold_10_12 + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15
   + 16 hold_10_16 + 17 hold_10_17 + 18 hold_10_18
   + 60 pair_ok_10_13_for_nic_out_10_nic_in_13 <= 60
 constr_pair_ok2_10_13: - start_10 + start_13 - hold_10_1 - 2 hold_10_2
   - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6 - 7 hold_10_7
   - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10 - 11 hold_10_11
   - 12 hold_10_12 - 13 hold_10_13 - 14 hold_10_14 - 15 hold_10_15
   - 16 hold_10_16 - 17 hold_10_17 - 18 hold_10_18
   + 60 pair_ok_10_13_for_nic_out_10_nic_in_13 <= 60
 constr_edge_switch_end_start_10_13_OR: switch_nic_out_10_nic_in_13
   - pair_ok_10_13_for_nic_out_10_nic_in_13 <= 0
 constr_pair_ok1_10_14: start_10 - start_14 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 12 hold_10_12 + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15
   + 16 hold_10_16 + 17 hold_10_17 + 18 hold_10_18
   + 60 pair_ok_10_14_for_nic_out_10_nic_in_14 <= 60
 constr_pair_ok2_10_14: - start_10 + start_14 - hold_10_1 - 2 hold_10_2
   - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6 - 7 hold_10_7
   - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10 - 11 hold_10_11
   - 12 hold_10_12 - 13 hold_10_13 - 14 hold_10_14 - 15 hold_10_15
   - 16 hold_10_16 - 17 hold_10_17 - 18 hold_10_18
   + 60 pair_ok_10_14_for_nic_out_10_nic_in_14 <= 60
 constr_edge_switch_end_start_10_14_OR: switch_nic_out_10_nic_in_14
   - pair_ok_10_14_for_nic_out_10_nic_in_14 <= 0
 constr_pair_ok1_10_15: start_10 - start_15 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 12 hold_10_12 + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15
   + 16 hold_10_16 + 17 hold_10_17 + 18 hold_10_18
   + 60 pair_ok_10_15_for_nic_out_10_nic_in_15 <= 60
 constr_pair_ok2_10_15: - start_10 + start_15 - hold_10_1 - 2 hold_10_2
   - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6 - 7 hold_10_7
   - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10 - 11 hold_10_11
   - 12 hold_10_12 - 13 hold_10_13 - 14 hold_10_14 - 15 hold_10_15
   - 16 hold_10_16 - 17 hold_10_17 - 18 hold_10_18
   + 60 pair_ok_10_15_for_nic_out_10_nic_in_15 <= 60
 constr_edge_switch_end_start_10_15_OR: switch_nic_out_10_nic_in_15
   - pair_ok_10_15_for_nic_out_10_nic_in_15 <= 0
 constr_is_start_zero_11: start_11
   + 60 is_start_zero_11_for_source_nic_in_11 <= 60
 constr_edge_switch_start_zero_11_OR: switch_source_nic_in_11
   - is_start_zero_11_for_source_nic_in_11 <= 0
 constr_is_end_complete_11: start_11 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 12 hold_11_12 + 13 hold_11_13 + 14 hold_11_14 + 15 hold_11_15
   + 16 hold_11_16 + 17 hold_11_17 + 18 hold_11_18 - 60 is_end_complete_11
   >= 0
 constr_edge_switch_end_max_11_OR: switch_nic_out_11_sink
   - is_end_complete_11 <= 0
 constr_pair_ok1_11_0: - start_0 + start_11 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 12 hold_11_12 + 13 hold_11_13 + 14 hold_11_14 + 15 hold_11_15
   + 16 hold_11_16 + 17 hold_11_17 + 18 hold_11_18
   + 60 pair_ok_11_0_for_nic_out_11_nic_in_0 <= 60
 constr_pair_ok2_11_0: start_0 - start_11 - hold_11_1 - 2 hold_11_2
   - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6 - 7 hold_11_7
   - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10 - 11 hold_11_11
   - 12 hold_11_12 - 13 hold_11_13 - 14 hold_11_14 - 15 hold_11_15
   - 16 hold_11_16 - 17 hold_11_17 - 18 hold_11_18
   + 60 pair_ok_11_0_for_nic_out_11_nic_in_0 <= 60
 constr_edge_switch_end_start_11_0_OR: switch_nic_out_11_nic_in_0
   - pair_ok_11_0_for_nic_out_11_nic_in_0 <= 0
 constr_pair_ok1_11_1: - start_1 + start_11 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 12 hold_11_12 + 13 hold_11_13 + 14 hold_11_14 + 15 hold_11_15
   + 16 hold_11_16 + 17 hold_11_17 + 18 hold_11_18
   + 60 pair_ok_11_1_for_nic_out_11_nic_in_1 <= 60
 constr_pair_ok2_11_1: start_1 - start_11 - hold_11_1 - 2 hold_11_2
   - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6 - 7 hold_11_7
   - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10 - 11 hold_11_11
   - 12 hold_11_12 - 13 hold_11_13 - 14 hold_11_14 - 15 hold_11_15
   - 16 hold_11_16 - 17 hold_11_17 - 18 hold_11_18
   + 60 pair_ok_11_1_for_nic_out_11_nic_in_1 <= 60
 constr_edge_switch_end_start_11_1_OR: switch_nic_out_11_nic_in_1
   - pair_ok_11_1_for_nic_out_11_nic_in_1 <= 0
 constr_pair_ok1_11_2: - start_2 + start_11 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 12 hold_11_12 + 13 hold_11_13 + 14 hold_11_14 + 15 hold_11_15
   + 16 hold_11_16 + 17 hold_11_17 + 18 hold_11_18
   + 60 pair_ok_11_2_for_nic_out_11_nic_in_2 <= 60
 constr_pair_ok2_11_2: start_2 - start_11 - hold_11_1 - 2 hold_11_2
   - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6 - 7 hold_11_7
   - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10 - 11 hold_11_11
   - 12 hold_11_12 - 13 hold_11_13 - 14 hold_11_14 - 15 hold_11_15
   - 16 hold_11_16 - 17 hold_11_17 - 18 hold_11_18
   + 60 pair_ok_11_2_for_nic_out_11_nic_in_2 <= 60
 constr_edge_switch_end_start_11_2_OR: switch_nic_out_11_nic_in_2
   - pair_ok_11_2_for_nic_out_11_nic_in_2 <= 0
 constr_pair_ok1_11_3: - start_3 + start_11 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 12 hold_11_12 + 13 hold_11_13 + 14 hold_11_14 + 15 hold_11_15
   + 16 hold_11_16 + 17 hold_11_17 + 18 hold_11_18
   + 60 pair_ok_11_3_for_nic_out_11_nic_in_3 <= 60
 constr_pair_ok2_11_3: start_3 - start_11 - hold_11_1 - 2 hold_11_2
   - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6 - 7 hold_11_7
   - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10 - 11 hold_11_11
   - 12 hold_11_12 - 13 hold_11_13 - 14 hold_11_14 - 15 hold_11_15
   - 16 hold_11_16 - 17 hold_11_17 - 18 hold_11_18
   + 60 pair_ok_11_3_for_nic_out_11_nic_in_3 <= 60
 constr_edge_switch_end_start_11_3_OR: switch_nic_out_11_nic_in_3
   - pair_ok_11_3_for_nic_out_11_nic_in_3 <= 0
 constr_pair_ok1_11_4: - start_4 + start_11 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 12 hold_11_12 + 13 hold_11_13 + 14 hold_11_14 + 15 hold_11_15
   + 16 hold_11_16 + 17 hold_11_17 + 18 hold_11_18
   + 60 pair_ok_11_4_for_nic_out_11_nic_in_4 <= 60
 constr_pair_ok2_11_4: start_4 - start_11 - hold_11_1 - 2 hold_11_2
   - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6 - 7 hold_11_7
   - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10 - 11 hold_11_11
   - 12 hold_11_12 - 13 hold_11_13 - 14 hold_11_14 - 15 hold_11_15
   - 16 hold_11_16 - 17 hold_11_17 - 18 hold_11_18
   + 60 pair_ok_11_4_for_nic_out_11_nic_in_4 <= 60
 constr_edge_switch_end_start_11_4_OR: switch_nic_out_11_nic_in_4
   - pair_ok_11_4_for_nic_out_11_nic_in_4 <= 0
 constr_pair_ok1_11_5: - start_5 + start_11 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 12 hold_11_12 + 13 hold_11_13 + 14 hold_11_14 + 15 hold_11_15
   + 16 hold_11_16 + 17 hold_11_17 + 18 hold_11_18
   + 60 pair_ok_11_5_for_nic_out_11_nic_in_5 <= 60
 constr_pair_ok2_11_5: start_5 - start_11 - hold_11_1 - 2 hold_11_2
   - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6 - 7 hold_11_7
   - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10 - 11 hold_11_11
   - 12 hold_11_12 - 13 hold_11_13 - 14 hold_11_14 - 15 hold_11_15
   - 16 hold_11_16 - 17 hold_11_17 - 18 hold_11_18
   + 60 pair_ok_11_5_for_nic_out_11_nic_in_5 <= 60
 constr_edge_switch_end_start_11_5_OR: switch_nic_out_11_nic_in_5
   - pair_ok_11_5_for_nic_out_11_nic_in_5 <= 0
 constr_pair_ok1_11_6: - start_6 + start_11 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 12 hold_11_12 + 13 hold_11_13 + 14 hold_11_14 + 15 hold_11_15
   + 16 hold_11_16 + 17 hold_11_17 + 18 hold_11_18
   + 60 pair_ok_11_6_for_nic_out_11_nic_in_6 <= 60
 constr_pair_ok2_11_6: start_6 - start_11 - hold_11_1 - 2 hold_11_2
   - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6 - 7 hold_11_7
   - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10 - 11 hold_11_11
   - 12 hold_11_12 - 13 hold_11_13 - 14 hold_11_14 - 15 hold_11_15
   - 16 hold_11_16 - 17 hold_11_17 - 18 hold_11_18
   + 60 pair_ok_11_6_for_nic_out_11_nic_in_6 <= 60
 constr_edge_switch_end_start_11_6_OR: switch_nic_out_11_nic_in_6
   - pair_ok_11_6_for_nic_out_11_nic_in_6 <= 0
 constr_pair_ok1_11_7: - start_7 + start_11 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 12 hold_11_12 + 13 hold_11_13 + 14 hold_11_14 + 15 hold_11_15
   + 16 hold_11_16 + 17 hold_11_17 + 18 hold_11_18
   + 60 pair_ok_11_7_for_nic_out_11_nic_in_7 <= 60
 constr_pair_ok2_11_7: start_7 - start_11 - hold_11_1 - 2 hold_11_2
   - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6 - 7 hold_11_7
   - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10 - 11 hold_11_11
   - 12 hold_11_12 - 13 hold_11_13 - 14 hold_11_14 - 15 hold_11_15
   - 16 hold_11_16 - 17 hold_11_17 - 18 hold_11_18
   + 60 pair_ok_11_7_for_nic_out_11_nic_in_7 <= 60
 constr_edge_switch_end_start_11_7_OR: switch_nic_out_11_nic_in_7
   - pair_ok_11_7_for_nic_out_11_nic_in_7 <= 0
 constr_pair_ok1_11_8: - start_8 + start_11 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 12 hold_11_12 + 13 hold_11_13 + 14 hold_11_14 + 15 hold_11_15
   + 16 hold_11_16 + 17 hold_11_17 + 18 hold_11_18
   + 60 pair_ok_11_8_for_nic_out_11_nic_in_8 <= 60
 constr_pair_ok2_11_8: start_8 - start_11 - hold_11_1 - 2 hold_11_2
   - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6 - 7 hold_11_7
   - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10 - 11 hold_11_11
   - 12 hold_11_12 - 13 hold_11_13 - 14 hold_11_14 - 15 hold_11_15
   - 16 hold_11_16 - 17 hold_11_17 - 18 hold_11_18
   + 60 pair_ok_11_8_for_nic_out_11_nic_in_8 <= 60
 constr_edge_switch_end_start_11_8_OR: switch_nic_out_11_nic_in_8
   - pair_ok_11_8_for_nic_out_11_nic_in_8 <= 0
 constr_pair_ok1_11_9: - start_9 + start_11 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 12 hold_11_12 + 13 hold_11_13 + 14 hold_11_14 + 15 hold_11_15
   + 16 hold_11_16 + 17 hold_11_17 + 18 hold_11_18
   + 60 pair_ok_11_9_for_nic_out_11_nic_in_9 <= 60
 constr_pair_ok2_11_9: start_9 - start_11 - hold_11_1 - 2 hold_11_2
   - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6 - 7 hold_11_7
   - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10 - 11 hold_11_11
   - 12 hold_11_12 - 13 hold_11_13 - 14 hold_11_14 - 15 hold_11_15
   - 16 hold_11_16 - 17 hold_11_17 - 18 hold_11_18
   + 60 pair_ok_11_9_for_nic_out_11_nic_in_9 <= 60
 constr_edge_switch_end_start_11_9_OR: switch_nic_out_11_nic_in_9
   - pair_ok_11_9_for_nic_out_11_nic_in_9 <= 0
 constr_pair_ok1_11_10: - start_10 + start_11 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 12 hold_11_12 + 13 hold_11_13 + 14 hold_11_14 + 15 hold_11_15
   + 16 hold_11_16 + 17 hold_11_17 + 18 hold_11_18
   + 60 pair_ok_11_10_for_nic_out_11_nic_in_10 <= 60
 constr_pair_ok2_11_10: start_10 - start_11 - hold_11_1 - 2 hold_11_2
   - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6 - 7 hold_11_7
   - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10 - 11 hold_11_11
   - 12 hold_11_12 - 13 hold_11_13 - 14 hold_11_14 - 15 hold_11_15
   - 16 hold_11_16 - 17 hold_11_17 - 18 hold_11_18
   + 60 pair_ok_11_10_for_nic_out_11_nic_in_10 <= 60
 constr_edge_switch_end_start_11_10_OR: switch_nic_out_11_nic_in_10
   - pair_ok_11_10_for_nic_out_11_nic_in_10 <= 0
 constr_pair_ok1_11_12: start_11 - start_12 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 12 hold_11_12 + 13 hold_11_13 + 14 hold_11_14 + 15 hold_11_15
   + 16 hold_11_16 + 17 hold_11_17 + 18 hold_11_18
   + 60 pair_ok_11_12_for_nic_out_11_nic_in_12 <= 60
 constr_pair_ok2_11_12: - start_11 + start_12 - hold_11_1 - 2 hold_11_2
   - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6 - 7 hold_11_7
   - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10 - 11 hold_11_11
   - 12 hold_11_12 - 13 hold_11_13 - 14 hold_11_14 - 15 hold_11_15
   - 16 hold_11_16 - 17 hold_11_17 - 18 hold_11_18
   + 60 pair_ok_11_12_for_nic_out_11_nic_in_12 <= 60
 constr_edge_switch_end_start_11_12_OR: switch_nic_out_11_nic_in_12
   - pair_ok_11_12_for_nic_out_11_nic_in_12 <= 0
 constr_pair_ok1_11_13: start_11 - start_13 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 12 hold_11_12 + 13 hold_11_13 + 14 hold_11_14 + 15 hold_11_15
   + 16 hold_11_16 + 17 hold_11_17 + 18 hold_11_18
   + 60 pair_ok_11_13_for_nic_out_11_nic_in_13 <= 60
 constr_pair_ok2_11_13: - start_11 + start_13 - hold_11_1 - 2 hold_11_2
   - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6 - 7 hold_11_7
   - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10 - 11 hold_11_11
   - 12 hold_11_12 - 13 hold_11_13 - 14 hold_11_14 - 15 hold_11_15
   - 16 hold_11_16 - 17 hold_11_17 - 18 hold_11_18
   + 60 pair_ok_11_13_for_nic_out_11_nic_in_13 <= 60
 constr_edge_switch_end_start_11_13_OR: switch_nic_out_11_nic_in_13
   - pair_ok_11_13_for_nic_out_11_nic_in_13 <= 0
 constr_pair_ok1_11_14: start_11 - start_14 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 12 hold_11_12 + 13 hold_11_13 + 14 hold_11_14 + 15 hold_11_15
   + 16 hold_11_16 + 17 hold_11_17 + 18 hold_11_18
   + 60 pair_ok_11_14_for_nic_out_11_nic_in_14 <= 60
 constr_pair_ok2_11_14: - start_11 + start_14 - hold_11_1 - 2 hold_11_2
   - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6 - 7 hold_11_7
   - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10 - 11 hold_11_11
   - 12 hold_11_12 - 13 hold_11_13 - 14 hold_11_14 - 15 hold_11_15
   - 16 hold_11_16 - 17 hold_11_17 - 18 hold_11_18
   + 60 pair_ok_11_14_for_nic_out_11_nic_in_14 <= 60
 constr_edge_switch_end_start_11_14_OR: switch_nic_out_11_nic_in_14
   - pair_ok_11_14_for_nic_out_11_nic_in_14 <= 0
 constr_pair_ok1_11_15: start_11 - start_15 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 12 hold_11_12 + 13 hold_11_13 + 14 hold_11_14 + 15 hold_11_15
   + 16 hold_11_16 + 17 hold_11_17 + 18 hold_11_18
   + 60 pair_ok_11_15_for_nic_out_11_nic_in_15 <= 60
 constr_pair_ok2_11_15: - start_11 + start_15 - hold_11_1 - 2 hold_11_2
   - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6 - 7 hold_11_7
   - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10 - 11 hold_11_11
   - 12 hold_11_12 - 13 hold_11_13 - 14 hold_11_14 - 15 hold_11_15
   - 16 hold_11_16 - 17 hold_11_17 - 18 hold_11_18
   + 60 pair_ok_11_15_for_nic_out_11_nic_in_15 <= 60
 constr_edge_switch_end_start_11_15_OR: switch_nic_out_11_nic_in_15
   - pair_ok_11_15_for_nic_out_11_nic_in_15 <= 0
 constr_is_start_zero_12: start_12
   + 60 is_start_zero_12_for_source_nic_in_12 <= 60
 constr_edge_switch_start_zero_12_OR: switch_source_nic_in_12
   - is_start_zero_12_for_source_nic_in_12 <= 0
 constr_is_end_complete_12: start_12 + hold_12_1 + 2 hold_12_2
   + 3 hold_12_3 + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6 + 7 hold_12_7
   + 8 hold_12_8 + 9 hold_12_9 + 10 hold_12_10 + 11 hold_12_11
   + 12 hold_12_12 + 13 hold_12_13 + 14 hold_12_14 + 15 hold_12_15
   + 16 hold_12_16 + 17 hold_12_17 + 18 hold_12_18 - 60 is_end_complete_12
   >= 0
 constr_edge_switch_end_max_12_OR: switch_nic_out_12_sink
   - is_end_complete_12 <= 0
 constr_pair_ok1_12_0: - start_0 + start_12 + hold_12_1 + 2 hold_12_2
   + 3 hold_12_3 + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6 + 7 hold_12_7
   + 8 hold_12_8 + 9 hold_12_9 + 10 hold_12_10 + 11 hold_12_11
   + 12 hold_12_12 + 13 hold_12_13 + 14 hold_12_14 + 15 hold_12_15
   + 16 hold_12_16 + 17 hold_12_17 + 18 hold_12_18
   + 60 pair_ok_12_0_for_nic_out_12_nic_in_0 <= 60
 constr_pair_ok2_12_0: start_0 - start_12 - hold_12_1 - 2 hold_12_2
   - 3 hold_12_3 - 4 hold_12_4 - 5 hold_12_5 - 6 hold_12_6 - 7 hold_12_7
   - 8 hold_12_8 - 9 hold_12_9 - 10 hold_12_10 - 11 hold_12_11
   - 12 hold_12_12 - 13 hold_12_13 - 14 hold_12_14 - 15 hold_12_15
   - 16 hold_12_16 - 17 hold_12_17 - 18 hold_12_18
   + 60 pair_ok_12_0_for_nic_out_12_nic_in_0 <= 60
 constr_edge_switch_end_start_12_0_OR: switch_nic_out_12_nic_in_0
   - pair_ok_12_0_for_nic_out_12_nic_in_0 <= 0
 constr_pair_ok1_12_1: - start_1 + start_12 + hold_12_1 + 2 hold_12_2
   + 3 hold_12_3 + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6 + 7 hold_12_7
   + 8 hold_12_8 + 9 hold_12_9 + 10 hold_12_10 + 11 hold_12_11
   + 12 hold_12_12 + 13 hold_12_13 + 14 hold_12_14 + 15 hold_12_15
   + 16 hold_12_16 + 17 hold_12_17 + 18 hold_12_18
   + 60 pair_ok_12_1_for_nic_out_12_nic_in_1 <= 60
 constr_pair_ok2_12_1: start_1 - start_12 - hold_12_1 - 2 hold_12_2
   - 3 hold_12_3 - 4 hold_12_4 - 5 hold_12_5 - 6 hold_12_6 - 7 hold_12_7
   - 8 hold_12_8 - 9 hold_12_9 - 10 hold_12_10 - 11 hold_12_11
   - 12 hold_12_12 - 13 hold_12_13 - 14 hold_12_14 - 15 hold_12_15
   - 16 hold_12_16 - 17 hold_12_17 - 18 hold_12_18
   + 60 pair_ok_12_1_for_nic_out_12_nic_in_1 <= 60
 constr_edge_switch_end_start_12_1_OR: switch_nic_out_12_nic_in_1
   - pair_ok_12_1_for_nic_out_12_nic_in_1 <= 0
 constr_pair_ok1_12_2: - start_2 + start_12 + hold_12_1 + 2 hold_12_2
   + 3 hold_12_3 + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6 + 7 hold_12_7
   + 8 hold_12_8 + 9 hold_12_9 + 10 hold_12_10 + 11 hold_12_11
   + 12 hold_12_12 + 13 hold_12_13 + 14 hold_12_14 + 15 hold_12_15
   + 16 hold_12_16 + 17 hold_12_17 + 18 hold_12_18
   + 60 pair_ok_12_2_for_nic_out_12_nic_in_2 <= 60
 constr_pair_ok2_12_2: start_2 - start_12 - hold_12_1 - 2 hold_12_2
   - 3 hold_12_3 - 4 hold_12_4 - 5 hold_12_5 - 6 hold_12_6 - 7 hold_12_7
   - 8 hold_12_8 - 9 hold_12_9 - 10 hold_12_10 - 11 hold_12_11
   - 12 hold_12_12 - 13 hold_12_13 - 14 hold_12_14 - 15 hold_12_15
   - 16 hold_12_16 - 17 hold_12_17 - 18 hold_12_18
   + 60 pair_ok_12_2_for_nic_out_12_nic_in_2 <= 60
 constr_edge_switch_end_start_12_2_OR: switch_nic_out_12_nic_in_2
   - pair_ok_12_2_for_nic_out_12_nic_in_2 <= 0
 constr_pair_ok1_12_3: - start_3 + start_12 + hold_12_1 + 2 hold_12_2
   + 3 hold_12_3 + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6 + 7 hold_12_7
   + 8 hold_12_8 + 9 hold_12_9 + 10 hold_12_10 + 11 hold_12_11
   + 12 hold_12_12 + 13 hold_12_13 + 14 hold_12_14 + 15 hold_12_15
   + 16 hold_12_16 + 17 hold_12_17 + 18 hold_12_18
   + 60 pair_ok_12_3_for_nic_out_12_nic_in_3 <= 60
 constr_pair_ok2_12_3: start_3 - start_12 - hold_12_1 - 2 hold_12_2
   - 3 hold_12_3 - 4 hold_12_4 - 5 hold_12_5 - 6 hold_12_6 - 7 hold_12_7
   - 8 hold_12_8 - 9 hold_12_9 - 10 hold_12_10 - 11 hold_12_11
   - 12 hold_12_12 - 13 hold_12_13 - 14 hold_12_14 - 15 hold_12_15
   - 16 hold_12_16 - 17 hold_12_17 - 18 hold_12_18
   + 60 pair_ok_12_3_for_nic_out_12_nic_in_3 <= 60
 constr_edge_switch_end_start_12_3_OR: switch_nic_out_12_nic_in_3
   - pair_ok_12_3_for_nic_out_12_nic_in_3 <= 0
 constr_pair_ok1_12_4: - start_4 + start_12 + hold_12_1 + 2 hold_12_2
   + 3 hold_12_3 + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6 + 7 hold_12_7
   + 8 hold_12_8 + 9 hold_12_9 + 10 hold_12_10 + 11 hold_12_11
   + 12 hold_12_12 + 13 hold_12_13 + 14 hold_12_14 + 15 hold_12_15
   + 16 hold_12_16 + 17 hold_12_17 + 18 hold_12_18
   + 60 pair_ok_12_4_for_nic_out_12_nic_in_4 <= 60
 constr_pair_ok2_12_4: start_4 - start_12 - hold_12_1 - 2 hold_12_2
   - 3 hold_12_3 - 4 hold_12_4 - 5 hold_12_5 - 6 hold_12_6 - 7 hold_12_7
   - 8 hold_12_8 - 9 hold_12_9 - 10 hold_12_10 - 11 hold_12_11
   - 12 hold_12_12 - 13 hold_12_13 - 14 hold_12_14 - 15 hold_12_15
   - 16 hold_12_16 - 17 hold_12_17 - 18 hold_12_18
   + 60 pair_ok_12_4_for_nic_out_12_nic_in_4 <= 60
 constr_edge_switch_end_start_12_4_OR: switch_nic_out_12_nic_in_4
   - pair_ok_12_4_for_nic_out_12_nic_in_4 <= 0
 constr_pair_ok1_12_5: - start_5 + start_12 + hold_12_1 + 2 hold_12_2
   + 3 hold_12_3 + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6 + 7 hold_12_7
   + 8 hold_12_8 + 9 hold_12_9 + 10 hold_12_10 + 11 hold_12_11
   + 12 hold_12_12 + 13 hold_12_13 + 14 hold_12_14 + 15 hold_12_15
   + 16 hold_12_16 + 17 hold_12_17 + 18 hold_12_18
   + 60 pair_ok_12_5_for_nic_out_12_nic_in_5 <= 60
 constr_pair_ok2_12_5: start_5 - start_12 - hold_12_1 - 2 hold_12_2
   - 3 hold_12_3 - 4 hold_12_4 - 5 hold_12_5 - 6 hold_12_6 - 7 hold_12_7
   - 8 hold_12_8 - 9 hold_12_9 - 10 hold_12_10 - 11 hold_12_11
   - 12 hold_12_12 - 13 hold_12_13 - 14 hold_12_14 - 15 hold_12_15
   - 16 hold_12_16 - 17 hold_12_17 - 18 hold_12_18
   + 60 pair_ok_12_5_for_nic_out_12_nic_in_5 <= 60
 constr_edge_switch_end_start_12_5_OR: switch_nic_out_12_nic_in_5
   - pair_ok_12_5_for_nic_out_12_nic_in_5 <= 0
 constr_pair_ok1_12_6: - start_6 + start_12 + hold_12_1 + 2 hold_12_2
   + 3 hold_12_3 + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6 + 7 hold_12_7
   + 8 hold_12_8 + 9 hold_12_9 + 10 hold_12_10 + 11 hold_12_11
   + 12 hold_12_12 + 13 hold_12_13 + 14 hold_12_14 + 15 hold_12_15
   + 16 hold_12_16 + 17 hold_12_17 + 18 hold_12_18
   + 60 pair_ok_12_6_for_nic_out_12_nic_in_6 <= 60
 constr_pair_ok2_12_6: start_6 - start_12 - hold_12_1 - 2 hold_12_2
   - 3 hold_12_3 - 4 hold_12_4 - 5 hold_12_5 - 6 hold_12_6 - 7 hold_12_7
   - 8 hold_12_8 - 9 hold_12_9 - 10 hold_12_10 - 11 hold_12_11
   - 12 hold_12_12 - 13 hold_12_13 - 14 hold_12_14 - 15 hold_12_15
   - 16 hold_12_16 - 17 hold_12_17 - 18 hold_12_18
   + 60 pair_ok_12_6_for_nic_out_12_nic_in_6 <= 60
 constr_edge_switch_end_start_12_6_OR: switch_nic_out_12_nic_in_6
   - pair_ok_12_6_for_nic_out_12_nic_in_6 <= 0
 constr_pair_ok1_12_7: - start_7 + start_12 + hold_12_1 + 2 hold_12_2
   + 3 hold_12_3 + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6 + 7 hold_12_7
   + 8 hold_12_8 + 9 hold_12_9 + 10 hold_12_10 + 11 hold_12_11
   + 12 hold_12_12 + 13 hold_12_13 + 14 hold_12_14 + 15 hold_12_15
   + 16 hold_12_16 + 17 hold_12_17 + 18 hold_12_18
   + 60 pair_ok_12_7_for_nic_out_12_nic_in_7 <= 60
 constr_pair_ok2_12_7: start_7 - start_12 - hold_12_1 - 2 hold_12_2
   - 3 hold_12_3 - 4 hold_12_4 - 5 hold_12_5 - 6 hold_12_6 - 7 hold_12_7
   - 8 hold_12_8 - 9 hold_12_9 - 10 hold_12_10 - 11 hold_12_11
   - 12 hold_12_12 - 13 hold_12_13 - 14 hold_12_14 - 15 hold_12_15
   - 16 hold_12_16 - 17 hold_12_17 - 18 hold_12_18
   + 60 pair_ok_12_7_for_nic_out_12_nic_in_7 <= 60
 constr_edge_switch_end_start_12_7_OR: switch_nic_out_12_nic_in_7
   - pair_ok_12_7_for_nic_out_12_nic_in_7 <= 0
 constr_pair_ok1_12_8: - start_8 + start_12 + hold_12_1 + 2 hold_12_2
   + 3 hold_12_3 + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6 + 7 hold_12_7
   + 8 hold_12_8 + 9 hold_12_9 + 10 hold_12_10 + 11 hold_12_11
   + 12 hold_12_12 + 13 hold_12_13 + 14 hold_12_14 + 15 hold_12_15
   + 16 hold_12_16 + 17 hold_12_17 + 18 hold_12_18
   + 60 pair_ok_12_8_for_nic_out_12_nic_in_8 <= 60
 constr_pair_ok2_12_8: start_8 - start_12 - hold_12_1 - 2 hold_12_2
   - 3 hold_12_3 - 4 hold_12_4 - 5 hold_12_5 - 6 hold_12_6 - 7 hold_12_7
   - 8 hold_12_8 - 9 hold_12_9 - 10 hold_12_10 - 11 hold_12_11
   - 12 hold_12_12 - 13 hold_12_13 - 14 hold_12_14 - 15 hold_12_15
   - 16 hold_12_16 - 17 hold_12_17 - 18 hold_12_18
   + 60 pair_ok_12_8_for_nic_out_12_nic_in_8 <= 60
 constr_edge_switch_end_start_12_8_OR: switch_nic_out_12_nic_in_8
   - pair_ok_12_8_for_nic_out_12_nic_in_8 <= 0
 constr_pair_ok1_12_9: - start_9 + start_12 + hold_12_1 + 2 hold_12_2
   + 3 hold_12_3 + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6 + 7 hold_12_7
   + 8 hold_12_8 + 9 hold_12_9 + 10 hold_12_10 + 11 hold_12_11
   + 12 hold_12_12 + 13 hold_12_13 + 14 hold_12_14 + 15 hold_12_15
   + 16 hold_12_16 + 17 hold_12_17 + 18 hold_12_18
   + 60 pair_ok_12_9_for_nic_out_12_nic_in_9 <= 60
 constr_pair_ok2_12_9: start_9 - start_12 - hold_12_1 - 2 hold_12_2
   - 3 hold_12_3 - 4 hold_12_4 - 5 hold_12_5 - 6 hold_12_6 - 7 hold_12_7
   - 8 hold_12_8 - 9 hold_12_9 - 10 hold_12_10 - 11 hold_12_11
   - 12 hold_12_12 - 13 hold_12_13 - 14 hold_12_14 - 15 hold_12_15
   - 16 hold_12_16 - 17 hold_12_17 - 18 hold_12_18
   + 60 pair_ok_12_9_for_nic_out_12_nic_in_9 <= 60
 constr_edge_switch_end_start_12_9_OR: switch_nic_out_12_nic_in_9
   - pair_ok_12_9_for_nic_out_12_nic_in_9 <= 0
 constr_pair_ok1_12_10: - start_10 + start_12 + hold_12_1 + 2 hold_12_2
   + 3 hold_12_3 + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6 + 7 hold_12_7
   + 8 hold_12_8 + 9 hold_12_9 + 10 hold_12_10 + 11 hold_12_11
   + 12 hold_12_12 + 13 hold_12_13 + 14 hold_12_14 + 15 hold_12_15
   + 16 hold_12_16 + 17 hold_12_17 + 18 hold_12_18
   + 60 pair_ok_12_10_for_nic_out_12_nic_in_10 <= 60
 constr_pair_ok2_12_10: start_10 - start_12 - hold_12_1 - 2 hold_12_2
   - 3 hold_12_3 - 4 hold_12_4 - 5 hold_12_5 - 6 hold_12_6 - 7 hold_12_7
   - 8 hold_12_8 - 9 hold_12_9 - 10 hold_12_10 - 11 hold_12_11
   - 12 hold_12_12 - 13 hold_12_13 - 14 hold_12_14 - 15 hold_12_15
   - 16 hold_12_16 - 17 hold_12_17 - 18 hold_12_18
   + 60 pair_ok_12_10_for_nic_out_12_nic_in_10 <= 60
 constr_edge_switch_end_start_12_10_OR: switch_nic_out_12_nic_in_10
   - pair_ok_12_10_for_nic_out_12_nic_in_10 <= 0
 constr_pair_ok1_12_11: - start_11 + start_12 + hold_12_1 + 2 hold_12_2
   + 3 hold_12_3 + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6 + 7 hold_12_7
   + 8 hold_12_8 + 9 hold_12_9 + 10 hold_12_10 + 11 hold_12_11
   + 12 hold_12_12 + 13 hold_12_13 + 14 hold_12_14 + 15 hold_12_15
   + 16 hold_12_16 + 17 hold_12_17 + 18 hold_12_18
   + 60 pair_ok_12_11_for_nic_out_12_nic_in_11 <= 60
 constr_pair_ok2_12_11: start_11 - start_12 - hold_12_1 - 2 hold_12_2
   - 3 hold_12_3 - 4 hold_12_4 - 5 hold_12_5 - 6 hold_12_6 - 7 hold_12_7
   - 8 hold_12_8 - 9 hold_12_9 - 10 hold_12_10 - 11 hold_12_11
   - 12 hold_12_12 - 13 hold_12_13 - 14 hold_12_14 - 15 hold_12_15
   - 16 hold_12_16 - 17 hold_12_17 - 18 hold_12_18
   + 60 pair_ok_12_11_for_nic_out_12_nic_in_11 <= 60
 constr_edge_switch_end_start_12_11_OR: switch_nic_out_12_nic_in_11
   - pair_ok_12_11_for_nic_out_12_nic_in_11 <= 0
 constr_pair_ok1_12_13: start_12 - start_13 + hold_12_1 + 2 hold_12_2
   + 3 hold_12_3 + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6 + 7 hold_12_7
   + 8 hold_12_8 + 9 hold_12_9 + 10 hold_12_10 + 11 hold_12_11
   + 12 hold_12_12 + 13 hold_12_13 + 14 hold_12_14 + 15 hold_12_15
   + 16 hold_12_16 + 17 hold_12_17 + 18 hold_12_18
   + 60 pair_ok_12_13_for_nic_out_12_nic_in_13 <= 60
 constr_pair_ok2_12_13: - start_12 + start_13 - hold_12_1 - 2 hold_12_2
   - 3 hold_12_3 - 4 hold_12_4 - 5 hold_12_5 - 6 hold_12_6 - 7 hold_12_7
   - 8 hold_12_8 - 9 hold_12_9 - 10 hold_12_10 - 11 hold_12_11
   - 12 hold_12_12 - 13 hold_12_13 - 14 hold_12_14 - 15 hold_12_15
   - 16 hold_12_16 - 17 hold_12_17 - 18 hold_12_18
   + 60 pair_ok_12_13_for_nic_out_12_nic_in_13 <= 60
 constr_edge_switch_end_start_12_13_OR: switch_nic_out_12_nic_in_13
   - pair_ok_12_13_for_nic_out_12_nic_in_13 <= 0
 constr_pair_ok1_12_14: start_12 - start_14 + hold_12_1 + 2 hold_12_2
   + 3 hold_12_3 + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6 + 7 hold_12_7
   + 8 hold_12_8 + 9 hold_12_9 + 10 hold_12_10 + 11 hold_12_11
   + 12 hold_12_12 + 13 hold_12_13 + 14 hold_12_14 + 15 hold_12_15
   + 16 hold_12_16 + 17 hold_12_17 + 18 hold_12_18
   + 60 pair_ok_12_14_for_nic_out_12_nic_in_14 <= 60
 constr_pair_ok2_12_14: - start_12 + start_14 - hold_12_1 - 2 hold_12_2
   - 3 hold_12_3 - 4 hold_12_4 - 5 hold_12_5 - 6 hold_12_6 - 7 hold_12_7
   - 8 hold_12_8 - 9 hold_12_9 - 10 hold_12_10 - 11 hold_12_11
   - 12 hold_12_12 - 13 hold_12_13 - 14 hold_12_14 - 15 hold_12_15
   - 16 hold_12_16 - 17 hold_12_17 - 18 hold_12_18
   + 60 pair_ok_12_14_for_nic_out_12_nic_in_14 <= 60
 constr_edge_switch_end_start_12_14_OR: switch_nic_out_12_nic_in_14
   - pair_ok_12_14_for_nic_out_12_nic_in_14 <= 0
 constr_pair_ok1_12_15: start_12 - start_15 + hold_12_1 + 2 hold_12_2
   + 3 hold_12_3 + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6 + 7 hold_12_7
   + 8 hold_12_8 + 9 hold_12_9 + 10 hold_12_10 + 11 hold_12_11
   + 12 hold_12_12 + 13 hold_12_13 + 14 hold_12_14 + 15 hold_12_15
   + 16 hold_12_16 + 17 hold_12_17 + 18 hold_12_18
   + 60 pair_ok_12_15_for_nic_out_12_nic_in_15 <= 60
 constr_pair_ok2_12_15: - start_12 + start_15 - hold_12_1 - 2 hold_12_2
   - 3 hold_12_3 - 4 hold_12_4 - 5 hold_12_5 - 6 hold_12_6 - 7 hold_12_7
   - 8 hold_12_8 - 9 hold_12_9 - 10 hold_12_10 - 11 hold_12_11
   - 12 hold_12_12 - 13 hold_12_13 - 14 hold_12_14 - 15 hold_12_15
   - 16 hold_12_16 - 17 hold_12_17 - 18 hold_12_18
   + 60 pair_ok_12_15_for_nic_out_12_nic_in_15 <= 60
 constr_edge_switch_end_start_12_15_OR: switch_nic_out_12_nic_in_15
   - pair_ok_12_15_for_nic_out_12_nic_in_15 <= 0
 constr_is_start_zero_13: start_13
   + 60 is_start_zero_13_for_source_nic_in_13 <= 60
 constr_edge_switch_start_zero_13_OR: switch_source_nic_in_13
   - is_start_zero_13_for_source_nic_in_13 <= 0
 constr_is_end_complete_13: start_13 + hold_13_1 + 2 hold_13_2
   + 3 hold_13_3 + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6 + 7 hold_13_7
   + 8 hold_13_8 + 9 hold_13_9 + 10 hold_13_10 + 11 hold_13_11
   + 12 hold_13_12 + 13 hold_13_13 + 14 hold_13_14 + 15 hold_13_15
   + 16 hold_13_16 + 17 hold_13_17 + 18 hold_13_18 - 60 is_end_complete_13
   >= 0
 constr_edge_switch_end_max_13_OR: switch_nic_out_13_sink
   - is_end_complete_13 <= 0
 constr_pair_ok1_13_0: - start_0 + start_13 + hold_13_1 + 2 hold_13_2
   + 3 hold_13_3 + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6 + 7 hold_13_7
   + 8 hold_13_8 + 9 hold_13_9 + 10 hold_13_10 + 11 hold_13_11
   + 12 hold_13_12 + 13 hold_13_13 + 14 hold_13_14 + 15 hold_13_15
   + 16 hold_13_16 + 17 hold_13_17 + 18 hold_13_18
   + 60 pair_ok_13_0_for_nic_out_13_nic_in_0 <= 60
 constr_pair_ok2_13_0: start_0 - start_13 - hold_13_1 - 2 hold_13_2
   - 3 hold_13_3 - 4 hold_13_4 - 5 hold_13_5 - 6 hold_13_6 - 7 hold_13_7
   - 8 hold_13_8 - 9 hold_13_9 - 10 hold_13_10 - 11 hold_13_11
   - 12 hold_13_12 - 13 hold_13_13 - 14 hold_13_14 - 15 hold_13_15
   - 16 hold_13_16 - 17 hold_13_17 - 18 hold_13_18
   + 60 pair_ok_13_0_for_nic_out_13_nic_in_0 <= 60
 constr_edge_switch_end_start_13_0_OR: switch_nic_out_13_nic_in_0
   - pair_ok_13_0_for_nic_out_13_nic_in_0 <= 0
 constr_pair_ok1_13_1: - start_1 + start_13 + hold_13_1 + 2 hold_13_2
   + 3 hold_13_3 + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6 + 7 hold_13_7
   + 8 hold_13_8 + 9 hold_13_9 + 10 hold_13_10 + 11 hold_13_11
   + 12 hold_13_12 + 13 hold_13_13 + 14 hold_13_14 + 15 hold_13_15
   + 16 hold_13_16 + 17 hold_13_17 + 18 hold_13_18
   + 60 pair_ok_13_1_for_nic_out_13_nic_in_1 <= 60
 constr_pair_ok2_13_1: start_1 - start_13 - hold_13_1 - 2 hold_13_2
   - 3 hold_13_3 - 4 hold_13_4 - 5 hold_13_5 - 6 hold_13_6 - 7 hold_13_7
   - 8 hold_13_8 - 9 hold_13_9 - 10 hold_13_10 - 11 hold_13_11
   - 12 hold_13_12 - 13 hold_13_13 - 14 hold_13_14 - 15 hold_13_15
   - 16 hold_13_16 - 17 hold_13_17 - 18 hold_13_18
   + 60 pair_ok_13_1_for_nic_out_13_nic_in_1 <= 60
 constr_edge_switch_end_start_13_1_OR: switch_nic_out_13_nic_in_1
   - pair_ok_13_1_for_nic_out_13_nic_in_1 <= 0
 constr_pair_ok1_13_2: - start_2 + start_13 + hold_13_1 + 2 hold_13_2
   + 3 hold_13_3 + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6 + 7 hold_13_7
   + 8 hold_13_8 + 9 hold_13_9 + 10 hold_13_10 + 11 hold_13_11
   + 12 hold_13_12 + 13 hold_13_13 + 14 hold_13_14 + 15 hold_13_15
   + 16 hold_13_16 + 17 hold_13_17 + 18 hold_13_18
   + 60 pair_ok_13_2_for_nic_out_13_nic_in_2 <= 60
 constr_pair_ok2_13_2: start_2 - start_13 - hold_13_1 - 2 hold_13_2
   - 3 hold_13_3 - 4 hold_13_4 - 5 hold_13_5 - 6 hold_13_6 - 7 hold_13_7
   - 8 hold_13_8 - 9 hold_13_9 - 10 hold_13_10 - 11 hold_13_11
   - 12 hold_13_12 - 13 hold_13_13 - 14 hold_13_14 - 15 hold_13_15
   - 16 hold_13_16 - 17 hold_13_17 - 18 hold_13_18
   + 60 pair_ok_13_2_for_nic_out_13_nic_in_2 <= 60
 constr_edge_switch_end_start_13_2_OR: switch_nic_out_13_nic_in_2
   - pair_ok_13_2_for_nic_out_13_nic_in_2 <= 0
 constr_pair_ok1_13_3: - start_3 + start_13 + hold_13_1 + 2 hold_13_2
   + 3 hold_13_3 + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6 + 7 hold_13_7
   + 8 hold_13_8 + 9 hold_13_9 + 10 hold_13_10 + 11 hold_13_11
   + 12 hold_13_12 + 13 hold_13_13 + 14 hold_13_14 + 15 hold_13_15
   + 16 hold_13_16 + 17 hold_13_17 + 18 hold_13_18
   + 60 pair_ok_13_3_for_nic_out_13_nic_in_3 <= 60
 constr_pair_ok2_13_3: start_3 - start_13 - hold_13_1 - 2 hold_13_2
   - 3 hold_13_3 - 4 hold_13_4 - 5 hold_13_5 - 6 hold_13_6 - 7 hold_13_7
   - 8 hold_13_8 - 9 hold_13_9 - 10 hold_13_10 - 11 hold_13_11
   - 12 hold_13_12 - 13 hold_13_13 - 14 hold_13_14 - 15 hold_13_15
   - 16 hold_13_16 - 17 hold_13_17 - 18 hold_13_18
   + 60 pair_ok_13_3_for_nic_out_13_nic_in_3 <= 60
 constr_edge_switch_end_start_13_3_OR: switch_nic_out_13_nic_in_3
   - pair_ok_13_3_for_nic_out_13_nic_in_3 <= 0
 constr_pair_ok1_13_4: - start_4 + start_13 + hold_13_1 + 2 hold_13_2
   + 3 hold_13_3 + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6 + 7 hold_13_7
   + 8 hold_13_8 + 9 hold_13_9 + 10 hold_13_10 + 11 hold_13_11
   + 12 hold_13_12 + 13 hold_13_13 + 14 hold_13_14 + 15 hold_13_15
   + 16 hold_13_16 + 17 hold_13_17 + 18 hold_13_18
   + 60 pair_ok_13_4_for_nic_out_13_nic_in_4 <= 60
 constr_pair_ok2_13_4: start_4 - start_13 - hold_13_1 - 2 hold_13_2
   - 3 hold_13_3 - 4 hold_13_4 - 5 hold_13_5 - 6 hold_13_6 - 7 hold_13_7
   - 8 hold_13_8 - 9 hold_13_9 - 10 hold_13_10 - 11 hold_13_11
   - 12 hold_13_12 - 13 hold_13_13 - 14 hold_13_14 - 15 hold_13_15
   - 16 hold_13_16 - 17 hold_13_17 - 18 hold_13_18
   + 60 pair_ok_13_4_for_nic_out_13_nic_in_4 <= 60
 constr_edge_switch_end_start_13_4_OR: switch_nic_out_13_nic_in_4
   - pair_ok_13_4_for_nic_out_13_nic_in_4 <= 0
 constr_pair_ok1_13_5: - start_5 + start_13 + hold_13_1 + 2 hold_13_2
   + 3 hold_13_3 + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6 + 7 hold_13_7
   + 8 hold_13_8 + 9 hold_13_9 + 10 hold_13_10 + 11 hold_13_11
   + 12 hold_13_12 + 13 hold_13_13 + 14 hold_13_14 + 15 hold_13_15
   + 16 hold_13_16 + 17 hold_13_17 + 18 hold_13_18
   + 60 pair_ok_13_5_for_nic_out_13_nic_in_5 <= 60
 constr_pair_ok2_13_5: start_5 - start_13 - hold_13_1 - 2 hold_13_2
   - 3 hold_13_3 - 4 hold_13_4 - 5 hold_13_5 - 6 hold_13_6 - 7 hold_13_7
   - 8 hold_13_8 - 9 hold_13_9 - 10 hold_13_10 - 11 hold_13_11
   - 12 hold_13_12 - 13 hold_13_13 - 14 hold_13_14 - 15 hold_13_15
   - 16 hold_13_16 - 17 hold_13_17 - 18 hold_13_18
   + 60 pair_ok_13_5_for_nic_out_13_nic_in_5 <= 60
 constr_edge_switch_end_start_13_5_OR: switch_nic_out_13_nic_in_5
   - pair_ok_13_5_for_nic_out_13_nic_in_5 <= 0
 constr_pair_ok1_13_6: - start_6 + start_13 + hold_13_1 + 2 hold_13_2
   + 3 hold_13_3 + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6 + 7 hold_13_7
   + 8 hold_13_8 + 9 hold_13_9 + 10 hold_13_10 + 11 hold_13_11
   + 12 hold_13_12 + 13 hold_13_13 + 14 hold_13_14 + 15 hold_13_15
   + 16 hold_13_16 + 17 hold_13_17 + 18 hold_13_18
   + 60 pair_ok_13_6_for_nic_out_13_nic_in_6 <= 60
 constr_pair_ok2_13_6: start_6 - start_13 - hold_13_1 - 2 hold_13_2
   - 3 hold_13_3 - 4 hold_13_4 - 5 hold_13_5 - 6 hold_13_6 - 7 hold_13_7
   - 8 hold_13_8 - 9 hold_13_9 - 10 hold_13_10 - 11 hold_13_11
   - 12 hold_13_12 - 13 hold_13_13 - 14 hold_13_14 - 15 hold_13_15
   - 16 hold_13_16 - 17 hold_13_17 - 18 hold_13_18
   + 60 pair_ok_13_6_for_nic_out_13_nic_in_6 <= 60
 constr_edge_switch_end_start_13_6_OR: switch_nic_out_13_nic_in_6
   - pair_ok_13_6_for_nic_out_13_nic_in_6 <= 0
 constr_pair_ok1_13_7: - start_7 + start_13 + hold_13_1 + 2 hold_13_2
   + 3 hold_13_3 + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6 + 7 hold_13_7
   + 8 hold_13_8 + 9 hold_13_9 + 10 hold_13_10 + 11 hold_13_11
   + 12 hold_13_12 + 13 hold_13_13 + 14 hold_13_14 + 15 hold_13_15
   + 16 hold_13_16 + 17 hold_13_17 + 18 hold_13_18
   + 60 pair_ok_13_7_for_nic_out_13_nic_in_7 <= 60
 constr_pair_ok2_13_7: start_7 - start_13 - hold_13_1 - 2 hold_13_2
   - 3 hold_13_3 - 4 hold_13_4 - 5 hold_13_5 - 6 hold_13_6 - 7 hold_13_7
   - 8 hold_13_8 - 9 hold_13_9 - 10 hold_13_10 - 11 hold_13_11
   - 12 hold_13_12 - 13 hold_13_13 - 14 hold_13_14 - 15 hold_13_15
   - 16 hold_13_16 - 17 hold_13_17 - 18 hold_13_18
   + 60 pair_ok_13_7_for_nic_out_13_nic_in_7 <= 60
 constr_edge_switch_end_start_13_7_OR: switch_nic_out_13_nic_in_7
   - pair_ok_13_7_for_nic_out_13_nic_in_7 <= 0
 constr_pair_ok1_13_8: - start_8 + start_13 + hold_13_1 + 2 hold_13_2
   + 3 hold_13_3 + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6 + 7 hold_13_7
   + 8 hold_13_8 + 9 hold_13_9 + 10 hold_13_10 + 11 hold_13_11
   + 12 hold_13_12 + 13 hold_13_13 + 14 hold_13_14 + 15 hold_13_15
   + 16 hold_13_16 + 17 hold_13_17 + 18 hold_13_18
   + 60 pair_ok_13_8_for_nic_out_13_nic_in_8 <= 60
 constr_pair_ok2_13_8: start_8 - start_13 - hold_13_1 - 2 hold_13_2
   - 3 hold_13_3 - 4 hold_13_4 - 5 hold_13_5 - 6 hold_13_6 - 7 hold_13_7
   - 8 hold_13_8 - 9 hold_13_9 - 10 hold_13_10 - 11 hold_13_11
   - 12 hold_13_12 - 13 hold_13_13 - 14 hold_13_14 - 15 hold_13_15
   - 16 hold_13_16 - 17 hold_13_17 - 18 hold_13_18
   + 60 pair_ok_13_8_for_nic_out_13_nic_in_8 <= 60
 constr_edge_switch_end_start_13_8_OR: switch_nic_out_13_nic_in_8
   - pair_ok_13_8_for_nic_out_13_nic_in_8 <= 0
 constr_pair_ok1_13_9: - start_9 + start_13 + hold_13_1 + 2 hold_13_2
   + 3 hold_13_3 + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6 + 7 hold_13_7
   + 8 hold_13_8 + 9 hold_13_9 + 10 hold_13_10 + 11 hold_13_11
   + 12 hold_13_12 + 13 hold_13_13 + 14 hold_13_14 + 15 hold_13_15
   + 16 hold_13_16 + 17 hold_13_17 + 18 hold_13_18
   + 60 pair_ok_13_9_for_nic_out_13_nic_in_9 <= 60
 constr_pair_ok2_13_9: start_9 - start_13 - hold_13_1 - 2 hold_13_2
   - 3 hold_13_3 - 4 hold_13_4 - 5 hold_13_5 - 6 hold_13_6 - 7 hold_13_7
   - 8 hold_13_8 - 9 hold_13_9 - 10 hold_13_10 - 11 hold_13_11
   - 12 hold_13_12 - 13 hold_13_13 - 14 hold_13_14 - 15 hold_13_15
   - 16 hold_13_16 - 17 hold_13_17 - 18 hold_13_18
   + 60 pair_ok_13_9_for_nic_out_13_nic_in_9 <= 60
 constr_edge_switch_end_start_13_9_OR: switch_nic_out_13_nic_in_9
   - pair_ok_13_9_for_nic_out_13_nic_in_9 <= 0
 constr_pair_ok1_13_10: - start_10 + start_13 + hold_13_1 + 2 hold_13_2
   + 3 hold_13_3 + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6 + 7 hold_13_7
   + 8 hold_13_8 + 9 hold_13_9 + 10 hold_13_10 + 11 hold_13_11
   + 12 hold_13_12 + 13 hold_13_13 + 14 hold_13_14 + 15 hold_13_15
   + 16 hold_13_16 + 17 hold_13_17 + 18 hold_13_18
   + 60 pair_ok_13_10_for_nic_out_13_nic_in_10 <= 60
 constr_pair_ok2_13_10: start_10 - start_13 - hold_13_1 - 2 hold_13_2
   - 3 hold_13_3 - 4 hold_13_4 - 5 hold_13_5 - 6 hold_13_6 - 7 hold_13_7
   - 8 hold_13_8 - 9 hold_13_9 - 10 hold_13_10 - 11 hold_13_11
   - 12 hold_13_12 - 13 hold_13_13 - 14 hold_13_14 - 15 hold_13_15
   - 16 hold_13_16 - 17 hold_13_17 - 18 hold_13_18
   + 60 pair_ok_13_10_for_nic_out_13_nic_in_10 <= 60
 constr_edge_switch_end_start_13_10_OR: switch_nic_out_13_nic_in_10
   - pair_ok_13_10_for_nic_out_13_nic_in_10 <= 0
 constr_pair_ok1_13_11: - start_11 + start_13 + hold_13_1 + 2 hold_13_2
   + 3 hold_13_3 + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6 + 7 hold_13_7
   + 8 hold_13_8 + 9 hold_13_9 + 10 hold_13_10 + 11 hold_13_11
   + 12 hold_13_12 + 13 hold_13_13 + 14 hold_13_14 + 15 hold_13_15
   + 16 hold_13_16 + 17 hold_13_17 + 18 hold_13_18
   + 60 pair_ok_13_11_for_nic_out_13_nic_in_11 <= 60
 constr_pair_ok2_13_11: start_11 - start_13 - hold_13_1 - 2 hold_13_2
   - 3 hold_13_3 - 4 hold_13_4 - 5 hold_13_5 - 6 hold_13_6 - 7 hold_13_7
   - 8 hold_13_8 - 9 hold_13_9 - 10 hold_13_10 - 11 hold_13_11
   - 12 hold_13_12 - 13 hold_13_13 - 14 hold_13_14 - 15 hold_13_15
   - 16 hold_13_16 - 17 hold_13_17 - 18 hold_13_18
   + 60 pair_ok_13_11_for_nic_out_13_nic_in_11 <= 60
 constr_edge_switch_end_start_13_11_OR: switch_nic_out_13_nic_in_11
   - pair_ok_13_11_for_nic_out_13_nic_in_11 <= 0
 constr_pair_ok1_13_12: - start_12 + start_13 + hold_13_1 + 2 hold_13_2
   + 3 hold_13_3 + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6 + 7 hold_13_7
   + 8 hold_13_8 + 9 hold_13_9 + 10 hold_13_10 + 11 hold_13_11
   + 12 hold_13_12 + 13 hold_13_13 + 14 hold_13_14 + 15 hold_13_15
   + 16 hold_13_16 + 17 hold_13_17 + 18 hold_13_18
   + 60 pair_ok_13_12_for_nic_out_13_nic_in_12 <= 60
 constr_pair_ok2_13_12: start_12 - start_13 - hold_13_1 - 2 hold_13_2
   - 3 hold_13_3 - 4 hold_13_4 - 5 hold_13_5 - 6 hold_13_6 - 7 hold_13_7
   - 8 hold_13_8 - 9 hold_13_9 - 10 hold_13_10 - 11 hold_13_11
   - 12 hold_13_12 - 13 hold_13_13 - 14 hold_13_14 - 15 hold_13_15
   - 16 hold_13_16 - 17 hold_13_17 - 18 hold_13_18
   + 60 pair_ok_13_12_for_nic_out_13_nic_in_12 <= 60
 constr_edge_switch_end_start_13_12_OR: switch_nic_out_13_nic_in_12
   - pair_ok_13_12_for_nic_out_13_nic_in_12 <= 0
 constr_pair_ok1_13_14: start_13 - start_14 + hold_13_1 + 2 hold_13_2
   + 3 hold_13_3 + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6 + 7 hold_13_7
   + 8 hold_13_8 + 9 hold_13_9 + 10 hold_13_10 + 11 hold_13_11
   + 12 hold_13_12 + 13 hold_13_13 + 14 hold_13_14 + 15 hold_13_15
   + 16 hold_13_16 + 17 hold_13_17 + 18 hold_13_18
   + 60 pair_ok_13_14_for_nic_out_13_nic_in_14 <= 60
 constr_pair_ok2_13_14: - start_13 + start_14 - hold_13_1 - 2 hold_13_2
   - 3 hold_13_3 - 4 hold_13_4 - 5 hold_13_5 - 6 hold_13_6 - 7 hold_13_7
   - 8 hold_13_8 - 9 hold_13_9 - 10 hold_13_10 - 11 hold_13_11
   - 12 hold_13_12 - 13 hold_13_13 - 14 hold_13_14 - 15 hold_13_15
   - 16 hold_13_16 - 17 hold_13_17 - 18 hold_13_18
   + 60 pair_ok_13_14_for_nic_out_13_nic_in_14 <= 60
 constr_edge_switch_end_start_13_14_OR: switch_nic_out_13_nic_in_14
   - pair_ok_13_14_for_nic_out_13_nic_in_14 <= 0
 constr_pair_ok1_13_15: start_13 - start_15 + hold_13_1 + 2 hold_13_2
   + 3 hold_13_3 + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6 + 7 hold_13_7
   + 8 hold_13_8 + 9 hold_13_9 + 10 hold_13_10 + 11 hold_13_11
   + 12 hold_13_12 + 13 hold_13_13 + 14 hold_13_14 + 15 hold_13_15
   + 16 hold_13_16 + 17 hold_13_17 + 18 hold_13_18
   + 60 pair_ok_13_15_for_nic_out_13_nic_in_15 <= 60
 constr_pair_ok2_13_15: - start_13 + start_15 - hold_13_1 - 2 hold_13_2
   - 3 hold_13_3 - 4 hold_13_4 - 5 hold_13_5 - 6 hold_13_6 - 7 hold_13_7
   - 8 hold_13_8 - 9 hold_13_9 - 10 hold_13_10 - 11 hold_13_11
   - 12 hold_13_12 - 13 hold_13_13 - 14 hold_13_14 - 15 hold_13_15
   - 16 hold_13_16 - 17 hold_13_17 - 18 hold_13_18
   + 60 pair_ok_13_15_for_nic_out_13_nic_in_15 <= 60
 constr_edge_switch_end_start_13_15_OR: switch_nic_out_13_nic_in_15
   - pair_ok_13_15_for_nic_out_13_nic_in_15 <= 0
 constr_is_start_zero_14: start_14
   + 60 is_start_zero_14_for_source_nic_in_14 <= 60
 constr_edge_switch_start_zero_14_OR: switch_source_nic_in_14
   - is_start_zero_14_for_source_nic_in_14 <= 0
 constr_is_end_complete_14: start_14 + hold_14_1 + 2 hold_14_2
   + 3 hold_14_3 + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6 + 7 hold_14_7
   + 8 hold_14_8 + 9 hold_14_9 + 10 hold_14_10 + 11 hold_14_11
   + 12 hold_14_12 + 13 hold_14_13 + 14 hold_14_14 + 15 hold_14_15
   + 16 hold_14_16 + 17 hold_14_17 + 18 hold_14_18 - 60 is_end_complete_14
   >= 0
 constr_edge_switch_end_max_14_OR: switch_nic_out_14_sink
   - is_end_complete_14 <= 0
 constr_pair_ok1_14_0: - start_0 + start_14 + hold_14_1 + 2 hold_14_2
   + 3 hold_14_3 + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6 + 7 hold_14_7
   + 8 hold_14_8 + 9 hold_14_9 + 10 hold_14_10 + 11 hold_14_11
   + 12 hold_14_12 + 13 hold_14_13 + 14 hold_14_14 + 15 hold_14_15
   + 16 hold_14_16 + 17 hold_14_17 + 18 hold_14_18
   + 60 pair_ok_14_0_for_nic_out_14_nic_in_0 <= 60
 constr_pair_ok2_14_0: start_0 - start_14 - hold_14_1 - 2 hold_14_2
   - 3 hold_14_3 - 4 hold_14_4 - 5 hold_14_5 - 6 hold_14_6 - 7 hold_14_7
   - 8 hold_14_8 - 9 hold_14_9 - 10 hold_14_10 - 11 hold_14_11
   - 12 hold_14_12 - 13 hold_14_13 - 14 hold_14_14 - 15 hold_14_15
   - 16 hold_14_16 - 17 hold_14_17 - 18 hold_14_18
   + 60 pair_ok_14_0_for_nic_out_14_nic_in_0 <= 60
 constr_edge_switch_end_start_14_0_OR: switch_nic_out_14_nic_in_0
   - pair_ok_14_0_for_nic_out_14_nic_in_0 <= 0
 constr_pair_ok1_14_1: - start_1 + start_14 + hold_14_1 + 2 hold_14_2
   + 3 hold_14_3 + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6 + 7 hold_14_7
   + 8 hold_14_8 + 9 hold_14_9 + 10 hold_14_10 + 11 hold_14_11
   + 12 hold_14_12 + 13 hold_14_13 + 14 hold_14_14 + 15 hold_14_15
   + 16 hold_14_16 + 17 hold_14_17 + 18 hold_14_18
   + 60 pair_ok_14_1_for_nic_out_14_nic_in_1 <= 60
 constr_pair_ok2_14_1: start_1 - start_14 - hold_14_1 - 2 hold_14_2
   - 3 hold_14_3 - 4 hold_14_4 - 5 hold_14_5 - 6 hold_14_6 - 7 hold_14_7
   - 8 hold_14_8 - 9 hold_14_9 - 10 hold_14_10 - 11 hold_14_11
   - 12 hold_14_12 - 13 hold_14_13 - 14 hold_14_14 - 15 hold_14_15
   - 16 hold_14_16 - 17 hold_14_17 - 18 hold_14_18
   + 60 pair_ok_14_1_for_nic_out_14_nic_in_1 <= 60
 constr_edge_switch_end_start_14_1_OR: switch_nic_out_14_nic_in_1
   - pair_ok_14_1_for_nic_out_14_nic_in_1 <= 0
 constr_pair_ok1_14_2: - start_2 + start_14 + hold_14_1 + 2 hold_14_2
   + 3 hold_14_3 + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6 + 7 hold_14_7
   + 8 hold_14_8 + 9 hold_14_9 + 10 hold_14_10 + 11 hold_14_11
   + 12 hold_14_12 + 13 hold_14_13 + 14 hold_14_14 + 15 hold_14_15
   + 16 hold_14_16 + 17 hold_14_17 + 18 hold_14_18
   + 60 pair_ok_14_2_for_nic_out_14_nic_in_2 <= 60
 constr_pair_ok2_14_2: start_2 - start_14 - hold_14_1 - 2 hold_14_2
   - 3 hold_14_3 - 4 hold_14_4 - 5 hold_14_5 - 6 hold_14_6 - 7 hold_14_7
   - 8 hold_14_8 - 9 hold_14_9 - 10 hold_14_10 - 11 hold_14_11
   - 12 hold_14_12 - 13 hold_14_13 - 14 hold_14_14 - 15 hold_14_15
   - 16 hold_14_16 - 17 hold_14_17 - 18 hold_14_18
   + 60 pair_ok_14_2_for_nic_out_14_nic_in_2 <= 60
 constr_edge_switch_end_start_14_2_OR: switch_nic_out_14_nic_in_2
   - pair_ok_14_2_for_nic_out_14_nic_in_2 <= 0
 constr_pair_ok1_14_3: - start_3 + start_14 + hold_14_1 + 2 hold_14_2
   + 3 hold_14_3 + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6 + 7 hold_14_7
   + 8 hold_14_8 + 9 hold_14_9 + 10 hold_14_10 + 11 hold_14_11
   + 12 hold_14_12 + 13 hold_14_13 + 14 hold_14_14 + 15 hold_14_15
   + 16 hold_14_16 + 17 hold_14_17 + 18 hold_14_18
   + 60 pair_ok_14_3_for_nic_out_14_nic_in_3 <= 60
 constr_pair_ok2_14_3: start_3 - start_14 - hold_14_1 - 2 hold_14_2
   - 3 hold_14_3 - 4 hold_14_4 - 5 hold_14_5 - 6 hold_14_6 - 7 hold_14_7
   - 8 hold_14_8 - 9 hold_14_9 - 10 hold_14_10 - 11 hold_14_11
   - 12 hold_14_12 - 13 hold_14_13 - 14 hold_14_14 - 15 hold_14_15
   - 16 hold_14_16 - 17 hold_14_17 - 18 hold_14_18
   + 60 pair_ok_14_3_for_nic_out_14_nic_in_3 <= 60
 constr_edge_switch_end_start_14_3_OR: switch_nic_out_14_nic_in_3
   - pair_ok_14_3_for_nic_out_14_nic_in_3 <= 0
 constr_pair_ok1_14_4: - start_4 + start_14 + hold_14_1 + 2 hold_14_2
   + 3 hold_14_3 + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6 + 7 hold_14_7
   + 8 hold_14_8 + 9 hold_14_9 + 10 hold_14_10 + 11 hold_14_11
   + 12 hold_14_12 + 13 hold_14_13 + 14 hold_14_14 + 15 hold_14_15
   + 16 hold_14_16 + 17 hold_14_17 + 18 hold_14_18
   + 60 pair_ok_14_4_for_nic_out_14_nic_in_4 <= 60
 constr_pair_ok2_14_4: start_4 - start_14 - hold_14_1 - 2 hold_14_2
   - 3 hold_14_3 - 4 hold_14_4 - 5 hold_14_5 - 6 hold_14_6 - 7 hold_14_7
   - 8 hold_14_8 - 9 hold_14_9 - 10 hold_14_10 - 11 hold_14_11
   - 12 hold_14_12 - 13 hold_14_13 - 14 hold_14_14 - 15 hold_14_15
   - 16 hold_14_16 - 17 hold_14_17 - 18 hold_14_18
   + 60 pair_ok_14_4_for_nic_out_14_nic_in_4 <= 60
 constr_edge_switch_end_start_14_4_OR: switch_nic_out_14_nic_in_4
   - pair_ok_14_4_for_nic_out_14_nic_in_4 <= 0
 constr_pair_ok1_14_5: - start_5 + start_14 + hold_14_1 + 2 hold_14_2
   + 3 hold_14_3 + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6 + 7 hold_14_7
   + 8 hold_14_8 + 9 hold_14_9 + 10 hold_14_10 + 11 hold_14_11
   + 12 hold_14_12 + 13 hold_14_13 + 14 hold_14_14 + 15 hold_14_15
   + 16 hold_14_16 + 17 hold_14_17 + 18 hold_14_18
   + 60 pair_ok_14_5_for_nic_out_14_nic_in_5 <= 60
 constr_pair_ok2_14_5: start_5 - start_14 - hold_14_1 - 2 hold_14_2
   - 3 hold_14_3 - 4 hold_14_4 - 5 hold_14_5 - 6 hold_14_6 - 7 hold_14_7
   - 8 hold_14_8 - 9 hold_14_9 - 10 hold_14_10 - 11 hold_14_11
   - 12 hold_14_12 - 13 hold_14_13 - 14 hold_14_14 - 15 hold_14_15
   - 16 hold_14_16 - 17 hold_14_17 - 18 hold_14_18
   + 60 pair_ok_14_5_for_nic_out_14_nic_in_5 <= 60
 constr_edge_switch_end_start_14_5_OR: switch_nic_out_14_nic_in_5
   - pair_ok_14_5_for_nic_out_14_nic_in_5 <= 0
 constr_pair_ok1_14_6: - start_6 + start_14 + hold_14_1 + 2 hold_14_2
   + 3 hold_14_3 + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6 + 7 hold_14_7
   + 8 hold_14_8 + 9 hold_14_9 + 10 hold_14_10 + 11 hold_14_11
   + 12 hold_14_12 + 13 hold_14_13 + 14 hold_14_14 + 15 hold_14_15
   + 16 hold_14_16 + 17 hold_14_17 + 18 hold_14_18
   + 60 pair_ok_14_6_for_nic_out_14_nic_in_6 <= 60
 constr_pair_ok2_14_6: start_6 - start_14 - hold_14_1 - 2 hold_14_2
   - 3 hold_14_3 - 4 hold_14_4 - 5 hold_14_5 - 6 hold_14_6 - 7 hold_14_7
   - 8 hold_14_8 - 9 hold_14_9 - 10 hold_14_10 - 11 hold_14_11
   - 12 hold_14_12 - 13 hold_14_13 - 14 hold_14_14 - 15 hold_14_15
   - 16 hold_14_16 - 17 hold_14_17 - 18 hold_14_18
   + 60 pair_ok_14_6_for_nic_out_14_nic_in_6 <= 60
 constr_edge_switch_end_start_14_6_OR: switch_nic_out_14_nic_in_6
   - pair_ok_14_6_for_nic_out_14_nic_in_6 <= 0
 constr_pair_ok1_14_7: - start_7 + start_14 + hold_14_1 + 2 hold_14_2
   + 3 hold_14_3 + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6 + 7 hold_14_7
   + 8 hold_14_8 + 9 hold_14_9 + 10 hold_14_10 + 11 hold_14_11
   + 12 hold_14_12 + 13 hold_14_13 + 14 hold_14_14 + 15 hold_14_15
   + 16 hold_14_16 + 17 hold_14_17 + 18 hold_14_18
   + 60 pair_ok_14_7_for_nic_out_14_nic_in_7 <= 60
 constr_pair_ok2_14_7: start_7 - start_14 - hold_14_1 - 2 hold_14_2
   - 3 hold_14_3 - 4 hold_14_4 - 5 hold_14_5 - 6 hold_14_6 - 7 hold_14_7
   - 8 hold_14_8 - 9 hold_14_9 - 10 hold_14_10 - 11 hold_14_11
   - 12 hold_14_12 - 13 hold_14_13 - 14 hold_14_14 - 15 hold_14_15
   - 16 hold_14_16 - 17 hold_14_17 - 18 hold_14_18
   + 60 pair_ok_14_7_for_nic_out_14_nic_in_7 <= 60
 constr_edge_switch_end_start_14_7_OR: switch_nic_out_14_nic_in_7
   - pair_ok_14_7_for_nic_out_14_nic_in_7 <= 0
 constr_pair_ok1_14_8: - start_8 + start_14 + hold_14_1 + 2 hold_14_2
   + 3 hold_14_3 + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6 + 7 hold_14_7
   + 8 hold_14_8 + 9 hold_14_9 + 10 hold_14_10 + 11 hold_14_11
   + 12 hold_14_12 + 13 hold_14_13 + 14 hold_14_14 + 15 hold_14_15
   + 16 hold_14_16 + 17 hold_14_17 + 18 hold_14_18
   + 60 pair_ok_14_8_for_nic_out_14_nic_in_8 <= 60
 constr_pair_ok2_14_8: start_8 - start_14 - hold_14_1 - 2 hold_14_2
   - 3 hold_14_3 - 4 hold_14_4 - 5 hold_14_5 - 6 hold_14_6 - 7 hold_14_7
   - 8 hold_14_8 - 9 hold_14_9 - 10 hold_14_10 - 11 hold_14_11
   - 12 hold_14_12 - 13 hold_14_13 - 14 hold_14_14 - 15 hold_14_15
   - 16 hold_14_16 - 17 hold_14_17 - 18 hold_14_18
   + 60 pair_ok_14_8_for_nic_out_14_nic_in_8 <= 60
 constr_edge_switch_end_start_14_8_OR: switch_nic_out_14_nic_in_8
   - pair_ok_14_8_for_nic_out_14_nic_in_8 <= 0
 constr_pair_ok1_14_9: - start_9 + start_14 + hold_14_1 + 2 hold_14_2
   + 3 hold_14_3 + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6 + 7 hold_14_7
   + 8 hold_14_8 + 9 hold_14_9 + 10 hold_14_10 + 11 hold_14_11
   + 12 hold_14_12 + 13 hold_14_13 + 14 hold_14_14 + 15 hold_14_15
   + 16 hold_14_16 + 17 hold_14_17 + 18 hold_14_18
   + 60 pair_ok_14_9_for_nic_out_14_nic_in_9 <= 60
 constr_pair_ok2_14_9: start_9 - start_14 - hold_14_1 - 2 hold_14_2
   - 3 hold_14_3 - 4 hold_14_4 - 5 hold_14_5 - 6 hold_14_6 - 7 hold_14_7
   - 8 hold_14_8 - 9 hold_14_9 - 10 hold_14_10 - 11 hold_14_11
   - 12 hold_14_12 - 13 hold_14_13 - 14 hold_14_14 - 15 hold_14_15
   - 16 hold_14_16 - 17 hold_14_17 - 18 hold_14_18
   + 60 pair_ok_14_9_for_nic_out_14_nic_in_9 <= 60
 constr_edge_switch_end_start_14_9_OR: switch_nic_out_14_nic_in_9
   - pair_ok_14_9_for_nic_out_14_nic_in_9 <= 0
 constr_pair_ok1_14_10: - start_10 + start_14 + hold_14_1 + 2 hold_14_2
   + 3 hold_14_3 + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6 + 7 hold_14_7
   + 8 hold_14_8 + 9 hold_14_9 + 10 hold_14_10 + 11 hold_14_11
   + 12 hold_14_12 + 13 hold_14_13 + 14 hold_14_14 + 15 hold_14_15
   + 16 hold_14_16 + 17 hold_14_17 + 18 hold_14_18
   + 60 pair_ok_14_10_for_nic_out_14_nic_in_10 <= 60
 constr_pair_ok2_14_10: start_10 - start_14 - hold_14_1 - 2 hold_14_2
   - 3 hold_14_3 - 4 hold_14_4 - 5 hold_14_5 - 6 hold_14_6 - 7 hold_14_7
   - 8 hold_14_8 - 9 hold_14_9 - 10 hold_14_10 - 11 hold_14_11
   - 12 hold_14_12 - 13 hold_14_13 - 14 hold_14_14 - 15 hold_14_15
   - 16 hold_14_16 - 17 hold_14_17 - 18 hold_14_18
   + 60 pair_ok_14_10_for_nic_out_14_nic_in_10 <= 60
 constr_edge_switch_end_start_14_10_OR: switch_nic_out_14_nic_in_10
   - pair_ok_14_10_for_nic_out_14_nic_in_10 <= 0
 constr_pair_ok1_14_11: - start_11 + start_14 + hold_14_1 + 2 hold_14_2
   + 3 hold_14_3 + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6 + 7 hold_14_7
   + 8 hold_14_8 + 9 hold_14_9 + 10 hold_14_10 + 11 hold_14_11
   + 12 hold_14_12 + 13 hold_14_13 + 14 hold_14_14 + 15 hold_14_15
   + 16 hold_14_16 + 17 hold_14_17 + 18 hold_14_18
   + 60 pair_ok_14_11_for_nic_out_14_nic_in_11 <= 60
 constr_pair_ok2_14_11: start_11 - start_14 - hold_14_1 - 2 hold_14_2
   - 3 hold_14_3 - 4 hold_14_4 - 5 hold_14_5 - 6 hold_14_6 - 7 hold_14_7
   - 8 hold_14_8 - 9 hold_14_9 - 10 hold_14_10 - 11 hold_14_11
   - 12 hold_14_12 - 13 hold_14_13 - 14 hold_14_14 - 15 hold_14_15
   - 16 hold_14_16 - 17 hold_14_17 - 18 hold_14_18
   + 60 pair_ok_14_11_for_nic_out_14_nic_in_11 <= 60
 constr_edge_switch_end_start_14_11_OR: switch_nic_out_14_nic_in_11
   - pair_ok_14_11_for_nic_out_14_nic_in_11 <= 0
 constr_pair_ok1_14_12: - start_12 + start_14 + hold_14_1 + 2 hold_14_2
   + 3 hold_14_3 + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6 + 7 hold_14_7
   + 8 hold_14_8 + 9 hold_14_9 + 10 hold_14_10 + 11 hold_14_11
   + 12 hold_14_12 + 13 hold_14_13 + 14 hold_14_14 + 15 hold_14_15
   + 16 hold_14_16 + 17 hold_14_17 + 18 hold_14_18
   + 60 pair_ok_14_12_for_nic_out_14_nic_in_12 <= 60
 constr_pair_ok2_14_12: start_12 - start_14 - hold_14_1 - 2 hold_14_2
   - 3 hold_14_3 - 4 hold_14_4 - 5 hold_14_5 - 6 hold_14_6 - 7 hold_14_7
   - 8 hold_14_8 - 9 hold_14_9 - 10 hold_14_10 - 11 hold_14_11
   - 12 hold_14_12 - 13 hold_14_13 - 14 hold_14_14 - 15 hold_14_15
   - 16 hold_14_16 - 17 hold_14_17 - 18 hold_14_18
   + 60 pair_ok_14_12_for_nic_out_14_nic_in_12 <= 60
 constr_edge_switch_end_start_14_12_OR: switch_nic_out_14_nic_in_12
   - pair_ok_14_12_for_nic_out_14_nic_in_12 <= 0
 constr_pair_ok1_14_13: - start_13 + start_14 + hold_14_1 + 2 hold_14_2
   + 3 hold_14_3 + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6 + 7 hold_14_7
   + 8 hold_14_8 + 9 hold_14_9 + 10 hold_14_10 + 11 hold_14_11
   + 12 hold_14_12 + 13 hold_14_13 + 14 hold_14_14 + 15 hold_14_15
   + 16 hold_14_16 + 17 hold_14_17 + 18 hold_14_18
   + 60 pair_ok_14_13_for_nic_out_14_nic_in_13 <= 60
 constr_pair_ok2_14_13: start_13 - start_14 - hold_14_1 - 2 hold_14_2
   - 3 hold_14_3 - 4 hold_14_4 - 5 hold_14_5 - 6 hold_14_6 - 7 hold_14_7
   - 8 hold_14_8 - 9 hold_14_9 - 10 hold_14_10 - 11 hold_14_11
   - 12 hold_14_12 - 13 hold_14_13 - 14 hold_14_14 - 15 hold_14_15
   - 16 hold_14_16 - 17 hold_14_17 - 18 hold_14_18
   + 60 pair_ok_14_13_for_nic_out_14_nic_in_13 <= 60
 constr_edge_switch_end_start_14_13_OR: switch_nic_out_14_nic_in_13
   - pair_ok_14_13_for_nic_out_14_nic_in_13 <= 0
 constr_pair_ok1_14_15: start_14 - start_15 + hold_14_1 + 2 hold_14_2
   + 3 hold_14_3 + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6 + 7 hold_14_7
   + 8 hold_14_8 + 9 hold_14_9 + 10 hold_14_10 + 11 hold_14_11
   + 12 hold_14_12 + 13 hold_14_13 + 14 hold_14_14 + 15 hold_14_15
   + 16 hold_14_16 + 17 hold_14_17 + 18 hold_14_18
   + 60 pair_ok_14_15_for_nic_out_14_nic_in_15 <= 60
 constr_pair_ok2_14_15: - start_14 + start_15 - hold_14_1 - 2 hold_14_2
   - 3 hold_14_3 - 4 hold_14_4 - 5 hold_14_5 - 6 hold_14_6 - 7 hold_14_7
   - 8 hold_14_8 - 9 hold_14_9 - 10 hold_14_10 - 11 hold_14_11
   - 12 hold_14_12 - 13 hold_14_13 - 14 hold_14_14 - 15 hold_14_15
   - 16 hold_14_16 - 17 hold_14_17 - 18 hold_14_18
   + 60 pair_ok_14_15_for_nic_out_14_nic_in_15 <= 60
 constr_edge_switch_end_start_14_15_OR: switch_nic_out_14_nic_in_15
   - pair_ok_14_15_for_nic_out_14_nic_in_15 <= 0
 constr_is_start_zero_15: start_15
   + 60 is_start_zero_15_for_source_nic_in_15 <= 60
 constr_edge_switch_start_zero_15_OR: switch_source_nic_in_15
   - is_start_zero_15_for_source_nic_in_15 <= 0
 constr_is_end_complete_15: start_15 + hold_15_1 + 2 hold_15_2
   + 3 hold_15_3 + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6 + 7 hold_15_7
   + 8 hold_15_8 + 9 hold_15_9 + 10 hold_15_10 + 11 hold_15_11
   + 12 hold_15_12 + 13 hold_15_13 + 14 hold_15_14 + 15 hold_15_15
   + 16 hold_15_16 + 17 hold_15_17 + 18 hold_15_18 - 60 is_end_complete_15
   >= 0
 constr_edge_switch_end_max_15_OR: switch_nic_out_15_sink
   - is_end_complete_15 <= 0
 constr_pair_ok1_15_0: - start_0 + start_15 + hold_15_1 + 2 hold_15_2
   + 3 hold_15_3 + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6 + 7 hold_15_7
   + 8 hold_15_8 + 9 hold_15_9 + 10 hold_15_10 + 11 hold_15_11
   + 12 hold_15_12 + 13 hold_15_13 + 14 hold_15_14 + 15 hold_15_15
   + 16 hold_15_16 + 17 hold_15_17 + 18 hold_15_18
   + 60 pair_ok_15_0_for_nic_out_15_nic_in_0 <= 60
 constr_pair_ok2_15_0: start_0 - start_15 - hold_15_1 - 2 hold_15_2
   - 3 hold_15_3 - 4 hold_15_4 - 5 hold_15_5 - 6 hold_15_6 - 7 hold_15_7
   - 8 hold_15_8 - 9 hold_15_9 - 10 hold_15_10 - 11 hold_15_11
   - 12 hold_15_12 - 13 hold_15_13 - 14 hold_15_14 - 15 hold_15_15
   - 16 hold_15_16 - 17 hold_15_17 - 18 hold_15_18
   + 60 pair_ok_15_0_for_nic_out_15_nic_in_0 <= 60
 constr_edge_switch_end_start_15_0_OR: switch_nic_out_15_nic_in_0
   - pair_ok_15_0_for_nic_out_15_nic_in_0 <= 0
 constr_pair_ok1_15_1: - start_1 + start_15 + hold_15_1 + 2 hold_15_2
   + 3 hold_15_3 + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6 + 7 hold_15_7
   + 8 hold_15_8 + 9 hold_15_9 + 10 hold_15_10 + 11 hold_15_11
   + 12 hold_15_12 + 13 hold_15_13 + 14 hold_15_14 + 15 hold_15_15
   + 16 hold_15_16 + 17 hold_15_17 + 18 hold_15_18
   + 60 pair_ok_15_1_for_nic_out_15_nic_in_1 <= 60
 constr_pair_ok2_15_1: start_1 - start_15 - hold_15_1 - 2 hold_15_2
   - 3 hold_15_3 - 4 hold_15_4 - 5 hold_15_5 - 6 hold_15_6 - 7 hold_15_7
   - 8 hold_15_8 - 9 hold_15_9 - 10 hold_15_10 - 11 hold_15_11
   - 12 hold_15_12 - 13 hold_15_13 - 14 hold_15_14 - 15 hold_15_15
   - 16 hold_15_16 - 17 hold_15_17 - 18 hold_15_18
   + 60 pair_ok_15_1_for_nic_out_15_nic_in_1 <= 60
 constr_edge_switch_end_start_15_1_OR: switch_nic_out_15_nic_in_1
   - pair_ok_15_1_for_nic_out_15_nic_in_1 <= 0
 constr_pair_ok1_15_2: - start_2 + start_15 + hold_15_1 + 2 hold_15_2
   + 3 hold_15_3 + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6 + 7 hold_15_7
   + 8 hold_15_8 + 9 hold_15_9 + 10 hold_15_10 + 11 hold_15_11
   + 12 hold_15_12 + 13 hold_15_13 + 14 hold_15_14 + 15 hold_15_15
   + 16 hold_15_16 + 17 hold_15_17 + 18 hold_15_18
   + 60 pair_ok_15_2_for_nic_out_15_nic_in_2 <= 60
 constr_pair_ok2_15_2: start_2 - start_15 - hold_15_1 - 2 hold_15_2
   - 3 hold_15_3 - 4 hold_15_4 - 5 hold_15_5 - 6 hold_15_6 - 7 hold_15_7
   - 8 hold_15_8 - 9 hold_15_9 - 10 hold_15_10 - 11 hold_15_11
   - 12 hold_15_12 - 13 hold_15_13 - 14 hold_15_14 - 15 hold_15_15
   - 16 hold_15_16 - 17 hold_15_17 - 18 hold_15_18
   + 60 pair_ok_15_2_for_nic_out_15_nic_in_2 <= 60
 constr_edge_switch_end_start_15_2_OR: switch_nic_out_15_nic_in_2
   - pair_ok_15_2_for_nic_out_15_nic_in_2 <= 0
 constr_pair_ok1_15_3: - start_3 + start_15 + hold_15_1 + 2 hold_15_2
   + 3 hold_15_3 + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6 + 7 hold_15_7
   + 8 hold_15_8 + 9 hold_15_9 + 10 hold_15_10 + 11 hold_15_11
   + 12 hold_15_12 + 13 hold_15_13 + 14 hold_15_14 + 15 hold_15_15
   + 16 hold_15_16 + 17 hold_15_17 + 18 hold_15_18
   + 60 pair_ok_15_3_for_nic_out_15_nic_in_3 <= 60
 constr_pair_ok2_15_3: start_3 - start_15 - hold_15_1 - 2 hold_15_2
   - 3 hold_15_3 - 4 hold_15_4 - 5 hold_15_5 - 6 hold_15_6 - 7 hold_15_7
   - 8 hold_15_8 - 9 hold_15_9 - 10 hold_15_10 - 11 hold_15_11
   - 12 hold_15_12 - 13 hold_15_13 - 14 hold_15_14 - 15 hold_15_15
   - 16 hold_15_16 - 17 hold_15_17 - 18 hold_15_18
   + 60 pair_ok_15_3_for_nic_out_15_nic_in_3 <= 60
 constr_edge_switch_end_start_15_3_OR: switch_nic_out_15_nic_in_3
   - pair_ok_15_3_for_nic_out_15_nic_in_3 <= 0
 constr_pair_ok1_15_4: - start_4 + start_15 + hold_15_1 + 2 hold_15_2
   + 3 hold_15_3 + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6 + 7 hold_15_7
   + 8 hold_15_8 + 9 hold_15_9 + 10 hold_15_10 + 11 hold_15_11
   + 12 hold_15_12 + 13 hold_15_13 + 14 hold_15_14 + 15 hold_15_15
   + 16 hold_15_16 + 17 hold_15_17 + 18 hold_15_18
   + 60 pair_ok_15_4_for_nic_out_15_nic_in_4 <= 60
 constr_pair_ok2_15_4: start_4 - start_15 - hold_15_1 - 2 hold_15_2
   - 3 hold_15_3 - 4 hold_15_4 - 5 hold_15_5 - 6 hold_15_6 - 7 hold_15_7
   - 8 hold_15_8 - 9 hold_15_9 - 10 hold_15_10 - 11 hold_15_11
   - 12 hold_15_12 - 13 hold_15_13 - 14 hold_15_14 - 15 hold_15_15
   - 16 hold_15_16 - 17 hold_15_17 - 18 hold_15_18
   + 60 pair_ok_15_4_for_nic_out_15_nic_in_4 <= 60
 constr_edge_switch_end_start_15_4_OR: switch_nic_out_15_nic_in_4
   - pair_ok_15_4_for_nic_out_15_nic_in_4 <= 0
 constr_pair_ok1_15_5: - start_5 + start_15 + hold_15_1 + 2 hold_15_2
   + 3 hold_15_3 + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6 + 7 hold_15_7
   + 8 hold_15_8 + 9 hold_15_9 + 10 hold_15_10 + 11 hold_15_11
   + 12 hold_15_12 + 13 hold_15_13 + 14 hold_15_14 + 15 hold_15_15
   + 16 hold_15_16 + 17 hold_15_17 + 18 hold_15_18
   + 60 pair_ok_15_5_for_nic_out_15_nic_in_5 <= 60
 constr_pair_ok2_15_5: start_5 - start_15 - hold_15_1 - 2 hold_15_2
   - 3 hold_15_3 - 4 hold_15_4 - 5 hold_15_5 - 6 hold_15_6 - 7 hold_15_7
   - 8 hold_15_8 - 9 hold_15_9 - 10 hold_15_10 - 11 hold_15_11
   - 12 hold_15_12 - 13 hold_15_13 - 14 hold_15_14 - 15 hold_15_15
   - 16 hold_15_16 - 17 hold_15_17 - 18 hold_15_18
   + 60 pair_ok_15_5_for_nic_out_15_nic_in_5 <= 60
 constr_edge_switch_end_start_15_5_OR: switch_nic_out_15_nic_in_5
   - pair_ok_15_5_for_nic_out_15_nic_in_5 <= 0
 constr_pair_ok1_15_6: - start_6 + start_15 + hold_15_1 + 2 hold_15_2
   + 3 hold_15_3 + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6 + 7 hold_15_7
   + 8 hold_15_8 + 9 hold_15_9 + 10 hold_15_10 + 11 hold_15_11
   + 12 hold_15_12 + 13 hold_15_13 + 14 hold_15_14 + 15 hold_15_15
   + 16 hold_15_16 + 17 hold_15_17 + 18 hold_15_18
   + 60 pair_ok_15_6_for_nic_out_15_nic_in_6 <= 60
 constr_pair_ok2_15_6: start_6 - start_15 - hold_15_1 - 2 hold_15_2
   - 3 hold_15_3 - 4 hold_15_4 - 5 hold_15_5 - 6 hold_15_6 - 7 hold_15_7
   - 8 hold_15_8 - 9 hold_15_9 - 10 hold_15_10 - 11 hold_15_11
   - 12 hold_15_12 - 13 hold_15_13 - 14 hold_15_14 - 15 hold_15_15
   - 16 hold_15_16 - 17 hold_15_17 - 18 hold_15_18
   + 60 pair_ok_15_6_for_nic_out_15_nic_in_6 <= 60
 constr_edge_switch_end_start_15_6_OR: switch_nic_out_15_nic_in_6
   - pair_ok_15_6_for_nic_out_15_nic_in_6 <= 0
 constr_pair_ok1_15_7: - start_7 + start_15 + hold_15_1 + 2 hold_15_2
   + 3 hold_15_3 + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6 + 7 hold_15_7
   + 8 hold_15_8 + 9 hold_15_9 + 10 hold_15_10 + 11 hold_15_11
   + 12 hold_15_12 + 13 hold_15_13 + 14 hold_15_14 + 15 hold_15_15
   + 16 hold_15_16 + 17 hold_15_17 + 18 hold_15_18
   + 60 pair_ok_15_7_for_nic_out_15_nic_in_7 <= 60
 constr_pair_ok2_15_7: start_7 - start_15 - hold_15_1 - 2 hold_15_2
   - 3 hold_15_3 - 4 hold_15_4 - 5 hold_15_5 - 6 hold_15_6 - 7 hold_15_7
   - 8 hold_15_8 - 9 hold_15_9 - 10 hold_15_10 - 11 hold_15_11
   - 12 hold_15_12 - 13 hold_15_13 - 14 hold_15_14 - 15 hold_15_15
   - 16 hold_15_16 - 17 hold_15_17 - 18 hold_15_18
   + 60 pair_ok_15_7_for_nic_out_15_nic_in_7 <= 60
 constr_edge_switch_end_start_15_7_OR: switch_nic_out_15_nic_in_7
   - pair_ok_15_7_for_nic_out_15_nic_in_7 <= 0
 constr_pair_ok1_15_8: - start_8 + start_15 + hold_15_1 + 2 hold_15_2
   + 3 hold_15_3 + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6 + 7 hold_15_7
   + 8 hold_15_8 + 9 hold_15_9 + 10 hold_15_10 + 11 hold_15_11
   + 12 hold_15_12 + 13 hold_15_13 + 14 hold_15_14 + 15 hold_15_15
   + 16 hold_15_16 + 17 hold_15_17 + 18 hold_15_18
   + 60 pair_ok_15_8_for_nic_out_15_nic_in_8 <= 60
 constr_pair_ok2_15_8: start_8 - start_15 - hold_15_1 - 2 hold_15_2
   - 3 hold_15_3 - 4 hold_15_4 - 5 hold_15_5 - 6 hold_15_6 - 7 hold_15_7
   - 8 hold_15_8 - 9 hold_15_9 - 10 hold_15_10 - 11 hold_15_11
   - 12 hold_15_12 - 13 hold_15_13 - 14 hold_15_14 - 15 hold_15_15
   - 16 hold_15_16 - 17 hold_15_17 - 18 hold_15_18
   + 60 pair_ok_15_8_for_nic_out_15_nic_in_8 <= 60
 constr_edge_switch_end_start_15_8_OR: switch_nic_out_15_nic_in_8
   - pair_ok_15_8_for_nic_out_15_nic_in_8 <= 0
 constr_pair_ok1_15_9: - start_9 + start_15 + hold_15_1 + 2 hold_15_2
   + 3 hold_15_3 + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6 + 7 hold_15_7
   + 8 hold_15_8 + 9 hold_15_9 + 10 hold_15_10 + 11 hold_15_11
   + 12 hold_15_12 + 13 hold_15_13 + 14 hold_15_14 + 15 hold_15_15
   + 16 hold_15_16 + 17 hold_15_17 + 18 hold_15_18
   + 60 pair_ok_15_9_for_nic_out_15_nic_in_9 <= 60
 constr_pair_ok2_15_9: start_9 - start_15 - hold_15_1 - 2 hold_15_2
   - 3 hold_15_3 - 4 hold_15_4 - 5 hold_15_5 - 6 hold_15_6 - 7 hold_15_7
   - 8 hold_15_8 - 9 hold_15_9 - 10 hold_15_10 - 11 hold_15_11
   - 12 hold_15_12 - 13 hold_15_13 - 14 hold_15_14 - 15 hold_15_15
   - 16 hold_15_16 - 17 hold_15_17 - 18 hold_15_18
   + 60 pair_ok_15_9_for_nic_out_15_nic_in_9 <= 60
 constr_edge_switch_end_start_15_9_OR: switch_nic_out_15_nic_in_9
   - pair_ok_15_9_for_nic_out_15_nic_in_9 <= 0
 constr_pair_ok1_15_10: - start_10 + start_15 + hold_15_1 + 2 hold_15_2
   + 3 hold_15_3 + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6 + 7 hold_15_7
   + 8 hold_15_8 + 9 hold_15_9 + 10 hold_15_10 + 11 hold_15_11
   + 12 hold_15_12 + 13 hold_15_13 + 14 hold_15_14 + 15 hold_15_15
   + 16 hold_15_16 + 17 hold_15_17 + 18 hold_15_18
   + 60 pair_ok_15_10_for_nic_out_15_nic_in_10 <= 60
 constr_pair_ok2_15_10: start_10 - start_15 - hold_15_1 - 2 hold_15_2
   - 3 hold_15_3 - 4 hold_15_4 - 5 hold_15_5 - 6 hold_15_6 - 7 hold_15_7
   - 8 hold_15_8 - 9 hold_15_9 - 10 hold_15_10 - 11 hold_15_11
   - 12 hold_15_12 - 13 hold_15_13 - 14 hold_15_14 - 15 hold_15_15
   - 16 hold_15_16 - 17 hold_15_17 - 18 hold_15_18
   + 60 pair_ok_15_10_for_nic_out_15_nic_in_10 <= 60
 constr_edge_switch_end_start_15_10_OR: switch_nic_out_15_nic_in_10
   - pair_ok_15_10_for_nic_out_15_nic_in_10 <= 0
 constr_pair_ok1_15_11: - start_11 + start_15 + hold_15_1 + 2 hold_15_2
   + 3 hold_15_3 + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6 + 7 hold_15_7
   + 8 hold_15_8 + 9 hold_15_9 + 10 hold_15_10 + 11 hold_15_11
   + 12 hold_15_12 + 13 hold_15_13 + 14 hold_15_14 + 15 hold_15_15
   + 16 hold_15_16 + 17 hold_15_17 + 18 hold_15_18
   + 60 pair_ok_15_11_for_nic_out_15_nic_in_11 <= 60
 constr_pair_ok2_15_11: start_11 - start_15 - hold_15_1 - 2 hold_15_2
   - 3 hold_15_3 - 4 hold_15_4 - 5 hold_15_5 - 6 hold_15_6 - 7 hold_15_7
   - 8 hold_15_8 - 9 hold_15_9 - 10 hold_15_10 - 11 hold_15_11
   - 12 hold_15_12 - 13 hold_15_13 - 14 hold_15_14 - 15 hold_15_15
   - 16 hold_15_16 - 17 hold_15_17 - 18 hold_15_18
   + 60 pair_ok_15_11_for_nic_out_15_nic_in_11 <= 60
 constr_edge_switch_end_start_15_11_OR: switch_nic_out_15_nic_in_11
   - pair_ok_15_11_for_nic_out_15_nic_in_11 <= 0
 constr_pair_ok1_15_12: - start_12 + start_15 + hold_15_1 + 2 hold_15_2
   + 3 hold_15_3 + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6 + 7 hold_15_7
   + 8 hold_15_8 + 9 hold_15_9 + 10 hold_15_10 + 11 hold_15_11
   + 12 hold_15_12 + 13 hold_15_13 + 14 hold_15_14 + 15 hold_15_15
   + 16 hold_15_16 + 17 hold_15_17 + 18 hold_15_18
   + 60 pair_ok_15_12_for_nic_out_15_nic_in_12 <= 60
 constr_pair_ok2_15_12: start_12 - start_15 - hold_15_1 - 2 hold_15_2
   - 3 hold_15_3 - 4 hold_15_4 - 5 hold_15_5 - 6 hold_15_6 - 7 hold_15_7
   - 8 hold_15_8 - 9 hold_15_9 - 10 hold_15_10 - 11 hold_15_11
   - 12 hold_15_12 - 13 hold_15_13 - 14 hold_15_14 - 15 hold_15_15
   - 16 hold_15_16 - 17 hold_15_17 - 18 hold_15_18
   + 60 pair_ok_15_12_for_nic_out_15_nic_in_12 <= 60
 constr_edge_switch_end_start_15_12_OR: switch_nic_out_15_nic_in_12
   - pair_ok_15_12_for_nic_out_15_nic_in_12 <= 0
 constr_pair_ok1_15_13: - start_13 + start_15 + hold_15_1 + 2 hold_15_2
   + 3 hold_15_3 + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6 + 7 hold_15_7
   + 8 hold_15_8 + 9 hold_15_9 + 10 hold_15_10 + 11 hold_15_11
   + 12 hold_15_12 + 13 hold_15_13 + 14 hold_15_14 + 15 hold_15_15
   + 16 hold_15_16 + 17 hold_15_17 + 18 hold_15_18
   + 60 pair_ok_15_13_for_nic_out_15_nic_in_13 <= 60
 constr_pair_ok2_15_13: start_13 - start_15 - hold_15_1 - 2 hold_15_2
   - 3 hold_15_3 - 4 hold_15_4 - 5 hold_15_5 - 6 hold_15_6 - 7 hold_15_7
   - 8 hold_15_8 - 9 hold_15_9 - 10 hold_15_10 - 11 hold_15_11
   - 12 hold_15_12 - 13 hold_15_13 - 14 hold_15_14 - 15 hold_15_15
   - 16 hold_15_16 - 17 hold_15_17 - 18 hold_15_18
   + 60 pair_ok_15_13_for_nic_out_15_nic_in_13 <= 60
 constr_edge_switch_end_start_15_13_OR: switch_nic_out_15_nic_in_13
   - pair_ok_15_13_for_nic_out_15_nic_in_13 <= 0
 constr_pair_ok1_15_14: - start_14 + start_15 + hold_15_1 + 2 hold_15_2
   + 3 hold_15_3 + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6 + 7 hold_15_7
   + 8 hold_15_8 + 9 hold_15_9 + 10 hold_15_10 + 11 hold_15_11
   + 12 hold_15_12 + 13 hold_15_13 + 14 hold_15_14 + 15 hold_15_15
   + 16 hold_15_16 + 17 hold_15_17 + 18 hold_15_18
   + 60 pair_ok_15_14_for_nic_out_15_nic_in_14 <= 60
 constr_pair_ok2_15_14: start_14 - start_15 - hold_15_1 - 2 hold_15_2
   - 3 hold_15_3 - 4 hold_15_4 - 5 hold_15_5 - 6 hold_15_6 - 7 hold_15_7
   - 8 hold_15_8 - 9 hold_15_9 - 10 hold_15_10 - 11 hold_15_11
   - 12 hold_15_12 - 13 hold_15_13 - 14 hold_15_14 - 15 hold_15_15
   - 16 hold_15_16 - 17 hold_15_17 - 18 hold_15_18
   + 60 pair_ok_15_14_for_nic_out_15_nic_in_14 <= 60
 constr_edge_switch_end_start_15_14_OR: switch_nic_out_15_nic_in_14
   - pair_ok_15_14_for_nic_out_15_nic_in_14 <= 0
 link_activation_dst_nic_in_0_0: switch_nic_in_0_0 - is_active_0 <= 0
 link_activation_src_0_nic_out_0: switch_0_nic_out_0 - is_active_0 <= 0
 link_activation_dst_nic_in_1_1: switch_nic_in_1_1 - is_active_1 <= 0
 link_activation_src_1_nic_out_1: switch_1_nic_out_1 - is_active_1 <= 0
 link_activation_dst_nic_in_2_2: switch_nic_in_2_2 - is_active_2 <= 0
 link_activation_src_2_nic_out_2: switch_2_nic_out_2 - is_active_2 <= 0
 link_activation_dst_nic_in_3_3: switch_nic_in_3_3 - is_active_3 <= 0
 link_activation_src_3_nic_out_3: switch_3_nic_out_3 - is_active_3 <= 0
 link_activation_dst_nic_in_4_4: switch_nic_in_4_4 - is_active_4 <= 0
 link_activation_src_4_nic_out_4: switch_4_nic_out_4 - is_active_4 <= 0
 link_activation_dst_nic_in_5_5: switch_nic_in_5_5 - is_active_5 <= 0
 link_activation_src_5_nic_out_5: switch_5_nic_out_5 - is_active_5 <= 0
 link_activation_dst_nic_in_6_6: switch_nic_in_6_6 - is_active_6 <= 0
 link_activation_src_6_nic_out_6: switch_6_nic_out_6 - is_active_6 <= 0
 link_activation_dst_nic_in_7_7: switch_nic_in_7_7 - is_active_7 <= 0
 link_activation_src_7_nic_out_7: switch_7_nic_out_7 - is_active_7 <= 0
 link_activation_dst_nic_in_8_8: switch_nic_in_8_8 - is_active_8 <= 0
 link_activation_src_8_nic_out_8: switch_8_nic_out_8 - is_active_8 <= 0
 link_activation_dst_nic_in_9_9: switch_nic_in_9_9 - is_active_9 <= 0
 link_activation_src_9_nic_out_9: switch_9_nic_out_9 - is_active_9 <= 0
 link_activation_dst_nic_in_10_10: switch_nic_in_10_10 - is_active_10 <= 0
 link_activation_src_10_nic_out_10: switch_10_nic_out_10 - is_active_10
   <= 0
 link_activation_dst_nic_in_11_11: switch_nic_in_11_11 - is_active_11 <= 0
 link_activation_src_11_nic_out_11: switch_11_nic_out_11 - is_active_11
   <= 0
 link_activation_dst_nic_in_12_12: switch_nic_in_12_12 - is_active_12 <= 0
 link_activation_src_12_nic_out_12: switch_12_nic_out_12 - is_active_12
   <= 0
 link_activation_dst_nic_in_13_13: switch_nic_in_13_13 - is_active_13 <= 0
 link_activation_src_13_nic_out_13: switch_13_nic_out_13 - is_active_13
   <= 0
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
 edge_flow_constr_nic_out_0_nic_in_1: flow_nic_out_0_nic_in_1
   - 75120.19230769231 switch_nic_out_0_nic_in_1 <= 0
 edge_flow_constr_nic_out_0_nic_in_2: flow_nic_out_0_nic_in_2
   - 75120.19230769231 switch_nic_out_0_nic_in_2 <= 0
 edge_flow_constr_nic_out_0_nic_in_3: flow_nic_out_0_nic_in_3
   - 75120.19230769231 switch_nic_out_0_nic_in_3 <= 0
 edge_flow_constr_nic_out_0_nic_in_4: flow_nic_out_0_nic_in_4
   - 75120.19230769231 switch_nic_out_0_nic_in_4 <= 0
 edge_flow_constr_nic_out_0_nic_in_5: flow_nic_out_0_nic_in_5
   - 75120.19230769231 switch_nic_out_0_nic_in_5 <= 0
 edge_flow_constr_nic_out_0_nic_in_6: flow_nic_out_0_nic_in_6
   - 75120.19230769231 switch_nic_out_0_nic_in_6 <= 0
 edge_flow_constr_nic_out_0_nic_in_7: flow_nic_out_0_nic_in_7
   - 75120.19230769231 switch_nic_out_0_nic_in_7 <= 0
 edge_flow_constr_nic_out_0_nic_in_8: flow_nic_out_0_nic_in_8
   - 75120.19230769231 switch_nic_out_0_nic_in_8 <= 0
 edge_flow_constr_nic_out_0_nic_in_9: flow_nic_out_0_nic_in_9
   - 75120.19230769231 switch_nic_out_0_nic_in_9 <= 0
 edge_flow_constr_nic_out_0_nic_in_10: flow_nic_out_0_nic_in_10
   - 75120.19230769231 switch_nic_out_0_nic_in_10 <= 0
 edge_flow_constr_nic_out_0_nic_in_11: flow_nic_out_0_nic_in_11
   - 75120.19230769231 switch_nic_out_0_nic_in_11 <= 0
 edge_flow_constr_nic_out_0_nic_in_12: flow_nic_out_0_nic_in_12
   - 75120.19230769231 switch_nic_out_0_nic_in_12 <= 0
 edge_flow_constr_nic_out_0_nic_in_13: flow_nic_out_0_nic_in_13
   - 75120.19230769231 switch_nic_out_0_nic_in_13 <= 0
 edge_flow_constr_nic_out_0_nic_in_14: flow_nic_out_0_nic_in_14
   - 75120.19230769231 switch_nic_out_0_nic_in_14 <= 0
 edge_flow_constr_nic_out_0_nic_in_15: flow_nic_out_0_nic_in_15
   - 75120.19230769231 switch_nic_out_0_nic_in_15 <= 0
 edge_flow_constr_nic_in_0_0: flow_nic_in_0_0
   - 4.807692307692308e+06 switch_nic_in_0_0 <= 0
 edge_flow_constr_0_nic_out_0: flow_0_nic_out_0
   - 4.807692307692308e+06 switch_0_nic_out_0 <= 0
 edge_flow_constr_source_nic_in_1: flow_source_nic_in_1
   - 5e+08 switch_source_nic_in_1 <= 0
 edge_flow_constr_nic_out_1_sink: flow_nic_out_1_sink
   - 5e+08 switch_nic_out_1_sink <= 0
 edge_flow_constr_nic_out_1_nic_in_0: flow_nic_out_1_nic_in_0
   - 75120.19230769231 switch_nic_out_1_nic_in_0 <= 0
 edge_flow_constr_nic_out_1_nic_in_2: flow_nic_out_1_nic_in_2
   - 75120.19230769231 switch_nic_out_1_nic_in_2 <= 0
 edge_flow_constr_nic_out_1_nic_in_3: flow_nic_out_1_nic_in_3
   - 75120.19230769231 switch_nic_out_1_nic_in_3 <= 0
 edge_flow_constr_nic_out_1_nic_in_4: flow_nic_out_1_nic_in_4
   - 75120.19230769231 switch_nic_out_1_nic_in_4 <= 0
 edge_flow_constr_nic_out_1_nic_in_5: flow_nic_out_1_nic_in_5
   - 75120.19230769231 switch_nic_out_1_nic_in_5 <= 0
 edge_flow_constr_nic_out_1_nic_in_6: flow_nic_out_1_nic_in_6
   - 75120.19230769231 switch_nic_out_1_nic_in_6 <= 0
 edge_flow_constr_nic_out_1_nic_in_7: flow_nic_out_1_nic_in_7
   - 75120.19230769231 switch_nic_out_1_nic_in_7 <= 0
 edge_flow_constr_nic_out_1_nic_in_8: flow_nic_out_1_nic_in_8
   - 75120.19230769231 switch_nic_out_1_nic_in_8 <= 0
 edge_flow_constr_nic_out_1_nic_in_9: flow_nic_out_1_nic_in_9
   - 75120.19230769231 switch_nic_out_1_nic_in_9 <= 0
 edge_flow_constr_nic_out_1_nic_in_10: flow_nic_out_1_nic_in_10
   - 75120.19230769231 switch_nic_out_1_nic_in_10 <= 0
 edge_flow_constr_nic_out_1_nic_in_11: flow_nic_out_1_nic_in_11
   - 75120.19230769231 switch_nic_out_1_nic_in_11 <= 0
 edge_flow_constr_nic_out_1_nic_in_12: flow_nic_out_1_nic_in_12
   - 75120.19230769231 switch_nic_out_1_nic_in_12 <= 0
 edge_flow_constr_nic_out_1_nic_in_13: flow_nic_out_1_nic_in_13
   - 75120.19230769231 switch_nic_out_1_nic_in_13 <= 0
 edge_flow_constr_nic_out_1_nic_in_14: flow_nic_out_1_nic_in_14
   - 75120.19230769231 switch_nic_out_1_nic_in_14 <= 0
 edge_flow_constr_nic_out_1_nic_in_15: flow_nic_out_1_nic_in_15
   - 75120.19230769231 switch_nic_out_1_nic_in_15 <= 0
 edge_flow_constr_nic_in_1_1: flow_nic_in_1_1
   - 4.807692307692308e+06 switch_nic_in_1_1 <= 0
 edge_flow_constr_1_nic_out_1: flow_1_nic_out_1
   - 4.807692307692308e+06 switch_1_nic_out_1 <= 0
 edge_flow_constr_source_nic_in_2: flow_source_nic_in_2
   - 5e+08 switch_source_nic_in_2 <= 0
 edge_flow_constr_nic_out_2_sink: flow_nic_out_2_sink
   - 5e+08 switch_nic_out_2_sink <= 0
 edge_flow_constr_nic_out_2_nic_in_0: flow_nic_out_2_nic_in_0
   - 75120.19230769231 switch_nic_out_2_nic_in_0 <= 0
 edge_flow_constr_nic_out_2_nic_in_1: flow_nic_out_2_nic_in_1
   - 75120.19230769231 switch_nic_out_2_nic_in_1 <= 0
 edge_flow_constr_nic_out_2_nic_in_3: flow_nic_out_2_nic_in_3
   - 75120.19230769231 switch_nic_out_2_nic_in_3 <= 0
 edge_flow_constr_nic_out_2_nic_in_4: flow_nic_out_2_nic_in_4
   - 75120.19230769231 switch_nic_out_2_nic_in_4 <= 0
 edge_flow_constr_nic_out_2_nic_in_5: flow_nic_out_2_nic_in_5
   - 75120.19230769231 switch_nic_out_2_nic_in_5 <= 0
 edge_flow_constr_nic_out_2_nic_in_6: flow_nic_out_2_nic_in_6
   - 75120.19230769231 switch_nic_out_2_nic_in_6 <= 0
 edge_flow_constr_nic_out_2_nic_in_7: flow_nic_out_2_nic_in_7
   - 75120.19230769231 switch_nic_out_2_nic_in_7 <= 0
 edge_flow_constr_nic_out_2_nic_in_8: flow_nic_out_2_nic_in_8
   - 75120.19230769231 switch_nic_out_2_nic_in_8 <= 0
 edge_flow_constr_nic_out_2_nic_in_9: flow_nic_out_2_nic_in_9
   - 75120.19230769231 switch_nic_out_2_nic_in_9 <= 0
 edge_flow_constr_nic_out_2_nic_in_10: flow_nic_out_2_nic_in_10
   - 75120.19230769231 switch_nic_out_2_nic_in_10 <= 0
 edge_flow_constr_nic_out_2_nic_in_11: flow_nic_out_2_nic_in_11
   - 75120.19230769231 switch_nic_out_2_nic_in_11 <= 0
 edge_flow_constr_nic_out_2_nic_in_12: flow_nic_out_2_nic_in_12
   - 75120.19230769231 switch_nic_out_2_nic_in_12 <= 0
 edge_flow_constr_nic_out_2_nic_in_13: flow_nic_out_2_nic_in_13
   - 75120.19230769231 switch_nic_out_2_nic_in_13 <= 0
 edge_flow_constr_nic_out_2_nic_in_14: flow_nic_out_2_nic_in_14
   - 75120.19230769231 switch_nic_out_2_nic_in_14 <= 0
 edge_flow_constr_nic_out_2_nic_in_15: flow_nic_out_2_nic_in_15
   - 75120.19230769231 switch_nic_out_2_nic_in_15 <= 0
 edge_flow_constr_nic_in_2_2: flow_nic_in_2_2
   - 4.807692307692308e+06 switch_nic_in_2_2 <= 0
 edge_flow_constr_2_nic_out_2: flow_2_nic_out_2
   - 4.807692307692308e+06 switch_2_nic_out_2 <= 0
 edge_flow_constr_source_nic_in_3: flow_source_nic_in_3
   - 5e+08 switch_source_nic_in_3 <= 0
 edge_flow_constr_nic_out_3_sink: flow_nic_out_3_sink
   - 5e+08 switch_nic_out_3_sink <= 0
 edge_flow_constr_nic_out_3_nic_in_0: flow_nic_out_3_nic_in_0
   - 75120.19230769231 switch_nic_out_3_nic_in_0 <= 0
 edge_flow_constr_nic_out_3_nic_in_1: flow_nic_out_3_nic_in_1
   - 75120.19230769231 switch_nic_out_3_nic_in_1 <= 0
 edge_flow_constr_nic_out_3_nic_in_2: flow_nic_out_3_nic_in_2
   - 75120.19230769231 switch_nic_out_3_nic_in_2 <= 0
 edge_flow_constr_nic_out_3_nic_in_4: flow_nic_out_3_nic_in_4
   - 75120.19230769231 switch_nic_out_3_nic_in_4 <= 0
 edge_flow_constr_nic_out_3_nic_in_5: flow_nic_out_3_nic_in_5
   - 75120.19230769231 switch_nic_out_3_nic_in_5 <= 0
 edge_flow_constr_nic_out_3_nic_in_6: flow_nic_out_3_nic_in_6
   - 75120.19230769231 switch_nic_out_3_nic_in_6 <= 0
 edge_flow_constr_nic_out_3_nic_in_7: flow_nic_out_3_nic_in_7
   - 75120.19230769231 switch_nic_out_3_nic_in_7 <= 0
 edge_flow_constr_nic_out_3_nic_in_8: flow_nic_out_3_nic_in_8
   - 75120.19230769231 switch_nic_out_3_nic_in_8 <= 0
 edge_flow_constr_nic_out_3_nic_in_9: flow_nic_out_3_nic_in_9
   - 75120.19230769231 switch_nic_out_3_nic_in_9 <= 0
 edge_flow_constr_nic_out_3_nic_in_10: flow_nic_out_3_nic_in_10
   - 75120.19230769231 switch_nic_out_3_nic_in_10 <= 0
 edge_flow_constr_nic_out_3_nic_in_11: flow_nic_out_3_nic_in_11
   - 75120.19230769231 switch_nic_out_3_nic_in_11 <= 0
 edge_flow_constr_nic_out_3_nic_in_12: flow_nic_out_3_nic_in_12
   - 75120.19230769231 switch_nic_out_3_nic_in_12 <= 0
 edge_flow_constr_nic_out_3_nic_in_13: flow_nic_out_3_nic_in_13
   - 75120.19230769231 switch_nic_out_3_nic_in_13 <= 0
 edge_flow_constr_nic_out_3_nic_in_14: flow_nic_out_3_nic_in_14
   - 75120.19230769231 switch_nic_out_3_nic_in_14 <= 0
 edge_flow_constr_nic_out_3_nic_in_15: flow_nic_out_3_nic_in_15
   - 75120.19230769231 switch_nic_out_3_nic_in_15 <= 0
 edge_flow_constr_nic_in_3_3: flow_nic_in_3_3
   - 4.807692307692308e+06 switch_nic_in_3_3 <= 0
 edge_flow_constr_3_nic_out_3: flow_3_nic_out_3
   - 4.807692307692308e+06 switch_3_nic_out_3 <= 0
 edge_flow_constr_source_nic_in_4: flow_source_nic_in_4
   - 5e+08 switch_source_nic_in_4 <= 0
 edge_flow_constr_nic_out_4_sink: flow_nic_out_4_sink
   - 5e+08 switch_nic_out_4_sink <= 0
 edge_flow_constr_nic_out_4_nic_in_0: flow_nic_out_4_nic_in_0
   - 75120.19230769231 switch_nic_out_4_nic_in_0 <= 0
 edge_flow_constr_nic_out_4_nic_in_1: flow_nic_out_4_nic_in_1
   - 75120.19230769231 switch_nic_out_4_nic_in_1 <= 0
 edge_flow_constr_nic_out_4_nic_in_2: flow_nic_out_4_nic_in_2
   - 75120.19230769231 switch_nic_out_4_nic_in_2 <= 0
 edge_flow_constr_nic_out_4_nic_in_3: flow_nic_out_4_nic_in_3
   - 75120.19230769231 switch_nic_out_4_nic_in_3 <= 0
 edge_flow_constr_nic_out_4_nic_in_5: flow_nic_out_4_nic_in_5
   - 75120.19230769231 switch_nic_out_4_nic_in_5 <= 0
 edge_flow_constr_nic_out_4_nic_in_6: flow_nic_out_4_nic_in_6
   - 75120.19230769231 switch_nic_out_4_nic_in_6 <= 0
 edge_flow_constr_nic_out_4_nic_in_7: flow_nic_out_4_nic_in_7
   - 75120.19230769231 switch_nic_out_4_nic_in_7 <= 0
 edge_flow_constr_nic_out_4_nic_in_8: flow_nic_out_4_nic_in_8
   - 75120.19230769231 switch_nic_out_4_nic_in_8 <= 0
 edge_flow_constr_nic_out_4_nic_in_9: flow_nic_out_4_nic_in_9
   - 75120.19230769231 switch_nic_out_4_nic_in_9 <= 0
 edge_flow_constr_nic_out_4_nic_in_10: flow_nic_out_4_nic_in_10
   - 75120.19230769231 switch_nic_out_4_nic_in_10 <= 0
 edge_flow_constr_nic_out_4_nic_in_11: flow_nic_out_4_nic_in_11
   - 75120.19230769231 switch_nic_out_4_nic_in_11 <= 0
 edge_flow_constr_nic_out_4_nic_in_12: flow_nic_out_4_nic_in_12
   - 75120.19230769231 switch_nic_out_4_nic_in_12 <= 0
 edge_flow_constr_nic_out_4_nic_in_13: flow_nic_out_4_nic_in_13
   - 75120.19230769231 switch_nic_out_4_nic_in_13 <= 0
 edge_flow_constr_nic_out_4_nic_in_14: flow_nic_out_4_nic_in_14
   - 75120.19230769231 switch_nic_out_4_nic_in_14 <= 0
 edge_flow_constr_nic_out_4_nic_in_15: flow_nic_out_4_nic_in_15
   - 75120.19230769231 switch_nic_out_4_nic_in_15 <= 0
 edge_flow_constr_nic_in_4_4: flow_nic_in_4_4
   - 4.807692307692308e+06 switch_nic_in_4_4 <= 0
 edge_flow_constr_4_nic_out_4: flow_4_nic_out_4
   - 4.807692307692308e+06 switch_4_nic_out_4 <= 0
 edge_flow_constr_source_nic_in_5: flow_source_nic_in_5
   - 5e+08 switch_source_nic_in_5 <= 0
 edge_flow_constr_nic_out_5_sink: flow_nic_out_5_sink
   - 5e+08 switch_nic_out_5_sink <= 0
 edge_flow_constr_nic_out_5_nic_in_0: flow_nic_out_5_nic_in_0
   - 75120.19230769231 switch_nic_out_5_nic_in_0 <= 0
 edge_flow_constr_nic_out_5_nic_in_1: flow_nic_out_5_nic_in_1
   - 75120.19230769231 switch_nic_out_5_nic_in_1 <= 0
 edge_flow_constr_nic_out_5_nic_in_2: flow_nic_out_5_nic_in_2
   - 75120.19230769231 switch_nic_out_5_nic_in_2 <= 0
 edge_flow_constr_nic_out_5_nic_in_3: flow_nic_out_5_nic_in_3
   - 75120.19230769231 switch_nic_out_5_nic_in_3 <= 0
 edge_flow_constr_nic_out_5_nic_in_4: flow_nic_out_5_nic_in_4
   - 75120.19230769231 switch_nic_out_5_nic_in_4 <= 0
 edge_flow_constr_nic_out_5_nic_in_6: flow_nic_out_5_nic_in_6
   - 75120.19230769231 switch_nic_out_5_nic_in_6 <= 0
 edge_flow_constr_nic_out_5_nic_in_7: flow_nic_out_5_nic_in_7
   - 75120.19230769231 switch_nic_out_5_nic_in_7 <= 0
 edge_flow_constr_nic_out_5_nic_in_8: flow_nic_out_5_nic_in_8
   - 75120.19230769231 switch_nic_out_5_nic_in_8 <= 0
 edge_flow_constr_nic_out_5_nic_in_9: flow_nic_out_5_nic_in_9
   - 75120.19230769231 switch_nic_out_5_nic_in_9 <= 0
 edge_flow_constr_nic_out_5_nic_in_10: flow_nic_out_5_nic_in_10
   - 75120.19230769231 switch_nic_out_5_nic_in_10 <= 0
 edge_flow_constr_nic_out_5_nic_in_11: flow_nic_out_5_nic_in_11
   - 75120.19230769231 switch_nic_out_5_nic_in_11 <= 0
 edge_flow_constr_nic_out_5_nic_in_12: flow_nic_out_5_nic_in_12
   - 75120.19230769231 switch_nic_out_5_nic_in_12 <= 0
 edge_flow_constr_nic_out_5_nic_in_13: flow_nic_out_5_nic_in_13
   - 75120.19230769231 switch_nic_out_5_nic_in_13 <= 0
 edge_flow_constr_nic_out_5_nic_in_14: flow_nic_out_5_nic_in_14
   - 75120.19230769231 switch_nic_out_5_nic_in_14 <= 0
 edge_flow_constr_nic_out_5_nic_in_15: flow_nic_out_5_nic_in_15
   - 75120.19230769231 switch_nic_out_5_nic_in_15 <= 0
 edge_flow_constr_nic_in_5_5: flow_nic_in_5_5
   - 4.807692307692308e+06 switch_nic_in_5_5 <= 0
 edge_flow_constr_5_nic_out_5: flow_5_nic_out_5
   - 4.807692307692308e+06 switch_5_nic_out_5 <= 0
 edge_flow_constr_source_nic_in_6: flow_source_nic_in_6
   - 5e+08 switch_source_nic_in_6 <= 0
 edge_flow_constr_nic_out_6_sink: flow_nic_out_6_sink
   - 5e+08 switch_nic_out_6_sink <= 0
 edge_flow_constr_nic_out_6_nic_in_0: flow_nic_out_6_nic_in_0
   - 75120.19230769231 switch_nic_out_6_nic_in_0 <= 0
 edge_flow_constr_nic_out_6_nic_in_1: flow_nic_out_6_nic_in_1
   - 75120.19230769231 switch_nic_out_6_nic_in_1 <= 0
 edge_flow_constr_nic_out_6_nic_in_2: flow_nic_out_6_nic_in_2
   - 75120.19230769231 switch_nic_out_6_nic_in_2 <= 0
 edge_flow_constr_nic_out_6_nic_in_3: flow_nic_out_6_nic_in_3
   - 75120.19230769231 switch_nic_out_6_nic_in_3 <= 0
 edge_flow_constr_nic_out_6_nic_in_4: flow_nic_out_6_nic_in_4
   - 75120.19230769231 switch_nic_out_6_nic_in_4 <= 0
 edge_flow_constr_nic_out_6_nic_in_5: flow_nic_out_6_nic_in_5
   - 75120.19230769231 switch_nic_out_6_nic_in_5 <= 0
 edge_flow_constr_nic_out_6_nic_in_7: flow_nic_out_6_nic_in_7
   - 75120.19230769231 switch_nic_out_6_nic_in_7 <= 0
 edge_flow_constr_nic_out_6_nic_in_8: flow_nic_out_6_nic_in_8
   - 75120.19230769231 switch_nic_out_6_nic_in_8 <= 0
 edge_flow_constr_nic_out_6_nic_in_9: flow_nic_out_6_nic_in_9
   - 75120.19230769231 switch_nic_out_6_nic_in_9 <= 0
 edge_flow_constr_nic_out_6_nic_in_10: flow_nic_out_6_nic_in_10
   - 75120.19230769231 switch_nic_out_6_nic_in_10 <= 0
 edge_flow_constr_nic_out_6_nic_in_11: flow_nic_out_6_nic_in_11
   - 75120.19230769231 switch_nic_out_6_nic_in_11 <= 0
 edge_flow_constr_nic_out_6_nic_in_12: flow_nic_out_6_nic_in_12
   - 75120.19230769231 switch_nic_out_6_nic_in_12 <= 0
 edge_flow_constr_nic_out_6_nic_in_13: flow_nic_out_6_nic_in_13
   - 75120.19230769231 switch_nic_out_6_nic_in_13 <= 0
 edge_flow_constr_nic_out_6_nic_in_14: flow_nic_out_6_nic_in_14
   - 75120.19230769231 switch_nic_out_6_nic_in_14 <= 0
 edge_flow_constr_nic_out_6_nic_in_15: flow_nic_out_6_nic_in_15
   - 75120.19230769231 switch_nic_out_6_nic_in_15 <= 0
 edge_flow_constr_nic_in_6_6: flow_nic_in_6_6
   - 4.807692307692308e+06 switch_nic_in_6_6 <= 0
 edge_flow_constr_6_nic_out_6: flow_6_nic_out_6
   - 4.807692307692308e+06 switch_6_nic_out_6 <= 0
 edge_flow_constr_source_nic_in_7: flow_source_nic_in_7
   - 5e+08 switch_source_nic_in_7 <= 0
 edge_flow_constr_nic_out_7_sink: flow_nic_out_7_sink
   - 5e+08 switch_nic_out_7_sink <= 0
 edge_flow_constr_nic_out_7_nic_in_0: flow_nic_out_7_nic_in_0
   - 75120.19230769231 switch_nic_out_7_nic_in_0 <= 0
 edge_flow_constr_nic_out_7_nic_in_1: flow_nic_out_7_nic_in_1
   - 75120.19230769231 switch_nic_out_7_nic_in_1 <= 0
 edge_flow_constr_nic_out_7_nic_in_2: flow_nic_out_7_nic_in_2
   - 75120.19230769231 switch_nic_out_7_nic_in_2 <= 0
 edge_flow_constr_nic_out_7_nic_in_3: flow_nic_out_7_nic_in_3
   - 75120.19230769231 switch_nic_out_7_nic_in_3 <= 0
 edge_flow_constr_nic_out_7_nic_in_4: flow_nic_out_7_nic_in_4
   - 75120.19230769231 switch_nic_out_7_nic_in_4 <= 0
 edge_flow_constr_nic_out_7_nic_in_5: flow_nic_out_7_nic_in_5
   - 75120.19230769231 switch_nic_out_7_nic_in_5 <= 0
 edge_flow_constr_nic_out_7_nic_in_6: flow_nic_out_7_nic_in_6
   - 75120.19230769231 switch_nic_out_7_nic_in_6 <= 0
 edge_flow_constr_nic_out_7_nic_in_8: flow_nic_out_7_nic_in_8
   - 75120.19230769231 switch_nic_out_7_nic_in_8 <= 0
 edge_flow_constr_nic_out_7_nic_in_9: flow_nic_out_7_nic_in_9
   - 75120.19230769231 switch_nic_out_7_nic_in_9 <= 0
 edge_flow_constr_nic_out_7_nic_in_10: flow_nic_out_7_nic_in_10
   - 75120.19230769231 switch_nic_out_7_nic_in_10 <= 0
 edge_flow_constr_nic_out_7_nic_in_11: flow_nic_out_7_nic_in_11
   - 75120.19230769231 switch_nic_out_7_nic_in_11 <= 0
 edge_flow_constr_nic_out_7_nic_in_12: flow_nic_out_7_nic_in_12
   - 75120.19230769231 switch_nic_out_7_nic_in_12 <= 0
 edge_flow_constr_nic_out_7_nic_in_13: flow_nic_out_7_nic_in_13
   - 75120.19230769231 switch_nic_out_7_nic_in_13 <= 0
 edge_flow_constr_nic_out_7_nic_in_14: flow_nic_out_7_nic_in_14
   - 75120.19230769231 switch_nic_out_7_nic_in_14 <= 0
 edge_flow_constr_nic_out_7_nic_in_15: flow_nic_out_7_nic_in_15
   - 75120.19230769231 switch_nic_out_7_nic_in_15 <= 0
 edge_flow_constr_nic_in_7_7: flow_nic_in_7_7
   - 4.807692307692308e+06 switch_nic_in_7_7 <= 0
 edge_flow_constr_7_nic_out_7: flow_7_nic_out_7
   - 4.807692307692308e+06 switch_7_nic_out_7 <= 0
 edge_flow_constr_source_nic_in_8: flow_source_nic_in_8
   - 5e+08 switch_source_nic_in_8 <= 0
 edge_flow_constr_nic_out_8_sink: flow_nic_out_8_sink
   - 5e+08 switch_nic_out_8_sink <= 0
 edge_flow_constr_nic_out_8_nic_in_0: flow_nic_out_8_nic_in_0
   - 75120.19230769231 switch_nic_out_8_nic_in_0 <= 0
 edge_flow_constr_nic_out_8_nic_in_1: flow_nic_out_8_nic_in_1
   - 75120.19230769231 switch_nic_out_8_nic_in_1 <= 0
 edge_flow_constr_nic_out_8_nic_in_2: flow_nic_out_8_nic_in_2
   - 75120.19230769231 switch_nic_out_8_nic_in_2 <= 0
 edge_flow_constr_nic_out_8_nic_in_3: flow_nic_out_8_nic_in_3
   - 75120.19230769231 switch_nic_out_8_nic_in_3 <= 0
 edge_flow_constr_nic_out_8_nic_in_4: flow_nic_out_8_nic_in_4
   - 75120.19230769231 switch_nic_out_8_nic_in_4 <= 0
 edge_flow_constr_nic_out_8_nic_in_5: flow_nic_out_8_nic_in_5
   - 75120.19230769231 switch_nic_out_8_nic_in_5 <= 0
 edge_flow_constr_nic_out_8_nic_in_6: flow_nic_out_8_nic_in_6
   - 75120.19230769231 switch_nic_out_8_nic_in_6 <= 0
 edge_flow_constr_nic_out_8_nic_in_7: flow_nic_out_8_nic_in_7
   - 75120.19230769231 switch_nic_out_8_nic_in_7 <= 0
 edge_flow_constr_nic_out_8_nic_in_9: flow_nic_out_8_nic_in_9
   - 75120.19230769231 switch_nic_out_8_nic_in_9 <= 0
 edge_flow_constr_nic_out_8_nic_in_10: flow_nic_out_8_nic_in_10
   - 75120.19230769231 switch_nic_out_8_nic_in_10 <= 0
 edge_flow_constr_nic_out_8_nic_in_11: flow_nic_out_8_nic_in_11
   - 75120.19230769231 switch_nic_out_8_nic_in_11 <= 0
 edge_flow_constr_nic_out_8_nic_in_12: flow_nic_out_8_nic_in_12
   - 75120.19230769231 switch_nic_out_8_nic_in_12 <= 0
 edge_flow_constr_nic_out_8_nic_in_13: flow_nic_out_8_nic_in_13
   - 75120.19230769231 switch_nic_out_8_nic_in_13 <= 0
 edge_flow_constr_nic_out_8_nic_in_14: flow_nic_out_8_nic_in_14
   - 75120.19230769231 switch_nic_out_8_nic_in_14 <= 0
 edge_flow_constr_nic_out_8_nic_in_15: flow_nic_out_8_nic_in_15
   - 75120.19230769231 switch_nic_out_8_nic_in_15 <= 0
 edge_flow_constr_nic_in_8_8: flow_nic_in_8_8
   - 4.807692307692308e+06 switch_nic_in_8_8 <= 0
 edge_flow_constr_8_nic_out_8: flow_8_nic_out_8
   - 4.807692307692308e+06 switch_8_nic_out_8 <= 0
 edge_flow_constr_source_nic_in_9: flow_source_nic_in_9
   - 5e+08 switch_source_nic_in_9 <= 0
 edge_flow_constr_nic_out_9_sink: flow_nic_out_9_sink
   - 5e+08 switch_nic_out_9_sink <= 0
 edge_flow_constr_nic_out_9_nic_in_0: flow_nic_out_9_nic_in_0
   - 75120.19230769231 switch_nic_out_9_nic_in_0 <= 0
 edge_flow_constr_nic_out_9_nic_in_1: flow_nic_out_9_nic_in_1
   - 75120.19230769231 switch_nic_out_9_nic_in_1 <= 0
 edge_flow_constr_nic_out_9_nic_in_2: flow_nic_out_9_nic_in_2
   - 75120.19230769231 switch_nic_out_9_nic_in_2 <= 0
 edge_flow_constr_nic_out_9_nic_in_3: flow_nic_out_9_nic_in_3
   - 75120.19230769231 switch_nic_out_9_nic_in_3 <= 0
 edge_flow_constr_nic_out_9_nic_in_4: flow_nic_out_9_nic_in_4
   - 75120.19230769231 switch_nic_out_9_nic_in_4 <= 0
 edge_flow_constr_nic_out_9_nic_in_5: flow_nic_out_9_nic_in_5
   - 75120.19230769231 switch_nic_out_9_nic_in_5 <= 0
 edge_flow_constr_nic_out_9_nic_in_6: flow_nic_out_9_nic_in_6
   - 75120.19230769231 switch_nic_out_9_nic_in_6 <= 0
 edge_flow_constr_nic_out_9_nic_in_7: flow_nic_out_9_nic_in_7
   - 75120.19230769231 switch_nic_out_9_nic_in_7 <= 0
 edge_flow_constr_nic_out_9_nic_in_8: flow_nic_out_9_nic_in_8
   - 75120.19230769231 switch_nic_out_9_nic_in_8 <= 0
 edge_flow_constr_nic_out_9_nic_in_10: flow_nic_out_9_nic_in_10
   - 75120.19230769231 switch_nic_out_9_nic_in_10 <= 0
 edge_flow_constr_nic_out_9_nic_in_11: flow_nic_out_9_nic_in_11
   - 75120.19230769231 switch_nic_out_9_nic_in_11 <= 0
 edge_flow_constr_nic_out_9_nic_in_12: flow_nic_out_9_nic_in_12
   - 75120.19230769231 switch_nic_out_9_nic_in_12 <= 0
 edge_flow_constr_nic_out_9_nic_in_13: flow_nic_out_9_nic_in_13
   - 75120.19230769231 switch_nic_out_9_nic_in_13 <= 0
 edge_flow_constr_nic_out_9_nic_in_14: flow_nic_out_9_nic_in_14
   - 75120.19230769231 switch_nic_out_9_nic_in_14 <= 0
 edge_flow_constr_nic_out_9_nic_in_15: flow_nic_out_9_nic_in_15
   - 75120.19230769231 switch_nic_out_9_nic_in_15 <= 0
 edge_flow_constr_nic_in_9_9: flow_nic_in_9_9
   - 4.807692307692308e+06 switch_nic_in_9_9 <= 0
 edge_flow_constr_9_nic_out_9: flow_9_nic_out_9
   - 4.807692307692308e+06 switch_9_nic_out_9 <= 0
 edge_flow_constr_source_nic_in_10: flow_source_nic_in_10
   - 5e+08 switch_source_nic_in_10 <= 0
 edge_flow_constr_nic_out_10_sink: flow_nic_out_10_sink
   - 5e+08 switch_nic_out_10_sink <= 0
 edge_flow_constr_nic_out_10_nic_in_0: flow_nic_out_10_nic_in_0
   - 75120.19230769231 switch_nic_out_10_nic_in_0 <= 0
 edge_flow_constr_nic_out_10_nic_in_1: flow_nic_out_10_nic_in_1
   - 75120.19230769231 switch_nic_out_10_nic_in_1 <= 0
 edge_flow_constr_nic_out_10_nic_in_2: flow_nic_out_10_nic_in_2
   - 75120.19230769231 switch_nic_out_10_nic_in_2 <= 0
 edge_flow_constr_nic_out_10_nic_in_3: flow_nic_out_10_nic_in_3
   - 75120.19230769231 switch_nic_out_10_nic_in_3 <= 0
 edge_flow_constr_nic_out_10_nic_in_4: flow_nic_out_10_nic_in_4
   - 75120.19230769231 switch_nic_out_10_nic_in_4 <= 0
 edge_flow_constr_nic_out_10_nic_in_5: flow_nic_out_10_nic_in_5
   - 75120.19230769231 switch_nic_out_10_nic_in_5 <= 0
 edge_flow_constr_nic_out_10_nic_in_6: flow_nic_out_10_nic_in_6
   - 75120.19230769231 switch_nic_out_10_nic_in_6 <= 0
 edge_flow_constr_nic_out_10_nic_in_7: flow_nic_out_10_nic_in_7
   - 75120.19230769231 switch_nic_out_10_nic_in_7 <= 0
 edge_flow_constr_nic_out_10_nic_in_8: flow_nic_out_10_nic_in_8
   - 75120.19230769231 switch_nic_out_10_nic_in_8 <= 0
 edge_flow_constr_nic_out_10_nic_in_9: flow_nic_out_10_nic_in_9
   - 75120.19230769231 switch_nic_out_10_nic_in_9 <= 0
 edge_flow_constr_nic_out_10_nic_in_11: flow_nic_out_10_nic_in_11
   - 75120.19230769231 switch_nic_out_10_nic_in_11 <= 0
 edge_flow_constr_nic_out_10_nic_in_12: flow_nic_out_10_nic_in_12
   - 75120.19230769231 switch_nic_out_10_nic_in_12 <= 0
 edge_flow_constr_nic_out_10_nic_in_13: flow_nic_out_10_nic_in_13
   - 75120.19230769231 switch_nic_out_10_nic_in_13 <= 0
 edge_flow_constr_nic_out_10_nic_in_14: flow_nic_out_10_nic_in_14
   - 75120.19230769231 switch_nic_out_10_nic_in_14 <= 0
 edge_flow_constr_nic_out_10_nic_in_15: flow_nic_out_10_nic_in_15
   - 75120.19230769231 switch_nic_out_10_nic_in_15 <= 0
 edge_flow_constr_nic_in_10_10: flow_nic_in_10_10
   - 4.807692307692308e+06 switch_nic_in_10_10 <= 0
 edge_flow_constr_10_nic_out_10: flow_10_nic_out_10
   - 4.807692307692308e+06 switch_10_nic_out_10 <= 0
 edge_flow_constr_source_nic_in_11: flow_source_nic_in_11
   - 5e+08 switch_source_nic_in_11 <= 0
 edge_flow_constr_nic_out_11_sink: flow_nic_out_11_sink
   - 5e+08 switch_nic_out_11_sink <= 0
 edge_flow_constr_nic_out_11_nic_in_0: flow_nic_out_11_nic_in_0
   - 75120.19230769231 switch_nic_out_11_nic_in_0 <= 0
 edge_flow_constr_nic_out_11_nic_in_1: flow_nic_out_11_nic_in_1
   - 75120.19230769231 switch_nic_out_11_nic_in_1 <= 0
 edge_flow_constr_nic_out_11_nic_in_2: flow_nic_out_11_nic_in_2
   - 75120.19230769231 switch_nic_out_11_nic_in_2 <= 0
 edge_flow_constr_nic_out_11_nic_in_3: flow_nic_out_11_nic_in_3
   - 75120.19230769231 switch_nic_out_11_nic_in_3 <= 0
 edge_flow_constr_nic_out_11_nic_in_4: flow_nic_out_11_nic_in_4
   - 75120.19230769231 switch_nic_out_11_nic_in_4 <= 0
 edge_flow_constr_nic_out_11_nic_in_5: flow_nic_out_11_nic_in_5
   - 75120.19230769231 switch_nic_out_11_nic_in_5 <= 0
 edge_flow_constr_nic_out_11_nic_in_6: flow_nic_out_11_nic_in_6
   - 75120.19230769231 switch_nic_out_11_nic_in_6 <= 0
 edge_flow_constr_nic_out_11_nic_in_7: flow_nic_out_11_nic_in_7
   - 75120.19230769231 switch_nic_out_11_nic_in_7 <= 0
 edge_flow_constr_nic_out_11_nic_in_8: flow_nic_out_11_nic_in_8
   - 75120.19230769231 switch_nic_out_11_nic_in_8 <= 0
 edge_flow_constr_nic_out_11_nic_in_9: flow_nic_out_11_nic_in_9
   - 75120.19230769231 switch_nic_out_11_nic_in_9 <= 0
 edge_flow_constr_nic_out_11_nic_in_10: flow_nic_out_11_nic_in_10
   - 75120.19230769231 switch_nic_out_11_nic_in_10 <= 0
 edge_flow_constr_nic_out_11_nic_in_12: flow_nic_out_11_nic_in_12
   - 75120.19230769231 switch_nic_out_11_nic_in_12 <= 0
 edge_flow_constr_nic_out_11_nic_in_13: flow_nic_out_11_nic_in_13
   - 75120.19230769231 switch_nic_out_11_nic_in_13 <= 0
 edge_flow_constr_nic_out_11_nic_in_14: flow_nic_out_11_nic_in_14
   - 75120.19230769231 switch_nic_out_11_nic_in_14 <= 0
 edge_flow_constr_nic_out_11_nic_in_15: flow_nic_out_11_nic_in_15
   - 75120.19230769231 switch_nic_out_11_nic_in_15 <= 0
 edge_flow_constr_nic_in_11_11: flow_nic_in_11_11
   - 4.807692307692308e+06 switch_nic_in_11_11 <= 0
 edge_flow_constr_11_nic_out_11: flow_11_nic_out_11
   - 4.807692307692308e+06 switch_11_nic_out_11 <= 0
 edge_flow_constr_source_nic_in_12: flow_source_nic_in_12
   - 5e+08 switch_source_nic_in_12 <= 0
 edge_flow_constr_nic_out_12_sink: flow_nic_out_12_sink
   - 5e+08 switch_nic_out_12_sink <= 0
 edge_flow_constr_nic_out_12_nic_in_0: flow_nic_out_12_nic_in_0
   - 75120.19230769231 switch_nic_out_12_nic_in_0 <= 0
 edge_flow_constr_nic_out_12_nic_in_1: flow_nic_out_12_nic_in_1
   - 75120.19230769231 switch_nic_out_12_nic_in_1 <= 0
 edge_flow_constr_nic_out_12_nic_in_2: flow_nic_out_12_nic_in_2
   - 75120.19230769231 switch_nic_out_12_nic_in_2 <= 0
 edge_flow_constr_nic_out_12_nic_in_3: flow_nic_out_12_nic_in_3
   - 75120.19230769231 switch_nic_out_12_nic_in_3 <= 0
 edge_flow_constr_nic_out_12_nic_in_4: flow_nic_out_12_nic_in_4
   - 75120.19230769231 switch_nic_out_12_nic_in_4 <= 0
 edge_flow_constr_nic_out_12_nic_in_5: flow_nic_out_12_nic_in_5
   - 75120.19230769231 switch_nic_out_12_nic_in_5 <= 0
 edge_flow_constr_nic_out_12_nic_in_6: flow_nic_out_12_nic_in_6
   - 75120.19230769231 switch_nic_out_12_nic_in_6 <= 0
 edge_flow_constr_nic_out_12_nic_in_7: flow_nic_out_12_nic_in_7
   - 75120.19230769231 switch_nic_out_12_nic_in_7 <= 0
 edge_flow_constr_nic_out_12_nic_in_8: flow_nic_out_12_nic_in_8
   - 75120.19230769231 switch_nic_out_12_nic_in_8 <= 0
 edge_flow_constr_nic_out_12_nic_in_9: flow_nic_out_12_nic_in_9
   - 75120.19230769231 switch_nic_out_12_nic_in_9 <= 0
 edge_flow_constr_nic_out_12_nic_in_10: flow_nic_out_12_nic_in_10
   - 75120.19230769231 switch_nic_out_12_nic_in_10 <= 0
 edge_flow_constr_nic_out_12_nic_in_11: flow_nic_out_12_nic_in_11
   - 75120.19230769231 switch_nic_out_12_nic_in_11 <= 0
 edge_flow_constr_nic_out_12_nic_in_13: flow_nic_out_12_nic_in_13
   - 75120.19230769231 switch_nic_out_12_nic_in_13 <= 0
 edge_flow_constr_nic_out_12_nic_in_14: flow_nic_out_12_nic_in_14
   - 75120.19230769231 switch_nic_out_12_nic_in_14 <= 0
 edge_flow_constr_nic_out_12_nic_in_15: flow_nic_out_12_nic_in_15
   - 75120.19230769231 switch_nic_out_12_nic_in_15 <= 0
 edge_flow_constr_nic_in_12_12: flow_nic_in_12_12
   - 4.807692307692308e+06 switch_nic_in_12_12 <= 0
 edge_flow_constr_12_nic_out_12: flow_12_nic_out_12
   - 4.807692307692308e+06 switch_12_nic_out_12 <= 0
 edge_flow_constr_source_nic_in_13: flow_source_nic_in_13
   - 5e+08 switch_source_nic_in_13 <= 0
 edge_flow_constr_nic_out_13_sink: flow_nic_out_13_sink
   - 5e+08 switch_nic_out_13_sink <= 0
 edge_flow_constr_nic_out_13_nic_in_0: flow_nic_out_13_nic_in_0
   - 75120.19230769231 switch_nic_out_13_nic_in_0 <= 0
 edge_flow_constr_nic_out_13_nic_in_1: flow_nic_out_13_nic_in_1
   - 75120.19230769231 switch_nic_out_13_nic_in_1 <= 0
 edge_flow_constr_nic_out_13_nic_in_2: flow_nic_out_13_nic_in_2
   - 75120.19230769231 switch_nic_out_13_nic_in_2 <= 0
 edge_flow_constr_nic_out_13_nic_in_3: flow_nic_out_13_nic_in_3
   - 75120.19230769231 switch_nic_out_13_nic_in_3 <= 0
 edge_flow_constr_nic_out_13_nic_in_4: flow_nic_out_13_nic_in_4
   - 75120.19230769231 switch_nic_out_13_nic_in_4 <= 0
 edge_flow_constr_nic_out_13_nic_in_5: flow_nic_out_13_nic_in_5
   - 75120.19230769231 switch_nic_out_13_nic_in_5 <= 0
 edge_flow_constr_nic_out_13_nic_in_6: flow_nic_out_13_nic_in_6
   - 75120.19230769231 switch_nic_out_13_nic_in_6 <= 0
 edge_flow_constr_nic_out_13_nic_in_7: flow_nic_out_13_nic_in_7
   - 75120.19230769231 switch_nic_out_13_nic_in_7 <= 0
 edge_flow_constr_nic_out_13_nic_in_8: flow_nic_out_13_nic_in_8
   - 75120.19230769231 switch_nic_out_13_nic_in_8 <= 0
 edge_flow_constr_nic_out_13_nic_in_9: flow_nic_out_13_nic_in_9
   - 75120.19230769231 switch_nic_out_13_nic_in_9 <= 0
 edge_flow_constr_nic_out_13_nic_in_10: flow_nic_out_13_nic_in_10
   - 75120.19230769231 switch_nic_out_13_nic_in_10 <= 0
 edge_flow_constr_nic_out_13_nic_in_11: flow_nic_out_13_nic_in_11
   - 75120.19230769231 switch_nic_out_13_nic_in_11 <= 0
 edge_flow_constr_nic_out_13_nic_in_12: flow_nic_out_13_nic_in_12
   - 75120.19230769231 switch_nic_out_13_nic_in_12 <= 0
 edge_flow_constr_nic_out_13_nic_in_14: flow_nic_out_13_nic_in_14
   - 75120.19230769231 switch_nic_out_13_nic_in_14 <= 0
 edge_flow_constr_nic_out_13_nic_in_15: flow_nic_out_13_nic_in_15
   - 75120.19230769231 switch_nic_out_13_nic_in_15 <= 0
 edge_flow_constr_nic_in_13_13: flow_nic_in_13_13
   - 4.807692307692308e+06 switch_nic_in_13_13 <= 0
 edge_flow_constr_13_nic_out_13: flow_13_nic_out_13
   - 4.807692307692308e+06 switch_13_nic_out_13 <= 0
 edge_flow_constr_source_nic_in_14: flow_source_nic_in_14
   - 5e+08 switch_source_nic_in_14 <= 0
 edge_flow_constr_nic_out_14_sink: flow_nic_out_14_sink
   - 5e+08 switch_nic_out_14_sink <= 0
 edge_flow_constr_nic_out_14_nic_in_0: flow_nic_out_14_nic_in_0
   - 75120.19230769231 switch_nic_out_14_nic_in_0 <= 0
 edge_flow_constr_nic_out_14_nic_in_1: flow_nic_out_14_nic_in_1
   - 75120.19230769231 switch_nic_out_14_nic_in_1 <= 0
 edge_flow_constr_nic_out_14_nic_in_2: flow_nic_out_14_nic_in_2
   - 75120.19230769231 switch_nic_out_14_nic_in_2 <= 0
 edge_flow_constr_nic_out_14_nic_in_3: flow_nic_out_14_nic_in_3
   - 75120.19230769231 switch_nic_out_14_nic_in_3 <= 0
 edge_flow_constr_nic_out_14_nic_in_4: flow_nic_out_14_nic_in_4
   - 75120.19230769231 switch_nic_out_14_nic_in_4 <= 0
 edge_flow_constr_nic_out_14_nic_in_5: flow_nic_out_14_nic_in_5
   - 75120.19230769231 switch_nic_out_14_nic_in_5 <= 0
 edge_flow_constr_nic_out_14_nic_in_6: flow_nic_out_14_nic_in_6
   - 75120.19230769231 switch_nic_out_14_nic_in_6 <= 0
 edge_flow_constr_nic_out_14_nic_in_7: flow_nic_out_14_nic_in_7
   - 75120.19230769231 switch_nic_out_14_nic_in_7 <= 0
 edge_flow_constr_nic_out_14_nic_in_8: flow_nic_out_14_nic_in_8
   - 75120.19230769231 switch_nic_out_14_nic_in_8 <= 0
 edge_flow_constr_nic_out_14_nic_in_9: flow_nic_out_14_nic_in_9
   - 75120.19230769231 switch_nic_out_14_nic_in_9 <= 0
 edge_flow_constr_nic_out_14_nic_in_10: flow_nic_out_14_nic_in_10
   - 75120.19230769231 switch_nic_out_14_nic_in_10 <= 0
 edge_flow_constr_nic_out_14_nic_in_11: flow_nic_out_14_nic_in_11
   - 75120.19230769231 switch_nic_out_14_nic_in_11 <= 0
 edge_flow_constr_nic_out_14_nic_in_12: flow_nic_out_14_nic_in_12
   - 75120.19230769231 switch_nic_out_14_nic_in_12 <= 0
 edge_flow_constr_nic_out_14_nic_in_13: flow_nic_out_14_nic_in_13
   - 75120.19230769231 switch_nic_out_14_nic_in_13 <= 0
 edge_flow_constr_nic_out_14_nic_in_15: flow_nic_out_14_nic_in_15
   - 75120.19230769231 switch_nic_out_14_nic_in_15 <= 0
 edge_flow_constr_nic_in_14_14: flow_nic_in_14_14
   - 4.807692307692308e+06 switch_nic_in_14_14 <= 0
 edge_flow_constr_14_nic_out_14: flow_14_nic_out_14
   - 4.807692307692308e+06 switch_14_nic_out_14 <= 0
 edge_flow_constr_source_nic_in_15: flow_source_nic_in_15
   - 5e+08 switch_source_nic_in_15 <= 0
 edge_flow_constr_nic_out_15_sink: flow_nic_out_15_sink
   - 5e+08 switch_nic_out_15_sink <= 0
 edge_flow_constr_nic_out_15_nic_in_0: flow_nic_out_15_nic_in_0
   - 75120.19230769231 switch_nic_out_15_nic_in_0 <= 0
 edge_flow_constr_nic_out_15_nic_in_1: flow_nic_out_15_nic_in_1
   - 75120.19230769231 switch_nic_out_15_nic_in_1 <= 0
 edge_flow_constr_nic_out_15_nic_in_2: flow_nic_out_15_nic_in_2
   - 75120.19230769231 switch_nic_out_15_nic_in_2 <= 0
 edge_flow_constr_nic_out_15_nic_in_3: flow_nic_out_15_nic_in_3
   - 75120.19230769231 switch_nic_out_15_nic_in_3 <= 0
 edge_flow_constr_nic_out_15_nic_in_4: flow_nic_out_15_nic_in_4
   - 75120.19230769231 switch_nic_out_15_nic_in_4 <= 0
 edge_flow_constr_nic_out_15_nic_in_5: flow_nic_out_15_nic_in_5
   - 75120.19230769231 switch_nic_out_15_nic_in_5 <= 0
 edge_flow_constr_nic_out_15_nic_in_6: flow_nic_out_15_nic_in_6
   - 75120.19230769231 switch_nic_out_15_nic_in_6 <= 0
 edge_flow_constr_nic_out_15_nic_in_7: flow_nic_out_15_nic_in_7
   - 75120.19230769231 switch_nic_out_15_nic_in_7 <= 0
 edge_flow_constr_nic_out_15_nic_in_8: flow_nic_out_15_nic_in_8
   - 75120.19230769231 switch_nic_out_15_nic_in_8 <= 0
 edge_flow_constr_nic_out_15_nic_in_9: flow_nic_out_15_nic_in_9
   - 75120.19230769231 switch_nic_out_15_nic_in_9 <= 0
 edge_flow_constr_nic_out_15_nic_in_10: flow_nic_out_15_nic_in_10
   - 75120.19230769231 switch_nic_out_15_nic_in_10 <= 0
 edge_flow_constr_nic_out_15_nic_in_11: flow_nic_out_15_nic_in_11
   - 75120.19230769231 switch_nic_out_15_nic_in_11 <= 0
 edge_flow_constr_nic_out_15_nic_in_12: flow_nic_out_15_nic_in_12
   - 75120.19230769231 switch_nic_out_15_nic_in_12 <= 0
 edge_flow_constr_nic_out_15_nic_in_13: flow_nic_out_15_nic_in_13
   - 75120.19230769231 switch_nic_out_15_nic_in_13 <= 0
 edge_flow_constr_nic_out_15_nic_in_14: flow_nic_out_15_nic_in_14
   - 75120.19230769231 switch_nic_out_15_nic_in_14 <= 0
 edge_flow_constr_nic_in_15_15: flow_nic_in_15_15
   - 4.807692307692308e+06 switch_nic_in_15_15 <= 0
 edge_flow_constr_15_nic_out_15: flow_15_nic_out_15
   - 4.807692307692308e+06 switch_15_nic_out_15 <= 0
 force_group_off_constr_0: group_tp_active_0 = 0
 force_group_off_constr_1: group_tp_active_1 = 0
 force_group_off_constr_2: group_tp_active_2 = 0
 force_group_off_constr_3: group_tp_active_3 = 0
 force_group_off_constr_4: group_tp_active_4 = 0
 force_group_off_constr_5: group_tp_active_5 = 0
 force_group_off_constr_6: group_tp_active_6 = 0
 force_group_off_constr_7: group_tp_active_7 = 0
 force_group_off_constr_8: group_tp_active_8 = 0
 force_group_off_constr_9: group_tp_active_9 = 0
 force_group_off_constr_10: group_tp_active_10 = 0
 force_group_off_constr_11: group_tp_active_11 = 0
 force_group_off_constr_12: group_tp_active_12 = 0
 force_group_off_constr_13: group_tp_active_13 = 0
 force_group_off_constr_14: group_tp_active_14 = 0
 force_group_off_constr_15: group_tp_active_15 = 0
 force_min_throughput: flow_source_nic_in_0 + flow_source_nic_in_1
   + flow_source_nic_in_2 + flow_source_nic_in_3 + flow_source_nic_in_4
   + flow_source_nic_in_5 + flow_source_nic_in_6 + flow_source_nic_in_7
   + flow_source_nic_in_8 + flow_source_nic_in_9 + flow_source_nic_in_10
   + flow_source_nic_in_11 + flow_source_nic_in_12 + flow_source_nic_in_13
   + flow_source_nic_in_14 + flow_source_nic_in_15 >= 1
 inactive_start_invalid_0: is_active_0 = 0 -> start_0 = 60
 inactive_start_invalid_1: is_active_1 = 0 -> start_1 = 60
 inactive_start_invalid_2: is_active_2 = 0 -> start_2 = 60
 inactive_start_invalid_3: is_active_3 = 0 -> start_3 = 60
 inactive_start_invalid_4: is_active_4 = 0 -> start_4 = 60
 inactive_start_invalid_5: is_active_5 = 0 -> start_5 = 60
 inactive_start_invalid_6: is_active_6 = 0 -> start_6 = 60
 inactive_start_invalid_7: is_active_7 = 0 -> start_7 = 60
 inactive_start_invalid_8: is_active_8 = 0 -> start_8 = 60
 inactive_start_invalid_9: is_active_9 = 0 -> start_9 = 60
 inactive_start_invalid_10: is_active_10 = 0 -> start_10 = 60
 inactive_start_invalid_11: is_active_11 = 0 -> start_11 = 60
 inactive_start_invalid_12: is_active_12 = 0 -> start_12 = 60
 inactive_start_invalid_13: is_active_13 = 0 -> start_13 = 60
 inactive_start_invalid_14: is_active_14 = 0 -> start_14 = 60
 inactive_start_invalid_15: is_active_15 = 0 -> start_15 = 60
 node_throughput_constraint_0: group_tp_active_0 = 0 -> - 2560 hold_0_1
   - 1280 hold_0_2 - 853.3333333333334 hold_0_3 - 640 hold_0_4
   - 512 hold_0_5 - 426.6666666666667 hold_0_6 - 365.7142857142857 hold_0_7
   + flow_nic_in_0_0 <= 0
 node_throughput_constraint_1: group_tp_active_1 = 0 -> - 2560 hold_1_1
   - 1280 hold_1_2 - 853.3333333333334 hold_1_3 - 640 hold_1_4
   - 512 hold_1_5 - 426.6666666666667 hold_1_6 - 365.7142857142857 hold_1_7
   + flow_nic_in_1_1 <= 0
 node_throughput_constraint_2: group_tp_active_2 = 0 -> - 2400 hold_2_1
   - 1200 hold_2_2 - 800 hold_2_3 - 600 hold_2_4 - 480 hold_2_5
   - 400 hold_2_6 - 342.8571428571428 hold_2_7 - 300 hold_2_8
   - 266.6666666666667 hold_2_9 - 240 hold_2_10
   - 218.1818181818182 hold_2_11 + flow_nic_in_2_2 <= 0
 node_throughput_constraint_3: group_tp_active_3 = 0 -> - 2560 hold_3_1
   - 1280 hold_3_2 - 853.3333333333334 hold_3_3 - 640 hold_3_4
   - 512 hold_3_5 - 426.6666666666667 hold_3_6 - 365.7142857142857 hold_3_7
   + flow_nic_in_3_3 <= 0
 node_throughput_constraint_4: group_tp_active_4 = 0 -> - 2560 hold_4_1
   - 1280 hold_4_2 - 853.3333333333334 hold_4_3 - 640 hold_4_4
   - 512 hold_4_5 - 426.6666666666667 hold_4_6 - 365.7142857142857 hold_4_7
   + flow_nic_in_4_4 <= 0
 node_throughput_constraint_5: group_tp_active_5 = 0 -> - 12440 hold_5_1
   - 6220 hold_5_2 - 4146.666666666667 hold_5_3 - 3110 hold_5_4
   - 2488 hold_5_5 - 2073.333333333333 hold_5_6
   - 1777.142857142857 hold_5_7 - 1555 hold_5_8
   - 1382.222222222222 hold_5_9 - 1244 hold_5_10
   - 1130.909090909091 hold_5_11 - 1036.666666666667 hold_5_12
   - 956.9230769230769 hold_5_13 - 888.5714285714287 hold_5_14
   - 829.3333333333333 hold_5_15 - 777.5 hold_5_16
   - 731.7647058823529 hold_5_17 - 691.1111111111112 hold_5_18
   + flow_nic_in_5_5 <= 0
 node_throughput_constraint_6: group_tp_active_6 = 0 -> - 2400 hold_6_1
   - 1200 hold_6_2 - 800 hold_6_3 - 600 hold_6_4 - 480 hold_6_5
   - 400 hold_6_6 - 342.8571428571428 hold_6_7 - 300 hold_6_8
   - 266.6666666666667 hold_6_9 - 240 hold_6_10
   - 218.1818181818182 hold_6_11 + flow_nic_in_6_6 <= 0
 node_throughput_constraint_7: group_tp_active_7 = 0 -> - 2560 hold_7_1
   - 1280 hold_7_2 - 853.3333333333334 hold_7_3 - 640 hold_7_4
   - 512 hold_7_5 - 426.6666666666667 hold_7_6 - 365.7142857142857 hold_7_7
   + flow_nic_in_7_7 <= 0
 node_throughput_constraint_8: group_tp_active_8 = 0 -> - 2400 hold_8_1
   - 1200 hold_8_2 - 800 hold_8_3 - 600 hold_8_4 - 480 hold_8_5
   - 400 hold_8_6 - 342.8571428571428 hold_8_7 - 300 hold_8_8
   - 266.6666666666667 hold_8_9 - 240 hold_8_10
   - 218.1818181818182 hold_8_11 + flow_nic_in_8_8 <= 0
 node_throughput_constraint_9: group_tp_active_9 = 0 -> - 2560 hold_9_1
   - 1280 hold_9_2 - 853.3333333333334 hold_9_3 - 640 hold_9_4
   - 512 hold_9_5 - 426.6666666666667 hold_9_6 - 365.7142857142857 hold_9_7
   + flow_nic_in_9_9 <= 0
 node_throughput_constraint_10: group_tp_active_10 = 0 -> - 12440 hold_10_1
   - 6220 hold_10_2 - 4146.666666666667 hold_10_3 - 3110 hold_10_4
   - 2488 hold_10_5 - 2073.333333333333 hold_10_6
   - 1777.142857142857 hold_10_7 - 1555 hold_10_8
   - 1382.222222222222 hold_10_9 - 1244 hold_10_10
   - 1130.909090909091 hold_10_11 - 1036.666666666667 hold_10_12
   - 956.9230769230769 hold_10_13 - 888.5714285714287 hold_10_14
   - 829.3333333333333 hold_10_15 - 777.5 hold_10_16
   - 731.7647058823529 hold_10_17 - 691.1111111111112 hold_10_18
   + flow_nic_in_10_10 <= 0
 node_throughput_constraint_11: group_tp_active_11 = 0 -> - 2400 hold_11_1
   - 1200 hold_11_2 - 800 hold_11_3 - 600 hold_11_4 - 480 hold_11_5
   - 400 hold_11_6 - 342.8571428571428 hold_11_7 - 300 hold_11_8
   - 266.6666666666667 hold_11_9 - 240 hold_11_10
   - 218.1818181818182 hold_11_11 + flow_nic_in_11_11 <= 0
 node_throughput_constraint_12: group_tp_active_12 = 0 -> - 2560 hold_12_1
   - 1280 hold_12_2 - 853.3333333333334 hold_12_3 - 640 hold_12_4
   - 512 hold_12_5 - 426.6666666666667 hold_12_6
   - 365.7142857142857 hold_12_7 + flow_nic_in_12_12 <= 0
 node_throughput_constraint_13: group_tp_active_13 = 0 -> - 2400 hold_13_1
   - 1200 hold_13_2 - 800 hold_13_3 - 600 hold_13_4 - 480 hold_13_5
   - 400 hold_13_6 - 342.8571428571428 hold_13_7 - 300 hold_13_8
   - 266.6666666666667 hold_13_9 - 240 hold_13_10
   - 218.1818181818182 hold_13_11 + flow_nic_in_13_13 <= 0
 node_throughput_constraint_14: group_tp_active_14 = 0 -> - 2400 hold_14_1
   - 1200 hold_14_2 - 800 hold_14_3 - 600 hold_14_4 - 480 hold_14_5
   - 400 hold_14_6 - 342.8571428571428 hold_14_7 - 300 hold_14_8
   - 266.6666666666667 hold_14_9 - 240 hold_14_10
   - 218.1818181818182 hold_14_11 + flow_nic_in_14_14 <= 0
 node_throughput_constraint_15: group_tp_active_15 = 0 -> - 2560 hold_15_1
   - 1280 hold_15_2 - 853.3333333333334 hold_15_3 - 640 hold_15_4
   - 512 hold_15_5 - 426.6666666666667 hold_15_6
   - 365.7142857142857 hold_15_7 + flow_nic_in_15_15 <= 0
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
 tp_off_limit_constr_1_8: group_tp_active_1 = 0 -> hold_1_8 = 0
 tp_off_limit_constr_1_9: group_tp_active_1 = 0 -> hold_1_9 = 0
 tp_off_limit_constr_1_10: group_tp_active_1 = 0 -> hold_1_10 = 0
 tp_off_limit_constr_1_11: group_tp_active_1 = 0 -> hold_1_11 = 0
 tp_off_limit_constr_1_12: group_tp_active_1 = 0 -> hold_1_12 = 0
 tp_off_limit_constr_1_13: group_tp_active_1 = 0 -> hold_1_13 = 0
 tp_off_limit_constr_1_14: group_tp_active_1 = 0 -> hold_1_14 = 0
 tp_off_limit_constr_1_15: group_tp_active_1 = 0 -> hold_1_15 = 0
 tp_off_limit_constr_1_16: group_tp_active_1 = 0 -> hold_1_16 = 0
 tp_off_limit_constr_1_17: group_tp_active_1 = 0 -> hold_1_17 = 0
 tp_off_limit_constr_1_18: group_tp_active_1 = 0 -> hold_1_18 = 0
 tp_off_limit_constr_2_12: group_tp_active_2 = 0 -> hold_2_12 = 0
 tp_off_limit_constr_2_13: group_tp_active_2 = 0 -> hold_2_13 = 0
 tp_off_limit_constr_2_14: group_tp_active_2 = 0 -> hold_2_14 = 0
 tp_off_limit_constr_2_15: group_tp_active_2 = 0 -> hold_2_15 = 0
 tp_off_limit_constr_2_16: group_tp_active_2 = 0 -> hold_2_16 = 0
 tp_off_limit_constr_2_17: group_tp_active_2 = 0 -> hold_2_17 = 0
 tp_off_limit_constr_2_18: group_tp_active_2 = 0 -> hold_2_18 = 0
 tp_off_limit_constr_3_8: group_tp_active_3 = 0 -> hold_3_8 = 0
 tp_off_limit_constr_3_9: group_tp_active_3 = 0 -> hold_3_9 = 0
 tp_off_limit_constr_3_10: group_tp_active_3 = 0 -> hold_3_10 = 0
 tp_off_limit_constr_3_11: group_tp_active_3 = 0 -> hold_3_11 = 0
 tp_off_limit_constr_3_12: group_tp_active_3 = 0 -> hold_3_12 = 0
 tp_off_limit_constr_3_13: group_tp_active_3 = 0 -> hold_3_13 = 0
 tp_off_limit_constr_3_14: group_tp_active_3 = 0 -> hold_3_14 = 0
 tp_off_limit_constr_3_15: group_tp_active_3 = 0 -> hold_3_15 = 0
 tp_off_limit_constr_3_16: group_tp_active_3 = 0 -> hold_3_16 = 0
 tp_off_limit_constr_3_17: group_tp_active_3 = 0 -> hold_3_17 = 0
 tp_off_limit_constr_3_18: group_tp_active_3 = 0 -> hold_3_18 = 0
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
 tp_off_limit_constr_6_12: group_tp_active_6 = 0 -> hold_6_12 = 0
 tp_off_limit_constr_6_13: group_tp_active_6 = 0 -> hold_6_13 = 0
 tp_off_limit_constr_6_14: group_tp_active_6 = 0 -> hold_6_14 = 0
 tp_off_limit_constr_6_15: group_tp_active_6 = 0 -> hold_6_15 = 0
 tp_off_limit_constr_6_16: group_tp_active_6 = 0 -> hold_6_16 = 0
 tp_off_limit_constr_6_17: group_tp_active_6 = 0 -> hold_6_17 = 0
 tp_off_limit_constr_6_18: group_tp_active_6 = 0 -> hold_6_18 = 0
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
 tp_off_limit_constr_8_12: group_tp_active_8 = 0 -> hold_8_12 = 0
 tp_off_limit_constr_8_13: group_tp_active_8 = 0 -> hold_8_13 = 0
 tp_off_limit_constr_8_14: group_tp_active_8 = 0 -> hold_8_14 = 0
 tp_off_limit_constr_8_15: group_tp_active_8 = 0 -> hold_8_15 = 0
 tp_off_limit_constr_8_16: group_tp_active_8 = 0 -> hold_8_16 = 0
 tp_off_limit_constr_8_17: group_tp_active_8 = 0 -> hold_8_17 = 0
 tp_off_limit_constr_8_18: group_tp_active_8 = 0 -> hold_8_18 = 0
 tp_off_limit_constr_9_8: group_tp_active_9 = 0 -> hold_9_8 = 0
 tp_off_limit_constr_9_9: group_tp_active_9 = 0 -> hold_9_9 = 0
 tp_off_limit_constr_9_10: group_tp_active_9 = 0 -> hold_9_10 = 0
 tp_off_limit_constr_9_11: group_tp_active_9 = 0 -> hold_9_11 = 0
 tp_off_limit_constr_9_12: group_tp_active_9 = 0 -> hold_9_12 = 0
 tp_off_limit_constr_9_13: group_tp_active_9 = 0 -> hold_9_13 = 0
 tp_off_limit_constr_9_14: group_tp_active_9 = 0 -> hold_9_14 = 0
 tp_off_limit_constr_9_15: group_tp_active_9 = 0 -> hold_9_15 = 0
 tp_off_limit_constr_9_16: group_tp_active_9 = 0 -> hold_9_16 = 0
 tp_off_limit_constr_9_17: group_tp_active_9 = 0 -> hold_9_17 = 0
 tp_off_limit_constr_9_18: group_tp_active_9 = 0 -> hold_9_18 = 0
 tp_off_limit_constr_11_12: group_tp_active_11 = 0 -> hold_11_12 = 0
 tp_off_limit_constr_11_13: group_tp_active_11 = 0 -> hold_11_13 = 0
 tp_off_limit_constr_11_14: group_tp_active_11 = 0 -> hold_11_14 = 0
 tp_off_limit_constr_11_15: group_tp_active_11 = 0 -> hold_11_15 = 0
 tp_off_limit_constr_11_16: group_tp_active_11 = 0 -> hold_11_16 = 0
 tp_off_limit_constr_11_17: group_tp_active_11 = 0 -> hold_11_17 = 0
 tp_off_limit_constr_11_18: group_tp_active_11 = 0 -> hold_11_18 = 0
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
 tp_off_limit_constr_13_12: group_tp_active_13 = 0 -> hold_13_12 = 0
 tp_off_limit_constr_13_13: group_tp_active_13 = 0 -> hold_13_13 = 0
 tp_off_limit_constr_13_14: group_tp_active_13 = 0 -> hold_13_14 = 0
 tp_off_limit_constr_13_15: group_tp_active_13 = 0 -> hold_13_15 = 0
 tp_off_limit_constr_13_16: group_tp_active_13 = 0 -> hold_13_16 = 0
 tp_off_limit_constr_13_17: group_tp_active_13 = 0 -> hold_13_17 = 0
 tp_off_limit_constr_13_18: group_tp_active_13 = 0 -> hold_13_18 = 0
 tp_off_limit_constr_14_12: group_tp_active_14 = 0 -> hold_14_12 = 0
 tp_off_limit_constr_14_13: group_tp_active_14 = 0 -> hold_14_13 = 0
 tp_off_limit_constr_14_14: group_tp_active_14 = 0 -> hold_14_14 = 0
 tp_off_limit_constr_14_15: group_tp_active_14 = 0 -> hold_14_15 = 0
 tp_off_limit_constr_14_16: group_tp_active_14 = 0 -> hold_14_16 = 0
 tp_off_limit_constr_14_17: group_tp_active_14 = 0 -> hold_14_17 = 0
 tp_off_limit_constr_14_18: group_tp_active_14 = 0 -> hold_14_18 = 0
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
 group_throughput_constr_0: group_tp_active_0 = 1 -> - 2560 hold_0_1
   - 1280 hold_0_2 - 853.3333333333334 hold_0_3 - 640 hold_0_4
   - 512 hold_0_5 - 426.6666666666667 hold_0_6 - 365.7142857142857 hold_0_7
   + flow_nic_in_0_0 <= 0
 group_throughput_constr_1: group_tp_active_1 = 1 -> - 2560 hold_1_1
   - 1280 hold_1_2 - 853.3333333333334 hold_1_3 - 640 hold_1_4
   - 512 hold_1_5 - 426.6666666666667 hold_1_6 - 365.7142857142857 hold_1_7
   + flow_nic_in_1_1 <= 0
 group_throughput_constr_2: group_tp_active_2 = 1 -> - 2400 hold_2_1
   - 1200 hold_2_2 - 800 hold_2_3 - 600 hold_2_4 - 480 hold_2_5
   - 400 hold_2_6 - 342.8571428571428 hold_2_7 - 300 hold_2_8
   - 266.6666666666667 hold_2_9 - 240 hold_2_10
   - 218.1818181818182 hold_2_11 + flow_nic_in_2_2 <= 0
 group_throughput_constr_3: group_tp_active_3 = 1 -> - 2560 hold_3_1
   - 1280 hold_3_2 - 853.3333333333334 hold_3_3 - 640 hold_3_4
   - 512 hold_3_5 - 426.6666666666667 hold_3_6 - 365.7142857142857 hold_3_7
   + flow_nic_in_3_3 <= 0
 group_throughput_constr_4: group_tp_active_4 = 1 -> - 2560 hold_4_1
   - 1280 hold_4_2 - 853.3333333333334 hold_4_3 - 640 hold_4_4
   - 512 hold_4_5 - 426.6666666666667 hold_4_6 - 365.7142857142857 hold_4_7
   + flow_nic_in_4_4 <= 0
 group_throughput_constr_5: group_tp_active_5 = 1 -> - 12440 hold_5_1
   - 6220 hold_5_2 - 4146.666666666667 hold_5_3 - 3110 hold_5_4
   - 2488 hold_5_5 - 2073.333333333333 hold_5_6
   - 1777.142857142857 hold_5_7 - 1555 hold_5_8
   - 1382.222222222222 hold_5_9 - 1244 hold_5_10
   - 1130.909090909091 hold_5_11 - 1036.666666666667 hold_5_12
   - 956.9230769230769 hold_5_13 - 888.5714285714287 hold_5_14
   - 829.3333333333333 hold_5_15 - 777.5 hold_5_16
   - 731.7647058823529 hold_5_17 - 691.1111111111112 hold_5_18
   + flow_nic_in_5_5 <= 0
 group_throughput_constr_6: group_tp_active_6 = 1 -> - 2400 hold_6_1
   - 1200 hold_6_2 - 800 hold_6_3 - 600 hold_6_4 - 480 hold_6_5
   - 400 hold_6_6 - 342.8571428571428 hold_6_7 - 300 hold_6_8
   - 266.6666666666667 hold_6_9 - 240 hold_6_10
   - 218.1818181818182 hold_6_11 + flow_nic_in_6_6 <= 0
 group_throughput_constr_7: group_tp_active_7 = 1 -> - 2560 hold_7_1
   - 1280 hold_7_2 - 853.3333333333334 hold_7_3 - 640 hold_7_4
   - 512 hold_7_5 - 426.6666666666667 hold_7_6 - 365.7142857142857 hold_7_7
   + flow_nic_in_7_7 <= 0
 group_throughput_constr_8: group_tp_active_8 = 1 -> - 2400 hold_8_1
   - 1200 hold_8_2 - 800 hold_8_3 - 600 hold_8_4 - 480 hold_8_5
   - 400 hold_8_6 - 342.8571428571428 hold_8_7 - 300 hold_8_8
   - 266.6666666666667 hold_8_9 - 240 hold_8_10
   - 218.1818181818182 hold_8_11 + flow_nic_in_8_8 <= 0
 group_throughput_constr_9: group_tp_active_9 = 1 -> - 2560 hold_9_1
   - 1280 hold_9_2 - 853.3333333333334 hold_9_3 - 640 hold_9_4
   - 512 hold_9_5 - 426.6666666666667 hold_9_6 - 365.7142857142857 hold_9_7
   + flow_nic_in_9_9 <= 0
 group_throughput_constr_10: group_tp_active_10 = 1 -> - 12440 hold_10_1
   - 6220 hold_10_2 - 4146.666666666667 hold_10_3 - 3110 hold_10_4
   - 2488 hold_10_5 - 2073.333333333333 hold_10_6
   - 1777.142857142857 hold_10_7 - 1555 hold_10_8
   - 1382.222222222222 hold_10_9 - 1244 hold_10_10
   - 1130.909090909091 hold_10_11 - 1036.666666666667 hold_10_12
   - 956.9230769230769 hold_10_13 - 888.5714285714287 hold_10_14
   - 829.3333333333333 hold_10_15 - 777.5 hold_10_16
   - 731.7647058823529 hold_10_17 - 691.1111111111112 hold_10_18
   + flow_nic_in_10_10 <= 0
 group_throughput_constr_11: group_tp_active_11 = 1 -> - 2400 hold_11_1
   - 1200 hold_11_2 - 800 hold_11_3 - 600 hold_11_4 - 480 hold_11_5
   - 400 hold_11_6 - 342.8571428571428 hold_11_7 - 300 hold_11_8
   - 266.6666666666667 hold_11_9 - 240 hold_11_10
   - 218.1818181818182 hold_11_11 + flow_nic_in_11_11 <= 0
 group_throughput_constr_12: group_tp_active_12 = 1 -> - 2560 hold_12_1
   - 1280 hold_12_2 - 853.3333333333334 hold_12_3 - 640 hold_12_4
   - 512 hold_12_5 - 426.6666666666667 hold_12_6
   - 365.7142857142857 hold_12_7 + flow_nic_in_12_12 <= 0
 group_throughput_constr_13: group_tp_active_13 = 1 -> - 2400 hold_13_1
   - 1200 hold_13_2 - 800 hold_13_3 - 600 hold_13_4 - 480 hold_13_5
   - 400 hold_13_6 - 342.8571428571428 hold_13_7 - 300 hold_13_8
   - 266.6666666666667 hold_13_9 - 240 hold_13_10
   - 218.1818181818182 hold_13_11 + flow_nic_in_13_13 <= 0
 group_throughput_constr_14: group_tp_active_14 = 1 -> - 2400 hold_14_1
   - 1200 hold_14_2 - 800 hold_14_3 - 600 hold_14_4 - 480 hold_14_5
   - 400 hold_14_6 - 342.8571428571428 hold_14_7 - 300 hold_14_8
   - 266.6666666666667 hold_14_9 - 240 hold_14_10
   - 218.1818181818182 hold_14_11 + flow_nic_in_14_14 <= 0
 group_throughput_constr_15: group_tp_active_15 = 1 -> - 2560 hold_15_1
   - 1280 hold_15_2 - 853.3333333333334 hold_15_3 - 640 hold_15_4
   - 512 hold_15_5 - 426.6666666666667 hold_15_6
   - 365.7142857142857 hold_15_7 + flow_nic_in_15_15 <= 0
Bounds
Binaries
 hold_0_1 hold_0_2 hold_0_3 hold_0_4 hold_0_5 hold_0_6 hold_0_7 hold_0_8
 hold_0_9 hold_0_10 hold_0_11 hold_0_12 hold_0_13 hold_0_14 hold_0_15
 hold_0_16 hold_0_17 hold_0_18 hold_1_1 hold_1_2 hold_1_3 hold_1_4 hold_1_5
 hold_1_6 hold_1_7 hold_1_8 hold_1_9 hold_1_10 hold_1_11 hold_1_12
 hold_1_13 hold_1_14 hold_1_15 hold_1_16 hold_1_17 hold_1_18 hold_2_1
 hold_2_2 hold_2_3 hold_2_4 hold_2_5 hold_2_6 hold_2_7 hold_2_8 hold_2_9
 hold_2_10 hold_2_11 hold_2_12 hold_2_13 hold_2_14 hold_2_15 hold_2_16
 hold_2_17 hold_2_18 hold_3_1 hold_3_2 hold_3_3 hold_3_4 hold_3_5 hold_3_6
 hold_3_7 hold_3_8 hold_3_9 hold_3_10 hold_3_11 hold_3_12 hold_3_13
 hold_3_14 hold_3_15 hold_3_16 hold_3_17 hold_3_18 hold_4_1 hold_4_2
 hold_4_3 hold_4_4 hold_4_5 hold_4_6 hold_4_7 hold_4_8 hold_4_9 hold_4_10
 hold_4_11 hold_4_12 hold_4_13 hold_4_14 hold_4_15 hold_4_16 hold_4_17
 hold_4_18 hold_5_1 hold_5_2 hold_5_3 hold_5_4 hold_5_5 hold_5_6 hold_5_7
 hold_5_8 hold_5_9 hold_5_10 hold_5_11 hold_5_12 hold_5_13 hold_5_14
 hold_5_15 hold_5_16 hold_5_17 hold_5_18 hold_6_1 hold_6_2 hold_6_3
 hold_6_4 hold_6_5 hold_6_6 hold_6_7 hold_6_8 hold_6_9 hold_6_10 hold_6_11
 hold_6_12 hold_6_13 hold_6_14 hold_6_15 hold_6_16 hold_6_17 hold_6_18
 hold_7_1 hold_7_2 hold_7_3 hold_7_4 hold_7_5 hold_7_6 hold_7_7 hold_7_8
 hold_7_9 hold_7_10 hold_7_11 hold_7_12 hold_7_13 hold_7_14 hold_7_15
 hold_7_16 hold_7_17 hold_7_18 hold_8_1 hold_8_2 hold_8_3 hold_8_4 hold_8_5
 hold_8_6 hold_8_7 hold_8_8 hold_8_9 hold_8_10 hold_8_11 hold_8_12
 hold_8_13 hold_8_14 hold_8_15 hold_8_16 hold_8_17 hold_8_18 hold_9_1
 hold_9_2 hold_9_3 hold_9_4 hold_9_5 hold_9_6 hold_9_7 hold_9_8 hold_9_9
 hold_9_10 hold_9_11 hold_9_12 hold_9_13 hold_9_14 hold_9_15 hold_9_16
 hold_9_17 hold_9_18 hold_10_1 hold_10_2 hold_10_3 hold_10_4 hold_10_5
 hold_10_6 hold_10_7 hold_10_8 hold_10_9 hold_10_10 hold_10_11 hold_10_12
 hold_10_13 hold_10_14 hold_10_15 hold_10_16 hold_10_17 hold_10_18
 hold_11_1 hold_11_2 hold_11_3 hold_11_4 hold_11_5 hold_11_6 hold_11_7
 hold_11_8 hold_11_9 hold_11_10 hold_11_11 hold_11_12 hold_11_13 hold_11_14
 hold_11_15 hold_11_16 hold_11_17 hold_11_18 hold_12_1 hold_12_2 hold_12_3
 hold_12_4 hold_12_5 hold_12_6 hold_12_7 hold_12_8 hold_12_9 hold_12_10
 hold_12_11 hold_12_12 hold_12_13 hold_12_14 hold_12_15 hold_12_16
 hold_12_17 hold_12_18 hold_13_1 hold_13_2 hold_13_3 hold_13_4 hold_13_5
 hold_13_6 hold_13_7 hold_13_8 hold_13_9 hold_13_10 hold_13_11 hold_13_12
 hold_13_13 hold_13_14 hold_13_15 hold_13_16 hold_13_17 hold_13_18
 hold_14_1 hold_14_2 hold_14_3 hold_14_4 hold_14_5 hold_14_6 hold_14_7
 hold_14_8 hold_14_9 hold_14_10 hold_14_11 hold_14_12 hold_14_13 hold_14_14
 hold_14_15 hold_14_16 hold_14_17 hold_14_18 hold_15_1 hold_15_2 hold_15_3
 hold_15_4 hold_15_5 hold_15_6 hold_15_7 hold_15_8 hold_15_9 hold_15_10
 hold_15_11 hold_15_12 hold_15_13 hold_15_14 hold_15_15 hold_15_16
 hold_15_17 hold_15_18 switch_source_nic_in_0 switch_nic_out_0_sink
 switch_nic_out_0_nic_in_1 switch_nic_out_0_nic_in_2
 switch_nic_out_0_nic_in_3 switch_nic_out_0_nic_in_4
 switch_nic_out_0_nic_in_5 switch_nic_out_0_nic_in_6
 switch_nic_out_0_nic_in_7 switch_nic_out_0_nic_in_8
 switch_nic_out_0_nic_in_9 switch_nic_out_0_nic_in_10
 switch_nic_out_0_nic_in_11 switch_nic_out_0_nic_in_12
 switch_nic_out_0_nic_in_13 switch_nic_out_0_nic_in_14
 switch_nic_out_0_nic_in_15 switch_nic_in_0_0 switch_0_nic_out_0
 switch_source_nic_in_1 switch_nic_out_1_sink switch_nic_out_1_nic_in_0
 switch_nic_out_1_nic_in_2 switch_nic_out_1_nic_in_3
 switch_nic_out_1_nic_in_4 switch_nic_out_1_nic_in_5
 switch_nic_out_1_nic_in_6 switch_nic_out_1_nic_in_7
 switch_nic_out_1_nic_in_8 switch_nic_out_1_nic_in_9
 switch_nic_out_1_nic_in_10 switch_nic_out_1_nic_in_11
 switch_nic_out_1_nic_in_12 switch_nic_out_1_nic_in_13
 switch_nic_out_1_nic_in_14 switch_nic_out_1_nic_in_15 switch_nic_in_1_1
 switch_1_nic_out_1 switch_source_nic_in_2 switch_nic_out_2_sink
 switch_nic_out_2_nic_in_0 switch_nic_out_2_nic_in_1
 switch_nic_out_2_nic_in_3 switch_nic_out_2_nic_in_4
 switch_nic_out_2_nic_in_5 switch_nic_out_2_nic_in_6
 switch_nic_out_2_nic_in_7 switch_nic_out_2_nic_in_8
 switch_nic_out_2_nic_in_9 switch_nic_out_2_nic_in_10
 switch_nic_out_2_nic_in_11 switch_nic_out_2_nic_in_12
 switch_nic_out_2_nic_in_13 switch_nic_out_2_nic_in_14
 switch_nic_out_2_nic_in_15 switch_nic_in_2_2 switch_2_nic_out_2
 switch_source_nic_in_3 switch_nic_out_3_sink switch_nic_out_3_nic_in_0
 switch_nic_out_3_nic_in_1 switch_nic_out_3_nic_in_2
 switch_nic_out_3_nic_in_4 switch_nic_out_3_nic_in_5
 switch_nic_out_3_nic_in_6 switch_nic_out_3_nic_in_7
 switch_nic_out_3_nic_in_8 switch_nic_out_3_nic_in_9
 switch_nic_out_3_nic_in_10 switch_nic_out_3_nic_in_11
 switch_nic_out_3_nic_in_12 switch_nic_out_3_nic_in_13
 switch_nic_out_3_nic_in_14 switch_nic_out_3_nic_in_15 switch_nic_in_3_3
 switch_3_nic_out_3 switch_source_nic_in_4 switch_nic_out_4_sink
 switch_nic_out_4_nic_in_0 switch_nic_out_4_nic_in_1
 switch_nic_out_4_nic_in_2 switch_nic_out_4_nic_in_3
 switch_nic_out_4_nic_in_5 switch_nic_out_4_nic_in_6
 switch_nic_out_4_nic_in_7 switch_nic_out_4_nic_in_8
 switch_nic_out_4_nic_in_9 switch_nic_out_4_nic_in_10
 switch_nic_out_4_nic_in_11 switch_nic_out_4_nic_in_12
 switch_nic_out_4_nic_in_13 switch_nic_out_4_nic_in_14
 switch_nic_out_4_nic_in_15 switch_nic_in_4_4 switch_4_nic_out_4
 switch_source_nic_in_5 switch_nic_out_5_sink switch_nic_out_5_nic_in_0
 switch_nic_out_5_nic_in_1 switch_nic_out_5_nic_in_2
 switch_nic_out_5_nic_in_3 switch_nic_out_5_nic_in_4
 switch_nic_out_5_nic_in_6 switch_nic_out_5_nic_in_7
 switch_nic_out_5_nic_in_8 switch_nic_out_5_nic_in_9
 switch_nic_out_5_nic_in_10 switch_nic_out_5_nic_in_11
 switch_nic_out_5_nic_in_12 switch_nic_out_5_nic_in_13
 switch_nic_out_5_nic_in_14 switch_nic_out_5_nic_in_15 switch_nic_in_5_5
 switch_5_nic_out_5 switch_source_nic_in_6 switch_nic_out_6_sink
 switch_nic_out_6_nic_in_0 switch_nic_out_6_nic_in_1
 switch_nic_out_6_nic_in_2 switch_nic_out_6_nic_in_3
 switch_nic_out_6_nic_in_4 switch_nic_out_6_nic_in_5
 switch_nic_out_6_nic_in_7 switch_nic_out_6_nic_in_8
 switch_nic_out_6_nic_in_9 switch_nic_out_6_nic_in_10
 switch_nic_out_6_nic_in_11 switch_nic_out_6_nic_in_12
 switch_nic_out_6_nic_in_13 switch_nic_out_6_nic_in_14
 switch_nic_out_6_nic_in_15 switch_nic_in_6_6 switch_6_nic_out_6
 switch_source_nic_in_7 switch_nic_out_7_sink switch_nic_out_7_nic_in_0
 switch_nic_out_7_nic_in_1 switch_nic_out_7_nic_in_2
 switch_nic_out_7_nic_in_3 switch_nic_out_7_nic_in_4
 switch_nic_out_7_nic_in_5 switch_nic_out_7_nic_in_6
 switch_nic_out_7_nic_in_8 switch_nic_out_7_nic_in_9
 switch_nic_out_7_nic_in_10 switch_nic_out_7_nic_in_11
 switch_nic_out_7_nic_in_12 switch_nic_out_7_nic_in_13
 switch_nic_out_7_nic_in_14 switch_nic_out_7_nic_in_15 switch_nic_in_7_7
 switch_7_nic_out_7 switch_source_nic_in_8 switch_nic_out_8_sink
 switch_nic_out_8_nic_in_0 switch_nic_out_8_nic_in_1
 switch_nic_out_8_nic_in_2 switch_nic_out_8_nic_in_3
 switch_nic_out_8_nic_in_4 switch_nic_out_8_nic_in_5
 switch_nic_out_8_nic_in_6 switch_nic_out_8_nic_in_7
 switch_nic_out_8_nic_in_9 switch_nic_out_8_nic_in_10
 switch_nic_out_8_nic_in_11 switch_nic_out_8_nic_in_12
 switch_nic_out_8_nic_in_13 switch_nic_out_8_nic_in_14
 switch_nic_out_8_nic_in_15 switch_nic_in_8_8 switch_8_nic_out_8
 switch_source_nic_in_9 switch_nic_out_9_sink switch_nic_out_9_nic_in_0
 switch_nic_out_9_nic_in_1 switch_nic_out_9_nic_in_2
 switch_nic_out_9_nic_in_3 switch_nic_out_9_nic_in_4
 switch_nic_out_9_nic_in_5 switch_nic_out_9_nic_in_6
 switch_nic_out_9_nic_in_7 switch_nic_out_9_nic_in_8
 switch_nic_out_9_nic_in_10 switch_nic_out_9_nic_in_11
 switch_nic_out_9_nic_in_12 switch_nic_out_9_nic_in_13
 switch_nic_out_9_nic_in_14 switch_nic_out_9_nic_in_15 switch_nic_in_9_9
 switch_9_nic_out_9 switch_source_nic_in_10 switch_nic_out_10_sink
 switch_nic_out_10_nic_in_0 switch_nic_out_10_nic_in_1
 switch_nic_out_10_nic_in_2 switch_nic_out_10_nic_in_3
 switch_nic_out_10_nic_in_4 switch_nic_out_10_nic_in_5
 switch_nic_out_10_nic_in_6 switch_nic_out_10_nic_in_7
 switch_nic_out_10_nic_in_8 switch_nic_out_10_nic_in_9
 switch_nic_out_10_nic_in_11 switch_nic_out_10_nic_in_12
 switch_nic_out_10_nic_in_13 switch_nic_out_10_nic_in_14
 switch_nic_out_10_nic_in_15 switch_nic_in_10_10 switch_10_nic_out_10
 switch_source_nic_in_11 switch_nic_out_11_sink switch_nic_out_11_nic_in_0
 switch_nic_out_11_nic_in_1 switch_nic_out_11_nic_in_2
 switch_nic_out_11_nic_in_3 switch_nic_out_11_nic_in_4
 switch_nic_out_11_nic_in_5 switch_nic_out_11_nic_in_6
 switch_nic_out_11_nic_in_7 switch_nic_out_11_nic_in_8
 switch_nic_out_11_nic_in_9 switch_nic_out_11_nic_in_10
 switch_nic_out_11_nic_in_12 switch_nic_out_11_nic_in_13
 switch_nic_out_11_nic_in_14 switch_nic_out_11_nic_in_15
 switch_nic_in_11_11 switch_11_nic_out_11 switch_source_nic_in_12
 switch_nic_out_12_sink switch_nic_out_12_nic_in_0
 switch_nic_out_12_nic_in_1 switch_nic_out_12_nic_in_2
 switch_nic_out_12_nic_in_3 switch_nic_out_12_nic_in_4
 switch_nic_out_12_nic_in_5 switch_nic_out_12_nic_in_6
 switch_nic_out_12_nic_in_7 switch_nic_out_12_nic_in_8
 switch_nic_out_12_nic_in_9 switch_nic_out_12_nic_in_10
 switch_nic_out_12_nic_in_11 switch_nic_out_12_nic_in_13
 switch_nic_out_12_nic_in_14 switch_nic_out_12_nic_in_15
 switch_nic_in_12_12 switch_12_nic_out_12 switch_source_nic_in_13
 switch_nic_out_13_sink switch_nic_out_13_nic_in_0
 switch_nic_out_13_nic_in_1 switch_nic_out_13_nic_in_2
 switch_nic_out_13_nic_in_3 switch_nic_out_13_nic_in_4
 switch_nic_out_13_nic_in_5 switch_nic_out_13_nic_in_6
 switch_nic_out_13_nic_in_7 switch_nic_out_13_nic_in_8
 switch_nic_out_13_nic_in_9 switch_nic_out_13_nic_in_10
 switch_nic_out_13_nic_in_11 switch_nic_out_13_nic_in_12
 switch_nic_out_13_nic_in_14 switch_nic_out_13_nic_in_15
 switch_nic_in_13_13 switch_13_nic_out_13 switch_source_nic_in_14
 switch_nic_out_14_sink switch_nic_out_14_nic_in_0
 switch_nic_out_14_nic_in_1 switch_nic_out_14_nic_in_2
 switch_nic_out_14_nic_in_3 switch_nic_out_14_nic_in_4
 switch_nic_out_14_nic_in_5 switch_nic_out_14_nic_in_6
 switch_nic_out_14_nic_in_7 switch_nic_out_14_nic_in_8
 switch_nic_out_14_nic_in_9 switch_nic_out_14_nic_in_10
 switch_nic_out_14_nic_in_11 switch_nic_out_14_nic_in_12
 switch_nic_out_14_nic_in_13 switch_nic_out_14_nic_in_15
 switch_nic_in_14_14 switch_14_nic_out_14 switch_source_nic_in_15
 switch_nic_out_15_sink switch_nic_out_15_nic_in_0
 switch_nic_out_15_nic_in_1 switch_nic_out_15_nic_in_2
 switch_nic_out_15_nic_in_3 switch_nic_out_15_nic_in_4
 switch_nic_out_15_nic_in_5 switch_nic_out_15_nic_in_6
 switch_nic_out_15_nic_in_7 switch_nic_out_15_nic_in_8
 switch_nic_out_15_nic_in_9 switch_nic_out_15_nic_in_10
 switch_nic_out_15_nic_in_11 switch_nic_out_15_nic_in_12
 switch_nic_out_15_nic_in_13 switch_nic_out_15_nic_in_14
 switch_nic_in_15_15 switch_15_nic_out_15 group_tp_active_0
 group_tp_active_1 group_tp_active_2 group_tp_active_3 group_tp_active_4
 group_tp_active_5 group_tp_active_6 group_tp_active_7 group_tp_active_8
 group_tp_active_9 group_tp_active_10 group_tp_active_11 group_tp_active_12
 group_tp_active_13 group_tp_active_14 group_tp_active_15 is_active_0
 is_active_1 is_active_2 is_active_3 is_active_4 is_active_5 is_active_6
 is_active_7 is_active_8 is_active_9 is_active_10 is_active_11 is_active_12
 is_active_13 is_active_14 is_active_15 is_start_zero_0_for_source_nic_in_0
 is_end_complete_0 pair_ok_0_1_for_nic_out_0_nic_in_1
 pair_ok_0_2_for_nic_out_0_nic_in_2 pair_ok_0_3_for_nic_out_0_nic_in_3
 pair_ok_0_4_for_nic_out_0_nic_in_4 pair_ok_0_5_for_nic_out_0_nic_in_5
 pair_ok_0_6_for_nic_out_0_nic_in_6 pair_ok_0_7_for_nic_out_0_nic_in_7
 pair_ok_0_8_for_nic_out_0_nic_in_8 pair_ok_0_9_for_nic_out_0_nic_in_9
 pair_ok_0_10_for_nic_out_0_nic_in_10 pair_ok_0_11_for_nic_out_0_nic_in_11
 pair_ok_0_12_for_nic_out_0_nic_in_12 pair_ok_0_13_for_nic_out_0_nic_in_13
 pair_ok_0_14_for_nic_out_0_nic_in_14 pair_ok_0_15_for_nic_out_0_nic_in_15
 is_start_zero_1_for_source_nic_in_1 is_end_complete_1
 pair_ok_1_0_for_nic_out_1_nic_in_0 pair_ok_1_2_for_nic_out_1_nic_in_2
 pair_ok_1_3_for_nic_out_1_nic_in_3 pair_ok_1_4_for_nic_out_1_nic_in_4
 pair_ok_1_5_for_nic_out_1_nic_in_5 pair_ok_1_6_for_nic_out_1_nic_in_6
 pair_ok_1_7_for_nic_out_1_nic_in_7 pair_ok_1_8_for_nic_out_1_nic_in_8
 pair_ok_1_9_for_nic_out_1_nic_in_9 pair_ok_1_10_for_nic_out_1_nic_in_10
 pair_ok_1_11_for_nic_out_1_nic_in_11 pair_ok_1_12_for_nic_out_1_nic_in_12
 pair_ok_1_13_for_nic_out_1_nic_in_13 pair_ok_1_14_for_nic_out_1_nic_in_14
 pair_ok_1_15_for_nic_out_1_nic_in_15 is_start_zero_2_for_source_nic_in_2
 is_end_complete_2 pair_ok_2_0_for_nic_out_2_nic_in_0
 pair_ok_2_1_for_nic_out_2_nic_in_1 pair_ok_2_3_for_nic_out_2_nic_in_3
 pair_ok_2_4_for_nic_out_2_nic_in_4 pair_ok_2_5_for_nic_out_2_nic_in_5
 pair_ok_2_6_for_nic_out_2_nic_in_6 pair_ok_2_7_for_nic_out_2_nic_in_7
 pair_ok_2_8_for_nic_out_2_nic_in_8 pair_ok_2_9_for_nic_out_2_nic_in_9
 pair_ok_2_10_for_nic_out_2_nic_in_10 pair_ok_2_11_for_nic_out_2_nic_in_11
 pair_ok_2_12_for_nic_out_2_nic_in_12 pair_ok_2_13_for_nic_out_2_nic_in_13
 pair_ok_2_14_for_nic_out_2_nic_in_14 pair_ok_2_15_for_nic_out_2_nic_in_15
 is_start_zero_3_for_source_nic_in_3 is_end_complete_3
 pair_ok_3_0_for_nic_out_3_nic_in_0 pair_ok_3_1_for_nic_out_3_nic_in_1
 pair_ok_3_2_for_nic_out_3_nic_in_2 pair_ok_3_4_for_nic_out_3_nic_in_4
 pair_ok_3_5_for_nic_out_3_nic_in_5 pair_ok_3_6_for_nic_out_3_nic_in_6
 pair_ok_3_7_for_nic_out_3_nic_in_7 pair_ok_3_8_for_nic_out_3_nic_in_8
 pair_ok_3_9_for_nic_out_3_nic_in_9 pair_ok_3_10_for_nic_out_3_nic_in_10
 pair_ok_3_11_for_nic_out_3_nic_in_11 pair_ok_3_12_for_nic_out_3_nic_in_12
 pair_ok_3_13_for_nic_out_3_nic_in_13 pair_ok_3_14_for_nic_out_3_nic_in_14
 pair_ok_3_15_for_nic_out_3_nic_in_15 is_start_zero_4_for_source_nic_in_4
 is_end_complete_4 pair_ok_4_0_for_nic_out_4_nic_in_0
 pair_ok_4_1_for_nic_out_4_nic_in_1 pair_ok_4_2_for_nic_out_4_nic_in_2
 pair_ok_4_3_for_nic_out_4_nic_in_3 pair_ok_4_5_for_nic_out_4_nic_in_5
 pair_ok_4_6_for_nic_out_4_nic_in_6 pair_ok_4_7_for_nic_out_4_nic_in_7
 pair_ok_4_8_for_nic_out_4_nic_in_8 pair_ok_4_9_for_nic_out_4_nic_in_9
 pair_ok_4_10_for_nic_out_4_nic_in_10 pair_ok_4_11_for_nic_out_4_nic_in_11
 pair_ok_4_12_for_nic_out_4_nic_in_12 pair_ok_4_13_for_nic_out_4_nic_in_13
 pair_ok_4_14_for_nic_out_4_nic_in_14 pair_ok_4_15_for_nic_out_4_nic_in_15
 is_start_zero_5_for_source_nic_in_5 is_end_complete_5
 pair_ok_5_0_for_nic_out_5_nic_in_0 pair_ok_5_1_for_nic_out_5_nic_in_1
 pair_ok_5_2_for_nic_out_5_nic_in_2 pair_ok_5_3_for_nic_out_5_nic_in_3
 pair_ok_5_4_for_nic_out_5_nic_in_4 pair_ok_5_6_for_nic_out_5_nic_in_6
 pair_ok_5_7_for_nic_out_5_nic_in_7 pair_ok_5_8_for_nic_out_5_nic_in_8
 pair_ok_5_9_for_nic_out_5_nic_in_9 pair_ok_5_10_for_nic_out_5_nic_in_10
 pair_ok_5_11_for_nic_out_5_nic_in_11 pair_ok_5_12_for_nic_out_5_nic_in_12
 pair_ok_5_13_for_nic_out_5_nic_in_13 pair_ok_5_14_for_nic_out_5_nic_in_14
 pair_ok_5_15_for_nic_out_5_nic_in_15 is_start_zero_6_for_source_nic_in_6
 is_end_complete_6 pair_ok_6_0_for_nic_out_6_nic_in_0
 pair_ok_6_1_for_nic_out_6_nic_in_1 pair_ok_6_2_for_nic_out_6_nic_in_2
 pair_ok_6_3_for_nic_out_6_nic_in_3 pair_ok_6_4_for_nic_out_6_nic_in_4
 pair_ok_6_5_for_nic_out_6_nic_in_5 pair_ok_6_7_for_nic_out_6_nic_in_7
 pair_ok_6_8_for_nic_out_6_nic_in_8 pair_ok_6_9_for_nic_out_6_nic_in_9
 pair_ok_6_10_for_nic_out_6_nic_in_10 pair_ok_6_11_for_nic_out_6_nic_in_11
 pair_ok_6_12_for_nic_out_6_nic_in_12 pair_ok_6_13_for_nic_out_6_nic_in_13
 pair_ok_6_14_for_nic_out_6_nic_in_14 pair_ok_6_15_for_nic_out_6_nic_in_15
 is_start_zero_7_for_source_nic_in_7 is_end_complete_7
 pair_ok_7_0_for_nic_out_7_nic_in_0 pair_ok_7_1_for_nic_out_7_nic_in_1
 pair_ok_7_2_for_nic_out_7_nic_in_2 pair_ok_7_3_for_nic_out_7_nic_in_3
 pair_ok_7_4_for_nic_out_7_nic_in_4 pair_ok_7_5_for_nic_out_7_nic_in_5
 pair_ok_7_6_for_nic_out_7_nic_in_6 pair_ok_7_8_for_nic_out_7_nic_in_8
 pair_ok_7_9_for_nic_out_7_nic_in_9 pair_ok_7_10_for_nic_out_7_nic_in_10
 pair_ok_7_11_for_nic_out_7_nic_in_11 pair_ok_7_12_for_nic_out_7_nic_in_12
 pair_ok_7_13_for_nic_out_7_nic_in_13 pair_ok_7_14_for_nic_out_7_nic_in_14
 pair_ok_7_15_for_nic_out_7_nic_in_15 is_start_zero_8_for_source_nic_in_8
 is_end_complete_8 pair_ok_8_0_for_nic_out_8_nic_in_0
 pair_ok_8_1_for_nic_out_8_nic_in_1 pair_ok_8_2_for_nic_out_8_nic_in_2
 pair_ok_8_3_for_nic_out_8_nic_in_3 pair_ok_8_4_for_nic_out_8_nic_in_4
 pair_ok_8_5_for_nic_out_8_nic_in_5 pair_ok_8_6_for_nic_out_8_nic_in_6
 pair_ok_8_7_for_nic_out_8_nic_in_7 pair_ok_8_9_for_nic_out_8_nic_in_9
 pair_ok_8_10_for_nic_out_8_nic_in_10 pair_ok_8_11_for_nic_out_8_nic_in_11
 pair_ok_8_12_for_nic_out_8_nic_in_12 pair_ok_8_13_for_nic_out_8_nic_in_13
 pair_ok_8_14_for_nic_out_8_nic_in_14 pair_ok_8_15_for_nic_out_8_nic_in_15
 is_start_zero_9_for_source_nic_in_9 is_end_complete_9
 pair_ok_9_0_for_nic_out_9_nic_in_0 pair_ok_9_1_for_nic_out_9_nic_in_1
 pair_ok_9_2_for_nic_out_9_nic_in_2 pair_ok_9_3_for_nic_out_9_nic_in_3
 pair_ok_9_4_for_nic_out_9_nic_in_4 pair_ok_9_5_for_nic_out_9_nic_in_5
 pair_ok_9_6_for_nic_out_9_nic_in_6 pair_ok_9_7_for_nic_out_9_nic_in_7
 pair_ok_9_8_for_nic_out_9_nic_in_8 pair_ok_9_10_for_nic_out_9_nic_in_10
 pair_ok_9_11_for_nic_out_9_nic_in_11 pair_ok_9_12_for_nic_out_9_nic_in_12
 pair_ok_9_13_for_nic_out_9_nic_in_13 pair_ok_9_14_for_nic_out_9_nic_in_14
 pair_ok_9_15_for_nic_out_9_nic_in_15 is_start_zero_10_for_source_nic_in_10
 is_end_complete_10 pair_ok_10_0_for_nic_out_10_nic_in_0
 pair_ok_10_1_for_nic_out_10_nic_in_1 pair_ok_10_2_for_nic_out_10_nic_in_2
 pair_ok_10_3_for_nic_out_10_nic_in_3 pair_ok_10_4_for_nic_out_10_nic_in_4
 pair_ok_10_5_for_nic_out_10_nic_in_5 pair_ok_10_6_for_nic_out_10_nic_in_6
 pair_ok_10_7_for_nic_out_10_nic_in_7 pair_ok_10_8_for_nic_out_10_nic_in_8
 pair_ok_10_9_for_nic_out_10_nic_in_9
 pair_ok_10_11_for_nic_out_10_nic_in_11
 pair_ok_10_12_for_nic_out_10_nic_in_12
 pair_ok_10_13_for_nic_out_10_nic_in_13
 pair_ok_10_14_for_nic_out_10_nic_in_14
 pair_ok_10_15_for_nic_out_10_nic_in_15
 is_start_zero_11_for_source_nic_in_11 is_end_complete_11
 pair_ok_11_0_for_nic_out_11_nic_in_0 pair_ok_11_1_for_nic_out_11_nic_in_1
 pair_ok_11_2_for_nic_out_11_nic_in_2 pair_ok_11_3_for_nic_out_11_nic_in_3
 pair_ok_11_4_for_nic_out_11_nic_in_4 pair_ok_11_5_for_nic_out_11_nic_in_5
 pair_ok_11_6_for_nic_out_11_nic_in_6 pair_ok_11_7_for_nic_out_11_nic_in_7
 pair_ok_11_8_for_nic_out_11_nic_in_8 pair_ok_11_9_for_nic_out_11_nic_in_9
 pair_ok_11_10_for_nic_out_11_nic_in_10
 pair_ok_11_12_for_nic_out_11_nic_in_12
 pair_ok_11_13_for_nic_out_11_nic_in_13
 pair_ok_11_14_for_nic_out_11_nic_in_14
 pair_ok_11_15_for_nic_out_11_nic_in_15
 is_start_zero_12_for_source_nic_in_12 is_end_complete_12
 pair_ok_12_0_for_nic_out_12_nic_in_0 pair_ok_12_1_for_nic_out_12_nic_in_1
 pair_ok_12_2_for_nic_out_12_nic_in_2 pair_ok_12_3_for_nic_out_12_nic_in_3
 pair_ok_12_4_for_nic_out_12_nic_in_4 pair_ok_12_5_for_nic_out_12_nic_in_5
 pair_ok_12_6_for_nic_out_12_nic_in_6 pair_ok_12_7_for_nic_out_12_nic_in_7
 pair_ok_12_8_for_nic_out_12_nic_in_8 pair_ok_12_9_for_nic_out_12_nic_in_9
 pair_ok_12_10_for_nic_out_12_nic_in_10
 pair_ok_12_11_for_nic_out_12_nic_in_11
 pair_ok_12_13_for_nic_out_12_nic_in_13
 pair_ok_12_14_for_nic_out_12_nic_in_14
 pair_ok_12_15_for_nic_out_12_nic_in_15
 is_start_zero_13_for_source_nic_in_13 is_end_complete_13
 pair_ok_13_0_for_nic_out_13_nic_in_0 pair_ok_13_1_for_nic_out_13_nic_in_1
 pair_ok_13_2_for_nic_out_13_nic_in_2 pair_ok_13_3_for_nic_out_13_nic_in_3
 pair_ok_13_4_for_nic_out_13_nic_in_4 pair_ok_13_5_for_nic_out_13_nic_in_5
 pair_ok_13_6_for_nic_out_13_nic_in_6 pair_ok_13_7_for_nic_out_13_nic_in_7
 pair_ok_13_8_for_nic_out_13_nic_in_8 pair_ok_13_9_for_nic_out_13_nic_in_9
 pair_ok_13_10_for_nic_out_13_nic_in_10
 pair_ok_13_11_for_nic_out_13_nic_in_11
 pair_ok_13_12_for_nic_out_13_nic_in_12
 pair_ok_13_14_for_nic_out_13_nic_in_14
 pair_ok_13_15_for_nic_out_13_nic_in_15
 is_start_zero_14_for_source_nic_in_14 is_end_complete_14
 pair_ok_14_0_for_nic_out_14_nic_in_0 pair_ok_14_1_for_nic_out_14_nic_in_1
 pair_ok_14_2_for_nic_out_14_nic_in_2 pair_ok_14_3_for_nic_out_14_nic_in_3
 pair_ok_14_4_for_nic_out_14_nic_in_4 pair_ok_14_5_for_nic_out_14_nic_in_5
 pair_ok_14_6_for_nic_out_14_nic_in_6 pair_ok_14_7_for_nic_out_14_nic_in_7
 pair_ok_14_8_for_nic_out_14_nic_in_8 pair_ok_14_9_for_nic_out_14_nic_in_9
 pair_ok_14_10_for_nic_out_14_nic_in_10
 pair_ok_14_11_for_nic_out_14_nic_in_11
 pair_ok_14_12_for_nic_out_14_nic_in_12
 pair_ok_14_13_for_nic_out_14_nic_in_13
 pair_ok_14_15_for_nic_out_14_nic_in_15
 is_start_zero_15_for_source_nic_in_15 is_end_complete_15
 pair_ok_15_0_for_nic_out_15_nic_in_0 pair_ok_15_1_for_nic_out_15_nic_in_1
 pair_ok_15_2_for_nic_out_15_nic_in_2 pair_ok_15_3_for_nic_out_15_nic_in_3
 pair_ok_15_4_for_nic_out_15_nic_in_4 pair_ok_15_5_for_nic_out_15_nic_in_5
 pair_ok_15_6_for_nic_out_15_nic_in_6 pair_ok_15_7_for_nic_out_15_nic_in_7
 pair_ok_15_8_for_nic_out_15_nic_in_8 pair_ok_15_9_for_nic_out_15_nic_in_9
 pair_ok_15_10_for_nic_out_15_nic_in_10
 pair_ok_15_11_for_nic_out_15_nic_in_11
 pair_ok_15_12_for_nic_out_15_nic_in_12
 pair_ok_15_13_for_nic_out_15_nic_in_13
 pair_ok_15_14_for_nic_out_15_nic_in_14
Generals
 start_0 start_1 start_2 start_3 start_4 start_5 start_6 start_7 start_8
 start_9 start_10 start_11 start_12 start_13 start_14 start_15
End
