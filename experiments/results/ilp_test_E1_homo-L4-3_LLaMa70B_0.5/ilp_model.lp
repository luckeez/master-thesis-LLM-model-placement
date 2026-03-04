\ Model 2026-03-03_16-41-56
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  9.1145833333333313e-04 flow_source_nic_in_0
   + 9.1145833333333313e-04 flow_source_nic_in_1
   + 9.1145833333333313e-04 flow_source_nic_in_2
   + 9.1145833333333313e-04 flow_source_nic_in_3
   + 9.1145833333333313e-04 flow_source_nic_in_4
   + 9.1145833333333313e-04 flow_source_nic_in_5
   + 9.1145833333333313e-04 flow_source_nic_in_6
   + 9.1145833333333313e-04 flow_source_nic_in_7
   + 9.1145833333333313e-04 flow_source_nic_in_8
   + 9.1145833333333313e-04 flow_source_nic_in_9
   + 9.1145833333333313e-04 flow_source_nic_in_10
   + 9.1145833333333313e-04 flow_source_nic_in_11
   + 9.1145833333333313e-04 flow_source_nic_in_12
   + 9.1145833333333313e-04 flow_source_nic_in_13
   + 9.1145833333333313e-04 flow_source_nic_in_14
   + 9.1145833333333313e-04 flow_source_nic_in_15
   + 9.1145833333333313e-04 flow_source_nic_in_16
   + 9.1145833333333313e-04 flow_source_nic_in_17
   + 9.1145833333333313e-04 flow_source_nic_in_18
   + 9.1145833333333313e-04 flow_source_nic_in_19
   + 9.1145833333333313e-04 flow_source_nic_in_20
   + 9.1145833333333313e-04 flow_source_nic_in_21
   + 9.1145833333333313e-04 flow_source_nic_in_22
   + 9.1145833333333313e-04 flow_source_nic_in_23
   + 9.1145833333333313e-04 flow_source_nic_in_24
   + 9.1145833333333313e-04 flow_source_nic_in_25
   + 9.1145833333333313e-04 flow_source_nic_in_26
   + 9.1145833333333313e-04 flow_source_nic_in_27
   + 9.1145833333333313e-04 flow_source_nic_in_28
   + 9.1145833333333313e-04 flow_source_nic_in_29
   + 9.1145833333333313e-04 flow_source_nic_in_30
   + 9.1145833333333313e-04 flow_source_nic_in_31 - 0.015625 is_active_0
   - 0.015625 is_active_1 - 0.015625 is_active_2 - 0.015625 is_active_3
   - 0.015625 is_active_4 - 0.015625 is_active_5 - 0.015625 is_active_6
   - 0.015625 is_active_7 - 0.015625 is_active_8 - 0.015625 is_active_9
   - 0.015625 is_active_10 - 0.015625 is_active_11 - 0.015625 is_active_12
   - 0.015625 is_active_13 - 0.015625 is_active_14 - 0.015625 is_active_15
   - 0.015625 is_active_16 - 0.015625 is_active_17 - 0.015625 is_active_18
   - 0.015625 is_active_19 - 0.015625 is_active_20 - 0.015625 is_active_21
   - 0.015625 is_active_22 - 0.015625 is_active_23 - 0.015625 is_active_24
   - 0.015625 is_active_25 - 0.015625 is_active_26 - 0.015625 is_active_27
   - 0.015625 is_active_28 - 0.015625 is_active_29 - 0.015625 is_active_30
   - 0.015625 is_active_31 + 0 val_end_0 + 0 val_end_1 + 0 val_end_2
   + 0 val_end_3 + 0 val_end_4 + 0 val_end_5 + 0 val_end_6 + 0 val_end_7
   + 0 val_end_8 + 0 val_end_9 + 0 val_end_10 + 0 val_end_11 + 0 val_end_12
   + 0 val_end_13 + 0 val_end_14 + 0 val_end_15 + 0 val_end_16
   + 0 val_end_17 + 0 val_end_18 + 0 val_end_19 + 0 val_end_20
   + 0 val_end_21 + 0 val_end_22 + 0 val_end_23 + 0 val_end_24
   + 0 val_end_25 + 0 val_end_26 + 0 val_end_27 + 0 val_end_28
   + 0 val_end_29 + 0 val_end_30 + 0 val_end_31
Subject To
 mem_active_constr_0: hold_0_1 + hold_0_2 + hold_0_3 + hold_0_4 + hold_0_5
   + hold_0_6 - 81 is_active_0 <= 0
 mem_active_constr2_0: hold_0_1 + hold_0_2 + hold_0_3 + hold_0_4 + hold_0_5
   + hold_0_6 - is_active_0 >= 0
 mem_active_constr_1: hold_1_1 + hold_1_2 + hold_1_3 + hold_1_4 + hold_1_5
   + hold_1_6 - 81 is_active_1 <= 0
 mem_active_constr2_1: hold_1_1 + hold_1_2 + hold_1_3 + hold_1_4 + hold_1_5
   + hold_1_6 - is_active_1 >= 0
 mem_active_constr_2: hold_2_1 + hold_2_2 + hold_2_3 + hold_2_4 + hold_2_5
   + hold_2_6 - 81 is_active_2 <= 0
 mem_active_constr2_2: hold_2_1 + hold_2_2 + hold_2_3 + hold_2_4 + hold_2_5
   + hold_2_6 - is_active_2 >= 0
 mem_active_constr_3: hold_3_1 + hold_3_2 + hold_3_3 + hold_3_4 + hold_3_5
   + hold_3_6 - 81 is_active_3 <= 0
 mem_active_constr2_3: hold_3_1 + hold_3_2 + hold_3_3 + hold_3_4 + hold_3_5
   + hold_3_6 - is_active_3 >= 0
 mem_active_constr_4: hold_4_1 + hold_4_2 + hold_4_3 + hold_4_4 + hold_4_5
   + hold_4_6 - 81 is_active_4 <= 0
 mem_active_constr2_4: hold_4_1 + hold_4_2 + hold_4_3 + hold_4_4 + hold_4_5
   + hold_4_6 - is_active_4 >= 0
 mem_active_constr_5: hold_5_1 + hold_5_2 + hold_5_3 + hold_5_4 + hold_5_5
   + hold_5_6 - 81 is_active_5 <= 0
 mem_active_constr2_5: hold_5_1 + hold_5_2 + hold_5_3 + hold_5_4 + hold_5_5
   + hold_5_6 - is_active_5 >= 0
 mem_active_constr_6: hold_6_1 + hold_6_2 + hold_6_3 + hold_6_4 + hold_6_5
   + hold_6_6 - 81 is_active_6 <= 0
 mem_active_constr2_6: hold_6_1 + hold_6_2 + hold_6_3 + hold_6_4 + hold_6_5
   + hold_6_6 - is_active_6 >= 0
 mem_active_constr_7: hold_7_1 + hold_7_2 + hold_7_3 + hold_7_4 + hold_7_5
   + hold_7_6 - 81 is_active_7 <= 0
 mem_active_constr2_7: hold_7_1 + hold_7_2 + hold_7_3 + hold_7_4 + hold_7_5
   + hold_7_6 - is_active_7 >= 0
 mem_active_constr_8: hold_8_1 + hold_8_2 + hold_8_3 + hold_8_4 + hold_8_5
   + hold_8_6 - 81 is_active_8 <= 0
 mem_active_constr2_8: hold_8_1 + hold_8_2 + hold_8_3 + hold_8_4 + hold_8_5
   + hold_8_6 - is_active_8 >= 0
 mem_active_constr_9: hold_9_1 + hold_9_2 + hold_9_3 + hold_9_4 + hold_9_5
   + hold_9_6 - 81 is_active_9 <= 0
 mem_active_constr2_9: hold_9_1 + hold_9_2 + hold_9_3 + hold_9_4 + hold_9_5
   + hold_9_6 - is_active_9 >= 0
 mem_active_constr_10: hold_10_1 + hold_10_2 + hold_10_3 + hold_10_4
   + hold_10_5 + hold_10_6 - 81 is_active_10 <= 0
 mem_active_constr2_10: hold_10_1 + hold_10_2 + hold_10_3 + hold_10_4
   + hold_10_5 + hold_10_6 - is_active_10 >= 0
 mem_active_constr_11: hold_11_1 + hold_11_2 + hold_11_3 + hold_11_4
   + hold_11_5 + hold_11_6 - 81 is_active_11 <= 0
 mem_active_constr2_11: hold_11_1 + hold_11_2 + hold_11_3 + hold_11_4
   + hold_11_5 + hold_11_6 - is_active_11 >= 0
 mem_active_constr_12: hold_12_1 + hold_12_2 + hold_12_3 + hold_12_4
   + hold_12_5 + hold_12_6 - 81 is_active_12 <= 0
 mem_active_constr2_12: hold_12_1 + hold_12_2 + hold_12_3 + hold_12_4
   + hold_12_5 + hold_12_6 - is_active_12 >= 0
 mem_active_constr_13: hold_13_1 + hold_13_2 + hold_13_3 + hold_13_4
   + hold_13_5 + hold_13_6 - 81 is_active_13 <= 0
 mem_active_constr2_13: hold_13_1 + hold_13_2 + hold_13_3 + hold_13_4
   + hold_13_5 + hold_13_6 - is_active_13 >= 0
 mem_active_constr_14: hold_14_1 + hold_14_2 + hold_14_3 + hold_14_4
   + hold_14_5 + hold_14_6 - 81 is_active_14 <= 0
 mem_active_constr2_14: hold_14_1 + hold_14_2 + hold_14_3 + hold_14_4
   + hold_14_5 + hold_14_6 - is_active_14 >= 0
 mem_active_constr_15: hold_15_1 + hold_15_2 + hold_15_3 + hold_15_4
   + hold_15_5 + hold_15_6 - 81 is_active_15 <= 0
 mem_active_constr2_15: hold_15_1 + hold_15_2 + hold_15_3 + hold_15_4
   + hold_15_5 + hold_15_6 - is_active_15 >= 0
 mem_active_constr_16: hold_16_1 + hold_16_2 + hold_16_3 + hold_16_4
   + hold_16_5 + hold_16_6 - 81 is_active_16 <= 0
 mem_active_constr2_16: hold_16_1 + hold_16_2 + hold_16_3 + hold_16_4
   + hold_16_5 + hold_16_6 - is_active_16 >= 0
 mem_active_constr_17: hold_17_1 + hold_17_2 + hold_17_3 + hold_17_4
   + hold_17_5 + hold_17_6 - 81 is_active_17 <= 0
 mem_active_constr2_17: hold_17_1 + hold_17_2 + hold_17_3 + hold_17_4
   + hold_17_5 + hold_17_6 - is_active_17 >= 0
 mem_active_constr_18: hold_18_1 + hold_18_2 + hold_18_3 + hold_18_4
   + hold_18_5 + hold_18_6 - 81 is_active_18 <= 0
 mem_active_constr2_18: hold_18_1 + hold_18_2 + hold_18_3 + hold_18_4
   + hold_18_5 + hold_18_6 - is_active_18 >= 0
 mem_active_constr_19: hold_19_1 + hold_19_2 + hold_19_3 + hold_19_4
   + hold_19_5 + hold_19_6 - 81 is_active_19 <= 0
 mem_active_constr2_19: hold_19_1 + hold_19_2 + hold_19_3 + hold_19_4
   + hold_19_5 + hold_19_6 - is_active_19 >= 0
 mem_active_constr_20: hold_20_1 + hold_20_2 + hold_20_3 + hold_20_4
   + hold_20_5 + hold_20_6 - 81 is_active_20 <= 0
 mem_active_constr2_20: hold_20_1 + hold_20_2 + hold_20_3 + hold_20_4
   + hold_20_5 + hold_20_6 - is_active_20 >= 0
 mem_active_constr_21: hold_21_1 + hold_21_2 + hold_21_3 + hold_21_4
   + hold_21_5 + hold_21_6 - 81 is_active_21 <= 0
 mem_active_constr2_21: hold_21_1 + hold_21_2 + hold_21_3 + hold_21_4
   + hold_21_5 + hold_21_6 - is_active_21 >= 0
 mem_active_constr_22: hold_22_1 + hold_22_2 + hold_22_3 + hold_22_4
   + hold_22_5 + hold_22_6 - 81 is_active_22 <= 0
 mem_active_constr2_22: hold_22_1 + hold_22_2 + hold_22_3 + hold_22_4
   + hold_22_5 + hold_22_6 - is_active_22 >= 0
 mem_active_constr_23: hold_23_1 + hold_23_2 + hold_23_3 + hold_23_4
   + hold_23_5 + hold_23_6 - 81 is_active_23 <= 0
 mem_active_constr2_23: hold_23_1 + hold_23_2 + hold_23_3 + hold_23_4
   + hold_23_5 + hold_23_6 - is_active_23 >= 0
 mem_active_constr_24: hold_24_1 + hold_24_2 + hold_24_3 + hold_24_4
   + hold_24_5 + hold_24_6 - 81 is_active_24 <= 0
 mem_active_constr2_24: hold_24_1 + hold_24_2 + hold_24_3 + hold_24_4
   + hold_24_5 + hold_24_6 - is_active_24 >= 0
 mem_active_constr_25: hold_25_1 + hold_25_2 + hold_25_3 + hold_25_4
   + hold_25_5 + hold_25_6 - 81 is_active_25 <= 0
 mem_active_constr2_25: hold_25_1 + hold_25_2 + hold_25_3 + hold_25_4
   + hold_25_5 + hold_25_6 - is_active_25 >= 0
 mem_active_constr_26: hold_26_1 + hold_26_2 + hold_26_3 + hold_26_4
   + hold_26_5 + hold_26_6 - 81 is_active_26 <= 0
 mem_active_constr2_26: hold_26_1 + hold_26_2 + hold_26_3 + hold_26_4
   + hold_26_5 + hold_26_6 - is_active_26 >= 0
 mem_active_constr_27: hold_27_1 + hold_27_2 + hold_27_3 + hold_27_4
   + hold_27_5 + hold_27_6 - 81 is_active_27 <= 0
 mem_active_constr2_27: hold_27_1 + hold_27_2 + hold_27_3 + hold_27_4
   + hold_27_5 + hold_27_6 - is_active_27 >= 0
 mem_active_constr_28: hold_28_1 + hold_28_2 + hold_28_3 + hold_28_4
   + hold_28_5 + hold_28_6 - 81 is_active_28 <= 0
 mem_active_constr2_28: hold_28_1 + hold_28_2 + hold_28_3 + hold_28_4
   + hold_28_5 + hold_28_6 - is_active_28 >= 0
 mem_active_constr_29: hold_29_1 + hold_29_2 + hold_29_3 + hold_29_4
   + hold_29_5 + hold_29_6 - 81 is_active_29 <= 0
 mem_active_constr2_29: hold_29_1 + hold_29_2 + hold_29_3 + hold_29_4
   + hold_29_5 + hold_29_6 - is_active_29 >= 0
 mem_active_constr_30: hold_30_1 + hold_30_2 + hold_30_3 + hold_30_4
   + hold_30_5 + hold_30_6 - 81 is_active_30 <= 0
 mem_active_constr2_30: hold_30_1 + hold_30_2 + hold_30_3 + hold_30_4
   + hold_30_5 + hold_30_6 - is_active_30 >= 0
 mem_active_constr_31: hold_31_1 + hold_31_2 + hold_31_3 + hold_31_4
   + hold_31_5 + hold_31_6 - 81 is_active_31 <= 0
 mem_active_constr2_31: hold_31_1 + hold_31_2 + hold_31_3 + hold_31_4
   + hold_31_5 + hold_31_6 - is_active_31 >= 0
 hold_constraint_0: hold_0_1 + hold_0_2 + hold_0_3 + hold_0_4 + hold_0_5
   + hold_0_6 <= 1
 end_constraint_0: start_0 + hold_0_1 + 2 hold_0_2 + 3 hold_0_3
   + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 <= 80
 hold_constraint_1: hold_1_1 + hold_1_2 + hold_1_3 + hold_1_4 + hold_1_5
   + hold_1_6 <= 1
 end_constraint_1: start_1 + hold_1_1 + 2 hold_1_2 + 3 hold_1_3
   + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 <= 80
 hold_constraint_2: hold_2_1 + hold_2_2 + hold_2_3 + hold_2_4 + hold_2_5
   + hold_2_6 <= 1
 end_constraint_2: start_2 + hold_2_1 + 2 hold_2_2 + 3 hold_2_3
   + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 <= 80
 hold_constraint_3: hold_3_1 + hold_3_2 + hold_3_3 + hold_3_4 + hold_3_5
   + hold_3_6 <= 1
 end_constraint_3: start_3 + hold_3_1 + 2 hold_3_2 + 3 hold_3_3
   + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 <= 80
 hold_constraint_4: hold_4_1 + hold_4_2 + hold_4_3 + hold_4_4 + hold_4_5
   + hold_4_6 <= 1
 end_constraint_4: start_4 + hold_4_1 + 2 hold_4_2 + 3 hold_4_3
   + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 <= 80
 hold_constraint_5: hold_5_1 + hold_5_2 + hold_5_3 + hold_5_4 + hold_5_5
   + hold_5_6 <= 1
 end_constraint_5: start_5 + hold_5_1 + 2 hold_5_2 + 3 hold_5_3
   + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 <= 80
 hold_constraint_6: hold_6_1 + hold_6_2 + hold_6_3 + hold_6_4 + hold_6_5
   + hold_6_6 <= 1
 end_constraint_6: start_6 + hold_6_1 + 2 hold_6_2 + 3 hold_6_3
   + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 <= 80
 hold_constraint_7: hold_7_1 + hold_7_2 + hold_7_3 + hold_7_4 + hold_7_5
   + hold_7_6 <= 1
 end_constraint_7: start_7 + hold_7_1 + 2 hold_7_2 + 3 hold_7_3
   + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 <= 80
 hold_constraint_8: hold_8_1 + hold_8_2 + hold_8_3 + hold_8_4 + hold_8_5
   + hold_8_6 <= 1
 end_constraint_8: start_8 + hold_8_1 + 2 hold_8_2 + 3 hold_8_3
   + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 <= 80
 hold_constraint_9: hold_9_1 + hold_9_2 + hold_9_3 + hold_9_4 + hold_9_5
   + hold_9_6 <= 1
 end_constraint_9: start_9 + hold_9_1 + 2 hold_9_2 + 3 hold_9_3
   + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 <= 80
 hold_constraint_10: hold_10_1 + hold_10_2 + hold_10_3 + hold_10_4
   + hold_10_5 + hold_10_6 <= 1
 end_constraint_10: start_10 + hold_10_1 + 2 hold_10_2 + 3 hold_10_3
   + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 <= 80
 hold_constraint_11: hold_11_1 + hold_11_2 + hold_11_3 + hold_11_4
   + hold_11_5 + hold_11_6 <= 1
 end_constraint_11: start_11 + hold_11_1 + 2 hold_11_2 + 3 hold_11_3
   + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 <= 80
 hold_constraint_12: hold_12_1 + hold_12_2 + hold_12_3 + hold_12_4
   + hold_12_5 + hold_12_6 <= 1
 end_constraint_12: start_12 + hold_12_1 + 2 hold_12_2 + 3 hold_12_3
   + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6 <= 80
 hold_constraint_13: hold_13_1 + hold_13_2 + hold_13_3 + hold_13_4
   + hold_13_5 + hold_13_6 <= 1
 end_constraint_13: start_13 + hold_13_1 + 2 hold_13_2 + 3 hold_13_3
   + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6 <= 80
 hold_constraint_14: hold_14_1 + hold_14_2 + hold_14_3 + hold_14_4
   + hold_14_5 + hold_14_6 <= 1
 end_constraint_14: start_14 + hold_14_1 + 2 hold_14_2 + 3 hold_14_3
   + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6 <= 80
 hold_constraint_15: hold_15_1 + hold_15_2 + hold_15_3 + hold_15_4
   + hold_15_5 + hold_15_6 <= 1
 end_constraint_15: start_15 + hold_15_1 + 2 hold_15_2 + 3 hold_15_3
   + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6 <= 80
 hold_constraint_16: hold_16_1 + hold_16_2 + hold_16_3 + hold_16_4
   + hold_16_5 + hold_16_6 <= 1
 end_constraint_16: start_16 + hold_16_1 + 2 hold_16_2 + 3 hold_16_3
   + 4 hold_16_4 + 5 hold_16_5 + 6 hold_16_6 <= 80
 hold_constraint_17: hold_17_1 + hold_17_2 + hold_17_3 + hold_17_4
   + hold_17_5 + hold_17_6 <= 1
 end_constraint_17: start_17 + hold_17_1 + 2 hold_17_2 + 3 hold_17_3
   + 4 hold_17_4 + 5 hold_17_5 + 6 hold_17_6 <= 80
 hold_constraint_18: hold_18_1 + hold_18_2 + hold_18_3 + hold_18_4
   + hold_18_5 + hold_18_6 <= 1
 end_constraint_18: start_18 + hold_18_1 + 2 hold_18_2 + 3 hold_18_3
   + 4 hold_18_4 + 5 hold_18_5 + 6 hold_18_6 <= 80
 hold_constraint_19: hold_19_1 + hold_19_2 + hold_19_3 + hold_19_4
   + hold_19_5 + hold_19_6 <= 1
 end_constraint_19: start_19 + hold_19_1 + 2 hold_19_2 + 3 hold_19_3
   + 4 hold_19_4 + 5 hold_19_5 + 6 hold_19_6 <= 80
 hold_constraint_20: hold_20_1 + hold_20_2 + hold_20_3 + hold_20_4
   + hold_20_5 + hold_20_6 <= 1
 end_constraint_20: start_20 + hold_20_1 + 2 hold_20_2 + 3 hold_20_3
   + 4 hold_20_4 + 5 hold_20_5 + 6 hold_20_6 <= 80
 hold_constraint_21: hold_21_1 + hold_21_2 + hold_21_3 + hold_21_4
   + hold_21_5 + hold_21_6 <= 1
 end_constraint_21: start_21 + hold_21_1 + 2 hold_21_2 + 3 hold_21_3
   + 4 hold_21_4 + 5 hold_21_5 + 6 hold_21_6 <= 80
 hold_constraint_22: hold_22_1 + hold_22_2 + hold_22_3 + hold_22_4
   + hold_22_5 + hold_22_6 <= 1
 end_constraint_22: start_22 + hold_22_1 + 2 hold_22_2 + 3 hold_22_3
   + 4 hold_22_4 + 5 hold_22_5 + 6 hold_22_6 <= 80
 hold_constraint_23: hold_23_1 + hold_23_2 + hold_23_3 + hold_23_4
   + hold_23_5 + hold_23_6 <= 1
 end_constraint_23: start_23 + hold_23_1 + 2 hold_23_2 + 3 hold_23_3
   + 4 hold_23_4 + 5 hold_23_5 + 6 hold_23_6 <= 80
 hold_constraint_24: hold_24_1 + hold_24_2 + hold_24_3 + hold_24_4
   + hold_24_5 + hold_24_6 <= 1
 end_constraint_24: start_24 + hold_24_1 + 2 hold_24_2 + 3 hold_24_3
   + 4 hold_24_4 + 5 hold_24_5 + 6 hold_24_6 <= 80
 hold_constraint_25: hold_25_1 + hold_25_2 + hold_25_3 + hold_25_4
   + hold_25_5 + hold_25_6 <= 1
 end_constraint_25: start_25 + hold_25_1 + 2 hold_25_2 + 3 hold_25_3
   + 4 hold_25_4 + 5 hold_25_5 + 6 hold_25_6 <= 80
 hold_constraint_26: hold_26_1 + hold_26_2 + hold_26_3 + hold_26_4
   + hold_26_5 + hold_26_6 <= 1
 end_constraint_26: start_26 + hold_26_1 + 2 hold_26_2 + 3 hold_26_3
   + 4 hold_26_4 + 5 hold_26_5 + 6 hold_26_6 <= 80
 hold_constraint_27: hold_27_1 + hold_27_2 + hold_27_3 + hold_27_4
   + hold_27_5 + hold_27_6 <= 1
 end_constraint_27: start_27 + hold_27_1 + 2 hold_27_2 + 3 hold_27_3
   + 4 hold_27_4 + 5 hold_27_5 + 6 hold_27_6 <= 80
 hold_constraint_28: hold_28_1 + hold_28_2 + hold_28_3 + hold_28_4
   + hold_28_5 + hold_28_6 <= 1
 end_constraint_28: start_28 + hold_28_1 + 2 hold_28_2 + 3 hold_28_3
   + 4 hold_28_4 + 5 hold_28_5 + 6 hold_28_6 <= 80
 hold_constraint_29: hold_29_1 + hold_29_2 + hold_29_3 + hold_29_4
   + hold_29_5 + hold_29_6 <= 1
 end_constraint_29: start_29 + hold_29_1 + 2 hold_29_2 + 3 hold_29_3
   + 4 hold_29_4 + 5 hold_29_5 + 6 hold_29_6 <= 80
 hold_constraint_30: hold_30_1 + hold_30_2 + hold_30_3 + hold_30_4
   + hold_30_5 + hold_30_6 <= 1
 end_constraint_30: start_30 + hold_30_1 + 2 hold_30_2 + 3 hold_30_3
   + 4 hold_30_4 + 5 hold_30_5 + 6 hold_30_6 <= 80
 hold_constraint_31: hold_31_1 + hold_31_2 + hold_31_3 + hold_31_4
   + hold_31_5 + hold_31_6 <= 1
 end_constraint_31: start_31 + hold_31_1 + 2 hold_31_2 + 3 hold_31_3
   + 4 hold_31_4 + 5 hold_31_5 + 6 hold_31_6 <= 80
 node_flow_constraint_nic_in_0: flow_source_nic_in_0 - flow_nic_in_0_0
   + flow_nic_out_1_nic_in_0 + flow_nic_out_2_nic_in_0
   + flow_nic_out_3_nic_in_0 + flow_nic_out_4_nic_in_0
   + flow_nic_out_5_nic_in_0 + flow_nic_out_6_nic_in_0
   + flow_nic_out_7_nic_in_0 + flow_nic_out_8_nic_in_0
   + flow_nic_out_9_nic_in_0 + flow_nic_out_10_nic_in_0
   + flow_nic_out_11_nic_in_0 + flow_nic_out_12_nic_in_0
   + flow_nic_out_13_nic_in_0 + flow_nic_out_14_nic_in_0
   + flow_nic_out_15_nic_in_0 + flow_nic_out_16_nic_in_0
   + flow_nic_out_17_nic_in_0 + flow_nic_out_18_nic_in_0
   + flow_nic_out_19_nic_in_0 + flow_nic_out_20_nic_in_0
   + flow_nic_out_21_nic_in_0 + flow_nic_out_22_nic_in_0
   + flow_nic_out_23_nic_in_0 + flow_nic_out_24_nic_in_0
   + flow_nic_out_25_nic_in_0 + flow_nic_out_26_nic_in_0
   + flow_nic_out_27_nic_in_0 + flow_nic_out_28_nic_in_0
   + flow_nic_out_29_nic_in_0 + flow_nic_out_30_nic_in_0
   + flow_nic_out_31_nic_in_0 = 0
 node_flow_constraint_nic_out_0: - flow_nic_out_0_sink
   - flow_nic_out_0_nic_in_1 - flow_nic_out_0_nic_in_2
   - flow_nic_out_0_nic_in_3 - flow_nic_out_0_nic_in_4
   - flow_nic_out_0_nic_in_5 - flow_nic_out_0_nic_in_6
   - flow_nic_out_0_nic_in_7 - flow_nic_out_0_nic_in_8
   - flow_nic_out_0_nic_in_9 - flow_nic_out_0_nic_in_10
   - flow_nic_out_0_nic_in_11 - flow_nic_out_0_nic_in_12
   - flow_nic_out_0_nic_in_13 - flow_nic_out_0_nic_in_14
   - flow_nic_out_0_nic_in_15 - flow_nic_out_0_nic_in_16
   - flow_nic_out_0_nic_in_17 - flow_nic_out_0_nic_in_18
   - flow_nic_out_0_nic_in_19 - flow_nic_out_0_nic_in_20
   - flow_nic_out_0_nic_in_21 - flow_nic_out_0_nic_in_22
   - flow_nic_out_0_nic_in_23 - flow_nic_out_0_nic_in_24
   - flow_nic_out_0_nic_in_25 - flow_nic_out_0_nic_in_26
   - flow_nic_out_0_nic_in_27 - flow_nic_out_0_nic_in_28
   - flow_nic_out_0_nic_in_29 - flow_nic_out_0_nic_in_30
   - flow_nic_out_0_nic_in_31 + flow_0_nic_out_0 = 0
 node_flow_constraint_nic_in_1: flow_nic_out_0_nic_in_1
   + flow_source_nic_in_1 - flow_nic_in_1_1 + flow_nic_out_2_nic_in_1
   + flow_nic_out_3_nic_in_1 + flow_nic_out_4_nic_in_1
   + flow_nic_out_5_nic_in_1 + flow_nic_out_6_nic_in_1
   + flow_nic_out_7_nic_in_1 + flow_nic_out_8_nic_in_1
   + flow_nic_out_9_nic_in_1 + flow_nic_out_10_nic_in_1
   + flow_nic_out_11_nic_in_1 + flow_nic_out_12_nic_in_1
   + flow_nic_out_13_nic_in_1 + flow_nic_out_14_nic_in_1
   + flow_nic_out_15_nic_in_1 + flow_nic_out_16_nic_in_1
   + flow_nic_out_17_nic_in_1 + flow_nic_out_18_nic_in_1
   + flow_nic_out_19_nic_in_1 + flow_nic_out_20_nic_in_1
   + flow_nic_out_21_nic_in_1 + flow_nic_out_22_nic_in_1
   + flow_nic_out_23_nic_in_1 + flow_nic_out_24_nic_in_1
   + flow_nic_out_25_nic_in_1 + flow_nic_out_26_nic_in_1
   + flow_nic_out_27_nic_in_1 + flow_nic_out_28_nic_in_1
   + flow_nic_out_29_nic_in_1 + flow_nic_out_30_nic_in_1
   + flow_nic_out_31_nic_in_1 = 0
 node_flow_constraint_nic_out_1: - flow_nic_out_1_sink
   - flow_nic_out_1_nic_in_0 - flow_nic_out_1_nic_in_2
   - flow_nic_out_1_nic_in_3 - flow_nic_out_1_nic_in_4
   - flow_nic_out_1_nic_in_5 - flow_nic_out_1_nic_in_6
   - flow_nic_out_1_nic_in_7 - flow_nic_out_1_nic_in_8
   - flow_nic_out_1_nic_in_9 - flow_nic_out_1_nic_in_10
   - flow_nic_out_1_nic_in_11 - flow_nic_out_1_nic_in_12
   - flow_nic_out_1_nic_in_13 - flow_nic_out_1_nic_in_14
   - flow_nic_out_1_nic_in_15 - flow_nic_out_1_nic_in_16
   - flow_nic_out_1_nic_in_17 - flow_nic_out_1_nic_in_18
   - flow_nic_out_1_nic_in_19 - flow_nic_out_1_nic_in_20
   - flow_nic_out_1_nic_in_21 - flow_nic_out_1_nic_in_22
   - flow_nic_out_1_nic_in_23 - flow_nic_out_1_nic_in_24
   - flow_nic_out_1_nic_in_25 - flow_nic_out_1_nic_in_26
   - flow_nic_out_1_nic_in_27 - flow_nic_out_1_nic_in_28
   - flow_nic_out_1_nic_in_29 - flow_nic_out_1_nic_in_30
   - flow_nic_out_1_nic_in_31 + flow_1_nic_out_1 = 0
 node_flow_constraint_nic_in_2: flow_nic_out_0_nic_in_2
   + flow_nic_out_1_nic_in_2 + flow_source_nic_in_2 - flow_nic_in_2_2
   + flow_nic_out_3_nic_in_2 + flow_nic_out_4_nic_in_2
   + flow_nic_out_5_nic_in_2 + flow_nic_out_6_nic_in_2
   + flow_nic_out_7_nic_in_2 + flow_nic_out_8_nic_in_2
   + flow_nic_out_9_nic_in_2 + flow_nic_out_10_nic_in_2
   + flow_nic_out_11_nic_in_2 + flow_nic_out_12_nic_in_2
   + flow_nic_out_13_nic_in_2 + flow_nic_out_14_nic_in_2
   + flow_nic_out_15_nic_in_2 + flow_nic_out_16_nic_in_2
   + flow_nic_out_17_nic_in_2 + flow_nic_out_18_nic_in_2
   + flow_nic_out_19_nic_in_2 + flow_nic_out_20_nic_in_2
   + flow_nic_out_21_nic_in_2 + flow_nic_out_22_nic_in_2
   + flow_nic_out_23_nic_in_2 + flow_nic_out_24_nic_in_2
   + flow_nic_out_25_nic_in_2 + flow_nic_out_26_nic_in_2
   + flow_nic_out_27_nic_in_2 + flow_nic_out_28_nic_in_2
   + flow_nic_out_29_nic_in_2 + flow_nic_out_30_nic_in_2
   + flow_nic_out_31_nic_in_2 = 0
 node_flow_constraint_nic_out_2: - flow_nic_out_2_sink
   - flow_nic_out_2_nic_in_0 - flow_nic_out_2_nic_in_1
   - flow_nic_out_2_nic_in_3 - flow_nic_out_2_nic_in_4
   - flow_nic_out_2_nic_in_5 - flow_nic_out_2_nic_in_6
   - flow_nic_out_2_nic_in_7 - flow_nic_out_2_nic_in_8
   - flow_nic_out_2_nic_in_9 - flow_nic_out_2_nic_in_10
   - flow_nic_out_2_nic_in_11 - flow_nic_out_2_nic_in_12
   - flow_nic_out_2_nic_in_13 - flow_nic_out_2_nic_in_14
   - flow_nic_out_2_nic_in_15 - flow_nic_out_2_nic_in_16
   - flow_nic_out_2_nic_in_17 - flow_nic_out_2_nic_in_18
   - flow_nic_out_2_nic_in_19 - flow_nic_out_2_nic_in_20
   - flow_nic_out_2_nic_in_21 - flow_nic_out_2_nic_in_22
   - flow_nic_out_2_nic_in_23 - flow_nic_out_2_nic_in_24
   - flow_nic_out_2_nic_in_25 - flow_nic_out_2_nic_in_26
   - flow_nic_out_2_nic_in_27 - flow_nic_out_2_nic_in_28
   - flow_nic_out_2_nic_in_29 - flow_nic_out_2_nic_in_30
   - flow_nic_out_2_nic_in_31 + flow_2_nic_out_2 = 0
 node_flow_constraint_nic_in_3: flow_nic_out_0_nic_in_3
   + flow_nic_out_1_nic_in_3 + flow_nic_out_2_nic_in_3
   + flow_source_nic_in_3 - flow_nic_in_3_3 + flow_nic_out_4_nic_in_3
   + flow_nic_out_5_nic_in_3 + flow_nic_out_6_nic_in_3
   + flow_nic_out_7_nic_in_3 + flow_nic_out_8_nic_in_3
   + flow_nic_out_9_nic_in_3 + flow_nic_out_10_nic_in_3
   + flow_nic_out_11_nic_in_3 + flow_nic_out_12_nic_in_3
   + flow_nic_out_13_nic_in_3 + flow_nic_out_14_nic_in_3
   + flow_nic_out_15_nic_in_3 + flow_nic_out_16_nic_in_3
   + flow_nic_out_17_nic_in_3 + flow_nic_out_18_nic_in_3
   + flow_nic_out_19_nic_in_3 + flow_nic_out_20_nic_in_3
   + flow_nic_out_21_nic_in_3 + flow_nic_out_22_nic_in_3
   + flow_nic_out_23_nic_in_3 + flow_nic_out_24_nic_in_3
   + flow_nic_out_25_nic_in_3 + flow_nic_out_26_nic_in_3
   + flow_nic_out_27_nic_in_3 + flow_nic_out_28_nic_in_3
   + flow_nic_out_29_nic_in_3 + flow_nic_out_30_nic_in_3
   + flow_nic_out_31_nic_in_3 = 0
 node_flow_constraint_nic_out_3: - flow_nic_out_3_sink
   - flow_nic_out_3_nic_in_0 - flow_nic_out_3_nic_in_1
   - flow_nic_out_3_nic_in_2 - flow_nic_out_3_nic_in_4
   - flow_nic_out_3_nic_in_5 - flow_nic_out_3_nic_in_6
   - flow_nic_out_3_nic_in_7 - flow_nic_out_3_nic_in_8
   - flow_nic_out_3_nic_in_9 - flow_nic_out_3_nic_in_10
   - flow_nic_out_3_nic_in_11 - flow_nic_out_3_nic_in_12
   - flow_nic_out_3_nic_in_13 - flow_nic_out_3_nic_in_14
   - flow_nic_out_3_nic_in_15 - flow_nic_out_3_nic_in_16
   - flow_nic_out_3_nic_in_17 - flow_nic_out_3_nic_in_18
   - flow_nic_out_3_nic_in_19 - flow_nic_out_3_nic_in_20
   - flow_nic_out_3_nic_in_21 - flow_nic_out_3_nic_in_22
   - flow_nic_out_3_nic_in_23 - flow_nic_out_3_nic_in_24
   - flow_nic_out_3_nic_in_25 - flow_nic_out_3_nic_in_26
   - flow_nic_out_3_nic_in_27 - flow_nic_out_3_nic_in_28
   - flow_nic_out_3_nic_in_29 - flow_nic_out_3_nic_in_30
   - flow_nic_out_3_nic_in_31 + flow_3_nic_out_3 = 0
 node_flow_constraint_nic_in_4: flow_nic_out_0_nic_in_4
   + flow_nic_out_1_nic_in_4 + flow_nic_out_2_nic_in_4
   + flow_nic_out_3_nic_in_4 + flow_source_nic_in_4 - flow_nic_in_4_4
   + flow_nic_out_5_nic_in_4 + flow_nic_out_6_nic_in_4
   + flow_nic_out_7_nic_in_4 + flow_nic_out_8_nic_in_4
   + flow_nic_out_9_nic_in_4 + flow_nic_out_10_nic_in_4
   + flow_nic_out_11_nic_in_4 + flow_nic_out_12_nic_in_4
   + flow_nic_out_13_nic_in_4 + flow_nic_out_14_nic_in_4
   + flow_nic_out_15_nic_in_4 + flow_nic_out_16_nic_in_4
   + flow_nic_out_17_nic_in_4 + flow_nic_out_18_nic_in_4
   + flow_nic_out_19_nic_in_4 + flow_nic_out_20_nic_in_4
   + flow_nic_out_21_nic_in_4 + flow_nic_out_22_nic_in_4
   + flow_nic_out_23_nic_in_4 + flow_nic_out_24_nic_in_4
   + flow_nic_out_25_nic_in_4 + flow_nic_out_26_nic_in_4
   + flow_nic_out_27_nic_in_4 + flow_nic_out_28_nic_in_4
   + flow_nic_out_29_nic_in_4 + flow_nic_out_30_nic_in_4
   + flow_nic_out_31_nic_in_4 = 0
 node_flow_constraint_nic_out_4: - flow_nic_out_4_sink
   - flow_nic_out_4_nic_in_0 - flow_nic_out_4_nic_in_1
   - flow_nic_out_4_nic_in_2 - flow_nic_out_4_nic_in_3
   - flow_nic_out_4_nic_in_5 - flow_nic_out_4_nic_in_6
   - flow_nic_out_4_nic_in_7 - flow_nic_out_4_nic_in_8
   - flow_nic_out_4_nic_in_9 - flow_nic_out_4_nic_in_10
   - flow_nic_out_4_nic_in_11 - flow_nic_out_4_nic_in_12
   - flow_nic_out_4_nic_in_13 - flow_nic_out_4_nic_in_14
   - flow_nic_out_4_nic_in_15 - flow_nic_out_4_nic_in_16
   - flow_nic_out_4_nic_in_17 - flow_nic_out_4_nic_in_18
   - flow_nic_out_4_nic_in_19 - flow_nic_out_4_nic_in_20
   - flow_nic_out_4_nic_in_21 - flow_nic_out_4_nic_in_22
   - flow_nic_out_4_nic_in_23 - flow_nic_out_4_nic_in_24
   - flow_nic_out_4_nic_in_25 - flow_nic_out_4_nic_in_26
   - flow_nic_out_4_nic_in_27 - flow_nic_out_4_nic_in_28
   - flow_nic_out_4_nic_in_29 - flow_nic_out_4_nic_in_30
   - flow_nic_out_4_nic_in_31 + flow_4_nic_out_4 = 0
 node_flow_constraint_nic_in_5: flow_nic_out_0_nic_in_5
   + flow_nic_out_1_nic_in_5 + flow_nic_out_2_nic_in_5
   + flow_nic_out_3_nic_in_5 + flow_nic_out_4_nic_in_5
   + flow_source_nic_in_5 - flow_nic_in_5_5 + flow_nic_out_6_nic_in_5
   + flow_nic_out_7_nic_in_5 + flow_nic_out_8_nic_in_5
   + flow_nic_out_9_nic_in_5 + flow_nic_out_10_nic_in_5
   + flow_nic_out_11_nic_in_5 + flow_nic_out_12_nic_in_5
   + flow_nic_out_13_nic_in_5 + flow_nic_out_14_nic_in_5
   + flow_nic_out_15_nic_in_5 + flow_nic_out_16_nic_in_5
   + flow_nic_out_17_nic_in_5 + flow_nic_out_18_nic_in_5
   + flow_nic_out_19_nic_in_5 + flow_nic_out_20_nic_in_5
   + flow_nic_out_21_nic_in_5 + flow_nic_out_22_nic_in_5
   + flow_nic_out_23_nic_in_5 + flow_nic_out_24_nic_in_5
   + flow_nic_out_25_nic_in_5 + flow_nic_out_26_nic_in_5
   + flow_nic_out_27_nic_in_5 + flow_nic_out_28_nic_in_5
   + flow_nic_out_29_nic_in_5 + flow_nic_out_30_nic_in_5
   + flow_nic_out_31_nic_in_5 = 0
 node_flow_constraint_nic_out_5: - flow_nic_out_5_sink
   - flow_nic_out_5_nic_in_0 - flow_nic_out_5_nic_in_1
   - flow_nic_out_5_nic_in_2 - flow_nic_out_5_nic_in_3
   - flow_nic_out_5_nic_in_4 - flow_nic_out_5_nic_in_6
   - flow_nic_out_5_nic_in_7 - flow_nic_out_5_nic_in_8
   - flow_nic_out_5_nic_in_9 - flow_nic_out_5_nic_in_10
   - flow_nic_out_5_nic_in_11 - flow_nic_out_5_nic_in_12
   - flow_nic_out_5_nic_in_13 - flow_nic_out_5_nic_in_14
   - flow_nic_out_5_nic_in_15 - flow_nic_out_5_nic_in_16
   - flow_nic_out_5_nic_in_17 - flow_nic_out_5_nic_in_18
   - flow_nic_out_5_nic_in_19 - flow_nic_out_5_nic_in_20
   - flow_nic_out_5_nic_in_21 - flow_nic_out_5_nic_in_22
   - flow_nic_out_5_nic_in_23 - flow_nic_out_5_nic_in_24
   - flow_nic_out_5_nic_in_25 - flow_nic_out_5_nic_in_26
   - flow_nic_out_5_nic_in_27 - flow_nic_out_5_nic_in_28
   - flow_nic_out_5_nic_in_29 - flow_nic_out_5_nic_in_30
   - flow_nic_out_5_nic_in_31 + flow_5_nic_out_5 = 0
 node_flow_constraint_nic_in_6: flow_nic_out_0_nic_in_6
   + flow_nic_out_1_nic_in_6 + flow_nic_out_2_nic_in_6
   + flow_nic_out_3_nic_in_6 + flow_nic_out_4_nic_in_6
   + flow_nic_out_5_nic_in_6 + flow_source_nic_in_6 - flow_nic_in_6_6
   + flow_nic_out_7_nic_in_6 + flow_nic_out_8_nic_in_6
   + flow_nic_out_9_nic_in_6 + flow_nic_out_10_nic_in_6
   + flow_nic_out_11_nic_in_6 + flow_nic_out_12_nic_in_6
   + flow_nic_out_13_nic_in_6 + flow_nic_out_14_nic_in_6
   + flow_nic_out_15_nic_in_6 + flow_nic_out_16_nic_in_6
   + flow_nic_out_17_nic_in_6 + flow_nic_out_18_nic_in_6
   + flow_nic_out_19_nic_in_6 + flow_nic_out_20_nic_in_6
   + flow_nic_out_21_nic_in_6 + flow_nic_out_22_nic_in_6
   + flow_nic_out_23_nic_in_6 + flow_nic_out_24_nic_in_6
   + flow_nic_out_25_nic_in_6 + flow_nic_out_26_nic_in_6
   + flow_nic_out_27_nic_in_6 + flow_nic_out_28_nic_in_6
   + flow_nic_out_29_nic_in_6 + flow_nic_out_30_nic_in_6
   + flow_nic_out_31_nic_in_6 = 0
 node_flow_constraint_nic_out_6: - flow_nic_out_6_sink
   - flow_nic_out_6_nic_in_0 - flow_nic_out_6_nic_in_1
   - flow_nic_out_6_nic_in_2 - flow_nic_out_6_nic_in_3
   - flow_nic_out_6_nic_in_4 - flow_nic_out_6_nic_in_5
   - flow_nic_out_6_nic_in_7 - flow_nic_out_6_nic_in_8
   - flow_nic_out_6_nic_in_9 - flow_nic_out_6_nic_in_10
   - flow_nic_out_6_nic_in_11 - flow_nic_out_6_nic_in_12
   - flow_nic_out_6_nic_in_13 - flow_nic_out_6_nic_in_14
   - flow_nic_out_6_nic_in_15 - flow_nic_out_6_nic_in_16
   - flow_nic_out_6_nic_in_17 - flow_nic_out_6_nic_in_18
   - flow_nic_out_6_nic_in_19 - flow_nic_out_6_nic_in_20
   - flow_nic_out_6_nic_in_21 - flow_nic_out_6_nic_in_22
   - flow_nic_out_6_nic_in_23 - flow_nic_out_6_nic_in_24
   - flow_nic_out_6_nic_in_25 - flow_nic_out_6_nic_in_26
   - flow_nic_out_6_nic_in_27 - flow_nic_out_6_nic_in_28
   - flow_nic_out_6_nic_in_29 - flow_nic_out_6_nic_in_30
   - flow_nic_out_6_nic_in_31 + flow_6_nic_out_6 = 0
 node_flow_constraint_nic_in_7: flow_nic_out_0_nic_in_7
   + flow_nic_out_1_nic_in_7 + flow_nic_out_2_nic_in_7
   + flow_nic_out_3_nic_in_7 + flow_nic_out_4_nic_in_7
   + flow_nic_out_5_nic_in_7 + flow_nic_out_6_nic_in_7
   + flow_source_nic_in_7 - flow_nic_in_7_7 + flow_nic_out_8_nic_in_7
   + flow_nic_out_9_nic_in_7 + flow_nic_out_10_nic_in_7
   + flow_nic_out_11_nic_in_7 + flow_nic_out_12_nic_in_7
   + flow_nic_out_13_nic_in_7 + flow_nic_out_14_nic_in_7
   + flow_nic_out_15_nic_in_7 + flow_nic_out_16_nic_in_7
   + flow_nic_out_17_nic_in_7 + flow_nic_out_18_nic_in_7
   + flow_nic_out_19_nic_in_7 + flow_nic_out_20_nic_in_7
   + flow_nic_out_21_nic_in_7 + flow_nic_out_22_nic_in_7
   + flow_nic_out_23_nic_in_7 + flow_nic_out_24_nic_in_7
   + flow_nic_out_25_nic_in_7 + flow_nic_out_26_nic_in_7
   + flow_nic_out_27_nic_in_7 + flow_nic_out_28_nic_in_7
   + flow_nic_out_29_nic_in_7 + flow_nic_out_30_nic_in_7
   + flow_nic_out_31_nic_in_7 = 0
 node_flow_constraint_nic_out_7: - flow_nic_out_7_sink
   - flow_nic_out_7_nic_in_0 - flow_nic_out_7_nic_in_1
   - flow_nic_out_7_nic_in_2 - flow_nic_out_7_nic_in_3
   - flow_nic_out_7_nic_in_4 - flow_nic_out_7_nic_in_5
   - flow_nic_out_7_nic_in_6 - flow_nic_out_7_nic_in_8
   - flow_nic_out_7_nic_in_9 - flow_nic_out_7_nic_in_10
   - flow_nic_out_7_nic_in_11 - flow_nic_out_7_nic_in_12
   - flow_nic_out_7_nic_in_13 - flow_nic_out_7_nic_in_14
   - flow_nic_out_7_nic_in_15 - flow_nic_out_7_nic_in_16
   - flow_nic_out_7_nic_in_17 - flow_nic_out_7_nic_in_18
   - flow_nic_out_7_nic_in_19 - flow_nic_out_7_nic_in_20
   - flow_nic_out_7_nic_in_21 - flow_nic_out_7_nic_in_22
   - flow_nic_out_7_nic_in_23 - flow_nic_out_7_nic_in_24
   - flow_nic_out_7_nic_in_25 - flow_nic_out_7_nic_in_26
   - flow_nic_out_7_nic_in_27 - flow_nic_out_7_nic_in_28
   - flow_nic_out_7_nic_in_29 - flow_nic_out_7_nic_in_30
   - flow_nic_out_7_nic_in_31 + flow_7_nic_out_7 = 0
 node_flow_constraint_nic_in_8: flow_nic_out_0_nic_in_8
   + flow_nic_out_1_nic_in_8 + flow_nic_out_2_nic_in_8
   + flow_nic_out_3_nic_in_8 + flow_nic_out_4_nic_in_8
   + flow_nic_out_5_nic_in_8 + flow_nic_out_6_nic_in_8
   + flow_nic_out_7_nic_in_8 + flow_source_nic_in_8 - flow_nic_in_8_8
   + flow_nic_out_9_nic_in_8 + flow_nic_out_10_nic_in_8
   + flow_nic_out_11_nic_in_8 + flow_nic_out_12_nic_in_8
   + flow_nic_out_13_nic_in_8 + flow_nic_out_14_nic_in_8
   + flow_nic_out_15_nic_in_8 + flow_nic_out_16_nic_in_8
   + flow_nic_out_17_nic_in_8 + flow_nic_out_18_nic_in_8
   + flow_nic_out_19_nic_in_8 + flow_nic_out_20_nic_in_8
   + flow_nic_out_21_nic_in_8 + flow_nic_out_22_nic_in_8
   + flow_nic_out_23_nic_in_8 + flow_nic_out_24_nic_in_8
   + flow_nic_out_25_nic_in_8 + flow_nic_out_26_nic_in_8
   + flow_nic_out_27_nic_in_8 + flow_nic_out_28_nic_in_8
   + flow_nic_out_29_nic_in_8 + flow_nic_out_30_nic_in_8
   + flow_nic_out_31_nic_in_8 = 0
 node_flow_constraint_nic_out_8: - flow_nic_out_8_sink
   - flow_nic_out_8_nic_in_0 - flow_nic_out_8_nic_in_1
   - flow_nic_out_8_nic_in_2 - flow_nic_out_8_nic_in_3
   - flow_nic_out_8_nic_in_4 - flow_nic_out_8_nic_in_5
   - flow_nic_out_8_nic_in_6 - flow_nic_out_8_nic_in_7
   - flow_nic_out_8_nic_in_9 - flow_nic_out_8_nic_in_10
   - flow_nic_out_8_nic_in_11 - flow_nic_out_8_nic_in_12
   - flow_nic_out_8_nic_in_13 - flow_nic_out_8_nic_in_14
   - flow_nic_out_8_nic_in_15 - flow_nic_out_8_nic_in_16
   - flow_nic_out_8_nic_in_17 - flow_nic_out_8_nic_in_18
   - flow_nic_out_8_nic_in_19 - flow_nic_out_8_nic_in_20
   - flow_nic_out_8_nic_in_21 - flow_nic_out_8_nic_in_22
   - flow_nic_out_8_nic_in_23 - flow_nic_out_8_nic_in_24
   - flow_nic_out_8_nic_in_25 - flow_nic_out_8_nic_in_26
   - flow_nic_out_8_nic_in_27 - flow_nic_out_8_nic_in_28
   - flow_nic_out_8_nic_in_29 - flow_nic_out_8_nic_in_30
   - flow_nic_out_8_nic_in_31 + flow_8_nic_out_8 = 0
 node_flow_constraint_nic_in_9: flow_nic_out_0_nic_in_9
   + flow_nic_out_1_nic_in_9 + flow_nic_out_2_nic_in_9
   + flow_nic_out_3_nic_in_9 + flow_nic_out_4_nic_in_9
   + flow_nic_out_5_nic_in_9 + flow_nic_out_6_nic_in_9
   + flow_nic_out_7_nic_in_9 + flow_nic_out_8_nic_in_9
   + flow_source_nic_in_9 - flow_nic_in_9_9 + flow_nic_out_10_nic_in_9
   + flow_nic_out_11_nic_in_9 + flow_nic_out_12_nic_in_9
   + flow_nic_out_13_nic_in_9 + flow_nic_out_14_nic_in_9
   + flow_nic_out_15_nic_in_9 + flow_nic_out_16_nic_in_9
   + flow_nic_out_17_nic_in_9 + flow_nic_out_18_nic_in_9
   + flow_nic_out_19_nic_in_9 + flow_nic_out_20_nic_in_9
   + flow_nic_out_21_nic_in_9 + flow_nic_out_22_nic_in_9
   + flow_nic_out_23_nic_in_9 + flow_nic_out_24_nic_in_9
   + flow_nic_out_25_nic_in_9 + flow_nic_out_26_nic_in_9
   + flow_nic_out_27_nic_in_9 + flow_nic_out_28_nic_in_9
   + flow_nic_out_29_nic_in_9 + flow_nic_out_30_nic_in_9
   + flow_nic_out_31_nic_in_9 = 0
 node_flow_constraint_nic_out_9: - flow_nic_out_9_sink
   - flow_nic_out_9_nic_in_0 - flow_nic_out_9_nic_in_1
   - flow_nic_out_9_nic_in_2 - flow_nic_out_9_nic_in_3
   - flow_nic_out_9_nic_in_4 - flow_nic_out_9_nic_in_5
   - flow_nic_out_9_nic_in_6 - flow_nic_out_9_nic_in_7
   - flow_nic_out_9_nic_in_8 - flow_nic_out_9_nic_in_10
   - flow_nic_out_9_nic_in_11 - flow_nic_out_9_nic_in_12
   - flow_nic_out_9_nic_in_13 - flow_nic_out_9_nic_in_14
   - flow_nic_out_9_nic_in_15 - flow_nic_out_9_nic_in_16
   - flow_nic_out_9_nic_in_17 - flow_nic_out_9_nic_in_18
   - flow_nic_out_9_nic_in_19 - flow_nic_out_9_nic_in_20
   - flow_nic_out_9_nic_in_21 - flow_nic_out_9_nic_in_22
   - flow_nic_out_9_nic_in_23 - flow_nic_out_9_nic_in_24
   - flow_nic_out_9_nic_in_25 - flow_nic_out_9_nic_in_26
   - flow_nic_out_9_nic_in_27 - flow_nic_out_9_nic_in_28
   - flow_nic_out_9_nic_in_29 - flow_nic_out_9_nic_in_30
   - flow_nic_out_9_nic_in_31 + flow_9_nic_out_9 = 0
 node_flow_constraint_nic_in_10: flow_nic_out_0_nic_in_10
   + flow_nic_out_1_nic_in_10 + flow_nic_out_2_nic_in_10
   + flow_nic_out_3_nic_in_10 + flow_nic_out_4_nic_in_10
   + flow_nic_out_5_nic_in_10 + flow_nic_out_6_nic_in_10
   + flow_nic_out_7_nic_in_10 + flow_nic_out_8_nic_in_10
   + flow_nic_out_9_nic_in_10 + flow_source_nic_in_10 - flow_nic_in_10_10
   + flow_nic_out_11_nic_in_10 + flow_nic_out_12_nic_in_10
   + flow_nic_out_13_nic_in_10 + flow_nic_out_14_nic_in_10
   + flow_nic_out_15_nic_in_10 + flow_nic_out_16_nic_in_10
   + flow_nic_out_17_nic_in_10 + flow_nic_out_18_nic_in_10
   + flow_nic_out_19_nic_in_10 + flow_nic_out_20_nic_in_10
   + flow_nic_out_21_nic_in_10 + flow_nic_out_22_nic_in_10
   + flow_nic_out_23_nic_in_10 + flow_nic_out_24_nic_in_10
   + flow_nic_out_25_nic_in_10 + flow_nic_out_26_nic_in_10
   + flow_nic_out_27_nic_in_10 + flow_nic_out_28_nic_in_10
   + flow_nic_out_29_nic_in_10 + flow_nic_out_30_nic_in_10
   + flow_nic_out_31_nic_in_10 = 0
 node_flow_constraint_nic_out_10: - flow_nic_out_10_sink
   - flow_nic_out_10_nic_in_0 - flow_nic_out_10_nic_in_1
   - flow_nic_out_10_nic_in_2 - flow_nic_out_10_nic_in_3
   - flow_nic_out_10_nic_in_4 - flow_nic_out_10_nic_in_5
   - flow_nic_out_10_nic_in_6 - flow_nic_out_10_nic_in_7
   - flow_nic_out_10_nic_in_8 - flow_nic_out_10_nic_in_9
   - flow_nic_out_10_nic_in_11 - flow_nic_out_10_nic_in_12
   - flow_nic_out_10_nic_in_13 - flow_nic_out_10_nic_in_14
   - flow_nic_out_10_nic_in_15 - flow_nic_out_10_nic_in_16
   - flow_nic_out_10_nic_in_17 - flow_nic_out_10_nic_in_18
   - flow_nic_out_10_nic_in_19 - flow_nic_out_10_nic_in_20
   - flow_nic_out_10_nic_in_21 - flow_nic_out_10_nic_in_22
   - flow_nic_out_10_nic_in_23 - flow_nic_out_10_nic_in_24
   - flow_nic_out_10_nic_in_25 - flow_nic_out_10_nic_in_26
   - flow_nic_out_10_nic_in_27 - flow_nic_out_10_nic_in_28
   - flow_nic_out_10_nic_in_29 - flow_nic_out_10_nic_in_30
   - flow_nic_out_10_nic_in_31 + flow_10_nic_out_10 = 0
 node_flow_constraint_nic_in_11: flow_nic_out_0_nic_in_11
   + flow_nic_out_1_nic_in_11 + flow_nic_out_2_nic_in_11
   + flow_nic_out_3_nic_in_11 + flow_nic_out_4_nic_in_11
   + flow_nic_out_5_nic_in_11 + flow_nic_out_6_nic_in_11
   + flow_nic_out_7_nic_in_11 + flow_nic_out_8_nic_in_11
   + flow_nic_out_9_nic_in_11 + flow_nic_out_10_nic_in_11
   + flow_source_nic_in_11 - flow_nic_in_11_11 + flow_nic_out_12_nic_in_11
   + flow_nic_out_13_nic_in_11 + flow_nic_out_14_nic_in_11
   + flow_nic_out_15_nic_in_11 + flow_nic_out_16_nic_in_11
   + flow_nic_out_17_nic_in_11 + flow_nic_out_18_nic_in_11
   + flow_nic_out_19_nic_in_11 + flow_nic_out_20_nic_in_11
   + flow_nic_out_21_nic_in_11 + flow_nic_out_22_nic_in_11
   + flow_nic_out_23_nic_in_11 + flow_nic_out_24_nic_in_11
   + flow_nic_out_25_nic_in_11 + flow_nic_out_26_nic_in_11
   + flow_nic_out_27_nic_in_11 + flow_nic_out_28_nic_in_11
   + flow_nic_out_29_nic_in_11 + flow_nic_out_30_nic_in_11
   + flow_nic_out_31_nic_in_11 = 0
 node_flow_constraint_nic_out_11: - flow_nic_out_11_sink
   - flow_nic_out_11_nic_in_0 - flow_nic_out_11_nic_in_1
   - flow_nic_out_11_nic_in_2 - flow_nic_out_11_nic_in_3
   - flow_nic_out_11_nic_in_4 - flow_nic_out_11_nic_in_5
   - flow_nic_out_11_nic_in_6 - flow_nic_out_11_nic_in_7
   - flow_nic_out_11_nic_in_8 - flow_nic_out_11_nic_in_9
   - flow_nic_out_11_nic_in_10 - flow_nic_out_11_nic_in_12
   - flow_nic_out_11_nic_in_13 - flow_nic_out_11_nic_in_14
   - flow_nic_out_11_nic_in_15 - flow_nic_out_11_nic_in_16
   - flow_nic_out_11_nic_in_17 - flow_nic_out_11_nic_in_18
   - flow_nic_out_11_nic_in_19 - flow_nic_out_11_nic_in_20
   - flow_nic_out_11_nic_in_21 - flow_nic_out_11_nic_in_22
   - flow_nic_out_11_nic_in_23 - flow_nic_out_11_nic_in_24
   - flow_nic_out_11_nic_in_25 - flow_nic_out_11_nic_in_26
   - flow_nic_out_11_nic_in_27 - flow_nic_out_11_nic_in_28
   - flow_nic_out_11_nic_in_29 - flow_nic_out_11_nic_in_30
   - flow_nic_out_11_nic_in_31 + flow_11_nic_out_11 = 0
 node_flow_constraint_nic_in_12: flow_nic_out_0_nic_in_12
   + flow_nic_out_1_nic_in_12 + flow_nic_out_2_nic_in_12
   + flow_nic_out_3_nic_in_12 + flow_nic_out_4_nic_in_12
   + flow_nic_out_5_nic_in_12 + flow_nic_out_6_nic_in_12
   + flow_nic_out_7_nic_in_12 + flow_nic_out_8_nic_in_12
   + flow_nic_out_9_nic_in_12 + flow_nic_out_10_nic_in_12
   + flow_nic_out_11_nic_in_12 + flow_source_nic_in_12 - flow_nic_in_12_12
   + flow_nic_out_13_nic_in_12 + flow_nic_out_14_nic_in_12
   + flow_nic_out_15_nic_in_12 + flow_nic_out_16_nic_in_12
   + flow_nic_out_17_nic_in_12 + flow_nic_out_18_nic_in_12
   + flow_nic_out_19_nic_in_12 + flow_nic_out_20_nic_in_12
   + flow_nic_out_21_nic_in_12 + flow_nic_out_22_nic_in_12
   + flow_nic_out_23_nic_in_12 + flow_nic_out_24_nic_in_12
   + flow_nic_out_25_nic_in_12 + flow_nic_out_26_nic_in_12
   + flow_nic_out_27_nic_in_12 + flow_nic_out_28_nic_in_12
   + flow_nic_out_29_nic_in_12 + flow_nic_out_30_nic_in_12
   + flow_nic_out_31_nic_in_12 = 0
 node_flow_constraint_nic_out_12: - flow_nic_out_12_sink
   - flow_nic_out_12_nic_in_0 - flow_nic_out_12_nic_in_1
   - flow_nic_out_12_nic_in_2 - flow_nic_out_12_nic_in_3
   - flow_nic_out_12_nic_in_4 - flow_nic_out_12_nic_in_5
   - flow_nic_out_12_nic_in_6 - flow_nic_out_12_nic_in_7
   - flow_nic_out_12_nic_in_8 - flow_nic_out_12_nic_in_9
   - flow_nic_out_12_nic_in_10 - flow_nic_out_12_nic_in_11
   - flow_nic_out_12_nic_in_13 - flow_nic_out_12_nic_in_14
   - flow_nic_out_12_nic_in_15 - flow_nic_out_12_nic_in_16
   - flow_nic_out_12_nic_in_17 - flow_nic_out_12_nic_in_18
   - flow_nic_out_12_nic_in_19 - flow_nic_out_12_nic_in_20
   - flow_nic_out_12_nic_in_21 - flow_nic_out_12_nic_in_22
   - flow_nic_out_12_nic_in_23 - flow_nic_out_12_nic_in_24
   - flow_nic_out_12_nic_in_25 - flow_nic_out_12_nic_in_26
   - flow_nic_out_12_nic_in_27 - flow_nic_out_12_nic_in_28
   - flow_nic_out_12_nic_in_29 - flow_nic_out_12_nic_in_30
   - flow_nic_out_12_nic_in_31 + flow_12_nic_out_12 = 0
 node_flow_constraint_nic_in_13: flow_nic_out_0_nic_in_13
   + flow_nic_out_1_nic_in_13 + flow_nic_out_2_nic_in_13
   + flow_nic_out_3_nic_in_13 + flow_nic_out_4_nic_in_13
   + flow_nic_out_5_nic_in_13 + flow_nic_out_6_nic_in_13
   + flow_nic_out_7_nic_in_13 + flow_nic_out_8_nic_in_13
   + flow_nic_out_9_nic_in_13 + flow_nic_out_10_nic_in_13
   + flow_nic_out_11_nic_in_13 + flow_nic_out_12_nic_in_13
   + flow_source_nic_in_13 - flow_nic_in_13_13 + flow_nic_out_14_nic_in_13
   + flow_nic_out_15_nic_in_13 + flow_nic_out_16_nic_in_13
   + flow_nic_out_17_nic_in_13 + flow_nic_out_18_nic_in_13
   + flow_nic_out_19_nic_in_13 + flow_nic_out_20_nic_in_13
   + flow_nic_out_21_nic_in_13 + flow_nic_out_22_nic_in_13
   + flow_nic_out_23_nic_in_13 + flow_nic_out_24_nic_in_13
   + flow_nic_out_25_nic_in_13 + flow_nic_out_26_nic_in_13
   + flow_nic_out_27_nic_in_13 + flow_nic_out_28_nic_in_13
   + flow_nic_out_29_nic_in_13 + flow_nic_out_30_nic_in_13
   + flow_nic_out_31_nic_in_13 = 0
 node_flow_constraint_nic_out_13: - flow_nic_out_13_sink
   - flow_nic_out_13_nic_in_0 - flow_nic_out_13_nic_in_1
   - flow_nic_out_13_nic_in_2 - flow_nic_out_13_nic_in_3
   - flow_nic_out_13_nic_in_4 - flow_nic_out_13_nic_in_5
   - flow_nic_out_13_nic_in_6 - flow_nic_out_13_nic_in_7
   - flow_nic_out_13_nic_in_8 - flow_nic_out_13_nic_in_9
   - flow_nic_out_13_nic_in_10 - flow_nic_out_13_nic_in_11
   - flow_nic_out_13_nic_in_12 - flow_nic_out_13_nic_in_14
   - flow_nic_out_13_nic_in_15 - flow_nic_out_13_nic_in_16
   - flow_nic_out_13_nic_in_17 - flow_nic_out_13_nic_in_18
   - flow_nic_out_13_nic_in_19 - flow_nic_out_13_nic_in_20
   - flow_nic_out_13_nic_in_21 - flow_nic_out_13_nic_in_22
   - flow_nic_out_13_nic_in_23 - flow_nic_out_13_nic_in_24
   - flow_nic_out_13_nic_in_25 - flow_nic_out_13_nic_in_26
   - flow_nic_out_13_nic_in_27 - flow_nic_out_13_nic_in_28
   - flow_nic_out_13_nic_in_29 - flow_nic_out_13_nic_in_30
   - flow_nic_out_13_nic_in_31 + flow_13_nic_out_13 = 0
 node_flow_constraint_nic_in_14: flow_nic_out_0_nic_in_14
   + flow_nic_out_1_nic_in_14 + flow_nic_out_2_nic_in_14
   + flow_nic_out_3_nic_in_14 + flow_nic_out_4_nic_in_14
   + flow_nic_out_5_nic_in_14 + flow_nic_out_6_nic_in_14
   + flow_nic_out_7_nic_in_14 + flow_nic_out_8_nic_in_14
   + flow_nic_out_9_nic_in_14 + flow_nic_out_10_nic_in_14
   + flow_nic_out_11_nic_in_14 + flow_nic_out_12_nic_in_14
   + flow_nic_out_13_nic_in_14 + flow_source_nic_in_14 - flow_nic_in_14_14
   + flow_nic_out_15_nic_in_14 + flow_nic_out_16_nic_in_14
   + flow_nic_out_17_nic_in_14 + flow_nic_out_18_nic_in_14
   + flow_nic_out_19_nic_in_14 + flow_nic_out_20_nic_in_14
   + flow_nic_out_21_nic_in_14 + flow_nic_out_22_nic_in_14
   + flow_nic_out_23_nic_in_14 + flow_nic_out_24_nic_in_14
   + flow_nic_out_25_nic_in_14 + flow_nic_out_26_nic_in_14
   + flow_nic_out_27_nic_in_14 + flow_nic_out_28_nic_in_14
   + flow_nic_out_29_nic_in_14 + flow_nic_out_30_nic_in_14
   + flow_nic_out_31_nic_in_14 = 0
 node_flow_constraint_nic_out_14: - flow_nic_out_14_sink
   - flow_nic_out_14_nic_in_0 - flow_nic_out_14_nic_in_1
   - flow_nic_out_14_nic_in_2 - flow_nic_out_14_nic_in_3
   - flow_nic_out_14_nic_in_4 - flow_nic_out_14_nic_in_5
   - flow_nic_out_14_nic_in_6 - flow_nic_out_14_nic_in_7
   - flow_nic_out_14_nic_in_8 - flow_nic_out_14_nic_in_9
   - flow_nic_out_14_nic_in_10 - flow_nic_out_14_nic_in_11
   - flow_nic_out_14_nic_in_12 - flow_nic_out_14_nic_in_13
   - flow_nic_out_14_nic_in_15 - flow_nic_out_14_nic_in_16
   - flow_nic_out_14_nic_in_17 - flow_nic_out_14_nic_in_18
   - flow_nic_out_14_nic_in_19 - flow_nic_out_14_nic_in_20
   - flow_nic_out_14_nic_in_21 - flow_nic_out_14_nic_in_22
   - flow_nic_out_14_nic_in_23 - flow_nic_out_14_nic_in_24
   - flow_nic_out_14_nic_in_25 - flow_nic_out_14_nic_in_26
   - flow_nic_out_14_nic_in_27 - flow_nic_out_14_nic_in_28
   - flow_nic_out_14_nic_in_29 - flow_nic_out_14_nic_in_30
   - flow_nic_out_14_nic_in_31 + flow_14_nic_out_14 = 0
 node_flow_constraint_nic_in_15: flow_nic_out_0_nic_in_15
   + flow_nic_out_1_nic_in_15 + flow_nic_out_2_nic_in_15
   + flow_nic_out_3_nic_in_15 + flow_nic_out_4_nic_in_15
   + flow_nic_out_5_nic_in_15 + flow_nic_out_6_nic_in_15
   + flow_nic_out_7_nic_in_15 + flow_nic_out_8_nic_in_15
   + flow_nic_out_9_nic_in_15 + flow_nic_out_10_nic_in_15
   + flow_nic_out_11_nic_in_15 + flow_nic_out_12_nic_in_15
   + flow_nic_out_13_nic_in_15 + flow_nic_out_14_nic_in_15
   + flow_source_nic_in_15 - flow_nic_in_15_15 + flow_nic_out_16_nic_in_15
   + flow_nic_out_17_nic_in_15 + flow_nic_out_18_nic_in_15
   + flow_nic_out_19_nic_in_15 + flow_nic_out_20_nic_in_15
   + flow_nic_out_21_nic_in_15 + flow_nic_out_22_nic_in_15
   + flow_nic_out_23_nic_in_15 + flow_nic_out_24_nic_in_15
   + flow_nic_out_25_nic_in_15 + flow_nic_out_26_nic_in_15
   + flow_nic_out_27_nic_in_15 + flow_nic_out_28_nic_in_15
   + flow_nic_out_29_nic_in_15 + flow_nic_out_30_nic_in_15
   + flow_nic_out_31_nic_in_15 = 0
 node_flow_constraint_nic_out_15: - flow_nic_out_15_sink
   - flow_nic_out_15_nic_in_0 - flow_nic_out_15_nic_in_1
   - flow_nic_out_15_nic_in_2 - flow_nic_out_15_nic_in_3
   - flow_nic_out_15_nic_in_4 - flow_nic_out_15_nic_in_5
   - flow_nic_out_15_nic_in_6 - flow_nic_out_15_nic_in_7
   - flow_nic_out_15_nic_in_8 - flow_nic_out_15_nic_in_9
   - flow_nic_out_15_nic_in_10 - flow_nic_out_15_nic_in_11
   - flow_nic_out_15_nic_in_12 - flow_nic_out_15_nic_in_13
   - flow_nic_out_15_nic_in_14 - flow_nic_out_15_nic_in_16
   - flow_nic_out_15_nic_in_17 - flow_nic_out_15_nic_in_18
   - flow_nic_out_15_nic_in_19 - flow_nic_out_15_nic_in_20
   - flow_nic_out_15_nic_in_21 - flow_nic_out_15_nic_in_22
   - flow_nic_out_15_nic_in_23 - flow_nic_out_15_nic_in_24
   - flow_nic_out_15_nic_in_25 - flow_nic_out_15_nic_in_26
   - flow_nic_out_15_nic_in_27 - flow_nic_out_15_nic_in_28
   - flow_nic_out_15_nic_in_29 - flow_nic_out_15_nic_in_30
   - flow_nic_out_15_nic_in_31 + flow_15_nic_out_15 = 0
 node_flow_constraint_nic_in_16: flow_nic_out_0_nic_in_16
   + flow_nic_out_1_nic_in_16 + flow_nic_out_2_nic_in_16
   + flow_nic_out_3_nic_in_16 + flow_nic_out_4_nic_in_16
   + flow_nic_out_5_nic_in_16 + flow_nic_out_6_nic_in_16
   + flow_nic_out_7_nic_in_16 + flow_nic_out_8_nic_in_16
   + flow_nic_out_9_nic_in_16 + flow_nic_out_10_nic_in_16
   + flow_nic_out_11_nic_in_16 + flow_nic_out_12_nic_in_16
   + flow_nic_out_13_nic_in_16 + flow_nic_out_14_nic_in_16
   + flow_nic_out_15_nic_in_16 + flow_source_nic_in_16 - flow_nic_in_16_16
   + flow_nic_out_17_nic_in_16 + flow_nic_out_18_nic_in_16
   + flow_nic_out_19_nic_in_16 + flow_nic_out_20_nic_in_16
   + flow_nic_out_21_nic_in_16 + flow_nic_out_22_nic_in_16
   + flow_nic_out_23_nic_in_16 + flow_nic_out_24_nic_in_16
   + flow_nic_out_25_nic_in_16 + flow_nic_out_26_nic_in_16
   + flow_nic_out_27_nic_in_16 + flow_nic_out_28_nic_in_16
   + flow_nic_out_29_nic_in_16 + flow_nic_out_30_nic_in_16
   + flow_nic_out_31_nic_in_16 = 0
 node_flow_constraint_nic_out_16: - flow_nic_out_16_sink
   - flow_nic_out_16_nic_in_0 - flow_nic_out_16_nic_in_1
   - flow_nic_out_16_nic_in_2 - flow_nic_out_16_nic_in_3
   - flow_nic_out_16_nic_in_4 - flow_nic_out_16_nic_in_5
   - flow_nic_out_16_nic_in_6 - flow_nic_out_16_nic_in_7
   - flow_nic_out_16_nic_in_8 - flow_nic_out_16_nic_in_9
   - flow_nic_out_16_nic_in_10 - flow_nic_out_16_nic_in_11
   - flow_nic_out_16_nic_in_12 - flow_nic_out_16_nic_in_13
   - flow_nic_out_16_nic_in_14 - flow_nic_out_16_nic_in_15
   - flow_nic_out_16_nic_in_17 - flow_nic_out_16_nic_in_18
   - flow_nic_out_16_nic_in_19 - flow_nic_out_16_nic_in_20
   - flow_nic_out_16_nic_in_21 - flow_nic_out_16_nic_in_22
   - flow_nic_out_16_nic_in_23 - flow_nic_out_16_nic_in_24
   - flow_nic_out_16_nic_in_25 - flow_nic_out_16_nic_in_26
   - flow_nic_out_16_nic_in_27 - flow_nic_out_16_nic_in_28
   - flow_nic_out_16_nic_in_29 - flow_nic_out_16_nic_in_30
   - flow_nic_out_16_nic_in_31 + flow_16_nic_out_16 = 0
 node_flow_constraint_nic_in_17: flow_nic_out_0_nic_in_17
   + flow_nic_out_1_nic_in_17 + flow_nic_out_2_nic_in_17
   + flow_nic_out_3_nic_in_17 + flow_nic_out_4_nic_in_17
   + flow_nic_out_5_nic_in_17 + flow_nic_out_6_nic_in_17
   + flow_nic_out_7_nic_in_17 + flow_nic_out_8_nic_in_17
   + flow_nic_out_9_nic_in_17 + flow_nic_out_10_nic_in_17
   + flow_nic_out_11_nic_in_17 + flow_nic_out_12_nic_in_17
   + flow_nic_out_13_nic_in_17 + flow_nic_out_14_nic_in_17
   + flow_nic_out_15_nic_in_17 + flow_nic_out_16_nic_in_17
   + flow_source_nic_in_17 - flow_nic_in_17_17 + flow_nic_out_18_nic_in_17
   + flow_nic_out_19_nic_in_17 + flow_nic_out_20_nic_in_17
   + flow_nic_out_21_nic_in_17 + flow_nic_out_22_nic_in_17
   + flow_nic_out_23_nic_in_17 + flow_nic_out_24_nic_in_17
   + flow_nic_out_25_nic_in_17 + flow_nic_out_26_nic_in_17
   + flow_nic_out_27_nic_in_17 + flow_nic_out_28_nic_in_17
   + flow_nic_out_29_nic_in_17 + flow_nic_out_30_nic_in_17
   + flow_nic_out_31_nic_in_17 = 0
 node_flow_constraint_nic_out_17: - flow_nic_out_17_sink
   - flow_nic_out_17_nic_in_0 - flow_nic_out_17_nic_in_1
   - flow_nic_out_17_nic_in_2 - flow_nic_out_17_nic_in_3
   - flow_nic_out_17_nic_in_4 - flow_nic_out_17_nic_in_5
   - flow_nic_out_17_nic_in_6 - flow_nic_out_17_nic_in_7
   - flow_nic_out_17_nic_in_8 - flow_nic_out_17_nic_in_9
   - flow_nic_out_17_nic_in_10 - flow_nic_out_17_nic_in_11
   - flow_nic_out_17_nic_in_12 - flow_nic_out_17_nic_in_13
   - flow_nic_out_17_nic_in_14 - flow_nic_out_17_nic_in_15
   - flow_nic_out_17_nic_in_16 - flow_nic_out_17_nic_in_18
   - flow_nic_out_17_nic_in_19 - flow_nic_out_17_nic_in_20
   - flow_nic_out_17_nic_in_21 - flow_nic_out_17_nic_in_22
   - flow_nic_out_17_nic_in_23 - flow_nic_out_17_nic_in_24
   - flow_nic_out_17_nic_in_25 - flow_nic_out_17_nic_in_26
   - flow_nic_out_17_nic_in_27 - flow_nic_out_17_nic_in_28
   - flow_nic_out_17_nic_in_29 - flow_nic_out_17_nic_in_30
   - flow_nic_out_17_nic_in_31 + flow_17_nic_out_17 = 0
 node_flow_constraint_nic_in_18: flow_nic_out_0_nic_in_18
   + flow_nic_out_1_nic_in_18 + flow_nic_out_2_nic_in_18
   + flow_nic_out_3_nic_in_18 + flow_nic_out_4_nic_in_18
   + flow_nic_out_5_nic_in_18 + flow_nic_out_6_nic_in_18
   + flow_nic_out_7_nic_in_18 + flow_nic_out_8_nic_in_18
   + flow_nic_out_9_nic_in_18 + flow_nic_out_10_nic_in_18
   + flow_nic_out_11_nic_in_18 + flow_nic_out_12_nic_in_18
   + flow_nic_out_13_nic_in_18 + flow_nic_out_14_nic_in_18
   + flow_nic_out_15_nic_in_18 + flow_nic_out_16_nic_in_18
   + flow_nic_out_17_nic_in_18 + flow_source_nic_in_18 - flow_nic_in_18_18
   + flow_nic_out_19_nic_in_18 + flow_nic_out_20_nic_in_18
   + flow_nic_out_21_nic_in_18 + flow_nic_out_22_nic_in_18
   + flow_nic_out_23_nic_in_18 + flow_nic_out_24_nic_in_18
   + flow_nic_out_25_nic_in_18 + flow_nic_out_26_nic_in_18
   + flow_nic_out_27_nic_in_18 + flow_nic_out_28_nic_in_18
   + flow_nic_out_29_nic_in_18 + flow_nic_out_30_nic_in_18
   + flow_nic_out_31_nic_in_18 = 0
 node_flow_constraint_nic_out_18: - flow_nic_out_18_sink
   - flow_nic_out_18_nic_in_0 - flow_nic_out_18_nic_in_1
   - flow_nic_out_18_nic_in_2 - flow_nic_out_18_nic_in_3
   - flow_nic_out_18_nic_in_4 - flow_nic_out_18_nic_in_5
   - flow_nic_out_18_nic_in_6 - flow_nic_out_18_nic_in_7
   - flow_nic_out_18_nic_in_8 - flow_nic_out_18_nic_in_9
   - flow_nic_out_18_nic_in_10 - flow_nic_out_18_nic_in_11
   - flow_nic_out_18_nic_in_12 - flow_nic_out_18_nic_in_13
   - flow_nic_out_18_nic_in_14 - flow_nic_out_18_nic_in_15
   - flow_nic_out_18_nic_in_16 - flow_nic_out_18_nic_in_17
   - flow_nic_out_18_nic_in_19 - flow_nic_out_18_nic_in_20
   - flow_nic_out_18_nic_in_21 - flow_nic_out_18_nic_in_22
   - flow_nic_out_18_nic_in_23 - flow_nic_out_18_nic_in_24
   - flow_nic_out_18_nic_in_25 - flow_nic_out_18_nic_in_26
   - flow_nic_out_18_nic_in_27 - flow_nic_out_18_nic_in_28
   - flow_nic_out_18_nic_in_29 - flow_nic_out_18_nic_in_30
   - flow_nic_out_18_nic_in_31 + flow_18_nic_out_18 = 0
 node_flow_constraint_nic_in_19: flow_nic_out_0_nic_in_19
   + flow_nic_out_1_nic_in_19 + flow_nic_out_2_nic_in_19
   + flow_nic_out_3_nic_in_19 + flow_nic_out_4_nic_in_19
   + flow_nic_out_5_nic_in_19 + flow_nic_out_6_nic_in_19
   + flow_nic_out_7_nic_in_19 + flow_nic_out_8_nic_in_19
   + flow_nic_out_9_nic_in_19 + flow_nic_out_10_nic_in_19
   + flow_nic_out_11_nic_in_19 + flow_nic_out_12_nic_in_19
   + flow_nic_out_13_nic_in_19 + flow_nic_out_14_nic_in_19
   + flow_nic_out_15_nic_in_19 + flow_nic_out_16_nic_in_19
   + flow_nic_out_17_nic_in_19 + flow_nic_out_18_nic_in_19
   + flow_source_nic_in_19 - flow_nic_in_19_19 + flow_nic_out_20_nic_in_19
   + flow_nic_out_21_nic_in_19 + flow_nic_out_22_nic_in_19
   + flow_nic_out_23_nic_in_19 + flow_nic_out_24_nic_in_19
   + flow_nic_out_25_nic_in_19 + flow_nic_out_26_nic_in_19
   + flow_nic_out_27_nic_in_19 + flow_nic_out_28_nic_in_19
   + flow_nic_out_29_nic_in_19 + flow_nic_out_30_nic_in_19
   + flow_nic_out_31_nic_in_19 = 0
 node_flow_constraint_nic_out_19: - flow_nic_out_19_sink
   - flow_nic_out_19_nic_in_0 - flow_nic_out_19_nic_in_1
   - flow_nic_out_19_nic_in_2 - flow_nic_out_19_nic_in_3
   - flow_nic_out_19_nic_in_4 - flow_nic_out_19_nic_in_5
   - flow_nic_out_19_nic_in_6 - flow_nic_out_19_nic_in_7
   - flow_nic_out_19_nic_in_8 - flow_nic_out_19_nic_in_9
   - flow_nic_out_19_nic_in_10 - flow_nic_out_19_nic_in_11
   - flow_nic_out_19_nic_in_12 - flow_nic_out_19_nic_in_13
   - flow_nic_out_19_nic_in_14 - flow_nic_out_19_nic_in_15
   - flow_nic_out_19_nic_in_16 - flow_nic_out_19_nic_in_17
   - flow_nic_out_19_nic_in_18 - flow_nic_out_19_nic_in_20
   - flow_nic_out_19_nic_in_21 - flow_nic_out_19_nic_in_22
   - flow_nic_out_19_nic_in_23 - flow_nic_out_19_nic_in_24
   - flow_nic_out_19_nic_in_25 - flow_nic_out_19_nic_in_26
   - flow_nic_out_19_nic_in_27 - flow_nic_out_19_nic_in_28
   - flow_nic_out_19_nic_in_29 - flow_nic_out_19_nic_in_30
   - flow_nic_out_19_nic_in_31 + flow_19_nic_out_19 = 0
 node_flow_constraint_nic_in_20: flow_nic_out_0_nic_in_20
   + flow_nic_out_1_nic_in_20 + flow_nic_out_2_nic_in_20
   + flow_nic_out_3_nic_in_20 + flow_nic_out_4_nic_in_20
   + flow_nic_out_5_nic_in_20 + flow_nic_out_6_nic_in_20
   + flow_nic_out_7_nic_in_20 + flow_nic_out_8_nic_in_20
   + flow_nic_out_9_nic_in_20 + flow_nic_out_10_nic_in_20
   + flow_nic_out_11_nic_in_20 + flow_nic_out_12_nic_in_20
   + flow_nic_out_13_nic_in_20 + flow_nic_out_14_nic_in_20
   + flow_nic_out_15_nic_in_20 + flow_nic_out_16_nic_in_20
   + flow_nic_out_17_nic_in_20 + flow_nic_out_18_nic_in_20
   + flow_nic_out_19_nic_in_20 + flow_source_nic_in_20 - flow_nic_in_20_20
   + flow_nic_out_21_nic_in_20 + flow_nic_out_22_nic_in_20
   + flow_nic_out_23_nic_in_20 + flow_nic_out_24_nic_in_20
   + flow_nic_out_25_nic_in_20 + flow_nic_out_26_nic_in_20
   + flow_nic_out_27_nic_in_20 + flow_nic_out_28_nic_in_20
   + flow_nic_out_29_nic_in_20 + flow_nic_out_30_nic_in_20
   + flow_nic_out_31_nic_in_20 = 0
 node_flow_constraint_nic_out_20: - flow_nic_out_20_sink
   - flow_nic_out_20_nic_in_0 - flow_nic_out_20_nic_in_1
   - flow_nic_out_20_nic_in_2 - flow_nic_out_20_nic_in_3
   - flow_nic_out_20_nic_in_4 - flow_nic_out_20_nic_in_5
   - flow_nic_out_20_nic_in_6 - flow_nic_out_20_nic_in_7
   - flow_nic_out_20_nic_in_8 - flow_nic_out_20_nic_in_9
   - flow_nic_out_20_nic_in_10 - flow_nic_out_20_nic_in_11
   - flow_nic_out_20_nic_in_12 - flow_nic_out_20_nic_in_13
   - flow_nic_out_20_nic_in_14 - flow_nic_out_20_nic_in_15
   - flow_nic_out_20_nic_in_16 - flow_nic_out_20_nic_in_17
   - flow_nic_out_20_nic_in_18 - flow_nic_out_20_nic_in_19
   - flow_nic_out_20_nic_in_21 - flow_nic_out_20_nic_in_22
   - flow_nic_out_20_nic_in_23 - flow_nic_out_20_nic_in_24
   - flow_nic_out_20_nic_in_25 - flow_nic_out_20_nic_in_26
   - flow_nic_out_20_nic_in_27 - flow_nic_out_20_nic_in_28
   - flow_nic_out_20_nic_in_29 - flow_nic_out_20_nic_in_30
   - flow_nic_out_20_nic_in_31 + flow_20_nic_out_20 = 0
 node_flow_constraint_nic_in_21: flow_nic_out_0_nic_in_21
   + flow_nic_out_1_nic_in_21 + flow_nic_out_2_nic_in_21
   + flow_nic_out_3_nic_in_21 + flow_nic_out_4_nic_in_21
   + flow_nic_out_5_nic_in_21 + flow_nic_out_6_nic_in_21
   + flow_nic_out_7_nic_in_21 + flow_nic_out_8_nic_in_21
   + flow_nic_out_9_nic_in_21 + flow_nic_out_10_nic_in_21
   + flow_nic_out_11_nic_in_21 + flow_nic_out_12_nic_in_21
   + flow_nic_out_13_nic_in_21 + flow_nic_out_14_nic_in_21
   + flow_nic_out_15_nic_in_21 + flow_nic_out_16_nic_in_21
   + flow_nic_out_17_nic_in_21 + flow_nic_out_18_nic_in_21
   + flow_nic_out_19_nic_in_21 + flow_nic_out_20_nic_in_21
   + flow_source_nic_in_21 - flow_nic_in_21_21 + flow_nic_out_22_nic_in_21
   + flow_nic_out_23_nic_in_21 + flow_nic_out_24_nic_in_21
   + flow_nic_out_25_nic_in_21 + flow_nic_out_26_nic_in_21
   + flow_nic_out_27_nic_in_21 + flow_nic_out_28_nic_in_21
   + flow_nic_out_29_nic_in_21 + flow_nic_out_30_nic_in_21
   + flow_nic_out_31_nic_in_21 = 0
 node_flow_constraint_nic_out_21: - flow_nic_out_21_sink
   - flow_nic_out_21_nic_in_0 - flow_nic_out_21_nic_in_1
   - flow_nic_out_21_nic_in_2 - flow_nic_out_21_nic_in_3
   - flow_nic_out_21_nic_in_4 - flow_nic_out_21_nic_in_5
   - flow_nic_out_21_nic_in_6 - flow_nic_out_21_nic_in_7
   - flow_nic_out_21_nic_in_8 - flow_nic_out_21_nic_in_9
   - flow_nic_out_21_nic_in_10 - flow_nic_out_21_nic_in_11
   - flow_nic_out_21_nic_in_12 - flow_nic_out_21_nic_in_13
   - flow_nic_out_21_nic_in_14 - flow_nic_out_21_nic_in_15
   - flow_nic_out_21_nic_in_16 - flow_nic_out_21_nic_in_17
   - flow_nic_out_21_nic_in_18 - flow_nic_out_21_nic_in_19
   - flow_nic_out_21_nic_in_20 - flow_nic_out_21_nic_in_22
   - flow_nic_out_21_nic_in_23 - flow_nic_out_21_nic_in_24
   - flow_nic_out_21_nic_in_25 - flow_nic_out_21_nic_in_26
   - flow_nic_out_21_nic_in_27 - flow_nic_out_21_nic_in_28
   - flow_nic_out_21_nic_in_29 - flow_nic_out_21_nic_in_30
   - flow_nic_out_21_nic_in_31 + flow_21_nic_out_21 = 0
 node_flow_constraint_nic_in_22: flow_nic_out_0_nic_in_22
   + flow_nic_out_1_nic_in_22 + flow_nic_out_2_nic_in_22
   + flow_nic_out_3_nic_in_22 + flow_nic_out_4_nic_in_22
   + flow_nic_out_5_nic_in_22 + flow_nic_out_6_nic_in_22
   + flow_nic_out_7_nic_in_22 + flow_nic_out_8_nic_in_22
   + flow_nic_out_9_nic_in_22 + flow_nic_out_10_nic_in_22
   + flow_nic_out_11_nic_in_22 + flow_nic_out_12_nic_in_22
   + flow_nic_out_13_nic_in_22 + flow_nic_out_14_nic_in_22
   + flow_nic_out_15_nic_in_22 + flow_nic_out_16_nic_in_22
   + flow_nic_out_17_nic_in_22 + flow_nic_out_18_nic_in_22
   + flow_nic_out_19_nic_in_22 + flow_nic_out_20_nic_in_22
   + flow_nic_out_21_nic_in_22 + flow_source_nic_in_22 - flow_nic_in_22_22
   + flow_nic_out_23_nic_in_22 + flow_nic_out_24_nic_in_22
   + flow_nic_out_25_nic_in_22 + flow_nic_out_26_nic_in_22
   + flow_nic_out_27_nic_in_22 + flow_nic_out_28_nic_in_22
   + flow_nic_out_29_nic_in_22 + flow_nic_out_30_nic_in_22
   + flow_nic_out_31_nic_in_22 = 0
 node_flow_constraint_nic_out_22: - flow_nic_out_22_sink
   - flow_nic_out_22_nic_in_0 - flow_nic_out_22_nic_in_1
   - flow_nic_out_22_nic_in_2 - flow_nic_out_22_nic_in_3
   - flow_nic_out_22_nic_in_4 - flow_nic_out_22_nic_in_5
   - flow_nic_out_22_nic_in_6 - flow_nic_out_22_nic_in_7
   - flow_nic_out_22_nic_in_8 - flow_nic_out_22_nic_in_9
   - flow_nic_out_22_nic_in_10 - flow_nic_out_22_nic_in_11
   - flow_nic_out_22_nic_in_12 - flow_nic_out_22_nic_in_13
   - flow_nic_out_22_nic_in_14 - flow_nic_out_22_nic_in_15
   - flow_nic_out_22_nic_in_16 - flow_nic_out_22_nic_in_17
   - flow_nic_out_22_nic_in_18 - flow_nic_out_22_nic_in_19
   - flow_nic_out_22_nic_in_20 - flow_nic_out_22_nic_in_21
   - flow_nic_out_22_nic_in_23 - flow_nic_out_22_nic_in_24
   - flow_nic_out_22_nic_in_25 - flow_nic_out_22_nic_in_26
   - flow_nic_out_22_nic_in_27 - flow_nic_out_22_nic_in_28
   - flow_nic_out_22_nic_in_29 - flow_nic_out_22_nic_in_30
   - flow_nic_out_22_nic_in_31 + flow_22_nic_out_22 = 0
 node_flow_constraint_nic_in_23: flow_nic_out_0_nic_in_23
   + flow_nic_out_1_nic_in_23 + flow_nic_out_2_nic_in_23
   + flow_nic_out_3_nic_in_23 + flow_nic_out_4_nic_in_23
   + flow_nic_out_5_nic_in_23 + flow_nic_out_6_nic_in_23
   + flow_nic_out_7_nic_in_23 + flow_nic_out_8_nic_in_23
   + flow_nic_out_9_nic_in_23 + flow_nic_out_10_nic_in_23
   + flow_nic_out_11_nic_in_23 + flow_nic_out_12_nic_in_23
   + flow_nic_out_13_nic_in_23 + flow_nic_out_14_nic_in_23
   + flow_nic_out_15_nic_in_23 + flow_nic_out_16_nic_in_23
   + flow_nic_out_17_nic_in_23 + flow_nic_out_18_nic_in_23
   + flow_nic_out_19_nic_in_23 + flow_nic_out_20_nic_in_23
   + flow_nic_out_21_nic_in_23 + flow_nic_out_22_nic_in_23
   + flow_source_nic_in_23 - flow_nic_in_23_23 + flow_nic_out_24_nic_in_23
   + flow_nic_out_25_nic_in_23 + flow_nic_out_26_nic_in_23
   + flow_nic_out_27_nic_in_23 + flow_nic_out_28_nic_in_23
   + flow_nic_out_29_nic_in_23 + flow_nic_out_30_nic_in_23
   + flow_nic_out_31_nic_in_23 = 0
 node_flow_constraint_nic_out_23: - flow_nic_out_23_sink
   - flow_nic_out_23_nic_in_0 - flow_nic_out_23_nic_in_1
   - flow_nic_out_23_nic_in_2 - flow_nic_out_23_nic_in_3
   - flow_nic_out_23_nic_in_4 - flow_nic_out_23_nic_in_5
   - flow_nic_out_23_nic_in_6 - flow_nic_out_23_nic_in_7
   - flow_nic_out_23_nic_in_8 - flow_nic_out_23_nic_in_9
   - flow_nic_out_23_nic_in_10 - flow_nic_out_23_nic_in_11
   - flow_nic_out_23_nic_in_12 - flow_nic_out_23_nic_in_13
   - flow_nic_out_23_nic_in_14 - flow_nic_out_23_nic_in_15
   - flow_nic_out_23_nic_in_16 - flow_nic_out_23_nic_in_17
   - flow_nic_out_23_nic_in_18 - flow_nic_out_23_nic_in_19
   - flow_nic_out_23_nic_in_20 - flow_nic_out_23_nic_in_21
   - flow_nic_out_23_nic_in_22 - flow_nic_out_23_nic_in_24
   - flow_nic_out_23_nic_in_25 - flow_nic_out_23_nic_in_26
   - flow_nic_out_23_nic_in_27 - flow_nic_out_23_nic_in_28
   - flow_nic_out_23_nic_in_29 - flow_nic_out_23_nic_in_30
   - flow_nic_out_23_nic_in_31 + flow_23_nic_out_23 = 0
 node_flow_constraint_nic_in_24: flow_nic_out_0_nic_in_24
   + flow_nic_out_1_nic_in_24 + flow_nic_out_2_nic_in_24
   + flow_nic_out_3_nic_in_24 + flow_nic_out_4_nic_in_24
   + flow_nic_out_5_nic_in_24 + flow_nic_out_6_nic_in_24
   + flow_nic_out_7_nic_in_24 + flow_nic_out_8_nic_in_24
   + flow_nic_out_9_nic_in_24 + flow_nic_out_10_nic_in_24
   + flow_nic_out_11_nic_in_24 + flow_nic_out_12_nic_in_24
   + flow_nic_out_13_nic_in_24 + flow_nic_out_14_nic_in_24
   + flow_nic_out_15_nic_in_24 + flow_nic_out_16_nic_in_24
   + flow_nic_out_17_nic_in_24 + flow_nic_out_18_nic_in_24
   + flow_nic_out_19_nic_in_24 + flow_nic_out_20_nic_in_24
   + flow_nic_out_21_nic_in_24 + flow_nic_out_22_nic_in_24
   + flow_nic_out_23_nic_in_24 + flow_source_nic_in_24 - flow_nic_in_24_24
   + flow_nic_out_25_nic_in_24 + flow_nic_out_26_nic_in_24
   + flow_nic_out_27_nic_in_24 + flow_nic_out_28_nic_in_24
   + flow_nic_out_29_nic_in_24 + flow_nic_out_30_nic_in_24
   + flow_nic_out_31_nic_in_24 = 0
 node_flow_constraint_nic_out_24: - flow_nic_out_24_sink
   - flow_nic_out_24_nic_in_0 - flow_nic_out_24_nic_in_1
   - flow_nic_out_24_nic_in_2 - flow_nic_out_24_nic_in_3
   - flow_nic_out_24_nic_in_4 - flow_nic_out_24_nic_in_5
   - flow_nic_out_24_nic_in_6 - flow_nic_out_24_nic_in_7
   - flow_nic_out_24_nic_in_8 - flow_nic_out_24_nic_in_9
   - flow_nic_out_24_nic_in_10 - flow_nic_out_24_nic_in_11
   - flow_nic_out_24_nic_in_12 - flow_nic_out_24_nic_in_13
   - flow_nic_out_24_nic_in_14 - flow_nic_out_24_nic_in_15
   - flow_nic_out_24_nic_in_16 - flow_nic_out_24_nic_in_17
   - flow_nic_out_24_nic_in_18 - flow_nic_out_24_nic_in_19
   - flow_nic_out_24_nic_in_20 - flow_nic_out_24_nic_in_21
   - flow_nic_out_24_nic_in_22 - flow_nic_out_24_nic_in_23
   - flow_nic_out_24_nic_in_25 - flow_nic_out_24_nic_in_26
   - flow_nic_out_24_nic_in_27 - flow_nic_out_24_nic_in_28
   - flow_nic_out_24_nic_in_29 - flow_nic_out_24_nic_in_30
   - flow_nic_out_24_nic_in_31 + flow_24_nic_out_24 = 0
 node_flow_constraint_nic_in_25: flow_nic_out_0_nic_in_25
   + flow_nic_out_1_nic_in_25 + flow_nic_out_2_nic_in_25
   + flow_nic_out_3_nic_in_25 + flow_nic_out_4_nic_in_25
   + flow_nic_out_5_nic_in_25 + flow_nic_out_6_nic_in_25
   + flow_nic_out_7_nic_in_25 + flow_nic_out_8_nic_in_25
   + flow_nic_out_9_nic_in_25 + flow_nic_out_10_nic_in_25
   + flow_nic_out_11_nic_in_25 + flow_nic_out_12_nic_in_25
   + flow_nic_out_13_nic_in_25 + flow_nic_out_14_nic_in_25
   + flow_nic_out_15_nic_in_25 + flow_nic_out_16_nic_in_25
   + flow_nic_out_17_nic_in_25 + flow_nic_out_18_nic_in_25
   + flow_nic_out_19_nic_in_25 + flow_nic_out_20_nic_in_25
   + flow_nic_out_21_nic_in_25 + flow_nic_out_22_nic_in_25
   + flow_nic_out_23_nic_in_25 + flow_nic_out_24_nic_in_25
   + flow_source_nic_in_25 - flow_nic_in_25_25 + flow_nic_out_26_nic_in_25
   + flow_nic_out_27_nic_in_25 + flow_nic_out_28_nic_in_25
   + flow_nic_out_29_nic_in_25 + flow_nic_out_30_nic_in_25
   + flow_nic_out_31_nic_in_25 = 0
 node_flow_constraint_nic_out_25: - flow_nic_out_25_sink
   - flow_nic_out_25_nic_in_0 - flow_nic_out_25_nic_in_1
   - flow_nic_out_25_nic_in_2 - flow_nic_out_25_nic_in_3
   - flow_nic_out_25_nic_in_4 - flow_nic_out_25_nic_in_5
   - flow_nic_out_25_nic_in_6 - flow_nic_out_25_nic_in_7
   - flow_nic_out_25_nic_in_8 - flow_nic_out_25_nic_in_9
   - flow_nic_out_25_nic_in_10 - flow_nic_out_25_nic_in_11
   - flow_nic_out_25_nic_in_12 - flow_nic_out_25_nic_in_13
   - flow_nic_out_25_nic_in_14 - flow_nic_out_25_nic_in_15
   - flow_nic_out_25_nic_in_16 - flow_nic_out_25_nic_in_17
   - flow_nic_out_25_nic_in_18 - flow_nic_out_25_nic_in_19
   - flow_nic_out_25_nic_in_20 - flow_nic_out_25_nic_in_21
   - flow_nic_out_25_nic_in_22 - flow_nic_out_25_nic_in_23
   - flow_nic_out_25_nic_in_24 - flow_nic_out_25_nic_in_26
   - flow_nic_out_25_nic_in_27 - flow_nic_out_25_nic_in_28
   - flow_nic_out_25_nic_in_29 - flow_nic_out_25_nic_in_30
   - flow_nic_out_25_nic_in_31 + flow_25_nic_out_25 = 0
 node_flow_constraint_nic_in_26: flow_nic_out_0_nic_in_26
   + flow_nic_out_1_nic_in_26 + flow_nic_out_2_nic_in_26
   + flow_nic_out_3_nic_in_26 + flow_nic_out_4_nic_in_26
   + flow_nic_out_5_nic_in_26 + flow_nic_out_6_nic_in_26
   + flow_nic_out_7_nic_in_26 + flow_nic_out_8_nic_in_26
   + flow_nic_out_9_nic_in_26 + flow_nic_out_10_nic_in_26
   + flow_nic_out_11_nic_in_26 + flow_nic_out_12_nic_in_26
   + flow_nic_out_13_nic_in_26 + flow_nic_out_14_nic_in_26
   + flow_nic_out_15_nic_in_26 + flow_nic_out_16_nic_in_26
   + flow_nic_out_17_nic_in_26 + flow_nic_out_18_nic_in_26
   + flow_nic_out_19_nic_in_26 + flow_nic_out_20_nic_in_26
   + flow_nic_out_21_nic_in_26 + flow_nic_out_22_nic_in_26
   + flow_nic_out_23_nic_in_26 + flow_nic_out_24_nic_in_26
   + flow_nic_out_25_nic_in_26 + flow_source_nic_in_26 - flow_nic_in_26_26
   + flow_nic_out_27_nic_in_26 + flow_nic_out_28_nic_in_26
   + flow_nic_out_29_nic_in_26 + flow_nic_out_30_nic_in_26
   + flow_nic_out_31_nic_in_26 = 0
 node_flow_constraint_nic_out_26: - flow_nic_out_26_sink
   - flow_nic_out_26_nic_in_0 - flow_nic_out_26_nic_in_1
   - flow_nic_out_26_nic_in_2 - flow_nic_out_26_nic_in_3
   - flow_nic_out_26_nic_in_4 - flow_nic_out_26_nic_in_5
   - flow_nic_out_26_nic_in_6 - flow_nic_out_26_nic_in_7
   - flow_nic_out_26_nic_in_8 - flow_nic_out_26_nic_in_9
   - flow_nic_out_26_nic_in_10 - flow_nic_out_26_nic_in_11
   - flow_nic_out_26_nic_in_12 - flow_nic_out_26_nic_in_13
   - flow_nic_out_26_nic_in_14 - flow_nic_out_26_nic_in_15
   - flow_nic_out_26_nic_in_16 - flow_nic_out_26_nic_in_17
   - flow_nic_out_26_nic_in_18 - flow_nic_out_26_nic_in_19
   - flow_nic_out_26_nic_in_20 - flow_nic_out_26_nic_in_21
   - flow_nic_out_26_nic_in_22 - flow_nic_out_26_nic_in_23
   - flow_nic_out_26_nic_in_24 - flow_nic_out_26_nic_in_25
   - flow_nic_out_26_nic_in_27 - flow_nic_out_26_nic_in_28
   - flow_nic_out_26_nic_in_29 - flow_nic_out_26_nic_in_30
   - flow_nic_out_26_nic_in_31 + flow_26_nic_out_26 = 0
 node_flow_constraint_nic_in_27: flow_nic_out_0_nic_in_27
   + flow_nic_out_1_nic_in_27 + flow_nic_out_2_nic_in_27
   + flow_nic_out_3_nic_in_27 + flow_nic_out_4_nic_in_27
   + flow_nic_out_5_nic_in_27 + flow_nic_out_6_nic_in_27
   + flow_nic_out_7_nic_in_27 + flow_nic_out_8_nic_in_27
   + flow_nic_out_9_nic_in_27 + flow_nic_out_10_nic_in_27
   + flow_nic_out_11_nic_in_27 + flow_nic_out_12_nic_in_27
   + flow_nic_out_13_nic_in_27 + flow_nic_out_14_nic_in_27
   + flow_nic_out_15_nic_in_27 + flow_nic_out_16_nic_in_27
   + flow_nic_out_17_nic_in_27 + flow_nic_out_18_nic_in_27
   + flow_nic_out_19_nic_in_27 + flow_nic_out_20_nic_in_27
   + flow_nic_out_21_nic_in_27 + flow_nic_out_22_nic_in_27
   + flow_nic_out_23_nic_in_27 + flow_nic_out_24_nic_in_27
   + flow_nic_out_25_nic_in_27 + flow_nic_out_26_nic_in_27
   + flow_source_nic_in_27 - flow_nic_in_27_27 + flow_nic_out_28_nic_in_27
   + flow_nic_out_29_nic_in_27 + flow_nic_out_30_nic_in_27
   + flow_nic_out_31_nic_in_27 = 0
 node_flow_constraint_nic_out_27: - flow_nic_out_27_sink
   - flow_nic_out_27_nic_in_0 - flow_nic_out_27_nic_in_1
   - flow_nic_out_27_nic_in_2 - flow_nic_out_27_nic_in_3
   - flow_nic_out_27_nic_in_4 - flow_nic_out_27_nic_in_5
   - flow_nic_out_27_nic_in_6 - flow_nic_out_27_nic_in_7
   - flow_nic_out_27_nic_in_8 - flow_nic_out_27_nic_in_9
   - flow_nic_out_27_nic_in_10 - flow_nic_out_27_nic_in_11
   - flow_nic_out_27_nic_in_12 - flow_nic_out_27_nic_in_13
   - flow_nic_out_27_nic_in_14 - flow_nic_out_27_nic_in_15
   - flow_nic_out_27_nic_in_16 - flow_nic_out_27_nic_in_17
   - flow_nic_out_27_nic_in_18 - flow_nic_out_27_nic_in_19
   - flow_nic_out_27_nic_in_20 - flow_nic_out_27_nic_in_21
   - flow_nic_out_27_nic_in_22 - flow_nic_out_27_nic_in_23
   - flow_nic_out_27_nic_in_24 - flow_nic_out_27_nic_in_25
   - flow_nic_out_27_nic_in_26 - flow_nic_out_27_nic_in_28
   - flow_nic_out_27_nic_in_29 - flow_nic_out_27_nic_in_30
   - flow_nic_out_27_nic_in_31 + flow_27_nic_out_27 = 0
 node_flow_constraint_nic_in_28: flow_nic_out_0_nic_in_28
   + flow_nic_out_1_nic_in_28 + flow_nic_out_2_nic_in_28
   + flow_nic_out_3_nic_in_28 + flow_nic_out_4_nic_in_28
   + flow_nic_out_5_nic_in_28 + flow_nic_out_6_nic_in_28
   + flow_nic_out_7_nic_in_28 + flow_nic_out_8_nic_in_28
   + flow_nic_out_9_nic_in_28 + flow_nic_out_10_nic_in_28
   + flow_nic_out_11_nic_in_28 + flow_nic_out_12_nic_in_28
   + flow_nic_out_13_nic_in_28 + flow_nic_out_14_nic_in_28
   + flow_nic_out_15_nic_in_28 + flow_nic_out_16_nic_in_28
   + flow_nic_out_17_nic_in_28 + flow_nic_out_18_nic_in_28
   + flow_nic_out_19_nic_in_28 + flow_nic_out_20_nic_in_28
   + flow_nic_out_21_nic_in_28 + flow_nic_out_22_nic_in_28
   + flow_nic_out_23_nic_in_28 + flow_nic_out_24_nic_in_28
   + flow_nic_out_25_nic_in_28 + flow_nic_out_26_nic_in_28
   + flow_nic_out_27_nic_in_28 + flow_source_nic_in_28 - flow_nic_in_28_28
   + flow_nic_out_29_nic_in_28 + flow_nic_out_30_nic_in_28
   + flow_nic_out_31_nic_in_28 = 0
 node_flow_constraint_nic_out_28: - flow_nic_out_28_sink
   - flow_nic_out_28_nic_in_0 - flow_nic_out_28_nic_in_1
   - flow_nic_out_28_nic_in_2 - flow_nic_out_28_nic_in_3
   - flow_nic_out_28_nic_in_4 - flow_nic_out_28_nic_in_5
   - flow_nic_out_28_nic_in_6 - flow_nic_out_28_nic_in_7
   - flow_nic_out_28_nic_in_8 - flow_nic_out_28_nic_in_9
   - flow_nic_out_28_nic_in_10 - flow_nic_out_28_nic_in_11
   - flow_nic_out_28_nic_in_12 - flow_nic_out_28_nic_in_13
   - flow_nic_out_28_nic_in_14 - flow_nic_out_28_nic_in_15
   - flow_nic_out_28_nic_in_16 - flow_nic_out_28_nic_in_17
   - flow_nic_out_28_nic_in_18 - flow_nic_out_28_nic_in_19
   - flow_nic_out_28_nic_in_20 - flow_nic_out_28_nic_in_21
   - flow_nic_out_28_nic_in_22 - flow_nic_out_28_nic_in_23
   - flow_nic_out_28_nic_in_24 - flow_nic_out_28_nic_in_25
   - flow_nic_out_28_nic_in_26 - flow_nic_out_28_nic_in_27
   - flow_nic_out_28_nic_in_29 - flow_nic_out_28_nic_in_30
   - flow_nic_out_28_nic_in_31 + flow_28_nic_out_28 = 0
 node_flow_constraint_nic_in_29: flow_nic_out_0_nic_in_29
   + flow_nic_out_1_nic_in_29 + flow_nic_out_2_nic_in_29
   + flow_nic_out_3_nic_in_29 + flow_nic_out_4_nic_in_29
   + flow_nic_out_5_nic_in_29 + flow_nic_out_6_nic_in_29
   + flow_nic_out_7_nic_in_29 + flow_nic_out_8_nic_in_29
   + flow_nic_out_9_nic_in_29 + flow_nic_out_10_nic_in_29
   + flow_nic_out_11_nic_in_29 + flow_nic_out_12_nic_in_29
   + flow_nic_out_13_nic_in_29 + flow_nic_out_14_nic_in_29
   + flow_nic_out_15_nic_in_29 + flow_nic_out_16_nic_in_29
   + flow_nic_out_17_nic_in_29 + flow_nic_out_18_nic_in_29
   + flow_nic_out_19_nic_in_29 + flow_nic_out_20_nic_in_29
   + flow_nic_out_21_nic_in_29 + flow_nic_out_22_nic_in_29
   + flow_nic_out_23_nic_in_29 + flow_nic_out_24_nic_in_29
   + flow_nic_out_25_nic_in_29 + flow_nic_out_26_nic_in_29
   + flow_nic_out_27_nic_in_29 + flow_nic_out_28_nic_in_29
   + flow_source_nic_in_29 - flow_nic_in_29_29 + flow_nic_out_30_nic_in_29
   + flow_nic_out_31_nic_in_29 = 0
 node_flow_constraint_nic_out_29: - flow_nic_out_29_sink
   - flow_nic_out_29_nic_in_0 - flow_nic_out_29_nic_in_1
   - flow_nic_out_29_nic_in_2 - flow_nic_out_29_nic_in_3
   - flow_nic_out_29_nic_in_4 - flow_nic_out_29_nic_in_5
   - flow_nic_out_29_nic_in_6 - flow_nic_out_29_nic_in_7
   - flow_nic_out_29_nic_in_8 - flow_nic_out_29_nic_in_9
   - flow_nic_out_29_nic_in_10 - flow_nic_out_29_nic_in_11
   - flow_nic_out_29_nic_in_12 - flow_nic_out_29_nic_in_13
   - flow_nic_out_29_nic_in_14 - flow_nic_out_29_nic_in_15
   - flow_nic_out_29_nic_in_16 - flow_nic_out_29_nic_in_17
   - flow_nic_out_29_nic_in_18 - flow_nic_out_29_nic_in_19
   - flow_nic_out_29_nic_in_20 - flow_nic_out_29_nic_in_21
   - flow_nic_out_29_nic_in_22 - flow_nic_out_29_nic_in_23
   - flow_nic_out_29_nic_in_24 - flow_nic_out_29_nic_in_25
   - flow_nic_out_29_nic_in_26 - flow_nic_out_29_nic_in_27
   - flow_nic_out_29_nic_in_28 - flow_nic_out_29_nic_in_30
   - flow_nic_out_29_nic_in_31 + flow_29_nic_out_29 = 0
 node_flow_constraint_nic_in_30: flow_nic_out_0_nic_in_30
   + flow_nic_out_1_nic_in_30 + flow_nic_out_2_nic_in_30
   + flow_nic_out_3_nic_in_30 + flow_nic_out_4_nic_in_30
   + flow_nic_out_5_nic_in_30 + flow_nic_out_6_nic_in_30
   + flow_nic_out_7_nic_in_30 + flow_nic_out_8_nic_in_30
   + flow_nic_out_9_nic_in_30 + flow_nic_out_10_nic_in_30
   + flow_nic_out_11_nic_in_30 + flow_nic_out_12_nic_in_30
   + flow_nic_out_13_nic_in_30 + flow_nic_out_14_nic_in_30
   + flow_nic_out_15_nic_in_30 + flow_nic_out_16_nic_in_30
   + flow_nic_out_17_nic_in_30 + flow_nic_out_18_nic_in_30
   + flow_nic_out_19_nic_in_30 + flow_nic_out_20_nic_in_30
   + flow_nic_out_21_nic_in_30 + flow_nic_out_22_nic_in_30
   + flow_nic_out_23_nic_in_30 + flow_nic_out_24_nic_in_30
   + flow_nic_out_25_nic_in_30 + flow_nic_out_26_nic_in_30
   + flow_nic_out_27_nic_in_30 + flow_nic_out_28_nic_in_30
   + flow_nic_out_29_nic_in_30 + flow_source_nic_in_30 - flow_nic_in_30_30
   + flow_nic_out_31_nic_in_30 = 0
 node_flow_constraint_nic_out_30: - flow_nic_out_30_sink
   - flow_nic_out_30_nic_in_0 - flow_nic_out_30_nic_in_1
   - flow_nic_out_30_nic_in_2 - flow_nic_out_30_nic_in_3
   - flow_nic_out_30_nic_in_4 - flow_nic_out_30_nic_in_5
   - flow_nic_out_30_nic_in_6 - flow_nic_out_30_nic_in_7
   - flow_nic_out_30_nic_in_8 - flow_nic_out_30_nic_in_9
   - flow_nic_out_30_nic_in_10 - flow_nic_out_30_nic_in_11
   - flow_nic_out_30_nic_in_12 - flow_nic_out_30_nic_in_13
   - flow_nic_out_30_nic_in_14 - flow_nic_out_30_nic_in_15
   - flow_nic_out_30_nic_in_16 - flow_nic_out_30_nic_in_17
   - flow_nic_out_30_nic_in_18 - flow_nic_out_30_nic_in_19
   - flow_nic_out_30_nic_in_20 - flow_nic_out_30_nic_in_21
   - flow_nic_out_30_nic_in_22 - flow_nic_out_30_nic_in_23
   - flow_nic_out_30_nic_in_24 - flow_nic_out_30_nic_in_25
   - flow_nic_out_30_nic_in_26 - flow_nic_out_30_nic_in_27
   - flow_nic_out_30_nic_in_28 - flow_nic_out_30_nic_in_29
   - flow_nic_out_30_nic_in_31 + flow_30_nic_out_30 = 0
 node_flow_constraint_nic_in_31: flow_nic_out_0_nic_in_31
   + flow_nic_out_1_nic_in_31 + flow_nic_out_2_nic_in_31
   + flow_nic_out_3_nic_in_31 + flow_nic_out_4_nic_in_31
   + flow_nic_out_5_nic_in_31 + flow_nic_out_6_nic_in_31
   + flow_nic_out_7_nic_in_31 + flow_nic_out_8_nic_in_31
   + flow_nic_out_9_nic_in_31 + flow_nic_out_10_nic_in_31
   + flow_nic_out_11_nic_in_31 + flow_nic_out_12_nic_in_31
   + flow_nic_out_13_nic_in_31 + flow_nic_out_14_nic_in_31
   + flow_nic_out_15_nic_in_31 + flow_nic_out_16_nic_in_31
   + flow_nic_out_17_nic_in_31 + flow_nic_out_18_nic_in_31
   + flow_nic_out_19_nic_in_31 + flow_nic_out_20_nic_in_31
   + flow_nic_out_21_nic_in_31 + flow_nic_out_22_nic_in_31
   + flow_nic_out_23_nic_in_31 + flow_nic_out_24_nic_in_31
   + flow_nic_out_25_nic_in_31 + flow_nic_out_26_nic_in_31
   + flow_nic_out_27_nic_in_31 + flow_nic_out_28_nic_in_31
   + flow_nic_out_29_nic_in_31 + flow_nic_out_30_nic_in_31
   + flow_source_nic_in_31 - flow_nic_in_31_31 = 0
 node_flow_constraint_nic_out_31: - flow_nic_out_31_sink
   - flow_nic_out_31_nic_in_0 - flow_nic_out_31_nic_in_1
   - flow_nic_out_31_nic_in_2 - flow_nic_out_31_nic_in_3
   - flow_nic_out_31_nic_in_4 - flow_nic_out_31_nic_in_5
   - flow_nic_out_31_nic_in_6 - flow_nic_out_31_nic_in_7
   - flow_nic_out_31_nic_in_8 - flow_nic_out_31_nic_in_9
   - flow_nic_out_31_nic_in_10 - flow_nic_out_31_nic_in_11
   - flow_nic_out_31_nic_in_12 - flow_nic_out_31_nic_in_13
   - flow_nic_out_31_nic_in_14 - flow_nic_out_31_nic_in_15
   - flow_nic_out_31_nic_in_16 - flow_nic_out_31_nic_in_17
   - flow_nic_out_31_nic_in_18 - flow_nic_out_31_nic_in_19
   - flow_nic_out_31_nic_in_20 - flow_nic_out_31_nic_in_21
   - flow_nic_out_31_nic_in_22 - flow_nic_out_31_nic_in_23
   - flow_nic_out_31_nic_in_24 - flow_nic_out_31_nic_in_25
   - flow_nic_out_31_nic_in_26 - flow_nic_out_31_nic_in_27
   - flow_nic_out_31_nic_in_28 - flow_nic_out_31_nic_in_29
   - flow_nic_out_31_nic_in_30 + flow_31_nic_out_31 = 0
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
 node_flow_constraint_16: flow_nic_in_16_16 - flow_16_nic_out_16 = 0
 node_flow_constraint_17: flow_nic_in_17_17 - flow_17_nic_out_17 = 0
 node_flow_constraint_18: flow_nic_in_18_18 - flow_18_nic_out_18 = 0
 node_flow_constraint_19: flow_nic_in_19_19 - flow_19_nic_out_19 = 0
 node_flow_constraint_20: flow_nic_in_20_20 - flow_20_nic_out_20 = 0
 node_flow_constraint_21: flow_nic_in_21_21 - flow_21_nic_out_21 = 0
 node_flow_constraint_22: flow_nic_in_22_22 - flow_22_nic_out_22 = 0
 node_flow_constraint_23: flow_nic_in_23_23 - flow_23_nic_out_23 = 0
 node_flow_constraint_24: flow_nic_in_24_24 - flow_24_nic_out_24 = 0
 node_flow_constraint_25: flow_nic_in_25_25 - flow_25_nic_out_25 = 0
 node_flow_constraint_26: flow_nic_in_26_26 - flow_26_nic_out_26 = 0
 node_flow_constraint_27: flow_nic_in_27_27 - flow_27_nic_out_27 = 0
 node_flow_constraint_28: flow_nic_in_28_28 - flow_28_nic_out_28 = 0
 node_flow_constraint_29: flow_nic_in_29_29 - flow_29_nic_out_29 = 0
 node_flow_constraint_30: flow_nic_in_30_30 - flow_30_nic_out_30 = 0
 node_flow_constraint_31: flow_nic_in_31_31 - flow_31_nic_out_31 = 0
 constr_is_start_zero_0: start_0 + 80 is_start_zero_0_for_source_nic_in_0
   <= 80
 constr_edge_switch_start_zero_0_OR: switch_source_nic_in_0
   - is_start_zero_0_for_source_nic_in_0 <= 0
 constr_is_end_complete_0: start_0 + hold_0_1 + 2 hold_0_2 + 3 hold_0_3
   + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 - 80 is_end_complete_0 >= 0
 constr_edge_switch_end_max_0_OR: switch_nic_out_0_sink - is_end_complete_0
   <= 0
 constr_link_match_1_0_1: 80 switch_nic_out_0_nic_in_1 - group_start_1
   + group_end_0 <= 80
 constr_link_match_2_0_1: 80 switch_nic_out_0_nic_in_1 + group_start_1
   - group_end_0 <= 80
 constr_link_match_1_0_2: 80 switch_nic_out_0_nic_in_2 + group_end_0
   - group_start_2 <= 80
 constr_link_match_2_0_2: 80 switch_nic_out_0_nic_in_2 - group_end_0
   + group_start_2 <= 80
 constr_link_match_1_0_3: 80 switch_nic_out_0_nic_in_3 + group_end_0
   - group_start_3 <= 80
 constr_link_match_2_0_3: 80 switch_nic_out_0_nic_in_3 - group_end_0
   + group_start_3 <= 80
 constr_link_match_1_0_4: 80 switch_nic_out_0_nic_in_4 + group_end_0
   - group_start_4 <= 80
 constr_link_match_2_0_4: 80 switch_nic_out_0_nic_in_4 - group_end_0
   + group_start_4 <= 80
 constr_link_match_1_0_5: 80 switch_nic_out_0_nic_in_5 + group_end_0
   - group_start_5 <= 80
 constr_link_match_2_0_5: 80 switch_nic_out_0_nic_in_5 - group_end_0
   + group_start_5 <= 80
 constr_link_match_1_0_6: 80 switch_nic_out_0_nic_in_6 + group_end_0
   - group_start_6 <= 80
 constr_link_match_2_0_6: 80 switch_nic_out_0_nic_in_6 - group_end_0
   + group_start_6 <= 80
 constr_link_match_1_0_7: 80 switch_nic_out_0_nic_in_7 + group_end_0
   - group_start_7 <= 80
 constr_link_match_2_0_7: 80 switch_nic_out_0_nic_in_7 - group_end_0
   + group_start_7 <= 80
 constr_link_match_1_0_8: 80 switch_nic_out_0_nic_in_8 + group_end_0
   - group_start_8 <= 80
 constr_link_match_2_0_8: 80 switch_nic_out_0_nic_in_8 - group_end_0
   + group_start_8 <= 80
 constr_link_match_1_0_9: 80 switch_nic_out_0_nic_in_9 + group_end_0
   - group_start_9 <= 80
 constr_link_match_2_0_9: 80 switch_nic_out_0_nic_in_9 - group_end_0
   + group_start_9 <= 80
 constr_link_match_1_0_10: 80 switch_nic_out_0_nic_in_10 + group_end_0
   - group_start_10 <= 80
 constr_link_match_2_0_10: 80 switch_nic_out_0_nic_in_10 - group_end_0
   + group_start_10 <= 80
 constr_link_match_1_0_11: 80 switch_nic_out_0_nic_in_11 + group_end_0
   - group_start_11 <= 80
 constr_link_match_2_0_11: 80 switch_nic_out_0_nic_in_11 - group_end_0
   + group_start_11 <= 80
 constr_link_match_1_0_12: 80 switch_nic_out_0_nic_in_12 + group_end_0
   - group_start_12 <= 80
 constr_link_match_2_0_12: 80 switch_nic_out_0_nic_in_12 - group_end_0
   + group_start_12 <= 80
 constr_link_match_1_0_13: 80 switch_nic_out_0_nic_in_13 + group_end_0
   - group_start_13 <= 80
 constr_link_match_2_0_13: 80 switch_nic_out_0_nic_in_13 - group_end_0
   + group_start_13 <= 80
 constr_link_match_1_0_14: 80 switch_nic_out_0_nic_in_14 + group_end_0
   - group_start_14 <= 80
 constr_link_match_2_0_14: 80 switch_nic_out_0_nic_in_14 - group_end_0
   + group_start_14 <= 80
 constr_link_match_1_0_15: 80 switch_nic_out_0_nic_in_15 + group_end_0
   - group_start_15 <= 80
 constr_link_match_2_0_15: 80 switch_nic_out_0_nic_in_15 - group_end_0
   + group_start_15 <= 80
 constr_link_match_1_0_16: 80 switch_nic_out_0_nic_in_16 + group_end_0
   - group_start_16 <= 80
 constr_link_match_2_0_16: 80 switch_nic_out_0_nic_in_16 - group_end_0
   + group_start_16 <= 80
 constr_link_match_1_0_17: 80 switch_nic_out_0_nic_in_17 + group_end_0
   - group_start_17 <= 80
 constr_link_match_2_0_17: 80 switch_nic_out_0_nic_in_17 - group_end_0
   + group_start_17 <= 80
 constr_link_match_1_0_18: 80 switch_nic_out_0_nic_in_18 + group_end_0
   - group_start_18 <= 80
 constr_link_match_2_0_18: 80 switch_nic_out_0_nic_in_18 - group_end_0
   + group_start_18 <= 80
 constr_link_match_1_0_19: 80 switch_nic_out_0_nic_in_19 + group_end_0
   - group_start_19 <= 80
 constr_link_match_2_0_19: 80 switch_nic_out_0_nic_in_19 - group_end_0
   + group_start_19 <= 80
 constr_link_match_1_0_20: 80 switch_nic_out_0_nic_in_20 + group_end_0
   - group_start_20 <= 80
 constr_link_match_2_0_20: 80 switch_nic_out_0_nic_in_20 - group_end_0
   + group_start_20 <= 80
 constr_link_match_1_0_21: 80 switch_nic_out_0_nic_in_21 + group_end_0
   - group_start_21 <= 80
 constr_link_match_2_0_21: 80 switch_nic_out_0_nic_in_21 - group_end_0
   + group_start_21 <= 80
 constr_link_match_1_0_22: 80 switch_nic_out_0_nic_in_22 + group_end_0
   - group_start_22 <= 80
 constr_link_match_2_0_22: 80 switch_nic_out_0_nic_in_22 - group_end_0
   + group_start_22 <= 80
 constr_link_match_1_0_23: 80 switch_nic_out_0_nic_in_23 + group_end_0
   - group_start_23 <= 80
 constr_link_match_2_0_23: 80 switch_nic_out_0_nic_in_23 - group_end_0
   + group_start_23 <= 80
 constr_link_match_1_0_24: 80 switch_nic_out_0_nic_in_24 + group_end_0
   - group_start_24 <= 80
 constr_link_match_2_0_24: 80 switch_nic_out_0_nic_in_24 - group_end_0
   + group_start_24 <= 80
 constr_link_match_1_0_25: 80 switch_nic_out_0_nic_in_25 + group_end_0
   - group_start_25 <= 80
 constr_link_match_2_0_25: 80 switch_nic_out_0_nic_in_25 - group_end_0
   + group_start_25 <= 80
 constr_link_match_1_0_26: 80 switch_nic_out_0_nic_in_26 + group_end_0
   - group_start_26 <= 80
 constr_link_match_2_0_26: 80 switch_nic_out_0_nic_in_26 - group_end_0
   + group_start_26 <= 80
 constr_link_match_1_0_27: 80 switch_nic_out_0_nic_in_27 + group_end_0
   - group_start_27 <= 80
 constr_link_match_2_0_27: 80 switch_nic_out_0_nic_in_27 - group_end_0
   + group_start_27 <= 80
 constr_link_match_1_0_28: 80 switch_nic_out_0_nic_in_28 + group_end_0
   - group_start_28 <= 80
 constr_link_match_2_0_28: 80 switch_nic_out_0_nic_in_28 - group_end_0
   + group_start_28 <= 80
 constr_link_match_1_0_29: 80 switch_nic_out_0_nic_in_29 + group_end_0
   - group_start_29 <= 80
 constr_link_match_2_0_29: 80 switch_nic_out_0_nic_in_29 - group_end_0
   + group_start_29 <= 80
 constr_link_match_1_0_30: 80 switch_nic_out_0_nic_in_30 + group_end_0
   - group_start_30 <= 80
 constr_link_match_2_0_30: 80 switch_nic_out_0_nic_in_30 - group_end_0
   + group_start_30 <= 80
 constr_link_match_1_0_31: 80 switch_nic_out_0_nic_in_31 + group_end_0
   - group_start_31 <= 80
 constr_link_match_2_0_31: 80 switch_nic_out_0_nic_in_31 - group_end_0
   + group_start_31 <= 80
 constr_is_start_zero_1: start_1 + 80 is_start_zero_1_for_source_nic_in_1
   <= 80
 constr_edge_switch_start_zero_1_OR: switch_source_nic_in_1
   - is_start_zero_1_for_source_nic_in_1 <= 0
 constr_is_end_complete_1: start_1 + hold_1_1 + 2 hold_1_2 + 3 hold_1_3
   + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 - 80 is_end_complete_1 >= 0
 constr_edge_switch_end_max_1_OR: switch_nic_out_1_sink - is_end_complete_1
   <= 0
 constr_link_match_1_1_0: 80 switch_nic_out_1_nic_in_0 - group_start_0
   + group_end_1 <= 80
 constr_link_match_2_1_0: 80 switch_nic_out_1_nic_in_0 + group_start_0
   - group_end_1 <= 80
 constr_link_match_1_1_2: 80 switch_nic_out_1_nic_in_2 - group_start_2
   + group_end_1 <= 80
 constr_link_match_2_1_2: 80 switch_nic_out_1_nic_in_2 + group_start_2
   - group_end_1 <= 80
 constr_link_match_1_1_3: 80 switch_nic_out_1_nic_in_3 - group_start_3
   + group_end_1 <= 80
 constr_link_match_2_1_3: 80 switch_nic_out_1_nic_in_3 + group_start_3
   - group_end_1 <= 80
 constr_link_match_1_1_4: 80 switch_nic_out_1_nic_in_4 - group_start_4
   + group_end_1 <= 80
 constr_link_match_2_1_4: 80 switch_nic_out_1_nic_in_4 + group_start_4
   - group_end_1 <= 80
 constr_link_match_1_1_5: 80 switch_nic_out_1_nic_in_5 - group_start_5
   + group_end_1 <= 80
 constr_link_match_2_1_5: 80 switch_nic_out_1_nic_in_5 + group_start_5
   - group_end_1 <= 80
 constr_link_match_1_1_6: 80 switch_nic_out_1_nic_in_6 - group_start_6
   + group_end_1 <= 80
 constr_link_match_2_1_6: 80 switch_nic_out_1_nic_in_6 + group_start_6
   - group_end_1 <= 80
 constr_link_match_1_1_7: 80 switch_nic_out_1_nic_in_7 - group_start_7
   + group_end_1 <= 80
 constr_link_match_2_1_7: 80 switch_nic_out_1_nic_in_7 + group_start_7
   - group_end_1 <= 80
 constr_link_match_1_1_8: 80 switch_nic_out_1_nic_in_8 - group_start_8
   + group_end_1 <= 80
 constr_link_match_2_1_8: 80 switch_nic_out_1_nic_in_8 + group_start_8
   - group_end_1 <= 80
 constr_link_match_1_1_9: 80 switch_nic_out_1_nic_in_9 - group_start_9
   + group_end_1 <= 80
 constr_link_match_2_1_9: 80 switch_nic_out_1_nic_in_9 + group_start_9
   - group_end_1 <= 80
 constr_link_match_1_1_10: 80 switch_nic_out_1_nic_in_10 - group_start_10
   + group_end_1 <= 80
 constr_link_match_2_1_10: 80 switch_nic_out_1_nic_in_10 + group_start_10
   - group_end_1 <= 80
 constr_link_match_1_1_11: 80 switch_nic_out_1_nic_in_11 - group_start_11
   + group_end_1 <= 80
 constr_link_match_2_1_11: 80 switch_nic_out_1_nic_in_11 + group_start_11
   - group_end_1 <= 80
 constr_link_match_1_1_12: 80 switch_nic_out_1_nic_in_12 - group_start_12
   + group_end_1 <= 80
 constr_link_match_2_1_12: 80 switch_nic_out_1_nic_in_12 + group_start_12
   - group_end_1 <= 80
 constr_link_match_1_1_13: 80 switch_nic_out_1_nic_in_13 - group_start_13
   + group_end_1 <= 80
 constr_link_match_2_1_13: 80 switch_nic_out_1_nic_in_13 + group_start_13
   - group_end_1 <= 80
 constr_link_match_1_1_14: 80 switch_nic_out_1_nic_in_14 - group_start_14
   + group_end_1 <= 80
 constr_link_match_2_1_14: 80 switch_nic_out_1_nic_in_14 + group_start_14
   - group_end_1 <= 80
 constr_link_match_1_1_15: 80 switch_nic_out_1_nic_in_15 - group_start_15
   + group_end_1 <= 80
 constr_link_match_2_1_15: 80 switch_nic_out_1_nic_in_15 + group_start_15
   - group_end_1 <= 80
 constr_link_match_1_1_16: 80 switch_nic_out_1_nic_in_16 - group_start_16
   + group_end_1 <= 80
 constr_link_match_2_1_16: 80 switch_nic_out_1_nic_in_16 + group_start_16
   - group_end_1 <= 80
 constr_link_match_1_1_17: 80 switch_nic_out_1_nic_in_17 - group_start_17
   + group_end_1 <= 80
 constr_link_match_2_1_17: 80 switch_nic_out_1_nic_in_17 + group_start_17
   - group_end_1 <= 80
 constr_link_match_1_1_18: 80 switch_nic_out_1_nic_in_18 - group_start_18
   + group_end_1 <= 80
 constr_link_match_2_1_18: 80 switch_nic_out_1_nic_in_18 + group_start_18
   - group_end_1 <= 80
 constr_link_match_1_1_19: 80 switch_nic_out_1_nic_in_19 - group_start_19
   + group_end_1 <= 80
 constr_link_match_2_1_19: 80 switch_nic_out_1_nic_in_19 + group_start_19
   - group_end_1 <= 80
 constr_link_match_1_1_20: 80 switch_nic_out_1_nic_in_20 - group_start_20
   + group_end_1 <= 80
 constr_link_match_2_1_20: 80 switch_nic_out_1_nic_in_20 + group_start_20
   - group_end_1 <= 80
 constr_link_match_1_1_21: 80 switch_nic_out_1_nic_in_21 - group_start_21
   + group_end_1 <= 80
 constr_link_match_2_1_21: 80 switch_nic_out_1_nic_in_21 + group_start_21
   - group_end_1 <= 80
 constr_link_match_1_1_22: 80 switch_nic_out_1_nic_in_22 - group_start_22
   + group_end_1 <= 80
 constr_link_match_2_1_22: 80 switch_nic_out_1_nic_in_22 + group_start_22
   - group_end_1 <= 80
 constr_link_match_1_1_23: 80 switch_nic_out_1_nic_in_23 - group_start_23
   + group_end_1 <= 80
 constr_link_match_2_1_23: 80 switch_nic_out_1_nic_in_23 + group_start_23
   - group_end_1 <= 80
 constr_link_match_1_1_24: 80 switch_nic_out_1_nic_in_24 - group_start_24
   + group_end_1 <= 80
 constr_link_match_2_1_24: 80 switch_nic_out_1_nic_in_24 + group_start_24
   - group_end_1 <= 80
 constr_link_match_1_1_25: 80 switch_nic_out_1_nic_in_25 - group_start_25
   + group_end_1 <= 80
 constr_link_match_2_1_25: 80 switch_nic_out_1_nic_in_25 + group_start_25
   - group_end_1 <= 80
 constr_link_match_1_1_26: 80 switch_nic_out_1_nic_in_26 - group_start_26
   + group_end_1 <= 80
 constr_link_match_2_1_26: 80 switch_nic_out_1_nic_in_26 + group_start_26
   - group_end_1 <= 80
 constr_link_match_1_1_27: 80 switch_nic_out_1_nic_in_27 - group_start_27
   + group_end_1 <= 80
 constr_link_match_2_1_27: 80 switch_nic_out_1_nic_in_27 + group_start_27
   - group_end_1 <= 80
 constr_link_match_1_1_28: 80 switch_nic_out_1_nic_in_28 - group_start_28
   + group_end_1 <= 80
 constr_link_match_2_1_28: 80 switch_nic_out_1_nic_in_28 + group_start_28
   - group_end_1 <= 80
 constr_link_match_1_1_29: 80 switch_nic_out_1_nic_in_29 - group_start_29
   + group_end_1 <= 80
 constr_link_match_2_1_29: 80 switch_nic_out_1_nic_in_29 + group_start_29
   - group_end_1 <= 80
 constr_link_match_1_1_30: 80 switch_nic_out_1_nic_in_30 - group_start_30
   + group_end_1 <= 80
 constr_link_match_2_1_30: 80 switch_nic_out_1_nic_in_30 + group_start_30
   - group_end_1 <= 80
 constr_link_match_1_1_31: 80 switch_nic_out_1_nic_in_31 - group_start_31
   + group_end_1 <= 80
 constr_link_match_2_1_31: 80 switch_nic_out_1_nic_in_31 + group_start_31
   - group_end_1 <= 80
 constr_is_start_zero_2: start_2 + 80 is_start_zero_2_for_source_nic_in_2
   <= 80
 constr_edge_switch_start_zero_2_OR: switch_source_nic_in_2
   - is_start_zero_2_for_source_nic_in_2 <= 0
 constr_is_end_complete_2: start_2 + hold_2_1 + 2 hold_2_2 + 3 hold_2_3
   + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 - 80 is_end_complete_2 >= 0
 constr_edge_switch_end_max_2_OR: switch_nic_out_2_sink - is_end_complete_2
   <= 0
 constr_link_match_1_2_0: 80 switch_nic_out_2_nic_in_0 - group_start_0
   + group_end_2 <= 80
 constr_link_match_2_2_0: 80 switch_nic_out_2_nic_in_0 + group_start_0
   - group_end_2 <= 80
 constr_link_match_1_2_1: 80 switch_nic_out_2_nic_in_1 - group_start_1
   + group_end_2 <= 80
 constr_link_match_2_2_1: 80 switch_nic_out_2_nic_in_1 + group_start_1
   - group_end_2 <= 80
 constr_link_match_1_2_3: 80 switch_nic_out_2_nic_in_3 - group_start_3
   + group_end_2 <= 80
 constr_link_match_2_2_3: 80 switch_nic_out_2_nic_in_3 + group_start_3
   - group_end_2 <= 80
 constr_link_match_1_2_4: 80 switch_nic_out_2_nic_in_4 - group_start_4
   + group_end_2 <= 80
 constr_link_match_2_2_4: 80 switch_nic_out_2_nic_in_4 + group_start_4
   - group_end_2 <= 80
 constr_link_match_1_2_5: 80 switch_nic_out_2_nic_in_5 - group_start_5
   + group_end_2 <= 80
 constr_link_match_2_2_5: 80 switch_nic_out_2_nic_in_5 + group_start_5
   - group_end_2 <= 80
 constr_link_match_1_2_6: 80 switch_nic_out_2_nic_in_6 - group_start_6
   + group_end_2 <= 80
 constr_link_match_2_2_6: 80 switch_nic_out_2_nic_in_6 + group_start_6
   - group_end_2 <= 80
 constr_link_match_1_2_7: 80 switch_nic_out_2_nic_in_7 - group_start_7
   + group_end_2 <= 80
 constr_link_match_2_2_7: 80 switch_nic_out_2_nic_in_7 + group_start_7
   - group_end_2 <= 80
 constr_link_match_1_2_8: 80 switch_nic_out_2_nic_in_8 - group_start_8
   + group_end_2 <= 80
 constr_link_match_2_2_8: 80 switch_nic_out_2_nic_in_8 + group_start_8
   - group_end_2 <= 80
 constr_link_match_1_2_9: 80 switch_nic_out_2_nic_in_9 - group_start_9
   + group_end_2 <= 80
 constr_link_match_2_2_9: 80 switch_nic_out_2_nic_in_9 + group_start_9
   - group_end_2 <= 80
 constr_link_match_1_2_10: 80 switch_nic_out_2_nic_in_10 - group_start_10
   + group_end_2 <= 80
 constr_link_match_2_2_10: 80 switch_nic_out_2_nic_in_10 + group_start_10
   - group_end_2 <= 80
 constr_link_match_1_2_11: 80 switch_nic_out_2_nic_in_11 - group_start_11
   + group_end_2 <= 80
 constr_link_match_2_2_11: 80 switch_nic_out_2_nic_in_11 + group_start_11
   - group_end_2 <= 80
 constr_link_match_1_2_12: 80 switch_nic_out_2_nic_in_12 - group_start_12
   + group_end_2 <= 80
 constr_link_match_2_2_12: 80 switch_nic_out_2_nic_in_12 + group_start_12
   - group_end_2 <= 80
 constr_link_match_1_2_13: 80 switch_nic_out_2_nic_in_13 - group_start_13
   + group_end_2 <= 80
 constr_link_match_2_2_13: 80 switch_nic_out_2_nic_in_13 + group_start_13
   - group_end_2 <= 80
 constr_link_match_1_2_14: 80 switch_nic_out_2_nic_in_14 - group_start_14
   + group_end_2 <= 80
 constr_link_match_2_2_14: 80 switch_nic_out_2_nic_in_14 + group_start_14
   - group_end_2 <= 80
 constr_link_match_1_2_15: 80 switch_nic_out_2_nic_in_15 - group_start_15
   + group_end_2 <= 80
 constr_link_match_2_2_15: 80 switch_nic_out_2_nic_in_15 + group_start_15
   - group_end_2 <= 80
 constr_link_match_1_2_16: 80 switch_nic_out_2_nic_in_16 - group_start_16
   + group_end_2 <= 80
 constr_link_match_2_2_16: 80 switch_nic_out_2_nic_in_16 + group_start_16
   - group_end_2 <= 80
 constr_link_match_1_2_17: 80 switch_nic_out_2_nic_in_17 - group_start_17
   + group_end_2 <= 80
 constr_link_match_2_2_17: 80 switch_nic_out_2_nic_in_17 + group_start_17
   - group_end_2 <= 80
 constr_link_match_1_2_18: 80 switch_nic_out_2_nic_in_18 - group_start_18
   + group_end_2 <= 80
 constr_link_match_2_2_18: 80 switch_nic_out_2_nic_in_18 + group_start_18
   - group_end_2 <= 80
 constr_link_match_1_2_19: 80 switch_nic_out_2_nic_in_19 - group_start_19
   + group_end_2 <= 80
 constr_link_match_2_2_19: 80 switch_nic_out_2_nic_in_19 + group_start_19
   - group_end_2 <= 80
 constr_link_match_1_2_20: 80 switch_nic_out_2_nic_in_20 - group_start_20
   + group_end_2 <= 80
 constr_link_match_2_2_20: 80 switch_nic_out_2_nic_in_20 + group_start_20
   - group_end_2 <= 80
 constr_link_match_1_2_21: 80 switch_nic_out_2_nic_in_21 - group_start_21
   + group_end_2 <= 80
 constr_link_match_2_2_21: 80 switch_nic_out_2_nic_in_21 + group_start_21
   - group_end_2 <= 80
 constr_link_match_1_2_22: 80 switch_nic_out_2_nic_in_22 - group_start_22
   + group_end_2 <= 80
 constr_link_match_2_2_22: 80 switch_nic_out_2_nic_in_22 + group_start_22
   - group_end_2 <= 80
 constr_link_match_1_2_23: 80 switch_nic_out_2_nic_in_23 - group_start_23
   + group_end_2 <= 80
 constr_link_match_2_2_23: 80 switch_nic_out_2_nic_in_23 + group_start_23
   - group_end_2 <= 80
 constr_link_match_1_2_24: 80 switch_nic_out_2_nic_in_24 - group_start_24
   + group_end_2 <= 80
 constr_link_match_2_2_24: 80 switch_nic_out_2_nic_in_24 + group_start_24
   - group_end_2 <= 80
 constr_link_match_1_2_25: 80 switch_nic_out_2_nic_in_25 - group_start_25
   + group_end_2 <= 80
 constr_link_match_2_2_25: 80 switch_nic_out_2_nic_in_25 + group_start_25
   - group_end_2 <= 80
 constr_link_match_1_2_26: 80 switch_nic_out_2_nic_in_26 - group_start_26
   + group_end_2 <= 80
 constr_link_match_2_2_26: 80 switch_nic_out_2_nic_in_26 + group_start_26
   - group_end_2 <= 80
 constr_link_match_1_2_27: 80 switch_nic_out_2_nic_in_27 - group_start_27
   + group_end_2 <= 80
 constr_link_match_2_2_27: 80 switch_nic_out_2_nic_in_27 + group_start_27
   - group_end_2 <= 80
 constr_link_match_1_2_28: 80 switch_nic_out_2_nic_in_28 - group_start_28
   + group_end_2 <= 80
 constr_link_match_2_2_28: 80 switch_nic_out_2_nic_in_28 + group_start_28
   - group_end_2 <= 80
 constr_link_match_1_2_29: 80 switch_nic_out_2_nic_in_29 - group_start_29
   + group_end_2 <= 80
 constr_link_match_2_2_29: 80 switch_nic_out_2_nic_in_29 + group_start_29
   - group_end_2 <= 80
 constr_link_match_1_2_30: 80 switch_nic_out_2_nic_in_30 - group_start_30
   + group_end_2 <= 80
 constr_link_match_2_2_30: 80 switch_nic_out_2_nic_in_30 + group_start_30
   - group_end_2 <= 80
 constr_link_match_1_2_31: 80 switch_nic_out_2_nic_in_31 - group_start_31
   + group_end_2 <= 80
 constr_link_match_2_2_31: 80 switch_nic_out_2_nic_in_31 + group_start_31
   - group_end_2 <= 80
 constr_is_start_zero_3: start_3 + 80 is_start_zero_3_for_source_nic_in_3
   <= 80
 constr_edge_switch_start_zero_3_OR: switch_source_nic_in_3
   - is_start_zero_3_for_source_nic_in_3 <= 0
 constr_is_end_complete_3: start_3 + hold_3_1 + 2 hold_3_2 + 3 hold_3_3
   + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 - 80 is_end_complete_3 >= 0
 constr_edge_switch_end_max_3_OR: switch_nic_out_3_sink - is_end_complete_3
   <= 0
 constr_link_match_1_3_0: 80 switch_nic_out_3_nic_in_0 - group_start_0
   + group_end_3 <= 80
 constr_link_match_2_3_0: 80 switch_nic_out_3_nic_in_0 + group_start_0
   - group_end_3 <= 80
 constr_link_match_1_3_1: 80 switch_nic_out_3_nic_in_1 - group_start_1
   + group_end_3 <= 80
 constr_link_match_2_3_1: 80 switch_nic_out_3_nic_in_1 + group_start_1
   - group_end_3 <= 80
 constr_link_match_1_3_2: 80 switch_nic_out_3_nic_in_2 - group_start_2
   + group_end_3 <= 80
 constr_link_match_2_3_2: 80 switch_nic_out_3_nic_in_2 + group_start_2
   - group_end_3 <= 80
 constr_link_match_1_3_4: 80 switch_nic_out_3_nic_in_4 - group_start_4
   + group_end_3 <= 80
 constr_link_match_2_3_4: 80 switch_nic_out_3_nic_in_4 + group_start_4
   - group_end_3 <= 80
 constr_link_match_1_3_5: 80 switch_nic_out_3_nic_in_5 - group_start_5
   + group_end_3 <= 80
 constr_link_match_2_3_5: 80 switch_nic_out_3_nic_in_5 + group_start_5
   - group_end_3 <= 80
 constr_link_match_1_3_6: 80 switch_nic_out_3_nic_in_6 - group_start_6
   + group_end_3 <= 80
 constr_link_match_2_3_6: 80 switch_nic_out_3_nic_in_6 + group_start_6
   - group_end_3 <= 80
 constr_link_match_1_3_7: 80 switch_nic_out_3_nic_in_7 - group_start_7
   + group_end_3 <= 80
 constr_link_match_2_3_7: 80 switch_nic_out_3_nic_in_7 + group_start_7
   - group_end_3 <= 80
 constr_link_match_1_3_8: 80 switch_nic_out_3_nic_in_8 - group_start_8
   + group_end_3 <= 80
 constr_link_match_2_3_8: 80 switch_nic_out_3_nic_in_8 + group_start_8
   - group_end_3 <= 80
 constr_link_match_1_3_9: 80 switch_nic_out_3_nic_in_9 - group_start_9
   + group_end_3 <= 80
 constr_link_match_2_3_9: 80 switch_nic_out_3_nic_in_9 + group_start_9
   - group_end_3 <= 80
 constr_link_match_1_3_10: 80 switch_nic_out_3_nic_in_10 - group_start_10
   + group_end_3 <= 80
 constr_link_match_2_3_10: 80 switch_nic_out_3_nic_in_10 + group_start_10
   - group_end_3 <= 80
 constr_link_match_1_3_11: 80 switch_nic_out_3_nic_in_11 - group_start_11
   + group_end_3 <= 80
 constr_link_match_2_3_11: 80 switch_nic_out_3_nic_in_11 + group_start_11
   - group_end_3 <= 80
 constr_link_match_1_3_12: 80 switch_nic_out_3_nic_in_12 - group_start_12
   + group_end_3 <= 80
 constr_link_match_2_3_12: 80 switch_nic_out_3_nic_in_12 + group_start_12
   - group_end_3 <= 80
 constr_link_match_1_3_13: 80 switch_nic_out_3_nic_in_13 - group_start_13
   + group_end_3 <= 80
 constr_link_match_2_3_13: 80 switch_nic_out_3_nic_in_13 + group_start_13
   - group_end_3 <= 80
 constr_link_match_1_3_14: 80 switch_nic_out_3_nic_in_14 - group_start_14
   + group_end_3 <= 80
 constr_link_match_2_3_14: 80 switch_nic_out_3_nic_in_14 + group_start_14
   - group_end_3 <= 80
 constr_link_match_1_3_15: 80 switch_nic_out_3_nic_in_15 - group_start_15
   + group_end_3 <= 80
 constr_link_match_2_3_15: 80 switch_nic_out_3_nic_in_15 + group_start_15
   - group_end_3 <= 80
 constr_link_match_1_3_16: 80 switch_nic_out_3_nic_in_16 - group_start_16
   + group_end_3 <= 80
 constr_link_match_2_3_16: 80 switch_nic_out_3_nic_in_16 + group_start_16
   - group_end_3 <= 80
 constr_link_match_1_3_17: 80 switch_nic_out_3_nic_in_17 - group_start_17
   + group_end_3 <= 80
 constr_link_match_2_3_17: 80 switch_nic_out_3_nic_in_17 + group_start_17
   - group_end_3 <= 80
 constr_link_match_1_3_18: 80 switch_nic_out_3_nic_in_18 - group_start_18
   + group_end_3 <= 80
 constr_link_match_2_3_18: 80 switch_nic_out_3_nic_in_18 + group_start_18
   - group_end_3 <= 80
 constr_link_match_1_3_19: 80 switch_nic_out_3_nic_in_19 - group_start_19
   + group_end_3 <= 80
 constr_link_match_2_3_19: 80 switch_nic_out_3_nic_in_19 + group_start_19
   - group_end_3 <= 80
 constr_link_match_1_3_20: 80 switch_nic_out_3_nic_in_20 - group_start_20
   + group_end_3 <= 80
 constr_link_match_2_3_20: 80 switch_nic_out_3_nic_in_20 + group_start_20
   - group_end_3 <= 80
 constr_link_match_1_3_21: 80 switch_nic_out_3_nic_in_21 - group_start_21
   + group_end_3 <= 80
 constr_link_match_2_3_21: 80 switch_nic_out_3_nic_in_21 + group_start_21
   - group_end_3 <= 80
 constr_link_match_1_3_22: 80 switch_nic_out_3_nic_in_22 - group_start_22
   + group_end_3 <= 80
 constr_link_match_2_3_22: 80 switch_nic_out_3_nic_in_22 + group_start_22
   - group_end_3 <= 80
 constr_link_match_1_3_23: 80 switch_nic_out_3_nic_in_23 - group_start_23
   + group_end_3 <= 80
 constr_link_match_2_3_23: 80 switch_nic_out_3_nic_in_23 + group_start_23
   - group_end_3 <= 80
 constr_link_match_1_3_24: 80 switch_nic_out_3_nic_in_24 - group_start_24
   + group_end_3 <= 80
 constr_link_match_2_3_24: 80 switch_nic_out_3_nic_in_24 + group_start_24
   - group_end_3 <= 80
 constr_link_match_1_3_25: 80 switch_nic_out_3_nic_in_25 - group_start_25
   + group_end_3 <= 80
 constr_link_match_2_3_25: 80 switch_nic_out_3_nic_in_25 + group_start_25
   - group_end_3 <= 80
 constr_link_match_1_3_26: 80 switch_nic_out_3_nic_in_26 - group_start_26
   + group_end_3 <= 80
 constr_link_match_2_3_26: 80 switch_nic_out_3_nic_in_26 + group_start_26
   - group_end_3 <= 80
 constr_link_match_1_3_27: 80 switch_nic_out_3_nic_in_27 - group_start_27
   + group_end_3 <= 80
 constr_link_match_2_3_27: 80 switch_nic_out_3_nic_in_27 + group_start_27
   - group_end_3 <= 80
 constr_link_match_1_3_28: 80 switch_nic_out_3_nic_in_28 - group_start_28
   + group_end_3 <= 80
 constr_link_match_2_3_28: 80 switch_nic_out_3_nic_in_28 + group_start_28
   - group_end_3 <= 80
 constr_link_match_1_3_29: 80 switch_nic_out_3_nic_in_29 - group_start_29
   + group_end_3 <= 80
 constr_link_match_2_3_29: 80 switch_nic_out_3_nic_in_29 + group_start_29
   - group_end_3 <= 80
 constr_link_match_1_3_30: 80 switch_nic_out_3_nic_in_30 - group_start_30
   + group_end_3 <= 80
 constr_link_match_2_3_30: 80 switch_nic_out_3_nic_in_30 + group_start_30
   - group_end_3 <= 80
 constr_link_match_1_3_31: 80 switch_nic_out_3_nic_in_31 - group_start_31
   + group_end_3 <= 80
 constr_link_match_2_3_31: 80 switch_nic_out_3_nic_in_31 + group_start_31
   - group_end_3 <= 80
 constr_is_start_zero_4: start_4 + 80 is_start_zero_4_for_source_nic_in_4
   <= 80
 constr_edge_switch_start_zero_4_OR: switch_source_nic_in_4
   - is_start_zero_4_for_source_nic_in_4 <= 0
 constr_is_end_complete_4: start_4 + hold_4_1 + 2 hold_4_2 + 3 hold_4_3
   + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 - 80 is_end_complete_4 >= 0
 constr_edge_switch_end_max_4_OR: switch_nic_out_4_sink - is_end_complete_4
   <= 0
 constr_link_match_1_4_0: 80 switch_nic_out_4_nic_in_0 - group_start_0
   + group_end_4 <= 80
 constr_link_match_2_4_0: 80 switch_nic_out_4_nic_in_0 + group_start_0
   - group_end_4 <= 80
 constr_link_match_1_4_1: 80 switch_nic_out_4_nic_in_1 - group_start_1
   + group_end_4 <= 80
 constr_link_match_2_4_1: 80 switch_nic_out_4_nic_in_1 + group_start_1
   - group_end_4 <= 80
 constr_link_match_1_4_2: 80 switch_nic_out_4_nic_in_2 - group_start_2
   + group_end_4 <= 80
 constr_link_match_2_4_2: 80 switch_nic_out_4_nic_in_2 + group_start_2
   - group_end_4 <= 80
 constr_link_match_1_4_3: 80 switch_nic_out_4_nic_in_3 - group_start_3
   + group_end_4 <= 80
 constr_link_match_2_4_3: 80 switch_nic_out_4_nic_in_3 + group_start_3
   - group_end_4 <= 80
 constr_link_match_1_4_5: 80 switch_nic_out_4_nic_in_5 - group_start_5
   + group_end_4 <= 80
 constr_link_match_2_4_5: 80 switch_nic_out_4_nic_in_5 + group_start_5
   - group_end_4 <= 80
 constr_link_match_1_4_6: 80 switch_nic_out_4_nic_in_6 - group_start_6
   + group_end_4 <= 80
 constr_link_match_2_4_6: 80 switch_nic_out_4_nic_in_6 + group_start_6
   - group_end_4 <= 80
 constr_link_match_1_4_7: 80 switch_nic_out_4_nic_in_7 - group_start_7
   + group_end_4 <= 80
 constr_link_match_2_4_7: 80 switch_nic_out_4_nic_in_7 + group_start_7
   - group_end_4 <= 80
 constr_link_match_1_4_8: 80 switch_nic_out_4_nic_in_8 - group_start_8
   + group_end_4 <= 80
 constr_link_match_2_4_8: 80 switch_nic_out_4_nic_in_8 + group_start_8
   - group_end_4 <= 80
 constr_link_match_1_4_9: 80 switch_nic_out_4_nic_in_9 - group_start_9
   + group_end_4 <= 80
 constr_link_match_2_4_9: 80 switch_nic_out_4_nic_in_9 + group_start_9
   - group_end_4 <= 80
 constr_link_match_1_4_10: 80 switch_nic_out_4_nic_in_10 - group_start_10
   + group_end_4 <= 80
 constr_link_match_2_4_10: 80 switch_nic_out_4_nic_in_10 + group_start_10
   - group_end_4 <= 80
 constr_link_match_1_4_11: 80 switch_nic_out_4_nic_in_11 - group_start_11
   + group_end_4 <= 80
 constr_link_match_2_4_11: 80 switch_nic_out_4_nic_in_11 + group_start_11
   - group_end_4 <= 80
 constr_link_match_1_4_12: 80 switch_nic_out_4_nic_in_12 - group_start_12
   + group_end_4 <= 80
 constr_link_match_2_4_12: 80 switch_nic_out_4_nic_in_12 + group_start_12
   - group_end_4 <= 80
 constr_link_match_1_4_13: 80 switch_nic_out_4_nic_in_13 - group_start_13
   + group_end_4 <= 80
 constr_link_match_2_4_13: 80 switch_nic_out_4_nic_in_13 + group_start_13
   - group_end_4 <= 80
 constr_link_match_1_4_14: 80 switch_nic_out_4_nic_in_14 - group_start_14
   + group_end_4 <= 80
 constr_link_match_2_4_14: 80 switch_nic_out_4_nic_in_14 + group_start_14
   - group_end_4 <= 80
 constr_link_match_1_4_15: 80 switch_nic_out_4_nic_in_15 - group_start_15
   + group_end_4 <= 80
 constr_link_match_2_4_15: 80 switch_nic_out_4_nic_in_15 + group_start_15
   - group_end_4 <= 80
 constr_link_match_1_4_16: 80 switch_nic_out_4_nic_in_16 - group_start_16
   + group_end_4 <= 80
 constr_link_match_2_4_16: 80 switch_nic_out_4_nic_in_16 + group_start_16
   - group_end_4 <= 80
 constr_link_match_1_4_17: 80 switch_nic_out_4_nic_in_17 - group_start_17
   + group_end_4 <= 80
 constr_link_match_2_4_17: 80 switch_nic_out_4_nic_in_17 + group_start_17
   - group_end_4 <= 80
 constr_link_match_1_4_18: 80 switch_nic_out_4_nic_in_18 - group_start_18
   + group_end_4 <= 80
 constr_link_match_2_4_18: 80 switch_nic_out_4_nic_in_18 + group_start_18
   - group_end_4 <= 80
 constr_link_match_1_4_19: 80 switch_nic_out_4_nic_in_19 - group_start_19
   + group_end_4 <= 80
 constr_link_match_2_4_19: 80 switch_nic_out_4_nic_in_19 + group_start_19
   - group_end_4 <= 80
 constr_link_match_1_4_20: 80 switch_nic_out_4_nic_in_20 - group_start_20
   + group_end_4 <= 80
 constr_link_match_2_4_20: 80 switch_nic_out_4_nic_in_20 + group_start_20
   - group_end_4 <= 80
 constr_link_match_1_4_21: 80 switch_nic_out_4_nic_in_21 - group_start_21
   + group_end_4 <= 80
 constr_link_match_2_4_21: 80 switch_nic_out_4_nic_in_21 + group_start_21
   - group_end_4 <= 80
 constr_link_match_1_4_22: 80 switch_nic_out_4_nic_in_22 - group_start_22
   + group_end_4 <= 80
 constr_link_match_2_4_22: 80 switch_nic_out_4_nic_in_22 + group_start_22
   - group_end_4 <= 80
 constr_link_match_1_4_23: 80 switch_nic_out_4_nic_in_23 - group_start_23
   + group_end_4 <= 80
 constr_link_match_2_4_23: 80 switch_nic_out_4_nic_in_23 + group_start_23
   - group_end_4 <= 80
 constr_link_match_1_4_24: 80 switch_nic_out_4_nic_in_24 - group_start_24
   + group_end_4 <= 80
 constr_link_match_2_4_24: 80 switch_nic_out_4_nic_in_24 + group_start_24
   - group_end_4 <= 80
 constr_link_match_1_4_25: 80 switch_nic_out_4_nic_in_25 - group_start_25
   + group_end_4 <= 80
 constr_link_match_2_4_25: 80 switch_nic_out_4_nic_in_25 + group_start_25
   - group_end_4 <= 80
 constr_link_match_1_4_26: 80 switch_nic_out_4_nic_in_26 - group_start_26
   + group_end_4 <= 80
 constr_link_match_2_4_26: 80 switch_nic_out_4_nic_in_26 + group_start_26
   - group_end_4 <= 80
 constr_link_match_1_4_27: 80 switch_nic_out_4_nic_in_27 - group_start_27
   + group_end_4 <= 80
 constr_link_match_2_4_27: 80 switch_nic_out_4_nic_in_27 + group_start_27
   - group_end_4 <= 80
 constr_link_match_1_4_28: 80 switch_nic_out_4_nic_in_28 - group_start_28
   + group_end_4 <= 80
 constr_link_match_2_4_28: 80 switch_nic_out_4_nic_in_28 + group_start_28
   - group_end_4 <= 80
 constr_link_match_1_4_29: 80 switch_nic_out_4_nic_in_29 - group_start_29
   + group_end_4 <= 80
 constr_link_match_2_4_29: 80 switch_nic_out_4_nic_in_29 + group_start_29
   - group_end_4 <= 80
 constr_link_match_1_4_30: 80 switch_nic_out_4_nic_in_30 - group_start_30
   + group_end_4 <= 80
 constr_link_match_2_4_30: 80 switch_nic_out_4_nic_in_30 + group_start_30
   - group_end_4 <= 80
 constr_link_match_1_4_31: 80 switch_nic_out_4_nic_in_31 - group_start_31
   + group_end_4 <= 80
 constr_link_match_2_4_31: 80 switch_nic_out_4_nic_in_31 + group_start_31
   - group_end_4 <= 80
 constr_is_start_zero_5: start_5 + 80 is_start_zero_5_for_source_nic_in_5
   <= 80
 constr_edge_switch_start_zero_5_OR: switch_source_nic_in_5
   - is_start_zero_5_for_source_nic_in_5 <= 0
 constr_is_end_complete_5: start_5 + hold_5_1 + 2 hold_5_2 + 3 hold_5_3
   + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 - 80 is_end_complete_5 >= 0
 constr_edge_switch_end_max_5_OR: switch_nic_out_5_sink - is_end_complete_5
   <= 0
 constr_link_match_1_5_0: 80 switch_nic_out_5_nic_in_0 - group_start_0
   + group_end_5 <= 80
 constr_link_match_2_5_0: 80 switch_nic_out_5_nic_in_0 + group_start_0
   - group_end_5 <= 80
 constr_link_match_1_5_1: 80 switch_nic_out_5_nic_in_1 - group_start_1
   + group_end_5 <= 80
 constr_link_match_2_5_1: 80 switch_nic_out_5_nic_in_1 + group_start_1
   - group_end_5 <= 80
 constr_link_match_1_5_2: 80 switch_nic_out_5_nic_in_2 - group_start_2
   + group_end_5 <= 80
 constr_link_match_2_5_2: 80 switch_nic_out_5_nic_in_2 + group_start_2
   - group_end_5 <= 80
 constr_link_match_1_5_3: 80 switch_nic_out_5_nic_in_3 - group_start_3
   + group_end_5 <= 80
 constr_link_match_2_5_3: 80 switch_nic_out_5_nic_in_3 + group_start_3
   - group_end_5 <= 80
 constr_link_match_1_5_4: 80 switch_nic_out_5_nic_in_4 - group_start_4
   + group_end_5 <= 80
 constr_link_match_2_5_4: 80 switch_nic_out_5_nic_in_4 + group_start_4
   - group_end_5 <= 80
 constr_link_match_1_5_6: 80 switch_nic_out_5_nic_in_6 - group_start_6
   + group_end_5 <= 80
 constr_link_match_2_5_6: 80 switch_nic_out_5_nic_in_6 + group_start_6
   - group_end_5 <= 80
 constr_link_match_1_5_7: 80 switch_nic_out_5_nic_in_7 - group_start_7
   + group_end_5 <= 80
 constr_link_match_2_5_7: 80 switch_nic_out_5_nic_in_7 + group_start_7
   - group_end_5 <= 80
 constr_link_match_1_5_8: 80 switch_nic_out_5_nic_in_8 - group_start_8
   + group_end_5 <= 80
 constr_link_match_2_5_8: 80 switch_nic_out_5_nic_in_8 + group_start_8
   - group_end_5 <= 80
 constr_link_match_1_5_9: 80 switch_nic_out_5_nic_in_9 - group_start_9
   + group_end_5 <= 80
 constr_link_match_2_5_9: 80 switch_nic_out_5_nic_in_9 + group_start_9
   - group_end_5 <= 80
 constr_link_match_1_5_10: 80 switch_nic_out_5_nic_in_10 - group_start_10
   + group_end_5 <= 80
 constr_link_match_2_5_10: 80 switch_nic_out_5_nic_in_10 + group_start_10
   - group_end_5 <= 80
 constr_link_match_1_5_11: 80 switch_nic_out_5_nic_in_11 - group_start_11
   + group_end_5 <= 80
 constr_link_match_2_5_11: 80 switch_nic_out_5_nic_in_11 + group_start_11
   - group_end_5 <= 80
 constr_link_match_1_5_12: 80 switch_nic_out_5_nic_in_12 - group_start_12
   + group_end_5 <= 80
 constr_link_match_2_5_12: 80 switch_nic_out_5_nic_in_12 + group_start_12
   - group_end_5 <= 80
 constr_link_match_1_5_13: 80 switch_nic_out_5_nic_in_13 - group_start_13
   + group_end_5 <= 80
 constr_link_match_2_5_13: 80 switch_nic_out_5_nic_in_13 + group_start_13
   - group_end_5 <= 80
 constr_link_match_1_5_14: 80 switch_nic_out_5_nic_in_14 - group_start_14
   + group_end_5 <= 80
 constr_link_match_2_5_14: 80 switch_nic_out_5_nic_in_14 + group_start_14
   - group_end_5 <= 80
 constr_link_match_1_5_15: 80 switch_nic_out_5_nic_in_15 - group_start_15
   + group_end_5 <= 80
 constr_link_match_2_5_15: 80 switch_nic_out_5_nic_in_15 + group_start_15
   - group_end_5 <= 80
 constr_link_match_1_5_16: 80 switch_nic_out_5_nic_in_16 - group_start_16
   + group_end_5 <= 80
 constr_link_match_2_5_16: 80 switch_nic_out_5_nic_in_16 + group_start_16
   - group_end_5 <= 80
 constr_link_match_1_5_17: 80 switch_nic_out_5_nic_in_17 - group_start_17
   + group_end_5 <= 80
 constr_link_match_2_5_17: 80 switch_nic_out_5_nic_in_17 + group_start_17
   - group_end_5 <= 80
 constr_link_match_1_5_18: 80 switch_nic_out_5_nic_in_18 - group_start_18
   + group_end_5 <= 80
 constr_link_match_2_5_18: 80 switch_nic_out_5_nic_in_18 + group_start_18
   - group_end_5 <= 80
 constr_link_match_1_5_19: 80 switch_nic_out_5_nic_in_19 - group_start_19
   + group_end_5 <= 80
 constr_link_match_2_5_19: 80 switch_nic_out_5_nic_in_19 + group_start_19
   - group_end_5 <= 80
 constr_link_match_1_5_20: 80 switch_nic_out_5_nic_in_20 - group_start_20
   + group_end_5 <= 80
 constr_link_match_2_5_20: 80 switch_nic_out_5_nic_in_20 + group_start_20
   - group_end_5 <= 80
 constr_link_match_1_5_21: 80 switch_nic_out_5_nic_in_21 - group_start_21
   + group_end_5 <= 80
 constr_link_match_2_5_21: 80 switch_nic_out_5_nic_in_21 + group_start_21
   - group_end_5 <= 80
 constr_link_match_1_5_22: 80 switch_nic_out_5_nic_in_22 - group_start_22
   + group_end_5 <= 80
 constr_link_match_2_5_22: 80 switch_nic_out_5_nic_in_22 + group_start_22
   - group_end_5 <= 80
 constr_link_match_1_5_23: 80 switch_nic_out_5_nic_in_23 - group_start_23
   + group_end_5 <= 80
 constr_link_match_2_5_23: 80 switch_nic_out_5_nic_in_23 + group_start_23
   - group_end_5 <= 80
 constr_link_match_1_5_24: 80 switch_nic_out_5_nic_in_24 - group_start_24
   + group_end_5 <= 80
 constr_link_match_2_5_24: 80 switch_nic_out_5_nic_in_24 + group_start_24
   - group_end_5 <= 80
 constr_link_match_1_5_25: 80 switch_nic_out_5_nic_in_25 - group_start_25
   + group_end_5 <= 80
 constr_link_match_2_5_25: 80 switch_nic_out_5_nic_in_25 + group_start_25
   - group_end_5 <= 80
 constr_link_match_1_5_26: 80 switch_nic_out_5_nic_in_26 - group_start_26
   + group_end_5 <= 80
 constr_link_match_2_5_26: 80 switch_nic_out_5_nic_in_26 + group_start_26
   - group_end_5 <= 80
 constr_link_match_1_5_27: 80 switch_nic_out_5_nic_in_27 - group_start_27
   + group_end_5 <= 80
 constr_link_match_2_5_27: 80 switch_nic_out_5_nic_in_27 + group_start_27
   - group_end_5 <= 80
 constr_link_match_1_5_28: 80 switch_nic_out_5_nic_in_28 - group_start_28
   + group_end_5 <= 80
 constr_link_match_2_5_28: 80 switch_nic_out_5_nic_in_28 + group_start_28
   - group_end_5 <= 80
 constr_link_match_1_5_29: 80 switch_nic_out_5_nic_in_29 - group_start_29
   + group_end_5 <= 80
 constr_link_match_2_5_29: 80 switch_nic_out_5_nic_in_29 + group_start_29
   - group_end_5 <= 80
 constr_link_match_1_5_30: 80 switch_nic_out_5_nic_in_30 - group_start_30
   + group_end_5 <= 80
 constr_link_match_2_5_30: 80 switch_nic_out_5_nic_in_30 + group_start_30
   - group_end_5 <= 80
 constr_link_match_1_5_31: 80 switch_nic_out_5_nic_in_31 - group_start_31
   + group_end_5 <= 80
 constr_link_match_2_5_31: 80 switch_nic_out_5_nic_in_31 + group_start_31
   - group_end_5 <= 80
 constr_is_start_zero_6: start_6 + 80 is_start_zero_6_for_source_nic_in_6
   <= 80
 constr_edge_switch_start_zero_6_OR: switch_source_nic_in_6
   - is_start_zero_6_for_source_nic_in_6 <= 0
 constr_is_end_complete_6: start_6 + hold_6_1 + 2 hold_6_2 + 3 hold_6_3
   + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 - 80 is_end_complete_6 >= 0
 constr_edge_switch_end_max_6_OR: switch_nic_out_6_sink - is_end_complete_6
   <= 0
 constr_link_match_1_6_0: 80 switch_nic_out_6_nic_in_0 - group_start_0
   + group_end_6 <= 80
 constr_link_match_2_6_0: 80 switch_nic_out_6_nic_in_0 + group_start_0
   - group_end_6 <= 80
 constr_link_match_1_6_1: 80 switch_nic_out_6_nic_in_1 - group_start_1
   + group_end_6 <= 80
 constr_link_match_2_6_1: 80 switch_nic_out_6_nic_in_1 + group_start_1
   - group_end_6 <= 80
 constr_link_match_1_6_2: 80 switch_nic_out_6_nic_in_2 - group_start_2
   + group_end_6 <= 80
 constr_link_match_2_6_2: 80 switch_nic_out_6_nic_in_2 + group_start_2
   - group_end_6 <= 80
 constr_link_match_1_6_3: 80 switch_nic_out_6_nic_in_3 - group_start_3
   + group_end_6 <= 80
 constr_link_match_2_6_3: 80 switch_nic_out_6_nic_in_3 + group_start_3
   - group_end_6 <= 80
 constr_link_match_1_6_4: 80 switch_nic_out_6_nic_in_4 - group_start_4
   + group_end_6 <= 80
 constr_link_match_2_6_4: 80 switch_nic_out_6_nic_in_4 + group_start_4
   - group_end_6 <= 80
 constr_link_match_1_6_5: 80 switch_nic_out_6_nic_in_5 - group_start_5
   + group_end_6 <= 80
 constr_link_match_2_6_5: 80 switch_nic_out_6_nic_in_5 + group_start_5
   - group_end_6 <= 80
 constr_link_match_1_6_7: 80 switch_nic_out_6_nic_in_7 - group_start_7
   + group_end_6 <= 80
 constr_link_match_2_6_7: 80 switch_nic_out_6_nic_in_7 + group_start_7
   - group_end_6 <= 80
 constr_link_match_1_6_8: 80 switch_nic_out_6_nic_in_8 - group_start_8
   + group_end_6 <= 80
 constr_link_match_2_6_8: 80 switch_nic_out_6_nic_in_8 + group_start_8
   - group_end_6 <= 80
 constr_link_match_1_6_9: 80 switch_nic_out_6_nic_in_9 - group_start_9
   + group_end_6 <= 80
 constr_link_match_2_6_9: 80 switch_nic_out_6_nic_in_9 + group_start_9
   - group_end_6 <= 80
 constr_link_match_1_6_10: 80 switch_nic_out_6_nic_in_10 - group_start_10
   + group_end_6 <= 80
 constr_link_match_2_6_10: 80 switch_nic_out_6_nic_in_10 + group_start_10
   - group_end_6 <= 80
 constr_link_match_1_6_11: 80 switch_nic_out_6_nic_in_11 - group_start_11
   + group_end_6 <= 80
 constr_link_match_2_6_11: 80 switch_nic_out_6_nic_in_11 + group_start_11
   - group_end_6 <= 80
 constr_link_match_1_6_12: 80 switch_nic_out_6_nic_in_12 - group_start_12
   + group_end_6 <= 80
 constr_link_match_2_6_12: 80 switch_nic_out_6_nic_in_12 + group_start_12
   - group_end_6 <= 80
 constr_link_match_1_6_13: 80 switch_nic_out_6_nic_in_13 - group_start_13
   + group_end_6 <= 80
 constr_link_match_2_6_13: 80 switch_nic_out_6_nic_in_13 + group_start_13
   - group_end_6 <= 80
 constr_link_match_1_6_14: 80 switch_nic_out_6_nic_in_14 - group_start_14
   + group_end_6 <= 80
 constr_link_match_2_6_14: 80 switch_nic_out_6_nic_in_14 + group_start_14
   - group_end_6 <= 80
 constr_link_match_1_6_15: 80 switch_nic_out_6_nic_in_15 - group_start_15
   + group_end_6 <= 80
 constr_link_match_2_6_15: 80 switch_nic_out_6_nic_in_15 + group_start_15
   - group_end_6 <= 80
 constr_link_match_1_6_16: 80 switch_nic_out_6_nic_in_16 - group_start_16
   + group_end_6 <= 80
 constr_link_match_2_6_16: 80 switch_nic_out_6_nic_in_16 + group_start_16
   - group_end_6 <= 80
 constr_link_match_1_6_17: 80 switch_nic_out_6_nic_in_17 - group_start_17
   + group_end_6 <= 80
 constr_link_match_2_6_17: 80 switch_nic_out_6_nic_in_17 + group_start_17
   - group_end_6 <= 80
 constr_link_match_1_6_18: 80 switch_nic_out_6_nic_in_18 - group_start_18
   + group_end_6 <= 80
 constr_link_match_2_6_18: 80 switch_nic_out_6_nic_in_18 + group_start_18
   - group_end_6 <= 80
 constr_link_match_1_6_19: 80 switch_nic_out_6_nic_in_19 - group_start_19
   + group_end_6 <= 80
 constr_link_match_2_6_19: 80 switch_nic_out_6_nic_in_19 + group_start_19
   - group_end_6 <= 80
 constr_link_match_1_6_20: 80 switch_nic_out_6_nic_in_20 - group_start_20
   + group_end_6 <= 80
 constr_link_match_2_6_20: 80 switch_nic_out_6_nic_in_20 + group_start_20
   - group_end_6 <= 80
 constr_link_match_1_6_21: 80 switch_nic_out_6_nic_in_21 - group_start_21
   + group_end_6 <= 80
 constr_link_match_2_6_21: 80 switch_nic_out_6_nic_in_21 + group_start_21
   - group_end_6 <= 80
 constr_link_match_1_6_22: 80 switch_nic_out_6_nic_in_22 - group_start_22
   + group_end_6 <= 80
 constr_link_match_2_6_22: 80 switch_nic_out_6_nic_in_22 + group_start_22
   - group_end_6 <= 80
 constr_link_match_1_6_23: 80 switch_nic_out_6_nic_in_23 - group_start_23
   + group_end_6 <= 80
 constr_link_match_2_6_23: 80 switch_nic_out_6_nic_in_23 + group_start_23
   - group_end_6 <= 80
 constr_link_match_1_6_24: 80 switch_nic_out_6_nic_in_24 - group_start_24
   + group_end_6 <= 80
 constr_link_match_2_6_24: 80 switch_nic_out_6_nic_in_24 + group_start_24
   - group_end_6 <= 80
 constr_link_match_1_6_25: 80 switch_nic_out_6_nic_in_25 - group_start_25
   + group_end_6 <= 80
 constr_link_match_2_6_25: 80 switch_nic_out_6_nic_in_25 + group_start_25
   - group_end_6 <= 80
 constr_link_match_1_6_26: 80 switch_nic_out_6_nic_in_26 - group_start_26
   + group_end_6 <= 80
 constr_link_match_2_6_26: 80 switch_nic_out_6_nic_in_26 + group_start_26
   - group_end_6 <= 80
 constr_link_match_1_6_27: 80 switch_nic_out_6_nic_in_27 - group_start_27
   + group_end_6 <= 80
 constr_link_match_2_6_27: 80 switch_nic_out_6_nic_in_27 + group_start_27
   - group_end_6 <= 80
 constr_link_match_1_6_28: 80 switch_nic_out_6_nic_in_28 - group_start_28
   + group_end_6 <= 80
 constr_link_match_2_6_28: 80 switch_nic_out_6_nic_in_28 + group_start_28
   - group_end_6 <= 80
 constr_link_match_1_6_29: 80 switch_nic_out_6_nic_in_29 - group_start_29
   + group_end_6 <= 80
 constr_link_match_2_6_29: 80 switch_nic_out_6_nic_in_29 + group_start_29
   - group_end_6 <= 80
 constr_link_match_1_6_30: 80 switch_nic_out_6_nic_in_30 - group_start_30
   + group_end_6 <= 80
 constr_link_match_2_6_30: 80 switch_nic_out_6_nic_in_30 + group_start_30
   - group_end_6 <= 80
 constr_link_match_1_6_31: 80 switch_nic_out_6_nic_in_31 - group_start_31
   + group_end_6 <= 80
 constr_link_match_2_6_31: 80 switch_nic_out_6_nic_in_31 + group_start_31
   - group_end_6 <= 80
 constr_is_start_zero_7: start_7 + 80 is_start_zero_7_for_source_nic_in_7
   <= 80
 constr_edge_switch_start_zero_7_OR: switch_source_nic_in_7
   - is_start_zero_7_for_source_nic_in_7 <= 0
 constr_is_end_complete_7: start_7 + hold_7_1 + 2 hold_7_2 + 3 hold_7_3
   + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 - 80 is_end_complete_7 >= 0
 constr_edge_switch_end_max_7_OR: switch_nic_out_7_sink - is_end_complete_7
   <= 0
 constr_link_match_1_7_0: 80 switch_nic_out_7_nic_in_0 - group_start_0
   + group_end_7 <= 80
 constr_link_match_2_7_0: 80 switch_nic_out_7_nic_in_0 + group_start_0
   - group_end_7 <= 80
 constr_link_match_1_7_1: 80 switch_nic_out_7_nic_in_1 - group_start_1
   + group_end_7 <= 80
 constr_link_match_2_7_1: 80 switch_nic_out_7_nic_in_1 + group_start_1
   - group_end_7 <= 80
 constr_link_match_1_7_2: 80 switch_nic_out_7_nic_in_2 - group_start_2
   + group_end_7 <= 80
 constr_link_match_2_7_2: 80 switch_nic_out_7_nic_in_2 + group_start_2
   - group_end_7 <= 80
 constr_link_match_1_7_3: 80 switch_nic_out_7_nic_in_3 - group_start_3
   + group_end_7 <= 80
 constr_link_match_2_7_3: 80 switch_nic_out_7_nic_in_3 + group_start_3
   - group_end_7 <= 80
 constr_link_match_1_7_4: 80 switch_nic_out_7_nic_in_4 - group_start_4
   + group_end_7 <= 80
 constr_link_match_2_7_4: 80 switch_nic_out_7_nic_in_4 + group_start_4
   - group_end_7 <= 80
 constr_link_match_1_7_5: 80 switch_nic_out_7_nic_in_5 - group_start_5
   + group_end_7 <= 80
 constr_link_match_2_7_5: 80 switch_nic_out_7_nic_in_5 + group_start_5
   - group_end_7 <= 80
 constr_link_match_1_7_6: 80 switch_nic_out_7_nic_in_6 - group_start_6
   + group_end_7 <= 80
 constr_link_match_2_7_6: 80 switch_nic_out_7_nic_in_6 + group_start_6
   - group_end_7 <= 80
 constr_link_match_1_7_8: 80 switch_nic_out_7_nic_in_8 - group_start_8
   + group_end_7 <= 80
 constr_link_match_2_7_8: 80 switch_nic_out_7_nic_in_8 + group_start_8
   - group_end_7 <= 80
 constr_link_match_1_7_9: 80 switch_nic_out_7_nic_in_9 - group_start_9
   + group_end_7 <= 80
 constr_link_match_2_7_9: 80 switch_nic_out_7_nic_in_9 + group_start_9
   - group_end_7 <= 80
 constr_link_match_1_7_10: 80 switch_nic_out_7_nic_in_10 - group_start_10
   + group_end_7 <= 80
 constr_link_match_2_7_10: 80 switch_nic_out_7_nic_in_10 + group_start_10
   - group_end_7 <= 80
 constr_link_match_1_7_11: 80 switch_nic_out_7_nic_in_11 - group_start_11
   + group_end_7 <= 80
 constr_link_match_2_7_11: 80 switch_nic_out_7_nic_in_11 + group_start_11
   - group_end_7 <= 80
 constr_link_match_1_7_12: 80 switch_nic_out_7_nic_in_12 - group_start_12
   + group_end_7 <= 80
 constr_link_match_2_7_12: 80 switch_nic_out_7_nic_in_12 + group_start_12
   - group_end_7 <= 80
 constr_link_match_1_7_13: 80 switch_nic_out_7_nic_in_13 - group_start_13
   + group_end_7 <= 80
 constr_link_match_2_7_13: 80 switch_nic_out_7_nic_in_13 + group_start_13
   - group_end_7 <= 80
 constr_link_match_1_7_14: 80 switch_nic_out_7_nic_in_14 - group_start_14
   + group_end_7 <= 80
 constr_link_match_2_7_14: 80 switch_nic_out_7_nic_in_14 + group_start_14
   - group_end_7 <= 80
 constr_link_match_1_7_15: 80 switch_nic_out_7_nic_in_15 - group_start_15
   + group_end_7 <= 80
 constr_link_match_2_7_15: 80 switch_nic_out_7_nic_in_15 + group_start_15
   - group_end_7 <= 80
 constr_link_match_1_7_16: 80 switch_nic_out_7_nic_in_16 - group_start_16
   + group_end_7 <= 80
 constr_link_match_2_7_16: 80 switch_nic_out_7_nic_in_16 + group_start_16
   - group_end_7 <= 80
 constr_link_match_1_7_17: 80 switch_nic_out_7_nic_in_17 - group_start_17
   + group_end_7 <= 80
 constr_link_match_2_7_17: 80 switch_nic_out_7_nic_in_17 + group_start_17
   - group_end_7 <= 80
 constr_link_match_1_7_18: 80 switch_nic_out_7_nic_in_18 - group_start_18
   + group_end_7 <= 80
 constr_link_match_2_7_18: 80 switch_nic_out_7_nic_in_18 + group_start_18
   - group_end_7 <= 80
 constr_link_match_1_7_19: 80 switch_nic_out_7_nic_in_19 - group_start_19
   + group_end_7 <= 80
 constr_link_match_2_7_19: 80 switch_nic_out_7_nic_in_19 + group_start_19
   - group_end_7 <= 80
 constr_link_match_1_7_20: 80 switch_nic_out_7_nic_in_20 - group_start_20
   + group_end_7 <= 80
 constr_link_match_2_7_20: 80 switch_nic_out_7_nic_in_20 + group_start_20
   - group_end_7 <= 80
 constr_link_match_1_7_21: 80 switch_nic_out_7_nic_in_21 - group_start_21
   + group_end_7 <= 80
 constr_link_match_2_7_21: 80 switch_nic_out_7_nic_in_21 + group_start_21
   - group_end_7 <= 80
 constr_link_match_1_7_22: 80 switch_nic_out_7_nic_in_22 - group_start_22
   + group_end_7 <= 80
 constr_link_match_2_7_22: 80 switch_nic_out_7_nic_in_22 + group_start_22
   - group_end_7 <= 80
 constr_link_match_1_7_23: 80 switch_nic_out_7_nic_in_23 - group_start_23
   + group_end_7 <= 80
 constr_link_match_2_7_23: 80 switch_nic_out_7_nic_in_23 + group_start_23
   - group_end_7 <= 80
 constr_link_match_1_7_24: 80 switch_nic_out_7_nic_in_24 - group_start_24
   + group_end_7 <= 80
 constr_link_match_2_7_24: 80 switch_nic_out_7_nic_in_24 + group_start_24
   - group_end_7 <= 80
 constr_link_match_1_7_25: 80 switch_nic_out_7_nic_in_25 - group_start_25
   + group_end_7 <= 80
 constr_link_match_2_7_25: 80 switch_nic_out_7_nic_in_25 + group_start_25
   - group_end_7 <= 80
 constr_link_match_1_7_26: 80 switch_nic_out_7_nic_in_26 - group_start_26
   + group_end_7 <= 80
 constr_link_match_2_7_26: 80 switch_nic_out_7_nic_in_26 + group_start_26
   - group_end_7 <= 80
 constr_link_match_1_7_27: 80 switch_nic_out_7_nic_in_27 - group_start_27
   + group_end_7 <= 80
 constr_link_match_2_7_27: 80 switch_nic_out_7_nic_in_27 + group_start_27
   - group_end_7 <= 80
 constr_link_match_1_7_28: 80 switch_nic_out_7_nic_in_28 - group_start_28
   + group_end_7 <= 80
 constr_link_match_2_7_28: 80 switch_nic_out_7_nic_in_28 + group_start_28
   - group_end_7 <= 80
 constr_link_match_1_7_29: 80 switch_nic_out_7_nic_in_29 - group_start_29
   + group_end_7 <= 80
 constr_link_match_2_7_29: 80 switch_nic_out_7_nic_in_29 + group_start_29
   - group_end_7 <= 80
 constr_link_match_1_7_30: 80 switch_nic_out_7_nic_in_30 - group_start_30
   + group_end_7 <= 80
 constr_link_match_2_7_30: 80 switch_nic_out_7_nic_in_30 + group_start_30
   - group_end_7 <= 80
 constr_link_match_1_7_31: 80 switch_nic_out_7_nic_in_31 - group_start_31
   + group_end_7 <= 80
 constr_link_match_2_7_31: 80 switch_nic_out_7_nic_in_31 + group_start_31
   - group_end_7 <= 80
 constr_is_start_zero_8: start_8 + 80 is_start_zero_8_for_source_nic_in_8
   <= 80
 constr_edge_switch_start_zero_8_OR: switch_source_nic_in_8
   - is_start_zero_8_for_source_nic_in_8 <= 0
 constr_is_end_complete_8: start_8 + hold_8_1 + 2 hold_8_2 + 3 hold_8_3
   + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 - 80 is_end_complete_8 >= 0
 constr_edge_switch_end_max_8_OR: switch_nic_out_8_sink - is_end_complete_8
   <= 0
 constr_link_match_1_8_0: 80 switch_nic_out_8_nic_in_0 - group_start_0
   + group_end_8 <= 80
 constr_link_match_2_8_0: 80 switch_nic_out_8_nic_in_0 + group_start_0
   - group_end_8 <= 80
 constr_link_match_1_8_1: 80 switch_nic_out_8_nic_in_1 - group_start_1
   + group_end_8 <= 80
 constr_link_match_2_8_1: 80 switch_nic_out_8_nic_in_1 + group_start_1
   - group_end_8 <= 80
 constr_link_match_1_8_2: 80 switch_nic_out_8_nic_in_2 - group_start_2
   + group_end_8 <= 80
 constr_link_match_2_8_2: 80 switch_nic_out_8_nic_in_2 + group_start_2
   - group_end_8 <= 80
 constr_link_match_1_8_3: 80 switch_nic_out_8_nic_in_3 - group_start_3
   + group_end_8 <= 80
 constr_link_match_2_8_3: 80 switch_nic_out_8_nic_in_3 + group_start_3
   - group_end_8 <= 80
 constr_link_match_1_8_4: 80 switch_nic_out_8_nic_in_4 - group_start_4
   + group_end_8 <= 80
 constr_link_match_2_8_4: 80 switch_nic_out_8_nic_in_4 + group_start_4
   - group_end_8 <= 80
 constr_link_match_1_8_5: 80 switch_nic_out_8_nic_in_5 - group_start_5
   + group_end_8 <= 80
 constr_link_match_2_8_5: 80 switch_nic_out_8_nic_in_5 + group_start_5
   - group_end_8 <= 80
 constr_link_match_1_8_6: 80 switch_nic_out_8_nic_in_6 - group_start_6
   + group_end_8 <= 80
 constr_link_match_2_8_6: 80 switch_nic_out_8_nic_in_6 + group_start_6
   - group_end_8 <= 80
 constr_link_match_1_8_7: 80 switch_nic_out_8_nic_in_7 - group_start_7
   + group_end_8 <= 80
 constr_link_match_2_8_7: 80 switch_nic_out_8_nic_in_7 + group_start_7
   - group_end_8 <= 80
 constr_link_match_1_8_9: 80 switch_nic_out_8_nic_in_9 - group_start_9
   + group_end_8 <= 80
 constr_link_match_2_8_9: 80 switch_nic_out_8_nic_in_9 + group_start_9
   - group_end_8 <= 80
 constr_link_match_1_8_10: 80 switch_nic_out_8_nic_in_10 - group_start_10
   + group_end_8 <= 80
 constr_link_match_2_8_10: 80 switch_nic_out_8_nic_in_10 + group_start_10
   - group_end_8 <= 80
 constr_link_match_1_8_11: 80 switch_nic_out_8_nic_in_11 - group_start_11
   + group_end_8 <= 80
 constr_link_match_2_8_11: 80 switch_nic_out_8_nic_in_11 + group_start_11
   - group_end_8 <= 80
 constr_link_match_1_8_12: 80 switch_nic_out_8_nic_in_12 - group_start_12
   + group_end_8 <= 80
 constr_link_match_2_8_12: 80 switch_nic_out_8_nic_in_12 + group_start_12
   - group_end_8 <= 80
 constr_link_match_1_8_13: 80 switch_nic_out_8_nic_in_13 - group_start_13
   + group_end_8 <= 80
 constr_link_match_2_8_13: 80 switch_nic_out_8_nic_in_13 + group_start_13
   - group_end_8 <= 80
 constr_link_match_1_8_14: 80 switch_nic_out_8_nic_in_14 - group_start_14
   + group_end_8 <= 80
 constr_link_match_2_8_14: 80 switch_nic_out_8_nic_in_14 + group_start_14
   - group_end_8 <= 80
 constr_link_match_1_8_15: 80 switch_nic_out_8_nic_in_15 - group_start_15
   + group_end_8 <= 80
 constr_link_match_2_8_15: 80 switch_nic_out_8_nic_in_15 + group_start_15
   - group_end_8 <= 80
 constr_link_match_1_8_16: 80 switch_nic_out_8_nic_in_16 - group_start_16
   + group_end_8 <= 80
 constr_link_match_2_8_16: 80 switch_nic_out_8_nic_in_16 + group_start_16
   - group_end_8 <= 80
 constr_link_match_1_8_17: 80 switch_nic_out_8_nic_in_17 - group_start_17
   + group_end_8 <= 80
 constr_link_match_2_8_17: 80 switch_nic_out_8_nic_in_17 + group_start_17
   - group_end_8 <= 80
 constr_link_match_1_8_18: 80 switch_nic_out_8_nic_in_18 - group_start_18
   + group_end_8 <= 80
 constr_link_match_2_8_18: 80 switch_nic_out_8_nic_in_18 + group_start_18
   - group_end_8 <= 80
 constr_link_match_1_8_19: 80 switch_nic_out_8_nic_in_19 - group_start_19
   + group_end_8 <= 80
 constr_link_match_2_8_19: 80 switch_nic_out_8_nic_in_19 + group_start_19
   - group_end_8 <= 80
 constr_link_match_1_8_20: 80 switch_nic_out_8_nic_in_20 - group_start_20
   + group_end_8 <= 80
 constr_link_match_2_8_20: 80 switch_nic_out_8_nic_in_20 + group_start_20
   - group_end_8 <= 80
 constr_link_match_1_8_21: 80 switch_nic_out_8_nic_in_21 - group_start_21
   + group_end_8 <= 80
 constr_link_match_2_8_21: 80 switch_nic_out_8_nic_in_21 + group_start_21
   - group_end_8 <= 80
 constr_link_match_1_8_22: 80 switch_nic_out_8_nic_in_22 - group_start_22
   + group_end_8 <= 80
 constr_link_match_2_8_22: 80 switch_nic_out_8_nic_in_22 + group_start_22
   - group_end_8 <= 80
 constr_link_match_1_8_23: 80 switch_nic_out_8_nic_in_23 - group_start_23
   + group_end_8 <= 80
 constr_link_match_2_8_23: 80 switch_nic_out_8_nic_in_23 + group_start_23
   - group_end_8 <= 80
 constr_link_match_1_8_24: 80 switch_nic_out_8_nic_in_24 - group_start_24
   + group_end_8 <= 80
 constr_link_match_2_8_24: 80 switch_nic_out_8_nic_in_24 + group_start_24
   - group_end_8 <= 80
 constr_link_match_1_8_25: 80 switch_nic_out_8_nic_in_25 - group_start_25
   + group_end_8 <= 80
 constr_link_match_2_8_25: 80 switch_nic_out_8_nic_in_25 + group_start_25
   - group_end_8 <= 80
 constr_link_match_1_8_26: 80 switch_nic_out_8_nic_in_26 - group_start_26
   + group_end_8 <= 80
 constr_link_match_2_8_26: 80 switch_nic_out_8_nic_in_26 + group_start_26
   - group_end_8 <= 80
 constr_link_match_1_8_27: 80 switch_nic_out_8_nic_in_27 - group_start_27
   + group_end_8 <= 80
 constr_link_match_2_8_27: 80 switch_nic_out_8_nic_in_27 + group_start_27
   - group_end_8 <= 80
 constr_link_match_1_8_28: 80 switch_nic_out_8_nic_in_28 - group_start_28
   + group_end_8 <= 80
 constr_link_match_2_8_28: 80 switch_nic_out_8_nic_in_28 + group_start_28
   - group_end_8 <= 80
 constr_link_match_1_8_29: 80 switch_nic_out_8_nic_in_29 - group_start_29
   + group_end_8 <= 80
 constr_link_match_2_8_29: 80 switch_nic_out_8_nic_in_29 + group_start_29
   - group_end_8 <= 80
 constr_link_match_1_8_30: 80 switch_nic_out_8_nic_in_30 - group_start_30
   + group_end_8 <= 80
 constr_link_match_2_8_30: 80 switch_nic_out_8_nic_in_30 + group_start_30
   - group_end_8 <= 80
 constr_link_match_1_8_31: 80 switch_nic_out_8_nic_in_31 - group_start_31
   + group_end_8 <= 80
 constr_link_match_2_8_31: 80 switch_nic_out_8_nic_in_31 + group_start_31
   - group_end_8 <= 80
 constr_is_start_zero_9: start_9 + 80 is_start_zero_9_for_source_nic_in_9
   <= 80
 constr_edge_switch_start_zero_9_OR: switch_source_nic_in_9
   - is_start_zero_9_for_source_nic_in_9 <= 0
 constr_is_end_complete_9: start_9 + hold_9_1 + 2 hold_9_2 + 3 hold_9_3
   + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 - 80 is_end_complete_9 >= 0
 constr_edge_switch_end_max_9_OR: switch_nic_out_9_sink - is_end_complete_9
   <= 0
 constr_link_match_1_9_0: 80 switch_nic_out_9_nic_in_0 - group_start_0
   + group_end_9 <= 80
 constr_link_match_2_9_0: 80 switch_nic_out_9_nic_in_0 + group_start_0
   - group_end_9 <= 80
 constr_link_match_1_9_1: 80 switch_nic_out_9_nic_in_1 - group_start_1
   + group_end_9 <= 80
 constr_link_match_2_9_1: 80 switch_nic_out_9_nic_in_1 + group_start_1
   - group_end_9 <= 80
 constr_link_match_1_9_2: 80 switch_nic_out_9_nic_in_2 - group_start_2
   + group_end_9 <= 80
 constr_link_match_2_9_2: 80 switch_nic_out_9_nic_in_2 + group_start_2
   - group_end_9 <= 80
 constr_link_match_1_9_3: 80 switch_nic_out_9_nic_in_3 - group_start_3
   + group_end_9 <= 80
 constr_link_match_2_9_3: 80 switch_nic_out_9_nic_in_3 + group_start_3
   - group_end_9 <= 80
 constr_link_match_1_9_4: 80 switch_nic_out_9_nic_in_4 - group_start_4
   + group_end_9 <= 80
 constr_link_match_2_9_4: 80 switch_nic_out_9_nic_in_4 + group_start_4
   - group_end_9 <= 80
 constr_link_match_1_9_5: 80 switch_nic_out_9_nic_in_5 - group_start_5
   + group_end_9 <= 80
 constr_link_match_2_9_5: 80 switch_nic_out_9_nic_in_5 + group_start_5
   - group_end_9 <= 80
 constr_link_match_1_9_6: 80 switch_nic_out_9_nic_in_6 - group_start_6
   + group_end_9 <= 80
 constr_link_match_2_9_6: 80 switch_nic_out_9_nic_in_6 + group_start_6
   - group_end_9 <= 80
 constr_link_match_1_9_7: 80 switch_nic_out_9_nic_in_7 - group_start_7
   + group_end_9 <= 80
 constr_link_match_2_9_7: 80 switch_nic_out_9_nic_in_7 + group_start_7
   - group_end_9 <= 80
 constr_link_match_1_9_8: 80 switch_nic_out_9_nic_in_8 - group_start_8
   + group_end_9 <= 80
 constr_link_match_2_9_8: 80 switch_nic_out_9_nic_in_8 + group_start_8
   - group_end_9 <= 80
 constr_link_match_1_9_10: 80 switch_nic_out_9_nic_in_10 - group_start_10
   + group_end_9 <= 80
 constr_link_match_2_9_10: 80 switch_nic_out_9_nic_in_10 + group_start_10
   - group_end_9 <= 80
 constr_link_match_1_9_11: 80 switch_nic_out_9_nic_in_11 - group_start_11
   + group_end_9 <= 80
 constr_link_match_2_9_11: 80 switch_nic_out_9_nic_in_11 + group_start_11
   - group_end_9 <= 80
 constr_link_match_1_9_12: 80 switch_nic_out_9_nic_in_12 - group_start_12
   + group_end_9 <= 80
 constr_link_match_2_9_12: 80 switch_nic_out_9_nic_in_12 + group_start_12
   - group_end_9 <= 80
 constr_link_match_1_9_13: 80 switch_nic_out_9_nic_in_13 - group_start_13
   + group_end_9 <= 80
 constr_link_match_2_9_13: 80 switch_nic_out_9_nic_in_13 + group_start_13
   - group_end_9 <= 80
 constr_link_match_1_9_14: 80 switch_nic_out_9_nic_in_14 - group_start_14
   + group_end_9 <= 80
 constr_link_match_2_9_14: 80 switch_nic_out_9_nic_in_14 + group_start_14
   - group_end_9 <= 80
 constr_link_match_1_9_15: 80 switch_nic_out_9_nic_in_15 - group_start_15
   + group_end_9 <= 80
 constr_link_match_2_9_15: 80 switch_nic_out_9_nic_in_15 + group_start_15
   - group_end_9 <= 80
 constr_link_match_1_9_16: 80 switch_nic_out_9_nic_in_16 - group_start_16
   + group_end_9 <= 80
 constr_link_match_2_9_16: 80 switch_nic_out_9_nic_in_16 + group_start_16
   - group_end_9 <= 80
 constr_link_match_1_9_17: 80 switch_nic_out_9_nic_in_17 - group_start_17
   + group_end_9 <= 80
 constr_link_match_2_9_17: 80 switch_nic_out_9_nic_in_17 + group_start_17
   - group_end_9 <= 80
 constr_link_match_1_9_18: 80 switch_nic_out_9_nic_in_18 - group_start_18
   + group_end_9 <= 80
 constr_link_match_2_9_18: 80 switch_nic_out_9_nic_in_18 + group_start_18
   - group_end_9 <= 80
 constr_link_match_1_9_19: 80 switch_nic_out_9_nic_in_19 - group_start_19
   + group_end_9 <= 80
 constr_link_match_2_9_19: 80 switch_nic_out_9_nic_in_19 + group_start_19
   - group_end_9 <= 80
 constr_link_match_1_9_20: 80 switch_nic_out_9_nic_in_20 - group_start_20
   + group_end_9 <= 80
 constr_link_match_2_9_20: 80 switch_nic_out_9_nic_in_20 + group_start_20
   - group_end_9 <= 80
 constr_link_match_1_9_21: 80 switch_nic_out_9_nic_in_21 - group_start_21
   + group_end_9 <= 80
 constr_link_match_2_9_21: 80 switch_nic_out_9_nic_in_21 + group_start_21
   - group_end_9 <= 80
 constr_link_match_1_9_22: 80 switch_nic_out_9_nic_in_22 - group_start_22
   + group_end_9 <= 80
 constr_link_match_2_9_22: 80 switch_nic_out_9_nic_in_22 + group_start_22
   - group_end_9 <= 80
 constr_link_match_1_9_23: 80 switch_nic_out_9_nic_in_23 - group_start_23
   + group_end_9 <= 80
 constr_link_match_2_9_23: 80 switch_nic_out_9_nic_in_23 + group_start_23
   - group_end_9 <= 80
 constr_link_match_1_9_24: 80 switch_nic_out_9_nic_in_24 - group_start_24
   + group_end_9 <= 80
 constr_link_match_2_9_24: 80 switch_nic_out_9_nic_in_24 + group_start_24
   - group_end_9 <= 80
 constr_link_match_1_9_25: 80 switch_nic_out_9_nic_in_25 - group_start_25
   + group_end_9 <= 80
 constr_link_match_2_9_25: 80 switch_nic_out_9_nic_in_25 + group_start_25
   - group_end_9 <= 80
 constr_link_match_1_9_26: 80 switch_nic_out_9_nic_in_26 - group_start_26
   + group_end_9 <= 80
 constr_link_match_2_9_26: 80 switch_nic_out_9_nic_in_26 + group_start_26
   - group_end_9 <= 80
 constr_link_match_1_9_27: 80 switch_nic_out_9_nic_in_27 - group_start_27
   + group_end_9 <= 80
 constr_link_match_2_9_27: 80 switch_nic_out_9_nic_in_27 + group_start_27
   - group_end_9 <= 80
 constr_link_match_1_9_28: 80 switch_nic_out_9_nic_in_28 - group_start_28
   + group_end_9 <= 80
 constr_link_match_2_9_28: 80 switch_nic_out_9_nic_in_28 + group_start_28
   - group_end_9 <= 80
 constr_link_match_1_9_29: 80 switch_nic_out_9_nic_in_29 - group_start_29
   + group_end_9 <= 80
 constr_link_match_2_9_29: 80 switch_nic_out_9_nic_in_29 + group_start_29
   - group_end_9 <= 80
 constr_link_match_1_9_30: 80 switch_nic_out_9_nic_in_30 - group_start_30
   + group_end_9 <= 80
 constr_link_match_2_9_30: 80 switch_nic_out_9_nic_in_30 + group_start_30
   - group_end_9 <= 80
 constr_link_match_1_9_31: 80 switch_nic_out_9_nic_in_31 - group_start_31
   + group_end_9 <= 80
 constr_link_match_2_9_31: 80 switch_nic_out_9_nic_in_31 + group_start_31
   - group_end_9 <= 80
 constr_is_start_zero_10: start_10
   + 80 is_start_zero_10_for_source_nic_in_10 <= 80
 constr_edge_switch_start_zero_10_OR: switch_source_nic_in_10
   - is_start_zero_10_for_source_nic_in_10 <= 0
 constr_is_end_complete_10: start_10 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6
   - 80 is_end_complete_10 >= 0
 constr_edge_switch_end_max_10_OR: switch_nic_out_10_sink
   - is_end_complete_10 <= 0
 constr_link_match_1_10_0: 80 switch_nic_out_10_nic_in_0 - group_start_0
   + group_end_10 <= 80
 constr_link_match_2_10_0: 80 switch_nic_out_10_nic_in_0 + group_start_0
   - group_end_10 <= 80
 constr_link_match_1_10_1: 80 switch_nic_out_10_nic_in_1 - group_start_1
   + group_end_10 <= 80
 constr_link_match_2_10_1: 80 switch_nic_out_10_nic_in_1 + group_start_1
   - group_end_10 <= 80
 constr_link_match_1_10_2: 80 switch_nic_out_10_nic_in_2 - group_start_2
   + group_end_10 <= 80
 constr_link_match_2_10_2: 80 switch_nic_out_10_nic_in_2 + group_start_2
   - group_end_10 <= 80
 constr_link_match_1_10_3: 80 switch_nic_out_10_nic_in_3 - group_start_3
   + group_end_10 <= 80
 constr_link_match_2_10_3: 80 switch_nic_out_10_nic_in_3 + group_start_3
   - group_end_10 <= 80
 constr_link_match_1_10_4: 80 switch_nic_out_10_nic_in_4 - group_start_4
   + group_end_10 <= 80
 constr_link_match_2_10_4: 80 switch_nic_out_10_nic_in_4 + group_start_4
   - group_end_10 <= 80
 constr_link_match_1_10_5: 80 switch_nic_out_10_nic_in_5 - group_start_5
   + group_end_10 <= 80
 constr_link_match_2_10_5: 80 switch_nic_out_10_nic_in_5 + group_start_5
   - group_end_10 <= 80
 constr_link_match_1_10_6: 80 switch_nic_out_10_nic_in_6 - group_start_6
   + group_end_10 <= 80
 constr_link_match_2_10_6: 80 switch_nic_out_10_nic_in_6 + group_start_6
   - group_end_10 <= 80
 constr_link_match_1_10_7: 80 switch_nic_out_10_nic_in_7 - group_start_7
   + group_end_10 <= 80
 constr_link_match_2_10_7: 80 switch_nic_out_10_nic_in_7 + group_start_7
   - group_end_10 <= 80
 constr_link_match_1_10_8: 80 switch_nic_out_10_nic_in_8 - group_start_8
   + group_end_10 <= 80
 constr_link_match_2_10_8: 80 switch_nic_out_10_nic_in_8 + group_start_8
   - group_end_10 <= 80
 constr_link_match_1_10_9: 80 switch_nic_out_10_nic_in_9 - group_start_9
   + group_end_10 <= 80
 constr_link_match_2_10_9: 80 switch_nic_out_10_nic_in_9 + group_start_9
   - group_end_10 <= 80
 constr_link_match_1_10_11: 80 switch_nic_out_10_nic_in_11 - group_start_11
   + group_end_10 <= 80
 constr_link_match_2_10_11: 80 switch_nic_out_10_nic_in_11 + group_start_11
   - group_end_10 <= 80
 constr_link_match_1_10_12: 80 switch_nic_out_10_nic_in_12 - group_start_12
   + group_end_10 <= 80
 constr_link_match_2_10_12: 80 switch_nic_out_10_nic_in_12 + group_start_12
   - group_end_10 <= 80
 constr_link_match_1_10_13: 80 switch_nic_out_10_nic_in_13 - group_start_13
   + group_end_10 <= 80
 constr_link_match_2_10_13: 80 switch_nic_out_10_nic_in_13 + group_start_13
   - group_end_10 <= 80
 constr_link_match_1_10_14: 80 switch_nic_out_10_nic_in_14 - group_start_14
   + group_end_10 <= 80
 constr_link_match_2_10_14: 80 switch_nic_out_10_nic_in_14 + group_start_14
   - group_end_10 <= 80
 constr_link_match_1_10_15: 80 switch_nic_out_10_nic_in_15 - group_start_15
   + group_end_10 <= 80
 constr_link_match_2_10_15: 80 switch_nic_out_10_nic_in_15 + group_start_15
   - group_end_10 <= 80
 constr_link_match_1_10_16: 80 switch_nic_out_10_nic_in_16 - group_start_16
   + group_end_10 <= 80
 constr_link_match_2_10_16: 80 switch_nic_out_10_nic_in_16 + group_start_16
   - group_end_10 <= 80
 constr_link_match_1_10_17: 80 switch_nic_out_10_nic_in_17 - group_start_17
   + group_end_10 <= 80
 constr_link_match_2_10_17: 80 switch_nic_out_10_nic_in_17 + group_start_17
   - group_end_10 <= 80
 constr_link_match_1_10_18: 80 switch_nic_out_10_nic_in_18 - group_start_18
   + group_end_10 <= 80
 constr_link_match_2_10_18: 80 switch_nic_out_10_nic_in_18 + group_start_18
   - group_end_10 <= 80
 constr_link_match_1_10_19: 80 switch_nic_out_10_nic_in_19 - group_start_19
   + group_end_10 <= 80
 constr_link_match_2_10_19: 80 switch_nic_out_10_nic_in_19 + group_start_19
   - group_end_10 <= 80
 constr_link_match_1_10_20: 80 switch_nic_out_10_nic_in_20 - group_start_20
   + group_end_10 <= 80
 constr_link_match_2_10_20: 80 switch_nic_out_10_nic_in_20 + group_start_20
   - group_end_10 <= 80
 constr_link_match_1_10_21: 80 switch_nic_out_10_nic_in_21 - group_start_21
   + group_end_10 <= 80
 constr_link_match_2_10_21: 80 switch_nic_out_10_nic_in_21 + group_start_21
   - group_end_10 <= 80
 constr_link_match_1_10_22: 80 switch_nic_out_10_nic_in_22 - group_start_22
   + group_end_10 <= 80
 constr_link_match_2_10_22: 80 switch_nic_out_10_nic_in_22 + group_start_22
   - group_end_10 <= 80
 constr_link_match_1_10_23: 80 switch_nic_out_10_nic_in_23 - group_start_23
   + group_end_10 <= 80
 constr_link_match_2_10_23: 80 switch_nic_out_10_nic_in_23 + group_start_23
   - group_end_10 <= 80
 constr_link_match_1_10_24: 80 switch_nic_out_10_nic_in_24 - group_start_24
   + group_end_10 <= 80
 constr_link_match_2_10_24: 80 switch_nic_out_10_nic_in_24 + group_start_24
   - group_end_10 <= 80
 constr_link_match_1_10_25: 80 switch_nic_out_10_nic_in_25 - group_start_25
   + group_end_10 <= 80
 constr_link_match_2_10_25: 80 switch_nic_out_10_nic_in_25 + group_start_25
   - group_end_10 <= 80
 constr_link_match_1_10_26: 80 switch_nic_out_10_nic_in_26 - group_start_26
   + group_end_10 <= 80
 constr_link_match_2_10_26: 80 switch_nic_out_10_nic_in_26 + group_start_26
   - group_end_10 <= 80
 constr_link_match_1_10_27: 80 switch_nic_out_10_nic_in_27 - group_start_27
   + group_end_10 <= 80
 constr_link_match_2_10_27: 80 switch_nic_out_10_nic_in_27 + group_start_27
   - group_end_10 <= 80
 constr_link_match_1_10_28: 80 switch_nic_out_10_nic_in_28 - group_start_28
   + group_end_10 <= 80
 constr_link_match_2_10_28: 80 switch_nic_out_10_nic_in_28 + group_start_28
   - group_end_10 <= 80
 constr_link_match_1_10_29: 80 switch_nic_out_10_nic_in_29 - group_start_29
   + group_end_10 <= 80
 constr_link_match_2_10_29: 80 switch_nic_out_10_nic_in_29 + group_start_29
   - group_end_10 <= 80
 constr_link_match_1_10_30: 80 switch_nic_out_10_nic_in_30 - group_start_30
   + group_end_10 <= 80
 constr_link_match_2_10_30: 80 switch_nic_out_10_nic_in_30 + group_start_30
   - group_end_10 <= 80
 constr_link_match_1_10_31: 80 switch_nic_out_10_nic_in_31 - group_start_31
   + group_end_10 <= 80
 constr_link_match_2_10_31: 80 switch_nic_out_10_nic_in_31 + group_start_31
   - group_end_10 <= 80
 constr_is_start_zero_11: start_11
   + 80 is_start_zero_11_for_source_nic_in_11 <= 80
 constr_edge_switch_start_zero_11_OR: switch_source_nic_in_11
   - is_start_zero_11_for_source_nic_in_11 <= 0
 constr_is_end_complete_11: start_11 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6
   - 80 is_end_complete_11 >= 0
 constr_edge_switch_end_max_11_OR: switch_nic_out_11_sink
   - is_end_complete_11 <= 0
 constr_link_match_1_11_0: 80 switch_nic_out_11_nic_in_0 - group_start_0
   + group_end_11 <= 80
 constr_link_match_2_11_0: 80 switch_nic_out_11_nic_in_0 + group_start_0
   - group_end_11 <= 80
 constr_link_match_1_11_1: 80 switch_nic_out_11_nic_in_1 - group_start_1
   + group_end_11 <= 80
 constr_link_match_2_11_1: 80 switch_nic_out_11_nic_in_1 + group_start_1
   - group_end_11 <= 80
 constr_link_match_1_11_2: 80 switch_nic_out_11_nic_in_2 - group_start_2
   + group_end_11 <= 80
 constr_link_match_2_11_2: 80 switch_nic_out_11_nic_in_2 + group_start_2
   - group_end_11 <= 80
 constr_link_match_1_11_3: 80 switch_nic_out_11_nic_in_3 - group_start_3
   + group_end_11 <= 80
 constr_link_match_2_11_3: 80 switch_nic_out_11_nic_in_3 + group_start_3
   - group_end_11 <= 80
 constr_link_match_1_11_4: 80 switch_nic_out_11_nic_in_4 - group_start_4
   + group_end_11 <= 80
 constr_link_match_2_11_4: 80 switch_nic_out_11_nic_in_4 + group_start_4
   - group_end_11 <= 80
 constr_link_match_1_11_5: 80 switch_nic_out_11_nic_in_5 - group_start_5
   + group_end_11 <= 80
 constr_link_match_2_11_5: 80 switch_nic_out_11_nic_in_5 + group_start_5
   - group_end_11 <= 80
 constr_link_match_1_11_6: 80 switch_nic_out_11_nic_in_6 - group_start_6
   + group_end_11 <= 80
 constr_link_match_2_11_6: 80 switch_nic_out_11_nic_in_6 + group_start_6
   - group_end_11 <= 80
 constr_link_match_1_11_7: 80 switch_nic_out_11_nic_in_7 - group_start_7
   + group_end_11 <= 80
 constr_link_match_2_11_7: 80 switch_nic_out_11_nic_in_7 + group_start_7
   - group_end_11 <= 80
 constr_link_match_1_11_8: 80 switch_nic_out_11_nic_in_8 - group_start_8
   + group_end_11 <= 80
 constr_link_match_2_11_8: 80 switch_nic_out_11_nic_in_8 + group_start_8
   - group_end_11 <= 80
 constr_link_match_1_11_9: 80 switch_nic_out_11_nic_in_9 - group_start_9
   + group_end_11 <= 80
 constr_link_match_2_11_9: 80 switch_nic_out_11_nic_in_9 + group_start_9
   - group_end_11 <= 80
 constr_link_match_1_11_10: 80 switch_nic_out_11_nic_in_10 - group_start_10
   + group_end_11 <= 80
 constr_link_match_2_11_10: 80 switch_nic_out_11_nic_in_10 + group_start_10
   - group_end_11 <= 80
 constr_link_match_1_11_12: 80 switch_nic_out_11_nic_in_12 - group_start_12
   + group_end_11 <= 80
 constr_link_match_2_11_12: 80 switch_nic_out_11_nic_in_12 + group_start_12
   - group_end_11 <= 80
 constr_link_match_1_11_13: 80 switch_nic_out_11_nic_in_13 - group_start_13
   + group_end_11 <= 80
 constr_link_match_2_11_13: 80 switch_nic_out_11_nic_in_13 + group_start_13
   - group_end_11 <= 80
 constr_link_match_1_11_14: 80 switch_nic_out_11_nic_in_14 - group_start_14
   + group_end_11 <= 80
 constr_link_match_2_11_14: 80 switch_nic_out_11_nic_in_14 + group_start_14
   - group_end_11 <= 80
 constr_link_match_1_11_15: 80 switch_nic_out_11_nic_in_15 - group_start_15
   + group_end_11 <= 80
 constr_link_match_2_11_15: 80 switch_nic_out_11_nic_in_15 + group_start_15
   - group_end_11 <= 80
 constr_link_match_1_11_16: 80 switch_nic_out_11_nic_in_16 - group_start_16
   + group_end_11 <= 80
 constr_link_match_2_11_16: 80 switch_nic_out_11_nic_in_16 + group_start_16
   - group_end_11 <= 80
 constr_link_match_1_11_17: 80 switch_nic_out_11_nic_in_17 - group_start_17
   + group_end_11 <= 80
 constr_link_match_2_11_17: 80 switch_nic_out_11_nic_in_17 + group_start_17
   - group_end_11 <= 80
 constr_link_match_1_11_18: 80 switch_nic_out_11_nic_in_18 - group_start_18
   + group_end_11 <= 80
 constr_link_match_2_11_18: 80 switch_nic_out_11_nic_in_18 + group_start_18
   - group_end_11 <= 80
 constr_link_match_1_11_19: 80 switch_nic_out_11_nic_in_19 - group_start_19
   + group_end_11 <= 80
 constr_link_match_2_11_19: 80 switch_nic_out_11_nic_in_19 + group_start_19
   - group_end_11 <= 80
 constr_link_match_1_11_20: 80 switch_nic_out_11_nic_in_20 - group_start_20
   + group_end_11 <= 80
 constr_link_match_2_11_20: 80 switch_nic_out_11_nic_in_20 + group_start_20
   - group_end_11 <= 80
 constr_link_match_1_11_21: 80 switch_nic_out_11_nic_in_21 - group_start_21
   + group_end_11 <= 80
 constr_link_match_2_11_21: 80 switch_nic_out_11_nic_in_21 + group_start_21
   - group_end_11 <= 80
 constr_link_match_1_11_22: 80 switch_nic_out_11_nic_in_22 - group_start_22
   + group_end_11 <= 80
 constr_link_match_2_11_22: 80 switch_nic_out_11_nic_in_22 + group_start_22
   - group_end_11 <= 80
 constr_link_match_1_11_23: 80 switch_nic_out_11_nic_in_23 - group_start_23
   + group_end_11 <= 80
 constr_link_match_2_11_23: 80 switch_nic_out_11_nic_in_23 + group_start_23
   - group_end_11 <= 80
 constr_link_match_1_11_24: 80 switch_nic_out_11_nic_in_24 - group_start_24
   + group_end_11 <= 80
 constr_link_match_2_11_24: 80 switch_nic_out_11_nic_in_24 + group_start_24
   - group_end_11 <= 80
 constr_link_match_1_11_25: 80 switch_nic_out_11_nic_in_25 - group_start_25
   + group_end_11 <= 80
 constr_link_match_2_11_25: 80 switch_nic_out_11_nic_in_25 + group_start_25
   - group_end_11 <= 80
 constr_link_match_1_11_26: 80 switch_nic_out_11_nic_in_26 - group_start_26
   + group_end_11 <= 80
 constr_link_match_2_11_26: 80 switch_nic_out_11_nic_in_26 + group_start_26
   - group_end_11 <= 80
 constr_link_match_1_11_27: 80 switch_nic_out_11_nic_in_27 - group_start_27
   + group_end_11 <= 80
 constr_link_match_2_11_27: 80 switch_nic_out_11_nic_in_27 + group_start_27
   - group_end_11 <= 80
 constr_link_match_1_11_28: 80 switch_nic_out_11_nic_in_28 - group_start_28
   + group_end_11 <= 80
 constr_link_match_2_11_28: 80 switch_nic_out_11_nic_in_28 + group_start_28
   - group_end_11 <= 80
 constr_link_match_1_11_29: 80 switch_nic_out_11_nic_in_29 - group_start_29
   + group_end_11 <= 80
 constr_link_match_2_11_29: 80 switch_nic_out_11_nic_in_29 + group_start_29
   - group_end_11 <= 80
 constr_link_match_1_11_30: 80 switch_nic_out_11_nic_in_30 - group_start_30
   + group_end_11 <= 80
 constr_link_match_2_11_30: 80 switch_nic_out_11_nic_in_30 + group_start_30
   - group_end_11 <= 80
 constr_link_match_1_11_31: 80 switch_nic_out_11_nic_in_31 - group_start_31
   + group_end_11 <= 80
 constr_link_match_2_11_31: 80 switch_nic_out_11_nic_in_31 + group_start_31
   - group_end_11 <= 80
 constr_is_start_zero_12: start_12
   + 80 is_start_zero_12_for_source_nic_in_12 <= 80
 constr_edge_switch_start_zero_12_OR: switch_source_nic_in_12
   - is_start_zero_12_for_source_nic_in_12 <= 0
 constr_is_end_complete_12: start_12 + hold_12_1 + 2 hold_12_2
   + 3 hold_12_3 + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6
   - 80 is_end_complete_12 >= 0
 constr_edge_switch_end_max_12_OR: switch_nic_out_12_sink
   - is_end_complete_12 <= 0
 constr_link_match_1_12_0: 80 switch_nic_out_12_nic_in_0 - group_start_0
   + group_end_12 <= 80
 constr_link_match_2_12_0: 80 switch_nic_out_12_nic_in_0 + group_start_0
   - group_end_12 <= 80
 constr_link_match_1_12_1: 80 switch_nic_out_12_nic_in_1 - group_start_1
   + group_end_12 <= 80
 constr_link_match_2_12_1: 80 switch_nic_out_12_nic_in_1 + group_start_1
   - group_end_12 <= 80
 constr_link_match_1_12_2: 80 switch_nic_out_12_nic_in_2 - group_start_2
   + group_end_12 <= 80
 constr_link_match_2_12_2: 80 switch_nic_out_12_nic_in_2 + group_start_2
   - group_end_12 <= 80
 constr_link_match_1_12_3: 80 switch_nic_out_12_nic_in_3 - group_start_3
   + group_end_12 <= 80
 constr_link_match_2_12_3: 80 switch_nic_out_12_nic_in_3 + group_start_3
   - group_end_12 <= 80
 constr_link_match_1_12_4: 80 switch_nic_out_12_nic_in_4 - group_start_4
   + group_end_12 <= 80
 constr_link_match_2_12_4: 80 switch_nic_out_12_nic_in_4 + group_start_4
   - group_end_12 <= 80
 constr_link_match_1_12_5: 80 switch_nic_out_12_nic_in_5 - group_start_5
   + group_end_12 <= 80
 constr_link_match_2_12_5: 80 switch_nic_out_12_nic_in_5 + group_start_5
   - group_end_12 <= 80
 constr_link_match_1_12_6: 80 switch_nic_out_12_nic_in_6 - group_start_6
   + group_end_12 <= 80
 constr_link_match_2_12_6: 80 switch_nic_out_12_nic_in_6 + group_start_6
   - group_end_12 <= 80
 constr_link_match_1_12_7: 80 switch_nic_out_12_nic_in_7 - group_start_7
   + group_end_12 <= 80
 constr_link_match_2_12_7: 80 switch_nic_out_12_nic_in_7 + group_start_7
   - group_end_12 <= 80
 constr_link_match_1_12_8: 80 switch_nic_out_12_nic_in_8 - group_start_8
   + group_end_12 <= 80
 constr_link_match_2_12_8: 80 switch_nic_out_12_nic_in_8 + group_start_8
   - group_end_12 <= 80
 constr_link_match_1_12_9: 80 switch_nic_out_12_nic_in_9 - group_start_9
   + group_end_12 <= 80
 constr_link_match_2_12_9: 80 switch_nic_out_12_nic_in_9 + group_start_9
   - group_end_12 <= 80
 constr_link_match_1_12_10: 80 switch_nic_out_12_nic_in_10 - group_start_10
   + group_end_12 <= 80
 constr_link_match_2_12_10: 80 switch_nic_out_12_nic_in_10 + group_start_10
   - group_end_12 <= 80
 constr_link_match_1_12_11: 80 switch_nic_out_12_nic_in_11 - group_start_11
   + group_end_12 <= 80
 constr_link_match_2_12_11: 80 switch_nic_out_12_nic_in_11 + group_start_11
   - group_end_12 <= 80
 constr_link_match_1_12_13: 80 switch_nic_out_12_nic_in_13 - group_start_13
   + group_end_12 <= 80
 constr_link_match_2_12_13: 80 switch_nic_out_12_nic_in_13 + group_start_13
   - group_end_12 <= 80
 constr_link_match_1_12_14: 80 switch_nic_out_12_nic_in_14 - group_start_14
   + group_end_12 <= 80
 constr_link_match_2_12_14: 80 switch_nic_out_12_nic_in_14 + group_start_14
   - group_end_12 <= 80
 constr_link_match_1_12_15: 80 switch_nic_out_12_nic_in_15 - group_start_15
   + group_end_12 <= 80
 constr_link_match_2_12_15: 80 switch_nic_out_12_nic_in_15 + group_start_15
   - group_end_12 <= 80
 constr_link_match_1_12_16: 80 switch_nic_out_12_nic_in_16 - group_start_16
   + group_end_12 <= 80
 constr_link_match_2_12_16: 80 switch_nic_out_12_nic_in_16 + group_start_16
   - group_end_12 <= 80
 constr_link_match_1_12_17: 80 switch_nic_out_12_nic_in_17 - group_start_17
   + group_end_12 <= 80
 constr_link_match_2_12_17: 80 switch_nic_out_12_nic_in_17 + group_start_17
   - group_end_12 <= 80
 constr_link_match_1_12_18: 80 switch_nic_out_12_nic_in_18 - group_start_18
   + group_end_12 <= 80
 constr_link_match_2_12_18: 80 switch_nic_out_12_nic_in_18 + group_start_18
   - group_end_12 <= 80
 constr_link_match_1_12_19: 80 switch_nic_out_12_nic_in_19 - group_start_19
   + group_end_12 <= 80
 constr_link_match_2_12_19: 80 switch_nic_out_12_nic_in_19 + group_start_19
   - group_end_12 <= 80
 constr_link_match_1_12_20: 80 switch_nic_out_12_nic_in_20 - group_start_20
   + group_end_12 <= 80
 constr_link_match_2_12_20: 80 switch_nic_out_12_nic_in_20 + group_start_20
   - group_end_12 <= 80
 constr_link_match_1_12_21: 80 switch_nic_out_12_nic_in_21 - group_start_21
   + group_end_12 <= 80
 constr_link_match_2_12_21: 80 switch_nic_out_12_nic_in_21 + group_start_21
   - group_end_12 <= 80
 constr_link_match_1_12_22: 80 switch_nic_out_12_nic_in_22 - group_start_22
   + group_end_12 <= 80
 constr_link_match_2_12_22: 80 switch_nic_out_12_nic_in_22 + group_start_22
   - group_end_12 <= 80
 constr_link_match_1_12_23: 80 switch_nic_out_12_nic_in_23 - group_start_23
   + group_end_12 <= 80
 constr_link_match_2_12_23: 80 switch_nic_out_12_nic_in_23 + group_start_23
   - group_end_12 <= 80
 constr_link_match_1_12_24: 80 switch_nic_out_12_nic_in_24 - group_start_24
   + group_end_12 <= 80
 constr_link_match_2_12_24: 80 switch_nic_out_12_nic_in_24 + group_start_24
   - group_end_12 <= 80
 constr_link_match_1_12_25: 80 switch_nic_out_12_nic_in_25 - group_start_25
   + group_end_12 <= 80
 constr_link_match_2_12_25: 80 switch_nic_out_12_nic_in_25 + group_start_25
   - group_end_12 <= 80
 constr_link_match_1_12_26: 80 switch_nic_out_12_nic_in_26 - group_start_26
   + group_end_12 <= 80
 constr_link_match_2_12_26: 80 switch_nic_out_12_nic_in_26 + group_start_26
   - group_end_12 <= 80
 constr_link_match_1_12_27: 80 switch_nic_out_12_nic_in_27 - group_start_27
   + group_end_12 <= 80
 constr_link_match_2_12_27: 80 switch_nic_out_12_nic_in_27 + group_start_27
   - group_end_12 <= 80
 constr_link_match_1_12_28: 80 switch_nic_out_12_nic_in_28 - group_start_28
   + group_end_12 <= 80
 constr_link_match_2_12_28: 80 switch_nic_out_12_nic_in_28 + group_start_28
   - group_end_12 <= 80
 constr_link_match_1_12_29: 80 switch_nic_out_12_nic_in_29 - group_start_29
   + group_end_12 <= 80
 constr_link_match_2_12_29: 80 switch_nic_out_12_nic_in_29 + group_start_29
   - group_end_12 <= 80
 constr_link_match_1_12_30: 80 switch_nic_out_12_nic_in_30 - group_start_30
   + group_end_12 <= 80
 constr_link_match_2_12_30: 80 switch_nic_out_12_nic_in_30 + group_start_30
   - group_end_12 <= 80
 constr_link_match_1_12_31: 80 switch_nic_out_12_nic_in_31 - group_start_31
   + group_end_12 <= 80
 constr_link_match_2_12_31: 80 switch_nic_out_12_nic_in_31 + group_start_31
   - group_end_12 <= 80
 constr_is_start_zero_13: start_13
   + 80 is_start_zero_13_for_source_nic_in_13 <= 80
 constr_edge_switch_start_zero_13_OR: switch_source_nic_in_13
   - is_start_zero_13_for_source_nic_in_13 <= 0
 constr_is_end_complete_13: start_13 + hold_13_1 + 2 hold_13_2
   + 3 hold_13_3 + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6
   - 80 is_end_complete_13 >= 0
 constr_edge_switch_end_max_13_OR: switch_nic_out_13_sink
   - is_end_complete_13 <= 0
 constr_link_match_1_13_0: 80 switch_nic_out_13_nic_in_0 - group_start_0
   + group_end_13 <= 80
 constr_link_match_2_13_0: 80 switch_nic_out_13_nic_in_0 + group_start_0
   - group_end_13 <= 80
 constr_link_match_1_13_1: 80 switch_nic_out_13_nic_in_1 - group_start_1
   + group_end_13 <= 80
 constr_link_match_2_13_1: 80 switch_nic_out_13_nic_in_1 + group_start_1
   - group_end_13 <= 80
 constr_link_match_1_13_2: 80 switch_nic_out_13_nic_in_2 - group_start_2
   + group_end_13 <= 80
 constr_link_match_2_13_2: 80 switch_nic_out_13_nic_in_2 + group_start_2
   - group_end_13 <= 80
 constr_link_match_1_13_3: 80 switch_nic_out_13_nic_in_3 - group_start_3
   + group_end_13 <= 80
 constr_link_match_2_13_3: 80 switch_nic_out_13_nic_in_3 + group_start_3
   - group_end_13 <= 80
 constr_link_match_1_13_4: 80 switch_nic_out_13_nic_in_4 - group_start_4
   + group_end_13 <= 80
 constr_link_match_2_13_4: 80 switch_nic_out_13_nic_in_4 + group_start_4
   - group_end_13 <= 80
 constr_link_match_1_13_5: 80 switch_nic_out_13_nic_in_5 - group_start_5
   + group_end_13 <= 80
 constr_link_match_2_13_5: 80 switch_nic_out_13_nic_in_5 + group_start_5
   - group_end_13 <= 80
 constr_link_match_1_13_6: 80 switch_nic_out_13_nic_in_6 - group_start_6
   + group_end_13 <= 80
 constr_link_match_2_13_6: 80 switch_nic_out_13_nic_in_6 + group_start_6
   - group_end_13 <= 80
 constr_link_match_1_13_7: 80 switch_nic_out_13_nic_in_7 - group_start_7
   + group_end_13 <= 80
 constr_link_match_2_13_7: 80 switch_nic_out_13_nic_in_7 + group_start_7
   - group_end_13 <= 80
 constr_link_match_1_13_8: 80 switch_nic_out_13_nic_in_8 - group_start_8
   + group_end_13 <= 80
 constr_link_match_2_13_8: 80 switch_nic_out_13_nic_in_8 + group_start_8
   - group_end_13 <= 80
 constr_link_match_1_13_9: 80 switch_nic_out_13_nic_in_9 - group_start_9
   + group_end_13 <= 80
 constr_link_match_2_13_9: 80 switch_nic_out_13_nic_in_9 + group_start_9
   - group_end_13 <= 80
 constr_link_match_1_13_10: 80 switch_nic_out_13_nic_in_10 - group_start_10
   + group_end_13 <= 80
 constr_link_match_2_13_10: 80 switch_nic_out_13_nic_in_10 + group_start_10
   - group_end_13 <= 80
 constr_link_match_1_13_11: 80 switch_nic_out_13_nic_in_11 - group_start_11
   + group_end_13 <= 80
 constr_link_match_2_13_11: 80 switch_nic_out_13_nic_in_11 + group_start_11
   - group_end_13 <= 80
 constr_link_match_1_13_12: 80 switch_nic_out_13_nic_in_12 - group_start_12
   + group_end_13 <= 80
 constr_link_match_2_13_12: 80 switch_nic_out_13_nic_in_12 + group_start_12
   - group_end_13 <= 80
 constr_link_match_1_13_14: 80 switch_nic_out_13_nic_in_14 - group_start_14
   + group_end_13 <= 80
 constr_link_match_2_13_14: 80 switch_nic_out_13_nic_in_14 + group_start_14
   - group_end_13 <= 80
 constr_link_match_1_13_15: 80 switch_nic_out_13_nic_in_15 - group_start_15
   + group_end_13 <= 80
 constr_link_match_2_13_15: 80 switch_nic_out_13_nic_in_15 + group_start_15
   - group_end_13 <= 80
 constr_link_match_1_13_16: 80 switch_nic_out_13_nic_in_16 - group_start_16
   + group_end_13 <= 80
 constr_link_match_2_13_16: 80 switch_nic_out_13_nic_in_16 + group_start_16
   - group_end_13 <= 80
 constr_link_match_1_13_17: 80 switch_nic_out_13_nic_in_17 - group_start_17
   + group_end_13 <= 80
 constr_link_match_2_13_17: 80 switch_nic_out_13_nic_in_17 + group_start_17
   - group_end_13 <= 80
 constr_link_match_1_13_18: 80 switch_nic_out_13_nic_in_18 - group_start_18
   + group_end_13 <= 80
 constr_link_match_2_13_18: 80 switch_nic_out_13_nic_in_18 + group_start_18
   - group_end_13 <= 80
 constr_link_match_1_13_19: 80 switch_nic_out_13_nic_in_19 - group_start_19
   + group_end_13 <= 80
 constr_link_match_2_13_19: 80 switch_nic_out_13_nic_in_19 + group_start_19
   - group_end_13 <= 80
 constr_link_match_1_13_20: 80 switch_nic_out_13_nic_in_20 - group_start_20
   + group_end_13 <= 80
 constr_link_match_2_13_20: 80 switch_nic_out_13_nic_in_20 + group_start_20
   - group_end_13 <= 80
 constr_link_match_1_13_21: 80 switch_nic_out_13_nic_in_21 - group_start_21
   + group_end_13 <= 80
 constr_link_match_2_13_21: 80 switch_nic_out_13_nic_in_21 + group_start_21
   - group_end_13 <= 80
 constr_link_match_1_13_22: 80 switch_nic_out_13_nic_in_22 - group_start_22
   + group_end_13 <= 80
 constr_link_match_2_13_22: 80 switch_nic_out_13_nic_in_22 + group_start_22
   - group_end_13 <= 80
 constr_link_match_1_13_23: 80 switch_nic_out_13_nic_in_23 - group_start_23
   + group_end_13 <= 80
 constr_link_match_2_13_23: 80 switch_nic_out_13_nic_in_23 + group_start_23
   - group_end_13 <= 80
 constr_link_match_1_13_24: 80 switch_nic_out_13_nic_in_24 - group_start_24
   + group_end_13 <= 80
 constr_link_match_2_13_24: 80 switch_nic_out_13_nic_in_24 + group_start_24
   - group_end_13 <= 80
 constr_link_match_1_13_25: 80 switch_nic_out_13_nic_in_25 - group_start_25
   + group_end_13 <= 80
 constr_link_match_2_13_25: 80 switch_nic_out_13_nic_in_25 + group_start_25
   - group_end_13 <= 80
 constr_link_match_1_13_26: 80 switch_nic_out_13_nic_in_26 - group_start_26
   + group_end_13 <= 80
 constr_link_match_2_13_26: 80 switch_nic_out_13_nic_in_26 + group_start_26
   - group_end_13 <= 80
 constr_link_match_1_13_27: 80 switch_nic_out_13_nic_in_27 - group_start_27
   + group_end_13 <= 80
 constr_link_match_2_13_27: 80 switch_nic_out_13_nic_in_27 + group_start_27
   - group_end_13 <= 80
 constr_link_match_1_13_28: 80 switch_nic_out_13_nic_in_28 - group_start_28
   + group_end_13 <= 80
 constr_link_match_2_13_28: 80 switch_nic_out_13_nic_in_28 + group_start_28
   - group_end_13 <= 80
 constr_link_match_1_13_29: 80 switch_nic_out_13_nic_in_29 - group_start_29
   + group_end_13 <= 80
 constr_link_match_2_13_29: 80 switch_nic_out_13_nic_in_29 + group_start_29
   - group_end_13 <= 80
 constr_link_match_1_13_30: 80 switch_nic_out_13_nic_in_30 - group_start_30
   + group_end_13 <= 80
 constr_link_match_2_13_30: 80 switch_nic_out_13_nic_in_30 + group_start_30
   - group_end_13 <= 80
 constr_link_match_1_13_31: 80 switch_nic_out_13_nic_in_31 - group_start_31
   + group_end_13 <= 80
 constr_link_match_2_13_31: 80 switch_nic_out_13_nic_in_31 + group_start_31
   - group_end_13 <= 80
 constr_is_start_zero_14: start_14
   + 80 is_start_zero_14_for_source_nic_in_14 <= 80
 constr_edge_switch_start_zero_14_OR: switch_source_nic_in_14
   - is_start_zero_14_for_source_nic_in_14 <= 0
 constr_is_end_complete_14: start_14 + hold_14_1 + 2 hold_14_2
   + 3 hold_14_3 + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6
   - 80 is_end_complete_14 >= 0
 constr_edge_switch_end_max_14_OR: switch_nic_out_14_sink
   - is_end_complete_14 <= 0
 constr_link_match_1_14_0: 80 switch_nic_out_14_nic_in_0 - group_start_0
   + group_end_14 <= 80
 constr_link_match_2_14_0: 80 switch_nic_out_14_nic_in_0 + group_start_0
   - group_end_14 <= 80
 constr_link_match_1_14_1: 80 switch_nic_out_14_nic_in_1 - group_start_1
   + group_end_14 <= 80
 constr_link_match_2_14_1: 80 switch_nic_out_14_nic_in_1 + group_start_1
   - group_end_14 <= 80
 constr_link_match_1_14_2: 80 switch_nic_out_14_nic_in_2 - group_start_2
   + group_end_14 <= 80
 constr_link_match_2_14_2: 80 switch_nic_out_14_nic_in_2 + group_start_2
   - group_end_14 <= 80
 constr_link_match_1_14_3: 80 switch_nic_out_14_nic_in_3 - group_start_3
   + group_end_14 <= 80
 constr_link_match_2_14_3: 80 switch_nic_out_14_nic_in_3 + group_start_3
   - group_end_14 <= 80
 constr_link_match_1_14_4: 80 switch_nic_out_14_nic_in_4 - group_start_4
   + group_end_14 <= 80
 constr_link_match_2_14_4: 80 switch_nic_out_14_nic_in_4 + group_start_4
   - group_end_14 <= 80
 constr_link_match_1_14_5: 80 switch_nic_out_14_nic_in_5 - group_start_5
   + group_end_14 <= 80
 constr_link_match_2_14_5: 80 switch_nic_out_14_nic_in_5 + group_start_5
   - group_end_14 <= 80
 constr_link_match_1_14_6: 80 switch_nic_out_14_nic_in_6 - group_start_6
   + group_end_14 <= 80
 constr_link_match_2_14_6: 80 switch_nic_out_14_nic_in_6 + group_start_6
   - group_end_14 <= 80
 constr_link_match_1_14_7: 80 switch_nic_out_14_nic_in_7 - group_start_7
   + group_end_14 <= 80
 constr_link_match_2_14_7: 80 switch_nic_out_14_nic_in_7 + group_start_7
   - group_end_14 <= 80
 constr_link_match_1_14_8: 80 switch_nic_out_14_nic_in_8 - group_start_8
   + group_end_14 <= 80
 constr_link_match_2_14_8: 80 switch_nic_out_14_nic_in_8 + group_start_8
   - group_end_14 <= 80
 constr_link_match_1_14_9: 80 switch_nic_out_14_nic_in_9 - group_start_9
   + group_end_14 <= 80
 constr_link_match_2_14_9: 80 switch_nic_out_14_nic_in_9 + group_start_9
   - group_end_14 <= 80
 constr_link_match_1_14_10: 80 switch_nic_out_14_nic_in_10 - group_start_10
   + group_end_14 <= 80
 constr_link_match_2_14_10: 80 switch_nic_out_14_nic_in_10 + group_start_10
   - group_end_14 <= 80
 constr_link_match_1_14_11: 80 switch_nic_out_14_nic_in_11 - group_start_11
   + group_end_14 <= 80
 constr_link_match_2_14_11: 80 switch_nic_out_14_nic_in_11 + group_start_11
   - group_end_14 <= 80
 constr_link_match_1_14_12: 80 switch_nic_out_14_nic_in_12 - group_start_12
   + group_end_14 <= 80
 constr_link_match_2_14_12: 80 switch_nic_out_14_nic_in_12 + group_start_12
   - group_end_14 <= 80
 constr_link_match_1_14_13: 80 switch_nic_out_14_nic_in_13 - group_start_13
   + group_end_14 <= 80
 constr_link_match_2_14_13: 80 switch_nic_out_14_nic_in_13 + group_start_13
   - group_end_14 <= 80
 constr_link_match_1_14_15: 80 switch_nic_out_14_nic_in_15 - group_start_15
   + group_end_14 <= 80
 constr_link_match_2_14_15: 80 switch_nic_out_14_nic_in_15 + group_start_15
   - group_end_14 <= 80
 constr_link_match_1_14_16: 80 switch_nic_out_14_nic_in_16 - group_start_16
   + group_end_14 <= 80
 constr_link_match_2_14_16: 80 switch_nic_out_14_nic_in_16 + group_start_16
   - group_end_14 <= 80
 constr_link_match_1_14_17: 80 switch_nic_out_14_nic_in_17 - group_start_17
   + group_end_14 <= 80
 constr_link_match_2_14_17: 80 switch_nic_out_14_nic_in_17 + group_start_17
   - group_end_14 <= 80
 constr_link_match_1_14_18: 80 switch_nic_out_14_nic_in_18 - group_start_18
   + group_end_14 <= 80
 constr_link_match_2_14_18: 80 switch_nic_out_14_nic_in_18 + group_start_18
   - group_end_14 <= 80
 constr_link_match_1_14_19: 80 switch_nic_out_14_nic_in_19 - group_start_19
   + group_end_14 <= 80
 constr_link_match_2_14_19: 80 switch_nic_out_14_nic_in_19 + group_start_19
   - group_end_14 <= 80
 constr_link_match_1_14_20: 80 switch_nic_out_14_nic_in_20 - group_start_20
   + group_end_14 <= 80
 constr_link_match_2_14_20: 80 switch_nic_out_14_nic_in_20 + group_start_20
   - group_end_14 <= 80
 constr_link_match_1_14_21: 80 switch_nic_out_14_nic_in_21 - group_start_21
   + group_end_14 <= 80
 constr_link_match_2_14_21: 80 switch_nic_out_14_nic_in_21 + group_start_21
   - group_end_14 <= 80
 constr_link_match_1_14_22: 80 switch_nic_out_14_nic_in_22 - group_start_22
   + group_end_14 <= 80
 constr_link_match_2_14_22: 80 switch_nic_out_14_nic_in_22 + group_start_22
   - group_end_14 <= 80
 constr_link_match_1_14_23: 80 switch_nic_out_14_nic_in_23 - group_start_23
   + group_end_14 <= 80
 constr_link_match_2_14_23: 80 switch_nic_out_14_nic_in_23 + group_start_23
   - group_end_14 <= 80
 constr_link_match_1_14_24: 80 switch_nic_out_14_nic_in_24 - group_start_24
   + group_end_14 <= 80
 constr_link_match_2_14_24: 80 switch_nic_out_14_nic_in_24 + group_start_24
   - group_end_14 <= 80
 constr_link_match_1_14_25: 80 switch_nic_out_14_nic_in_25 - group_start_25
   + group_end_14 <= 80
 constr_link_match_2_14_25: 80 switch_nic_out_14_nic_in_25 + group_start_25
   - group_end_14 <= 80
 constr_link_match_1_14_26: 80 switch_nic_out_14_nic_in_26 - group_start_26
   + group_end_14 <= 80
 constr_link_match_2_14_26: 80 switch_nic_out_14_nic_in_26 + group_start_26
   - group_end_14 <= 80
 constr_link_match_1_14_27: 80 switch_nic_out_14_nic_in_27 - group_start_27
   + group_end_14 <= 80
 constr_link_match_2_14_27: 80 switch_nic_out_14_nic_in_27 + group_start_27
   - group_end_14 <= 80
 constr_link_match_1_14_28: 80 switch_nic_out_14_nic_in_28 - group_start_28
   + group_end_14 <= 80
 constr_link_match_2_14_28: 80 switch_nic_out_14_nic_in_28 + group_start_28
   - group_end_14 <= 80
 constr_link_match_1_14_29: 80 switch_nic_out_14_nic_in_29 - group_start_29
   + group_end_14 <= 80
 constr_link_match_2_14_29: 80 switch_nic_out_14_nic_in_29 + group_start_29
   - group_end_14 <= 80
 constr_link_match_1_14_30: 80 switch_nic_out_14_nic_in_30 - group_start_30
   + group_end_14 <= 80
 constr_link_match_2_14_30: 80 switch_nic_out_14_nic_in_30 + group_start_30
   - group_end_14 <= 80
 constr_link_match_1_14_31: 80 switch_nic_out_14_nic_in_31 - group_start_31
   + group_end_14 <= 80
 constr_link_match_2_14_31: 80 switch_nic_out_14_nic_in_31 + group_start_31
   - group_end_14 <= 80
 constr_is_start_zero_15: start_15
   + 80 is_start_zero_15_for_source_nic_in_15 <= 80
 constr_edge_switch_start_zero_15_OR: switch_source_nic_in_15
   - is_start_zero_15_for_source_nic_in_15 <= 0
 constr_is_end_complete_15: start_15 + hold_15_1 + 2 hold_15_2
   + 3 hold_15_3 + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6
   - 80 is_end_complete_15 >= 0
 constr_edge_switch_end_max_15_OR: switch_nic_out_15_sink
   - is_end_complete_15 <= 0
 constr_link_match_1_15_0: 80 switch_nic_out_15_nic_in_0 - group_start_0
   + group_end_15 <= 80
 constr_link_match_2_15_0: 80 switch_nic_out_15_nic_in_0 + group_start_0
   - group_end_15 <= 80
 constr_link_match_1_15_1: 80 switch_nic_out_15_nic_in_1 - group_start_1
   + group_end_15 <= 80
 constr_link_match_2_15_1: 80 switch_nic_out_15_nic_in_1 + group_start_1
   - group_end_15 <= 80
 constr_link_match_1_15_2: 80 switch_nic_out_15_nic_in_2 - group_start_2
   + group_end_15 <= 80
 constr_link_match_2_15_2: 80 switch_nic_out_15_nic_in_2 + group_start_2
   - group_end_15 <= 80
 constr_link_match_1_15_3: 80 switch_nic_out_15_nic_in_3 - group_start_3
   + group_end_15 <= 80
 constr_link_match_2_15_3: 80 switch_nic_out_15_nic_in_3 + group_start_3
   - group_end_15 <= 80
 constr_link_match_1_15_4: 80 switch_nic_out_15_nic_in_4 - group_start_4
   + group_end_15 <= 80
 constr_link_match_2_15_4: 80 switch_nic_out_15_nic_in_4 + group_start_4
   - group_end_15 <= 80
 constr_link_match_1_15_5: 80 switch_nic_out_15_nic_in_5 - group_start_5
   + group_end_15 <= 80
 constr_link_match_2_15_5: 80 switch_nic_out_15_nic_in_5 + group_start_5
   - group_end_15 <= 80
 constr_link_match_1_15_6: 80 switch_nic_out_15_nic_in_6 - group_start_6
   + group_end_15 <= 80
 constr_link_match_2_15_6: 80 switch_nic_out_15_nic_in_6 + group_start_6
   - group_end_15 <= 80
 constr_link_match_1_15_7: 80 switch_nic_out_15_nic_in_7 - group_start_7
   + group_end_15 <= 80
 constr_link_match_2_15_7: 80 switch_nic_out_15_nic_in_7 + group_start_7
   - group_end_15 <= 80
 constr_link_match_1_15_8: 80 switch_nic_out_15_nic_in_8 - group_start_8
   + group_end_15 <= 80
 constr_link_match_2_15_8: 80 switch_nic_out_15_nic_in_8 + group_start_8
   - group_end_15 <= 80
 constr_link_match_1_15_9: 80 switch_nic_out_15_nic_in_9 - group_start_9
   + group_end_15 <= 80
 constr_link_match_2_15_9: 80 switch_nic_out_15_nic_in_9 + group_start_9
   - group_end_15 <= 80
 constr_link_match_1_15_10: 80 switch_nic_out_15_nic_in_10 - group_start_10
   + group_end_15 <= 80
 constr_link_match_2_15_10: 80 switch_nic_out_15_nic_in_10 + group_start_10
   - group_end_15 <= 80
 constr_link_match_1_15_11: 80 switch_nic_out_15_nic_in_11 - group_start_11
   + group_end_15 <= 80
 constr_link_match_2_15_11: 80 switch_nic_out_15_nic_in_11 + group_start_11
   - group_end_15 <= 80
 constr_link_match_1_15_12: 80 switch_nic_out_15_nic_in_12 - group_start_12
   + group_end_15 <= 80
 constr_link_match_2_15_12: 80 switch_nic_out_15_nic_in_12 + group_start_12
   - group_end_15 <= 80
 constr_link_match_1_15_13: 80 switch_nic_out_15_nic_in_13 - group_start_13
   + group_end_15 <= 80
 constr_link_match_2_15_13: 80 switch_nic_out_15_nic_in_13 + group_start_13
   - group_end_15 <= 80
 constr_link_match_1_15_14: 80 switch_nic_out_15_nic_in_14 - group_start_14
   + group_end_15 <= 80
 constr_link_match_2_15_14: 80 switch_nic_out_15_nic_in_14 + group_start_14
   - group_end_15 <= 80
 constr_link_match_1_15_16: 80 switch_nic_out_15_nic_in_16 - group_start_16
   + group_end_15 <= 80
 constr_link_match_2_15_16: 80 switch_nic_out_15_nic_in_16 + group_start_16
   - group_end_15 <= 80
 constr_link_match_1_15_17: 80 switch_nic_out_15_nic_in_17 - group_start_17
   + group_end_15 <= 80
 constr_link_match_2_15_17: 80 switch_nic_out_15_nic_in_17 + group_start_17
   - group_end_15 <= 80
 constr_link_match_1_15_18: 80 switch_nic_out_15_nic_in_18 - group_start_18
   + group_end_15 <= 80
 constr_link_match_2_15_18: 80 switch_nic_out_15_nic_in_18 + group_start_18
   - group_end_15 <= 80
 constr_link_match_1_15_19: 80 switch_nic_out_15_nic_in_19 - group_start_19
   + group_end_15 <= 80
 constr_link_match_2_15_19: 80 switch_nic_out_15_nic_in_19 + group_start_19
   - group_end_15 <= 80
 constr_link_match_1_15_20: 80 switch_nic_out_15_nic_in_20 - group_start_20
   + group_end_15 <= 80
 constr_link_match_2_15_20: 80 switch_nic_out_15_nic_in_20 + group_start_20
   - group_end_15 <= 80
 constr_link_match_1_15_21: 80 switch_nic_out_15_nic_in_21 - group_start_21
   + group_end_15 <= 80
 constr_link_match_2_15_21: 80 switch_nic_out_15_nic_in_21 + group_start_21
   - group_end_15 <= 80
 constr_link_match_1_15_22: 80 switch_nic_out_15_nic_in_22 - group_start_22
   + group_end_15 <= 80
 constr_link_match_2_15_22: 80 switch_nic_out_15_nic_in_22 + group_start_22
   - group_end_15 <= 80
 constr_link_match_1_15_23: 80 switch_nic_out_15_nic_in_23 - group_start_23
   + group_end_15 <= 80
 constr_link_match_2_15_23: 80 switch_nic_out_15_nic_in_23 + group_start_23
   - group_end_15 <= 80
 constr_link_match_1_15_24: 80 switch_nic_out_15_nic_in_24 - group_start_24
   + group_end_15 <= 80
 constr_link_match_2_15_24: 80 switch_nic_out_15_nic_in_24 + group_start_24
   - group_end_15 <= 80
 constr_link_match_1_15_25: 80 switch_nic_out_15_nic_in_25 - group_start_25
   + group_end_15 <= 80
 constr_link_match_2_15_25: 80 switch_nic_out_15_nic_in_25 + group_start_25
   - group_end_15 <= 80
 constr_link_match_1_15_26: 80 switch_nic_out_15_nic_in_26 - group_start_26
   + group_end_15 <= 80
 constr_link_match_2_15_26: 80 switch_nic_out_15_nic_in_26 + group_start_26
   - group_end_15 <= 80
 constr_link_match_1_15_27: 80 switch_nic_out_15_nic_in_27 - group_start_27
   + group_end_15 <= 80
 constr_link_match_2_15_27: 80 switch_nic_out_15_nic_in_27 + group_start_27
   - group_end_15 <= 80
 constr_link_match_1_15_28: 80 switch_nic_out_15_nic_in_28 - group_start_28
   + group_end_15 <= 80
 constr_link_match_2_15_28: 80 switch_nic_out_15_nic_in_28 + group_start_28
   - group_end_15 <= 80
 constr_link_match_1_15_29: 80 switch_nic_out_15_nic_in_29 - group_start_29
   + group_end_15 <= 80
 constr_link_match_2_15_29: 80 switch_nic_out_15_nic_in_29 + group_start_29
   - group_end_15 <= 80
 constr_link_match_1_15_30: 80 switch_nic_out_15_nic_in_30 - group_start_30
   + group_end_15 <= 80
 constr_link_match_2_15_30: 80 switch_nic_out_15_nic_in_30 + group_start_30
   - group_end_15 <= 80
 constr_link_match_1_15_31: 80 switch_nic_out_15_nic_in_31 - group_start_31
   + group_end_15 <= 80
 constr_link_match_2_15_31: 80 switch_nic_out_15_nic_in_31 + group_start_31
   - group_end_15 <= 80
 constr_is_start_zero_16: start_16
   + 80 is_start_zero_16_for_source_nic_in_16 <= 80
 constr_edge_switch_start_zero_16_OR: switch_source_nic_in_16
   - is_start_zero_16_for_source_nic_in_16 <= 0
 constr_is_end_complete_16: start_16 + hold_16_1 + 2 hold_16_2
   + 3 hold_16_3 + 4 hold_16_4 + 5 hold_16_5 + 6 hold_16_6
   - 80 is_end_complete_16 >= 0
 constr_edge_switch_end_max_16_OR: switch_nic_out_16_sink
   - is_end_complete_16 <= 0
 constr_link_match_1_16_0: 80 switch_nic_out_16_nic_in_0 - group_start_0
   + group_end_16 <= 80
 constr_link_match_2_16_0: 80 switch_nic_out_16_nic_in_0 + group_start_0
   - group_end_16 <= 80
 constr_link_match_1_16_1: 80 switch_nic_out_16_nic_in_1 - group_start_1
   + group_end_16 <= 80
 constr_link_match_2_16_1: 80 switch_nic_out_16_nic_in_1 + group_start_1
   - group_end_16 <= 80
 constr_link_match_1_16_2: 80 switch_nic_out_16_nic_in_2 - group_start_2
   + group_end_16 <= 80
 constr_link_match_2_16_2: 80 switch_nic_out_16_nic_in_2 + group_start_2
   - group_end_16 <= 80
 constr_link_match_1_16_3: 80 switch_nic_out_16_nic_in_3 - group_start_3
   + group_end_16 <= 80
 constr_link_match_2_16_3: 80 switch_nic_out_16_nic_in_3 + group_start_3
   - group_end_16 <= 80
 constr_link_match_1_16_4: 80 switch_nic_out_16_nic_in_4 - group_start_4
   + group_end_16 <= 80
 constr_link_match_2_16_4: 80 switch_nic_out_16_nic_in_4 + group_start_4
   - group_end_16 <= 80
 constr_link_match_1_16_5: 80 switch_nic_out_16_nic_in_5 - group_start_5
   + group_end_16 <= 80
 constr_link_match_2_16_5: 80 switch_nic_out_16_nic_in_5 + group_start_5
   - group_end_16 <= 80
 constr_link_match_1_16_6: 80 switch_nic_out_16_nic_in_6 - group_start_6
   + group_end_16 <= 80
 constr_link_match_2_16_6: 80 switch_nic_out_16_nic_in_6 + group_start_6
   - group_end_16 <= 80
 constr_link_match_1_16_7: 80 switch_nic_out_16_nic_in_7 - group_start_7
   + group_end_16 <= 80
 constr_link_match_2_16_7: 80 switch_nic_out_16_nic_in_7 + group_start_7
   - group_end_16 <= 80
 constr_link_match_1_16_8: 80 switch_nic_out_16_nic_in_8 - group_start_8
   + group_end_16 <= 80
 constr_link_match_2_16_8: 80 switch_nic_out_16_nic_in_8 + group_start_8
   - group_end_16 <= 80
 constr_link_match_1_16_9: 80 switch_nic_out_16_nic_in_9 - group_start_9
   + group_end_16 <= 80
 constr_link_match_2_16_9: 80 switch_nic_out_16_nic_in_9 + group_start_9
   - group_end_16 <= 80
 constr_link_match_1_16_10: 80 switch_nic_out_16_nic_in_10 - group_start_10
   + group_end_16 <= 80
 constr_link_match_2_16_10: 80 switch_nic_out_16_nic_in_10 + group_start_10
   - group_end_16 <= 80
 constr_link_match_1_16_11: 80 switch_nic_out_16_nic_in_11 - group_start_11
   + group_end_16 <= 80
 constr_link_match_2_16_11: 80 switch_nic_out_16_nic_in_11 + group_start_11
   - group_end_16 <= 80
 constr_link_match_1_16_12: 80 switch_nic_out_16_nic_in_12 - group_start_12
   + group_end_16 <= 80
 constr_link_match_2_16_12: 80 switch_nic_out_16_nic_in_12 + group_start_12
   - group_end_16 <= 80
 constr_link_match_1_16_13: 80 switch_nic_out_16_nic_in_13 - group_start_13
   + group_end_16 <= 80
 constr_link_match_2_16_13: 80 switch_nic_out_16_nic_in_13 + group_start_13
   - group_end_16 <= 80
 constr_link_match_1_16_14: 80 switch_nic_out_16_nic_in_14 - group_start_14
   + group_end_16 <= 80
 constr_link_match_2_16_14: 80 switch_nic_out_16_nic_in_14 + group_start_14
   - group_end_16 <= 80
 constr_link_match_1_16_15: 80 switch_nic_out_16_nic_in_15 - group_start_15
   + group_end_16 <= 80
 constr_link_match_2_16_15: 80 switch_nic_out_16_nic_in_15 + group_start_15
   - group_end_16 <= 80
 constr_link_match_1_16_17: 80 switch_nic_out_16_nic_in_17 - group_start_17
   + group_end_16 <= 80
 constr_link_match_2_16_17: 80 switch_nic_out_16_nic_in_17 + group_start_17
   - group_end_16 <= 80
 constr_link_match_1_16_18: 80 switch_nic_out_16_nic_in_18 - group_start_18
   + group_end_16 <= 80
 constr_link_match_2_16_18: 80 switch_nic_out_16_nic_in_18 + group_start_18
   - group_end_16 <= 80
 constr_link_match_1_16_19: 80 switch_nic_out_16_nic_in_19 - group_start_19
   + group_end_16 <= 80
 constr_link_match_2_16_19: 80 switch_nic_out_16_nic_in_19 + group_start_19
   - group_end_16 <= 80
 constr_link_match_1_16_20: 80 switch_nic_out_16_nic_in_20 - group_start_20
   + group_end_16 <= 80
 constr_link_match_2_16_20: 80 switch_nic_out_16_nic_in_20 + group_start_20
   - group_end_16 <= 80
 constr_link_match_1_16_21: 80 switch_nic_out_16_nic_in_21 - group_start_21
   + group_end_16 <= 80
 constr_link_match_2_16_21: 80 switch_nic_out_16_nic_in_21 + group_start_21
   - group_end_16 <= 80
 constr_link_match_1_16_22: 80 switch_nic_out_16_nic_in_22 - group_start_22
   + group_end_16 <= 80
 constr_link_match_2_16_22: 80 switch_nic_out_16_nic_in_22 + group_start_22
   - group_end_16 <= 80
 constr_link_match_1_16_23: 80 switch_nic_out_16_nic_in_23 - group_start_23
   + group_end_16 <= 80
 constr_link_match_2_16_23: 80 switch_nic_out_16_nic_in_23 + group_start_23
   - group_end_16 <= 80
 constr_link_match_1_16_24: 80 switch_nic_out_16_nic_in_24 - group_start_24
   + group_end_16 <= 80
 constr_link_match_2_16_24: 80 switch_nic_out_16_nic_in_24 + group_start_24
   - group_end_16 <= 80
 constr_link_match_1_16_25: 80 switch_nic_out_16_nic_in_25 - group_start_25
   + group_end_16 <= 80
 constr_link_match_2_16_25: 80 switch_nic_out_16_nic_in_25 + group_start_25
   - group_end_16 <= 80
 constr_link_match_1_16_26: 80 switch_nic_out_16_nic_in_26 - group_start_26
   + group_end_16 <= 80
 constr_link_match_2_16_26: 80 switch_nic_out_16_nic_in_26 + group_start_26
   - group_end_16 <= 80
 constr_link_match_1_16_27: 80 switch_nic_out_16_nic_in_27 - group_start_27
   + group_end_16 <= 80
 constr_link_match_2_16_27: 80 switch_nic_out_16_nic_in_27 + group_start_27
   - group_end_16 <= 80
 constr_link_match_1_16_28: 80 switch_nic_out_16_nic_in_28 - group_start_28
   + group_end_16 <= 80
 constr_link_match_2_16_28: 80 switch_nic_out_16_nic_in_28 + group_start_28
   - group_end_16 <= 80
 constr_link_match_1_16_29: 80 switch_nic_out_16_nic_in_29 - group_start_29
   + group_end_16 <= 80
 constr_link_match_2_16_29: 80 switch_nic_out_16_nic_in_29 + group_start_29
   - group_end_16 <= 80
 constr_link_match_1_16_30: 80 switch_nic_out_16_nic_in_30 - group_start_30
   + group_end_16 <= 80
 constr_link_match_2_16_30: 80 switch_nic_out_16_nic_in_30 + group_start_30
   - group_end_16 <= 80
 constr_link_match_1_16_31: 80 switch_nic_out_16_nic_in_31 - group_start_31
   + group_end_16 <= 80
 constr_link_match_2_16_31: 80 switch_nic_out_16_nic_in_31 + group_start_31
   - group_end_16 <= 80
 constr_is_start_zero_17: start_17
   + 80 is_start_zero_17_for_source_nic_in_17 <= 80
 constr_edge_switch_start_zero_17_OR: switch_source_nic_in_17
   - is_start_zero_17_for_source_nic_in_17 <= 0
 constr_is_end_complete_17: start_17 + hold_17_1 + 2 hold_17_2
   + 3 hold_17_3 + 4 hold_17_4 + 5 hold_17_5 + 6 hold_17_6
   - 80 is_end_complete_17 >= 0
 constr_edge_switch_end_max_17_OR: switch_nic_out_17_sink
   - is_end_complete_17 <= 0
 constr_link_match_1_17_0: 80 switch_nic_out_17_nic_in_0 - group_start_0
   + group_end_17 <= 80
 constr_link_match_2_17_0: 80 switch_nic_out_17_nic_in_0 + group_start_0
   - group_end_17 <= 80
 constr_link_match_1_17_1: 80 switch_nic_out_17_nic_in_1 - group_start_1
   + group_end_17 <= 80
 constr_link_match_2_17_1: 80 switch_nic_out_17_nic_in_1 + group_start_1
   - group_end_17 <= 80
 constr_link_match_1_17_2: 80 switch_nic_out_17_nic_in_2 - group_start_2
   + group_end_17 <= 80
 constr_link_match_2_17_2: 80 switch_nic_out_17_nic_in_2 + group_start_2
   - group_end_17 <= 80
 constr_link_match_1_17_3: 80 switch_nic_out_17_nic_in_3 - group_start_3
   + group_end_17 <= 80
 constr_link_match_2_17_3: 80 switch_nic_out_17_nic_in_3 + group_start_3
   - group_end_17 <= 80
 constr_link_match_1_17_4: 80 switch_nic_out_17_nic_in_4 - group_start_4
   + group_end_17 <= 80
 constr_link_match_2_17_4: 80 switch_nic_out_17_nic_in_4 + group_start_4
   - group_end_17 <= 80
 constr_link_match_1_17_5: 80 switch_nic_out_17_nic_in_5 - group_start_5
   + group_end_17 <= 80
 constr_link_match_2_17_5: 80 switch_nic_out_17_nic_in_5 + group_start_5
   - group_end_17 <= 80
 constr_link_match_1_17_6: 80 switch_nic_out_17_nic_in_6 - group_start_6
   + group_end_17 <= 80
 constr_link_match_2_17_6: 80 switch_nic_out_17_nic_in_6 + group_start_6
   - group_end_17 <= 80
 constr_link_match_1_17_7: 80 switch_nic_out_17_nic_in_7 - group_start_7
   + group_end_17 <= 80
 constr_link_match_2_17_7: 80 switch_nic_out_17_nic_in_7 + group_start_7
   - group_end_17 <= 80
 constr_link_match_1_17_8: 80 switch_nic_out_17_nic_in_8 - group_start_8
   + group_end_17 <= 80
 constr_link_match_2_17_8: 80 switch_nic_out_17_nic_in_8 + group_start_8
   - group_end_17 <= 80
 constr_link_match_1_17_9: 80 switch_nic_out_17_nic_in_9 - group_start_9
   + group_end_17 <= 80
 constr_link_match_2_17_9: 80 switch_nic_out_17_nic_in_9 + group_start_9
   - group_end_17 <= 80
 constr_link_match_1_17_10: 80 switch_nic_out_17_nic_in_10 - group_start_10
   + group_end_17 <= 80
 constr_link_match_2_17_10: 80 switch_nic_out_17_nic_in_10 + group_start_10
   - group_end_17 <= 80
 constr_link_match_1_17_11: 80 switch_nic_out_17_nic_in_11 - group_start_11
   + group_end_17 <= 80
 constr_link_match_2_17_11: 80 switch_nic_out_17_nic_in_11 + group_start_11
   - group_end_17 <= 80
 constr_link_match_1_17_12: 80 switch_nic_out_17_nic_in_12 - group_start_12
   + group_end_17 <= 80
 constr_link_match_2_17_12: 80 switch_nic_out_17_nic_in_12 + group_start_12
   - group_end_17 <= 80
 constr_link_match_1_17_13: 80 switch_nic_out_17_nic_in_13 - group_start_13
   + group_end_17 <= 80
 constr_link_match_2_17_13: 80 switch_nic_out_17_nic_in_13 + group_start_13
   - group_end_17 <= 80
 constr_link_match_1_17_14: 80 switch_nic_out_17_nic_in_14 - group_start_14
   + group_end_17 <= 80
 constr_link_match_2_17_14: 80 switch_nic_out_17_nic_in_14 + group_start_14
   - group_end_17 <= 80
 constr_link_match_1_17_15: 80 switch_nic_out_17_nic_in_15 - group_start_15
   + group_end_17 <= 80
 constr_link_match_2_17_15: 80 switch_nic_out_17_nic_in_15 + group_start_15
   - group_end_17 <= 80
 constr_link_match_1_17_16: 80 switch_nic_out_17_nic_in_16 - group_start_16
   + group_end_17 <= 80
 constr_link_match_2_17_16: 80 switch_nic_out_17_nic_in_16 + group_start_16
   - group_end_17 <= 80
 constr_link_match_1_17_18: 80 switch_nic_out_17_nic_in_18 - group_start_18
   + group_end_17 <= 80
 constr_link_match_2_17_18: 80 switch_nic_out_17_nic_in_18 + group_start_18
   - group_end_17 <= 80
 constr_link_match_1_17_19: 80 switch_nic_out_17_nic_in_19 - group_start_19
   + group_end_17 <= 80
 constr_link_match_2_17_19: 80 switch_nic_out_17_nic_in_19 + group_start_19
   - group_end_17 <= 80
 constr_link_match_1_17_20: 80 switch_nic_out_17_nic_in_20 - group_start_20
   + group_end_17 <= 80
 constr_link_match_2_17_20: 80 switch_nic_out_17_nic_in_20 + group_start_20
   - group_end_17 <= 80
 constr_link_match_1_17_21: 80 switch_nic_out_17_nic_in_21 - group_start_21
   + group_end_17 <= 80
 constr_link_match_2_17_21: 80 switch_nic_out_17_nic_in_21 + group_start_21
   - group_end_17 <= 80
 constr_link_match_1_17_22: 80 switch_nic_out_17_nic_in_22 - group_start_22
   + group_end_17 <= 80
 constr_link_match_2_17_22: 80 switch_nic_out_17_nic_in_22 + group_start_22
   - group_end_17 <= 80
 constr_link_match_1_17_23: 80 switch_nic_out_17_nic_in_23 - group_start_23
   + group_end_17 <= 80
 constr_link_match_2_17_23: 80 switch_nic_out_17_nic_in_23 + group_start_23
   - group_end_17 <= 80
 constr_link_match_1_17_24: 80 switch_nic_out_17_nic_in_24 - group_start_24
   + group_end_17 <= 80
 constr_link_match_2_17_24: 80 switch_nic_out_17_nic_in_24 + group_start_24
   - group_end_17 <= 80
 constr_link_match_1_17_25: 80 switch_nic_out_17_nic_in_25 - group_start_25
   + group_end_17 <= 80
 constr_link_match_2_17_25: 80 switch_nic_out_17_nic_in_25 + group_start_25
   - group_end_17 <= 80
 constr_link_match_1_17_26: 80 switch_nic_out_17_nic_in_26 - group_start_26
   + group_end_17 <= 80
 constr_link_match_2_17_26: 80 switch_nic_out_17_nic_in_26 + group_start_26
   - group_end_17 <= 80
 constr_link_match_1_17_27: 80 switch_nic_out_17_nic_in_27 - group_start_27
   + group_end_17 <= 80
 constr_link_match_2_17_27: 80 switch_nic_out_17_nic_in_27 + group_start_27
   - group_end_17 <= 80
 constr_link_match_1_17_28: 80 switch_nic_out_17_nic_in_28 - group_start_28
   + group_end_17 <= 80
 constr_link_match_2_17_28: 80 switch_nic_out_17_nic_in_28 + group_start_28
   - group_end_17 <= 80
 constr_link_match_1_17_29: 80 switch_nic_out_17_nic_in_29 - group_start_29
   + group_end_17 <= 80
 constr_link_match_2_17_29: 80 switch_nic_out_17_nic_in_29 + group_start_29
   - group_end_17 <= 80
 constr_link_match_1_17_30: 80 switch_nic_out_17_nic_in_30 - group_start_30
   + group_end_17 <= 80
 constr_link_match_2_17_30: 80 switch_nic_out_17_nic_in_30 + group_start_30
   - group_end_17 <= 80
 constr_link_match_1_17_31: 80 switch_nic_out_17_nic_in_31 - group_start_31
   + group_end_17 <= 80
 constr_link_match_2_17_31: 80 switch_nic_out_17_nic_in_31 + group_start_31
   - group_end_17 <= 80
 constr_is_start_zero_18: start_18
   + 80 is_start_zero_18_for_source_nic_in_18 <= 80
 constr_edge_switch_start_zero_18_OR: switch_source_nic_in_18
   - is_start_zero_18_for_source_nic_in_18 <= 0
 constr_is_end_complete_18: start_18 + hold_18_1 + 2 hold_18_2
   + 3 hold_18_3 + 4 hold_18_4 + 5 hold_18_5 + 6 hold_18_6
   - 80 is_end_complete_18 >= 0
 constr_edge_switch_end_max_18_OR: switch_nic_out_18_sink
   - is_end_complete_18 <= 0
 constr_link_match_1_18_0: 80 switch_nic_out_18_nic_in_0 - group_start_0
   + group_end_18 <= 80
 constr_link_match_2_18_0: 80 switch_nic_out_18_nic_in_0 + group_start_0
   - group_end_18 <= 80
 constr_link_match_1_18_1: 80 switch_nic_out_18_nic_in_1 - group_start_1
   + group_end_18 <= 80
 constr_link_match_2_18_1: 80 switch_nic_out_18_nic_in_1 + group_start_1
   - group_end_18 <= 80
 constr_link_match_1_18_2: 80 switch_nic_out_18_nic_in_2 - group_start_2
   + group_end_18 <= 80
 constr_link_match_2_18_2: 80 switch_nic_out_18_nic_in_2 + group_start_2
   - group_end_18 <= 80
 constr_link_match_1_18_3: 80 switch_nic_out_18_nic_in_3 - group_start_3
   + group_end_18 <= 80
 constr_link_match_2_18_3: 80 switch_nic_out_18_nic_in_3 + group_start_3
   - group_end_18 <= 80
 constr_link_match_1_18_4: 80 switch_nic_out_18_nic_in_4 - group_start_4
   + group_end_18 <= 80
 constr_link_match_2_18_4: 80 switch_nic_out_18_nic_in_4 + group_start_4
   - group_end_18 <= 80
 constr_link_match_1_18_5: 80 switch_nic_out_18_nic_in_5 - group_start_5
   + group_end_18 <= 80
 constr_link_match_2_18_5: 80 switch_nic_out_18_nic_in_5 + group_start_5
   - group_end_18 <= 80
 constr_link_match_1_18_6: 80 switch_nic_out_18_nic_in_6 - group_start_6
   + group_end_18 <= 80
 constr_link_match_2_18_6: 80 switch_nic_out_18_nic_in_6 + group_start_6
   - group_end_18 <= 80
 constr_link_match_1_18_7: 80 switch_nic_out_18_nic_in_7 - group_start_7
   + group_end_18 <= 80
 constr_link_match_2_18_7: 80 switch_nic_out_18_nic_in_7 + group_start_7
   - group_end_18 <= 80
 constr_link_match_1_18_8: 80 switch_nic_out_18_nic_in_8 - group_start_8
   + group_end_18 <= 80
 constr_link_match_2_18_8: 80 switch_nic_out_18_nic_in_8 + group_start_8
   - group_end_18 <= 80
 constr_link_match_1_18_9: 80 switch_nic_out_18_nic_in_9 - group_start_9
   + group_end_18 <= 80
 constr_link_match_2_18_9: 80 switch_nic_out_18_nic_in_9 + group_start_9
   - group_end_18 <= 80
 constr_link_match_1_18_10: 80 switch_nic_out_18_nic_in_10 - group_start_10
   + group_end_18 <= 80
 constr_link_match_2_18_10: 80 switch_nic_out_18_nic_in_10 + group_start_10
   - group_end_18 <= 80
 constr_link_match_1_18_11: 80 switch_nic_out_18_nic_in_11 - group_start_11
   + group_end_18 <= 80
 constr_link_match_2_18_11: 80 switch_nic_out_18_nic_in_11 + group_start_11
   - group_end_18 <= 80
 constr_link_match_1_18_12: 80 switch_nic_out_18_nic_in_12 - group_start_12
   + group_end_18 <= 80
 constr_link_match_2_18_12: 80 switch_nic_out_18_nic_in_12 + group_start_12
   - group_end_18 <= 80
 constr_link_match_1_18_13: 80 switch_nic_out_18_nic_in_13 - group_start_13
   + group_end_18 <= 80
 constr_link_match_2_18_13: 80 switch_nic_out_18_nic_in_13 + group_start_13
   - group_end_18 <= 80
 constr_link_match_1_18_14: 80 switch_nic_out_18_nic_in_14 - group_start_14
   + group_end_18 <= 80
 constr_link_match_2_18_14: 80 switch_nic_out_18_nic_in_14 + group_start_14
   - group_end_18 <= 80
 constr_link_match_1_18_15: 80 switch_nic_out_18_nic_in_15 - group_start_15
   + group_end_18 <= 80
 constr_link_match_2_18_15: 80 switch_nic_out_18_nic_in_15 + group_start_15
   - group_end_18 <= 80
 constr_link_match_1_18_16: 80 switch_nic_out_18_nic_in_16 - group_start_16
   + group_end_18 <= 80
 constr_link_match_2_18_16: 80 switch_nic_out_18_nic_in_16 + group_start_16
   - group_end_18 <= 80
 constr_link_match_1_18_17: 80 switch_nic_out_18_nic_in_17 - group_start_17
   + group_end_18 <= 80
 constr_link_match_2_18_17: 80 switch_nic_out_18_nic_in_17 + group_start_17
   - group_end_18 <= 80
 constr_link_match_1_18_19: 80 switch_nic_out_18_nic_in_19 - group_start_19
   + group_end_18 <= 80
 constr_link_match_2_18_19: 80 switch_nic_out_18_nic_in_19 + group_start_19
   - group_end_18 <= 80
 constr_link_match_1_18_20: 80 switch_nic_out_18_nic_in_20 - group_start_20
   + group_end_18 <= 80
 constr_link_match_2_18_20: 80 switch_nic_out_18_nic_in_20 + group_start_20
   - group_end_18 <= 80
 constr_link_match_1_18_21: 80 switch_nic_out_18_nic_in_21 - group_start_21
   + group_end_18 <= 80
 constr_link_match_2_18_21: 80 switch_nic_out_18_nic_in_21 + group_start_21
   - group_end_18 <= 80
 constr_link_match_1_18_22: 80 switch_nic_out_18_nic_in_22 - group_start_22
   + group_end_18 <= 80
 constr_link_match_2_18_22: 80 switch_nic_out_18_nic_in_22 + group_start_22
   - group_end_18 <= 80
 constr_link_match_1_18_23: 80 switch_nic_out_18_nic_in_23 - group_start_23
   + group_end_18 <= 80
 constr_link_match_2_18_23: 80 switch_nic_out_18_nic_in_23 + group_start_23
   - group_end_18 <= 80
 constr_link_match_1_18_24: 80 switch_nic_out_18_nic_in_24 - group_start_24
   + group_end_18 <= 80
 constr_link_match_2_18_24: 80 switch_nic_out_18_nic_in_24 + group_start_24
   - group_end_18 <= 80
 constr_link_match_1_18_25: 80 switch_nic_out_18_nic_in_25 - group_start_25
   + group_end_18 <= 80
 constr_link_match_2_18_25: 80 switch_nic_out_18_nic_in_25 + group_start_25
   - group_end_18 <= 80
 constr_link_match_1_18_26: 80 switch_nic_out_18_nic_in_26 - group_start_26
   + group_end_18 <= 80
 constr_link_match_2_18_26: 80 switch_nic_out_18_nic_in_26 + group_start_26
   - group_end_18 <= 80
 constr_link_match_1_18_27: 80 switch_nic_out_18_nic_in_27 - group_start_27
   + group_end_18 <= 80
 constr_link_match_2_18_27: 80 switch_nic_out_18_nic_in_27 + group_start_27
   - group_end_18 <= 80
 constr_link_match_1_18_28: 80 switch_nic_out_18_nic_in_28 - group_start_28
   + group_end_18 <= 80
 constr_link_match_2_18_28: 80 switch_nic_out_18_nic_in_28 + group_start_28
   - group_end_18 <= 80
 constr_link_match_1_18_29: 80 switch_nic_out_18_nic_in_29 - group_start_29
   + group_end_18 <= 80
 constr_link_match_2_18_29: 80 switch_nic_out_18_nic_in_29 + group_start_29
   - group_end_18 <= 80
 constr_link_match_1_18_30: 80 switch_nic_out_18_nic_in_30 - group_start_30
   + group_end_18 <= 80
 constr_link_match_2_18_30: 80 switch_nic_out_18_nic_in_30 + group_start_30
   - group_end_18 <= 80
 constr_link_match_1_18_31: 80 switch_nic_out_18_nic_in_31 - group_start_31
   + group_end_18 <= 80
 constr_link_match_2_18_31: 80 switch_nic_out_18_nic_in_31 + group_start_31
   - group_end_18 <= 80
 constr_is_start_zero_19: start_19
   + 80 is_start_zero_19_for_source_nic_in_19 <= 80
 constr_edge_switch_start_zero_19_OR: switch_source_nic_in_19
   - is_start_zero_19_for_source_nic_in_19 <= 0
 constr_is_end_complete_19: start_19 + hold_19_1 + 2 hold_19_2
   + 3 hold_19_3 + 4 hold_19_4 + 5 hold_19_5 + 6 hold_19_6
   - 80 is_end_complete_19 >= 0
 constr_edge_switch_end_max_19_OR: switch_nic_out_19_sink
   - is_end_complete_19 <= 0
 constr_link_match_1_19_0: 80 switch_nic_out_19_nic_in_0 - group_start_0
   + group_end_19 <= 80
 constr_link_match_2_19_0: 80 switch_nic_out_19_nic_in_0 + group_start_0
   - group_end_19 <= 80
 constr_link_match_1_19_1: 80 switch_nic_out_19_nic_in_1 - group_start_1
   + group_end_19 <= 80
 constr_link_match_2_19_1: 80 switch_nic_out_19_nic_in_1 + group_start_1
   - group_end_19 <= 80
 constr_link_match_1_19_2: 80 switch_nic_out_19_nic_in_2 - group_start_2
   + group_end_19 <= 80
 constr_link_match_2_19_2: 80 switch_nic_out_19_nic_in_2 + group_start_2
   - group_end_19 <= 80
 constr_link_match_1_19_3: 80 switch_nic_out_19_nic_in_3 - group_start_3
   + group_end_19 <= 80
 constr_link_match_2_19_3: 80 switch_nic_out_19_nic_in_3 + group_start_3
   - group_end_19 <= 80
 constr_link_match_1_19_4: 80 switch_nic_out_19_nic_in_4 - group_start_4
   + group_end_19 <= 80
 constr_link_match_2_19_4: 80 switch_nic_out_19_nic_in_4 + group_start_4
   - group_end_19 <= 80
 constr_link_match_1_19_5: 80 switch_nic_out_19_nic_in_5 - group_start_5
   + group_end_19 <= 80
 constr_link_match_2_19_5: 80 switch_nic_out_19_nic_in_5 + group_start_5
   - group_end_19 <= 80
 constr_link_match_1_19_6: 80 switch_nic_out_19_nic_in_6 - group_start_6
   + group_end_19 <= 80
 constr_link_match_2_19_6: 80 switch_nic_out_19_nic_in_6 + group_start_6
   - group_end_19 <= 80
 constr_link_match_1_19_7: 80 switch_nic_out_19_nic_in_7 - group_start_7
   + group_end_19 <= 80
 constr_link_match_2_19_7: 80 switch_nic_out_19_nic_in_7 + group_start_7
   - group_end_19 <= 80
 constr_link_match_1_19_8: 80 switch_nic_out_19_nic_in_8 - group_start_8
   + group_end_19 <= 80
 constr_link_match_2_19_8: 80 switch_nic_out_19_nic_in_8 + group_start_8
   - group_end_19 <= 80
 constr_link_match_1_19_9: 80 switch_nic_out_19_nic_in_9 - group_start_9
   + group_end_19 <= 80
 constr_link_match_2_19_9: 80 switch_nic_out_19_nic_in_9 + group_start_9
   - group_end_19 <= 80
 constr_link_match_1_19_10: 80 switch_nic_out_19_nic_in_10 - group_start_10
   + group_end_19 <= 80
 constr_link_match_2_19_10: 80 switch_nic_out_19_nic_in_10 + group_start_10
   - group_end_19 <= 80
 constr_link_match_1_19_11: 80 switch_nic_out_19_nic_in_11 - group_start_11
   + group_end_19 <= 80
 constr_link_match_2_19_11: 80 switch_nic_out_19_nic_in_11 + group_start_11
   - group_end_19 <= 80
 constr_link_match_1_19_12: 80 switch_nic_out_19_nic_in_12 - group_start_12
   + group_end_19 <= 80
 constr_link_match_2_19_12: 80 switch_nic_out_19_nic_in_12 + group_start_12
   - group_end_19 <= 80
 constr_link_match_1_19_13: 80 switch_nic_out_19_nic_in_13 - group_start_13
   + group_end_19 <= 80
 constr_link_match_2_19_13: 80 switch_nic_out_19_nic_in_13 + group_start_13
   - group_end_19 <= 80
 constr_link_match_1_19_14: 80 switch_nic_out_19_nic_in_14 - group_start_14
   + group_end_19 <= 80
 constr_link_match_2_19_14: 80 switch_nic_out_19_nic_in_14 + group_start_14
   - group_end_19 <= 80
 constr_link_match_1_19_15: 80 switch_nic_out_19_nic_in_15 - group_start_15
   + group_end_19 <= 80
 constr_link_match_2_19_15: 80 switch_nic_out_19_nic_in_15 + group_start_15
   - group_end_19 <= 80
 constr_link_match_1_19_16: 80 switch_nic_out_19_nic_in_16 - group_start_16
   + group_end_19 <= 80
 constr_link_match_2_19_16: 80 switch_nic_out_19_nic_in_16 + group_start_16
   - group_end_19 <= 80
 constr_link_match_1_19_17: 80 switch_nic_out_19_nic_in_17 - group_start_17
   + group_end_19 <= 80
 constr_link_match_2_19_17: 80 switch_nic_out_19_nic_in_17 + group_start_17
   - group_end_19 <= 80
 constr_link_match_1_19_18: 80 switch_nic_out_19_nic_in_18 - group_start_18
   + group_end_19 <= 80
 constr_link_match_2_19_18: 80 switch_nic_out_19_nic_in_18 + group_start_18
   - group_end_19 <= 80
 constr_link_match_1_19_20: 80 switch_nic_out_19_nic_in_20 - group_start_20
   + group_end_19 <= 80
 constr_link_match_2_19_20: 80 switch_nic_out_19_nic_in_20 + group_start_20
   - group_end_19 <= 80
 constr_link_match_1_19_21: 80 switch_nic_out_19_nic_in_21 - group_start_21
   + group_end_19 <= 80
 constr_link_match_2_19_21: 80 switch_nic_out_19_nic_in_21 + group_start_21
   - group_end_19 <= 80
 constr_link_match_1_19_22: 80 switch_nic_out_19_nic_in_22 - group_start_22
   + group_end_19 <= 80
 constr_link_match_2_19_22: 80 switch_nic_out_19_nic_in_22 + group_start_22
   - group_end_19 <= 80
 constr_link_match_1_19_23: 80 switch_nic_out_19_nic_in_23 - group_start_23
   + group_end_19 <= 80
 constr_link_match_2_19_23: 80 switch_nic_out_19_nic_in_23 + group_start_23
   - group_end_19 <= 80
 constr_link_match_1_19_24: 80 switch_nic_out_19_nic_in_24 - group_start_24
   + group_end_19 <= 80
 constr_link_match_2_19_24: 80 switch_nic_out_19_nic_in_24 + group_start_24
   - group_end_19 <= 80
 constr_link_match_1_19_25: 80 switch_nic_out_19_nic_in_25 - group_start_25
   + group_end_19 <= 80
 constr_link_match_2_19_25: 80 switch_nic_out_19_nic_in_25 + group_start_25
   - group_end_19 <= 80
 constr_link_match_1_19_26: 80 switch_nic_out_19_nic_in_26 - group_start_26
   + group_end_19 <= 80
 constr_link_match_2_19_26: 80 switch_nic_out_19_nic_in_26 + group_start_26
   - group_end_19 <= 80
 constr_link_match_1_19_27: 80 switch_nic_out_19_nic_in_27 - group_start_27
   + group_end_19 <= 80
 constr_link_match_2_19_27: 80 switch_nic_out_19_nic_in_27 + group_start_27
   - group_end_19 <= 80
 constr_link_match_1_19_28: 80 switch_nic_out_19_nic_in_28 - group_start_28
   + group_end_19 <= 80
 constr_link_match_2_19_28: 80 switch_nic_out_19_nic_in_28 + group_start_28
   - group_end_19 <= 80
 constr_link_match_1_19_29: 80 switch_nic_out_19_nic_in_29 - group_start_29
   + group_end_19 <= 80
 constr_link_match_2_19_29: 80 switch_nic_out_19_nic_in_29 + group_start_29
   - group_end_19 <= 80
 constr_link_match_1_19_30: 80 switch_nic_out_19_nic_in_30 - group_start_30
   + group_end_19 <= 80
 constr_link_match_2_19_30: 80 switch_nic_out_19_nic_in_30 + group_start_30
   - group_end_19 <= 80
 constr_link_match_1_19_31: 80 switch_nic_out_19_nic_in_31 - group_start_31
   + group_end_19 <= 80
 constr_link_match_2_19_31: 80 switch_nic_out_19_nic_in_31 + group_start_31
   - group_end_19 <= 80
 constr_is_start_zero_20: start_20
   + 80 is_start_zero_20_for_source_nic_in_20 <= 80
 constr_edge_switch_start_zero_20_OR: switch_source_nic_in_20
   - is_start_zero_20_for_source_nic_in_20 <= 0
 constr_is_end_complete_20: start_20 + hold_20_1 + 2 hold_20_2
   + 3 hold_20_3 + 4 hold_20_4 + 5 hold_20_5 + 6 hold_20_6
   - 80 is_end_complete_20 >= 0
 constr_edge_switch_end_max_20_OR: switch_nic_out_20_sink
   - is_end_complete_20 <= 0
 constr_link_match_1_20_0: 80 switch_nic_out_20_nic_in_0 - group_start_0
   + group_end_20 <= 80
 constr_link_match_2_20_0: 80 switch_nic_out_20_nic_in_0 + group_start_0
   - group_end_20 <= 80
 constr_link_match_1_20_1: 80 switch_nic_out_20_nic_in_1 - group_start_1
   + group_end_20 <= 80
 constr_link_match_2_20_1: 80 switch_nic_out_20_nic_in_1 + group_start_1
   - group_end_20 <= 80
 constr_link_match_1_20_2: 80 switch_nic_out_20_nic_in_2 - group_start_2
   + group_end_20 <= 80
 constr_link_match_2_20_2: 80 switch_nic_out_20_nic_in_2 + group_start_2
   - group_end_20 <= 80
 constr_link_match_1_20_3: 80 switch_nic_out_20_nic_in_3 - group_start_3
   + group_end_20 <= 80
 constr_link_match_2_20_3: 80 switch_nic_out_20_nic_in_3 + group_start_3
   - group_end_20 <= 80
 constr_link_match_1_20_4: 80 switch_nic_out_20_nic_in_4 - group_start_4
   + group_end_20 <= 80
 constr_link_match_2_20_4: 80 switch_nic_out_20_nic_in_4 + group_start_4
   - group_end_20 <= 80
 constr_link_match_1_20_5: 80 switch_nic_out_20_nic_in_5 - group_start_5
   + group_end_20 <= 80
 constr_link_match_2_20_5: 80 switch_nic_out_20_nic_in_5 + group_start_5
   - group_end_20 <= 80
 constr_link_match_1_20_6: 80 switch_nic_out_20_nic_in_6 - group_start_6
   + group_end_20 <= 80
 constr_link_match_2_20_6: 80 switch_nic_out_20_nic_in_6 + group_start_6
   - group_end_20 <= 80
 constr_link_match_1_20_7: 80 switch_nic_out_20_nic_in_7 - group_start_7
   + group_end_20 <= 80
 constr_link_match_2_20_7: 80 switch_nic_out_20_nic_in_7 + group_start_7
   - group_end_20 <= 80
 constr_link_match_1_20_8: 80 switch_nic_out_20_nic_in_8 - group_start_8
   + group_end_20 <= 80
 constr_link_match_2_20_8: 80 switch_nic_out_20_nic_in_8 + group_start_8
   - group_end_20 <= 80
 constr_link_match_1_20_9: 80 switch_nic_out_20_nic_in_9 - group_start_9
   + group_end_20 <= 80
 constr_link_match_2_20_9: 80 switch_nic_out_20_nic_in_9 + group_start_9
   - group_end_20 <= 80
 constr_link_match_1_20_10: 80 switch_nic_out_20_nic_in_10 - group_start_10
   + group_end_20 <= 80
 constr_link_match_2_20_10: 80 switch_nic_out_20_nic_in_10 + group_start_10
   - group_end_20 <= 80
 constr_link_match_1_20_11: 80 switch_nic_out_20_nic_in_11 - group_start_11
   + group_end_20 <= 80
 constr_link_match_2_20_11: 80 switch_nic_out_20_nic_in_11 + group_start_11
   - group_end_20 <= 80
 constr_link_match_1_20_12: 80 switch_nic_out_20_nic_in_12 - group_start_12
   + group_end_20 <= 80
 constr_link_match_2_20_12: 80 switch_nic_out_20_nic_in_12 + group_start_12
   - group_end_20 <= 80
 constr_link_match_1_20_13: 80 switch_nic_out_20_nic_in_13 - group_start_13
   + group_end_20 <= 80
 constr_link_match_2_20_13: 80 switch_nic_out_20_nic_in_13 + group_start_13
   - group_end_20 <= 80
 constr_link_match_1_20_14: 80 switch_nic_out_20_nic_in_14 - group_start_14
   + group_end_20 <= 80
 constr_link_match_2_20_14: 80 switch_nic_out_20_nic_in_14 + group_start_14
   - group_end_20 <= 80
 constr_link_match_1_20_15: 80 switch_nic_out_20_nic_in_15 - group_start_15
   + group_end_20 <= 80
 constr_link_match_2_20_15: 80 switch_nic_out_20_nic_in_15 + group_start_15
   - group_end_20 <= 80
 constr_link_match_1_20_16: 80 switch_nic_out_20_nic_in_16 - group_start_16
   + group_end_20 <= 80
 constr_link_match_2_20_16: 80 switch_nic_out_20_nic_in_16 + group_start_16
   - group_end_20 <= 80
 constr_link_match_1_20_17: 80 switch_nic_out_20_nic_in_17 - group_start_17
   + group_end_20 <= 80
 constr_link_match_2_20_17: 80 switch_nic_out_20_nic_in_17 + group_start_17
   - group_end_20 <= 80
 constr_link_match_1_20_18: 80 switch_nic_out_20_nic_in_18 - group_start_18
   + group_end_20 <= 80
 constr_link_match_2_20_18: 80 switch_nic_out_20_nic_in_18 + group_start_18
   - group_end_20 <= 80
 constr_link_match_1_20_19: 80 switch_nic_out_20_nic_in_19 - group_start_19
   + group_end_20 <= 80
 constr_link_match_2_20_19: 80 switch_nic_out_20_nic_in_19 + group_start_19
   - group_end_20 <= 80
 constr_link_match_1_20_21: 80 switch_nic_out_20_nic_in_21 - group_start_21
   + group_end_20 <= 80
 constr_link_match_2_20_21: 80 switch_nic_out_20_nic_in_21 + group_start_21
   - group_end_20 <= 80
 constr_link_match_1_20_22: 80 switch_nic_out_20_nic_in_22 - group_start_22
   + group_end_20 <= 80
 constr_link_match_2_20_22: 80 switch_nic_out_20_nic_in_22 + group_start_22
   - group_end_20 <= 80
 constr_link_match_1_20_23: 80 switch_nic_out_20_nic_in_23 - group_start_23
   + group_end_20 <= 80
 constr_link_match_2_20_23: 80 switch_nic_out_20_nic_in_23 + group_start_23
   - group_end_20 <= 80
 constr_link_match_1_20_24: 80 switch_nic_out_20_nic_in_24 - group_start_24
   + group_end_20 <= 80
 constr_link_match_2_20_24: 80 switch_nic_out_20_nic_in_24 + group_start_24
   - group_end_20 <= 80
 constr_link_match_1_20_25: 80 switch_nic_out_20_nic_in_25 - group_start_25
   + group_end_20 <= 80
 constr_link_match_2_20_25: 80 switch_nic_out_20_nic_in_25 + group_start_25
   - group_end_20 <= 80
 constr_link_match_1_20_26: 80 switch_nic_out_20_nic_in_26 - group_start_26
   + group_end_20 <= 80
 constr_link_match_2_20_26: 80 switch_nic_out_20_nic_in_26 + group_start_26
   - group_end_20 <= 80
 constr_link_match_1_20_27: 80 switch_nic_out_20_nic_in_27 - group_start_27
   + group_end_20 <= 80
 constr_link_match_2_20_27: 80 switch_nic_out_20_nic_in_27 + group_start_27
   - group_end_20 <= 80
 constr_link_match_1_20_28: 80 switch_nic_out_20_nic_in_28 - group_start_28
   + group_end_20 <= 80
 constr_link_match_2_20_28: 80 switch_nic_out_20_nic_in_28 + group_start_28
   - group_end_20 <= 80
 constr_link_match_1_20_29: 80 switch_nic_out_20_nic_in_29 - group_start_29
   + group_end_20 <= 80
 constr_link_match_2_20_29: 80 switch_nic_out_20_nic_in_29 + group_start_29
   - group_end_20 <= 80
 constr_link_match_1_20_30: 80 switch_nic_out_20_nic_in_30 - group_start_30
   + group_end_20 <= 80
 constr_link_match_2_20_30: 80 switch_nic_out_20_nic_in_30 + group_start_30
   - group_end_20 <= 80
 constr_link_match_1_20_31: 80 switch_nic_out_20_nic_in_31 - group_start_31
   + group_end_20 <= 80
 constr_link_match_2_20_31: 80 switch_nic_out_20_nic_in_31 + group_start_31
   - group_end_20 <= 80
 constr_is_start_zero_21: start_21
   + 80 is_start_zero_21_for_source_nic_in_21 <= 80
 constr_edge_switch_start_zero_21_OR: switch_source_nic_in_21
   - is_start_zero_21_for_source_nic_in_21 <= 0
 constr_is_end_complete_21: start_21 + hold_21_1 + 2 hold_21_2
   + 3 hold_21_3 + 4 hold_21_4 + 5 hold_21_5 + 6 hold_21_6
   - 80 is_end_complete_21 >= 0
 constr_edge_switch_end_max_21_OR: switch_nic_out_21_sink
   - is_end_complete_21 <= 0
 constr_link_match_1_21_0: 80 switch_nic_out_21_nic_in_0 - group_start_0
   + group_end_21 <= 80
 constr_link_match_2_21_0: 80 switch_nic_out_21_nic_in_0 + group_start_0
   - group_end_21 <= 80
 constr_link_match_1_21_1: 80 switch_nic_out_21_nic_in_1 - group_start_1
   + group_end_21 <= 80
 constr_link_match_2_21_1: 80 switch_nic_out_21_nic_in_1 + group_start_1
   - group_end_21 <= 80
 constr_link_match_1_21_2: 80 switch_nic_out_21_nic_in_2 - group_start_2
   + group_end_21 <= 80
 constr_link_match_2_21_2: 80 switch_nic_out_21_nic_in_2 + group_start_2
   - group_end_21 <= 80
 constr_link_match_1_21_3: 80 switch_nic_out_21_nic_in_3 - group_start_3
   + group_end_21 <= 80
 constr_link_match_2_21_3: 80 switch_nic_out_21_nic_in_3 + group_start_3
   - group_end_21 <= 80
 constr_link_match_1_21_4: 80 switch_nic_out_21_nic_in_4 - group_start_4
   + group_end_21 <= 80
 constr_link_match_2_21_4: 80 switch_nic_out_21_nic_in_4 + group_start_4
   - group_end_21 <= 80
 constr_link_match_1_21_5: 80 switch_nic_out_21_nic_in_5 - group_start_5
   + group_end_21 <= 80
 constr_link_match_2_21_5: 80 switch_nic_out_21_nic_in_5 + group_start_5
   - group_end_21 <= 80
 constr_link_match_1_21_6: 80 switch_nic_out_21_nic_in_6 - group_start_6
   + group_end_21 <= 80
 constr_link_match_2_21_6: 80 switch_nic_out_21_nic_in_6 + group_start_6
   - group_end_21 <= 80
 constr_link_match_1_21_7: 80 switch_nic_out_21_nic_in_7 - group_start_7
   + group_end_21 <= 80
 constr_link_match_2_21_7: 80 switch_nic_out_21_nic_in_7 + group_start_7
   - group_end_21 <= 80
 constr_link_match_1_21_8: 80 switch_nic_out_21_nic_in_8 - group_start_8
   + group_end_21 <= 80
 constr_link_match_2_21_8: 80 switch_nic_out_21_nic_in_8 + group_start_8
   - group_end_21 <= 80
 constr_link_match_1_21_9: 80 switch_nic_out_21_nic_in_9 - group_start_9
   + group_end_21 <= 80
 constr_link_match_2_21_9: 80 switch_nic_out_21_nic_in_9 + group_start_9
   - group_end_21 <= 80
 constr_link_match_1_21_10: 80 switch_nic_out_21_nic_in_10 - group_start_10
   + group_end_21 <= 80
 constr_link_match_2_21_10: 80 switch_nic_out_21_nic_in_10 + group_start_10
   - group_end_21 <= 80
 constr_link_match_1_21_11: 80 switch_nic_out_21_nic_in_11 - group_start_11
   + group_end_21 <= 80
 constr_link_match_2_21_11: 80 switch_nic_out_21_nic_in_11 + group_start_11
   - group_end_21 <= 80
 constr_link_match_1_21_12: 80 switch_nic_out_21_nic_in_12 - group_start_12
   + group_end_21 <= 80
 constr_link_match_2_21_12: 80 switch_nic_out_21_nic_in_12 + group_start_12
   - group_end_21 <= 80
 constr_link_match_1_21_13: 80 switch_nic_out_21_nic_in_13 - group_start_13
   + group_end_21 <= 80
 constr_link_match_2_21_13: 80 switch_nic_out_21_nic_in_13 + group_start_13
   - group_end_21 <= 80
 constr_link_match_1_21_14: 80 switch_nic_out_21_nic_in_14 - group_start_14
   + group_end_21 <= 80
 constr_link_match_2_21_14: 80 switch_nic_out_21_nic_in_14 + group_start_14
   - group_end_21 <= 80
 constr_link_match_1_21_15: 80 switch_nic_out_21_nic_in_15 - group_start_15
   + group_end_21 <= 80
 constr_link_match_2_21_15: 80 switch_nic_out_21_nic_in_15 + group_start_15
   - group_end_21 <= 80
 constr_link_match_1_21_16: 80 switch_nic_out_21_nic_in_16 - group_start_16
   + group_end_21 <= 80
 constr_link_match_2_21_16: 80 switch_nic_out_21_nic_in_16 + group_start_16
   - group_end_21 <= 80
 constr_link_match_1_21_17: 80 switch_nic_out_21_nic_in_17 - group_start_17
   + group_end_21 <= 80
 constr_link_match_2_21_17: 80 switch_nic_out_21_nic_in_17 + group_start_17
   - group_end_21 <= 80
 constr_link_match_1_21_18: 80 switch_nic_out_21_nic_in_18 - group_start_18
   + group_end_21 <= 80
 constr_link_match_2_21_18: 80 switch_nic_out_21_nic_in_18 + group_start_18
   - group_end_21 <= 80
 constr_link_match_1_21_19: 80 switch_nic_out_21_nic_in_19 - group_start_19
   + group_end_21 <= 80
 constr_link_match_2_21_19: 80 switch_nic_out_21_nic_in_19 + group_start_19
   - group_end_21 <= 80
 constr_link_match_1_21_20: 80 switch_nic_out_21_nic_in_20 - group_start_20
   + group_end_21 <= 80
 constr_link_match_2_21_20: 80 switch_nic_out_21_nic_in_20 + group_start_20
   - group_end_21 <= 80
 constr_link_match_1_21_22: 80 switch_nic_out_21_nic_in_22 - group_start_22
   + group_end_21 <= 80
 constr_link_match_2_21_22: 80 switch_nic_out_21_nic_in_22 + group_start_22
   - group_end_21 <= 80
 constr_link_match_1_21_23: 80 switch_nic_out_21_nic_in_23 - group_start_23
   + group_end_21 <= 80
 constr_link_match_2_21_23: 80 switch_nic_out_21_nic_in_23 + group_start_23
   - group_end_21 <= 80
 constr_link_match_1_21_24: 80 switch_nic_out_21_nic_in_24 - group_start_24
   + group_end_21 <= 80
 constr_link_match_2_21_24: 80 switch_nic_out_21_nic_in_24 + group_start_24
   - group_end_21 <= 80
 constr_link_match_1_21_25: 80 switch_nic_out_21_nic_in_25 - group_start_25
   + group_end_21 <= 80
 constr_link_match_2_21_25: 80 switch_nic_out_21_nic_in_25 + group_start_25
   - group_end_21 <= 80
 constr_link_match_1_21_26: 80 switch_nic_out_21_nic_in_26 - group_start_26
   + group_end_21 <= 80
 constr_link_match_2_21_26: 80 switch_nic_out_21_nic_in_26 + group_start_26
   - group_end_21 <= 80
 constr_link_match_1_21_27: 80 switch_nic_out_21_nic_in_27 - group_start_27
   + group_end_21 <= 80
 constr_link_match_2_21_27: 80 switch_nic_out_21_nic_in_27 + group_start_27
   - group_end_21 <= 80
 constr_link_match_1_21_28: 80 switch_nic_out_21_nic_in_28 - group_start_28
   + group_end_21 <= 80
 constr_link_match_2_21_28: 80 switch_nic_out_21_nic_in_28 + group_start_28
   - group_end_21 <= 80
 constr_link_match_1_21_29: 80 switch_nic_out_21_nic_in_29 - group_start_29
   + group_end_21 <= 80
 constr_link_match_2_21_29: 80 switch_nic_out_21_nic_in_29 + group_start_29
   - group_end_21 <= 80
 constr_link_match_1_21_30: 80 switch_nic_out_21_nic_in_30 - group_start_30
   + group_end_21 <= 80
 constr_link_match_2_21_30: 80 switch_nic_out_21_nic_in_30 + group_start_30
   - group_end_21 <= 80
 constr_link_match_1_21_31: 80 switch_nic_out_21_nic_in_31 - group_start_31
   + group_end_21 <= 80
 constr_link_match_2_21_31: 80 switch_nic_out_21_nic_in_31 + group_start_31
   - group_end_21 <= 80
 constr_is_start_zero_22: start_22
   + 80 is_start_zero_22_for_source_nic_in_22 <= 80
 constr_edge_switch_start_zero_22_OR: switch_source_nic_in_22
   - is_start_zero_22_for_source_nic_in_22 <= 0
 constr_is_end_complete_22: start_22 + hold_22_1 + 2 hold_22_2
   + 3 hold_22_3 + 4 hold_22_4 + 5 hold_22_5 + 6 hold_22_6
   - 80 is_end_complete_22 >= 0
 constr_edge_switch_end_max_22_OR: switch_nic_out_22_sink
   - is_end_complete_22 <= 0
 constr_link_match_1_22_0: 80 switch_nic_out_22_nic_in_0 - group_start_0
   + group_end_22 <= 80
 constr_link_match_2_22_0: 80 switch_nic_out_22_nic_in_0 + group_start_0
   - group_end_22 <= 80
 constr_link_match_1_22_1: 80 switch_nic_out_22_nic_in_1 - group_start_1
   + group_end_22 <= 80
 constr_link_match_2_22_1: 80 switch_nic_out_22_nic_in_1 + group_start_1
   - group_end_22 <= 80
 constr_link_match_1_22_2: 80 switch_nic_out_22_nic_in_2 - group_start_2
   + group_end_22 <= 80
 constr_link_match_2_22_2: 80 switch_nic_out_22_nic_in_2 + group_start_2
   - group_end_22 <= 80
 constr_link_match_1_22_3: 80 switch_nic_out_22_nic_in_3 - group_start_3
   + group_end_22 <= 80
 constr_link_match_2_22_3: 80 switch_nic_out_22_nic_in_3 + group_start_3
   - group_end_22 <= 80
 constr_link_match_1_22_4: 80 switch_nic_out_22_nic_in_4 - group_start_4
   + group_end_22 <= 80
 constr_link_match_2_22_4: 80 switch_nic_out_22_nic_in_4 + group_start_4
   - group_end_22 <= 80
 constr_link_match_1_22_5: 80 switch_nic_out_22_nic_in_5 - group_start_5
   + group_end_22 <= 80
 constr_link_match_2_22_5: 80 switch_nic_out_22_nic_in_5 + group_start_5
   - group_end_22 <= 80
 constr_link_match_1_22_6: 80 switch_nic_out_22_nic_in_6 - group_start_6
   + group_end_22 <= 80
 constr_link_match_2_22_6: 80 switch_nic_out_22_nic_in_6 + group_start_6
   - group_end_22 <= 80
 constr_link_match_1_22_7: 80 switch_nic_out_22_nic_in_7 - group_start_7
   + group_end_22 <= 80
 constr_link_match_2_22_7: 80 switch_nic_out_22_nic_in_7 + group_start_7
   - group_end_22 <= 80
 constr_link_match_1_22_8: 80 switch_nic_out_22_nic_in_8 - group_start_8
   + group_end_22 <= 80
 constr_link_match_2_22_8: 80 switch_nic_out_22_nic_in_8 + group_start_8
   - group_end_22 <= 80
 constr_link_match_1_22_9: 80 switch_nic_out_22_nic_in_9 - group_start_9
   + group_end_22 <= 80
 constr_link_match_2_22_9: 80 switch_nic_out_22_nic_in_9 + group_start_9
   - group_end_22 <= 80
 constr_link_match_1_22_10: 80 switch_nic_out_22_nic_in_10 - group_start_10
   + group_end_22 <= 80
 constr_link_match_2_22_10: 80 switch_nic_out_22_nic_in_10 + group_start_10
   - group_end_22 <= 80
 constr_link_match_1_22_11: 80 switch_nic_out_22_nic_in_11 - group_start_11
   + group_end_22 <= 80
 constr_link_match_2_22_11: 80 switch_nic_out_22_nic_in_11 + group_start_11
   - group_end_22 <= 80
 constr_link_match_1_22_12: 80 switch_nic_out_22_nic_in_12 - group_start_12
   + group_end_22 <= 80
 constr_link_match_2_22_12: 80 switch_nic_out_22_nic_in_12 + group_start_12
   - group_end_22 <= 80
 constr_link_match_1_22_13: 80 switch_nic_out_22_nic_in_13 - group_start_13
   + group_end_22 <= 80
 constr_link_match_2_22_13: 80 switch_nic_out_22_nic_in_13 + group_start_13
   - group_end_22 <= 80
 constr_link_match_1_22_14: 80 switch_nic_out_22_nic_in_14 - group_start_14
   + group_end_22 <= 80
 constr_link_match_2_22_14: 80 switch_nic_out_22_nic_in_14 + group_start_14
   - group_end_22 <= 80
 constr_link_match_1_22_15: 80 switch_nic_out_22_nic_in_15 - group_start_15
   + group_end_22 <= 80
 constr_link_match_2_22_15: 80 switch_nic_out_22_nic_in_15 + group_start_15
   - group_end_22 <= 80
 constr_link_match_1_22_16: 80 switch_nic_out_22_nic_in_16 - group_start_16
   + group_end_22 <= 80
 constr_link_match_2_22_16: 80 switch_nic_out_22_nic_in_16 + group_start_16
   - group_end_22 <= 80
 constr_link_match_1_22_17: 80 switch_nic_out_22_nic_in_17 - group_start_17
   + group_end_22 <= 80
 constr_link_match_2_22_17: 80 switch_nic_out_22_nic_in_17 + group_start_17
   - group_end_22 <= 80
 constr_link_match_1_22_18: 80 switch_nic_out_22_nic_in_18 - group_start_18
   + group_end_22 <= 80
 constr_link_match_2_22_18: 80 switch_nic_out_22_nic_in_18 + group_start_18
   - group_end_22 <= 80
 constr_link_match_1_22_19: 80 switch_nic_out_22_nic_in_19 - group_start_19
   + group_end_22 <= 80
 constr_link_match_2_22_19: 80 switch_nic_out_22_nic_in_19 + group_start_19
   - group_end_22 <= 80
 constr_link_match_1_22_20: 80 switch_nic_out_22_nic_in_20 - group_start_20
   + group_end_22 <= 80
 constr_link_match_2_22_20: 80 switch_nic_out_22_nic_in_20 + group_start_20
   - group_end_22 <= 80
 constr_link_match_1_22_21: 80 switch_nic_out_22_nic_in_21 - group_start_21
   + group_end_22 <= 80
 constr_link_match_2_22_21: 80 switch_nic_out_22_nic_in_21 + group_start_21
   - group_end_22 <= 80
 constr_link_match_1_22_23: 80 switch_nic_out_22_nic_in_23 - group_start_23
   + group_end_22 <= 80
 constr_link_match_2_22_23: 80 switch_nic_out_22_nic_in_23 + group_start_23
   - group_end_22 <= 80
 constr_link_match_1_22_24: 80 switch_nic_out_22_nic_in_24 - group_start_24
   + group_end_22 <= 80
 constr_link_match_2_22_24: 80 switch_nic_out_22_nic_in_24 + group_start_24
   - group_end_22 <= 80
 constr_link_match_1_22_25: 80 switch_nic_out_22_nic_in_25 - group_start_25
   + group_end_22 <= 80
 constr_link_match_2_22_25: 80 switch_nic_out_22_nic_in_25 + group_start_25
   - group_end_22 <= 80
 constr_link_match_1_22_26: 80 switch_nic_out_22_nic_in_26 - group_start_26
   + group_end_22 <= 80
 constr_link_match_2_22_26: 80 switch_nic_out_22_nic_in_26 + group_start_26
   - group_end_22 <= 80
 constr_link_match_1_22_27: 80 switch_nic_out_22_nic_in_27 - group_start_27
   + group_end_22 <= 80
 constr_link_match_2_22_27: 80 switch_nic_out_22_nic_in_27 + group_start_27
   - group_end_22 <= 80
 constr_link_match_1_22_28: 80 switch_nic_out_22_nic_in_28 - group_start_28
   + group_end_22 <= 80
 constr_link_match_2_22_28: 80 switch_nic_out_22_nic_in_28 + group_start_28
   - group_end_22 <= 80
 constr_link_match_1_22_29: 80 switch_nic_out_22_nic_in_29 - group_start_29
   + group_end_22 <= 80
 constr_link_match_2_22_29: 80 switch_nic_out_22_nic_in_29 + group_start_29
   - group_end_22 <= 80
 constr_link_match_1_22_30: 80 switch_nic_out_22_nic_in_30 - group_start_30
   + group_end_22 <= 80
 constr_link_match_2_22_30: 80 switch_nic_out_22_nic_in_30 + group_start_30
   - group_end_22 <= 80
 constr_link_match_1_22_31: 80 switch_nic_out_22_nic_in_31 - group_start_31
   + group_end_22 <= 80
 constr_link_match_2_22_31: 80 switch_nic_out_22_nic_in_31 + group_start_31
   - group_end_22 <= 80
 constr_is_start_zero_23: start_23
   + 80 is_start_zero_23_for_source_nic_in_23 <= 80
 constr_edge_switch_start_zero_23_OR: switch_source_nic_in_23
   - is_start_zero_23_for_source_nic_in_23 <= 0
 constr_is_end_complete_23: start_23 + hold_23_1 + 2 hold_23_2
   + 3 hold_23_3 + 4 hold_23_4 + 5 hold_23_5 + 6 hold_23_6
   - 80 is_end_complete_23 >= 0
 constr_edge_switch_end_max_23_OR: switch_nic_out_23_sink
   - is_end_complete_23 <= 0
 constr_link_match_1_23_0: 80 switch_nic_out_23_nic_in_0 - group_start_0
   + group_end_23 <= 80
 constr_link_match_2_23_0: 80 switch_nic_out_23_nic_in_0 + group_start_0
   - group_end_23 <= 80
 constr_link_match_1_23_1: 80 switch_nic_out_23_nic_in_1 - group_start_1
   + group_end_23 <= 80
 constr_link_match_2_23_1: 80 switch_nic_out_23_nic_in_1 + group_start_1
   - group_end_23 <= 80
 constr_link_match_1_23_2: 80 switch_nic_out_23_nic_in_2 - group_start_2
   + group_end_23 <= 80
 constr_link_match_2_23_2: 80 switch_nic_out_23_nic_in_2 + group_start_2
   - group_end_23 <= 80
 constr_link_match_1_23_3: 80 switch_nic_out_23_nic_in_3 - group_start_3
   + group_end_23 <= 80
 constr_link_match_2_23_3: 80 switch_nic_out_23_nic_in_3 + group_start_3
   - group_end_23 <= 80
 constr_link_match_1_23_4: 80 switch_nic_out_23_nic_in_4 - group_start_4
   + group_end_23 <= 80
 constr_link_match_2_23_4: 80 switch_nic_out_23_nic_in_4 + group_start_4
   - group_end_23 <= 80
 constr_link_match_1_23_5: 80 switch_nic_out_23_nic_in_5 - group_start_5
   + group_end_23 <= 80
 constr_link_match_2_23_5: 80 switch_nic_out_23_nic_in_5 + group_start_5
   - group_end_23 <= 80
 constr_link_match_1_23_6: 80 switch_nic_out_23_nic_in_6 - group_start_6
   + group_end_23 <= 80
 constr_link_match_2_23_6: 80 switch_nic_out_23_nic_in_6 + group_start_6
   - group_end_23 <= 80
 constr_link_match_1_23_7: 80 switch_nic_out_23_nic_in_7 - group_start_7
   + group_end_23 <= 80
 constr_link_match_2_23_7: 80 switch_nic_out_23_nic_in_7 + group_start_7
   - group_end_23 <= 80
 constr_link_match_1_23_8: 80 switch_nic_out_23_nic_in_8 - group_start_8
   + group_end_23 <= 80
 constr_link_match_2_23_8: 80 switch_nic_out_23_nic_in_8 + group_start_8
   - group_end_23 <= 80
 constr_link_match_1_23_9: 80 switch_nic_out_23_nic_in_9 - group_start_9
   + group_end_23 <= 80
 constr_link_match_2_23_9: 80 switch_nic_out_23_nic_in_9 + group_start_9
   - group_end_23 <= 80
 constr_link_match_1_23_10: 80 switch_nic_out_23_nic_in_10 - group_start_10
   + group_end_23 <= 80
 constr_link_match_2_23_10: 80 switch_nic_out_23_nic_in_10 + group_start_10
   - group_end_23 <= 80
 constr_link_match_1_23_11: 80 switch_nic_out_23_nic_in_11 - group_start_11
   + group_end_23 <= 80
 constr_link_match_2_23_11: 80 switch_nic_out_23_nic_in_11 + group_start_11
   - group_end_23 <= 80
 constr_link_match_1_23_12: 80 switch_nic_out_23_nic_in_12 - group_start_12
   + group_end_23 <= 80
 constr_link_match_2_23_12: 80 switch_nic_out_23_nic_in_12 + group_start_12
   - group_end_23 <= 80
 constr_link_match_1_23_13: 80 switch_nic_out_23_nic_in_13 - group_start_13
   + group_end_23 <= 80
 constr_link_match_2_23_13: 80 switch_nic_out_23_nic_in_13 + group_start_13
   - group_end_23 <= 80
 constr_link_match_1_23_14: 80 switch_nic_out_23_nic_in_14 - group_start_14
   + group_end_23 <= 80
 constr_link_match_2_23_14: 80 switch_nic_out_23_nic_in_14 + group_start_14
   - group_end_23 <= 80
 constr_link_match_1_23_15: 80 switch_nic_out_23_nic_in_15 - group_start_15
   + group_end_23 <= 80
 constr_link_match_2_23_15: 80 switch_nic_out_23_nic_in_15 + group_start_15
   - group_end_23 <= 80
 constr_link_match_1_23_16: 80 switch_nic_out_23_nic_in_16 - group_start_16
   + group_end_23 <= 80
 constr_link_match_2_23_16: 80 switch_nic_out_23_nic_in_16 + group_start_16
   - group_end_23 <= 80
 constr_link_match_1_23_17: 80 switch_nic_out_23_nic_in_17 - group_start_17
   + group_end_23 <= 80
 constr_link_match_2_23_17: 80 switch_nic_out_23_nic_in_17 + group_start_17
   - group_end_23 <= 80
 constr_link_match_1_23_18: 80 switch_nic_out_23_nic_in_18 - group_start_18
   + group_end_23 <= 80
 constr_link_match_2_23_18: 80 switch_nic_out_23_nic_in_18 + group_start_18
   - group_end_23 <= 80
 constr_link_match_1_23_19: 80 switch_nic_out_23_nic_in_19 - group_start_19
   + group_end_23 <= 80
 constr_link_match_2_23_19: 80 switch_nic_out_23_nic_in_19 + group_start_19
   - group_end_23 <= 80
 constr_link_match_1_23_20: 80 switch_nic_out_23_nic_in_20 - group_start_20
   + group_end_23 <= 80
 constr_link_match_2_23_20: 80 switch_nic_out_23_nic_in_20 + group_start_20
   - group_end_23 <= 80
 constr_link_match_1_23_21: 80 switch_nic_out_23_nic_in_21 - group_start_21
   + group_end_23 <= 80
 constr_link_match_2_23_21: 80 switch_nic_out_23_nic_in_21 + group_start_21
   - group_end_23 <= 80
 constr_link_match_1_23_22: 80 switch_nic_out_23_nic_in_22 - group_start_22
   + group_end_23 <= 80
 constr_link_match_2_23_22: 80 switch_nic_out_23_nic_in_22 + group_start_22
   - group_end_23 <= 80
 constr_link_match_1_23_24: 80 switch_nic_out_23_nic_in_24 - group_start_24
   + group_end_23 <= 80
 constr_link_match_2_23_24: 80 switch_nic_out_23_nic_in_24 + group_start_24
   - group_end_23 <= 80
 constr_link_match_1_23_25: 80 switch_nic_out_23_nic_in_25 - group_start_25
   + group_end_23 <= 80
 constr_link_match_2_23_25: 80 switch_nic_out_23_nic_in_25 + group_start_25
   - group_end_23 <= 80
 constr_link_match_1_23_26: 80 switch_nic_out_23_nic_in_26 - group_start_26
   + group_end_23 <= 80
 constr_link_match_2_23_26: 80 switch_nic_out_23_nic_in_26 + group_start_26
   - group_end_23 <= 80
 constr_link_match_1_23_27: 80 switch_nic_out_23_nic_in_27 - group_start_27
   + group_end_23 <= 80
 constr_link_match_2_23_27: 80 switch_nic_out_23_nic_in_27 + group_start_27
   - group_end_23 <= 80
 constr_link_match_1_23_28: 80 switch_nic_out_23_nic_in_28 - group_start_28
   + group_end_23 <= 80
 constr_link_match_2_23_28: 80 switch_nic_out_23_nic_in_28 + group_start_28
   - group_end_23 <= 80
 constr_link_match_1_23_29: 80 switch_nic_out_23_nic_in_29 - group_start_29
   + group_end_23 <= 80
 constr_link_match_2_23_29: 80 switch_nic_out_23_nic_in_29 + group_start_29
   - group_end_23 <= 80
 constr_link_match_1_23_30: 80 switch_nic_out_23_nic_in_30 - group_start_30
   + group_end_23 <= 80
 constr_link_match_2_23_30: 80 switch_nic_out_23_nic_in_30 + group_start_30
   - group_end_23 <= 80
 constr_link_match_1_23_31: 80 switch_nic_out_23_nic_in_31 - group_start_31
   + group_end_23 <= 80
 constr_link_match_2_23_31: 80 switch_nic_out_23_nic_in_31 + group_start_31
   - group_end_23 <= 80
 constr_is_start_zero_24: start_24
   + 80 is_start_zero_24_for_source_nic_in_24 <= 80
 constr_edge_switch_start_zero_24_OR: switch_source_nic_in_24
   - is_start_zero_24_for_source_nic_in_24 <= 0
 constr_is_end_complete_24: start_24 + hold_24_1 + 2 hold_24_2
   + 3 hold_24_3 + 4 hold_24_4 + 5 hold_24_5 + 6 hold_24_6
   - 80 is_end_complete_24 >= 0
 constr_edge_switch_end_max_24_OR: switch_nic_out_24_sink
   - is_end_complete_24 <= 0
 constr_link_match_1_24_0: 80 switch_nic_out_24_nic_in_0 - group_start_0
   + group_end_24 <= 80
 constr_link_match_2_24_0: 80 switch_nic_out_24_nic_in_0 + group_start_0
   - group_end_24 <= 80
 constr_link_match_1_24_1: 80 switch_nic_out_24_nic_in_1 - group_start_1
   + group_end_24 <= 80
 constr_link_match_2_24_1: 80 switch_nic_out_24_nic_in_1 + group_start_1
   - group_end_24 <= 80
 constr_link_match_1_24_2: 80 switch_nic_out_24_nic_in_2 - group_start_2
   + group_end_24 <= 80
 constr_link_match_2_24_2: 80 switch_nic_out_24_nic_in_2 + group_start_2
   - group_end_24 <= 80
 constr_link_match_1_24_3: 80 switch_nic_out_24_nic_in_3 - group_start_3
   + group_end_24 <= 80
 constr_link_match_2_24_3: 80 switch_nic_out_24_nic_in_3 + group_start_3
   - group_end_24 <= 80
 constr_link_match_1_24_4: 80 switch_nic_out_24_nic_in_4 - group_start_4
   + group_end_24 <= 80
 constr_link_match_2_24_4: 80 switch_nic_out_24_nic_in_4 + group_start_4
   - group_end_24 <= 80
 constr_link_match_1_24_5: 80 switch_nic_out_24_nic_in_5 - group_start_5
   + group_end_24 <= 80
 constr_link_match_2_24_5: 80 switch_nic_out_24_nic_in_5 + group_start_5
   - group_end_24 <= 80
 constr_link_match_1_24_6: 80 switch_nic_out_24_nic_in_6 - group_start_6
   + group_end_24 <= 80
 constr_link_match_2_24_6: 80 switch_nic_out_24_nic_in_6 + group_start_6
   - group_end_24 <= 80
 constr_link_match_1_24_7: 80 switch_nic_out_24_nic_in_7 - group_start_7
   + group_end_24 <= 80
 constr_link_match_2_24_7: 80 switch_nic_out_24_nic_in_7 + group_start_7
   - group_end_24 <= 80
 constr_link_match_1_24_8: 80 switch_nic_out_24_nic_in_8 - group_start_8
   + group_end_24 <= 80
 constr_link_match_2_24_8: 80 switch_nic_out_24_nic_in_8 + group_start_8
   - group_end_24 <= 80
 constr_link_match_1_24_9: 80 switch_nic_out_24_nic_in_9 - group_start_9
   + group_end_24 <= 80
 constr_link_match_2_24_9: 80 switch_nic_out_24_nic_in_9 + group_start_9
   - group_end_24 <= 80
 constr_link_match_1_24_10: 80 switch_nic_out_24_nic_in_10 - group_start_10
   + group_end_24 <= 80
 constr_link_match_2_24_10: 80 switch_nic_out_24_nic_in_10 + group_start_10
   - group_end_24 <= 80
 constr_link_match_1_24_11: 80 switch_nic_out_24_nic_in_11 - group_start_11
   + group_end_24 <= 80
 constr_link_match_2_24_11: 80 switch_nic_out_24_nic_in_11 + group_start_11
   - group_end_24 <= 80
 constr_link_match_1_24_12: 80 switch_nic_out_24_nic_in_12 - group_start_12
   + group_end_24 <= 80
 constr_link_match_2_24_12: 80 switch_nic_out_24_nic_in_12 + group_start_12
   - group_end_24 <= 80
 constr_link_match_1_24_13: 80 switch_nic_out_24_nic_in_13 - group_start_13
   + group_end_24 <= 80
 constr_link_match_2_24_13: 80 switch_nic_out_24_nic_in_13 + group_start_13
   - group_end_24 <= 80
 constr_link_match_1_24_14: 80 switch_nic_out_24_nic_in_14 - group_start_14
   + group_end_24 <= 80
 constr_link_match_2_24_14: 80 switch_nic_out_24_nic_in_14 + group_start_14
   - group_end_24 <= 80
 constr_link_match_1_24_15: 80 switch_nic_out_24_nic_in_15 - group_start_15
   + group_end_24 <= 80
 constr_link_match_2_24_15: 80 switch_nic_out_24_nic_in_15 + group_start_15
   - group_end_24 <= 80
 constr_link_match_1_24_16: 80 switch_nic_out_24_nic_in_16 - group_start_16
   + group_end_24 <= 80
 constr_link_match_2_24_16: 80 switch_nic_out_24_nic_in_16 + group_start_16
   - group_end_24 <= 80
 constr_link_match_1_24_17: 80 switch_nic_out_24_nic_in_17 - group_start_17
   + group_end_24 <= 80
 constr_link_match_2_24_17: 80 switch_nic_out_24_nic_in_17 + group_start_17
   - group_end_24 <= 80
 constr_link_match_1_24_18: 80 switch_nic_out_24_nic_in_18 - group_start_18
   + group_end_24 <= 80
 constr_link_match_2_24_18: 80 switch_nic_out_24_nic_in_18 + group_start_18
   - group_end_24 <= 80
 constr_link_match_1_24_19: 80 switch_nic_out_24_nic_in_19 - group_start_19
   + group_end_24 <= 80
 constr_link_match_2_24_19: 80 switch_nic_out_24_nic_in_19 + group_start_19
   - group_end_24 <= 80
 constr_link_match_1_24_20: 80 switch_nic_out_24_nic_in_20 - group_start_20
   + group_end_24 <= 80
 constr_link_match_2_24_20: 80 switch_nic_out_24_nic_in_20 + group_start_20
   - group_end_24 <= 80
 constr_link_match_1_24_21: 80 switch_nic_out_24_nic_in_21 - group_start_21
   + group_end_24 <= 80
 constr_link_match_2_24_21: 80 switch_nic_out_24_nic_in_21 + group_start_21
   - group_end_24 <= 80
 constr_link_match_1_24_22: 80 switch_nic_out_24_nic_in_22 - group_start_22
   + group_end_24 <= 80
 constr_link_match_2_24_22: 80 switch_nic_out_24_nic_in_22 + group_start_22
   - group_end_24 <= 80
 constr_link_match_1_24_23: 80 switch_nic_out_24_nic_in_23 - group_start_23
   + group_end_24 <= 80
 constr_link_match_2_24_23: 80 switch_nic_out_24_nic_in_23 + group_start_23
   - group_end_24 <= 80
 constr_link_match_1_24_25: 80 switch_nic_out_24_nic_in_25 - group_start_25
   + group_end_24 <= 80
 constr_link_match_2_24_25: 80 switch_nic_out_24_nic_in_25 + group_start_25
   - group_end_24 <= 80
 constr_link_match_1_24_26: 80 switch_nic_out_24_nic_in_26 - group_start_26
   + group_end_24 <= 80
 constr_link_match_2_24_26: 80 switch_nic_out_24_nic_in_26 + group_start_26
   - group_end_24 <= 80
 constr_link_match_1_24_27: 80 switch_nic_out_24_nic_in_27 - group_start_27
   + group_end_24 <= 80
 constr_link_match_2_24_27: 80 switch_nic_out_24_nic_in_27 + group_start_27
   - group_end_24 <= 80
 constr_link_match_1_24_28: 80 switch_nic_out_24_nic_in_28 - group_start_28
   + group_end_24 <= 80
 constr_link_match_2_24_28: 80 switch_nic_out_24_nic_in_28 + group_start_28
   - group_end_24 <= 80
 constr_link_match_1_24_29: 80 switch_nic_out_24_nic_in_29 - group_start_29
   + group_end_24 <= 80
 constr_link_match_2_24_29: 80 switch_nic_out_24_nic_in_29 + group_start_29
   - group_end_24 <= 80
 constr_link_match_1_24_30: 80 switch_nic_out_24_nic_in_30 - group_start_30
   + group_end_24 <= 80
 constr_link_match_2_24_30: 80 switch_nic_out_24_nic_in_30 + group_start_30
   - group_end_24 <= 80
 constr_link_match_1_24_31: 80 switch_nic_out_24_nic_in_31 - group_start_31
   + group_end_24 <= 80
 constr_link_match_2_24_31: 80 switch_nic_out_24_nic_in_31 + group_start_31
   - group_end_24 <= 80
 constr_is_start_zero_25: start_25
   + 80 is_start_zero_25_for_source_nic_in_25 <= 80
 constr_edge_switch_start_zero_25_OR: switch_source_nic_in_25
   - is_start_zero_25_for_source_nic_in_25 <= 0
 constr_is_end_complete_25: start_25 + hold_25_1 + 2 hold_25_2
   + 3 hold_25_3 + 4 hold_25_4 + 5 hold_25_5 + 6 hold_25_6
   - 80 is_end_complete_25 >= 0
 constr_edge_switch_end_max_25_OR: switch_nic_out_25_sink
   - is_end_complete_25 <= 0
 constr_link_match_1_25_0: 80 switch_nic_out_25_nic_in_0 - group_start_0
   + group_end_25 <= 80
 constr_link_match_2_25_0: 80 switch_nic_out_25_nic_in_0 + group_start_0
   - group_end_25 <= 80
 constr_link_match_1_25_1: 80 switch_nic_out_25_nic_in_1 - group_start_1
   + group_end_25 <= 80
 constr_link_match_2_25_1: 80 switch_nic_out_25_nic_in_1 + group_start_1
   - group_end_25 <= 80
 constr_link_match_1_25_2: 80 switch_nic_out_25_nic_in_2 - group_start_2
   + group_end_25 <= 80
 constr_link_match_2_25_2: 80 switch_nic_out_25_nic_in_2 + group_start_2
   - group_end_25 <= 80
 constr_link_match_1_25_3: 80 switch_nic_out_25_nic_in_3 - group_start_3
   + group_end_25 <= 80
 constr_link_match_2_25_3: 80 switch_nic_out_25_nic_in_3 + group_start_3
   - group_end_25 <= 80
 constr_link_match_1_25_4: 80 switch_nic_out_25_nic_in_4 - group_start_4
   + group_end_25 <= 80
 constr_link_match_2_25_4: 80 switch_nic_out_25_nic_in_4 + group_start_4
   - group_end_25 <= 80
 constr_link_match_1_25_5: 80 switch_nic_out_25_nic_in_5 - group_start_5
   + group_end_25 <= 80
 constr_link_match_2_25_5: 80 switch_nic_out_25_nic_in_5 + group_start_5
   - group_end_25 <= 80
 constr_link_match_1_25_6: 80 switch_nic_out_25_nic_in_6 - group_start_6
   + group_end_25 <= 80
 constr_link_match_2_25_6: 80 switch_nic_out_25_nic_in_6 + group_start_6
   - group_end_25 <= 80
 constr_link_match_1_25_7: 80 switch_nic_out_25_nic_in_7 - group_start_7
   + group_end_25 <= 80
 constr_link_match_2_25_7: 80 switch_nic_out_25_nic_in_7 + group_start_7
   - group_end_25 <= 80
 constr_link_match_1_25_8: 80 switch_nic_out_25_nic_in_8 - group_start_8
   + group_end_25 <= 80
 constr_link_match_2_25_8: 80 switch_nic_out_25_nic_in_8 + group_start_8
   - group_end_25 <= 80
 constr_link_match_1_25_9: 80 switch_nic_out_25_nic_in_9 - group_start_9
   + group_end_25 <= 80
 constr_link_match_2_25_9: 80 switch_nic_out_25_nic_in_9 + group_start_9
   - group_end_25 <= 80
 constr_link_match_1_25_10: 80 switch_nic_out_25_nic_in_10 - group_start_10
   + group_end_25 <= 80
 constr_link_match_2_25_10: 80 switch_nic_out_25_nic_in_10 + group_start_10
   - group_end_25 <= 80
 constr_link_match_1_25_11: 80 switch_nic_out_25_nic_in_11 - group_start_11
   + group_end_25 <= 80
 constr_link_match_2_25_11: 80 switch_nic_out_25_nic_in_11 + group_start_11
   - group_end_25 <= 80
 constr_link_match_1_25_12: 80 switch_nic_out_25_nic_in_12 - group_start_12
   + group_end_25 <= 80
 constr_link_match_2_25_12: 80 switch_nic_out_25_nic_in_12 + group_start_12
   - group_end_25 <= 80
 constr_link_match_1_25_13: 80 switch_nic_out_25_nic_in_13 - group_start_13
   + group_end_25 <= 80
 constr_link_match_2_25_13: 80 switch_nic_out_25_nic_in_13 + group_start_13
   - group_end_25 <= 80
 constr_link_match_1_25_14: 80 switch_nic_out_25_nic_in_14 - group_start_14
   + group_end_25 <= 80
 constr_link_match_2_25_14: 80 switch_nic_out_25_nic_in_14 + group_start_14
   - group_end_25 <= 80
 constr_link_match_1_25_15: 80 switch_nic_out_25_nic_in_15 - group_start_15
   + group_end_25 <= 80
 constr_link_match_2_25_15: 80 switch_nic_out_25_nic_in_15 + group_start_15
   - group_end_25 <= 80
 constr_link_match_1_25_16: 80 switch_nic_out_25_nic_in_16 - group_start_16
   + group_end_25 <= 80
 constr_link_match_2_25_16: 80 switch_nic_out_25_nic_in_16 + group_start_16
   - group_end_25 <= 80
 constr_link_match_1_25_17: 80 switch_nic_out_25_nic_in_17 - group_start_17
   + group_end_25 <= 80
 constr_link_match_2_25_17: 80 switch_nic_out_25_nic_in_17 + group_start_17
   - group_end_25 <= 80
 constr_link_match_1_25_18: 80 switch_nic_out_25_nic_in_18 - group_start_18
   + group_end_25 <= 80
 constr_link_match_2_25_18: 80 switch_nic_out_25_nic_in_18 + group_start_18
   - group_end_25 <= 80
 constr_link_match_1_25_19: 80 switch_nic_out_25_nic_in_19 - group_start_19
   + group_end_25 <= 80
 constr_link_match_2_25_19: 80 switch_nic_out_25_nic_in_19 + group_start_19
   - group_end_25 <= 80
 constr_link_match_1_25_20: 80 switch_nic_out_25_nic_in_20 - group_start_20
   + group_end_25 <= 80
 constr_link_match_2_25_20: 80 switch_nic_out_25_nic_in_20 + group_start_20
   - group_end_25 <= 80
 constr_link_match_1_25_21: 80 switch_nic_out_25_nic_in_21 - group_start_21
   + group_end_25 <= 80
 constr_link_match_2_25_21: 80 switch_nic_out_25_nic_in_21 + group_start_21
   - group_end_25 <= 80
 constr_link_match_1_25_22: 80 switch_nic_out_25_nic_in_22 - group_start_22
   + group_end_25 <= 80
 constr_link_match_2_25_22: 80 switch_nic_out_25_nic_in_22 + group_start_22
   - group_end_25 <= 80
 constr_link_match_1_25_23: 80 switch_nic_out_25_nic_in_23 - group_start_23
   + group_end_25 <= 80
 constr_link_match_2_25_23: 80 switch_nic_out_25_nic_in_23 + group_start_23
   - group_end_25 <= 80
 constr_link_match_1_25_24: 80 switch_nic_out_25_nic_in_24 - group_start_24
   + group_end_25 <= 80
 constr_link_match_2_25_24: 80 switch_nic_out_25_nic_in_24 + group_start_24
   - group_end_25 <= 80
 constr_link_match_1_25_26: 80 switch_nic_out_25_nic_in_26 - group_start_26
   + group_end_25 <= 80
 constr_link_match_2_25_26: 80 switch_nic_out_25_nic_in_26 + group_start_26
   - group_end_25 <= 80
 constr_link_match_1_25_27: 80 switch_nic_out_25_nic_in_27 - group_start_27
   + group_end_25 <= 80
 constr_link_match_2_25_27: 80 switch_nic_out_25_nic_in_27 + group_start_27
   - group_end_25 <= 80
 constr_link_match_1_25_28: 80 switch_nic_out_25_nic_in_28 - group_start_28
   + group_end_25 <= 80
 constr_link_match_2_25_28: 80 switch_nic_out_25_nic_in_28 + group_start_28
   - group_end_25 <= 80
 constr_link_match_1_25_29: 80 switch_nic_out_25_nic_in_29 - group_start_29
   + group_end_25 <= 80
 constr_link_match_2_25_29: 80 switch_nic_out_25_nic_in_29 + group_start_29
   - group_end_25 <= 80
 constr_link_match_1_25_30: 80 switch_nic_out_25_nic_in_30 - group_start_30
   + group_end_25 <= 80
 constr_link_match_2_25_30: 80 switch_nic_out_25_nic_in_30 + group_start_30
   - group_end_25 <= 80
 constr_link_match_1_25_31: 80 switch_nic_out_25_nic_in_31 - group_start_31
   + group_end_25 <= 80
 constr_link_match_2_25_31: 80 switch_nic_out_25_nic_in_31 + group_start_31
   - group_end_25 <= 80
 constr_is_start_zero_26: start_26
   + 80 is_start_zero_26_for_source_nic_in_26 <= 80
 constr_edge_switch_start_zero_26_OR: switch_source_nic_in_26
   - is_start_zero_26_for_source_nic_in_26 <= 0
 constr_is_end_complete_26: start_26 + hold_26_1 + 2 hold_26_2
   + 3 hold_26_3 + 4 hold_26_4 + 5 hold_26_5 + 6 hold_26_6
   - 80 is_end_complete_26 >= 0
 constr_edge_switch_end_max_26_OR: switch_nic_out_26_sink
   - is_end_complete_26 <= 0
 constr_link_match_1_26_0: 80 switch_nic_out_26_nic_in_0 - group_start_0
   + group_end_26 <= 80
 constr_link_match_2_26_0: 80 switch_nic_out_26_nic_in_0 + group_start_0
   - group_end_26 <= 80
 constr_link_match_1_26_1: 80 switch_nic_out_26_nic_in_1 - group_start_1
   + group_end_26 <= 80
 constr_link_match_2_26_1: 80 switch_nic_out_26_nic_in_1 + group_start_1
   - group_end_26 <= 80
 constr_link_match_1_26_2: 80 switch_nic_out_26_nic_in_2 - group_start_2
   + group_end_26 <= 80
 constr_link_match_2_26_2: 80 switch_nic_out_26_nic_in_2 + group_start_2
   - group_end_26 <= 80
 constr_link_match_1_26_3: 80 switch_nic_out_26_nic_in_3 - group_start_3
   + group_end_26 <= 80
 constr_link_match_2_26_3: 80 switch_nic_out_26_nic_in_3 + group_start_3
   - group_end_26 <= 80
 constr_link_match_1_26_4: 80 switch_nic_out_26_nic_in_4 - group_start_4
   + group_end_26 <= 80
 constr_link_match_2_26_4: 80 switch_nic_out_26_nic_in_4 + group_start_4
   - group_end_26 <= 80
 constr_link_match_1_26_5: 80 switch_nic_out_26_nic_in_5 - group_start_5
   + group_end_26 <= 80
 constr_link_match_2_26_5: 80 switch_nic_out_26_nic_in_5 + group_start_5
   - group_end_26 <= 80
 constr_link_match_1_26_6: 80 switch_nic_out_26_nic_in_6 - group_start_6
   + group_end_26 <= 80
 constr_link_match_2_26_6: 80 switch_nic_out_26_nic_in_6 + group_start_6
   - group_end_26 <= 80
 constr_link_match_1_26_7: 80 switch_nic_out_26_nic_in_7 - group_start_7
   + group_end_26 <= 80
 constr_link_match_2_26_7: 80 switch_nic_out_26_nic_in_7 + group_start_7
   - group_end_26 <= 80
 constr_link_match_1_26_8: 80 switch_nic_out_26_nic_in_8 - group_start_8
   + group_end_26 <= 80
 constr_link_match_2_26_8: 80 switch_nic_out_26_nic_in_8 + group_start_8
   - group_end_26 <= 80
 constr_link_match_1_26_9: 80 switch_nic_out_26_nic_in_9 - group_start_9
   + group_end_26 <= 80
 constr_link_match_2_26_9: 80 switch_nic_out_26_nic_in_9 + group_start_9
   - group_end_26 <= 80
 constr_link_match_1_26_10: 80 switch_nic_out_26_nic_in_10 - group_start_10
   + group_end_26 <= 80
 constr_link_match_2_26_10: 80 switch_nic_out_26_nic_in_10 + group_start_10
   - group_end_26 <= 80
 constr_link_match_1_26_11: 80 switch_nic_out_26_nic_in_11 - group_start_11
   + group_end_26 <= 80
 constr_link_match_2_26_11: 80 switch_nic_out_26_nic_in_11 + group_start_11
   - group_end_26 <= 80
 constr_link_match_1_26_12: 80 switch_nic_out_26_nic_in_12 - group_start_12
   + group_end_26 <= 80
 constr_link_match_2_26_12: 80 switch_nic_out_26_nic_in_12 + group_start_12
   - group_end_26 <= 80
 constr_link_match_1_26_13: 80 switch_nic_out_26_nic_in_13 - group_start_13
   + group_end_26 <= 80
 constr_link_match_2_26_13: 80 switch_nic_out_26_nic_in_13 + group_start_13
   - group_end_26 <= 80
 constr_link_match_1_26_14: 80 switch_nic_out_26_nic_in_14 - group_start_14
   + group_end_26 <= 80
 constr_link_match_2_26_14: 80 switch_nic_out_26_nic_in_14 + group_start_14
   - group_end_26 <= 80
 constr_link_match_1_26_15: 80 switch_nic_out_26_nic_in_15 - group_start_15
   + group_end_26 <= 80
 constr_link_match_2_26_15: 80 switch_nic_out_26_nic_in_15 + group_start_15
   - group_end_26 <= 80
 constr_link_match_1_26_16: 80 switch_nic_out_26_nic_in_16 - group_start_16
   + group_end_26 <= 80
 constr_link_match_2_26_16: 80 switch_nic_out_26_nic_in_16 + group_start_16
   - group_end_26 <= 80
 constr_link_match_1_26_17: 80 switch_nic_out_26_nic_in_17 - group_start_17
   + group_end_26 <= 80
 constr_link_match_2_26_17: 80 switch_nic_out_26_nic_in_17 + group_start_17
   - group_end_26 <= 80
 constr_link_match_1_26_18: 80 switch_nic_out_26_nic_in_18 - group_start_18
   + group_end_26 <= 80
 constr_link_match_2_26_18: 80 switch_nic_out_26_nic_in_18 + group_start_18
   - group_end_26 <= 80
 constr_link_match_1_26_19: 80 switch_nic_out_26_nic_in_19 - group_start_19
   + group_end_26 <= 80
 constr_link_match_2_26_19: 80 switch_nic_out_26_nic_in_19 + group_start_19
   - group_end_26 <= 80
 constr_link_match_1_26_20: 80 switch_nic_out_26_nic_in_20 - group_start_20
   + group_end_26 <= 80
 constr_link_match_2_26_20: 80 switch_nic_out_26_nic_in_20 + group_start_20
   - group_end_26 <= 80
 constr_link_match_1_26_21: 80 switch_nic_out_26_nic_in_21 - group_start_21
   + group_end_26 <= 80
 constr_link_match_2_26_21: 80 switch_nic_out_26_nic_in_21 + group_start_21
   - group_end_26 <= 80
 constr_link_match_1_26_22: 80 switch_nic_out_26_nic_in_22 - group_start_22
   + group_end_26 <= 80
 constr_link_match_2_26_22: 80 switch_nic_out_26_nic_in_22 + group_start_22
   - group_end_26 <= 80
 constr_link_match_1_26_23: 80 switch_nic_out_26_nic_in_23 - group_start_23
   + group_end_26 <= 80
 constr_link_match_2_26_23: 80 switch_nic_out_26_nic_in_23 + group_start_23
   - group_end_26 <= 80
 constr_link_match_1_26_24: 80 switch_nic_out_26_nic_in_24 - group_start_24
   + group_end_26 <= 80
 constr_link_match_2_26_24: 80 switch_nic_out_26_nic_in_24 + group_start_24
   - group_end_26 <= 80
 constr_link_match_1_26_25: 80 switch_nic_out_26_nic_in_25 - group_start_25
   + group_end_26 <= 80
 constr_link_match_2_26_25: 80 switch_nic_out_26_nic_in_25 + group_start_25
   - group_end_26 <= 80
 constr_link_match_1_26_27: 80 switch_nic_out_26_nic_in_27 - group_start_27
   + group_end_26 <= 80
 constr_link_match_2_26_27: 80 switch_nic_out_26_nic_in_27 + group_start_27
   - group_end_26 <= 80
 constr_link_match_1_26_28: 80 switch_nic_out_26_nic_in_28 - group_start_28
   + group_end_26 <= 80
 constr_link_match_2_26_28: 80 switch_nic_out_26_nic_in_28 + group_start_28
   - group_end_26 <= 80
 constr_link_match_1_26_29: 80 switch_nic_out_26_nic_in_29 - group_start_29
   + group_end_26 <= 80
 constr_link_match_2_26_29: 80 switch_nic_out_26_nic_in_29 + group_start_29
   - group_end_26 <= 80
 constr_link_match_1_26_30: 80 switch_nic_out_26_nic_in_30 - group_start_30
   + group_end_26 <= 80
 constr_link_match_2_26_30: 80 switch_nic_out_26_nic_in_30 + group_start_30
   - group_end_26 <= 80
 constr_link_match_1_26_31: 80 switch_nic_out_26_nic_in_31 - group_start_31
   + group_end_26 <= 80
 constr_link_match_2_26_31: 80 switch_nic_out_26_nic_in_31 + group_start_31
   - group_end_26 <= 80
 constr_is_start_zero_27: start_27
   + 80 is_start_zero_27_for_source_nic_in_27 <= 80
 constr_edge_switch_start_zero_27_OR: switch_source_nic_in_27
   - is_start_zero_27_for_source_nic_in_27 <= 0
 constr_is_end_complete_27: start_27 + hold_27_1 + 2 hold_27_2
   + 3 hold_27_3 + 4 hold_27_4 + 5 hold_27_5 + 6 hold_27_6
   - 80 is_end_complete_27 >= 0
 constr_edge_switch_end_max_27_OR: switch_nic_out_27_sink
   - is_end_complete_27 <= 0
 constr_link_match_1_27_0: 80 switch_nic_out_27_nic_in_0 - group_start_0
   + group_end_27 <= 80
 constr_link_match_2_27_0: 80 switch_nic_out_27_nic_in_0 + group_start_0
   - group_end_27 <= 80
 constr_link_match_1_27_1: 80 switch_nic_out_27_nic_in_1 - group_start_1
   + group_end_27 <= 80
 constr_link_match_2_27_1: 80 switch_nic_out_27_nic_in_1 + group_start_1
   - group_end_27 <= 80
 constr_link_match_1_27_2: 80 switch_nic_out_27_nic_in_2 - group_start_2
   + group_end_27 <= 80
 constr_link_match_2_27_2: 80 switch_nic_out_27_nic_in_2 + group_start_2
   - group_end_27 <= 80
 constr_link_match_1_27_3: 80 switch_nic_out_27_nic_in_3 - group_start_3
   + group_end_27 <= 80
 constr_link_match_2_27_3: 80 switch_nic_out_27_nic_in_3 + group_start_3
   - group_end_27 <= 80
 constr_link_match_1_27_4: 80 switch_nic_out_27_nic_in_4 - group_start_4
   + group_end_27 <= 80
 constr_link_match_2_27_4: 80 switch_nic_out_27_nic_in_4 + group_start_4
   - group_end_27 <= 80
 constr_link_match_1_27_5: 80 switch_nic_out_27_nic_in_5 - group_start_5
   + group_end_27 <= 80
 constr_link_match_2_27_5: 80 switch_nic_out_27_nic_in_5 + group_start_5
   - group_end_27 <= 80
 constr_link_match_1_27_6: 80 switch_nic_out_27_nic_in_6 - group_start_6
   + group_end_27 <= 80
 constr_link_match_2_27_6: 80 switch_nic_out_27_nic_in_6 + group_start_6
   - group_end_27 <= 80
 constr_link_match_1_27_7: 80 switch_nic_out_27_nic_in_7 - group_start_7
   + group_end_27 <= 80
 constr_link_match_2_27_7: 80 switch_nic_out_27_nic_in_7 + group_start_7
   - group_end_27 <= 80
 constr_link_match_1_27_8: 80 switch_nic_out_27_nic_in_8 - group_start_8
   + group_end_27 <= 80
 constr_link_match_2_27_8: 80 switch_nic_out_27_nic_in_8 + group_start_8
   - group_end_27 <= 80
 constr_link_match_1_27_9: 80 switch_nic_out_27_nic_in_9 - group_start_9
   + group_end_27 <= 80
 constr_link_match_2_27_9: 80 switch_nic_out_27_nic_in_9 + group_start_9
   - group_end_27 <= 80
 constr_link_match_1_27_10: 80 switch_nic_out_27_nic_in_10 - group_start_10
   + group_end_27 <= 80
 constr_link_match_2_27_10: 80 switch_nic_out_27_nic_in_10 + group_start_10
   - group_end_27 <= 80
 constr_link_match_1_27_11: 80 switch_nic_out_27_nic_in_11 - group_start_11
   + group_end_27 <= 80
 constr_link_match_2_27_11: 80 switch_nic_out_27_nic_in_11 + group_start_11
   - group_end_27 <= 80
 constr_link_match_1_27_12: 80 switch_nic_out_27_nic_in_12 - group_start_12
   + group_end_27 <= 80
 constr_link_match_2_27_12: 80 switch_nic_out_27_nic_in_12 + group_start_12
   - group_end_27 <= 80
 constr_link_match_1_27_13: 80 switch_nic_out_27_nic_in_13 - group_start_13
   + group_end_27 <= 80
 constr_link_match_2_27_13: 80 switch_nic_out_27_nic_in_13 + group_start_13
   - group_end_27 <= 80
 constr_link_match_1_27_14: 80 switch_nic_out_27_nic_in_14 - group_start_14
   + group_end_27 <= 80
 constr_link_match_2_27_14: 80 switch_nic_out_27_nic_in_14 + group_start_14
   - group_end_27 <= 80
 constr_link_match_1_27_15: 80 switch_nic_out_27_nic_in_15 - group_start_15
   + group_end_27 <= 80
 constr_link_match_2_27_15: 80 switch_nic_out_27_nic_in_15 + group_start_15
   - group_end_27 <= 80
 constr_link_match_1_27_16: 80 switch_nic_out_27_nic_in_16 - group_start_16
   + group_end_27 <= 80
 constr_link_match_2_27_16: 80 switch_nic_out_27_nic_in_16 + group_start_16
   - group_end_27 <= 80
 constr_link_match_1_27_17: 80 switch_nic_out_27_nic_in_17 - group_start_17
   + group_end_27 <= 80
 constr_link_match_2_27_17: 80 switch_nic_out_27_nic_in_17 + group_start_17
   - group_end_27 <= 80
 constr_link_match_1_27_18: 80 switch_nic_out_27_nic_in_18 - group_start_18
   + group_end_27 <= 80
 constr_link_match_2_27_18: 80 switch_nic_out_27_nic_in_18 + group_start_18
   - group_end_27 <= 80
 constr_link_match_1_27_19: 80 switch_nic_out_27_nic_in_19 - group_start_19
   + group_end_27 <= 80
 constr_link_match_2_27_19: 80 switch_nic_out_27_nic_in_19 + group_start_19
   - group_end_27 <= 80
 constr_link_match_1_27_20: 80 switch_nic_out_27_nic_in_20 - group_start_20
   + group_end_27 <= 80
 constr_link_match_2_27_20: 80 switch_nic_out_27_nic_in_20 + group_start_20
   - group_end_27 <= 80
 constr_link_match_1_27_21: 80 switch_nic_out_27_nic_in_21 - group_start_21
   + group_end_27 <= 80
 constr_link_match_2_27_21: 80 switch_nic_out_27_nic_in_21 + group_start_21
   - group_end_27 <= 80
 constr_link_match_1_27_22: 80 switch_nic_out_27_nic_in_22 - group_start_22
   + group_end_27 <= 80
 constr_link_match_2_27_22: 80 switch_nic_out_27_nic_in_22 + group_start_22
   - group_end_27 <= 80
 constr_link_match_1_27_23: 80 switch_nic_out_27_nic_in_23 - group_start_23
   + group_end_27 <= 80
 constr_link_match_2_27_23: 80 switch_nic_out_27_nic_in_23 + group_start_23
   - group_end_27 <= 80
 constr_link_match_1_27_24: 80 switch_nic_out_27_nic_in_24 - group_start_24
   + group_end_27 <= 80
 constr_link_match_2_27_24: 80 switch_nic_out_27_nic_in_24 + group_start_24
   - group_end_27 <= 80
 constr_link_match_1_27_25: 80 switch_nic_out_27_nic_in_25 - group_start_25
   + group_end_27 <= 80
 constr_link_match_2_27_25: 80 switch_nic_out_27_nic_in_25 + group_start_25
   - group_end_27 <= 80
 constr_link_match_1_27_26: 80 switch_nic_out_27_nic_in_26 - group_start_26
   + group_end_27 <= 80
 constr_link_match_2_27_26: 80 switch_nic_out_27_nic_in_26 + group_start_26
   - group_end_27 <= 80
 constr_link_match_1_27_28: 80 switch_nic_out_27_nic_in_28 - group_start_28
   + group_end_27 <= 80
 constr_link_match_2_27_28: 80 switch_nic_out_27_nic_in_28 + group_start_28
   - group_end_27 <= 80
 constr_link_match_1_27_29: 80 switch_nic_out_27_nic_in_29 - group_start_29
   + group_end_27 <= 80
 constr_link_match_2_27_29: 80 switch_nic_out_27_nic_in_29 + group_start_29
   - group_end_27 <= 80
 constr_link_match_1_27_30: 80 switch_nic_out_27_nic_in_30 - group_start_30
   + group_end_27 <= 80
 constr_link_match_2_27_30: 80 switch_nic_out_27_nic_in_30 + group_start_30
   - group_end_27 <= 80
 constr_link_match_1_27_31: 80 switch_nic_out_27_nic_in_31 - group_start_31
   + group_end_27 <= 80
 constr_link_match_2_27_31: 80 switch_nic_out_27_nic_in_31 + group_start_31
   - group_end_27 <= 80
 constr_is_start_zero_28: start_28
   + 80 is_start_zero_28_for_source_nic_in_28 <= 80
 constr_edge_switch_start_zero_28_OR: switch_source_nic_in_28
   - is_start_zero_28_for_source_nic_in_28 <= 0
 constr_is_end_complete_28: start_28 + hold_28_1 + 2 hold_28_2
   + 3 hold_28_3 + 4 hold_28_4 + 5 hold_28_5 + 6 hold_28_6
   - 80 is_end_complete_28 >= 0
 constr_edge_switch_end_max_28_OR: switch_nic_out_28_sink
   - is_end_complete_28 <= 0
 constr_link_match_1_28_0: 80 switch_nic_out_28_nic_in_0 - group_start_0
   + group_end_28 <= 80
 constr_link_match_2_28_0: 80 switch_nic_out_28_nic_in_0 + group_start_0
   - group_end_28 <= 80
 constr_link_match_1_28_1: 80 switch_nic_out_28_nic_in_1 - group_start_1
   + group_end_28 <= 80
 constr_link_match_2_28_1: 80 switch_nic_out_28_nic_in_1 + group_start_1
   - group_end_28 <= 80
 constr_link_match_1_28_2: 80 switch_nic_out_28_nic_in_2 - group_start_2
   + group_end_28 <= 80
 constr_link_match_2_28_2: 80 switch_nic_out_28_nic_in_2 + group_start_2
   - group_end_28 <= 80
 constr_link_match_1_28_3: 80 switch_nic_out_28_nic_in_3 - group_start_3
   + group_end_28 <= 80
 constr_link_match_2_28_3: 80 switch_nic_out_28_nic_in_3 + group_start_3
   - group_end_28 <= 80
 constr_link_match_1_28_4: 80 switch_nic_out_28_nic_in_4 - group_start_4
   + group_end_28 <= 80
 constr_link_match_2_28_4: 80 switch_nic_out_28_nic_in_4 + group_start_4
   - group_end_28 <= 80
 constr_link_match_1_28_5: 80 switch_nic_out_28_nic_in_5 - group_start_5
   + group_end_28 <= 80
 constr_link_match_2_28_5: 80 switch_nic_out_28_nic_in_5 + group_start_5
   - group_end_28 <= 80
 constr_link_match_1_28_6: 80 switch_nic_out_28_nic_in_6 - group_start_6
   + group_end_28 <= 80
 constr_link_match_2_28_6: 80 switch_nic_out_28_nic_in_6 + group_start_6
   - group_end_28 <= 80
 constr_link_match_1_28_7: 80 switch_nic_out_28_nic_in_7 - group_start_7
   + group_end_28 <= 80
 constr_link_match_2_28_7: 80 switch_nic_out_28_nic_in_7 + group_start_7
   - group_end_28 <= 80
 constr_link_match_1_28_8: 80 switch_nic_out_28_nic_in_8 - group_start_8
   + group_end_28 <= 80
 constr_link_match_2_28_8: 80 switch_nic_out_28_nic_in_8 + group_start_8
   - group_end_28 <= 80
 constr_link_match_1_28_9: 80 switch_nic_out_28_nic_in_9 - group_start_9
   + group_end_28 <= 80
 constr_link_match_2_28_9: 80 switch_nic_out_28_nic_in_9 + group_start_9
   - group_end_28 <= 80
 constr_link_match_1_28_10: 80 switch_nic_out_28_nic_in_10 - group_start_10
   + group_end_28 <= 80
 constr_link_match_2_28_10: 80 switch_nic_out_28_nic_in_10 + group_start_10
   - group_end_28 <= 80
 constr_link_match_1_28_11: 80 switch_nic_out_28_nic_in_11 - group_start_11
   + group_end_28 <= 80
 constr_link_match_2_28_11: 80 switch_nic_out_28_nic_in_11 + group_start_11
   - group_end_28 <= 80
 constr_link_match_1_28_12: 80 switch_nic_out_28_nic_in_12 - group_start_12
   + group_end_28 <= 80
 constr_link_match_2_28_12: 80 switch_nic_out_28_nic_in_12 + group_start_12
   - group_end_28 <= 80
 constr_link_match_1_28_13: 80 switch_nic_out_28_nic_in_13 - group_start_13
   + group_end_28 <= 80
 constr_link_match_2_28_13: 80 switch_nic_out_28_nic_in_13 + group_start_13
   - group_end_28 <= 80
 constr_link_match_1_28_14: 80 switch_nic_out_28_nic_in_14 - group_start_14
   + group_end_28 <= 80
 constr_link_match_2_28_14: 80 switch_nic_out_28_nic_in_14 + group_start_14
   - group_end_28 <= 80
 constr_link_match_1_28_15: 80 switch_nic_out_28_nic_in_15 - group_start_15
   + group_end_28 <= 80
 constr_link_match_2_28_15: 80 switch_nic_out_28_nic_in_15 + group_start_15
   - group_end_28 <= 80
 constr_link_match_1_28_16: 80 switch_nic_out_28_nic_in_16 - group_start_16
   + group_end_28 <= 80
 constr_link_match_2_28_16: 80 switch_nic_out_28_nic_in_16 + group_start_16
   - group_end_28 <= 80
 constr_link_match_1_28_17: 80 switch_nic_out_28_nic_in_17 - group_start_17
   + group_end_28 <= 80
 constr_link_match_2_28_17: 80 switch_nic_out_28_nic_in_17 + group_start_17
   - group_end_28 <= 80
 constr_link_match_1_28_18: 80 switch_nic_out_28_nic_in_18 - group_start_18
   + group_end_28 <= 80
 constr_link_match_2_28_18: 80 switch_nic_out_28_nic_in_18 + group_start_18
   - group_end_28 <= 80
 constr_link_match_1_28_19: 80 switch_nic_out_28_nic_in_19 - group_start_19
   + group_end_28 <= 80
 constr_link_match_2_28_19: 80 switch_nic_out_28_nic_in_19 + group_start_19
   - group_end_28 <= 80
 constr_link_match_1_28_20: 80 switch_nic_out_28_nic_in_20 - group_start_20
   + group_end_28 <= 80
 constr_link_match_2_28_20: 80 switch_nic_out_28_nic_in_20 + group_start_20
   - group_end_28 <= 80
 constr_link_match_1_28_21: 80 switch_nic_out_28_nic_in_21 - group_start_21
   + group_end_28 <= 80
 constr_link_match_2_28_21: 80 switch_nic_out_28_nic_in_21 + group_start_21
   - group_end_28 <= 80
 constr_link_match_1_28_22: 80 switch_nic_out_28_nic_in_22 - group_start_22
   + group_end_28 <= 80
 constr_link_match_2_28_22: 80 switch_nic_out_28_nic_in_22 + group_start_22
   - group_end_28 <= 80
 constr_link_match_1_28_23: 80 switch_nic_out_28_nic_in_23 - group_start_23
   + group_end_28 <= 80
 constr_link_match_2_28_23: 80 switch_nic_out_28_nic_in_23 + group_start_23
   - group_end_28 <= 80
 constr_link_match_1_28_24: 80 switch_nic_out_28_nic_in_24 - group_start_24
   + group_end_28 <= 80
 constr_link_match_2_28_24: 80 switch_nic_out_28_nic_in_24 + group_start_24
   - group_end_28 <= 80
 constr_link_match_1_28_25: 80 switch_nic_out_28_nic_in_25 - group_start_25
   + group_end_28 <= 80
 constr_link_match_2_28_25: 80 switch_nic_out_28_nic_in_25 + group_start_25
   - group_end_28 <= 80
 constr_link_match_1_28_26: 80 switch_nic_out_28_nic_in_26 - group_start_26
   + group_end_28 <= 80
 constr_link_match_2_28_26: 80 switch_nic_out_28_nic_in_26 + group_start_26
   - group_end_28 <= 80
 constr_link_match_1_28_27: 80 switch_nic_out_28_nic_in_27 - group_start_27
   + group_end_28 <= 80
 constr_link_match_2_28_27: 80 switch_nic_out_28_nic_in_27 + group_start_27
   - group_end_28 <= 80
 constr_link_match_1_28_29: 80 switch_nic_out_28_nic_in_29 - group_start_29
   + group_end_28 <= 80
 constr_link_match_2_28_29: 80 switch_nic_out_28_nic_in_29 + group_start_29
   - group_end_28 <= 80
 constr_link_match_1_28_30: 80 switch_nic_out_28_nic_in_30 - group_start_30
   + group_end_28 <= 80
 constr_link_match_2_28_30: 80 switch_nic_out_28_nic_in_30 + group_start_30
   - group_end_28 <= 80
 constr_link_match_1_28_31: 80 switch_nic_out_28_nic_in_31 - group_start_31
   + group_end_28 <= 80
 constr_link_match_2_28_31: 80 switch_nic_out_28_nic_in_31 + group_start_31
   - group_end_28 <= 80
 constr_is_start_zero_29: start_29
   + 80 is_start_zero_29_for_source_nic_in_29 <= 80
 constr_edge_switch_start_zero_29_OR: switch_source_nic_in_29
   - is_start_zero_29_for_source_nic_in_29 <= 0
 constr_is_end_complete_29: start_29 + hold_29_1 + 2 hold_29_2
   + 3 hold_29_3 + 4 hold_29_4 + 5 hold_29_5 + 6 hold_29_6
   - 80 is_end_complete_29 >= 0
 constr_edge_switch_end_max_29_OR: switch_nic_out_29_sink
   - is_end_complete_29 <= 0
 constr_link_match_1_29_0: 80 switch_nic_out_29_nic_in_0 - group_start_0
   + group_end_29 <= 80
 constr_link_match_2_29_0: 80 switch_nic_out_29_nic_in_0 + group_start_0
   - group_end_29 <= 80
 constr_link_match_1_29_1: 80 switch_nic_out_29_nic_in_1 - group_start_1
   + group_end_29 <= 80
 constr_link_match_2_29_1: 80 switch_nic_out_29_nic_in_1 + group_start_1
   - group_end_29 <= 80
 constr_link_match_1_29_2: 80 switch_nic_out_29_nic_in_2 - group_start_2
   + group_end_29 <= 80
 constr_link_match_2_29_2: 80 switch_nic_out_29_nic_in_2 + group_start_2
   - group_end_29 <= 80
 constr_link_match_1_29_3: 80 switch_nic_out_29_nic_in_3 - group_start_3
   + group_end_29 <= 80
 constr_link_match_2_29_3: 80 switch_nic_out_29_nic_in_3 + group_start_3
   - group_end_29 <= 80
 constr_link_match_1_29_4: 80 switch_nic_out_29_nic_in_4 - group_start_4
   + group_end_29 <= 80
 constr_link_match_2_29_4: 80 switch_nic_out_29_nic_in_4 + group_start_4
   - group_end_29 <= 80
 constr_link_match_1_29_5: 80 switch_nic_out_29_nic_in_5 - group_start_5
   + group_end_29 <= 80
 constr_link_match_2_29_5: 80 switch_nic_out_29_nic_in_5 + group_start_5
   - group_end_29 <= 80
 constr_link_match_1_29_6: 80 switch_nic_out_29_nic_in_6 - group_start_6
   + group_end_29 <= 80
 constr_link_match_2_29_6: 80 switch_nic_out_29_nic_in_6 + group_start_6
   - group_end_29 <= 80
 constr_link_match_1_29_7: 80 switch_nic_out_29_nic_in_7 - group_start_7
   + group_end_29 <= 80
 constr_link_match_2_29_7: 80 switch_nic_out_29_nic_in_7 + group_start_7
   - group_end_29 <= 80
 constr_link_match_1_29_8: 80 switch_nic_out_29_nic_in_8 - group_start_8
   + group_end_29 <= 80
 constr_link_match_2_29_8: 80 switch_nic_out_29_nic_in_8 + group_start_8
   - group_end_29 <= 80
 constr_link_match_1_29_9: 80 switch_nic_out_29_nic_in_9 - group_start_9
   + group_end_29 <= 80
 constr_link_match_2_29_9: 80 switch_nic_out_29_nic_in_9 + group_start_9
   - group_end_29 <= 80
 constr_link_match_1_29_10: 80 switch_nic_out_29_nic_in_10 - group_start_10
   + group_end_29 <= 80
 constr_link_match_2_29_10: 80 switch_nic_out_29_nic_in_10 + group_start_10
   - group_end_29 <= 80
 constr_link_match_1_29_11: 80 switch_nic_out_29_nic_in_11 - group_start_11
   + group_end_29 <= 80
 constr_link_match_2_29_11: 80 switch_nic_out_29_nic_in_11 + group_start_11
   - group_end_29 <= 80
 constr_link_match_1_29_12: 80 switch_nic_out_29_nic_in_12 - group_start_12
   + group_end_29 <= 80
 constr_link_match_2_29_12: 80 switch_nic_out_29_nic_in_12 + group_start_12
   - group_end_29 <= 80
 constr_link_match_1_29_13: 80 switch_nic_out_29_nic_in_13 - group_start_13
   + group_end_29 <= 80
 constr_link_match_2_29_13: 80 switch_nic_out_29_nic_in_13 + group_start_13
   - group_end_29 <= 80
 constr_link_match_1_29_14: 80 switch_nic_out_29_nic_in_14 - group_start_14
   + group_end_29 <= 80
 constr_link_match_2_29_14: 80 switch_nic_out_29_nic_in_14 + group_start_14
   - group_end_29 <= 80
 constr_link_match_1_29_15: 80 switch_nic_out_29_nic_in_15 - group_start_15
   + group_end_29 <= 80
 constr_link_match_2_29_15: 80 switch_nic_out_29_nic_in_15 + group_start_15
   - group_end_29 <= 80
 constr_link_match_1_29_16: 80 switch_nic_out_29_nic_in_16 - group_start_16
   + group_end_29 <= 80
 constr_link_match_2_29_16: 80 switch_nic_out_29_nic_in_16 + group_start_16
   - group_end_29 <= 80
 constr_link_match_1_29_17: 80 switch_nic_out_29_nic_in_17 - group_start_17
   + group_end_29 <= 80
 constr_link_match_2_29_17: 80 switch_nic_out_29_nic_in_17 + group_start_17
   - group_end_29 <= 80
 constr_link_match_1_29_18: 80 switch_nic_out_29_nic_in_18 - group_start_18
   + group_end_29 <= 80
 constr_link_match_2_29_18: 80 switch_nic_out_29_nic_in_18 + group_start_18
   - group_end_29 <= 80
 constr_link_match_1_29_19: 80 switch_nic_out_29_nic_in_19 - group_start_19
   + group_end_29 <= 80
 constr_link_match_2_29_19: 80 switch_nic_out_29_nic_in_19 + group_start_19
   - group_end_29 <= 80
 constr_link_match_1_29_20: 80 switch_nic_out_29_nic_in_20 - group_start_20
   + group_end_29 <= 80
 constr_link_match_2_29_20: 80 switch_nic_out_29_nic_in_20 + group_start_20
   - group_end_29 <= 80
 constr_link_match_1_29_21: 80 switch_nic_out_29_nic_in_21 - group_start_21
   + group_end_29 <= 80
 constr_link_match_2_29_21: 80 switch_nic_out_29_nic_in_21 + group_start_21
   - group_end_29 <= 80
 constr_link_match_1_29_22: 80 switch_nic_out_29_nic_in_22 - group_start_22
   + group_end_29 <= 80
 constr_link_match_2_29_22: 80 switch_nic_out_29_nic_in_22 + group_start_22
   - group_end_29 <= 80
 constr_link_match_1_29_23: 80 switch_nic_out_29_nic_in_23 - group_start_23
   + group_end_29 <= 80
 constr_link_match_2_29_23: 80 switch_nic_out_29_nic_in_23 + group_start_23
   - group_end_29 <= 80
 constr_link_match_1_29_24: 80 switch_nic_out_29_nic_in_24 - group_start_24
   + group_end_29 <= 80
 constr_link_match_2_29_24: 80 switch_nic_out_29_nic_in_24 + group_start_24
   - group_end_29 <= 80
 constr_link_match_1_29_25: 80 switch_nic_out_29_nic_in_25 - group_start_25
   + group_end_29 <= 80
 constr_link_match_2_29_25: 80 switch_nic_out_29_nic_in_25 + group_start_25
   - group_end_29 <= 80
 constr_link_match_1_29_26: 80 switch_nic_out_29_nic_in_26 - group_start_26
   + group_end_29 <= 80
 constr_link_match_2_29_26: 80 switch_nic_out_29_nic_in_26 + group_start_26
   - group_end_29 <= 80
 constr_link_match_1_29_27: 80 switch_nic_out_29_nic_in_27 - group_start_27
   + group_end_29 <= 80
 constr_link_match_2_29_27: 80 switch_nic_out_29_nic_in_27 + group_start_27
   - group_end_29 <= 80
 constr_link_match_1_29_28: 80 switch_nic_out_29_nic_in_28 - group_start_28
   + group_end_29 <= 80
 constr_link_match_2_29_28: 80 switch_nic_out_29_nic_in_28 + group_start_28
   - group_end_29 <= 80
 constr_link_match_1_29_30: 80 switch_nic_out_29_nic_in_30 - group_start_30
   + group_end_29 <= 80
 constr_link_match_2_29_30: 80 switch_nic_out_29_nic_in_30 + group_start_30
   - group_end_29 <= 80
 constr_link_match_1_29_31: 80 switch_nic_out_29_nic_in_31 - group_start_31
   + group_end_29 <= 80
 constr_link_match_2_29_31: 80 switch_nic_out_29_nic_in_31 + group_start_31
   - group_end_29 <= 80
 constr_is_start_zero_30: start_30
   + 80 is_start_zero_30_for_source_nic_in_30 <= 80
 constr_edge_switch_start_zero_30_OR: switch_source_nic_in_30
   - is_start_zero_30_for_source_nic_in_30 <= 0
 constr_is_end_complete_30: start_30 + hold_30_1 + 2 hold_30_2
   + 3 hold_30_3 + 4 hold_30_4 + 5 hold_30_5 + 6 hold_30_6
   - 80 is_end_complete_30 >= 0
 constr_edge_switch_end_max_30_OR: switch_nic_out_30_sink
   - is_end_complete_30 <= 0
 constr_link_match_1_30_0: 80 switch_nic_out_30_nic_in_0 - group_start_0
   + group_end_30 <= 80
 constr_link_match_2_30_0: 80 switch_nic_out_30_nic_in_0 + group_start_0
   - group_end_30 <= 80
 constr_link_match_1_30_1: 80 switch_nic_out_30_nic_in_1 - group_start_1
   + group_end_30 <= 80
 constr_link_match_2_30_1: 80 switch_nic_out_30_nic_in_1 + group_start_1
   - group_end_30 <= 80
 constr_link_match_1_30_2: 80 switch_nic_out_30_nic_in_2 - group_start_2
   + group_end_30 <= 80
 constr_link_match_2_30_2: 80 switch_nic_out_30_nic_in_2 + group_start_2
   - group_end_30 <= 80
 constr_link_match_1_30_3: 80 switch_nic_out_30_nic_in_3 - group_start_3
   + group_end_30 <= 80
 constr_link_match_2_30_3: 80 switch_nic_out_30_nic_in_3 + group_start_3
   - group_end_30 <= 80
 constr_link_match_1_30_4: 80 switch_nic_out_30_nic_in_4 - group_start_4
   + group_end_30 <= 80
 constr_link_match_2_30_4: 80 switch_nic_out_30_nic_in_4 + group_start_4
   - group_end_30 <= 80
 constr_link_match_1_30_5: 80 switch_nic_out_30_nic_in_5 - group_start_5
   + group_end_30 <= 80
 constr_link_match_2_30_5: 80 switch_nic_out_30_nic_in_5 + group_start_5
   - group_end_30 <= 80
 constr_link_match_1_30_6: 80 switch_nic_out_30_nic_in_6 - group_start_6
   + group_end_30 <= 80
 constr_link_match_2_30_6: 80 switch_nic_out_30_nic_in_6 + group_start_6
   - group_end_30 <= 80
 constr_link_match_1_30_7: 80 switch_nic_out_30_nic_in_7 - group_start_7
   + group_end_30 <= 80
 constr_link_match_2_30_7: 80 switch_nic_out_30_nic_in_7 + group_start_7
   - group_end_30 <= 80
 constr_link_match_1_30_8: 80 switch_nic_out_30_nic_in_8 - group_start_8
   + group_end_30 <= 80
 constr_link_match_2_30_8: 80 switch_nic_out_30_nic_in_8 + group_start_8
   - group_end_30 <= 80
 constr_link_match_1_30_9: 80 switch_nic_out_30_nic_in_9 - group_start_9
   + group_end_30 <= 80
 constr_link_match_2_30_9: 80 switch_nic_out_30_nic_in_9 + group_start_9
   - group_end_30 <= 80
 constr_link_match_1_30_10: 80 switch_nic_out_30_nic_in_10 - group_start_10
   + group_end_30 <= 80
 constr_link_match_2_30_10: 80 switch_nic_out_30_nic_in_10 + group_start_10
   - group_end_30 <= 80
 constr_link_match_1_30_11: 80 switch_nic_out_30_nic_in_11 - group_start_11
   + group_end_30 <= 80
 constr_link_match_2_30_11: 80 switch_nic_out_30_nic_in_11 + group_start_11
   - group_end_30 <= 80
 constr_link_match_1_30_12: 80 switch_nic_out_30_nic_in_12 - group_start_12
   + group_end_30 <= 80
 constr_link_match_2_30_12: 80 switch_nic_out_30_nic_in_12 + group_start_12
   - group_end_30 <= 80
 constr_link_match_1_30_13: 80 switch_nic_out_30_nic_in_13 - group_start_13
   + group_end_30 <= 80
 constr_link_match_2_30_13: 80 switch_nic_out_30_nic_in_13 + group_start_13
   - group_end_30 <= 80
 constr_link_match_1_30_14: 80 switch_nic_out_30_nic_in_14 - group_start_14
   + group_end_30 <= 80
 constr_link_match_2_30_14: 80 switch_nic_out_30_nic_in_14 + group_start_14
   - group_end_30 <= 80
 constr_link_match_1_30_15: 80 switch_nic_out_30_nic_in_15 - group_start_15
   + group_end_30 <= 80
 constr_link_match_2_30_15: 80 switch_nic_out_30_nic_in_15 + group_start_15
   - group_end_30 <= 80
 constr_link_match_1_30_16: 80 switch_nic_out_30_nic_in_16 - group_start_16
   + group_end_30 <= 80
 constr_link_match_2_30_16: 80 switch_nic_out_30_nic_in_16 + group_start_16
   - group_end_30 <= 80
 constr_link_match_1_30_17: 80 switch_nic_out_30_nic_in_17 - group_start_17
   + group_end_30 <= 80
 constr_link_match_2_30_17: 80 switch_nic_out_30_nic_in_17 + group_start_17
   - group_end_30 <= 80
 constr_link_match_1_30_18: 80 switch_nic_out_30_nic_in_18 - group_start_18
   + group_end_30 <= 80
 constr_link_match_2_30_18: 80 switch_nic_out_30_nic_in_18 + group_start_18
   - group_end_30 <= 80
 constr_link_match_1_30_19: 80 switch_nic_out_30_nic_in_19 - group_start_19
   + group_end_30 <= 80
 constr_link_match_2_30_19: 80 switch_nic_out_30_nic_in_19 + group_start_19
   - group_end_30 <= 80
 constr_link_match_1_30_20: 80 switch_nic_out_30_nic_in_20 - group_start_20
   + group_end_30 <= 80
 constr_link_match_2_30_20: 80 switch_nic_out_30_nic_in_20 + group_start_20
   - group_end_30 <= 80
 constr_link_match_1_30_21: 80 switch_nic_out_30_nic_in_21 - group_start_21
   + group_end_30 <= 80
 constr_link_match_2_30_21: 80 switch_nic_out_30_nic_in_21 + group_start_21
   - group_end_30 <= 80
 constr_link_match_1_30_22: 80 switch_nic_out_30_nic_in_22 - group_start_22
   + group_end_30 <= 80
 constr_link_match_2_30_22: 80 switch_nic_out_30_nic_in_22 + group_start_22
   - group_end_30 <= 80
 constr_link_match_1_30_23: 80 switch_nic_out_30_nic_in_23 - group_start_23
   + group_end_30 <= 80
 constr_link_match_2_30_23: 80 switch_nic_out_30_nic_in_23 + group_start_23
   - group_end_30 <= 80
 constr_link_match_1_30_24: 80 switch_nic_out_30_nic_in_24 - group_start_24
   + group_end_30 <= 80
 constr_link_match_2_30_24: 80 switch_nic_out_30_nic_in_24 + group_start_24
   - group_end_30 <= 80
 constr_link_match_1_30_25: 80 switch_nic_out_30_nic_in_25 - group_start_25
   + group_end_30 <= 80
 constr_link_match_2_30_25: 80 switch_nic_out_30_nic_in_25 + group_start_25
   - group_end_30 <= 80
 constr_link_match_1_30_26: 80 switch_nic_out_30_nic_in_26 - group_start_26
   + group_end_30 <= 80
 constr_link_match_2_30_26: 80 switch_nic_out_30_nic_in_26 + group_start_26
   - group_end_30 <= 80
 constr_link_match_1_30_27: 80 switch_nic_out_30_nic_in_27 - group_start_27
   + group_end_30 <= 80
 constr_link_match_2_30_27: 80 switch_nic_out_30_nic_in_27 + group_start_27
   - group_end_30 <= 80
 constr_link_match_1_30_28: 80 switch_nic_out_30_nic_in_28 - group_start_28
   + group_end_30 <= 80
 constr_link_match_2_30_28: 80 switch_nic_out_30_nic_in_28 + group_start_28
   - group_end_30 <= 80
 constr_link_match_1_30_29: 80 switch_nic_out_30_nic_in_29 - group_start_29
   + group_end_30 <= 80
 constr_link_match_2_30_29: 80 switch_nic_out_30_nic_in_29 + group_start_29
   - group_end_30 <= 80
 constr_link_match_1_30_31: 80 switch_nic_out_30_nic_in_31 - group_start_31
   + group_end_30 <= 80
 constr_link_match_2_30_31: 80 switch_nic_out_30_nic_in_31 + group_start_31
   - group_end_30 <= 80
 constr_is_start_zero_31: start_31
   + 80 is_start_zero_31_for_source_nic_in_31 <= 80
 constr_edge_switch_start_zero_31_OR: switch_source_nic_in_31
   - is_start_zero_31_for_source_nic_in_31 <= 0
 constr_is_end_complete_31: start_31 + hold_31_1 + 2 hold_31_2
   + 3 hold_31_3 + 4 hold_31_4 + 5 hold_31_5 + 6 hold_31_6
   - 80 is_end_complete_31 >= 0
 constr_edge_switch_end_max_31_OR: switch_nic_out_31_sink
   - is_end_complete_31 <= 0
 constr_link_match_1_31_0: 80 switch_nic_out_31_nic_in_0 - group_start_0
   + group_end_31 <= 80
 constr_link_match_2_31_0: 80 switch_nic_out_31_nic_in_0 + group_start_0
   - group_end_31 <= 80
 constr_link_match_1_31_1: 80 switch_nic_out_31_nic_in_1 - group_start_1
   + group_end_31 <= 80
 constr_link_match_2_31_1: 80 switch_nic_out_31_nic_in_1 + group_start_1
   - group_end_31 <= 80
 constr_link_match_1_31_2: 80 switch_nic_out_31_nic_in_2 - group_start_2
   + group_end_31 <= 80
 constr_link_match_2_31_2: 80 switch_nic_out_31_nic_in_2 + group_start_2
   - group_end_31 <= 80
 constr_link_match_1_31_3: 80 switch_nic_out_31_nic_in_3 - group_start_3
   + group_end_31 <= 80
 constr_link_match_2_31_3: 80 switch_nic_out_31_nic_in_3 + group_start_3
   - group_end_31 <= 80
 constr_link_match_1_31_4: 80 switch_nic_out_31_nic_in_4 - group_start_4
   + group_end_31 <= 80
 constr_link_match_2_31_4: 80 switch_nic_out_31_nic_in_4 + group_start_4
   - group_end_31 <= 80
 constr_link_match_1_31_5: 80 switch_nic_out_31_nic_in_5 - group_start_5
   + group_end_31 <= 80
 constr_link_match_2_31_5: 80 switch_nic_out_31_nic_in_5 + group_start_5
   - group_end_31 <= 80
 constr_link_match_1_31_6: 80 switch_nic_out_31_nic_in_6 - group_start_6
   + group_end_31 <= 80
 constr_link_match_2_31_6: 80 switch_nic_out_31_nic_in_6 + group_start_6
   - group_end_31 <= 80
 constr_link_match_1_31_7: 80 switch_nic_out_31_nic_in_7 - group_start_7
   + group_end_31 <= 80
 constr_link_match_2_31_7: 80 switch_nic_out_31_nic_in_7 + group_start_7
   - group_end_31 <= 80
 constr_link_match_1_31_8: 80 switch_nic_out_31_nic_in_8 - group_start_8
   + group_end_31 <= 80
 constr_link_match_2_31_8: 80 switch_nic_out_31_nic_in_8 + group_start_8
   - group_end_31 <= 80
 constr_link_match_1_31_9: 80 switch_nic_out_31_nic_in_9 - group_start_9
   + group_end_31 <= 80
 constr_link_match_2_31_9: 80 switch_nic_out_31_nic_in_9 + group_start_9
   - group_end_31 <= 80
 constr_link_match_1_31_10: 80 switch_nic_out_31_nic_in_10 - group_start_10
   + group_end_31 <= 80
 constr_link_match_2_31_10: 80 switch_nic_out_31_nic_in_10 + group_start_10
   - group_end_31 <= 80
 constr_link_match_1_31_11: 80 switch_nic_out_31_nic_in_11 - group_start_11
   + group_end_31 <= 80
 constr_link_match_2_31_11: 80 switch_nic_out_31_nic_in_11 + group_start_11
   - group_end_31 <= 80
 constr_link_match_1_31_12: 80 switch_nic_out_31_nic_in_12 - group_start_12
   + group_end_31 <= 80
 constr_link_match_2_31_12: 80 switch_nic_out_31_nic_in_12 + group_start_12
   - group_end_31 <= 80
 constr_link_match_1_31_13: 80 switch_nic_out_31_nic_in_13 - group_start_13
   + group_end_31 <= 80
 constr_link_match_2_31_13: 80 switch_nic_out_31_nic_in_13 + group_start_13
   - group_end_31 <= 80
 constr_link_match_1_31_14: 80 switch_nic_out_31_nic_in_14 - group_start_14
   + group_end_31 <= 80
 constr_link_match_2_31_14: 80 switch_nic_out_31_nic_in_14 + group_start_14
   - group_end_31 <= 80
 constr_link_match_1_31_15: 80 switch_nic_out_31_nic_in_15 - group_start_15
   + group_end_31 <= 80
 constr_link_match_2_31_15: 80 switch_nic_out_31_nic_in_15 + group_start_15
   - group_end_31 <= 80
 constr_link_match_1_31_16: 80 switch_nic_out_31_nic_in_16 - group_start_16
   + group_end_31 <= 80
 constr_link_match_2_31_16: 80 switch_nic_out_31_nic_in_16 + group_start_16
   - group_end_31 <= 80
 constr_link_match_1_31_17: 80 switch_nic_out_31_nic_in_17 - group_start_17
   + group_end_31 <= 80
 constr_link_match_2_31_17: 80 switch_nic_out_31_nic_in_17 + group_start_17
   - group_end_31 <= 80
 constr_link_match_1_31_18: 80 switch_nic_out_31_nic_in_18 - group_start_18
   + group_end_31 <= 80
 constr_link_match_2_31_18: 80 switch_nic_out_31_nic_in_18 + group_start_18
   - group_end_31 <= 80
 constr_link_match_1_31_19: 80 switch_nic_out_31_nic_in_19 - group_start_19
   + group_end_31 <= 80
 constr_link_match_2_31_19: 80 switch_nic_out_31_nic_in_19 + group_start_19
   - group_end_31 <= 80
 constr_link_match_1_31_20: 80 switch_nic_out_31_nic_in_20 - group_start_20
   + group_end_31 <= 80
 constr_link_match_2_31_20: 80 switch_nic_out_31_nic_in_20 + group_start_20
   - group_end_31 <= 80
 constr_link_match_1_31_21: 80 switch_nic_out_31_nic_in_21 - group_start_21
   + group_end_31 <= 80
 constr_link_match_2_31_21: 80 switch_nic_out_31_nic_in_21 + group_start_21
   - group_end_31 <= 80
 constr_link_match_1_31_22: 80 switch_nic_out_31_nic_in_22 - group_start_22
   + group_end_31 <= 80
 constr_link_match_2_31_22: 80 switch_nic_out_31_nic_in_22 + group_start_22
   - group_end_31 <= 80
 constr_link_match_1_31_23: 80 switch_nic_out_31_nic_in_23 - group_start_23
   + group_end_31 <= 80
 constr_link_match_2_31_23: 80 switch_nic_out_31_nic_in_23 + group_start_23
   - group_end_31 <= 80
 constr_link_match_1_31_24: 80 switch_nic_out_31_nic_in_24 - group_start_24
   + group_end_31 <= 80
 constr_link_match_2_31_24: 80 switch_nic_out_31_nic_in_24 + group_start_24
   - group_end_31 <= 80
 constr_link_match_1_31_25: 80 switch_nic_out_31_nic_in_25 - group_start_25
   + group_end_31 <= 80
 constr_link_match_2_31_25: 80 switch_nic_out_31_nic_in_25 + group_start_25
   - group_end_31 <= 80
 constr_link_match_1_31_26: 80 switch_nic_out_31_nic_in_26 - group_start_26
   + group_end_31 <= 80
 constr_link_match_2_31_26: 80 switch_nic_out_31_nic_in_26 + group_start_26
   - group_end_31 <= 80
 constr_link_match_1_31_27: 80 switch_nic_out_31_nic_in_27 - group_start_27
   + group_end_31 <= 80
 constr_link_match_2_31_27: 80 switch_nic_out_31_nic_in_27 + group_start_27
   - group_end_31 <= 80
 constr_link_match_1_31_28: 80 switch_nic_out_31_nic_in_28 - group_start_28
   + group_end_31 <= 80
 constr_link_match_2_31_28: 80 switch_nic_out_31_nic_in_28 + group_start_28
   - group_end_31 <= 80
 constr_link_match_1_31_29: 80 switch_nic_out_31_nic_in_29 - group_start_29
   + group_end_31 <= 80
 constr_link_match_2_31_29: 80 switch_nic_out_31_nic_in_29 + group_start_29
   - group_end_31 <= 80
 constr_link_match_1_31_30: 80 switch_nic_out_31_nic_in_30 - group_start_30
   + group_end_31 <= 80
 constr_link_match_2_31_30: 80 switch_nic_out_31_nic_in_30 + group_start_30
   - group_end_31 <= 80
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
 link_activation_dst_nic_in_16_16: switch_nic_in_16_16 - is_active_16 <= 0
 link_activation_src_16_nic_out_16: switch_16_nic_out_16 - is_active_16
   <= 0
 link_activation_dst_nic_in_17_17: switch_nic_in_17_17 - is_active_17 <= 0
 link_activation_src_17_nic_out_17: switch_17_nic_out_17 - is_active_17
   <= 0
 link_activation_dst_nic_in_18_18: switch_nic_in_18_18 - is_active_18 <= 0
 link_activation_src_18_nic_out_18: switch_18_nic_out_18 - is_active_18
   <= 0
 link_activation_dst_nic_in_19_19: switch_nic_in_19_19 - is_active_19 <= 0
 link_activation_src_19_nic_out_19: switch_19_nic_out_19 - is_active_19
   <= 0
 link_activation_dst_nic_in_20_20: switch_nic_in_20_20 - is_active_20 <= 0
 link_activation_src_20_nic_out_20: switch_20_nic_out_20 - is_active_20
   <= 0
 link_activation_dst_nic_in_21_21: switch_nic_in_21_21 - is_active_21 <= 0
 link_activation_src_21_nic_out_21: switch_21_nic_out_21 - is_active_21
   <= 0
 link_activation_dst_nic_in_22_22: switch_nic_in_22_22 - is_active_22 <= 0
 link_activation_src_22_nic_out_22: switch_22_nic_out_22 - is_active_22
   <= 0
 link_activation_dst_nic_in_23_23: switch_nic_in_23_23 - is_active_23 <= 0
 link_activation_src_23_nic_out_23: switch_23_nic_out_23 - is_active_23
   <= 0
 link_activation_dst_nic_in_24_24: switch_nic_in_24_24 - is_active_24 <= 0
 link_activation_src_24_nic_out_24: switch_24_nic_out_24 - is_active_24
   <= 0
 link_activation_dst_nic_in_25_25: switch_nic_in_25_25 - is_active_25 <= 0
 link_activation_src_25_nic_out_25: switch_25_nic_out_25 - is_active_25
   <= 0
 link_activation_dst_nic_in_26_26: switch_nic_in_26_26 - is_active_26 <= 0
 link_activation_src_26_nic_out_26: switch_26_nic_out_26 - is_active_26
   <= 0
 link_activation_dst_nic_in_27_27: switch_nic_in_27_27 - is_active_27 <= 0
 link_activation_src_27_nic_out_27: switch_27_nic_out_27 - is_active_27
   <= 0
 link_activation_dst_nic_in_28_28: switch_nic_in_28_28 - is_active_28 <= 0
 link_activation_src_28_nic_out_28: switch_28_nic_out_28 - is_active_28
   <= 0
 link_activation_dst_nic_in_29_29: switch_nic_in_29_29 - is_active_29 <= 0
 link_activation_src_29_nic_out_29: switch_29_nic_out_29 - is_active_29
   <= 0
 link_activation_dst_nic_in_30_30: switch_nic_in_30_30 - is_active_30 <= 0
 link_activation_src_30_nic_out_30: switch_30_nic_out_30 - is_active_30
   <= 0
 link_activation_dst_nic_in_31_31: switch_nic_in_31_31 - is_active_31 <= 0
 link_activation_src_31_nic_out_31: switch_31_nic_out_31 - is_active_31
   <= 0
 edge_flow_constr_source_nic_in_0: flow_source_nic_in_0
   - 5e+08 switch_source_nic_in_0 <= 0
 edge_flow_constr_nic_out_0_sink: flow_nic_out_0_sink
   - 5e+08 switch_nic_out_0_sink <= 0
 edge_flow_constr_nic_out_0_nic_in_1: flow_nic_out_0_nic_in_1
   - 61035.15625 switch_nic_out_0_nic_in_1 <= 0
 edge_flow_constr_nic_out_0_nic_in_2: flow_nic_out_0_nic_in_2
   - 61035.15625 switch_nic_out_0_nic_in_2 <= 0
 edge_flow_constr_nic_out_0_nic_in_3: flow_nic_out_0_nic_in_3
   - 61035.15625 switch_nic_out_0_nic_in_3 <= 0
 edge_flow_constr_nic_out_0_nic_in_4: flow_nic_out_0_nic_in_4
   - 61035.15625 switch_nic_out_0_nic_in_4 <= 0
 edge_flow_constr_nic_out_0_nic_in_5: flow_nic_out_0_nic_in_5
   - 61035.15625 switch_nic_out_0_nic_in_5 <= 0
 edge_flow_constr_nic_out_0_nic_in_6: flow_nic_out_0_nic_in_6
   - 61035.15625 switch_nic_out_0_nic_in_6 <= 0
 edge_flow_constr_nic_out_0_nic_in_7: flow_nic_out_0_nic_in_7
   - 61035.15625 switch_nic_out_0_nic_in_7 <= 0
 edge_flow_constr_nic_out_0_nic_in_8: flow_nic_out_0_nic_in_8
   - 61035.15625 switch_nic_out_0_nic_in_8 <= 0
 edge_flow_constr_nic_out_0_nic_in_9: flow_nic_out_0_nic_in_9
   - 61035.15625 switch_nic_out_0_nic_in_9 <= 0
 edge_flow_constr_nic_out_0_nic_in_10: flow_nic_out_0_nic_in_10
   - 61035.15625 switch_nic_out_0_nic_in_10 <= 0
 edge_flow_constr_nic_out_0_nic_in_11: flow_nic_out_0_nic_in_11
   - 61035.15625 switch_nic_out_0_nic_in_11 <= 0
 edge_flow_constr_nic_out_0_nic_in_12: flow_nic_out_0_nic_in_12
   - 61035.15625 switch_nic_out_0_nic_in_12 <= 0
 edge_flow_constr_nic_out_0_nic_in_13: flow_nic_out_0_nic_in_13
   - 61035.15625 switch_nic_out_0_nic_in_13 <= 0
 edge_flow_constr_nic_out_0_nic_in_14: flow_nic_out_0_nic_in_14
   - 61035.15625 switch_nic_out_0_nic_in_14 <= 0
 edge_flow_constr_nic_out_0_nic_in_15: flow_nic_out_0_nic_in_15
   - 61035.15625 switch_nic_out_0_nic_in_15 <= 0
 edge_flow_constr_nic_out_0_nic_in_16: flow_nic_out_0_nic_in_16
   - 61035.15625 switch_nic_out_0_nic_in_16 <= 0
 edge_flow_constr_nic_out_0_nic_in_17: flow_nic_out_0_nic_in_17
   - 61035.15625 switch_nic_out_0_nic_in_17 <= 0
 edge_flow_constr_nic_out_0_nic_in_18: flow_nic_out_0_nic_in_18
   - 61035.15625 switch_nic_out_0_nic_in_18 <= 0
 edge_flow_constr_nic_out_0_nic_in_19: flow_nic_out_0_nic_in_19
   - 61035.15625 switch_nic_out_0_nic_in_19 <= 0
 edge_flow_constr_nic_out_0_nic_in_20: flow_nic_out_0_nic_in_20
   - 61035.15625 switch_nic_out_0_nic_in_20 <= 0
 edge_flow_constr_nic_out_0_nic_in_21: flow_nic_out_0_nic_in_21
   - 61035.15625 switch_nic_out_0_nic_in_21 <= 0
 edge_flow_constr_nic_out_0_nic_in_22: flow_nic_out_0_nic_in_22
   - 61035.15625 switch_nic_out_0_nic_in_22 <= 0
 edge_flow_constr_nic_out_0_nic_in_23: flow_nic_out_0_nic_in_23
   - 61035.15625 switch_nic_out_0_nic_in_23 <= 0
 edge_flow_constr_nic_out_0_nic_in_24: flow_nic_out_0_nic_in_24
   - 61035.15625 switch_nic_out_0_nic_in_24 <= 0
 edge_flow_constr_nic_out_0_nic_in_25: flow_nic_out_0_nic_in_25
   - 61035.15625 switch_nic_out_0_nic_in_25 <= 0
 edge_flow_constr_nic_out_0_nic_in_26: flow_nic_out_0_nic_in_26
   - 61035.15625 switch_nic_out_0_nic_in_26 <= 0
 edge_flow_constr_nic_out_0_nic_in_27: flow_nic_out_0_nic_in_27
   - 61035.15625 switch_nic_out_0_nic_in_27 <= 0
 edge_flow_constr_nic_out_0_nic_in_28: flow_nic_out_0_nic_in_28
   - 61035.15625 switch_nic_out_0_nic_in_28 <= 0
 edge_flow_constr_nic_out_0_nic_in_29: flow_nic_out_0_nic_in_29
   - 61035.15625 switch_nic_out_0_nic_in_29 <= 0
 edge_flow_constr_nic_out_0_nic_in_30: flow_nic_out_0_nic_in_30
   - 61035.15625 switch_nic_out_0_nic_in_30 <= 0
 edge_flow_constr_nic_out_0_nic_in_31: flow_nic_out_0_nic_in_31
   - 61035.15625 switch_nic_out_0_nic_in_31 <= 0
 edge_flow_constr_nic_in_0_0: flow_nic_in_0_0
   - 3.90625e+06 switch_nic_in_0_0 <= 0
 edge_flow_constr_0_nic_out_0: flow_0_nic_out_0
   - 3.90625e+06 switch_0_nic_out_0 <= 0
 edge_flow_constr_source_nic_in_1: flow_source_nic_in_1
   - 5e+08 switch_source_nic_in_1 <= 0
 edge_flow_constr_nic_out_1_sink: flow_nic_out_1_sink
   - 5e+08 switch_nic_out_1_sink <= 0
 edge_flow_constr_nic_out_1_nic_in_0: flow_nic_out_1_nic_in_0
   - 61035.15625 switch_nic_out_1_nic_in_0 <= 0
 edge_flow_constr_nic_out_1_nic_in_2: flow_nic_out_1_nic_in_2
   - 61035.15625 switch_nic_out_1_nic_in_2 <= 0
 edge_flow_constr_nic_out_1_nic_in_3: flow_nic_out_1_nic_in_3
   - 61035.15625 switch_nic_out_1_nic_in_3 <= 0
 edge_flow_constr_nic_out_1_nic_in_4: flow_nic_out_1_nic_in_4
   - 61035.15625 switch_nic_out_1_nic_in_4 <= 0
 edge_flow_constr_nic_out_1_nic_in_5: flow_nic_out_1_nic_in_5
   - 61035.15625 switch_nic_out_1_nic_in_5 <= 0
 edge_flow_constr_nic_out_1_nic_in_6: flow_nic_out_1_nic_in_6
   - 61035.15625 switch_nic_out_1_nic_in_6 <= 0
 edge_flow_constr_nic_out_1_nic_in_7: flow_nic_out_1_nic_in_7
   - 61035.15625 switch_nic_out_1_nic_in_7 <= 0
 edge_flow_constr_nic_out_1_nic_in_8: flow_nic_out_1_nic_in_8
   - 61035.15625 switch_nic_out_1_nic_in_8 <= 0
 edge_flow_constr_nic_out_1_nic_in_9: flow_nic_out_1_nic_in_9
   - 61035.15625 switch_nic_out_1_nic_in_9 <= 0
 edge_flow_constr_nic_out_1_nic_in_10: flow_nic_out_1_nic_in_10
   - 61035.15625 switch_nic_out_1_nic_in_10 <= 0
 edge_flow_constr_nic_out_1_nic_in_11: flow_nic_out_1_nic_in_11
   - 61035.15625 switch_nic_out_1_nic_in_11 <= 0
 edge_flow_constr_nic_out_1_nic_in_12: flow_nic_out_1_nic_in_12
   - 61035.15625 switch_nic_out_1_nic_in_12 <= 0
 edge_flow_constr_nic_out_1_nic_in_13: flow_nic_out_1_nic_in_13
   - 61035.15625 switch_nic_out_1_nic_in_13 <= 0
 edge_flow_constr_nic_out_1_nic_in_14: flow_nic_out_1_nic_in_14
   - 61035.15625 switch_nic_out_1_nic_in_14 <= 0
 edge_flow_constr_nic_out_1_nic_in_15: flow_nic_out_1_nic_in_15
   - 61035.15625 switch_nic_out_1_nic_in_15 <= 0
 edge_flow_constr_nic_out_1_nic_in_16: flow_nic_out_1_nic_in_16
   - 61035.15625 switch_nic_out_1_nic_in_16 <= 0
 edge_flow_constr_nic_out_1_nic_in_17: flow_nic_out_1_nic_in_17
   - 61035.15625 switch_nic_out_1_nic_in_17 <= 0
 edge_flow_constr_nic_out_1_nic_in_18: flow_nic_out_1_nic_in_18
   - 61035.15625 switch_nic_out_1_nic_in_18 <= 0
 edge_flow_constr_nic_out_1_nic_in_19: flow_nic_out_1_nic_in_19
   - 61035.15625 switch_nic_out_1_nic_in_19 <= 0
 edge_flow_constr_nic_out_1_nic_in_20: flow_nic_out_1_nic_in_20
   - 61035.15625 switch_nic_out_1_nic_in_20 <= 0
 edge_flow_constr_nic_out_1_nic_in_21: flow_nic_out_1_nic_in_21
   - 61035.15625 switch_nic_out_1_nic_in_21 <= 0
 edge_flow_constr_nic_out_1_nic_in_22: flow_nic_out_1_nic_in_22
   - 61035.15625 switch_nic_out_1_nic_in_22 <= 0
 edge_flow_constr_nic_out_1_nic_in_23: flow_nic_out_1_nic_in_23
   - 61035.15625 switch_nic_out_1_nic_in_23 <= 0
 edge_flow_constr_nic_out_1_nic_in_24: flow_nic_out_1_nic_in_24
   - 61035.15625 switch_nic_out_1_nic_in_24 <= 0
 edge_flow_constr_nic_out_1_nic_in_25: flow_nic_out_1_nic_in_25
   - 61035.15625 switch_nic_out_1_nic_in_25 <= 0
 edge_flow_constr_nic_out_1_nic_in_26: flow_nic_out_1_nic_in_26
   - 61035.15625 switch_nic_out_1_nic_in_26 <= 0
 edge_flow_constr_nic_out_1_nic_in_27: flow_nic_out_1_nic_in_27
   - 61035.15625 switch_nic_out_1_nic_in_27 <= 0
 edge_flow_constr_nic_out_1_nic_in_28: flow_nic_out_1_nic_in_28
   - 61035.15625 switch_nic_out_1_nic_in_28 <= 0
 edge_flow_constr_nic_out_1_nic_in_29: flow_nic_out_1_nic_in_29
   - 61035.15625 switch_nic_out_1_nic_in_29 <= 0
 edge_flow_constr_nic_out_1_nic_in_30: flow_nic_out_1_nic_in_30
   - 61035.15625 switch_nic_out_1_nic_in_30 <= 0
 edge_flow_constr_nic_out_1_nic_in_31: flow_nic_out_1_nic_in_31
   - 61035.15625 switch_nic_out_1_nic_in_31 <= 0
 edge_flow_constr_nic_in_1_1: flow_nic_in_1_1
   - 3.90625e+06 switch_nic_in_1_1 <= 0
 edge_flow_constr_1_nic_out_1: flow_1_nic_out_1
   - 3.90625e+06 switch_1_nic_out_1 <= 0
 edge_flow_constr_source_nic_in_2: flow_source_nic_in_2
   - 5e+08 switch_source_nic_in_2 <= 0
 edge_flow_constr_nic_out_2_sink: flow_nic_out_2_sink
   - 5e+08 switch_nic_out_2_sink <= 0
 edge_flow_constr_nic_out_2_nic_in_0: flow_nic_out_2_nic_in_0
   - 61035.15625 switch_nic_out_2_nic_in_0 <= 0
 edge_flow_constr_nic_out_2_nic_in_1: flow_nic_out_2_nic_in_1
   - 61035.15625 switch_nic_out_2_nic_in_1 <= 0
 edge_flow_constr_nic_out_2_nic_in_3: flow_nic_out_2_nic_in_3
   - 61035.15625 switch_nic_out_2_nic_in_3 <= 0
 edge_flow_constr_nic_out_2_nic_in_4: flow_nic_out_2_nic_in_4
   - 61035.15625 switch_nic_out_2_nic_in_4 <= 0
 edge_flow_constr_nic_out_2_nic_in_5: flow_nic_out_2_nic_in_5
   - 61035.15625 switch_nic_out_2_nic_in_5 <= 0
 edge_flow_constr_nic_out_2_nic_in_6: flow_nic_out_2_nic_in_6
   - 61035.15625 switch_nic_out_2_nic_in_6 <= 0
 edge_flow_constr_nic_out_2_nic_in_7: flow_nic_out_2_nic_in_7
   - 61035.15625 switch_nic_out_2_nic_in_7 <= 0
 edge_flow_constr_nic_out_2_nic_in_8: flow_nic_out_2_nic_in_8
   - 61035.15625 switch_nic_out_2_nic_in_8 <= 0
 edge_flow_constr_nic_out_2_nic_in_9: flow_nic_out_2_nic_in_9
   - 61035.15625 switch_nic_out_2_nic_in_9 <= 0
 edge_flow_constr_nic_out_2_nic_in_10: flow_nic_out_2_nic_in_10
   - 61035.15625 switch_nic_out_2_nic_in_10 <= 0
 edge_flow_constr_nic_out_2_nic_in_11: flow_nic_out_2_nic_in_11
   - 61035.15625 switch_nic_out_2_nic_in_11 <= 0
 edge_flow_constr_nic_out_2_nic_in_12: flow_nic_out_2_nic_in_12
   - 61035.15625 switch_nic_out_2_nic_in_12 <= 0
 edge_flow_constr_nic_out_2_nic_in_13: flow_nic_out_2_nic_in_13
   - 61035.15625 switch_nic_out_2_nic_in_13 <= 0
 edge_flow_constr_nic_out_2_nic_in_14: flow_nic_out_2_nic_in_14
   - 61035.15625 switch_nic_out_2_nic_in_14 <= 0
 edge_flow_constr_nic_out_2_nic_in_15: flow_nic_out_2_nic_in_15
   - 61035.15625 switch_nic_out_2_nic_in_15 <= 0
 edge_flow_constr_nic_out_2_nic_in_16: flow_nic_out_2_nic_in_16
   - 61035.15625 switch_nic_out_2_nic_in_16 <= 0
 edge_flow_constr_nic_out_2_nic_in_17: flow_nic_out_2_nic_in_17
   - 61035.15625 switch_nic_out_2_nic_in_17 <= 0
 edge_flow_constr_nic_out_2_nic_in_18: flow_nic_out_2_nic_in_18
   - 61035.15625 switch_nic_out_2_nic_in_18 <= 0
 edge_flow_constr_nic_out_2_nic_in_19: flow_nic_out_2_nic_in_19
   - 61035.15625 switch_nic_out_2_nic_in_19 <= 0
 edge_flow_constr_nic_out_2_nic_in_20: flow_nic_out_2_nic_in_20
   - 61035.15625 switch_nic_out_2_nic_in_20 <= 0
 edge_flow_constr_nic_out_2_nic_in_21: flow_nic_out_2_nic_in_21
   - 61035.15625 switch_nic_out_2_nic_in_21 <= 0
 edge_flow_constr_nic_out_2_nic_in_22: flow_nic_out_2_nic_in_22
   - 61035.15625 switch_nic_out_2_nic_in_22 <= 0
 edge_flow_constr_nic_out_2_nic_in_23: flow_nic_out_2_nic_in_23
   - 61035.15625 switch_nic_out_2_nic_in_23 <= 0
 edge_flow_constr_nic_out_2_nic_in_24: flow_nic_out_2_nic_in_24
   - 61035.15625 switch_nic_out_2_nic_in_24 <= 0
 edge_flow_constr_nic_out_2_nic_in_25: flow_nic_out_2_nic_in_25
   - 61035.15625 switch_nic_out_2_nic_in_25 <= 0
 edge_flow_constr_nic_out_2_nic_in_26: flow_nic_out_2_nic_in_26
   - 61035.15625 switch_nic_out_2_nic_in_26 <= 0
 edge_flow_constr_nic_out_2_nic_in_27: flow_nic_out_2_nic_in_27
   - 61035.15625 switch_nic_out_2_nic_in_27 <= 0
 edge_flow_constr_nic_out_2_nic_in_28: flow_nic_out_2_nic_in_28
   - 61035.15625 switch_nic_out_2_nic_in_28 <= 0
 edge_flow_constr_nic_out_2_nic_in_29: flow_nic_out_2_nic_in_29
   - 61035.15625 switch_nic_out_2_nic_in_29 <= 0
 edge_flow_constr_nic_out_2_nic_in_30: flow_nic_out_2_nic_in_30
   - 61035.15625 switch_nic_out_2_nic_in_30 <= 0
 edge_flow_constr_nic_out_2_nic_in_31: flow_nic_out_2_nic_in_31
   - 61035.15625 switch_nic_out_2_nic_in_31 <= 0
 edge_flow_constr_nic_in_2_2: flow_nic_in_2_2
   - 3.90625e+06 switch_nic_in_2_2 <= 0
 edge_flow_constr_2_nic_out_2: flow_2_nic_out_2
   - 3.90625e+06 switch_2_nic_out_2 <= 0
 edge_flow_constr_source_nic_in_3: flow_source_nic_in_3
   - 5e+08 switch_source_nic_in_3 <= 0
 edge_flow_constr_nic_out_3_sink: flow_nic_out_3_sink
   - 5e+08 switch_nic_out_3_sink <= 0
 edge_flow_constr_nic_out_3_nic_in_0: flow_nic_out_3_nic_in_0
   - 61035.15625 switch_nic_out_3_nic_in_0 <= 0
 edge_flow_constr_nic_out_3_nic_in_1: flow_nic_out_3_nic_in_1
   - 61035.15625 switch_nic_out_3_nic_in_1 <= 0
 edge_flow_constr_nic_out_3_nic_in_2: flow_nic_out_3_nic_in_2
   - 61035.15625 switch_nic_out_3_nic_in_2 <= 0
 edge_flow_constr_nic_out_3_nic_in_4: flow_nic_out_3_nic_in_4
   - 61035.15625 switch_nic_out_3_nic_in_4 <= 0
 edge_flow_constr_nic_out_3_nic_in_5: flow_nic_out_3_nic_in_5
   - 61035.15625 switch_nic_out_3_nic_in_5 <= 0
 edge_flow_constr_nic_out_3_nic_in_6: flow_nic_out_3_nic_in_6
   - 61035.15625 switch_nic_out_3_nic_in_6 <= 0
 edge_flow_constr_nic_out_3_nic_in_7: flow_nic_out_3_nic_in_7
   - 61035.15625 switch_nic_out_3_nic_in_7 <= 0
 edge_flow_constr_nic_out_3_nic_in_8: flow_nic_out_3_nic_in_8
   - 61035.15625 switch_nic_out_3_nic_in_8 <= 0
 edge_flow_constr_nic_out_3_nic_in_9: flow_nic_out_3_nic_in_9
   - 61035.15625 switch_nic_out_3_nic_in_9 <= 0
 edge_flow_constr_nic_out_3_nic_in_10: flow_nic_out_3_nic_in_10
   - 61035.15625 switch_nic_out_3_nic_in_10 <= 0
 edge_flow_constr_nic_out_3_nic_in_11: flow_nic_out_3_nic_in_11
   - 61035.15625 switch_nic_out_3_nic_in_11 <= 0
 edge_flow_constr_nic_out_3_nic_in_12: flow_nic_out_3_nic_in_12
   - 61035.15625 switch_nic_out_3_nic_in_12 <= 0
 edge_flow_constr_nic_out_3_nic_in_13: flow_nic_out_3_nic_in_13
   - 61035.15625 switch_nic_out_3_nic_in_13 <= 0
 edge_flow_constr_nic_out_3_nic_in_14: flow_nic_out_3_nic_in_14
   - 61035.15625 switch_nic_out_3_nic_in_14 <= 0
 edge_flow_constr_nic_out_3_nic_in_15: flow_nic_out_3_nic_in_15
   - 61035.15625 switch_nic_out_3_nic_in_15 <= 0
 edge_flow_constr_nic_out_3_nic_in_16: flow_nic_out_3_nic_in_16
   - 61035.15625 switch_nic_out_3_nic_in_16 <= 0
 edge_flow_constr_nic_out_3_nic_in_17: flow_nic_out_3_nic_in_17
   - 61035.15625 switch_nic_out_3_nic_in_17 <= 0
 edge_flow_constr_nic_out_3_nic_in_18: flow_nic_out_3_nic_in_18
   - 61035.15625 switch_nic_out_3_nic_in_18 <= 0
 edge_flow_constr_nic_out_3_nic_in_19: flow_nic_out_3_nic_in_19
   - 61035.15625 switch_nic_out_3_nic_in_19 <= 0
 edge_flow_constr_nic_out_3_nic_in_20: flow_nic_out_3_nic_in_20
   - 61035.15625 switch_nic_out_3_nic_in_20 <= 0
 edge_flow_constr_nic_out_3_nic_in_21: flow_nic_out_3_nic_in_21
   - 61035.15625 switch_nic_out_3_nic_in_21 <= 0
 edge_flow_constr_nic_out_3_nic_in_22: flow_nic_out_3_nic_in_22
   - 61035.15625 switch_nic_out_3_nic_in_22 <= 0
 edge_flow_constr_nic_out_3_nic_in_23: flow_nic_out_3_nic_in_23
   - 61035.15625 switch_nic_out_3_nic_in_23 <= 0
 edge_flow_constr_nic_out_3_nic_in_24: flow_nic_out_3_nic_in_24
   - 61035.15625 switch_nic_out_3_nic_in_24 <= 0
 edge_flow_constr_nic_out_3_nic_in_25: flow_nic_out_3_nic_in_25
   - 61035.15625 switch_nic_out_3_nic_in_25 <= 0
 edge_flow_constr_nic_out_3_nic_in_26: flow_nic_out_3_nic_in_26
   - 61035.15625 switch_nic_out_3_nic_in_26 <= 0
 edge_flow_constr_nic_out_3_nic_in_27: flow_nic_out_3_nic_in_27
   - 61035.15625 switch_nic_out_3_nic_in_27 <= 0
 edge_flow_constr_nic_out_3_nic_in_28: flow_nic_out_3_nic_in_28
   - 61035.15625 switch_nic_out_3_nic_in_28 <= 0
 edge_flow_constr_nic_out_3_nic_in_29: flow_nic_out_3_nic_in_29
   - 61035.15625 switch_nic_out_3_nic_in_29 <= 0
 edge_flow_constr_nic_out_3_nic_in_30: flow_nic_out_3_nic_in_30
   - 61035.15625 switch_nic_out_3_nic_in_30 <= 0
 edge_flow_constr_nic_out_3_nic_in_31: flow_nic_out_3_nic_in_31
   - 61035.15625 switch_nic_out_3_nic_in_31 <= 0
 edge_flow_constr_nic_in_3_3: flow_nic_in_3_3
   - 3.90625e+06 switch_nic_in_3_3 <= 0
 edge_flow_constr_3_nic_out_3: flow_3_nic_out_3
   - 3.90625e+06 switch_3_nic_out_3 <= 0
 edge_flow_constr_source_nic_in_4: flow_source_nic_in_4
   - 5e+08 switch_source_nic_in_4 <= 0
 edge_flow_constr_nic_out_4_sink: flow_nic_out_4_sink
   - 5e+08 switch_nic_out_4_sink <= 0
 edge_flow_constr_nic_out_4_nic_in_0: flow_nic_out_4_nic_in_0
   - 61035.15625 switch_nic_out_4_nic_in_0 <= 0
 edge_flow_constr_nic_out_4_nic_in_1: flow_nic_out_4_nic_in_1
   - 61035.15625 switch_nic_out_4_nic_in_1 <= 0
 edge_flow_constr_nic_out_4_nic_in_2: flow_nic_out_4_nic_in_2
   - 61035.15625 switch_nic_out_4_nic_in_2 <= 0
 edge_flow_constr_nic_out_4_nic_in_3: flow_nic_out_4_nic_in_3
   - 61035.15625 switch_nic_out_4_nic_in_3 <= 0
 edge_flow_constr_nic_out_4_nic_in_5: flow_nic_out_4_nic_in_5
   - 61035.15625 switch_nic_out_4_nic_in_5 <= 0
 edge_flow_constr_nic_out_4_nic_in_6: flow_nic_out_4_nic_in_6
   - 61035.15625 switch_nic_out_4_nic_in_6 <= 0
 edge_flow_constr_nic_out_4_nic_in_7: flow_nic_out_4_nic_in_7
   - 61035.15625 switch_nic_out_4_nic_in_7 <= 0
 edge_flow_constr_nic_out_4_nic_in_8: flow_nic_out_4_nic_in_8
   - 61035.15625 switch_nic_out_4_nic_in_8 <= 0
 edge_flow_constr_nic_out_4_nic_in_9: flow_nic_out_4_nic_in_9
   - 61035.15625 switch_nic_out_4_nic_in_9 <= 0
 edge_flow_constr_nic_out_4_nic_in_10: flow_nic_out_4_nic_in_10
   - 61035.15625 switch_nic_out_4_nic_in_10 <= 0
 edge_flow_constr_nic_out_4_nic_in_11: flow_nic_out_4_nic_in_11
   - 61035.15625 switch_nic_out_4_nic_in_11 <= 0
 edge_flow_constr_nic_out_4_nic_in_12: flow_nic_out_4_nic_in_12
   - 61035.15625 switch_nic_out_4_nic_in_12 <= 0
 edge_flow_constr_nic_out_4_nic_in_13: flow_nic_out_4_nic_in_13
   - 61035.15625 switch_nic_out_4_nic_in_13 <= 0
 edge_flow_constr_nic_out_4_nic_in_14: flow_nic_out_4_nic_in_14
   - 61035.15625 switch_nic_out_4_nic_in_14 <= 0
 edge_flow_constr_nic_out_4_nic_in_15: flow_nic_out_4_nic_in_15
   - 61035.15625 switch_nic_out_4_nic_in_15 <= 0
 edge_flow_constr_nic_out_4_nic_in_16: flow_nic_out_4_nic_in_16
   - 61035.15625 switch_nic_out_4_nic_in_16 <= 0
 edge_flow_constr_nic_out_4_nic_in_17: flow_nic_out_4_nic_in_17
   - 61035.15625 switch_nic_out_4_nic_in_17 <= 0
 edge_flow_constr_nic_out_4_nic_in_18: flow_nic_out_4_nic_in_18
   - 61035.15625 switch_nic_out_4_nic_in_18 <= 0
 edge_flow_constr_nic_out_4_nic_in_19: flow_nic_out_4_nic_in_19
   - 61035.15625 switch_nic_out_4_nic_in_19 <= 0
 edge_flow_constr_nic_out_4_nic_in_20: flow_nic_out_4_nic_in_20
   - 61035.15625 switch_nic_out_4_nic_in_20 <= 0
 edge_flow_constr_nic_out_4_nic_in_21: flow_nic_out_4_nic_in_21
   - 61035.15625 switch_nic_out_4_nic_in_21 <= 0
 edge_flow_constr_nic_out_4_nic_in_22: flow_nic_out_4_nic_in_22
   - 61035.15625 switch_nic_out_4_nic_in_22 <= 0
 edge_flow_constr_nic_out_4_nic_in_23: flow_nic_out_4_nic_in_23
   - 61035.15625 switch_nic_out_4_nic_in_23 <= 0
 edge_flow_constr_nic_out_4_nic_in_24: flow_nic_out_4_nic_in_24
   - 61035.15625 switch_nic_out_4_nic_in_24 <= 0
 edge_flow_constr_nic_out_4_nic_in_25: flow_nic_out_4_nic_in_25
   - 61035.15625 switch_nic_out_4_nic_in_25 <= 0
 edge_flow_constr_nic_out_4_nic_in_26: flow_nic_out_4_nic_in_26
   - 61035.15625 switch_nic_out_4_nic_in_26 <= 0
 edge_flow_constr_nic_out_4_nic_in_27: flow_nic_out_4_nic_in_27
   - 61035.15625 switch_nic_out_4_nic_in_27 <= 0
 edge_flow_constr_nic_out_4_nic_in_28: flow_nic_out_4_nic_in_28
   - 61035.15625 switch_nic_out_4_nic_in_28 <= 0
 edge_flow_constr_nic_out_4_nic_in_29: flow_nic_out_4_nic_in_29
   - 61035.15625 switch_nic_out_4_nic_in_29 <= 0
 edge_flow_constr_nic_out_4_nic_in_30: flow_nic_out_4_nic_in_30
   - 61035.15625 switch_nic_out_4_nic_in_30 <= 0
 edge_flow_constr_nic_out_4_nic_in_31: flow_nic_out_4_nic_in_31
   - 61035.15625 switch_nic_out_4_nic_in_31 <= 0
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
 edge_flow_constr_nic_out_5_nic_in_1: flow_nic_out_5_nic_in_1
   - 61035.15625 switch_nic_out_5_nic_in_1 <= 0
 edge_flow_constr_nic_out_5_nic_in_2: flow_nic_out_5_nic_in_2
   - 61035.15625 switch_nic_out_5_nic_in_2 <= 0
 edge_flow_constr_nic_out_5_nic_in_3: flow_nic_out_5_nic_in_3
   - 61035.15625 switch_nic_out_5_nic_in_3 <= 0
 edge_flow_constr_nic_out_5_nic_in_4: flow_nic_out_5_nic_in_4
   - 61035.15625 switch_nic_out_5_nic_in_4 <= 0
 edge_flow_constr_nic_out_5_nic_in_6: flow_nic_out_5_nic_in_6
   - 61035.15625 switch_nic_out_5_nic_in_6 <= 0
 edge_flow_constr_nic_out_5_nic_in_7: flow_nic_out_5_nic_in_7
   - 61035.15625 switch_nic_out_5_nic_in_7 <= 0
 edge_flow_constr_nic_out_5_nic_in_8: flow_nic_out_5_nic_in_8
   - 61035.15625 switch_nic_out_5_nic_in_8 <= 0
 edge_flow_constr_nic_out_5_nic_in_9: flow_nic_out_5_nic_in_9
   - 61035.15625 switch_nic_out_5_nic_in_9 <= 0
 edge_flow_constr_nic_out_5_nic_in_10: flow_nic_out_5_nic_in_10
   - 61035.15625 switch_nic_out_5_nic_in_10 <= 0
 edge_flow_constr_nic_out_5_nic_in_11: flow_nic_out_5_nic_in_11
   - 61035.15625 switch_nic_out_5_nic_in_11 <= 0
 edge_flow_constr_nic_out_5_nic_in_12: flow_nic_out_5_nic_in_12
   - 61035.15625 switch_nic_out_5_nic_in_12 <= 0
 edge_flow_constr_nic_out_5_nic_in_13: flow_nic_out_5_nic_in_13
   - 61035.15625 switch_nic_out_5_nic_in_13 <= 0
 edge_flow_constr_nic_out_5_nic_in_14: flow_nic_out_5_nic_in_14
   - 61035.15625 switch_nic_out_5_nic_in_14 <= 0
 edge_flow_constr_nic_out_5_nic_in_15: flow_nic_out_5_nic_in_15
   - 61035.15625 switch_nic_out_5_nic_in_15 <= 0
 edge_flow_constr_nic_out_5_nic_in_16: flow_nic_out_5_nic_in_16
   - 61035.15625 switch_nic_out_5_nic_in_16 <= 0
 edge_flow_constr_nic_out_5_nic_in_17: flow_nic_out_5_nic_in_17
   - 61035.15625 switch_nic_out_5_nic_in_17 <= 0
 edge_flow_constr_nic_out_5_nic_in_18: flow_nic_out_5_nic_in_18
   - 61035.15625 switch_nic_out_5_nic_in_18 <= 0
 edge_flow_constr_nic_out_5_nic_in_19: flow_nic_out_5_nic_in_19
   - 61035.15625 switch_nic_out_5_nic_in_19 <= 0
 edge_flow_constr_nic_out_5_nic_in_20: flow_nic_out_5_nic_in_20
   - 61035.15625 switch_nic_out_5_nic_in_20 <= 0
 edge_flow_constr_nic_out_5_nic_in_21: flow_nic_out_5_nic_in_21
   - 61035.15625 switch_nic_out_5_nic_in_21 <= 0
 edge_flow_constr_nic_out_5_nic_in_22: flow_nic_out_5_nic_in_22
   - 61035.15625 switch_nic_out_5_nic_in_22 <= 0
 edge_flow_constr_nic_out_5_nic_in_23: flow_nic_out_5_nic_in_23
   - 61035.15625 switch_nic_out_5_nic_in_23 <= 0
 edge_flow_constr_nic_out_5_nic_in_24: flow_nic_out_5_nic_in_24
   - 61035.15625 switch_nic_out_5_nic_in_24 <= 0
 edge_flow_constr_nic_out_5_nic_in_25: flow_nic_out_5_nic_in_25
   - 61035.15625 switch_nic_out_5_nic_in_25 <= 0
 edge_flow_constr_nic_out_5_nic_in_26: flow_nic_out_5_nic_in_26
   - 61035.15625 switch_nic_out_5_nic_in_26 <= 0
 edge_flow_constr_nic_out_5_nic_in_27: flow_nic_out_5_nic_in_27
   - 61035.15625 switch_nic_out_5_nic_in_27 <= 0
 edge_flow_constr_nic_out_5_nic_in_28: flow_nic_out_5_nic_in_28
   - 61035.15625 switch_nic_out_5_nic_in_28 <= 0
 edge_flow_constr_nic_out_5_nic_in_29: flow_nic_out_5_nic_in_29
   - 61035.15625 switch_nic_out_5_nic_in_29 <= 0
 edge_flow_constr_nic_out_5_nic_in_30: flow_nic_out_5_nic_in_30
   - 61035.15625 switch_nic_out_5_nic_in_30 <= 0
 edge_flow_constr_nic_out_5_nic_in_31: flow_nic_out_5_nic_in_31
   - 61035.15625 switch_nic_out_5_nic_in_31 <= 0
 edge_flow_constr_nic_in_5_5: flow_nic_in_5_5
   - 3.90625e+06 switch_nic_in_5_5 <= 0
 edge_flow_constr_5_nic_out_5: flow_5_nic_out_5
   - 3.90625e+06 switch_5_nic_out_5 <= 0
 edge_flow_constr_source_nic_in_6: flow_source_nic_in_6
   - 5e+08 switch_source_nic_in_6 <= 0
 edge_flow_constr_nic_out_6_sink: flow_nic_out_6_sink
   - 5e+08 switch_nic_out_6_sink <= 0
 edge_flow_constr_nic_out_6_nic_in_0: flow_nic_out_6_nic_in_0
   - 61035.15625 switch_nic_out_6_nic_in_0 <= 0
 edge_flow_constr_nic_out_6_nic_in_1: flow_nic_out_6_nic_in_1
   - 61035.15625 switch_nic_out_6_nic_in_1 <= 0
 edge_flow_constr_nic_out_6_nic_in_2: flow_nic_out_6_nic_in_2
   - 61035.15625 switch_nic_out_6_nic_in_2 <= 0
 edge_flow_constr_nic_out_6_nic_in_3: flow_nic_out_6_nic_in_3
   - 61035.15625 switch_nic_out_6_nic_in_3 <= 0
 edge_flow_constr_nic_out_6_nic_in_4: flow_nic_out_6_nic_in_4
   - 61035.15625 switch_nic_out_6_nic_in_4 <= 0
 edge_flow_constr_nic_out_6_nic_in_5: flow_nic_out_6_nic_in_5
   - 61035.15625 switch_nic_out_6_nic_in_5 <= 0
 edge_flow_constr_nic_out_6_nic_in_7: flow_nic_out_6_nic_in_7
   - 61035.15625 switch_nic_out_6_nic_in_7 <= 0
 edge_flow_constr_nic_out_6_nic_in_8: flow_nic_out_6_nic_in_8
   - 61035.15625 switch_nic_out_6_nic_in_8 <= 0
 edge_flow_constr_nic_out_6_nic_in_9: flow_nic_out_6_nic_in_9
   - 61035.15625 switch_nic_out_6_nic_in_9 <= 0
 edge_flow_constr_nic_out_6_nic_in_10: flow_nic_out_6_nic_in_10
   - 61035.15625 switch_nic_out_6_nic_in_10 <= 0
 edge_flow_constr_nic_out_6_nic_in_11: flow_nic_out_6_nic_in_11
   - 61035.15625 switch_nic_out_6_nic_in_11 <= 0
 edge_flow_constr_nic_out_6_nic_in_12: flow_nic_out_6_nic_in_12
   - 61035.15625 switch_nic_out_6_nic_in_12 <= 0
 edge_flow_constr_nic_out_6_nic_in_13: flow_nic_out_6_nic_in_13
   - 61035.15625 switch_nic_out_6_nic_in_13 <= 0
 edge_flow_constr_nic_out_6_nic_in_14: flow_nic_out_6_nic_in_14
   - 61035.15625 switch_nic_out_6_nic_in_14 <= 0
 edge_flow_constr_nic_out_6_nic_in_15: flow_nic_out_6_nic_in_15
   - 61035.15625 switch_nic_out_6_nic_in_15 <= 0
 edge_flow_constr_nic_out_6_nic_in_16: flow_nic_out_6_nic_in_16
   - 61035.15625 switch_nic_out_6_nic_in_16 <= 0
 edge_flow_constr_nic_out_6_nic_in_17: flow_nic_out_6_nic_in_17
   - 61035.15625 switch_nic_out_6_nic_in_17 <= 0
 edge_flow_constr_nic_out_6_nic_in_18: flow_nic_out_6_nic_in_18
   - 61035.15625 switch_nic_out_6_nic_in_18 <= 0
 edge_flow_constr_nic_out_6_nic_in_19: flow_nic_out_6_nic_in_19
   - 61035.15625 switch_nic_out_6_nic_in_19 <= 0
 edge_flow_constr_nic_out_6_nic_in_20: flow_nic_out_6_nic_in_20
   - 61035.15625 switch_nic_out_6_nic_in_20 <= 0
 edge_flow_constr_nic_out_6_nic_in_21: flow_nic_out_6_nic_in_21
   - 61035.15625 switch_nic_out_6_nic_in_21 <= 0
 edge_flow_constr_nic_out_6_nic_in_22: flow_nic_out_6_nic_in_22
   - 61035.15625 switch_nic_out_6_nic_in_22 <= 0
 edge_flow_constr_nic_out_6_nic_in_23: flow_nic_out_6_nic_in_23
   - 61035.15625 switch_nic_out_6_nic_in_23 <= 0
 edge_flow_constr_nic_out_6_nic_in_24: flow_nic_out_6_nic_in_24
   - 61035.15625 switch_nic_out_6_nic_in_24 <= 0
 edge_flow_constr_nic_out_6_nic_in_25: flow_nic_out_6_nic_in_25
   - 61035.15625 switch_nic_out_6_nic_in_25 <= 0
 edge_flow_constr_nic_out_6_nic_in_26: flow_nic_out_6_nic_in_26
   - 61035.15625 switch_nic_out_6_nic_in_26 <= 0
 edge_flow_constr_nic_out_6_nic_in_27: flow_nic_out_6_nic_in_27
   - 61035.15625 switch_nic_out_6_nic_in_27 <= 0
 edge_flow_constr_nic_out_6_nic_in_28: flow_nic_out_6_nic_in_28
   - 61035.15625 switch_nic_out_6_nic_in_28 <= 0
 edge_flow_constr_nic_out_6_nic_in_29: flow_nic_out_6_nic_in_29
   - 61035.15625 switch_nic_out_6_nic_in_29 <= 0
 edge_flow_constr_nic_out_6_nic_in_30: flow_nic_out_6_nic_in_30
   - 61035.15625 switch_nic_out_6_nic_in_30 <= 0
 edge_flow_constr_nic_out_6_nic_in_31: flow_nic_out_6_nic_in_31
   - 61035.15625 switch_nic_out_6_nic_in_31 <= 0
 edge_flow_constr_nic_in_6_6: flow_nic_in_6_6
   - 3.90625e+06 switch_nic_in_6_6 <= 0
 edge_flow_constr_6_nic_out_6: flow_6_nic_out_6
   - 3.90625e+06 switch_6_nic_out_6 <= 0
 edge_flow_constr_source_nic_in_7: flow_source_nic_in_7
   - 5e+08 switch_source_nic_in_7 <= 0
 edge_flow_constr_nic_out_7_sink: flow_nic_out_7_sink
   - 5e+08 switch_nic_out_7_sink <= 0
 edge_flow_constr_nic_out_7_nic_in_0: flow_nic_out_7_nic_in_0
   - 61035.15625 switch_nic_out_7_nic_in_0 <= 0
 edge_flow_constr_nic_out_7_nic_in_1: flow_nic_out_7_nic_in_1
   - 61035.15625 switch_nic_out_7_nic_in_1 <= 0
 edge_flow_constr_nic_out_7_nic_in_2: flow_nic_out_7_nic_in_2
   - 61035.15625 switch_nic_out_7_nic_in_2 <= 0
 edge_flow_constr_nic_out_7_nic_in_3: flow_nic_out_7_nic_in_3
   - 61035.15625 switch_nic_out_7_nic_in_3 <= 0
 edge_flow_constr_nic_out_7_nic_in_4: flow_nic_out_7_nic_in_4
   - 61035.15625 switch_nic_out_7_nic_in_4 <= 0
 edge_flow_constr_nic_out_7_nic_in_5: flow_nic_out_7_nic_in_5
   - 61035.15625 switch_nic_out_7_nic_in_5 <= 0
 edge_flow_constr_nic_out_7_nic_in_6: flow_nic_out_7_nic_in_6
   - 61035.15625 switch_nic_out_7_nic_in_6 <= 0
 edge_flow_constr_nic_out_7_nic_in_8: flow_nic_out_7_nic_in_8
   - 61035.15625 switch_nic_out_7_nic_in_8 <= 0
 edge_flow_constr_nic_out_7_nic_in_9: flow_nic_out_7_nic_in_9
   - 61035.15625 switch_nic_out_7_nic_in_9 <= 0
 edge_flow_constr_nic_out_7_nic_in_10: flow_nic_out_7_nic_in_10
   - 61035.15625 switch_nic_out_7_nic_in_10 <= 0
 edge_flow_constr_nic_out_7_nic_in_11: flow_nic_out_7_nic_in_11
   - 61035.15625 switch_nic_out_7_nic_in_11 <= 0
 edge_flow_constr_nic_out_7_nic_in_12: flow_nic_out_7_nic_in_12
   - 61035.15625 switch_nic_out_7_nic_in_12 <= 0
 edge_flow_constr_nic_out_7_nic_in_13: flow_nic_out_7_nic_in_13
   - 61035.15625 switch_nic_out_7_nic_in_13 <= 0
 edge_flow_constr_nic_out_7_nic_in_14: flow_nic_out_7_nic_in_14
   - 61035.15625 switch_nic_out_7_nic_in_14 <= 0
 edge_flow_constr_nic_out_7_nic_in_15: flow_nic_out_7_nic_in_15
   - 61035.15625 switch_nic_out_7_nic_in_15 <= 0
 edge_flow_constr_nic_out_7_nic_in_16: flow_nic_out_7_nic_in_16
   - 61035.15625 switch_nic_out_7_nic_in_16 <= 0
 edge_flow_constr_nic_out_7_nic_in_17: flow_nic_out_7_nic_in_17
   - 61035.15625 switch_nic_out_7_nic_in_17 <= 0
 edge_flow_constr_nic_out_7_nic_in_18: flow_nic_out_7_nic_in_18
   - 61035.15625 switch_nic_out_7_nic_in_18 <= 0
 edge_flow_constr_nic_out_7_nic_in_19: flow_nic_out_7_nic_in_19
   - 61035.15625 switch_nic_out_7_nic_in_19 <= 0
 edge_flow_constr_nic_out_7_nic_in_20: flow_nic_out_7_nic_in_20
   - 61035.15625 switch_nic_out_7_nic_in_20 <= 0
 edge_flow_constr_nic_out_7_nic_in_21: flow_nic_out_7_nic_in_21
   - 61035.15625 switch_nic_out_7_nic_in_21 <= 0
 edge_flow_constr_nic_out_7_nic_in_22: flow_nic_out_7_nic_in_22
   - 61035.15625 switch_nic_out_7_nic_in_22 <= 0
 edge_flow_constr_nic_out_7_nic_in_23: flow_nic_out_7_nic_in_23
   - 61035.15625 switch_nic_out_7_nic_in_23 <= 0
 edge_flow_constr_nic_out_7_nic_in_24: flow_nic_out_7_nic_in_24
   - 61035.15625 switch_nic_out_7_nic_in_24 <= 0
 edge_flow_constr_nic_out_7_nic_in_25: flow_nic_out_7_nic_in_25
   - 61035.15625 switch_nic_out_7_nic_in_25 <= 0
 edge_flow_constr_nic_out_7_nic_in_26: flow_nic_out_7_nic_in_26
   - 61035.15625 switch_nic_out_7_nic_in_26 <= 0
 edge_flow_constr_nic_out_7_nic_in_27: flow_nic_out_7_nic_in_27
   - 61035.15625 switch_nic_out_7_nic_in_27 <= 0
 edge_flow_constr_nic_out_7_nic_in_28: flow_nic_out_7_nic_in_28
   - 61035.15625 switch_nic_out_7_nic_in_28 <= 0
 edge_flow_constr_nic_out_7_nic_in_29: flow_nic_out_7_nic_in_29
   - 61035.15625 switch_nic_out_7_nic_in_29 <= 0
 edge_flow_constr_nic_out_7_nic_in_30: flow_nic_out_7_nic_in_30
   - 61035.15625 switch_nic_out_7_nic_in_30 <= 0
 edge_flow_constr_nic_out_7_nic_in_31: flow_nic_out_7_nic_in_31
   - 61035.15625 switch_nic_out_7_nic_in_31 <= 0
 edge_flow_constr_nic_in_7_7: flow_nic_in_7_7
   - 3.90625e+06 switch_nic_in_7_7 <= 0
 edge_flow_constr_7_nic_out_7: flow_7_nic_out_7
   - 3.90625e+06 switch_7_nic_out_7 <= 0
 edge_flow_constr_source_nic_in_8: flow_source_nic_in_8
   - 5e+08 switch_source_nic_in_8 <= 0
 edge_flow_constr_nic_out_8_sink: flow_nic_out_8_sink
   - 5e+08 switch_nic_out_8_sink <= 0
 edge_flow_constr_nic_out_8_nic_in_0: flow_nic_out_8_nic_in_0
   - 61035.15625 switch_nic_out_8_nic_in_0 <= 0
 edge_flow_constr_nic_out_8_nic_in_1: flow_nic_out_8_nic_in_1
   - 61035.15625 switch_nic_out_8_nic_in_1 <= 0
 edge_flow_constr_nic_out_8_nic_in_2: flow_nic_out_8_nic_in_2
   - 61035.15625 switch_nic_out_8_nic_in_2 <= 0
 edge_flow_constr_nic_out_8_nic_in_3: flow_nic_out_8_nic_in_3
   - 61035.15625 switch_nic_out_8_nic_in_3 <= 0
 edge_flow_constr_nic_out_8_nic_in_4: flow_nic_out_8_nic_in_4
   - 61035.15625 switch_nic_out_8_nic_in_4 <= 0
 edge_flow_constr_nic_out_8_nic_in_5: flow_nic_out_8_nic_in_5
   - 61035.15625 switch_nic_out_8_nic_in_5 <= 0
 edge_flow_constr_nic_out_8_nic_in_6: flow_nic_out_8_nic_in_6
   - 61035.15625 switch_nic_out_8_nic_in_6 <= 0
 edge_flow_constr_nic_out_8_nic_in_7: flow_nic_out_8_nic_in_7
   - 61035.15625 switch_nic_out_8_nic_in_7 <= 0
 edge_flow_constr_nic_out_8_nic_in_9: flow_nic_out_8_nic_in_9
   - 61035.15625 switch_nic_out_8_nic_in_9 <= 0
 edge_flow_constr_nic_out_8_nic_in_10: flow_nic_out_8_nic_in_10
   - 61035.15625 switch_nic_out_8_nic_in_10 <= 0
 edge_flow_constr_nic_out_8_nic_in_11: flow_nic_out_8_nic_in_11
   - 61035.15625 switch_nic_out_8_nic_in_11 <= 0
 edge_flow_constr_nic_out_8_nic_in_12: flow_nic_out_8_nic_in_12
   - 61035.15625 switch_nic_out_8_nic_in_12 <= 0
 edge_flow_constr_nic_out_8_nic_in_13: flow_nic_out_8_nic_in_13
   - 61035.15625 switch_nic_out_8_nic_in_13 <= 0
 edge_flow_constr_nic_out_8_nic_in_14: flow_nic_out_8_nic_in_14
   - 61035.15625 switch_nic_out_8_nic_in_14 <= 0
 edge_flow_constr_nic_out_8_nic_in_15: flow_nic_out_8_nic_in_15
   - 61035.15625 switch_nic_out_8_nic_in_15 <= 0
 edge_flow_constr_nic_out_8_nic_in_16: flow_nic_out_8_nic_in_16
   - 61035.15625 switch_nic_out_8_nic_in_16 <= 0
 edge_flow_constr_nic_out_8_nic_in_17: flow_nic_out_8_nic_in_17
   - 61035.15625 switch_nic_out_8_nic_in_17 <= 0
 edge_flow_constr_nic_out_8_nic_in_18: flow_nic_out_8_nic_in_18
   - 61035.15625 switch_nic_out_8_nic_in_18 <= 0
 edge_flow_constr_nic_out_8_nic_in_19: flow_nic_out_8_nic_in_19
   - 61035.15625 switch_nic_out_8_nic_in_19 <= 0
 edge_flow_constr_nic_out_8_nic_in_20: flow_nic_out_8_nic_in_20
   - 61035.15625 switch_nic_out_8_nic_in_20 <= 0
 edge_flow_constr_nic_out_8_nic_in_21: flow_nic_out_8_nic_in_21
   - 61035.15625 switch_nic_out_8_nic_in_21 <= 0
 edge_flow_constr_nic_out_8_nic_in_22: flow_nic_out_8_nic_in_22
   - 61035.15625 switch_nic_out_8_nic_in_22 <= 0
 edge_flow_constr_nic_out_8_nic_in_23: flow_nic_out_8_nic_in_23
   - 61035.15625 switch_nic_out_8_nic_in_23 <= 0
 edge_flow_constr_nic_out_8_nic_in_24: flow_nic_out_8_nic_in_24
   - 61035.15625 switch_nic_out_8_nic_in_24 <= 0
 edge_flow_constr_nic_out_8_nic_in_25: flow_nic_out_8_nic_in_25
   - 61035.15625 switch_nic_out_8_nic_in_25 <= 0
 edge_flow_constr_nic_out_8_nic_in_26: flow_nic_out_8_nic_in_26
   - 61035.15625 switch_nic_out_8_nic_in_26 <= 0
 edge_flow_constr_nic_out_8_nic_in_27: flow_nic_out_8_nic_in_27
   - 61035.15625 switch_nic_out_8_nic_in_27 <= 0
 edge_flow_constr_nic_out_8_nic_in_28: flow_nic_out_8_nic_in_28
   - 61035.15625 switch_nic_out_8_nic_in_28 <= 0
 edge_flow_constr_nic_out_8_nic_in_29: flow_nic_out_8_nic_in_29
   - 61035.15625 switch_nic_out_8_nic_in_29 <= 0
 edge_flow_constr_nic_out_8_nic_in_30: flow_nic_out_8_nic_in_30
   - 61035.15625 switch_nic_out_8_nic_in_30 <= 0
 edge_flow_constr_nic_out_8_nic_in_31: flow_nic_out_8_nic_in_31
   - 61035.15625 switch_nic_out_8_nic_in_31 <= 0
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
 edge_flow_constr_nic_out_9_nic_in_1: flow_nic_out_9_nic_in_1
   - 61035.15625 switch_nic_out_9_nic_in_1 <= 0
 edge_flow_constr_nic_out_9_nic_in_2: flow_nic_out_9_nic_in_2
   - 61035.15625 switch_nic_out_9_nic_in_2 <= 0
 edge_flow_constr_nic_out_9_nic_in_3: flow_nic_out_9_nic_in_3
   - 61035.15625 switch_nic_out_9_nic_in_3 <= 0
 edge_flow_constr_nic_out_9_nic_in_4: flow_nic_out_9_nic_in_4
   - 61035.15625 switch_nic_out_9_nic_in_4 <= 0
 edge_flow_constr_nic_out_9_nic_in_5: flow_nic_out_9_nic_in_5
   - 61035.15625 switch_nic_out_9_nic_in_5 <= 0
 edge_flow_constr_nic_out_9_nic_in_6: flow_nic_out_9_nic_in_6
   - 61035.15625 switch_nic_out_9_nic_in_6 <= 0
 edge_flow_constr_nic_out_9_nic_in_7: flow_nic_out_9_nic_in_7
   - 61035.15625 switch_nic_out_9_nic_in_7 <= 0
 edge_flow_constr_nic_out_9_nic_in_8: flow_nic_out_9_nic_in_8
   - 61035.15625 switch_nic_out_9_nic_in_8 <= 0
 edge_flow_constr_nic_out_9_nic_in_10: flow_nic_out_9_nic_in_10
   - 61035.15625 switch_nic_out_9_nic_in_10 <= 0
 edge_flow_constr_nic_out_9_nic_in_11: flow_nic_out_9_nic_in_11
   - 61035.15625 switch_nic_out_9_nic_in_11 <= 0
 edge_flow_constr_nic_out_9_nic_in_12: flow_nic_out_9_nic_in_12
   - 61035.15625 switch_nic_out_9_nic_in_12 <= 0
 edge_flow_constr_nic_out_9_nic_in_13: flow_nic_out_9_nic_in_13
   - 61035.15625 switch_nic_out_9_nic_in_13 <= 0
 edge_flow_constr_nic_out_9_nic_in_14: flow_nic_out_9_nic_in_14
   - 61035.15625 switch_nic_out_9_nic_in_14 <= 0
 edge_flow_constr_nic_out_9_nic_in_15: flow_nic_out_9_nic_in_15
   - 61035.15625 switch_nic_out_9_nic_in_15 <= 0
 edge_flow_constr_nic_out_9_nic_in_16: flow_nic_out_9_nic_in_16
   - 61035.15625 switch_nic_out_9_nic_in_16 <= 0
 edge_flow_constr_nic_out_9_nic_in_17: flow_nic_out_9_nic_in_17
   - 61035.15625 switch_nic_out_9_nic_in_17 <= 0
 edge_flow_constr_nic_out_9_nic_in_18: flow_nic_out_9_nic_in_18
   - 61035.15625 switch_nic_out_9_nic_in_18 <= 0
 edge_flow_constr_nic_out_9_nic_in_19: flow_nic_out_9_nic_in_19
   - 61035.15625 switch_nic_out_9_nic_in_19 <= 0
 edge_flow_constr_nic_out_9_nic_in_20: flow_nic_out_9_nic_in_20
   - 61035.15625 switch_nic_out_9_nic_in_20 <= 0
 edge_flow_constr_nic_out_9_nic_in_21: flow_nic_out_9_nic_in_21
   - 61035.15625 switch_nic_out_9_nic_in_21 <= 0
 edge_flow_constr_nic_out_9_nic_in_22: flow_nic_out_9_nic_in_22
   - 61035.15625 switch_nic_out_9_nic_in_22 <= 0
 edge_flow_constr_nic_out_9_nic_in_23: flow_nic_out_9_nic_in_23
   - 61035.15625 switch_nic_out_9_nic_in_23 <= 0
 edge_flow_constr_nic_out_9_nic_in_24: flow_nic_out_9_nic_in_24
   - 61035.15625 switch_nic_out_9_nic_in_24 <= 0
 edge_flow_constr_nic_out_9_nic_in_25: flow_nic_out_9_nic_in_25
   - 61035.15625 switch_nic_out_9_nic_in_25 <= 0
 edge_flow_constr_nic_out_9_nic_in_26: flow_nic_out_9_nic_in_26
   - 61035.15625 switch_nic_out_9_nic_in_26 <= 0
 edge_flow_constr_nic_out_9_nic_in_27: flow_nic_out_9_nic_in_27
   - 61035.15625 switch_nic_out_9_nic_in_27 <= 0
 edge_flow_constr_nic_out_9_nic_in_28: flow_nic_out_9_nic_in_28
   - 61035.15625 switch_nic_out_9_nic_in_28 <= 0
 edge_flow_constr_nic_out_9_nic_in_29: flow_nic_out_9_nic_in_29
   - 61035.15625 switch_nic_out_9_nic_in_29 <= 0
 edge_flow_constr_nic_out_9_nic_in_30: flow_nic_out_9_nic_in_30
   - 61035.15625 switch_nic_out_9_nic_in_30 <= 0
 edge_flow_constr_nic_out_9_nic_in_31: flow_nic_out_9_nic_in_31
   - 61035.15625 switch_nic_out_9_nic_in_31 <= 0
 edge_flow_constr_nic_in_9_9: flow_nic_in_9_9
   - 3.90625e+06 switch_nic_in_9_9 <= 0
 edge_flow_constr_9_nic_out_9: flow_9_nic_out_9
   - 3.90625e+06 switch_9_nic_out_9 <= 0
 edge_flow_constr_source_nic_in_10: flow_source_nic_in_10
   - 5e+08 switch_source_nic_in_10 <= 0
 edge_flow_constr_nic_out_10_sink: flow_nic_out_10_sink
   - 5e+08 switch_nic_out_10_sink <= 0
 edge_flow_constr_nic_out_10_nic_in_0: flow_nic_out_10_nic_in_0
   - 61035.15625 switch_nic_out_10_nic_in_0 <= 0
 edge_flow_constr_nic_out_10_nic_in_1: flow_nic_out_10_nic_in_1
   - 61035.15625 switch_nic_out_10_nic_in_1 <= 0
 edge_flow_constr_nic_out_10_nic_in_2: flow_nic_out_10_nic_in_2
   - 61035.15625 switch_nic_out_10_nic_in_2 <= 0
 edge_flow_constr_nic_out_10_nic_in_3: flow_nic_out_10_nic_in_3
   - 61035.15625 switch_nic_out_10_nic_in_3 <= 0
 edge_flow_constr_nic_out_10_nic_in_4: flow_nic_out_10_nic_in_4
   - 61035.15625 switch_nic_out_10_nic_in_4 <= 0
 edge_flow_constr_nic_out_10_nic_in_5: flow_nic_out_10_nic_in_5
   - 61035.15625 switch_nic_out_10_nic_in_5 <= 0
 edge_flow_constr_nic_out_10_nic_in_6: flow_nic_out_10_nic_in_6
   - 61035.15625 switch_nic_out_10_nic_in_6 <= 0
 edge_flow_constr_nic_out_10_nic_in_7: flow_nic_out_10_nic_in_7
   - 61035.15625 switch_nic_out_10_nic_in_7 <= 0
 edge_flow_constr_nic_out_10_nic_in_8: flow_nic_out_10_nic_in_8
   - 61035.15625 switch_nic_out_10_nic_in_8 <= 0
 edge_flow_constr_nic_out_10_nic_in_9: flow_nic_out_10_nic_in_9
   - 61035.15625 switch_nic_out_10_nic_in_9 <= 0
 edge_flow_constr_nic_out_10_nic_in_11: flow_nic_out_10_nic_in_11
   - 61035.15625 switch_nic_out_10_nic_in_11 <= 0
 edge_flow_constr_nic_out_10_nic_in_12: flow_nic_out_10_nic_in_12
   - 61035.15625 switch_nic_out_10_nic_in_12 <= 0
 edge_flow_constr_nic_out_10_nic_in_13: flow_nic_out_10_nic_in_13
   - 61035.15625 switch_nic_out_10_nic_in_13 <= 0
 edge_flow_constr_nic_out_10_nic_in_14: flow_nic_out_10_nic_in_14
   - 61035.15625 switch_nic_out_10_nic_in_14 <= 0
 edge_flow_constr_nic_out_10_nic_in_15: flow_nic_out_10_nic_in_15
   - 61035.15625 switch_nic_out_10_nic_in_15 <= 0
 edge_flow_constr_nic_out_10_nic_in_16: flow_nic_out_10_nic_in_16
   - 61035.15625 switch_nic_out_10_nic_in_16 <= 0
 edge_flow_constr_nic_out_10_nic_in_17: flow_nic_out_10_nic_in_17
   - 61035.15625 switch_nic_out_10_nic_in_17 <= 0
 edge_flow_constr_nic_out_10_nic_in_18: flow_nic_out_10_nic_in_18
   - 61035.15625 switch_nic_out_10_nic_in_18 <= 0
 edge_flow_constr_nic_out_10_nic_in_19: flow_nic_out_10_nic_in_19
   - 61035.15625 switch_nic_out_10_nic_in_19 <= 0
 edge_flow_constr_nic_out_10_nic_in_20: flow_nic_out_10_nic_in_20
   - 61035.15625 switch_nic_out_10_nic_in_20 <= 0
 edge_flow_constr_nic_out_10_nic_in_21: flow_nic_out_10_nic_in_21
   - 61035.15625 switch_nic_out_10_nic_in_21 <= 0
 edge_flow_constr_nic_out_10_nic_in_22: flow_nic_out_10_nic_in_22
   - 61035.15625 switch_nic_out_10_nic_in_22 <= 0
 edge_flow_constr_nic_out_10_nic_in_23: flow_nic_out_10_nic_in_23
   - 61035.15625 switch_nic_out_10_nic_in_23 <= 0
 edge_flow_constr_nic_out_10_nic_in_24: flow_nic_out_10_nic_in_24
   - 61035.15625 switch_nic_out_10_nic_in_24 <= 0
 edge_flow_constr_nic_out_10_nic_in_25: flow_nic_out_10_nic_in_25
   - 61035.15625 switch_nic_out_10_nic_in_25 <= 0
 edge_flow_constr_nic_out_10_nic_in_26: flow_nic_out_10_nic_in_26
   - 61035.15625 switch_nic_out_10_nic_in_26 <= 0
 edge_flow_constr_nic_out_10_nic_in_27: flow_nic_out_10_nic_in_27
   - 61035.15625 switch_nic_out_10_nic_in_27 <= 0
 edge_flow_constr_nic_out_10_nic_in_28: flow_nic_out_10_nic_in_28
   - 61035.15625 switch_nic_out_10_nic_in_28 <= 0
 edge_flow_constr_nic_out_10_nic_in_29: flow_nic_out_10_nic_in_29
   - 61035.15625 switch_nic_out_10_nic_in_29 <= 0
 edge_flow_constr_nic_out_10_nic_in_30: flow_nic_out_10_nic_in_30
   - 61035.15625 switch_nic_out_10_nic_in_30 <= 0
 edge_flow_constr_nic_out_10_nic_in_31: flow_nic_out_10_nic_in_31
   - 61035.15625 switch_nic_out_10_nic_in_31 <= 0
 edge_flow_constr_nic_in_10_10: flow_nic_in_10_10
   - 3.90625e+06 switch_nic_in_10_10 <= 0
 edge_flow_constr_10_nic_out_10: flow_10_nic_out_10
   - 3.90625e+06 switch_10_nic_out_10 <= 0
 edge_flow_constr_source_nic_in_11: flow_source_nic_in_11
   - 5e+08 switch_source_nic_in_11 <= 0
 edge_flow_constr_nic_out_11_sink: flow_nic_out_11_sink
   - 5e+08 switch_nic_out_11_sink <= 0
 edge_flow_constr_nic_out_11_nic_in_0: flow_nic_out_11_nic_in_0
   - 61035.15625 switch_nic_out_11_nic_in_0 <= 0
 edge_flow_constr_nic_out_11_nic_in_1: flow_nic_out_11_nic_in_1
   - 61035.15625 switch_nic_out_11_nic_in_1 <= 0
 edge_flow_constr_nic_out_11_nic_in_2: flow_nic_out_11_nic_in_2
   - 61035.15625 switch_nic_out_11_nic_in_2 <= 0
 edge_flow_constr_nic_out_11_nic_in_3: flow_nic_out_11_nic_in_3
   - 61035.15625 switch_nic_out_11_nic_in_3 <= 0
 edge_flow_constr_nic_out_11_nic_in_4: flow_nic_out_11_nic_in_4
   - 61035.15625 switch_nic_out_11_nic_in_4 <= 0
 edge_flow_constr_nic_out_11_nic_in_5: flow_nic_out_11_nic_in_5
   - 61035.15625 switch_nic_out_11_nic_in_5 <= 0
 edge_flow_constr_nic_out_11_nic_in_6: flow_nic_out_11_nic_in_6
   - 61035.15625 switch_nic_out_11_nic_in_6 <= 0
 edge_flow_constr_nic_out_11_nic_in_7: flow_nic_out_11_nic_in_7
   - 61035.15625 switch_nic_out_11_nic_in_7 <= 0
 edge_flow_constr_nic_out_11_nic_in_8: flow_nic_out_11_nic_in_8
   - 61035.15625 switch_nic_out_11_nic_in_8 <= 0
 edge_flow_constr_nic_out_11_nic_in_9: flow_nic_out_11_nic_in_9
   - 61035.15625 switch_nic_out_11_nic_in_9 <= 0
 edge_flow_constr_nic_out_11_nic_in_10: flow_nic_out_11_nic_in_10
   - 61035.15625 switch_nic_out_11_nic_in_10 <= 0
 edge_flow_constr_nic_out_11_nic_in_12: flow_nic_out_11_nic_in_12
   - 61035.15625 switch_nic_out_11_nic_in_12 <= 0
 edge_flow_constr_nic_out_11_nic_in_13: flow_nic_out_11_nic_in_13
   - 61035.15625 switch_nic_out_11_nic_in_13 <= 0
 edge_flow_constr_nic_out_11_nic_in_14: flow_nic_out_11_nic_in_14
   - 61035.15625 switch_nic_out_11_nic_in_14 <= 0
 edge_flow_constr_nic_out_11_nic_in_15: flow_nic_out_11_nic_in_15
   - 61035.15625 switch_nic_out_11_nic_in_15 <= 0
 edge_flow_constr_nic_out_11_nic_in_16: flow_nic_out_11_nic_in_16
   - 61035.15625 switch_nic_out_11_nic_in_16 <= 0
 edge_flow_constr_nic_out_11_nic_in_17: flow_nic_out_11_nic_in_17
   - 61035.15625 switch_nic_out_11_nic_in_17 <= 0
 edge_flow_constr_nic_out_11_nic_in_18: flow_nic_out_11_nic_in_18
   - 61035.15625 switch_nic_out_11_nic_in_18 <= 0
 edge_flow_constr_nic_out_11_nic_in_19: flow_nic_out_11_nic_in_19
   - 61035.15625 switch_nic_out_11_nic_in_19 <= 0
 edge_flow_constr_nic_out_11_nic_in_20: flow_nic_out_11_nic_in_20
   - 61035.15625 switch_nic_out_11_nic_in_20 <= 0
 edge_flow_constr_nic_out_11_nic_in_21: flow_nic_out_11_nic_in_21
   - 61035.15625 switch_nic_out_11_nic_in_21 <= 0
 edge_flow_constr_nic_out_11_nic_in_22: flow_nic_out_11_nic_in_22
   - 61035.15625 switch_nic_out_11_nic_in_22 <= 0
 edge_flow_constr_nic_out_11_nic_in_23: flow_nic_out_11_nic_in_23
   - 61035.15625 switch_nic_out_11_nic_in_23 <= 0
 edge_flow_constr_nic_out_11_nic_in_24: flow_nic_out_11_nic_in_24
   - 61035.15625 switch_nic_out_11_nic_in_24 <= 0
 edge_flow_constr_nic_out_11_nic_in_25: flow_nic_out_11_nic_in_25
   - 61035.15625 switch_nic_out_11_nic_in_25 <= 0
 edge_flow_constr_nic_out_11_nic_in_26: flow_nic_out_11_nic_in_26
   - 61035.15625 switch_nic_out_11_nic_in_26 <= 0
 edge_flow_constr_nic_out_11_nic_in_27: flow_nic_out_11_nic_in_27
   - 61035.15625 switch_nic_out_11_nic_in_27 <= 0
 edge_flow_constr_nic_out_11_nic_in_28: flow_nic_out_11_nic_in_28
   - 61035.15625 switch_nic_out_11_nic_in_28 <= 0
 edge_flow_constr_nic_out_11_nic_in_29: flow_nic_out_11_nic_in_29
   - 61035.15625 switch_nic_out_11_nic_in_29 <= 0
 edge_flow_constr_nic_out_11_nic_in_30: flow_nic_out_11_nic_in_30
   - 61035.15625 switch_nic_out_11_nic_in_30 <= 0
 edge_flow_constr_nic_out_11_nic_in_31: flow_nic_out_11_nic_in_31
   - 61035.15625 switch_nic_out_11_nic_in_31 <= 0
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
 edge_flow_constr_nic_out_12_nic_in_1: flow_nic_out_12_nic_in_1
   - 61035.15625 switch_nic_out_12_nic_in_1 <= 0
 edge_flow_constr_nic_out_12_nic_in_2: flow_nic_out_12_nic_in_2
   - 61035.15625 switch_nic_out_12_nic_in_2 <= 0
 edge_flow_constr_nic_out_12_nic_in_3: flow_nic_out_12_nic_in_3
   - 61035.15625 switch_nic_out_12_nic_in_3 <= 0
 edge_flow_constr_nic_out_12_nic_in_4: flow_nic_out_12_nic_in_4
   - 61035.15625 switch_nic_out_12_nic_in_4 <= 0
 edge_flow_constr_nic_out_12_nic_in_5: flow_nic_out_12_nic_in_5
   - 61035.15625 switch_nic_out_12_nic_in_5 <= 0
 edge_flow_constr_nic_out_12_nic_in_6: flow_nic_out_12_nic_in_6
   - 61035.15625 switch_nic_out_12_nic_in_6 <= 0
 edge_flow_constr_nic_out_12_nic_in_7: flow_nic_out_12_nic_in_7
   - 61035.15625 switch_nic_out_12_nic_in_7 <= 0
 edge_flow_constr_nic_out_12_nic_in_8: flow_nic_out_12_nic_in_8
   - 61035.15625 switch_nic_out_12_nic_in_8 <= 0
 edge_flow_constr_nic_out_12_nic_in_9: flow_nic_out_12_nic_in_9
   - 61035.15625 switch_nic_out_12_nic_in_9 <= 0
 edge_flow_constr_nic_out_12_nic_in_10: flow_nic_out_12_nic_in_10
   - 61035.15625 switch_nic_out_12_nic_in_10 <= 0
 edge_flow_constr_nic_out_12_nic_in_11: flow_nic_out_12_nic_in_11
   - 61035.15625 switch_nic_out_12_nic_in_11 <= 0
 edge_flow_constr_nic_out_12_nic_in_13: flow_nic_out_12_nic_in_13
   - 61035.15625 switch_nic_out_12_nic_in_13 <= 0
 edge_flow_constr_nic_out_12_nic_in_14: flow_nic_out_12_nic_in_14
   - 61035.15625 switch_nic_out_12_nic_in_14 <= 0
 edge_flow_constr_nic_out_12_nic_in_15: flow_nic_out_12_nic_in_15
   - 61035.15625 switch_nic_out_12_nic_in_15 <= 0
 edge_flow_constr_nic_out_12_nic_in_16: flow_nic_out_12_nic_in_16
   - 61035.15625 switch_nic_out_12_nic_in_16 <= 0
 edge_flow_constr_nic_out_12_nic_in_17: flow_nic_out_12_nic_in_17
   - 61035.15625 switch_nic_out_12_nic_in_17 <= 0
 edge_flow_constr_nic_out_12_nic_in_18: flow_nic_out_12_nic_in_18
   - 61035.15625 switch_nic_out_12_nic_in_18 <= 0
 edge_flow_constr_nic_out_12_nic_in_19: flow_nic_out_12_nic_in_19
   - 61035.15625 switch_nic_out_12_nic_in_19 <= 0
 edge_flow_constr_nic_out_12_nic_in_20: flow_nic_out_12_nic_in_20
   - 61035.15625 switch_nic_out_12_nic_in_20 <= 0
 edge_flow_constr_nic_out_12_nic_in_21: flow_nic_out_12_nic_in_21
   - 61035.15625 switch_nic_out_12_nic_in_21 <= 0
 edge_flow_constr_nic_out_12_nic_in_22: flow_nic_out_12_nic_in_22
   - 61035.15625 switch_nic_out_12_nic_in_22 <= 0
 edge_flow_constr_nic_out_12_nic_in_23: flow_nic_out_12_nic_in_23
   - 61035.15625 switch_nic_out_12_nic_in_23 <= 0
 edge_flow_constr_nic_out_12_nic_in_24: flow_nic_out_12_nic_in_24
   - 61035.15625 switch_nic_out_12_nic_in_24 <= 0
 edge_flow_constr_nic_out_12_nic_in_25: flow_nic_out_12_nic_in_25
   - 61035.15625 switch_nic_out_12_nic_in_25 <= 0
 edge_flow_constr_nic_out_12_nic_in_26: flow_nic_out_12_nic_in_26
   - 61035.15625 switch_nic_out_12_nic_in_26 <= 0
 edge_flow_constr_nic_out_12_nic_in_27: flow_nic_out_12_nic_in_27
   - 61035.15625 switch_nic_out_12_nic_in_27 <= 0
 edge_flow_constr_nic_out_12_nic_in_28: flow_nic_out_12_nic_in_28
   - 61035.15625 switch_nic_out_12_nic_in_28 <= 0
 edge_flow_constr_nic_out_12_nic_in_29: flow_nic_out_12_nic_in_29
   - 61035.15625 switch_nic_out_12_nic_in_29 <= 0
 edge_flow_constr_nic_out_12_nic_in_30: flow_nic_out_12_nic_in_30
   - 61035.15625 switch_nic_out_12_nic_in_30 <= 0
 edge_flow_constr_nic_out_12_nic_in_31: flow_nic_out_12_nic_in_31
   - 61035.15625 switch_nic_out_12_nic_in_31 <= 0
 edge_flow_constr_nic_in_12_12: flow_nic_in_12_12
   - 3.90625e+06 switch_nic_in_12_12 <= 0
 edge_flow_constr_12_nic_out_12: flow_12_nic_out_12
   - 3.90625e+06 switch_12_nic_out_12 <= 0
 edge_flow_constr_source_nic_in_13: flow_source_nic_in_13
   - 5e+08 switch_source_nic_in_13 <= 0
 edge_flow_constr_nic_out_13_sink: flow_nic_out_13_sink
   - 5e+08 switch_nic_out_13_sink <= 0
 edge_flow_constr_nic_out_13_nic_in_0: flow_nic_out_13_nic_in_0
   - 61035.15625 switch_nic_out_13_nic_in_0 <= 0
 edge_flow_constr_nic_out_13_nic_in_1: flow_nic_out_13_nic_in_1
   - 61035.15625 switch_nic_out_13_nic_in_1 <= 0
 edge_flow_constr_nic_out_13_nic_in_2: flow_nic_out_13_nic_in_2
   - 61035.15625 switch_nic_out_13_nic_in_2 <= 0
 edge_flow_constr_nic_out_13_nic_in_3: flow_nic_out_13_nic_in_3
   - 61035.15625 switch_nic_out_13_nic_in_3 <= 0
 edge_flow_constr_nic_out_13_nic_in_4: flow_nic_out_13_nic_in_4
   - 61035.15625 switch_nic_out_13_nic_in_4 <= 0
 edge_flow_constr_nic_out_13_nic_in_5: flow_nic_out_13_nic_in_5
   - 61035.15625 switch_nic_out_13_nic_in_5 <= 0
 edge_flow_constr_nic_out_13_nic_in_6: flow_nic_out_13_nic_in_6
   - 61035.15625 switch_nic_out_13_nic_in_6 <= 0
 edge_flow_constr_nic_out_13_nic_in_7: flow_nic_out_13_nic_in_7
   - 61035.15625 switch_nic_out_13_nic_in_7 <= 0
 edge_flow_constr_nic_out_13_nic_in_8: flow_nic_out_13_nic_in_8
   - 61035.15625 switch_nic_out_13_nic_in_8 <= 0
 edge_flow_constr_nic_out_13_nic_in_9: flow_nic_out_13_nic_in_9
   - 61035.15625 switch_nic_out_13_nic_in_9 <= 0
 edge_flow_constr_nic_out_13_nic_in_10: flow_nic_out_13_nic_in_10
   - 61035.15625 switch_nic_out_13_nic_in_10 <= 0
 edge_flow_constr_nic_out_13_nic_in_11: flow_nic_out_13_nic_in_11
   - 61035.15625 switch_nic_out_13_nic_in_11 <= 0
 edge_flow_constr_nic_out_13_nic_in_12: flow_nic_out_13_nic_in_12
   - 61035.15625 switch_nic_out_13_nic_in_12 <= 0
 edge_flow_constr_nic_out_13_nic_in_14: flow_nic_out_13_nic_in_14
   - 61035.15625 switch_nic_out_13_nic_in_14 <= 0
 edge_flow_constr_nic_out_13_nic_in_15: flow_nic_out_13_nic_in_15
   - 61035.15625 switch_nic_out_13_nic_in_15 <= 0
 edge_flow_constr_nic_out_13_nic_in_16: flow_nic_out_13_nic_in_16
   - 61035.15625 switch_nic_out_13_nic_in_16 <= 0
 edge_flow_constr_nic_out_13_nic_in_17: flow_nic_out_13_nic_in_17
   - 61035.15625 switch_nic_out_13_nic_in_17 <= 0
 edge_flow_constr_nic_out_13_nic_in_18: flow_nic_out_13_nic_in_18
   - 61035.15625 switch_nic_out_13_nic_in_18 <= 0
 edge_flow_constr_nic_out_13_nic_in_19: flow_nic_out_13_nic_in_19
   - 61035.15625 switch_nic_out_13_nic_in_19 <= 0
 edge_flow_constr_nic_out_13_nic_in_20: flow_nic_out_13_nic_in_20
   - 61035.15625 switch_nic_out_13_nic_in_20 <= 0
 edge_flow_constr_nic_out_13_nic_in_21: flow_nic_out_13_nic_in_21
   - 61035.15625 switch_nic_out_13_nic_in_21 <= 0
 edge_flow_constr_nic_out_13_nic_in_22: flow_nic_out_13_nic_in_22
   - 61035.15625 switch_nic_out_13_nic_in_22 <= 0
 edge_flow_constr_nic_out_13_nic_in_23: flow_nic_out_13_nic_in_23
   - 61035.15625 switch_nic_out_13_nic_in_23 <= 0
 edge_flow_constr_nic_out_13_nic_in_24: flow_nic_out_13_nic_in_24
   - 61035.15625 switch_nic_out_13_nic_in_24 <= 0
 edge_flow_constr_nic_out_13_nic_in_25: flow_nic_out_13_nic_in_25
   - 61035.15625 switch_nic_out_13_nic_in_25 <= 0
 edge_flow_constr_nic_out_13_nic_in_26: flow_nic_out_13_nic_in_26
   - 61035.15625 switch_nic_out_13_nic_in_26 <= 0
 edge_flow_constr_nic_out_13_nic_in_27: flow_nic_out_13_nic_in_27
   - 61035.15625 switch_nic_out_13_nic_in_27 <= 0
 edge_flow_constr_nic_out_13_nic_in_28: flow_nic_out_13_nic_in_28
   - 61035.15625 switch_nic_out_13_nic_in_28 <= 0
 edge_flow_constr_nic_out_13_nic_in_29: flow_nic_out_13_nic_in_29
   - 61035.15625 switch_nic_out_13_nic_in_29 <= 0
 edge_flow_constr_nic_out_13_nic_in_30: flow_nic_out_13_nic_in_30
   - 61035.15625 switch_nic_out_13_nic_in_30 <= 0
 edge_flow_constr_nic_out_13_nic_in_31: flow_nic_out_13_nic_in_31
   - 61035.15625 switch_nic_out_13_nic_in_31 <= 0
 edge_flow_constr_nic_in_13_13: flow_nic_in_13_13
   - 3.90625e+06 switch_nic_in_13_13 <= 0
 edge_flow_constr_13_nic_out_13: flow_13_nic_out_13
   - 3.90625e+06 switch_13_nic_out_13 <= 0
 edge_flow_constr_source_nic_in_14: flow_source_nic_in_14
   - 5e+08 switch_source_nic_in_14 <= 0
 edge_flow_constr_nic_out_14_sink: flow_nic_out_14_sink
   - 5e+08 switch_nic_out_14_sink <= 0
 edge_flow_constr_nic_out_14_nic_in_0: flow_nic_out_14_nic_in_0
   - 61035.15625 switch_nic_out_14_nic_in_0 <= 0
 edge_flow_constr_nic_out_14_nic_in_1: flow_nic_out_14_nic_in_1
   - 61035.15625 switch_nic_out_14_nic_in_1 <= 0
 edge_flow_constr_nic_out_14_nic_in_2: flow_nic_out_14_nic_in_2
   - 61035.15625 switch_nic_out_14_nic_in_2 <= 0
 edge_flow_constr_nic_out_14_nic_in_3: flow_nic_out_14_nic_in_3
   - 61035.15625 switch_nic_out_14_nic_in_3 <= 0
 edge_flow_constr_nic_out_14_nic_in_4: flow_nic_out_14_nic_in_4
   - 61035.15625 switch_nic_out_14_nic_in_4 <= 0
 edge_flow_constr_nic_out_14_nic_in_5: flow_nic_out_14_nic_in_5
   - 61035.15625 switch_nic_out_14_nic_in_5 <= 0
 edge_flow_constr_nic_out_14_nic_in_6: flow_nic_out_14_nic_in_6
   - 61035.15625 switch_nic_out_14_nic_in_6 <= 0
 edge_flow_constr_nic_out_14_nic_in_7: flow_nic_out_14_nic_in_7
   - 61035.15625 switch_nic_out_14_nic_in_7 <= 0
 edge_flow_constr_nic_out_14_nic_in_8: flow_nic_out_14_nic_in_8
   - 61035.15625 switch_nic_out_14_nic_in_8 <= 0
 edge_flow_constr_nic_out_14_nic_in_9: flow_nic_out_14_nic_in_9
   - 61035.15625 switch_nic_out_14_nic_in_9 <= 0
 edge_flow_constr_nic_out_14_nic_in_10: flow_nic_out_14_nic_in_10
   - 61035.15625 switch_nic_out_14_nic_in_10 <= 0
 edge_flow_constr_nic_out_14_nic_in_11: flow_nic_out_14_nic_in_11
   - 61035.15625 switch_nic_out_14_nic_in_11 <= 0
 edge_flow_constr_nic_out_14_nic_in_12: flow_nic_out_14_nic_in_12
   - 61035.15625 switch_nic_out_14_nic_in_12 <= 0
 edge_flow_constr_nic_out_14_nic_in_13: flow_nic_out_14_nic_in_13
   - 61035.15625 switch_nic_out_14_nic_in_13 <= 0
 edge_flow_constr_nic_out_14_nic_in_15: flow_nic_out_14_nic_in_15
   - 61035.15625 switch_nic_out_14_nic_in_15 <= 0
 edge_flow_constr_nic_out_14_nic_in_16: flow_nic_out_14_nic_in_16
   - 61035.15625 switch_nic_out_14_nic_in_16 <= 0
 edge_flow_constr_nic_out_14_nic_in_17: flow_nic_out_14_nic_in_17
   - 61035.15625 switch_nic_out_14_nic_in_17 <= 0
 edge_flow_constr_nic_out_14_nic_in_18: flow_nic_out_14_nic_in_18
   - 61035.15625 switch_nic_out_14_nic_in_18 <= 0
 edge_flow_constr_nic_out_14_nic_in_19: flow_nic_out_14_nic_in_19
   - 61035.15625 switch_nic_out_14_nic_in_19 <= 0
 edge_flow_constr_nic_out_14_nic_in_20: flow_nic_out_14_nic_in_20
   - 61035.15625 switch_nic_out_14_nic_in_20 <= 0
 edge_flow_constr_nic_out_14_nic_in_21: flow_nic_out_14_nic_in_21
   - 61035.15625 switch_nic_out_14_nic_in_21 <= 0
 edge_flow_constr_nic_out_14_nic_in_22: flow_nic_out_14_nic_in_22
   - 61035.15625 switch_nic_out_14_nic_in_22 <= 0
 edge_flow_constr_nic_out_14_nic_in_23: flow_nic_out_14_nic_in_23
   - 61035.15625 switch_nic_out_14_nic_in_23 <= 0
 edge_flow_constr_nic_out_14_nic_in_24: flow_nic_out_14_nic_in_24
   - 61035.15625 switch_nic_out_14_nic_in_24 <= 0
 edge_flow_constr_nic_out_14_nic_in_25: flow_nic_out_14_nic_in_25
   - 61035.15625 switch_nic_out_14_nic_in_25 <= 0
 edge_flow_constr_nic_out_14_nic_in_26: flow_nic_out_14_nic_in_26
   - 61035.15625 switch_nic_out_14_nic_in_26 <= 0
 edge_flow_constr_nic_out_14_nic_in_27: flow_nic_out_14_nic_in_27
   - 61035.15625 switch_nic_out_14_nic_in_27 <= 0
 edge_flow_constr_nic_out_14_nic_in_28: flow_nic_out_14_nic_in_28
   - 61035.15625 switch_nic_out_14_nic_in_28 <= 0
 edge_flow_constr_nic_out_14_nic_in_29: flow_nic_out_14_nic_in_29
   - 61035.15625 switch_nic_out_14_nic_in_29 <= 0
 edge_flow_constr_nic_out_14_nic_in_30: flow_nic_out_14_nic_in_30
   - 61035.15625 switch_nic_out_14_nic_in_30 <= 0
 edge_flow_constr_nic_out_14_nic_in_31: flow_nic_out_14_nic_in_31
   - 61035.15625 switch_nic_out_14_nic_in_31 <= 0
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
 edge_flow_constr_nic_out_15_nic_in_1: flow_nic_out_15_nic_in_1
   - 61035.15625 switch_nic_out_15_nic_in_1 <= 0
 edge_flow_constr_nic_out_15_nic_in_2: flow_nic_out_15_nic_in_2
   - 61035.15625 switch_nic_out_15_nic_in_2 <= 0
 edge_flow_constr_nic_out_15_nic_in_3: flow_nic_out_15_nic_in_3
   - 61035.15625 switch_nic_out_15_nic_in_3 <= 0
 edge_flow_constr_nic_out_15_nic_in_4: flow_nic_out_15_nic_in_4
   - 61035.15625 switch_nic_out_15_nic_in_4 <= 0
 edge_flow_constr_nic_out_15_nic_in_5: flow_nic_out_15_nic_in_5
   - 61035.15625 switch_nic_out_15_nic_in_5 <= 0
 edge_flow_constr_nic_out_15_nic_in_6: flow_nic_out_15_nic_in_6
   - 61035.15625 switch_nic_out_15_nic_in_6 <= 0
 edge_flow_constr_nic_out_15_nic_in_7: flow_nic_out_15_nic_in_7
   - 61035.15625 switch_nic_out_15_nic_in_7 <= 0
 edge_flow_constr_nic_out_15_nic_in_8: flow_nic_out_15_nic_in_8
   - 61035.15625 switch_nic_out_15_nic_in_8 <= 0
 edge_flow_constr_nic_out_15_nic_in_9: flow_nic_out_15_nic_in_9
   - 61035.15625 switch_nic_out_15_nic_in_9 <= 0
 edge_flow_constr_nic_out_15_nic_in_10: flow_nic_out_15_nic_in_10
   - 61035.15625 switch_nic_out_15_nic_in_10 <= 0
 edge_flow_constr_nic_out_15_nic_in_11: flow_nic_out_15_nic_in_11
   - 61035.15625 switch_nic_out_15_nic_in_11 <= 0
 edge_flow_constr_nic_out_15_nic_in_12: flow_nic_out_15_nic_in_12
   - 61035.15625 switch_nic_out_15_nic_in_12 <= 0
 edge_flow_constr_nic_out_15_nic_in_13: flow_nic_out_15_nic_in_13
   - 61035.15625 switch_nic_out_15_nic_in_13 <= 0
 edge_flow_constr_nic_out_15_nic_in_14: flow_nic_out_15_nic_in_14
   - 61035.15625 switch_nic_out_15_nic_in_14 <= 0
 edge_flow_constr_nic_out_15_nic_in_16: flow_nic_out_15_nic_in_16
   - 61035.15625 switch_nic_out_15_nic_in_16 <= 0
 edge_flow_constr_nic_out_15_nic_in_17: flow_nic_out_15_nic_in_17
   - 61035.15625 switch_nic_out_15_nic_in_17 <= 0
 edge_flow_constr_nic_out_15_nic_in_18: flow_nic_out_15_nic_in_18
   - 61035.15625 switch_nic_out_15_nic_in_18 <= 0
 edge_flow_constr_nic_out_15_nic_in_19: flow_nic_out_15_nic_in_19
   - 61035.15625 switch_nic_out_15_nic_in_19 <= 0
 edge_flow_constr_nic_out_15_nic_in_20: flow_nic_out_15_nic_in_20
   - 61035.15625 switch_nic_out_15_nic_in_20 <= 0
 edge_flow_constr_nic_out_15_nic_in_21: flow_nic_out_15_nic_in_21
   - 61035.15625 switch_nic_out_15_nic_in_21 <= 0
 edge_flow_constr_nic_out_15_nic_in_22: flow_nic_out_15_nic_in_22
   - 61035.15625 switch_nic_out_15_nic_in_22 <= 0
 edge_flow_constr_nic_out_15_nic_in_23: flow_nic_out_15_nic_in_23
   - 61035.15625 switch_nic_out_15_nic_in_23 <= 0
 edge_flow_constr_nic_out_15_nic_in_24: flow_nic_out_15_nic_in_24
   - 61035.15625 switch_nic_out_15_nic_in_24 <= 0
 edge_flow_constr_nic_out_15_nic_in_25: flow_nic_out_15_nic_in_25
   - 61035.15625 switch_nic_out_15_nic_in_25 <= 0
 edge_flow_constr_nic_out_15_nic_in_26: flow_nic_out_15_nic_in_26
   - 61035.15625 switch_nic_out_15_nic_in_26 <= 0
 edge_flow_constr_nic_out_15_nic_in_27: flow_nic_out_15_nic_in_27
   - 61035.15625 switch_nic_out_15_nic_in_27 <= 0
 edge_flow_constr_nic_out_15_nic_in_28: flow_nic_out_15_nic_in_28
   - 61035.15625 switch_nic_out_15_nic_in_28 <= 0
 edge_flow_constr_nic_out_15_nic_in_29: flow_nic_out_15_nic_in_29
   - 61035.15625 switch_nic_out_15_nic_in_29 <= 0
 edge_flow_constr_nic_out_15_nic_in_30: flow_nic_out_15_nic_in_30
   - 61035.15625 switch_nic_out_15_nic_in_30 <= 0
 edge_flow_constr_nic_out_15_nic_in_31: flow_nic_out_15_nic_in_31
   - 61035.15625 switch_nic_out_15_nic_in_31 <= 0
 edge_flow_constr_nic_in_15_15: flow_nic_in_15_15
   - 3.90625e+06 switch_nic_in_15_15 <= 0
 edge_flow_constr_15_nic_out_15: flow_15_nic_out_15
   - 3.90625e+06 switch_15_nic_out_15 <= 0
 edge_flow_constr_source_nic_in_16: flow_source_nic_in_16
   - 5e+08 switch_source_nic_in_16 <= 0
 edge_flow_constr_nic_out_16_sink: flow_nic_out_16_sink
   - 5e+08 switch_nic_out_16_sink <= 0
 edge_flow_constr_nic_out_16_nic_in_0: flow_nic_out_16_nic_in_0
   - 61035.15625 switch_nic_out_16_nic_in_0 <= 0
 edge_flow_constr_nic_out_16_nic_in_1: flow_nic_out_16_nic_in_1
   - 61035.15625 switch_nic_out_16_nic_in_1 <= 0
 edge_flow_constr_nic_out_16_nic_in_2: flow_nic_out_16_nic_in_2
   - 61035.15625 switch_nic_out_16_nic_in_2 <= 0
 edge_flow_constr_nic_out_16_nic_in_3: flow_nic_out_16_nic_in_3
   - 61035.15625 switch_nic_out_16_nic_in_3 <= 0
 edge_flow_constr_nic_out_16_nic_in_4: flow_nic_out_16_nic_in_4
   - 61035.15625 switch_nic_out_16_nic_in_4 <= 0
 edge_flow_constr_nic_out_16_nic_in_5: flow_nic_out_16_nic_in_5
   - 61035.15625 switch_nic_out_16_nic_in_5 <= 0
 edge_flow_constr_nic_out_16_nic_in_6: flow_nic_out_16_nic_in_6
   - 61035.15625 switch_nic_out_16_nic_in_6 <= 0
 edge_flow_constr_nic_out_16_nic_in_7: flow_nic_out_16_nic_in_7
   - 61035.15625 switch_nic_out_16_nic_in_7 <= 0
 edge_flow_constr_nic_out_16_nic_in_8: flow_nic_out_16_nic_in_8
   - 61035.15625 switch_nic_out_16_nic_in_8 <= 0
 edge_flow_constr_nic_out_16_nic_in_9: flow_nic_out_16_nic_in_9
   - 61035.15625 switch_nic_out_16_nic_in_9 <= 0
 edge_flow_constr_nic_out_16_nic_in_10: flow_nic_out_16_nic_in_10
   - 61035.15625 switch_nic_out_16_nic_in_10 <= 0
 edge_flow_constr_nic_out_16_nic_in_11: flow_nic_out_16_nic_in_11
   - 61035.15625 switch_nic_out_16_nic_in_11 <= 0
 edge_flow_constr_nic_out_16_nic_in_12: flow_nic_out_16_nic_in_12
   - 61035.15625 switch_nic_out_16_nic_in_12 <= 0
 edge_flow_constr_nic_out_16_nic_in_13: flow_nic_out_16_nic_in_13
   - 61035.15625 switch_nic_out_16_nic_in_13 <= 0
 edge_flow_constr_nic_out_16_nic_in_14: flow_nic_out_16_nic_in_14
   - 61035.15625 switch_nic_out_16_nic_in_14 <= 0
 edge_flow_constr_nic_out_16_nic_in_15: flow_nic_out_16_nic_in_15
   - 61035.15625 switch_nic_out_16_nic_in_15 <= 0
 edge_flow_constr_nic_out_16_nic_in_17: flow_nic_out_16_nic_in_17
   - 61035.15625 switch_nic_out_16_nic_in_17 <= 0
 edge_flow_constr_nic_out_16_nic_in_18: flow_nic_out_16_nic_in_18
   - 61035.15625 switch_nic_out_16_nic_in_18 <= 0
 edge_flow_constr_nic_out_16_nic_in_19: flow_nic_out_16_nic_in_19
   - 61035.15625 switch_nic_out_16_nic_in_19 <= 0
 edge_flow_constr_nic_out_16_nic_in_20: flow_nic_out_16_nic_in_20
   - 61035.15625 switch_nic_out_16_nic_in_20 <= 0
 edge_flow_constr_nic_out_16_nic_in_21: flow_nic_out_16_nic_in_21
   - 61035.15625 switch_nic_out_16_nic_in_21 <= 0
 edge_flow_constr_nic_out_16_nic_in_22: flow_nic_out_16_nic_in_22
   - 61035.15625 switch_nic_out_16_nic_in_22 <= 0
 edge_flow_constr_nic_out_16_nic_in_23: flow_nic_out_16_nic_in_23
   - 61035.15625 switch_nic_out_16_nic_in_23 <= 0
 edge_flow_constr_nic_out_16_nic_in_24: flow_nic_out_16_nic_in_24
   - 61035.15625 switch_nic_out_16_nic_in_24 <= 0
 edge_flow_constr_nic_out_16_nic_in_25: flow_nic_out_16_nic_in_25
   - 61035.15625 switch_nic_out_16_nic_in_25 <= 0
 edge_flow_constr_nic_out_16_nic_in_26: flow_nic_out_16_nic_in_26
   - 61035.15625 switch_nic_out_16_nic_in_26 <= 0
 edge_flow_constr_nic_out_16_nic_in_27: flow_nic_out_16_nic_in_27
   - 61035.15625 switch_nic_out_16_nic_in_27 <= 0
 edge_flow_constr_nic_out_16_nic_in_28: flow_nic_out_16_nic_in_28
   - 61035.15625 switch_nic_out_16_nic_in_28 <= 0
 edge_flow_constr_nic_out_16_nic_in_29: flow_nic_out_16_nic_in_29
   - 61035.15625 switch_nic_out_16_nic_in_29 <= 0
 edge_flow_constr_nic_out_16_nic_in_30: flow_nic_out_16_nic_in_30
   - 61035.15625 switch_nic_out_16_nic_in_30 <= 0
 edge_flow_constr_nic_out_16_nic_in_31: flow_nic_out_16_nic_in_31
   - 61035.15625 switch_nic_out_16_nic_in_31 <= 0
 edge_flow_constr_nic_in_16_16: flow_nic_in_16_16
   - 3.90625e+06 switch_nic_in_16_16 <= 0
 edge_flow_constr_16_nic_out_16: flow_16_nic_out_16
   - 3.90625e+06 switch_16_nic_out_16 <= 0
 edge_flow_constr_source_nic_in_17: flow_source_nic_in_17
   - 5e+08 switch_source_nic_in_17 <= 0
 edge_flow_constr_nic_out_17_sink: flow_nic_out_17_sink
   - 5e+08 switch_nic_out_17_sink <= 0
 edge_flow_constr_nic_out_17_nic_in_0: flow_nic_out_17_nic_in_0
   - 61035.15625 switch_nic_out_17_nic_in_0 <= 0
 edge_flow_constr_nic_out_17_nic_in_1: flow_nic_out_17_nic_in_1
   - 61035.15625 switch_nic_out_17_nic_in_1 <= 0
 edge_flow_constr_nic_out_17_nic_in_2: flow_nic_out_17_nic_in_2
   - 61035.15625 switch_nic_out_17_nic_in_2 <= 0
 edge_flow_constr_nic_out_17_nic_in_3: flow_nic_out_17_nic_in_3
   - 61035.15625 switch_nic_out_17_nic_in_3 <= 0
 edge_flow_constr_nic_out_17_nic_in_4: flow_nic_out_17_nic_in_4
   - 61035.15625 switch_nic_out_17_nic_in_4 <= 0
 edge_flow_constr_nic_out_17_nic_in_5: flow_nic_out_17_nic_in_5
   - 61035.15625 switch_nic_out_17_nic_in_5 <= 0
 edge_flow_constr_nic_out_17_nic_in_6: flow_nic_out_17_nic_in_6
   - 61035.15625 switch_nic_out_17_nic_in_6 <= 0
 edge_flow_constr_nic_out_17_nic_in_7: flow_nic_out_17_nic_in_7
   - 61035.15625 switch_nic_out_17_nic_in_7 <= 0
 edge_flow_constr_nic_out_17_nic_in_8: flow_nic_out_17_nic_in_8
   - 61035.15625 switch_nic_out_17_nic_in_8 <= 0
 edge_flow_constr_nic_out_17_nic_in_9: flow_nic_out_17_nic_in_9
   - 61035.15625 switch_nic_out_17_nic_in_9 <= 0
 edge_flow_constr_nic_out_17_nic_in_10: flow_nic_out_17_nic_in_10
   - 61035.15625 switch_nic_out_17_nic_in_10 <= 0
 edge_flow_constr_nic_out_17_nic_in_11: flow_nic_out_17_nic_in_11
   - 61035.15625 switch_nic_out_17_nic_in_11 <= 0
 edge_flow_constr_nic_out_17_nic_in_12: flow_nic_out_17_nic_in_12
   - 61035.15625 switch_nic_out_17_nic_in_12 <= 0
 edge_flow_constr_nic_out_17_nic_in_13: flow_nic_out_17_nic_in_13
   - 61035.15625 switch_nic_out_17_nic_in_13 <= 0
 edge_flow_constr_nic_out_17_nic_in_14: flow_nic_out_17_nic_in_14
   - 61035.15625 switch_nic_out_17_nic_in_14 <= 0
 edge_flow_constr_nic_out_17_nic_in_15: flow_nic_out_17_nic_in_15
   - 61035.15625 switch_nic_out_17_nic_in_15 <= 0
 edge_flow_constr_nic_out_17_nic_in_16: flow_nic_out_17_nic_in_16
   - 61035.15625 switch_nic_out_17_nic_in_16 <= 0
 edge_flow_constr_nic_out_17_nic_in_18: flow_nic_out_17_nic_in_18
   - 61035.15625 switch_nic_out_17_nic_in_18 <= 0
 edge_flow_constr_nic_out_17_nic_in_19: flow_nic_out_17_nic_in_19
   - 61035.15625 switch_nic_out_17_nic_in_19 <= 0
 edge_flow_constr_nic_out_17_nic_in_20: flow_nic_out_17_nic_in_20
   - 61035.15625 switch_nic_out_17_nic_in_20 <= 0
 edge_flow_constr_nic_out_17_nic_in_21: flow_nic_out_17_nic_in_21
   - 61035.15625 switch_nic_out_17_nic_in_21 <= 0
 edge_flow_constr_nic_out_17_nic_in_22: flow_nic_out_17_nic_in_22
   - 61035.15625 switch_nic_out_17_nic_in_22 <= 0
 edge_flow_constr_nic_out_17_nic_in_23: flow_nic_out_17_nic_in_23
   - 61035.15625 switch_nic_out_17_nic_in_23 <= 0
 edge_flow_constr_nic_out_17_nic_in_24: flow_nic_out_17_nic_in_24
   - 61035.15625 switch_nic_out_17_nic_in_24 <= 0
 edge_flow_constr_nic_out_17_nic_in_25: flow_nic_out_17_nic_in_25
   - 61035.15625 switch_nic_out_17_nic_in_25 <= 0
 edge_flow_constr_nic_out_17_nic_in_26: flow_nic_out_17_nic_in_26
   - 61035.15625 switch_nic_out_17_nic_in_26 <= 0
 edge_flow_constr_nic_out_17_nic_in_27: flow_nic_out_17_nic_in_27
   - 61035.15625 switch_nic_out_17_nic_in_27 <= 0
 edge_flow_constr_nic_out_17_nic_in_28: flow_nic_out_17_nic_in_28
   - 61035.15625 switch_nic_out_17_nic_in_28 <= 0
 edge_flow_constr_nic_out_17_nic_in_29: flow_nic_out_17_nic_in_29
   - 61035.15625 switch_nic_out_17_nic_in_29 <= 0
 edge_flow_constr_nic_out_17_nic_in_30: flow_nic_out_17_nic_in_30
   - 61035.15625 switch_nic_out_17_nic_in_30 <= 0
 edge_flow_constr_nic_out_17_nic_in_31: flow_nic_out_17_nic_in_31
   - 61035.15625 switch_nic_out_17_nic_in_31 <= 0
 edge_flow_constr_nic_in_17_17: flow_nic_in_17_17
   - 3.90625e+06 switch_nic_in_17_17 <= 0
 edge_flow_constr_17_nic_out_17: flow_17_nic_out_17
   - 3.90625e+06 switch_17_nic_out_17 <= 0
 edge_flow_constr_source_nic_in_18: flow_source_nic_in_18
   - 5e+08 switch_source_nic_in_18 <= 0
 edge_flow_constr_nic_out_18_sink: flow_nic_out_18_sink
   - 5e+08 switch_nic_out_18_sink <= 0
 edge_flow_constr_nic_out_18_nic_in_0: flow_nic_out_18_nic_in_0
   - 61035.15625 switch_nic_out_18_nic_in_0 <= 0
 edge_flow_constr_nic_out_18_nic_in_1: flow_nic_out_18_nic_in_1
   - 61035.15625 switch_nic_out_18_nic_in_1 <= 0
 edge_flow_constr_nic_out_18_nic_in_2: flow_nic_out_18_nic_in_2
   - 61035.15625 switch_nic_out_18_nic_in_2 <= 0
 edge_flow_constr_nic_out_18_nic_in_3: flow_nic_out_18_nic_in_3
   - 61035.15625 switch_nic_out_18_nic_in_3 <= 0
 edge_flow_constr_nic_out_18_nic_in_4: flow_nic_out_18_nic_in_4
   - 61035.15625 switch_nic_out_18_nic_in_4 <= 0
 edge_flow_constr_nic_out_18_nic_in_5: flow_nic_out_18_nic_in_5
   - 61035.15625 switch_nic_out_18_nic_in_5 <= 0
 edge_flow_constr_nic_out_18_nic_in_6: flow_nic_out_18_nic_in_6
   - 61035.15625 switch_nic_out_18_nic_in_6 <= 0
 edge_flow_constr_nic_out_18_nic_in_7: flow_nic_out_18_nic_in_7
   - 61035.15625 switch_nic_out_18_nic_in_7 <= 0
 edge_flow_constr_nic_out_18_nic_in_8: flow_nic_out_18_nic_in_8
   - 61035.15625 switch_nic_out_18_nic_in_8 <= 0
 edge_flow_constr_nic_out_18_nic_in_9: flow_nic_out_18_nic_in_9
   - 61035.15625 switch_nic_out_18_nic_in_9 <= 0
 edge_flow_constr_nic_out_18_nic_in_10: flow_nic_out_18_nic_in_10
   - 61035.15625 switch_nic_out_18_nic_in_10 <= 0
 edge_flow_constr_nic_out_18_nic_in_11: flow_nic_out_18_nic_in_11
   - 61035.15625 switch_nic_out_18_nic_in_11 <= 0
 edge_flow_constr_nic_out_18_nic_in_12: flow_nic_out_18_nic_in_12
   - 61035.15625 switch_nic_out_18_nic_in_12 <= 0
 edge_flow_constr_nic_out_18_nic_in_13: flow_nic_out_18_nic_in_13
   - 61035.15625 switch_nic_out_18_nic_in_13 <= 0
 edge_flow_constr_nic_out_18_nic_in_14: flow_nic_out_18_nic_in_14
   - 61035.15625 switch_nic_out_18_nic_in_14 <= 0
 edge_flow_constr_nic_out_18_nic_in_15: flow_nic_out_18_nic_in_15
   - 61035.15625 switch_nic_out_18_nic_in_15 <= 0
 edge_flow_constr_nic_out_18_nic_in_16: flow_nic_out_18_nic_in_16
   - 61035.15625 switch_nic_out_18_nic_in_16 <= 0
 edge_flow_constr_nic_out_18_nic_in_17: flow_nic_out_18_nic_in_17
   - 61035.15625 switch_nic_out_18_nic_in_17 <= 0
 edge_flow_constr_nic_out_18_nic_in_19: flow_nic_out_18_nic_in_19
   - 61035.15625 switch_nic_out_18_nic_in_19 <= 0
 edge_flow_constr_nic_out_18_nic_in_20: flow_nic_out_18_nic_in_20
   - 61035.15625 switch_nic_out_18_nic_in_20 <= 0
 edge_flow_constr_nic_out_18_nic_in_21: flow_nic_out_18_nic_in_21
   - 61035.15625 switch_nic_out_18_nic_in_21 <= 0
 edge_flow_constr_nic_out_18_nic_in_22: flow_nic_out_18_nic_in_22
   - 61035.15625 switch_nic_out_18_nic_in_22 <= 0
 edge_flow_constr_nic_out_18_nic_in_23: flow_nic_out_18_nic_in_23
   - 61035.15625 switch_nic_out_18_nic_in_23 <= 0
 edge_flow_constr_nic_out_18_nic_in_24: flow_nic_out_18_nic_in_24
   - 61035.15625 switch_nic_out_18_nic_in_24 <= 0
 edge_flow_constr_nic_out_18_nic_in_25: flow_nic_out_18_nic_in_25
   - 61035.15625 switch_nic_out_18_nic_in_25 <= 0
 edge_flow_constr_nic_out_18_nic_in_26: flow_nic_out_18_nic_in_26
   - 61035.15625 switch_nic_out_18_nic_in_26 <= 0
 edge_flow_constr_nic_out_18_nic_in_27: flow_nic_out_18_nic_in_27
   - 61035.15625 switch_nic_out_18_nic_in_27 <= 0
 edge_flow_constr_nic_out_18_nic_in_28: flow_nic_out_18_nic_in_28
   - 61035.15625 switch_nic_out_18_nic_in_28 <= 0
 edge_flow_constr_nic_out_18_nic_in_29: flow_nic_out_18_nic_in_29
   - 61035.15625 switch_nic_out_18_nic_in_29 <= 0
 edge_flow_constr_nic_out_18_nic_in_30: flow_nic_out_18_nic_in_30
   - 61035.15625 switch_nic_out_18_nic_in_30 <= 0
 edge_flow_constr_nic_out_18_nic_in_31: flow_nic_out_18_nic_in_31
   - 61035.15625 switch_nic_out_18_nic_in_31 <= 0
 edge_flow_constr_nic_in_18_18: flow_nic_in_18_18
   - 3.90625e+06 switch_nic_in_18_18 <= 0
 edge_flow_constr_18_nic_out_18: flow_18_nic_out_18
   - 3.90625e+06 switch_18_nic_out_18 <= 0
 edge_flow_constr_source_nic_in_19: flow_source_nic_in_19
   - 5e+08 switch_source_nic_in_19 <= 0
 edge_flow_constr_nic_out_19_sink: flow_nic_out_19_sink
   - 5e+08 switch_nic_out_19_sink <= 0
 edge_flow_constr_nic_out_19_nic_in_0: flow_nic_out_19_nic_in_0
   - 61035.15625 switch_nic_out_19_nic_in_0 <= 0
 edge_flow_constr_nic_out_19_nic_in_1: flow_nic_out_19_nic_in_1
   - 61035.15625 switch_nic_out_19_nic_in_1 <= 0
 edge_flow_constr_nic_out_19_nic_in_2: flow_nic_out_19_nic_in_2
   - 61035.15625 switch_nic_out_19_nic_in_2 <= 0
 edge_flow_constr_nic_out_19_nic_in_3: flow_nic_out_19_nic_in_3
   - 61035.15625 switch_nic_out_19_nic_in_3 <= 0
 edge_flow_constr_nic_out_19_nic_in_4: flow_nic_out_19_nic_in_4
   - 61035.15625 switch_nic_out_19_nic_in_4 <= 0
 edge_flow_constr_nic_out_19_nic_in_5: flow_nic_out_19_nic_in_5
   - 61035.15625 switch_nic_out_19_nic_in_5 <= 0
 edge_flow_constr_nic_out_19_nic_in_6: flow_nic_out_19_nic_in_6
   - 61035.15625 switch_nic_out_19_nic_in_6 <= 0
 edge_flow_constr_nic_out_19_nic_in_7: flow_nic_out_19_nic_in_7
   - 61035.15625 switch_nic_out_19_nic_in_7 <= 0
 edge_flow_constr_nic_out_19_nic_in_8: flow_nic_out_19_nic_in_8
   - 61035.15625 switch_nic_out_19_nic_in_8 <= 0
 edge_flow_constr_nic_out_19_nic_in_9: flow_nic_out_19_nic_in_9
   - 61035.15625 switch_nic_out_19_nic_in_9 <= 0
 edge_flow_constr_nic_out_19_nic_in_10: flow_nic_out_19_nic_in_10
   - 61035.15625 switch_nic_out_19_nic_in_10 <= 0
 edge_flow_constr_nic_out_19_nic_in_11: flow_nic_out_19_nic_in_11
   - 61035.15625 switch_nic_out_19_nic_in_11 <= 0
 edge_flow_constr_nic_out_19_nic_in_12: flow_nic_out_19_nic_in_12
   - 61035.15625 switch_nic_out_19_nic_in_12 <= 0
 edge_flow_constr_nic_out_19_nic_in_13: flow_nic_out_19_nic_in_13
   - 61035.15625 switch_nic_out_19_nic_in_13 <= 0
 edge_flow_constr_nic_out_19_nic_in_14: flow_nic_out_19_nic_in_14
   - 61035.15625 switch_nic_out_19_nic_in_14 <= 0
 edge_flow_constr_nic_out_19_nic_in_15: flow_nic_out_19_nic_in_15
   - 61035.15625 switch_nic_out_19_nic_in_15 <= 0
 edge_flow_constr_nic_out_19_nic_in_16: flow_nic_out_19_nic_in_16
   - 61035.15625 switch_nic_out_19_nic_in_16 <= 0
 edge_flow_constr_nic_out_19_nic_in_17: flow_nic_out_19_nic_in_17
   - 61035.15625 switch_nic_out_19_nic_in_17 <= 0
 edge_flow_constr_nic_out_19_nic_in_18: flow_nic_out_19_nic_in_18
   - 61035.15625 switch_nic_out_19_nic_in_18 <= 0
 edge_flow_constr_nic_out_19_nic_in_20: flow_nic_out_19_nic_in_20
   - 61035.15625 switch_nic_out_19_nic_in_20 <= 0
 edge_flow_constr_nic_out_19_nic_in_21: flow_nic_out_19_nic_in_21
   - 61035.15625 switch_nic_out_19_nic_in_21 <= 0
 edge_flow_constr_nic_out_19_nic_in_22: flow_nic_out_19_nic_in_22
   - 61035.15625 switch_nic_out_19_nic_in_22 <= 0
 edge_flow_constr_nic_out_19_nic_in_23: flow_nic_out_19_nic_in_23
   - 61035.15625 switch_nic_out_19_nic_in_23 <= 0
 edge_flow_constr_nic_out_19_nic_in_24: flow_nic_out_19_nic_in_24
   - 61035.15625 switch_nic_out_19_nic_in_24 <= 0
 edge_flow_constr_nic_out_19_nic_in_25: flow_nic_out_19_nic_in_25
   - 61035.15625 switch_nic_out_19_nic_in_25 <= 0
 edge_flow_constr_nic_out_19_nic_in_26: flow_nic_out_19_nic_in_26
   - 61035.15625 switch_nic_out_19_nic_in_26 <= 0
 edge_flow_constr_nic_out_19_nic_in_27: flow_nic_out_19_nic_in_27
   - 61035.15625 switch_nic_out_19_nic_in_27 <= 0
 edge_flow_constr_nic_out_19_nic_in_28: flow_nic_out_19_nic_in_28
   - 61035.15625 switch_nic_out_19_nic_in_28 <= 0
 edge_flow_constr_nic_out_19_nic_in_29: flow_nic_out_19_nic_in_29
   - 61035.15625 switch_nic_out_19_nic_in_29 <= 0
 edge_flow_constr_nic_out_19_nic_in_30: flow_nic_out_19_nic_in_30
   - 61035.15625 switch_nic_out_19_nic_in_30 <= 0
 edge_flow_constr_nic_out_19_nic_in_31: flow_nic_out_19_nic_in_31
   - 61035.15625 switch_nic_out_19_nic_in_31 <= 0
 edge_flow_constr_nic_in_19_19: flow_nic_in_19_19
   - 3.90625e+06 switch_nic_in_19_19 <= 0
 edge_flow_constr_19_nic_out_19: flow_19_nic_out_19
   - 3.90625e+06 switch_19_nic_out_19 <= 0
 edge_flow_constr_source_nic_in_20: flow_source_nic_in_20
   - 5e+08 switch_source_nic_in_20 <= 0
 edge_flow_constr_nic_out_20_sink: flow_nic_out_20_sink
   - 5e+08 switch_nic_out_20_sink <= 0
 edge_flow_constr_nic_out_20_nic_in_0: flow_nic_out_20_nic_in_0
   - 61035.15625 switch_nic_out_20_nic_in_0 <= 0
 edge_flow_constr_nic_out_20_nic_in_1: flow_nic_out_20_nic_in_1
   - 61035.15625 switch_nic_out_20_nic_in_1 <= 0
 edge_flow_constr_nic_out_20_nic_in_2: flow_nic_out_20_nic_in_2
   - 61035.15625 switch_nic_out_20_nic_in_2 <= 0
 edge_flow_constr_nic_out_20_nic_in_3: flow_nic_out_20_nic_in_3
   - 61035.15625 switch_nic_out_20_nic_in_3 <= 0
 edge_flow_constr_nic_out_20_nic_in_4: flow_nic_out_20_nic_in_4
   - 61035.15625 switch_nic_out_20_nic_in_4 <= 0
 edge_flow_constr_nic_out_20_nic_in_5: flow_nic_out_20_nic_in_5
   - 61035.15625 switch_nic_out_20_nic_in_5 <= 0
 edge_flow_constr_nic_out_20_nic_in_6: flow_nic_out_20_nic_in_6
   - 61035.15625 switch_nic_out_20_nic_in_6 <= 0
 edge_flow_constr_nic_out_20_nic_in_7: flow_nic_out_20_nic_in_7
   - 61035.15625 switch_nic_out_20_nic_in_7 <= 0
 edge_flow_constr_nic_out_20_nic_in_8: flow_nic_out_20_nic_in_8
   - 61035.15625 switch_nic_out_20_nic_in_8 <= 0
 edge_flow_constr_nic_out_20_nic_in_9: flow_nic_out_20_nic_in_9
   - 61035.15625 switch_nic_out_20_nic_in_9 <= 0
 edge_flow_constr_nic_out_20_nic_in_10: flow_nic_out_20_nic_in_10
   - 61035.15625 switch_nic_out_20_nic_in_10 <= 0
 edge_flow_constr_nic_out_20_nic_in_11: flow_nic_out_20_nic_in_11
   - 61035.15625 switch_nic_out_20_nic_in_11 <= 0
 edge_flow_constr_nic_out_20_nic_in_12: flow_nic_out_20_nic_in_12
   - 61035.15625 switch_nic_out_20_nic_in_12 <= 0
 edge_flow_constr_nic_out_20_nic_in_13: flow_nic_out_20_nic_in_13
   - 61035.15625 switch_nic_out_20_nic_in_13 <= 0
 edge_flow_constr_nic_out_20_nic_in_14: flow_nic_out_20_nic_in_14
   - 61035.15625 switch_nic_out_20_nic_in_14 <= 0
 edge_flow_constr_nic_out_20_nic_in_15: flow_nic_out_20_nic_in_15
   - 61035.15625 switch_nic_out_20_nic_in_15 <= 0
 edge_flow_constr_nic_out_20_nic_in_16: flow_nic_out_20_nic_in_16
   - 61035.15625 switch_nic_out_20_nic_in_16 <= 0
 edge_flow_constr_nic_out_20_nic_in_17: flow_nic_out_20_nic_in_17
   - 61035.15625 switch_nic_out_20_nic_in_17 <= 0
 edge_flow_constr_nic_out_20_nic_in_18: flow_nic_out_20_nic_in_18
   - 61035.15625 switch_nic_out_20_nic_in_18 <= 0
 edge_flow_constr_nic_out_20_nic_in_19: flow_nic_out_20_nic_in_19
   - 61035.15625 switch_nic_out_20_nic_in_19 <= 0
 edge_flow_constr_nic_out_20_nic_in_21: flow_nic_out_20_nic_in_21
   - 61035.15625 switch_nic_out_20_nic_in_21 <= 0
 edge_flow_constr_nic_out_20_nic_in_22: flow_nic_out_20_nic_in_22
   - 61035.15625 switch_nic_out_20_nic_in_22 <= 0
 edge_flow_constr_nic_out_20_nic_in_23: flow_nic_out_20_nic_in_23
   - 61035.15625 switch_nic_out_20_nic_in_23 <= 0
 edge_flow_constr_nic_out_20_nic_in_24: flow_nic_out_20_nic_in_24
   - 61035.15625 switch_nic_out_20_nic_in_24 <= 0
 edge_flow_constr_nic_out_20_nic_in_25: flow_nic_out_20_nic_in_25
   - 61035.15625 switch_nic_out_20_nic_in_25 <= 0
 edge_flow_constr_nic_out_20_nic_in_26: flow_nic_out_20_nic_in_26
   - 61035.15625 switch_nic_out_20_nic_in_26 <= 0
 edge_flow_constr_nic_out_20_nic_in_27: flow_nic_out_20_nic_in_27
   - 61035.15625 switch_nic_out_20_nic_in_27 <= 0
 edge_flow_constr_nic_out_20_nic_in_28: flow_nic_out_20_nic_in_28
   - 61035.15625 switch_nic_out_20_nic_in_28 <= 0
 edge_flow_constr_nic_out_20_nic_in_29: flow_nic_out_20_nic_in_29
   - 61035.15625 switch_nic_out_20_nic_in_29 <= 0
 edge_flow_constr_nic_out_20_nic_in_30: flow_nic_out_20_nic_in_30
   - 61035.15625 switch_nic_out_20_nic_in_30 <= 0
 edge_flow_constr_nic_out_20_nic_in_31: flow_nic_out_20_nic_in_31
   - 61035.15625 switch_nic_out_20_nic_in_31 <= 0
 edge_flow_constr_nic_in_20_20: flow_nic_in_20_20
   - 3.90625e+06 switch_nic_in_20_20 <= 0
 edge_flow_constr_20_nic_out_20: flow_20_nic_out_20
   - 3.90625e+06 switch_20_nic_out_20 <= 0
 edge_flow_constr_source_nic_in_21: flow_source_nic_in_21
   - 5e+08 switch_source_nic_in_21 <= 0
 edge_flow_constr_nic_out_21_sink: flow_nic_out_21_sink
   - 5e+08 switch_nic_out_21_sink <= 0
 edge_flow_constr_nic_out_21_nic_in_0: flow_nic_out_21_nic_in_0
   - 61035.15625 switch_nic_out_21_nic_in_0 <= 0
 edge_flow_constr_nic_out_21_nic_in_1: flow_nic_out_21_nic_in_1
   - 61035.15625 switch_nic_out_21_nic_in_1 <= 0
 edge_flow_constr_nic_out_21_nic_in_2: flow_nic_out_21_nic_in_2
   - 61035.15625 switch_nic_out_21_nic_in_2 <= 0
 edge_flow_constr_nic_out_21_nic_in_3: flow_nic_out_21_nic_in_3
   - 61035.15625 switch_nic_out_21_nic_in_3 <= 0
 edge_flow_constr_nic_out_21_nic_in_4: flow_nic_out_21_nic_in_4
   - 61035.15625 switch_nic_out_21_nic_in_4 <= 0
 edge_flow_constr_nic_out_21_nic_in_5: flow_nic_out_21_nic_in_5
   - 61035.15625 switch_nic_out_21_nic_in_5 <= 0
 edge_flow_constr_nic_out_21_nic_in_6: flow_nic_out_21_nic_in_6
   - 61035.15625 switch_nic_out_21_nic_in_6 <= 0
 edge_flow_constr_nic_out_21_nic_in_7: flow_nic_out_21_nic_in_7
   - 61035.15625 switch_nic_out_21_nic_in_7 <= 0
 edge_flow_constr_nic_out_21_nic_in_8: flow_nic_out_21_nic_in_8
   - 61035.15625 switch_nic_out_21_nic_in_8 <= 0
 edge_flow_constr_nic_out_21_nic_in_9: flow_nic_out_21_nic_in_9
   - 61035.15625 switch_nic_out_21_nic_in_9 <= 0
 edge_flow_constr_nic_out_21_nic_in_10: flow_nic_out_21_nic_in_10
   - 61035.15625 switch_nic_out_21_nic_in_10 <= 0
 edge_flow_constr_nic_out_21_nic_in_11: flow_nic_out_21_nic_in_11
   - 61035.15625 switch_nic_out_21_nic_in_11 <= 0
 edge_flow_constr_nic_out_21_nic_in_12: flow_nic_out_21_nic_in_12
   - 61035.15625 switch_nic_out_21_nic_in_12 <= 0
 edge_flow_constr_nic_out_21_nic_in_13: flow_nic_out_21_nic_in_13
   - 61035.15625 switch_nic_out_21_nic_in_13 <= 0
 edge_flow_constr_nic_out_21_nic_in_14: flow_nic_out_21_nic_in_14
   - 61035.15625 switch_nic_out_21_nic_in_14 <= 0
 edge_flow_constr_nic_out_21_nic_in_15: flow_nic_out_21_nic_in_15
   - 61035.15625 switch_nic_out_21_nic_in_15 <= 0
 edge_flow_constr_nic_out_21_nic_in_16: flow_nic_out_21_nic_in_16
   - 61035.15625 switch_nic_out_21_nic_in_16 <= 0
 edge_flow_constr_nic_out_21_nic_in_17: flow_nic_out_21_nic_in_17
   - 61035.15625 switch_nic_out_21_nic_in_17 <= 0
 edge_flow_constr_nic_out_21_nic_in_18: flow_nic_out_21_nic_in_18
   - 61035.15625 switch_nic_out_21_nic_in_18 <= 0
 edge_flow_constr_nic_out_21_nic_in_19: flow_nic_out_21_nic_in_19
   - 61035.15625 switch_nic_out_21_nic_in_19 <= 0
 edge_flow_constr_nic_out_21_nic_in_20: flow_nic_out_21_nic_in_20
   - 61035.15625 switch_nic_out_21_nic_in_20 <= 0
 edge_flow_constr_nic_out_21_nic_in_22: flow_nic_out_21_nic_in_22
   - 61035.15625 switch_nic_out_21_nic_in_22 <= 0
 edge_flow_constr_nic_out_21_nic_in_23: flow_nic_out_21_nic_in_23
   - 61035.15625 switch_nic_out_21_nic_in_23 <= 0
 edge_flow_constr_nic_out_21_nic_in_24: flow_nic_out_21_nic_in_24
   - 61035.15625 switch_nic_out_21_nic_in_24 <= 0
 edge_flow_constr_nic_out_21_nic_in_25: flow_nic_out_21_nic_in_25
   - 61035.15625 switch_nic_out_21_nic_in_25 <= 0
 edge_flow_constr_nic_out_21_nic_in_26: flow_nic_out_21_nic_in_26
   - 61035.15625 switch_nic_out_21_nic_in_26 <= 0
 edge_flow_constr_nic_out_21_nic_in_27: flow_nic_out_21_nic_in_27
   - 61035.15625 switch_nic_out_21_nic_in_27 <= 0
 edge_flow_constr_nic_out_21_nic_in_28: flow_nic_out_21_nic_in_28
   - 61035.15625 switch_nic_out_21_nic_in_28 <= 0
 edge_flow_constr_nic_out_21_nic_in_29: flow_nic_out_21_nic_in_29
   - 61035.15625 switch_nic_out_21_nic_in_29 <= 0
 edge_flow_constr_nic_out_21_nic_in_30: flow_nic_out_21_nic_in_30
   - 61035.15625 switch_nic_out_21_nic_in_30 <= 0
 edge_flow_constr_nic_out_21_nic_in_31: flow_nic_out_21_nic_in_31
   - 61035.15625 switch_nic_out_21_nic_in_31 <= 0
 edge_flow_constr_nic_in_21_21: flow_nic_in_21_21
   - 3.90625e+06 switch_nic_in_21_21 <= 0
 edge_flow_constr_21_nic_out_21: flow_21_nic_out_21
   - 3.90625e+06 switch_21_nic_out_21 <= 0
 edge_flow_constr_source_nic_in_22: flow_source_nic_in_22
   - 5e+08 switch_source_nic_in_22 <= 0
 edge_flow_constr_nic_out_22_sink: flow_nic_out_22_sink
   - 5e+08 switch_nic_out_22_sink <= 0
 edge_flow_constr_nic_out_22_nic_in_0: flow_nic_out_22_nic_in_0
   - 61035.15625 switch_nic_out_22_nic_in_0 <= 0
 edge_flow_constr_nic_out_22_nic_in_1: flow_nic_out_22_nic_in_1
   - 61035.15625 switch_nic_out_22_nic_in_1 <= 0
 edge_flow_constr_nic_out_22_nic_in_2: flow_nic_out_22_nic_in_2
   - 61035.15625 switch_nic_out_22_nic_in_2 <= 0
 edge_flow_constr_nic_out_22_nic_in_3: flow_nic_out_22_nic_in_3
   - 61035.15625 switch_nic_out_22_nic_in_3 <= 0
 edge_flow_constr_nic_out_22_nic_in_4: flow_nic_out_22_nic_in_4
   - 61035.15625 switch_nic_out_22_nic_in_4 <= 0
 edge_flow_constr_nic_out_22_nic_in_5: flow_nic_out_22_nic_in_5
   - 61035.15625 switch_nic_out_22_nic_in_5 <= 0
 edge_flow_constr_nic_out_22_nic_in_6: flow_nic_out_22_nic_in_6
   - 61035.15625 switch_nic_out_22_nic_in_6 <= 0
 edge_flow_constr_nic_out_22_nic_in_7: flow_nic_out_22_nic_in_7
   - 61035.15625 switch_nic_out_22_nic_in_7 <= 0
 edge_flow_constr_nic_out_22_nic_in_8: flow_nic_out_22_nic_in_8
   - 61035.15625 switch_nic_out_22_nic_in_8 <= 0
 edge_flow_constr_nic_out_22_nic_in_9: flow_nic_out_22_nic_in_9
   - 61035.15625 switch_nic_out_22_nic_in_9 <= 0
 edge_flow_constr_nic_out_22_nic_in_10: flow_nic_out_22_nic_in_10
   - 61035.15625 switch_nic_out_22_nic_in_10 <= 0
 edge_flow_constr_nic_out_22_nic_in_11: flow_nic_out_22_nic_in_11
   - 61035.15625 switch_nic_out_22_nic_in_11 <= 0
 edge_flow_constr_nic_out_22_nic_in_12: flow_nic_out_22_nic_in_12
   - 61035.15625 switch_nic_out_22_nic_in_12 <= 0
 edge_flow_constr_nic_out_22_nic_in_13: flow_nic_out_22_nic_in_13
   - 61035.15625 switch_nic_out_22_nic_in_13 <= 0
 edge_flow_constr_nic_out_22_nic_in_14: flow_nic_out_22_nic_in_14
   - 61035.15625 switch_nic_out_22_nic_in_14 <= 0
 edge_flow_constr_nic_out_22_nic_in_15: flow_nic_out_22_nic_in_15
   - 61035.15625 switch_nic_out_22_nic_in_15 <= 0
 edge_flow_constr_nic_out_22_nic_in_16: flow_nic_out_22_nic_in_16
   - 61035.15625 switch_nic_out_22_nic_in_16 <= 0
 edge_flow_constr_nic_out_22_nic_in_17: flow_nic_out_22_nic_in_17
   - 61035.15625 switch_nic_out_22_nic_in_17 <= 0
 edge_flow_constr_nic_out_22_nic_in_18: flow_nic_out_22_nic_in_18
   - 61035.15625 switch_nic_out_22_nic_in_18 <= 0
 edge_flow_constr_nic_out_22_nic_in_19: flow_nic_out_22_nic_in_19
   - 61035.15625 switch_nic_out_22_nic_in_19 <= 0
 edge_flow_constr_nic_out_22_nic_in_20: flow_nic_out_22_nic_in_20
   - 61035.15625 switch_nic_out_22_nic_in_20 <= 0
 edge_flow_constr_nic_out_22_nic_in_21: flow_nic_out_22_nic_in_21
   - 61035.15625 switch_nic_out_22_nic_in_21 <= 0
 edge_flow_constr_nic_out_22_nic_in_23: flow_nic_out_22_nic_in_23
   - 61035.15625 switch_nic_out_22_nic_in_23 <= 0
 edge_flow_constr_nic_out_22_nic_in_24: flow_nic_out_22_nic_in_24
   - 61035.15625 switch_nic_out_22_nic_in_24 <= 0
 edge_flow_constr_nic_out_22_nic_in_25: flow_nic_out_22_nic_in_25
   - 61035.15625 switch_nic_out_22_nic_in_25 <= 0
 edge_flow_constr_nic_out_22_nic_in_26: flow_nic_out_22_nic_in_26
   - 61035.15625 switch_nic_out_22_nic_in_26 <= 0
 edge_flow_constr_nic_out_22_nic_in_27: flow_nic_out_22_nic_in_27
   - 61035.15625 switch_nic_out_22_nic_in_27 <= 0
 edge_flow_constr_nic_out_22_nic_in_28: flow_nic_out_22_nic_in_28
   - 61035.15625 switch_nic_out_22_nic_in_28 <= 0
 edge_flow_constr_nic_out_22_nic_in_29: flow_nic_out_22_nic_in_29
   - 61035.15625 switch_nic_out_22_nic_in_29 <= 0
 edge_flow_constr_nic_out_22_nic_in_30: flow_nic_out_22_nic_in_30
   - 61035.15625 switch_nic_out_22_nic_in_30 <= 0
 edge_flow_constr_nic_out_22_nic_in_31: flow_nic_out_22_nic_in_31
   - 61035.15625 switch_nic_out_22_nic_in_31 <= 0
 edge_flow_constr_nic_in_22_22: flow_nic_in_22_22
   - 3.90625e+06 switch_nic_in_22_22 <= 0
 edge_flow_constr_22_nic_out_22: flow_22_nic_out_22
   - 3.90625e+06 switch_22_nic_out_22 <= 0
 edge_flow_constr_source_nic_in_23: flow_source_nic_in_23
   - 5e+08 switch_source_nic_in_23 <= 0
 edge_flow_constr_nic_out_23_sink: flow_nic_out_23_sink
   - 5e+08 switch_nic_out_23_sink <= 0
 edge_flow_constr_nic_out_23_nic_in_0: flow_nic_out_23_nic_in_0
   - 61035.15625 switch_nic_out_23_nic_in_0 <= 0
 edge_flow_constr_nic_out_23_nic_in_1: flow_nic_out_23_nic_in_1
   - 61035.15625 switch_nic_out_23_nic_in_1 <= 0
 edge_flow_constr_nic_out_23_nic_in_2: flow_nic_out_23_nic_in_2
   - 61035.15625 switch_nic_out_23_nic_in_2 <= 0
 edge_flow_constr_nic_out_23_nic_in_3: flow_nic_out_23_nic_in_3
   - 61035.15625 switch_nic_out_23_nic_in_3 <= 0
 edge_flow_constr_nic_out_23_nic_in_4: flow_nic_out_23_nic_in_4
   - 61035.15625 switch_nic_out_23_nic_in_4 <= 0
 edge_flow_constr_nic_out_23_nic_in_5: flow_nic_out_23_nic_in_5
   - 61035.15625 switch_nic_out_23_nic_in_5 <= 0
 edge_flow_constr_nic_out_23_nic_in_6: flow_nic_out_23_nic_in_6
   - 61035.15625 switch_nic_out_23_nic_in_6 <= 0
 edge_flow_constr_nic_out_23_nic_in_7: flow_nic_out_23_nic_in_7
   - 61035.15625 switch_nic_out_23_nic_in_7 <= 0
 edge_flow_constr_nic_out_23_nic_in_8: flow_nic_out_23_nic_in_8
   - 61035.15625 switch_nic_out_23_nic_in_8 <= 0
 edge_flow_constr_nic_out_23_nic_in_9: flow_nic_out_23_nic_in_9
   - 61035.15625 switch_nic_out_23_nic_in_9 <= 0
 edge_flow_constr_nic_out_23_nic_in_10: flow_nic_out_23_nic_in_10
   - 61035.15625 switch_nic_out_23_nic_in_10 <= 0
 edge_flow_constr_nic_out_23_nic_in_11: flow_nic_out_23_nic_in_11
   - 61035.15625 switch_nic_out_23_nic_in_11 <= 0
 edge_flow_constr_nic_out_23_nic_in_12: flow_nic_out_23_nic_in_12
   - 61035.15625 switch_nic_out_23_nic_in_12 <= 0
 edge_flow_constr_nic_out_23_nic_in_13: flow_nic_out_23_nic_in_13
   - 61035.15625 switch_nic_out_23_nic_in_13 <= 0
 edge_flow_constr_nic_out_23_nic_in_14: flow_nic_out_23_nic_in_14
   - 61035.15625 switch_nic_out_23_nic_in_14 <= 0
 edge_flow_constr_nic_out_23_nic_in_15: flow_nic_out_23_nic_in_15
   - 61035.15625 switch_nic_out_23_nic_in_15 <= 0
 edge_flow_constr_nic_out_23_nic_in_16: flow_nic_out_23_nic_in_16
   - 61035.15625 switch_nic_out_23_nic_in_16 <= 0
 edge_flow_constr_nic_out_23_nic_in_17: flow_nic_out_23_nic_in_17
   - 61035.15625 switch_nic_out_23_nic_in_17 <= 0
 edge_flow_constr_nic_out_23_nic_in_18: flow_nic_out_23_nic_in_18
   - 61035.15625 switch_nic_out_23_nic_in_18 <= 0
 edge_flow_constr_nic_out_23_nic_in_19: flow_nic_out_23_nic_in_19
   - 61035.15625 switch_nic_out_23_nic_in_19 <= 0
 edge_flow_constr_nic_out_23_nic_in_20: flow_nic_out_23_nic_in_20
   - 61035.15625 switch_nic_out_23_nic_in_20 <= 0
 edge_flow_constr_nic_out_23_nic_in_21: flow_nic_out_23_nic_in_21
   - 61035.15625 switch_nic_out_23_nic_in_21 <= 0
 edge_flow_constr_nic_out_23_nic_in_22: flow_nic_out_23_nic_in_22
   - 61035.15625 switch_nic_out_23_nic_in_22 <= 0
 edge_flow_constr_nic_out_23_nic_in_24: flow_nic_out_23_nic_in_24
   - 61035.15625 switch_nic_out_23_nic_in_24 <= 0
 edge_flow_constr_nic_out_23_nic_in_25: flow_nic_out_23_nic_in_25
   - 61035.15625 switch_nic_out_23_nic_in_25 <= 0
 edge_flow_constr_nic_out_23_nic_in_26: flow_nic_out_23_nic_in_26
   - 61035.15625 switch_nic_out_23_nic_in_26 <= 0
 edge_flow_constr_nic_out_23_nic_in_27: flow_nic_out_23_nic_in_27
   - 61035.15625 switch_nic_out_23_nic_in_27 <= 0
 edge_flow_constr_nic_out_23_nic_in_28: flow_nic_out_23_nic_in_28
   - 61035.15625 switch_nic_out_23_nic_in_28 <= 0
 edge_flow_constr_nic_out_23_nic_in_29: flow_nic_out_23_nic_in_29
   - 61035.15625 switch_nic_out_23_nic_in_29 <= 0
 edge_flow_constr_nic_out_23_nic_in_30: flow_nic_out_23_nic_in_30
   - 61035.15625 switch_nic_out_23_nic_in_30 <= 0
 edge_flow_constr_nic_out_23_nic_in_31: flow_nic_out_23_nic_in_31
   - 61035.15625 switch_nic_out_23_nic_in_31 <= 0
 edge_flow_constr_nic_in_23_23: flow_nic_in_23_23
   - 3.90625e+06 switch_nic_in_23_23 <= 0
 edge_flow_constr_23_nic_out_23: flow_23_nic_out_23
   - 3.90625e+06 switch_23_nic_out_23 <= 0
 edge_flow_constr_source_nic_in_24: flow_source_nic_in_24
   - 5e+08 switch_source_nic_in_24 <= 0
 edge_flow_constr_nic_out_24_sink: flow_nic_out_24_sink
   - 5e+08 switch_nic_out_24_sink <= 0
 edge_flow_constr_nic_out_24_nic_in_0: flow_nic_out_24_nic_in_0
   - 61035.15625 switch_nic_out_24_nic_in_0 <= 0
 edge_flow_constr_nic_out_24_nic_in_1: flow_nic_out_24_nic_in_1
   - 61035.15625 switch_nic_out_24_nic_in_1 <= 0
 edge_flow_constr_nic_out_24_nic_in_2: flow_nic_out_24_nic_in_2
   - 61035.15625 switch_nic_out_24_nic_in_2 <= 0
 edge_flow_constr_nic_out_24_nic_in_3: flow_nic_out_24_nic_in_3
   - 61035.15625 switch_nic_out_24_nic_in_3 <= 0
 edge_flow_constr_nic_out_24_nic_in_4: flow_nic_out_24_nic_in_4
   - 61035.15625 switch_nic_out_24_nic_in_4 <= 0
 edge_flow_constr_nic_out_24_nic_in_5: flow_nic_out_24_nic_in_5
   - 61035.15625 switch_nic_out_24_nic_in_5 <= 0
 edge_flow_constr_nic_out_24_nic_in_6: flow_nic_out_24_nic_in_6
   - 61035.15625 switch_nic_out_24_nic_in_6 <= 0
 edge_flow_constr_nic_out_24_nic_in_7: flow_nic_out_24_nic_in_7
   - 61035.15625 switch_nic_out_24_nic_in_7 <= 0
 edge_flow_constr_nic_out_24_nic_in_8: flow_nic_out_24_nic_in_8
   - 61035.15625 switch_nic_out_24_nic_in_8 <= 0
 edge_flow_constr_nic_out_24_nic_in_9: flow_nic_out_24_nic_in_9
   - 61035.15625 switch_nic_out_24_nic_in_9 <= 0
 edge_flow_constr_nic_out_24_nic_in_10: flow_nic_out_24_nic_in_10
   - 61035.15625 switch_nic_out_24_nic_in_10 <= 0
 edge_flow_constr_nic_out_24_nic_in_11: flow_nic_out_24_nic_in_11
   - 61035.15625 switch_nic_out_24_nic_in_11 <= 0
 edge_flow_constr_nic_out_24_nic_in_12: flow_nic_out_24_nic_in_12
   - 61035.15625 switch_nic_out_24_nic_in_12 <= 0
 edge_flow_constr_nic_out_24_nic_in_13: flow_nic_out_24_nic_in_13
   - 61035.15625 switch_nic_out_24_nic_in_13 <= 0
 edge_flow_constr_nic_out_24_nic_in_14: flow_nic_out_24_nic_in_14
   - 61035.15625 switch_nic_out_24_nic_in_14 <= 0
 edge_flow_constr_nic_out_24_nic_in_15: flow_nic_out_24_nic_in_15
   - 61035.15625 switch_nic_out_24_nic_in_15 <= 0
 edge_flow_constr_nic_out_24_nic_in_16: flow_nic_out_24_nic_in_16
   - 61035.15625 switch_nic_out_24_nic_in_16 <= 0
 edge_flow_constr_nic_out_24_nic_in_17: flow_nic_out_24_nic_in_17
   - 61035.15625 switch_nic_out_24_nic_in_17 <= 0
 edge_flow_constr_nic_out_24_nic_in_18: flow_nic_out_24_nic_in_18
   - 61035.15625 switch_nic_out_24_nic_in_18 <= 0
 edge_flow_constr_nic_out_24_nic_in_19: flow_nic_out_24_nic_in_19
   - 61035.15625 switch_nic_out_24_nic_in_19 <= 0
 edge_flow_constr_nic_out_24_nic_in_20: flow_nic_out_24_nic_in_20
   - 61035.15625 switch_nic_out_24_nic_in_20 <= 0
 edge_flow_constr_nic_out_24_nic_in_21: flow_nic_out_24_nic_in_21
   - 61035.15625 switch_nic_out_24_nic_in_21 <= 0
 edge_flow_constr_nic_out_24_nic_in_22: flow_nic_out_24_nic_in_22
   - 61035.15625 switch_nic_out_24_nic_in_22 <= 0
 edge_flow_constr_nic_out_24_nic_in_23: flow_nic_out_24_nic_in_23
   - 61035.15625 switch_nic_out_24_nic_in_23 <= 0
 edge_flow_constr_nic_out_24_nic_in_25: flow_nic_out_24_nic_in_25
   - 61035.15625 switch_nic_out_24_nic_in_25 <= 0
 edge_flow_constr_nic_out_24_nic_in_26: flow_nic_out_24_nic_in_26
   - 61035.15625 switch_nic_out_24_nic_in_26 <= 0
 edge_flow_constr_nic_out_24_nic_in_27: flow_nic_out_24_nic_in_27
   - 61035.15625 switch_nic_out_24_nic_in_27 <= 0
 edge_flow_constr_nic_out_24_nic_in_28: flow_nic_out_24_nic_in_28
   - 61035.15625 switch_nic_out_24_nic_in_28 <= 0
 edge_flow_constr_nic_out_24_nic_in_29: flow_nic_out_24_nic_in_29
   - 61035.15625 switch_nic_out_24_nic_in_29 <= 0
 edge_flow_constr_nic_out_24_nic_in_30: flow_nic_out_24_nic_in_30
   - 61035.15625 switch_nic_out_24_nic_in_30 <= 0
 edge_flow_constr_nic_out_24_nic_in_31: flow_nic_out_24_nic_in_31
   - 61035.15625 switch_nic_out_24_nic_in_31 <= 0
 edge_flow_constr_nic_in_24_24: flow_nic_in_24_24
   - 3.90625e+06 switch_nic_in_24_24 <= 0
 edge_flow_constr_24_nic_out_24: flow_24_nic_out_24
   - 3.90625e+06 switch_24_nic_out_24 <= 0
 edge_flow_constr_source_nic_in_25: flow_source_nic_in_25
   - 5e+08 switch_source_nic_in_25 <= 0
 edge_flow_constr_nic_out_25_sink: flow_nic_out_25_sink
   - 5e+08 switch_nic_out_25_sink <= 0
 edge_flow_constr_nic_out_25_nic_in_0: flow_nic_out_25_nic_in_0
   - 61035.15625 switch_nic_out_25_nic_in_0 <= 0
 edge_flow_constr_nic_out_25_nic_in_1: flow_nic_out_25_nic_in_1
   - 61035.15625 switch_nic_out_25_nic_in_1 <= 0
 edge_flow_constr_nic_out_25_nic_in_2: flow_nic_out_25_nic_in_2
   - 61035.15625 switch_nic_out_25_nic_in_2 <= 0
 edge_flow_constr_nic_out_25_nic_in_3: flow_nic_out_25_nic_in_3
   - 61035.15625 switch_nic_out_25_nic_in_3 <= 0
 edge_flow_constr_nic_out_25_nic_in_4: flow_nic_out_25_nic_in_4
   - 61035.15625 switch_nic_out_25_nic_in_4 <= 0
 edge_flow_constr_nic_out_25_nic_in_5: flow_nic_out_25_nic_in_5
   - 61035.15625 switch_nic_out_25_nic_in_5 <= 0
 edge_flow_constr_nic_out_25_nic_in_6: flow_nic_out_25_nic_in_6
   - 61035.15625 switch_nic_out_25_nic_in_6 <= 0
 edge_flow_constr_nic_out_25_nic_in_7: flow_nic_out_25_nic_in_7
   - 61035.15625 switch_nic_out_25_nic_in_7 <= 0
 edge_flow_constr_nic_out_25_nic_in_8: flow_nic_out_25_nic_in_8
   - 61035.15625 switch_nic_out_25_nic_in_8 <= 0
 edge_flow_constr_nic_out_25_nic_in_9: flow_nic_out_25_nic_in_9
   - 61035.15625 switch_nic_out_25_nic_in_9 <= 0
 edge_flow_constr_nic_out_25_nic_in_10: flow_nic_out_25_nic_in_10
   - 61035.15625 switch_nic_out_25_nic_in_10 <= 0
 edge_flow_constr_nic_out_25_nic_in_11: flow_nic_out_25_nic_in_11
   - 61035.15625 switch_nic_out_25_nic_in_11 <= 0
 edge_flow_constr_nic_out_25_nic_in_12: flow_nic_out_25_nic_in_12
   - 61035.15625 switch_nic_out_25_nic_in_12 <= 0
 edge_flow_constr_nic_out_25_nic_in_13: flow_nic_out_25_nic_in_13
   - 61035.15625 switch_nic_out_25_nic_in_13 <= 0
 edge_flow_constr_nic_out_25_nic_in_14: flow_nic_out_25_nic_in_14
   - 61035.15625 switch_nic_out_25_nic_in_14 <= 0
 edge_flow_constr_nic_out_25_nic_in_15: flow_nic_out_25_nic_in_15
   - 61035.15625 switch_nic_out_25_nic_in_15 <= 0
 edge_flow_constr_nic_out_25_nic_in_16: flow_nic_out_25_nic_in_16
   - 61035.15625 switch_nic_out_25_nic_in_16 <= 0
 edge_flow_constr_nic_out_25_nic_in_17: flow_nic_out_25_nic_in_17
   - 61035.15625 switch_nic_out_25_nic_in_17 <= 0
 edge_flow_constr_nic_out_25_nic_in_18: flow_nic_out_25_nic_in_18
   - 61035.15625 switch_nic_out_25_nic_in_18 <= 0
 edge_flow_constr_nic_out_25_nic_in_19: flow_nic_out_25_nic_in_19
   - 61035.15625 switch_nic_out_25_nic_in_19 <= 0
 edge_flow_constr_nic_out_25_nic_in_20: flow_nic_out_25_nic_in_20
   - 61035.15625 switch_nic_out_25_nic_in_20 <= 0
 edge_flow_constr_nic_out_25_nic_in_21: flow_nic_out_25_nic_in_21
   - 61035.15625 switch_nic_out_25_nic_in_21 <= 0
 edge_flow_constr_nic_out_25_nic_in_22: flow_nic_out_25_nic_in_22
   - 61035.15625 switch_nic_out_25_nic_in_22 <= 0
 edge_flow_constr_nic_out_25_nic_in_23: flow_nic_out_25_nic_in_23
   - 61035.15625 switch_nic_out_25_nic_in_23 <= 0
 edge_flow_constr_nic_out_25_nic_in_24: flow_nic_out_25_nic_in_24
   - 61035.15625 switch_nic_out_25_nic_in_24 <= 0
 edge_flow_constr_nic_out_25_nic_in_26: flow_nic_out_25_nic_in_26
   - 61035.15625 switch_nic_out_25_nic_in_26 <= 0
 edge_flow_constr_nic_out_25_nic_in_27: flow_nic_out_25_nic_in_27
   - 61035.15625 switch_nic_out_25_nic_in_27 <= 0
 edge_flow_constr_nic_out_25_nic_in_28: flow_nic_out_25_nic_in_28
   - 61035.15625 switch_nic_out_25_nic_in_28 <= 0
 edge_flow_constr_nic_out_25_nic_in_29: flow_nic_out_25_nic_in_29
   - 61035.15625 switch_nic_out_25_nic_in_29 <= 0
 edge_flow_constr_nic_out_25_nic_in_30: flow_nic_out_25_nic_in_30
   - 61035.15625 switch_nic_out_25_nic_in_30 <= 0
 edge_flow_constr_nic_out_25_nic_in_31: flow_nic_out_25_nic_in_31
   - 61035.15625 switch_nic_out_25_nic_in_31 <= 0
 edge_flow_constr_nic_in_25_25: flow_nic_in_25_25
   - 3.90625e+06 switch_nic_in_25_25 <= 0
 edge_flow_constr_25_nic_out_25: flow_25_nic_out_25
   - 3.90625e+06 switch_25_nic_out_25 <= 0
 edge_flow_constr_source_nic_in_26: flow_source_nic_in_26
   - 5e+08 switch_source_nic_in_26 <= 0
 edge_flow_constr_nic_out_26_sink: flow_nic_out_26_sink
   - 5e+08 switch_nic_out_26_sink <= 0
 edge_flow_constr_nic_out_26_nic_in_0: flow_nic_out_26_nic_in_0
   - 61035.15625 switch_nic_out_26_nic_in_0 <= 0
 edge_flow_constr_nic_out_26_nic_in_1: flow_nic_out_26_nic_in_1
   - 61035.15625 switch_nic_out_26_nic_in_1 <= 0
 edge_flow_constr_nic_out_26_nic_in_2: flow_nic_out_26_nic_in_2
   - 61035.15625 switch_nic_out_26_nic_in_2 <= 0
 edge_flow_constr_nic_out_26_nic_in_3: flow_nic_out_26_nic_in_3
   - 61035.15625 switch_nic_out_26_nic_in_3 <= 0
 edge_flow_constr_nic_out_26_nic_in_4: flow_nic_out_26_nic_in_4
   - 61035.15625 switch_nic_out_26_nic_in_4 <= 0
 edge_flow_constr_nic_out_26_nic_in_5: flow_nic_out_26_nic_in_5
   - 61035.15625 switch_nic_out_26_nic_in_5 <= 0
 edge_flow_constr_nic_out_26_nic_in_6: flow_nic_out_26_nic_in_6
   - 61035.15625 switch_nic_out_26_nic_in_6 <= 0
 edge_flow_constr_nic_out_26_nic_in_7: flow_nic_out_26_nic_in_7
   - 61035.15625 switch_nic_out_26_nic_in_7 <= 0
 edge_flow_constr_nic_out_26_nic_in_8: flow_nic_out_26_nic_in_8
   - 61035.15625 switch_nic_out_26_nic_in_8 <= 0
 edge_flow_constr_nic_out_26_nic_in_9: flow_nic_out_26_nic_in_9
   - 61035.15625 switch_nic_out_26_nic_in_9 <= 0
 edge_flow_constr_nic_out_26_nic_in_10: flow_nic_out_26_nic_in_10
   - 61035.15625 switch_nic_out_26_nic_in_10 <= 0
 edge_flow_constr_nic_out_26_nic_in_11: flow_nic_out_26_nic_in_11
   - 61035.15625 switch_nic_out_26_nic_in_11 <= 0
 edge_flow_constr_nic_out_26_nic_in_12: flow_nic_out_26_nic_in_12
   - 61035.15625 switch_nic_out_26_nic_in_12 <= 0
 edge_flow_constr_nic_out_26_nic_in_13: flow_nic_out_26_nic_in_13
   - 61035.15625 switch_nic_out_26_nic_in_13 <= 0
 edge_flow_constr_nic_out_26_nic_in_14: flow_nic_out_26_nic_in_14
   - 61035.15625 switch_nic_out_26_nic_in_14 <= 0
 edge_flow_constr_nic_out_26_nic_in_15: flow_nic_out_26_nic_in_15
   - 61035.15625 switch_nic_out_26_nic_in_15 <= 0
 edge_flow_constr_nic_out_26_nic_in_16: flow_nic_out_26_nic_in_16
   - 61035.15625 switch_nic_out_26_nic_in_16 <= 0
 edge_flow_constr_nic_out_26_nic_in_17: flow_nic_out_26_nic_in_17
   - 61035.15625 switch_nic_out_26_nic_in_17 <= 0
 edge_flow_constr_nic_out_26_nic_in_18: flow_nic_out_26_nic_in_18
   - 61035.15625 switch_nic_out_26_nic_in_18 <= 0
 edge_flow_constr_nic_out_26_nic_in_19: flow_nic_out_26_nic_in_19
   - 61035.15625 switch_nic_out_26_nic_in_19 <= 0
 edge_flow_constr_nic_out_26_nic_in_20: flow_nic_out_26_nic_in_20
   - 61035.15625 switch_nic_out_26_nic_in_20 <= 0
 edge_flow_constr_nic_out_26_nic_in_21: flow_nic_out_26_nic_in_21
   - 61035.15625 switch_nic_out_26_nic_in_21 <= 0
 edge_flow_constr_nic_out_26_nic_in_22: flow_nic_out_26_nic_in_22
   - 61035.15625 switch_nic_out_26_nic_in_22 <= 0
 edge_flow_constr_nic_out_26_nic_in_23: flow_nic_out_26_nic_in_23
   - 61035.15625 switch_nic_out_26_nic_in_23 <= 0
 edge_flow_constr_nic_out_26_nic_in_24: flow_nic_out_26_nic_in_24
   - 61035.15625 switch_nic_out_26_nic_in_24 <= 0
 edge_flow_constr_nic_out_26_nic_in_25: flow_nic_out_26_nic_in_25
   - 61035.15625 switch_nic_out_26_nic_in_25 <= 0
 edge_flow_constr_nic_out_26_nic_in_27: flow_nic_out_26_nic_in_27
   - 61035.15625 switch_nic_out_26_nic_in_27 <= 0
 edge_flow_constr_nic_out_26_nic_in_28: flow_nic_out_26_nic_in_28
   - 61035.15625 switch_nic_out_26_nic_in_28 <= 0
 edge_flow_constr_nic_out_26_nic_in_29: flow_nic_out_26_nic_in_29
   - 61035.15625 switch_nic_out_26_nic_in_29 <= 0
 edge_flow_constr_nic_out_26_nic_in_30: flow_nic_out_26_nic_in_30
   - 61035.15625 switch_nic_out_26_nic_in_30 <= 0
 edge_flow_constr_nic_out_26_nic_in_31: flow_nic_out_26_nic_in_31
   - 61035.15625 switch_nic_out_26_nic_in_31 <= 0
 edge_flow_constr_nic_in_26_26: flow_nic_in_26_26
   - 3.90625e+06 switch_nic_in_26_26 <= 0
 edge_flow_constr_26_nic_out_26: flow_26_nic_out_26
   - 3.90625e+06 switch_26_nic_out_26 <= 0
 edge_flow_constr_source_nic_in_27: flow_source_nic_in_27
   - 5e+08 switch_source_nic_in_27 <= 0
 edge_flow_constr_nic_out_27_sink: flow_nic_out_27_sink
   - 5e+08 switch_nic_out_27_sink <= 0
 edge_flow_constr_nic_out_27_nic_in_0: flow_nic_out_27_nic_in_0
   - 61035.15625 switch_nic_out_27_nic_in_0 <= 0
 edge_flow_constr_nic_out_27_nic_in_1: flow_nic_out_27_nic_in_1
   - 61035.15625 switch_nic_out_27_nic_in_1 <= 0
 edge_flow_constr_nic_out_27_nic_in_2: flow_nic_out_27_nic_in_2
   - 61035.15625 switch_nic_out_27_nic_in_2 <= 0
 edge_flow_constr_nic_out_27_nic_in_3: flow_nic_out_27_nic_in_3
   - 61035.15625 switch_nic_out_27_nic_in_3 <= 0
 edge_flow_constr_nic_out_27_nic_in_4: flow_nic_out_27_nic_in_4
   - 61035.15625 switch_nic_out_27_nic_in_4 <= 0
 edge_flow_constr_nic_out_27_nic_in_5: flow_nic_out_27_nic_in_5
   - 61035.15625 switch_nic_out_27_nic_in_5 <= 0
 edge_flow_constr_nic_out_27_nic_in_6: flow_nic_out_27_nic_in_6
   - 61035.15625 switch_nic_out_27_nic_in_6 <= 0
 edge_flow_constr_nic_out_27_nic_in_7: flow_nic_out_27_nic_in_7
   - 61035.15625 switch_nic_out_27_nic_in_7 <= 0
 edge_flow_constr_nic_out_27_nic_in_8: flow_nic_out_27_nic_in_8
   - 61035.15625 switch_nic_out_27_nic_in_8 <= 0
 edge_flow_constr_nic_out_27_nic_in_9: flow_nic_out_27_nic_in_9
   - 61035.15625 switch_nic_out_27_nic_in_9 <= 0
 edge_flow_constr_nic_out_27_nic_in_10: flow_nic_out_27_nic_in_10
   - 61035.15625 switch_nic_out_27_nic_in_10 <= 0
 edge_flow_constr_nic_out_27_nic_in_11: flow_nic_out_27_nic_in_11
   - 61035.15625 switch_nic_out_27_nic_in_11 <= 0
 edge_flow_constr_nic_out_27_nic_in_12: flow_nic_out_27_nic_in_12
   - 61035.15625 switch_nic_out_27_nic_in_12 <= 0
 edge_flow_constr_nic_out_27_nic_in_13: flow_nic_out_27_nic_in_13
   - 61035.15625 switch_nic_out_27_nic_in_13 <= 0
 edge_flow_constr_nic_out_27_nic_in_14: flow_nic_out_27_nic_in_14
   - 61035.15625 switch_nic_out_27_nic_in_14 <= 0
 edge_flow_constr_nic_out_27_nic_in_15: flow_nic_out_27_nic_in_15
   - 61035.15625 switch_nic_out_27_nic_in_15 <= 0
 edge_flow_constr_nic_out_27_nic_in_16: flow_nic_out_27_nic_in_16
   - 61035.15625 switch_nic_out_27_nic_in_16 <= 0
 edge_flow_constr_nic_out_27_nic_in_17: flow_nic_out_27_nic_in_17
   - 61035.15625 switch_nic_out_27_nic_in_17 <= 0
 edge_flow_constr_nic_out_27_nic_in_18: flow_nic_out_27_nic_in_18
   - 61035.15625 switch_nic_out_27_nic_in_18 <= 0
 edge_flow_constr_nic_out_27_nic_in_19: flow_nic_out_27_nic_in_19
   - 61035.15625 switch_nic_out_27_nic_in_19 <= 0
 edge_flow_constr_nic_out_27_nic_in_20: flow_nic_out_27_nic_in_20
   - 61035.15625 switch_nic_out_27_nic_in_20 <= 0
 edge_flow_constr_nic_out_27_nic_in_21: flow_nic_out_27_nic_in_21
   - 61035.15625 switch_nic_out_27_nic_in_21 <= 0
 edge_flow_constr_nic_out_27_nic_in_22: flow_nic_out_27_nic_in_22
   - 61035.15625 switch_nic_out_27_nic_in_22 <= 0
 edge_flow_constr_nic_out_27_nic_in_23: flow_nic_out_27_nic_in_23
   - 61035.15625 switch_nic_out_27_nic_in_23 <= 0
 edge_flow_constr_nic_out_27_nic_in_24: flow_nic_out_27_nic_in_24
   - 61035.15625 switch_nic_out_27_nic_in_24 <= 0
 edge_flow_constr_nic_out_27_nic_in_25: flow_nic_out_27_nic_in_25
   - 61035.15625 switch_nic_out_27_nic_in_25 <= 0
 edge_flow_constr_nic_out_27_nic_in_26: flow_nic_out_27_nic_in_26
   - 61035.15625 switch_nic_out_27_nic_in_26 <= 0
 edge_flow_constr_nic_out_27_nic_in_28: flow_nic_out_27_nic_in_28
   - 61035.15625 switch_nic_out_27_nic_in_28 <= 0
 edge_flow_constr_nic_out_27_nic_in_29: flow_nic_out_27_nic_in_29
   - 61035.15625 switch_nic_out_27_nic_in_29 <= 0
 edge_flow_constr_nic_out_27_nic_in_30: flow_nic_out_27_nic_in_30
   - 61035.15625 switch_nic_out_27_nic_in_30 <= 0
 edge_flow_constr_nic_out_27_nic_in_31: flow_nic_out_27_nic_in_31
   - 61035.15625 switch_nic_out_27_nic_in_31 <= 0
 edge_flow_constr_nic_in_27_27: flow_nic_in_27_27
   - 3.90625e+06 switch_nic_in_27_27 <= 0
 edge_flow_constr_27_nic_out_27: flow_27_nic_out_27
   - 3.90625e+06 switch_27_nic_out_27 <= 0
 edge_flow_constr_source_nic_in_28: flow_source_nic_in_28
   - 5e+08 switch_source_nic_in_28 <= 0
 edge_flow_constr_nic_out_28_sink: flow_nic_out_28_sink
   - 5e+08 switch_nic_out_28_sink <= 0
 edge_flow_constr_nic_out_28_nic_in_0: flow_nic_out_28_nic_in_0
   - 61035.15625 switch_nic_out_28_nic_in_0 <= 0
 edge_flow_constr_nic_out_28_nic_in_1: flow_nic_out_28_nic_in_1
   - 61035.15625 switch_nic_out_28_nic_in_1 <= 0
 edge_flow_constr_nic_out_28_nic_in_2: flow_nic_out_28_nic_in_2
   - 61035.15625 switch_nic_out_28_nic_in_2 <= 0
 edge_flow_constr_nic_out_28_nic_in_3: flow_nic_out_28_nic_in_3
   - 61035.15625 switch_nic_out_28_nic_in_3 <= 0
 edge_flow_constr_nic_out_28_nic_in_4: flow_nic_out_28_nic_in_4
   - 61035.15625 switch_nic_out_28_nic_in_4 <= 0
 edge_flow_constr_nic_out_28_nic_in_5: flow_nic_out_28_nic_in_5
   - 61035.15625 switch_nic_out_28_nic_in_5 <= 0
 edge_flow_constr_nic_out_28_nic_in_6: flow_nic_out_28_nic_in_6
   - 61035.15625 switch_nic_out_28_nic_in_6 <= 0
 edge_flow_constr_nic_out_28_nic_in_7: flow_nic_out_28_nic_in_7
   - 61035.15625 switch_nic_out_28_nic_in_7 <= 0
 edge_flow_constr_nic_out_28_nic_in_8: flow_nic_out_28_nic_in_8
   - 61035.15625 switch_nic_out_28_nic_in_8 <= 0
 edge_flow_constr_nic_out_28_nic_in_9: flow_nic_out_28_nic_in_9
   - 61035.15625 switch_nic_out_28_nic_in_9 <= 0
 edge_flow_constr_nic_out_28_nic_in_10: flow_nic_out_28_nic_in_10
   - 61035.15625 switch_nic_out_28_nic_in_10 <= 0
 edge_flow_constr_nic_out_28_nic_in_11: flow_nic_out_28_nic_in_11
   - 61035.15625 switch_nic_out_28_nic_in_11 <= 0
 edge_flow_constr_nic_out_28_nic_in_12: flow_nic_out_28_nic_in_12
   - 61035.15625 switch_nic_out_28_nic_in_12 <= 0
 edge_flow_constr_nic_out_28_nic_in_13: flow_nic_out_28_nic_in_13
   - 61035.15625 switch_nic_out_28_nic_in_13 <= 0
 edge_flow_constr_nic_out_28_nic_in_14: flow_nic_out_28_nic_in_14
   - 61035.15625 switch_nic_out_28_nic_in_14 <= 0
 edge_flow_constr_nic_out_28_nic_in_15: flow_nic_out_28_nic_in_15
   - 61035.15625 switch_nic_out_28_nic_in_15 <= 0
 edge_flow_constr_nic_out_28_nic_in_16: flow_nic_out_28_nic_in_16
   - 61035.15625 switch_nic_out_28_nic_in_16 <= 0
 edge_flow_constr_nic_out_28_nic_in_17: flow_nic_out_28_nic_in_17
   - 61035.15625 switch_nic_out_28_nic_in_17 <= 0
 edge_flow_constr_nic_out_28_nic_in_18: flow_nic_out_28_nic_in_18
   - 61035.15625 switch_nic_out_28_nic_in_18 <= 0
 edge_flow_constr_nic_out_28_nic_in_19: flow_nic_out_28_nic_in_19
   - 61035.15625 switch_nic_out_28_nic_in_19 <= 0
 edge_flow_constr_nic_out_28_nic_in_20: flow_nic_out_28_nic_in_20
   - 61035.15625 switch_nic_out_28_nic_in_20 <= 0
 edge_flow_constr_nic_out_28_nic_in_21: flow_nic_out_28_nic_in_21
   - 61035.15625 switch_nic_out_28_nic_in_21 <= 0
 edge_flow_constr_nic_out_28_nic_in_22: flow_nic_out_28_nic_in_22
   - 61035.15625 switch_nic_out_28_nic_in_22 <= 0
 edge_flow_constr_nic_out_28_nic_in_23: flow_nic_out_28_nic_in_23
   - 61035.15625 switch_nic_out_28_nic_in_23 <= 0
 edge_flow_constr_nic_out_28_nic_in_24: flow_nic_out_28_nic_in_24
   - 61035.15625 switch_nic_out_28_nic_in_24 <= 0
 edge_flow_constr_nic_out_28_nic_in_25: flow_nic_out_28_nic_in_25
   - 61035.15625 switch_nic_out_28_nic_in_25 <= 0
 edge_flow_constr_nic_out_28_nic_in_26: flow_nic_out_28_nic_in_26
   - 61035.15625 switch_nic_out_28_nic_in_26 <= 0
 edge_flow_constr_nic_out_28_nic_in_27: flow_nic_out_28_nic_in_27
   - 61035.15625 switch_nic_out_28_nic_in_27 <= 0
 edge_flow_constr_nic_out_28_nic_in_29: flow_nic_out_28_nic_in_29
   - 61035.15625 switch_nic_out_28_nic_in_29 <= 0
 edge_flow_constr_nic_out_28_nic_in_30: flow_nic_out_28_nic_in_30
   - 61035.15625 switch_nic_out_28_nic_in_30 <= 0
 edge_flow_constr_nic_out_28_nic_in_31: flow_nic_out_28_nic_in_31
   - 61035.15625 switch_nic_out_28_nic_in_31 <= 0
 edge_flow_constr_nic_in_28_28: flow_nic_in_28_28
   - 3.90625e+06 switch_nic_in_28_28 <= 0
 edge_flow_constr_28_nic_out_28: flow_28_nic_out_28
   - 3.90625e+06 switch_28_nic_out_28 <= 0
 edge_flow_constr_source_nic_in_29: flow_source_nic_in_29
   - 5e+08 switch_source_nic_in_29 <= 0
 edge_flow_constr_nic_out_29_sink: flow_nic_out_29_sink
   - 5e+08 switch_nic_out_29_sink <= 0
 edge_flow_constr_nic_out_29_nic_in_0: flow_nic_out_29_nic_in_0
   - 61035.15625 switch_nic_out_29_nic_in_0 <= 0
 edge_flow_constr_nic_out_29_nic_in_1: flow_nic_out_29_nic_in_1
   - 61035.15625 switch_nic_out_29_nic_in_1 <= 0
 edge_flow_constr_nic_out_29_nic_in_2: flow_nic_out_29_nic_in_2
   - 61035.15625 switch_nic_out_29_nic_in_2 <= 0
 edge_flow_constr_nic_out_29_nic_in_3: flow_nic_out_29_nic_in_3
   - 61035.15625 switch_nic_out_29_nic_in_3 <= 0
 edge_flow_constr_nic_out_29_nic_in_4: flow_nic_out_29_nic_in_4
   - 61035.15625 switch_nic_out_29_nic_in_4 <= 0
 edge_flow_constr_nic_out_29_nic_in_5: flow_nic_out_29_nic_in_5
   - 61035.15625 switch_nic_out_29_nic_in_5 <= 0
 edge_flow_constr_nic_out_29_nic_in_6: flow_nic_out_29_nic_in_6
   - 61035.15625 switch_nic_out_29_nic_in_6 <= 0
 edge_flow_constr_nic_out_29_nic_in_7: flow_nic_out_29_nic_in_7
   - 61035.15625 switch_nic_out_29_nic_in_7 <= 0
 edge_flow_constr_nic_out_29_nic_in_8: flow_nic_out_29_nic_in_8
   - 61035.15625 switch_nic_out_29_nic_in_8 <= 0
 edge_flow_constr_nic_out_29_nic_in_9: flow_nic_out_29_nic_in_9
   - 61035.15625 switch_nic_out_29_nic_in_9 <= 0
 edge_flow_constr_nic_out_29_nic_in_10: flow_nic_out_29_nic_in_10
   - 61035.15625 switch_nic_out_29_nic_in_10 <= 0
 edge_flow_constr_nic_out_29_nic_in_11: flow_nic_out_29_nic_in_11
   - 61035.15625 switch_nic_out_29_nic_in_11 <= 0
 edge_flow_constr_nic_out_29_nic_in_12: flow_nic_out_29_nic_in_12
   - 61035.15625 switch_nic_out_29_nic_in_12 <= 0
 edge_flow_constr_nic_out_29_nic_in_13: flow_nic_out_29_nic_in_13
   - 61035.15625 switch_nic_out_29_nic_in_13 <= 0
 edge_flow_constr_nic_out_29_nic_in_14: flow_nic_out_29_nic_in_14
   - 61035.15625 switch_nic_out_29_nic_in_14 <= 0
 edge_flow_constr_nic_out_29_nic_in_15: flow_nic_out_29_nic_in_15
   - 61035.15625 switch_nic_out_29_nic_in_15 <= 0
 edge_flow_constr_nic_out_29_nic_in_16: flow_nic_out_29_nic_in_16
   - 61035.15625 switch_nic_out_29_nic_in_16 <= 0
 edge_flow_constr_nic_out_29_nic_in_17: flow_nic_out_29_nic_in_17
   - 61035.15625 switch_nic_out_29_nic_in_17 <= 0
 edge_flow_constr_nic_out_29_nic_in_18: flow_nic_out_29_nic_in_18
   - 61035.15625 switch_nic_out_29_nic_in_18 <= 0
 edge_flow_constr_nic_out_29_nic_in_19: flow_nic_out_29_nic_in_19
   - 61035.15625 switch_nic_out_29_nic_in_19 <= 0
 edge_flow_constr_nic_out_29_nic_in_20: flow_nic_out_29_nic_in_20
   - 61035.15625 switch_nic_out_29_nic_in_20 <= 0
 edge_flow_constr_nic_out_29_nic_in_21: flow_nic_out_29_nic_in_21
   - 61035.15625 switch_nic_out_29_nic_in_21 <= 0
 edge_flow_constr_nic_out_29_nic_in_22: flow_nic_out_29_nic_in_22
   - 61035.15625 switch_nic_out_29_nic_in_22 <= 0
 edge_flow_constr_nic_out_29_nic_in_23: flow_nic_out_29_nic_in_23
   - 61035.15625 switch_nic_out_29_nic_in_23 <= 0
 edge_flow_constr_nic_out_29_nic_in_24: flow_nic_out_29_nic_in_24
   - 61035.15625 switch_nic_out_29_nic_in_24 <= 0
 edge_flow_constr_nic_out_29_nic_in_25: flow_nic_out_29_nic_in_25
   - 61035.15625 switch_nic_out_29_nic_in_25 <= 0
 edge_flow_constr_nic_out_29_nic_in_26: flow_nic_out_29_nic_in_26
   - 61035.15625 switch_nic_out_29_nic_in_26 <= 0
 edge_flow_constr_nic_out_29_nic_in_27: flow_nic_out_29_nic_in_27
   - 61035.15625 switch_nic_out_29_nic_in_27 <= 0
 edge_flow_constr_nic_out_29_nic_in_28: flow_nic_out_29_nic_in_28
   - 61035.15625 switch_nic_out_29_nic_in_28 <= 0
 edge_flow_constr_nic_out_29_nic_in_30: flow_nic_out_29_nic_in_30
   - 61035.15625 switch_nic_out_29_nic_in_30 <= 0
 edge_flow_constr_nic_out_29_nic_in_31: flow_nic_out_29_nic_in_31
   - 61035.15625 switch_nic_out_29_nic_in_31 <= 0
 edge_flow_constr_nic_in_29_29: flow_nic_in_29_29
   - 3.90625e+06 switch_nic_in_29_29 <= 0
 edge_flow_constr_29_nic_out_29: flow_29_nic_out_29
   - 3.90625e+06 switch_29_nic_out_29 <= 0
 edge_flow_constr_source_nic_in_30: flow_source_nic_in_30
   - 5e+08 switch_source_nic_in_30 <= 0
 edge_flow_constr_nic_out_30_sink: flow_nic_out_30_sink
   - 5e+08 switch_nic_out_30_sink <= 0
 edge_flow_constr_nic_out_30_nic_in_0: flow_nic_out_30_nic_in_0
   - 61035.15625 switch_nic_out_30_nic_in_0 <= 0
 edge_flow_constr_nic_out_30_nic_in_1: flow_nic_out_30_nic_in_1
   - 61035.15625 switch_nic_out_30_nic_in_1 <= 0
 edge_flow_constr_nic_out_30_nic_in_2: flow_nic_out_30_nic_in_2
   - 61035.15625 switch_nic_out_30_nic_in_2 <= 0
 edge_flow_constr_nic_out_30_nic_in_3: flow_nic_out_30_nic_in_3
   - 61035.15625 switch_nic_out_30_nic_in_3 <= 0
 edge_flow_constr_nic_out_30_nic_in_4: flow_nic_out_30_nic_in_4
   - 61035.15625 switch_nic_out_30_nic_in_4 <= 0
 edge_flow_constr_nic_out_30_nic_in_5: flow_nic_out_30_nic_in_5
   - 61035.15625 switch_nic_out_30_nic_in_5 <= 0
 edge_flow_constr_nic_out_30_nic_in_6: flow_nic_out_30_nic_in_6
   - 61035.15625 switch_nic_out_30_nic_in_6 <= 0
 edge_flow_constr_nic_out_30_nic_in_7: flow_nic_out_30_nic_in_7
   - 61035.15625 switch_nic_out_30_nic_in_7 <= 0
 edge_flow_constr_nic_out_30_nic_in_8: flow_nic_out_30_nic_in_8
   - 61035.15625 switch_nic_out_30_nic_in_8 <= 0
 edge_flow_constr_nic_out_30_nic_in_9: flow_nic_out_30_nic_in_9
   - 61035.15625 switch_nic_out_30_nic_in_9 <= 0
 edge_flow_constr_nic_out_30_nic_in_10: flow_nic_out_30_nic_in_10
   - 61035.15625 switch_nic_out_30_nic_in_10 <= 0
 edge_flow_constr_nic_out_30_nic_in_11: flow_nic_out_30_nic_in_11
   - 61035.15625 switch_nic_out_30_nic_in_11 <= 0
 edge_flow_constr_nic_out_30_nic_in_12: flow_nic_out_30_nic_in_12
   - 61035.15625 switch_nic_out_30_nic_in_12 <= 0
 edge_flow_constr_nic_out_30_nic_in_13: flow_nic_out_30_nic_in_13
   - 61035.15625 switch_nic_out_30_nic_in_13 <= 0
 edge_flow_constr_nic_out_30_nic_in_14: flow_nic_out_30_nic_in_14
   - 61035.15625 switch_nic_out_30_nic_in_14 <= 0
 edge_flow_constr_nic_out_30_nic_in_15: flow_nic_out_30_nic_in_15
   - 61035.15625 switch_nic_out_30_nic_in_15 <= 0
 edge_flow_constr_nic_out_30_nic_in_16: flow_nic_out_30_nic_in_16
   - 61035.15625 switch_nic_out_30_nic_in_16 <= 0
 edge_flow_constr_nic_out_30_nic_in_17: flow_nic_out_30_nic_in_17
   - 61035.15625 switch_nic_out_30_nic_in_17 <= 0
 edge_flow_constr_nic_out_30_nic_in_18: flow_nic_out_30_nic_in_18
   - 61035.15625 switch_nic_out_30_nic_in_18 <= 0
 edge_flow_constr_nic_out_30_nic_in_19: flow_nic_out_30_nic_in_19
   - 61035.15625 switch_nic_out_30_nic_in_19 <= 0
 edge_flow_constr_nic_out_30_nic_in_20: flow_nic_out_30_nic_in_20
   - 61035.15625 switch_nic_out_30_nic_in_20 <= 0
 edge_flow_constr_nic_out_30_nic_in_21: flow_nic_out_30_nic_in_21
   - 61035.15625 switch_nic_out_30_nic_in_21 <= 0
 edge_flow_constr_nic_out_30_nic_in_22: flow_nic_out_30_nic_in_22
   - 61035.15625 switch_nic_out_30_nic_in_22 <= 0
 edge_flow_constr_nic_out_30_nic_in_23: flow_nic_out_30_nic_in_23
   - 61035.15625 switch_nic_out_30_nic_in_23 <= 0
 edge_flow_constr_nic_out_30_nic_in_24: flow_nic_out_30_nic_in_24
   - 61035.15625 switch_nic_out_30_nic_in_24 <= 0
 edge_flow_constr_nic_out_30_nic_in_25: flow_nic_out_30_nic_in_25
   - 61035.15625 switch_nic_out_30_nic_in_25 <= 0
 edge_flow_constr_nic_out_30_nic_in_26: flow_nic_out_30_nic_in_26
   - 61035.15625 switch_nic_out_30_nic_in_26 <= 0
 edge_flow_constr_nic_out_30_nic_in_27: flow_nic_out_30_nic_in_27
   - 61035.15625 switch_nic_out_30_nic_in_27 <= 0
 edge_flow_constr_nic_out_30_nic_in_28: flow_nic_out_30_nic_in_28
   - 61035.15625 switch_nic_out_30_nic_in_28 <= 0
 edge_flow_constr_nic_out_30_nic_in_29: flow_nic_out_30_nic_in_29
   - 61035.15625 switch_nic_out_30_nic_in_29 <= 0
 edge_flow_constr_nic_out_30_nic_in_31: flow_nic_out_30_nic_in_31
   - 61035.15625 switch_nic_out_30_nic_in_31 <= 0
 edge_flow_constr_nic_in_30_30: flow_nic_in_30_30
   - 3.90625e+06 switch_nic_in_30_30 <= 0
 edge_flow_constr_30_nic_out_30: flow_30_nic_out_30
   - 3.90625e+06 switch_30_nic_out_30 <= 0
 edge_flow_constr_source_nic_in_31: flow_source_nic_in_31
   - 5e+08 switch_source_nic_in_31 <= 0
 edge_flow_constr_nic_out_31_sink: flow_nic_out_31_sink
   - 5e+08 switch_nic_out_31_sink <= 0
 edge_flow_constr_nic_out_31_nic_in_0: flow_nic_out_31_nic_in_0
   - 61035.15625 switch_nic_out_31_nic_in_0 <= 0
 edge_flow_constr_nic_out_31_nic_in_1: flow_nic_out_31_nic_in_1
   - 61035.15625 switch_nic_out_31_nic_in_1 <= 0
 edge_flow_constr_nic_out_31_nic_in_2: flow_nic_out_31_nic_in_2
   - 61035.15625 switch_nic_out_31_nic_in_2 <= 0
 edge_flow_constr_nic_out_31_nic_in_3: flow_nic_out_31_nic_in_3
   - 61035.15625 switch_nic_out_31_nic_in_3 <= 0
 edge_flow_constr_nic_out_31_nic_in_4: flow_nic_out_31_nic_in_4
   - 61035.15625 switch_nic_out_31_nic_in_4 <= 0
 edge_flow_constr_nic_out_31_nic_in_5: flow_nic_out_31_nic_in_5
   - 61035.15625 switch_nic_out_31_nic_in_5 <= 0
 edge_flow_constr_nic_out_31_nic_in_6: flow_nic_out_31_nic_in_6
   - 61035.15625 switch_nic_out_31_nic_in_6 <= 0
 edge_flow_constr_nic_out_31_nic_in_7: flow_nic_out_31_nic_in_7
   - 61035.15625 switch_nic_out_31_nic_in_7 <= 0
 edge_flow_constr_nic_out_31_nic_in_8: flow_nic_out_31_nic_in_8
   - 61035.15625 switch_nic_out_31_nic_in_8 <= 0
 edge_flow_constr_nic_out_31_nic_in_9: flow_nic_out_31_nic_in_9
   - 61035.15625 switch_nic_out_31_nic_in_9 <= 0
 edge_flow_constr_nic_out_31_nic_in_10: flow_nic_out_31_nic_in_10
   - 61035.15625 switch_nic_out_31_nic_in_10 <= 0
 edge_flow_constr_nic_out_31_nic_in_11: flow_nic_out_31_nic_in_11
   - 61035.15625 switch_nic_out_31_nic_in_11 <= 0
 edge_flow_constr_nic_out_31_nic_in_12: flow_nic_out_31_nic_in_12
   - 61035.15625 switch_nic_out_31_nic_in_12 <= 0
 edge_flow_constr_nic_out_31_nic_in_13: flow_nic_out_31_nic_in_13
   - 61035.15625 switch_nic_out_31_nic_in_13 <= 0
 edge_flow_constr_nic_out_31_nic_in_14: flow_nic_out_31_nic_in_14
   - 61035.15625 switch_nic_out_31_nic_in_14 <= 0
 edge_flow_constr_nic_out_31_nic_in_15: flow_nic_out_31_nic_in_15
   - 61035.15625 switch_nic_out_31_nic_in_15 <= 0
 edge_flow_constr_nic_out_31_nic_in_16: flow_nic_out_31_nic_in_16
   - 61035.15625 switch_nic_out_31_nic_in_16 <= 0
 edge_flow_constr_nic_out_31_nic_in_17: flow_nic_out_31_nic_in_17
   - 61035.15625 switch_nic_out_31_nic_in_17 <= 0
 edge_flow_constr_nic_out_31_nic_in_18: flow_nic_out_31_nic_in_18
   - 61035.15625 switch_nic_out_31_nic_in_18 <= 0
 edge_flow_constr_nic_out_31_nic_in_19: flow_nic_out_31_nic_in_19
   - 61035.15625 switch_nic_out_31_nic_in_19 <= 0
 edge_flow_constr_nic_out_31_nic_in_20: flow_nic_out_31_nic_in_20
   - 61035.15625 switch_nic_out_31_nic_in_20 <= 0
 edge_flow_constr_nic_out_31_nic_in_21: flow_nic_out_31_nic_in_21
   - 61035.15625 switch_nic_out_31_nic_in_21 <= 0
 edge_flow_constr_nic_out_31_nic_in_22: flow_nic_out_31_nic_in_22
   - 61035.15625 switch_nic_out_31_nic_in_22 <= 0
 edge_flow_constr_nic_out_31_nic_in_23: flow_nic_out_31_nic_in_23
   - 61035.15625 switch_nic_out_31_nic_in_23 <= 0
 edge_flow_constr_nic_out_31_nic_in_24: flow_nic_out_31_nic_in_24
   - 61035.15625 switch_nic_out_31_nic_in_24 <= 0
 edge_flow_constr_nic_out_31_nic_in_25: flow_nic_out_31_nic_in_25
   - 61035.15625 switch_nic_out_31_nic_in_25 <= 0
 edge_flow_constr_nic_out_31_nic_in_26: flow_nic_out_31_nic_in_26
   - 61035.15625 switch_nic_out_31_nic_in_26 <= 0
 edge_flow_constr_nic_out_31_nic_in_27: flow_nic_out_31_nic_in_27
   - 61035.15625 switch_nic_out_31_nic_in_27 <= 0
 edge_flow_constr_nic_out_31_nic_in_28: flow_nic_out_31_nic_in_28
   - 61035.15625 switch_nic_out_31_nic_in_28 <= 0
 edge_flow_constr_nic_out_31_nic_in_29: flow_nic_out_31_nic_in_29
   - 61035.15625 switch_nic_out_31_nic_in_29 <= 0
 edge_flow_constr_nic_out_31_nic_in_30: flow_nic_out_31_nic_in_30
   - 61035.15625 switch_nic_out_31_nic_in_30 <= 0
 edge_flow_constr_nic_in_31_31: flow_nic_in_31_31
   - 3.90625e+06 switch_nic_in_31_31 <= 0
 edge_flow_constr_31_nic_out_31: flow_31_nic_out_31
   - 3.90625e+06 switch_31_nic_out_31 <= 0
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
 force_group_off_constr_16: group_tp_active_16 = 0
 force_group_off_constr_17: group_tp_active_17 = 0
 force_group_off_constr_18: group_tp_active_18 = 0
 force_group_off_constr_19: group_tp_active_19 = 0
 force_group_off_constr_20: group_tp_active_20 = 0
 force_group_off_constr_21: group_tp_active_21 = 0
 force_group_off_constr_22: group_tp_active_22 = 0
 force_group_off_constr_23: group_tp_active_23 = 0
 force_group_off_constr_24: group_tp_active_24 = 0
 force_group_off_constr_25: group_tp_active_25 = 0
 force_group_off_constr_26: group_tp_active_26 = 0
 force_group_off_constr_27: group_tp_active_27 = 0
 force_group_off_constr_28: group_tp_active_28 = 0
 force_group_off_constr_29: group_tp_active_29 = 0
 force_group_off_constr_30: group_tp_active_30 = 0
 force_group_off_constr_31: group_tp_active_31 = 0
 force_min_throughput: flow_source_nic_in_0 + flow_source_nic_in_1
   + flow_source_nic_in_2 + flow_source_nic_in_3 + flow_source_nic_in_4
   + flow_source_nic_in_5 + flow_source_nic_in_6 + flow_source_nic_in_7
   + flow_source_nic_in_8 + flow_source_nic_in_9 + flow_source_nic_in_10
   + flow_source_nic_in_11 + flow_source_nic_in_12 + flow_source_nic_in_13
   + flow_source_nic_in_14 + flow_source_nic_in_15 + flow_source_nic_in_16
   + flow_source_nic_in_17 + flow_source_nic_in_18 + flow_source_nic_in_19
   + flow_source_nic_in_20 + flow_source_nic_in_21 + flow_source_nic_in_22
   + flow_source_nic_in_23 + flow_source_nic_in_24 + flow_source_nic_in_25
   + flow_source_nic_in_26 + flow_source_nic_in_27 + flow_source_nic_in_28
   + flow_source_nic_in_29 + flow_source_nic_in_30 + flow_source_nic_in_31
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
 inactive_start_invalid_10: is_active_10 = 0 -> start_10 = 80
 inactive_start_invalid_11: is_active_11 = 0 -> start_11 = 80
 inactive_start_invalid_12: is_active_12 = 0 -> start_12 = 80
 inactive_start_invalid_13: is_active_13 = 0 -> start_13 = 80
 inactive_start_invalid_14: is_active_14 = 0 -> start_14 = 80
 inactive_start_invalid_15: is_active_15 = 0 -> start_15 = 80
 inactive_start_invalid_16: is_active_16 = 0 -> start_16 = 80
 inactive_start_invalid_17: is_active_17 = 0 -> start_17 = 80
 inactive_start_invalid_18: is_active_18 = 0 -> start_18 = 80
 inactive_start_invalid_19: is_active_19 = 0 -> start_19 = 80
 inactive_start_invalid_20: is_active_20 = 0 -> start_20 = 80
 inactive_start_invalid_21: is_active_21 = 0 -> start_21 = 80
 inactive_start_invalid_22: is_active_22 = 0 -> start_22 = 80
 inactive_start_invalid_23: is_active_23 = 0 -> start_23 = 80
 inactive_start_invalid_24: is_active_24 = 0 -> start_24 = 80
 inactive_start_invalid_25: is_active_25 = 0 -> start_25 = 80
 inactive_start_invalid_26: is_active_26 = 0 -> start_26 = 80
 inactive_start_invalid_27: is_active_27 = 0 -> start_27 = 80
 inactive_start_invalid_28: is_active_28 = 0 -> start_28 = 80
 inactive_start_invalid_29: is_active_29 = 0 -> start_29 = 80
 inactive_start_invalid_30: is_active_30 = 0 -> start_30 = 80
 inactive_start_invalid_31: is_active_31 = 0 -> start_31 = 80
 node_throughput_constraint_0: group_tp_active_0 = 0 ->
   - 1371.428571428571 hold_0_1 - 685.7142857142857 hold_0_2
   - 457.1428571428571 hold_0_3 - 342.8571428571428 hold_0_4
   - 274.2857142857143 hold_0_5 - 228.5714285714286 hold_0_6
   + flow_nic_in_0_0 <= 0
 node_throughput_constraint_1: group_tp_active_1 = 0 ->
   - 1371.428571428571 hold_1_1 - 685.7142857142857 hold_1_2
   - 457.1428571428571 hold_1_3 - 342.8571428571428 hold_1_4
   - 274.2857142857143 hold_1_5 - 228.5714285714286 hold_1_6
   + flow_nic_in_1_1 <= 0
 node_throughput_constraint_2: group_tp_active_2 = 0 ->
   - 1371.428571428571 hold_2_1 - 685.7142857142857 hold_2_2
   - 457.1428571428571 hold_2_3 - 342.8571428571428 hold_2_4
   - 274.2857142857143 hold_2_5 - 228.5714285714286 hold_2_6
   + flow_nic_in_2_2 <= 0
 node_throughput_constraint_3: group_tp_active_3 = 0 ->
   - 1371.428571428571 hold_3_1 - 685.7142857142857 hold_3_2
   - 457.1428571428571 hold_3_3 - 342.8571428571428 hold_3_4
   - 274.2857142857143 hold_3_5 - 228.5714285714286 hold_3_6
   + flow_nic_in_3_3 <= 0
 node_throughput_constraint_4: group_tp_active_4 = 0 ->
   - 1371.428571428571 hold_4_1 - 685.7142857142857 hold_4_2
   - 457.1428571428571 hold_4_3 - 342.8571428571428 hold_4_4
   - 274.2857142857143 hold_4_5 - 228.5714285714286 hold_4_6
   + flow_nic_in_4_4 <= 0
 node_throughput_constraint_5: group_tp_active_5 = 0 ->
   - 1371.428571428571 hold_5_1 - 685.7142857142857 hold_5_2
   - 457.1428571428571 hold_5_3 - 342.8571428571428 hold_5_4
   - 274.2857142857143 hold_5_5 - 228.5714285714286 hold_5_6
   + flow_nic_in_5_5 <= 0
 node_throughput_constraint_6: group_tp_active_6 = 0 ->
   - 1371.428571428571 hold_6_1 - 685.7142857142857 hold_6_2
   - 457.1428571428571 hold_6_3 - 342.8571428571428 hold_6_4
   - 274.2857142857143 hold_6_5 - 228.5714285714286 hold_6_6
   + flow_nic_in_6_6 <= 0
 node_throughput_constraint_7: group_tp_active_7 = 0 ->
   - 1371.428571428571 hold_7_1 - 685.7142857142857 hold_7_2
   - 457.1428571428571 hold_7_3 - 342.8571428571428 hold_7_4
   - 274.2857142857143 hold_7_5 - 228.5714285714286 hold_7_6
   + flow_nic_in_7_7 <= 0
 node_throughput_constraint_8: group_tp_active_8 = 0 ->
   - 1371.428571428571 hold_8_1 - 685.7142857142857 hold_8_2
   - 457.1428571428571 hold_8_3 - 342.8571428571428 hold_8_4
   - 274.2857142857143 hold_8_5 - 228.5714285714286 hold_8_6
   + flow_nic_in_8_8 <= 0
 node_throughput_constraint_9: group_tp_active_9 = 0 ->
   - 1371.428571428571 hold_9_1 - 685.7142857142857 hold_9_2
   - 457.1428571428571 hold_9_3 - 342.8571428571428 hold_9_4
   - 274.2857142857143 hold_9_5 - 228.5714285714286 hold_9_6
   + flow_nic_in_9_9 <= 0
 node_throughput_constraint_10: group_tp_active_10 = 0 ->
   - 1371.428571428571 hold_10_1 - 685.7142857142857 hold_10_2
   - 457.1428571428571 hold_10_3 - 342.8571428571428 hold_10_4
   - 274.2857142857143 hold_10_5 - 228.5714285714286 hold_10_6
   + flow_nic_in_10_10 <= 0
 node_throughput_constraint_11: group_tp_active_11 = 0 ->
   - 1371.428571428571 hold_11_1 - 685.7142857142857 hold_11_2
   - 457.1428571428571 hold_11_3 - 342.8571428571428 hold_11_4
   - 274.2857142857143 hold_11_5 - 228.5714285714286 hold_11_6
   + flow_nic_in_11_11 <= 0
 node_throughput_constraint_12: group_tp_active_12 = 0 ->
   - 1371.428571428571 hold_12_1 - 685.7142857142857 hold_12_2
   - 457.1428571428571 hold_12_3 - 342.8571428571428 hold_12_4
   - 274.2857142857143 hold_12_5 - 228.5714285714286 hold_12_6
   + flow_nic_in_12_12 <= 0
 node_throughput_constraint_13: group_tp_active_13 = 0 ->
   - 1371.428571428571 hold_13_1 - 685.7142857142857 hold_13_2
   - 457.1428571428571 hold_13_3 - 342.8571428571428 hold_13_4
   - 274.2857142857143 hold_13_5 - 228.5714285714286 hold_13_6
   + flow_nic_in_13_13 <= 0
 node_throughput_constraint_14: group_tp_active_14 = 0 ->
   - 1371.428571428571 hold_14_1 - 685.7142857142857 hold_14_2
   - 457.1428571428571 hold_14_3 - 342.8571428571428 hold_14_4
   - 274.2857142857143 hold_14_5 - 228.5714285714286 hold_14_6
   + flow_nic_in_14_14 <= 0
 node_throughput_constraint_15: group_tp_active_15 = 0 ->
   - 1371.428571428571 hold_15_1 - 685.7142857142857 hold_15_2
   - 457.1428571428571 hold_15_3 - 342.8571428571428 hold_15_4
   - 274.2857142857143 hold_15_5 - 228.5714285714286 hold_15_6
   + flow_nic_in_15_15 <= 0
 node_throughput_constraint_16: group_tp_active_16 = 0 ->
   - 1371.428571428571 hold_16_1 - 685.7142857142857 hold_16_2
   - 457.1428571428571 hold_16_3 - 342.8571428571428 hold_16_4
   - 274.2857142857143 hold_16_5 - 228.5714285714286 hold_16_6
   + flow_nic_in_16_16 <= 0
 node_throughput_constraint_17: group_tp_active_17 = 0 ->
   - 1371.428571428571 hold_17_1 - 685.7142857142857 hold_17_2
   - 457.1428571428571 hold_17_3 - 342.8571428571428 hold_17_4
   - 274.2857142857143 hold_17_5 - 228.5714285714286 hold_17_6
   + flow_nic_in_17_17 <= 0
 node_throughput_constraint_18: group_tp_active_18 = 0 ->
   - 1371.428571428571 hold_18_1 - 685.7142857142857 hold_18_2
   - 457.1428571428571 hold_18_3 - 342.8571428571428 hold_18_4
   - 274.2857142857143 hold_18_5 - 228.5714285714286 hold_18_6
   + flow_nic_in_18_18 <= 0
 node_throughput_constraint_19: group_tp_active_19 = 0 ->
   - 1371.428571428571 hold_19_1 - 685.7142857142857 hold_19_2
   - 457.1428571428571 hold_19_3 - 342.8571428571428 hold_19_4
   - 274.2857142857143 hold_19_5 - 228.5714285714286 hold_19_6
   + flow_nic_in_19_19 <= 0
 node_throughput_constraint_20: group_tp_active_20 = 0 ->
   - 1371.428571428571 hold_20_1 - 685.7142857142857 hold_20_2
   - 457.1428571428571 hold_20_3 - 342.8571428571428 hold_20_4
   - 274.2857142857143 hold_20_5 - 228.5714285714286 hold_20_6
   + flow_nic_in_20_20 <= 0
 node_throughput_constraint_21: group_tp_active_21 = 0 ->
   - 1371.428571428571 hold_21_1 - 685.7142857142857 hold_21_2
   - 457.1428571428571 hold_21_3 - 342.8571428571428 hold_21_4
   - 274.2857142857143 hold_21_5 - 228.5714285714286 hold_21_6
   + flow_nic_in_21_21 <= 0
 node_throughput_constraint_22: group_tp_active_22 = 0 ->
   - 1371.428571428571 hold_22_1 - 685.7142857142857 hold_22_2
   - 457.1428571428571 hold_22_3 - 342.8571428571428 hold_22_4
   - 274.2857142857143 hold_22_5 - 228.5714285714286 hold_22_6
   + flow_nic_in_22_22 <= 0
 node_throughput_constraint_23: group_tp_active_23 = 0 ->
   - 1371.428571428571 hold_23_1 - 685.7142857142857 hold_23_2
   - 457.1428571428571 hold_23_3 - 342.8571428571428 hold_23_4
   - 274.2857142857143 hold_23_5 - 228.5714285714286 hold_23_6
   + flow_nic_in_23_23 <= 0
 node_throughput_constraint_24: group_tp_active_24 = 0 ->
   - 1371.428571428571 hold_24_1 - 685.7142857142857 hold_24_2
   - 457.1428571428571 hold_24_3 - 342.8571428571428 hold_24_4
   - 274.2857142857143 hold_24_5 - 228.5714285714286 hold_24_6
   + flow_nic_in_24_24 <= 0
 node_throughput_constraint_25: group_tp_active_25 = 0 ->
   - 1371.428571428571 hold_25_1 - 685.7142857142857 hold_25_2
   - 457.1428571428571 hold_25_3 - 342.8571428571428 hold_25_4
   - 274.2857142857143 hold_25_5 - 228.5714285714286 hold_25_6
   + flow_nic_in_25_25 <= 0
 node_throughput_constraint_26: group_tp_active_26 = 0 ->
   - 1371.428571428571 hold_26_1 - 685.7142857142857 hold_26_2
   - 457.1428571428571 hold_26_3 - 342.8571428571428 hold_26_4
   - 274.2857142857143 hold_26_5 - 228.5714285714286 hold_26_6
   + flow_nic_in_26_26 <= 0
 node_throughput_constraint_27: group_tp_active_27 = 0 ->
   - 1371.428571428571 hold_27_1 - 685.7142857142857 hold_27_2
   - 457.1428571428571 hold_27_3 - 342.8571428571428 hold_27_4
   - 274.2857142857143 hold_27_5 - 228.5714285714286 hold_27_6
   + flow_nic_in_27_27 <= 0
 node_throughput_constraint_28: group_tp_active_28 = 0 ->
   - 1371.428571428571 hold_28_1 - 685.7142857142857 hold_28_2
   - 457.1428571428571 hold_28_3 - 342.8571428571428 hold_28_4
   - 274.2857142857143 hold_28_5 - 228.5714285714286 hold_28_6
   + flow_nic_in_28_28 <= 0
 node_throughput_constraint_29: group_tp_active_29 = 0 ->
   - 1371.428571428571 hold_29_1 - 685.7142857142857 hold_29_2
   - 457.1428571428571 hold_29_3 - 342.8571428571428 hold_29_4
   - 274.2857142857143 hold_29_5 - 228.5714285714286 hold_29_6
   + flow_nic_in_29_29 <= 0
 node_throughput_constraint_30: group_tp_active_30 = 0 ->
   - 1371.428571428571 hold_30_1 - 685.7142857142857 hold_30_2
   - 457.1428571428571 hold_30_3 - 342.8571428571428 hold_30_4
   - 274.2857142857143 hold_30_5 - 228.5714285714286 hold_30_6
   + flow_nic_in_30_30 <= 0
 node_throughput_constraint_31: group_tp_active_31 = 0 ->
   - 1371.428571428571 hold_31_1 - 685.7142857142857 hold_31_2
   - 457.1428571428571 hold_31_3 - 342.8571428571428 hold_31_4
   - 274.2857142857143 hold_31_5 - 228.5714285714286 hold_31_6
   + flow_nic_in_31_31 <= 0
 constr_intergroup_throughput_0_source_nic_in_0: switch_source_nic_in_0 = 1 ->
   - 1348.314606741573 hold_0_1 - 679.8866855524079 hold_0_2
   - 454.5454545454545 hold_0_3 - 341.3940256045519 hold_0_4
   - 273.3485193621868 hold_0_5 - 227.9202279202279 hold_0_6
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_0_nic_in_1: switch_nic_out_0_nic_in_1 = 1 ->
   - 1263.157894736842 hold_1_1 - 657.5342465753424 hold_1_2
   - 444.4444444444444 hold_1_3 - 335.6643356643356 hold_1_4
   - 269.6629213483146 hold_1_5 - 225.3521126760563 hold_1_6
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_0_nic_in_2: switch_nic_out_0_nic_in_2 = 1 ->
   - 1263.157894736842 hold_2_1 - 657.5342465753424 hold_2_2
   - 444.4444444444444 hold_2_3 - 335.6643356643356 hold_2_4
   - 269.6629213483146 hold_2_5 - 225.3521126760563 hold_2_6
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_0_nic_in_3: switch_nic_out_0_nic_in_3 = 1 ->
   - 1263.157894736842 hold_3_1 - 657.5342465753424 hold_3_2
   - 444.4444444444444 hold_3_3 - 335.6643356643356 hold_3_4
   - 269.6629213483146 hold_3_5 - 225.3521126760563 hold_3_6
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_0_nic_in_4: switch_nic_out_0_nic_in_4 = 1 ->
   - 1263.157894736842 hold_4_1 - 657.5342465753424 hold_4_2
   - 444.4444444444444 hold_4_3 - 335.6643356643356 hold_4_4
   - 269.6629213483146 hold_4_5 - 225.3521126760563 hold_4_6
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_0_nic_in_5: switch_nic_out_0_nic_in_5 = 1 ->
   - 1263.157894736842 hold_5_1 - 657.5342465753424 hold_5_2
   - 444.4444444444444 hold_5_3 - 335.6643356643356 hold_5_4
   - 269.6629213483146 hold_5_5 - 225.3521126760563 hold_5_6
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_0_nic_in_6: switch_nic_out_0_nic_in_6 = 1 ->
   - 1263.157894736842 hold_6_1 - 657.5342465753424 hold_6_2
   - 444.4444444444444 hold_6_3 - 335.6643356643356 hold_6_4
   - 269.6629213483146 hold_6_5 - 225.3521126760563 hold_6_6
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_0_nic_in_7: switch_nic_out_0_nic_in_7 = 1 ->
   - 1263.157894736842 hold_7_1 - 657.5342465753424 hold_7_2
   - 444.4444444444444 hold_7_3 - 335.6643356643356 hold_7_4
   - 269.6629213483146 hold_7_5 - 225.3521126760563 hold_7_6
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_0_nic_in_8: switch_nic_out_0_nic_in_8 = 1 ->
   - 1263.157894736842 hold_8_1 - 657.5342465753424 hold_8_2
   - 444.4444444444444 hold_8_3 - 335.6643356643356 hold_8_4
   - 269.6629213483146 hold_8_5 - 225.3521126760563 hold_8_6
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_0_nic_in_9: switch_nic_out_0_nic_in_9 = 1 ->
   - 1263.157894736842 hold_9_1 - 657.5342465753424 hold_9_2
   - 444.4444444444444 hold_9_3 - 335.6643356643356 hold_9_4
   - 269.6629213483146 hold_9_5 - 225.3521126760563 hold_9_6
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_0_nic_in_10: switch_nic_out_0_nic_in_10 = 1 ->
   - 1263.157894736842 hold_10_1 - 657.5342465753424 hold_10_2
   - 444.4444444444444 hold_10_3 - 335.6643356643356 hold_10_4
   - 269.6629213483146 hold_10_5 - 225.3521126760563 hold_10_6
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_0_nic_in_11: switch_nic_out_0_nic_in_11 = 1 ->
   - 1263.157894736842 hold_11_1 - 657.5342465753424 hold_11_2
   - 444.4444444444444 hold_11_3 - 335.6643356643356 hold_11_4
   - 269.6629213483146 hold_11_5 - 225.3521126760563 hold_11_6
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_0_nic_in_12: switch_nic_out_0_nic_in_12 = 1 ->
   - 1263.157894736842 hold_12_1 - 657.5342465753424 hold_12_2
   - 444.4444444444444 hold_12_3 - 335.6643356643356 hold_12_4
   - 269.6629213483146 hold_12_5 - 225.3521126760563 hold_12_6
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_0_nic_in_13: switch_nic_out_0_nic_in_13 = 1 ->
   - 1263.157894736842 hold_13_1 - 657.5342465753424 hold_13_2
   - 444.4444444444444 hold_13_3 - 335.6643356643356 hold_13_4
   - 269.6629213483146 hold_13_5 - 225.3521126760563 hold_13_6
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_0_nic_in_14: switch_nic_out_0_nic_in_14 = 1 ->
   - 1263.157894736842 hold_14_1 - 657.5342465753424 hold_14_2
   - 444.4444444444444 hold_14_3 - 335.6643356643356 hold_14_4
   - 269.6629213483146 hold_14_5 - 225.3521126760563 hold_14_6
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_0_nic_in_15: switch_nic_out_0_nic_in_15 = 1 ->
   - 1263.157894736842 hold_15_1 - 657.5342465753424 hold_15_2
   - 444.4444444444444 hold_15_3 - 335.6643356643356 hold_15_4
   - 269.6629213483146 hold_15_5 - 225.3521126760563 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_16_nic_out_0_nic_in_16: switch_nic_out_0_nic_in_16 = 1 ->
   - 1263.157894736842 hold_16_1 - 657.5342465753424 hold_16_2
   - 444.4444444444444 hold_16_3 - 335.6643356643356 hold_16_4
   - 269.6629213483146 hold_16_5 - 225.3521126760563 hold_16_6
   + flow_nic_in_16_16 <= 0
 constr_intergroup_throughput_17_nic_out_0_nic_in_17: switch_nic_out_0_nic_in_17 = 1 ->
   - 1263.157894736842 hold_17_1 - 657.5342465753424 hold_17_2
   - 444.4444444444444 hold_17_3 - 335.6643356643356 hold_17_4
   - 269.6629213483146 hold_17_5 - 225.3521126760563 hold_17_6
   + flow_nic_in_17_17 <= 0
 constr_intergroup_throughput_18_nic_out_0_nic_in_18: switch_nic_out_0_nic_in_18 = 1 ->
   - 1263.157894736842 hold_18_1 - 657.5342465753424 hold_18_2
   - 444.4444444444444 hold_18_3 - 335.6643356643356 hold_18_4
   - 269.6629213483146 hold_18_5 - 225.3521126760563 hold_18_6
   + flow_nic_in_18_18 <= 0
 constr_intergroup_throughput_19_nic_out_0_nic_in_19: switch_nic_out_0_nic_in_19 = 1 ->
   - 1263.157894736842 hold_19_1 - 657.5342465753424 hold_19_2
   - 444.4444444444444 hold_19_3 - 335.6643356643356 hold_19_4
   - 269.6629213483146 hold_19_5 - 225.3521126760563 hold_19_6
   + flow_nic_in_19_19 <= 0
 constr_intergroup_throughput_20_nic_out_0_nic_in_20: switch_nic_out_0_nic_in_20 = 1 ->
   - 1263.157894736842 hold_20_1 - 657.5342465753424 hold_20_2
   - 444.4444444444444 hold_20_3 - 335.6643356643356 hold_20_4
   - 269.6629213483146 hold_20_5 - 225.3521126760563 hold_20_6
   + flow_nic_in_20_20 <= 0
 constr_intergroup_throughput_21_nic_out_0_nic_in_21: switch_nic_out_0_nic_in_21 = 1 ->
   - 1263.157894736842 hold_21_1 - 657.5342465753424 hold_21_2
   - 444.4444444444444 hold_21_3 - 335.6643356643356 hold_21_4
   - 269.6629213483146 hold_21_5 - 225.3521126760563 hold_21_6
   + flow_nic_in_21_21 <= 0
 constr_intergroup_throughput_22_nic_out_0_nic_in_22: switch_nic_out_0_nic_in_22 = 1 ->
   - 1263.157894736842 hold_22_1 - 657.5342465753424 hold_22_2
   - 444.4444444444444 hold_22_3 - 335.6643356643356 hold_22_4
   - 269.6629213483146 hold_22_5 - 225.3521126760563 hold_22_6
   + flow_nic_in_22_22 <= 0
 constr_intergroup_throughput_23_nic_out_0_nic_in_23: switch_nic_out_0_nic_in_23 = 1 ->
   - 1263.157894736842 hold_23_1 - 657.5342465753424 hold_23_2
   - 444.4444444444444 hold_23_3 - 335.6643356643356 hold_23_4
   - 269.6629213483146 hold_23_5 - 225.3521126760563 hold_23_6
   + flow_nic_in_23_23 <= 0
 constr_intergroup_throughput_24_nic_out_0_nic_in_24: switch_nic_out_0_nic_in_24 = 1 ->
   - 1263.157894736842 hold_24_1 - 657.5342465753424 hold_24_2
   - 444.4444444444444 hold_24_3 - 335.6643356643356 hold_24_4
   - 269.6629213483146 hold_24_5 - 225.3521126760563 hold_24_6
   + flow_nic_in_24_24 <= 0
 constr_intergroup_throughput_25_nic_out_0_nic_in_25: switch_nic_out_0_nic_in_25 = 1 ->
   - 1263.157894736842 hold_25_1 - 657.5342465753424 hold_25_2
   - 444.4444444444444 hold_25_3 - 335.6643356643356 hold_25_4
   - 269.6629213483146 hold_25_5 - 225.3521126760563 hold_25_6
   + flow_nic_in_25_25 <= 0
 constr_intergroup_throughput_26_nic_out_0_nic_in_26: switch_nic_out_0_nic_in_26 = 1 ->
   - 1263.157894736842 hold_26_1 - 657.5342465753424 hold_26_2
   - 444.4444444444444 hold_26_3 - 335.6643356643356 hold_26_4
   - 269.6629213483146 hold_26_5 - 225.3521126760563 hold_26_6
   + flow_nic_in_26_26 <= 0
 constr_intergroup_throughput_27_nic_out_0_nic_in_27: switch_nic_out_0_nic_in_27 = 1 ->
   - 1263.157894736842 hold_27_1 - 657.5342465753424 hold_27_2
   - 444.4444444444444 hold_27_3 - 335.6643356643356 hold_27_4
   - 269.6629213483146 hold_27_5 - 225.3521126760563 hold_27_6
   + flow_nic_in_27_27 <= 0
 constr_intergroup_throughput_28_nic_out_0_nic_in_28: switch_nic_out_0_nic_in_28 = 1 ->
   - 1263.157894736842 hold_28_1 - 657.5342465753424 hold_28_2
   - 444.4444444444444 hold_28_3 - 335.6643356643356 hold_28_4
   - 269.6629213483146 hold_28_5 - 225.3521126760563 hold_28_6
   + flow_nic_in_28_28 <= 0
 constr_intergroup_throughput_29_nic_out_0_nic_in_29: switch_nic_out_0_nic_in_29 = 1 ->
   - 1263.157894736842 hold_29_1 - 657.5342465753424 hold_29_2
   - 444.4444444444444 hold_29_3 - 335.6643356643356 hold_29_4
   - 269.6629213483146 hold_29_5 - 225.3521126760563 hold_29_6
   + flow_nic_in_29_29 <= 0
 constr_intergroup_throughput_30_nic_out_0_nic_in_30: switch_nic_out_0_nic_in_30 = 1 ->
   - 1263.157894736842 hold_30_1 - 657.5342465753424 hold_30_2
   - 444.4444444444444 hold_30_3 - 335.6643356643356 hold_30_4
   - 269.6629213483146 hold_30_5 - 225.3521126760563 hold_30_6
   + flow_nic_in_30_30 <= 0
 constr_intergroup_throughput_31_nic_out_0_nic_in_31: switch_nic_out_0_nic_in_31 = 1 ->
   - 1263.157894736842 hold_31_1 - 657.5342465753424 hold_31_2
   - 444.4444444444444 hold_31_3 - 335.6643356643356 hold_31_4
   - 269.6629213483146 hold_31_5 - 225.3521126760563 hold_31_6
   + flow_nic_in_31_31 <= 0
 constr_intergroup_throughput_1_source_nic_in_1: switch_source_nic_in_1 = 1 ->
   - 1348.314606741573 hold_1_1 - 679.8866855524079 hold_1_2
   - 454.5454545454545 hold_1_3 - 341.3940256045519 hold_1_4
   - 273.3485193621868 hold_1_5 - 227.9202279202279 hold_1_6
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_0_nic_out_1_nic_in_0: switch_nic_out_1_nic_in_0 = 1 ->
   - 1263.157894736842 hold_0_1 - 657.5342465753424 hold_0_2
   - 444.4444444444444 hold_0_3 - 335.6643356643356 hold_0_4
   - 269.6629213483146 hold_0_5 - 225.3521126760563 hold_0_6
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_2_nic_out_1_nic_in_2: switch_nic_out_1_nic_in_2 = 1 ->
   - 1263.157894736842 hold_2_1 - 657.5342465753424 hold_2_2
   - 444.4444444444444 hold_2_3 - 335.6643356643356 hold_2_4
   - 269.6629213483146 hold_2_5 - 225.3521126760563 hold_2_6
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_1_nic_in_3: switch_nic_out_1_nic_in_3 = 1 ->
   - 1263.157894736842 hold_3_1 - 657.5342465753424 hold_3_2
   - 444.4444444444444 hold_3_3 - 335.6643356643356 hold_3_4
   - 269.6629213483146 hold_3_5 - 225.3521126760563 hold_3_6
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_1_nic_in_4: switch_nic_out_1_nic_in_4 = 1 ->
   - 1263.157894736842 hold_4_1 - 657.5342465753424 hold_4_2
   - 444.4444444444444 hold_4_3 - 335.6643356643356 hold_4_4
   - 269.6629213483146 hold_4_5 - 225.3521126760563 hold_4_6
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_1_nic_in_5: switch_nic_out_1_nic_in_5 = 1 ->
   - 1263.157894736842 hold_5_1 - 657.5342465753424 hold_5_2
   - 444.4444444444444 hold_5_3 - 335.6643356643356 hold_5_4
   - 269.6629213483146 hold_5_5 - 225.3521126760563 hold_5_6
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_1_nic_in_6: switch_nic_out_1_nic_in_6 = 1 ->
   - 1263.157894736842 hold_6_1 - 657.5342465753424 hold_6_2
   - 444.4444444444444 hold_6_3 - 335.6643356643356 hold_6_4
   - 269.6629213483146 hold_6_5 - 225.3521126760563 hold_6_6
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_1_nic_in_7: switch_nic_out_1_nic_in_7 = 1 ->
   - 1263.157894736842 hold_7_1 - 657.5342465753424 hold_7_2
   - 444.4444444444444 hold_7_3 - 335.6643356643356 hold_7_4
   - 269.6629213483146 hold_7_5 - 225.3521126760563 hold_7_6
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_1_nic_in_8: switch_nic_out_1_nic_in_8 = 1 ->
   - 1263.157894736842 hold_8_1 - 657.5342465753424 hold_8_2
   - 444.4444444444444 hold_8_3 - 335.6643356643356 hold_8_4
   - 269.6629213483146 hold_8_5 - 225.3521126760563 hold_8_6
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_1_nic_in_9: switch_nic_out_1_nic_in_9 = 1 ->
   - 1263.157894736842 hold_9_1 - 657.5342465753424 hold_9_2
   - 444.4444444444444 hold_9_3 - 335.6643356643356 hold_9_4
   - 269.6629213483146 hold_9_5 - 225.3521126760563 hold_9_6
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_1_nic_in_10: switch_nic_out_1_nic_in_10 = 1 ->
   - 1263.157894736842 hold_10_1 - 657.5342465753424 hold_10_2
   - 444.4444444444444 hold_10_3 - 335.6643356643356 hold_10_4
   - 269.6629213483146 hold_10_5 - 225.3521126760563 hold_10_6
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_1_nic_in_11: switch_nic_out_1_nic_in_11 = 1 ->
   - 1263.157894736842 hold_11_1 - 657.5342465753424 hold_11_2
   - 444.4444444444444 hold_11_3 - 335.6643356643356 hold_11_4
   - 269.6629213483146 hold_11_5 - 225.3521126760563 hold_11_6
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_1_nic_in_12: switch_nic_out_1_nic_in_12 = 1 ->
   - 1263.157894736842 hold_12_1 - 657.5342465753424 hold_12_2
   - 444.4444444444444 hold_12_3 - 335.6643356643356 hold_12_4
   - 269.6629213483146 hold_12_5 - 225.3521126760563 hold_12_6
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_1_nic_in_13: switch_nic_out_1_nic_in_13 = 1 ->
   - 1263.157894736842 hold_13_1 - 657.5342465753424 hold_13_2
   - 444.4444444444444 hold_13_3 - 335.6643356643356 hold_13_4
   - 269.6629213483146 hold_13_5 - 225.3521126760563 hold_13_6
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_1_nic_in_14: switch_nic_out_1_nic_in_14 = 1 ->
   - 1263.157894736842 hold_14_1 - 657.5342465753424 hold_14_2
   - 444.4444444444444 hold_14_3 - 335.6643356643356 hold_14_4
   - 269.6629213483146 hold_14_5 - 225.3521126760563 hold_14_6
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_1_nic_in_15: switch_nic_out_1_nic_in_15 = 1 ->
   - 1263.157894736842 hold_15_1 - 657.5342465753424 hold_15_2
   - 444.4444444444444 hold_15_3 - 335.6643356643356 hold_15_4
   - 269.6629213483146 hold_15_5 - 225.3521126760563 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_16_nic_out_1_nic_in_16: switch_nic_out_1_nic_in_16 = 1 ->
   - 1263.157894736842 hold_16_1 - 657.5342465753424 hold_16_2
   - 444.4444444444444 hold_16_3 - 335.6643356643356 hold_16_4
   - 269.6629213483146 hold_16_5 - 225.3521126760563 hold_16_6
   + flow_nic_in_16_16 <= 0
 constr_intergroup_throughput_17_nic_out_1_nic_in_17: switch_nic_out_1_nic_in_17 = 1 ->
   - 1263.157894736842 hold_17_1 - 657.5342465753424 hold_17_2
   - 444.4444444444444 hold_17_3 - 335.6643356643356 hold_17_4
   - 269.6629213483146 hold_17_5 - 225.3521126760563 hold_17_6
   + flow_nic_in_17_17 <= 0
 constr_intergroup_throughput_18_nic_out_1_nic_in_18: switch_nic_out_1_nic_in_18 = 1 ->
   - 1263.157894736842 hold_18_1 - 657.5342465753424 hold_18_2
   - 444.4444444444444 hold_18_3 - 335.6643356643356 hold_18_4
   - 269.6629213483146 hold_18_5 - 225.3521126760563 hold_18_6
   + flow_nic_in_18_18 <= 0
 constr_intergroup_throughput_19_nic_out_1_nic_in_19: switch_nic_out_1_nic_in_19 = 1 ->
   - 1263.157894736842 hold_19_1 - 657.5342465753424 hold_19_2
   - 444.4444444444444 hold_19_3 - 335.6643356643356 hold_19_4
   - 269.6629213483146 hold_19_5 - 225.3521126760563 hold_19_6
   + flow_nic_in_19_19 <= 0
 constr_intergroup_throughput_20_nic_out_1_nic_in_20: switch_nic_out_1_nic_in_20 = 1 ->
   - 1263.157894736842 hold_20_1 - 657.5342465753424 hold_20_2
   - 444.4444444444444 hold_20_3 - 335.6643356643356 hold_20_4
   - 269.6629213483146 hold_20_5 - 225.3521126760563 hold_20_6
   + flow_nic_in_20_20 <= 0
 constr_intergroup_throughput_21_nic_out_1_nic_in_21: switch_nic_out_1_nic_in_21 = 1 ->
   - 1263.157894736842 hold_21_1 - 657.5342465753424 hold_21_2
   - 444.4444444444444 hold_21_3 - 335.6643356643356 hold_21_4
   - 269.6629213483146 hold_21_5 - 225.3521126760563 hold_21_6
   + flow_nic_in_21_21 <= 0
 constr_intergroup_throughput_22_nic_out_1_nic_in_22: switch_nic_out_1_nic_in_22 = 1 ->
   - 1263.157894736842 hold_22_1 - 657.5342465753424 hold_22_2
   - 444.4444444444444 hold_22_3 - 335.6643356643356 hold_22_4
   - 269.6629213483146 hold_22_5 - 225.3521126760563 hold_22_6
   + flow_nic_in_22_22 <= 0
 constr_intergroup_throughput_23_nic_out_1_nic_in_23: switch_nic_out_1_nic_in_23 = 1 ->
   - 1263.157894736842 hold_23_1 - 657.5342465753424 hold_23_2
   - 444.4444444444444 hold_23_3 - 335.6643356643356 hold_23_4
   - 269.6629213483146 hold_23_5 - 225.3521126760563 hold_23_6
   + flow_nic_in_23_23 <= 0
 constr_intergroup_throughput_24_nic_out_1_nic_in_24: switch_nic_out_1_nic_in_24 = 1 ->
   - 1263.157894736842 hold_24_1 - 657.5342465753424 hold_24_2
   - 444.4444444444444 hold_24_3 - 335.6643356643356 hold_24_4
   - 269.6629213483146 hold_24_5 - 225.3521126760563 hold_24_6
   + flow_nic_in_24_24 <= 0
 constr_intergroup_throughput_25_nic_out_1_nic_in_25: switch_nic_out_1_nic_in_25 = 1 ->
   - 1263.157894736842 hold_25_1 - 657.5342465753424 hold_25_2
   - 444.4444444444444 hold_25_3 - 335.6643356643356 hold_25_4
   - 269.6629213483146 hold_25_5 - 225.3521126760563 hold_25_6
   + flow_nic_in_25_25 <= 0
 constr_intergroup_throughput_26_nic_out_1_nic_in_26: switch_nic_out_1_nic_in_26 = 1 ->
   - 1263.157894736842 hold_26_1 - 657.5342465753424 hold_26_2
   - 444.4444444444444 hold_26_3 - 335.6643356643356 hold_26_4
   - 269.6629213483146 hold_26_5 - 225.3521126760563 hold_26_6
   + flow_nic_in_26_26 <= 0
 constr_intergroup_throughput_27_nic_out_1_nic_in_27: switch_nic_out_1_nic_in_27 = 1 ->
   - 1263.157894736842 hold_27_1 - 657.5342465753424 hold_27_2
   - 444.4444444444444 hold_27_3 - 335.6643356643356 hold_27_4
   - 269.6629213483146 hold_27_5 - 225.3521126760563 hold_27_6
   + flow_nic_in_27_27 <= 0
 constr_intergroup_throughput_28_nic_out_1_nic_in_28: switch_nic_out_1_nic_in_28 = 1 ->
   - 1263.157894736842 hold_28_1 - 657.5342465753424 hold_28_2
   - 444.4444444444444 hold_28_3 - 335.6643356643356 hold_28_4
   - 269.6629213483146 hold_28_5 - 225.3521126760563 hold_28_6
   + flow_nic_in_28_28 <= 0
 constr_intergroup_throughput_29_nic_out_1_nic_in_29: switch_nic_out_1_nic_in_29 = 1 ->
   - 1263.157894736842 hold_29_1 - 657.5342465753424 hold_29_2
   - 444.4444444444444 hold_29_3 - 335.6643356643356 hold_29_4
   - 269.6629213483146 hold_29_5 - 225.3521126760563 hold_29_6
   + flow_nic_in_29_29 <= 0
 constr_intergroup_throughput_30_nic_out_1_nic_in_30: switch_nic_out_1_nic_in_30 = 1 ->
   - 1263.157894736842 hold_30_1 - 657.5342465753424 hold_30_2
   - 444.4444444444444 hold_30_3 - 335.6643356643356 hold_30_4
   - 269.6629213483146 hold_30_5 - 225.3521126760563 hold_30_6
   + flow_nic_in_30_30 <= 0
 constr_intergroup_throughput_31_nic_out_1_nic_in_31: switch_nic_out_1_nic_in_31 = 1 ->
   - 1263.157894736842 hold_31_1 - 657.5342465753424 hold_31_2
   - 444.4444444444444 hold_31_3 - 335.6643356643356 hold_31_4
   - 269.6629213483146 hold_31_5 - 225.3521126760563 hold_31_6
   + flow_nic_in_31_31 <= 0
 constr_intergroup_throughput_2_source_nic_in_2: switch_source_nic_in_2 = 1 ->
   - 1348.314606741573 hold_2_1 - 679.8866855524079 hold_2_2
   - 454.5454545454545 hold_2_3 - 341.3940256045519 hold_2_4
   - 273.3485193621868 hold_2_5 - 227.9202279202279 hold_2_6
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_0_nic_out_2_nic_in_0: switch_nic_out_2_nic_in_0 = 1 ->
   - 1263.157894736842 hold_0_1 - 657.5342465753424 hold_0_2
   - 444.4444444444444 hold_0_3 - 335.6643356643356 hold_0_4
   - 269.6629213483146 hold_0_5 - 225.3521126760563 hold_0_6
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_2_nic_in_1: switch_nic_out_2_nic_in_1 = 1 ->
   - 1263.157894736842 hold_1_1 - 657.5342465753424 hold_1_2
   - 444.4444444444444 hold_1_3 - 335.6643356643356 hold_1_4
   - 269.6629213483146 hold_1_5 - 225.3521126760563 hold_1_6
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_3_nic_out_2_nic_in_3: switch_nic_out_2_nic_in_3 = 1 ->
   - 1263.157894736842 hold_3_1 - 657.5342465753424 hold_3_2
   - 444.4444444444444 hold_3_3 - 335.6643356643356 hold_3_4
   - 269.6629213483146 hold_3_5 - 225.3521126760563 hold_3_6
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_2_nic_in_4: switch_nic_out_2_nic_in_4 = 1 ->
   - 1263.157894736842 hold_4_1 - 657.5342465753424 hold_4_2
   - 444.4444444444444 hold_4_3 - 335.6643356643356 hold_4_4
   - 269.6629213483146 hold_4_5 - 225.3521126760563 hold_4_6
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_2_nic_in_5: switch_nic_out_2_nic_in_5 = 1 ->
   - 1263.157894736842 hold_5_1 - 657.5342465753424 hold_5_2
   - 444.4444444444444 hold_5_3 - 335.6643356643356 hold_5_4
   - 269.6629213483146 hold_5_5 - 225.3521126760563 hold_5_6
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_2_nic_in_6: switch_nic_out_2_nic_in_6 = 1 ->
   - 1263.157894736842 hold_6_1 - 657.5342465753424 hold_6_2
   - 444.4444444444444 hold_6_3 - 335.6643356643356 hold_6_4
   - 269.6629213483146 hold_6_5 - 225.3521126760563 hold_6_6
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_2_nic_in_7: switch_nic_out_2_nic_in_7 = 1 ->
   - 1263.157894736842 hold_7_1 - 657.5342465753424 hold_7_2
   - 444.4444444444444 hold_7_3 - 335.6643356643356 hold_7_4
   - 269.6629213483146 hold_7_5 - 225.3521126760563 hold_7_6
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_2_nic_in_8: switch_nic_out_2_nic_in_8 = 1 ->
   - 1263.157894736842 hold_8_1 - 657.5342465753424 hold_8_2
   - 444.4444444444444 hold_8_3 - 335.6643356643356 hold_8_4
   - 269.6629213483146 hold_8_5 - 225.3521126760563 hold_8_6
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_2_nic_in_9: switch_nic_out_2_nic_in_9 = 1 ->
   - 1263.157894736842 hold_9_1 - 657.5342465753424 hold_9_2
   - 444.4444444444444 hold_9_3 - 335.6643356643356 hold_9_4
   - 269.6629213483146 hold_9_5 - 225.3521126760563 hold_9_6
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_2_nic_in_10: switch_nic_out_2_nic_in_10 = 1 ->
   - 1263.157894736842 hold_10_1 - 657.5342465753424 hold_10_2
   - 444.4444444444444 hold_10_3 - 335.6643356643356 hold_10_4
   - 269.6629213483146 hold_10_5 - 225.3521126760563 hold_10_6
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_2_nic_in_11: switch_nic_out_2_nic_in_11 = 1 ->
   - 1263.157894736842 hold_11_1 - 657.5342465753424 hold_11_2
   - 444.4444444444444 hold_11_3 - 335.6643356643356 hold_11_4
   - 269.6629213483146 hold_11_5 - 225.3521126760563 hold_11_6
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_2_nic_in_12: switch_nic_out_2_nic_in_12 = 1 ->
   - 1263.157894736842 hold_12_1 - 657.5342465753424 hold_12_2
   - 444.4444444444444 hold_12_3 - 335.6643356643356 hold_12_4
   - 269.6629213483146 hold_12_5 - 225.3521126760563 hold_12_6
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_2_nic_in_13: switch_nic_out_2_nic_in_13 = 1 ->
   - 1263.157894736842 hold_13_1 - 657.5342465753424 hold_13_2
   - 444.4444444444444 hold_13_3 - 335.6643356643356 hold_13_4
   - 269.6629213483146 hold_13_5 - 225.3521126760563 hold_13_6
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_2_nic_in_14: switch_nic_out_2_nic_in_14 = 1 ->
   - 1263.157894736842 hold_14_1 - 657.5342465753424 hold_14_2
   - 444.4444444444444 hold_14_3 - 335.6643356643356 hold_14_4
   - 269.6629213483146 hold_14_5 - 225.3521126760563 hold_14_6
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_2_nic_in_15: switch_nic_out_2_nic_in_15 = 1 ->
   - 1263.157894736842 hold_15_1 - 657.5342465753424 hold_15_2
   - 444.4444444444444 hold_15_3 - 335.6643356643356 hold_15_4
   - 269.6629213483146 hold_15_5 - 225.3521126760563 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_16_nic_out_2_nic_in_16: switch_nic_out_2_nic_in_16 = 1 ->
   - 1263.157894736842 hold_16_1 - 657.5342465753424 hold_16_2
   - 444.4444444444444 hold_16_3 - 335.6643356643356 hold_16_4
   - 269.6629213483146 hold_16_5 - 225.3521126760563 hold_16_6
   + flow_nic_in_16_16 <= 0
 constr_intergroup_throughput_17_nic_out_2_nic_in_17: switch_nic_out_2_nic_in_17 = 1 ->
   - 1263.157894736842 hold_17_1 - 657.5342465753424 hold_17_2
   - 444.4444444444444 hold_17_3 - 335.6643356643356 hold_17_4
   - 269.6629213483146 hold_17_5 - 225.3521126760563 hold_17_6
   + flow_nic_in_17_17 <= 0
 constr_intergroup_throughput_18_nic_out_2_nic_in_18: switch_nic_out_2_nic_in_18 = 1 ->
   - 1263.157894736842 hold_18_1 - 657.5342465753424 hold_18_2
   - 444.4444444444444 hold_18_3 - 335.6643356643356 hold_18_4
   - 269.6629213483146 hold_18_5 - 225.3521126760563 hold_18_6
   + flow_nic_in_18_18 <= 0
 constr_intergroup_throughput_19_nic_out_2_nic_in_19: switch_nic_out_2_nic_in_19 = 1 ->
   - 1263.157894736842 hold_19_1 - 657.5342465753424 hold_19_2
   - 444.4444444444444 hold_19_3 - 335.6643356643356 hold_19_4
   - 269.6629213483146 hold_19_5 - 225.3521126760563 hold_19_6
   + flow_nic_in_19_19 <= 0
 constr_intergroup_throughput_20_nic_out_2_nic_in_20: switch_nic_out_2_nic_in_20 = 1 ->
   - 1263.157894736842 hold_20_1 - 657.5342465753424 hold_20_2
   - 444.4444444444444 hold_20_3 - 335.6643356643356 hold_20_4
   - 269.6629213483146 hold_20_5 - 225.3521126760563 hold_20_6
   + flow_nic_in_20_20 <= 0
 constr_intergroup_throughput_21_nic_out_2_nic_in_21: switch_nic_out_2_nic_in_21 = 1 ->
   - 1263.157894736842 hold_21_1 - 657.5342465753424 hold_21_2
   - 444.4444444444444 hold_21_3 - 335.6643356643356 hold_21_4
   - 269.6629213483146 hold_21_5 - 225.3521126760563 hold_21_6
   + flow_nic_in_21_21 <= 0
 constr_intergroup_throughput_22_nic_out_2_nic_in_22: switch_nic_out_2_nic_in_22 = 1 ->
   - 1263.157894736842 hold_22_1 - 657.5342465753424 hold_22_2
   - 444.4444444444444 hold_22_3 - 335.6643356643356 hold_22_4
   - 269.6629213483146 hold_22_5 - 225.3521126760563 hold_22_6
   + flow_nic_in_22_22 <= 0
 constr_intergroup_throughput_23_nic_out_2_nic_in_23: switch_nic_out_2_nic_in_23 = 1 ->
   - 1263.157894736842 hold_23_1 - 657.5342465753424 hold_23_2
   - 444.4444444444444 hold_23_3 - 335.6643356643356 hold_23_4
   - 269.6629213483146 hold_23_5 - 225.3521126760563 hold_23_6
   + flow_nic_in_23_23 <= 0
 constr_intergroup_throughput_24_nic_out_2_nic_in_24: switch_nic_out_2_nic_in_24 = 1 ->
   - 1263.157894736842 hold_24_1 - 657.5342465753424 hold_24_2
   - 444.4444444444444 hold_24_3 - 335.6643356643356 hold_24_4
   - 269.6629213483146 hold_24_5 - 225.3521126760563 hold_24_6
   + flow_nic_in_24_24 <= 0
 constr_intergroup_throughput_25_nic_out_2_nic_in_25: switch_nic_out_2_nic_in_25 = 1 ->
   - 1263.157894736842 hold_25_1 - 657.5342465753424 hold_25_2
   - 444.4444444444444 hold_25_3 - 335.6643356643356 hold_25_4
   - 269.6629213483146 hold_25_5 - 225.3521126760563 hold_25_6
   + flow_nic_in_25_25 <= 0
 constr_intergroup_throughput_26_nic_out_2_nic_in_26: switch_nic_out_2_nic_in_26 = 1 ->
   - 1263.157894736842 hold_26_1 - 657.5342465753424 hold_26_2
   - 444.4444444444444 hold_26_3 - 335.6643356643356 hold_26_4
   - 269.6629213483146 hold_26_5 - 225.3521126760563 hold_26_6
   + flow_nic_in_26_26 <= 0
 constr_intergroup_throughput_27_nic_out_2_nic_in_27: switch_nic_out_2_nic_in_27 = 1 ->
   - 1263.157894736842 hold_27_1 - 657.5342465753424 hold_27_2
   - 444.4444444444444 hold_27_3 - 335.6643356643356 hold_27_4
   - 269.6629213483146 hold_27_5 - 225.3521126760563 hold_27_6
   + flow_nic_in_27_27 <= 0
 constr_intergroup_throughput_28_nic_out_2_nic_in_28: switch_nic_out_2_nic_in_28 = 1 ->
   - 1263.157894736842 hold_28_1 - 657.5342465753424 hold_28_2
   - 444.4444444444444 hold_28_3 - 335.6643356643356 hold_28_4
   - 269.6629213483146 hold_28_5 - 225.3521126760563 hold_28_6
   + flow_nic_in_28_28 <= 0
 constr_intergroup_throughput_29_nic_out_2_nic_in_29: switch_nic_out_2_nic_in_29 = 1 ->
   - 1263.157894736842 hold_29_1 - 657.5342465753424 hold_29_2
   - 444.4444444444444 hold_29_3 - 335.6643356643356 hold_29_4
   - 269.6629213483146 hold_29_5 - 225.3521126760563 hold_29_6
   + flow_nic_in_29_29 <= 0
 constr_intergroup_throughput_30_nic_out_2_nic_in_30: switch_nic_out_2_nic_in_30 = 1 ->
   - 1263.157894736842 hold_30_1 - 657.5342465753424 hold_30_2
   - 444.4444444444444 hold_30_3 - 335.6643356643356 hold_30_4
   - 269.6629213483146 hold_30_5 - 225.3521126760563 hold_30_6
   + flow_nic_in_30_30 <= 0
 constr_intergroup_throughput_31_nic_out_2_nic_in_31: switch_nic_out_2_nic_in_31 = 1 ->
   - 1263.157894736842 hold_31_1 - 657.5342465753424 hold_31_2
   - 444.4444444444444 hold_31_3 - 335.6643356643356 hold_31_4
   - 269.6629213483146 hold_31_5 - 225.3521126760563 hold_31_6
   + flow_nic_in_31_31 <= 0
 constr_intergroup_throughput_3_source_nic_in_3: switch_source_nic_in_3 = 1 ->
   - 1348.314606741573 hold_3_1 - 679.8866855524079 hold_3_2
   - 454.5454545454545 hold_3_3 - 341.3940256045519 hold_3_4
   - 273.3485193621868 hold_3_5 - 227.9202279202279 hold_3_6
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_0_nic_out_3_nic_in_0: switch_nic_out_3_nic_in_0 = 1 ->
   - 1263.157894736842 hold_0_1 - 657.5342465753424 hold_0_2
   - 444.4444444444444 hold_0_3 - 335.6643356643356 hold_0_4
   - 269.6629213483146 hold_0_5 - 225.3521126760563 hold_0_6
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_3_nic_in_1: switch_nic_out_3_nic_in_1 = 1 ->
   - 1263.157894736842 hold_1_1 - 657.5342465753424 hold_1_2
   - 444.4444444444444 hold_1_3 - 335.6643356643356 hold_1_4
   - 269.6629213483146 hold_1_5 - 225.3521126760563 hold_1_6
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_3_nic_in_2: switch_nic_out_3_nic_in_2 = 1 ->
   - 1263.157894736842 hold_2_1 - 657.5342465753424 hold_2_2
   - 444.4444444444444 hold_2_3 - 335.6643356643356 hold_2_4
   - 269.6629213483146 hold_2_5 - 225.3521126760563 hold_2_6
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_4_nic_out_3_nic_in_4: switch_nic_out_3_nic_in_4 = 1 ->
   - 1263.157894736842 hold_4_1 - 657.5342465753424 hold_4_2
   - 444.4444444444444 hold_4_3 - 335.6643356643356 hold_4_4
   - 269.6629213483146 hold_4_5 - 225.3521126760563 hold_4_6
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_3_nic_in_5: switch_nic_out_3_nic_in_5 = 1 ->
   - 1263.157894736842 hold_5_1 - 657.5342465753424 hold_5_2
   - 444.4444444444444 hold_5_3 - 335.6643356643356 hold_5_4
   - 269.6629213483146 hold_5_5 - 225.3521126760563 hold_5_6
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_3_nic_in_6: switch_nic_out_3_nic_in_6 = 1 ->
   - 1263.157894736842 hold_6_1 - 657.5342465753424 hold_6_2
   - 444.4444444444444 hold_6_3 - 335.6643356643356 hold_6_4
   - 269.6629213483146 hold_6_5 - 225.3521126760563 hold_6_6
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_3_nic_in_7: switch_nic_out_3_nic_in_7 = 1 ->
   - 1263.157894736842 hold_7_1 - 657.5342465753424 hold_7_2
   - 444.4444444444444 hold_7_3 - 335.6643356643356 hold_7_4
   - 269.6629213483146 hold_7_5 - 225.3521126760563 hold_7_6
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_3_nic_in_8: switch_nic_out_3_nic_in_8 = 1 ->
   - 1263.157894736842 hold_8_1 - 657.5342465753424 hold_8_2
   - 444.4444444444444 hold_8_3 - 335.6643356643356 hold_8_4
   - 269.6629213483146 hold_8_5 - 225.3521126760563 hold_8_6
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_3_nic_in_9: switch_nic_out_3_nic_in_9 = 1 ->
   - 1263.157894736842 hold_9_1 - 657.5342465753424 hold_9_2
   - 444.4444444444444 hold_9_3 - 335.6643356643356 hold_9_4
   - 269.6629213483146 hold_9_5 - 225.3521126760563 hold_9_6
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_3_nic_in_10: switch_nic_out_3_nic_in_10 = 1 ->
   - 1263.157894736842 hold_10_1 - 657.5342465753424 hold_10_2
   - 444.4444444444444 hold_10_3 - 335.6643356643356 hold_10_4
   - 269.6629213483146 hold_10_5 - 225.3521126760563 hold_10_6
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_3_nic_in_11: switch_nic_out_3_nic_in_11 = 1 ->
   - 1263.157894736842 hold_11_1 - 657.5342465753424 hold_11_2
   - 444.4444444444444 hold_11_3 - 335.6643356643356 hold_11_4
   - 269.6629213483146 hold_11_5 - 225.3521126760563 hold_11_6
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_3_nic_in_12: switch_nic_out_3_nic_in_12 = 1 ->
   - 1263.157894736842 hold_12_1 - 657.5342465753424 hold_12_2
   - 444.4444444444444 hold_12_3 - 335.6643356643356 hold_12_4
   - 269.6629213483146 hold_12_5 - 225.3521126760563 hold_12_6
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_3_nic_in_13: switch_nic_out_3_nic_in_13 = 1 ->
   - 1263.157894736842 hold_13_1 - 657.5342465753424 hold_13_2
   - 444.4444444444444 hold_13_3 - 335.6643356643356 hold_13_4
   - 269.6629213483146 hold_13_5 - 225.3521126760563 hold_13_6
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_3_nic_in_14: switch_nic_out_3_nic_in_14 = 1 ->
   - 1263.157894736842 hold_14_1 - 657.5342465753424 hold_14_2
   - 444.4444444444444 hold_14_3 - 335.6643356643356 hold_14_4
   - 269.6629213483146 hold_14_5 - 225.3521126760563 hold_14_6
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_3_nic_in_15: switch_nic_out_3_nic_in_15 = 1 ->
   - 1263.157894736842 hold_15_1 - 657.5342465753424 hold_15_2
   - 444.4444444444444 hold_15_3 - 335.6643356643356 hold_15_4
   - 269.6629213483146 hold_15_5 - 225.3521126760563 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_16_nic_out_3_nic_in_16: switch_nic_out_3_nic_in_16 = 1 ->
   - 1263.157894736842 hold_16_1 - 657.5342465753424 hold_16_2
   - 444.4444444444444 hold_16_3 - 335.6643356643356 hold_16_4
   - 269.6629213483146 hold_16_5 - 225.3521126760563 hold_16_6
   + flow_nic_in_16_16 <= 0
 constr_intergroup_throughput_17_nic_out_3_nic_in_17: switch_nic_out_3_nic_in_17 = 1 ->
   - 1263.157894736842 hold_17_1 - 657.5342465753424 hold_17_2
   - 444.4444444444444 hold_17_3 - 335.6643356643356 hold_17_4
   - 269.6629213483146 hold_17_5 - 225.3521126760563 hold_17_6
   + flow_nic_in_17_17 <= 0
 constr_intergroup_throughput_18_nic_out_3_nic_in_18: switch_nic_out_3_nic_in_18 = 1 ->
   - 1263.157894736842 hold_18_1 - 657.5342465753424 hold_18_2
   - 444.4444444444444 hold_18_3 - 335.6643356643356 hold_18_4
   - 269.6629213483146 hold_18_5 - 225.3521126760563 hold_18_6
   + flow_nic_in_18_18 <= 0
 constr_intergroup_throughput_19_nic_out_3_nic_in_19: switch_nic_out_3_nic_in_19 = 1 ->
   - 1263.157894736842 hold_19_1 - 657.5342465753424 hold_19_2
   - 444.4444444444444 hold_19_3 - 335.6643356643356 hold_19_4
   - 269.6629213483146 hold_19_5 - 225.3521126760563 hold_19_6
   + flow_nic_in_19_19 <= 0
 constr_intergroup_throughput_20_nic_out_3_nic_in_20: switch_nic_out_3_nic_in_20 = 1 ->
   - 1263.157894736842 hold_20_1 - 657.5342465753424 hold_20_2
   - 444.4444444444444 hold_20_3 - 335.6643356643356 hold_20_4
   - 269.6629213483146 hold_20_5 - 225.3521126760563 hold_20_6
   + flow_nic_in_20_20 <= 0
 constr_intergroup_throughput_21_nic_out_3_nic_in_21: switch_nic_out_3_nic_in_21 = 1 ->
   - 1263.157894736842 hold_21_1 - 657.5342465753424 hold_21_2
   - 444.4444444444444 hold_21_3 - 335.6643356643356 hold_21_4
   - 269.6629213483146 hold_21_5 - 225.3521126760563 hold_21_6
   + flow_nic_in_21_21 <= 0
 constr_intergroup_throughput_22_nic_out_3_nic_in_22: switch_nic_out_3_nic_in_22 = 1 ->
   - 1263.157894736842 hold_22_1 - 657.5342465753424 hold_22_2
   - 444.4444444444444 hold_22_3 - 335.6643356643356 hold_22_4
   - 269.6629213483146 hold_22_5 - 225.3521126760563 hold_22_6
   + flow_nic_in_22_22 <= 0
 constr_intergroup_throughput_23_nic_out_3_nic_in_23: switch_nic_out_3_nic_in_23 = 1 ->
   - 1263.157894736842 hold_23_1 - 657.5342465753424 hold_23_2
   - 444.4444444444444 hold_23_3 - 335.6643356643356 hold_23_4
   - 269.6629213483146 hold_23_5 - 225.3521126760563 hold_23_6
   + flow_nic_in_23_23 <= 0
 constr_intergroup_throughput_24_nic_out_3_nic_in_24: switch_nic_out_3_nic_in_24 = 1 ->
   - 1263.157894736842 hold_24_1 - 657.5342465753424 hold_24_2
   - 444.4444444444444 hold_24_3 - 335.6643356643356 hold_24_4
   - 269.6629213483146 hold_24_5 - 225.3521126760563 hold_24_6
   + flow_nic_in_24_24 <= 0
 constr_intergroup_throughput_25_nic_out_3_nic_in_25: switch_nic_out_3_nic_in_25 = 1 ->
   - 1263.157894736842 hold_25_1 - 657.5342465753424 hold_25_2
   - 444.4444444444444 hold_25_3 - 335.6643356643356 hold_25_4
   - 269.6629213483146 hold_25_5 - 225.3521126760563 hold_25_6
   + flow_nic_in_25_25 <= 0
 constr_intergroup_throughput_26_nic_out_3_nic_in_26: switch_nic_out_3_nic_in_26 = 1 ->
   - 1263.157894736842 hold_26_1 - 657.5342465753424 hold_26_2
   - 444.4444444444444 hold_26_3 - 335.6643356643356 hold_26_4
   - 269.6629213483146 hold_26_5 - 225.3521126760563 hold_26_6
   + flow_nic_in_26_26 <= 0
 constr_intergroup_throughput_27_nic_out_3_nic_in_27: switch_nic_out_3_nic_in_27 = 1 ->
   - 1263.157894736842 hold_27_1 - 657.5342465753424 hold_27_2
   - 444.4444444444444 hold_27_3 - 335.6643356643356 hold_27_4
   - 269.6629213483146 hold_27_5 - 225.3521126760563 hold_27_6
   + flow_nic_in_27_27 <= 0
 constr_intergroup_throughput_28_nic_out_3_nic_in_28: switch_nic_out_3_nic_in_28 = 1 ->
   - 1263.157894736842 hold_28_1 - 657.5342465753424 hold_28_2
   - 444.4444444444444 hold_28_3 - 335.6643356643356 hold_28_4
   - 269.6629213483146 hold_28_5 - 225.3521126760563 hold_28_6
   + flow_nic_in_28_28 <= 0
 constr_intergroup_throughput_29_nic_out_3_nic_in_29: switch_nic_out_3_nic_in_29 = 1 ->
   - 1263.157894736842 hold_29_1 - 657.5342465753424 hold_29_2
   - 444.4444444444444 hold_29_3 - 335.6643356643356 hold_29_4
   - 269.6629213483146 hold_29_5 - 225.3521126760563 hold_29_6
   + flow_nic_in_29_29 <= 0
 constr_intergroup_throughput_30_nic_out_3_nic_in_30: switch_nic_out_3_nic_in_30 = 1 ->
   - 1263.157894736842 hold_30_1 - 657.5342465753424 hold_30_2
   - 444.4444444444444 hold_30_3 - 335.6643356643356 hold_30_4
   - 269.6629213483146 hold_30_5 - 225.3521126760563 hold_30_6
   + flow_nic_in_30_30 <= 0
 constr_intergroup_throughput_31_nic_out_3_nic_in_31: switch_nic_out_3_nic_in_31 = 1 ->
   - 1263.157894736842 hold_31_1 - 657.5342465753424 hold_31_2
   - 444.4444444444444 hold_31_3 - 335.6643356643356 hold_31_4
   - 269.6629213483146 hold_31_5 - 225.3521126760563 hold_31_6
   + flow_nic_in_31_31 <= 0
 constr_intergroup_throughput_4_source_nic_in_4: switch_source_nic_in_4 = 1 ->
   - 1348.314606741573 hold_4_1 - 679.8866855524079 hold_4_2
   - 454.5454545454545 hold_4_3 - 341.3940256045519 hold_4_4
   - 273.3485193621868 hold_4_5 - 227.9202279202279 hold_4_6
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_0_nic_out_4_nic_in_0: switch_nic_out_4_nic_in_0 = 1 ->
   - 1263.157894736842 hold_0_1 - 657.5342465753424 hold_0_2
   - 444.4444444444444 hold_0_3 - 335.6643356643356 hold_0_4
   - 269.6629213483146 hold_0_5 - 225.3521126760563 hold_0_6
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_4_nic_in_1: switch_nic_out_4_nic_in_1 = 1 ->
   - 1263.157894736842 hold_1_1 - 657.5342465753424 hold_1_2
   - 444.4444444444444 hold_1_3 - 335.6643356643356 hold_1_4
   - 269.6629213483146 hold_1_5 - 225.3521126760563 hold_1_6
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_4_nic_in_2: switch_nic_out_4_nic_in_2 = 1 ->
   - 1263.157894736842 hold_2_1 - 657.5342465753424 hold_2_2
   - 444.4444444444444 hold_2_3 - 335.6643356643356 hold_2_4
   - 269.6629213483146 hold_2_5 - 225.3521126760563 hold_2_6
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_4_nic_in_3: switch_nic_out_4_nic_in_3 = 1 ->
   - 1263.157894736842 hold_3_1 - 657.5342465753424 hold_3_2
   - 444.4444444444444 hold_3_3 - 335.6643356643356 hold_3_4
   - 269.6629213483146 hold_3_5 - 225.3521126760563 hold_3_6
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_5_nic_out_4_nic_in_5: switch_nic_out_4_nic_in_5 = 1 ->
   - 1263.157894736842 hold_5_1 - 657.5342465753424 hold_5_2
   - 444.4444444444444 hold_5_3 - 335.6643356643356 hold_5_4
   - 269.6629213483146 hold_5_5 - 225.3521126760563 hold_5_6
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_4_nic_in_6: switch_nic_out_4_nic_in_6 = 1 ->
   - 1263.157894736842 hold_6_1 - 657.5342465753424 hold_6_2
   - 444.4444444444444 hold_6_3 - 335.6643356643356 hold_6_4
   - 269.6629213483146 hold_6_5 - 225.3521126760563 hold_6_6
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_4_nic_in_7: switch_nic_out_4_nic_in_7 = 1 ->
   - 1263.157894736842 hold_7_1 - 657.5342465753424 hold_7_2
   - 444.4444444444444 hold_7_3 - 335.6643356643356 hold_7_4
   - 269.6629213483146 hold_7_5 - 225.3521126760563 hold_7_6
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_4_nic_in_8: switch_nic_out_4_nic_in_8 = 1 ->
   - 1263.157894736842 hold_8_1 - 657.5342465753424 hold_8_2
   - 444.4444444444444 hold_8_3 - 335.6643356643356 hold_8_4
   - 269.6629213483146 hold_8_5 - 225.3521126760563 hold_8_6
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_4_nic_in_9: switch_nic_out_4_nic_in_9 = 1 ->
   - 1263.157894736842 hold_9_1 - 657.5342465753424 hold_9_2
   - 444.4444444444444 hold_9_3 - 335.6643356643356 hold_9_4
   - 269.6629213483146 hold_9_5 - 225.3521126760563 hold_9_6
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_4_nic_in_10: switch_nic_out_4_nic_in_10 = 1 ->
   - 1263.157894736842 hold_10_1 - 657.5342465753424 hold_10_2
   - 444.4444444444444 hold_10_3 - 335.6643356643356 hold_10_4
   - 269.6629213483146 hold_10_5 - 225.3521126760563 hold_10_6
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_4_nic_in_11: switch_nic_out_4_nic_in_11 = 1 ->
   - 1263.157894736842 hold_11_1 - 657.5342465753424 hold_11_2
   - 444.4444444444444 hold_11_3 - 335.6643356643356 hold_11_4
   - 269.6629213483146 hold_11_5 - 225.3521126760563 hold_11_6
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_4_nic_in_12: switch_nic_out_4_nic_in_12 = 1 ->
   - 1263.157894736842 hold_12_1 - 657.5342465753424 hold_12_2
   - 444.4444444444444 hold_12_3 - 335.6643356643356 hold_12_4
   - 269.6629213483146 hold_12_5 - 225.3521126760563 hold_12_6
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_4_nic_in_13: switch_nic_out_4_nic_in_13 = 1 ->
   - 1263.157894736842 hold_13_1 - 657.5342465753424 hold_13_2
   - 444.4444444444444 hold_13_3 - 335.6643356643356 hold_13_4
   - 269.6629213483146 hold_13_5 - 225.3521126760563 hold_13_6
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_4_nic_in_14: switch_nic_out_4_nic_in_14 = 1 ->
   - 1263.157894736842 hold_14_1 - 657.5342465753424 hold_14_2
   - 444.4444444444444 hold_14_3 - 335.6643356643356 hold_14_4
   - 269.6629213483146 hold_14_5 - 225.3521126760563 hold_14_6
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_4_nic_in_15: switch_nic_out_4_nic_in_15 = 1 ->
   - 1263.157894736842 hold_15_1 - 657.5342465753424 hold_15_2
   - 444.4444444444444 hold_15_3 - 335.6643356643356 hold_15_4
   - 269.6629213483146 hold_15_5 - 225.3521126760563 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_16_nic_out_4_nic_in_16: switch_nic_out_4_nic_in_16 = 1 ->
   - 1263.157894736842 hold_16_1 - 657.5342465753424 hold_16_2
   - 444.4444444444444 hold_16_3 - 335.6643356643356 hold_16_4
   - 269.6629213483146 hold_16_5 - 225.3521126760563 hold_16_6
   + flow_nic_in_16_16 <= 0
 constr_intergroup_throughput_17_nic_out_4_nic_in_17: switch_nic_out_4_nic_in_17 = 1 ->
   - 1263.157894736842 hold_17_1 - 657.5342465753424 hold_17_2
   - 444.4444444444444 hold_17_3 - 335.6643356643356 hold_17_4
   - 269.6629213483146 hold_17_5 - 225.3521126760563 hold_17_6
   + flow_nic_in_17_17 <= 0
 constr_intergroup_throughput_18_nic_out_4_nic_in_18: switch_nic_out_4_nic_in_18 = 1 ->
   - 1263.157894736842 hold_18_1 - 657.5342465753424 hold_18_2
   - 444.4444444444444 hold_18_3 - 335.6643356643356 hold_18_4
   - 269.6629213483146 hold_18_5 - 225.3521126760563 hold_18_6
   + flow_nic_in_18_18 <= 0
 constr_intergroup_throughput_19_nic_out_4_nic_in_19: switch_nic_out_4_nic_in_19 = 1 ->
   - 1263.157894736842 hold_19_1 - 657.5342465753424 hold_19_2
   - 444.4444444444444 hold_19_3 - 335.6643356643356 hold_19_4
   - 269.6629213483146 hold_19_5 - 225.3521126760563 hold_19_6
   + flow_nic_in_19_19 <= 0
 constr_intergroup_throughput_20_nic_out_4_nic_in_20: switch_nic_out_4_nic_in_20 = 1 ->
   - 1263.157894736842 hold_20_1 - 657.5342465753424 hold_20_2
   - 444.4444444444444 hold_20_3 - 335.6643356643356 hold_20_4
   - 269.6629213483146 hold_20_5 - 225.3521126760563 hold_20_6
   + flow_nic_in_20_20 <= 0
 constr_intergroup_throughput_21_nic_out_4_nic_in_21: switch_nic_out_4_nic_in_21 = 1 ->
   - 1263.157894736842 hold_21_1 - 657.5342465753424 hold_21_2
   - 444.4444444444444 hold_21_3 - 335.6643356643356 hold_21_4
   - 269.6629213483146 hold_21_5 - 225.3521126760563 hold_21_6
   + flow_nic_in_21_21 <= 0
 constr_intergroup_throughput_22_nic_out_4_nic_in_22: switch_nic_out_4_nic_in_22 = 1 ->
   - 1263.157894736842 hold_22_1 - 657.5342465753424 hold_22_2
   - 444.4444444444444 hold_22_3 - 335.6643356643356 hold_22_4
   - 269.6629213483146 hold_22_5 - 225.3521126760563 hold_22_6
   + flow_nic_in_22_22 <= 0
 constr_intergroup_throughput_23_nic_out_4_nic_in_23: switch_nic_out_4_nic_in_23 = 1 ->
   - 1263.157894736842 hold_23_1 - 657.5342465753424 hold_23_2
   - 444.4444444444444 hold_23_3 - 335.6643356643356 hold_23_4
   - 269.6629213483146 hold_23_5 - 225.3521126760563 hold_23_6
   + flow_nic_in_23_23 <= 0
 constr_intergroup_throughput_24_nic_out_4_nic_in_24: switch_nic_out_4_nic_in_24 = 1 ->
   - 1263.157894736842 hold_24_1 - 657.5342465753424 hold_24_2
   - 444.4444444444444 hold_24_3 - 335.6643356643356 hold_24_4
   - 269.6629213483146 hold_24_5 - 225.3521126760563 hold_24_6
   + flow_nic_in_24_24 <= 0
 constr_intergroup_throughput_25_nic_out_4_nic_in_25: switch_nic_out_4_nic_in_25 = 1 ->
   - 1263.157894736842 hold_25_1 - 657.5342465753424 hold_25_2
   - 444.4444444444444 hold_25_3 - 335.6643356643356 hold_25_4
   - 269.6629213483146 hold_25_5 - 225.3521126760563 hold_25_6
   + flow_nic_in_25_25 <= 0
 constr_intergroup_throughput_26_nic_out_4_nic_in_26: switch_nic_out_4_nic_in_26 = 1 ->
   - 1263.157894736842 hold_26_1 - 657.5342465753424 hold_26_2
   - 444.4444444444444 hold_26_3 - 335.6643356643356 hold_26_4
   - 269.6629213483146 hold_26_5 - 225.3521126760563 hold_26_6
   + flow_nic_in_26_26 <= 0
 constr_intergroup_throughput_27_nic_out_4_nic_in_27: switch_nic_out_4_nic_in_27 = 1 ->
   - 1263.157894736842 hold_27_1 - 657.5342465753424 hold_27_2
   - 444.4444444444444 hold_27_3 - 335.6643356643356 hold_27_4
   - 269.6629213483146 hold_27_5 - 225.3521126760563 hold_27_6
   + flow_nic_in_27_27 <= 0
 constr_intergroup_throughput_28_nic_out_4_nic_in_28: switch_nic_out_4_nic_in_28 = 1 ->
   - 1263.157894736842 hold_28_1 - 657.5342465753424 hold_28_2
   - 444.4444444444444 hold_28_3 - 335.6643356643356 hold_28_4
   - 269.6629213483146 hold_28_5 - 225.3521126760563 hold_28_6
   + flow_nic_in_28_28 <= 0
 constr_intergroup_throughput_29_nic_out_4_nic_in_29: switch_nic_out_4_nic_in_29 = 1 ->
   - 1263.157894736842 hold_29_1 - 657.5342465753424 hold_29_2
   - 444.4444444444444 hold_29_3 - 335.6643356643356 hold_29_4
   - 269.6629213483146 hold_29_5 - 225.3521126760563 hold_29_6
   + flow_nic_in_29_29 <= 0
 constr_intergroup_throughput_30_nic_out_4_nic_in_30: switch_nic_out_4_nic_in_30 = 1 ->
   - 1263.157894736842 hold_30_1 - 657.5342465753424 hold_30_2
   - 444.4444444444444 hold_30_3 - 335.6643356643356 hold_30_4
   - 269.6629213483146 hold_30_5 - 225.3521126760563 hold_30_6
   + flow_nic_in_30_30 <= 0
 constr_intergroup_throughput_31_nic_out_4_nic_in_31: switch_nic_out_4_nic_in_31 = 1 ->
   - 1263.157894736842 hold_31_1 - 657.5342465753424 hold_31_2
   - 444.4444444444444 hold_31_3 - 335.6643356643356 hold_31_4
   - 269.6629213483146 hold_31_5 - 225.3521126760563 hold_31_6
   + flow_nic_in_31_31 <= 0
 constr_intergroup_throughput_5_source_nic_in_5: switch_source_nic_in_5 = 1 ->
   - 1348.314606741573 hold_5_1 - 679.8866855524079 hold_5_2
   - 454.5454545454545 hold_5_3 - 341.3940256045519 hold_5_4
   - 273.3485193621868 hold_5_5 - 227.9202279202279 hold_5_6
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_0_nic_out_5_nic_in_0: switch_nic_out_5_nic_in_0 = 1 ->
   - 1263.157894736842 hold_0_1 - 657.5342465753424 hold_0_2
   - 444.4444444444444 hold_0_3 - 335.6643356643356 hold_0_4
   - 269.6629213483146 hold_0_5 - 225.3521126760563 hold_0_6
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_5_nic_in_1: switch_nic_out_5_nic_in_1 = 1 ->
   - 1263.157894736842 hold_1_1 - 657.5342465753424 hold_1_2
   - 444.4444444444444 hold_1_3 - 335.6643356643356 hold_1_4
   - 269.6629213483146 hold_1_5 - 225.3521126760563 hold_1_6
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_5_nic_in_2: switch_nic_out_5_nic_in_2 = 1 ->
   - 1263.157894736842 hold_2_1 - 657.5342465753424 hold_2_2
   - 444.4444444444444 hold_2_3 - 335.6643356643356 hold_2_4
   - 269.6629213483146 hold_2_5 - 225.3521126760563 hold_2_6
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_5_nic_in_3: switch_nic_out_5_nic_in_3 = 1 ->
   - 1263.157894736842 hold_3_1 - 657.5342465753424 hold_3_2
   - 444.4444444444444 hold_3_3 - 335.6643356643356 hold_3_4
   - 269.6629213483146 hold_3_5 - 225.3521126760563 hold_3_6
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_5_nic_in_4: switch_nic_out_5_nic_in_4 = 1 ->
   - 1263.157894736842 hold_4_1 - 657.5342465753424 hold_4_2
   - 444.4444444444444 hold_4_3 - 335.6643356643356 hold_4_4
   - 269.6629213483146 hold_4_5 - 225.3521126760563 hold_4_6
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_6_nic_out_5_nic_in_6: switch_nic_out_5_nic_in_6 = 1 ->
   - 1263.157894736842 hold_6_1 - 657.5342465753424 hold_6_2
   - 444.4444444444444 hold_6_3 - 335.6643356643356 hold_6_4
   - 269.6629213483146 hold_6_5 - 225.3521126760563 hold_6_6
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_5_nic_in_7: switch_nic_out_5_nic_in_7 = 1 ->
   - 1263.157894736842 hold_7_1 - 657.5342465753424 hold_7_2
   - 444.4444444444444 hold_7_3 - 335.6643356643356 hold_7_4
   - 269.6629213483146 hold_7_5 - 225.3521126760563 hold_7_6
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_5_nic_in_8: switch_nic_out_5_nic_in_8 = 1 ->
   - 1263.157894736842 hold_8_1 - 657.5342465753424 hold_8_2
   - 444.4444444444444 hold_8_3 - 335.6643356643356 hold_8_4
   - 269.6629213483146 hold_8_5 - 225.3521126760563 hold_8_6
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_5_nic_in_9: switch_nic_out_5_nic_in_9 = 1 ->
   - 1263.157894736842 hold_9_1 - 657.5342465753424 hold_9_2
   - 444.4444444444444 hold_9_3 - 335.6643356643356 hold_9_4
   - 269.6629213483146 hold_9_5 - 225.3521126760563 hold_9_6
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_5_nic_in_10: switch_nic_out_5_nic_in_10 = 1 ->
   - 1263.157894736842 hold_10_1 - 657.5342465753424 hold_10_2
   - 444.4444444444444 hold_10_3 - 335.6643356643356 hold_10_4
   - 269.6629213483146 hold_10_5 - 225.3521126760563 hold_10_6
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_5_nic_in_11: switch_nic_out_5_nic_in_11 = 1 ->
   - 1263.157894736842 hold_11_1 - 657.5342465753424 hold_11_2
   - 444.4444444444444 hold_11_3 - 335.6643356643356 hold_11_4
   - 269.6629213483146 hold_11_5 - 225.3521126760563 hold_11_6
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_5_nic_in_12: switch_nic_out_5_nic_in_12 = 1 ->
   - 1263.157894736842 hold_12_1 - 657.5342465753424 hold_12_2
   - 444.4444444444444 hold_12_3 - 335.6643356643356 hold_12_4
   - 269.6629213483146 hold_12_5 - 225.3521126760563 hold_12_6
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_5_nic_in_13: switch_nic_out_5_nic_in_13 = 1 ->
   - 1263.157894736842 hold_13_1 - 657.5342465753424 hold_13_2
   - 444.4444444444444 hold_13_3 - 335.6643356643356 hold_13_4
   - 269.6629213483146 hold_13_5 - 225.3521126760563 hold_13_6
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_5_nic_in_14: switch_nic_out_5_nic_in_14 = 1 ->
   - 1263.157894736842 hold_14_1 - 657.5342465753424 hold_14_2
   - 444.4444444444444 hold_14_3 - 335.6643356643356 hold_14_4
   - 269.6629213483146 hold_14_5 - 225.3521126760563 hold_14_6
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_5_nic_in_15: switch_nic_out_5_nic_in_15 = 1 ->
   - 1263.157894736842 hold_15_1 - 657.5342465753424 hold_15_2
   - 444.4444444444444 hold_15_3 - 335.6643356643356 hold_15_4
   - 269.6629213483146 hold_15_5 - 225.3521126760563 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_16_nic_out_5_nic_in_16: switch_nic_out_5_nic_in_16 = 1 ->
   - 1263.157894736842 hold_16_1 - 657.5342465753424 hold_16_2
   - 444.4444444444444 hold_16_3 - 335.6643356643356 hold_16_4
   - 269.6629213483146 hold_16_5 - 225.3521126760563 hold_16_6
   + flow_nic_in_16_16 <= 0
 constr_intergroup_throughput_17_nic_out_5_nic_in_17: switch_nic_out_5_nic_in_17 = 1 ->
   - 1263.157894736842 hold_17_1 - 657.5342465753424 hold_17_2
   - 444.4444444444444 hold_17_3 - 335.6643356643356 hold_17_4
   - 269.6629213483146 hold_17_5 - 225.3521126760563 hold_17_6
   + flow_nic_in_17_17 <= 0
 constr_intergroup_throughput_18_nic_out_5_nic_in_18: switch_nic_out_5_nic_in_18 = 1 ->
   - 1263.157894736842 hold_18_1 - 657.5342465753424 hold_18_2
   - 444.4444444444444 hold_18_3 - 335.6643356643356 hold_18_4
   - 269.6629213483146 hold_18_5 - 225.3521126760563 hold_18_6
   + flow_nic_in_18_18 <= 0
 constr_intergroup_throughput_19_nic_out_5_nic_in_19: switch_nic_out_5_nic_in_19 = 1 ->
   - 1263.157894736842 hold_19_1 - 657.5342465753424 hold_19_2
   - 444.4444444444444 hold_19_3 - 335.6643356643356 hold_19_4
   - 269.6629213483146 hold_19_5 - 225.3521126760563 hold_19_6
   + flow_nic_in_19_19 <= 0
 constr_intergroup_throughput_20_nic_out_5_nic_in_20: switch_nic_out_5_nic_in_20 = 1 ->
   - 1263.157894736842 hold_20_1 - 657.5342465753424 hold_20_2
   - 444.4444444444444 hold_20_3 - 335.6643356643356 hold_20_4
   - 269.6629213483146 hold_20_5 - 225.3521126760563 hold_20_6
   + flow_nic_in_20_20 <= 0
 constr_intergroup_throughput_21_nic_out_5_nic_in_21: switch_nic_out_5_nic_in_21 = 1 ->
   - 1263.157894736842 hold_21_1 - 657.5342465753424 hold_21_2
   - 444.4444444444444 hold_21_3 - 335.6643356643356 hold_21_4
   - 269.6629213483146 hold_21_5 - 225.3521126760563 hold_21_6
   + flow_nic_in_21_21 <= 0
 constr_intergroup_throughput_22_nic_out_5_nic_in_22: switch_nic_out_5_nic_in_22 = 1 ->
   - 1263.157894736842 hold_22_1 - 657.5342465753424 hold_22_2
   - 444.4444444444444 hold_22_3 - 335.6643356643356 hold_22_4
   - 269.6629213483146 hold_22_5 - 225.3521126760563 hold_22_6
   + flow_nic_in_22_22 <= 0
 constr_intergroup_throughput_23_nic_out_5_nic_in_23: switch_nic_out_5_nic_in_23 = 1 ->
   - 1263.157894736842 hold_23_1 - 657.5342465753424 hold_23_2
   - 444.4444444444444 hold_23_3 - 335.6643356643356 hold_23_4
   - 269.6629213483146 hold_23_5 - 225.3521126760563 hold_23_6
   + flow_nic_in_23_23 <= 0
 constr_intergroup_throughput_24_nic_out_5_nic_in_24: switch_nic_out_5_nic_in_24 = 1 ->
   - 1263.157894736842 hold_24_1 - 657.5342465753424 hold_24_2
   - 444.4444444444444 hold_24_3 - 335.6643356643356 hold_24_4
   - 269.6629213483146 hold_24_5 - 225.3521126760563 hold_24_6
   + flow_nic_in_24_24 <= 0
 constr_intergroup_throughput_25_nic_out_5_nic_in_25: switch_nic_out_5_nic_in_25 = 1 ->
   - 1263.157894736842 hold_25_1 - 657.5342465753424 hold_25_2
   - 444.4444444444444 hold_25_3 - 335.6643356643356 hold_25_4
   - 269.6629213483146 hold_25_5 - 225.3521126760563 hold_25_6
   + flow_nic_in_25_25 <= 0
 constr_intergroup_throughput_26_nic_out_5_nic_in_26: switch_nic_out_5_nic_in_26 = 1 ->
   - 1263.157894736842 hold_26_1 - 657.5342465753424 hold_26_2
   - 444.4444444444444 hold_26_3 - 335.6643356643356 hold_26_4
   - 269.6629213483146 hold_26_5 - 225.3521126760563 hold_26_6
   + flow_nic_in_26_26 <= 0
 constr_intergroup_throughput_27_nic_out_5_nic_in_27: switch_nic_out_5_nic_in_27 = 1 ->
   - 1263.157894736842 hold_27_1 - 657.5342465753424 hold_27_2
   - 444.4444444444444 hold_27_3 - 335.6643356643356 hold_27_4
   - 269.6629213483146 hold_27_5 - 225.3521126760563 hold_27_6
   + flow_nic_in_27_27 <= 0
 constr_intergroup_throughput_28_nic_out_5_nic_in_28: switch_nic_out_5_nic_in_28 = 1 ->
   - 1263.157894736842 hold_28_1 - 657.5342465753424 hold_28_2
   - 444.4444444444444 hold_28_3 - 335.6643356643356 hold_28_4
   - 269.6629213483146 hold_28_5 - 225.3521126760563 hold_28_6
   + flow_nic_in_28_28 <= 0
 constr_intergroup_throughput_29_nic_out_5_nic_in_29: switch_nic_out_5_nic_in_29 = 1 ->
   - 1263.157894736842 hold_29_1 - 657.5342465753424 hold_29_2
   - 444.4444444444444 hold_29_3 - 335.6643356643356 hold_29_4
   - 269.6629213483146 hold_29_5 - 225.3521126760563 hold_29_6
   + flow_nic_in_29_29 <= 0
 constr_intergroup_throughput_30_nic_out_5_nic_in_30: switch_nic_out_5_nic_in_30 = 1 ->
   - 1263.157894736842 hold_30_1 - 657.5342465753424 hold_30_2
   - 444.4444444444444 hold_30_3 - 335.6643356643356 hold_30_4
   - 269.6629213483146 hold_30_5 - 225.3521126760563 hold_30_6
   + flow_nic_in_30_30 <= 0
 constr_intergroup_throughput_31_nic_out_5_nic_in_31: switch_nic_out_5_nic_in_31 = 1 ->
   - 1263.157894736842 hold_31_1 - 657.5342465753424 hold_31_2
   - 444.4444444444444 hold_31_3 - 335.6643356643356 hold_31_4
   - 269.6629213483146 hold_31_5 - 225.3521126760563 hold_31_6
   + flow_nic_in_31_31 <= 0
 constr_intergroup_throughput_6_source_nic_in_6: switch_source_nic_in_6 = 1 ->
   - 1348.314606741573 hold_6_1 - 679.8866855524079 hold_6_2
   - 454.5454545454545 hold_6_3 - 341.3940256045519 hold_6_4
   - 273.3485193621868 hold_6_5 - 227.9202279202279 hold_6_6
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_0_nic_out_6_nic_in_0: switch_nic_out_6_nic_in_0 = 1 ->
   - 1263.157894736842 hold_0_1 - 657.5342465753424 hold_0_2
   - 444.4444444444444 hold_0_3 - 335.6643356643356 hold_0_4
   - 269.6629213483146 hold_0_5 - 225.3521126760563 hold_0_6
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_6_nic_in_1: switch_nic_out_6_nic_in_1 = 1 ->
   - 1263.157894736842 hold_1_1 - 657.5342465753424 hold_1_2
   - 444.4444444444444 hold_1_3 - 335.6643356643356 hold_1_4
   - 269.6629213483146 hold_1_5 - 225.3521126760563 hold_1_6
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_6_nic_in_2: switch_nic_out_6_nic_in_2 = 1 ->
   - 1263.157894736842 hold_2_1 - 657.5342465753424 hold_2_2
   - 444.4444444444444 hold_2_3 - 335.6643356643356 hold_2_4
   - 269.6629213483146 hold_2_5 - 225.3521126760563 hold_2_6
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_6_nic_in_3: switch_nic_out_6_nic_in_3 = 1 ->
   - 1263.157894736842 hold_3_1 - 657.5342465753424 hold_3_2
   - 444.4444444444444 hold_3_3 - 335.6643356643356 hold_3_4
   - 269.6629213483146 hold_3_5 - 225.3521126760563 hold_3_6
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_6_nic_in_4: switch_nic_out_6_nic_in_4 = 1 ->
   - 1263.157894736842 hold_4_1 - 657.5342465753424 hold_4_2
   - 444.4444444444444 hold_4_3 - 335.6643356643356 hold_4_4
   - 269.6629213483146 hold_4_5 - 225.3521126760563 hold_4_6
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_6_nic_in_5: switch_nic_out_6_nic_in_5 = 1 ->
   - 1263.157894736842 hold_5_1 - 657.5342465753424 hold_5_2
   - 444.4444444444444 hold_5_3 - 335.6643356643356 hold_5_4
   - 269.6629213483146 hold_5_5 - 225.3521126760563 hold_5_6
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_7_nic_out_6_nic_in_7: switch_nic_out_6_nic_in_7 = 1 ->
   - 1263.157894736842 hold_7_1 - 657.5342465753424 hold_7_2
   - 444.4444444444444 hold_7_3 - 335.6643356643356 hold_7_4
   - 269.6629213483146 hold_7_5 - 225.3521126760563 hold_7_6
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_6_nic_in_8: switch_nic_out_6_nic_in_8 = 1 ->
   - 1263.157894736842 hold_8_1 - 657.5342465753424 hold_8_2
   - 444.4444444444444 hold_8_3 - 335.6643356643356 hold_8_4
   - 269.6629213483146 hold_8_5 - 225.3521126760563 hold_8_6
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_6_nic_in_9: switch_nic_out_6_nic_in_9 = 1 ->
   - 1263.157894736842 hold_9_1 - 657.5342465753424 hold_9_2
   - 444.4444444444444 hold_9_3 - 335.6643356643356 hold_9_4
   - 269.6629213483146 hold_9_5 - 225.3521126760563 hold_9_6
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_6_nic_in_10: switch_nic_out_6_nic_in_10 = 1 ->
   - 1263.157894736842 hold_10_1 - 657.5342465753424 hold_10_2
   - 444.4444444444444 hold_10_3 - 335.6643356643356 hold_10_4
   - 269.6629213483146 hold_10_5 - 225.3521126760563 hold_10_6
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_6_nic_in_11: switch_nic_out_6_nic_in_11 = 1 ->
   - 1263.157894736842 hold_11_1 - 657.5342465753424 hold_11_2
   - 444.4444444444444 hold_11_3 - 335.6643356643356 hold_11_4
   - 269.6629213483146 hold_11_5 - 225.3521126760563 hold_11_6
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_6_nic_in_12: switch_nic_out_6_nic_in_12 = 1 ->
   - 1263.157894736842 hold_12_1 - 657.5342465753424 hold_12_2
   - 444.4444444444444 hold_12_3 - 335.6643356643356 hold_12_4
   - 269.6629213483146 hold_12_5 - 225.3521126760563 hold_12_6
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_6_nic_in_13: switch_nic_out_6_nic_in_13 = 1 ->
   - 1263.157894736842 hold_13_1 - 657.5342465753424 hold_13_2
   - 444.4444444444444 hold_13_3 - 335.6643356643356 hold_13_4
   - 269.6629213483146 hold_13_5 - 225.3521126760563 hold_13_6
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_6_nic_in_14: switch_nic_out_6_nic_in_14 = 1 ->
   - 1263.157894736842 hold_14_1 - 657.5342465753424 hold_14_2
   - 444.4444444444444 hold_14_3 - 335.6643356643356 hold_14_4
   - 269.6629213483146 hold_14_5 - 225.3521126760563 hold_14_6
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_6_nic_in_15: switch_nic_out_6_nic_in_15 = 1 ->
   - 1263.157894736842 hold_15_1 - 657.5342465753424 hold_15_2
   - 444.4444444444444 hold_15_3 - 335.6643356643356 hold_15_4
   - 269.6629213483146 hold_15_5 - 225.3521126760563 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_16_nic_out_6_nic_in_16: switch_nic_out_6_nic_in_16 = 1 ->
   - 1263.157894736842 hold_16_1 - 657.5342465753424 hold_16_2
   - 444.4444444444444 hold_16_3 - 335.6643356643356 hold_16_4
   - 269.6629213483146 hold_16_5 - 225.3521126760563 hold_16_6
   + flow_nic_in_16_16 <= 0
 constr_intergroup_throughput_17_nic_out_6_nic_in_17: switch_nic_out_6_nic_in_17 = 1 ->
   - 1263.157894736842 hold_17_1 - 657.5342465753424 hold_17_2
   - 444.4444444444444 hold_17_3 - 335.6643356643356 hold_17_4
   - 269.6629213483146 hold_17_5 - 225.3521126760563 hold_17_6
   + flow_nic_in_17_17 <= 0
 constr_intergroup_throughput_18_nic_out_6_nic_in_18: switch_nic_out_6_nic_in_18 = 1 ->
   - 1263.157894736842 hold_18_1 - 657.5342465753424 hold_18_2
   - 444.4444444444444 hold_18_3 - 335.6643356643356 hold_18_4
   - 269.6629213483146 hold_18_5 - 225.3521126760563 hold_18_6
   + flow_nic_in_18_18 <= 0
 constr_intergroup_throughput_19_nic_out_6_nic_in_19: switch_nic_out_6_nic_in_19 = 1 ->
   - 1263.157894736842 hold_19_1 - 657.5342465753424 hold_19_2
   - 444.4444444444444 hold_19_3 - 335.6643356643356 hold_19_4
   - 269.6629213483146 hold_19_5 - 225.3521126760563 hold_19_6
   + flow_nic_in_19_19 <= 0
 constr_intergroup_throughput_20_nic_out_6_nic_in_20: switch_nic_out_6_nic_in_20 = 1 ->
   - 1263.157894736842 hold_20_1 - 657.5342465753424 hold_20_2
   - 444.4444444444444 hold_20_3 - 335.6643356643356 hold_20_4
   - 269.6629213483146 hold_20_5 - 225.3521126760563 hold_20_6
   + flow_nic_in_20_20 <= 0
 constr_intergroup_throughput_21_nic_out_6_nic_in_21: switch_nic_out_6_nic_in_21 = 1 ->
   - 1263.157894736842 hold_21_1 - 657.5342465753424 hold_21_2
   - 444.4444444444444 hold_21_3 - 335.6643356643356 hold_21_4
   - 269.6629213483146 hold_21_5 - 225.3521126760563 hold_21_6
   + flow_nic_in_21_21 <= 0
 constr_intergroup_throughput_22_nic_out_6_nic_in_22: switch_nic_out_6_nic_in_22 = 1 ->
   - 1263.157894736842 hold_22_1 - 657.5342465753424 hold_22_2
   - 444.4444444444444 hold_22_3 - 335.6643356643356 hold_22_4
   - 269.6629213483146 hold_22_5 - 225.3521126760563 hold_22_6
   + flow_nic_in_22_22 <= 0
 constr_intergroup_throughput_23_nic_out_6_nic_in_23: switch_nic_out_6_nic_in_23 = 1 ->
   - 1263.157894736842 hold_23_1 - 657.5342465753424 hold_23_2
   - 444.4444444444444 hold_23_3 - 335.6643356643356 hold_23_4
   - 269.6629213483146 hold_23_5 - 225.3521126760563 hold_23_6
   + flow_nic_in_23_23 <= 0
 constr_intergroup_throughput_24_nic_out_6_nic_in_24: switch_nic_out_6_nic_in_24 = 1 ->
   - 1263.157894736842 hold_24_1 - 657.5342465753424 hold_24_2
   - 444.4444444444444 hold_24_3 - 335.6643356643356 hold_24_4
   - 269.6629213483146 hold_24_5 - 225.3521126760563 hold_24_6
   + flow_nic_in_24_24 <= 0
 constr_intergroup_throughput_25_nic_out_6_nic_in_25: switch_nic_out_6_nic_in_25 = 1 ->
   - 1263.157894736842 hold_25_1 - 657.5342465753424 hold_25_2
   - 444.4444444444444 hold_25_3 - 335.6643356643356 hold_25_4
   - 269.6629213483146 hold_25_5 - 225.3521126760563 hold_25_6
   + flow_nic_in_25_25 <= 0
 constr_intergroup_throughput_26_nic_out_6_nic_in_26: switch_nic_out_6_nic_in_26 = 1 ->
   - 1263.157894736842 hold_26_1 - 657.5342465753424 hold_26_2
   - 444.4444444444444 hold_26_3 - 335.6643356643356 hold_26_4
   - 269.6629213483146 hold_26_5 - 225.3521126760563 hold_26_6
   + flow_nic_in_26_26 <= 0
 constr_intergroup_throughput_27_nic_out_6_nic_in_27: switch_nic_out_6_nic_in_27 = 1 ->
   - 1263.157894736842 hold_27_1 - 657.5342465753424 hold_27_2
   - 444.4444444444444 hold_27_3 - 335.6643356643356 hold_27_4
   - 269.6629213483146 hold_27_5 - 225.3521126760563 hold_27_6
   + flow_nic_in_27_27 <= 0
 constr_intergroup_throughput_28_nic_out_6_nic_in_28: switch_nic_out_6_nic_in_28 = 1 ->
   - 1263.157894736842 hold_28_1 - 657.5342465753424 hold_28_2
   - 444.4444444444444 hold_28_3 - 335.6643356643356 hold_28_4
   - 269.6629213483146 hold_28_5 - 225.3521126760563 hold_28_6
   + flow_nic_in_28_28 <= 0
 constr_intergroup_throughput_29_nic_out_6_nic_in_29: switch_nic_out_6_nic_in_29 = 1 ->
   - 1263.157894736842 hold_29_1 - 657.5342465753424 hold_29_2
   - 444.4444444444444 hold_29_3 - 335.6643356643356 hold_29_4
   - 269.6629213483146 hold_29_5 - 225.3521126760563 hold_29_6
   + flow_nic_in_29_29 <= 0
 constr_intergroup_throughput_30_nic_out_6_nic_in_30: switch_nic_out_6_nic_in_30 = 1 ->
   - 1263.157894736842 hold_30_1 - 657.5342465753424 hold_30_2
   - 444.4444444444444 hold_30_3 - 335.6643356643356 hold_30_4
   - 269.6629213483146 hold_30_5 - 225.3521126760563 hold_30_6
   + flow_nic_in_30_30 <= 0
 constr_intergroup_throughput_31_nic_out_6_nic_in_31: switch_nic_out_6_nic_in_31 = 1 ->
   - 1263.157894736842 hold_31_1 - 657.5342465753424 hold_31_2
   - 444.4444444444444 hold_31_3 - 335.6643356643356 hold_31_4
   - 269.6629213483146 hold_31_5 - 225.3521126760563 hold_31_6
   + flow_nic_in_31_31 <= 0
 constr_intergroup_throughput_7_source_nic_in_7: switch_source_nic_in_7 = 1 ->
   - 1348.314606741573 hold_7_1 - 679.8866855524079 hold_7_2
   - 454.5454545454545 hold_7_3 - 341.3940256045519 hold_7_4
   - 273.3485193621868 hold_7_5 - 227.9202279202279 hold_7_6
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_0_nic_out_7_nic_in_0: switch_nic_out_7_nic_in_0 = 1 ->
   - 1263.157894736842 hold_0_1 - 657.5342465753424 hold_0_2
   - 444.4444444444444 hold_0_3 - 335.6643356643356 hold_0_4
   - 269.6629213483146 hold_0_5 - 225.3521126760563 hold_0_6
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_7_nic_in_1: switch_nic_out_7_nic_in_1 = 1 ->
   - 1263.157894736842 hold_1_1 - 657.5342465753424 hold_1_2
   - 444.4444444444444 hold_1_3 - 335.6643356643356 hold_1_4
   - 269.6629213483146 hold_1_5 - 225.3521126760563 hold_1_6
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_7_nic_in_2: switch_nic_out_7_nic_in_2 = 1 ->
   - 1263.157894736842 hold_2_1 - 657.5342465753424 hold_2_2
   - 444.4444444444444 hold_2_3 - 335.6643356643356 hold_2_4
   - 269.6629213483146 hold_2_5 - 225.3521126760563 hold_2_6
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_7_nic_in_3: switch_nic_out_7_nic_in_3 = 1 ->
   - 1263.157894736842 hold_3_1 - 657.5342465753424 hold_3_2
   - 444.4444444444444 hold_3_3 - 335.6643356643356 hold_3_4
   - 269.6629213483146 hold_3_5 - 225.3521126760563 hold_3_6
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_7_nic_in_4: switch_nic_out_7_nic_in_4 = 1 ->
   - 1263.157894736842 hold_4_1 - 657.5342465753424 hold_4_2
   - 444.4444444444444 hold_4_3 - 335.6643356643356 hold_4_4
   - 269.6629213483146 hold_4_5 - 225.3521126760563 hold_4_6
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_7_nic_in_5: switch_nic_out_7_nic_in_5 = 1 ->
   - 1263.157894736842 hold_5_1 - 657.5342465753424 hold_5_2
   - 444.4444444444444 hold_5_3 - 335.6643356643356 hold_5_4
   - 269.6629213483146 hold_5_5 - 225.3521126760563 hold_5_6
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_7_nic_in_6: switch_nic_out_7_nic_in_6 = 1 ->
   - 1263.157894736842 hold_6_1 - 657.5342465753424 hold_6_2
   - 444.4444444444444 hold_6_3 - 335.6643356643356 hold_6_4
   - 269.6629213483146 hold_6_5 - 225.3521126760563 hold_6_6
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_8_nic_out_7_nic_in_8: switch_nic_out_7_nic_in_8 = 1 ->
   - 1263.157894736842 hold_8_1 - 657.5342465753424 hold_8_2
   - 444.4444444444444 hold_8_3 - 335.6643356643356 hold_8_4
   - 269.6629213483146 hold_8_5 - 225.3521126760563 hold_8_6
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_7_nic_in_9: switch_nic_out_7_nic_in_9 = 1 ->
   - 1263.157894736842 hold_9_1 - 657.5342465753424 hold_9_2
   - 444.4444444444444 hold_9_3 - 335.6643356643356 hold_9_4
   - 269.6629213483146 hold_9_5 - 225.3521126760563 hold_9_6
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_7_nic_in_10: switch_nic_out_7_nic_in_10 = 1 ->
   - 1263.157894736842 hold_10_1 - 657.5342465753424 hold_10_2
   - 444.4444444444444 hold_10_3 - 335.6643356643356 hold_10_4
   - 269.6629213483146 hold_10_5 - 225.3521126760563 hold_10_6
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_7_nic_in_11: switch_nic_out_7_nic_in_11 = 1 ->
   - 1263.157894736842 hold_11_1 - 657.5342465753424 hold_11_2
   - 444.4444444444444 hold_11_3 - 335.6643356643356 hold_11_4
   - 269.6629213483146 hold_11_5 - 225.3521126760563 hold_11_6
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_7_nic_in_12: switch_nic_out_7_nic_in_12 = 1 ->
   - 1263.157894736842 hold_12_1 - 657.5342465753424 hold_12_2
   - 444.4444444444444 hold_12_3 - 335.6643356643356 hold_12_4
   - 269.6629213483146 hold_12_5 - 225.3521126760563 hold_12_6
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_7_nic_in_13: switch_nic_out_7_nic_in_13 = 1 ->
   - 1263.157894736842 hold_13_1 - 657.5342465753424 hold_13_2
   - 444.4444444444444 hold_13_3 - 335.6643356643356 hold_13_4
   - 269.6629213483146 hold_13_5 - 225.3521126760563 hold_13_6
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_7_nic_in_14: switch_nic_out_7_nic_in_14 = 1 ->
   - 1263.157894736842 hold_14_1 - 657.5342465753424 hold_14_2
   - 444.4444444444444 hold_14_3 - 335.6643356643356 hold_14_4
   - 269.6629213483146 hold_14_5 - 225.3521126760563 hold_14_6
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_7_nic_in_15: switch_nic_out_7_nic_in_15 = 1 ->
   - 1263.157894736842 hold_15_1 - 657.5342465753424 hold_15_2
   - 444.4444444444444 hold_15_3 - 335.6643356643356 hold_15_4
   - 269.6629213483146 hold_15_5 - 225.3521126760563 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_16_nic_out_7_nic_in_16: switch_nic_out_7_nic_in_16 = 1 ->
   - 1263.157894736842 hold_16_1 - 657.5342465753424 hold_16_2
   - 444.4444444444444 hold_16_3 - 335.6643356643356 hold_16_4
   - 269.6629213483146 hold_16_5 - 225.3521126760563 hold_16_6
   + flow_nic_in_16_16 <= 0
 constr_intergroup_throughput_17_nic_out_7_nic_in_17: switch_nic_out_7_nic_in_17 = 1 ->
   - 1263.157894736842 hold_17_1 - 657.5342465753424 hold_17_2
   - 444.4444444444444 hold_17_3 - 335.6643356643356 hold_17_4
   - 269.6629213483146 hold_17_5 - 225.3521126760563 hold_17_6
   + flow_nic_in_17_17 <= 0
 constr_intergroup_throughput_18_nic_out_7_nic_in_18: switch_nic_out_7_nic_in_18 = 1 ->
   - 1263.157894736842 hold_18_1 - 657.5342465753424 hold_18_2
   - 444.4444444444444 hold_18_3 - 335.6643356643356 hold_18_4
   - 269.6629213483146 hold_18_5 - 225.3521126760563 hold_18_6
   + flow_nic_in_18_18 <= 0
 constr_intergroup_throughput_19_nic_out_7_nic_in_19: switch_nic_out_7_nic_in_19 = 1 ->
   - 1263.157894736842 hold_19_1 - 657.5342465753424 hold_19_2
   - 444.4444444444444 hold_19_3 - 335.6643356643356 hold_19_4
   - 269.6629213483146 hold_19_5 - 225.3521126760563 hold_19_6
   + flow_nic_in_19_19 <= 0
 constr_intergroup_throughput_20_nic_out_7_nic_in_20: switch_nic_out_7_nic_in_20 = 1 ->
   - 1263.157894736842 hold_20_1 - 657.5342465753424 hold_20_2
   - 444.4444444444444 hold_20_3 - 335.6643356643356 hold_20_4
   - 269.6629213483146 hold_20_5 - 225.3521126760563 hold_20_6
   + flow_nic_in_20_20 <= 0
 constr_intergroup_throughput_21_nic_out_7_nic_in_21: switch_nic_out_7_nic_in_21 = 1 ->
   - 1263.157894736842 hold_21_1 - 657.5342465753424 hold_21_2
   - 444.4444444444444 hold_21_3 - 335.6643356643356 hold_21_4
   - 269.6629213483146 hold_21_5 - 225.3521126760563 hold_21_6
   + flow_nic_in_21_21 <= 0
 constr_intergroup_throughput_22_nic_out_7_nic_in_22: switch_nic_out_7_nic_in_22 = 1 ->
   - 1263.157894736842 hold_22_1 - 657.5342465753424 hold_22_2
   - 444.4444444444444 hold_22_3 - 335.6643356643356 hold_22_4
   - 269.6629213483146 hold_22_5 - 225.3521126760563 hold_22_6
   + flow_nic_in_22_22 <= 0
 constr_intergroup_throughput_23_nic_out_7_nic_in_23: switch_nic_out_7_nic_in_23 = 1 ->
   - 1263.157894736842 hold_23_1 - 657.5342465753424 hold_23_2
   - 444.4444444444444 hold_23_3 - 335.6643356643356 hold_23_4
   - 269.6629213483146 hold_23_5 - 225.3521126760563 hold_23_6
   + flow_nic_in_23_23 <= 0
 constr_intergroup_throughput_24_nic_out_7_nic_in_24: switch_nic_out_7_nic_in_24 = 1 ->
   - 1263.157894736842 hold_24_1 - 657.5342465753424 hold_24_2
   - 444.4444444444444 hold_24_3 - 335.6643356643356 hold_24_4
   - 269.6629213483146 hold_24_5 - 225.3521126760563 hold_24_6
   + flow_nic_in_24_24 <= 0
 constr_intergroup_throughput_25_nic_out_7_nic_in_25: switch_nic_out_7_nic_in_25 = 1 ->
   - 1263.157894736842 hold_25_1 - 657.5342465753424 hold_25_2
   - 444.4444444444444 hold_25_3 - 335.6643356643356 hold_25_4
   - 269.6629213483146 hold_25_5 - 225.3521126760563 hold_25_6
   + flow_nic_in_25_25 <= 0
 constr_intergroup_throughput_26_nic_out_7_nic_in_26: switch_nic_out_7_nic_in_26 = 1 ->
   - 1263.157894736842 hold_26_1 - 657.5342465753424 hold_26_2
   - 444.4444444444444 hold_26_3 - 335.6643356643356 hold_26_4
   - 269.6629213483146 hold_26_5 - 225.3521126760563 hold_26_6
   + flow_nic_in_26_26 <= 0
 constr_intergroup_throughput_27_nic_out_7_nic_in_27: switch_nic_out_7_nic_in_27 = 1 ->
   - 1263.157894736842 hold_27_1 - 657.5342465753424 hold_27_2
   - 444.4444444444444 hold_27_3 - 335.6643356643356 hold_27_4
   - 269.6629213483146 hold_27_5 - 225.3521126760563 hold_27_6
   + flow_nic_in_27_27 <= 0
 constr_intergroup_throughput_28_nic_out_7_nic_in_28: switch_nic_out_7_nic_in_28 = 1 ->
   - 1263.157894736842 hold_28_1 - 657.5342465753424 hold_28_2
   - 444.4444444444444 hold_28_3 - 335.6643356643356 hold_28_4
   - 269.6629213483146 hold_28_5 - 225.3521126760563 hold_28_6
   + flow_nic_in_28_28 <= 0
 constr_intergroup_throughput_29_nic_out_7_nic_in_29: switch_nic_out_7_nic_in_29 = 1 ->
   - 1263.157894736842 hold_29_1 - 657.5342465753424 hold_29_2
   - 444.4444444444444 hold_29_3 - 335.6643356643356 hold_29_4
   - 269.6629213483146 hold_29_5 - 225.3521126760563 hold_29_6
   + flow_nic_in_29_29 <= 0
 constr_intergroup_throughput_30_nic_out_7_nic_in_30: switch_nic_out_7_nic_in_30 = 1 ->
   - 1263.157894736842 hold_30_1 - 657.5342465753424 hold_30_2
   - 444.4444444444444 hold_30_3 - 335.6643356643356 hold_30_4
   - 269.6629213483146 hold_30_5 - 225.3521126760563 hold_30_6
   + flow_nic_in_30_30 <= 0
 constr_intergroup_throughput_31_nic_out_7_nic_in_31: switch_nic_out_7_nic_in_31 = 1 ->
   - 1263.157894736842 hold_31_1 - 657.5342465753424 hold_31_2
   - 444.4444444444444 hold_31_3 - 335.6643356643356 hold_31_4
   - 269.6629213483146 hold_31_5 - 225.3521126760563 hold_31_6
   + flow_nic_in_31_31 <= 0
 constr_intergroup_throughput_8_source_nic_in_8: switch_source_nic_in_8 = 1 ->
   - 1348.314606741573 hold_8_1 - 679.8866855524079 hold_8_2
   - 454.5454545454545 hold_8_3 - 341.3940256045519 hold_8_4
   - 273.3485193621868 hold_8_5 - 227.9202279202279 hold_8_6
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_0_nic_out_8_nic_in_0: switch_nic_out_8_nic_in_0 = 1 ->
   - 1263.157894736842 hold_0_1 - 657.5342465753424 hold_0_2
   - 444.4444444444444 hold_0_3 - 335.6643356643356 hold_0_4
   - 269.6629213483146 hold_0_5 - 225.3521126760563 hold_0_6
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_8_nic_in_1: switch_nic_out_8_nic_in_1 = 1 ->
   - 1263.157894736842 hold_1_1 - 657.5342465753424 hold_1_2
   - 444.4444444444444 hold_1_3 - 335.6643356643356 hold_1_4
   - 269.6629213483146 hold_1_5 - 225.3521126760563 hold_1_6
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_8_nic_in_2: switch_nic_out_8_nic_in_2 = 1 ->
   - 1263.157894736842 hold_2_1 - 657.5342465753424 hold_2_2
   - 444.4444444444444 hold_2_3 - 335.6643356643356 hold_2_4
   - 269.6629213483146 hold_2_5 - 225.3521126760563 hold_2_6
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_8_nic_in_3: switch_nic_out_8_nic_in_3 = 1 ->
   - 1263.157894736842 hold_3_1 - 657.5342465753424 hold_3_2
   - 444.4444444444444 hold_3_3 - 335.6643356643356 hold_3_4
   - 269.6629213483146 hold_3_5 - 225.3521126760563 hold_3_6
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_8_nic_in_4: switch_nic_out_8_nic_in_4 = 1 ->
   - 1263.157894736842 hold_4_1 - 657.5342465753424 hold_4_2
   - 444.4444444444444 hold_4_3 - 335.6643356643356 hold_4_4
   - 269.6629213483146 hold_4_5 - 225.3521126760563 hold_4_6
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_8_nic_in_5: switch_nic_out_8_nic_in_5 = 1 ->
   - 1263.157894736842 hold_5_1 - 657.5342465753424 hold_5_2
   - 444.4444444444444 hold_5_3 - 335.6643356643356 hold_5_4
   - 269.6629213483146 hold_5_5 - 225.3521126760563 hold_5_6
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_8_nic_in_6: switch_nic_out_8_nic_in_6 = 1 ->
   - 1263.157894736842 hold_6_1 - 657.5342465753424 hold_6_2
   - 444.4444444444444 hold_6_3 - 335.6643356643356 hold_6_4
   - 269.6629213483146 hold_6_5 - 225.3521126760563 hold_6_6
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_8_nic_in_7: switch_nic_out_8_nic_in_7 = 1 ->
   - 1263.157894736842 hold_7_1 - 657.5342465753424 hold_7_2
   - 444.4444444444444 hold_7_3 - 335.6643356643356 hold_7_4
   - 269.6629213483146 hold_7_5 - 225.3521126760563 hold_7_6
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_9_nic_out_8_nic_in_9: switch_nic_out_8_nic_in_9 = 1 ->
   - 1263.157894736842 hold_9_1 - 657.5342465753424 hold_9_2
   - 444.4444444444444 hold_9_3 - 335.6643356643356 hold_9_4
   - 269.6629213483146 hold_9_5 - 225.3521126760563 hold_9_6
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_8_nic_in_10: switch_nic_out_8_nic_in_10 = 1 ->
   - 1263.157894736842 hold_10_1 - 657.5342465753424 hold_10_2
   - 444.4444444444444 hold_10_3 - 335.6643356643356 hold_10_4
   - 269.6629213483146 hold_10_5 - 225.3521126760563 hold_10_6
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_8_nic_in_11: switch_nic_out_8_nic_in_11 = 1 ->
   - 1263.157894736842 hold_11_1 - 657.5342465753424 hold_11_2
   - 444.4444444444444 hold_11_3 - 335.6643356643356 hold_11_4
   - 269.6629213483146 hold_11_5 - 225.3521126760563 hold_11_6
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_8_nic_in_12: switch_nic_out_8_nic_in_12 = 1 ->
   - 1263.157894736842 hold_12_1 - 657.5342465753424 hold_12_2
   - 444.4444444444444 hold_12_3 - 335.6643356643356 hold_12_4
   - 269.6629213483146 hold_12_5 - 225.3521126760563 hold_12_6
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_8_nic_in_13: switch_nic_out_8_nic_in_13 = 1 ->
   - 1263.157894736842 hold_13_1 - 657.5342465753424 hold_13_2
   - 444.4444444444444 hold_13_3 - 335.6643356643356 hold_13_4
   - 269.6629213483146 hold_13_5 - 225.3521126760563 hold_13_6
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_8_nic_in_14: switch_nic_out_8_nic_in_14 = 1 ->
   - 1263.157894736842 hold_14_1 - 657.5342465753424 hold_14_2
   - 444.4444444444444 hold_14_3 - 335.6643356643356 hold_14_4
   - 269.6629213483146 hold_14_5 - 225.3521126760563 hold_14_6
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_8_nic_in_15: switch_nic_out_8_nic_in_15 = 1 ->
   - 1263.157894736842 hold_15_1 - 657.5342465753424 hold_15_2
   - 444.4444444444444 hold_15_3 - 335.6643356643356 hold_15_4
   - 269.6629213483146 hold_15_5 - 225.3521126760563 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_16_nic_out_8_nic_in_16: switch_nic_out_8_nic_in_16 = 1 ->
   - 1263.157894736842 hold_16_1 - 657.5342465753424 hold_16_2
   - 444.4444444444444 hold_16_3 - 335.6643356643356 hold_16_4
   - 269.6629213483146 hold_16_5 - 225.3521126760563 hold_16_6
   + flow_nic_in_16_16 <= 0
 constr_intergroup_throughput_17_nic_out_8_nic_in_17: switch_nic_out_8_nic_in_17 = 1 ->
   - 1263.157894736842 hold_17_1 - 657.5342465753424 hold_17_2
   - 444.4444444444444 hold_17_3 - 335.6643356643356 hold_17_4
   - 269.6629213483146 hold_17_5 - 225.3521126760563 hold_17_6
   + flow_nic_in_17_17 <= 0
 constr_intergroup_throughput_18_nic_out_8_nic_in_18: switch_nic_out_8_nic_in_18 = 1 ->
   - 1263.157894736842 hold_18_1 - 657.5342465753424 hold_18_2
   - 444.4444444444444 hold_18_3 - 335.6643356643356 hold_18_4
   - 269.6629213483146 hold_18_5 - 225.3521126760563 hold_18_6
   + flow_nic_in_18_18 <= 0
 constr_intergroup_throughput_19_nic_out_8_nic_in_19: switch_nic_out_8_nic_in_19 = 1 ->
   - 1263.157894736842 hold_19_1 - 657.5342465753424 hold_19_2
   - 444.4444444444444 hold_19_3 - 335.6643356643356 hold_19_4
   - 269.6629213483146 hold_19_5 - 225.3521126760563 hold_19_6
   + flow_nic_in_19_19 <= 0
 constr_intergroup_throughput_20_nic_out_8_nic_in_20: switch_nic_out_8_nic_in_20 = 1 ->
   - 1263.157894736842 hold_20_1 - 657.5342465753424 hold_20_2
   - 444.4444444444444 hold_20_3 - 335.6643356643356 hold_20_4
   - 269.6629213483146 hold_20_5 - 225.3521126760563 hold_20_6
   + flow_nic_in_20_20 <= 0
 constr_intergroup_throughput_21_nic_out_8_nic_in_21: switch_nic_out_8_nic_in_21 = 1 ->
   - 1263.157894736842 hold_21_1 - 657.5342465753424 hold_21_2
   - 444.4444444444444 hold_21_3 - 335.6643356643356 hold_21_4
   - 269.6629213483146 hold_21_5 - 225.3521126760563 hold_21_6
   + flow_nic_in_21_21 <= 0
 constr_intergroup_throughput_22_nic_out_8_nic_in_22: switch_nic_out_8_nic_in_22 = 1 ->
   - 1263.157894736842 hold_22_1 - 657.5342465753424 hold_22_2
   - 444.4444444444444 hold_22_3 - 335.6643356643356 hold_22_4
   - 269.6629213483146 hold_22_5 - 225.3521126760563 hold_22_6
   + flow_nic_in_22_22 <= 0
 constr_intergroup_throughput_23_nic_out_8_nic_in_23: switch_nic_out_8_nic_in_23 = 1 ->
   - 1263.157894736842 hold_23_1 - 657.5342465753424 hold_23_2
   - 444.4444444444444 hold_23_3 - 335.6643356643356 hold_23_4
   - 269.6629213483146 hold_23_5 - 225.3521126760563 hold_23_6
   + flow_nic_in_23_23 <= 0
 constr_intergroup_throughput_24_nic_out_8_nic_in_24: switch_nic_out_8_nic_in_24 = 1 ->
   - 1263.157894736842 hold_24_1 - 657.5342465753424 hold_24_2
   - 444.4444444444444 hold_24_3 - 335.6643356643356 hold_24_4
   - 269.6629213483146 hold_24_5 - 225.3521126760563 hold_24_6
   + flow_nic_in_24_24 <= 0
 constr_intergroup_throughput_25_nic_out_8_nic_in_25: switch_nic_out_8_nic_in_25 = 1 ->
   - 1263.157894736842 hold_25_1 - 657.5342465753424 hold_25_2
   - 444.4444444444444 hold_25_3 - 335.6643356643356 hold_25_4
   - 269.6629213483146 hold_25_5 - 225.3521126760563 hold_25_6
   + flow_nic_in_25_25 <= 0
 constr_intergroup_throughput_26_nic_out_8_nic_in_26: switch_nic_out_8_nic_in_26 = 1 ->
   - 1263.157894736842 hold_26_1 - 657.5342465753424 hold_26_2
   - 444.4444444444444 hold_26_3 - 335.6643356643356 hold_26_4
   - 269.6629213483146 hold_26_5 - 225.3521126760563 hold_26_6
   + flow_nic_in_26_26 <= 0
 constr_intergroup_throughput_27_nic_out_8_nic_in_27: switch_nic_out_8_nic_in_27 = 1 ->
   - 1263.157894736842 hold_27_1 - 657.5342465753424 hold_27_2
   - 444.4444444444444 hold_27_3 - 335.6643356643356 hold_27_4
   - 269.6629213483146 hold_27_5 - 225.3521126760563 hold_27_6
   + flow_nic_in_27_27 <= 0
 constr_intergroup_throughput_28_nic_out_8_nic_in_28: switch_nic_out_8_nic_in_28 = 1 ->
   - 1263.157894736842 hold_28_1 - 657.5342465753424 hold_28_2
   - 444.4444444444444 hold_28_3 - 335.6643356643356 hold_28_4
   - 269.6629213483146 hold_28_5 - 225.3521126760563 hold_28_6
   + flow_nic_in_28_28 <= 0
 constr_intergroup_throughput_29_nic_out_8_nic_in_29: switch_nic_out_8_nic_in_29 = 1 ->
   - 1263.157894736842 hold_29_1 - 657.5342465753424 hold_29_2
   - 444.4444444444444 hold_29_3 - 335.6643356643356 hold_29_4
   - 269.6629213483146 hold_29_5 - 225.3521126760563 hold_29_6
   + flow_nic_in_29_29 <= 0
 constr_intergroup_throughput_30_nic_out_8_nic_in_30: switch_nic_out_8_nic_in_30 = 1 ->
   - 1263.157894736842 hold_30_1 - 657.5342465753424 hold_30_2
   - 444.4444444444444 hold_30_3 - 335.6643356643356 hold_30_4
   - 269.6629213483146 hold_30_5 - 225.3521126760563 hold_30_6
   + flow_nic_in_30_30 <= 0
 constr_intergroup_throughput_31_nic_out_8_nic_in_31: switch_nic_out_8_nic_in_31 = 1 ->
   - 1263.157894736842 hold_31_1 - 657.5342465753424 hold_31_2
   - 444.4444444444444 hold_31_3 - 335.6643356643356 hold_31_4
   - 269.6629213483146 hold_31_5 - 225.3521126760563 hold_31_6
   + flow_nic_in_31_31 <= 0
 constr_intergroup_throughput_9_source_nic_in_9: switch_source_nic_in_9 = 1 ->
   - 1348.314606741573 hold_9_1 - 679.8866855524079 hold_9_2
   - 454.5454545454545 hold_9_3 - 341.3940256045519 hold_9_4
   - 273.3485193621868 hold_9_5 - 227.9202279202279 hold_9_6
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_0_nic_out_9_nic_in_0: switch_nic_out_9_nic_in_0 = 1 ->
   - 1263.157894736842 hold_0_1 - 657.5342465753424 hold_0_2
   - 444.4444444444444 hold_0_3 - 335.6643356643356 hold_0_4
   - 269.6629213483146 hold_0_5 - 225.3521126760563 hold_0_6
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_9_nic_in_1: switch_nic_out_9_nic_in_1 = 1 ->
   - 1263.157894736842 hold_1_1 - 657.5342465753424 hold_1_2
   - 444.4444444444444 hold_1_3 - 335.6643356643356 hold_1_4
   - 269.6629213483146 hold_1_5 - 225.3521126760563 hold_1_6
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_9_nic_in_2: switch_nic_out_9_nic_in_2 = 1 ->
   - 1263.157894736842 hold_2_1 - 657.5342465753424 hold_2_2
   - 444.4444444444444 hold_2_3 - 335.6643356643356 hold_2_4
   - 269.6629213483146 hold_2_5 - 225.3521126760563 hold_2_6
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_9_nic_in_3: switch_nic_out_9_nic_in_3 = 1 ->
   - 1263.157894736842 hold_3_1 - 657.5342465753424 hold_3_2
   - 444.4444444444444 hold_3_3 - 335.6643356643356 hold_3_4
   - 269.6629213483146 hold_3_5 - 225.3521126760563 hold_3_6
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_9_nic_in_4: switch_nic_out_9_nic_in_4 = 1 ->
   - 1263.157894736842 hold_4_1 - 657.5342465753424 hold_4_2
   - 444.4444444444444 hold_4_3 - 335.6643356643356 hold_4_4
   - 269.6629213483146 hold_4_5 - 225.3521126760563 hold_4_6
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_9_nic_in_5: switch_nic_out_9_nic_in_5 = 1 ->
   - 1263.157894736842 hold_5_1 - 657.5342465753424 hold_5_2
   - 444.4444444444444 hold_5_3 - 335.6643356643356 hold_5_4
   - 269.6629213483146 hold_5_5 - 225.3521126760563 hold_5_6
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_9_nic_in_6: switch_nic_out_9_nic_in_6 = 1 ->
   - 1263.157894736842 hold_6_1 - 657.5342465753424 hold_6_2
   - 444.4444444444444 hold_6_3 - 335.6643356643356 hold_6_4
   - 269.6629213483146 hold_6_5 - 225.3521126760563 hold_6_6
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_9_nic_in_7: switch_nic_out_9_nic_in_7 = 1 ->
   - 1263.157894736842 hold_7_1 - 657.5342465753424 hold_7_2
   - 444.4444444444444 hold_7_3 - 335.6643356643356 hold_7_4
   - 269.6629213483146 hold_7_5 - 225.3521126760563 hold_7_6
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_9_nic_in_8: switch_nic_out_9_nic_in_8 = 1 ->
   - 1263.157894736842 hold_8_1 - 657.5342465753424 hold_8_2
   - 444.4444444444444 hold_8_3 - 335.6643356643356 hold_8_4
   - 269.6629213483146 hold_8_5 - 225.3521126760563 hold_8_6
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_10_nic_out_9_nic_in_10: switch_nic_out_9_nic_in_10 = 1 ->
   - 1263.157894736842 hold_10_1 - 657.5342465753424 hold_10_2
   - 444.4444444444444 hold_10_3 - 335.6643356643356 hold_10_4
   - 269.6629213483146 hold_10_5 - 225.3521126760563 hold_10_6
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_9_nic_in_11: switch_nic_out_9_nic_in_11 = 1 ->
   - 1263.157894736842 hold_11_1 - 657.5342465753424 hold_11_2
   - 444.4444444444444 hold_11_3 - 335.6643356643356 hold_11_4
   - 269.6629213483146 hold_11_5 - 225.3521126760563 hold_11_6
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_9_nic_in_12: switch_nic_out_9_nic_in_12 = 1 ->
   - 1263.157894736842 hold_12_1 - 657.5342465753424 hold_12_2
   - 444.4444444444444 hold_12_3 - 335.6643356643356 hold_12_4
   - 269.6629213483146 hold_12_5 - 225.3521126760563 hold_12_6
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_9_nic_in_13: switch_nic_out_9_nic_in_13 = 1 ->
   - 1263.157894736842 hold_13_1 - 657.5342465753424 hold_13_2
   - 444.4444444444444 hold_13_3 - 335.6643356643356 hold_13_4
   - 269.6629213483146 hold_13_5 - 225.3521126760563 hold_13_6
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_9_nic_in_14: switch_nic_out_9_nic_in_14 = 1 ->
   - 1263.157894736842 hold_14_1 - 657.5342465753424 hold_14_2
   - 444.4444444444444 hold_14_3 - 335.6643356643356 hold_14_4
   - 269.6629213483146 hold_14_5 - 225.3521126760563 hold_14_6
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_9_nic_in_15: switch_nic_out_9_nic_in_15 = 1 ->
   - 1263.157894736842 hold_15_1 - 657.5342465753424 hold_15_2
   - 444.4444444444444 hold_15_3 - 335.6643356643356 hold_15_4
   - 269.6629213483146 hold_15_5 - 225.3521126760563 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_16_nic_out_9_nic_in_16: switch_nic_out_9_nic_in_16 = 1 ->
   - 1263.157894736842 hold_16_1 - 657.5342465753424 hold_16_2
   - 444.4444444444444 hold_16_3 - 335.6643356643356 hold_16_4
   - 269.6629213483146 hold_16_5 - 225.3521126760563 hold_16_6
   + flow_nic_in_16_16 <= 0
 constr_intergroup_throughput_17_nic_out_9_nic_in_17: switch_nic_out_9_nic_in_17 = 1 ->
   - 1263.157894736842 hold_17_1 - 657.5342465753424 hold_17_2
   - 444.4444444444444 hold_17_3 - 335.6643356643356 hold_17_4
   - 269.6629213483146 hold_17_5 - 225.3521126760563 hold_17_6
   + flow_nic_in_17_17 <= 0
 constr_intergroup_throughput_18_nic_out_9_nic_in_18: switch_nic_out_9_nic_in_18 = 1 ->
   - 1263.157894736842 hold_18_1 - 657.5342465753424 hold_18_2
   - 444.4444444444444 hold_18_3 - 335.6643356643356 hold_18_4
   - 269.6629213483146 hold_18_5 - 225.3521126760563 hold_18_6
   + flow_nic_in_18_18 <= 0
 constr_intergroup_throughput_19_nic_out_9_nic_in_19: switch_nic_out_9_nic_in_19 = 1 ->
   - 1263.157894736842 hold_19_1 - 657.5342465753424 hold_19_2
   - 444.4444444444444 hold_19_3 - 335.6643356643356 hold_19_4
   - 269.6629213483146 hold_19_5 - 225.3521126760563 hold_19_6
   + flow_nic_in_19_19 <= 0
 constr_intergroup_throughput_20_nic_out_9_nic_in_20: switch_nic_out_9_nic_in_20 = 1 ->
   - 1263.157894736842 hold_20_1 - 657.5342465753424 hold_20_2
   - 444.4444444444444 hold_20_3 - 335.6643356643356 hold_20_4
   - 269.6629213483146 hold_20_5 - 225.3521126760563 hold_20_6
   + flow_nic_in_20_20 <= 0
 constr_intergroup_throughput_21_nic_out_9_nic_in_21: switch_nic_out_9_nic_in_21 = 1 ->
   - 1263.157894736842 hold_21_1 - 657.5342465753424 hold_21_2
   - 444.4444444444444 hold_21_3 - 335.6643356643356 hold_21_4
   - 269.6629213483146 hold_21_5 - 225.3521126760563 hold_21_6
   + flow_nic_in_21_21 <= 0
 constr_intergroup_throughput_22_nic_out_9_nic_in_22: switch_nic_out_9_nic_in_22 = 1 ->
   - 1263.157894736842 hold_22_1 - 657.5342465753424 hold_22_2
   - 444.4444444444444 hold_22_3 - 335.6643356643356 hold_22_4
   - 269.6629213483146 hold_22_5 - 225.3521126760563 hold_22_6
   + flow_nic_in_22_22 <= 0
 constr_intergroup_throughput_23_nic_out_9_nic_in_23: switch_nic_out_9_nic_in_23 = 1 ->
   - 1263.157894736842 hold_23_1 - 657.5342465753424 hold_23_2
   - 444.4444444444444 hold_23_3 - 335.6643356643356 hold_23_4
   - 269.6629213483146 hold_23_5 - 225.3521126760563 hold_23_6
   + flow_nic_in_23_23 <= 0
 constr_intergroup_throughput_24_nic_out_9_nic_in_24: switch_nic_out_9_nic_in_24 = 1 ->
   - 1263.157894736842 hold_24_1 - 657.5342465753424 hold_24_2
   - 444.4444444444444 hold_24_3 - 335.6643356643356 hold_24_4
   - 269.6629213483146 hold_24_5 - 225.3521126760563 hold_24_6
   + flow_nic_in_24_24 <= 0
 constr_intergroup_throughput_25_nic_out_9_nic_in_25: switch_nic_out_9_nic_in_25 = 1 ->
   - 1263.157894736842 hold_25_1 - 657.5342465753424 hold_25_2
   - 444.4444444444444 hold_25_3 - 335.6643356643356 hold_25_4
   - 269.6629213483146 hold_25_5 - 225.3521126760563 hold_25_6
   + flow_nic_in_25_25 <= 0
 constr_intergroup_throughput_26_nic_out_9_nic_in_26: switch_nic_out_9_nic_in_26 = 1 ->
   - 1263.157894736842 hold_26_1 - 657.5342465753424 hold_26_2
   - 444.4444444444444 hold_26_3 - 335.6643356643356 hold_26_4
   - 269.6629213483146 hold_26_5 - 225.3521126760563 hold_26_6
   + flow_nic_in_26_26 <= 0
 constr_intergroup_throughput_27_nic_out_9_nic_in_27: switch_nic_out_9_nic_in_27 = 1 ->
   - 1263.157894736842 hold_27_1 - 657.5342465753424 hold_27_2
   - 444.4444444444444 hold_27_3 - 335.6643356643356 hold_27_4
   - 269.6629213483146 hold_27_5 - 225.3521126760563 hold_27_6
   + flow_nic_in_27_27 <= 0
 constr_intergroup_throughput_28_nic_out_9_nic_in_28: switch_nic_out_9_nic_in_28 = 1 ->
   - 1263.157894736842 hold_28_1 - 657.5342465753424 hold_28_2
   - 444.4444444444444 hold_28_3 - 335.6643356643356 hold_28_4
   - 269.6629213483146 hold_28_5 - 225.3521126760563 hold_28_6
   + flow_nic_in_28_28 <= 0
 constr_intergroup_throughput_29_nic_out_9_nic_in_29: switch_nic_out_9_nic_in_29 = 1 ->
   - 1263.157894736842 hold_29_1 - 657.5342465753424 hold_29_2
   - 444.4444444444444 hold_29_3 - 335.6643356643356 hold_29_4
   - 269.6629213483146 hold_29_5 - 225.3521126760563 hold_29_6
   + flow_nic_in_29_29 <= 0
 constr_intergroup_throughput_30_nic_out_9_nic_in_30: switch_nic_out_9_nic_in_30 = 1 ->
   - 1263.157894736842 hold_30_1 - 657.5342465753424 hold_30_2
   - 444.4444444444444 hold_30_3 - 335.6643356643356 hold_30_4
   - 269.6629213483146 hold_30_5 - 225.3521126760563 hold_30_6
   + flow_nic_in_30_30 <= 0
 constr_intergroup_throughput_31_nic_out_9_nic_in_31: switch_nic_out_9_nic_in_31 = 1 ->
   - 1263.157894736842 hold_31_1 - 657.5342465753424 hold_31_2
   - 444.4444444444444 hold_31_3 - 335.6643356643356 hold_31_4
   - 269.6629213483146 hold_31_5 - 225.3521126760563 hold_31_6
   + flow_nic_in_31_31 <= 0
 constr_intergroup_throughput_10_source_nic_in_10: switch_source_nic_in_10 = 1 ->
   - 1348.314606741573 hold_10_1 - 679.8866855524079 hold_10_2
   - 454.5454545454545 hold_10_3 - 341.3940256045519 hold_10_4
   - 273.3485193621868 hold_10_5 - 227.9202279202279 hold_10_6
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_0_nic_out_10_nic_in_0: switch_nic_out_10_nic_in_0 = 1 ->
   - 1263.157894736842 hold_0_1 - 657.5342465753424 hold_0_2
   - 444.4444444444444 hold_0_3 - 335.6643356643356 hold_0_4
   - 269.6629213483146 hold_0_5 - 225.3521126760563 hold_0_6
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_10_nic_in_1: switch_nic_out_10_nic_in_1 = 1 ->
   - 1263.157894736842 hold_1_1 - 657.5342465753424 hold_1_2
   - 444.4444444444444 hold_1_3 - 335.6643356643356 hold_1_4
   - 269.6629213483146 hold_1_5 - 225.3521126760563 hold_1_6
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_10_nic_in_2: switch_nic_out_10_nic_in_2 = 1 ->
   - 1263.157894736842 hold_2_1 - 657.5342465753424 hold_2_2
   - 444.4444444444444 hold_2_3 - 335.6643356643356 hold_2_4
   - 269.6629213483146 hold_2_5 - 225.3521126760563 hold_2_6
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_10_nic_in_3: switch_nic_out_10_nic_in_3 = 1 ->
   - 1263.157894736842 hold_3_1 - 657.5342465753424 hold_3_2
   - 444.4444444444444 hold_3_3 - 335.6643356643356 hold_3_4
   - 269.6629213483146 hold_3_5 - 225.3521126760563 hold_3_6
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_10_nic_in_4: switch_nic_out_10_nic_in_4 = 1 ->
   - 1263.157894736842 hold_4_1 - 657.5342465753424 hold_4_2
   - 444.4444444444444 hold_4_3 - 335.6643356643356 hold_4_4
   - 269.6629213483146 hold_4_5 - 225.3521126760563 hold_4_6
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_10_nic_in_5: switch_nic_out_10_nic_in_5 = 1 ->
   - 1263.157894736842 hold_5_1 - 657.5342465753424 hold_5_2
   - 444.4444444444444 hold_5_3 - 335.6643356643356 hold_5_4
   - 269.6629213483146 hold_5_5 - 225.3521126760563 hold_5_6
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_10_nic_in_6: switch_nic_out_10_nic_in_6 = 1 ->
   - 1263.157894736842 hold_6_1 - 657.5342465753424 hold_6_2
   - 444.4444444444444 hold_6_3 - 335.6643356643356 hold_6_4
   - 269.6629213483146 hold_6_5 - 225.3521126760563 hold_6_6
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_10_nic_in_7: switch_nic_out_10_nic_in_7 = 1 ->
   - 1263.157894736842 hold_7_1 - 657.5342465753424 hold_7_2
   - 444.4444444444444 hold_7_3 - 335.6643356643356 hold_7_4
   - 269.6629213483146 hold_7_5 - 225.3521126760563 hold_7_6
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_10_nic_in_8: switch_nic_out_10_nic_in_8 = 1 ->
   - 1263.157894736842 hold_8_1 - 657.5342465753424 hold_8_2
   - 444.4444444444444 hold_8_3 - 335.6643356643356 hold_8_4
   - 269.6629213483146 hold_8_5 - 225.3521126760563 hold_8_6
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_10_nic_in_9: switch_nic_out_10_nic_in_9 = 1 ->
   - 1263.157894736842 hold_9_1 - 657.5342465753424 hold_9_2
   - 444.4444444444444 hold_9_3 - 335.6643356643356 hold_9_4
   - 269.6629213483146 hold_9_5 - 225.3521126760563 hold_9_6
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_11_nic_out_10_nic_in_11: switch_nic_out_10_nic_in_11 = 1 ->
   - 1263.157894736842 hold_11_1 - 657.5342465753424 hold_11_2
   - 444.4444444444444 hold_11_3 - 335.6643356643356 hold_11_4
   - 269.6629213483146 hold_11_5 - 225.3521126760563 hold_11_6
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_10_nic_in_12: switch_nic_out_10_nic_in_12 = 1 ->
   - 1263.157894736842 hold_12_1 - 657.5342465753424 hold_12_2
   - 444.4444444444444 hold_12_3 - 335.6643356643356 hold_12_4
   - 269.6629213483146 hold_12_5 - 225.3521126760563 hold_12_6
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_10_nic_in_13: switch_nic_out_10_nic_in_13 = 1 ->
   - 1263.157894736842 hold_13_1 - 657.5342465753424 hold_13_2
   - 444.4444444444444 hold_13_3 - 335.6643356643356 hold_13_4
   - 269.6629213483146 hold_13_5 - 225.3521126760563 hold_13_6
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_10_nic_in_14: switch_nic_out_10_nic_in_14 = 1 ->
   - 1263.157894736842 hold_14_1 - 657.5342465753424 hold_14_2
   - 444.4444444444444 hold_14_3 - 335.6643356643356 hold_14_4
   - 269.6629213483146 hold_14_5 - 225.3521126760563 hold_14_6
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_10_nic_in_15: switch_nic_out_10_nic_in_15 = 1 ->
   - 1263.157894736842 hold_15_1 - 657.5342465753424 hold_15_2
   - 444.4444444444444 hold_15_3 - 335.6643356643356 hold_15_4
   - 269.6629213483146 hold_15_5 - 225.3521126760563 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_16_nic_out_10_nic_in_16: switch_nic_out_10_nic_in_16 = 1 ->
   - 1263.157894736842 hold_16_1 - 657.5342465753424 hold_16_2
   - 444.4444444444444 hold_16_3 - 335.6643356643356 hold_16_4
   - 269.6629213483146 hold_16_5 - 225.3521126760563 hold_16_6
   + flow_nic_in_16_16 <= 0
 constr_intergroup_throughput_17_nic_out_10_nic_in_17: switch_nic_out_10_nic_in_17 = 1 ->
   - 1263.157894736842 hold_17_1 - 657.5342465753424 hold_17_2
   - 444.4444444444444 hold_17_3 - 335.6643356643356 hold_17_4
   - 269.6629213483146 hold_17_5 - 225.3521126760563 hold_17_6
   + flow_nic_in_17_17 <= 0
 constr_intergroup_throughput_18_nic_out_10_nic_in_18: switch_nic_out_10_nic_in_18 = 1 ->
   - 1263.157894736842 hold_18_1 - 657.5342465753424 hold_18_2
   - 444.4444444444444 hold_18_3 - 335.6643356643356 hold_18_4
   - 269.6629213483146 hold_18_5 - 225.3521126760563 hold_18_6
   + flow_nic_in_18_18 <= 0
 constr_intergroup_throughput_19_nic_out_10_nic_in_19: switch_nic_out_10_nic_in_19 = 1 ->
   - 1263.157894736842 hold_19_1 - 657.5342465753424 hold_19_2
   - 444.4444444444444 hold_19_3 - 335.6643356643356 hold_19_4
   - 269.6629213483146 hold_19_5 - 225.3521126760563 hold_19_6
   + flow_nic_in_19_19 <= 0
 constr_intergroup_throughput_20_nic_out_10_nic_in_20: switch_nic_out_10_nic_in_20 = 1 ->
   - 1263.157894736842 hold_20_1 - 657.5342465753424 hold_20_2
   - 444.4444444444444 hold_20_3 - 335.6643356643356 hold_20_4
   - 269.6629213483146 hold_20_5 - 225.3521126760563 hold_20_6
   + flow_nic_in_20_20 <= 0
 constr_intergroup_throughput_21_nic_out_10_nic_in_21: switch_nic_out_10_nic_in_21 = 1 ->
   - 1263.157894736842 hold_21_1 - 657.5342465753424 hold_21_2
   - 444.4444444444444 hold_21_3 - 335.6643356643356 hold_21_4
   - 269.6629213483146 hold_21_5 - 225.3521126760563 hold_21_6
   + flow_nic_in_21_21 <= 0
 constr_intergroup_throughput_22_nic_out_10_nic_in_22: switch_nic_out_10_nic_in_22 = 1 ->
   - 1263.157894736842 hold_22_1 - 657.5342465753424 hold_22_2
   - 444.4444444444444 hold_22_3 - 335.6643356643356 hold_22_4
   - 269.6629213483146 hold_22_5 - 225.3521126760563 hold_22_6
   + flow_nic_in_22_22 <= 0
 constr_intergroup_throughput_23_nic_out_10_nic_in_23: switch_nic_out_10_nic_in_23 = 1 ->
   - 1263.157894736842 hold_23_1 - 657.5342465753424 hold_23_2
   - 444.4444444444444 hold_23_3 - 335.6643356643356 hold_23_4
   - 269.6629213483146 hold_23_5 - 225.3521126760563 hold_23_6
   + flow_nic_in_23_23 <= 0
 constr_intergroup_throughput_24_nic_out_10_nic_in_24: switch_nic_out_10_nic_in_24 = 1 ->
   - 1263.157894736842 hold_24_1 - 657.5342465753424 hold_24_2
   - 444.4444444444444 hold_24_3 - 335.6643356643356 hold_24_4
   - 269.6629213483146 hold_24_5 - 225.3521126760563 hold_24_6
   + flow_nic_in_24_24 <= 0
 constr_intergroup_throughput_25_nic_out_10_nic_in_25: switch_nic_out_10_nic_in_25 = 1 ->
   - 1263.157894736842 hold_25_1 - 657.5342465753424 hold_25_2
   - 444.4444444444444 hold_25_3 - 335.6643356643356 hold_25_4
   - 269.6629213483146 hold_25_5 - 225.3521126760563 hold_25_6
   + flow_nic_in_25_25 <= 0
 constr_intergroup_throughput_26_nic_out_10_nic_in_26: switch_nic_out_10_nic_in_26 = 1 ->
   - 1263.157894736842 hold_26_1 - 657.5342465753424 hold_26_2
   - 444.4444444444444 hold_26_3 - 335.6643356643356 hold_26_4
   - 269.6629213483146 hold_26_5 - 225.3521126760563 hold_26_6
   + flow_nic_in_26_26 <= 0
 constr_intergroup_throughput_27_nic_out_10_nic_in_27: switch_nic_out_10_nic_in_27 = 1 ->
   - 1263.157894736842 hold_27_1 - 657.5342465753424 hold_27_2
   - 444.4444444444444 hold_27_3 - 335.6643356643356 hold_27_4
   - 269.6629213483146 hold_27_5 - 225.3521126760563 hold_27_6
   + flow_nic_in_27_27 <= 0
 constr_intergroup_throughput_28_nic_out_10_nic_in_28: switch_nic_out_10_nic_in_28 = 1 ->
   - 1263.157894736842 hold_28_1 - 657.5342465753424 hold_28_2
   - 444.4444444444444 hold_28_3 - 335.6643356643356 hold_28_4
   - 269.6629213483146 hold_28_5 - 225.3521126760563 hold_28_6
   + flow_nic_in_28_28 <= 0
 constr_intergroup_throughput_29_nic_out_10_nic_in_29: switch_nic_out_10_nic_in_29 = 1 ->
   - 1263.157894736842 hold_29_1 - 657.5342465753424 hold_29_2
   - 444.4444444444444 hold_29_3 - 335.6643356643356 hold_29_4
   - 269.6629213483146 hold_29_5 - 225.3521126760563 hold_29_6
   + flow_nic_in_29_29 <= 0
 constr_intergroup_throughput_30_nic_out_10_nic_in_30: switch_nic_out_10_nic_in_30 = 1 ->
   - 1263.157894736842 hold_30_1 - 657.5342465753424 hold_30_2
   - 444.4444444444444 hold_30_3 - 335.6643356643356 hold_30_4
   - 269.6629213483146 hold_30_5 - 225.3521126760563 hold_30_6
   + flow_nic_in_30_30 <= 0
 constr_intergroup_throughput_31_nic_out_10_nic_in_31: switch_nic_out_10_nic_in_31 = 1 ->
   - 1263.157894736842 hold_31_1 - 657.5342465753424 hold_31_2
   - 444.4444444444444 hold_31_3 - 335.6643356643356 hold_31_4
   - 269.6629213483146 hold_31_5 - 225.3521126760563 hold_31_6
   + flow_nic_in_31_31 <= 0
 constr_intergroup_throughput_11_source_nic_in_11: switch_source_nic_in_11 = 1 ->
   - 1348.314606741573 hold_11_1 - 679.8866855524079 hold_11_2
   - 454.5454545454545 hold_11_3 - 341.3940256045519 hold_11_4
   - 273.3485193621868 hold_11_5 - 227.9202279202279 hold_11_6
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_0_nic_out_11_nic_in_0: switch_nic_out_11_nic_in_0 = 1 ->
   - 1263.157894736842 hold_0_1 - 657.5342465753424 hold_0_2
   - 444.4444444444444 hold_0_3 - 335.6643356643356 hold_0_4
   - 269.6629213483146 hold_0_5 - 225.3521126760563 hold_0_6
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_11_nic_in_1: switch_nic_out_11_nic_in_1 = 1 ->
   - 1263.157894736842 hold_1_1 - 657.5342465753424 hold_1_2
   - 444.4444444444444 hold_1_3 - 335.6643356643356 hold_1_4
   - 269.6629213483146 hold_1_5 - 225.3521126760563 hold_1_6
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_11_nic_in_2: switch_nic_out_11_nic_in_2 = 1 ->
   - 1263.157894736842 hold_2_1 - 657.5342465753424 hold_2_2
   - 444.4444444444444 hold_2_3 - 335.6643356643356 hold_2_4
   - 269.6629213483146 hold_2_5 - 225.3521126760563 hold_2_6
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_11_nic_in_3: switch_nic_out_11_nic_in_3 = 1 ->
   - 1263.157894736842 hold_3_1 - 657.5342465753424 hold_3_2
   - 444.4444444444444 hold_3_3 - 335.6643356643356 hold_3_4
   - 269.6629213483146 hold_3_5 - 225.3521126760563 hold_3_6
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_11_nic_in_4: switch_nic_out_11_nic_in_4 = 1 ->
   - 1263.157894736842 hold_4_1 - 657.5342465753424 hold_4_2
   - 444.4444444444444 hold_4_3 - 335.6643356643356 hold_4_4
   - 269.6629213483146 hold_4_5 - 225.3521126760563 hold_4_6
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_11_nic_in_5: switch_nic_out_11_nic_in_5 = 1 ->
   - 1263.157894736842 hold_5_1 - 657.5342465753424 hold_5_2
   - 444.4444444444444 hold_5_3 - 335.6643356643356 hold_5_4
   - 269.6629213483146 hold_5_5 - 225.3521126760563 hold_5_6
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_11_nic_in_6: switch_nic_out_11_nic_in_6 = 1 ->
   - 1263.157894736842 hold_6_1 - 657.5342465753424 hold_6_2
   - 444.4444444444444 hold_6_3 - 335.6643356643356 hold_6_4
   - 269.6629213483146 hold_6_5 - 225.3521126760563 hold_6_6
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_11_nic_in_7: switch_nic_out_11_nic_in_7 = 1 ->
   - 1263.157894736842 hold_7_1 - 657.5342465753424 hold_7_2
   - 444.4444444444444 hold_7_3 - 335.6643356643356 hold_7_4
   - 269.6629213483146 hold_7_5 - 225.3521126760563 hold_7_6
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_11_nic_in_8: switch_nic_out_11_nic_in_8 = 1 ->
   - 1263.157894736842 hold_8_1 - 657.5342465753424 hold_8_2
   - 444.4444444444444 hold_8_3 - 335.6643356643356 hold_8_4
   - 269.6629213483146 hold_8_5 - 225.3521126760563 hold_8_6
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_11_nic_in_9: switch_nic_out_11_nic_in_9 = 1 ->
   - 1263.157894736842 hold_9_1 - 657.5342465753424 hold_9_2
   - 444.4444444444444 hold_9_3 - 335.6643356643356 hold_9_4
   - 269.6629213483146 hold_9_5 - 225.3521126760563 hold_9_6
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_11_nic_in_10: switch_nic_out_11_nic_in_10 = 1 ->
   - 1263.157894736842 hold_10_1 - 657.5342465753424 hold_10_2
   - 444.4444444444444 hold_10_3 - 335.6643356643356 hold_10_4
   - 269.6629213483146 hold_10_5 - 225.3521126760563 hold_10_6
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_12_nic_out_11_nic_in_12: switch_nic_out_11_nic_in_12 = 1 ->
   - 1263.157894736842 hold_12_1 - 657.5342465753424 hold_12_2
   - 444.4444444444444 hold_12_3 - 335.6643356643356 hold_12_4
   - 269.6629213483146 hold_12_5 - 225.3521126760563 hold_12_6
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_11_nic_in_13: switch_nic_out_11_nic_in_13 = 1 ->
   - 1263.157894736842 hold_13_1 - 657.5342465753424 hold_13_2
   - 444.4444444444444 hold_13_3 - 335.6643356643356 hold_13_4
   - 269.6629213483146 hold_13_5 - 225.3521126760563 hold_13_6
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_11_nic_in_14: switch_nic_out_11_nic_in_14 = 1 ->
   - 1263.157894736842 hold_14_1 - 657.5342465753424 hold_14_2
   - 444.4444444444444 hold_14_3 - 335.6643356643356 hold_14_4
   - 269.6629213483146 hold_14_5 - 225.3521126760563 hold_14_6
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_11_nic_in_15: switch_nic_out_11_nic_in_15 = 1 ->
   - 1263.157894736842 hold_15_1 - 657.5342465753424 hold_15_2
   - 444.4444444444444 hold_15_3 - 335.6643356643356 hold_15_4
   - 269.6629213483146 hold_15_5 - 225.3521126760563 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_16_nic_out_11_nic_in_16: switch_nic_out_11_nic_in_16 = 1 ->
   - 1263.157894736842 hold_16_1 - 657.5342465753424 hold_16_2
   - 444.4444444444444 hold_16_3 - 335.6643356643356 hold_16_4
   - 269.6629213483146 hold_16_5 - 225.3521126760563 hold_16_6
   + flow_nic_in_16_16 <= 0
 constr_intergroup_throughput_17_nic_out_11_nic_in_17: switch_nic_out_11_nic_in_17 = 1 ->
   - 1263.157894736842 hold_17_1 - 657.5342465753424 hold_17_2
   - 444.4444444444444 hold_17_3 - 335.6643356643356 hold_17_4
   - 269.6629213483146 hold_17_5 - 225.3521126760563 hold_17_6
   + flow_nic_in_17_17 <= 0
 constr_intergroup_throughput_18_nic_out_11_nic_in_18: switch_nic_out_11_nic_in_18 = 1 ->
   - 1263.157894736842 hold_18_1 - 657.5342465753424 hold_18_2
   - 444.4444444444444 hold_18_3 - 335.6643356643356 hold_18_4
   - 269.6629213483146 hold_18_5 - 225.3521126760563 hold_18_6
   + flow_nic_in_18_18 <= 0
 constr_intergroup_throughput_19_nic_out_11_nic_in_19: switch_nic_out_11_nic_in_19 = 1 ->
   - 1263.157894736842 hold_19_1 - 657.5342465753424 hold_19_2
   - 444.4444444444444 hold_19_3 - 335.6643356643356 hold_19_4
   - 269.6629213483146 hold_19_5 - 225.3521126760563 hold_19_6
   + flow_nic_in_19_19 <= 0
 constr_intergroup_throughput_20_nic_out_11_nic_in_20: switch_nic_out_11_nic_in_20 = 1 ->
   - 1263.157894736842 hold_20_1 - 657.5342465753424 hold_20_2
   - 444.4444444444444 hold_20_3 - 335.6643356643356 hold_20_4
   - 269.6629213483146 hold_20_5 - 225.3521126760563 hold_20_6
   + flow_nic_in_20_20 <= 0
 constr_intergroup_throughput_21_nic_out_11_nic_in_21: switch_nic_out_11_nic_in_21 = 1 ->
   - 1263.157894736842 hold_21_1 - 657.5342465753424 hold_21_2
   - 444.4444444444444 hold_21_3 - 335.6643356643356 hold_21_4
   - 269.6629213483146 hold_21_5 - 225.3521126760563 hold_21_6
   + flow_nic_in_21_21 <= 0
 constr_intergroup_throughput_22_nic_out_11_nic_in_22: switch_nic_out_11_nic_in_22 = 1 ->
   - 1263.157894736842 hold_22_1 - 657.5342465753424 hold_22_2
   - 444.4444444444444 hold_22_3 - 335.6643356643356 hold_22_4
   - 269.6629213483146 hold_22_5 - 225.3521126760563 hold_22_6
   + flow_nic_in_22_22 <= 0
 constr_intergroup_throughput_23_nic_out_11_nic_in_23: switch_nic_out_11_nic_in_23 = 1 ->
   - 1263.157894736842 hold_23_1 - 657.5342465753424 hold_23_2
   - 444.4444444444444 hold_23_3 - 335.6643356643356 hold_23_4
   - 269.6629213483146 hold_23_5 - 225.3521126760563 hold_23_6
   + flow_nic_in_23_23 <= 0
 constr_intergroup_throughput_24_nic_out_11_nic_in_24: switch_nic_out_11_nic_in_24 = 1 ->
   - 1263.157894736842 hold_24_1 - 657.5342465753424 hold_24_2
   - 444.4444444444444 hold_24_3 - 335.6643356643356 hold_24_4
   - 269.6629213483146 hold_24_5 - 225.3521126760563 hold_24_6
   + flow_nic_in_24_24 <= 0
 constr_intergroup_throughput_25_nic_out_11_nic_in_25: switch_nic_out_11_nic_in_25 = 1 ->
   - 1263.157894736842 hold_25_1 - 657.5342465753424 hold_25_2
   - 444.4444444444444 hold_25_3 - 335.6643356643356 hold_25_4
   - 269.6629213483146 hold_25_5 - 225.3521126760563 hold_25_6
   + flow_nic_in_25_25 <= 0
 constr_intergroup_throughput_26_nic_out_11_nic_in_26: switch_nic_out_11_nic_in_26 = 1 ->
   - 1263.157894736842 hold_26_1 - 657.5342465753424 hold_26_2
   - 444.4444444444444 hold_26_3 - 335.6643356643356 hold_26_4
   - 269.6629213483146 hold_26_5 - 225.3521126760563 hold_26_6
   + flow_nic_in_26_26 <= 0
 constr_intergroup_throughput_27_nic_out_11_nic_in_27: switch_nic_out_11_nic_in_27 = 1 ->
   - 1263.157894736842 hold_27_1 - 657.5342465753424 hold_27_2
   - 444.4444444444444 hold_27_3 - 335.6643356643356 hold_27_4
   - 269.6629213483146 hold_27_5 - 225.3521126760563 hold_27_6
   + flow_nic_in_27_27 <= 0
 constr_intergroup_throughput_28_nic_out_11_nic_in_28: switch_nic_out_11_nic_in_28 = 1 ->
   - 1263.157894736842 hold_28_1 - 657.5342465753424 hold_28_2
   - 444.4444444444444 hold_28_3 - 335.6643356643356 hold_28_4
   - 269.6629213483146 hold_28_5 - 225.3521126760563 hold_28_6
   + flow_nic_in_28_28 <= 0
 constr_intergroup_throughput_29_nic_out_11_nic_in_29: switch_nic_out_11_nic_in_29 = 1 ->
   - 1263.157894736842 hold_29_1 - 657.5342465753424 hold_29_2
   - 444.4444444444444 hold_29_3 - 335.6643356643356 hold_29_4
   - 269.6629213483146 hold_29_5 - 225.3521126760563 hold_29_6
   + flow_nic_in_29_29 <= 0
 constr_intergroup_throughput_30_nic_out_11_nic_in_30: switch_nic_out_11_nic_in_30 = 1 ->
   - 1263.157894736842 hold_30_1 - 657.5342465753424 hold_30_2
   - 444.4444444444444 hold_30_3 - 335.6643356643356 hold_30_4
   - 269.6629213483146 hold_30_5 - 225.3521126760563 hold_30_6
   + flow_nic_in_30_30 <= 0
 constr_intergroup_throughput_31_nic_out_11_nic_in_31: switch_nic_out_11_nic_in_31 = 1 ->
   - 1263.157894736842 hold_31_1 - 657.5342465753424 hold_31_2
   - 444.4444444444444 hold_31_3 - 335.6643356643356 hold_31_4
   - 269.6629213483146 hold_31_5 - 225.3521126760563 hold_31_6
   + flow_nic_in_31_31 <= 0
 constr_intergroup_throughput_12_source_nic_in_12: switch_source_nic_in_12 = 1 ->
   - 1348.314606741573 hold_12_1 - 679.8866855524079 hold_12_2
   - 454.5454545454545 hold_12_3 - 341.3940256045519 hold_12_4
   - 273.3485193621868 hold_12_5 - 227.9202279202279 hold_12_6
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_0_nic_out_12_nic_in_0: switch_nic_out_12_nic_in_0 = 1 ->
   - 1263.157894736842 hold_0_1 - 657.5342465753424 hold_0_2
   - 444.4444444444444 hold_0_3 - 335.6643356643356 hold_0_4
   - 269.6629213483146 hold_0_5 - 225.3521126760563 hold_0_6
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_12_nic_in_1: switch_nic_out_12_nic_in_1 = 1 ->
   - 1263.157894736842 hold_1_1 - 657.5342465753424 hold_1_2
   - 444.4444444444444 hold_1_3 - 335.6643356643356 hold_1_4
   - 269.6629213483146 hold_1_5 - 225.3521126760563 hold_1_6
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_12_nic_in_2: switch_nic_out_12_nic_in_2 = 1 ->
   - 1263.157894736842 hold_2_1 - 657.5342465753424 hold_2_2
   - 444.4444444444444 hold_2_3 - 335.6643356643356 hold_2_4
   - 269.6629213483146 hold_2_5 - 225.3521126760563 hold_2_6
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_12_nic_in_3: switch_nic_out_12_nic_in_3 = 1 ->
   - 1263.157894736842 hold_3_1 - 657.5342465753424 hold_3_2
   - 444.4444444444444 hold_3_3 - 335.6643356643356 hold_3_4
   - 269.6629213483146 hold_3_5 - 225.3521126760563 hold_3_6
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_12_nic_in_4: switch_nic_out_12_nic_in_4 = 1 ->
   - 1263.157894736842 hold_4_1 - 657.5342465753424 hold_4_2
   - 444.4444444444444 hold_4_3 - 335.6643356643356 hold_4_4
   - 269.6629213483146 hold_4_5 - 225.3521126760563 hold_4_6
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_12_nic_in_5: switch_nic_out_12_nic_in_5 = 1 ->
   - 1263.157894736842 hold_5_1 - 657.5342465753424 hold_5_2
   - 444.4444444444444 hold_5_3 - 335.6643356643356 hold_5_4
   - 269.6629213483146 hold_5_5 - 225.3521126760563 hold_5_6
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_12_nic_in_6: switch_nic_out_12_nic_in_6 = 1 ->
   - 1263.157894736842 hold_6_1 - 657.5342465753424 hold_6_2
   - 444.4444444444444 hold_6_3 - 335.6643356643356 hold_6_4
   - 269.6629213483146 hold_6_5 - 225.3521126760563 hold_6_6
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_12_nic_in_7: switch_nic_out_12_nic_in_7 = 1 ->
   - 1263.157894736842 hold_7_1 - 657.5342465753424 hold_7_2
   - 444.4444444444444 hold_7_3 - 335.6643356643356 hold_7_4
   - 269.6629213483146 hold_7_5 - 225.3521126760563 hold_7_6
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_12_nic_in_8: switch_nic_out_12_nic_in_8 = 1 ->
   - 1263.157894736842 hold_8_1 - 657.5342465753424 hold_8_2
   - 444.4444444444444 hold_8_3 - 335.6643356643356 hold_8_4
   - 269.6629213483146 hold_8_5 - 225.3521126760563 hold_8_6
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_12_nic_in_9: switch_nic_out_12_nic_in_9 = 1 ->
   - 1263.157894736842 hold_9_1 - 657.5342465753424 hold_9_2
   - 444.4444444444444 hold_9_3 - 335.6643356643356 hold_9_4
   - 269.6629213483146 hold_9_5 - 225.3521126760563 hold_9_6
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_12_nic_in_10: switch_nic_out_12_nic_in_10 = 1 ->
   - 1263.157894736842 hold_10_1 - 657.5342465753424 hold_10_2
   - 444.4444444444444 hold_10_3 - 335.6643356643356 hold_10_4
   - 269.6629213483146 hold_10_5 - 225.3521126760563 hold_10_6
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_12_nic_in_11: switch_nic_out_12_nic_in_11 = 1 ->
   - 1263.157894736842 hold_11_1 - 657.5342465753424 hold_11_2
   - 444.4444444444444 hold_11_3 - 335.6643356643356 hold_11_4
   - 269.6629213483146 hold_11_5 - 225.3521126760563 hold_11_6
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_13_nic_out_12_nic_in_13: switch_nic_out_12_nic_in_13 = 1 ->
   - 1263.157894736842 hold_13_1 - 657.5342465753424 hold_13_2
   - 444.4444444444444 hold_13_3 - 335.6643356643356 hold_13_4
   - 269.6629213483146 hold_13_5 - 225.3521126760563 hold_13_6
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_12_nic_in_14: switch_nic_out_12_nic_in_14 = 1 ->
   - 1263.157894736842 hold_14_1 - 657.5342465753424 hold_14_2
   - 444.4444444444444 hold_14_3 - 335.6643356643356 hold_14_4
   - 269.6629213483146 hold_14_5 - 225.3521126760563 hold_14_6
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_12_nic_in_15: switch_nic_out_12_nic_in_15 = 1 ->
   - 1263.157894736842 hold_15_1 - 657.5342465753424 hold_15_2
   - 444.4444444444444 hold_15_3 - 335.6643356643356 hold_15_4
   - 269.6629213483146 hold_15_5 - 225.3521126760563 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_16_nic_out_12_nic_in_16: switch_nic_out_12_nic_in_16 = 1 ->
   - 1263.157894736842 hold_16_1 - 657.5342465753424 hold_16_2
   - 444.4444444444444 hold_16_3 - 335.6643356643356 hold_16_4
   - 269.6629213483146 hold_16_5 - 225.3521126760563 hold_16_6
   + flow_nic_in_16_16 <= 0
 constr_intergroup_throughput_17_nic_out_12_nic_in_17: switch_nic_out_12_nic_in_17 = 1 ->
   - 1263.157894736842 hold_17_1 - 657.5342465753424 hold_17_2
   - 444.4444444444444 hold_17_3 - 335.6643356643356 hold_17_4
   - 269.6629213483146 hold_17_5 - 225.3521126760563 hold_17_6
   + flow_nic_in_17_17 <= 0
 constr_intergroup_throughput_18_nic_out_12_nic_in_18: switch_nic_out_12_nic_in_18 = 1 ->
   - 1263.157894736842 hold_18_1 - 657.5342465753424 hold_18_2
   - 444.4444444444444 hold_18_3 - 335.6643356643356 hold_18_4
   - 269.6629213483146 hold_18_5 - 225.3521126760563 hold_18_6
   + flow_nic_in_18_18 <= 0
 constr_intergroup_throughput_19_nic_out_12_nic_in_19: switch_nic_out_12_nic_in_19 = 1 ->
   - 1263.157894736842 hold_19_1 - 657.5342465753424 hold_19_2
   - 444.4444444444444 hold_19_3 - 335.6643356643356 hold_19_4
   - 269.6629213483146 hold_19_5 - 225.3521126760563 hold_19_6
   + flow_nic_in_19_19 <= 0
 constr_intergroup_throughput_20_nic_out_12_nic_in_20: switch_nic_out_12_nic_in_20 = 1 ->
   - 1263.157894736842 hold_20_1 - 657.5342465753424 hold_20_2
   - 444.4444444444444 hold_20_3 - 335.6643356643356 hold_20_4
   - 269.6629213483146 hold_20_5 - 225.3521126760563 hold_20_6
   + flow_nic_in_20_20 <= 0
 constr_intergroup_throughput_21_nic_out_12_nic_in_21: switch_nic_out_12_nic_in_21 = 1 ->
   - 1263.157894736842 hold_21_1 - 657.5342465753424 hold_21_2
   - 444.4444444444444 hold_21_3 - 335.6643356643356 hold_21_4
   - 269.6629213483146 hold_21_5 - 225.3521126760563 hold_21_6
   + flow_nic_in_21_21 <= 0
 constr_intergroup_throughput_22_nic_out_12_nic_in_22: switch_nic_out_12_nic_in_22 = 1 ->
   - 1263.157894736842 hold_22_1 - 657.5342465753424 hold_22_2
   - 444.4444444444444 hold_22_3 - 335.6643356643356 hold_22_4
   - 269.6629213483146 hold_22_5 - 225.3521126760563 hold_22_6
   + flow_nic_in_22_22 <= 0
 constr_intergroup_throughput_23_nic_out_12_nic_in_23: switch_nic_out_12_nic_in_23 = 1 ->
   - 1263.157894736842 hold_23_1 - 657.5342465753424 hold_23_2
   - 444.4444444444444 hold_23_3 - 335.6643356643356 hold_23_4
   - 269.6629213483146 hold_23_5 - 225.3521126760563 hold_23_6
   + flow_nic_in_23_23 <= 0
 constr_intergroup_throughput_24_nic_out_12_nic_in_24: switch_nic_out_12_nic_in_24 = 1 ->
   - 1263.157894736842 hold_24_1 - 657.5342465753424 hold_24_2
   - 444.4444444444444 hold_24_3 - 335.6643356643356 hold_24_4
   - 269.6629213483146 hold_24_5 - 225.3521126760563 hold_24_6
   + flow_nic_in_24_24 <= 0
 constr_intergroup_throughput_25_nic_out_12_nic_in_25: switch_nic_out_12_nic_in_25 = 1 ->
   - 1263.157894736842 hold_25_1 - 657.5342465753424 hold_25_2
   - 444.4444444444444 hold_25_3 - 335.6643356643356 hold_25_4
   - 269.6629213483146 hold_25_5 - 225.3521126760563 hold_25_6
   + flow_nic_in_25_25 <= 0
 constr_intergroup_throughput_26_nic_out_12_nic_in_26: switch_nic_out_12_nic_in_26 = 1 ->
   - 1263.157894736842 hold_26_1 - 657.5342465753424 hold_26_2
   - 444.4444444444444 hold_26_3 - 335.6643356643356 hold_26_4
   - 269.6629213483146 hold_26_5 - 225.3521126760563 hold_26_6
   + flow_nic_in_26_26 <= 0
 constr_intergroup_throughput_27_nic_out_12_nic_in_27: switch_nic_out_12_nic_in_27 = 1 ->
   - 1263.157894736842 hold_27_1 - 657.5342465753424 hold_27_2
   - 444.4444444444444 hold_27_3 - 335.6643356643356 hold_27_4
   - 269.6629213483146 hold_27_5 - 225.3521126760563 hold_27_6
   + flow_nic_in_27_27 <= 0
 constr_intergroup_throughput_28_nic_out_12_nic_in_28: switch_nic_out_12_nic_in_28 = 1 ->
   - 1263.157894736842 hold_28_1 - 657.5342465753424 hold_28_2
   - 444.4444444444444 hold_28_3 - 335.6643356643356 hold_28_4
   - 269.6629213483146 hold_28_5 - 225.3521126760563 hold_28_6
   + flow_nic_in_28_28 <= 0
 constr_intergroup_throughput_29_nic_out_12_nic_in_29: switch_nic_out_12_nic_in_29 = 1 ->
   - 1263.157894736842 hold_29_1 - 657.5342465753424 hold_29_2
   - 444.4444444444444 hold_29_3 - 335.6643356643356 hold_29_4
   - 269.6629213483146 hold_29_5 - 225.3521126760563 hold_29_6
   + flow_nic_in_29_29 <= 0
 constr_intergroup_throughput_30_nic_out_12_nic_in_30: switch_nic_out_12_nic_in_30 = 1 ->
   - 1263.157894736842 hold_30_1 - 657.5342465753424 hold_30_2
   - 444.4444444444444 hold_30_3 - 335.6643356643356 hold_30_4
   - 269.6629213483146 hold_30_5 - 225.3521126760563 hold_30_6
   + flow_nic_in_30_30 <= 0
 constr_intergroup_throughput_31_nic_out_12_nic_in_31: switch_nic_out_12_nic_in_31 = 1 ->
   - 1263.157894736842 hold_31_1 - 657.5342465753424 hold_31_2
   - 444.4444444444444 hold_31_3 - 335.6643356643356 hold_31_4
   - 269.6629213483146 hold_31_5 - 225.3521126760563 hold_31_6
   + flow_nic_in_31_31 <= 0
 constr_intergroup_throughput_13_source_nic_in_13: switch_source_nic_in_13 = 1 ->
   - 1348.314606741573 hold_13_1 - 679.8866855524079 hold_13_2
   - 454.5454545454545 hold_13_3 - 341.3940256045519 hold_13_4
   - 273.3485193621868 hold_13_5 - 227.9202279202279 hold_13_6
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_0_nic_out_13_nic_in_0: switch_nic_out_13_nic_in_0 = 1 ->
   - 1263.157894736842 hold_0_1 - 657.5342465753424 hold_0_2
   - 444.4444444444444 hold_0_3 - 335.6643356643356 hold_0_4
   - 269.6629213483146 hold_0_5 - 225.3521126760563 hold_0_6
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_13_nic_in_1: switch_nic_out_13_nic_in_1 = 1 ->
   - 1263.157894736842 hold_1_1 - 657.5342465753424 hold_1_2
   - 444.4444444444444 hold_1_3 - 335.6643356643356 hold_1_4
   - 269.6629213483146 hold_1_5 - 225.3521126760563 hold_1_6
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_13_nic_in_2: switch_nic_out_13_nic_in_2 = 1 ->
   - 1263.157894736842 hold_2_1 - 657.5342465753424 hold_2_2
   - 444.4444444444444 hold_2_3 - 335.6643356643356 hold_2_4
   - 269.6629213483146 hold_2_5 - 225.3521126760563 hold_2_6
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_13_nic_in_3: switch_nic_out_13_nic_in_3 = 1 ->
   - 1263.157894736842 hold_3_1 - 657.5342465753424 hold_3_2
   - 444.4444444444444 hold_3_3 - 335.6643356643356 hold_3_4
   - 269.6629213483146 hold_3_5 - 225.3521126760563 hold_3_6
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_13_nic_in_4: switch_nic_out_13_nic_in_4 = 1 ->
   - 1263.157894736842 hold_4_1 - 657.5342465753424 hold_4_2
   - 444.4444444444444 hold_4_3 - 335.6643356643356 hold_4_4
   - 269.6629213483146 hold_4_5 - 225.3521126760563 hold_4_6
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_13_nic_in_5: switch_nic_out_13_nic_in_5 = 1 ->
   - 1263.157894736842 hold_5_1 - 657.5342465753424 hold_5_2
   - 444.4444444444444 hold_5_3 - 335.6643356643356 hold_5_4
   - 269.6629213483146 hold_5_5 - 225.3521126760563 hold_5_6
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_13_nic_in_6: switch_nic_out_13_nic_in_6 = 1 ->
   - 1263.157894736842 hold_6_1 - 657.5342465753424 hold_6_2
   - 444.4444444444444 hold_6_3 - 335.6643356643356 hold_6_4
   - 269.6629213483146 hold_6_5 - 225.3521126760563 hold_6_6
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_13_nic_in_7: switch_nic_out_13_nic_in_7 = 1 ->
   - 1263.157894736842 hold_7_1 - 657.5342465753424 hold_7_2
   - 444.4444444444444 hold_7_3 - 335.6643356643356 hold_7_4
   - 269.6629213483146 hold_7_5 - 225.3521126760563 hold_7_6
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_13_nic_in_8: switch_nic_out_13_nic_in_8 = 1 ->
   - 1263.157894736842 hold_8_1 - 657.5342465753424 hold_8_2
   - 444.4444444444444 hold_8_3 - 335.6643356643356 hold_8_4
   - 269.6629213483146 hold_8_5 - 225.3521126760563 hold_8_6
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_13_nic_in_9: switch_nic_out_13_nic_in_9 = 1 ->
   - 1263.157894736842 hold_9_1 - 657.5342465753424 hold_9_2
   - 444.4444444444444 hold_9_3 - 335.6643356643356 hold_9_4
   - 269.6629213483146 hold_9_5 - 225.3521126760563 hold_9_6
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_13_nic_in_10: switch_nic_out_13_nic_in_10 = 1 ->
   - 1263.157894736842 hold_10_1 - 657.5342465753424 hold_10_2
   - 444.4444444444444 hold_10_3 - 335.6643356643356 hold_10_4
   - 269.6629213483146 hold_10_5 - 225.3521126760563 hold_10_6
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_13_nic_in_11: switch_nic_out_13_nic_in_11 = 1 ->
   - 1263.157894736842 hold_11_1 - 657.5342465753424 hold_11_2
   - 444.4444444444444 hold_11_3 - 335.6643356643356 hold_11_4
   - 269.6629213483146 hold_11_5 - 225.3521126760563 hold_11_6
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_13_nic_in_12: switch_nic_out_13_nic_in_12 = 1 ->
   - 1263.157894736842 hold_12_1 - 657.5342465753424 hold_12_2
   - 444.4444444444444 hold_12_3 - 335.6643356643356 hold_12_4
   - 269.6629213483146 hold_12_5 - 225.3521126760563 hold_12_6
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_14_nic_out_13_nic_in_14: switch_nic_out_13_nic_in_14 = 1 ->
   - 1263.157894736842 hold_14_1 - 657.5342465753424 hold_14_2
   - 444.4444444444444 hold_14_3 - 335.6643356643356 hold_14_4
   - 269.6629213483146 hold_14_5 - 225.3521126760563 hold_14_6
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_13_nic_in_15: switch_nic_out_13_nic_in_15 = 1 ->
   - 1263.157894736842 hold_15_1 - 657.5342465753424 hold_15_2
   - 444.4444444444444 hold_15_3 - 335.6643356643356 hold_15_4
   - 269.6629213483146 hold_15_5 - 225.3521126760563 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_16_nic_out_13_nic_in_16: switch_nic_out_13_nic_in_16 = 1 ->
   - 1263.157894736842 hold_16_1 - 657.5342465753424 hold_16_2
   - 444.4444444444444 hold_16_3 - 335.6643356643356 hold_16_4
   - 269.6629213483146 hold_16_5 - 225.3521126760563 hold_16_6
   + flow_nic_in_16_16 <= 0
 constr_intergroup_throughput_17_nic_out_13_nic_in_17: switch_nic_out_13_nic_in_17 = 1 ->
   - 1263.157894736842 hold_17_1 - 657.5342465753424 hold_17_2
   - 444.4444444444444 hold_17_3 - 335.6643356643356 hold_17_4
   - 269.6629213483146 hold_17_5 - 225.3521126760563 hold_17_6
   + flow_nic_in_17_17 <= 0
 constr_intergroup_throughput_18_nic_out_13_nic_in_18: switch_nic_out_13_nic_in_18 = 1 ->
   - 1263.157894736842 hold_18_1 - 657.5342465753424 hold_18_2
   - 444.4444444444444 hold_18_3 - 335.6643356643356 hold_18_4
   - 269.6629213483146 hold_18_5 - 225.3521126760563 hold_18_6
   + flow_nic_in_18_18 <= 0
 constr_intergroup_throughput_19_nic_out_13_nic_in_19: switch_nic_out_13_nic_in_19 = 1 ->
   - 1263.157894736842 hold_19_1 - 657.5342465753424 hold_19_2
   - 444.4444444444444 hold_19_3 - 335.6643356643356 hold_19_4
   - 269.6629213483146 hold_19_5 - 225.3521126760563 hold_19_6
   + flow_nic_in_19_19 <= 0
 constr_intergroup_throughput_20_nic_out_13_nic_in_20: switch_nic_out_13_nic_in_20 = 1 ->
   - 1263.157894736842 hold_20_1 - 657.5342465753424 hold_20_2
   - 444.4444444444444 hold_20_3 - 335.6643356643356 hold_20_4
   - 269.6629213483146 hold_20_5 - 225.3521126760563 hold_20_6
   + flow_nic_in_20_20 <= 0
 constr_intergroup_throughput_21_nic_out_13_nic_in_21: switch_nic_out_13_nic_in_21 = 1 ->
   - 1263.157894736842 hold_21_1 - 657.5342465753424 hold_21_2
   - 444.4444444444444 hold_21_3 - 335.6643356643356 hold_21_4
   - 269.6629213483146 hold_21_5 - 225.3521126760563 hold_21_6
   + flow_nic_in_21_21 <= 0
 constr_intergroup_throughput_22_nic_out_13_nic_in_22: switch_nic_out_13_nic_in_22 = 1 ->
   - 1263.157894736842 hold_22_1 - 657.5342465753424 hold_22_2
   - 444.4444444444444 hold_22_3 - 335.6643356643356 hold_22_4
   - 269.6629213483146 hold_22_5 - 225.3521126760563 hold_22_6
   + flow_nic_in_22_22 <= 0
 constr_intergroup_throughput_23_nic_out_13_nic_in_23: switch_nic_out_13_nic_in_23 = 1 ->
   - 1263.157894736842 hold_23_1 - 657.5342465753424 hold_23_2
   - 444.4444444444444 hold_23_3 - 335.6643356643356 hold_23_4
   - 269.6629213483146 hold_23_5 - 225.3521126760563 hold_23_6
   + flow_nic_in_23_23 <= 0
 constr_intergroup_throughput_24_nic_out_13_nic_in_24: switch_nic_out_13_nic_in_24 = 1 ->
   - 1263.157894736842 hold_24_1 - 657.5342465753424 hold_24_2
   - 444.4444444444444 hold_24_3 - 335.6643356643356 hold_24_4
   - 269.6629213483146 hold_24_5 - 225.3521126760563 hold_24_6
   + flow_nic_in_24_24 <= 0
 constr_intergroup_throughput_25_nic_out_13_nic_in_25: switch_nic_out_13_nic_in_25 = 1 ->
   - 1263.157894736842 hold_25_1 - 657.5342465753424 hold_25_2
   - 444.4444444444444 hold_25_3 - 335.6643356643356 hold_25_4
   - 269.6629213483146 hold_25_5 - 225.3521126760563 hold_25_6
   + flow_nic_in_25_25 <= 0
 constr_intergroup_throughput_26_nic_out_13_nic_in_26: switch_nic_out_13_nic_in_26 = 1 ->
   - 1263.157894736842 hold_26_1 - 657.5342465753424 hold_26_2
   - 444.4444444444444 hold_26_3 - 335.6643356643356 hold_26_4
   - 269.6629213483146 hold_26_5 - 225.3521126760563 hold_26_6
   + flow_nic_in_26_26 <= 0
 constr_intergroup_throughput_27_nic_out_13_nic_in_27: switch_nic_out_13_nic_in_27 = 1 ->
   - 1263.157894736842 hold_27_1 - 657.5342465753424 hold_27_2
   - 444.4444444444444 hold_27_3 - 335.6643356643356 hold_27_4
   - 269.6629213483146 hold_27_5 - 225.3521126760563 hold_27_6
   + flow_nic_in_27_27 <= 0
 constr_intergroup_throughput_28_nic_out_13_nic_in_28: switch_nic_out_13_nic_in_28 = 1 ->
   - 1263.157894736842 hold_28_1 - 657.5342465753424 hold_28_2
   - 444.4444444444444 hold_28_3 - 335.6643356643356 hold_28_4
   - 269.6629213483146 hold_28_5 - 225.3521126760563 hold_28_6
   + flow_nic_in_28_28 <= 0
 constr_intergroup_throughput_29_nic_out_13_nic_in_29: switch_nic_out_13_nic_in_29 = 1 ->
   - 1263.157894736842 hold_29_1 - 657.5342465753424 hold_29_2
   - 444.4444444444444 hold_29_3 - 335.6643356643356 hold_29_4
   - 269.6629213483146 hold_29_5 - 225.3521126760563 hold_29_6
   + flow_nic_in_29_29 <= 0
 constr_intergroup_throughput_30_nic_out_13_nic_in_30: switch_nic_out_13_nic_in_30 = 1 ->
   - 1263.157894736842 hold_30_1 - 657.5342465753424 hold_30_2
   - 444.4444444444444 hold_30_3 - 335.6643356643356 hold_30_4
   - 269.6629213483146 hold_30_5 - 225.3521126760563 hold_30_6
   + flow_nic_in_30_30 <= 0
 constr_intergroup_throughput_31_nic_out_13_nic_in_31: switch_nic_out_13_nic_in_31 = 1 ->
   - 1263.157894736842 hold_31_1 - 657.5342465753424 hold_31_2
   - 444.4444444444444 hold_31_3 - 335.6643356643356 hold_31_4
   - 269.6629213483146 hold_31_5 - 225.3521126760563 hold_31_6
   + flow_nic_in_31_31 <= 0
 constr_intergroup_throughput_14_source_nic_in_14: switch_source_nic_in_14 = 1 ->
   - 1348.314606741573 hold_14_1 - 679.8866855524079 hold_14_2
   - 454.5454545454545 hold_14_3 - 341.3940256045519 hold_14_4
   - 273.3485193621868 hold_14_5 - 227.9202279202279 hold_14_6
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_0_nic_out_14_nic_in_0: switch_nic_out_14_nic_in_0 = 1 ->
   - 1263.157894736842 hold_0_1 - 657.5342465753424 hold_0_2
   - 444.4444444444444 hold_0_3 - 335.6643356643356 hold_0_4
   - 269.6629213483146 hold_0_5 - 225.3521126760563 hold_0_6
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_14_nic_in_1: switch_nic_out_14_nic_in_1 = 1 ->
   - 1263.157894736842 hold_1_1 - 657.5342465753424 hold_1_2
   - 444.4444444444444 hold_1_3 - 335.6643356643356 hold_1_4
   - 269.6629213483146 hold_1_5 - 225.3521126760563 hold_1_6
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_14_nic_in_2: switch_nic_out_14_nic_in_2 = 1 ->
   - 1263.157894736842 hold_2_1 - 657.5342465753424 hold_2_2
   - 444.4444444444444 hold_2_3 - 335.6643356643356 hold_2_4
   - 269.6629213483146 hold_2_5 - 225.3521126760563 hold_2_6
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_14_nic_in_3: switch_nic_out_14_nic_in_3 = 1 ->
   - 1263.157894736842 hold_3_1 - 657.5342465753424 hold_3_2
   - 444.4444444444444 hold_3_3 - 335.6643356643356 hold_3_4
   - 269.6629213483146 hold_3_5 - 225.3521126760563 hold_3_6
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_14_nic_in_4: switch_nic_out_14_nic_in_4 = 1 ->
   - 1263.157894736842 hold_4_1 - 657.5342465753424 hold_4_2
   - 444.4444444444444 hold_4_3 - 335.6643356643356 hold_4_4
   - 269.6629213483146 hold_4_5 - 225.3521126760563 hold_4_6
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_14_nic_in_5: switch_nic_out_14_nic_in_5 = 1 ->
   - 1263.157894736842 hold_5_1 - 657.5342465753424 hold_5_2
   - 444.4444444444444 hold_5_3 - 335.6643356643356 hold_5_4
   - 269.6629213483146 hold_5_5 - 225.3521126760563 hold_5_6
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_14_nic_in_6: switch_nic_out_14_nic_in_6 = 1 ->
   - 1263.157894736842 hold_6_1 - 657.5342465753424 hold_6_2
   - 444.4444444444444 hold_6_3 - 335.6643356643356 hold_6_4
   - 269.6629213483146 hold_6_5 - 225.3521126760563 hold_6_6
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_14_nic_in_7: switch_nic_out_14_nic_in_7 = 1 ->
   - 1263.157894736842 hold_7_1 - 657.5342465753424 hold_7_2
   - 444.4444444444444 hold_7_3 - 335.6643356643356 hold_7_4
   - 269.6629213483146 hold_7_5 - 225.3521126760563 hold_7_6
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_14_nic_in_8: switch_nic_out_14_nic_in_8 = 1 ->
   - 1263.157894736842 hold_8_1 - 657.5342465753424 hold_8_2
   - 444.4444444444444 hold_8_3 - 335.6643356643356 hold_8_4
   - 269.6629213483146 hold_8_5 - 225.3521126760563 hold_8_6
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_14_nic_in_9: switch_nic_out_14_nic_in_9 = 1 ->
   - 1263.157894736842 hold_9_1 - 657.5342465753424 hold_9_2
   - 444.4444444444444 hold_9_3 - 335.6643356643356 hold_9_4
   - 269.6629213483146 hold_9_5 - 225.3521126760563 hold_9_6
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_14_nic_in_10: switch_nic_out_14_nic_in_10 = 1 ->
   - 1263.157894736842 hold_10_1 - 657.5342465753424 hold_10_2
   - 444.4444444444444 hold_10_3 - 335.6643356643356 hold_10_4
   - 269.6629213483146 hold_10_5 - 225.3521126760563 hold_10_6
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_14_nic_in_11: switch_nic_out_14_nic_in_11 = 1 ->
   - 1263.157894736842 hold_11_1 - 657.5342465753424 hold_11_2
   - 444.4444444444444 hold_11_3 - 335.6643356643356 hold_11_4
   - 269.6629213483146 hold_11_5 - 225.3521126760563 hold_11_6
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_14_nic_in_12: switch_nic_out_14_nic_in_12 = 1 ->
   - 1263.157894736842 hold_12_1 - 657.5342465753424 hold_12_2
   - 444.4444444444444 hold_12_3 - 335.6643356643356 hold_12_4
   - 269.6629213483146 hold_12_5 - 225.3521126760563 hold_12_6
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_14_nic_in_13: switch_nic_out_14_nic_in_13 = 1 ->
   - 1263.157894736842 hold_13_1 - 657.5342465753424 hold_13_2
   - 444.4444444444444 hold_13_3 - 335.6643356643356 hold_13_4
   - 269.6629213483146 hold_13_5 - 225.3521126760563 hold_13_6
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_15_nic_out_14_nic_in_15: switch_nic_out_14_nic_in_15 = 1 ->
   - 1263.157894736842 hold_15_1 - 657.5342465753424 hold_15_2
   - 444.4444444444444 hold_15_3 - 335.6643356643356 hold_15_4
   - 269.6629213483146 hold_15_5 - 225.3521126760563 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_16_nic_out_14_nic_in_16: switch_nic_out_14_nic_in_16 = 1 ->
   - 1263.157894736842 hold_16_1 - 657.5342465753424 hold_16_2
   - 444.4444444444444 hold_16_3 - 335.6643356643356 hold_16_4
   - 269.6629213483146 hold_16_5 - 225.3521126760563 hold_16_6
   + flow_nic_in_16_16 <= 0
 constr_intergroup_throughput_17_nic_out_14_nic_in_17: switch_nic_out_14_nic_in_17 = 1 ->
   - 1263.157894736842 hold_17_1 - 657.5342465753424 hold_17_2
   - 444.4444444444444 hold_17_3 - 335.6643356643356 hold_17_4
   - 269.6629213483146 hold_17_5 - 225.3521126760563 hold_17_6
   + flow_nic_in_17_17 <= 0
 constr_intergroup_throughput_18_nic_out_14_nic_in_18: switch_nic_out_14_nic_in_18 = 1 ->
   - 1263.157894736842 hold_18_1 - 657.5342465753424 hold_18_2
   - 444.4444444444444 hold_18_3 - 335.6643356643356 hold_18_4
   - 269.6629213483146 hold_18_5 - 225.3521126760563 hold_18_6
   + flow_nic_in_18_18 <= 0
 constr_intergroup_throughput_19_nic_out_14_nic_in_19: switch_nic_out_14_nic_in_19 = 1 ->
   - 1263.157894736842 hold_19_1 - 657.5342465753424 hold_19_2
   - 444.4444444444444 hold_19_3 - 335.6643356643356 hold_19_4
   - 269.6629213483146 hold_19_5 - 225.3521126760563 hold_19_6
   + flow_nic_in_19_19 <= 0
 constr_intergroup_throughput_20_nic_out_14_nic_in_20: switch_nic_out_14_nic_in_20 = 1 ->
   - 1263.157894736842 hold_20_1 - 657.5342465753424 hold_20_2
   - 444.4444444444444 hold_20_3 - 335.6643356643356 hold_20_4
   - 269.6629213483146 hold_20_5 - 225.3521126760563 hold_20_6
   + flow_nic_in_20_20 <= 0
 constr_intergroup_throughput_21_nic_out_14_nic_in_21: switch_nic_out_14_nic_in_21 = 1 ->
   - 1263.157894736842 hold_21_1 - 657.5342465753424 hold_21_2
   - 444.4444444444444 hold_21_3 - 335.6643356643356 hold_21_4
   - 269.6629213483146 hold_21_5 - 225.3521126760563 hold_21_6
   + flow_nic_in_21_21 <= 0
 constr_intergroup_throughput_22_nic_out_14_nic_in_22: switch_nic_out_14_nic_in_22 = 1 ->
   - 1263.157894736842 hold_22_1 - 657.5342465753424 hold_22_2
   - 444.4444444444444 hold_22_3 - 335.6643356643356 hold_22_4
   - 269.6629213483146 hold_22_5 - 225.3521126760563 hold_22_6
   + flow_nic_in_22_22 <= 0
 constr_intergroup_throughput_23_nic_out_14_nic_in_23: switch_nic_out_14_nic_in_23 = 1 ->
   - 1263.157894736842 hold_23_1 - 657.5342465753424 hold_23_2
   - 444.4444444444444 hold_23_3 - 335.6643356643356 hold_23_4
   - 269.6629213483146 hold_23_5 - 225.3521126760563 hold_23_6
   + flow_nic_in_23_23 <= 0
 constr_intergroup_throughput_24_nic_out_14_nic_in_24: switch_nic_out_14_nic_in_24 = 1 ->
   - 1263.157894736842 hold_24_1 - 657.5342465753424 hold_24_2
   - 444.4444444444444 hold_24_3 - 335.6643356643356 hold_24_4
   - 269.6629213483146 hold_24_5 - 225.3521126760563 hold_24_6
   + flow_nic_in_24_24 <= 0
 constr_intergroup_throughput_25_nic_out_14_nic_in_25: switch_nic_out_14_nic_in_25 = 1 ->
   - 1263.157894736842 hold_25_1 - 657.5342465753424 hold_25_2
   - 444.4444444444444 hold_25_3 - 335.6643356643356 hold_25_4
   - 269.6629213483146 hold_25_5 - 225.3521126760563 hold_25_6
   + flow_nic_in_25_25 <= 0
 constr_intergroup_throughput_26_nic_out_14_nic_in_26: switch_nic_out_14_nic_in_26 = 1 ->
   - 1263.157894736842 hold_26_1 - 657.5342465753424 hold_26_2
   - 444.4444444444444 hold_26_3 - 335.6643356643356 hold_26_4
   - 269.6629213483146 hold_26_5 - 225.3521126760563 hold_26_6
   + flow_nic_in_26_26 <= 0
 constr_intergroup_throughput_27_nic_out_14_nic_in_27: switch_nic_out_14_nic_in_27 = 1 ->
   - 1263.157894736842 hold_27_1 - 657.5342465753424 hold_27_2
   - 444.4444444444444 hold_27_3 - 335.6643356643356 hold_27_4
   - 269.6629213483146 hold_27_5 - 225.3521126760563 hold_27_6
   + flow_nic_in_27_27 <= 0
 constr_intergroup_throughput_28_nic_out_14_nic_in_28: switch_nic_out_14_nic_in_28 = 1 ->
   - 1263.157894736842 hold_28_1 - 657.5342465753424 hold_28_2
   - 444.4444444444444 hold_28_3 - 335.6643356643356 hold_28_4
   - 269.6629213483146 hold_28_5 - 225.3521126760563 hold_28_6
   + flow_nic_in_28_28 <= 0
 constr_intergroup_throughput_29_nic_out_14_nic_in_29: switch_nic_out_14_nic_in_29 = 1 ->
   - 1263.157894736842 hold_29_1 - 657.5342465753424 hold_29_2
   - 444.4444444444444 hold_29_3 - 335.6643356643356 hold_29_4
   - 269.6629213483146 hold_29_5 - 225.3521126760563 hold_29_6
   + flow_nic_in_29_29 <= 0
 constr_intergroup_throughput_30_nic_out_14_nic_in_30: switch_nic_out_14_nic_in_30 = 1 ->
   - 1263.157894736842 hold_30_1 - 657.5342465753424 hold_30_2
   - 444.4444444444444 hold_30_3 - 335.6643356643356 hold_30_4
   - 269.6629213483146 hold_30_5 - 225.3521126760563 hold_30_6
   + flow_nic_in_30_30 <= 0
 constr_intergroup_throughput_31_nic_out_14_nic_in_31: switch_nic_out_14_nic_in_31 = 1 ->
   - 1263.157894736842 hold_31_1 - 657.5342465753424 hold_31_2
   - 444.4444444444444 hold_31_3 - 335.6643356643356 hold_31_4
   - 269.6629213483146 hold_31_5 - 225.3521126760563 hold_31_6
   + flow_nic_in_31_31 <= 0
 constr_intergroup_throughput_15_source_nic_in_15: switch_source_nic_in_15 = 1 ->
   - 1348.314606741573 hold_15_1 - 679.8866855524079 hold_15_2
   - 454.5454545454545 hold_15_3 - 341.3940256045519 hold_15_4
   - 273.3485193621868 hold_15_5 - 227.9202279202279 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_0_nic_out_15_nic_in_0: switch_nic_out_15_nic_in_0 = 1 ->
   - 1263.157894736842 hold_0_1 - 657.5342465753424 hold_0_2
   - 444.4444444444444 hold_0_3 - 335.6643356643356 hold_0_4
   - 269.6629213483146 hold_0_5 - 225.3521126760563 hold_0_6
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_15_nic_in_1: switch_nic_out_15_nic_in_1 = 1 ->
   - 1263.157894736842 hold_1_1 - 657.5342465753424 hold_1_2
   - 444.4444444444444 hold_1_3 - 335.6643356643356 hold_1_4
   - 269.6629213483146 hold_1_5 - 225.3521126760563 hold_1_6
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_15_nic_in_2: switch_nic_out_15_nic_in_2 = 1 ->
   - 1263.157894736842 hold_2_1 - 657.5342465753424 hold_2_2
   - 444.4444444444444 hold_2_3 - 335.6643356643356 hold_2_4
   - 269.6629213483146 hold_2_5 - 225.3521126760563 hold_2_6
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_15_nic_in_3: switch_nic_out_15_nic_in_3 = 1 ->
   - 1263.157894736842 hold_3_1 - 657.5342465753424 hold_3_2
   - 444.4444444444444 hold_3_3 - 335.6643356643356 hold_3_4
   - 269.6629213483146 hold_3_5 - 225.3521126760563 hold_3_6
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_15_nic_in_4: switch_nic_out_15_nic_in_4 = 1 ->
   - 1263.157894736842 hold_4_1 - 657.5342465753424 hold_4_2
   - 444.4444444444444 hold_4_3 - 335.6643356643356 hold_4_4
   - 269.6629213483146 hold_4_5 - 225.3521126760563 hold_4_6
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_15_nic_in_5: switch_nic_out_15_nic_in_5 = 1 ->
   - 1263.157894736842 hold_5_1 - 657.5342465753424 hold_5_2
   - 444.4444444444444 hold_5_3 - 335.6643356643356 hold_5_4
   - 269.6629213483146 hold_5_5 - 225.3521126760563 hold_5_6
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_15_nic_in_6: switch_nic_out_15_nic_in_6 = 1 ->
   - 1263.157894736842 hold_6_1 - 657.5342465753424 hold_6_2
   - 444.4444444444444 hold_6_3 - 335.6643356643356 hold_6_4
   - 269.6629213483146 hold_6_5 - 225.3521126760563 hold_6_6
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_15_nic_in_7: switch_nic_out_15_nic_in_7 = 1 ->
   - 1263.157894736842 hold_7_1 - 657.5342465753424 hold_7_2
   - 444.4444444444444 hold_7_3 - 335.6643356643356 hold_7_4
   - 269.6629213483146 hold_7_5 - 225.3521126760563 hold_7_6
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_15_nic_in_8: switch_nic_out_15_nic_in_8 = 1 ->
   - 1263.157894736842 hold_8_1 - 657.5342465753424 hold_8_2
   - 444.4444444444444 hold_8_3 - 335.6643356643356 hold_8_4
   - 269.6629213483146 hold_8_5 - 225.3521126760563 hold_8_6
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_15_nic_in_9: switch_nic_out_15_nic_in_9 = 1 ->
   - 1263.157894736842 hold_9_1 - 657.5342465753424 hold_9_2
   - 444.4444444444444 hold_9_3 - 335.6643356643356 hold_9_4
   - 269.6629213483146 hold_9_5 - 225.3521126760563 hold_9_6
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_15_nic_in_10: switch_nic_out_15_nic_in_10 = 1 ->
   - 1263.157894736842 hold_10_1 - 657.5342465753424 hold_10_2
   - 444.4444444444444 hold_10_3 - 335.6643356643356 hold_10_4
   - 269.6629213483146 hold_10_5 - 225.3521126760563 hold_10_6
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_15_nic_in_11: switch_nic_out_15_nic_in_11 = 1 ->
   - 1263.157894736842 hold_11_1 - 657.5342465753424 hold_11_2
   - 444.4444444444444 hold_11_3 - 335.6643356643356 hold_11_4
   - 269.6629213483146 hold_11_5 - 225.3521126760563 hold_11_6
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_15_nic_in_12: switch_nic_out_15_nic_in_12 = 1 ->
   - 1263.157894736842 hold_12_1 - 657.5342465753424 hold_12_2
   - 444.4444444444444 hold_12_3 - 335.6643356643356 hold_12_4
   - 269.6629213483146 hold_12_5 - 225.3521126760563 hold_12_6
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_15_nic_in_13: switch_nic_out_15_nic_in_13 = 1 ->
   - 1263.157894736842 hold_13_1 - 657.5342465753424 hold_13_2
   - 444.4444444444444 hold_13_3 - 335.6643356643356 hold_13_4
   - 269.6629213483146 hold_13_5 - 225.3521126760563 hold_13_6
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_15_nic_in_14: switch_nic_out_15_nic_in_14 = 1 ->
   - 1263.157894736842 hold_14_1 - 657.5342465753424 hold_14_2
   - 444.4444444444444 hold_14_3 - 335.6643356643356 hold_14_4
   - 269.6629213483146 hold_14_5 - 225.3521126760563 hold_14_6
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_16_nic_out_15_nic_in_16: switch_nic_out_15_nic_in_16 = 1 ->
   - 1263.157894736842 hold_16_1 - 657.5342465753424 hold_16_2
   - 444.4444444444444 hold_16_3 - 335.6643356643356 hold_16_4
   - 269.6629213483146 hold_16_5 - 225.3521126760563 hold_16_6
   + flow_nic_in_16_16 <= 0
 constr_intergroup_throughput_17_nic_out_15_nic_in_17: switch_nic_out_15_nic_in_17 = 1 ->
   - 1263.157894736842 hold_17_1 - 657.5342465753424 hold_17_2
   - 444.4444444444444 hold_17_3 - 335.6643356643356 hold_17_4
   - 269.6629213483146 hold_17_5 - 225.3521126760563 hold_17_6
   + flow_nic_in_17_17 <= 0
 constr_intergroup_throughput_18_nic_out_15_nic_in_18: switch_nic_out_15_nic_in_18 = 1 ->
   - 1263.157894736842 hold_18_1 - 657.5342465753424 hold_18_2
   - 444.4444444444444 hold_18_3 - 335.6643356643356 hold_18_4
   - 269.6629213483146 hold_18_5 - 225.3521126760563 hold_18_6
   + flow_nic_in_18_18 <= 0
 constr_intergroup_throughput_19_nic_out_15_nic_in_19: switch_nic_out_15_nic_in_19 = 1 ->
   - 1263.157894736842 hold_19_1 - 657.5342465753424 hold_19_2
   - 444.4444444444444 hold_19_3 - 335.6643356643356 hold_19_4
   - 269.6629213483146 hold_19_5 - 225.3521126760563 hold_19_6
   + flow_nic_in_19_19 <= 0
 constr_intergroup_throughput_20_nic_out_15_nic_in_20: switch_nic_out_15_nic_in_20 = 1 ->
   - 1263.157894736842 hold_20_1 - 657.5342465753424 hold_20_2
   - 444.4444444444444 hold_20_3 - 335.6643356643356 hold_20_4
   - 269.6629213483146 hold_20_5 - 225.3521126760563 hold_20_6
   + flow_nic_in_20_20 <= 0
 constr_intergroup_throughput_21_nic_out_15_nic_in_21: switch_nic_out_15_nic_in_21 = 1 ->
   - 1263.157894736842 hold_21_1 - 657.5342465753424 hold_21_2
   - 444.4444444444444 hold_21_3 - 335.6643356643356 hold_21_4
   - 269.6629213483146 hold_21_5 - 225.3521126760563 hold_21_6
   + flow_nic_in_21_21 <= 0
 constr_intergroup_throughput_22_nic_out_15_nic_in_22: switch_nic_out_15_nic_in_22 = 1 ->
   - 1263.157894736842 hold_22_1 - 657.5342465753424 hold_22_2
   - 444.4444444444444 hold_22_3 - 335.6643356643356 hold_22_4
   - 269.6629213483146 hold_22_5 - 225.3521126760563 hold_22_6
   + flow_nic_in_22_22 <= 0
 constr_intergroup_throughput_23_nic_out_15_nic_in_23: switch_nic_out_15_nic_in_23 = 1 ->
   - 1263.157894736842 hold_23_1 - 657.5342465753424 hold_23_2
   - 444.4444444444444 hold_23_3 - 335.6643356643356 hold_23_4
   - 269.6629213483146 hold_23_5 - 225.3521126760563 hold_23_6
   + flow_nic_in_23_23 <= 0
 constr_intergroup_throughput_24_nic_out_15_nic_in_24: switch_nic_out_15_nic_in_24 = 1 ->
   - 1263.157894736842 hold_24_1 - 657.5342465753424 hold_24_2
   - 444.4444444444444 hold_24_3 - 335.6643356643356 hold_24_4
   - 269.6629213483146 hold_24_5 - 225.3521126760563 hold_24_6
   + flow_nic_in_24_24 <= 0
 constr_intergroup_throughput_25_nic_out_15_nic_in_25: switch_nic_out_15_nic_in_25 = 1 ->
   - 1263.157894736842 hold_25_1 - 657.5342465753424 hold_25_2
   - 444.4444444444444 hold_25_3 - 335.6643356643356 hold_25_4
   - 269.6629213483146 hold_25_5 - 225.3521126760563 hold_25_6
   + flow_nic_in_25_25 <= 0
 constr_intergroup_throughput_26_nic_out_15_nic_in_26: switch_nic_out_15_nic_in_26 = 1 ->
   - 1263.157894736842 hold_26_1 - 657.5342465753424 hold_26_2
   - 444.4444444444444 hold_26_3 - 335.6643356643356 hold_26_4
   - 269.6629213483146 hold_26_5 - 225.3521126760563 hold_26_6
   + flow_nic_in_26_26 <= 0
 constr_intergroup_throughput_27_nic_out_15_nic_in_27: switch_nic_out_15_nic_in_27 = 1 ->
   - 1263.157894736842 hold_27_1 - 657.5342465753424 hold_27_2
   - 444.4444444444444 hold_27_3 - 335.6643356643356 hold_27_4
   - 269.6629213483146 hold_27_5 - 225.3521126760563 hold_27_6
   + flow_nic_in_27_27 <= 0
 constr_intergroup_throughput_28_nic_out_15_nic_in_28: switch_nic_out_15_nic_in_28 = 1 ->
   - 1263.157894736842 hold_28_1 - 657.5342465753424 hold_28_2
   - 444.4444444444444 hold_28_3 - 335.6643356643356 hold_28_4
   - 269.6629213483146 hold_28_5 - 225.3521126760563 hold_28_6
   + flow_nic_in_28_28 <= 0
 constr_intergroup_throughput_29_nic_out_15_nic_in_29: switch_nic_out_15_nic_in_29 = 1 ->
   - 1263.157894736842 hold_29_1 - 657.5342465753424 hold_29_2
   - 444.4444444444444 hold_29_3 - 335.6643356643356 hold_29_4
   - 269.6629213483146 hold_29_5 - 225.3521126760563 hold_29_6
   + flow_nic_in_29_29 <= 0
 constr_intergroup_throughput_30_nic_out_15_nic_in_30: switch_nic_out_15_nic_in_30 = 1 ->
   - 1263.157894736842 hold_30_1 - 657.5342465753424 hold_30_2
   - 444.4444444444444 hold_30_3 - 335.6643356643356 hold_30_4
   - 269.6629213483146 hold_30_5 - 225.3521126760563 hold_30_6
   + flow_nic_in_30_30 <= 0
 constr_intergroup_throughput_31_nic_out_15_nic_in_31: switch_nic_out_15_nic_in_31 = 1 ->
   - 1263.157894736842 hold_31_1 - 657.5342465753424 hold_31_2
   - 444.4444444444444 hold_31_3 - 335.6643356643356 hold_31_4
   - 269.6629213483146 hold_31_5 - 225.3521126760563 hold_31_6
   + flow_nic_in_31_31 <= 0
 constr_intergroup_throughput_16_source_nic_in_16: switch_source_nic_in_16 = 1 ->
   - 1348.314606741573 hold_16_1 - 679.8866855524079 hold_16_2
   - 454.5454545454545 hold_16_3 - 341.3940256045519 hold_16_4
   - 273.3485193621868 hold_16_5 - 227.9202279202279 hold_16_6
   + flow_nic_in_16_16 <= 0
 constr_intergroup_throughput_0_nic_out_16_nic_in_0: switch_nic_out_16_nic_in_0 = 1 ->
   - 1263.157894736842 hold_0_1 - 657.5342465753424 hold_0_2
   - 444.4444444444444 hold_0_3 - 335.6643356643356 hold_0_4
   - 269.6629213483146 hold_0_5 - 225.3521126760563 hold_0_6
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_16_nic_in_1: switch_nic_out_16_nic_in_1 = 1 ->
   - 1263.157894736842 hold_1_1 - 657.5342465753424 hold_1_2
   - 444.4444444444444 hold_1_3 - 335.6643356643356 hold_1_4
   - 269.6629213483146 hold_1_5 - 225.3521126760563 hold_1_6
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_16_nic_in_2: switch_nic_out_16_nic_in_2 = 1 ->
   - 1263.157894736842 hold_2_1 - 657.5342465753424 hold_2_2
   - 444.4444444444444 hold_2_3 - 335.6643356643356 hold_2_4
   - 269.6629213483146 hold_2_5 - 225.3521126760563 hold_2_6
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_16_nic_in_3: switch_nic_out_16_nic_in_3 = 1 ->
   - 1263.157894736842 hold_3_1 - 657.5342465753424 hold_3_2
   - 444.4444444444444 hold_3_3 - 335.6643356643356 hold_3_4
   - 269.6629213483146 hold_3_5 - 225.3521126760563 hold_3_6
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_16_nic_in_4: switch_nic_out_16_nic_in_4 = 1 ->
   - 1263.157894736842 hold_4_1 - 657.5342465753424 hold_4_2
   - 444.4444444444444 hold_4_3 - 335.6643356643356 hold_4_4
   - 269.6629213483146 hold_4_5 - 225.3521126760563 hold_4_6
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_16_nic_in_5: switch_nic_out_16_nic_in_5 = 1 ->
   - 1263.157894736842 hold_5_1 - 657.5342465753424 hold_5_2
   - 444.4444444444444 hold_5_3 - 335.6643356643356 hold_5_4
   - 269.6629213483146 hold_5_5 - 225.3521126760563 hold_5_6
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_16_nic_in_6: switch_nic_out_16_nic_in_6 = 1 ->
   - 1263.157894736842 hold_6_1 - 657.5342465753424 hold_6_2
   - 444.4444444444444 hold_6_3 - 335.6643356643356 hold_6_4
   - 269.6629213483146 hold_6_5 - 225.3521126760563 hold_6_6
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_16_nic_in_7: switch_nic_out_16_nic_in_7 = 1 ->
   - 1263.157894736842 hold_7_1 - 657.5342465753424 hold_7_2
   - 444.4444444444444 hold_7_3 - 335.6643356643356 hold_7_4
   - 269.6629213483146 hold_7_5 - 225.3521126760563 hold_7_6
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_16_nic_in_8: switch_nic_out_16_nic_in_8 = 1 ->
   - 1263.157894736842 hold_8_1 - 657.5342465753424 hold_8_2
   - 444.4444444444444 hold_8_3 - 335.6643356643356 hold_8_4
   - 269.6629213483146 hold_8_5 - 225.3521126760563 hold_8_6
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_16_nic_in_9: switch_nic_out_16_nic_in_9 = 1 ->
   - 1263.157894736842 hold_9_1 - 657.5342465753424 hold_9_2
   - 444.4444444444444 hold_9_3 - 335.6643356643356 hold_9_4
   - 269.6629213483146 hold_9_5 - 225.3521126760563 hold_9_6
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_16_nic_in_10: switch_nic_out_16_nic_in_10 = 1 ->
   - 1263.157894736842 hold_10_1 - 657.5342465753424 hold_10_2
   - 444.4444444444444 hold_10_3 - 335.6643356643356 hold_10_4
   - 269.6629213483146 hold_10_5 - 225.3521126760563 hold_10_6
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_16_nic_in_11: switch_nic_out_16_nic_in_11 = 1 ->
   - 1263.157894736842 hold_11_1 - 657.5342465753424 hold_11_2
   - 444.4444444444444 hold_11_3 - 335.6643356643356 hold_11_4
   - 269.6629213483146 hold_11_5 - 225.3521126760563 hold_11_6
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_16_nic_in_12: switch_nic_out_16_nic_in_12 = 1 ->
   - 1263.157894736842 hold_12_1 - 657.5342465753424 hold_12_2
   - 444.4444444444444 hold_12_3 - 335.6643356643356 hold_12_4
   - 269.6629213483146 hold_12_5 - 225.3521126760563 hold_12_6
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_16_nic_in_13: switch_nic_out_16_nic_in_13 = 1 ->
   - 1263.157894736842 hold_13_1 - 657.5342465753424 hold_13_2
   - 444.4444444444444 hold_13_3 - 335.6643356643356 hold_13_4
   - 269.6629213483146 hold_13_5 - 225.3521126760563 hold_13_6
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_16_nic_in_14: switch_nic_out_16_nic_in_14 = 1 ->
   - 1263.157894736842 hold_14_1 - 657.5342465753424 hold_14_2
   - 444.4444444444444 hold_14_3 - 335.6643356643356 hold_14_4
   - 269.6629213483146 hold_14_5 - 225.3521126760563 hold_14_6
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_16_nic_in_15: switch_nic_out_16_nic_in_15 = 1 ->
   - 1263.157894736842 hold_15_1 - 657.5342465753424 hold_15_2
   - 444.4444444444444 hold_15_3 - 335.6643356643356 hold_15_4
   - 269.6629213483146 hold_15_5 - 225.3521126760563 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_17_nic_out_16_nic_in_17: switch_nic_out_16_nic_in_17 = 1 ->
   - 1263.157894736842 hold_17_1 - 657.5342465753424 hold_17_2
   - 444.4444444444444 hold_17_3 - 335.6643356643356 hold_17_4
   - 269.6629213483146 hold_17_5 - 225.3521126760563 hold_17_6
   + flow_nic_in_17_17 <= 0
 constr_intergroup_throughput_18_nic_out_16_nic_in_18: switch_nic_out_16_nic_in_18 = 1 ->
   - 1263.157894736842 hold_18_1 - 657.5342465753424 hold_18_2
   - 444.4444444444444 hold_18_3 - 335.6643356643356 hold_18_4
   - 269.6629213483146 hold_18_5 - 225.3521126760563 hold_18_6
   + flow_nic_in_18_18 <= 0
 constr_intergroup_throughput_19_nic_out_16_nic_in_19: switch_nic_out_16_nic_in_19 = 1 ->
   - 1263.157894736842 hold_19_1 - 657.5342465753424 hold_19_2
   - 444.4444444444444 hold_19_3 - 335.6643356643356 hold_19_4
   - 269.6629213483146 hold_19_5 - 225.3521126760563 hold_19_6
   + flow_nic_in_19_19 <= 0
 constr_intergroup_throughput_20_nic_out_16_nic_in_20: switch_nic_out_16_nic_in_20 = 1 ->
   - 1263.157894736842 hold_20_1 - 657.5342465753424 hold_20_2
   - 444.4444444444444 hold_20_3 - 335.6643356643356 hold_20_4
   - 269.6629213483146 hold_20_5 - 225.3521126760563 hold_20_6
   + flow_nic_in_20_20 <= 0
 constr_intergroup_throughput_21_nic_out_16_nic_in_21: switch_nic_out_16_nic_in_21 = 1 ->
   - 1263.157894736842 hold_21_1 - 657.5342465753424 hold_21_2
   - 444.4444444444444 hold_21_3 - 335.6643356643356 hold_21_4
   - 269.6629213483146 hold_21_5 - 225.3521126760563 hold_21_6
   + flow_nic_in_21_21 <= 0
 constr_intergroup_throughput_22_nic_out_16_nic_in_22: switch_nic_out_16_nic_in_22 = 1 ->
   - 1263.157894736842 hold_22_1 - 657.5342465753424 hold_22_2
   - 444.4444444444444 hold_22_3 - 335.6643356643356 hold_22_4
   - 269.6629213483146 hold_22_5 - 225.3521126760563 hold_22_6
   + flow_nic_in_22_22 <= 0
 constr_intergroup_throughput_23_nic_out_16_nic_in_23: switch_nic_out_16_nic_in_23 = 1 ->
   - 1263.157894736842 hold_23_1 - 657.5342465753424 hold_23_2
   - 444.4444444444444 hold_23_3 - 335.6643356643356 hold_23_4
   - 269.6629213483146 hold_23_5 - 225.3521126760563 hold_23_6
   + flow_nic_in_23_23 <= 0
 constr_intergroup_throughput_24_nic_out_16_nic_in_24: switch_nic_out_16_nic_in_24 = 1 ->
   - 1263.157894736842 hold_24_1 - 657.5342465753424 hold_24_2
   - 444.4444444444444 hold_24_3 - 335.6643356643356 hold_24_4
   - 269.6629213483146 hold_24_5 - 225.3521126760563 hold_24_6
   + flow_nic_in_24_24 <= 0
 constr_intergroup_throughput_25_nic_out_16_nic_in_25: switch_nic_out_16_nic_in_25 = 1 ->
   - 1263.157894736842 hold_25_1 - 657.5342465753424 hold_25_2
   - 444.4444444444444 hold_25_3 - 335.6643356643356 hold_25_4
   - 269.6629213483146 hold_25_5 - 225.3521126760563 hold_25_6
   + flow_nic_in_25_25 <= 0
 constr_intergroup_throughput_26_nic_out_16_nic_in_26: switch_nic_out_16_nic_in_26 = 1 ->
   - 1263.157894736842 hold_26_1 - 657.5342465753424 hold_26_2
   - 444.4444444444444 hold_26_3 - 335.6643356643356 hold_26_4
   - 269.6629213483146 hold_26_5 - 225.3521126760563 hold_26_6
   + flow_nic_in_26_26 <= 0
 constr_intergroup_throughput_27_nic_out_16_nic_in_27: switch_nic_out_16_nic_in_27 = 1 ->
   - 1263.157894736842 hold_27_1 - 657.5342465753424 hold_27_2
   - 444.4444444444444 hold_27_3 - 335.6643356643356 hold_27_4
   - 269.6629213483146 hold_27_5 - 225.3521126760563 hold_27_6
   + flow_nic_in_27_27 <= 0
 constr_intergroup_throughput_28_nic_out_16_nic_in_28: switch_nic_out_16_nic_in_28 = 1 ->
   - 1263.157894736842 hold_28_1 - 657.5342465753424 hold_28_2
   - 444.4444444444444 hold_28_3 - 335.6643356643356 hold_28_4
   - 269.6629213483146 hold_28_5 - 225.3521126760563 hold_28_6
   + flow_nic_in_28_28 <= 0
 constr_intergroup_throughput_29_nic_out_16_nic_in_29: switch_nic_out_16_nic_in_29 = 1 ->
   - 1263.157894736842 hold_29_1 - 657.5342465753424 hold_29_2
   - 444.4444444444444 hold_29_3 - 335.6643356643356 hold_29_4
   - 269.6629213483146 hold_29_5 - 225.3521126760563 hold_29_6
   + flow_nic_in_29_29 <= 0
 constr_intergroup_throughput_30_nic_out_16_nic_in_30: switch_nic_out_16_nic_in_30 = 1 ->
   - 1263.157894736842 hold_30_1 - 657.5342465753424 hold_30_2
   - 444.4444444444444 hold_30_3 - 335.6643356643356 hold_30_4
   - 269.6629213483146 hold_30_5 - 225.3521126760563 hold_30_6
   + flow_nic_in_30_30 <= 0
 constr_intergroup_throughput_31_nic_out_16_nic_in_31: switch_nic_out_16_nic_in_31 = 1 ->
   - 1263.157894736842 hold_31_1 - 657.5342465753424 hold_31_2
   - 444.4444444444444 hold_31_3 - 335.6643356643356 hold_31_4
   - 269.6629213483146 hold_31_5 - 225.3521126760563 hold_31_6
   + flow_nic_in_31_31 <= 0
 constr_intergroup_throughput_17_source_nic_in_17: switch_source_nic_in_17 = 1 ->
   - 1348.314606741573 hold_17_1 - 679.8866855524079 hold_17_2
   - 454.5454545454545 hold_17_3 - 341.3940256045519 hold_17_4
   - 273.3485193621868 hold_17_5 - 227.9202279202279 hold_17_6
   + flow_nic_in_17_17 <= 0
 constr_intergroup_throughput_0_nic_out_17_nic_in_0: switch_nic_out_17_nic_in_0 = 1 ->
   - 1263.157894736842 hold_0_1 - 657.5342465753424 hold_0_2
   - 444.4444444444444 hold_0_3 - 335.6643356643356 hold_0_4
   - 269.6629213483146 hold_0_5 - 225.3521126760563 hold_0_6
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_17_nic_in_1: switch_nic_out_17_nic_in_1 = 1 ->
   - 1263.157894736842 hold_1_1 - 657.5342465753424 hold_1_2
   - 444.4444444444444 hold_1_3 - 335.6643356643356 hold_1_4
   - 269.6629213483146 hold_1_5 - 225.3521126760563 hold_1_6
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_17_nic_in_2: switch_nic_out_17_nic_in_2 = 1 ->
   - 1263.157894736842 hold_2_1 - 657.5342465753424 hold_2_2
   - 444.4444444444444 hold_2_3 - 335.6643356643356 hold_2_4
   - 269.6629213483146 hold_2_5 - 225.3521126760563 hold_2_6
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_17_nic_in_3: switch_nic_out_17_nic_in_3 = 1 ->
   - 1263.157894736842 hold_3_1 - 657.5342465753424 hold_3_2
   - 444.4444444444444 hold_3_3 - 335.6643356643356 hold_3_4
   - 269.6629213483146 hold_3_5 - 225.3521126760563 hold_3_6
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_17_nic_in_4: switch_nic_out_17_nic_in_4 = 1 ->
   - 1263.157894736842 hold_4_1 - 657.5342465753424 hold_4_2
   - 444.4444444444444 hold_4_3 - 335.6643356643356 hold_4_4
   - 269.6629213483146 hold_4_5 - 225.3521126760563 hold_4_6
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_17_nic_in_5: switch_nic_out_17_nic_in_5 = 1 ->
   - 1263.157894736842 hold_5_1 - 657.5342465753424 hold_5_2
   - 444.4444444444444 hold_5_3 - 335.6643356643356 hold_5_4
   - 269.6629213483146 hold_5_5 - 225.3521126760563 hold_5_6
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_17_nic_in_6: switch_nic_out_17_nic_in_6 = 1 ->
   - 1263.157894736842 hold_6_1 - 657.5342465753424 hold_6_2
   - 444.4444444444444 hold_6_3 - 335.6643356643356 hold_6_4
   - 269.6629213483146 hold_6_5 - 225.3521126760563 hold_6_6
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_17_nic_in_7: switch_nic_out_17_nic_in_7 = 1 ->
   - 1263.157894736842 hold_7_1 - 657.5342465753424 hold_7_2
   - 444.4444444444444 hold_7_3 - 335.6643356643356 hold_7_4
   - 269.6629213483146 hold_7_5 - 225.3521126760563 hold_7_6
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_17_nic_in_8: switch_nic_out_17_nic_in_8 = 1 ->
   - 1263.157894736842 hold_8_1 - 657.5342465753424 hold_8_2
   - 444.4444444444444 hold_8_3 - 335.6643356643356 hold_8_4
   - 269.6629213483146 hold_8_5 - 225.3521126760563 hold_8_6
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_17_nic_in_9: switch_nic_out_17_nic_in_9 = 1 ->
   - 1263.157894736842 hold_9_1 - 657.5342465753424 hold_9_2
   - 444.4444444444444 hold_9_3 - 335.6643356643356 hold_9_4
   - 269.6629213483146 hold_9_5 - 225.3521126760563 hold_9_6
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_17_nic_in_10: switch_nic_out_17_nic_in_10 = 1 ->
   - 1263.157894736842 hold_10_1 - 657.5342465753424 hold_10_2
   - 444.4444444444444 hold_10_3 - 335.6643356643356 hold_10_4
   - 269.6629213483146 hold_10_5 - 225.3521126760563 hold_10_6
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_17_nic_in_11: switch_nic_out_17_nic_in_11 = 1 ->
   - 1263.157894736842 hold_11_1 - 657.5342465753424 hold_11_2
   - 444.4444444444444 hold_11_3 - 335.6643356643356 hold_11_4
   - 269.6629213483146 hold_11_5 - 225.3521126760563 hold_11_6
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_17_nic_in_12: switch_nic_out_17_nic_in_12 = 1 ->
   - 1263.157894736842 hold_12_1 - 657.5342465753424 hold_12_2
   - 444.4444444444444 hold_12_3 - 335.6643356643356 hold_12_4
   - 269.6629213483146 hold_12_5 - 225.3521126760563 hold_12_6
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_17_nic_in_13: switch_nic_out_17_nic_in_13 = 1 ->
   - 1263.157894736842 hold_13_1 - 657.5342465753424 hold_13_2
   - 444.4444444444444 hold_13_3 - 335.6643356643356 hold_13_4
   - 269.6629213483146 hold_13_5 - 225.3521126760563 hold_13_6
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_17_nic_in_14: switch_nic_out_17_nic_in_14 = 1 ->
   - 1263.157894736842 hold_14_1 - 657.5342465753424 hold_14_2
   - 444.4444444444444 hold_14_3 - 335.6643356643356 hold_14_4
   - 269.6629213483146 hold_14_5 - 225.3521126760563 hold_14_6
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_17_nic_in_15: switch_nic_out_17_nic_in_15 = 1 ->
   - 1263.157894736842 hold_15_1 - 657.5342465753424 hold_15_2
   - 444.4444444444444 hold_15_3 - 335.6643356643356 hold_15_4
   - 269.6629213483146 hold_15_5 - 225.3521126760563 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_16_nic_out_17_nic_in_16: switch_nic_out_17_nic_in_16 = 1 ->
   - 1263.157894736842 hold_16_1 - 657.5342465753424 hold_16_2
   - 444.4444444444444 hold_16_3 - 335.6643356643356 hold_16_4
   - 269.6629213483146 hold_16_5 - 225.3521126760563 hold_16_6
   + flow_nic_in_16_16 <= 0
 constr_intergroup_throughput_18_nic_out_17_nic_in_18: switch_nic_out_17_nic_in_18 = 1 ->
   - 1263.157894736842 hold_18_1 - 657.5342465753424 hold_18_2
   - 444.4444444444444 hold_18_3 - 335.6643356643356 hold_18_4
   - 269.6629213483146 hold_18_5 - 225.3521126760563 hold_18_6
   + flow_nic_in_18_18 <= 0
 constr_intergroup_throughput_19_nic_out_17_nic_in_19: switch_nic_out_17_nic_in_19 = 1 ->
   - 1263.157894736842 hold_19_1 - 657.5342465753424 hold_19_2
   - 444.4444444444444 hold_19_3 - 335.6643356643356 hold_19_4
   - 269.6629213483146 hold_19_5 - 225.3521126760563 hold_19_6
   + flow_nic_in_19_19 <= 0
 constr_intergroup_throughput_20_nic_out_17_nic_in_20: switch_nic_out_17_nic_in_20 = 1 ->
   - 1263.157894736842 hold_20_1 - 657.5342465753424 hold_20_2
   - 444.4444444444444 hold_20_3 - 335.6643356643356 hold_20_4
   - 269.6629213483146 hold_20_5 - 225.3521126760563 hold_20_6
   + flow_nic_in_20_20 <= 0
 constr_intergroup_throughput_21_nic_out_17_nic_in_21: switch_nic_out_17_nic_in_21 = 1 ->
   - 1263.157894736842 hold_21_1 - 657.5342465753424 hold_21_2
   - 444.4444444444444 hold_21_3 - 335.6643356643356 hold_21_4
   - 269.6629213483146 hold_21_5 - 225.3521126760563 hold_21_6
   + flow_nic_in_21_21 <= 0
 constr_intergroup_throughput_22_nic_out_17_nic_in_22: switch_nic_out_17_nic_in_22 = 1 ->
   - 1263.157894736842 hold_22_1 - 657.5342465753424 hold_22_2
   - 444.4444444444444 hold_22_3 - 335.6643356643356 hold_22_4
   - 269.6629213483146 hold_22_5 - 225.3521126760563 hold_22_6
   + flow_nic_in_22_22 <= 0
 constr_intergroup_throughput_23_nic_out_17_nic_in_23: switch_nic_out_17_nic_in_23 = 1 ->
   - 1263.157894736842 hold_23_1 - 657.5342465753424 hold_23_2
   - 444.4444444444444 hold_23_3 - 335.6643356643356 hold_23_4
   - 269.6629213483146 hold_23_5 - 225.3521126760563 hold_23_6
   + flow_nic_in_23_23 <= 0
 constr_intergroup_throughput_24_nic_out_17_nic_in_24: switch_nic_out_17_nic_in_24 = 1 ->
   - 1263.157894736842 hold_24_1 - 657.5342465753424 hold_24_2
   - 444.4444444444444 hold_24_3 - 335.6643356643356 hold_24_4
   - 269.6629213483146 hold_24_5 - 225.3521126760563 hold_24_6
   + flow_nic_in_24_24 <= 0
 constr_intergroup_throughput_25_nic_out_17_nic_in_25: switch_nic_out_17_nic_in_25 = 1 ->
   - 1263.157894736842 hold_25_1 - 657.5342465753424 hold_25_2
   - 444.4444444444444 hold_25_3 - 335.6643356643356 hold_25_4
   - 269.6629213483146 hold_25_5 - 225.3521126760563 hold_25_6
   + flow_nic_in_25_25 <= 0
 constr_intergroup_throughput_26_nic_out_17_nic_in_26: switch_nic_out_17_nic_in_26 = 1 ->
   - 1263.157894736842 hold_26_1 - 657.5342465753424 hold_26_2
   - 444.4444444444444 hold_26_3 - 335.6643356643356 hold_26_4
   - 269.6629213483146 hold_26_5 - 225.3521126760563 hold_26_6
   + flow_nic_in_26_26 <= 0
 constr_intergroup_throughput_27_nic_out_17_nic_in_27: switch_nic_out_17_nic_in_27 = 1 ->
   - 1263.157894736842 hold_27_1 - 657.5342465753424 hold_27_2
   - 444.4444444444444 hold_27_3 - 335.6643356643356 hold_27_4
   - 269.6629213483146 hold_27_5 - 225.3521126760563 hold_27_6
   + flow_nic_in_27_27 <= 0
 constr_intergroup_throughput_28_nic_out_17_nic_in_28: switch_nic_out_17_nic_in_28 = 1 ->
   - 1263.157894736842 hold_28_1 - 657.5342465753424 hold_28_2
   - 444.4444444444444 hold_28_3 - 335.6643356643356 hold_28_4
   - 269.6629213483146 hold_28_5 - 225.3521126760563 hold_28_6
   + flow_nic_in_28_28 <= 0
 constr_intergroup_throughput_29_nic_out_17_nic_in_29: switch_nic_out_17_nic_in_29 = 1 ->
   - 1263.157894736842 hold_29_1 - 657.5342465753424 hold_29_2
   - 444.4444444444444 hold_29_3 - 335.6643356643356 hold_29_4
   - 269.6629213483146 hold_29_5 - 225.3521126760563 hold_29_6
   + flow_nic_in_29_29 <= 0
 constr_intergroup_throughput_30_nic_out_17_nic_in_30: switch_nic_out_17_nic_in_30 = 1 ->
   - 1263.157894736842 hold_30_1 - 657.5342465753424 hold_30_2
   - 444.4444444444444 hold_30_3 - 335.6643356643356 hold_30_4
   - 269.6629213483146 hold_30_5 - 225.3521126760563 hold_30_6
   + flow_nic_in_30_30 <= 0
 constr_intergroup_throughput_31_nic_out_17_nic_in_31: switch_nic_out_17_nic_in_31 = 1 ->
   - 1263.157894736842 hold_31_1 - 657.5342465753424 hold_31_2
   - 444.4444444444444 hold_31_3 - 335.6643356643356 hold_31_4
   - 269.6629213483146 hold_31_5 - 225.3521126760563 hold_31_6
   + flow_nic_in_31_31 <= 0
 constr_intergroup_throughput_18_source_nic_in_18: switch_source_nic_in_18 = 1 ->
   - 1348.314606741573 hold_18_1 - 679.8866855524079 hold_18_2
   - 454.5454545454545 hold_18_3 - 341.3940256045519 hold_18_4
   - 273.3485193621868 hold_18_5 - 227.9202279202279 hold_18_6
   + flow_nic_in_18_18 <= 0
 constr_intergroup_throughput_0_nic_out_18_nic_in_0: switch_nic_out_18_nic_in_0 = 1 ->
   - 1263.157894736842 hold_0_1 - 657.5342465753424 hold_0_2
   - 444.4444444444444 hold_0_3 - 335.6643356643356 hold_0_4
   - 269.6629213483146 hold_0_5 - 225.3521126760563 hold_0_6
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_18_nic_in_1: switch_nic_out_18_nic_in_1 = 1 ->
   - 1263.157894736842 hold_1_1 - 657.5342465753424 hold_1_2
   - 444.4444444444444 hold_1_3 - 335.6643356643356 hold_1_4
   - 269.6629213483146 hold_1_5 - 225.3521126760563 hold_1_6
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_18_nic_in_2: switch_nic_out_18_nic_in_2 = 1 ->
   - 1263.157894736842 hold_2_1 - 657.5342465753424 hold_2_2
   - 444.4444444444444 hold_2_3 - 335.6643356643356 hold_2_4
   - 269.6629213483146 hold_2_5 - 225.3521126760563 hold_2_6
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_18_nic_in_3: switch_nic_out_18_nic_in_3 = 1 ->
   - 1263.157894736842 hold_3_1 - 657.5342465753424 hold_3_2
   - 444.4444444444444 hold_3_3 - 335.6643356643356 hold_3_4
   - 269.6629213483146 hold_3_5 - 225.3521126760563 hold_3_6
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_18_nic_in_4: switch_nic_out_18_nic_in_4 = 1 ->
   - 1263.157894736842 hold_4_1 - 657.5342465753424 hold_4_2
   - 444.4444444444444 hold_4_3 - 335.6643356643356 hold_4_4
   - 269.6629213483146 hold_4_5 - 225.3521126760563 hold_4_6
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_18_nic_in_5: switch_nic_out_18_nic_in_5 = 1 ->
   - 1263.157894736842 hold_5_1 - 657.5342465753424 hold_5_2
   - 444.4444444444444 hold_5_3 - 335.6643356643356 hold_5_4
   - 269.6629213483146 hold_5_5 - 225.3521126760563 hold_5_6
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_18_nic_in_6: switch_nic_out_18_nic_in_6 = 1 ->
   - 1263.157894736842 hold_6_1 - 657.5342465753424 hold_6_2
   - 444.4444444444444 hold_6_3 - 335.6643356643356 hold_6_4
   - 269.6629213483146 hold_6_5 - 225.3521126760563 hold_6_6
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_18_nic_in_7: switch_nic_out_18_nic_in_7 = 1 ->
   - 1263.157894736842 hold_7_1 - 657.5342465753424 hold_7_2
   - 444.4444444444444 hold_7_3 - 335.6643356643356 hold_7_4
   - 269.6629213483146 hold_7_5 - 225.3521126760563 hold_7_6
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_18_nic_in_8: switch_nic_out_18_nic_in_8 = 1 ->
   - 1263.157894736842 hold_8_1 - 657.5342465753424 hold_8_2
   - 444.4444444444444 hold_8_3 - 335.6643356643356 hold_8_4
   - 269.6629213483146 hold_8_5 - 225.3521126760563 hold_8_6
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_18_nic_in_9: switch_nic_out_18_nic_in_9 = 1 ->
   - 1263.157894736842 hold_9_1 - 657.5342465753424 hold_9_2
   - 444.4444444444444 hold_9_3 - 335.6643356643356 hold_9_4
   - 269.6629213483146 hold_9_5 - 225.3521126760563 hold_9_6
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_18_nic_in_10: switch_nic_out_18_nic_in_10 = 1 ->
   - 1263.157894736842 hold_10_1 - 657.5342465753424 hold_10_2
   - 444.4444444444444 hold_10_3 - 335.6643356643356 hold_10_4
   - 269.6629213483146 hold_10_5 - 225.3521126760563 hold_10_6
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_18_nic_in_11: switch_nic_out_18_nic_in_11 = 1 ->
   - 1263.157894736842 hold_11_1 - 657.5342465753424 hold_11_2
   - 444.4444444444444 hold_11_3 - 335.6643356643356 hold_11_4
   - 269.6629213483146 hold_11_5 - 225.3521126760563 hold_11_6
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_18_nic_in_12: switch_nic_out_18_nic_in_12 = 1 ->
   - 1263.157894736842 hold_12_1 - 657.5342465753424 hold_12_2
   - 444.4444444444444 hold_12_3 - 335.6643356643356 hold_12_4
   - 269.6629213483146 hold_12_5 - 225.3521126760563 hold_12_6
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_18_nic_in_13: switch_nic_out_18_nic_in_13 = 1 ->
   - 1263.157894736842 hold_13_1 - 657.5342465753424 hold_13_2
   - 444.4444444444444 hold_13_3 - 335.6643356643356 hold_13_4
   - 269.6629213483146 hold_13_5 - 225.3521126760563 hold_13_6
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_18_nic_in_14: switch_nic_out_18_nic_in_14 = 1 ->
   - 1263.157894736842 hold_14_1 - 657.5342465753424 hold_14_2
   - 444.4444444444444 hold_14_3 - 335.6643356643356 hold_14_4
   - 269.6629213483146 hold_14_5 - 225.3521126760563 hold_14_6
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_18_nic_in_15: switch_nic_out_18_nic_in_15 = 1 ->
   - 1263.157894736842 hold_15_1 - 657.5342465753424 hold_15_2
   - 444.4444444444444 hold_15_3 - 335.6643356643356 hold_15_4
   - 269.6629213483146 hold_15_5 - 225.3521126760563 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_16_nic_out_18_nic_in_16: switch_nic_out_18_nic_in_16 = 1 ->
   - 1263.157894736842 hold_16_1 - 657.5342465753424 hold_16_2
   - 444.4444444444444 hold_16_3 - 335.6643356643356 hold_16_4
   - 269.6629213483146 hold_16_5 - 225.3521126760563 hold_16_6
   + flow_nic_in_16_16 <= 0
 constr_intergroup_throughput_17_nic_out_18_nic_in_17: switch_nic_out_18_nic_in_17 = 1 ->
   - 1263.157894736842 hold_17_1 - 657.5342465753424 hold_17_2
   - 444.4444444444444 hold_17_3 - 335.6643356643356 hold_17_4
   - 269.6629213483146 hold_17_5 - 225.3521126760563 hold_17_6
   + flow_nic_in_17_17 <= 0
 constr_intergroup_throughput_19_nic_out_18_nic_in_19: switch_nic_out_18_nic_in_19 = 1 ->
   - 1263.157894736842 hold_19_1 - 657.5342465753424 hold_19_2
   - 444.4444444444444 hold_19_3 - 335.6643356643356 hold_19_4
   - 269.6629213483146 hold_19_5 - 225.3521126760563 hold_19_6
   + flow_nic_in_19_19 <= 0
 constr_intergroup_throughput_20_nic_out_18_nic_in_20: switch_nic_out_18_nic_in_20 = 1 ->
   - 1263.157894736842 hold_20_1 - 657.5342465753424 hold_20_2
   - 444.4444444444444 hold_20_3 - 335.6643356643356 hold_20_4
   - 269.6629213483146 hold_20_5 - 225.3521126760563 hold_20_6
   + flow_nic_in_20_20 <= 0
 constr_intergroup_throughput_21_nic_out_18_nic_in_21: switch_nic_out_18_nic_in_21 = 1 ->
   - 1263.157894736842 hold_21_1 - 657.5342465753424 hold_21_2
   - 444.4444444444444 hold_21_3 - 335.6643356643356 hold_21_4
   - 269.6629213483146 hold_21_5 - 225.3521126760563 hold_21_6
   + flow_nic_in_21_21 <= 0
 constr_intergroup_throughput_22_nic_out_18_nic_in_22: switch_nic_out_18_nic_in_22 = 1 ->
   - 1263.157894736842 hold_22_1 - 657.5342465753424 hold_22_2
   - 444.4444444444444 hold_22_3 - 335.6643356643356 hold_22_4
   - 269.6629213483146 hold_22_5 - 225.3521126760563 hold_22_6
   + flow_nic_in_22_22 <= 0
 constr_intergroup_throughput_23_nic_out_18_nic_in_23: switch_nic_out_18_nic_in_23 = 1 ->
   - 1263.157894736842 hold_23_1 - 657.5342465753424 hold_23_2
   - 444.4444444444444 hold_23_3 - 335.6643356643356 hold_23_4
   - 269.6629213483146 hold_23_5 - 225.3521126760563 hold_23_6
   + flow_nic_in_23_23 <= 0
 constr_intergroup_throughput_24_nic_out_18_nic_in_24: switch_nic_out_18_nic_in_24 = 1 ->
   - 1263.157894736842 hold_24_1 - 657.5342465753424 hold_24_2
   - 444.4444444444444 hold_24_3 - 335.6643356643356 hold_24_4
   - 269.6629213483146 hold_24_5 - 225.3521126760563 hold_24_6
   + flow_nic_in_24_24 <= 0
 constr_intergroup_throughput_25_nic_out_18_nic_in_25: switch_nic_out_18_nic_in_25 = 1 ->
   - 1263.157894736842 hold_25_1 - 657.5342465753424 hold_25_2
   - 444.4444444444444 hold_25_3 - 335.6643356643356 hold_25_4
   - 269.6629213483146 hold_25_5 - 225.3521126760563 hold_25_6
   + flow_nic_in_25_25 <= 0
 constr_intergroup_throughput_26_nic_out_18_nic_in_26: switch_nic_out_18_nic_in_26 = 1 ->
   - 1263.157894736842 hold_26_1 - 657.5342465753424 hold_26_2
   - 444.4444444444444 hold_26_3 - 335.6643356643356 hold_26_4
   - 269.6629213483146 hold_26_5 - 225.3521126760563 hold_26_6
   + flow_nic_in_26_26 <= 0
 constr_intergroup_throughput_27_nic_out_18_nic_in_27: switch_nic_out_18_nic_in_27 = 1 ->
   - 1263.157894736842 hold_27_1 - 657.5342465753424 hold_27_2
   - 444.4444444444444 hold_27_3 - 335.6643356643356 hold_27_4
   - 269.6629213483146 hold_27_5 - 225.3521126760563 hold_27_6
   + flow_nic_in_27_27 <= 0
 constr_intergroup_throughput_28_nic_out_18_nic_in_28: switch_nic_out_18_nic_in_28 = 1 ->
   - 1263.157894736842 hold_28_1 - 657.5342465753424 hold_28_2
   - 444.4444444444444 hold_28_3 - 335.6643356643356 hold_28_4
   - 269.6629213483146 hold_28_5 - 225.3521126760563 hold_28_6
   + flow_nic_in_28_28 <= 0
 constr_intergroup_throughput_29_nic_out_18_nic_in_29: switch_nic_out_18_nic_in_29 = 1 ->
   - 1263.157894736842 hold_29_1 - 657.5342465753424 hold_29_2
   - 444.4444444444444 hold_29_3 - 335.6643356643356 hold_29_4
   - 269.6629213483146 hold_29_5 - 225.3521126760563 hold_29_6
   + flow_nic_in_29_29 <= 0
 constr_intergroup_throughput_30_nic_out_18_nic_in_30: switch_nic_out_18_nic_in_30 = 1 ->
   - 1263.157894736842 hold_30_1 - 657.5342465753424 hold_30_2
   - 444.4444444444444 hold_30_3 - 335.6643356643356 hold_30_4
   - 269.6629213483146 hold_30_5 - 225.3521126760563 hold_30_6
   + flow_nic_in_30_30 <= 0
 constr_intergroup_throughput_31_nic_out_18_nic_in_31: switch_nic_out_18_nic_in_31 = 1 ->
   - 1263.157894736842 hold_31_1 - 657.5342465753424 hold_31_2
   - 444.4444444444444 hold_31_3 - 335.6643356643356 hold_31_4
   - 269.6629213483146 hold_31_5 - 225.3521126760563 hold_31_6
   + flow_nic_in_31_31 <= 0
 constr_intergroup_throughput_19_source_nic_in_19: switch_source_nic_in_19 = 1 ->
   - 1348.314606741573 hold_19_1 - 679.8866855524079 hold_19_2
   - 454.5454545454545 hold_19_3 - 341.3940256045519 hold_19_4
   - 273.3485193621868 hold_19_5 - 227.9202279202279 hold_19_6
   + flow_nic_in_19_19 <= 0
 constr_intergroup_throughput_0_nic_out_19_nic_in_0: switch_nic_out_19_nic_in_0 = 1 ->
   - 1263.157894736842 hold_0_1 - 657.5342465753424 hold_0_2
   - 444.4444444444444 hold_0_3 - 335.6643356643356 hold_0_4
   - 269.6629213483146 hold_0_5 - 225.3521126760563 hold_0_6
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_19_nic_in_1: switch_nic_out_19_nic_in_1 = 1 ->
   - 1263.157894736842 hold_1_1 - 657.5342465753424 hold_1_2
   - 444.4444444444444 hold_1_3 - 335.6643356643356 hold_1_4
   - 269.6629213483146 hold_1_5 - 225.3521126760563 hold_1_6
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_19_nic_in_2: switch_nic_out_19_nic_in_2 = 1 ->
   - 1263.157894736842 hold_2_1 - 657.5342465753424 hold_2_2
   - 444.4444444444444 hold_2_3 - 335.6643356643356 hold_2_4
   - 269.6629213483146 hold_2_5 - 225.3521126760563 hold_2_6
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_19_nic_in_3: switch_nic_out_19_nic_in_3 = 1 ->
   - 1263.157894736842 hold_3_1 - 657.5342465753424 hold_3_2
   - 444.4444444444444 hold_3_3 - 335.6643356643356 hold_3_4
   - 269.6629213483146 hold_3_5 - 225.3521126760563 hold_3_6
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_19_nic_in_4: switch_nic_out_19_nic_in_4 = 1 ->
   - 1263.157894736842 hold_4_1 - 657.5342465753424 hold_4_2
   - 444.4444444444444 hold_4_3 - 335.6643356643356 hold_4_4
   - 269.6629213483146 hold_4_5 - 225.3521126760563 hold_4_6
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_19_nic_in_5: switch_nic_out_19_nic_in_5 = 1 ->
   - 1263.157894736842 hold_5_1 - 657.5342465753424 hold_5_2
   - 444.4444444444444 hold_5_3 - 335.6643356643356 hold_5_4
   - 269.6629213483146 hold_5_5 - 225.3521126760563 hold_5_6
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_19_nic_in_6: switch_nic_out_19_nic_in_6 = 1 ->
   - 1263.157894736842 hold_6_1 - 657.5342465753424 hold_6_2
   - 444.4444444444444 hold_6_3 - 335.6643356643356 hold_6_4
   - 269.6629213483146 hold_6_5 - 225.3521126760563 hold_6_6
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_19_nic_in_7: switch_nic_out_19_nic_in_7 = 1 ->
   - 1263.157894736842 hold_7_1 - 657.5342465753424 hold_7_2
   - 444.4444444444444 hold_7_3 - 335.6643356643356 hold_7_4
   - 269.6629213483146 hold_7_5 - 225.3521126760563 hold_7_6
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_19_nic_in_8: switch_nic_out_19_nic_in_8 = 1 ->
   - 1263.157894736842 hold_8_1 - 657.5342465753424 hold_8_2
   - 444.4444444444444 hold_8_3 - 335.6643356643356 hold_8_4
   - 269.6629213483146 hold_8_5 - 225.3521126760563 hold_8_6
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_19_nic_in_9: switch_nic_out_19_nic_in_9 = 1 ->
   - 1263.157894736842 hold_9_1 - 657.5342465753424 hold_9_2
   - 444.4444444444444 hold_9_3 - 335.6643356643356 hold_9_4
   - 269.6629213483146 hold_9_5 - 225.3521126760563 hold_9_6
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_19_nic_in_10: switch_nic_out_19_nic_in_10 = 1 ->
   - 1263.157894736842 hold_10_1 - 657.5342465753424 hold_10_2
   - 444.4444444444444 hold_10_3 - 335.6643356643356 hold_10_4
   - 269.6629213483146 hold_10_5 - 225.3521126760563 hold_10_6
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_19_nic_in_11: switch_nic_out_19_nic_in_11 = 1 ->
   - 1263.157894736842 hold_11_1 - 657.5342465753424 hold_11_2
   - 444.4444444444444 hold_11_3 - 335.6643356643356 hold_11_4
   - 269.6629213483146 hold_11_5 - 225.3521126760563 hold_11_6
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_19_nic_in_12: switch_nic_out_19_nic_in_12 = 1 ->
   - 1263.157894736842 hold_12_1 - 657.5342465753424 hold_12_2
   - 444.4444444444444 hold_12_3 - 335.6643356643356 hold_12_4
   - 269.6629213483146 hold_12_5 - 225.3521126760563 hold_12_6
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_19_nic_in_13: switch_nic_out_19_nic_in_13 = 1 ->
   - 1263.157894736842 hold_13_1 - 657.5342465753424 hold_13_2
   - 444.4444444444444 hold_13_3 - 335.6643356643356 hold_13_4
   - 269.6629213483146 hold_13_5 - 225.3521126760563 hold_13_6
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_19_nic_in_14: switch_nic_out_19_nic_in_14 = 1 ->
   - 1263.157894736842 hold_14_1 - 657.5342465753424 hold_14_2
   - 444.4444444444444 hold_14_3 - 335.6643356643356 hold_14_4
   - 269.6629213483146 hold_14_5 - 225.3521126760563 hold_14_6
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_19_nic_in_15: switch_nic_out_19_nic_in_15 = 1 ->
   - 1263.157894736842 hold_15_1 - 657.5342465753424 hold_15_2
   - 444.4444444444444 hold_15_3 - 335.6643356643356 hold_15_4
   - 269.6629213483146 hold_15_5 - 225.3521126760563 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_16_nic_out_19_nic_in_16: switch_nic_out_19_nic_in_16 = 1 ->
   - 1263.157894736842 hold_16_1 - 657.5342465753424 hold_16_2
   - 444.4444444444444 hold_16_3 - 335.6643356643356 hold_16_4
   - 269.6629213483146 hold_16_5 - 225.3521126760563 hold_16_6
   + flow_nic_in_16_16 <= 0
 constr_intergroup_throughput_17_nic_out_19_nic_in_17: switch_nic_out_19_nic_in_17 = 1 ->
   - 1263.157894736842 hold_17_1 - 657.5342465753424 hold_17_2
   - 444.4444444444444 hold_17_3 - 335.6643356643356 hold_17_4
   - 269.6629213483146 hold_17_5 - 225.3521126760563 hold_17_6
   + flow_nic_in_17_17 <= 0
 constr_intergroup_throughput_18_nic_out_19_nic_in_18: switch_nic_out_19_nic_in_18 = 1 ->
   - 1263.157894736842 hold_18_1 - 657.5342465753424 hold_18_2
   - 444.4444444444444 hold_18_3 - 335.6643356643356 hold_18_4
   - 269.6629213483146 hold_18_5 - 225.3521126760563 hold_18_6
   + flow_nic_in_18_18 <= 0
 constr_intergroup_throughput_20_nic_out_19_nic_in_20: switch_nic_out_19_nic_in_20 = 1 ->
   - 1263.157894736842 hold_20_1 - 657.5342465753424 hold_20_2
   - 444.4444444444444 hold_20_3 - 335.6643356643356 hold_20_4
   - 269.6629213483146 hold_20_5 - 225.3521126760563 hold_20_6
   + flow_nic_in_20_20 <= 0
 constr_intergroup_throughput_21_nic_out_19_nic_in_21: switch_nic_out_19_nic_in_21 = 1 ->
   - 1263.157894736842 hold_21_1 - 657.5342465753424 hold_21_2
   - 444.4444444444444 hold_21_3 - 335.6643356643356 hold_21_4
   - 269.6629213483146 hold_21_5 - 225.3521126760563 hold_21_6
   + flow_nic_in_21_21 <= 0
 constr_intergroup_throughput_22_nic_out_19_nic_in_22: switch_nic_out_19_nic_in_22 = 1 ->
   - 1263.157894736842 hold_22_1 - 657.5342465753424 hold_22_2
   - 444.4444444444444 hold_22_3 - 335.6643356643356 hold_22_4
   - 269.6629213483146 hold_22_5 - 225.3521126760563 hold_22_6
   + flow_nic_in_22_22 <= 0
 constr_intergroup_throughput_23_nic_out_19_nic_in_23: switch_nic_out_19_nic_in_23 = 1 ->
   - 1263.157894736842 hold_23_1 - 657.5342465753424 hold_23_2
   - 444.4444444444444 hold_23_3 - 335.6643356643356 hold_23_4
   - 269.6629213483146 hold_23_5 - 225.3521126760563 hold_23_6
   + flow_nic_in_23_23 <= 0
 constr_intergroup_throughput_24_nic_out_19_nic_in_24: switch_nic_out_19_nic_in_24 = 1 ->
   - 1263.157894736842 hold_24_1 - 657.5342465753424 hold_24_2
   - 444.4444444444444 hold_24_3 - 335.6643356643356 hold_24_4
   - 269.6629213483146 hold_24_5 - 225.3521126760563 hold_24_6
   + flow_nic_in_24_24 <= 0
 constr_intergroup_throughput_25_nic_out_19_nic_in_25: switch_nic_out_19_nic_in_25 = 1 ->
   - 1263.157894736842 hold_25_1 - 657.5342465753424 hold_25_2
   - 444.4444444444444 hold_25_3 - 335.6643356643356 hold_25_4
   - 269.6629213483146 hold_25_5 - 225.3521126760563 hold_25_6
   + flow_nic_in_25_25 <= 0
 constr_intergroup_throughput_26_nic_out_19_nic_in_26: switch_nic_out_19_nic_in_26 = 1 ->
   - 1263.157894736842 hold_26_1 - 657.5342465753424 hold_26_2
   - 444.4444444444444 hold_26_3 - 335.6643356643356 hold_26_4
   - 269.6629213483146 hold_26_5 - 225.3521126760563 hold_26_6
   + flow_nic_in_26_26 <= 0
 constr_intergroup_throughput_27_nic_out_19_nic_in_27: switch_nic_out_19_nic_in_27 = 1 ->
   - 1263.157894736842 hold_27_1 - 657.5342465753424 hold_27_2
   - 444.4444444444444 hold_27_3 - 335.6643356643356 hold_27_4
   - 269.6629213483146 hold_27_5 - 225.3521126760563 hold_27_6
   + flow_nic_in_27_27 <= 0
 constr_intergroup_throughput_28_nic_out_19_nic_in_28: switch_nic_out_19_nic_in_28 = 1 ->
   - 1263.157894736842 hold_28_1 - 657.5342465753424 hold_28_2
   - 444.4444444444444 hold_28_3 - 335.6643356643356 hold_28_4
   - 269.6629213483146 hold_28_5 - 225.3521126760563 hold_28_6
   + flow_nic_in_28_28 <= 0
 constr_intergroup_throughput_29_nic_out_19_nic_in_29: switch_nic_out_19_nic_in_29 = 1 ->
   - 1263.157894736842 hold_29_1 - 657.5342465753424 hold_29_2
   - 444.4444444444444 hold_29_3 - 335.6643356643356 hold_29_4
   - 269.6629213483146 hold_29_5 - 225.3521126760563 hold_29_6
   + flow_nic_in_29_29 <= 0
 constr_intergroup_throughput_30_nic_out_19_nic_in_30: switch_nic_out_19_nic_in_30 = 1 ->
   - 1263.157894736842 hold_30_1 - 657.5342465753424 hold_30_2
   - 444.4444444444444 hold_30_3 - 335.6643356643356 hold_30_4
   - 269.6629213483146 hold_30_5 - 225.3521126760563 hold_30_6
   + flow_nic_in_30_30 <= 0
 constr_intergroup_throughput_31_nic_out_19_nic_in_31: switch_nic_out_19_nic_in_31 = 1 ->
   - 1263.157894736842 hold_31_1 - 657.5342465753424 hold_31_2
   - 444.4444444444444 hold_31_3 - 335.6643356643356 hold_31_4
   - 269.6629213483146 hold_31_5 - 225.3521126760563 hold_31_6
   + flow_nic_in_31_31 <= 0
 constr_intergroup_throughput_20_source_nic_in_20: switch_source_nic_in_20 = 1 ->
   - 1348.314606741573 hold_20_1 - 679.8866855524079 hold_20_2
   - 454.5454545454545 hold_20_3 - 341.3940256045519 hold_20_4
   - 273.3485193621868 hold_20_5 - 227.9202279202279 hold_20_6
   + flow_nic_in_20_20 <= 0
 constr_intergroup_throughput_0_nic_out_20_nic_in_0: switch_nic_out_20_nic_in_0 = 1 ->
   - 1263.157894736842 hold_0_1 - 657.5342465753424 hold_0_2
   - 444.4444444444444 hold_0_3 - 335.6643356643356 hold_0_4
   - 269.6629213483146 hold_0_5 - 225.3521126760563 hold_0_6
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_20_nic_in_1: switch_nic_out_20_nic_in_1 = 1 ->
   - 1263.157894736842 hold_1_1 - 657.5342465753424 hold_1_2
   - 444.4444444444444 hold_1_3 - 335.6643356643356 hold_1_4
   - 269.6629213483146 hold_1_5 - 225.3521126760563 hold_1_6
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_20_nic_in_2: switch_nic_out_20_nic_in_2 = 1 ->
   - 1263.157894736842 hold_2_1 - 657.5342465753424 hold_2_2
   - 444.4444444444444 hold_2_3 - 335.6643356643356 hold_2_4
   - 269.6629213483146 hold_2_5 - 225.3521126760563 hold_2_6
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_20_nic_in_3: switch_nic_out_20_nic_in_3 = 1 ->
   - 1263.157894736842 hold_3_1 - 657.5342465753424 hold_3_2
   - 444.4444444444444 hold_3_3 - 335.6643356643356 hold_3_4
   - 269.6629213483146 hold_3_5 - 225.3521126760563 hold_3_6
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_20_nic_in_4: switch_nic_out_20_nic_in_4 = 1 ->
   - 1263.157894736842 hold_4_1 - 657.5342465753424 hold_4_2
   - 444.4444444444444 hold_4_3 - 335.6643356643356 hold_4_4
   - 269.6629213483146 hold_4_5 - 225.3521126760563 hold_4_6
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_20_nic_in_5: switch_nic_out_20_nic_in_5 = 1 ->
   - 1263.157894736842 hold_5_1 - 657.5342465753424 hold_5_2
   - 444.4444444444444 hold_5_3 - 335.6643356643356 hold_5_4
   - 269.6629213483146 hold_5_5 - 225.3521126760563 hold_5_6
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_20_nic_in_6: switch_nic_out_20_nic_in_6 = 1 ->
   - 1263.157894736842 hold_6_1 - 657.5342465753424 hold_6_2
   - 444.4444444444444 hold_6_3 - 335.6643356643356 hold_6_4
   - 269.6629213483146 hold_6_5 - 225.3521126760563 hold_6_6
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_20_nic_in_7: switch_nic_out_20_nic_in_7 = 1 ->
   - 1263.157894736842 hold_7_1 - 657.5342465753424 hold_7_2
   - 444.4444444444444 hold_7_3 - 335.6643356643356 hold_7_4
   - 269.6629213483146 hold_7_5 - 225.3521126760563 hold_7_6
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_20_nic_in_8: switch_nic_out_20_nic_in_8 = 1 ->
   - 1263.157894736842 hold_8_1 - 657.5342465753424 hold_8_2
   - 444.4444444444444 hold_8_3 - 335.6643356643356 hold_8_4
   - 269.6629213483146 hold_8_5 - 225.3521126760563 hold_8_6
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_20_nic_in_9: switch_nic_out_20_nic_in_9 = 1 ->
   - 1263.157894736842 hold_9_1 - 657.5342465753424 hold_9_2
   - 444.4444444444444 hold_9_3 - 335.6643356643356 hold_9_4
   - 269.6629213483146 hold_9_5 - 225.3521126760563 hold_9_6
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_20_nic_in_10: switch_nic_out_20_nic_in_10 = 1 ->
   - 1263.157894736842 hold_10_1 - 657.5342465753424 hold_10_2
   - 444.4444444444444 hold_10_3 - 335.6643356643356 hold_10_4
   - 269.6629213483146 hold_10_5 - 225.3521126760563 hold_10_6
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_20_nic_in_11: switch_nic_out_20_nic_in_11 = 1 ->
   - 1263.157894736842 hold_11_1 - 657.5342465753424 hold_11_2
   - 444.4444444444444 hold_11_3 - 335.6643356643356 hold_11_4
   - 269.6629213483146 hold_11_5 - 225.3521126760563 hold_11_6
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_20_nic_in_12: switch_nic_out_20_nic_in_12 = 1 ->
   - 1263.157894736842 hold_12_1 - 657.5342465753424 hold_12_2
   - 444.4444444444444 hold_12_3 - 335.6643356643356 hold_12_4
   - 269.6629213483146 hold_12_5 - 225.3521126760563 hold_12_6
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_20_nic_in_13: switch_nic_out_20_nic_in_13 = 1 ->
   - 1263.157894736842 hold_13_1 - 657.5342465753424 hold_13_2
   - 444.4444444444444 hold_13_3 - 335.6643356643356 hold_13_4
   - 269.6629213483146 hold_13_5 - 225.3521126760563 hold_13_6
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_20_nic_in_14: switch_nic_out_20_nic_in_14 = 1 ->
   - 1263.157894736842 hold_14_1 - 657.5342465753424 hold_14_2
   - 444.4444444444444 hold_14_3 - 335.6643356643356 hold_14_4
   - 269.6629213483146 hold_14_5 - 225.3521126760563 hold_14_6
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_20_nic_in_15: switch_nic_out_20_nic_in_15 = 1 ->
   - 1263.157894736842 hold_15_1 - 657.5342465753424 hold_15_2
   - 444.4444444444444 hold_15_3 - 335.6643356643356 hold_15_4
   - 269.6629213483146 hold_15_5 - 225.3521126760563 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_16_nic_out_20_nic_in_16: switch_nic_out_20_nic_in_16 = 1 ->
   - 1263.157894736842 hold_16_1 - 657.5342465753424 hold_16_2
   - 444.4444444444444 hold_16_3 - 335.6643356643356 hold_16_4
   - 269.6629213483146 hold_16_5 - 225.3521126760563 hold_16_6
   + flow_nic_in_16_16 <= 0
 constr_intergroup_throughput_17_nic_out_20_nic_in_17: switch_nic_out_20_nic_in_17 = 1 ->
   - 1263.157894736842 hold_17_1 - 657.5342465753424 hold_17_2
   - 444.4444444444444 hold_17_3 - 335.6643356643356 hold_17_4
   - 269.6629213483146 hold_17_5 - 225.3521126760563 hold_17_6
   + flow_nic_in_17_17 <= 0
 constr_intergroup_throughput_18_nic_out_20_nic_in_18: switch_nic_out_20_nic_in_18 = 1 ->
   - 1263.157894736842 hold_18_1 - 657.5342465753424 hold_18_2
   - 444.4444444444444 hold_18_3 - 335.6643356643356 hold_18_4
   - 269.6629213483146 hold_18_5 - 225.3521126760563 hold_18_6
   + flow_nic_in_18_18 <= 0
 constr_intergroup_throughput_19_nic_out_20_nic_in_19: switch_nic_out_20_nic_in_19 = 1 ->
   - 1263.157894736842 hold_19_1 - 657.5342465753424 hold_19_2
   - 444.4444444444444 hold_19_3 - 335.6643356643356 hold_19_4
   - 269.6629213483146 hold_19_5 - 225.3521126760563 hold_19_6
   + flow_nic_in_19_19 <= 0
 constr_intergroup_throughput_21_nic_out_20_nic_in_21: switch_nic_out_20_nic_in_21 = 1 ->
   - 1263.157894736842 hold_21_1 - 657.5342465753424 hold_21_2
   - 444.4444444444444 hold_21_3 - 335.6643356643356 hold_21_4
   - 269.6629213483146 hold_21_5 - 225.3521126760563 hold_21_6
   + flow_nic_in_21_21 <= 0
 constr_intergroup_throughput_22_nic_out_20_nic_in_22: switch_nic_out_20_nic_in_22 = 1 ->
   - 1263.157894736842 hold_22_1 - 657.5342465753424 hold_22_2
   - 444.4444444444444 hold_22_3 - 335.6643356643356 hold_22_4
   - 269.6629213483146 hold_22_5 - 225.3521126760563 hold_22_6
   + flow_nic_in_22_22 <= 0
 constr_intergroup_throughput_23_nic_out_20_nic_in_23: switch_nic_out_20_nic_in_23 = 1 ->
   - 1263.157894736842 hold_23_1 - 657.5342465753424 hold_23_2
   - 444.4444444444444 hold_23_3 - 335.6643356643356 hold_23_4
   - 269.6629213483146 hold_23_5 - 225.3521126760563 hold_23_6
   + flow_nic_in_23_23 <= 0
 constr_intergroup_throughput_24_nic_out_20_nic_in_24: switch_nic_out_20_nic_in_24 = 1 ->
   - 1263.157894736842 hold_24_1 - 657.5342465753424 hold_24_2
   - 444.4444444444444 hold_24_3 - 335.6643356643356 hold_24_4
   - 269.6629213483146 hold_24_5 - 225.3521126760563 hold_24_6
   + flow_nic_in_24_24 <= 0
 constr_intergroup_throughput_25_nic_out_20_nic_in_25: switch_nic_out_20_nic_in_25 = 1 ->
   - 1263.157894736842 hold_25_1 - 657.5342465753424 hold_25_2
   - 444.4444444444444 hold_25_3 - 335.6643356643356 hold_25_4
   - 269.6629213483146 hold_25_5 - 225.3521126760563 hold_25_6
   + flow_nic_in_25_25 <= 0
 constr_intergroup_throughput_26_nic_out_20_nic_in_26: switch_nic_out_20_nic_in_26 = 1 ->
   - 1263.157894736842 hold_26_1 - 657.5342465753424 hold_26_2
   - 444.4444444444444 hold_26_3 - 335.6643356643356 hold_26_4
   - 269.6629213483146 hold_26_5 - 225.3521126760563 hold_26_6
   + flow_nic_in_26_26 <= 0
 constr_intergroup_throughput_27_nic_out_20_nic_in_27: switch_nic_out_20_nic_in_27 = 1 ->
   - 1263.157894736842 hold_27_1 - 657.5342465753424 hold_27_2
   - 444.4444444444444 hold_27_3 - 335.6643356643356 hold_27_4
   - 269.6629213483146 hold_27_5 - 225.3521126760563 hold_27_6
   + flow_nic_in_27_27 <= 0
 constr_intergroup_throughput_28_nic_out_20_nic_in_28: switch_nic_out_20_nic_in_28 = 1 ->
   - 1263.157894736842 hold_28_1 - 657.5342465753424 hold_28_2
   - 444.4444444444444 hold_28_3 - 335.6643356643356 hold_28_4
   - 269.6629213483146 hold_28_5 - 225.3521126760563 hold_28_6
   + flow_nic_in_28_28 <= 0
 constr_intergroup_throughput_29_nic_out_20_nic_in_29: switch_nic_out_20_nic_in_29 = 1 ->
   - 1263.157894736842 hold_29_1 - 657.5342465753424 hold_29_2
   - 444.4444444444444 hold_29_3 - 335.6643356643356 hold_29_4
   - 269.6629213483146 hold_29_5 - 225.3521126760563 hold_29_6
   + flow_nic_in_29_29 <= 0
 constr_intergroup_throughput_30_nic_out_20_nic_in_30: switch_nic_out_20_nic_in_30 = 1 ->
   - 1263.157894736842 hold_30_1 - 657.5342465753424 hold_30_2
   - 444.4444444444444 hold_30_3 - 335.6643356643356 hold_30_4
   - 269.6629213483146 hold_30_5 - 225.3521126760563 hold_30_6
   + flow_nic_in_30_30 <= 0
 constr_intergroup_throughput_31_nic_out_20_nic_in_31: switch_nic_out_20_nic_in_31 = 1 ->
   - 1263.157894736842 hold_31_1 - 657.5342465753424 hold_31_2
   - 444.4444444444444 hold_31_3 - 335.6643356643356 hold_31_4
   - 269.6629213483146 hold_31_5 - 225.3521126760563 hold_31_6
   + flow_nic_in_31_31 <= 0
 constr_intergroup_throughput_21_source_nic_in_21: switch_source_nic_in_21 = 1 ->
   - 1348.314606741573 hold_21_1 - 679.8866855524079 hold_21_2
   - 454.5454545454545 hold_21_3 - 341.3940256045519 hold_21_4
   - 273.3485193621868 hold_21_5 - 227.9202279202279 hold_21_6
   + flow_nic_in_21_21 <= 0
 constr_intergroup_throughput_0_nic_out_21_nic_in_0: switch_nic_out_21_nic_in_0 = 1 ->
   - 1263.157894736842 hold_0_1 - 657.5342465753424 hold_0_2
   - 444.4444444444444 hold_0_3 - 335.6643356643356 hold_0_4
   - 269.6629213483146 hold_0_5 - 225.3521126760563 hold_0_6
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_21_nic_in_1: switch_nic_out_21_nic_in_1 = 1 ->
   - 1263.157894736842 hold_1_1 - 657.5342465753424 hold_1_2
   - 444.4444444444444 hold_1_3 - 335.6643356643356 hold_1_4
   - 269.6629213483146 hold_1_5 - 225.3521126760563 hold_1_6
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_21_nic_in_2: switch_nic_out_21_nic_in_2 = 1 ->
   - 1263.157894736842 hold_2_1 - 657.5342465753424 hold_2_2
   - 444.4444444444444 hold_2_3 - 335.6643356643356 hold_2_4
   - 269.6629213483146 hold_2_5 - 225.3521126760563 hold_2_6
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_21_nic_in_3: switch_nic_out_21_nic_in_3 = 1 ->
   - 1263.157894736842 hold_3_1 - 657.5342465753424 hold_3_2
   - 444.4444444444444 hold_3_3 - 335.6643356643356 hold_3_4
   - 269.6629213483146 hold_3_5 - 225.3521126760563 hold_3_6
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_21_nic_in_4: switch_nic_out_21_nic_in_4 = 1 ->
   - 1263.157894736842 hold_4_1 - 657.5342465753424 hold_4_2
   - 444.4444444444444 hold_4_3 - 335.6643356643356 hold_4_4
   - 269.6629213483146 hold_4_5 - 225.3521126760563 hold_4_6
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_21_nic_in_5: switch_nic_out_21_nic_in_5 = 1 ->
   - 1263.157894736842 hold_5_1 - 657.5342465753424 hold_5_2
   - 444.4444444444444 hold_5_3 - 335.6643356643356 hold_5_4
   - 269.6629213483146 hold_5_5 - 225.3521126760563 hold_5_6
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_21_nic_in_6: switch_nic_out_21_nic_in_6 = 1 ->
   - 1263.157894736842 hold_6_1 - 657.5342465753424 hold_6_2
   - 444.4444444444444 hold_6_3 - 335.6643356643356 hold_6_4
   - 269.6629213483146 hold_6_5 - 225.3521126760563 hold_6_6
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_21_nic_in_7: switch_nic_out_21_nic_in_7 = 1 ->
   - 1263.157894736842 hold_7_1 - 657.5342465753424 hold_7_2
   - 444.4444444444444 hold_7_3 - 335.6643356643356 hold_7_4
   - 269.6629213483146 hold_7_5 - 225.3521126760563 hold_7_6
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_21_nic_in_8: switch_nic_out_21_nic_in_8 = 1 ->
   - 1263.157894736842 hold_8_1 - 657.5342465753424 hold_8_2
   - 444.4444444444444 hold_8_3 - 335.6643356643356 hold_8_4
   - 269.6629213483146 hold_8_5 - 225.3521126760563 hold_8_6
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_21_nic_in_9: switch_nic_out_21_nic_in_9 = 1 ->
   - 1263.157894736842 hold_9_1 - 657.5342465753424 hold_9_2
   - 444.4444444444444 hold_9_3 - 335.6643356643356 hold_9_4
   - 269.6629213483146 hold_9_5 - 225.3521126760563 hold_9_6
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_21_nic_in_10: switch_nic_out_21_nic_in_10 = 1 ->
   - 1263.157894736842 hold_10_1 - 657.5342465753424 hold_10_2
   - 444.4444444444444 hold_10_3 - 335.6643356643356 hold_10_4
   - 269.6629213483146 hold_10_5 - 225.3521126760563 hold_10_6
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_21_nic_in_11: switch_nic_out_21_nic_in_11 = 1 ->
   - 1263.157894736842 hold_11_1 - 657.5342465753424 hold_11_2
   - 444.4444444444444 hold_11_3 - 335.6643356643356 hold_11_4
   - 269.6629213483146 hold_11_5 - 225.3521126760563 hold_11_6
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_21_nic_in_12: switch_nic_out_21_nic_in_12 = 1 ->
   - 1263.157894736842 hold_12_1 - 657.5342465753424 hold_12_2
   - 444.4444444444444 hold_12_3 - 335.6643356643356 hold_12_4
   - 269.6629213483146 hold_12_5 - 225.3521126760563 hold_12_6
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_21_nic_in_13: switch_nic_out_21_nic_in_13 = 1 ->
   - 1263.157894736842 hold_13_1 - 657.5342465753424 hold_13_2
   - 444.4444444444444 hold_13_3 - 335.6643356643356 hold_13_4
   - 269.6629213483146 hold_13_5 - 225.3521126760563 hold_13_6
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_21_nic_in_14: switch_nic_out_21_nic_in_14 = 1 ->
   - 1263.157894736842 hold_14_1 - 657.5342465753424 hold_14_2
   - 444.4444444444444 hold_14_3 - 335.6643356643356 hold_14_4
   - 269.6629213483146 hold_14_5 - 225.3521126760563 hold_14_6
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_21_nic_in_15: switch_nic_out_21_nic_in_15 = 1 ->
   - 1263.157894736842 hold_15_1 - 657.5342465753424 hold_15_2
   - 444.4444444444444 hold_15_3 - 335.6643356643356 hold_15_4
   - 269.6629213483146 hold_15_5 - 225.3521126760563 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_16_nic_out_21_nic_in_16: switch_nic_out_21_nic_in_16 = 1 ->
   - 1263.157894736842 hold_16_1 - 657.5342465753424 hold_16_2
   - 444.4444444444444 hold_16_3 - 335.6643356643356 hold_16_4
   - 269.6629213483146 hold_16_5 - 225.3521126760563 hold_16_6
   + flow_nic_in_16_16 <= 0
 constr_intergroup_throughput_17_nic_out_21_nic_in_17: switch_nic_out_21_nic_in_17 = 1 ->
   - 1263.157894736842 hold_17_1 - 657.5342465753424 hold_17_2
   - 444.4444444444444 hold_17_3 - 335.6643356643356 hold_17_4
   - 269.6629213483146 hold_17_5 - 225.3521126760563 hold_17_6
   + flow_nic_in_17_17 <= 0
 constr_intergroup_throughput_18_nic_out_21_nic_in_18: switch_nic_out_21_nic_in_18 = 1 ->
   - 1263.157894736842 hold_18_1 - 657.5342465753424 hold_18_2
   - 444.4444444444444 hold_18_3 - 335.6643356643356 hold_18_4
   - 269.6629213483146 hold_18_5 - 225.3521126760563 hold_18_6
   + flow_nic_in_18_18 <= 0
 constr_intergroup_throughput_19_nic_out_21_nic_in_19: switch_nic_out_21_nic_in_19 = 1 ->
   - 1263.157894736842 hold_19_1 - 657.5342465753424 hold_19_2
   - 444.4444444444444 hold_19_3 - 335.6643356643356 hold_19_4
   - 269.6629213483146 hold_19_5 - 225.3521126760563 hold_19_6
   + flow_nic_in_19_19 <= 0
 constr_intergroup_throughput_20_nic_out_21_nic_in_20: switch_nic_out_21_nic_in_20 = 1 ->
   - 1263.157894736842 hold_20_1 - 657.5342465753424 hold_20_2
   - 444.4444444444444 hold_20_3 - 335.6643356643356 hold_20_4
   - 269.6629213483146 hold_20_5 - 225.3521126760563 hold_20_6
   + flow_nic_in_20_20 <= 0
 constr_intergroup_throughput_22_nic_out_21_nic_in_22: switch_nic_out_21_nic_in_22 = 1 ->
   - 1263.157894736842 hold_22_1 - 657.5342465753424 hold_22_2
   - 444.4444444444444 hold_22_3 - 335.6643356643356 hold_22_4
   - 269.6629213483146 hold_22_5 - 225.3521126760563 hold_22_6
   + flow_nic_in_22_22 <= 0
 constr_intergroup_throughput_23_nic_out_21_nic_in_23: switch_nic_out_21_nic_in_23 = 1 ->
   - 1263.157894736842 hold_23_1 - 657.5342465753424 hold_23_2
   - 444.4444444444444 hold_23_3 - 335.6643356643356 hold_23_4
   - 269.6629213483146 hold_23_5 - 225.3521126760563 hold_23_6
   + flow_nic_in_23_23 <= 0
 constr_intergroup_throughput_24_nic_out_21_nic_in_24: switch_nic_out_21_nic_in_24 = 1 ->
   - 1263.157894736842 hold_24_1 - 657.5342465753424 hold_24_2
   - 444.4444444444444 hold_24_3 - 335.6643356643356 hold_24_4
   - 269.6629213483146 hold_24_5 - 225.3521126760563 hold_24_6
   + flow_nic_in_24_24 <= 0
 constr_intergroup_throughput_25_nic_out_21_nic_in_25: switch_nic_out_21_nic_in_25 = 1 ->
   - 1263.157894736842 hold_25_1 - 657.5342465753424 hold_25_2
   - 444.4444444444444 hold_25_3 - 335.6643356643356 hold_25_4
   - 269.6629213483146 hold_25_5 - 225.3521126760563 hold_25_6
   + flow_nic_in_25_25 <= 0
 constr_intergroup_throughput_26_nic_out_21_nic_in_26: switch_nic_out_21_nic_in_26 = 1 ->
   - 1263.157894736842 hold_26_1 - 657.5342465753424 hold_26_2
   - 444.4444444444444 hold_26_3 - 335.6643356643356 hold_26_4
   - 269.6629213483146 hold_26_5 - 225.3521126760563 hold_26_6
   + flow_nic_in_26_26 <= 0
 constr_intergroup_throughput_27_nic_out_21_nic_in_27: switch_nic_out_21_nic_in_27 = 1 ->
   - 1263.157894736842 hold_27_1 - 657.5342465753424 hold_27_2
   - 444.4444444444444 hold_27_3 - 335.6643356643356 hold_27_4
   - 269.6629213483146 hold_27_5 - 225.3521126760563 hold_27_6
   + flow_nic_in_27_27 <= 0
 constr_intergroup_throughput_28_nic_out_21_nic_in_28: switch_nic_out_21_nic_in_28 = 1 ->
   - 1263.157894736842 hold_28_1 - 657.5342465753424 hold_28_2
   - 444.4444444444444 hold_28_3 - 335.6643356643356 hold_28_4
   - 269.6629213483146 hold_28_5 - 225.3521126760563 hold_28_6
   + flow_nic_in_28_28 <= 0
 constr_intergroup_throughput_29_nic_out_21_nic_in_29: switch_nic_out_21_nic_in_29 = 1 ->
   - 1263.157894736842 hold_29_1 - 657.5342465753424 hold_29_2
   - 444.4444444444444 hold_29_3 - 335.6643356643356 hold_29_4
   - 269.6629213483146 hold_29_5 - 225.3521126760563 hold_29_6
   + flow_nic_in_29_29 <= 0
 constr_intergroup_throughput_30_nic_out_21_nic_in_30: switch_nic_out_21_nic_in_30 = 1 ->
   - 1263.157894736842 hold_30_1 - 657.5342465753424 hold_30_2
   - 444.4444444444444 hold_30_3 - 335.6643356643356 hold_30_4
   - 269.6629213483146 hold_30_5 - 225.3521126760563 hold_30_6
   + flow_nic_in_30_30 <= 0
 constr_intergroup_throughput_31_nic_out_21_nic_in_31: switch_nic_out_21_nic_in_31 = 1 ->
   - 1263.157894736842 hold_31_1 - 657.5342465753424 hold_31_2
   - 444.4444444444444 hold_31_3 - 335.6643356643356 hold_31_4
   - 269.6629213483146 hold_31_5 - 225.3521126760563 hold_31_6
   + flow_nic_in_31_31 <= 0
 constr_intergroup_throughput_22_source_nic_in_22: switch_source_nic_in_22 = 1 ->
   - 1348.314606741573 hold_22_1 - 679.8866855524079 hold_22_2
   - 454.5454545454545 hold_22_3 - 341.3940256045519 hold_22_4
   - 273.3485193621868 hold_22_5 - 227.9202279202279 hold_22_6
   + flow_nic_in_22_22 <= 0
 constr_intergroup_throughput_0_nic_out_22_nic_in_0: switch_nic_out_22_nic_in_0 = 1 ->
   - 1263.157894736842 hold_0_1 - 657.5342465753424 hold_0_2
   - 444.4444444444444 hold_0_3 - 335.6643356643356 hold_0_4
   - 269.6629213483146 hold_0_5 - 225.3521126760563 hold_0_6
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_22_nic_in_1: switch_nic_out_22_nic_in_1 = 1 ->
   - 1263.157894736842 hold_1_1 - 657.5342465753424 hold_1_2
   - 444.4444444444444 hold_1_3 - 335.6643356643356 hold_1_4
   - 269.6629213483146 hold_1_5 - 225.3521126760563 hold_1_6
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_22_nic_in_2: switch_nic_out_22_nic_in_2 = 1 ->
   - 1263.157894736842 hold_2_1 - 657.5342465753424 hold_2_2
   - 444.4444444444444 hold_2_3 - 335.6643356643356 hold_2_4
   - 269.6629213483146 hold_2_5 - 225.3521126760563 hold_2_6
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_22_nic_in_3: switch_nic_out_22_nic_in_3 = 1 ->
   - 1263.157894736842 hold_3_1 - 657.5342465753424 hold_3_2
   - 444.4444444444444 hold_3_3 - 335.6643356643356 hold_3_4
   - 269.6629213483146 hold_3_5 - 225.3521126760563 hold_3_6
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_22_nic_in_4: switch_nic_out_22_nic_in_4 = 1 ->
   - 1263.157894736842 hold_4_1 - 657.5342465753424 hold_4_2
   - 444.4444444444444 hold_4_3 - 335.6643356643356 hold_4_4
   - 269.6629213483146 hold_4_5 - 225.3521126760563 hold_4_6
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_22_nic_in_5: switch_nic_out_22_nic_in_5 = 1 ->
   - 1263.157894736842 hold_5_1 - 657.5342465753424 hold_5_2
   - 444.4444444444444 hold_5_3 - 335.6643356643356 hold_5_4
   - 269.6629213483146 hold_5_5 - 225.3521126760563 hold_5_6
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_22_nic_in_6: switch_nic_out_22_nic_in_6 = 1 ->
   - 1263.157894736842 hold_6_1 - 657.5342465753424 hold_6_2
   - 444.4444444444444 hold_6_3 - 335.6643356643356 hold_6_4
   - 269.6629213483146 hold_6_5 - 225.3521126760563 hold_6_6
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_22_nic_in_7: switch_nic_out_22_nic_in_7 = 1 ->
   - 1263.157894736842 hold_7_1 - 657.5342465753424 hold_7_2
   - 444.4444444444444 hold_7_3 - 335.6643356643356 hold_7_4
   - 269.6629213483146 hold_7_5 - 225.3521126760563 hold_7_6
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_22_nic_in_8: switch_nic_out_22_nic_in_8 = 1 ->
   - 1263.157894736842 hold_8_1 - 657.5342465753424 hold_8_2
   - 444.4444444444444 hold_8_3 - 335.6643356643356 hold_8_4
   - 269.6629213483146 hold_8_5 - 225.3521126760563 hold_8_6
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_22_nic_in_9: switch_nic_out_22_nic_in_9 = 1 ->
   - 1263.157894736842 hold_9_1 - 657.5342465753424 hold_9_2
   - 444.4444444444444 hold_9_3 - 335.6643356643356 hold_9_4
   - 269.6629213483146 hold_9_5 - 225.3521126760563 hold_9_6
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_22_nic_in_10: switch_nic_out_22_nic_in_10 = 1 ->
   - 1263.157894736842 hold_10_1 - 657.5342465753424 hold_10_2
   - 444.4444444444444 hold_10_3 - 335.6643356643356 hold_10_4
   - 269.6629213483146 hold_10_5 - 225.3521126760563 hold_10_6
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_22_nic_in_11: switch_nic_out_22_nic_in_11 = 1 ->
   - 1263.157894736842 hold_11_1 - 657.5342465753424 hold_11_2
   - 444.4444444444444 hold_11_3 - 335.6643356643356 hold_11_4
   - 269.6629213483146 hold_11_5 - 225.3521126760563 hold_11_6
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_22_nic_in_12: switch_nic_out_22_nic_in_12 = 1 ->
   - 1263.157894736842 hold_12_1 - 657.5342465753424 hold_12_2
   - 444.4444444444444 hold_12_3 - 335.6643356643356 hold_12_4
   - 269.6629213483146 hold_12_5 - 225.3521126760563 hold_12_6
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_22_nic_in_13: switch_nic_out_22_nic_in_13 = 1 ->
   - 1263.157894736842 hold_13_1 - 657.5342465753424 hold_13_2
   - 444.4444444444444 hold_13_3 - 335.6643356643356 hold_13_4
   - 269.6629213483146 hold_13_5 - 225.3521126760563 hold_13_6
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_22_nic_in_14: switch_nic_out_22_nic_in_14 = 1 ->
   - 1263.157894736842 hold_14_1 - 657.5342465753424 hold_14_2
   - 444.4444444444444 hold_14_3 - 335.6643356643356 hold_14_4
   - 269.6629213483146 hold_14_5 - 225.3521126760563 hold_14_6
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_22_nic_in_15: switch_nic_out_22_nic_in_15 = 1 ->
   - 1263.157894736842 hold_15_1 - 657.5342465753424 hold_15_2
   - 444.4444444444444 hold_15_3 - 335.6643356643356 hold_15_4
   - 269.6629213483146 hold_15_5 - 225.3521126760563 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_16_nic_out_22_nic_in_16: switch_nic_out_22_nic_in_16 = 1 ->
   - 1263.157894736842 hold_16_1 - 657.5342465753424 hold_16_2
   - 444.4444444444444 hold_16_3 - 335.6643356643356 hold_16_4
   - 269.6629213483146 hold_16_5 - 225.3521126760563 hold_16_6
   + flow_nic_in_16_16 <= 0
 constr_intergroup_throughput_17_nic_out_22_nic_in_17: switch_nic_out_22_nic_in_17 = 1 ->
   - 1263.157894736842 hold_17_1 - 657.5342465753424 hold_17_2
   - 444.4444444444444 hold_17_3 - 335.6643356643356 hold_17_4
   - 269.6629213483146 hold_17_5 - 225.3521126760563 hold_17_6
   + flow_nic_in_17_17 <= 0
 constr_intergroup_throughput_18_nic_out_22_nic_in_18: switch_nic_out_22_nic_in_18 = 1 ->
   - 1263.157894736842 hold_18_1 - 657.5342465753424 hold_18_2
   - 444.4444444444444 hold_18_3 - 335.6643356643356 hold_18_4
   - 269.6629213483146 hold_18_5 - 225.3521126760563 hold_18_6
   + flow_nic_in_18_18 <= 0
 constr_intergroup_throughput_19_nic_out_22_nic_in_19: switch_nic_out_22_nic_in_19 = 1 ->
   - 1263.157894736842 hold_19_1 - 657.5342465753424 hold_19_2
   - 444.4444444444444 hold_19_3 - 335.6643356643356 hold_19_4
   - 269.6629213483146 hold_19_5 - 225.3521126760563 hold_19_6
   + flow_nic_in_19_19 <= 0
 constr_intergroup_throughput_20_nic_out_22_nic_in_20: switch_nic_out_22_nic_in_20 = 1 ->
   - 1263.157894736842 hold_20_1 - 657.5342465753424 hold_20_2
   - 444.4444444444444 hold_20_3 - 335.6643356643356 hold_20_4
   - 269.6629213483146 hold_20_5 - 225.3521126760563 hold_20_6
   + flow_nic_in_20_20 <= 0
 constr_intergroup_throughput_21_nic_out_22_nic_in_21: switch_nic_out_22_nic_in_21 = 1 ->
   - 1263.157894736842 hold_21_1 - 657.5342465753424 hold_21_2
   - 444.4444444444444 hold_21_3 - 335.6643356643356 hold_21_4
   - 269.6629213483146 hold_21_5 - 225.3521126760563 hold_21_6
   + flow_nic_in_21_21 <= 0
 constr_intergroup_throughput_23_nic_out_22_nic_in_23: switch_nic_out_22_nic_in_23 = 1 ->
   - 1263.157894736842 hold_23_1 - 657.5342465753424 hold_23_2
   - 444.4444444444444 hold_23_3 - 335.6643356643356 hold_23_4
   - 269.6629213483146 hold_23_5 - 225.3521126760563 hold_23_6
   + flow_nic_in_23_23 <= 0
 constr_intergroup_throughput_24_nic_out_22_nic_in_24: switch_nic_out_22_nic_in_24 = 1 ->
   - 1263.157894736842 hold_24_1 - 657.5342465753424 hold_24_2
   - 444.4444444444444 hold_24_3 - 335.6643356643356 hold_24_4
   - 269.6629213483146 hold_24_5 - 225.3521126760563 hold_24_6
   + flow_nic_in_24_24 <= 0
 constr_intergroup_throughput_25_nic_out_22_nic_in_25: switch_nic_out_22_nic_in_25 = 1 ->
   - 1263.157894736842 hold_25_1 - 657.5342465753424 hold_25_2
   - 444.4444444444444 hold_25_3 - 335.6643356643356 hold_25_4
   - 269.6629213483146 hold_25_5 - 225.3521126760563 hold_25_6
   + flow_nic_in_25_25 <= 0
 constr_intergroup_throughput_26_nic_out_22_nic_in_26: switch_nic_out_22_nic_in_26 = 1 ->
   - 1263.157894736842 hold_26_1 - 657.5342465753424 hold_26_2
   - 444.4444444444444 hold_26_3 - 335.6643356643356 hold_26_4
   - 269.6629213483146 hold_26_5 - 225.3521126760563 hold_26_6
   + flow_nic_in_26_26 <= 0
 constr_intergroup_throughput_27_nic_out_22_nic_in_27: switch_nic_out_22_nic_in_27 = 1 ->
   - 1263.157894736842 hold_27_1 - 657.5342465753424 hold_27_2
   - 444.4444444444444 hold_27_3 - 335.6643356643356 hold_27_4
   - 269.6629213483146 hold_27_5 - 225.3521126760563 hold_27_6
   + flow_nic_in_27_27 <= 0
 constr_intergroup_throughput_28_nic_out_22_nic_in_28: switch_nic_out_22_nic_in_28 = 1 ->
   - 1263.157894736842 hold_28_1 - 657.5342465753424 hold_28_2
   - 444.4444444444444 hold_28_3 - 335.6643356643356 hold_28_4
   - 269.6629213483146 hold_28_5 - 225.3521126760563 hold_28_6
   + flow_nic_in_28_28 <= 0
 constr_intergroup_throughput_29_nic_out_22_nic_in_29: switch_nic_out_22_nic_in_29 = 1 ->
   - 1263.157894736842 hold_29_1 - 657.5342465753424 hold_29_2
   - 444.4444444444444 hold_29_3 - 335.6643356643356 hold_29_4
   - 269.6629213483146 hold_29_5 - 225.3521126760563 hold_29_6
   + flow_nic_in_29_29 <= 0
 constr_intergroup_throughput_30_nic_out_22_nic_in_30: switch_nic_out_22_nic_in_30 = 1 ->
   - 1263.157894736842 hold_30_1 - 657.5342465753424 hold_30_2
   - 444.4444444444444 hold_30_3 - 335.6643356643356 hold_30_4
   - 269.6629213483146 hold_30_5 - 225.3521126760563 hold_30_6
   + flow_nic_in_30_30 <= 0
 constr_intergroup_throughput_31_nic_out_22_nic_in_31: switch_nic_out_22_nic_in_31 = 1 ->
   - 1263.157894736842 hold_31_1 - 657.5342465753424 hold_31_2
   - 444.4444444444444 hold_31_3 - 335.6643356643356 hold_31_4
   - 269.6629213483146 hold_31_5 - 225.3521126760563 hold_31_6
   + flow_nic_in_31_31 <= 0
 constr_intergroup_throughput_23_source_nic_in_23: switch_source_nic_in_23 = 1 ->
   - 1348.314606741573 hold_23_1 - 679.8866855524079 hold_23_2
   - 454.5454545454545 hold_23_3 - 341.3940256045519 hold_23_4
   - 273.3485193621868 hold_23_5 - 227.9202279202279 hold_23_6
   + flow_nic_in_23_23 <= 0
 constr_intergroup_throughput_0_nic_out_23_nic_in_0: switch_nic_out_23_nic_in_0 = 1 ->
   - 1263.157894736842 hold_0_1 - 657.5342465753424 hold_0_2
   - 444.4444444444444 hold_0_3 - 335.6643356643356 hold_0_4
   - 269.6629213483146 hold_0_5 - 225.3521126760563 hold_0_6
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_23_nic_in_1: switch_nic_out_23_nic_in_1 = 1 ->
   - 1263.157894736842 hold_1_1 - 657.5342465753424 hold_1_2
   - 444.4444444444444 hold_1_3 - 335.6643356643356 hold_1_4
   - 269.6629213483146 hold_1_5 - 225.3521126760563 hold_1_6
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_23_nic_in_2: switch_nic_out_23_nic_in_2 = 1 ->
   - 1263.157894736842 hold_2_1 - 657.5342465753424 hold_2_2
   - 444.4444444444444 hold_2_3 - 335.6643356643356 hold_2_4
   - 269.6629213483146 hold_2_5 - 225.3521126760563 hold_2_6
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_23_nic_in_3: switch_nic_out_23_nic_in_3 = 1 ->
   - 1263.157894736842 hold_3_1 - 657.5342465753424 hold_3_2
   - 444.4444444444444 hold_3_3 - 335.6643356643356 hold_3_4
   - 269.6629213483146 hold_3_5 - 225.3521126760563 hold_3_6
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_23_nic_in_4: switch_nic_out_23_nic_in_4 = 1 ->
   - 1263.157894736842 hold_4_1 - 657.5342465753424 hold_4_2
   - 444.4444444444444 hold_4_3 - 335.6643356643356 hold_4_4
   - 269.6629213483146 hold_4_5 - 225.3521126760563 hold_4_6
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_23_nic_in_5: switch_nic_out_23_nic_in_5 = 1 ->
   - 1263.157894736842 hold_5_1 - 657.5342465753424 hold_5_2
   - 444.4444444444444 hold_5_3 - 335.6643356643356 hold_5_4
   - 269.6629213483146 hold_5_5 - 225.3521126760563 hold_5_6
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_23_nic_in_6: switch_nic_out_23_nic_in_6 = 1 ->
   - 1263.157894736842 hold_6_1 - 657.5342465753424 hold_6_2
   - 444.4444444444444 hold_6_3 - 335.6643356643356 hold_6_4
   - 269.6629213483146 hold_6_5 - 225.3521126760563 hold_6_6
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_23_nic_in_7: switch_nic_out_23_nic_in_7 = 1 ->
   - 1263.157894736842 hold_7_1 - 657.5342465753424 hold_7_2
   - 444.4444444444444 hold_7_3 - 335.6643356643356 hold_7_4
   - 269.6629213483146 hold_7_5 - 225.3521126760563 hold_7_6
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_23_nic_in_8: switch_nic_out_23_nic_in_8 = 1 ->
   - 1263.157894736842 hold_8_1 - 657.5342465753424 hold_8_2
   - 444.4444444444444 hold_8_3 - 335.6643356643356 hold_8_4
   - 269.6629213483146 hold_8_5 - 225.3521126760563 hold_8_6
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_23_nic_in_9: switch_nic_out_23_nic_in_9 = 1 ->
   - 1263.157894736842 hold_9_1 - 657.5342465753424 hold_9_2
   - 444.4444444444444 hold_9_3 - 335.6643356643356 hold_9_4
   - 269.6629213483146 hold_9_5 - 225.3521126760563 hold_9_6
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_23_nic_in_10: switch_nic_out_23_nic_in_10 = 1 ->
   - 1263.157894736842 hold_10_1 - 657.5342465753424 hold_10_2
   - 444.4444444444444 hold_10_3 - 335.6643356643356 hold_10_4
   - 269.6629213483146 hold_10_5 - 225.3521126760563 hold_10_6
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_23_nic_in_11: switch_nic_out_23_nic_in_11 = 1 ->
   - 1263.157894736842 hold_11_1 - 657.5342465753424 hold_11_2
   - 444.4444444444444 hold_11_3 - 335.6643356643356 hold_11_4
   - 269.6629213483146 hold_11_5 - 225.3521126760563 hold_11_6
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_23_nic_in_12: switch_nic_out_23_nic_in_12 = 1 ->
   - 1263.157894736842 hold_12_1 - 657.5342465753424 hold_12_2
   - 444.4444444444444 hold_12_3 - 335.6643356643356 hold_12_4
   - 269.6629213483146 hold_12_5 - 225.3521126760563 hold_12_6
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_23_nic_in_13: switch_nic_out_23_nic_in_13 = 1 ->
   - 1263.157894736842 hold_13_1 - 657.5342465753424 hold_13_2
   - 444.4444444444444 hold_13_3 - 335.6643356643356 hold_13_4
   - 269.6629213483146 hold_13_5 - 225.3521126760563 hold_13_6
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_23_nic_in_14: switch_nic_out_23_nic_in_14 = 1 ->
   - 1263.157894736842 hold_14_1 - 657.5342465753424 hold_14_2
   - 444.4444444444444 hold_14_3 - 335.6643356643356 hold_14_4
   - 269.6629213483146 hold_14_5 - 225.3521126760563 hold_14_6
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_23_nic_in_15: switch_nic_out_23_nic_in_15 = 1 ->
   - 1263.157894736842 hold_15_1 - 657.5342465753424 hold_15_2
   - 444.4444444444444 hold_15_3 - 335.6643356643356 hold_15_4
   - 269.6629213483146 hold_15_5 - 225.3521126760563 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_16_nic_out_23_nic_in_16: switch_nic_out_23_nic_in_16 = 1 ->
   - 1263.157894736842 hold_16_1 - 657.5342465753424 hold_16_2
   - 444.4444444444444 hold_16_3 - 335.6643356643356 hold_16_4
   - 269.6629213483146 hold_16_5 - 225.3521126760563 hold_16_6
   + flow_nic_in_16_16 <= 0
 constr_intergroup_throughput_17_nic_out_23_nic_in_17: switch_nic_out_23_nic_in_17 = 1 ->
   - 1263.157894736842 hold_17_1 - 657.5342465753424 hold_17_2
   - 444.4444444444444 hold_17_3 - 335.6643356643356 hold_17_4
   - 269.6629213483146 hold_17_5 - 225.3521126760563 hold_17_6
   + flow_nic_in_17_17 <= 0
 constr_intergroup_throughput_18_nic_out_23_nic_in_18: switch_nic_out_23_nic_in_18 = 1 ->
   - 1263.157894736842 hold_18_1 - 657.5342465753424 hold_18_2
   - 444.4444444444444 hold_18_3 - 335.6643356643356 hold_18_4
   - 269.6629213483146 hold_18_5 - 225.3521126760563 hold_18_6
   + flow_nic_in_18_18 <= 0
 constr_intergroup_throughput_19_nic_out_23_nic_in_19: switch_nic_out_23_nic_in_19 = 1 ->
   - 1263.157894736842 hold_19_1 - 657.5342465753424 hold_19_2
   - 444.4444444444444 hold_19_3 - 335.6643356643356 hold_19_4
   - 269.6629213483146 hold_19_5 - 225.3521126760563 hold_19_6
   + flow_nic_in_19_19 <= 0
 constr_intergroup_throughput_20_nic_out_23_nic_in_20: switch_nic_out_23_nic_in_20 = 1 ->
   - 1263.157894736842 hold_20_1 - 657.5342465753424 hold_20_2
   - 444.4444444444444 hold_20_3 - 335.6643356643356 hold_20_4
   - 269.6629213483146 hold_20_5 - 225.3521126760563 hold_20_6
   + flow_nic_in_20_20 <= 0
 constr_intergroup_throughput_21_nic_out_23_nic_in_21: switch_nic_out_23_nic_in_21 = 1 ->
   - 1263.157894736842 hold_21_1 - 657.5342465753424 hold_21_2
   - 444.4444444444444 hold_21_3 - 335.6643356643356 hold_21_4
   - 269.6629213483146 hold_21_5 - 225.3521126760563 hold_21_6
   + flow_nic_in_21_21 <= 0
 constr_intergroup_throughput_22_nic_out_23_nic_in_22: switch_nic_out_23_nic_in_22 = 1 ->
   - 1263.157894736842 hold_22_1 - 657.5342465753424 hold_22_2
   - 444.4444444444444 hold_22_3 - 335.6643356643356 hold_22_4
   - 269.6629213483146 hold_22_5 - 225.3521126760563 hold_22_6
   + flow_nic_in_22_22 <= 0
 constr_intergroup_throughput_24_nic_out_23_nic_in_24: switch_nic_out_23_nic_in_24 = 1 ->
   - 1263.157894736842 hold_24_1 - 657.5342465753424 hold_24_2
   - 444.4444444444444 hold_24_3 - 335.6643356643356 hold_24_4
   - 269.6629213483146 hold_24_5 - 225.3521126760563 hold_24_6
   + flow_nic_in_24_24 <= 0
 constr_intergroup_throughput_25_nic_out_23_nic_in_25: switch_nic_out_23_nic_in_25 = 1 ->
   - 1263.157894736842 hold_25_1 - 657.5342465753424 hold_25_2
   - 444.4444444444444 hold_25_3 - 335.6643356643356 hold_25_4
   - 269.6629213483146 hold_25_5 - 225.3521126760563 hold_25_6
   + flow_nic_in_25_25 <= 0
 constr_intergroup_throughput_26_nic_out_23_nic_in_26: switch_nic_out_23_nic_in_26 = 1 ->
   - 1263.157894736842 hold_26_1 - 657.5342465753424 hold_26_2
   - 444.4444444444444 hold_26_3 - 335.6643356643356 hold_26_4
   - 269.6629213483146 hold_26_5 - 225.3521126760563 hold_26_6
   + flow_nic_in_26_26 <= 0
 constr_intergroup_throughput_27_nic_out_23_nic_in_27: switch_nic_out_23_nic_in_27 = 1 ->
   - 1263.157894736842 hold_27_1 - 657.5342465753424 hold_27_2
   - 444.4444444444444 hold_27_3 - 335.6643356643356 hold_27_4
   - 269.6629213483146 hold_27_5 - 225.3521126760563 hold_27_6
   + flow_nic_in_27_27 <= 0
 constr_intergroup_throughput_28_nic_out_23_nic_in_28: switch_nic_out_23_nic_in_28 = 1 ->
   - 1263.157894736842 hold_28_1 - 657.5342465753424 hold_28_2
   - 444.4444444444444 hold_28_3 - 335.6643356643356 hold_28_4
   - 269.6629213483146 hold_28_5 - 225.3521126760563 hold_28_6
   + flow_nic_in_28_28 <= 0
 constr_intergroup_throughput_29_nic_out_23_nic_in_29: switch_nic_out_23_nic_in_29 = 1 ->
   - 1263.157894736842 hold_29_1 - 657.5342465753424 hold_29_2
   - 444.4444444444444 hold_29_3 - 335.6643356643356 hold_29_4
   - 269.6629213483146 hold_29_5 - 225.3521126760563 hold_29_6
   + flow_nic_in_29_29 <= 0
 constr_intergroup_throughput_30_nic_out_23_nic_in_30: switch_nic_out_23_nic_in_30 = 1 ->
   - 1263.157894736842 hold_30_1 - 657.5342465753424 hold_30_2
   - 444.4444444444444 hold_30_3 - 335.6643356643356 hold_30_4
   - 269.6629213483146 hold_30_5 - 225.3521126760563 hold_30_6
   + flow_nic_in_30_30 <= 0
 constr_intergroup_throughput_31_nic_out_23_nic_in_31: switch_nic_out_23_nic_in_31 = 1 ->
   - 1263.157894736842 hold_31_1 - 657.5342465753424 hold_31_2
   - 444.4444444444444 hold_31_3 - 335.6643356643356 hold_31_4
   - 269.6629213483146 hold_31_5 - 225.3521126760563 hold_31_6
   + flow_nic_in_31_31 <= 0
 constr_intergroup_throughput_24_source_nic_in_24: switch_source_nic_in_24 = 1 ->
   - 1348.314606741573 hold_24_1 - 679.8866855524079 hold_24_2
   - 454.5454545454545 hold_24_3 - 341.3940256045519 hold_24_4
   - 273.3485193621868 hold_24_5 - 227.9202279202279 hold_24_6
   + flow_nic_in_24_24 <= 0
 constr_intergroup_throughput_0_nic_out_24_nic_in_0: switch_nic_out_24_nic_in_0 = 1 ->
   - 1263.157894736842 hold_0_1 - 657.5342465753424 hold_0_2
   - 444.4444444444444 hold_0_3 - 335.6643356643356 hold_0_4
   - 269.6629213483146 hold_0_5 - 225.3521126760563 hold_0_6
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_24_nic_in_1: switch_nic_out_24_nic_in_1 = 1 ->
   - 1263.157894736842 hold_1_1 - 657.5342465753424 hold_1_2
   - 444.4444444444444 hold_1_3 - 335.6643356643356 hold_1_4
   - 269.6629213483146 hold_1_5 - 225.3521126760563 hold_1_6
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_24_nic_in_2: switch_nic_out_24_nic_in_2 = 1 ->
   - 1263.157894736842 hold_2_1 - 657.5342465753424 hold_2_2
   - 444.4444444444444 hold_2_3 - 335.6643356643356 hold_2_4
   - 269.6629213483146 hold_2_5 - 225.3521126760563 hold_2_6
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_24_nic_in_3: switch_nic_out_24_nic_in_3 = 1 ->
   - 1263.157894736842 hold_3_1 - 657.5342465753424 hold_3_2
   - 444.4444444444444 hold_3_3 - 335.6643356643356 hold_3_4
   - 269.6629213483146 hold_3_5 - 225.3521126760563 hold_3_6
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_24_nic_in_4: switch_nic_out_24_nic_in_4 = 1 ->
   - 1263.157894736842 hold_4_1 - 657.5342465753424 hold_4_2
   - 444.4444444444444 hold_4_3 - 335.6643356643356 hold_4_4
   - 269.6629213483146 hold_4_5 - 225.3521126760563 hold_4_6
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_24_nic_in_5: switch_nic_out_24_nic_in_5 = 1 ->
   - 1263.157894736842 hold_5_1 - 657.5342465753424 hold_5_2
   - 444.4444444444444 hold_5_3 - 335.6643356643356 hold_5_4
   - 269.6629213483146 hold_5_5 - 225.3521126760563 hold_5_6
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_24_nic_in_6: switch_nic_out_24_nic_in_6 = 1 ->
   - 1263.157894736842 hold_6_1 - 657.5342465753424 hold_6_2
   - 444.4444444444444 hold_6_3 - 335.6643356643356 hold_6_4
   - 269.6629213483146 hold_6_5 - 225.3521126760563 hold_6_6
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_24_nic_in_7: switch_nic_out_24_nic_in_7 = 1 ->
   - 1263.157894736842 hold_7_1 - 657.5342465753424 hold_7_2
   - 444.4444444444444 hold_7_3 - 335.6643356643356 hold_7_4
   - 269.6629213483146 hold_7_5 - 225.3521126760563 hold_7_6
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_24_nic_in_8: switch_nic_out_24_nic_in_8 = 1 ->
   - 1263.157894736842 hold_8_1 - 657.5342465753424 hold_8_2
   - 444.4444444444444 hold_8_3 - 335.6643356643356 hold_8_4
   - 269.6629213483146 hold_8_5 - 225.3521126760563 hold_8_6
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_24_nic_in_9: switch_nic_out_24_nic_in_9 = 1 ->
   - 1263.157894736842 hold_9_1 - 657.5342465753424 hold_9_2
   - 444.4444444444444 hold_9_3 - 335.6643356643356 hold_9_4
   - 269.6629213483146 hold_9_5 - 225.3521126760563 hold_9_6
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_24_nic_in_10: switch_nic_out_24_nic_in_10 = 1 ->
   - 1263.157894736842 hold_10_1 - 657.5342465753424 hold_10_2
   - 444.4444444444444 hold_10_3 - 335.6643356643356 hold_10_4
   - 269.6629213483146 hold_10_5 - 225.3521126760563 hold_10_6
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_24_nic_in_11: switch_nic_out_24_nic_in_11 = 1 ->
   - 1263.157894736842 hold_11_1 - 657.5342465753424 hold_11_2
   - 444.4444444444444 hold_11_3 - 335.6643356643356 hold_11_4
   - 269.6629213483146 hold_11_5 - 225.3521126760563 hold_11_6
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_24_nic_in_12: switch_nic_out_24_nic_in_12 = 1 ->
   - 1263.157894736842 hold_12_1 - 657.5342465753424 hold_12_2
   - 444.4444444444444 hold_12_3 - 335.6643356643356 hold_12_4
   - 269.6629213483146 hold_12_5 - 225.3521126760563 hold_12_6
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_24_nic_in_13: switch_nic_out_24_nic_in_13 = 1 ->
   - 1263.157894736842 hold_13_1 - 657.5342465753424 hold_13_2
   - 444.4444444444444 hold_13_3 - 335.6643356643356 hold_13_4
   - 269.6629213483146 hold_13_5 - 225.3521126760563 hold_13_6
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_24_nic_in_14: switch_nic_out_24_nic_in_14 = 1 ->
   - 1263.157894736842 hold_14_1 - 657.5342465753424 hold_14_2
   - 444.4444444444444 hold_14_3 - 335.6643356643356 hold_14_4
   - 269.6629213483146 hold_14_5 - 225.3521126760563 hold_14_6
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_24_nic_in_15: switch_nic_out_24_nic_in_15 = 1 ->
   - 1263.157894736842 hold_15_1 - 657.5342465753424 hold_15_2
   - 444.4444444444444 hold_15_3 - 335.6643356643356 hold_15_4
   - 269.6629213483146 hold_15_5 - 225.3521126760563 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_16_nic_out_24_nic_in_16: switch_nic_out_24_nic_in_16 = 1 ->
   - 1263.157894736842 hold_16_1 - 657.5342465753424 hold_16_2
   - 444.4444444444444 hold_16_3 - 335.6643356643356 hold_16_4
   - 269.6629213483146 hold_16_5 - 225.3521126760563 hold_16_6
   + flow_nic_in_16_16 <= 0
 constr_intergroup_throughput_17_nic_out_24_nic_in_17: switch_nic_out_24_nic_in_17 = 1 ->
   - 1263.157894736842 hold_17_1 - 657.5342465753424 hold_17_2
   - 444.4444444444444 hold_17_3 - 335.6643356643356 hold_17_4
   - 269.6629213483146 hold_17_5 - 225.3521126760563 hold_17_6
   + flow_nic_in_17_17 <= 0
 constr_intergroup_throughput_18_nic_out_24_nic_in_18: switch_nic_out_24_nic_in_18 = 1 ->
   - 1263.157894736842 hold_18_1 - 657.5342465753424 hold_18_2
   - 444.4444444444444 hold_18_3 - 335.6643356643356 hold_18_4
   - 269.6629213483146 hold_18_5 - 225.3521126760563 hold_18_6
   + flow_nic_in_18_18 <= 0
 constr_intergroup_throughput_19_nic_out_24_nic_in_19: switch_nic_out_24_nic_in_19 = 1 ->
   - 1263.157894736842 hold_19_1 - 657.5342465753424 hold_19_2
   - 444.4444444444444 hold_19_3 - 335.6643356643356 hold_19_4
   - 269.6629213483146 hold_19_5 - 225.3521126760563 hold_19_6
   + flow_nic_in_19_19 <= 0
 constr_intergroup_throughput_20_nic_out_24_nic_in_20: switch_nic_out_24_nic_in_20 = 1 ->
   - 1263.157894736842 hold_20_1 - 657.5342465753424 hold_20_2
   - 444.4444444444444 hold_20_3 - 335.6643356643356 hold_20_4
   - 269.6629213483146 hold_20_5 - 225.3521126760563 hold_20_6
   + flow_nic_in_20_20 <= 0
 constr_intergroup_throughput_21_nic_out_24_nic_in_21: switch_nic_out_24_nic_in_21 = 1 ->
   - 1263.157894736842 hold_21_1 - 657.5342465753424 hold_21_2
   - 444.4444444444444 hold_21_3 - 335.6643356643356 hold_21_4
   - 269.6629213483146 hold_21_5 - 225.3521126760563 hold_21_6
   + flow_nic_in_21_21 <= 0
 constr_intergroup_throughput_22_nic_out_24_nic_in_22: switch_nic_out_24_nic_in_22 = 1 ->
   - 1263.157894736842 hold_22_1 - 657.5342465753424 hold_22_2
   - 444.4444444444444 hold_22_3 - 335.6643356643356 hold_22_4
   - 269.6629213483146 hold_22_5 - 225.3521126760563 hold_22_6
   + flow_nic_in_22_22 <= 0
 constr_intergroup_throughput_23_nic_out_24_nic_in_23: switch_nic_out_24_nic_in_23 = 1 ->
   - 1263.157894736842 hold_23_1 - 657.5342465753424 hold_23_2
   - 444.4444444444444 hold_23_3 - 335.6643356643356 hold_23_4
   - 269.6629213483146 hold_23_5 - 225.3521126760563 hold_23_6
   + flow_nic_in_23_23 <= 0
 constr_intergroup_throughput_25_nic_out_24_nic_in_25: switch_nic_out_24_nic_in_25 = 1 ->
   - 1263.157894736842 hold_25_1 - 657.5342465753424 hold_25_2
   - 444.4444444444444 hold_25_3 - 335.6643356643356 hold_25_4
   - 269.6629213483146 hold_25_5 - 225.3521126760563 hold_25_6
   + flow_nic_in_25_25 <= 0
 constr_intergroup_throughput_26_nic_out_24_nic_in_26: switch_nic_out_24_nic_in_26 = 1 ->
   - 1263.157894736842 hold_26_1 - 657.5342465753424 hold_26_2
   - 444.4444444444444 hold_26_3 - 335.6643356643356 hold_26_4
   - 269.6629213483146 hold_26_5 - 225.3521126760563 hold_26_6
   + flow_nic_in_26_26 <= 0
 constr_intergroup_throughput_27_nic_out_24_nic_in_27: switch_nic_out_24_nic_in_27 = 1 ->
   - 1263.157894736842 hold_27_1 - 657.5342465753424 hold_27_2
   - 444.4444444444444 hold_27_3 - 335.6643356643356 hold_27_4
   - 269.6629213483146 hold_27_5 - 225.3521126760563 hold_27_6
   + flow_nic_in_27_27 <= 0
 constr_intergroup_throughput_28_nic_out_24_nic_in_28: switch_nic_out_24_nic_in_28 = 1 ->
   - 1263.157894736842 hold_28_1 - 657.5342465753424 hold_28_2
   - 444.4444444444444 hold_28_3 - 335.6643356643356 hold_28_4
   - 269.6629213483146 hold_28_5 - 225.3521126760563 hold_28_6
   + flow_nic_in_28_28 <= 0
 constr_intergroup_throughput_29_nic_out_24_nic_in_29: switch_nic_out_24_nic_in_29 = 1 ->
   - 1263.157894736842 hold_29_1 - 657.5342465753424 hold_29_2
   - 444.4444444444444 hold_29_3 - 335.6643356643356 hold_29_4
   - 269.6629213483146 hold_29_5 - 225.3521126760563 hold_29_6
   + flow_nic_in_29_29 <= 0
 constr_intergroup_throughput_30_nic_out_24_nic_in_30: switch_nic_out_24_nic_in_30 = 1 ->
   - 1263.157894736842 hold_30_1 - 657.5342465753424 hold_30_2
   - 444.4444444444444 hold_30_3 - 335.6643356643356 hold_30_4
   - 269.6629213483146 hold_30_5 - 225.3521126760563 hold_30_6
   + flow_nic_in_30_30 <= 0
 constr_intergroup_throughput_31_nic_out_24_nic_in_31: switch_nic_out_24_nic_in_31 = 1 ->
   - 1263.157894736842 hold_31_1 - 657.5342465753424 hold_31_2
   - 444.4444444444444 hold_31_3 - 335.6643356643356 hold_31_4
   - 269.6629213483146 hold_31_5 - 225.3521126760563 hold_31_6
   + flow_nic_in_31_31 <= 0
 constr_intergroup_throughput_25_source_nic_in_25: switch_source_nic_in_25 = 1 ->
   - 1348.314606741573 hold_25_1 - 679.8866855524079 hold_25_2
   - 454.5454545454545 hold_25_3 - 341.3940256045519 hold_25_4
   - 273.3485193621868 hold_25_5 - 227.9202279202279 hold_25_6
   + flow_nic_in_25_25 <= 0
 constr_intergroup_throughput_0_nic_out_25_nic_in_0: switch_nic_out_25_nic_in_0 = 1 ->
   - 1263.157894736842 hold_0_1 - 657.5342465753424 hold_0_2
   - 444.4444444444444 hold_0_3 - 335.6643356643356 hold_0_4
   - 269.6629213483146 hold_0_5 - 225.3521126760563 hold_0_6
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_25_nic_in_1: switch_nic_out_25_nic_in_1 = 1 ->
   - 1263.157894736842 hold_1_1 - 657.5342465753424 hold_1_2
   - 444.4444444444444 hold_1_3 - 335.6643356643356 hold_1_4
   - 269.6629213483146 hold_1_5 - 225.3521126760563 hold_1_6
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_25_nic_in_2: switch_nic_out_25_nic_in_2 = 1 ->
   - 1263.157894736842 hold_2_1 - 657.5342465753424 hold_2_2
   - 444.4444444444444 hold_2_3 - 335.6643356643356 hold_2_4
   - 269.6629213483146 hold_2_5 - 225.3521126760563 hold_2_6
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_25_nic_in_3: switch_nic_out_25_nic_in_3 = 1 ->
   - 1263.157894736842 hold_3_1 - 657.5342465753424 hold_3_2
   - 444.4444444444444 hold_3_3 - 335.6643356643356 hold_3_4
   - 269.6629213483146 hold_3_5 - 225.3521126760563 hold_3_6
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_25_nic_in_4: switch_nic_out_25_nic_in_4 = 1 ->
   - 1263.157894736842 hold_4_1 - 657.5342465753424 hold_4_2
   - 444.4444444444444 hold_4_3 - 335.6643356643356 hold_4_4
   - 269.6629213483146 hold_4_5 - 225.3521126760563 hold_4_6
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_25_nic_in_5: switch_nic_out_25_nic_in_5 = 1 ->
   - 1263.157894736842 hold_5_1 - 657.5342465753424 hold_5_2
   - 444.4444444444444 hold_5_3 - 335.6643356643356 hold_5_4
   - 269.6629213483146 hold_5_5 - 225.3521126760563 hold_5_6
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_25_nic_in_6: switch_nic_out_25_nic_in_6 = 1 ->
   - 1263.157894736842 hold_6_1 - 657.5342465753424 hold_6_2
   - 444.4444444444444 hold_6_3 - 335.6643356643356 hold_6_4
   - 269.6629213483146 hold_6_5 - 225.3521126760563 hold_6_6
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_25_nic_in_7: switch_nic_out_25_nic_in_7 = 1 ->
   - 1263.157894736842 hold_7_1 - 657.5342465753424 hold_7_2
   - 444.4444444444444 hold_7_3 - 335.6643356643356 hold_7_4
   - 269.6629213483146 hold_7_5 - 225.3521126760563 hold_7_6
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_25_nic_in_8: switch_nic_out_25_nic_in_8 = 1 ->
   - 1263.157894736842 hold_8_1 - 657.5342465753424 hold_8_2
   - 444.4444444444444 hold_8_3 - 335.6643356643356 hold_8_4
   - 269.6629213483146 hold_8_5 - 225.3521126760563 hold_8_6
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_25_nic_in_9: switch_nic_out_25_nic_in_9 = 1 ->
   - 1263.157894736842 hold_9_1 - 657.5342465753424 hold_9_2
   - 444.4444444444444 hold_9_3 - 335.6643356643356 hold_9_4
   - 269.6629213483146 hold_9_5 - 225.3521126760563 hold_9_6
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_25_nic_in_10: switch_nic_out_25_nic_in_10 = 1 ->
   - 1263.157894736842 hold_10_1 - 657.5342465753424 hold_10_2
   - 444.4444444444444 hold_10_3 - 335.6643356643356 hold_10_4
   - 269.6629213483146 hold_10_5 - 225.3521126760563 hold_10_6
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_25_nic_in_11: switch_nic_out_25_nic_in_11 = 1 ->
   - 1263.157894736842 hold_11_1 - 657.5342465753424 hold_11_2
   - 444.4444444444444 hold_11_3 - 335.6643356643356 hold_11_4
   - 269.6629213483146 hold_11_5 - 225.3521126760563 hold_11_6
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_25_nic_in_12: switch_nic_out_25_nic_in_12 = 1 ->
   - 1263.157894736842 hold_12_1 - 657.5342465753424 hold_12_2
   - 444.4444444444444 hold_12_3 - 335.6643356643356 hold_12_4
   - 269.6629213483146 hold_12_5 - 225.3521126760563 hold_12_6
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_25_nic_in_13: switch_nic_out_25_nic_in_13 = 1 ->
   - 1263.157894736842 hold_13_1 - 657.5342465753424 hold_13_2
   - 444.4444444444444 hold_13_3 - 335.6643356643356 hold_13_4
   - 269.6629213483146 hold_13_5 - 225.3521126760563 hold_13_6
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_25_nic_in_14: switch_nic_out_25_nic_in_14 = 1 ->
   - 1263.157894736842 hold_14_1 - 657.5342465753424 hold_14_2
   - 444.4444444444444 hold_14_3 - 335.6643356643356 hold_14_4
   - 269.6629213483146 hold_14_5 - 225.3521126760563 hold_14_6
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_25_nic_in_15: switch_nic_out_25_nic_in_15 = 1 ->
   - 1263.157894736842 hold_15_1 - 657.5342465753424 hold_15_2
   - 444.4444444444444 hold_15_3 - 335.6643356643356 hold_15_4
   - 269.6629213483146 hold_15_5 - 225.3521126760563 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_16_nic_out_25_nic_in_16: switch_nic_out_25_nic_in_16 = 1 ->
   - 1263.157894736842 hold_16_1 - 657.5342465753424 hold_16_2
   - 444.4444444444444 hold_16_3 - 335.6643356643356 hold_16_4
   - 269.6629213483146 hold_16_5 - 225.3521126760563 hold_16_6
   + flow_nic_in_16_16 <= 0
 constr_intergroup_throughput_17_nic_out_25_nic_in_17: switch_nic_out_25_nic_in_17 = 1 ->
   - 1263.157894736842 hold_17_1 - 657.5342465753424 hold_17_2
   - 444.4444444444444 hold_17_3 - 335.6643356643356 hold_17_4
   - 269.6629213483146 hold_17_5 - 225.3521126760563 hold_17_6
   + flow_nic_in_17_17 <= 0
 constr_intergroup_throughput_18_nic_out_25_nic_in_18: switch_nic_out_25_nic_in_18 = 1 ->
   - 1263.157894736842 hold_18_1 - 657.5342465753424 hold_18_2
   - 444.4444444444444 hold_18_3 - 335.6643356643356 hold_18_4
   - 269.6629213483146 hold_18_5 - 225.3521126760563 hold_18_6
   + flow_nic_in_18_18 <= 0
 constr_intergroup_throughput_19_nic_out_25_nic_in_19: switch_nic_out_25_nic_in_19 = 1 ->
   - 1263.157894736842 hold_19_1 - 657.5342465753424 hold_19_2
   - 444.4444444444444 hold_19_3 - 335.6643356643356 hold_19_4
   - 269.6629213483146 hold_19_5 - 225.3521126760563 hold_19_6
   + flow_nic_in_19_19 <= 0
 constr_intergroup_throughput_20_nic_out_25_nic_in_20: switch_nic_out_25_nic_in_20 = 1 ->
   - 1263.157894736842 hold_20_1 - 657.5342465753424 hold_20_2
   - 444.4444444444444 hold_20_3 - 335.6643356643356 hold_20_4
   - 269.6629213483146 hold_20_5 - 225.3521126760563 hold_20_6
   + flow_nic_in_20_20 <= 0
 constr_intergroup_throughput_21_nic_out_25_nic_in_21: switch_nic_out_25_nic_in_21 = 1 ->
   - 1263.157894736842 hold_21_1 - 657.5342465753424 hold_21_2
   - 444.4444444444444 hold_21_3 - 335.6643356643356 hold_21_4
   - 269.6629213483146 hold_21_5 - 225.3521126760563 hold_21_6
   + flow_nic_in_21_21 <= 0
 constr_intergroup_throughput_22_nic_out_25_nic_in_22: switch_nic_out_25_nic_in_22 = 1 ->
   - 1263.157894736842 hold_22_1 - 657.5342465753424 hold_22_2
   - 444.4444444444444 hold_22_3 - 335.6643356643356 hold_22_4
   - 269.6629213483146 hold_22_5 - 225.3521126760563 hold_22_6
   + flow_nic_in_22_22 <= 0
 constr_intergroup_throughput_23_nic_out_25_nic_in_23: switch_nic_out_25_nic_in_23 = 1 ->
   - 1263.157894736842 hold_23_1 - 657.5342465753424 hold_23_2
   - 444.4444444444444 hold_23_3 - 335.6643356643356 hold_23_4
   - 269.6629213483146 hold_23_5 - 225.3521126760563 hold_23_6
   + flow_nic_in_23_23 <= 0
 constr_intergroup_throughput_24_nic_out_25_nic_in_24: switch_nic_out_25_nic_in_24 = 1 ->
   - 1263.157894736842 hold_24_1 - 657.5342465753424 hold_24_2
   - 444.4444444444444 hold_24_3 - 335.6643356643356 hold_24_4
   - 269.6629213483146 hold_24_5 - 225.3521126760563 hold_24_6
   + flow_nic_in_24_24 <= 0
 constr_intergroup_throughput_26_nic_out_25_nic_in_26: switch_nic_out_25_nic_in_26 = 1 ->
   - 1263.157894736842 hold_26_1 - 657.5342465753424 hold_26_2
   - 444.4444444444444 hold_26_3 - 335.6643356643356 hold_26_4
   - 269.6629213483146 hold_26_5 - 225.3521126760563 hold_26_6
   + flow_nic_in_26_26 <= 0
 constr_intergroup_throughput_27_nic_out_25_nic_in_27: switch_nic_out_25_nic_in_27 = 1 ->
   - 1263.157894736842 hold_27_1 - 657.5342465753424 hold_27_2
   - 444.4444444444444 hold_27_3 - 335.6643356643356 hold_27_4
   - 269.6629213483146 hold_27_5 - 225.3521126760563 hold_27_6
   + flow_nic_in_27_27 <= 0
 constr_intergroup_throughput_28_nic_out_25_nic_in_28: switch_nic_out_25_nic_in_28 = 1 ->
   - 1263.157894736842 hold_28_1 - 657.5342465753424 hold_28_2
   - 444.4444444444444 hold_28_3 - 335.6643356643356 hold_28_4
   - 269.6629213483146 hold_28_5 - 225.3521126760563 hold_28_6
   + flow_nic_in_28_28 <= 0
 constr_intergroup_throughput_29_nic_out_25_nic_in_29: switch_nic_out_25_nic_in_29 = 1 ->
   - 1263.157894736842 hold_29_1 - 657.5342465753424 hold_29_2
   - 444.4444444444444 hold_29_3 - 335.6643356643356 hold_29_4
   - 269.6629213483146 hold_29_5 - 225.3521126760563 hold_29_6
   + flow_nic_in_29_29 <= 0
 constr_intergroup_throughput_30_nic_out_25_nic_in_30: switch_nic_out_25_nic_in_30 = 1 ->
   - 1263.157894736842 hold_30_1 - 657.5342465753424 hold_30_2
   - 444.4444444444444 hold_30_3 - 335.6643356643356 hold_30_4
   - 269.6629213483146 hold_30_5 - 225.3521126760563 hold_30_6
   + flow_nic_in_30_30 <= 0
 constr_intergroup_throughput_31_nic_out_25_nic_in_31: switch_nic_out_25_nic_in_31 = 1 ->
   - 1263.157894736842 hold_31_1 - 657.5342465753424 hold_31_2
   - 444.4444444444444 hold_31_3 - 335.6643356643356 hold_31_4
   - 269.6629213483146 hold_31_5 - 225.3521126760563 hold_31_6
   + flow_nic_in_31_31 <= 0
 constr_intergroup_throughput_26_source_nic_in_26: switch_source_nic_in_26 = 1 ->
   - 1348.314606741573 hold_26_1 - 679.8866855524079 hold_26_2
   - 454.5454545454545 hold_26_3 - 341.3940256045519 hold_26_4
   - 273.3485193621868 hold_26_5 - 227.9202279202279 hold_26_6
   + flow_nic_in_26_26 <= 0
 constr_intergroup_throughput_0_nic_out_26_nic_in_0: switch_nic_out_26_nic_in_0 = 1 ->
   - 1263.157894736842 hold_0_1 - 657.5342465753424 hold_0_2
   - 444.4444444444444 hold_0_3 - 335.6643356643356 hold_0_4
   - 269.6629213483146 hold_0_5 - 225.3521126760563 hold_0_6
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_26_nic_in_1: switch_nic_out_26_nic_in_1 = 1 ->
   - 1263.157894736842 hold_1_1 - 657.5342465753424 hold_1_2
   - 444.4444444444444 hold_1_3 - 335.6643356643356 hold_1_4
   - 269.6629213483146 hold_1_5 - 225.3521126760563 hold_1_6
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_26_nic_in_2: switch_nic_out_26_nic_in_2 = 1 ->
   - 1263.157894736842 hold_2_1 - 657.5342465753424 hold_2_2
   - 444.4444444444444 hold_2_3 - 335.6643356643356 hold_2_4
   - 269.6629213483146 hold_2_5 - 225.3521126760563 hold_2_6
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_26_nic_in_3: switch_nic_out_26_nic_in_3 = 1 ->
   - 1263.157894736842 hold_3_1 - 657.5342465753424 hold_3_2
   - 444.4444444444444 hold_3_3 - 335.6643356643356 hold_3_4
   - 269.6629213483146 hold_3_5 - 225.3521126760563 hold_3_6
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_26_nic_in_4: switch_nic_out_26_nic_in_4 = 1 ->
   - 1263.157894736842 hold_4_1 - 657.5342465753424 hold_4_2
   - 444.4444444444444 hold_4_3 - 335.6643356643356 hold_4_4
   - 269.6629213483146 hold_4_5 - 225.3521126760563 hold_4_6
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_26_nic_in_5: switch_nic_out_26_nic_in_5 = 1 ->
   - 1263.157894736842 hold_5_1 - 657.5342465753424 hold_5_2
   - 444.4444444444444 hold_5_3 - 335.6643356643356 hold_5_4
   - 269.6629213483146 hold_5_5 - 225.3521126760563 hold_5_6
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_26_nic_in_6: switch_nic_out_26_nic_in_6 = 1 ->
   - 1263.157894736842 hold_6_1 - 657.5342465753424 hold_6_2
   - 444.4444444444444 hold_6_3 - 335.6643356643356 hold_6_4
   - 269.6629213483146 hold_6_5 - 225.3521126760563 hold_6_6
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_26_nic_in_7: switch_nic_out_26_nic_in_7 = 1 ->
   - 1263.157894736842 hold_7_1 - 657.5342465753424 hold_7_2
   - 444.4444444444444 hold_7_3 - 335.6643356643356 hold_7_4
   - 269.6629213483146 hold_7_5 - 225.3521126760563 hold_7_6
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_26_nic_in_8: switch_nic_out_26_nic_in_8 = 1 ->
   - 1263.157894736842 hold_8_1 - 657.5342465753424 hold_8_2
   - 444.4444444444444 hold_8_3 - 335.6643356643356 hold_8_4
   - 269.6629213483146 hold_8_5 - 225.3521126760563 hold_8_6
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_26_nic_in_9: switch_nic_out_26_nic_in_9 = 1 ->
   - 1263.157894736842 hold_9_1 - 657.5342465753424 hold_9_2
   - 444.4444444444444 hold_9_3 - 335.6643356643356 hold_9_4
   - 269.6629213483146 hold_9_5 - 225.3521126760563 hold_9_6
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_26_nic_in_10: switch_nic_out_26_nic_in_10 = 1 ->
   - 1263.157894736842 hold_10_1 - 657.5342465753424 hold_10_2
   - 444.4444444444444 hold_10_3 - 335.6643356643356 hold_10_4
   - 269.6629213483146 hold_10_5 - 225.3521126760563 hold_10_6
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_26_nic_in_11: switch_nic_out_26_nic_in_11 = 1 ->
   - 1263.157894736842 hold_11_1 - 657.5342465753424 hold_11_2
   - 444.4444444444444 hold_11_3 - 335.6643356643356 hold_11_4
   - 269.6629213483146 hold_11_5 - 225.3521126760563 hold_11_6
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_26_nic_in_12: switch_nic_out_26_nic_in_12 = 1 ->
   - 1263.157894736842 hold_12_1 - 657.5342465753424 hold_12_2
   - 444.4444444444444 hold_12_3 - 335.6643356643356 hold_12_4
   - 269.6629213483146 hold_12_5 - 225.3521126760563 hold_12_6
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_26_nic_in_13: switch_nic_out_26_nic_in_13 = 1 ->
   - 1263.157894736842 hold_13_1 - 657.5342465753424 hold_13_2
   - 444.4444444444444 hold_13_3 - 335.6643356643356 hold_13_4
   - 269.6629213483146 hold_13_5 - 225.3521126760563 hold_13_6
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_26_nic_in_14: switch_nic_out_26_nic_in_14 = 1 ->
   - 1263.157894736842 hold_14_1 - 657.5342465753424 hold_14_2
   - 444.4444444444444 hold_14_3 - 335.6643356643356 hold_14_4
   - 269.6629213483146 hold_14_5 - 225.3521126760563 hold_14_6
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_26_nic_in_15: switch_nic_out_26_nic_in_15 = 1 ->
   - 1263.157894736842 hold_15_1 - 657.5342465753424 hold_15_2
   - 444.4444444444444 hold_15_3 - 335.6643356643356 hold_15_4
   - 269.6629213483146 hold_15_5 - 225.3521126760563 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_16_nic_out_26_nic_in_16: switch_nic_out_26_nic_in_16 = 1 ->
   - 1263.157894736842 hold_16_1 - 657.5342465753424 hold_16_2
   - 444.4444444444444 hold_16_3 - 335.6643356643356 hold_16_4
   - 269.6629213483146 hold_16_5 - 225.3521126760563 hold_16_6
   + flow_nic_in_16_16 <= 0
 constr_intergroup_throughput_17_nic_out_26_nic_in_17: switch_nic_out_26_nic_in_17 = 1 ->
   - 1263.157894736842 hold_17_1 - 657.5342465753424 hold_17_2
   - 444.4444444444444 hold_17_3 - 335.6643356643356 hold_17_4
   - 269.6629213483146 hold_17_5 - 225.3521126760563 hold_17_6
   + flow_nic_in_17_17 <= 0
 constr_intergroup_throughput_18_nic_out_26_nic_in_18: switch_nic_out_26_nic_in_18 = 1 ->
   - 1263.157894736842 hold_18_1 - 657.5342465753424 hold_18_2
   - 444.4444444444444 hold_18_3 - 335.6643356643356 hold_18_4
   - 269.6629213483146 hold_18_5 - 225.3521126760563 hold_18_6
   + flow_nic_in_18_18 <= 0
 constr_intergroup_throughput_19_nic_out_26_nic_in_19: switch_nic_out_26_nic_in_19 = 1 ->
   - 1263.157894736842 hold_19_1 - 657.5342465753424 hold_19_2
   - 444.4444444444444 hold_19_3 - 335.6643356643356 hold_19_4
   - 269.6629213483146 hold_19_5 - 225.3521126760563 hold_19_6
   + flow_nic_in_19_19 <= 0
 constr_intergroup_throughput_20_nic_out_26_nic_in_20: switch_nic_out_26_nic_in_20 = 1 ->
   - 1263.157894736842 hold_20_1 - 657.5342465753424 hold_20_2
   - 444.4444444444444 hold_20_3 - 335.6643356643356 hold_20_4
   - 269.6629213483146 hold_20_5 - 225.3521126760563 hold_20_6
   + flow_nic_in_20_20 <= 0
 constr_intergroup_throughput_21_nic_out_26_nic_in_21: switch_nic_out_26_nic_in_21 = 1 ->
   - 1263.157894736842 hold_21_1 - 657.5342465753424 hold_21_2
   - 444.4444444444444 hold_21_3 - 335.6643356643356 hold_21_4
   - 269.6629213483146 hold_21_5 - 225.3521126760563 hold_21_6
   + flow_nic_in_21_21 <= 0
 constr_intergroup_throughput_22_nic_out_26_nic_in_22: switch_nic_out_26_nic_in_22 = 1 ->
   - 1263.157894736842 hold_22_1 - 657.5342465753424 hold_22_2
   - 444.4444444444444 hold_22_3 - 335.6643356643356 hold_22_4
   - 269.6629213483146 hold_22_5 - 225.3521126760563 hold_22_6
   + flow_nic_in_22_22 <= 0
 constr_intergroup_throughput_23_nic_out_26_nic_in_23: switch_nic_out_26_nic_in_23 = 1 ->
   - 1263.157894736842 hold_23_1 - 657.5342465753424 hold_23_2
   - 444.4444444444444 hold_23_3 - 335.6643356643356 hold_23_4
   - 269.6629213483146 hold_23_5 - 225.3521126760563 hold_23_6
   + flow_nic_in_23_23 <= 0
 constr_intergroup_throughput_24_nic_out_26_nic_in_24: switch_nic_out_26_nic_in_24 = 1 ->
   - 1263.157894736842 hold_24_1 - 657.5342465753424 hold_24_2
   - 444.4444444444444 hold_24_3 - 335.6643356643356 hold_24_4
   - 269.6629213483146 hold_24_5 - 225.3521126760563 hold_24_6
   + flow_nic_in_24_24 <= 0
 constr_intergroup_throughput_25_nic_out_26_nic_in_25: switch_nic_out_26_nic_in_25 = 1 ->
   - 1263.157894736842 hold_25_1 - 657.5342465753424 hold_25_2
   - 444.4444444444444 hold_25_3 - 335.6643356643356 hold_25_4
   - 269.6629213483146 hold_25_5 - 225.3521126760563 hold_25_6
   + flow_nic_in_25_25 <= 0
 constr_intergroup_throughput_27_nic_out_26_nic_in_27: switch_nic_out_26_nic_in_27 = 1 ->
   - 1263.157894736842 hold_27_1 - 657.5342465753424 hold_27_2
   - 444.4444444444444 hold_27_3 - 335.6643356643356 hold_27_4
   - 269.6629213483146 hold_27_5 - 225.3521126760563 hold_27_6
   + flow_nic_in_27_27 <= 0
 constr_intergroup_throughput_28_nic_out_26_nic_in_28: switch_nic_out_26_nic_in_28 = 1 ->
   - 1263.157894736842 hold_28_1 - 657.5342465753424 hold_28_2
   - 444.4444444444444 hold_28_3 - 335.6643356643356 hold_28_4
   - 269.6629213483146 hold_28_5 - 225.3521126760563 hold_28_6
   + flow_nic_in_28_28 <= 0
 constr_intergroup_throughput_29_nic_out_26_nic_in_29: switch_nic_out_26_nic_in_29 = 1 ->
   - 1263.157894736842 hold_29_1 - 657.5342465753424 hold_29_2
   - 444.4444444444444 hold_29_3 - 335.6643356643356 hold_29_4
   - 269.6629213483146 hold_29_5 - 225.3521126760563 hold_29_6
   + flow_nic_in_29_29 <= 0
 constr_intergroup_throughput_30_nic_out_26_nic_in_30: switch_nic_out_26_nic_in_30 = 1 ->
   - 1263.157894736842 hold_30_1 - 657.5342465753424 hold_30_2
   - 444.4444444444444 hold_30_3 - 335.6643356643356 hold_30_4
   - 269.6629213483146 hold_30_5 - 225.3521126760563 hold_30_6
   + flow_nic_in_30_30 <= 0
 constr_intergroup_throughput_31_nic_out_26_nic_in_31: switch_nic_out_26_nic_in_31 = 1 ->
   - 1263.157894736842 hold_31_1 - 657.5342465753424 hold_31_2
   - 444.4444444444444 hold_31_3 - 335.6643356643356 hold_31_4
   - 269.6629213483146 hold_31_5 - 225.3521126760563 hold_31_6
   + flow_nic_in_31_31 <= 0
 constr_intergroup_throughput_27_source_nic_in_27: switch_source_nic_in_27 = 1 ->
   - 1348.314606741573 hold_27_1 - 679.8866855524079 hold_27_2
   - 454.5454545454545 hold_27_3 - 341.3940256045519 hold_27_4
   - 273.3485193621868 hold_27_5 - 227.9202279202279 hold_27_6
   + flow_nic_in_27_27 <= 0
 constr_intergroup_throughput_0_nic_out_27_nic_in_0: switch_nic_out_27_nic_in_0 = 1 ->
   - 1263.157894736842 hold_0_1 - 657.5342465753424 hold_0_2
   - 444.4444444444444 hold_0_3 - 335.6643356643356 hold_0_4
   - 269.6629213483146 hold_0_5 - 225.3521126760563 hold_0_6
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_27_nic_in_1: switch_nic_out_27_nic_in_1 = 1 ->
   - 1263.157894736842 hold_1_1 - 657.5342465753424 hold_1_2
   - 444.4444444444444 hold_1_3 - 335.6643356643356 hold_1_4
   - 269.6629213483146 hold_1_5 - 225.3521126760563 hold_1_6
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_27_nic_in_2: switch_nic_out_27_nic_in_2 = 1 ->
   - 1263.157894736842 hold_2_1 - 657.5342465753424 hold_2_2
   - 444.4444444444444 hold_2_3 - 335.6643356643356 hold_2_4
   - 269.6629213483146 hold_2_5 - 225.3521126760563 hold_2_6
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_27_nic_in_3: switch_nic_out_27_nic_in_3 = 1 ->
   - 1263.157894736842 hold_3_1 - 657.5342465753424 hold_3_2
   - 444.4444444444444 hold_3_3 - 335.6643356643356 hold_3_4
   - 269.6629213483146 hold_3_5 - 225.3521126760563 hold_3_6
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_27_nic_in_4: switch_nic_out_27_nic_in_4 = 1 ->
   - 1263.157894736842 hold_4_1 - 657.5342465753424 hold_4_2
   - 444.4444444444444 hold_4_3 - 335.6643356643356 hold_4_4
   - 269.6629213483146 hold_4_5 - 225.3521126760563 hold_4_6
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_27_nic_in_5: switch_nic_out_27_nic_in_5 = 1 ->
   - 1263.157894736842 hold_5_1 - 657.5342465753424 hold_5_2
   - 444.4444444444444 hold_5_3 - 335.6643356643356 hold_5_4
   - 269.6629213483146 hold_5_5 - 225.3521126760563 hold_5_6
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_27_nic_in_6: switch_nic_out_27_nic_in_6 = 1 ->
   - 1263.157894736842 hold_6_1 - 657.5342465753424 hold_6_2
   - 444.4444444444444 hold_6_3 - 335.6643356643356 hold_6_4
   - 269.6629213483146 hold_6_5 - 225.3521126760563 hold_6_6
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_27_nic_in_7: switch_nic_out_27_nic_in_7 = 1 ->
   - 1263.157894736842 hold_7_1 - 657.5342465753424 hold_7_2
   - 444.4444444444444 hold_7_3 - 335.6643356643356 hold_7_4
   - 269.6629213483146 hold_7_5 - 225.3521126760563 hold_7_6
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_27_nic_in_8: switch_nic_out_27_nic_in_8 = 1 ->
   - 1263.157894736842 hold_8_1 - 657.5342465753424 hold_8_2
   - 444.4444444444444 hold_8_3 - 335.6643356643356 hold_8_4
   - 269.6629213483146 hold_8_5 - 225.3521126760563 hold_8_6
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_27_nic_in_9: switch_nic_out_27_nic_in_9 = 1 ->
   - 1263.157894736842 hold_9_1 - 657.5342465753424 hold_9_2
   - 444.4444444444444 hold_9_3 - 335.6643356643356 hold_9_4
   - 269.6629213483146 hold_9_5 - 225.3521126760563 hold_9_6
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_27_nic_in_10: switch_nic_out_27_nic_in_10 = 1 ->
   - 1263.157894736842 hold_10_1 - 657.5342465753424 hold_10_2
   - 444.4444444444444 hold_10_3 - 335.6643356643356 hold_10_4
   - 269.6629213483146 hold_10_5 - 225.3521126760563 hold_10_6
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_27_nic_in_11: switch_nic_out_27_nic_in_11 = 1 ->
   - 1263.157894736842 hold_11_1 - 657.5342465753424 hold_11_2
   - 444.4444444444444 hold_11_3 - 335.6643356643356 hold_11_4
   - 269.6629213483146 hold_11_5 - 225.3521126760563 hold_11_6
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_27_nic_in_12: switch_nic_out_27_nic_in_12 = 1 ->
   - 1263.157894736842 hold_12_1 - 657.5342465753424 hold_12_2
   - 444.4444444444444 hold_12_3 - 335.6643356643356 hold_12_4
   - 269.6629213483146 hold_12_5 - 225.3521126760563 hold_12_6
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_27_nic_in_13: switch_nic_out_27_nic_in_13 = 1 ->
   - 1263.157894736842 hold_13_1 - 657.5342465753424 hold_13_2
   - 444.4444444444444 hold_13_3 - 335.6643356643356 hold_13_4
   - 269.6629213483146 hold_13_5 - 225.3521126760563 hold_13_6
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_27_nic_in_14: switch_nic_out_27_nic_in_14 = 1 ->
   - 1263.157894736842 hold_14_1 - 657.5342465753424 hold_14_2
   - 444.4444444444444 hold_14_3 - 335.6643356643356 hold_14_4
   - 269.6629213483146 hold_14_5 - 225.3521126760563 hold_14_6
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_27_nic_in_15: switch_nic_out_27_nic_in_15 = 1 ->
   - 1263.157894736842 hold_15_1 - 657.5342465753424 hold_15_2
   - 444.4444444444444 hold_15_3 - 335.6643356643356 hold_15_4
   - 269.6629213483146 hold_15_5 - 225.3521126760563 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_16_nic_out_27_nic_in_16: switch_nic_out_27_nic_in_16 = 1 ->
   - 1263.157894736842 hold_16_1 - 657.5342465753424 hold_16_2
   - 444.4444444444444 hold_16_3 - 335.6643356643356 hold_16_4
   - 269.6629213483146 hold_16_5 - 225.3521126760563 hold_16_6
   + flow_nic_in_16_16 <= 0
 constr_intergroup_throughput_17_nic_out_27_nic_in_17: switch_nic_out_27_nic_in_17 = 1 ->
   - 1263.157894736842 hold_17_1 - 657.5342465753424 hold_17_2
   - 444.4444444444444 hold_17_3 - 335.6643356643356 hold_17_4
   - 269.6629213483146 hold_17_5 - 225.3521126760563 hold_17_6
   + flow_nic_in_17_17 <= 0
 constr_intergroup_throughput_18_nic_out_27_nic_in_18: switch_nic_out_27_nic_in_18 = 1 ->
   - 1263.157894736842 hold_18_1 - 657.5342465753424 hold_18_2
   - 444.4444444444444 hold_18_3 - 335.6643356643356 hold_18_4
   - 269.6629213483146 hold_18_5 - 225.3521126760563 hold_18_6
   + flow_nic_in_18_18 <= 0
 constr_intergroup_throughput_19_nic_out_27_nic_in_19: switch_nic_out_27_nic_in_19 = 1 ->
   - 1263.157894736842 hold_19_1 - 657.5342465753424 hold_19_2
   - 444.4444444444444 hold_19_3 - 335.6643356643356 hold_19_4
   - 269.6629213483146 hold_19_5 - 225.3521126760563 hold_19_6
   + flow_nic_in_19_19 <= 0
 constr_intergroup_throughput_20_nic_out_27_nic_in_20: switch_nic_out_27_nic_in_20 = 1 ->
   - 1263.157894736842 hold_20_1 - 657.5342465753424 hold_20_2
   - 444.4444444444444 hold_20_3 - 335.6643356643356 hold_20_4
   - 269.6629213483146 hold_20_5 - 225.3521126760563 hold_20_6
   + flow_nic_in_20_20 <= 0
 constr_intergroup_throughput_21_nic_out_27_nic_in_21: switch_nic_out_27_nic_in_21 = 1 ->
   - 1263.157894736842 hold_21_1 - 657.5342465753424 hold_21_2
   - 444.4444444444444 hold_21_3 - 335.6643356643356 hold_21_4
   - 269.6629213483146 hold_21_5 - 225.3521126760563 hold_21_6
   + flow_nic_in_21_21 <= 0
 constr_intergroup_throughput_22_nic_out_27_nic_in_22: switch_nic_out_27_nic_in_22 = 1 ->
   - 1263.157894736842 hold_22_1 - 657.5342465753424 hold_22_2
   - 444.4444444444444 hold_22_3 - 335.6643356643356 hold_22_4
   - 269.6629213483146 hold_22_5 - 225.3521126760563 hold_22_6
   + flow_nic_in_22_22 <= 0
 constr_intergroup_throughput_23_nic_out_27_nic_in_23: switch_nic_out_27_nic_in_23 = 1 ->
   - 1263.157894736842 hold_23_1 - 657.5342465753424 hold_23_2
   - 444.4444444444444 hold_23_3 - 335.6643356643356 hold_23_4
   - 269.6629213483146 hold_23_5 - 225.3521126760563 hold_23_6
   + flow_nic_in_23_23 <= 0
 constr_intergroup_throughput_24_nic_out_27_nic_in_24: switch_nic_out_27_nic_in_24 = 1 ->
   - 1263.157894736842 hold_24_1 - 657.5342465753424 hold_24_2
   - 444.4444444444444 hold_24_3 - 335.6643356643356 hold_24_4
   - 269.6629213483146 hold_24_5 - 225.3521126760563 hold_24_6
   + flow_nic_in_24_24 <= 0
 constr_intergroup_throughput_25_nic_out_27_nic_in_25: switch_nic_out_27_nic_in_25 = 1 ->
   - 1263.157894736842 hold_25_1 - 657.5342465753424 hold_25_2
   - 444.4444444444444 hold_25_3 - 335.6643356643356 hold_25_4
   - 269.6629213483146 hold_25_5 - 225.3521126760563 hold_25_6
   + flow_nic_in_25_25 <= 0
 constr_intergroup_throughput_26_nic_out_27_nic_in_26: switch_nic_out_27_nic_in_26 = 1 ->
   - 1263.157894736842 hold_26_1 - 657.5342465753424 hold_26_2
   - 444.4444444444444 hold_26_3 - 335.6643356643356 hold_26_4
   - 269.6629213483146 hold_26_5 - 225.3521126760563 hold_26_6
   + flow_nic_in_26_26 <= 0
 constr_intergroup_throughput_28_nic_out_27_nic_in_28: switch_nic_out_27_nic_in_28 = 1 ->
   - 1263.157894736842 hold_28_1 - 657.5342465753424 hold_28_2
   - 444.4444444444444 hold_28_3 - 335.6643356643356 hold_28_4
   - 269.6629213483146 hold_28_5 - 225.3521126760563 hold_28_6
   + flow_nic_in_28_28 <= 0
 constr_intergroup_throughput_29_nic_out_27_nic_in_29: switch_nic_out_27_nic_in_29 = 1 ->
   - 1263.157894736842 hold_29_1 - 657.5342465753424 hold_29_2
   - 444.4444444444444 hold_29_3 - 335.6643356643356 hold_29_4
   - 269.6629213483146 hold_29_5 - 225.3521126760563 hold_29_6
   + flow_nic_in_29_29 <= 0
 constr_intergroup_throughput_30_nic_out_27_nic_in_30: switch_nic_out_27_nic_in_30 = 1 ->
   - 1263.157894736842 hold_30_1 - 657.5342465753424 hold_30_2
   - 444.4444444444444 hold_30_3 - 335.6643356643356 hold_30_4
   - 269.6629213483146 hold_30_5 - 225.3521126760563 hold_30_6
   + flow_nic_in_30_30 <= 0
 constr_intergroup_throughput_31_nic_out_27_nic_in_31: switch_nic_out_27_nic_in_31 = 1 ->
   - 1263.157894736842 hold_31_1 - 657.5342465753424 hold_31_2
   - 444.4444444444444 hold_31_3 - 335.6643356643356 hold_31_4
   - 269.6629213483146 hold_31_5 - 225.3521126760563 hold_31_6
   + flow_nic_in_31_31 <= 0
 constr_intergroup_throughput_28_source_nic_in_28: switch_source_nic_in_28 = 1 ->
   - 1348.314606741573 hold_28_1 - 679.8866855524079 hold_28_2
   - 454.5454545454545 hold_28_3 - 341.3940256045519 hold_28_4
   - 273.3485193621868 hold_28_5 - 227.9202279202279 hold_28_6
   + flow_nic_in_28_28 <= 0
 constr_intergroup_throughput_0_nic_out_28_nic_in_0: switch_nic_out_28_nic_in_0 = 1 ->
   - 1263.157894736842 hold_0_1 - 657.5342465753424 hold_0_2
   - 444.4444444444444 hold_0_3 - 335.6643356643356 hold_0_4
   - 269.6629213483146 hold_0_5 - 225.3521126760563 hold_0_6
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_28_nic_in_1: switch_nic_out_28_nic_in_1 = 1 ->
   - 1263.157894736842 hold_1_1 - 657.5342465753424 hold_1_2
   - 444.4444444444444 hold_1_3 - 335.6643356643356 hold_1_4
   - 269.6629213483146 hold_1_5 - 225.3521126760563 hold_1_6
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_28_nic_in_2: switch_nic_out_28_nic_in_2 = 1 ->
   - 1263.157894736842 hold_2_1 - 657.5342465753424 hold_2_2
   - 444.4444444444444 hold_2_3 - 335.6643356643356 hold_2_4
   - 269.6629213483146 hold_2_5 - 225.3521126760563 hold_2_6
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_28_nic_in_3: switch_nic_out_28_nic_in_3 = 1 ->
   - 1263.157894736842 hold_3_1 - 657.5342465753424 hold_3_2
   - 444.4444444444444 hold_3_3 - 335.6643356643356 hold_3_4
   - 269.6629213483146 hold_3_5 - 225.3521126760563 hold_3_6
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_28_nic_in_4: switch_nic_out_28_nic_in_4 = 1 ->
   - 1263.157894736842 hold_4_1 - 657.5342465753424 hold_4_2
   - 444.4444444444444 hold_4_3 - 335.6643356643356 hold_4_4
   - 269.6629213483146 hold_4_5 - 225.3521126760563 hold_4_6
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_28_nic_in_5: switch_nic_out_28_nic_in_5 = 1 ->
   - 1263.157894736842 hold_5_1 - 657.5342465753424 hold_5_2
   - 444.4444444444444 hold_5_3 - 335.6643356643356 hold_5_4
   - 269.6629213483146 hold_5_5 - 225.3521126760563 hold_5_6
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_28_nic_in_6: switch_nic_out_28_nic_in_6 = 1 ->
   - 1263.157894736842 hold_6_1 - 657.5342465753424 hold_6_2
   - 444.4444444444444 hold_6_3 - 335.6643356643356 hold_6_4
   - 269.6629213483146 hold_6_5 - 225.3521126760563 hold_6_6
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_28_nic_in_7: switch_nic_out_28_nic_in_7 = 1 ->
   - 1263.157894736842 hold_7_1 - 657.5342465753424 hold_7_2
   - 444.4444444444444 hold_7_3 - 335.6643356643356 hold_7_4
   - 269.6629213483146 hold_7_5 - 225.3521126760563 hold_7_6
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_28_nic_in_8: switch_nic_out_28_nic_in_8 = 1 ->
   - 1263.157894736842 hold_8_1 - 657.5342465753424 hold_8_2
   - 444.4444444444444 hold_8_3 - 335.6643356643356 hold_8_4
   - 269.6629213483146 hold_8_5 - 225.3521126760563 hold_8_6
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_28_nic_in_9: switch_nic_out_28_nic_in_9 = 1 ->
   - 1263.157894736842 hold_9_1 - 657.5342465753424 hold_9_2
   - 444.4444444444444 hold_9_3 - 335.6643356643356 hold_9_4
   - 269.6629213483146 hold_9_5 - 225.3521126760563 hold_9_6
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_28_nic_in_10: switch_nic_out_28_nic_in_10 = 1 ->
   - 1263.157894736842 hold_10_1 - 657.5342465753424 hold_10_2
   - 444.4444444444444 hold_10_3 - 335.6643356643356 hold_10_4
   - 269.6629213483146 hold_10_5 - 225.3521126760563 hold_10_6
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_28_nic_in_11: switch_nic_out_28_nic_in_11 = 1 ->
   - 1263.157894736842 hold_11_1 - 657.5342465753424 hold_11_2
   - 444.4444444444444 hold_11_3 - 335.6643356643356 hold_11_4
   - 269.6629213483146 hold_11_5 - 225.3521126760563 hold_11_6
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_28_nic_in_12: switch_nic_out_28_nic_in_12 = 1 ->
   - 1263.157894736842 hold_12_1 - 657.5342465753424 hold_12_2
   - 444.4444444444444 hold_12_3 - 335.6643356643356 hold_12_4
   - 269.6629213483146 hold_12_5 - 225.3521126760563 hold_12_6
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_28_nic_in_13: switch_nic_out_28_nic_in_13 = 1 ->
   - 1263.157894736842 hold_13_1 - 657.5342465753424 hold_13_2
   - 444.4444444444444 hold_13_3 - 335.6643356643356 hold_13_4
   - 269.6629213483146 hold_13_5 - 225.3521126760563 hold_13_6
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_28_nic_in_14: switch_nic_out_28_nic_in_14 = 1 ->
   - 1263.157894736842 hold_14_1 - 657.5342465753424 hold_14_2
   - 444.4444444444444 hold_14_3 - 335.6643356643356 hold_14_4
   - 269.6629213483146 hold_14_5 - 225.3521126760563 hold_14_6
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_28_nic_in_15: switch_nic_out_28_nic_in_15 = 1 ->
   - 1263.157894736842 hold_15_1 - 657.5342465753424 hold_15_2
   - 444.4444444444444 hold_15_3 - 335.6643356643356 hold_15_4
   - 269.6629213483146 hold_15_5 - 225.3521126760563 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_16_nic_out_28_nic_in_16: switch_nic_out_28_nic_in_16 = 1 ->
   - 1263.157894736842 hold_16_1 - 657.5342465753424 hold_16_2
   - 444.4444444444444 hold_16_3 - 335.6643356643356 hold_16_4
   - 269.6629213483146 hold_16_5 - 225.3521126760563 hold_16_6
   + flow_nic_in_16_16 <= 0
 constr_intergroup_throughput_17_nic_out_28_nic_in_17: switch_nic_out_28_nic_in_17 = 1 ->
   - 1263.157894736842 hold_17_1 - 657.5342465753424 hold_17_2
   - 444.4444444444444 hold_17_3 - 335.6643356643356 hold_17_4
   - 269.6629213483146 hold_17_5 - 225.3521126760563 hold_17_6
   + flow_nic_in_17_17 <= 0
 constr_intergroup_throughput_18_nic_out_28_nic_in_18: switch_nic_out_28_nic_in_18 = 1 ->
   - 1263.157894736842 hold_18_1 - 657.5342465753424 hold_18_2
   - 444.4444444444444 hold_18_3 - 335.6643356643356 hold_18_4
   - 269.6629213483146 hold_18_5 - 225.3521126760563 hold_18_6
   + flow_nic_in_18_18 <= 0
 constr_intergroup_throughput_19_nic_out_28_nic_in_19: switch_nic_out_28_nic_in_19 = 1 ->
   - 1263.157894736842 hold_19_1 - 657.5342465753424 hold_19_2
   - 444.4444444444444 hold_19_3 - 335.6643356643356 hold_19_4
   - 269.6629213483146 hold_19_5 - 225.3521126760563 hold_19_6
   + flow_nic_in_19_19 <= 0
 constr_intergroup_throughput_20_nic_out_28_nic_in_20: switch_nic_out_28_nic_in_20 = 1 ->
   - 1263.157894736842 hold_20_1 - 657.5342465753424 hold_20_2
   - 444.4444444444444 hold_20_3 - 335.6643356643356 hold_20_4
   - 269.6629213483146 hold_20_5 - 225.3521126760563 hold_20_6
   + flow_nic_in_20_20 <= 0
 constr_intergroup_throughput_21_nic_out_28_nic_in_21: switch_nic_out_28_nic_in_21 = 1 ->
   - 1263.157894736842 hold_21_1 - 657.5342465753424 hold_21_2
   - 444.4444444444444 hold_21_3 - 335.6643356643356 hold_21_4
   - 269.6629213483146 hold_21_5 - 225.3521126760563 hold_21_6
   + flow_nic_in_21_21 <= 0
 constr_intergroup_throughput_22_nic_out_28_nic_in_22: switch_nic_out_28_nic_in_22 = 1 ->
   - 1263.157894736842 hold_22_1 - 657.5342465753424 hold_22_2
   - 444.4444444444444 hold_22_3 - 335.6643356643356 hold_22_4
   - 269.6629213483146 hold_22_5 - 225.3521126760563 hold_22_6
   + flow_nic_in_22_22 <= 0
 constr_intergroup_throughput_23_nic_out_28_nic_in_23: switch_nic_out_28_nic_in_23 = 1 ->
   - 1263.157894736842 hold_23_1 - 657.5342465753424 hold_23_2
   - 444.4444444444444 hold_23_3 - 335.6643356643356 hold_23_4
   - 269.6629213483146 hold_23_5 - 225.3521126760563 hold_23_6
   + flow_nic_in_23_23 <= 0
 constr_intergroup_throughput_24_nic_out_28_nic_in_24: switch_nic_out_28_nic_in_24 = 1 ->
   - 1263.157894736842 hold_24_1 - 657.5342465753424 hold_24_2
   - 444.4444444444444 hold_24_3 - 335.6643356643356 hold_24_4
   - 269.6629213483146 hold_24_5 - 225.3521126760563 hold_24_6
   + flow_nic_in_24_24 <= 0
 constr_intergroup_throughput_25_nic_out_28_nic_in_25: switch_nic_out_28_nic_in_25 = 1 ->
   - 1263.157894736842 hold_25_1 - 657.5342465753424 hold_25_2
   - 444.4444444444444 hold_25_3 - 335.6643356643356 hold_25_4
   - 269.6629213483146 hold_25_5 - 225.3521126760563 hold_25_6
   + flow_nic_in_25_25 <= 0
 constr_intergroup_throughput_26_nic_out_28_nic_in_26: switch_nic_out_28_nic_in_26 = 1 ->
   - 1263.157894736842 hold_26_1 - 657.5342465753424 hold_26_2
   - 444.4444444444444 hold_26_3 - 335.6643356643356 hold_26_4
   - 269.6629213483146 hold_26_5 - 225.3521126760563 hold_26_6
   + flow_nic_in_26_26 <= 0
 constr_intergroup_throughput_27_nic_out_28_nic_in_27: switch_nic_out_28_nic_in_27 = 1 ->
   - 1263.157894736842 hold_27_1 - 657.5342465753424 hold_27_2
   - 444.4444444444444 hold_27_3 - 335.6643356643356 hold_27_4
   - 269.6629213483146 hold_27_5 - 225.3521126760563 hold_27_6
   + flow_nic_in_27_27 <= 0
 constr_intergroup_throughput_29_nic_out_28_nic_in_29: switch_nic_out_28_nic_in_29 = 1 ->
   - 1263.157894736842 hold_29_1 - 657.5342465753424 hold_29_2
   - 444.4444444444444 hold_29_3 - 335.6643356643356 hold_29_4
   - 269.6629213483146 hold_29_5 - 225.3521126760563 hold_29_6
   + flow_nic_in_29_29 <= 0
 constr_intergroup_throughput_30_nic_out_28_nic_in_30: switch_nic_out_28_nic_in_30 = 1 ->
   - 1263.157894736842 hold_30_1 - 657.5342465753424 hold_30_2
   - 444.4444444444444 hold_30_3 - 335.6643356643356 hold_30_4
   - 269.6629213483146 hold_30_5 - 225.3521126760563 hold_30_6
   + flow_nic_in_30_30 <= 0
 constr_intergroup_throughput_31_nic_out_28_nic_in_31: switch_nic_out_28_nic_in_31 = 1 ->
   - 1263.157894736842 hold_31_1 - 657.5342465753424 hold_31_2
   - 444.4444444444444 hold_31_3 - 335.6643356643356 hold_31_4
   - 269.6629213483146 hold_31_5 - 225.3521126760563 hold_31_6
   + flow_nic_in_31_31 <= 0
 constr_intergroup_throughput_29_source_nic_in_29: switch_source_nic_in_29 = 1 ->
   - 1348.314606741573 hold_29_1 - 679.8866855524079 hold_29_2
   - 454.5454545454545 hold_29_3 - 341.3940256045519 hold_29_4
   - 273.3485193621868 hold_29_5 - 227.9202279202279 hold_29_6
   + flow_nic_in_29_29 <= 0
 constr_intergroup_throughput_0_nic_out_29_nic_in_0: switch_nic_out_29_nic_in_0 = 1 ->
   - 1263.157894736842 hold_0_1 - 657.5342465753424 hold_0_2
   - 444.4444444444444 hold_0_3 - 335.6643356643356 hold_0_4
   - 269.6629213483146 hold_0_5 - 225.3521126760563 hold_0_6
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_29_nic_in_1: switch_nic_out_29_nic_in_1 = 1 ->
   - 1263.157894736842 hold_1_1 - 657.5342465753424 hold_1_2
   - 444.4444444444444 hold_1_3 - 335.6643356643356 hold_1_4
   - 269.6629213483146 hold_1_5 - 225.3521126760563 hold_1_6
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_29_nic_in_2: switch_nic_out_29_nic_in_2 = 1 ->
   - 1263.157894736842 hold_2_1 - 657.5342465753424 hold_2_2
   - 444.4444444444444 hold_2_3 - 335.6643356643356 hold_2_4
   - 269.6629213483146 hold_2_5 - 225.3521126760563 hold_2_6
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_29_nic_in_3: switch_nic_out_29_nic_in_3 = 1 ->
   - 1263.157894736842 hold_3_1 - 657.5342465753424 hold_3_2
   - 444.4444444444444 hold_3_3 - 335.6643356643356 hold_3_4
   - 269.6629213483146 hold_3_5 - 225.3521126760563 hold_3_6
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_29_nic_in_4: switch_nic_out_29_nic_in_4 = 1 ->
   - 1263.157894736842 hold_4_1 - 657.5342465753424 hold_4_2
   - 444.4444444444444 hold_4_3 - 335.6643356643356 hold_4_4
   - 269.6629213483146 hold_4_5 - 225.3521126760563 hold_4_6
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_29_nic_in_5: switch_nic_out_29_nic_in_5 = 1 ->
   - 1263.157894736842 hold_5_1 - 657.5342465753424 hold_5_2
   - 444.4444444444444 hold_5_3 - 335.6643356643356 hold_5_4
   - 269.6629213483146 hold_5_5 - 225.3521126760563 hold_5_6
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_29_nic_in_6: switch_nic_out_29_nic_in_6 = 1 ->
   - 1263.157894736842 hold_6_1 - 657.5342465753424 hold_6_2
   - 444.4444444444444 hold_6_3 - 335.6643356643356 hold_6_4
   - 269.6629213483146 hold_6_5 - 225.3521126760563 hold_6_6
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_29_nic_in_7: switch_nic_out_29_nic_in_7 = 1 ->
   - 1263.157894736842 hold_7_1 - 657.5342465753424 hold_7_2
   - 444.4444444444444 hold_7_3 - 335.6643356643356 hold_7_4
   - 269.6629213483146 hold_7_5 - 225.3521126760563 hold_7_6
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_29_nic_in_8: switch_nic_out_29_nic_in_8 = 1 ->
   - 1263.157894736842 hold_8_1 - 657.5342465753424 hold_8_2
   - 444.4444444444444 hold_8_3 - 335.6643356643356 hold_8_4
   - 269.6629213483146 hold_8_5 - 225.3521126760563 hold_8_6
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_29_nic_in_9: switch_nic_out_29_nic_in_9 = 1 ->
   - 1263.157894736842 hold_9_1 - 657.5342465753424 hold_9_2
   - 444.4444444444444 hold_9_3 - 335.6643356643356 hold_9_4
   - 269.6629213483146 hold_9_5 - 225.3521126760563 hold_9_6
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_29_nic_in_10: switch_nic_out_29_nic_in_10 = 1 ->
   - 1263.157894736842 hold_10_1 - 657.5342465753424 hold_10_2
   - 444.4444444444444 hold_10_3 - 335.6643356643356 hold_10_4
   - 269.6629213483146 hold_10_5 - 225.3521126760563 hold_10_6
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_29_nic_in_11: switch_nic_out_29_nic_in_11 = 1 ->
   - 1263.157894736842 hold_11_1 - 657.5342465753424 hold_11_2
   - 444.4444444444444 hold_11_3 - 335.6643356643356 hold_11_4
   - 269.6629213483146 hold_11_5 - 225.3521126760563 hold_11_6
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_29_nic_in_12: switch_nic_out_29_nic_in_12 = 1 ->
   - 1263.157894736842 hold_12_1 - 657.5342465753424 hold_12_2
   - 444.4444444444444 hold_12_3 - 335.6643356643356 hold_12_4
   - 269.6629213483146 hold_12_5 - 225.3521126760563 hold_12_6
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_29_nic_in_13: switch_nic_out_29_nic_in_13 = 1 ->
   - 1263.157894736842 hold_13_1 - 657.5342465753424 hold_13_2
   - 444.4444444444444 hold_13_3 - 335.6643356643356 hold_13_4
   - 269.6629213483146 hold_13_5 - 225.3521126760563 hold_13_6
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_29_nic_in_14: switch_nic_out_29_nic_in_14 = 1 ->
   - 1263.157894736842 hold_14_1 - 657.5342465753424 hold_14_2
   - 444.4444444444444 hold_14_3 - 335.6643356643356 hold_14_4
   - 269.6629213483146 hold_14_5 - 225.3521126760563 hold_14_6
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_29_nic_in_15: switch_nic_out_29_nic_in_15 = 1 ->
   - 1263.157894736842 hold_15_1 - 657.5342465753424 hold_15_2
   - 444.4444444444444 hold_15_3 - 335.6643356643356 hold_15_4
   - 269.6629213483146 hold_15_5 - 225.3521126760563 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_16_nic_out_29_nic_in_16: switch_nic_out_29_nic_in_16 = 1 ->
   - 1263.157894736842 hold_16_1 - 657.5342465753424 hold_16_2
   - 444.4444444444444 hold_16_3 - 335.6643356643356 hold_16_4
   - 269.6629213483146 hold_16_5 - 225.3521126760563 hold_16_6
   + flow_nic_in_16_16 <= 0
 constr_intergroup_throughput_17_nic_out_29_nic_in_17: switch_nic_out_29_nic_in_17 = 1 ->
   - 1263.157894736842 hold_17_1 - 657.5342465753424 hold_17_2
   - 444.4444444444444 hold_17_3 - 335.6643356643356 hold_17_4
   - 269.6629213483146 hold_17_5 - 225.3521126760563 hold_17_6
   + flow_nic_in_17_17 <= 0
 constr_intergroup_throughput_18_nic_out_29_nic_in_18: switch_nic_out_29_nic_in_18 = 1 ->
   - 1263.157894736842 hold_18_1 - 657.5342465753424 hold_18_2
   - 444.4444444444444 hold_18_3 - 335.6643356643356 hold_18_4
   - 269.6629213483146 hold_18_5 - 225.3521126760563 hold_18_6
   + flow_nic_in_18_18 <= 0
 constr_intergroup_throughput_19_nic_out_29_nic_in_19: switch_nic_out_29_nic_in_19 = 1 ->
   - 1263.157894736842 hold_19_1 - 657.5342465753424 hold_19_2
   - 444.4444444444444 hold_19_3 - 335.6643356643356 hold_19_4
   - 269.6629213483146 hold_19_5 - 225.3521126760563 hold_19_6
   + flow_nic_in_19_19 <= 0
 constr_intergroup_throughput_20_nic_out_29_nic_in_20: switch_nic_out_29_nic_in_20 = 1 ->
   - 1263.157894736842 hold_20_1 - 657.5342465753424 hold_20_2
   - 444.4444444444444 hold_20_3 - 335.6643356643356 hold_20_4
   - 269.6629213483146 hold_20_5 - 225.3521126760563 hold_20_6
   + flow_nic_in_20_20 <= 0
 constr_intergroup_throughput_21_nic_out_29_nic_in_21: switch_nic_out_29_nic_in_21 = 1 ->
   - 1263.157894736842 hold_21_1 - 657.5342465753424 hold_21_2
   - 444.4444444444444 hold_21_3 - 335.6643356643356 hold_21_4
   - 269.6629213483146 hold_21_5 - 225.3521126760563 hold_21_6
   + flow_nic_in_21_21 <= 0
 constr_intergroup_throughput_22_nic_out_29_nic_in_22: switch_nic_out_29_nic_in_22 = 1 ->
   - 1263.157894736842 hold_22_1 - 657.5342465753424 hold_22_2
   - 444.4444444444444 hold_22_3 - 335.6643356643356 hold_22_4
   - 269.6629213483146 hold_22_5 - 225.3521126760563 hold_22_6
   + flow_nic_in_22_22 <= 0
 constr_intergroup_throughput_23_nic_out_29_nic_in_23: switch_nic_out_29_nic_in_23 = 1 ->
   - 1263.157894736842 hold_23_1 - 657.5342465753424 hold_23_2
   - 444.4444444444444 hold_23_3 - 335.6643356643356 hold_23_4
   - 269.6629213483146 hold_23_5 - 225.3521126760563 hold_23_6
   + flow_nic_in_23_23 <= 0
 constr_intergroup_throughput_24_nic_out_29_nic_in_24: switch_nic_out_29_nic_in_24 = 1 ->
   - 1263.157894736842 hold_24_1 - 657.5342465753424 hold_24_2
   - 444.4444444444444 hold_24_3 - 335.6643356643356 hold_24_4
   - 269.6629213483146 hold_24_5 - 225.3521126760563 hold_24_6
   + flow_nic_in_24_24 <= 0
 constr_intergroup_throughput_25_nic_out_29_nic_in_25: switch_nic_out_29_nic_in_25 = 1 ->
   - 1263.157894736842 hold_25_1 - 657.5342465753424 hold_25_2
   - 444.4444444444444 hold_25_3 - 335.6643356643356 hold_25_4
   - 269.6629213483146 hold_25_5 - 225.3521126760563 hold_25_6
   + flow_nic_in_25_25 <= 0
 constr_intergroup_throughput_26_nic_out_29_nic_in_26: switch_nic_out_29_nic_in_26 = 1 ->
   - 1263.157894736842 hold_26_1 - 657.5342465753424 hold_26_2
   - 444.4444444444444 hold_26_3 - 335.6643356643356 hold_26_4
   - 269.6629213483146 hold_26_5 - 225.3521126760563 hold_26_6
   + flow_nic_in_26_26 <= 0
 constr_intergroup_throughput_27_nic_out_29_nic_in_27: switch_nic_out_29_nic_in_27 = 1 ->
   - 1263.157894736842 hold_27_1 - 657.5342465753424 hold_27_2
   - 444.4444444444444 hold_27_3 - 335.6643356643356 hold_27_4
   - 269.6629213483146 hold_27_5 - 225.3521126760563 hold_27_6
   + flow_nic_in_27_27 <= 0
 constr_intergroup_throughput_28_nic_out_29_nic_in_28: switch_nic_out_29_nic_in_28 = 1 ->
   - 1263.157894736842 hold_28_1 - 657.5342465753424 hold_28_2
   - 444.4444444444444 hold_28_3 - 335.6643356643356 hold_28_4
   - 269.6629213483146 hold_28_5 - 225.3521126760563 hold_28_6
   + flow_nic_in_28_28 <= 0
 constr_intergroup_throughput_30_nic_out_29_nic_in_30: switch_nic_out_29_nic_in_30 = 1 ->
   - 1263.157894736842 hold_30_1 - 657.5342465753424 hold_30_2
   - 444.4444444444444 hold_30_3 - 335.6643356643356 hold_30_4
   - 269.6629213483146 hold_30_5 - 225.3521126760563 hold_30_6
   + flow_nic_in_30_30 <= 0
 constr_intergroup_throughput_31_nic_out_29_nic_in_31: switch_nic_out_29_nic_in_31 = 1 ->
   - 1263.157894736842 hold_31_1 - 657.5342465753424 hold_31_2
   - 444.4444444444444 hold_31_3 - 335.6643356643356 hold_31_4
   - 269.6629213483146 hold_31_5 - 225.3521126760563 hold_31_6
   + flow_nic_in_31_31 <= 0
 constr_intergroup_throughput_30_source_nic_in_30: switch_source_nic_in_30 = 1 ->
   - 1348.314606741573 hold_30_1 - 679.8866855524079 hold_30_2
   - 454.5454545454545 hold_30_3 - 341.3940256045519 hold_30_4
   - 273.3485193621868 hold_30_5 - 227.9202279202279 hold_30_6
   + flow_nic_in_30_30 <= 0
 constr_intergroup_throughput_0_nic_out_30_nic_in_0: switch_nic_out_30_nic_in_0 = 1 ->
   - 1263.157894736842 hold_0_1 - 657.5342465753424 hold_0_2
   - 444.4444444444444 hold_0_3 - 335.6643356643356 hold_0_4
   - 269.6629213483146 hold_0_5 - 225.3521126760563 hold_0_6
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_30_nic_in_1: switch_nic_out_30_nic_in_1 = 1 ->
   - 1263.157894736842 hold_1_1 - 657.5342465753424 hold_1_2
   - 444.4444444444444 hold_1_3 - 335.6643356643356 hold_1_4
   - 269.6629213483146 hold_1_5 - 225.3521126760563 hold_1_6
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_30_nic_in_2: switch_nic_out_30_nic_in_2 = 1 ->
   - 1263.157894736842 hold_2_1 - 657.5342465753424 hold_2_2
   - 444.4444444444444 hold_2_3 - 335.6643356643356 hold_2_4
   - 269.6629213483146 hold_2_5 - 225.3521126760563 hold_2_6
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_30_nic_in_3: switch_nic_out_30_nic_in_3 = 1 ->
   - 1263.157894736842 hold_3_1 - 657.5342465753424 hold_3_2
   - 444.4444444444444 hold_3_3 - 335.6643356643356 hold_3_4
   - 269.6629213483146 hold_3_5 - 225.3521126760563 hold_3_6
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_30_nic_in_4: switch_nic_out_30_nic_in_4 = 1 ->
   - 1263.157894736842 hold_4_1 - 657.5342465753424 hold_4_2
   - 444.4444444444444 hold_4_3 - 335.6643356643356 hold_4_4
   - 269.6629213483146 hold_4_5 - 225.3521126760563 hold_4_6
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_30_nic_in_5: switch_nic_out_30_nic_in_5 = 1 ->
   - 1263.157894736842 hold_5_1 - 657.5342465753424 hold_5_2
   - 444.4444444444444 hold_5_3 - 335.6643356643356 hold_5_4
   - 269.6629213483146 hold_5_5 - 225.3521126760563 hold_5_6
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_30_nic_in_6: switch_nic_out_30_nic_in_6 = 1 ->
   - 1263.157894736842 hold_6_1 - 657.5342465753424 hold_6_2
   - 444.4444444444444 hold_6_3 - 335.6643356643356 hold_6_4
   - 269.6629213483146 hold_6_5 - 225.3521126760563 hold_6_6
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_30_nic_in_7: switch_nic_out_30_nic_in_7 = 1 ->
   - 1263.157894736842 hold_7_1 - 657.5342465753424 hold_7_2
   - 444.4444444444444 hold_7_3 - 335.6643356643356 hold_7_4
   - 269.6629213483146 hold_7_5 - 225.3521126760563 hold_7_6
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_30_nic_in_8: switch_nic_out_30_nic_in_8 = 1 ->
   - 1263.157894736842 hold_8_1 - 657.5342465753424 hold_8_2
   - 444.4444444444444 hold_8_3 - 335.6643356643356 hold_8_4
   - 269.6629213483146 hold_8_5 - 225.3521126760563 hold_8_6
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_30_nic_in_9: switch_nic_out_30_nic_in_9 = 1 ->
   - 1263.157894736842 hold_9_1 - 657.5342465753424 hold_9_2
   - 444.4444444444444 hold_9_3 - 335.6643356643356 hold_9_4
   - 269.6629213483146 hold_9_5 - 225.3521126760563 hold_9_6
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_30_nic_in_10: switch_nic_out_30_nic_in_10 = 1 ->
   - 1263.157894736842 hold_10_1 - 657.5342465753424 hold_10_2
   - 444.4444444444444 hold_10_3 - 335.6643356643356 hold_10_4
   - 269.6629213483146 hold_10_5 - 225.3521126760563 hold_10_6
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_30_nic_in_11: switch_nic_out_30_nic_in_11 = 1 ->
   - 1263.157894736842 hold_11_1 - 657.5342465753424 hold_11_2
   - 444.4444444444444 hold_11_3 - 335.6643356643356 hold_11_4
   - 269.6629213483146 hold_11_5 - 225.3521126760563 hold_11_6
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_30_nic_in_12: switch_nic_out_30_nic_in_12 = 1 ->
   - 1263.157894736842 hold_12_1 - 657.5342465753424 hold_12_2
   - 444.4444444444444 hold_12_3 - 335.6643356643356 hold_12_4
   - 269.6629213483146 hold_12_5 - 225.3521126760563 hold_12_6
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_30_nic_in_13: switch_nic_out_30_nic_in_13 = 1 ->
   - 1263.157894736842 hold_13_1 - 657.5342465753424 hold_13_2
   - 444.4444444444444 hold_13_3 - 335.6643356643356 hold_13_4
   - 269.6629213483146 hold_13_5 - 225.3521126760563 hold_13_6
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_30_nic_in_14: switch_nic_out_30_nic_in_14 = 1 ->
   - 1263.157894736842 hold_14_1 - 657.5342465753424 hold_14_2
   - 444.4444444444444 hold_14_3 - 335.6643356643356 hold_14_4
   - 269.6629213483146 hold_14_5 - 225.3521126760563 hold_14_6
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_30_nic_in_15: switch_nic_out_30_nic_in_15 = 1 ->
   - 1263.157894736842 hold_15_1 - 657.5342465753424 hold_15_2
   - 444.4444444444444 hold_15_3 - 335.6643356643356 hold_15_4
   - 269.6629213483146 hold_15_5 - 225.3521126760563 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_16_nic_out_30_nic_in_16: switch_nic_out_30_nic_in_16 = 1 ->
   - 1263.157894736842 hold_16_1 - 657.5342465753424 hold_16_2
   - 444.4444444444444 hold_16_3 - 335.6643356643356 hold_16_4
   - 269.6629213483146 hold_16_5 - 225.3521126760563 hold_16_6
   + flow_nic_in_16_16 <= 0
 constr_intergroup_throughput_17_nic_out_30_nic_in_17: switch_nic_out_30_nic_in_17 = 1 ->
   - 1263.157894736842 hold_17_1 - 657.5342465753424 hold_17_2
   - 444.4444444444444 hold_17_3 - 335.6643356643356 hold_17_4
   - 269.6629213483146 hold_17_5 - 225.3521126760563 hold_17_6
   + flow_nic_in_17_17 <= 0
 constr_intergroup_throughput_18_nic_out_30_nic_in_18: switch_nic_out_30_nic_in_18 = 1 ->
   - 1263.157894736842 hold_18_1 - 657.5342465753424 hold_18_2
   - 444.4444444444444 hold_18_3 - 335.6643356643356 hold_18_4
   - 269.6629213483146 hold_18_5 - 225.3521126760563 hold_18_6
   + flow_nic_in_18_18 <= 0
 constr_intergroup_throughput_19_nic_out_30_nic_in_19: switch_nic_out_30_nic_in_19 = 1 ->
   - 1263.157894736842 hold_19_1 - 657.5342465753424 hold_19_2
   - 444.4444444444444 hold_19_3 - 335.6643356643356 hold_19_4
   - 269.6629213483146 hold_19_5 - 225.3521126760563 hold_19_6
   + flow_nic_in_19_19 <= 0
 constr_intergroup_throughput_20_nic_out_30_nic_in_20: switch_nic_out_30_nic_in_20 = 1 ->
   - 1263.157894736842 hold_20_1 - 657.5342465753424 hold_20_2
   - 444.4444444444444 hold_20_3 - 335.6643356643356 hold_20_4
   - 269.6629213483146 hold_20_5 - 225.3521126760563 hold_20_6
   + flow_nic_in_20_20 <= 0
 constr_intergroup_throughput_21_nic_out_30_nic_in_21: switch_nic_out_30_nic_in_21 = 1 ->
   - 1263.157894736842 hold_21_1 - 657.5342465753424 hold_21_2
   - 444.4444444444444 hold_21_3 - 335.6643356643356 hold_21_4
   - 269.6629213483146 hold_21_5 - 225.3521126760563 hold_21_6
   + flow_nic_in_21_21 <= 0
 constr_intergroup_throughput_22_nic_out_30_nic_in_22: switch_nic_out_30_nic_in_22 = 1 ->
   - 1263.157894736842 hold_22_1 - 657.5342465753424 hold_22_2
   - 444.4444444444444 hold_22_3 - 335.6643356643356 hold_22_4
   - 269.6629213483146 hold_22_5 - 225.3521126760563 hold_22_6
   + flow_nic_in_22_22 <= 0
 constr_intergroup_throughput_23_nic_out_30_nic_in_23: switch_nic_out_30_nic_in_23 = 1 ->
   - 1263.157894736842 hold_23_1 - 657.5342465753424 hold_23_2
   - 444.4444444444444 hold_23_3 - 335.6643356643356 hold_23_4
   - 269.6629213483146 hold_23_5 - 225.3521126760563 hold_23_6
   + flow_nic_in_23_23 <= 0
 constr_intergroup_throughput_24_nic_out_30_nic_in_24: switch_nic_out_30_nic_in_24 = 1 ->
   - 1263.157894736842 hold_24_1 - 657.5342465753424 hold_24_2
   - 444.4444444444444 hold_24_3 - 335.6643356643356 hold_24_4
   - 269.6629213483146 hold_24_5 - 225.3521126760563 hold_24_6
   + flow_nic_in_24_24 <= 0
 constr_intergroup_throughput_25_nic_out_30_nic_in_25: switch_nic_out_30_nic_in_25 = 1 ->
   - 1263.157894736842 hold_25_1 - 657.5342465753424 hold_25_2
   - 444.4444444444444 hold_25_3 - 335.6643356643356 hold_25_4
   - 269.6629213483146 hold_25_5 - 225.3521126760563 hold_25_6
   + flow_nic_in_25_25 <= 0
 constr_intergroup_throughput_26_nic_out_30_nic_in_26: switch_nic_out_30_nic_in_26 = 1 ->
   - 1263.157894736842 hold_26_1 - 657.5342465753424 hold_26_2
   - 444.4444444444444 hold_26_3 - 335.6643356643356 hold_26_4
   - 269.6629213483146 hold_26_5 - 225.3521126760563 hold_26_6
   + flow_nic_in_26_26 <= 0
 constr_intergroup_throughput_27_nic_out_30_nic_in_27: switch_nic_out_30_nic_in_27 = 1 ->
   - 1263.157894736842 hold_27_1 - 657.5342465753424 hold_27_2
   - 444.4444444444444 hold_27_3 - 335.6643356643356 hold_27_4
   - 269.6629213483146 hold_27_5 - 225.3521126760563 hold_27_6
   + flow_nic_in_27_27 <= 0
 constr_intergroup_throughput_28_nic_out_30_nic_in_28: switch_nic_out_30_nic_in_28 = 1 ->
   - 1263.157894736842 hold_28_1 - 657.5342465753424 hold_28_2
   - 444.4444444444444 hold_28_3 - 335.6643356643356 hold_28_4
   - 269.6629213483146 hold_28_5 - 225.3521126760563 hold_28_6
   + flow_nic_in_28_28 <= 0
 constr_intergroup_throughput_29_nic_out_30_nic_in_29: switch_nic_out_30_nic_in_29 = 1 ->
   - 1263.157894736842 hold_29_1 - 657.5342465753424 hold_29_2
   - 444.4444444444444 hold_29_3 - 335.6643356643356 hold_29_4
   - 269.6629213483146 hold_29_5 - 225.3521126760563 hold_29_6
   + flow_nic_in_29_29 <= 0
 constr_intergroup_throughput_31_nic_out_30_nic_in_31: switch_nic_out_30_nic_in_31 = 1 ->
   - 1263.157894736842 hold_31_1 - 657.5342465753424 hold_31_2
   - 444.4444444444444 hold_31_3 - 335.6643356643356 hold_31_4
   - 269.6629213483146 hold_31_5 - 225.3521126760563 hold_31_6
   + flow_nic_in_31_31 <= 0
 constr_intergroup_throughput_31_source_nic_in_31: switch_source_nic_in_31 = 1 ->
   - 1348.314606741573 hold_31_1 - 679.8866855524079 hold_31_2
   - 454.5454545454545 hold_31_3 - 341.3940256045519 hold_31_4
   - 273.3485193621868 hold_31_5 - 227.9202279202279 hold_31_6
   + flow_nic_in_31_31 <= 0
 constr_intergroup_throughput_0_nic_out_31_nic_in_0: switch_nic_out_31_nic_in_0 = 1 ->
   - 1263.157894736842 hold_0_1 - 657.5342465753424 hold_0_2
   - 444.4444444444444 hold_0_3 - 335.6643356643356 hold_0_4
   - 269.6629213483146 hold_0_5 - 225.3521126760563 hold_0_6
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_31_nic_in_1: switch_nic_out_31_nic_in_1 = 1 ->
   - 1263.157894736842 hold_1_1 - 657.5342465753424 hold_1_2
   - 444.4444444444444 hold_1_3 - 335.6643356643356 hold_1_4
   - 269.6629213483146 hold_1_5 - 225.3521126760563 hold_1_6
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_31_nic_in_2: switch_nic_out_31_nic_in_2 = 1 ->
   - 1263.157894736842 hold_2_1 - 657.5342465753424 hold_2_2
   - 444.4444444444444 hold_2_3 - 335.6643356643356 hold_2_4
   - 269.6629213483146 hold_2_5 - 225.3521126760563 hold_2_6
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_31_nic_in_3: switch_nic_out_31_nic_in_3 = 1 ->
   - 1263.157894736842 hold_3_1 - 657.5342465753424 hold_3_2
   - 444.4444444444444 hold_3_3 - 335.6643356643356 hold_3_4
   - 269.6629213483146 hold_3_5 - 225.3521126760563 hold_3_6
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_31_nic_in_4: switch_nic_out_31_nic_in_4 = 1 ->
   - 1263.157894736842 hold_4_1 - 657.5342465753424 hold_4_2
   - 444.4444444444444 hold_4_3 - 335.6643356643356 hold_4_4
   - 269.6629213483146 hold_4_5 - 225.3521126760563 hold_4_6
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_31_nic_in_5: switch_nic_out_31_nic_in_5 = 1 ->
   - 1263.157894736842 hold_5_1 - 657.5342465753424 hold_5_2
   - 444.4444444444444 hold_5_3 - 335.6643356643356 hold_5_4
   - 269.6629213483146 hold_5_5 - 225.3521126760563 hold_5_6
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_31_nic_in_6: switch_nic_out_31_nic_in_6 = 1 ->
   - 1263.157894736842 hold_6_1 - 657.5342465753424 hold_6_2
   - 444.4444444444444 hold_6_3 - 335.6643356643356 hold_6_4
   - 269.6629213483146 hold_6_5 - 225.3521126760563 hold_6_6
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_31_nic_in_7: switch_nic_out_31_nic_in_7 = 1 ->
   - 1263.157894736842 hold_7_1 - 657.5342465753424 hold_7_2
   - 444.4444444444444 hold_7_3 - 335.6643356643356 hold_7_4
   - 269.6629213483146 hold_7_5 - 225.3521126760563 hold_7_6
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_31_nic_in_8: switch_nic_out_31_nic_in_8 = 1 ->
   - 1263.157894736842 hold_8_1 - 657.5342465753424 hold_8_2
   - 444.4444444444444 hold_8_3 - 335.6643356643356 hold_8_4
   - 269.6629213483146 hold_8_5 - 225.3521126760563 hold_8_6
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_31_nic_in_9: switch_nic_out_31_nic_in_9 = 1 ->
   - 1263.157894736842 hold_9_1 - 657.5342465753424 hold_9_2
   - 444.4444444444444 hold_9_3 - 335.6643356643356 hold_9_4
   - 269.6629213483146 hold_9_5 - 225.3521126760563 hold_9_6
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_31_nic_in_10: switch_nic_out_31_nic_in_10 = 1 ->
   - 1263.157894736842 hold_10_1 - 657.5342465753424 hold_10_2
   - 444.4444444444444 hold_10_3 - 335.6643356643356 hold_10_4
   - 269.6629213483146 hold_10_5 - 225.3521126760563 hold_10_6
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_31_nic_in_11: switch_nic_out_31_nic_in_11 = 1 ->
   - 1263.157894736842 hold_11_1 - 657.5342465753424 hold_11_2
   - 444.4444444444444 hold_11_3 - 335.6643356643356 hold_11_4
   - 269.6629213483146 hold_11_5 - 225.3521126760563 hold_11_6
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_31_nic_in_12: switch_nic_out_31_nic_in_12 = 1 ->
   - 1263.157894736842 hold_12_1 - 657.5342465753424 hold_12_2
   - 444.4444444444444 hold_12_3 - 335.6643356643356 hold_12_4
   - 269.6629213483146 hold_12_5 - 225.3521126760563 hold_12_6
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_31_nic_in_13: switch_nic_out_31_nic_in_13 = 1 ->
   - 1263.157894736842 hold_13_1 - 657.5342465753424 hold_13_2
   - 444.4444444444444 hold_13_3 - 335.6643356643356 hold_13_4
   - 269.6629213483146 hold_13_5 - 225.3521126760563 hold_13_6
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_31_nic_in_14: switch_nic_out_31_nic_in_14 = 1 ->
   - 1263.157894736842 hold_14_1 - 657.5342465753424 hold_14_2
   - 444.4444444444444 hold_14_3 - 335.6643356643356 hold_14_4
   - 269.6629213483146 hold_14_5 - 225.3521126760563 hold_14_6
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_31_nic_in_15: switch_nic_out_31_nic_in_15 = 1 ->
   - 1263.157894736842 hold_15_1 - 657.5342465753424 hold_15_2
   - 444.4444444444444 hold_15_3 - 335.6643356643356 hold_15_4
   - 269.6629213483146 hold_15_5 - 225.3521126760563 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_16_nic_out_31_nic_in_16: switch_nic_out_31_nic_in_16 = 1 ->
   - 1263.157894736842 hold_16_1 - 657.5342465753424 hold_16_2
   - 444.4444444444444 hold_16_3 - 335.6643356643356 hold_16_4
   - 269.6629213483146 hold_16_5 - 225.3521126760563 hold_16_6
   + flow_nic_in_16_16 <= 0
 constr_intergroup_throughput_17_nic_out_31_nic_in_17: switch_nic_out_31_nic_in_17 = 1 ->
   - 1263.157894736842 hold_17_1 - 657.5342465753424 hold_17_2
   - 444.4444444444444 hold_17_3 - 335.6643356643356 hold_17_4
   - 269.6629213483146 hold_17_5 - 225.3521126760563 hold_17_6
   + flow_nic_in_17_17 <= 0
 constr_intergroup_throughput_18_nic_out_31_nic_in_18: switch_nic_out_31_nic_in_18 = 1 ->
   - 1263.157894736842 hold_18_1 - 657.5342465753424 hold_18_2
   - 444.4444444444444 hold_18_3 - 335.6643356643356 hold_18_4
   - 269.6629213483146 hold_18_5 - 225.3521126760563 hold_18_6
   + flow_nic_in_18_18 <= 0
 constr_intergroup_throughput_19_nic_out_31_nic_in_19: switch_nic_out_31_nic_in_19 = 1 ->
   - 1263.157894736842 hold_19_1 - 657.5342465753424 hold_19_2
   - 444.4444444444444 hold_19_3 - 335.6643356643356 hold_19_4
   - 269.6629213483146 hold_19_5 - 225.3521126760563 hold_19_6
   + flow_nic_in_19_19 <= 0
 constr_intergroup_throughput_20_nic_out_31_nic_in_20: switch_nic_out_31_nic_in_20 = 1 ->
   - 1263.157894736842 hold_20_1 - 657.5342465753424 hold_20_2
   - 444.4444444444444 hold_20_3 - 335.6643356643356 hold_20_4
   - 269.6629213483146 hold_20_5 - 225.3521126760563 hold_20_6
   + flow_nic_in_20_20 <= 0
 constr_intergroup_throughput_21_nic_out_31_nic_in_21: switch_nic_out_31_nic_in_21 = 1 ->
   - 1263.157894736842 hold_21_1 - 657.5342465753424 hold_21_2
   - 444.4444444444444 hold_21_3 - 335.6643356643356 hold_21_4
   - 269.6629213483146 hold_21_5 - 225.3521126760563 hold_21_6
   + flow_nic_in_21_21 <= 0
 constr_intergroup_throughput_22_nic_out_31_nic_in_22: switch_nic_out_31_nic_in_22 = 1 ->
   - 1263.157894736842 hold_22_1 - 657.5342465753424 hold_22_2
   - 444.4444444444444 hold_22_3 - 335.6643356643356 hold_22_4
   - 269.6629213483146 hold_22_5 - 225.3521126760563 hold_22_6
   + flow_nic_in_22_22 <= 0
 constr_intergroup_throughput_23_nic_out_31_nic_in_23: switch_nic_out_31_nic_in_23 = 1 ->
   - 1263.157894736842 hold_23_1 - 657.5342465753424 hold_23_2
   - 444.4444444444444 hold_23_3 - 335.6643356643356 hold_23_4
   - 269.6629213483146 hold_23_5 - 225.3521126760563 hold_23_6
   + flow_nic_in_23_23 <= 0
 constr_intergroup_throughput_24_nic_out_31_nic_in_24: switch_nic_out_31_nic_in_24 = 1 ->
   - 1263.157894736842 hold_24_1 - 657.5342465753424 hold_24_2
   - 444.4444444444444 hold_24_3 - 335.6643356643356 hold_24_4
   - 269.6629213483146 hold_24_5 - 225.3521126760563 hold_24_6
   + flow_nic_in_24_24 <= 0
 constr_intergroup_throughput_25_nic_out_31_nic_in_25: switch_nic_out_31_nic_in_25 = 1 ->
   - 1263.157894736842 hold_25_1 - 657.5342465753424 hold_25_2
   - 444.4444444444444 hold_25_3 - 335.6643356643356 hold_25_4
   - 269.6629213483146 hold_25_5 - 225.3521126760563 hold_25_6
   + flow_nic_in_25_25 <= 0
 constr_intergroup_throughput_26_nic_out_31_nic_in_26: switch_nic_out_31_nic_in_26 = 1 ->
   - 1263.157894736842 hold_26_1 - 657.5342465753424 hold_26_2
   - 444.4444444444444 hold_26_3 - 335.6643356643356 hold_26_4
   - 269.6629213483146 hold_26_5 - 225.3521126760563 hold_26_6
   + flow_nic_in_26_26 <= 0
 constr_intergroup_throughput_27_nic_out_31_nic_in_27: switch_nic_out_31_nic_in_27 = 1 ->
   - 1263.157894736842 hold_27_1 - 657.5342465753424 hold_27_2
   - 444.4444444444444 hold_27_3 - 335.6643356643356 hold_27_4
   - 269.6629213483146 hold_27_5 - 225.3521126760563 hold_27_6
   + flow_nic_in_27_27 <= 0
 constr_intergroup_throughput_28_nic_out_31_nic_in_28: switch_nic_out_31_nic_in_28 = 1 ->
   - 1263.157894736842 hold_28_1 - 657.5342465753424 hold_28_2
   - 444.4444444444444 hold_28_3 - 335.6643356643356 hold_28_4
   - 269.6629213483146 hold_28_5 - 225.3521126760563 hold_28_6
   + flow_nic_in_28_28 <= 0
 constr_intergroup_throughput_29_nic_out_31_nic_in_29: switch_nic_out_31_nic_in_29 = 1 ->
   - 1263.157894736842 hold_29_1 - 657.5342465753424 hold_29_2
   - 444.4444444444444 hold_29_3 - 335.6643356643356 hold_29_4
   - 269.6629213483146 hold_29_5 - 225.3521126760563 hold_29_6
   + flow_nic_in_29_29 <= 0
 constr_intergroup_throughput_30_nic_out_31_nic_in_30: switch_nic_out_31_nic_in_30 = 1 ->
   - 1263.157894736842 hold_30_1 - 657.5342465753424 hold_30_2
   - 444.4444444444444 hold_30_3 - 335.6643356643356 hold_30_4
   - 269.6629213483146 hold_30_5 - 225.3521126760563 hold_30_6
   + flow_nic_in_30_30 <= 0
 inactive_end_not_complete_0: is_active_0 = 0 -> is_end_complete_0 = 0
 active_end_val_0: is_active_0 = 1 -> - start_0 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 + val_end_0 = 0
 inactive_end_zero_0: is_active_0 = 0 -> val_end_0 = 0
 inactive_end_not_complete_1: is_active_1 = 0 -> is_end_complete_1 = 0
 active_end_val_1: is_active_1 = 1 -> - start_1 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 + val_end_1 = 0
 inactive_end_zero_1: is_active_1 = 0 -> val_end_1 = 0
 inactive_end_not_complete_2: is_active_2 = 0 -> is_end_complete_2 = 0
 active_end_val_2: is_active_2 = 1 -> - start_2 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 + val_end_2 = 0
 inactive_end_zero_2: is_active_2 = 0 -> val_end_2 = 0
 inactive_end_not_complete_3: is_active_3 = 0 -> is_end_complete_3 = 0
 active_end_val_3: is_active_3 = 1 -> - start_3 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 + val_end_3 = 0
 inactive_end_zero_3: is_active_3 = 0 -> val_end_3 = 0
 inactive_end_not_complete_4: is_active_4 = 0 -> is_end_complete_4 = 0
 active_end_val_4: is_active_4 = 1 -> - start_4 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 + val_end_4 = 0
 inactive_end_zero_4: is_active_4 = 0 -> val_end_4 = 0
 inactive_end_not_complete_5: is_active_5 = 0 -> is_end_complete_5 = 0
 active_end_val_5: is_active_5 = 1 -> - start_5 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 + val_end_5 = 0
 inactive_end_zero_5: is_active_5 = 0 -> val_end_5 = 0
 inactive_end_not_complete_6: is_active_6 = 0 -> is_end_complete_6 = 0
 active_end_val_6: is_active_6 = 1 -> - start_6 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 + val_end_6 = 0
 inactive_end_zero_6: is_active_6 = 0 -> val_end_6 = 0
 inactive_end_not_complete_7: is_active_7 = 0 -> is_end_complete_7 = 0
 active_end_val_7: is_active_7 = 1 -> - start_7 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 + val_end_7 = 0
 inactive_end_zero_7: is_active_7 = 0 -> val_end_7 = 0
 inactive_end_not_complete_8: is_active_8 = 0 -> is_end_complete_8 = 0
 active_end_val_8: is_active_8 = 1 -> - start_8 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 + val_end_8 = 0
 inactive_end_zero_8: is_active_8 = 0 -> val_end_8 = 0
 inactive_end_not_complete_9: is_active_9 = 0 -> is_end_complete_9 = 0
 active_end_val_9: is_active_9 = 1 -> - start_9 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 + val_end_9 = 0
 inactive_end_zero_9: is_active_9 = 0 -> val_end_9 = 0
 inactive_end_not_complete_10: is_active_10 = 0 -> is_end_complete_10 = 0
 active_end_val_10: is_active_10 = 1 -> - start_10 - hold_10_1
   - 2 hold_10_2 - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6
   + val_end_10 = 0
 inactive_end_zero_10: is_active_10 = 0 -> val_end_10 = 0
 inactive_end_not_complete_11: is_active_11 = 0 -> is_end_complete_11 = 0
 active_end_val_11: is_active_11 = 1 -> - start_11 - hold_11_1
   - 2 hold_11_2 - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6
   + val_end_11 = 0
 inactive_end_zero_11: is_active_11 = 0 -> val_end_11 = 0
 inactive_end_not_complete_12: is_active_12 = 0 -> is_end_complete_12 = 0
 active_end_val_12: is_active_12 = 1 -> - start_12 - hold_12_1
   - 2 hold_12_2 - 3 hold_12_3 - 4 hold_12_4 - 5 hold_12_5 - 6 hold_12_6
   + val_end_12 = 0
 inactive_end_zero_12: is_active_12 = 0 -> val_end_12 = 0
 inactive_end_not_complete_13: is_active_13 = 0 -> is_end_complete_13 = 0
 active_end_val_13: is_active_13 = 1 -> - start_13 - hold_13_1
   - 2 hold_13_2 - 3 hold_13_3 - 4 hold_13_4 - 5 hold_13_5 - 6 hold_13_6
   + val_end_13 = 0
 inactive_end_zero_13: is_active_13 = 0 -> val_end_13 = 0
 inactive_end_not_complete_14: is_active_14 = 0 -> is_end_complete_14 = 0
 active_end_val_14: is_active_14 = 1 -> - start_14 - hold_14_1
   - 2 hold_14_2 - 3 hold_14_3 - 4 hold_14_4 - 5 hold_14_5 - 6 hold_14_6
   + val_end_14 = 0
 inactive_end_zero_14: is_active_14 = 0 -> val_end_14 = 0
 inactive_end_not_complete_15: is_active_15 = 0 -> is_end_complete_15 = 0
 active_end_val_15: is_active_15 = 1 -> - start_15 - hold_15_1
   - 2 hold_15_2 - 3 hold_15_3 - 4 hold_15_4 - 5 hold_15_5 - 6 hold_15_6
   + val_end_15 = 0
 inactive_end_zero_15: is_active_15 = 0 -> val_end_15 = 0
 inactive_end_not_complete_16: is_active_16 = 0 -> is_end_complete_16 = 0
 active_end_val_16: is_active_16 = 1 -> - start_16 - hold_16_1
   - 2 hold_16_2 - 3 hold_16_3 - 4 hold_16_4 - 5 hold_16_5 - 6 hold_16_6
   + val_end_16 = 0
 inactive_end_zero_16: is_active_16 = 0 -> val_end_16 = 0
 inactive_end_not_complete_17: is_active_17 = 0 -> is_end_complete_17 = 0
 active_end_val_17: is_active_17 = 1 -> - start_17 - hold_17_1
   - 2 hold_17_2 - 3 hold_17_3 - 4 hold_17_4 - 5 hold_17_5 - 6 hold_17_6
   + val_end_17 = 0
 inactive_end_zero_17: is_active_17 = 0 -> val_end_17 = 0
 inactive_end_not_complete_18: is_active_18 = 0 -> is_end_complete_18 = 0
 active_end_val_18: is_active_18 = 1 -> - start_18 - hold_18_1
   - 2 hold_18_2 - 3 hold_18_3 - 4 hold_18_4 - 5 hold_18_5 - 6 hold_18_6
   + val_end_18 = 0
 inactive_end_zero_18: is_active_18 = 0 -> val_end_18 = 0
 inactive_end_not_complete_19: is_active_19 = 0 -> is_end_complete_19 = 0
 active_end_val_19: is_active_19 = 1 -> - start_19 - hold_19_1
   - 2 hold_19_2 - 3 hold_19_3 - 4 hold_19_4 - 5 hold_19_5 - 6 hold_19_6
   + val_end_19 = 0
 inactive_end_zero_19: is_active_19 = 0 -> val_end_19 = 0
 inactive_end_not_complete_20: is_active_20 = 0 -> is_end_complete_20 = 0
 active_end_val_20: is_active_20 = 1 -> - start_20 - hold_20_1
   - 2 hold_20_2 - 3 hold_20_3 - 4 hold_20_4 - 5 hold_20_5 - 6 hold_20_6
   + val_end_20 = 0
 inactive_end_zero_20: is_active_20 = 0 -> val_end_20 = 0
 inactive_end_not_complete_21: is_active_21 = 0 -> is_end_complete_21 = 0
 active_end_val_21: is_active_21 = 1 -> - start_21 - hold_21_1
   - 2 hold_21_2 - 3 hold_21_3 - 4 hold_21_4 - 5 hold_21_5 - 6 hold_21_6
   + val_end_21 = 0
 inactive_end_zero_21: is_active_21 = 0 -> val_end_21 = 0
 inactive_end_not_complete_22: is_active_22 = 0 -> is_end_complete_22 = 0
 active_end_val_22: is_active_22 = 1 -> - start_22 - hold_22_1
   - 2 hold_22_2 - 3 hold_22_3 - 4 hold_22_4 - 5 hold_22_5 - 6 hold_22_6
   + val_end_22 = 0
 inactive_end_zero_22: is_active_22 = 0 -> val_end_22 = 0
 inactive_end_not_complete_23: is_active_23 = 0 -> is_end_complete_23 = 0
 active_end_val_23: is_active_23 = 1 -> - start_23 - hold_23_1
   - 2 hold_23_2 - 3 hold_23_3 - 4 hold_23_4 - 5 hold_23_5 - 6 hold_23_6
   + val_end_23 = 0
 inactive_end_zero_23: is_active_23 = 0 -> val_end_23 = 0
 inactive_end_not_complete_24: is_active_24 = 0 -> is_end_complete_24 = 0
 active_end_val_24: is_active_24 = 1 -> - start_24 - hold_24_1
   - 2 hold_24_2 - 3 hold_24_3 - 4 hold_24_4 - 5 hold_24_5 - 6 hold_24_6
   + val_end_24 = 0
 inactive_end_zero_24: is_active_24 = 0 -> val_end_24 = 0
 inactive_end_not_complete_25: is_active_25 = 0 -> is_end_complete_25 = 0
 active_end_val_25: is_active_25 = 1 -> - start_25 - hold_25_1
   - 2 hold_25_2 - 3 hold_25_3 - 4 hold_25_4 - 5 hold_25_5 - 6 hold_25_6
   + val_end_25 = 0
 inactive_end_zero_25: is_active_25 = 0 -> val_end_25 = 0
 inactive_end_not_complete_26: is_active_26 = 0 -> is_end_complete_26 = 0
 active_end_val_26: is_active_26 = 1 -> - start_26 - hold_26_1
   - 2 hold_26_2 - 3 hold_26_3 - 4 hold_26_4 - 5 hold_26_5 - 6 hold_26_6
   + val_end_26 = 0
 inactive_end_zero_26: is_active_26 = 0 -> val_end_26 = 0
 inactive_end_not_complete_27: is_active_27 = 0 -> is_end_complete_27 = 0
 active_end_val_27: is_active_27 = 1 -> - start_27 - hold_27_1
   - 2 hold_27_2 - 3 hold_27_3 - 4 hold_27_4 - 5 hold_27_5 - 6 hold_27_6
   + val_end_27 = 0
 inactive_end_zero_27: is_active_27 = 0 -> val_end_27 = 0
 inactive_end_not_complete_28: is_active_28 = 0 -> is_end_complete_28 = 0
 active_end_val_28: is_active_28 = 1 -> - start_28 - hold_28_1
   - 2 hold_28_2 - 3 hold_28_3 - 4 hold_28_4 - 5 hold_28_5 - 6 hold_28_6
   + val_end_28 = 0
 inactive_end_zero_28: is_active_28 = 0 -> val_end_28 = 0
 inactive_end_not_complete_29: is_active_29 = 0 -> is_end_complete_29 = 0
 active_end_val_29: is_active_29 = 1 -> - start_29 - hold_29_1
   - 2 hold_29_2 - 3 hold_29_3 - 4 hold_29_4 - 5 hold_29_5 - 6 hold_29_6
   + val_end_29 = 0
 inactive_end_zero_29: is_active_29 = 0 -> val_end_29 = 0
 inactive_end_not_complete_30: is_active_30 = 0 -> is_end_complete_30 = 0
 active_end_val_30: is_active_30 = 1 -> - start_30 - hold_30_1
   - 2 hold_30_2 - 3 hold_30_3 - 4 hold_30_4 - 5 hold_30_5 - 6 hold_30_6
   + val_end_30 = 0
 inactive_end_zero_30: is_active_30 = 0 -> val_end_30 = 0
 inactive_end_not_complete_31: is_active_31 = 0 -> is_end_complete_31 = 0
 active_end_val_31: is_active_31 = 1 -> - start_31 - hold_31_1
   - 2 hold_31_2 - 3 hold_31_3 - 4 hold_31_4 - 5 hold_31_5 - 6 hold_31_6
   + val_end_31 = 0
 inactive_end_zero_31: is_active_31 = 0 -> val_end_31 = 0
 group_throughput_constr_0: group_tp_active_0 = 1 ->
   - 1371.428571428571 hold_0_1 - 685.7142857142857 hold_0_2
   - 457.1428571428571 hold_0_3 - 342.8571428571428 hold_0_4
   - 274.2857142857143 hold_0_5 - 228.5714285714286 hold_0_6
   + flow_nic_in_0_0 <= 0
 group_throughput_constr_1: group_tp_active_1 = 1 ->
   - 1371.428571428571 hold_1_1 - 685.7142857142857 hold_1_2
   - 457.1428571428571 hold_1_3 - 342.8571428571428 hold_1_4
   - 274.2857142857143 hold_1_5 - 228.5714285714286 hold_1_6
   + flow_nic_in_1_1 <= 0
 group_throughput_constr_2: group_tp_active_2 = 1 ->
   - 1371.428571428571 hold_2_1 - 685.7142857142857 hold_2_2
   - 457.1428571428571 hold_2_3 - 342.8571428571428 hold_2_4
   - 274.2857142857143 hold_2_5 - 228.5714285714286 hold_2_6
   + flow_nic_in_2_2 <= 0
 group_throughput_constr_3: group_tp_active_3 = 1 ->
   - 1371.428571428571 hold_3_1 - 685.7142857142857 hold_3_2
   - 457.1428571428571 hold_3_3 - 342.8571428571428 hold_3_4
   - 274.2857142857143 hold_3_5 - 228.5714285714286 hold_3_6
   + flow_nic_in_3_3 <= 0
 group_throughput_constr_4: group_tp_active_4 = 1 ->
   - 1371.428571428571 hold_4_1 - 685.7142857142857 hold_4_2
   - 457.1428571428571 hold_4_3 - 342.8571428571428 hold_4_4
   - 274.2857142857143 hold_4_5 - 228.5714285714286 hold_4_6
   + flow_nic_in_4_4 <= 0
 group_throughput_constr_5: group_tp_active_5 = 1 ->
   - 1371.428571428571 hold_5_1 - 685.7142857142857 hold_5_2
   - 457.1428571428571 hold_5_3 - 342.8571428571428 hold_5_4
   - 274.2857142857143 hold_5_5 - 228.5714285714286 hold_5_6
   + flow_nic_in_5_5 <= 0
 group_throughput_constr_6: group_tp_active_6 = 1 ->
   - 1371.428571428571 hold_6_1 - 685.7142857142857 hold_6_2
   - 457.1428571428571 hold_6_3 - 342.8571428571428 hold_6_4
   - 274.2857142857143 hold_6_5 - 228.5714285714286 hold_6_6
   + flow_nic_in_6_6 <= 0
 group_throughput_constr_7: group_tp_active_7 = 1 ->
   - 1371.428571428571 hold_7_1 - 685.7142857142857 hold_7_2
   - 457.1428571428571 hold_7_3 - 342.8571428571428 hold_7_4
   - 274.2857142857143 hold_7_5 - 228.5714285714286 hold_7_6
   + flow_nic_in_7_7 <= 0
 group_throughput_constr_8: group_tp_active_8 = 1 ->
   - 1371.428571428571 hold_8_1 - 685.7142857142857 hold_8_2
   - 457.1428571428571 hold_8_3 - 342.8571428571428 hold_8_4
   - 274.2857142857143 hold_8_5 - 228.5714285714286 hold_8_6
   + flow_nic_in_8_8 <= 0
 group_throughput_constr_9: group_tp_active_9 = 1 ->
   - 1371.428571428571 hold_9_1 - 685.7142857142857 hold_9_2
   - 457.1428571428571 hold_9_3 - 342.8571428571428 hold_9_4
   - 274.2857142857143 hold_9_5 - 228.5714285714286 hold_9_6
   + flow_nic_in_9_9 <= 0
 group_throughput_constr_10: group_tp_active_10 = 1 ->
   - 1371.428571428571 hold_10_1 - 685.7142857142857 hold_10_2
   - 457.1428571428571 hold_10_3 - 342.8571428571428 hold_10_4
   - 274.2857142857143 hold_10_5 - 228.5714285714286 hold_10_6
   + flow_nic_in_10_10 <= 0
 group_throughput_constr_11: group_tp_active_11 = 1 ->
   - 1371.428571428571 hold_11_1 - 685.7142857142857 hold_11_2
   - 457.1428571428571 hold_11_3 - 342.8571428571428 hold_11_4
   - 274.2857142857143 hold_11_5 - 228.5714285714286 hold_11_6
   + flow_nic_in_11_11 <= 0
 group_throughput_constr_12: group_tp_active_12 = 1 ->
   - 1371.428571428571 hold_12_1 - 685.7142857142857 hold_12_2
   - 457.1428571428571 hold_12_3 - 342.8571428571428 hold_12_4
   - 274.2857142857143 hold_12_5 - 228.5714285714286 hold_12_6
   + flow_nic_in_12_12 <= 0
 group_throughput_constr_13: group_tp_active_13 = 1 ->
   - 1371.428571428571 hold_13_1 - 685.7142857142857 hold_13_2
   - 457.1428571428571 hold_13_3 - 342.8571428571428 hold_13_4
   - 274.2857142857143 hold_13_5 - 228.5714285714286 hold_13_6
   + flow_nic_in_13_13 <= 0
 group_throughput_constr_14: group_tp_active_14 = 1 ->
   - 1371.428571428571 hold_14_1 - 685.7142857142857 hold_14_2
   - 457.1428571428571 hold_14_3 - 342.8571428571428 hold_14_4
   - 274.2857142857143 hold_14_5 - 228.5714285714286 hold_14_6
   + flow_nic_in_14_14 <= 0
 group_throughput_constr_15: group_tp_active_15 = 1 ->
   - 1371.428571428571 hold_15_1 - 685.7142857142857 hold_15_2
   - 457.1428571428571 hold_15_3 - 342.8571428571428 hold_15_4
   - 274.2857142857143 hold_15_5 - 228.5714285714286 hold_15_6
   + flow_nic_in_15_15 <= 0
 group_throughput_constr_16: group_tp_active_16 = 1 ->
   - 1371.428571428571 hold_16_1 - 685.7142857142857 hold_16_2
   - 457.1428571428571 hold_16_3 - 342.8571428571428 hold_16_4
   - 274.2857142857143 hold_16_5 - 228.5714285714286 hold_16_6
   + flow_nic_in_16_16 <= 0
 group_throughput_constr_17: group_tp_active_17 = 1 ->
   - 1371.428571428571 hold_17_1 - 685.7142857142857 hold_17_2
   - 457.1428571428571 hold_17_3 - 342.8571428571428 hold_17_4
   - 274.2857142857143 hold_17_5 - 228.5714285714286 hold_17_6
   + flow_nic_in_17_17 <= 0
 group_throughput_constr_18: group_tp_active_18 = 1 ->
   - 1371.428571428571 hold_18_1 - 685.7142857142857 hold_18_2
   - 457.1428571428571 hold_18_3 - 342.8571428571428 hold_18_4
   - 274.2857142857143 hold_18_5 - 228.5714285714286 hold_18_6
   + flow_nic_in_18_18 <= 0
 group_throughput_constr_19: group_tp_active_19 = 1 ->
   - 1371.428571428571 hold_19_1 - 685.7142857142857 hold_19_2
   - 457.1428571428571 hold_19_3 - 342.8571428571428 hold_19_4
   - 274.2857142857143 hold_19_5 - 228.5714285714286 hold_19_6
   + flow_nic_in_19_19 <= 0
 group_throughput_constr_20: group_tp_active_20 = 1 ->
   - 1371.428571428571 hold_20_1 - 685.7142857142857 hold_20_2
   - 457.1428571428571 hold_20_3 - 342.8571428571428 hold_20_4
   - 274.2857142857143 hold_20_5 - 228.5714285714286 hold_20_6
   + flow_nic_in_20_20 <= 0
 group_throughput_constr_21: group_tp_active_21 = 1 ->
   - 1371.428571428571 hold_21_1 - 685.7142857142857 hold_21_2
   - 457.1428571428571 hold_21_3 - 342.8571428571428 hold_21_4
   - 274.2857142857143 hold_21_5 - 228.5714285714286 hold_21_6
   + flow_nic_in_21_21 <= 0
 group_throughput_constr_22: group_tp_active_22 = 1 ->
   - 1371.428571428571 hold_22_1 - 685.7142857142857 hold_22_2
   - 457.1428571428571 hold_22_3 - 342.8571428571428 hold_22_4
   - 274.2857142857143 hold_22_5 - 228.5714285714286 hold_22_6
   + flow_nic_in_22_22 <= 0
 group_throughput_constr_23: group_tp_active_23 = 1 ->
   - 1371.428571428571 hold_23_1 - 685.7142857142857 hold_23_2
   - 457.1428571428571 hold_23_3 - 342.8571428571428 hold_23_4
   - 274.2857142857143 hold_23_5 - 228.5714285714286 hold_23_6
   + flow_nic_in_23_23 <= 0
 group_throughput_constr_24: group_tp_active_24 = 1 ->
   - 1371.428571428571 hold_24_1 - 685.7142857142857 hold_24_2
   - 457.1428571428571 hold_24_3 - 342.8571428571428 hold_24_4
   - 274.2857142857143 hold_24_5 - 228.5714285714286 hold_24_6
   + flow_nic_in_24_24 <= 0
 group_throughput_constr_25: group_tp_active_25 = 1 ->
   - 1371.428571428571 hold_25_1 - 685.7142857142857 hold_25_2
   - 457.1428571428571 hold_25_3 - 342.8571428571428 hold_25_4
   - 274.2857142857143 hold_25_5 - 228.5714285714286 hold_25_6
   + flow_nic_in_25_25 <= 0
 group_throughput_constr_26: group_tp_active_26 = 1 ->
   - 1371.428571428571 hold_26_1 - 685.7142857142857 hold_26_2
   - 457.1428571428571 hold_26_3 - 342.8571428571428 hold_26_4
   - 274.2857142857143 hold_26_5 - 228.5714285714286 hold_26_6
   + flow_nic_in_26_26 <= 0
 group_throughput_constr_27: group_tp_active_27 = 1 ->
   - 1371.428571428571 hold_27_1 - 685.7142857142857 hold_27_2
   - 457.1428571428571 hold_27_3 - 342.8571428571428 hold_27_4
   - 274.2857142857143 hold_27_5 - 228.5714285714286 hold_27_6
   + flow_nic_in_27_27 <= 0
 group_throughput_constr_28: group_tp_active_28 = 1 ->
   - 1371.428571428571 hold_28_1 - 685.7142857142857 hold_28_2
   - 457.1428571428571 hold_28_3 - 342.8571428571428 hold_28_4
   - 274.2857142857143 hold_28_5 - 228.5714285714286 hold_28_6
   + flow_nic_in_28_28 <= 0
 group_throughput_constr_29: group_tp_active_29 = 1 ->
   - 1371.428571428571 hold_29_1 - 685.7142857142857 hold_29_2
   - 457.1428571428571 hold_29_3 - 342.8571428571428 hold_29_4
   - 274.2857142857143 hold_29_5 - 228.5714285714286 hold_29_6
   + flow_nic_in_29_29 <= 0
 group_throughput_constr_30: group_tp_active_30 = 1 ->
   - 1371.428571428571 hold_30_1 - 685.7142857142857 hold_30_2
   - 457.1428571428571 hold_30_3 - 342.8571428571428 hold_30_4
   - 274.2857142857143 hold_30_5 - 228.5714285714286 hold_30_6
   + flow_nic_in_30_30 <= 0
 group_throughput_constr_31: group_tp_active_31 = 1 ->
   - 1371.428571428571 hold_31_1 - 685.7142857142857 hold_31_2
   - 457.1428571428571 hold_31_3 - 342.8571428571428 hold_31_4
   - 274.2857142857143 hold_31_5 - 228.5714285714286 hold_31_6
   + flow_nic_in_31_31 <= 0
Bounds
Binaries
 hold_0_1 hold_0_2 hold_0_3 hold_0_4 hold_0_5 hold_0_6 hold_1_1 hold_1_2
 hold_1_3 hold_1_4 hold_1_5 hold_1_6 hold_2_1 hold_2_2 hold_2_3 hold_2_4
 hold_2_5 hold_2_6 hold_3_1 hold_3_2 hold_3_3 hold_3_4 hold_3_5 hold_3_6
 hold_4_1 hold_4_2 hold_4_3 hold_4_4 hold_4_5 hold_4_6 hold_5_1 hold_5_2
 hold_5_3 hold_5_4 hold_5_5 hold_5_6 hold_6_1 hold_6_2 hold_6_3 hold_6_4
 hold_6_5 hold_6_6 hold_7_1 hold_7_2 hold_7_3 hold_7_4 hold_7_5 hold_7_6
 hold_8_1 hold_8_2 hold_8_3 hold_8_4 hold_8_5 hold_8_6 hold_9_1 hold_9_2
 hold_9_3 hold_9_4 hold_9_5 hold_9_6 hold_10_1 hold_10_2 hold_10_3
 hold_10_4 hold_10_5 hold_10_6 hold_11_1 hold_11_2 hold_11_3 hold_11_4
 hold_11_5 hold_11_6 hold_12_1 hold_12_2 hold_12_3 hold_12_4 hold_12_5
 hold_12_6 hold_13_1 hold_13_2 hold_13_3 hold_13_4 hold_13_5 hold_13_6
 hold_14_1 hold_14_2 hold_14_3 hold_14_4 hold_14_5 hold_14_6 hold_15_1
 hold_15_2 hold_15_3 hold_15_4 hold_15_5 hold_15_6 hold_16_1 hold_16_2
 hold_16_3 hold_16_4 hold_16_5 hold_16_6 hold_17_1 hold_17_2 hold_17_3
 hold_17_4 hold_17_5 hold_17_6 hold_18_1 hold_18_2 hold_18_3 hold_18_4
 hold_18_5 hold_18_6 hold_19_1 hold_19_2 hold_19_3 hold_19_4 hold_19_5
 hold_19_6 hold_20_1 hold_20_2 hold_20_3 hold_20_4 hold_20_5 hold_20_6
 hold_21_1 hold_21_2 hold_21_3 hold_21_4 hold_21_5 hold_21_6 hold_22_1
 hold_22_2 hold_22_3 hold_22_4 hold_22_5 hold_22_6 hold_23_1 hold_23_2
 hold_23_3 hold_23_4 hold_23_5 hold_23_6 hold_24_1 hold_24_2 hold_24_3
 hold_24_4 hold_24_5 hold_24_6 hold_25_1 hold_25_2 hold_25_3 hold_25_4
 hold_25_5 hold_25_6 hold_26_1 hold_26_2 hold_26_3 hold_26_4 hold_26_5
 hold_26_6 hold_27_1 hold_27_2 hold_27_3 hold_27_4 hold_27_5 hold_27_6
 hold_28_1 hold_28_2 hold_28_3 hold_28_4 hold_28_5 hold_28_6 hold_29_1
 hold_29_2 hold_29_3 hold_29_4 hold_29_5 hold_29_6 hold_30_1 hold_30_2
 hold_30_3 hold_30_4 hold_30_5 hold_30_6 hold_31_1 hold_31_2 hold_31_3
 hold_31_4 hold_31_5 hold_31_6 switch_source_nic_in_0 switch_nic_out_0_sink
 switch_nic_out_0_nic_in_1 switch_nic_out_0_nic_in_2
 switch_nic_out_0_nic_in_3 switch_nic_out_0_nic_in_4
 switch_nic_out_0_nic_in_5 switch_nic_out_0_nic_in_6
 switch_nic_out_0_nic_in_7 switch_nic_out_0_nic_in_8
 switch_nic_out_0_nic_in_9 switch_nic_out_0_nic_in_10
 switch_nic_out_0_nic_in_11 switch_nic_out_0_nic_in_12
 switch_nic_out_0_nic_in_13 switch_nic_out_0_nic_in_14
 switch_nic_out_0_nic_in_15 switch_nic_out_0_nic_in_16
 switch_nic_out_0_nic_in_17 switch_nic_out_0_nic_in_18
 switch_nic_out_0_nic_in_19 switch_nic_out_0_nic_in_20
 switch_nic_out_0_nic_in_21 switch_nic_out_0_nic_in_22
 switch_nic_out_0_nic_in_23 switch_nic_out_0_nic_in_24
 switch_nic_out_0_nic_in_25 switch_nic_out_0_nic_in_26
 switch_nic_out_0_nic_in_27 switch_nic_out_0_nic_in_28
 switch_nic_out_0_nic_in_29 switch_nic_out_0_nic_in_30
 switch_nic_out_0_nic_in_31 switch_nic_in_0_0 switch_0_nic_out_0
 switch_source_nic_in_1 switch_nic_out_1_sink switch_nic_out_1_nic_in_0
 switch_nic_out_1_nic_in_2 switch_nic_out_1_nic_in_3
 switch_nic_out_1_nic_in_4 switch_nic_out_1_nic_in_5
 switch_nic_out_1_nic_in_6 switch_nic_out_1_nic_in_7
 switch_nic_out_1_nic_in_8 switch_nic_out_1_nic_in_9
 switch_nic_out_1_nic_in_10 switch_nic_out_1_nic_in_11
 switch_nic_out_1_nic_in_12 switch_nic_out_1_nic_in_13
 switch_nic_out_1_nic_in_14 switch_nic_out_1_nic_in_15
 switch_nic_out_1_nic_in_16 switch_nic_out_1_nic_in_17
 switch_nic_out_1_nic_in_18 switch_nic_out_1_nic_in_19
 switch_nic_out_1_nic_in_20 switch_nic_out_1_nic_in_21
 switch_nic_out_1_nic_in_22 switch_nic_out_1_nic_in_23
 switch_nic_out_1_nic_in_24 switch_nic_out_1_nic_in_25
 switch_nic_out_1_nic_in_26 switch_nic_out_1_nic_in_27
 switch_nic_out_1_nic_in_28 switch_nic_out_1_nic_in_29
 switch_nic_out_1_nic_in_30 switch_nic_out_1_nic_in_31 switch_nic_in_1_1
 switch_1_nic_out_1 switch_source_nic_in_2 switch_nic_out_2_sink
 switch_nic_out_2_nic_in_0 switch_nic_out_2_nic_in_1
 switch_nic_out_2_nic_in_3 switch_nic_out_2_nic_in_4
 switch_nic_out_2_nic_in_5 switch_nic_out_2_nic_in_6
 switch_nic_out_2_nic_in_7 switch_nic_out_2_nic_in_8
 switch_nic_out_2_nic_in_9 switch_nic_out_2_nic_in_10
 switch_nic_out_2_nic_in_11 switch_nic_out_2_nic_in_12
 switch_nic_out_2_nic_in_13 switch_nic_out_2_nic_in_14
 switch_nic_out_2_nic_in_15 switch_nic_out_2_nic_in_16
 switch_nic_out_2_nic_in_17 switch_nic_out_2_nic_in_18
 switch_nic_out_2_nic_in_19 switch_nic_out_2_nic_in_20
 switch_nic_out_2_nic_in_21 switch_nic_out_2_nic_in_22
 switch_nic_out_2_nic_in_23 switch_nic_out_2_nic_in_24
 switch_nic_out_2_nic_in_25 switch_nic_out_2_nic_in_26
 switch_nic_out_2_nic_in_27 switch_nic_out_2_nic_in_28
 switch_nic_out_2_nic_in_29 switch_nic_out_2_nic_in_30
 switch_nic_out_2_nic_in_31 switch_nic_in_2_2 switch_2_nic_out_2
 switch_source_nic_in_3 switch_nic_out_3_sink switch_nic_out_3_nic_in_0
 switch_nic_out_3_nic_in_1 switch_nic_out_3_nic_in_2
 switch_nic_out_3_nic_in_4 switch_nic_out_3_nic_in_5
 switch_nic_out_3_nic_in_6 switch_nic_out_3_nic_in_7
 switch_nic_out_3_nic_in_8 switch_nic_out_3_nic_in_9
 switch_nic_out_3_nic_in_10 switch_nic_out_3_nic_in_11
 switch_nic_out_3_nic_in_12 switch_nic_out_3_nic_in_13
 switch_nic_out_3_nic_in_14 switch_nic_out_3_nic_in_15
 switch_nic_out_3_nic_in_16 switch_nic_out_3_nic_in_17
 switch_nic_out_3_nic_in_18 switch_nic_out_3_nic_in_19
 switch_nic_out_3_nic_in_20 switch_nic_out_3_nic_in_21
 switch_nic_out_3_nic_in_22 switch_nic_out_3_nic_in_23
 switch_nic_out_3_nic_in_24 switch_nic_out_3_nic_in_25
 switch_nic_out_3_nic_in_26 switch_nic_out_3_nic_in_27
 switch_nic_out_3_nic_in_28 switch_nic_out_3_nic_in_29
 switch_nic_out_3_nic_in_30 switch_nic_out_3_nic_in_31 switch_nic_in_3_3
 switch_3_nic_out_3 switch_source_nic_in_4 switch_nic_out_4_sink
 switch_nic_out_4_nic_in_0 switch_nic_out_4_nic_in_1
 switch_nic_out_4_nic_in_2 switch_nic_out_4_nic_in_3
 switch_nic_out_4_nic_in_5 switch_nic_out_4_nic_in_6
 switch_nic_out_4_nic_in_7 switch_nic_out_4_nic_in_8
 switch_nic_out_4_nic_in_9 switch_nic_out_4_nic_in_10
 switch_nic_out_4_nic_in_11 switch_nic_out_4_nic_in_12
 switch_nic_out_4_nic_in_13 switch_nic_out_4_nic_in_14
 switch_nic_out_4_nic_in_15 switch_nic_out_4_nic_in_16
 switch_nic_out_4_nic_in_17 switch_nic_out_4_nic_in_18
 switch_nic_out_4_nic_in_19 switch_nic_out_4_nic_in_20
 switch_nic_out_4_nic_in_21 switch_nic_out_4_nic_in_22
 switch_nic_out_4_nic_in_23 switch_nic_out_4_nic_in_24
 switch_nic_out_4_nic_in_25 switch_nic_out_4_nic_in_26
 switch_nic_out_4_nic_in_27 switch_nic_out_4_nic_in_28
 switch_nic_out_4_nic_in_29 switch_nic_out_4_nic_in_30
 switch_nic_out_4_nic_in_31 switch_nic_in_4_4 switch_4_nic_out_4
 switch_source_nic_in_5 switch_nic_out_5_sink switch_nic_out_5_nic_in_0
 switch_nic_out_5_nic_in_1 switch_nic_out_5_nic_in_2
 switch_nic_out_5_nic_in_3 switch_nic_out_5_nic_in_4
 switch_nic_out_5_nic_in_6 switch_nic_out_5_nic_in_7
 switch_nic_out_5_nic_in_8 switch_nic_out_5_nic_in_9
 switch_nic_out_5_nic_in_10 switch_nic_out_5_nic_in_11
 switch_nic_out_5_nic_in_12 switch_nic_out_5_nic_in_13
 switch_nic_out_5_nic_in_14 switch_nic_out_5_nic_in_15
 switch_nic_out_5_nic_in_16 switch_nic_out_5_nic_in_17
 switch_nic_out_5_nic_in_18 switch_nic_out_5_nic_in_19
 switch_nic_out_5_nic_in_20 switch_nic_out_5_nic_in_21
 switch_nic_out_5_nic_in_22 switch_nic_out_5_nic_in_23
 switch_nic_out_5_nic_in_24 switch_nic_out_5_nic_in_25
 switch_nic_out_5_nic_in_26 switch_nic_out_5_nic_in_27
 switch_nic_out_5_nic_in_28 switch_nic_out_5_nic_in_29
 switch_nic_out_5_nic_in_30 switch_nic_out_5_nic_in_31 switch_nic_in_5_5
 switch_5_nic_out_5 switch_source_nic_in_6 switch_nic_out_6_sink
 switch_nic_out_6_nic_in_0 switch_nic_out_6_nic_in_1
 switch_nic_out_6_nic_in_2 switch_nic_out_6_nic_in_3
 switch_nic_out_6_nic_in_4 switch_nic_out_6_nic_in_5
 switch_nic_out_6_nic_in_7 switch_nic_out_6_nic_in_8
 switch_nic_out_6_nic_in_9 switch_nic_out_6_nic_in_10
 switch_nic_out_6_nic_in_11 switch_nic_out_6_nic_in_12
 switch_nic_out_6_nic_in_13 switch_nic_out_6_nic_in_14
 switch_nic_out_6_nic_in_15 switch_nic_out_6_nic_in_16
 switch_nic_out_6_nic_in_17 switch_nic_out_6_nic_in_18
 switch_nic_out_6_nic_in_19 switch_nic_out_6_nic_in_20
 switch_nic_out_6_nic_in_21 switch_nic_out_6_nic_in_22
 switch_nic_out_6_nic_in_23 switch_nic_out_6_nic_in_24
 switch_nic_out_6_nic_in_25 switch_nic_out_6_nic_in_26
 switch_nic_out_6_nic_in_27 switch_nic_out_6_nic_in_28
 switch_nic_out_6_nic_in_29 switch_nic_out_6_nic_in_30
 switch_nic_out_6_nic_in_31 switch_nic_in_6_6 switch_6_nic_out_6
 switch_source_nic_in_7 switch_nic_out_7_sink switch_nic_out_7_nic_in_0
 switch_nic_out_7_nic_in_1 switch_nic_out_7_nic_in_2
 switch_nic_out_7_nic_in_3 switch_nic_out_7_nic_in_4
 switch_nic_out_7_nic_in_5 switch_nic_out_7_nic_in_6
 switch_nic_out_7_nic_in_8 switch_nic_out_7_nic_in_9
 switch_nic_out_7_nic_in_10 switch_nic_out_7_nic_in_11
 switch_nic_out_7_nic_in_12 switch_nic_out_7_nic_in_13
 switch_nic_out_7_nic_in_14 switch_nic_out_7_nic_in_15
 switch_nic_out_7_nic_in_16 switch_nic_out_7_nic_in_17
 switch_nic_out_7_nic_in_18 switch_nic_out_7_nic_in_19
 switch_nic_out_7_nic_in_20 switch_nic_out_7_nic_in_21
 switch_nic_out_7_nic_in_22 switch_nic_out_7_nic_in_23
 switch_nic_out_7_nic_in_24 switch_nic_out_7_nic_in_25
 switch_nic_out_7_nic_in_26 switch_nic_out_7_nic_in_27
 switch_nic_out_7_nic_in_28 switch_nic_out_7_nic_in_29
 switch_nic_out_7_nic_in_30 switch_nic_out_7_nic_in_31 switch_nic_in_7_7
 switch_7_nic_out_7 switch_source_nic_in_8 switch_nic_out_8_sink
 switch_nic_out_8_nic_in_0 switch_nic_out_8_nic_in_1
 switch_nic_out_8_nic_in_2 switch_nic_out_8_nic_in_3
 switch_nic_out_8_nic_in_4 switch_nic_out_8_nic_in_5
 switch_nic_out_8_nic_in_6 switch_nic_out_8_nic_in_7
 switch_nic_out_8_nic_in_9 switch_nic_out_8_nic_in_10
 switch_nic_out_8_nic_in_11 switch_nic_out_8_nic_in_12
 switch_nic_out_8_nic_in_13 switch_nic_out_8_nic_in_14
 switch_nic_out_8_nic_in_15 switch_nic_out_8_nic_in_16
 switch_nic_out_8_nic_in_17 switch_nic_out_8_nic_in_18
 switch_nic_out_8_nic_in_19 switch_nic_out_8_nic_in_20
 switch_nic_out_8_nic_in_21 switch_nic_out_8_nic_in_22
 switch_nic_out_8_nic_in_23 switch_nic_out_8_nic_in_24
 switch_nic_out_8_nic_in_25 switch_nic_out_8_nic_in_26
 switch_nic_out_8_nic_in_27 switch_nic_out_8_nic_in_28
 switch_nic_out_8_nic_in_29 switch_nic_out_8_nic_in_30
 switch_nic_out_8_nic_in_31 switch_nic_in_8_8 switch_8_nic_out_8
 switch_source_nic_in_9 switch_nic_out_9_sink switch_nic_out_9_nic_in_0
 switch_nic_out_9_nic_in_1 switch_nic_out_9_nic_in_2
 switch_nic_out_9_nic_in_3 switch_nic_out_9_nic_in_4
 switch_nic_out_9_nic_in_5 switch_nic_out_9_nic_in_6
 switch_nic_out_9_nic_in_7 switch_nic_out_9_nic_in_8
 switch_nic_out_9_nic_in_10 switch_nic_out_9_nic_in_11
 switch_nic_out_9_nic_in_12 switch_nic_out_9_nic_in_13
 switch_nic_out_9_nic_in_14 switch_nic_out_9_nic_in_15
 switch_nic_out_9_nic_in_16 switch_nic_out_9_nic_in_17
 switch_nic_out_9_nic_in_18 switch_nic_out_9_nic_in_19
 switch_nic_out_9_nic_in_20 switch_nic_out_9_nic_in_21
 switch_nic_out_9_nic_in_22 switch_nic_out_9_nic_in_23
 switch_nic_out_9_nic_in_24 switch_nic_out_9_nic_in_25
 switch_nic_out_9_nic_in_26 switch_nic_out_9_nic_in_27
 switch_nic_out_9_nic_in_28 switch_nic_out_9_nic_in_29
 switch_nic_out_9_nic_in_30 switch_nic_out_9_nic_in_31 switch_nic_in_9_9
 switch_9_nic_out_9 switch_source_nic_in_10 switch_nic_out_10_sink
 switch_nic_out_10_nic_in_0 switch_nic_out_10_nic_in_1
 switch_nic_out_10_nic_in_2 switch_nic_out_10_nic_in_3
 switch_nic_out_10_nic_in_4 switch_nic_out_10_nic_in_5
 switch_nic_out_10_nic_in_6 switch_nic_out_10_nic_in_7
 switch_nic_out_10_nic_in_8 switch_nic_out_10_nic_in_9
 switch_nic_out_10_nic_in_11 switch_nic_out_10_nic_in_12
 switch_nic_out_10_nic_in_13 switch_nic_out_10_nic_in_14
 switch_nic_out_10_nic_in_15 switch_nic_out_10_nic_in_16
 switch_nic_out_10_nic_in_17 switch_nic_out_10_nic_in_18
 switch_nic_out_10_nic_in_19 switch_nic_out_10_nic_in_20
 switch_nic_out_10_nic_in_21 switch_nic_out_10_nic_in_22
 switch_nic_out_10_nic_in_23 switch_nic_out_10_nic_in_24
 switch_nic_out_10_nic_in_25 switch_nic_out_10_nic_in_26
 switch_nic_out_10_nic_in_27 switch_nic_out_10_nic_in_28
 switch_nic_out_10_nic_in_29 switch_nic_out_10_nic_in_30
 switch_nic_out_10_nic_in_31 switch_nic_in_10_10 switch_10_nic_out_10
 switch_source_nic_in_11 switch_nic_out_11_sink switch_nic_out_11_nic_in_0
 switch_nic_out_11_nic_in_1 switch_nic_out_11_nic_in_2
 switch_nic_out_11_nic_in_3 switch_nic_out_11_nic_in_4
 switch_nic_out_11_nic_in_5 switch_nic_out_11_nic_in_6
 switch_nic_out_11_nic_in_7 switch_nic_out_11_nic_in_8
 switch_nic_out_11_nic_in_9 switch_nic_out_11_nic_in_10
 switch_nic_out_11_nic_in_12 switch_nic_out_11_nic_in_13
 switch_nic_out_11_nic_in_14 switch_nic_out_11_nic_in_15
 switch_nic_out_11_nic_in_16 switch_nic_out_11_nic_in_17
 switch_nic_out_11_nic_in_18 switch_nic_out_11_nic_in_19
 switch_nic_out_11_nic_in_20 switch_nic_out_11_nic_in_21
 switch_nic_out_11_nic_in_22 switch_nic_out_11_nic_in_23
 switch_nic_out_11_nic_in_24 switch_nic_out_11_nic_in_25
 switch_nic_out_11_nic_in_26 switch_nic_out_11_nic_in_27
 switch_nic_out_11_nic_in_28 switch_nic_out_11_nic_in_29
 switch_nic_out_11_nic_in_30 switch_nic_out_11_nic_in_31
 switch_nic_in_11_11 switch_11_nic_out_11 switch_source_nic_in_12
 switch_nic_out_12_sink switch_nic_out_12_nic_in_0
 switch_nic_out_12_nic_in_1 switch_nic_out_12_nic_in_2
 switch_nic_out_12_nic_in_3 switch_nic_out_12_nic_in_4
 switch_nic_out_12_nic_in_5 switch_nic_out_12_nic_in_6
 switch_nic_out_12_nic_in_7 switch_nic_out_12_nic_in_8
 switch_nic_out_12_nic_in_9 switch_nic_out_12_nic_in_10
 switch_nic_out_12_nic_in_11 switch_nic_out_12_nic_in_13
 switch_nic_out_12_nic_in_14 switch_nic_out_12_nic_in_15
 switch_nic_out_12_nic_in_16 switch_nic_out_12_nic_in_17
 switch_nic_out_12_nic_in_18 switch_nic_out_12_nic_in_19
 switch_nic_out_12_nic_in_20 switch_nic_out_12_nic_in_21
 switch_nic_out_12_nic_in_22 switch_nic_out_12_nic_in_23
 switch_nic_out_12_nic_in_24 switch_nic_out_12_nic_in_25
 switch_nic_out_12_nic_in_26 switch_nic_out_12_nic_in_27
 switch_nic_out_12_nic_in_28 switch_nic_out_12_nic_in_29
 switch_nic_out_12_nic_in_30 switch_nic_out_12_nic_in_31
 switch_nic_in_12_12 switch_12_nic_out_12 switch_source_nic_in_13
 switch_nic_out_13_sink switch_nic_out_13_nic_in_0
 switch_nic_out_13_nic_in_1 switch_nic_out_13_nic_in_2
 switch_nic_out_13_nic_in_3 switch_nic_out_13_nic_in_4
 switch_nic_out_13_nic_in_5 switch_nic_out_13_nic_in_6
 switch_nic_out_13_nic_in_7 switch_nic_out_13_nic_in_8
 switch_nic_out_13_nic_in_9 switch_nic_out_13_nic_in_10
 switch_nic_out_13_nic_in_11 switch_nic_out_13_nic_in_12
 switch_nic_out_13_nic_in_14 switch_nic_out_13_nic_in_15
 switch_nic_out_13_nic_in_16 switch_nic_out_13_nic_in_17
 switch_nic_out_13_nic_in_18 switch_nic_out_13_nic_in_19
 switch_nic_out_13_nic_in_20 switch_nic_out_13_nic_in_21
 switch_nic_out_13_nic_in_22 switch_nic_out_13_nic_in_23
 switch_nic_out_13_nic_in_24 switch_nic_out_13_nic_in_25
 switch_nic_out_13_nic_in_26 switch_nic_out_13_nic_in_27
 switch_nic_out_13_nic_in_28 switch_nic_out_13_nic_in_29
 switch_nic_out_13_nic_in_30 switch_nic_out_13_nic_in_31
 switch_nic_in_13_13 switch_13_nic_out_13 switch_source_nic_in_14
 switch_nic_out_14_sink switch_nic_out_14_nic_in_0
 switch_nic_out_14_nic_in_1 switch_nic_out_14_nic_in_2
 switch_nic_out_14_nic_in_3 switch_nic_out_14_nic_in_4
 switch_nic_out_14_nic_in_5 switch_nic_out_14_nic_in_6
 switch_nic_out_14_nic_in_7 switch_nic_out_14_nic_in_8
 switch_nic_out_14_nic_in_9 switch_nic_out_14_nic_in_10
 switch_nic_out_14_nic_in_11 switch_nic_out_14_nic_in_12
 switch_nic_out_14_nic_in_13 switch_nic_out_14_nic_in_15
 switch_nic_out_14_nic_in_16 switch_nic_out_14_nic_in_17
 switch_nic_out_14_nic_in_18 switch_nic_out_14_nic_in_19
 switch_nic_out_14_nic_in_20 switch_nic_out_14_nic_in_21
 switch_nic_out_14_nic_in_22 switch_nic_out_14_nic_in_23
 switch_nic_out_14_nic_in_24 switch_nic_out_14_nic_in_25
 switch_nic_out_14_nic_in_26 switch_nic_out_14_nic_in_27
 switch_nic_out_14_nic_in_28 switch_nic_out_14_nic_in_29
 switch_nic_out_14_nic_in_30 switch_nic_out_14_nic_in_31
 switch_nic_in_14_14 switch_14_nic_out_14 switch_source_nic_in_15
 switch_nic_out_15_sink switch_nic_out_15_nic_in_0
 switch_nic_out_15_nic_in_1 switch_nic_out_15_nic_in_2
 switch_nic_out_15_nic_in_3 switch_nic_out_15_nic_in_4
 switch_nic_out_15_nic_in_5 switch_nic_out_15_nic_in_6
 switch_nic_out_15_nic_in_7 switch_nic_out_15_nic_in_8
 switch_nic_out_15_nic_in_9 switch_nic_out_15_nic_in_10
 switch_nic_out_15_nic_in_11 switch_nic_out_15_nic_in_12
 switch_nic_out_15_nic_in_13 switch_nic_out_15_nic_in_14
 switch_nic_out_15_nic_in_16 switch_nic_out_15_nic_in_17
 switch_nic_out_15_nic_in_18 switch_nic_out_15_nic_in_19
 switch_nic_out_15_nic_in_20 switch_nic_out_15_nic_in_21
 switch_nic_out_15_nic_in_22 switch_nic_out_15_nic_in_23
 switch_nic_out_15_nic_in_24 switch_nic_out_15_nic_in_25
 switch_nic_out_15_nic_in_26 switch_nic_out_15_nic_in_27
 switch_nic_out_15_nic_in_28 switch_nic_out_15_nic_in_29
 switch_nic_out_15_nic_in_30 switch_nic_out_15_nic_in_31
 switch_nic_in_15_15 switch_15_nic_out_15 switch_source_nic_in_16
 switch_nic_out_16_sink switch_nic_out_16_nic_in_0
 switch_nic_out_16_nic_in_1 switch_nic_out_16_nic_in_2
 switch_nic_out_16_nic_in_3 switch_nic_out_16_nic_in_4
 switch_nic_out_16_nic_in_5 switch_nic_out_16_nic_in_6
 switch_nic_out_16_nic_in_7 switch_nic_out_16_nic_in_8
 switch_nic_out_16_nic_in_9 switch_nic_out_16_nic_in_10
 switch_nic_out_16_nic_in_11 switch_nic_out_16_nic_in_12
 switch_nic_out_16_nic_in_13 switch_nic_out_16_nic_in_14
 switch_nic_out_16_nic_in_15 switch_nic_out_16_nic_in_17
 switch_nic_out_16_nic_in_18 switch_nic_out_16_nic_in_19
 switch_nic_out_16_nic_in_20 switch_nic_out_16_nic_in_21
 switch_nic_out_16_nic_in_22 switch_nic_out_16_nic_in_23
 switch_nic_out_16_nic_in_24 switch_nic_out_16_nic_in_25
 switch_nic_out_16_nic_in_26 switch_nic_out_16_nic_in_27
 switch_nic_out_16_nic_in_28 switch_nic_out_16_nic_in_29
 switch_nic_out_16_nic_in_30 switch_nic_out_16_nic_in_31
 switch_nic_in_16_16 switch_16_nic_out_16 switch_source_nic_in_17
 switch_nic_out_17_sink switch_nic_out_17_nic_in_0
 switch_nic_out_17_nic_in_1 switch_nic_out_17_nic_in_2
 switch_nic_out_17_nic_in_3 switch_nic_out_17_nic_in_4
 switch_nic_out_17_nic_in_5 switch_nic_out_17_nic_in_6
 switch_nic_out_17_nic_in_7 switch_nic_out_17_nic_in_8
 switch_nic_out_17_nic_in_9 switch_nic_out_17_nic_in_10
 switch_nic_out_17_nic_in_11 switch_nic_out_17_nic_in_12
 switch_nic_out_17_nic_in_13 switch_nic_out_17_nic_in_14
 switch_nic_out_17_nic_in_15 switch_nic_out_17_nic_in_16
 switch_nic_out_17_nic_in_18 switch_nic_out_17_nic_in_19
 switch_nic_out_17_nic_in_20 switch_nic_out_17_nic_in_21
 switch_nic_out_17_nic_in_22 switch_nic_out_17_nic_in_23
 switch_nic_out_17_nic_in_24 switch_nic_out_17_nic_in_25
 switch_nic_out_17_nic_in_26 switch_nic_out_17_nic_in_27
 switch_nic_out_17_nic_in_28 switch_nic_out_17_nic_in_29
 switch_nic_out_17_nic_in_30 switch_nic_out_17_nic_in_31
 switch_nic_in_17_17 switch_17_nic_out_17 switch_source_nic_in_18
 switch_nic_out_18_sink switch_nic_out_18_nic_in_0
 switch_nic_out_18_nic_in_1 switch_nic_out_18_nic_in_2
 switch_nic_out_18_nic_in_3 switch_nic_out_18_nic_in_4
 switch_nic_out_18_nic_in_5 switch_nic_out_18_nic_in_6
 switch_nic_out_18_nic_in_7 switch_nic_out_18_nic_in_8
 switch_nic_out_18_nic_in_9 switch_nic_out_18_nic_in_10
 switch_nic_out_18_nic_in_11 switch_nic_out_18_nic_in_12
 switch_nic_out_18_nic_in_13 switch_nic_out_18_nic_in_14
 switch_nic_out_18_nic_in_15 switch_nic_out_18_nic_in_16
 switch_nic_out_18_nic_in_17 switch_nic_out_18_nic_in_19
 switch_nic_out_18_nic_in_20 switch_nic_out_18_nic_in_21
 switch_nic_out_18_nic_in_22 switch_nic_out_18_nic_in_23
 switch_nic_out_18_nic_in_24 switch_nic_out_18_nic_in_25
 switch_nic_out_18_nic_in_26 switch_nic_out_18_nic_in_27
 switch_nic_out_18_nic_in_28 switch_nic_out_18_nic_in_29
 switch_nic_out_18_nic_in_30 switch_nic_out_18_nic_in_31
 switch_nic_in_18_18 switch_18_nic_out_18 switch_source_nic_in_19
 switch_nic_out_19_sink switch_nic_out_19_nic_in_0
 switch_nic_out_19_nic_in_1 switch_nic_out_19_nic_in_2
 switch_nic_out_19_nic_in_3 switch_nic_out_19_nic_in_4
 switch_nic_out_19_nic_in_5 switch_nic_out_19_nic_in_6
 switch_nic_out_19_nic_in_7 switch_nic_out_19_nic_in_8
 switch_nic_out_19_nic_in_9 switch_nic_out_19_nic_in_10
 switch_nic_out_19_nic_in_11 switch_nic_out_19_nic_in_12
 switch_nic_out_19_nic_in_13 switch_nic_out_19_nic_in_14
 switch_nic_out_19_nic_in_15 switch_nic_out_19_nic_in_16
 switch_nic_out_19_nic_in_17 switch_nic_out_19_nic_in_18
 switch_nic_out_19_nic_in_20 switch_nic_out_19_nic_in_21
 switch_nic_out_19_nic_in_22 switch_nic_out_19_nic_in_23
 switch_nic_out_19_nic_in_24 switch_nic_out_19_nic_in_25
 switch_nic_out_19_nic_in_26 switch_nic_out_19_nic_in_27
 switch_nic_out_19_nic_in_28 switch_nic_out_19_nic_in_29
 switch_nic_out_19_nic_in_30 switch_nic_out_19_nic_in_31
 switch_nic_in_19_19 switch_19_nic_out_19 switch_source_nic_in_20
 switch_nic_out_20_sink switch_nic_out_20_nic_in_0
 switch_nic_out_20_nic_in_1 switch_nic_out_20_nic_in_2
 switch_nic_out_20_nic_in_3 switch_nic_out_20_nic_in_4
 switch_nic_out_20_nic_in_5 switch_nic_out_20_nic_in_6
 switch_nic_out_20_nic_in_7 switch_nic_out_20_nic_in_8
 switch_nic_out_20_nic_in_9 switch_nic_out_20_nic_in_10
 switch_nic_out_20_nic_in_11 switch_nic_out_20_nic_in_12
 switch_nic_out_20_nic_in_13 switch_nic_out_20_nic_in_14
 switch_nic_out_20_nic_in_15 switch_nic_out_20_nic_in_16
 switch_nic_out_20_nic_in_17 switch_nic_out_20_nic_in_18
 switch_nic_out_20_nic_in_19 switch_nic_out_20_nic_in_21
 switch_nic_out_20_nic_in_22 switch_nic_out_20_nic_in_23
 switch_nic_out_20_nic_in_24 switch_nic_out_20_nic_in_25
 switch_nic_out_20_nic_in_26 switch_nic_out_20_nic_in_27
 switch_nic_out_20_nic_in_28 switch_nic_out_20_nic_in_29
 switch_nic_out_20_nic_in_30 switch_nic_out_20_nic_in_31
 switch_nic_in_20_20 switch_20_nic_out_20 switch_source_nic_in_21
 switch_nic_out_21_sink switch_nic_out_21_nic_in_0
 switch_nic_out_21_nic_in_1 switch_nic_out_21_nic_in_2
 switch_nic_out_21_nic_in_3 switch_nic_out_21_nic_in_4
 switch_nic_out_21_nic_in_5 switch_nic_out_21_nic_in_6
 switch_nic_out_21_nic_in_7 switch_nic_out_21_nic_in_8
 switch_nic_out_21_nic_in_9 switch_nic_out_21_nic_in_10
 switch_nic_out_21_nic_in_11 switch_nic_out_21_nic_in_12
 switch_nic_out_21_nic_in_13 switch_nic_out_21_nic_in_14
 switch_nic_out_21_nic_in_15 switch_nic_out_21_nic_in_16
 switch_nic_out_21_nic_in_17 switch_nic_out_21_nic_in_18
 switch_nic_out_21_nic_in_19 switch_nic_out_21_nic_in_20
 switch_nic_out_21_nic_in_22 switch_nic_out_21_nic_in_23
 switch_nic_out_21_nic_in_24 switch_nic_out_21_nic_in_25
 switch_nic_out_21_nic_in_26 switch_nic_out_21_nic_in_27
 switch_nic_out_21_nic_in_28 switch_nic_out_21_nic_in_29
 switch_nic_out_21_nic_in_30 switch_nic_out_21_nic_in_31
 switch_nic_in_21_21 switch_21_nic_out_21 switch_source_nic_in_22
 switch_nic_out_22_sink switch_nic_out_22_nic_in_0
 switch_nic_out_22_nic_in_1 switch_nic_out_22_nic_in_2
 switch_nic_out_22_nic_in_3 switch_nic_out_22_nic_in_4
 switch_nic_out_22_nic_in_5 switch_nic_out_22_nic_in_6
 switch_nic_out_22_nic_in_7 switch_nic_out_22_nic_in_8
 switch_nic_out_22_nic_in_9 switch_nic_out_22_nic_in_10
 switch_nic_out_22_nic_in_11 switch_nic_out_22_nic_in_12
 switch_nic_out_22_nic_in_13 switch_nic_out_22_nic_in_14
 switch_nic_out_22_nic_in_15 switch_nic_out_22_nic_in_16
 switch_nic_out_22_nic_in_17 switch_nic_out_22_nic_in_18
 switch_nic_out_22_nic_in_19 switch_nic_out_22_nic_in_20
 switch_nic_out_22_nic_in_21 switch_nic_out_22_nic_in_23
 switch_nic_out_22_nic_in_24 switch_nic_out_22_nic_in_25
 switch_nic_out_22_nic_in_26 switch_nic_out_22_nic_in_27
 switch_nic_out_22_nic_in_28 switch_nic_out_22_nic_in_29
 switch_nic_out_22_nic_in_30 switch_nic_out_22_nic_in_31
 switch_nic_in_22_22 switch_22_nic_out_22 switch_source_nic_in_23
 switch_nic_out_23_sink switch_nic_out_23_nic_in_0
 switch_nic_out_23_nic_in_1 switch_nic_out_23_nic_in_2
 switch_nic_out_23_nic_in_3 switch_nic_out_23_nic_in_4
 switch_nic_out_23_nic_in_5 switch_nic_out_23_nic_in_6
 switch_nic_out_23_nic_in_7 switch_nic_out_23_nic_in_8
 switch_nic_out_23_nic_in_9 switch_nic_out_23_nic_in_10
 switch_nic_out_23_nic_in_11 switch_nic_out_23_nic_in_12
 switch_nic_out_23_nic_in_13 switch_nic_out_23_nic_in_14
 switch_nic_out_23_nic_in_15 switch_nic_out_23_nic_in_16
 switch_nic_out_23_nic_in_17 switch_nic_out_23_nic_in_18
 switch_nic_out_23_nic_in_19 switch_nic_out_23_nic_in_20
 switch_nic_out_23_nic_in_21 switch_nic_out_23_nic_in_22
 switch_nic_out_23_nic_in_24 switch_nic_out_23_nic_in_25
 switch_nic_out_23_nic_in_26 switch_nic_out_23_nic_in_27
 switch_nic_out_23_nic_in_28 switch_nic_out_23_nic_in_29
 switch_nic_out_23_nic_in_30 switch_nic_out_23_nic_in_31
 switch_nic_in_23_23 switch_23_nic_out_23 switch_source_nic_in_24
 switch_nic_out_24_sink switch_nic_out_24_nic_in_0
 switch_nic_out_24_nic_in_1 switch_nic_out_24_nic_in_2
 switch_nic_out_24_nic_in_3 switch_nic_out_24_nic_in_4
 switch_nic_out_24_nic_in_5 switch_nic_out_24_nic_in_6
 switch_nic_out_24_nic_in_7 switch_nic_out_24_nic_in_8
 switch_nic_out_24_nic_in_9 switch_nic_out_24_nic_in_10
 switch_nic_out_24_nic_in_11 switch_nic_out_24_nic_in_12
 switch_nic_out_24_nic_in_13 switch_nic_out_24_nic_in_14
 switch_nic_out_24_nic_in_15 switch_nic_out_24_nic_in_16
 switch_nic_out_24_nic_in_17 switch_nic_out_24_nic_in_18
 switch_nic_out_24_nic_in_19 switch_nic_out_24_nic_in_20
 switch_nic_out_24_nic_in_21 switch_nic_out_24_nic_in_22
 switch_nic_out_24_nic_in_23 switch_nic_out_24_nic_in_25
 switch_nic_out_24_nic_in_26 switch_nic_out_24_nic_in_27
 switch_nic_out_24_nic_in_28 switch_nic_out_24_nic_in_29
 switch_nic_out_24_nic_in_30 switch_nic_out_24_nic_in_31
 switch_nic_in_24_24 switch_24_nic_out_24 switch_source_nic_in_25
 switch_nic_out_25_sink switch_nic_out_25_nic_in_0
 switch_nic_out_25_nic_in_1 switch_nic_out_25_nic_in_2
 switch_nic_out_25_nic_in_3 switch_nic_out_25_nic_in_4
 switch_nic_out_25_nic_in_5 switch_nic_out_25_nic_in_6
 switch_nic_out_25_nic_in_7 switch_nic_out_25_nic_in_8
 switch_nic_out_25_nic_in_9 switch_nic_out_25_nic_in_10
 switch_nic_out_25_nic_in_11 switch_nic_out_25_nic_in_12
 switch_nic_out_25_nic_in_13 switch_nic_out_25_nic_in_14
 switch_nic_out_25_nic_in_15 switch_nic_out_25_nic_in_16
 switch_nic_out_25_nic_in_17 switch_nic_out_25_nic_in_18
 switch_nic_out_25_nic_in_19 switch_nic_out_25_nic_in_20
 switch_nic_out_25_nic_in_21 switch_nic_out_25_nic_in_22
 switch_nic_out_25_nic_in_23 switch_nic_out_25_nic_in_24
 switch_nic_out_25_nic_in_26 switch_nic_out_25_nic_in_27
 switch_nic_out_25_nic_in_28 switch_nic_out_25_nic_in_29
 switch_nic_out_25_nic_in_30 switch_nic_out_25_nic_in_31
 switch_nic_in_25_25 switch_25_nic_out_25 switch_source_nic_in_26
 switch_nic_out_26_sink switch_nic_out_26_nic_in_0
 switch_nic_out_26_nic_in_1 switch_nic_out_26_nic_in_2
 switch_nic_out_26_nic_in_3 switch_nic_out_26_nic_in_4
 switch_nic_out_26_nic_in_5 switch_nic_out_26_nic_in_6
 switch_nic_out_26_nic_in_7 switch_nic_out_26_nic_in_8
 switch_nic_out_26_nic_in_9 switch_nic_out_26_nic_in_10
 switch_nic_out_26_nic_in_11 switch_nic_out_26_nic_in_12
 switch_nic_out_26_nic_in_13 switch_nic_out_26_nic_in_14
 switch_nic_out_26_nic_in_15 switch_nic_out_26_nic_in_16
 switch_nic_out_26_nic_in_17 switch_nic_out_26_nic_in_18
 switch_nic_out_26_nic_in_19 switch_nic_out_26_nic_in_20
 switch_nic_out_26_nic_in_21 switch_nic_out_26_nic_in_22
 switch_nic_out_26_nic_in_23 switch_nic_out_26_nic_in_24
 switch_nic_out_26_nic_in_25 switch_nic_out_26_nic_in_27
 switch_nic_out_26_nic_in_28 switch_nic_out_26_nic_in_29
 switch_nic_out_26_nic_in_30 switch_nic_out_26_nic_in_31
 switch_nic_in_26_26 switch_26_nic_out_26 switch_source_nic_in_27
 switch_nic_out_27_sink switch_nic_out_27_nic_in_0
 switch_nic_out_27_nic_in_1 switch_nic_out_27_nic_in_2
 switch_nic_out_27_nic_in_3 switch_nic_out_27_nic_in_4
 switch_nic_out_27_nic_in_5 switch_nic_out_27_nic_in_6
 switch_nic_out_27_nic_in_7 switch_nic_out_27_nic_in_8
 switch_nic_out_27_nic_in_9 switch_nic_out_27_nic_in_10
 switch_nic_out_27_nic_in_11 switch_nic_out_27_nic_in_12
 switch_nic_out_27_nic_in_13 switch_nic_out_27_nic_in_14
 switch_nic_out_27_nic_in_15 switch_nic_out_27_nic_in_16
 switch_nic_out_27_nic_in_17 switch_nic_out_27_nic_in_18
 switch_nic_out_27_nic_in_19 switch_nic_out_27_nic_in_20
 switch_nic_out_27_nic_in_21 switch_nic_out_27_nic_in_22
 switch_nic_out_27_nic_in_23 switch_nic_out_27_nic_in_24
 switch_nic_out_27_nic_in_25 switch_nic_out_27_nic_in_26
 switch_nic_out_27_nic_in_28 switch_nic_out_27_nic_in_29
 switch_nic_out_27_nic_in_30 switch_nic_out_27_nic_in_31
 switch_nic_in_27_27 switch_27_nic_out_27 switch_source_nic_in_28
 switch_nic_out_28_sink switch_nic_out_28_nic_in_0
 switch_nic_out_28_nic_in_1 switch_nic_out_28_nic_in_2
 switch_nic_out_28_nic_in_3 switch_nic_out_28_nic_in_4
 switch_nic_out_28_nic_in_5 switch_nic_out_28_nic_in_6
 switch_nic_out_28_nic_in_7 switch_nic_out_28_nic_in_8
 switch_nic_out_28_nic_in_9 switch_nic_out_28_nic_in_10
 switch_nic_out_28_nic_in_11 switch_nic_out_28_nic_in_12
 switch_nic_out_28_nic_in_13 switch_nic_out_28_nic_in_14
 switch_nic_out_28_nic_in_15 switch_nic_out_28_nic_in_16
 switch_nic_out_28_nic_in_17 switch_nic_out_28_nic_in_18
 switch_nic_out_28_nic_in_19 switch_nic_out_28_nic_in_20
 switch_nic_out_28_nic_in_21 switch_nic_out_28_nic_in_22
 switch_nic_out_28_nic_in_23 switch_nic_out_28_nic_in_24
 switch_nic_out_28_nic_in_25 switch_nic_out_28_nic_in_26
 switch_nic_out_28_nic_in_27 switch_nic_out_28_nic_in_29
 switch_nic_out_28_nic_in_30 switch_nic_out_28_nic_in_31
 switch_nic_in_28_28 switch_28_nic_out_28 switch_source_nic_in_29
 switch_nic_out_29_sink switch_nic_out_29_nic_in_0
 switch_nic_out_29_nic_in_1 switch_nic_out_29_nic_in_2
 switch_nic_out_29_nic_in_3 switch_nic_out_29_nic_in_4
 switch_nic_out_29_nic_in_5 switch_nic_out_29_nic_in_6
 switch_nic_out_29_nic_in_7 switch_nic_out_29_nic_in_8
 switch_nic_out_29_nic_in_9 switch_nic_out_29_nic_in_10
 switch_nic_out_29_nic_in_11 switch_nic_out_29_nic_in_12
 switch_nic_out_29_nic_in_13 switch_nic_out_29_nic_in_14
 switch_nic_out_29_nic_in_15 switch_nic_out_29_nic_in_16
 switch_nic_out_29_nic_in_17 switch_nic_out_29_nic_in_18
 switch_nic_out_29_nic_in_19 switch_nic_out_29_nic_in_20
 switch_nic_out_29_nic_in_21 switch_nic_out_29_nic_in_22
 switch_nic_out_29_nic_in_23 switch_nic_out_29_nic_in_24
 switch_nic_out_29_nic_in_25 switch_nic_out_29_nic_in_26
 switch_nic_out_29_nic_in_27 switch_nic_out_29_nic_in_28
 switch_nic_out_29_nic_in_30 switch_nic_out_29_nic_in_31
 switch_nic_in_29_29 switch_29_nic_out_29 switch_source_nic_in_30
 switch_nic_out_30_sink switch_nic_out_30_nic_in_0
 switch_nic_out_30_nic_in_1 switch_nic_out_30_nic_in_2
 switch_nic_out_30_nic_in_3 switch_nic_out_30_nic_in_4
 switch_nic_out_30_nic_in_5 switch_nic_out_30_nic_in_6
 switch_nic_out_30_nic_in_7 switch_nic_out_30_nic_in_8
 switch_nic_out_30_nic_in_9 switch_nic_out_30_nic_in_10
 switch_nic_out_30_nic_in_11 switch_nic_out_30_nic_in_12
 switch_nic_out_30_nic_in_13 switch_nic_out_30_nic_in_14
 switch_nic_out_30_nic_in_15 switch_nic_out_30_nic_in_16
 switch_nic_out_30_nic_in_17 switch_nic_out_30_nic_in_18
 switch_nic_out_30_nic_in_19 switch_nic_out_30_nic_in_20
 switch_nic_out_30_nic_in_21 switch_nic_out_30_nic_in_22
 switch_nic_out_30_nic_in_23 switch_nic_out_30_nic_in_24
 switch_nic_out_30_nic_in_25 switch_nic_out_30_nic_in_26
 switch_nic_out_30_nic_in_27 switch_nic_out_30_nic_in_28
 switch_nic_out_30_nic_in_29 switch_nic_out_30_nic_in_31
 switch_nic_in_30_30 switch_30_nic_out_30 switch_source_nic_in_31
 switch_nic_out_31_sink switch_nic_out_31_nic_in_0
 switch_nic_out_31_nic_in_1 switch_nic_out_31_nic_in_2
 switch_nic_out_31_nic_in_3 switch_nic_out_31_nic_in_4
 switch_nic_out_31_nic_in_5 switch_nic_out_31_nic_in_6
 switch_nic_out_31_nic_in_7 switch_nic_out_31_nic_in_8
 switch_nic_out_31_nic_in_9 switch_nic_out_31_nic_in_10
 switch_nic_out_31_nic_in_11 switch_nic_out_31_nic_in_12
 switch_nic_out_31_nic_in_13 switch_nic_out_31_nic_in_14
 switch_nic_out_31_nic_in_15 switch_nic_out_31_nic_in_16
 switch_nic_out_31_nic_in_17 switch_nic_out_31_nic_in_18
 switch_nic_out_31_nic_in_19 switch_nic_out_31_nic_in_20
 switch_nic_out_31_nic_in_21 switch_nic_out_31_nic_in_22
 switch_nic_out_31_nic_in_23 switch_nic_out_31_nic_in_24
 switch_nic_out_31_nic_in_25 switch_nic_out_31_nic_in_26
 switch_nic_out_31_nic_in_27 switch_nic_out_31_nic_in_28
 switch_nic_out_31_nic_in_29 switch_nic_out_31_nic_in_30
 switch_nic_in_31_31 switch_31_nic_out_31 group_tp_active_0
 group_tp_active_1 group_tp_active_2 group_tp_active_3 group_tp_active_4
 group_tp_active_5 group_tp_active_6 group_tp_active_7 group_tp_active_8
 group_tp_active_9 group_tp_active_10 group_tp_active_11 group_tp_active_12
 group_tp_active_13 group_tp_active_14 group_tp_active_15
 group_tp_active_16 group_tp_active_17 group_tp_active_18
 group_tp_active_19 group_tp_active_20 group_tp_active_21
 group_tp_active_22 group_tp_active_23 group_tp_active_24
 group_tp_active_25 group_tp_active_26 group_tp_active_27
 group_tp_active_28 group_tp_active_29 group_tp_active_30
 group_tp_active_31 is_active_0 is_active_1 is_active_2 is_active_3
 is_active_4 is_active_5 is_active_6 is_active_7 is_active_8 is_active_9
 is_active_10 is_active_11 is_active_12 is_active_13 is_active_14
 is_active_15 is_active_16 is_active_17 is_active_18 is_active_19
 is_active_20 is_active_21 is_active_22 is_active_23 is_active_24
 is_active_25 is_active_26 is_active_27 is_active_28 is_active_29
 is_active_30 is_active_31 is_start_zero_0_for_source_nic_in_0
 is_end_complete_0 is_start_zero_1_for_source_nic_in_1 is_end_complete_1
 is_start_zero_2_for_source_nic_in_2 is_end_complete_2
 is_start_zero_3_for_source_nic_in_3 is_end_complete_3
 is_start_zero_4_for_source_nic_in_4 is_end_complete_4
 is_start_zero_5_for_source_nic_in_5 is_end_complete_5
 is_start_zero_6_for_source_nic_in_6 is_end_complete_6
 is_start_zero_7_for_source_nic_in_7 is_end_complete_7
 is_start_zero_8_for_source_nic_in_8 is_end_complete_8
 is_start_zero_9_for_source_nic_in_9 is_end_complete_9
 is_start_zero_10_for_source_nic_in_10 is_end_complete_10
 is_start_zero_11_for_source_nic_in_11 is_end_complete_11
 is_start_zero_12_for_source_nic_in_12 is_end_complete_12
 is_start_zero_13_for_source_nic_in_13 is_end_complete_13
 is_start_zero_14_for_source_nic_in_14 is_end_complete_14
 is_start_zero_15_for_source_nic_in_15 is_end_complete_15
 is_start_zero_16_for_source_nic_in_16 is_end_complete_16
 is_start_zero_17_for_source_nic_in_17 is_end_complete_17
 is_start_zero_18_for_source_nic_in_18 is_end_complete_18
 is_start_zero_19_for_source_nic_in_19 is_end_complete_19
 is_start_zero_20_for_source_nic_in_20 is_end_complete_20
 is_start_zero_21_for_source_nic_in_21 is_end_complete_21
 is_start_zero_22_for_source_nic_in_22 is_end_complete_22
 is_start_zero_23_for_source_nic_in_23 is_end_complete_23
 is_start_zero_24_for_source_nic_in_24 is_end_complete_24
 is_start_zero_25_for_source_nic_in_25 is_end_complete_25
 is_start_zero_26_for_source_nic_in_26 is_end_complete_26
 is_start_zero_27_for_source_nic_in_27 is_end_complete_27
 is_start_zero_28_for_source_nic_in_28 is_end_complete_28
 is_start_zero_29_for_source_nic_in_29 is_end_complete_29
 is_start_zero_30_for_source_nic_in_30 is_end_complete_30
 is_start_zero_31_for_source_nic_in_31 is_end_complete_31
Generals
 start_0 start_1 start_2 start_3 start_4 start_5 start_6 start_7 start_8
 start_9 start_10 start_11 start_12 start_13 start_14 start_15 start_16
 start_17 start_18 start_19 start_20 start_21 start_22 start_23 start_24
 start_25 start_26 start_27 start_28 start_29 start_30 start_31
 group_start_1 val_end_0 group_end_0 group_start_2 group_start_3
 group_start_4 group_start_5 group_start_6 group_start_7 group_start_8
 group_start_9 group_start_10 group_start_11 group_start_12 group_start_13
 group_start_14 group_start_15 group_start_16 group_start_17 group_start_18
 group_start_19 group_start_20 group_start_21 group_start_22 group_start_23
 group_start_24 group_start_25 group_start_26 group_start_27 group_start_28
 group_start_29 group_start_30 group_start_31 group_start_0 val_end_1
 group_end_1 val_end_2 group_end_2 val_end_3 group_end_3 val_end_4
 group_end_4 val_end_5 group_end_5 val_end_6 group_end_6 val_end_7
 group_end_7 val_end_8 group_end_8 val_end_9 group_end_9 val_end_10
 group_end_10 val_end_11 group_end_11 val_end_12 group_end_12 val_end_13
 group_end_13 val_end_14 group_end_14 val_end_15 group_end_15 val_end_16
 group_end_16 val_end_17 group_end_17 val_end_18 group_end_18 val_end_19
 group_end_19 val_end_20 group_end_20 val_end_21 group_end_21 val_end_22
 group_end_22 val_end_23 group_end_23 val_end_24 group_end_24 val_end_25
 group_end_25 val_end_26 group_end_26 val_end_27 group_end_27 val_end_28
 group_end_28 val_end_29 group_end_29 val_end_30 group_end_30 val_end_31
 group_end_31
General Constraints
 min_start_1: group_start_1 = MIN ( start_1 )
 max_end_0: group_end_0 = MAX ( val_end_0 )
 min_start_2: group_start_2 = MIN ( start_2 )
 min_start_3: group_start_3 = MIN ( start_3 )
 min_start_4: group_start_4 = MIN ( start_4 )
 min_start_5: group_start_5 = MIN ( start_5 )
 min_start_6: group_start_6 = MIN ( start_6 )
 min_start_7: group_start_7 = MIN ( start_7 )
 min_start_8: group_start_8 = MIN ( start_8 )
 min_start_9: group_start_9 = MIN ( start_9 )
 min_start_10: group_start_10 = MIN ( start_10 )
 min_start_11: group_start_11 = MIN ( start_11 )
 min_start_12: group_start_12 = MIN ( start_12 )
 min_start_13: group_start_13 = MIN ( start_13 )
 min_start_14: group_start_14 = MIN ( start_14 )
 min_start_15: group_start_15 = MIN ( start_15 )
 min_start_16: group_start_16 = MIN ( start_16 )
 min_start_17: group_start_17 = MIN ( start_17 )
 min_start_18: group_start_18 = MIN ( start_18 )
 min_start_19: group_start_19 = MIN ( start_19 )
 min_start_20: group_start_20 = MIN ( start_20 )
 min_start_21: group_start_21 = MIN ( start_21 )
 min_start_22: group_start_22 = MIN ( start_22 )
 min_start_23: group_start_23 = MIN ( start_23 )
 min_start_24: group_start_24 = MIN ( start_24 )
 min_start_25: group_start_25 = MIN ( start_25 )
 min_start_26: group_start_26 = MIN ( start_26 )
 min_start_27: group_start_27 = MIN ( start_27 )
 min_start_28: group_start_28 = MIN ( start_28 )
 min_start_29: group_start_29 = MIN ( start_29 )
 min_start_30: group_start_30 = MIN ( start_30 )
 min_start_31: group_start_31 = MIN ( start_31 )
 min_start_0: group_start_0 = MIN ( start_0 )
 max_end_1: group_end_1 = MAX ( val_end_1 )
 max_end_2: group_end_2 = MAX ( val_end_2 )
 max_end_3: group_end_3 = MAX ( val_end_3 )
 max_end_4: group_end_4 = MAX ( val_end_4 )
 max_end_5: group_end_5 = MAX ( val_end_5 )
 max_end_6: group_end_6 = MAX ( val_end_6 )
 max_end_7: group_end_7 = MAX ( val_end_7 )
 max_end_8: group_end_8 = MAX ( val_end_8 )
 max_end_9: group_end_9 = MAX ( val_end_9 )
 max_end_10: group_end_10 = MAX ( val_end_10 )
 max_end_11: group_end_11 = MAX ( val_end_11 )
 max_end_12: group_end_12 = MAX ( val_end_12 )
 max_end_13: group_end_13 = MAX ( val_end_13 )
 max_end_14: group_end_14 = MAX ( val_end_14 )
 max_end_15: group_end_15 = MAX ( val_end_15 )
 max_end_16: group_end_16 = MAX ( val_end_16 )
 max_end_17: group_end_17 = MAX ( val_end_17 )
 max_end_18: group_end_18 = MAX ( val_end_18 )
 max_end_19: group_end_19 = MAX ( val_end_19 )
 max_end_20: group_end_20 = MAX ( val_end_20 )
 max_end_21: group_end_21 = MAX ( val_end_21 )
 max_end_22: group_end_22 = MAX ( val_end_22 )
 max_end_23: group_end_23 = MAX ( val_end_23 )
 max_end_24: group_end_24 = MAX ( val_end_24 )
 max_end_25: group_end_25 = MAX ( val_end_25 )
 max_end_26: group_end_26 = MAX ( val_end_26 )
 max_end_27: group_end_27 = MAX ( val_end_27 )
 max_end_28: group_end_28 = MAX ( val_end_28 )
 max_end_29: group_end_29 = MAX ( val_end_29 )
 max_end_30: group_end_30 = MAX ( val_end_30 )
 max_end_31: group_end_31 = MAX ( val_end_31 )
End
