\ Model 2026-02-26_19-42-16
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  1.2307475912511426e-04 flow_source_nic_in_0
   + 1.2307475912511426e-04 flow_source_nic_in_4
   + 1.2307475912511426e-04 flow_source_nic_in_6
   + 1.2307475912511426e-04 flow_source_nic_in_7
   + 1.2307475912511426e-04 flow_source_nic_in_8
   + 1.2307475912511426e-04 flow_source_nic_in_10
   + 1.2307475912511426e-04 flow_source_nic_in_12
   + 1.2307475912511426e-04 flow_source_nic_in_13
   + 1.2307475912511426e-04 flow_source_nic_in_14
   + 1.2307475912511426e-04 flow_source_nic_in_15
   - 0.0279397930385701 is_active_0 - 0.0279397930385701 is_active_1
   - 0.0279397930385701 is_active_2 - 0.0279397930385701 is_active_3
   - 0.1151458137347131 is_active_4 - 0.1151458137347131 is_active_5
   - 0.1396989651928504 is_active_6 - 0.160865475070555 is_active_7
   - 0.0397930385700847 is_active_8 - 0.0397930385700847 is_active_9
   - 0.0253998118532455 is_active_10 - 0.0253998118532455 is_active_11
   - 0.0296331138287864 is_active_12 - 0.0296331138287864 is_active_13
   - 0.0397930385700847 is_active_14 - 0.0279397930385701 is_active_15
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
   - flow_nic_in_0_1 - flow_nic_in_0_2 - flow_nic_in_0_3
   + flow_nic_out_4_nic_in_0 + flow_nic_out_6_nic_in_0
   + flow_nic_out_7_nic_in_0 + flow_nic_out_8_nic_in_0
   + flow_nic_out_10_nic_in_0 + flow_nic_out_12_nic_in_0
   + flow_nic_out_13_nic_in_0 + flow_nic_out_14_nic_in_0
   + flow_nic_out_15_nic_in_0 = 0
 node_flow_constraint_nic_out_0: - flow_nic_out_0_sink
   - flow_nic_out_0_nic_in_4 - flow_nic_out_0_nic_in_6
   - flow_nic_out_0_nic_in_7 - flow_nic_out_0_nic_in_8
   - flow_nic_out_0_nic_in_10 - flow_nic_out_0_nic_in_12
   - flow_nic_out_0_nic_in_13 - flow_nic_out_0_nic_in_14
   - flow_nic_out_0_nic_in_15 + flow_0_nic_out_0 + flow_1_nic_out_0
   + flow_2_nic_out_0 + flow_3_nic_out_0 = 0
 node_flow_constraint_nic_in_4: flow_nic_out_0_nic_in_4
   + flow_source_nic_in_4 - flow_nic_in_4_4 - flow_nic_in_4_5
   + flow_nic_out_6_nic_in_4 + flow_nic_out_7_nic_in_4
   + flow_nic_out_8_nic_in_4 + flow_nic_out_10_nic_in_4
   + flow_nic_out_12_nic_in_4 + flow_nic_out_13_nic_in_4
   + flow_nic_out_14_nic_in_4 + flow_nic_out_15_nic_in_4 = 0
 node_flow_constraint_nic_out_4: - flow_nic_out_4_sink
   - flow_nic_out_4_nic_in_0 - flow_nic_out_4_nic_in_6
   - flow_nic_out_4_nic_in_7 - flow_nic_out_4_nic_in_8
   - flow_nic_out_4_nic_in_10 - flow_nic_out_4_nic_in_12
   - flow_nic_out_4_nic_in_13 - flow_nic_out_4_nic_in_14
   - flow_nic_out_4_nic_in_15 + flow_4_nic_out_4 + flow_5_nic_out_4 = 0
 node_flow_constraint_nic_in_6: flow_nic_out_0_nic_in_6
   + flow_nic_out_4_nic_in_6 + flow_source_nic_in_6 - flow_nic_in_6_6
   + flow_nic_out_7_nic_in_6 + flow_nic_out_8_nic_in_6
   + flow_nic_out_10_nic_in_6 + flow_nic_out_12_nic_in_6
   + flow_nic_out_13_nic_in_6 + flow_nic_out_14_nic_in_6
   + flow_nic_out_15_nic_in_6 = 0
 node_flow_constraint_nic_out_6: - flow_nic_out_6_sink
   - flow_nic_out_6_nic_in_0 - flow_nic_out_6_nic_in_4
   - flow_nic_out_6_nic_in_7 - flow_nic_out_6_nic_in_8
   - flow_nic_out_6_nic_in_10 - flow_nic_out_6_nic_in_12
   - flow_nic_out_6_nic_in_13 - flow_nic_out_6_nic_in_14
   - flow_nic_out_6_nic_in_15 + flow_6_nic_out_6 = 0
 node_flow_constraint_nic_in_7: flow_nic_out_0_nic_in_7
   + flow_nic_out_4_nic_in_7 + flow_nic_out_6_nic_in_7
   + flow_source_nic_in_7 - flow_nic_in_7_7 + flow_nic_out_8_nic_in_7
   + flow_nic_out_10_nic_in_7 + flow_nic_out_12_nic_in_7
   + flow_nic_out_13_nic_in_7 + flow_nic_out_14_nic_in_7
   + flow_nic_out_15_nic_in_7 = 0
 node_flow_constraint_nic_out_7: - flow_nic_out_7_sink
   - flow_nic_out_7_nic_in_0 - flow_nic_out_7_nic_in_4
   - flow_nic_out_7_nic_in_6 - flow_nic_out_7_nic_in_8
   - flow_nic_out_7_nic_in_10 - flow_nic_out_7_nic_in_12
   - flow_nic_out_7_nic_in_13 - flow_nic_out_7_nic_in_14
   - flow_nic_out_7_nic_in_15 + flow_7_nic_out_7 = 0
 node_flow_constraint_nic_in_8: flow_nic_out_0_nic_in_8
   + flow_nic_out_4_nic_in_8 + flow_nic_out_6_nic_in_8
   + flow_nic_out_7_nic_in_8 + flow_source_nic_in_8 - flow_nic_in_8_8
   - flow_nic_in_8_9 + flow_nic_out_10_nic_in_8 + flow_nic_out_12_nic_in_8
   + flow_nic_out_13_nic_in_8 + flow_nic_out_14_nic_in_8
   + flow_nic_out_15_nic_in_8 = 0
 node_flow_constraint_nic_out_8: - flow_nic_out_8_sink
   - flow_nic_out_8_nic_in_0 - flow_nic_out_8_nic_in_4
   - flow_nic_out_8_nic_in_6 - flow_nic_out_8_nic_in_7
   - flow_nic_out_8_nic_in_10 - flow_nic_out_8_nic_in_12
   - flow_nic_out_8_nic_in_13 - flow_nic_out_8_nic_in_14
   - flow_nic_out_8_nic_in_15 + flow_8_nic_out_8 + flow_9_nic_out_8 = 0
 node_flow_constraint_nic_in_10: flow_nic_out_0_nic_in_10
   + flow_nic_out_4_nic_in_10 + flow_nic_out_6_nic_in_10
   + flow_nic_out_7_nic_in_10 + flow_nic_out_8_nic_in_10
   + flow_source_nic_in_10 - flow_nic_in_10_10 - flow_nic_in_10_11
   + flow_nic_out_12_nic_in_10 + flow_nic_out_13_nic_in_10
   + flow_nic_out_14_nic_in_10 + flow_nic_out_15_nic_in_10 = 0
 node_flow_constraint_nic_out_10: - flow_nic_out_10_sink
   - flow_nic_out_10_nic_in_0 - flow_nic_out_10_nic_in_4
   - flow_nic_out_10_nic_in_6 - flow_nic_out_10_nic_in_7
   - flow_nic_out_10_nic_in_8 - flow_nic_out_10_nic_in_12
   - flow_nic_out_10_nic_in_13 - flow_nic_out_10_nic_in_14
   - flow_nic_out_10_nic_in_15 + flow_10_nic_out_10 + flow_11_nic_out_10
   = 0
 node_flow_constraint_nic_in_12: flow_nic_out_0_nic_in_12
   + flow_nic_out_4_nic_in_12 + flow_nic_out_6_nic_in_12
   + flow_nic_out_7_nic_in_12 + flow_nic_out_8_nic_in_12
   + flow_nic_out_10_nic_in_12 + flow_source_nic_in_12 - flow_nic_in_12_12
   + flow_nic_out_13_nic_in_12 + flow_nic_out_14_nic_in_12
   + flow_nic_out_15_nic_in_12 = 0
 node_flow_constraint_nic_out_12: - flow_nic_out_12_sink
   - flow_nic_out_12_nic_in_0 - flow_nic_out_12_nic_in_4
   - flow_nic_out_12_nic_in_6 - flow_nic_out_12_nic_in_7
   - flow_nic_out_12_nic_in_8 - flow_nic_out_12_nic_in_10
   - flow_nic_out_12_nic_in_13 - flow_nic_out_12_nic_in_14
   - flow_nic_out_12_nic_in_15 + flow_12_nic_out_12 = 0
 node_flow_constraint_nic_in_13: flow_nic_out_0_nic_in_13
   + flow_nic_out_4_nic_in_13 + flow_nic_out_6_nic_in_13
   + flow_nic_out_7_nic_in_13 + flow_nic_out_8_nic_in_13
   + flow_nic_out_10_nic_in_13 + flow_nic_out_12_nic_in_13
   + flow_source_nic_in_13 - flow_nic_in_13_13 + flow_nic_out_14_nic_in_13
   + flow_nic_out_15_nic_in_13 = 0
 node_flow_constraint_nic_out_13: - flow_nic_out_13_sink
   - flow_nic_out_13_nic_in_0 - flow_nic_out_13_nic_in_4
   - flow_nic_out_13_nic_in_6 - flow_nic_out_13_nic_in_7
   - flow_nic_out_13_nic_in_8 - flow_nic_out_13_nic_in_10
   - flow_nic_out_13_nic_in_12 - flow_nic_out_13_nic_in_14
   - flow_nic_out_13_nic_in_15 + flow_13_nic_out_13 = 0
 node_flow_constraint_nic_in_14: flow_nic_out_0_nic_in_14
   + flow_nic_out_4_nic_in_14 + flow_nic_out_6_nic_in_14
   + flow_nic_out_7_nic_in_14 + flow_nic_out_8_nic_in_14
   + flow_nic_out_10_nic_in_14 + flow_nic_out_12_nic_in_14
   + flow_nic_out_13_nic_in_14 + flow_source_nic_in_14 - flow_nic_in_14_14
   + flow_nic_out_15_nic_in_14 = 0
 node_flow_constraint_nic_out_14: - flow_nic_out_14_sink
   - flow_nic_out_14_nic_in_0 - flow_nic_out_14_nic_in_4
   - flow_nic_out_14_nic_in_6 - flow_nic_out_14_nic_in_7
   - flow_nic_out_14_nic_in_8 - flow_nic_out_14_nic_in_10
   - flow_nic_out_14_nic_in_12 - flow_nic_out_14_nic_in_13
   - flow_nic_out_14_nic_in_15 + flow_14_nic_out_14 = 0
 node_flow_constraint_nic_in_15: flow_nic_out_0_nic_in_15
   + flow_nic_out_4_nic_in_15 + flow_nic_out_6_nic_in_15
   + flow_nic_out_7_nic_in_15 + flow_nic_out_8_nic_in_15
   + flow_nic_out_10_nic_in_15 + flow_nic_out_12_nic_in_15
   + flow_nic_out_13_nic_in_15 + flow_nic_out_14_nic_in_15
   + flow_source_nic_in_15 - flow_nic_in_15_15 = 0
 node_flow_constraint_nic_out_15: - flow_nic_out_15_sink
   - flow_nic_out_15_nic_in_0 - flow_nic_out_15_nic_in_4
   - flow_nic_out_15_nic_in_6 - flow_nic_out_15_nic_in_7
   - flow_nic_out_15_nic_in_8 - flow_nic_out_15_nic_in_10
   - flow_nic_out_15_nic_in_12 - flow_nic_out_15_nic_in_13
   - flow_nic_out_15_nic_in_14 + flow_15_nic_out_15 = 0
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
 node_flow_constraint_6: flow_nic_in_6_6 - flow_6_nic_out_6 = 0
 node_flow_constraint_7: flow_nic_in_7_7 - flow_7_nic_out_7 = 0
 node_flow_constraint_8: flow_nic_in_8_8 - flow_8_nic_out_8 - flow_8_9
   + flow_9_8 = 0
 node_flow_constraint_9: flow_nic_in_8_9 - flow_9_nic_out_8 + flow_8_9
   - flow_9_8 = 0
 node_flow_constraint_10: flow_nic_in_10_10 - flow_10_nic_out_10
   - flow_10_11 + flow_11_10 = 0
 node_flow_constraint_11: flow_nic_in_10_11 - flow_11_nic_out_10
   + flow_10_11 - flow_11_10 = 0
 node_flow_constraint_12: flow_nic_in_12_12 - flow_12_nic_out_12 = 0
 node_flow_constraint_13: flow_nic_in_13_13 - flow_13_nic_out_13 = 0
 node_flow_constraint_14: flow_nic_in_14_14 - flow_14_nic_out_14 = 0
 node_flow_constraint_15: flow_nic_in_15_15 - flow_15_nic_out_15 = 0
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
   + 22 hold_0_22 + 23 hold_0_23 + 24 hold_0_24 + 25 hold_0_25
   + 26 hold_0_26 - 80 is_end_complete_0 >= 0
 constr_is_end_complete_1: start_1 + hold_1_1 + 2 hold_1_2 + 3 hold_1_3
   + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7 + 8 hold_1_8
   + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 + 12 hold_1_12 + 13 hold_1_13
   + 14 hold_1_14 + 15 hold_1_15 + 16 hold_1_16 + 17 hold_1_17
   + 18 hold_1_18 + 19 hold_1_19 + 20 hold_1_20 + 21 hold_1_21
   + 22 hold_1_22 + 23 hold_1_23 + 24 hold_1_24 + 25 hold_1_25
   + 26 hold_1_26 - 80 is_end_complete_1 >= 0
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
 constr_link_match_1_0_7: 80 switch_nic_out_0_nic_in_7 + group_end_0
   - group_start_7 <= 80
 constr_link_match_2_0_7: 80 switch_nic_out_0_nic_in_7 - group_end_0
   + group_start_7 <= 80
 constr_link_match_1_0_8: 80 switch_nic_out_0_nic_in_8 + group_end_0
   - group_start_8 <= 80
 constr_link_match_2_0_8: 80 switch_nic_out_0_nic_in_8 - group_end_0
   + group_start_8 <= 80
 constr_link_match_1_0_10: 80 switch_nic_out_0_nic_in_10 + group_end_0
   - group_start_10 <= 80
 constr_link_match_2_0_10: 80 switch_nic_out_0_nic_in_10 - group_end_0
   + group_start_10 <= 80
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
 constr_pp_permissionA_0_2: start_0 - start_2 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 8 hold_0_8 + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12
   + 13 hold_0_13 + 14 hold_0_14 + 15 hold_0_15 + 16 hold_0_16
   + 17 hold_0_17 + 18 hold_0_18 + 19 hold_0_19 + 20 hold_0_20
   + 21 hold_0_21 + 22 hold_0_22 + 23 hold_0_23 + 24 hold_0_24
   + 25 hold_0_25 + 26 hold_0_26 + 80 pp_permission_0_2 <= 80
 constr_pp_permissionB_0_2: - start_0 + start_2 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11 - 12 hold_0_12
   - 13 hold_0_13 - 14 hold_0_14 - 15 hold_0_15 - 16 hold_0_16
   - 17 hold_0_17 - 18 hold_0_18 - 19 hold_0_19 - 20 hold_0_20
   - 21 hold_0_21 - 22 hold_0_22 - 23 hold_0_23 - 24 hold_0_24
   - 25 hold_0_25 - 26 hold_0_26 + 80 pp_permission_0_2 <= 80
 constr_internal_compute_0_2: switch_0_2 - group_tp_active_0
   - pp_permission_0_2 <= 0
 constr_pp_permissionA_0_3: start_0 - start_3 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 8 hold_0_8 + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12
   + 13 hold_0_13 + 14 hold_0_14 + 15 hold_0_15 + 16 hold_0_16
   + 17 hold_0_17 + 18 hold_0_18 + 19 hold_0_19 + 20 hold_0_20
   + 21 hold_0_21 + 22 hold_0_22 + 23 hold_0_23 + 24 hold_0_24
   + 25 hold_0_25 + 26 hold_0_26 + 80 pp_permission_0_3 <= 80
 constr_pp_permissionB_0_3: - start_0 + start_3 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11 - 12 hold_0_12
   - 13 hold_0_13 - 14 hold_0_14 - 15 hold_0_15 - 16 hold_0_16
   - 17 hold_0_17 - 18 hold_0_18 - 19 hold_0_19 - 20 hold_0_20
   - 21 hold_0_21 - 22 hold_0_22 - 23 hold_0_23 - 24 hold_0_24
   - 25 hold_0_25 - 26 hold_0_26 + 80 pp_permission_0_3 <= 80
 constr_internal_compute_0_3: switch_0_3 - group_tp_active_0
   - pp_permission_0_3 <= 0
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
 constr_pp_permissionA_1_2: start_1 - start_2 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 8 hold_1_8 + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 + 12 hold_1_12
   + 13 hold_1_13 + 14 hold_1_14 + 15 hold_1_15 + 16 hold_1_16
   + 17 hold_1_17 + 18 hold_1_18 + 19 hold_1_19 + 20 hold_1_20
   + 21 hold_1_21 + 22 hold_1_22 + 23 hold_1_23 + 24 hold_1_24
   + 25 hold_1_25 + 26 hold_1_26 + 80 pp_permission_1_2 <= 80
 constr_pp_permissionB_1_2: - start_1 + start_2 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11 - 12 hold_1_12
   - 13 hold_1_13 - 14 hold_1_14 - 15 hold_1_15 - 16 hold_1_16
   - 17 hold_1_17 - 18 hold_1_18 - 19 hold_1_19 - 20 hold_1_20
   - 21 hold_1_21 - 22 hold_1_22 - 23 hold_1_23 - 24 hold_1_24
   - 25 hold_1_25 - 26 hold_1_26 + 80 pp_permission_1_2 <= 80
 constr_internal_compute_1_2: switch_1_2 - group_tp_active_0
   - pp_permission_1_2 <= 0
 constr_pp_permissionA_1_3: start_1 - start_3 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 8 hold_1_8 + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 + 12 hold_1_12
   + 13 hold_1_13 + 14 hold_1_14 + 15 hold_1_15 + 16 hold_1_16
   + 17 hold_1_17 + 18 hold_1_18 + 19 hold_1_19 + 20 hold_1_20
   + 21 hold_1_21 + 22 hold_1_22 + 23 hold_1_23 + 24 hold_1_24
   + 25 hold_1_25 + 26 hold_1_26 + 80 pp_permission_1_3 <= 80
 constr_pp_permissionB_1_3: - start_1 + start_3 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11 - 12 hold_1_12
   - 13 hold_1_13 - 14 hold_1_14 - 15 hold_1_15 - 16 hold_1_16
   - 17 hold_1_17 - 18 hold_1_18 - 19 hold_1_19 - 20 hold_1_20
   - 21 hold_1_21 - 22 hold_1_22 - 23 hold_1_23 - 24 hold_1_24
   - 25 hold_1_25 - 26 hold_1_26 + 80 pp_permission_1_3 <= 80
 constr_internal_compute_1_3: switch_1_3 - group_tp_active_0
   - pp_permission_1_3 <= 0
 constr_pp_permissionA_2_0: - start_0 + start_2 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 12 hold_2_12
   + 13 hold_2_13 + 14 hold_2_14 + 15 hold_2_15 + 16 hold_2_16
   + 17 hold_2_17 + 18 hold_2_18 + 19 hold_2_19 + 20 hold_2_20
   + 21 hold_2_21 + 22 hold_2_22 + 23 hold_2_23 + 24 hold_2_24
   + 25 hold_2_25 + 26 hold_2_26 + 80 pp_permission_2_0 <= 80
 constr_pp_permissionB_2_0: start_0 - start_2 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 - 12 hold_2_12
   - 13 hold_2_13 - 14 hold_2_14 - 15 hold_2_15 - 16 hold_2_16
   - 17 hold_2_17 - 18 hold_2_18 - 19 hold_2_19 - 20 hold_2_20
   - 21 hold_2_21 - 22 hold_2_22 - 23 hold_2_23 - 24 hold_2_24
   - 25 hold_2_25 - 26 hold_2_26 + 80 pp_permission_2_0 <= 80
 constr_internal_compute_2_0: switch_2_0 - group_tp_active_0
   - pp_permission_2_0 <= 0
 constr_pp_permissionA_2_1: - start_1 + start_2 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 12 hold_2_12
   + 13 hold_2_13 + 14 hold_2_14 + 15 hold_2_15 + 16 hold_2_16
   + 17 hold_2_17 + 18 hold_2_18 + 19 hold_2_19 + 20 hold_2_20
   + 21 hold_2_21 + 22 hold_2_22 + 23 hold_2_23 + 24 hold_2_24
   + 25 hold_2_25 + 26 hold_2_26 + 80 pp_permission_2_1 <= 80
 constr_pp_permissionB_2_1: start_1 - start_2 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 - 12 hold_2_12
   - 13 hold_2_13 - 14 hold_2_14 - 15 hold_2_15 - 16 hold_2_16
   - 17 hold_2_17 - 18 hold_2_18 - 19 hold_2_19 - 20 hold_2_20
   - 21 hold_2_21 - 22 hold_2_22 - 23 hold_2_23 - 24 hold_2_24
   - 25 hold_2_25 - 26 hold_2_26 + 80 pp_permission_2_1 <= 80
 constr_internal_compute_2_1: switch_2_1 - group_tp_active_0
   - pp_permission_2_1 <= 0
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
 constr_internal_compute_2_3: switch_2_3 - group_tp_active_0
   - pp_permission_2_3 <= 0
 constr_pp_permissionA_3_0: - start_0 + start_3 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 8 hold_3_8 + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11 + 12 hold_3_12
   + 13 hold_3_13 + 14 hold_3_14 + 15 hold_3_15 + 16 hold_3_16
   + 17 hold_3_17 + 18 hold_3_18 + 19 hold_3_19 + 20 hold_3_20
   + 21 hold_3_21 + 22 hold_3_22 + 23 hold_3_23 + 24 hold_3_24
   + 25 hold_3_25 + 26 hold_3_26 + 80 pp_permission_3_0 <= 80
 constr_pp_permissionB_3_0: start_0 - start_3 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   - 8 hold_3_8 - 9 hold_3_9 - 10 hold_3_10 - 11 hold_3_11 - 12 hold_3_12
   - 13 hold_3_13 - 14 hold_3_14 - 15 hold_3_15 - 16 hold_3_16
   - 17 hold_3_17 - 18 hold_3_18 - 19 hold_3_19 - 20 hold_3_20
   - 21 hold_3_21 - 22 hold_3_22 - 23 hold_3_23 - 24 hold_3_24
   - 25 hold_3_25 - 26 hold_3_26 + 80 pp_permission_3_0 <= 80
 constr_internal_compute_3_0: switch_3_0 - group_tp_active_0
   - pp_permission_3_0 <= 0
 constr_pp_permissionA_3_1: - start_1 + start_3 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 8 hold_3_8 + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11 + 12 hold_3_12
   + 13 hold_3_13 + 14 hold_3_14 + 15 hold_3_15 + 16 hold_3_16
   + 17 hold_3_17 + 18 hold_3_18 + 19 hold_3_19 + 20 hold_3_20
   + 21 hold_3_21 + 22 hold_3_22 + 23 hold_3_23 + 24 hold_3_24
   + 25 hold_3_25 + 26 hold_3_26 + 80 pp_permission_3_1 <= 80
 constr_pp_permissionB_3_1: start_1 - start_3 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   - 8 hold_3_8 - 9 hold_3_9 - 10 hold_3_10 - 11 hold_3_11 - 12 hold_3_12
   - 13 hold_3_13 - 14 hold_3_14 - 15 hold_3_15 - 16 hold_3_16
   - 17 hold_3_17 - 18 hold_3_18 - 19 hold_3_19 - 20 hold_3_20
   - 21 hold_3_21 - 22 hold_3_22 - 23 hold_3_23 - 24 hold_3_24
   - 25 hold_3_25 - 26 hold_3_26 + 80 pp_permission_3_1 <= 80
 constr_internal_compute_3_1: switch_3_1 - group_tp_active_0
   - pp_permission_3_1 <= 0
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
   + 22 hold_4_22 + 23 hold_4_23 + 24 hold_4_24 + 25 hold_4_25
   + 26 hold_4_26 - 80 is_end_complete_4 >= 0
 constr_is_end_complete_5: start_5 + hold_5_1 + 2 hold_5_2 + 3 hold_5_3
   + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7 + 8 hold_5_8
   + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12 + 13 hold_5_13
   + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16 + 17 hold_5_17
   + 18 hold_5_18 + 19 hold_5_19 + 20 hold_5_20 + 21 hold_5_21
   + 22 hold_5_22 + 23 hold_5_23 + 24 hold_5_24 + 25 hold_5_25
   + 26 hold_5_26 - 80 is_end_complete_5 >= 0
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
 constr_link_match_1_4_7: 80 switch_nic_out_4_nic_in_7 - group_start_7
   + group_end_4 <= 80
 constr_link_match_2_4_7: 80 switch_nic_out_4_nic_in_7 + group_start_7
   - group_end_4 <= 80
 constr_link_match_1_4_8: 80 switch_nic_out_4_nic_in_8 - group_start_8
   + group_end_4 <= 80
 constr_link_match_2_4_8: 80 switch_nic_out_4_nic_in_8 + group_start_8
   - group_end_4 <= 80
 constr_link_match_1_4_10: 80 switch_nic_out_4_nic_in_10 - group_start_10
   + group_end_4 <= 80
 constr_link_match_2_4_10: 80 switch_nic_out_4_nic_in_10 + group_start_10
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
 constr_pp_permissionA_4_5: start_4 - start_5 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 8 hold_4_8 + 9 hold_4_9 + 10 hold_4_10 + 11 hold_4_11 + 12 hold_4_12
   + 13 hold_4_13 + 14 hold_4_14 + 15 hold_4_15 + 16 hold_4_16
   + 17 hold_4_17 + 18 hold_4_18 + 19 hold_4_19 + 20 hold_4_20
   + 21 hold_4_21 + 22 hold_4_22 + 23 hold_4_23 + 24 hold_4_24
   + 25 hold_4_25 + 26 hold_4_26 + 80 pp_permission_4_5 <= 80
 constr_pp_permissionB_4_5: - start_4 + start_5 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   - 8 hold_4_8 - 9 hold_4_9 - 10 hold_4_10 - 11 hold_4_11 - 12 hold_4_12
   - 13 hold_4_13 - 14 hold_4_14 - 15 hold_4_15 - 16 hold_4_16
   - 17 hold_4_17 - 18 hold_4_18 - 19 hold_4_19 - 20 hold_4_20
   - 21 hold_4_21 - 22 hold_4_22 - 23 hold_4_23 - 24 hold_4_24
   - 25 hold_4_25 - 26 hold_4_26 + 80 pp_permission_4_5 <= 80
 constr_internal_compute_4_5: switch_4_5 - group_tp_active_4
   - pp_permission_4_5 <= 0
 constr_pp_permissionA_5_4: - start_4 + start_5 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12
   + 13 hold_5_13 + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16
   + 17 hold_5_17 + 18 hold_5_18 + 19 hold_5_19 + 20 hold_5_20
   + 21 hold_5_21 + 22 hold_5_22 + 23 hold_5_23 + 24 hold_5_24
   + 25 hold_5_25 + 26 hold_5_26 + 80 pp_permission_5_4 <= 80
 constr_pp_permissionB_5_4: start_4 - start_5 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11 - 12 hold_5_12
   - 13 hold_5_13 - 14 hold_5_14 - 15 hold_5_15 - 16 hold_5_16
   - 17 hold_5_17 - 18 hold_5_18 - 19 hold_5_19 - 20 hold_5_20
   - 21 hold_5_21 - 22 hold_5_22 - 23 hold_5_23 - 24 hold_5_24
   - 25 hold_5_25 - 26 hold_5_26 + 80 pp_permission_5_4 <= 80
 constr_internal_compute_5_4: switch_5_4 - group_tp_active_4
   - pp_permission_5_4 <= 0
 constr_is_start_zero_6: start_6 + 80 is_start_zero_6_for_source_nic_in_6
   <= 80
 constr_edge_switch_start_zero_6_OR: switch_source_nic_in_6
   - is_start_zero_6_for_source_nic_in_6 <= 0
 constr_is_end_complete_6: start_6 + hold_6_1 + 2 hold_6_2 + 3 hold_6_3
   + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7 + 8 hold_6_8
   + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 12 hold_6_12 + 13 hold_6_13
   + 14 hold_6_14 + 15 hold_6_15 + 16 hold_6_16 + 17 hold_6_17
   + 18 hold_6_18 + 19 hold_6_19 + 20 hold_6_20 + 21 hold_6_21
   + 22 hold_6_22 + 23 hold_6_23 + 24 hold_6_24 + 25 hold_6_25
   + 26 hold_6_26 - 80 is_end_complete_6 >= 0
 constr_edge_switch_end_max_6_OR: switch_nic_out_6_sink - is_end_complete_6
   <= 0
 constr_link_match_1_6_0: 80 switch_nic_out_6_nic_in_0 - group_start_0
   + group_end_6 <= 80
 constr_link_match_2_6_0: 80 switch_nic_out_6_nic_in_0 + group_start_0
   - group_end_6 <= 80
 constr_link_match_1_6_4: 80 switch_nic_out_6_nic_in_4 - group_start_4
   + group_end_6 <= 80
 constr_link_match_2_6_4: 80 switch_nic_out_6_nic_in_4 + group_start_4
   - group_end_6 <= 80
 constr_link_match_1_6_7: 80 switch_nic_out_6_nic_in_7 - group_start_7
   + group_end_6 <= 80
 constr_link_match_2_6_7: 80 switch_nic_out_6_nic_in_7 + group_start_7
   - group_end_6 <= 80
 constr_link_match_1_6_8: 80 switch_nic_out_6_nic_in_8 - group_start_8
   + group_end_6 <= 80
 constr_link_match_2_6_8: 80 switch_nic_out_6_nic_in_8 + group_start_8
   - group_end_6 <= 80
 constr_link_match_1_6_10: 80 switch_nic_out_6_nic_in_10 - group_start_10
   + group_end_6 <= 80
 constr_link_match_2_6_10: 80 switch_nic_out_6_nic_in_10 + group_start_10
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
 constr_is_start_zero_7: start_7 + 80 is_start_zero_7_for_source_nic_in_7
   <= 80
 constr_edge_switch_start_zero_7_OR: switch_source_nic_in_7
   - is_start_zero_7_for_source_nic_in_7 <= 0
 constr_is_end_complete_7: start_7 + hold_7_1 + 2 hold_7_2 + 3 hold_7_3
   + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7 + 8 hold_7_8
   + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11 + 12 hold_7_12 + 13 hold_7_13
   + 14 hold_7_14 + 15 hold_7_15 + 16 hold_7_16 + 17 hold_7_17
   + 18 hold_7_18 + 19 hold_7_19 + 20 hold_7_20 + 21 hold_7_21
   + 22 hold_7_22 + 23 hold_7_23 + 24 hold_7_24 + 25 hold_7_25
   + 26 hold_7_26 - 80 is_end_complete_7 >= 0
 constr_edge_switch_end_max_7_OR: switch_nic_out_7_sink - is_end_complete_7
   <= 0
 constr_link_match_1_7_0: 80 switch_nic_out_7_nic_in_0 - group_start_0
   + group_end_7 <= 80
 constr_link_match_2_7_0: 80 switch_nic_out_7_nic_in_0 + group_start_0
   - group_end_7 <= 80
 constr_link_match_1_7_4: 80 switch_nic_out_7_nic_in_4 - group_start_4
   + group_end_7 <= 80
 constr_link_match_2_7_4: 80 switch_nic_out_7_nic_in_4 + group_start_4
   - group_end_7 <= 80
 constr_link_match_1_7_6: 80 switch_nic_out_7_nic_in_6 - group_start_6
   + group_end_7 <= 80
 constr_link_match_2_7_6: 80 switch_nic_out_7_nic_in_6 + group_start_6
   - group_end_7 <= 80
 constr_link_match_1_7_8: 80 switch_nic_out_7_nic_in_8 - group_start_8
   + group_end_7 <= 80
 constr_link_match_2_7_8: 80 switch_nic_out_7_nic_in_8 + group_start_8
   - group_end_7 <= 80
 constr_link_match_1_7_10: 80 switch_nic_out_7_nic_in_10 - group_start_10
   + group_end_7 <= 80
 constr_link_match_2_7_10: 80 switch_nic_out_7_nic_in_10 + group_start_10
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
 constr_is_start_zero_8: start_8 + 80 is_start_zero_8_for_source_nic_in_8
   <= 80
 constr_is_start_zero_9: start_9 + 80 is_start_zero_9_for_source_nic_in_8
   <= 80
 constr_edge_switch_start_zero_8_OR: switch_source_nic_in_8
   - is_start_zero_8_for_source_nic_in_8
   - is_start_zero_9_for_source_nic_in_8 <= 0
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
 constr_edge_switch_end_max_8_OR: switch_nic_out_8_sink - is_end_complete_8
   - is_end_complete_9 <= 0
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
 constr_link_match_1_8_7: 80 switch_nic_out_8_nic_in_7 - group_start_7
   + group_end_8 <= 80
 constr_link_match_2_8_7: 80 switch_nic_out_8_nic_in_7 + group_start_7
   - group_end_8 <= 80
 constr_link_match_1_8_10: 80 switch_nic_out_8_nic_in_10 - group_start_10
   + group_end_8 <= 80
 constr_link_match_2_8_10: 80 switch_nic_out_8_nic_in_10 + group_start_10
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
 constr_internal_compute_8_9: switch_8_9 - group_tp_active_8
   - pp_permission_8_9 <= 0
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
 constr_internal_compute_9_8: switch_9_8 - group_tp_active_8
   - pp_permission_9_8 <= 0
 constr_is_start_zero_10: start_10
   + 80 is_start_zero_10_for_source_nic_in_10 <= 80
 constr_is_start_zero_11: start_11
   + 80 is_start_zero_11_for_source_nic_in_10 <= 80
 constr_edge_switch_start_zero_10_OR: switch_source_nic_in_10
   - is_start_zero_10_for_source_nic_in_10
   - is_start_zero_11_for_source_nic_in_10 <= 0
 constr_is_end_complete_10: start_10 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 12 hold_10_12 + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15
   + 16 hold_10_16 + 17 hold_10_17 + 18 hold_10_18 + 19 hold_10_19
   + 20 hold_10_20 + 21 hold_10_21 + 22 hold_10_22 + 23 hold_10_23
   + 24 hold_10_24 + 25 hold_10_25 + 26 hold_10_26 - 80 is_end_complete_10
   >= 0
 constr_is_end_complete_11: start_11 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 12 hold_11_12 + 13 hold_11_13 + 14 hold_11_14 + 15 hold_11_15
   + 16 hold_11_16 + 17 hold_11_17 + 18 hold_11_18 + 19 hold_11_19
   + 20 hold_11_20 + 21 hold_11_21 + 22 hold_11_22 + 23 hold_11_23
   + 24 hold_11_24 + 25 hold_11_25 + 26 hold_11_26 - 80 is_end_complete_11
   >= 0
 constr_edge_switch_end_max_10_OR: switch_nic_out_10_sink
   - is_end_complete_10 - is_end_complete_11 <= 0
 constr_link_match_1_10_0: 80 switch_nic_out_10_nic_in_0 - group_start_0
   + group_end_10 <= 80
 constr_link_match_2_10_0: 80 switch_nic_out_10_nic_in_0 + group_start_0
   - group_end_10 <= 80
 constr_link_match_1_10_4: 80 switch_nic_out_10_nic_in_4 - group_start_4
   + group_end_10 <= 80
 constr_link_match_2_10_4: 80 switch_nic_out_10_nic_in_4 + group_start_4
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
 constr_pp_permissionA_10_11: start_10 - start_11 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 12 hold_10_12 + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15
   + 16 hold_10_16 + 17 hold_10_17 + 18 hold_10_18 + 19 hold_10_19
   + 20 hold_10_20 + 21 hold_10_21 + 22 hold_10_22 + 23 hold_10_23
   + 24 hold_10_24 + 25 hold_10_25 + 26 hold_10_26 + 80 pp_permission_10_11
   <= 80
 constr_pp_permissionB_10_11: - start_10 + start_11 - hold_10_1
   - 2 hold_10_2 - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6
   - 7 hold_10_7 - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10
   - 11 hold_10_11 - 12 hold_10_12 - 13 hold_10_13 - 14 hold_10_14
   - 15 hold_10_15 - 16 hold_10_16 - 17 hold_10_17 - 18 hold_10_18
   - 19 hold_10_19 - 20 hold_10_20 - 21 hold_10_21 - 22 hold_10_22
   - 23 hold_10_23 - 24 hold_10_24 - 25 hold_10_25 - 26 hold_10_26
   + 80 pp_permission_10_11 <= 80
 constr_internal_compute_10_11: switch_10_11 - group_tp_active_10
   - pp_permission_10_11 <= 0
 constr_pp_permissionA_11_10: - start_10 + start_11 + hold_11_1
   + 2 hold_11_2 + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6
   + 7 hold_11_7 + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10
   + 11 hold_11_11 + 12 hold_11_12 + 13 hold_11_13 + 14 hold_11_14
   + 15 hold_11_15 + 16 hold_11_16 + 17 hold_11_17 + 18 hold_11_18
   + 19 hold_11_19 + 20 hold_11_20 + 21 hold_11_21 + 22 hold_11_22
   + 23 hold_11_23 + 24 hold_11_24 + 25 hold_11_25 + 26 hold_11_26
   + 80 pp_permission_11_10 <= 80
 constr_pp_permissionB_11_10: start_10 - start_11 - hold_11_1 - 2 hold_11_2
   - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6 - 7 hold_11_7
   - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10 - 11 hold_11_11
   - 12 hold_11_12 - 13 hold_11_13 - 14 hold_11_14 - 15 hold_11_15
   - 16 hold_11_16 - 17 hold_11_17 - 18 hold_11_18 - 19 hold_11_19
   - 20 hold_11_20 - 21 hold_11_21 - 22 hold_11_22 - 23 hold_11_23
   - 24 hold_11_24 - 25 hold_11_25 - 26 hold_11_26 + 80 pp_permission_11_10
   <= 80
 constr_internal_compute_11_10: switch_11_10 - group_tp_active_10
   - pp_permission_11_10 <= 0
 constr_is_start_zero_12: start_12
   + 80 is_start_zero_12_for_source_nic_in_12 <= 80
 constr_edge_switch_start_zero_12_OR: switch_source_nic_in_12
   - is_start_zero_12_for_source_nic_in_12 <= 0
 constr_is_end_complete_12: start_12 + hold_12_1 + 2 hold_12_2
   + 3 hold_12_3 + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6 + 7 hold_12_7
   + 8 hold_12_8 + 9 hold_12_9 + 10 hold_12_10 + 11 hold_12_11
   + 12 hold_12_12 + 13 hold_12_13 + 14 hold_12_14 + 15 hold_12_15
   + 16 hold_12_16 + 17 hold_12_17 + 18 hold_12_18 + 19 hold_12_19
   + 20 hold_12_20 + 21 hold_12_21 + 22 hold_12_22 + 23 hold_12_23
   + 24 hold_12_24 + 25 hold_12_25 + 26 hold_12_26 - 80 is_end_complete_12
   >= 0
 constr_edge_switch_end_max_12_OR: switch_nic_out_12_sink
   - is_end_complete_12 <= 0
 constr_link_match_1_12_0: 80 switch_nic_out_12_nic_in_0 - group_start_0
   + group_end_12 <= 80
 constr_link_match_2_12_0: 80 switch_nic_out_12_nic_in_0 + group_start_0
   - group_end_12 <= 80
 constr_link_match_1_12_4: 80 switch_nic_out_12_nic_in_4 - group_start_4
   + group_end_12 <= 80
 constr_link_match_2_12_4: 80 switch_nic_out_12_nic_in_4 + group_start_4
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
 constr_link_match_1_12_10: 80 switch_nic_out_12_nic_in_10 - group_start_10
   + group_end_12 <= 80
 constr_link_match_2_12_10: 80 switch_nic_out_12_nic_in_10 + group_start_10
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
 constr_is_start_zero_13: start_13
   + 80 is_start_zero_13_for_source_nic_in_13 <= 80
 constr_edge_switch_start_zero_13_OR: switch_source_nic_in_13
   - is_start_zero_13_for_source_nic_in_13 <= 0
 constr_is_end_complete_13: start_13 + hold_13_1 + 2 hold_13_2
   + 3 hold_13_3 + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6 + 7 hold_13_7
   + 8 hold_13_8 + 9 hold_13_9 + 10 hold_13_10 + 11 hold_13_11
   + 12 hold_13_12 + 13 hold_13_13 + 14 hold_13_14 + 15 hold_13_15
   + 16 hold_13_16 + 17 hold_13_17 + 18 hold_13_18 + 19 hold_13_19
   + 20 hold_13_20 + 21 hold_13_21 + 22 hold_13_22 + 23 hold_13_23
   + 24 hold_13_24 + 25 hold_13_25 + 26 hold_13_26 - 80 is_end_complete_13
   >= 0
 constr_edge_switch_end_max_13_OR: switch_nic_out_13_sink
   - is_end_complete_13 <= 0
 constr_link_match_1_13_0: 80 switch_nic_out_13_nic_in_0 - group_start_0
   + group_end_13 <= 80
 constr_link_match_2_13_0: 80 switch_nic_out_13_nic_in_0 + group_start_0
   - group_end_13 <= 80
 constr_link_match_1_13_4: 80 switch_nic_out_13_nic_in_4 - group_start_4
   + group_end_13 <= 80
 constr_link_match_2_13_4: 80 switch_nic_out_13_nic_in_4 + group_start_4
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
 constr_link_match_1_13_10: 80 switch_nic_out_13_nic_in_10 - group_start_10
   + group_end_13 <= 80
 constr_link_match_2_13_10: 80 switch_nic_out_13_nic_in_10 + group_start_10
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
 constr_link_match_1_14_4: 80 switch_nic_out_14_nic_in_4 - group_start_4
   + group_end_14 <= 80
 constr_link_match_2_14_4: 80 switch_nic_out_14_nic_in_4 + group_start_4
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
 constr_link_match_1_14_10: 80 switch_nic_out_14_nic_in_10 - group_start_10
   + group_end_14 <= 80
 constr_link_match_2_14_10: 80 switch_nic_out_14_nic_in_10 + group_start_10
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
 constr_link_match_1_15_4: 80 switch_nic_out_15_nic_in_4 - group_start_4
   + group_end_15 <= 80
 constr_link_match_2_15_4: 80 switch_nic_out_15_nic_in_4 + group_start_4
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
 constr_link_match_1_15_10: 80 switch_nic_out_15_nic_in_10 - group_start_10
   + group_end_15 <= 80
 constr_link_match_2_15_10: 80 switch_nic_out_15_nic_in_10 + group_start_10
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
 link_activation_dst_nic_in_7_7: switch_nic_in_7_7 - is_active_7 <= 0
 link_activation_src_7_nic_out_7: switch_7_nic_out_7 - is_active_7 <= 0
 link_activation_dst_nic_in_8_8: switch_nic_in_8_8 - is_active_8 <= 0
 link_activation_src_8_nic_out_8: switch_8_nic_out_8 - is_active_8 <= 0
 link_activation_dst_nic_in_8_9: switch_nic_in_8_9 - is_active_9 <= 0
 link_activation_src_9_nic_out_8: switch_9_nic_out_8 - is_active_9 <= 0
 link_activation_src_8_9: switch_8_9 - is_active_8 <= 0
 link_activation_dst_8_9: switch_8_9 - is_active_9 <= 0
 link_activation_src_9_8: switch_9_8 - is_active_9 <= 0
 link_activation_dst_9_8: switch_9_8 - is_active_8 <= 0
 link_activation_dst_nic_in_10_10: switch_nic_in_10_10 - is_active_10 <= 0
 link_activation_src_10_nic_out_10: switch_10_nic_out_10 - is_active_10
   <= 0
 link_activation_dst_nic_in_10_11: switch_nic_in_10_11 - is_active_11 <= 0
 link_activation_src_11_nic_out_10: switch_11_nic_out_10 - is_active_11
   <= 0
 link_activation_src_10_11: switch_10_11 - is_active_10 <= 0
 link_activation_dst_10_11: switch_10_11 - is_active_11 <= 0
 link_activation_src_11_10: switch_11_10 - is_active_11 <= 0
 link_activation_dst_11_10: switch_11_10 - is_active_10 <= 0
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
 edge_flow_constr_nic_out_0_nic_in_4: flow_nic_out_0_nic_in_4
   - 61035.15625 switch_nic_out_0_nic_in_4 <= 0
 edge_flow_constr_nic_out_0_nic_in_6: flow_nic_out_0_nic_in_6
   - 61035.15625 switch_nic_out_0_nic_in_6 <= 0
 edge_flow_constr_nic_out_0_nic_in_7: flow_nic_out_0_nic_in_7
   - 61035.15625 switch_nic_out_0_nic_in_7 <= 0
 edge_flow_constr_nic_out_0_nic_in_8: flow_nic_out_0_nic_in_8
   - 61035.15625 switch_nic_out_0_nic_in_8 <= 0
 edge_flow_constr_nic_out_0_nic_in_10: flow_nic_out_0_nic_in_10
   - 61035.15625 switch_nic_out_0_nic_in_10 <= 0
 edge_flow_constr_nic_out_0_nic_in_12: flow_nic_out_0_nic_in_12
   - 61035.15625 switch_nic_out_0_nic_in_12 <= 0
 edge_flow_constr_nic_out_0_nic_in_13: flow_nic_out_0_nic_in_13
   - 61035.15625 switch_nic_out_0_nic_in_13 <= 0
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
 edge_flow_constr_nic_out_4_nic_in_7: flow_nic_out_4_nic_in_7
   - 61035.15625 switch_nic_out_4_nic_in_7 <= 0
 edge_flow_constr_nic_out_4_nic_in_8: flow_nic_out_4_nic_in_8
   - 61035.15625 switch_nic_out_4_nic_in_8 <= 0
 edge_flow_constr_nic_out_4_nic_in_10: flow_nic_out_4_nic_in_10
   - 61035.15625 switch_nic_out_4_nic_in_10 <= 0
 edge_flow_constr_nic_out_4_nic_in_12: flow_nic_out_4_nic_in_12
   - 61035.15625 switch_nic_out_4_nic_in_12 <= 0
 edge_flow_constr_nic_out_4_nic_in_13: flow_nic_out_4_nic_in_13
   - 61035.15625 switch_nic_out_4_nic_in_13 <= 0
 edge_flow_constr_nic_out_4_nic_in_14: flow_nic_out_4_nic_in_14
   - 61035.15625 switch_nic_out_4_nic_in_14 <= 0
 edge_flow_constr_nic_out_4_nic_in_15: flow_nic_out_4_nic_in_15
   - 61035.15625 switch_nic_out_4_nic_in_15 <= 0
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
 edge_flow_constr_nic_out_6_nic_in_7: flow_nic_out_6_nic_in_7
   - 61035.15625 switch_nic_out_6_nic_in_7 <= 0
 edge_flow_constr_nic_out_6_nic_in_8: flow_nic_out_6_nic_in_8
   - 61035.15625 switch_nic_out_6_nic_in_8 <= 0
 edge_flow_constr_nic_out_6_nic_in_10: flow_nic_out_6_nic_in_10
   - 61035.15625 switch_nic_out_6_nic_in_10 <= 0
 edge_flow_constr_nic_out_6_nic_in_12: flow_nic_out_6_nic_in_12
   - 61035.15625 switch_nic_out_6_nic_in_12 <= 0
 edge_flow_constr_nic_out_6_nic_in_13: flow_nic_out_6_nic_in_13
   - 61035.15625 switch_nic_out_6_nic_in_13 <= 0
 edge_flow_constr_nic_out_6_nic_in_14: flow_nic_out_6_nic_in_14
   - 61035.15625 switch_nic_out_6_nic_in_14 <= 0
 edge_flow_constr_nic_out_6_nic_in_15: flow_nic_out_6_nic_in_15
   - 61035.15625 switch_nic_out_6_nic_in_15 <= 0
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
 edge_flow_constr_nic_out_7_nic_in_4: flow_nic_out_7_nic_in_4
   - 61035.15625 switch_nic_out_7_nic_in_4 <= 0
 edge_flow_constr_nic_out_7_nic_in_6: flow_nic_out_7_nic_in_6
   - 61035.15625 switch_nic_out_7_nic_in_6 <= 0
 edge_flow_constr_nic_out_7_nic_in_8: flow_nic_out_7_nic_in_8
   - 61035.15625 switch_nic_out_7_nic_in_8 <= 0
 edge_flow_constr_nic_out_7_nic_in_10: flow_nic_out_7_nic_in_10
   - 61035.15625 switch_nic_out_7_nic_in_10 <= 0
 edge_flow_constr_nic_out_7_nic_in_12: flow_nic_out_7_nic_in_12
   - 61035.15625 switch_nic_out_7_nic_in_12 <= 0
 edge_flow_constr_nic_out_7_nic_in_13: flow_nic_out_7_nic_in_13
   - 61035.15625 switch_nic_out_7_nic_in_13 <= 0
 edge_flow_constr_nic_out_7_nic_in_14: flow_nic_out_7_nic_in_14
   - 61035.15625 switch_nic_out_7_nic_in_14 <= 0
 edge_flow_constr_nic_out_7_nic_in_15: flow_nic_out_7_nic_in_15
   - 61035.15625 switch_nic_out_7_nic_in_15 <= 0
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
 edge_flow_constr_nic_out_8_nic_in_4: flow_nic_out_8_nic_in_4
   - 61035.15625 switch_nic_out_8_nic_in_4 <= 0
 edge_flow_constr_nic_out_8_nic_in_6: flow_nic_out_8_nic_in_6
   - 61035.15625 switch_nic_out_8_nic_in_6 <= 0
 edge_flow_constr_nic_out_8_nic_in_7: flow_nic_out_8_nic_in_7
   - 61035.15625 switch_nic_out_8_nic_in_7 <= 0
 edge_flow_constr_nic_out_8_nic_in_10: flow_nic_out_8_nic_in_10
   - 61035.15625 switch_nic_out_8_nic_in_10 <= 0
 edge_flow_constr_nic_out_8_nic_in_12: flow_nic_out_8_nic_in_12
   - 61035.15625 switch_nic_out_8_nic_in_12 <= 0
 edge_flow_constr_nic_out_8_nic_in_13: flow_nic_out_8_nic_in_13
   - 61035.15625 switch_nic_out_8_nic_in_13 <= 0
 edge_flow_constr_nic_out_8_nic_in_14: flow_nic_out_8_nic_in_14
   - 61035.15625 switch_nic_out_8_nic_in_14 <= 0
 edge_flow_constr_nic_out_8_nic_in_15: flow_nic_out_8_nic_in_15
   - 61035.15625 switch_nic_out_8_nic_in_15 <= 0
 edge_flow_constr_nic_in_8_8: flow_nic_in_8_8
   - 3.90625e+06 switch_nic_in_8_8 <= 0
 edge_flow_constr_8_nic_out_8: flow_8_nic_out_8
   - 3.90625e+06 switch_8_nic_out_8 <= 0
 edge_flow_constr_nic_in_8_9: flow_nic_in_8_9
   - 3.90625e+06 switch_nic_in_8_9 <= 0
 edge_flow_constr_9_nic_out_8: flow_9_nic_out_8
   - 3.90625e+06 switch_9_nic_out_8 <= 0
 edge_flow_constr_8_9: flow_8_9 - 3.90625e+06 switch_8_9 <= 0
 edge_flow_constr_9_8: flow_9_8 - 3.90625e+06 switch_9_8 <= 0
 edge_flow_constr_source_nic_in_10: flow_source_nic_in_10
   - 5e+08 switch_source_nic_in_10 <= 0
 edge_flow_constr_nic_out_10_sink: flow_nic_out_10_sink
   - 5e+08 switch_nic_out_10_sink <= 0
 edge_flow_constr_nic_out_10_nic_in_0: flow_nic_out_10_nic_in_0
   - 61035.15625 switch_nic_out_10_nic_in_0 <= 0
 edge_flow_constr_nic_out_10_nic_in_4: flow_nic_out_10_nic_in_4
   - 61035.15625 switch_nic_out_10_nic_in_4 <= 0
 edge_flow_constr_nic_out_10_nic_in_6: flow_nic_out_10_nic_in_6
   - 61035.15625 switch_nic_out_10_nic_in_6 <= 0
 edge_flow_constr_nic_out_10_nic_in_7: flow_nic_out_10_nic_in_7
   - 61035.15625 switch_nic_out_10_nic_in_7 <= 0
 edge_flow_constr_nic_out_10_nic_in_8: flow_nic_out_10_nic_in_8
   - 61035.15625 switch_nic_out_10_nic_in_8 <= 0
 edge_flow_constr_nic_out_10_nic_in_12: flow_nic_out_10_nic_in_12
   - 61035.15625 switch_nic_out_10_nic_in_12 <= 0
 edge_flow_constr_nic_out_10_nic_in_13: flow_nic_out_10_nic_in_13
   - 61035.15625 switch_nic_out_10_nic_in_13 <= 0
 edge_flow_constr_nic_out_10_nic_in_14: flow_nic_out_10_nic_in_14
   - 61035.15625 switch_nic_out_10_nic_in_14 <= 0
 edge_flow_constr_nic_out_10_nic_in_15: flow_nic_out_10_nic_in_15
   - 61035.15625 switch_nic_out_10_nic_in_15 <= 0
 edge_flow_constr_nic_in_10_10: flow_nic_in_10_10
   - 3.90625e+06 switch_nic_in_10_10 <= 0
 edge_flow_constr_10_nic_out_10: flow_10_nic_out_10
   - 3.90625e+06 switch_10_nic_out_10 <= 0
 edge_flow_constr_nic_in_10_11: flow_nic_in_10_11
   - 3.90625e+06 switch_nic_in_10_11 <= 0
 edge_flow_constr_11_nic_out_10: flow_11_nic_out_10
   - 3.90625e+06 switch_11_nic_out_10 <= 0
 edge_flow_constr_10_11: flow_10_11 - 3.90625e+06 switch_10_11 <= 0
 edge_flow_constr_11_10: flow_11_10 - 3.90625e+06 switch_11_10 <= 0
 edge_flow_constr_source_nic_in_12: flow_source_nic_in_12
   - 5e+08 switch_source_nic_in_12 <= 0
 edge_flow_constr_nic_out_12_sink: flow_nic_out_12_sink
   - 5e+08 switch_nic_out_12_sink <= 0
 edge_flow_constr_nic_out_12_nic_in_0: flow_nic_out_12_nic_in_0
   - 61035.15625 switch_nic_out_12_nic_in_0 <= 0
 edge_flow_constr_nic_out_12_nic_in_4: flow_nic_out_12_nic_in_4
   - 61035.15625 switch_nic_out_12_nic_in_4 <= 0
 edge_flow_constr_nic_out_12_nic_in_6: flow_nic_out_12_nic_in_6
   - 61035.15625 switch_nic_out_12_nic_in_6 <= 0
 edge_flow_constr_nic_out_12_nic_in_7: flow_nic_out_12_nic_in_7
   - 61035.15625 switch_nic_out_12_nic_in_7 <= 0
 edge_flow_constr_nic_out_12_nic_in_8: flow_nic_out_12_nic_in_8
   - 61035.15625 switch_nic_out_12_nic_in_8 <= 0
 edge_flow_constr_nic_out_12_nic_in_10: flow_nic_out_12_nic_in_10
   - 61035.15625 switch_nic_out_12_nic_in_10 <= 0
 edge_flow_constr_nic_out_12_nic_in_13: flow_nic_out_12_nic_in_13
   - 61035.15625 switch_nic_out_12_nic_in_13 <= 0
 edge_flow_constr_nic_out_12_nic_in_14: flow_nic_out_12_nic_in_14
   - 61035.15625 switch_nic_out_12_nic_in_14 <= 0
 edge_flow_constr_nic_out_12_nic_in_15: flow_nic_out_12_nic_in_15
   - 61035.15625 switch_nic_out_12_nic_in_15 <= 0
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
 edge_flow_constr_nic_out_13_nic_in_4: flow_nic_out_13_nic_in_4
   - 61035.15625 switch_nic_out_13_nic_in_4 <= 0
 edge_flow_constr_nic_out_13_nic_in_6: flow_nic_out_13_nic_in_6
   - 61035.15625 switch_nic_out_13_nic_in_6 <= 0
 edge_flow_constr_nic_out_13_nic_in_7: flow_nic_out_13_nic_in_7
   - 61035.15625 switch_nic_out_13_nic_in_7 <= 0
 edge_flow_constr_nic_out_13_nic_in_8: flow_nic_out_13_nic_in_8
   - 61035.15625 switch_nic_out_13_nic_in_8 <= 0
 edge_flow_constr_nic_out_13_nic_in_10: flow_nic_out_13_nic_in_10
   - 61035.15625 switch_nic_out_13_nic_in_10 <= 0
 edge_flow_constr_nic_out_13_nic_in_12: flow_nic_out_13_nic_in_12
   - 61035.15625 switch_nic_out_13_nic_in_12 <= 0
 edge_flow_constr_nic_out_13_nic_in_14: flow_nic_out_13_nic_in_14
   - 61035.15625 switch_nic_out_13_nic_in_14 <= 0
 edge_flow_constr_nic_out_13_nic_in_15: flow_nic_out_13_nic_in_15
   - 61035.15625 switch_nic_out_13_nic_in_15 <= 0
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
 edge_flow_constr_nic_out_14_nic_in_4: flow_nic_out_14_nic_in_4
   - 61035.15625 switch_nic_out_14_nic_in_4 <= 0
 edge_flow_constr_nic_out_14_nic_in_6: flow_nic_out_14_nic_in_6
   - 61035.15625 switch_nic_out_14_nic_in_6 <= 0
 edge_flow_constr_nic_out_14_nic_in_7: flow_nic_out_14_nic_in_7
   - 61035.15625 switch_nic_out_14_nic_in_7 <= 0
 edge_flow_constr_nic_out_14_nic_in_8: flow_nic_out_14_nic_in_8
   - 61035.15625 switch_nic_out_14_nic_in_8 <= 0
 edge_flow_constr_nic_out_14_nic_in_10: flow_nic_out_14_nic_in_10
   - 61035.15625 switch_nic_out_14_nic_in_10 <= 0
 edge_flow_constr_nic_out_14_nic_in_12: flow_nic_out_14_nic_in_12
   - 61035.15625 switch_nic_out_14_nic_in_12 <= 0
 edge_flow_constr_nic_out_14_nic_in_13: flow_nic_out_14_nic_in_13
   - 61035.15625 switch_nic_out_14_nic_in_13 <= 0
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
 edge_flow_constr_nic_out_15_nic_in_4: flow_nic_out_15_nic_in_4
   - 61035.15625 switch_nic_out_15_nic_in_4 <= 0
 edge_flow_constr_nic_out_15_nic_in_6: flow_nic_out_15_nic_in_6
   - 61035.15625 switch_nic_out_15_nic_in_6 <= 0
 edge_flow_constr_nic_out_15_nic_in_7: flow_nic_out_15_nic_in_7
   - 61035.15625 switch_nic_out_15_nic_in_7 <= 0
 edge_flow_constr_nic_out_15_nic_in_8: flow_nic_out_15_nic_in_8
   - 61035.15625 switch_nic_out_15_nic_in_8 <= 0
 edge_flow_constr_nic_out_15_nic_in_10: flow_nic_out_15_nic_in_10
   - 61035.15625 switch_nic_out_15_nic_in_10 <= 0
 edge_flow_constr_nic_out_15_nic_in_12: flow_nic_out_15_nic_in_12
   - 61035.15625 switch_nic_out_15_nic_in_12 <= 0
 edge_flow_constr_nic_out_15_nic_in_13: flow_nic_out_15_nic_in_13
   - 61035.15625 switch_nic_out_15_nic_in_13 <= 0
 edge_flow_constr_nic_out_15_nic_in_14: flow_nic_out_15_nic_in_14
   - 61035.15625 switch_nic_out_15_nic_in_14 <= 0
 edge_flow_constr_nic_in_15_15: flow_nic_in_15_15
   - 3.90625e+06 switch_nic_in_15_15 <= 0
 edge_flow_constr_15_nic_out_15: flow_15_nic_out_15
   - 3.90625e+06 switch_15_nic_out_15 <= 0
 force_group_off_constr_6: group_tp_active_6 = 0
 force_group_off_constr_7: group_tp_active_7 = 0
 force_group_off_constr_12: group_tp_active_12 = 0
 force_group_off_constr_13: group_tp_active_13 = 0
 force_group_off_constr_14: group_tp_active_14 = 0
 force_group_off_constr_15: group_tp_active_15 = 0
 R620: - group_tp_active_0 + tp_on_and_switch_source_nic_in_0_0 <= 0
 R621: - switch_source_nic_in_0 + tp_on_and_switch_source_nic_in_0_0 <= 0
 R622: - switch_source_nic_in_0 - group_tp_active_0
   + tp_on_and_switch_source_nic_in_0_0 >= -1
 R623: - group_tp_active_4 + tp_on_and_switch_nic_out_0_nic_in_4_4 <= 0
 R624: - switch_nic_out_0_nic_in_4 + tp_on_and_switch_nic_out_0_nic_in_4_4
   <= 0
 R625: - switch_nic_out_0_nic_in_4 - group_tp_active_4
   + tp_on_and_switch_nic_out_0_nic_in_4_4 >= -1
 R626: - group_tp_active_8 + tp_on_and_switch_nic_out_0_nic_in_8_8 <= 0
 R627: - switch_nic_out_0_nic_in_8 + tp_on_and_switch_nic_out_0_nic_in_8_8
   <= 0
 R628: - switch_nic_out_0_nic_in_8 - group_tp_active_8
   + tp_on_and_switch_nic_out_0_nic_in_8_8 >= -1
 R629: - group_tp_active_10 + tp_on_and_switch_nic_out_0_nic_in_10_10 <= 0
 R630: - switch_nic_out_0_nic_in_10
   + tp_on_and_switch_nic_out_0_nic_in_10_10 <= 0
 R631: - switch_nic_out_0_nic_in_10 - group_tp_active_10
   + tp_on_and_switch_nic_out_0_nic_in_10_10 >= -1
 R632: - group_tp_active_4 + tp_on_and_switch_source_nic_in_4_4 <= 0
 R633: - switch_source_nic_in_4 + tp_on_and_switch_source_nic_in_4_4 <= 0
 R634: - switch_source_nic_in_4 - group_tp_active_4
   + tp_on_and_switch_source_nic_in_4_4 >= -1
 R635: - group_tp_active_0 + tp_on_and_switch_nic_out_4_nic_in_0_0 <= 0
 R636: - switch_nic_out_4_nic_in_0 + tp_on_and_switch_nic_out_4_nic_in_0_0
   <= 0
 R637: - switch_nic_out_4_nic_in_0 - group_tp_active_0
   + tp_on_and_switch_nic_out_4_nic_in_0_0 >= -1
 R638: - group_tp_active_8 + tp_on_and_switch_nic_out_4_nic_in_8_8 <= 0
 R639: - switch_nic_out_4_nic_in_8 + tp_on_and_switch_nic_out_4_nic_in_8_8
   <= 0
 R640: - switch_nic_out_4_nic_in_8 - group_tp_active_8
   + tp_on_and_switch_nic_out_4_nic_in_8_8 >= -1
 R641: - group_tp_active_10 + tp_on_and_switch_nic_out_4_nic_in_10_10 <= 0
 R642: - switch_nic_out_4_nic_in_10
   + tp_on_and_switch_nic_out_4_nic_in_10_10 <= 0
 R643: - switch_nic_out_4_nic_in_10 - group_tp_active_10
   + tp_on_and_switch_nic_out_4_nic_in_10_10 >= -1
 R644: - group_tp_active_0 + tp_on_and_switch_nic_out_6_nic_in_0_0 <= 0
 R645: - switch_nic_out_6_nic_in_0 + tp_on_and_switch_nic_out_6_nic_in_0_0
   <= 0
 R646: - switch_nic_out_6_nic_in_0 - group_tp_active_0
   + tp_on_and_switch_nic_out_6_nic_in_0_0 >= -1
 R647: - group_tp_active_4 + tp_on_and_switch_nic_out_6_nic_in_4_4 <= 0
 R648: - switch_nic_out_6_nic_in_4 + tp_on_and_switch_nic_out_6_nic_in_4_4
   <= 0
 R649: - switch_nic_out_6_nic_in_4 - group_tp_active_4
   + tp_on_and_switch_nic_out_6_nic_in_4_4 >= -1
 R650: - group_tp_active_8 + tp_on_and_switch_nic_out_6_nic_in_8_8 <= 0
 R651: - switch_nic_out_6_nic_in_8 + tp_on_and_switch_nic_out_6_nic_in_8_8
   <= 0
 R652: - switch_nic_out_6_nic_in_8 - group_tp_active_8
   + tp_on_and_switch_nic_out_6_nic_in_8_8 >= -1
 R653: - group_tp_active_10 + tp_on_and_switch_nic_out_6_nic_in_10_10 <= 0
 R654: - switch_nic_out_6_nic_in_10
   + tp_on_and_switch_nic_out_6_nic_in_10_10 <= 0
 R655: - switch_nic_out_6_nic_in_10 - group_tp_active_10
   + tp_on_and_switch_nic_out_6_nic_in_10_10 >= -1
 R656: - group_tp_active_0 + tp_on_and_switch_nic_out_7_nic_in_0_0 <= 0
 R657: - switch_nic_out_7_nic_in_0 + tp_on_and_switch_nic_out_7_nic_in_0_0
   <= 0
 R658: - switch_nic_out_7_nic_in_0 - group_tp_active_0
   + tp_on_and_switch_nic_out_7_nic_in_0_0 >= -1
 R659: - group_tp_active_4 + tp_on_and_switch_nic_out_7_nic_in_4_4 <= 0
 R660: - switch_nic_out_7_nic_in_4 + tp_on_and_switch_nic_out_7_nic_in_4_4
   <= 0
 R661: - switch_nic_out_7_nic_in_4 - group_tp_active_4
   + tp_on_and_switch_nic_out_7_nic_in_4_4 >= -1
 R662: - group_tp_active_8 + tp_on_and_switch_nic_out_7_nic_in_8_8 <= 0
 R663: - switch_nic_out_7_nic_in_8 + tp_on_and_switch_nic_out_7_nic_in_8_8
   <= 0
 R664: - switch_nic_out_7_nic_in_8 - group_tp_active_8
   + tp_on_and_switch_nic_out_7_nic_in_8_8 >= -1
 R665: - group_tp_active_10 + tp_on_and_switch_nic_out_7_nic_in_10_10 <= 0
 R666: - switch_nic_out_7_nic_in_10
   + tp_on_and_switch_nic_out_7_nic_in_10_10 <= 0
 R667: - switch_nic_out_7_nic_in_10 - group_tp_active_10
   + tp_on_and_switch_nic_out_7_nic_in_10_10 >= -1
 R668: - group_tp_active_8 + tp_on_and_switch_source_nic_in_8_8 <= 0
 R669: - switch_source_nic_in_8 + tp_on_and_switch_source_nic_in_8_8 <= 0
 R670: - switch_source_nic_in_8 - group_tp_active_8
   + tp_on_and_switch_source_nic_in_8_8 >= -1
 R671: - group_tp_active_0 + tp_on_and_switch_nic_out_8_nic_in_0_0 <= 0
 R672: - switch_nic_out_8_nic_in_0 + tp_on_and_switch_nic_out_8_nic_in_0_0
   <= 0
 R673: - switch_nic_out_8_nic_in_0 - group_tp_active_0
   + tp_on_and_switch_nic_out_8_nic_in_0_0 >= -1
 R674: - group_tp_active_4 + tp_on_and_switch_nic_out_8_nic_in_4_4 <= 0
 R675: - switch_nic_out_8_nic_in_4 + tp_on_and_switch_nic_out_8_nic_in_4_4
   <= 0
 R676: - switch_nic_out_8_nic_in_4 - group_tp_active_4
   + tp_on_and_switch_nic_out_8_nic_in_4_4 >= -1
 R677: - group_tp_active_10 + tp_on_and_switch_nic_out_8_nic_in_10_10 <= 0
 R678: - switch_nic_out_8_nic_in_10
   + tp_on_and_switch_nic_out_8_nic_in_10_10 <= 0
 R679: - switch_nic_out_8_nic_in_10 - group_tp_active_10
   + tp_on_and_switch_nic_out_8_nic_in_10_10 >= -1
 R680: - group_tp_active_10 + tp_on_and_switch_source_nic_in_10_10 <= 0
 R681: - switch_source_nic_in_10 + tp_on_and_switch_source_nic_in_10_10
   <= 0
 R682: - switch_source_nic_in_10 - group_tp_active_10
   + tp_on_and_switch_source_nic_in_10_10 >= -1
 R683: - group_tp_active_0 + tp_on_and_switch_nic_out_10_nic_in_0_0 <= 0
 R684: - switch_nic_out_10_nic_in_0
   + tp_on_and_switch_nic_out_10_nic_in_0_0 <= 0
 R685: - switch_nic_out_10_nic_in_0 - group_tp_active_0
   + tp_on_and_switch_nic_out_10_nic_in_0_0 >= -1
 R686: - group_tp_active_4 + tp_on_and_switch_nic_out_10_nic_in_4_4 <= 0
 R687: - switch_nic_out_10_nic_in_4
   + tp_on_and_switch_nic_out_10_nic_in_4_4 <= 0
 R688: - switch_nic_out_10_nic_in_4 - group_tp_active_4
   + tp_on_and_switch_nic_out_10_nic_in_4_4 >= -1
 R689: - group_tp_active_8 + tp_on_and_switch_nic_out_10_nic_in_8_8 <= 0
 R690: - switch_nic_out_10_nic_in_8
   + tp_on_and_switch_nic_out_10_nic_in_8_8 <= 0
 R691: - switch_nic_out_10_nic_in_8 - group_tp_active_8
   + tp_on_and_switch_nic_out_10_nic_in_8_8 >= -1
 R692: - group_tp_active_0 + tp_on_and_switch_nic_out_12_nic_in_0_0 <= 0
 R693: - switch_nic_out_12_nic_in_0
   + tp_on_and_switch_nic_out_12_nic_in_0_0 <= 0
 R694: - switch_nic_out_12_nic_in_0 - group_tp_active_0
   + tp_on_and_switch_nic_out_12_nic_in_0_0 >= -1
 R695: - group_tp_active_4 + tp_on_and_switch_nic_out_12_nic_in_4_4 <= 0
 R696: - switch_nic_out_12_nic_in_4
   + tp_on_and_switch_nic_out_12_nic_in_4_4 <= 0
 R697: - switch_nic_out_12_nic_in_4 - group_tp_active_4
   + tp_on_and_switch_nic_out_12_nic_in_4_4 >= -1
 R698: - group_tp_active_8 + tp_on_and_switch_nic_out_12_nic_in_8_8 <= 0
 R699: - switch_nic_out_12_nic_in_8
   + tp_on_and_switch_nic_out_12_nic_in_8_8 <= 0
 R700: - switch_nic_out_12_nic_in_8 - group_tp_active_8
   + tp_on_and_switch_nic_out_12_nic_in_8_8 >= -1
 R701: - group_tp_active_10 + tp_on_and_switch_nic_out_12_nic_in_10_10
   <= 0
 R702: - switch_nic_out_12_nic_in_10
   + tp_on_and_switch_nic_out_12_nic_in_10_10 <= 0
 R703: - switch_nic_out_12_nic_in_10 - group_tp_active_10
   + tp_on_and_switch_nic_out_12_nic_in_10_10 >= -1
 R704: - group_tp_active_0 + tp_on_and_switch_nic_out_13_nic_in_0_0 <= 0
 R705: - switch_nic_out_13_nic_in_0
   + tp_on_and_switch_nic_out_13_nic_in_0_0 <= 0
 R706: - switch_nic_out_13_nic_in_0 - group_tp_active_0
   + tp_on_and_switch_nic_out_13_nic_in_0_0 >= -1
 R707: - group_tp_active_4 + tp_on_and_switch_nic_out_13_nic_in_4_4 <= 0
 R708: - switch_nic_out_13_nic_in_4
   + tp_on_and_switch_nic_out_13_nic_in_4_4 <= 0
 R709: - switch_nic_out_13_nic_in_4 - group_tp_active_4
   + tp_on_and_switch_nic_out_13_nic_in_4_4 >= -1
 R710: - group_tp_active_8 + tp_on_and_switch_nic_out_13_nic_in_8_8 <= 0
 R711: - switch_nic_out_13_nic_in_8
   + tp_on_and_switch_nic_out_13_nic_in_8_8 <= 0
 R712: - switch_nic_out_13_nic_in_8 - group_tp_active_8
   + tp_on_and_switch_nic_out_13_nic_in_8_8 >= -1
 R713: - group_tp_active_10 + tp_on_and_switch_nic_out_13_nic_in_10_10
   <= 0
 R714: - switch_nic_out_13_nic_in_10
   + tp_on_and_switch_nic_out_13_nic_in_10_10 <= 0
 R715: - switch_nic_out_13_nic_in_10 - group_tp_active_10
   + tp_on_and_switch_nic_out_13_nic_in_10_10 >= -1
 R716: - group_tp_active_0 + tp_on_and_switch_nic_out_14_nic_in_0_0 <= 0
 R717: - switch_nic_out_14_nic_in_0
   + tp_on_and_switch_nic_out_14_nic_in_0_0 <= 0
 R718: - switch_nic_out_14_nic_in_0 - group_tp_active_0
   + tp_on_and_switch_nic_out_14_nic_in_0_0 >= -1
 R719: - group_tp_active_4 + tp_on_and_switch_nic_out_14_nic_in_4_4 <= 0
 R720: - switch_nic_out_14_nic_in_4
   + tp_on_and_switch_nic_out_14_nic_in_4_4 <= 0
 R721: - switch_nic_out_14_nic_in_4 - group_tp_active_4
   + tp_on_and_switch_nic_out_14_nic_in_4_4 >= -1
 R722: - group_tp_active_8 + tp_on_and_switch_nic_out_14_nic_in_8_8 <= 0
 R723: - switch_nic_out_14_nic_in_8
   + tp_on_and_switch_nic_out_14_nic_in_8_8 <= 0
 R724: - switch_nic_out_14_nic_in_8 - group_tp_active_8
   + tp_on_and_switch_nic_out_14_nic_in_8_8 >= -1
 R725: - group_tp_active_10 + tp_on_and_switch_nic_out_14_nic_in_10_10
   <= 0
 R726: - switch_nic_out_14_nic_in_10
   + tp_on_and_switch_nic_out_14_nic_in_10_10 <= 0
 R727: - switch_nic_out_14_nic_in_10 - group_tp_active_10
   + tp_on_and_switch_nic_out_14_nic_in_10_10 >= -1
 R728: - group_tp_active_0 + tp_on_and_switch_nic_out_15_nic_in_0_0 <= 0
 R729: - switch_nic_out_15_nic_in_0
   + tp_on_and_switch_nic_out_15_nic_in_0_0 <= 0
 R730: - switch_nic_out_15_nic_in_0 - group_tp_active_0
   + tp_on_and_switch_nic_out_15_nic_in_0_0 >= -1
 R731: - group_tp_active_4 + tp_on_and_switch_nic_out_15_nic_in_4_4 <= 0
 R732: - switch_nic_out_15_nic_in_4
   + tp_on_and_switch_nic_out_15_nic_in_4_4 <= 0
 R733: - switch_nic_out_15_nic_in_4 - group_tp_active_4
   + tp_on_and_switch_nic_out_15_nic_in_4_4 >= -1
 R734: - group_tp_active_8 + tp_on_and_switch_nic_out_15_nic_in_8_8 <= 0
 R735: - switch_nic_out_15_nic_in_8
   + tp_on_and_switch_nic_out_15_nic_in_8_8 <= 0
 R736: - switch_nic_out_15_nic_in_8 - group_tp_active_8
   + tp_on_and_switch_nic_out_15_nic_in_8_8 >= -1
 R737: - group_tp_active_10 + tp_on_and_switch_nic_out_15_nic_in_10_10
   <= 0
 R738: - switch_nic_out_15_nic_in_10
   + tp_on_and_switch_nic_out_15_nic_in_10_10 <= 0
 R739: - switch_nic_out_15_nic_in_10 - group_tp_active_10
   + tp_on_and_switch_nic_out_15_nic_in_10_10 >= -1
 force_min_throughput: flow_source_nic_in_0 + flow_source_nic_in_4
   + flow_source_nic_in_6 + flow_source_nic_in_7 + flow_source_nic_in_8
   + flow_source_nic_in_10 + flow_source_nic_in_12 + flow_source_nic_in_13
   + flow_source_nic_in_14 + flow_source_nic_in_15 >= 1
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
   - 3949.714285714286 hold_4_1 - 1974.857142857143 hold_4_2
   - 1316.571428571428 hold_4_3 - 987.4285714285714 hold_4_4
   - 789.9428571428572 hold_4_5 - 658.2857142857142 hold_4_6
   - 564.2448979591836 hold_4_7 - 493.7142857142857 hold_4_8
   - 438.8571428571428 hold_4_9 - 394.9714285714286 hold_4_10
   - 359.0649350649351 hold_4_11 - 329.1428571428571 hold_4_12
   - 303.8241758241758 hold_4_13 + flow_nic_in_4_4 + flow_5_4 <= 0
 node_throughput_constraint_5: group_tp_active_4 = 0 ->
   - 3949.714285714286 hold_5_1 - 1974.857142857143 hold_5_2
   - 1316.571428571428 hold_5_3 - 987.4285714285714 hold_5_4
   - 789.9428571428572 hold_5_5 - 658.2857142857142 hold_5_6
   - 564.2448979591836 hold_5_7 - 493.7142857142857 hold_5_8
   - 438.8571428571428 hold_5_9 - 394.9714285714286 hold_5_10
   - 359.0649350649351 hold_5_11 - 329.1428571428571 hold_5_12
   - 303.8241758241758 hold_5_13 + flow_nic_in_4_5 + flow_4_5 <= 0
 node_throughput_constraint_6: group_tp_active_6 = 0 ->
   - 9325.714285714286 hold_6_1 - 4662.857142857143 hold_6_2
   - 3108.571428571428 hold_6_3 - 2331.428571428572 hold_6_4
   - 1865.142857142857 hold_6_5 - 1554.285714285714 hold_6_6
   - 1332.244897959184 hold_6_7 - 1165.714285714286 hold_6_8
   - 1036.190476190476 hold_6_9 - 932.5714285714284 hold_6_10
   - 847.7922077922078 hold_6_11 - 777.1428571428571 hold_6_12
   - 717.3626373626373 hold_6_13 - 666.1224489795918 hold_6_14
   - 621.7142857142858 hold_6_15 - 582.8571428571429 hold_6_16
   - 548.5714285714286 hold_6_17 - 518.0952380952381 hold_6_18
   - 490.8270676691729 hold_6_19 - 466.2857142857142 hold_6_20
   - 444.0816326530613 hold_6_21 - 423.8961038961039 hold_6_22
   + flow_nic_in_6_6 <= 0
 node_throughput_constraint_7: group_tp_active_7 = 0 ->
   - 15314.28571428572 hold_7_1 - 7657.142857142858 hold_7_2
   - 5104.761904761905 hold_7_3 - 3828.571428571429 hold_7_4
   - 3062.857142857143 hold_7_5 - 2552.380952380952 hold_7_6
   - 2187.755102040816 hold_7_7 - 1914.285714285714 hold_7_8
   - 1701.587301587301 hold_7_9 - 1531.428571428571 hold_7_10
   - 1392.207792207792 hold_7_11 - 1276.190476190476 hold_7_12
   - 1178.021978021978 hold_7_13 - 1093.877551020408 hold_7_14
   - 1020.952380952381 hold_7_15 - 957.1428571428572 hold_7_16
   - 900.8403361344537 hold_7_17 - 850.7936507936507 hold_7_18
   - 806.015037593985 hold_7_19 - 765.7142857142857 hold_7_20
   - 729.2517006802721 hold_7_21 - 696.1038961038961 hold_7_22
   + flow_nic_in_7_7 <= 0
 node_throughput_constraint_8: group_tp_active_8 = 0 ->
   - 1371.428571428571 hold_8_1 - 685.7142857142857 hold_8_2
   - 457.1428571428571 hold_8_3 - 342.8571428571428 hold_8_4
   - 274.2857142857143 hold_8_5 - 228.5714285714286 hold_8_6
   + flow_nic_in_8_8 + flow_9_8 <= 0
 node_throughput_constraint_9: group_tp_active_8 = 0 ->
   - 1371.428571428571 hold_9_1 - 685.7142857142857 hold_9_2
   - 457.1428571428571 hold_9_3 - 342.8571428571428 hold_9_4
   - 274.2857142857143 hold_9_5 - 228.5714285714286 hold_9_6
   + flow_nic_in_8_9 + flow_8_9 <= 0
 node_throughput_constraint_10: group_tp_active_10 = 0 -> - 2048 hold_10_1
   - 1024 hold_10_2 - 682.6666666666666 hold_10_3 - 512 hold_10_4
   + flow_nic_in_10_10 + flow_11_10 <= 0
 node_throughput_constraint_11: group_tp_active_10 = 0 -> - 2048 hold_11_1
   - 1024 hold_11_2 - 682.6666666666666 hold_11_3 - 512 hold_11_4
   + flow_nic_in_10_11 + flow_10_11 <= 0
 node_throughput_constraint_12: group_tp_active_12 = 0 ->
   - 1462.857142857143 hold_12_1 - 731.4285714285714 hold_12_2
   - 487.6190476190476 hold_12_3 - 365.7142857142857 hold_12_4
   + flow_nic_in_12_12 <= 0
 node_throughput_constraint_13: group_tp_active_13 = 0 ->
   - 1462.857142857143 hold_13_1 - 731.4285714285714 hold_13_2
   - 487.6190476190476 hold_13_3 - 365.7142857142857 hold_13_4
   + flow_nic_in_13_13 <= 0
 node_throughput_constraint_14: group_tp_active_14 = 0 ->
   - 1371.428571428571 hold_14_1 - 685.7142857142857 hold_14_2
   - 457.1428571428571 hold_14_3 - 342.8571428571428 hold_14_4
   - 274.2857142857143 hold_14_5 - 228.5714285714286 hold_14_6
   + flow_nic_in_14_14 <= 0
 node_throughput_constraint_15: group_tp_active_15 = 0 ->
   - 4265.142857142857 hold_15_1 - 2132.571428571428 hold_15_2
   - 1421.714285714286 hold_15_3 - 1066.285714285714 hold_15_4
   - 853.0285714285714 hold_15_5 - 710.8571428571429 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_0_source_nic_in_0: switch_source_nic_in_0 = 1 ->
   - 4049.259480279933 hold_0_1 - 2077.199231903821 hold_0_2
   - 1396.889562629835 hold_0_3 - 1052.260583931315 hold_0_4
   - 844.0288127735122 hold_0_5 - 704.5963014358132 hold_0_6
   + flow_nic_in_0_0 + flow_1_0 + flow_2_0 + flow_3_0 <= 0
 constr_intergroup_throughput_4_nic_out_0_nic_in_4: switch_nic_out_0_nic_in_4 = 1 ->
   - 3167.736021998167 hold_4_1 - 1757.884028484232 hold_4_2
   - 1216.473072861668 hold_4_3 - 930.032292787944 hold_4_4
   - 752.7771727292528 hold_4_5 - 632.2722283205268 hold_4_6
   - 545.0244440939914 hold_4_7 - 478.9356984478936 hold_4_8
   - 427.1412680756396 hold_4_9 - 385.4561677448137 hold_4_10
   - 351.1838227822376 hold_4_11 - 322.5083986562149 hold_4_12
   - 298.1623673539816 hold_4_13 + flow_nic_in_4_4 + flow_5_4 <= 0
 constr_intergroup_throughput_6_nic_out_0_nic_in_6: switch_nic_out_0_nic_in_6 = 1 ->
   - 5891.696750902527 hold_6_1 - 3610.619469026549 hold_6_2
   - 2602.870813397129 hold_6_3 - 2034.912718204489 hold_6_4
   - 1670.419651995905 hold_6_5 - 1416.666666666667 hold_6_6
   - 1229.841748304446 hold_6_7 - 1086.551264980027 hold_6_8
   - 973.1663685152056 hold_6_9 - 881.2095032397407 hold_6_10
   - 805.1307350764677 hold_6_11 - 741.1444141689373 hold_6_12
   - 686.5797223390829 hold_6_13 - 639.4984326018808 hold_6_14
   - 598.4598459845985 hold_6_15 - 562.3707787732598 hold_6_16
   - 530.3867403314916 hold_6_17 - 501.8450184501845 hold_6_18
   - 476.2182667055733 hold_6_19 - 453.081621321488 hold_6_20
   - 432.0889594916601 hold_6_21 - 412.9554655870446 hold_6_22
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_0_nic_in_7: switch_nic_out_0_nic_in_7 = 1 ->
   - 7824.817518248176 hold_7_1 - 5178.743961352658 hold_7_2
   - 3870.036101083032 hold_7_3 - 3089.337175792507 hold_7_4
   - 2570.743405275779 hold_7_5 - 2201.23203285421 hold_7_6
   - 1924.5960502693 hold_7_7 - 1709.728867623605 hold_7_8
   - 1538.020086083214 hold_7_9 - 1397.653194263364 hold_7_10
   - 1280.764635603345 hold_7_11 - 1181.918412348401 hold_7_12
   - 1097.236438075742 hold_7_13 - 1023.877745940783 hold_7_14
   - 959.7135183527306 hold_7_15 - 903.117101937658 hold_7_16
   - 852.8241845664279 hold_7_17 - 807.8372268274302 hold_7_18
   - 767.3586256263421 hold_7_19 - 730.7430129516018 hold_7_20
   - 697.4625894599869 hold_7_21 - 667.0815183571873 hold_7_22
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_0_nic_in_8: switch_nic_out_0_nic_in_8 = 1 ->
   - 1263.157894736842 hold_8_1 - 657.5342465753424 hold_8_2
   - 444.4444444444444 hold_8_3 - 335.6643356643356 hold_8_4
   - 269.6629213483146 hold_8_5 - 225.3521126760563 hold_8_6
   + flow_nic_in_8_8 + flow_9_8 <= 0
 constr_intergroup_throughput_10_nic_out_0_nic_in_10: switch_nic_out_0_nic_in_10 = 1 ->
   - 1815.602836879433 hold_10_1 - 962.406015037594 hold_10_2
   - 654.7314578005115 hold_10_3 - 496.1240310077519 hold_10_4
   + flow_nic_in_10_10 + flow_11_10 <= 0
 constr_intergroup_throughput_12_nic_out_0_nic_in_12: switch_nic_out_0_nic_in_12 = 1 ->
   - 1340.314136125654 hold_12_1 - 699.4535519125683 hold_12_2
   - 473.1977818853974 hold_12_3 - 357.5418994413408 hold_12_4
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_0_nic_in_13: switch_nic_out_0_nic_in_13 = 1 ->
   - 1340.314136125654 hold_13_1 - 699.4535519125683 hold_13_2
   - 473.1977818853974 hold_13_3 - 357.5418994413408 hold_13_4
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_0_nic_in_14: switch_nic_out_0_nic_in_14 = 1 ->
   - 1263.157894736842 hold_14_1 - 657.5342465753424 hold_14_2
   - 444.4444444444444 hold_14_3 - 335.6643356643356 hold_14_4
   - 269.6629213483146 hold_14_5 - 225.3521126760563 hold_14_6
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_0_nic_in_15: switch_nic_out_0_nic_in_15 = 1 ->
   - 3367.47123843898 hold_15_1 - 1881.759737804109 hold_15_2
   - 1305.694043558121 hold_15_3 - 999.665171097569 hold_15_4
   - 809.8518960559865 hold_15_5 - 680.6182464779099 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_4_source_nic_in_4: switch_source_nic_in_4 = 1 ->
   - 3763.885863646265 hold_4_1 - 1927.280838724069 hold_4_2
   - 1295.25522824376 hold_4_3 - 975.3894784375707 hold_4_4
   - 782.2190032139786 hold_4_5 - 652.9131716164135 hold_4_6
   - 560.2931163062157 hold_4_7 - 490.6860517946388 hold_4_8
   - 436.4628324619231 hold_4_9 - 393.0309784833735 hold_4_10
   - 357.4605407418134 hold_4_11 - 327.7942180742089 hold_4_12
   - 302.6746772696222 hold_4_13 + flow_nic_in_4_4 + flow_5_4 <= 0
 constr_intergroup_throughput_0_nic_out_4_nic_in_0: switch_nic_out_4_nic_in_0 = 1 ->
   - 3367.47123843898 hold_0_1 - 1881.759737804109 hold_0_2
   - 1305.694043558121 hold_0_3 - 999.665171097569 hold_0_4
   - 809.8518960559865 hold_0_5 - 680.6182464779099 hold_0_6
   + flow_nic_in_0_0 + flow_1_0 + flow_2_0 + flow_3_0 <= 0
 constr_intergroup_throughput_6_nic_out_4_nic_in_6: switch_nic_out_4_nic_in_6 = 1 ->
   - 5891.696750902527 hold_6_1 - 3610.619469026549 hold_6_2
   - 2602.870813397129 hold_6_3 - 2034.912718204489 hold_6_4
   - 1670.419651995905 hold_6_5 - 1416.666666666667 hold_6_6
   - 1229.841748304446 hold_6_7 - 1086.551264980027 hold_6_8
   - 973.1663685152056 hold_6_9 - 881.2095032397407 hold_6_10
   - 805.1307350764677 hold_6_11 - 741.1444141689373 hold_6_12
   - 686.5797223390829 hold_6_13 - 639.4984326018808 hold_6_14
   - 598.4598459845985 hold_6_15 - 562.3707787732598 hold_6_16
   - 530.3867403314916 hold_6_17 - 501.8450184501845 hold_6_18
   - 476.2182667055733 hold_6_19 - 453.081621321488 hold_6_20
   - 432.0889594916601 hold_6_21 - 412.9554655870446 hold_6_22
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_4_nic_in_7: switch_nic_out_4_nic_in_7 = 1 ->
   - 7824.817518248176 hold_7_1 - 5178.743961352658 hold_7_2
   - 3870.036101083032 hold_7_3 - 3089.337175792507 hold_7_4
   - 2570.743405275779 hold_7_5 - 2201.23203285421 hold_7_6
   - 1924.5960502693 hold_7_7 - 1709.728867623605 hold_7_8
   - 1538.020086083214 hold_7_9 - 1397.653194263364 hold_7_10
   - 1280.764635603345 hold_7_11 - 1181.918412348401 hold_7_12
   - 1097.236438075742 hold_7_13 - 1023.877745940783 hold_7_14
   - 959.7135183527306 hold_7_15 - 903.117101937658 hold_7_16
   - 852.8241845664279 hold_7_17 - 807.8372268274302 hold_7_18
   - 767.3586256263421 hold_7_19 - 730.7430129516018 hold_7_20
   - 697.4625894599869 hold_7_21 - 667.0815183571873 hold_7_22
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_4_nic_in_8: switch_nic_out_4_nic_in_8 = 1 ->
   - 1263.157894736842 hold_8_1 - 657.5342465753424 hold_8_2
   - 444.4444444444444 hold_8_3 - 335.6643356643356 hold_8_4
   - 269.6629213483146 hold_8_5 - 225.3521126760563 hold_8_6
   + flow_nic_in_8_8 + flow_9_8 <= 0
 constr_intergroup_throughput_10_nic_out_4_nic_in_10: switch_nic_out_4_nic_in_10 = 1 ->
   - 1815.602836879433 hold_10_1 - 962.406015037594 hold_10_2
   - 654.7314578005115 hold_10_3 - 496.1240310077519 hold_10_4
   + flow_nic_in_10_10 + flow_11_10 <= 0
 constr_intergroup_throughput_12_nic_out_4_nic_in_12: switch_nic_out_4_nic_in_12 = 1 ->
   - 1340.314136125654 hold_12_1 - 699.4535519125683 hold_12_2
   - 473.1977818853974 hold_12_3 - 357.5418994413408 hold_12_4
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_4_nic_in_13: switch_nic_out_4_nic_in_13 = 1 ->
   - 1340.314136125654 hold_13_1 - 699.4535519125683 hold_13_2
   - 473.1977818853974 hold_13_3 - 357.5418994413408 hold_13_4
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_4_nic_in_14: switch_nic_out_4_nic_in_14 = 1 ->
   - 1263.157894736842 hold_14_1 - 657.5342465753424 hold_14_2
   - 444.4444444444444 hold_14_3 - 335.6643356643356 hold_14_4
   - 269.6629213483146 hold_14_5 - 225.3521126760563 hold_14_6
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_4_nic_in_15: switch_nic_out_4_nic_in_15 = 1 ->
   - 3367.47123843898 hold_15_1 - 1881.759737804109 hold_15_2
   - 1305.694043558121 hold_15_3 - 999.665171097569 hold_15_4
   - 809.8518960559865 hold_15_5 - 680.6182464779099 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_6_source_nic_in_6: switch_source_nic_in_6 = 1 ->
   - 8352.09825997953 hold_6_1 - 4406.047516198704 hold_6_2
   - 2992.299229922993 hold_6_3 - 2265.40810660744 hold_6_4
   - 1822.649095376368 hold_6_5 - 1524.663677130045 hold_6_6
   - 1310.42235426369 hold_6_7 - 1148.972120529428 hold_6_8
   - 1022.940955246333 hold_6_9 - 921.825576140985 hold_6_10
   - 838.9020252904288 hold_6_11 - 769.6661007357103 hold_6_12
   - 710.9871917748541 hold_6_13 - 660.6217616580311 hold_6_14
   - 616.9199364935361 hold_6_15 - 578.6413274712806 hold_6_16
   - 544.8354143019296 hold_6_17 - 514.7615442846328 hold_6_18
   - 487.8340407724039 hold_6_19 - 463.5836836723099 hold_6_20
   - 441.6301347621368 hold_6_21 - 421.6618437370815 hold_6_22
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_0_nic_out_6_nic_in_0: switch_nic_out_6_nic_in_0 = 1 ->
   - 3367.47123843898 hold_0_1 - 1881.759737804109 hold_0_2
   - 1305.694043558121 hold_0_3 - 999.665171097569 hold_0_4
   - 809.8518960559865 hold_0_5 - 680.6182464779099 hold_0_6
   + flow_nic_in_0_0 + flow_1_0 + flow_2_0 + flow_3_0 <= 0
 constr_intergroup_throughput_4_nic_out_6_nic_in_4: switch_nic_out_6_nic_in_4 = 1 ->
   - 3167.736021998167 hold_4_1 - 1757.884028484232 hold_4_2
   - 1216.473072861668 hold_4_3 - 930.032292787944 hold_4_4
   - 752.7771727292528 hold_4_5 - 632.2722283205268 hold_4_6
   - 545.0244440939914 hold_4_7 - 478.9356984478936 hold_4_8
   - 427.1412680756396 hold_4_9 - 385.4561677448137 hold_4_10
   - 351.1838227822376 hold_4_11 - 322.5083986562149 hold_4_12
   - 298.1623673539816 hold_4_13 + flow_nic_in_4_4 + flow_5_4 <= 0
 constr_intergroup_throughput_7_nic_out_6_nic_in_7: switch_nic_out_6_nic_in_7 = 1 ->
   - 7824.817518248176 hold_7_1 - 5178.743961352658 hold_7_2
   - 3870.036101083032 hold_7_3 - 3089.337175792507 hold_7_4
   - 2570.743405275779 hold_7_5 - 2201.23203285421 hold_7_6
   - 1924.5960502693 hold_7_7 - 1709.728867623605 hold_7_8
   - 1538.020086083214 hold_7_9 - 1397.653194263364 hold_7_10
   - 1280.764635603345 hold_7_11 - 1181.918412348401 hold_7_12
   - 1097.236438075742 hold_7_13 - 1023.877745940783 hold_7_14
   - 959.7135183527306 hold_7_15 - 903.117101937658 hold_7_16
   - 852.8241845664279 hold_7_17 - 807.8372268274302 hold_7_18
   - 767.3586256263421 hold_7_19 - 730.7430129516018 hold_7_20
   - 697.4625894599869 hold_7_21 - 667.0815183571873 hold_7_22
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_6_nic_in_8: switch_nic_out_6_nic_in_8 = 1 ->
   - 1263.157894736842 hold_8_1 - 657.5342465753424 hold_8_2
   - 444.4444444444444 hold_8_3 - 335.6643356643356 hold_8_4
   - 269.6629213483146 hold_8_5 - 225.3521126760563 hold_8_6
   + flow_nic_in_8_8 + flow_9_8 <= 0
 constr_intergroup_throughput_10_nic_out_6_nic_in_10: switch_nic_out_6_nic_in_10 = 1 ->
   - 1815.602836879433 hold_10_1 - 962.406015037594 hold_10_2
   - 654.7314578005115 hold_10_3 - 496.1240310077519 hold_10_4
   + flow_nic_in_10_10 + flow_11_10 <= 0
 constr_intergroup_throughput_12_nic_out_6_nic_in_12: switch_nic_out_6_nic_in_12 = 1 ->
   - 1340.314136125654 hold_12_1 - 699.4535519125683 hold_12_2
   - 473.1977818853974 hold_12_3 - 357.5418994413408 hold_12_4
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_6_nic_in_13: switch_nic_out_6_nic_in_13 = 1 ->
   - 1340.314136125654 hold_13_1 - 699.4535519125683 hold_13_2
   - 473.1977818853974 hold_13_3 - 357.5418994413408 hold_13_4
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_6_nic_in_14: switch_nic_out_6_nic_in_14 = 1 ->
   - 1263.157894736842 hold_14_1 - 657.5342465753424 hold_14_2
   - 444.4444444444444 hold_14_3 - 335.6643356643356 hold_14_4
   - 269.6629213483146 hold_14_5 - 225.3521126760563 hold_14_6
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_6_nic_in_15: switch_nic_out_6_nic_in_15 = 1 ->
   - 3367.47123843898 hold_15_1 - 1881.759737804109 hold_15_2
   - 1305.694043558121 hold_15_3 - 999.665171097569 hold_15_4
   - 809.8518960559865 hold_15_5 - 680.6182464779099 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_7_source_nic_in_7: switch_source_nic_in_7 = 1 ->
   - 12853.7170263789 hold_7_1 - 6988.265971316819 hold_7_2
   - 4798.567591763653 hold_7_3 - 3653.71506475801 hold_7_4
   - 2949.917446340121 hold_7_5 - 2473.465620673743 hold_7_6
   - 2129.519268970997 hold_7_7 - 1869.550052319498 hold_7_8
   - 1666.148585638794 hold_7_9 - 1502.663302495094 hold_7_10
   - 1368.39417921879 hold_7_11 - 1256.151863135693 hold_7_12
   - 1160.927008880225 hold_7_13 - 1079.122206563318 hold_7_14
   - 1008.087267255972 hold_7_15 - 945.8267160755252 hold_7_16
   - 890.8093734419145 hold_7_17 - 841.8407413224438 hold_7_18
   - 797.9752865862737 hold_7_19 - 758.4547898684025 hold_7_20
   - 722.664149925846 hold_7_21 - 690.0991373760783 hold_7_22
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_0_nic_out_7_nic_in_0: switch_nic_out_7_nic_in_0 = 1 ->
   - 3367.47123843898 hold_0_1 - 1881.759737804109 hold_0_2
   - 1305.694043558121 hold_0_3 - 999.665171097569 hold_0_4
   - 809.8518960559865 hold_0_5 - 680.6182464779099 hold_0_6
   + flow_nic_in_0_0 + flow_1_0 + flow_2_0 + flow_3_0 <= 0
 constr_intergroup_throughput_4_nic_out_7_nic_in_4: switch_nic_out_7_nic_in_4 = 1 ->
   - 3167.736021998167 hold_4_1 - 1757.884028484232 hold_4_2
   - 1216.473072861668 hold_4_3 - 930.032292787944 hold_4_4
   - 752.7771727292528 hold_4_5 - 632.2722283205268 hold_4_6
   - 545.0244440939914 hold_4_7 - 478.9356984478936 hold_4_8
   - 427.1412680756396 hold_4_9 - 385.4561677448137 hold_4_10
   - 351.1838227822376 hold_4_11 - 322.5083986562149 hold_4_12
   - 298.1623673539816 hold_4_13 + flow_nic_in_4_4 + flow_5_4 <= 0
 constr_intergroup_throughput_6_nic_out_7_nic_in_6: switch_nic_out_7_nic_in_6 = 1 ->
   - 5891.696750902527 hold_6_1 - 3610.619469026549 hold_6_2
   - 2602.870813397129 hold_6_3 - 2034.912718204489 hold_6_4
   - 1670.419651995905 hold_6_5 - 1416.666666666667 hold_6_6
   - 1229.841748304446 hold_6_7 - 1086.551264980027 hold_6_8
   - 973.1663685152056 hold_6_9 - 881.2095032397407 hold_6_10
   - 805.1307350764677 hold_6_11 - 741.1444141689373 hold_6_12
   - 686.5797223390829 hold_6_13 - 639.4984326018808 hold_6_14
   - 598.4598459845985 hold_6_15 - 562.3707787732598 hold_6_16
   - 530.3867403314916 hold_6_17 - 501.8450184501845 hold_6_18
   - 476.2182667055733 hold_6_19 - 453.081621321488 hold_6_20
   - 432.0889594916601 hold_6_21 - 412.9554655870446 hold_6_22
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_8_nic_out_7_nic_in_8: switch_nic_out_7_nic_in_8 = 1 ->
   - 1263.157894736842 hold_8_1 - 657.5342465753424 hold_8_2
   - 444.4444444444444 hold_8_3 - 335.6643356643356 hold_8_4
   - 269.6629213483146 hold_8_5 - 225.3521126760563 hold_8_6
   + flow_nic_in_8_8 + flow_9_8 <= 0
 constr_intergroup_throughput_10_nic_out_7_nic_in_10: switch_nic_out_7_nic_in_10 = 1 ->
   - 1815.602836879433 hold_10_1 - 962.406015037594 hold_10_2
   - 654.7314578005115 hold_10_3 - 496.1240310077519 hold_10_4
   + flow_nic_in_10_10 + flow_11_10 <= 0
 constr_intergroup_throughput_12_nic_out_7_nic_in_12: switch_nic_out_7_nic_in_12 = 1 ->
   - 1340.314136125654 hold_12_1 - 699.4535519125683 hold_12_2
   - 473.1977818853974 hold_12_3 - 357.5418994413408 hold_12_4
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_7_nic_in_13: switch_nic_out_7_nic_in_13 = 1 ->
   - 1340.314136125654 hold_13_1 - 699.4535519125683 hold_13_2
   - 473.1977818853974 hold_13_3 - 357.5418994413408 hold_13_4
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_7_nic_in_14: switch_nic_out_7_nic_in_14 = 1 ->
   - 1263.157894736842 hold_14_1 - 657.5342465753424 hold_14_2
   - 444.4444444444444 hold_14_3 - 335.6643356643356 hold_14_4
   - 269.6629213483146 hold_14_5 - 225.3521126760563 hold_14_6
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_7_nic_in_15: switch_nic_out_7_nic_in_15 = 1 ->
   - 3367.47123843898 hold_15_1 - 1881.759737804109 hold_15_2
   - 1305.694043558121 hold_15_3 - 999.665171097569 hold_15_4
   - 809.8518960559865 hold_15_5 - 680.6182464779099 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_8_source_nic_in_8: switch_source_nic_in_8 = 1 ->
   - 1348.314606741573 hold_8_1 - 679.8866855524079 hold_8_2
   - 454.5454545454545 hold_8_3 - 341.3940256045519 hold_8_4
   - 273.3485193621868 hold_8_5 - 227.9202279202279 hold_8_6
   + flow_nic_in_8_8 + flow_9_8 <= 0
 constr_intergroup_throughput_0_nic_out_8_nic_in_0: switch_nic_out_8_nic_in_0 = 1 ->
   - 3367.47123843898 hold_0_1 - 1881.759737804109 hold_0_2
   - 1305.694043558121 hold_0_3 - 999.665171097569 hold_0_4
   - 809.8518960559865 hold_0_5 - 680.6182464779099 hold_0_6
   + flow_nic_in_0_0 + flow_1_0 + flow_2_0 + flow_3_0 <= 0
 constr_intergroup_throughput_4_nic_out_8_nic_in_4: switch_nic_out_8_nic_in_4 = 1 ->
   - 3167.736021998167 hold_4_1 - 1757.884028484232 hold_4_2
   - 1216.473072861668 hold_4_3 - 930.032292787944 hold_4_4
   - 752.7771727292528 hold_4_5 - 632.2722283205268 hold_4_6
   - 545.0244440939914 hold_4_7 - 478.9356984478936 hold_4_8
   - 427.1412680756396 hold_4_9 - 385.4561677448137 hold_4_10
   - 351.1838227822376 hold_4_11 - 322.5083986562149 hold_4_12
   - 298.1623673539816 hold_4_13 + flow_nic_in_4_4 + flow_5_4 <= 0
 constr_intergroup_throughput_6_nic_out_8_nic_in_6: switch_nic_out_8_nic_in_6 = 1 ->
   - 5891.696750902527 hold_6_1 - 3610.619469026549 hold_6_2
   - 2602.870813397129 hold_6_3 - 2034.912718204489 hold_6_4
   - 1670.419651995905 hold_6_5 - 1416.666666666667 hold_6_6
   - 1229.841748304446 hold_6_7 - 1086.551264980027 hold_6_8
   - 973.1663685152056 hold_6_9 - 881.2095032397407 hold_6_10
   - 805.1307350764677 hold_6_11 - 741.1444141689373 hold_6_12
   - 686.5797223390829 hold_6_13 - 639.4984326018808 hold_6_14
   - 598.4598459845985 hold_6_15 - 562.3707787732598 hold_6_16
   - 530.3867403314916 hold_6_17 - 501.8450184501845 hold_6_18
   - 476.2182667055733 hold_6_19 - 453.081621321488 hold_6_20
   - 432.0889594916601 hold_6_21 - 412.9554655870446 hold_6_22
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_8_nic_in_7: switch_nic_out_8_nic_in_7 = 1 ->
   - 7824.817518248176 hold_7_1 - 5178.743961352658 hold_7_2
   - 3870.036101083032 hold_7_3 - 3089.337175792507 hold_7_4
   - 2570.743405275779 hold_7_5 - 2201.23203285421 hold_7_6
   - 1924.5960502693 hold_7_7 - 1709.728867623605 hold_7_8
   - 1538.020086083214 hold_7_9 - 1397.653194263364 hold_7_10
   - 1280.764635603345 hold_7_11 - 1181.918412348401 hold_7_12
   - 1097.236438075742 hold_7_13 - 1023.877745940783 hold_7_14
   - 959.7135183527306 hold_7_15 - 903.117101937658 hold_7_16
   - 852.8241845664279 hold_7_17 - 807.8372268274302 hold_7_18
   - 767.3586256263421 hold_7_19 - 730.7430129516018 hold_7_20
   - 697.4625894599869 hold_7_21 - 667.0815183571873 hold_7_22
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_10_nic_out_8_nic_in_10: switch_nic_out_8_nic_in_10 = 1 ->
   - 1815.602836879433 hold_10_1 - 962.406015037594 hold_10_2
   - 654.7314578005115 hold_10_3 - 496.1240310077519 hold_10_4
   + flow_nic_in_10_10 + flow_11_10 <= 0
 constr_intergroup_throughput_12_nic_out_8_nic_in_12: switch_nic_out_8_nic_in_12 = 1 ->
   - 1340.314136125654 hold_12_1 - 699.4535519125683 hold_12_2
   - 473.1977818853974 hold_12_3 - 357.5418994413408 hold_12_4
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_8_nic_in_13: switch_nic_out_8_nic_in_13 = 1 ->
   - 1340.314136125654 hold_13_1 - 699.4535519125683 hold_13_2
   - 473.1977818853974 hold_13_3 - 357.5418994413408 hold_13_4
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_8_nic_in_14: switch_nic_out_8_nic_in_14 = 1 ->
   - 1263.157894736842 hold_14_1 - 657.5342465753424 hold_14_2
   - 444.4444444444444 hold_14_3 - 335.6643356643356 hold_14_4
   - 269.6629213483146 hold_14_5 - 225.3521126760563 hold_14_6
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_8_nic_in_15: switch_nic_out_8_nic_in_15 = 1 ->
   - 3367.47123843898 hold_15_1 - 1881.759737804109 hold_15_2
   - 1305.694043558121 hold_15_3 - 999.665171097569 hold_15_4
   - 809.8518960559865 hold_15_5 - 680.6182464779099 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_10_source_nic_in_10: switch_source_nic_in_10 = 1 ->
   - 1996.879875195008 hold_10_1 - 1011.058451816746 hold_10_2
   - 676.8905341089371 hold_10_3 - 508.7440381558029 hold_10_4
   + flow_nic_in_10_10 + flow_11_10 <= 0
 constr_intergroup_throughput_0_nic_out_10_nic_in_0: switch_nic_out_10_nic_in_0 = 1 ->
   - 3367.47123843898 hold_0_1 - 1881.759737804109 hold_0_2
   - 1305.694043558121 hold_0_3 - 999.665171097569 hold_0_4
   - 809.8518960559865 hold_0_5 - 680.6182464779099 hold_0_6
   + flow_nic_in_0_0 + flow_1_0 + flow_2_0 + flow_3_0 <= 0
 constr_intergroup_throughput_4_nic_out_10_nic_in_4: switch_nic_out_10_nic_in_4 = 1 ->
   - 3167.736021998167 hold_4_1 - 1757.884028484232 hold_4_2
   - 1216.473072861668 hold_4_3 - 930.032292787944 hold_4_4
   - 752.7771727292528 hold_4_5 - 632.2722283205268 hold_4_6
   - 545.0244440939914 hold_4_7 - 478.9356984478936 hold_4_8
   - 427.1412680756396 hold_4_9 - 385.4561677448137 hold_4_10
   - 351.1838227822376 hold_4_11 - 322.5083986562149 hold_4_12
   - 298.1623673539816 hold_4_13 + flow_nic_in_4_4 + flow_5_4 <= 0
 constr_intergroup_throughput_6_nic_out_10_nic_in_6: switch_nic_out_10_nic_in_6 = 1 ->
   - 5891.696750902527 hold_6_1 - 3610.619469026549 hold_6_2
   - 2602.870813397129 hold_6_3 - 2034.912718204489 hold_6_4
   - 1670.419651995905 hold_6_5 - 1416.666666666667 hold_6_6
   - 1229.841748304446 hold_6_7 - 1086.551264980027 hold_6_8
   - 973.1663685152056 hold_6_9 - 881.2095032397407 hold_6_10
   - 805.1307350764677 hold_6_11 - 741.1444141689373 hold_6_12
   - 686.5797223390829 hold_6_13 - 639.4984326018808 hold_6_14
   - 598.4598459845985 hold_6_15 - 562.3707787732598 hold_6_16
   - 530.3867403314916 hold_6_17 - 501.8450184501845 hold_6_18
   - 476.2182667055733 hold_6_19 - 453.081621321488 hold_6_20
   - 432.0889594916601 hold_6_21 - 412.9554655870446 hold_6_22
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_10_nic_in_7: switch_nic_out_10_nic_in_7 = 1 ->
   - 7824.817518248176 hold_7_1 - 5178.743961352658 hold_7_2
   - 3870.036101083032 hold_7_3 - 3089.337175792507 hold_7_4
   - 2570.743405275779 hold_7_5 - 2201.23203285421 hold_7_6
   - 1924.5960502693 hold_7_7 - 1709.728867623605 hold_7_8
   - 1538.020086083214 hold_7_9 - 1397.653194263364 hold_7_10
   - 1280.764635603345 hold_7_11 - 1181.918412348401 hold_7_12
   - 1097.236438075742 hold_7_13 - 1023.877745940783 hold_7_14
   - 959.7135183527306 hold_7_15 - 903.117101937658 hold_7_16
   - 852.8241845664279 hold_7_17 - 807.8372268274302 hold_7_18
   - 767.3586256263421 hold_7_19 - 730.7430129516018 hold_7_20
   - 697.4625894599869 hold_7_21 - 667.0815183571873 hold_7_22
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_10_nic_in_8: switch_nic_out_10_nic_in_8 = 1 ->
   - 1263.157894736842 hold_8_1 - 657.5342465753424 hold_8_2
   - 444.4444444444444 hold_8_3 - 335.6643356643356 hold_8_4
   - 269.6629213483146 hold_8_5 - 225.3521126760563 hold_8_6
   + flow_nic_in_8_8 + flow_9_8 <= 0
 constr_intergroup_throughput_12_nic_out_10_nic_in_12: switch_nic_out_10_nic_in_12 = 1 ->
   - 1340.314136125654 hold_12_1 - 699.4535519125683 hold_12_2
   - 473.1977818853974 hold_12_3 - 357.5418994413408 hold_12_4
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_10_nic_in_13: switch_nic_out_10_nic_in_13 = 1 ->
   - 1340.314136125654 hold_13_1 - 699.4535519125683 hold_13_2
   - 473.1977818853974 hold_13_3 - 357.5418994413408 hold_13_4
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_10_nic_in_14: switch_nic_out_10_nic_in_14 = 1 ->
   - 1263.157894736842 hold_14_1 - 657.5342465753424 hold_14_2
   - 444.4444444444444 hold_14_3 - 335.6643356643356 hold_14_4
   - 269.6629213483146 hold_14_5 - 225.3521126760563 hold_14_6
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_10_nic_in_15: switch_nic_out_10_nic_in_15 = 1 ->
   - 3367.47123843898 hold_15_1 - 1881.759737804109 hold_15_2
   - 1305.694043558121 hold_15_3 - 999.665171097569 hold_15_4
   - 809.8518960559865 hold_15_5 - 680.6182464779099 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_12_source_nic_in_12: switch_source_nic_in_12 = 1 ->
   - 1436.58810325477 hold_12_1 - 724.8018120045301 hold_12_2
   - 484.66489965922 hold_12_3 - 364.0500568828214 hold_12_4
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_0_nic_out_12_nic_in_0: switch_nic_out_12_nic_in_0 = 1 ->
   - 3367.47123843898 hold_0_1 - 1881.759737804109 hold_0_2
   - 1305.694043558121 hold_0_3 - 999.665171097569 hold_0_4
   - 809.8518960559865 hold_0_5 - 680.6182464779099 hold_0_6
   + flow_nic_in_0_0 + flow_1_0 + flow_2_0 + flow_3_0 <= 0
 constr_intergroup_throughput_4_nic_out_12_nic_in_4: switch_nic_out_12_nic_in_4 = 1 ->
   - 3167.736021998167 hold_4_1 - 1757.884028484232 hold_4_2
   - 1216.473072861668 hold_4_3 - 930.032292787944 hold_4_4
   - 752.7771727292528 hold_4_5 - 632.2722283205268 hold_4_6
   - 545.0244440939914 hold_4_7 - 478.9356984478936 hold_4_8
   - 427.1412680756396 hold_4_9 - 385.4561677448137 hold_4_10
   - 351.1838227822376 hold_4_11 - 322.5083986562149 hold_4_12
   - 298.1623673539816 hold_4_13 + flow_nic_in_4_4 + flow_5_4 <= 0
 constr_intergroup_throughput_6_nic_out_12_nic_in_6: switch_nic_out_12_nic_in_6 = 1 ->
   - 5891.696750902527 hold_6_1 - 3610.619469026549 hold_6_2
   - 2602.870813397129 hold_6_3 - 2034.912718204489 hold_6_4
   - 1670.419651995905 hold_6_5 - 1416.666666666667 hold_6_6
   - 1229.841748304446 hold_6_7 - 1086.551264980027 hold_6_8
   - 973.1663685152056 hold_6_9 - 881.2095032397407 hold_6_10
   - 805.1307350764677 hold_6_11 - 741.1444141689373 hold_6_12
   - 686.5797223390829 hold_6_13 - 639.4984326018808 hold_6_14
   - 598.4598459845985 hold_6_15 - 562.3707787732598 hold_6_16
   - 530.3867403314916 hold_6_17 - 501.8450184501845 hold_6_18
   - 476.2182667055733 hold_6_19 - 453.081621321488 hold_6_20
   - 432.0889594916601 hold_6_21 - 412.9554655870446 hold_6_22
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_12_nic_in_7: switch_nic_out_12_nic_in_7 = 1 ->
   - 7824.817518248176 hold_7_1 - 5178.743961352658 hold_7_2
   - 3870.036101083032 hold_7_3 - 3089.337175792507 hold_7_4
   - 2570.743405275779 hold_7_5 - 2201.23203285421 hold_7_6
   - 1924.5960502693 hold_7_7 - 1709.728867623605 hold_7_8
   - 1538.020086083214 hold_7_9 - 1397.653194263364 hold_7_10
   - 1280.764635603345 hold_7_11 - 1181.918412348401 hold_7_12
   - 1097.236438075742 hold_7_13 - 1023.877745940783 hold_7_14
   - 959.7135183527306 hold_7_15 - 903.117101937658 hold_7_16
   - 852.8241845664279 hold_7_17 - 807.8372268274302 hold_7_18
   - 767.3586256263421 hold_7_19 - 730.7430129516018 hold_7_20
   - 697.4625894599869 hold_7_21 - 667.0815183571873 hold_7_22
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_12_nic_in_8: switch_nic_out_12_nic_in_8 = 1 ->
   - 1263.157894736842 hold_8_1 - 657.5342465753424 hold_8_2
   - 444.4444444444444 hold_8_3 - 335.6643356643356 hold_8_4
   - 269.6629213483146 hold_8_5 - 225.3521126760563 hold_8_6
   + flow_nic_in_8_8 + flow_9_8 <= 0
 constr_intergroup_throughput_10_nic_out_12_nic_in_10: switch_nic_out_12_nic_in_10 = 1 ->
   - 1815.602836879433 hold_10_1 - 962.406015037594 hold_10_2
   - 654.7314578005115 hold_10_3 - 496.1240310077519 hold_10_4
   + flow_nic_in_10_10 + flow_11_10 <= 0
 constr_intergroup_throughput_13_nic_out_12_nic_in_13: switch_nic_out_12_nic_in_13 = 1 ->
   - 1340.314136125654 hold_13_1 - 699.4535519125683 hold_13_2
   - 473.1977818853974 hold_13_3 - 357.5418994413408 hold_13_4
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_12_nic_in_14: switch_nic_out_12_nic_in_14 = 1 ->
   - 1263.157894736842 hold_14_1 - 657.5342465753424 hold_14_2
   - 444.4444444444444 hold_14_3 - 335.6643356643356 hold_14_4
   - 269.6629213483146 hold_14_5 - 225.3521126760563 hold_14_6
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_12_nic_in_15: switch_nic_out_12_nic_in_15 = 1 ->
   - 3367.47123843898 hold_15_1 - 1881.759737804109 hold_15_2
   - 1305.694043558121 hold_15_3 - 999.665171097569 hold_15_4
   - 809.8518960559865 hold_15_5 - 680.6182464779099 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_13_source_nic_in_13: switch_source_nic_in_13 = 1 ->
   - 1436.58810325477 hold_13_1 - 724.8018120045301 hold_13_2
   - 484.66489965922 hold_13_3 - 364.0500568828214 hold_13_4
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_0_nic_out_13_nic_in_0: switch_nic_out_13_nic_in_0 = 1 ->
   - 3367.47123843898 hold_0_1 - 1881.759737804109 hold_0_2
   - 1305.694043558121 hold_0_3 - 999.665171097569 hold_0_4
   - 809.8518960559865 hold_0_5 - 680.6182464779099 hold_0_6
   + flow_nic_in_0_0 + flow_1_0 + flow_2_0 + flow_3_0 <= 0
 constr_intergroup_throughput_4_nic_out_13_nic_in_4: switch_nic_out_13_nic_in_4 = 1 ->
   - 3167.736021998167 hold_4_1 - 1757.884028484232 hold_4_2
   - 1216.473072861668 hold_4_3 - 930.032292787944 hold_4_4
   - 752.7771727292528 hold_4_5 - 632.2722283205268 hold_4_6
   - 545.0244440939914 hold_4_7 - 478.9356984478936 hold_4_8
   - 427.1412680756396 hold_4_9 - 385.4561677448137 hold_4_10
   - 351.1838227822376 hold_4_11 - 322.5083986562149 hold_4_12
   - 298.1623673539816 hold_4_13 + flow_nic_in_4_4 + flow_5_4 <= 0
 constr_intergroup_throughput_6_nic_out_13_nic_in_6: switch_nic_out_13_nic_in_6 = 1 ->
   - 5891.696750902527 hold_6_1 - 3610.619469026549 hold_6_2
   - 2602.870813397129 hold_6_3 - 2034.912718204489 hold_6_4
   - 1670.419651995905 hold_6_5 - 1416.666666666667 hold_6_6
   - 1229.841748304446 hold_6_7 - 1086.551264980027 hold_6_8
   - 973.1663685152056 hold_6_9 - 881.2095032397407 hold_6_10
   - 805.1307350764677 hold_6_11 - 741.1444141689373 hold_6_12
   - 686.5797223390829 hold_6_13 - 639.4984326018808 hold_6_14
   - 598.4598459845985 hold_6_15 - 562.3707787732598 hold_6_16
   - 530.3867403314916 hold_6_17 - 501.8450184501845 hold_6_18
   - 476.2182667055733 hold_6_19 - 453.081621321488 hold_6_20
   - 432.0889594916601 hold_6_21 - 412.9554655870446 hold_6_22
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_13_nic_in_7: switch_nic_out_13_nic_in_7 = 1 ->
   - 7824.817518248176 hold_7_1 - 5178.743961352658 hold_7_2
   - 3870.036101083032 hold_7_3 - 3089.337175792507 hold_7_4
   - 2570.743405275779 hold_7_5 - 2201.23203285421 hold_7_6
   - 1924.5960502693 hold_7_7 - 1709.728867623605 hold_7_8
   - 1538.020086083214 hold_7_9 - 1397.653194263364 hold_7_10
   - 1280.764635603345 hold_7_11 - 1181.918412348401 hold_7_12
   - 1097.236438075742 hold_7_13 - 1023.877745940783 hold_7_14
   - 959.7135183527306 hold_7_15 - 903.117101937658 hold_7_16
   - 852.8241845664279 hold_7_17 - 807.8372268274302 hold_7_18
   - 767.3586256263421 hold_7_19 - 730.7430129516018 hold_7_20
   - 697.4625894599869 hold_7_21 - 667.0815183571873 hold_7_22
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_13_nic_in_8: switch_nic_out_13_nic_in_8 = 1 ->
   - 1263.157894736842 hold_8_1 - 657.5342465753424 hold_8_2
   - 444.4444444444444 hold_8_3 - 335.6643356643356 hold_8_4
   - 269.6629213483146 hold_8_5 - 225.3521126760563 hold_8_6
   + flow_nic_in_8_8 + flow_9_8 <= 0
 constr_intergroup_throughput_10_nic_out_13_nic_in_10: switch_nic_out_13_nic_in_10 = 1 ->
   - 1815.602836879433 hold_10_1 - 962.406015037594 hold_10_2
   - 654.7314578005115 hold_10_3 - 496.1240310077519 hold_10_4
   + flow_nic_in_10_10 + flow_11_10 <= 0
 constr_intergroup_throughput_12_nic_out_13_nic_in_12: switch_nic_out_13_nic_in_12 = 1 ->
   - 1340.314136125654 hold_12_1 - 699.4535519125683 hold_12_2
   - 473.1977818853974 hold_12_3 - 357.5418994413408 hold_12_4
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_14_nic_out_13_nic_in_14: switch_nic_out_13_nic_in_14 = 1 ->
   - 1263.157894736842 hold_14_1 - 657.5342465753424 hold_14_2
   - 444.4444444444444 hold_14_3 - 335.6643356643356 hold_14_4
   - 269.6629213483146 hold_14_5 - 225.3521126760563 hold_14_6
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_13_nic_in_15: switch_nic_out_13_nic_in_15 = 1 ->
   - 3367.47123843898 hold_15_1 - 1881.759737804109 hold_15_2
   - 1305.694043558121 hold_15_3 - 999.665171097569 hold_15_4
   - 809.8518960559865 hold_15_5 - 680.6182464779099 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_14_source_nic_in_14: switch_source_nic_in_14 = 1 ->
   - 1348.314606741573 hold_14_1 - 679.8866855524079 hold_14_2
   - 454.5454545454545 hold_14_3 - 341.3940256045519 hold_14_4
   - 273.3485193621868 hold_14_5 - 227.9202279202279 hold_14_6
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_0_nic_out_14_nic_in_0: switch_nic_out_14_nic_in_0 = 1 ->
   - 3367.47123843898 hold_0_1 - 1881.759737804109 hold_0_2
   - 1305.694043558121 hold_0_3 - 999.665171097569 hold_0_4
   - 809.8518960559865 hold_0_5 - 680.6182464779099 hold_0_6
   + flow_nic_in_0_0 + flow_1_0 + flow_2_0 + flow_3_0 <= 0
 constr_intergroup_throughput_4_nic_out_14_nic_in_4: switch_nic_out_14_nic_in_4 = 1 ->
   - 3167.736021998167 hold_4_1 - 1757.884028484232 hold_4_2
   - 1216.473072861668 hold_4_3 - 930.032292787944 hold_4_4
   - 752.7771727292528 hold_4_5 - 632.2722283205268 hold_4_6
   - 545.0244440939914 hold_4_7 - 478.9356984478936 hold_4_8
   - 427.1412680756396 hold_4_9 - 385.4561677448137 hold_4_10
   - 351.1838227822376 hold_4_11 - 322.5083986562149 hold_4_12
   - 298.1623673539816 hold_4_13 + flow_nic_in_4_4 + flow_5_4 <= 0
 constr_intergroup_throughput_6_nic_out_14_nic_in_6: switch_nic_out_14_nic_in_6 = 1 ->
   - 5891.696750902527 hold_6_1 - 3610.619469026549 hold_6_2
   - 2602.870813397129 hold_6_3 - 2034.912718204489 hold_6_4
   - 1670.419651995905 hold_6_5 - 1416.666666666667 hold_6_6
   - 1229.841748304446 hold_6_7 - 1086.551264980027 hold_6_8
   - 973.1663685152056 hold_6_9 - 881.2095032397407 hold_6_10
   - 805.1307350764677 hold_6_11 - 741.1444141689373 hold_6_12
   - 686.5797223390829 hold_6_13 - 639.4984326018808 hold_6_14
   - 598.4598459845985 hold_6_15 - 562.3707787732598 hold_6_16
   - 530.3867403314916 hold_6_17 - 501.8450184501845 hold_6_18
   - 476.2182667055733 hold_6_19 - 453.081621321488 hold_6_20
   - 432.0889594916601 hold_6_21 - 412.9554655870446 hold_6_22
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_14_nic_in_7: switch_nic_out_14_nic_in_7 = 1 ->
   - 7824.817518248176 hold_7_1 - 5178.743961352658 hold_7_2
   - 3870.036101083032 hold_7_3 - 3089.337175792507 hold_7_4
   - 2570.743405275779 hold_7_5 - 2201.23203285421 hold_7_6
   - 1924.5960502693 hold_7_7 - 1709.728867623605 hold_7_8
   - 1538.020086083214 hold_7_9 - 1397.653194263364 hold_7_10
   - 1280.764635603345 hold_7_11 - 1181.918412348401 hold_7_12
   - 1097.236438075742 hold_7_13 - 1023.877745940783 hold_7_14
   - 959.7135183527306 hold_7_15 - 903.117101937658 hold_7_16
   - 852.8241845664279 hold_7_17 - 807.8372268274302 hold_7_18
   - 767.3586256263421 hold_7_19 - 730.7430129516018 hold_7_20
   - 697.4625894599869 hold_7_21 - 667.0815183571873 hold_7_22
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_14_nic_in_8: switch_nic_out_14_nic_in_8 = 1 ->
   - 1263.157894736842 hold_8_1 - 657.5342465753424 hold_8_2
   - 444.4444444444444 hold_8_3 - 335.6643356643356 hold_8_4
   - 269.6629213483146 hold_8_5 - 225.3521126760563 hold_8_6
   + flow_nic_in_8_8 + flow_9_8 <= 0
 constr_intergroup_throughput_10_nic_out_14_nic_in_10: switch_nic_out_14_nic_in_10 = 1 ->
   - 1815.602836879433 hold_10_1 - 962.406015037594 hold_10_2
   - 654.7314578005115 hold_10_3 - 496.1240310077519 hold_10_4
   + flow_nic_in_10_10 + flow_11_10 <= 0
 constr_intergroup_throughput_12_nic_out_14_nic_in_12: switch_nic_out_14_nic_in_12 = 1 ->
   - 1340.314136125654 hold_12_1 - 699.4535519125683 hold_12_2
   - 473.1977818853974 hold_12_3 - 357.5418994413408 hold_12_4
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_14_nic_in_13: switch_nic_out_14_nic_in_13 = 1 ->
   - 1340.314136125654 hold_13_1 - 699.4535519125683 hold_13_2
   - 473.1977818853974 hold_13_3 - 357.5418994413408 hold_13_4
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_15_nic_out_14_nic_in_15: switch_nic_out_14_nic_in_15 = 1 ->
   - 3367.47123843898 hold_15_1 - 1881.759737804109 hold_15_2
   - 1305.694043558121 hold_15_3 - 999.665171097569 hold_15_4
   - 809.8518960559865 hold_15_5 - 680.6182464779099 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_15_source_nic_in_15: switch_source_nic_in_15 = 1 ->
   - 4049.259480279933 hold_15_1 - 2077.199231903821 hold_15_2
   - 1396.889562629835 hold_15_3 - 1052.260583931315 hold_15_4
   - 844.0288127735122 hold_15_5 - 704.5963014358132 hold_15_6
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_0_nic_out_15_nic_in_0: switch_nic_out_15_nic_in_0 = 1 ->
   - 3367.47123843898 hold_0_1 - 1881.759737804109 hold_0_2
   - 1305.694043558121 hold_0_3 - 999.665171097569 hold_0_4
   - 809.8518960559865 hold_0_5 - 680.6182464779099 hold_0_6
   + flow_nic_in_0_0 + flow_1_0 + flow_2_0 + flow_3_0 <= 0
 constr_intergroup_throughput_4_nic_out_15_nic_in_4: switch_nic_out_15_nic_in_4 = 1 ->
   - 3167.736021998167 hold_4_1 - 1757.884028484232 hold_4_2
   - 1216.473072861668 hold_4_3 - 930.032292787944 hold_4_4
   - 752.7771727292528 hold_4_5 - 632.2722283205268 hold_4_6
   - 545.0244440939914 hold_4_7 - 478.9356984478936 hold_4_8
   - 427.1412680756396 hold_4_9 - 385.4561677448137 hold_4_10
   - 351.1838227822376 hold_4_11 - 322.5083986562149 hold_4_12
   - 298.1623673539816 hold_4_13 + flow_nic_in_4_4 + flow_5_4 <= 0
 constr_intergroup_throughput_6_nic_out_15_nic_in_6: switch_nic_out_15_nic_in_6 = 1 ->
   - 5891.696750902527 hold_6_1 - 3610.619469026549 hold_6_2
   - 2602.870813397129 hold_6_3 - 2034.912718204489 hold_6_4
   - 1670.419651995905 hold_6_5 - 1416.666666666667 hold_6_6
   - 1229.841748304446 hold_6_7 - 1086.551264980027 hold_6_8
   - 973.1663685152056 hold_6_9 - 881.2095032397407 hold_6_10
   - 805.1307350764677 hold_6_11 - 741.1444141689373 hold_6_12
   - 686.5797223390829 hold_6_13 - 639.4984326018808 hold_6_14
   - 598.4598459845985 hold_6_15 - 562.3707787732598 hold_6_16
   - 530.3867403314916 hold_6_17 - 501.8450184501845 hold_6_18
   - 476.2182667055733 hold_6_19 - 453.081621321488 hold_6_20
   - 432.0889594916601 hold_6_21 - 412.9554655870446 hold_6_22
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_15_nic_in_7: switch_nic_out_15_nic_in_7 = 1 ->
   - 7824.817518248176 hold_7_1 - 5178.743961352658 hold_7_2
   - 3870.036101083032 hold_7_3 - 3089.337175792507 hold_7_4
   - 2570.743405275779 hold_7_5 - 2201.23203285421 hold_7_6
   - 1924.5960502693 hold_7_7 - 1709.728867623605 hold_7_8
   - 1538.020086083214 hold_7_9 - 1397.653194263364 hold_7_10
   - 1280.764635603345 hold_7_11 - 1181.918412348401 hold_7_12
   - 1097.236438075742 hold_7_13 - 1023.877745940783 hold_7_14
   - 959.7135183527306 hold_7_15 - 903.117101937658 hold_7_16
   - 852.8241845664279 hold_7_17 - 807.8372268274302 hold_7_18
   - 767.3586256263421 hold_7_19 - 730.7430129516018 hold_7_20
   - 697.4625894599869 hold_7_21 - 667.0815183571873 hold_7_22
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_15_nic_in_8: switch_nic_out_15_nic_in_8 = 1 ->
   - 1263.157894736842 hold_8_1 - 657.5342465753424 hold_8_2
   - 444.4444444444444 hold_8_3 - 335.6643356643356 hold_8_4
   - 269.6629213483146 hold_8_5 - 225.3521126760563 hold_8_6
   + flow_nic_in_8_8 + flow_9_8 <= 0
 constr_intergroup_throughput_10_nic_out_15_nic_in_10: switch_nic_out_15_nic_in_10 = 1 ->
   - 1815.602836879433 hold_10_1 - 962.406015037594 hold_10_2
   - 654.7314578005115 hold_10_3 - 496.1240310077519 hold_10_4
   + flow_nic_in_10_10 + flow_11_10 <= 0
 constr_intergroup_throughput_12_nic_out_15_nic_in_12: switch_nic_out_15_nic_in_12 = 1 ->
   - 1340.314136125654 hold_12_1 - 699.4535519125683 hold_12_2
   - 473.1977818853974 hold_12_3 - 357.5418994413408 hold_12_4
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_15_nic_in_13: switch_nic_out_15_nic_in_13 = 1 ->
   - 1340.314136125654 hold_13_1 - 699.4535519125683 hold_13_2
   - 473.1977818853974 hold_13_3 - 357.5418994413408 hold_13_4
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_15_nic_in_14: switch_nic_out_15_nic_in_14 = 1 ->
   - 1263.157894736842 hold_14_1 - 657.5342465753424 hold_14_2
   - 444.4444444444444 hold_14_3 - 335.6643356643356 hold_14_4
   - 269.6629213483146 hold_14_5 - 225.3521126760563 hold_14_6
   + flow_nic_in_14_14 <= 0
 inactive_end_not_complete_0: is_active_0 = 0 -> is_end_complete_0 = 0
 inactive_end_not_complete_1: is_active_1 = 0 -> is_end_complete_1 = 0
 inactive_end_not_complete_2: is_active_2 = 0 -> is_end_complete_2 = 0
 inactive_end_not_complete_3: is_active_3 = 0 -> is_end_complete_3 = 0
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
 inactive_end_not_complete_5: is_active_5 = 0 -> is_end_complete_5 = 0
 active_end_val_4: is_active_4 = 1 -> - start_4 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   - 8 hold_4_8 - 9 hold_4_9 - 10 hold_4_10 - 11 hold_4_11 - 12 hold_4_12
   - 13 hold_4_13 - 14 hold_4_14 - 15 hold_4_15 - 16 hold_4_16
   - 17 hold_4_17 - 18 hold_4_18 - 19 hold_4_19 - 20 hold_4_20
   - 21 hold_4_21 - 22 hold_4_22 - 23 hold_4_23 - 24 hold_4_24
   - 25 hold_4_25 - 26 hold_4_26 + val_end_4 = 0
 inactive_end_zero_4: is_active_4 = 0 -> val_end_4 = 0
 active_end_val_5: is_active_5 = 1 -> - start_5 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11 - 12 hold_5_12
   - 13 hold_5_13 - 14 hold_5_14 - 15 hold_5_15 - 16 hold_5_16
   - 17 hold_5_17 - 18 hold_5_18 - 19 hold_5_19 - 20 hold_5_20
   - 21 hold_5_21 - 22 hold_5_22 - 23 hold_5_23 - 24 hold_5_24
   - 25 hold_5_25 - 26 hold_5_26 + val_end_5 = 0
 inactive_end_zero_5: is_active_5 = 0 -> val_end_5 = 0
 inactive_end_not_complete_6: is_active_6 = 0 -> is_end_complete_6 = 0
 active_end_val_6: is_active_6 = 1 -> - start_6 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11 - 12 hold_6_12
   - 13 hold_6_13 - 14 hold_6_14 - 15 hold_6_15 - 16 hold_6_16
   - 17 hold_6_17 - 18 hold_6_18 - 19 hold_6_19 - 20 hold_6_20
   - 21 hold_6_21 - 22 hold_6_22 - 23 hold_6_23 - 24 hold_6_24
   - 25 hold_6_25 - 26 hold_6_26 + val_end_6 = 0
 inactive_end_zero_6: is_active_6 = 0 -> val_end_6 = 0
 inactive_end_not_complete_7: is_active_7 = 0 -> is_end_complete_7 = 0
 active_end_val_7: is_active_7 = 1 -> - start_7 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   - 8 hold_7_8 - 9 hold_7_9 - 10 hold_7_10 - 11 hold_7_11 - 12 hold_7_12
   - 13 hold_7_13 - 14 hold_7_14 - 15 hold_7_15 - 16 hold_7_16
   - 17 hold_7_17 - 18 hold_7_18 - 19 hold_7_19 - 20 hold_7_20
   - 21 hold_7_21 - 22 hold_7_22 - 23 hold_7_23 - 24 hold_7_24
   - 25 hold_7_25 - 26 hold_7_26 + val_end_7 = 0
 inactive_end_zero_7: is_active_7 = 0 -> val_end_7 = 0
 inactive_end_not_complete_8: is_active_8 = 0 -> is_end_complete_8 = 0
 inactive_end_not_complete_9: is_active_9 = 0 -> is_end_complete_9 = 0
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
 inactive_end_not_complete_10: is_active_10 = 0 -> is_end_complete_10 = 0
 inactive_end_not_complete_11: is_active_11 = 0 -> is_end_complete_11 = 0
 active_end_val_10: is_active_10 = 1 -> - start_10 - hold_10_1
   - 2 hold_10_2 - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6
   - 7 hold_10_7 - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10
   - 11 hold_10_11 - 12 hold_10_12 - 13 hold_10_13 - 14 hold_10_14
   - 15 hold_10_15 - 16 hold_10_16 - 17 hold_10_17 - 18 hold_10_18
   - 19 hold_10_19 - 20 hold_10_20 - 21 hold_10_21 - 22 hold_10_22
   - 23 hold_10_23 - 24 hold_10_24 - 25 hold_10_25 - 26 hold_10_26
   + val_end_10 = 0
 inactive_end_zero_10: is_active_10 = 0 -> val_end_10 = 0
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
 active_end_val_12: is_active_12 = 1 -> - start_12 - hold_12_1
   - 2 hold_12_2 - 3 hold_12_3 - 4 hold_12_4 - 5 hold_12_5 - 6 hold_12_6
   - 7 hold_12_7 - 8 hold_12_8 - 9 hold_12_9 - 10 hold_12_10
   - 11 hold_12_11 - 12 hold_12_12 - 13 hold_12_13 - 14 hold_12_14
   - 15 hold_12_15 - 16 hold_12_16 - 17 hold_12_17 - 18 hold_12_18
   - 19 hold_12_19 - 20 hold_12_20 - 21 hold_12_21 - 22 hold_12_22
   - 23 hold_12_23 - 24 hold_12_24 - 25 hold_12_25 - 26 hold_12_26
   + val_end_12 = 0
 inactive_end_zero_12: is_active_12 = 0 -> val_end_12 = 0
 inactive_end_not_complete_13: is_active_13 = 0 -> is_end_complete_13 = 0
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
 group_synch_hold_constr_0_1_25: group_tp_active_0 = 1 -> hold_0_25
   - hold_1_25 = 0
 group_synch_hold_constr_0_2_25: group_tp_active_0 = 1 -> hold_0_25
   - hold_2_25 = 0
 group_synch_hold_constr_0_3_25: group_tp_active_0 = 1 -> hold_0_25
   - hold_3_25 = 0
 group_synch_hold_constr_0_1_26: group_tp_active_0 = 1 -> hold_0_26
   - hold_1_26 = 0
 group_synch_hold_constr_0_2_26: group_tp_active_0 = 1 -> hold_0_26
   - hold_2_26 = 0
 group_synch_hold_constr_0_3_26: group_tp_active_0 = 1 -> hold_0_26
   - hold_3_26 = 0
 tp_on_limit_constr_0_25: group_tp_active_0 = 1 -> hold_0_25 = 0
 tp_on_limit_constr_0_26: group_tp_active_0 = 1 -> hold_0_26 = 0
 tp_on_limit_constr_1_25: group_tp_active_0 = 1 -> hold_1_25 = 0
 tp_on_limit_constr_1_26: group_tp_active_0 = 1 -> hold_1_26 = 0
 tp_on_limit_constr_2_25: group_tp_active_0 = 1 -> hold_2_25 = 0
 tp_on_limit_constr_2_26: group_tp_active_0 = 1 -> hold_2_26 = 0
 tp_on_limit_constr_3_25: group_tp_active_0 = 1 -> hold_3_25 = 0
 tp_on_limit_constr_3_26: group_tp_active_0 = 1 -> hold_3_26 = 0
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
 tp_off_limit_constr_2_25: group_tp_active_0 = 0 -> hold_2_25 = 0
 tp_off_limit_constr_2_26: group_tp_active_0 = 0 -> hold_2_26 = 0
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
 tp_off_limit_constr_3_25: group_tp_active_0 = 0 -> hold_3_25 = 0
 tp_off_limit_constr_3_26: group_tp_active_0 = 0 -> hold_3_26 = 0
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
 group_synch_hold_constr_4_5_25: group_tp_active_4 = 1 -> hold_4_25
   - hold_5_25 = 0
 group_synch_hold_constr_4_5_26: group_tp_active_4 = 1 -> hold_4_26
   - hold_5_26 = 0
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
 tp_off_limit_constr_4_25: group_tp_active_4 = 0 -> hold_4_25 = 0
 tp_off_limit_constr_4_26: group_tp_active_4 = 0 -> hold_4_26 = 0
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
 tp_off_limit_constr_5_25: group_tp_active_4 = 0 -> hold_5_25 = 0
 tp_off_limit_constr_5_26: group_tp_active_4 = 0 -> hold_5_26 = 0
 tp_off_limit_constr_6_23: group_tp_active_6 = 0 -> hold_6_23 = 0
 tp_off_limit_constr_6_24: group_tp_active_6 = 0 -> hold_6_24 = 0
 tp_off_limit_constr_6_25: group_tp_active_6 = 0 -> hold_6_25 = 0
 tp_off_limit_constr_6_26: group_tp_active_6 = 0 -> hold_6_26 = 0
 tp_off_limit_constr_7_23: group_tp_active_7 = 0 -> hold_7_23 = 0
 tp_off_limit_constr_7_24: group_tp_active_7 = 0 -> hold_7_24 = 0
 tp_off_limit_constr_7_25: group_tp_active_7 = 0 -> hold_7_25 = 0
 tp_off_limit_constr_7_26: group_tp_active_7 = 0 -> hold_7_26 = 0
 group_synch_start_constr_8_9: group_tp_active_8 = 1 -> start_8 - start_9
   = 0
 group_synch_hold_constr_8_9_1: group_tp_active_8 = 1 -> hold_8_1
   - hold_9_1 = 0
 group_synch_hold_constr_8_9_2: group_tp_active_8 = 1 -> hold_8_2
   - hold_9_2 = 0
 group_synch_hold_constr_8_9_3: group_tp_active_8 = 1 -> hold_8_3
   - hold_9_3 = 0
 group_synch_hold_constr_8_9_4: group_tp_active_8 = 1 -> hold_8_4
   - hold_9_4 = 0
 group_synch_hold_constr_8_9_5: group_tp_active_8 = 1 -> hold_8_5
   - hold_9_5 = 0
 group_synch_hold_constr_8_9_6: group_tp_active_8 = 1 -> hold_8_6
   - hold_9_6 = 0
 group_synch_hold_constr_8_9_7: group_tp_active_8 = 1 -> hold_8_7
   - hold_9_7 = 0
 group_synch_hold_constr_8_9_8: group_tp_active_8 = 1 -> hold_8_8
   - hold_9_8 = 0
 group_synch_hold_constr_8_9_9: group_tp_active_8 = 1 -> hold_8_9
   - hold_9_9 = 0
 group_synch_hold_constr_8_9_10: group_tp_active_8 = 1 -> hold_8_10
   - hold_9_10 = 0
 group_synch_hold_constr_8_9_11: group_tp_active_8 = 1 -> hold_8_11
   - hold_9_11 = 0
 group_synch_hold_constr_8_9_12: group_tp_active_8 = 1 -> hold_8_12
   - hold_9_12 = 0
 group_synch_hold_constr_8_9_13: group_tp_active_8 = 1 -> hold_8_13
   - hold_9_13 = 0
 group_synch_hold_constr_8_9_14: group_tp_active_8 = 1 -> hold_8_14
   - hold_9_14 = 0
 group_synch_hold_constr_8_9_15: group_tp_active_8 = 1 -> hold_8_15
   - hold_9_15 = 0
 group_synch_hold_constr_8_9_16: group_tp_active_8 = 1 -> hold_8_16
   - hold_9_16 = 0
 group_synch_hold_constr_8_9_17: group_tp_active_8 = 1 -> hold_8_17
   - hold_9_17 = 0
 group_synch_hold_constr_8_9_18: group_tp_active_8 = 1 -> hold_8_18
   - hold_9_18 = 0
 group_synch_hold_constr_8_9_19: group_tp_active_8 = 1 -> hold_8_19
   - hold_9_19 = 0
 group_synch_hold_constr_8_9_20: group_tp_active_8 = 1 -> hold_8_20
   - hold_9_20 = 0
 group_synch_hold_constr_8_9_21: group_tp_active_8 = 1 -> hold_8_21
   - hold_9_21 = 0
 group_synch_hold_constr_8_9_22: group_tp_active_8 = 1 -> hold_8_22
   - hold_9_22 = 0
 group_synch_hold_constr_8_9_23: group_tp_active_8 = 1 -> hold_8_23
   - hold_9_23 = 0
 group_synch_hold_constr_8_9_24: group_tp_active_8 = 1 -> hold_8_24
   - hold_9_24 = 0
 group_synch_hold_constr_8_9_25: group_tp_active_8 = 1 -> hold_8_25
   - hold_9_25 = 0
 group_synch_hold_constr_8_9_26: group_tp_active_8 = 1 -> hold_8_26
   - hold_9_26 = 0
 tp_on_limit_constr_8_13: group_tp_active_8 = 1 -> hold_8_13 = 0
 tp_on_limit_constr_8_14: group_tp_active_8 = 1 -> hold_8_14 = 0
 tp_on_limit_constr_8_15: group_tp_active_8 = 1 -> hold_8_15 = 0
 tp_on_limit_constr_8_16: group_tp_active_8 = 1 -> hold_8_16 = 0
 tp_on_limit_constr_8_17: group_tp_active_8 = 1 -> hold_8_17 = 0
 tp_on_limit_constr_8_18: group_tp_active_8 = 1 -> hold_8_18 = 0
 tp_on_limit_constr_8_19: group_tp_active_8 = 1 -> hold_8_19 = 0
 tp_on_limit_constr_8_20: group_tp_active_8 = 1 -> hold_8_20 = 0
 tp_on_limit_constr_8_21: group_tp_active_8 = 1 -> hold_8_21 = 0
 tp_on_limit_constr_8_22: group_tp_active_8 = 1 -> hold_8_22 = 0
 tp_on_limit_constr_8_23: group_tp_active_8 = 1 -> hold_8_23 = 0
 tp_on_limit_constr_8_24: group_tp_active_8 = 1 -> hold_8_24 = 0
 tp_on_limit_constr_8_25: group_tp_active_8 = 1 -> hold_8_25 = 0
 tp_on_limit_constr_8_26: group_tp_active_8 = 1 -> hold_8_26 = 0
 tp_on_limit_constr_9_13: group_tp_active_8 = 1 -> hold_9_13 = 0
 tp_on_limit_constr_9_14: group_tp_active_8 = 1 -> hold_9_14 = 0
 tp_on_limit_constr_9_15: group_tp_active_8 = 1 -> hold_9_15 = 0
 tp_on_limit_constr_9_16: group_tp_active_8 = 1 -> hold_9_16 = 0
 tp_on_limit_constr_9_17: group_tp_active_8 = 1 -> hold_9_17 = 0
 tp_on_limit_constr_9_18: group_tp_active_8 = 1 -> hold_9_18 = 0
 tp_on_limit_constr_9_19: group_tp_active_8 = 1 -> hold_9_19 = 0
 tp_on_limit_constr_9_20: group_tp_active_8 = 1 -> hold_9_20 = 0
 tp_on_limit_constr_9_21: group_tp_active_8 = 1 -> hold_9_21 = 0
 tp_on_limit_constr_9_22: group_tp_active_8 = 1 -> hold_9_22 = 0
 tp_on_limit_constr_9_23: group_tp_active_8 = 1 -> hold_9_23 = 0
 tp_on_limit_constr_9_24: group_tp_active_8 = 1 -> hold_9_24 = 0
 tp_on_limit_constr_9_25: group_tp_active_8 = 1 -> hold_9_25 = 0
 tp_on_limit_constr_9_26: group_tp_active_8 = 1 -> hold_9_26 = 0
 tp_off_limit_constr_8_1: group_tp_active_8 = 0 -> hold_8_1 = 0
 tp_off_limit_constr_8_2: group_tp_active_8 = 0 -> hold_8_2 = 0
 tp_off_limit_constr_8_3: group_tp_active_8 = 0 -> hold_8_3 = 0
 tp_off_limit_constr_8_4: group_tp_active_8 = 0 -> hold_8_4 = 0
 tp_off_limit_constr_8_5: group_tp_active_8 = 0 -> hold_8_5 = 0
 tp_off_limit_constr_8_6: group_tp_active_8 = 0 -> hold_8_6 = 0
 tp_off_limit_constr_8_7: group_tp_active_8 = 0 -> hold_8_7 = 0
 tp_off_limit_constr_8_8: group_tp_active_8 = 0 -> hold_8_8 = 0
 tp_off_limit_constr_8_9: group_tp_active_8 = 0 -> hold_8_9 = 0
 tp_off_limit_constr_8_10: group_tp_active_8 = 0 -> hold_8_10 = 0
 tp_off_limit_constr_8_11: group_tp_active_8 = 0 -> hold_8_11 = 0
 tp_off_limit_constr_8_12: group_tp_active_8 = 0 -> hold_8_12 = 0
 tp_off_limit_constr_8_13: group_tp_active_8 = 0 -> hold_8_13 = 0
 tp_off_limit_constr_8_14: group_tp_active_8 = 0 -> hold_8_14 = 0
 tp_off_limit_constr_8_15: group_tp_active_8 = 0 -> hold_8_15 = 0
 tp_off_limit_constr_8_16: group_tp_active_8 = 0 -> hold_8_16 = 0
 tp_off_limit_constr_8_17: group_tp_active_8 = 0 -> hold_8_17 = 0
 tp_off_limit_constr_8_18: group_tp_active_8 = 0 -> hold_8_18 = 0
 tp_off_limit_constr_8_19: group_tp_active_8 = 0 -> hold_8_19 = 0
 tp_off_limit_constr_8_20: group_tp_active_8 = 0 -> hold_8_20 = 0
 tp_off_limit_constr_8_21: group_tp_active_8 = 0 -> hold_8_21 = 0
 tp_off_limit_constr_8_22: group_tp_active_8 = 0 -> hold_8_22 = 0
 tp_off_limit_constr_8_23: group_tp_active_8 = 0 -> hold_8_23 = 0
 tp_off_limit_constr_8_24: group_tp_active_8 = 0 -> hold_8_24 = 0
 tp_off_limit_constr_8_25: group_tp_active_8 = 0 -> hold_8_25 = 0
 tp_off_limit_constr_8_26: group_tp_active_8 = 0 -> hold_8_26 = 0
 tp_off_limit_constr_9_1: group_tp_active_8 = 0 -> hold_9_1 = 0
 tp_off_limit_constr_9_2: group_tp_active_8 = 0 -> hold_9_2 = 0
 tp_off_limit_constr_9_3: group_tp_active_8 = 0 -> hold_9_3 = 0
 tp_off_limit_constr_9_4: group_tp_active_8 = 0 -> hold_9_4 = 0
 tp_off_limit_constr_9_5: group_tp_active_8 = 0 -> hold_9_5 = 0
 tp_off_limit_constr_9_6: group_tp_active_8 = 0 -> hold_9_6 = 0
 tp_off_limit_constr_9_7: group_tp_active_8 = 0 -> hold_9_7 = 0
 tp_off_limit_constr_9_8: group_tp_active_8 = 0 -> hold_9_8 = 0
 tp_off_limit_constr_9_9: group_tp_active_8 = 0 -> hold_9_9 = 0
 tp_off_limit_constr_9_10: group_tp_active_8 = 0 -> hold_9_10 = 0
 tp_off_limit_constr_9_11: group_tp_active_8 = 0 -> hold_9_11 = 0
 tp_off_limit_constr_9_12: group_tp_active_8 = 0 -> hold_9_12 = 0
 tp_off_limit_constr_9_13: group_tp_active_8 = 0 -> hold_9_13 = 0
 tp_off_limit_constr_9_14: group_tp_active_8 = 0 -> hold_9_14 = 0
 tp_off_limit_constr_9_15: group_tp_active_8 = 0 -> hold_9_15 = 0
 tp_off_limit_constr_9_16: group_tp_active_8 = 0 -> hold_9_16 = 0
 tp_off_limit_constr_9_17: group_tp_active_8 = 0 -> hold_9_17 = 0
 tp_off_limit_constr_9_18: group_tp_active_8 = 0 -> hold_9_18 = 0
 tp_off_limit_constr_9_19: group_tp_active_8 = 0 -> hold_9_19 = 0
 tp_off_limit_constr_9_20: group_tp_active_8 = 0 -> hold_9_20 = 0
 tp_off_limit_constr_9_21: group_tp_active_8 = 0 -> hold_9_21 = 0
 tp_off_limit_constr_9_22: group_tp_active_8 = 0 -> hold_9_22 = 0
 tp_off_limit_constr_9_23: group_tp_active_8 = 0 -> hold_9_23 = 0
 tp_off_limit_constr_9_24: group_tp_active_8 = 0 -> hold_9_24 = 0
 tp_off_limit_constr_9_25: group_tp_active_8 = 0 -> hold_9_25 = 0
 tp_off_limit_constr_9_26: group_tp_active_8 = 0 -> hold_9_26 = 0
 group_synch_start_constr_10_11: group_tp_active_10 = 1 -> start_10
   - start_11 = 0
 group_synch_hold_constr_10_11_1: group_tp_active_10 = 1 -> hold_10_1
   - hold_11_1 = 0
 group_synch_hold_constr_10_11_2: group_tp_active_10 = 1 -> hold_10_2
   - hold_11_2 = 0
 group_synch_hold_constr_10_11_3: group_tp_active_10 = 1 -> hold_10_3
   - hold_11_3 = 0
 group_synch_hold_constr_10_11_4: group_tp_active_10 = 1 -> hold_10_4
   - hold_11_4 = 0
 group_synch_hold_constr_10_11_5: group_tp_active_10 = 1 -> hold_10_5
   - hold_11_5 = 0
 group_synch_hold_constr_10_11_6: group_tp_active_10 = 1 -> hold_10_6
   - hold_11_6 = 0
 group_synch_hold_constr_10_11_7: group_tp_active_10 = 1 -> hold_10_7
   - hold_11_7 = 0
 group_synch_hold_constr_10_11_8: group_tp_active_10 = 1 -> hold_10_8
   - hold_11_8 = 0
 group_synch_hold_constr_10_11_9: group_tp_active_10 = 1 -> hold_10_9
   - hold_11_9 = 0
 group_synch_hold_constr_10_11_10: group_tp_active_10 = 1 -> hold_10_10
   - hold_11_10 = 0
 group_synch_hold_constr_10_11_11: group_tp_active_10 = 1 -> hold_10_11
   - hold_11_11 = 0
 group_synch_hold_constr_10_11_12: group_tp_active_10 = 1 -> hold_10_12
   - hold_11_12 = 0
 group_synch_hold_constr_10_11_13: group_tp_active_10 = 1 -> hold_10_13
   - hold_11_13 = 0
 group_synch_hold_constr_10_11_14: group_tp_active_10 = 1 -> hold_10_14
   - hold_11_14 = 0
 group_synch_hold_constr_10_11_15: group_tp_active_10 = 1 -> hold_10_15
   - hold_11_15 = 0
 group_synch_hold_constr_10_11_16: group_tp_active_10 = 1 -> hold_10_16
   - hold_11_16 = 0
 group_synch_hold_constr_10_11_17: group_tp_active_10 = 1 -> hold_10_17
   - hold_11_17 = 0
 group_synch_hold_constr_10_11_18: group_tp_active_10 = 1 -> hold_10_18
   - hold_11_18 = 0
 group_synch_hold_constr_10_11_19: group_tp_active_10 = 1 -> hold_10_19
   - hold_11_19 = 0
 group_synch_hold_constr_10_11_20: group_tp_active_10 = 1 -> hold_10_20
   - hold_11_20 = 0
 group_synch_hold_constr_10_11_21: group_tp_active_10 = 1 -> hold_10_21
   - hold_11_21 = 0
 group_synch_hold_constr_10_11_22: group_tp_active_10 = 1 -> hold_10_22
   - hold_11_22 = 0
 group_synch_hold_constr_10_11_23: group_tp_active_10 = 1 -> hold_10_23
   - hold_11_23 = 0
 group_synch_hold_constr_10_11_24: group_tp_active_10 = 1 -> hold_10_24
   - hold_11_24 = 0
 group_synch_hold_constr_10_11_25: group_tp_active_10 = 1 -> hold_10_25
   - hold_11_25 = 0
 group_synch_hold_constr_10_11_26: group_tp_active_10 = 1 -> hold_10_26
   - hold_11_26 = 0
 tp_on_limit_constr_10_9: group_tp_active_10 = 1 -> hold_10_9 = 0
 tp_on_limit_constr_10_10: group_tp_active_10 = 1 -> hold_10_10 = 0
 tp_on_limit_constr_10_11: group_tp_active_10 = 1 -> hold_10_11 = 0
 tp_on_limit_constr_10_12: group_tp_active_10 = 1 -> hold_10_12 = 0
 tp_on_limit_constr_10_13: group_tp_active_10 = 1 -> hold_10_13 = 0
 tp_on_limit_constr_10_14: group_tp_active_10 = 1 -> hold_10_14 = 0
 tp_on_limit_constr_10_15: group_tp_active_10 = 1 -> hold_10_15 = 0
 tp_on_limit_constr_10_16: group_tp_active_10 = 1 -> hold_10_16 = 0
 tp_on_limit_constr_10_17: group_tp_active_10 = 1 -> hold_10_17 = 0
 tp_on_limit_constr_10_18: group_tp_active_10 = 1 -> hold_10_18 = 0
 tp_on_limit_constr_10_19: group_tp_active_10 = 1 -> hold_10_19 = 0
 tp_on_limit_constr_10_20: group_tp_active_10 = 1 -> hold_10_20 = 0
 tp_on_limit_constr_10_21: group_tp_active_10 = 1 -> hold_10_21 = 0
 tp_on_limit_constr_10_22: group_tp_active_10 = 1 -> hold_10_22 = 0
 tp_on_limit_constr_10_23: group_tp_active_10 = 1 -> hold_10_23 = 0
 tp_on_limit_constr_10_24: group_tp_active_10 = 1 -> hold_10_24 = 0
 tp_on_limit_constr_10_25: group_tp_active_10 = 1 -> hold_10_25 = 0
 tp_on_limit_constr_10_26: group_tp_active_10 = 1 -> hold_10_26 = 0
 tp_on_limit_constr_11_9: group_tp_active_10 = 1 -> hold_11_9 = 0
 tp_on_limit_constr_11_10: group_tp_active_10 = 1 -> hold_11_10 = 0
 tp_on_limit_constr_11_11: group_tp_active_10 = 1 -> hold_11_11 = 0
 tp_on_limit_constr_11_12: group_tp_active_10 = 1 -> hold_11_12 = 0
 tp_on_limit_constr_11_13: group_tp_active_10 = 1 -> hold_11_13 = 0
 tp_on_limit_constr_11_14: group_tp_active_10 = 1 -> hold_11_14 = 0
 tp_on_limit_constr_11_15: group_tp_active_10 = 1 -> hold_11_15 = 0
 tp_on_limit_constr_11_16: group_tp_active_10 = 1 -> hold_11_16 = 0
 tp_on_limit_constr_11_17: group_tp_active_10 = 1 -> hold_11_17 = 0
 tp_on_limit_constr_11_18: group_tp_active_10 = 1 -> hold_11_18 = 0
 tp_on_limit_constr_11_19: group_tp_active_10 = 1 -> hold_11_19 = 0
 tp_on_limit_constr_11_20: group_tp_active_10 = 1 -> hold_11_20 = 0
 tp_on_limit_constr_11_21: group_tp_active_10 = 1 -> hold_11_21 = 0
 tp_on_limit_constr_11_22: group_tp_active_10 = 1 -> hold_11_22 = 0
 tp_on_limit_constr_11_23: group_tp_active_10 = 1 -> hold_11_23 = 0
 tp_on_limit_constr_11_24: group_tp_active_10 = 1 -> hold_11_24 = 0
 tp_on_limit_constr_11_25: group_tp_active_10 = 1 -> hold_11_25 = 0
 tp_on_limit_constr_11_26: group_tp_active_10 = 1 -> hold_11_26 = 0
 tp_off_limit_constr_10_1: group_tp_active_10 = 0 -> hold_10_1 = 0
 tp_off_limit_constr_10_2: group_tp_active_10 = 0 -> hold_10_2 = 0
 tp_off_limit_constr_10_3: group_tp_active_10 = 0 -> hold_10_3 = 0
 tp_off_limit_constr_10_4: group_tp_active_10 = 0 -> hold_10_4 = 0
 tp_off_limit_constr_10_5: group_tp_active_10 = 0 -> hold_10_5 = 0
 tp_off_limit_constr_10_6: group_tp_active_10 = 0 -> hold_10_6 = 0
 tp_off_limit_constr_10_7: group_tp_active_10 = 0 -> hold_10_7 = 0
 tp_off_limit_constr_10_8: group_tp_active_10 = 0 -> hold_10_8 = 0
 tp_off_limit_constr_10_9: group_tp_active_10 = 0 -> hold_10_9 = 0
 tp_off_limit_constr_10_10: group_tp_active_10 = 0 -> hold_10_10 = 0
 tp_off_limit_constr_10_11: group_tp_active_10 = 0 -> hold_10_11 = 0
 tp_off_limit_constr_10_12: group_tp_active_10 = 0 -> hold_10_12 = 0
 tp_off_limit_constr_10_13: group_tp_active_10 = 0 -> hold_10_13 = 0
 tp_off_limit_constr_10_14: group_tp_active_10 = 0 -> hold_10_14 = 0
 tp_off_limit_constr_10_15: group_tp_active_10 = 0 -> hold_10_15 = 0
 tp_off_limit_constr_10_16: group_tp_active_10 = 0 -> hold_10_16 = 0
 tp_off_limit_constr_10_17: group_tp_active_10 = 0 -> hold_10_17 = 0
 tp_off_limit_constr_10_18: group_tp_active_10 = 0 -> hold_10_18 = 0
 tp_off_limit_constr_10_19: group_tp_active_10 = 0 -> hold_10_19 = 0
 tp_off_limit_constr_10_20: group_tp_active_10 = 0 -> hold_10_20 = 0
 tp_off_limit_constr_10_21: group_tp_active_10 = 0 -> hold_10_21 = 0
 tp_off_limit_constr_10_22: group_tp_active_10 = 0 -> hold_10_22 = 0
 tp_off_limit_constr_10_23: group_tp_active_10 = 0 -> hold_10_23 = 0
 tp_off_limit_constr_10_24: group_tp_active_10 = 0 -> hold_10_24 = 0
 tp_off_limit_constr_10_25: group_tp_active_10 = 0 -> hold_10_25 = 0
 tp_off_limit_constr_10_26: group_tp_active_10 = 0 -> hold_10_26 = 0
 tp_off_limit_constr_11_1: group_tp_active_10 = 0 -> hold_11_1 = 0
 tp_off_limit_constr_11_2: group_tp_active_10 = 0 -> hold_11_2 = 0
 tp_off_limit_constr_11_3: group_tp_active_10 = 0 -> hold_11_3 = 0
 tp_off_limit_constr_11_4: group_tp_active_10 = 0 -> hold_11_4 = 0
 tp_off_limit_constr_11_5: group_tp_active_10 = 0 -> hold_11_5 = 0
 tp_off_limit_constr_11_6: group_tp_active_10 = 0 -> hold_11_6 = 0
 tp_off_limit_constr_11_7: group_tp_active_10 = 0 -> hold_11_7 = 0
 tp_off_limit_constr_11_8: group_tp_active_10 = 0 -> hold_11_8 = 0
 tp_off_limit_constr_11_9: group_tp_active_10 = 0 -> hold_11_9 = 0
 tp_off_limit_constr_11_10: group_tp_active_10 = 0 -> hold_11_10 = 0
 tp_off_limit_constr_11_11: group_tp_active_10 = 0 -> hold_11_11 = 0
 tp_off_limit_constr_11_12: group_tp_active_10 = 0 -> hold_11_12 = 0
 tp_off_limit_constr_11_13: group_tp_active_10 = 0 -> hold_11_13 = 0
 tp_off_limit_constr_11_14: group_tp_active_10 = 0 -> hold_11_14 = 0
 tp_off_limit_constr_11_15: group_tp_active_10 = 0 -> hold_11_15 = 0
 tp_off_limit_constr_11_16: group_tp_active_10 = 0 -> hold_11_16 = 0
 tp_off_limit_constr_11_17: group_tp_active_10 = 0 -> hold_11_17 = 0
 tp_off_limit_constr_11_18: group_tp_active_10 = 0 -> hold_11_18 = 0
 tp_off_limit_constr_11_19: group_tp_active_10 = 0 -> hold_11_19 = 0
 tp_off_limit_constr_11_20: group_tp_active_10 = 0 -> hold_11_20 = 0
 tp_off_limit_constr_11_21: group_tp_active_10 = 0 -> hold_11_21 = 0
 tp_off_limit_constr_11_22: group_tp_active_10 = 0 -> hold_11_22 = 0
 tp_off_limit_constr_11_23: group_tp_active_10 = 0 -> hold_11_23 = 0
 tp_off_limit_constr_11_24: group_tp_active_10 = 0 -> hold_11_24 = 0
 tp_off_limit_constr_11_25: group_tp_active_10 = 0 -> hold_11_25 = 0
 tp_off_limit_constr_11_26: group_tp_active_10 = 0 -> hold_11_26 = 0
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
 tp_off_limit_constr_13_5: group_tp_active_13 = 0 -> hold_13_5 = 0
 tp_off_limit_constr_13_6: group_tp_active_13 = 0 -> hold_13_6 = 0
 tp_off_limit_constr_13_7: group_tp_active_13 = 0 -> hold_13_7 = 0
 tp_off_limit_constr_13_8: group_tp_active_13 = 0 -> hold_13_8 = 0
 tp_off_limit_constr_13_9: group_tp_active_13 = 0 -> hold_13_9 = 0
 tp_off_limit_constr_13_10: group_tp_active_13 = 0 -> hold_13_10 = 0
 tp_off_limit_constr_13_11: group_tp_active_13 = 0 -> hold_13_11 = 0
 tp_off_limit_constr_13_12: group_tp_active_13 = 0 -> hold_13_12 = 0
 tp_off_limit_constr_13_13: group_tp_active_13 = 0 -> hold_13_13 = 0
 tp_off_limit_constr_13_14: group_tp_active_13 = 0 -> hold_13_14 = 0
 tp_off_limit_constr_13_15: group_tp_active_13 = 0 -> hold_13_15 = 0
 tp_off_limit_constr_13_16: group_tp_active_13 = 0 -> hold_13_16 = 0
 tp_off_limit_constr_13_17: group_tp_active_13 = 0 -> hold_13_17 = 0
 tp_off_limit_constr_13_18: group_tp_active_13 = 0 -> hold_13_18 = 0
 tp_off_limit_constr_13_19: group_tp_active_13 = 0 -> hold_13_19 = 0
 tp_off_limit_constr_13_20: group_tp_active_13 = 0 -> hold_13_20 = 0
 tp_off_limit_constr_13_21: group_tp_active_13 = 0 -> hold_13_21 = 0
 tp_off_limit_constr_13_22: group_tp_active_13 = 0 -> hold_13_22 = 0
 tp_off_limit_constr_13_23: group_tp_active_13 = 0 -> hold_13_23 = 0
 tp_off_limit_constr_13_24: group_tp_active_13 = 0 -> hold_13_24 = 0
 tp_off_limit_constr_13_25: group_tp_active_13 = 0 -> hold_13_25 = 0
 tp_off_limit_constr_13_26: group_tp_active_13 = 0 -> hold_13_26 = 0
 tp_off_limit_constr_14_7: group_tp_active_14 = 0 -> hold_14_7 = 0
 tp_off_limit_constr_14_8: group_tp_active_14 = 0 -> hold_14_8 = 0
 tp_off_limit_constr_14_9: group_tp_active_14 = 0 -> hold_14_9 = 0
 tp_off_limit_constr_14_10: group_tp_active_14 = 0 -> hold_14_10 = 0
 tp_off_limit_constr_14_11: group_tp_active_14 = 0 -> hold_14_11 = 0
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
   - 7506.63764007438 hold_4_1 - 3753.31882003719 hold_4_2
   - 2502.21254669146 hold_4_3 - 1876.659410018595 hold_4_4
   - 1501.327528014876 hold_4_5 - 1251.10627334573 hold_4_6
   - 1072.376805724911 hold_4_7 - 938.3297050092975 hold_4_8
   - 834.0708488971533 hold_4_9 - 750.663764007438 hold_4_10
   - 682.4216036431255 hold_4_11 - 625.553136672865 hold_4_12
   - 577.4336646211061 hold_4_13 - 536.1884028624556 hold_4_14
   - 500.442509338292 hold_4_15 - 469.1648525046488 hold_4_16
   - 441.5669200043752 hold_4_17 - 417.0354244485766 hold_4_18
   - 395.0861915828621 hold_4_19 - 375.331882003719 hold_4_20
   - 357.4589352416371 hold_4_21 - 341.2108018215628 hold_4_22
   - 326.3755495684513 hold_4_23 - 312.7765683364325 hold_4_24
   - 300.2655056029752 hold_4_25 - 288.7168323105531 hold_4_26
   + flow_nic_in_4_4 + flow_nic_in_4_5 <= 0
 group_throughput_constr_6: group_tp_active_6 = 1 ->
   - 9325.714285714286 hold_6_1 - 4662.857142857143 hold_6_2
   - 3108.571428571428 hold_6_3 - 2331.428571428572 hold_6_4
   - 1865.142857142857 hold_6_5 - 1554.285714285714 hold_6_6
   - 1332.244897959184 hold_6_7 - 1165.714285714286 hold_6_8
   - 1036.190476190476 hold_6_9 - 932.5714285714284 hold_6_10
   - 847.7922077922078 hold_6_11 - 777.1428571428571 hold_6_12
   - 717.3626373626373 hold_6_13 - 666.1224489795918 hold_6_14
   - 621.7142857142858 hold_6_15 - 582.8571428571429 hold_6_16
   - 548.5714285714286 hold_6_17 - 518.0952380952381 hold_6_18
   - 490.8270676691729 hold_6_19 - 466.2857142857142 hold_6_20
   - 444.0816326530613 hold_6_21 - 423.8961038961039 hold_6_22
   + flow_nic_in_6_6 <= 0
 group_throughput_constr_7: group_tp_active_7 = 1 ->
   - 15314.28571428572 hold_7_1 - 7657.142857142858 hold_7_2
   - 5104.761904761905 hold_7_3 - 3828.571428571429 hold_7_4
   - 3062.857142857143 hold_7_5 - 2552.380952380952 hold_7_6
   - 2187.755102040816 hold_7_7 - 1914.285714285714 hold_7_8
   - 1701.587301587301 hold_7_9 - 1531.428571428571 hold_7_10
   - 1392.207792207792 hold_7_11 - 1276.190476190476 hold_7_12
   - 1178.021978021978 hold_7_13 - 1093.877551020408 hold_7_14
   - 1020.952380952381 hold_7_15 - 957.1428571428572 hold_7_16
   - 900.8403361344537 hold_7_17 - 850.7936507936507 hold_7_18
   - 806.015037593985 hold_7_19 - 765.7142857142857 hold_7_20
   - 729.2517006802721 hold_7_21 - 696.1038961038961 hold_7_22
   + flow_nic_in_7_7 <= 0
 group_throughput_constr_8: group_tp_active_8 = 1 ->
   - 2693.912296991259 hold_8_1 - 1346.95614849563 hold_8_2
   - 897.970765663753 hold_8_3 - 673.4780742478148 hold_8_4
   - 538.7824593982518 hold_8_5 - 448.9853828318765 hold_8_6
   - 384.8446138558941 hold_8_7 - 336.7390371239074 hold_8_8
   - 299.3235885545843 hold_8_9 - 269.3912296991259 hold_8_10
   - 244.9011179082963 hold_8_11 - 224.4926914159383 hold_8_12
   + flow_nic_in_8_8 + flow_nic_in_8_9 <= 0
 group_throughput_constr_10: group_tp_active_10 = 1 ->
   - 3987.803303595952 hold_10_1 - 1993.901651797976 hold_10_2
   - 1329.267767865317 hold_10_3 - 996.950825898988 hold_10_4
   - 797.5606607191904 hold_10_5 - 664.6338839326586 hold_10_6
   - 569.6861862279932 hold_10_7 - 498.475412949494 hold_10_8
   + flow_nic_in_10_10 + flow_nic_in_10_11 <= 0
 group_throughput_constr_12: group_tp_active_12 = 1 ->
   - 1462.857142857143 hold_12_1 - 731.4285714285714 hold_12_2
   - 487.6190476190476 hold_12_3 - 365.7142857142857 hold_12_4
   + flow_nic_in_12_12 <= 0
 group_throughput_constr_13: group_tp_active_13 = 1 ->
   - 1462.857142857143 hold_13_1 - 731.4285714285714 hold_13_2
   - 487.6190476190476 hold_13_3 - 365.7142857142857 hold_13_4
   + flow_nic_in_13_13 <= 0
 group_throughput_constr_14: group_tp_active_14 = 1 ->
   - 1371.428571428571 hold_14_1 - 685.7142857142857 hold_14_2
   - 457.1428571428571 hold_14_3 - 342.8571428571428 hold_14_4
   - 274.2857142857143 hold_14_5 - 228.5714285714286 hold_14_6
   + flow_nic_in_14_14 <= 0
 group_throughput_constr_15: group_tp_active_15 = 1 ->
   - 4265.142857142857 hold_15_1 - 2132.571428571428 hold_15_2
   - 1421.714285714286 hold_15_3 - 1066.285714285714 hold_15_4
   - 853.0285714285714 hold_15_5 - 710.8571428571429 hold_15_6
   + flow_nic_in_15_15 <= 0
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
   - 5109.459042174185 hold_4_1 - 3040.15247603248 hold_4_2
   - 2163.816929787807 hold_4_3 - 1679.651095409346 hold_4_4
   - 1372.538192304928 hold_4_5 - 1160.371981735487 hold_4_6
   - 1005.016998327084 hold_4_7 - 886.3492175210624 hold_4_8
   - 792.7454804093763 hold_4_9 - 717.0235396836346 hold_4_10
   - 654.5060374151892 hold_4_11 - 602.0160715548276 hold_4_12
   - 557.3201993053406 hold_4_13 - 518.8024130345685 hold_4_14
   - 485.2645706247653 hold_4_15 - 455.7995324779789 hold_4_16
   - 429.7078711806942 hold_4_17 - 406.441639349837 hold_4_18
   - 385.5654670831283 hold_4_19 - 366.7290626737931 hold_4_20
   - 349.6473985665369 hold_4_21 - 334.0861883096475 hold_4_22
   - 319.8510763911254 hold_4_23 - 306.7794787980271 hold_4_24
   - 294.7343457685529 hold_4_25 - 283.5993383963553 hold_4_26
   + flow_nic_in_4_4 + flow_nic_in_4_5 <= 0
 intergroup_group_tp_8_via_nic_out_0_nic_in_8: tp_on_and_switch_nic_out_0_nic_in_8_8 = 1 ->
   - 2305.702309237719 hold_8_1 - 1242.367720967523 hold_8_2
   - 850.2519296467542 hold_8_3 - 646.2748288017978 hold_8_4
   - 521.2305906758797 hold_8_5 - 436.730044930787 hold_8_6
   - 375.8054450200391 hold_8_7 - 329.7980448692438 hold_8_8
   - 293.8267585678413 hold_8_9 - 264.9306058433094 hold_8_10
   - 241.2090943547263 hold_8_11 - 221.3864637231713 hold_8_12
   + flow_nic_in_8_8 + flow_nic_in_8_9 <= 0
 intergroup_group_tp_10_via_nic_out_0_nic_in_10: tp_on_and_switch_nic_out_0_nic_in_10_10 = 1 ->
   - 3192.189351095739 hold_10_1 - 1772.957696786115 hold_10_2
   - 1227.304267597741 hold_10_3 - 938.4749875300135 hold_10_4
   - 759.6918879625396 hold_10_5 - 638.1263589100226 hold_10_6
   - 550.0996746229188 hold_10_7 - 483.4147645504219 hold_10_8
   + flow_nic_in_10_10 + flow_nic_in_10_11 <= 0
 intergroup_group_tp_4_via_source_nic_in_4: tp_on_and_switch_source_nic_in_4_4 = 1 ->
   - 6862.690961524641 hold_4_1 - 3585.117698418173 hold_4_2
   - 2426.322853123827 hold_4_3 - 1833.645313368965 hold_4_4
   - 1473.671728842764 hold_4_5 - 1231.841712172382 hold_4_6
   - 1058.192048119833 hold_4_7 - 927.4515136936157 hold_4_8
   - 825.4646488917072 hold_4_9 - 743.6855416581999 hold_4_10
   - 676.6495998303666 hold_4_11 - 620.6996291733516 hold_4_12
   - 573.2956619338331 hold_4_13 - 532.6186038805861 hold_4_14
   - 497.3314369348856 hold_4_15 - 466.429448710796 hold_4_16
   - 439.1430320530621 hold_4_17 - 414.8727167047877 hold_4_18
   - 393.144616467096 hold_4_19 - 373.5791797958418 hold_4_20
   - 355.8688290825181 hold_4_21 - 339.7616724131586 hold_4_22
   - 325.0494471689925 hold_4_23 - 311.5584659885816 hold_4_24
   - 299.1427275736955 hold_4_25 - 287.6786115922728 hold_4_26
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
 intergroup_group_tp_8_via_nic_out_4_nic_in_8: tp_on_and_switch_nic_out_4_nic_in_8_8 = 1 ->
   - 2305.702309237719 hold_8_1 - 1242.367720967523 hold_8_2
   - 850.2519296467542 hold_8_3 - 646.2748288017978 hold_8_4
   - 521.2305906758797 hold_8_5 - 436.730044930787 hold_8_6
   - 375.8054450200391 hold_8_7 - 329.7980448692438 hold_8_8
   - 293.8267585678413 hold_8_9 - 264.9306058433094 hold_8_10
   - 241.2090943547263 hold_8_11 - 221.3864637231713 hold_8_12
   + flow_nic_in_8_8 + flow_nic_in_8_9 <= 0
 intergroup_group_tp_10_via_nic_out_4_nic_in_10: tp_on_and_switch_nic_out_4_nic_in_10_10 = 1 ->
   - 3192.189351095739 hold_10_1 - 1772.957696786115 hold_10_2
   - 1227.304267597741 hold_10_3 - 938.4749875300135 hold_10_4
   - 759.6918879625396 hold_10_5 - 638.1263589100226 hold_10_6
   - 550.0996746229188 hold_10_7 - 483.4147645504219 hold_10_8
   + flow_nic_in_10_10 + flow_nic_in_10_11 <= 0
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
   - 5109.459042174185 hold_4_1 - 3040.15247603248 hold_4_2
   - 2163.816929787807 hold_4_3 - 1679.651095409346 hold_4_4
   - 1372.538192304928 hold_4_5 - 1160.371981735487 hold_4_6
   - 1005.016998327084 hold_4_7 - 886.3492175210624 hold_4_8
   - 792.7454804093763 hold_4_9 - 717.0235396836346 hold_4_10
   - 654.5060374151892 hold_4_11 - 602.0160715548276 hold_4_12
   - 557.3201993053406 hold_4_13 - 518.8024130345685 hold_4_14
   - 485.2645706247653 hold_4_15 - 455.7995324779789 hold_4_16
   - 429.7078711806942 hold_4_17 - 406.441639349837 hold_4_18
   - 385.5654670831283 hold_4_19 - 366.7290626737931 hold_4_20
   - 349.6473985665369 hold_4_21 - 334.0861883096475 hold_4_22
   - 319.8510763911254 hold_4_23 - 306.7794787980271 hold_4_24
   - 294.7343457685529 hold_4_25 - 283.5993383963553 hold_4_26
   + flow_nic_in_4_4 + flow_nic_in_4_5 <= 0
 intergroup_group_tp_8_via_nic_out_6_nic_in_8: tp_on_and_switch_nic_out_6_nic_in_8_8 = 1 ->
   - 2305.702309237719 hold_8_1 - 1242.367720967523 hold_8_2
   - 850.2519296467542 hold_8_3 - 646.2748288017978 hold_8_4
   - 521.2305906758797 hold_8_5 - 436.730044930787 hold_8_6
   - 375.8054450200391 hold_8_7 - 329.7980448692438 hold_8_8
   - 293.8267585678413 hold_8_9 - 264.9306058433094 hold_8_10
   - 241.2090943547263 hold_8_11 - 221.3864637231713 hold_8_12
   + flow_nic_in_8_8 + flow_nic_in_8_9 <= 0
 intergroup_group_tp_10_via_nic_out_6_nic_in_10: tp_on_and_switch_nic_out_6_nic_in_10_10 = 1 ->
   - 3192.189351095739 hold_10_1 - 1772.957696786115 hold_10_2
   - 1227.304267597741 hold_10_3 - 938.4749875300135 hold_10_4
   - 759.6918879625396 hold_10_5 - 638.1263589100226 hold_10_6
   - 550.0996746229188 hold_10_7 - 483.4147645504219 hold_10_8
   + flow_nic_in_10_10 + flow_nic_in_10_11 <= 0
 intergroup_group_tp_0_via_nic_out_7_nic_in_0: tp_on_and_switch_nic_out_7_nic_in_0_0 = 1 ->
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
 intergroup_group_tp_4_via_nic_out_7_nic_in_4: tp_on_and_switch_nic_out_7_nic_in_4_4 = 1 ->
   - 5109.459042174185 hold_4_1 - 3040.15247603248 hold_4_2
   - 2163.816929787807 hold_4_3 - 1679.651095409346 hold_4_4
   - 1372.538192304928 hold_4_5 - 1160.371981735487 hold_4_6
   - 1005.016998327084 hold_4_7 - 886.3492175210624 hold_4_8
   - 792.7454804093763 hold_4_9 - 717.0235396836346 hold_4_10
   - 654.5060374151892 hold_4_11 - 602.0160715548276 hold_4_12
   - 557.3201993053406 hold_4_13 - 518.8024130345685 hold_4_14
   - 485.2645706247653 hold_4_15 - 455.7995324779789 hold_4_16
   - 429.7078711806942 hold_4_17 - 406.441639349837 hold_4_18
   - 385.5654670831283 hold_4_19 - 366.7290626737931 hold_4_20
   - 349.6473985665369 hold_4_21 - 334.0861883096475 hold_4_22
   - 319.8510763911254 hold_4_23 - 306.7794787980271 hold_4_24
   - 294.7343457685529 hold_4_25 - 283.5993383963553 hold_4_26
   + flow_nic_in_4_4 + flow_nic_in_4_5 <= 0
 intergroup_group_tp_8_via_nic_out_7_nic_in_8: tp_on_and_switch_nic_out_7_nic_in_8_8 = 1 ->
   - 2305.702309237719 hold_8_1 - 1242.367720967523 hold_8_2
   - 850.2519296467542 hold_8_3 - 646.2748288017978 hold_8_4
   - 521.2305906758797 hold_8_5 - 436.730044930787 hold_8_6
   - 375.8054450200391 hold_8_7 - 329.7980448692438 hold_8_8
   - 293.8267585678413 hold_8_9 - 264.9306058433094 hold_8_10
   - 241.2090943547263 hold_8_11 - 221.3864637231713 hold_8_12
   + flow_nic_in_8_8 + flow_nic_in_8_9 <= 0
 intergroup_group_tp_10_via_nic_out_7_nic_in_10: tp_on_and_switch_nic_out_7_nic_in_10_10 = 1 ->
   - 3192.189351095739 hold_10_1 - 1772.957696786115 hold_10_2
   - 1227.304267597741 hold_10_3 - 938.4749875300135 hold_10_4
   - 759.6918879625396 hold_10_5 - 638.1263589100226 hold_10_6
   - 550.0996746229188 hold_10_7 - 483.4147645504219 hold_10_8
   + flow_nic_in_10_10 + flow_nic_in_10_11 <= 0
 intergroup_group_tp_8_via_source_nic_in_8: tp_on_and_switch_source_nic_in_8_8 = 1 ->
   - 2606.152953379399 hold_8_1 - 1324.653029216547 hold_8_2
   - 888.0032536439213 hold_8_3 - 667.8557467206057 hold_8_4
   - 535.1781518871273 hold_8_5 - 446.4795976683049 hold_8_6
   - 383.0021598768469 hold_8_7 - 335.3275636127566 hold_8_8
   - 298.2078305797816 hold_8_9 - 268.4871287270489 hold_8_10
   - 244.1536989855088 hold_8_11 - 223.8644921365358 hold_8_12
   + flow_nic_in_8_8 + flow_nic_in_8_9 <= 0
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
   - 5109.459042174185 hold_4_1 - 3040.15247603248 hold_4_2
   - 2163.816929787807 hold_4_3 - 1679.651095409346 hold_4_4
   - 1372.538192304928 hold_4_5 - 1160.371981735487 hold_4_6
   - 1005.016998327084 hold_4_7 - 886.3492175210624 hold_4_8
   - 792.7454804093763 hold_4_9 - 717.0235396836346 hold_4_10
   - 654.5060374151892 hold_4_11 - 602.0160715548276 hold_4_12
   - 557.3201993053406 hold_4_13 - 518.8024130345685 hold_4_14
   - 485.2645706247653 hold_4_15 - 455.7995324779789 hold_4_16
   - 429.7078711806942 hold_4_17 - 406.441639349837 hold_4_18
   - 385.5654670831283 hold_4_19 - 366.7290626737931 hold_4_20
   - 349.6473985665369 hold_4_21 - 334.0861883096475 hold_4_22
   - 319.8510763911254 hold_4_23 - 306.7794787980271 hold_4_24
   - 294.7343457685529 hold_4_25 - 283.5993383963553 hold_4_26
   + flow_nic_in_4_4 + flow_nic_in_4_5 <= 0
 intergroup_group_tp_10_via_nic_out_8_nic_in_10: tp_on_and_switch_nic_out_8_nic_in_10_10 = 1 ->
   - 3192.189351095739 hold_10_1 - 1772.957696786115 hold_10_2
   - 1227.304267597741 hold_10_3 - 938.4749875300135 hold_10_4
   - 759.6918879625396 hold_10_5 - 638.1263589100226 hold_10_6
   - 550.0996746229188 hold_10_7 - 483.4147645504219 hold_10_8
   + flow_nic_in_10_10 + flow_nic_in_10_11 <= 0
 intergroup_group_tp_10_via_source_nic_in_10: tp_on_and_switch_source_nic_in_10_10 = 1 ->
   - 3798.459439812698 hold_10_1 - 1945.414584870413 hold_10_2
   - 1307.54185237166 hold_10_3 - 984.67985842273 hold_10_4
   - 789.6878610662662 hold_10_5 - 659.1576525484435 hold_10_6
   - 565.6580912192966 hold_10_7 - 495.3886745232008 hold_10_8
   + flow_nic_in_10_10 + flow_nic_in_10_11 <= 0
 intergroup_group_tp_0_via_nic_out_10_nic_in_0: tp_on_and_switch_nic_out_10_nic_in_0_0 = 1 ->
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
 intergroup_group_tp_4_via_nic_out_10_nic_in_4: tp_on_and_switch_nic_out_10_nic_in_4_4 = 1 ->
   - 5109.459042174185 hold_4_1 - 3040.15247603248 hold_4_2
   - 2163.816929787807 hold_4_3 - 1679.651095409346 hold_4_4
   - 1372.538192304928 hold_4_5 - 1160.371981735487 hold_4_6
   - 1005.016998327084 hold_4_7 - 886.3492175210624 hold_4_8
   - 792.7454804093763 hold_4_9 - 717.0235396836346 hold_4_10
   - 654.5060374151892 hold_4_11 - 602.0160715548276 hold_4_12
   - 557.3201993053406 hold_4_13 - 518.8024130345685 hold_4_14
   - 485.2645706247653 hold_4_15 - 455.7995324779789 hold_4_16
   - 429.7078711806942 hold_4_17 - 406.441639349837 hold_4_18
   - 385.5654670831283 hold_4_19 - 366.7290626737931 hold_4_20
   - 349.6473985665369 hold_4_21 - 334.0861883096475 hold_4_22
   - 319.8510763911254 hold_4_23 - 306.7794787980271 hold_4_24
   - 294.7343457685529 hold_4_25 - 283.5993383963553 hold_4_26
   + flow_nic_in_4_4 + flow_nic_in_4_5 <= 0
 intergroup_group_tp_8_via_nic_out_10_nic_in_8: tp_on_and_switch_nic_out_10_nic_in_8_8 = 1 ->
   - 2305.702309237719 hold_8_1 - 1242.367720967523 hold_8_2
   - 850.2519296467542 hold_8_3 - 646.2748288017978 hold_8_4
   - 521.2305906758797 hold_8_5 - 436.730044930787 hold_8_6
   - 375.8054450200391 hold_8_7 - 329.7980448692438 hold_8_8
   - 293.8267585678413 hold_8_9 - 264.9306058433094 hold_8_10
   - 241.2090943547263 hold_8_11 - 221.3864637231713 hold_8_12
   + flow_nic_in_8_8 + flow_nic_in_8_9 <= 0
 intergroup_group_tp_0_via_nic_out_12_nic_in_0: tp_on_and_switch_nic_out_12_nic_in_0_0 = 1 ->
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
 intergroup_group_tp_4_via_nic_out_12_nic_in_4: tp_on_and_switch_nic_out_12_nic_in_4_4 = 1 ->
   - 5109.459042174185 hold_4_1 - 3040.15247603248 hold_4_2
   - 2163.816929787807 hold_4_3 - 1679.651095409346 hold_4_4
   - 1372.538192304928 hold_4_5 - 1160.371981735487 hold_4_6
   - 1005.016998327084 hold_4_7 - 886.3492175210624 hold_4_8
   - 792.7454804093763 hold_4_9 - 717.0235396836346 hold_4_10
   - 654.5060374151892 hold_4_11 - 602.0160715548276 hold_4_12
   - 557.3201993053406 hold_4_13 - 518.8024130345685 hold_4_14
   - 485.2645706247653 hold_4_15 - 455.7995324779789 hold_4_16
   - 429.7078711806942 hold_4_17 - 406.441639349837 hold_4_18
   - 385.5654670831283 hold_4_19 - 366.7290626737931 hold_4_20
   - 349.6473985665369 hold_4_21 - 334.0861883096475 hold_4_22
   - 319.8510763911254 hold_4_23 - 306.7794787980271 hold_4_24
   - 294.7343457685529 hold_4_25 - 283.5993383963553 hold_4_26
   + flow_nic_in_4_4 + flow_nic_in_4_5 <= 0
 intergroup_group_tp_8_via_nic_out_12_nic_in_8: tp_on_and_switch_nic_out_12_nic_in_8_8 = 1 ->
   - 2305.702309237719 hold_8_1 - 1242.367720967523 hold_8_2
   - 850.2519296467542 hold_8_3 - 646.2748288017978 hold_8_4
   - 521.2305906758797 hold_8_5 - 436.730044930787 hold_8_6
   - 375.8054450200391 hold_8_7 - 329.7980448692438 hold_8_8
   - 293.8267585678413 hold_8_9 - 264.9306058433094 hold_8_10
   - 241.2090943547263 hold_8_11 - 221.3864637231713 hold_8_12
   + flow_nic_in_8_8 + flow_nic_in_8_9 <= 0
 intergroup_group_tp_10_via_nic_out_12_nic_in_10: tp_on_and_switch_nic_out_12_nic_in_10_10 = 1 ->
   - 3192.189351095739 hold_10_1 - 1772.957696786115 hold_10_2
   - 1227.304267597741 hold_10_3 - 938.4749875300135 hold_10_4
   - 759.6918879625396 hold_10_5 - 638.1263589100226 hold_10_6
   - 550.0996746229188 hold_10_7 - 483.4147645504219 hold_10_8
   + flow_nic_in_10_10 + flow_nic_in_10_11 <= 0
 intergroup_group_tp_0_via_nic_out_13_nic_in_0: tp_on_and_switch_nic_out_13_nic_in_0_0 = 1 ->
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
 intergroup_group_tp_4_via_nic_out_13_nic_in_4: tp_on_and_switch_nic_out_13_nic_in_4_4 = 1 ->
   - 5109.459042174185 hold_4_1 - 3040.15247603248 hold_4_2
   - 2163.816929787807 hold_4_3 - 1679.651095409346 hold_4_4
   - 1372.538192304928 hold_4_5 - 1160.371981735487 hold_4_6
   - 1005.016998327084 hold_4_7 - 886.3492175210624 hold_4_8
   - 792.7454804093763 hold_4_9 - 717.0235396836346 hold_4_10
   - 654.5060374151892 hold_4_11 - 602.0160715548276 hold_4_12
   - 557.3201993053406 hold_4_13 - 518.8024130345685 hold_4_14
   - 485.2645706247653 hold_4_15 - 455.7995324779789 hold_4_16
   - 429.7078711806942 hold_4_17 - 406.441639349837 hold_4_18
   - 385.5654670831283 hold_4_19 - 366.7290626737931 hold_4_20
   - 349.6473985665369 hold_4_21 - 334.0861883096475 hold_4_22
   - 319.8510763911254 hold_4_23 - 306.7794787980271 hold_4_24
   - 294.7343457685529 hold_4_25 - 283.5993383963553 hold_4_26
   + flow_nic_in_4_4 + flow_nic_in_4_5 <= 0
 intergroup_group_tp_8_via_nic_out_13_nic_in_8: tp_on_and_switch_nic_out_13_nic_in_8_8 = 1 ->
   - 2305.702309237719 hold_8_1 - 1242.367720967523 hold_8_2
   - 850.2519296467542 hold_8_3 - 646.2748288017978 hold_8_4
   - 521.2305906758797 hold_8_5 - 436.730044930787 hold_8_6
   - 375.8054450200391 hold_8_7 - 329.7980448692438 hold_8_8
   - 293.8267585678413 hold_8_9 - 264.9306058433094 hold_8_10
   - 241.2090943547263 hold_8_11 - 221.3864637231713 hold_8_12
   + flow_nic_in_8_8 + flow_nic_in_8_9 <= 0
 intergroup_group_tp_10_via_nic_out_13_nic_in_10: tp_on_and_switch_nic_out_13_nic_in_10_10 = 1 ->
   - 3192.189351095739 hold_10_1 - 1772.957696786115 hold_10_2
   - 1227.304267597741 hold_10_3 - 938.4749875300135 hold_10_4
   - 759.6918879625396 hold_10_5 - 638.1263589100226 hold_10_6
   - 550.0996746229188 hold_10_7 - 483.4147645504219 hold_10_8
   + flow_nic_in_10_10 + flow_nic_in_10_11 <= 0
 intergroup_group_tp_0_via_nic_out_14_nic_in_0: tp_on_and_switch_nic_out_14_nic_in_0_0 = 1 ->
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
 intergroup_group_tp_4_via_nic_out_14_nic_in_4: tp_on_and_switch_nic_out_14_nic_in_4_4 = 1 ->
   - 5109.459042174185 hold_4_1 - 3040.15247603248 hold_4_2
   - 2163.816929787807 hold_4_3 - 1679.651095409346 hold_4_4
   - 1372.538192304928 hold_4_5 - 1160.371981735487 hold_4_6
   - 1005.016998327084 hold_4_7 - 886.3492175210624 hold_4_8
   - 792.7454804093763 hold_4_9 - 717.0235396836346 hold_4_10
   - 654.5060374151892 hold_4_11 - 602.0160715548276 hold_4_12
   - 557.3201993053406 hold_4_13 - 518.8024130345685 hold_4_14
   - 485.2645706247653 hold_4_15 - 455.7995324779789 hold_4_16
   - 429.7078711806942 hold_4_17 - 406.441639349837 hold_4_18
   - 385.5654670831283 hold_4_19 - 366.7290626737931 hold_4_20
   - 349.6473985665369 hold_4_21 - 334.0861883096475 hold_4_22
   - 319.8510763911254 hold_4_23 - 306.7794787980271 hold_4_24
   - 294.7343457685529 hold_4_25 - 283.5993383963553 hold_4_26
   + flow_nic_in_4_4 + flow_nic_in_4_5 <= 0
 intergroup_group_tp_8_via_nic_out_14_nic_in_8: tp_on_and_switch_nic_out_14_nic_in_8_8 = 1 ->
   - 2305.702309237719 hold_8_1 - 1242.367720967523 hold_8_2
   - 850.2519296467542 hold_8_3 - 646.2748288017978 hold_8_4
   - 521.2305906758797 hold_8_5 - 436.730044930787 hold_8_6
   - 375.8054450200391 hold_8_7 - 329.7980448692438 hold_8_8
   - 293.8267585678413 hold_8_9 - 264.9306058433094 hold_8_10
   - 241.2090943547263 hold_8_11 - 221.3864637231713 hold_8_12
   + flow_nic_in_8_8 + flow_nic_in_8_9 <= 0
 intergroup_group_tp_10_via_nic_out_14_nic_in_10: tp_on_and_switch_nic_out_14_nic_in_10_10 = 1 ->
   - 3192.189351095739 hold_10_1 - 1772.957696786115 hold_10_2
   - 1227.304267597741 hold_10_3 - 938.4749875300135 hold_10_4
   - 759.6918879625396 hold_10_5 - 638.1263589100226 hold_10_6
   - 550.0996746229188 hold_10_7 - 483.4147645504219 hold_10_8
   + flow_nic_in_10_10 + flow_nic_in_10_11 <= 0
 intergroup_group_tp_0_via_nic_out_15_nic_in_0: tp_on_and_switch_nic_out_15_nic_in_0_0 = 1 ->
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
 intergroup_group_tp_4_via_nic_out_15_nic_in_4: tp_on_and_switch_nic_out_15_nic_in_4_4 = 1 ->
   - 5109.459042174185 hold_4_1 - 3040.15247603248 hold_4_2
   - 2163.816929787807 hold_4_3 - 1679.651095409346 hold_4_4
   - 1372.538192304928 hold_4_5 - 1160.371981735487 hold_4_6
   - 1005.016998327084 hold_4_7 - 886.3492175210624 hold_4_8
   - 792.7454804093763 hold_4_9 - 717.0235396836346 hold_4_10
   - 654.5060374151892 hold_4_11 - 602.0160715548276 hold_4_12
   - 557.3201993053406 hold_4_13 - 518.8024130345685 hold_4_14
   - 485.2645706247653 hold_4_15 - 455.7995324779789 hold_4_16
   - 429.7078711806942 hold_4_17 - 406.441639349837 hold_4_18
   - 385.5654670831283 hold_4_19 - 366.7290626737931 hold_4_20
   - 349.6473985665369 hold_4_21 - 334.0861883096475 hold_4_22
   - 319.8510763911254 hold_4_23 - 306.7794787980271 hold_4_24
   - 294.7343457685529 hold_4_25 - 283.5993383963553 hold_4_26
   + flow_nic_in_4_4 + flow_nic_in_4_5 <= 0
 intergroup_group_tp_8_via_nic_out_15_nic_in_8: tp_on_and_switch_nic_out_15_nic_in_8_8 = 1 ->
   - 2305.702309237719 hold_8_1 - 1242.367720967523 hold_8_2
   - 850.2519296467542 hold_8_3 - 646.2748288017978 hold_8_4
   - 521.2305906758797 hold_8_5 - 436.730044930787 hold_8_6
   - 375.8054450200391 hold_8_7 - 329.7980448692438 hold_8_8
   - 293.8267585678413 hold_8_9 - 264.9306058433094 hold_8_10
   - 241.2090943547263 hold_8_11 - 221.3864637231713 hold_8_12
   + flow_nic_in_8_8 + flow_nic_in_8_9 <= 0
 intergroup_group_tp_10_via_nic_out_15_nic_in_10: tp_on_and_switch_nic_out_15_nic_in_10_10 = 1 ->
   - 3192.189351095739 hold_10_1 - 1772.957696786115 hold_10_2
   - 1227.304267597741 hold_10_3 - 938.4749875300135 hold_10_4
   - 759.6918879625396 hold_10_5 - 638.1263589100226 hold_10_6
   - 550.0996746229188 hold_10_7 - 483.4147645504219 hold_10_8
   + flow_nic_in_10_10 + flow_nic_in_10_11 <= 0
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
 switch_source_nic_in_0 switch_nic_out_0_sink switch_nic_out_0_nic_in_4
 switch_nic_out_0_nic_in_6 switch_nic_out_0_nic_in_7
 switch_nic_out_0_nic_in_8 switch_nic_out_0_nic_in_10
 switch_nic_out_0_nic_in_12 switch_nic_out_0_nic_in_13
 switch_nic_out_0_nic_in_14 switch_nic_out_0_nic_in_15 switch_nic_in_0_0
 switch_0_nic_out_0 switch_nic_in_0_1 switch_1_nic_out_0 switch_nic_in_0_2
 switch_2_nic_out_0 switch_nic_in_0_3 switch_3_nic_out_0 switch_0_1
 switch_0_2 switch_0_3 switch_1_0 switch_1_2 switch_1_3 switch_2_0
 switch_2_1 switch_2_3 switch_3_0 switch_3_1 switch_3_2
 switch_source_nic_in_4 switch_nic_out_4_sink switch_nic_out_4_nic_in_0
 switch_nic_out_4_nic_in_6 switch_nic_out_4_nic_in_7
 switch_nic_out_4_nic_in_8 switch_nic_out_4_nic_in_10
 switch_nic_out_4_nic_in_12 switch_nic_out_4_nic_in_13
 switch_nic_out_4_nic_in_14 switch_nic_out_4_nic_in_15 switch_nic_in_4_4
 switch_4_nic_out_4 switch_nic_in_4_5 switch_5_nic_out_4 switch_4_5
 switch_5_4 switch_source_nic_in_6 switch_nic_out_6_sink
 switch_nic_out_6_nic_in_0 switch_nic_out_6_nic_in_4
 switch_nic_out_6_nic_in_7 switch_nic_out_6_nic_in_8
 switch_nic_out_6_nic_in_10 switch_nic_out_6_nic_in_12
 switch_nic_out_6_nic_in_13 switch_nic_out_6_nic_in_14
 switch_nic_out_6_nic_in_15 switch_nic_in_6_6 switch_6_nic_out_6
 switch_source_nic_in_7 switch_nic_out_7_sink switch_nic_out_7_nic_in_0
 switch_nic_out_7_nic_in_4 switch_nic_out_7_nic_in_6
 switch_nic_out_7_nic_in_8 switch_nic_out_7_nic_in_10
 switch_nic_out_7_nic_in_12 switch_nic_out_7_nic_in_13
 switch_nic_out_7_nic_in_14 switch_nic_out_7_nic_in_15 switch_nic_in_7_7
 switch_7_nic_out_7 switch_source_nic_in_8 switch_nic_out_8_sink
 switch_nic_out_8_nic_in_0 switch_nic_out_8_nic_in_4
 switch_nic_out_8_nic_in_6 switch_nic_out_8_nic_in_7
 switch_nic_out_8_nic_in_10 switch_nic_out_8_nic_in_12
 switch_nic_out_8_nic_in_13 switch_nic_out_8_nic_in_14
 switch_nic_out_8_nic_in_15 switch_nic_in_8_8 switch_8_nic_out_8
 switch_nic_in_8_9 switch_9_nic_out_8 switch_8_9 switch_9_8
 switch_source_nic_in_10 switch_nic_out_10_sink switch_nic_out_10_nic_in_0
 switch_nic_out_10_nic_in_4 switch_nic_out_10_nic_in_6
 switch_nic_out_10_nic_in_7 switch_nic_out_10_nic_in_8
 switch_nic_out_10_nic_in_12 switch_nic_out_10_nic_in_13
 switch_nic_out_10_nic_in_14 switch_nic_out_10_nic_in_15
 switch_nic_in_10_10 switch_10_nic_out_10 switch_nic_in_10_11
 switch_11_nic_out_10 switch_10_11 switch_11_10 switch_source_nic_in_12
 switch_nic_out_12_sink switch_nic_out_12_nic_in_0
 switch_nic_out_12_nic_in_4 switch_nic_out_12_nic_in_6
 switch_nic_out_12_nic_in_7 switch_nic_out_12_nic_in_8
 switch_nic_out_12_nic_in_10 switch_nic_out_12_nic_in_13
 switch_nic_out_12_nic_in_14 switch_nic_out_12_nic_in_15
 switch_nic_in_12_12 switch_12_nic_out_12 switch_source_nic_in_13
 switch_nic_out_13_sink switch_nic_out_13_nic_in_0
 switch_nic_out_13_nic_in_4 switch_nic_out_13_nic_in_6
 switch_nic_out_13_nic_in_7 switch_nic_out_13_nic_in_8
 switch_nic_out_13_nic_in_10 switch_nic_out_13_nic_in_12
 switch_nic_out_13_nic_in_14 switch_nic_out_13_nic_in_15
 switch_nic_in_13_13 switch_13_nic_out_13 switch_source_nic_in_14
 switch_nic_out_14_sink switch_nic_out_14_nic_in_0
 switch_nic_out_14_nic_in_4 switch_nic_out_14_nic_in_6
 switch_nic_out_14_nic_in_7 switch_nic_out_14_nic_in_8
 switch_nic_out_14_nic_in_10 switch_nic_out_14_nic_in_12
 switch_nic_out_14_nic_in_13 switch_nic_out_14_nic_in_15
 switch_nic_in_14_14 switch_14_nic_out_14 switch_source_nic_in_15
 switch_nic_out_15_sink switch_nic_out_15_nic_in_0
 switch_nic_out_15_nic_in_4 switch_nic_out_15_nic_in_6
 switch_nic_out_15_nic_in_7 switch_nic_out_15_nic_in_8
 switch_nic_out_15_nic_in_10 switch_nic_out_15_nic_in_12
 switch_nic_out_15_nic_in_13 switch_nic_out_15_nic_in_14
 switch_nic_in_15_15 switch_15_nic_out_15 group_tp_active_0
 group_tp_active_4 group_tp_active_6 group_tp_active_7 group_tp_active_8
 group_tp_active_10 group_tp_active_12 group_tp_active_13
 group_tp_active_14 group_tp_active_15 is_active_0 is_active_1 is_active_2
 is_active_3 is_active_4 is_active_5 is_active_6 is_active_7 is_active_8
 is_active_9 is_active_10 is_active_11 is_active_12 is_active_13
 is_active_14 is_active_15 is_start_zero_0_for_source_nic_in_0
 is_start_zero_1_for_source_nic_in_0 is_start_zero_2_for_source_nic_in_0
 is_start_zero_3_for_source_nic_in_0 is_end_complete_0 is_end_complete_1
 is_end_complete_2 is_end_complete_3 pp_permission_0_1 pp_permission_0_2
 pp_permission_0_3 pp_permission_1_0 pp_permission_1_2 pp_permission_1_3
 pp_permission_2_0 pp_permission_2_1 pp_permission_2_3 pp_permission_3_0
 pp_permission_3_1 pp_permission_3_2 is_start_zero_4_for_source_nic_in_4
 is_start_zero_5_for_source_nic_in_4 is_end_complete_4 is_end_complete_5
 pp_permission_4_5 pp_permission_5_4 is_start_zero_6_for_source_nic_in_6
 is_end_complete_6 is_start_zero_7_for_source_nic_in_7 is_end_complete_7
 is_start_zero_8_for_source_nic_in_8 is_start_zero_9_for_source_nic_in_8
 is_end_complete_8 is_end_complete_9 pp_permission_8_9 pp_permission_9_8
 is_start_zero_10_for_source_nic_in_10
 is_start_zero_11_for_source_nic_in_10 is_end_complete_10
 is_end_complete_11 pp_permission_10_11 pp_permission_11_10
 is_start_zero_12_for_source_nic_in_12 is_end_complete_12
 is_start_zero_13_for_source_nic_in_13 is_end_complete_13
 is_start_zero_14_for_source_nic_in_14 is_end_complete_14
 is_start_zero_15_for_source_nic_in_15 is_end_complete_15
 tp_on_and_switch_source_nic_in_0_0 tp_on_and_switch_nic_out_0_nic_in_4_4
 tp_on_and_switch_nic_out_0_nic_in_8_8
 tp_on_and_switch_nic_out_0_nic_in_10_10 tp_on_and_switch_source_nic_in_4_4
 tp_on_and_switch_nic_out_4_nic_in_0_0
 tp_on_and_switch_nic_out_4_nic_in_8_8
 tp_on_and_switch_nic_out_4_nic_in_10_10
 tp_on_and_switch_nic_out_6_nic_in_0_0
 tp_on_and_switch_nic_out_6_nic_in_4_4
 tp_on_and_switch_nic_out_6_nic_in_8_8
 tp_on_and_switch_nic_out_6_nic_in_10_10
 tp_on_and_switch_nic_out_7_nic_in_0_0
 tp_on_and_switch_nic_out_7_nic_in_4_4
 tp_on_and_switch_nic_out_7_nic_in_8_8
 tp_on_and_switch_nic_out_7_nic_in_10_10 tp_on_and_switch_source_nic_in_8_8
 tp_on_and_switch_nic_out_8_nic_in_0_0
 tp_on_and_switch_nic_out_8_nic_in_4_4
 tp_on_and_switch_nic_out_8_nic_in_10_10
 tp_on_and_switch_source_nic_in_10_10
 tp_on_and_switch_nic_out_10_nic_in_0_0
 tp_on_and_switch_nic_out_10_nic_in_4_4
 tp_on_and_switch_nic_out_10_nic_in_8_8
 tp_on_and_switch_nic_out_12_nic_in_0_0
 tp_on_and_switch_nic_out_12_nic_in_4_4
 tp_on_and_switch_nic_out_12_nic_in_8_8
 tp_on_and_switch_nic_out_12_nic_in_10_10
 tp_on_and_switch_nic_out_13_nic_in_0_0
 tp_on_and_switch_nic_out_13_nic_in_4_4
 tp_on_and_switch_nic_out_13_nic_in_8_8
 tp_on_and_switch_nic_out_13_nic_in_10_10
 tp_on_and_switch_nic_out_14_nic_in_0_0
 tp_on_and_switch_nic_out_14_nic_in_4_4
 tp_on_and_switch_nic_out_14_nic_in_8_8
 tp_on_and_switch_nic_out_14_nic_in_10_10
 tp_on_and_switch_nic_out_15_nic_in_0_0
 tp_on_and_switch_nic_out_15_nic_in_4_4
 tp_on_and_switch_nic_out_15_nic_in_8_8
 tp_on_and_switch_nic_out_15_nic_in_10_10
Generals
 start_0 start_1 start_2 start_3 start_4 start_5 start_6 start_7 start_8
 start_9 start_10 start_11 start_12 start_13 start_14 start_15
 group_start_4 val_end_0 val_end_1 val_end_2 val_end_3 group_end_0
 group_start_6 group_start_7 group_start_8 group_start_10 group_start_12
 group_start_13 group_start_14 group_start_15 group_start_0 val_end_4
 val_end_5 group_end_4 val_end_6 group_end_6 val_end_7 group_end_7
 val_end_8 val_end_9 group_end_8 val_end_10 val_end_11 group_end_10
 val_end_12 group_end_12 val_end_13 group_end_13 val_end_14 group_end_14
 val_end_15 group_end_15
General Constraints
 min_start_4: group_start_4 = MIN ( start_4 , start_5 )
 max_end_0: group_end_0 = MAX ( val_end_0 , val_end_1 , val_end_2 ,
   val_end_3 )
 min_start_6: group_start_6 = MIN ( start_6 )
 min_start_7: group_start_7 = MIN ( start_7 )
 min_start_8: group_start_8 = MIN ( start_8 , start_9 )
 min_start_10: group_start_10 = MIN ( start_10 , start_11 )
 min_start_12: group_start_12 = MIN ( start_12 )
 min_start_13: group_start_13 = MIN ( start_13 )
 min_start_14: group_start_14 = MIN ( start_14 )
 min_start_15: group_start_15 = MIN ( start_15 )
 min_start_0: group_start_0 = MIN ( start_0 , start_1 , start_2 , start_3 )
 max_end_4: group_end_4 = MAX ( val_end_4 , val_end_5 )
 max_end_6: group_end_6 = MAX ( val_end_6 )
 max_end_7: group_end_7 = MAX ( val_end_7 )
 max_end_8: group_end_8 = MAX ( val_end_8 , val_end_9 )
 max_end_10: group_end_10 = MAX ( val_end_10 , val_end_11 )
 max_end_12: group_end_12 = MAX ( val_end_12 )
 max_end_13: group_end_13 = MAX ( val_end_13 )
 max_end_14: group_end_14 = MAX ( val_end_14 )
 max_end_15: group_end_15 = MAX ( val_end_15 )
End
