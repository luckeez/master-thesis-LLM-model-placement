\ Model 2026-02-22_08-22-39
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  9.75365e-05 flow_source_nic_in_0 + 9.75365e-05 flow_source_nic_in_1
   + 9.75365e-05 flow_source_nic_in_2 + 9.75365e-05 flow_source_nic_in_3
   + 9.75365e-05 flow_source_nic_in_4 + 9.75365e-05 flow_source_nic_in_6
   + 9.75365e-05 flow_source_nic_in_7 + 9.75365e-05 flow_source_nic_in_8
   + 9.75365e-05 flow_source_nic_in_10 + 9.75365e-05 flow_source_nic_in_12
   + 9.75365e-05 flow_source_nic_in_14 - 0.1000585137507314 is_active_0
   - 0.0679344645991808 is_active_1 - 0.0868929198361615 is_active_2
   - 0.0679344645991808 is_active_3 - 0.0716208308952604 is_active_4
   - 0.0716208308952604 is_active_5 - 0.0716208308952604 is_active_6
   - 0.0716208308952604 is_active_7 - 0.0716208308952604 is_active_8
   - 0.0716208308952604 is_active_9 - 0.0173785839672323 is_active_10
   - 0.0173785839672323 is_active_11 - 0.0247513165593915 is_active_12
   - 0.0247513165593915 is_active_13 - 0.0157987126974839 is_active_14
   - 0.0157987126974839 is_active_15 - 0.0157987126974839 is_active_16
   - 0.0157987126974839 is_active_17 + 0 val_end_0 + 0 val_end_1
   + 0 val_end_2 + 0 val_end_3 + 0 val_end_4 + 0 val_end_5 + 0 val_end_6
   + 0 val_end_7 + 0 val_end_8 + 0 val_end_9 + 0 val_end_10 + 0 val_end_11
   + 0 val_end_12 + 0 val_end_13 + 0 val_end_14 + 0 val_end_15
   + 0 val_end_16 + 0 val_end_17
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
 mem_active_constr_16: hold_16_1 + hold_16_2 + hold_16_3 + hold_16_4
   + hold_16_5 + hold_16_6 + hold_16_7 + hold_16_8 + hold_16_9 + hold_16_10
   + hold_16_11 + hold_16_12 + hold_16_13 + hold_16_14 + hold_16_15
   + hold_16_16 + hold_16_17 + hold_16_18 + hold_16_19 + hold_16_20
   + hold_16_21 + hold_16_22 + hold_16_23 + hold_16_24 + hold_16_25
   + hold_16_26 - 81 is_active_16 <= 0
 mem_active_constr2_16: hold_16_1 + hold_16_2 + hold_16_3 + hold_16_4
   + hold_16_5 + hold_16_6 + hold_16_7 + hold_16_8 + hold_16_9 + hold_16_10
   + hold_16_11 + hold_16_12 + hold_16_13 + hold_16_14 + hold_16_15
   + hold_16_16 + hold_16_17 + hold_16_18 + hold_16_19 + hold_16_20
   + hold_16_21 + hold_16_22 + hold_16_23 + hold_16_24 + hold_16_25
   + hold_16_26 - is_active_16 >= 0
 mem_active_constr_17: hold_17_1 + hold_17_2 + hold_17_3 + hold_17_4
   + hold_17_5 + hold_17_6 + hold_17_7 + hold_17_8 + hold_17_9 + hold_17_10
   + hold_17_11 + hold_17_12 + hold_17_13 + hold_17_14 + hold_17_15
   + hold_17_16 + hold_17_17 + hold_17_18 + hold_17_19 + hold_17_20
   + hold_17_21 + hold_17_22 + hold_17_23 + hold_17_24 + hold_17_25
   + hold_17_26 - 81 is_active_17 <= 0
 mem_active_constr2_17: hold_17_1 + hold_17_2 + hold_17_3 + hold_17_4
   + hold_17_5 + hold_17_6 + hold_17_7 + hold_17_8 + hold_17_9 + hold_17_10
   + hold_17_11 + hold_17_12 + hold_17_13 + hold_17_14 + hold_17_15
   + hold_17_16 + hold_17_17 + hold_17_18 + hold_17_19 + hold_17_20
   + hold_17_21 + hold_17_22 + hold_17_23 + hold_17_24 + hold_17_25
   + hold_17_26 - is_active_17 >= 0
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
 hold_constraint_16: hold_16_1 + hold_16_2 + hold_16_3 + hold_16_4
   + hold_16_5 + hold_16_6 + hold_16_7 + hold_16_8 + hold_16_9 + hold_16_10
   + hold_16_11 + hold_16_12 + hold_16_13 + hold_16_14 + hold_16_15
   + hold_16_16 + hold_16_17 + hold_16_18 + hold_16_19 + hold_16_20
   + hold_16_21 + hold_16_22 + hold_16_23 + hold_16_24 + hold_16_25
   + hold_16_26 <= 1
 end_constraint_16: start_16 + hold_16_1 + 2 hold_16_2 + 3 hold_16_3
   + 4 hold_16_4 + 5 hold_16_5 + 6 hold_16_6 + 7 hold_16_7 + 8 hold_16_8
   + 9 hold_16_9 + 10 hold_16_10 + 11 hold_16_11 + 12 hold_16_12
   + 13 hold_16_13 + 14 hold_16_14 + 15 hold_16_15 + 16 hold_16_16
   + 17 hold_16_17 + 18 hold_16_18 + 19 hold_16_19 + 20 hold_16_20
   + 21 hold_16_21 + 22 hold_16_22 + 23 hold_16_23 + 24 hold_16_24
   + 25 hold_16_25 + 26 hold_16_26 <= 80
 hold_constraint_17: hold_17_1 + hold_17_2 + hold_17_3 + hold_17_4
   + hold_17_5 + hold_17_6 + hold_17_7 + hold_17_8 + hold_17_9 + hold_17_10
   + hold_17_11 + hold_17_12 + hold_17_13 + hold_17_14 + hold_17_15
   + hold_17_16 + hold_17_17 + hold_17_18 + hold_17_19 + hold_17_20
   + hold_17_21 + hold_17_22 + hold_17_23 + hold_17_24 + hold_17_25
   + hold_17_26 <= 1
 end_constraint_17: start_17 + hold_17_1 + 2 hold_17_2 + 3 hold_17_3
   + 4 hold_17_4 + 5 hold_17_5 + 6 hold_17_6 + 7 hold_17_7 + 8 hold_17_8
   + 9 hold_17_9 + 10 hold_17_10 + 11 hold_17_11 + 12 hold_17_12
   + 13 hold_17_13 + 14 hold_17_14 + 15 hold_17_15 + 16 hold_17_16
   + 17 hold_17_17 + 18 hold_17_18 + 19 hold_17_19 + 20 hold_17_20
   + 21 hold_17_21 + 22 hold_17_22 + 23 hold_17_23 + 24 hold_17_24
   + 25 hold_17_25 + 26 hold_17_26 <= 80
 node_flow_constraint_nic_in_0: flow_source_nic_in_0 - flow_nic_in_0_0
   + flow_nic_out_1_nic_in_0 + flow_nic_out_2_nic_in_0
   + flow_nic_out_3_nic_in_0 + flow_nic_out_4_nic_in_0
   + flow_nic_out_6_nic_in_0 + flow_nic_out_7_nic_in_0
   + flow_nic_out_8_nic_in_0 + flow_nic_out_10_nic_in_0
   + flow_nic_out_12_nic_in_0 + flow_nic_out_14_nic_in_0 = 0
 node_flow_constraint_nic_out_0: - flow_nic_out_0_sink
   - flow_nic_out_0_nic_in_1 - flow_nic_out_0_nic_in_2
   - flow_nic_out_0_nic_in_3 - flow_nic_out_0_nic_in_4
   - flow_nic_out_0_nic_in_6 - flow_nic_out_0_nic_in_7
   - flow_nic_out_0_nic_in_8 - flow_nic_out_0_nic_in_10
   - flow_nic_out_0_nic_in_12 - flow_nic_out_0_nic_in_14 + flow_0_nic_out_0
   = 0
 node_flow_constraint_nic_in_1: flow_nic_out_0_nic_in_1
   + flow_source_nic_in_1 - flow_nic_in_1_1 + flow_nic_out_2_nic_in_1
   + flow_nic_out_3_nic_in_1 + flow_nic_out_4_nic_in_1
   + flow_nic_out_6_nic_in_1 + flow_nic_out_7_nic_in_1
   + flow_nic_out_8_nic_in_1 + flow_nic_out_10_nic_in_1
   + flow_nic_out_12_nic_in_1 + flow_nic_out_14_nic_in_1 = 0
 node_flow_constraint_nic_out_1: - flow_nic_out_1_sink
   - flow_nic_out_1_nic_in_0 - flow_nic_out_1_nic_in_2
   - flow_nic_out_1_nic_in_3 - flow_nic_out_1_nic_in_4
   - flow_nic_out_1_nic_in_6 - flow_nic_out_1_nic_in_7
   - flow_nic_out_1_nic_in_8 - flow_nic_out_1_nic_in_10
   - flow_nic_out_1_nic_in_12 - flow_nic_out_1_nic_in_14 + flow_1_nic_out_1
   = 0
 node_flow_constraint_nic_in_2: flow_nic_out_0_nic_in_2
   + flow_nic_out_1_nic_in_2 + flow_source_nic_in_2 - flow_nic_in_2_2
   + flow_nic_out_3_nic_in_2 + flow_nic_out_4_nic_in_2
   + flow_nic_out_6_nic_in_2 + flow_nic_out_7_nic_in_2
   + flow_nic_out_8_nic_in_2 + flow_nic_out_10_nic_in_2
   + flow_nic_out_12_nic_in_2 + flow_nic_out_14_nic_in_2 = 0
 node_flow_constraint_nic_out_2: - flow_nic_out_2_sink
   - flow_nic_out_2_nic_in_0 - flow_nic_out_2_nic_in_1
   - flow_nic_out_2_nic_in_3 - flow_nic_out_2_nic_in_4
   - flow_nic_out_2_nic_in_6 - flow_nic_out_2_nic_in_7
   - flow_nic_out_2_nic_in_8 - flow_nic_out_2_nic_in_10
   - flow_nic_out_2_nic_in_12 - flow_nic_out_2_nic_in_14 + flow_2_nic_out_2
   = 0
 node_flow_constraint_nic_in_3: flow_nic_out_0_nic_in_3
   + flow_nic_out_1_nic_in_3 + flow_nic_out_2_nic_in_3
   + flow_source_nic_in_3 - flow_nic_in_3_3 + flow_nic_out_4_nic_in_3
   + flow_nic_out_6_nic_in_3 + flow_nic_out_7_nic_in_3
   + flow_nic_out_8_nic_in_3 + flow_nic_out_10_nic_in_3
   + flow_nic_out_12_nic_in_3 + flow_nic_out_14_nic_in_3 = 0
 node_flow_constraint_nic_out_3: - flow_nic_out_3_sink
   - flow_nic_out_3_nic_in_0 - flow_nic_out_3_nic_in_1
   - flow_nic_out_3_nic_in_2 - flow_nic_out_3_nic_in_4
   - flow_nic_out_3_nic_in_6 - flow_nic_out_3_nic_in_7
   - flow_nic_out_3_nic_in_8 - flow_nic_out_3_nic_in_10
   - flow_nic_out_3_nic_in_12 - flow_nic_out_3_nic_in_14 + flow_3_nic_out_3
   = 0
 node_flow_constraint_nic_in_4: flow_nic_out_0_nic_in_4
   + flow_nic_out_1_nic_in_4 + flow_nic_out_2_nic_in_4
   + flow_nic_out_3_nic_in_4 + flow_source_nic_in_4 - flow_nic_in_4_4
   - flow_nic_in_4_5 + flow_nic_out_6_nic_in_4 + flow_nic_out_7_nic_in_4
   + flow_nic_out_8_nic_in_4 + flow_nic_out_10_nic_in_4
   + flow_nic_out_12_nic_in_4 + flow_nic_out_14_nic_in_4 = 0
 node_flow_constraint_nic_out_4: - flow_nic_out_4_sink
   - flow_nic_out_4_nic_in_0 - flow_nic_out_4_nic_in_1
   - flow_nic_out_4_nic_in_2 - flow_nic_out_4_nic_in_3
   - flow_nic_out_4_nic_in_6 - flow_nic_out_4_nic_in_7
   - flow_nic_out_4_nic_in_8 - flow_nic_out_4_nic_in_10
   - flow_nic_out_4_nic_in_12 - flow_nic_out_4_nic_in_14 + flow_4_nic_out_4
   + flow_5_nic_out_4 = 0
 node_flow_constraint_nic_in_6: flow_nic_out_0_nic_in_6
   + flow_nic_out_1_nic_in_6 + flow_nic_out_2_nic_in_6
   + flow_nic_out_3_nic_in_6 + flow_nic_out_4_nic_in_6
   + flow_source_nic_in_6 - flow_nic_in_6_6 + flow_nic_out_7_nic_in_6
   + flow_nic_out_8_nic_in_6 + flow_nic_out_10_nic_in_6
   + flow_nic_out_12_nic_in_6 + flow_nic_out_14_nic_in_6 = 0
 node_flow_constraint_nic_out_6: - flow_nic_out_6_sink
   - flow_nic_out_6_nic_in_0 - flow_nic_out_6_nic_in_1
   - flow_nic_out_6_nic_in_2 - flow_nic_out_6_nic_in_3
   - flow_nic_out_6_nic_in_4 - flow_nic_out_6_nic_in_7
   - flow_nic_out_6_nic_in_8 - flow_nic_out_6_nic_in_10
   - flow_nic_out_6_nic_in_12 - flow_nic_out_6_nic_in_14 + flow_6_nic_out_6
   = 0
 node_flow_constraint_nic_in_7: flow_nic_out_0_nic_in_7
   + flow_nic_out_1_nic_in_7 + flow_nic_out_2_nic_in_7
   + flow_nic_out_3_nic_in_7 + flow_nic_out_4_nic_in_7
   + flow_nic_out_6_nic_in_7 + flow_source_nic_in_7 - flow_nic_in_7_7
   + flow_nic_out_8_nic_in_7 + flow_nic_out_10_nic_in_7
   + flow_nic_out_12_nic_in_7 + flow_nic_out_14_nic_in_7 = 0
 node_flow_constraint_nic_out_7: - flow_nic_out_7_sink
   - flow_nic_out_7_nic_in_0 - flow_nic_out_7_nic_in_1
   - flow_nic_out_7_nic_in_2 - flow_nic_out_7_nic_in_3
   - flow_nic_out_7_nic_in_4 - flow_nic_out_7_nic_in_6
   - flow_nic_out_7_nic_in_8 - flow_nic_out_7_nic_in_10
   - flow_nic_out_7_nic_in_12 - flow_nic_out_7_nic_in_14 + flow_7_nic_out_7
   = 0
 node_flow_constraint_nic_in_8: flow_nic_out_0_nic_in_8
   + flow_nic_out_1_nic_in_8 + flow_nic_out_2_nic_in_8
   + flow_nic_out_3_nic_in_8 + flow_nic_out_4_nic_in_8
   + flow_nic_out_6_nic_in_8 + flow_nic_out_7_nic_in_8
   + flow_source_nic_in_8 - flow_nic_in_8_8 - flow_nic_in_8_9
   + flow_nic_out_10_nic_in_8 + flow_nic_out_12_nic_in_8
   + flow_nic_out_14_nic_in_8 = 0
 node_flow_constraint_nic_out_8: - flow_nic_out_8_sink
   - flow_nic_out_8_nic_in_0 - flow_nic_out_8_nic_in_1
   - flow_nic_out_8_nic_in_2 - flow_nic_out_8_nic_in_3
   - flow_nic_out_8_nic_in_4 - flow_nic_out_8_nic_in_6
   - flow_nic_out_8_nic_in_7 - flow_nic_out_8_nic_in_10
   - flow_nic_out_8_nic_in_12 - flow_nic_out_8_nic_in_14 + flow_8_nic_out_8
   + flow_9_nic_out_8 = 0
 node_flow_constraint_nic_in_10: flow_nic_out_0_nic_in_10
   + flow_nic_out_1_nic_in_10 + flow_nic_out_2_nic_in_10
   + flow_nic_out_3_nic_in_10 + flow_nic_out_4_nic_in_10
   + flow_nic_out_6_nic_in_10 + flow_nic_out_7_nic_in_10
   + flow_nic_out_8_nic_in_10 + flow_source_nic_in_10 - flow_nic_in_10_10
   - flow_nic_in_10_11 + flow_nic_out_12_nic_in_10
   + flow_nic_out_14_nic_in_10 = 0
 node_flow_constraint_nic_out_10: - flow_nic_out_10_sink
   - flow_nic_out_10_nic_in_0 - flow_nic_out_10_nic_in_1
   - flow_nic_out_10_nic_in_2 - flow_nic_out_10_nic_in_3
   - flow_nic_out_10_nic_in_4 - flow_nic_out_10_nic_in_6
   - flow_nic_out_10_nic_in_7 - flow_nic_out_10_nic_in_8
   - flow_nic_out_10_nic_in_12 - flow_nic_out_10_nic_in_14
   + flow_10_nic_out_10 + flow_11_nic_out_10 = 0
 node_flow_constraint_nic_in_12: flow_nic_out_0_nic_in_12
   + flow_nic_out_1_nic_in_12 + flow_nic_out_2_nic_in_12
   + flow_nic_out_3_nic_in_12 + flow_nic_out_4_nic_in_12
   + flow_nic_out_6_nic_in_12 + flow_nic_out_7_nic_in_12
   + flow_nic_out_8_nic_in_12 + flow_nic_out_10_nic_in_12
   + flow_source_nic_in_12 - flow_nic_in_12_12 - flow_nic_in_12_13
   + flow_nic_out_14_nic_in_12 = 0
 node_flow_constraint_nic_out_12: - flow_nic_out_12_sink
   - flow_nic_out_12_nic_in_0 - flow_nic_out_12_nic_in_1
   - flow_nic_out_12_nic_in_2 - flow_nic_out_12_nic_in_3
   - flow_nic_out_12_nic_in_4 - flow_nic_out_12_nic_in_6
   - flow_nic_out_12_nic_in_7 - flow_nic_out_12_nic_in_8
   - flow_nic_out_12_nic_in_10 - flow_nic_out_12_nic_in_14
   + flow_12_nic_out_12 + flow_13_nic_out_12 = 0
 node_flow_constraint_nic_in_14: flow_nic_out_0_nic_in_14
   + flow_nic_out_1_nic_in_14 + flow_nic_out_2_nic_in_14
   + flow_nic_out_3_nic_in_14 + flow_nic_out_4_nic_in_14
   + flow_nic_out_6_nic_in_14 + flow_nic_out_7_nic_in_14
   + flow_nic_out_8_nic_in_14 + flow_nic_out_10_nic_in_14
   + flow_nic_out_12_nic_in_14 + flow_source_nic_in_14 - flow_nic_in_14_14
   - flow_nic_in_14_15 - flow_nic_in_14_16 - flow_nic_in_14_17 = 0
 node_flow_constraint_nic_out_14: - flow_nic_out_14_sink
   - flow_nic_out_14_nic_in_0 - flow_nic_out_14_nic_in_1
   - flow_nic_out_14_nic_in_2 - flow_nic_out_14_nic_in_3
   - flow_nic_out_14_nic_in_4 - flow_nic_out_14_nic_in_6
   - flow_nic_out_14_nic_in_7 - flow_nic_out_14_nic_in_8
   - flow_nic_out_14_nic_in_10 - flow_nic_out_14_nic_in_12
   + flow_14_nic_out_14 + flow_15_nic_out_14 + flow_16_nic_out_14
   + flow_17_nic_out_14 = 0
 node_flow_constraint_0: flow_nic_in_0_0 - flow_0_nic_out_0 = 0
 node_flow_constraint_1: flow_nic_in_1_1 - flow_1_nic_out_1 = 0
 node_flow_constraint_2: flow_nic_in_2_2 - flow_2_nic_out_2 = 0
 node_flow_constraint_3: flow_nic_in_3_3 - flow_3_nic_out_3 = 0
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
 node_flow_constraint_12: flow_nic_in_12_12 - flow_12_nic_out_12
   - flow_12_13 + flow_13_12 = 0
 node_flow_constraint_13: flow_nic_in_12_13 - flow_13_nic_out_12
   + flow_12_13 - flow_13_12 = 0
 node_flow_constraint_14: flow_nic_in_14_14 - flow_14_nic_out_14
   - flow_14_15 - flow_14_16 - flow_14_17 + flow_15_14 + flow_16_14
   + flow_17_14 = 0
 node_flow_constraint_15: flow_nic_in_14_15 - flow_15_nic_out_14
   + flow_14_15 - flow_15_14 - flow_15_16 - flow_15_17 + flow_16_15
   + flow_17_15 = 0
 node_flow_constraint_16: flow_nic_in_14_16 - flow_16_nic_out_14
   + flow_14_16 + flow_15_16 - flow_16_14 - flow_16_15 - flow_16_17
   + flow_17_16 = 0
 node_flow_constraint_17: flow_nic_in_14_17 - flow_17_nic_out_14
   + flow_14_17 + flow_15_17 + flow_16_17 - flow_17_14 - flow_17_15
   - flow_17_16 = 0
 constr_is_start_zero_0: start_0 + 80 is_start_zero_0_for_source_nic_in_0
   <= 80
 constr_edge_switch_start_zero_0_OR: switch_source_nic_in_0
   - is_start_zero_0_for_source_nic_in_0 <= 0
 constr_is_end_complete_0: start_0 + hold_0_1 + 2 hold_0_2 + 3 hold_0_3
   + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7 + 8 hold_0_8
   + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12 + 13 hold_0_13
   + 14 hold_0_14 + 15 hold_0_15 + 16 hold_0_16 + 17 hold_0_17
   + 18 hold_0_18 + 19 hold_0_19 + 20 hold_0_20 + 21 hold_0_21
   + 22 hold_0_22 + 23 hold_0_23 + 24 hold_0_24 + 25 hold_0_25
   + 26 hold_0_26 - 80 is_end_complete_0 >= 0
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
 constr_link_match_1_0_14: 80 switch_nic_out_0_nic_in_14 + group_end_0
   - group_start_14 <= 80
 constr_link_match_2_0_14: 80 switch_nic_out_0_nic_in_14 - group_end_0
   + group_start_14 <= 80
 constr_is_start_zero_1: start_1 + 80 is_start_zero_1_for_source_nic_in_1
   <= 80
 constr_edge_switch_start_zero_1_OR: switch_source_nic_in_1
   - is_start_zero_1_for_source_nic_in_1 <= 0
 constr_is_end_complete_1: start_1 + hold_1_1 + 2 hold_1_2 + 3 hold_1_3
   + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7 + 8 hold_1_8
   + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 + 12 hold_1_12 + 13 hold_1_13
   + 14 hold_1_14 + 15 hold_1_15 + 16 hold_1_16 + 17 hold_1_17
   + 18 hold_1_18 + 19 hold_1_19 + 20 hold_1_20 + 21 hold_1_21
   + 22 hold_1_22 + 23 hold_1_23 + 24 hold_1_24 + 25 hold_1_25
   + 26 hold_1_26 - 80 is_end_complete_1 >= 0
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
 constr_link_match_1_1_10: 80 switch_nic_out_1_nic_in_10 - group_start_10
   + group_end_1 <= 80
 constr_link_match_2_1_10: 80 switch_nic_out_1_nic_in_10 + group_start_10
   - group_end_1 <= 80
 constr_link_match_1_1_12: 80 switch_nic_out_1_nic_in_12 - group_start_12
   + group_end_1 <= 80
 constr_link_match_2_1_12: 80 switch_nic_out_1_nic_in_12 + group_start_12
   - group_end_1 <= 80
 constr_link_match_1_1_14: 80 switch_nic_out_1_nic_in_14 - group_start_14
   + group_end_1 <= 80
 constr_link_match_2_1_14: 80 switch_nic_out_1_nic_in_14 + group_start_14
   - group_end_1 <= 80
 constr_is_start_zero_2: start_2 + 80 is_start_zero_2_for_source_nic_in_2
   <= 80
 constr_edge_switch_start_zero_2_OR: switch_source_nic_in_2
   - is_start_zero_2_for_source_nic_in_2 <= 0
 constr_is_end_complete_2: start_2 + hold_2_1 + 2 hold_2_2 + 3 hold_2_3
   + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7 + 8 hold_2_8
   + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 12 hold_2_12 + 13 hold_2_13
   + 14 hold_2_14 + 15 hold_2_15 + 16 hold_2_16 + 17 hold_2_17
   + 18 hold_2_18 + 19 hold_2_19 + 20 hold_2_20 + 21 hold_2_21
   + 22 hold_2_22 + 23 hold_2_23 + 24 hold_2_24 + 25 hold_2_25
   + 26 hold_2_26 - 80 is_end_complete_2 >= 0
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
 constr_link_match_1_2_10: 80 switch_nic_out_2_nic_in_10 - group_start_10
   + group_end_2 <= 80
 constr_link_match_2_2_10: 80 switch_nic_out_2_nic_in_10 + group_start_10
   - group_end_2 <= 80
 constr_link_match_1_2_12: 80 switch_nic_out_2_nic_in_12 - group_start_12
   + group_end_2 <= 80
 constr_link_match_2_2_12: 80 switch_nic_out_2_nic_in_12 + group_start_12
   - group_end_2 <= 80
 constr_link_match_1_2_14: 80 switch_nic_out_2_nic_in_14 - group_start_14
   + group_end_2 <= 80
 constr_link_match_2_2_14: 80 switch_nic_out_2_nic_in_14 + group_start_14
   - group_end_2 <= 80
 constr_is_start_zero_3: start_3 + 80 is_start_zero_3_for_source_nic_in_3
   <= 80
 constr_edge_switch_start_zero_3_OR: switch_source_nic_in_3
   - is_start_zero_3_for_source_nic_in_3 <= 0
 constr_is_end_complete_3: start_3 + hold_3_1 + 2 hold_3_2 + 3 hold_3_3
   + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7 + 8 hold_3_8
   + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11 + 12 hold_3_12 + 13 hold_3_13
   + 14 hold_3_14 + 15 hold_3_15 + 16 hold_3_16 + 17 hold_3_17
   + 18 hold_3_18 + 19 hold_3_19 + 20 hold_3_20 + 21 hold_3_21
   + 22 hold_3_22 + 23 hold_3_23 + 24 hold_3_24 + 25 hold_3_25
   + 26 hold_3_26 - 80 is_end_complete_3 >= 0
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
 constr_link_match_1_3_10: 80 switch_nic_out_3_nic_in_10 - group_start_10
   + group_end_3 <= 80
 constr_link_match_2_3_10: 80 switch_nic_out_3_nic_in_10 + group_start_10
   - group_end_3 <= 80
 constr_link_match_1_3_12: 80 switch_nic_out_3_nic_in_12 - group_start_12
   + group_end_3 <= 80
 constr_link_match_2_3_12: 80 switch_nic_out_3_nic_in_12 + group_start_12
   - group_end_3 <= 80
 constr_link_match_1_3_14: 80 switch_nic_out_3_nic_in_14 - group_start_14
   + group_end_3 <= 80
 constr_link_match_2_3_14: 80 switch_nic_out_3_nic_in_14 + group_start_14
   - group_end_3 <= 80
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
 constr_link_match_1_4_14: 80 switch_nic_out_4_nic_in_14 - group_start_14
   + group_end_4 <= 80
 constr_link_match_2_4_14: 80 switch_nic_out_4_nic_in_14 + group_start_14
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
 constr_link_match_1_6_14: 80 switch_nic_out_6_nic_in_14 - group_start_14
   + group_end_6 <= 80
 constr_link_match_2_6_14: 80 switch_nic_out_6_nic_in_14 + group_start_14
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
 constr_link_match_1_7_14: 80 switch_nic_out_7_nic_in_14 - group_start_14
   + group_end_7 <= 80
 constr_link_match_2_7_14: 80 switch_nic_out_7_nic_in_14 + group_start_14
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
 constr_link_match_1_8_14: 80 switch_nic_out_8_nic_in_14 - group_start_14
   + group_end_8 <= 80
 constr_link_match_2_8_14: 80 switch_nic_out_8_nic_in_14 + group_start_14
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
 constr_link_match_1_10_14: 80 switch_nic_out_10_nic_in_14 - group_start_14
   + group_end_10 <= 80
 constr_link_match_2_10_14: 80 switch_nic_out_10_nic_in_14 + group_start_14
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
 constr_link_match_1_12_14: 80 switch_nic_out_12_nic_in_14 - group_start_14
   + group_end_12 <= 80
 constr_link_match_2_12_14: 80 switch_nic_out_12_nic_in_14 + group_start_14
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
 constr_is_start_zero_15: start_15
   + 80 is_start_zero_15_for_source_nic_in_14 <= 80
 constr_is_start_zero_16: start_16
   + 80 is_start_zero_16_for_source_nic_in_14 <= 80
 constr_is_start_zero_17: start_17
   + 80 is_start_zero_17_for_source_nic_in_14 <= 80
 constr_edge_switch_start_zero_14_OR: switch_source_nic_in_14
   - is_start_zero_14_for_source_nic_in_14
   - is_start_zero_15_for_source_nic_in_14
   - is_start_zero_16_for_source_nic_in_14
   - is_start_zero_17_for_source_nic_in_14 <= 0
 constr_is_end_complete_14: start_14 + hold_14_1 + 2 hold_14_2
   + 3 hold_14_3 + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6 + 7 hold_14_7
   + 8 hold_14_8 + 9 hold_14_9 + 10 hold_14_10 + 11 hold_14_11
   + 12 hold_14_12 + 13 hold_14_13 + 14 hold_14_14 + 15 hold_14_15
   + 16 hold_14_16 + 17 hold_14_17 + 18 hold_14_18 + 19 hold_14_19
   + 20 hold_14_20 + 21 hold_14_21 + 22 hold_14_22 + 23 hold_14_23
   + 24 hold_14_24 + 25 hold_14_25 + 26 hold_14_26 - 80 is_end_complete_14
   >= 0
 constr_is_end_complete_15: start_15 + hold_15_1 + 2 hold_15_2
   + 3 hold_15_3 + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6 + 7 hold_15_7
   + 8 hold_15_8 + 9 hold_15_9 + 10 hold_15_10 + 11 hold_15_11
   + 12 hold_15_12 + 13 hold_15_13 + 14 hold_15_14 + 15 hold_15_15
   + 16 hold_15_16 + 17 hold_15_17 + 18 hold_15_18 + 19 hold_15_19
   + 20 hold_15_20 + 21 hold_15_21 + 22 hold_15_22 + 23 hold_15_23
   + 24 hold_15_24 + 25 hold_15_25 + 26 hold_15_26 - 80 is_end_complete_15
   >= 0
 constr_is_end_complete_16: start_16 + hold_16_1 + 2 hold_16_2
   + 3 hold_16_3 + 4 hold_16_4 + 5 hold_16_5 + 6 hold_16_6 + 7 hold_16_7
   + 8 hold_16_8 + 9 hold_16_9 + 10 hold_16_10 + 11 hold_16_11
   + 12 hold_16_12 + 13 hold_16_13 + 14 hold_16_14 + 15 hold_16_15
   + 16 hold_16_16 + 17 hold_16_17 + 18 hold_16_18 + 19 hold_16_19
   + 20 hold_16_20 + 21 hold_16_21 + 22 hold_16_22 + 23 hold_16_23
   + 24 hold_16_24 + 25 hold_16_25 + 26 hold_16_26 - 80 is_end_complete_16
   >= 0
 constr_is_end_complete_17: start_17 + hold_17_1 + 2 hold_17_2
   + 3 hold_17_3 + 4 hold_17_4 + 5 hold_17_5 + 6 hold_17_6 + 7 hold_17_7
   + 8 hold_17_8 + 9 hold_17_9 + 10 hold_17_10 + 11 hold_17_11
   + 12 hold_17_12 + 13 hold_17_13 + 14 hold_17_14 + 15 hold_17_15
   + 16 hold_17_16 + 17 hold_17_17 + 18 hold_17_18 + 19 hold_17_19
   + 20 hold_17_20 + 21 hold_17_21 + 22 hold_17_22 + 23 hold_17_23
   + 24 hold_17_24 + 25 hold_17_25 + 26 hold_17_26 - 80 is_end_complete_17
   >= 0
 constr_edge_switch_end_max_14_OR: switch_nic_out_14_sink
   - is_end_complete_14 - is_end_complete_15 - is_end_complete_16
   - is_end_complete_17 <= 0
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
 constr_pp_permissionA_14_15: start_14 - start_15 + hold_14_1 + 2 hold_14_2
   + 3 hold_14_3 + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6 + 7 hold_14_7
   + 8 hold_14_8 + 9 hold_14_9 + 10 hold_14_10 + 11 hold_14_11
   + 12 hold_14_12 + 13 hold_14_13 + 14 hold_14_14 + 15 hold_14_15
   + 16 hold_14_16 + 17 hold_14_17 + 18 hold_14_18 + 19 hold_14_19
   + 20 hold_14_20 + 21 hold_14_21 + 22 hold_14_22 + 23 hold_14_23
   + 24 hold_14_24 + 25 hold_14_25 + 26 hold_14_26 + 80 pp_permission_14_15
   <= 80
 constr_pp_permissionB_14_15: - start_14 + start_15 - hold_14_1
   - 2 hold_14_2 - 3 hold_14_3 - 4 hold_14_4 - 5 hold_14_5 - 6 hold_14_6
   - 7 hold_14_7 - 8 hold_14_8 - 9 hold_14_9 - 10 hold_14_10
   - 11 hold_14_11 - 12 hold_14_12 - 13 hold_14_13 - 14 hold_14_14
   - 15 hold_14_15 - 16 hold_14_16 - 17 hold_14_17 - 18 hold_14_18
   - 19 hold_14_19 - 20 hold_14_20 - 21 hold_14_21 - 22 hold_14_22
   - 23 hold_14_23 - 24 hold_14_24 - 25 hold_14_25 - 26 hold_14_26
   + 80 pp_permission_14_15 <= 80
 constr_internal_compute_14_15: switch_14_15 - group_tp_active_14
   - pp_permission_14_15 <= 0
 constr_pp_permissionA_14_16: start_14 - start_16 + hold_14_1 + 2 hold_14_2
   + 3 hold_14_3 + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6 + 7 hold_14_7
   + 8 hold_14_8 + 9 hold_14_9 + 10 hold_14_10 + 11 hold_14_11
   + 12 hold_14_12 + 13 hold_14_13 + 14 hold_14_14 + 15 hold_14_15
   + 16 hold_14_16 + 17 hold_14_17 + 18 hold_14_18 + 19 hold_14_19
   + 20 hold_14_20 + 21 hold_14_21 + 22 hold_14_22 + 23 hold_14_23
   + 24 hold_14_24 + 25 hold_14_25 + 26 hold_14_26 + 80 pp_permission_14_16
   <= 80
 constr_pp_permissionB_14_16: - start_14 + start_16 - hold_14_1
   - 2 hold_14_2 - 3 hold_14_3 - 4 hold_14_4 - 5 hold_14_5 - 6 hold_14_6
   - 7 hold_14_7 - 8 hold_14_8 - 9 hold_14_9 - 10 hold_14_10
   - 11 hold_14_11 - 12 hold_14_12 - 13 hold_14_13 - 14 hold_14_14
   - 15 hold_14_15 - 16 hold_14_16 - 17 hold_14_17 - 18 hold_14_18
   - 19 hold_14_19 - 20 hold_14_20 - 21 hold_14_21 - 22 hold_14_22
   - 23 hold_14_23 - 24 hold_14_24 - 25 hold_14_25 - 26 hold_14_26
   + 80 pp_permission_14_16 <= 80
 constr_internal_compute_14_16: switch_14_16 - group_tp_active_14
   - pp_permission_14_16 <= 0
 constr_pp_permissionA_14_17: start_14 - start_17 + hold_14_1 + 2 hold_14_2
   + 3 hold_14_3 + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6 + 7 hold_14_7
   + 8 hold_14_8 + 9 hold_14_9 + 10 hold_14_10 + 11 hold_14_11
   + 12 hold_14_12 + 13 hold_14_13 + 14 hold_14_14 + 15 hold_14_15
   + 16 hold_14_16 + 17 hold_14_17 + 18 hold_14_18 + 19 hold_14_19
   + 20 hold_14_20 + 21 hold_14_21 + 22 hold_14_22 + 23 hold_14_23
   + 24 hold_14_24 + 25 hold_14_25 + 26 hold_14_26 + 80 pp_permission_14_17
   <= 80
 constr_pp_permissionB_14_17: - start_14 + start_17 - hold_14_1
   - 2 hold_14_2 - 3 hold_14_3 - 4 hold_14_4 - 5 hold_14_5 - 6 hold_14_6
   - 7 hold_14_7 - 8 hold_14_8 - 9 hold_14_9 - 10 hold_14_10
   - 11 hold_14_11 - 12 hold_14_12 - 13 hold_14_13 - 14 hold_14_14
   - 15 hold_14_15 - 16 hold_14_16 - 17 hold_14_17 - 18 hold_14_18
   - 19 hold_14_19 - 20 hold_14_20 - 21 hold_14_21 - 22 hold_14_22
   - 23 hold_14_23 - 24 hold_14_24 - 25 hold_14_25 - 26 hold_14_26
   + 80 pp_permission_14_17 <= 80
 constr_internal_compute_14_17: switch_14_17 - group_tp_active_14
   - pp_permission_14_17 <= 0
 constr_pp_permissionA_15_14: - start_14 + start_15 + hold_15_1
   + 2 hold_15_2 + 3 hold_15_3 + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6
   + 7 hold_15_7 + 8 hold_15_8 + 9 hold_15_9 + 10 hold_15_10
   + 11 hold_15_11 + 12 hold_15_12 + 13 hold_15_13 + 14 hold_15_14
   + 15 hold_15_15 + 16 hold_15_16 + 17 hold_15_17 + 18 hold_15_18
   + 19 hold_15_19 + 20 hold_15_20 + 21 hold_15_21 + 22 hold_15_22
   + 23 hold_15_23 + 24 hold_15_24 + 25 hold_15_25 + 26 hold_15_26
   + 80 pp_permission_15_14 <= 80
 constr_pp_permissionB_15_14: start_14 - start_15 - hold_15_1 - 2 hold_15_2
   - 3 hold_15_3 - 4 hold_15_4 - 5 hold_15_5 - 6 hold_15_6 - 7 hold_15_7
   - 8 hold_15_8 - 9 hold_15_9 - 10 hold_15_10 - 11 hold_15_11
   - 12 hold_15_12 - 13 hold_15_13 - 14 hold_15_14 - 15 hold_15_15
   - 16 hold_15_16 - 17 hold_15_17 - 18 hold_15_18 - 19 hold_15_19
   - 20 hold_15_20 - 21 hold_15_21 - 22 hold_15_22 - 23 hold_15_23
   - 24 hold_15_24 - 25 hold_15_25 - 26 hold_15_26 + 80 pp_permission_15_14
   <= 80
 constr_internal_compute_15_14: switch_15_14 - group_tp_active_14
   - pp_permission_15_14 <= 0
 constr_pp_permissionA_15_16: start_15 - start_16 + hold_15_1 + 2 hold_15_2
   + 3 hold_15_3 + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6 + 7 hold_15_7
   + 8 hold_15_8 + 9 hold_15_9 + 10 hold_15_10 + 11 hold_15_11
   + 12 hold_15_12 + 13 hold_15_13 + 14 hold_15_14 + 15 hold_15_15
   + 16 hold_15_16 + 17 hold_15_17 + 18 hold_15_18 + 19 hold_15_19
   + 20 hold_15_20 + 21 hold_15_21 + 22 hold_15_22 + 23 hold_15_23
   + 24 hold_15_24 + 25 hold_15_25 + 26 hold_15_26 + 80 pp_permission_15_16
   <= 80
 constr_pp_permissionB_15_16: - start_15 + start_16 - hold_15_1
   - 2 hold_15_2 - 3 hold_15_3 - 4 hold_15_4 - 5 hold_15_5 - 6 hold_15_6
   - 7 hold_15_7 - 8 hold_15_8 - 9 hold_15_9 - 10 hold_15_10
   - 11 hold_15_11 - 12 hold_15_12 - 13 hold_15_13 - 14 hold_15_14
   - 15 hold_15_15 - 16 hold_15_16 - 17 hold_15_17 - 18 hold_15_18
   - 19 hold_15_19 - 20 hold_15_20 - 21 hold_15_21 - 22 hold_15_22
   - 23 hold_15_23 - 24 hold_15_24 - 25 hold_15_25 - 26 hold_15_26
   + 80 pp_permission_15_16 <= 80
 constr_internal_compute_15_16: switch_15_16 - group_tp_active_14
   - pp_permission_15_16 <= 0
 constr_pp_permissionA_15_17: start_15 - start_17 + hold_15_1 + 2 hold_15_2
   + 3 hold_15_3 + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6 + 7 hold_15_7
   + 8 hold_15_8 + 9 hold_15_9 + 10 hold_15_10 + 11 hold_15_11
   + 12 hold_15_12 + 13 hold_15_13 + 14 hold_15_14 + 15 hold_15_15
   + 16 hold_15_16 + 17 hold_15_17 + 18 hold_15_18 + 19 hold_15_19
   + 20 hold_15_20 + 21 hold_15_21 + 22 hold_15_22 + 23 hold_15_23
   + 24 hold_15_24 + 25 hold_15_25 + 26 hold_15_26 + 80 pp_permission_15_17
   <= 80
 constr_pp_permissionB_15_17: - start_15 + start_17 - hold_15_1
   - 2 hold_15_2 - 3 hold_15_3 - 4 hold_15_4 - 5 hold_15_5 - 6 hold_15_6
   - 7 hold_15_7 - 8 hold_15_8 - 9 hold_15_9 - 10 hold_15_10
   - 11 hold_15_11 - 12 hold_15_12 - 13 hold_15_13 - 14 hold_15_14
   - 15 hold_15_15 - 16 hold_15_16 - 17 hold_15_17 - 18 hold_15_18
   - 19 hold_15_19 - 20 hold_15_20 - 21 hold_15_21 - 22 hold_15_22
   - 23 hold_15_23 - 24 hold_15_24 - 25 hold_15_25 - 26 hold_15_26
   + 80 pp_permission_15_17 <= 80
 constr_internal_compute_15_17: switch_15_17 - group_tp_active_14
   - pp_permission_15_17 <= 0
 constr_pp_permissionA_16_14: - start_14 + start_16 + hold_16_1
   + 2 hold_16_2 + 3 hold_16_3 + 4 hold_16_4 + 5 hold_16_5 + 6 hold_16_6
   + 7 hold_16_7 + 8 hold_16_8 + 9 hold_16_9 + 10 hold_16_10
   + 11 hold_16_11 + 12 hold_16_12 + 13 hold_16_13 + 14 hold_16_14
   + 15 hold_16_15 + 16 hold_16_16 + 17 hold_16_17 + 18 hold_16_18
   + 19 hold_16_19 + 20 hold_16_20 + 21 hold_16_21 + 22 hold_16_22
   + 23 hold_16_23 + 24 hold_16_24 + 25 hold_16_25 + 26 hold_16_26
   + 80 pp_permission_16_14 <= 80
 constr_pp_permissionB_16_14: start_14 - start_16 - hold_16_1 - 2 hold_16_2
   - 3 hold_16_3 - 4 hold_16_4 - 5 hold_16_5 - 6 hold_16_6 - 7 hold_16_7
   - 8 hold_16_8 - 9 hold_16_9 - 10 hold_16_10 - 11 hold_16_11
   - 12 hold_16_12 - 13 hold_16_13 - 14 hold_16_14 - 15 hold_16_15
   - 16 hold_16_16 - 17 hold_16_17 - 18 hold_16_18 - 19 hold_16_19
   - 20 hold_16_20 - 21 hold_16_21 - 22 hold_16_22 - 23 hold_16_23
   - 24 hold_16_24 - 25 hold_16_25 - 26 hold_16_26 + 80 pp_permission_16_14
   <= 80
 constr_internal_compute_16_14: switch_16_14 - group_tp_active_14
   - pp_permission_16_14 <= 0
 constr_pp_permissionA_16_15: - start_15 + start_16 + hold_16_1
   + 2 hold_16_2 + 3 hold_16_3 + 4 hold_16_4 + 5 hold_16_5 + 6 hold_16_6
   + 7 hold_16_7 + 8 hold_16_8 + 9 hold_16_9 + 10 hold_16_10
   + 11 hold_16_11 + 12 hold_16_12 + 13 hold_16_13 + 14 hold_16_14
   + 15 hold_16_15 + 16 hold_16_16 + 17 hold_16_17 + 18 hold_16_18
   + 19 hold_16_19 + 20 hold_16_20 + 21 hold_16_21 + 22 hold_16_22
   + 23 hold_16_23 + 24 hold_16_24 + 25 hold_16_25 + 26 hold_16_26
   + 80 pp_permission_16_15 <= 80
 constr_pp_permissionB_16_15: start_15 - start_16 - hold_16_1 - 2 hold_16_2
   - 3 hold_16_3 - 4 hold_16_4 - 5 hold_16_5 - 6 hold_16_6 - 7 hold_16_7
   - 8 hold_16_8 - 9 hold_16_9 - 10 hold_16_10 - 11 hold_16_11
   - 12 hold_16_12 - 13 hold_16_13 - 14 hold_16_14 - 15 hold_16_15
   - 16 hold_16_16 - 17 hold_16_17 - 18 hold_16_18 - 19 hold_16_19
   - 20 hold_16_20 - 21 hold_16_21 - 22 hold_16_22 - 23 hold_16_23
   - 24 hold_16_24 - 25 hold_16_25 - 26 hold_16_26 + 80 pp_permission_16_15
   <= 80
 constr_internal_compute_16_15: switch_16_15 - group_tp_active_14
   - pp_permission_16_15 <= 0
 constr_pp_permissionA_16_17: start_16 - start_17 + hold_16_1 + 2 hold_16_2
   + 3 hold_16_3 + 4 hold_16_4 + 5 hold_16_5 + 6 hold_16_6 + 7 hold_16_7
   + 8 hold_16_8 + 9 hold_16_9 + 10 hold_16_10 + 11 hold_16_11
   + 12 hold_16_12 + 13 hold_16_13 + 14 hold_16_14 + 15 hold_16_15
   + 16 hold_16_16 + 17 hold_16_17 + 18 hold_16_18 + 19 hold_16_19
   + 20 hold_16_20 + 21 hold_16_21 + 22 hold_16_22 + 23 hold_16_23
   + 24 hold_16_24 + 25 hold_16_25 + 26 hold_16_26 + 80 pp_permission_16_17
   <= 80
 constr_pp_permissionB_16_17: - start_16 + start_17 - hold_16_1
   - 2 hold_16_2 - 3 hold_16_3 - 4 hold_16_4 - 5 hold_16_5 - 6 hold_16_6
   - 7 hold_16_7 - 8 hold_16_8 - 9 hold_16_9 - 10 hold_16_10
   - 11 hold_16_11 - 12 hold_16_12 - 13 hold_16_13 - 14 hold_16_14
   - 15 hold_16_15 - 16 hold_16_16 - 17 hold_16_17 - 18 hold_16_18
   - 19 hold_16_19 - 20 hold_16_20 - 21 hold_16_21 - 22 hold_16_22
   - 23 hold_16_23 - 24 hold_16_24 - 25 hold_16_25 - 26 hold_16_26
   + 80 pp_permission_16_17 <= 80
 constr_internal_compute_16_17: switch_16_17 - group_tp_active_14
   - pp_permission_16_17 <= 0
 constr_pp_permissionA_17_14: - start_14 + start_17 + hold_17_1
   + 2 hold_17_2 + 3 hold_17_3 + 4 hold_17_4 + 5 hold_17_5 + 6 hold_17_6
   + 7 hold_17_7 + 8 hold_17_8 + 9 hold_17_9 + 10 hold_17_10
   + 11 hold_17_11 + 12 hold_17_12 + 13 hold_17_13 + 14 hold_17_14
   + 15 hold_17_15 + 16 hold_17_16 + 17 hold_17_17 + 18 hold_17_18
   + 19 hold_17_19 + 20 hold_17_20 + 21 hold_17_21 + 22 hold_17_22
   + 23 hold_17_23 + 24 hold_17_24 + 25 hold_17_25 + 26 hold_17_26
   + 80 pp_permission_17_14 <= 80
 constr_pp_permissionB_17_14: start_14 - start_17 - hold_17_1 - 2 hold_17_2
   - 3 hold_17_3 - 4 hold_17_4 - 5 hold_17_5 - 6 hold_17_6 - 7 hold_17_7
   - 8 hold_17_8 - 9 hold_17_9 - 10 hold_17_10 - 11 hold_17_11
   - 12 hold_17_12 - 13 hold_17_13 - 14 hold_17_14 - 15 hold_17_15
   - 16 hold_17_16 - 17 hold_17_17 - 18 hold_17_18 - 19 hold_17_19
   - 20 hold_17_20 - 21 hold_17_21 - 22 hold_17_22 - 23 hold_17_23
   - 24 hold_17_24 - 25 hold_17_25 - 26 hold_17_26 + 80 pp_permission_17_14
   <= 80
 constr_internal_compute_17_14: switch_17_14 - group_tp_active_14
   - pp_permission_17_14 <= 0
 constr_pp_permissionA_17_15: - start_15 + start_17 + hold_17_1
   + 2 hold_17_2 + 3 hold_17_3 + 4 hold_17_4 + 5 hold_17_5 + 6 hold_17_6
   + 7 hold_17_7 + 8 hold_17_8 + 9 hold_17_9 + 10 hold_17_10
   + 11 hold_17_11 + 12 hold_17_12 + 13 hold_17_13 + 14 hold_17_14
   + 15 hold_17_15 + 16 hold_17_16 + 17 hold_17_17 + 18 hold_17_18
   + 19 hold_17_19 + 20 hold_17_20 + 21 hold_17_21 + 22 hold_17_22
   + 23 hold_17_23 + 24 hold_17_24 + 25 hold_17_25 + 26 hold_17_26
   + 80 pp_permission_17_15 <= 80
 constr_pp_permissionB_17_15: start_15 - start_17 - hold_17_1 - 2 hold_17_2
   - 3 hold_17_3 - 4 hold_17_4 - 5 hold_17_5 - 6 hold_17_6 - 7 hold_17_7
   - 8 hold_17_8 - 9 hold_17_9 - 10 hold_17_10 - 11 hold_17_11
   - 12 hold_17_12 - 13 hold_17_13 - 14 hold_17_14 - 15 hold_17_15
   - 16 hold_17_16 - 17 hold_17_17 - 18 hold_17_18 - 19 hold_17_19
   - 20 hold_17_20 - 21 hold_17_21 - 22 hold_17_22 - 23 hold_17_23
   - 24 hold_17_24 - 25 hold_17_25 - 26 hold_17_26 + 80 pp_permission_17_15
   <= 80
 constr_internal_compute_17_15: switch_17_15 - group_tp_active_14
   - pp_permission_17_15 <= 0
 constr_pp_permissionA_17_16: - start_16 + start_17 + hold_17_1
   + 2 hold_17_2 + 3 hold_17_3 + 4 hold_17_4 + 5 hold_17_5 + 6 hold_17_6
   + 7 hold_17_7 + 8 hold_17_8 + 9 hold_17_9 + 10 hold_17_10
   + 11 hold_17_11 + 12 hold_17_12 + 13 hold_17_13 + 14 hold_17_14
   + 15 hold_17_15 + 16 hold_17_16 + 17 hold_17_17 + 18 hold_17_18
   + 19 hold_17_19 + 20 hold_17_20 + 21 hold_17_21 + 22 hold_17_22
   + 23 hold_17_23 + 24 hold_17_24 + 25 hold_17_25 + 26 hold_17_26
   + 80 pp_permission_17_16 <= 80
 constr_pp_permissionB_17_16: start_16 - start_17 - hold_17_1 - 2 hold_17_2
   - 3 hold_17_3 - 4 hold_17_4 - 5 hold_17_5 - 6 hold_17_6 - 7 hold_17_7
   - 8 hold_17_8 - 9 hold_17_9 - 10 hold_17_10 - 11 hold_17_11
   - 12 hold_17_12 - 13 hold_17_13 - 14 hold_17_14 - 15 hold_17_15
   - 16 hold_17_16 - 17 hold_17_17 - 18 hold_17_18 - 19 hold_17_19
   - 20 hold_17_20 - 21 hold_17_21 - 22 hold_17_22 - 23 hold_17_23
   - 24 hold_17_24 - 25 hold_17_25 - 26 hold_17_26 + 80 pp_permission_17_16
   <= 80
 constr_internal_compute_17_16: switch_17_16 - group_tp_active_14
   - pp_permission_17_16 <= 0
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
 link_activation_dst_nic_in_14_15: switch_nic_in_14_15 - is_active_15 <= 0
 link_activation_src_15_nic_out_14: switch_15_nic_out_14 - is_active_15
   <= 0
 link_activation_dst_nic_in_14_16: switch_nic_in_14_16 - is_active_16 <= 0
 link_activation_src_16_nic_out_14: switch_16_nic_out_14 - is_active_16
   <= 0
 link_activation_dst_nic_in_14_17: switch_nic_in_14_17 - is_active_17 <= 0
 link_activation_src_17_nic_out_14: switch_17_nic_out_14 - is_active_17
   <= 0
 link_activation_src_14_15: switch_14_15 - is_active_14 <= 0
 link_activation_dst_14_15: switch_14_15 - is_active_15 <= 0
 link_activation_src_14_16: switch_14_16 - is_active_14 <= 0
 link_activation_dst_14_16: switch_14_16 - is_active_16 <= 0
 link_activation_src_14_17: switch_14_17 - is_active_14 <= 0
 link_activation_dst_14_17: switch_14_17 - is_active_17 <= 0
 link_activation_src_15_14: switch_15_14 - is_active_15 <= 0
 link_activation_dst_15_14: switch_15_14 - is_active_14 <= 0
 link_activation_src_15_16: switch_15_16 - is_active_15 <= 0
 link_activation_dst_15_16: switch_15_16 - is_active_16 <= 0
 link_activation_src_15_17: switch_15_17 - is_active_15 <= 0
 link_activation_dst_15_17: switch_15_17 - is_active_17 <= 0
 link_activation_src_16_14: switch_16_14 - is_active_16 <= 0
 link_activation_dst_16_14: switch_16_14 - is_active_14 <= 0
 link_activation_src_16_15: switch_16_15 - is_active_16 <= 0
 link_activation_dst_16_15: switch_16_15 - is_active_15 <= 0
 link_activation_src_16_17: switch_16_17 - is_active_16 <= 0
 link_activation_dst_16_17: switch_16_17 - is_active_17 <= 0
 link_activation_src_17_14: switch_17_14 - is_active_17 <= 0
 link_activation_dst_17_14: switch_17_14 - is_active_14 <= 0
 link_activation_src_17_15: switch_17_15 - is_active_17 <= 0
 link_activation_dst_17_15: switch_17_15 - is_active_15 <= 0
 link_activation_src_17_16: switch_17_16 - is_active_17 <= 0
 link_activation_dst_17_16: switch_17_16 - is_active_16 <= 0
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
 edge_flow_constr_nic_out_0_nic_in_14: flow_nic_out_0_nic_in_14
   - 61035.15625 switch_nic_out_0_nic_in_14 <= 0
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
 edge_flow_constr_nic_out_1_nic_in_6: flow_nic_out_1_nic_in_6
   - 61035.15625 switch_nic_out_1_nic_in_6 <= 0
 edge_flow_constr_nic_out_1_nic_in_7: flow_nic_out_1_nic_in_7
   - 61035.15625 switch_nic_out_1_nic_in_7 <= 0
 edge_flow_constr_nic_out_1_nic_in_8: flow_nic_out_1_nic_in_8
   - 61035.15625 switch_nic_out_1_nic_in_8 <= 0
 edge_flow_constr_nic_out_1_nic_in_10: flow_nic_out_1_nic_in_10
   - 61035.15625 switch_nic_out_1_nic_in_10 <= 0
 edge_flow_constr_nic_out_1_nic_in_12: flow_nic_out_1_nic_in_12
   - 61035.15625 switch_nic_out_1_nic_in_12 <= 0
 edge_flow_constr_nic_out_1_nic_in_14: flow_nic_out_1_nic_in_14
   - 61035.15625 switch_nic_out_1_nic_in_14 <= 0
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
 edge_flow_constr_nic_out_2_nic_in_6: flow_nic_out_2_nic_in_6
   - 61035.15625 switch_nic_out_2_nic_in_6 <= 0
 edge_flow_constr_nic_out_2_nic_in_7: flow_nic_out_2_nic_in_7
   - 61035.15625 switch_nic_out_2_nic_in_7 <= 0
 edge_flow_constr_nic_out_2_nic_in_8: flow_nic_out_2_nic_in_8
   - 61035.15625 switch_nic_out_2_nic_in_8 <= 0
 edge_flow_constr_nic_out_2_nic_in_10: flow_nic_out_2_nic_in_10
   - 61035.15625 switch_nic_out_2_nic_in_10 <= 0
 edge_flow_constr_nic_out_2_nic_in_12: flow_nic_out_2_nic_in_12
   - 61035.15625 switch_nic_out_2_nic_in_12 <= 0
 edge_flow_constr_nic_out_2_nic_in_14: flow_nic_out_2_nic_in_14
   - 61035.15625 switch_nic_out_2_nic_in_14 <= 0
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
 edge_flow_constr_nic_out_3_nic_in_6: flow_nic_out_3_nic_in_6
   - 61035.15625 switch_nic_out_3_nic_in_6 <= 0
 edge_flow_constr_nic_out_3_nic_in_7: flow_nic_out_3_nic_in_7
   - 61035.15625 switch_nic_out_3_nic_in_7 <= 0
 edge_flow_constr_nic_out_3_nic_in_8: flow_nic_out_3_nic_in_8
   - 61035.15625 switch_nic_out_3_nic_in_8 <= 0
 edge_flow_constr_nic_out_3_nic_in_10: flow_nic_out_3_nic_in_10
   - 61035.15625 switch_nic_out_3_nic_in_10 <= 0
 edge_flow_constr_nic_out_3_nic_in_12: flow_nic_out_3_nic_in_12
   - 61035.15625 switch_nic_out_3_nic_in_12 <= 0
 edge_flow_constr_nic_out_3_nic_in_14: flow_nic_out_3_nic_in_14
   - 61035.15625 switch_nic_out_3_nic_in_14 <= 0
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
 edge_flow_constr_nic_out_4_nic_in_14: flow_nic_out_4_nic_in_14
   - 61035.15625 switch_nic_out_4_nic_in_14 <= 0
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
 edge_flow_constr_nic_out_6_nic_in_1: flow_nic_out_6_nic_in_1
   - 61035.15625 switch_nic_out_6_nic_in_1 <= 0
 edge_flow_constr_nic_out_6_nic_in_2: flow_nic_out_6_nic_in_2
   - 61035.15625 switch_nic_out_6_nic_in_2 <= 0
 edge_flow_constr_nic_out_6_nic_in_3: flow_nic_out_6_nic_in_3
   - 61035.15625 switch_nic_out_6_nic_in_3 <= 0
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
 edge_flow_constr_nic_out_6_nic_in_14: flow_nic_out_6_nic_in_14
   - 61035.15625 switch_nic_out_6_nic_in_14 <= 0
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
 edge_flow_constr_nic_out_7_nic_in_6: flow_nic_out_7_nic_in_6
   - 61035.15625 switch_nic_out_7_nic_in_6 <= 0
 edge_flow_constr_nic_out_7_nic_in_8: flow_nic_out_7_nic_in_8
   - 61035.15625 switch_nic_out_7_nic_in_8 <= 0
 edge_flow_constr_nic_out_7_nic_in_10: flow_nic_out_7_nic_in_10
   - 61035.15625 switch_nic_out_7_nic_in_10 <= 0
 edge_flow_constr_nic_out_7_nic_in_12: flow_nic_out_7_nic_in_12
   - 61035.15625 switch_nic_out_7_nic_in_12 <= 0
 edge_flow_constr_nic_out_7_nic_in_14: flow_nic_out_7_nic_in_14
   - 61035.15625 switch_nic_out_7_nic_in_14 <= 0
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
 edge_flow_constr_nic_out_8_nic_in_6: flow_nic_out_8_nic_in_6
   - 61035.15625 switch_nic_out_8_nic_in_6 <= 0
 edge_flow_constr_nic_out_8_nic_in_7: flow_nic_out_8_nic_in_7
   - 61035.15625 switch_nic_out_8_nic_in_7 <= 0
 edge_flow_constr_nic_out_8_nic_in_10: flow_nic_out_8_nic_in_10
   - 61035.15625 switch_nic_out_8_nic_in_10 <= 0
 edge_flow_constr_nic_out_8_nic_in_12: flow_nic_out_8_nic_in_12
   - 61035.15625 switch_nic_out_8_nic_in_12 <= 0
 edge_flow_constr_nic_out_8_nic_in_14: flow_nic_out_8_nic_in_14
   - 61035.15625 switch_nic_out_8_nic_in_14 <= 0
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
 edge_flow_constr_nic_out_10_nic_in_1: flow_nic_out_10_nic_in_1
   - 61035.15625 switch_nic_out_10_nic_in_1 <= 0
 edge_flow_constr_nic_out_10_nic_in_2: flow_nic_out_10_nic_in_2
   - 61035.15625 switch_nic_out_10_nic_in_2 <= 0
 edge_flow_constr_nic_out_10_nic_in_3: flow_nic_out_10_nic_in_3
   - 61035.15625 switch_nic_out_10_nic_in_3 <= 0
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
 edge_flow_constr_nic_out_10_nic_in_14: flow_nic_out_10_nic_in_14
   - 61035.15625 switch_nic_out_10_nic_in_14 <= 0
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
 edge_flow_constr_nic_out_12_nic_in_1: flow_nic_out_12_nic_in_1
   - 61035.15625 switch_nic_out_12_nic_in_1 <= 0
 edge_flow_constr_nic_out_12_nic_in_2: flow_nic_out_12_nic_in_2
   - 61035.15625 switch_nic_out_12_nic_in_2 <= 0
 edge_flow_constr_nic_out_12_nic_in_3: flow_nic_out_12_nic_in_3
   - 61035.15625 switch_nic_out_12_nic_in_3 <= 0
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
 edge_flow_constr_nic_out_12_nic_in_14: flow_nic_out_12_nic_in_14
   - 61035.15625 switch_nic_out_12_nic_in_14 <= 0
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
 edge_flow_constr_nic_out_14_nic_in_1: flow_nic_out_14_nic_in_1
   - 61035.15625 switch_nic_out_14_nic_in_1 <= 0
 edge_flow_constr_nic_out_14_nic_in_2: flow_nic_out_14_nic_in_2
   - 61035.15625 switch_nic_out_14_nic_in_2 <= 0
 edge_flow_constr_nic_out_14_nic_in_3: flow_nic_out_14_nic_in_3
   - 61035.15625 switch_nic_out_14_nic_in_3 <= 0
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
 edge_flow_constr_nic_in_14_14: flow_nic_in_14_14
   - 3.90625e+06 switch_nic_in_14_14 <= 0
 edge_flow_constr_14_nic_out_14: flow_14_nic_out_14
   - 3.90625e+06 switch_14_nic_out_14 <= 0
 edge_flow_constr_nic_in_14_15: flow_nic_in_14_15
   - 3.90625e+06 switch_nic_in_14_15 <= 0
 edge_flow_constr_15_nic_out_14: flow_15_nic_out_14
   - 3.90625e+06 switch_15_nic_out_14 <= 0
 edge_flow_constr_nic_in_14_16: flow_nic_in_14_16
   - 3.90625e+06 switch_nic_in_14_16 <= 0
 edge_flow_constr_16_nic_out_14: flow_16_nic_out_14
   - 3.90625e+06 switch_16_nic_out_14 <= 0
 edge_flow_constr_nic_in_14_17: flow_nic_in_14_17
   - 3.90625e+06 switch_nic_in_14_17 <= 0
 edge_flow_constr_17_nic_out_14: flow_17_nic_out_14
   - 3.90625e+06 switch_17_nic_out_14 <= 0
 edge_flow_constr_14_15: flow_14_15 - 3.90625e+06 switch_14_15 <= 0
 edge_flow_constr_14_16: flow_14_16 - 3.90625e+06 switch_14_16 <= 0
 edge_flow_constr_14_17: flow_14_17 - 3.90625e+06 switch_14_17 <= 0
 edge_flow_constr_15_14: flow_15_14 - 3.90625e+06 switch_15_14 <= 0
 edge_flow_constr_15_16: flow_15_16 - 3.90625e+06 switch_15_16 <= 0
 edge_flow_constr_15_17: flow_15_17 - 3.90625e+06 switch_15_17 <= 0
 edge_flow_constr_16_14: flow_16_14 - 3.90625e+06 switch_16_14 <= 0
 edge_flow_constr_16_15: flow_16_15 - 3.90625e+06 switch_16_15 <= 0
 edge_flow_constr_16_17: flow_16_17 - 3.90625e+06 switch_16_17 <= 0
 edge_flow_constr_17_14: flow_17_14 - 3.90625e+06 switch_17_14 <= 0
 edge_flow_constr_17_15: flow_17_15 - 3.90625e+06 switch_17_15 <= 0
 edge_flow_constr_17_16: flow_17_16 - 3.90625e+06 switch_17_16 <= 0
 force_group_off_constr_0: group_tp_active_0 = 0
 force_group_off_constr_1: group_tp_active_1 = 0
 force_group_off_constr_2: group_tp_active_2 = 0
 force_group_off_constr_3: group_tp_active_3 = 0
 force_group_off_constr_6: group_tp_active_6 = 0
 force_group_off_constr_7: group_tp_active_7 = 0
 R720: - group_tp_active_4 + tp_on_and_switch_nic_out_0_nic_in_4_4 <= 0
 R721: - switch_nic_out_0_nic_in_4 + tp_on_and_switch_nic_out_0_nic_in_4_4
   <= 0
 R722: - switch_nic_out_0_nic_in_4 - group_tp_active_4
   + tp_on_and_switch_nic_out_0_nic_in_4_4 >= -1
 R723: - group_tp_active_8 + tp_on_and_switch_nic_out_0_nic_in_8_8 <= 0
 R724: - switch_nic_out_0_nic_in_8 + tp_on_and_switch_nic_out_0_nic_in_8_8
   <= 0
 R725: - switch_nic_out_0_nic_in_8 - group_tp_active_8
   + tp_on_and_switch_nic_out_0_nic_in_8_8 >= -1
 R726: - group_tp_active_10 + tp_on_and_switch_nic_out_0_nic_in_10_10 <= 0
 R727: - switch_nic_out_0_nic_in_10
   + tp_on_and_switch_nic_out_0_nic_in_10_10 <= 0
 R728: - switch_nic_out_0_nic_in_10 - group_tp_active_10
   + tp_on_and_switch_nic_out_0_nic_in_10_10 >= -1
 R729: - group_tp_active_12 + tp_on_and_switch_nic_out_0_nic_in_12_12 <= 0
 R730: - switch_nic_out_0_nic_in_12
   + tp_on_and_switch_nic_out_0_nic_in_12_12 <= 0
 R731: - switch_nic_out_0_nic_in_12 - group_tp_active_12
   + tp_on_and_switch_nic_out_0_nic_in_12_12 >= -1
 R732: - group_tp_active_14 + tp_on_and_switch_nic_out_0_nic_in_14_14 <= 0
 R733: - switch_nic_out_0_nic_in_14
   + tp_on_and_switch_nic_out_0_nic_in_14_14 <= 0
 R734: - switch_nic_out_0_nic_in_14 - group_tp_active_14
   + tp_on_and_switch_nic_out_0_nic_in_14_14 >= -1
 R735: - group_tp_active_4 + tp_on_and_switch_nic_out_1_nic_in_4_4 <= 0
 R736: - switch_nic_out_1_nic_in_4 + tp_on_and_switch_nic_out_1_nic_in_4_4
   <= 0
 R737: - switch_nic_out_1_nic_in_4 - group_tp_active_4
   + tp_on_and_switch_nic_out_1_nic_in_4_4 >= -1
 R738: - group_tp_active_8 + tp_on_and_switch_nic_out_1_nic_in_8_8 <= 0
 R739: - switch_nic_out_1_nic_in_8 + tp_on_and_switch_nic_out_1_nic_in_8_8
   <= 0
 R740: - switch_nic_out_1_nic_in_8 - group_tp_active_8
   + tp_on_and_switch_nic_out_1_nic_in_8_8 >= -1
 R741: - group_tp_active_10 + tp_on_and_switch_nic_out_1_nic_in_10_10 <= 0
 R742: - switch_nic_out_1_nic_in_10
   + tp_on_and_switch_nic_out_1_nic_in_10_10 <= 0
 R743: - switch_nic_out_1_nic_in_10 - group_tp_active_10
   + tp_on_and_switch_nic_out_1_nic_in_10_10 >= -1
 R744: - group_tp_active_12 + tp_on_and_switch_nic_out_1_nic_in_12_12 <= 0
 R745: - switch_nic_out_1_nic_in_12
   + tp_on_and_switch_nic_out_1_nic_in_12_12 <= 0
 R746: - switch_nic_out_1_nic_in_12 - group_tp_active_12
   + tp_on_and_switch_nic_out_1_nic_in_12_12 >= -1
 R747: - group_tp_active_14 + tp_on_and_switch_nic_out_1_nic_in_14_14 <= 0
 R748: - switch_nic_out_1_nic_in_14
   + tp_on_and_switch_nic_out_1_nic_in_14_14 <= 0
 R749: - switch_nic_out_1_nic_in_14 - group_tp_active_14
   + tp_on_and_switch_nic_out_1_nic_in_14_14 >= -1
 R750: - group_tp_active_4 + tp_on_and_switch_nic_out_2_nic_in_4_4 <= 0
 R751: - switch_nic_out_2_nic_in_4 + tp_on_and_switch_nic_out_2_nic_in_4_4
   <= 0
 R752: - switch_nic_out_2_nic_in_4 - group_tp_active_4
   + tp_on_and_switch_nic_out_2_nic_in_4_4 >= -1
 R753: - group_tp_active_8 + tp_on_and_switch_nic_out_2_nic_in_8_8 <= 0
 R754: - switch_nic_out_2_nic_in_8 + tp_on_and_switch_nic_out_2_nic_in_8_8
   <= 0
 R755: - switch_nic_out_2_nic_in_8 - group_tp_active_8
   + tp_on_and_switch_nic_out_2_nic_in_8_8 >= -1
 R756: - group_tp_active_10 + tp_on_and_switch_nic_out_2_nic_in_10_10 <= 0
 R757: - switch_nic_out_2_nic_in_10
   + tp_on_and_switch_nic_out_2_nic_in_10_10 <= 0
 R758: - switch_nic_out_2_nic_in_10 - group_tp_active_10
   + tp_on_and_switch_nic_out_2_nic_in_10_10 >= -1
 R759: - group_tp_active_12 + tp_on_and_switch_nic_out_2_nic_in_12_12 <= 0
 R760: - switch_nic_out_2_nic_in_12
   + tp_on_and_switch_nic_out_2_nic_in_12_12 <= 0
 R761: - switch_nic_out_2_nic_in_12 - group_tp_active_12
   + tp_on_and_switch_nic_out_2_nic_in_12_12 >= -1
 R762: - group_tp_active_14 + tp_on_and_switch_nic_out_2_nic_in_14_14 <= 0
 R763: - switch_nic_out_2_nic_in_14
   + tp_on_and_switch_nic_out_2_nic_in_14_14 <= 0
 R764: - switch_nic_out_2_nic_in_14 - group_tp_active_14
   + tp_on_and_switch_nic_out_2_nic_in_14_14 >= -1
 R765: - group_tp_active_4 + tp_on_and_switch_nic_out_3_nic_in_4_4 <= 0
 R766: - switch_nic_out_3_nic_in_4 + tp_on_and_switch_nic_out_3_nic_in_4_4
   <= 0
 R767: - switch_nic_out_3_nic_in_4 - group_tp_active_4
   + tp_on_and_switch_nic_out_3_nic_in_4_4 >= -1
 R768: - group_tp_active_8 + tp_on_and_switch_nic_out_3_nic_in_8_8 <= 0
 R769: - switch_nic_out_3_nic_in_8 + tp_on_and_switch_nic_out_3_nic_in_8_8
   <= 0
 R770: - switch_nic_out_3_nic_in_8 - group_tp_active_8
   + tp_on_and_switch_nic_out_3_nic_in_8_8 >= -1
 R771: - group_tp_active_10 + tp_on_and_switch_nic_out_3_nic_in_10_10 <= 0
 R772: - switch_nic_out_3_nic_in_10
   + tp_on_and_switch_nic_out_3_nic_in_10_10 <= 0
 R773: - switch_nic_out_3_nic_in_10 - group_tp_active_10
   + tp_on_and_switch_nic_out_3_nic_in_10_10 >= -1
 R774: - group_tp_active_12 + tp_on_and_switch_nic_out_3_nic_in_12_12 <= 0
 R775: - switch_nic_out_3_nic_in_12
   + tp_on_and_switch_nic_out_3_nic_in_12_12 <= 0
 R776: - switch_nic_out_3_nic_in_12 - group_tp_active_12
   + tp_on_and_switch_nic_out_3_nic_in_12_12 >= -1
 R777: - group_tp_active_14 + tp_on_and_switch_nic_out_3_nic_in_14_14 <= 0
 R778: - switch_nic_out_3_nic_in_14
   + tp_on_and_switch_nic_out_3_nic_in_14_14 <= 0
 R779: - switch_nic_out_3_nic_in_14 - group_tp_active_14
   + tp_on_and_switch_nic_out_3_nic_in_14_14 >= -1
 R780: - group_tp_active_4 + tp_on_and_switch_source_nic_in_4_4 <= 0
 R781: - switch_source_nic_in_4 + tp_on_and_switch_source_nic_in_4_4 <= 0
 R782: - switch_source_nic_in_4 - group_tp_active_4
   + tp_on_and_switch_source_nic_in_4_4 >= -1
 R783: - group_tp_active_8 + tp_on_and_switch_nic_out_4_nic_in_8_8 <= 0
 R784: - switch_nic_out_4_nic_in_8 + tp_on_and_switch_nic_out_4_nic_in_8_8
   <= 0
 R785: - switch_nic_out_4_nic_in_8 - group_tp_active_8
   + tp_on_and_switch_nic_out_4_nic_in_8_8 >= -1
 R786: - group_tp_active_10 + tp_on_and_switch_nic_out_4_nic_in_10_10 <= 0
 R787: - switch_nic_out_4_nic_in_10
   + tp_on_and_switch_nic_out_4_nic_in_10_10 <= 0
 R788: - switch_nic_out_4_nic_in_10 - group_tp_active_10
   + tp_on_and_switch_nic_out_4_nic_in_10_10 >= -1
 R789: - group_tp_active_12 + tp_on_and_switch_nic_out_4_nic_in_12_12 <= 0
 R790: - switch_nic_out_4_nic_in_12
   + tp_on_and_switch_nic_out_4_nic_in_12_12 <= 0
 R791: - switch_nic_out_4_nic_in_12 - group_tp_active_12
   + tp_on_and_switch_nic_out_4_nic_in_12_12 >= -1
 R792: - group_tp_active_14 + tp_on_and_switch_nic_out_4_nic_in_14_14 <= 0
 R793: - switch_nic_out_4_nic_in_14
   + tp_on_and_switch_nic_out_4_nic_in_14_14 <= 0
 R794: - switch_nic_out_4_nic_in_14 - group_tp_active_14
   + tp_on_and_switch_nic_out_4_nic_in_14_14 >= -1
 R795: - group_tp_active_4 + tp_on_and_switch_nic_out_6_nic_in_4_4 <= 0
 R796: - switch_nic_out_6_nic_in_4 + tp_on_and_switch_nic_out_6_nic_in_4_4
   <= 0
 R797: - switch_nic_out_6_nic_in_4 - group_tp_active_4
   + tp_on_and_switch_nic_out_6_nic_in_4_4 >= -1
 R798: - group_tp_active_8 + tp_on_and_switch_nic_out_6_nic_in_8_8 <= 0
 R799: - switch_nic_out_6_nic_in_8 + tp_on_and_switch_nic_out_6_nic_in_8_8
   <= 0
 R800: - switch_nic_out_6_nic_in_8 - group_tp_active_8
   + tp_on_and_switch_nic_out_6_nic_in_8_8 >= -1
 R801: - group_tp_active_10 + tp_on_and_switch_nic_out_6_nic_in_10_10 <= 0
 R802: - switch_nic_out_6_nic_in_10
   + tp_on_and_switch_nic_out_6_nic_in_10_10 <= 0
 R803: - switch_nic_out_6_nic_in_10 - group_tp_active_10
   + tp_on_and_switch_nic_out_6_nic_in_10_10 >= -1
 R804: - group_tp_active_12 + tp_on_and_switch_nic_out_6_nic_in_12_12 <= 0
 R805: - switch_nic_out_6_nic_in_12
   + tp_on_and_switch_nic_out_6_nic_in_12_12 <= 0
 R806: - switch_nic_out_6_nic_in_12 - group_tp_active_12
   + tp_on_and_switch_nic_out_6_nic_in_12_12 >= -1
 R807: - group_tp_active_14 + tp_on_and_switch_nic_out_6_nic_in_14_14 <= 0
 R808: - switch_nic_out_6_nic_in_14
   + tp_on_and_switch_nic_out_6_nic_in_14_14 <= 0
 R809: - switch_nic_out_6_nic_in_14 - group_tp_active_14
   + tp_on_and_switch_nic_out_6_nic_in_14_14 >= -1
 R810: - group_tp_active_4 + tp_on_and_switch_nic_out_7_nic_in_4_4 <= 0
 R811: - switch_nic_out_7_nic_in_4 + tp_on_and_switch_nic_out_7_nic_in_4_4
   <= 0
 R812: - switch_nic_out_7_nic_in_4 - group_tp_active_4
   + tp_on_and_switch_nic_out_7_nic_in_4_4 >= -1
 R813: - group_tp_active_8 + tp_on_and_switch_nic_out_7_nic_in_8_8 <= 0
 R814: - switch_nic_out_7_nic_in_8 + tp_on_and_switch_nic_out_7_nic_in_8_8
   <= 0
 R815: - switch_nic_out_7_nic_in_8 - group_tp_active_8
   + tp_on_and_switch_nic_out_7_nic_in_8_8 >= -1
 R816: - group_tp_active_10 + tp_on_and_switch_nic_out_7_nic_in_10_10 <= 0
 R817: - switch_nic_out_7_nic_in_10
   + tp_on_and_switch_nic_out_7_nic_in_10_10 <= 0
 R818: - switch_nic_out_7_nic_in_10 - group_tp_active_10
   + tp_on_and_switch_nic_out_7_nic_in_10_10 >= -1
 R819: - group_tp_active_12 + tp_on_and_switch_nic_out_7_nic_in_12_12 <= 0
 R820: - switch_nic_out_7_nic_in_12
   + tp_on_and_switch_nic_out_7_nic_in_12_12 <= 0
 R821: - switch_nic_out_7_nic_in_12 - group_tp_active_12
   + tp_on_and_switch_nic_out_7_nic_in_12_12 >= -1
 R822: - group_tp_active_14 + tp_on_and_switch_nic_out_7_nic_in_14_14 <= 0
 R823: - switch_nic_out_7_nic_in_14
   + tp_on_and_switch_nic_out_7_nic_in_14_14 <= 0
 R824: - switch_nic_out_7_nic_in_14 - group_tp_active_14
   + tp_on_and_switch_nic_out_7_nic_in_14_14 >= -1
 R825: - group_tp_active_8 + tp_on_and_switch_source_nic_in_8_8 <= 0
 R826: - switch_source_nic_in_8 + tp_on_and_switch_source_nic_in_8_8 <= 0
 R827: - switch_source_nic_in_8 - group_tp_active_8
   + tp_on_and_switch_source_nic_in_8_8 >= -1
 R828: - group_tp_active_4 + tp_on_and_switch_nic_out_8_nic_in_4_4 <= 0
 R829: - switch_nic_out_8_nic_in_4 + tp_on_and_switch_nic_out_8_nic_in_4_4
   <= 0
 R830: - switch_nic_out_8_nic_in_4 - group_tp_active_4
   + tp_on_and_switch_nic_out_8_nic_in_4_4 >= -1
 R831: - group_tp_active_10 + tp_on_and_switch_nic_out_8_nic_in_10_10 <= 0
 R832: - switch_nic_out_8_nic_in_10
   + tp_on_and_switch_nic_out_8_nic_in_10_10 <= 0
 R833: - switch_nic_out_8_nic_in_10 - group_tp_active_10
   + tp_on_and_switch_nic_out_8_nic_in_10_10 >= -1
 R834: - group_tp_active_12 + tp_on_and_switch_nic_out_8_nic_in_12_12 <= 0
 R835: - switch_nic_out_8_nic_in_12
   + tp_on_and_switch_nic_out_8_nic_in_12_12 <= 0
 R836: - switch_nic_out_8_nic_in_12 - group_tp_active_12
   + tp_on_and_switch_nic_out_8_nic_in_12_12 >= -1
 R837: - group_tp_active_14 + tp_on_and_switch_nic_out_8_nic_in_14_14 <= 0
 R838: - switch_nic_out_8_nic_in_14
   + tp_on_and_switch_nic_out_8_nic_in_14_14 <= 0
 R839: - switch_nic_out_8_nic_in_14 - group_tp_active_14
   + tp_on_and_switch_nic_out_8_nic_in_14_14 >= -1
 R840: - group_tp_active_10 + tp_on_and_switch_source_nic_in_10_10 <= 0
 R841: - switch_source_nic_in_10 + tp_on_and_switch_source_nic_in_10_10
   <= 0
 R842: - switch_source_nic_in_10 - group_tp_active_10
   + tp_on_and_switch_source_nic_in_10_10 >= -1
 R843: - group_tp_active_4 + tp_on_and_switch_nic_out_10_nic_in_4_4 <= 0
 R844: - switch_nic_out_10_nic_in_4
   + tp_on_and_switch_nic_out_10_nic_in_4_4 <= 0
 R845: - switch_nic_out_10_nic_in_4 - group_tp_active_4
   + tp_on_and_switch_nic_out_10_nic_in_4_4 >= -1
 R846: - group_tp_active_8 + tp_on_and_switch_nic_out_10_nic_in_8_8 <= 0
 R847: - switch_nic_out_10_nic_in_8
   + tp_on_and_switch_nic_out_10_nic_in_8_8 <= 0
 R848: - switch_nic_out_10_nic_in_8 - group_tp_active_8
   + tp_on_and_switch_nic_out_10_nic_in_8_8 >= -1
 R849: - group_tp_active_12 + tp_on_and_switch_nic_out_10_nic_in_12_12
   <= 0
 R850: - switch_nic_out_10_nic_in_12
   + tp_on_and_switch_nic_out_10_nic_in_12_12 <= 0
 R851: - switch_nic_out_10_nic_in_12 - group_tp_active_12
   + tp_on_and_switch_nic_out_10_nic_in_12_12 >= -1
 R852: - group_tp_active_14 + tp_on_and_switch_nic_out_10_nic_in_14_14
   <= 0
 R853: - switch_nic_out_10_nic_in_14
   + tp_on_and_switch_nic_out_10_nic_in_14_14 <= 0
 R854: - switch_nic_out_10_nic_in_14 - group_tp_active_14
   + tp_on_and_switch_nic_out_10_nic_in_14_14 >= -1
 R855: - group_tp_active_12 + tp_on_and_switch_source_nic_in_12_12 <= 0
 R856: - switch_source_nic_in_12 + tp_on_and_switch_source_nic_in_12_12
   <= 0
 R857: - switch_source_nic_in_12 - group_tp_active_12
   + tp_on_and_switch_source_nic_in_12_12 >= -1
 R858: - group_tp_active_4 + tp_on_and_switch_nic_out_12_nic_in_4_4 <= 0
 R859: - switch_nic_out_12_nic_in_4
   + tp_on_and_switch_nic_out_12_nic_in_4_4 <= 0
 R860: - switch_nic_out_12_nic_in_4 - group_tp_active_4
   + tp_on_and_switch_nic_out_12_nic_in_4_4 >= -1
 R861: - group_tp_active_8 + tp_on_and_switch_nic_out_12_nic_in_8_8 <= 0
 R862: - switch_nic_out_12_nic_in_8
   + tp_on_and_switch_nic_out_12_nic_in_8_8 <= 0
 R863: - switch_nic_out_12_nic_in_8 - group_tp_active_8
   + tp_on_and_switch_nic_out_12_nic_in_8_8 >= -1
 R864: - group_tp_active_10 + tp_on_and_switch_nic_out_12_nic_in_10_10
   <= 0
 R865: - switch_nic_out_12_nic_in_10
   + tp_on_and_switch_nic_out_12_nic_in_10_10 <= 0
 R866: - switch_nic_out_12_nic_in_10 - group_tp_active_10
   + tp_on_and_switch_nic_out_12_nic_in_10_10 >= -1
 R867: - group_tp_active_14 + tp_on_and_switch_nic_out_12_nic_in_14_14
   <= 0
 R868: - switch_nic_out_12_nic_in_14
   + tp_on_and_switch_nic_out_12_nic_in_14_14 <= 0
 R869: - switch_nic_out_12_nic_in_14 - group_tp_active_14
   + tp_on_and_switch_nic_out_12_nic_in_14_14 >= -1
 R870: - group_tp_active_14 + tp_on_and_switch_source_nic_in_14_14 <= 0
 R871: - switch_source_nic_in_14 + tp_on_and_switch_source_nic_in_14_14
   <= 0
 R872: - switch_source_nic_in_14 - group_tp_active_14
   + tp_on_and_switch_source_nic_in_14_14 >= -1
 R873: - group_tp_active_4 + tp_on_and_switch_nic_out_14_nic_in_4_4 <= 0
 R874: - switch_nic_out_14_nic_in_4
   + tp_on_and_switch_nic_out_14_nic_in_4_4 <= 0
 R875: - switch_nic_out_14_nic_in_4 - group_tp_active_4
   + tp_on_and_switch_nic_out_14_nic_in_4_4 >= -1
 R876: - group_tp_active_8 + tp_on_and_switch_nic_out_14_nic_in_8_8 <= 0
 R877: - switch_nic_out_14_nic_in_8
   + tp_on_and_switch_nic_out_14_nic_in_8_8 <= 0
 R878: - switch_nic_out_14_nic_in_8 - group_tp_active_8
   + tp_on_and_switch_nic_out_14_nic_in_8_8 >= -1
 R879: - group_tp_active_10 + tp_on_and_switch_nic_out_14_nic_in_10_10
   <= 0
 R880: - switch_nic_out_14_nic_in_10
   + tp_on_and_switch_nic_out_14_nic_in_10_10 <= 0
 R881: - switch_nic_out_14_nic_in_10 - group_tp_active_10
   + tp_on_and_switch_nic_out_14_nic_in_10_10 >= -1
 R882: - group_tp_active_12 + tp_on_and_switch_nic_out_14_nic_in_12_12
   <= 0
 R883: - switch_nic_out_14_nic_in_12
   + tp_on_and_switch_nic_out_14_nic_in_12_12 <= 0
 R884: - switch_nic_out_14_nic_in_12 - group_tp_active_12
   + tp_on_and_switch_nic_out_14_nic_in_12_12 >= -1
 force_min_throughput: flow_source_nic_in_0 + flow_source_nic_in_1
   + flow_source_nic_in_2 + flow_source_nic_in_3 + flow_source_nic_in_4
   + flow_source_nic_in_6 + flow_source_nic_in_7 + flow_source_nic_in_8
   + flow_source_nic_in_10 + flow_source_nic_in_12 + flow_source_nic_in_14
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
 node_throughput_constraint_0: group_tp_active_0 = 0 ->
   - 15314.28571428572 hold_0_1 - 7657.142857142858 hold_0_2
   - 5104.761904761905 hold_0_3 - 3828.571428571429 hold_0_4
   - 3062.857142857143 hold_0_5 - 2552.380952380952 hold_0_6
   - 2187.755102040816 hold_0_7 - 1914.285714285714 hold_0_8
   - 1701.587301587301 hold_0_9 - 1531.428571428571 hold_0_10
   - 1392.207792207792 hold_0_11 - 1276.190476190476 hold_0_12
   - 1178.021978021978 hold_0_13 - 1093.877551020408 hold_0_14
   - 1020.952380952381 hold_0_15 - 957.1428571428572 hold_0_16
   - 900.8403361344537 hold_0_17 - 850.7936507936507 hold_0_18
   - 806.015037593985 hold_0_19 - 765.7142857142857 hold_0_20
   - 729.2517006802721 hold_0_21 - 696.1038961038961 hold_0_22
   + flow_nic_in_0_0 <= 0
 node_throughput_constraint_1: group_tp_active_1 = 0 ->
   - 7108.571428571429 hold_1_1 - 3554.285714285715 hold_1_2
   - 2369.52380952381 hold_1_3 - 1777.142857142857 hold_1_4
   - 1421.714285714286 hold_1_5 - 1184.761904761905 hold_1_6
   - 1015.510204081633 hold_1_7 - 888.5714285714287 hold_1_8
   - 789.8412698412698 hold_1_9 - 710.8571428571429 hold_1_10
   - 646.2337662337662 hold_1_11 + flow_nic_in_1_1 <= 0
 node_throughput_constraint_2: group_tp_active_2 = 0 ->
   - 9325.714285714286 hold_2_1 - 4662.857142857143 hold_2_2
   - 3108.571428571428 hold_2_3 - 2331.428571428572 hold_2_4
   - 1865.142857142857 hold_2_5 - 1554.285714285714 hold_2_6
   - 1332.244897959184 hold_2_7 - 1165.714285714286 hold_2_8
   - 1036.190476190476 hold_2_9 - 932.5714285714284 hold_2_10
   - 847.7922077922078 hold_2_11 - 777.1428571428571 hold_2_12
   - 717.3626373626373 hold_2_13 - 666.1224489795918 hold_2_14
   - 621.7142857142858 hold_2_15 - 582.8571428571429 hold_2_16
   - 548.5714285714286 hold_2_17 - 518.0952380952381 hold_2_18
   - 490.8270676691729 hold_2_19 - 466.2857142857142 hold_2_20
   - 444.0816326530613 hold_2_21 - 423.8961038961039 hold_2_22
   + flow_nic_in_2_2 <= 0
 node_throughput_constraint_3: group_tp_active_3 = 0 ->
   - 7108.571428571429 hold_3_1 - 3554.285714285715 hold_3_2
   - 2369.52380952381 hold_3_3 - 1777.142857142857 hold_3_4
   - 1421.714285714286 hold_3_5 - 1184.761904761905 hold_3_6
   - 1015.510204081633 hold_3_7 - 888.5714285714287 hold_3_8
   - 789.8412698412698 hold_3_9 - 710.8571428571429 hold_3_10
   - 646.2337662337662 hold_3_11 + flow_nic_in_3_3 <= 0
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
   - 3949.714285714286 hold_6_1 - 1974.857142857143 hold_6_2
   - 1316.571428571428 hold_6_3 - 987.4285714285714 hold_6_4
   - 789.9428571428572 hold_6_5 - 658.2857142857142 hold_6_6
   - 564.2448979591836 hold_6_7 - 493.7142857142857 hold_6_8
   - 438.8571428571428 hold_6_9 - 394.9714285714286 hold_6_10
   - 359.0649350649351 hold_6_11 - 329.1428571428571 hold_6_12
   - 303.8241758241758 hold_6_13 + flow_nic_in_6_6 <= 0
 node_throughput_constraint_7: group_tp_active_7 = 0 ->
   - 3949.714285714286 hold_7_1 - 1974.857142857143 hold_7_2
   - 1316.571428571428 hold_7_3 - 987.4285714285714 hold_7_4
   - 789.9428571428572 hold_7_5 - 658.2857142857142 hold_7_6
   - 564.2448979591836 hold_7_7 - 493.7142857142857 hold_7_8
   - 438.8571428571428 hold_7_9 - 394.9714285714286 hold_7_10
   - 359.0649350649351 hold_7_11 - 329.1428571428571 hold_7_12
   - 303.8241758241758 hold_7_13 + flow_nic_in_7_7 <= 0
 node_throughput_constraint_8: group_tp_active_8 = 0 ->
   - 3949.714285714286 hold_8_1 - 1974.857142857143 hold_8_2
   - 1316.571428571428 hold_8_3 - 987.4285714285714 hold_8_4
   - 789.9428571428572 hold_8_5 - 658.2857142857142 hold_8_6
   - 564.2448979591836 hold_8_7 - 493.7142857142857 hold_8_8
   - 438.8571428571428 hold_8_9 - 394.9714285714286 hold_8_10
   - 359.0649350649351 hold_8_11 - 329.1428571428571 hold_8_12
   - 303.8241758241758 hold_8_13 + flow_nic_in_8_8 + flow_9_8 <= 0
 node_throughput_constraint_9: group_tp_active_8 = 0 ->
   - 3949.714285714286 hold_9_1 - 1974.857142857143 hold_9_2
   - 1316.571428571428 hold_9_3 - 987.4285714285714 hold_9_4
   - 789.9428571428572 hold_9_5 - 658.2857142857142 hold_9_6
   - 564.2448979591836 hold_9_7 - 493.7142857142857 hold_9_8
   - 438.8571428571428 hold_9_9 - 394.9714285714286 hold_9_10
   - 359.0649350649351 hold_9_11 - 329.1428571428571 hold_9_12
   - 303.8241758241758 hold_9_13 + flow_nic_in_8_9 + flow_8_9 <= 0
 node_throughput_constraint_10: group_tp_active_10 = 0 ->
   - 4265.142857142857 hold_10_1 - 2132.571428571428 hold_10_2
   - 1421.714285714286 hold_10_3 - 1066.285714285714 hold_10_4
   - 853.0285714285714 hold_10_5 - 710.8571428571429 hold_10_6
   + flow_nic_in_10_10 + flow_11_10 <= 0
 node_throughput_constraint_11: group_tp_active_10 = 0 ->
   - 4265.142857142857 hold_11_1 - 2132.571428571428 hold_11_2
   - 1421.714285714286 hold_11_3 - 1066.285714285714 hold_11_4
   - 853.0285714285714 hold_11_5 - 710.8571428571429 hold_11_6
   + flow_nic_in_10_11 + flow_10_11 <= 0
 node_throughput_constraint_12: group_tp_active_12 = 0 ->
   - 1371.428571428571 hold_12_1 - 685.7142857142857 hold_12_2
   - 457.1428571428571 hold_12_3 - 342.8571428571428 hold_12_4
   - 274.2857142857143 hold_12_5 - 228.5714285714286 hold_12_6
   + flow_nic_in_12_12 + flow_13_12 <= 0
 node_throughput_constraint_13: group_tp_active_12 = 0 ->
   - 1371.428571428571 hold_13_1 - 685.7142857142857 hold_13_2
   - 457.1428571428571 hold_13_3 - 342.8571428571428 hold_13_4
   - 274.2857142857143 hold_13_5 - 228.5714285714286 hold_13_6
   + flow_nic_in_12_13 + flow_12_13 <= 0
 node_throughput_constraint_14: group_tp_active_14 = 0 -> - 2048 hold_14_1
   - 1024 hold_14_2 - 682.6666666666666 hold_14_3 - 512 hold_14_4
   + flow_nic_in_14_14 + flow_15_14 + flow_16_14 + flow_17_14 <= 0
 node_throughput_constraint_15: group_tp_active_14 = 0 -> - 2048 hold_15_1
   - 1024 hold_15_2 - 682.6666666666666 hold_15_3 - 512 hold_15_4
   + flow_nic_in_14_15 + flow_14_15 + flow_16_15 + flow_17_15 <= 0
 node_throughput_constraint_16: group_tp_active_14 = 0 -> - 2048 hold_16_1
   - 1024 hold_16_2 - 682.6666666666666 hold_16_3 - 512 hold_16_4
   + flow_nic_in_14_16 + flow_14_16 + flow_15_16 + flow_17_16 <= 0
 node_throughput_constraint_17: group_tp_active_14 = 0 -> - 2048 hold_17_1
   - 1024 hold_17_2 - 682.6666666666666 hold_17_3 - 512 hold_17_4
   + flow_nic_in_14_17 + flow_14_17 + flow_15_17 + flow_16_17 <= 0
 constr_intergroup_throughput_0_source_nic_in_0: switch_source_nic_in_0 = 1 ->
   - 12853.7170263789 hold_0_1 - 6988.265971316819 hold_0_2
   - 4798.567591763653 hold_0_3 - 3653.71506475801 hold_0_4
   - 2949.917446340121 hold_0_5 - 2473.465620673743 hold_0_6
   - 2129.519268970997 hold_0_7 - 1869.550052319498 hold_0_8
   - 1666.148585638794 hold_0_9 - 1502.663302495094 hold_0_10
   - 1368.39417921879 hold_0_11 - 1256.151863135693 hold_0_12
   - 1160.927008880225 hold_0_13 - 1079.122206563318 hold_0_14
   - 1008.087267255972 hold_0_15 - 945.8267160755252 hold_0_16
   - 890.8093734419145 hold_0_17 - 841.8407413224438 hold_0_18
   - 797.9752865862737 hold_0_19 - 758.4547898684025 hold_0_20
   - 722.664149925846 hold_0_21 - 690.0991373760783 hold_0_22
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_0_nic_in_1: switch_nic_out_0_nic_in_1 = 1 ->
   - 4921.859545004946 hold_1_1 - 2908.240794856809 hold_1_2
   - 2063.873911240149 hold_1_3 - 1599.485695917711 hold_1_4
   - 1305.694043558121 hold_1_5 - 1103.081356683662 hold_1_6
   - 954.9030896181155 hold_1_7 - 841.8203349687025 hold_1_8
   - 752.6849190742701 hold_1_9 - 680.6182464779099 hold_1_10
   - 621.1459243540132 hold_1_11 + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_0_nic_in_2: switch_nic_out_0_nic_in_2 = 1 ->
   - 5891.696750902527 hold_2_1 - 3610.619469026549 hold_2_2
   - 2602.870813397129 hold_2_3 - 2034.912718204489 hold_2_4
   - 1670.419651995905 hold_2_5 - 1416.666666666667 hold_2_6
   - 1229.841748304446 hold_2_7 - 1086.551264980027 hold_2_8
   - 973.1663685152056 hold_2_9 - 881.2095032397407 hold_2_10
   - 805.1307350764677 hold_2_11 - 741.1444141689373 hold_2_12
   - 686.5797223390829 hold_2_13 - 639.4984326018808 hold_2_14
   - 598.4598459845985 hold_2_15 - 562.3707787732598 hold_2_16
   - 530.3867403314916 hold_2_17 - 501.8450184501845 hold_2_18
   - 476.2182667055733 hold_2_19 - 453.081621321488 hold_2_20
   - 432.0889594916601 hold_2_21 - 412.9554655870446 hold_2_22
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_0_nic_in_3: switch_nic_out_0_nic_in_3 = 1 ->
   - 4921.859545004946 hold_3_1 - 2908.240794856809 hold_3_2
   - 2063.873911240149 hold_3_3 - 1599.485695917711 hold_3_4
   - 1305.694043558121 hold_3_5 - 1103.081356683662 hold_3_6
   - 954.9030896181155 hold_3_7 - 841.8203349687025 hold_3_8
   - 752.6849190742701 hold_3_9 - 680.6182464779099 hold_3_10
   - 621.1459243540132 hold_3_11 + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_0_nic_in_4: switch_nic_out_0_nic_in_4 = 1 ->
   - 3167.736021998167 hold_4_1 - 1757.884028484232 hold_4_2
   - 1216.473072861668 hold_4_3 - 930.032292787944 hold_4_4
   - 752.7771727292528 hold_4_5 - 632.2722283205268 hold_4_6
   - 545.0244440939914 hold_4_7 - 478.9356984478936 hold_4_8
   - 427.1412680756396 hold_4_9 - 385.4561677448137 hold_4_10
   - 351.1838227822376 hold_4_11 - 322.5083986562149 hold_4_12
   - 298.1623673539816 hold_4_13 + flow_nic_in_4_4 + flow_5_4 <= 0
 constr_intergroup_throughput_6_nic_out_0_nic_in_6: switch_nic_out_0_nic_in_6 = 1 ->
   - 3167.736021998167 hold_6_1 - 1757.884028484232 hold_6_2
   - 1216.473072861668 hold_6_3 - 930.032292787944 hold_6_4
   - 752.7771727292528 hold_6_5 - 632.2722283205268 hold_6_6
   - 545.0244440939914 hold_6_7 - 478.9356984478936 hold_6_8
   - 427.1412680756396 hold_6_9 - 385.4561677448137 hold_6_10
   - 351.1838227822376 hold_6_11 - 322.5083986562149 hold_6_12
   - 298.1623673539816 hold_6_13 + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_0_nic_in_7: switch_nic_out_0_nic_in_7 = 1 ->
   - 3167.736021998167 hold_7_1 - 1757.884028484232 hold_7_2
   - 1216.473072861668 hold_7_3 - 930.032292787944 hold_7_4
   - 752.7771727292528 hold_7_5 - 632.2722283205268 hold_7_6
   - 545.0244440939914 hold_7_7 - 478.9356984478936 hold_7_8
   - 427.1412680756396 hold_7_9 - 385.4561677448137 hold_7_10
   - 351.1838227822376 hold_7_11 - 322.5083986562149 hold_7_12
   - 298.1623673539816 hold_7_13 + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_0_nic_in_8: switch_nic_out_0_nic_in_8 = 1 ->
   - 3167.736021998167 hold_8_1 - 1757.884028484232 hold_8_2
   - 1216.473072861668 hold_8_3 - 930.032292787944 hold_8_4
   - 752.7771727292528 hold_8_5 - 632.2722283205268 hold_8_6
   - 545.0244440939914 hold_8_7 - 478.9356984478936 hold_8_8
   - 427.1412680756396 hold_8_9 - 385.4561677448137 hold_8_10
   - 351.1838227822376 hold_8_11 - 322.5083986562149 hold_8_12
   - 298.1623673539816 hold_8_13 + flow_nic_in_8_8 + flow_9_8 <= 0
 constr_intergroup_throughput_10_nic_out_0_nic_in_10: switch_nic_out_0_nic_in_10 = 1 ->
   - 3367.47123843898 hold_10_1 - 1881.759737804109 hold_10_2
   - 1305.694043558121 hold_10_3 - 999.665171097569 hold_10_4
   - 809.8518960559865 hold_10_5 - 680.6182464779099 hold_10_6
   + flow_nic_in_10_10 + flow_11_10 <= 0
 constr_intergroup_throughput_12_nic_out_0_nic_in_12: switch_nic_out_0_nic_in_12 = 1 ->
   - 1263.157894736842 hold_12_1 - 657.5342465753424 hold_12_2
   - 444.4444444444444 hold_12_3 - 335.6643356643356 hold_12_4
   - 269.6629213483146 hold_12_5 - 225.3521126760563 hold_12_6
   + flow_nic_in_12_12 + flow_13_12 <= 0
 constr_intergroup_throughput_14_nic_out_0_nic_in_14: switch_nic_out_0_nic_in_14 = 1 ->
   - 1815.602836879433 hold_14_1 - 962.406015037594 hold_14_2
   - 654.7314578005115 hold_14_3 - 496.1240310077519 hold_14_4
   + flow_nic_in_14_14 + flow_15_14 + flow_16_14 + flow_17_14 <= 0
 constr_intergroup_throughput_1_source_nic_in_1: switch_source_nic_in_1 = 1 ->
   - 6528.470217790607 hold_1_1 - 3403.09123238955 hold_1_2
   - 2301.359726204792 hold_1_3 - 1738.522814618126 hold_1_4
   - 1396.889562629835 hold_1_5 - 1167.472197456713 hold_1_6
   - 1002.781024545564 hold_1_7 - 878.810356398573 hold_1_8
   - 782.1193926629153 hold_1_9 - 704.5963014358132 hold_1_10
   - 641.0553709000026 hold_1_11 + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_0_nic_out_1_nic_in_0: switch_nic_out_1_nic_in_0 = 1 ->
   - 7824.817518248176 hold_0_1 - 5178.743961352658 hold_0_2
   - 3870.036101083032 hold_0_3 - 3089.337175792507 hold_0_4
   - 2570.743405275779 hold_0_5 - 2201.23203285421 hold_0_6
   - 1924.5960502693 hold_0_7 - 1709.728867623605 hold_0_8
   - 1538.020086083214 hold_0_9 - 1397.653194263364 hold_0_10
   - 1280.764635603345 hold_0_11 - 1181.918412348401 hold_0_12
   - 1097.236438075742 hold_0_13 - 1023.877745940783 hold_0_14
   - 959.7135183527306 hold_0_15 - 903.117101937658 hold_0_16
   - 852.8241845664279 hold_0_17 - 807.8372268274302 hold_0_18
   - 767.3586256263421 hold_0_19 - 730.7430129516018 hold_0_20
   - 697.4625894599869 hold_0_21 - 667.0815183571873 hold_0_22
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_2_nic_out_1_nic_in_2: switch_nic_out_1_nic_in_2 = 1 ->
   - 5891.696750902527 hold_2_1 - 3610.619469026549 hold_2_2
   - 2602.870813397129 hold_2_3 - 2034.912718204489 hold_2_4
   - 1670.419651995905 hold_2_5 - 1416.666666666667 hold_2_6
   - 1229.841748304446 hold_2_7 - 1086.551264980027 hold_2_8
   - 973.1663685152056 hold_2_9 - 881.2095032397407 hold_2_10
   - 805.1307350764677 hold_2_11 - 741.1444141689373 hold_2_12
   - 686.5797223390829 hold_2_13 - 639.4984326018808 hold_2_14
   - 598.4598459845985 hold_2_15 - 562.3707787732598 hold_2_16
   - 530.3867403314916 hold_2_17 - 501.8450184501845 hold_2_18
   - 476.2182667055733 hold_2_19 - 453.081621321488 hold_2_20
   - 432.0889594916601 hold_2_21 - 412.9554655870446 hold_2_22
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_1_nic_in_3: switch_nic_out_1_nic_in_3 = 1 ->
   - 4921.859545004946 hold_3_1 - 2908.240794856809 hold_3_2
   - 2063.873911240149 hold_3_3 - 1599.485695917711 hold_3_4
   - 1305.694043558121 hold_3_5 - 1103.081356683662 hold_3_6
   - 954.9030896181155 hold_3_7 - 841.8203349687025 hold_3_8
   - 752.6849190742701 hold_3_9 - 680.6182464779099 hold_3_10
   - 621.1459243540132 hold_3_11 + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_1_nic_in_4: switch_nic_out_1_nic_in_4 = 1 ->
   - 3167.736021998167 hold_4_1 - 1757.884028484232 hold_4_2
   - 1216.473072861668 hold_4_3 - 930.032292787944 hold_4_4
   - 752.7771727292528 hold_4_5 - 632.2722283205268 hold_4_6
   - 545.0244440939914 hold_4_7 - 478.9356984478936 hold_4_8
   - 427.1412680756396 hold_4_9 - 385.4561677448137 hold_4_10
   - 351.1838227822376 hold_4_11 - 322.5083986562149 hold_4_12
   - 298.1623673539816 hold_4_13 + flow_nic_in_4_4 + flow_5_4 <= 0
 constr_intergroup_throughput_6_nic_out_1_nic_in_6: switch_nic_out_1_nic_in_6 = 1 ->
   - 3167.736021998167 hold_6_1 - 1757.884028484232 hold_6_2
   - 1216.473072861668 hold_6_3 - 930.032292787944 hold_6_4
   - 752.7771727292528 hold_6_5 - 632.2722283205268 hold_6_6
   - 545.0244440939914 hold_6_7 - 478.9356984478936 hold_6_8
   - 427.1412680756396 hold_6_9 - 385.4561677448137 hold_6_10
   - 351.1838227822376 hold_6_11 - 322.5083986562149 hold_6_12
   - 298.1623673539816 hold_6_13 + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_1_nic_in_7: switch_nic_out_1_nic_in_7 = 1 ->
   - 3167.736021998167 hold_7_1 - 1757.884028484232 hold_7_2
   - 1216.473072861668 hold_7_3 - 930.032292787944 hold_7_4
   - 752.7771727292528 hold_7_5 - 632.2722283205268 hold_7_6
   - 545.0244440939914 hold_7_7 - 478.9356984478936 hold_7_8
   - 427.1412680756396 hold_7_9 - 385.4561677448137 hold_7_10
   - 351.1838227822376 hold_7_11 - 322.5083986562149 hold_7_12
   - 298.1623673539816 hold_7_13 + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_1_nic_in_8: switch_nic_out_1_nic_in_8 = 1 ->
   - 3167.736021998167 hold_8_1 - 1757.884028484232 hold_8_2
   - 1216.473072861668 hold_8_3 - 930.032292787944 hold_8_4
   - 752.7771727292528 hold_8_5 - 632.2722283205268 hold_8_6
   - 545.0244440939914 hold_8_7 - 478.9356984478936 hold_8_8
   - 427.1412680756396 hold_8_9 - 385.4561677448137 hold_8_10
   - 351.1838227822376 hold_8_11 - 322.5083986562149 hold_8_12
   - 298.1623673539816 hold_8_13 + flow_nic_in_8_8 + flow_9_8 <= 0
 constr_intergroup_throughput_10_nic_out_1_nic_in_10: switch_nic_out_1_nic_in_10 = 1 ->
   - 3367.47123843898 hold_10_1 - 1881.759737804109 hold_10_2
   - 1305.694043558121 hold_10_3 - 999.665171097569 hold_10_4
   - 809.8518960559865 hold_10_5 - 680.6182464779099 hold_10_6
   + flow_nic_in_10_10 + flow_11_10 <= 0
 constr_intergroup_throughput_12_nic_out_1_nic_in_12: switch_nic_out_1_nic_in_12 = 1 ->
   - 1263.157894736842 hold_12_1 - 657.5342465753424 hold_12_2
   - 444.4444444444444 hold_12_3 - 335.6643356643356 hold_12_4
   - 269.6629213483146 hold_12_5 - 225.3521126760563 hold_12_6
   + flow_nic_in_12_12 + flow_13_12 <= 0
 constr_intergroup_throughput_14_nic_out_1_nic_in_14: switch_nic_out_1_nic_in_14 = 1 ->
   - 1815.602836879433 hold_14_1 - 962.406015037594 hold_14_2
   - 654.7314578005115 hold_14_3 - 496.1240310077519 hold_14_4
   + flow_nic_in_14_14 + flow_15_14 + flow_16_14 + flow_17_14 <= 0
 constr_intergroup_throughput_2_source_nic_in_2: switch_source_nic_in_2 = 1 ->
   - 8352.09825997953 hold_2_1 - 4406.047516198704 hold_2_2
   - 2992.299229922993 hold_2_3 - 2265.40810660744 hold_2_4
   - 1822.649095376368 hold_2_5 - 1524.663677130045 hold_2_6
   - 1310.42235426369 hold_2_7 - 1148.972120529428 hold_2_8
   - 1022.940955246333 hold_2_9 - 921.825576140985 hold_2_10
   - 838.9020252904288 hold_2_11 - 769.6661007357103 hold_2_12
   - 710.9871917748541 hold_2_13 - 660.6217616580311 hold_2_14
   - 616.9199364935361 hold_2_15 - 578.6413274712806 hold_2_16
   - 544.8354143019296 hold_2_17 - 514.7615442846328 hold_2_18
   - 487.8340407724039 hold_2_19 - 463.5836836723099 hold_2_20
   - 441.6301347621368 hold_2_21 - 421.6618437370815 hold_2_22
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_0_nic_out_2_nic_in_0: switch_nic_out_2_nic_in_0 = 1 ->
   - 7824.817518248176 hold_0_1 - 5178.743961352658 hold_0_2
   - 3870.036101083032 hold_0_3 - 3089.337175792507 hold_0_4
   - 2570.743405275779 hold_0_5 - 2201.23203285421 hold_0_6
   - 1924.5960502693 hold_0_7 - 1709.728867623605 hold_0_8
   - 1538.020086083214 hold_0_9 - 1397.653194263364 hold_0_10
   - 1280.764635603345 hold_0_11 - 1181.918412348401 hold_0_12
   - 1097.236438075742 hold_0_13 - 1023.877745940783 hold_0_14
   - 959.7135183527306 hold_0_15 - 903.117101937658 hold_0_16
   - 852.8241845664279 hold_0_17 - 807.8372268274302 hold_0_18
   - 767.3586256263421 hold_0_19 - 730.7430129516018 hold_0_20
   - 697.4625894599869 hold_0_21 - 667.0815183571873 hold_0_22
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_2_nic_in_1: switch_nic_out_2_nic_in_1 = 1 ->
   - 4921.859545004946 hold_1_1 - 2908.240794856809 hold_1_2
   - 2063.873911240149 hold_1_3 - 1599.485695917711 hold_1_4
   - 1305.694043558121 hold_1_5 - 1103.081356683662 hold_1_6
   - 954.9030896181155 hold_1_7 - 841.8203349687025 hold_1_8
   - 752.6849190742701 hold_1_9 - 680.6182464779099 hold_1_10
   - 621.1459243540132 hold_1_11 + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_3_nic_out_2_nic_in_3: switch_nic_out_2_nic_in_3 = 1 ->
   - 4921.859545004946 hold_3_1 - 2908.240794856809 hold_3_2
   - 2063.873911240149 hold_3_3 - 1599.485695917711 hold_3_4
   - 1305.694043558121 hold_3_5 - 1103.081356683662 hold_3_6
   - 954.9030896181155 hold_3_7 - 841.8203349687025 hold_3_8
   - 752.6849190742701 hold_3_9 - 680.6182464779099 hold_3_10
   - 621.1459243540132 hold_3_11 + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_2_nic_in_4: switch_nic_out_2_nic_in_4 = 1 ->
   - 3167.736021998167 hold_4_1 - 1757.884028484232 hold_4_2
   - 1216.473072861668 hold_4_3 - 930.032292787944 hold_4_4
   - 752.7771727292528 hold_4_5 - 632.2722283205268 hold_4_6
   - 545.0244440939914 hold_4_7 - 478.9356984478936 hold_4_8
   - 427.1412680756396 hold_4_9 - 385.4561677448137 hold_4_10
   - 351.1838227822376 hold_4_11 - 322.5083986562149 hold_4_12
   - 298.1623673539816 hold_4_13 + flow_nic_in_4_4 + flow_5_4 <= 0
 constr_intergroup_throughput_6_nic_out_2_nic_in_6: switch_nic_out_2_nic_in_6 = 1 ->
   - 3167.736021998167 hold_6_1 - 1757.884028484232 hold_6_2
   - 1216.473072861668 hold_6_3 - 930.032292787944 hold_6_4
   - 752.7771727292528 hold_6_5 - 632.2722283205268 hold_6_6
   - 545.0244440939914 hold_6_7 - 478.9356984478936 hold_6_8
   - 427.1412680756396 hold_6_9 - 385.4561677448137 hold_6_10
   - 351.1838227822376 hold_6_11 - 322.5083986562149 hold_6_12
   - 298.1623673539816 hold_6_13 + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_2_nic_in_7: switch_nic_out_2_nic_in_7 = 1 ->
   - 3167.736021998167 hold_7_1 - 1757.884028484232 hold_7_2
   - 1216.473072861668 hold_7_3 - 930.032292787944 hold_7_4
   - 752.7771727292528 hold_7_5 - 632.2722283205268 hold_7_6
   - 545.0244440939914 hold_7_7 - 478.9356984478936 hold_7_8
   - 427.1412680756396 hold_7_9 - 385.4561677448137 hold_7_10
   - 351.1838227822376 hold_7_11 - 322.5083986562149 hold_7_12
   - 298.1623673539816 hold_7_13 + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_2_nic_in_8: switch_nic_out_2_nic_in_8 = 1 ->
   - 3167.736021998167 hold_8_1 - 1757.884028484232 hold_8_2
   - 1216.473072861668 hold_8_3 - 930.032292787944 hold_8_4
   - 752.7771727292528 hold_8_5 - 632.2722283205268 hold_8_6
   - 545.0244440939914 hold_8_7 - 478.9356984478936 hold_8_8
   - 427.1412680756396 hold_8_9 - 385.4561677448137 hold_8_10
   - 351.1838227822376 hold_8_11 - 322.5083986562149 hold_8_12
   - 298.1623673539816 hold_8_13 + flow_nic_in_8_8 + flow_9_8 <= 0
 constr_intergroup_throughput_10_nic_out_2_nic_in_10: switch_nic_out_2_nic_in_10 = 1 ->
   - 3367.47123843898 hold_10_1 - 1881.759737804109 hold_10_2
   - 1305.694043558121 hold_10_3 - 999.665171097569 hold_10_4
   - 809.8518960559865 hold_10_5 - 680.6182464779099 hold_10_6
   + flow_nic_in_10_10 + flow_11_10 <= 0
 constr_intergroup_throughput_12_nic_out_2_nic_in_12: switch_nic_out_2_nic_in_12 = 1 ->
   - 1263.157894736842 hold_12_1 - 657.5342465753424 hold_12_2
   - 444.4444444444444 hold_12_3 - 335.6643356643356 hold_12_4
   - 269.6629213483146 hold_12_5 - 225.3521126760563 hold_12_6
   + flow_nic_in_12_12 + flow_13_12 <= 0
 constr_intergroup_throughput_14_nic_out_2_nic_in_14: switch_nic_out_2_nic_in_14 = 1 ->
   - 1815.602836879433 hold_14_1 - 962.406015037594 hold_14_2
   - 654.7314578005115 hold_14_3 - 496.1240310077519 hold_14_4
   + flow_nic_in_14_14 + flow_15_14 + flow_16_14 + flow_17_14 <= 0
 constr_intergroup_throughput_3_source_nic_in_3: switch_source_nic_in_3 = 1 ->
   - 6528.470217790607 hold_3_1 - 3403.09123238955 hold_3_2
   - 2301.359726204792 hold_3_3 - 1738.522814618126 hold_3_4
   - 1396.889562629835 hold_3_5 - 1167.472197456713 hold_3_6
   - 1002.781024545564 hold_3_7 - 878.810356398573 hold_3_8
   - 782.1193926629153 hold_3_9 - 704.5963014358132 hold_3_10
   - 641.0553709000026 hold_3_11 + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_0_nic_out_3_nic_in_0: switch_nic_out_3_nic_in_0 = 1 ->
   - 7824.817518248176 hold_0_1 - 5178.743961352658 hold_0_2
   - 3870.036101083032 hold_0_3 - 3089.337175792507 hold_0_4
   - 2570.743405275779 hold_0_5 - 2201.23203285421 hold_0_6
   - 1924.5960502693 hold_0_7 - 1709.728867623605 hold_0_8
   - 1538.020086083214 hold_0_9 - 1397.653194263364 hold_0_10
   - 1280.764635603345 hold_0_11 - 1181.918412348401 hold_0_12
   - 1097.236438075742 hold_0_13 - 1023.877745940783 hold_0_14
   - 959.7135183527306 hold_0_15 - 903.117101937658 hold_0_16
   - 852.8241845664279 hold_0_17 - 807.8372268274302 hold_0_18
   - 767.3586256263421 hold_0_19 - 730.7430129516018 hold_0_20
   - 697.4625894599869 hold_0_21 - 667.0815183571873 hold_0_22
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_3_nic_in_1: switch_nic_out_3_nic_in_1 = 1 ->
   - 4921.859545004946 hold_1_1 - 2908.240794856809 hold_1_2
   - 2063.873911240149 hold_1_3 - 1599.485695917711 hold_1_4
   - 1305.694043558121 hold_1_5 - 1103.081356683662 hold_1_6
   - 954.9030896181155 hold_1_7 - 841.8203349687025 hold_1_8
   - 752.6849190742701 hold_1_9 - 680.6182464779099 hold_1_10
   - 621.1459243540132 hold_1_11 + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_3_nic_in_2: switch_nic_out_3_nic_in_2 = 1 ->
   - 5891.696750902527 hold_2_1 - 3610.619469026549 hold_2_2
   - 2602.870813397129 hold_2_3 - 2034.912718204489 hold_2_4
   - 1670.419651995905 hold_2_5 - 1416.666666666667 hold_2_6
   - 1229.841748304446 hold_2_7 - 1086.551264980027 hold_2_8
   - 973.1663685152056 hold_2_9 - 881.2095032397407 hold_2_10
   - 805.1307350764677 hold_2_11 - 741.1444141689373 hold_2_12
   - 686.5797223390829 hold_2_13 - 639.4984326018808 hold_2_14
   - 598.4598459845985 hold_2_15 - 562.3707787732598 hold_2_16
   - 530.3867403314916 hold_2_17 - 501.8450184501845 hold_2_18
   - 476.2182667055733 hold_2_19 - 453.081621321488 hold_2_20
   - 432.0889594916601 hold_2_21 - 412.9554655870446 hold_2_22
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_4_nic_out_3_nic_in_4: switch_nic_out_3_nic_in_4 = 1 ->
   - 3167.736021998167 hold_4_1 - 1757.884028484232 hold_4_2
   - 1216.473072861668 hold_4_3 - 930.032292787944 hold_4_4
   - 752.7771727292528 hold_4_5 - 632.2722283205268 hold_4_6
   - 545.0244440939914 hold_4_7 - 478.9356984478936 hold_4_8
   - 427.1412680756396 hold_4_9 - 385.4561677448137 hold_4_10
   - 351.1838227822376 hold_4_11 - 322.5083986562149 hold_4_12
   - 298.1623673539816 hold_4_13 + flow_nic_in_4_4 + flow_5_4 <= 0
 constr_intergroup_throughput_6_nic_out_3_nic_in_6: switch_nic_out_3_nic_in_6 = 1 ->
   - 3167.736021998167 hold_6_1 - 1757.884028484232 hold_6_2
   - 1216.473072861668 hold_6_3 - 930.032292787944 hold_6_4
   - 752.7771727292528 hold_6_5 - 632.2722283205268 hold_6_6
   - 545.0244440939914 hold_6_7 - 478.9356984478936 hold_6_8
   - 427.1412680756396 hold_6_9 - 385.4561677448137 hold_6_10
   - 351.1838227822376 hold_6_11 - 322.5083986562149 hold_6_12
   - 298.1623673539816 hold_6_13 + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_3_nic_in_7: switch_nic_out_3_nic_in_7 = 1 ->
   - 3167.736021998167 hold_7_1 - 1757.884028484232 hold_7_2
   - 1216.473072861668 hold_7_3 - 930.032292787944 hold_7_4
   - 752.7771727292528 hold_7_5 - 632.2722283205268 hold_7_6
   - 545.0244440939914 hold_7_7 - 478.9356984478936 hold_7_8
   - 427.1412680756396 hold_7_9 - 385.4561677448137 hold_7_10
   - 351.1838227822376 hold_7_11 - 322.5083986562149 hold_7_12
   - 298.1623673539816 hold_7_13 + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_3_nic_in_8: switch_nic_out_3_nic_in_8 = 1 ->
   - 3167.736021998167 hold_8_1 - 1757.884028484232 hold_8_2
   - 1216.473072861668 hold_8_3 - 930.032292787944 hold_8_4
   - 752.7771727292528 hold_8_5 - 632.2722283205268 hold_8_6
   - 545.0244440939914 hold_8_7 - 478.9356984478936 hold_8_8
   - 427.1412680756396 hold_8_9 - 385.4561677448137 hold_8_10
   - 351.1838227822376 hold_8_11 - 322.5083986562149 hold_8_12
   - 298.1623673539816 hold_8_13 + flow_nic_in_8_8 + flow_9_8 <= 0
 constr_intergroup_throughput_10_nic_out_3_nic_in_10: switch_nic_out_3_nic_in_10 = 1 ->
   - 3367.47123843898 hold_10_1 - 1881.759737804109 hold_10_2
   - 1305.694043558121 hold_10_3 - 999.665171097569 hold_10_4
   - 809.8518960559865 hold_10_5 - 680.6182464779099 hold_10_6
   + flow_nic_in_10_10 + flow_11_10 <= 0
 constr_intergroup_throughput_12_nic_out_3_nic_in_12: switch_nic_out_3_nic_in_12 = 1 ->
   - 1263.157894736842 hold_12_1 - 657.5342465753424 hold_12_2
   - 444.4444444444444 hold_12_3 - 335.6643356643356 hold_12_4
   - 269.6629213483146 hold_12_5 - 225.3521126760563 hold_12_6
   + flow_nic_in_12_12 + flow_13_12 <= 0
 constr_intergroup_throughput_14_nic_out_3_nic_in_14: switch_nic_out_3_nic_in_14 = 1 ->
   - 1815.602836879433 hold_14_1 - 962.406015037594 hold_14_2
   - 654.7314578005115 hold_14_3 - 496.1240310077519 hold_14_4
   + flow_nic_in_14_14 + flow_15_14 + flow_16_14 + flow_17_14 <= 0
 constr_intergroup_throughput_4_source_nic_in_4: switch_source_nic_in_4 = 1 ->
   - 3763.885863646265 hold_4_1 - 1927.280838724069 hold_4_2
   - 1295.25522824376 hold_4_3 - 975.3894784375707 hold_4_4
   - 782.2190032139786 hold_4_5 - 652.9131716164135 hold_4_6
   - 560.2931163062157 hold_4_7 - 490.6860517946388 hold_4_8
   - 436.4628324619231 hold_4_9 - 393.0309784833735 hold_4_10
   - 357.4605407418134 hold_4_11 - 327.7942180742089 hold_4_12
   - 302.6746772696222 hold_4_13 + flow_nic_in_4_4 + flow_5_4 <= 0
 constr_intergroup_throughput_0_nic_out_4_nic_in_0: switch_nic_out_4_nic_in_0 = 1 ->
   - 7824.817518248176 hold_0_1 - 5178.743961352658 hold_0_2
   - 3870.036101083032 hold_0_3 - 3089.337175792507 hold_0_4
   - 2570.743405275779 hold_0_5 - 2201.23203285421 hold_0_6
   - 1924.5960502693 hold_0_7 - 1709.728867623605 hold_0_8
   - 1538.020086083214 hold_0_9 - 1397.653194263364 hold_0_10
   - 1280.764635603345 hold_0_11 - 1181.918412348401 hold_0_12
   - 1097.236438075742 hold_0_13 - 1023.877745940783 hold_0_14
   - 959.7135183527306 hold_0_15 - 903.117101937658 hold_0_16
   - 852.8241845664279 hold_0_17 - 807.8372268274302 hold_0_18
   - 767.3586256263421 hold_0_19 - 730.7430129516018 hold_0_20
   - 697.4625894599869 hold_0_21 - 667.0815183571873 hold_0_22
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_4_nic_in_1: switch_nic_out_4_nic_in_1 = 1 ->
   - 4921.859545004946 hold_1_1 - 2908.240794856809 hold_1_2
   - 2063.873911240149 hold_1_3 - 1599.485695917711 hold_1_4
   - 1305.694043558121 hold_1_5 - 1103.081356683662 hold_1_6
   - 954.9030896181155 hold_1_7 - 841.8203349687025 hold_1_8
   - 752.6849190742701 hold_1_9 - 680.6182464779099 hold_1_10
   - 621.1459243540132 hold_1_11 + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_4_nic_in_2: switch_nic_out_4_nic_in_2 = 1 ->
   - 5891.696750902527 hold_2_1 - 3610.619469026549 hold_2_2
   - 2602.870813397129 hold_2_3 - 2034.912718204489 hold_2_4
   - 1670.419651995905 hold_2_5 - 1416.666666666667 hold_2_6
   - 1229.841748304446 hold_2_7 - 1086.551264980027 hold_2_8
   - 973.1663685152056 hold_2_9 - 881.2095032397407 hold_2_10
   - 805.1307350764677 hold_2_11 - 741.1444141689373 hold_2_12
   - 686.5797223390829 hold_2_13 - 639.4984326018808 hold_2_14
   - 598.4598459845985 hold_2_15 - 562.3707787732598 hold_2_16
   - 530.3867403314916 hold_2_17 - 501.8450184501845 hold_2_18
   - 476.2182667055733 hold_2_19 - 453.081621321488 hold_2_20
   - 432.0889594916601 hold_2_21 - 412.9554655870446 hold_2_22
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_4_nic_in_3: switch_nic_out_4_nic_in_3 = 1 ->
   - 4921.859545004946 hold_3_1 - 2908.240794856809 hold_3_2
   - 2063.873911240149 hold_3_3 - 1599.485695917711 hold_3_4
   - 1305.694043558121 hold_3_5 - 1103.081356683662 hold_3_6
   - 954.9030896181155 hold_3_7 - 841.8203349687025 hold_3_8
   - 752.6849190742701 hold_3_9 - 680.6182464779099 hold_3_10
   - 621.1459243540132 hold_3_11 + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_6_nic_out_4_nic_in_6: switch_nic_out_4_nic_in_6 = 1 ->
   - 3167.736021998167 hold_6_1 - 1757.884028484232 hold_6_2
   - 1216.473072861668 hold_6_3 - 930.032292787944 hold_6_4
   - 752.7771727292528 hold_6_5 - 632.2722283205268 hold_6_6
   - 545.0244440939914 hold_6_7 - 478.9356984478936 hold_6_8
   - 427.1412680756396 hold_6_9 - 385.4561677448137 hold_6_10
   - 351.1838227822376 hold_6_11 - 322.5083986562149 hold_6_12
   - 298.1623673539816 hold_6_13 + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_4_nic_in_7: switch_nic_out_4_nic_in_7 = 1 ->
   - 3167.736021998167 hold_7_1 - 1757.884028484232 hold_7_2
   - 1216.473072861668 hold_7_3 - 930.032292787944 hold_7_4
   - 752.7771727292528 hold_7_5 - 632.2722283205268 hold_7_6
   - 545.0244440939914 hold_7_7 - 478.9356984478936 hold_7_8
   - 427.1412680756396 hold_7_9 - 385.4561677448137 hold_7_10
   - 351.1838227822376 hold_7_11 - 322.5083986562149 hold_7_12
   - 298.1623673539816 hold_7_13 + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_4_nic_in_8: switch_nic_out_4_nic_in_8 = 1 ->
   - 3167.736021998167 hold_8_1 - 1757.884028484232 hold_8_2
   - 1216.473072861668 hold_8_3 - 930.032292787944 hold_8_4
   - 752.7771727292528 hold_8_5 - 632.2722283205268 hold_8_6
   - 545.0244440939914 hold_8_7 - 478.9356984478936 hold_8_8
   - 427.1412680756396 hold_8_9 - 385.4561677448137 hold_8_10
   - 351.1838227822376 hold_8_11 - 322.5083986562149 hold_8_12
   - 298.1623673539816 hold_8_13 + flow_nic_in_8_8 + flow_9_8 <= 0
 constr_intergroup_throughput_10_nic_out_4_nic_in_10: switch_nic_out_4_nic_in_10 = 1 ->
   - 3367.47123843898 hold_10_1 - 1881.759737804109 hold_10_2
   - 1305.694043558121 hold_10_3 - 999.665171097569 hold_10_4
   - 809.8518960559865 hold_10_5 - 680.6182464779099 hold_10_6
   + flow_nic_in_10_10 + flow_11_10 <= 0
 constr_intergroup_throughput_12_nic_out_4_nic_in_12: switch_nic_out_4_nic_in_12 = 1 ->
   - 1263.157894736842 hold_12_1 - 657.5342465753424 hold_12_2
   - 444.4444444444444 hold_12_3 - 335.6643356643356 hold_12_4
   - 269.6629213483146 hold_12_5 - 225.3521126760563 hold_12_6
   + flow_nic_in_12_12 + flow_13_12 <= 0
 constr_intergroup_throughput_14_nic_out_4_nic_in_14: switch_nic_out_4_nic_in_14 = 1 ->
   - 1815.602836879433 hold_14_1 - 962.406015037594 hold_14_2
   - 654.7314578005115 hold_14_3 - 496.1240310077519 hold_14_4
   + flow_nic_in_14_14 + flow_15_14 + flow_16_14 + flow_17_14 <= 0
 constr_intergroup_throughput_6_source_nic_in_6: switch_source_nic_in_6 = 1 ->
   - 3763.885863646265 hold_6_1 - 1927.280838724069 hold_6_2
   - 1295.25522824376 hold_6_3 - 975.3894784375707 hold_6_4
   - 782.2190032139786 hold_6_5 - 652.9131716164135 hold_6_6
   - 560.2931163062157 hold_6_7 - 490.6860517946388 hold_6_8
   - 436.4628324619231 hold_6_9 - 393.0309784833735 hold_6_10
   - 357.4605407418134 hold_6_11 - 327.7942180742089 hold_6_12
   - 302.6746772696222 hold_6_13 + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_0_nic_out_6_nic_in_0: switch_nic_out_6_nic_in_0 = 1 ->
   - 7824.817518248176 hold_0_1 - 5178.743961352658 hold_0_2
   - 3870.036101083032 hold_0_3 - 3089.337175792507 hold_0_4
   - 2570.743405275779 hold_0_5 - 2201.23203285421 hold_0_6
   - 1924.5960502693 hold_0_7 - 1709.728867623605 hold_0_8
   - 1538.020086083214 hold_0_9 - 1397.653194263364 hold_0_10
   - 1280.764635603345 hold_0_11 - 1181.918412348401 hold_0_12
   - 1097.236438075742 hold_0_13 - 1023.877745940783 hold_0_14
   - 959.7135183527306 hold_0_15 - 903.117101937658 hold_0_16
   - 852.8241845664279 hold_0_17 - 807.8372268274302 hold_0_18
   - 767.3586256263421 hold_0_19 - 730.7430129516018 hold_0_20
   - 697.4625894599869 hold_0_21 - 667.0815183571873 hold_0_22
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_6_nic_in_1: switch_nic_out_6_nic_in_1 = 1 ->
   - 4921.859545004946 hold_1_1 - 2908.240794856809 hold_1_2
   - 2063.873911240149 hold_1_3 - 1599.485695917711 hold_1_4
   - 1305.694043558121 hold_1_5 - 1103.081356683662 hold_1_6
   - 954.9030896181155 hold_1_7 - 841.8203349687025 hold_1_8
   - 752.6849190742701 hold_1_9 - 680.6182464779099 hold_1_10
   - 621.1459243540132 hold_1_11 + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_6_nic_in_2: switch_nic_out_6_nic_in_2 = 1 ->
   - 5891.696750902527 hold_2_1 - 3610.619469026549 hold_2_2
   - 2602.870813397129 hold_2_3 - 2034.912718204489 hold_2_4
   - 1670.419651995905 hold_2_5 - 1416.666666666667 hold_2_6
   - 1229.841748304446 hold_2_7 - 1086.551264980027 hold_2_8
   - 973.1663685152056 hold_2_9 - 881.2095032397407 hold_2_10
   - 805.1307350764677 hold_2_11 - 741.1444141689373 hold_2_12
   - 686.5797223390829 hold_2_13 - 639.4984326018808 hold_2_14
   - 598.4598459845985 hold_2_15 - 562.3707787732598 hold_2_16
   - 530.3867403314916 hold_2_17 - 501.8450184501845 hold_2_18
   - 476.2182667055733 hold_2_19 - 453.081621321488 hold_2_20
   - 432.0889594916601 hold_2_21 - 412.9554655870446 hold_2_22
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_6_nic_in_3: switch_nic_out_6_nic_in_3 = 1 ->
   - 4921.859545004946 hold_3_1 - 2908.240794856809 hold_3_2
   - 2063.873911240149 hold_3_3 - 1599.485695917711 hold_3_4
   - 1305.694043558121 hold_3_5 - 1103.081356683662 hold_3_6
   - 954.9030896181155 hold_3_7 - 841.8203349687025 hold_3_8
   - 752.6849190742701 hold_3_9 - 680.6182464779099 hold_3_10
   - 621.1459243540132 hold_3_11 + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_6_nic_in_4: switch_nic_out_6_nic_in_4 = 1 ->
   - 3167.736021998167 hold_4_1 - 1757.884028484232 hold_4_2
   - 1216.473072861668 hold_4_3 - 930.032292787944 hold_4_4
   - 752.7771727292528 hold_4_5 - 632.2722283205268 hold_4_6
   - 545.0244440939914 hold_4_7 - 478.9356984478936 hold_4_8
   - 427.1412680756396 hold_4_9 - 385.4561677448137 hold_4_10
   - 351.1838227822376 hold_4_11 - 322.5083986562149 hold_4_12
   - 298.1623673539816 hold_4_13 + flow_nic_in_4_4 + flow_5_4 <= 0
 constr_intergroup_throughput_7_nic_out_6_nic_in_7: switch_nic_out_6_nic_in_7 = 1 ->
   - 3167.736021998167 hold_7_1 - 1757.884028484232 hold_7_2
   - 1216.473072861668 hold_7_3 - 930.032292787944 hold_7_4
   - 752.7771727292528 hold_7_5 - 632.2722283205268 hold_7_6
   - 545.0244440939914 hold_7_7 - 478.9356984478936 hold_7_8
   - 427.1412680756396 hold_7_9 - 385.4561677448137 hold_7_10
   - 351.1838227822376 hold_7_11 - 322.5083986562149 hold_7_12
   - 298.1623673539816 hold_7_13 + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_6_nic_in_8: switch_nic_out_6_nic_in_8 = 1 ->
   - 3167.736021998167 hold_8_1 - 1757.884028484232 hold_8_2
   - 1216.473072861668 hold_8_3 - 930.032292787944 hold_8_4
   - 752.7771727292528 hold_8_5 - 632.2722283205268 hold_8_6
   - 545.0244440939914 hold_8_7 - 478.9356984478936 hold_8_8
   - 427.1412680756396 hold_8_9 - 385.4561677448137 hold_8_10
   - 351.1838227822376 hold_8_11 - 322.5083986562149 hold_8_12
   - 298.1623673539816 hold_8_13 + flow_nic_in_8_8 + flow_9_8 <= 0
 constr_intergroup_throughput_10_nic_out_6_nic_in_10: switch_nic_out_6_nic_in_10 = 1 ->
   - 3367.47123843898 hold_10_1 - 1881.759737804109 hold_10_2
   - 1305.694043558121 hold_10_3 - 999.665171097569 hold_10_4
   - 809.8518960559865 hold_10_5 - 680.6182464779099 hold_10_6
   + flow_nic_in_10_10 + flow_11_10 <= 0
 constr_intergroup_throughput_12_nic_out_6_nic_in_12: switch_nic_out_6_nic_in_12 = 1 ->
   - 1263.157894736842 hold_12_1 - 657.5342465753424 hold_12_2
   - 444.4444444444444 hold_12_3 - 335.6643356643356 hold_12_4
   - 269.6629213483146 hold_12_5 - 225.3521126760563 hold_12_6
   + flow_nic_in_12_12 + flow_13_12 <= 0
 constr_intergroup_throughput_14_nic_out_6_nic_in_14: switch_nic_out_6_nic_in_14 = 1 ->
   - 1815.602836879433 hold_14_1 - 962.406015037594 hold_14_2
   - 654.7314578005115 hold_14_3 - 496.1240310077519 hold_14_4
   + flow_nic_in_14_14 + flow_15_14 + flow_16_14 + flow_17_14 <= 0
 constr_intergroup_throughput_7_source_nic_in_7: switch_source_nic_in_7 = 1 ->
   - 3763.885863646265 hold_7_1 - 1927.280838724069 hold_7_2
   - 1295.25522824376 hold_7_3 - 975.3894784375707 hold_7_4
   - 782.2190032139786 hold_7_5 - 652.9131716164135 hold_7_6
   - 560.2931163062157 hold_7_7 - 490.6860517946388 hold_7_8
   - 436.4628324619231 hold_7_9 - 393.0309784833735 hold_7_10
   - 357.4605407418134 hold_7_11 - 327.7942180742089 hold_7_12
   - 302.6746772696222 hold_7_13 + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_0_nic_out_7_nic_in_0: switch_nic_out_7_nic_in_0 = 1 ->
   - 7824.817518248176 hold_0_1 - 5178.743961352658 hold_0_2
   - 3870.036101083032 hold_0_3 - 3089.337175792507 hold_0_4
   - 2570.743405275779 hold_0_5 - 2201.23203285421 hold_0_6
   - 1924.5960502693 hold_0_7 - 1709.728867623605 hold_0_8
   - 1538.020086083214 hold_0_9 - 1397.653194263364 hold_0_10
   - 1280.764635603345 hold_0_11 - 1181.918412348401 hold_0_12
   - 1097.236438075742 hold_0_13 - 1023.877745940783 hold_0_14
   - 959.7135183527306 hold_0_15 - 903.117101937658 hold_0_16
   - 852.8241845664279 hold_0_17 - 807.8372268274302 hold_0_18
   - 767.3586256263421 hold_0_19 - 730.7430129516018 hold_0_20
   - 697.4625894599869 hold_0_21 - 667.0815183571873 hold_0_22
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_7_nic_in_1: switch_nic_out_7_nic_in_1 = 1 ->
   - 4921.859545004946 hold_1_1 - 2908.240794856809 hold_1_2
   - 2063.873911240149 hold_1_3 - 1599.485695917711 hold_1_4
   - 1305.694043558121 hold_1_5 - 1103.081356683662 hold_1_6
   - 954.9030896181155 hold_1_7 - 841.8203349687025 hold_1_8
   - 752.6849190742701 hold_1_9 - 680.6182464779099 hold_1_10
   - 621.1459243540132 hold_1_11 + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_7_nic_in_2: switch_nic_out_7_nic_in_2 = 1 ->
   - 5891.696750902527 hold_2_1 - 3610.619469026549 hold_2_2
   - 2602.870813397129 hold_2_3 - 2034.912718204489 hold_2_4
   - 1670.419651995905 hold_2_5 - 1416.666666666667 hold_2_6
   - 1229.841748304446 hold_2_7 - 1086.551264980027 hold_2_8
   - 973.1663685152056 hold_2_9 - 881.2095032397407 hold_2_10
   - 805.1307350764677 hold_2_11 - 741.1444141689373 hold_2_12
   - 686.5797223390829 hold_2_13 - 639.4984326018808 hold_2_14
   - 598.4598459845985 hold_2_15 - 562.3707787732598 hold_2_16
   - 530.3867403314916 hold_2_17 - 501.8450184501845 hold_2_18
   - 476.2182667055733 hold_2_19 - 453.081621321488 hold_2_20
   - 432.0889594916601 hold_2_21 - 412.9554655870446 hold_2_22
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_7_nic_in_3: switch_nic_out_7_nic_in_3 = 1 ->
   - 4921.859545004946 hold_3_1 - 2908.240794856809 hold_3_2
   - 2063.873911240149 hold_3_3 - 1599.485695917711 hold_3_4
   - 1305.694043558121 hold_3_5 - 1103.081356683662 hold_3_6
   - 954.9030896181155 hold_3_7 - 841.8203349687025 hold_3_8
   - 752.6849190742701 hold_3_9 - 680.6182464779099 hold_3_10
   - 621.1459243540132 hold_3_11 + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_7_nic_in_4: switch_nic_out_7_nic_in_4 = 1 ->
   - 3167.736021998167 hold_4_1 - 1757.884028484232 hold_4_2
   - 1216.473072861668 hold_4_3 - 930.032292787944 hold_4_4
   - 752.7771727292528 hold_4_5 - 632.2722283205268 hold_4_6
   - 545.0244440939914 hold_4_7 - 478.9356984478936 hold_4_8
   - 427.1412680756396 hold_4_9 - 385.4561677448137 hold_4_10
   - 351.1838227822376 hold_4_11 - 322.5083986562149 hold_4_12
   - 298.1623673539816 hold_4_13 + flow_nic_in_4_4 + flow_5_4 <= 0
 constr_intergroup_throughput_6_nic_out_7_nic_in_6: switch_nic_out_7_nic_in_6 = 1 ->
   - 3167.736021998167 hold_6_1 - 1757.884028484232 hold_6_2
   - 1216.473072861668 hold_6_3 - 930.032292787944 hold_6_4
   - 752.7771727292528 hold_6_5 - 632.2722283205268 hold_6_6
   - 545.0244440939914 hold_6_7 - 478.9356984478936 hold_6_8
   - 427.1412680756396 hold_6_9 - 385.4561677448137 hold_6_10
   - 351.1838227822376 hold_6_11 - 322.5083986562149 hold_6_12
   - 298.1623673539816 hold_6_13 + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_8_nic_out_7_nic_in_8: switch_nic_out_7_nic_in_8 = 1 ->
   - 3167.736021998167 hold_8_1 - 1757.884028484232 hold_8_2
   - 1216.473072861668 hold_8_3 - 930.032292787944 hold_8_4
   - 752.7771727292528 hold_8_5 - 632.2722283205268 hold_8_6
   - 545.0244440939914 hold_8_7 - 478.9356984478936 hold_8_8
   - 427.1412680756396 hold_8_9 - 385.4561677448137 hold_8_10
   - 351.1838227822376 hold_8_11 - 322.5083986562149 hold_8_12
   - 298.1623673539816 hold_8_13 + flow_nic_in_8_8 + flow_9_8 <= 0
 constr_intergroup_throughput_10_nic_out_7_nic_in_10: switch_nic_out_7_nic_in_10 = 1 ->
   - 3367.47123843898 hold_10_1 - 1881.759737804109 hold_10_2
   - 1305.694043558121 hold_10_3 - 999.665171097569 hold_10_4
   - 809.8518960559865 hold_10_5 - 680.6182464779099 hold_10_6
   + flow_nic_in_10_10 + flow_11_10 <= 0
 constr_intergroup_throughput_12_nic_out_7_nic_in_12: switch_nic_out_7_nic_in_12 = 1 ->
   - 1263.157894736842 hold_12_1 - 657.5342465753424 hold_12_2
   - 444.4444444444444 hold_12_3 - 335.6643356643356 hold_12_4
   - 269.6629213483146 hold_12_5 - 225.3521126760563 hold_12_6
   + flow_nic_in_12_12 + flow_13_12 <= 0
 constr_intergroup_throughput_14_nic_out_7_nic_in_14: switch_nic_out_7_nic_in_14 = 1 ->
   - 1815.602836879433 hold_14_1 - 962.406015037594 hold_14_2
   - 654.7314578005115 hold_14_3 - 496.1240310077519 hold_14_4
   + flow_nic_in_14_14 + flow_15_14 + flow_16_14 + flow_17_14 <= 0
 constr_intergroup_throughput_8_source_nic_in_8: switch_source_nic_in_8 = 1 ->
   - 3763.885863646265 hold_8_1 - 1927.280838724069 hold_8_2
   - 1295.25522824376 hold_8_3 - 975.3894784375707 hold_8_4
   - 782.2190032139786 hold_8_5 - 652.9131716164135 hold_8_6
   - 560.2931163062157 hold_8_7 - 490.6860517946388 hold_8_8
   - 436.4628324619231 hold_8_9 - 393.0309784833735 hold_8_10
   - 357.4605407418134 hold_8_11 - 327.7942180742089 hold_8_12
   - 302.6746772696222 hold_8_13 + flow_nic_in_8_8 + flow_9_8 <= 0
 constr_intergroup_throughput_0_nic_out_8_nic_in_0: switch_nic_out_8_nic_in_0 = 1 ->
   - 7824.817518248176 hold_0_1 - 5178.743961352658 hold_0_2
   - 3870.036101083032 hold_0_3 - 3089.337175792507 hold_0_4
   - 2570.743405275779 hold_0_5 - 2201.23203285421 hold_0_6
   - 1924.5960502693 hold_0_7 - 1709.728867623605 hold_0_8
   - 1538.020086083214 hold_0_9 - 1397.653194263364 hold_0_10
   - 1280.764635603345 hold_0_11 - 1181.918412348401 hold_0_12
   - 1097.236438075742 hold_0_13 - 1023.877745940783 hold_0_14
   - 959.7135183527306 hold_0_15 - 903.117101937658 hold_0_16
   - 852.8241845664279 hold_0_17 - 807.8372268274302 hold_0_18
   - 767.3586256263421 hold_0_19 - 730.7430129516018 hold_0_20
   - 697.4625894599869 hold_0_21 - 667.0815183571873 hold_0_22
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_8_nic_in_1: switch_nic_out_8_nic_in_1 = 1 ->
   - 4921.859545004946 hold_1_1 - 2908.240794856809 hold_1_2
   - 2063.873911240149 hold_1_3 - 1599.485695917711 hold_1_4
   - 1305.694043558121 hold_1_5 - 1103.081356683662 hold_1_6
   - 954.9030896181155 hold_1_7 - 841.8203349687025 hold_1_8
   - 752.6849190742701 hold_1_9 - 680.6182464779099 hold_1_10
   - 621.1459243540132 hold_1_11 + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_8_nic_in_2: switch_nic_out_8_nic_in_2 = 1 ->
   - 5891.696750902527 hold_2_1 - 3610.619469026549 hold_2_2
   - 2602.870813397129 hold_2_3 - 2034.912718204489 hold_2_4
   - 1670.419651995905 hold_2_5 - 1416.666666666667 hold_2_6
   - 1229.841748304446 hold_2_7 - 1086.551264980027 hold_2_8
   - 973.1663685152056 hold_2_9 - 881.2095032397407 hold_2_10
   - 805.1307350764677 hold_2_11 - 741.1444141689373 hold_2_12
   - 686.5797223390829 hold_2_13 - 639.4984326018808 hold_2_14
   - 598.4598459845985 hold_2_15 - 562.3707787732598 hold_2_16
   - 530.3867403314916 hold_2_17 - 501.8450184501845 hold_2_18
   - 476.2182667055733 hold_2_19 - 453.081621321488 hold_2_20
   - 432.0889594916601 hold_2_21 - 412.9554655870446 hold_2_22
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_8_nic_in_3: switch_nic_out_8_nic_in_3 = 1 ->
   - 4921.859545004946 hold_3_1 - 2908.240794856809 hold_3_2
   - 2063.873911240149 hold_3_3 - 1599.485695917711 hold_3_4
   - 1305.694043558121 hold_3_5 - 1103.081356683662 hold_3_6
   - 954.9030896181155 hold_3_7 - 841.8203349687025 hold_3_8
   - 752.6849190742701 hold_3_9 - 680.6182464779099 hold_3_10
   - 621.1459243540132 hold_3_11 + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_8_nic_in_4: switch_nic_out_8_nic_in_4 = 1 ->
   - 3167.736021998167 hold_4_1 - 1757.884028484232 hold_4_2
   - 1216.473072861668 hold_4_3 - 930.032292787944 hold_4_4
   - 752.7771727292528 hold_4_5 - 632.2722283205268 hold_4_6
   - 545.0244440939914 hold_4_7 - 478.9356984478936 hold_4_8
   - 427.1412680756396 hold_4_9 - 385.4561677448137 hold_4_10
   - 351.1838227822376 hold_4_11 - 322.5083986562149 hold_4_12
   - 298.1623673539816 hold_4_13 + flow_nic_in_4_4 + flow_5_4 <= 0
 constr_intergroup_throughput_6_nic_out_8_nic_in_6: switch_nic_out_8_nic_in_6 = 1 ->
   - 3167.736021998167 hold_6_1 - 1757.884028484232 hold_6_2
   - 1216.473072861668 hold_6_3 - 930.032292787944 hold_6_4
   - 752.7771727292528 hold_6_5 - 632.2722283205268 hold_6_6
   - 545.0244440939914 hold_6_7 - 478.9356984478936 hold_6_8
   - 427.1412680756396 hold_6_9 - 385.4561677448137 hold_6_10
   - 351.1838227822376 hold_6_11 - 322.5083986562149 hold_6_12
   - 298.1623673539816 hold_6_13 + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_8_nic_in_7: switch_nic_out_8_nic_in_7 = 1 ->
   - 3167.736021998167 hold_7_1 - 1757.884028484232 hold_7_2
   - 1216.473072861668 hold_7_3 - 930.032292787944 hold_7_4
   - 752.7771727292528 hold_7_5 - 632.2722283205268 hold_7_6
   - 545.0244440939914 hold_7_7 - 478.9356984478936 hold_7_8
   - 427.1412680756396 hold_7_9 - 385.4561677448137 hold_7_10
   - 351.1838227822376 hold_7_11 - 322.5083986562149 hold_7_12
   - 298.1623673539816 hold_7_13 + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_10_nic_out_8_nic_in_10: switch_nic_out_8_nic_in_10 = 1 ->
   - 3367.47123843898 hold_10_1 - 1881.759737804109 hold_10_2
   - 1305.694043558121 hold_10_3 - 999.665171097569 hold_10_4
   - 809.8518960559865 hold_10_5 - 680.6182464779099 hold_10_6
   + flow_nic_in_10_10 + flow_11_10 <= 0
 constr_intergroup_throughput_12_nic_out_8_nic_in_12: switch_nic_out_8_nic_in_12 = 1 ->
   - 1263.157894736842 hold_12_1 - 657.5342465753424 hold_12_2
   - 444.4444444444444 hold_12_3 - 335.6643356643356 hold_12_4
   - 269.6629213483146 hold_12_5 - 225.3521126760563 hold_12_6
   + flow_nic_in_12_12 + flow_13_12 <= 0
 constr_intergroup_throughput_14_nic_out_8_nic_in_14: switch_nic_out_8_nic_in_14 = 1 ->
   - 1815.602836879433 hold_14_1 - 962.406015037594 hold_14_2
   - 654.7314578005115 hold_14_3 - 496.1240310077519 hold_14_4
   + flow_nic_in_14_14 + flow_15_14 + flow_16_14 + flow_17_14 <= 0
 constr_intergroup_throughput_10_source_nic_in_10: switch_source_nic_in_10 = 1 ->
   - 4049.259480279933 hold_10_1 - 2077.199231903821 hold_10_2
   - 1396.889562629835 hold_10_3 - 1052.260583931315 hold_10_4
   - 844.0288127735122 hold_10_5 - 704.5963014358132 hold_10_6
   + flow_nic_in_10_10 + flow_11_10 <= 0
 constr_intergroup_throughput_0_nic_out_10_nic_in_0: switch_nic_out_10_nic_in_0 = 1 ->
   - 7824.817518248176 hold_0_1 - 5178.743961352658 hold_0_2
   - 3870.036101083032 hold_0_3 - 3089.337175792507 hold_0_4
   - 2570.743405275779 hold_0_5 - 2201.23203285421 hold_0_6
   - 1924.5960502693 hold_0_7 - 1709.728867623605 hold_0_8
   - 1538.020086083214 hold_0_9 - 1397.653194263364 hold_0_10
   - 1280.764635603345 hold_0_11 - 1181.918412348401 hold_0_12
   - 1097.236438075742 hold_0_13 - 1023.877745940783 hold_0_14
   - 959.7135183527306 hold_0_15 - 903.117101937658 hold_0_16
   - 852.8241845664279 hold_0_17 - 807.8372268274302 hold_0_18
   - 767.3586256263421 hold_0_19 - 730.7430129516018 hold_0_20
   - 697.4625894599869 hold_0_21 - 667.0815183571873 hold_0_22
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_10_nic_in_1: switch_nic_out_10_nic_in_1 = 1 ->
   - 4921.859545004946 hold_1_1 - 2908.240794856809 hold_1_2
   - 2063.873911240149 hold_1_3 - 1599.485695917711 hold_1_4
   - 1305.694043558121 hold_1_5 - 1103.081356683662 hold_1_6
   - 954.9030896181155 hold_1_7 - 841.8203349687025 hold_1_8
   - 752.6849190742701 hold_1_9 - 680.6182464779099 hold_1_10
   - 621.1459243540132 hold_1_11 + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_10_nic_in_2: switch_nic_out_10_nic_in_2 = 1 ->
   - 5891.696750902527 hold_2_1 - 3610.619469026549 hold_2_2
   - 2602.870813397129 hold_2_3 - 2034.912718204489 hold_2_4
   - 1670.419651995905 hold_2_5 - 1416.666666666667 hold_2_6
   - 1229.841748304446 hold_2_7 - 1086.551264980027 hold_2_8
   - 973.1663685152056 hold_2_9 - 881.2095032397407 hold_2_10
   - 805.1307350764677 hold_2_11 - 741.1444141689373 hold_2_12
   - 686.5797223390829 hold_2_13 - 639.4984326018808 hold_2_14
   - 598.4598459845985 hold_2_15 - 562.3707787732598 hold_2_16
   - 530.3867403314916 hold_2_17 - 501.8450184501845 hold_2_18
   - 476.2182667055733 hold_2_19 - 453.081621321488 hold_2_20
   - 432.0889594916601 hold_2_21 - 412.9554655870446 hold_2_22
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_10_nic_in_3: switch_nic_out_10_nic_in_3 = 1 ->
   - 4921.859545004946 hold_3_1 - 2908.240794856809 hold_3_2
   - 2063.873911240149 hold_3_3 - 1599.485695917711 hold_3_4
   - 1305.694043558121 hold_3_5 - 1103.081356683662 hold_3_6
   - 954.9030896181155 hold_3_7 - 841.8203349687025 hold_3_8
   - 752.6849190742701 hold_3_9 - 680.6182464779099 hold_3_10
   - 621.1459243540132 hold_3_11 + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_10_nic_in_4: switch_nic_out_10_nic_in_4 = 1 ->
   - 3167.736021998167 hold_4_1 - 1757.884028484232 hold_4_2
   - 1216.473072861668 hold_4_3 - 930.032292787944 hold_4_4
   - 752.7771727292528 hold_4_5 - 632.2722283205268 hold_4_6
   - 545.0244440939914 hold_4_7 - 478.9356984478936 hold_4_8
   - 427.1412680756396 hold_4_9 - 385.4561677448137 hold_4_10
   - 351.1838227822376 hold_4_11 - 322.5083986562149 hold_4_12
   - 298.1623673539816 hold_4_13 + flow_nic_in_4_4 + flow_5_4 <= 0
 constr_intergroup_throughput_6_nic_out_10_nic_in_6: switch_nic_out_10_nic_in_6 = 1 ->
   - 3167.736021998167 hold_6_1 - 1757.884028484232 hold_6_2
   - 1216.473072861668 hold_6_3 - 930.032292787944 hold_6_4
   - 752.7771727292528 hold_6_5 - 632.2722283205268 hold_6_6
   - 545.0244440939914 hold_6_7 - 478.9356984478936 hold_6_8
   - 427.1412680756396 hold_6_9 - 385.4561677448137 hold_6_10
   - 351.1838227822376 hold_6_11 - 322.5083986562149 hold_6_12
   - 298.1623673539816 hold_6_13 + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_10_nic_in_7: switch_nic_out_10_nic_in_7 = 1 ->
   - 3167.736021998167 hold_7_1 - 1757.884028484232 hold_7_2
   - 1216.473072861668 hold_7_3 - 930.032292787944 hold_7_4
   - 752.7771727292528 hold_7_5 - 632.2722283205268 hold_7_6
   - 545.0244440939914 hold_7_7 - 478.9356984478936 hold_7_8
   - 427.1412680756396 hold_7_9 - 385.4561677448137 hold_7_10
   - 351.1838227822376 hold_7_11 - 322.5083986562149 hold_7_12
   - 298.1623673539816 hold_7_13 + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_10_nic_in_8: switch_nic_out_10_nic_in_8 = 1 ->
   - 3167.736021998167 hold_8_1 - 1757.884028484232 hold_8_2
   - 1216.473072861668 hold_8_3 - 930.032292787944 hold_8_4
   - 752.7771727292528 hold_8_5 - 632.2722283205268 hold_8_6
   - 545.0244440939914 hold_8_7 - 478.9356984478936 hold_8_8
   - 427.1412680756396 hold_8_9 - 385.4561677448137 hold_8_10
   - 351.1838227822376 hold_8_11 - 322.5083986562149 hold_8_12
   - 298.1623673539816 hold_8_13 + flow_nic_in_8_8 + flow_9_8 <= 0
 constr_intergroup_throughput_12_nic_out_10_nic_in_12: switch_nic_out_10_nic_in_12 = 1 ->
   - 1263.157894736842 hold_12_1 - 657.5342465753424 hold_12_2
   - 444.4444444444444 hold_12_3 - 335.6643356643356 hold_12_4
   - 269.6629213483146 hold_12_5 - 225.3521126760563 hold_12_6
   + flow_nic_in_12_12 + flow_13_12 <= 0
 constr_intergroup_throughput_14_nic_out_10_nic_in_14: switch_nic_out_10_nic_in_14 = 1 ->
   - 1815.602836879433 hold_14_1 - 962.406015037594 hold_14_2
   - 654.7314578005115 hold_14_3 - 496.1240310077519 hold_14_4
   + flow_nic_in_14_14 + flow_15_14 + flow_16_14 + flow_17_14 <= 0
 constr_intergroup_throughput_12_source_nic_in_12: switch_source_nic_in_12 = 1 ->
   - 1348.314606741573 hold_12_1 - 679.8866855524079 hold_12_2
   - 454.5454545454545 hold_12_3 - 341.3940256045519 hold_12_4
   - 273.3485193621868 hold_12_5 - 227.9202279202279 hold_12_6
   + flow_nic_in_12_12 + flow_13_12 <= 0
 constr_intergroup_throughput_0_nic_out_12_nic_in_0: switch_nic_out_12_nic_in_0 = 1 ->
   - 7824.817518248176 hold_0_1 - 5178.743961352658 hold_0_2
   - 3870.036101083032 hold_0_3 - 3089.337175792507 hold_0_4
   - 2570.743405275779 hold_0_5 - 2201.23203285421 hold_0_6
   - 1924.5960502693 hold_0_7 - 1709.728867623605 hold_0_8
   - 1538.020086083214 hold_0_9 - 1397.653194263364 hold_0_10
   - 1280.764635603345 hold_0_11 - 1181.918412348401 hold_0_12
   - 1097.236438075742 hold_0_13 - 1023.877745940783 hold_0_14
   - 959.7135183527306 hold_0_15 - 903.117101937658 hold_0_16
   - 852.8241845664279 hold_0_17 - 807.8372268274302 hold_0_18
   - 767.3586256263421 hold_0_19 - 730.7430129516018 hold_0_20
   - 697.4625894599869 hold_0_21 - 667.0815183571873 hold_0_22
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_12_nic_in_1: switch_nic_out_12_nic_in_1 = 1 ->
   - 4921.859545004946 hold_1_1 - 2908.240794856809 hold_1_2
   - 2063.873911240149 hold_1_3 - 1599.485695917711 hold_1_4
   - 1305.694043558121 hold_1_5 - 1103.081356683662 hold_1_6
   - 954.9030896181155 hold_1_7 - 841.8203349687025 hold_1_8
   - 752.6849190742701 hold_1_9 - 680.6182464779099 hold_1_10
   - 621.1459243540132 hold_1_11 + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_12_nic_in_2: switch_nic_out_12_nic_in_2 = 1 ->
   - 5891.696750902527 hold_2_1 - 3610.619469026549 hold_2_2
   - 2602.870813397129 hold_2_3 - 2034.912718204489 hold_2_4
   - 1670.419651995905 hold_2_5 - 1416.666666666667 hold_2_6
   - 1229.841748304446 hold_2_7 - 1086.551264980027 hold_2_8
   - 973.1663685152056 hold_2_9 - 881.2095032397407 hold_2_10
   - 805.1307350764677 hold_2_11 - 741.1444141689373 hold_2_12
   - 686.5797223390829 hold_2_13 - 639.4984326018808 hold_2_14
   - 598.4598459845985 hold_2_15 - 562.3707787732598 hold_2_16
   - 530.3867403314916 hold_2_17 - 501.8450184501845 hold_2_18
   - 476.2182667055733 hold_2_19 - 453.081621321488 hold_2_20
   - 432.0889594916601 hold_2_21 - 412.9554655870446 hold_2_22
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_12_nic_in_3: switch_nic_out_12_nic_in_3 = 1 ->
   - 4921.859545004946 hold_3_1 - 2908.240794856809 hold_3_2
   - 2063.873911240149 hold_3_3 - 1599.485695917711 hold_3_4
   - 1305.694043558121 hold_3_5 - 1103.081356683662 hold_3_6
   - 954.9030896181155 hold_3_7 - 841.8203349687025 hold_3_8
   - 752.6849190742701 hold_3_9 - 680.6182464779099 hold_3_10
   - 621.1459243540132 hold_3_11 + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_12_nic_in_4: switch_nic_out_12_nic_in_4 = 1 ->
   - 3167.736021998167 hold_4_1 - 1757.884028484232 hold_4_2
   - 1216.473072861668 hold_4_3 - 930.032292787944 hold_4_4
   - 752.7771727292528 hold_4_5 - 632.2722283205268 hold_4_6
   - 545.0244440939914 hold_4_7 - 478.9356984478936 hold_4_8
   - 427.1412680756396 hold_4_9 - 385.4561677448137 hold_4_10
   - 351.1838227822376 hold_4_11 - 322.5083986562149 hold_4_12
   - 298.1623673539816 hold_4_13 + flow_nic_in_4_4 + flow_5_4 <= 0
 constr_intergroup_throughput_6_nic_out_12_nic_in_6: switch_nic_out_12_nic_in_6 = 1 ->
   - 3167.736021998167 hold_6_1 - 1757.884028484232 hold_6_2
   - 1216.473072861668 hold_6_3 - 930.032292787944 hold_6_4
   - 752.7771727292528 hold_6_5 - 632.2722283205268 hold_6_6
   - 545.0244440939914 hold_6_7 - 478.9356984478936 hold_6_8
   - 427.1412680756396 hold_6_9 - 385.4561677448137 hold_6_10
   - 351.1838227822376 hold_6_11 - 322.5083986562149 hold_6_12
   - 298.1623673539816 hold_6_13 + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_12_nic_in_7: switch_nic_out_12_nic_in_7 = 1 ->
   - 3167.736021998167 hold_7_1 - 1757.884028484232 hold_7_2
   - 1216.473072861668 hold_7_3 - 930.032292787944 hold_7_4
   - 752.7771727292528 hold_7_5 - 632.2722283205268 hold_7_6
   - 545.0244440939914 hold_7_7 - 478.9356984478936 hold_7_8
   - 427.1412680756396 hold_7_9 - 385.4561677448137 hold_7_10
   - 351.1838227822376 hold_7_11 - 322.5083986562149 hold_7_12
   - 298.1623673539816 hold_7_13 + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_12_nic_in_8: switch_nic_out_12_nic_in_8 = 1 ->
   - 3167.736021998167 hold_8_1 - 1757.884028484232 hold_8_2
   - 1216.473072861668 hold_8_3 - 930.032292787944 hold_8_4
   - 752.7771727292528 hold_8_5 - 632.2722283205268 hold_8_6
   - 545.0244440939914 hold_8_7 - 478.9356984478936 hold_8_8
   - 427.1412680756396 hold_8_9 - 385.4561677448137 hold_8_10
   - 351.1838227822376 hold_8_11 - 322.5083986562149 hold_8_12
   - 298.1623673539816 hold_8_13 + flow_nic_in_8_8 + flow_9_8 <= 0
 constr_intergroup_throughput_10_nic_out_12_nic_in_10: switch_nic_out_12_nic_in_10 = 1 ->
   - 3367.47123843898 hold_10_1 - 1881.759737804109 hold_10_2
   - 1305.694043558121 hold_10_3 - 999.665171097569 hold_10_4
   - 809.8518960559865 hold_10_5 - 680.6182464779099 hold_10_6
   + flow_nic_in_10_10 + flow_11_10 <= 0
 constr_intergroup_throughput_14_nic_out_12_nic_in_14: switch_nic_out_12_nic_in_14 = 1 ->
   - 1815.602836879433 hold_14_1 - 962.406015037594 hold_14_2
   - 654.7314578005115 hold_14_3 - 496.1240310077519 hold_14_4
   + flow_nic_in_14_14 + flow_15_14 + flow_16_14 + flow_17_14 <= 0
 constr_intergroup_throughput_14_source_nic_in_14: switch_source_nic_in_14 = 1 ->
   - 1996.879875195008 hold_14_1 - 1011.058451816746 hold_14_2
   - 676.8905341089371 hold_14_3 - 508.7440381558029 hold_14_4
   + flow_nic_in_14_14 + flow_15_14 + flow_16_14 + flow_17_14 <= 0
 constr_intergroup_throughput_0_nic_out_14_nic_in_0: switch_nic_out_14_nic_in_0 = 1 ->
   - 7824.817518248176 hold_0_1 - 5178.743961352658 hold_0_2
   - 3870.036101083032 hold_0_3 - 3089.337175792507 hold_0_4
   - 2570.743405275779 hold_0_5 - 2201.23203285421 hold_0_6
   - 1924.5960502693 hold_0_7 - 1709.728867623605 hold_0_8
   - 1538.020086083214 hold_0_9 - 1397.653194263364 hold_0_10
   - 1280.764635603345 hold_0_11 - 1181.918412348401 hold_0_12
   - 1097.236438075742 hold_0_13 - 1023.877745940783 hold_0_14
   - 959.7135183527306 hold_0_15 - 903.117101937658 hold_0_16
   - 852.8241845664279 hold_0_17 - 807.8372268274302 hold_0_18
   - 767.3586256263421 hold_0_19 - 730.7430129516018 hold_0_20
   - 697.4625894599869 hold_0_21 - 667.0815183571873 hold_0_22
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_14_nic_in_1: switch_nic_out_14_nic_in_1 = 1 ->
   - 4921.859545004946 hold_1_1 - 2908.240794856809 hold_1_2
   - 2063.873911240149 hold_1_3 - 1599.485695917711 hold_1_4
   - 1305.694043558121 hold_1_5 - 1103.081356683662 hold_1_6
   - 954.9030896181155 hold_1_7 - 841.8203349687025 hold_1_8
   - 752.6849190742701 hold_1_9 - 680.6182464779099 hold_1_10
   - 621.1459243540132 hold_1_11 + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_14_nic_in_2: switch_nic_out_14_nic_in_2 = 1 ->
   - 5891.696750902527 hold_2_1 - 3610.619469026549 hold_2_2
   - 2602.870813397129 hold_2_3 - 2034.912718204489 hold_2_4
   - 1670.419651995905 hold_2_5 - 1416.666666666667 hold_2_6
   - 1229.841748304446 hold_2_7 - 1086.551264980027 hold_2_8
   - 973.1663685152056 hold_2_9 - 881.2095032397407 hold_2_10
   - 805.1307350764677 hold_2_11 - 741.1444141689373 hold_2_12
   - 686.5797223390829 hold_2_13 - 639.4984326018808 hold_2_14
   - 598.4598459845985 hold_2_15 - 562.3707787732598 hold_2_16
   - 530.3867403314916 hold_2_17 - 501.8450184501845 hold_2_18
   - 476.2182667055733 hold_2_19 - 453.081621321488 hold_2_20
   - 432.0889594916601 hold_2_21 - 412.9554655870446 hold_2_22
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_14_nic_in_3: switch_nic_out_14_nic_in_3 = 1 ->
   - 4921.859545004946 hold_3_1 - 2908.240794856809 hold_3_2
   - 2063.873911240149 hold_3_3 - 1599.485695917711 hold_3_4
   - 1305.694043558121 hold_3_5 - 1103.081356683662 hold_3_6
   - 954.9030896181155 hold_3_7 - 841.8203349687025 hold_3_8
   - 752.6849190742701 hold_3_9 - 680.6182464779099 hold_3_10
   - 621.1459243540132 hold_3_11 + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_14_nic_in_4: switch_nic_out_14_nic_in_4 = 1 ->
   - 3167.736021998167 hold_4_1 - 1757.884028484232 hold_4_2
   - 1216.473072861668 hold_4_3 - 930.032292787944 hold_4_4
   - 752.7771727292528 hold_4_5 - 632.2722283205268 hold_4_6
   - 545.0244440939914 hold_4_7 - 478.9356984478936 hold_4_8
   - 427.1412680756396 hold_4_9 - 385.4561677448137 hold_4_10
   - 351.1838227822376 hold_4_11 - 322.5083986562149 hold_4_12
   - 298.1623673539816 hold_4_13 + flow_nic_in_4_4 + flow_5_4 <= 0
 constr_intergroup_throughput_6_nic_out_14_nic_in_6: switch_nic_out_14_nic_in_6 = 1 ->
   - 3167.736021998167 hold_6_1 - 1757.884028484232 hold_6_2
   - 1216.473072861668 hold_6_3 - 930.032292787944 hold_6_4
   - 752.7771727292528 hold_6_5 - 632.2722283205268 hold_6_6
   - 545.0244440939914 hold_6_7 - 478.9356984478936 hold_6_8
   - 427.1412680756396 hold_6_9 - 385.4561677448137 hold_6_10
   - 351.1838227822376 hold_6_11 - 322.5083986562149 hold_6_12
   - 298.1623673539816 hold_6_13 + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_14_nic_in_7: switch_nic_out_14_nic_in_7 = 1 ->
   - 3167.736021998167 hold_7_1 - 1757.884028484232 hold_7_2
   - 1216.473072861668 hold_7_3 - 930.032292787944 hold_7_4
   - 752.7771727292528 hold_7_5 - 632.2722283205268 hold_7_6
   - 545.0244440939914 hold_7_7 - 478.9356984478936 hold_7_8
   - 427.1412680756396 hold_7_9 - 385.4561677448137 hold_7_10
   - 351.1838227822376 hold_7_11 - 322.5083986562149 hold_7_12
   - 298.1623673539816 hold_7_13 + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_14_nic_in_8: switch_nic_out_14_nic_in_8 = 1 ->
   - 3167.736021998167 hold_8_1 - 1757.884028484232 hold_8_2
   - 1216.473072861668 hold_8_3 - 930.032292787944 hold_8_4
   - 752.7771727292528 hold_8_5 - 632.2722283205268 hold_8_6
   - 545.0244440939914 hold_8_7 - 478.9356984478936 hold_8_8
   - 427.1412680756396 hold_8_9 - 385.4561677448137 hold_8_10
   - 351.1838227822376 hold_8_11 - 322.5083986562149 hold_8_12
   - 298.1623673539816 hold_8_13 + flow_nic_in_8_8 + flow_9_8 <= 0
 constr_intergroup_throughput_10_nic_out_14_nic_in_10: switch_nic_out_14_nic_in_10 = 1 ->
   - 3367.47123843898 hold_10_1 - 1881.759737804109 hold_10_2
   - 1305.694043558121 hold_10_3 - 999.665171097569 hold_10_4
   - 809.8518960559865 hold_10_5 - 680.6182464779099 hold_10_6
   + flow_nic_in_10_10 + flow_11_10 <= 0
 constr_intergroup_throughput_12_nic_out_14_nic_in_12: switch_nic_out_14_nic_in_12 = 1 ->
   - 1263.157894736842 hold_12_1 - 657.5342465753424 hold_12_2
   - 444.4444444444444 hold_12_3 - 335.6643356643356 hold_12_4
   - 269.6629213483146 hold_12_5 - 225.3521126760563 hold_12_6
   + flow_nic_in_12_12 + flow_13_12 <= 0
 inactive_end_not_complete_0: is_active_0 = 0 -> is_end_complete_0 = 0
 active_end_val_0: is_active_0 = 1 -> - start_0 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11 - 12 hold_0_12
   - 13 hold_0_13 - 14 hold_0_14 - 15 hold_0_15 - 16 hold_0_16
   - 17 hold_0_17 - 18 hold_0_18 - 19 hold_0_19 - 20 hold_0_20
   - 21 hold_0_21 - 22 hold_0_22 - 23 hold_0_23 - 24 hold_0_24
   - 25 hold_0_25 - 26 hold_0_26 + val_end_0 = 0
 inactive_end_zero_0: is_active_0 = 0 -> val_end_0 = 0
 inactive_end_not_complete_1: is_active_1 = 0 -> is_end_complete_1 = 0
 active_end_val_1: is_active_1 = 1 -> - start_1 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11 - 12 hold_1_12
   - 13 hold_1_13 - 14 hold_1_14 - 15 hold_1_15 - 16 hold_1_16
   - 17 hold_1_17 - 18 hold_1_18 - 19 hold_1_19 - 20 hold_1_20
   - 21 hold_1_21 - 22 hold_1_22 - 23 hold_1_23 - 24 hold_1_24
   - 25 hold_1_25 - 26 hold_1_26 + val_end_1 = 0
 inactive_end_zero_1: is_active_1 = 0 -> val_end_1 = 0
 inactive_end_not_complete_2: is_active_2 = 0 -> is_end_complete_2 = 0
 active_end_val_2: is_active_2 = 1 -> - start_2 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 - 12 hold_2_12
   - 13 hold_2_13 - 14 hold_2_14 - 15 hold_2_15 - 16 hold_2_16
   - 17 hold_2_17 - 18 hold_2_18 - 19 hold_2_19 - 20 hold_2_20
   - 21 hold_2_21 - 22 hold_2_22 - 23 hold_2_23 - 24 hold_2_24
   - 25 hold_2_25 - 26 hold_2_26 + val_end_2 = 0
 inactive_end_zero_2: is_active_2 = 0 -> val_end_2 = 0
 inactive_end_not_complete_3: is_active_3 = 0 -> is_end_complete_3 = 0
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
 inactive_end_not_complete_15: is_active_15 = 0 -> is_end_complete_15 = 0
 inactive_end_not_complete_16: is_active_16 = 0 -> is_end_complete_16 = 0
 inactive_end_not_complete_17: is_active_17 = 0 -> is_end_complete_17 = 0
 active_end_val_14: is_active_14 = 1 -> - start_14 - hold_14_1
   - 2 hold_14_2 - 3 hold_14_3 - 4 hold_14_4 - 5 hold_14_5 - 6 hold_14_6
   - 7 hold_14_7 - 8 hold_14_8 - 9 hold_14_9 - 10 hold_14_10
   - 11 hold_14_11 - 12 hold_14_12 - 13 hold_14_13 - 14 hold_14_14
   - 15 hold_14_15 - 16 hold_14_16 - 17 hold_14_17 - 18 hold_14_18
   - 19 hold_14_19 - 20 hold_14_20 - 21 hold_14_21 - 22 hold_14_22
   - 23 hold_14_23 - 24 hold_14_24 - 25 hold_14_25 - 26 hold_14_26
   + val_end_14 = 0
 inactive_end_zero_14: is_active_14 = 0 -> val_end_14 = 0
 active_end_val_15: is_active_15 = 1 -> - start_15 - hold_15_1
   - 2 hold_15_2 - 3 hold_15_3 - 4 hold_15_4 - 5 hold_15_5 - 6 hold_15_6
   - 7 hold_15_7 - 8 hold_15_8 - 9 hold_15_9 - 10 hold_15_10
   - 11 hold_15_11 - 12 hold_15_12 - 13 hold_15_13 - 14 hold_15_14
   - 15 hold_15_15 - 16 hold_15_16 - 17 hold_15_17 - 18 hold_15_18
   - 19 hold_15_19 - 20 hold_15_20 - 21 hold_15_21 - 22 hold_15_22
   - 23 hold_15_23 - 24 hold_15_24 - 25 hold_15_25 - 26 hold_15_26
   + val_end_15 = 0
 inactive_end_zero_15: is_active_15 = 0 -> val_end_15 = 0
 active_end_val_16: is_active_16 = 1 -> - start_16 - hold_16_1
   - 2 hold_16_2 - 3 hold_16_3 - 4 hold_16_4 - 5 hold_16_5 - 6 hold_16_6
   - 7 hold_16_7 - 8 hold_16_8 - 9 hold_16_9 - 10 hold_16_10
   - 11 hold_16_11 - 12 hold_16_12 - 13 hold_16_13 - 14 hold_16_14
   - 15 hold_16_15 - 16 hold_16_16 - 17 hold_16_17 - 18 hold_16_18
   - 19 hold_16_19 - 20 hold_16_20 - 21 hold_16_21 - 22 hold_16_22
   - 23 hold_16_23 - 24 hold_16_24 - 25 hold_16_25 - 26 hold_16_26
   + val_end_16 = 0
 inactive_end_zero_16: is_active_16 = 0 -> val_end_16 = 0
 active_end_val_17: is_active_17 = 1 -> - start_17 - hold_17_1
   - 2 hold_17_2 - 3 hold_17_3 - 4 hold_17_4 - 5 hold_17_5 - 6 hold_17_6
   - 7 hold_17_7 - 8 hold_17_8 - 9 hold_17_9 - 10 hold_17_10
   - 11 hold_17_11 - 12 hold_17_12 - 13 hold_17_13 - 14 hold_17_14
   - 15 hold_17_15 - 16 hold_17_16 - 17 hold_17_17 - 18 hold_17_18
   - 19 hold_17_19 - 20 hold_17_20 - 21 hold_17_21 - 22 hold_17_22
   - 23 hold_17_23 - 24 hold_17_24 - 25 hold_17_25 - 26 hold_17_26
   + val_end_17 = 0
 inactive_end_zero_17: is_active_17 = 0 -> val_end_17 = 0
 tp_off_limit_constr_0_23: group_tp_active_0 = 0 -> hold_0_23 = 0
 tp_off_limit_constr_0_24: group_tp_active_0 = 0 -> hold_0_24 = 0
 tp_off_limit_constr_0_25: group_tp_active_0 = 0 -> hold_0_25 = 0
 tp_off_limit_constr_0_26: group_tp_active_0 = 0 -> hold_0_26 = 0
 tp_off_limit_constr_1_12: group_tp_active_1 = 0 -> hold_1_12 = 0
 tp_off_limit_constr_1_13: group_tp_active_1 = 0 -> hold_1_13 = 0
 tp_off_limit_constr_1_14: group_tp_active_1 = 0 -> hold_1_14 = 0
 tp_off_limit_constr_1_15: group_tp_active_1 = 0 -> hold_1_15 = 0
 tp_off_limit_constr_1_16: group_tp_active_1 = 0 -> hold_1_16 = 0
 tp_off_limit_constr_1_17: group_tp_active_1 = 0 -> hold_1_17 = 0
 tp_off_limit_constr_1_18: group_tp_active_1 = 0 -> hold_1_18 = 0
 tp_off_limit_constr_1_19: group_tp_active_1 = 0 -> hold_1_19 = 0
 tp_off_limit_constr_1_20: group_tp_active_1 = 0 -> hold_1_20 = 0
 tp_off_limit_constr_1_21: group_tp_active_1 = 0 -> hold_1_21 = 0
 tp_off_limit_constr_1_22: group_tp_active_1 = 0 -> hold_1_22 = 0
 tp_off_limit_constr_1_23: group_tp_active_1 = 0 -> hold_1_23 = 0
 tp_off_limit_constr_1_24: group_tp_active_1 = 0 -> hold_1_24 = 0
 tp_off_limit_constr_1_25: group_tp_active_1 = 0 -> hold_1_25 = 0
 tp_off_limit_constr_1_26: group_tp_active_1 = 0 -> hold_1_26 = 0
 tp_off_limit_constr_2_23: group_tp_active_2 = 0 -> hold_2_23 = 0
 tp_off_limit_constr_2_24: group_tp_active_2 = 0 -> hold_2_24 = 0
 tp_off_limit_constr_2_25: group_tp_active_2 = 0 -> hold_2_25 = 0
 tp_off_limit_constr_2_26: group_tp_active_2 = 0 -> hold_2_26 = 0
 tp_off_limit_constr_3_12: group_tp_active_3 = 0 -> hold_3_12 = 0
 tp_off_limit_constr_3_13: group_tp_active_3 = 0 -> hold_3_13 = 0
 tp_off_limit_constr_3_14: group_tp_active_3 = 0 -> hold_3_14 = 0
 tp_off_limit_constr_3_15: group_tp_active_3 = 0 -> hold_3_15 = 0
 tp_off_limit_constr_3_16: group_tp_active_3 = 0 -> hold_3_16 = 0
 tp_off_limit_constr_3_17: group_tp_active_3 = 0 -> hold_3_17 = 0
 tp_off_limit_constr_3_18: group_tp_active_3 = 0 -> hold_3_18 = 0
 tp_off_limit_constr_3_19: group_tp_active_3 = 0 -> hold_3_19 = 0
 tp_off_limit_constr_3_20: group_tp_active_3 = 0 -> hold_3_20 = 0
 tp_off_limit_constr_3_21: group_tp_active_3 = 0 -> hold_3_21 = 0
 tp_off_limit_constr_3_22: group_tp_active_3 = 0 -> hold_3_22 = 0
 tp_off_limit_constr_3_23: group_tp_active_3 = 0 -> hold_3_23 = 0
 tp_off_limit_constr_3_24: group_tp_active_3 = 0 -> hold_3_24 = 0
 tp_off_limit_constr_3_25: group_tp_active_3 = 0 -> hold_3_25 = 0
 tp_off_limit_constr_3_26: group_tp_active_3 = 0 -> hold_3_26 = 0
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
 tp_off_limit_constr_6_25: group_tp_active_6 = 0 -> hold_6_25 = 0
 tp_off_limit_constr_6_26: group_tp_active_6 = 0 -> hold_6_26 = 0
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
 group_synch_start_constr_14_15: group_tp_active_14 = 1 -> start_14
   - start_15 = 0
 group_synch_start_constr_14_16: group_tp_active_14 = 1 -> start_14
   - start_16 = 0
 group_synch_start_constr_14_17: group_tp_active_14 = 1 -> start_14
   - start_17 = 0
 group_synch_hold_constr_14_15_1: group_tp_active_14 = 1 -> hold_14_1
   - hold_15_1 = 0
 group_synch_hold_constr_14_16_1: group_tp_active_14 = 1 -> hold_14_1
   - hold_16_1 = 0
 group_synch_hold_constr_14_17_1: group_tp_active_14 = 1 -> hold_14_1
   - hold_17_1 = 0
 group_synch_hold_constr_14_15_2: group_tp_active_14 = 1 -> hold_14_2
   - hold_15_2 = 0
 group_synch_hold_constr_14_16_2: group_tp_active_14 = 1 -> hold_14_2
   - hold_16_2 = 0
 group_synch_hold_constr_14_17_2: group_tp_active_14 = 1 -> hold_14_2
   - hold_17_2 = 0
 group_synch_hold_constr_14_15_3: group_tp_active_14 = 1 -> hold_14_3
   - hold_15_3 = 0
 group_synch_hold_constr_14_16_3: group_tp_active_14 = 1 -> hold_14_3
   - hold_16_3 = 0
 group_synch_hold_constr_14_17_3: group_tp_active_14 = 1 -> hold_14_3
   - hold_17_3 = 0
 group_synch_hold_constr_14_15_4: group_tp_active_14 = 1 -> hold_14_4
   - hold_15_4 = 0
 group_synch_hold_constr_14_16_4: group_tp_active_14 = 1 -> hold_14_4
   - hold_16_4 = 0
 group_synch_hold_constr_14_17_4: group_tp_active_14 = 1 -> hold_14_4
   - hold_17_4 = 0
 group_synch_hold_constr_14_15_5: group_tp_active_14 = 1 -> hold_14_5
   - hold_15_5 = 0
 group_synch_hold_constr_14_16_5: group_tp_active_14 = 1 -> hold_14_5
   - hold_16_5 = 0
 group_synch_hold_constr_14_17_5: group_tp_active_14 = 1 -> hold_14_5
   - hold_17_5 = 0
 group_synch_hold_constr_14_15_6: group_tp_active_14 = 1 -> hold_14_6
   - hold_15_6 = 0
 group_synch_hold_constr_14_16_6: group_tp_active_14 = 1 -> hold_14_6
   - hold_16_6 = 0
 group_synch_hold_constr_14_17_6: group_tp_active_14 = 1 -> hold_14_6
   - hold_17_6 = 0
 group_synch_hold_constr_14_15_7: group_tp_active_14 = 1 -> hold_14_7
   - hold_15_7 = 0
 group_synch_hold_constr_14_16_7: group_tp_active_14 = 1 -> hold_14_7
   - hold_16_7 = 0
 group_synch_hold_constr_14_17_7: group_tp_active_14 = 1 -> hold_14_7
   - hold_17_7 = 0
 group_synch_hold_constr_14_15_8: group_tp_active_14 = 1 -> hold_14_8
   - hold_15_8 = 0
 group_synch_hold_constr_14_16_8: group_tp_active_14 = 1 -> hold_14_8
   - hold_16_8 = 0
 group_synch_hold_constr_14_17_8: group_tp_active_14 = 1 -> hold_14_8
   - hold_17_8 = 0
 group_synch_hold_constr_14_15_9: group_tp_active_14 = 1 -> hold_14_9
   - hold_15_9 = 0
 group_synch_hold_constr_14_16_9: group_tp_active_14 = 1 -> hold_14_9
   - hold_16_9 = 0
 group_synch_hold_constr_14_17_9: group_tp_active_14 = 1 -> hold_14_9
   - hold_17_9 = 0
 group_synch_hold_constr_14_15_10: group_tp_active_14 = 1 -> hold_14_10
   - hold_15_10 = 0
 group_synch_hold_constr_14_16_10: group_tp_active_14 = 1 -> hold_14_10
   - hold_16_10 = 0
 group_synch_hold_constr_14_17_10: group_tp_active_14 = 1 -> hold_14_10
   - hold_17_10 = 0
 group_synch_hold_constr_14_15_11: group_tp_active_14 = 1 -> hold_14_11
   - hold_15_11 = 0
 group_synch_hold_constr_14_16_11: group_tp_active_14 = 1 -> hold_14_11
   - hold_16_11 = 0
 group_synch_hold_constr_14_17_11: group_tp_active_14 = 1 -> hold_14_11
   - hold_17_11 = 0
 group_synch_hold_constr_14_15_12: group_tp_active_14 = 1 -> hold_14_12
   - hold_15_12 = 0
 group_synch_hold_constr_14_16_12: group_tp_active_14 = 1 -> hold_14_12
   - hold_16_12 = 0
 group_synch_hold_constr_14_17_12: group_tp_active_14 = 1 -> hold_14_12
   - hold_17_12 = 0
 group_synch_hold_constr_14_15_13: group_tp_active_14 = 1 -> hold_14_13
   - hold_15_13 = 0
 group_synch_hold_constr_14_16_13: group_tp_active_14 = 1 -> hold_14_13
   - hold_16_13 = 0
 group_synch_hold_constr_14_17_13: group_tp_active_14 = 1 -> hold_14_13
   - hold_17_13 = 0
 group_synch_hold_constr_14_15_14: group_tp_active_14 = 1 -> hold_14_14
   - hold_15_14 = 0
 group_synch_hold_constr_14_16_14: group_tp_active_14 = 1 -> hold_14_14
   - hold_16_14 = 0
 group_synch_hold_constr_14_17_14: group_tp_active_14 = 1 -> hold_14_14
   - hold_17_14 = 0
 group_synch_hold_constr_14_15_15: group_tp_active_14 = 1 -> hold_14_15
   - hold_15_15 = 0
 group_synch_hold_constr_14_16_15: group_tp_active_14 = 1 -> hold_14_15
   - hold_16_15 = 0
 group_synch_hold_constr_14_17_15: group_tp_active_14 = 1 -> hold_14_15
   - hold_17_15 = 0
 group_synch_hold_constr_14_15_16: group_tp_active_14 = 1 -> hold_14_16
   - hold_15_16 = 0
 group_synch_hold_constr_14_16_16: group_tp_active_14 = 1 -> hold_14_16
   - hold_16_16 = 0
 group_synch_hold_constr_14_17_16: group_tp_active_14 = 1 -> hold_14_16
   - hold_17_16 = 0
 group_synch_hold_constr_14_15_17: group_tp_active_14 = 1 -> hold_14_17
   - hold_15_17 = 0
 group_synch_hold_constr_14_16_17: group_tp_active_14 = 1 -> hold_14_17
   - hold_16_17 = 0
 group_synch_hold_constr_14_17_17: group_tp_active_14 = 1 -> hold_14_17
   - hold_17_17 = 0
 group_synch_hold_constr_14_15_18: group_tp_active_14 = 1 -> hold_14_18
   - hold_15_18 = 0
 group_synch_hold_constr_14_16_18: group_tp_active_14 = 1 -> hold_14_18
   - hold_16_18 = 0
 group_synch_hold_constr_14_17_18: group_tp_active_14 = 1 -> hold_14_18
   - hold_17_18 = 0
 group_synch_hold_constr_14_15_19: group_tp_active_14 = 1 -> hold_14_19
   - hold_15_19 = 0
 group_synch_hold_constr_14_16_19: group_tp_active_14 = 1 -> hold_14_19
   - hold_16_19 = 0
 group_synch_hold_constr_14_17_19: group_tp_active_14 = 1 -> hold_14_19
   - hold_17_19 = 0
 group_synch_hold_constr_14_15_20: group_tp_active_14 = 1 -> hold_14_20
   - hold_15_20 = 0
 group_synch_hold_constr_14_16_20: group_tp_active_14 = 1 -> hold_14_20
   - hold_16_20 = 0
 group_synch_hold_constr_14_17_20: group_tp_active_14 = 1 -> hold_14_20
   - hold_17_20 = 0
 group_synch_hold_constr_14_15_21: group_tp_active_14 = 1 -> hold_14_21
   - hold_15_21 = 0
 group_synch_hold_constr_14_16_21: group_tp_active_14 = 1 -> hold_14_21
   - hold_16_21 = 0
 group_synch_hold_constr_14_17_21: group_tp_active_14 = 1 -> hold_14_21
   - hold_17_21 = 0
 group_synch_hold_constr_14_15_22: group_tp_active_14 = 1 -> hold_14_22
   - hold_15_22 = 0
 group_synch_hold_constr_14_16_22: group_tp_active_14 = 1 -> hold_14_22
   - hold_16_22 = 0
 group_synch_hold_constr_14_17_22: group_tp_active_14 = 1 -> hold_14_22
   - hold_17_22 = 0
 group_synch_hold_constr_14_15_23: group_tp_active_14 = 1 -> hold_14_23
   - hold_15_23 = 0
 group_synch_hold_constr_14_16_23: group_tp_active_14 = 1 -> hold_14_23
   - hold_16_23 = 0
 group_synch_hold_constr_14_17_23: group_tp_active_14 = 1 -> hold_14_23
   - hold_17_23 = 0
 group_synch_hold_constr_14_15_24: group_tp_active_14 = 1 -> hold_14_24
   - hold_15_24 = 0
 group_synch_hold_constr_14_16_24: group_tp_active_14 = 1 -> hold_14_24
   - hold_16_24 = 0
 group_synch_hold_constr_14_17_24: group_tp_active_14 = 1 -> hold_14_24
   - hold_17_24 = 0
 group_synch_hold_constr_14_15_25: group_tp_active_14 = 1 -> hold_14_25
   - hold_15_25 = 0
 group_synch_hold_constr_14_16_25: group_tp_active_14 = 1 -> hold_14_25
   - hold_16_25 = 0
 group_synch_hold_constr_14_17_25: group_tp_active_14 = 1 -> hold_14_25
   - hold_17_25 = 0
 group_synch_hold_constr_14_15_26: group_tp_active_14 = 1 -> hold_14_26
   - hold_15_26 = 0
 group_synch_hold_constr_14_16_26: group_tp_active_14 = 1 -> hold_14_26
   - hold_16_26 = 0
 group_synch_hold_constr_14_17_26: group_tp_active_14 = 1 -> hold_14_26
   - hold_17_26 = 0
 tp_on_limit_constr_14_17: group_tp_active_14 = 1 -> hold_14_17 = 0
 tp_on_limit_constr_14_18: group_tp_active_14 = 1 -> hold_14_18 = 0
 tp_on_limit_constr_14_19: group_tp_active_14 = 1 -> hold_14_19 = 0
 tp_on_limit_constr_14_20: group_tp_active_14 = 1 -> hold_14_20 = 0
 tp_on_limit_constr_14_21: group_tp_active_14 = 1 -> hold_14_21 = 0
 tp_on_limit_constr_14_22: group_tp_active_14 = 1 -> hold_14_22 = 0
 tp_on_limit_constr_14_23: group_tp_active_14 = 1 -> hold_14_23 = 0
 tp_on_limit_constr_14_24: group_tp_active_14 = 1 -> hold_14_24 = 0
 tp_on_limit_constr_14_25: group_tp_active_14 = 1 -> hold_14_25 = 0
 tp_on_limit_constr_14_26: group_tp_active_14 = 1 -> hold_14_26 = 0
 tp_on_limit_constr_15_17: group_tp_active_14 = 1 -> hold_15_17 = 0
 tp_on_limit_constr_15_18: group_tp_active_14 = 1 -> hold_15_18 = 0
 tp_on_limit_constr_15_19: group_tp_active_14 = 1 -> hold_15_19 = 0
 tp_on_limit_constr_15_20: group_tp_active_14 = 1 -> hold_15_20 = 0
 tp_on_limit_constr_15_21: group_tp_active_14 = 1 -> hold_15_21 = 0
 tp_on_limit_constr_15_22: group_tp_active_14 = 1 -> hold_15_22 = 0
 tp_on_limit_constr_15_23: group_tp_active_14 = 1 -> hold_15_23 = 0
 tp_on_limit_constr_15_24: group_tp_active_14 = 1 -> hold_15_24 = 0
 tp_on_limit_constr_15_25: group_tp_active_14 = 1 -> hold_15_25 = 0
 tp_on_limit_constr_15_26: group_tp_active_14 = 1 -> hold_15_26 = 0
 tp_on_limit_constr_16_17: group_tp_active_14 = 1 -> hold_16_17 = 0
 tp_on_limit_constr_16_18: group_tp_active_14 = 1 -> hold_16_18 = 0
 tp_on_limit_constr_16_19: group_tp_active_14 = 1 -> hold_16_19 = 0
 tp_on_limit_constr_16_20: group_tp_active_14 = 1 -> hold_16_20 = 0
 tp_on_limit_constr_16_21: group_tp_active_14 = 1 -> hold_16_21 = 0
 tp_on_limit_constr_16_22: group_tp_active_14 = 1 -> hold_16_22 = 0
 tp_on_limit_constr_16_23: group_tp_active_14 = 1 -> hold_16_23 = 0
 tp_on_limit_constr_16_24: group_tp_active_14 = 1 -> hold_16_24 = 0
 tp_on_limit_constr_16_25: group_tp_active_14 = 1 -> hold_16_25 = 0
 tp_on_limit_constr_16_26: group_tp_active_14 = 1 -> hold_16_26 = 0
 tp_on_limit_constr_17_17: group_tp_active_14 = 1 -> hold_17_17 = 0
 tp_on_limit_constr_17_18: group_tp_active_14 = 1 -> hold_17_18 = 0
 tp_on_limit_constr_17_19: group_tp_active_14 = 1 -> hold_17_19 = 0
 tp_on_limit_constr_17_20: group_tp_active_14 = 1 -> hold_17_20 = 0
 tp_on_limit_constr_17_21: group_tp_active_14 = 1 -> hold_17_21 = 0
 tp_on_limit_constr_17_22: group_tp_active_14 = 1 -> hold_17_22 = 0
 tp_on_limit_constr_17_23: group_tp_active_14 = 1 -> hold_17_23 = 0
 tp_on_limit_constr_17_24: group_tp_active_14 = 1 -> hold_17_24 = 0
 tp_on_limit_constr_17_25: group_tp_active_14 = 1 -> hold_17_25 = 0
 tp_on_limit_constr_17_26: group_tp_active_14 = 1 -> hold_17_26 = 0
 tp_off_limit_constr_14_1: group_tp_active_14 = 0 -> hold_14_1 = 0
 tp_off_limit_constr_14_2: group_tp_active_14 = 0 -> hold_14_2 = 0
 tp_off_limit_constr_14_3: group_tp_active_14 = 0 -> hold_14_3 = 0
 tp_off_limit_constr_14_4: group_tp_active_14 = 0 -> hold_14_4 = 0
 tp_off_limit_constr_14_5: group_tp_active_14 = 0 -> hold_14_5 = 0
 tp_off_limit_constr_14_6: group_tp_active_14 = 0 -> hold_14_6 = 0
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
 tp_off_limit_constr_15_1: group_tp_active_14 = 0 -> hold_15_1 = 0
 tp_off_limit_constr_15_2: group_tp_active_14 = 0 -> hold_15_2 = 0
 tp_off_limit_constr_15_3: group_tp_active_14 = 0 -> hold_15_3 = 0
 tp_off_limit_constr_15_4: group_tp_active_14 = 0 -> hold_15_4 = 0
 tp_off_limit_constr_15_5: group_tp_active_14 = 0 -> hold_15_5 = 0
 tp_off_limit_constr_15_6: group_tp_active_14 = 0 -> hold_15_6 = 0
 tp_off_limit_constr_15_7: group_tp_active_14 = 0 -> hold_15_7 = 0
 tp_off_limit_constr_15_8: group_tp_active_14 = 0 -> hold_15_8 = 0
 tp_off_limit_constr_15_9: group_tp_active_14 = 0 -> hold_15_9 = 0
 tp_off_limit_constr_15_10: group_tp_active_14 = 0 -> hold_15_10 = 0
 tp_off_limit_constr_15_11: group_tp_active_14 = 0 -> hold_15_11 = 0
 tp_off_limit_constr_15_12: group_tp_active_14 = 0 -> hold_15_12 = 0
 tp_off_limit_constr_15_13: group_tp_active_14 = 0 -> hold_15_13 = 0
 tp_off_limit_constr_15_14: group_tp_active_14 = 0 -> hold_15_14 = 0
 tp_off_limit_constr_15_15: group_tp_active_14 = 0 -> hold_15_15 = 0
 tp_off_limit_constr_15_16: group_tp_active_14 = 0 -> hold_15_16 = 0
 tp_off_limit_constr_15_17: group_tp_active_14 = 0 -> hold_15_17 = 0
 tp_off_limit_constr_15_18: group_tp_active_14 = 0 -> hold_15_18 = 0
 tp_off_limit_constr_15_19: group_tp_active_14 = 0 -> hold_15_19 = 0
 tp_off_limit_constr_15_20: group_tp_active_14 = 0 -> hold_15_20 = 0
 tp_off_limit_constr_15_21: group_tp_active_14 = 0 -> hold_15_21 = 0
 tp_off_limit_constr_15_22: group_tp_active_14 = 0 -> hold_15_22 = 0
 tp_off_limit_constr_15_23: group_tp_active_14 = 0 -> hold_15_23 = 0
 tp_off_limit_constr_15_24: group_tp_active_14 = 0 -> hold_15_24 = 0
 tp_off_limit_constr_15_25: group_tp_active_14 = 0 -> hold_15_25 = 0
 tp_off_limit_constr_15_26: group_tp_active_14 = 0 -> hold_15_26 = 0
 tp_off_limit_constr_16_1: group_tp_active_14 = 0 -> hold_16_1 = 0
 tp_off_limit_constr_16_2: group_tp_active_14 = 0 -> hold_16_2 = 0
 tp_off_limit_constr_16_3: group_tp_active_14 = 0 -> hold_16_3 = 0
 tp_off_limit_constr_16_4: group_tp_active_14 = 0 -> hold_16_4 = 0
 tp_off_limit_constr_16_5: group_tp_active_14 = 0 -> hold_16_5 = 0
 tp_off_limit_constr_16_6: group_tp_active_14 = 0 -> hold_16_6 = 0
 tp_off_limit_constr_16_7: group_tp_active_14 = 0 -> hold_16_7 = 0
 tp_off_limit_constr_16_8: group_tp_active_14 = 0 -> hold_16_8 = 0
 tp_off_limit_constr_16_9: group_tp_active_14 = 0 -> hold_16_9 = 0
 tp_off_limit_constr_16_10: group_tp_active_14 = 0 -> hold_16_10 = 0
 tp_off_limit_constr_16_11: group_tp_active_14 = 0 -> hold_16_11 = 0
 tp_off_limit_constr_16_12: group_tp_active_14 = 0 -> hold_16_12 = 0
 tp_off_limit_constr_16_13: group_tp_active_14 = 0 -> hold_16_13 = 0
 tp_off_limit_constr_16_14: group_tp_active_14 = 0 -> hold_16_14 = 0
 tp_off_limit_constr_16_15: group_tp_active_14 = 0 -> hold_16_15 = 0
 tp_off_limit_constr_16_16: group_tp_active_14 = 0 -> hold_16_16 = 0
 tp_off_limit_constr_16_17: group_tp_active_14 = 0 -> hold_16_17 = 0
 tp_off_limit_constr_16_18: group_tp_active_14 = 0 -> hold_16_18 = 0
 tp_off_limit_constr_16_19: group_tp_active_14 = 0 -> hold_16_19 = 0
 tp_off_limit_constr_16_20: group_tp_active_14 = 0 -> hold_16_20 = 0
 tp_off_limit_constr_16_21: group_tp_active_14 = 0 -> hold_16_21 = 0
 tp_off_limit_constr_16_22: group_tp_active_14 = 0 -> hold_16_22 = 0
 tp_off_limit_constr_16_23: group_tp_active_14 = 0 -> hold_16_23 = 0
 tp_off_limit_constr_16_24: group_tp_active_14 = 0 -> hold_16_24 = 0
 tp_off_limit_constr_16_25: group_tp_active_14 = 0 -> hold_16_25 = 0
 tp_off_limit_constr_16_26: group_tp_active_14 = 0 -> hold_16_26 = 0
 tp_off_limit_constr_17_1: group_tp_active_14 = 0 -> hold_17_1 = 0
 tp_off_limit_constr_17_2: group_tp_active_14 = 0 -> hold_17_2 = 0
 tp_off_limit_constr_17_3: group_tp_active_14 = 0 -> hold_17_3 = 0
 tp_off_limit_constr_17_4: group_tp_active_14 = 0 -> hold_17_4 = 0
 tp_off_limit_constr_17_5: group_tp_active_14 = 0 -> hold_17_5 = 0
 tp_off_limit_constr_17_6: group_tp_active_14 = 0 -> hold_17_6 = 0
 tp_off_limit_constr_17_7: group_tp_active_14 = 0 -> hold_17_7 = 0
 tp_off_limit_constr_17_8: group_tp_active_14 = 0 -> hold_17_8 = 0
 tp_off_limit_constr_17_9: group_tp_active_14 = 0 -> hold_17_9 = 0
 tp_off_limit_constr_17_10: group_tp_active_14 = 0 -> hold_17_10 = 0
 tp_off_limit_constr_17_11: group_tp_active_14 = 0 -> hold_17_11 = 0
 tp_off_limit_constr_17_12: group_tp_active_14 = 0 -> hold_17_12 = 0
 tp_off_limit_constr_17_13: group_tp_active_14 = 0 -> hold_17_13 = 0
 tp_off_limit_constr_17_14: group_tp_active_14 = 0 -> hold_17_14 = 0
 tp_off_limit_constr_17_15: group_tp_active_14 = 0 -> hold_17_15 = 0
 tp_off_limit_constr_17_16: group_tp_active_14 = 0 -> hold_17_16 = 0
 tp_off_limit_constr_17_17: group_tp_active_14 = 0 -> hold_17_17 = 0
 tp_off_limit_constr_17_18: group_tp_active_14 = 0 -> hold_17_18 = 0
 tp_off_limit_constr_17_19: group_tp_active_14 = 0 -> hold_17_19 = 0
 tp_off_limit_constr_17_20: group_tp_active_14 = 0 -> hold_17_20 = 0
 tp_off_limit_constr_17_21: group_tp_active_14 = 0 -> hold_17_21 = 0
 tp_off_limit_constr_17_22: group_tp_active_14 = 0 -> hold_17_22 = 0
 tp_off_limit_constr_17_23: group_tp_active_14 = 0 -> hold_17_23 = 0
 tp_off_limit_constr_17_24: group_tp_active_14 = 0 -> hold_17_24 = 0
 tp_off_limit_constr_17_25: group_tp_active_14 = 0 -> hold_17_25 = 0
 tp_off_limit_constr_17_26: group_tp_active_14 = 0 -> hold_17_26 = 0
 group_throughput_constr_0: group_tp_active_0 = 1 ->
   - 15314.28571428572 hold_0_1 - 7657.142857142858 hold_0_2
   - 5104.761904761905 hold_0_3 - 3828.571428571429 hold_0_4
   - 3062.857142857143 hold_0_5 - 2552.380952380952 hold_0_6
   - 2187.755102040816 hold_0_7 - 1914.285714285714 hold_0_8
   - 1701.587301587301 hold_0_9 - 1531.428571428571 hold_0_10
   - 1392.207792207792 hold_0_11 - 1276.190476190476 hold_0_12
   - 1178.021978021978 hold_0_13 - 1093.877551020408 hold_0_14
   - 1020.952380952381 hold_0_15 - 957.1428571428572 hold_0_16
   - 900.8403361344537 hold_0_17 - 850.7936507936507 hold_0_18
   - 806.015037593985 hold_0_19 - 765.7142857142857 hold_0_20
   - 729.2517006802721 hold_0_21 - 696.1038961038961 hold_0_22
   + flow_nic_in_0_0 <= 0
 group_throughput_constr_1: group_tp_active_1 = 1 ->
   - 7108.571428571429 hold_1_1 - 3554.285714285715 hold_1_2
   - 2369.52380952381 hold_1_3 - 1777.142857142857 hold_1_4
   - 1421.714285714286 hold_1_5 - 1184.761904761905 hold_1_6
   - 1015.510204081633 hold_1_7 - 888.5714285714287 hold_1_8
   - 789.8412698412698 hold_1_9 - 710.8571428571429 hold_1_10
   - 646.2337662337662 hold_1_11 + flow_nic_in_1_1 <= 0
 group_throughput_constr_2: group_tp_active_2 = 1 ->
   - 9325.714285714286 hold_2_1 - 4662.857142857143 hold_2_2
   - 3108.571428571428 hold_2_3 - 2331.428571428572 hold_2_4
   - 1865.142857142857 hold_2_5 - 1554.285714285714 hold_2_6
   - 1332.244897959184 hold_2_7 - 1165.714285714286 hold_2_8
   - 1036.190476190476 hold_2_9 - 932.5714285714284 hold_2_10
   - 847.7922077922078 hold_2_11 - 777.1428571428571 hold_2_12
   - 717.3626373626373 hold_2_13 - 666.1224489795918 hold_2_14
   - 621.7142857142858 hold_2_15 - 582.8571428571429 hold_2_16
   - 548.5714285714286 hold_2_17 - 518.0952380952381 hold_2_18
   - 490.8270676691729 hold_2_19 - 466.2857142857142 hold_2_20
   - 444.0816326530613 hold_2_21 - 423.8961038961039 hold_2_22
   + flow_nic_in_2_2 <= 0
 group_throughput_constr_3: group_tp_active_3 = 1 ->
   - 7108.571428571429 hold_3_1 - 3554.285714285715 hold_3_2
   - 2369.52380952381 hold_3_3 - 1777.142857142857 hold_3_4
   - 1421.714285714286 hold_3_5 - 1184.761904761905 hold_3_6
   - 1015.510204081633 hold_3_7 - 888.5714285714287 hold_3_8
   - 789.8412698412698 hold_3_9 - 710.8571428571429 hold_3_10
   - 646.2337662337662 hold_3_11 + flow_nic_in_3_3 <= 0
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
   - 3949.714285714286 hold_6_1 - 1974.857142857143 hold_6_2
   - 1316.571428571428 hold_6_3 - 987.4285714285714 hold_6_4
   - 789.9428571428572 hold_6_5 - 658.2857142857142 hold_6_6
   - 564.2448979591836 hold_6_7 - 493.7142857142857 hold_6_8
   - 438.8571428571428 hold_6_9 - 394.9714285714286 hold_6_10
   - 359.0649350649351 hold_6_11 - 329.1428571428571 hold_6_12
   - 303.8241758241758 hold_6_13 + flow_nic_in_6_6 <= 0
 group_throughput_constr_7: group_tp_active_7 = 1 ->
   - 3949.714285714286 hold_7_1 - 1974.857142857143 hold_7_2
   - 1316.571428571428 hold_7_3 - 987.4285714285714 hold_7_4
   - 789.9428571428572 hold_7_5 - 658.2857142857142 hold_7_6
   - 564.2448979591836 hold_7_7 - 493.7142857142857 hold_7_8
   - 438.8571428571428 hold_7_9 - 394.9714285714286 hold_7_10
   - 359.0649350649351 hold_7_11 - 329.1428571428571 hold_7_12
   - 303.8241758241758 hold_7_13 + flow_nic_in_7_7 <= 0
 group_throughput_constr_8: group_tp_active_8 = 1 ->
   - 7506.63764007438 hold_8_1 - 3753.31882003719 hold_8_2
   - 2502.21254669146 hold_8_3 - 1876.659410018595 hold_8_4
   - 1501.327528014876 hold_8_5 - 1251.10627334573 hold_8_6
   - 1072.376805724911 hold_8_7 - 938.3297050092975 hold_8_8
   - 834.0708488971533 hold_8_9 - 750.663764007438 hold_8_10
   - 682.4216036431255 hold_8_11 - 625.553136672865 hold_8_12
   - 577.4336646211061 hold_8_13 - 536.1884028624556 hold_8_14
   - 500.442509338292 hold_8_15 - 469.1648525046488 hold_8_16
   - 441.5669200043752 hold_8_17 - 417.0354244485766 hold_8_18
   - 395.0861915828621 hold_8_19 - 375.331882003719 hold_8_20
   - 357.4589352416371 hold_8_21 - 341.2108018215628 hold_8_22
   - 326.3755495684513 hold_8_23 - 312.7765683364325 hold_8_24
   - 300.2655056029752 hold_8_25 - 288.7168323105531 hold_8_26
   + flow_nic_in_8_8 + flow_nic_in_8_9 <= 0
 group_throughput_constr_10: group_tp_active_10 = 1 ->
   - 8074.06386378554 hold_10_1 - 4037.03193189277 hold_10_2
   - 2691.354621261847 hold_10_3 - 2018.515965946385 hold_10_4
   - 1614.812772757108 hold_10_5 - 1345.677310630924 hold_10_6
   - 1153.437694826506 hold_10_7 - 1009.257982973192 hold_10_8
   - 897.1182070872823 hold_10_9 - 807.4063863785541 hold_10_10
   - 734.0058057986855 hold_10_11 - 672.8386553154618 hold_10_12
   + flow_nic_in_10_10 + flow_nic_in_10_11 <= 0
 group_throughput_constr_12: group_tp_active_12 = 1 ->
   - 2693.912296991259 hold_12_1 - 1346.95614849563 hold_12_2
   - 897.970765663753 hold_12_3 - 673.4780742478148 hold_12_4
   - 538.7824593982518 hold_12_5 - 448.9853828318765 hold_12_6
   - 384.8446138558941 hold_12_7 - 336.7390371239074 hold_12_8
   - 299.3235885545843 hold_12_9 - 269.3912296991259 hold_12_10
   - 244.9011179082963 hold_12_11 - 224.4926914159383 hold_12_12
   + flow_nic_in_12_12 + flow_nic_in_12_13 <= 0
 group_throughput_constr_14: group_tp_active_14 = 1 ->
   - 7621.584517894289 hold_14_1 - 3810.792258947145 hold_14_2
   - 2540.52817263143 hold_14_3 - 1905.396129473572 hold_14_4
   - 1524.316903578858 hold_14_5 - 1270.264086315715 hold_14_6
   - 1088.797788270613 hold_14_7 - 952.6980647367861 hold_14_8
   - 846.8427242104766 hold_14_9 - 762.158451789429 hold_14_10
   - 692.8713198085718 hold_14_11 - 635.1320431578575 hold_14_12
   - 586.2757321457145 hold_14_13 - 544.3988941353064 hold_14_14
   - 508.1056345262859 hold_14_15 - 476.3490323683931 hold_14_16
   + flow_nic_in_14_14 + flow_nic_in_14_15 + flow_nic_in_14_16
   + flow_nic_in_14_17 <= 0
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
   - 5109.459042174185 hold_8_1 - 3040.15247603248 hold_8_2
   - 2163.816929787807 hold_8_3 - 1679.651095409346 hold_8_4
   - 1372.538192304928 hold_8_5 - 1160.371981735487 hold_8_6
   - 1005.016998327084 hold_8_7 - 886.3492175210624 hold_8_8
   - 792.7454804093763 hold_8_9 - 717.0235396836346 hold_8_10
   - 654.5060374151892 hold_8_11 - 602.0160715548276 hold_8_12
   - 557.3201993053406 hold_8_13 - 518.8024130345685 hold_8_14
   - 485.2645706247653 hold_8_15 - 455.7995324779789 hold_8_16
   - 429.7078711806942 hold_8_17 - 406.441639349837 hold_8_18
   - 385.5654670831283 hold_8_19 - 366.7290626737931 hold_8_20
   - 349.6473985665369 hold_8_21 - 334.0861883096475 hold_8_22
   - 319.8510763911254 hold_8_23 - 306.7794787980271 hold_8_24
   - 294.7343457685529 hold_8_25 - 283.5993383963553 hold_8_26
   + flow_nic_in_8_8 + flow_nic_in_8_9 <= 0
 intergroup_group_tp_10_via_nic_out_0_nic_in_10: tp_on_and_switch_nic_out_0_nic_in_10_10 = 1 ->
   - 5366.149336128531 hold_10_1 - 3223.656633868661 hold_10_2
   - 2303.828417615377 hold_10_3 - 1792.392643000101 hold_10_4
   - 1466.777120905479 hold_10_5 - 1241.279690871386 hold_10_6
   - 1075.877817936759 hold_10_7 - 949.3728499935663 hold_10_8
   - 849.4875361276669 hold_10_9 - 768.6196123946033 hold_10_10
   - 701.8100166255107 hold_10_11 - 645.6859991034142 hold_10_12
   + flow_nic_in_10_10 + flow_nic_in_10_11 <= 0
 intergroup_group_tp_12_via_nic_out_0_nic_in_12: tp_on_and_switch_nic_out_0_nic_in_12_12 = 1 ->
   - 2305.702309237719 hold_12_1 - 1242.367720967523 hold_12_2
   - 850.2519296467542 hold_12_3 - 646.2748288017978 hold_12_4
   - 521.2305906758797 hold_12_5 - 436.730044930787 hold_12_6
   - 375.8054450200391 hold_12_7 - 329.7980448692438 hold_12_8
   - 293.8267585678413 hold_12_9 - 264.9306058433094 hold_12_10
   - 241.2090943547263 hold_12_11 - 221.3864637231713 hold_12_12
   + flow_nic_in_12_12 + flow_nic_in_12_13 <= 0
 intergroup_group_tp_14_via_nic_out_0_nic_in_14: tp_on_and_switch_nic_out_0_nic_in_14_14 = 1 ->
   - 5162.454372776313 hold_14_1 - 3077.750518312423 hold_14_2
   - 2192.410614391559 hold_14_3 - 1702.634102654364 hold_14_4
   - 1391.727311909141 hold_14_5 - 1176.833502919945 hold_14_6
   - 1019.425990533228 hold_14_7 - 899.1588818239977 hold_14_8
   - 804.2743562801687 hold_14_9 - 727.5038750948597 hold_14_10
   - 664.112297072705 hold_14_11 - 610.8825985968332 hold_14_12
   - 565.5526210836673 hold_14_13 - 526.4852692383147 hold_14_14
   - 492.4665695994538 hold_14_15 - 462.5772677503618 hold_14_16
   + flow_nic_in_14_14 + flow_nic_in_14_15 + flow_nic_in_14_16
   + flow_nic_in_14_17 <= 0
 intergroup_group_tp_4_via_nic_out_1_nic_in_4: tp_on_and_switch_nic_out_1_nic_in_4_4 = 1 ->
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
 intergroup_group_tp_8_via_nic_out_1_nic_in_8: tp_on_and_switch_nic_out_1_nic_in_8_8 = 1 ->
   - 5109.459042174185 hold_8_1 - 3040.15247603248 hold_8_2
   - 2163.816929787807 hold_8_3 - 1679.651095409346 hold_8_4
   - 1372.538192304928 hold_8_5 - 1160.371981735487 hold_8_6
   - 1005.016998327084 hold_8_7 - 886.3492175210624 hold_8_8
   - 792.7454804093763 hold_8_9 - 717.0235396836346 hold_8_10
   - 654.5060374151892 hold_8_11 - 602.0160715548276 hold_8_12
   - 557.3201993053406 hold_8_13 - 518.8024130345685 hold_8_14
   - 485.2645706247653 hold_8_15 - 455.7995324779789 hold_8_16
   - 429.7078711806942 hold_8_17 - 406.441639349837 hold_8_18
   - 385.5654670831283 hold_8_19 - 366.7290626737931 hold_8_20
   - 349.6473985665369 hold_8_21 - 334.0861883096475 hold_8_22
   - 319.8510763911254 hold_8_23 - 306.7794787980271 hold_8_24
   - 294.7343457685529 hold_8_25 - 283.5993383963553 hold_8_26
   + flow_nic_in_8_8 + flow_nic_in_8_9 <= 0
 intergroup_group_tp_10_via_nic_out_1_nic_in_10: tp_on_and_switch_nic_out_1_nic_in_10_10 = 1 ->
   - 5366.149336128531 hold_10_1 - 3223.656633868661 hold_10_2
   - 2303.828417615377 hold_10_3 - 1792.392643000101 hold_10_4
   - 1466.777120905479 hold_10_5 - 1241.279690871386 hold_10_6
   - 1075.877817936759 hold_10_7 - 949.3728499935663 hold_10_8
   - 849.4875361276669 hold_10_9 - 768.6196123946033 hold_10_10
   - 701.8100166255107 hold_10_11 - 645.6859991034142 hold_10_12
   + flow_nic_in_10_10 + flow_nic_in_10_11 <= 0
 intergroup_group_tp_12_via_nic_out_1_nic_in_12: tp_on_and_switch_nic_out_1_nic_in_12_12 = 1 ->
   - 2305.702309237719 hold_12_1 - 1242.367720967523 hold_12_2
   - 850.2519296467542 hold_12_3 - 646.2748288017978 hold_12_4
   - 521.2305906758797 hold_12_5 - 436.730044930787 hold_12_6
   - 375.8054450200391 hold_12_7 - 329.7980448692438 hold_12_8
   - 293.8267585678413 hold_12_9 - 264.9306058433094 hold_12_10
   - 241.2090943547263 hold_12_11 - 221.3864637231713 hold_12_12
   + flow_nic_in_12_12 + flow_nic_in_12_13 <= 0
 intergroup_group_tp_14_via_nic_out_1_nic_in_14: tp_on_and_switch_nic_out_1_nic_in_14_14 = 1 ->
   - 5162.454372776313 hold_14_1 - 3077.750518312423 hold_14_2
   - 2192.410614391559 hold_14_3 - 1702.634102654364 hold_14_4
   - 1391.727311909141 hold_14_5 - 1176.833502919945 hold_14_6
   - 1019.425990533228 hold_14_7 - 899.1588818239977 hold_14_8
   - 804.2743562801687 hold_14_9 - 727.5038750948597 hold_14_10
   - 664.112297072705 hold_14_11 - 610.8825985968332 hold_14_12
   - 565.5526210836673 hold_14_13 - 526.4852692383147 hold_14_14
   - 492.4665695994538 hold_14_15 - 462.5772677503618 hold_14_16
   + flow_nic_in_14_14 + flow_nic_in_14_15 + flow_nic_in_14_16
   + flow_nic_in_14_17 <= 0
 intergroup_group_tp_4_via_nic_out_2_nic_in_4: tp_on_and_switch_nic_out_2_nic_in_4_4 = 1 ->
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
 intergroup_group_tp_8_via_nic_out_2_nic_in_8: tp_on_and_switch_nic_out_2_nic_in_8_8 = 1 ->
   - 5109.459042174185 hold_8_1 - 3040.15247603248 hold_8_2
   - 2163.816929787807 hold_8_3 - 1679.651095409346 hold_8_4
   - 1372.538192304928 hold_8_5 - 1160.371981735487 hold_8_6
   - 1005.016998327084 hold_8_7 - 886.3492175210624 hold_8_8
   - 792.7454804093763 hold_8_9 - 717.0235396836346 hold_8_10
   - 654.5060374151892 hold_8_11 - 602.0160715548276 hold_8_12
   - 557.3201993053406 hold_8_13 - 518.8024130345685 hold_8_14
   - 485.2645706247653 hold_8_15 - 455.7995324779789 hold_8_16
   - 429.7078711806942 hold_8_17 - 406.441639349837 hold_8_18
   - 385.5654670831283 hold_8_19 - 366.7290626737931 hold_8_20
   - 349.6473985665369 hold_8_21 - 334.0861883096475 hold_8_22
   - 319.8510763911254 hold_8_23 - 306.7794787980271 hold_8_24
   - 294.7343457685529 hold_8_25 - 283.5993383963553 hold_8_26
   + flow_nic_in_8_8 + flow_nic_in_8_9 <= 0
 intergroup_group_tp_10_via_nic_out_2_nic_in_10: tp_on_and_switch_nic_out_2_nic_in_10_10 = 1 ->
   - 5366.149336128531 hold_10_1 - 3223.656633868661 hold_10_2
   - 2303.828417615377 hold_10_3 - 1792.392643000101 hold_10_4
   - 1466.777120905479 hold_10_5 - 1241.279690871386 hold_10_6
   - 1075.877817936759 hold_10_7 - 949.3728499935663 hold_10_8
   - 849.4875361276669 hold_10_9 - 768.6196123946033 hold_10_10
   - 701.8100166255107 hold_10_11 - 645.6859991034142 hold_10_12
   + flow_nic_in_10_10 + flow_nic_in_10_11 <= 0
 intergroup_group_tp_12_via_nic_out_2_nic_in_12: tp_on_and_switch_nic_out_2_nic_in_12_12 = 1 ->
   - 2305.702309237719 hold_12_1 - 1242.367720967523 hold_12_2
   - 850.2519296467542 hold_12_3 - 646.2748288017978 hold_12_4
   - 521.2305906758797 hold_12_5 - 436.730044930787 hold_12_6
   - 375.8054450200391 hold_12_7 - 329.7980448692438 hold_12_8
   - 293.8267585678413 hold_12_9 - 264.9306058433094 hold_12_10
   - 241.2090943547263 hold_12_11 - 221.3864637231713 hold_12_12
   + flow_nic_in_12_12 + flow_nic_in_12_13 <= 0
 intergroup_group_tp_14_via_nic_out_2_nic_in_14: tp_on_and_switch_nic_out_2_nic_in_14_14 = 1 ->
   - 5162.454372776313 hold_14_1 - 3077.750518312423 hold_14_2
   - 2192.410614391559 hold_14_3 - 1702.634102654364 hold_14_4
   - 1391.727311909141 hold_14_5 - 1176.833502919945 hold_14_6
   - 1019.425990533228 hold_14_7 - 899.1588818239977 hold_14_8
   - 804.2743562801687 hold_14_9 - 727.5038750948597 hold_14_10
   - 664.112297072705 hold_14_11 - 610.8825985968332 hold_14_12
   - 565.5526210836673 hold_14_13 - 526.4852692383147 hold_14_14
   - 492.4665695994538 hold_14_15 - 462.5772677503618 hold_14_16
   + flow_nic_in_14_14 + flow_nic_in_14_15 + flow_nic_in_14_16
   + flow_nic_in_14_17 <= 0
 intergroup_group_tp_4_via_nic_out_3_nic_in_4: tp_on_and_switch_nic_out_3_nic_in_4_4 = 1 ->
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
 intergroup_group_tp_8_via_nic_out_3_nic_in_8: tp_on_and_switch_nic_out_3_nic_in_8_8 = 1 ->
   - 5109.459042174185 hold_8_1 - 3040.15247603248 hold_8_2
   - 2163.816929787807 hold_8_3 - 1679.651095409346 hold_8_4
   - 1372.538192304928 hold_8_5 - 1160.371981735487 hold_8_6
   - 1005.016998327084 hold_8_7 - 886.3492175210624 hold_8_8
   - 792.7454804093763 hold_8_9 - 717.0235396836346 hold_8_10
   - 654.5060374151892 hold_8_11 - 602.0160715548276 hold_8_12
   - 557.3201993053406 hold_8_13 - 518.8024130345685 hold_8_14
   - 485.2645706247653 hold_8_15 - 455.7995324779789 hold_8_16
   - 429.7078711806942 hold_8_17 - 406.441639349837 hold_8_18
   - 385.5654670831283 hold_8_19 - 366.7290626737931 hold_8_20
   - 349.6473985665369 hold_8_21 - 334.0861883096475 hold_8_22
   - 319.8510763911254 hold_8_23 - 306.7794787980271 hold_8_24
   - 294.7343457685529 hold_8_25 - 283.5993383963553 hold_8_26
   + flow_nic_in_8_8 + flow_nic_in_8_9 <= 0
 intergroup_group_tp_10_via_nic_out_3_nic_in_10: tp_on_and_switch_nic_out_3_nic_in_10_10 = 1 ->
   - 5366.149336128531 hold_10_1 - 3223.656633868661 hold_10_2
   - 2303.828417615377 hold_10_3 - 1792.392643000101 hold_10_4
   - 1466.777120905479 hold_10_5 - 1241.279690871386 hold_10_6
   - 1075.877817936759 hold_10_7 - 949.3728499935663 hold_10_8
   - 849.4875361276669 hold_10_9 - 768.6196123946033 hold_10_10
   - 701.8100166255107 hold_10_11 - 645.6859991034142 hold_10_12
   + flow_nic_in_10_10 + flow_nic_in_10_11 <= 0
 intergroup_group_tp_12_via_nic_out_3_nic_in_12: tp_on_and_switch_nic_out_3_nic_in_12_12 = 1 ->
   - 2305.702309237719 hold_12_1 - 1242.367720967523 hold_12_2
   - 850.2519296467542 hold_12_3 - 646.2748288017978 hold_12_4
   - 521.2305906758797 hold_12_5 - 436.730044930787 hold_12_6
   - 375.8054450200391 hold_12_7 - 329.7980448692438 hold_12_8
   - 293.8267585678413 hold_12_9 - 264.9306058433094 hold_12_10
   - 241.2090943547263 hold_12_11 - 221.3864637231713 hold_12_12
   + flow_nic_in_12_12 + flow_nic_in_12_13 <= 0
 intergroup_group_tp_14_via_nic_out_3_nic_in_14: tp_on_and_switch_nic_out_3_nic_in_14_14 = 1 ->
   - 5162.454372776313 hold_14_1 - 3077.750518312423 hold_14_2
   - 2192.410614391559 hold_14_3 - 1702.634102654364 hold_14_4
   - 1391.727311909141 hold_14_5 - 1176.833502919945 hold_14_6
   - 1019.425990533228 hold_14_7 - 899.1588818239977 hold_14_8
   - 804.2743562801687 hold_14_9 - 727.5038750948597 hold_14_10
   - 664.112297072705 hold_14_11 - 610.8825985968332 hold_14_12
   - 565.5526210836673 hold_14_13 - 526.4852692383147 hold_14_14
   - 492.4665695994538 hold_14_15 - 462.5772677503618 hold_14_16
   + flow_nic_in_14_14 + flow_nic_in_14_15 + flow_nic_in_14_16
   + flow_nic_in_14_17 <= 0
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
 intergroup_group_tp_8_via_nic_out_4_nic_in_8: tp_on_and_switch_nic_out_4_nic_in_8_8 = 1 ->
   - 5109.459042174185 hold_8_1 - 3040.15247603248 hold_8_2
   - 2163.816929787807 hold_8_3 - 1679.651095409346 hold_8_4
   - 1372.538192304928 hold_8_5 - 1160.371981735487 hold_8_6
   - 1005.016998327084 hold_8_7 - 886.3492175210624 hold_8_8
   - 792.7454804093763 hold_8_9 - 717.0235396836346 hold_8_10
   - 654.5060374151892 hold_8_11 - 602.0160715548276 hold_8_12
   - 557.3201993053406 hold_8_13 - 518.8024130345685 hold_8_14
   - 485.2645706247653 hold_8_15 - 455.7995324779789 hold_8_16
   - 429.7078711806942 hold_8_17 - 406.441639349837 hold_8_18
   - 385.5654670831283 hold_8_19 - 366.7290626737931 hold_8_20
   - 349.6473985665369 hold_8_21 - 334.0861883096475 hold_8_22
   - 319.8510763911254 hold_8_23 - 306.7794787980271 hold_8_24
   - 294.7343457685529 hold_8_25 - 283.5993383963553 hold_8_26
   + flow_nic_in_8_8 + flow_nic_in_8_9 <= 0
 intergroup_group_tp_10_via_nic_out_4_nic_in_10: tp_on_and_switch_nic_out_4_nic_in_10_10 = 1 ->
   - 5366.149336128531 hold_10_1 - 3223.656633868661 hold_10_2
   - 2303.828417615377 hold_10_3 - 1792.392643000101 hold_10_4
   - 1466.777120905479 hold_10_5 - 1241.279690871386 hold_10_6
   - 1075.877817936759 hold_10_7 - 949.3728499935663 hold_10_8
   - 849.4875361276669 hold_10_9 - 768.6196123946033 hold_10_10
   - 701.8100166255107 hold_10_11 - 645.6859991034142 hold_10_12
   + flow_nic_in_10_10 + flow_nic_in_10_11 <= 0
 intergroup_group_tp_12_via_nic_out_4_nic_in_12: tp_on_and_switch_nic_out_4_nic_in_12_12 = 1 ->
   - 2305.702309237719 hold_12_1 - 1242.367720967523 hold_12_2
   - 850.2519296467542 hold_12_3 - 646.2748288017978 hold_12_4
   - 521.2305906758797 hold_12_5 - 436.730044930787 hold_12_6
   - 375.8054450200391 hold_12_7 - 329.7980448692438 hold_12_8
   - 293.8267585678413 hold_12_9 - 264.9306058433094 hold_12_10
   - 241.2090943547263 hold_12_11 - 221.3864637231713 hold_12_12
   + flow_nic_in_12_12 + flow_nic_in_12_13 <= 0
 intergroup_group_tp_14_via_nic_out_4_nic_in_14: tp_on_and_switch_nic_out_4_nic_in_14_14 = 1 ->
   - 5162.454372776313 hold_14_1 - 3077.750518312423 hold_14_2
   - 2192.410614391559 hold_14_3 - 1702.634102654364 hold_14_4
   - 1391.727311909141 hold_14_5 - 1176.833502919945 hold_14_6
   - 1019.425990533228 hold_14_7 - 899.1588818239977 hold_14_8
   - 804.2743562801687 hold_14_9 - 727.5038750948597 hold_14_10
   - 664.112297072705 hold_14_11 - 610.8825985968332 hold_14_12
   - 565.5526210836673 hold_14_13 - 526.4852692383147 hold_14_14
   - 492.4665695994538 hold_14_15 - 462.5772677503618 hold_14_16
   + flow_nic_in_14_14 + flow_nic_in_14_15 + flow_nic_in_14_16
   + flow_nic_in_14_17 <= 0
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
   - 5109.459042174185 hold_8_1 - 3040.15247603248 hold_8_2
   - 2163.816929787807 hold_8_3 - 1679.651095409346 hold_8_4
   - 1372.538192304928 hold_8_5 - 1160.371981735487 hold_8_6
   - 1005.016998327084 hold_8_7 - 886.3492175210624 hold_8_8
   - 792.7454804093763 hold_8_9 - 717.0235396836346 hold_8_10
   - 654.5060374151892 hold_8_11 - 602.0160715548276 hold_8_12
   - 557.3201993053406 hold_8_13 - 518.8024130345685 hold_8_14
   - 485.2645706247653 hold_8_15 - 455.7995324779789 hold_8_16
   - 429.7078711806942 hold_8_17 - 406.441639349837 hold_8_18
   - 385.5654670831283 hold_8_19 - 366.7290626737931 hold_8_20
   - 349.6473985665369 hold_8_21 - 334.0861883096475 hold_8_22
   - 319.8510763911254 hold_8_23 - 306.7794787980271 hold_8_24
   - 294.7343457685529 hold_8_25 - 283.5993383963553 hold_8_26
   + flow_nic_in_8_8 + flow_nic_in_8_9 <= 0
 intergroup_group_tp_10_via_nic_out_6_nic_in_10: tp_on_and_switch_nic_out_6_nic_in_10_10 = 1 ->
   - 5366.149336128531 hold_10_1 - 3223.656633868661 hold_10_2
   - 2303.828417615377 hold_10_3 - 1792.392643000101 hold_10_4
   - 1466.777120905479 hold_10_5 - 1241.279690871386 hold_10_6
   - 1075.877817936759 hold_10_7 - 949.3728499935663 hold_10_8
   - 849.4875361276669 hold_10_9 - 768.6196123946033 hold_10_10
   - 701.8100166255107 hold_10_11 - 645.6859991034142 hold_10_12
   + flow_nic_in_10_10 + flow_nic_in_10_11 <= 0
 intergroup_group_tp_12_via_nic_out_6_nic_in_12: tp_on_and_switch_nic_out_6_nic_in_12_12 = 1 ->
   - 2305.702309237719 hold_12_1 - 1242.367720967523 hold_12_2
   - 850.2519296467542 hold_12_3 - 646.2748288017978 hold_12_4
   - 521.2305906758797 hold_12_5 - 436.730044930787 hold_12_6
   - 375.8054450200391 hold_12_7 - 329.7980448692438 hold_12_8
   - 293.8267585678413 hold_12_9 - 264.9306058433094 hold_12_10
   - 241.2090943547263 hold_12_11 - 221.3864637231713 hold_12_12
   + flow_nic_in_12_12 + flow_nic_in_12_13 <= 0
 intergroup_group_tp_14_via_nic_out_6_nic_in_14: tp_on_and_switch_nic_out_6_nic_in_14_14 = 1 ->
   - 5162.454372776313 hold_14_1 - 3077.750518312423 hold_14_2
   - 2192.410614391559 hold_14_3 - 1702.634102654364 hold_14_4
   - 1391.727311909141 hold_14_5 - 1176.833502919945 hold_14_6
   - 1019.425990533228 hold_14_7 - 899.1588818239977 hold_14_8
   - 804.2743562801687 hold_14_9 - 727.5038750948597 hold_14_10
   - 664.112297072705 hold_14_11 - 610.8825985968332 hold_14_12
   - 565.5526210836673 hold_14_13 - 526.4852692383147 hold_14_14
   - 492.4665695994538 hold_14_15 - 462.5772677503618 hold_14_16
   + flow_nic_in_14_14 + flow_nic_in_14_15 + flow_nic_in_14_16
   + flow_nic_in_14_17 <= 0
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
   - 5109.459042174185 hold_8_1 - 3040.15247603248 hold_8_2
   - 2163.816929787807 hold_8_3 - 1679.651095409346 hold_8_4
   - 1372.538192304928 hold_8_5 - 1160.371981735487 hold_8_6
   - 1005.016998327084 hold_8_7 - 886.3492175210624 hold_8_8
   - 792.7454804093763 hold_8_9 - 717.0235396836346 hold_8_10
   - 654.5060374151892 hold_8_11 - 602.0160715548276 hold_8_12
   - 557.3201993053406 hold_8_13 - 518.8024130345685 hold_8_14
   - 485.2645706247653 hold_8_15 - 455.7995324779789 hold_8_16
   - 429.7078711806942 hold_8_17 - 406.441639349837 hold_8_18
   - 385.5654670831283 hold_8_19 - 366.7290626737931 hold_8_20
   - 349.6473985665369 hold_8_21 - 334.0861883096475 hold_8_22
   - 319.8510763911254 hold_8_23 - 306.7794787980271 hold_8_24
   - 294.7343457685529 hold_8_25 - 283.5993383963553 hold_8_26
   + flow_nic_in_8_8 + flow_nic_in_8_9 <= 0
 intergroup_group_tp_10_via_nic_out_7_nic_in_10: tp_on_and_switch_nic_out_7_nic_in_10_10 = 1 ->
   - 5366.149336128531 hold_10_1 - 3223.656633868661 hold_10_2
   - 2303.828417615377 hold_10_3 - 1792.392643000101 hold_10_4
   - 1466.777120905479 hold_10_5 - 1241.279690871386 hold_10_6
   - 1075.877817936759 hold_10_7 - 949.3728499935663 hold_10_8
   - 849.4875361276669 hold_10_9 - 768.6196123946033 hold_10_10
   - 701.8100166255107 hold_10_11 - 645.6859991034142 hold_10_12
   + flow_nic_in_10_10 + flow_nic_in_10_11 <= 0
 intergroup_group_tp_12_via_nic_out_7_nic_in_12: tp_on_and_switch_nic_out_7_nic_in_12_12 = 1 ->
   - 2305.702309237719 hold_12_1 - 1242.367720967523 hold_12_2
   - 850.2519296467542 hold_12_3 - 646.2748288017978 hold_12_4
   - 521.2305906758797 hold_12_5 - 436.730044930787 hold_12_6
   - 375.8054450200391 hold_12_7 - 329.7980448692438 hold_12_8
   - 293.8267585678413 hold_12_9 - 264.9306058433094 hold_12_10
   - 241.2090943547263 hold_12_11 - 221.3864637231713 hold_12_12
   + flow_nic_in_12_12 + flow_nic_in_12_13 <= 0
 intergroup_group_tp_14_via_nic_out_7_nic_in_14: tp_on_and_switch_nic_out_7_nic_in_14_14 = 1 ->
   - 5162.454372776313 hold_14_1 - 3077.750518312423 hold_14_2
   - 2192.410614391559 hold_14_3 - 1702.634102654364 hold_14_4
   - 1391.727311909141 hold_14_5 - 1176.833502919945 hold_14_6
   - 1019.425990533228 hold_14_7 - 899.1588818239977 hold_14_8
   - 804.2743562801687 hold_14_9 - 727.5038750948597 hold_14_10
   - 664.112297072705 hold_14_11 - 610.8825985968332 hold_14_12
   - 565.5526210836673 hold_14_13 - 526.4852692383147 hold_14_14
   - 492.4665695994538 hold_14_15 - 462.5772677503618 hold_14_16
   + flow_nic_in_14_14 + flow_nic_in_14_15 + flow_nic_in_14_16
   + flow_nic_in_14_17 <= 0
 intergroup_group_tp_8_via_source_nic_in_8: tp_on_and_switch_source_nic_in_8_8 = 1 ->
   - 6862.690961524641 hold_8_1 - 3585.117698418173 hold_8_2
   - 2426.322853123827 hold_8_3 - 1833.645313368965 hold_8_4
   - 1473.671728842764 hold_8_5 - 1231.841712172382 hold_8_6
   - 1058.192048119833 hold_8_7 - 927.4515136936157 hold_8_8
   - 825.4646488917072 hold_8_9 - 743.6855416581999 hold_8_10
   - 676.6495998303666 hold_8_11 - 620.6996291733516 hold_8_12
   - 573.2956619338331 hold_8_13 - 532.6186038805861 hold_8_14
   - 497.3314369348856 hold_8_15 - 466.429448710796 hold_8_16
   - 439.1430320530621 hold_8_17 - 414.8727167047877 hold_8_18
   - 393.144616467096 hold_8_19 - 373.5791797958418 hold_8_20
   - 355.8688290825181 hold_8_21 - 339.7616724131586 hold_8_22
   - 325.0494471689925 hold_8_23 - 311.5584659885816 hold_8_24
   - 299.1427275736955 hold_8_25 - 287.6786115922728 hold_8_26
   + flow_nic_in_8_8 + flow_nic_in_8_9 <= 0
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
   - 5366.149336128531 hold_10_1 - 3223.656633868661 hold_10_2
   - 2303.828417615377 hold_10_3 - 1792.392643000101 hold_10_4
   - 1466.777120905479 hold_10_5 - 1241.279690871386 hold_10_6
   - 1075.877817936759 hold_10_7 - 949.3728499935663 hold_10_8
   - 849.4875361276669 hold_10_9 - 768.6196123946033 hold_10_10
   - 701.8100166255107 hold_10_11 - 645.6859991034142 hold_10_12
   + flow_nic_in_10_10 + flow_nic_in_10_11 <= 0
 intergroup_group_tp_12_via_nic_out_8_nic_in_12: tp_on_and_switch_nic_out_8_nic_in_12_12 = 1 ->
   - 2305.702309237719 hold_12_1 - 1242.367720967523 hold_12_2
   - 850.2519296467542 hold_12_3 - 646.2748288017978 hold_12_4
   - 521.2305906758797 hold_12_5 - 436.730044930787 hold_12_6
   - 375.8054450200391 hold_12_7 - 329.7980448692438 hold_12_8
   - 293.8267585678413 hold_12_9 - 264.9306058433094 hold_12_10
   - 241.2090943547263 hold_12_11 - 221.3864637231713 hold_12_12
   + flow_nic_in_12_12 + flow_nic_in_12_13 <= 0
 intergroup_group_tp_14_via_nic_out_8_nic_in_14: tp_on_and_switch_nic_out_8_nic_in_14_14 = 1 ->
   - 5162.454372776313 hold_14_1 - 3077.750518312423 hold_14_2
   - 2192.410614391559 hold_14_3 - 1702.634102654364 hold_14_4
   - 1391.727311909141 hold_14_5 - 1176.833502919945 hold_14_6
   - 1019.425990533228 hold_14_7 - 899.1588818239977 hold_14_8
   - 804.2743562801687 hold_14_9 - 727.5038750948597 hold_14_10
   - 664.112297072705 hold_14_11 - 610.8825985968332 hold_14_12
   - 565.5526210836673 hold_14_13 - 526.4852692383147 hold_14_14
   - 492.4665695994538 hold_14_15 - 462.5772677503618 hold_14_16
   + flow_nic_in_14_14 + flow_nic_in_14_15 + flow_nic_in_14_16
   + flow_nic_in_14_17 <= 0
 intergroup_group_tp_10_via_source_nic_in_10: tp_on_and_switch_source_nic_in_10_10 = 1 ->
   - 7333.885604527395 hold_10_1 - 3843.098061973017 hold_10_2
   - 2603.759131617697 hold_10_3 - 1968.839296516373 hold_10_4
   - 1582.862441653366 hold_10_5 - 1323.416171696253 hold_10_6
   - 1137.043829654094 hold_10_7 - 996.6840907841146 hold_10_8
   - 887.1695081060992 hold_10_9 - 799.3389937728837 hold_10_10
   - 727.332477532996 hold_10_11 - 667.2269542320156 hold_10_12
   + flow_nic_in_10_10 + flow_nic_in_10_11 <= 0
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
   - 5109.459042174185 hold_8_1 - 3040.15247603248 hold_8_2
   - 2163.816929787807 hold_8_3 - 1679.651095409346 hold_8_4
   - 1372.538192304928 hold_8_5 - 1160.371981735487 hold_8_6
   - 1005.016998327084 hold_8_7 - 886.3492175210624 hold_8_8
   - 792.7454804093763 hold_8_9 - 717.0235396836346 hold_8_10
   - 654.5060374151892 hold_8_11 - 602.0160715548276 hold_8_12
   - 557.3201993053406 hold_8_13 - 518.8024130345685 hold_8_14
   - 485.2645706247653 hold_8_15 - 455.7995324779789 hold_8_16
   - 429.7078711806942 hold_8_17 - 406.441639349837 hold_8_18
   - 385.5654670831283 hold_8_19 - 366.7290626737931 hold_8_20
   - 349.6473985665369 hold_8_21 - 334.0861883096475 hold_8_22
   - 319.8510763911254 hold_8_23 - 306.7794787980271 hold_8_24
   - 294.7343457685529 hold_8_25 - 283.5993383963553 hold_8_26
   + flow_nic_in_8_8 + flow_nic_in_8_9 <= 0
 intergroup_group_tp_12_via_nic_out_10_nic_in_12: tp_on_and_switch_nic_out_10_nic_in_12_12 = 1 ->
   - 2305.702309237719 hold_12_1 - 1242.367720967523 hold_12_2
   - 850.2519296467542 hold_12_3 - 646.2748288017978 hold_12_4
   - 521.2305906758797 hold_12_5 - 436.730044930787 hold_12_6
   - 375.8054450200391 hold_12_7 - 329.7980448692438 hold_12_8
   - 293.8267585678413 hold_12_9 - 264.9306058433094 hold_12_10
   - 241.2090943547263 hold_12_11 - 221.3864637231713 hold_12_12
   + flow_nic_in_12_12 + flow_nic_in_12_13 <= 0
 intergroup_group_tp_14_via_nic_out_10_nic_in_14: tp_on_and_switch_nic_out_10_nic_in_14_14 = 1 ->
   - 5162.454372776313 hold_14_1 - 3077.750518312423 hold_14_2
   - 2192.410614391559 hold_14_3 - 1702.634102654364 hold_14_4
   - 1391.727311909141 hold_14_5 - 1176.833502919945 hold_14_6
   - 1019.425990533228 hold_14_7 - 899.1588818239977 hold_14_8
   - 804.2743562801687 hold_14_9 - 727.5038750948597 hold_14_10
   - 664.112297072705 hold_14_11 - 610.8825985968332 hold_14_12
   - 565.5526210836673 hold_14_13 - 526.4852692383147 hold_14_14
   - 492.4665695994538 hold_14_15 - 462.5772677503618 hold_14_16
   + flow_nic_in_14_14 + flow_nic_in_14_15 + flow_nic_in_14_16
   + flow_nic_in_14_17 <= 0
 intergroup_group_tp_12_via_source_nic_in_12: tp_on_and_switch_source_nic_in_12_12 = 1 ->
   - 2606.152953379399 hold_12_1 - 1324.653029216547 hold_12_2
   - 888.0032536439213 hold_12_3 - 667.8557467206057 hold_12_4
   - 535.1781518871273 hold_12_5 - 446.4795976683049 hold_12_6
   - 383.0021598768469 hold_12_7 - 335.3275636127566 hold_12_8
   - 298.2078305797816 hold_12_9 - 268.4871287270489 hold_12_10
   - 244.1536989855088 hold_12_11 - 223.8644921365358 hold_12_12
   + flow_nic_in_12_12 + flow_nic_in_12_13 <= 0
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
   - 5109.459042174185 hold_8_1 - 3040.15247603248 hold_8_2
   - 2163.816929787807 hold_8_3 - 1679.651095409346 hold_8_4
   - 1372.538192304928 hold_8_5 - 1160.371981735487 hold_8_6
   - 1005.016998327084 hold_8_7 - 886.3492175210624 hold_8_8
   - 792.7454804093763 hold_8_9 - 717.0235396836346 hold_8_10
   - 654.5060374151892 hold_8_11 - 602.0160715548276 hold_8_12
   - 557.3201993053406 hold_8_13 - 518.8024130345685 hold_8_14
   - 485.2645706247653 hold_8_15 - 455.7995324779789 hold_8_16
   - 429.7078711806942 hold_8_17 - 406.441639349837 hold_8_18
   - 385.5654670831283 hold_8_19 - 366.7290626737931 hold_8_20
   - 349.6473985665369 hold_8_21 - 334.0861883096475 hold_8_22
   - 319.8510763911254 hold_8_23 - 306.7794787980271 hold_8_24
   - 294.7343457685529 hold_8_25 - 283.5993383963553 hold_8_26
   + flow_nic_in_8_8 + flow_nic_in_8_9 <= 0
 intergroup_group_tp_10_via_nic_out_12_nic_in_10: tp_on_and_switch_nic_out_12_nic_in_10_10 = 1 ->
   - 5366.149336128531 hold_10_1 - 3223.656633868661 hold_10_2
   - 2303.828417615377 hold_10_3 - 1792.392643000101 hold_10_4
   - 1466.777120905479 hold_10_5 - 1241.279690871386 hold_10_6
   - 1075.877817936759 hold_10_7 - 949.3728499935663 hold_10_8
   - 849.4875361276669 hold_10_9 - 768.6196123946033 hold_10_10
   - 701.8100166255107 hold_10_11 - 645.6859991034142 hold_10_12
   + flow_nic_in_10_10 + flow_nic_in_10_11 <= 0
 intergroup_group_tp_14_via_nic_out_12_nic_in_14: tp_on_and_switch_nic_out_12_nic_in_14_14 = 1 ->
   - 5162.454372776313 hold_14_1 - 3077.750518312423 hold_14_2
   - 2192.410614391559 hold_14_3 - 1702.634102654364 hold_14_4
   - 1391.727311909141 hold_14_5 - 1176.833502919945 hold_14_6
   - 1019.425990533228 hold_14_7 - 899.1588818239977 hold_14_8
   - 804.2743562801687 hold_14_9 - 727.5038750948597 hold_14_10
   - 664.112297072705 hold_14_11 - 610.8825985968332 hold_14_12
   - 565.5526210836673 hold_14_13 - 526.4852692383147 hold_14_14
   - 492.4665695994538 hold_14_15 - 462.5772677503618 hold_14_16
   + flow_nic_in_14_14 + flow_nic_in_14_15 + flow_nic_in_14_16
   + flow_nic_in_14_17 <= 0
 intergroup_group_tp_14_via_source_nic_in_14: tp_on_and_switch_source_nic_in_14_14 = 1 ->
   - 6958.636559545705 hold_14_1 - 3637.519375474299 hold_14_2
   - 2462.33284799727 hold_14_3 - 1861.070180491239 hold_14_4
   - 1495.8156893303 hold_14_5 - 1250.409704554758 hold_14_6
   - 1074.178251958848 hold_14_7 - 941.4861641547031 hold_14_8
   - 837.9723394757928 hold_14_9 - 754.9659062115293 hold_14_10
   - 686.921963217819 hold_14_11 - 630.1293513779275 hold_14_12
   - 582.0104992511525 hold_14_13 - 540.7193067275552 hold_14_14
   - 504.8988600803767 hold_14_15 - 473.5294660813211 hold_14_16
   + flow_nic_in_14_14 + flow_nic_in_14_15 + flow_nic_in_14_16
   + flow_nic_in_14_17 <= 0
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
   - 5109.459042174185 hold_8_1 - 3040.15247603248 hold_8_2
   - 2163.816929787807 hold_8_3 - 1679.651095409346 hold_8_4
   - 1372.538192304928 hold_8_5 - 1160.371981735487 hold_8_6
   - 1005.016998327084 hold_8_7 - 886.3492175210624 hold_8_8
   - 792.7454804093763 hold_8_9 - 717.0235396836346 hold_8_10
   - 654.5060374151892 hold_8_11 - 602.0160715548276 hold_8_12
   - 557.3201993053406 hold_8_13 - 518.8024130345685 hold_8_14
   - 485.2645706247653 hold_8_15 - 455.7995324779789 hold_8_16
   - 429.7078711806942 hold_8_17 - 406.441639349837 hold_8_18
   - 385.5654670831283 hold_8_19 - 366.7290626737931 hold_8_20
   - 349.6473985665369 hold_8_21 - 334.0861883096475 hold_8_22
   - 319.8510763911254 hold_8_23 - 306.7794787980271 hold_8_24
   - 294.7343457685529 hold_8_25 - 283.5993383963553 hold_8_26
   + flow_nic_in_8_8 + flow_nic_in_8_9 <= 0
 intergroup_group_tp_10_via_nic_out_14_nic_in_10: tp_on_and_switch_nic_out_14_nic_in_10_10 = 1 ->
   - 5366.149336128531 hold_10_1 - 3223.656633868661 hold_10_2
   - 2303.828417615377 hold_10_3 - 1792.392643000101 hold_10_4
   - 1466.777120905479 hold_10_5 - 1241.279690871386 hold_10_6
   - 1075.877817936759 hold_10_7 - 949.3728499935663 hold_10_8
   - 849.4875361276669 hold_10_9 - 768.6196123946033 hold_10_10
   - 701.8100166255107 hold_10_11 - 645.6859991034142 hold_10_12
   + flow_nic_in_10_10 + flow_nic_in_10_11 <= 0
 intergroup_group_tp_12_via_nic_out_14_nic_in_12: tp_on_and_switch_nic_out_14_nic_in_12_12 = 1 ->
   - 2305.702309237719 hold_12_1 - 1242.367720967523 hold_12_2
   - 850.2519296467542 hold_12_3 - 646.2748288017978 hold_12_4
   - 521.2305906758797 hold_12_5 - 436.730044930787 hold_12_6
   - 375.8054450200391 hold_12_7 - 329.7980448692438 hold_12_8
   - 293.8267585678413 hold_12_9 - 264.9306058433094 hold_12_10
   - 241.2090943547263 hold_12_11 - 221.3864637231713 hold_12_12
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
 hold_15_22 hold_15_23 hold_15_24 hold_15_25 hold_15_26 hold_16_1 hold_16_2
 hold_16_3 hold_16_4 hold_16_5 hold_16_6 hold_16_7 hold_16_8 hold_16_9
 hold_16_10 hold_16_11 hold_16_12 hold_16_13 hold_16_14 hold_16_15
 hold_16_16 hold_16_17 hold_16_18 hold_16_19 hold_16_20 hold_16_21
 hold_16_22 hold_16_23 hold_16_24 hold_16_25 hold_16_26 hold_17_1 hold_17_2
 hold_17_3 hold_17_4 hold_17_5 hold_17_6 hold_17_7 hold_17_8 hold_17_9
 hold_17_10 hold_17_11 hold_17_12 hold_17_13 hold_17_14 hold_17_15
 hold_17_16 hold_17_17 hold_17_18 hold_17_19 hold_17_20 hold_17_21
 hold_17_22 hold_17_23 hold_17_24 hold_17_25 hold_17_26
 switch_source_nic_in_0 switch_nic_out_0_sink switch_nic_out_0_nic_in_1
 switch_nic_out_0_nic_in_2 switch_nic_out_0_nic_in_3
 switch_nic_out_0_nic_in_4 switch_nic_out_0_nic_in_6
 switch_nic_out_0_nic_in_7 switch_nic_out_0_nic_in_8
 switch_nic_out_0_nic_in_10 switch_nic_out_0_nic_in_12
 switch_nic_out_0_nic_in_14 switch_nic_in_0_0 switch_0_nic_out_0
 switch_source_nic_in_1 switch_nic_out_1_sink switch_nic_out_1_nic_in_0
 switch_nic_out_1_nic_in_2 switch_nic_out_1_nic_in_3
 switch_nic_out_1_nic_in_4 switch_nic_out_1_nic_in_6
 switch_nic_out_1_nic_in_7 switch_nic_out_1_nic_in_8
 switch_nic_out_1_nic_in_10 switch_nic_out_1_nic_in_12
 switch_nic_out_1_nic_in_14 switch_nic_in_1_1 switch_1_nic_out_1
 switch_source_nic_in_2 switch_nic_out_2_sink switch_nic_out_2_nic_in_0
 switch_nic_out_2_nic_in_1 switch_nic_out_2_nic_in_3
 switch_nic_out_2_nic_in_4 switch_nic_out_2_nic_in_6
 switch_nic_out_2_nic_in_7 switch_nic_out_2_nic_in_8
 switch_nic_out_2_nic_in_10 switch_nic_out_2_nic_in_12
 switch_nic_out_2_nic_in_14 switch_nic_in_2_2 switch_2_nic_out_2
 switch_source_nic_in_3 switch_nic_out_3_sink switch_nic_out_3_nic_in_0
 switch_nic_out_3_nic_in_1 switch_nic_out_3_nic_in_2
 switch_nic_out_3_nic_in_4 switch_nic_out_3_nic_in_6
 switch_nic_out_3_nic_in_7 switch_nic_out_3_nic_in_8
 switch_nic_out_3_nic_in_10 switch_nic_out_3_nic_in_12
 switch_nic_out_3_nic_in_14 switch_nic_in_3_3 switch_3_nic_out_3
 switch_source_nic_in_4 switch_nic_out_4_sink switch_nic_out_4_nic_in_0
 switch_nic_out_4_nic_in_1 switch_nic_out_4_nic_in_2
 switch_nic_out_4_nic_in_3 switch_nic_out_4_nic_in_6
 switch_nic_out_4_nic_in_7 switch_nic_out_4_nic_in_8
 switch_nic_out_4_nic_in_10 switch_nic_out_4_nic_in_12
 switch_nic_out_4_nic_in_14 switch_nic_in_4_4 switch_4_nic_out_4
 switch_nic_in_4_5 switch_5_nic_out_4 switch_4_5 switch_5_4
 switch_source_nic_in_6 switch_nic_out_6_sink switch_nic_out_6_nic_in_0
 switch_nic_out_6_nic_in_1 switch_nic_out_6_nic_in_2
 switch_nic_out_6_nic_in_3 switch_nic_out_6_nic_in_4
 switch_nic_out_6_nic_in_7 switch_nic_out_6_nic_in_8
 switch_nic_out_6_nic_in_10 switch_nic_out_6_nic_in_12
 switch_nic_out_6_nic_in_14 switch_nic_in_6_6 switch_6_nic_out_6
 switch_source_nic_in_7 switch_nic_out_7_sink switch_nic_out_7_nic_in_0
 switch_nic_out_7_nic_in_1 switch_nic_out_7_nic_in_2
 switch_nic_out_7_nic_in_3 switch_nic_out_7_nic_in_4
 switch_nic_out_7_nic_in_6 switch_nic_out_7_nic_in_8
 switch_nic_out_7_nic_in_10 switch_nic_out_7_nic_in_12
 switch_nic_out_7_nic_in_14 switch_nic_in_7_7 switch_7_nic_out_7
 switch_source_nic_in_8 switch_nic_out_8_sink switch_nic_out_8_nic_in_0
 switch_nic_out_8_nic_in_1 switch_nic_out_8_nic_in_2
 switch_nic_out_8_nic_in_3 switch_nic_out_8_nic_in_4
 switch_nic_out_8_nic_in_6 switch_nic_out_8_nic_in_7
 switch_nic_out_8_nic_in_10 switch_nic_out_8_nic_in_12
 switch_nic_out_8_nic_in_14 switch_nic_in_8_8 switch_8_nic_out_8
 switch_nic_in_8_9 switch_9_nic_out_8 switch_8_9 switch_9_8
 switch_source_nic_in_10 switch_nic_out_10_sink switch_nic_out_10_nic_in_0
 switch_nic_out_10_nic_in_1 switch_nic_out_10_nic_in_2
 switch_nic_out_10_nic_in_3 switch_nic_out_10_nic_in_4
 switch_nic_out_10_nic_in_6 switch_nic_out_10_nic_in_7
 switch_nic_out_10_nic_in_8 switch_nic_out_10_nic_in_12
 switch_nic_out_10_nic_in_14 switch_nic_in_10_10 switch_10_nic_out_10
 switch_nic_in_10_11 switch_11_nic_out_10 switch_10_11 switch_11_10
 switch_source_nic_in_12 switch_nic_out_12_sink switch_nic_out_12_nic_in_0
 switch_nic_out_12_nic_in_1 switch_nic_out_12_nic_in_2
 switch_nic_out_12_nic_in_3 switch_nic_out_12_nic_in_4
 switch_nic_out_12_nic_in_6 switch_nic_out_12_nic_in_7
 switch_nic_out_12_nic_in_8 switch_nic_out_12_nic_in_10
 switch_nic_out_12_nic_in_14 switch_nic_in_12_12 switch_12_nic_out_12
 switch_nic_in_12_13 switch_13_nic_out_12 switch_12_13 switch_13_12
 switch_source_nic_in_14 switch_nic_out_14_sink switch_nic_out_14_nic_in_0
 switch_nic_out_14_nic_in_1 switch_nic_out_14_nic_in_2
 switch_nic_out_14_nic_in_3 switch_nic_out_14_nic_in_4
 switch_nic_out_14_nic_in_6 switch_nic_out_14_nic_in_7
 switch_nic_out_14_nic_in_8 switch_nic_out_14_nic_in_10
 switch_nic_out_14_nic_in_12 switch_nic_in_14_14 switch_14_nic_out_14
 switch_nic_in_14_15 switch_15_nic_out_14 switch_nic_in_14_16
 switch_16_nic_out_14 switch_nic_in_14_17 switch_17_nic_out_14 switch_14_15
 switch_14_16 switch_14_17 switch_15_14 switch_15_16 switch_15_17
 switch_16_14 switch_16_15 switch_16_17 switch_17_14 switch_17_15
 switch_17_16 group_tp_active_0 group_tp_active_1 group_tp_active_2
 group_tp_active_3 group_tp_active_4 group_tp_active_6 group_tp_active_7
 group_tp_active_8 group_tp_active_10 group_tp_active_12 group_tp_active_14
 is_active_0 is_active_1 is_active_2 is_active_3 is_active_4 is_active_5
 is_active_6 is_active_7 is_active_8 is_active_9 is_active_10 is_active_11
 is_active_12 is_active_13 is_active_14 is_active_15 is_active_16
 is_active_17 is_start_zero_0_for_source_nic_in_0 is_end_complete_0
 is_start_zero_1_for_source_nic_in_1 is_end_complete_1
 is_start_zero_2_for_source_nic_in_2 is_end_complete_2
 is_start_zero_3_for_source_nic_in_3 is_end_complete_3
 is_start_zero_4_for_source_nic_in_4 is_start_zero_5_for_source_nic_in_4
 is_end_complete_4 is_end_complete_5 pp_permission_4_5 pp_permission_5_4
 is_start_zero_6_for_source_nic_in_6 is_end_complete_6
 is_start_zero_7_for_source_nic_in_7 is_end_complete_7
 is_start_zero_8_for_source_nic_in_8 is_start_zero_9_for_source_nic_in_8
 is_end_complete_8 is_end_complete_9 pp_permission_8_9 pp_permission_9_8
 is_start_zero_10_for_source_nic_in_10
 is_start_zero_11_for_source_nic_in_10 is_end_complete_10
 is_end_complete_11 pp_permission_10_11 pp_permission_11_10
 is_start_zero_12_for_source_nic_in_12
 is_start_zero_13_for_source_nic_in_12 is_end_complete_12
 is_end_complete_13 pp_permission_12_13 pp_permission_13_12
 is_start_zero_14_for_source_nic_in_14
 is_start_zero_15_for_source_nic_in_14
 is_start_zero_16_for_source_nic_in_14
 is_start_zero_17_for_source_nic_in_14 is_end_complete_14
 is_end_complete_15 is_end_complete_16 is_end_complete_17
 pp_permission_14_15 pp_permission_14_16 pp_permission_14_17
 pp_permission_15_14 pp_permission_15_16 pp_permission_15_17
 pp_permission_16_14 pp_permission_16_15 pp_permission_16_17
 pp_permission_17_14 pp_permission_17_15 pp_permission_17_16
 tp_on_and_switch_nic_out_0_nic_in_4_4
 tp_on_and_switch_nic_out_0_nic_in_8_8
 tp_on_and_switch_nic_out_0_nic_in_10_10
 tp_on_and_switch_nic_out_0_nic_in_12_12
 tp_on_and_switch_nic_out_0_nic_in_14_14
 tp_on_and_switch_nic_out_1_nic_in_4_4
 tp_on_and_switch_nic_out_1_nic_in_8_8
 tp_on_and_switch_nic_out_1_nic_in_10_10
 tp_on_and_switch_nic_out_1_nic_in_12_12
 tp_on_and_switch_nic_out_1_nic_in_14_14
 tp_on_and_switch_nic_out_2_nic_in_4_4
 tp_on_and_switch_nic_out_2_nic_in_8_8
 tp_on_and_switch_nic_out_2_nic_in_10_10
 tp_on_and_switch_nic_out_2_nic_in_12_12
 tp_on_and_switch_nic_out_2_nic_in_14_14
 tp_on_and_switch_nic_out_3_nic_in_4_4
 tp_on_and_switch_nic_out_3_nic_in_8_8
 tp_on_and_switch_nic_out_3_nic_in_10_10
 tp_on_and_switch_nic_out_3_nic_in_12_12
 tp_on_and_switch_nic_out_3_nic_in_14_14 tp_on_and_switch_source_nic_in_4_4
 tp_on_and_switch_nic_out_4_nic_in_8_8
 tp_on_and_switch_nic_out_4_nic_in_10_10
 tp_on_and_switch_nic_out_4_nic_in_12_12
 tp_on_and_switch_nic_out_4_nic_in_14_14
 tp_on_and_switch_nic_out_6_nic_in_4_4
 tp_on_and_switch_nic_out_6_nic_in_8_8
 tp_on_and_switch_nic_out_6_nic_in_10_10
 tp_on_and_switch_nic_out_6_nic_in_12_12
 tp_on_and_switch_nic_out_6_nic_in_14_14
 tp_on_and_switch_nic_out_7_nic_in_4_4
 tp_on_and_switch_nic_out_7_nic_in_8_8
 tp_on_and_switch_nic_out_7_nic_in_10_10
 tp_on_and_switch_nic_out_7_nic_in_12_12
 tp_on_and_switch_nic_out_7_nic_in_14_14 tp_on_and_switch_source_nic_in_8_8
 tp_on_and_switch_nic_out_8_nic_in_4_4
 tp_on_and_switch_nic_out_8_nic_in_10_10
 tp_on_and_switch_nic_out_8_nic_in_12_12
 tp_on_and_switch_nic_out_8_nic_in_14_14
 tp_on_and_switch_source_nic_in_10_10
 tp_on_and_switch_nic_out_10_nic_in_4_4
 tp_on_and_switch_nic_out_10_nic_in_8_8
 tp_on_and_switch_nic_out_10_nic_in_12_12
 tp_on_and_switch_nic_out_10_nic_in_14_14
 tp_on_and_switch_source_nic_in_12_12
 tp_on_and_switch_nic_out_12_nic_in_4_4
 tp_on_and_switch_nic_out_12_nic_in_8_8
 tp_on_and_switch_nic_out_12_nic_in_10_10
 tp_on_and_switch_nic_out_12_nic_in_14_14
 tp_on_and_switch_source_nic_in_14_14
 tp_on_and_switch_nic_out_14_nic_in_4_4
 tp_on_and_switch_nic_out_14_nic_in_8_8
 tp_on_and_switch_nic_out_14_nic_in_10_10
 tp_on_and_switch_nic_out_14_nic_in_12_12
Generals
 start_0 start_1 start_2 start_3 start_4 start_5 start_6 start_7 start_8
 start_9 start_10 start_11 start_12 start_13 start_14 start_15 start_16
 start_17 group_start_1 val_end_0 group_end_0 group_start_2 group_start_3
 group_start_4 group_start_6 group_start_7 group_start_8 group_start_10
 group_start_12 group_start_14 group_start_0 val_end_1 group_end_1
 val_end_2 group_end_2 val_end_3 group_end_3 val_end_4 val_end_5
 group_end_4 val_end_6 group_end_6 val_end_7 group_end_7 val_end_8
 val_end_9 group_end_8 val_end_10 val_end_11 group_end_10 val_end_12
 val_end_13 group_end_12 val_end_14 val_end_15 val_end_16 val_end_17
 group_end_14
General Constraints
 min_start_1: group_start_1 = MIN ( start_1 )
 max_end_0: group_end_0 = MAX ( val_end_0 )
 min_start_2: group_start_2 = MIN ( start_2 )
 min_start_3: group_start_3 = MIN ( start_3 )
 min_start_4: group_start_4 = MIN ( start_4 , start_5 )
 min_start_6: group_start_6 = MIN ( start_6 )
 min_start_7: group_start_7 = MIN ( start_7 )
 min_start_8: group_start_8 = MIN ( start_8 , start_9 )
 min_start_10: group_start_10 = MIN ( start_10 , start_11 )
 min_start_12: group_start_12 = MIN ( start_12 , start_13 )
 min_start_14: group_start_14 = MIN ( start_14 , start_15 , start_16 ,
   start_17 )
 min_start_0: group_start_0 = MIN ( start_0 )
 max_end_1: group_end_1 = MAX ( val_end_1 )
 max_end_2: group_end_2 = MAX ( val_end_2 )
 max_end_3: group_end_3 = MAX ( val_end_3 )
 max_end_4: group_end_4 = MAX ( val_end_4 , val_end_5 )
 max_end_6: group_end_6 = MAX ( val_end_6 )
 max_end_7: group_end_7 = MAX ( val_end_7 )
 max_end_8: group_end_8 = MAX ( val_end_8 , val_end_9 )
 max_end_10: group_end_10 = MAX ( val_end_10 , val_end_11 )
 max_end_12: group_end_12 = MAX ( val_end_12 , val_end_13 )
 max_end_14: group_end_14 = MAX ( val_end_14 , val_end_15 , val_end_16 ,
   val_end_17 )
End
