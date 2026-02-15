\ Model 2026-02-15_14-50-10
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  5.2301255230125519e-04 flow_source_nic_in_0
   + 5.2301255230125519e-04 flow_source_nic_in_4
   + 5.2301255230125519e-04 flow_source_nic_in_5
   + 5.2301255230125519e-04 flow_source_nic_in_6
   + 5.2301255230125519e-04 flow_source_nic_in_7
   + 5.2301255230125519e-04 flow_source_nic_in_8
   + 5.2301255230125519e-04 flow_source_nic_in_9
   + 5.2301255230125519e-04 flow_source_nic_in_10
   + 5.2301255230125519e-04 flow_source_nic_in_11
   + 5.2301255230125519e-04 flow_source_nic_in_12
   + 5.2301255230125519e-04 flow_source_nic_in_13
   + 5.2301255230125519e-04 flow_source_nic_in_14
   - 0.024390243902439 is_active_0 - 0.024390243902439 is_active_1
   - 0.024390243902439 is_active_2 - 0.024390243902439 is_active_3
   - 0.0609756097560976 is_active_4 - 0.0365853658536585 is_active_5
   - 0.0365853658536585 is_active_6 - 0.0365853658536585 is_active_7
   - 0.0365853658536585 is_active_8 - 0.024390243902439 is_active_9
   - 0.024390243902439 is_active_10 - 0.024390243902439 is_active_11
   - 0.0609756097560976 is_active_12 - 0.024390243902439 is_active_13
   - 0.0365853658536585 is_active_14 + 0 val_end_0 + 0 val_end_1
   + 0 val_end_2 + 0 val_end_3 + 0 val_end_4 + 0 val_end_5 + 0 val_end_6
   + 0 val_end_7 + 0 val_end_8 + 0 val_end_9 + 0 val_end_10 + 0 val_end_11
   + 0 val_end_12 + 0 val_end_13 + 0 val_end_14
Subject To
 mem_active_constr_0: hold_0_1 + hold_0_2 + hold_0_3 + hold_0_4 + hold_0_5
   + hold_0_6 + hold_0_7 + hold_0_8 + hold_0_9 + hold_0_10 + hold_0_11
   + hold_0_12 + hold_0_13 + hold_0_14 + hold_0_15 + hold_0_16 + hold_0_17
   + hold_0_18 + hold_0_19 + hold_0_20 + hold_0_21 + hold_0_22 + hold_0_23
   + hold_0_24 + hold_0_25 + hold_0_26 + hold_0_27 + hold_0_28 + hold_0_29
   + hold_0_30 + hold_0_31 + hold_0_32 - 61 is_active_0 <= 0
 mem_active_constr2_0: hold_0_1 + hold_0_2 + hold_0_3 + hold_0_4 + hold_0_5
   + hold_0_6 + hold_0_7 + hold_0_8 + hold_0_9 + hold_0_10 + hold_0_11
   + hold_0_12 + hold_0_13 + hold_0_14 + hold_0_15 + hold_0_16 + hold_0_17
   + hold_0_18 + hold_0_19 + hold_0_20 + hold_0_21 + hold_0_22 + hold_0_23
   + hold_0_24 + hold_0_25 + hold_0_26 + hold_0_27 + hold_0_28 + hold_0_29
   + hold_0_30 + hold_0_31 + hold_0_32 - is_active_0 >= 0
 mem_active_constr_1: hold_1_1 + hold_1_2 + hold_1_3 + hold_1_4 + hold_1_5
   + hold_1_6 + hold_1_7 + hold_1_8 + hold_1_9 + hold_1_10 + hold_1_11
   + hold_1_12 + hold_1_13 + hold_1_14 + hold_1_15 + hold_1_16 + hold_1_17
   + hold_1_18 + hold_1_19 + hold_1_20 + hold_1_21 + hold_1_22 + hold_1_23
   + hold_1_24 + hold_1_25 + hold_1_26 + hold_1_27 + hold_1_28 + hold_1_29
   + hold_1_30 + hold_1_31 + hold_1_32 - 61 is_active_1 <= 0
 mem_active_constr2_1: hold_1_1 + hold_1_2 + hold_1_3 + hold_1_4 + hold_1_5
   + hold_1_6 + hold_1_7 + hold_1_8 + hold_1_9 + hold_1_10 + hold_1_11
   + hold_1_12 + hold_1_13 + hold_1_14 + hold_1_15 + hold_1_16 + hold_1_17
   + hold_1_18 + hold_1_19 + hold_1_20 + hold_1_21 + hold_1_22 + hold_1_23
   + hold_1_24 + hold_1_25 + hold_1_26 + hold_1_27 + hold_1_28 + hold_1_29
   + hold_1_30 + hold_1_31 + hold_1_32 - is_active_1 >= 0
 mem_active_constr_2: hold_2_1 + hold_2_2 + hold_2_3 + hold_2_4 + hold_2_5
   + hold_2_6 + hold_2_7 + hold_2_8 + hold_2_9 + hold_2_10 + hold_2_11
   + hold_2_12 + hold_2_13 + hold_2_14 + hold_2_15 + hold_2_16 + hold_2_17
   + hold_2_18 + hold_2_19 + hold_2_20 + hold_2_21 + hold_2_22 + hold_2_23
   + hold_2_24 + hold_2_25 + hold_2_26 + hold_2_27 + hold_2_28 + hold_2_29
   + hold_2_30 + hold_2_31 + hold_2_32 - 61 is_active_2 <= 0
 mem_active_constr2_2: hold_2_1 + hold_2_2 + hold_2_3 + hold_2_4 + hold_2_5
   + hold_2_6 + hold_2_7 + hold_2_8 + hold_2_9 + hold_2_10 + hold_2_11
   + hold_2_12 + hold_2_13 + hold_2_14 + hold_2_15 + hold_2_16 + hold_2_17
   + hold_2_18 + hold_2_19 + hold_2_20 + hold_2_21 + hold_2_22 + hold_2_23
   + hold_2_24 + hold_2_25 + hold_2_26 + hold_2_27 + hold_2_28 + hold_2_29
   + hold_2_30 + hold_2_31 + hold_2_32 - is_active_2 >= 0
 mem_active_constr_3: hold_3_1 + hold_3_2 + hold_3_3 + hold_3_4 + hold_3_5
   + hold_3_6 + hold_3_7 + hold_3_8 + hold_3_9 + hold_3_10 + hold_3_11
   + hold_3_12 + hold_3_13 + hold_3_14 + hold_3_15 + hold_3_16 + hold_3_17
   + hold_3_18 + hold_3_19 + hold_3_20 + hold_3_21 + hold_3_22 + hold_3_23
   + hold_3_24 + hold_3_25 + hold_3_26 + hold_3_27 + hold_3_28 + hold_3_29
   + hold_3_30 + hold_3_31 + hold_3_32 - 61 is_active_3 <= 0
 mem_active_constr2_3: hold_3_1 + hold_3_2 + hold_3_3 + hold_3_4 + hold_3_5
   + hold_3_6 + hold_3_7 + hold_3_8 + hold_3_9 + hold_3_10 + hold_3_11
   + hold_3_12 + hold_3_13 + hold_3_14 + hold_3_15 + hold_3_16 + hold_3_17
   + hold_3_18 + hold_3_19 + hold_3_20 + hold_3_21 + hold_3_22 + hold_3_23
   + hold_3_24 + hold_3_25 + hold_3_26 + hold_3_27 + hold_3_28 + hold_3_29
   + hold_3_30 + hold_3_31 + hold_3_32 - is_active_3 >= 0
 mem_active_constr_4: hold_4_1 + hold_4_2 + hold_4_3 + hold_4_4 + hold_4_5
   + hold_4_6 + hold_4_7 + hold_4_8 + hold_4_9 + hold_4_10 + hold_4_11
   + hold_4_12 + hold_4_13 + hold_4_14 + hold_4_15 + hold_4_16 + hold_4_17
   + hold_4_18 + hold_4_19 + hold_4_20 + hold_4_21 + hold_4_22 + hold_4_23
   + hold_4_24 + hold_4_25 + hold_4_26 + hold_4_27 + hold_4_28 + hold_4_29
   + hold_4_30 + hold_4_31 + hold_4_32 - 61 is_active_4 <= 0
 mem_active_constr2_4: hold_4_1 + hold_4_2 + hold_4_3 + hold_4_4 + hold_4_5
   + hold_4_6 + hold_4_7 + hold_4_8 + hold_4_9 + hold_4_10 + hold_4_11
   + hold_4_12 + hold_4_13 + hold_4_14 + hold_4_15 + hold_4_16 + hold_4_17
   + hold_4_18 + hold_4_19 + hold_4_20 + hold_4_21 + hold_4_22 + hold_4_23
   + hold_4_24 + hold_4_25 + hold_4_26 + hold_4_27 + hold_4_28 + hold_4_29
   + hold_4_30 + hold_4_31 + hold_4_32 - is_active_4 >= 0
 mem_active_constr_5: hold_5_1 + hold_5_2 + hold_5_3 + hold_5_4 + hold_5_5
   + hold_5_6 + hold_5_7 + hold_5_8 + hold_5_9 + hold_5_10 + hold_5_11
   + hold_5_12 + hold_5_13 + hold_5_14 + hold_5_15 + hold_5_16 + hold_5_17
   + hold_5_18 + hold_5_19 + hold_5_20 + hold_5_21 + hold_5_22 + hold_5_23
   + hold_5_24 + hold_5_25 + hold_5_26 + hold_5_27 + hold_5_28 + hold_5_29
   + hold_5_30 + hold_5_31 + hold_5_32 - 61 is_active_5 <= 0
 mem_active_constr2_5: hold_5_1 + hold_5_2 + hold_5_3 + hold_5_4 + hold_5_5
   + hold_5_6 + hold_5_7 + hold_5_8 + hold_5_9 + hold_5_10 + hold_5_11
   + hold_5_12 + hold_5_13 + hold_5_14 + hold_5_15 + hold_5_16 + hold_5_17
   + hold_5_18 + hold_5_19 + hold_5_20 + hold_5_21 + hold_5_22 + hold_5_23
   + hold_5_24 + hold_5_25 + hold_5_26 + hold_5_27 + hold_5_28 + hold_5_29
   + hold_5_30 + hold_5_31 + hold_5_32 - is_active_5 >= 0
 mem_active_constr_6: hold_6_1 + hold_6_2 + hold_6_3 + hold_6_4 + hold_6_5
   + hold_6_6 + hold_6_7 + hold_6_8 + hold_6_9 + hold_6_10 + hold_6_11
   + hold_6_12 + hold_6_13 + hold_6_14 + hold_6_15 + hold_6_16 + hold_6_17
   + hold_6_18 + hold_6_19 + hold_6_20 + hold_6_21 + hold_6_22 + hold_6_23
   + hold_6_24 + hold_6_25 + hold_6_26 + hold_6_27 + hold_6_28 + hold_6_29
   + hold_6_30 + hold_6_31 + hold_6_32 - 61 is_active_6 <= 0
 mem_active_constr2_6: hold_6_1 + hold_6_2 + hold_6_3 + hold_6_4 + hold_6_5
   + hold_6_6 + hold_6_7 + hold_6_8 + hold_6_9 + hold_6_10 + hold_6_11
   + hold_6_12 + hold_6_13 + hold_6_14 + hold_6_15 + hold_6_16 + hold_6_17
   + hold_6_18 + hold_6_19 + hold_6_20 + hold_6_21 + hold_6_22 + hold_6_23
   + hold_6_24 + hold_6_25 + hold_6_26 + hold_6_27 + hold_6_28 + hold_6_29
   + hold_6_30 + hold_6_31 + hold_6_32 - is_active_6 >= 0
 mem_active_constr_7: hold_7_1 + hold_7_2 + hold_7_3 + hold_7_4 + hold_7_5
   + hold_7_6 + hold_7_7 + hold_7_8 + hold_7_9 + hold_7_10 + hold_7_11
   + hold_7_12 + hold_7_13 + hold_7_14 + hold_7_15 + hold_7_16 + hold_7_17
   + hold_7_18 + hold_7_19 + hold_7_20 + hold_7_21 + hold_7_22 + hold_7_23
   + hold_7_24 + hold_7_25 + hold_7_26 + hold_7_27 + hold_7_28 + hold_7_29
   + hold_7_30 + hold_7_31 + hold_7_32 - 61 is_active_7 <= 0
 mem_active_constr2_7: hold_7_1 + hold_7_2 + hold_7_3 + hold_7_4 + hold_7_5
   + hold_7_6 + hold_7_7 + hold_7_8 + hold_7_9 + hold_7_10 + hold_7_11
   + hold_7_12 + hold_7_13 + hold_7_14 + hold_7_15 + hold_7_16 + hold_7_17
   + hold_7_18 + hold_7_19 + hold_7_20 + hold_7_21 + hold_7_22 + hold_7_23
   + hold_7_24 + hold_7_25 + hold_7_26 + hold_7_27 + hold_7_28 + hold_7_29
   + hold_7_30 + hold_7_31 + hold_7_32 - is_active_7 >= 0
 mem_active_constr_8: hold_8_1 + hold_8_2 + hold_8_3 + hold_8_4 + hold_8_5
   + hold_8_6 + hold_8_7 + hold_8_8 + hold_8_9 + hold_8_10 + hold_8_11
   + hold_8_12 + hold_8_13 + hold_8_14 + hold_8_15 + hold_8_16 + hold_8_17
   + hold_8_18 + hold_8_19 + hold_8_20 + hold_8_21 + hold_8_22 + hold_8_23
   + hold_8_24 + hold_8_25 + hold_8_26 + hold_8_27 + hold_8_28 + hold_8_29
   + hold_8_30 + hold_8_31 + hold_8_32 - 61 is_active_8 <= 0
 mem_active_constr2_8: hold_8_1 + hold_8_2 + hold_8_3 + hold_8_4 + hold_8_5
   + hold_8_6 + hold_8_7 + hold_8_8 + hold_8_9 + hold_8_10 + hold_8_11
   + hold_8_12 + hold_8_13 + hold_8_14 + hold_8_15 + hold_8_16 + hold_8_17
   + hold_8_18 + hold_8_19 + hold_8_20 + hold_8_21 + hold_8_22 + hold_8_23
   + hold_8_24 + hold_8_25 + hold_8_26 + hold_8_27 + hold_8_28 + hold_8_29
   + hold_8_30 + hold_8_31 + hold_8_32 - is_active_8 >= 0
 mem_active_constr_9: hold_9_1 + hold_9_2 + hold_9_3 + hold_9_4 + hold_9_5
   + hold_9_6 + hold_9_7 + hold_9_8 + hold_9_9 + hold_9_10 + hold_9_11
   + hold_9_12 + hold_9_13 + hold_9_14 + hold_9_15 + hold_9_16 + hold_9_17
   + hold_9_18 + hold_9_19 + hold_9_20 + hold_9_21 + hold_9_22 + hold_9_23
   + hold_9_24 + hold_9_25 + hold_9_26 + hold_9_27 + hold_9_28 + hold_9_29
   + hold_9_30 + hold_9_31 + hold_9_32 - 61 is_active_9 <= 0
 mem_active_constr2_9: hold_9_1 + hold_9_2 + hold_9_3 + hold_9_4 + hold_9_5
   + hold_9_6 + hold_9_7 + hold_9_8 + hold_9_9 + hold_9_10 + hold_9_11
   + hold_9_12 + hold_9_13 + hold_9_14 + hold_9_15 + hold_9_16 + hold_9_17
   + hold_9_18 + hold_9_19 + hold_9_20 + hold_9_21 + hold_9_22 + hold_9_23
   + hold_9_24 + hold_9_25 + hold_9_26 + hold_9_27 + hold_9_28 + hold_9_29
   + hold_9_30 + hold_9_31 + hold_9_32 - is_active_9 >= 0
 mem_active_constr_10: hold_10_1 + hold_10_2 + hold_10_3 + hold_10_4
   + hold_10_5 + hold_10_6 + hold_10_7 + hold_10_8 + hold_10_9 + hold_10_10
   + hold_10_11 + hold_10_12 + hold_10_13 + hold_10_14 + hold_10_15
   + hold_10_16 + hold_10_17 + hold_10_18 + hold_10_19 + hold_10_20
   + hold_10_21 + hold_10_22 + hold_10_23 + hold_10_24 + hold_10_25
   + hold_10_26 + hold_10_27 + hold_10_28 + hold_10_29 + hold_10_30
   + hold_10_31 + hold_10_32 - 61 is_active_10 <= 0
 mem_active_constr2_10: hold_10_1 + hold_10_2 + hold_10_3 + hold_10_4
   + hold_10_5 + hold_10_6 + hold_10_7 + hold_10_8 + hold_10_9 + hold_10_10
   + hold_10_11 + hold_10_12 + hold_10_13 + hold_10_14 + hold_10_15
   + hold_10_16 + hold_10_17 + hold_10_18 + hold_10_19 + hold_10_20
   + hold_10_21 + hold_10_22 + hold_10_23 + hold_10_24 + hold_10_25
   + hold_10_26 + hold_10_27 + hold_10_28 + hold_10_29 + hold_10_30
   + hold_10_31 + hold_10_32 - is_active_10 >= 0
 mem_active_constr_11: hold_11_1 + hold_11_2 + hold_11_3 + hold_11_4
   + hold_11_5 + hold_11_6 + hold_11_7 + hold_11_8 + hold_11_9 + hold_11_10
   + hold_11_11 + hold_11_12 + hold_11_13 + hold_11_14 + hold_11_15
   + hold_11_16 + hold_11_17 + hold_11_18 + hold_11_19 + hold_11_20
   + hold_11_21 + hold_11_22 + hold_11_23 + hold_11_24 + hold_11_25
   + hold_11_26 + hold_11_27 + hold_11_28 + hold_11_29 + hold_11_30
   + hold_11_31 + hold_11_32 - 61 is_active_11 <= 0
 mem_active_constr2_11: hold_11_1 + hold_11_2 + hold_11_3 + hold_11_4
   + hold_11_5 + hold_11_6 + hold_11_7 + hold_11_8 + hold_11_9 + hold_11_10
   + hold_11_11 + hold_11_12 + hold_11_13 + hold_11_14 + hold_11_15
   + hold_11_16 + hold_11_17 + hold_11_18 + hold_11_19 + hold_11_20
   + hold_11_21 + hold_11_22 + hold_11_23 + hold_11_24 + hold_11_25
   + hold_11_26 + hold_11_27 + hold_11_28 + hold_11_29 + hold_11_30
   + hold_11_31 + hold_11_32 - is_active_11 >= 0
 mem_active_constr_12: hold_12_1 + hold_12_2 + hold_12_3 + hold_12_4
   + hold_12_5 + hold_12_6 + hold_12_7 + hold_12_8 + hold_12_9 + hold_12_10
   + hold_12_11 + hold_12_12 + hold_12_13 + hold_12_14 + hold_12_15
   + hold_12_16 + hold_12_17 + hold_12_18 + hold_12_19 + hold_12_20
   + hold_12_21 + hold_12_22 + hold_12_23 + hold_12_24 + hold_12_25
   + hold_12_26 + hold_12_27 + hold_12_28 + hold_12_29 + hold_12_30
   + hold_12_31 + hold_12_32 - 61 is_active_12 <= 0
 mem_active_constr2_12: hold_12_1 + hold_12_2 + hold_12_3 + hold_12_4
   + hold_12_5 + hold_12_6 + hold_12_7 + hold_12_8 + hold_12_9 + hold_12_10
   + hold_12_11 + hold_12_12 + hold_12_13 + hold_12_14 + hold_12_15
   + hold_12_16 + hold_12_17 + hold_12_18 + hold_12_19 + hold_12_20
   + hold_12_21 + hold_12_22 + hold_12_23 + hold_12_24 + hold_12_25
   + hold_12_26 + hold_12_27 + hold_12_28 + hold_12_29 + hold_12_30
   + hold_12_31 + hold_12_32 - is_active_12 >= 0
 mem_active_constr_13: hold_13_1 + hold_13_2 + hold_13_3 + hold_13_4
   + hold_13_5 + hold_13_6 + hold_13_7 + hold_13_8 + hold_13_9 + hold_13_10
   + hold_13_11 + hold_13_12 + hold_13_13 + hold_13_14 + hold_13_15
   + hold_13_16 + hold_13_17 + hold_13_18 + hold_13_19 + hold_13_20
   + hold_13_21 + hold_13_22 + hold_13_23 + hold_13_24 + hold_13_25
   + hold_13_26 + hold_13_27 + hold_13_28 + hold_13_29 + hold_13_30
   + hold_13_31 + hold_13_32 - 61 is_active_13 <= 0
 mem_active_constr2_13: hold_13_1 + hold_13_2 + hold_13_3 + hold_13_4
   + hold_13_5 + hold_13_6 + hold_13_7 + hold_13_8 + hold_13_9 + hold_13_10
   + hold_13_11 + hold_13_12 + hold_13_13 + hold_13_14 + hold_13_15
   + hold_13_16 + hold_13_17 + hold_13_18 + hold_13_19 + hold_13_20
   + hold_13_21 + hold_13_22 + hold_13_23 + hold_13_24 + hold_13_25
   + hold_13_26 + hold_13_27 + hold_13_28 + hold_13_29 + hold_13_30
   + hold_13_31 + hold_13_32 - is_active_13 >= 0
 mem_active_constr_14: hold_14_1 + hold_14_2 + hold_14_3 + hold_14_4
   + hold_14_5 + hold_14_6 + hold_14_7 + hold_14_8 + hold_14_9 + hold_14_10
   + hold_14_11 + hold_14_12 + hold_14_13 + hold_14_14 + hold_14_15
   + hold_14_16 + hold_14_17 + hold_14_18 + hold_14_19 + hold_14_20
   + hold_14_21 + hold_14_22 + hold_14_23 + hold_14_24 + hold_14_25
   + hold_14_26 + hold_14_27 + hold_14_28 + hold_14_29 + hold_14_30
   + hold_14_31 + hold_14_32 - 61 is_active_14 <= 0
 mem_active_constr2_14: hold_14_1 + hold_14_2 + hold_14_3 + hold_14_4
   + hold_14_5 + hold_14_6 + hold_14_7 + hold_14_8 + hold_14_9 + hold_14_10
   + hold_14_11 + hold_14_12 + hold_14_13 + hold_14_14 + hold_14_15
   + hold_14_16 + hold_14_17 + hold_14_18 + hold_14_19 + hold_14_20
   + hold_14_21 + hold_14_22 + hold_14_23 + hold_14_24 + hold_14_25
   + hold_14_26 + hold_14_27 + hold_14_28 + hold_14_29 + hold_14_30
   + hold_14_31 + hold_14_32 - is_active_14 >= 0
 hold_constraint_0: hold_0_1 + hold_0_2 + hold_0_3 + hold_0_4 + hold_0_5
   + hold_0_6 + hold_0_7 + hold_0_8 + hold_0_9 + hold_0_10 + hold_0_11
   + hold_0_12 + hold_0_13 + hold_0_14 + hold_0_15 + hold_0_16 + hold_0_17
   + hold_0_18 + hold_0_19 + hold_0_20 + hold_0_21 + hold_0_22 + hold_0_23
   + hold_0_24 + hold_0_25 + hold_0_26 + hold_0_27 + hold_0_28 + hold_0_29
   + hold_0_30 + hold_0_31 + hold_0_32 <= 1
 end_constraint_0: start_0 + hold_0_1 + 2 hold_0_2 + 3 hold_0_3
   + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7 + 8 hold_0_8
   + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12 + 13 hold_0_13
   + 14 hold_0_14 + 15 hold_0_15 + 16 hold_0_16 + 17 hold_0_17
   + 18 hold_0_18 + 19 hold_0_19 + 20 hold_0_20 + 21 hold_0_21
   + 22 hold_0_22 + 23 hold_0_23 + 24 hold_0_24 + 25 hold_0_25
   + 26 hold_0_26 + 27 hold_0_27 + 28 hold_0_28 + 29 hold_0_29
   + 30 hold_0_30 + 31 hold_0_31 + 32 hold_0_32 <= 60
 hold_constraint_1: hold_1_1 + hold_1_2 + hold_1_3 + hold_1_4 + hold_1_5
   + hold_1_6 + hold_1_7 + hold_1_8 + hold_1_9 + hold_1_10 + hold_1_11
   + hold_1_12 + hold_1_13 + hold_1_14 + hold_1_15 + hold_1_16 + hold_1_17
   + hold_1_18 + hold_1_19 + hold_1_20 + hold_1_21 + hold_1_22 + hold_1_23
   + hold_1_24 + hold_1_25 + hold_1_26 + hold_1_27 + hold_1_28 + hold_1_29
   + hold_1_30 + hold_1_31 + hold_1_32 <= 1
 end_constraint_1: start_1 + hold_1_1 + 2 hold_1_2 + 3 hold_1_3
   + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7 + 8 hold_1_8
   + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 + 12 hold_1_12 + 13 hold_1_13
   + 14 hold_1_14 + 15 hold_1_15 + 16 hold_1_16 + 17 hold_1_17
   + 18 hold_1_18 + 19 hold_1_19 + 20 hold_1_20 + 21 hold_1_21
   + 22 hold_1_22 + 23 hold_1_23 + 24 hold_1_24 + 25 hold_1_25
   + 26 hold_1_26 + 27 hold_1_27 + 28 hold_1_28 + 29 hold_1_29
   + 30 hold_1_30 + 31 hold_1_31 + 32 hold_1_32 <= 60
 hold_constraint_2: hold_2_1 + hold_2_2 + hold_2_3 + hold_2_4 + hold_2_5
   + hold_2_6 + hold_2_7 + hold_2_8 + hold_2_9 + hold_2_10 + hold_2_11
   + hold_2_12 + hold_2_13 + hold_2_14 + hold_2_15 + hold_2_16 + hold_2_17
   + hold_2_18 + hold_2_19 + hold_2_20 + hold_2_21 + hold_2_22 + hold_2_23
   + hold_2_24 + hold_2_25 + hold_2_26 + hold_2_27 + hold_2_28 + hold_2_29
   + hold_2_30 + hold_2_31 + hold_2_32 <= 1
 end_constraint_2: start_2 + hold_2_1 + 2 hold_2_2 + 3 hold_2_3
   + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7 + 8 hold_2_8
   + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 12 hold_2_12 + 13 hold_2_13
   + 14 hold_2_14 + 15 hold_2_15 + 16 hold_2_16 + 17 hold_2_17
   + 18 hold_2_18 + 19 hold_2_19 + 20 hold_2_20 + 21 hold_2_21
   + 22 hold_2_22 + 23 hold_2_23 + 24 hold_2_24 + 25 hold_2_25
   + 26 hold_2_26 + 27 hold_2_27 + 28 hold_2_28 + 29 hold_2_29
   + 30 hold_2_30 + 31 hold_2_31 + 32 hold_2_32 <= 60
 hold_constraint_3: hold_3_1 + hold_3_2 + hold_3_3 + hold_3_4 + hold_3_5
   + hold_3_6 + hold_3_7 + hold_3_8 + hold_3_9 + hold_3_10 + hold_3_11
   + hold_3_12 + hold_3_13 + hold_3_14 + hold_3_15 + hold_3_16 + hold_3_17
   + hold_3_18 + hold_3_19 + hold_3_20 + hold_3_21 + hold_3_22 + hold_3_23
   + hold_3_24 + hold_3_25 + hold_3_26 + hold_3_27 + hold_3_28 + hold_3_29
   + hold_3_30 + hold_3_31 + hold_3_32 <= 1
 end_constraint_3: start_3 + hold_3_1 + 2 hold_3_2 + 3 hold_3_3
   + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7 + 8 hold_3_8
   + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11 + 12 hold_3_12 + 13 hold_3_13
   + 14 hold_3_14 + 15 hold_3_15 + 16 hold_3_16 + 17 hold_3_17
   + 18 hold_3_18 + 19 hold_3_19 + 20 hold_3_20 + 21 hold_3_21
   + 22 hold_3_22 + 23 hold_3_23 + 24 hold_3_24 + 25 hold_3_25
   + 26 hold_3_26 + 27 hold_3_27 + 28 hold_3_28 + 29 hold_3_29
   + 30 hold_3_30 + 31 hold_3_31 + 32 hold_3_32 <= 60
 hold_constraint_4: hold_4_1 + hold_4_2 + hold_4_3 + hold_4_4 + hold_4_5
   + hold_4_6 + hold_4_7 + hold_4_8 + hold_4_9 + hold_4_10 + hold_4_11
   + hold_4_12 + hold_4_13 + hold_4_14 + hold_4_15 + hold_4_16 + hold_4_17
   + hold_4_18 + hold_4_19 + hold_4_20 + hold_4_21 + hold_4_22 + hold_4_23
   + hold_4_24 + hold_4_25 + hold_4_26 + hold_4_27 + hold_4_28 + hold_4_29
   + hold_4_30 + hold_4_31 + hold_4_32 <= 1
 end_constraint_4: start_4 + hold_4_1 + 2 hold_4_2 + 3 hold_4_3
   + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7 + 8 hold_4_8
   + 9 hold_4_9 + 10 hold_4_10 + 11 hold_4_11 + 12 hold_4_12 + 13 hold_4_13
   + 14 hold_4_14 + 15 hold_4_15 + 16 hold_4_16 + 17 hold_4_17
   + 18 hold_4_18 + 19 hold_4_19 + 20 hold_4_20 + 21 hold_4_21
   + 22 hold_4_22 + 23 hold_4_23 + 24 hold_4_24 + 25 hold_4_25
   + 26 hold_4_26 + 27 hold_4_27 + 28 hold_4_28 + 29 hold_4_29
   + 30 hold_4_30 + 31 hold_4_31 + 32 hold_4_32 <= 60
 hold_constraint_5: hold_5_1 + hold_5_2 + hold_5_3 + hold_5_4 + hold_5_5
   + hold_5_6 + hold_5_7 + hold_5_8 + hold_5_9 + hold_5_10 + hold_5_11
   + hold_5_12 + hold_5_13 + hold_5_14 + hold_5_15 + hold_5_16 + hold_5_17
   + hold_5_18 + hold_5_19 + hold_5_20 + hold_5_21 + hold_5_22 + hold_5_23
   + hold_5_24 + hold_5_25 + hold_5_26 + hold_5_27 + hold_5_28 + hold_5_29
   + hold_5_30 + hold_5_31 + hold_5_32 <= 1
 end_constraint_5: start_5 + hold_5_1 + 2 hold_5_2 + 3 hold_5_3
   + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7 + 8 hold_5_8
   + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12 + 13 hold_5_13
   + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16 + 17 hold_5_17
   + 18 hold_5_18 + 19 hold_5_19 + 20 hold_5_20 + 21 hold_5_21
   + 22 hold_5_22 + 23 hold_5_23 + 24 hold_5_24 + 25 hold_5_25
   + 26 hold_5_26 + 27 hold_5_27 + 28 hold_5_28 + 29 hold_5_29
   + 30 hold_5_30 + 31 hold_5_31 + 32 hold_5_32 <= 60
 hold_constraint_6: hold_6_1 + hold_6_2 + hold_6_3 + hold_6_4 + hold_6_5
   + hold_6_6 + hold_6_7 + hold_6_8 + hold_6_9 + hold_6_10 + hold_6_11
   + hold_6_12 + hold_6_13 + hold_6_14 + hold_6_15 + hold_6_16 + hold_6_17
   + hold_6_18 + hold_6_19 + hold_6_20 + hold_6_21 + hold_6_22 + hold_6_23
   + hold_6_24 + hold_6_25 + hold_6_26 + hold_6_27 + hold_6_28 + hold_6_29
   + hold_6_30 + hold_6_31 + hold_6_32 <= 1
 end_constraint_6: start_6 + hold_6_1 + 2 hold_6_2 + 3 hold_6_3
   + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7 + 8 hold_6_8
   + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 12 hold_6_12 + 13 hold_6_13
   + 14 hold_6_14 + 15 hold_6_15 + 16 hold_6_16 + 17 hold_6_17
   + 18 hold_6_18 + 19 hold_6_19 + 20 hold_6_20 + 21 hold_6_21
   + 22 hold_6_22 + 23 hold_6_23 + 24 hold_6_24 + 25 hold_6_25
   + 26 hold_6_26 + 27 hold_6_27 + 28 hold_6_28 + 29 hold_6_29
   + 30 hold_6_30 + 31 hold_6_31 + 32 hold_6_32 <= 60
 hold_constraint_7: hold_7_1 + hold_7_2 + hold_7_3 + hold_7_4 + hold_7_5
   + hold_7_6 + hold_7_7 + hold_7_8 + hold_7_9 + hold_7_10 + hold_7_11
   + hold_7_12 + hold_7_13 + hold_7_14 + hold_7_15 + hold_7_16 + hold_7_17
   + hold_7_18 + hold_7_19 + hold_7_20 + hold_7_21 + hold_7_22 + hold_7_23
   + hold_7_24 + hold_7_25 + hold_7_26 + hold_7_27 + hold_7_28 + hold_7_29
   + hold_7_30 + hold_7_31 + hold_7_32 <= 1
 end_constraint_7: start_7 + hold_7_1 + 2 hold_7_2 + 3 hold_7_3
   + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7 + 8 hold_7_8
   + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11 + 12 hold_7_12 + 13 hold_7_13
   + 14 hold_7_14 + 15 hold_7_15 + 16 hold_7_16 + 17 hold_7_17
   + 18 hold_7_18 + 19 hold_7_19 + 20 hold_7_20 + 21 hold_7_21
   + 22 hold_7_22 + 23 hold_7_23 + 24 hold_7_24 + 25 hold_7_25
   + 26 hold_7_26 + 27 hold_7_27 + 28 hold_7_28 + 29 hold_7_29
   + 30 hold_7_30 + 31 hold_7_31 + 32 hold_7_32 <= 60
 hold_constraint_8: hold_8_1 + hold_8_2 + hold_8_3 + hold_8_4 + hold_8_5
   + hold_8_6 + hold_8_7 + hold_8_8 + hold_8_9 + hold_8_10 + hold_8_11
   + hold_8_12 + hold_8_13 + hold_8_14 + hold_8_15 + hold_8_16 + hold_8_17
   + hold_8_18 + hold_8_19 + hold_8_20 + hold_8_21 + hold_8_22 + hold_8_23
   + hold_8_24 + hold_8_25 + hold_8_26 + hold_8_27 + hold_8_28 + hold_8_29
   + hold_8_30 + hold_8_31 + hold_8_32 <= 1
 end_constraint_8: start_8 + hold_8_1 + 2 hold_8_2 + 3 hold_8_3
   + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7 + 8 hold_8_8
   + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 12 hold_8_12 + 13 hold_8_13
   + 14 hold_8_14 + 15 hold_8_15 + 16 hold_8_16 + 17 hold_8_17
   + 18 hold_8_18 + 19 hold_8_19 + 20 hold_8_20 + 21 hold_8_21
   + 22 hold_8_22 + 23 hold_8_23 + 24 hold_8_24 + 25 hold_8_25
   + 26 hold_8_26 + 27 hold_8_27 + 28 hold_8_28 + 29 hold_8_29
   + 30 hold_8_30 + 31 hold_8_31 + 32 hold_8_32 <= 60
 hold_constraint_9: hold_9_1 + hold_9_2 + hold_9_3 + hold_9_4 + hold_9_5
   + hold_9_6 + hold_9_7 + hold_9_8 + hold_9_9 + hold_9_10 + hold_9_11
   + hold_9_12 + hold_9_13 + hold_9_14 + hold_9_15 + hold_9_16 + hold_9_17
   + hold_9_18 + hold_9_19 + hold_9_20 + hold_9_21 + hold_9_22 + hold_9_23
   + hold_9_24 + hold_9_25 + hold_9_26 + hold_9_27 + hold_9_28 + hold_9_29
   + hold_9_30 + hold_9_31 + hold_9_32 <= 1
 end_constraint_9: start_9 + hold_9_1 + 2 hold_9_2 + 3 hold_9_3
   + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7 + 8 hold_9_8
   + 9 hold_9_9 + 10 hold_9_10 + 11 hold_9_11 + 12 hold_9_12 + 13 hold_9_13
   + 14 hold_9_14 + 15 hold_9_15 + 16 hold_9_16 + 17 hold_9_17
   + 18 hold_9_18 + 19 hold_9_19 + 20 hold_9_20 + 21 hold_9_21
   + 22 hold_9_22 + 23 hold_9_23 + 24 hold_9_24 + 25 hold_9_25
   + 26 hold_9_26 + 27 hold_9_27 + 28 hold_9_28 + 29 hold_9_29
   + 30 hold_9_30 + 31 hold_9_31 + 32 hold_9_32 <= 60
 hold_constraint_10: hold_10_1 + hold_10_2 + hold_10_3 + hold_10_4
   + hold_10_5 + hold_10_6 + hold_10_7 + hold_10_8 + hold_10_9 + hold_10_10
   + hold_10_11 + hold_10_12 + hold_10_13 + hold_10_14 + hold_10_15
   + hold_10_16 + hold_10_17 + hold_10_18 + hold_10_19 + hold_10_20
   + hold_10_21 + hold_10_22 + hold_10_23 + hold_10_24 + hold_10_25
   + hold_10_26 + hold_10_27 + hold_10_28 + hold_10_29 + hold_10_30
   + hold_10_31 + hold_10_32 <= 1
 end_constraint_10: start_10 + hold_10_1 + 2 hold_10_2 + 3 hold_10_3
   + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7 + 8 hold_10_8
   + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11 + 12 hold_10_12
   + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15 + 16 hold_10_16
   + 17 hold_10_17 + 18 hold_10_18 + 19 hold_10_19 + 20 hold_10_20
   + 21 hold_10_21 + 22 hold_10_22 + 23 hold_10_23 + 24 hold_10_24
   + 25 hold_10_25 + 26 hold_10_26 + 27 hold_10_27 + 28 hold_10_28
   + 29 hold_10_29 + 30 hold_10_30 + 31 hold_10_31 + 32 hold_10_32 <= 60
 hold_constraint_11: hold_11_1 + hold_11_2 + hold_11_3 + hold_11_4
   + hold_11_5 + hold_11_6 + hold_11_7 + hold_11_8 + hold_11_9 + hold_11_10
   + hold_11_11 + hold_11_12 + hold_11_13 + hold_11_14 + hold_11_15
   + hold_11_16 + hold_11_17 + hold_11_18 + hold_11_19 + hold_11_20
   + hold_11_21 + hold_11_22 + hold_11_23 + hold_11_24 + hold_11_25
   + hold_11_26 + hold_11_27 + hold_11_28 + hold_11_29 + hold_11_30
   + hold_11_31 + hold_11_32 <= 1
 end_constraint_11: start_11 + hold_11_1 + 2 hold_11_2 + 3 hold_11_3
   + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7 + 8 hold_11_8
   + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11 + 12 hold_11_12
   + 13 hold_11_13 + 14 hold_11_14 + 15 hold_11_15 + 16 hold_11_16
   + 17 hold_11_17 + 18 hold_11_18 + 19 hold_11_19 + 20 hold_11_20
   + 21 hold_11_21 + 22 hold_11_22 + 23 hold_11_23 + 24 hold_11_24
   + 25 hold_11_25 + 26 hold_11_26 + 27 hold_11_27 + 28 hold_11_28
   + 29 hold_11_29 + 30 hold_11_30 + 31 hold_11_31 + 32 hold_11_32 <= 60
 hold_constraint_12: hold_12_1 + hold_12_2 + hold_12_3 + hold_12_4
   + hold_12_5 + hold_12_6 + hold_12_7 + hold_12_8 + hold_12_9 + hold_12_10
   + hold_12_11 + hold_12_12 + hold_12_13 + hold_12_14 + hold_12_15
   + hold_12_16 + hold_12_17 + hold_12_18 + hold_12_19 + hold_12_20
   + hold_12_21 + hold_12_22 + hold_12_23 + hold_12_24 + hold_12_25
   + hold_12_26 + hold_12_27 + hold_12_28 + hold_12_29 + hold_12_30
   + hold_12_31 + hold_12_32 <= 1
 end_constraint_12: start_12 + hold_12_1 + 2 hold_12_2 + 3 hold_12_3
   + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6 + 7 hold_12_7 + 8 hold_12_8
   + 9 hold_12_9 + 10 hold_12_10 + 11 hold_12_11 + 12 hold_12_12
   + 13 hold_12_13 + 14 hold_12_14 + 15 hold_12_15 + 16 hold_12_16
   + 17 hold_12_17 + 18 hold_12_18 + 19 hold_12_19 + 20 hold_12_20
   + 21 hold_12_21 + 22 hold_12_22 + 23 hold_12_23 + 24 hold_12_24
   + 25 hold_12_25 + 26 hold_12_26 + 27 hold_12_27 + 28 hold_12_28
   + 29 hold_12_29 + 30 hold_12_30 + 31 hold_12_31 + 32 hold_12_32 <= 60
 hold_constraint_13: hold_13_1 + hold_13_2 + hold_13_3 + hold_13_4
   + hold_13_5 + hold_13_6 + hold_13_7 + hold_13_8 + hold_13_9 + hold_13_10
   + hold_13_11 + hold_13_12 + hold_13_13 + hold_13_14 + hold_13_15
   + hold_13_16 + hold_13_17 + hold_13_18 + hold_13_19 + hold_13_20
   + hold_13_21 + hold_13_22 + hold_13_23 + hold_13_24 + hold_13_25
   + hold_13_26 + hold_13_27 + hold_13_28 + hold_13_29 + hold_13_30
   + hold_13_31 + hold_13_32 <= 1
 end_constraint_13: start_13 + hold_13_1 + 2 hold_13_2 + 3 hold_13_3
   + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6 + 7 hold_13_7 + 8 hold_13_8
   + 9 hold_13_9 + 10 hold_13_10 + 11 hold_13_11 + 12 hold_13_12
   + 13 hold_13_13 + 14 hold_13_14 + 15 hold_13_15 + 16 hold_13_16
   + 17 hold_13_17 + 18 hold_13_18 + 19 hold_13_19 + 20 hold_13_20
   + 21 hold_13_21 + 22 hold_13_22 + 23 hold_13_23 + 24 hold_13_24
   + 25 hold_13_25 + 26 hold_13_26 + 27 hold_13_27 + 28 hold_13_28
   + 29 hold_13_29 + 30 hold_13_30 + 31 hold_13_31 + 32 hold_13_32 <= 60
 hold_constraint_14: hold_14_1 + hold_14_2 + hold_14_3 + hold_14_4
   + hold_14_5 + hold_14_6 + hold_14_7 + hold_14_8 + hold_14_9 + hold_14_10
   + hold_14_11 + hold_14_12 + hold_14_13 + hold_14_14 + hold_14_15
   + hold_14_16 + hold_14_17 + hold_14_18 + hold_14_19 + hold_14_20
   + hold_14_21 + hold_14_22 + hold_14_23 + hold_14_24 + hold_14_25
   + hold_14_26 + hold_14_27 + hold_14_28 + hold_14_29 + hold_14_30
   + hold_14_31 + hold_14_32 <= 1
 end_constraint_14: start_14 + hold_14_1 + 2 hold_14_2 + 3 hold_14_3
   + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6 + 7 hold_14_7 + 8 hold_14_8
   + 9 hold_14_9 + 10 hold_14_10 + 11 hold_14_11 + 12 hold_14_12
   + 13 hold_14_13 + 14 hold_14_14 + 15 hold_14_15 + 16 hold_14_16
   + 17 hold_14_17 + 18 hold_14_18 + 19 hold_14_19 + 20 hold_14_20
   + 21 hold_14_21 + 22 hold_14_22 + 23 hold_14_23 + 24 hold_14_24
   + 25 hold_14_25 + 26 hold_14_26 + 27 hold_14_27 + 28 hold_14_28
   + 29 hold_14_29 + 30 hold_14_30 + 31 hold_14_31 + 32 hold_14_32 <= 60
 node_flow_constraint_nic_in_0: flow_source_nic_in_0 - flow_nic_in_0_0
   - flow_nic_in_0_1 - flow_nic_in_0_2 - flow_nic_in_0_3
   + flow_nic_out_4_nic_in_0 + flow_nic_out_5_nic_in_0
   + flow_nic_out_6_nic_in_0 + flow_nic_out_7_nic_in_0
   + flow_nic_out_8_nic_in_0 + flow_nic_out_9_nic_in_0
   + flow_nic_out_10_nic_in_0 + flow_nic_out_11_nic_in_0
   + flow_nic_out_12_nic_in_0 + flow_nic_out_13_nic_in_0
   + flow_nic_out_14_nic_in_0 = 0
 node_flow_constraint_nic_out_0: - flow_nic_out_0_sink
   - flow_nic_out_0_nic_in_4 - flow_nic_out_0_nic_in_5
   - flow_nic_out_0_nic_in_6 - flow_nic_out_0_nic_in_7
   - flow_nic_out_0_nic_in_8 - flow_nic_out_0_nic_in_9
   - flow_nic_out_0_nic_in_10 - flow_nic_out_0_nic_in_11
   - flow_nic_out_0_nic_in_12 - flow_nic_out_0_nic_in_13
   - flow_nic_out_0_nic_in_14 + flow_0_nic_out_0 + flow_1_nic_out_0
   + flow_2_nic_out_0 + flow_3_nic_out_0 = 0
 node_flow_constraint_nic_in_4: flow_nic_out_0_nic_in_4
   + flow_source_nic_in_4 - flow_nic_in_4_4 + flow_nic_out_5_nic_in_4
   + flow_nic_out_6_nic_in_4 + flow_nic_out_7_nic_in_4
   + flow_nic_out_8_nic_in_4 + flow_nic_out_9_nic_in_4
   + flow_nic_out_10_nic_in_4 + flow_nic_out_11_nic_in_4
   + flow_nic_out_12_nic_in_4 + flow_nic_out_13_nic_in_4
   + flow_nic_out_14_nic_in_4 = 0
 node_flow_constraint_nic_out_4: - flow_nic_out_4_sink
   - flow_nic_out_4_nic_in_0 - flow_nic_out_4_nic_in_5
   - flow_nic_out_4_nic_in_6 - flow_nic_out_4_nic_in_7
   - flow_nic_out_4_nic_in_8 - flow_nic_out_4_nic_in_9
   - flow_nic_out_4_nic_in_10 - flow_nic_out_4_nic_in_11
   - flow_nic_out_4_nic_in_12 - flow_nic_out_4_nic_in_13
   - flow_nic_out_4_nic_in_14 + flow_4_nic_out_4 = 0
 node_flow_constraint_nic_in_5: flow_nic_out_0_nic_in_5
   + flow_nic_out_4_nic_in_5 + flow_source_nic_in_5 - flow_nic_in_5_5
   + flow_nic_out_6_nic_in_5 + flow_nic_out_7_nic_in_5
   + flow_nic_out_8_nic_in_5 + flow_nic_out_9_nic_in_5
   + flow_nic_out_10_nic_in_5 + flow_nic_out_11_nic_in_5
   + flow_nic_out_12_nic_in_5 + flow_nic_out_13_nic_in_5
   + flow_nic_out_14_nic_in_5 = 0
 node_flow_constraint_nic_out_5: - flow_nic_out_5_sink
   - flow_nic_out_5_nic_in_0 - flow_nic_out_5_nic_in_4
   - flow_nic_out_5_nic_in_6 - flow_nic_out_5_nic_in_7
   - flow_nic_out_5_nic_in_8 - flow_nic_out_5_nic_in_9
   - flow_nic_out_5_nic_in_10 - flow_nic_out_5_nic_in_11
   - flow_nic_out_5_nic_in_12 - flow_nic_out_5_nic_in_13
   - flow_nic_out_5_nic_in_14 + flow_5_nic_out_5 = 0
 node_flow_constraint_nic_in_6: flow_nic_out_0_nic_in_6
   + flow_nic_out_4_nic_in_6 + flow_nic_out_5_nic_in_6
   + flow_source_nic_in_6 - flow_nic_in_6_6 + flow_nic_out_7_nic_in_6
   + flow_nic_out_8_nic_in_6 + flow_nic_out_9_nic_in_6
   + flow_nic_out_10_nic_in_6 + flow_nic_out_11_nic_in_6
   + flow_nic_out_12_nic_in_6 + flow_nic_out_13_nic_in_6
   + flow_nic_out_14_nic_in_6 = 0
 node_flow_constraint_nic_out_6: - flow_nic_out_6_sink
   - flow_nic_out_6_nic_in_0 - flow_nic_out_6_nic_in_4
   - flow_nic_out_6_nic_in_5 - flow_nic_out_6_nic_in_7
   - flow_nic_out_6_nic_in_8 - flow_nic_out_6_nic_in_9
   - flow_nic_out_6_nic_in_10 - flow_nic_out_6_nic_in_11
   - flow_nic_out_6_nic_in_12 - flow_nic_out_6_nic_in_13
   - flow_nic_out_6_nic_in_14 + flow_6_nic_out_6 = 0
 node_flow_constraint_nic_in_7: flow_nic_out_0_nic_in_7
   + flow_nic_out_4_nic_in_7 + flow_nic_out_5_nic_in_7
   + flow_nic_out_6_nic_in_7 + flow_source_nic_in_7 - flow_nic_in_7_7
   + flow_nic_out_8_nic_in_7 + flow_nic_out_9_nic_in_7
   + flow_nic_out_10_nic_in_7 + flow_nic_out_11_nic_in_7
   + flow_nic_out_12_nic_in_7 + flow_nic_out_13_nic_in_7
   + flow_nic_out_14_nic_in_7 = 0
 node_flow_constraint_nic_out_7: - flow_nic_out_7_sink
   - flow_nic_out_7_nic_in_0 - flow_nic_out_7_nic_in_4
   - flow_nic_out_7_nic_in_5 - flow_nic_out_7_nic_in_6
   - flow_nic_out_7_nic_in_8 - flow_nic_out_7_nic_in_9
   - flow_nic_out_7_nic_in_10 - flow_nic_out_7_nic_in_11
   - flow_nic_out_7_nic_in_12 - flow_nic_out_7_nic_in_13
   - flow_nic_out_7_nic_in_14 + flow_7_nic_out_7 = 0
 node_flow_constraint_nic_in_8: flow_nic_out_0_nic_in_8
   + flow_nic_out_4_nic_in_8 + flow_nic_out_5_nic_in_8
   + flow_nic_out_6_nic_in_8 + flow_nic_out_7_nic_in_8
   + flow_source_nic_in_8 - flow_nic_in_8_8 + flow_nic_out_9_nic_in_8
   + flow_nic_out_10_nic_in_8 + flow_nic_out_11_nic_in_8
   + flow_nic_out_12_nic_in_8 + flow_nic_out_13_nic_in_8
   + flow_nic_out_14_nic_in_8 = 0
 node_flow_constraint_nic_out_8: - flow_nic_out_8_sink
   - flow_nic_out_8_nic_in_0 - flow_nic_out_8_nic_in_4
   - flow_nic_out_8_nic_in_5 - flow_nic_out_8_nic_in_6
   - flow_nic_out_8_nic_in_7 - flow_nic_out_8_nic_in_9
   - flow_nic_out_8_nic_in_10 - flow_nic_out_8_nic_in_11
   - flow_nic_out_8_nic_in_12 - flow_nic_out_8_nic_in_13
   - flow_nic_out_8_nic_in_14 + flow_8_nic_out_8 = 0
 node_flow_constraint_nic_in_9: flow_nic_out_0_nic_in_9
   + flow_nic_out_4_nic_in_9 + flow_nic_out_5_nic_in_9
   + flow_nic_out_6_nic_in_9 + flow_nic_out_7_nic_in_9
   + flow_nic_out_8_nic_in_9 + flow_source_nic_in_9 - flow_nic_in_9_9
   + flow_nic_out_10_nic_in_9 + flow_nic_out_11_nic_in_9
   + flow_nic_out_12_nic_in_9 + flow_nic_out_13_nic_in_9
   + flow_nic_out_14_nic_in_9 = 0
 node_flow_constraint_nic_out_9: - flow_nic_out_9_sink
   - flow_nic_out_9_nic_in_0 - flow_nic_out_9_nic_in_4
   - flow_nic_out_9_nic_in_5 - flow_nic_out_9_nic_in_6
   - flow_nic_out_9_nic_in_7 - flow_nic_out_9_nic_in_8
   - flow_nic_out_9_nic_in_10 - flow_nic_out_9_nic_in_11
   - flow_nic_out_9_nic_in_12 - flow_nic_out_9_nic_in_13
   - flow_nic_out_9_nic_in_14 + flow_9_nic_out_9 = 0
 node_flow_constraint_nic_in_10: flow_nic_out_0_nic_in_10
   + flow_nic_out_4_nic_in_10 + flow_nic_out_5_nic_in_10
   + flow_nic_out_6_nic_in_10 + flow_nic_out_7_nic_in_10
   + flow_nic_out_8_nic_in_10 + flow_nic_out_9_nic_in_10
   + flow_source_nic_in_10 - flow_nic_in_10_10 + flow_nic_out_11_nic_in_10
   + flow_nic_out_12_nic_in_10 + flow_nic_out_13_nic_in_10
   + flow_nic_out_14_nic_in_10 = 0
 node_flow_constraint_nic_out_10: - flow_nic_out_10_sink
   - flow_nic_out_10_nic_in_0 - flow_nic_out_10_nic_in_4
   - flow_nic_out_10_nic_in_5 - flow_nic_out_10_nic_in_6
   - flow_nic_out_10_nic_in_7 - flow_nic_out_10_nic_in_8
   - flow_nic_out_10_nic_in_9 - flow_nic_out_10_nic_in_11
   - flow_nic_out_10_nic_in_12 - flow_nic_out_10_nic_in_13
   - flow_nic_out_10_nic_in_14 + flow_10_nic_out_10 = 0
 node_flow_constraint_nic_in_11: flow_nic_out_0_nic_in_11
   + flow_nic_out_4_nic_in_11 + flow_nic_out_5_nic_in_11
   + flow_nic_out_6_nic_in_11 + flow_nic_out_7_nic_in_11
   + flow_nic_out_8_nic_in_11 + flow_nic_out_9_nic_in_11
   + flow_nic_out_10_nic_in_11 + flow_source_nic_in_11 - flow_nic_in_11_11
   + flow_nic_out_12_nic_in_11 + flow_nic_out_13_nic_in_11
   + flow_nic_out_14_nic_in_11 = 0
 node_flow_constraint_nic_out_11: - flow_nic_out_11_sink
   - flow_nic_out_11_nic_in_0 - flow_nic_out_11_nic_in_4
   - flow_nic_out_11_nic_in_5 - flow_nic_out_11_nic_in_6
   - flow_nic_out_11_nic_in_7 - flow_nic_out_11_nic_in_8
   - flow_nic_out_11_nic_in_9 - flow_nic_out_11_nic_in_10
   - flow_nic_out_11_nic_in_12 - flow_nic_out_11_nic_in_13
   - flow_nic_out_11_nic_in_14 + flow_11_nic_out_11 = 0
 node_flow_constraint_nic_in_12: flow_nic_out_0_nic_in_12
   + flow_nic_out_4_nic_in_12 + flow_nic_out_5_nic_in_12
   + flow_nic_out_6_nic_in_12 + flow_nic_out_7_nic_in_12
   + flow_nic_out_8_nic_in_12 + flow_nic_out_9_nic_in_12
   + flow_nic_out_10_nic_in_12 + flow_nic_out_11_nic_in_12
   + flow_source_nic_in_12 - flow_nic_in_12_12 + flow_nic_out_13_nic_in_12
   + flow_nic_out_14_nic_in_12 = 0
 node_flow_constraint_nic_out_12: - flow_nic_out_12_sink
   - flow_nic_out_12_nic_in_0 - flow_nic_out_12_nic_in_4
   - flow_nic_out_12_nic_in_5 - flow_nic_out_12_nic_in_6
   - flow_nic_out_12_nic_in_7 - flow_nic_out_12_nic_in_8
   - flow_nic_out_12_nic_in_9 - flow_nic_out_12_nic_in_10
   - flow_nic_out_12_nic_in_11 - flow_nic_out_12_nic_in_13
   - flow_nic_out_12_nic_in_14 + flow_12_nic_out_12 = 0
 node_flow_constraint_nic_in_13: flow_nic_out_0_nic_in_13
   + flow_nic_out_4_nic_in_13 + flow_nic_out_5_nic_in_13
   + flow_nic_out_6_nic_in_13 + flow_nic_out_7_nic_in_13
   + flow_nic_out_8_nic_in_13 + flow_nic_out_9_nic_in_13
   + flow_nic_out_10_nic_in_13 + flow_nic_out_11_nic_in_13
   + flow_nic_out_12_nic_in_13 + flow_source_nic_in_13 - flow_nic_in_13_13
   + flow_nic_out_14_nic_in_13 = 0
 node_flow_constraint_nic_out_13: - flow_nic_out_13_sink
   - flow_nic_out_13_nic_in_0 - flow_nic_out_13_nic_in_4
   - flow_nic_out_13_nic_in_5 - flow_nic_out_13_nic_in_6
   - flow_nic_out_13_nic_in_7 - flow_nic_out_13_nic_in_8
   - flow_nic_out_13_nic_in_9 - flow_nic_out_13_nic_in_10
   - flow_nic_out_13_nic_in_11 - flow_nic_out_13_nic_in_12
   - flow_nic_out_13_nic_in_14 + flow_13_nic_out_13 = 0
 node_flow_constraint_nic_in_14: flow_nic_out_0_nic_in_14
   + flow_nic_out_4_nic_in_14 + flow_nic_out_5_nic_in_14
   + flow_nic_out_6_nic_in_14 + flow_nic_out_7_nic_in_14
   + flow_nic_out_8_nic_in_14 + flow_nic_out_9_nic_in_14
   + flow_nic_out_10_nic_in_14 + flow_nic_out_11_nic_in_14
   + flow_nic_out_12_nic_in_14 + flow_nic_out_13_nic_in_14
   + flow_source_nic_in_14 - flow_nic_in_14_14 = 0
 node_flow_constraint_nic_out_14: - flow_nic_out_14_sink
   - flow_nic_out_14_nic_in_0 - flow_nic_out_14_nic_in_4
   - flow_nic_out_14_nic_in_5 - flow_nic_out_14_nic_in_6
   - flow_nic_out_14_nic_in_7 - flow_nic_out_14_nic_in_8
   - flow_nic_out_14_nic_in_9 - flow_nic_out_14_nic_in_10
   - flow_nic_out_14_nic_in_11 - flow_nic_out_14_nic_in_12
   - flow_nic_out_14_nic_in_13 + flow_14_nic_out_14 = 0
 node_flow_constraint_0: flow_nic_in_0_0 - flow_0_nic_out_0 - flow_0_1
   - flow_0_2 - flow_0_3 + flow_1_0 + flow_2_0 + flow_3_0 = 0
 node_flow_constraint_1: flow_nic_in_0_1 - flow_1_nic_out_0 + flow_0_1
   - flow_1_0 - flow_1_2 - flow_1_3 + flow_2_1 + flow_3_1 = 0
 node_flow_constraint_2: flow_nic_in_0_2 - flow_2_nic_out_0 + flow_0_2
   + flow_1_2 - flow_2_0 - flow_2_1 - flow_2_3 + flow_3_2 = 0
 node_flow_constraint_3: flow_nic_in_0_3 - flow_3_nic_out_0 + flow_0_3
   + flow_1_3 + flow_2_3 - flow_3_0 - flow_3_1 - flow_3_2 = 0
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
 constr_is_start_zero_0: start_0 + 60 is_start_zero_0_for_source_nic_in_0
   <= 60
 constr_is_start_zero_1: start_1 + 60 is_start_zero_1_for_source_nic_in_0
   <= 60
 constr_is_start_zero_2: start_2 + 60 is_start_zero_2_for_source_nic_in_0
   <= 60
 constr_is_start_zero_3: start_3 + 60 is_start_zero_3_for_source_nic_in_0
   <= 60
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
   + 26 hold_0_26 + 27 hold_0_27 + 28 hold_0_28 + 29 hold_0_29
   + 30 hold_0_30 + 31 hold_0_31 + 32 hold_0_32 - 60 is_end_complete_0
   >= 0
 constr_is_end_complete_1: start_1 + hold_1_1 + 2 hold_1_2 + 3 hold_1_3
   + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7 + 8 hold_1_8
   + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 + 12 hold_1_12 + 13 hold_1_13
   + 14 hold_1_14 + 15 hold_1_15 + 16 hold_1_16 + 17 hold_1_17
   + 18 hold_1_18 + 19 hold_1_19 + 20 hold_1_20 + 21 hold_1_21
   + 22 hold_1_22 + 23 hold_1_23 + 24 hold_1_24 + 25 hold_1_25
   + 26 hold_1_26 + 27 hold_1_27 + 28 hold_1_28 + 29 hold_1_29
   + 30 hold_1_30 + 31 hold_1_31 + 32 hold_1_32 - 60 is_end_complete_1
   >= 0
 constr_is_end_complete_2: start_2 + hold_2_1 + 2 hold_2_2 + 3 hold_2_3
   + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7 + 8 hold_2_8
   + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 12 hold_2_12 + 13 hold_2_13
   + 14 hold_2_14 + 15 hold_2_15 + 16 hold_2_16 + 17 hold_2_17
   + 18 hold_2_18 + 19 hold_2_19 + 20 hold_2_20 + 21 hold_2_21
   + 22 hold_2_22 + 23 hold_2_23 + 24 hold_2_24 + 25 hold_2_25
   + 26 hold_2_26 + 27 hold_2_27 + 28 hold_2_28 + 29 hold_2_29
   + 30 hold_2_30 + 31 hold_2_31 + 32 hold_2_32 - 60 is_end_complete_2
   >= 0
 constr_is_end_complete_3: start_3 + hold_3_1 + 2 hold_3_2 + 3 hold_3_3
   + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7 + 8 hold_3_8
   + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11 + 12 hold_3_12 + 13 hold_3_13
   + 14 hold_3_14 + 15 hold_3_15 + 16 hold_3_16 + 17 hold_3_17
   + 18 hold_3_18 + 19 hold_3_19 + 20 hold_3_20 + 21 hold_3_21
   + 22 hold_3_22 + 23 hold_3_23 + 24 hold_3_24 + 25 hold_3_25
   + 26 hold_3_26 + 27 hold_3_27 + 28 hold_3_28 + 29 hold_3_29
   + 30 hold_3_30 + 31 hold_3_31 + 32 hold_3_32 - 60 is_end_complete_3
   >= 0
 constr_edge_switch_end_max_0_OR: switch_nic_out_0_sink - is_end_complete_0
   - is_end_complete_1 - is_end_complete_2 - is_end_complete_3 <= 0
 constr_link_match_1_0_4: 60 switch_nic_out_0_nic_in_4 - group_start_4
   + group_end_0 <= 60
 constr_link_match_2_0_4: 60 switch_nic_out_0_nic_in_4 + group_start_4
   - group_end_0 <= 60
 constr_link_match_1_0_5: 60 switch_nic_out_0_nic_in_5 + group_end_0
   - group_start_5 <= 60
 constr_link_match_2_0_5: 60 switch_nic_out_0_nic_in_5 - group_end_0
   + group_start_5 <= 60
 constr_link_match_1_0_6: 60 switch_nic_out_0_nic_in_6 + group_end_0
   - group_start_6 <= 60
 constr_link_match_2_0_6: 60 switch_nic_out_0_nic_in_6 - group_end_0
   + group_start_6 <= 60
 constr_link_match_1_0_7: 60 switch_nic_out_0_nic_in_7 + group_end_0
   - group_start_7 <= 60
 constr_link_match_2_0_7: 60 switch_nic_out_0_nic_in_7 - group_end_0
   + group_start_7 <= 60
 constr_link_match_1_0_8: 60 switch_nic_out_0_nic_in_8 + group_end_0
   - group_start_8 <= 60
 constr_link_match_2_0_8: 60 switch_nic_out_0_nic_in_8 - group_end_0
   + group_start_8 <= 60
 constr_link_match_1_0_9: 60 switch_nic_out_0_nic_in_9 + group_end_0
   - group_start_9 <= 60
 constr_link_match_2_0_9: 60 switch_nic_out_0_nic_in_9 - group_end_0
   + group_start_9 <= 60
 constr_link_match_1_0_10: 60 switch_nic_out_0_nic_in_10 + group_end_0
   - group_start_10 <= 60
 constr_link_match_2_0_10: 60 switch_nic_out_0_nic_in_10 - group_end_0
   + group_start_10 <= 60
 constr_link_match_1_0_11: 60 switch_nic_out_0_nic_in_11 + group_end_0
   - group_start_11 <= 60
 constr_link_match_2_0_11: 60 switch_nic_out_0_nic_in_11 - group_end_0
   + group_start_11 <= 60
 constr_link_match_1_0_12: 60 switch_nic_out_0_nic_in_12 + group_end_0
   - group_start_12 <= 60
 constr_link_match_2_0_12: 60 switch_nic_out_0_nic_in_12 - group_end_0
   + group_start_12 <= 60
 constr_link_match_1_0_13: 60 switch_nic_out_0_nic_in_13 + group_end_0
   - group_start_13 <= 60
 constr_link_match_2_0_13: 60 switch_nic_out_0_nic_in_13 - group_end_0
   + group_start_13 <= 60
 constr_link_match_1_0_14: 60 switch_nic_out_0_nic_in_14 + group_end_0
   - group_start_14 <= 60
 constr_link_match_2_0_14: 60 switch_nic_out_0_nic_in_14 - group_end_0
   + group_start_14 <= 60
 constr_pp_permissionA_0_1: start_0 - start_1 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 8 hold_0_8 + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12
   + 13 hold_0_13 + 14 hold_0_14 + 15 hold_0_15 + 16 hold_0_16
   + 17 hold_0_17 + 18 hold_0_18 + 19 hold_0_19 + 20 hold_0_20
   + 21 hold_0_21 + 22 hold_0_22 + 23 hold_0_23 + 24 hold_0_24
   + 25 hold_0_25 + 26 hold_0_26 + 27 hold_0_27 + 28 hold_0_28
   + 29 hold_0_29 + 30 hold_0_30 + 31 hold_0_31 + 32 hold_0_32
   + 60 pp_permission_0_1 <= 60
 constr_pp_permissionB_0_1: - start_0 + start_1 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11 - 12 hold_0_12
   - 13 hold_0_13 - 14 hold_0_14 - 15 hold_0_15 - 16 hold_0_16
   - 17 hold_0_17 - 18 hold_0_18 - 19 hold_0_19 - 20 hold_0_20
   - 21 hold_0_21 - 22 hold_0_22 - 23 hold_0_23 - 24 hold_0_24
   - 25 hold_0_25 - 26 hold_0_26 - 27 hold_0_27 - 28 hold_0_28
   - 29 hold_0_29 - 30 hold_0_30 - 31 hold_0_31 - 32 hold_0_32
   + 60 pp_permission_0_1 <= 60
 constr_internal_compute_0_1: switch_0_1 - group_tp_active_0
   - pp_permission_0_1 <= 0
 constr_pp_permissionA_0_2: start_0 - start_2 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 8 hold_0_8 + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12
   + 13 hold_0_13 + 14 hold_0_14 + 15 hold_0_15 + 16 hold_0_16
   + 17 hold_0_17 + 18 hold_0_18 + 19 hold_0_19 + 20 hold_0_20
   + 21 hold_0_21 + 22 hold_0_22 + 23 hold_0_23 + 24 hold_0_24
   + 25 hold_0_25 + 26 hold_0_26 + 27 hold_0_27 + 28 hold_0_28
   + 29 hold_0_29 + 30 hold_0_30 + 31 hold_0_31 + 32 hold_0_32
   + 60 pp_permission_0_2 <= 60
 constr_pp_permissionB_0_2: - start_0 + start_2 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11 - 12 hold_0_12
   - 13 hold_0_13 - 14 hold_0_14 - 15 hold_0_15 - 16 hold_0_16
   - 17 hold_0_17 - 18 hold_0_18 - 19 hold_0_19 - 20 hold_0_20
   - 21 hold_0_21 - 22 hold_0_22 - 23 hold_0_23 - 24 hold_0_24
   - 25 hold_0_25 - 26 hold_0_26 - 27 hold_0_27 - 28 hold_0_28
   - 29 hold_0_29 - 30 hold_0_30 - 31 hold_0_31 - 32 hold_0_32
   + 60 pp_permission_0_2 <= 60
 constr_internal_compute_0_2: switch_0_2 - group_tp_active_0
   - pp_permission_0_2 <= 0
 constr_pp_permissionA_0_3: start_0 - start_3 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 8 hold_0_8 + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12
   + 13 hold_0_13 + 14 hold_0_14 + 15 hold_0_15 + 16 hold_0_16
   + 17 hold_0_17 + 18 hold_0_18 + 19 hold_0_19 + 20 hold_0_20
   + 21 hold_0_21 + 22 hold_0_22 + 23 hold_0_23 + 24 hold_0_24
   + 25 hold_0_25 + 26 hold_0_26 + 27 hold_0_27 + 28 hold_0_28
   + 29 hold_0_29 + 30 hold_0_30 + 31 hold_0_31 + 32 hold_0_32
   + 60 pp_permission_0_3 <= 60
 constr_pp_permissionB_0_3: - start_0 + start_3 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11 - 12 hold_0_12
   - 13 hold_0_13 - 14 hold_0_14 - 15 hold_0_15 - 16 hold_0_16
   - 17 hold_0_17 - 18 hold_0_18 - 19 hold_0_19 - 20 hold_0_20
   - 21 hold_0_21 - 22 hold_0_22 - 23 hold_0_23 - 24 hold_0_24
   - 25 hold_0_25 - 26 hold_0_26 - 27 hold_0_27 - 28 hold_0_28
   - 29 hold_0_29 - 30 hold_0_30 - 31 hold_0_31 - 32 hold_0_32
   + 60 pp_permission_0_3 <= 60
 constr_internal_compute_0_3: switch_0_3 - group_tp_active_0
   - pp_permission_0_3 <= 0
 constr_pp_permissionA_1_0: - start_0 + start_1 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 8 hold_1_8 + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 + 12 hold_1_12
   + 13 hold_1_13 + 14 hold_1_14 + 15 hold_1_15 + 16 hold_1_16
   + 17 hold_1_17 + 18 hold_1_18 + 19 hold_1_19 + 20 hold_1_20
   + 21 hold_1_21 + 22 hold_1_22 + 23 hold_1_23 + 24 hold_1_24
   + 25 hold_1_25 + 26 hold_1_26 + 27 hold_1_27 + 28 hold_1_28
   + 29 hold_1_29 + 30 hold_1_30 + 31 hold_1_31 + 32 hold_1_32
   + 60 pp_permission_1_0 <= 60
 constr_pp_permissionB_1_0: start_0 - start_1 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11 - 12 hold_1_12
   - 13 hold_1_13 - 14 hold_1_14 - 15 hold_1_15 - 16 hold_1_16
   - 17 hold_1_17 - 18 hold_1_18 - 19 hold_1_19 - 20 hold_1_20
   - 21 hold_1_21 - 22 hold_1_22 - 23 hold_1_23 - 24 hold_1_24
   - 25 hold_1_25 - 26 hold_1_26 - 27 hold_1_27 - 28 hold_1_28
   - 29 hold_1_29 - 30 hold_1_30 - 31 hold_1_31 - 32 hold_1_32
   + 60 pp_permission_1_0 <= 60
 constr_internal_compute_1_0: switch_1_0 - group_tp_active_0
   - pp_permission_1_0 <= 0
 constr_pp_permissionA_1_2: start_1 - start_2 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 8 hold_1_8 + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 + 12 hold_1_12
   + 13 hold_1_13 + 14 hold_1_14 + 15 hold_1_15 + 16 hold_1_16
   + 17 hold_1_17 + 18 hold_1_18 + 19 hold_1_19 + 20 hold_1_20
   + 21 hold_1_21 + 22 hold_1_22 + 23 hold_1_23 + 24 hold_1_24
   + 25 hold_1_25 + 26 hold_1_26 + 27 hold_1_27 + 28 hold_1_28
   + 29 hold_1_29 + 30 hold_1_30 + 31 hold_1_31 + 32 hold_1_32
   + 60 pp_permission_1_2 <= 60
 constr_pp_permissionB_1_2: - start_1 + start_2 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11 - 12 hold_1_12
   - 13 hold_1_13 - 14 hold_1_14 - 15 hold_1_15 - 16 hold_1_16
   - 17 hold_1_17 - 18 hold_1_18 - 19 hold_1_19 - 20 hold_1_20
   - 21 hold_1_21 - 22 hold_1_22 - 23 hold_1_23 - 24 hold_1_24
   - 25 hold_1_25 - 26 hold_1_26 - 27 hold_1_27 - 28 hold_1_28
   - 29 hold_1_29 - 30 hold_1_30 - 31 hold_1_31 - 32 hold_1_32
   + 60 pp_permission_1_2 <= 60
 constr_internal_compute_1_2: switch_1_2 - group_tp_active_0
   - pp_permission_1_2 <= 0
 constr_pp_permissionA_1_3: start_1 - start_3 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 8 hold_1_8 + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 + 12 hold_1_12
   + 13 hold_1_13 + 14 hold_1_14 + 15 hold_1_15 + 16 hold_1_16
   + 17 hold_1_17 + 18 hold_1_18 + 19 hold_1_19 + 20 hold_1_20
   + 21 hold_1_21 + 22 hold_1_22 + 23 hold_1_23 + 24 hold_1_24
   + 25 hold_1_25 + 26 hold_1_26 + 27 hold_1_27 + 28 hold_1_28
   + 29 hold_1_29 + 30 hold_1_30 + 31 hold_1_31 + 32 hold_1_32
   + 60 pp_permission_1_3 <= 60
 constr_pp_permissionB_1_3: - start_1 + start_3 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11 - 12 hold_1_12
   - 13 hold_1_13 - 14 hold_1_14 - 15 hold_1_15 - 16 hold_1_16
   - 17 hold_1_17 - 18 hold_1_18 - 19 hold_1_19 - 20 hold_1_20
   - 21 hold_1_21 - 22 hold_1_22 - 23 hold_1_23 - 24 hold_1_24
   - 25 hold_1_25 - 26 hold_1_26 - 27 hold_1_27 - 28 hold_1_28
   - 29 hold_1_29 - 30 hold_1_30 - 31 hold_1_31 - 32 hold_1_32
   + 60 pp_permission_1_3 <= 60
 constr_internal_compute_1_3: switch_1_3 - group_tp_active_0
   - pp_permission_1_3 <= 0
 constr_pp_permissionA_2_0: - start_0 + start_2 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 12 hold_2_12
   + 13 hold_2_13 + 14 hold_2_14 + 15 hold_2_15 + 16 hold_2_16
   + 17 hold_2_17 + 18 hold_2_18 + 19 hold_2_19 + 20 hold_2_20
   + 21 hold_2_21 + 22 hold_2_22 + 23 hold_2_23 + 24 hold_2_24
   + 25 hold_2_25 + 26 hold_2_26 + 27 hold_2_27 + 28 hold_2_28
   + 29 hold_2_29 + 30 hold_2_30 + 31 hold_2_31 + 32 hold_2_32
   + 60 pp_permission_2_0 <= 60
 constr_pp_permissionB_2_0: start_0 - start_2 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 - 12 hold_2_12
   - 13 hold_2_13 - 14 hold_2_14 - 15 hold_2_15 - 16 hold_2_16
   - 17 hold_2_17 - 18 hold_2_18 - 19 hold_2_19 - 20 hold_2_20
   - 21 hold_2_21 - 22 hold_2_22 - 23 hold_2_23 - 24 hold_2_24
   - 25 hold_2_25 - 26 hold_2_26 - 27 hold_2_27 - 28 hold_2_28
   - 29 hold_2_29 - 30 hold_2_30 - 31 hold_2_31 - 32 hold_2_32
   + 60 pp_permission_2_0 <= 60
 constr_internal_compute_2_0: switch_2_0 - group_tp_active_0
   - pp_permission_2_0 <= 0
 constr_pp_permissionA_2_1: - start_1 + start_2 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 12 hold_2_12
   + 13 hold_2_13 + 14 hold_2_14 + 15 hold_2_15 + 16 hold_2_16
   + 17 hold_2_17 + 18 hold_2_18 + 19 hold_2_19 + 20 hold_2_20
   + 21 hold_2_21 + 22 hold_2_22 + 23 hold_2_23 + 24 hold_2_24
   + 25 hold_2_25 + 26 hold_2_26 + 27 hold_2_27 + 28 hold_2_28
   + 29 hold_2_29 + 30 hold_2_30 + 31 hold_2_31 + 32 hold_2_32
   + 60 pp_permission_2_1 <= 60
 constr_pp_permissionB_2_1: start_1 - start_2 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 - 12 hold_2_12
   - 13 hold_2_13 - 14 hold_2_14 - 15 hold_2_15 - 16 hold_2_16
   - 17 hold_2_17 - 18 hold_2_18 - 19 hold_2_19 - 20 hold_2_20
   - 21 hold_2_21 - 22 hold_2_22 - 23 hold_2_23 - 24 hold_2_24
   - 25 hold_2_25 - 26 hold_2_26 - 27 hold_2_27 - 28 hold_2_28
   - 29 hold_2_29 - 30 hold_2_30 - 31 hold_2_31 - 32 hold_2_32
   + 60 pp_permission_2_1 <= 60
 constr_internal_compute_2_1: switch_2_1 - group_tp_active_0
   - pp_permission_2_1 <= 0
 constr_pp_permissionA_2_3: start_2 - start_3 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 12 hold_2_12
   + 13 hold_2_13 + 14 hold_2_14 + 15 hold_2_15 + 16 hold_2_16
   + 17 hold_2_17 + 18 hold_2_18 + 19 hold_2_19 + 20 hold_2_20
   + 21 hold_2_21 + 22 hold_2_22 + 23 hold_2_23 + 24 hold_2_24
   + 25 hold_2_25 + 26 hold_2_26 + 27 hold_2_27 + 28 hold_2_28
   + 29 hold_2_29 + 30 hold_2_30 + 31 hold_2_31 + 32 hold_2_32
   + 60 pp_permission_2_3 <= 60
 constr_pp_permissionB_2_3: - start_2 + start_3 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 - 12 hold_2_12
   - 13 hold_2_13 - 14 hold_2_14 - 15 hold_2_15 - 16 hold_2_16
   - 17 hold_2_17 - 18 hold_2_18 - 19 hold_2_19 - 20 hold_2_20
   - 21 hold_2_21 - 22 hold_2_22 - 23 hold_2_23 - 24 hold_2_24
   - 25 hold_2_25 - 26 hold_2_26 - 27 hold_2_27 - 28 hold_2_28
   - 29 hold_2_29 - 30 hold_2_30 - 31 hold_2_31 - 32 hold_2_32
   + 60 pp_permission_2_3 <= 60
 constr_internal_compute_2_3: switch_2_3 - group_tp_active_0
   - pp_permission_2_3 <= 0
 constr_pp_permissionA_3_0: - start_0 + start_3 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 8 hold_3_8 + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11 + 12 hold_3_12
   + 13 hold_3_13 + 14 hold_3_14 + 15 hold_3_15 + 16 hold_3_16
   + 17 hold_3_17 + 18 hold_3_18 + 19 hold_3_19 + 20 hold_3_20
   + 21 hold_3_21 + 22 hold_3_22 + 23 hold_3_23 + 24 hold_3_24
   + 25 hold_3_25 + 26 hold_3_26 + 27 hold_3_27 + 28 hold_3_28
   + 29 hold_3_29 + 30 hold_3_30 + 31 hold_3_31 + 32 hold_3_32
   + 60 pp_permission_3_0 <= 60
 constr_pp_permissionB_3_0: start_0 - start_3 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   - 8 hold_3_8 - 9 hold_3_9 - 10 hold_3_10 - 11 hold_3_11 - 12 hold_3_12
   - 13 hold_3_13 - 14 hold_3_14 - 15 hold_3_15 - 16 hold_3_16
   - 17 hold_3_17 - 18 hold_3_18 - 19 hold_3_19 - 20 hold_3_20
   - 21 hold_3_21 - 22 hold_3_22 - 23 hold_3_23 - 24 hold_3_24
   - 25 hold_3_25 - 26 hold_3_26 - 27 hold_3_27 - 28 hold_3_28
   - 29 hold_3_29 - 30 hold_3_30 - 31 hold_3_31 - 32 hold_3_32
   + 60 pp_permission_3_0 <= 60
 constr_internal_compute_3_0: switch_3_0 - group_tp_active_0
   - pp_permission_3_0 <= 0
 constr_pp_permissionA_3_1: - start_1 + start_3 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 8 hold_3_8 + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11 + 12 hold_3_12
   + 13 hold_3_13 + 14 hold_3_14 + 15 hold_3_15 + 16 hold_3_16
   + 17 hold_3_17 + 18 hold_3_18 + 19 hold_3_19 + 20 hold_3_20
   + 21 hold_3_21 + 22 hold_3_22 + 23 hold_3_23 + 24 hold_3_24
   + 25 hold_3_25 + 26 hold_3_26 + 27 hold_3_27 + 28 hold_3_28
   + 29 hold_3_29 + 30 hold_3_30 + 31 hold_3_31 + 32 hold_3_32
   + 60 pp_permission_3_1 <= 60
 constr_pp_permissionB_3_1: start_1 - start_3 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   - 8 hold_3_8 - 9 hold_3_9 - 10 hold_3_10 - 11 hold_3_11 - 12 hold_3_12
   - 13 hold_3_13 - 14 hold_3_14 - 15 hold_3_15 - 16 hold_3_16
   - 17 hold_3_17 - 18 hold_3_18 - 19 hold_3_19 - 20 hold_3_20
   - 21 hold_3_21 - 22 hold_3_22 - 23 hold_3_23 - 24 hold_3_24
   - 25 hold_3_25 - 26 hold_3_26 - 27 hold_3_27 - 28 hold_3_28
   - 29 hold_3_29 - 30 hold_3_30 - 31 hold_3_31 - 32 hold_3_32
   + 60 pp_permission_3_1 <= 60
 constr_internal_compute_3_1: switch_3_1 - group_tp_active_0
   - pp_permission_3_1 <= 0
 constr_pp_permissionA_3_2: - start_2 + start_3 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 8 hold_3_8 + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11 + 12 hold_3_12
   + 13 hold_3_13 + 14 hold_3_14 + 15 hold_3_15 + 16 hold_3_16
   + 17 hold_3_17 + 18 hold_3_18 + 19 hold_3_19 + 20 hold_3_20
   + 21 hold_3_21 + 22 hold_3_22 + 23 hold_3_23 + 24 hold_3_24
   + 25 hold_3_25 + 26 hold_3_26 + 27 hold_3_27 + 28 hold_3_28
   + 29 hold_3_29 + 30 hold_3_30 + 31 hold_3_31 + 32 hold_3_32
   + 60 pp_permission_3_2 <= 60
 constr_pp_permissionB_3_2: start_2 - start_3 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   - 8 hold_3_8 - 9 hold_3_9 - 10 hold_3_10 - 11 hold_3_11 - 12 hold_3_12
   - 13 hold_3_13 - 14 hold_3_14 - 15 hold_3_15 - 16 hold_3_16
   - 17 hold_3_17 - 18 hold_3_18 - 19 hold_3_19 - 20 hold_3_20
   - 21 hold_3_21 - 22 hold_3_22 - 23 hold_3_23 - 24 hold_3_24
   - 25 hold_3_25 - 26 hold_3_26 - 27 hold_3_27 - 28 hold_3_28
   - 29 hold_3_29 - 30 hold_3_30 - 31 hold_3_31 - 32 hold_3_32
   + 60 pp_permission_3_2 <= 60
 constr_internal_compute_3_2: switch_3_2 - group_tp_active_0
   - pp_permission_3_2 <= 0
 constr_is_start_zero_4: start_4 + 60 is_start_zero_4_for_source_nic_in_4
   <= 60
 constr_edge_switch_start_zero_4_OR: switch_source_nic_in_4
   - is_start_zero_4_for_source_nic_in_4 <= 0
 constr_is_end_complete_4: start_4 + hold_4_1 + 2 hold_4_2 + 3 hold_4_3
   + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7 + 8 hold_4_8
   + 9 hold_4_9 + 10 hold_4_10 + 11 hold_4_11 + 12 hold_4_12 + 13 hold_4_13
   + 14 hold_4_14 + 15 hold_4_15 + 16 hold_4_16 + 17 hold_4_17
   + 18 hold_4_18 + 19 hold_4_19 + 20 hold_4_20 + 21 hold_4_21
   + 22 hold_4_22 + 23 hold_4_23 + 24 hold_4_24 + 25 hold_4_25
   + 26 hold_4_26 + 27 hold_4_27 + 28 hold_4_28 + 29 hold_4_29
   + 30 hold_4_30 + 31 hold_4_31 + 32 hold_4_32 - 60 is_end_complete_4
   >= 0
 constr_edge_switch_end_max_4_OR: switch_nic_out_4_sink - is_end_complete_4
   <= 0
 constr_link_match_1_4_0: 60 switch_nic_out_4_nic_in_0 - group_start_0
   + group_end_4 <= 60
 constr_link_match_2_4_0: 60 switch_nic_out_4_nic_in_0 + group_start_0
   - group_end_4 <= 60
 constr_link_match_1_4_5: 60 switch_nic_out_4_nic_in_5 - group_start_5
   + group_end_4 <= 60
 constr_link_match_2_4_5: 60 switch_nic_out_4_nic_in_5 + group_start_5
   - group_end_4 <= 60
 constr_link_match_1_4_6: 60 switch_nic_out_4_nic_in_6 - group_start_6
   + group_end_4 <= 60
 constr_link_match_2_4_6: 60 switch_nic_out_4_nic_in_6 + group_start_6
   - group_end_4 <= 60
 constr_link_match_1_4_7: 60 switch_nic_out_4_nic_in_7 - group_start_7
   + group_end_4 <= 60
 constr_link_match_2_4_7: 60 switch_nic_out_4_nic_in_7 + group_start_7
   - group_end_4 <= 60
 constr_link_match_1_4_8: 60 switch_nic_out_4_nic_in_8 - group_start_8
   + group_end_4 <= 60
 constr_link_match_2_4_8: 60 switch_nic_out_4_nic_in_8 + group_start_8
   - group_end_4 <= 60
 constr_link_match_1_4_9: 60 switch_nic_out_4_nic_in_9 - group_start_9
   + group_end_4 <= 60
 constr_link_match_2_4_9: 60 switch_nic_out_4_nic_in_9 + group_start_9
   - group_end_4 <= 60
 constr_link_match_1_4_10: 60 switch_nic_out_4_nic_in_10 - group_start_10
   + group_end_4 <= 60
 constr_link_match_2_4_10: 60 switch_nic_out_4_nic_in_10 + group_start_10
   - group_end_4 <= 60
 constr_link_match_1_4_11: 60 switch_nic_out_4_nic_in_11 - group_start_11
   + group_end_4 <= 60
 constr_link_match_2_4_11: 60 switch_nic_out_4_nic_in_11 + group_start_11
   - group_end_4 <= 60
 constr_link_match_1_4_12: 60 switch_nic_out_4_nic_in_12 - group_start_12
   + group_end_4 <= 60
 constr_link_match_2_4_12: 60 switch_nic_out_4_nic_in_12 + group_start_12
   - group_end_4 <= 60
 constr_link_match_1_4_13: 60 switch_nic_out_4_nic_in_13 - group_start_13
   + group_end_4 <= 60
 constr_link_match_2_4_13: 60 switch_nic_out_4_nic_in_13 + group_start_13
   - group_end_4 <= 60
 constr_link_match_1_4_14: 60 switch_nic_out_4_nic_in_14 - group_start_14
   + group_end_4 <= 60
 constr_link_match_2_4_14: 60 switch_nic_out_4_nic_in_14 + group_start_14
   - group_end_4 <= 60
 constr_is_start_zero_5: start_5 + 60 is_start_zero_5_for_source_nic_in_5
   <= 60
 constr_edge_switch_start_zero_5_OR: switch_source_nic_in_5
   - is_start_zero_5_for_source_nic_in_5 <= 0
 constr_is_end_complete_5: start_5 + hold_5_1 + 2 hold_5_2 + 3 hold_5_3
   + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7 + 8 hold_5_8
   + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12 + 13 hold_5_13
   + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16 + 17 hold_5_17
   + 18 hold_5_18 + 19 hold_5_19 + 20 hold_5_20 + 21 hold_5_21
   + 22 hold_5_22 + 23 hold_5_23 + 24 hold_5_24 + 25 hold_5_25
   + 26 hold_5_26 + 27 hold_5_27 + 28 hold_5_28 + 29 hold_5_29
   + 30 hold_5_30 + 31 hold_5_31 + 32 hold_5_32 - 60 is_end_complete_5
   >= 0
 constr_edge_switch_end_max_5_OR: switch_nic_out_5_sink - is_end_complete_5
   <= 0
 constr_link_match_1_5_0: 60 switch_nic_out_5_nic_in_0 - group_start_0
   + group_end_5 <= 60
 constr_link_match_2_5_0: 60 switch_nic_out_5_nic_in_0 + group_start_0
   - group_end_5 <= 60
 constr_link_match_1_5_4: 60 switch_nic_out_5_nic_in_4 - group_start_4
   + group_end_5 <= 60
 constr_link_match_2_5_4: 60 switch_nic_out_5_nic_in_4 + group_start_4
   - group_end_5 <= 60
 constr_link_match_1_5_6: 60 switch_nic_out_5_nic_in_6 - group_start_6
   + group_end_5 <= 60
 constr_link_match_2_5_6: 60 switch_nic_out_5_nic_in_6 + group_start_6
   - group_end_5 <= 60
 constr_link_match_1_5_7: 60 switch_nic_out_5_nic_in_7 - group_start_7
   + group_end_5 <= 60
 constr_link_match_2_5_7: 60 switch_nic_out_5_nic_in_7 + group_start_7
   - group_end_5 <= 60
 constr_link_match_1_5_8: 60 switch_nic_out_5_nic_in_8 - group_start_8
   + group_end_5 <= 60
 constr_link_match_2_5_8: 60 switch_nic_out_5_nic_in_8 + group_start_8
   - group_end_5 <= 60
 constr_link_match_1_5_9: 60 switch_nic_out_5_nic_in_9 - group_start_9
   + group_end_5 <= 60
 constr_link_match_2_5_9: 60 switch_nic_out_5_nic_in_9 + group_start_9
   - group_end_5 <= 60
 constr_link_match_1_5_10: 60 switch_nic_out_5_nic_in_10 - group_start_10
   + group_end_5 <= 60
 constr_link_match_2_5_10: 60 switch_nic_out_5_nic_in_10 + group_start_10
   - group_end_5 <= 60
 constr_link_match_1_5_11: 60 switch_nic_out_5_nic_in_11 - group_start_11
   + group_end_5 <= 60
 constr_link_match_2_5_11: 60 switch_nic_out_5_nic_in_11 + group_start_11
   - group_end_5 <= 60
 constr_link_match_1_5_12: 60 switch_nic_out_5_nic_in_12 - group_start_12
   + group_end_5 <= 60
 constr_link_match_2_5_12: 60 switch_nic_out_5_nic_in_12 + group_start_12
   - group_end_5 <= 60
 constr_link_match_1_5_13: 60 switch_nic_out_5_nic_in_13 - group_start_13
   + group_end_5 <= 60
 constr_link_match_2_5_13: 60 switch_nic_out_5_nic_in_13 + group_start_13
   - group_end_5 <= 60
 constr_link_match_1_5_14: 60 switch_nic_out_5_nic_in_14 - group_start_14
   + group_end_5 <= 60
 constr_link_match_2_5_14: 60 switch_nic_out_5_nic_in_14 + group_start_14
   - group_end_5 <= 60
 constr_is_start_zero_6: start_6 + 60 is_start_zero_6_for_source_nic_in_6
   <= 60
 constr_edge_switch_start_zero_6_OR: switch_source_nic_in_6
   - is_start_zero_6_for_source_nic_in_6 <= 0
 constr_is_end_complete_6: start_6 + hold_6_1 + 2 hold_6_2 + 3 hold_6_3
   + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7 + 8 hold_6_8
   + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 12 hold_6_12 + 13 hold_6_13
   + 14 hold_6_14 + 15 hold_6_15 + 16 hold_6_16 + 17 hold_6_17
   + 18 hold_6_18 + 19 hold_6_19 + 20 hold_6_20 + 21 hold_6_21
   + 22 hold_6_22 + 23 hold_6_23 + 24 hold_6_24 + 25 hold_6_25
   + 26 hold_6_26 + 27 hold_6_27 + 28 hold_6_28 + 29 hold_6_29
   + 30 hold_6_30 + 31 hold_6_31 + 32 hold_6_32 - 60 is_end_complete_6
   >= 0
 constr_edge_switch_end_max_6_OR: switch_nic_out_6_sink - is_end_complete_6
   <= 0
 constr_link_match_1_6_0: 60 switch_nic_out_6_nic_in_0 - group_start_0
   + group_end_6 <= 60
 constr_link_match_2_6_0: 60 switch_nic_out_6_nic_in_0 + group_start_0
   - group_end_6 <= 60
 constr_link_match_1_6_4: 60 switch_nic_out_6_nic_in_4 - group_start_4
   + group_end_6 <= 60
 constr_link_match_2_6_4: 60 switch_nic_out_6_nic_in_4 + group_start_4
   - group_end_6 <= 60
 constr_link_match_1_6_5: 60 switch_nic_out_6_nic_in_5 - group_start_5
   + group_end_6 <= 60
 constr_link_match_2_6_5: 60 switch_nic_out_6_nic_in_5 + group_start_5
   - group_end_6 <= 60
 constr_link_match_1_6_7: 60 switch_nic_out_6_nic_in_7 - group_start_7
   + group_end_6 <= 60
 constr_link_match_2_6_7: 60 switch_nic_out_6_nic_in_7 + group_start_7
   - group_end_6 <= 60
 constr_link_match_1_6_8: 60 switch_nic_out_6_nic_in_8 - group_start_8
   + group_end_6 <= 60
 constr_link_match_2_6_8: 60 switch_nic_out_6_nic_in_8 + group_start_8
   - group_end_6 <= 60
 constr_link_match_1_6_9: 60 switch_nic_out_6_nic_in_9 - group_start_9
   + group_end_6 <= 60
 constr_link_match_2_6_9: 60 switch_nic_out_6_nic_in_9 + group_start_9
   - group_end_6 <= 60
 constr_link_match_1_6_10: 60 switch_nic_out_6_nic_in_10 - group_start_10
   + group_end_6 <= 60
 constr_link_match_2_6_10: 60 switch_nic_out_6_nic_in_10 + group_start_10
   - group_end_6 <= 60
 constr_link_match_1_6_11: 60 switch_nic_out_6_nic_in_11 - group_start_11
   + group_end_6 <= 60
 constr_link_match_2_6_11: 60 switch_nic_out_6_nic_in_11 + group_start_11
   - group_end_6 <= 60
 constr_link_match_1_6_12: 60 switch_nic_out_6_nic_in_12 - group_start_12
   + group_end_6 <= 60
 constr_link_match_2_6_12: 60 switch_nic_out_6_nic_in_12 + group_start_12
   - group_end_6 <= 60
 constr_link_match_1_6_13: 60 switch_nic_out_6_nic_in_13 - group_start_13
   + group_end_6 <= 60
 constr_link_match_2_6_13: 60 switch_nic_out_6_nic_in_13 + group_start_13
   - group_end_6 <= 60
 constr_link_match_1_6_14: 60 switch_nic_out_6_nic_in_14 - group_start_14
   + group_end_6 <= 60
 constr_link_match_2_6_14: 60 switch_nic_out_6_nic_in_14 + group_start_14
   - group_end_6 <= 60
 constr_is_start_zero_7: start_7 + 60 is_start_zero_7_for_source_nic_in_7
   <= 60
 constr_edge_switch_start_zero_7_OR: switch_source_nic_in_7
   - is_start_zero_7_for_source_nic_in_7 <= 0
 constr_is_end_complete_7: start_7 + hold_7_1 + 2 hold_7_2 + 3 hold_7_3
   + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7 + 8 hold_7_8
   + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11 + 12 hold_7_12 + 13 hold_7_13
   + 14 hold_7_14 + 15 hold_7_15 + 16 hold_7_16 + 17 hold_7_17
   + 18 hold_7_18 + 19 hold_7_19 + 20 hold_7_20 + 21 hold_7_21
   + 22 hold_7_22 + 23 hold_7_23 + 24 hold_7_24 + 25 hold_7_25
   + 26 hold_7_26 + 27 hold_7_27 + 28 hold_7_28 + 29 hold_7_29
   + 30 hold_7_30 + 31 hold_7_31 + 32 hold_7_32 - 60 is_end_complete_7
   >= 0
 constr_edge_switch_end_max_7_OR: switch_nic_out_7_sink - is_end_complete_7
   <= 0
 constr_link_match_1_7_0: 60 switch_nic_out_7_nic_in_0 - group_start_0
   + group_end_7 <= 60
 constr_link_match_2_7_0: 60 switch_nic_out_7_nic_in_0 + group_start_0
   - group_end_7 <= 60
 constr_link_match_1_7_4: 60 switch_nic_out_7_nic_in_4 - group_start_4
   + group_end_7 <= 60
 constr_link_match_2_7_4: 60 switch_nic_out_7_nic_in_4 + group_start_4
   - group_end_7 <= 60
 constr_link_match_1_7_5: 60 switch_nic_out_7_nic_in_5 - group_start_5
   + group_end_7 <= 60
 constr_link_match_2_7_5: 60 switch_nic_out_7_nic_in_5 + group_start_5
   - group_end_7 <= 60
 constr_link_match_1_7_6: 60 switch_nic_out_7_nic_in_6 - group_start_6
   + group_end_7 <= 60
 constr_link_match_2_7_6: 60 switch_nic_out_7_nic_in_6 + group_start_6
   - group_end_7 <= 60
 constr_link_match_1_7_8: 60 switch_nic_out_7_nic_in_8 - group_start_8
   + group_end_7 <= 60
 constr_link_match_2_7_8: 60 switch_nic_out_7_nic_in_8 + group_start_8
   - group_end_7 <= 60
 constr_link_match_1_7_9: 60 switch_nic_out_7_nic_in_9 - group_start_9
   + group_end_7 <= 60
 constr_link_match_2_7_9: 60 switch_nic_out_7_nic_in_9 + group_start_9
   - group_end_7 <= 60
 constr_link_match_1_7_10: 60 switch_nic_out_7_nic_in_10 - group_start_10
   + group_end_7 <= 60
 constr_link_match_2_7_10: 60 switch_nic_out_7_nic_in_10 + group_start_10
   - group_end_7 <= 60
 constr_link_match_1_7_11: 60 switch_nic_out_7_nic_in_11 - group_start_11
   + group_end_7 <= 60
 constr_link_match_2_7_11: 60 switch_nic_out_7_nic_in_11 + group_start_11
   - group_end_7 <= 60
 constr_link_match_1_7_12: 60 switch_nic_out_7_nic_in_12 - group_start_12
   + group_end_7 <= 60
 constr_link_match_2_7_12: 60 switch_nic_out_7_nic_in_12 + group_start_12
   - group_end_7 <= 60
 constr_link_match_1_7_13: 60 switch_nic_out_7_nic_in_13 - group_start_13
   + group_end_7 <= 60
 constr_link_match_2_7_13: 60 switch_nic_out_7_nic_in_13 + group_start_13
   - group_end_7 <= 60
 constr_link_match_1_7_14: 60 switch_nic_out_7_nic_in_14 - group_start_14
   + group_end_7 <= 60
 constr_link_match_2_7_14: 60 switch_nic_out_7_nic_in_14 + group_start_14
   - group_end_7 <= 60
 constr_is_start_zero_8: start_8 + 60 is_start_zero_8_for_source_nic_in_8
   <= 60
 constr_edge_switch_start_zero_8_OR: switch_source_nic_in_8
   - is_start_zero_8_for_source_nic_in_8 <= 0
 constr_is_end_complete_8: start_8 + hold_8_1 + 2 hold_8_2 + 3 hold_8_3
   + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7 + 8 hold_8_8
   + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 12 hold_8_12 + 13 hold_8_13
   + 14 hold_8_14 + 15 hold_8_15 + 16 hold_8_16 + 17 hold_8_17
   + 18 hold_8_18 + 19 hold_8_19 + 20 hold_8_20 + 21 hold_8_21
   + 22 hold_8_22 + 23 hold_8_23 + 24 hold_8_24 + 25 hold_8_25
   + 26 hold_8_26 + 27 hold_8_27 + 28 hold_8_28 + 29 hold_8_29
   + 30 hold_8_30 + 31 hold_8_31 + 32 hold_8_32 - 60 is_end_complete_8
   >= 0
 constr_edge_switch_end_max_8_OR: switch_nic_out_8_sink - is_end_complete_8
   <= 0
 constr_link_match_1_8_0: 60 switch_nic_out_8_nic_in_0 - group_start_0
   + group_end_8 <= 60
 constr_link_match_2_8_0: 60 switch_nic_out_8_nic_in_0 + group_start_0
   - group_end_8 <= 60
 constr_link_match_1_8_4: 60 switch_nic_out_8_nic_in_4 - group_start_4
   + group_end_8 <= 60
 constr_link_match_2_8_4: 60 switch_nic_out_8_nic_in_4 + group_start_4
   - group_end_8 <= 60
 constr_link_match_1_8_5: 60 switch_nic_out_8_nic_in_5 - group_start_5
   + group_end_8 <= 60
 constr_link_match_2_8_5: 60 switch_nic_out_8_nic_in_5 + group_start_5
   - group_end_8 <= 60
 constr_link_match_1_8_6: 60 switch_nic_out_8_nic_in_6 - group_start_6
   + group_end_8 <= 60
 constr_link_match_2_8_6: 60 switch_nic_out_8_nic_in_6 + group_start_6
   - group_end_8 <= 60
 constr_link_match_1_8_7: 60 switch_nic_out_8_nic_in_7 - group_start_7
   + group_end_8 <= 60
 constr_link_match_2_8_7: 60 switch_nic_out_8_nic_in_7 + group_start_7
   - group_end_8 <= 60
 constr_link_match_1_8_9: 60 switch_nic_out_8_nic_in_9 - group_start_9
   + group_end_8 <= 60
 constr_link_match_2_8_9: 60 switch_nic_out_8_nic_in_9 + group_start_9
   - group_end_8 <= 60
 constr_link_match_1_8_10: 60 switch_nic_out_8_nic_in_10 - group_start_10
   + group_end_8 <= 60
 constr_link_match_2_8_10: 60 switch_nic_out_8_nic_in_10 + group_start_10
   - group_end_8 <= 60
 constr_link_match_1_8_11: 60 switch_nic_out_8_nic_in_11 - group_start_11
   + group_end_8 <= 60
 constr_link_match_2_8_11: 60 switch_nic_out_8_nic_in_11 + group_start_11
   - group_end_8 <= 60
 constr_link_match_1_8_12: 60 switch_nic_out_8_nic_in_12 - group_start_12
   + group_end_8 <= 60
 constr_link_match_2_8_12: 60 switch_nic_out_8_nic_in_12 + group_start_12
   - group_end_8 <= 60
 constr_link_match_1_8_13: 60 switch_nic_out_8_nic_in_13 - group_start_13
   + group_end_8 <= 60
 constr_link_match_2_8_13: 60 switch_nic_out_8_nic_in_13 + group_start_13
   - group_end_8 <= 60
 constr_link_match_1_8_14: 60 switch_nic_out_8_nic_in_14 - group_start_14
   + group_end_8 <= 60
 constr_link_match_2_8_14: 60 switch_nic_out_8_nic_in_14 + group_start_14
   - group_end_8 <= 60
 constr_is_start_zero_9: start_9 + 60 is_start_zero_9_for_source_nic_in_9
   <= 60
 constr_edge_switch_start_zero_9_OR: switch_source_nic_in_9
   - is_start_zero_9_for_source_nic_in_9 <= 0
 constr_is_end_complete_9: start_9 + hold_9_1 + 2 hold_9_2 + 3 hold_9_3
   + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7 + 8 hold_9_8
   + 9 hold_9_9 + 10 hold_9_10 + 11 hold_9_11 + 12 hold_9_12 + 13 hold_9_13
   + 14 hold_9_14 + 15 hold_9_15 + 16 hold_9_16 + 17 hold_9_17
   + 18 hold_9_18 + 19 hold_9_19 + 20 hold_9_20 + 21 hold_9_21
   + 22 hold_9_22 + 23 hold_9_23 + 24 hold_9_24 + 25 hold_9_25
   + 26 hold_9_26 + 27 hold_9_27 + 28 hold_9_28 + 29 hold_9_29
   + 30 hold_9_30 + 31 hold_9_31 + 32 hold_9_32 - 60 is_end_complete_9
   >= 0
 constr_edge_switch_end_max_9_OR: switch_nic_out_9_sink - is_end_complete_9
   <= 0
 constr_link_match_1_9_0: 60 switch_nic_out_9_nic_in_0 - group_start_0
   + group_end_9 <= 60
 constr_link_match_2_9_0: 60 switch_nic_out_9_nic_in_0 + group_start_0
   - group_end_9 <= 60
 constr_link_match_1_9_4: 60 switch_nic_out_9_nic_in_4 - group_start_4
   + group_end_9 <= 60
 constr_link_match_2_9_4: 60 switch_nic_out_9_nic_in_4 + group_start_4
   - group_end_9 <= 60
 constr_link_match_1_9_5: 60 switch_nic_out_9_nic_in_5 - group_start_5
   + group_end_9 <= 60
 constr_link_match_2_9_5: 60 switch_nic_out_9_nic_in_5 + group_start_5
   - group_end_9 <= 60
 constr_link_match_1_9_6: 60 switch_nic_out_9_nic_in_6 - group_start_6
   + group_end_9 <= 60
 constr_link_match_2_9_6: 60 switch_nic_out_9_nic_in_6 + group_start_6
   - group_end_9 <= 60
 constr_link_match_1_9_7: 60 switch_nic_out_9_nic_in_7 - group_start_7
   + group_end_9 <= 60
 constr_link_match_2_9_7: 60 switch_nic_out_9_nic_in_7 + group_start_7
   - group_end_9 <= 60
 constr_link_match_1_9_8: 60 switch_nic_out_9_nic_in_8 - group_start_8
   + group_end_9 <= 60
 constr_link_match_2_9_8: 60 switch_nic_out_9_nic_in_8 + group_start_8
   - group_end_9 <= 60
 constr_link_match_1_9_10: 60 switch_nic_out_9_nic_in_10 - group_start_10
   + group_end_9 <= 60
 constr_link_match_2_9_10: 60 switch_nic_out_9_nic_in_10 + group_start_10
   - group_end_9 <= 60
 constr_link_match_1_9_11: 60 switch_nic_out_9_nic_in_11 - group_start_11
   + group_end_9 <= 60
 constr_link_match_2_9_11: 60 switch_nic_out_9_nic_in_11 + group_start_11
   - group_end_9 <= 60
 constr_link_match_1_9_12: 60 switch_nic_out_9_nic_in_12 - group_start_12
   + group_end_9 <= 60
 constr_link_match_2_9_12: 60 switch_nic_out_9_nic_in_12 + group_start_12
   - group_end_9 <= 60
 constr_link_match_1_9_13: 60 switch_nic_out_9_nic_in_13 - group_start_13
   + group_end_9 <= 60
 constr_link_match_2_9_13: 60 switch_nic_out_9_nic_in_13 + group_start_13
   - group_end_9 <= 60
 constr_link_match_1_9_14: 60 switch_nic_out_9_nic_in_14 - group_start_14
   + group_end_9 <= 60
 constr_link_match_2_9_14: 60 switch_nic_out_9_nic_in_14 + group_start_14
   - group_end_9 <= 60
 constr_is_start_zero_10: start_10
   + 60 is_start_zero_10_for_source_nic_in_10 <= 60
 constr_edge_switch_start_zero_10_OR: switch_source_nic_in_10
   - is_start_zero_10_for_source_nic_in_10 <= 0
 constr_is_end_complete_10: start_10 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 12 hold_10_12 + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15
   + 16 hold_10_16 + 17 hold_10_17 + 18 hold_10_18 + 19 hold_10_19
   + 20 hold_10_20 + 21 hold_10_21 + 22 hold_10_22 + 23 hold_10_23
   + 24 hold_10_24 + 25 hold_10_25 + 26 hold_10_26 + 27 hold_10_27
   + 28 hold_10_28 + 29 hold_10_29 + 30 hold_10_30 + 31 hold_10_31
   + 32 hold_10_32 - 60 is_end_complete_10 >= 0
 constr_edge_switch_end_max_10_OR: switch_nic_out_10_sink
   - is_end_complete_10 <= 0
 constr_link_match_1_10_0: 60 switch_nic_out_10_nic_in_0 - group_start_0
   + group_end_10 <= 60
 constr_link_match_2_10_0: 60 switch_nic_out_10_nic_in_0 + group_start_0
   - group_end_10 <= 60
 constr_link_match_1_10_4: 60 switch_nic_out_10_nic_in_4 - group_start_4
   + group_end_10 <= 60
 constr_link_match_2_10_4: 60 switch_nic_out_10_nic_in_4 + group_start_4
   - group_end_10 <= 60
 constr_link_match_1_10_5: 60 switch_nic_out_10_nic_in_5 - group_start_5
   + group_end_10 <= 60
 constr_link_match_2_10_5: 60 switch_nic_out_10_nic_in_5 + group_start_5
   - group_end_10 <= 60
 constr_link_match_1_10_6: 60 switch_nic_out_10_nic_in_6 - group_start_6
   + group_end_10 <= 60
 constr_link_match_2_10_6: 60 switch_nic_out_10_nic_in_6 + group_start_6
   - group_end_10 <= 60
 constr_link_match_1_10_7: 60 switch_nic_out_10_nic_in_7 - group_start_7
   + group_end_10 <= 60
 constr_link_match_2_10_7: 60 switch_nic_out_10_nic_in_7 + group_start_7
   - group_end_10 <= 60
 constr_link_match_1_10_8: 60 switch_nic_out_10_nic_in_8 - group_start_8
   + group_end_10 <= 60
 constr_link_match_2_10_8: 60 switch_nic_out_10_nic_in_8 + group_start_8
   - group_end_10 <= 60
 constr_link_match_1_10_9: 60 switch_nic_out_10_nic_in_9 - group_start_9
   + group_end_10 <= 60
 constr_link_match_2_10_9: 60 switch_nic_out_10_nic_in_9 + group_start_9
   - group_end_10 <= 60
 constr_link_match_1_10_11: 60 switch_nic_out_10_nic_in_11 - group_start_11
   + group_end_10 <= 60
 constr_link_match_2_10_11: 60 switch_nic_out_10_nic_in_11 + group_start_11
   - group_end_10 <= 60
 constr_link_match_1_10_12: 60 switch_nic_out_10_nic_in_12 - group_start_12
   + group_end_10 <= 60
 constr_link_match_2_10_12: 60 switch_nic_out_10_nic_in_12 + group_start_12
   - group_end_10 <= 60
 constr_link_match_1_10_13: 60 switch_nic_out_10_nic_in_13 - group_start_13
   + group_end_10 <= 60
 constr_link_match_2_10_13: 60 switch_nic_out_10_nic_in_13 + group_start_13
   - group_end_10 <= 60
 constr_link_match_1_10_14: 60 switch_nic_out_10_nic_in_14 - group_start_14
   + group_end_10 <= 60
 constr_link_match_2_10_14: 60 switch_nic_out_10_nic_in_14 + group_start_14
   - group_end_10 <= 60
 constr_is_start_zero_11: start_11
   + 60 is_start_zero_11_for_source_nic_in_11 <= 60
 constr_edge_switch_start_zero_11_OR: switch_source_nic_in_11
   - is_start_zero_11_for_source_nic_in_11 <= 0
 constr_is_end_complete_11: start_11 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 12 hold_11_12 + 13 hold_11_13 + 14 hold_11_14 + 15 hold_11_15
   + 16 hold_11_16 + 17 hold_11_17 + 18 hold_11_18 + 19 hold_11_19
   + 20 hold_11_20 + 21 hold_11_21 + 22 hold_11_22 + 23 hold_11_23
   + 24 hold_11_24 + 25 hold_11_25 + 26 hold_11_26 + 27 hold_11_27
   + 28 hold_11_28 + 29 hold_11_29 + 30 hold_11_30 + 31 hold_11_31
   + 32 hold_11_32 - 60 is_end_complete_11 >= 0
 constr_edge_switch_end_max_11_OR: switch_nic_out_11_sink
   - is_end_complete_11 <= 0
 constr_link_match_1_11_0: 60 switch_nic_out_11_nic_in_0 - group_start_0
   + group_end_11 <= 60
 constr_link_match_2_11_0: 60 switch_nic_out_11_nic_in_0 + group_start_0
   - group_end_11 <= 60
 constr_link_match_1_11_4: 60 switch_nic_out_11_nic_in_4 - group_start_4
   + group_end_11 <= 60
 constr_link_match_2_11_4: 60 switch_nic_out_11_nic_in_4 + group_start_4
   - group_end_11 <= 60
 constr_link_match_1_11_5: 60 switch_nic_out_11_nic_in_5 - group_start_5
   + group_end_11 <= 60
 constr_link_match_2_11_5: 60 switch_nic_out_11_nic_in_5 + group_start_5
   - group_end_11 <= 60
 constr_link_match_1_11_6: 60 switch_nic_out_11_nic_in_6 - group_start_6
   + group_end_11 <= 60
 constr_link_match_2_11_6: 60 switch_nic_out_11_nic_in_6 + group_start_6
   - group_end_11 <= 60
 constr_link_match_1_11_7: 60 switch_nic_out_11_nic_in_7 - group_start_7
   + group_end_11 <= 60
 constr_link_match_2_11_7: 60 switch_nic_out_11_nic_in_7 + group_start_7
   - group_end_11 <= 60
 constr_link_match_1_11_8: 60 switch_nic_out_11_nic_in_8 - group_start_8
   + group_end_11 <= 60
 constr_link_match_2_11_8: 60 switch_nic_out_11_nic_in_8 + group_start_8
   - group_end_11 <= 60
 constr_link_match_1_11_9: 60 switch_nic_out_11_nic_in_9 - group_start_9
   + group_end_11 <= 60
 constr_link_match_2_11_9: 60 switch_nic_out_11_nic_in_9 + group_start_9
   - group_end_11 <= 60
 constr_link_match_1_11_10: 60 switch_nic_out_11_nic_in_10 - group_start_10
   + group_end_11 <= 60
 constr_link_match_2_11_10: 60 switch_nic_out_11_nic_in_10 + group_start_10
   - group_end_11 <= 60
 constr_link_match_1_11_12: 60 switch_nic_out_11_nic_in_12 - group_start_12
   + group_end_11 <= 60
 constr_link_match_2_11_12: 60 switch_nic_out_11_nic_in_12 + group_start_12
   - group_end_11 <= 60
 constr_link_match_1_11_13: 60 switch_nic_out_11_nic_in_13 - group_start_13
   + group_end_11 <= 60
 constr_link_match_2_11_13: 60 switch_nic_out_11_nic_in_13 + group_start_13
   - group_end_11 <= 60
 constr_link_match_1_11_14: 60 switch_nic_out_11_nic_in_14 - group_start_14
   + group_end_11 <= 60
 constr_link_match_2_11_14: 60 switch_nic_out_11_nic_in_14 + group_start_14
   - group_end_11 <= 60
 constr_is_start_zero_12: start_12
   + 60 is_start_zero_12_for_source_nic_in_12 <= 60
 constr_edge_switch_start_zero_12_OR: switch_source_nic_in_12
   - is_start_zero_12_for_source_nic_in_12 <= 0
 constr_is_end_complete_12: start_12 + hold_12_1 + 2 hold_12_2
   + 3 hold_12_3 + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6 + 7 hold_12_7
   + 8 hold_12_8 + 9 hold_12_9 + 10 hold_12_10 + 11 hold_12_11
   + 12 hold_12_12 + 13 hold_12_13 + 14 hold_12_14 + 15 hold_12_15
   + 16 hold_12_16 + 17 hold_12_17 + 18 hold_12_18 + 19 hold_12_19
   + 20 hold_12_20 + 21 hold_12_21 + 22 hold_12_22 + 23 hold_12_23
   + 24 hold_12_24 + 25 hold_12_25 + 26 hold_12_26 + 27 hold_12_27
   + 28 hold_12_28 + 29 hold_12_29 + 30 hold_12_30 + 31 hold_12_31
   + 32 hold_12_32 - 60 is_end_complete_12 >= 0
 constr_edge_switch_end_max_12_OR: switch_nic_out_12_sink
   - is_end_complete_12 <= 0
 constr_link_match_1_12_0: 60 switch_nic_out_12_nic_in_0 - group_start_0
   + group_end_12 <= 60
 constr_link_match_2_12_0: 60 switch_nic_out_12_nic_in_0 + group_start_0
   - group_end_12 <= 60
 constr_link_match_1_12_4: 60 switch_nic_out_12_nic_in_4 - group_start_4
   + group_end_12 <= 60
 constr_link_match_2_12_4: 60 switch_nic_out_12_nic_in_4 + group_start_4
   - group_end_12 <= 60
 constr_link_match_1_12_5: 60 switch_nic_out_12_nic_in_5 - group_start_5
   + group_end_12 <= 60
 constr_link_match_2_12_5: 60 switch_nic_out_12_nic_in_5 + group_start_5
   - group_end_12 <= 60
 constr_link_match_1_12_6: 60 switch_nic_out_12_nic_in_6 - group_start_6
   + group_end_12 <= 60
 constr_link_match_2_12_6: 60 switch_nic_out_12_nic_in_6 + group_start_6
   - group_end_12 <= 60
 constr_link_match_1_12_7: 60 switch_nic_out_12_nic_in_7 - group_start_7
   + group_end_12 <= 60
 constr_link_match_2_12_7: 60 switch_nic_out_12_nic_in_7 + group_start_7
   - group_end_12 <= 60
 constr_link_match_1_12_8: 60 switch_nic_out_12_nic_in_8 - group_start_8
   + group_end_12 <= 60
 constr_link_match_2_12_8: 60 switch_nic_out_12_nic_in_8 + group_start_8
   - group_end_12 <= 60
 constr_link_match_1_12_9: 60 switch_nic_out_12_nic_in_9 - group_start_9
   + group_end_12 <= 60
 constr_link_match_2_12_9: 60 switch_nic_out_12_nic_in_9 + group_start_9
   - group_end_12 <= 60
 constr_link_match_1_12_10: 60 switch_nic_out_12_nic_in_10 - group_start_10
   + group_end_12 <= 60
 constr_link_match_2_12_10: 60 switch_nic_out_12_nic_in_10 + group_start_10
   - group_end_12 <= 60
 constr_link_match_1_12_11: 60 switch_nic_out_12_nic_in_11 - group_start_11
   + group_end_12 <= 60
 constr_link_match_2_12_11: 60 switch_nic_out_12_nic_in_11 + group_start_11
   - group_end_12 <= 60
 constr_link_match_1_12_13: 60 switch_nic_out_12_nic_in_13 - group_start_13
   + group_end_12 <= 60
 constr_link_match_2_12_13: 60 switch_nic_out_12_nic_in_13 + group_start_13
   - group_end_12 <= 60
 constr_link_match_1_12_14: 60 switch_nic_out_12_nic_in_14 - group_start_14
   + group_end_12 <= 60
 constr_link_match_2_12_14: 60 switch_nic_out_12_nic_in_14 + group_start_14
   - group_end_12 <= 60
 constr_is_start_zero_13: start_13
   + 60 is_start_zero_13_for_source_nic_in_13 <= 60
 constr_edge_switch_start_zero_13_OR: switch_source_nic_in_13
   - is_start_zero_13_for_source_nic_in_13 <= 0
 constr_is_end_complete_13: start_13 + hold_13_1 + 2 hold_13_2
   + 3 hold_13_3 + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6 + 7 hold_13_7
   + 8 hold_13_8 + 9 hold_13_9 + 10 hold_13_10 + 11 hold_13_11
   + 12 hold_13_12 + 13 hold_13_13 + 14 hold_13_14 + 15 hold_13_15
   + 16 hold_13_16 + 17 hold_13_17 + 18 hold_13_18 + 19 hold_13_19
   + 20 hold_13_20 + 21 hold_13_21 + 22 hold_13_22 + 23 hold_13_23
   + 24 hold_13_24 + 25 hold_13_25 + 26 hold_13_26 + 27 hold_13_27
   + 28 hold_13_28 + 29 hold_13_29 + 30 hold_13_30 + 31 hold_13_31
   + 32 hold_13_32 - 60 is_end_complete_13 >= 0
 constr_edge_switch_end_max_13_OR: switch_nic_out_13_sink
   - is_end_complete_13 <= 0
 constr_link_match_1_13_0: 60 switch_nic_out_13_nic_in_0 - group_start_0
   + group_end_13 <= 60
 constr_link_match_2_13_0: 60 switch_nic_out_13_nic_in_0 + group_start_0
   - group_end_13 <= 60
 constr_link_match_1_13_4: 60 switch_nic_out_13_nic_in_4 - group_start_4
   + group_end_13 <= 60
 constr_link_match_2_13_4: 60 switch_nic_out_13_nic_in_4 + group_start_4
   - group_end_13 <= 60
 constr_link_match_1_13_5: 60 switch_nic_out_13_nic_in_5 - group_start_5
   + group_end_13 <= 60
 constr_link_match_2_13_5: 60 switch_nic_out_13_nic_in_5 + group_start_5
   - group_end_13 <= 60
 constr_link_match_1_13_6: 60 switch_nic_out_13_nic_in_6 - group_start_6
   + group_end_13 <= 60
 constr_link_match_2_13_6: 60 switch_nic_out_13_nic_in_6 + group_start_6
   - group_end_13 <= 60
 constr_link_match_1_13_7: 60 switch_nic_out_13_nic_in_7 - group_start_7
   + group_end_13 <= 60
 constr_link_match_2_13_7: 60 switch_nic_out_13_nic_in_7 + group_start_7
   - group_end_13 <= 60
 constr_link_match_1_13_8: 60 switch_nic_out_13_nic_in_8 - group_start_8
   + group_end_13 <= 60
 constr_link_match_2_13_8: 60 switch_nic_out_13_nic_in_8 + group_start_8
   - group_end_13 <= 60
 constr_link_match_1_13_9: 60 switch_nic_out_13_nic_in_9 - group_start_9
   + group_end_13 <= 60
 constr_link_match_2_13_9: 60 switch_nic_out_13_nic_in_9 + group_start_9
   - group_end_13 <= 60
 constr_link_match_1_13_10: 60 switch_nic_out_13_nic_in_10 - group_start_10
   + group_end_13 <= 60
 constr_link_match_2_13_10: 60 switch_nic_out_13_nic_in_10 + group_start_10
   - group_end_13 <= 60
 constr_link_match_1_13_11: 60 switch_nic_out_13_nic_in_11 - group_start_11
   + group_end_13 <= 60
 constr_link_match_2_13_11: 60 switch_nic_out_13_nic_in_11 + group_start_11
   - group_end_13 <= 60
 constr_link_match_1_13_12: 60 switch_nic_out_13_nic_in_12 - group_start_12
   + group_end_13 <= 60
 constr_link_match_2_13_12: 60 switch_nic_out_13_nic_in_12 + group_start_12
   - group_end_13 <= 60
 constr_link_match_1_13_14: 60 switch_nic_out_13_nic_in_14 - group_start_14
   + group_end_13 <= 60
 constr_link_match_2_13_14: 60 switch_nic_out_13_nic_in_14 + group_start_14
   - group_end_13 <= 60
 constr_is_start_zero_14: start_14
   + 60 is_start_zero_14_for_source_nic_in_14 <= 60
 constr_edge_switch_start_zero_14_OR: switch_source_nic_in_14
   - is_start_zero_14_for_source_nic_in_14 <= 0
 constr_is_end_complete_14: start_14 + hold_14_1 + 2 hold_14_2
   + 3 hold_14_3 + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6 + 7 hold_14_7
   + 8 hold_14_8 + 9 hold_14_9 + 10 hold_14_10 + 11 hold_14_11
   + 12 hold_14_12 + 13 hold_14_13 + 14 hold_14_14 + 15 hold_14_15
   + 16 hold_14_16 + 17 hold_14_17 + 18 hold_14_18 + 19 hold_14_19
   + 20 hold_14_20 + 21 hold_14_21 + 22 hold_14_22 + 23 hold_14_23
   + 24 hold_14_24 + 25 hold_14_25 + 26 hold_14_26 + 27 hold_14_27
   + 28 hold_14_28 + 29 hold_14_29 + 30 hold_14_30 + 31 hold_14_31
   + 32 hold_14_32 - 60 is_end_complete_14 >= 0
 constr_edge_switch_end_max_14_OR: switch_nic_out_14_sink
   - is_end_complete_14 <= 0
 constr_link_match_1_14_0: 60 switch_nic_out_14_nic_in_0 - group_start_0
   + group_end_14 <= 60
 constr_link_match_2_14_0: 60 switch_nic_out_14_nic_in_0 + group_start_0
   - group_end_14 <= 60
 constr_link_match_1_14_4: 60 switch_nic_out_14_nic_in_4 - group_start_4
   + group_end_14 <= 60
 constr_link_match_2_14_4: 60 switch_nic_out_14_nic_in_4 + group_start_4
   - group_end_14 <= 60
 constr_link_match_1_14_5: 60 switch_nic_out_14_nic_in_5 - group_start_5
   + group_end_14 <= 60
 constr_link_match_2_14_5: 60 switch_nic_out_14_nic_in_5 + group_start_5
   - group_end_14 <= 60
 constr_link_match_1_14_6: 60 switch_nic_out_14_nic_in_6 - group_start_6
   + group_end_14 <= 60
 constr_link_match_2_14_6: 60 switch_nic_out_14_nic_in_6 + group_start_6
   - group_end_14 <= 60
 constr_link_match_1_14_7: 60 switch_nic_out_14_nic_in_7 - group_start_7
   + group_end_14 <= 60
 constr_link_match_2_14_7: 60 switch_nic_out_14_nic_in_7 + group_start_7
   - group_end_14 <= 60
 constr_link_match_1_14_8: 60 switch_nic_out_14_nic_in_8 - group_start_8
   + group_end_14 <= 60
 constr_link_match_2_14_8: 60 switch_nic_out_14_nic_in_8 + group_start_8
   - group_end_14 <= 60
 constr_link_match_1_14_9: 60 switch_nic_out_14_nic_in_9 - group_start_9
   + group_end_14 <= 60
 constr_link_match_2_14_9: 60 switch_nic_out_14_nic_in_9 + group_start_9
   - group_end_14 <= 60
 constr_link_match_1_14_10: 60 switch_nic_out_14_nic_in_10 - group_start_10
   + group_end_14 <= 60
 constr_link_match_2_14_10: 60 switch_nic_out_14_nic_in_10 + group_start_10
   - group_end_14 <= 60
 constr_link_match_1_14_11: 60 switch_nic_out_14_nic_in_11 - group_start_11
   + group_end_14 <= 60
 constr_link_match_2_14_11: 60 switch_nic_out_14_nic_in_11 + group_start_11
   - group_end_14 <= 60
 constr_link_match_1_14_12: 60 switch_nic_out_14_nic_in_12 - group_start_12
   + group_end_14 <= 60
 constr_link_match_2_14_12: 60 switch_nic_out_14_nic_in_12 + group_start_12
   - group_end_14 <= 60
 constr_link_match_1_14_13: 60 switch_nic_out_14_nic_in_13 - group_start_13
   + group_end_14 <= 60
 constr_link_match_2_14_13: 60 switch_nic_out_14_nic_in_13 + group_start_13
   - group_end_14 <= 60
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
 edge_flow_constr_source_nic_in_0: flow_source_nic_in_0
   - 5e+08 switch_source_nic_in_0 <= 0
 edge_flow_constr_nic_out_0_sink: flow_nic_out_0_sink
   - 5e+08 switch_nic_out_0_sink <= 0
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
 edge_flow_constr_nic_in_0_0: flow_nic_in_0_0
   - 4.807692307692308e+06 switch_nic_in_0_0 <= 0
 edge_flow_constr_0_nic_out_0: flow_0_nic_out_0
   - 4.807692307692308e+06 switch_0_nic_out_0 <= 0
 edge_flow_constr_nic_in_0_1: flow_nic_in_0_1
   - 4.807692307692308e+06 switch_nic_in_0_1 <= 0
 edge_flow_constr_1_nic_out_0: flow_1_nic_out_0
   - 4.807692307692308e+06 switch_1_nic_out_0 <= 0
 edge_flow_constr_nic_in_0_2: flow_nic_in_0_2
   - 4.807692307692308e+06 switch_nic_in_0_2 <= 0
 edge_flow_constr_2_nic_out_0: flow_2_nic_out_0
   - 4.807692307692308e+06 switch_2_nic_out_0 <= 0
 edge_flow_constr_nic_in_0_3: flow_nic_in_0_3
   - 4.807692307692308e+06 switch_nic_in_0_3 <= 0
 edge_flow_constr_3_nic_out_0: flow_3_nic_out_0
   - 4.807692307692308e+06 switch_3_nic_out_0 <= 0
 edge_flow_constr_0_1: flow_0_1 - 4.807692307692308e+06 switch_0_1 <= 0
 edge_flow_constr_0_2: flow_0_2 - 4.807692307692308e+06 switch_0_2 <= 0
 edge_flow_constr_0_3: flow_0_3 - 4.807692307692308e+06 switch_0_3 <= 0
 edge_flow_constr_1_0: flow_1_0 - 4.807692307692308e+06 switch_1_0 <= 0
 edge_flow_constr_1_2: flow_1_2 - 4.807692307692308e+06 switch_1_2 <= 0
 edge_flow_constr_1_3: flow_1_3 - 4.807692307692308e+06 switch_1_3 <= 0
 edge_flow_constr_2_0: flow_2_0 - 4.807692307692308e+06 switch_2_0 <= 0
 edge_flow_constr_2_1: flow_2_1 - 4.807692307692308e+06 switch_2_1 <= 0
 edge_flow_constr_2_3: flow_2_3 - 4.807692307692308e+06 switch_2_3 <= 0
 edge_flow_constr_3_0: flow_3_0 - 4.807692307692308e+06 switch_3_0 <= 0
 edge_flow_constr_3_1: flow_3_1 - 4.807692307692308e+06 switch_3_1 <= 0
 edge_flow_constr_3_2: flow_3_2 - 4.807692307692308e+06 switch_3_2 <= 0
 edge_flow_constr_source_nic_in_4: flow_source_nic_in_4
   - 5e+08 switch_source_nic_in_4 <= 0
 edge_flow_constr_nic_out_4_sink: flow_nic_out_4_sink
   - 5e+08 switch_nic_out_4_sink <= 0
 edge_flow_constr_nic_out_4_nic_in_0: flow_nic_out_4_nic_in_0
   - 75120.19230769231 switch_nic_out_4_nic_in_0 <= 0
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
 edge_flow_constr_nic_in_14_14: flow_nic_in_14_14
   - 4.807692307692308e+06 switch_nic_in_14_14 <= 0
 edge_flow_constr_14_nic_out_14: flow_14_nic_out_14
   - 4.807692307692308e+06 switch_14_nic_out_14 <= 0
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
 R716: - group_tp_active_0 + tp_on_and_switch_source_nic_in_0_0 <= 0
 R717: - switch_source_nic_in_0 + tp_on_and_switch_source_nic_in_0_0 <= 0
 R718: - switch_source_nic_in_0 - group_tp_active_0
   + tp_on_and_switch_source_nic_in_0_0 >= -1
 R719: - group_tp_active_0 + tp_on_and_switch_nic_out_4_nic_in_0_0 <= 0
 R720: - switch_nic_out_4_nic_in_0 + tp_on_and_switch_nic_out_4_nic_in_0_0
   <= 0
 R721: - switch_nic_out_4_nic_in_0 - group_tp_active_0
   + tp_on_and_switch_nic_out_4_nic_in_0_0 >= -1
 R722: - group_tp_active_0 + tp_on_and_switch_nic_out_5_nic_in_0_0 <= 0
 R723: - switch_nic_out_5_nic_in_0 + tp_on_and_switch_nic_out_5_nic_in_0_0
   <= 0
 R724: - switch_nic_out_5_nic_in_0 - group_tp_active_0
   + tp_on_and_switch_nic_out_5_nic_in_0_0 >= -1
 R725: - group_tp_active_0 + tp_on_and_switch_nic_out_6_nic_in_0_0 <= 0
 R726: - switch_nic_out_6_nic_in_0 + tp_on_and_switch_nic_out_6_nic_in_0_0
   <= 0
 R727: - switch_nic_out_6_nic_in_0 - group_tp_active_0
   + tp_on_and_switch_nic_out_6_nic_in_0_0 >= -1
 R728: - group_tp_active_0 + tp_on_and_switch_nic_out_7_nic_in_0_0 <= 0
 R729: - switch_nic_out_7_nic_in_0 + tp_on_and_switch_nic_out_7_nic_in_0_0
   <= 0
 R730: - switch_nic_out_7_nic_in_0 - group_tp_active_0
   + tp_on_and_switch_nic_out_7_nic_in_0_0 >= -1
 R731: - group_tp_active_0 + tp_on_and_switch_nic_out_8_nic_in_0_0 <= 0
 R732: - switch_nic_out_8_nic_in_0 + tp_on_and_switch_nic_out_8_nic_in_0_0
   <= 0
 R733: - switch_nic_out_8_nic_in_0 - group_tp_active_0
   + tp_on_and_switch_nic_out_8_nic_in_0_0 >= -1
 R734: - group_tp_active_0 + tp_on_and_switch_nic_out_9_nic_in_0_0 <= 0
 R735: - switch_nic_out_9_nic_in_0 + tp_on_and_switch_nic_out_9_nic_in_0_0
   <= 0
 R736: - switch_nic_out_9_nic_in_0 - group_tp_active_0
   + tp_on_and_switch_nic_out_9_nic_in_0_0 >= -1
 R737: - group_tp_active_0 + tp_on_and_switch_nic_out_10_nic_in_0_0 <= 0
 R738: - switch_nic_out_10_nic_in_0
   + tp_on_and_switch_nic_out_10_nic_in_0_0 <= 0
 R739: - switch_nic_out_10_nic_in_0 - group_tp_active_0
   + tp_on_and_switch_nic_out_10_nic_in_0_0 >= -1
 R740: - group_tp_active_0 + tp_on_and_switch_nic_out_11_nic_in_0_0 <= 0
 R741: - switch_nic_out_11_nic_in_0
   + tp_on_and_switch_nic_out_11_nic_in_0_0 <= 0
 R742: - switch_nic_out_11_nic_in_0 - group_tp_active_0
   + tp_on_and_switch_nic_out_11_nic_in_0_0 >= -1
 R743: - group_tp_active_0 + tp_on_and_switch_nic_out_12_nic_in_0_0 <= 0
 R744: - switch_nic_out_12_nic_in_0
   + tp_on_and_switch_nic_out_12_nic_in_0_0 <= 0
 R745: - switch_nic_out_12_nic_in_0 - group_tp_active_0
   + tp_on_and_switch_nic_out_12_nic_in_0_0 >= -1
 R746: - group_tp_active_0 + tp_on_and_switch_nic_out_13_nic_in_0_0 <= 0
 R747: - switch_nic_out_13_nic_in_0
   + tp_on_and_switch_nic_out_13_nic_in_0_0 <= 0
 R748: - switch_nic_out_13_nic_in_0 - group_tp_active_0
   + tp_on_and_switch_nic_out_13_nic_in_0_0 >= -1
 R749: - group_tp_active_0 + tp_on_and_switch_nic_out_14_nic_in_0_0 <= 0
 R750: - switch_nic_out_14_nic_in_0
   + tp_on_and_switch_nic_out_14_nic_in_0_0 <= 0
 R751: - switch_nic_out_14_nic_in_0 - group_tp_active_0
   + tp_on_and_switch_nic_out_14_nic_in_0_0 >= -1
 pipeline_ordering_start_0_1: switch_nic_in_0_1 - group_tp_active_0 <= 0
 pipeline_ordering_start_0_2: switch_nic_in_0_2 - group_tp_active_0 <= 0
 pipeline_ordering_start_0_3: switch_nic_in_0_3 - group_tp_active_0 <= 0
 pipeline_ordering_constr1_0_1: - start_0 + start_1 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11 - 12 hold_0_12
   - 13 hold_0_13 - 14 hold_0_14 - 15 hold_0_15 - 16 hold_0_16
   - 17 hold_0_17 - 18 hold_0_18 - 19 hold_0_19 - 20 hold_0_20
   - 21 hold_0_21 - 22 hold_0_22 - 23 hold_0_23 - 24 hold_0_24
   - 25 hold_0_25 - 26 hold_0_26 - 27 hold_0_27 - 28 hold_0_28
   - 29 hold_0_29 - 30 hold_0_30 - 31 hold_0_31 - 32 hold_0_32
   - 120 group_tp_active_0 + 120 is_active_1 <= 120
 pipeline_ordering_constr2_0_1: start_0 - start_1 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 8 hold_0_8 + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12
   + 13 hold_0_13 + 14 hold_0_14 + 15 hold_0_15 + 16 hold_0_16
   + 17 hold_0_17 + 18 hold_0_18 + 19 hold_0_19 + 20 hold_0_20
   + 21 hold_0_21 + 22 hold_0_22 + 23 hold_0_23 + 24 hold_0_24
   + 25 hold_0_25 + 26 hold_0_26 + 27 hold_0_27 + 28 hold_0_28
   + 29 hold_0_29 + 30 hold_0_30 + 31 hold_0_31 + 32 hold_0_32
   - 120 group_tp_active_0 + 120 is_active_1 <= 120
 pipeline_ordering_constr3_0_1: start_1 + 120 group_tp_active_0
   + 120 is_active_1 >= 60
 pipeline_ordering_constr4_0_1: switch_0_nic_out_0 - group_tp_active_0
   + is_active_1 <= 1
 pipeline_ordering_constr1_1_2: - start_1 + start_2 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11 - 12 hold_1_12
   - 13 hold_1_13 - 14 hold_1_14 - 15 hold_1_15 - 16 hold_1_16
   - 17 hold_1_17 - 18 hold_1_18 - 19 hold_1_19 - 20 hold_1_20
   - 21 hold_1_21 - 22 hold_1_22 - 23 hold_1_23 - 24 hold_1_24
   - 25 hold_1_25 - 26 hold_1_26 - 27 hold_1_27 - 28 hold_1_28
   - 29 hold_1_29 - 30 hold_1_30 - 31 hold_1_31 - 32 hold_1_32
   - 120 group_tp_active_0 + 120 is_active_2 <= 120
 pipeline_ordering_constr2_1_2: start_1 - start_2 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 8 hold_1_8 + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 + 12 hold_1_12
   + 13 hold_1_13 + 14 hold_1_14 + 15 hold_1_15 + 16 hold_1_16
   + 17 hold_1_17 + 18 hold_1_18 + 19 hold_1_19 + 20 hold_1_20
   + 21 hold_1_21 + 22 hold_1_22 + 23 hold_1_23 + 24 hold_1_24
   + 25 hold_1_25 + 26 hold_1_26 + 27 hold_1_27 + 28 hold_1_28
   + 29 hold_1_29 + 30 hold_1_30 + 31 hold_1_31 + 32 hold_1_32
   - 120 group_tp_active_0 + 120 is_active_2 <= 120
 pipeline_ordering_constr3_1_2: start_2 + 120 group_tp_active_0
   + 120 is_active_2 >= 60
 pipeline_ordering_constr4_1_2: switch_1_nic_out_0 - group_tp_active_0
   + is_active_2 <= 1
 pipeline_ordering_constr1_2_3: - start_2 + start_3 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 - 12 hold_2_12
   - 13 hold_2_13 - 14 hold_2_14 - 15 hold_2_15 - 16 hold_2_16
   - 17 hold_2_17 - 18 hold_2_18 - 19 hold_2_19 - 20 hold_2_20
   - 21 hold_2_21 - 22 hold_2_22 - 23 hold_2_23 - 24 hold_2_24
   - 25 hold_2_25 - 26 hold_2_26 - 27 hold_2_27 - 28 hold_2_28
   - 29 hold_2_29 - 30 hold_2_30 - 31 hold_2_31 - 32 hold_2_32
   - 120 group_tp_active_0 + 120 is_active_3 <= 120
 pipeline_ordering_constr2_2_3: start_2 - start_3 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 12 hold_2_12
   + 13 hold_2_13 + 14 hold_2_14 + 15 hold_2_15 + 16 hold_2_16
   + 17 hold_2_17 + 18 hold_2_18 + 19 hold_2_19 + 20 hold_2_20
   + 21 hold_2_21 + 22 hold_2_22 + 23 hold_2_23 + 24 hold_2_24
   + 25 hold_2_25 + 26 hold_2_26 + 27 hold_2_27 + 28 hold_2_28
   + 29 hold_2_29 + 30 hold_2_30 + 31 hold_2_31 + 32 hold_2_32
   - 120 group_tp_active_0 + 120 is_active_3 <= 120
 pipeline_ordering_constr3_2_3: start_3 + 120 group_tp_active_0
   + 120 is_active_3 >= 60
 pipeline_ordering_constr4_2_3: switch_2_nic_out_0 - group_tp_active_0
   + is_active_3 <= 1
 force_min_throughput: flow_source_nic_in_0 + flow_source_nic_in_4
   + flow_source_nic_in_5 + flow_source_nic_in_6 + flow_source_nic_in_7
   + flow_source_nic_in_8 + flow_source_nic_in_9 + flow_source_nic_in_10
   + flow_source_nic_in_11 + flow_source_nic_in_12 + flow_source_nic_in_13
   + flow_source_nic_in_14 >= 1
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
 node_throughput_constraint_0: group_tp_active_0 = 0 -> - 2560 hold_0_1
   - 1280 hold_0_2 - 853.3333333333334 hold_0_3 - 640 hold_0_4
   - 512 hold_0_5 - 426.6666666666667 hold_0_6 - 365.7142857142857 hold_0_7
   - 320 hold_0_8 + flow_nic_in_0_0 + flow_1_0 + flow_2_0 + flow_3_0 <= 0
 node_throughput_constraint_1: group_tp_active_0 = 0 -> - 2560 hold_1_1
   - 1280 hold_1_2 - 853.3333333333334 hold_1_3 - 640 hold_1_4
   - 512 hold_1_5 - 426.6666666666667 hold_1_6 - 365.7142857142857 hold_1_7
   - 320 hold_1_8 + flow_nic_in_0_1 + flow_0_1 + flow_2_1 + flow_3_1 <= 0
 node_throughput_constraint_2: group_tp_active_0 = 0 -> - 2560 hold_2_1
   - 1280 hold_2_2 - 853.3333333333334 hold_2_3 - 640 hold_2_4
   - 512 hold_2_5 - 426.6666666666667 hold_2_6 - 365.7142857142857 hold_2_7
   - 320 hold_2_8 + flow_nic_in_0_2 + flow_0_2 + flow_1_2 + flow_3_2 <= 0
 node_throughput_constraint_3: group_tp_active_0 = 0 -> - 2560 hold_3_1
   - 1280 hold_3_2 - 853.3333333333334 hold_3_3 - 640 hold_3_4
   - 512 hold_3_5 - 426.6666666666667 hold_3_6 - 365.7142857142857 hold_3_7
   - 320 hold_3_8 + flow_nic_in_0_3 + flow_0_3 + flow_1_3 + flow_2_3 <= 0
 node_throughput_constraint_4: group_tp_active_4 = 0 -> - 12440 hold_4_1
   - 6220 hold_4_2 - 4146.666666666667 hold_4_3 - 3110 hold_4_4
   - 2488 hold_4_5 - 2073.333333333333 hold_4_6
   - 1777.142857142857 hold_4_7 - 1555 hold_4_8
   - 1382.222222222222 hold_4_9 - 1244 hold_4_10
   - 1130.909090909091 hold_4_11 - 1036.666666666667 hold_4_12
   - 956.9230769230769 hold_4_13 - 888.5714285714287 hold_4_14
   - 829.3333333333333 hold_4_15 - 777.5 hold_4_16
   - 731.7647058823529 hold_4_17 - 691.1111111111112 hold_4_18
   - 654.7368421052631 hold_4_19 - 622 hold_4_20 + flow_nic_in_4_4 <= 0
 node_throughput_constraint_5: group_tp_active_5 = 0 -> - 2400 hold_5_1
   - 1200 hold_5_2 - 800 hold_5_3 - 600 hold_5_4 - 480 hold_5_5
   - 400 hold_5_6 - 342.8571428571428 hold_5_7 - 300 hold_5_8
   - 266.6666666666667 hold_5_9 - 240 hold_5_10
   - 218.1818181818182 hold_5_11 - 200 hold_5_12 + flow_nic_in_5_5 <= 0
 node_throughput_constraint_6: group_tp_active_6 = 0 -> - 2400 hold_6_1
   - 1200 hold_6_2 - 800 hold_6_3 - 600 hold_6_4 - 480 hold_6_5
   - 400 hold_6_6 - 342.8571428571428 hold_6_7 - 300 hold_6_8
   - 266.6666666666667 hold_6_9 - 240 hold_6_10
   - 218.1818181818182 hold_6_11 - 200 hold_6_12 + flow_nic_in_6_6 <= 0
 node_throughput_constraint_7: group_tp_active_7 = 0 -> - 2400 hold_7_1
   - 1200 hold_7_2 - 800 hold_7_3 - 600 hold_7_4 - 480 hold_7_5
   - 400 hold_7_6 - 342.8571428571428 hold_7_7 - 300 hold_7_8
   - 266.6666666666667 hold_7_9 - 240 hold_7_10
   - 218.1818181818182 hold_7_11 - 200 hold_7_12 + flow_nic_in_7_7 <= 0
 node_throughput_constraint_8: group_tp_active_8 = 0 -> - 2400 hold_8_1
   - 1200 hold_8_2 - 800 hold_8_3 - 600 hold_8_4 - 480 hold_8_5
   - 400 hold_8_6 - 342.8571428571428 hold_8_7 - 300 hold_8_8
   - 266.6666666666667 hold_8_9 - 240 hold_8_10
   - 218.1818181818182 hold_8_11 - 200 hold_8_12 + flow_nic_in_8_8 <= 0
 node_throughput_constraint_9: group_tp_active_9 = 0 -> - 2560 hold_9_1
   - 1280 hold_9_2 - 853.3333333333334 hold_9_3 - 640 hold_9_4
   - 512 hold_9_5 - 426.6666666666667 hold_9_6 - 365.7142857142857 hold_9_7
   - 320 hold_9_8 + flow_nic_in_9_9 <= 0
 node_throughput_constraint_10: group_tp_active_10 = 0 -> - 2560 hold_10_1
   - 1280 hold_10_2 - 853.3333333333334 hold_10_3 - 640 hold_10_4
   - 512 hold_10_5 - 426.6666666666667 hold_10_6
   - 365.7142857142857 hold_10_7 - 320 hold_10_8 + flow_nic_in_10_10 <= 0
 node_throughput_constraint_11: group_tp_active_11 = 0 -> - 2560 hold_11_1
   - 1280 hold_11_2 - 853.3333333333334 hold_11_3 - 640 hold_11_4
   - 512 hold_11_5 - 426.6666666666667 hold_11_6
   - 365.7142857142857 hold_11_7 - 320 hold_11_8 + flow_nic_in_11_11 <= 0
 node_throughput_constraint_12: group_tp_active_12 = 0 -> - 12440 hold_12_1
   - 6220 hold_12_2 - 4146.666666666667 hold_12_3 - 3110 hold_12_4
   - 2488 hold_12_5 - 2073.333333333333 hold_12_6
   - 1777.142857142857 hold_12_7 - 1555 hold_12_8
   - 1382.222222222222 hold_12_9 - 1244 hold_12_10
   - 1130.909090909091 hold_12_11 - 1036.666666666667 hold_12_12
   - 956.9230769230769 hold_12_13 - 888.5714285714287 hold_12_14
   - 829.3333333333333 hold_12_15 - 777.5 hold_12_16
   - 731.7647058823529 hold_12_17 - 691.1111111111112 hold_12_18
   - 654.7368421052631 hold_12_19 - 622 hold_12_20 + flow_nic_in_12_12
   <= 0
 node_throughput_constraint_13: group_tp_active_13 = 0 -> - 2560 hold_13_1
   - 1280 hold_13_2 - 853.3333333333334 hold_13_3 - 640 hold_13_4
   - 512 hold_13_5 - 426.6666666666667 hold_13_6
   - 365.7142857142857 hold_13_7 - 320 hold_13_8 + flow_nic_in_13_13 <= 0
 node_throughput_constraint_14: group_tp_active_14 = 0 -> - 2400 hold_14_1
   - 1200 hold_14_2 - 800 hold_14_3 - 600 hold_14_4 - 480 hold_14_5
   - 400 hold_14_6 - 342.8571428571428 hold_14_7 - 300 hold_14_8
   - 266.6666666666667 hold_14_9 - 240 hold_14_10
   - 218.1818181818182 hold_14_11 - 200 hold_14_12 + flow_nic_in_14_14
   <= 0
 constr_intergroup_throughput_0_source_nic_in_0: switch_source_nic_in_0 = 1 ->
   - 2480.62015503876 hold_0_1 - 1259.842519685039 hold_0_2
   - 844.3271767810028 hold_0_3 - 634.9206349206349 hold_0_4
   - 508.7440381558029 hold_0_5 - 424.4031830238727 hold_0_6
   - 364.0500568828214 hold_0_7 - 318.7250996015936 hold_0_8
   + flow_nic_in_0_0 + flow_1_0 + flow_2_0 + flow_3_0 <= 0
 constr_intergroup_throughput_4_nic_out_0_nic_in_4: switch_nic_out_0_nic_in_4 = 1 ->
   - 11542.56553003943 hold_4_1 - 5987.245818794369 hold_4_2
   - 4041.913735683535 hold_4_3 - 3050.701980258721 hold_4_4
   - 2449.903992910246 hold_4_5 - 2046.810085969314 hold_4_6
   - 1757.620712797146 hold_4_7 - 1540.032806165083 hold_4_8
   - 1370.383630304867 hold_4_9 - 1234.40252040386 hold_4_10
   - 1122.971722597098 hold_4_11 - 1029.993169257519 hold_4_12
   - 951.2339660874386 hold_4_13 - 883.6639377741473 hold_4_14
   - 825.0567889771352 hold_4_15 - 773.7401066691546 hold_4_16
   - 728.4331952394198 hold_4_17 - 688.1387340791858 hold_4_18
   - 652.0685091271246 hold_4_19 - 619.5913386709168 hold_4_20
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_0_nic_in_5: switch_nic_out_0_nic_in_5 = 1 ->
   - 2364.532019704433 hold_5_1 - 1191.06699751861 hold_5_2
   - 796.0199004975125 hold_5_3 - 597.7584059775841 hold_5_4
   - 478.5643070787636 hold_5_5 - 399.002493765586 hold_5_6
   - 342.1240199572345 hold_5_7 - 299.4385527136619 hold_5_8
   - 266.2229617304492 hold_5_9 - 239.6405391912132 hold_5_10
   - 217.8847026781661 hold_5_11 - 199.7503121098626 hold_5_12
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_0_nic_in_6: switch_nic_out_0_nic_in_6 = 1 ->
   - 2364.532019704433 hold_6_1 - 1191.06699751861 hold_6_2
   - 796.0199004975125 hold_6_3 - 597.7584059775841 hold_6_4
   - 478.5643070787636 hold_6_5 - 399.002493765586 hold_6_6
   - 342.1240199572345 hold_6_7 - 299.4385527136619 hold_6_8
   - 266.2229617304492 hold_6_9 - 239.6405391912132 hold_6_10
   - 217.8847026781661 hold_6_11 - 199.7503121098626 hold_6_12
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_0_nic_in_7: switch_nic_out_0_nic_in_7 = 1 ->
   - 2330.097087378641 hold_7_1 - 1182.266009852217 hold_7_2
   - 792.0792079207921 hold_7_3 - 595.5334987593052 hold_7_4
   - 477.1371769383698 hold_7_5 - 398.0099502487562 hold_7_6
   - 341.3940256045519 hold_7_7 - 298.879202988792 hold_7_8
   - 265.78073089701 hold_7_9 - 239.2821535393818 hold_7_10
   - 217.5883952855847 hold_7_11 - 199.501246882793 hold_7_12
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_0_nic_in_8: switch_nic_out_0_nic_in_8 = 1 ->
   - 2330.097087378641 hold_8_1 - 1182.266009852217 hold_8_2
   - 792.0792079207921 hold_8_3 - 595.5334987593052 hold_8_4
   - 477.1371769383698 hold_8_5 - 398.0099502487562 hold_8_6
   - 341.3940256045519 hold_8_7 - 298.879202988792 hold_8_8
   - 265.78073089701 hold_8_9 - 239.2821535393818 hold_8_10
   - 217.5883952855847 hold_8_11 - 199.501246882793 hold_8_12
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_0_nic_in_9: switch_nic_out_0_nic_in_9 = 1 ->
   - 2480.62015503876 hold_9_1 - 1259.842519685039 hold_9_2
   - 844.3271767810028 hold_9_3 - 634.9206349206349 hold_9_4
   - 508.7440381558029 hold_9_5 - 424.4031830238727 hold_9_6
   - 364.0500568828214 hold_9_7 - 318.7250996015936 hold_9_8
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_0_nic_in_10: switch_nic_out_0_nic_in_10 = 1 ->
   - 2480.62015503876 hold_10_1 - 1259.842519685039 hold_10_2
   - 844.3271767810028 hold_10_3 - 634.9206349206349 hold_10_4
   - 508.7440381558029 hold_10_5 - 424.4031830238727 hold_10_6
   - 364.0500568828214 hold_10_7 - 318.7250996015936 hold_10_8
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_0_nic_in_11: switch_nic_out_0_nic_in_11 = 1 ->
   - 2480.62015503876 hold_11_1 - 1259.842519685039 hold_11_2
   - 844.3271767810028 hold_11_3 - 634.9206349206349 hold_11_4
   - 508.7440381558029 hold_11_5 - 424.4031830238727 hold_11_6
   - 364.0500568828214 hold_11_7 - 318.7250996015936 hold_11_8
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_0_nic_in_12: switch_nic_out_0_nic_in_12 = 1 ->
   - 10765.90220683687 hold_12_1 - 5771.282765019717 hold_12_2
   - 3942.322928220568 hold_12_3 - 2993.622909397185 hold_12_4
   - 2412.957036174959 hold_12_5 - 2020.956867841768 hold_12_6
   - 1738.522814618126 hold_12_7 - 1525.350990129361 hold_12_8
   - 1358.746108896292 hold_12_9 - 1224.951996455123 hold_12_10
   - 1115.14499574201 hold_12_11 - 1023.405042984657 hold_12_12
   - 945.6121014024553 hold_12_13 - 878.810356398573 hold_12_14
   - 820.8241232555838 hold_12_15 - 770.0164030825416 hold_12_16
   - 725.1318819037627 hold_12_17 - 685.1918151524333 hold_12_18
   - 649.4218370702931 hold_12_19 - 617.20126020193 hold_12_20
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_0_nic_in_13: switch_nic_out_0_nic_in_13 = 1 ->
   - 2480.62015503876 hold_13_1 - 1259.842519685039 hold_13_2
   - 844.3271767810028 hold_13_3 - 634.9206349206349 hold_13_4
   - 508.7440381558029 hold_13_5 - 424.4031830238727 hold_13_6
   - 364.0500568828214 hold_13_7 - 318.7250996015936 hold_13_8
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_0_nic_in_14: switch_nic_out_0_nic_in_14 = 1 ->
   - 2330.097087378641 hold_14_1 - 1182.266009852217 hold_14_2
   - 792.0792079207921 hold_14_3 - 595.5334987593052 hold_14_4
   - 477.1371769383698 hold_14_5 - 398.0099502487562 hold_14_6
   - 341.3940256045519 hold_14_7 - 298.879202988792 hold_14_8
   - 265.78073089701 hold_14_9 - 239.2821535393818 hold_14_10
   - 217.5883952855847 hold_14_11 - 199.501246882793 hold_14_12
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_4_source_nic_in_4: switch_source_nic_in_4 = 1 ->
   - 10765.90220683687 hold_4_1 - 5771.282765019717 hold_4_2
   - 3942.322928220568 hold_4_3 - 2993.622909397185 hold_4_4
   - 2412.957036174959 hold_4_5 - 2020.956867841768 hold_4_6
   - 1738.522814618126 hold_4_7 - 1525.350990129361 hold_4_8
   - 1358.746108896292 hold_4_9 - 1224.951996455123 hold_4_10
   - 1115.14499574201 hold_4_11 - 1023.405042984657 hold_4_12
   - 945.6121014024553 hold_4_13 - 878.810356398573 hold_4_14
   - 820.8241232555838 hold_4_15 - 770.0164030825416 hold_4_16
   - 725.1318819037627 hold_4_17 - 685.1918151524333 hold_4_18
   - 649.4218370702931 hold_4_19 - 617.20126020193 hold_4_20
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_0_nic_out_4_nic_in_0: switch_nic_out_4_nic_in_0 = 1 ->
   - 2519.685039370078 hold_0_1 - 1269.84126984127 hold_0_2
   - 848.8063660477454 hold_0_3 - 637.4501992031873 hold_0_4
   - 510.3668261562998 hold_0_5 - 425.531914893617 hold_0_6
   - 364.8802736602053 hold_0_7 - 319.3612774451097 hold_0_8
   + flow_nic_in_0_0 + flow_1_0 + flow_2_0 + flow_3_0 <= 0
 constr_intergroup_throughput_5_nic_out_4_nic_in_5: switch_nic_out_4_nic_in_5 = 1 ->
   - 2364.532019704433 hold_5_1 - 1191.06699751861 hold_5_2
   - 796.0199004975125 hold_5_3 - 597.7584059775841 hold_5_4
   - 478.5643070787636 hold_5_5 - 399.002493765586 hold_5_6
   - 342.1240199572345 hold_5_7 - 299.4385527136619 hold_5_8
   - 266.2229617304492 hold_5_9 - 239.6405391912132 hold_5_10
   - 217.8847026781661 hold_5_11 - 199.7503121098626 hold_5_12
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_4_nic_in_6: switch_nic_out_4_nic_in_6 = 1 ->
   - 2364.532019704433 hold_6_1 - 1191.06699751861 hold_6_2
   - 796.0199004975125 hold_6_3 - 597.7584059775841 hold_6_4
   - 478.5643070787636 hold_6_5 - 399.002493765586 hold_6_6
   - 342.1240199572345 hold_6_7 - 299.4385527136619 hold_6_8
   - 266.2229617304492 hold_6_9 - 239.6405391912132 hold_6_10
   - 217.8847026781661 hold_6_11 - 199.7503121098626 hold_6_12
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_4_nic_in_7: switch_nic_out_4_nic_in_7 = 1 ->
   - 2330.097087378641 hold_7_1 - 1182.266009852217 hold_7_2
   - 792.0792079207921 hold_7_3 - 595.5334987593052 hold_7_4
   - 477.1371769383698 hold_7_5 - 398.0099502487562 hold_7_6
   - 341.3940256045519 hold_7_7 - 298.879202988792 hold_7_8
   - 265.78073089701 hold_7_9 - 239.2821535393818 hold_7_10
   - 217.5883952855847 hold_7_11 - 199.501246882793 hold_7_12
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_4_nic_in_8: switch_nic_out_4_nic_in_8 = 1 ->
   - 2330.097087378641 hold_8_1 - 1182.266009852217 hold_8_2
   - 792.0792079207921 hold_8_3 - 595.5334987593052 hold_8_4
   - 477.1371769383698 hold_8_5 - 398.0099502487562 hold_8_6
   - 341.3940256045519 hold_8_7 - 298.879202988792 hold_8_8
   - 265.78073089701 hold_8_9 - 239.2821535393818 hold_8_10
   - 217.5883952855847 hold_8_11 - 199.501246882793 hold_8_12
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_4_nic_in_9: switch_nic_out_4_nic_in_9 = 1 ->
   - 2480.62015503876 hold_9_1 - 1259.842519685039 hold_9_2
   - 844.3271767810028 hold_9_3 - 634.9206349206349 hold_9_4
   - 508.7440381558029 hold_9_5 - 424.4031830238727 hold_9_6
   - 364.0500568828214 hold_9_7 - 318.7250996015936 hold_9_8
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_4_nic_in_10: switch_nic_out_4_nic_in_10 = 1 ->
   - 2480.62015503876 hold_10_1 - 1259.842519685039 hold_10_2
   - 844.3271767810028 hold_10_3 - 634.9206349206349 hold_10_4
   - 508.7440381558029 hold_10_5 - 424.4031830238727 hold_10_6
   - 364.0500568828214 hold_10_7 - 318.7250996015936 hold_10_8
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_4_nic_in_11: switch_nic_out_4_nic_in_11 = 1 ->
   - 2480.62015503876 hold_11_1 - 1259.842519685039 hold_11_2
   - 844.3271767810028 hold_11_3 - 634.9206349206349 hold_11_4
   - 508.7440381558029 hold_11_5 - 424.4031830238727 hold_11_6
   - 364.0500568828214 hold_11_7 - 318.7250996015936 hold_11_8
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_4_nic_in_12: switch_nic_out_4_nic_in_12 = 1 ->
   - 10765.90220683687 hold_12_1 - 5771.282765019717 hold_12_2
   - 3942.322928220568 hold_12_3 - 2993.622909397185 hold_12_4
   - 2412.957036174959 hold_12_5 - 2020.956867841768 hold_12_6
   - 1738.522814618126 hold_12_7 - 1525.350990129361 hold_12_8
   - 1358.746108896292 hold_12_9 - 1224.951996455123 hold_12_10
   - 1115.14499574201 hold_12_11 - 1023.405042984657 hold_12_12
   - 945.6121014024553 hold_12_13 - 878.810356398573 hold_12_14
   - 820.8241232555838 hold_12_15 - 770.0164030825416 hold_12_16
   - 725.1318819037627 hold_12_17 - 685.1918151524333 hold_12_18
   - 649.4218370702931 hold_12_19 - 617.20126020193 hold_12_20
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_4_nic_in_13: switch_nic_out_4_nic_in_13 = 1 ->
   - 2480.62015503876 hold_13_1 - 1259.842519685039 hold_13_2
   - 844.3271767810028 hold_13_3 - 634.9206349206349 hold_13_4
   - 508.7440381558029 hold_13_5 - 424.4031830238727 hold_13_6
   - 364.0500568828214 hold_13_7 - 318.7250996015936 hold_13_8
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_4_nic_in_14: switch_nic_out_4_nic_in_14 = 1 ->
   - 2330.097087378641 hold_14_1 - 1182.266009852217 hold_14_2
   - 792.0792079207921 hold_14_3 - 595.5334987593052 hold_14_4
   - 477.1371769383698 hold_14_5 - 398.0099502487562 hold_14_6
   - 341.3940256045519 hold_14_7 - 298.879202988792 hold_14_8
   - 265.78073089701 hold_14_9 - 239.2821535393818 hold_14_10
   - 217.5883952855847 hold_14_11 - 199.501246882793 hold_14_12
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_5_source_nic_in_5: switch_source_nic_in_5 = 1 ->
   - 2330.097087378641 hold_5_1 - 1182.266009852217 hold_5_2
   - 792.0792079207921 hold_5_3 - 595.5334987593052 hold_5_4
   - 477.1371769383698 hold_5_5 - 398.0099502487562 hold_5_6
   - 341.3940256045519 hold_5_7 - 298.879202988792 hold_5_8
   - 265.78073089701 hold_5_9 - 239.2821535393818 hold_5_10
   - 217.5883952855847 hold_5_11 - 199.501246882793 hold_5_12
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_0_nic_out_5_nic_in_0: switch_nic_out_5_nic_in_0 = 1 ->
   - 2519.685039370078 hold_0_1 - 1269.84126984127 hold_0_2
   - 848.8063660477454 hold_0_3 - 637.4501992031873 hold_0_4
   - 510.3668261562998 hold_0_5 - 425.531914893617 hold_0_6
   - 364.8802736602053 hold_0_7 - 319.3612774451097 hold_0_8
   + flow_nic_in_0_0 + flow_1_0 + flow_2_0 + flow_3_0 <= 0
 constr_intergroup_throughput_4_nic_out_5_nic_in_4: switch_nic_out_5_nic_in_4 = 1 ->
   - 11542.56553003943 hold_4_1 - 5987.245818794369 hold_4_2
   - 4041.913735683535 hold_4_3 - 3050.701980258721 hold_4_4
   - 2449.903992910246 hold_4_5 - 2046.810085969314 hold_4_6
   - 1757.620712797146 hold_4_7 - 1540.032806165083 hold_4_8
   - 1370.383630304867 hold_4_9 - 1234.40252040386 hold_4_10
   - 1122.971722597098 hold_4_11 - 1029.993169257519 hold_4_12
   - 951.2339660874386 hold_4_13 - 883.6639377741473 hold_4_14
   - 825.0567889771352 hold_4_15 - 773.7401066691546 hold_4_16
   - 728.4331952394198 hold_4_17 - 688.1387340791858 hold_4_18
   - 652.0685091271246 hold_4_19 - 619.5913386709168 hold_4_20
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_6_nic_out_5_nic_in_6: switch_nic_out_5_nic_in_6 = 1 ->
   - 2364.532019704433 hold_6_1 - 1191.06699751861 hold_6_2
   - 796.0199004975125 hold_6_3 - 597.7584059775841 hold_6_4
   - 478.5643070787636 hold_6_5 - 399.002493765586 hold_6_6
   - 342.1240199572345 hold_6_7 - 299.4385527136619 hold_6_8
   - 266.2229617304492 hold_6_9 - 239.6405391912132 hold_6_10
   - 217.8847026781661 hold_6_11 - 199.7503121098626 hold_6_12
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_5_nic_in_7: switch_nic_out_5_nic_in_7 = 1 ->
   - 2330.097087378641 hold_7_1 - 1182.266009852217 hold_7_2
   - 792.0792079207921 hold_7_3 - 595.5334987593052 hold_7_4
   - 477.1371769383698 hold_7_5 - 398.0099502487562 hold_7_6
   - 341.3940256045519 hold_7_7 - 298.879202988792 hold_7_8
   - 265.78073089701 hold_7_9 - 239.2821535393818 hold_7_10
   - 217.5883952855847 hold_7_11 - 199.501246882793 hold_7_12
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_5_nic_in_8: switch_nic_out_5_nic_in_8 = 1 ->
   - 2330.097087378641 hold_8_1 - 1182.266009852217 hold_8_2
   - 792.0792079207921 hold_8_3 - 595.5334987593052 hold_8_4
   - 477.1371769383698 hold_8_5 - 398.0099502487562 hold_8_6
   - 341.3940256045519 hold_8_7 - 298.879202988792 hold_8_8
   - 265.78073089701 hold_8_9 - 239.2821535393818 hold_8_10
   - 217.5883952855847 hold_8_11 - 199.501246882793 hold_8_12
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_5_nic_in_9: switch_nic_out_5_nic_in_9 = 1 ->
   - 2480.62015503876 hold_9_1 - 1259.842519685039 hold_9_2
   - 844.3271767810028 hold_9_3 - 634.9206349206349 hold_9_4
   - 508.7440381558029 hold_9_5 - 424.4031830238727 hold_9_6
   - 364.0500568828214 hold_9_7 - 318.7250996015936 hold_9_8
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_5_nic_in_10: switch_nic_out_5_nic_in_10 = 1 ->
   - 2480.62015503876 hold_10_1 - 1259.842519685039 hold_10_2
   - 844.3271767810028 hold_10_3 - 634.9206349206349 hold_10_4
   - 508.7440381558029 hold_10_5 - 424.4031830238727 hold_10_6
   - 364.0500568828214 hold_10_7 - 318.7250996015936 hold_10_8
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_5_nic_in_11: switch_nic_out_5_nic_in_11 = 1 ->
   - 2480.62015503876 hold_11_1 - 1259.842519685039 hold_11_2
   - 844.3271767810028 hold_11_3 - 634.9206349206349 hold_11_4
   - 508.7440381558029 hold_11_5 - 424.4031830238727 hold_11_6
   - 364.0500568828214 hold_11_7 - 318.7250996015936 hold_11_8
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_5_nic_in_12: switch_nic_out_5_nic_in_12 = 1 ->
   - 10765.90220683687 hold_12_1 - 5771.282765019717 hold_12_2
   - 3942.322928220568 hold_12_3 - 2993.622909397185 hold_12_4
   - 2412.957036174959 hold_12_5 - 2020.956867841768 hold_12_6
   - 1738.522814618126 hold_12_7 - 1525.350990129361 hold_12_8
   - 1358.746108896292 hold_12_9 - 1224.951996455123 hold_12_10
   - 1115.14499574201 hold_12_11 - 1023.405042984657 hold_12_12
   - 945.6121014024553 hold_12_13 - 878.810356398573 hold_12_14
   - 820.8241232555838 hold_12_15 - 770.0164030825416 hold_12_16
   - 725.1318819037627 hold_12_17 - 685.1918151524333 hold_12_18
   - 649.4218370702931 hold_12_19 - 617.20126020193 hold_12_20
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_5_nic_in_13: switch_nic_out_5_nic_in_13 = 1 ->
   - 2480.62015503876 hold_13_1 - 1259.842519685039 hold_13_2
   - 844.3271767810028 hold_13_3 - 634.9206349206349 hold_13_4
   - 508.7440381558029 hold_13_5 - 424.4031830238727 hold_13_6
   - 364.0500568828214 hold_13_7 - 318.7250996015936 hold_13_8
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_5_nic_in_14: switch_nic_out_5_nic_in_14 = 1 ->
   - 2330.097087378641 hold_14_1 - 1182.266009852217 hold_14_2
   - 792.0792079207921 hold_14_3 - 595.5334987593052 hold_14_4
   - 477.1371769383698 hold_14_5 - 398.0099502487562 hold_14_6
   - 341.3940256045519 hold_14_7 - 298.879202988792 hold_14_8
   - 265.78073089701 hold_14_9 - 239.2821535393818 hold_14_10
   - 217.5883952855847 hold_14_11 - 199.501246882793 hold_14_12
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_6_source_nic_in_6: switch_source_nic_in_6 = 1 ->
   - 2330.097087378641 hold_6_1 - 1182.266009852217 hold_6_2
   - 792.0792079207921 hold_6_3 - 595.5334987593052 hold_6_4
   - 477.1371769383698 hold_6_5 - 398.0099502487562 hold_6_6
   - 341.3940256045519 hold_6_7 - 298.879202988792 hold_6_8
   - 265.78073089701 hold_6_9 - 239.2821535393818 hold_6_10
   - 217.5883952855847 hold_6_11 - 199.501246882793 hold_6_12
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_0_nic_out_6_nic_in_0: switch_nic_out_6_nic_in_0 = 1 ->
   - 2519.685039370078 hold_0_1 - 1269.84126984127 hold_0_2
   - 848.8063660477454 hold_0_3 - 637.4501992031873 hold_0_4
   - 510.3668261562998 hold_0_5 - 425.531914893617 hold_0_6
   - 364.8802736602053 hold_0_7 - 319.3612774451097 hold_0_8
   + flow_nic_in_0_0 + flow_1_0 + flow_2_0 + flow_3_0 <= 0
 constr_intergroup_throughput_4_nic_out_6_nic_in_4: switch_nic_out_6_nic_in_4 = 1 ->
   - 11542.56553003943 hold_4_1 - 5987.245818794369 hold_4_2
   - 4041.913735683535 hold_4_3 - 3050.701980258721 hold_4_4
   - 2449.903992910246 hold_4_5 - 2046.810085969314 hold_4_6
   - 1757.620712797146 hold_4_7 - 1540.032806165083 hold_4_8
   - 1370.383630304867 hold_4_9 - 1234.40252040386 hold_4_10
   - 1122.971722597098 hold_4_11 - 1029.993169257519 hold_4_12
   - 951.2339660874386 hold_4_13 - 883.6639377741473 hold_4_14
   - 825.0567889771352 hold_4_15 - 773.7401066691546 hold_4_16
   - 728.4331952394198 hold_4_17 - 688.1387340791858 hold_4_18
   - 652.0685091271246 hold_4_19 - 619.5913386709168 hold_4_20
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_6_nic_in_5: switch_nic_out_6_nic_in_5 = 1 ->
   - 2364.532019704433 hold_5_1 - 1191.06699751861 hold_5_2
   - 796.0199004975125 hold_5_3 - 597.7584059775841 hold_5_4
   - 478.5643070787636 hold_5_5 - 399.002493765586 hold_5_6
   - 342.1240199572345 hold_5_7 - 299.4385527136619 hold_5_8
   - 266.2229617304492 hold_5_9 - 239.6405391912132 hold_5_10
   - 217.8847026781661 hold_5_11 - 199.7503121098626 hold_5_12
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_7_nic_out_6_nic_in_7: switch_nic_out_6_nic_in_7 = 1 ->
   - 2330.097087378641 hold_7_1 - 1182.266009852217 hold_7_2
   - 792.0792079207921 hold_7_3 - 595.5334987593052 hold_7_4
   - 477.1371769383698 hold_7_5 - 398.0099502487562 hold_7_6
   - 341.3940256045519 hold_7_7 - 298.879202988792 hold_7_8
   - 265.78073089701 hold_7_9 - 239.2821535393818 hold_7_10
   - 217.5883952855847 hold_7_11 - 199.501246882793 hold_7_12
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_6_nic_in_8: switch_nic_out_6_nic_in_8 = 1 ->
   - 2330.097087378641 hold_8_1 - 1182.266009852217 hold_8_2
   - 792.0792079207921 hold_8_3 - 595.5334987593052 hold_8_4
   - 477.1371769383698 hold_8_5 - 398.0099502487562 hold_8_6
   - 341.3940256045519 hold_8_7 - 298.879202988792 hold_8_8
   - 265.78073089701 hold_8_9 - 239.2821535393818 hold_8_10
   - 217.5883952855847 hold_8_11 - 199.501246882793 hold_8_12
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_6_nic_in_9: switch_nic_out_6_nic_in_9 = 1 ->
   - 2480.62015503876 hold_9_1 - 1259.842519685039 hold_9_2
   - 844.3271767810028 hold_9_3 - 634.9206349206349 hold_9_4
   - 508.7440381558029 hold_9_5 - 424.4031830238727 hold_9_6
   - 364.0500568828214 hold_9_7 - 318.7250996015936 hold_9_8
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_6_nic_in_10: switch_nic_out_6_nic_in_10 = 1 ->
   - 2480.62015503876 hold_10_1 - 1259.842519685039 hold_10_2
   - 844.3271767810028 hold_10_3 - 634.9206349206349 hold_10_4
   - 508.7440381558029 hold_10_5 - 424.4031830238727 hold_10_6
   - 364.0500568828214 hold_10_7 - 318.7250996015936 hold_10_8
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_6_nic_in_11: switch_nic_out_6_nic_in_11 = 1 ->
   - 2480.62015503876 hold_11_1 - 1259.842519685039 hold_11_2
   - 844.3271767810028 hold_11_3 - 634.9206349206349 hold_11_4
   - 508.7440381558029 hold_11_5 - 424.4031830238727 hold_11_6
   - 364.0500568828214 hold_11_7 - 318.7250996015936 hold_11_8
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_6_nic_in_12: switch_nic_out_6_nic_in_12 = 1 ->
   - 10765.90220683687 hold_12_1 - 5771.282765019717 hold_12_2
   - 3942.322928220568 hold_12_3 - 2993.622909397185 hold_12_4
   - 2412.957036174959 hold_12_5 - 2020.956867841768 hold_12_6
   - 1738.522814618126 hold_12_7 - 1525.350990129361 hold_12_8
   - 1358.746108896292 hold_12_9 - 1224.951996455123 hold_12_10
   - 1115.14499574201 hold_12_11 - 1023.405042984657 hold_12_12
   - 945.6121014024553 hold_12_13 - 878.810356398573 hold_12_14
   - 820.8241232555838 hold_12_15 - 770.0164030825416 hold_12_16
   - 725.1318819037627 hold_12_17 - 685.1918151524333 hold_12_18
   - 649.4218370702931 hold_12_19 - 617.20126020193 hold_12_20
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_6_nic_in_13: switch_nic_out_6_nic_in_13 = 1 ->
   - 2480.62015503876 hold_13_1 - 1259.842519685039 hold_13_2
   - 844.3271767810028 hold_13_3 - 634.9206349206349 hold_13_4
   - 508.7440381558029 hold_13_5 - 424.4031830238727 hold_13_6
   - 364.0500568828214 hold_13_7 - 318.7250996015936 hold_13_8
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_6_nic_in_14: switch_nic_out_6_nic_in_14 = 1 ->
   - 2330.097087378641 hold_14_1 - 1182.266009852217 hold_14_2
   - 792.0792079207921 hold_14_3 - 595.5334987593052 hold_14_4
   - 477.1371769383698 hold_14_5 - 398.0099502487562 hold_14_6
   - 341.3940256045519 hold_14_7 - 298.879202988792 hold_14_8
   - 265.78073089701 hold_14_9 - 239.2821535393818 hold_14_10
   - 217.5883952855847 hold_14_11 - 199.501246882793 hold_14_12
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_7_source_nic_in_7: switch_source_nic_in_7 = 1 ->
   - 2330.097087378641 hold_7_1 - 1182.266009852217 hold_7_2
   - 792.0792079207921 hold_7_3 - 595.5334987593052 hold_7_4
   - 477.1371769383698 hold_7_5 - 398.0099502487562 hold_7_6
   - 341.3940256045519 hold_7_7 - 298.879202988792 hold_7_8
   - 265.78073089701 hold_7_9 - 239.2821535393818 hold_7_10
   - 217.5883952855847 hold_7_11 - 199.501246882793 hold_7_12
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_0_nic_out_7_nic_in_0: switch_nic_out_7_nic_in_0 = 1 ->
   - 2480.62015503876 hold_0_1 - 1259.842519685039 hold_0_2
   - 844.3271767810028 hold_0_3 - 634.9206349206349 hold_0_4
   - 508.7440381558029 hold_0_5 - 424.4031830238727 hold_0_6
   - 364.0500568828214 hold_0_7 - 318.7250996015936 hold_0_8
   + flow_nic_in_0_0 + flow_1_0 + flow_2_0 + flow_3_0 <= 0
 constr_intergroup_throughput_4_nic_out_7_nic_in_4: switch_nic_out_7_nic_in_4 = 1 ->
   - 10765.90220683687 hold_4_1 - 5771.282765019717 hold_4_2
   - 3942.322928220568 hold_4_3 - 2993.622909397185 hold_4_4
   - 2412.957036174959 hold_4_5 - 2020.956867841768 hold_4_6
   - 1738.522814618126 hold_4_7 - 1525.350990129361 hold_4_8
   - 1358.746108896292 hold_4_9 - 1224.951996455123 hold_4_10
   - 1115.14499574201 hold_4_11 - 1023.405042984657 hold_4_12
   - 945.6121014024553 hold_4_13 - 878.810356398573 hold_4_14
   - 820.8241232555838 hold_4_15 - 770.0164030825416 hold_4_16
   - 725.1318819037627 hold_4_17 - 685.1918151524333 hold_4_18
   - 649.4218370702931 hold_4_19 - 617.20126020193 hold_4_20
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_7_nic_in_5: switch_nic_out_7_nic_in_5 = 1 ->
   - 2330.097087378641 hold_5_1 - 1182.266009852217 hold_5_2
   - 792.0792079207921 hold_5_3 - 595.5334987593052 hold_5_4
   - 477.1371769383698 hold_5_5 - 398.0099502487562 hold_5_6
   - 341.3940256045519 hold_5_7 - 298.879202988792 hold_5_8
   - 265.78073089701 hold_5_9 - 239.2821535393818 hold_5_10
   - 217.5883952855847 hold_5_11 - 199.501246882793 hold_5_12
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_7_nic_in_6: switch_nic_out_7_nic_in_6 = 1 ->
   - 2330.097087378641 hold_6_1 - 1182.266009852217 hold_6_2
   - 792.0792079207921 hold_6_3 - 595.5334987593052 hold_6_4
   - 477.1371769383698 hold_6_5 - 398.0099502487562 hold_6_6
   - 341.3940256045519 hold_6_7 - 298.879202988792 hold_6_8
   - 265.78073089701 hold_6_9 - 239.2821535393818 hold_6_10
   - 217.5883952855847 hold_6_11 - 199.501246882793 hold_6_12
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_8_nic_out_7_nic_in_8: switch_nic_out_7_nic_in_8 = 1 ->
   - 2364.532019704433 hold_8_1 - 1191.06699751861 hold_8_2
   - 796.0199004975125 hold_8_3 - 597.7584059775841 hold_8_4
   - 478.5643070787636 hold_8_5 - 399.002493765586 hold_8_6
   - 342.1240199572345 hold_8_7 - 299.4385527136619 hold_8_8
   - 266.2229617304492 hold_8_9 - 239.6405391912132 hold_8_10
   - 217.8847026781661 hold_8_11 - 199.7503121098626 hold_8_12
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_7_nic_in_9: switch_nic_out_7_nic_in_9 = 1 ->
   - 2519.685039370078 hold_9_1 - 1269.84126984127 hold_9_2
   - 848.8063660477454 hold_9_3 - 637.4501992031873 hold_9_4
   - 510.3668261562998 hold_9_5 - 425.531914893617 hold_9_6
   - 364.8802736602053 hold_9_7 - 319.3612774451097 hold_9_8
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_7_nic_in_10: switch_nic_out_7_nic_in_10 = 1 ->
   - 2519.685039370078 hold_10_1 - 1269.84126984127 hold_10_2
   - 848.8063660477454 hold_10_3 - 637.4501992031873 hold_10_4
   - 510.3668261562998 hold_10_5 - 425.531914893617 hold_10_6
   - 364.8802736602053 hold_10_7 - 319.3612774451097 hold_10_8
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_7_nic_in_11: switch_nic_out_7_nic_in_11 = 1 ->
   - 2480.62015503876 hold_11_1 - 1259.842519685039 hold_11_2
   - 844.3271767810028 hold_11_3 - 634.9206349206349 hold_11_4
   - 508.7440381558029 hold_11_5 - 424.4031830238727 hold_11_6
   - 364.0500568828214 hold_11_7 - 318.7250996015936 hold_11_8
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_7_nic_in_12: switch_nic_out_7_nic_in_12 = 1 ->
   - 10765.90220683687 hold_12_1 - 5771.282765019717 hold_12_2
   - 3942.322928220568 hold_12_3 - 2993.622909397185 hold_12_4
   - 2412.957036174959 hold_12_5 - 2020.956867841768 hold_12_6
   - 1738.522814618126 hold_12_7 - 1525.350990129361 hold_12_8
   - 1358.746108896292 hold_12_9 - 1224.951996455123 hold_12_10
   - 1115.14499574201 hold_12_11 - 1023.405042984657 hold_12_12
   - 945.6121014024553 hold_12_13 - 878.810356398573 hold_12_14
   - 820.8241232555838 hold_12_15 - 770.0164030825416 hold_12_16
   - 725.1318819037627 hold_12_17 - 685.1918151524333 hold_12_18
   - 649.4218370702931 hold_12_19 - 617.20126020193 hold_12_20
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_7_nic_in_13: switch_nic_out_7_nic_in_13 = 1 ->
   - 2480.62015503876 hold_13_1 - 1259.842519685039 hold_13_2
   - 844.3271767810028 hold_13_3 - 634.9206349206349 hold_13_4
   - 508.7440381558029 hold_13_5 - 424.4031830238727 hold_13_6
   - 364.0500568828214 hold_13_7 - 318.7250996015936 hold_13_8
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_7_nic_in_14: switch_nic_out_7_nic_in_14 = 1 ->
   - 2330.097087378641 hold_14_1 - 1182.266009852217 hold_14_2
   - 792.0792079207921 hold_14_3 - 595.5334987593052 hold_14_4
   - 477.1371769383698 hold_14_5 - 398.0099502487562 hold_14_6
   - 341.3940256045519 hold_14_7 - 298.879202988792 hold_14_8
   - 265.78073089701 hold_14_9 - 239.2821535393818 hold_14_10
   - 217.5883952855847 hold_14_11 - 199.501246882793 hold_14_12
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_8_source_nic_in_8: switch_source_nic_in_8 = 1 ->
   - 2330.097087378641 hold_8_1 - 1182.266009852217 hold_8_2
   - 792.0792079207921 hold_8_3 - 595.5334987593052 hold_8_4
   - 477.1371769383698 hold_8_5 - 398.0099502487562 hold_8_6
   - 341.3940256045519 hold_8_7 - 298.879202988792 hold_8_8
   - 265.78073089701 hold_8_9 - 239.2821535393818 hold_8_10
   - 217.5883952855847 hold_8_11 - 199.501246882793 hold_8_12
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_0_nic_out_8_nic_in_0: switch_nic_out_8_nic_in_0 = 1 ->
   - 2480.62015503876 hold_0_1 - 1259.842519685039 hold_0_2
   - 844.3271767810028 hold_0_3 - 634.9206349206349 hold_0_4
   - 508.7440381558029 hold_0_5 - 424.4031830238727 hold_0_6
   - 364.0500568828214 hold_0_7 - 318.7250996015936 hold_0_8
   + flow_nic_in_0_0 + flow_1_0 + flow_2_0 + flow_3_0 <= 0
 constr_intergroup_throughput_4_nic_out_8_nic_in_4: switch_nic_out_8_nic_in_4 = 1 ->
   - 10765.90220683687 hold_4_1 - 5771.282765019717 hold_4_2
   - 3942.322928220568 hold_4_3 - 2993.622909397185 hold_4_4
   - 2412.957036174959 hold_4_5 - 2020.956867841768 hold_4_6
   - 1738.522814618126 hold_4_7 - 1525.350990129361 hold_4_8
   - 1358.746108896292 hold_4_9 - 1224.951996455123 hold_4_10
   - 1115.14499574201 hold_4_11 - 1023.405042984657 hold_4_12
   - 945.6121014024553 hold_4_13 - 878.810356398573 hold_4_14
   - 820.8241232555838 hold_4_15 - 770.0164030825416 hold_4_16
   - 725.1318819037627 hold_4_17 - 685.1918151524333 hold_4_18
   - 649.4218370702931 hold_4_19 - 617.20126020193 hold_4_20
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_8_nic_in_5: switch_nic_out_8_nic_in_5 = 1 ->
   - 2330.097087378641 hold_5_1 - 1182.266009852217 hold_5_2
   - 792.0792079207921 hold_5_3 - 595.5334987593052 hold_5_4
   - 477.1371769383698 hold_5_5 - 398.0099502487562 hold_5_6
   - 341.3940256045519 hold_5_7 - 298.879202988792 hold_5_8
   - 265.78073089701 hold_5_9 - 239.2821535393818 hold_5_10
   - 217.5883952855847 hold_5_11 - 199.501246882793 hold_5_12
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_8_nic_in_6: switch_nic_out_8_nic_in_6 = 1 ->
   - 2330.097087378641 hold_6_1 - 1182.266009852217 hold_6_2
   - 792.0792079207921 hold_6_3 - 595.5334987593052 hold_6_4
   - 477.1371769383698 hold_6_5 - 398.0099502487562 hold_6_6
   - 341.3940256045519 hold_6_7 - 298.879202988792 hold_6_8
   - 265.78073089701 hold_6_9 - 239.2821535393818 hold_6_10
   - 217.5883952855847 hold_6_11 - 199.501246882793 hold_6_12
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_8_nic_in_7: switch_nic_out_8_nic_in_7 = 1 ->
   - 2364.532019704433 hold_7_1 - 1191.06699751861 hold_7_2
   - 796.0199004975125 hold_7_3 - 597.7584059775841 hold_7_4
   - 478.5643070787636 hold_7_5 - 399.002493765586 hold_7_6
   - 342.1240199572345 hold_7_7 - 299.4385527136619 hold_7_8
   - 266.2229617304492 hold_7_9 - 239.6405391912132 hold_7_10
   - 217.8847026781661 hold_7_11 - 199.7503121098626 hold_7_12
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_9_nic_out_8_nic_in_9: switch_nic_out_8_nic_in_9 = 1 ->
   - 2519.685039370078 hold_9_1 - 1269.84126984127 hold_9_2
   - 848.8063660477454 hold_9_3 - 637.4501992031873 hold_9_4
   - 510.3668261562998 hold_9_5 - 425.531914893617 hold_9_6
   - 364.8802736602053 hold_9_7 - 319.3612774451097 hold_9_8
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_8_nic_in_10: switch_nic_out_8_nic_in_10 = 1 ->
   - 2519.685039370078 hold_10_1 - 1269.84126984127 hold_10_2
   - 848.8063660477454 hold_10_3 - 637.4501992031873 hold_10_4
   - 510.3668261562998 hold_10_5 - 425.531914893617 hold_10_6
   - 364.8802736602053 hold_10_7 - 319.3612774451097 hold_10_8
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_8_nic_in_11: switch_nic_out_8_nic_in_11 = 1 ->
   - 2480.62015503876 hold_11_1 - 1259.842519685039 hold_11_2
   - 844.3271767810028 hold_11_3 - 634.9206349206349 hold_11_4
   - 508.7440381558029 hold_11_5 - 424.4031830238727 hold_11_6
   - 364.0500568828214 hold_11_7 - 318.7250996015936 hold_11_8
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_8_nic_in_12: switch_nic_out_8_nic_in_12 = 1 ->
   - 10765.90220683687 hold_12_1 - 5771.282765019717 hold_12_2
   - 3942.322928220568 hold_12_3 - 2993.622909397185 hold_12_4
   - 2412.957036174959 hold_12_5 - 2020.956867841768 hold_12_6
   - 1738.522814618126 hold_12_7 - 1525.350990129361 hold_12_8
   - 1358.746108896292 hold_12_9 - 1224.951996455123 hold_12_10
   - 1115.14499574201 hold_12_11 - 1023.405042984657 hold_12_12
   - 945.6121014024553 hold_12_13 - 878.810356398573 hold_12_14
   - 820.8241232555838 hold_12_15 - 770.0164030825416 hold_12_16
   - 725.1318819037627 hold_12_17 - 685.1918151524333 hold_12_18
   - 649.4218370702931 hold_12_19 - 617.20126020193 hold_12_20
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_8_nic_in_13: switch_nic_out_8_nic_in_13 = 1 ->
   - 2480.62015503876 hold_13_1 - 1259.842519685039 hold_13_2
   - 844.3271767810028 hold_13_3 - 634.9206349206349 hold_13_4
   - 508.7440381558029 hold_13_5 - 424.4031830238727 hold_13_6
   - 364.0500568828214 hold_13_7 - 318.7250996015936 hold_13_8
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_8_nic_in_14: switch_nic_out_8_nic_in_14 = 1 ->
   - 2330.097087378641 hold_14_1 - 1182.266009852217 hold_14_2
   - 792.0792079207921 hold_14_3 - 595.5334987593052 hold_14_4
   - 477.1371769383698 hold_14_5 - 398.0099502487562 hold_14_6
   - 341.3940256045519 hold_14_7 - 298.879202988792 hold_14_8
   - 265.78073089701 hold_14_9 - 239.2821535393818 hold_14_10
   - 217.5883952855847 hold_14_11 - 199.501246882793 hold_14_12
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_9_source_nic_in_9: switch_source_nic_in_9 = 1 ->
   - 2480.62015503876 hold_9_1 - 1259.842519685039 hold_9_2
   - 844.3271767810028 hold_9_3 - 634.9206349206349 hold_9_4
   - 508.7440381558029 hold_9_5 - 424.4031830238727 hold_9_6
   - 364.0500568828214 hold_9_7 - 318.7250996015936 hold_9_8
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_0_nic_out_9_nic_in_0: switch_nic_out_9_nic_in_0 = 1 ->
   - 2480.62015503876 hold_0_1 - 1259.842519685039 hold_0_2
   - 844.3271767810028 hold_0_3 - 634.9206349206349 hold_0_4
   - 508.7440381558029 hold_0_5 - 424.4031830238727 hold_0_6
   - 364.0500568828214 hold_0_7 - 318.7250996015936 hold_0_8
   + flow_nic_in_0_0 + flow_1_0 + flow_2_0 + flow_3_0 <= 0
 constr_intergroup_throughput_4_nic_out_9_nic_in_4: switch_nic_out_9_nic_in_4 = 1 ->
   - 10765.90220683687 hold_4_1 - 5771.282765019717 hold_4_2
   - 3942.322928220568 hold_4_3 - 2993.622909397185 hold_4_4
   - 2412.957036174959 hold_4_5 - 2020.956867841768 hold_4_6
   - 1738.522814618126 hold_4_7 - 1525.350990129361 hold_4_8
   - 1358.746108896292 hold_4_9 - 1224.951996455123 hold_4_10
   - 1115.14499574201 hold_4_11 - 1023.405042984657 hold_4_12
   - 945.6121014024553 hold_4_13 - 878.810356398573 hold_4_14
   - 820.8241232555838 hold_4_15 - 770.0164030825416 hold_4_16
   - 725.1318819037627 hold_4_17 - 685.1918151524333 hold_4_18
   - 649.4218370702931 hold_4_19 - 617.20126020193 hold_4_20
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_9_nic_in_5: switch_nic_out_9_nic_in_5 = 1 ->
   - 2330.097087378641 hold_5_1 - 1182.266009852217 hold_5_2
   - 792.0792079207921 hold_5_3 - 595.5334987593052 hold_5_4
   - 477.1371769383698 hold_5_5 - 398.0099502487562 hold_5_6
   - 341.3940256045519 hold_5_7 - 298.879202988792 hold_5_8
   - 265.78073089701 hold_5_9 - 239.2821535393818 hold_5_10
   - 217.5883952855847 hold_5_11 - 199.501246882793 hold_5_12
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_9_nic_in_6: switch_nic_out_9_nic_in_6 = 1 ->
   - 2330.097087378641 hold_6_1 - 1182.266009852217 hold_6_2
   - 792.0792079207921 hold_6_3 - 595.5334987593052 hold_6_4
   - 477.1371769383698 hold_6_5 - 398.0099502487562 hold_6_6
   - 341.3940256045519 hold_6_7 - 298.879202988792 hold_6_8
   - 265.78073089701 hold_6_9 - 239.2821535393818 hold_6_10
   - 217.5883952855847 hold_6_11 - 199.501246882793 hold_6_12
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_9_nic_in_7: switch_nic_out_9_nic_in_7 = 1 ->
   - 2364.532019704433 hold_7_1 - 1191.06699751861 hold_7_2
   - 796.0199004975125 hold_7_3 - 597.7584059775841 hold_7_4
   - 478.5643070787636 hold_7_5 - 399.002493765586 hold_7_6
   - 342.1240199572345 hold_7_7 - 299.4385527136619 hold_7_8
   - 266.2229617304492 hold_7_9 - 239.6405391912132 hold_7_10
   - 217.8847026781661 hold_7_11 - 199.7503121098626 hold_7_12
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_9_nic_in_8: switch_nic_out_9_nic_in_8 = 1 ->
   - 2364.532019704433 hold_8_1 - 1191.06699751861 hold_8_2
   - 796.0199004975125 hold_8_3 - 597.7584059775841 hold_8_4
   - 478.5643070787636 hold_8_5 - 399.002493765586 hold_8_6
   - 342.1240199572345 hold_8_7 - 299.4385527136619 hold_8_8
   - 266.2229617304492 hold_8_9 - 239.6405391912132 hold_8_10
   - 217.8847026781661 hold_8_11 - 199.7503121098626 hold_8_12
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_10_nic_out_9_nic_in_10: switch_nic_out_9_nic_in_10 = 1 ->
   - 2519.685039370078 hold_10_1 - 1269.84126984127 hold_10_2
   - 848.8063660477454 hold_10_3 - 637.4501992031873 hold_10_4
   - 510.3668261562998 hold_10_5 - 425.531914893617 hold_10_6
   - 364.8802736602053 hold_10_7 - 319.3612774451097 hold_10_8
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_9_nic_in_11: switch_nic_out_9_nic_in_11 = 1 ->
   - 2480.62015503876 hold_11_1 - 1259.842519685039 hold_11_2
   - 844.3271767810028 hold_11_3 - 634.9206349206349 hold_11_4
   - 508.7440381558029 hold_11_5 - 424.4031830238727 hold_11_6
   - 364.0500568828214 hold_11_7 - 318.7250996015936 hold_11_8
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_9_nic_in_12: switch_nic_out_9_nic_in_12 = 1 ->
   - 10765.90220683687 hold_12_1 - 5771.282765019717 hold_12_2
   - 3942.322928220568 hold_12_3 - 2993.622909397185 hold_12_4
   - 2412.957036174959 hold_12_5 - 2020.956867841768 hold_12_6
   - 1738.522814618126 hold_12_7 - 1525.350990129361 hold_12_8
   - 1358.746108896292 hold_12_9 - 1224.951996455123 hold_12_10
   - 1115.14499574201 hold_12_11 - 1023.405042984657 hold_12_12
   - 945.6121014024553 hold_12_13 - 878.810356398573 hold_12_14
   - 820.8241232555838 hold_12_15 - 770.0164030825416 hold_12_16
   - 725.1318819037627 hold_12_17 - 685.1918151524333 hold_12_18
   - 649.4218370702931 hold_12_19 - 617.20126020193 hold_12_20
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_9_nic_in_13: switch_nic_out_9_nic_in_13 = 1 ->
   - 2480.62015503876 hold_13_1 - 1259.842519685039 hold_13_2
   - 844.3271767810028 hold_13_3 - 634.9206349206349 hold_13_4
   - 508.7440381558029 hold_13_5 - 424.4031830238727 hold_13_6
   - 364.0500568828214 hold_13_7 - 318.7250996015936 hold_13_8
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_9_nic_in_14: switch_nic_out_9_nic_in_14 = 1 ->
   - 2330.097087378641 hold_14_1 - 1182.266009852217 hold_14_2
   - 792.0792079207921 hold_14_3 - 595.5334987593052 hold_14_4
   - 477.1371769383698 hold_14_5 - 398.0099502487562 hold_14_6
   - 341.3940256045519 hold_14_7 - 298.879202988792 hold_14_8
   - 265.78073089701 hold_14_9 - 239.2821535393818 hold_14_10
   - 217.5883952855847 hold_14_11 - 199.501246882793 hold_14_12
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_10_source_nic_in_10: switch_source_nic_in_10 = 1 ->
   - 2480.62015503876 hold_10_1 - 1259.842519685039 hold_10_2
   - 844.3271767810028 hold_10_3 - 634.9206349206349 hold_10_4
   - 508.7440381558029 hold_10_5 - 424.4031830238727 hold_10_6
   - 364.0500568828214 hold_10_7 - 318.7250996015936 hold_10_8
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_0_nic_out_10_nic_in_0: switch_nic_out_10_nic_in_0 = 1 ->
   - 2480.62015503876 hold_0_1 - 1259.842519685039 hold_0_2
   - 844.3271767810028 hold_0_3 - 634.9206349206349 hold_0_4
   - 508.7440381558029 hold_0_5 - 424.4031830238727 hold_0_6
   - 364.0500568828214 hold_0_7 - 318.7250996015936 hold_0_8
   + flow_nic_in_0_0 + flow_1_0 + flow_2_0 + flow_3_0 <= 0
 constr_intergroup_throughput_4_nic_out_10_nic_in_4: switch_nic_out_10_nic_in_4 = 1 ->
   - 10765.90220683687 hold_4_1 - 5771.282765019717 hold_4_2
   - 3942.322928220568 hold_4_3 - 2993.622909397185 hold_4_4
   - 2412.957036174959 hold_4_5 - 2020.956867841768 hold_4_6
   - 1738.522814618126 hold_4_7 - 1525.350990129361 hold_4_8
   - 1358.746108896292 hold_4_9 - 1224.951996455123 hold_4_10
   - 1115.14499574201 hold_4_11 - 1023.405042984657 hold_4_12
   - 945.6121014024553 hold_4_13 - 878.810356398573 hold_4_14
   - 820.8241232555838 hold_4_15 - 770.0164030825416 hold_4_16
   - 725.1318819037627 hold_4_17 - 685.1918151524333 hold_4_18
   - 649.4218370702931 hold_4_19 - 617.20126020193 hold_4_20
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_10_nic_in_5: switch_nic_out_10_nic_in_5 = 1 ->
   - 2330.097087378641 hold_5_1 - 1182.266009852217 hold_5_2
   - 792.0792079207921 hold_5_3 - 595.5334987593052 hold_5_4
   - 477.1371769383698 hold_5_5 - 398.0099502487562 hold_5_6
   - 341.3940256045519 hold_5_7 - 298.879202988792 hold_5_8
   - 265.78073089701 hold_5_9 - 239.2821535393818 hold_5_10
   - 217.5883952855847 hold_5_11 - 199.501246882793 hold_5_12
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_10_nic_in_6: switch_nic_out_10_nic_in_6 = 1 ->
   - 2330.097087378641 hold_6_1 - 1182.266009852217 hold_6_2
   - 792.0792079207921 hold_6_3 - 595.5334987593052 hold_6_4
   - 477.1371769383698 hold_6_5 - 398.0099502487562 hold_6_6
   - 341.3940256045519 hold_6_7 - 298.879202988792 hold_6_8
   - 265.78073089701 hold_6_9 - 239.2821535393818 hold_6_10
   - 217.5883952855847 hold_6_11 - 199.501246882793 hold_6_12
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_10_nic_in_7: switch_nic_out_10_nic_in_7 = 1 ->
   - 2364.532019704433 hold_7_1 - 1191.06699751861 hold_7_2
   - 796.0199004975125 hold_7_3 - 597.7584059775841 hold_7_4
   - 478.5643070787636 hold_7_5 - 399.002493765586 hold_7_6
   - 342.1240199572345 hold_7_7 - 299.4385527136619 hold_7_8
   - 266.2229617304492 hold_7_9 - 239.6405391912132 hold_7_10
   - 217.8847026781661 hold_7_11 - 199.7503121098626 hold_7_12
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_10_nic_in_8: switch_nic_out_10_nic_in_8 = 1 ->
   - 2364.532019704433 hold_8_1 - 1191.06699751861 hold_8_2
   - 796.0199004975125 hold_8_3 - 597.7584059775841 hold_8_4
   - 478.5643070787636 hold_8_5 - 399.002493765586 hold_8_6
   - 342.1240199572345 hold_8_7 - 299.4385527136619 hold_8_8
   - 266.2229617304492 hold_8_9 - 239.6405391912132 hold_8_10
   - 217.8847026781661 hold_8_11 - 199.7503121098626 hold_8_12
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_10_nic_in_9: switch_nic_out_10_nic_in_9 = 1 ->
   - 2519.685039370078 hold_9_1 - 1269.84126984127 hold_9_2
   - 848.8063660477454 hold_9_3 - 637.4501992031873 hold_9_4
   - 510.3668261562998 hold_9_5 - 425.531914893617 hold_9_6
   - 364.8802736602053 hold_9_7 - 319.3612774451097 hold_9_8
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_11_nic_out_10_nic_in_11: switch_nic_out_10_nic_in_11 = 1 ->
   - 2480.62015503876 hold_11_1 - 1259.842519685039 hold_11_2
   - 844.3271767810028 hold_11_3 - 634.9206349206349 hold_11_4
   - 508.7440381558029 hold_11_5 - 424.4031830238727 hold_11_6
   - 364.0500568828214 hold_11_7 - 318.7250996015936 hold_11_8
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_10_nic_in_12: switch_nic_out_10_nic_in_12 = 1 ->
   - 10765.90220683687 hold_12_1 - 5771.282765019717 hold_12_2
   - 3942.322928220568 hold_12_3 - 2993.622909397185 hold_12_4
   - 2412.957036174959 hold_12_5 - 2020.956867841768 hold_12_6
   - 1738.522814618126 hold_12_7 - 1525.350990129361 hold_12_8
   - 1358.746108896292 hold_12_9 - 1224.951996455123 hold_12_10
   - 1115.14499574201 hold_12_11 - 1023.405042984657 hold_12_12
   - 945.6121014024553 hold_12_13 - 878.810356398573 hold_12_14
   - 820.8241232555838 hold_12_15 - 770.0164030825416 hold_12_16
   - 725.1318819037627 hold_12_17 - 685.1918151524333 hold_12_18
   - 649.4218370702931 hold_12_19 - 617.20126020193 hold_12_20
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_10_nic_in_13: switch_nic_out_10_nic_in_13 = 1 ->
   - 2480.62015503876 hold_13_1 - 1259.842519685039 hold_13_2
   - 844.3271767810028 hold_13_3 - 634.9206349206349 hold_13_4
   - 508.7440381558029 hold_13_5 - 424.4031830238727 hold_13_6
   - 364.0500568828214 hold_13_7 - 318.7250996015936 hold_13_8
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_10_nic_in_14: switch_nic_out_10_nic_in_14 = 1 ->
   - 2330.097087378641 hold_14_1 - 1182.266009852217 hold_14_2
   - 792.0792079207921 hold_14_3 - 595.5334987593052 hold_14_4
   - 477.1371769383698 hold_14_5 - 398.0099502487562 hold_14_6
   - 341.3940256045519 hold_14_7 - 298.879202988792 hold_14_8
   - 265.78073089701 hold_14_9 - 239.2821535393818 hold_14_10
   - 217.5883952855847 hold_14_11 - 199.501246882793 hold_14_12
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_11_source_nic_in_11: switch_source_nic_in_11 = 1 ->
   - 2480.62015503876 hold_11_1 - 1259.842519685039 hold_11_2
   - 844.3271767810028 hold_11_3 - 634.9206349206349 hold_11_4
   - 508.7440381558029 hold_11_5 - 424.4031830238727 hold_11_6
   - 364.0500568828214 hold_11_7 - 318.7250996015936 hold_11_8
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_0_nic_out_11_nic_in_0: switch_nic_out_11_nic_in_0 = 1 ->
   - 2480.62015503876 hold_0_1 - 1259.842519685039 hold_0_2
   - 844.3271767810028 hold_0_3 - 634.9206349206349 hold_0_4
   - 508.7440381558029 hold_0_5 - 424.4031830238727 hold_0_6
   - 364.0500568828214 hold_0_7 - 318.7250996015936 hold_0_8
   + flow_nic_in_0_0 + flow_1_0 + flow_2_0 + flow_3_0 <= 0
 constr_intergroup_throughput_4_nic_out_11_nic_in_4: switch_nic_out_11_nic_in_4 = 1 ->
   - 10765.90220683687 hold_4_1 - 5771.282765019717 hold_4_2
   - 3942.322928220568 hold_4_3 - 2993.622909397185 hold_4_4
   - 2412.957036174959 hold_4_5 - 2020.956867841768 hold_4_6
   - 1738.522814618126 hold_4_7 - 1525.350990129361 hold_4_8
   - 1358.746108896292 hold_4_9 - 1224.951996455123 hold_4_10
   - 1115.14499574201 hold_4_11 - 1023.405042984657 hold_4_12
   - 945.6121014024553 hold_4_13 - 878.810356398573 hold_4_14
   - 820.8241232555838 hold_4_15 - 770.0164030825416 hold_4_16
   - 725.1318819037627 hold_4_17 - 685.1918151524333 hold_4_18
   - 649.4218370702931 hold_4_19 - 617.20126020193 hold_4_20
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_11_nic_in_5: switch_nic_out_11_nic_in_5 = 1 ->
   - 2330.097087378641 hold_5_1 - 1182.266009852217 hold_5_2
   - 792.0792079207921 hold_5_3 - 595.5334987593052 hold_5_4
   - 477.1371769383698 hold_5_5 - 398.0099502487562 hold_5_6
   - 341.3940256045519 hold_5_7 - 298.879202988792 hold_5_8
   - 265.78073089701 hold_5_9 - 239.2821535393818 hold_5_10
   - 217.5883952855847 hold_5_11 - 199.501246882793 hold_5_12
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_11_nic_in_6: switch_nic_out_11_nic_in_6 = 1 ->
   - 2330.097087378641 hold_6_1 - 1182.266009852217 hold_6_2
   - 792.0792079207921 hold_6_3 - 595.5334987593052 hold_6_4
   - 477.1371769383698 hold_6_5 - 398.0099502487562 hold_6_6
   - 341.3940256045519 hold_6_7 - 298.879202988792 hold_6_8
   - 265.78073089701 hold_6_9 - 239.2821535393818 hold_6_10
   - 217.5883952855847 hold_6_11 - 199.501246882793 hold_6_12
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_11_nic_in_7: switch_nic_out_11_nic_in_7 = 1 ->
   - 2330.097087378641 hold_7_1 - 1182.266009852217 hold_7_2
   - 792.0792079207921 hold_7_3 - 595.5334987593052 hold_7_4
   - 477.1371769383698 hold_7_5 - 398.0099502487562 hold_7_6
   - 341.3940256045519 hold_7_7 - 298.879202988792 hold_7_8
   - 265.78073089701 hold_7_9 - 239.2821535393818 hold_7_10
   - 217.5883952855847 hold_7_11 - 199.501246882793 hold_7_12
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_11_nic_in_8: switch_nic_out_11_nic_in_8 = 1 ->
   - 2330.097087378641 hold_8_1 - 1182.266009852217 hold_8_2
   - 792.0792079207921 hold_8_3 - 595.5334987593052 hold_8_4
   - 477.1371769383698 hold_8_5 - 398.0099502487562 hold_8_6
   - 341.3940256045519 hold_8_7 - 298.879202988792 hold_8_8
   - 265.78073089701 hold_8_9 - 239.2821535393818 hold_8_10
   - 217.5883952855847 hold_8_11 - 199.501246882793 hold_8_12
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_11_nic_in_9: switch_nic_out_11_nic_in_9 = 1 ->
   - 2480.62015503876 hold_9_1 - 1259.842519685039 hold_9_2
   - 844.3271767810028 hold_9_3 - 634.9206349206349 hold_9_4
   - 508.7440381558029 hold_9_5 - 424.4031830238727 hold_9_6
   - 364.0500568828214 hold_9_7 - 318.7250996015936 hold_9_8
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_11_nic_in_10: switch_nic_out_11_nic_in_10 = 1 ->
   - 2480.62015503876 hold_10_1 - 1259.842519685039 hold_10_2
   - 844.3271767810028 hold_10_3 - 634.9206349206349 hold_10_4
   - 508.7440381558029 hold_10_5 - 424.4031830238727 hold_10_6
   - 364.0500568828214 hold_10_7 - 318.7250996015936 hold_10_8
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_12_nic_out_11_nic_in_12: switch_nic_out_11_nic_in_12 = 1 ->
   - 11542.56553003943 hold_12_1 - 5987.245818794369 hold_12_2
   - 4041.913735683535 hold_12_3 - 3050.701980258721 hold_12_4
   - 2449.903992910246 hold_12_5 - 2046.810085969314 hold_12_6
   - 1757.620712797146 hold_12_7 - 1540.032806165083 hold_12_8
   - 1370.383630304867 hold_12_9 - 1234.40252040386 hold_12_10
   - 1122.971722597098 hold_12_11 - 1029.993169257519 hold_12_12
   - 951.2339660874386 hold_12_13 - 883.6639377741473 hold_12_14
   - 825.0567889771352 hold_12_15 - 773.7401066691546 hold_12_16
   - 728.4331952394198 hold_12_17 - 688.1387340791858 hold_12_18
   - 652.0685091271246 hold_12_19 - 619.5913386709168 hold_12_20
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_11_nic_in_13: switch_nic_out_11_nic_in_13 = 1 ->
   - 2519.685039370078 hold_13_1 - 1269.84126984127 hold_13_2
   - 848.8063660477454 hold_13_3 - 637.4501992031873 hold_13_4
   - 510.3668261562998 hold_13_5 - 425.531914893617 hold_13_6
   - 364.8802736602053 hold_13_7 - 319.3612774451097 hold_13_8
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_11_nic_in_14: switch_nic_out_11_nic_in_14 = 1 ->
   - 2364.532019704433 hold_14_1 - 1191.06699751861 hold_14_2
   - 796.0199004975125 hold_14_3 - 597.7584059775841 hold_14_4
   - 478.5643070787636 hold_14_5 - 399.002493765586 hold_14_6
   - 342.1240199572345 hold_14_7 - 299.4385527136619 hold_14_8
   - 266.2229617304492 hold_14_9 - 239.6405391912132 hold_14_10
   - 217.8847026781661 hold_14_11 - 199.7503121098626 hold_14_12
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_12_source_nic_in_12: switch_source_nic_in_12 = 1 ->
   - 10765.90220683687 hold_12_1 - 5771.282765019717 hold_12_2
   - 3942.322928220568 hold_12_3 - 2993.622909397185 hold_12_4
   - 2412.957036174959 hold_12_5 - 2020.956867841768 hold_12_6
   - 1738.522814618126 hold_12_7 - 1525.350990129361 hold_12_8
   - 1358.746108896292 hold_12_9 - 1224.951996455123 hold_12_10
   - 1115.14499574201 hold_12_11 - 1023.405042984657 hold_12_12
   - 945.6121014024553 hold_12_13 - 878.810356398573 hold_12_14
   - 820.8241232555838 hold_12_15 - 770.0164030825416 hold_12_16
   - 725.1318819037627 hold_12_17 - 685.1918151524333 hold_12_18
   - 649.4218370702931 hold_12_19 - 617.20126020193 hold_12_20
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_0_nic_out_12_nic_in_0: switch_nic_out_12_nic_in_0 = 1 ->
   - 2480.62015503876 hold_0_1 - 1259.842519685039 hold_0_2
   - 844.3271767810028 hold_0_3 - 634.9206349206349 hold_0_4
   - 508.7440381558029 hold_0_5 - 424.4031830238727 hold_0_6
   - 364.0500568828214 hold_0_7 - 318.7250996015936 hold_0_8
   + flow_nic_in_0_0 + flow_1_0 + flow_2_0 + flow_3_0 <= 0
 constr_intergroup_throughput_4_nic_out_12_nic_in_4: switch_nic_out_12_nic_in_4 = 1 ->
   - 10765.90220683687 hold_4_1 - 5771.282765019717 hold_4_2
   - 3942.322928220568 hold_4_3 - 2993.622909397185 hold_4_4
   - 2412.957036174959 hold_4_5 - 2020.956867841768 hold_4_6
   - 1738.522814618126 hold_4_7 - 1525.350990129361 hold_4_8
   - 1358.746108896292 hold_4_9 - 1224.951996455123 hold_4_10
   - 1115.14499574201 hold_4_11 - 1023.405042984657 hold_4_12
   - 945.6121014024553 hold_4_13 - 878.810356398573 hold_4_14
   - 820.8241232555838 hold_4_15 - 770.0164030825416 hold_4_16
   - 725.1318819037627 hold_4_17 - 685.1918151524333 hold_4_18
   - 649.4218370702931 hold_4_19 - 617.20126020193 hold_4_20
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_12_nic_in_5: switch_nic_out_12_nic_in_5 = 1 ->
   - 2330.097087378641 hold_5_1 - 1182.266009852217 hold_5_2
   - 792.0792079207921 hold_5_3 - 595.5334987593052 hold_5_4
   - 477.1371769383698 hold_5_5 - 398.0099502487562 hold_5_6
   - 341.3940256045519 hold_5_7 - 298.879202988792 hold_5_8
   - 265.78073089701 hold_5_9 - 239.2821535393818 hold_5_10
   - 217.5883952855847 hold_5_11 - 199.501246882793 hold_5_12
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_12_nic_in_6: switch_nic_out_12_nic_in_6 = 1 ->
   - 2330.097087378641 hold_6_1 - 1182.266009852217 hold_6_2
   - 792.0792079207921 hold_6_3 - 595.5334987593052 hold_6_4
   - 477.1371769383698 hold_6_5 - 398.0099502487562 hold_6_6
   - 341.3940256045519 hold_6_7 - 298.879202988792 hold_6_8
   - 265.78073089701 hold_6_9 - 239.2821535393818 hold_6_10
   - 217.5883952855847 hold_6_11 - 199.501246882793 hold_6_12
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_12_nic_in_7: switch_nic_out_12_nic_in_7 = 1 ->
   - 2330.097087378641 hold_7_1 - 1182.266009852217 hold_7_2
   - 792.0792079207921 hold_7_3 - 595.5334987593052 hold_7_4
   - 477.1371769383698 hold_7_5 - 398.0099502487562 hold_7_6
   - 341.3940256045519 hold_7_7 - 298.879202988792 hold_7_8
   - 265.78073089701 hold_7_9 - 239.2821535393818 hold_7_10
   - 217.5883952855847 hold_7_11 - 199.501246882793 hold_7_12
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_12_nic_in_8: switch_nic_out_12_nic_in_8 = 1 ->
   - 2330.097087378641 hold_8_1 - 1182.266009852217 hold_8_2
   - 792.0792079207921 hold_8_3 - 595.5334987593052 hold_8_4
   - 477.1371769383698 hold_8_5 - 398.0099502487562 hold_8_6
   - 341.3940256045519 hold_8_7 - 298.879202988792 hold_8_8
   - 265.78073089701 hold_8_9 - 239.2821535393818 hold_8_10
   - 217.5883952855847 hold_8_11 - 199.501246882793 hold_8_12
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_12_nic_in_9: switch_nic_out_12_nic_in_9 = 1 ->
   - 2480.62015503876 hold_9_1 - 1259.842519685039 hold_9_2
   - 844.3271767810028 hold_9_3 - 634.9206349206349 hold_9_4
   - 508.7440381558029 hold_9_5 - 424.4031830238727 hold_9_6
   - 364.0500568828214 hold_9_7 - 318.7250996015936 hold_9_8
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_12_nic_in_10: switch_nic_out_12_nic_in_10 = 1 ->
   - 2480.62015503876 hold_10_1 - 1259.842519685039 hold_10_2
   - 844.3271767810028 hold_10_3 - 634.9206349206349 hold_10_4
   - 508.7440381558029 hold_10_5 - 424.4031830238727 hold_10_6
   - 364.0500568828214 hold_10_7 - 318.7250996015936 hold_10_8
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_12_nic_in_11: switch_nic_out_12_nic_in_11 = 1 ->
   - 2519.685039370078 hold_11_1 - 1269.84126984127 hold_11_2
   - 848.8063660477454 hold_11_3 - 637.4501992031873 hold_11_4
   - 510.3668261562998 hold_11_5 - 425.531914893617 hold_11_6
   - 364.8802736602053 hold_11_7 - 319.3612774451097 hold_11_8
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_13_nic_out_12_nic_in_13: switch_nic_out_12_nic_in_13 = 1 ->
   - 2519.685039370078 hold_13_1 - 1269.84126984127 hold_13_2
   - 848.8063660477454 hold_13_3 - 637.4501992031873 hold_13_4
   - 510.3668261562998 hold_13_5 - 425.531914893617 hold_13_6
   - 364.8802736602053 hold_13_7 - 319.3612774451097 hold_13_8
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_12_nic_in_14: switch_nic_out_12_nic_in_14 = 1 ->
   - 2364.532019704433 hold_14_1 - 1191.06699751861 hold_14_2
   - 796.0199004975125 hold_14_3 - 597.7584059775841 hold_14_4
   - 478.5643070787636 hold_14_5 - 399.002493765586 hold_14_6
   - 342.1240199572345 hold_14_7 - 299.4385527136619 hold_14_8
   - 266.2229617304492 hold_14_9 - 239.6405391912132 hold_14_10
   - 217.8847026781661 hold_14_11 - 199.7503121098626 hold_14_12
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_13_source_nic_in_13: switch_source_nic_in_13 = 1 ->
   - 2480.62015503876 hold_13_1 - 1259.842519685039 hold_13_2
   - 844.3271767810028 hold_13_3 - 634.9206349206349 hold_13_4
   - 508.7440381558029 hold_13_5 - 424.4031830238727 hold_13_6
   - 364.0500568828214 hold_13_7 - 318.7250996015936 hold_13_8
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_0_nic_out_13_nic_in_0: switch_nic_out_13_nic_in_0 = 1 ->
   - 2480.62015503876 hold_0_1 - 1259.842519685039 hold_0_2
   - 844.3271767810028 hold_0_3 - 634.9206349206349 hold_0_4
   - 508.7440381558029 hold_0_5 - 424.4031830238727 hold_0_6
   - 364.0500568828214 hold_0_7 - 318.7250996015936 hold_0_8
   + flow_nic_in_0_0 + flow_1_0 + flow_2_0 + flow_3_0 <= 0
 constr_intergroup_throughput_4_nic_out_13_nic_in_4: switch_nic_out_13_nic_in_4 = 1 ->
   - 10765.90220683687 hold_4_1 - 5771.282765019717 hold_4_2
   - 3942.322928220568 hold_4_3 - 2993.622909397185 hold_4_4
   - 2412.957036174959 hold_4_5 - 2020.956867841768 hold_4_6
   - 1738.522814618126 hold_4_7 - 1525.350990129361 hold_4_8
   - 1358.746108896292 hold_4_9 - 1224.951996455123 hold_4_10
   - 1115.14499574201 hold_4_11 - 1023.405042984657 hold_4_12
   - 945.6121014024553 hold_4_13 - 878.810356398573 hold_4_14
   - 820.8241232555838 hold_4_15 - 770.0164030825416 hold_4_16
   - 725.1318819037627 hold_4_17 - 685.1918151524333 hold_4_18
   - 649.4218370702931 hold_4_19 - 617.20126020193 hold_4_20
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_13_nic_in_5: switch_nic_out_13_nic_in_5 = 1 ->
   - 2330.097087378641 hold_5_1 - 1182.266009852217 hold_5_2
   - 792.0792079207921 hold_5_3 - 595.5334987593052 hold_5_4
   - 477.1371769383698 hold_5_5 - 398.0099502487562 hold_5_6
   - 341.3940256045519 hold_5_7 - 298.879202988792 hold_5_8
   - 265.78073089701 hold_5_9 - 239.2821535393818 hold_5_10
   - 217.5883952855847 hold_5_11 - 199.501246882793 hold_5_12
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_13_nic_in_6: switch_nic_out_13_nic_in_6 = 1 ->
   - 2330.097087378641 hold_6_1 - 1182.266009852217 hold_6_2
   - 792.0792079207921 hold_6_3 - 595.5334987593052 hold_6_4
   - 477.1371769383698 hold_6_5 - 398.0099502487562 hold_6_6
   - 341.3940256045519 hold_6_7 - 298.879202988792 hold_6_8
   - 265.78073089701 hold_6_9 - 239.2821535393818 hold_6_10
   - 217.5883952855847 hold_6_11 - 199.501246882793 hold_6_12
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_13_nic_in_7: switch_nic_out_13_nic_in_7 = 1 ->
   - 2330.097087378641 hold_7_1 - 1182.266009852217 hold_7_2
   - 792.0792079207921 hold_7_3 - 595.5334987593052 hold_7_4
   - 477.1371769383698 hold_7_5 - 398.0099502487562 hold_7_6
   - 341.3940256045519 hold_7_7 - 298.879202988792 hold_7_8
   - 265.78073089701 hold_7_9 - 239.2821535393818 hold_7_10
   - 217.5883952855847 hold_7_11 - 199.501246882793 hold_7_12
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_13_nic_in_8: switch_nic_out_13_nic_in_8 = 1 ->
   - 2330.097087378641 hold_8_1 - 1182.266009852217 hold_8_2
   - 792.0792079207921 hold_8_3 - 595.5334987593052 hold_8_4
   - 477.1371769383698 hold_8_5 - 398.0099502487562 hold_8_6
   - 341.3940256045519 hold_8_7 - 298.879202988792 hold_8_8
   - 265.78073089701 hold_8_9 - 239.2821535393818 hold_8_10
   - 217.5883952855847 hold_8_11 - 199.501246882793 hold_8_12
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_13_nic_in_9: switch_nic_out_13_nic_in_9 = 1 ->
   - 2480.62015503876 hold_9_1 - 1259.842519685039 hold_9_2
   - 844.3271767810028 hold_9_3 - 634.9206349206349 hold_9_4
   - 508.7440381558029 hold_9_5 - 424.4031830238727 hold_9_6
   - 364.0500568828214 hold_9_7 - 318.7250996015936 hold_9_8
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_13_nic_in_10: switch_nic_out_13_nic_in_10 = 1 ->
   - 2480.62015503876 hold_10_1 - 1259.842519685039 hold_10_2
   - 844.3271767810028 hold_10_3 - 634.9206349206349 hold_10_4
   - 508.7440381558029 hold_10_5 - 424.4031830238727 hold_10_6
   - 364.0500568828214 hold_10_7 - 318.7250996015936 hold_10_8
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_13_nic_in_11: switch_nic_out_13_nic_in_11 = 1 ->
   - 2519.685039370078 hold_11_1 - 1269.84126984127 hold_11_2
   - 848.8063660477454 hold_11_3 - 637.4501992031873 hold_11_4
   - 510.3668261562998 hold_11_5 - 425.531914893617 hold_11_6
   - 364.8802736602053 hold_11_7 - 319.3612774451097 hold_11_8
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_13_nic_in_12: switch_nic_out_13_nic_in_12 = 1 ->
   - 11542.56553003943 hold_12_1 - 5987.245818794369 hold_12_2
   - 4041.913735683535 hold_12_3 - 3050.701980258721 hold_12_4
   - 2449.903992910246 hold_12_5 - 2046.810085969314 hold_12_6
   - 1757.620712797146 hold_12_7 - 1540.032806165083 hold_12_8
   - 1370.383630304867 hold_12_9 - 1234.40252040386 hold_12_10
   - 1122.971722597098 hold_12_11 - 1029.993169257519 hold_12_12
   - 951.2339660874386 hold_12_13 - 883.6639377741473 hold_12_14
   - 825.0567889771352 hold_12_15 - 773.7401066691546 hold_12_16
   - 728.4331952394198 hold_12_17 - 688.1387340791858 hold_12_18
   - 652.0685091271246 hold_12_19 - 619.5913386709168 hold_12_20
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_14_nic_out_13_nic_in_14: switch_nic_out_13_nic_in_14 = 1 ->
   - 2364.532019704433 hold_14_1 - 1191.06699751861 hold_14_2
   - 796.0199004975125 hold_14_3 - 597.7584059775841 hold_14_4
   - 478.5643070787636 hold_14_5 - 399.002493765586 hold_14_6
   - 342.1240199572345 hold_14_7 - 299.4385527136619 hold_14_8
   - 266.2229617304492 hold_14_9 - 239.6405391912132 hold_14_10
   - 217.8847026781661 hold_14_11 - 199.7503121098626 hold_14_12
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_14_source_nic_in_14: switch_source_nic_in_14 = 1 ->
   - 2330.097087378641 hold_14_1 - 1182.266009852217 hold_14_2
   - 792.0792079207921 hold_14_3 - 595.5334987593052 hold_14_4
   - 477.1371769383698 hold_14_5 - 398.0099502487562 hold_14_6
   - 341.3940256045519 hold_14_7 - 298.879202988792 hold_14_8
   - 265.78073089701 hold_14_9 - 239.2821535393818 hold_14_10
   - 217.5883952855847 hold_14_11 - 199.501246882793 hold_14_12
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_0_nic_out_14_nic_in_0: switch_nic_out_14_nic_in_0 = 1 ->
   - 2480.62015503876 hold_0_1 - 1259.842519685039 hold_0_2
   - 844.3271767810028 hold_0_3 - 634.9206349206349 hold_0_4
   - 508.7440381558029 hold_0_5 - 424.4031830238727 hold_0_6
   - 364.0500568828214 hold_0_7 - 318.7250996015936 hold_0_8
   + flow_nic_in_0_0 + flow_1_0 + flow_2_0 + flow_3_0 <= 0
 constr_intergroup_throughput_4_nic_out_14_nic_in_4: switch_nic_out_14_nic_in_4 = 1 ->
   - 10765.90220683687 hold_4_1 - 5771.282765019717 hold_4_2
   - 3942.322928220568 hold_4_3 - 2993.622909397185 hold_4_4
   - 2412.957036174959 hold_4_5 - 2020.956867841768 hold_4_6
   - 1738.522814618126 hold_4_7 - 1525.350990129361 hold_4_8
   - 1358.746108896292 hold_4_9 - 1224.951996455123 hold_4_10
   - 1115.14499574201 hold_4_11 - 1023.405042984657 hold_4_12
   - 945.6121014024553 hold_4_13 - 878.810356398573 hold_4_14
   - 820.8241232555838 hold_4_15 - 770.0164030825416 hold_4_16
   - 725.1318819037627 hold_4_17 - 685.1918151524333 hold_4_18
   - 649.4218370702931 hold_4_19 - 617.20126020193 hold_4_20
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_14_nic_in_5: switch_nic_out_14_nic_in_5 = 1 ->
   - 2330.097087378641 hold_5_1 - 1182.266009852217 hold_5_2
   - 792.0792079207921 hold_5_3 - 595.5334987593052 hold_5_4
   - 477.1371769383698 hold_5_5 - 398.0099502487562 hold_5_6
   - 341.3940256045519 hold_5_7 - 298.879202988792 hold_5_8
   - 265.78073089701 hold_5_9 - 239.2821535393818 hold_5_10
   - 217.5883952855847 hold_5_11 - 199.501246882793 hold_5_12
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_14_nic_in_6: switch_nic_out_14_nic_in_6 = 1 ->
   - 2330.097087378641 hold_6_1 - 1182.266009852217 hold_6_2
   - 792.0792079207921 hold_6_3 - 595.5334987593052 hold_6_4
   - 477.1371769383698 hold_6_5 - 398.0099502487562 hold_6_6
   - 341.3940256045519 hold_6_7 - 298.879202988792 hold_6_8
   - 265.78073089701 hold_6_9 - 239.2821535393818 hold_6_10
   - 217.5883952855847 hold_6_11 - 199.501246882793 hold_6_12
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_14_nic_in_7: switch_nic_out_14_nic_in_7 = 1 ->
   - 2330.097087378641 hold_7_1 - 1182.266009852217 hold_7_2
   - 792.0792079207921 hold_7_3 - 595.5334987593052 hold_7_4
   - 477.1371769383698 hold_7_5 - 398.0099502487562 hold_7_6
   - 341.3940256045519 hold_7_7 - 298.879202988792 hold_7_8
   - 265.78073089701 hold_7_9 - 239.2821535393818 hold_7_10
   - 217.5883952855847 hold_7_11 - 199.501246882793 hold_7_12
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_14_nic_in_8: switch_nic_out_14_nic_in_8 = 1 ->
   - 2330.097087378641 hold_8_1 - 1182.266009852217 hold_8_2
   - 792.0792079207921 hold_8_3 - 595.5334987593052 hold_8_4
   - 477.1371769383698 hold_8_5 - 398.0099502487562 hold_8_6
   - 341.3940256045519 hold_8_7 - 298.879202988792 hold_8_8
   - 265.78073089701 hold_8_9 - 239.2821535393818 hold_8_10
   - 217.5883952855847 hold_8_11 - 199.501246882793 hold_8_12
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_14_nic_in_9: switch_nic_out_14_nic_in_9 = 1 ->
   - 2480.62015503876 hold_9_1 - 1259.842519685039 hold_9_2
   - 844.3271767810028 hold_9_3 - 634.9206349206349 hold_9_4
   - 508.7440381558029 hold_9_5 - 424.4031830238727 hold_9_6
   - 364.0500568828214 hold_9_7 - 318.7250996015936 hold_9_8
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_14_nic_in_10: switch_nic_out_14_nic_in_10 = 1 ->
   - 2480.62015503876 hold_10_1 - 1259.842519685039 hold_10_2
   - 844.3271767810028 hold_10_3 - 634.9206349206349 hold_10_4
   - 508.7440381558029 hold_10_5 - 424.4031830238727 hold_10_6
   - 364.0500568828214 hold_10_7 - 318.7250996015936 hold_10_8
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_14_nic_in_11: switch_nic_out_14_nic_in_11 = 1 ->
   - 2519.685039370078 hold_11_1 - 1269.84126984127 hold_11_2
   - 848.8063660477454 hold_11_3 - 637.4501992031873 hold_11_4
   - 510.3668261562998 hold_11_5 - 425.531914893617 hold_11_6
   - 364.8802736602053 hold_11_7 - 319.3612774451097 hold_11_8
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_14_nic_in_12: switch_nic_out_14_nic_in_12 = 1 ->
   - 11542.56553003943 hold_12_1 - 5987.245818794369 hold_12_2
   - 4041.913735683535 hold_12_3 - 3050.701980258721 hold_12_4
   - 2449.903992910246 hold_12_5 - 2046.810085969314 hold_12_6
   - 1757.620712797146 hold_12_7 - 1540.032806165083 hold_12_8
   - 1370.383630304867 hold_12_9 - 1234.40252040386 hold_12_10
   - 1122.971722597098 hold_12_11 - 1029.993169257519 hold_12_12
   - 951.2339660874386 hold_12_13 - 883.6639377741473 hold_12_14
   - 825.0567889771352 hold_12_15 - 773.7401066691546 hold_12_16
   - 728.4331952394198 hold_12_17 - 688.1387340791858 hold_12_18
   - 652.0685091271246 hold_12_19 - 619.5913386709168 hold_12_20
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_14_nic_in_13: switch_nic_out_14_nic_in_13 = 1 ->
   - 2519.685039370078 hold_13_1 - 1269.84126984127 hold_13_2
   - 848.8063660477454 hold_13_3 - 637.4501992031873 hold_13_4
   - 510.3668261562998 hold_13_5 - 425.531914893617 hold_13_6
   - 364.8802736602053 hold_13_7 - 319.3612774451097 hold_13_8
   + flow_nic_in_13_13 <= 0
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
   - 25 hold_0_25 - 26 hold_0_26 - 27 hold_0_27 - 28 hold_0_28
   - 29 hold_0_29 - 30 hold_0_30 - 31 hold_0_31 - 32 hold_0_32 + val_end_0
   = 0
 inactive_end_zero_0: is_active_0 = 0 -> val_end_0 = 0
 active_end_val_1: is_active_1 = 1 -> - start_1 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11 - 12 hold_1_12
   - 13 hold_1_13 - 14 hold_1_14 - 15 hold_1_15 - 16 hold_1_16
   - 17 hold_1_17 - 18 hold_1_18 - 19 hold_1_19 - 20 hold_1_20
   - 21 hold_1_21 - 22 hold_1_22 - 23 hold_1_23 - 24 hold_1_24
   - 25 hold_1_25 - 26 hold_1_26 - 27 hold_1_27 - 28 hold_1_28
   - 29 hold_1_29 - 30 hold_1_30 - 31 hold_1_31 - 32 hold_1_32 + val_end_1
   = 0
 inactive_end_zero_1: is_active_1 = 0 -> val_end_1 = 0
 active_end_val_2: is_active_2 = 1 -> - start_2 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 - 12 hold_2_12
   - 13 hold_2_13 - 14 hold_2_14 - 15 hold_2_15 - 16 hold_2_16
   - 17 hold_2_17 - 18 hold_2_18 - 19 hold_2_19 - 20 hold_2_20
   - 21 hold_2_21 - 22 hold_2_22 - 23 hold_2_23 - 24 hold_2_24
   - 25 hold_2_25 - 26 hold_2_26 - 27 hold_2_27 - 28 hold_2_28
   - 29 hold_2_29 - 30 hold_2_30 - 31 hold_2_31 - 32 hold_2_32 + val_end_2
   = 0
 inactive_end_zero_2: is_active_2 = 0 -> val_end_2 = 0
 active_end_val_3: is_active_3 = 1 -> - start_3 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   - 8 hold_3_8 - 9 hold_3_9 - 10 hold_3_10 - 11 hold_3_11 - 12 hold_3_12
   - 13 hold_3_13 - 14 hold_3_14 - 15 hold_3_15 - 16 hold_3_16
   - 17 hold_3_17 - 18 hold_3_18 - 19 hold_3_19 - 20 hold_3_20
   - 21 hold_3_21 - 22 hold_3_22 - 23 hold_3_23 - 24 hold_3_24
   - 25 hold_3_25 - 26 hold_3_26 - 27 hold_3_27 - 28 hold_3_28
   - 29 hold_3_29 - 30 hold_3_30 - 31 hold_3_31 - 32 hold_3_32 + val_end_3
   = 0
 inactive_end_zero_3: is_active_3 = 0 -> val_end_3 = 0
 inactive_end_not_complete_4: is_active_4 = 0 -> is_end_complete_4 = 0
 active_end_val_4: is_active_4 = 1 -> - start_4 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   - 8 hold_4_8 - 9 hold_4_9 - 10 hold_4_10 - 11 hold_4_11 - 12 hold_4_12
   - 13 hold_4_13 - 14 hold_4_14 - 15 hold_4_15 - 16 hold_4_16
   - 17 hold_4_17 - 18 hold_4_18 - 19 hold_4_19 - 20 hold_4_20
   - 21 hold_4_21 - 22 hold_4_22 - 23 hold_4_23 - 24 hold_4_24
   - 25 hold_4_25 - 26 hold_4_26 - 27 hold_4_27 - 28 hold_4_28
   - 29 hold_4_29 - 30 hold_4_30 - 31 hold_4_31 - 32 hold_4_32 + val_end_4
   = 0
 inactive_end_zero_4: is_active_4 = 0 -> val_end_4 = 0
 inactive_end_not_complete_5: is_active_5 = 0 -> is_end_complete_5 = 0
 active_end_val_5: is_active_5 = 1 -> - start_5 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11 - 12 hold_5_12
   - 13 hold_5_13 - 14 hold_5_14 - 15 hold_5_15 - 16 hold_5_16
   - 17 hold_5_17 - 18 hold_5_18 - 19 hold_5_19 - 20 hold_5_20
   - 21 hold_5_21 - 22 hold_5_22 - 23 hold_5_23 - 24 hold_5_24
   - 25 hold_5_25 - 26 hold_5_26 - 27 hold_5_27 - 28 hold_5_28
   - 29 hold_5_29 - 30 hold_5_30 - 31 hold_5_31 - 32 hold_5_32 + val_end_5
   = 0
 inactive_end_zero_5: is_active_5 = 0 -> val_end_5 = 0
 inactive_end_not_complete_6: is_active_6 = 0 -> is_end_complete_6 = 0
 active_end_val_6: is_active_6 = 1 -> - start_6 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11 - 12 hold_6_12
   - 13 hold_6_13 - 14 hold_6_14 - 15 hold_6_15 - 16 hold_6_16
   - 17 hold_6_17 - 18 hold_6_18 - 19 hold_6_19 - 20 hold_6_20
   - 21 hold_6_21 - 22 hold_6_22 - 23 hold_6_23 - 24 hold_6_24
   - 25 hold_6_25 - 26 hold_6_26 - 27 hold_6_27 - 28 hold_6_28
   - 29 hold_6_29 - 30 hold_6_30 - 31 hold_6_31 - 32 hold_6_32 + val_end_6
   = 0
 inactive_end_zero_6: is_active_6 = 0 -> val_end_6 = 0
 inactive_end_not_complete_7: is_active_7 = 0 -> is_end_complete_7 = 0
 active_end_val_7: is_active_7 = 1 -> - start_7 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   - 8 hold_7_8 - 9 hold_7_9 - 10 hold_7_10 - 11 hold_7_11 - 12 hold_7_12
   - 13 hold_7_13 - 14 hold_7_14 - 15 hold_7_15 - 16 hold_7_16
   - 17 hold_7_17 - 18 hold_7_18 - 19 hold_7_19 - 20 hold_7_20
   - 21 hold_7_21 - 22 hold_7_22 - 23 hold_7_23 - 24 hold_7_24
   - 25 hold_7_25 - 26 hold_7_26 - 27 hold_7_27 - 28 hold_7_28
   - 29 hold_7_29 - 30 hold_7_30 - 31 hold_7_31 - 32 hold_7_32 + val_end_7
   = 0
 inactive_end_zero_7: is_active_7 = 0 -> val_end_7 = 0
 inactive_end_not_complete_8: is_active_8 = 0 -> is_end_complete_8 = 0
 active_end_val_8: is_active_8 = 1 -> - start_8 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11 - 12 hold_8_12
   - 13 hold_8_13 - 14 hold_8_14 - 15 hold_8_15 - 16 hold_8_16
   - 17 hold_8_17 - 18 hold_8_18 - 19 hold_8_19 - 20 hold_8_20
   - 21 hold_8_21 - 22 hold_8_22 - 23 hold_8_23 - 24 hold_8_24
   - 25 hold_8_25 - 26 hold_8_26 - 27 hold_8_27 - 28 hold_8_28
   - 29 hold_8_29 - 30 hold_8_30 - 31 hold_8_31 - 32 hold_8_32 + val_end_8
   = 0
 inactive_end_zero_8: is_active_8 = 0 -> val_end_8 = 0
 inactive_end_not_complete_9: is_active_9 = 0 -> is_end_complete_9 = 0
 active_end_val_9: is_active_9 = 1 -> - start_9 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   - 8 hold_9_8 - 9 hold_9_9 - 10 hold_9_10 - 11 hold_9_11 - 12 hold_9_12
   - 13 hold_9_13 - 14 hold_9_14 - 15 hold_9_15 - 16 hold_9_16
   - 17 hold_9_17 - 18 hold_9_18 - 19 hold_9_19 - 20 hold_9_20
   - 21 hold_9_21 - 22 hold_9_22 - 23 hold_9_23 - 24 hold_9_24
   - 25 hold_9_25 - 26 hold_9_26 - 27 hold_9_27 - 28 hold_9_28
   - 29 hold_9_29 - 30 hold_9_30 - 31 hold_9_31 - 32 hold_9_32 + val_end_9
   = 0
 inactive_end_zero_9: is_active_9 = 0 -> val_end_9 = 0
 inactive_end_not_complete_10: is_active_10 = 0 -> is_end_complete_10 = 0
 active_end_val_10: is_active_10 = 1 -> - start_10 - hold_10_1
   - 2 hold_10_2 - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6
   - 7 hold_10_7 - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10
   - 11 hold_10_11 - 12 hold_10_12 - 13 hold_10_13 - 14 hold_10_14
   - 15 hold_10_15 - 16 hold_10_16 - 17 hold_10_17 - 18 hold_10_18
   - 19 hold_10_19 - 20 hold_10_20 - 21 hold_10_21 - 22 hold_10_22
   - 23 hold_10_23 - 24 hold_10_24 - 25 hold_10_25 - 26 hold_10_26
   - 27 hold_10_27 - 28 hold_10_28 - 29 hold_10_29 - 30 hold_10_30
   - 31 hold_10_31 - 32 hold_10_32 + val_end_10 = 0
 inactive_end_zero_10: is_active_10 = 0 -> val_end_10 = 0
 inactive_end_not_complete_11: is_active_11 = 0 -> is_end_complete_11 = 0
 active_end_val_11: is_active_11 = 1 -> - start_11 - hold_11_1
   - 2 hold_11_2 - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6
   - 7 hold_11_7 - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10
   - 11 hold_11_11 - 12 hold_11_12 - 13 hold_11_13 - 14 hold_11_14
   - 15 hold_11_15 - 16 hold_11_16 - 17 hold_11_17 - 18 hold_11_18
   - 19 hold_11_19 - 20 hold_11_20 - 21 hold_11_21 - 22 hold_11_22
   - 23 hold_11_23 - 24 hold_11_24 - 25 hold_11_25 - 26 hold_11_26
   - 27 hold_11_27 - 28 hold_11_28 - 29 hold_11_29 - 30 hold_11_30
   - 31 hold_11_31 - 32 hold_11_32 + val_end_11 = 0
 inactive_end_zero_11: is_active_11 = 0 -> val_end_11 = 0
 inactive_end_not_complete_12: is_active_12 = 0 -> is_end_complete_12 = 0
 active_end_val_12: is_active_12 = 1 -> - start_12 - hold_12_1
   - 2 hold_12_2 - 3 hold_12_3 - 4 hold_12_4 - 5 hold_12_5 - 6 hold_12_6
   - 7 hold_12_7 - 8 hold_12_8 - 9 hold_12_9 - 10 hold_12_10
   - 11 hold_12_11 - 12 hold_12_12 - 13 hold_12_13 - 14 hold_12_14
   - 15 hold_12_15 - 16 hold_12_16 - 17 hold_12_17 - 18 hold_12_18
   - 19 hold_12_19 - 20 hold_12_20 - 21 hold_12_21 - 22 hold_12_22
   - 23 hold_12_23 - 24 hold_12_24 - 25 hold_12_25 - 26 hold_12_26
   - 27 hold_12_27 - 28 hold_12_28 - 29 hold_12_29 - 30 hold_12_30
   - 31 hold_12_31 - 32 hold_12_32 + val_end_12 = 0
 inactive_end_zero_12: is_active_12 = 0 -> val_end_12 = 0
 inactive_end_not_complete_13: is_active_13 = 0 -> is_end_complete_13 = 0
 active_end_val_13: is_active_13 = 1 -> - start_13 - hold_13_1
   - 2 hold_13_2 - 3 hold_13_3 - 4 hold_13_4 - 5 hold_13_5 - 6 hold_13_6
   - 7 hold_13_7 - 8 hold_13_8 - 9 hold_13_9 - 10 hold_13_10
   - 11 hold_13_11 - 12 hold_13_12 - 13 hold_13_13 - 14 hold_13_14
   - 15 hold_13_15 - 16 hold_13_16 - 17 hold_13_17 - 18 hold_13_18
   - 19 hold_13_19 - 20 hold_13_20 - 21 hold_13_21 - 22 hold_13_22
   - 23 hold_13_23 - 24 hold_13_24 - 25 hold_13_25 - 26 hold_13_26
   - 27 hold_13_27 - 28 hold_13_28 - 29 hold_13_29 - 30 hold_13_30
   - 31 hold_13_31 - 32 hold_13_32 + val_end_13 = 0
 inactive_end_zero_13: is_active_13 = 0 -> val_end_13 = 0
 inactive_end_not_complete_14: is_active_14 = 0 -> is_end_complete_14 = 0
 active_end_val_14: is_active_14 = 1 -> - start_14 - hold_14_1
   - 2 hold_14_2 - 3 hold_14_3 - 4 hold_14_4 - 5 hold_14_5 - 6 hold_14_6
   - 7 hold_14_7 - 8 hold_14_8 - 9 hold_14_9 - 10 hold_14_10
   - 11 hold_14_11 - 12 hold_14_12 - 13 hold_14_13 - 14 hold_14_14
   - 15 hold_14_15 - 16 hold_14_16 - 17 hold_14_17 - 18 hold_14_18
   - 19 hold_14_19 - 20 hold_14_20 - 21 hold_14_21 - 22 hold_14_22
   - 23 hold_14_23 - 24 hold_14_24 - 25 hold_14_25 - 26 hold_14_26
   - 27 hold_14_27 - 28 hold_14_28 - 29 hold_14_29 - 30 hold_14_30
   - 31 hold_14_31 - 32 hold_14_32 + val_end_14 = 0
 inactive_end_zero_14: is_active_14 = 0 -> val_end_14 = 0
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
 group_synch_hold_constr_0_1_27: group_tp_active_0 = 1 -> hold_0_27
   - hold_1_27 = 0
 group_synch_hold_constr_0_2_27: group_tp_active_0 = 1 -> hold_0_27
   - hold_2_27 = 0
 group_synch_hold_constr_0_3_27: group_tp_active_0 = 1 -> hold_0_27
   - hold_3_27 = 0
 group_synch_hold_constr_0_1_28: group_tp_active_0 = 1 -> hold_0_28
   - hold_1_28 = 0
 group_synch_hold_constr_0_2_28: group_tp_active_0 = 1 -> hold_0_28
   - hold_2_28 = 0
 group_synch_hold_constr_0_3_28: group_tp_active_0 = 1 -> hold_0_28
   - hold_3_28 = 0
 group_synch_hold_constr_0_1_29: group_tp_active_0 = 1 -> hold_0_29
   - hold_1_29 = 0
 group_synch_hold_constr_0_2_29: group_tp_active_0 = 1 -> hold_0_29
   - hold_2_29 = 0
 group_synch_hold_constr_0_3_29: group_tp_active_0 = 1 -> hold_0_29
   - hold_3_29 = 0
 group_synch_hold_constr_0_1_30: group_tp_active_0 = 1 -> hold_0_30
   - hold_1_30 = 0
 group_synch_hold_constr_0_2_30: group_tp_active_0 = 1 -> hold_0_30
   - hold_2_30 = 0
 group_synch_hold_constr_0_3_30: group_tp_active_0 = 1 -> hold_0_30
   - hold_3_30 = 0
 group_synch_hold_constr_0_1_31: group_tp_active_0 = 1 -> hold_0_31
   - hold_1_31 = 0
 group_synch_hold_constr_0_2_31: group_tp_active_0 = 1 -> hold_0_31
   - hold_2_31 = 0
 group_synch_hold_constr_0_3_31: group_tp_active_0 = 1 -> hold_0_31
   - hold_3_31 = 0
 group_synch_hold_constr_0_1_32: group_tp_active_0 = 1 -> hold_0_32
   - hold_1_32 = 0
 group_synch_hold_constr_0_2_32: group_tp_active_0 = 1 -> hold_0_32
   - hold_2_32 = 0
 group_synch_hold_constr_0_3_32: group_tp_active_0 = 1 -> hold_0_32
   - hold_3_32 = 0
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
 tp_off_limit_constr_0_27: group_tp_active_0 = 0 -> hold_0_27 = 0
 tp_off_limit_constr_0_28: group_tp_active_0 = 0 -> hold_0_28 = 0
 tp_off_limit_constr_0_29: group_tp_active_0 = 0 -> hold_0_29 = 0
 tp_off_limit_constr_0_30: group_tp_active_0 = 0 -> hold_0_30 = 0
 tp_off_limit_constr_0_31: group_tp_active_0 = 0 -> hold_0_31 = 0
 tp_off_limit_constr_0_32: group_tp_active_0 = 0 -> hold_0_32 = 0
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
 tp_off_limit_constr_1_27: group_tp_active_0 = 0 -> hold_1_27 = 0
 tp_off_limit_constr_1_28: group_tp_active_0 = 0 -> hold_1_28 = 0
 tp_off_limit_constr_1_29: group_tp_active_0 = 0 -> hold_1_29 = 0
 tp_off_limit_constr_1_30: group_tp_active_0 = 0 -> hold_1_30 = 0
 tp_off_limit_constr_1_31: group_tp_active_0 = 0 -> hold_1_31 = 0
 tp_off_limit_constr_1_32: group_tp_active_0 = 0 -> hold_1_32 = 0
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
 tp_off_limit_constr_2_27: group_tp_active_0 = 0 -> hold_2_27 = 0
 tp_off_limit_constr_2_28: group_tp_active_0 = 0 -> hold_2_28 = 0
 tp_off_limit_constr_2_29: group_tp_active_0 = 0 -> hold_2_29 = 0
 tp_off_limit_constr_2_30: group_tp_active_0 = 0 -> hold_2_30 = 0
 tp_off_limit_constr_2_31: group_tp_active_0 = 0 -> hold_2_31 = 0
 tp_off_limit_constr_2_32: group_tp_active_0 = 0 -> hold_2_32 = 0
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
 tp_off_limit_constr_3_27: group_tp_active_0 = 0 -> hold_3_27 = 0
 tp_off_limit_constr_3_28: group_tp_active_0 = 0 -> hold_3_28 = 0
 tp_off_limit_constr_3_29: group_tp_active_0 = 0 -> hold_3_29 = 0
 tp_off_limit_constr_3_30: group_tp_active_0 = 0 -> hold_3_30 = 0
 tp_off_limit_constr_3_31: group_tp_active_0 = 0 -> hold_3_31 = 0
 tp_off_limit_constr_3_32: group_tp_active_0 = 0 -> hold_3_32 = 0
 tp_off_limit_constr_4_21: group_tp_active_4 = 0 -> hold_4_21 = 0
 tp_off_limit_constr_4_22: group_tp_active_4 = 0 -> hold_4_22 = 0
 tp_off_limit_constr_4_23: group_tp_active_4 = 0 -> hold_4_23 = 0
 tp_off_limit_constr_4_24: group_tp_active_4 = 0 -> hold_4_24 = 0
 tp_off_limit_constr_4_25: group_tp_active_4 = 0 -> hold_4_25 = 0
 tp_off_limit_constr_4_26: group_tp_active_4 = 0 -> hold_4_26 = 0
 tp_off_limit_constr_4_27: group_tp_active_4 = 0 -> hold_4_27 = 0
 tp_off_limit_constr_4_28: group_tp_active_4 = 0 -> hold_4_28 = 0
 tp_off_limit_constr_4_29: group_tp_active_4 = 0 -> hold_4_29 = 0
 tp_off_limit_constr_4_30: group_tp_active_4 = 0 -> hold_4_30 = 0
 tp_off_limit_constr_4_31: group_tp_active_4 = 0 -> hold_4_31 = 0
 tp_off_limit_constr_4_32: group_tp_active_4 = 0 -> hold_4_32 = 0
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
 tp_off_limit_constr_5_27: group_tp_active_5 = 0 -> hold_5_27 = 0
 tp_off_limit_constr_5_28: group_tp_active_5 = 0 -> hold_5_28 = 0
 tp_off_limit_constr_5_29: group_tp_active_5 = 0 -> hold_5_29 = 0
 tp_off_limit_constr_5_30: group_tp_active_5 = 0 -> hold_5_30 = 0
 tp_off_limit_constr_5_31: group_tp_active_5 = 0 -> hold_5_31 = 0
 tp_off_limit_constr_5_32: group_tp_active_5 = 0 -> hold_5_32 = 0
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
 tp_off_limit_constr_6_25: group_tp_active_6 = 0 -> hold_6_25 = 0
 tp_off_limit_constr_6_26: group_tp_active_6 = 0 -> hold_6_26 = 0
 tp_off_limit_constr_6_27: group_tp_active_6 = 0 -> hold_6_27 = 0
 tp_off_limit_constr_6_28: group_tp_active_6 = 0 -> hold_6_28 = 0
 tp_off_limit_constr_6_29: group_tp_active_6 = 0 -> hold_6_29 = 0
 tp_off_limit_constr_6_30: group_tp_active_6 = 0 -> hold_6_30 = 0
 tp_off_limit_constr_6_31: group_tp_active_6 = 0 -> hold_6_31 = 0
 tp_off_limit_constr_6_32: group_tp_active_6 = 0 -> hold_6_32 = 0
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
 tp_off_limit_constr_7_27: group_tp_active_7 = 0 -> hold_7_27 = 0
 tp_off_limit_constr_7_28: group_tp_active_7 = 0 -> hold_7_28 = 0
 tp_off_limit_constr_7_29: group_tp_active_7 = 0 -> hold_7_29 = 0
 tp_off_limit_constr_7_30: group_tp_active_7 = 0 -> hold_7_30 = 0
 tp_off_limit_constr_7_31: group_tp_active_7 = 0 -> hold_7_31 = 0
 tp_off_limit_constr_7_32: group_tp_active_7 = 0 -> hold_7_32 = 0
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
 tp_off_limit_constr_8_27: group_tp_active_8 = 0 -> hold_8_27 = 0
 tp_off_limit_constr_8_28: group_tp_active_8 = 0 -> hold_8_28 = 0
 tp_off_limit_constr_8_29: group_tp_active_8 = 0 -> hold_8_29 = 0
 tp_off_limit_constr_8_30: group_tp_active_8 = 0 -> hold_8_30 = 0
 tp_off_limit_constr_8_31: group_tp_active_8 = 0 -> hold_8_31 = 0
 tp_off_limit_constr_8_32: group_tp_active_8 = 0 -> hold_8_32 = 0
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
 tp_off_limit_constr_9_19: group_tp_active_9 = 0 -> hold_9_19 = 0
 tp_off_limit_constr_9_20: group_tp_active_9 = 0 -> hold_9_20 = 0
 tp_off_limit_constr_9_21: group_tp_active_9 = 0 -> hold_9_21 = 0
 tp_off_limit_constr_9_22: group_tp_active_9 = 0 -> hold_9_22 = 0
 tp_off_limit_constr_9_23: group_tp_active_9 = 0 -> hold_9_23 = 0
 tp_off_limit_constr_9_24: group_tp_active_9 = 0 -> hold_9_24 = 0
 tp_off_limit_constr_9_25: group_tp_active_9 = 0 -> hold_9_25 = 0
 tp_off_limit_constr_9_26: group_tp_active_9 = 0 -> hold_9_26 = 0
 tp_off_limit_constr_9_27: group_tp_active_9 = 0 -> hold_9_27 = 0
 tp_off_limit_constr_9_28: group_tp_active_9 = 0 -> hold_9_28 = 0
 tp_off_limit_constr_9_29: group_tp_active_9 = 0 -> hold_9_29 = 0
 tp_off_limit_constr_9_30: group_tp_active_9 = 0 -> hold_9_30 = 0
 tp_off_limit_constr_9_31: group_tp_active_9 = 0 -> hold_9_31 = 0
 tp_off_limit_constr_9_32: group_tp_active_9 = 0 -> hold_9_32 = 0
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
 tp_off_limit_constr_10_27: group_tp_active_10 = 0 -> hold_10_27 = 0
 tp_off_limit_constr_10_28: group_tp_active_10 = 0 -> hold_10_28 = 0
 tp_off_limit_constr_10_29: group_tp_active_10 = 0 -> hold_10_29 = 0
 tp_off_limit_constr_10_30: group_tp_active_10 = 0 -> hold_10_30 = 0
 tp_off_limit_constr_10_31: group_tp_active_10 = 0 -> hold_10_31 = 0
 tp_off_limit_constr_10_32: group_tp_active_10 = 0 -> hold_10_32 = 0
 tp_off_limit_constr_11_9: group_tp_active_11 = 0 -> hold_11_9 = 0
 tp_off_limit_constr_11_10: group_tp_active_11 = 0 -> hold_11_10 = 0
 tp_off_limit_constr_11_11: group_tp_active_11 = 0 -> hold_11_11 = 0
 tp_off_limit_constr_11_12: group_tp_active_11 = 0 -> hold_11_12 = 0
 tp_off_limit_constr_11_13: group_tp_active_11 = 0 -> hold_11_13 = 0
 tp_off_limit_constr_11_14: group_tp_active_11 = 0 -> hold_11_14 = 0
 tp_off_limit_constr_11_15: group_tp_active_11 = 0 -> hold_11_15 = 0
 tp_off_limit_constr_11_16: group_tp_active_11 = 0 -> hold_11_16 = 0
 tp_off_limit_constr_11_17: group_tp_active_11 = 0 -> hold_11_17 = 0
 tp_off_limit_constr_11_18: group_tp_active_11 = 0 -> hold_11_18 = 0
 tp_off_limit_constr_11_19: group_tp_active_11 = 0 -> hold_11_19 = 0
 tp_off_limit_constr_11_20: group_tp_active_11 = 0 -> hold_11_20 = 0
 tp_off_limit_constr_11_21: group_tp_active_11 = 0 -> hold_11_21 = 0
 tp_off_limit_constr_11_22: group_tp_active_11 = 0 -> hold_11_22 = 0
 tp_off_limit_constr_11_23: group_tp_active_11 = 0 -> hold_11_23 = 0
 tp_off_limit_constr_11_24: group_tp_active_11 = 0 -> hold_11_24 = 0
 tp_off_limit_constr_11_25: group_tp_active_11 = 0 -> hold_11_25 = 0
 tp_off_limit_constr_11_26: group_tp_active_11 = 0 -> hold_11_26 = 0
 tp_off_limit_constr_11_27: group_tp_active_11 = 0 -> hold_11_27 = 0
 tp_off_limit_constr_11_28: group_tp_active_11 = 0 -> hold_11_28 = 0
 tp_off_limit_constr_11_29: group_tp_active_11 = 0 -> hold_11_29 = 0
 tp_off_limit_constr_11_30: group_tp_active_11 = 0 -> hold_11_30 = 0
 tp_off_limit_constr_11_31: group_tp_active_11 = 0 -> hold_11_31 = 0
 tp_off_limit_constr_11_32: group_tp_active_11 = 0 -> hold_11_32 = 0
 tp_off_limit_constr_12_21: group_tp_active_12 = 0 -> hold_12_21 = 0
 tp_off_limit_constr_12_22: group_tp_active_12 = 0 -> hold_12_22 = 0
 tp_off_limit_constr_12_23: group_tp_active_12 = 0 -> hold_12_23 = 0
 tp_off_limit_constr_12_24: group_tp_active_12 = 0 -> hold_12_24 = 0
 tp_off_limit_constr_12_25: group_tp_active_12 = 0 -> hold_12_25 = 0
 tp_off_limit_constr_12_26: group_tp_active_12 = 0 -> hold_12_26 = 0
 tp_off_limit_constr_12_27: group_tp_active_12 = 0 -> hold_12_27 = 0
 tp_off_limit_constr_12_28: group_tp_active_12 = 0 -> hold_12_28 = 0
 tp_off_limit_constr_12_29: group_tp_active_12 = 0 -> hold_12_29 = 0
 tp_off_limit_constr_12_30: group_tp_active_12 = 0 -> hold_12_30 = 0
 tp_off_limit_constr_12_31: group_tp_active_12 = 0 -> hold_12_31 = 0
 tp_off_limit_constr_12_32: group_tp_active_12 = 0 -> hold_12_32 = 0
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
 tp_off_limit_constr_13_27: group_tp_active_13 = 0 -> hold_13_27 = 0
 tp_off_limit_constr_13_28: group_tp_active_13 = 0 -> hold_13_28 = 0
 tp_off_limit_constr_13_29: group_tp_active_13 = 0 -> hold_13_29 = 0
 tp_off_limit_constr_13_30: group_tp_active_13 = 0 -> hold_13_30 = 0
 tp_off_limit_constr_13_31: group_tp_active_13 = 0 -> hold_13_31 = 0
 tp_off_limit_constr_13_32: group_tp_active_13 = 0 -> hold_13_32 = 0
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
 tp_off_limit_constr_14_27: group_tp_active_14 = 0 -> hold_14_27 = 0
 tp_off_limit_constr_14_28: group_tp_active_14 = 0 -> hold_14_28 = 0
 tp_off_limit_constr_14_29: group_tp_active_14 = 0 -> hold_14_29 = 0
 tp_off_limit_constr_14_30: group_tp_active_14 = 0 -> hold_14_30 = 0
 tp_off_limit_constr_14_31: group_tp_active_14 = 0 -> hold_14_31 = 0
 tp_off_limit_constr_14_32: group_tp_active_14 = 0 -> hold_14_32 = 0
 group_throughput_constr_0: group_tp_active_0 = 1 ->
   - 9471.827233871254 hold_0_1 - 4735.913616935627 hold_0_2
   - 3157.275744623752 hold_0_3 - 2367.956808467814 hold_0_4
   - 1894.365446774251 hold_0_5 - 1578.637872311876 hold_0_6
   - 1353.118176267322 hold_0_7 - 1183.978404233907 hold_0_8
   - 1052.425248207917 hold_0_9 - 947.1827233871254 hold_0_10
   - 861.0752030792048 hold_0_11 - 789.3189361559379 hold_0_12
   - 728.6020949131733 hold_0_13 - 676.559088133661 hold_0_14
   - 631.4551489247502 hold_0_15 - 591.9892021169534 hold_0_16
   - 557.1663078747797 hold_0_17 - 526.2126241039585 hold_0_18
   - 498.5172228353292 hold_0_19 - 473.5913616935627 hold_0_20
   - 451.0393920891073 hold_0_21 - 430.5376015396024 hold_0_22
   - 411.8185753857068 hold_0_23 - 394.6594680779689 hold_0_24
   - 378.8730893548502 hold_0_25 - 364.3010474565867 hold_0_26
   - 350.8084160693057 hold_0_27 - 338.2795440668305 hold_0_28
   - 326.614732202457 hold_0_29 - 315.7275744623751 hold_0_30
   - 305.5428139958469 hold_0_31 - 295.9946010584767 hold_0_32
   + flow_nic_in_0_0 + flow_nic_in_0_1 + flow_nic_in_0_2 + flow_nic_in_0_3
   <= 0
 group_throughput_constr_4: group_tp_active_4 = 1 -> - 12440 hold_4_1
   - 6220 hold_4_2 - 4146.666666666667 hold_4_3 - 3110 hold_4_4
   - 2488 hold_4_5 - 2073.333333333333 hold_4_6
   - 1777.142857142857 hold_4_7 - 1555 hold_4_8
   - 1382.222222222222 hold_4_9 - 1244 hold_4_10
   - 1130.909090909091 hold_4_11 - 1036.666666666667 hold_4_12
   - 956.9230769230769 hold_4_13 - 888.5714285714287 hold_4_14
   - 829.3333333333333 hold_4_15 - 777.5 hold_4_16
   - 731.7647058823529 hold_4_17 - 691.1111111111112 hold_4_18
   - 654.7368421052631 hold_4_19 - 622 hold_4_20 + flow_nic_in_4_4 <= 0
 group_throughput_constr_5: group_tp_active_5 = 1 -> - 2400 hold_5_1
   - 1200 hold_5_2 - 800 hold_5_3 - 600 hold_5_4 - 480 hold_5_5
   - 400 hold_5_6 - 342.8571428571428 hold_5_7 - 300 hold_5_8
   - 266.6666666666667 hold_5_9 - 240 hold_5_10
   - 218.1818181818182 hold_5_11 - 200 hold_5_12 + flow_nic_in_5_5 <= 0
 group_throughput_constr_6: group_tp_active_6 = 1 -> - 2400 hold_6_1
   - 1200 hold_6_2 - 800 hold_6_3 - 600 hold_6_4 - 480 hold_6_5
   - 400 hold_6_6 - 342.8571428571428 hold_6_7 - 300 hold_6_8
   - 266.6666666666667 hold_6_9 - 240 hold_6_10
   - 218.1818181818182 hold_6_11 - 200 hold_6_12 + flow_nic_in_6_6 <= 0
 group_throughput_constr_7: group_tp_active_7 = 1 -> - 2400 hold_7_1
   - 1200 hold_7_2 - 800 hold_7_3 - 600 hold_7_4 - 480 hold_7_5
   - 400 hold_7_6 - 342.8571428571428 hold_7_7 - 300 hold_7_8
   - 266.6666666666667 hold_7_9 - 240 hold_7_10
   - 218.1818181818182 hold_7_11 - 200 hold_7_12 + flow_nic_in_7_7 <= 0
 group_throughput_constr_8: group_tp_active_8 = 1 -> - 2400 hold_8_1
   - 1200 hold_8_2 - 800 hold_8_3 - 600 hold_8_4 - 480 hold_8_5
   - 400 hold_8_6 - 342.8571428571428 hold_8_7 - 300 hold_8_8
   - 266.6666666666667 hold_8_9 - 240 hold_8_10
   - 218.1818181818182 hold_8_11 - 200 hold_8_12 + flow_nic_in_8_8 <= 0
 group_throughput_constr_9: group_tp_active_9 = 1 -> - 2560 hold_9_1
   - 1280 hold_9_2 - 853.3333333333334 hold_9_3 - 640 hold_9_4
   - 512 hold_9_5 - 426.6666666666667 hold_9_6 - 365.7142857142857 hold_9_7
   - 320 hold_9_8 + flow_nic_in_9_9 <= 0
 group_throughput_constr_10: group_tp_active_10 = 1 -> - 2560 hold_10_1
   - 1280 hold_10_2 - 853.3333333333334 hold_10_3 - 640 hold_10_4
   - 512 hold_10_5 - 426.6666666666667 hold_10_6
   - 365.7142857142857 hold_10_7 - 320 hold_10_8 + flow_nic_in_10_10 <= 0
 group_throughput_constr_11: group_tp_active_11 = 1 -> - 2560 hold_11_1
   - 1280 hold_11_2 - 853.3333333333334 hold_11_3 - 640 hold_11_4
   - 512 hold_11_5 - 426.6666666666667 hold_11_6
   - 365.7142857142857 hold_11_7 - 320 hold_11_8 + flow_nic_in_11_11 <= 0
 group_throughput_constr_12: group_tp_active_12 = 1 -> - 12440 hold_12_1
   - 6220 hold_12_2 - 4146.666666666667 hold_12_3 - 3110 hold_12_4
   - 2488 hold_12_5 - 2073.333333333333 hold_12_6
   - 1777.142857142857 hold_12_7 - 1555 hold_12_8
   - 1382.222222222222 hold_12_9 - 1244 hold_12_10
   - 1130.909090909091 hold_12_11 - 1036.666666666667 hold_12_12
   - 956.9230769230769 hold_12_13 - 888.5714285714287 hold_12_14
   - 829.3333333333333 hold_12_15 - 777.5 hold_12_16
   - 731.7647058823529 hold_12_17 - 691.1111111111112 hold_12_18
   - 654.7368421052631 hold_12_19 - 622 hold_12_20 + flow_nic_in_12_12
   <= 0
 group_throughput_constr_13: group_tp_active_13 = 1 -> - 2560 hold_13_1
   - 1280 hold_13_2 - 853.3333333333334 hold_13_3 - 640 hold_13_4
   - 512 hold_13_5 - 426.6666666666667 hold_13_6
   - 365.7142857142857 hold_13_7 - 320 hold_13_8 + flow_nic_in_13_13 <= 0
 group_throughput_constr_14: group_tp_active_14 = 1 -> - 2400 hold_14_1
   - 1200 hold_14_2 - 800 hold_14_3 - 600 hold_14_4 - 480 hold_14_5
   - 400 hold_14_6 - 342.8571428571428 hold_14_7 - 300 hold_14_8
   - 266.6666666666667 hold_14_9 - 240 hold_14_10
   - 218.1818181818182 hold_14_11 - 200 hold_14_12 + flow_nic_in_14_14
   <= 0
 intergroup_group_tp_0_via_source_nic_in_0: tp_on_and_switch_source_nic_in_0_0 = 1 ->
   - 8469.103651242311 hold_0_1 - 4471.222096779602 hold_0_2
   - 3037.401806494724 hold_0_3 - 2299.881556099861 hold_0_4
   - 1850.545332577694 hold_0_5 - 1548.08946409013 hold_0_6
   - 1330.612231320283 hold_0_7 - 1166.711390603306 hold_0_8
   - 1038.760032149623 hold_0_9 - 936.0995073776344 hold_0_10
   - 851.9057664436448 hold_0_11 - 781.6072189242741 hold_0_12
   - 722.0262221873243 hold_0_13 - 670.88541780228 hold_0_14
   - 626.5099869607609 hold_0_15 - 587.6407399572198 hold_0_16
   - 553.3127178236551 hold_0_17 - 522.7739956531342 hold_0_18
   - 495.4299681872031 hold_0_19 - 470.8042513623898 hold_0_20
   - 448.5106922145832 hold_0_21 - 428.2329715835306 hold_0_22
   - 409.7095005999684 hold_0_23 - 392.7220745150864 hold_0_24
   - 377.0872367754328 hold_0_25 - 362.6496269695274 hold_0_26
   - 349.2768005546515 hold_0_27 - 336.8551539680695 hold_0_28
   - 325.2866893906527 hold_0_29 - 314.4864240141833 hold_0_30
   - 304.3802988329679 hold_0_31 - 294.9034780916206 hold_0_32
   + flow_nic_in_0_0 + flow_nic_in_0_1 + flow_nic_in_0_2 + flow_nic_in_0_3
   <= 0
 intergroup_group_tp_0_via_nic_out_4_nic_in_0: tp_on_and_switch_nic_out_4_nic_in_0_0 = 1 ->
   - 8942.444193559204 hold_0_1 - 4599.763112199722 hold_0_2
   - 3096.178928180259 hold_0_3 - 2333.422781206613 hold_0_4
   - 1872.199014755269 hold_0_5 - 1563.214437848548 hold_0_6
   - 1341.77083560456 hold_0_7 - 1175.28147991444 hold_0_8
   - 1045.547991306268 hold_0_9 - 941.6085027247797 hold_0_10
   - 856.4659431670611 hold_0_11 - 785.4441490301729 hold_0_12
   - 725.2992539390549 hold_0_13 - 673.7103079361391 hold_0_14
   - 628.9728480283667 hold_0_15 - 589.8069561832413 hold_0_16
   - 555.2328264751669 hold_0_17 - 524.4876738840541 hold_0_18
   - 496.9688009198892 hold_0_19 - 472.1936938532186 hold_0_20
   - 449.7714879733914 hold_0_21 - 429.3821941644812 hold_0_22
   - 410.7613307229451 hold_0_23 - 393.6883877673144 hold_0_24
   - 377.97805364926 hold_0_25 - 363.4734614395548 hold_0_26
   - 350.0409329115923 hold_0_27 - 337.5658464379208 hold_0_28
   - 325.9493580631111 hold_0_29 - 315.1057770705403 hold_0_30
   - 304.9604485358277 hold_0_31 - 295.4480321683818 hold_0_32
   + flow_nic_in_0_0 + flow_nic_in_0_1 + flow_nic_in_0_2 + flow_nic_in_0_3
   <= 0
 intergroup_group_tp_0_via_nic_out_5_nic_in_0: tp_on_and_switch_nic_out_5_nic_in_0_0 = 1 ->
   - 8942.444193559204 hold_0_1 - 4599.763112199722 hold_0_2
   - 3096.178928180259 hold_0_3 - 2333.422781206613 hold_0_4
   - 1872.199014755269 hold_0_5 - 1563.214437848548 hold_0_6
   - 1341.77083560456 hold_0_7 - 1175.28147991444 hold_0_8
   - 1045.547991306268 hold_0_9 - 941.6085027247797 hold_0_10
   - 856.4659431670611 hold_0_11 - 785.4441490301729 hold_0_12
   - 725.2992539390549 hold_0_13 - 673.7103079361391 hold_0_14
   - 628.9728480283667 hold_0_15 - 589.8069561832413 hold_0_16
   - 555.2328264751669 hold_0_17 - 524.4876738840541 hold_0_18
   - 496.9688009198892 hold_0_19 - 472.1936938532186 hold_0_20
   - 449.7714879733914 hold_0_21 - 429.3821941644812 hold_0_22
   - 410.7613307229451 hold_0_23 - 393.6883877673144 hold_0_24
   - 377.97805364926 hold_0_25 - 363.4734614395548 hold_0_26
   - 350.0409329115923 hold_0_27 - 337.5658464379208 hold_0_28
   - 325.9493580631111 hold_0_29 - 315.1057770705403 hold_0_30
   - 304.9604485358277 hold_0_31 - 295.4480321683818 hold_0_32
   + flow_nic_in_0_0 + flow_nic_in_0_1 + flow_nic_in_0_2 + flow_nic_in_0_3
   <= 0
 intergroup_group_tp_0_via_nic_out_6_nic_in_0: tp_on_and_switch_nic_out_6_nic_in_0_0 = 1 ->
   - 8942.444193559204 hold_0_1 - 4599.763112199722 hold_0_2
   - 3096.178928180259 hold_0_3 - 2333.422781206613 hold_0_4
   - 1872.199014755269 hold_0_5 - 1563.214437848548 hold_0_6
   - 1341.77083560456 hold_0_7 - 1175.28147991444 hold_0_8
   - 1045.547991306268 hold_0_9 - 941.6085027247797 hold_0_10
   - 856.4659431670611 hold_0_11 - 785.4441490301729 hold_0_12
   - 725.2992539390549 hold_0_13 - 673.7103079361391 hold_0_14
   - 628.9728480283667 hold_0_15 - 589.8069561832413 hold_0_16
   - 555.2328264751669 hold_0_17 - 524.4876738840541 hold_0_18
   - 496.9688009198892 hold_0_19 - 472.1936938532186 hold_0_20
   - 449.7714879733914 hold_0_21 - 429.3821941644812 hold_0_22
   - 410.7613307229451 hold_0_23 - 393.6883877673144 hold_0_24
   - 377.97805364926 hold_0_25 - 363.4734614395548 hold_0_26
   - 350.0409329115923 hold_0_27 - 337.5658464379208 hold_0_28
   - 325.9493580631111 hold_0_29 - 315.1057770705403 hold_0_30
   - 304.9604485358277 hold_0_31 - 295.4480321683818 hold_0_32
   + flow_nic_in_0_0 + flow_nic_in_0_1 + flow_nic_in_0_2 + flow_nic_in_0_3
   <= 0
 intergroup_group_tp_0_via_nic_out_7_nic_in_0: tp_on_and_switch_nic_out_7_nic_in_0_0 = 1 ->
   - 8469.103651242311 hold_0_1 - 4471.222096779602 hold_0_2
   - 3037.401806494724 hold_0_3 - 2299.881556099861 hold_0_4
   - 1850.545332577694 hold_0_5 - 1548.08946409013 hold_0_6
   - 1330.612231320283 hold_0_7 - 1166.711390603306 hold_0_8
   - 1038.760032149623 hold_0_9 - 936.0995073776344 hold_0_10
   - 851.9057664436448 hold_0_11 - 781.6072189242741 hold_0_12
   - 722.0262221873243 hold_0_13 - 670.88541780228 hold_0_14
   - 626.5099869607609 hold_0_15 - 587.6407399572198 hold_0_16
   - 553.3127178236551 hold_0_17 - 522.7739956531342 hold_0_18
   - 495.4299681872031 hold_0_19 - 470.8042513623898 hold_0_20
   - 448.5106922145832 hold_0_21 - 428.2329715835306 hold_0_22
   - 409.7095005999684 hold_0_23 - 392.7220745150864 hold_0_24
   - 377.0872367754328 hold_0_25 - 362.6496269695274 hold_0_26
   - 349.2768005546515 hold_0_27 - 336.8551539680695 hold_0_28
   - 325.2866893906527 hold_0_29 - 314.4864240141833 hold_0_30
   - 304.3802988329679 hold_0_31 - 294.9034780916206 hold_0_32
   + flow_nic_in_0_0 + flow_nic_in_0_1 + flow_nic_in_0_2 + flow_nic_in_0_3
   <= 0
 intergroup_group_tp_0_via_nic_out_8_nic_in_0: tp_on_and_switch_nic_out_8_nic_in_0_0 = 1 ->
   - 8469.103651242311 hold_0_1 - 4471.222096779602 hold_0_2
   - 3037.401806494724 hold_0_3 - 2299.881556099861 hold_0_4
   - 1850.545332577694 hold_0_5 - 1548.08946409013 hold_0_6
   - 1330.612231320283 hold_0_7 - 1166.711390603306 hold_0_8
   - 1038.760032149623 hold_0_9 - 936.0995073776344 hold_0_10
   - 851.9057664436448 hold_0_11 - 781.6072189242741 hold_0_12
   - 722.0262221873243 hold_0_13 - 670.88541780228 hold_0_14
   - 626.5099869607609 hold_0_15 - 587.6407399572198 hold_0_16
   - 553.3127178236551 hold_0_17 - 522.7739956531342 hold_0_18
   - 495.4299681872031 hold_0_19 - 470.8042513623898 hold_0_20
   - 448.5106922145832 hold_0_21 - 428.2329715835306 hold_0_22
   - 409.7095005999684 hold_0_23 - 392.7220745150864 hold_0_24
   - 377.0872367754328 hold_0_25 - 362.6496269695274 hold_0_26
   - 349.2768005546515 hold_0_27 - 336.8551539680695 hold_0_28
   - 325.2866893906527 hold_0_29 - 314.4864240141833 hold_0_30
   - 304.3802988329679 hold_0_31 - 294.9034780916206 hold_0_32
   + flow_nic_in_0_0 + flow_nic_in_0_1 + flow_nic_in_0_2 + flow_nic_in_0_3
   <= 0
 intergroup_group_tp_0_via_nic_out_9_nic_in_0: tp_on_and_switch_nic_out_9_nic_in_0_0 = 1 ->
   - 8469.103651242311 hold_0_1 - 4471.222096779602 hold_0_2
   - 3037.401806494724 hold_0_3 - 2299.881556099861 hold_0_4
   - 1850.545332577694 hold_0_5 - 1548.08946409013 hold_0_6
   - 1330.612231320283 hold_0_7 - 1166.711390603306 hold_0_8
   - 1038.760032149623 hold_0_9 - 936.0995073776344 hold_0_10
   - 851.9057664436448 hold_0_11 - 781.6072189242741 hold_0_12
   - 722.0262221873243 hold_0_13 - 670.88541780228 hold_0_14
   - 626.5099869607609 hold_0_15 - 587.6407399572198 hold_0_16
   - 553.3127178236551 hold_0_17 - 522.7739956531342 hold_0_18
   - 495.4299681872031 hold_0_19 - 470.8042513623898 hold_0_20
   - 448.5106922145832 hold_0_21 - 428.2329715835306 hold_0_22
   - 409.7095005999684 hold_0_23 - 392.7220745150864 hold_0_24
   - 377.0872367754328 hold_0_25 - 362.6496269695274 hold_0_26
   - 349.2768005546515 hold_0_27 - 336.8551539680695 hold_0_28
   - 325.2866893906527 hold_0_29 - 314.4864240141833 hold_0_30
   - 304.3802988329679 hold_0_31 - 294.9034780916206 hold_0_32
   + flow_nic_in_0_0 + flow_nic_in_0_1 + flow_nic_in_0_2 + flow_nic_in_0_3
   <= 0
 intergroup_group_tp_0_via_nic_out_10_nic_in_0: tp_on_and_switch_nic_out_10_nic_in_0_0 = 1 ->
   - 8469.103651242311 hold_0_1 - 4471.222096779602 hold_0_2
   - 3037.401806494724 hold_0_3 - 2299.881556099861 hold_0_4
   - 1850.545332577694 hold_0_5 - 1548.08946409013 hold_0_6
   - 1330.612231320283 hold_0_7 - 1166.711390603306 hold_0_8
   - 1038.760032149623 hold_0_9 - 936.0995073776344 hold_0_10
   - 851.9057664436448 hold_0_11 - 781.6072189242741 hold_0_12
   - 722.0262221873243 hold_0_13 - 670.88541780228 hold_0_14
   - 626.5099869607609 hold_0_15 - 587.6407399572198 hold_0_16
   - 553.3127178236551 hold_0_17 - 522.7739956531342 hold_0_18
   - 495.4299681872031 hold_0_19 - 470.8042513623898 hold_0_20
   - 448.5106922145832 hold_0_21 - 428.2329715835306 hold_0_22
   - 409.7095005999684 hold_0_23 - 392.7220745150864 hold_0_24
   - 377.0872367754328 hold_0_25 - 362.6496269695274 hold_0_26
   - 349.2768005546515 hold_0_27 - 336.8551539680695 hold_0_28
   - 325.2866893906527 hold_0_29 - 314.4864240141833 hold_0_30
   - 304.3802988329679 hold_0_31 - 294.9034780916206 hold_0_32
   + flow_nic_in_0_0 + flow_nic_in_0_1 + flow_nic_in_0_2 + flow_nic_in_0_3
   <= 0
 intergroup_group_tp_0_via_nic_out_11_nic_in_0: tp_on_and_switch_nic_out_11_nic_in_0_0 = 1 ->
   - 8469.103651242311 hold_0_1 - 4471.222096779602 hold_0_2
   - 3037.401806494724 hold_0_3 - 2299.881556099861 hold_0_4
   - 1850.545332577694 hold_0_5 - 1548.08946409013 hold_0_6
   - 1330.612231320283 hold_0_7 - 1166.711390603306 hold_0_8
   - 1038.760032149623 hold_0_9 - 936.0995073776344 hold_0_10
   - 851.9057664436448 hold_0_11 - 781.6072189242741 hold_0_12
   - 722.0262221873243 hold_0_13 - 670.88541780228 hold_0_14
   - 626.5099869607609 hold_0_15 - 587.6407399572198 hold_0_16
   - 553.3127178236551 hold_0_17 - 522.7739956531342 hold_0_18
   - 495.4299681872031 hold_0_19 - 470.8042513623898 hold_0_20
   - 448.5106922145832 hold_0_21 - 428.2329715835306 hold_0_22
   - 409.7095005999684 hold_0_23 - 392.7220745150864 hold_0_24
   - 377.0872367754328 hold_0_25 - 362.6496269695274 hold_0_26
   - 349.2768005546515 hold_0_27 - 336.8551539680695 hold_0_28
   - 325.2866893906527 hold_0_29 - 314.4864240141833 hold_0_30
   - 304.3802988329679 hold_0_31 - 294.9034780916206 hold_0_32
   + flow_nic_in_0_0 + flow_nic_in_0_1 + flow_nic_in_0_2 + flow_nic_in_0_3
   <= 0
 intergroup_group_tp_0_via_nic_out_12_nic_in_0: tp_on_and_switch_nic_out_12_nic_in_0_0 = 1 ->
   - 8469.103651242311 hold_0_1 - 4471.222096779602 hold_0_2
   - 3037.401806494724 hold_0_3 - 2299.881556099861 hold_0_4
   - 1850.545332577694 hold_0_5 - 1548.08946409013 hold_0_6
   - 1330.612231320283 hold_0_7 - 1166.711390603306 hold_0_8
   - 1038.760032149623 hold_0_9 - 936.0995073776344 hold_0_10
   - 851.9057664436448 hold_0_11 - 781.6072189242741 hold_0_12
   - 722.0262221873243 hold_0_13 - 670.88541780228 hold_0_14
   - 626.5099869607609 hold_0_15 - 587.6407399572198 hold_0_16
   - 553.3127178236551 hold_0_17 - 522.7739956531342 hold_0_18
   - 495.4299681872031 hold_0_19 - 470.8042513623898 hold_0_20
   - 448.5106922145832 hold_0_21 - 428.2329715835306 hold_0_22
   - 409.7095005999684 hold_0_23 - 392.7220745150864 hold_0_24
   - 377.0872367754328 hold_0_25 - 362.6496269695274 hold_0_26
   - 349.2768005546515 hold_0_27 - 336.8551539680695 hold_0_28
   - 325.2866893906527 hold_0_29 - 314.4864240141833 hold_0_30
   - 304.3802988329679 hold_0_31 - 294.9034780916206 hold_0_32
   + flow_nic_in_0_0 + flow_nic_in_0_1 + flow_nic_in_0_2 + flow_nic_in_0_3
   <= 0
 intergroup_group_tp_0_via_nic_out_13_nic_in_0: tp_on_and_switch_nic_out_13_nic_in_0_0 = 1 ->
   - 8469.103651242311 hold_0_1 - 4471.222096779602 hold_0_2
   - 3037.401806494724 hold_0_3 - 2299.881556099861 hold_0_4
   - 1850.545332577694 hold_0_5 - 1548.08946409013 hold_0_6
   - 1330.612231320283 hold_0_7 - 1166.711390603306 hold_0_8
   - 1038.760032149623 hold_0_9 - 936.0995073776344 hold_0_10
   - 851.9057664436448 hold_0_11 - 781.6072189242741 hold_0_12
   - 722.0262221873243 hold_0_13 - 670.88541780228 hold_0_14
   - 626.5099869607609 hold_0_15 - 587.6407399572198 hold_0_16
   - 553.3127178236551 hold_0_17 - 522.7739956531342 hold_0_18
   - 495.4299681872031 hold_0_19 - 470.8042513623898 hold_0_20
   - 448.5106922145832 hold_0_21 - 428.2329715835306 hold_0_22
   - 409.7095005999684 hold_0_23 - 392.7220745150864 hold_0_24
   - 377.0872367754328 hold_0_25 - 362.6496269695274 hold_0_26
   - 349.2768005546515 hold_0_27 - 336.8551539680695 hold_0_28
   - 325.2866893906527 hold_0_29 - 314.4864240141833 hold_0_30
   - 304.3802988329679 hold_0_31 - 294.9034780916206 hold_0_32
   + flow_nic_in_0_0 + flow_nic_in_0_1 + flow_nic_in_0_2 + flow_nic_in_0_3
   <= 0
 intergroup_group_tp_0_via_nic_out_14_nic_in_0: tp_on_and_switch_nic_out_14_nic_in_0_0 = 1 ->
   - 8469.103651242311 hold_0_1 - 4471.222096779602 hold_0_2
   - 3037.401806494724 hold_0_3 - 2299.881556099861 hold_0_4
   - 1850.545332577694 hold_0_5 - 1548.08946409013 hold_0_6
   - 1330.612231320283 hold_0_7 - 1166.711390603306 hold_0_8
   - 1038.760032149623 hold_0_9 - 936.0995073776344 hold_0_10
   - 851.9057664436448 hold_0_11 - 781.6072189242741 hold_0_12
   - 722.0262221873243 hold_0_13 - 670.88541780228 hold_0_14
   - 626.5099869607609 hold_0_15 - 587.6407399572198 hold_0_16
   - 553.3127178236551 hold_0_17 - 522.7739956531342 hold_0_18
   - 495.4299681872031 hold_0_19 - 470.8042513623898 hold_0_20
   - 448.5106922145832 hold_0_21 - 428.2329715835306 hold_0_22
   - 409.7095005999684 hold_0_23 - 392.7220745150864 hold_0_24
   - 377.0872367754328 hold_0_25 - 362.6496269695274 hold_0_26
   - 349.2768005546515 hold_0_27 - 336.8551539680695 hold_0_28
   - 325.2866893906527 hold_0_29 - 314.4864240141833 hold_0_30
   - 304.3802988329679 hold_0_31 - 294.9034780916206 hold_0_32
   + flow_nic_in_0_0 + flow_nic_in_0_1 + flow_nic_in_0_2 + flow_nic_in_0_3
   <= 0
Bounds
Binaries
 hold_0_1 hold_0_2 hold_0_3 hold_0_4 hold_0_5 hold_0_6 hold_0_7 hold_0_8
 hold_0_9 hold_0_10 hold_0_11 hold_0_12 hold_0_13 hold_0_14 hold_0_15
 hold_0_16 hold_0_17 hold_0_18 hold_0_19 hold_0_20 hold_0_21 hold_0_22
 hold_0_23 hold_0_24 hold_0_25 hold_0_26 hold_0_27 hold_0_28 hold_0_29
 hold_0_30 hold_0_31 hold_0_32 hold_1_1 hold_1_2 hold_1_3 hold_1_4 hold_1_5
 hold_1_6 hold_1_7 hold_1_8 hold_1_9 hold_1_10 hold_1_11 hold_1_12
 hold_1_13 hold_1_14 hold_1_15 hold_1_16 hold_1_17 hold_1_18 hold_1_19
 hold_1_20 hold_1_21 hold_1_22 hold_1_23 hold_1_24 hold_1_25 hold_1_26
 hold_1_27 hold_1_28 hold_1_29 hold_1_30 hold_1_31 hold_1_32 hold_2_1
 hold_2_2 hold_2_3 hold_2_4 hold_2_5 hold_2_6 hold_2_7 hold_2_8 hold_2_9
 hold_2_10 hold_2_11 hold_2_12 hold_2_13 hold_2_14 hold_2_15 hold_2_16
 hold_2_17 hold_2_18 hold_2_19 hold_2_20 hold_2_21 hold_2_22 hold_2_23
 hold_2_24 hold_2_25 hold_2_26 hold_2_27 hold_2_28 hold_2_29 hold_2_30
 hold_2_31 hold_2_32 hold_3_1 hold_3_2 hold_3_3 hold_3_4 hold_3_5 hold_3_6
 hold_3_7 hold_3_8 hold_3_9 hold_3_10 hold_3_11 hold_3_12 hold_3_13
 hold_3_14 hold_3_15 hold_3_16 hold_3_17 hold_3_18 hold_3_19 hold_3_20
 hold_3_21 hold_3_22 hold_3_23 hold_3_24 hold_3_25 hold_3_26 hold_3_27
 hold_3_28 hold_3_29 hold_3_30 hold_3_31 hold_3_32 hold_4_1 hold_4_2
 hold_4_3 hold_4_4 hold_4_5 hold_4_6 hold_4_7 hold_4_8 hold_4_9 hold_4_10
 hold_4_11 hold_4_12 hold_4_13 hold_4_14 hold_4_15 hold_4_16 hold_4_17
 hold_4_18 hold_4_19 hold_4_20 hold_4_21 hold_4_22 hold_4_23 hold_4_24
 hold_4_25 hold_4_26 hold_4_27 hold_4_28 hold_4_29 hold_4_30 hold_4_31
 hold_4_32 hold_5_1 hold_5_2 hold_5_3 hold_5_4 hold_5_5 hold_5_6 hold_5_7
 hold_5_8 hold_5_9 hold_5_10 hold_5_11 hold_5_12 hold_5_13 hold_5_14
 hold_5_15 hold_5_16 hold_5_17 hold_5_18 hold_5_19 hold_5_20 hold_5_21
 hold_5_22 hold_5_23 hold_5_24 hold_5_25 hold_5_26 hold_5_27 hold_5_28
 hold_5_29 hold_5_30 hold_5_31 hold_5_32 hold_6_1 hold_6_2 hold_6_3
 hold_6_4 hold_6_5 hold_6_6 hold_6_7 hold_6_8 hold_6_9 hold_6_10 hold_6_11
 hold_6_12 hold_6_13 hold_6_14 hold_6_15 hold_6_16 hold_6_17 hold_6_18
 hold_6_19 hold_6_20 hold_6_21 hold_6_22 hold_6_23 hold_6_24 hold_6_25
 hold_6_26 hold_6_27 hold_6_28 hold_6_29 hold_6_30 hold_6_31 hold_6_32
 hold_7_1 hold_7_2 hold_7_3 hold_7_4 hold_7_5 hold_7_6 hold_7_7 hold_7_8
 hold_7_9 hold_7_10 hold_7_11 hold_7_12 hold_7_13 hold_7_14 hold_7_15
 hold_7_16 hold_7_17 hold_7_18 hold_7_19 hold_7_20 hold_7_21 hold_7_22
 hold_7_23 hold_7_24 hold_7_25 hold_7_26 hold_7_27 hold_7_28 hold_7_29
 hold_7_30 hold_7_31 hold_7_32 hold_8_1 hold_8_2 hold_8_3 hold_8_4 hold_8_5
 hold_8_6 hold_8_7 hold_8_8 hold_8_9 hold_8_10 hold_8_11 hold_8_12
 hold_8_13 hold_8_14 hold_8_15 hold_8_16 hold_8_17 hold_8_18 hold_8_19
 hold_8_20 hold_8_21 hold_8_22 hold_8_23 hold_8_24 hold_8_25 hold_8_26
 hold_8_27 hold_8_28 hold_8_29 hold_8_30 hold_8_31 hold_8_32 hold_9_1
 hold_9_2 hold_9_3 hold_9_4 hold_9_5 hold_9_6 hold_9_7 hold_9_8 hold_9_9
 hold_9_10 hold_9_11 hold_9_12 hold_9_13 hold_9_14 hold_9_15 hold_9_16
 hold_9_17 hold_9_18 hold_9_19 hold_9_20 hold_9_21 hold_9_22 hold_9_23
 hold_9_24 hold_9_25 hold_9_26 hold_9_27 hold_9_28 hold_9_29 hold_9_30
 hold_9_31 hold_9_32 hold_10_1 hold_10_2 hold_10_3 hold_10_4 hold_10_5
 hold_10_6 hold_10_7 hold_10_8 hold_10_9 hold_10_10 hold_10_11 hold_10_12
 hold_10_13 hold_10_14 hold_10_15 hold_10_16 hold_10_17 hold_10_18
 hold_10_19 hold_10_20 hold_10_21 hold_10_22 hold_10_23 hold_10_24
 hold_10_25 hold_10_26 hold_10_27 hold_10_28 hold_10_29 hold_10_30
 hold_10_31 hold_10_32 hold_11_1 hold_11_2 hold_11_3 hold_11_4 hold_11_5
 hold_11_6 hold_11_7 hold_11_8 hold_11_9 hold_11_10 hold_11_11 hold_11_12
 hold_11_13 hold_11_14 hold_11_15 hold_11_16 hold_11_17 hold_11_18
 hold_11_19 hold_11_20 hold_11_21 hold_11_22 hold_11_23 hold_11_24
 hold_11_25 hold_11_26 hold_11_27 hold_11_28 hold_11_29 hold_11_30
 hold_11_31 hold_11_32 hold_12_1 hold_12_2 hold_12_3 hold_12_4 hold_12_5
 hold_12_6 hold_12_7 hold_12_8 hold_12_9 hold_12_10 hold_12_11 hold_12_12
 hold_12_13 hold_12_14 hold_12_15 hold_12_16 hold_12_17 hold_12_18
 hold_12_19 hold_12_20 hold_12_21 hold_12_22 hold_12_23 hold_12_24
 hold_12_25 hold_12_26 hold_12_27 hold_12_28 hold_12_29 hold_12_30
 hold_12_31 hold_12_32 hold_13_1 hold_13_2 hold_13_3 hold_13_4 hold_13_5
 hold_13_6 hold_13_7 hold_13_8 hold_13_9 hold_13_10 hold_13_11 hold_13_12
 hold_13_13 hold_13_14 hold_13_15 hold_13_16 hold_13_17 hold_13_18
 hold_13_19 hold_13_20 hold_13_21 hold_13_22 hold_13_23 hold_13_24
 hold_13_25 hold_13_26 hold_13_27 hold_13_28 hold_13_29 hold_13_30
 hold_13_31 hold_13_32 hold_14_1 hold_14_2 hold_14_3 hold_14_4 hold_14_5
 hold_14_6 hold_14_7 hold_14_8 hold_14_9 hold_14_10 hold_14_11 hold_14_12
 hold_14_13 hold_14_14 hold_14_15 hold_14_16 hold_14_17 hold_14_18
 hold_14_19 hold_14_20 hold_14_21 hold_14_22 hold_14_23 hold_14_24
 hold_14_25 hold_14_26 hold_14_27 hold_14_28 hold_14_29 hold_14_30
 hold_14_31 hold_14_32 switch_source_nic_in_0 switch_nic_out_0_sink
 switch_nic_out_0_nic_in_4 switch_nic_out_0_nic_in_5
 switch_nic_out_0_nic_in_6 switch_nic_out_0_nic_in_7
 switch_nic_out_0_nic_in_8 switch_nic_out_0_nic_in_9
 switch_nic_out_0_nic_in_10 switch_nic_out_0_nic_in_11
 switch_nic_out_0_nic_in_12 switch_nic_out_0_nic_in_13
 switch_nic_out_0_nic_in_14 switch_nic_in_0_0 switch_0_nic_out_0
 switch_nic_in_0_1 switch_1_nic_out_0 switch_nic_in_0_2 switch_2_nic_out_0
 switch_nic_in_0_3 switch_3_nic_out_0 switch_0_1 switch_0_2 switch_0_3
 switch_1_0 switch_1_2 switch_1_3 switch_2_0 switch_2_1 switch_2_3
 switch_3_0 switch_3_1 switch_3_2 switch_source_nic_in_4
 switch_nic_out_4_sink switch_nic_out_4_nic_in_0 switch_nic_out_4_nic_in_5
 switch_nic_out_4_nic_in_6 switch_nic_out_4_nic_in_7
 switch_nic_out_4_nic_in_8 switch_nic_out_4_nic_in_9
 switch_nic_out_4_nic_in_10 switch_nic_out_4_nic_in_11
 switch_nic_out_4_nic_in_12 switch_nic_out_4_nic_in_13
 switch_nic_out_4_nic_in_14 switch_nic_in_4_4 switch_4_nic_out_4
 switch_source_nic_in_5 switch_nic_out_5_sink switch_nic_out_5_nic_in_0
 switch_nic_out_5_nic_in_4 switch_nic_out_5_nic_in_6
 switch_nic_out_5_nic_in_7 switch_nic_out_5_nic_in_8
 switch_nic_out_5_nic_in_9 switch_nic_out_5_nic_in_10
 switch_nic_out_5_nic_in_11 switch_nic_out_5_nic_in_12
 switch_nic_out_5_nic_in_13 switch_nic_out_5_nic_in_14 switch_nic_in_5_5
 switch_5_nic_out_5 switch_source_nic_in_6 switch_nic_out_6_sink
 switch_nic_out_6_nic_in_0 switch_nic_out_6_nic_in_4
 switch_nic_out_6_nic_in_5 switch_nic_out_6_nic_in_7
 switch_nic_out_6_nic_in_8 switch_nic_out_6_nic_in_9
 switch_nic_out_6_nic_in_10 switch_nic_out_6_nic_in_11
 switch_nic_out_6_nic_in_12 switch_nic_out_6_nic_in_13
 switch_nic_out_6_nic_in_14 switch_nic_in_6_6 switch_6_nic_out_6
 switch_source_nic_in_7 switch_nic_out_7_sink switch_nic_out_7_nic_in_0
 switch_nic_out_7_nic_in_4 switch_nic_out_7_nic_in_5
 switch_nic_out_7_nic_in_6 switch_nic_out_7_nic_in_8
 switch_nic_out_7_nic_in_9 switch_nic_out_7_nic_in_10
 switch_nic_out_7_nic_in_11 switch_nic_out_7_nic_in_12
 switch_nic_out_7_nic_in_13 switch_nic_out_7_nic_in_14 switch_nic_in_7_7
 switch_7_nic_out_7 switch_source_nic_in_8 switch_nic_out_8_sink
 switch_nic_out_8_nic_in_0 switch_nic_out_8_nic_in_4
 switch_nic_out_8_nic_in_5 switch_nic_out_8_nic_in_6
 switch_nic_out_8_nic_in_7 switch_nic_out_8_nic_in_9
 switch_nic_out_8_nic_in_10 switch_nic_out_8_nic_in_11
 switch_nic_out_8_nic_in_12 switch_nic_out_8_nic_in_13
 switch_nic_out_8_nic_in_14 switch_nic_in_8_8 switch_8_nic_out_8
 switch_source_nic_in_9 switch_nic_out_9_sink switch_nic_out_9_nic_in_0
 switch_nic_out_9_nic_in_4 switch_nic_out_9_nic_in_5
 switch_nic_out_9_nic_in_6 switch_nic_out_9_nic_in_7
 switch_nic_out_9_nic_in_8 switch_nic_out_9_nic_in_10
 switch_nic_out_9_nic_in_11 switch_nic_out_9_nic_in_12
 switch_nic_out_9_nic_in_13 switch_nic_out_9_nic_in_14 switch_nic_in_9_9
 switch_9_nic_out_9 switch_source_nic_in_10 switch_nic_out_10_sink
 switch_nic_out_10_nic_in_0 switch_nic_out_10_nic_in_4
 switch_nic_out_10_nic_in_5 switch_nic_out_10_nic_in_6
 switch_nic_out_10_nic_in_7 switch_nic_out_10_nic_in_8
 switch_nic_out_10_nic_in_9 switch_nic_out_10_nic_in_11
 switch_nic_out_10_nic_in_12 switch_nic_out_10_nic_in_13
 switch_nic_out_10_nic_in_14 switch_nic_in_10_10 switch_10_nic_out_10
 switch_source_nic_in_11 switch_nic_out_11_sink switch_nic_out_11_nic_in_0
 switch_nic_out_11_nic_in_4 switch_nic_out_11_nic_in_5
 switch_nic_out_11_nic_in_6 switch_nic_out_11_nic_in_7
 switch_nic_out_11_nic_in_8 switch_nic_out_11_nic_in_9
 switch_nic_out_11_nic_in_10 switch_nic_out_11_nic_in_12
 switch_nic_out_11_nic_in_13 switch_nic_out_11_nic_in_14
 switch_nic_in_11_11 switch_11_nic_out_11 switch_source_nic_in_12
 switch_nic_out_12_sink switch_nic_out_12_nic_in_0
 switch_nic_out_12_nic_in_4 switch_nic_out_12_nic_in_5
 switch_nic_out_12_nic_in_6 switch_nic_out_12_nic_in_7
 switch_nic_out_12_nic_in_8 switch_nic_out_12_nic_in_9
 switch_nic_out_12_nic_in_10 switch_nic_out_12_nic_in_11
 switch_nic_out_12_nic_in_13 switch_nic_out_12_nic_in_14
 switch_nic_in_12_12 switch_12_nic_out_12 switch_source_nic_in_13
 switch_nic_out_13_sink switch_nic_out_13_nic_in_0
 switch_nic_out_13_nic_in_4 switch_nic_out_13_nic_in_5
 switch_nic_out_13_nic_in_6 switch_nic_out_13_nic_in_7
 switch_nic_out_13_nic_in_8 switch_nic_out_13_nic_in_9
 switch_nic_out_13_nic_in_10 switch_nic_out_13_nic_in_11
 switch_nic_out_13_nic_in_12 switch_nic_out_13_nic_in_14
 switch_nic_in_13_13 switch_13_nic_out_13 switch_source_nic_in_14
 switch_nic_out_14_sink switch_nic_out_14_nic_in_0
 switch_nic_out_14_nic_in_4 switch_nic_out_14_nic_in_5
 switch_nic_out_14_nic_in_6 switch_nic_out_14_nic_in_7
 switch_nic_out_14_nic_in_8 switch_nic_out_14_nic_in_9
 switch_nic_out_14_nic_in_10 switch_nic_out_14_nic_in_11
 switch_nic_out_14_nic_in_12 switch_nic_out_14_nic_in_13
 switch_nic_in_14_14 switch_14_nic_out_14 group_tp_active_0
 group_tp_active_4 group_tp_active_5 group_tp_active_6 group_tp_active_7
 group_tp_active_8 group_tp_active_9 group_tp_active_10 group_tp_active_11
 group_tp_active_12 group_tp_active_13 group_tp_active_14 is_active_0
 is_active_1 is_active_2 is_active_3 is_active_4 is_active_5 is_active_6
 is_active_7 is_active_8 is_active_9 is_active_10 is_active_11 is_active_12
 is_active_13 is_active_14 is_start_zero_0_for_source_nic_in_0
 is_start_zero_1_for_source_nic_in_0 is_start_zero_2_for_source_nic_in_0
 is_start_zero_3_for_source_nic_in_0 is_end_complete_0 is_end_complete_1
 is_end_complete_2 is_end_complete_3 pp_permission_0_1 pp_permission_0_2
 pp_permission_0_3 pp_permission_1_0 pp_permission_1_2 pp_permission_1_3
 pp_permission_2_0 pp_permission_2_1 pp_permission_2_3 pp_permission_3_0
 pp_permission_3_1 pp_permission_3_2 is_start_zero_4_for_source_nic_in_4
 is_end_complete_4 is_start_zero_5_for_source_nic_in_5 is_end_complete_5
 is_start_zero_6_for_source_nic_in_6 is_end_complete_6
 is_start_zero_7_for_source_nic_in_7 is_end_complete_7
 is_start_zero_8_for_source_nic_in_8 is_end_complete_8
 is_start_zero_9_for_source_nic_in_9 is_end_complete_9
 is_start_zero_10_for_source_nic_in_10 is_end_complete_10
 is_start_zero_11_for_source_nic_in_11 is_end_complete_11
 is_start_zero_12_for_source_nic_in_12 is_end_complete_12
 is_start_zero_13_for_source_nic_in_13 is_end_complete_13
 is_start_zero_14_for_source_nic_in_14 is_end_complete_14
 tp_on_and_switch_source_nic_in_0_0 tp_on_and_switch_nic_out_4_nic_in_0_0
 tp_on_and_switch_nic_out_5_nic_in_0_0
 tp_on_and_switch_nic_out_6_nic_in_0_0
 tp_on_and_switch_nic_out_7_nic_in_0_0
 tp_on_and_switch_nic_out_8_nic_in_0_0
 tp_on_and_switch_nic_out_9_nic_in_0_0
 tp_on_and_switch_nic_out_10_nic_in_0_0
 tp_on_and_switch_nic_out_11_nic_in_0_0
 tp_on_and_switch_nic_out_12_nic_in_0_0
 tp_on_and_switch_nic_out_13_nic_in_0_0
 tp_on_and_switch_nic_out_14_nic_in_0_0
Generals
 start_0 start_1 start_2 start_3 start_4 start_5 start_6 start_7 start_8
 start_9 start_10 start_11 start_12 start_13 start_14 group_start_4
 val_end_0 val_end_1 val_end_2 val_end_3 group_end_0 group_start_5
 group_start_6 group_start_7 group_start_8 group_start_9 group_start_10
 group_start_11 group_start_12 group_start_13 group_start_14 group_start_0
 val_end_4 group_end_4 val_end_5 group_end_5 val_end_6 group_end_6
 val_end_7 group_end_7 val_end_8 group_end_8 val_end_9 group_end_9
 val_end_10 group_end_10 val_end_11 group_end_11 val_end_12 group_end_12
 val_end_13 group_end_13 val_end_14 group_end_14
General Constraints
 min_start_4: group_start_4 = MIN ( start_4 )
 max_end_0: group_end_0 = MAX ( val_end_0 , val_end_1 , val_end_2 ,
   val_end_3 )
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
 min_start_0: group_start_0 = MIN ( start_0 , start_1 , start_2 , start_3 )
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
End
