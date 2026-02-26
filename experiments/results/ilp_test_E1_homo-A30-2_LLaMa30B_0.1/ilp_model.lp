\ Model 2026-02-24_03-52-34
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  5.02412e-05 flow_source_nic_in_0 + 5.02412e-05 flow_source_nic_in_1
   + 5.02412e-05 flow_source_nic_in_2 + 5.02412e-05 flow_source_nic_in_3
   + 5.02412e-05 flow_source_nic_in_4 + 5.02412e-05 flow_source_nic_in_5
   + 5.02412e-05 flow_source_nic_in_6 + 5.02412e-05 flow_source_nic_in_7
   + 5.02412e-05 flow_source_nic_in_8 + 5.02412e-05 flow_source_nic_in_9
   + 5.02412e-05 flow_source_nic_in_10 + 5.02412e-05 flow_source_nic_in_11
   + 5.02412e-05 flow_source_nic_in_12 + 5.02412e-05 flow_source_nic_in_13
   + 5.02412e-05 flow_source_nic_in_14 + 5.02412e-05 flow_source_nic_in_15
   - 0.05625 is_active_0 - 0.05625 is_active_1 - 0.05625 is_active_2
   - 0.05625 is_active_3 - 0.05625 is_active_4 - 0.05625 is_active_5
   - 0.05625 is_active_6 - 0.05625 is_active_7 - 0.05625 is_active_8
   - 0.05625 is_active_9 - 0.05625 is_active_10 - 0.05625 is_active_11
   - 0.05625 is_active_12 - 0.05625 is_active_13 - 0.05625 is_active_14
   - 0.05625 is_active_15 + 0 val_end_0 + 0 val_end_1 + 0 val_end_2
   + 0 val_end_3 + 0 val_end_4 + 0 val_end_5 + 0 val_end_6 + 0 val_end_7
   + 0 val_end_8 + 0 val_end_9 + 0 val_end_10 + 0 val_end_11 + 0 val_end_12
   + 0 val_end_13 + 0 val_end_14 + 0 val_end_15
Subject To
 mem_active_constr_0: hold_0_1 + hold_0_2 + hold_0_3 + hold_0_4 + hold_0_5
   + hold_0_6 + hold_0_7 + hold_0_8 + hold_0_9 + hold_0_10 + hold_0_11
   + hold_0_12 - 61 is_active_0 <= 0
 mem_active_constr2_0: hold_0_1 + hold_0_2 + hold_0_3 + hold_0_4 + hold_0_5
   + hold_0_6 + hold_0_7 + hold_0_8 + hold_0_9 + hold_0_10 + hold_0_11
   + hold_0_12 - is_active_0 >= 0
 mem_active_constr_1: hold_1_1 + hold_1_2 + hold_1_3 + hold_1_4 + hold_1_5
   + hold_1_6 + hold_1_7 + hold_1_8 + hold_1_9 + hold_1_10 + hold_1_11
   + hold_1_12 - 61 is_active_1 <= 0
 mem_active_constr2_1: hold_1_1 + hold_1_2 + hold_1_3 + hold_1_4 + hold_1_5
   + hold_1_6 + hold_1_7 + hold_1_8 + hold_1_9 + hold_1_10 + hold_1_11
   + hold_1_12 - is_active_1 >= 0
 mem_active_constr_2: hold_2_1 + hold_2_2 + hold_2_3 + hold_2_4 + hold_2_5
   + hold_2_6 + hold_2_7 + hold_2_8 + hold_2_9 + hold_2_10 + hold_2_11
   + hold_2_12 - 61 is_active_2 <= 0
 mem_active_constr2_2: hold_2_1 + hold_2_2 + hold_2_3 + hold_2_4 + hold_2_5
   + hold_2_6 + hold_2_7 + hold_2_8 + hold_2_9 + hold_2_10 + hold_2_11
   + hold_2_12 - is_active_2 >= 0
 mem_active_constr_3: hold_3_1 + hold_3_2 + hold_3_3 + hold_3_4 + hold_3_5
   + hold_3_6 + hold_3_7 + hold_3_8 + hold_3_9 + hold_3_10 + hold_3_11
   + hold_3_12 - 61 is_active_3 <= 0
 mem_active_constr2_3: hold_3_1 + hold_3_2 + hold_3_3 + hold_3_4 + hold_3_5
   + hold_3_6 + hold_3_7 + hold_3_8 + hold_3_9 + hold_3_10 + hold_3_11
   + hold_3_12 - is_active_3 >= 0
 mem_active_constr_4: hold_4_1 + hold_4_2 + hold_4_3 + hold_4_4 + hold_4_5
   + hold_4_6 + hold_4_7 + hold_4_8 + hold_4_9 + hold_4_10 + hold_4_11
   + hold_4_12 - 61 is_active_4 <= 0
 mem_active_constr2_4: hold_4_1 + hold_4_2 + hold_4_3 + hold_4_4 + hold_4_5
   + hold_4_6 + hold_4_7 + hold_4_8 + hold_4_9 + hold_4_10 + hold_4_11
   + hold_4_12 - is_active_4 >= 0
 mem_active_constr_5: hold_5_1 + hold_5_2 + hold_5_3 + hold_5_4 + hold_5_5
   + hold_5_6 + hold_5_7 + hold_5_8 + hold_5_9 + hold_5_10 + hold_5_11
   + hold_5_12 - 61 is_active_5 <= 0
 mem_active_constr2_5: hold_5_1 + hold_5_2 + hold_5_3 + hold_5_4 + hold_5_5
   + hold_5_6 + hold_5_7 + hold_5_8 + hold_5_9 + hold_5_10 + hold_5_11
   + hold_5_12 - is_active_5 >= 0
 mem_active_constr_6: hold_6_1 + hold_6_2 + hold_6_3 + hold_6_4 + hold_6_5
   + hold_6_6 + hold_6_7 + hold_6_8 + hold_6_9 + hold_6_10 + hold_6_11
   + hold_6_12 - 61 is_active_6 <= 0
 mem_active_constr2_6: hold_6_1 + hold_6_2 + hold_6_3 + hold_6_4 + hold_6_5
   + hold_6_6 + hold_6_7 + hold_6_8 + hold_6_9 + hold_6_10 + hold_6_11
   + hold_6_12 - is_active_6 >= 0
 mem_active_constr_7: hold_7_1 + hold_7_2 + hold_7_3 + hold_7_4 + hold_7_5
   + hold_7_6 + hold_7_7 + hold_7_8 + hold_7_9 + hold_7_10 + hold_7_11
   + hold_7_12 - 61 is_active_7 <= 0
 mem_active_constr2_7: hold_7_1 + hold_7_2 + hold_7_3 + hold_7_4 + hold_7_5
   + hold_7_6 + hold_7_7 + hold_7_8 + hold_7_9 + hold_7_10 + hold_7_11
   + hold_7_12 - is_active_7 >= 0
 mem_active_constr_8: hold_8_1 + hold_8_2 + hold_8_3 + hold_8_4 + hold_8_5
   + hold_8_6 + hold_8_7 + hold_8_8 + hold_8_9 + hold_8_10 + hold_8_11
   + hold_8_12 - 61 is_active_8 <= 0
 mem_active_constr2_8: hold_8_1 + hold_8_2 + hold_8_3 + hold_8_4 + hold_8_5
   + hold_8_6 + hold_8_7 + hold_8_8 + hold_8_9 + hold_8_10 + hold_8_11
   + hold_8_12 - is_active_8 >= 0
 mem_active_constr_9: hold_9_1 + hold_9_2 + hold_9_3 + hold_9_4 + hold_9_5
   + hold_9_6 + hold_9_7 + hold_9_8 + hold_9_9 + hold_9_10 + hold_9_11
   + hold_9_12 - 61 is_active_9 <= 0
 mem_active_constr2_9: hold_9_1 + hold_9_2 + hold_9_3 + hold_9_4 + hold_9_5
   + hold_9_6 + hold_9_7 + hold_9_8 + hold_9_9 + hold_9_10 + hold_9_11
   + hold_9_12 - is_active_9 >= 0
 mem_active_constr_10: hold_10_1 + hold_10_2 + hold_10_3 + hold_10_4
   + hold_10_5 + hold_10_6 + hold_10_7 + hold_10_8 + hold_10_9 + hold_10_10
   + hold_10_11 + hold_10_12 - 61 is_active_10 <= 0
 mem_active_constr2_10: hold_10_1 + hold_10_2 + hold_10_3 + hold_10_4
   + hold_10_5 + hold_10_6 + hold_10_7 + hold_10_8 + hold_10_9 + hold_10_10
   + hold_10_11 + hold_10_12 - is_active_10 >= 0
 mem_active_constr_11: hold_11_1 + hold_11_2 + hold_11_3 + hold_11_4
   + hold_11_5 + hold_11_6 + hold_11_7 + hold_11_8 + hold_11_9 + hold_11_10
   + hold_11_11 + hold_11_12 - 61 is_active_11 <= 0
 mem_active_constr2_11: hold_11_1 + hold_11_2 + hold_11_3 + hold_11_4
   + hold_11_5 + hold_11_6 + hold_11_7 + hold_11_8 + hold_11_9 + hold_11_10
   + hold_11_11 + hold_11_12 - is_active_11 >= 0
 mem_active_constr_12: hold_12_1 + hold_12_2 + hold_12_3 + hold_12_4
   + hold_12_5 + hold_12_6 + hold_12_7 + hold_12_8 + hold_12_9 + hold_12_10
   + hold_12_11 + hold_12_12 - 61 is_active_12 <= 0
 mem_active_constr2_12: hold_12_1 + hold_12_2 + hold_12_3 + hold_12_4
   + hold_12_5 + hold_12_6 + hold_12_7 + hold_12_8 + hold_12_9 + hold_12_10
   + hold_12_11 + hold_12_12 - is_active_12 >= 0
 mem_active_constr_13: hold_13_1 + hold_13_2 + hold_13_3 + hold_13_4
   + hold_13_5 + hold_13_6 + hold_13_7 + hold_13_8 + hold_13_9 + hold_13_10
   + hold_13_11 + hold_13_12 - 61 is_active_13 <= 0
 mem_active_constr2_13: hold_13_1 + hold_13_2 + hold_13_3 + hold_13_4
   + hold_13_5 + hold_13_6 + hold_13_7 + hold_13_8 + hold_13_9 + hold_13_10
   + hold_13_11 + hold_13_12 - is_active_13 >= 0
 mem_active_constr_14: hold_14_1 + hold_14_2 + hold_14_3 + hold_14_4
   + hold_14_5 + hold_14_6 + hold_14_7 + hold_14_8 + hold_14_9 + hold_14_10
   + hold_14_11 + hold_14_12 - 61 is_active_14 <= 0
 mem_active_constr2_14: hold_14_1 + hold_14_2 + hold_14_3 + hold_14_4
   + hold_14_5 + hold_14_6 + hold_14_7 + hold_14_8 + hold_14_9 + hold_14_10
   + hold_14_11 + hold_14_12 - is_active_14 >= 0
 mem_active_constr_15: hold_15_1 + hold_15_2 + hold_15_3 + hold_15_4
   + hold_15_5 + hold_15_6 + hold_15_7 + hold_15_8 + hold_15_9 + hold_15_10
   + hold_15_11 + hold_15_12 - 61 is_active_15 <= 0
 mem_active_constr2_15: hold_15_1 + hold_15_2 + hold_15_3 + hold_15_4
   + hold_15_5 + hold_15_6 + hold_15_7 + hold_15_8 + hold_15_9 + hold_15_10
   + hold_15_11 + hold_15_12 - is_active_15 >= 0
 hold_constraint_0: hold_0_1 + hold_0_2 + hold_0_3 + hold_0_4 + hold_0_5
   + hold_0_6 + hold_0_7 + hold_0_8 + hold_0_9 + hold_0_10 + hold_0_11
   + hold_0_12 <= 1
 end_constraint_0: start_0 + hold_0_1 + 2 hold_0_2 + 3 hold_0_3
   + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7 + 8 hold_0_8
   + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12 <= 60
 hold_constraint_1: hold_1_1 + hold_1_2 + hold_1_3 + hold_1_4 + hold_1_5
   + hold_1_6 + hold_1_7 + hold_1_8 + hold_1_9 + hold_1_10 + hold_1_11
   + hold_1_12 <= 1
 end_constraint_1: start_1 + hold_1_1 + 2 hold_1_2 + 3 hold_1_3
   + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7 + 8 hold_1_8
   + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 + 12 hold_1_12 <= 60
 hold_constraint_2: hold_2_1 + hold_2_2 + hold_2_3 + hold_2_4 + hold_2_5
   + hold_2_6 + hold_2_7 + hold_2_8 + hold_2_9 + hold_2_10 + hold_2_11
   + hold_2_12 <= 1
 end_constraint_2: start_2 + hold_2_1 + 2 hold_2_2 + 3 hold_2_3
   + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7 + 8 hold_2_8
   + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 12 hold_2_12 <= 60
 hold_constraint_3: hold_3_1 + hold_3_2 + hold_3_3 + hold_3_4 + hold_3_5
   + hold_3_6 + hold_3_7 + hold_3_8 + hold_3_9 + hold_3_10 + hold_3_11
   + hold_3_12 <= 1
 end_constraint_3: start_3 + hold_3_1 + 2 hold_3_2 + 3 hold_3_3
   + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7 + 8 hold_3_8
   + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11 + 12 hold_3_12 <= 60
 hold_constraint_4: hold_4_1 + hold_4_2 + hold_4_3 + hold_4_4 + hold_4_5
   + hold_4_6 + hold_4_7 + hold_4_8 + hold_4_9 + hold_4_10 + hold_4_11
   + hold_4_12 <= 1
 end_constraint_4: start_4 + hold_4_1 + 2 hold_4_2 + 3 hold_4_3
   + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7 + 8 hold_4_8
   + 9 hold_4_9 + 10 hold_4_10 + 11 hold_4_11 + 12 hold_4_12 <= 60
 hold_constraint_5: hold_5_1 + hold_5_2 + hold_5_3 + hold_5_4 + hold_5_5
   + hold_5_6 + hold_5_7 + hold_5_8 + hold_5_9 + hold_5_10 + hold_5_11
   + hold_5_12 <= 1
 end_constraint_5: start_5 + hold_5_1 + 2 hold_5_2 + 3 hold_5_3
   + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7 + 8 hold_5_8
   + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12 <= 60
 hold_constraint_6: hold_6_1 + hold_6_2 + hold_6_3 + hold_6_4 + hold_6_5
   + hold_6_6 + hold_6_7 + hold_6_8 + hold_6_9 + hold_6_10 + hold_6_11
   + hold_6_12 <= 1
 end_constraint_6: start_6 + hold_6_1 + 2 hold_6_2 + 3 hold_6_3
   + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7 + 8 hold_6_8
   + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 12 hold_6_12 <= 60
 hold_constraint_7: hold_7_1 + hold_7_2 + hold_7_3 + hold_7_4 + hold_7_5
   + hold_7_6 + hold_7_7 + hold_7_8 + hold_7_9 + hold_7_10 + hold_7_11
   + hold_7_12 <= 1
 end_constraint_7: start_7 + hold_7_1 + 2 hold_7_2 + 3 hold_7_3
   + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7 + 8 hold_7_8
   + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11 + 12 hold_7_12 <= 60
 hold_constraint_8: hold_8_1 + hold_8_2 + hold_8_3 + hold_8_4 + hold_8_5
   + hold_8_6 + hold_8_7 + hold_8_8 + hold_8_9 + hold_8_10 + hold_8_11
   + hold_8_12 <= 1
 end_constraint_8: start_8 + hold_8_1 + 2 hold_8_2 + 3 hold_8_3
   + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7 + 8 hold_8_8
   + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 12 hold_8_12 <= 60
 hold_constraint_9: hold_9_1 + hold_9_2 + hold_9_3 + hold_9_4 + hold_9_5
   + hold_9_6 + hold_9_7 + hold_9_8 + hold_9_9 + hold_9_10 + hold_9_11
   + hold_9_12 <= 1
 end_constraint_9: start_9 + hold_9_1 + 2 hold_9_2 + 3 hold_9_3
   + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7 + 8 hold_9_8
   + 9 hold_9_9 + 10 hold_9_10 + 11 hold_9_11 + 12 hold_9_12 <= 60
 hold_constraint_10: hold_10_1 + hold_10_2 + hold_10_3 + hold_10_4
   + hold_10_5 + hold_10_6 + hold_10_7 + hold_10_8 + hold_10_9 + hold_10_10
   + hold_10_11 + hold_10_12 <= 1
 end_constraint_10: start_10 + hold_10_1 + 2 hold_10_2 + 3 hold_10_3
   + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7 + 8 hold_10_8
   + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11 + 12 hold_10_12 <= 60
 hold_constraint_11: hold_11_1 + hold_11_2 + hold_11_3 + hold_11_4
   + hold_11_5 + hold_11_6 + hold_11_7 + hold_11_8 + hold_11_9 + hold_11_10
   + hold_11_11 + hold_11_12 <= 1
 end_constraint_11: start_11 + hold_11_1 + 2 hold_11_2 + 3 hold_11_3
   + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7 + 8 hold_11_8
   + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11 + 12 hold_11_12 <= 60
 hold_constraint_12: hold_12_1 + hold_12_2 + hold_12_3 + hold_12_4
   + hold_12_5 + hold_12_6 + hold_12_7 + hold_12_8 + hold_12_9 + hold_12_10
   + hold_12_11 + hold_12_12 <= 1
 end_constraint_12: start_12 + hold_12_1 + 2 hold_12_2 + 3 hold_12_3
   + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6 + 7 hold_12_7 + 8 hold_12_8
   + 9 hold_12_9 + 10 hold_12_10 + 11 hold_12_11 + 12 hold_12_12 <= 60
 hold_constraint_13: hold_13_1 + hold_13_2 + hold_13_3 + hold_13_4
   + hold_13_5 + hold_13_6 + hold_13_7 + hold_13_8 + hold_13_9 + hold_13_10
   + hold_13_11 + hold_13_12 <= 1
 end_constraint_13: start_13 + hold_13_1 + 2 hold_13_2 + 3 hold_13_3
   + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6 + 7 hold_13_7 + 8 hold_13_8
   + 9 hold_13_9 + 10 hold_13_10 + 11 hold_13_11 + 12 hold_13_12 <= 60
 hold_constraint_14: hold_14_1 + hold_14_2 + hold_14_3 + hold_14_4
   + hold_14_5 + hold_14_6 + hold_14_7 + hold_14_8 + hold_14_9 + hold_14_10
   + hold_14_11 + hold_14_12 <= 1
 end_constraint_14: start_14 + hold_14_1 + 2 hold_14_2 + 3 hold_14_3
   + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6 + 7 hold_14_7 + 8 hold_14_8
   + 9 hold_14_9 + 10 hold_14_10 + 11 hold_14_11 + 12 hold_14_12 <= 60
 hold_constraint_15: hold_15_1 + hold_15_2 + hold_15_3 + hold_15_4
   + hold_15_5 + hold_15_6 + hold_15_7 + hold_15_8 + hold_15_9 + hold_15_10
   + hold_15_11 + hold_15_12 <= 1
 end_constraint_15: start_15 + hold_15_1 + 2 hold_15_2 + 3 hold_15_3
   + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6 + 7 hold_15_7 + 8 hold_15_8
   + 9 hold_15_9 + 10 hold_15_10 + 11 hold_15_11 + 12 hold_15_12 <= 60
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
   + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12
   - 60 is_end_complete_0 >= 0
 constr_edge_switch_end_max_0_OR: switch_nic_out_0_sink - is_end_complete_0
   <= 0
 constr_link_match_1_0_1: 60 switch_nic_out_0_nic_in_1 - group_start_1
   + group_end_0 <= 60
 constr_link_match_2_0_1: 60 switch_nic_out_0_nic_in_1 + group_start_1
   - group_end_0 <= 60
 constr_link_match_1_0_2: 60 switch_nic_out_0_nic_in_2 + group_end_0
   - group_start_2 <= 60
 constr_link_match_2_0_2: 60 switch_nic_out_0_nic_in_2 - group_end_0
   + group_start_2 <= 60
 constr_link_match_1_0_3: 60 switch_nic_out_0_nic_in_3 + group_end_0
   - group_start_3 <= 60
 constr_link_match_2_0_3: 60 switch_nic_out_0_nic_in_3 - group_end_0
   + group_start_3 <= 60
 constr_link_match_1_0_4: 60 switch_nic_out_0_nic_in_4 + group_end_0
   - group_start_4 <= 60
 constr_link_match_2_0_4: 60 switch_nic_out_0_nic_in_4 - group_end_0
   + group_start_4 <= 60
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
 constr_link_match_1_0_15: 60 switch_nic_out_0_nic_in_15 + group_end_0
   - group_start_15 <= 60
 constr_link_match_2_0_15: 60 switch_nic_out_0_nic_in_15 - group_end_0
   + group_start_15 <= 60
 constr_is_start_zero_1: start_1 + 60 is_start_zero_1_for_source_nic_in_1
   <= 60
 constr_edge_switch_start_zero_1_OR: switch_source_nic_in_1
   - is_start_zero_1_for_source_nic_in_1 <= 0
 constr_is_end_complete_1: start_1 + hold_1_1 + 2 hold_1_2 + 3 hold_1_3
   + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7 + 8 hold_1_8
   + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 + 12 hold_1_12
   - 60 is_end_complete_1 >= 0
 constr_edge_switch_end_max_1_OR: switch_nic_out_1_sink - is_end_complete_1
   <= 0
 constr_link_match_1_1_0: 60 switch_nic_out_1_nic_in_0 - group_start_0
   + group_end_1 <= 60
 constr_link_match_2_1_0: 60 switch_nic_out_1_nic_in_0 + group_start_0
   - group_end_1 <= 60
 constr_link_match_1_1_2: 60 switch_nic_out_1_nic_in_2 - group_start_2
   + group_end_1 <= 60
 constr_link_match_2_1_2: 60 switch_nic_out_1_nic_in_2 + group_start_2
   - group_end_1 <= 60
 constr_link_match_1_1_3: 60 switch_nic_out_1_nic_in_3 - group_start_3
   + group_end_1 <= 60
 constr_link_match_2_1_3: 60 switch_nic_out_1_nic_in_3 + group_start_3
   - group_end_1 <= 60
 constr_link_match_1_1_4: 60 switch_nic_out_1_nic_in_4 - group_start_4
   + group_end_1 <= 60
 constr_link_match_2_1_4: 60 switch_nic_out_1_nic_in_4 + group_start_4
   - group_end_1 <= 60
 constr_link_match_1_1_5: 60 switch_nic_out_1_nic_in_5 - group_start_5
   + group_end_1 <= 60
 constr_link_match_2_1_5: 60 switch_nic_out_1_nic_in_5 + group_start_5
   - group_end_1 <= 60
 constr_link_match_1_1_6: 60 switch_nic_out_1_nic_in_6 - group_start_6
   + group_end_1 <= 60
 constr_link_match_2_1_6: 60 switch_nic_out_1_nic_in_6 + group_start_6
   - group_end_1 <= 60
 constr_link_match_1_1_7: 60 switch_nic_out_1_nic_in_7 - group_start_7
   + group_end_1 <= 60
 constr_link_match_2_1_7: 60 switch_nic_out_1_nic_in_7 + group_start_7
   - group_end_1 <= 60
 constr_link_match_1_1_8: 60 switch_nic_out_1_nic_in_8 - group_start_8
   + group_end_1 <= 60
 constr_link_match_2_1_8: 60 switch_nic_out_1_nic_in_8 + group_start_8
   - group_end_1 <= 60
 constr_link_match_1_1_9: 60 switch_nic_out_1_nic_in_9 - group_start_9
   + group_end_1 <= 60
 constr_link_match_2_1_9: 60 switch_nic_out_1_nic_in_9 + group_start_9
   - group_end_1 <= 60
 constr_link_match_1_1_10: 60 switch_nic_out_1_nic_in_10 - group_start_10
   + group_end_1 <= 60
 constr_link_match_2_1_10: 60 switch_nic_out_1_nic_in_10 + group_start_10
   - group_end_1 <= 60
 constr_link_match_1_1_11: 60 switch_nic_out_1_nic_in_11 - group_start_11
   + group_end_1 <= 60
 constr_link_match_2_1_11: 60 switch_nic_out_1_nic_in_11 + group_start_11
   - group_end_1 <= 60
 constr_link_match_1_1_12: 60 switch_nic_out_1_nic_in_12 - group_start_12
   + group_end_1 <= 60
 constr_link_match_2_1_12: 60 switch_nic_out_1_nic_in_12 + group_start_12
   - group_end_1 <= 60
 constr_link_match_1_1_13: 60 switch_nic_out_1_nic_in_13 - group_start_13
   + group_end_1 <= 60
 constr_link_match_2_1_13: 60 switch_nic_out_1_nic_in_13 + group_start_13
   - group_end_1 <= 60
 constr_link_match_1_1_14: 60 switch_nic_out_1_nic_in_14 - group_start_14
   + group_end_1 <= 60
 constr_link_match_2_1_14: 60 switch_nic_out_1_nic_in_14 + group_start_14
   - group_end_1 <= 60
 constr_link_match_1_1_15: 60 switch_nic_out_1_nic_in_15 - group_start_15
   + group_end_1 <= 60
 constr_link_match_2_1_15: 60 switch_nic_out_1_nic_in_15 + group_start_15
   - group_end_1 <= 60
 constr_is_start_zero_2: start_2 + 60 is_start_zero_2_for_source_nic_in_2
   <= 60
 constr_edge_switch_start_zero_2_OR: switch_source_nic_in_2
   - is_start_zero_2_for_source_nic_in_2 <= 0
 constr_is_end_complete_2: start_2 + hold_2_1 + 2 hold_2_2 + 3 hold_2_3
   + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7 + 8 hold_2_8
   + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 12 hold_2_12
   - 60 is_end_complete_2 >= 0
 constr_edge_switch_end_max_2_OR: switch_nic_out_2_sink - is_end_complete_2
   <= 0
 constr_link_match_1_2_0: 60 switch_nic_out_2_nic_in_0 - group_start_0
   + group_end_2 <= 60
 constr_link_match_2_2_0: 60 switch_nic_out_2_nic_in_0 + group_start_0
   - group_end_2 <= 60
 constr_link_match_1_2_1: 60 switch_nic_out_2_nic_in_1 - group_start_1
   + group_end_2 <= 60
 constr_link_match_2_2_1: 60 switch_nic_out_2_nic_in_1 + group_start_1
   - group_end_2 <= 60
 constr_link_match_1_2_3: 60 switch_nic_out_2_nic_in_3 - group_start_3
   + group_end_2 <= 60
 constr_link_match_2_2_3: 60 switch_nic_out_2_nic_in_3 + group_start_3
   - group_end_2 <= 60
 constr_link_match_1_2_4: 60 switch_nic_out_2_nic_in_4 - group_start_4
   + group_end_2 <= 60
 constr_link_match_2_2_4: 60 switch_nic_out_2_nic_in_4 + group_start_4
   - group_end_2 <= 60
 constr_link_match_1_2_5: 60 switch_nic_out_2_nic_in_5 - group_start_5
   + group_end_2 <= 60
 constr_link_match_2_2_5: 60 switch_nic_out_2_nic_in_5 + group_start_5
   - group_end_2 <= 60
 constr_link_match_1_2_6: 60 switch_nic_out_2_nic_in_6 - group_start_6
   + group_end_2 <= 60
 constr_link_match_2_2_6: 60 switch_nic_out_2_nic_in_6 + group_start_6
   - group_end_2 <= 60
 constr_link_match_1_2_7: 60 switch_nic_out_2_nic_in_7 - group_start_7
   + group_end_2 <= 60
 constr_link_match_2_2_7: 60 switch_nic_out_2_nic_in_7 + group_start_7
   - group_end_2 <= 60
 constr_link_match_1_2_8: 60 switch_nic_out_2_nic_in_8 - group_start_8
   + group_end_2 <= 60
 constr_link_match_2_2_8: 60 switch_nic_out_2_nic_in_8 + group_start_8
   - group_end_2 <= 60
 constr_link_match_1_2_9: 60 switch_nic_out_2_nic_in_9 - group_start_9
   + group_end_2 <= 60
 constr_link_match_2_2_9: 60 switch_nic_out_2_nic_in_9 + group_start_9
   - group_end_2 <= 60
 constr_link_match_1_2_10: 60 switch_nic_out_2_nic_in_10 - group_start_10
   + group_end_2 <= 60
 constr_link_match_2_2_10: 60 switch_nic_out_2_nic_in_10 + group_start_10
   - group_end_2 <= 60
 constr_link_match_1_2_11: 60 switch_nic_out_2_nic_in_11 - group_start_11
   + group_end_2 <= 60
 constr_link_match_2_2_11: 60 switch_nic_out_2_nic_in_11 + group_start_11
   - group_end_2 <= 60
 constr_link_match_1_2_12: 60 switch_nic_out_2_nic_in_12 - group_start_12
   + group_end_2 <= 60
 constr_link_match_2_2_12: 60 switch_nic_out_2_nic_in_12 + group_start_12
   - group_end_2 <= 60
 constr_link_match_1_2_13: 60 switch_nic_out_2_nic_in_13 - group_start_13
   + group_end_2 <= 60
 constr_link_match_2_2_13: 60 switch_nic_out_2_nic_in_13 + group_start_13
   - group_end_2 <= 60
 constr_link_match_1_2_14: 60 switch_nic_out_2_nic_in_14 - group_start_14
   + group_end_2 <= 60
 constr_link_match_2_2_14: 60 switch_nic_out_2_nic_in_14 + group_start_14
   - group_end_2 <= 60
 constr_link_match_1_2_15: 60 switch_nic_out_2_nic_in_15 - group_start_15
   + group_end_2 <= 60
 constr_link_match_2_2_15: 60 switch_nic_out_2_nic_in_15 + group_start_15
   - group_end_2 <= 60
 constr_is_start_zero_3: start_3 + 60 is_start_zero_3_for_source_nic_in_3
   <= 60
 constr_edge_switch_start_zero_3_OR: switch_source_nic_in_3
   - is_start_zero_3_for_source_nic_in_3 <= 0
 constr_is_end_complete_3: start_3 + hold_3_1 + 2 hold_3_2 + 3 hold_3_3
   + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7 + 8 hold_3_8
   + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11 + 12 hold_3_12
   - 60 is_end_complete_3 >= 0
 constr_edge_switch_end_max_3_OR: switch_nic_out_3_sink - is_end_complete_3
   <= 0
 constr_link_match_1_3_0: 60 switch_nic_out_3_nic_in_0 - group_start_0
   + group_end_3 <= 60
 constr_link_match_2_3_0: 60 switch_nic_out_3_nic_in_0 + group_start_0
   - group_end_3 <= 60
 constr_link_match_1_3_1: 60 switch_nic_out_3_nic_in_1 - group_start_1
   + group_end_3 <= 60
 constr_link_match_2_3_1: 60 switch_nic_out_3_nic_in_1 + group_start_1
   - group_end_3 <= 60
 constr_link_match_1_3_2: 60 switch_nic_out_3_nic_in_2 - group_start_2
   + group_end_3 <= 60
 constr_link_match_2_3_2: 60 switch_nic_out_3_nic_in_2 + group_start_2
   - group_end_3 <= 60
 constr_link_match_1_3_4: 60 switch_nic_out_3_nic_in_4 - group_start_4
   + group_end_3 <= 60
 constr_link_match_2_3_4: 60 switch_nic_out_3_nic_in_4 + group_start_4
   - group_end_3 <= 60
 constr_link_match_1_3_5: 60 switch_nic_out_3_nic_in_5 - group_start_5
   + group_end_3 <= 60
 constr_link_match_2_3_5: 60 switch_nic_out_3_nic_in_5 + group_start_5
   - group_end_3 <= 60
 constr_link_match_1_3_6: 60 switch_nic_out_3_nic_in_6 - group_start_6
   + group_end_3 <= 60
 constr_link_match_2_3_6: 60 switch_nic_out_3_nic_in_6 + group_start_6
   - group_end_3 <= 60
 constr_link_match_1_3_7: 60 switch_nic_out_3_nic_in_7 - group_start_7
   + group_end_3 <= 60
 constr_link_match_2_3_7: 60 switch_nic_out_3_nic_in_7 + group_start_7
   - group_end_3 <= 60
 constr_link_match_1_3_8: 60 switch_nic_out_3_nic_in_8 - group_start_8
   + group_end_3 <= 60
 constr_link_match_2_3_8: 60 switch_nic_out_3_nic_in_8 + group_start_8
   - group_end_3 <= 60
 constr_link_match_1_3_9: 60 switch_nic_out_3_nic_in_9 - group_start_9
   + group_end_3 <= 60
 constr_link_match_2_3_9: 60 switch_nic_out_3_nic_in_9 + group_start_9
   - group_end_3 <= 60
 constr_link_match_1_3_10: 60 switch_nic_out_3_nic_in_10 - group_start_10
   + group_end_3 <= 60
 constr_link_match_2_3_10: 60 switch_nic_out_3_nic_in_10 + group_start_10
   - group_end_3 <= 60
 constr_link_match_1_3_11: 60 switch_nic_out_3_nic_in_11 - group_start_11
   + group_end_3 <= 60
 constr_link_match_2_3_11: 60 switch_nic_out_3_nic_in_11 + group_start_11
   - group_end_3 <= 60
 constr_link_match_1_3_12: 60 switch_nic_out_3_nic_in_12 - group_start_12
   + group_end_3 <= 60
 constr_link_match_2_3_12: 60 switch_nic_out_3_nic_in_12 + group_start_12
   - group_end_3 <= 60
 constr_link_match_1_3_13: 60 switch_nic_out_3_nic_in_13 - group_start_13
   + group_end_3 <= 60
 constr_link_match_2_3_13: 60 switch_nic_out_3_nic_in_13 + group_start_13
   - group_end_3 <= 60
 constr_link_match_1_3_14: 60 switch_nic_out_3_nic_in_14 - group_start_14
   + group_end_3 <= 60
 constr_link_match_2_3_14: 60 switch_nic_out_3_nic_in_14 + group_start_14
   - group_end_3 <= 60
 constr_link_match_1_3_15: 60 switch_nic_out_3_nic_in_15 - group_start_15
   + group_end_3 <= 60
 constr_link_match_2_3_15: 60 switch_nic_out_3_nic_in_15 + group_start_15
   - group_end_3 <= 60
 constr_is_start_zero_4: start_4 + 60 is_start_zero_4_for_source_nic_in_4
   <= 60
 constr_edge_switch_start_zero_4_OR: switch_source_nic_in_4
   - is_start_zero_4_for_source_nic_in_4 <= 0
 constr_is_end_complete_4: start_4 + hold_4_1 + 2 hold_4_2 + 3 hold_4_3
   + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7 + 8 hold_4_8
   + 9 hold_4_9 + 10 hold_4_10 + 11 hold_4_11 + 12 hold_4_12
   - 60 is_end_complete_4 >= 0
 constr_edge_switch_end_max_4_OR: switch_nic_out_4_sink - is_end_complete_4
   <= 0
 constr_link_match_1_4_0: 60 switch_nic_out_4_nic_in_0 - group_start_0
   + group_end_4 <= 60
 constr_link_match_2_4_0: 60 switch_nic_out_4_nic_in_0 + group_start_0
   - group_end_4 <= 60
 constr_link_match_1_4_1: 60 switch_nic_out_4_nic_in_1 - group_start_1
   + group_end_4 <= 60
 constr_link_match_2_4_1: 60 switch_nic_out_4_nic_in_1 + group_start_1
   - group_end_4 <= 60
 constr_link_match_1_4_2: 60 switch_nic_out_4_nic_in_2 - group_start_2
   + group_end_4 <= 60
 constr_link_match_2_4_2: 60 switch_nic_out_4_nic_in_2 + group_start_2
   - group_end_4 <= 60
 constr_link_match_1_4_3: 60 switch_nic_out_4_nic_in_3 - group_start_3
   + group_end_4 <= 60
 constr_link_match_2_4_3: 60 switch_nic_out_4_nic_in_3 + group_start_3
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
 constr_link_match_1_4_15: 60 switch_nic_out_4_nic_in_15 - group_start_15
   + group_end_4 <= 60
 constr_link_match_2_4_15: 60 switch_nic_out_4_nic_in_15 + group_start_15
   - group_end_4 <= 60
 constr_is_start_zero_5: start_5 + 60 is_start_zero_5_for_source_nic_in_5
   <= 60
 constr_edge_switch_start_zero_5_OR: switch_source_nic_in_5
   - is_start_zero_5_for_source_nic_in_5 <= 0
 constr_is_end_complete_5: start_5 + hold_5_1 + 2 hold_5_2 + 3 hold_5_3
   + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7 + 8 hold_5_8
   + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12
   - 60 is_end_complete_5 >= 0
 constr_edge_switch_end_max_5_OR: switch_nic_out_5_sink - is_end_complete_5
   <= 0
 constr_link_match_1_5_0: 60 switch_nic_out_5_nic_in_0 - group_start_0
   + group_end_5 <= 60
 constr_link_match_2_5_0: 60 switch_nic_out_5_nic_in_0 + group_start_0
   - group_end_5 <= 60
 constr_link_match_1_5_1: 60 switch_nic_out_5_nic_in_1 - group_start_1
   + group_end_5 <= 60
 constr_link_match_2_5_1: 60 switch_nic_out_5_nic_in_1 + group_start_1
   - group_end_5 <= 60
 constr_link_match_1_5_2: 60 switch_nic_out_5_nic_in_2 - group_start_2
   + group_end_5 <= 60
 constr_link_match_2_5_2: 60 switch_nic_out_5_nic_in_2 + group_start_2
   - group_end_5 <= 60
 constr_link_match_1_5_3: 60 switch_nic_out_5_nic_in_3 - group_start_3
   + group_end_5 <= 60
 constr_link_match_2_5_3: 60 switch_nic_out_5_nic_in_3 + group_start_3
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
 constr_link_match_1_5_15: 60 switch_nic_out_5_nic_in_15 - group_start_15
   + group_end_5 <= 60
 constr_link_match_2_5_15: 60 switch_nic_out_5_nic_in_15 + group_start_15
   - group_end_5 <= 60
 constr_is_start_zero_6: start_6 + 60 is_start_zero_6_for_source_nic_in_6
   <= 60
 constr_edge_switch_start_zero_6_OR: switch_source_nic_in_6
   - is_start_zero_6_for_source_nic_in_6 <= 0
 constr_is_end_complete_6: start_6 + hold_6_1 + 2 hold_6_2 + 3 hold_6_3
   + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7 + 8 hold_6_8
   + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 12 hold_6_12
   - 60 is_end_complete_6 >= 0
 constr_edge_switch_end_max_6_OR: switch_nic_out_6_sink - is_end_complete_6
   <= 0
 constr_link_match_1_6_0: 60 switch_nic_out_6_nic_in_0 - group_start_0
   + group_end_6 <= 60
 constr_link_match_2_6_0: 60 switch_nic_out_6_nic_in_0 + group_start_0
   - group_end_6 <= 60
 constr_link_match_1_6_1: 60 switch_nic_out_6_nic_in_1 - group_start_1
   + group_end_6 <= 60
 constr_link_match_2_6_1: 60 switch_nic_out_6_nic_in_1 + group_start_1
   - group_end_6 <= 60
 constr_link_match_1_6_2: 60 switch_nic_out_6_nic_in_2 - group_start_2
   + group_end_6 <= 60
 constr_link_match_2_6_2: 60 switch_nic_out_6_nic_in_2 + group_start_2
   - group_end_6 <= 60
 constr_link_match_1_6_3: 60 switch_nic_out_6_nic_in_3 - group_start_3
   + group_end_6 <= 60
 constr_link_match_2_6_3: 60 switch_nic_out_6_nic_in_3 + group_start_3
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
 constr_link_match_1_6_15: 60 switch_nic_out_6_nic_in_15 - group_start_15
   + group_end_6 <= 60
 constr_link_match_2_6_15: 60 switch_nic_out_6_nic_in_15 + group_start_15
   - group_end_6 <= 60
 constr_is_start_zero_7: start_7 + 60 is_start_zero_7_for_source_nic_in_7
   <= 60
 constr_edge_switch_start_zero_7_OR: switch_source_nic_in_7
   - is_start_zero_7_for_source_nic_in_7 <= 0
 constr_is_end_complete_7: start_7 + hold_7_1 + 2 hold_7_2 + 3 hold_7_3
   + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7 + 8 hold_7_8
   + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11 + 12 hold_7_12
   - 60 is_end_complete_7 >= 0
 constr_edge_switch_end_max_7_OR: switch_nic_out_7_sink - is_end_complete_7
   <= 0
 constr_link_match_1_7_0: 60 switch_nic_out_7_nic_in_0 - group_start_0
   + group_end_7 <= 60
 constr_link_match_2_7_0: 60 switch_nic_out_7_nic_in_0 + group_start_0
   - group_end_7 <= 60
 constr_link_match_1_7_1: 60 switch_nic_out_7_nic_in_1 - group_start_1
   + group_end_7 <= 60
 constr_link_match_2_7_1: 60 switch_nic_out_7_nic_in_1 + group_start_1
   - group_end_7 <= 60
 constr_link_match_1_7_2: 60 switch_nic_out_7_nic_in_2 - group_start_2
   + group_end_7 <= 60
 constr_link_match_2_7_2: 60 switch_nic_out_7_nic_in_2 + group_start_2
   - group_end_7 <= 60
 constr_link_match_1_7_3: 60 switch_nic_out_7_nic_in_3 - group_start_3
   + group_end_7 <= 60
 constr_link_match_2_7_3: 60 switch_nic_out_7_nic_in_3 + group_start_3
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
 constr_link_match_1_7_15: 60 switch_nic_out_7_nic_in_15 - group_start_15
   + group_end_7 <= 60
 constr_link_match_2_7_15: 60 switch_nic_out_7_nic_in_15 + group_start_15
   - group_end_7 <= 60
 constr_is_start_zero_8: start_8 + 60 is_start_zero_8_for_source_nic_in_8
   <= 60
 constr_edge_switch_start_zero_8_OR: switch_source_nic_in_8
   - is_start_zero_8_for_source_nic_in_8 <= 0
 constr_is_end_complete_8: start_8 + hold_8_1 + 2 hold_8_2 + 3 hold_8_3
   + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7 + 8 hold_8_8
   + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 12 hold_8_12
   - 60 is_end_complete_8 >= 0
 constr_edge_switch_end_max_8_OR: switch_nic_out_8_sink - is_end_complete_8
   <= 0
 constr_link_match_1_8_0: 60 switch_nic_out_8_nic_in_0 - group_start_0
   + group_end_8 <= 60
 constr_link_match_2_8_0: 60 switch_nic_out_8_nic_in_0 + group_start_0
   - group_end_8 <= 60
 constr_link_match_1_8_1: 60 switch_nic_out_8_nic_in_1 - group_start_1
   + group_end_8 <= 60
 constr_link_match_2_8_1: 60 switch_nic_out_8_nic_in_1 + group_start_1
   - group_end_8 <= 60
 constr_link_match_1_8_2: 60 switch_nic_out_8_nic_in_2 - group_start_2
   + group_end_8 <= 60
 constr_link_match_2_8_2: 60 switch_nic_out_8_nic_in_2 + group_start_2
   - group_end_8 <= 60
 constr_link_match_1_8_3: 60 switch_nic_out_8_nic_in_3 - group_start_3
   + group_end_8 <= 60
 constr_link_match_2_8_3: 60 switch_nic_out_8_nic_in_3 + group_start_3
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
 constr_link_match_1_8_15: 60 switch_nic_out_8_nic_in_15 - group_start_15
   + group_end_8 <= 60
 constr_link_match_2_8_15: 60 switch_nic_out_8_nic_in_15 + group_start_15
   - group_end_8 <= 60
 constr_is_start_zero_9: start_9 + 60 is_start_zero_9_for_source_nic_in_9
   <= 60
 constr_edge_switch_start_zero_9_OR: switch_source_nic_in_9
   - is_start_zero_9_for_source_nic_in_9 <= 0
 constr_is_end_complete_9: start_9 + hold_9_1 + 2 hold_9_2 + 3 hold_9_3
   + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7 + 8 hold_9_8
   + 9 hold_9_9 + 10 hold_9_10 + 11 hold_9_11 + 12 hold_9_12
   - 60 is_end_complete_9 >= 0
 constr_edge_switch_end_max_9_OR: switch_nic_out_9_sink - is_end_complete_9
   <= 0
 constr_link_match_1_9_0: 60 switch_nic_out_9_nic_in_0 - group_start_0
   + group_end_9 <= 60
 constr_link_match_2_9_0: 60 switch_nic_out_9_nic_in_0 + group_start_0
   - group_end_9 <= 60
 constr_link_match_1_9_1: 60 switch_nic_out_9_nic_in_1 - group_start_1
   + group_end_9 <= 60
 constr_link_match_2_9_1: 60 switch_nic_out_9_nic_in_1 + group_start_1
   - group_end_9 <= 60
 constr_link_match_1_9_2: 60 switch_nic_out_9_nic_in_2 - group_start_2
   + group_end_9 <= 60
 constr_link_match_2_9_2: 60 switch_nic_out_9_nic_in_2 + group_start_2
   - group_end_9 <= 60
 constr_link_match_1_9_3: 60 switch_nic_out_9_nic_in_3 - group_start_3
   + group_end_9 <= 60
 constr_link_match_2_9_3: 60 switch_nic_out_9_nic_in_3 + group_start_3
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
 constr_link_match_1_9_15: 60 switch_nic_out_9_nic_in_15 - group_start_15
   + group_end_9 <= 60
 constr_link_match_2_9_15: 60 switch_nic_out_9_nic_in_15 + group_start_15
   - group_end_9 <= 60
 constr_is_start_zero_10: start_10
   + 60 is_start_zero_10_for_source_nic_in_10 <= 60
 constr_edge_switch_start_zero_10_OR: switch_source_nic_in_10
   - is_start_zero_10_for_source_nic_in_10 <= 0
 constr_is_end_complete_10: start_10 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 12 hold_10_12 - 60 is_end_complete_10 >= 0
 constr_edge_switch_end_max_10_OR: switch_nic_out_10_sink
   - is_end_complete_10 <= 0
 constr_link_match_1_10_0: 60 switch_nic_out_10_nic_in_0 - group_start_0
   + group_end_10 <= 60
 constr_link_match_2_10_0: 60 switch_nic_out_10_nic_in_0 + group_start_0
   - group_end_10 <= 60
 constr_link_match_1_10_1: 60 switch_nic_out_10_nic_in_1 - group_start_1
   + group_end_10 <= 60
 constr_link_match_2_10_1: 60 switch_nic_out_10_nic_in_1 + group_start_1
   - group_end_10 <= 60
 constr_link_match_1_10_2: 60 switch_nic_out_10_nic_in_2 - group_start_2
   + group_end_10 <= 60
 constr_link_match_2_10_2: 60 switch_nic_out_10_nic_in_2 + group_start_2
   - group_end_10 <= 60
 constr_link_match_1_10_3: 60 switch_nic_out_10_nic_in_3 - group_start_3
   + group_end_10 <= 60
 constr_link_match_2_10_3: 60 switch_nic_out_10_nic_in_3 + group_start_3
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
 constr_link_match_1_10_15: 60 switch_nic_out_10_nic_in_15 - group_start_15
   + group_end_10 <= 60
 constr_link_match_2_10_15: 60 switch_nic_out_10_nic_in_15 + group_start_15
   - group_end_10 <= 60
 constr_is_start_zero_11: start_11
   + 60 is_start_zero_11_for_source_nic_in_11 <= 60
 constr_edge_switch_start_zero_11_OR: switch_source_nic_in_11
   - is_start_zero_11_for_source_nic_in_11 <= 0
 constr_is_end_complete_11: start_11 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 12 hold_11_12 - 60 is_end_complete_11 >= 0
 constr_edge_switch_end_max_11_OR: switch_nic_out_11_sink
   - is_end_complete_11 <= 0
 constr_link_match_1_11_0: 60 switch_nic_out_11_nic_in_0 - group_start_0
   + group_end_11 <= 60
 constr_link_match_2_11_0: 60 switch_nic_out_11_nic_in_0 + group_start_0
   - group_end_11 <= 60
 constr_link_match_1_11_1: 60 switch_nic_out_11_nic_in_1 - group_start_1
   + group_end_11 <= 60
 constr_link_match_2_11_1: 60 switch_nic_out_11_nic_in_1 + group_start_1
   - group_end_11 <= 60
 constr_link_match_1_11_2: 60 switch_nic_out_11_nic_in_2 - group_start_2
   + group_end_11 <= 60
 constr_link_match_2_11_2: 60 switch_nic_out_11_nic_in_2 + group_start_2
   - group_end_11 <= 60
 constr_link_match_1_11_3: 60 switch_nic_out_11_nic_in_3 - group_start_3
   + group_end_11 <= 60
 constr_link_match_2_11_3: 60 switch_nic_out_11_nic_in_3 + group_start_3
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
 constr_link_match_1_11_15: 60 switch_nic_out_11_nic_in_15 - group_start_15
   + group_end_11 <= 60
 constr_link_match_2_11_15: 60 switch_nic_out_11_nic_in_15 + group_start_15
   - group_end_11 <= 60
 constr_is_start_zero_12: start_12
   + 60 is_start_zero_12_for_source_nic_in_12 <= 60
 constr_edge_switch_start_zero_12_OR: switch_source_nic_in_12
   - is_start_zero_12_for_source_nic_in_12 <= 0
 constr_is_end_complete_12: start_12 + hold_12_1 + 2 hold_12_2
   + 3 hold_12_3 + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6 + 7 hold_12_7
   + 8 hold_12_8 + 9 hold_12_9 + 10 hold_12_10 + 11 hold_12_11
   + 12 hold_12_12 - 60 is_end_complete_12 >= 0
 constr_edge_switch_end_max_12_OR: switch_nic_out_12_sink
   - is_end_complete_12 <= 0
 constr_link_match_1_12_0: 60 switch_nic_out_12_nic_in_0 - group_start_0
   + group_end_12 <= 60
 constr_link_match_2_12_0: 60 switch_nic_out_12_nic_in_0 + group_start_0
   - group_end_12 <= 60
 constr_link_match_1_12_1: 60 switch_nic_out_12_nic_in_1 - group_start_1
   + group_end_12 <= 60
 constr_link_match_2_12_1: 60 switch_nic_out_12_nic_in_1 + group_start_1
   - group_end_12 <= 60
 constr_link_match_1_12_2: 60 switch_nic_out_12_nic_in_2 - group_start_2
   + group_end_12 <= 60
 constr_link_match_2_12_2: 60 switch_nic_out_12_nic_in_2 + group_start_2
   - group_end_12 <= 60
 constr_link_match_1_12_3: 60 switch_nic_out_12_nic_in_3 - group_start_3
   + group_end_12 <= 60
 constr_link_match_2_12_3: 60 switch_nic_out_12_nic_in_3 + group_start_3
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
 constr_link_match_1_12_15: 60 switch_nic_out_12_nic_in_15 - group_start_15
   + group_end_12 <= 60
 constr_link_match_2_12_15: 60 switch_nic_out_12_nic_in_15 + group_start_15
   - group_end_12 <= 60
 constr_is_start_zero_13: start_13
   + 60 is_start_zero_13_for_source_nic_in_13 <= 60
 constr_edge_switch_start_zero_13_OR: switch_source_nic_in_13
   - is_start_zero_13_for_source_nic_in_13 <= 0
 constr_is_end_complete_13: start_13 + hold_13_1 + 2 hold_13_2
   + 3 hold_13_3 + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6 + 7 hold_13_7
   + 8 hold_13_8 + 9 hold_13_9 + 10 hold_13_10 + 11 hold_13_11
   + 12 hold_13_12 - 60 is_end_complete_13 >= 0
 constr_edge_switch_end_max_13_OR: switch_nic_out_13_sink
   - is_end_complete_13 <= 0
 constr_link_match_1_13_0: 60 switch_nic_out_13_nic_in_0 - group_start_0
   + group_end_13 <= 60
 constr_link_match_2_13_0: 60 switch_nic_out_13_nic_in_0 + group_start_0
   - group_end_13 <= 60
 constr_link_match_1_13_1: 60 switch_nic_out_13_nic_in_1 - group_start_1
   + group_end_13 <= 60
 constr_link_match_2_13_1: 60 switch_nic_out_13_nic_in_1 + group_start_1
   - group_end_13 <= 60
 constr_link_match_1_13_2: 60 switch_nic_out_13_nic_in_2 - group_start_2
   + group_end_13 <= 60
 constr_link_match_2_13_2: 60 switch_nic_out_13_nic_in_2 + group_start_2
   - group_end_13 <= 60
 constr_link_match_1_13_3: 60 switch_nic_out_13_nic_in_3 - group_start_3
   + group_end_13 <= 60
 constr_link_match_2_13_3: 60 switch_nic_out_13_nic_in_3 + group_start_3
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
 constr_link_match_1_13_15: 60 switch_nic_out_13_nic_in_15 - group_start_15
   + group_end_13 <= 60
 constr_link_match_2_13_15: 60 switch_nic_out_13_nic_in_15 + group_start_15
   - group_end_13 <= 60
 constr_is_start_zero_14: start_14
   + 60 is_start_zero_14_for_source_nic_in_14 <= 60
 constr_edge_switch_start_zero_14_OR: switch_source_nic_in_14
   - is_start_zero_14_for_source_nic_in_14 <= 0
 constr_is_end_complete_14: start_14 + hold_14_1 + 2 hold_14_2
   + 3 hold_14_3 + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6 + 7 hold_14_7
   + 8 hold_14_8 + 9 hold_14_9 + 10 hold_14_10 + 11 hold_14_11
   + 12 hold_14_12 - 60 is_end_complete_14 >= 0
 constr_edge_switch_end_max_14_OR: switch_nic_out_14_sink
   - is_end_complete_14 <= 0
 constr_link_match_1_14_0: 60 switch_nic_out_14_nic_in_0 - group_start_0
   + group_end_14 <= 60
 constr_link_match_2_14_0: 60 switch_nic_out_14_nic_in_0 + group_start_0
   - group_end_14 <= 60
 constr_link_match_1_14_1: 60 switch_nic_out_14_nic_in_1 - group_start_1
   + group_end_14 <= 60
 constr_link_match_2_14_1: 60 switch_nic_out_14_nic_in_1 + group_start_1
   - group_end_14 <= 60
 constr_link_match_1_14_2: 60 switch_nic_out_14_nic_in_2 - group_start_2
   + group_end_14 <= 60
 constr_link_match_2_14_2: 60 switch_nic_out_14_nic_in_2 + group_start_2
   - group_end_14 <= 60
 constr_link_match_1_14_3: 60 switch_nic_out_14_nic_in_3 - group_start_3
   + group_end_14 <= 60
 constr_link_match_2_14_3: 60 switch_nic_out_14_nic_in_3 + group_start_3
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
 constr_link_match_1_14_15: 60 switch_nic_out_14_nic_in_15 - group_start_15
   + group_end_14 <= 60
 constr_link_match_2_14_15: 60 switch_nic_out_14_nic_in_15 + group_start_15
   - group_end_14 <= 60
 constr_is_start_zero_15: start_15
   + 60 is_start_zero_15_for_source_nic_in_15 <= 60
 constr_edge_switch_start_zero_15_OR: switch_source_nic_in_15
   - is_start_zero_15_for_source_nic_in_15 <= 0
 constr_is_end_complete_15: start_15 + hold_15_1 + 2 hold_15_2
   + 3 hold_15_3 + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6 + 7 hold_15_7
   + 8 hold_15_8 + 9 hold_15_9 + 10 hold_15_10 + 11 hold_15_11
   + 12 hold_15_12 - 60 is_end_complete_15 >= 0
 constr_edge_switch_end_max_15_OR: switch_nic_out_15_sink
   - is_end_complete_15 <= 0
 constr_link_match_1_15_0: 60 switch_nic_out_15_nic_in_0 - group_start_0
   + group_end_15 <= 60
 constr_link_match_2_15_0: 60 switch_nic_out_15_nic_in_0 + group_start_0
   - group_end_15 <= 60
 constr_link_match_1_15_1: 60 switch_nic_out_15_nic_in_1 - group_start_1
   + group_end_15 <= 60
 constr_link_match_2_15_1: 60 switch_nic_out_15_nic_in_1 + group_start_1
   - group_end_15 <= 60
 constr_link_match_1_15_2: 60 switch_nic_out_15_nic_in_2 - group_start_2
   + group_end_15 <= 60
 constr_link_match_2_15_2: 60 switch_nic_out_15_nic_in_2 + group_start_2
   - group_end_15 <= 60
 constr_link_match_1_15_3: 60 switch_nic_out_15_nic_in_3 - group_start_3
   + group_end_15 <= 60
 constr_link_match_2_15_3: 60 switch_nic_out_15_nic_in_3 + group_start_3
   - group_end_15 <= 60
 constr_link_match_1_15_4: 60 switch_nic_out_15_nic_in_4 - group_start_4
   + group_end_15 <= 60
 constr_link_match_2_15_4: 60 switch_nic_out_15_nic_in_4 + group_start_4
   - group_end_15 <= 60
 constr_link_match_1_15_5: 60 switch_nic_out_15_nic_in_5 - group_start_5
   + group_end_15 <= 60
 constr_link_match_2_15_5: 60 switch_nic_out_15_nic_in_5 + group_start_5
   - group_end_15 <= 60
 constr_link_match_1_15_6: 60 switch_nic_out_15_nic_in_6 - group_start_6
   + group_end_15 <= 60
 constr_link_match_2_15_6: 60 switch_nic_out_15_nic_in_6 + group_start_6
   - group_end_15 <= 60
 constr_link_match_1_15_7: 60 switch_nic_out_15_nic_in_7 - group_start_7
   + group_end_15 <= 60
 constr_link_match_2_15_7: 60 switch_nic_out_15_nic_in_7 + group_start_7
   - group_end_15 <= 60
 constr_link_match_1_15_8: 60 switch_nic_out_15_nic_in_8 - group_start_8
   + group_end_15 <= 60
 constr_link_match_2_15_8: 60 switch_nic_out_15_nic_in_8 + group_start_8
   - group_end_15 <= 60
 constr_link_match_1_15_9: 60 switch_nic_out_15_nic_in_9 - group_start_9
   + group_end_15 <= 60
 constr_link_match_2_15_9: 60 switch_nic_out_15_nic_in_9 + group_start_9
   - group_end_15 <= 60
 constr_link_match_1_15_10: 60 switch_nic_out_15_nic_in_10 - group_start_10
   + group_end_15 <= 60
 constr_link_match_2_15_10: 60 switch_nic_out_15_nic_in_10 + group_start_10
   - group_end_15 <= 60
 constr_link_match_1_15_11: 60 switch_nic_out_15_nic_in_11 - group_start_11
   + group_end_15 <= 60
 constr_link_match_2_15_11: 60 switch_nic_out_15_nic_in_11 + group_start_11
   - group_end_15 <= 60
 constr_link_match_1_15_12: 60 switch_nic_out_15_nic_in_12 - group_start_12
   + group_end_15 <= 60
 constr_link_match_2_15_12: 60 switch_nic_out_15_nic_in_12 + group_start_12
   - group_end_15 <= 60
 constr_link_match_1_15_13: 60 switch_nic_out_15_nic_in_13 - group_start_13
   + group_end_15 <= 60
 constr_link_match_2_15_13: 60 switch_nic_out_15_nic_in_13 + group_start_13
   - group_end_15 <= 60
 constr_link_match_1_15_14: 60 switch_nic_out_15_nic_in_14 - group_start_14
   + group_end_15 <= 60
 constr_link_match_2_15_14: 60 switch_nic_out_15_nic_in_14 + group_start_14
   - group_end_15 <= 60
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
 node_throughput_constraint_0: group_tp_active_0 = 0 -> - 7464 hold_0_1
   - 3732 hold_0_2 - 2488 hold_0_3 - 1866 hold_0_4 - 1492.8 hold_0_5
   - 1244 hold_0_6 - 1066.285714285714 hold_0_7 - 933 hold_0_8
   - 829.3333333333333 hold_0_9 - 746.4 hold_0_10
   - 678.5454545454545 hold_0_11 - 622 hold_0_12 + flow_nic_in_0_0 <= 0
 node_throughput_constraint_1: group_tp_active_1 = 0 -> - 7464 hold_1_1
   - 3732 hold_1_2 - 2488 hold_1_3 - 1866 hold_1_4 - 1492.8 hold_1_5
   - 1244 hold_1_6 - 1066.285714285714 hold_1_7 - 933 hold_1_8
   - 829.3333333333333 hold_1_9 - 746.4 hold_1_10
   - 678.5454545454545 hold_1_11 - 622 hold_1_12 + flow_nic_in_1_1 <= 0
 node_throughput_constraint_2: group_tp_active_2 = 0 -> - 7464 hold_2_1
   - 3732 hold_2_2 - 2488 hold_2_3 - 1866 hold_2_4 - 1492.8 hold_2_5
   - 1244 hold_2_6 - 1066.285714285714 hold_2_7 - 933 hold_2_8
   - 829.3333333333333 hold_2_9 - 746.4 hold_2_10
   - 678.5454545454545 hold_2_11 - 622 hold_2_12 + flow_nic_in_2_2 <= 0
 node_throughput_constraint_3: group_tp_active_3 = 0 -> - 7464 hold_3_1
   - 3732 hold_3_2 - 2488 hold_3_3 - 1866 hold_3_4 - 1492.8 hold_3_5
   - 1244 hold_3_6 - 1066.285714285714 hold_3_7 - 933 hold_3_8
   - 829.3333333333333 hold_3_9 - 746.4 hold_3_10
   - 678.5454545454545 hold_3_11 - 622 hold_3_12 + flow_nic_in_3_3 <= 0
 node_throughput_constraint_4: group_tp_active_4 = 0 -> - 7464 hold_4_1
   - 3732 hold_4_2 - 2488 hold_4_3 - 1866 hold_4_4 - 1492.8 hold_4_5
   - 1244 hold_4_6 - 1066.285714285714 hold_4_7 - 933 hold_4_8
   - 829.3333333333333 hold_4_9 - 746.4 hold_4_10
   - 678.5454545454545 hold_4_11 - 622 hold_4_12 + flow_nic_in_4_4 <= 0
 node_throughput_constraint_5: group_tp_active_5 = 0 -> - 7464 hold_5_1
   - 3732 hold_5_2 - 2488 hold_5_3 - 1866 hold_5_4 - 1492.8 hold_5_5
   - 1244 hold_5_6 - 1066.285714285714 hold_5_7 - 933 hold_5_8
   - 829.3333333333333 hold_5_9 - 746.4 hold_5_10
   - 678.5454545454545 hold_5_11 - 622 hold_5_12 + flow_nic_in_5_5 <= 0
 node_throughput_constraint_6: group_tp_active_6 = 0 -> - 7464 hold_6_1
   - 3732 hold_6_2 - 2488 hold_6_3 - 1866 hold_6_4 - 1492.8 hold_6_5
   - 1244 hold_6_6 - 1066.285714285714 hold_6_7 - 933 hold_6_8
   - 829.3333333333333 hold_6_9 - 746.4 hold_6_10
   - 678.5454545454545 hold_6_11 - 622 hold_6_12 + flow_nic_in_6_6 <= 0
 node_throughput_constraint_7: group_tp_active_7 = 0 -> - 7464 hold_7_1
   - 3732 hold_7_2 - 2488 hold_7_3 - 1866 hold_7_4 - 1492.8 hold_7_5
   - 1244 hold_7_6 - 1066.285714285714 hold_7_7 - 933 hold_7_8
   - 829.3333333333333 hold_7_9 - 746.4 hold_7_10
   - 678.5454545454545 hold_7_11 - 622 hold_7_12 + flow_nic_in_7_7 <= 0
 node_throughput_constraint_8: group_tp_active_8 = 0 -> - 7464 hold_8_1
   - 3732 hold_8_2 - 2488 hold_8_3 - 1866 hold_8_4 - 1492.8 hold_8_5
   - 1244 hold_8_6 - 1066.285714285714 hold_8_7 - 933 hold_8_8
   - 829.3333333333333 hold_8_9 - 746.4 hold_8_10
   - 678.5454545454545 hold_8_11 - 622 hold_8_12 + flow_nic_in_8_8 <= 0
 node_throughput_constraint_9: group_tp_active_9 = 0 -> - 7464 hold_9_1
   - 3732 hold_9_2 - 2488 hold_9_3 - 1866 hold_9_4 - 1492.8 hold_9_5
   - 1244 hold_9_6 - 1066.285714285714 hold_9_7 - 933 hold_9_8
   - 829.3333333333333 hold_9_9 - 746.4 hold_9_10
   - 678.5454545454545 hold_9_11 - 622 hold_9_12 + flow_nic_in_9_9 <= 0
 node_throughput_constraint_10: group_tp_active_10 = 0 -> - 7464 hold_10_1
   - 3732 hold_10_2 - 2488 hold_10_3 - 1866 hold_10_4 - 1492.8 hold_10_5
   - 1244 hold_10_6 - 1066.285714285714 hold_10_7 - 933 hold_10_8
   - 829.3333333333333 hold_10_9 - 746.4 hold_10_10
   - 678.5454545454545 hold_10_11 - 622 hold_10_12 + flow_nic_in_10_10
   <= 0
 node_throughput_constraint_11: group_tp_active_11 = 0 -> - 7464 hold_11_1
   - 3732 hold_11_2 - 2488 hold_11_3 - 1866 hold_11_4 - 1492.8 hold_11_5
   - 1244 hold_11_6 - 1066.285714285714 hold_11_7 - 933 hold_11_8
   - 829.3333333333333 hold_11_9 - 746.4 hold_11_10
   - 678.5454545454545 hold_11_11 - 622 hold_11_12 + flow_nic_in_11_11
   <= 0
 node_throughput_constraint_12: group_tp_active_12 = 0 -> - 7464 hold_12_1
   - 3732 hold_12_2 - 2488 hold_12_3 - 1866 hold_12_4 - 1492.8 hold_12_5
   - 1244 hold_12_6 - 1066.285714285714 hold_12_7 - 933 hold_12_8
   - 829.3333333333333 hold_12_9 - 746.4 hold_12_10
   - 678.5454545454545 hold_12_11 - 622 hold_12_12 + flow_nic_in_12_12
   <= 0
 node_throughput_constraint_13: group_tp_active_13 = 0 -> - 7464 hold_13_1
   - 3732 hold_13_2 - 2488 hold_13_3 - 1866 hold_13_4 - 1492.8 hold_13_5
   - 1244 hold_13_6 - 1066.285714285714 hold_13_7 - 933 hold_13_8
   - 829.3333333333333 hold_13_9 - 746.4 hold_13_10
   - 678.5454545454545 hold_13_11 - 622 hold_13_12 + flow_nic_in_13_13
   <= 0
 node_throughput_constraint_14: group_tp_active_14 = 0 -> - 7464 hold_14_1
   - 3732 hold_14_2 - 2488 hold_14_3 - 1866 hold_14_4 - 1492.8 hold_14_5
   - 1244 hold_14_6 - 1066.285714285714 hold_14_7 - 933 hold_14_8
   - 829.3333333333333 hold_14_9 - 746.4 hold_14_10
   - 678.5454545454545 hold_14_11 - 622 hold_14_12 + flow_nic_in_14_14
   <= 0
 node_throughput_constraint_15: group_tp_active_15 = 0 -> - 7464 hold_15_1
   - 3732 hold_15_2 - 2488 hold_15_3 - 1866 hold_15_4 - 1492.8 hold_15_5
   - 1244 hold_15_6 - 1066.285714285714 hold_15_7 - 933 hold_15_8
   - 829.3333333333333 hold_15_9 - 746.4 hold_15_10
   - 678.5454545454545 hold_15_11 - 622 hold_15_12 + flow_nic_in_15_15
   <= 0
 constr_intergroup_throughput_0_source_nic_in_0: switch_source_nic_in_0 = 1 ->
   - 6827.037409677124 hold_0_1 - 3565.661873596714 hold_0_2
   - 2412.957036174959 hold_0_3 - 1823.467617814477 hold_0_4
   - 1465.454616849587 hold_0_5 - 1224.951996455123 hold_0_6
   - 1052.260583931315 hold_0_7 - 922.2443255532355 hold_0_8
   - 820.8241232555838 hold_0_9 - 739.5004607016537 hold_0_10
   - 672.8385602120198 hold_0_11 - 617.20126020193 hold_0_12
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_0_nic_in_1: switch_nic_out_0_nic_in_1 = 1 ->
   - 5089.669280600068 hold_1_1 - 3026.150415568619 hold_1_2
   - 2153.180441367374 hold_1_3 - 1671.107130863092 hold_1_4
   - 1365.407481935425 hold_1_5 - 1154.256553003943 hold_1_6
   - 999.665171097569 hold_1_7 - 881.5921573259316 hold_1_8
   - 788.4645856441134 hold_1_9 - 713.1323747193427 hold_1_10
   - 650.9396938908995 hold_1_11 - 598.7245818794369 hold_1_12
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_0_nic_in_2: switch_nic_out_0_nic_in_2 = 1 ->
   - 5089.669280600068 hold_2_1 - 3026.150415568619 hold_2_2
   - 2153.180441367374 hold_2_3 - 1671.107130863092 hold_2_4
   - 1365.407481935425 hold_2_5 - 1154.256553003943 hold_2_6
   - 999.665171097569 hold_2_7 - 881.5921573259316 hold_2_8
   - 788.4645856441134 hold_2_9 - 713.1323747193427 hold_2_10
   - 650.9396938908995 hold_2_11 - 598.7245818794369 hold_2_12
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_0_nic_in_3: switch_nic_out_0_nic_in_3 = 1 ->
   - 5089.669280600068 hold_3_1 - 3026.150415568619 hold_3_2
   - 2153.180441367374 hold_3_3 - 1671.107130863092 hold_3_4
   - 1365.407481935425 hold_3_5 - 1154.256553003943 hold_3_6
   - 999.665171097569 hold_3_7 - 881.5921573259316 hold_3_8
   - 788.4645856441134 hold_3_9 - 713.1323747193427 hold_3_10
   - 650.9396938908995 hold_3_11 - 598.7245818794369 hold_3_12
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_0_nic_in_4: switch_nic_out_0_nic_in_4 = 1 ->
   - 5089.669280600068 hold_4_1 - 3026.150415568619 hold_4_2
   - 2153.180441367374 hold_4_3 - 1671.107130863092 hold_4_4
   - 1365.407481935425 hold_4_5 - 1154.256553003943 hold_4_6
   - 999.665171097569 hold_4_7 - 881.5921573259316 hold_4_8
   - 788.4645856441134 hold_4_9 - 713.1323747193427 hold_4_10
   - 650.9396938908995 hold_4_11 - 598.7245818794369 hold_4_12
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_0_nic_in_5: switch_nic_out_0_nic_in_5 = 1 ->
   - 5089.669280600068 hold_5_1 - 3026.150415568619 hold_5_2
   - 2153.180441367374 hold_5_3 - 1671.107130863092 hold_5_4
   - 1365.407481935425 hold_5_5 - 1154.256553003943 hold_5_6
   - 999.665171097569 hold_5_7 - 881.5921573259316 hold_5_8
   - 788.4645856441134 hold_5_9 - 713.1323747193427 hold_5_10
   - 650.9396938908995 hold_5_11 - 598.7245818794369 hold_5_12
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_0_nic_in_6: switch_nic_out_0_nic_in_6 = 1 ->
   - 5089.669280600068 hold_6_1 - 3026.150415568619 hold_6_2
   - 2153.180441367374 hold_6_3 - 1671.107130863092 hold_6_4
   - 1365.407481935425 hold_6_5 - 1154.256553003943 hold_6_6
   - 999.665171097569 hold_6_7 - 881.5921573259316 hold_6_8
   - 788.4645856441134 hold_6_9 - 713.1323747193427 hold_6_10
   - 650.9396938908995 hold_6_11 - 598.7245818794369 hold_6_12
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_0_nic_in_7: switch_nic_out_0_nic_in_7 = 1 ->
   - 5089.669280600068 hold_7_1 - 3026.150415568619 hold_7_2
   - 2153.180441367374 hold_7_3 - 1671.107130863092 hold_7_4
   - 1365.407481935425 hold_7_5 - 1154.256553003943 hold_7_6
   - 999.665171097569 hold_7_7 - 881.5921573259316 hold_7_8
   - 788.4645856441134 hold_7_9 - 713.1323747193427 hold_7_10
   - 650.9396938908995 hold_7_11 - 598.7245818794369 hold_7_12
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_0_nic_in_8: switch_nic_out_0_nic_in_8 = 1 ->
   - 5089.669280600068 hold_8_1 - 3026.150415568619 hold_8_2
   - 2153.180441367374 hold_8_3 - 1671.107130863092 hold_8_4
   - 1365.407481935425 hold_8_5 - 1154.256553003943 hold_8_6
   - 999.665171097569 hold_8_7 - 881.5921573259316 hold_8_8
   - 788.4645856441134 hold_8_9 - 713.1323747193427 hold_8_10
   - 650.9396938908995 hold_8_11 - 598.7245818794369 hold_8_12
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_0_nic_in_9: switch_nic_out_0_nic_in_9 = 1 ->
   - 5089.669280600068 hold_9_1 - 3026.150415568619 hold_9_2
   - 2153.180441367374 hold_9_3 - 1671.107130863092 hold_9_4
   - 1365.407481935425 hold_9_5 - 1154.256553003943 hold_9_6
   - 999.665171097569 hold_9_7 - 881.5921573259316 hold_9_8
   - 788.4645856441134 hold_9_9 - 713.1323747193427 hold_9_10
   - 650.9396938908995 hold_9_11 - 598.7245818794369 hold_9_12
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_0_nic_in_10: switch_nic_out_0_nic_in_10 = 1 ->
   - 5089.669280600068 hold_10_1 - 3026.150415568619 hold_10_2
   - 2153.180441367374 hold_10_3 - 1671.107130863092 hold_10_4
   - 1365.407481935425 hold_10_5 - 1154.256553003943 hold_10_6
   - 999.665171097569 hold_10_7 - 881.5921573259316 hold_10_8
   - 788.4645856441134 hold_10_9 - 713.1323747193427 hold_10_10
   - 650.9396938908995 hold_10_11 - 598.7245818794369 hold_10_12
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_0_nic_in_11: switch_nic_out_0_nic_in_11 = 1 ->
   - 5089.669280600068 hold_11_1 - 3026.150415568619 hold_11_2
   - 2153.180441367374 hold_11_3 - 1671.107130863092 hold_11_4
   - 1365.407481935425 hold_11_5 - 1154.256553003943 hold_11_6
   - 999.665171097569 hold_11_7 - 881.5921573259316 hold_11_8
   - 788.4645856441134 hold_11_9 - 713.1323747193427 hold_11_10
   - 650.9396938908995 hold_11_11 - 598.7245818794369 hold_11_12
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_0_nic_in_12: switch_nic_out_0_nic_in_12 = 1 ->
   - 5089.669280600068 hold_12_1 - 3026.150415568619 hold_12_2
   - 2153.180441367374 hold_12_3 - 1671.107130863092 hold_12_4
   - 1365.407481935425 hold_12_5 - 1154.256553003943 hold_12_6
   - 999.665171097569 hold_12_7 - 881.5921573259316 hold_12_8
   - 788.4645856441134 hold_12_9 - 713.1323747193427 hold_12_10
   - 650.9396938908995 hold_12_11 - 598.7245818794369 hold_12_12
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_0_nic_in_13: switch_nic_out_0_nic_in_13 = 1 ->
   - 5089.669280600068 hold_13_1 - 3026.150415568619 hold_13_2
   - 2153.180441367374 hold_13_3 - 1671.107130863092 hold_13_4
   - 1365.407481935425 hold_13_5 - 1154.256553003943 hold_13_6
   - 999.665171097569 hold_13_7 - 881.5921573259316 hold_13_8
   - 788.4645856441134 hold_13_9 - 713.1323747193427 hold_13_10
   - 650.9396938908995 hold_13_11 - 598.7245818794369 hold_13_12
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_0_nic_in_14: switch_nic_out_0_nic_in_14 = 1 ->
   - 5089.669280600068 hold_14_1 - 3026.150415568619 hold_14_2
   - 2153.180441367374 hold_14_3 - 1671.107130863092 hold_14_4
   - 1365.407481935425 hold_14_5 - 1154.256553003943 hold_14_6
   - 999.665171097569 hold_14_7 - 881.5921573259316 hold_14_8
   - 788.4645856441134 hold_14_9 - 713.1323747193427 hold_14_10
   - 650.9396938908995 hold_14_11 - 598.7245818794369 hold_14_12
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_0_nic_in_15: switch_nic_out_0_nic_in_15 = 1 ->
   - 5089.669280600068 hold_15_1 - 3026.150415568619 hold_15_2
   - 2153.180441367374 hold_15_3 - 1671.107130863092 hold_15_4
   - 1365.407481935425 hold_15_5 - 1154.256553003943 hold_15_6
   - 999.665171097569 hold_15_7 - 881.5921573259316 hold_15_8
   - 788.4645856441134 hold_15_9 - 713.1323747193427 hold_15_10
   - 650.9396938908995 hold_15_11 - 598.7245818794369 hold_15_12
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_1_source_nic_in_1: switch_source_nic_in_1 = 1 ->
   - 6827.037409677124 hold_1_1 - 3565.661873596714 hold_1_2
   - 2412.957036174959 hold_1_3 - 1823.467617814477 hold_1_4
   - 1465.454616849587 hold_1_5 - 1224.951996455123 hold_1_6
   - 1052.260583931315 hold_1_7 - 922.2443255532355 hold_1_8
   - 820.8241232555838 hold_1_9 - 739.5004607016537 hold_1_10
   - 672.8385602120198 hold_1_11 - 617.20126020193 hold_1_12
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_0_nic_out_1_nic_in_0: switch_nic_out_1_nic_in_0 = 1 ->
   - 5089.669280600068 hold_0_1 - 3026.150415568619 hold_0_2
   - 2153.180441367374 hold_0_3 - 1671.107130863092 hold_0_4
   - 1365.407481935425 hold_0_5 - 1154.256553003943 hold_0_6
   - 999.665171097569 hold_0_7 - 881.5921573259316 hold_0_8
   - 788.4645856441134 hold_0_9 - 713.1323747193427 hold_0_10
   - 650.9396938908995 hold_0_11 - 598.7245818794369 hold_0_12
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_2_nic_out_1_nic_in_2: switch_nic_out_1_nic_in_2 = 1 ->
   - 5089.669280600068 hold_2_1 - 3026.150415568619 hold_2_2
   - 2153.180441367374 hold_2_3 - 1671.107130863092 hold_2_4
   - 1365.407481935425 hold_2_5 - 1154.256553003943 hold_2_6
   - 999.665171097569 hold_2_7 - 881.5921573259316 hold_2_8
   - 788.4645856441134 hold_2_9 - 713.1323747193427 hold_2_10
   - 650.9396938908995 hold_2_11 - 598.7245818794369 hold_2_12
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_1_nic_in_3: switch_nic_out_1_nic_in_3 = 1 ->
   - 5089.669280600068 hold_3_1 - 3026.150415568619 hold_3_2
   - 2153.180441367374 hold_3_3 - 1671.107130863092 hold_3_4
   - 1365.407481935425 hold_3_5 - 1154.256553003943 hold_3_6
   - 999.665171097569 hold_3_7 - 881.5921573259316 hold_3_8
   - 788.4645856441134 hold_3_9 - 713.1323747193427 hold_3_10
   - 650.9396938908995 hold_3_11 - 598.7245818794369 hold_3_12
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_1_nic_in_4: switch_nic_out_1_nic_in_4 = 1 ->
   - 5089.669280600068 hold_4_1 - 3026.150415568619 hold_4_2
   - 2153.180441367374 hold_4_3 - 1671.107130863092 hold_4_4
   - 1365.407481935425 hold_4_5 - 1154.256553003943 hold_4_6
   - 999.665171097569 hold_4_7 - 881.5921573259316 hold_4_8
   - 788.4645856441134 hold_4_9 - 713.1323747193427 hold_4_10
   - 650.9396938908995 hold_4_11 - 598.7245818794369 hold_4_12
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_1_nic_in_5: switch_nic_out_1_nic_in_5 = 1 ->
   - 5089.669280600068 hold_5_1 - 3026.150415568619 hold_5_2
   - 2153.180441367374 hold_5_3 - 1671.107130863092 hold_5_4
   - 1365.407481935425 hold_5_5 - 1154.256553003943 hold_5_6
   - 999.665171097569 hold_5_7 - 881.5921573259316 hold_5_8
   - 788.4645856441134 hold_5_9 - 713.1323747193427 hold_5_10
   - 650.9396938908995 hold_5_11 - 598.7245818794369 hold_5_12
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_1_nic_in_6: switch_nic_out_1_nic_in_6 = 1 ->
   - 5089.669280600068 hold_6_1 - 3026.150415568619 hold_6_2
   - 2153.180441367374 hold_6_3 - 1671.107130863092 hold_6_4
   - 1365.407481935425 hold_6_5 - 1154.256553003943 hold_6_6
   - 999.665171097569 hold_6_7 - 881.5921573259316 hold_6_8
   - 788.4645856441134 hold_6_9 - 713.1323747193427 hold_6_10
   - 650.9396938908995 hold_6_11 - 598.7245818794369 hold_6_12
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_1_nic_in_7: switch_nic_out_1_nic_in_7 = 1 ->
   - 5089.669280600068 hold_7_1 - 3026.150415568619 hold_7_2
   - 2153.180441367374 hold_7_3 - 1671.107130863092 hold_7_4
   - 1365.407481935425 hold_7_5 - 1154.256553003943 hold_7_6
   - 999.665171097569 hold_7_7 - 881.5921573259316 hold_7_8
   - 788.4645856441134 hold_7_9 - 713.1323747193427 hold_7_10
   - 650.9396938908995 hold_7_11 - 598.7245818794369 hold_7_12
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_1_nic_in_8: switch_nic_out_1_nic_in_8 = 1 ->
   - 5089.669280600068 hold_8_1 - 3026.150415568619 hold_8_2
   - 2153.180441367374 hold_8_3 - 1671.107130863092 hold_8_4
   - 1365.407481935425 hold_8_5 - 1154.256553003943 hold_8_6
   - 999.665171097569 hold_8_7 - 881.5921573259316 hold_8_8
   - 788.4645856441134 hold_8_9 - 713.1323747193427 hold_8_10
   - 650.9396938908995 hold_8_11 - 598.7245818794369 hold_8_12
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_1_nic_in_9: switch_nic_out_1_nic_in_9 = 1 ->
   - 5089.669280600068 hold_9_1 - 3026.150415568619 hold_9_2
   - 2153.180441367374 hold_9_3 - 1671.107130863092 hold_9_4
   - 1365.407481935425 hold_9_5 - 1154.256553003943 hold_9_6
   - 999.665171097569 hold_9_7 - 881.5921573259316 hold_9_8
   - 788.4645856441134 hold_9_9 - 713.1323747193427 hold_9_10
   - 650.9396938908995 hold_9_11 - 598.7245818794369 hold_9_12
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_1_nic_in_10: switch_nic_out_1_nic_in_10 = 1 ->
   - 5089.669280600068 hold_10_1 - 3026.150415568619 hold_10_2
   - 2153.180441367374 hold_10_3 - 1671.107130863092 hold_10_4
   - 1365.407481935425 hold_10_5 - 1154.256553003943 hold_10_6
   - 999.665171097569 hold_10_7 - 881.5921573259316 hold_10_8
   - 788.4645856441134 hold_10_9 - 713.1323747193427 hold_10_10
   - 650.9396938908995 hold_10_11 - 598.7245818794369 hold_10_12
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_1_nic_in_11: switch_nic_out_1_nic_in_11 = 1 ->
   - 5089.669280600068 hold_11_1 - 3026.150415568619 hold_11_2
   - 2153.180441367374 hold_11_3 - 1671.107130863092 hold_11_4
   - 1365.407481935425 hold_11_5 - 1154.256553003943 hold_11_6
   - 999.665171097569 hold_11_7 - 881.5921573259316 hold_11_8
   - 788.4645856441134 hold_11_9 - 713.1323747193427 hold_11_10
   - 650.9396938908995 hold_11_11 - 598.7245818794369 hold_11_12
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_1_nic_in_12: switch_nic_out_1_nic_in_12 = 1 ->
   - 5089.669280600068 hold_12_1 - 3026.150415568619 hold_12_2
   - 2153.180441367374 hold_12_3 - 1671.107130863092 hold_12_4
   - 1365.407481935425 hold_12_5 - 1154.256553003943 hold_12_6
   - 999.665171097569 hold_12_7 - 881.5921573259316 hold_12_8
   - 788.4645856441134 hold_12_9 - 713.1323747193427 hold_12_10
   - 650.9396938908995 hold_12_11 - 598.7245818794369 hold_12_12
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_1_nic_in_13: switch_nic_out_1_nic_in_13 = 1 ->
   - 5089.669280600068 hold_13_1 - 3026.150415568619 hold_13_2
   - 2153.180441367374 hold_13_3 - 1671.107130863092 hold_13_4
   - 1365.407481935425 hold_13_5 - 1154.256553003943 hold_13_6
   - 999.665171097569 hold_13_7 - 881.5921573259316 hold_13_8
   - 788.4645856441134 hold_13_9 - 713.1323747193427 hold_13_10
   - 650.9396938908995 hold_13_11 - 598.7245818794369 hold_13_12
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_1_nic_in_14: switch_nic_out_1_nic_in_14 = 1 ->
   - 5089.669280600068 hold_14_1 - 3026.150415568619 hold_14_2
   - 2153.180441367374 hold_14_3 - 1671.107130863092 hold_14_4
   - 1365.407481935425 hold_14_5 - 1154.256553003943 hold_14_6
   - 999.665171097569 hold_14_7 - 881.5921573259316 hold_14_8
   - 788.4645856441134 hold_14_9 - 713.1323747193427 hold_14_10
   - 650.9396938908995 hold_14_11 - 598.7245818794369 hold_14_12
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_1_nic_in_15: switch_nic_out_1_nic_in_15 = 1 ->
   - 5089.669280600068 hold_15_1 - 3026.150415568619 hold_15_2
   - 2153.180441367374 hold_15_3 - 1671.107130863092 hold_15_4
   - 1365.407481935425 hold_15_5 - 1154.256553003943 hold_15_6
   - 999.665171097569 hold_15_7 - 881.5921573259316 hold_15_8
   - 788.4645856441134 hold_15_9 - 713.1323747193427 hold_15_10
   - 650.9396938908995 hold_15_11 - 598.7245818794369 hold_15_12
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_2_source_nic_in_2: switch_source_nic_in_2 = 1 ->
   - 6827.037409677124 hold_2_1 - 3565.661873596714 hold_2_2
   - 2412.957036174959 hold_2_3 - 1823.467617814477 hold_2_4
   - 1465.454616849587 hold_2_5 - 1224.951996455123 hold_2_6
   - 1052.260583931315 hold_2_7 - 922.2443255532355 hold_2_8
   - 820.8241232555838 hold_2_9 - 739.5004607016537 hold_2_10
   - 672.8385602120198 hold_2_11 - 617.20126020193 hold_2_12
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_0_nic_out_2_nic_in_0: switch_nic_out_2_nic_in_0 = 1 ->
   - 5089.669280600068 hold_0_1 - 3026.150415568619 hold_0_2
   - 2153.180441367374 hold_0_3 - 1671.107130863092 hold_0_4
   - 1365.407481935425 hold_0_5 - 1154.256553003943 hold_0_6
   - 999.665171097569 hold_0_7 - 881.5921573259316 hold_0_8
   - 788.4645856441134 hold_0_9 - 713.1323747193427 hold_0_10
   - 650.9396938908995 hold_0_11 - 598.7245818794369 hold_0_12
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_2_nic_in_1: switch_nic_out_2_nic_in_1 = 1 ->
   - 5089.669280600068 hold_1_1 - 3026.150415568619 hold_1_2
   - 2153.180441367374 hold_1_3 - 1671.107130863092 hold_1_4
   - 1365.407481935425 hold_1_5 - 1154.256553003943 hold_1_6
   - 999.665171097569 hold_1_7 - 881.5921573259316 hold_1_8
   - 788.4645856441134 hold_1_9 - 713.1323747193427 hold_1_10
   - 650.9396938908995 hold_1_11 - 598.7245818794369 hold_1_12
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_3_nic_out_2_nic_in_3: switch_nic_out_2_nic_in_3 = 1 ->
   - 5089.669280600068 hold_3_1 - 3026.150415568619 hold_3_2
   - 2153.180441367374 hold_3_3 - 1671.107130863092 hold_3_4
   - 1365.407481935425 hold_3_5 - 1154.256553003943 hold_3_6
   - 999.665171097569 hold_3_7 - 881.5921573259316 hold_3_8
   - 788.4645856441134 hold_3_9 - 713.1323747193427 hold_3_10
   - 650.9396938908995 hold_3_11 - 598.7245818794369 hold_3_12
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_2_nic_in_4: switch_nic_out_2_nic_in_4 = 1 ->
   - 5089.669280600068 hold_4_1 - 3026.150415568619 hold_4_2
   - 2153.180441367374 hold_4_3 - 1671.107130863092 hold_4_4
   - 1365.407481935425 hold_4_5 - 1154.256553003943 hold_4_6
   - 999.665171097569 hold_4_7 - 881.5921573259316 hold_4_8
   - 788.4645856441134 hold_4_9 - 713.1323747193427 hold_4_10
   - 650.9396938908995 hold_4_11 - 598.7245818794369 hold_4_12
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_2_nic_in_5: switch_nic_out_2_nic_in_5 = 1 ->
   - 5089.669280600068 hold_5_1 - 3026.150415568619 hold_5_2
   - 2153.180441367374 hold_5_3 - 1671.107130863092 hold_5_4
   - 1365.407481935425 hold_5_5 - 1154.256553003943 hold_5_6
   - 999.665171097569 hold_5_7 - 881.5921573259316 hold_5_8
   - 788.4645856441134 hold_5_9 - 713.1323747193427 hold_5_10
   - 650.9396938908995 hold_5_11 - 598.7245818794369 hold_5_12
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_2_nic_in_6: switch_nic_out_2_nic_in_6 = 1 ->
   - 5089.669280600068 hold_6_1 - 3026.150415568619 hold_6_2
   - 2153.180441367374 hold_6_3 - 1671.107130863092 hold_6_4
   - 1365.407481935425 hold_6_5 - 1154.256553003943 hold_6_6
   - 999.665171097569 hold_6_7 - 881.5921573259316 hold_6_8
   - 788.4645856441134 hold_6_9 - 713.1323747193427 hold_6_10
   - 650.9396938908995 hold_6_11 - 598.7245818794369 hold_6_12
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_2_nic_in_7: switch_nic_out_2_nic_in_7 = 1 ->
   - 5089.669280600068 hold_7_1 - 3026.150415568619 hold_7_2
   - 2153.180441367374 hold_7_3 - 1671.107130863092 hold_7_4
   - 1365.407481935425 hold_7_5 - 1154.256553003943 hold_7_6
   - 999.665171097569 hold_7_7 - 881.5921573259316 hold_7_8
   - 788.4645856441134 hold_7_9 - 713.1323747193427 hold_7_10
   - 650.9396938908995 hold_7_11 - 598.7245818794369 hold_7_12
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_2_nic_in_8: switch_nic_out_2_nic_in_8 = 1 ->
   - 5089.669280600068 hold_8_1 - 3026.150415568619 hold_8_2
   - 2153.180441367374 hold_8_3 - 1671.107130863092 hold_8_4
   - 1365.407481935425 hold_8_5 - 1154.256553003943 hold_8_6
   - 999.665171097569 hold_8_7 - 881.5921573259316 hold_8_8
   - 788.4645856441134 hold_8_9 - 713.1323747193427 hold_8_10
   - 650.9396938908995 hold_8_11 - 598.7245818794369 hold_8_12
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_2_nic_in_9: switch_nic_out_2_nic_in_9 = 1 ->
   - 5089.669280600068 hold_9_1 - 3026.150415568619 hold_9_2
   - 2153.180441367374 hold_9_3 - 1671.107130863092 hold_9_4
   - 1365.407481935425 hold_9_5 - 1154.256553003943 hold_9_6
   - 999.665171097569 hold_9_7 - 881.5921573259316 hold_9_8
   - 788.4645856441134 hold_9_9 - 713.1323747193427 hold_9_10
   - 650.9396938908995 hold_9_11 - 598.7245818794369 hold_9_12
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_2_nic_in_10: switch_nic_out_2_nic_in_10 = 1 ->
   - 5089.669280600068 hold_10_1 - 3026.150415568619 hold_10_2
   - 2153.180441367374 hold_10_3 - 1671.107130863092 hold_10_4
   - 1365.407481935425 hold_10_5 - 1154.256553003943 hold_10_6
   - 999.665171097569 hold_10_7 - 881.5921573259316 hold_10_8
   - 788.4645856441134 hold_10_9 - 713.1323747193427 hold_10_10
   - 650.9396938908995 hold_10_11 - 598.7245818794369 hold_10_12
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_2_nic_in_11: switch_nic_out_2_nic_in_11 = 1 ->
   - 5089.669280600068 hold_11_1 - 3026.150415568619 hold_11_2
   - 2153.180441367374 hold_11_3 - 1671.107130863092 hold_11_4
   - 1365.407481935425 hold_11_5 - 1154.256553003943 hold_11_6
   - 999.665171097569 hold_11_7 - 881.5921573259316 hold_11_8
   - 788.4645856441134 hold_11_9 - 713.1323747193427 hold_11_10
   - 650.9396938908995 hold_11_11 - 598.7245818794369 hold_11_12
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_2_nic_in_12: switch_nic_out_2_nic_in_12 = 1 ->
   - 5089.669280600068 hold_12_1 - 3026.150415568619 hold_12_2
   - 2153.180441367374 hold_12_3 - 1671.107130863092 hold_12_4
   - 1365.407481935425 hold_12_5 - 1154.256553003943 hold_12_6
   - 999.665171097569 hold_12_7 - 881.5921573259316 hold_12_8
   - 788.4645856441134 hold_12_9 - 713.1323747193427 hold_12_10
   - 650.9396938908995 hold_12_11 - 598.7245818794369 hold_12_12
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_2_nic_in_13: switch_nic_out_2_nic_in_13 = 1 ->
   - 5089.669280600068 hold_13_1 - 3026.150415568619 hold_13_2
   - 2153.180441367374 hold_13_3 - 1671.107130863092 hold_13_4
   - 1365.407481935425 hold_13_5 - 1154.256553003943 hold_13_6
   - 999.665171097569 hold_13_7 - 881.5921573259316 hold_13_8
   - 788.4645856441134 hold_13_9 - 713.1323747193427 hold_13_10
   - 650.9396938908995 hold_13_11 - 598.7245818794369 hold_13_12
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_2_nic_in_14: switch_nic_out_2_nic_in_14 = 1 ->
   - 5089.669280600068 hold_14_1 - 3026.150415568619 hold_14_2
   - 2153.180441367374 hold_14_3 - 1671.107130863092 hold_14_4
   - 1365.407481935425 hold_14_5 - 1154.256553003943 hold_14_6
   - 999.665171097569 hold_14_7 - 881.5921573259316 hold_14_8
   - 788.4645856441134 hold_14_9 - 713.1323747193427 hold_14_10
   - 650.9396938908995 hold_14_11 - 598.7245818794369 hold_14_12
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_2_nic_in_15: switch_nic_out_2_nic_in_15 = 1 ->
   - 5089.669280600068 hold_15_1 - 3026.150415568619 hold_15_2
   - 2153.180441367374 hold_15_3 - 1671.107130863092 hold_15_4
   - 1365.407481935425 hold_15_5 - 1154.256553003943 hold_15_6
   - 999.665171097569 hold_15_7 - 881.5921573259316 hold_15_8
   - 788.4645856441134 hold_15_9 - 713.1323747193427 hold_15_10
   - 650.9396938908995 hold_15_11 - 598.7245818794369 hold_15_12
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_3_source_nic_in_3: switch_source_nic_in_3 = 1 ->
   - 6827.037409677124 hold_3_1 - 3565.661873596714 hold_3_2
   - 2412.957036174959 hold_3_3 - 1823.467617814477 hold_3_4
   - 1465.454616849587 hold_3_5 - 1224.951996455123 hold_3_6
   - 1052.260583931315 hold_3_7 - 922.2443255532355 hold_3_8
   - 820.8241232555838 hold_3_9 - 739.5004607016537 hold_3_10
   - 672.8385602120198 hold_3_11 - 617.20126020193 hold_3_12
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_0_nic_out_3_nic_in_0: switch_nic_out_3_nic_in_0 = 1 ->
   - 5089.669280600068 hold_0_1 - 3026.150415568619 hold_0_2
   - 2153.180441367374 hold_0_3 - 1671.107130863092 hold_0_4
   - 1365.407481935425 hold_0_5 - 1154.256553003943 hold_0_6
   - 999.665171097569 hold_0_7 - 881.5921573259316 hold_0_8
   - 788.4645856441134 hold_0_9 - 713.1323747193427 hold_0_10
   - 650.9396938908995 hold_0_11 - 598.7245818794369 hold_0_12
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_3_nic_in_1: switch_nic_out_3_nic_in_1 = 1 ->
   - 5089.669280600068 hold_1_1 - 3026.150415568619 hold_1_2
   - 2153.180441367374 hold_1_3 - 1671.107130863092 hold_1_4
   - 1365.407481935425 hold_1_5 - 1154.256553003943 hold_1_6
   - 999.665171097569 hold_1_7 - 881.5921573259316 hold_1_8
   - 788.4645856441134 hold_1_9 - 713.1323747193427 hold_1_10
   - 650.9396938908995 hold_1_11 - 598.7245818794369 hold_1_12
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_3_nic_in_2: switch_nic_out_3_nic_in_2 = 1 ->
   - 5089.669280600068 hold_2_1 - 3026.150415568619 hold_2_2
   - 2153.180441367374 hold_2_3 - 1671.107130863092 hold_2_4
   - 1365.407481935425 hold_2_5 - 1154.256553003943 hold_2_6
   - 999.665171097569 hold_2_7 - 881.5921573259316 hold_2_8
   - 788.4645856441134 hold_2_9 - 713.1323747193427 hold_2_10
   - 650.9396938908995 hold_2_11 - 598.7245818794369 hold_2_12
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_4_nic_out_3_nic_in_4: switch_nic_out_3_nic_in_4 = 1 ->
   - 5089.669280600068 hold_4_1 - 3026.150415568619 hold_4_2
   - 2153.180441367374 hold_4_3 - 1671.107130863092 hold_4_4
   - 1365.407481935425 hold_4_5 - 1154.256553003943 hold_4_6
   - 999.665171097569 hold_4_7 - 881.5921573259316 hold_4_8
   - 788.4645856441134 hold_4_9 - 713.1323747193427 hold_4_10
   - 650.9396938908995 hold_4_11 - 598.7245818794369 hold_4_12
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_3_nic_in_5: switch_nic_out_3_nic_in_5 = 1 ->
   - 5089.669280600068 hold_5_1 - 3026.150415568619 hold_5_2
   - 2153.180441367374 hold_5_3 - 1671.107130863092 hold_5_4
   - 1365.407481935425 hold_5_5 - 1154.256553003943 hold_5_6
   - 999.665171097569 hold_5_7 - 881.5921573259316 hold_5_8
   - 788.4645856441134 hold_5_9 - 713.1323747193427 hold_5_10
   - 650.9396938908995 hold_5_11 - 598.7245818794369 hold_5_12
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_3_nic_in_6: switch_nic_out_3_nic_in_6 = 1 ->
   - 5089.669280600068 hold_6_1 - 3026.150415568619 hold_6_2
   - 2153.180441367374 hold_6_3 - 1671.107130863092 hold_6_4
   - 1365.407481935425 hold_6_5 - 1154.256553003943 hold_6_6
   - 999.665171097569 hold_6_7 - 881.5921573259316 hold_6_8
   - 788.4645856441134 hold_6_9 - 713.1323747193427 hold_6_10
   - 650.9396938908995 hold_6_11 - 598.7245818794369 hold_6_12
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_3_nic_in_7: switch_nic_out_3_nic_in_7 = 1 ->
   - 5089.669280600068 hold_7_1 - 3026.150415568619 hold_7_2
   - 2153.180441367374 hold_7_3 - 1671.107130863092 hold_7_4
   - 1365.407481935425 hold_7_5 - 1154.256553003943 hold_7_6
   - 999.665171097569 hold_7_7 - 881.5921573259316 hold_7_8
   - 788.4645856441134 hold_7_9 - 713.1323747193427 hold_7_10
   - 650.9396938908995 hold_7_11 - 598.7245818794369 hold_7_12
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_3_nic_in_8: switch_nic_out_3_nic_in_8 = 1 ->
   - 5089.669280600068 hold_8_1 - 3026.150415568619 hold_8_2
   - 2153.180441367374 hold_8_3 - 1671.107130863092 hold_8_4
   - 1365.407481935425 hold_8_5 - 1154.256553003943 hold_8_6
   - 999.665171097569 hold_8_7 - 881.5921573259316 hold_8_8
   - 788.4645856441134 hold_8_9 - 713.1323747193427 hold_8_10
   - 650.9396938908995 hold_8_11 - 598.7245818794369 hold_8_12
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_3_nic_in_9: switch_nic_out_3_nic_in_9 = 1 ->
   - 5089.669280600068 hold_9_1 - 3026.150415568619 hold_9_2
   - 2153.180441367374 hold_9_3 - 1671.107130863092 hold_9_4
   - 1365.407481935425 hold_9_5 - 1154.256553003943 hold_9_6
   - 999.665171097569 hold_9_7 - 881.5921573259316 hold_9_8
   - 788.4645856441134 hold_9_9 - 713.1323747193427 hold_9_10
   - 650.9396938908995 hold_9_11 - 598.7245818794369 hold_9_12
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_3_nic_in_10: switch_nic_out_3_nic_in_10 = 1 ->
   - 5089.669280600068 hold_10_1 - 3026.150415568619 hold_10_2
   - 2153.180441367374 hold_10_3 - 1671.107130863092 hold_10_4
   - 1365.407481935425 hold_10_5 - 1154.256553003943 hold_10_6
   - 999.665171097569 hold_10_7 - 881.5921573259316 hold_10_8
   - 788.4645856441134 hold_10_9 - 713.1323747193427 hold_10_10
   - 650.9396938908995 hold_10_11 - 598.7245818794369 hold_10_12
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_3_nic_in_11: switch_nic_out_3_nic_in_11 = 1 ->
   - 5089.669280600068 hold_11_1 - 3026.150415568619 hold_11_2
   - 2153.180441367374 hold_11_3 - 1671.107130863092 hold_11_4
   - 1365.407481935425 hold_11_5 - 1154.256553003943 hold_11_6
   - 999.665171097569 hold_11_7 - 881.5921573259316 hold_11_8
   - 788.4645856441134 hold_11_9 - 713.1323747193427 hold_11_10
   - 650.9396938908995 hold_11_11 - 598.7245818794369 hold_11_12
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_3_nic_in_12: switch_nic_out_3_nic_in_12 = 1 ->
   - 5089.669280600068 hold_12_1 - 3026.150415568619 hold_12_2
   - 2153.180441367374 hold_12_3 - 1671.107130863092 hold_12_4
   - 1365.407481935425 hold_12_5 - 1154.256553003943 hold_12_6
   - 999.665171097569 hold_12_7 - 881.5921573259316 hold_12_8
   - 788.4645856441134 hold_12_9 - 713.1323747193427 hold_12_10
   - 650.9396938908995 hold_12_11 - 598.7245818794369 hold_12_12
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_3_nic_in_13: switch_nic_out_3_nic_in_13 = 1 ->
   - 5089.669280600068 hold_13_1 - 3026.150415568619 hold_13_2
   - 2153.180441367374 hold_13_3 - 1671.107130863092 hold_13_4
   - 1365.407481935425 hold_13_5 - 1154.256553003943 hold_13_6
   - 999.665171097569 hold_13_7 - 881.5921573259316 hold_13_8
   - 788.4645856441134 hold_13_9 - 713.1323747193427 hold_13_10
   - 650.9396938908995 hold_13_11 - 598.7245818794369 hold_13_12
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_3_nic_in_14: switch_nic_out_3_nic_in_14 = 1 ->
   - 5089.669280600068 hold_14_1 - 3026.150415568619 hold_14_2
   - 2153.180441367374 hold_14_3 - 1671.107130863092 hold_14_4
   - 1365.407481935425 hold_14_5 - 1154.256553003943 hold_14_6
   - 999.665171097569 hold_14_7 - 881.5921573259316 hold_14_8
   - 788.4645856441134 hold_14_9 - 713.1323747193427 hold_14_10
   - 650.9396938908995 hold_14_11 - 598.7245818794369 hold_14_12
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_3_nic_in_15: switch_nic_out_3_nic_in_15 = 1 ->
   - 5089.669280600068 hold_15_1 - 3026.150415568619 hold_15_2
   - 2153.180441367374 hold_15_3 - 1671.107130863092 hold_15_4
   - 1365.407481935425 hold_15_5 - 1154.256553003943 hold_15_6
   - 999.665171097569 hold_15_7 - 881.5921573259316 hold_15_8
   - 788.4645856441134 hold_15_9 - 713.1323747193427 hold_15_10
   - 650.9396938908995 hold_15_11 - 598.7245818794369 hold_15_12
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_4_source_nic_in_4: switch_source_nic_in_4 = 1 ->
   - 6827.037409677124 hold_4_1 - 3565.661873596714 hold_4_2
   - 2412.957036174959 hold_4_3 - 1823.467617814477 hold_4_4
   - 1465.454616849587 hold_4_5 - 1224.951996455123 hold_4_6
   - 1052.260583931315 hold_4_7 - 922.2443255532355 hold_4_8
   - 820.8241232555838 hold_4_9 - 739.5004607016537 hold_4_10
   - 672.8385602120198 hold_4_11 - 617.20126020193 hold_4_12
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_0_nic_out_4_nic_in_0: switch_nic_out_4_nic_in_0 = 1 ->
   - 5089.669280600068 hold_0_1 - 3026.150415568619 hold_0_2
   - 2153.180441367374 hold_0_3 - 1671.107130863092 hold_0_4
   - 1365.407481935425 hold_0_5 - 1154.256553003943 hold_0_6
   - 999.665171097569 hold_0_7 - 881.5921573259316 hold_0_8
   - 788.4645856441134 hold_0_9 - 713.1323747193427 hold_0_10
   - 650.9396938908995 hold_0_11 - 598.7245818794369 hold_0_12
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_4_nic_in_1: switch_nic_out_4_nic_in_1 = 1 ->
   - 5089.669280600068 hold_1_1 - 3026.150415568619 hold_1_2
   - 2153.180441367374 hold_1_3 - 1671.107130863092 hold_1_4
   - 1365.407481935425 hold_1_5 - 1154.256553003943 hold_1_6
   - 999.665171097569 hold_1_7 - 881.5921573259316 hold_1_8
   - 788.4645856441134 hold_1_9 - 713.1323747193427 hold_1_10
   - 650.9396938908995 hold_1_11 - 598.7245818794369 hold_1_12
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_4_nic_in_2: switch_nic_out_4_nic_in_2 = 1 ->
   - 5089.669280600068 hold_2_1 - 3026.150415568619 hold_2_2
   - 2153.180441367374 hold_2_3 - 1671.107130863092 hold_2_4
   - 1365.407481935425 hold_2_5 - 1154.256553003943 hold_2_6
   - 999.665171097569 hold_2_7 - 881.5921573259316 hold_2_8
   - 788.4645856441134 hold_2_9 - 713.1323747193427 hold_2_10
   - 650.9396938908995 hold_2_11 - 598.7245818794369 hold_2_12
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_4_nic_in_3: switch_nic_out_4_nic_in_3 = 1 ->
   - 5089.669280600068 hold_3_1 - 3026.150415568619 hold_3_2
   - 2153.180441367374 hold_3_3 - 1671.107130863092 hold_3_4
   - 1365.407481935425 hold_3_5 - 1154.256553003943 hold_3_6
   - 999.665171097569 hold_3_7 - 881.5921573259316 hold_3_8
   - 788.4645856441134 hold_3_9 - 713.1323747193427 hold_3_10
   - 650.9396938908995 hold_3_11 - 598.7245818794369 hold_3_12
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_5_nic_out_4_nic_in_5: switch_nic_out_4_nic_in_5 = 1 ->
   - 5089.669280600068 hold_5_1 - 3026.150415568619 hold_5_2
   - 2153.180441367374 hold_5_3 - 1671.107130863092 hold_5_4
   - 1365.407481935425 hold_5_5 - 1154.256553003943 hold_5_6
   - 999.665171097569 hold_5_7 - 881.5921573259316 hold_5_8
   - 788.4645856441134 hold_5_9 - 713.1323747193427 hold_5_10
   - 650.9396938908995 hold_5_11 - 598.7245818794369 hold_5_12
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_4_nic_in_6: switch_nic_out_4_nic_in_6 = 1 ->
   - 5089.669280600068 hold_6_1 - 3026.150415568619 hold_6_2
   - 2153.180441367374 hold_6_3 - 1671.107130863092 hold_6_4
   - 1365.407481935425 hold_6_5 - 1154.256553003943 hold_6_6
   - 999.665171097569 hold_6_7 - 881.5921573259316 hold_6_8
   - 788.4645856441134 hold_6_9 - 713.1323747193427 hold_6_10
   - 650.9396938908995 hold_6_11 - 598.7245818794369 hold_6_12
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_4_nic_in_7: switch_nic_out_4_nic_in_7 = 1 ->
   - 5089.669280600068 hold_7_1 - 3026.150415568619 hold_7_2
   - 2153.180441367374 hold_7_3 - 1671.107130863092 hold_7_4
   - 1365.407481935425 hold_7_5 - 1154.256553003943 hold_7_6
   - 999.665171097569 hold_7_7 - 881.5921573259316 hold_7_8
   - 788.4645856441134 hold_7_9 - 713.1323747193427 hold_7_10
   - 650.9396938908995 hold_7_11 - 598.7245818794369 hold_7_12
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_4_nic_in_8: switch_nic_out_4_nic_in_8 = 1 ->
   - 5089.669280600068 hold_8_1 - 3026.150415568619 hold_8_2
   - 2153.180441367374 hold_8_3 - 1671.107130863092 hold_8_4
   - 1365.407481935425 hold_8_5 - 1154.256553003943 hold_8_6
   - 999.665171097569 hold_8_7 - 881.5921573259316 hold_8_8
   - 788.4645856441134 hold_8_9 - 713.1323747193427 hold_8_10
   - 650.9396938908995 hold_8_11 - 598.7245818794369 hold_8_12
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_4_nic_in_9: switch_nic_out_4_nic_in_9 = 1 ->
   - 5089.669280600068 hold_9_1 - 3026.150415568619 hold_9_2
   - 2153.180441367374 hold_9_3 - 1671.107130863092 hold_9_4
   - 1365.407481935425 hold_9_5 - 1154.256553003943 hold_9_6
   - 999.665171097569 hold_9_7 - 881.5921573259316 hold_9_8
   - 788.4645856441134 hold_9_9 - 713.1323747193427 hold_9_10
   - 650.9396938908995 hold_9_11 - 598.7245818794369 hold_9_12
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_4_nic_in_10: switch_nic_out_4_nic_in_10 = 1 ->
   - 5089.669280600068 hold_10_1 - 3026.150415568619 hold_10_2
   - 2153.180441367374 hold_10_3 - 1671.107130863092 hold_10_4
   - 1365.407481935425 hold_10_5 - 1154.256553003943 hold_10_6
   - 999.665171097569 hold_10_7 - 881.5921573259316 hold_10_8
   - 788.4645856441134 hold_10_9 - 713.1323747193427 hold_10_10
   - 650.9396938908995 hold_10_11 - 598.7245818794369 hold_10_12
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_4_nic_in_11: switch_nic_out_4_nic_in_11 = 1 ->
   - 5089.669280600068 hold_11_1 - 3026.150415568619 hold_11_2
   - 2153.180441367374 hold_11_3 - 1671.107130863092 hold_11_4
   - 1365.407481935425 hold_11_5 - 1154.256553003943 hold_11_6
   - 999.665171097569 hold_11_7 - 881.5921573259316 hold_11_8
   - 788.4645856441134 hold_11_9 - 713.1323747193427 hold_11_10
   - 650.9396938908995 hold_11_11 - 598.7245818794369 hold_11_12
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_4_nic_in_12: switch_nic_out_4_nic_in_12 = 1 ->
   - 5089.669280600068 hold_12_1 - 3026.150415568619 hold_12_2
   - 2153.180441367374 hold_12_3 - 1671.107130863092 hold_12_4
   - 1365.407481935425 hold_12_5 - 1154.256553003943 hold_12_6
   - 999.665171097569 hold_12_7 - 881.5921573259316 hold_12_8
   - 788.4645856441134 hold_12_9 - 713.1323747193427 hold_12_10
   - 650.9396938908995 hold_12_11 - 598.7245818794369 hold_12_12
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_4_nic_in_13: switch_nic_out_4_nic_in_13 = 1 ->
   - 5089.669280600068 hold_13_1 - 3026.150415568619 hold_13_2
   - 2153.180441367374 hold_13_3 - 1671.107130863092 hold_13_4
   - 1365.407481935425 hold_13_5 - 1154.256553003943 hold_13_6
   - 999.665171097569 hold_13_7 - 881.5921573259316 hold_13_8
   - 788.4645856441134 hold_13_9 - 713.1323747193427 hold_13_10
   - 650.9396938908995 hold_13_11 - 598.7245818794369 hold_13_12
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_4_nic_in_14: switch_nic_out_4_nic_in_14 = 1 ->
   - 5089.669280600068 hold_14_1 - 3026.150415568619 hold_14_2
   - 2153.180441367374 hold_14_3 - 1671.107130863092 hold_14_4
   - 1365.407481935425 hold_14_5 - 1154.256553003943 hold_14_6
   - 999.665171097569 hold_14_7 - 881.5921573259316 hold_14_8
   - 788.4645856441134 hold_14_9 - 713.1323747193427 hold_14_10
   - 650.9396938908995 hold_14_11 - 598.7245818794369 hold_14_12
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_4_nic_in_15: switch_nic_out_4_nic_in_15 = 1 ->
   - 5089.669280600068 hold_15_1 - 3026.150415568619 hold_15_2
   - 2153.180441367374 hold_15_3 - 1671.107130863092 hold_15_4
   - 1365.407481935425 hold_15_5 - 1154.256553003943 hold_15_6
   - 999.665171097569 hold_15_7 - 881.5921573259316 hold_15_8
   - 788.4645856441134 hold_15_9 - 713.1323747193427 hold_15_10
   - 650.9396938908995 hold_15_11 - 598.7245818794369 hold_15_12
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_5_source_nic_in_5: switch_source_nic_in_5 = 1 ->
   - 6827.037409677124 hold_5_1 - 3565.661873596714 hold_5_2
   - 2412.957036174959 hold_5_3 - 1823.467617814477 hold_5_4
   - 1465.454616849587 hold_5_5 - 1224.951996455123 hold_5_6
   - 1052.260583931315 hold_5_7 - 922.2443255532355 hold_5_8
   - 820.8241232555838 hold_5_9 - 739.5004607016537 hold_5_10
   - 672.8385602120198 hold_5_11 - 617.20126020193 hold_5_12
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_0_nic_out_5_nic_in_0: switch_nic_out_5_nic_in_0 = 1 ->
   - 5089.669280600068 hold_0_1 - 3026.150415568619 hold_0_2
   - 2153.180441367374 hold_0_3 - 1671.107130863092 hold_0_4
   - 1365.407481935425 hold_0_5 - 1154.256553003943 hold_0_6
   - 999.665171097569 hold_0_7 - 881.5921573259316 hold_0_8
   - 788.4645856441134 hold_0_9 - 713.1323747193427 hold_0_10
   - 650.9396938908995 hold_0_11 - 598.7245818794369 hold_0_12
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_5_nic_in_1: switch_nic_out_5_nic_in_1 = 1 ->
   - 5089.669280600068 hold_1_1 - 3026.150415568619 hold_1_2
   - 2153.180441367374 hold_1_3 - 1671.107130863092 hold_1_4
   - 1365.407481935425 hold_1_5 - 1154.256553003943 hold_1_6
   - 999.665171097569 hold_1_7 - 881.5921573259316 hold_1_8
   - 788.4645856441134 hold_1_9 - 713.1323747193427 hold_1_10
   - 650.9396938908995 hold_1_11 - 598.7245818794369 hold_1_12
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_5_nic_in_2: switch_nic_out_5_nic_in_2 = 1 ->
   - 5089.669280600068 hold_2_1 - 3026.150415568619 hold_2_2
   - 2153.180441367374 hold_2_3 - 1671.107130863092 hold_2_4
   - 1365.407481935425 hold_2_5 - 1154.256553003943 hold_2_6
   - 999.665171097569 hold_2_7 - 881.5921573259316 hold_2_8
   - 788.4645856441134 hold_2_9 - 713.1323747193427 hold_2_10
   - 650.9396938908995 hold_2_11 - 598.7245818794369 hold_2_12
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_5_nic_in_3: switch_nic_out_5_nic_in_3 = 1 ->
   - 5089.669280600068 hold_3_1 - 3026.150415568619 hold_3_2
   - 2153.180441367374 hold_3_3 - 1671.107130863092 hold_3_4
   - 1365.407481935425 hold_3_5 - 1154.256553003943 hold_3_6
   - 999.665171097569 hold_3_7 - 881.5921573259316 hold_3_8
   - 788.4645856441134 hold_3_9 - 713.1323747193427 hold_3_10
   - 650.9396938908995 hold_3_11 - 598.7245818794369 hold_3_12
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_5_nic_in_4: switch_nic_out_5_nic_in_4 = 1 ->
   - 5089.669280600068 hold_4_1 - 3026.150415568619 hold_4_2
   - 2153.180441367374 hold_4_3 - 1671.107130863092 hold_4_4
   - 1365.407481935425 hold_4_5 - 1154.256553003943 hold_4_6
   - 999.665171097569 hold_4_7 - 881.5921573259316 hold_4_8
   - 788.4645856441134 hold_4_9 - 713.1323747193427 hold_4_10
   - 650.9396938908995 hold_4_11 - 598.7245818794369 hold_4_12
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_6_nic_out_5_nic_in_6: switch_nic_out_5_nic_in_6 = 1 ->
   - 5089.669280600068 hold_6_1 - 3026.150415568619 hold_6_2
   - 2153.180441367374 hold_6_3 - 1671.107130863092 hold_6_4
   - 1365.407481935425 hold_6_5 - 1154.256553003943 hold_6_6
   - 999.665171097569 hold_6_7 - 881.5921573259316 hold_6_8
   - 788.4645856441134 hold_6_9 - 713.1323747193427 hold_6_10
   - 650.9396938908995 hold_6_11 - 598.7245818794369 hold_6_12
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_5_nic_in_7: switch_nic_out_5_nic_in_7 = 1 ->
   - 5089.669280600068 hold_7_1 - 3026.150415568619 hold_7_2
   - 2153.180441367374 hold_7_3 - 1671.107130863092 hold_7_4
   - 1365.407481935425 hold_7_5 - 1154.256553003943 hold_7_6
   - 999.665171097569 hold_7_7 - 881.5921573259316 hold_7_8
   - 788.4645856441134 hold_7_9 - 713.1323747193427 hold_7_10
   - 650.9396938908995 hold_7_11 - 598.7245818794369 hold_7_12
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_5_nic_in_8: switch_nic_out_5_nic_in_8 = 1 ->
   - 5089.669280600068 hold_8_1 - 3026.150415568619 hold_8_2
   - 2153.180441367374 hold_8_3 - 1671.107130863092 hold_8_4
   - 1365.407481935425 hold_8_5 - 1154.256553003943 hold_8_6
   - 999.665171097569 hold_8_7 - 881.5921573259316 hold_8_8
   - 788.4645856441134 hold_8_9 - 713.1323747193427 hold_8_10
   - 650.9396938908995 hold_8_11 - 598.7245818794369 hold_8_12
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_5_nic_in_9: switch_nic_out_5_nic_in_9 = 1 ->
   - 5089.669280600068 hold_9_1 - 3026.150415568619 hold_9_2
   - 2153.180441367374 hold_9_3 - 1671.107130863092 hold_9_4
   - 1365.407481935425 hold_9_5 - 1154.256553003943 hold_9_6
   - 999.665171097569 hold_9_7 - 881.5921573259316 hold_9_8
   - 788.4645856441134 hold_9_9 - 713.1323747193427 hold_9_10
   - 650.9396938908995 hold_9_11 - 598.7245818794369 hold_9_12
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_5_nic_in_10: switch_nic_out_5_nic_in_10 = 1 ->
   - 5089.669280600068 hold_10_1 - 3026.150415568619 hold_10_2
   - 2153.180441367374 hold_10_3 - 1671.107130863092 hold_10_4
   - 1365.407481935425 hold_10_5 - 1154.256553003943 hold_10_6
   - 999.665171097569 hold_10_7 - 881.5921573259316 hold_10_8
   - 788.4645856441134 hold_10_9 - 713.1323747193427 hold_10_10
   - 650.9396938908995 hold_10_11 - 598.7245818794369 hold_10_12
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_5_nic_in_11: switch_nic_out_5_nic_in_11 = 1 ->
   - 5089.669280600068 hold_11_1 - 3026.150415568619 hold_11_2
   - 2153.180441367374 hold_11_3 - 1671.107130863092 hold_11_4
   - 1365.407481935425 hold_11_5 - 1154.256553003943 hold_11_6
   - 999.665171097569 hold_11_7 - 881.5921573259316 hold_11_8
   - 788.4645856441134 hold_11_9 - 713.1323747193427 hold_11_10
   - 650.9396938908995 hold_11_11 - 598.7245818794369 hold_11_12
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_5_nic_in_12: switch_nic_out_5_nic_in_12 = 1 ->
   - 5089.669280600068 hold_12_1 - 3026.150415568619 hold_12_2
   - 2153.180441367374 hold_12_3 - 1671.107130863092 hold_12_4
   - 1365.407481935425 hold_12_5 - 1154.256553003943 hold_12_6
   - 999.665171097569 hold_12_7 - 881.5921573259316 hold_12_8
   - 788.4645856441134 hold_12_9 - 713.1323747193427 hold_12_10
   - 650.9396938908995 hold_12_11 - 598.7245818794369 hold_12_12
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_5_nic_in_13: switch_nic_out_5_nic_in_13 = 1 ->
   - 5089.669280600068 hold_13_1 - 3026.150415568619 hold_13_2
   - 2153.180441367374 hold_13_3 - 1671.107130863092 hold_13_4
   - 1365.407481935425 hold_13_5 - 1154.256553003943 hold_13_6
   - 999.665171097569 hold_13_7 - 881.5921573259316 hold_13_8
   - 788.4645856441134 hold_13_9 - 713.1323747193427 hold_13_10
   - 650.9396938908995 hold_13_11 - 598.7245818794369 hold_13_12
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_5_nic_in_14: switch_nic_out_5_nic_in_14 = 1 ->
   - 5089.669280600068 hold_14_1 - 3026.150415568619 hold_14_2
   - 2153.180441367374 hold_14_3 - 1671.107130863092 hold_14_4
   - 1365.407481935425 hold_14_5 - 1154.256553003943 hold_14_6
   - 999.665171097569 hold_14_7 - 881.5921573259316 hold_14_8
   - 788.4645856441134 hold_14_9 - 713.1323747193427 hold_14_10
   - 650.9396938908995 hold_14_11 - 598.7245818794369 hold_14_12
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_5_nic_in_15: switch_nic_out_5_nic_in_15 = 1 ->
   - 5089.669280600068 hold_15_1 - 3026.150415568619 hold_15_2
   - 2153.180441367374 hold_15_3 - 1671.107130863092 hold_15_4
   - 1365.407481935425 hold_15_5 - 1154.256553003943 hold_15_6
   - 999.665171097569 hold_15_7 - 881.5921573259316 hold_15_8
   - 788.4645856441134 hold_15_9 - 713.1323747193427 hold_15_10
   - 650.9396938908995 hold_15_11 - 598.7245818794369 hold_15_12
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_6_source_nic_in_6: switch_source_nic_in_6 = 1 ->
   - 6827.037409677124 hold_6_1 - 3565.661873596714 hold_6_2
   - 2412.957036174959 hold_6_3 - 1823.467617814477 hold_6_4
   - 1465.454616849587 hold_6_5 - 1224.951996455123 hold_6_6
   - 1052.260583931315 hold_6_7 - 922.2443255532355 hold_6_8
   - 820.8241232555838 hold_6_9 - 739.5004607016537 hold_6_10
   - 672.8385602120198 hold_6_11 - 617.20126020193 hold_6_12
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_0_nic_out_6_nic_in_0: switch_nic_out_6_nic_in_0 = 1 ->
   - 5089.669280600068 hold_0_1 - 3026.150415568619 hold_0_2
   - 2153.180441367374 hold_0_3 - 1671.107130863092 hold_0_4
   - 1365.407481935425 hold_0_5 - 1154.256553003943 hold_0_6
   - 999.665171097569 hold_0_7 - 881.5921573259316 hold_0_8
   - 788.4645856441134 hold_0_9 - 713.1323747193427 hold_0_10
   - 650.9396938908995 hold_0_11 - 598.7245818794369 hold_0_12
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_6_nic_in_1: switch_nic_out_6_nic_in_1 = 1 ->
   - 5089.669280600068 hold_1_1 - 3026.150415568619 hold_1_2
   - 2153.180441367374 hold_1_3 - 1671.107130863092 hold_1_4
   - 1365.407481935425 hold_1_5 - 1154.256553003943 hold_1_6
   - 999.665171097569 hold_1_7 - 881.5921573259316 hold_1_8
   - 788.4645856441134 hold_1_9 - 713.1323747193427 hold_1_10
   - 650.9396938908995 hold_1_11 - 598.7245818794369 hold_1_12
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_6_nic_in_2: switch_nic_out_6_nic_in_2 = 1 ->
   - 5089.669280600068 hold_2_1 - 3026.150415568619 hold_2_2
   - 2153.180441367374 hold_2_3 - 1671.107130863092 hold_2_4
   - 1365.407481935425 hold_2_5 - 1154.256553003943 hold_2_6
   - 999.665171097569 hold_2_7 - 881.5921573259316 hold_2_8
   - 788.4645856441134 hold_2_9 - 713.1323747193427 hold_2_10
   - 650.9396938908995 hold_2_11 - 598.7245818794369 hold_2_12
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_6_nic_in_3: switch_nic_out_6_nic_in_3 = 1 ->
   - 5089.669280600068 hold_3_1 - 3026.150415568619 hold_3_2
   - 2153.180441367374 hold_3_3 - 1671.107130863092 hold_3_4
   - 1365.407481935425 hold_3_5 - 1154.256553003943 hold_3_6
   - 999.665171097569 hold_3_7 - 881.5921573259316 hold_3_8
   - 788.4645856441134 hold_3_9 - 713.1323747193427 hold_3_10
   - 650.9396938908995 hold_3_11 - 598.7245818794369 hold_3_12
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_6_nic_in_4: switch_nic_out_6_nic_in_4 = 1 ->
   - 5089.669280600068 hold_4_1 - 3026.150415568619 hold_4_2
   - 2153.180441367374 hold_4_3 - 1671.107130863092 hold_4_4
   - 1365.407481935425 hold_4_5 - 1154.256553003943 hold_4_6
   - 999.665171097569 hold_4_7 - 881.5921573259316 hold_4_8
   - 788.4645856441134 hold_4_9 - 713.1323747193427 hold_4_10
   - 650.9396938908995 hold_4_11 - 598.7245818794369 hold_4_12
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_6_nic_in_5: switch_nic_out_6_nic_in_5 = 1 ->
   - 5089.669280600068 hold_5_1 - 3026.150415568619 hold_5_2
   - 2153.180441367374 hold_5_3 - 1671.107130863092 hold_5_4
   - 1365.407481935425 hold_5_5 - 1154.256553003943 hold_5_6
   - 999.665171097569 hold_5_7 - 881.5921573259316 hold_5_8
   - 788.4645856441134 hold_5_9 - 713.1323747193427 hold_5_10
   - 650.9396938908995 hold_5_11 - 598.7245818794369 hold_5_12
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_7_nic_out_6_nic_in_7: switch_nic_out_6_nic_in_7 = 1 ->
   - 5089.669280600068 hold_7_1 - 3026.150415568619 hold_7_2
   - 2153.180441367374 hold_7_3 - 1671.107130863092 hold_7_4
   - 1365.407481935425 hold_7_5 - 1154.256553003943 hold_7_6
   - 999.665171097569 hold_7_7 - 881.5921573259316 hold_7_8
   - 788.4645856441134 hold_7_9 - 713.1323747193427 hold_7_10
   - 650.9396938908995 hold_7_11 - 598.7245818794369 hold_7_12
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_6_nic_in_8: switch_nic_out_6_nic_in_8 = 1 ->
   - 5089.669280600068 hold_8_1 - 3026.150415568619 hold_8_2
   - 2153.180441367374 hold_8_3 - 1671.107130863092 hold_8_4
   - 1365.407481935425 hold_8_5 - 1154.256553003943 hold_8_6
   - 999.665171097569 hold_8_7 - 881.5921573259316 hold_8_8
   - 788.4645856441134 hold_8_9 - 713.1323747193427 hold_8_10
   - 650.9396938908995 hold_8_11 - 598.7245818794369 hold_8_12
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_6_nic_in_9: switch_nic_out_6_nic_in_9 = 1 ->
   - 5089.669280600068 hold_9_1 - 3026.150415568619 hold_9_2
   - 2153.180441367374 hold_9_3 - 1671.107130863092 hold_9_4
   - 1365.407481935425 hold_9_5 - 1154.256553003943 hold_9_6
   - 999.665171097569 hold_9_7 - 881.5921573259316 hold_9_8
   - 788.4645856441134 hold_9_9 - 713.1323747193427 hold_9_10
   - 650.9396938908995 hold_9_11 - 598.7245818794369 hold_9_12
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_6_nic_in_10: switch_nic_out_6_nic_in_10 = 1 ->
   - 5089.669280600068 hold_10_1 - 3026.150415568619 hold_10_2
   - 2153.180441367374 hold_10_3 - 1671.107130863092 hold_10_4
   - 1365.407481935425 hold_10_5 - 1154.256553003943 hold_10_6
   - 999.665171097569 hold_10_7 - 881.5921573259316 hold_10_8
   - 788.4645856441134 hold_10_9 - 713.1323747193427 hold_10_10
   - 650.9396938908995 hold_10_11 - 598.7245818794369 hold_10_12
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_6_nic_in_11: switch_nic_out_6_nic_in_11 = 1 ->
   - 5089.669280600068 hold_11_1 - 3026.150415568619 hold_11_2
   - 2153.180441367374 hold_11_3 - 1671.107130863092 hold_11_4
   - 1365.407481935425 hold_11_5 - 1154.256553003943 hold_11_6
   - 999.665171097569 hold_11_7 - 881.5921573259316 hold_11_8
   - 788.4645856441134 hold_11_9 - 713.1323747193427 hold_11_10
   - 650.9396938908995 hold_11_11 - 598.7245818794369 hold_11_12
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_6_nic_in_12: switch_nic_out_6_nic_in_12 = 1 ->
   - 5089.669280600068 hold_12_1 - 3026.150415568619 hold_12_2
   - 2153.180441367374 hold_12_3 - 1671.107130863092 hold_12_4
   - 1365.407481935425 hold_12_5 - 1154.256553003943 hold_12_6
   - 999.665171097569 hold_12_7 - 881.5921573259316 hold_12_8
   - 788.4645856441134 hold_12_9 - 713.1323747193427 hold_12_10
   - 650.9396938908995 hold_12_11 - 598.7245818794369 hold_12_12
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_6_nic_in_13: switch_nic_out_6_nic_in_13 = 1 ->
   - 5089.669280600068 hold_13_1 - 3026.150415568619 hold_13_2
   - 2153.180441367374 hold_13_3 - 1671.107130863092 hold_13_4
   - 1365.407481935425 hold_13_5 - 1154.256553003943 hold_13_6
   - 999.665171097569 hold_13_7 - 881.5921573259316 hold_13_8
   - 788.4645856441134 hold_13_9 - 713.1323747193427 hold_13_10
   - 650.9396938908995 hold_13_11 - 598.7245818794369 hold_13_12
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_6_nic_in_14: switch_nic_out_6_nic_in_14 = 1 ->
   - 5089.669280600068 hold_14_1 - 3026.150415568619 hold_14_2
   - 2153.180441367374 hold_14_3 - 1671.107130863092 hold_14_4
   - 1365.407481935425 hold_14_5 - 1154.256553003943 hold_14_6
   - 999.665171097569 hold_14_7 - 881.5921573259316 hold_14_8
   - 788.4645856441134 hold_14_9 - 713.1323747193427 hold_14_10
   - 650.9396938908995 hold_14_11 - 598.7245818794369 hold_14_12
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_6_nic_in_15: switch_nic_out_6_nic_in_15 = 1 ->
   - 5089.669280600068 hold_15_1 - 3026.150415568619 hold_15_2
   - 2153.180441367374 hold_15_3 - 1671.107130863092 hold_15_4
   - 1365.407481935425 hold_15_5 - 1154.256553003943 hold_15_6
   - 999.665171097569 hold_15_7 - 881.5921573259316 hold_15_8
   - 788.4645856441134 hold_15_9 - 713.1323747193427 hold_15_10
   - 650.9396938908995 hold_15_11 - 598.7245818794369 hold_15_12
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_7_source_nic_in_7: switch_source_nic_in_7 = 1 ->
   - 6827.037409677124 hold_7_1 - 3565.661873596714 hold_7_2
   - 2412.957036174959 hold_7_3 - 1823.467617814477 hold_7_4
   - 1465.454616849587 hold_7_5 - 1224.951996455123 hold_7_6
   - 1052.260583931315 hold_7_7 - 922.2443255532355 hold_7_8
   - 820.8241232555838 hold_7_9 - 739.5004607016537 hold_7_10
   - 672.8385602120198 hold_7_11 - 617.20126020193 hold_7_12
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_0_nic_out_7_nic_in_0: switch_nic_out_7_nic_in_0 = 1 ->
   - 5089.669280600068 hold_0_1 - 3026.150415568619 hold_0_2
   - 2153.180441367374 hold_0_3 - 1671.107130863092 hold_0_4
   - 1365.407481935425 hold_0_5 - 1154.256553003943 hold_0_6
   - 999.665171097569 hold_0_7 - 881.5921573259316 hold_0_8
   - 788.4645856441134 hold_0_9 - 713.1323747193427 hold_0_10
   - 650.9396938908995 hold_0_11 - 598.7245818794369 hold_0_12
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_7_nic_in_1: switch_nic_out_7_nic_in_1 = 1 ->
   - 5089.669280600068 hold_1_1 - 3026.150415568619 hold_1_2
   - 2153.180441367374 hold_1_3 - 1671.107130863092 hold_1_4
   - 1365.407481935425 hold_1_5 - 1154.256553003943 hold_1_6
   - 999.665171097569 hold_1_7 - 881.5921573259316 hold_1_8
   - 788.4645856441134 hold_1_9 - 713.1323747193427 hold_1_10
   - 650.9396938908995 hold_1_11 - 598.7245818794369 hold_1_12
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_7_nic_in_2: switch_nic_out_7_nic_in_2 = 1 ->
   - 5089.669280600068 hold_2_1 - 3026.150415568619 hold_2_2
   - 2153.180441367374 hold_2_3 - 1671.107130863092 hold_2_4
   - 1365.407481935425 hold_2_5 - 1154.256553003943 hold_2_6
   - 999.665171097569 hold_2_7 - 881.5921573259316 hold_2_8
   - 788.4645856441134 hold_2_9 - 713.1323747193427 hold_2_10
   - 650.9396938908995 hold_2_11 - 598.7245818794369 hold_2_12
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_7_nic_in_3: switch_nic_out_7_nic_in_3 = 1 ->
   - 5089.669280600068 hold_3_1 - 3026.150415568619 hold_3_2
   - 2153.180441367374 hold_3_3 - 1671.107130863092 hold_3_4
   - 1365.407481935425 hold_3_5 - 1154.256553003943 hold_3_6
   - 999.665171097569 hold_3_7 - 881.5921573259316 hold_3_8
   - 788.4645856441134 hold_3_9 - 713.1323747193427 hold_3_10
   - 650.9396938908995 hold_3_11 - 598.7245818794369 hold_3_12
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_7_nic_in_4: switch_nic_out_7_nic_in_4 = 1 ->
   - 5089.669280600068 hold_4_1 - 3026.150415568619 hold_4_2
   - 2153.180441367374 hold_4_3 - 1671.107130863092 hold_4_4
   - 1365.407481935425 hold_4_5 - 1154.256553003943 hold_4_6
   - 999.665171097569 hold_4_7 - 881.5921573259316 hold_4_8
   - 788.4645856441134 hold_4_9 - 713.1323747193427 hold_4_10
   - 650.9396938908995 hold_4_11 - 598.7245818794369 hold_4_12
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_7_nic_in_5: switch_nic_out_7_nic_in_5 = 1 ->
   - 5089.669280600068 hold_5_1 - 3026.150415568619 hold_5_2
   - 2153.180441367374 hold_5_3 - 1671.107130863092 hold_5_4
   - 1365.407481935425 hold_5_5 - 1154.256553003943 hold_5_6
   - 999.665171097569 hold_5_7 - 881.5921573259316 hold_5_8
   - 788.4645856441134 hold_5_9 - 713.1323747193427 hold_5_10
   - 650.9396938908995 hold_5_11 - 598.7245818794369 hold_5_12
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_7_nic_in_6: switch_nic_out_7_nic_in_6 = 1 ->
   - 5089.669280600068 hold_6_1 - 3026.150415568619 hold_6_2
   - 2153.180441367374 hold_6_3 - 1671.107130863092 hold_6_4
   - 1365.407481935425 hold_6_5 - 1154.256553003943 hold_6_6
   - 999.665171097569 hold_6_7 - 881.5921573259316 hold_6_8
   - 788.4645856441134 hold_6_9 - 713.1323747193427 hold_6_10
   - 650.9396938908995 hold_6_11 - 598.7245818794369 hold_6_12
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_8_nic_out_7_nic_in_8: switch_nic_out_7_nic_in_8 = 1 ->
   - 5089.669280600068 hold_8_1 - 3026.150415568619 hold_8_2
   - 2153.180441367374 hold_8_3 - 1671.107130863092 hold_8_4
   - 1365.407481935425 hold_8_5 - 1154.256553003943 hold_8_6
   - 999.665171097569 hold_8_7 - 881.5921573259316 hold_8_8
   - 788.4645856441134 hold_8_9 - 713.1323747193427 hold_8_10
   - 650.9396938908995 hold_8_11 - 598.7245818794369 hold_8_12
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_7_nic_in_9: switch_nic_out_7_nic_in_9 = 1 ->
   - 5089.669280600068 hold_9_1 - 3026.150415568619 hold_9_2
   - 2153.180441367374 hold_9_3 - 1671.107130863092 hold_9_4
   - 1365.407481935425 hold_9_5 - 1154.256553003943 hold_9_6
   - 999.665171097569 hold_9_7 - 881.5921573259316 hold_9_8
   - 788.4645856441134 hold_9_9 - 713.1323747193427 hold_9_10
   - 650.9396938908995 hold_9_11 - 598.7245818794369 hold_9_12
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_7_nic_in_10: switch_nic_out_7_nic_in_10 = 1 ->
   - 5089.669280600068 hold_10_1 - 3026.150415568619 hold_10_2
   - 2153.180441367374 hold_10_3 - 1671.107130863092 hold_10_4
   - 1365.407481935425 hold_10_5 - 1154.256553003943 hold_10_6
   - 999.665171097569 hold_10_7 - 881.5921573259316 hold_10_8
   - 788.4645856441134 hold_10_9 - 713.1323747193427 hold_10_10
   - 650.9396938908995 hold_10_11 - 598.7245818794369 hold_10_12
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_7_nic_in_11: switch_nic_out_7_nic_in_11 = 1 ->
   - 5089.669280600068 hold_11_1 - 3026.150415568619 hold_11_2
   - 2153.180441367374 hold_11_3 - 1671.107130863092 hold_11_4
   - 1365.407481935425 hold_11_5 - 1154.256553003943 hold_11_6
   - 999.665171097569 hold_11_7 - 881.5921573259316 hold_11_8
   - 788.4645856441134 hold_11_9 - 713.1323747193427 hold_11_10
   - 650.9396938908995 hold_11_11 - 598.7245818794369 hold_11_12
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_7_nic_in_12: switch_nic_out_7_nic_in_12 = 1 ->
   - 5089.669280600068 hold_12_1 - 3026.150415568619 hold_12_2
   - 2153.180441367374 hold_12_3 - 1671.107130863092 hold_12_4
   - 1365.407481935425 hold_12_5 - 1154.256553003943 hold_12_6
   - 999.665171097569 hold_12_7 - 881.5921573259316 hold_12_8
   - 788.4645856441134 hold_12_9 - 713.1323747193427 hold_12_10
   - 650.9396938908995 hold_12_11 - 598.7245818794369 hold_12_12
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_7_nic_in_13: switch_nic_out_7_nic_in_13 = 1 ->
   - 5089.669280600068 hold_13_1 - 3026.150415568619 hold_13_2
   - 2153.180441367374 hold_13_3 - 1671.107130863092 hold_13_4
   - 1365.407481935425 hold_13_5 - 1154.256553003943 hold_13_6
   - 999.665171097569 hold_13_7 - 881.5921573259316 hold_13_8
   - 788.4645856441134 hold_13_9 - 713.1323747193427 hold_13_10
   - 650.9396938908995 hold_13_11 - 598.7245818794369 hold_13_12
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_7_nic_in_14: switch_nic_out_7_nic_in_14 = 1 ->
   - 5089.669280600068 hold_14_1 - 3026.150415568619 hold_14_2
   - 2153.180441367374 hold_14_3 - 1671.107130863092 hold_14_4
   - 1365.407481935425 hold_14_5 - 1154.256553003943 hold_14_6
   - 999.665171097569 hold_14_7 - 881.5921573259316 hold_14_8
   - 788.4645856441134 hold_14_9 - 713.1323747193427 hold_14_10
   - 650.9396938908995 hold_14_11 - 598.7245818794369 hold_14_12
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_7_nic_in_15: switch_nic_out_7_nic_in_15 = 1 ->
   - 5089.669280600068 hold_15_1 - 3026.150415568619 hold_15_2
   - 2153.180441367374 hold_15_3 - 1671.107130863092 hold_15_4
   - 1365.407481935425 hold_15_5 - 1154.256553003943 hold_15_6
   - 999.665171097569 hold_15_7 - 881.5921573259316 hold_15_8
   - 788.4645856441134 hold_15_9 - 713.1323747193427 hold_15_10
   - 650.9396938908995 hold_15_11 - 598.7245818794369 hold_15_12
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_8_source_nic_in_8: switch_source_nic_in_8 = 1 ->
   - 6827.037409677124 hold_8_1 - 3565.661873596714 hold_8_2
   - 2412.957036174959 hold_8_3 - 1823.467617814477 hold_8_4
   - 1465.454616849587 hold_8_5 - 1224.951996455123 hold_8_6
   - 1052.260583931315 hold_8_7 - 922.2443255532355 hold_8_8
   - 820.8241232555838 hold_8_9 - 739.5004607016537 hold_8_10
   - 672.8385602120198 hold_8_11 - 617.20126020193 hold_8_12
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_0_nic_out_8_nic_in_0: switch_nic_out_8_nic_in_0 = 1 ->
   - 5089.669280600068 hold_0_1 - 3026.150415568619 hold_0_2
   - 2153.180441367374 hold_0_3 - 1671.107130863092 hold_0_4
   - 1365.407481935425 hold_0_5 - 1154.256553003943 hold_0_6
   - 999.665171097569 hold_0_7 - 881.5921573259316 hold_0_8
   - 788.4645856441134 hold_0_9 - 713.1323747193427 hold_0_10
   - 650.9396938908995 hold_0_11 - 598.7245818794369 hold_0_12
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_8_nic_in_1: switch_nic_out_8_nic_in_1 = 1 ->
   - 5089.669280600068 hold_1_1 - 3026.150415568619 hold_1_2
   - 2153.180441367374 hold_1_3 - 1671.107130863092 hold_1_4
   - 1365.407481935425 hold_1_5 - 1154.256553003943 hold_1_6
   - 999.665171097569 hold_1_7 - 881.5921573259316 hold_1_8
   - 788.4645856441134 hold_1_9 - 713.1323747193427 hold_1_10
   - 650.9396938908995 hold_1_11 - 598.7245818794369 hold_1_12
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_8_nic_in_2: switch_nic_out_8_nic_in_2 = 1 ->
   - 5089.669280600068 hold_2_1 - 3026.150415568619 hold_2_2
   - 2153.180441367374 hold_2_3 - 1671.107130863092 hold_2_4
   - 1365.407481935425 hold_2_5 - 1154.256553003943 hold_2_6
   - 999.665171097569 hold_2_7 - 881.5921573259316 hold_2_8
   - 788.4645856441134 hold_2_9 - 713.1323747193427 hold_2_10
   - 650.9396938908995 hold_2_11 - 598.7245818794369 hold_2_12
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_8_nic_in_3: switch_nic_out_8_nic_in_3 = 1 ->
   - 5089.669280600068 hold_3_1 - 3026.150415568619 hold_3_2
   - 2153.180441367374 hold_3_3 - 1671.107130863092 hold_3_4
   - 1365.407481935425 hold_3_5 - 1154.256553003943 hold_3_6
   - 999.665171097569 hold_3_7 - 881.5921573259316 hold_3_8
   - 788.4645856441134 hold_3_9 - 713.1323747193427 hold_3_10
   - 650.9396938908995 hold_3_11 - 598.7245818794369 hold_3_12
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_8_nic_in_4: switch_nic_out_8_nic_in_4 = 1 ->
   - 5089.669280600068 hold_4_1 - 3026.150415568619 hold_4_2
   - 2153.180441367374 hold_4_3 - 1671.107130863092 hold_4_4
   - 1365.407481935425 hold_4_5 - 1154.256553003943 hold_4_6
   - 999.665171097569 hold_4_7 - 881.5921573259316 hold_4_8
   - 788.4645856441134 hold_4_9 - 713.1323747193427 hold_4_10
   - 650.9396938908995 hold_4_11 - 598.7245818794369 hold_4_12
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_8_nic_in_5: switch_nic_out_8_nic_in_5 = 1 ->
   - 5089.669280600068 hold_5_1 - 3026.150415568619 hold_5_2
   - 2153.180441367374 hold_5_3 - 1671.107130863092 hold_5_4
   - 1365.407481935425 hold_5_5 - 1154.256553003943 hold_5_6
   - 999.665171097569 hold_5_7 - 881.5921573259316 hold_5_8
   - 788.4645856441134 hold_5_9 - 713.1323747193427 hold_5_10
   - 650.9396938908995 hold_5_11 - 598.7245818794369 hold_5_12
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_8_nic_in_6: switch_nic_out_8_nic_in_6 = 1 ->
   - 5089.669280600068 hold_6_1 - 3026.150415568619 hold_6_2
   - 2153.180441367374 hold_6_3 - 1671.107130863092 hold_6_4
   - 1365.407481935425 hold_6_5 - 1154.256553003943 hold_6_6
   - 999.665171097569 hold_6_7 - 881.5921573259316 hold_6_8
   - 788.4645856441134 hold_6_9 - 713.1323747193427 hold_6_10
   - 650.9396938908995 hold_6_11 - 598.7245818794369 hold_6_12
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_8_nic_in_7: switch_nic_out_8_nic_in_7 = 1 ->
   - 5089.669280600068 hold_7_1 - 3026.150415568619 hold_7_2
   - 2153.180441367374 hold_7_3 - 1671.107130863092 hold_7_4
   - 1365.407481935425 hold_7_5 - 1154.256553003943 hold_7_6
   - 999.665171097569 hold_7_7 - 881.5921573259316 hold_7_8
   - 788.4645856441134 hold_7_9 - 713.1323747193427 hold_7_10
   - 650.9396938908995 hold_7_11 - 598.7245818794369 hold_7_12
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_9_nic_out_8_nic_in_9: switch_nic_out_8_nic_in_9 = 1 ->
   - 5089.669280600068 hold_9_1 - 3026.150415568619 hold_9_2
   - 2153.180441367374 hold_9_3 - 1671.107130863092 hold_9_4
   - 1365.407481935425 hold_9_5 - 1154.256553003943 hold_9_6
   - 999.665171097569 hold_9_7 - 881.5921573259316 hold_9_8
   - 788.4645856441134 hold_9_9 - 713.1323747193427 hold_9_10
   - 650.9396938908995 hold_9_11 - 598.7245818794369 hold_9_12
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_8_nic_in_10: switch_nic_out_8_nic_in_10 = 1 ->
   - 5089.669280600068 hold_10_1 - 3026.150415568619 hold_10_2
   - 2153.180441367374 hold_10_3 - 1671.107130863092 hold_10_4
   - 1365.407481935425 hold_10_5 - 1154.256553003943 hold_10_6
   - 999.665171097569 hold_10_7 - 881.5921573259316 hold_10_8
   - 788.4645856441134 hold_10_9 - 713.1323747193427 hold_10_10
   - 650.9396938908995 hold_10_11 - 598.7245818794369 hold_10_12
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_8_nic_in_11: switch_nic_out_8_nic_in_11 = 1 ->
   - 5089.669280600068 hold_11_1 - 3026.150415568619 hold_11_2
   - 2153.180441367374 hold_11_3 - 1671.107130863092 hold_11_4
   - 1365.407481935425 hold_11_5 - 1154.256553003943 hold_11_6
   - 999.665171097569 hold_11_7 - 881.5921573259316 hold_11_8
   - 788.4645856441134 hold_11_9 - 713.1323747193427 hold_11_10
   - 650.9396938908995 hold_11_11 - 598.7245818794369 hold_11_12
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_8_nic_in_12: switch_nic_out_8_nic_in_12 = 1 ->
   - 5089.669280600068 hold_12_1 - 3026.150415568619 hold_12_2
   - 2153.180441367374 hold_12_3 - 1671.107130863092 hold_12_4
   - 1365.407481935425 hold_12_5 - 1154.256553003943 hold_12_6
   - 999.665171097569 hold_12_7 - 881.5921573259316 hold_12_8
   - 788.4645856441134 hold_12_9 - 713.1323747193427 hold_12_10
   - 650.9396938908995 hold_12_11 - 598.7245818794369 hold_12_12
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_8_nic_in_13: switch_nic_out_8_nic_in_13 = 1 ->
   - 5089.669280600068 hold_13_1 - 3026.150415568619 hold_13_2
   - 2153.180441367374 hold_13_3 - 1671.107130863092 hold_13_4
   - 1365.407481935425 hold_13_5 - 1154.256553003943 hold_13_6
   - 999.665171097569 hold_13_7 - 881.5921573259316 hold_13_8
   - 788.4645856441134 hold_13_9 - 713.1323747193427 hold_13_10
   - 650.9396938908995 hold_13_11 - 598.7245818794369 hold_13_12
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_8_nic_in_14: switch_nic_out_8_nic_in_14 = 1 ->
   - 5089.669280600068 hold_14_1 - 3026.150415568619 hold_14_2
   - 2153.180441367374 hold_14_3 - 1671.107130863092 hold_14_4
   - 1365.407481935425 hold_14_5 - 1154.256553003943 hold_14_6
   - 999.665171097569 hold_14_7 - 881.5921573259316 hold_14_8
   - 788.4645856441134 hold_14_9 - 713.1323747193427 hold_14_10
   - 650.9396938908995 hold_14_11 - 598.7245818794369 hold_14_12
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_8_nic_in_15: switch_nic_out_8_nic_in_15 = 1 ->
   - 5089.669280600068 hold_15_1 - 3026.150415568619 hold_15_2
   - 2153.180441367374 hold_15_3 - 1671.107130863092 hold_15_4
   - 1365.407481935425 hold_15_5 - 1154.256553003943 hold_15_6
   - 999.665171097569 hold_15_7 - 881.5921573259316 hold_15_8
   - 788.4645856441134 hold_15_9 - 713.1323747193427 hold_15_10
   - 650.9396938908995 hold_15_11 - 598.7245818794369 hold_15_12
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_9_source_nic_in_9: switch_source_nic_in_9 = 1 ->
   - 6827.037409677124 hold_9_1 - 3565.661873596714 hold_9_2
   - 2412.957036174959 hold_9_3 - 1823.467617814477 hold_9_4
   - 1465.454616849587 hold_9_5 - 1224.951996455123 hold_9_6
   - 1052.260583931315 hold_9_7 - 922.2443255532355 hold_9_8
   - 820.8241232555838 hold_9_9 - 739.5004607016537 hold_9_10
   - 672.8385602120198 hold_9_11 - 617.20126020193 hold_9_12
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_0_nic_out_9_nic_in_0: switch_nic_out_9_nic_in_0 = 1 ->
   - 5089.669280600068 hold_0_1 - 3026.150415568619 hold_0_2
   - 2153.180441367374 hold_0_3 - 1671.107130863092 hold_0_4
   - 1365.407481935425 hold_0_5 - 1154.256553003943 hold_0_6
   - 999.665171097569 hold_0_7 - 881.5921573259316 hold_0_8
   - 788.4645856441134 hold_0_9 - 713.1323747193427 hold_0_10
   - 650.9396938908995 hold_0_11 - 598.7245818794369 hold_0_12
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_9_nic_in_1: switch_nic_out_9_nic_in_1 = 1 ->
   - 5089.669280600068 hold_1_1 - 3026.150415568619 hold_1_2
   - 2153.180441367374 hold_1_3 - 1671.107130863092 hold_1_4
   - 1365.407481935425 hold_1_5 - 1154.256553003943 hold_1_6
   - 999.665171097569 hold_1_7 - 881.5921573259316 hold_1_8
   - 788.4645856441134 hold_1_9 - 713.1323747193427 hold_1_10
   - 650.9396938908995 hold_1_11 - 598.7245818794369 hold_1_12
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_9_nic_in_2: switch_nic_out_9_nic_in_2 = 1 ->
   - 5089.669280600068 hold_2_1 - 3026.150415568619 hold_2_2
   - 2153.180441367374 hold_2_3 - 1671.107130863092 hold_2_4
   - 1365.407481935425 hold_2_5 - 1154.256553003943 hold_2_6
   - 999.665171097569 hold_2_7 - 881.5921573259316 hold_2_8
   - 788.4645856441134 hold_2_9 - 713.1323747193427 hold_2_10
   - 650.9396938908995 hold_2_11 - 598.7245818794369 hold_2_12
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_9_nic_in_3: switch_nic_out_9_nic_in_3 = 1 ->
   - 5089.669280600068 hold_3_1 - 3026.150415568619 hold_3_2
   - 2153.180441367374 hold_3_3 - 1671.107130863092 hold_3_4
   - 1365.407481935425 hold_3_5 - 1154.256553003943 hold_3_6
   - 999.665171097569 hold_3_7 - 881.5921573259316 hold_3_8
   - 788.4645856441134 hold_3_9 - 713.1323747193427 hold_3_10
   - 650.9396938908995 hold_3_11 - 598.7245818794369 hold_3_12
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_9_nic_in_4: switch_nic_out_9_nic_in_4 = 1 ->
   - 5089.669280600068 hold_4_1 - 3026.150415568619 hold_4_2
   - 2153.180441367374 hold_4_3 - 1671.107130863092 hold_4_4
   - 1365.407481935425 hold_4_5 - 1154.256553003943 hold_4_6
   - 999.665171097569 hold_4_7 - 881.5921573259316 hold_4_8
   - 788.4645856441134 hold_4_9 - 713.1323747193427 hold_4_10
   - 650.9396938908995 hold_4_11 - 598.7245818794369 hold_4_12
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_9_nic_in_5: switch_nic_out_9_nic_in_5 = 1 ->
   - 5089.669280600068 hold_5_1 - 3026.150415568619 hold_5_2
   - 2153.180441367374 hold_5_3 - 1671.107130863092 hold_5_4
   - 1365.407481935425 hold_5_5 - 1154.256553003943 hold_5_6
   - 999.665171097569 hold_5_7 - 881.5921573259316 hold_5_8
   - 788.4645856441134 hold_5_9 - 713.1323747193427 hold_5_10
   - 650.9396938908995 hold_5_11 - 598.7245818794369 hold_5_12
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_9_nic_in_6: switch_nic_out_9_nic_in_6 = 1 ->
   - 5089.669280600068 hold_6_1 - 3026.150415568619 hold_6_2
   - 2153.180441367374 hold_6_3 - 1671.107130863092 hold_6_4
   - 1365.407481935425 hold_6_5 - 1154.256553003943 hold_6_6
   - 999.665171097569 hold_6_7 - 881.5921573259316 hold_6_8
   - 788.4645856441134 hold_6_9 - 713.1323747193427 hold_6_10
   - 650.9396938908995 hold_6_11 - 598.7245818794369 hold_6_12
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_9_nic_in_7: switch_nic_out_9_nic_in_7 = 1 ->
   - 5089.669280600068 hold_7_1 - 3026.150415568619 hold_7_2
   - 2153.180441367374 hold_7_3 - 1671.107130863092 hold_7_4
   - 1365.407481935425 hold_7_5 - 1154.256553003943 hold_7_6
   - 999.665171097569 hold_7_7 - 881.5921573259316 hold_7_8
   - 788.4645856441134 hold_7_9 - 713.1323747193427 hold_7_10
   - 650.9396938908995 hold_7_11 - 598.7245818794369 hold_7_12
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_9_nic_in_8: switch_nic_out_9_nic_in_8 = 1 ->
   - 5089.669280600068 hold_8_1 - 3026.150415568619 hold_8_2
   - 2153.180441367374 hold_8_3 - 1671.107130863092 hold_8_4
   - 1365.407481935425 hold_8_5 - 1154.256553003943 hold_8_6
   - 999.665171097569 hold_8_7 - 881.5921573259316 hold_8_8
   - 788.4645856441134 hold_8_9 - 713.1323747193427 hold_8_10
   - 650.9396938908995 hold_8_11 - 598.7245818794369 hold_8_12
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_10_nic_out_9_nic_in_10: switch_nic_out_9_nic_in_10 = 1 ->
   - 5089.669280600068 hold_10_1 - 3026.150415568619 hold_10_2
   - 2153.180441367374 hold_10_3 - 1671.107130863092 hold_10_4
   - 1365.407481935425 hold_10_5 - 1154.256553003943 hold_10_6
   - 999.665171097569 hold_10_7 - 881.5921573259316 hold_10_8
   - 788.4645856441134 hold_10_9 - 713.1323747193427 hold_10_10
   - 650.9396938908995 hold_10_11 - 598.7245818794369 hold_10_12
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_9_nic_in_11: switch_nic_out_9_nic_in_11 = 1 ->
   - 5089.669280600068 hold_11_1 - 3026.150415568619 hold_11_2
   - 2153.180441367374 hold_11_3 - 1671.107130863092 hold_11_4
   - 1365.407481935425 hold_11_5 - 1154.256553003943 hold_11_6
   - 999.665171097569 hold_11_7 - 881.5921573259316 hold_11_8
   - 788.4645856441134 hold_11_9 - 713.1323747193427 hold_11_10
   - 650.9396938908995 hold_11_11 - 598.7245818794369 hold_11_12
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_9_nic_in_12: switch_nic_out_9_nic_in_12 = 1 ->
   - 5089.669280600068 hold_12_1 - 3026.150415568619 hold_12_2
   - 2153.180441367374 hold_12_3 - 1671.107130863092 hold_12_4
   - 1365.407481935425 hold_12_5 - 1154.256553003943 hold_12_6
   - 999.665171097569 hold_12_7 - 881.5921573259316 hold_12_8
   - 788.4645856441134 hold_12_9 - 713.1323747193427 hold_12_10
   - 650.9396938908995 hold_12_11 - 598.7245818794369 hold_12_12
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_9_nic_in_13: switch_nic_out_9_nic_in_13 = 1 ->
   - 5089.669280600068 hold_13_1 - 3026.150415568619 hold_13_2
   - 2153.180441367374 hold_13_3 - 1671.107130863092 hold_13_4
   - 1365.407481935425 hold_13_5 - 1154.256553003943 hold_13_6
   - 999.665171097569 hold_13_7 - 881.5921573259316 hold_13_8
   - 788.4645856441134 hold_13_9 - 713.1323747193427 hold_13_10
   - 650.9396938908995 hold_13_11 - 598.7245818794369 hold_13_12
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_9_nic_in_14: switch_nic_out_9_nic_in_14 = 1 ->
   - 5089.669280600068 hold_14_1 - 3026.150415568619 hold_14_2
   - 2153.180441367374 hold_14_3 - 1671.107130863092 hold_14_4
   - 1365.407481935425 hold_14_5 - 1154.256553003943 hold_14_6
   - 999.665171097569 hold_14_7 - 881.5921573259316 hold_14_8
   - 788.4645856441134 hold_14_9 - 713.1323747193427 hold_14_10
   - 650.9396938908995 hold_14_11 - 598.7245818794369 hold_14_12
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_9_nic_in_15: switch_nic_out_9_nic_in_15 = 1 ->
   - 5089.669280600068 hold_15_1 - 3026.150415568619 hold_15_2
   - 2153.180441367374 hold_15_3 - 1671.107130863092 hold_15_4
   - 1365.407481935425 hold_15_5 - 1154.256553003943 hold_15_6
   - 999.665171097569 hold_15_7 - 881.5921573259316 hold_15_8
   - 788.4645856441134 hold_15_9 - 713.1323747193427 hold_15_10
   - 650.9396938908995 hold_15_11 - 598.7245818794369 hold_15_12
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_10_source_nic_in_10: switch_source_nic_in_10 = 1 ->
   - 6827.037409677124 hold_10_1 - 3565.661873596714 hold_10_2
   - 2412.957036174959 hold_10_3 - 1823.467617814477 hold_10_4
   - 1465.454616849587 hold_10_5 - 1224.951996455123 hold_10_6
   - 1052.260583931315 hold_10_7 - 922.2443255532355 hold_10_8
   - 820.8241232555838 hold_10_9 - 739.5004607016537 hold_10_10
   - 672.8385602120198 hold_10_11 - 617.20126020193 hold_10_12
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_0_nic_out_10_nic_in_0: switch_nic_out_10_nic_in_0 = 1 ->
   - 5089.669280600068 hold_0_1 - 3026.150415568619 hold_0_2
   - 2153.180441367374 hold_0_3 - 1671.107130863092 hold_0_4
   - 1365.407481935425 hold_0_5 - 1154.256553003943 hold_0_6
   - 999.665171097569 hold_0_7 - 881.5921573259316 hold_0_8
   - 788.4645856441134 hold_0_9 - 713.1323747193427 hold_0_10
   - 650.9396938908995 hold_0_11 - 598.7245818794369 hold_0_12
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_10_nic_in_1: switch_nic_out_10_nic_in_1 = 1 ->
   - 5089.669280600068 hold_1_1 - 3026.150415568619 hold_1_2
   - 2153.180441367374 hold_1_3 - 1671.107130863092 hold_1_4
   - 1365.407481935425 hold_1_5 - 1154.256553003943 hold_1_6
   - 999.665171097569 hold_1_7 - 881.5921573259316 hold_1_8
   - 788.4645856441134 hold_1_9 - 713.1323747193427 hold_1_10
   - 650.9396938908995 hold_1_11 - 598.7245818794369 hold_1_12
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_10_nic_in_2: switch_nic_out_10_nic_in_2 = 1 ->
   - 5089.669280600068 hold_2_1 - 3026.150415568619 hold_2_2
   - 2153.180441367374 hold_2_3 - 1671.107130863092 hold_2_4
   - 1365.407481935425 hold_2_5 - 1154.256553003943 hold_2_6
   - 999.665171097569 hold_2_7 - 881.5921573259316 hold_2_8
   - 788.4645856441134 hold_2_9 - 713.1323747193427 hold_2_10
   - 650.9396938908995 hold_2_11 - 598.7245818794369 hold_2_12
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_10_nic_in_3: switch_nic_out_10_nic_in_3 = 1 ->
   - 5089.669280600068 hold_3_1 - 3026.150415568619 hold_3_2
   - 2153.180441367374 hold_3_3 - 1671.107130863092 hold_3_4
   - 1365.407481935425 hold_3_5 - 1154.256553003943 hold_3_6
   - 999.665171097569 hold_3_7 - 881.5921573259316 hold_3_8
   - 788.4645856441134 hold_3_9 - 713.1323747193427 hold_3_10
   - 650.9396938908995 hold_3_11 - 598.7245818794369 hold_3_12
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_10_nic_in_4: switch_nic_out_10_nic_in_4 = 1 ->
   - 5089.669280600068 hold_4_1 - 3026.150415568619 hold_4_2
   - 2153.180441367374 hold_4_3 - 1671.107130863092 hold_4_4
   - 1365.407481935425 hold_4_5 - 1154.256553003943 hold_4_6
   - 999.665171097569 hold_4_7 - 881.5921573259316 hold_4_8
   - 788.4645856441134 hold_4_9 - 713.1323747193427 hold_4_10
   - 650.9396938908995 hold_4_11 - 598.7245818794369 hold_4_12
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_10_nic_in_5: switch_nic_out_10_nic_in_5 = 1 ->
   - 5089.669280600068 hold_5_1 - 3026.150415568619 hold_5_2
   - 2153.180441367374 hold_5_3 - 1671.107130863092 hold_5_4
   - 1365.407481935425 hold_5_5 - 1154.256553003943 hold_5_6
   - 999.665171097569 hold_5_7 - 881.5921573259316 hold_5_8
   - 788.4645856441134 hold_5_9 - 713.1323747193427 hold_5_10
   - 650.9396938908995 hold_5_11 - 598.7245818794369 hold_5_12
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_10_nic_in_6: switch_nic_out_10_nic_in_6 = 1 ->
   - 5089.669280600068 hold_6_1 - 3026.150415568619 hold_6_2
   - 2153.180441367374 hold_6_3 - 1671.107130863092 hold_6_4
   - 1365.407481935425 hold_6_5 - 1154.256553003943 hold_6_6
   - 999.665171097569 hold_6_7 - 881.5921573259316 hold_6_8
   - 788.4645856441134 hold_6_9 - 713.1323747193427 hold_6_10
   - 650.9396938908995 hold_6_11 - 598.7245818794369 hold_6_12
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_10_nic_in_7: switch_nic_out_10_nic_in_7 = 1 ->
   - 5089.669280600068 hold_7_1 - 3026.150415568619 hold_7_2
   - 2153.180441367374 hold_7_3 - 1671.107130863092 hold_7_4
   - 1365.407481935425 hold_7_5 - 1154.256553003943 hold_7_6
   - 999.665171097569 hold_7_7 - 881.5921573259316 hold_7_8
   - 788.4645856441134 hold_7_9 - 713.1323747193427 hold_7_10
   - 650.9396938908995 hold_7_11 - 598.7245818794369 hold_7_12
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_10_nic_in_8: switch_nic_out_10_nic_in_8 = 1 ->
   - 5089.669280600068 hold_8_1 - 3026.150415568619 hold_8_2
   - 2153.180441367374 hold_8_3 - 1671.107130863092 hold_8_4
   - 1365.407481935425 hold_8_5 - 1154.256553003943 hold_8_6
   - 999.665171097569 hold_8_7 - 881.5921573259316 hold_8_8
   - 788.4645856441134 hold_8_9 - 713.1323747193427 hold_8_10
   - 650.9396938908995 hold_8_11 - 598.7245818794369 hold_8_12
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_10_nic_in_9: switch_nic_out_10_nic_in_9 = 1 ->
   - 5089.669280600068 hold_9_1 - 3026.150415568619 hold_9_2
   - 2153.180441367374 hold_9_3 - 1671.107130863092 hold_9_4
   - 1365.407481935425 hold_9_5 - 1154.256553003943 hold_9_6
   - 999.665171097569 hold_9_7 - 881.5921573259316 hold_9_8
   - 788.4645856441134 hold_9_9 - 713.1323747193427 hold_9_10
   - 650.9396938908995 hold_9_11 - 598.7245818794369 hold_9_12
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_11_nic_out_10_nic_in_11: switch_nic_out_10_nic_in_11 = 1 ->
   - 5089.669280600068 hold_11_1 - 3026.150415568619 hold_11_2
   - 2153.180441367374 hold_11_3 - 1671.107130863092 hold_11_4
   - 1365.407481935425 hold_11_5 - 1154.256553003943 hold_11_6
   - 999.665171097569 hold_11_7 - 881.5921573259316 hold_11_8
   - 788.4645856441134 hold_11_9 - 713.1323747193427 hold_11_10
   - 650.9396938908995 hold_11_11 - 598.7245818794369 hold_11_12
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_10_nic_in_12: switch_nic_out_10_nic_in_12 = 1 ->
   - 5089.669280600068 hold_12_1 - 3026.150415568619 hold_12_2
   - 2153.180441367374 hold_12_3 - 1671.107130863092 hold_12_4
   - 1365.407481935425 hold_12_5 - 1154.256553003943 hold_12_6
   - 999.665171097569 hold_12_7 - 881.5921573259316 hold_12_8
   - 788.4645856441134 hold_12_9 - 713.1323747193427 hold_12_10
   - 650.9396938908995 hold_12_11 - 598.7245818794369 hold_12_12
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_10_nic_in_13: switch_nic_out_10_nic_in_13 = 1 ->
   - 5089.669280600068 hold_13_1 - 3026.150415568619 hold_13_2
   - 2153.180441367374 hold_13_3 - 1671.107130863092 hold_13_4
   - 1365.407481935425 hold_13_5 - 1154.256553003943 hold_13_6
   - 999.665171097569 hold_13_7 - 881.5921573259316 hold_13_8
   - 788.4645856441134 hold_13_9 - 713.1323747193427 hold_13_10
   - 650.9396938908995 hold_13_11 - 598.7245818794369 hold_13_12
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_10_nic_in_14: switch_nic_out_10_nic_in_14 = 1 ->
   - 5089.669280600068 hold_14_1 - 3026.150415568619 hold_14_2
   - 2153.180441367374 hold_14_3 - 1671.107130863092 hold_14_4
   - 1365.407481935425 hold_14_5 - 1154.256553003943 hold_14_6
   - 999.665171097569 hold_14_7 - 881.5921573259316 hold_14_8
   - 788.4645856441134 hold_14_9 - 713.1323747193427 hold_14_10
   - 650.9396938908995 hold_14_11 - 598.7245818794369 hold_14_12
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_10_nic_in_15: switch_nic_out_10_nic_in_15 = 1 ->
   - 5089.669280600068 hold_15_1 - 3026.150415568619 hold_15_2
   - 2153.180441367374 hold_15_3 - 1671.107130863092 hold_15_4
   - 1365.407481935425 hold_15_5 - 1154.256553003943 hold_15_6
   - 999.665171097569 hold_15_7 - 881.5921573259316 hold_15_8
   - 788.4645856441134 hold_15_9 - 713.1323747193427 hold_15_10
   - 650.9396938908995 hold_15_11 - 598.7245818794369 hold_15_12
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_11_source_nic_in_11: switch_source_nic_in_11 = 1 ->
   - 6827.037409677124 hold_11_1 - 3565.661873596714 hold_11_2
   - 2412.957036174959 hold_11_3 - 1823.467617814477 hold_11_4
   - 1465.454616849587 hold_11_5 - 1224.951996455123 hold_11_6
   - 1052.260583931315 hold_11_7 - 922.2443255532355 hold_11_8
   - 820.8241232555838 hold_11_9 - 739.5004607016537 hold_11_10
   - 672.8385602120198 hold_11_11 - 617.20126020193 hold_11_12
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_0_nic_out_11_nic_in_0: switch_nic_out_11_nic_in_0 = 1 ->
   - 5089.669280600068 hold_0_1 - 3026.150415568619 hold_0_2
   - 2153.180441367374 hold_0_3 - 1671.107130863092 hold_0_4
   - 1365.407481935425 hold_0_5 - 1154.256553003943 hold_0_6
   - 999.665171097569 hold_0_7 - 881.5921573259316 hold_0_8
   - 788.4645856441134 hold_0_9 - 713.1323747193427 hold_0_10
   - 650.9396938908995 hold_0_11 - 598.7245818794369 hold_0_12
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_11_nic_in_1: switch_nic_out_11_nic_in_1 = 1 ->
   - 5089.669280600068 hold_1_1 - 3026.150415568619 hold_1_2
   - 2153.180441367374 hold_1_3 - 1671.107130863092 hold_1_4
   - 1365.407481935425 hold_1_5 - 1154.256553003943 hold_1_6
   - 999.665171097569 hold_1_7 - 881.5921573259316 hold_1_8
   - 788.4645856441134 hold_1_9 - 713.1323747193427 hold_1_10
   - 650.9396938908995 hold_1_11 - 598.7245818794369 hold_1_12
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_11_nic_in_2: switch_nic_out_11_nic_in_2 = 1 ->
   - 5089.669280600068 hold_2_1 - 3026.150415568619 hold_2_2
   - 2153.180441367374 hold_2_3 - 1671.107130863092 hold_2_4
   - 1365.407481935425 hold_2_5 - 1154.256553003943 hold_2_6
   - 999.665171097569 hold_2_7 - 881.5921573259316 hold_2_8
   - 788.4645856441134 hold_2_9 - 713.1323747193427 hold_2_10
   - 650.9396938908995 hold_2_11 - 598.7245818794369 hold_2_12
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_11_nic_in_3: switch_nic_out_11_nic_in_3 = 1 ->
   - 5089.669280600068 hold_3_1 - 3026.150415568619 hold_3_2
   - 2153.180441367374 hold_3_3 - 1671.107130863092 hold_3_4
   - 1365.407481935425 hold_3_5 - 1154.256553003943 hold_3_6
   - 999.665171097569 hold_3_7 - 881.5921573259316 hold_3_8
   - 788.4645856441134 hold_3_9 - 713.1323747193427 hold_3_10
   - 650.9396938908995 hold_3_11 - 598.7245818794369 hold_3_12
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_11_nic_in_4: switch_nic_out_11_nic_in_4 = 1 ->
   - 5089.669280600068 hold_4_1 - 3026.150415568619 hold_4_2
   - 2153.180441367374 hold_4_3 - 1671.107130863092 hold_4_4
   - 1365.407481935425 hold_4_5 - 1154.256553003943 hold_4_6
   - 999.665171097569 hold_4_7 - 881.5921573259316 hold_4_8
   - 788.4645856441134 hold_4_9 - 713.1323747193427 hold_4_10
   - 650.9396938908995 hold_4_11 - 598.7245818794369 hold_4_12
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_11_nic_in_5: switch_nic_out_11_nic_in_5 = 1 ->
   - 5089.669280600068 hold_5_1 - 3026.150415568619 hold_5_2
   - 2153.180441367374 hold_5_3 - 1671.107130863092 hold_5_4
   - 1365.407481935425 hold_5_5 - 1154.256553003943 hold_5_6
   - 999.665171097569 hold_5_7 - 881.5921573259316 hold_5_8
   - 788.4645856441134 hold_5_9 - 713.1323747193427 hold_5_10
   - 650.9396938908995 hold_5_11 - 598.7245818794369 hold_5_12
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_11_nic_in_6: switch_nic_out_11_nic_in_6 = 1 ->
   - 5089.669280600068 hold_6_1 - 3026.150415568619 hold_6_2
   - 2153.180441367374 hold_6_3 - 1671.107130863092 hold_6_4
   - 1365.407481935425 hold_6_5 - 1154.256553003943 hold_6_6
   - 999.665171097569 hold_6_7 - 881.5921573259316 hold_6_8
   - 788.4645856441134 hold_6_9 - 713.1323747193427 hold_6_10
   - 650.9396938908995 hold_6_11 - 598.7245818794369 hold_6_12
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_11_nic_in_7: switch_nic_out_11_nic_in_7 = 1 ->
   - 5089.669280600068 hold_7_1 - 3026.150415568619 hold_7_2
   - 2153.180441367374 hold_7_3 - 1671.107130863092 hold_7_4
   - 1365.407481935425 hold_7_5 - 1154.256553003943 hold_7_6
   - 999.665171097569 hold_7_7 - 881.5921573259316 hold_7_8
   - 788.4645856441134 hold_7_9 - 713.1323747193427 hold_7_10
   - 650.9396938908995 hold_7_11 - 598.7245818794369 hold_7_12
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_11_nic_in_8: switch_nic_out_11_nic_in_8 = 1 ->
   - 5089.669280600068 hold_8_1 - 3026.150415568619 hold_8_2
   - 2153.180441367374 hold_8_3 - 1671.107130863092 hold_8_4
   - 1365.407481935425 hold_8_5 - 1154.256553003943 hold_8_6
   - 999.665171097569 hold_8_7 - 881.5921573259316 hold_8_8
   - 788.4645856441134 hold_8_9 - 713.1323747193427 hold_8_10
   - 650.9396938908995 hold_8_11 - 598.7245818794369 hold_8_12
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_11_nic_in_9: switch_nic_out_11_nic_in_9 = 1 ->
   - 5089.669280600068 hold_9_1 - 3026.150415568619 hold_9_2
   - 2153.180441367374 hold_9_3 - 1671.107130863092 hold_9_4
   - 1365.407481935425 hold_9_5 - 1154.256553003943 hold_9_6
   - 999.665171097569 hold_9_7 - 881.5921573259316 hold_9_8
   - 788.4645856441134 hold_9_9 - 713.1323747193427 hold_9_10
   - 650.9396938908995 hold_9_11 - 598.7245818794369 hold_9_12
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_11_nic_in_10: switch_nic_out_11_nic_in_10 = 1 ->
   - 5089.669280600068 hold_10_1 - 3026.150415568619 hold_10_2
   - 2153.180441367374 hold_10_3 - 1671.107130863092 hold_10_4
   - 1365.407481935425 hold_10_5 - 1154.256553003943 hold_10_6
   - 999.665171097569 hold_10_7 - 881.5921573259316 hold_10_8
   - 788.4645856441134 hold_10_9 - 713.1323747193427 hold_10_10
   - 650.9396938908995 hold_10_11 - 598.7245818794369 hold_10_12
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_12_nic_out_11_nic_in_12: switch_nic_out_11_nic_in_12 = 1 ->
   - 5089.669280600068 hold_12_1 - 3026.150415568619 hold_12_2
   - 2153.180441367374 hold_12_3 - 1671.107130863092 hold_12_4
   - 1365.407481935425 hold_12_5 - 1154.256553003943 hold_12_6
   - 999.665171097569 hold_12_7 - 881.5921573259316 hold_12_8
   - 788.4645856441134 hold_12_9 - 713.1323747193427 hold_12_10
   - 650.9396938908995 hold_12_11 - 598.7245818794369 hold_12_12
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_11_nic_in_13: switch_nic_out_11_nic_in_13 = 1 ->
   - 5089.669280600068 hold_13_1 - 3026.150415568619 hold_13_2
   - 2153.180441367374 hold_13_3 - 1671.107130863092 hold_13_4
   - 1365.407481935425 hold_13_5 - 1154.256553003943 hold_13_6
   - 999.665171097569 hold_13_7 - 881.5921573259316 hold_13_8
   - 788.4645856441134 hold_13_9 - 713.1323747193427 hold_13_10
   - 650.9396938908995 hold_13_11 - 598.7245818794369 hold_13_12
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_11_nic_in_14: switch_nic_out_11_nic_in_14 = 1 ->
   - 5089.669280600068 hold_14_1 - 3026.150415568619 hold_14_2
   - 2153.180441367374 hold_14_3 - 1671.107130863092 hold_14_4
   - 1365.407481935425 hold_14_5 - 1154.256553003943 hold_14_6
   - 999.665171097569 hold_14_7 - 881.5921573259316 hold_14_8
   - 788.4645856441134 hold_14_9 - 713.1323747193427 hold_14_10
   - 650.9396938908995 hold_14_11 - 598.7245818794369 hold_14_12
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_11_nic_in_15: switch_nic_out_11_nic_in_15 = 1 ->
   - 5089.669280600068 hold_15_1 - 3026.150415568619 hold_15_2
   - 2153.180441367374 hold_15_3 - 1671.107130863092 hold_15_4
   - 1365.407481935425 hold_15_5 - 1154.256553003943 hold_15_6
   - 999.665171097569 hold_15_7 - 881.5921573259316 hold_15_8
   - 788.4645856441134 hold_15_9 - 713.1323747193427 hold_15_10
   - 650.9396938908995 hold_15_11 - 598.7245818794369 hold_15_12
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_12_source_nic_in_12: switch_source_nic_in_12 = 1 ->
   - 6827.037409677124 hold_12_1 - 3565.661873596714 hold_12_2
   - 2412.957036174959 hold_12_3 - 1823.467617814477 hold_12_4
   - 1465.454616849587 hold_12_5 - 1224.951996455123 hold_12_6
   - 1052.260583931315 hold_12_7 - 922.2443255532355 hold_12_8
   - 820.8241232555838 hold_12_9 - 739.5004607016537 hold_12_10
   - 672.8385602120198 hold_12_11 - 617.20126020193 hold_12_12
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_0_nic_out_12_nic_in_0: switch_nic_out_12_nic_in_0 = 1 ->
   - 5089.669280600068 hold_0_1 - 3026.150415568619 hold_0_2
   - 2153.180441367374 hold_0_3 - 1671.107130863092 hold_0_4
   - 1365.407481935425 hold_0_5 - 1154.256553003943 hold_0_6
   - 999.665171097569 hold_0_7 - 881.5921573259316 hold_0_8
   - 788.4645856441134 hold_0_9 - 713.1323747193427 hold_0_10
   - 650.9396938908995 hold_0_11 - 598.7245818794369 hold_0_12
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_12_nic_in_1: switch_nic_out_12_nic_in_1 = 1 ->
   - 5089.669280600068 hold_1_1 - 3026.150415568619 hold_1_2
   - 2153.180441367374 hold_1_3 - 1671.107130863092 hold_1_4
   - 1365.407481935425 hold_1_5 - 1154.256553003943 hold_1_6
   - 999.665171097569 hold_1_7 - 881.5921573259316 hold_1_8
   - 788.4645856441134 hold_1_9 - 713.1323747193427 hold_1_10
   - 650.9396938908995 hold_1_11 - 598.7245818794369 hold_1_12
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_12_nic_in_2: switch_nic_out_12_nic_in_2 = 1 ->
   - 5089.669280600068 hold_2_1 - 3026.150415568619 hold_2_2
   - 2153.180441367374 hold_2_3 - 1671.107130863092 hold_2_4
   - 1365.407481935425 hold_2_5 - 1154.256553003943 hold_2_6
   - 999.665171097569 hold_2_7 - 881.5921573259316 hold_2_8
   - 788.4645856441134 hold_2_9 - 713.1323747193427 hold_2_10
   - 650.9396938908995 hold_2_11 - 598.7245818794369 hold_2_12
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_12_nic_in_3: switch_nic_out_12_nic_in_3 = 1 ->
   - 5089.669280600068 hold_3_1 - 3026.150415568619 hold_3_2
   - 2153.180441367374 hold_3_3 - 1671.107130863092 hold_3_4
   - 1365.407481935425 hold_3_5 - 1154.256553003943 hold_3_6
   - 999.665171097569 hold_3_7 - 881.5921573259316 hold_3_8
   - 788.4645856441134 hold_3_9 - 713.1323747193427 hold_3_10
   - 650.9396938908995 hold_3_11 - 598.7245818794369 hold_3_12
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_12_nic_in_4: switch_nic_out_12_nic_in_4 = 1 ->
   - 5089.669280600068 hold_4_1 - 3026.150415568619 hold_4_2
   - 2153.180441367374 hold_4_3 - 1671.107130863092 hold_4_4
   - 1365.407481935425 hold_4_5 - 1154.256553003943 hold_4_6
   - 999.665171097569 hold_4_7 - 881.5921573259316 hold_4_8
   - 788.4645856441134 hold_4_9 - 713.1323747193427 hold_4_10
   - 650.9396938908995 hold_4_11 - 598.7245818794369 hold_4_12
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_12_nic_in_5: switch_nic_out_12_nic_in_5 = 1 ->
   - 5089.669280600068 hold_5_1 - 3026.150415568619 hold_5_2
   - 2153.180441367374 hold_5_3 - 1671.107130863092 hold_5_4
   - 1365.407481935425 hold_5_5 - 1154.256553003943 hold_5_6
   - 999.665171097569 hold_5_7 - 881.5921573259316 hold_5_8
   - 788.4645856441134 hold_5_9 - 713.1323747193427 hold_5_10
   - 650.9396938908995 hold_5_11 - 598.7245818794369 hold_5_12
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_12_nic_in_6: switch_nic_out_12_nic_in_6 = 1 ->
   - 5089.669280600068 hold_6_1 - 3026.150415568619 hold_6_2
   - 2153.180441367374 hold_6_3 - 1671.107130863092 hold_6_4
   - 1365.407481935425 hold_6_5 - 1154.256553003943 hold_6_6
   - 999.665171097569 hold_6_7 - 881.5921573259316 hold_6_8
   - 788.4645856441134 hold_6_9 - 713.1323747193427 hold_6_10
   - 650.9396938908995 hold_6_11 - 598.7245818794369 hold_6_12
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_12_nic_in_7: switch_nic_out_12_nic_in_7 = 1 ->
   - 5089.669280600068 hold_7_1 - 3026.150415568619 hold_7_2
   - 2153.180441367374 hold_7_3 - 1671.107130863092 hold_7_4
   - 1365.407481935425 hold_7_5 - 1154.256553003943 hold_7_6
   - 999.665171097569 hold_7_7 - 881.5921573259316 hold_7_8
   - 788.4645856441134 hold_7_9 - 713.1323747193427 hold_7_10
   - 650.9396938908995 hold_7_11 - 598.7245818794369 hold_7_12
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_12_nic_in_8: switch_nic_out_12_nic_in_8 = 1 ->
   - 5089.669280600068 hold_8_1 - 3026.150415568619 hold_8_2
   - 2153.180441367374 hold_8_3 - 1671.107130863092 hold_8_4
   - 1365.407481935425 hold_8_5 - 1154.256553003943 hold_8_6
   - 999.665171097569 hold_8_7 - 881.5921573259316 hold_8_8
   - 788.4645856441134 hold_8_9 - 713.1323747193427 hold_8_10
   - 650.9396938908995 hold_8_11 - 598.7245818794369 hold_8_12
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_12_nic_in_9: switch_nic_out_12_nic_in_9 = 1 ->
   - 5089.669280600068 hold_9_1 - 3026.150415568619 hold_9_2
   - 2153.180441367374 hold_9_3 - 1671.107130863092 hold_9_4
   - 1365.407481935425 hold_9_5 - 1154.256553003943 hold_9_6
   - 999.665171097569 hold_9_7 - 881.5921573259316 hold_9_8
   - 788.4645856441134 hold_9_9 - 713.1323747193427 hold_9_10
   - 650.9396938908995 hold_9_11 - 598.7245818794369 hold_9_12
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_12_nic_in_10: switch_nic_out_12_nic_in_10 = 1 ->
   - 5089.669280600068 hold_10_1 - 3026.150415568619 hold_10_2
   - 2153.180441367374 hold_10_3 - 1671.107130863092 hold_10_4
   - 1365.407481935425 hold_10_5 - 1154.256553003943 hold_10_6
   - 999.665171097569 hold_10_7 - 881.5921573259316 hold_10_8
   - 788.4645856441134 hold_10_9 - 713.1323747193427 hold_10_10
   - 650.9396938908995 hold_10_11 - 598.7245818794369 hold_10_12
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_12_nic_in_11: switch_nic_out_12_nic_in_11 = 1 ->
   - 5089.669280600068 hold_11_1 - 3026.150415568619 hold_11_2
   - 2153.180441367374 hold_11_3 - 1671.107130863092 hold_11_4
   - 1365.407481935425 hold_11_5 - 1154.256553003943 hold_11_6
   - 999.665171097569 hold_11_7 - 881.5921573259316 hold_11_8
   - 788.4645856441134 hold_11_9 - 713.1323747193427 hold_11_10
   - 650.9396938908995 hold_11_11 - 598.7245818794369 hold_11_12
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_13_nic_out_12_nic_in_13: switch_nic_out_12_nic_in_13 = 1 ->
   - 5089.669280600068 hold_13_1 - 3026.150415568619 hold_13_2
   - 2153.180441367374 hold_13_3 - 1671.107130863092 hold_13_4
   - 1365.407481935425 hold_13_5 - 1154.256553003943 hold_13_6
   - 999.665171097569 hold_13_7 - 881.5921573259316 hold_13_8
   - 788.4645856441134 hold_13_9 - 713.1323747193427 hold_13_10
   - 650.9396938908995 hold_13_11 - 598.7245818794369 hold_13_12
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_12_nic_in_14: switch_nic_out_12_nic_in_14 = 1 ->
   - 5089.669280600068 hold_14_1 - 3026.150415568619 hold_14_2
   - 2153.180441367374 hold_14_3 - 1671.107130863092 hold_14_4
   - 1365.407481935425 hold_14_5 - 1154.256553003943 hold_14_6
   - 999.665171097569 hold_14_7 - 881.5921573259316 hold_14_8
   - 788.4645856441134 hold_14_9 - 713.1323747193427 hold_14_10
   - 650.9396938908995 hold_14_11 - 598.7245818794369 hold_14_12
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_12_nic_in_15: switch_nic_out_12_nic_in_15 = 1 ->
   - 5089.669280600068 hold_15_1 - 3026.150415568619 hold_15_2
   - 2153.180441367374 hold_15_3 - 1671.107130863092 hold_15_4
   - 1365.407481935425 hold_15_5 - 1154.256553003943 hold_15_6
   - 999.665171097569 hold_15_7 - 881.5921573259316 hold_15_8
   - 788.4645856441134 hold_15_9 - 713.1323747193427 hold_15_10
   - 650.9396938908995 hold_15_11 - 598.7245818794369 hold_15_12
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_13_source_nic_in_13: switch_source_nic_in_13 = 1 ->
   - 6827.037409677124 hold_13_1 - 3565.661873596714 hold_13_2
   - 2412.957036174959 hold_13_3 - 1823.467617814477 hold_13_4
   - 1465.454616849587 hold_13_5 - 1224.951996455123 hold_13_6
   - 1052.260583931315 hold_13_7 - 922.2443255532355 hold_13_8
   - 820.8241232555838 hold_13_9 - 739.5004607016537 hold_13_10
   - 672.8385602120198 hold_13_11 - 617.20126020193 hold_13_12
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_0_nic_out_13_nic_in_0: switch_nic_out_13_nic_in_0 = 1 ->
   - 5089.669280600068 hold_0_1 - 3026.150415568619 hold_0_2
   - 2153.180441367374 hold_0_3 - 1671.107130863092 hold_0_4
   - 1365.407481935425 hold_0_5 - 1154.256553003943 hold_0_6
   - 999.665171097569 hold_0_7 - 881.5921573259316 hold_0_8
   - 788.4645856441134 hold_0_9 - 713.1323747193427 hold_0_10
   - 650.9396938908995 hold_0_11 - 598.7245818794369 hold_0_12
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_13_nic_in_1: switch_nic_out_13_nic_in_1 = 1 ->
   - 5089.669280600068 hold_1_1 - 3026.150415568619 hold_1_2
   - 2153.180441367374 hold_1_3 - 1671.107130863092 hold_1_4
   - 1365.407481935425 hold_1_5 - 1154.256553003943 hold_1_6
   - 999.665171097569 hold_1_7 - 881.5921573259316 hold_1_8
   - 788.4645856441134 hold_1_9 - 713.1323747193427 hold_1_10
   - 650.9396938908995 hold_1_11 - 598.7245818794369 hold_1_12
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_13_nic_in_2: switch_nic_out_13_nic_in_2 = 1 ->
   - 5089.669280600068 hold_2_1 - 3026.150415568619 hold_2_2
   - 2153.180441367374 hold_2_3 - 1671.107130863092 hold_2_4
   - 1365.407481935425 hold_2_5 - 1154.256553003943 hold_2_6
   - 999.665171097569 hold_2_7 - 881.5921573259316 hold_2_8
   - 788.4645856441134 hold_2_9 - 713.1323747193427 hold_2_10
   - 650.9396938908995 hold_2_11 - 598.7245818794369 hold_2_12
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_13_nic_in_3: switch_nic_out_13_nic_in_3 = 1 ->
   - 5089.669280600068 hold_3_1 - 3026.150415568619 hold_3_2
   - 2153.180441367374 hold_3_3 - 1671.107130863092 hold_3_4
   - 1365.407481935425 hold_3_5 - 1154.256553003943 hold_3_6
   - 999.665171097569 hold_3_7 - 881.5921573259316 hold_3_8
   - 788.4645856441134 hold_3_9 - 713.1323747193427 hold_3_10
   - 650.9396938908995 hold_3_11 - 598.7245818794369 hold_3_12
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_13_nic_in_4: switch_nic_out_13_nic_in_4 = 1 ->
   - 5089.669280600068 hold_4_1 - 3026.150415568619 hold_4_2
   - 2153.180441367374 hold_4_3 - 1671.107130863092 hold_4_4
   - 1365.407481935425 hold_4_5 - 1154.256553003943 hold_4_6
   - 999.665171097569 hold_4_7 - 881.5921573259316 hold_4_8
   - 788.4645856441134 hold_4_9 - 713.1323747193427 hold_4_10
   - 650.9396938908995 hold_4_11 - 598.7245818794369 hold_4_12
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_13_nic_in_5: switch_nic_out_13_nic_in_5 = 1 ->
   - 5089.669280600068 hold_5_1 - 3026.150415568619 hold_5_2
   - 2153.180441367374 hold_5_3 - 1671.107130863092 hold_5_4
   - 1365.407481935425 hold_5_5 - 1154.256553003943 hold_5_6
   - 999.665171097569 hold_5_7 - 881.5921573259316 hold_5_8
   - 788.4645856441134 hold_5_9 - 713.1323747193427 hold_5_10
   - 650.9396938908995 hold_5_11 - 598.7245818794369 hold_5_12
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_13_nic_in_6: switch_nic_out_13_nic_in_6 = 1 ->
   - 5089.669280600068 hold_6_1 - 3026.150415568619 hold_6_2
   - 2153.180441367374 hold_6_3 - 1671.107130863092 hold_6_4
   - 1365.407481935425 hold_6_5 - 1154.256553003943 hold_6_6
   - 999.665171097569 hold_6_7 - 881.5921573259316 hold_6_8
   - 788.4645856441134 hold_6_9 - 713.1323747193427 hold_6_10
   - 650.9396938908995 hold_6_11 - 598.7245818794369 hold_6_12
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_13_nic_in_7: switch_nic_out_13_nic_in_7 = 1 ->
   - 5089.669280600068 hold_7_1 - 3026.150415568619 hold_7_2
   - 2153.180441367374 hold_7_3 - 1671.107130863092 hold_7_4
   - 1365.407481935425 hold_7_5 - 1154.256553003943 hold_7_6
   - 999.665171097569 hold_7_7 - 881.5921573259316 hold_7_8
   - 788.4645856441134 hold_7_9 - 713.1323747193427 hold_7_10
   - 650.9396938908995 hold_7_11 - 598.7245818794369 hold_7_12
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_13_nic_in_8: switch_nic_out_13_nic_in_8 = 1 ->
   - 5089.669280600068 hold_8_1 - 3026.150415568619 hold_8_2
   - 2153.180441367374 hold_8_3 - 1671.107130863092 hold_8_4
   - 1365.407481935425 hold_8_5 - 1154.256553003943 hold_8_6
   - 999.665171097569 hold_8_7 - 881.5921573259316 hold_8_8
   - 788.4645856441134 hold_8_9 - 713.1323747193427 hold_8_10
   - 650.9396938908995 hold_8_11 - 598.7245818794369 hold_8_12
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_13_nic_in_9: switch_nic_out_13_nic_in_9 = 1 ->
   - 5089.669280600068 hold_9_1 - 3026.150415568619 hold_9_2
   - 2153.180441367374 hold_9_3 - 1671.107130863092 hold_9_4
   - 1365.407481935425 hold_9_5 - 1154.256553003943 hold_9_6
   - 999.665171097569 hold_9_7 - 881.5921573259316 hold_9_8
   - 788.4645856441134 hold_9_9 - 713.1323747193427 hold_9_10
   - 650.9396938908995 hold_9_11 - 598.7245818794369 hold_9_12
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_13_nic_in_10: switch_nic_out_13_nic_in_10 = 1 ->
   - 5089.669280600068 hold_10_1 - 3026.150415568619 hold_10_2
   - 2153.180441367374 hold_10_3 - 1671.107130863092 hold_10_4
   - 1365.407481935425 hold_10_5 - 1154.256553003943 hold_10_6
   - 999.665171097569 hold_10_7 - 881.5921573259316 hold_10_8
   - 788.4645856441134 hold_10_9 - 713.1323747193427 hold_10_10
   - 650.9396938908995 hold_10_11 - 598.7245818794369 hold_10_12
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_13_nic_in_11: switch_nic_out_13_nic_in_11 = 1 ->
   - 5089.669280600068 hold_11_1 - 3026.150415568619 hold_11_2
   - 2153.180441367374 hold_11_3 - 1671.107130863092 hold_11_4
   - 1365.407481935425 hold_11_5 - 1154.256553003943 hold_11_6
   - 999.665171097569 hold_11_7 - 881.5921573259316 hold_11_8
   - 788.4645856441134 hold_11_9 - 713.1323747193427 hold_11_10
   - 650.9396938908995 hold_11_11 - 598.7245818794369 hold_11_12
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_13_nic_in_12: switch_nic_out_13_nic_in_12 = 1 ->
   - 5089.669280600068 hold_12_1 - 3026.150415568619 hold_12_2
   - 2153.180441367374 hold_12_3 - 1671.107130863092 hold_12_4
   - 1365.407481935425 hold_12_5 - 1154.256553003943 hold_12_6
   - 999.665171097569 hold_12_7 - 881.5921573259316 hold_12_8
   - 788.4645856441134 hold_12_9 - 713.1323747193427 hold_12_10
   - 650.9396938908995 hold_12_11 - 598.7245818794369 hold_12_12
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_14_nic_out_13_nic_in_14: switch_nic_out_13_nic_in_14 = 1 ->
   - 5089.669280600068 hold_14_1 - 3026.150415568619 hold_14_2
   - 2153.180441367374 hold_14_3 - 1671.107130863092 hold_14_4
   - 1365.407481935425 hold_14_5 - 1154.256553003943 hold_14_6
   - 999.665171097569 hold_14_7 - 881.5921573259316 hold_14_8
   - 788.4645856441134 hold_14_9 - 713.1323747193427 hold_14_10
   - 650.9396938908995 hold_14_11 - 598.7245818794369 hold_14_12
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_15_nic_out_13_nic_in_15: switch_nic_out_13_nic_in_15 = 1 ->
   - 5089.669280600068 hold_15_1 - 3026.150415568619 hold_15_2
   - 2153.180441367374 hold_15_3 - 1671.107130863092 hold_15_4
   - 1365.407481935425 hold_15_5 - 1154.256553003943 hold_15_6
   - 999.665171097569 hold_15_7 - 881.5921573259316 hold_15_8
   - 788.4645856441134 hold_15_9 - 713.1323747193427 hold_15_10
   - 650.9396938908995 hold_15_11 - 598.7245818794369 hold_15_12
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_14_source_nic_in_14: switch_source_nic_in_14 = 1 ->
   - 6827.037409677124 hold_14_1 - 3565.661873596714 hold_14_2
   - 2412.957036174959 hold_14_3 - 1823.467617814477 hold_14_4
   - 1465.454616849587 hold_14_5 - 1224.951996455123 hold_14_6
   - 1052.260583931315 hold_14_7 - 922.2443255532355 hold_14_8
   - 820.8241232555838 hold_14_9 - 739.5004607016537 hold_14_10
   - 672.8385602120198 hold_14_11 - 617.20126020193 hold_14_12
   + flow_nic_in_14_14 <= 0
 constr_intergroup_throughput_0_nic_out_14_nic_in_0: switch_nic_out_14_nic_in_0 = 1 ->
   - 5089.669280600068 hold_0_1 - 3026.150415568619 hold_0_2
   - 2153.180441367374 hold_0_3 - 1671.107130863092 hold_0_4
   - 1365.407481935425 hold_0_5 - 1154.256553003943 hold_0_6
   - 999.665171097569 hold_0_7 - 881.5921573259316 hold_0_8
   - 788.4645856441134 hold_0_9 - 713.1323747193427 hold_0_10
   - 650.9396938908995 hold_0_11 - 598.7245818794369 hold_0_12
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_14_nic_in_1: switch_nic_out_14_nic_in_1 = 1 ->
   - 5089.669280600068 hold_1_1 - 3026.150415568619 hold_1_2
   - 2153.180441367374 hold_1_3 - 1671.107130863092 hold_1_4
   - 1365.407481935425 hold_1_5 - 1154.256553003943 hold_1_6
   - 999.665171097569 hold_1_7 - 881.5921573259316 hold_1_8
   - 788.4645856441134 hold_1_9 - 713.1323747193427 hold_1_10
   - 650.9396938908995 hold_1_11 - 598.7245818794369 hold_1_12
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_14_nic_in_2: switch_nic_out_14_nic_in_2 = 1 ->
   - 5089.669280600068 hold_2_1 - 3026.150415568619 hold_2_2
   - 2153.180441367374 hold_2_3 - 1671.107130863092 hold_2_4
   - 1365.407481935425 hold_2_5 - 1154.256553003943 hold_2_6
   - 999.665171097569 hold_2_7 - 881.5921573259316 hold_2_8
   - 788.4645856441134 hold_2_9 - 713.1323747193427 hold_2_10
   - 650.9396938908995 hold_2_11 - 598.7245818794369 hold_2_12
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_14_nic_in_3: switch_nic_out_14_nic_in_3 = 1 ->
   - 5089.669280600068 hold_3_1 - 3026.150415568619 hold_3_2
   - 2153.180441367374 hold_3_3 - 1671.107130863092 hold_3_4
   - 1365.407481935425 hold_3_5 - 1154.256553003943 hold_3_6
   - 999.665171097569 hold_3_7 - 881.5921573259316 hold_3_8
   - 788.4645856441134 hold_3_9 - 713.1323747193427 hold_3_10
   - 650.9396938908995 hold_3_11 - 598.7245818794369 hold_3_12
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_14_nic_in_4: switch_nic_out_14_nic_in_4 = 1 ->
   - 5089.669280600068 hold_4_1 - 3026.150415568619 hold_4_2
   - 2153.180441367374 hold_4_3 - 1671.107130863092 hold_4_4
   - 1365.407481935425 hold_4_5 - 1154.256553003943 hold_4_6
   - 999.665171097569 hold_4_7 - 881.5921573259316 hold_4_8
   - 788.4645856441134 hold_4_9 - 713.1323747193427 hold_4_10
   - 650.9396938908995 hold_4_11 - 598.7245818794369 hold_4_12
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_14_nic_in_5: switch_nic_out_14_nic_in_5 = 1 ->
   - 5089.669280600068 hold_5_1 - 3026.150415568619 hold_5_2
   - 2153.180441367374 hold_5_3 - 1671.107130863092 hold_5_4
   - 1365.407481935425 hold_5_5 - 1154.256553003943 hold_5_6
   - 999.665171097569 hold_5_7 - 881.5921573259316 hold_5_8
   - 788.4645856441134 hold_5_9 - 713.1323747193427 hold_5_10
   - 650.9396938908995 hold_5_11 - 598.7245818794369 hold_5_12
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_14_nic_in_6: switch_nic_out_14_nic_in_6 = 1 ->
   - 5089.669280600068 hold_6_1 - 3026.150415568619 hold_6_2
   - 2153.180441367374 hold_6_3 - 1671.107130863092 hold_6_4
   - 1365.407481935425 hold_6_5 - 1154.256553003943 hold_6_6
   - 999.665171097569 hold_6_7 - 881.5921573259316 hold_6_8
   - 788.4645856441134 hold_6_9 - 713.1323747193427 hold_6_10
   - 650.9396938908995 hold_6_11 - 598.7245818794369 hold_6_12
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_14_nic_in_7: switch_nic_out_14_nic_in_7 = 1 ->
   - 5089.669280600068 hold_7_1 - 3026.150415568619 hold_7_2
   - 2153.180441367374 hold_7_3 - 1671.107130863092 hold_7_4
   - 1365.407481935425 hold_7_5 - 1154.256553003943 hold_7_6
   - 999.665171097569 hold_7_7 - 881.5921573259316 hold_7_8
   - 788.4645856441134 hold_7_9 - 713.1323747193427 hold_7_10
   - 650.9396938908995 hold_7_11 - 598.7245818794369 hold_7_12
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_14_nic_in_8: switch_nic_out_14_nic_in_8 = 1 ->
   - 5089.669280600068 hold_8_1 - 3026.150415568619 hold_8_2
   - 2153.180441367374 hold_8_3 - 1671.107130863092 hold_8_4
   - 1365.407481935425 hold_8_5 - 1154.256553003943 hold_8_6
   - 999.665171097569 hold_8_7 - 881.5921573259316 hold_8_8
   - 788.4645856441134 hold_8_9 - 713.1323747193427 hold_8_10
   - 650.9396938908995 hold_8_11 - 598.7245818794369 hold_8_12
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_14_nic_in_9: switch_nic_out_14_nic_in_9 = 1 ->
   - 5089.669280600068 hold_9_1 - 3026.150415568619 hold_9_2
   - 2153.180441367374 hold_9_3 - 1671.107130863092 hold_9_4
   - 1365.407481935425 hold_9_5 - 1154.256553003943 hold_9_6
   - 999.665171097569 hold_9_7 - 881.5921573259316 hold_9_8
   - 788.4645856441134 hold_9_9 - 713.1323747193427 hold_9_10
   - 650.9396938908995 hold_9_11 - 598.7245818794369 hold_9_12
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_14_nic_in_10: switch_nic_out_14_nic_in_10 = 1 ->
   - 5089.669280600068 hold_10_1 - 3026.150415568619 hold_10_2
   - 2153.180441367374 hold_10_3 - 1671.107130863092 hold_10_4
   - 1365.407481935425 hold_10_5 - 1154.256553003943 hold_10_6
   - 999.665171097569 hold_10_7 - 881.5921573259316 hold_10_8
   - 788.4645856441134 hold_10_9 - 713.1323747193427 hold_10_10
   - 650.9396938908995 hold_10_11 - 598.7245818794369 hold_10_12
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_14_nic_in_11: switch_nic_out_14_nic_in_11 = 1 ->
   - 5089.669280600068 hold_11_1 - 3026.150415568619 hold_11_2
   - 2153.180441367374 hold_11_3 - 1671.107130863092 hold_11_4
   - 1365.407481935425 hold_11_5 - 1154.256553003943 hold_11_6
   - 999.665171097569 hold_11_7 - 881.5921573259316 hold_11_8
   - 788.4645856441134 hold_11_9 - 713.1323747193427 hold_11_10
   - 650.9396938908995 hold_11_11 - 598.7245818794369 hold_11_12
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_14_nic_in_12: switch_nic_out_14_nic_in_12 = 1 ->
   - 5089.669280600068 hold_12_1 - 3026.150415568619 hold_12_2
   - 2153.180441367374 hold_12_3 - 1671.107130863092 hold_12_4
   - 1365.407481935425 hold_12_5 - 1154.256553003943 hold_12_6
   - 999.665171097569 hold_12_7 - 881.5921573259316 hold_12_8
   - 788.4645856441134 hold_12_9 - 713.1323747193427 hold_12_10
   - 650.9396938908995 hold_12_11 - 598.7245818794369 hold_12_12
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_14_nic_in_13: switch_nic_out_14_nic_in_13 = 1 ->
   - 5089.669280600068 hold_13_1 - 3026.150415568619 hold_13_2
   - 2153.180441367374 hold_13_3 - 1671.107130863092 hold_13_4
   - 1365.407481935425 hold_13_5 - 1154.256553003943 hold_13_6
   - 999.665171097569 hold_13_7 - 881.5921573259316 hold_13_8
   - 788.4645856441134 hold_13_9 - 713.1323747193427 hold_13_10
   - 650.9396938908995 hold_13_11 - 598.7245818794369 hold_13_12
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_15_nic_out_14_nic_in_15: switch_nic_out_14_nic_in_15 = 1 ->
   - 5089.669280600068 hold_15_1 - 3026.150415568619 hold_15_2
   - 2153.180441367374 hold_15_3 - 1671.107130863092 hold_15_4
   - 1365.407481935425 hold_15_5 - 1154.256553003943 hold_15_6
   - 999.665171097569 hold_15_7 - 881.5921573259316 hold_15_8
   - 788.4645856441134 hold_15_9 - 713.1323747193427 hold_15_10
   - 650.9396938908995 hold_15_11 - 598.7245818794369 hold_15_12
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_15_source_nic_in_15: switch_source_nic_in_15 = 1 ->
   - 6827.037409677124 hold_15_1 - 3565.661873596714 hold_15_2
   - 2412.957036174959 hold_15_3 - 1823.467617814477 hold_15_4
   - 1465.454616849587 hold_15_5 - 1224.951996455123 hold_15_6
   - 1052.260583931315 hold_15_7 - 922.2443255532355 hold_15_8
   - 820.8241232555838 hold_15_9 - 739.5004607016537 hold_15_10
   - 672.8385602120198 hold_15_11 - 617.20126020193 hold_15_12
   + flow_nic_in_15_15 <= 0
 constr_intergroup_throughput_0_nic_out_15_nic_in_0: switch_nic_out_15_nic_in_0 = 1 ->
   - 5089.669280600068 hold_0_1 - 3026.150415568619 hold_0_2
   - 2153.180441367374 hold_0_3 - 1671.107130863092 hold_0_4
   - 1365.407481935425 hold_0_5 - 1154.256553003943 hold_0_6
   - 999.665171097569 hold_0_7 - 881.5921573259316 hold_0_8
   - 788.4645856441134 hold_0_9 - 713.1323747193427 hold_0_10
   - 650.9396938908995 hold_0_11 - 598.7245818794369 hold_0_12
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_15_nic_in_1: switch_nic_out_15_nic_in_1 = 1 ->
   - 5089.669280600068 hold_1_1 - 3026.150415568619 hold_1_2
   - 2153.180441367374 hold_1_3 - 1671.107130863092 hold_1_4
   - 1365.407481935425 hold_1_5 - 1154.256553003943 hold_1_6
   - 999.665171097569 hold_1_7 - 881.5921573259316 hold_1_8
   - 788.4645856441134 hold_1_9 - 713.1323747193427 hold_1_10
   - 650.9396938908995 hold_1_11 - 598.7245818794369 hold_1_12
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_15_nic_in_2: switch_nic_out_15_nic_in_2 = 1 ->
   - 5089.669280600068 hold_2_1 - 3026.150415568619 hold_2_2
   - 2153.180441367374 hold_2_3 - 1671.107130863092 hold_2_4
   - 1365.407481935425 hold_2_5 - 1154.256553003943 hold_2_6
   - 999.665171097569 hold_2_7 - 881.5921573259316 hold_2_8
   - 788.4645856441134 hold_2_9 - 713.1323747193427 hold_2_10
   - 650.9396938908995 hold_2_11 - 598.7245818794369 hold_2_12
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_15_nic_in_3: switch_nic_out_15_nic_in_3 = 1 ->
   - 5089.669280600068 hold_3_1 - 3026.150415568619 hold_3_2
   - 2153.180441367374 hold_3_3 - 1671.107130863092 hold_3_4
   - 1365.407481935425 hold_3_5 - 1154.256553003943 hold_3_6
   - 999.665171097569 hold_3_7 - 881.5921573259316 hold_3_8
   - 788.4645856441134 hold_3_9 - 713.1323747193427 hold_3_10
   - 650.9396938908995 hold_3_11 - 598.7245818794369 hold_3_12
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_15_nic_in_4: switch_nic_out_15_nic_in_4 = 1 ->
   - 5089.669280600068 hold_4_1 - 3026.150415568619 hold_4_2
   - 2153.180441367374 hold_4_3 - 1671.107130863092 hold_4_4
   - 1365.407481935425 hold_4_5 - 1154.256553003943 hold_4_6
   - 999.665171097569 hold_4_7 - 881.5921573259316 hold_4_8
   - 788.4645856441134 hold_4_9 - 713.1323747193427 hold_4_10
   - 650.9396938908995 hold_4_11 - 598.7245818794369 hold_4_12
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_15_nic_in_5: switch_nic_out_15_nic_in_5 = 1 ->
   - 5089.669280600068 hold_5_1 - 3026.150415568619 hold_5_2
   - 2153.180441367374 hold_5_3 - 1671.107130863092 hold_5_4
   - 1365.407481935425 hold_5_5 - 1154.256553003943 hold_5_6
   - 999.665171097569 hold_5_7 - 881.5921573259316 hold_5_8
   - 788.4645856441134 hold_5_9 - 713.1323747193427 hold_5_10
   - 650.9396938908995 hold_5_11 - 598.7245818794369 hold_5_12
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_15_nic_in_6: switch_nic_out_15_nic_in_6 = 1 ->
   - 5089.669280600068 hold_6_1 - 3026.150415568619 hold_6_2
   - 2153.180441367374 hold_6_3 - 1671.107130863092 hold_6_4
   - 1365.407481935425 hold_6_5 - 1154.256553003943 hold_6_6
   - 999.665171097569 hold_6_7 - 881.5921573259316 hold_6_8
   - 788.4645856441134 hold_6_9 - 713.1323747193427 hold_6_10
   - 650.9396938908995 hold_6_11 - 598.7245818794369 hold_6_12
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_15_nic_in_7: switch_nic_out_15_nic_in_7 = 1 ->
   - 5089.669280600068 hold_7_1 - 3026.150415568619 hold_7_2
   - 2153.180441367374 hold_7_3 - 1671.107130863092 hold_7_4
   - 1365.407481935425 hold_7_5 - 1154.256553003943 hold_7_6
   - 999.665171097569 hold_7_7 - 881.5921573259316 hold_7_8
   - 788.4645856441134 hold_7_9 - 713.1323747193427 hold_7_10
   - 650.9396938908995 hold_7_11 - 598.7245818794369 hold_7_12
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_15_nic_in_8: switch_nic_out_15_nic_in_8 = 1 ->
   - 5089.669280600068 hold_8_1 - 3026.150415568619 hold_8_2
   - 2153.180441367374 hold_8_3 - 1671.107130863092 hold_8_4
   - 1365.407481935425 hold_8_5 - 1154.256553003943 hold_8_6
   - 999.665171097569 hold_8_7 - 881.5921573259316 hold_8_8
   - 788.4645856441134 hold_8_9 - 713.1323747193427 hold_8_10
   - 650.9396938908995 hold_8_11 - 598.7245818794369 hold_8_12
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_15_nic_in_9: switch_nic_out_15_nic_in_9 = 1 ->
   - 5089.669280600068 hold_9_1 - 3026.150415568619 hold_9_2
   - 2153.180441367374 hold_9_3 - 1671.107130863092 hold_9_4
   - 1365.407481935425 hold_9_5 - 1154.256553003943 hold_9_6
   - 999.665171097569 hold_9_7 - 881.5921573259316 hold_9_8
   - 788.4645856441134 hold_9_9 - 713.1323747193427 hold_9_10
   - 650.9396938908995 hold_9_11 - 598.7245818794369 hold_9_12
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_15_nic_in_10: switch_nic_out_15_nic_in_10 = 1 ->
   - 5089.669280600068 hold_10_1 - 3026.150415568619 hold_10_2
   - 2153.180441367374 hold_10_3 - 1671.107130863092 hold_10_4
   - 1365.407481935425 hold_10_5 - 1154.256553003943 hold_10_6
   - 999.665171097569 hold_10_7 - 881.5921573259316 hold_10_8
   - 788.4645856441134 hold_10_9 - 713.1323747193427 hold_10_10
   - 650.9396938908995 hold_10_11 - 598.7245818794369 hold_10_12
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_15_nic_in_11: switch_nic_out_15_nic_in_11 = 1 ->
   - 5089.669280600068 hold_11_1 - 3026.150415568619 hold_11_2
   - 2153.180441367374 hold_11_3 - 1671.107130863092 hold_11_4
   - 1365.407481935425 hold_11_5 - 1154.256553003943 hold_11_6
   - 999.665171097569 hold_11_7 - 881.5921573259316 hold_11_8
   - 788.4645856441134 hold_11_9 - 713.1323747193427 hold_11_10
   - 650.9396938908995 hold_11_11 - 598.7245818794369 hold_11_12
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_12_nic_out_15_nic_in_12: switch_nic_out_15_nic_in_12 = 1 ->
   - 5089.669280600068 hold_12_1 - 3026.150415568619 hold_12_2
   - 2153.180441367374 hold_12_3 - 1671.107130863092 hold_12_4
   - 1365.407481935425 hold_12_5 - 1154.256553003943 hold_12_6
   - 999.665171097569 hold_12_7 - 881.5921573259316 hold_12_8
   - 788.4645856441134 hold_12_9 - 713.1323747193427 hold_12_10
   - 650.9396938908995 hold_12_11 - 598.7245818794369 hold_12_12
   + flow_nic_in_12_12 <= 0
 constr_intergroup_throughput_13_nic_out_15_nic_in_13: switch_nic_out_15_nic_in_13 = 1 ->
   - 5089.669280600068 hold_13_1 - 3026.150415568619 hold_13_2
   - 2153.180441367374 hold_13_3 - 1671.107130863092 hold_13_4
   - 1365.407481935425 hold_13_5 - 1154.256553003943 hold_13_6
   - 999.665171097569 hold_13_7 - 881.5921573259316 hold_13_8
   - 788.4645856441134 hold_13_9 - 713.1323747193427 hold_13_10
   - 650.9396938908995 hold_13_11 - 598.7245818794369 hold_13_12
   + flow_nic_in_13_13 <= 0
 constr_intergroup_throughput_14_nic_out_15_nic_in_14: switch_nic_out_15_nic_in_14 = 1 ->
   - 5089.669280600068 hold_14_1 - 3026.150415568619 hold_14_2
   - 2153.180441367374 hold_14_3 - 1671.107130863092 hold_14_4
   - 1365.407481935425 hold_14_5 - 1154.256553003943 hold_14_6
   - 999.665171097569 hold_14_7 - 881.5921573259316 hold_14_8
   - 788.4645856441134 hold_14_9 - 713.1323747193427 hold_14_10
   - 650.9396938908995 hold_14_11 - 598.7245818794369 hold_14_12
   + flow_nic_in_14_14 <= 0
 inactive_end_not_complete_0: is_active_0 = 0 -> is_end_complete_0 = 0
 active_end_val_0: is_active_0 = 1 -> - start_0 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11 - 12 hold_0_12
   + val_end_0 = 0
 inactive_end_zero_0: is_active_0 = 0 -> val_end_0 = 0
 inactive_end_not_complete_1: is_active_1 = 0 -> is_end_complete_1 = 0
 active_end_val_1: is_active_1 = 1 -> - start_1 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11 - 12 hold_1_12
   + val_end_1 = 0
 inactive_end_zero_1: is_active_1 = 0 -> val_end_1 = 0
 inactive_end_not_complete_2: is_active_2 = 0 -> is_end_complete_2 = 0
 active_end_val_2: is_active_2 = 1 -> - start_2 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 - 12 hold_2_12
   + val_end_2 = 0
 inactive_end_zero_2: is_active_2 = 0 -> val_end_2 = 0
 inactive_end_not_complete_3: is_active_3 = 0 -> is_end_complete_3 = 0
 active_end_val_3: is_active_3 = 1 -> - start_3 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   - 8 hold_3_8 - 9 hold_3_9 - 10 hold_3_10 - 11 hold_3_11 - 12 hold_3_12
   + val_end_3 = 0
 inactive_end_zero_3: is_active_3 = 0 -> val_end_3 = 0
 inactive_end_not_complete_4: is_active_4 = 0 -> is_end_complete_4 = 0
 active_end_val_4: is_active_4 = 1 -> - start_4 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   - 8 hold_4_8 - 9 hold_4_9 - 10 hold_4_10 - 11 hold_4_11 - 12 hold_4_12
   + val_end_4 = 0
 inactive_end_zero_4: is_active_4 = 0 -> val_end_4 = 0
 inactive_end_not_complete_5: is_active_5 = 0 -> is_end_complete_5 = 0
 active_end_val_5: is_active_5 = 1 -> - start_5 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11 - 12 hold_5_12
   + val_end_5 = 0
 inactive_end_zero_5: is_active_5 = 0 -> val_end_5 = 0
 inactive_end_not_complete_6: is_active_6 = 0 -> is_end_complete_6 = 0
 active_end_val_6: is_active_6 = 1 -> - start_6 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11 - 12 hold_6_12
   + val_end_6 = 0
 inactive_end_zero_6: is_active_6 = 0 -> val_end_6 = 0
 inactive_end_not_complete_7: is_active_7 = 0 -> is_end_complete_7 = 0
 active_end_val_7: is_active_7 = 1 -> - start_7 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   - 8 hold_7_8 - 9 hold_7_9 - 10 hold_7_10 - 11 hold_7_11 - 12 hold_7_12
   + val_end_7 = 0
 inactive_end_zero_7: is_active_7 = 0 -> val_end_7 = 0
 inactive_end_not_complete_8: is_active_8 = 0 -> is_end_complete_8 = 0
 active_end_val_8: is_active_8 = 1 -> - start_8 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11 - 12 hold_8_12
   + val_end_8 = 0
 inactive_end_zero_8: is_active_8 = 0 -> val_end_8 = 0
 inactive_end_not_complete_9: is_active_9 = 0 -> is_end_complete_9 = 0
 active_end_val_9: is_active_9 = 1 -> - start_9 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   - 8 hold_9_8 - 9 hold_9_9 - 10 hold_9_10 - 11 hold_9_11 - 12 hold_9_12
   + val_end_9 = 0
 inactive_end_zero_9: is_active_9 = 0 -> val_end_9 = 0
 inactive_end_not_complete_10: is_active_10 = 0 -> is_end_complete_10 = 0
 active_end_val_10: is_active_10 = 1 -> - start_10 - hold_10_1
   - 2 hold_10_2 - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6
   - 7 hold_10_7 - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10
   - 11 hold_10_11 - 12 hold_10_12 + val_end_10 = 0
 inactive_end_zero_10: is_active_10 = 0 -> val_end_10 = 0
 inactive_end_not_complete_11: is_active_11 = 0 -> is_end_complete_11 = 0
 active_end_val_11: is_active_11 = 1 -> - start_11 - hold_11_1
   - 2 hold_11_2 - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6
   - 7 hold_11_7 - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10
   - 11 hold_11_11 - 12 hold_11_12 + val_end_11 = 0
 inactive_end_zero_11: is_active_11 = 0 -> val_end_11 = 0
 inactive_end_not_complete_12: is_active_12 = 0 -> is_end_complete_12 = 0
 active_end_val_12: is_active_12 = 1 -> - start_12 - hold_12_1
   - 2 hold_12_2 - 3 hold_12_3 - 4 hold_12_4 - 5 hold_12_5 - 6 hold_12_6
   - 7 hold_12_7 - 8 hold_12_8 - 9 hold_12_9 - 10 hold_12_10
   - 11 hold_12_11 - 12 hold_12_12 + val_end_12 = 0
 inactive_end_zero_12: is_active_12 = 0 -> val_end_12 = 0
 inactive_end_not_complete_13: is_active_13 = 0 -> is_end_complete_13 = 0
 active_end_val_13: is_active_13 = 1 -> - start_13 - hold_13_1
   - 2 hold_13_2 - 3 hold_13_3 - 4 hold_13_4 - 5 hold_13_5 - 6 hold_13_6
   - 7 hold_13_7 - 8 hold_13_8 - 9 hold_13_9 - 10 hold_13_10
   - 11 hold_13_11 - 12 hold_13_12 + val_end_13 = 0
 inactive_end_zero_13: is_active_13 = 0 -> val_end_13 = 0
 inactive_end_not_complete_14: is_active_14 = 0 -> is_end_complete_14 = 0
 active_end_val_14: is_active_14 = 1 -> - start_14 - hold_14_1
   - 2 hold_14_2 - 3 hold_14_3 - 4 hold_14_4 - 5 hold_14_5 - 6 hold_14_6
   - 7 hold_14_7 - 8 hold_14_8 - 9 hold_14_9 - 10 hold_14_10
   - 11 hold_14_11 - 12 hold_14_12 + val_end_14 = 0
 inactive_end_zero_14: is_active_14 = 0 -> val_end_14 = 0
 inactive_end_not_complete_15: is_active_15 = 0 -> is_end_complete_15 = 0
 active_end_val_15: is_active_15 = 1 -> - start_15 - hold_15_1
   - 2 hold_15_2 - 3 hold_15_3 - 4 hold_15_4 - 5 hold_15_5 - 6 hold_15_6
   - 7 hold_15_7 - 8 hold_15_8 - 9 hold_15_9 - 10 hold_15_10
   - 11 hold_15_11 - 12 hold_15_12 + val_end_15 = 0
 inactive_end_zero_15: is_active_15 = 0 -> val_end_15 = 0
 group_throughput_constr_0: group_tp_active_0 = 1 -> - 7464 hold_0_1
   - 3732 hold_0_2 - 2488 hold_0_3 - 1866 hold_0_4 - 1492.8 hold_0_5
   - 1244 hold_0_6 - 1066.285714285714 hold_0_7 - 933 hold_0_8
   - 829.3333333333333 hold_0_9 - 746.4 hold_0_10
   - 678.5454545454545 hold_0_11 - 622 hold_0_12 + flow_nic_in_0_0 <= 0
 group_throughput_constr_1: group_tp_active_1 = 1 -> - 7464 hold_1_1
   - 3732 hold_1_2 - 2488 hold_1_3 - 1866 hold_1_4 - 1492.8 hold_1_5
   - 1244 hold_1_6 - 1066.285714285714 hold_1_7 - 933 hold_1_8
   - 829.3333333333333 hold_1_9 - 746.4 hold_1_10
   - 678.5454545454545 hold_1_11 - 622 hold_1_12 + flow_nic_in_1_1 <= 0
 group_throughput_constr_2: group_tp_active_2 = 1 -> - 7464 hold_2_1
   - 3732 hold_2_2 - 2488 hold_2_3 - 1866 hold_2_4 - 1492.8 hold_2_5
   - 1244 hold_2_6 - 1066.285714285714 hold_2_7 - 933 hold_2_8
   - 829.3333333333333 hold_2_9 - 746.4 hold_2_10
   - 678.5454545454545 hold_2_11 - 622 hold_2_12 + flow_nic_in_2_2 <= 0
 group_throughput_constr_3: group_tp_active_3 = 1 -> - 7464 hold_3_1
   - 3732 hold_3_2 - 2488 hold_3_3 - 1866 hold_3_4 - 1492.8 hold_3_5
   - 1244 hold_3_6 - 1066.285714285714 hold_3_7 - 933 hold_3_8
   - 829.3333333333333 hold_3_9 - 746.4 hold_3_10
   - 678.5454545454545 hold_3_11 - 622 hold_3_12 + flow_nic_in_3_3 <= 0
 group_throughput_constr_4: group_tp_active_4 = 1 -> - 7464 hold_4_1
   - 3732 hold_4_2 - 2488 hold_4_3 - 1866 hold_4_4 - 1492.8 hold_4_5
   - 1244 hold_4_6 - 1066.285714285714 hold_4_7 - 933 hold_4_8
   - 829.3333333333333 hold_4_9 - 746.4 hold_4_10
   - 678.5454545454545 hold_4_11 - 622 hold_4_12 + flow_nic_in_4_4 <= 0
 group_throughput_constr_5: group_tp_active_5 = 1 -> - 7464 hold_5_1
   - 3732 hold_5_2 - 2488 hold_5_3 - 1866 hold_5_4 - 1492.8 hold_5_5
   - 1244 hold_5_6 - 1066.285714285714 hold_5_7 - 933 hold_5_8
   - 829.3333333333333 hold_5_9 - 746.4 hold_5_10
   - 678.5454545454545 hold_5_11 - 622 hold_5_12 + flow_nic_in_5_5 <= 0
 group_throughput_constr_6: group_tp_active_6 = 1 -> - 7464 hold_6_1
   - 3732 hold_6_2 - 2488 hold_6_3 - 1866 hold_6_4 - 1492.8 hold_6_5
   - 1244 hold_6_6 - 1066.285714285714 hold_6_7 - 933 hold_6_8
   - 829.3333333333333 hold_6_9 - 746.4 hold_6_10
   - 678.5454545454545 hold_6_11 - 622 hold_6_12 + flow_nic_in_6_6 <= 0
 group_throughput_constr_7: group_tp_active_7 = 1 -> - 7464 hold_7_1
   - 3732 hold_7_2 - 2488 hold_7_3 - 1866 hold_7_4 - 1492.8 hold_7_5
   - 1244 hold_7_6 - 1066.285714285714 hold_7_7 - 933 hold_7_8
   - 829.3333333333333 hold_7_9 - 746.4 hold_7_10
   - 678.5454545454545 hold_7_11 - 622 hold_7_12 + flow_nic_in_7_7 <= 0
 group_throughput_constr_8: group_tp_active_8 = 1 -> - 7464 hold_8_1
   - 3732 hold_8_2 - 2488 hold_8_3 - 1866 hold_8_4 - 1492.8 hold_8_5
   - 1244 hold_8_6 - 1066.285714285714 hold_8_7 - 933 hold_8_8
   - 829.3333333333333 hold_8_9 - 746.4 hold_8_10
   - 678.5454545454545 hold_8_11 - 622 hold_8_12 + flow_nic_in_8_8 <= 0
 group_throughput_constr_9: group_tp_active_9 = 1 -> - 7464 hold_9_1
   - 3732 hold_9_2 - 2488 hold_9_3 - 1866 hold_9_4 - 1492.8 hold_9_5
   - 1244 hold_9_6 - 1066.285714285714 hold_9_7 - 933 hold_9_8
   - 829.3333333333333 hold_9_9 - 746.4 hold_9_10
   - 678.5454545454545 hold_9_11 - 622 hold_9_12 + flow_nic_in_9_9 <= 0
 group_throughput_constr_10: group_tp_active_10 = 1 -> - 7464 hold_10_1
   - 3732 hold_10_2 - 2488 hold_10_3 - 1866 hold_10_4 - 1492.8 hold_10_5
   - 1244 hold_10_6 - 1066.285714285714 hold_10_7 - 933 hold_10_8
   - 829.3333333333333 hold_10_9 - 746.4 hold_10_10
   - 678.5454545454545 hold_10_11 - 622 hold_10_12 + flow_nic_in_10_10
   <= 0
 group_throughput_constr_11: group_tp_active_11 = 1 -> - 7464 hold_11_1
   - 3732 hold_11_2 - 2488 hold_11_3 - 1866 hold_11_4 - 1492.8 hold_11_5
   - 1244 hold_11_6 - 1066.285714285714 hold_11_7 - 933 hold_11_8
   - 829.3333333333333 hold_11_9 - 746.4 hold_11_10
   - 678.5454545454545 hold_11_11 - 622 hold_11_12 + flow_nic_in_11_11
   <= 0
 group_throughput_constr_12: group_tp_active_12 = 1 -> - 7464 hold_12_1
   - 3732 hold_12_2 - 2488 hold_12_3 - 1866 hold_12_4 - 1492.8 hold_12_5
   - 1244 hold_12_6 - 1066.285714285714 hold_12_7 - 933 hold_12_8
   - 829.3333333333333 hold_12_9 - 746.4 hold_12_10
   - 678.5454545454545 hold_12_11 - 622 hold_12_12 + flow_nic_in_12_12
   <= 0
 group_throughput_constr_13: group_tp_active_13 = 1 -> - 7464 hold_13_1
   - 3732 hold_13_2 - 2488 hold_13_3 - 1866 hold_13_4 - 1492.8 hold_13_5
   - 1244 hold_13_6 - 1066.285714285714 hold_13_7 - 933 hold_13_8
   - 829.3333333333333 hold_13_9 - 746.4 hold_13_10
   - 678.5454545454545 hold_13_11 - 622 hold_13_12 + flow_nic_in_13_13
   <= 0
 group_throughput_constr_14: group_tp_active_14 = 1 -> - 7464 hold_14_1
   - 3732 hold_14_2 - 2488 hold_14_3 - 1866 hold_14_4 - 1492.8 hold_14_5
   - 1244 hold_14_6 - 1066.285714285714 hold_14_7 - 933 hold_14_8
   - 829.3333333333333 hold_14_9 - 746.4 hold_14_10
   - 678.5454545454545 hold_14_11 - 622 hold_14_12 + flow_nic_in_14_14
   <= 0
 group_throughput_constr_15: group_tp_active_15 = 1 -> - 7464 hold_15_1
   - 3732 hold_15_2 - 2488 hold_15_3 - 1866 hold_15_4 - 1492.8 hold_15_5
   - 1244 hold_15_6 - 1066.285714285714 hold_15_7 - 933 hold_15_8
   - 829.3333333333333 hold_15_9 - 746.4 hold_15_10
   - 678.5454545454545 hold_15_11 - 622 hold_15_12 + flow_nic_in_15_15
   <= 0
Bounds
Binaries
 hold_0_1 hold_0_2 hold_0_3 hold_0_4 hold_0_5 hold_0_6 hold_0_7 hold_0_8
 hold_0_9 hold_0_10 hold_0_11 hold_0_12 hold_1_1 hold_1_2 hold_1_3 hold_1_4
 hold_1_5 hold_1_6 hold_1_7 hold_1_8 hold_1_9 hold_1_10 hold_1_11 hold_1_12
 hold_2_1 hold_2_2 hold_2_3 hold_2_4 hold_2_5 hold_2_6 hold_2_7 hold_2_8
 hold_2_9 hold_2_10 hold_2_11 hold_2_12 hold_3_1 hold_3_2 hold_3_3 hold_3_4
 hold_3_5 hold_3_6 hold_3_7 hold_3_8 hold_3_9 hold_3_10 hold_3_11 hold_3_12
 hold_4_1 hold_4_2 hold_4_3 hold_4_4 hold_4_5 hold_4_6 hold_4_7 hold_4_8
 hold_4_9 hold_4_10 hold_4_11 hold_4_12 hold_5_1 hold_5_2 hold_5_3 hold_5_4
 hold_5_5 hold_5_6 hold_5_7 hold_5_8 hold_5_9 hold_5_10 hold_5_11 hold_5_12
 hold_6_1 hold_6_2 hold_6_3 hold_6_4 hold_6_5 hold_6_6 hold_6_7 hold_6_8
 hold_6_9 hold_6_10 hold_6_11 hold_6_12 hold_7_1 hold_7_2 hold_7_3 hold_7_4
 hold_7_5 hold_7_6 hold_7_7 hold_7_8 hold_7_9 hold_7_10 hold_7_11 hold_7_12
 hold_8_1 hold_8_2 hold_8_3 hold_8_4 hold_8_5 hold_8_6 hold_8_7 hold_8_8
 hold_8_9 hold_8_10 hold_8_11 hold_8_12 hold_9_1 hold_9_2 hold_9_3 hold_9_4
 hold_9_5 hold_9_6 hold_9_7 hold_9_8 hold_9_9 hold_9_10 hold_9_11 hold_9_12
 hold_10_1 hold_10_2 hold_10_3 hold_10_4 hold_10_5 hold_10_6 hold_10_7
 hold_10_8 hold_10_9 hold_10_10 hold_10_11 hold_10_12 hold_11_1 hold_11_2
 hold_11_3 hold_11_4 hold_11_5 hold_11_6 hold_11_7 hold_11_8 hold_11_9
 hold_11_10 hold_11_11 hold_11_12 hold_12_1 hold_12_2 hold_12_3 hold_12_4
 hold_12_5 hold_12_6 hold_12_7 hold_12_8 hold_12_9 hold_12_10 hold_12_11
 hold_12_12 hold_13_1 hold_13_2 hold_13_3 hold_13_4 hold_13_5 hold_13_6
 hold_13_7 hold_13_8 hold_13_9 hold_13_10 hold_13_11 hold_13_12 hold_14_1
 hold_14_2 hold_14_3 hold_14_4 hold_14_5 hold_14_6 hold_14_7 hold_14_8
 hold_14_9 hold_14_10 hold_14_11 hold_14_12 hold_15_1 hold_15_2 hold_15_3
 hold_15_4 hold_15_5 hold_15_6 hold_15_7 hold_15_8 hold_15_9 hold_15_10
 hold_15_11 hold_15_12 switch_source_nic_in_0 switch_nic_out_0_sink
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
Generals
 start_0 start_1 start_2 start_3 start_4 start_5 start_6 start_7 start_8
 start_9 start_10 start_11 start_12 start_13 start_14 start_15
 group_start_1 val_end_0 group_end_0 group_start_2 group_start_3
 group_start_4 group_start_5 group_start_6 group_start_7 group_start_8
 group_start_9 group_start_10 group_start_11 group_start_12 group_start_13
 group_start_14 group_start_15 group_start_0 val_end_1 group_end_1
 val_end_2 group_end_2 val_end_3 group_end_3 val_end_4 group_end_4
 val_end_5 group_end_5 val_end_6 group_end_6 val_end_7 group_end_7
 val_end_8 group_end_8 val_end_9 group_end_9 val_end_10 group_end_10
 val_end_11 group_end_11 val_end_12 group_end_12 val_end_13 group_end_13
 val_end_14 group_end_14 val_end_15 group_end_15
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
End
