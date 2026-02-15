\ Model 2026-02-15_17-36-33
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  0.00208333333333333 flow_source_nic_in_0
   + 0.00208333333333333 flow_source_nic_in_1
   + 0.00208333333333333 flow_source_nic_in_2
   + 0.00208333333333333 flow_source_nic_in_3
   + 0.00208333333333333 flow_source_nic_in_4
   + 0.00208333333333333 flow_source_nic_in_5
   + 0.00208333333333333 flow_source_nic_in_6
   + 0.00208333333333333 flow_source_nic_in_7
   + 0.00208333333333333 flow_source_nic_in_8
   + 0.00208333333333333 flow_source_nic_in_9
   + 0.00208333333333333 flow_source_nic_in_10
   + 0.00208333333333333 flow_source_nic_in_11 + 0 val_end_0 + 0 val_end_1
   + 0 val_end_2 + 0 val_end_3 + 0 val_end_4 + 0 val_end_5 + 0 val_end_6
   + 0 val_end_7 + 0 val_end_8 + 0 val_end_9 + 0 val_end_10 + 0 val_end_11
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
 node_flow_constraint_nic_in_0: flow_source_nic_in_0 - flow_nic_in_0_0
   + flow_nic_out_1_nic_in_0 + flow_nic_out_2_nic_in_0
   + flow_nic_out_3_nic_in_0 + flow_nic_out_4_nic_in_0
   + flow_nic_out_5_nic_in_0 + flow_nic_out_6_nic_in_0
   + flow_nic_out_7_nic_in_0 + flow_nic_out_8_nic_in_0
   + flow_nic_out_9_nic_in_0 + flow_nic_out_10_nic_in_0
   + flow_nic_out_11_nic_in_0 = 0
 node_flow_constraint_nic_out_0: - flow_nic_out_0_sink
   - flow_nic_out_0_nic_in_1 - flow_nic_out_0_nic_in_2
   - flow_nic_out_0_nic_in_3 - flow_nic_out_0_nic_in_4
   - flow_nic_out_0_nic_in_5 - flow_nic_out_0_nic_in_6
   - flow_nic_out_0_nic_in_7 - flow_nic_out_0_nic_in_8
   - flow_nic_out_0_nic_in_9 - flow_nic_out_0_nic_in_10
   - flow_nic_out_0_nic_in_11 + flow_0_nic_out_0 = 0
 node_flow_constraint_nic_in_1: flow_nic_out_0_nic_in_1
   + flow_source_nic_in_1 - flow_nic_in_1_1 + flow_nic_out_2_nic_in_1
   + flow_nic_out_3_nic_in_1 + flow_nic_out_4_nic_in_1
   + flow_nic_out_5_nic_in_1 + flow_nic_out_6_nic_in_1
   + flow_nic_out_7_nic_in_1 + flow_nic_out_8_nic_in_1
   + flow_nic_out_9_nic_in_1 + flow_nic_out_10_nic_in_1
   + flow_nic_out_11_nic_in_1 = 0
 node_flow_constraint_nic_out_1: - flow_nic_out_1_sink
   - flow_nic_out_1_nic_in_0 - flow_nic_out_1_nic_in_2
   - flow_nic_out_1_nic_in_3 - flow_nic_out_1_nic_in_4
   - flow_nic_out_1_nic_in_5 - flow_nic_out_1_nic_in_6
   - flow_nic_out_1_nic_in_7 - flow_nic_out_1_nic_in_8
   - flow_nic_out_1_nic_in_9 - flow_nic_out_1_nic_in_10
   - flow_nic_out_1_nic_in_11 + flow_1_nic_out_1 = 0
 node_flow_constraint_nic_in_2: flow_nic_out_0_nic_in_2
   + flow_nic_out_1_nic_in_2 + flow_source_nic_in_2 - flow_nic_in_2_2
   + flow_nic_out_3_nic_in_2 + flow_nic_out_4_nic_in_2
   + flow_nic_out_5_nic_in_2 + flow_nic_out_6_nic_in_2
   + flow_nic_out_7_nic_in_2 + flow_nic_out_8_nic_in_2
   + flow_nic_out_9_nic_in_2 + flow_nic_out_10_nic_in_2
   + flow_nic_out_11_nic_in_2 = 0
 node_flow_constraint_nic_out_2: - flow_nic_out_2_sink
   - flow_nic_out_2_nic_in_0 - flow_nic_out_2_nic_in_1
   - flow_nic_out_2_nic_in_3 - flow_nic_out_2_nic_in_4
   - flow_nic_out_2_nic_in_5 - flow_nic_out_2_nic_in_6
   - flow_nic_out_2_nic_in_7 - flow_nic_out_2_nic_in_8
   - flow_nic_out_2_nic_in_9 - flow_nic_out_2_nic_in_10
   - flow_nic_out_2_nic_in_11 + flow_2_nic_out_2 = 0
 node_flow_constraint_nic_in_3: flow_nic_out_0_nic_in_3
   + flow_nic_out_1_nic_in_3 + flow_nic_out_2_nic_in_3
   + flow_source_nic_in_3 - flow_nic_in_3_3 + flow_nic_out_4_nic_in_3
   + flow_nic_out_5_nic_in_3 + flow_nic_out_6_nic_in_3
   + flow_nic_out_7_nic_in_3 + flow_nic_out_8_nic_in_3
   + flow_nic_out_9_nic_in_3 + flow_nic_out_10_nic_in_3
   + flow_nic_out_11_nic_in_3 = 0
 node_flow_constraint_nic_out_3: - flow_nic_out_3_sink
   - flow_nic_out_3_nic_in_0 - flow_nic_out_3_nic_in_1
   - flow_nic_out_3_nic_in_2 - flow_nic_out_3_nic_in_4
   - flow_nic_out_3_nic_in_5 - flow_nic_out_3_nic_in_6
   - flow_nic_out_3_nic_in_7 - flow_nic_out_3_nic_in_8
   - flow_nic_out_3_nic_in_9 - flow_nic_out_3_nic_in_10
   - flow_nic_out_3_nic_in_11 + flow_3_nic_out_3 = 0
 node_flow_constraint_nic_in_4: flow_nic_out_0_nic_in_4
   + flow_nic_out_1_nic_in_4 + flow_nic_out_2_nic_in_4
   + flow_nic_out_3_nic_in_4 + flow_source_nic_in_4 - flow_nic_in_4_4
   + flow_nic_out_5_nic_in_4 + flow_nic_out_6_nic_in_4
   + flow_nic_out_7_nic_in_4 + flow_nic_out_8_nic_in_4
   + flow_nic_out_9_nic_in_4 + flow_nic_out_10_nic_in_4
   + flow_nic_out_11_nic_in_4 = 0
 node_flow_constraint_nic_out_4: - flow_nic_out_4_sink
   - flow_nic_out_4_nic_in_0 - flow_nic_out_4_nic_in_1
   - flow_nic_out_4_nic_in_2 - flow_nic_out_4_nic_in_3
   - flow_nic_out_4_nic_in_5 - flow_nic_out_4_nic_in_6
   - flow_nic_out_4_nic_in_7 - flow_nic_out_4_nic_in_8
   - flow_nic_out_4_nic_in_9 - flow_nic_out_4_nic_in_10
   - flow_nic_out_4_nic_in_11 + flow_4_nic_out_4 = 0
 node_flow_constraint_nic_in_5: flow_nic_out_0_nic_in_5
   + flow_nic_out_1_nic_in_5 + flow_nic_out_2_nic_in_5
   + flow_nic_out_3_nic_in_5 + flow_nic_out_4_nic_in_5
   + flow_source_nic_in_5 - flow_nic_in_5_5 + flow_nic_out_6_nic_in_5
   + flow_nic_out_7_nic_in_5 + flow_nic_out_8_nic_in_5
   + flow_nic_out_9_nic_in_5 + flow_nic_out_10_nic_in_5
   + flow_nic_out_11_nic_in_5 = 0
 node_flow_constraint_nic_out_5: - flow_nic_out_5_sink
   - flow_nic_out_5_nic_in_0 - flow_nic_out_5_nic_in_1
   - flow_nic_out_5_nic_in_2 - flow_nic_out_5_nic_in_3
   - flow_nic_out_5_nic_in_4 - flow_nic_out_5_nic_in_6
   - flow_nic_out_5_nic_in_7 - flow_nic_out_5_nic_in_8
   - flow_nic_out_5_nic_in_9 - flow_nic_out_5_nic_in_10
   - flow_nic_out_5_nic_in_11 + flow_5_nic_out_5 = 0
 node_flow_constraint_nic_in_6: flow_nic_out_0_nic_in_6
   + flow_nic_out_1_nic_in_6 + flow_nic_out_2_nic_in_6
   + flow_nic_out_3_nic_in_6 + flow_nic_out_4_nic_in_6
   + flow_nic_out_5_nic_in_6 + flow_source_nic_in_6 - flow_nic_in_6_6
   + flow_nic_out_7_nic_in_6 + flow_nic_out_8_nic_in_6
   + flow_nic_out_9_nic_in_6 + flow_nic_out_10_nic_in_6
   + flow_nic_out_11_nic_in_6 = 0
 node_flow_constraint_nic_out_6: - flow_nic_out_6_sink
   - flow_nic_out_6_nic_in_0 - flow_nic_out_6_nic_in_1
   - flow_nic_out_6_nic_in_2 - flow_nic_out_6_nic_in_3
   - flow_nic_out_6_nic_in_4 - flow_nic_out_6_nic_in_5
   - flow_nic_out_6_nic_in_7 - flow_nic_out_6_nic_in_8
   - flow_nic_out_6_nic_in_9 - flow_nic_out_6_nic_in_10
   - flow_nic_out_6_nic_in_11 + flow_6_nic_out_6 = 0
 node_flow_constraint_nic_in_7: flow_nic_out_0_nic_in_7
   + flow_nic_out_1_nic_in_7 + flow_nic_out_2_nic_in_7
   + flow_nic_out_3_nic_in_7 + flow_nic_out_4_nic_in_7
   + flow_nic_out_5_nic_in_7 + flow_nic_out_6_nic_in_7
   + flow_source_nic_in_7 - flow_nic_in_7_7 + flow_nic_out_8_nic_in_7
   + flow_nic_out_9_nic_in_7 + flow_nic_out_10_nic_in_7
   + flow_nic_out_11_nic_in_7 = 0
 node_flow_constraint_nic_out_7: - flow_nic_out_7_sink
   - flow_nic_out_7_nic_in_0 - flow_nic_out_7_nic_in_1
   - flow_nic_out_7_nic_in_2 - flow_nic_out_7_nic_in_3
   - flow_nic_out_7_nic_in_4 - flow_nic_out_7_nic_in_5
   - flow_nic_out_7_nic_in_6 - flow_nic_out_7_nic_in_8
   - flow_nic_out_7_nic_in_9 - flow_nic_out_7_nic_in_10
   - flow_nic_out_7_nic_in_11 + flow_7_nic_out_7 = 0
 node_flow_constraint_nic_in_8: flow_nic_out_0_nic_in_8
   + flow_nic_out_1_nic_in_8 + flow_nic_out_2_nic_in_8
   + flow_nic_out_3_nic_in_8 + flow_nic_out_4_nic_in_8
   + flow_nic_out_5_nic_in_8 + flow_nic_out_6_nic_in_8
   + flow_nic_out_7_nic_in_8 + flow_source_nic_in_8 - flow_nic_in_8_8
   + flow_nic_out_9_nic_in_8 + flow_nic_out_10_nic_in_8
   + flow_nic_out_11_nic_in_8 = 0
 node_flow_constraint_nic_out_8: - flow_nic_out_8_sink
   - flow_nic_out_8_nic_in_0 - flow_nic_out_8_nic_in_1
   - flow_nic_out_8_nic_in_2 - flow_nic_out_8_nic_in_3
   - flow_nic_out_8_nic_in_4 - flow_nic_out_8_nic_in_5
   - flow_nic_out_8_nic_in_6 - flow_nic_out_8_nic_in_7
   - flow_nic_out_8_nic_in_9 - flow_nic_out_8_nic_in_10
   - flow_nic_out_8_nic_in_11 + flow_8_nic_out_8 = 0
 node_flow_constraint_nic_in_9: flow_nic_out_0_nic_in_9
   + flow_nic_out_1_nic_in_9 + flow_nic_out_2_nic_in_9
   + flow_nic_out_3_nic_in_9 + flow_nic_out_4_nic_in_9
   + flow_nic_out_5_nic_in_9 + flow_nic_out_6_nic_in_9
   + flow_nic_out_7_nic_in_9 + flow_nic_out_8_nic_in_9
   + flow_source_nic_in_9 - flow_nic_in_9_9 + flow_nic_out_10_nic_in_9
   + flow_nic_out_11_nic_in_9 = 0
 node_flow_constraint_nic_out_9: - flow_nic_out_9_sink
   - flow_nic_out_9_nic_in_0 - flow_nic_out_9_nic_in_1
   - flow_nic_out_9_nic_in_2 - flow_nic_out_9_nic_in_3
   - flow_nic_out_9_nic_in_4 - flow_nic_out_9_nic_in_5
   - flow_nic_out_9_nic_in_6 - flow_nic_out_9_nic_in_7
   - flow_nic_out_9_nic_in_8 - flow_nic_out_9_nic_in_10
   - flow_nic_out_9_nic_in_11 + flow_9_nic_out_9 = 0
 node_flow_constraint_nic_in_10: flow_nic_out_0_nic_in_10
   + flow_nic_out_1_nic_in_10 + flow_nic_out_2_nic_in_10
   + flow_nic_out_3_nic_in_10 + flow_nic_out_4_nic_in_10
   + flow_nic_out_5_nic_in_10 + flow_nic_out_6_nic_in_10
   + flow_nic_out_7_nic_in_10 + flow_nic_out_8_nic_in_10
   + flow_nic_out_9_nic_in_10 + flow_source_nic_in_10 - flow_nic_in_10_10
   + flow_nic_out_11_nic_in_10 = 0
 node_flow_constraint_nic_out_10: - flow_nic_out_10_sink
   - flow_nic_out_10_nic_in_0 - flow_nic_out_10_nic_in_1
   - flow_nic_out_10_nic_in_2 - flow_nic_out_10_nic_in_3
   - flow_nic_out_10_nic_in_4 - flow_nic_out_10_nic_in_5
   - flow_nic_out_10_nic_in_6 - flow_nic_out_10_nic_in_7
   - flow_nic_out_10_nic_in_8 - flow_nic_out_10_nic_in_9
   - flow_nic_out_10_nic_in_11 + flow_10_nic_out_10 = 0
 node_flow_constraint_nic_in_11: flow_nic_out_0_nic_in_11
   + flow_nic_out_1_nic_in_11 + flow_nic_out_2_nic_in_11
   + flow_nic_out_3_nic_in_11 + flow_nic_out_4_nic_in_11
   + flow_nic_out_5_nic_in_11 + flow_nic_out_6_nic_in_11
   + flow_nic_out_7_nic_in_11 + flow_nic_out_8_nic_in_11
   + flow_nic_out_9_nic_in_11 + flow_nic_out_10_nic_in_11
   + flow_source_nic_in_11 - flow_nic_in_11_11 = 0
 node_flow_constraint_nic_out_11: - flow_nic_out_11_sink
   - flow_nic_out_11_nic_in_0 - flow_nic_out_11_nic_in_1
   - flow_nic_out_11_nic_in_2 - flow_nic_out_11_nic_in_3
   - flow_nic_out_11_nic_in_4 - flow_nic_out_11_nic_in_5
   - flow_nic_out_11_nic_in_6 - flow_nic_out_11_nic_in_7
   - flow_nic_out_11_nic_in_8 - flow_nic_out_11_nic_in_9
   - flow_nic_out_11_nic_in_10 + flow_11_nic_out_11 = 0
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
 edge_flow_constr_nic_in_11_11: flow_nic_in_11_11
   - 4.807692307692308e+06 switch_nic_in_11_11 <= 0
 edge_flow_constr_11_nic_out_11: flow_11_nic_out_11
   - 4.807692307692308e+06 switch_11_nic_out_11 <= 0
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
 force_min_throughput: flow_source_nic_in_0 + flow_source_nic_in_1
   + flow_source_nic_in_2 + flow_source_nic_in_3 + flow_source_nic_in_4
   + flow_source_nic_in_5 + flow_source_nic_in_6 + flow_source_nic_in_7
   + flow_source_nic_in_8 + flow_source_nic_in_9 + flow_source_nic_in_10
   + flow_source_nic_in_11 >= 1
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
 node_throughput_constraint_0: group_tp_active_0 = 0 -> - 2400 hold_0_1
   - 1200 hold_0_2 - 800 hold_0_3 - 600 hold_0_4 - 480 hold_0_5
   - 400 hold_0_6 - 342.8571428571428 hold_0_7 - 300 hold_0_8
   - 266.6666666666667 hold_0_9 - 240 hold_0_10
   - 218.1818181818182 hold_0_11 - 200 hold_0_12 + flow_nic_in_0_0 <= 0
 node_throughput_constraint_1: group_tp_active_1 = 0 -> - 2400 hold_1_1
   - 1200 hold_1_2 - 800 hold_1_3 - 600 hold_1_4 - 480 hold_1_5
   - 400 hold_1_6 - 342.8571428571428 hold_1_7 - 300 hold_1_8
   - 266.6666666666667 hold_1_9 - 240 hold_1_10
   - 218.1818181818182 hold_1_11 - 200 hold_1_12 + flow_nic_in_1_1 <= 0
 node_throughput_constraint_2: group_tp_active_2 = 0 -> - 2400 hold_2_1
   - 1200 hold_2_2 - 800 hold_2_3 - 600 hold_2_4 - 480 hold_2_5
   - 400 hold_2_6 - 342.8571428571428 hold_2_7 - 300 hold_2_8
   - 266.6666666666667 hold_2_9 - 240 hold_2_10
   - 218.1818181818182 hold_2_11 - 200 hold_2_12 + flow_nic_in_2_2 <= 0
 node_throughput_constraint_3: group_tp_active_3 = 0 -> - 2400 hold_3_1
   - 1200 hold_3_2 - 800 hold_3_3 - 600 hold_3_4 - 480 hold_3_5
   - 400 hold_3_6 - 342.8571428571428 hold_3_7 - 300 hold_3_8
   - 266.6666666666667 hold_3_9 - 240 hold_3_10
   - 218.1818181818182 hold_3_11 - 200 hold_3_12 + flow_nic_in_3_3 <= 0
 node_throughput_constraint_4: group_tp_active_4 = 0 -> - 2400 hold_4_1
   - 1200 hold_4_2 - 800 hold_4_3 - 600 hold_4_4 - 480 hold_4_5
   - 400 hold_4_6 - 342.8571428571428 hold_4_7 - 300 hold_4_8
   - 266.6666666666667 hold_4_9 - 240 hold_4_10
   - 218.1818181818182 hold_4_11 - 200 hold_4_12 + flow_nic_in_4_4 <= 0
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
 node_throughput_constraint_9: group_tp_active_9 = 0 -> - 2400 hold_9_1
   - 1200 hold_9_2 - 800 hold_9_3 - 600 hold_9_4 - 480 hold_9_5
   - 400 hold_9_6 - 342.8571428571428 hold_9_7 - 300 hold_9_8
   - 266.6666666666667 hold_9_9 - 240 hold_9_10
   - 218.1818181818182 hold_9_11 - 200 hold_9_12 + flow_nic_in_9_9 <= 0
 node_throughput_constraint_10: group_tp_active_10 = 0 -> - 2400 hold_10_1
   - 1200 hold_10_2 - 800 hold_10_3 - 600 hold_10_4 - 480 hold_10_5
   - 400 hold_10_6 - 342.8571428571428 hold_10_7 - 300 hold_10_8
   - 266.6666666666667 hold_10_9 - 240 hold_10_10
   - 218.1818181818182 hold_10_11 - 200 hold_10_12 + flow_nic_in_10_10
   <= 0
 node_throughput_constraint_11: group_tp_active_11 = 0 -> - 2400 hold_11_1
   - 1200 hold_11_2 - 800 hold_11_3 - 600 hold_11_4 - 480 hold_11_5
   - 400 hold_11_6 - 342.8571428571428 hold_11_7 - 300 hold_11_8
   - 266.6666666666667 hold_11_9 - 240 hold_11_10
   - 218.1818181818182 hold_11_11 - 200 hold_11_12 + flow_nic_in_11_11
   <= 0
 constr_intergroup_throughput_0_source_nic_in_0: switch_source_nic_in_0 = 1 ->
   - 1846.153846153846 hold_0_1 - 1043.478260869565 hold_0_2
   - 727.2727272727274 hold_0_3 - 558.1395348837209 hold_0_4
   - 452.8301886792453 hold_0_5 - 380.9523809523809 hold_0_6
   - 328.7671232876712 hold_0_7 - 289.1566265060241 hold_0_8
   - 258.0645161290323 hold_0_9 - 233.0097087378641 hold_0_10
   - 212.3893805309735 hold_0_11 - 195.1219512195122 hold_0_12
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_0_nic_in_1: switch_nic_out_0_nic_in_1 = 1 ->
   - 2364.532019704433 hold_1_1 - 1191.06699751861 hold_1_2
   - 796.0199004975125 hold_1_3 - 597.7584059775841 hold_1_4
   - 478.5643070787636 hold_1_5 - 399.002493765586 hold_1_6
   - 342.1240199572345 hold_1_7 - 299.4385527136619 hold_1_8
   - 266.2229617304492 hold_1_9 - 239.6405391912132 hold_1_10
   - 217.8847026781661 hold_1_11 - 199.7503121098626 hold_1_12
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_0_nic_in_2: switch_nic_out_0_nic_in_2 = 1 ->
   - 2364.532019704433 hold_2_1 - 1191.06699751861 hold_2_2
   - 796.0199004975125 hold_2_3 - 597.7584059775841 hold_2_4
   - 478.5643070787636 hold_2_5 - 399.002493765586 hold_2_6
   - 342.1240199572345 hold_2_7 - 299.4385527136619 hold_2_8
   - 266.2229617304492 hold_2_9 - 239.6405391912132 hold_2_10
   - 217.8847026781661 hold_2_11 - 199.7503121098626 hold_2_12
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_0_nic_in_3: switch_nic_out_0_nic_in_3 = 1 ->
   - 2364.532019704433 hold_3_1 - 1191.06699751861 hold_3_2
   - 796.0199004975125 hold_3_3 - 597.7584059775841 hold_3_4
   - 478.5643070787636 hold_3_5 - 399.002493765586 hold_3_6
   - 342.1240199572345 hold_3_7 - 299.4385527136619 hold_3_8
   - 266.2229617304492 hold_3_9 - 239.6405391912132 hold_3_10
   - 217.8847026781661 hold_3_11 - 199.7503121098626 hold_3_12
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_0_nic_in_4: switch_nic_out_0_nic_in_4 = 1 ->
   - 2330.097087378641 hold_4_1 - 1182.266009852217 hold_4_2
   - 792.0792079207921 hold_4_3 - 595.5334987593052 hold_4_4
   - 477.1371769383698 hold_4_5 - 398.0099502487562 hold_4_6
   - 341.3940256045519 hold_4_7 - 298.879202988792 hold_4_8
   - 265.78073089701 hold_4_9 - 239.2821535393818 hold_4_10
   - 217.5883952855847 hold_4_11 - 199.501246882793 hold_4_12
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_0_nic_in_5: switch_nic_out_0_nic_in_5 = 1 ->
   - 2330.097087378641 hold_5_1 - 1182.266009852217 hold_5_2
   - 792.0792079207921 hold_5_3 - 595.5334987593052 hold_5_4
   - 477.1371769383698 hold_5_5 - 398.0099502487562 hold_5_6
   - 341.3940256045519 hold_5_7 - 298.879202988792 hold_5_8
   - 265.78073089701 hold_5_9 - 239.2821535393818 hold_5_10
   - 217.5883952855847 hold_5_11 - 199.501246882793 hold_5_12
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_0_nic_in_6: switch_nic_out_0_nic_in_6 = 1 ->
   - 2330.097087378641 hold_6_1 - 1182.266009852217 hold_6_2
   - 792.0792079207921 hold_6_3 - 595.5334987593052 hold_6_4
   - 477.1371769383698 hold_6_5 - 398.0099502487562 hold_6_6
   - 341.3940256045519 hold_6_7 - 298.879202988792 hold_6_8
   - 265.78073089701 hold_6_9 - 239.2821535393818 hold_6_10
   - 217.5883952855847 hold_6_11 - 199.501246882793 hold_6_12
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
   - 2330.097087378641 hold_9_1 - 1182.266009852217 hold_9_2
   - 792.0792079207921 hold_9_3 - 595.5334987593052 hold_9_4
   - 477.1371769383698 hold_9_5 - 398.0099502487562 hold_9_6
   - 341.3940256045519 hold_9_7 - 298.879202988792 hold_9_8
   - 265.78073089701 hold_9_9 - 239.2821535393818 hold_9_10
   - 217.5883952855847 hold_9_11 - 199.501246882793 hold_9_12
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_0_nic_in_10: switch_nic_out_0_nic_in_10 = 1 ->
   - 2330.097087378641 hold_10_1 - 1182.266009852217 hold_10_2
   - 792.0792079207921 hold_10_3 - 595.5334987593052 hold_10_4
   - 477.1371769383698 hold_10_5 - 398.0099502487562 hold_10_6
   - 341.3940256045519 hold_10_7 - 298.879202988792 hold_10_8
   - 265.78073089701 hold_10_9 - 239.2821535393818 hold_10_10
   - 217.5883952855847 hold_10_11 - 199.501246882793 hold_10_12
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_0_nic_in_11: switch_nic_out_0_nic_in_11 = 1 ->
   - 2330.097087378641 hold_11_1 - 1182.266009852217 hold_11_2
   - 792.0792079207921 hold_11_3 - 595.5334987593052 hold_11_4
   - 477.1371769383698 hold_11_5 - 398.0099502487562 hold_11_6
   - 341.3940256045519 hold_11_7 - 298.879202988792 hold_11_8
   - 265.78073089701 hold_11_9 - 239.2821535393818 hold_11_10
   - 217.5883952855847 hold_11_11 - 199.501246882793 hold_11_12
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_1_source_nic_in_1: switch_source_nic_in_1 = 1 ->
   - 1846.153846153846 hold_1_1 - 1043.478260869565 hold_1_2
   - 727.2727272727274 hold_1_3 - 558.1395348837209 hold_1_4
   - 452.8301886792453 hold_1_5 - 380.9523809523809 hold_1_6
   - 328.7671232876712 hold_1_7 - 289.1566265060241 hold_1_8
   - 258.0645161290323 hold_1_9 - 233.0097087378641 hold_1_10
   - 212.3893805309735 hold_1_11 - 195.1219512195122 hold_1_12
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_0_nic_out_1_nic_in_0: switch_nic_out_1_nic_in_0 = 1 ->
   - 2364.532019704433 hold_0_1 - 1191.06699751861 hold_0_2
   - 796.0199004975125 hold_0_3 - 597.7584059775841 hold_0_4
   - 478.5643070787636 hold_0_5 - 399.002493765586 hold_0_6
   - 342.1240199572345 hold_0_7 - 299.4385527136619 hold_0_8
   - 266.2229617304492 hold_0_9 - 239.6405391912132 hold_0_10
   - 217.8847026781661 hold_0_11 - 199.7503121098626 hold_0_12
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_2_nic_out_1_nic_in_2: switch_nic_out_1_nic_in_2 = 1 ->
   - 2364.532019704433 hold_2_1 - 1191.06699751861 hold_2_2
   - 796.0199004975125 hold_2_3 - 597.7584059775841 hold_2_4
   - 478.5643070787636 hold_2_5 - 399.002493765586 hold_2_6
   - 342.1240199572345 hold_2_7 - 299.4385527136619 hold_2_8
   - 266.2229617304492 hold_2_9 - 239.6405391912132 hold_2_10
   - 217.8847026781661 hold_2_11 - 199.7503121098626 hold_2_12
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_1_nic_in_3: switch_nic_out_1_nic_in_3 = 1 ->
   - 2364.532019704433 hold_3_1 - 1191.06699751861 hold_3_2
   - 796.0199004975125 hold_3_3 - 597.7584059775841 hold_3_4
   - 478.5643070787636 hold_3_5 - 399.002493765586 hold_3_6
   - 342.1240199572345 hold_3_7 - 299.4385527136619 hold_3_8
   - 266.2229617304492 hold_3_9 - 239.6405391912132 hold_3_10
   - 217.8847026781661 hold_3_11 - 199.7503121098626 hold_3_12
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_1_nic_in_4: switch_nic_out_1_nic_in_4 = 1 ->
   - 2330.097087378641 hold_4_1 - 1182.266009852217 hold_4_2
   - 792.0792079207921 hold_4_3 - 595.5334987593052 hold_4_4
   - 477.1371769383698 hold_4_5 - 398.0099502487562 hold_4_6
   - 341.3940256045519 hold_4_7 - 298.879202988792 hold_4_8
   - 265.78073089701 hold_4_9 - 239.2821535393818 hold_4_10
   - 217.5883952855847 hold_4_11 - 199.501246882793 hold_4_12
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_1_nic_in_5: switch_nic_out_1_nic_in_5 = 1 ->
   - 2330.097087378641 hold_5_1 - 1182.266009852217 hold_5_2
   - 792.0792079207921 hold_5_3 - 595.5334987593052 hold_5_4
   - 477.1371769383698 hold_5_5 - 398.0099502487562 hold_5_6
   - 341.3940256045519 hold_5_7 - 298.879202988792 hold_5_8
   - 265.78073089701 hold_5_9 - 239.2821535393818 hold_5_10
   - 217.5883952855847 hold_5_11 - 199.501246882793 hold_5_12
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_1_nic_in_6: switch_nic_out_1_nic_in_6 = 1 ->
   - 2330.097087378641 hold_6_1 - 1182.266009852217 hold_6_2
   - 792.0792079207921 hold_6_3 - 595.5334987593052 hold_6_4
   - 477.1371769383698 hold_6_5 - 398.0099502487562 hold_6_6
   - 341.3940256045519 hold_6_7 - 298.879202988792 hold_6_8
   - 265.78073089701 hold_6_9 - 239.2821535393818 hold_6_10
   - 217.5883952855847 hold_6_11 - 199.501246882793 hold_6_12
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_1_nic_in_7: switch_nic_out_1_nic_in_7 = 1 ->
   - 2330.097087378641 hold_7_1 - 1182.266009852217 hold_7_2
   - 792.0792079207921 hold_7_3 - 595.5334987593052 hold_7_4
   - 477.1371769383698 hold_7_5 - 398.0099502487562 hold_7_6
   - 341.3940256045519 hold_7_7 - 298.879202988792 hold_7_8
   - 265.78073089701 hold_7_9 - 239.2821535393818 hold_7_10
   - 217.5883952855847 hold_7_11 - 199.501246882793 hold_7_12
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_1_nic_in_8: switch_nic_out_1_nic_in_8 = 1 ->
   - 2330.097087378641 hold_8_1 - 1182.266009852217 hold_8_2
   - 792.0792079207921 hold_8_3 - 595.5334987593052 hold_8_4
   - 477.1371769383698 hold_8_5 - 398.0099502487562 hold_8_6
   - 341.3940256045519 hold_8_7 - 298.879202988792 hold_8_8
   - 265.78073089701 hold_8_9 - 239.2821535393818 hold_8_10
   - 217.5883952855847 hold_8_11 - 199.501246882793 hold_8_12
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_1_nic_in_9: switch_nic_out_1_nic_in_9 = 1 ->
   - 2330.097087378641 hold_9_1 - 1182.266009852217 hold_9_2
   - 792.0792079207921 hold_9_3 - 595.5334987593052 hold_9_4
   - 477.1371769383698 hold_9_5 - 398.0099502487562 hold_9_6
   - 341.3940256045519 hold_9_7 - 298.879202988792 hold_9_8
   - 265.78073089701 hold_9_9 - 239.2821535393818 hold_9_10
   - 217.5883952855847 hold_9_11 - 199.501246882793 hold_9_12
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_1_nic_in_10: switch_nic_out_1_nic_in_10 = 1 ->
   - 2330.097087378641 hold_10_1 - 1182.266009852217 hold_10_2
   - 792.0792079207921 hold_10_3 - 595.5334987593052 hold_10_4
   - 477.1371769383698 hold_10_5 - 398.0099502487562 hold_10_6
   - 341.3940256045519 hold_10_7 - 298.879202988792 hold_10_8
   - 265.78073089701 hold_10_9 - 239.2821535393818 hold_10_10
   - 217.5883952855847 hold_10_11 - 199.501246882793 hold_10_12
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_1_nic_in_11: switch_nic_out_1_nic_in_11 = 1 ->
   - 2330.097087378641 hold_11_1 - 1182.266009852217 hold_11_2
   - 792.0792079207921 hold_11_3 - 595.5334987593052 hold_11_4
   - 477.1371769383698 hold_11_5 - 398.0099502487562 hold_11_6
   - 341.3940256045519 hold_11_7 - 298.879202988792 hold_11_8
   - 265.78073089701 hold_11_9 - 239.2821535393818 hold_11_10
   - 217.5883952855847 hold_11_11 - 199.501246882793 hold_11_12
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_2_source_nic_in_2: switch_source_nic_in_2 = 1 ->
   - 1846.153846153846 hold_2_1 - 1043.478260869565 hold_2_2
   - 727.2727272727274 hold_2_3 - 558.1395348837209 hold_2_4
   - 452.8301886792453 hold_2_5 - 380.9523809523809 hold_2_6
   - 328.7671232876712 hold_2_7 - 289.1566265060241 hold_2_8
   - 258.0645161290323 hold_2_9 - 233.0097087378641 hold_2_10
   - 212.3893805309735 hold_2_11 - 195.1219512195122 hold_2_12
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_0_nic_out_2_nic_in_0: switch_nic_out_2_nic_in_0 = 1 ->
   - 2364.532019704433 hold_0_1 - 1191.06699751861 hold_0_2
   - 796.0199004975125 hold_0_3 - 597.7584059775841 hold_0_4
   - 478.5643070787636 hold_0_5 - 399.002493765586 hold_0_6
   - 342.1240199572345 hold_0_7 - 299.4385527136619 hold_0_8
   - 266.2229617304492 hold_0_9 - 239.6405391912132 hold_0_10
   - 217.8847026781661 hold_0_11 - 199.7503121098626 hold_0_12
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_2_nic_in_1: switch_nic_out_2_nic_in_1 = 1 ->
   - 2364.532019704433 hold_1_1 - 1191.06699751861 hold_1_2
   - 796.0199004975125 hold_1_3 - 597.7584059775841 hold_1_4
   - 478.5643070787636 hold_1_5 - 399.002493765586 hold_1_6
   - 342.1240199572345 hold_1_7 - 299.4385527136619 hold_1_8
   - 266.2229617304492 hold_1_9 - 239.6405391912132 hold_1_10
   - 217.8847026781661 hold_1_11 - 199.7503121098626 hold_1_12
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_3_nic_out_2_nic_in_3: switch_nic_out_2_nic_in_3 = 1 ->
   - 2364.532019704433 hold_3_1 - 1191.06699751861 hold_3_2
   - 796.0199004975125 hold_3_3 - 597.7584059775841 hold_3_4
   - 478.5643070787636 hold_3_5 - 399.002493765586 hold_3_6
   - 342.1240199572345 hold_3_7 - 299.4385527136619 hold_3_8
   - 266.2229617304492 hold_3_9 - 239.6405391912132 hold_3_10
   - 217.8847026781661 hold_3_11 - 199.7503121098626 hold_3_12
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_2_nic_in_4: switch_nic_out_2_nic_in_4 = 1 ->
   - 2330.097087378641 hold_4_1 - 1182.266009852217 hold_4_2
   - 792.0792079207921 hold_4_3 - 595.5334987593052 hold_4_4
   - 477.1371769383698 hold_4_5 - 398.0099502487562 hold_4_6
   - 341.3940256045519 hold_4_7 - 298.879202988792 hold_4_8
   - 265.78073089701 hold_4_9 - 239.2821535393818 hold_4_10
   - 217.5883952855847 hold_4_11 - 199.501246882793 hold_4_12
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_2_nic_in_5: switch_nic_out_2_nic_in_5 = 1 ->
   - 2330.097087378641 hold_5_1 - 1182.266009852217 hold_5_2
   - 792.0792079207921 hold_5_3 - 595.5334987593052 hold_5_4
   - 477.1371769383698 hold_5_5 - 398.0099502487562 hold_5_6
   - 341.3940256045519 hold_5_7 - 298.879202988792 hold_5_8
   - 265.78073089701 hold_5_9 - 239.2821535393818 hold_5_10
   - 217.5883952855847 hold_5_11 - 199.501246882793 hold_5_12
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_2_nic_in_6: switch_nic_out_2_nic_in_6 = 1 ->
   - 2330.097087378641 hold_6_1 - 1182.266009852217 hold_6_2
   - 792.0792079207921 hold_6_3 - 595.5334987593052 hold_6_4
   - 477.1371769383698 hold_6_5 - 398.0099502487562 hold_6_6
   - 341.3940256045519 hold_6_7 - 298.879202988792 hold_6_8
   - 265.78073089701 hold_6_9 - 239.2821535393818 hold_6_10
   - 217.5883952855847 hold_6_11 - 199.501246882793 hold_6_12
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_2_nic_in_7: switch_nic_out_2_nic_in_7 = 1 ->
   - 2330.097087378641 hold_7_1 - 1182.266009852217 hold_7_2
   - 792.0792079207921 hold_7_3 - 595.5334987593052 hold_7_4
   - 477.1371769383698 hold_7_5 - 398.0099502487562 hold_7_6
   - 341.3940256045519 hold_7_7 - 298.879202988792 hold_7_8
   - 265.78073089701 hold_7_9 - 239.2821535393818 hold_7_10
   - 217.5883952855847 hold_7_11 - 199.501246882793 hold_7_12
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_2_nic_in_8: switch_nic_out_2_nic_in_8 = 1 ->
   - 2330.097087378641 hold_8_1 - 1182.266009852217 hold_8_2
   - 792.0792079207921 hold_8_3 - 595.5334987593052 hold_8_4
   - 477.1371769383698 hold_8_5 - 398.0099502487562 hold_8_6
   - 341.3940256045519 hold_8_7 - 298.879202988792 hold_8_8
   - 265.78073089701 hold_8_9 - 239.2821535393818 hold_8_10
   - 217.5883952855847 hold_8_11 - 199.501246882793 hold_8_12
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_2_nic_in_9: switch_nic_out_2_nic_in_9 = 1 ->
   - 2330.097087378641 hold_9_1 - 1182.266009852217 hold_9_2
   - 792.0792079207921 hold_9_3 - 595.5334987593052 hold_9_4
   - 477.1371769383698 hold_9_5 - 398.0099502487562 hold_9_6
   - 341.3940256045519 hold_9_7 - 298.879202988792 hold_9_8
   - 265.78073089701 hold_9_9 - 239.2821535393818 hold_9_10
   - 217.5883952855847 hold_9_11 - 199.501246882793 hold_9_12
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_2_nic_in_10: switch_nic_out_2_nic_in_10 = 1 ->
   - 2330.097087378641 hold_10_1 - 1182.266009852217 hold_10_2
   - 792.0792079207921 hold_10_3 - 595.5334987593052 hold_10_4
   - 477.1371769383698 hold_10_5 - 398.0099502487562 hold_10_6
   - 341.3940256045519 hold_10_7 - 298.879202988792 hold_10_8
   - 265.78073089701 hold_10_9 - 239.2821535393818 hold_10_10
   - 217.5883952855847 hold_10_11 - 199.501246882793 hold_10_12
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_2_nic_in_11: switch_nic_out_2_nic_in_11 = 1 ->
   - 2330.097087378641 hold_11_1 - 1182.266009852217 hold_11_2
   - 792.0792079207921 hold_11_3 - 595.5334987593052 hold_11_4
   - 477.1371769383698 hold_11_5 - 398.0099502487562 hold_11_6
   - 341.3940256045519 hold_11_7 - 298.879202988792 hold_11_8
   - 265.78073089701 hold_11_9 - 239.2821535393818 hold_11_10
   - 217.5883952855847 hold_11_11 - 199.501246882793 hold_11_12
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_3_source_nic_in_3: switch_source_nic_in_3 = 1 ->
   - 1846.153846153846 hold_3_1 - 1043.478260869565 hold_3_2
   - 727.2727272727274 hold_3_3 - 558.1395348837209 hold_3_4
   - 452.8301886792453 hold_3_5 - 380.9523809523809 hold_3_6
   - 328.7671232876712 hold_3_7 - 289.1566265060241 hold_3_8
   - 258.0645161290323 hold_3_9 - 233.0097087378641 hold_3_10
   - 212.3893805309735 hold_3_11 - 195.1219512195122 hold_3_12
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_0_nic_out_3_nic_in_0: switch_nic_out_3_nic_in_0 = 1 ->
   - 2364.532019704433 hold_0_1 - 1191.06699751861 hold_0_2
   - 796.0199004975125 hold_0_3 - 597.7584059775841 hold_0_4
   - 478.5643070787636 hold_0_5 - 399.002493765586 hold_0_6
   - 342.1240199572345 hold_0_7 - 299.4385527136619 hold_0_8
   - 266.2229617304492 hold_0_9 - 239.6405391912132 hold_0_10
   - 217.8847026781661 hold_0_11 - 199.7503121098626 hold_0_12
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_3_nic_in_1: switch_nic_out_3_nic_in_1 = 1 ->
   - 2364.532019704433 hold_1_1 - 1191.06699751861 hold_1_2
   - 796.0199004975125 hold_1_3 - 597.7584059775841 hold_1_4
   - 478.5643070787636 hold_1_5 - 399.002493765586 hold_1_6
   - 342.1240199572345 hold_1_7 - 299.4385527136619 hold_1_8
   - 266.2229617304492 hold_1_9 - 239.6405391912132 hold_1_10
   - 217.8847026781661 hold_1_11 - 199.7503121098626 hold_1_12
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_3_nic_in_2: switch_nic_out_3_nic_in_2 = 1 ->
   - 2364.532019704433 hold_2_1 - 1191.06699751861 hold_2_2
   - 796.0199004975125 hold_2_3 - 597.7584059775841 hold_2_4
   - 478.5643070787636 hold_2_5 - 399.002493765586 hold_2_6
   - 342.1240199572345 hold_2_7 - 299.4385527136619 hold_2_8
   - 266.2229617304492 hold_2_9 - 239.6405391912132 hold_2_10
   - 217.8847026781661 hold_2_11 - 199.7503121098626 hold_2_12
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_4_nic_out_3_nic_in_4: switch_nic_out_3_nic_in_4 = 1 ->
   - 2330.097087378641 hold_4_1 - 1182.266009852217 hold_4_2
   - 792.0792079207921 hold_4_3 - 595.5334987593052 hold_4_4
   - 477.1371769383698 hold_4_5 - 398.0099502487562 hold_4_6
   - 341.3940256045519 hold_4_7 - 298.879202988792 hold_4_8
   - 265.78073089701 hold_4_9 - 239.2821535393818 hold_4_10
   - 217.5883952855847 hold_4_11 - 199.501246882793 hold_4_12
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_3_nic_in_5: switch_nic_out_3_nic_in_5 = 1 ->
   - 2330.097087378641 hold_5_1 - 1182.266009852217 hold_5_2
   - 792.0792079207921 hold_5_3 - 595.5334987593052 hold_5_4
   - 477.1371769383698 hold_5_5 - 398.0099502487562 hold_5_6
   - 341.3940256045519 hold_5_7 - 298.879202988792 hold_5_8
   - 265.78073089701 hold_5_9 - 239.2821535393818 hold_5_10
   - 217.5883952855847 hold_5_11 - 199.501246882793 hold_5_12
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_3_nic_in_6: switch_nic_out_3_nic_in_6 = 1 ->
   - 2330.097087378641 hold_6_1 - 1182.266009852217 hold_6_2
   - 792.0792079207921 hold_6_3 - 595.5334987593052 hold_6_4
   - 477.1371769383698 hold_6_5 - 398.0099502487562 hold_6_6
   - 341.3940256045519 hold_6_7 - 298.879202988792 hold_6_8
   - 265.78073089701 hold_6_9 - 239.2821535393818 hold_6_10
   - 217.5883952855847 hold_6_11 - 199.501246882793 hold_6_12
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_7_nic_out_3_nic_in_7: switch_nic_out_3_nic_in_7 = 1 ->
   - 2330.097087378641 hold_7_1 - 1182.266009852217 hold_7_2
   - 792.0792079207921 hold_7_3 - 595.5334987593052 hold_7_4
   - 477.1371769383698 hold_7_5 - 398.0099502487562 hold_7_6
   - 341.3940256045519 hold_7_7 - 298.879202988792 hold_7_8
   - 265.78073089701 hold_7_9 - 239.2821535393818 hold_7_10
   - 217.5883952855847 hold_7_11 - 199.501246882793 hold_7_12
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_8_nic_out_3_nic_in_8: switch_nic_out_3_nic_in_8 = 1 ->
   - 2330.097087378641 hold_8_1 - 1182.266009852217 hold_8_2
   - 792.0792079207921 hold_8_3 - 595.5334987593052 hold_8_4
   - 477.1371769383698 hold_8_5 - 398.0099502487562 hold_8_6
   - 341.3940256045519 hold_8_7 - 298.879202988792 hold_8_8
   - 265.78073089701 hold_8_9 - 239.2821535393818 hold_8_10
   - 217.5883952855847 hold_8_11 - 199.501246882793 hold_8_12
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_3_nic_in_9: switch_nic_out_3_nic_in_9 = 1 ->
   - 2330.097087378641 hold_9_1 - 1182.266009852217 hold_9_2
   - 792.0792079207921 hold_9_3 - 595.5334987593052 hold_9_4
   - 477.1371769383698 hold_9_5 - 398.0099502487562 hold_9_6
   - 341.3940256045519 hold_9_7 - 298.879202988792 hold_9_8
   - 265.78073089701 hold_9_9 - 239.2821535393818 hold_9_10
   - 217.5883952855847 hold_9_11 - 199.501246882793 hold_9_12
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_3_nic_in_10: switch_nic_out_3_nic_in_10 = 1 ->
   - 2330.097087378641 hold_10_1 - 1182.266009852217 hold_10_2
   - 792.0792079207921 hold_10_3 - 595.5334987593052 hold_10_4
   - 477.1371769383698 hold_10_5 - 398.0099502487562 hold_10_6
   - 341.3940256045519 hold_10_7 - 298.879202988792 hold_10_8
   - 265.78073089701 hold_10_9 - 239.2821535393818 hold_10_10
   - 217.5883952855847 hold_10_11 - 199.501246882793 hold_10_12
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_3_nic_in_11: switch_nic_out_3_nic_in_11 = 1 ->
   - 2330.097087378641 hold_11_1 - 1182.266009852217 hold_11_2
   - 792.0792079207921 hold_11_3 - 595.5334987593052 hold_11_4
   - 477.1371769383698 hold_11_5 - 398.0099502487562 hold_11_6
   - 341.3940256045519 hold_11_7 - 298.879202988792 hold_11_8
   - 265.78073089701 hold_11_9 - 239.2821535393818 hold_11_10
   - 217.5883952855847 hold_11_11 - 199.501246882793 hold_11_12
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_4_source_nic_in_4: switch_source_nic_in_4 = 1 ->
   - 1846.153846153846 hold_4_1 - 1043.478260869565 hold_4_2
   - 727.2727272727274 hold_4_3 - 558.1395348837209 hold_4_4
   - 452.8301886792453 hold_4_5 - 380.9523809523809 hold_4_6
   - 328.7671232876712 hold_4_7 - 289.1566265060241 hold_4_8
   - 258.0645161290323 hold_4_9 - 233.0097087378641 hold_4_10
   - 212.3893805309735 hold_4_11 - 195.1219512195122 hold_4_12
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_0_nic_out_4_nic_in_0: switch_nic_out_4_nic_in_0 = 1 ->
   - 2330.097087378641 hold_0_1 - 1182.266009852217 hold_0_2
   - 792.0792079207921 hold_0_3 - 595.5334987593052 hold_0_4
   - 477.1371769383698 hold_0_5 - 398.0099502487562 hold_0_6
   - 341.3940256045519 hold_0_7 - 298.879202988792 hold_0_8
   - 265.78073089701 hold_0_9 - 239.2821535393818 hold_0_10
   - 217.5883952855847 hold_0_11 - 199.501246882793 hold_0_12
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_4_nic_in_1: switch_nic_out_4_nic_in_1 = 1 ->
   - 2330.097087378641 hold_1_1 - 1182.266009852217 hold_1_2
   - 792.0792079207921 hold_1_3 - 595.5334987593052 hold_1_4
   - 477.1371769383698 hold_1_5 - 398.0099502487562 hold_1_6
   - 341.3940256045519 hold_1_7 - 298.879202988792 hold_1_8
   - 265.78073089701 hold_1_9 - 239.2821535393818 hold_1_10
   - 217.5883952855847 hold_1_11 - 199.501246882793 hold_1_12
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_4_nic_in_2: switch_nic_out_4_nic_in_2 = 1 ->
   - 2330.097087378641 hold_2_1 - 1182.266009852217 hold_2_2
   - 792.0792079207921 hold_2_3 - 595.5334987593052 hold_2_4
   - 477.1371769383698 hold_2_5 - 398.0099502487562 hold_2_6
   - 341.3940256045519 hold_2_7 - 298.879202988792 hold_2_8
   - 265.78073089701 hold_2_9 - 239.2821535393818 hold_2_10
   - 217.5883952855847 hold_2_11 - 199.501246882793 hold_2_12
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_4_nic_in_3: switch_nic_out_4_nic_in_3 = 1 ->
   - 2330.097087378641 hold_3_1 - 1182.266009852217 hold_3_2
   - 792.0792079207921 hold_3_3 - 595.5334987593052 hold_3_4
   - 477.1371769383698 hold_3_5 - 398.0099502487562 hold_3_6
   - 341.3940256045519 hold_3_7 - 298.879202988792 hold_3_8
   - 265.78073089701 hold_3_9 - 239.2821535393818 hold_3_10
   - 217.5883952855847 hold_3_11 - 199.501246882793 hold_3_12
   + flow_nic_in_3_3 <= 0
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
   - 2364.532019704433 hold_7_1 - 1191.06699751861 hold_7_2
   - 796.0199004975125 hold_7_3 - 597.7584059775841 hold_7_4
   - 478.5643070787636 hold_7_5 - 399.002493765586 hold_7_6
   - 342.1240199572345 hold_7_7 - 299.4385527136619 hold_7_8
   - 266.2229617304492 hold_7_9 - 239.6405391912132 hold_7_10
   - 217.8847026781661 hold_7_11 - 199.7503121098626 hold_7_12
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
   - 2330.097087378641 hold_9_1 - 1182.266009852217 hold_9_2
   - 792.0792079207921 hold_9_3 - 595.5334987593052 hold_9_4
   - 477.1371769383698 hold_9_5 - 398.0099502487562 hold_9_6
   - 341.3940256045519 hold_9_7 - 298.879202988792 hold_9_8
   - 265.78073089701 hold_9_9 - 239.2821535393818 hold_9_10
   - 217.5883952855847 hold_9_11 - 199.501246882793 hold_9_12
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_4_nic_in_10: switch_nic_out_4_nic_in_10 = 1 ->
   - 2330.097087378641 hold_10_1 - 1182.266009852217 hold_10_2
   - 792.0792079207921 hold_10_3 - 595.5334987593052 hold_10_4
   - 477.1371769383698 hold_10_5 - 398.0099502487562 hold_10_6
   - 341.3940256045519 hold_10_7 - 298.879202988792 hold_10_8
   - 265.78073089701 hold_10_9 - 239.2821535393818 hold_10_10
   - 217.5883952855847 hold_10_11 - 199.501246882793 hold_10_12
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_4_nic_in_11: switch_nic_out_4_nic_in_11 = 1 ->
   - 2330.097087378641 hold_11_1 - 1182.266009852217 hold_11_2
   - 792.0792079207921 hold_11_3 - 595.5334987593052 hold_11_4
   - 477.1371769383698 hold_11_5 - 398.0099502487562 hold_11_6
   - 341.3940256045519 hold_11_7 - 298.879202988792 hold_11_8
   - 265.78073089701 hold_11_9 - 239.2821535393818 hold_11_10
   - 217.5883952855847 hold_11_11 - 199.501246882793 hold_11_12
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_5_source_nic_in_5: switch_source_nic_in_5 = 1 ->
   - 1846.153846153846 hold_5_1 - 1043.478260869565 hold_5_2
   - 727.2727272727274 hold_5_3 - 558.1395348837209 hold_5_4
   - 452.8301886792453 hold_5_5 - 380.9523809523809 hold_5_6
   - 328.7671232876712 hold_5_7 - 289.1566265060241 hold_5_8
   - 258.0645161290323 hold_5_9 - 233.0097087378641 hold_5_10
   - 212.3893805309735 hold_5_11 - 195.1219512195122 hold_5_12
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_0_nic_out_5_nic_in_0: switch_nic_out_5_nic_in_0 = 1 ->
   - 2330.097087378641 hold_0_1 - 1182.266009852217 hold_0_2
   - 792.0792079207921 hold_0_3 - 595.5334987593052 hold_0_4
   - 477.1371769383698 hold_0_5 - 398.0099502487562 hold_0_6
   - 341.3940256045519 hold_0_7 - 298.879202988792 hold_0_8
   - 265.78073089701 hold_0_9 - 239.2821535393818 hold_0_10
   - 217.5883952855847 hold_0_11 - 199.501246882793 hold_0_12
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_5_nic_in_1: switch_nic_out_5_nic_in_1 = 1 ->
   - 2330.097087378641 hold_1_1 - 1182.266009852217 hold_1_2
   - 792.0792079207921 hold_1_3 - 595.5334987593052 hold_1_4
   - 477.1371769383698 hold_1_5 - 398.0099502487562 hold_1_6
   - 341.3940256045519 hold_1_7 - 298.879202988792 hold_1_8
   - 265.78073089701 hold_1_9 - 239.2821535393818 hold_1_10
   - 217.5883952855847 hold_1_11 - 199.501246882793 hold_1_12
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_5_nic_in_2: switch_nic_out_5_nic_in_2 = 1 ->
   - 2330.097087378641 hold_2_1 - 1182.266009852217 hold_2_2
   - 792.0792079207921 hold_2_3 - 595.5334987593052 hold_2_4
   - 477.1371769383698 hold_2_5 - 398.0099502487562 hold_2_6
   - 341.3940256045519 hold_2_7 - 298.879202988792 hold_2_8
   - 265.78073089701 hold_2_9 - 239.2821535393818 hold_2_10
   - 217.5883952855847 hold_2_11 - 199.501246882793 hold_2_12
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_5_nic_in_3: switch_nic_out_5_nic_in_3 = 1 ->
   - 2330.097087378641 hold_3_1 - 1182.266009852217 hold_3_2
   - 792.0792079207921 hold_3_3 - 595.5334987593052 hold_3_4
   - 477.1371769383698 hold_3_5 - 398.0099502487562 hold_3_6
   - 341.3940256045519 hold_3_7 - 298.879202988792 hold_3_8
   - 265.78073089701 hold_3_9 - 239.2821535393818 hold_3_10
   - 217.5883952855847 hold_3_11 - 199.501246882793 hold_3_12
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_5_nic_in_4: switch_nic_out_5_nic_in_4 = 1 ->
   - 2364.532019704433 hold_4_1 - 1191.06699751861 hold_4_2
   - 796.0199004975125 hold_4_3 - 597.7584059775841 hold_4_4
   - 478.5643070787636 hold_4_5 - 399.002493765586 hold_4_6
   - 342.1240199572345 hold_4_7 - 299.4385527136619 hold_4_8
   - 266.2229617304492 hold_4_9 - 239.6405391912132 hold_4_10
   - 217.8847026781661 hold_4_11 - 199.7503121098626 hold_4_12
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
   - 2364.532019704433 hold_7_1 - 1191.06699751861 hold_7_2
   - 796.0199004975125 hold_7_3 - 597.7584059775841 hold_7_4
   - 478.5643070787636 hold_7_5 - 399.002493765586 hold_7_6
   - 342.1240199572345 hold_7_7 - 299.4385527136619 hold_7_8
   - 266.2229617304492 hold_7_9 - 239.6405391912132 hold_7_10
   - 217.8847026781661 hold_7_11 - 199.7503121098626 hold_7_12
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
   - 2330.097087378641 hold_9_1 - 1182.266009852217 hold_9_2
   - 792.0792079207921 hold_9_3 - 595.5334987593052 hold_9_4
   - 477.1371769383698 hold_9_5 - 398.0099502487562 hold_9_6
   - 341.3940256045519 hold_9_7 - 298.879202988792 hold_9_8
   - 265.78073089701 hold_9_9 - 239.2821535393818 hold_9_10
   - 217.5883952855847 hold_9_11 - 199.501246882793 hold_9_12
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_5_nic_in_10: switch_nic_out_5_nic_in_10 = 1 ->
   - 2330.097087378641 hold_10_1 - 1182.266009852217 hold_10_2
   - 792.0792079207921 hold_10_3 - 595.5334987593052 hold_10_4
   - 477.1371769383698 hold_10_5 - 398.0099502487562 hold_10_6
   - 341.3940256045519 hold_10_7 - 298.879202988792 hold_10_8
   - 265.78073089701 hold_10_9 - 239.2821535393818 hold_10_10
   - 217.5883952855847 hold_10_11 - 199.501246882793 hold_10_12
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_5_nic_in_11: switch_nic_out_5_nic_in_11 = 1 ->
   - 2330.097087378641 hold_11_1 - 1182.266009852217 hold_11_2
   - 792.0792079207921 hold_11_3 - 595.5334987593052 hold_11_4
   - 477.1371769383698 hold_11_5 - 398.0099502487562 hold_11_6
   - 341.3940256045519 hold_11_7 - 298.879202988792 hold_11_8
   - 265.78073089701 hold_11_9 - 239.2821535393818 hold_11_10
   - 217.5883952855847 hold_11_11 - 199.501246882793 hold_11_12
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_6_source_nic_in_6: switch_source_nic_in_6 = 1 ->
   - 1846.153846153846 hold_6_1 - 1043.478260869565 hold_6_2
   - 727.2727272727274 hold_6_3 - 558.1395348837209 hold_6_4
   - 452.8301886792453 hold_6_5 - 380.9523809523809 hold_6_6
   - 328.7671232876712 hold_6_7 - 289.1566265060241 hold_6_8
   - 258.0645161290323 hold_6_9 - 233.0097087378641 hold_6_10
   - 212.3893805309735 hold_6_11 - 195.1219512195122 hold_6_12
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_0_nic_out_6_nic_in_0: switch_nic_out_6_nic_in_0 = 1 ->
   - 2330.097087378641 hold_0_1 - 1182.266009852217 hold_0_2
   - 792.0792079207921 hold_0_3 - 595.5334987593052 hold_0_4
   - 477.1371769383698 hold_0_5 - 398.0099502487562 hold_0_6
   - 341.3940256045519 hold_0_7 - 298.879202988792 hold_0_8
   - 265.78073089701 hold_0_9 - 239.2821535393818 hold_0_10
   - 217.5883952855847 hold_0_11 - 199.501246882793 hold_0_12
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_6_nic_in_1: switch_nic_out_6_nic_in_1 = 1 ->
   - 2330.097087378641 hold_1_1 - 1182.266009852217 hold_1_2
   - 792.0792079207921 hold_1_3 - 595.5334987593052 hold_1_4
   - 477.1371769383698 hold_1_5 - 398.0099502487562 hold_1_6
   - 341.3940256045519 hold_1_7 - 298.879202988792 hold_1_8
   - 265.78073089701 hold_1_9 - 239.2821535393818 hold_1_10
   - 217.5883952855847 hold_1_11 - 199.501246882793 hold_1_12
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_6_nic_in_2: switch_nic_out_6_nic_in_2 = 1 ->
   - 2330.097087378641 hold_2_1 - 1182.266009852217 hold_2_2
   - 792.0792079207921 hold_2_3 - 595.5334987593052 hold_2_4
   - 477.1371769383698 hold_2_5 - 398.0099502487562 hold_2_6
   - 341.3940256045519 hold_2_7 - 298.879202988792 hold_2_8
   - 265.78073089701 hold_2_9 - 239.2821535393818 hold_2_10
   - 217.5883952855847 hold_2_11 - 199.501246882793 hold_2_12
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_6_nic_in_3: switch_nic_out_6_nic_in_3 = 1 ->
   - 2330.097087378641 hold_3_1 - 1182.266009852217 hold_3_2
   - 792.0792079207921 hold_3_3 - 595.5334987593052 hold_3_4
   - 477.1371769383698 hold_3_5 - 398.0099502487562 hold_3_6
   - 341.3940256045519 hold_3_7 - 298.879202988792 hold_3_8
   - 265.78073089701 hold_3_9 - 239.2821535393818 hold_3_10
   - 217.5883952855847 hold_3_11 - 199.501246882793 hold_3_12
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_6_nic_in_4: switch_nic_out_6_nic_in_4 = 1 ->
   - 2364.532019704433 hold_4_1 - 1191.06699751861 hold_4_2
   - 796.0199004975125 hold_4_3 - 597.7584059775841 hold_4_4
   - 478.5643070787636 hold_4_5 - 399.002493765586 hold_4_6
   - 342.1240199572345 hold_4_7 - 299.4385527136619 hold_4_8
   - 266.2229617304492 hold_4_9 - 239.6405391912132 hold_4_10
   - 217.8847026781661 hold_4_11 - 199.7503121098626 hold_4_12
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
   - 2364.532019704433 hold_7_1 - 1191.06699751861 hold_7_2
   - 796.0199004975125 hold_7_3 - 597.7584059775841 hold_7_4
   - 478.5643070787636 hold_7_5 - 399.002493765586 hold_7_6
   - 342.1240199572345 hold_7_7 - 299.4385527136619 hold_7_8
   - 266.2229617304492 hold_7_9 - 239.6405391912132 hold_7_10
   - 217.8847026781661 hold_7_11 - 199.7503121098626 hold_7_12
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
   - 2330.097087378641 hold_9_1 - 1182.266009852217 hold_9_2
   - 792.0792079207921 hold_9_3 - 595.5334987593052 hold_9_4
   - 477.1371769383698 hold_9_5 - 398.0099502487562 hold_9_6
   - 341.3940256045519 hold_9_7 - 298.879202988792 hold_9_8
   - 265.78073089701 hold_9_9 - 239.2821535393818 hold_9_10
   - 217.5883952855847 hold_9_11 - 199.501246882793 hold_9_12
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_6_nic_in_10: switch_nic_out_6_nic_in_10 = 1 ->
   - 2330.097087378641 hold_10_1 - 1182.266009852217 hold_10_2
   - 792.0792079207921 hold_10_3 - 595.5334987593052 hold_10_4
   - 477.1371769383698 hold_10_5 - 398.0099502487562 hold_10_6
   - 341.3940256045519 hold_10_7 - 298.879202988792 hold_10_8
   - 265.78073089701 hold_10_9 - 239.2821535393818 hold_10_10
   - 217.5883952855847 hold_10_11 - 199.501246882793 hold_10_12
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_6_nic_in_11: switch_nic_out_6_nic_in_11 = 1 ->
   - 2330.097087378641 hold_11_1 - 1182.266009852217 hold_11_2
   - 792.0792079207921 hold_11_3 - 595.5334987593052 hold_11_4
   - 477.1371769383698 hold_11_5 - 398.0099502487562 hold_11_6
   - 341.3940256045519 hold_11_7 - 298.879202988792 hold_11_8
   - 265.78073089701 hold_11_9 - 239.2821535393818 hold_11_10
   - 217.5883952855847 hold_11_11 - 199.501246882793 hold_11_12
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_7_source_nic_in_7: switch_source_nic_in_7 = 1 ->
   - 1846.153846153846 hold_7_1 - 1043.478260869565 hold_7_2
   - 727.2727272727274 hold_7_3 - 558.1395348837209 hold_7_4
   - 452.8301886792453 hold_7_5 - 380.9523809523809 hold_7_6
   - 328.7671232876712 hold_7_7 - 289.1566265060241 hold_7_8
   - 258.0645161290323 hold_7_9 - 233.0097087378641 hold_7_10
   - 212.3893805309735 hold_7_11 - 195.1219512195122 hold_7_12
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_0_nic_out_7_nic_in_0: switch_nic_out_7_nic_in_0 = 1 ->
   - 2330.097087378641 hold_0_1 - 1182.266009852217 hold_0_2
   - 792.0792079207921 hold_0_3 - 595.5334987593052 hold_0_4
   - 477.1371769383698 hold_0_5 - 398.0099502487562 hold_0_6
   - 341.3940256045519 hold_0_7 - 298.879202988792 hold_0_8
   - 265.78073089701 hold_0_9 - 239.2821535393818 hold_0_10
   - 217.5883952855847 hold_0_11 - 199.501246882793 hold_0_12
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_7_nic_in_1: switch_nic_out_7_nic_in_1 = 1 ->
   - 2330.097087378641 hold_1_1 - 1182.266009852217 hold_1_2
   - 792.0792079207921 hold_1_3 - 595.5334987593052 hold_1_4
   - 477.1371769383698 hold_1_5 - 398.0099502487562 hold_1_6
   - 341.3940256045519 hold_1_7 - 298.879202988792 hold_1_8
   - 265.78073089701 hold_1_9 - 239.2821535393818 hold_1_10
   - 217.5883952855847 hold_1_11 - 199.501246882793 hold_1_12
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_7_nic_in_2: switch_nic_out_7_nic_in_2 = 1 ->
   - 2330.097087378641 hold_2_1 - 1182.266009852217 hold_2_2
   - 792.0792079207921 hold_2_3 - 595.5334987593052 hold_2_4
   - 477.1371769383698 hold_2_5 - 398.0099502487562 hold_2_6
   - 341.3940256045519 hold_2_7 - 298.879202988792 hold_2_8
   - 265.78073089701 hold_2_9 - 239.2821535393818 hold_2_10
   - 217.5883952855847 hold_2_11 - 199.501246882793 hold_2_12
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_7_nic_in_3: switch_nic_out_7_nic_in_3 = 1 ->
   - 2330.097087378641 hold_3_1 - 1182.266009852217 hold_3_2
   - 792.0792079207921 hold_3_3 - 595.5334987593052 hold_3_4
   - 477.1371769383698 hold_3_5 - 398.0099502487562 hold_3_6
   - 341.3940256045519 hold_3_7 - 298.879202988792 hold_3_8
   - 265.78073089701 hold_3_9 - 239.2821535393818 hold_3_10
   - 217.5883952855847 hold_3_11 - 199.501246882793 hold_3_12
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_7_nic_in_4: switch_nic_out_7_nic_in_4 = 1 ->
   - 2364.532019704433 hold_4_1 - 1191.06699751861 hold_4_2
   - 796.0199004975125 hold_4_3 - 597.7584059775841 hold_4_4
   - 478.5643070787636 hold_4_5 - 399.002493765586 hold_4_6
   - 342.1240199572345 hold_4_7 - 299.4385527136619 hold_4_8
   - 266.2229617304492 hold_4_9 - 239.6405391912132 hold_4_10
   - 217.8847026781661 hold_4_11 - 199.7503121098626 hold_4_12
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_5_nic_out_7_nic_in_5: switch_nic_out_7_nic_in_5 = 1 ->
   - 2364.532019704433 hold_5_1 - 1191.06699751861 hold_5_2
   - 796.0199004975125 hold_5_3 - 597.7584059775841 hold_5_4
   - 478.5643070787636 hold_5_5 - 399.002493765586 hold_5_6
   - 342.1240199572345 hold_5_7 - 299.4385527136619 hold_5_8
   - 266.2229617304492 hold_5_9 - 239.6405391912132 hold_5_10
   - 217.8847026781661 hold_5_11 - 199.7503121098626 hold_5_12
   + flow_nic_in_5_5 <= 0
 constr_intergroup_throughput_6_nic_out_7_nic_in_6: switch_nic_out_7_nic_in_6 = 1 ->
   - 2364.532019704433 hold_6_1 - 1191.06699751861 hold_6_2
   - 796.0199004975125 hold_6_3 - 597.7584059775841 hold_6_4
   - 478.5643070787636 hold_6_5 - 399.002493765586 hold_6_6
   - 342.1240199572345 hold_6_7 - 299.4385527136619 hold_6_8
   - 266.2229617304492 hold_6_9 - 239.6405391912132 hold_6_10
   - 217.8847026781661 hold_6_11 - 199.7503121098626 hold_6_12
   + flow_nic_in_6_6 <= 0
 constr_intergroup_throughput_8_nic_out_7_nic_in_8: switch_nic_out_7_nic_in_8 = 1 ->
   - 2330.097087378641 hold_8_1 - 1182.266009852217 hold_8_2
   - 792.0792079207921 hold_8_3 - 595.5334987593052 hold_8_4
   - 477.1371769383698 hold_8_5 - 398.0099502487562 hold_8_6
   - 341.3940256045519 hold_8_7 - 298.879202988792 hold_8_8
   - 265.78073089701 hold_8_9 - 239.2821535393818 hold_8_10
   - 217.5883952855847 hold_8_11 - 199.501246882793 hold_8_12
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_7_nic_in_9: switch_nic_out_7_nic_in_9 = 1 ->
   - 2330.097087378641 hold_9_1 - 1182.266009852217 hold_9_2
   - 792.0792079207921 hold_9_3 - 595.5334987593052 hold_9_4
   - 477.1371769383698 hold_9_5 - 398.0099502487562 hold_9_6
   - 341.3940256045519 hold_9_7 - 298.879202988792 hold_9_8
   - 265.78073089701 hold_9_9 - 239.2821535393818 hold_9_10
   - 217.5883952855847 hold_9_11 - 199.501246882793 hold_9_12
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_7_nic_in_10: switch_nic_out_7_nic_in_10 = 1 ->
   - 2330.097087378641 hold_10_1 - 1182.266009852217 hold_10_2
   - 792.0792079207921 hold_10_3 - 595.5334987593052 hold_10_4
   - 477.1371769383698 hold_10_5 - 398.0099502487562 hold_10_6
   - 341.3940256045519 hold_10_7 - 298.879202988792 hold_10_8
   - 265.78073089701 hold_10_9 - 239.2821535393818 hold_10_10
   - 217.5883952855847 hold_10_11 - 199.501246882793 hold_10_12
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_7_nic_in_11: switch_nic_out_7_nic_in_11 = 1 ->
   - 2330.097087378641 hold_11_1 - 1182.266009852217 hold_11_2
   - 792.0792079207921 hold_11_3 - 595.5334987593052 hold_11_4
   - 477.1371769383698 hold_11_5 - 398.0099502487562 hold_11_6
   - 341.3940256045519 hold_11_7 - 298.879202988792 hold_11_8
   - 265.78073089701 hold_11_9 - 239.2821535393818 hold_11_10
   - 217.5883952855847 hold_11_11 - 199.501246882793 hold_11_12
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_8_source_nic_in_8: switch_source_nic_in_8 = 1 ->
   - 1846.153846153846 hold_8_1 - 1043.478260869565 hold_8_2
   - 727.2727272727274 hold_8_3 - 558.1395348837209 hold_8_4
   - 452.8301886792453 hold_8_5 - 380.9523809523809 hold_8_6
   - 328.7671232876712 hold_8_7 - 289.1566265060241 hold_8_8
   - 258.0645161290323 hold_8_9 - 233.0097087378641 hold_8_10
   - 212.3893805309735 hold_8_11 - 195.1219512195122 hold_8_12
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_0_nic_out_8_nic_in_0: switch_nic_out_8_nic_in_0 = 1 ->
   - 2330.097087378641 hold_0_1 - 1182.266009852217 hold_0_2
   - 792.0792079207921 hold_0_3 - 595.5334987593052 hold_0_4
   - 477.1371769383698 hold_0_5 - 398.0099502487562 hold_0_6
   - 341.3940256045519 hold_0_7 - 298.879202988792 hold_0_8
   - 265.78073089701 hold_0_9 - 239.2821535393818 hold_0_10
   - 217.5883952855847 hold_0_11 - 199.501246882793 hold_0_12
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_8_nic_in_1: switch_nic_out_8_nic_in_1 = 1 ->
   - 2330.097087378641 hold_1_1 - 1182.266009852217 hold_1_2
   - 792.0792079207921 hold_1_3 - 595.5334987593052 hold_1_4
   - 477.1371769383698 hold_1_5 - 398.0099502487562 hold_1_6
   - 341.3940256045519 hold_1_7 - 298.879202988792 hold_1_8
   - 265.78073089701 hold_1_9 - 239.2821535393818 hold_1_10
   - 217.5883952855847 hold_1_11 - 199.501246882793 hold_1_12
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_8_nic_in_2: switch_nic_out_8_nic_in_2 = 1 ->
   - 2330.097087378641 hold_2_1 - 1182.266009852217 hold_2_2
   - 792.0792079207921 hold_2_3 - 595.5334987593052 hold_2_4
   - 477.1371769383698 hold_2_5 - 398.0099502487562 hold_2_6
   - 341.3940256045519 hold_2_7 - 298.879202988792 hold_2_8
   - 265.78073089701 hold_2_9 - 239.2821535393818 hold_2_10
   - 217.5883952855847 hold_2_11 - 199.501246882793 hold_2_12
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_8_nic_in_3: switch_nic_out_8_nic_in_3 = 1 ->
   - 2330.097087378641 hold_3_1 - 1182.266009852217 hold_3_2
   - 792.0792079207921 hold_3_3 - 595.5334987593052 hold_3_4
   - 477.1371769383698 hold_3_5 - 398.0099502487562 hold_3_6
   - 341.3940256045519 hold_3_7 - 298.879202988792 hold_3_8
   - 265.78073089701 hold_3_9 - 239.2821535393818 hold_3_10
   - 217.5883952855847 hold_3_11 - 199.501246882793 hold_3_12
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_8_nic_in_4: switch_nic_out_8_nic_in_4 = 1 ->
   - 2330.097087378641 hold_4_1 - 1182.266009852217 hold_4_2
   - 792.0792079207921 hold_4_3 - 595.5334987593052 hold_4_4
   - 477.1371769383698 hold_4_5 - 398.0099502487562 hold_4_6
   - 341.3940256045519 hold_4_7 - 298.879202988792 hold_4_8
   - 265.78073089701 hold_4_9 - 239.2821535393818 hold_4_10
   - 217.5883952855847 hold_4_11 - 199.501246882793 hold_4_12
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
   - 2330.097087378641 hold_7_1 - 1182.266009852217 hold_7_2
   - 792.0792079207921 hold_7_3 - 595.5334987593052 hold_7_4
   - 477.1371769383698 hold_7_5 - 398.0099502487562 hold_7_6
   - 341.3940256045519 hold_7_7 - 298.879202988792 hold_7_8
   - 265.78073089701 hold_7_9 - 239.2821535393818 hold_7_10
   - 217.5883952855847 hold_7_11 - 199.501246882793 hold_7_12
   + flow_nic_in_7_7 <= 0
 constr_intergroup_throughput_9_nic_out_8_nic_in_9: switch_nic_out_8_nic_in_9 = 1 ->
   - 2364.532019704433 hold_9_1 - 1191.06699751861 hold_9_2
   - 796.0199004975125 hold_9_3 - 597.7584059775841 hold_9_4
   - 478.5643070787636 hold_9_5 - 399.002493765586 hold_9_6
   - 342.1240199572345 hold_9_7 - 299.4385527136619 hold_9_8
   - 266.2229617304492 hold_9_9 - 239.6405391912132 hold_9_10
   - 217.8847026781661 hold_9_11 - 199.7503121098626 hold_9_12
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_8_nic_in_10: switch_nic_out_8_nic_in_10 = 1 ->
   - 2364.532019704433 hold_10_1 - 1191.06699751861 hold_10_2
   - 796.0199004975125 hold_10_3 - 597.7584059775841 hold_10_4
   - 478.5643070787636 hold_10_5 - 399.002493765586 hold_10_6
   - 342.1240199572345 hold_10_7 - 299.4385527136619 hold_10_8
   - 266.2229617304492 hold_10_9 - 239.6405391912132 hold_10_10
   - 217.8847026781661 hold_10_11 - 199.7503121098626 hold_10_12
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_8_nic_in_11: switch_nic_out_8_nic_in_11 = 1 ->
   - 2364.532019704433 hold_11_1 - 1191.06699751861 hold_11_2
   - 796.0199004975125 hold_11_3 - 597.7584059775841 hold_11_4
   - 478.5643070787636 hold_11_5 - 399.002493765586 hold_11_6
   - 342.1240199572345 hold_11_7 - 299.4385527136619 hold_11_8
   - 266.2229617304492 hold_11_9 - 239.6405391912132 hold_11_10
   - 217.8847026781661 hold_11_11 - 199.7503121098626 hold_11_12
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_9_source_nic_in_9: switch_source_nic_in_9 = 1 ->
   - 1846.153846153846 hold_9_1 - 1043.478260869565 hold_9_2
   - 727.2727272727274 hold_9_3 - 558.1395348837209 hold_9_4
   - 452.8301886792453 hold_9_5 - 380.9523809523809 hold_9_6
   - 328.7671232876712 hold_9_7 - 289.1566265060241 hold_9_8
   - 258.0645161290323 hold_9_9 - 233.0097087378641 hold_9_10
   - 212.3893805309735 hold_9_11 - 195.1219512195122 hold_9_12
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_0_nic_out_9_nic_in_0: switch_nic_out_9_nic_in_0 = 1 ->
   - 2330.097087378641 hold_0_1 - 1182.266009852217 hold_0_2
   - 792.0792079207921 hold_0_3 - 595.5334987593052 hold_0_4
   - 477.1371769383698 hold_0_5 - 398.0099502487562 hold_0_6
   - 341.3940256045519 hold_0_7 - 298.879202988792 hold_0_8
   - 265.78073089701 hold_0_9 - 239.2821535393818 hold_0_10
   - 217.5883952855847 hold_0_11 - 199.501246882793 hold_0_12
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_9_nic_in_1: switch_nic_out_9_nic_in_1 = 1 ->
   - 2330.097087378641 hold_1_1 - 1182.266009852217 hold_1_2
   - 792.0792079207921 hold_1_3 - 595.5334987593052 hold_1_4
   - 477.1371769383698 hold_1_5 - 398.0099502487562 hold_1_6
   - 341.3940256045519 hold_1_7 - 298.879202988792 hold_1_8
   - 265.78073089701 hold_1_9 - 239.2821535393818 hold_1_10
   - 217.5883952855847 hold_1_11 - 199.501246882793 hold_1_12
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_9_nic_in_2: switch_nic_out_9_nic_in_2 = 1 ->
   - 2330.097087378641 hold_2_1 - 1182.266009852217 hold_2_2
   - 792.0792079207921 hold_2_3 - 595.5334987593052 hold_2_4
   - 477.1371769383698 hold_2_5 - 398.0099502487562 hold_2_6
   - 341.3940256045519 hold_2_7 - 298.879202988792 hold_2_8
   - 265.78073089701 hold_2_9 - 239.2821535393818 hold_2_10
   - 217.5883952855847 hold_2_11 - 199.501246882793 hold_2_12
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_9_nic_in_3: switch_nic_out_9_nic_in_3 = 1 ->
   - 2330.097087378641 hold_3_1 - 1182.266009852217 hold_3_2
   - 792.0792079207921 hold_3_3 - 595.5334987593052 hold_3_4
   - 477.1371769383698 hold_3_5 - 398.0099502487562 hold_3_6
   - 341.3940256045519 hold_3_7 - 298.879202988792 hold_3_8
   - 265.78073089701 hold_3_9 - 239.2821535393818 hold_3_10
   - 217.5883952855847 hold_3_11 - 199.501246882793 hold_3_12
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_9_nic_in_4: switch_nic_out_9_nic_in_4 = 1 ->
   - 2330.097087378641 hold_4_1 - 1182.266009852217 hold_4_2
   - 792.0792079207921 hold_4_3 - 595.5334987593052 hold_4_4
   - 477.1371769383698 hold_4_5 - 398.0099502487562 hold_4_6
   - 341.3940256045519 hold_4_7 - 298.879202988792 hold_4_8
   - 265.78073089701 hold_4_9 - 239.2821535393818 hold_4_10
   - 217.5883952855847 hold_4_11 - 199.501246882793 hold_4_12
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
   - 2330.097087378641 hold_7_1 - 1182.266009852217 hold_7_2
   - 792.0792079207921 hold_7_3 - 595.5334987593052 hold_7_4
   - 477.1371769383698 hold_7_5 - 398.0099502487562 hold_7_6
   - 341.3940256045519 hold_7_7 - 298.879202988792 hold_7_8
   - 265.78073089701 hold_7_9 - 239.2821535393818 hold_7_10
   - 217.5883952855847 hold_7_11 - 199.501246882793 hold_7_12
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
   - 2364.532019704433 hold_10_1 - 1191.06699751861 hold_10_2
   - 796.0199004975125 hold_10_3 - 597.7584059775841 hold_10_4
   - 478.5643070787636 hold_10_5 - 399.002493765586 hold_10_6
   - 342.1240199572345 hold_10_7 - 299.4385527136619 hold_10_8
   - 266.2229617304492 hold_10_9 - 239.6405391912132 hold_10_10
   - 217.8847026781661 hold_10_11 - 199.7503121098626 hold_10_12
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_11_nic_out_9_nic_in_11: switch_nic_out_9_nic_in_11 = 1 ->
   - 2364.532019704433 hold_11_1 - 1191.06699751861 hold_11_2
   - 796.0199004975125 hold_11_3 - 597.7584059775841 hold_11_4
   - 478.5643070787636 hold_11_5 - 399.002493765586 hold_11_6
   - 342.1240199572345 hold_11_7 - 299.4385527136619 hold_11_8
   - 266.2229617304492 hold_11_9 - 239.6405391912132 hold_11_10
   - 217.8847026781661 hold_11_11 - 199.7503121098626 hold_11_12
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_10_source_nic_in_10: switch_source_nic_in_10 = 1 ->
   - 1846.153846153846 hold_10_1 - 1043.478260869565 hold_10_2
   - 727.2727272727274 hold_10_3 - 558.1395348837209 hold_10_4
   - 452.8301886792453 hold_10_5 - 380.9523809523809 hold_10_6
   - 328.7671232876712 hold_10_7 - 289.1566265060241 hold_10_8
   - 258.0645161290323 hold_10_9 - 233.0097087378641 hold_10_10
   - 212.3893805309735 hold_10_11 - 195.1219512195122 hold_10_12
   + flow_nic_in_10_10 <= 0
 constr_intergroup_throughput_0_nic_out_10_nic_in_0: switch_nic_out_10_nic_in_0 = 1 ->
   - 2330.097087378641 hold_0_1 - 1182.266009852217 hold_0_2
   - 792.0792079207921 hold_0_3 - 595.5334987593052 hold_0_4
   - 477.1371769383698 hold_0_5 - 398.0099502487562 hold_0_6
   - 341.3940256045519 hold_0_7 - 298.879202988792 hold_0_8
   - 265.78073089701 hold_0_9 - 239.2821535393818 hold_0_10
   - 217.5883952855847 hold_0_11 - 199.501246882793 hold_0_12
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_10_nic_in_1: switch_nic_out_10_nic_in_1 = 1 ->
   - 2330.097087378641 hold_1_1 - 1182.266009852217 hold_1_2
   - 792.0792079207921 hold_1_3 - 595.5334987593052 hold_1_4
   - 477.1371769383698 hold_1_5 - 398.0099502487562 hold_1_6
   - 341.3940256045519 hold_1_7 - 298.879202988792 hold_1_8
   - 265.78073089701 hold_1_9 - 239.2821535393818 hold_1_10
   - 217.5883952855847 hold_1_11 - 199.501246882793 hold_1_12
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_10_nic_in_2: switch_nic_out_10_nic_in_2 = 1 ->
   - 2330.097087378641 hold_2_1 - 1182.266009852217 hold_2_2
   - 792.0792079207921 hold_2_3 - 595.5334987593052 hold_2_4
   - 477.1371769383698 hold_2_5 - 398.0099502487562 hold_2_6
   - 341.3940256045519 hold_2_7 - 298.879202988792 hold_2_8
   - 265.78073089701 hold_2_9 - 239.2821535393818 hold_2_10
   - 217.5883952855847 hold_2_11 - 199.501246882793 hold_2_12
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_10_nic_in_3: switch_nic_out_10_nic_in_3 = 1 ->
   - 2330.097087378641 hold_3_1 - 1182.266009852217 hold_3_2
   - 792.0792079207921 hold_3_3 - 595.5334987593052 hold_3_4
   - 477.1371769383698 hold_3_5 - 398.0099502487562 hold_3_6
   - 341.3940256045519 hold_3_7 - 298.879202988792 hold_3_8
   - 265.78073089701 hold_3_9 - 239.2821535393818 hold_3_10
   - 217.5883952855847 hold_3_11 - 199.501246882793 hold_3_12
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_10_nic_in_4: switch_nic_out_10_nic_in_4 = 1 ->
   - 2330.097087378641 hold_4_1 - 1182.266009852217 hold_4_2
   - 792.0792079207921 hold_4_3 - 595.5334987593052 hold_4_4
   - 477.1371769383698 hold_4_5 - 398.0099502487562 hold_4_6
   - 341.3940256045519 hold_4_7 - 298.879202988792 hold_4_8
   - 265.78073089701 hold_4_9 - 239.2821535393818 hold_4_10
   - 217.5883952855847 hold_4_11 - 199.501246882793 hold_4_12
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
   - 2330.097087378641 hold_7_1 - 1182.266009852217 hold_7_2
   - 792.0792079207921 hold_7_3 - 595.5334987593052 hold_7_4
   - 477.1371769383698 hold_7_5 - 398.0099502487562 hold_7_6
   - 341.3940256045519 hold_7_7 - 298.879202988792 hold_7_8
   - 265.78073089701 hold_7_9 - 239.2821535393818 hold_7_10
   - 217.5883952855847 hold_7_11 - 199.501246882793 hold_7_12
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
   - 2364.532019704433 hold_9_1 - 1191.06699751861 hold_9_2
   - 796.0199004975125 hold_9_3 - 597.7584059775841 hold_9_4
   - 478.5643070787636 hold_9_5 - 399.002493765586 hold_9_6
   - 342.1240199572345 hold_9_7 - 299.4385527136619 hold_9_8
   - 266.2229617304492 hold_9_9 - 239.6405391912132 hold_9_10
   - 217.8847026781661 hold_9_11 - 199.7503121098626 hold_9_12
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_11_nic_out_10_nic_in_11: switch_nic_out_10_nic_in_11 = 1 ->
   - 2364.532019704433 hold_11_1 - 1191.06699751861 hold_11_2
   - 796.0199004975125 hold_11_3 - 597.7584059775841 hold_11_4
   - 478.5643070787636 hold_11_5 - 399.002493765586 hold_11_6
   - 342.1240199572345 hold_11_7 - 299.4385527136619 hold_11_8
   - 266.2229617304492 hold_11_9 - 239.6405391912132 hold_11_10
   - 217.8847026781661 hold_11_11 - 199.7503121098626 hold_11_12
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_11_source_nic_in_11: switch_source_nic_in_11 = 1 ->
   - 1846.153846153846 hold_11_1 - 1043.478260869565 hold_11_2
   - 727.2727272727274 hold_11_3 - 558.1395348837209 hold_11_4
   - 452.8301886792453 hold_11_5 - 380.9523809523809 hold_11_6
   - 328.7671232876712 hold_11_7 - 289.1566265060241 hold_11_8
   - 258.0645161290323 hold_11_9 - 233.0097087378641 hold_11_10
   - 212.3893805309735 hold_11_11 - 195.1219512195122 hold_11_12
   + flow_nic_in_11_11 <= 0
 constr_intergroup_throughput_0_nic_out_11_nic_in_0: switch_nic_out_11_nic_in_0 = 1 ->
   - 2330.097087378641 hold_0_1 - 1182.266009852217 hold_0_2
   - 792.0792079207921 hold_0_3 - 595.5334987593052 hold_0_4
   - 477.1371769383698 hold_0_5 - 398.0099502487562 hold_0_6
   - 341.3940256045519 hold_0_7 - 298.879202988792 hold_0_8
   - 265.78073089701 hold_0_9 - 239.2821535393818 hold_0_10
   - 217.5883952855847 hold_0_11 - 199.501246882793 hold_0_12
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_11_nic_in_1: switch_nic_out_11_nic_in_1 = 1 ->
   - 2330.097087378641 hold_1_1 - 1182.266009852217 hold_1_2
   - 792.0792079207921 hold_1_3 - 595.5334987593052 hold_1_4
   - 477.1371769383698 hold_1_5 - 398.0099502487562 hold_1_6
   - 341.3940256045519 hold_1_7 - 298.879202988792 hold_1_8
   - 265.78073089701 hold_1_9 - 239.2821535393818 hold_1_10
   - 217.5883952855847 hold_1_11 - 199.501246882793 hold_1_12
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_11_nic_in_2: switch_nic_out_11_nic_in_2 = 1 ->
   - 2330.097087378641 hold_2_1 - 1182.266009852217 hold_2_2
   - 792.0792079207921 hold_2_3 - 595.5334987593052 hold_2_4
   - 477.1371769383698 hold_2_5 - 398.0099502487562 hold_2_6
   - 341.3940256045519 hold_2_7 - 298.879202988792 hold_2_8
   - 265.78073089701 hold_2_9 - 239.2821535393818 hold_2_10
   - 217.5883952855847 hold_2_11 - 199.501246882793 hold_2_12
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_11_nic_in_3: switch_nic_out_11_nic_in_3 = 1 ->
   - 2330.097087378641 hold_3_1 - 1182.266009852217 hold_3_2
   - 792.0792079207921 hold_3_3 - 595.5334987593052 hold_3_4
   - 477.1371769383698 hold_3_5 - 398.0099502487562 hold_3_6
   - 341.3940256045519 hold_3_7 - 298.879202988792 hold_3_8
   - 265.78073089701 hold_3_9 - 239.2821535393818 hold_3_10
   - 217.5883952855847 hold_3_11 - 199.501246882793 hold_3_12
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_4_nic_out_11_nic_in_4: switch_nic_out_11_nic_in_4 = 1 ->
   - 2330.097087378641 hold_4_1 - 1182.266009852217 hold_4_2
   - 792.0792079207921 hold_4_3 - 595.5334987593052 hold_4_4
   - 477.1371769383698 hold_4_5 - 398.0099502487562 hold_4_6
   - 341.3940256045519 hold_4_7 - 298.879202988792 hold_4_8
   - 265.78073089701 hold_4_9 - 239.2821535393818 hold_4_10
   - 217.5883952855847 hold_4_11 - 199.501246882793 hold_4_12
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
   - 2364.532019704433 hold_8_1 - 1191.06699751861 hold_8_2
   - 796.0199004975125 hold_8_3 - 597.7584059775841 hold_8_4
   - 478.5643070787636 hold_8_5 - 399.002493765586 hold_8_6
   - 342.1240199572345 hold_8_7 - 299.4385527136619 hold_8_8
   - 266.2229617304492 hold_8_9 - 239.6405391912132 hold_8_10
   - 217.8847026781661 hold_8_11 - 199.7503121098626 hold_8_12
   + flow_nic_in_8_8 <= 0
 constr_intergroup_throughput_9_nic_out_11_nic_in_9: switch_nic_out_11_nic_in_9 = 1 ->
   - 2364.532019704433 hold_9_1 - 1191.06699751861 hold_9_2
   - 796.0199004975125 hold_9_3 - 597.7584059775841 hold_9_4
   - 478.5643070787636 hold_9_5 - 399.002493765586 hold_9_6
   - 342.1240199572345 hold_9_7 - 299.4385527136619 hold_9_8
   - 266.2229617304492 hold_9_9 - 239.6405391912132 hold_9_10
   - 217.8847026781661 hold_9_11 - 199.7503121098626 hold_9_12
   + flow_nic_in_9_9 <= 0
 constr_intergroup_throughput_10_nic_out_11_nic_in_10: switch_nic_out_11_nic_in_10 = 1 ->
   - 2364.532019704433 hold_10_1 - 1191.06699751861 hold_10_2
   - 796.0199004975125 hold_10_3 - 597.7584059775841 hold_10_4
   - 478.5643070787636 hold_10_5 - 399.002493765586 hold_10_6
   - 342.1240199572345 hold_10_7 - 299.4385527136619 hold_10_8
   - 266.2229617304492 hold_10_9 - 239.6405391912132 hold_10_10
   - 217.8847026781661 hold_10_11 - 199.7503121098626 hold_10_12
   + flow_nic_in_10_10 <= 0
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
 group_throughput_constr_0: group_tp_active_0 = 1 -> - 2400 hold_0_1
   - 1200 hold_0_2 - 800 hold_0_3 - 600 hold_0_4 - 480 hold_0_5
   - 400 hold_0_6 - 342.8571428571428 hold_0_7 - 300 hold_0_8
   - 266.6666666666667 hold_0_9 - 240 hold_0_10
   - 218.1818181818182 hold_0_11 - 200 hold_0_12 + flow_nic_in_0_0 <= 0
 group_throughput_constr_1: group_tp_active_1 = 1 -> - 2400 hold_1_1
   - 1200 hold_1_2 - 800 hold_1_3 - 600 hold_1_4 - 480 hold_1_5
   - 400 hold_1_6 - 342.8571428571428 hold_1_7 - 300 hold_1_8
   - 266.6666666666667 hold_1_9 - 240 hold_1_10
   - 218.1818181818182 hold_1_11 - 200 hold_1_12 + flow_nic_in_1_1 <= 0
 group_throughput_constr_2: group_tp_active_2 = 1 -> - 2400 hold_2_1
   - 1200 hold_2_2 - 800 hold_2_3 - 600 hold_2_4 - 480 hold_2_5
   - 400 hold_2_6 - 342.8571428571428 hold_2_7 - 300 hold_2_8
   - 266.6666666666667 hold_2_9 - 240 hold_2_10
   - 218.1818181818182 hold_2_11 - 200 hold_2_12 + flow_nic_in_2_2 <= 0
 group_throughput_constr_3: group_tp_active_3 = 1 -> - 2400 hold_3_1
   - 1200 hold_3_2 - 800 hold_3_3 - 600 hold_3_4 - 480 hold_3_5
   - 400 hold_3_6 - 342.8571428571428 hold_3_7 - 300 hold_3_8
   - 266.6666666666667 hold_3_9 - 240 hold_3_10
   - 218.1818181818182 hold_3_11 - 200 hold_3_12 + flow_nic_in_3_3 <= 0
 group_throughput_constr_4: group_tp_active_4 = 1 -> - 2400 hold_4_1
   - 1200 hold_4_2 - 800 hold_4_3 - 600 hold_4_4 - 480 hold_4_5
   - 400 hold_4_6 - 342.8571428571428 hold_4_7 - 300 hold_4_8
   - 266.6666666666667 hold_4_9 - 240 hold_4_10
   - 218.1818181818182 hold_4_11 - 200 hold_4_12 + flow_nic_in_4_4 <= 0
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
 group_throughput_constr_9: group_tp_active_9 = 1 -> - 2400 hold_9_1
   - 1200 hold_9_2 - 800 hold_9_3 - 600 hold_9_4 - 480 hold_9_5
   - 400 hold_9_6 - 342.8571428571428 hold_9_7 - 300 hold_9_8
   - 266.6666666666667 hold_9_9 - 240 hold_9_10
   - 218.1818181818182 hold_9_11 - 200 hold_9_12 + flow_nic_in_9_9 <= 0
 group_throughput_constr_10: group_tp_active_10 = 1 -> - 2400 hold_10_1
   - 1200 hold_10_2 - 800 hold_10_3 - 600 hold_10_4 - 480 hold_10_5
   - 400 hold_10_6 - 342.8571428571428 hold_10_7 - 300 hold_10_8
   - 266.6666666666667 hold_10_9 - 240 hold_10_10
   - 218.1818181818182 hold_10_11 - 200 hold_10_12 + flow_nic_in_10_10
   <= 0
 group_throughput_constr_11: group_tp_active_11 = 1 -> - 2400 hold_11_1
   - 1200 hold_11_2 - 800 hold_11_3 - 600 hold_11_4 - 480 hold_11_5
   - 400 hold_11_6 - 342.8571428571428 hold_11_7 - 300 hold_11_8
   - 266.6666666666667 hold_11_9 - 240 hold_11_10
   - 218.1818181818182 hold_11_11 - 200 hold_11_12 + flow_nic_in_11_11
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
 hold_11_10 hold_11_11 hold_11_12 switch_source_nic_in_0
 switch_nic_out_0_sink switch_nic_out_0_nic_in_1 switch_nic_out_0_nic_in_2
 switch_nic_out_0_nic_in_3 switch_nic_out_0_nic_in_4
 switch_nic_out_0_nic_in_5 switch_nic_out_0_nic_in_6
 switch_nic_out_0_nic_in_7 switch_nic_out_0_nic_in_8
 switch_nic_out_0_nic_in_9 switch_nic_out_0_nic_in_10
 switch_nic_out_0_nic_in_11 switch_nic_in_0_0 switch_0_nic_out_0
 switch_source_nic_in_1 switch_nic_out_1_sink switch_nic_out_1_nic_in_0
 switch_nic_out_1_nic_in_2 switch_nic_out_1_nic_in_3
 switch_nic_out_1_nic_in_4 switch_nic_out_1_nic_in_5
 switch_nic_out_1_nic_in_6 switch_nic_out_1_nic_in_7
 switch_nic_out_1_nic_in_8 switch_nic_out_1_nic_in_9
 switch_nic_out_1_nic_in_10 switch_nic_out_1_nic_in_11 switch_nic_in_1_1
 switch_1_nic_out_1 switch_source_nic_in_2 switch_nic_out_2_sink
 switch_nic_out_2_nic_in_0 switch_nic_out_2_nic_in_1
 switch_nic_out_2_nic_in_3 switch_nic_out_2_nic_in_4
 switch_nic_out_2_nic_in_5 switch_nic_out_2_nic_in_6
 switch_nic_out_2_nic_in_7 switch_nic_out_2_nic_in_8
 switch_nic_out_2_nic_in_9 switch_nic_out_2_nic_in_10
 switch_nic_out_2_nic_in_11 switch_nic_in_2_2 switch_2_nic_out_2
 switch_source_nic_in_3 switch_nic_out_3_sink switch_nic_out_3_nic_in_0
 switch_nic_out_3_nic_in_1 switch_nic_out_3_nic_in_2
 switch_nic_out_3_nic_in_4 switch_nic_out_3_nic_in_5
 switch_nic_out_3_nic_in_6 switch_nic_out_3_nic_in_7
 switch_nic_out_3_nic_in_8 switch_nic_out_3_nic_in_9
 switch_nic_out_3_nic_in_10 switch_nic_out_3_nic_in_11 switch_nic_in_3_3
 switch_3_nic_out_3 switch_source_nic_in_4 switch_nic_out_4_sink
 switch_nic_out_4_nic_in_0 switch_nic_out_4_nic_in_1
 switch_nic_out_4_nic_in_2 switch_nic_out_4_nic_in_3
 switch_nic_out_4_nic_in_5 switch_nic_out_4_nic_in_6
 switch_nic_out_4_nic_in_7 switch_nic_out_4_nic_in_8
 switch_nic_out_4_nic_in_9 switch_nic_out_4_nic_in_10
 switch_nic_out_4_nic_in_11 switch_nic_in_4_4 switch_4_nic_out_4
 switch_source_nic_in_5 switch_nic_out_5_sink switch_nic_out_5_nic_in_0
 switch_nic_out_5_nic_in_1 switch_nic_out_5_nic_in_2
 switch_nic_out_5_nic_in_3 switch_nic_out_5_nic_in_4
 switch_nic_out_5_nic_in_6 switch_nic_out_5_nic_in_7
 switch_nic_out_5_nic_in_8 switch_nic_out_5_nic_in_9
 switch_nic_out_5_nic_in_10 switch_nic_out_5_nic_in_11 switch_nic_in_5_5
 switch_5_nic_out_5 switch_source_nic_in_6 switch_nic_out_6_sink
 switch_nic_out_6_nic_in_0 switch_nic_out_6_nic_in_1
 switch_nic_out_6_nic_in_2 switch_nic_out_6_nic_in_3
 switch_nic_out_6_nic_in_4 switch_nic_out_6_nic_in_5
 switch_nic_out_6_nic_in_7 switch_nic_out_6_nic_in_8
 switch_nic_out_6_nic_in_9 switch_nic_out_6_nic_in_10
 switch_nic_out_6_nic_in_11 switch_nic_in_6_6 switch_6_nic_out_6
 switch_source_nic_in_7 switch_nic_out_7_sink switch_nic_out_7_nic_in_0
 switch_nic_out_7_nic_in_1 switch_nic_out_7_nic_in_2
 switch_nic_out_7_nic_in_3 switch_nic_out_7_nic_in_4
 switch_nic_out_7_nic_in_5 switch_nic_out_7_nic_in_6
 switch_nic_out_7_nic_in_8 switch_nic_out_7_nic_in_9
 switch_nic_out_7_nic_in_10 switch_nic_out_7_nic_in_11 switch_nic_in_7_7
 switch_7_nic_out_7 switch_source_nic_in_8 switch_nic_out_8_sink
 switch_nic_out_8_nic_in_0 switch_nic_out_8_nic_in_1
 switch_nic_out_8_nic_in_2 switch_nic_out_8_nic_in_3
 switch_nic_out_8_nic_in_4 switch_nic_out_8_nic_in_5
 switch_nic_out_8_nic_in_6 switch_nic_out_8_nic_in_7
 switch_nic_out_8_nic_in_9 switch_nic_out_8_nic_in_10
 switch_nic_out_8_nic_in_11 switch_nic_in_8_8 switch_8_nic_out_8
 switch_source_nic_in_9 switch_nic_out_9_sink switch_nic_out_9_nic_in_0
 switch_nic_out_9_nic_in_1 switch_nic_out_9_nic_in_2
 switch_nic_out_9_nic_in_3 switch_nic_out_9_nic_in_4
 switch_nic_out_9_nic_in_5 switch_nic_out_9_nic_in_6
 switch_nic_out_9_nic_in_7 switch_nic_out_9_nic_in_8
 switch_nic_out_9_nic_in_10 switch_nic_out_9_nic_in_11 switch_nic_in_9_9
 switch_9_nic_out_9 switch_source_nic_in_10 switch_nic_out_10_sink
 switch_nic_out_10_nic_in_0 switch_nic_out_10_nic_in_1
 switch_nic_out_10_nic_in_2 switch_nic_out_10_nic_in_3
 switch_nic_out_10_nic_in_4 switch_nic_out_10_nic_in_5
 switch_nic_out_10_nic_in_6 switch_nic_out_10_nic_in_7
 switch_nic_out_10_nic_in_8 switch_nic_out_10_nic_in_9
 switch_nic_out_10_nic_in_11 switch_nic_in_10_10 switch_10_nic_out_10
 switch_source_nic_in_11 switch_nic_out_11_sink switch_nic_out_11_nic_in_0
 switch_nic_out_11_nic_in_1 switch_nic_out_11_nic_in_2
 switch_nic_out_11_nic_in_3 switch_nic_out_11_nic_in_4
 switch_nic_out_11_nic_in_5 switch_nic_out_11_nic_in_6
 switch_nic_out_11_nic_in_7 switch_nic_out_11_nic_in_8
 switch_nic_out_11_nic_in_9 switch_nic_out_11_nic_in_10 switch_nic_in_11_11
 switch_11_nic_out_11 group_tp_active_0 group_tp_active_1 group_tp_active_2
 group_tp_active_3 group_tp_active_4 group_tp_active_5 group_tp_active_6
 group_tp_active_7 group_tp_active_8 group_tp_active_9 group_tp_active_10
 group_tp_active_11 is_active_0 is_active_1 is_active_2 is_active_3
 is_active_4 is_active_5 is_active_6 is_active_7 is_active_8 is_active_9
 is_active_10 is_active_11 is_start_zero_0_for_source_nic_in_0
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
Generals
 start_0 start_1 start_2 start_3 start_4 start_5 start_6 start_7 start_8
 start_9 start_10 start_11 group_start_1 val_end_0 group_end_0
 group_start_2 group_start_3 group_start_4 group_start_5 group_start_6
 group_start_7 group_start_8 group_start_9 group_start_10 group_start_11
 group_start_0 val_end_1 group_end_1 val_end_2 group_end_2 val_end_3
 group_end_3 val_end_4 group_end_4 val_end_5 group_end_5 val_end_6
 group_end_6 val_end_7 group_end_7 val_end_8 group_end_8 val_end_9
 group_end_9 val_end_10 group_end_10 val_end_11 group_end_11
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
End
