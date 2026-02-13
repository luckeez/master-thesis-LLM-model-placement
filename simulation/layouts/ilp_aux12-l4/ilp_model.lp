\ Model 2026-01-30_12-59-08
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  flow_source_nic_in_0 + flow_source_nic_in_1 + flow_source_nic_in_2
   + flow_source_nic_in_3 + flow_source_nic_in_4 + flow_source_nic_in_5
   + flow_source_nic_in_6 + flow_source_nic_in_7 + flow_source_nic_in_8
   + flow_source_nic_in_9 + flow_source_nic_in_10 + flow_source_nic_in_11
Subject To
 hold_constraint_0: hold_0_1 + hold_0_2 + hold_0_3 + hold_0_4 + hold_0_5
   + hold_0_6 + hold_0_7 + hold_0_8 + hold_0_9 + hold_0_10 + hold_0_11 = 1
 end_constraint_0: start_0 + hold_0_1 + 2 hold_0_2 + 3 hold_0_3
   + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7 + 8 hold_0_8
   + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 <= 60
 hold_constraint_1: hold_1_1 + hold_1_2 + hold_1_3 + hold_1_4 + hold_1_5
   + hold_1_6 + hold_1_7 + hold_1_8 + hold_1_9 + hold_1_10 + hold_1_11 = 1
 end_constraint_1: start_1 + hold_1_1 + 2 hold_1_2 + 3 hold_1_3
   + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7 + 8 hold_1_8
   + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 <= 60
 hold_constraint_2: hold_2_1 + hold_2_2 + hold_2_3 + hold_2_4 + hold_2_5
   + hold_2_6 + hold_2_7 + hold_2_8 + hold_2_9 + hold_2_10 + hold_2_11 = 1
 end_constraint_2: start_2 + hold_2_1 + 2 hold_2_2 + 3 hold_2_3
   + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7 + 8 hold_2_8
   + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 <= 60
 hold_constraint_3: hold_3_1 + hold_3_2 + hold_3_3 + hold_3_4 + hold_3_5
   + hold_3_6 + hold_3_7 + hold_3_8 + hold_3_9 + hold_3_10 + hold_3_11 = 1
 end_constraint_3: start_3 + hold_3_1 + 2 hold_3_2 + 3 hold_3_3
   + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7 + 8 hold_3_8
   + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11 <= 60
 hold_constraint_4: hold_4_1 + hold_4_2 + hold_4_3 + hold_4_4 + hold_4_5
   + hold_4_6 + hold_4_7 + hold_4_8 + hold_4_9 + hold_4_10 + hold_4_11 = 1
 end_constraint_4: start_4 + hold_4_1 + 2 hold_4_2 + 3 hold_4_3
   + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7 + 8 hold_4_8
   + 9 hold_4_9 + 10 hold_4_10 + 11 hold_4_11 <= 60
 hold_constraint_5: hold_5_1 + hold_5_2 + hold_5_3 + hold_5_4 + hold_5_5
   + hold_5_6 + hold_5_7 + hold_5_8 + hold_5_9 + hold_5_10 + hold_5_11 = 1
 end_constraint_5: start_5 + hold_5_1 + 2 hold_5_2 + 3 hold_5_3
   + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7 + 8 hold_5_8
   + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 <= 60
 hold_constraint_6: hold_6_1 + hold_6_2 + hold_6_3 + hold_6_4 + hold_6_5
   + hold_6_6 + hold_6_7 + hold_6_8 + hold_6_9 + hold_6_10 + hold_6_11 = 1
 end_constraint_6: start_6 + hold_6_1 + 2 hold_6_2 + 3 hold_6_3
   + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7 + 8 hold_6_8
   + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 <= 60
 hold_constraint_7: hold_7_1 + hold_7_2 + hold_7_3 + hold_7_4 + hold_7_5
   + hold_7_6 + hold_7_7 + hold_7_8 + hold_7_9 + hold_7_10 + hold_7_11 = 1
 end_constraint_7: start_7 + hold_7_1 + 2 hold_7_2 + 3 hold_7_3
   + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7 + 8 hold_7_8
   + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11 <= 60
 hold_constraint_8: hold_8_1 + hold_8_2 + hold_8_3 + hold_8_4 + hold_8_5
   + hold_8_6 + hold_8_7 + hold_8_8 + hold_8_9 + hold_8_10 + hold_8_11 = 1
 end_constraint_8: start_8 + hold_8_1 + 2 hold_8_2 + 3 hold_8_3
   + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7 + 8 hold_8_8
   + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 <= 60
 hold_constraint_9: hold_9_1 + hold_9_2 + hold_9_3 + hold_9_4 + hold_9_5
   + hold_9_6 + hold_9_7 + hold_9_8 + hold_9_9 + hold_9_10 + hold_9_11 = 1
 end_constraint_9: start_9 + hold_9_1 + 2 hold_9_2 + 3 hold_9_3
   + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7 + 8 hold_9_8
   + 9 hold_9_9 + 10 hold_9_10 + 11 hold_9_11 <= 60
 hold_constraint_10: hold_10_1 + hold_10_2 + hold_10_3 + hold_10_4
   + hold_10_5 + hold_10_6 + hold_10_7 + hold_10_8 + hold_10_9 + hold_10_10
   + hold_10_11 = 1
 end_constraint_10: start_10 + hold_10_1 + 2 hold_10_2 + 3 hold_10_3
   + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7 + 8 hold_10_8
   + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11 <= 60
 hold_constraint_11: hold_11_1 + hold_11_2 + hold_11_3 + hold_11_4
   + hold_11_5 + hold_11_6 + hold_11_7 + hold_11_8 + hold_11_9 + hold_11_10
   + hold_11_11 = 1
 end_constraint_11: start_11 + hold_11_1 + 2 hold_11_2 + 3 hold_11_3
   + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7 + 8 hold_11_8
   + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11 <= 60
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
   + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 - 60 is_end_complete_0 >= 0
 constr_edge_switch_end_max_0_OR: switch_nic_out_0_sink - is_end_complete_0
   <= 0
 constr_pair_ok1_0_1: start_0 - start_1 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 8 hold_0_8 + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11
   + 60 pair_ok_0_1_for_nic_out_0_nic_in_1 <= 60
 constr_pair_ok2_0_1: - start_0 + start_1 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11
   + 60 pair_ok_0_1_for_nic_out_0_nic_in_1 <= 60
 constr_edge_switch_end_start_0_1_OR: switch_nic_out_0_nic_in_1
   - pair_ok_0_1_for_nic_out_0_nic_in_1 <= 0
 constr_pair_ok1_0_2: start_0 - start_2 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 8 hold_0_8 + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11
   + 60 pair_ok_0_2_for_nic_out_0_nic_in_2 <= 60
 constr_pair_ok2_0_2: - start_0 + start_2 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11
   + 60 pair_ok_0_2_for_nic_out_0_nic_in_2 <= 60
 constr_edge_switch_end_start_0_2_OR: switch_nic_out_0_nic_in_2
   - pair_ok_0_2_for_nic_out_0_nic_in_2 <= 0
 constr_pair_ok1_0_3: start_0 - start_3 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 8 hold_0_8 + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11
   + 60 pair_ok_0_3_for_nic_out_0_nic_in_3 <= 60
 constr_pair_ok2_0_3: - start_0 + start_3 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11
   + 60 pair_ok_0_3_for_nic_out_0_nic_in_3 <= 60
 constr_edge_switch_end_start_0_3_OR: switch_nic_out_0_nic_in_3
   - pair_ok_0_3_for_nic_out_0_nic_in_3 <= 0
 constr_pair_ok1_0_4: start_0 - start_4 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 8 hold_0_8 + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11
   + 60 pair_ok_0_4_for_nic_out_0_nic_in_4 <= 60
 constr_pair_ok2_0_4: - start_0 + start_4 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11
   + 60 pair_ok_0_4_for_nic_out_0_nic_in_4 <= 60
 constr_edge_switch_end_start_0_4_OR: switch_nic_out_0_nic_in_4
   - pair_ok_0_4_for_nic_out_0_nic_in_4 <= 0
 constr_pair_ok1_0_5: start_0 - start_5 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 8 hold_0_8 + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11
   + 60 pair_ok_0_5_for_nic_out_0_nic_in_5 <= 60
 constr_pair_ok2_0_5: - start_0 + start_5 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11
   + 60 pair_ok_0_5_for_nic_out_0_nic_in_5 <= 60
 constr_edge_switch_end_start_0_5_OR: switch_nic_out_0_nic_in_5
   - pair_ok_0_5_for_nic_out_0_nic_in_5 <= 0
 constr_pair_ok1_0_6: start_0 - start_6 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 8 hold_0_8 + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11
   + 60 pair_ok_0_6_for_nic_out_0_nic_in_6 <= 60
 constr_pair_ok2_0_6: - start_0 + start_6 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11
   + 60 pair_ok_0_6_for_nic_out_0_nic_in_6 <= 60
 constr_edge_switch_end_start_0_6_OR: switch_nic_out_0_nic_in_6
   - pair_ok_0_6_for_nic_out_0_nic_in_6 <= 0
 constr_pair_ok1_0_7: start_0 - start_7 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 8 hold_0_8 + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11
   + 60 pair_ok_0_7_for_nic_out_0_nic_in_7 <= 60
 constr_pair_ok2_0_7: - start_0 + start_7 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11
   + 60 pair_ok_0_7_for_nic_out_0_nic_in_7 <= 60
 constr_edge_switch_end_start_0_7_OR: switch_nic_out_0_nic_in_7
   - pair_ok_0_7_for_nic_out_0_nic_in_7 <= 0
 constr_pair_ok1_0_8: start_0 - start_8 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 8 hold_0_8 + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11
   + 60 pair_ok_0_8_for_nic_out_0_nic_in_8 <= 60
 constr_pair_ok2_0_8: - start_0 + start_8 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11
   + 60 pair_ok_0_8_for_nic_out_0_nic_in_8 <= 60
 constr_edge_switch_end_start_0_8_OR: switch_nic_out_0_nic_in_8
   - pair_ok_0_8_for_nic_out_0_nic_in_8 <= 0
 constr_pair_ok1_0_9: start_0 - start_9 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 8 hold_0_8 + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11
   + 60 pair_ok_0_9_for_nic_out_0_nic_in_9 <= 60
 constr_pair_ok2_0_9: - start_0 + start_9 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11
   + 60 pair_ok_0_9_for_nic_out_0_nic_in_9 <= 60
 constr_edge_switch_end_start_0_9_OR: switch_nic_out_0_nic_in_9
   - pair_ok_0_9_for_nic_out_0_nic_in_9 <= 0
 constr_pair_ok1_0_10: start_0 - start_10 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 8 hold_0_8 + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11
   + 60 pair_ok_0_10_for_nic_out_0_nic_in_10 <= 60
 constr_pair_ok2_0_10: - start_0 + start_10 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11
   + 60 pair_ok_0_10_for_nic_out_0_nic_in_10 <= 60
 constr_edge_switch_end_start_0_10_OR: switch_nic_out_0_nic_in_10
   - pair_ok_0_10_for_nic_out_0_nic_in_10 <= 0
 constr_pair_ok1_0_11: start_0 - start_11 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 8 hold_0_8 + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11
   + 60 pair_ok_0_11_for_nic_out_0_nic_in_11 <= 60
 constr_pair_ok2_0_11: - start_0 + start_11 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11
   + 60 pair_ok_0_11_for_nic_out_0_nic_in_11 <= 60
 constr_edge_switch_end_start_0_11_OR: switch_nic_out_0_nic_in_11
   - pair_ok_0_11_for_nic_out_0_nic_in_11 <= 0
 force_constr_distrib_nic_in_0_0: switch_nic_in_0_0 = 1
 force_constr_distrib_0_nic_out_0: switch_0_nic_out_0 = 1
 constr_is_start_zero_1: start_1 + 60 is_start_zero_1_for_source_nic_in_1
   <= 60
 constr_edge_switch_start_zero_1_OR: switch_source_nic_in_1
   - is_start_zero_1_for_source_nic_in_1 <= 0
 constr_is_end_complete_1: start_1 + hold_1_1 + 2 hold_1_2 + 3 hold_1_3
   + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7 + 8 hold_1_8
   + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 - 60 is_end_complete_1 >= 0
 constr_edge_switch_end_max_1_OR: switch_nic_out_1_sink - is_end_complete_1
   <= 0
 constr_pair_ok1_1_0: - start_0 + start_1 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 8 hold_1_8 + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11
   + 60 pair_ok_1_0_for_nic_out_1_nic_in_0 <= 60
 constr_pair_ok2_1_0: start_0 - start_1 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11
   + 60 pair_ok_1_0_for_nic_out_1_nic_in_0 <= 60
 constr_edge_switch_end_start_1_0_OR: switch_nic_out_1_nic_in_0
   - pair_ok_1_0_for_nic_out_1_nic_in_0 <= 0
 constr_pair_ok1_1_2: start_1 - start_2 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 8 hold_1_8 + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11
   + 60 pair_ok_1_2_for_nic_out_1_nic_in_2 <= 60
 constr_pair_ok2_1_2: - start_1 + start_2 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11
   + 60 pair_ok_1_2_for_nic_out_1_nic_in_2 <= 60
 constr_edge_switch_end_start_1_2_OR: switch_nic_out_1_nic_in_2
   - pair_ok_1_2_for_nic_out_1_nic_in_2 <= 0
 constr_pair_ok1_1_3: start_1 - start_3 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 8 hold_1_8 + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11
   + 60 pair_ok_1_3_for_nic_out_1_nic_in_3 <= 60
 constr_pair_ok2_1_3: - start_1 + start_3 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11
   + 60 pair_ok_1_3_for_nic_out_1_nic_in_3 <= 60
 constr_edge_switch_end_start_1_3_OR: switch_nic_out_1_nic_in_3
   - pair_ok_1_3_for_nic_out_1_nic_in_3 <= 0
 constr_pair_ok1_1_4: start_1 - start_4 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 8 hold_1_8 + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11
   + 60 pair_ok_1_4_for_nic_out_1_nic_in_4 <= 60
 constr_pair_ok2_1_4: - start_1 + start_4 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11
   + 60 pair_ok_1_4_for_nic_out_1_nic_in_4 <= 60
 constr_edge_switch_end_start_1_4_OR: switch_nic_out_1_nic_in_4
   - pair_ok_1_4_for_nic_out_1_nic_in_4 <= 0
 constr_pair_ok1_1_5: start_1 - start_5 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 8 hold_1_8 + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11
   + 60 pair_ok_1_5_for_nic_out_1_nic_in_5 <= 60
 constr_pair_ok2_1_5: - start_1 + start_5 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11
   + 60 pair_ok_1_5_for_nic_out_1_nic_in_5 <= 60
 constr_edge_switch_end_start_1_5_OR: switch_nic_out_1_nic_in_5
   - pair_ok_1_5_for_nic_out_1_nic_in_5 <= 0
 constr_pair_ok1_1_6: start_1 - start_6 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 8 hold_1_8 + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11
   + 60 pair_ok_1_6_for_nic_out_1_nic_in_6 <= 60
 constr_pair_ok2_1_6: - start_1 + start_6 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11
   + 60 pair_ok_1_6_for_nic_out_1_nic_in_6 <= 60
 constr_edge_switch_end_start_1_6_OR: switch_nic_out_1_nic_in_6
   - pair_ok_1_6_for_nic_out_1_nic_in_6 <= 0
 constr_pair_ok1_1_7: start_1 - start_7 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 8 hold_1_8 + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11
   + 60 pair_ok_1_7_for_nic_out_1_nic_in_7 <= 60
 constr_pair_ok2_1_7: - start_1 + start_7 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11
   + 60 pair_ok_1_7_for_nic_out_1_nic_in_7 <= 60
 constr_edge_switch_end_start_1_7_OR: switch_nic_out_1_nic_in_7
   - pair_ok_1_7_for_nic_out_1_nic_in_7 <= 0
 constr_pair_ok1_1_8: start_1 - start_8 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 8 hold_1_8 + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11
   + 60 pair_ok_1_8_for_nic_out_1_nic_in_8 <= 60
 constr_pair_ok2_1_8: - start_1 + start_8 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11
   + 60 pair_ok_1_8_for_nic_out_1_nic_in_8 <= 60
 constr_edge_switch_end_start_1_8_OR: switch_nic_out_1_nic_in_8
   - pair_ok_1_8_for_nic_out_1_nic_in_8 <= 0
 constr_pair_ok1_1_9: start_1 - start_9 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 8 hold_1_8 + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11
   + 60 pair_ok_1_9_for_nic_out_1_nic_in_9 <= 60
 constr_pair_ok2_1_9: - start_1 + start_9 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11
   + 60 pair_ok_1_9_for_nic_out_1_nic_in_9 <= 60
 constr_edge_switch_end_start_1_9_OR: switch_nic_out_1_nic_in_9
   - pair_ok_1_9_for_nic_out_1_nic_in_9 <= 0
 constr_pair_ok1_1_10: start_1 - start_10 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 8 hold_1_8 + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11
   + 60 pair_ok_1_10_for_nic_out_1_nic_in_10 <= 60
 constr_pair_ok2_1_10: - start_1 + start_10 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11
   + 60 pair_ok_1_10_for_nic_out_1_nic_in_10 <= 60
 constr_edge_switch_end_start_1_10_OR: switch_nic_out_1_nic_in_10
   - pair_ok_1_10_for_nic_out_1_nic_in_10 <= 0
 constr_pair_ok1_1_11: start_1 - start_11 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 8 hold_1_8 + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11
   + 60 pair_ok_1_11_for_nic_out_1_nic_in_11 <= 60
 constr_pair_ok2_1_11: - start_1 + start_11 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11
   + 60 pair_ok_1_11_for_nic_out_1_nic_in_11 <= 60
 constr_edge_switch_end_start_1_11_OR: switch_nic_out_1_nic_in_11
   - pair_ok_1_11_for_nic_out_1_nic_in_11 <= 0
 force_constr_distrib_nic_in_1_1: switch_nic_in_1_1 = 1
 force_constr_distrib_1_nic_out_1: switch_1_nic_out_1 = 1
 constr_is_start_zero_2: start_2 + 60 is_start_zero_2_for_source_nic_in_2
   <= 60
 constr_edge_switch_start_zero_2_OR: switch_source_nic_in_2
   - is_start_zero_2_for_source_nic_in_2 <= 0
 constr_is_end_complete_2: start_2 + hold_2_1 + 2 hold_2_2 + 3 hold_2_3
   + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7 + 8 hold_2_8
   + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 - 60 is_end_complete_2 >= 0
 constr_edge_switch_end_max_2_OR: switch_nic_out_2_sink - is_end_complete_2
   <= 0
 constr_pair_ok1_2_0: - start_0 + start_2 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11
   + 60 pair_ok_2_0_for_nic_out_2_nic_in_0 <= 60
 constr_pair_ok2_2_0: start_0 - start_2 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11
   + 60 pair_ok_2_0_for_nic_out_2_nic_in_0 <= 60
 constr_edge_switch_end_start_2_0_OR: switch_nic_out_2_nic_in_0
   - pair_ok_2_0_for_nic_out_2_nic_in_0 <= 0
 constr_pair_ok1_2_1: - start_1 + start_2 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11
   + 60 pair_ok_2_1_for_nic_out_2_nic_in_1 <= 60
 constr_pair_ok2_2_1: start_1 - start_2 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11
   + 60 pair_ok_2_1_for_nic_out_2_nic_in_1 <= 60
 constr_edge_switch_end_start_2_1_OR: switch_nic_out_2_nic_in_1
   - pair_ok_2_1_for_nic_out_2_nic_in_1 <= 0
 constr_pair_ok1_2_3: start_2 - start_3 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11
   + 60 pair_ok_2_3_for_nic_out_2_nic_in_3 <= 60
 constr_pair_ok2_2_3: - start_2 + start_3 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11
   + 60 pair_ok_2_3_for_nic_out_2_nic_in_3 <= 60
 constr_edge_switch_end_start_2_3_OR: switch_nic_out_2_nic_in_3
   - pair_ok_2_3_for_nic_out_2_nic_in_3 <= 0
 constr_pair_ok1_2_4: start_2 - start_4 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11
   + 60 pair_ok_2_4_for_nic_out_2_nic_in_4 <= 60
 constr_pair_ok2_2_4: - start_2 + start_4 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11
   + 60 pair_ok_2_4_for_nic_out_2_nic_in_4 <= 60
 constr_edge_switch_end_start_2_4_OR: switch_nic_out_2_nic_in_4
   - pair_ok_2_4_for_nic_out_2_nic_in_4 <= 0
 constr_pair_ok1_2_5: start_2 - start_5 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11
   + 60 pair_ok_2_5_for_nic_out_2_nic_in_5 <= 60
 constr_pair_ok2_2_5: - start_2 + start_5 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11
   + 60 pair_ok_2_5_for_nic_out_2_nic_in_5 <= 60
 constr_edge_switch_end_start_2_5_OR: switch_nic_out_2_nic_in_5
   - pair_ok_2_5_for_nic_out_2_nic_in_5 <= 0
 constr_pair_ok1_2_6: start_2 - start_6 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11
   + 60 pair_ok_2_6_for_nic_out_2_nic_in_6 <= 60
 constr_pair_ok2_2_6: - start_2 + start_6 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11
   + 60 pair_ok_2_6_for_nic_out_2_nic_in_6 <= 60
 constr_edge_switch_end_start_2_6_OR: switch_nic_out_2_nic_in_6
   - pair_ok_2_6_for_nic_out_2_nic_in_6 <= 0
 constr_pair_ok1_2_7: start_2 - start_7 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11
   + 60 pair_ok_2_7_for_nic_out_2_nic_in_7 <= 60
 constr_pair_ok2_2_7: - start_2 + start_7 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11
   + 60 pair_ok_2_7_for_nic_out_2_nic_in_7 <= 60
 constr_edge_switch_end_start_2_7_OR: switch_nic_out_2_nic_in_7
   - pair_ok_2_7_for_nic_out_2_nic_in_7 <= 0
 constr_pair_ok1_2_8: start_2 - start_8 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11
   + 60 pair_ok_2_8_for_nic_out_2_nic_in_8 <= 60
 constr_pair_ok2_2_8: - start_2 + start_8 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11
   + 60 pair_ok_2_8_for_nic_out_2_nic_in_8 <= 60
 constr_edge_switch_end_start_2_8_OR: switch_nic_out_2_nic_in_8
   - pair_ok_2_8_for_nic_out_2_nic_in_8 <= 0
 constr_pair_ok1_2_9: start_2 - start_9 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11
   + 60 pair_ok_2_9_for_nic_out_2_nic_in_9 <= 60
 constr_pair_ok2_2_9: - start_2 + start_9 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11
   + 60 pair_ok_2_9_for_nic_out_2_nic_in_9 <= 60
 constr_edge_switch_end_start_2_9_OR: switch_nic_out_2_nic_in_9
   - pair_ok_2_9_for_nic_out_2_nic_in_9 <= 0
 constr_pair_ok1_2_10: start_2 - start_10 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11
   + 60 pair_ok_2_10_for_nic_out_2_nic_in_10 <= 60
 constr_pair_ok2_2_10: - start_2 + start_10 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11
   + 60 pair_ok_2_10_for_nic_out_2_nic_in_10 <= 60
 constr_edge_switch_end_start_2_10_OR: switch_nic_out_2_nic_in_10
   - pair_ok_2_10_for_nic_out_2_nic_in_10 <= 0
 constr_pair_ok1_2_11: start_2 - start_11 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11
   + 60 pair_ok_2_11_for_nic_out_2_nic_in_11 <= 60
 constr_pair_ok2_2_11: - start_2 + start_11 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11
   + 60 pair_ok_2_11_for_nic_out_2_nic_in_11 <= 60
 constr_edge_switch_end_start_2_11_OR: switch_nic_out_2_nic_in_11
   - pair_ok_2_11_for_nic_out_2_nic_in_11 <= 0
 force_constr_distrib_nic_in_2_2: switch_nic_in_2_2 = 1
 force_constr_distrib_2_nic_out_2: switch_2_nic_out_2 = 1
 constr_is_start_zero_3: start_3 + 60 is_start_zero_3_for_source_nic_in_3
   <= 60
 constr_edge_switch_start_zero_3_OR: switch_source_nic_in_3
   - is_start_zero_3_for_source_nic_in_3 <= 0
 constr_is_end_complete_3: start_3 + hold_3_1 + 2 hold_3_2 + 3 hold_3_3
   + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7 + 8 hold_3_8
   + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11 - 60 is_end_complete_3 >= 0
 constr_edge_switch_end_max_3_OR: switch_nic_out_3_sink - is_end_complete_3
   <= 0
 constr_pair_ok1_3_0: - start_0 + start_3 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 8 hold_3_8 + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11
   + 60 pair_ok_3_0_for_nic_out_3_nic_in_0 <= 60
 constr_pair_ok2_3_0: start_0 - start_3 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   - 8 hold_3_8 - 9 hold_3_9 - 10 hold_3_10 - 11 hold_3_11
   + 60 pair_ok_3_0_for_nic_out_3_nic_in_0 <= 60
 constr_edge_switch_end_start_3_0_OR: switch_nic_out_3_nic_in_0
   - pair_ok_3_0_for_nic_out_3_nic_in_0 <= 0
 constr_pair_ok1_3_1: - start_1 + start_3 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 8 hold_3_8 + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11
   + 60 pair_ok_3_1_for_nic_out_3_nic_in_1 <= 60
 constr_pair_ok2_3_1: start_1 - start_3 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   - 8 hold_3_8 - 9 hold_3_9 - 10 hold_3_10 - 11 hold_3_11
   + 60 pair_ok_3_1_for_nic_out_3_nic_in_1 <= 60
 constr_edge_switch_end_start_3_1_OR: switch_nic_out_3_nic_in_1
   - pair_ok_3_1_for_nic_out_3_nic_in_1 <= 0
 constr_pair_ok1_3_2: - start_2 + start_3 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 8 hold_3_8 + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11
   + 60 pair_ok_3_2_for_nic_out_3_nic_in_2 <= 60
 constr_pair_ok2_3_2: start_2 - start_3 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   - 8 hold_3_8 - 9 hold_3_9 - 10 hold_3_10 - 11 hold_3_11
   + 60 pair_ok_3_2_for_nic_out_3_nic_in_2 <= 60
 constr_edge_switch_end_start_3_2_OR: switch_nic_out_3_nic_in_2
   - pair_ok_3_2_for_nic_out_3_nic_in_2 <= 0
 constr_pair_ok1_3_4: start_3 - start_4 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 8 hold_3_8 + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11
   + 60 pair_ok_3_4_for_nic_out_3_nic_in_4 <= 60
 constr_pair_ok2_3_4: - start_3 + start_4 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   - 8 hold_3_8 - 9 hold_3_9 - 10 hold_3_10 - 11 hold_3_11
   + 60 pair_ok_3_4_for_nic_out_3_nic_in_4 <= 60
 constr_edge_switch_end_start_3_4_OR: switch_nic_out_3_nic_in_4
   - pair_ok_3_4_for_nic_out_3_nic_in_4 <= 0
 constr_pair_ok1_3_5: start_3 - start_5 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 8 hold_3_8 + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11
   + 60 pair_ok_3_5_for_nic_out_3_nic_in_5 <= 60
 constr_pair_ok2_3_5: - start_3 + start_5 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   - 8 hold_3_8 - 9 hold_3_9 - 10 hold_3_10 - 11 hold_3_11
   + 60 pair_ok_3_5_for_nic_out_3_nic_in_5 <= 60
 constr_edge_switch_end_start_3_5_OR: switch_nic_out_3_nic_in_5
   - pair_ok_3_5_for_nic_out_3_nic_in_5 <= 0
 constr_pair_ok1_3_6: start_3 - start_6 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 8 hold_3_8 + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11
   + 60 pair_ok_3_6_for_nic_out_3_nic_in_6 <= 60
 constr_pair_ok2_3_6: - start_3 + start_6 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   - 8 hold_3_8 - 9 hold_3_9 - 10 hold_3_10 - 11 hold_3_11
   + 60 pair_ok_3_6_for_nic_out_3_nic_in_6 <= 60
 constr_edge_switch_end_start_3_6_OR: switch_nic_out_3_nic_in_6
   - pair_ok_3_6_for_nic_out_3_nic_in_6 <= 0
 constr_pair_ok1_3_7: start_3 - start_7 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 8 hold_3_8 + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11
   + 60 pair_ok_3_7_for_nic_out_3_nic_in_7 <= 60
 constr_pair_ok2_3_7: - start_3 + start_7 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   - 8 hold_3_8 - 9 hold_3_9 - 10 hold_3_10 - 11 hold_3_11
   + 60 pair_ok_3_7_for_nic_out_3_nic_in_7 <= 60
 constr_edge_switch_end_start_3_7_OR: switch_nic_out_3_nic_in_7
   - pair_ok_3_7_for_nic_out_3_nic_in_7 <= 0
 constr_pair_ok1_3_8: start_3 - start_8 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 8 hold_3_8 + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11
   + 60 pair_ok_3_8_for_nic_out_3_nic_in_8 <= 60
 constr_pair_ok2_3_8: - start_3 + start_8 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   - 8 hold_3_8 - 9 hold_3_9 - 10 hold_3_10 - 11 hold_3_11
   + 60 pair_ok_3_8_for_nic_out_3_nic_in_8 <= 60
 constr_edge_switch_end_start_3_8_OR: switch_nic_out_3_nic_in_8
   - pair_ok_3_8_for_nic_out_3_nic_in_8 <= 0
 constr_pair_ok1_3_9: start_3 - start_9 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 8 hold_3_8 + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11
   + 60 pair_ok_3_9_for_nic_out_3_nic_in_9 <= 60
 constr_pair_ok2_3_9: - start_3 + start_9 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   - 8 hold_3_8 - 9 hold_3_9 - 10 hold_3_10 - 11 hold_3_11
   + 60 pair_ok_3_9_for_nic_out_3_nic_in_9 <= 60
 constr_edge_switch_end_start_3_9_OR: switch_nic_out_3_nic_in_9
   - pair_ok_3_9_for_nic_out_3_nic_in_9 <= 0
 constr_pair_ok1_3_10: start_3 - start_10 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 8 hold_3_8 + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11
   + 60 pair_ok_3_10_for_nic_out_3_nic_in_10 <= 60
 constr_pair_ok2_3_10: - start_3 + start_10 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   - 8 hold_3_8 - 9 hold_3_9 - 10 hold_3_10 - 11 hold_3_11
   + 60 pair_ok_3_10_for_nic_out_3_nic_in_10 <= 60
 constr_edge_switch_end_start_3_10_OR: switch_nic_out_3_nic_in_10
   - pair_ok_3_10_for_nic_out_3_nic_in_10 <= 0
 constr_pair_ok1_3_11: start_3 - start_11 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 8 hold_3_8 + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11
   + 60 pair_ok_3_11_for_nic_out_3_nic_in_11 <= 60
 constr_pair_ok2_3_11: - start_3 + start_11 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   - 8 hold_3_8 - 9 hold_3_9 - 10 hold_3_10 - 11 hold_3_11
   + 60 pair_ok_3_11_for_nic_out_3_nic_in_11 <= 60
 constr_edge_switch_end_start_3_11_OR: switch_nic_out_3_nic_in_11
   - pair_ok_3_11_for_nic_out_3_nic_in_11 <= 0
 force_constr_distrib_nic_in_3_3: switch_nic_in_3_3 = 1
 force_constr_distrib_3_nic_out_3: switch_3_nic_out_3 = 1
 constr_is_start_zero_4: start_4 + 60 is_start_zero_4_for_source_nic_in_4
   <= 60
 constr_edge_switch_start_zero_4_OR: switch_source_nic_in_4
   - is_start_zero_4_for_source_nic_in_4 <= 0
 constr_is_end_complete_4: start_4 + hold_4_1 + 2 hold_4_2 + 3 hold_4_3
   + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7 + 8 hold_4_8
   + 9 hold_4_9 + 10 hold_4_10 + 11 hold_4_11 - 60 is_end_complete_4 >= 0
 constr_edge_switch_end_max_4_OR: switch_nic_out_4_sink - is_end_complete_4
   <= 0
 constr_pair_ok1_4_0: - start_0 + start_4 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 8 hold_4_8 + 9 hold_4_9 + 10 hold_4_10 + 11 hold_4_11
   + 60 pair_ok_4_0_for_nic_out_4_nic_in_0 <= 60
 constr_pair_ok2_4_0: start_0 - start_4 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   - 8 hold_4_8 - 9 hold_4_9 - 10 hold_4_10 - 11 hold_4_11
   + 60 pair_ok_4_0_for_nic_out_4_nic_in_0 <= 60
 constr_edge_switch_end_start_4_0_OR: switch_nic_out_4_nic_in_0
   - pair_ok_4_0_for_nic_out_4_nic_in_0 <= 0
 constr_pair_ok1_4_1: - start_1 + start_4 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 8 hold_4_8 + 9 hold_4_9 + 10 hold_4_10 + 11 hold_4_11
   + 60 pair_ok_4_1_for_nic_out_4_nic_in_1 <= 60
 constr_pair_ok2_4_1: start_1 - start_4 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   - 8 hold_4_8 - 9 hold_4_9 - 10 hold_4_10 - 11 hold_4_11
   + 60 pair_ok_4_1_for_nic_out_4_nic_in_1 <= 60
 constr_edge_switch_end_start_4_1_OR: switch_nic_out_4_nic_in_1
   - pair_ok_4_1_for_nic_out_4_nic_in_1 <= 0
 constr_pair_ok1_4_2: - start_2 + start_4 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 8 hold_4_8 + 9 hold_4_9 + 10 hold_4_10 + 11 hold_4_11
   + 60 pair_ok_4_2_for_nic_out_4_nic_in_2 <= 60
 constr_pair_ok2_4_2: start_2 - start_4 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   - 8 hold_4_8 - 9 hold_4_9 - 10 hold_4_10 - 11 hold_4_11
   + 60 pair_ok_4_2_for_nic_out_4_nic_in_2 <= 60
 constr_edge_switch_end_start_4_2_OR: switch_nic_out_4_nic_in_2
   - pair_ok_4_2_for_nic_out_4_nic_in_2 <= 0
 constr_pair_ok1_4_3: - start_3 + start_4 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 8 hold_4_8 + 9 hold_4_9 + 10 hold_4_10 + 11 hold_4_11
   + 60 pair_ok_4_3_for_nic_out_4_nic_in_3 <= 60
 constr_pair_ok2_4_3: start_3 - start_4 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   - 8 hold_4_8 - 9 hold_4_9 - 10 hold_4_10 - 11 hold_4_11
   + 60 pair_ok_4_3_for_nic_out_4_nic_in_3 <= 60
 constr_edge_switch_end_start_4_3_OR: switch_nic_out_4_nic_in_3
   - pair_ok_4_3_for_nic_out_4_nic_in_3 <= 0
 constr_pair_ok1_4_5: start_4 - start_5 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 8 hold_4_8 + 9 hold_4_9 + 10 hold_4_10 + 11 hold_4_11
   + 60 pair_ok_4_5_for_nic_out_4_nic_in_5 <= 60
 constr_pair_ok2_4_5: - start_4 + start_5 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   - 8 hold_4_8 - 9 hold_4_9 - 10 hold_4_10 - 11 hold_4_11
   + 60 pair_ok_4_5_for_nic_out_4_nic_in_5 <= 60
 constr_edge_switch_end_start_4_5_OR: switch_nic_out_4_nic_in_5
   - pair_ok_4_5_for_nic_out_4_nic_in_5 <= 0
 constr_pair_ok1_4_6: start_4 - start_6 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 8 hold_4_8 + 9 hold_4_9 + 10 hold_4_10 + 11 hold_4_11
   + 60 pair_ok_4_6_for_nic_out_4_nic_in_6 <= 60
 constr_pair_ok2_4_6: - start_4 + start_6 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   - 8 hold_4_8 - 9 hold_4_9 - 10 hold_4_10 - 11 hold_4_11
   + 60 pair_ok_4_6_for_nic_out_4_nic_in_6 <= 60
 constr_edge_switch_end_start_4_6_OR: switch_nic_out_4_nic_in_6
   - pair_ok_4_6_for_nic_out_4_nic_in_6 <= 0
 constr_pair_ok1_4_7: start_4 - start_7 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 8 hold_4_8 + 9 hold_4_9 + 10 hold_4_10 + 11 hold_4_11
   + 60 pair_ok_4_7_for_nic_out_4_nic_in_7 <= 60
 constr_pair_ok2_4_7: - start_4 + start_7 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   - 8 hold_4_8 - 9 hold_4_9 - 10 hold_4_10 - 11 hold_4_11
   + 60 pair_ok_4_7_for_nic_out_4_nic_in_7 <= 60
 constr_edge_switch_end_start_4_7_OR: switch_nic_out_4_nic_in_7
   - pair_ok_4_7_for_nic_out_4_nic_in_7 <= 0
 constr_pair_ok1_4_8: start_4 - start_8 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 8 hold_4_8 + 9 hold_4_9 + 10 hold_4_10 + 11 hold_4_11
   + 60 pair_ok_4_8_for_nic_out_4_nic_in_8 <= 60
 constr_pair_ok2_4_8: - start_4 + start_8 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   - 8 hold_4_8 - 9 hold_4_9 - 10 hold_4_10 - 11 hold_4_11
   + 60 pair_ok_4_8_for_nic_out_4_nic_in_8 <= 60
 constr_edge_switch_end_start_4_8_OR: switch_nic_out_4_nic_in_8
   - pair_ok_4_8_for_nic_out_4_nic_in_8 <= 0
 constr_pair_ok1_4_9: start_4 - start_9 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 8 hold_4_8 + 9 hold_4_9 + 10 hold_4_10 + 11 hold_4_11
   + 60 pair_ok_4_9_for_nic_out_4_nic_in_9 <= 60
 constr_pair_ok2_4_9: - start_4 + start_9 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   - 8 hold_4_8 - 9 hold_4_9 - 10 hold_4_10 - 11 hold_4_11
   + 60 pair_ok_4_9_for_nic_out_4_nic_in_9 <= 60
 constr_edge_switch_end_start_4_9_OR: switch_nic_out_4_nic_in_9
   - pair_ok_4_9_for_nic_out_4_nic_in_9 <= 0
 constr_pair_ok1_4_10: start_4 - start_10 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 8 hold_4_8 + 9 hold_4_9 + 10 hold_4_10 + 11 hold_4_11
   + 60 pair_ok_4_10_for_nic_out_4_nic_in_10 <= 60
 constr_pair_ok2_4_10: - start_4 + start_10 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   - 8 hold_4_8 - 9 hold_4_9 - 10 hold_4_10 - 11 hold_4_11
   + 60 pair_ok_4_10_for_nic_out_4_nic_in_10 <= 60
 constr_edge_switch_end_start_4_10_OR: switch_nic_out_4_nic_in_10
   - pair_ok_4_10_for_nic_out_4_nic_in_10 <= 0
 constr_pair_ok1_4_11: start_4 - start_11 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 8 hold_4_8 + 9 hold_4_9 + 10 hold_4_10 + 11 hold_4_11
   + 60 pair_ok_4_11_for_nic_out_4_nic_in_11 <= 60
 constr_pair_ok2_4_11: - start_4 + start_11 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   - 8 hold_4_8 - 9 hold_4_9 - 10 hold_4_10 - 11 hold_4_11
   + 60 pair_ok_4_11_for_nic_out_4_nic_in_11 <= 60
 constr_edge_switch_end_start_4_11_OR: switch_nic_out_4_nic_in_11
   - pair_ok_4_11_for_nic_out_4_nic_in_11 <= 0
 force_constr_distrib_nic_in_4_4: switch_nic_in_4_4 = 1
 force_constr_distrib_4_nic_out_4: switch_4_nic_out_4 = 1
 constr_is_start_zero_5: start_5 + 60 is_start_zero_5_for_source_nic_in_5
   <= 60
 constr_edge_switch_start_zero_5_OR: switch_source_nic_in_5
   - is_start_zero_5_for_source_nic_in_5 <= 0
 constr_is_end_complete_5: start_5 + hold_5_1 + 2 hold_5_2 + 3 hold_5_3
   + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7 + 8 hold_5_8
   + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 - 60 is_end_complete_5 >= 0
 constr_edge_switch_end_max_5_OR: switch_nic_out_5_sink - is_end_complete_5
   <= 0
 constr_pair_ok1_5_0: - start_0 + start_5 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11
   + 60 pair_ok_5_0_for_nic_out_5_nic_in_0 <= 60
 constr_pair_ok2_5_0: start_0 - start_5 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11
   + 60 pair_ok_5_0_for_nic_out_5_nic_in_0 <= 60
 constr_edge_switch_end_start_5_0_OR: switch_nic_out_5_nic_in_0
   - pair_ok_5_0_for_nic_out_5_nic_in_0 <= 0
 constr_pair_ok1_5_1: - start_1 + start_5 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11
   + 60 pair_ok_5_1_for_nic_out_5_nic_in_1 <= 60
 constr_pair_ok2_5_1: start_1 - start_5 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11
   + 60 pair_ok_5_1_for_nic_out_5_nic_in_1 <= 60
 constr_edge_switch_end_start_5_1_OR: switch_nic_out_5_nic_in_1
   - pair_ok_5_1_for_nic_out_5_nic_in_1 <= 0
 constr_pair_ok1_5_2: - start_2 + start_5 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11
   + 60 pair_ok_5_2_for_nic_out_5_nic_in_2 <= 60
 constr_pair_ok2_5_2: start_2 - start_5 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11
   + 60 pair_ok_5_2_for_nic_out_5_nic_in_2 <= 60
 constr_edge_switch_end_start_5_2_OR: switch_nic_out_5_nic_in_2
   - pair_ok_5_2_for_nic_out_5_nic_in_2 <= 0
 constr_pair_ok1_5_3: - start_3 + start_5 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11
   + 60 pair_ok_5_3_for_nic_out_5_nic_in_3 <= 60
 constr_pair_ok2_5_3: start_3 - start_5 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11
   + 60 pair_ok_5_3_for_nic_out_5_nic_in_3 <= 60
 constr_edge_switch_end_start_5_3_OR: switch_nic_out_5_nic_in_3
   - pair_ok_5_3_for_nic_out_5_nic_in_3 <= 0
 constr_pair_ok1_5_4: - start_4 + start_5 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11
   + 60 pair_ok_5_4_for_nic_out_5_nic_in_4 <= 60
 constr_pair_ok2_5_4: start_4 - start_5 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11
   + 60 pair_ok_5_4_for_nic_out_5_nic_in_4 <= 60
 constr_edge_switch_end_start_5_4_OR: switch_nic_out_5_nic_in_4
   - pair_ok_5_4_for_nic_out_5_nic_in_4 <= 0
 constr_pair_ok1_5_6: start_5 - start_6 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11
   + 60 pair_ok_5_6_for_nic_out_5_nic_in_6 <= 60
 constr_pair_ok2_5_6: - start_5 + start_6 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11
   + 60 pair_ok_5_6_for_nic_out_5_nic_in_6 <= 60
 constr_edge_switch_end_start_5_6_OR: switch_nic_out_5_nic_in_6
   - pair_ok_5_6_for_nic_out_5_nic_in_6 <= 0
 constr_pair_ok1_5_7: start_5 - start_7 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11
   + 60 pair_ok_5_7_for_nic_out_5_nic_in_7 <= 60
 constr_pair_ok2_5_7: - start_5 + start_7 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11
   + 60 pair_ok_5_7_for_nic_out_5_nic_in_7 <= 60
 constr_edge_switch_end_start_5_7_OR: switch_nic_out_5_nic_in_7
   - pair_ok_5_7_for_nic_out_5_nic_in_7 <= 0
 constr_pair_ok1_5_8: start_5 - start_8 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11
   + 60 pair_ok_5_8_for_nic_out_5_nic_in_8 <= 60
 constr_pair_ok2_5_8: - start_5 + start_8 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11
   + 60 pair_ok_5_8_for_nic_out_5_nic_in_8 <= 60
 constr_edge_switch_end_start_5_8_OR: switch_nic_out_5_nic_in_8
   - pair_ok_5_8_for_nic_out_5_nic_in_8 <= 0
 constr_pair_ok1_5_9: start_5 - start_9 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11
   + 60 pair_ok_5_9_for_nic_out_5_nic_in_9 <= 60
 constr_pair_ok2_5_9: - start_5 + start_9 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11
   + 60 pair_ok_5_9_for_nic_out_5_nic_in_9 <= 60
 constr_edge_switch_end_start_5_9_OR: switch_nic_out_5_nic_in_9
   - pair_ok_5_9_for_nic_out_5_nic_in_9 <= 0
 constr_pair_ok1_5_10: start_5 - start_10 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11
   + 60 pair_ok_5_10_for_nic_out_5_nic_in_10 <= 60
 constr_pair_ok2_5_10: - start_5 + start_10 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11
   + 60 pair_ok_5_10_for_nic_out_5_nic_in_10 <= 60
 constr_edge_switch_end_start_5_10_OR: switch_nic_out_5_nic_in_10
   - pair_ok_5_10_for_nic_out_5_nic_in_10 <= 0
 constr_pair_ok1_5_11: start_5 - start_11 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11
   + 60 pair_ok_5_11_for_nic_out_5_nic_in_11 <= 60
 constr_pair_ok2_5_11: - start_5 + start_11 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11
   + 60 pair_ok_5_11_for_nic_out_5_nic_in_11 <= 60
 constr_edge_switch_end_start_5_11_OR: switch_nic_out_5_nic_in_11
   - pair_ok_5_11_for_nic_out_5_nic_in_11 <= 0
 force_constr_distrib_nic_in_5_5: switch_nic_in_5_5 = 1
 force_constr_distrib_5_nic_out_5: switch_5_nic_out_5 = 1
 constr_is_start_zero_6: start_6 + 60 is_start_zero_6_for_source_nic_in_6
   <= 60
 constr_edge_switch_start_zero_6_OR: switch_source_nic_in_6
   - is_start_zero_6_for_source_nic_in_6 <= 0
 constr_is_end_complete_6: start_6 + hold_6_1 + 2 hold_6_2 + 3 hold_6_3
   + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7 + 8 hold_6_8
   + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 - 60 is_end_complete_6 >= 0
 constr_edge_switch_end_max_6_OR: switch_nic_out_6_sink - is_end_complete_6
   <= 0
 constr_pair_ok1_6_0: - start_0 + start_6 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11
   + 60 pair_ok_6_0_for_nic_out_6_nic_in_0 <= 60
 constr_pair_ok2_6_0: start_0 - start_6 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11
   + 60 pair_ok_6_0_for_nic_out_6_nic_in_0 <= 60
 constr_edge_switch_end_start_6_0_OR: switch_nic_out_6_nic_in_0
   - pair_ok_6_0_for_nic_out_6_nic_in_0 <= 0
 constr_pair_ok1_6_1: - start_1 + start_6 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11
   + 60 pair_ok_6_1_for_nic_out_6_nic_in_1 <= 60
 constr_pair_ok2_6_1: start_1 - start_6 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11
   + 60 pair_ok_6_1_for_nic_out_6_nic_in_1 <= 60
 constr_edge_switch_end_start_6_1_OR: switch_nic_out_6_nic_in_1
   - pair_ok_6_1_for_nic_out_6_nic_in_1 <= 0
 constr_pair_ok1_6_2: - start_2 + start_6 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11
   + 60 pair_ok_6_2_for_nic_out_6_nic_in_2 <= 60
 constr_pair_ok2_6_2: start_2 - start_6 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11
   + 60 pair_ok_6_2_for_nic_out_6_nic_in_2 <= 60
 constr_edge_switch_end_start_6_2_OR: switch_nic_out_6_nic_in_2
   - pair_ok_6_2_for_nic_out_6_nic_in_2 <= 0
 constr_pair_ok1_6_3: - start_3 + start_6 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11
   + 60 pair_ok_6_3_for_nic_out_6_nic_in_3 <= 60
 constr_pair_ok2_6_3: start_3 - start_6 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11
   + 60 pair_ok_6_3_for_nic_out_6_nic_in_3 <= 60
 constr_edge_switch_end_start_6_3_OR: switch_nic_out_6_nic_in_3
   - pair_ok_6_3_for_nic_out_6_nic_in_3 <= 0
 constr_pair_ok1_6_4: - start_4 + start_6 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11
   + 60 pair_ok_6_4_for_nic_out_6_nic_in_4 <= 60
 constr_pair_ok2_6_4: start_4 - start_6 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11
   + 60 pair_ok_6_4_for_nic_out_6_nic_in_4 <= 60
 constr_edge_switch_end_start_6_4_OR: switch_nic_out_6_nic_in_4
   - pair_ok_6_4_for_nic_out_6_nic_in_4 <= 0
 constr_pair_ok1_6_5: - start_5 + start_6 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11
   + 60 pair_ok_6_5_for_nic_out_6_nic_in_5 <= 60
 constr_pair_ok2_6_5: start_5 - start_6 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11
   + 60 pair_ok_6_5_for_nic_out_6_nic_in_5 <= 60
 constr_edge_switch_end_start_6_5_OR: switch_nic_out_6_nic_in_5
   - pair_ok_6_5_for_nic_out_6_nic_in_5 <= 0
 constr_pair_ok1_6_7: start_6 - start_7 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11
   + 60 pair_ok_6_7_for_nic_out_6_nic_in_7 <= 60
 constr_pair_ok2_6_7: - start_6 + start_7 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11
   + 60 pair_ok_6_7_for_nic_out_6_nic_in_7 <= 60
 constr_edge_switch_end_start_6_7_OR: switch_nic_out_6_nic_in_7
   - pair_ok_6_7_for_nic_out_6_nic_in_7 <= 0
 constr_pair_ok1_6_8: start_6 - start_8 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11
   + 60 pair_ok_6_8_for_nic_out_6_nic_in_8 <= 60
 constr_pair_ok2_6_8: - start_6 + start_8 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11
   + 60 pair_ok_6_8_for_nic_out_6_nic_in_8 <= 60
 constr_edge_switch_end_start_6_8_OR: switch_nic_out_6_nic_in_8
   - pair_ok_6_8_for_nic_out_6_nic_in_8 <= 0
 constr_pair_ok1_6_9: start_6 - start_9 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11
   + 60 pair_ok_6_9_for_nic_out_6_nic_in_9 <= 60
 constr_pair_ok2_6_9: - start_6 + start_9 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11
   + 60 pair_ok_6_9_for_nic_out_6_nic_in_9 <= 60
 constr_edge_switch_end_start_6_9_OR: switch_nic_out_6_nic_in_9
   - pair_ok_6_9_for_nic_out_6_nic_in_9 <= 0
 constr_pair_ok1_6_10: start_6 - start_10 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11
   + 60 pair_ok_6_10_for_nic_out_6_nic_in_10 <= 60
 constr_pair_ok2_6_10: - start_6 + start_10 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11
   + 60 pair_ok_6_10_for_nic_out_6_nic_in_10 <= 60
 constr_edge_switch_end_start_6_10_OR: switch_nic_out_6_nic_in_10
   - pair_ok_6_10_for_nic_out_6_nic_in_10 <= 0
 constr_pair_ok1_6_11: start_6 - start_11 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11
   + 60 pair_ok_6_11_for_nic_out_6_nic_in_11 <= 60
 constr_pair_ok2_6_11: - start_6 + start_11 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11
   + 60 pair_ok_6_11_for_nic_out_6_nic_in_11 <= 60
 constr_edge_switch_end_start_6_11_OR: switch_nic_out_6_nic_in_11
   - pair_ok_6_11_for_nic_out_6_nic_in_11 <= 0
 force_constr_distrib_nic_in_6_6: switch_nic_in_6_6 = 1
 force_constr_distrib_6_nic_out_6: switch_6_nic_out_6 = 1
 constr_is_start_zero_7: start_7 + 60 is_start_zero_7_for_source_nic_in_7
   <= 60
 constr_edge_switch_start_zero_7_OR: switch_source_nic_in_7
   - is_start_zero_7_for_source_nic_in_7 <= 0
 constr_is_end_complete_7: start_7 + hold_7_1 + 2 hold_7_2 + 3 hold_7_3
   + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7 + 8 hold_7_8
   + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11 - 60 is_end_complete_7 >= 0
 constr_edge_switch_end_max_7_OR: switch_nic_out_7_sink - is_end_complete_7
   <= 0
 constr_pair_ok1_7_0: - start_0 + start_7 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 8 hold_7_8 + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11
   + 60 pair_ok_7_0_for_nic_out_7_nic_in_0 <= 60
 constr_pair_ok2_7_0: start_0 - start_7 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   - 8 hold_7_8 - 9 hold_7_9 - 10 hold_7_10 - 11 hold_7_11
   + 60 pair_ok_7_0_for_nic_out_7_nic_in_0 <= 60
 constr_edge_switch_end_start_7_0_OR: switch_nic_out_7_nic_in_0
   - pair_ok_7_0_for_nic_out_7_nic_in_0 <= 0
 constr_pair_ok1_7_1: - start_1 + start_7 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 8 hold_7_8 + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11
   + 60 pair_ok_7_1_for_nic_out_7_nic_in_1 <= 60
 constr_pair_ok2_7_1: start_1 - start_7 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   - 8 hold_7_8 - 9 hold_7_9 - 10 hold_7_10 - 11 hold_7_11
   + 60 pair_ok_7_1_for_nic_out_7_nic_in_1 <= 60
 constr_edge_switch_end_start_7_1_OR: switch_nic_out_7_nic_in_1
   - pair_ok_7_1_for_nic_out_7_nic_in_1 <= 0
 constr_pair_ok1_7_2: - start_2 + start_7 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 8 hold_7_8 + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11
   + 60 pair_ok_7_2_for_nic_out_7_nic_in_2 <= 60
 constr_pair_ok2_7_2: start_2 - start_7 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   - 8 hold_7_8 - 9 hold_7_9 - 10 hold_7_10 - 11 hold_7_11
   + 60 pair_ok_7_2_for_nic_out_7_nic_in_2 <= 60
 constr_edge_switch_end_start_7_2_OR: switch_nic_out_7_nic_in_2
   - pair_ok_7_2_for_nic_out_7_nic_in_2 <= 0
 constr_pair_ok1_7_3: - start_3 + start_7 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 8 hold_7_8 + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11
   + 60 pair_ok_7_3_for_nic_out_7_nic_in_3 <= 60
 constr_pair_ok2_7_3: start_3 - start_7 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   - 8 hold_7_8 - 9 hold_7_9 - 10 hold_7_10 - 11 hold_7_11
   + 60 pair_ok_7_3_for_nic_out_7_nic_in_3 <= 60
 constr_edge_switch_end_start_7_3_OR: switch_nic_out_7_nic_in_3
   - pair_ok_7_3_for_nic_out_7_nic_in_3 <= 0
 constr_pair_ok1_7_4: - start_4 + start_7 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 8 hold_7_8 + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11
   + 60 pair_ok_7_4_for_nic_out_7_nic_in_4 <= 60
 constr_pair_ok2_7_4: start_4 - start_7 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   - 8 hold_7_8 - 9 hold_7_9 - 10 hold_7_10 - 11 hold_7_11
   + 60 pair_ok_7_4_for_nic_out_7_nic_in_4 <= 60
 constr_edge_switch_end_start_7_4_OR: switch_nic_out_7_nic_in_4
   - pair_ok_7_4_for_nic_out_7_nic_in_4 <= 0
 constr_pair_ok1_7_5: - start_5 + start_7 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 8 hold_7_8 + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11
   + 60 pair_ok_7_5_for_nic_out_7_nic_in_5 <= 60
 constr_pair_ok2_7_5: start_5 - start_7 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   - 8 hold_7_8 - 9 hold_7_9 - 10 hold_7_10 - 11 hold_7_11
   + 60 pair_ok_7_5_for_nic_out_7_nic_in_5 <= 60
 constr_edge_switch_end_start_7_5_OR: switch_nic_out_7_nic_in_5
   - pair_ok_7_5_for_nic_out_7_nic_in_5 <= 0
 constr_pair_ok1_7_6: - start_6 + start_7 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 8 hold_7_8 + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11
   + 60 pair_ok_7_6_for_nic_out_7_nic_in_6 <= 60
 constr_pair_ok2_7_6: start_6 - start_7 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   - 8 hold_7_8 - 9 hold_7_9 - 10 hold_7_10 - 11 hold_7_11
   + 60 pair_ok_7_6_for_nic_out_7_nic_in_6 <= 60
 constr_edge_switch_end_start_7_6_OR: switch_nic_out_7_nic_in_6
   - pair_ok_7_6_for_nic_out_7_nic_in_6 <= 0
 constr_pair_ok1_7_8: start_7 - start_8 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 8 hold_7_8 + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11
   + 60 pair_ok_7_8_for_nic_out_7_nic_in_8 <= 60
 constr_pair_ok2_7_8: - start_7 + start_8 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   - 8 hold_7_8 - 9 hold_7_9 - 10 hold_7_10 - 11 hold_7_11
   + 60 pair_ok_7_8_for_nic_out_7_nic_in_8 <= 60
 constr_edge_switch_end_start_7_8_OR: switch_nic_out_7_nic_in_8
   - pair_ok_7_8_for_nic_out_7_nic_in_8 <= 0
 constr_pair_ok1_7_9: start_7 - start_9 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 8 hold_7_8 + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11
   + 60 pair_ok_7_9_for_nic_out_7_nic_in_9 <= 60
 constr_pair_ok2_7_9: - start_7 + start_9 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   - 8 hold_7_8 - 9 hold_7_9 - 10 hold_7_10 - 11 hold_7_11
   + 60 pair_ok_7_9_for_nic_out_7_nic_in_9 <= 60
 constr_edge_switch_end_start_7_9_OR: switch_nic_out_7_nic_in_9
   - pair_ok_7_9_for_nic_out_7_nic_in_9 <= 0
 constr_pair_ok1_7_10: start_7 - start_10 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 8 hold_7_8 + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11
   + 60 pair_ok_7_10_for_nic_out_7_nic_in_10 <= 60
 constr_pair_ok2_7_10: - start_7 + start_10 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   - 8 hold_7_8 - 9 hold_7_9 - 10 hold_7_10 - 11 hold_7_11
   + 60 pair_ok_7_10_for_nic_out_7_nic_in_10 <= 60
 constr_edge_switch_end_start_7_10_OR: switch_nic_out_7_nic_in_10
   - pair_ok_7_10_for_nic_out_7_nic_in_10 <= 0
 constr_pair_ok1_7_11: start_7 - start_11 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 8 hold_7_8 + 9 hold_7_9 + 10 hold_7_10 + 11 hold_7_11
   + 60 pair_ok_7_11_for_nic_out_7_nic_in_11 <= 60
 constr_pair_ok2_7_11: - start_7 + start_11 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   - 8 hold_7_8 - 9 hold_7_9 - 10 hold_7_10 - 11 hold_7_11
   + 60 pair_ok_7_11_for_nic_out_7_nic_in_11 <= 60
 constr_edge_switch_end_start_7_11_OR: switch_nic_out_7_nic_in_11
   - pair_ok_7_11_for_nic_out_7_nic_in_11 <= 0
 force_constr_distrib_nic_in_7_7: switch_nic_in_7_7 = 1
 force_constr_distrib_7_nic_out_7: switch_7_nic_out_7 = 1
 constr_is_start_zero_8: start_8 + 60 is_start_zero_8_for_source_nic_in_8
   <= 60
 constr_edge_switch_start_zero_8_OR: switch_source_nic_in_8
   - is_start_zero_8_for_source_nic_in_8 <= 0
 constr_is_end_complete_8: start_8 + hold_8_1 + 2 hold_8_2 + 3 hold_8_3
   + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7 + 8 hold_8_8
   + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 - 60 is_end_complete_8 >= 0
 constr_edge_switch_end_max_8_OR: switch_nic_out_8_sink - is_end_complete_8
   <= 0
 constr_pair_ok1_8_0: - start_0 + start_8 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11
   + 60 pair_ok_8_0_for_nic_out_8_nic_in_0 <= 60
 constr_pair_ok2_8_0: start_0 - start_8 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11
   + 60 pair_ok_8_0_for_nic_out_8_nic_in_0 <= 60
 constr_edge_switch_end_start_8_0_OR: switch_nic_out_8_nic_in_0
   - pair_ok_8_0_for_nic_out_8_nic_in_0 <= 0
 constr_pair_ok1_8_1: - start_1 + start_8 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11
   + 60 pair_ok_8_1_for_nic_out_8_nic_in_1 <= 60
 constr_pair_ok2_8_1: start_1 - start_8 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11
   + 60 pair_ok_8_1_for_nic_out_8_nic_in_1 <= 60
 constr_edge_switch_end_start_8_1_OR: switch_nic_out_8_nic_in_1
   - pair_ok_8_1_for_nic_out_8_nic_in_1 <= 0
 constr_pair_ok1_8_2: - start_2 + start_8 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11
   + 60 pair_ok_8_2_for_nic_out_8_nic_in_2 <= 60
 constr_pair_ok2_8_2: start_2 - start_8 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11
   + 60 pair_ok_8_2_for_nic_out_8_nic_in_2 <= 60
 constr_edge_switch_end_start_8_2_OR: switch_nic_out_8_nic_in_2
   - pair_ok_8_2_for_nic_out_8_nic_in_2 <= 0
 constr_pair_ok1_8_3: - start_3 + start_8 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11
   + 60 pair_ok_8_3_for_nic_out_8_nic_in_3 <= 60
 constr_pair_ok2_8_3: start_3 - start_8 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11
   + 60 pair_ok_8_3_for_nic_out_8_nic_in_3 <= 60
 constr_edge_switch_end_start_8_3_OR: switch_nic_out_8_nic_in_3
   - pair_ok_8_3_for_nic_out_8_nic_in_3 <= 0
 constr_pair_ok1_8_4: - start_4 + start_8 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11
   + 60 pair_ok_8_4_for_nic_out_8_nic_in_4 <= 60
 constr_pair_ok2_8_4: start_4 - start_8 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11
   + 60 pair_ok_8_4_for_nic_out_8_nic_in_4 <= 60
 constr_edge_switch_end_start_8_4_OR: switch_nic_out_8_nic_in_4
   - pair_ok_8_4_for_nic_out_8_nic_in_4 <= 0
 constr_pair_ok1_8_5: - start_5 + start_8 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11
   + 60 pair_ok_8_5_for_nic_out_8_nic_in_5 <= 60
 constr_pair_ok2_8_5: start_5 - start_8 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11
   + 60 pair_ok_8_5_for_nic_out_8_nic_in_5 <= 60
 constr_edge_switch_end_start_8_5_OR: switch_nic_out_8_nic_in_5
   - pair_ok_8_5_for_nic_out_8_nic_in_5 <= 0
 constr_pair_ok1_8_6: - start_6 + start_8 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11
   + 60 pair_ok_8_6_for_nic_out_8_nic_in_6 <= 60
 constr_pair_ok2_8_6: start_6 - start_8 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11
   + 60 pair_ok_8_6_for_nic_out_8_nic_in_6 <= 60
 constr_edge_switch_end_start_8_6_OR: switch_nic_out_8_nic_in_6
   - pair_ok_8_6_for_nic_out_8_nic_in_6 <= 0
 constr_pair_ok1_8_7: - start_7 + start_8 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11
   + 60 pair_ok_8_7_for_nic_out_8_nic_in_7 <= 60
 constr_pair_ok2_8_7: start_7 - start_8 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11
   + 60 pair_ok_8_7_for_nic_out_8_nic_in_7 <= 60
 constr_edge_switch_end_start_8_7_OR: switch_nic_out_8_nic_in_7
   - pair_ok_8_7_for_nic_out_8_nic_in_7 <= 0
 constr_pair_ok1_8_9: start_8 - start_9 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11
   + 60 pair_ok_8_9_for_nic_out_8_nic_in_9 <= 60
 constr_pair_ok2_8_9: - start_8 + start_9 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11
   + 60 pair_ok_8_9_for_nic_out_8_nic_in_9 <= 60
 constr_edge_switch_end_start_8_9_OR: switch_nic_out_8_nic_in_9
   - pair_ok_8_9_for_nic_out_8_nic_in_9 <= 0
 constr_pair_ok1_8_10: start_8 - start_10 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11
   + 60 pair_ok_8_10_for_nic_out_8_nic_in_10 <= 60
 constr_pair_ok2_8_10: - start_8 + start_10 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11
   + 60 pair_ok_8_10_for_nic_out_8_nic_in_10 <= 60
 constr_edge_switch_end_start_8_10_OR: switch_nic_out_8_nic_in_10
   - pair_ok_8_10_for_nic_out_8_nic_in_10 <= 0
 constr_pair_ok1_8_11: start_8 - start_11 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11
   + 60 pair_ok_8_11_for_nic_out_8_nic_in_11 <= 60
 constr_pair_ok2_8_11: - start_8 + start_11 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11
   + 60 pair_ok_8_11_for_nic_out_8_nic_in_11 <= 60
 constr_edge_switch_end_start_8_11_OR: switch_nic_out_8_nic_in_11
   - pair_ok_8_11_for_nic_out_8_nic_in_11 <= 0
 force_constr_distrib_nic_in_8_8: switch_nic_in_8_8 = 1
 force_constr_distrib_8_nic_out_8: switch_8_nic_out_8 = 1
 constr_is_start_zero_9: start_9 + 60 is_start_zero_9_for_source_nic_in_9
   <= 60
 constr_edge_switch_start_zero_9_OR: switch_source_nic_in_9
   - is_start_zero_9_for_source_nic_in_9 <= 0
 constr_is_end_complete_9: start_9 + hold_9_1 + 2 hold_9_2 + 3 hold_9_3
   + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7 + 8 hold_9_8
   + 9 hold_9_9 + 10 hold_9_10 + 11 hold_9_11 - 60 is_end_complete_9 >= 0
 constr_edge_switch_end_max_9_OR: switch_nic_out_9_sink - is_end_complete_9
   <= 0
 constr_pair_ok1_9_0: - start_0 + start_9 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 8 hold_9_8 + 9 hold_9_9 + 10 hold_9_10 + 11 hold_9_11
   + 60 pair_ok_9_0_for_nic_out_9_nic_in_0 <= 60
 constr_pair_ok2_9_0: start_0 - start_9 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   - 8 hold_9_8 - 9 hold_9_9 - 10 hold_9_10 - 11 hold_9_11
   + 60 pair_ok_9_0_for_nic_out_9_nic_in_0 <= 60
 constr_edge_switch_end_start_9_0_OR: switch_nic_out_9_nic_in_0
   - pair_ok_9_0_for_nic_out_9_nic_in_0 <= 0
 constr_pair_ok1_9_1: - start_1 + start_9 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 8 hold_9_8 + 9 hold_9_9 + 10 hold_9_10 + 11 hold_9_11
   + 60 pair_ok_9_1_for_nic_out_9_nic_in_1 <= 60
 constr_pair_ok2_9_1: start_1 - start_9 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   - 8 hold_9_8 - 9 hold_9_9 - 10 hold_9_10 - 11 hold_9_11
   + 60 pair_ok_9_1_for_nic_out_9_nic_in_1 <= 60
 constr_edge_switch_end_start_9_1_OR: switch_nic_out_9_nic_in_1
   - pair_ok_9_1_for_nic_out_9_nic_in_1 <= 0
 constr_pair_ok1_9_2: - start_2 + start_9 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 8 hold_9_8 + 9 hold_9_9 + 10 hold_9_10 + 11 hold_9_11
   + 60 pair_ok_9_2_for_nic_out_9_nic_in_2 <= 60
 constr_pair_ok2_9_2: start_2 - start_9 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   - 8 hold_9_8 - 9 hold_9_9 - 10 hold_9_10 - 11 hold_9_11
   + 60 pair_ok_9_2_for_nic_out_9_nic_in_2 <= 60
 constr_edge_switch_end_start_9_2_OR: switch_nic_out_9_nic_in_2
   - pair_ok_9_2_for_nic_out_9_nic_in_2 <= 0
 constr_pair_ok1_9_3: - start_3 + start_9 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 8 hold_9_8 + 9 hold_9_9 + 10 hold_9_10 + 11 hold_9_11
   + 60 pair_ok_9_3_for_nic_out_9_nic_in_3 <= 60
 constr_pair_ok2_9_3: start_3 - start_9 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   - 8 hold_9_8 - 9 hold_9_9 - 10 hold_9_10 - 11 hold_9_11
   + 60 pair_ok_9_3_for_nic_out_9_nic_in_3 <= 60
 constr_edge_switch_end_start_9_3_OR: switch_nic_out_9_nic_in_3
   - pair_ok_9_3_for_nic_out_9_nic_in_3 <= 0
 constr_pair_ok1_9_4: - start_4 + start_9 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 8 hold_9_8 + 9 hold_9_9 + 10 hold_9_10 + 11 hold_9_11
   + 60 pair_ok_9_4_for_nic_out_9_nic_in_4 <= 60
 constr_pair_ok2_9_4: start_4 - start_9 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   - 8 hold_9_8 - 9 hold_9_9 - 10 hold_9_10 - 11 hold_9_11
   + 60 pair_ok_9_4_for_nic_out_9_nic_in_4 <= 60
 constr_edge_switch_end_start_9_4_OR: switch_nic_out_9_nic_in_4
   - pair_ok_9_4_for_nic_out_9_nic_in_4 <= 0
 constr_pair_ok1_9_5: - start_5 + start_9 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 8 hold_9_8 + 9 hold_9_9 + 10 hold_9_10 + 11 hold_9_11
   + 60 pair_ok_9_5_for_nic_out_9_nic_in_5 <= 60
 constr_pair_ok2_9_5: start_5 - start_9 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   - 8 hold_9_8 - 9 hold_9_9 - 10 hold_9_10 - 11 hold_9_11
   + 60 pair_ok_9_5_for_nic_out_9_nic_in_5 <= 60
 constr_edge_switch_end_start_9_5_OR: switch_nic_out_9_nic_in_5
   - pair_ok_9_5_for_nic_out_9_nic_in_5 <= 0
 constr_pair_ok1_9_6: - start_6 + start_9 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 8 hold_9_8 + 9 hold_9_9 + 10 hold_9_10 + 11 hold_9_11
   + 60 pair_ok_9_6_for_nic_out_9_nic_in_6 <= 60
 constr_pair_ok2_9_6: start_6 - start_9 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   - 8 hold_9_8 - 9 hold_9_9 - 10 hold_9_10 - 11 hold_9_11
   + 60 pair_ok_9_6_for_nic_out_9_nic_in_6 <= 60
 constr_edge_switch_end_start_9_6_OR: switch_nic_out_9_nic_in_6
   - pair_ok_9_6_for_nic_out_9_nic_in_6 <= 0
 constr_pair_ok1_9_7: - start_7 + start_9 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 8 hold_9_8 + 9 hold_9_9 + 10 hold_9_10 + 11 hold_9_11
   + 60 pair_ok_9_7_for_nic_out_9_nic_in_7 <= 60
 constr_pair_ok2_9_7: start_7 - start_9 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   - 8 hold_9_8 - 9 hold_9_9 - 10 hold_9_10 - 11 hold_9_11
   + 60 pair_ok_9_7_for_nic_out_9_nic_in_7 <= 60
 constr_edge_switch_end_start_9_7_OR: switch_nic_out_9_nic_in_7
   - pair_ok_9_7_for_nic_out_9_nic_in_7 <= 0
 constr_pair_ok1_9_8: - start_8 + start_9 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 8 hold_9_8 + 9 hold_9_9 + 10 hold_9_10 + 11 hold_9_11
   + 60 pair_ok_9_8_for_nic_out_9_nic_in_8 <= 60
 constr_pair_ok2_9_8: start_8 - start_9 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   - 8 hold_9_8 - 9 hold_9_9 - 10 hold_9_10 - 11 hold_9_11
   + 60 pair_ok_9_8_for_nic_out_9_nic_in_8 <= 60
 constr_edge_switch_end_start_9_8_OR: switch_nic_out_9_nic_in_8
   - pair_ok_9_8_for_nic_out_9_nic_in_8 <= 0
 constr_pair_ok1_9_10: start_9 - start_10 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 8 hold_9_8 + 9 hold_9_9 + 10 hold_9_10 + 11 hold_9_11
   + 60 pair_ok_9_10_for_nic_out_9_nic_in_10 <= 60
 constr_pair_ok2_9_10: - start_9 + start_10 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   - 8 hold_9_8 - 9 hold_9_9 - 10 hold_9_10 - 11 hold_9_11
   + 60 pair_ok_9_10_for_nic_out_9_nic_in_10 <= 60
 constr_edge_switch_end_start_9_10_OR: switch_nic_out_9_nic_in_10
   - pair_ok_9_10_for_nic_out_9_nic_in_10 <= 0
 constr_pair_ok1_9_11: start_9 - start_11 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 8 hold_9_8 + 9 hold_9_9 + 10 hold_9_10 + 11 hold_9_11
   + 60 pair_ok_9_11_for_nic_out_9_nic_in_11 <= 60
 constr_pair_ok2_9_11: - start_9 + start_11 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   - 8 hold_9_8 - 9 hold_9_9 - 10 hold_9_10 - 11 hold_9_11
   + 60 pair_ok_9_11_for_nic_out_9_nic_in_11 <= 60
 constr_edge_switch_end_start_9_11_OR: switch_nic_out_9_nic_in_11
   - pair_ok_9_11_for_nic_out_9_nic_in_11 <= 0
 force_constr_distrib_nic_in_9_9: switch_nic_in_9_9 = 1
 force_constr_distrib_9_nic_out_9: switch_9_nic_out_9 = 1
 constr_is_start_zero_10: start_10
   + 60 is_start_zero_10_for_source_nic_in_10 <= 60
 constr_edge_switch_start_zero_10_OR: switch_source_nic_in_10
   - is_start_zero_10_for_source_nic_in_10 <= 0
 constr_is_end_complete_10: start_10 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   - 60 is_end_complete_10 >= 0
 constr_edge_switch_end_max_10_OR: switch_nic_out_10_sink
   - is_end_complete_10 <= 0
 constr_pair_ok1_10_0: - start_0 + start_10 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 60 pair_ok_10_0_for_nic_out_10_nic_in_0 <= 60
 constr_pair_ok2_10_0: start_0 - start_10 - hold_10_1 - 2 hold_10_2
   - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6 - 7 hold_10_7
   - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10 - 11 hold_10_11
   + 60 pair_ok_10_0_for_nic_out_10_nic_in_0 <= 60
 constr_edge_switch_end_start_10_0_OR: switch_nic_out_10_nic_in_0
   - pair_ok_10_0_for_nic_out_10_nic_in_0 <= 0
 constr_pair_ok1_10_1: - start_1 + start_10 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 60 pair_ok_10_1_for_nic_out_10_nic_in_1 <= 60
 constr_pair_ok2_10_1: start_1 - start_10 - hold_10_1 - 2 hold_10_2
   - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6 - 7 hold_10_7
   - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10 - 11 hold_10_11
   + 60 pair_ok_10_1_for_nic_out_10_nic_in_1 <= 60
 constr_edge_switch_end_start_10_1_OR: switch_nic_out_10_nic_in_1
   - pair_ok_10_1_for_nic_out_10_nic_in_1 <= 0
 constr_pair_ok1_10_2: - start_2 + start_10 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 60 pair_ok_10_2_for_nic_out_10_nic_in_2 <= 60
 constr_pair_ok2_10_2: start_2 - start_10 - hold_10_1 - 2 hold_10_2
   - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6 - 7 hold_10_7
   - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10 - 11 hold_10_11
   + 60 pair_ok_10_2_for_nic_out_10_nic_in_2 <= 60
 constr_edge_switch_end_start_10_2_OR: switch_nic_out_10_nic_in_2
   - pair_ok_10_2_for_nic_out_10_nic_in_2 <= 0
 constr_pair_ok1_10_3: - start_3 + start_10 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 60 pair_ok_10_3_for_nic_out_10_nic_in_3 <= 60
 constr_pair_ok2_10_3: start_3 - start_10 - hold_10_1 - 2 hold_10_2
   - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6 - 7 hold_10_7
   - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10 - 11 hold_10_11
   + 60 pair_ok_10_3_for_nic_out_10_nic_in_3 <= 60
 constr_edge_switch_end_start_10_3_OR: switch_nic_out_10_nic_in_3
   - pair_ok_10_3_for_nic_out_10_nic_in_3 <= 0
 constr_pair_ok1_10_4: - start_4 + start_10 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 60 pair_ok_10_4_for_nic_out_10_nic_in_4 <= 60
 constr_pair_ok2_10_4: start_4 - start_10 - hold_10_1 - 2 hold_10_2
   - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6 - 7 hold_10_7
   - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10 - 11 hold_10_11
   + 60 pair_ok_10_4_for_nic_out_10_nic_in_4 <= 60
 constr_edge_switch_end_start_10_4_OR: switch_nic_out_10_nic_in_4
   - pair_ok_10_4_for_nic_out_10_nic_in_4 <= 0
 constr_pair_ok1_10_5: - start_5 + start_10 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 60 pair_ok_10_5_for_nic_out_10_nic_in_5 <= 60
 constr_pair_ok2_10_5: start_5 - start_10 - hold_10_1 - 2 hold_10_2
   - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6 - 7 hold_10_7
   - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10 - 11 hold_10_11
   + 60 pair_ok_10_5_for_nic_out_10_nic_in_5 <= 60
 constr_edge_switch_end_start_10_5_OR: switch_nic_out_10_nic_in_5
   - pair_ok_10_5_for_nic_out_10_nic_in_5 <= 0
 constr_pair_ok1_10_6: - start_6 + start_10 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 60 pair_ok_10_6_for_nic_out_10_nic_in_6 <= 60
 constr_pair_ok2_10_6: start_6 - start_10 - hold_10_1 - 2 hold_10_2
   - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6 - 7 hold_10_7
   - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10 - 11 hold_10_11
   + 60 pair_ok_10_6_for_nic_out_10_nic_in_6 <= 60
 constr_edge_switch_end_start_10_6_OR: switch_nic_out_10_nic_in_6
   - pair_ok_10_6_for_nic_out_10_nic_in_6 <= 0
 constr_pair_ok1_10_7: - start_7 + start_10 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 60 pair_ok_10_7_for_nic_out_10_nic_in_7 <= 60
 constr_pair_ok2_10_7: start_7 - start_10 - hold_10_1 - 2 hold_10_2
   - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6 - 7 hold_10_7
   - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10 - 11 hold_10_11
   + 60 pair_ok_10_7_for_nic_out_10_nic_in_7 <= 60
 constr_edge_switch_end_start_10_7_OR: switch_nic_out_10_nic_in_7
   - pair_ok_10_7_for_nic_out_10_nic_in_7 <= 0
 constr_pair_ok1_10_8: - start_8 + start_10 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 60 pair_ok_10_8_for_nic_out_10_nic_in_8 <= 60
 constr_pair_ok2_10_8: start_8 - start_10 - hold_10_1 - 2 hold_10_2
   - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6 - 7 hold_10_7
   - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10 - 11 hold_10_11
   + 60 pair_ok_10_8_for_nic_out_10_nic_in_8 <= 60
 constr_edge_switch_end_start_10_8_OR: switch_nic_out_10_nic_in_8
   - pair_ok_10_8_for_nic_out_10_nic_in_8 <= 0
 constr_pair_ok1_10_9: - start_9 + start_10 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 60 pair_ok_10_9_for_nic_out_10_nic_in_9 <= 60
 constr_pair_ok2_10_9: start_9 - start_10 - hold_10_1 - 2 hold_10_2
   - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6 - 7 hold_10_7
   - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10 - 11 hold_10_11
   + 60 pair_ok_10_9_for_nic_out_10_nic_in_9 <= 60
 constr_edge_switch_end_start_10_9_OR: switch_nic_out_10_nic_in_9
   - pair_ok_10_9_for_nic_out_10_nic_in_9 <= 0
 constr_pair_ok1_10_11: start_10 - start_11 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 60 pair_ok_10_11_for_nic_out_10_nic_in_11 <= 60
 constr_pair_ok2_10_11: - start_10 + start_11 - hold_10_1 - 2 hold_10_2
   - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6 - 7 hold_10_7
   - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10 - 11 hold_10_11
   + 60 pair_ok_10_11_for_nic_out_10_nic_in_11 <= 60
 constr_edge_switch_end_start_10_11_OR: switch_nic_out_10_nic_in_11
   - pair_ok_10_11_for_nic_out_10_nic_in_11 <= 0
 force_constr_distrib_nic_in_10_10: switch_nic_in_10_10 = 1
 force_constr_distrib_10_nic_out_10: switch_10_nic_out_10 = 1
 constr_is_start_zero_11: start_11
   + 60 is_start_zero_11_for_source_nic_in_11 <= 60
 constr_edge_switch_start_zero_11_OR: switch_source_nic_in_11
   - is_start_zero_11_for_source_nic_in_11 <= 0
 constr_is_end_complete_11: start_11 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   - 60 is_end_complete_11 >= 0
 constr_edge_switch_end_max_11_OR: switch_nic_out_11_sink
   - is_end_complete_11 <= 0
 constr_pair_ok1_11_0: - start_0 + start_11 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 60 pair_ok_11_0_for_nic_out_11_nic_in_0 <= 60
 constr_pair_ok2_11_0: start_0 - start_11 - hold_11_1 - 2 hold_11_2
   - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6 - 7 hold_11_7
   - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10 - 11 hold_11_11
   + 60 pair_ok_11_0_for_nic_out_11_nic_in_0 <= 60
 constr_edge_switch_end_start_11_0_OR: switch_nic_out_11_nic_in_0
   - pair_ok_11_0_for_nic_out_11_nic_in_0 <= 0
 constr_pair_ok1_11_1: - start_1 + start_11 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 60 pair_ok_11_1_for_nic_out_11_nic_in_1 <= 60
 constr_pair_ok2_11_1: start_1 - start_11 - hold_11_1 - 2 hold_11_2
   - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6 - 7 hold_11_7
   - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10 - 11 hold_11_11
   + 60 pair_ok_11_1_for_nic_out_11_nic_in_1 <= 60
 constr_edge_switch_end_start_11_1_OR: switch_nic_out_11_nic_in_1
   - pair_ok_11_1_for_nic_out_11_nic_in_1 <= 0
 constr_pair_ok1_11_2: - start_2 + start_11 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 60 pair_ok_11_2_for_nic_out_11_nic_in_2 <= 60
 constr_pair_ok2_11_2: start_2 - start_11 - hold_11_1 - 2 hold_11_2
   - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6 - 7 hold_11_7
   - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10 - 11 hold_11_11
   + 60 pair_ok_11_2_for_nic_out_11_nic_in_2 <= 60
 constr_edge_switch_end_start_11_2_OR: switch_nic_out_11_nic_in_2
   - pair_ok_11_2_for_nic_out_11_nic_in_2 <= 0
 constr_pair_ok1_11_3: - start_3 + start_11 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 60 pair_ok_11_3_for_nic_out_11_nic_in_3 <= 60
 constr_pair_ok2_11_3: start_3 - start_11 - hold_11_1 - 2 hold_11_2
   - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6 - 7 hold_11_7
   - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10 - 11 hold_11_11
   + 60 pair_ok_11_3_for_nic_out_11_nic_in_3 <= 60
 constr_edge_switch_end_start_11_3_OR: switch_nic_out_11_nic_in_3
   - pair_ok_11_3_for_nic_out_11_nic_in_3 <= 0
 constr_pair_ok1_11_4: - start_4 + start_11 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 60 pair_ok_11_4_for_nic_out_11_nic_in_4 <= 60
 constr_pair_ok2_11_4: start_4 - start_11 - hold_11_1 - 2 hold_11_2
   - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6 - 7 hold_11_7
   - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10 - 11 hold_11_11
   + 60 pair_ok_11_4_for_nic_out_11_nic_in_4 <= 60
 constr_edge_switch_end_start_11_4_OR: switch_nic_out_11_nic_in_4
   - pair_ok_11_4_for_nic_out_11_nic_in_4 <= 0
 constr_pair_ok1_11_5: - start_5 + start_11 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 60 pair_ok_11_5_for_nic_out_11_nic_in_5 <= 60
 constr_pair_ok2_11_5: start_5 - start_11 - hold_11_1 - 2 hold_11_2
   - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6 - 7 hold_11_7
   - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10 - 11 hold_11_11
   + 60 pair_ok_11_5_for_nic_out_11_nic_in_5 <= 60
 constr_edge_switch_end_start_11_5_OR: switch_nic_out_11_nic_in_5
   - pair_ok_11_5_for_nic_out_11_nic_in_5 <= 0
 constr_pair_ok1_11_6: - start_6 + start_11 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 60 pair_ok_11_6_for_nic_out_11_nic_in_6 <= 60
 constr_pair_ok2_11_6: start_6 - start_11 - hold_11_1 - 2 hold_11_2
   - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6 - 7 hold_11_7
   - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10 - 11 hold_11_11
   + 60 pair_ok_11_6_for_nic_out_11_nic_in_6 <= 60
 constr_edge_switch_end_start_11_6_OR: switch_nic_out_11_nic_in_6
   - pair_ok_11_6_for_nic_out_11_nic_in_6 <= 0
 constr_pair_ok1_11_7: - start_7 + start_11 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 60 pair_ok_11_7_for_nic_out_11_nic_in_7 <= 60
 constr_pair_ok2_11_7: start_7 - start_11 - hold_11_1 - 2 hold_11_2
   - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6 - 7 hold_11_7
   - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10 - 11 hold_11_11
   + 60 pair_ok_11_7_for_nic_out_11_nic_in_7 <= 60
 constr_edge_switch_end_start_11_7_OR: switch_nic_out_11_nic_in_7
   - pair_ok_11_7_for_nic_out_11_nic_in_7 <= 0
 constr_pair_ok1_11_8: - start_8 + start_11 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 60 pair_ok_11_8_for_nic_out_11_nic_in_8 <= 60
 constr_pair_ok2_11_8: start_8 - start_11 - hold_11_1 - 2 hold_11_2
   - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6 - 7 hold_11_7
   - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10 - 11 hold_11_11
   + 60 pair_ok_11_8_for_nic_out_11_nic_in_8 <= 60
 constr_edge_switch_end_start_11_8_OR: switch_nic_out_11_nic_in_8
   - pair_ok_11_8_for_nic_out_11_nic_in_8 <= 0
 constr_pair_ok1_11_9: - start_9 + start_11 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 60 pair_ok_11_9_for_nic_out_11_nic_in_9 <= 60
 constr_pair_ok2_11_9: start_9 - start_11 - hold_11_1 - 2 hold_11_2
   - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6 - 7 hold_11_7
   - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10 - 11 hold_11_11
   + 60 pair_ok_11_9_for_nic_out_11_nic_in_9 <= 60
 constr_edge_switch_end_start_11_9_OR: switch_nic_out_11_nic_in_9
   - pair_ok_11_9_for_nic_out_11_nic_in_9 <= 0
 constr_pair_ok1_11_10: - start_10 + start_11 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 60 pair_ok_11_10_for_nic_out_11_nic_in_10 <= 60
 constr_pair_ok2_11_10: start_10 - start_11 - hold_11_1 - 2 hold_11_2
   - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6 - 7 hold_11_7
   - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10 - 11 hold_11_11
   + 60 pair_ok_11_10_for_nic_out_11_nic_in_10 <= 60
 constr_edge_switch_end_start_11_10_OR: switch_nic_out_11_nic_in_10
   - pair_ok_11_10_for_nic_out_11_nic_in_10 <= 0
 force_constr_distrib_nic_in_11_11: switch_nic_in_11_11 = 1
 force_constr_distrib_11_nic_out_11: switch_11_nic_out_11 = 1
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
 node_throughput_constraint_0: group_tp_active_0 = 0 -> - 2400 hold_0_1
   - 1200 hold_0_2 - 800 hold_0_3 - 600 hold_0_4 - 480 hold_0_5
   - 400 hold_0_6 - 342.8571428571428 hold_0_7 - 300 hold_0_8
   - 266.6666666666667 hold_0_9 - 240 hold_0_10
   - 218.1818181818182 hold_0_11 + flow_nic_in_0_0 <= 0
 node_throughput_constraint_1: group_tp_active_1 = 0 -> - 2400 hold_1_1
   - 1200 hold_1_2 - 800 hold_1_3 - 600 hold_1_4 - 480 hold_1_5
   - 400 hold_1_6 - 342.8571428571428 hold_1_7 - 300 hold_1_8
   - 266.6666666666667 hold_1_9 - 240 hold_1_10
   - 218.1818181818182 hold_1_11 + flow_nic_in_1_1 <= 0
 node_throughput_constraint_2: group_tp_active_2 = 0 -> - 2400 hold_2_1
   - 1200 hold_2_2 - 800 hold_2_3 - 600 hold_2_4 - 480 hold_2_5
   - 400 hold_2_6 - 342.8571428571428 hold_2_7 - 300 hold_2_8
   - 266.6666666666667 hold_2_9 - 240 hold_2_10
   - 218.1818181818182 hold_2_11 + flow_nic_in_2_2 <= 0
 node_throughput_constraint_3: group_tp_active_3 = 0 -> - 2400 hold_3_1
   - 1200 hold_3_2 - 800 hold_3_3 - 600 hold_3_4 - 480 hold_3_5
   - 400 hold_3_6 - 342.8571428571428 hold_3_7 - 300 hold_3_8
   - 266.6666666666667 hold_3_9 - 240 hold_3_10
   - 218.1818181818182 hold_3_11 + flow_nic_in_3_3 <= 0
 node_throughput_constraint_4: group_tp_active_4 = 0 -> - 2400 hold_4_1
   - 1200 hold_4_2 - 800 hold_4_3 - 600 hold_4_4 - 480 hold_4_5
   - 400 hold_4_6 - 342.8571428571428 hold_4_7 - 300 hold_4_8
   - 266.6666666666667 hold_4_9 - 240 hold_4_10
   - 218.1818181818182 hold_4_11 + flow_nic_in_4_4 <= 0
 node_throughput_constraint_5: group_tp_active_5 = 0 -> - 2400 hold_5_1
   - 1200 hold_5_2 - 800 hold_5_3 - 600 hold_5_4 - 480 hold_5_5
   - 400 hold_5_6 - 342.8571428571428 hold_5_7 - 300 hold_5_8
   - 266.6666666666667 hold_5_9 - 240 hold_5_10
   - 218.1818181818182 hold_5_11 + flow_nic_in_5_5 <= 0
 node_throughput_constraint_6: group_tp_active_6 = 0 -> - 2400 hold_6_1
   - 1200 hold_6_2 - 800 hold_6_3 - 600 hold_6_4 - 480 hold_6_5
   - 400 hold_6_6 - 342.8571428571428 hold_6_7 - 300 hold_6_8
   - 266.6666666666667 hold_6_9 - 240 hold_6_10
   - 218.1818181818182 hold_6_11 + flow_nic_in_6_6 <= 0
 node_throughput_constraint_7: group_tp_active_7 = 0 -> - 2400 hold_7_1
   - 1200 hold_7_2 - 800 hold_7_3 - 600 hold_7_4 - 480 hold_7_5
   - 400 hold_7_6 - 342.8571428571428 hold_7_7 - 300 hold_7_8
   - 266.6666666666667 hold_7_9 - 240 hold_7_10
   - 218.1818181818182 hold_7_11 + flow_nic_in_7_7 <= 0
 node_throughput_constraint_8: group_tp_active_8 = 0 -> - 2400 hold_8_1
   - 1200 hold_8_2 - 800 hold_8_3 - 600 hold_8_4 - 480 hold_8_5
   - 400 hold_8_6 - 342.8571428571428 hold_8_7 - 300 hold_8_8
   - 266.6666666666667 hold_8_9 - 240 hold_8_10
   - 218.1818181818182 hold_8_11 + flow_nic_in_8_8 <= 0
 node_throughput_constraint_9: group_tp_active_9 = 0 -> - 2400 hold_9_1
   - 1200 hold_9_2 - 800 hold_9_3 - 600 hold_9_4 - 480 hold_9_5
   - 400 hold_9_6 - 342.8571428571428 hold_9_7 - 300 hold_9_8
   - 266.6666666666667 hold_9_9 - 240 hold_9_10
   - 218.1818181818182 hold_9_11 + flow_nic_in_9_9 <= 0
 node_throughput_constraint_10: group_tp_active_10 = 0 -> - 2400 hold_10_1
   - 1200 hold_10_2 - 800 hold_10_3 - 600 hold_10_4 - 480 hold_10_5
   - 400 hold_10_6 - 342.8571428571428 hold_10_7 - 300 hold_10_8
   - 266.6666666666667 hold_10_9 - 240 hold_10_10
   - 218.1818181818182 hold_10_11 + flow_nic_in_10_10 <= 0
 node_throughput_constraint_11: group_tp_active_11 = 0 -> - 2400 hold_11_1
   - 1200 hold_11_2 - 800 hold_11_3 - 600 hold_11_4 - 480 hold_11_5
   - 400 hold_11_6 - 342.8571428571428 hold_11_7 - 300 hold_11_8
   - 266.6666666666667 hold_11_9 - 240 hold_11_10
   - 218.1818181818182 hold_11_11 + flow_nic_in_11_11 <= 0
 group_throughput_constr_0: group_tp_active_0 = 1 -> - 2400 hold_0_1
   - 1200 hold_0_2 - 800 hold_0_3 - 600 hold_0_4 - 480 hold_0_5
   - 400 hold_0_6 - 342.8571428571428 hold_0_7 - 300 hold_0_8
   - 266.6666666666667 hold_0_9 - 240 hold_0_10
   - 218.1818181818182 hold_0_11 + flow_nic_in_0_0 <= 0
 group_throughput_constr_1: group_tp_active_1 = 1 -> - 2400 hold_1_1
   - 1200 hold_1_2 - 800 hold_1_3 - 600 hold_1_4 - 480 hold_1_5
   - 400 hold_1_6 - 342.8571428571428 hold_1_7 - 300 hold_1_8
   - 266.6666666666667 hold_1_9 - 240 hold_1_10
   - 218.1818181818182 hold_1_11 + flow_nic_in_1_1 <= 0
 group_throughput_constr_2: group_tp_active_2 = 1 -> - 2400 hold_2_1
   - 1200 hold_2_2 - 800 hold_2_3 - 600 hold_2_4 - 480 hold_2_5
   - 400 hold_2_6 - 342.8571428571428 hold_2_7 - 300 hold_2_8
   - 266.6666666666667 hold_2_9 - 240 hold_2_10
   - 218.1818181818182 hold_2_11 + flow_nic_in_2_2 <= 0
 group_throughput_constr_3: group_tp_active_3 = 1 -> - 2400 hold_3_1
   - 1200 hold_3_2 - 800 hold_3_3 - 600 hold_3_4 - 480 hold_3_5
   - 400 hold_3_6 - 342.8571428571428 hold_3_7 - 300 hold_3_8
   - 266.6666666666667 hold_3_9 - 240 hold_3_10
   - 218.1818181818182 hold_3_11 + flow_nic_in_3_3 <= 0
 group_throughput_constr_4: group_tp_active_4 = 1 -> - 2400 hold_4_1
   - 1200 hold_4_2 - 800 hold_4_3 - 600 hold_4_4 - 480 hold_4_5
   - 400 hold_4_6 - 342.8571428571428 hold_4_7 - 300 hold_4_8
   - 266.6666666666667 hold_4_9 - 240 hold_4_10
   - 218.1818181818182 hold_4_11 + flow_nic_in_4_4 <= 0
 group_throughput_constr_5: group_tp_active_5 = 1 -> - 2400 hold_5_1
   - 1200 hold_5_2 - 800 hold_5_3 - 600 hold_5_4 - 480 hold_5_5
   - 400 hold_5_6 - 342.8571428571428 hold_5_7 - 300 hold_5_8
   - 266.6666666666667 hold_5_9 - 240 hold_5_10
   - 218.1818181818182 hold_5_11 + flow_nic_in_5_5 <= 0
 group_throughput_constr_6: group_tp_active_6 = 1 -> - 2400 hold_6_1
   - 1200 hold_6_2 - 800 hold_6_3 - 600 hold_6_4 - 480 hold_6_5
   - 400 hold_6_6 - 342.8571428571428 hold_6_7 - 300 hold_6_8
   - 266.6666666666667 hold_6_9 - 240 hold_6_10
   - 218.1818181818182 hold_6_11 + flow_nic_in_6_6 <= 0
 group_throughput_constr_7: group_tp_active_7 = 1 -> - 2400 hold_7_1
   - 1200 hold_7_2 - 800 hold_7_3 - 600 hold_7_4 - 480 hold_7_5
   - 400 hold_7_6 - 342.8571428571428 hold_7_7 - 300 hold_7_8
   - 266.6666666666667 hold_7_9 - 240 hold_7_10
   - 218.1818181818182 hold_7_11 + flow_nic_in_7_7 <= 0
 group_throughput_constr_8: group_tp_active_8 = 1 -> - 2400 hold_8_1
   - 1200 hold_8_2 - 800 hold_8_3 - 600 hold_8_4 - 480 hold_8_5
   - 400 hold_8_6 - 342.8571428571428 hold_8_7 - 300 hold_8_8
   - 266.6666666666667 hold_8_9 - 240 hold_8_10
   - 218.1818181818182 hold_8_11 + flow_nic_in_8_8 <= 0
 group_throughput_constr_9: group_tp_active_9 = 1 -> - 2400 hold_9_1
   - 1200 hold_9_2 - 800 hold_9_3 - 600 hold_9_4 - 480 hold_9_5
   - 400 hold_9_6 - 342.8571428571428 hold_9_7 - 300 hold_9_8
   - 266.6666666666667 hold_9_9 - 240 hold_9_10
   - 218.1818181818182 hold_9_11 + flow_nic_in_9_9 <= 0
 group_throughput_constr_10: group_tp_active_10 = 1 -> - 2400 hold_10_1
   - 1200 hold_10_2 - 800 hold_10_3 - 600 hold_10_4 - 480 hold_10_5
   - 400 hold_10_6 - 342.8571428571428 hold_10_7 - 300 hold_10_8
   - 266.6666666666667 hold_10_9 - 240 hold_10_10
   - 218.1818181818182 hold_10_11 + flow_nic_in_10_10 <= 0
 group_throughput_constr_11: group_tp_active_11 = 1 -> - 2400 hold_11_1
   - 1200 hold_11_2 - 800 hold_11_3 - 600 hold_11_4 - 480 hold_11_5
   - 400 hold_11_6 - 342.8571428571428 hold_11_7 - 300 hold_11_8
   - 266.6666666666667 hold_11_9 - 240 hold_11_10
   - 218.1818181818182 hold_11_11 + flow_nic_in_11_11 <= 0
Bounds
Binaries
 hold_0_1 hold_0_2 hold_0_3 hold_0_4 hold_0_5 hold_0_6 hold_0_7 hold_0_8
 hold_0_9 hold_0_10 hold_0_11 hold_1_1 hold_1_2 hold_1_3 hold_1_4 hold_1_5
 hold_1_6 hold_1_7 hold_1_8 hold_1_9 hold_1_10 hold_1_11 hold_2_1 hold_2_2
 hold_2_3 hold_2_4 hold_2_5 hold_2_6 hold_2_7 hold_2_8 hold_2_9 hold_2_10
 hold_2_11 hold_3_1 hold_3_2 hold_3_3 hold_3_4 hold_3_5 hold_3_6 hold_3_7
 hold_3_8 hold_3_9 hold_3_10 hold_3_11 hold_4_1 hold_4_2 hold_4_3 hold_4_4
 hold_4_5 hold_4_6 hold_4_7 hold_4_8 hold_4_9 hold_4_10 hold_4_11 hold_5_1
 hold_5_2 hold_5_3 hold_5_4 hold_5_5 hold_5_6 hold_5_7 hold_5_8 hold_5_9
 hold_5_10 hold_5_11 hold_6_1 hold_6_2 hold_6_3 hold_6_4 hold_6_5 hold_6_6
 hold_6_7 hold_6_8 hold_6_9 hold_6_10 hold_6_11 hold_7_1 hold_7_2 hold_7_3
 hold_7_4 hold_7_5 hold_7_6 hold_7_7 hold_7_8 hold_7_9 hold_7_10 hold_7_11
 hold_8_1 hold_8_2 hold_8_3 hold_8_4 hold_8_5 hold_8_6 hold_8_7 hold_8_8
 hold_8_9 hold_8_10 hold_8_11 hold_9_1 hold_9_2 hold_9_3 hold_9_4 hold_9_5
 hold_9_6 hold_9_7 hold_9_8 hold_9_9 hold_9_10 hold_9_11 hold_10_1
 hold_10_2 hold_10_3 hold_10_4 hold_10_5 hold_10_6 hold_10_7 hold_10_8
 hold_10_9 hold_10_10 hold_10_11 hold_11_1 hold_11_2 hold_11_3 hold_11_4
 hold_11_5 hold_11_6 hold_11_7 hold_11_8 hold_11_9 hold_11_10 hold_11_11
 switch_source_nic_in_0 switch_nic_out_0_sink switch_nic_out_0_nic_in_1
 switch_nic_out_0_nic_in_2 switch_nic_out_0_nic_in_3
 switch_nic_out_0_nic_in_4 switch_nic_out_0_nic_in_5
 switch_nic_out_0_nic_in_6 switch_nic_out_0_nic_in_7
 switch_nic_out_0_nic_in_8 switch_nic_out_0_nic_in_9
 switch_nic_out_0_nic_in_10 switch_nic_out_0_nic_in_11 switch_nic_in_0_0
 switch_0_nic_out_0 switch_source_nic_in_1 switch_nic_out_1_sink
 switch_nic_out_1_nic_in_0 switch_nic_out_1_nic_in_2
 switch_nic_out_1_nic_in_3 switch_nic_out_1_nic_in_4
 switch_nic_out_1_nic_in_5 switch_nic_out_1_nic_in_6
 switch_nic_out_1_nic_in_7 switch_nic_out_1_nic_in_8
 switch_nic_out_1_nic_in_9 switch_nic_out_1_nic_in_10
 switch_nic_out_1_nic_in_11 switch_nic_in_1_1 switch_1_nic_out_1
 switch_source_nic_in_2 switch_nic_out_2_sink switch_nic_out_2_nic_in_0
 switch_nic_out_2_nic_in_1 switch_nic_out_2_nic_in_3
 switch_nic_out_2_nic_in_4 switch_nic_out_2_nic_in_5
 switch_nic_out_2_nic_in_6 switch_nic_out_2_nic_in_7
 switch_nic_out_2_nic_in_8 switch_nic_out_2_nic_in_9
 switch_nic_out_2_nic_in_10 switch_nic_out_2_nic_in_11 switch_nic_in_2_2
 switch_2_nic_out_2 switch_source_nic_in_3 switch_nic_out_3_sink
 switch_nic_out_3_nic_in_0 switch_nic_out_3_nic_in_1
 switch_nic_out_3_nic_in_2 switch_nic_out_3_nic_in_4
 switch_nic_out_3_nic_in_5 switch_nic_out_3_nic_in_6
 switch_nic_out_3_nic_in_7 switch_nic_out_3_nic_in_8
 switch_nic_out_3_nic_in_9 switch_nic_out_3_nic_in_10
 switch_nic_out_3_nic_in_11 switch_nic_in_3_3 switch_3_nic_out_3
 switch_source_nic_in_4 switch_nic_out_4_sink switch_nic_out_4_nic_in_0
 switch_nic_out_4_nic_in_1 switch_nic_out_4_nic_in_2
 switch_nic_out_4_nic_in_3 switch_nic_out_4_nic_in_5
 switch_nic_out_4_nic_in_6 switch_nic_out_4_nic_in_7
 switch_nic_out_4_nic_in_8 switch_nic_out_4_nic_in_9
 switch_nic_out_4_nic_in_10 switch_nic_out_4_nic_in_11 switch_nic_in_4_4
 switch_4_nic_out_4 switch_source_nic_in_5 switch_nic_out_5_sink
 switch_nic_out_5_nic_in_0 switch_nic_out_5_nic_in_1
 switch_nic_out_5_nic_in_2 switch_nic_out_5_nic_in_3
 switch_nic_out_5_nic_in_4 switch_nic_out_5_nic_in_6
 switch_nic_out_5_nic_in_7 switch_nic_out_5_nic_in_8
 switch_nic_out_5_nic_in_9 switch_nic_out_5_nic_in_10
 switch_nic_out_5_nic_in_11 switch_nic_in_5_5 switch_5_nic_out_5
 switch_source_nic_in_6 switch_nic_out_6_sink switch_nic_out_6_nic_in_0
 switch_nic_out_6_nic_in_1 switch_nic_out_6_nic_in_2
 switch_nic_out_6_nic_in_3 switch_nic_out_6_nic_in_4
 switch_nic_out_6_nic_in_5 switch_nic_out_6_nic_in_7
 switch_nic_out_6_nic_in_8 switch_nic_out_6_nic_in_9
 switch_nic_out_6_nic_in_10 switch_nic_out_6_nic_in_11 switch_nic_in_6_6
 switch_6_nic_out_6 switch_source_nic_in_7 switch_nic_out_7_sink
 switch_nic_out_7_nic_in_0 switch_nic_out_7_nic_in_1
 switch_nic_out_7_nic_in_2 switch_nic_out_7_nic_in_3
 switch_nic_out_7_nic_in_4 switch_nic_out_7_nic_in_5
 switch_nic_out_7_nic_in_6 switch_nic_out_7_nic_in_8
 switch_nic_out_7_nic_in_9 switch_nic_out_7_nic_in_10
 switch_nic_out_7_nic_in_11 switch_nic_in_7_7 switch_7_nic_out_7
 switch_source_nic_in_8 switch_nic_out_8_sink switch_nic_out_8_nic_in_0
 switch_nic_out_8_nic_in_1 switch_nic_out_8_nic_in_2
 switch_nic_out_8_nic_in_3 switch_nic_out_8_nic_in_4
 switch_nic_out_8_nic_in_5 switch_nic_out_8_nic_in_6
 switch_nic_out_8_nic_in_7 switch_nic_out_8_nic_in_9
 switch_nic_out_8_nic_in_10 switch_nic_out_8_nic_in_11 switch_nic_in_8_8
 switch_8_nic_out_8 switch_source_nic_in_9 switch_nic_out_9_sink
 switch_nic_out_9_nic_in_0 switch_nic_out_9_nic_in_1
 switch_nic_out_9_nic_in_2 switch_nic_out_9_nic_in_3
 switch_nic_out_9_nic_in_4 switch_nic_out_9_nic_in_5
 switch_nic_out_9_nic_in_6 switch_nic_out_9_nic_in_7
 switch_nic_out_9_nic_in_8 switch_nic_out_9_nic_in_10
 switch_nic_out_9_nic_in_11 switch_nic_in_9_9 switch_9_nic_out_9
 switch_source_nic_in_10 switch_nic_out_10_sink switch_nic_out_10_nic_in_0
 switch_nic_out_10_nic_in_1 switch_nic_out_10_nic_in_2
 switch_nic_out_10_nic_in_3 switch_nic_out_10_nic_in_4
 switch_nic_out_10_nic_in_5 switch_nic_out_10_nic_in_6
 switch_nic_out_10_nic_in_7 switch_nic_out_10_nic_in_8
 switch_nic_out_10_nic_in_9 switch_nic_out_10_nic_in_11 switch_nic_in_10_10
 switch_10_nic_out_10 switch_source_nic_in_11 switch_nic_out_11_sink
 switch_nic_out_11_nic_in_0 switch_nic_out_11_nic_in_1
 switch_nic_out_11_nic_in_2 switch_nic_out_11_nic_in_3
 switch_nic_out_11_nic_in_4 switch_nic_out_11_nic_in_5
 switch_nic_out_11_nic_in_6 switch_nic_out_11_nic_in_7
 switch_nic_out_11_nic_in_8 switch_nic_out_11_nic_in_9
 switch_nic_out_11_nic_in_10 switch_nic_in_11_11 switch_11_nic_out_11
 group_tp_active_0 group_tp_active_1 group_tp_active_2 group_tp_active_3
 group_tp_active_4 group_tp_active_5 group_tp_active_6 group_tp_active_7
 group_tp_active_8 group_tp_active_9 group_tp_active_10 group_tp_active_11
 is_start_zero_0_for_source_nic_in_0 is_end_complete_0
 pair_ok_0_1_for_nic_out_0_nic_in_1 pair_ok_0_2_for_nic_out_0_nic_in_2
 pair_ok_0_3_for_nic_out_0_nic_in_3 pair_ok_0_4_for_nic_out_0_nic_in_4
 pair_ok_0_5_for_nic_out_0_nic_in_5 pair_ok_0_6_for_nic_out_0_nic_in_6
 pair_ok_0_7_for_nic_out_0_nic_in_7 pair_ok_0_8_for_nic_out_0_nic_in_8
 pair_ok_0_9_for_nic_out_0_nic_in_9 pair_ok_0_10_for_nic_out_0_nic_in_10
 pair_ok_0_11_for_nic_out_0_nic_in_11 is_start_zero_1_for_source_nic_in_1
 is_end_complete_1 pair_ok_1_0_for_nic_out_1_nic_in_0
 pair_ok_1_2_for_nic_out_1_nic_in_2 pair_ok_1_3_for_nic_out_1_nic_in_3
 pair_ok_1_4_for_nic_out_1_nic_in_4 pair_ok_1_5_for_nic_out_1_nic_in_5
 pair_ok_1_6_for_nic_out_1_nic_in_6 pair_ok_1_7_for_nic_out_1_nic_in_7
 pair_ok_1_8_for_nic_out_1_nic_in_8 pair_ok_1_9_for_nic_out_1_nic_in_9
 pair_ok_1_10_for_nic_out_1_nic_in_10 pair_ok_1_11_for_nic_out_1_nic_in_11
 is_start_zero_2_for_source_nic_in_2 is_end_complete_2
 pair_ok_2_0_for_nic_out_2_nic_in_0 pair_ok_2_1_for_nic_out_2_nic_in_1
 pair_ok_2_3_for_nic_out_2_nic_in_3 pair_ok_2_4_for_nic_out_2_nic_in_4
 pair_ok_2_5_for_nic_out_2_nic_in_5 pair_ok_2_6_for_nic_out_2_nic_in_6
 pair_ok_2_7_for_nic_out_2_nic_in_7 pair_ok_2_8_for_nic_out_2_nic_in_8
 pair_ok_2_9_for_nic_out_2_nic_in_9 pair_ok_2_10_for_nic_out_2_nic_in_10
 pair_ok_2_11_for_nic_out_2_nic_in_11 is_start_zero_3_for_source_nic_in_3
 is_end_complete_3 pair_ok_3_0_for_nic_out_3_nic_in_0
 pair_ok_3_1_for_nic_out_3_nic_in_1 pair_ok_3_2_for_nic_out_3_nic_in_2
 pair_ok_3_4_for_nic_out_3_nic_in_4 pair_ok_3_5_for_nic_out_3_nic_in_5
 pair_ok_3_6_for_nic_out_3_nic_in_6 pair_ok_3_7_for_nic_out_3_nic_in_7
 pair_ok_3_8_for_nic_out_3_nic_in_8 pair_ok_3_9_for_nic_out_3_nic_in_9
 pair_ok_3_10_for_nic_out_3_nic_in_10 pair_ok_3_11_for_nic_out_3_nic_in_11
 is_start_zero_4_for_source_nic_in_4 is_end_complete_4
 pair_ok_4_0_for_nic_out_4_nic_in_0 pair_ok_4_1_for_nic_out_4_nic_in_1
 pair_ok_4_2_for_nic_out_4_nic_in_2 pair_ok_4_3_for_nic_out_4_nic_in_3
 pair_ok_4_5_for_nic_out_4_nic_in_5 pair_ok_4_6_for_nic_out_4_nic_in_6
 pair_ok_4_7_for_nic_out_4_nic_in_7 pair_ok_4_8_for_nic_out_4_nic_in_8
 pair_ok_4_9_for_nic_out_4_nic_in_9 pair_ok_4_10_for_nic_out_4_nic_in_10
 pair_ok_4_11_for_nic_out_4_nic_in_11 is_start_zero_5_for_source_nic_in_5
 is_end_complete_5 pair_ok_5_0_for_nic_out_5_nic_in_0
 pair_ok_5_1_for_nic_out_5_nic_in_1 pair_ok_5_2_for_nic_out_5_nic_in_2
 pair_ok_5_3_for_nic_out_5_nic_in_3 pair_ok_5_4_for_nic_out_5_nic_in_4
 pair_ok_5_6_for_nic_out_5_nic_in_6 pair_ok_5_7_for_nic_out_5_nic_in_7
 pair_ok_5_8_for_nic_out_5_nic_in_8 pair_ok_5_9_for_nic_out_5_nic_in_9
 pair_ok_5_10_for_nic_out_5_nic_in_10 pair_ok_5_11_for_nic_out_5_nic_in_11
 is_start_zero_6_for_source_nic_in_6 is_end_complete_6
 pair_ok_6_0_for_nic_out_6_nic_in_0 pair_ok_6_1_for_nic_out_6_nic_in_1
 pair_ok_6_2_for_nic_out_6_nic_in_2 pair_ok_6_3_for_nic_out_6_nic_in_3
 pair_ok_6_4_for_nic_out_6_nic_in_4 pair_ok_6_5_for_nic_out_6_nic_in_5
 pair_ok_6_7_for_nic_out_6_nic_in_7 pair_ok_6_8_for_nic_out_6_nic_in_8
 pair_ok_6_9_for_nic_out_6_nic_in_9 pair_ok_6_10_for_nic_out_6_nic_in_10
 pair_ok_6_11_for_nic_out_6_nic_in_11 is_start_zero_7_for_source_nic_in_7
 is_end_complete_7 pair_ok_7_0_for_nic_out_7_nic_in_0
 pair_ok_7_1_for_nic_out_7_nic_in_1 pair_ok_7_2_for_nic_out_7_nic_in_2
 pair_ok_7_3_for_nic_out_7_nic_in_3 pair_ok_7_4_for_nic_out_7_nic_in_4
 pair_ok_7_5_for_nic_out_7_nic_in_5 pair_ok_7_6_for_nic_out_7_nic_in_6
 pair_ok_7_8_for_nic_out_7_nic_in_8 pair_ok_7_9_for_nic_out_7_nic_in_9
 pair_ok_7_10_for_nic_out_7_nic_in_10 pair_ok_7_11_for_nic_out_7_nic_in_11
 is_start_zero_8_for_source_nic_in_8 is_end_complete_8
 pair_ok_8_0_for_nic_out_8_nic_in_0 pair_ok_8_1_for_nic_out_8_nic_in_1
 pair_ok_8_2_for_nic_out_8_nic_in_2 pair_ok_8_3_for_nic_out_8_nic_in_3
 pair_ok_8_4_for_nic_out_8_nic_in_4 pair_ok_8_5_for_nic_out_8_nic_in_5
 pair_ok_8_6_for_nic_out_8_nic_in_6 pair_ok_8_7_for_nic_out_8_nic_in_7
 pair_ok_8_9_for_nic_out_8_nic_in_9 pair_ok_8_10_for_nic_out_8_nic_in_10
 pair_ok_8_11_for_nic_out_8_nic_in_11 is_start_zero_9_for_source_nic_in_9
 is_end_complete_9 pair_ok_9_0_for_nic_out_9_nic_in_0
 pair_ok_9_1_for_nic_out_9_nic_in_1 pair_ok_9_2_for_nic_out_9_nic_in_2
 pair_ok_9_3_for_nic_out_9_nic_in_3 pair_ok_9_4_for_nic_out_9_nic_in_4
 pair_ok_9_5_for_nic_out_9_nic_in_5 pair_ok_9_6_for_nic_out_9_nic_in_6
 pair_ok_9_7_for_nic_out_9_nic_in_7 pair_ok_9_8_for_nic_out_9_nic_in_8
 pair_ok_9_10_for_nic_out_9_nic_in_10 pair_ok_9_11_for_nic_out_9_nic_in_11
 is_start_zero_10_for_source_nic_in_10 is_end_complete_10
 pair_ok_10_0_for_nic_out_10_nic_in_0 pair_ok_10_1_for_nic_out_10_nic_in_1
 pair_ok_10_2_for_nic_out_10_nic_in_2 pair_ok_10_3_for_nic_out_10_nic_in_3
 pair_ok_10_4_for_nic_out_10_nic_in_4 pair_ok_10_5_for_nic_out_10_nic_in_5
 pair_ok_10_6_for_nic_out_10_nic_in_6 pair_ok_10_7_for_nic_out_10_nic_in_7
 pair_ok_10_8_for_nic_out_10_nic_in_8 pair_ok_10_9_for_nic_out_10_nic_in_9
 pair_ok_10_11_for_nic_out_10_nic_in_11
 is_start_zero_11_for_source_nic_in_11 is_end_complete_11
 pair_ok_11_0_for_nic_out_11_nic_in_0 pair_ok_11_1_for_nic_out_11_nic_in_1
 pair_ok_11_2_for_nic_out_11_nic_in_2 pair_ok_11_3_for_nic_out_11_nic_in_3
 pair_ok_11_4_for_nic_out_11_nic_in_4 pair_ok_11_5_for_nic_out_11_nic_in_5
 pair_ok_11_6_for_nic_out_11_nic_in_6 pair_ok_11_7_for_nic_out_11_nic_in_7
 pair_ok_11_8_for_nic_out_11_nic_in_8 pair_ok_11_9_for_nic_out_11_nic_in_9
 pair_ok_11_10_for_nic_out_11_nic_in_10
Generals
 start_0 start_1 start_2 start_3 start_4 start_5 start_6 start_7 start_8
 start_9 start_10 start_11
End
