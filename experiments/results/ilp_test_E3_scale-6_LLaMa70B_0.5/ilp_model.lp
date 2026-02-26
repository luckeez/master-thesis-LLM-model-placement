\ Model 2026-02-23_23-01-16
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  8.554947203754399e-04 flow_source_nic_in_0
   + 8.554947203754399e-04 flow_source_nic_in_1
   + 8.554947203754399e-04 flow_source_nic_in_2
   + 8.554947203754399e-04 flow_source_nic_in_3
   + 8.554947203754399e-04 flow_source_nic_in_4
   - 0.1073446327683616 is_active_0 - 0.0728813559322034 is_active_1
   - 0.0932203389830508 is_active_2 - 0.0728813559322034 is_active_3
   - 0.0768361581920904 is_active_4 - 0.0768361581920904 is_active_5
   + 0 val_end_0 + 0 val_end_1 + 0 val_end_2 + 0 val_end_3 + 0 val_end_4
   + 0 val_end_5
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
 node_flow_constraint_nic_in_0: flow_source_nic_in_0 - flow_nic_in_0_0
   + flow_nic_out_1_nic_in_0 + flow_nic_out_2_nic_in_0
   + flow_nic_out_3_nic_in_0 + flow_nic_out_4_nic_in_0 = 0
 node_flow_constraint_nic_out_0: - flow_nic_out_0_sink
   - flow_nic_out_0_nic_in_1 - flow_nic_out_0_nic_in_2
   - flow_nic_out_0_nic_in_3 - flow_nic_out_0_nic_in_4 + flow_0_nic_out_0
   = 0
 node_flow_constraint_nic_in_1: flow_nic_out_0_nic_in_1
   + flow_source_nic_in_1 - flow_nic_in_1_1 + flow_nic_out_2_nic_in_1
   + flow_nic_out_3_nic_in_1 + flow_nic_out_4_nic_in_1 = 0
 node_flow_constraint_nic_out_1: - flow_nic_out_1_sink
   - flow_nic_out_1_nic_in_0 - flow_nic_out_1_nic_in_2
   - flow_nic_out_1_nic_in_3 - flow_nic_out_1_nic_in_4 + flow_1_nic_out_1
   = 0
 node_flow_constraint_nic_in_2: flow_nic_out_0_nic_in_2
   + flow_nic_out_1_nic_in_2 + flow_source_nic_in_2 - flow_nic_in_2_2
   + flow_nic_out_3_nic_in_2 + flow_nic_out_4_nic_in_2 = 0
 node_flow_constraint_nic_out_2: - flow_nic_out_2_sink
   - flow_nic_out_2_nic_in_0 - flow_nic_out_2_nic_in_1
   - flow_nic_out_2_nic_in_3 - flow_nic_out_2_nic_in_4 + flow_2_nic_out_2
   = 0
 node_flow_constraint_nic_in_3: flow_nic_out_0_nic_in_3
   + flow_nic_out_1_nic_in_3 + flow_nic_out_2_nic_in_3
   + flow_source_nic_in_3 - flow_nic_in_3_3 + flow_nic_out_4_nic_in_3 = 0
 node_flow_constraint_nic_out_3: - flow_nic_out_3_sink
   - flow_nic_out_3_nic_in_0 - flow_nic_out_3_nic_in_1
   - flow_nic_out_3_nic_in_2 - flow_nic_out_3_nic_in_4 + flow_3_nic_out_3
   = 0
 node_flow_constraint_nic_in_4: flow_nic_out_0_nic_in_4
   + flow_nic_out_1_nic_in_4 + flow_nic_out_2_nic_in_4
   + flow_nic_out_3_nic_in_4 + flow_source_nic_in_4 - flow_nic_in_4_4
   - flow_nic_in_4_5 = 0
 node_flow_constraint_nic_out_4: - flow_nic_out_4_sink
   - flow_nic_out_4_nic_in_0 - flow_nic_out_4_nic_in_1
   - flow_nic_out_4_nic_in_2 - flow_nic_out_4_nic_in_3 + flow_4_nic_out_4
   + flow_5_nic_out_4 = 0
 node_flow_constraint_0: flow_nic_in_0_0 - flow_0_nic_out_0 = 0
 node_flow_constraint_1: flow_nic_in_1_1 - flow_1_nic_out_1 = 0
 node_flow_constraint_2: flow_nic_in_2_2 - flow_2_nic_out_2 = 0
 node_flow_constraint_3: flow_nic_in_3_3 - flow_3_nic_out_3 = 0
 node_flow_constraint_4: flow_nic_in_4_4 - flow_4_nic_out_4 - flow_4_5
   + flow_5_4 = 0
 node_flow_constraint_5: flow_nic_in_4_5 - flow_5_nic_out_4 + flow_4_5
   - flow_5_4 = 0
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
 force_group_off_constr_0: group_tp_active_0 = 0
 force_group_off_constr_1: group_tp_active_1 = 0
 force_group_off_constr_2: group_tp_active_2 = 0
 force_group_off_constr_3: group_tp_active_3 = 0
 R172: - group_tp_active_4 + tp_on_and_switch_nic_out_0_nic_in_4_4 <= 0
 R173: - switch_nic_out_0_nic_in_4 + tp_on_and_switch_nic_out_0_nic_in_4_4
   <= 0
 R174: - switch_nic_out_0_nic_in_4 - group_tp_active_4
   + tp_on_and_switch_nic_out_0_nic_in_4_4 >= -1
 R175: - group_tp_active_4 + tp_on_and_switch_nic_out_1_nic_in_4_4 <= 0
 R176: - switch_nic_out_1_nic_in_4 + tp_on_and_switch_nic_out_1_nic_in_4_4
   <= 0
 R177: - switch_nic_out_1_nic_in_4 - group_tp_active_4
   + tp_on_and_switch_nic_out_1_nic_in_4_4 >= -1
 R178: - group_tp_active_4 + tp_on_and_switch_nic_out_2_nic_in_4_4 <= 0
 R179: - switch_nic_out_2_nic_in_4 + tp_on_and_switch_nic_out_2_nic_in_4_4
   <= 0
 R180: - switch_nic_out_2_nic_in_4 - group_tp_active_4
   + tp_on_and_switch_nic_out_2_nic_in_4_4 >= -1
 R181: - group_tp_active_4 + tp_on_and_switch_nic_out_3_nic_in_4_4 <= 0
 R182: - switch_nic_out_3_nic_in_4 + tp_on_and_switch_nic_out_3_nic_in_4_4
   <= 0
 R183: - switch_nic_out_3_nic_in_4 - group_tp_active_4
   + tp_on_and_switch_nic_out_3_nic_in_4_4 >= -1
 R184: - group_tp_active_4 + tp_on_and_switch_source_nic_in_4_4 <= 0
 R185: - switch_source_nic_in_4 + tp_on_and_switch_source_nic_in_4_4 <= 0
 R186: - switch_source_nic_in_4 - group_tp_active_4
   + tp_on_and_switch_source_nic_in_4_4 >= -1
 force_min_throughput: flow_source_nic_in_0 + flow_source_nic_in_1
   + flow_source_nic_in_2 + flow_source_nic_in_3 + flow_source_nic_in_4
   >= 1
 inactive_start_invalid_0: is_active_0 = 0 -> start_0 = 80
 inactive_start_invalid_1: is_active_1 = 0 -> start_1 = 80
 inactive_start_invalid_2: is_active_2 = 0 -> start_2 = 80
 inactive_start_invalid_3: is_active_3 = 0 -> start_3 = 80
 inactive_start_invalid_4: is_active_4 = 0 -> start_4 = 80
 inactive_start_invalid_5: is_active_5 = 0 -> start_5 = 80
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
 hold_5_24 hold_5_25 hold_5_26 switch_source_nic_in_0 switch_nic_out_0_sink
 switch_nic_out_0_nic_in_1 switch_nic_out_0_nic_in_2
 switch_nic_out_0_nic_in_3 switch_nic_out_0_nic_in_4 switch_nic_in_0_0
 switch_0_nic_out_0 switch_source_nic_in_1 switch_nic_out_1_sink
 switch_nic_out_1_nic_in_0 switch_nic_out_1_nic_in_2
 switch_nic_out_1_nic_in_3 switch_nic_out_1_nic_in_4 switch_nic_in_1_1
 switch_1_nic_out_1 switch_source_nic_in_2 switch_nic_out_2_sink
 switch_nic_out_2_nic_in_0 switch_nic_out_2_nic_in_1
 switch_nic_out_2_nic_in_3 switch_nic_out_2_nic_in_4 switch_nic_in_2_2
 switch_2_nic_out_2 switch_source_nic_in_3 switch_nic_out_3_sink
 switch_nic_out_3_nic_in_0 switch_nic_out_3_nic_in_1
 switch_nic_out_3_nic_in_2 switch_nic_out_3_nic_in_4 switch_nic_in_3_3
 switch_3_nic_out_3 switch_source_nic_in_4 switch_nic_out_4_sink
 switch_nic_out_4_nic_in_0 switch_nic_out_4_nic_in_1
 switch_nic_out_4_nic_in_2 switch_nic_out_4_nic_in_3 switch_nic_in_4_4
 switch_4_nic_out_4 switch_nic_in_4_5 switch_5_nic_out_4 switch_4_5
 switch_5_4 group_tp_active_0 group_tp_active_1 group_tp_active_2
 group_tp_active_3 group_tp_active_4 is_active_0 is_active_1 is_active_2
 is_active_3 is_active_4 is_active_5 is_start_zero_0_for_source_nic_in_0
 is_end_complete_0 is_start_zero_1_for_source_nic_in_1 is_end_complete_1
 is_start_zero_2_for_source_nic_in_2 is_end_complete_2
 is_start_zero_3_for_source_nic_in_3 is_end_complete_3
 is_start_zero_4_for_source_nic_in_4 is_start_zero_5_for_source_nic_in_4
 is_end_complete_4 is_end_complete_5 pp_permission_4_5 pp_permission_5_4
 tp_on_and_switch_nic_out_0_nic_in_4_4
 tp_on_and_switch_nic_out_1_nic_in_4_4
 tp_on_and_switch_nic_out_2_nic_in_4_4
 tp_on_and_switch_nic_out_3_nic_in_4_4 tp_on_and_switch_source_nic_in_4_4
Generals
 start_0 start_1 start_2 start_3 start_4 start_5 group_start_1 val_end_0
 group_end_0 group_start_2 group_start_3 group_start_4 group_start_0
 val_end_1 group_end_1 val_end_2 group_end_2 val_end_3 group_end_3
 val_end_4 val_end_5 group_end_4
General Constraints
 min_start_1: group_start_1 = MIN ( start_1 )
 max_end_0: group_end_0 = MAX ( val_end_0 )
 min_start_2: group_start_2 = MIN ( start_2 )
 min_start_3: group_start_3 = MIN ( start_3 )
 min_start_4: group_start_4 = MIN ( start_4 , start_5 )
 min_start_0: group_start_0 = MIN ( start_0 )
 max_end_1: group_end_1 = MAX ( val_end_1 )
 max_end_2: group_end_2 = MAX ( val_end_2 )
 max_end_3: group_end_3 = MAX ( val_end_3 )
 max_end_4: group_end_4 = MAX ( val_end_4 , val_end_5 )
End
