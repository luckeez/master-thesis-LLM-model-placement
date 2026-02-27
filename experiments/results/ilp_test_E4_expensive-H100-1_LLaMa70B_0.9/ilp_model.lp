\ Model 2026-02-27_11-02-58
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  0.00117537313432836 flow_source_nic_in_0
   + 0.00117537313432836 flow_source_nic_in_1
   + 0.00117537313432836 flow_source_nic_in_2
   + 0.00117537313432836 flow_source_nic_in_3 - 0.025 is_active_0
   - 0.025 is_active_1 - 0.025 is_active_2 - 0.025 is_active_3
   + 0 val_end_0 + 0 val_end_1 + 0 val_end_2 + 0 val_end_3
Subject To
 mem_active_constr_0: hold_0_1 + hold_0_2 + hold_0_3 + hold_0_4 + hold_0_5
   + hold_0_6 + hold_0_7 + hold_0_8 + hold_0_9 + hold_0_10 + hold_0_11
   + hold_0_12 + hold_0_13 + hold_0_14 + hold_0_15 + hold_0_16 + hold_0_17
   + hold_0_18 + hold_0_19 + hold_0_20 + hold_0_21 + hold_0_22
   - 81 is_active_0 <= 0
 mem_active_constr2_0: hold_0_1 + hold_0_2 + hold_0_3 + hold_0_4 + hold_0_5
   + hold_0_6 + hold_0_7 + hold_0_8 + hold_0_9 + hold_0_10 + hold_0_11
   + hold_0_12 + hold_0_13 + hold_0_14 + hold_0_15 + hold_0_16 + hold_0_17
   + hold_0_18 + hold_0_19 + hold_0_20 + hold_0_21 + hold_0_22
   - is_active_0 >= 0
 mem_active_constr_1: hold_1_1 + hold_1_2 + hold_1_3 + hold_1_4 + hold_1_5
   + hold_1_6 + hold_1_7 + hold_1_8 + hold_1_9 + hold_1_10 + hold_1_11
   + hold_1_12 + hold_1_13 + hold_1_14 + hold_1_15 + hold_1_16 + hold_1_17
   + hold_1_18 + hold_1_19 + hold_1_20 + hold_1_21 + hold_1_22
   - 81 is_active_1 <= 0
 mem_active_constr2_1: hold_1_1 + hold_1_2 + hold_1_3 + hold_1_4 + hold_1_5
   + hold_1_6 + hold_1_7 + hold_1_8 + hold_1_9 + hold_1_10 + hold_1_11
   + hold_1_12 + hold_1_13 + hold_1_14 + hold_1_15 + hold_1_16 + hold_1_17
   + hold_1_18 + hold_1_19 + hold_1_20 + hold_1_21 + hold_1_22
   - is_active_1 >= 0
 mem_active_constr_2: hold_2_1 + hold_2_2 + hold_2_3 + hold_2_4 + hold_2_5
   + hold_2_6 + hold_2_7 + hold_2_8 + hold_2_9 + hold_2_10 + hold_2_11
   + hold_2_12 + hold_2_13 + hold_2_14 + hold_2_15 + hold_2_16 + hold_2_17
   + hold_2_18 + hold_2_19 + hold_2_20 + hold_2_21 + hold_2_22
   - 81 is_active_2 <= 0
 mem_active_constr2_2: hold_2_1 + hold_2_2 + hold_2_3 + hold_2_4 + hold_2_5
   + hold_2_6 + hold_2_7 + hold_2_8 + hold_2_9 + hold_2_10 + hold_2_11
   + hold_2_12 + hold_2_13 + hold_2_14 + hold_2_15 + hold_2_16 + hold_2_17
   + hold_2_18 + hold_2_19 + hold_2_20 + hold_2_21 + hold_2_22
   - is_active_2 >= 0
 mem_active_constr_3: hold_3_1 + hold_3_2 + hold_3_3 + hold_3_4 + hold_3_5
   + hold_3_6 + hold_3_7 + hold_3_8 + hold_3_9 + hold_3_10 + hold_3_11
   + hold_3_12 + hold_3_13 + hold_3_14 + hold_3_15 + hold_3_16 + hold_3_17
   + hold_3_18 + hold_3_19 + hold_3_20 + hold_3_21 + hold_3_22
   - 81 is_active_3 <= 0
 mem_active_constr2_3: hold_3_1 + hold_3_2 + hold_3_3 + hold_3_4 + hold_3_5
   + hold_3_6 + hold_3_7 + hold_3_8 + hold_3_9 + hold_3_10 + hold_3_11
   + hold_3_12 + hold_3_13 + hold_3_14 + hold_3_15 + hold_3_16 + hold_3_17
   + hold_3_18 + hold_3_19 + hold_3_20 + hold_3_21 + hold_3_22
   - is_active_3 >= 0
 hold_constraint_0: hold_0_1 + hold_0_2 + hold_0_3 + hold_0_4 + hold_0_5
   + hold_0_6 + hold_0_7 + hold_0_8 + hold_0_9 + hold_0_10 + hold_0_11
   + hold_0_12 + hold_0_13 + hold_0_14 + hold_0_15 + hold_0_16 + hold_0_17
   + hold_0_18 + hold_0_19 + hold_0_20 + hold_0_21 + hold_0_22 <= 1
 end_constraint_0: start_0 + hold_0_1 + 2 hold_0_2 + 3 hold_0_3
   + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7 + 8 hold_0_8
   + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12 + 13 hold_0_13
   + 14 hold_0_14 + 15 hold_0_15 + 16 hold_0_16 + 17 hold_0_17
   + 18 hold_0_18 + 19 hold_0_19 + 20 hold_0_20 + 21 hold_0_21
   + 22 hold_0_22 <= 80
 hold_constraint_1: hold_1_1 + hold_1_2 + hold_1_3 + hold_1_4 + hold_1_5
   + hold_1_6 + hold_1_7 + hold_1_8 + hold_1_9 + hold_1_10 + hold_1_11
   + hold_1_12 + hold_1_13 + hold_1_14 + hold_1_15 + hold_1_16 + hold_1_17
   + hold_1_18 + hold_1_19 + hold_1_20 + hold_1_21 + hold_1_22 <= 1
 end_constraint_1: start_1 + hold_1_1 + 2 hold_1_2 + 3 hold_1_3
   + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7 + 8 hold_1_8
   + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 + 12 hold_1_12 + 13 hold_1_13
   + 14 hold_1_14 + 15 hold_1_15 + 16 hold_1_16 + 17 hold_1_17
   + 18 hold_1_18 + 19 hold_1_19 + 20 hold_1_20 + 21 hold_1_21
   + 22 hold_1_22 <= 80
 hold_constraint_2: hold_2_1 + hold_2_2 + hold_2_3 + hold_2_4 + hold_2_5
   + hold_2_6 + hold_2_7 + hold_2_8 + hold_2_9 + hold_2_10 + hold_2_11
   + hold_2_12 + hold_2_13 + hold_2_14 + hold_2_15 + hold_2_16 + hold_2_17
   + hold_2_18 + hold_2_19 + hold_2_20 + hold_2_21 + hold_2_22 <= 1
 end_constraint_2: start_2 + hold_2_1 + 2 hold_2_2 + 3 hold_2_3
   + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7 + 8 hold_2_8
   + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 12 hold_2_12 + 13 hold_2_13
   + 14 hold_2_14 + 15 hold_2_15 + 16 hold_2_16 + 17 hold_2_17
   + 18 hold_2_18 + 19 hold_2_19 + 20 hold_2_20 + 21 hold_2_21
   + 22 hold_2_22 <= 80
 hold_constraint_3: hold_3_1 + hold_3_2 + hold_3_3 + hold_3_4 + hold_3_5
   + hold_3_6 + hold_3_7 + hold_3_8 + hold_3_9 + hold_3_10 + hold_3_11
   + hold_3_12 + hold_3_13 + hold_3_14 + hold_3_15 + hold_3_16 + hold_3_17
   + hold_3_18 + hold_3_19 + hold_3_20 + hold_3_21 + hold_3_22 <= 1
 end_constraint_3: start_3 + hold_3_1 + 2 hold_3_2 + 3 hold_3_3
   + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7 + 8 hold_3_8
   + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11 + 12 hold_3_12 + 13 hold_3_13
   + 14 hold_3_14 + 15 hold_3_15 + 16 hold_3_16 + 17 hold_3_17
   + 18 hold_3_18 + 19 hold_3_19 + 20 hold_3_20 + 21 hold_3_21
   + 22 hold_3_22 <= 80
 node_flow_constraint_nic_in_0: flow_source_nic_in_0 - flow_nic_in_0_0
   + flow_nic_out_1_nic_in_0 + flow_nic_out_2_nic_in_0
   + flow_nic_out_3_nic_in_0 = 0
 node_flow_constraint_nic_out_0: - flow_nic_out_0_sink
   - flow_nic_out_0_nic_in_1 - flow_nic_out_0_nic_in_2
   - flow_nic_out_0_nic_in_3 + flow_0_nic_out_0 = 0
 node_flow_constraint_nic_in_1: flow_nic_out_0_nic_in_1
   + flow_source_nic_in_1 - flow_nic_in_1_1 + flow_nic_out_2_nic_in_1
   + flow_nic_out_3_nic_in_1 = 0
 node_flow_constraint_nic_out_1: - flow_nic_out_1_sink
   - flow_nic_out_1_nic_in_0 - flow_nic_out_1_nic_in_2
   - flow_nic_out_1_nic_in_3 + flow_1_nic_out_1 = 0
 node_flow_constraint_nic_in_2: flow_nic_out_0_nic_in_2
   + flow_nic_out_1_nic_in_2 + flow_source_nic_in_2 - flow_nic_in_2_2
   + flow_nic_out_3_nic_in_2 = 0
 node_flow_constraint_nic_out_2: - flow_nic_out_2_sink
   - flow_nic_out_2_nic_in_0 - flow_nic_out_2_nic_in_1
   - flow_nic_out_2_nic_in_3 + flow_2_nic_out_2 = 0
 node_flow_constraint_nic_in_3: flow_nic_out_0_nic_in_3
   + flow_nic_out_1_nic_in_3 + flow_nic_out_2_nic_in_3
   + flow_source_nic_in_3 - flow_nic_in_3_3 = 0
 node_flow_constraint_nic_out_3: - flow_nic_out_3_sink
   - flow_nic_out_3_nic_in_0 - flow_nic_out_3_nic_in_1
   - flow_nic_out_3_nic_in_2 + flow_3_nic_out_3 = 0
 node_flow_constraint_0: flow_nic_in_0_0 - flow_0_nic_out_0 = 0
 node_flow_constraint_1: flow_nic_in_1_1 - flow_1_nic_out_1 = 0
 node_flow_constraint_2: flow_nic_in_2_2 - flow_2_nic_out_2 = 0
 node_flow_constraint_3: flow_nic_in_3_3 - flow_3_nic_out_3 = 0
 constr_is_start_zero_0: start_0 + 80 is_start_zero_0_for_source_nic_in_0
   <= 80
 constr_edge_switch_start_zero_0_OR: switch_source_nic_in_0
   - is_start_zero_0_for_source_nic_in_0 <= 0
 constr_is_end_complete_0: start_0 + hold_0_1 + 2 hold_0_2 + 3 hold_0_3
   + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7 + 8 hold_0_8
   + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12 + 13 hold_0_13
   + 14 hold_0_14 + 15 hold_0_15 + 16 hold_0_16 + 17 hold_0_17
   + 18 hold_0_18 + 19 hold_0_19 + 20 hold_0_20 + 21 hold_0_21
   + 22 hold_0_22 - 80 is_end_complete_0 >= 0
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
 constr_is_start_zero_1: start_1 + 80 is_start_zero_1_for_source_nic_in_1
   <= 80
 constr_edge_switch_start_zero_1_OR: switch_source_nic_in_1
   - is_start_zero_1_for_source_nic_in_1 <= 0
 constr_is_end_complete_1: start_1 + hold_1_1 + 2 hold_1_2 + 3 hold_1_3
   + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7 + 8 hold_1_8
   + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 + 12 hold_1_12 + 13 hold_1_13
   + 14 hold_1_14 + 15 hold_1_15 + 16 hold_1_16 + 17 hold_1_17
   + 18 hold_1_18 + 19 hold_1_19 + 20 hold_1_20 + 21 hold_1_21
   + 22 hold_1_22 - 80 is_end_complete_1 >= 0
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
 constr_is_start_zero_2: start_2 + 80 is_start_zero_2_for_source_nic_in_2
   <= 80
 constr_edge_switch_start_zero_2_OR: switch_source_nic_in_2
   - is_start_zero_2_for_source_nic_in_2 <= 0
 constr_is_end_complete_2: start_2 + hold_2_1 + 2 hold_2_2 + 3 hold_2_3
   + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7 + 8 hold_2_8
   + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 12 hold_2_12 + 13 hold_2_13
   + 14 hold_2_14 + 15 hold_2_15 + 16 hold_2_16 + 17 hold_2_17
   + 18 hold_2_18 + 19 hold_2_19 + 20 hold_2_20 + 21 hold_2_21
   + 22 hold_2_22 - 80 is_end_complete_2 >= 0
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
 constr_is_start_zero_3: start_3 + 80 is_start_zero_3_for_source_nic_in_3
   <= 80
 constr_edge_switch_start_zero_3_OR: switch_source_nic_in_3
   - is_start_zero_3_for_source_nic_in_3 <= 0
 constr_is_end_complete_3: start_3 + hold_3_1 + 2 hold_3_2 + 3 hold_3_3
   + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7 + 8 hold_3_8
   + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11 + 12 hold_3_12 + 13 hold_3_13
   + 14 hold_3_14 + 15 hold_3_15 + 16 hold_3_16 + 17 hold_3_17
   + 18 hold_3_18 + 19 hold_3_19 + 20 hold_3_20 + 21 hold_3_21
   + 22 hold_3_22 - 80 is_end_complete_3 >= 0
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
 link_activation_dst_nic_in_0_0: switch_nic_in_0_0 - is_active_0 <= 0
 link_activation_src_0_nic_out_0: switch_0_nic_out_0 - is_active_0 <= 0
 link_activation_dst_nic_in_1_1: switch_nic_in_1_1 - is_active_1 <= 0
 link_activation_src_1_nic_out_1: switch_1_nic_out_1 - is_active_1 <= 0
 link_activation_dst_nic_in_2_2: switch_nic_in_2_2 - is_active_2 <= 0
 link_activation_src_2_nic_out_2: switch_2_nic_out_2 - is_active_2 <= 0
 link_activation_dst_nic_in_3_3: switch_nic_in_3_3 - is_active_3 <= 0
 link_activation_src_3_nic_out_3: switch_3_nic_out_3 - is_active_3 <= 0
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
 edge_flow_constr_nic_in_3_3: flow_nic_in_3_3
   - 3.90625e+06 switch_nic_in_3_3 <= 0
 edge_flow_constr_3_nic_out_3: flow_3_nic_out_3
   - 3.90625e+06 switch_3_nic_out_3 <= 0
 force_group_off_constr_0: group_tp_active_0 = 0
 force_group_off_constr_1: group_tp_active_1 = 0
 force_group_off_constr_2: group_tp_active_2 = 0
 force_group_off_constr_3: group_tp_active_3 = 0
 force_min_throughput: flow_source_nic_in_0 + flow_source_nic_in_1
   + flow_source_nic_in_2 + flow_source_nic_in_3 >= 1
 inactive_start_invalid_0: is_active_0 = 0 -> start_0 = 80
 inactive_start_invalid_1: is_active_1 = 0 -> start_1 = 80
 inactive_start_invalid_2: is_active_2 = 0 -> start_2 = 80
 inactive_start_invalid_3: is_active_3 = 0 -> start_3 = 80
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
   - 15314.28571428572 hold_1_1 - 7657.142857142858 hold_1_2
   - 5104.761904761905 hold_1_3 - 3828.571428571429 hold_1_4
   - 3062.857142857143 hold_1_5 - 2552.380952380952 hold_1_6
   - 2187.755102040816 hold_1_7 - 1914.285714285714 hold_1_8
   - 1701.587301587301 hold_1_9 - 1531.428571428571 hold_1_10
   - 1392.207792207792 hold_1_11 - 1276.190476190476 hold_1_12
   - 1178.021978021978 hold_1_13 - 1093.877551020408 hold_1_14
   - 1020.952380952381 hold_1_15 - 957.1428571428572 hold_1_16
   - 900.8403361344537 hold_1_17 - 850.7936507936507 hold_1_18
   - 806.015037593985 hold_1_19 - 765.7142857142857 hold_1_20
   - 729.2517006802721 hold_1_21 - 696.1038961038961 hold_1_22
   + flow_nic_in_1_1 <= 0
 node_throughput_constraint_2: group_tp_active_2 = 0 ->
   - 15314.28571428572 hold_2_1 - 7657.142857142858 hold_2_2
   - 5104.761904761905 hold_2_3 - 3828.571428571429 hold_2_4
   - 3062.857142857143 hold_2_5 - 2552.380952380952 hold_2_6
   - 2187.755102040816 hold_2_7 - 1914.285714285714 hold_2_8
   - 1701.587301587301 hold_2_9 - 1531.428571428571 hold_2_10
   - 1392.207792207792 hold_2_11 - 1276.190476190476 hold_2_12
   - 1178.021978021978 hold_2_13 - 1093.877551020408 hold_2_14
   - 1020.952380952381 hold_2_15 - 957.1428571428572 hold_2_16
   - 900.8403361344537 hold_2_17 - 850.7936507936507 hold_2_18
   - 806.015037593985 hold_2_19 - 765.7142857142857 hold_2_20
   - 729.2517006802721 hold_2_21 - 696.1038961038961 hold_2_22
   + flow_nic_in_2_2 <= 0
 node_throughput_constraint_3: group_tp_active_3 = 0 ->
   - 15314.28571428572 hold_3_1 - 7657.142857142858 hold_3_2
   - 5104.761904761905 hold_3_3 - 3828.571428571429 hold_3_4
   - 3062.857142857143 hold_3_5 - 2552.380952380952 hold_3_6
   - 2187.755102040816 hold_3_7 - 1914.285714285714 hold_3_8
   - 1701.587301587301 hold_3_9 - 1531.428571428571 hold_3_10
   - 1392.207792207792 hold_3_11 - 1276.190476190476 hold_3_12
   - 1178.021978021978 hold_3_13 - 1093.877551020408 hold_3_14
   - 1020.952380952381 hold_3_15 - 957.1428571428572 hold_3_16
   - 900.8403361344537 hold_3_17 - 850.7936507936507 hold_3_18
   - 806.015037593985 hold_3_19 - 765.7142857142857 hold_3_20
   - 729.2517006802721 hold_3_21 - 696.1038961038961 hold_3_22
   + flow_nic_in_3_3 <= 0
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
   - 7824.817518248176 hold_1_1 - 5178.743961352658 hold_1_2
   - 3870.036101083032 hold_1_3 - 3089.337175792507 hold_1_4
   - 2570.743405275779 hold_1_5 - 2201.23203285421 hold_1_6
   - 1924.5960502693 hold_1_7 - 1709.728867623605 hold_1_8
   - 1538.020086083214 hold_1_9 - 1397.653194263364 hold_1_10
   - 1280.764635603345 hold_1_11 - 1181.918412348401 hold_1_12
   - 1097.236438075742 hold_1_13 - 1023.877745940783 hold_1_14
   - 959.7135183527306 hold_1_15 - 903.117101937658 hold_1_16
   - 852.8241845664279 hold_1_17 - 807.8372268274302 hold_1_18
   - 767.3586256263421 hold_1_19 - 730.7430129516018 hold_1_20
   - 697.4625894599869 hold_1_21 - 667.0815183571873 hold_1_22
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_0_nic_in_2: switch_nic_out_0_nic_in_2 = 1 ->
   - 7824.817518248176 hold_2_1 - 5178.743961352658 hold_2_2
   - 3870.036101083032 hold_2_3 - 3089.337175792507 hold_2_4
   - 2570.743405275779 hold_2_5 - 2201.23203285421 hold_2_6
   - 1924.5960502693 hold_2_7 - 1709.728867623605 hold_2_8
   - 1538.020086083214 hold_2_9 - 1397.653194263364 hold_2_10
   - 1280.764635603345 hold_2_11 - 1181.918412348401 hold_2_12
   - 1097.236438075742 hold_2_13 - 1023.877745940783 hold_2_14
   - 959.7135183527306 hold_2_15 - 903.117101937658 hold_2_16
   - 852.8241845664279 hold_2_17 - 807.8372268274302 hold_2_18
   - 767.3586256263421 hold_2_19 - 730.7430129516018 hold_2_20
   - 697.4625894599869 hold_2_21 - 667.0815183571873 hold_2_22
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_0_nic_in_3: switch_nic_out_0_nic_in_3 = 1 ->
   - 7824.817518248176 hold_3_1 - 5178.743961352658 hold_3_2
   - 3870.036101083032 hold_3_3 - 3089.337175792507 hold_3_4
   - 2570.743405275779 hold_3_5 - 2201.23203285421 hold_3_6
   - 1924.5960502693 hold_3_7 - 1709.728867623605 hold_3_8
   - 1538.020086083214 hold_3_9 - 1397.653194263364 hold_3_10
   - 1280.764635603345 hold_3_11 - 1181.918412348401 hold_3_12
   - 1097.236438075742 hold_3_13 - 1023.877745940783 hold_3_14
   - 959.7135183527306 hold_3_15 - 903.117101937658 hold_3_16
   - 852.8241845664279 hold_3_17 - 807.8372268274302 hold_3_18
   - 767.3586256263421 hold_3_19 - 730.7430129516018 hold_3_20
   - 697.4625894599869 hold_3_21 - 667.0815183571873 hold_3_22
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_1_source_nic_in_1: switch_source_nic_in_1 = 1 ->
   - 12853.7170263789 hold_1_1 - 6988.265971316819 hold_1_2
   - 4798.567591763653 hold_1_3 - 3653.71506475801 hold_1_4
   - 2949.917446340121 hold_1_5 - 2473.465620673743 hold_1_6
   - 2129.519268970997 hold_1_7 - 1869.550052319498 hold_1_8
   - 1666.148585638794 hold_1_9 - 1502.663302495094 hold_1_10
   - 1368.39417921879 hold_1_11 - 1256.151863135693 hold_1_12
   - 1160.927008880225 hold_1_13 - 1079.122206563318 hold_1_14
   - 1008.087267255972 hold_1_15 - 945.8267160755252 hold_1_16
   - 890.8093734419145 hold_1_17 - 841.8407413224438 hold_1_18
   - 797.9752865862737 hold_1_19 - 758.4547898684025 hold_1_20
   - 722.664149925846 hold_1_21 - 690.0991373760783 hold_1_22
   + flow_nic_in_1_1 <= 0
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
   - 7824.817518248176 hold_2_1 - 5178.743961352658 hold_2_2
   - 3870.036101083032 hold_2_3 - 3089.337175792507 hold_2_4
   - 2570.743405275779 hold_2_5 - 2201.23203285421 hold_2_6
   - 1924.5960502693 hold_2_7 - 1709.728867623605 hold_2_8
   - 1538.020086083214 hold_2_9 - 1397.653194263364 hold_2_10
   - 1280.764635603345 hold_2_11 - 1181.918412348401 hold_2_12
   - 1097.236438075742 hold_2_13 - 1023.877745940783 hold_2_14
   - 959.7135183527306 hold_2_15 - 903.117101937658 hold_2_16
   - 852.8241845664279 hold_2_17 - 807.8372268274302 hold_2_18
   - 767.3586256263421 hold_2_19 - 730.7430129516018 hold_2_20
   - 697.4625894599869 hold_2_21 - 667.0815183571873 hold_2_22
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_1_nic_in_3: switch_nic_out_1_nic_in_3 = 1 ->
   - 7824.817518248176 hold_3_1 - 5178.743961352658 hold_3_2
   - 3870.036101083032 hold_3_3 - 3089.337175792507 hold_3_4
   - 2570.743405275779 hold_3_5 - 2201.23203285421 hold_3_6
   - 1924.5960502693 hold_3_7 - 1709.728867623605 hold_3_8
   - 1538.020086083214 hold_3_9 - 1397.653194263364 hold_3_10
   - 1280.764635603345 hold_3_11 - 1181.918412348401 hold_3_12
   - 1097.236438075742 hold_3_13 - 1023.877745940783 hold_3_14
   - 959.7135183527306 hold_3_15 - 903.117101937658 hold_3_16
   - 852.8241845664279 hold_3_17 - 807.8372268274302 hold_3_18
   - 767.3586256263421 hold_3_19 - 730.7430129516018 hold_3_20
   - 697.4625894599869 hold_3_21 - 667.0815183571873 hold_3_22
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_2_source_nic_in_2: switch_source_nic_in_2 = 1 ->
   - 12853.7170263789 hold_2_1 - 6988.265971316819 hold_2_2
   - 4798.567591763653 hold_2_3 - 3653.71506475801 hold_2_4
   - 2949.917446340121 hold_2_5 - 2473.465620673743 hold_2_6
   - 2129.519268970997 hold_2_7 - 1869.550052319498 hold_2_8
   - 1666.148585638794 hold_2_9 - 1502.663302495094 hold_2_10
   - 1368.39417921879 hold_2_11 - 1256.151863135693 hold_2_12
   - 1160.927008880225 hold_2_13 - 1079.122206563318 hold_2_14
   - 1008.087267255972 hold_2_15 - 945.8267160755252 hold_2_16
   - 890.8093734419145 hold_2_17 - 841.8407413224438 hold_2_18
   - 797.9752865862737 hold_2_19 - 758.4547898684025 hold_2_20
   - 722.664149925846 hold_2_21 - 690.0991373760783 hold_2_22
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
   - 7824.817518248176 hold_1_1 - 5178.743961352658 hold_1_2
   - 3870.036101083032 hold_1_3 - 3089.337175792507 hold_1_4
   - 2570.743405275779 hold_1_5 - 2201.23203285421 hold_1_6
   - 1924.5960502693 hold_1_7 - 1709.728867623605 hold_1_8
   - 1538.020086083214 hold_1_9 - 1397.653194263364 hold_1_10
   - 1280.764635603345 hold_1_11 - 1181.918412348401 hold_1_12
   - 1097.236438075742 hold_1_13 - 1023.877745940783 hold_1_14
   - 959.7135183527306 hold_1_15 - 903.117101937658 hold_1_16
   - 852.8241845664279 hold_1_17 - 807.8372268274302 hold_1_18
   - 767.3586256263421 hold_1_19 - 730.7430129516018 hold_1_20
   - 697.4625894599869 hold_1_21 - 667.0815183571873 hold_1_22
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_3_nic_out_2_nic_in_3: switch_nic_out_2_nic_in_3 = 1 ->
   - 7824.817518248176 hold_3_1 - 5178.743961352658 hold_3_2
   - 3870.036101083032 hold_3_3 - 3089.337175792507 hold_3_4
   - 2570.743405275779 hold_3_5 - 2201.23203285421 hold_3_6
   - 1924.5960502693 hold_3_7 - 1709.728867623605 hold_3_8
   - 1538.020086083214 hold_3_9 - 1397.653194263364 hold_3_10
   - 1280.764635603345 hold_3_11 - 1181.918412348401 hold_3_12
   - 1097.236438075742 hold_3_13 - 1023.877745940783 hold_3_14
   - 959.7135183527306 hold_3_15 - 903.117101937658 hold_3_16
   - 852.8241845664279 hold_3_17 - 807.8372268274302 hold_3_18
   - 767.3586256263421 hold_3_19 - 730.7430129516018 hold_3_20
   - 697.4625894599869 hold_3_21 - 667.0815183571873 hold_3_22
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_3_source_nic_in_3: switch_source_nic_in_3 = 1 ->
   - 12853.7170263789 hold_3_1 - 6988.265971316819 hold_3_2
   - 4798.567591763653 hold_3_3 - 3653.71506475801 hold_3_4
   - 2949.917446340121 hold_3_5 - 2473.465620673743 hold_3_6
   - 2129.519268970997 hold_3_7 - 1869.550052319498 hold_3_8
   - 1666.148585638794 hold_3_9 - 1502.663302495094 hold_3_10
   - 1368.39417921879 hold_3_11 - 1256.151863135693 hold_3_12
   - 1160.927008880225 hold_3_13 - 1079.122206563318 hold_3_14
   - 1008.087267255972 hold_3_15 - 945.8267160755252 hold_3_16
   - 890.8093734419145 hold_3_17 - 841.8407413224438 hold_3_18
   - 797.9752865862737 hold_3_19 - 758.4547898684025 hold_3_20
   - 722.664149925846 hold_3_21 - 690.0991373760783 hold_3_22
   + flow_nic_in_3_3 <= 0
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
   - 7824.817518248176 hold_1_1 - 5178.743961352658 hold_1_2
   - 3870.036101083032 hold_1_3 - 3089.337175792507 hold_1_4
   - 2570.743405275779 hold_1_5 - 2201.23203285421 hold_1_6
   - 1924.5960502693 hold_1_7 - 1709.728867623605 hold_1_8
   - 1538.020086083214 hold_1_9 - 1397.653194263364 hold_1_10
   - 1280.764635603345 hold_1_11 - 1181.918412348401 hold_1_12
   - 1097.236438075742 hold_1_13 - 1023.877745940783 hold_1_14
   - 959.7135183527306 hold_1_15 - 903.117101937658 hold_1_16
   - 852.8241845664279 hold_1_17 - 807.8372268274302 hold_1_18
   - 767.3586256263421 hold_1_19 - 730.7430129516018 hold_1_20
   - 697.4625894599869 hold_1_21 - 667.0815183571873 hold_1_22
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_3_nic_in_2: switch_nic_out_3_nic_in_2 = 1 ->
   - 7824.817518248176 hold_2_1 - 5178.743961352658 hold_2_2
   - 3870.036101083032 hold_2_3 - 3089.337175792507 hold_2_4
   - 2570.743405275779 hold_2_5 - 2201.23203285421 hold_2_6
   - 1924.5960502693 hold_2_7 - 1709.728867623605 hold_2_8
   - 1538.020086083214 hold_2_9 - 1397.653194263364 hold_2_10
   - 1280.764635603345 hold_2_11 - 1181.918412348401 hold_2_12
   - 1097.236438075742 hold_2_13 - 1023.877745940783 hold_2_14
   - 959.7135183527306 hold_2_15 - 903.117101937658 hold_2_16
   - 852.8241845664279 hold_2_17 - 807.8372268274302 hold_2_18
   - 767.3586256263421 hold_2_19 - 730.7430129516018 hold_2_20
   - 697.4625894599869 hold_2_21 - 667.0815183571873 hold_2_22
   + flow_nic_in_2_2 <= 0
 inactive_end_not_complete_0: is_active_0 = 0 -> is_end_complete_0 = 0
 active_end_val_0: is_active_0 = 1 -> - start_0 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11 - 12 hold_0_12
   - 13 hold_0_13 - 14 hold_0_14 - 15 hold_0_15 - 16 hold_0_16
   - 17 hold_0_17 - 18 hold_0_18 - 19 hold_0_19 - 20 hold_0_20
   - 21 hold_0_21 - 22 hold_0_22 + val_end_0 = 0
 inactive_end_zero_0: is_active_0 = 0 -> val_end_0 = 0
 inactive_end_not_complete_1: is_active_1 = 0 -> is_end_complete_1 = 0
 active_end_val_1: is_active_1 = 1 -> - start_1 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11 - 12 hold_1_12
   - 13 hold_1_13 - 14 hold_1_14 - 15 hold_1_15 - 16 hold_1_16
   - 17 hold_1_17 - 18 hold_1_18 - 19 hold_1_19 - 20 hold_1_20
   - 21 hold_1_21 - 22 hold_1_22 + val_end_1 = 0
 inactive_end_zero_1: is_active_1 = 0 -> val_end_1 = 0
 inactive_end_not_complete_2: is_active_2 = 0 -> is_end_complete_2 = 0
 active_end_val_2: is_active_2 = 1 -> - start_2 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 - 12 hold_2_12
   - 13 hold_2_13 - 14 hold_2_14 - 15 hold_2_15 - 16 hold_2_16
   - 17 hold_2_17 - 18 hold_2_18 - 19 hold_2_19 - 20 hold_2_20
   - 21 hold_2_21 - 22 hold_2_22 + val_end_2 = 0
 inactive_end_zero_2: is_active_2 = 0 -> val_end_2 = 0
 inactive_end_not_complete_3: is_active_3 = 0 -> is_end_complete_3 = 0
 active_end_val_3: is_active_3 = 1 -> - start_3 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   - 8 hold_3_8 - 9 hold_3_9 - 10 hold_3_10 - 11 hold_3_11 - 12 hold_3_12
   - 13 hold_3_13 - 14 hold_3_14 - 15 hold_3_15 - 16 hold_3_16
   - 17 hold_3_17 - 18 hold_3_18 - 19 hold_3_19 - 20 hold_3_20
   - 21 hold_3_21 - 22 hold_3_22 + val_end_3 = 0
 inactive_end_zero_3: is_active_3 = 0 -> val_end_3 = 0
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
   - 15314.28571428572 hold_1_1 - 7657.142857142858 hold_1_2
   - 5104.761904761905 hold_1_3 - 3828.571428571429 hold_1_4
   - 3062.857142857143 hold_1_5 - 2552.380952380952 hold_1_6
   - 2187.755102040816 hold_1_7 - 1914.285714285714 hold_1_8
   - 1701.587301587301 hold_1_9 - 1531.428571428571 hold_1_10
   - 1392.207792207792 hold_1_11 - 1276.190476190476 hold_1_12
   - 1178.021978021978 hold_1_13 - 1093.877551020408 hold_1_14
   - 1020.952380952381 hold_1_15 - 957.1428571428572 hold_1_16
   - 900.8403361344537 hold_1_17 - 850.7936507936507 hold_1_18
   - 806.015037593985 hold_1_19 - 765.7142857142857 hold_1_20
   - 729.2517006802721 hold_1_21 - 696.1038961038961 hold_1_22
   + flow_nic_in_1_1 <= 0
 group_throughput_constr_2: group_tp_active_2 = 1 ->
   - 15314.28571428572 hold_2_1 - 7657.142857142858 hold_2_2
   - 5104.761904761905 hold_2_3 - 3828.571428571429 hold_2_4
   - 3062.857142857143 hold_2_5 - 2552.380952380952 hold_2_6
   - 2187.755102040816 hold_2_7 - 1914.285714285714 hold_2_8
   - 1701.587301587301 hold_2_9 - 1531.428571428571 hold_2_10
   - 1392.207792207792 hold_2_11 - 1276.190476190476 hold_2_12
   - 1178.021978021978 hold_2_13 - 1093.877551020408 hold_2_14
   - 1020.952380952381 hold_2_15 - 957.1428571428572 hold_2_16
   - 900.8403361344537 hold_2_17 - 850.7936507936507 hold_2_18
   - 806.015037593985 hold_2_19 - 765.7142857142857 hold_2_20
   - 729.2517006802721 hold_2_21 - 696.1038961038961 hold_2_22
   + flow_nic_in_2_2 <= 0
 group_throughput_constr_3: group_tp_active_3 = 1 ->
   - 15314.28571428572 hold_3_1 - 7657.142857142858 hold_3_2
   - 5104.761904761905 hold_3_3 - 3828.571428571429 hold_3_4
   - 3062.857142857143 hold_3_5 - 2552.380952380952 hold_3_6
   - 2187.755102040816 hold_3_7 - 1914.285714285714 hold_3_8
   - 1701.587301587301 hold_3_9 - 1531.428571428571 hold_3_10
   - 1392.207792207792 hold_3_11 - 1276.190476190476 hold_3_12
   - 1178.021978021978 hold_3_13 - 1093.877551020408 hold_3_14
   - 1020.952380952381 hold_3_15 - 957.1428571428572 hold_3_16
   - 900.8403361344537 hold_3_17 - 850.7936507936507 hold_3_18
   - 806.015037593985 hold_3_19 - 765.7142857142857 hold_3_20
   - 729.2517006802721 hold_3_21 - 696.1038961038961 hold_3_22
   + flow_nic_in_3_3 <= 0
Bounds
Binaries
 hold_0_1 hold_0_2 hold_0_3 hold_0_4 hold_0_5 hold_0_6 hold_0_7 hold_0_8
 hold_0_9 hold_0_10 hold_0_11 hold_0_12 hold_0_13 hold_0_14 hold_0_15
 hold_0_16 hold_0_17 hold_0_18 hold_0_19 hold_0_20 hold_0_21 hold_0_22
 hold_1_1 hold_1_2 hold_1_3 hold_1_4 hold_1_5 hold_1_6 hold_1_7 hold_1_8
 hold_1_9 hold_1_10 hold_1_11 hold_1_12 hold_1_13 hold_1_14 hold_1_15
 hold_1_16 hold_1_17 hold_1_18 hold_1_19 hold_1_20 hold_1_21 hold_1_22
 hold_2_1 hold_2_2 hold_2_3 hold_2_4 hold_2_5 hold_2_6 hold_2_7 hold_2_8
 hold_2_9 hold_2_10 hold_2_11 hold_2_12 hold_2_13 hold_2_14 hold_2_15
 hold_2_16 hold_2_17 hold_2_18 hold_2_19 hold_2_20 hold_2_21 hold_2_22
 hold_3_1 hold_3_2 hold_3_3 hold_3_4 hold_3_5 hold_3_6 hold_3_7 hold_3_8
 hold_3_9 hold_3_10 hold_3_11 hold_3_12 hold_3_13 hold_3_14 hold_3_15
 hold_3_16 hold_3_17 hold_3_18 hold_3_19 hold_3_20 hold_3_21 hold_3_22
 switch_source_nic_in_0 switch_nic_out_0_sink switch_nic_out_0_nic_in_1
 switch_nic_out_0_nic_in_2 switch_nic_out_0_nic_in_3 switch_nic_in_0_0
 switch_0_nic_out_0 switch_source_nic_in_1 switch_nic_out_1_sink
 switch_nic_out_1_nic_in_0 switch_nic_out_1_nic_in_2
 switch_nic_out_1_nic_in_3 switch_nic_in_1_1 switch_1_nic_out_1
 switch_source_nic_in_2 switch_nic_out_2_sink switch_nic_out_2_nic_in_0
 switch_nic_out_2_nic_in_1 switch_nic_out_2_nic_in_3 switch_nic_in_2_2
 switch_2_nic_out_2 switch_source_nic_in_3 switch_nic_out_3_sink
 switch_nic_out_3_nic_in_0 switch_nic_out_3_nic_in_1
 switch_nic_out_3_nic_in_2 switch_nic_in_3_3 switch_3_nic_out_3
 group_tp_active_0 group_tp_active_1 group_tp_active_2 group_tp_active_3
 is_active_0 is_active_1 is_active_2 is_active_3
 is_start_zero_0_for_source_nic_in_0 is_end_complete_0
 is_start_zero_1_for_source_nic_in_1 is_end_complete_1
 is_start_zero_2_for_source_nic_in_2 is_end_complete_2
 is_start_zero_3_for_source_nic_in_3 is_end_complete_3
Generals
 start_0 start_1 start_2 start_3 group_start_1 val_end_0 group_end_0
 group_start_2 group_start_3 group_start_0 val_end_1 group_end_1 val_end_2
 group_end_2 val_end_3 group_end_3
General Constraints
 min_start_1: group_start_1 = MIN ( start_1 )
 max_end_0: group_end_0 = MAX ( val_end_0 )
 min_start_2: group_start_2 = MIN ( start_2 )
 min_start_3: group_start_3 = MIN ( start_3 )
 min_start_0: group_start_0 = MIN ( start_0 )
 max_end_1: group_end_1 = MAX ( val_end_1 )
 max_end_2: group_end_2 = MAX ( val_end_2 )
 max_end_3: group_end_3 = MAX ( val_end_3 )
End
