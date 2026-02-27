\ Model 2026-02-27_11-24-14
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  0.00193014705882353 flow_source_nic_in_0
   + 0.00193014705882353 flow_source_nic_in_1
   + 0.00193014705882353 flow_source_nic_in_2
   + 0.00193014705882353 flow_source_nic_in_3 - 0.025 is_active_0
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
   - 9325.714285714286 hold_0_1 - 4662.857142857143 hold_0_2
   - 3108.571428571428 hold_0_3 - 2331.428571428572 hold_0_4
   - 1865.142857142857 hold_0_5 - 1554.285714285714 hold_0_6
   - 1332.244897959184 hold_0_7 - 1165.714285714286 hold_0_8
   - 1036.190476190476 hold_0_9 - 932.5714285714284 hold_0_10
   - 847.7922077922078 hold_0_11 - 777.1428571428571 hold_0_12
   - 717.3626373626373 hold_0_13 - 666.1224489795918 hold_0_14
   - 621.7142857142858 hold_0_15 - 582.8571428571429 hold_0_16
   - 548.5714285714286 hold_0_17 - 518.0952380952381 hold_0_18
   - 490.8270676691729 hold_0_19 - 466.2857142857142 hold_0_20
   - 444.0816326530613 hold_0_21 - 423.8961038961039 hold_0_22
   + flow_nic_in_0_0 <= 0
 node_throughput_constraint_1: group_tp_active_1 = 0 ->
   - 9325.714285714286 hold_1_1 - 4662.857142857143 hold_1_2
   - 3108.571428571428 hold_1_3 - 2331.428571428572 hold_1_4
   - 1865.142857142857 hold_1_5 - 1554.285714285714 hold_1_6
   - 1332.244897959184 hold_1_7 - 1165.714285714286 hold_1_8
   - 1036.190476190476 hold_1_9 - 932.5714285714284 hold_1_10
   - 847.7922077922078 hold_1_11 - 777.1428571428571 hold_1_12
   - 717.3626373626373 hold_1_13 - 666.1224489795918 hold_1_14
   - 621.7142857142858 hold_1_15 - 582.8571428571429 hold_1_16
   - 548.5714285714286 hold_1_17 - 518.0952380952381 hold_1_18
   - 490.8270676691729 hold_1_19 - 466.2857142857142 hold_1_20
   - 444.0816326530613 hold_1_21 - 423.8961038961039 hold_1_22
   + flow_nic_in_1_1 <= 0
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
   - 9325.714285714286 hold_3_1 - 4662.857142857143 hold_3_2
   - 3108.571428571428 hold_3_3 - 2331.428571428572 hold_3_4
   - 1865.142857142857 hold_3_5 - 1554.285714285714 hold_3_6
   - 1332.244897959184 hold_3_7 - 1165.714285714286 hold_3_8
   - 1036.190476190476 hold_3_9 - 932.5714285714284 hold_3_10
   - 847.7922077922078 hold_3_11 - 777.1428571428571 hold_3_12
   - 717.3626373626373 hold_3_13 - 666.1224489795918 hold_3_14
   - 621.7142857142858 hold_3_15 - 582.8571428571429 hold_3_16
   - 548.5714285714286 hold_3_17 - 518.0952380952381 hold_3_18
   - 490.8270676691729 hold_3_19 - 466.2857142857142 hold_3_20
   - 444.0816326530613 hold_3_21 - 423.8961038961039 hold_3_22
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_0_source_nic_in_0: switch_source_nic_in_0 = 1 ->
   - 8352.09825997953 hold_0_1 - 4406.047516198704 hold_0_2
   - 2992.299229922993 hold_0_3 - 2265.40810660744 hold_0_4
   - 1822.649095376368 hold_0_5 - 1524.663677130045 hold_0_6
   - 1310.42235426369 hold_0_7 - 1148.972120529428 hold_0_8
   - 1022.940955246333 hold_0_9 - 921.825576140985 hold_0_10
   - 838.9020252904288 hold_0_11 - 769.6661007357103 hold_0_12
   - 710.9871917748541 hold_0_13 - 660.6217616580311 hold_0_14
   - 616.9199364935361 hold_0_15 - 578.6413274712806 hold_0_16
   - 544.8354143019296 hold_0_17 - 514.7615442846328 hold_0_18
   - 487.8340407724039 hold_0_19 - 463.5836836723099 hold_0_20
   - 441.6301347621368 hold_0_21 - 421.6618437370815 hold_0_22
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_0_nic_in_1: switch_nic_out_0_nic_in_1 = 1 ->
   - 5891.696750902527 hold_1_1 - 3610.619469026549 hold_1_2
   - 2602.870813397129 hold_1_3 - 2034.912718204489 hold_1_4
   - 1670.419651995905 hold_1_5 - 1416.666666666667 hold_1_6
   - 1229.841748304446 hold_1_7 - 1086.551264980027 hold_1_8
   - 973.1663685152056 hold_1_9 - 881.2095032397407 hold_1_10
   - 805.1307350764677 hold_1_11 - 741.1444141689373 hold_1_12
   - 686.5797223390829 hold_1_13 - 639.4984326018808 hold_1_14
   - 598.4598459845985 hold_1_15 - 562.3707787732598 hold_1_16
   - 530.3867403314916 hold_1_17 - 501.8450184501845 hold_1_18
   - 476.2182667055733 hold_1_19 - 453.081621321488 hold_1_20
   - 432.0889594916601 hold_1_21 - 412.9554655870446 hold_1_22
   + flow_nic_in_1_1 <= 0
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
   - 5891.696750902527 hold_3_1 - 3610.619469026549 hold_3_2
   - 2602.870813397129 hold_3_3 - 2034.912718204489 hold_3_4
   - 1670.419651995905 hold_3_5 - 1416.666666666667 hold_3_6
   - 1229.841748304446 hold_3_7 - 1086.551264980027 hold_3_8
   - 973.1663685152056 hold_3_9 - 881.2095032397407 hold_3_10
   - 805.1307350764677 hold_3_11 - 741.1444141689373 hold_3_12
   - 686.5797223390829 hold_3_13 - 639.4984326018808 hold_3_14
   - 598.4598459845985 hold_3_15 - 562.3707787732598 hold_3_16
   - 530.3867403314916 hold_3_17 - 501.8450184501845 hold_3_18
   - 476.2182667055733 hold_3_19 - 453.081621321488 hold_3_20
   - 432.0889594916601 hold_3_21 - 412.9554655870446 hold_3_22
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_1_source_nic_in_1: switch_source_nic_in_1 = 1 ->
   - 8352.09825997953 hold_1_1 - 4406.047516198704 hold_1_2
   - 2992.299229922993 hold_1_3 - 2265.40810660744 hold_1_4
   - 1822.649095376368 hold_1_5 - 1524.663677130045 hold_1_6
   - 1310.42235426369 hold_1_7 - 1148.972120529428 hold_1_8
   - 1022.940955246333 hold_1_9 - 921.825576140985 hold_1_10
   - 838.9020252904288 hold_1_11 - 769.6661007357103 hold_1_12
   - 710.9871917748541 hold_1_13 - 660.6217616580311 hold_1_14
   - 616.9199364935361 hold_1_15 - 578.6413274712806 hold_1_16
   - 544.8354143019296 hold_1_17 - 514.7615442846328 hold_1_18
   - 487.8340407724039 hold_1_19 - 463.5836836723099 hold_1_20
   - 441.6301347621368 hold_1_21 - 421.6618437370815 hold_1_22
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_0_nic_out_1_nic_in_0: switch_nic_out_1_nic_in_0 = 1 ->
   - 5891.696750902527 hold_0_1 - 3610.619469026549 hold_0_2
   - 2602.870813397129 hold_0_3 - 2034.912718204489 hold_0_4
   - 1670.419651995905 hold_0_5 - 1416.666666666667 hold_0_6
   - 1229.841748304446 hold_0_7 - 1086.551264980027 hold_0_8
   - 973.1663685152056 hold_0_9 - 881.2095032397407 hold_0_10
   - 805.1307350764677 hold_0_11 - 741.1444141689373 hold_0_12
   - 686.5797223390829 hold_0_13 - 639.4984326018808 hold_0_14
   - 598.4598459845985 hold_0_15 - 562.3707787732598 hold_0_16
   - 530.3867403314916 hold_0_17 - 501.8450184501845 hold_0_18
   - 476.2182667055733 hold_0_19 - 453.081621321488 hold_0_20
   - 432.0889594916601 hold_0_21 - 412.9554655870446 hold_0_22
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
   - 5891.696750902527 hold_3_1 - 3610.619469026549 hold_3_2
   - 2602.870813397129 hold_3_3 - 2034.912718204489 hold_3_4
   - 1670.419651995905 hold_3_5 - 1416.666666666667 hold_3_6
   - 1229.841748304446 hold_3_7 - 1086.551264980027 hold_3_8
   - 973.1663685152056 hold_3_9 - 881.2095032397407 hold_3_10
   - 805.1307350764677 hold_3_11 - 741.1444141689373 hold_3_12
   - 686.5797223390829 hold_3_13 - 639.4984326018808 hold_3_14
   - 598.4598459845985 hold_3_15 - 562.3707787732598 hold_3_16
   - 530.3867403314916 hold_3_17 - 501.8450184501845 hold_3_18
   - 476.2182667055733 hold_3_19 - 453.081621321488 hold_3_20
   - 432.0889594916601 hold_3_21 - 412.9554655870446 hold_3_22
   + flow_nic_in_3_3 <= 0
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
   - 5891.696750902527 hold_0_1 - 3610.619469026549 hold_0_2
   - 2602.870813397129 hold_0_3 - 2034.912718204489 hold_0_4
   - 1670.419651995905 hold_0_5 - 1416.666666666667 hold_0_6
   - 1229.841748304446 hold_0_7 - 1086.551264980027 hold_0_8
   - 973.1663685152056 hold_0_9 - 881.2095032397407 hold_0_10
   - 805.1307350764677 hold_0_11 - 741.1444141689373 hold_0_12
   - 686.5797223390829 hold_0_13 - 639.4984326018808 hold_0_14
   - 598.4598459845985 hold_0_15 - 562.3707787732598 hold_0_16
   - 530.3867403314916 hold_0_17 - 501.8450184501845 hold_0_18
   - 476.2182667055733 hold_0_19 - 453.081621321488 hold_0_20
   - 432.0889594916601 hold_0_21 - 412.9554655870446 hold_0_22
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_2_nic_in_1: switch_nic_out_2_nic_in_1 = 1 ->
   - 5891.696750902527 hold_1_1 - 3610.619469026549 hold_1_2
   - 2602.870813397129 hold_1_3 - 2034.912718204489 hold_1_4
   - 1670.419651995905 hold_1_5 - 1416.666666666667 hold_1_6
   - 1229.841748304446 hold_1_7 - 1086.551264980027 hold_1_8
   - 973.1663685152056 hold_1_9 - 881.2095032397407 hold_1_10
   - 805.1307350764677 hold_1_11 - 741.1444141689373 hold_1_12
   - 686.5797223390829 hold_1_13 - 639.4984326018808 hold_1_14
   - 598.4598459845985 hold_1_15 - 562.3707787732598 hold_1_16
   - 530.3867403314916 hold_1_17 - 501.8450184501845 hold_1_18
   - 476.2182667055733 hold_1_19 - 453.081621321488 hold_1_20
   - 432.0889594916601 hold_1_21 - 412.9554655870446 hold_1_22
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_3_nic_out_2_nic_in_3: switch_nic_out_2_nic_in_3 = 1 ->
   - 5891.696750902527 hold_3_1 - 3610.619469026549 hold_3_2
   - 2602.870813397129 hold_3_3 - 2034.912718204489 hold_3_4
   - 1670.419651995905 hold_3_5 - 1416.666666666667 hold_3_6
   - 1229.841748304446 hold_3_7 - 1086.551264980027 hold_3_8
   - 973.1663685152056 hold_3_9 - 881.2095032397407 hold_3_10
   - 805.1307350764677 hold_3_11 - 741.1444141689373 hold_3_12
   - 686.5797223390829 hold_3_13 - 639.4984326018808 hold_3_14
   - 598.4598459845985 hold_3_15 - 562.3707787732598 hold_3_16
   - 530.3867403314916 hold_3_17 - 501.8450184501845 hold_3_18
   - 476.2182667055733 hold_3_19 - 453.081621321488 hold_3_20
   - 432.0889594916601 hold_3_21 - 412.9554655870446 hold_3_22
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_3_source_nic_in_3: switch_source_nic_in_3 = 1 ->
   - 8352.09825997953 hold_3_1 - 4406.047516198704 hold_3_2
   - 2992.299229922993 hold_3_3 - 2265.40810660744 hold_3_4
   - 1822.649095376368 hold_3_5 - 1524.663677130045 hold_3_6
   - 1310.42235426369 hold_3_7 - 1148.972120529428 hold_3_8
   - 1022.940955246333 hold_3_9 - 921.825576140985 hold_3_10
   - 838.9020252904288 hold_3_11 - 769.6661007357103 hold_3_12
   - 710.9871917748541 hold_3_13 - 660.6217616580311 hold_3_14
   - 616.9199364935361 hold_3_15 - 578.6413274712806 hold_3_16
   - 544.8354143019296 hold_3_17 - 514.7615442846328 hold_3_18
   - 487.8340407724039 hold_3_19 - 463.5836836723099 hold_3_20
   - 441.6301347621368 hold_3_21 - 421.6618437370815 hold_3_22
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_0_nic_out_3_nic_in_0: switch_nic_out_3_nic_in_0 = 1 ->
   - 5891.696750902527 hold_0_1 - 3610.619469026549 hold_0_2
   - 2602.870813397129 hold_0_3 - 2034.912718204489 hold_0_4
   - 1670.419651995905 hold_0_5 - 1416.666666666667 hold_0_6
   - 1229.841748304446 hold_0_7 - 1086.551264980027 hold_0_8
   - 973.1663685152056 hold_0_9 - 881.2095032397407 hold_0_10
   - 805.1307350764677 hold_0_11 - 741.1444141689373 hold_0_12
   - 686.5797223390829 hold_0_13 - 639.4984326018808 hold_0_14
   - 598.4598459845985 hold_0_15 - 562.3707787732598 hold_0_16
   - 530.3867403314916 hold_0_17 - 501.8450184501845 hold_0_18
   - 476.2182667055733 hold_0_19 - 453.081621321488 hold_0_20
   - 432.0889594916601 hold_0_21 - 412.9554655870446 hold_0_22
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_3_nic_in_1: switch_nic_out_3_nic_in_1 = 1 ->
   - 5891.696750902527 hold_1_1 - 3610.619469026549 hold_1_2
   - 2602.870813397129 hold_1_3 - 2034.912718204489 hold_1_4
   - 1670.419651995905 hold_1_5 - 1416.666666666667 hold_1_6
   - 1229.841748304446 hold_1_7 - 1086.551264980027 hold_1_8
   - 973.1663685152056 hold_1_9 - 881.2095032397407 hold_1_10
   - 805.1307350764677 hold_1_11 - 741.1444141689373 hold_1_12
   - 686.5797223390829 hold_1_13 - 639.4984326018808 hold_1_14
   - 598.4598459845985 hold_1_15 - 562.3707787732598 hold_1_16
   - 530.3867403314916 hold_1_17 - 501.8450184501845 hold_1_18
   - 476.2182667055733 hold_1_19 - 453.081621321488 hold_1_20
   - 432.0889594916601 hold_1_21 - 412.9554655870446 hold_1_22
   + flow_nic_in_1_1 <= 0
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
   - 9325.714285714286 hold_0_1 - 4662.857142857143 hold_0_2
   - 3108.571428571428 hold_0_3 - 2331.428571428572 hold_0_4
   - 1865.142857142857 hold_0_5 - 1554.285714285714 hold_0_6
   - 1332.244897959184 hold_0_7 - 1165.714285714286 hold_0_8
   - 1036.190476190476 hold_0_9 - 932.5714285714284 hold_0_10
   - 847.7922077922078 hold_0_11 - 777.1428571428571 hold_0_12
   - 717.3626373626373 hold_0_13 - 666.1224489795918 hold_0_14
   - 621.7142857142858 hold_0_15 - 582.8571428571429 hold_0_16
   - 548.5714285714286 hold_0_17 - 518.0952380952381 hold_0_18
   - 490.8270676691729 hold_0_19 - 466.2857142857142 hold_0_20
   - 444.0816326530613 hold_0_21 - 423.8961038961039 hold_0_22
   + flow_nic_in_0_0 <= 0
 group_throughput_constr_1: group_tp_active_1 = 1 ->
   - 9325.714285714286 hold_1_1 - 4662.857142857143 hold_1_2
   - 3108.571428571428 hold_1_3 - 2331.428571428572 hold_1_4
   - 1865.142857142857 hold_1_5 - 1554.285714285714 hold_1_6
   - 1332.244897959184 hold_1_7 - 1165.714285714286 hold_1_8
   - 1036.190476190476 hold_1_9 - 932.5714285714284 hold_1_10
   - 847.7922077922078 hold_1_11 - 777.1428571428571 hold_1_12
   - 717.3626373626373 hold_1_13 - 666.1224489795918 hold_1_14
   - 621.7142857142858 hold_1_15 - 582.8571428571429 hold_1_16
   - 548.5714285714286 hold_1_17 - 518.0952380952381 hold_1_18
   - 490.8270676691729 hold_1_19 - 466.2857142857142 hold_1_20
   - 444.0816326530613 hold_1_21 - 423.8961038961039 hold_1_22
   + flow_nic_in_1_1 <= 0
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
   - 9325.714285714286 hold_3_1 - 4662.857142857143 hold_3_2
   - 3108.571428571428 hold_3_3 - 2331.428571428572 hold_3_4
   - 1865.142857142857 hold_3_5 - 1554.285714285714 hold_3_6
   - 1332.244897959184 hold_3_7 - 1165.714285714286 hold_3_8
   - 1036.190476190476 hold_3_9 - 932.5714285714284 hold_3_10
   - 847.7922077922078 hold_3_11 - 777.1428571428571 hold_3_12
   - 717.3626373626373 hold_3_13 - 666.1224489795918 hold_3_14
   - 621.7142857142858 hold_3_15 - 582.8571428571429 hold_3_16
   - 548.5714285714286 hold_3_17 - 518.0952380952381 hold_3_18
   - 490.8270676691729 hold_3_19 - 466.2857142857142 hold_3_20
   - 444.0816326530613 hold_3_21 - 423.8961038961039 hold_3_22
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
