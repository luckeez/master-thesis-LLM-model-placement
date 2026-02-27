\ Model 2026-02-27_12-26-24
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  0.00246710526315789 flow_source_nic_in_0
   + 0.00246710526315789 flow_source_nic_in_1
   + 0.00246710526315789 flow_source_nic_in_3
   + 0.00246710526315789 flow_source_nic_in_4
   - 0.0129251700680272 is_active_0 - 0.0129251700680272 is_active_1
   - 0.0129251700680272 is_active_2 - 0.0306122448979592 is_active_3
   - 0.0306122448979592 is_active_4 + 0 val_end_0 + 0 val_end_1
   + 0 val_end_2 + 0 val_end_3 + 0 val_end_4
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
 node_flow_constraint_nic_in_0: flow_source_nic_in_0 - flow_nic_in_0_0
   + flow_nic_out_1_nic_in_0 + flow_nic_out_3_nic_in_0
   + flow_nic_out_4_nic_in_0 = 0
 node_flow_constraint_nic_out_0: - flow_nic_out_0_sink
   - flow_nic_out_0_nic_in_1 - flow_nic_out_0_nic_in_3
   - flow_nic_out_0_nic_in_4 + flow_0_nic_out_0 = 0
 node_flow_constraint_nic_in_1: flow_nic_out_0_nic_in_1
   + flow_source_nic_in_1 - flow_nic_in_1_1 - flow_nic_in_1_2
   + flow_nic_out_3_nic_in_1 + flow_nic_out_4_nic_in_1 = 0
 node_flow_constraint_nic_out_1: - flow_nic_out_1_sink
   - flow_nic_out_1_nic_in_0 - flow_nic_out_1_nic_in_3
   - flow_nic_out_1_nic_in_4 + flow_1_nic_out_1 + flow_2_nic_out_1 = 0
 node_flow_constraint_nic_in_3: flow_nic_out_0_nic_in_3
   + flow_nic_out_1_nic_in_3 + flow_source_nic_in_3 - flow_nic_in_3_3
   + flow_nic_out_4_nic_in_3 = 0
 node_flow_constraint_nic_out_3: - flow_nic_out_3_sink
   - flow_nic_out_3_nic_in_0 - flow_nic_out_3_nic_in_1
   - flow_nic_out_3_nic_in_4 + flow_3_nic_out_3 = 0
 node_flow_constraint_nic_in_4: flow_nic_out_0_nic_in_4
   + flow_nic_out_1_nic_in_4 + flow_nic_out_3_nic_in_4
   + flow_source_nic_in_4 - flow_nic_in_4_4 = 0
 node_flow_constraint_nic_out_4: - flow_nic_out_4_sink
   - flow_nic_out_4_nic_in_0 - flow_nic_out_4_nic_in_1
   - flow_nic_out_4_nic_in_3 + flow_4_nic_out_4 = 0
 node_flow_constraint_0: flow_nic_in_0_0 - flow_0_nic_out_0 = 0
 node_flow_constraint_1: flow_nic_in_1_1 - flow_1_nic_out_1 - flow_1_2
   + flow_2_1 = 0
 node_flow_constraint_2: flow_nic_in_1_2 - flow_2_nic_out_1 + flow_1_2
   - flow_2_1 = 0
 node_flow_constraint_3: flow_nic_in_3_3 - flow_3_nic_out_3 = 0
 node_flow_constraint_4: flow_nic_in_4_4 - flow_4_nic_out_4 = 0
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
 constr_is_start_zero_2: start_2 + 80 is_start_zero_2_for_source_nic_in_1
   <= 80
 constr_edge_switch_start_zero_1_OR: switch_source_nic_in_1
   - is_start_zero_1_for_source_nic_in_1
   - is_start_zero_2_for_source_nic_in_1 <= 0
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
 constr_edge_switch_end_max_1_OR: switch_nic_out_1_sink - is_end_complete_1
   - is_end_complete_2 <= 0
 constr_link_match_1_1_0: 80 switch_nic_out_1_nic_in_0 - group_start_0
   + group_end_1 <= 80
 constr_link_match_2_1_0: 80 switch_nic_out_1_nic_in_0 + group_start_0
   - group_end_1 <= 80
 constr_link_match_1_1_3: 80 switch_nic_out_1_nic_in_3 - group_start_3
   + group_end_1 <= 80
 constr_link_match_2_1_3: 80 switch_nic_out_1_nic_in_3 + group_start_3
   - group_end_1 <= 80
 constr_link_match_1_1_4: 80 switch_nic_out_1_nic_in_4 - group_start_4
   + group_end_1 <= 80
 constr_link_match_2_1_4: 80 switch_nic_out_1_nic_in_4 + group_start_4
   - group_end_1 <= 80
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
 constr_internal_compute_1_2: switch_1_2 - group_tp_active_1
   - pp_permission_1_2 <= 0
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
 constr_internal_compute_2_1: switch_2_1 - group_tp_active_1
   - pp_permission_2_1 <= 0
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
 constr_link_match_1_3_4: 80 switch_nic_out_3_nic_in_4 - group_start_4
   + group_end_3 <= 80
 constr_link_match_2_3_4: 80 switch_nic_out_3_nic_in_4 + group_start_4
   - group_end_3 <= 80
 constr_is_start_zero_4: start_4 + 80 is_start_zero_4_for_source_nic_in_4
   <= 80
 constr_edge_switch_start_zero_4_OR: switch_source_nic_in_4
   - is_start_zero_4_for_source_nic_in_4 <= 0
 constr_is_end_complete_4: start_4 + hold_4_1 + 2 hold_4_2 + 3 hold_4_3
   + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7 + 8 hold_4_8
   + 9 hold_4_9 + 10 hold_4_10 + 11 hold_4_11 + 12 hold_4_12 + 13 hold_4_13
   + 14 hold_4_14 + 15 hold_4_15 + 16 hold_4_16 + 17 hold_4_17
   + 18 hold_4_18 + 19 hold_4_19 + 20 hold_4_20 + 21 hold_4_21
   + 22 hold_4_22 + 23 hold_4_23 + 24 hold_4_24 + 25 hold_4_25
   + 26 hold_4_26 - 80 is_end_complete_4 >= 0
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
 constr_link_match_1_4_3: 80 switch_nic_out_4_nic_in_3 - group_start_3
   + group_end_4 <= 80
 constr_link_match_2_4_3: 80 switch_nic_out_4_nic_in_3 + group_start_3
   - group_end_4 <= 80
 link_activation_dst_nic_in_0_0: switch_nic_in_0_0 - is_active_0 <= 0
 link_activation_src_0_nic_out_0: switch_0_nic_out_0 - is_active_0 <= 0
 link_activation_dst_nic_in_1_1: switch_nic_in_1_1 - is_active_1 <= 0
 link_activation_src_1_nic_out_1: switch_1_nic_out_1 - is_active_1 <= 0
 link_activation_dst_nic_in_1_2: switch_nic_in_1_2 - is_active_2 <= 0
 link_activation_src_2_nic_out_1: switch_2_nic_out_1 - is_active_2 <= 0
 link_activation_src_1_2: switch_1_2 - is_active_1 <= 0
 link_activation_dst_1_2: switch_1_2 - is_active_2 <= 0
 link_activation_src_2_1: switch_2_1 - is_active_2 <= 0
 link_activation_dst_2_1: switch_2_1 - is_active_1 <= 0
 link_activation_dst_nic_in_3_3: switch_nic_in_3_3 - is_active_3 <= 0
 link_activation_src_3_nic_out_3: switch_3_nic_out_3 - is_active_3 <= 0
 link_activation_dst_nic_in_4_4: switch_nic_in_4_4 - is_active_4 <= 0
 link_activation_src_4_nic_out_4: switch_4_nic_out_4 - is_active_4 <= 0
 edge_flow_constr_source_nic_in_0: flow_source_nic_in_0
   - 5e+08 switch_source_nic_in_0 <= 0
 edge_flow_constr_nic_out_0_sink: flow_nic_out_0_sink
   - 5e+08 switch_nic_out_0_sink <= 0
 edge_flow_constr_nic_out_0_nic_in_1: flow_nic_out_0_nic_in_1
   - 61035.15625 switch_nic_out_0_nic_in_1 <= 0
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
 edge_flow_constr_nic_out_1_nic_in_3: flow_nic_out_1_nic_in_3
   - 61035.15625 switch_nic_out_1_nic_in_3 <= 0
 edge_flow_constr_nic_out_1_nic_in_4: flow_nic_out_1_nic_in_4
   - 61035.15625 switch_nic_out_1_nic_in_4 <= 0
 edge_flow_constr_nic_in_1_1: flow_nic_in_1_1
   - 3.90625e+06 switch_nic_in_1_1 <= 0
 edge_flow_constr_1_nic_out_1: flow_1_nic_out_1
   - 3.90625e+06 switch_1_nic_out_1 <= 0
 edge_flow_constr_nic_in_1_2: flow_nic_in_1_2
   - 3.90625e+06 switch_nic_in_1_2 <= 0
 edge_flow_constr_2_nic_out_1: flow_2_nic_out_1
   - 3.90625e+06 switch_2_nic_out_1 <= 0
 edge_flow_constr_1_2: flow_1_2 - 3.90625e+06 switch_1_2 <= 0
 edge_flow_constr_2_1: flow_2_1 - 3.90625e+06 switch_2_1 <= 0
 edge_flow_constr_source_nic_in_3: flow_source_nic_in_3
   - 5e+08 switch_source_nic_in_3 <= 0
 edge_flow_constr_nic_out_3_sink: flow_nic_out_3_sink
   - 5e+08 switch_nic_out_3_sink <= 0
 edge_flow_constr_nic_out_3_nic_in_0: flow_nic_out_3_nic_in_0
   - 61035.15625 switch_nic_out_3_nic_in_0 <= 0
 edge_flow_constr_nic_out_3_nic_in_1: flow_nic_out_3_nic_in_1
   - 61035.15625 switch_nic_out_3_nic_in_1 <= 0
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
 edge_flow_constr_nic_out_4_nic_in_3: flow_nic_out_4_nic_in_3
   - 61035.15625 switch_nic_out_4_nic_in_3 <= 0
 edge_flow_constr_nic_in_4_4: flow_nic_in_4_4
   - 3.90625e+06 switch_nic_in_4_4 <= 0
 edge_flow_constr_4_nic_out_4: flow_4_nic_out_4
   - 3.90625e+06 switch_4_nic_out_4 <= 0
 force_group_off_constr_0: group_tp_active_0 = 0
 force_group_off_constr_3: group_tp_active_3 = 0
 force_group_off_constr_4: group_tp_active_4 = 0
 R130: - group_tp_active_1 + tp_on_and_switch_nic_out_0_nic_in_1_1 <= 0
 R131: - switch_nic_out_0_nic_in_1 + tp_on_and_switch_nic_out_0_nic_in_1_1
   <= 0
 R132: - switch_nic_out_0_nic_in_1 - group_tp_active_1
   + tp_on_and_switch_nic_out_0_nic_in_1_1 >= -1
 R133: - group_tp_active_1 + tp_on_and_switch_source_nic_in_1_1 <= 0
 R134: - switch_source_nic_in_1 + tp_on_and_switch_source_nic_in_1_1 <= 0
 R135: - switch_source_nic_in_1 - group_tp_active_1
   + tp_on_and_switch_source_nic_in_1_1 >= -1
 R136: - group_tp_active_1 + tp_on_and_switch_nic_out_3_nic_in_1_1 <= 0
 R137: - switch_nic_out_3_nic_in_1 + tp_on_and_switch_nic_out_3_nic_in_1_1
   <= 0
 R138: - switch_nic_out_3_nic_in_1 - group_tp_active_1
   + tp_on_and_switch_nic_out_3_nic_in_1_1 >= -1
 R139: - group_tp_active_1 + tp_on_and_switch_nic_out_4_nic_in_1_1 <= 0
 R140: - switch_nic_out_4_nic_in_1 + tp_on_and_switch_nic_out_4_nic_in_1_1
   <= 0
 R141: - switch_nic_out_4_nic_in_1 - group_tp_active_1
   + tp_on_and_switch_nic_out_4_nic_in_1_1 >= -1
 force_min_throughput: flow_source_nic_in_0 + flow_source_nic_in_1
   + flow_source_nic_in_3 + flow_source_nic_in_4 >= 1
 inactive_start_invalid_0: is_active_0 = 0 -> start_0 = 80
 inactive_start_invalid_1: is_active_1 = 0 -> start_1 = 80
 inactive_start_invalid_2: is_active_2 = 0 -> start_2 = 80
 inactive_start_invalid_3: is_active_3 = 0 -> start_3 = 80
 inactive_start_invalid_4: is_active_4 = 0 -> start_4 = 80
 node_throughput_constraint_0: group_tp_active_0 = 0 ->
   - 3510.857142857143 hold_0_1 - 1755.428571428571 hold_0_2
   - 1170.285714285714 hold_0_3 - 877.7142857142857 hold_0_4
   - 702.1714285714286 hold_0_5 - 585.1428571428571 hold_0_6
   - 501.5510204081633 hold_0_7 - 438.8571428571428 hold_0_8
   - 390.0952380952381 hold_0_9 - 351.0857142857143 hold_0_10
   - 319.1688311688312 hold_0_11 - 292.5714285714286 hold_0_12
   - 270.0659340659341 hold_0_13 + flow_nic_in_0_0 <= 0
 node_throughput_constraint_1: group_tp_active_1 = 0 ->
   - 3510.857142857143 hold_1_1 - 1755.428571428571 hold_1_2
   - 1170.285714285714 hold_1_3 - 877.7142857142857 hold_1_4
   - 702.1714285714286 hold_1_5 - 585.1428571428571 hold_1_6
   - 501.5510204081633 hold_1_7 - 438.8571428571428 hold_1_8
   - 390.0952380952381 hold_1_9 - 351.0857142857143 hold_1_10
   - 319.1688311688312 hold_1_11 - 292.5714285714286 hold_1_12
   - 270.0659340659341 hold_1_13 + flow_nic_in_1_1 + flow_2_1 <= 0
 node_throughput_constraint_2: group_tp_active_1 = 0 ->
   - 3510.857142857143 hold_2_1 - 1755.428571428571 hold_2_2
   - 1170.285714285714 hold_2_3 - 877.7142857142857 hold_2_4
   - 702.1714285714286 hold_2_5 - 585.1428571428571 hold_2_6
   - 501.5510204081633 hold_2_7 - 438.8571428571428 hold_2_8
   - 390.0952380952381 hold_2_9 - 351.0857142857143 hold_2_10
   - 319.1688311688312 hold_2_11 - 292.5714285714286 hold_2_12
   - 270.0659340659341 hold_2_13 + flow_nic_in_1_2 + flow_1_2 <= 0
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
 node_throughput_constraint_4: group_tp_active_4 = 0 ->
   - 9325.714285714286 hold_4_1 - 4662.857142857143 hold_4_2
   - 3108.571428571428 hold_4_3 - 2331.428571428572 hold_4_4
   - 1865.142857142857 hold_4_5 - 1554.285714285714 hold_4_6
   - 1332.244897959184 hold_4_7 - 1165.714285714286 hold_4_8
   - 1036.190476190476 hold_4_9 - 932.5714285714284 hold_4_10
   - 847.7922077922078 hold_4_11 - 777.1428571428571 hold_4_12
   - 717.3626373626373 hold_4_13 - 666.1224489795918 hold_4_14
   - 621.7142857142858 hold_4_15 - 582.8571428571429 hold_4_16
   - 548.5714285714286 hold_4_17 - 518.0952380952381 hold_4_18
   - 490.8270676691729 hold_4_19 - 466.2857142857142 hold_4_20
   - 444.0816326530613 hold_4_21 - 423.8961038961039 hold_4_22
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_0_source_nic_in_0: switch_source_nic_in_0 = 1 ->
   - 3363.258156338954 hold_0_1 - 1717.736524267501 hold_0_2
   - 1153.412930840279 hold_0_3 - 868.1890119828171 hold_0_4
   - 696.0619930212536 hold_0_5 - 580.8940322214659 hold_0_6
   - 498.4261933348477 hold_0_7 - 436.4628324619231 hold_0_8
   - 388.2022897869437 hold_0_9 - 349.5516817623231 hold_0_10
   - 317.9005319038848 hold_0_11 - 291.5053518560692 hold_0_12
   - 269.1573063241453 hold_0_13 + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_0_nic_in_1: switch_nic_out_0_nic_in_1 = 1 ->
   - 2879.100281162137 hold_1_1 - 1581.874356333677 hold_1_2
   - 1090.521831735889 hold_1_3 - 832.0693391115925 hold_1_4
   - 672.6516312677907 hold_1_5 - 564.4983461962513 hold_1_6
   - 486.3067911983537 hold_1_7 - 427.1412680756396 hold_1_8
   - 380.8107103012272 hold_1_9 - 343.5473048535003 hold_1_10
   - 312.9265559743302 hold_1_11 - 287.317620650954 hold_1_12
   - 265.5831244056367 hold_1_13 + flow_nic_in_1_1 + flow_2_1 <= 0
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
 constr_intergroup_throughput_4_nic_out_0_nic_in_4: switch_nic_out_0_nic_in_4 = 1 ->
   - 5891.696750902527 hold_4_1 - 3610.619469026549 hold_4_2
   - 2602.870813397129 hold_4_3 - 2034.912718204489 hold_4_4
   - 1670.419651995905 hold_4_5 - 1416.666666666667 hold_4_6
   - 1229.841748304446 hold_4_7 - 1086.551264980027 hold_4_8
   - 973.1663685152056 hold_4_9 - 881.2095032397407 hold_4_10
   - 805.1307350764677 hold_4_11 - 741.1444141689373 hold_4_12
   - 686.5797223390829 hold_4_13 - 639.4984326018808 hold_4_14
   - 598.4598459845985 hold_4_15 - 562.3707787732598 hold_4_16
   - 530.3867403314916 hold_4_17 - 501.8450184501845 hold_4_18
   - 476.2182667055733 hold_4_19 - 453.081621321488 hold_4_20
   - 432.0889594916601 hold_4_21 - 412.9554655870446 hold_4_22
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_1_source_nic_in_1: switch_source_nic_in_1 = 1 ->
   - 3363.258156338954 hold_1_1 - 1717.736524267501 hold_1_2
   - 1153.412930840279 hold_1_3 - 868.1890119828171 hold_1_4
   - 696.0619930212536 hold_1_5 - 580.8940322214659 hold_1_6
   - 498.4261933348477 hold_1_7 - 436.4628324619231 hold_1_8
   - 388.2022897869437 hold_1_9 - 349.5516817623231 hold_1_10
   - 317.9005319038848 hold_1_11 - 291.5053518560692 hold_1_12
   - 269.1573063241453 hold_1_13 + flow_nic_in_1_1 + flow_2_1 <= 0
 constr_intergroup_throughput_0_nic_out_1_nic_in_0: switch_nic_out_1_nic_in_0 = 1 ->
   - 2879.100281162137 hold_0_1 - 1581.874356333677 hold_0_2
   - 1090.521831735889 hold_0_3 - 832.0693391115925 hold_0_4
   - 672.6516312677907 hold_0_5 - 564.4983461962513 hold_0_6
   - 486.3067911983537 hold_0_7 - 427.1412680756396 hold_0_8
   - 380.8107103012272 hold_0_9 - 343.5473048535003 hold_0_10
   - 312.9265559743302 hold_0_11 - 287.317620650954 hold_0_12
   - 265.5831244056367 hold_0_13 + flow_nic_in_0_0 <= 0
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
 constr_intergroup_throughput_4_nic_out_1_nic_in_4: switch_nic_out_1_nic_in_4 = 1 ->
   - 5891.696750902527 hold_4_1 - 3610.619469026549 hold_4_2
   - 2602.870813397129 hold_4_3 - 2034.912718204489 hold_4_4
   - 1670.419651995905 hold_4_5 - 1416.666666666667 hold_4_6
   - 1229.841748304446 hold_4_7 - 1086.551264980027 hold_4_8
   - 973.1663685152056 hold_4_9 - 881.2095032397407 hold_4_10
   - 805.1307350764677 hold_4_11 - 741.1444141689373 hold_4_12
   - 686.5797223390829 hold_4_13 - 639.4984326018808 hold_4_14
   - 598.4598459845985 hold_4_15 - 562.3707787732598 hold_4_16
   - 530.3867403314916 hold_4_17 - 501.8450184501845 hold_4_18
   - 476.2182667055733 hold_4_19 - 453.081621321488 hold_4_20
   - 432.0889594916601 hold_4_21 - 412.9554655870446 hold_4_22
   + flow_nic_in_4_4 <= 0
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
   - 2879.100281162137 hold_0_1 - 1581.874356333677 hold_0_2
   - 1090.521831735889 hold_0_3 - 832.0693391115925 hold_0_4
   - 672.6516312677907 hold_0_5 - 564.4983461962513 hold_0_6
   - 486.3067911983537 hold_0_7 - 427.1412680756396 hold_0_8
   - 380.8107103012272 hold_0_9 - 343.5473048535003 hold_0_10
   - 312.9265559743302 hold_0_11 - 287.317620650954 hold_0_12
   - 265.5831244056367 hold_0_13 + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_3_nic_in_1: switch_nic_out_3_nic_in_1 = 1 ->
   - 2879.100281162137 hold_1_1 - 1581.874356333677 hold_1_2
   - 1090.521831735889 hold_1_3 - 832.0693391115925 hold_1_4
   - 672.6516312677907 hold_1_5 - 564.4983461962513 hold_1_6
   - 486.3067911983537 hold_1_7 - 427.1412680756396 hold_1_8
   - 380.8107103012272 hold_1_9 - 343.5473048535003 hold_1_10
   - 312.9265559743302 hold_1_11 - 287.317620650954 hold_1_12
   - 265.5831244056367 hold_1_13 + flow_nic_in_1_1 + flow_2_1 <= 0
 constr_intergroup_throughput_4_nic_out_3_nic_in_4: switch_nic_out_3_nic_in_4 = 1 ->
   - 5891.696750902527 hold_4_1 - 3610.619469026549 hold_4_2
   - 2602.870813397129 hold_4_3 - 2034.912718204489 hold_4_4
   - 1670.419651995905 hold_4_5 - 1416.666666666667 hold_4_6
   - 1229.841748304446 hold_4_7 - 1086.551264980027 hold_4_8
   - 973.1663685152056 hold_4_9 - 881.2095032397407 hold_4_10
   - 805.1307350764677 hold_4_11 - 741.1444141689373 hold_4_12
   - 686.5797223390829 hold_4_13 - 639.4984326018808 hold_4_14
   - 598.4598459845985 hold_4_15 - 562.3707787732598 hold_4_16
   - 530.3867403314916 hold_4_17 - 501.8450184501845 hold_4_18
   - 476.2182667055733 hold_4_19 - 453.081621321488 hold_4_20
   - 432.0889594916601 hold_4_21 - 412.9554655870446 hold_4_22
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_4_source_nic_in_4: switch_source_nic_in_4 = 1 ->
   - 8352.09825997953 hold_4_1 - 4406.047516198704 hold_4_2
   - 2992.299229922993 hold_4_3 - 2265.40810660744 hold_4_4
   - 1822.649095376368 hold_4_5 - 1524.663677130045 hold_4_6
   - 1310.42235426369 hold_4_7 - 1148.972120529428 hold_4_8
   - 1022.940955246333 hold_4_9 - 921.825576140985 hold_4_10
   - 838.9020252904288 hold_4_11 - 769.6661007357103 hold_4_12
   - 710.9871917748541 hold_4_13 - 660.6217616580311 hold_4_14
   - 616.9199364935361 hold_4_15 - 578.6413274712806 hold_4_16
   - 544.8354143019296 hold_4_17 - 514.7615442846328 hold_4_18
   - 487.8340407724039 hold_4_19 - 463.5836836723099 hold_4_20
   - 441.6301347621368 hold_4_21 - 421.6618437370815 hold_4_22
   + flow_nic_in_4_4 <= 0
 constr_intergroup_throughput_0_nic_out_4_nic_in_0: switch_nic_out_4_nic_in_0 = 1 ->
   - 2879.100281162137 hold_0_1 - 1581.874356333677 hold_0_2
   - 1090.521831735889 hold_0_3 - 832.0693391115925 hold_0_4
   - 672.6516312677907 hold_0_5 - 564.4983461962513 hold_0_6
   - 486.3067911983537 hold_0_7 - 427.1412680756396 hold_0_8
   - 380.8107103012272 hold_0_9 - 343.5473048535003 hold_0_10
   - 312.9265559743302 hold_0_11 - 287.317620650954 hold_0_12
   - 265.5831244056367 hold_0_13 + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_4_nic_in_1: switch_nic_out_4_nic_in_1 = 1 ->
   - 2879.100281162137 hold_1_1 - 1581.874356333677 hold_1_2
   - 1090.521831735889 hold_1_3 - 832.0693391115925 hold_1_4
   - 672.6516312677907 hold_1_5 - 564.4983461962513 hold_1_6
   - 486.3067911983537 hold_1_7 - 427.1412680756396 hold_1_8
   - 380.8107103012272 hold_1_9 - 343.5473048535003 hold_1_10
   - 312.9265559743302 hold_1_11 - 287.317620650954 hold_1_12
   - 265.5831244056367 hold_1_13 + flow_nic_in_1_1 + flow_2_1 <= 0
 constr_intergroup_throughput_3_nic_out_4_nic_in_3: switch_nic_out_4_nic_in_3 = 1 ->
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
 inactive_end_not_complete_2: is_active_2 = 0 -> is_end_complete_2 = 0
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
 active_end_val_4: is_active_4 = 1 -> - start_4 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   - 8 hold_4_8 - 9 hold_4_9 - 10 hold_4_10 - 11 hold_4_11 - 12 hold_4_12
   - 13 hold_4_13 - 14 hold_4_14 - 15 hold_4_15 - 16 hold_4_16
   - 17 hold_4_17 - 18 hold_4_18 - 19 hold_4_19 - 20 hold_4_20
   - 21 hold_4_21 - 22 hold_4_22 - 23 hold_4_23 - 24 hold_4_24
   - 25 hold_4_25 - 26 hold_4_26 + val_end_4 = 0
 inactive_end_zero_4: is_active_4 = 0 -> val_end_4 = 0
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
 group_synch_start_constr_1_2: group_tp_active_1 = 1 -> start_1 - start_2
   = 0
 group_synch_hold_constr_1_2_1: group_tp_active_1 = 1 -> hold_1_1
   - hold_2_1 = 0
 group_synch_hold_constr_1_2_2: group_tp_active_1 = 1 -> hold_1_2
   - hold_2_2 = 0
 group_synch_hold_constr_1_2_3: group_tp_active_1 = 1 -> hold_1_3
   - hold_2_3 = 0
 group_synch_hold_constr_1_2_4: group_tp_active_1 = 1 -> hold_1_4
   - hold_2_4 = 0
 group_synch_hold_constr_1_2_5: group_tp_active_1 = 1 -> hold_1_5
   - hold_2_5 = 0
 group_synch_hold_constr_1_2_6: group_tp_active_1 = 1 -> hold_1_6
   - hold_2_6 = 0
 group_synch_hold_constr_1_2_7: group_tp_active_1 = 1 -> hold_1_7
   - hold_2_7 = 0
 group_synch_hold_constr_1_2_8: group_tp_active_1 = 1 -> hold_1_8
   - hold_2_8 = 0
 group_synch_hold_constr_1_2_9: group_tp_active_1 = 1 -> hold_1_9
   - hold_2_9 = 0
 group_synch_hold_constr_1_2_10: group_tp_active_1 = 1 -> hold_1_10
   - hold_2_10 = 0
 group_synch_hold_constr_1_2_11: group_tp_active_1 = 1 -> hold_1_11
   - hold_2_11 = 0
 group_synch_hold_constr_1_2_12: group_tp_active_1 = 1 -> hold_1_12
   - hold_2_12 = 0
 group_synch_hold_constr_1_2_13: group_tp_active_1 = 1 -> hold_1_13
   - hold_2_13 = 0
 group_synch_hold_constr_1_2_14: group_tp_active_1 = 1 -> hold_1_14
   - hold_2_14 = 0
 group_synch_hold_constr_1_2_15: group_tp_active_1 = 1 -> hold_1_15
   - hold_2_15 = 0
 group_synch_hold_constr_1_2_16: group_tp_active_1 = 1 -> hold_1_16
   - hold_2_16 = 0
 group_synch_hold_constr_1_2_17: group_tp_active_1 = 1 -> hold_1_17
   - hold_2_17 = 0
 group_synch_hold_constr_1_2_18: group_tp_active_1 = 1 -> hold_1_18
   - hold_2_18 = 0
 group_synch_hold_constr_1_2_19: group_tp_active_1 = 1 -> hold_1_19
   - hold_2_19 = 0
 group_synch_hold_constr_1_2_20: group_tp_active_1 = 1 -> hold_1_20
   - hold_2_20 = 0
 group_synch_hold_constr_1_2_21: group_tp_active_1 = 1 -> hold_1_21
   - hold_2_21 = 0
 group_synch_hold_constr_1_2_22: group_tp_active_1 = 1 -> hold_1_22
   - hold_2_22 = 0
 group_synch_hold_constr_1_2_23: group_tp_active_1 = 1 -> hold_1_23
   - hold_2_23 = 0
 group_synch_hold_constr_1_2_24: group_tp_active_1 = 1 -> hold_1_24
   - hold_2_24 = 0
 group_synch_hold_constr_1_2_25: group_tp_active_1 = 1 -> hold_1_25
   - hold_2_25 = 0
 group_synch_hold_constr_1_2_26: group_tp_active_1 = 1 -> hold_1_26
   - hold_2_26 = 0
 tp_off_limit_constr_1_1: group_tp_active_1 = 0 -> hold_1_1 = 0
 tp_off_limit_constr_1_2: group_tp_active_1 = 0 -> hold_1_2 = 0
 tp_off_limit_constr_1_3: group_tp_active_1 = 0 -> hold_1_3 = 0
 tp_off_limit_constr_1_4: group_tp_active_1 = 0 -> hold_1_4 = 0
 tp_off_limit_constr_1_5: group_tp_active_1 = 0 -> hold_1_5 = 0
 tp_off_limit_constr_1_6: group_tp_active_1 = 0 -> hold_1_6 = 0
 tp_off_limit_constr_1_7: group_tp_active_1 = 0 -> hold_1_7 = 0
 tp_off_limit_constr_1_8: group_tp_active_1 = 0 -> hold_1_8 = 0
 tp_off_limit_constr_1_9: group_tp_active_1 = 0 -> hold_1_9 = 0
 tp_off_limit_constr_1_10: group_tp_active_1 = 0 -> hold_1_10 = 0
 tp_off_limit_constr_1_11: group_tp_active_1 = 0 -> hold_1_11 = 0
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
 tp_off_limit_constr_2_1: group_tp_active_1 = 0 -> hold_2_1 = 0
 tp_off_limit_constr_2_2: group_tp_active_1 = 0 -> hold_2_2 = 0
 tp_off_limit_constr_2_3: group_tp_active_1 = 0 -> hold_2_3 = 0
 tp_off_limit_constr_2_4: group_tp_active_1 = 0 -> hold_2_4 = 0
 tp_off_limit_constr_2_5: group_tp_active_1 = 0 -> hold_2_5 = 0
 tp_off_limit_constr_2_6: group_tp_active_1 = 0 -> hold_2_6 = 0
 tp_off_limit_constr_2_7: group_tp_active_1 = 0 -> hold_2_7 = 0
 tp_off_limit_constr_2_8: group_tp_active_1 = 0 -> hold_2_8 = 0
 tp_off_limit_constr_2_9: group_tp_active_1 = 0 -> hold_2_9 = 0
 tp_off_limit_constr_2_10: group_tp_active_1 = 0 -> hold_2_10 = 0
 tp_off_limit_constr_2_11: group_tp_active_1 = 0 -> hold_2_11 = 0
 tp_off_limit_constr_2_12: group_tp_active_1 = 0 -> hold_2_12 = 0
 tp_off_limit_constr_2_13: group_tp_active_1 = 0 -> hold_2_13 = 0
 tp_off_limit_constr_2_14: group_tp_active_1 = 0 -> hold_2_14 = 0
 tp_off_limit_constr_2_15: group_tp_active_1 = 0 -> hold_2_15 = 0
 tp_off_limit_constr_2_16: group_tp_active_1 = 0 -> hold_2_16 = 0
 tp_off_limit_constr_2_17: group_tp_active_1 = 0 -> hold_2_17 = 0
 tp_off_limit_constr_2_18: group_tp_active_1 = 0 -> hold_2_18 = 0
 tp_off_limit_constr_2_19: group_tp_active_1 = 0 -> hold_2_19 = 0
 tp_off_limit_constr_2_20: group_tp_active_1 = 0 -> hold_2_20 = 0
 tp_off_limit_constr_2_21: group_tp_active_1 = 0 -> hold_2_21 = 0
 tp_off_limit_constr_2_22: group_tp_active_1 = 0 -> hold_2_22 = 0
 tp_off_limit_constr_2_23: group_tp_active_1 = 0 -> hold_2_23 = 0
 tp_off_limit_constr_2_24: group_tp_active_1 = 0 -> hold_2_24 = 0
 tp_off_limit_constr_2_25: group_tp_active_1 = 0 -> hold_2_25 = 0
 tp_off_limit_constr_2_26: group_tp_active_1 = 0 -> hold_2_26 = 0
 tp_off_limit_constr_3_23: group_tp_active_3 = 0 -> hold_3_23 = 0
 tp_off_limit_constr_3_24: group_tp_active_3 = 0 -> hold_3_24 = 0
 tp_off_limit_constr_3_25: group_tp_active_3 = 0 -> hold_3_25 = 0
 tp_off_limit_constr_3_26: group_tp_active_3 = 0 -> hold_3_26 = 0
 tp_off_limit_constr_4_23: group_tp_active_4 = 0 -> hold_4_23 = 0
 tp_off_limit_constr_4_24: group_tp_active_4 = 0 -> hold_4_24 = 0
 tp_off_limit_constr_4_25: group_tp_active_4 = 0 -> hold_4_25 = 0
 tp_off_limit_constr_4_26: group_tp_active_4 = 0 -> hold_4_26 = 0
 group_throughput_constr_0: group_tp_active_0 = 1 ->
   - 3510.857142857143 hold_0_1 - 1755.428571428571 hold_0_2
   - 1170.285714285714 hold_0_3 - 877.7142857142857 hold_0_4
   - 702.1714285714286 hold_0_5 - 585.1428571428571 hold_0_6
   - 501.5510204081633 hold_0_7 - 438.8571428571428 hold_0_8
   - 390.0952380952381 hold_0_9 - 351.0857142857143 hold_0_10
   - 319.1688311688312 hold_0_11 - 292.5714285714286 hold_0_12
   - 270.0659340659341 hold_0_13 + flow_nic_in_0_0 <= 0
 group_throughput_constr_1: group_tp_active_1 = 1 ->
   - 6709.636764727775 hold_1_1 - 3354.818382363887 hold_1_2
   - 2236.545588242592 hold_1_3 - 1677.409191181944 hold_1_4
   - 1341.927352945555 hold_1_5 - 1118.272794121296 hold_1_6
   - 958.5195378182536 hold_1_7 - 838.7045955909718 hold_1_8
   - 745.5151960808639 hold_1_9 - 670.9636764727776 hold_1_10
   - 609.9669786116159 hold_1_11 - 559.1363970606479 hold_1_12
   - 516.1259049790597 hold_1_13 - 479.2597689091268 hold_1_14
   - 447.3091176485183 hold_1_15 - 419.3522977954859 hold_1_16
   - 394.684515572222 hold_1_17 - 372.757598040432 hold_1_18
   - 353.1387770909355 hold_1_19 - 335.4818382363888 hold_1_20
   - 319.5065126060845 hold_1_21 - 304.983489305808 hold_1_22
   - 291.7233375968598 hold_1_23 - 279.568198530324 hold_1_24
   - 268.385470589111 hold_1_25 - 258.0629524895298 hold_1_26
   + flow_nic_in_1_1 + flow_nic_in_1_2 <= 0
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
 group_throughput_constr_4: group_tp_active_4 = 1 ->
   - 9325.714285714286 hold_4_1 - 4662.857142857143 hold_4_2
   - 3108.571428571428 hold_4_3 - 2331.428571428572 hold_4_4
   - 1865.142857142857 hold_4_5 - 1554.285714285714 hold_4_6
   - 1332.244897959184 hold_4_7 - 1165.714285714286 hold_4_8
   - 1036.190476190476 hold_4_9 - 932.5714285714284 hold_4_10
   - 847.7922077922078 hold_4_11 - 777.1428571428571 hold_4_12
   - 717.3626373626373 hold_4_13 - 666.1224489795918 hold_4_14
   - 621.7142857142858 hold_4_15 - 582.8571428571429 hold_4_16
   - 548.5714285714286 hold_4_17 - 518.0952380952381 hold_4_18
   - 490.8270676691729 hold_4_19 - 466.2857142857142 hold_4_20
   - 444.0816326530613 hold_4_21 - 423.8961038961039 hold_4_22
   + flow_nic_in_4_4 <= 0
 intergroup_group_tp_1_via_nic_out_0_nic_in_1: tp_on_and_switch_nic_out_0_nic_in_1_1 = 1 ->
   - 4727.252546918105 hold_1_1 - 2773.319442084394 hold_1_2
   - 1962.253719528576 hold_1_3 - 1518.239848874405 hold_1_4
   - 1238.088316837634 hold_1_5 - 1045.220211240311 hold_1_6
   - 904.3426562614383 hold_1_7 - 796.9302776989886 hold_1_8
   - 712.3246432027078 hold_1_9 - 643.9591034989182 hold_1_10
   - 587.5671920572128 hold_1_11 - 540.2565773030514 hold_1_12
   - 499.9970651213938 hold_1_13 - 465.3216473359614 hold_1_14
   - 435.1438786236861 hold_1_15 - 408.6419880051317 hold_1_16
   - 385.1829074940356 hold_1_17 - 364.2710479852658 hold_1_18
   - 345.5129018638517 hold_1_19 - 328.5920467444093 hold_1_20
   - 313.2511511759551 hold_1_21 - 299.2787960866978 hold_1_22
   - 286.499672553257 hold_1_23 - 274.7671880442751 hold_1_24
   - 263.9578178909646 hold_1_25 - 253.9667395739921 hold_1_26
   + flow_nic_in_1_1 + flow_nic_in_1_2 <= 0
 intergroup_group_tp_1_via_source_nic_in_1: tp_on_and_switch_source_nic_in_1_1 = 1 ->
   - 6190.441584188167 hold_1_1 - 3219.795517494591 hold_1_2
   - 2175.719393118431 hold_1_3 - 1642.960233722046 hold_1_4
   - 1319.789089454823 hold_1_5 - 1102.856612304337 hold_1_6
   - 947.1710137886099 hold_1_7 - 830.0030039192051 hold_1_8
   - 738.6319294841023 hold_1_9 - 665.3830780191463 hold_1_10
   - 605.3514238738835 hold_1_11 - 555.2556018523049 hold_1_12
   - 512.8174255062044 hold_1_13 - 476.405742582911 hold_1_14
   - 444.8219561893465 hold_1_15 - 417.1655561140166 hold_1_16
   - 392.7468766876226 hold_1_17 - 371.0288004845265 hold_1_18
   - 351.5867904755622 hold_1_19 - 334.0808624631546 hold_1_20
   - 318.2355335372368 hold_1_21 - 303.8252183653559 hold_1_22
   - 290.663421304606 hold_1_23 - 278.5946210761428 hold_1_24
   - 267.4880966055551 hold_1_25 - 257.2331730879632 hold_1_26
   + flow_nic_in_1_1 + flow_nic_in_1_2 <= 0
 intergroup_group_tp_1_via_nic_out_3_nic_in_1: tp_on_and_switch_nic_out_3_nic_in_1_1 = 1 ->
   - 4727.252546918105 hold_1_1 - 2773.319442084394 hold_1_2
   - 1962.253719528576 hold_1_3 - 1518.239848874405 hold_1_4
   - 1238.088316837634 hold_1_5 - 1045.220211240311 hold_1_6
   - 904.3426562614383 hold_1_7 - 796.9302776989886 hold_1_8
   - 712.3246432027078 hold_1_9 - 643.9591034989182 hold_1_10
   - 587.5671920572128 hold_1_11 - 540.2565773030514 hold_1_12
   - 499.9970651213938 hold_1_13 - 465.3216473359614 hold_1_14
   - 435.1438786236861 hold_1_15 - 408.6419880051317 hold_1_16
   - 385.1829074940356 hold_1_17 - 364.2710479852658 hold_1_18
   - 345.5129018638517 hold_1_19 - 328.5920467444093 hold_1_20
   - 313.2511511759551 hold_1_21 - 299.2787960866978 hold_1_22
   - 286.499672553257 hold_1_23 - 274.7671880442751 hold_1_24
   - 263.9578178909646 hold_1_25 - 253.9667395739921 hold_1_26
   + flow_nic_in_1_1 + flow_nic_in_1_2 <= 0
 intergroup_group_tp_1_via_nic_out_4_nic_in_1: tp_on_and_switch_nic_out_4_nic_in_1_1 = 1 ->
   - 4727.252546918105 hold_1_1 - 2773.319442084394 hold_1_2
   - 1962.253719528576 hold_1_3 - 1518.239848874405 hold_1_4
   - 1238.088316837634 hold_1_5 - 1045.220211240311 hold_1_6
   - 904.3426562614383 hold_1_7 - 796.9302776989886 hold_1_8
   - 712.3246432027078 hold_1_9 - 643.9591034989182 hold_1_10
   - 587.5671920572128 hold_1_11 - 540.2565773030514 hold_1_12
   - 499.9970651213938 hold_1_13 - 465.3216473359614 hold_1_14
   - 435.1438786236861 hold_1_15 - 408.6419880051317 hold_1_16
   - 385.1829074940356 hold_1_17 - 364.2710479852658 hold_1_18
   - 345.5129018638517 hold_1_19 - 328.5920467444093 hold_1_20
   - 313.2511511759551 hold_1_21 - 299.2787960866978 hold_1_22
   - 286.499672553257 hold_1_23 - 274.7671880442751 hold_1_24
   - 263.9578178909646 hold_1_25 - 253.9667395739921 hold_1_26
   + flow_nic_in_1_1 + flow_nic_in_1_2 <= 0
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
 hold_4_21 hold_4_22 hold_4_23 hold_4_24 hold_4_25 hold_4_26
 switch_source_nic_in_0 switch_nic_out_0_sink switch_nic_out_0_nic_in_1
 switch_nic_out_0_nic_in_3 switch_nic_out_0_nic_in_4 switch_nic_in_0_0
 switch_0_nic_out_0 switch_source_nic_in_1 switch_nic_out_1_sink
 switch_nic_out_1_nic_in_0 switch_nic_out_1_nic_in_3
 switch_nic_out_1_nic_in_4 switch_nic_in_1_1 switch_1_nic_out_1
 switch_nic_in_1_2 switch_2_nic_out_1 switch_1_2 switch_2_1
 switch_source_nic_in_3 switch_nic_out_3_sink switch_nic_out_3_nic_in_0
 switch_nic_out_3_nic_in_1 switch_nic_out_3_nic_in_4 switch_nic_in_3_3
 switch_3_nic_out_3 switch_source_nic_in_4 switch_nic_out_4_sink
 switch_nic_out_4_nic_in_0 switch_nic_out_4_nic_in_1
 switch_nic_out_4_nic_in_3 switch_nic_in_4_4 switch_4_nic_out_4
 group_tp_active_0 group_tp_active_1 group_tp_active_3 group_tp_active_4
 is_active_0 is_active_1 is_active_2 is_active_3 is_active_4
 is_start_zero_0_for_source_nic_in_0 is_end_complete_0
 is_start_zero_1_for_source_nic_in_1 is_start_zero_2_for_source_nic_in_1
 is_end_complete_1 is_end_complete_2 pp_permission_1_2 pp_permission_2_1
 is_start_zero_3_for_source_nic_in_3 is_end_complete_3
 is_start_zero_4_for_source_nic_in_4 is_end_complete_4
 tp_on_and_switch_nic_out_0_nic_in_1_1 tp_on_and_switch_source_nic_in_1_1
 tp_on_and_switch_nic_out_3_nic_in_1_1
 tp_on_and_switch_nic_out_4_nic_in_1_1
Generals
 start_0 start_1 start_2 start_3 start_4 group_start_1 val_end_0
 group_end_0 group_start_3 group_start_4 group_start_0 val_end_1 val_end_2
 group_end_1 val_end_3 group_end_3 val_end_4 group_end_4
General Constraints
 min_start_1: group_start_1 = MIN ( start_1 , start_2 )
 max_end_0: group_end_0 = MAX ( val_end_0 )
 min_start_3: group_start_3 = MIN ( start_3 )
 min_start_4: group_start_4 = MIN ( start_4 )
 min_start_0: group_start_0 = MIN ( start_0 )
 max_end_1: group_end_1 = MAX ( val_end_1 , val_end_2 )
 max_end_3: group_end_3 = MAX ( val_end_3 )
 max_end_4: group_end_4 = MAX ( val_end_4 )
End
