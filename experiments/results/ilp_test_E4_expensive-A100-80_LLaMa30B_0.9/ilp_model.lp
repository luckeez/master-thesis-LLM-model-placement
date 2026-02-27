\ Model 2026-02-27_11-24-12
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  8.2720588235294095e-04 flow_source_nic_in_0
   + 8.2720588235294095e-04 flow_source_nic_in_1
   + 8.2720588235294095e-04 flow_source_nic_in_2
   + 8.2720588235294095e-04 flow_source_nic_in_3 - 0.025 is_active_0
   - 0.025 is_active_1 - 0.025 is_active_2 - 0.025 is_active_3
   + 0 val_end_0 + 0 val_end_1 + 0 val_end_2 + 0 val_end_3
Subject To
 mem_active_constr_0: hold_0_1 + hold_0_2 + hold_0_3 + hold_0_4 + hold_0_5
   + hold_0_6 + hold_0_7 + hold_0_8 + hold_0_9 + hold_0_10 + hold_0_11
   + hold_0_12 + hold_0_13 + hold_0_14 + hold_0_15 + hold_0_16 + hold_0_17
   + hold_0_18 + hold_0_19 + hold_0_20 + hold_0_21 + hold_0_22 + hold_0_23
   + hold_0_24 + hold_0_25 + hold_0_26 + hold_0_27 + hold_0_28 + hold_0_29
   + hold_0_30 + hold_0_31 + hold_0_32 + hold_0_33 + hold_0_34 + hold_0_35
   + hold_0_36 + hold_0_37 + hold_0_38 + hold_0_39 + hold_0_40
   - 61 is_active_0 <= 0
 mem_active_constr2_0: hold_0_1 + hold_0_2 + hold_0_3 + hold_0_4 + hold_0_5
   + hold_0_6 + hold_0_7 + hold_0_8 + hold_0_9 + hold_0_10 + hold_0_11
   + hold_0_12 + hold_0_13 + hold_0_14 + hold_0_15 + hold_0_16 + hold_0_17
   + hold_0_18 + hold_0_19 + hold_0_20 + hold_0_21 + hold_0_22 + hold_0_23
   + hold_0_24 + hold_0_25 + hold_0_26 + hold_0_27 + hold_0_28 + hold_0_29
   + hold_0_30 + hold_0_31 + hold_0_32 + hold_0_33 + hold_0_34 + hold_0_35
   + hold_0_36 + hold_0_37 + hold_0_38 + hold_0_39 + hold_0_40
   - is_active_0 >= 0
 mem_active_constr_1: hold_1_1 + hold_1_2 + hold_1_3 + hold_1_4 + hold_1_5
   + hold_1_6 + hold_1_7 + hold_1_8 + hold_1_9 + hold_1_10 + hold_1_11
   + hold_1_12 + hold_1_13 + hold_1_14 + hold_1_15 + hold_1_16 + hold_1_17
   + hold_1_18 + hold_1_19 + hold_1_20 + hold_1_21 + hold_1_22 + hold_1_23
   + hold_1_24 + hold_1_25 + hold_1_26 + hold_1_27 + hold_1_28 + hold_1_29
   + hold_1_30 + hold_1_31 + hold_1_32 + hold_1_33 + hold_1_34 + hold_1_35
   + hold_1_36 + hold_1_37 + hold_1_38 + hold_1_39 + hold_1_40
   - 61 is_active_1 <= 0
 mem_active_constr2_1: hold_1_1 + hold_1_2 + hold_1_3 + hold_1_4 + hold_1_5
   + hold_1_6 + hold_1_7 + hold_1_8 + hold_1_9 + hold_1_10 + hold_1_11
   + hold_1_12 + hold_1_13 + hold_1_14 + hold_1_15 + hold_1_16 + hold_1_17
   + hold_1_18 + hold_1_19 + hold_1_20 + hold_1_21 + hold_1_22 + hold_1_23
   + hold_1_24 + hold_1_25 + hold_1_26 + hold_1_27 + hold_1_28 + hold_1_29
   + hold_1_30 + hold_1_31 + hold_1_32 + hold_1_33 + hold_1_34 + hold_1_35
   + hold_1_36 + hold_1_37 + hold_1_38 + hold_1_39 + hold_1_40
   - is_active_1 >= 0
 mem_active_constr_2: hold_2_1 + hold_2_2 + hold_2_3 + hold_2_4 + hold_2_5
   + hold_2_6 + hold_2_7 + hold_2_8 + hold_2_9 + hold_2_10 + hold_2_11
   + hold_2_12 + hold_2_13 + hold_2_14 + hold_2_15 + hold_2_16 + hold_2_17
   + hold_2_18 + hold_2_19 + hold_2_20 + hold_2_21 + hold_2_22 + hold_2_23
   + hold_2_24 + hold_2_25 + hold_2_26 + hold_2_27 + hold_2_28 + hold_2_29
   + hold_2_30 + hold_2_31 + hold_2_32 + hold_2_33 + hold_2_34 + hold_2_35
   + hold_2_36 + hold_2_37 + hold_2_38 + hold_2_39 + hold_2_40
   - 61 is_active_2 <= 0
 mem_active_constr2_2: hold_2_1 + hold_2_2 + hold_2_3 + hold_2_4 + hold_2_5
   + hold_2_6 + hold_2_7 + hold_2_8 + hold_2_9 + hold_2_10 + hold_2_11
   + hold_2_12 + hold_2_13 + hold_2_14 + hold_2_15 + hold_2_16 + hold_2_17
   + hold_2_18 + hold_2_19 + hold_2_20 + hold_2_21 + hold_2_22 + hold_2_23
   + hold_2_24 + hold_2_25 + hold_2_26 + hold_2_27 + hold_2_28 + hold_2_29
   + hold_2_30 + hold_2_31 + hold_2_32 + hold_2_33 + hold_2_34 + hold_2_35
   + hold_2_36 + hold_2_37 + hold_2_38 + hold_2_39 + hold_2_40
   - is_active_2 >= 0
 mem_active_constr_3: hold_3_1 + hold_3_2 + hold_3_3 + hold_3_4 + hold_3_5
   + hold_3_6 + hold_3_7 + hold_3_8 + hold_3_9 + hold_3_10 + hold_3_11
   + hold_3_12 + hold_3_13 + hold_3_14 + hold_3_15 + hold_3_16 + hold_3_17
   + hold_3_18 + hold_3_19 + hold_3_20 + hold_3_21 + hold_3_22 + hold_3_23
   + hold_3_24 + hold_3_25 + hold_3_26 + hold_3_27 + hold_3_28 + hold_3_29
   + hold_3_30 + hold_3_31 + hold_3_32 + hold_3_33 + hold_3_34 + hold_3_35
   + hold_3_36 + hold_3_37 + hold_3_38 + hold_3_39 + hold_3_40
   - 61 is_active_3 <= 0
 mem_active_constr2_3: hold_3_1 + hold_3_2 + hold_3_3 + hold_3_4 + hold_3_5
   + hold_3_6 + hold_3_7 + hold_3_8 + hold_3_9 + hold_3_10 + hold_3_11
   + hold_3_12 + hold_3_13 + hold_3_14 + hold_3_15 + hold_3_16 + hold_3_17
   + hold_3_18 + hold_3_19 + hold_3_20 + hold_3_21 + hold_3_22 + hold_3_23
   + hold_3_24 + hold_3_25 + hold_3_26 + hold_3_27 + hold_3_28 + hold_3_29
   + hold_3_30 + hold_3_31 + hold_3_32 + hold_3_33 + hold_3_34 + hold_3_35
   + hold_3_36 + hold_3_37 + hold_3_38 + hold_3_39 + hold_3_40
   - is_active_3 >= 0
 hold_constraint_0: hold_0_1 + hold_0_2 + hold_0_3 + hold_0_4 + hold_0_5
   + hold_0_6 + hold_0_7 + hold_0_8 + hold_0_9 + hold_0_10 + hold_0_11
   + hold_0_12 + hold_0_13 + hold_0_14 + hold_0_15 + hold_0_16 + hold_0_17
   + hold_0_18 + hold_0_19 + hold_0_20 + hold_0_21 + hold_0_22 + hold_0_23
   + hold_0_24 + hold_0_25 + hold_0_26 + hold_0_27 + hold_0_28 + hold_0_29
   + hold_0_30 + hold_0_31 + hold_0_32 + hold_0_33 + hold_0_34 + hold_0_35
   + hold_0_36 + hold_0_37 + hold_0_38 + hold_0_39 + hold_0_40 <= 1
 end_constraint_0: start_0 + hold_0_1 + 2 hold_0_2 + 3 hold_0_3
   + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7 + 8 hold_0_8
   + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12 + 13 hold_0_13
   + 14 hold_0_14 + 15 hold_0_15 + 16 hold_0_16 + 17 hold_0_17
   + 18 hold_0_18 + 19 hold_0_19 + 20 hold_0_20 + 21 hold_0_21
   + 22 hold_0_22 + 23 hold_0_23 + 24 hold_0_24 + 25 hold_0_25
   + 26 hold_0_26 + 27 hold_0_27 + 28 hold_0_28 + 29 hold_0_29
   + 30 hold_0_30 + 31 hold_0_31 + 32 hold_0_32 + 33 hold_0_33
   + 34 hold_0_34 + 35 hold_0_35 + 36 hold_0_36 + 37 hold_0_37
   + 38 hold_0_38 + 39 hold_0_39 + 40 hold_0_40 <= 60
 hold_constraint_1: hold_1_1 + hold_1_2 + hold_1_3 + hold_1_4 + hold_1_5
   + hold_1_6 + hold_1_7 + hold_1_8 + hold_1_9 + hold_1_10 + hold_1_11
   + hold_1_12 + hold_1_13 + hold_1_14 + hold_1_15 + hold_1_16 + hold_1_17
   + hold_1_18 + hold_1_19 + hold_1_20 + hold_1_21 + hold_1_22 + hold_1_23
   + hold_1_24 + hold_1_25 + hold_1_26 + hold_1_27 + hold_1_28 + hold_1_29
   + hold_1_30 + hold_1_31 + hold_1_32 + hold_1_33 + hold_1_34 + hold_1_35
   + hold_1_36 + hold_1_37 + hold_1_38 + hold_1_39 + hold_1_40 <= 1
 end_constraint_1: start_1 + hold_1_1 + 2 hold_1_2 + 3 hold_1_3
   + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7 + 8 hold_1_8
   + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 + 12 hold_1_12 + 13 hold_1_13
   + 14 hold_1_14 + 15 hold_1_15 + 16 hold_1_16 + 17 hold_1_17
   + 18 hold_1_18 + 19 hold_1_19 + 20 hold_1_20 + 21 hold_1_21
   + 22 hold_1_22 + 23 hold_1_23 + 24 hold_1_24 + 25 hold_1_25
   + 26 hold_1_26 + 27 hold_1_27 + 28 hold_1_28 + 29 hold_1_29
   + 30 hold_1_30 + 31 hold_1_31 + 32 hold_1_32 + 33 hold_1_33
   + 34 hold_1_34 + 35 hold_1_35 + 36 hold_1_36 + 37 hold_1_37
   + 38 hold_1_38 + 39 hold_1_39 + 40 hold_1_40 <= 60
 hold_constraint_2: hold_2_1 + hold_2_2 + hold_2_3 + hold_2_4 + hold_2_5
   + hold_2_6 + hold_2_7 + hold_2_8 + hold_2_9 + hold_2_10 + hold_2_11
   + hold_2_12 + hold_2_13 + hold_2_14 + hold_2_15 + hold_2_16 + hold_2_17
   + hold_2_18 + hold_2_19 + hold_2_20 + hold_2_21 + hold_2_22 + hold_2_23
   + hold_2_24 + hold_2_25 + hold_2_26 + hold_2_27 + hold_2_28 + hold_2_29
   + hold_2_30 + hold_2_31 + hold_2_32 + hold_2_33 + hold_2_34 + hold_2_35
   + hold_2_36 + hold_2_37 + hold_2_38 + hold_2_39 + hold_2_40 <= 1
 end_constraint_2: start_2 + hold_2_1 + 2 hold_2_2 + 3 hold_2_3
   + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7 + 8 hold_2_8
   + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 12 hold_2_12 + 13 hold_2_13
   + 14 hold_2_14 + 15 hold_2_15 + 16 hold_2_16 + 17 hold_2_17
   + 18 hold_2_18 + 19 hold_2_19 + 20 hold_2_20 + 21 hold_2_21
   + 22 hold_2_22 + 23 hold_2_23 + 24 hold_2_24 + 25 hold_2_25
   + 26 hold_2_26 + 27 hold_2_27 + 28 hold_2_28 + 29 hold_2_29
   + 30 hold_2_30 + 31 hold_2_31 + 32 hold_2_32 + 33 hold_2_33
   + 34 hold_2_34 + 35 hold_2_35 + 36 hold_2_36 + 37 hold_2_37
   + 38 hold_2_38 + 39 hold_2_39 + 40 hold_2_40 <= 60
 hold_constraint_3: hold_3_1 + hold_3_2 + hold_3_3 + hold_3_4 + hold_3_5
   + hold_3_6 + hold_3_7 + hold_3_8 + hold_3_9 + hold_3_10 + hold_3_11
   + hold_3_12 + hold_3_13 + hold_3_14 + hold_3_15 + hold_3_16 + hold_3_17
   + hold_3_18 + hold_3_19 + hold_3_20 + hold_3_21 + hold_3_22 + hold_3_23
   + hold_3_24 + hold_3_25 + hold_3_26 + hold_3_27 + hold_3_28 + hold_3_29
   + hold_3_30 + hold_3_31 + hold_3_32 + hold_3_33 + hold_3_34 + hold_3_35
   + hold_3_36 + hold_3_37 + hold_3_38 + hold_3_39 + hold_3_40 <= 1
 end_constraint_3: start_3 + hold_3_1 + 2 hold_3_2 + 3 hold_3_3
   + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7 + 8 hold_3_8
   + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11 + 12 hold_3_12 + 13 hold_3_13
   + 14 hold_3_14 + 15 hold_3_15 + 16 hold_3_16 + 17 hold_3_17
   + 18 hold_3_18 + 19 hold_3_19 + 20 hold_3_20 + 21 hold_3_21
   + 22 hold_3_22 + 23 hold_3_23 + 24 hold_3_24 + 25 hold_3_25
   + 26 hold_3_26 + 27 hold_3_27 + 28 hold_3_28 + 29 hold_3_29
   + 30 hold_3_30 + 31 hold_3_31 + 32 hold_3_32 + 33 hold_3_33
   + 34 hold_3_34 + 35 hold_3_35 + 36 hold_3_36 + 37 hold_3_37
   + 38 hold_3_38 + 39 hold_3_39 + 40 hold_3_40 <= 60
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
 constr_is_start_zero_0: start_0 + 60 is_start_zero_0_for_source_nic_in_0
   <= 60
 constr_edge_switch_start_zero_0_OR: switch_source_nic_in_0
   - is_start_zero_0_for_source_nic_in_0 <= 0
 constr_is_end_complete_0: start_0 + hold_0_1 + 2 hold_0_2 + 3 hold_0_3
   + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7 + 8 hold_0_8
   + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12 + 13 hold_0_13
   + 14 hold_0_14 + 15 hold_0_15 + 16 hold_0_16 + 17 hold_0_17
   + 18 hold_0_18 + 19 hold_0_19 + 20 hold_0_20 + 21 hold_0_21
   + 22 hold_0_22 + 23 hold_0_23 + 24 hold_0_24 + 25 hold_0_25
   + 26 hold_0_26 + 27 hold_0_27 + 28 hold_0_28 + 29 hold_0_29
   + 30 hold_0_30 + 31 hold_0_31 + 32 hold_0_32 + 33 hold_0_33
   + 34 hold_0_34 + 35 hold_0_35 + 36 hold_0_36 + 37 hold_0_37
   + 38 hold_0_38 + 39 hold_0_39 + 40 hold_0_40 - 60 is_end_complete_0
   >= 0
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
 constr_is_start_zero_1: start_1 + 60 is_start_zero_1_for_source_nic_in_1
   <= 60
 constr_edge_switch_start_zero_1_OR: switch_source_nic_in_1
   - is_start_zero_1_for_source_nic_in_1 <= 0
 constr_is_end_complete_1: start_1 + hold_1_1 + 2 hold_1_2 + 3 hold_1_3
   + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7 + 8 hold_1_8
   + 9 hold_1_9 + 10 hold_1_10 + 11 hold_1_11 + 12 hold_1_12 + 13 hold_1_13
   + 14 hold_1_14 + 15 hold_1_15 + 16 hold_1_16 + 17 hold_1_17
   + 18 hold_1_18 + 19 hold_1_19 + 20 hold_1_20 + 21 hold_1_21
   + 22 hold_1_22 + 23 hold_1_23 + 24 hold_1_24 + 25 hold_1_25
   + 26 hold_1_26 + 27 hold_1_27 + 28 hold_1_28 + 29 hold_1_29
   + 30 hold_1_30 + 31 hold_1_31 + 32 hold_1_32 + 33 hold_1_33
   + 34 hold_1_34 + 35 hold_1_35 + 36 hold_1_36 + 37 hold_1_37
   + 38 hold_1_38 + 39 hold_1_39 + 40 hold_1_40 - 60 is_end_complete_1
   >= 0
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
 constr_is_start_zero_2: start_2 + 60 is_start_zero_2_for_source_nic_in_2
   <= 60
 constr_edge_switch_start_zero_2_OR: switch_source_nic_in_2
   - is_start_zero_2_for_source_nic_in_2 <= 0
 constr_is_end_complete_2: start_2 + hold_2_1 + 2 hold_2_2 + 3 hold_2_3
   + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7 + 8 hold_2_8
   + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 12 hold_2_12 + 13 hold_2_13
   + 14 hold_2_14 + 15 hold_2_15 + 16 hold_2_16 + 17 hold_2_17
   + 18 hold_2_18 + 19 hold_2_19 + 20 hold_2_20 + 21 hold_2_21
   + 22 hold_2_22 + 23 hold_2_23 + 24 hold_2_24 + 25 hold_2_25
   + 26 hold_2_26 + 27 hold_2_27 + 28 hold_2_28 + 29 hold_2_29
   + 30 hold_2_30 + 31 hold_2_31 + 32 hold_2_32 + 33 hold_2_33
   + 34 hold_2_34 + 35 hold_2_35 + 36 hold_2_36 + 37 hold_2_37
   + 38 hold_2_38 + 39 hold_2_39 + 40 hold_2_40 - 60 is_end_complete_2
   >= 0
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
 constr_is_start_zero_3: start_3 + 60 is_start_zero_3_for_source_nic_in_3
   <= 60
 constr_edge_switch_start_zero_3_OR: switch_source_nic_in_3
   - is_start_zero_3_for_source_nic_in_3 <= 0
 constr_is_end_complete_3: start_3 + hold_3_1 + 2 hold_3_2 + 3 hold_3_3
   + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7 + 8 hold_3_8
   + 9 hold_3_9 + 10 hold_3_10 + 11 hold_3_11 + 12 hold_3_12 + 13 hold_3_13
   + 14 hold_3_14 + 15 hold_3_15 + 16 hold_3_16 + 17 hold_3_17
   + 18 hold_3_18 + 19 hold_3_19 + 20 hold_3_20 + 21 hold_3_21
   + 22 hold_3_22 + 23 hold_3_23 + 24 hold_3_24 + 25 hold_3_25
   + 26 hold_3_26 + 27 hold_3_27 + 28 hold_3_28 + 29 hold_3_29
   + 30 hold_3_30 + 31 hold_3_31 + 32 hold_3_32 + 33 hold_3_33
   + 34 hold_3_34 + 35 hold_3_35 + 36 hold_3_36 + 37 hold_3_37
   + 38 hold_3_38 + 39 hold_3_39 + 40 hold_3_40 - 60 is_end_complete_3
   >= 0
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
   - 75120.19230769231 switch_nic_out_0_nic_in_1 <= 0
 edge_flow_constr_nic_out_0_nic_in_2: flow_nic_out_0_nic_in_2
   - 75120.19230769231 switch_nic_out_0_nic_in_2 <= 0
 edge_flow_constr_nic_out_0_nic_in_3: flow_nic_out_0_nic_in_3
   - 75120.19230769231 switch_nic_out_0_nic_in_3 <= 0
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
 edge_flow_constr_nic_in_3_3: flow_nic_in_3_3
   - 4.807692307692308e+06 switch_nic_in_3_3 <= 0
 edge_flow_constr_3_nic_out_3: flow_3_nic_out_3
   - 4.807692307692308e+06 switch_3_nic_out_3 <= 0
 force_group_off_constr_0: group_tp_active_0 = 0
 force_group_off_constr_1: group_tp_active_1 = 0
 force_group_off_constr_2: group_tp_active_2 = 0
 force_group_off_constr_3: group_tp_active_3 = 0
 force_min_throughput: flow_source_nic_in_0 + flow_source_nic_in_1
   + flow_source_nic_in_2 + flow_source_nic_in_3 >= 1
 inactive_start_invalid_0: is_active_0 = 0 -> start_0 = 60
 inactive_start_invalid_1: is_active_1 = 0 -> start_1 = 60
 inactive_start_invalid_2: is_active_2 = 0 -> start_2 = 60
 inactive_start_invalid_3: is_active_3 = 0 -> start_3 = 60
 node_throughput_constraint_0: group_tp_active_0 = 0 -> - 16320 hold_0_1
   - 8160 hold_0_2 - 5440 hold_0_3 - 4080 hold_0_4 - 3264 hold_0_5
   - 2720 hold_0_6 - 2331.428571428572 hold_0_7 - 2040 hold_0_8
   - 1813.333333333333 hold_0_9 - 1632 hold_0_10
   - 1483.636363636364 hold_0_11 - 1360 hold_0_12
   - 1255.384615384615 hold_0_13 - 1165.714285714286 hold_0_14
   - 1088 hold_0_15 - 1020 hold_0_16 - 960 hold_0_17
   - 906.6666666666666 hold_0_18 - 858.9473684210526 hold_0_19
   - 816 hold_0_20 - 777.1428571428571 hold_0_21
   - 741.8181818181819 hold_0_22 - 709.5652173913044 hold_0_23
   - 680 hold_0_24 - 652.8 hold_0_25 - 627.6923076923077 hold_0_26
   - 604.4444444444445 hold_0_27 - 582.8571428571429 hold_0_28
   - 562.7586206896552 hold_0_29 - 544 hold_0_30
   - 526.4516129032257 hold_0_31 - 510 hold_0_32
   - 494.5454545454545 hold_0_33 - 480 hold_0_34
   - 466.2857142857142 hold_0_35 - 453.3333333333333 hold_0_36
   - 441.081081081081 hold_0_37 - 429.4736842105263 hold_0_38
   - 418.4615384615385 hold_0_39 - 408 hold_0_40 + flow_nic_in_0_0 <= 0
 node_throughput_constraint_1: group_tp_active_1 = 0 -> - 16320 hold_1_1
   - 8160 hold_1_2 - 5440 hold_1_3 - 4080 hold_1_4 - 3264 hold_1_5
   - 2720 hold_1_6 - 2331.428571428572 hold_1_7 - 2040 hold_1_8
   - 1813.333333333333 hold_1_9 - 1632 hold_1_10
   - 1483.636363636364 hold_1_11 - 1360 hold_1_12
   - 1255.384615384615 hold_1_13 - 1165.714285714286 hold_1_14
   - 1088 hold_1_15 - 1020 hold_1_16 - 960 hold_1_17
   - 906.6666666666666 hold_1_18 - 858.9473684210526 hold_1_19
   - 816 hold_1_20 - 777.1428571428571 hold_1_21
   - 741.8181818181819 hold_1_22 - 709.5652173913044 hold_1_23
   - 680 hold_1_24 - 652.8 hold_1_25 - 627.6923076923077 hold_1_26
   - 604.4444444444445 hold_1_27 - 582.8571428571429 hold_1_28
   - 562.7586206896552 hold_1_29 - 544 hold_1_30
   - 526.4516129032257 hold_1_31 - 510 hold_1_32
   - 494.5454545454545 hold_1_33 - 480 hold_1_34
   - 466.2857142857142 hold_1_35 - 453.3333333333333 hold_1_36
   - 441.081081081081 hold_1_37 - 429.4736842105263 hold_1_38
   - 418.4615384615385 hold_1_39 - 408 hold_1_40 + flow_nic_in_1_1 <= 0
 node_throughput_constraint_2: group_tp_active_2 = 0 -> - 16320 hold_2_1
   - 8160 hold_2_2 - 5440 hold_2_3 - 4080 hold_2_4 - 3264 hold_2_5
   - 2720 hold_2_6 - 2331.428571428572 hold_2_7 - 2040 hold_2_8
   - 1813.333333333333 hold_2_9 - 1632 hold_2_10
   - 1483.636363636364 hold_2_11 - 1360 hold_2_12
   - 1255.384615384615 hold_2_13 - 1165.714285714286 hold_2_14
   - 1088 hold_2_15 - 1020 hold_2_16 - 960 hold_2_17
   - 906.6666666666666 hold_2_18 - 858.9473684210526 hold_2_19
   - 816 hold_2_20 - 777.1428571428571 hold_2_21
   - 741.8181818181819 hold_2_22 - 709.5652173913044 hold_2_23
   - 680 hold_2_24 - 652.8 hold_2_25 - 627.6923076923077 hold_2_26
   - 604.4444444444445 hold_2_27 - 582.8571428571429 hold_2_28
   - 562.7586206896552 hold_2_29 - 544 hold_2_30
   - 526.4516129032257 hold_2_31 - 510 hold_2_32
   - 494.5454545454545 hold_2_33 - 480 hold_2_34
   - 466.2857142857142 hold_2_35 - 453.3333333333333 hold_2_36
   - 441.081081081081 hold_2_37 - 429.4736842105263 hold_2_38
   - 418.4615384615385 hold_2_39 - 408 hold_2_40 + flow_nic_in_2_2 <= 0
 node_throughput_constraint_3: group_tp_active_3 = 0 -> - 16320 hold_3_1
   - 8160 hold_3_2 - 5440 hold_3_3 - 4080 hold_3_4 - 3264 hold_3_5
   - 2720 hold_3_6 - 2331.428571428572 hold_3_7 - 2040 hold_3_8
   - 1813.333333333333 hold_3_9 - 1632 hold_3_10
   - 1483.636363636364 hold_3_11 - 1360 hold_3_12
   - 1255.384615384615 hold_3_13 - 1165.714285714286 hold_3_14
   - 1088 hold_3_15 - 1020 hold_3_16 - 960 hold_3_17
   - 906.6666666666666 hold_3_18 - 858.9473684210526 hold_3_19
   - 816 hold_3_20 - 777.1428571428571 hold_3_21
   - 741.8181818181819 hold_3_22 - 709.5652173913044 hold_3_23
   - 680 hold_3_24 - 652.8 hold_3_25 - 627.6923076923077 hold_3_26
   - 604.4444444444445 hold_3_27 - 582.8571428571429 hold_3_28
   - 562.7586206896552 hold_3_29 - 544 hold_3_30
   - 526.4516129032257 hold_3_31 - 510 hold_3_32
   - 494.5454545454545 hold_3_33 - 480 hold_3_34
   - 466.2857142857142 hold_3_35 - 453.3333333333333 hold_3_36
   - 441.081081081081 hold_3_37 - 429.4736842105263 hold_3_38
   - 418.4615384615385 hold_3_39 - 408 hold_3_40 + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_0_source_nic_in_0: switch_source_nic_in_0 = 1 ->
   - 13554.81727574751 hold_0_1 - 7404.718693284936 hold_0_2
   - 5093.632958801498 hold_0_3 - 3882.017126546147 hold_0_4
   - 3136.049192928517 hold_0_5 - 2630.560928433269 hold_0_6
   - 2265.40810660744 hold_0_7 - 1989.273525109702 hold_0_8
   - 1773.142112125163 hold_0_9 - 1599.37279498236 hold_0_10
   - 1456.622634773295 hold_0_11 - 1337.266470009833 hold_0_12
   - 1235.989094213875 hold_0_13 - 1148.972120529428 hold_0_14
   - 1073.401736385162 hold_0_15 - 1007.158726240435 hold_0_16
   - 948.6166007905139 hold_0_17 - 896.5062623599209 hold_0_18
   - 849.8229535513435 hold_0_19 - 807.7608394377352 hold_0_20
   - 769.6661007357103 hold_0_21 - 735.0027022158171 hold_0_22
   - 703.3270125840373 hold_0_23 - 674.2687159147249 hold_0_24
   - 647.5162672591651 hold_0_25 - 622.805678522363 hold_0_26
   - 599.911777679753 hold_0_27 - 578.6413274712806 hold_0_28
   - 558.8275578687852 hold_0_29 - 540.3257846642829 hold_0_30
   - 523.0098705294192 hold_0_31 - 506.7693454229288 hold_0_32
   - 491.5070473436935 hold_0_33 - 477.1371769383698 hold_0_34
   - 463.5836836723099 hold_0_35 - 450.7789194564137 hold_0_36
   - 438.6625094075906 hold_0_37 - 427.1803999581196 hold_0_38
   - 416.284052647689 hold_0_39 - 405.9297582330117 hold_0_40
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_0_nic_in_1: switch_nic_out_0_nic_in_1 = 1 ->
   - 8079.207920792079 hold_1_1 - 5403.973509933775 hold_1_2
   - 4059.701492537313 hold_1_3 - 3250.996015936255 hold_1_4
   - 2710.963455149502 hold_1_5 - 2324.786324786325 hold_1_6
   - 2034.912718204489 hold_1_7 - 1809.312638580931 hold_1_8
   - 1628.74251497006 hold_1_9 - 1480.943738656987 hold_1_10
   - 1357.737104825291 hold_1_11 - 1253.456221198157 hold_1_12
   - 1164.051355206847 hold_1_13 - 1086.551264980027 hold_1_14
   - 1018.7265917603 hold_1_15 - 958.8719153936545 hold_1_16
   - 905.6603773584906 hold_1_17 - 858.0441640378549 hold_1_18
   - 815.1848151848152 hold_1_19 - 776.4034253092293 hold_1_20
   - 741.1444141689373 hold_1_21 - 708.9487402258906 hold_1_22
   - 679.4338051623647 hold_1_23 - 652.2781774580336 hold_1_24
   - 627.2098385857032 hold_1_25 - 603.9970392301998 hold_1_26
   - 582.441113490364 hold_1_27 - 562.3707787732598 hold_1_28
   - 543.6375749500334 hold_1_29 - 526.1121856866538 hold_1_30
   - 509.681449094316 hold_1_31 - 494.2459115687462 hold_1_32
   - 479.7178130511463 hold_1_33 - 466.0194174757282 hold_1_34
   - 453.081621321488 hold_1_35 - 440.8427876823339 hold_1_36
   - 429.2477643345607 hold_1_37 - 418.2470527934393 hold_1_38
   - 407.7961019490255 hold_1_39 - 397.8547050219405 hold_1_40
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_0_nic_in_2: switch_nic_out_0_nic_in_2 = 1 ->
   - 8079.207920792079 hold_2_1 - 5403.973509933775 hold_2_2
   - 4059.701492537313 hold_2_3 - 3250.996015936255 hold_2_4
   - 2710.963455149502 hold_2_5 - 2324.786324786325 hold_2_6
   - 2034.912718204489 hold_2_7 - 1809.312638580931 hold_2_8
   - 1628.74251497006 hold_2_9 - 1480.943738656987 hold_2_10
   - 1357.737104825291 hold_2_11 - 1253.456221198157 hold_2_12
   - 1164.051355206847 hold_2_13 - 1086.551264980027 hold_2_14
   - 1018.7265917603 hold_2_15 - 958.8719153936545 hold_2_16
   - 905.6603773584906 hold_2_17 - 858.0441640378549 hold_2_18
   - 815.1848151848152 hold_2_19 - 776.4034253092293 hold_2_20
   - 741.1444141689373 hold_2_21 - 708.9487402258906 hold_2_22
   - 679.4338051623647 hold_2_23 - 652.2781774580336 hold_2_24
   - 627.2098385857032 hold_2_25 - 603.9970392301998 hold_2_26
   - 582.441113490364 hold_2_27 - 562.3707787732598 hold_2_28
   - 543.6375749500334 hold_2_29 - 526.1121856866538 hold_2_30
   - 509.681449094316 hold_2_31 - 494.2459115687462 hold_2_32
   - 479.7178130511463 hold_2_33 - 466.0194174757282 hold_2_34
   - 453.081621321488 hold_2_35 - 440.8427876823339 hold_2_36
   - 429.2477643345607 hold_2_37 - 418.2470527934393 hold_2_38
   - 407.7961019490255 hold_2_39 - 397.8547050219405 hold_2_40
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_0_nic_in_3: switch_nic_out_0_nic_in_3 = 1 ->
   - 8079.207920792079 hold_3_1 - 5403.973509933775 hold_3_2
   - 4059.701492537313 hold_3_3 - 3250.996015936255 hold_3_4
   - 2710.963455149502 hold_3_5 - 2324.786324786325 hold_3_6
   - 2034.912718204489 hold_3_7 - 1809.312638580931 hold_3_8
   - 1628.74251497006 hold_3_9 - 1480.943738656987 hold_3_10
   - 1357.737104825291 hold_3_11 - 1253.456221198157 hold_3_12
   - 1164.051355206847 hold_3_13 - 1086.551264980027 hold_3_14
   - 1018.7265917603 hold_3_15 - 958.8719153936545 hold_3_16
   - 905.6603773584906 hold_3_17 - 858.0441640378549 hold_3_18
   - 815.1848151848152 hold_3_19 - 776.4034253092293 hold_3_20
   - 741.1444141689373 hold_3_21 - 708.9487402258906 hold_3_22
   - 679.4338051623647 hold_3_23 - 652.2781774580336 hold_3_24
   - 627.2098385857032 hold_3_25 - 603.9970392301998 hold_3_26
   - 582.441113490364 hold_3_27 - 562.3707787732598 hold_3_28
   - 543.6375749500334 hold_3_29 - 526.1121856866538 hold_3_30
   - 509.681449094316 hold_3_31 - 494.2459115687462 hold_3_32
   - 479.7178130511463 hold_3_33 - 466.0194174757282 hold_3_34
   - 453.081621321488 hold_3_35 - 440.8427876823339 hold_3_36
   - 429.2477643345607 hold_3_37 - 418.2470527934393 hold_3_38
   - 407.7961019490255 hold_3_39 - 397.8547050219405 hold_3_40
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_1_source_nic_in_1: switch_source_nic_in_1 = 1 ->
   - 13554.81727574751 hold_1_1 - 7404.718693284936 hold_1_2
   - 5093.632958801498 hold_1_3 - 3882.017126546147 hold_1_4
   - 3136.049192928517 hold_1_5 - 2630.560928433269 hold_1_6
   - 2265.40810660744 hold_1_7 - 1989.273525109702 hold_1_8
   - 1773.142112125163 hold_1_9 - 1599.37279498236 hold_1_10
   - 1456.622634773295 hold_1_11 - 1337.266470009833 hold_1_12
   - 1235.989094213875 hold_1_13 - 1148.972120529428 hold_1_14
   - 1073.401736385162 hold_1_15 - 1007.158726240435 hold_1_16
   - 948.6166007905139 hold_1_17 - 896.5062623599209 hold_1_18
   - 849.8229535513435 hold_1_19 - 807.7608394377352 hold_1_20
   - 769.6661007357103 hold_1_21 - 735.0027022158171 hold_1_22
   - 703.3270125840373 hold_1_23 - 674.2687159147249 hold_1_24
   - 647.5162672591651 hold_1_25 - 622.805678522363 hold_1_26
   - 599.911777679753 hold_1_27 - 578.6413274712806 hold_1_28
   - 558.8275578687852 hold_1_29 - 540.3257846642829 hold_1_30
   - 523.0098705294192 hold_1_31 - 506.7693454229288 hold_1_32
   - 491.5070473436935 hold_1_33 - 477.1371769383698 hold_1_34
   - 463.5836836723099 hold_1_35 - 450.7789194564137 hold_1_36
   - 438.6625094075906 hold_1_37 - 427.1803999581196 hold_1_38
   - 416.284052647689 hold_1_39 - 405.9297582330117 hold_1_40
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_0_nic_out_1_nic_in_0: switch_nic_out_1_nic_in_0 = 1 ->
   - 8079.207920792079 hold_0_1 - 5403.973509933775 hold_0_2
   - 4059.701492537313 hold_0_3 - 3250.996015936255 hold_0_4
   - 2710.963455149502 hold_0_5 - 2324.786324786325 hold_0_6
   - 2034.912718204489 hold_0_7 - 1809.312638580931 hold_0_8
   - 1628.74251497006 hold_0_9 - 1480.943738656987 hold_0_10
   - 1357.737104825291 hold_0_11 - 1253.456221198157 hold_0_12
   - 1164.051355206847 hold_0_13 - 1086.551264980027 hold_0_14
   - 1018.7265917603 hold_0_15 - 958.8719153936545 hold_0_16
   - 905.6603773584906 hold_0_17 - 858.0441640378549 hold_0_18
   - 815.1848151848152 hold_0_19 - 776.4034253092293 hold_0_20
   - 741.1444141689373 hold_0_21 - 708.9487402258906 hold_0_22
   - 679.4338051623647 hold_0_23 - 652.2781774580336 hold_0_24
   - 627.2098385857032 hold_0_25 - 603.9970392301998 hold_0_26
   - 582.441113490364 hold_0_27 - 562.3707787732598 hold_0_28
   - 543.6375749500334 hold_0_29 - 526.1121856866538 hold_0_30
   - 509.681449094316 hold_0_31 - 494.2459115687462 hold_0_32
   - 479.7178130511463 hold_0_33 - 466.0194174757282 hold_0_34
   - 453.081621321488 hold_0_35 - 440.8427876823339 hold_0_36
   - 429.2477643345607 hold_0_37 - 418.2470527934393 hold_0_38
   - 407.7961019490255 hold_0_39 - 397.8547050219405 hold_0_40
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_2_nic_out_1_nic_in_2: switch_nic_out_1_nic_in_2 = 1 ->
   - 8079.207920792079 hold_2_1 - 5403.973509933775 hold_2_2
   - 4059.701492537313 hold_2_3 - 3250.996015936255 hold_2_4
   - 2710.963455149502 hold_2_5 - 2324.786324786325 hold_2_6
   - 2034.912718204489 hold_2_7 - 1809.312638580931 hold_2_8
   - 1628.74251497006 hold_2_9 - 1480.943738656987 hold_2_10
   - 1357.737104825291 hold_2_11 - 1253.456221198157 hold_2_12
   - 1164.051355206847 hold_2_13 - 1086.551264980027 hold_2_14
   - 1018.7265917603 hold_2_15 - 958.8719153936545 hold_2_16
   - 905.6603773584906 hold_2_17 - 858.0441640378549 hold_2_18
   - 815.1848151848152 hold_2_19 - 776.4034253092293 hold_2_20
   - 741.1444141689373 hold_2_21 - 708.9487402258906 hold_2_22
   - 679.4338051623647 hold_2_23 - 652.2781774580336 hold_2_24
   - 627.2098385857032 hold_2_25 - 603.9970392301998 hold_2_26
   - 582.441113490364 hold_2_27 - 562.3707787732598 hold_2_28
   - 543.6375749500334 hold_2_29 - 526.1121856866538 hold_2_30
   - 509.681449094316 hold_2_31 - 494.2459115687462 hold_2_32
   - 479.7178130511463 hold_2_33 - 466.0194174757282 hold_2_34
   - 453.081621321488 hold_2_35 - 440.8427876823339 hold_2_36
   - 429.2477643345607 hold_2_37 - 418.2470527934393 hold_2_38
   - 407.7961019490255 hold_2_39 - 397.8547050219405 hold_2_40
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_3_nic_out_1_nic_in_3: switch_nic_out_1_nic_in_3 = 1 ->
   - 8079.207920792079 hold_3_1 - 5403.973509933775 hold_3_2
   - 4059.701492537313 hold_3_3 - 3250.996015936255 hold_3_4
   - 2710.963455149502 hold_3_5 - 2324.786324786325 hold_3_6
   - 2034.912718204489 hold_3_7 - 1809.312638580931 hold_3_8
   - 1628.74251497006 hold_3_9 - 1480.943738656987 hold_3_10
   - 1357.737104825291 hold_3_11 - 1253.456221198157 hold_3_12
   - 1164.051355206847 hold_3_13 - 1086.551264980027 hold_3_14
   - 1018.7265917603 hold_3_15 - 958.8719153936545 hold_3_16
   - 905.6603773584906 hold_3_17 - 858.0441640378549 hold_3_18
   - 815.1848151848152 hold_3_19 - 776.4034253092293 hold_3_20
   - 741.1444141689373 hold_3_21 - 708.9487402258906 hold_3_22
   - 679.4338051623647 hold_3_23 - 652.2781774580336 hold_3_24
   - 627.2098385857032 hold_3_25 - 603.9970392301998 hold_3_26
   - 582.441113490364 hold_3_27 - 562.3707787732598 hold_3_28
   - 543.6375749500334 hold_3_29 - 526.1121856866538 hold_3_30
   - 509.681449094316 hold_3_31 - 494.2459115687462 hold_3_32
   - 479.7178130511463 hold_3_33 - 466.0194174757282 hold_3_34
   - 453.081621321488 hold_3_35 - 440.8427876823339 hold_3_36
   - 429.2477643345607 hold_3_37 - 418.2470527934393 hold_3_38
   - 407.7961019490255 hold_3_39 - 397.8547050219405 hold_3_40
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_2_source_nic_in_2: switch_source_nic_in_2 = 1 ->
   - 13554.81727574751 hold_2_1 - 7404.718693284936 hold_2_2
   - 5093.632958801498 hold_2_3 - 3882.017126546147 hold_2_4
   - 3136.049192928517 hold_2_5 - 2630.560928433269 hold_2_6
   - 2265.40810660744 hold_2_7 - 1989.273525109702 hold_2_8
   - 1773.142112125163 hold_2_9 - 1599.37279498236 hold_2_10
   - 1456.622634773295 hold_2_11 - 1337.266470009833 hold_2_12
   - 1235.989094213875 hold_2_13 - 1148.972120529428 hold_2_14
   - 1073.401736385162 hold_2_15 - 1007.158726240435 hold_2_16
   - 948.6166007905139 hold_2_17 - 896.5062623599209 hold_2_18
   - 849.8229535513435 hold_2_19 - 807.7608394377352 hold_2_20
   - 769.6661007357103 hold_2_21 - 735.0027022158171 hold_2_22
   - 703.3270125840373 hold_2_23 - 674.2687159147249 hold_2_24
   - 647.5162672591651 hold_2_25 - 622.805678522363 hold_2_26
   - 599.911777679753 hold_2_27 - 578.6413274712806 hold_2_28
   - 558.8275578687852 hold_2_29 - 540.3257846642829 hold_2_30
   - 523.0098705294192 hold_2_31 - 506.7693454229288 hold_2_32
   - 491.5070473436935 hold_2_33 - 477.1371769383698 hold_2_34
   - 463.5836836723099 hold_2_35 - 450.7789194564137 hold_2_36
   - 438.6625094075906 hold_2_37 - 427.1803999581196 hold_2_38
   - 416.284052647689 hold_2_39 - 405.9297582330117 hold_2_40
   + flow_nic_in_2_2 <= 0
 constr_intergroup_throughput_0_nic_out_2_nic_in_0: switch_nic_out_2_nic_in_0 = 1 ->
   - 8079.207920792079 hold_0_1 - 5403.973509933775 hold_0_2
   - 4059.701492537313 hold_0_3 - 3250.996015936255 hold_0_4
   - 2710.963455149502 hold_0_5 - 2324.786324786325 hold_0_6
   - 2034.912718204489 hold_0_7 - 1809.312638580931 hold_0_8
   - 1628.74251497006 hold_0_9 - 1480.943738656987 hold_0_10
   - 1357.737104825291 hold_0_11 - 1253.456221198157 hold_0_12
   - 1164.051355206847 hold_0_13 - 1086.551264980027 hold_0_14
   - 1018.7265917603 hold_0_15 - 958.8719153936545 hold_0_16
   - 905.6603773584906 hold_0_17 - 858.0441640378549 hold_0_18
   - 815.1848151848152 hold_0_19 - 776.4034253092293 hold_0_20
   - 741.1444141689373 hold_0_21 - 708.9487402258906 hold_0_22
   - 679.4338051623647 hold_0_23 - 652.2781774580336 hold_0_24
   - 627.2098385857032 hold_0_25 - 603.9970392301998 hold_0_26
   - 582.441113490364 hold_0_27 - 562.3707787732598 hold_0_28
   - 543.6375749500334 hold_0_29 - 526.1121856866538 hold_0_30
   - 509.681449094316 hold_0_31 - 494.2459115687462 hold_0_32
   - 479.7178130511463 hold_0_33 - 466.0194174757282 hold_0_34
   - 453.081621321488 hold_0_35 - 440.8427876823339 hold_0_36
   - 429.2477643345607 hold_0_37 - 418.2470527934393 hold_0_38
   - 407.7961019490255 hold_0_39 - 397.8547050219405 hold_0_40
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_2_nic_in_1: switch_nic_out_2_nic_in_1 = 1 ->
   - 8079.207920792079 hold_1_1 - 5403.973509933775 hold_1_2
   - 4059.701492537313 hold_1_3 - 3250.996015936255 hold_1_4
   - 2710.963455149502 hold_1_5 - 2324.786324786325 hold_1_6
   - 2034.912718204489 hold_1_7 - 1809.312638580931 hold_1_8
   - 1628.74251497006 hold_1_9 - 1480.943738656987 hold_1_10
   - 1357.737104825291 hold_1_11 - 1253.456221198157 hold_1_12
   - 1164.051355206847 hold_1_13 - 1086.551264980027 hold_1_14
   - 1018.7265917603 hold_1_15 - 958.8719153936545 hold_1_16
   - 905.6603773584906 hold_1_17 - 858.0441640378549 hold_1_18
   - 815.1848151848152 hold_1_19 - 776.4034253092293 hold_1_20
   - 741.1444141689373 hold_1_21 - 708.9487402258906 hold_1_22
   - 679.4338051623647 hold_1_23 - 652.2781774580336 hold_1_24
   - 627.2098385857032 hold_1_25 - 603.9970392301998 hold_1_26
   - 582.441113490364 hold_1_27 - 562.3707787732598 hold_1_28
   - 543.6375749500334 hold_1_29 - 526.1121856866538 hold_1_30
   - 509.681449094316 hold_1_31 - 494.2459115687462 hold_1_32
   - 479.7178130511463 hold_1_33 - 466.0194174757282 hold_1_34
   - 453.081621321488 hold_1_35 - 440.8427876823339 hold_1_36
   - 429.2477643345607 hold_1_37 - 418.2470527934393 hold_1_38
   - 407.7961019490255 hold_1_39 - 397.8547050219405 hold_1_40
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_3_nic_out_2_nic_in_3: switch_nic_out_2_nic_in_3 = 1 ->
   - 8079.207920792079 hold_3_1 - 5403.973509933775 hold_3_2
   - 4059.701492537313 hold_3_3 - 3250.996015936255 hold_3_4
   - 2710.963455149502 hold_3_5 - 2324.786324786325 hold_3_6
   - 2034.912718204489 hold_3_7 - 1809.312638580931 hold_3_8
   - 1628.74251497006 hold_3_9 - 1480.943738656987 hold_3_10
   - 1357.737104825291 hold_3_11 - 1253.456221198157 hold_3_12
   - 1164.051355206847 hold_3_13 - 1086.551264980027 hold_3_14
   - 1018.7265917603 hold_3_15 - 958.8719153936545 hold_3_16
   - 905.6603773584906 hold_3_17 - 858.0441640378549 hold_3_18
   - 815.1848151848152 hold_3_19 - 776.4034253092293 hold_3_20
   - 741.1444141689373 hold_3_21 - 708.9487402258906 hold_3_22
   - 679.4338051623647 hold_3_23 - 652.2781774580336 hold_3_24
   - 627.2098385857032 hold_3_25 - 603.9970392301998 hold_3_26
   - 582.441113490364 hold_3_27 - 562.3707787732598 hold_3_28
   - 543.6375749500334 hold_3_29 - 526.1121856866538 hold_3_30
   - 509.681449094316 hold_3_31 - 494.2459115687462 hold_3_32
   - 479.7178130511463 hold_3_33 - 466.0194174757282 hold_3_34
   - 453.081621321488 hold_3_35 - 440.8427876823339 hold_3_36
   - 429.2477643345607 hold_3_37 - 418.2470527934393 hold_3_38
   - 407.7961019490255 hold_3_39 - 397.8547050219405 hold_3_40
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_3_source_nic_in_3: switch_source_nic_in_3 = 1 ->
   - 13554.81727574751 hold_3_1 - 7404.718693284936 hold_3_2
   - 5093.632958801498 hold_3_3 - 3882.017126546147 hold_3_4
   - 3136.049192928517 hold_3_5 - 2630.560928433269 hold_3_6
   - 2265.40810660744 hold_3_7 - 1989.273525109702 hold_3_8
   - 1773.142112125163 hold_3_9 - 1599.37279498236 hold_3_10
   - 1456.622634773295 hold_3_11 - 1337.266470009833 hold_3_12
   - 1235.989094213875 hold_3_13 - 1148.972120529428 hold_3_14
   - 1073.401736385162 hold_3_15 - 1007.158726240435 hold_3_16
   - 948.6166007905139 hold_3_17 - 896.5062623599209 hold_3_18
   - 849.8229535513435 hold_3_19 - 807.7608394377352 hold_3_20
   - 769.6661007357103 hold_3_21 - 735.0027022158171 hold_3_22
   - 703.3270125840373 hold_3_23 - 674.2687159147249 hold_3_24
   - 647.5162672591651 hold_3_25 - 622.805678522363 hold_3_26
   - 599.911777679753 hold_3_27 - 578.6413274712806 hold_3_28
   - 558.8275578687852 hold_3_29 - 540.3257846642829 hold_3_30
   - 523.0098705294192 hold_3_31 - 506.7693454229288 hold_3_32
   - 491.5070473436935 hold_3_33 - 477.1371769383698 hold_3_34
   - 463.5836836723099 hold_3_35 - 450.7789194564137 hold_3_36
   - 438.6625094075906 hold_3_37 - 427.1803999581196 hold_3_38
   - 416.284052647689 hold_3_39 - 405.9297582330117 hold_3_40
   + flow_nic_in_3_3 <= 0
 constr_intergroup_throughput_0_nic_out_3_nic_in_0: switch_nic_out_3_nic_in_0 = 1 ->
   - 8079.207920792079 hold_0_1 - 5403.973509933775 hold_0_2
   - 4059.701492537313 hold_0_3 - 3250.996015936255 hold_0_4
   - 2710.963455149502 hold_0_5 - 2324.786324786325 hold_0_6
   - 2034.912718204489 hold_0_7 - 1809.312638580931 hold_0_8
   - 1628.74251497006 hold_0_9 - 1480.943738656987 hold_0_10
   - 1357.737104825291 hold_0_11 - 1253.456221198157 hold_0_12
   - 1164.051355206847 hold_0_13 - 1086.551264980027 hold_0_14
   - 1018.7265917603 hold_0_15 - 958.8719153936545 hold_0_16
   - 905.6603773584906 hold_0_17 - 858.0441640378549 hold_0_18
   - 815.1848151848152 hold_0_19 - 776.4034253092293 hold_0_20
   - 741.1444141689373 hold_0_21 - 708.9487402258906 hold_0_22
   - 679.4338051623647 hold_0_23 - 652.2781774580336 hold_0_24
   - 627.2098385857032 hold_0_25 - 603.9970392301998 hold_0_26
   - 582.441113490364 hold_0_27 - 562.3707787732598 hold_0_28
   - 543.6375749500334 hold_0_29 - 526.1121856866538 hold_0_30
   - 509.681449094316 hold_0_31 - 494.2459115687462 hold_0_32
   - 479.7178130511463 hold_0_33 - 466.0194174757282 hold_0_34
   - 453.081621321488 hold_0_35 - 440.8427876823339 hold_0_36
   - 429.2477643345607 hold_0_37 - 418.2470527934393 hold_0_38
   - 407.7961019490255 hold_0_39 - 397.8547050219405 hold_0_40
   + flow_nic_in_0_0 <= 0
 constr_intergroup_throughput_1_nic_out_3_nic_in_1: switch_nic_out_3_nic_in_1 = 1 ->
   - 8079.207920792079 hold_1_1 - 5403.973509933775 hold_1_2
   - 4059.701492537313 hold_1_3 - 3250.996015936255 hold_1_4
   - 2710.963455149502 hold_1_5 - 2324.786324786325 hold_1_6
   - 2034.912718204489 hold_1_7 - 1809.312638580931 hold_1_8
   - 1628.74251497006 hold_1_9 - 1480.943738656987 hold_1_10
   - 1357.737104825291 hold_1_11 - 1253.456221198157 hold_1_12
   - 1164.051355206847 hold_1_13 - 1086.551264980027 hold_1_14
   - 1018.7265917603 hold_1_15 - 958.8719153936545 hold_1_16
   - 905.6603773584906 hold_1_17 - 858.0441640378549 hold_1_18
   - 815.1848151848152 hold_1_19 - 776.4034253092293 hold_1_20
   - 741.1444141689373 hold_1_21 - 708.9487402258906 hold_1_22
   - 679.4338051623647 hold_1_23 - 652.2781774580336 hold_1_24
   - 627.2098385857032 hold_1_25 - 603.9970392301998 hold_1_26
   - 582.441113490364 hold_1_27 - 562.3707787732598 hold_1_28
   - 543.6375749500334 hold_1_29 - 526.1121856866538 hold_1_30
   - 509.681449094316 hold_1_31 - 494.2459115687462 hold_1_32
   - 479.7178130511463 hold_1_33 - 466.0194174757282 hold_1_34
   - 453.081621321488 hold_1_35 - 440.8427876823339 hold_1_36
   - 429.2477643345607 hold_1_37 - 418.2470527934393 hold_1_38
   - 407.7961019490255 hold_1_39 - 397.8547050219405 hold_1_40
   + flow_nic_in_1_1 <= 0
 constr_intergroup_throughput_2_nic_out_3_nic_in_2: switch_nic_out_3_nic_in_2 = 1 ->
   - 8079.207920792079 hold_2_1 - 5403.973509933775 hold_2_2
   - 4059.701492537313 hold_2_3 - 3250.996015936255 hold_2_4
   - 2710.963455149502 hold_2_5 - 2324.786324786325 hold_2_6
   - 2034.912718204489 hold_2_7 - 1809.312638580931 hold_2_8
   - 1628.74251497006 hold_2_9 - 1480.943738656987 hold_2_10
   - 1357.737104825291 hold_2_11 - 1253.456221198157 hold_2_12
   - 1164.051355206847 hold_2_13 - 1086.551264980027 hold_2_14
   - 1018.7265917603 hold_2_15 - 958.8719153936545 hold_2_16
   - 905.6603773584906 hold_2_17 - 858.0441640378549 hold_2_18
   - 815.1848151848152 hold_2_19 - 776.4034253092293 hold_2_20
   - 741.1444141689373 hold_2_21 - 708.9487402258906 hold_2_22
   - 679.4338051623647 hold_2_23 - 652.2781774580336 hold_2_24
   - 627.2098385857032 hold_2_25 - 603.9970392301998 hold_2_26
   - 582.441113490364 hold_2_27 - 562.3707787732598 hold_2_28
   - 543.6375749500334 hold_2_29 - 526.1121856866538 hold_2_30
   - 509.681449094316 hold_2_31 - 494.2459115687462 hold_2_32
   - 479.7178130511463 hold_2_33 - 466.0194174757282 hold_2_34
   - 453.081621321488 hold_2_35 - 440.8427876823339 hold_2_36
   - 429.2477643345607 hold_2_37 - 418.2470527934393 hold_2_38
   - 407.7961019490255 hold_2_39 - 397.8547050219405 hold_2_40
   + flow_nic_in_2_2 <= 0
 inactive_end_not_complete_0: is_active_0 = 0 -> is_end_complete_0 = 0
 active_end_val_0: is_active_0 = 1 -> - start_0 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11 - 12 hold_0_12
   - 13 hold_0_13 - 14 hold_0_14 - 15 hold_0_15 - 16 hold_0_16
   - 17 hold_0_17 - 18 hold_0_18 - 19 hold_0_19 - 20 hold_0_20
   - 21 hold_0_21 - 22 hold_0_22 - 23 hold_0_23 - 24 hold_0_24
   - 25 hold_0_25 - 26 hold_0_26 - 27 hold_0_27 - 28 hold_0_28
   - 29 hold_0_29 - 30 hold_0_30 - 31 hold_0_31 - 32 hold_0_32
   - 33 hold_0_33 - 34 hold_0_34 - 35 hold_0_35 - 36 hold_0_36
   - 37 hold_0_37 - 38 hold_0_38 - 39 hold_0_39 - 40 hold_0_40 + val_end_0
   = 0
 inactive_end_zero_0: is_active_0 = 0 -> val_end_0 = 0
 inactive_end_not_complete_1: is_active_1 = 0 -> is_end_complete_1 = 0
 active_end_val_1: is_active_1 = 1 -> - start_1 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 - 10 hold_1_10 - 11 hold_1_11 - 12 hold_1_12
   - 13 hold_1_13 - 14 hold_1_14 - 15 hold_1_15 - 16 hold_1_16
   - 17 hold_1_17 - 18 hold_1_18 - 19 hold_1_19 - 20 hold_1_20
   - 21 hold_1_21 - 22 hold_1_22 - 23 hold_1_23 - 24 hold_1_24
   - 25 hold_1_25 - 26 hold_1_26 - 27 hold_1_27 - 28 hold_1_28
   - 29 hold_1_29 - 30 hold_1_30 - 31 hold_1_31 - 32 hold_1_32
   - 33 hold_1_33 - 34 hold_1_34 - 35 hold_1_35 - 36 hold_1_36
   - 37 hold_1_37 - 38 hold_1_38 - 39 hold_1_39 - 40 hold_1_40 + val_end_1
   = 0
 inactive_end_zero_1: is_active_1 = 0 -> val_end_1 = 0
 inactive_end_not_complete_2: is_active_2 = 0 -> is_end_complete_2 = 0
 active_end_val_2: is_active_2 = 1 -> - start_2 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 - 12 hold_2_12
   - 13 hold_2_13 - 14 hold_2_14 - 15 hold_2_15 - 16 hold_2_16
   - 17 hold_2_17 - 18 hold_2_18 - 19 hold_2_19 - 20 hold_2_20
   - 21 hold_2_21 - 22 hold_2_22 - 23 hold_2_23 - 24 hold_2_24
   - 25 hold_2_25 - 26 hold_2_26 - 27 hold_2_27 - 28 hold_2_28
   - 29 hold_2_29 - 30 hold_2_30 - 31 hold_2_31 - 32 hold_2_32
   - 33 hold_2_33 - 34 hold_2_34 - 35 hold_2_35 - 36 hold_2_36
   - 37 hold_2_37 - 38 hold_2_38 - 39 hold_2_39 - 40 hold_2_40 + val_end_2
   = 0
 inactive_end_zero_2: is_active_2 = 0 -> val_end_2 = 0
 inactive_end_not_complete_3: is_active_3 = 0 -> is_end_complete_3 = 0
 active_end_val_3: is_active_3 = 1 -> - start_3 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   - 8 hold_3_8 - 9 hold_3_9 - 10 hold_3_10 - 11 hold_3_11 - 12 hold_3_12
   - 13 hold_3_13 - 14 hold_3_14 - 15 hold_3_15 - 16 hold_3_16
   - 17 hold_3_17 - 18 hold_3_18 - 19 hold_3_19 - 20 hold_3_20
   - 21 hold_3_21 - 22 hold_3_22 - 23 hold_3_23 - 24 hold_3_24
   - 25 hold_3_25 - 26 hold_3_26 - 27 hold_3_27 - 28 hold_3_28
   - 29 hold_3_29 - 30 hold_3_30 - 31 hold_3_31 - 32 hold_3_32
   - 33 hold_3_33 - 34 hold_3_34 - 35 hold_3_35 - 36 hold_3_36
   - 37 hold_3_37 - 38 hold_3_38 - 39 hold_3_39 - 40 hold_3_40 + val_end_3
   = 0
 inactive_end_zero_3: is_active_3 = 0 -> val_end_3 = 0
 group_throughput_constr_0: group_tp_active_0 = 1 -> - 16320 hold_0_1
   - 8160 hold_0_2 - 5440 hold_0_3 - 4080 hold_0_4 - 3264 hold_0_5
   - 2720 hold_0_6 - 2331.428571428572 hold_0_7 - 2040 hold_0_8
   - 1813.333333333333 hold_0_9 - 1632 hold_0_10
   - 1483.636363636364 hold_0_11 - 1360 hold_0_12
   - 1255.384615384615 hold_0_13 - 1165.714285714286 hold_0_14
   - 1088 hold_0_15 - 1020 hold_0_16 - 960 hold_0_17
   - 906.6666666666666 hold_0_18 - 858.9473684210526 hold_0_19
   - 816 hold_0_20 - 777.1428571428571 hold_0_21
   - 741.8181818181819 hold_0_22 - 709.5652173913044 hold_0_23
   - 680 hold_0_24 - 652.8 hold_0_25 - 627.6923076923077 hold_0_26
   - 604.4444444444445 hold_0_27 - 582.8571428571429 hold_0_28
   - 562.7586206896552 hold_0_29 - 544 hold_0_30
   - 526.4516129032257 hold_0_31 - 510 hold_0_32
   - 494.5454545454545 hold_0_33 - 480 hold_0_34
   - 466.2857142857142 hold_0_35 - 453.3333333333333 hold_0_36
   - 441.081081081081 hold_0_37 - 429.4736842105263 hold_0_38
   - 418.4615384615385 hold_0_39 - 408 hold_0_40 + flow_nic_in_0_0 <= 0
 group_throughput_constr_1: group_tp_active_1 = 1 -> - 16320 hold_1_1
   - 8160 hold_1_2 - 5440 hold_1_3 - 4080 hold_1_4 - 3264 hold_1_5
   - 2720 hold_1_6 - 2331.428571428572 hold_1_7 - 2040 hold_1_8
   - 1813.333333333333 hold_1_9 - 1632 hold_1_10
   - 1483.636363636364 hold_1_11 - 1360 hold_1_12
   - 1255.384615384615 hold_1_13 - 1165.714285714286 hold_1_14
   - 1088 hold_1_15 - 1020 hold_1_16 - 960 hold_1_17
   - 906.6666666666666 hold_1_18 - 858.9473684210526 hold_1_19
   - 816 hold_1_20 - 777.1428571428571 hold_1_21
   - 741.8181818181819 hold_1_22 - 709.5652173913044 hold_1_23
   - 680 hold_1_24 - 652.8 hold_1_25 - 627.6923076923077 hold_1_26
   - 604.4444444444445 hold_1_27 - 582.8571428571429 hold_1_28
   - 562.7586206896552 hold_1_29 - 544 hold_1_30
   - 526.4516129032257 hold_1_31 - 510 hold_1_32
   - 494.5454545454545 hold_1_33 - 480 hold_1_34
   - 466.2857142857142 hold_1_35 - 453.3333333333333 hold_1_36
   - 441.081081081081 hold_1_37 - 429.4736842105263 hold_1_38
   - 418.4615384615385 hold_1_39 - 408 hold_1_40 + flow_nic_in_1_1 <= 0
 group_throughput_constr_2: group_tp_active_2 = 1 -> - 16320 hold_2_1
   - 8160 hold_2_2 - 5440 hold_2_3 - 4080 hold_2_4 - 3264 hold_2_5
   - 2720 hold_2_6 - 2331.428571428572 hold_2_7 - 2040 hold_2_8
   - 1813.333333333333 hold_2_9 - 1632 hold_2_10
   - 1483.636363636364 hold_2_11 - 1360 hold_2_12
   - 1255.384615384615 hold_2_13 - 1165.714285714286 hold_2_14
   - 1088 hold_2_15 - 1020 hold_2_16 - 960 hold_2_17
   - 906.6666666666666 hold_2_18 - 858.9473684210526 hold_2_19
   - 816 hold_2_20 - 777.1428571428571 hold_2_21
   - 741.8181818181819 hold_2_22 - 709.5652173913044 hold_2_23
   - 680 hold_2_24 - 652.8 hold_2_25 - 627.6923076923077 hold_2_26
   - 604.4444444444445 hold_2_27 - 582.8571428571429 hold_2_28
   - 562.7586206896552 hold_2_29 - 544 hold_2_30
   - 526.4516129032257 hold_2_31 - 510 hold_2_32
   - 494.5454545454545 hold_2_33 - 480 hold_2_34
   - 466.2857142857142 hold_2_35 - 453.3333333333333 hold_2_36
   - 441.081081081081 hold_2_37 - 429.4736842105263 hold_2_38
   - 418.4615384615385 hold_2_39 - 408 hold_2_40 + flow_nic_in_2_2 <= 0
 group_throughput_constr_3: group_tp_active_3 = 1 -> - 16320 hold_3_1
   - 8160 hold_3_2 - 5440 hold_3_3 - 4080 hold_3_4 - 3264 hold_3_5
   - 2720 hold_3_6 - 2331.428571428572 hold_3_7 - 2040 hold_3_8
   - 1813.333333333333 hold_3_9 - 1632 hold_3_10
   - 1483.636363636364 hold_3_11 - 1360 hold_3_12
   - 1255.384615384615 hold_3_13 - 1165.714285714286 hold_3_14
   - 1088 hold_3_15 - 1020 hold_3_16 - 960 hold_3_17
   - 906.6666666666666 hold_3_18 - 858.9473684210526 hold_3_19
   - 816 hold_3_20 - 777.1428571428571 hold_3_21
   - 741.8181818181819 hold_3_22 - 709.5652173913044 hold_3_23
   - 680 hold_3_24 - 652.8 hold_3_25 - 627.6923076923077 hold_3_26
   - 604.4444444444445 hold_3_27 - 582.8571428571429 hold_3_28
   - 562.7586206896552 hold_3_29 - 544 hold_3_30
   - 526.4516129032257 hold_3_31 - 510 hold_3_32
   - 494.5454545454545 hold_3_33 - 480 hold_3_34
   - 466.2857142857142 hold_3_35 - 453.3333333333333 hold_3_36
   - 441.081081081081 hold_3_37 - 429.4736842105263 hold_3_38
   - 418.4615384615385 hold_3_39 - 408 hold_3_40 + flow_nic_in_3_3 <= 0
Bounds
Binaries
 hold_0_1 hold_0_2 hold_0_3 hold_0_4 hold_0_5 hold_0_6 hold_0_7 hold_0_8
 hold_0_9 hold_0_10 hold_0_11 hold_0_12 hold_0_13 hold_0_14 hold_0_15
 hold_0_16 hold_0_17 hold_0_18 hold_0_19 hold_0_20 hold_0_21 hold_0_22
 hold_0_23 hold_0_24 hold_0_25 hold_0_26 hold_0_27 hold_0_28 hold_0_29
 hold_0_30 hold_0_31 hold_0_32 hold_0_33 hold_0_34 hold_0_35 hold_0_36
 hold_0_37 hold_0_38 hold_0_39 hold_0_40 hold_1_1 hold_1_2 hold_1_3
 hold_1_4 hold_1_5 hold_1_6 hold_1_7 hold_1_8 hold_1_9 hold_1_10 hold_1_11
 hold_1_12 hold_1_13 hold_1_14 hold_1_15 hold_1_16 hold_1_17 hold_1_18
 hold_1_19 hold_1_20 hold_1_21 hold_1_22 hold_1_23 hold_1_24 hold_1_25
 hold_1_26 hold_1_27 hold_1_28 hold_1_29 hold_1_30 hold_1_31 hold_1_32
 hold_1_33 hold_1_34 hold_1_35 hold_1_36 hold_1_37 hold_1_38 hold_1_39
 hold_1_40 hold_2_1 hold_2_2 hold_2_3 hold_2_4 hold_2_5 hold_2_6 hold_2_7
 hold_2_8 hold_2_9 hold_2_10 hold_2_11 hold_2_12 hold_2_13 hold_2_14
 hold_2_15 hold_2_16 hold_2_17 hold_2_18 hold_2_19 hold_2_20 hold_2_21
 hold_2_22 hold_2_23 hold_2_24 hold_2_25 hold_2_26 hold_2_27 hold_2_28
 hold_2_29 hold_2_30 hold_2_31 hold_2_32 hold_2_33 hold_2_34 hold_2_35
 hold_2_36 hold_2_37 hold_2_38 hold_2_39 hold_2_40 hold_3_1 hold_3_2
 hold_3_3 hold_3_4 hold_3_5 hold_3_6 hold_3_7 hold_3_8 hold_3_9 hold_3_10
 hold_3_11 hold_3_12 hold_3_13 hold_3_14 hold_3_15 hold_3_16 hold_3_17
 hold_3_18 hold_3_19 hold_3_20 hold_3_21 hold_3_22 hold_3_23 hold_3_24
 hold_3_25 hold_3_26 hold_3_27 hold_3_28 hold_3_29 hold_3_30 hold_3_31
 hold_3_32 hold_3_33 hold_3_34 hold_3_35 hold_3_36 hold_3_37 hold_3_38
 hold_3_39 hold_3_40 switch_source_nic_in_0 switch_nic_out_0_sink
 switch_nic_out_0_nic_in_1 switch_nic_out_0_nic_in_2
 switch_nic_out_0_nic_in_3 switch_nic_in_0_0 switch_0_nic_out_0
 switch_source_nic_in_1 switch_nic_out_1_sink switch_nic_out_1_nic_in_0
 switch_nic_out_1_nic_in_2 switch_nic_out_1_nic_in_3 switch_nic_in_1_1
 switch_1_nic_out_1 switch_source_nic_in_2 switch_nic_out_2_sink
 switch_nic_out_2_nic_in_0 switch_nic_out_2_nic_in_1
 switch_nic_out_2_nic_in_3 switch_nic_in_2_2 switch_2_nic_out_2
 switch_source_nic_in_3 switch_nic_out_3_sink switch_nic_out_3_nic_in_0
 switch_nic_out_3_nic_in_1 switch_nic_out_3_nic_in_2 switch_nic_in_3_3
 switch_3_nic_out_3 group_tp_active_0 group_tp_active_1 group_tp_active_2
 group_tp_active_3 is_active_0 is_active_1 is_active_2 is_active_3
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
