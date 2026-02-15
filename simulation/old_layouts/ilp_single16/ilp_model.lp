\ Model 2026-01-22_17-24-28
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  flow_source_0 + flow_source_1 + flow_source_2 + flow_source_3
   + flow_source_4 + flow_source_5 + flow_source_6 + flow_source_7
   + flow_source_8 + flow_source_9 + flow_source_10 + flow_source_11
   + flow_source_12 + flow_source_13 + flow_source_14 + flow_source_15
Subject To
 hold_constraint_0: hold_0_1 + hold_0_2 + hold_0_3 + hold_0_4 + hold_0_5
   + hold_0_6 + hold_0_7 = 1
 end_constraint_0: start_0 + hold_0_1 + 2 hold_0_2 + 3 hold_0_3
   + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7 <= 60
 hold_constraint_1: hold_1_1 + hold_1_2 + hold_1_3 + hold_1_4 + hold_1_5
   + hold_1_6 + hold_1_7 = 1
 end_constraint_1: start_1 + hold_1_1 + 2 hold_1_2 + 3 hold_1_3
   + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7 <= 60
 hold_constraint_2: hold_2_1 + hold_2_2 + hold_2_3 + hold_2_4 + hold_2_5
   + hold_2_6 + hold_2_7 + hold_2_8 + hold_2_9 + hold_2_10 + hold_2_11 = 1
 end_constraint_2: start_2 + hold_2_1 + 2 hold_2_2 + 3 hold_2_3
   + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7 + 8 hold_2_8
   + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 <= 60
 hold_constraint_3: hold_3_1 + hold_3_2 + hold_3_3 + hold_3_4 + hold_3_5
   + hold_3_6 + hold_3_7 = 1
 end_constraint_3: start_3 + hold_3_1 + 2 hold_3_2 + 3 hold_3_3
   + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7 <= 60
 hold_constraint_4: hold_4_1 + hold_4_2 + hold_4_3 + hold_4_4 + hold_4_5
   + hold_4_6 + hold_4_7 = 1
 end_constraint_4: start_4 + hold_4_1 + 2 hold_4_2 + 3 hold_4_3
   + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7 <= 60
 hold_constraint_5: hold_5_1 + hold_5_2 + hold_5_3 + hold_5_4 + hold_5_5
   + hold_5_6 + hold_5_7 + hold_5_8 + hold_5_9 + hold_5_10 + hold_5_11
   + hold_5_12 + hold_5_13 + hold_5_14 + hold_5_15 + hold_5_16 + hold_5_17
   + hold_5_18 = 1
 end_constraint_5: start_5 + hold_5_1 + 2 hold_5_2 + 3 hold_5_3
   + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7 + 8 hold_5_8
   + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12 + 13 hold_5_13
   + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16 + 17 hold_5_17
   + 18 hold_5_18 <= 60
 hold_constraint_6: hold_6_1 + hold_6_2 + hold_6_3 + hold_6_4 + hold_6_5
   + hold_6_6 + hold_6_7 + hold_6_8 + hold_6_9 + hold_6_10 + hold_6_11 = 1
 end_constraint_6: start_6 + hold_6_1 + 2 hold_6_2 + 3 hold_6_3
   + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7 + 8 hold_6_8
   + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 <= 60
 hold_constraint_7: hold_7_1 + hold_7_2 + hold_7_3 + hold_7_4 + hold_7_5
   + hold_7_6 + hold_7_7 = 1
 end_constraint_7: start_7 + hold_7_1 + 2 hold_7_2 + 3 hold_7_3
   + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7 <= 60
 hold_constraint_8: hold_8_1 + hold_8_2 + hold_8_3 + hold_8_4 + hold_8_5
   + hold_8_6 + hold_8_7 + hold_8_8 + hold_8_9 + hold_8_10 + hold_8_11 = 1
 end_constraint_8: start_8 + hold_8_1 + 2 hold_8_2 + 3 hold_8_3
   + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7 + 8 hold_8_8
   + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 <= 60
 hold_constraint_9: hold_9_1 + hold_9_2 + hold_9_3 + hold_9_4 + hold_9_5
   + hold_9_6 + hold_9_7 = 1
 end_constraint_9: start_9 + hold_9_1 + 2 hold_9_2 + 3 hold_9_3
   + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7 <= 60
 hold_constraint_10: hold_10_1 + hold_10_2 + hold_10_3 + hold_10_4
   + hold_10_5 + hold_10_6 + hold_10_7 + hold_10_8 + hold_10_9 + hold_10_10
   + hold_10_11 + hold_10_12 + hold_10_13 + hold_10_14 + hold_10_15
   + hold_10_16 + hold_10_17 + hold_10_18 = 1
 end_constraint_10: start_10 + hold_10_1 + 2 hold_10_2 + 3 hold_10_3
   + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7 + 8 hold_10_8
   + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11 + 12 hold_10_12
   + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15 + 16 hold_10_16
   + 17 hold_10_17 + 18 hold_10_18 <= 60
 hold_constraint_11: hold_11_1 + hold_11_2 + hold_11_3 + hold_11_4
   + hold_11_5 + hold_11_6 + hold_11_7 + hold_11_8 + hold_11_9 + hold_11_10
   + hold_11_11 = 1
 end_constraint_11: start_11 + hold_11_1 + 2 hold_11_2 + 3 hold_11_3
   + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7 + 8 hold_11_8
   + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11 <= 60
 hold_constraint_12: hold_12_1 + hold_12_2 + hold_12_3 + hold_12_4
   + hold_12_5 + hold_12_6 + hold_12_7 = 1
 end_constraint_12: start_12 + hold_12_1 + 2 hold_12_2 + 3 hold_12_3
   + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6 + 7 hold_12_7 <= 60
 hold_constraint_13: hold_13_1 + hold_13_2 + hold_13_3 + hold_13_4
   + hold_13_5 + hold_13_6 + hold_13_7 + hold_13_8 + hold_13_9 + hold_13_10
   + hold_13_11 = 1
 end_constraint_13: start_13 + hold_13_1 + 2 hold_13_2 + 3 hold_13_3
   + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6 + 7 hold_13_7 + 8 hold_13_8
   + 9 hold_13_9 + 10 hold_13_10 + 11 hold_13_11 <= 60
 hold_constraint_14: hold_14_1 + hold_14_2 + hold_14_3 + hold_14_4
   + hold_14_5 + hold_14_6 + hold_14_7 + hold_14_8 + hold_14_9 + hold_14_10
   + hold_14_11 = 1
 end_constraint_14: start_14 + hold_14_1 + 2 hold_14_2 + 3 hold_14_3
   + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6 + 7 hold_14_7 + 8 hold_14_8
   + 9 hold_14_9 + 10 hold_14_10 + 11 hold_14_11 <= 60
 hold_constraint_15: hold_15_1 + hold_15_2 + hold_15_3 + hold_15_4
   + hold_15_5 + hold_15_6 + hold_15_7 = 1
 end_constraint_15: start_15 + hold_15_1 + 2 hold_15_2 + 3 hold_15_3
   + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6 + 7 hold_15_7 <= 60
 node_flow_constraint_0: flow_source_0 - flow_0_sink - flow_0_6 + flow_6_0
   - flow_0_13 + flow_13_0 - flow_0_10 + flow_10_0 - flow_0_8 + flow_8_0
   - flow_0_2 + flow_2_0 - flow_0_12 + flow_12_0 - flow_0_1 + flow_1_0
   - flow_0_5 + flow_5_0 - flow_0_15 + flow_15_0 - flow_0_7 + flow_7_0
   - flow_0_14 + flow_14_0 - flow_0_11 + flow_11_0 - flow_0_9 + flow_9_0
   - flow_0_3 + flow_3_0 - flow_0_4 + flow_4_0 = 0
 node_flow_constraint_1: flow_source_1 - flow_1_sink - flow_1_14
   + flow_14_1 - flow_1_4 + flow_4_1 - flow_1_5 + flow_5_1 - flow_1_13
   + flow_13_1 + flow_0_1 - flow_1_0 - flow_1_2 + flow_2_1 - flow_1_15
   + flow_15_1 - flow_1_7 + flow_7_1 - flow_1_6 + flow_6_1 - flow_1_8
   + flow_8_1 - flow_1_11 + flow_11_1 - flow_1_3 + flow_3_1 - flow_1_10
   + flow_10_1 - flow_1_12 + flow_12_1 - flow_1_9 + flow_9_1 = 0
 node_flow_constraint_2: flow_source_2 - flow_2_sink - flow_2_7 + flow_7_2
   - flow_2_12 + flow_12_2 - flow_2_10 + flow_10_2 - flow_2_6 + flow_6_2
   - flow_2_13 + flow_13_2 - flow_2_14 + flow_14_2 + flow_0_2 - flow_2_0
   - flow_2_5 + flow_5_2 - flow_2_15 + flow_15_2 + flow_1_2 - flow_2_1
   - flow_2_8 + flow_8_2 - flow_2_9 + flow_9_2 - flow_2_11 + flow_11_2
   - flow_2_3 + flow_3_2 - flow_2_4 + flow_4_2 = 0
 node_flow_constraint_3: flow_source_3 - flow_3_sink - flow_3_11
   + flow_11_3 - flow_3_15 + flow_15_3 - flow_3_13 + flow_13_3 - flow_3_7
   + flow_7_3 - flow_3_4 + flow_4_3 - flow_3_14 + flow_14_3 - flow_3_12
   + flow_12_3 - flow_3_10 + flow_10_3 - flow_3_8 + flow_8_3 - flow_3_6
   + flow_6_3 - flow_3_5 + flow_5_3 + flow_0_3 - flow_3_0 + flow_2_3
   - flow_3_2 + flow_1_3 - flow_3_1 - flow_3_9 + flow_9_3 = 0
 node_flow_constraint_4: flow_source_4 - flow_4_sink - flow_4_13
   + flow_13_4 - flow_4_12 + flow_12_4 - flow_4_15 + flow_15_4 + flow_1_4
   - flow_4_1 + flow_3_4 - flow_4_3 - flow_4_11 + flow_11_4 - flow_4_6
   + flow_6_4 - flow_4_7 + flow_7_4 - flow_4_14 + flow_14_4 - flow_4_9
   + flow_9_4 - flow_4_5 + flow_5_4 + flow_2_4 - flow_4_2 - flow_4_10
   + flow_10_4 + flow_0_4 - flow_4_0 - flow_4_8 + flow_8_4 = 0
 node_flow_constraint_5: flow_source_5 - flow_5_sink - flow_5_6 + flow_6_5
   - flow_5_15 + flow_15_5 - flow_5_9 + flow_9_5 + flow_1_5 - flow_5_1
   - flow_5_12 + flow_12_5 - flow_5_11 + flow_11_5 - flow_5_7 + flow_7_5
   + flow_2_5 - flow_5_2 + flow_0_5 - flow_5_0 + flow_4_5 - flow_5_4
   - flow_5_13 + flow_13_5 + flow_3_5 - flow_5_3 - flow_5_8 + flow_8_5
   - flow_5_10 + flow_10_5 - flow_5_14 + flow_14_5 = 0
 node_flow_constraint_6: flow_source_6 - flow_6_sink + flow_0_6 - flow_6_0
   + flow_5_6 - flow_6_5 - flow_6_11 + flow_11_6 + flow_2_6 - flow_6_2
   - flow_6_9 + flow_9_6 + flow_4_6 - flow_6_4 - flow_6_10 + flow_10_6
   - flow_6_14 + flow_14_6 - flow_6_13 + flow_13_6 - flow_6_15 + flow_15_6
   + flow_1_6 - flow_6_1 - flow_6_12 + flow_12_6 + flow_3_6 - flow_6_3
   - flow_6_8 + flow_8_6 - flow_6_7 + flow_7_6 = 0
 node_flow_constraint_7: flow_source_7 - flow_7_sink + flow_2_7 - flow_7_2
   + flow_3_7 - flow_7_3 - flow_7_14 + flow_14_7 - flow_7_11 + flow_11_7
   - flow_7_13 + flow_13_7 + flow_4_7 - flow_7_4 + flow_5_7 - flow_7_5
   - flow_7_10 + flow_10_7 - flow_7_12 + flow_12_7 + flow_1_7 - flow_7_1
   + flow_0_7 - flow_7_0 - flow_7_15 + flow_15_7 - flow_7_8 + flow_8_7
   - flow_7_9 + flow_9_7 + flow_6_7 - flow_7_6 = 0
 node_flow_constraint_8: flow_source_8 - flow_8_sink - flow_8_14
   + flow_14_8 - flow_8_13 + flow_13_8 + flow_0_8 - flow_8_0 - flow_8_9
   + flow_9_8 - flow_8_11 + flow_11_8 + flow_2_8 - flow_8_2 - flow_8_12
   + flow_12_8 + flow_1_8 - flow_8_1 - flow_8_10 + flow_10_8 + flow_3_8
   - flow_8_3 - flow_8_15 + flow_15_8 + flow_7_8 - flow_8_7 + flow_6_8
   - flow_8_6 + flow_5_8 - flow_8_5 + flow_4_8 - flow_8_4 = 0
 node_flow_constraint_9: flow_source_9 - flow_9_sink - flow_9_11
   + flow_11_9 - flow_9_12 + flow_12_9 + flow_5_9 - flow_9_5 + flow_6_9
   - flow_9_6 + flow_8_9 - flow_9_8 + flow_4_9 - flow_9_4 - flow_9_13
   + flow_13_9 + flow_2_9 - flow_9_2 - flow_9_15 + flow_15_9 + flow_0_9
   - flow_9_0 + flow_7_9 - flow_9_7 + flow_3_9 - flow_9_3 + flow_1_9
   - flow_9_1 - flow_9_10 + flow_10_9 - flow_9_14 + flow_14_9 = 0
 node_flow_constraint_10: flow_source_10 - flow_10_sink - flow_10_14
   + flow_14_10 - flow_10_12 + flow_12_10 + flow_2_10 - flow_10_2
   + flow_0_10 - flow_10_0 - flow_10_11 + flow_11_10 + flow_6_10
   - flow_10_6 + flow_7_10 - flow_10_7 - flow_10_15 + flow_15_10
   + flow_8_10 - flow_10_8 + flow_3_10 - flow_10_3 - flow_10_13
   + flow_13_10 + flow_1_10 - flow_10_1 + flow_5_10 - flow_10_5 + flow_4_10
   - flow_10_4 + flow_9_10 - flow_10_9 = 0
 node_flow_constraint_11: flow_source_11 - flow_11_sink + flow_3_11
   - flow_11_3 - flow_11_15 + flow_15_11 + flow_9_11 - flow_11_9
   + flow_6_11 - flow_11_6 + flow_7_11 - flow_11_7 + flow_4_11 - flow_11_4
   + flow_10_11 - flow_11_10 + flow_5_11 - flow_11_5 - flow_11_12
   + flow_12_11 + flow_8_11 - flow_11_8 - flow_11_14 + flow_14_11
   + flow_1_11 - flow_11_1 + flow_0_11 - flow_11_0 + flow_2_11 - flow_11_2
   - flow_11_13 + flow_13_11 = 0
 node_flow_constraint_12: flow_source_12 - flow_12_sink + flow_10_12
   - flow_12_10 + flow_2_12 - flow_12_2 + flow_4_12 - flow_12_4 + flow_9_12
   - flow_12_9 + flow_5_12 - flow_12_5 - flow_12_15 + flow_15_12
   + flow_0_12 - flow_12_0 + flow_11_12 - flow_12_11 + flow_7_12
   - flow_12_7 - flow_12_13 + flow_13_12 + flow_8_12 - flow_12_8
   + flow_3_12 - flow_12_3 + flow_6_12 - flow_12_6 - flow_12_14
   + flow_14_12 + flow_1_12 - flow_12_1 = 0
 node_flow_constraint_13: flow_source_13 - flow_13_sink + flow_4_13
   - flow_13_4 + flow_3_13 - flow_13_3 + flow_8_13 - flow_13_8 + flow_0_13
   - flow_13_0 + flow_2_13 - flow_13_2 - flow_13_14 + flow_14_13
   + flow_7_13 - flow_13_7 + flow_1_13 - flow_13_1 + flow_6_13 - flow_13_6
   + flow_12_13 - flow_13_12 - flow_13_15 + flow_15_13 + flow_9_13
   - flow_13_9 + flow_5_13 - flow_13_5 + flow_10_13 - flow_13_10
   + flow_11_13 - flow_13_11 = 0
 node_flow_constraint_14: flow_source_14 - flow_14_sink + flow_10_14
   - flow_14_10 + flow_8_14 - flow_14_8 + flow_1_14 - flow_14_1 + flow_7_14
   - flow_14_7 + flow_13_14 - flow_14_13 + flow_2_14 - flow_14_2
   - flow_14_15 + flow_15_14 + flow_4_14 - flow_14_4 + flow_6_14
   - flow_14_6 + flow_11_14 - flow_14_11 + flow_3_14 - flow_14_3
   + flow_0_14 - flow_14_0 + flow_12_14 - flow_14_12 + flow_5_14
   - flow_14_5 + flow_9_14 - flow_14_9 = 0
 node_flow_constraint_15: flow_source_15 - flow_15_sink + flow_11_15
   - flow_15_11 + flow_3_15 - flow_15_3 + flow_5_15 - flow_15_5 + flow_4_15
   - flow_15_4 + flow_14_15 - flow_15_14 + flow_12_15 - flow_15_12
   + flow_2_15 - flow_15_2 + flow_6_15 - flow_15_6 + flow_10_15
   - flow_15_10 + flow_0_15 - flow_15_0 + flow_1_15 - flow_15_1
   + flow_13_15 - flow_15_13 + flow_7_15 - flow_15_7 + flow_9_15
   - flow_15_9 + flow_8_15 - flow_15_8 = 0
 node_throughput_constraint_0: - 2320.661144544786 hold_0_1
   - 1072.19081035998 hold_0_2 - 601.432327787442 hold_0_3
   - 451.0240438081751 hold_0_4 - 323.4054440555661 hold_0_5
   - 237.7979382152295 hold_0_6 - 283.9928300288351 hold_0_7
   + flow_source_0 + flow_6_0 + flow_13_0 + flow_10_0 + flow_8_0 + flow_2_0
   + flow_12_0 + flow_1_0 + flow_5_0 + flow_15_0 + flow_7_0 + flow_14_0
   + flow_11_0 + flow_9_0 + flow_3_0 + flow_4_0 <= 0
 node_throughput_constraint_1: - 2320.661144544786 hold_1_1
   - 1072.19081035998 hold_1_2 - 601.432327787442 hold_1_3
   - 451.0240438081751 hold_1_4 - 323.4054440555661 hold_1_5
   - 237.7979382152295 hold_1_6 - 283.9928300288351 hold_1_7
   + flow_source_1 + flow_14_1 + flow_4_1 + flow_5_1 + flow_13_1 + flow_0_1
   + flow_2_1 + flow_15_1 + flow_7_1 + flow_6_1 + flow_8_1 + flow_11_1
   + flow_3_1 + flow_10_1 + flow_12_1 + flow_9_1 <= 0
 node_throughput_constraint_2: - 3197.727458706798 hold_2_1
   - 1506.391731513355 hold_2_2 - 892.8016261598826 hold_2_3
   - 669.9110685954037 hold_2_4 - 497.1587154008199 hold_2_5
   - 381.7272691377779 hold_2_6 - 274.8565176483288 hold_2_7
   - 240.6052653661339 hold_2_8 - 191.0876197896771 hold_2_9
   - 151.2887918459155 hold_2_10 - 229.7552419484142 hold_2_11
   + flow_source_2 + flow_7_2 + flow_12_2 + flow_10_2 + flow_6_2
   + flow_13_2 + flow_14_2 + flow_0_2 + flow_5_2 + flow_15_2 + flow_1_2
   + flow_8_2 + flow_9_2 + flow_11_2 + flow_3_2 + flow_4_2 <= 0
 node_throughput_constraint_3: - 2320.661144544786 hold_3_1
   - 1072.19081035998 hold_3_2 - 601.432327787442 hold_3_3
   - 451.0240438081751 hold_3_4 - 323.4054440555661 hold_3_5
   - 237.7979382152295 hold_3_6 - 283.9928300288351 hold_3_7
   + flow_source_3 + flow_11_3 + flow_15_3 + flow_13_3 + flow_7_3
   + flow_4_3 + flow_14_3 + flow_12_3 + flow_10_3 + flow_8_3 + flow_6_3
   + flow_5_3 + flow_0_3 + flow_2_3 + flow_1_3 + flow_9_3 <= 0
 node_throughput_constraint_4: - 2320.661144544786 hold_4_1
   - 1072.19081035998 hold_4_2 - 601.432327787442 hold_4_3
   - 451.0240438081751 hold_4_4 - 323.4054440555661 hold_4_5
   - 237.7979382152295 hold_4_6 - 283.9928300288351 hold_4_7
   + flow_source_4 + flow_13_4 + flow_12_4 + flow_15_4 + flow_1_4
   + flow_3_4 + flow_11_4 + flow_6_4 + flow_7_4 + flow_14_4 + flow_9_4
   + flow_5_4 + flow_2_4 + flow_10_4 + flow_0_4 + flow_8_4 <= 0
 node_throughput_constraint_5: - 10626.99174975639 hold_5_1
   - 5311.087921272938 hold_5_2 - 3539.034251309976 hold_5_3
   - 2652.959761566372 hold_5_4 - 2121.878109506345 hold_5_5
   - 1768.658334718934 hold_5_6 - 1516.330221384021 hold_5_7
   - 1327.171593353353 hold_5_8 - 1179.995261704564 hold_5_9
   - 1062.300633344631 hold_5_10 - 782.2129404883333 hold_5_11
   - 716.5881848256739 hold_5_12 - 661.0512696181356 hold_5_13
   - 613.4564473534082 hold_5_14 - 572.3043740562133 hold_5_15
   - 536.7201941529252 hold_5_16 - 505.3169085062248 hold_5_17
   - 477.4205560692816 hold_5_18 + flow_source_5 + flow_6_5 + flow_15_5
   + flow_9_5 + flow_1_5 + flow_12_5 + flow_11_5 + flow_7_5 + flow_2_5
   + flow_0_5 + flow_4_5 + flow_13_5 + flow_3_5 + flow_8_5 + flow_10_5
   + flow_14_5 <= 0
 node_throughput_constraint_6: - 3197.727458706798 hold_6_1
   - 1506.391731513355 hold_6_2 - 892.8016261598826 hold_6_3
   - 669.9110685954037 hold_6_4 - 497.1587154008199 hold_6_5
   - 381.7272691377779 hold_6_6 - 274.8565176483288 hold_6_7
   - 240.6052653661339 hold_6_8 - 191.0876197896771 hold_6_9
   - 151.2887918459155 hold_6_10 - 229.7552419484142 hold_6_11
   + flow_source_6 + flow_0_6 + flow_5_6 + flow_11_6 + flow_2_6 + flow_9_6
   + flow_4_6 + flow_10_6 + flow_14_6 + flow_13_6 + flow_15_6 + flow_1_6
   + flow_12_6 + flow_3_6 + flow_8_6 + flow_7_6 <= 0
 node_throughput_constraint_7: - 2320.661144544786 hold_7_1
   - 1072.19081035998 hold_7_2 - 601.432327787442 hold_7_3
   - 451.0240438081751 hold_7_4 - 323.4054440555661 hold_7_5
   - 237.7979382152295 hold_7_6 - 283.9928300288351 hold_7_7
   + flow_source_7 + flow_2_7 + flow_3_7 + flow_14_7 + flow_11_7
   + flow_13_7 + flow_4_7 + flow_5_7 + flow_10_7 + flow_12_7 + flow_1_7
   + flow_0_7 + flow_15_7 + flow_8_7 + flow_9_7 + flow_6_7 <= 0
 node_throughput_constraint_8: - 3197.727458706798 hold_8_1
   - 1506.391731513355 hold_8_2 - 892.8016261598826 hold_8_3
   - 669.9110685954037 hold_8_4 - 497.1587154008199 hold_8_5
   - 381.7272691377779 hold_8_6 - 274.8565176483288 hold_8_7
   - 240.6052653661339 hold_8_8 - 191.0876197896771 hold_8_9
   - 151.2887918459155 hold_8_10 - 229.7552419484142 hold_8_11
   + flow_source_8 + flow_14_8 + flow_13_8 + flow_0_8 + flow_9_8
   + flow_11_8 + flow_2_8 + flow_12_8 + flow_1_8 + flow_10_8 + flow_3_8
   + flow_15_8 + flow_7_8 + flow_6_8 + flow_5_8 + flow_4_8 <= 0
 node_throughput_constraint_9: - 2320.661144544786 hold_9_1
   - 1072.19081035998 hold_9_2 - 601.432327787442 hold_9_3
   - 451.0240438081751 hold_9_4 - 323.4054440555661 hold_9_5
   - 237.7979382152295 hold_9_6 - 283.9928300288351 hold_9_7
   + flow_source_9 + flow_11_9 + flow_12_9 + flow_5_9 + flow_6_9 + flow_8_9
   + flow_4_9 + flow_13_9 + flow_2_9 + flow_15_9 + flow_0_9 + flow_7_9
   + flow_3_9 + flow_1_9 + flow_10_9 + flow_14_9 <= 0
 node_throughput_constraint_10: - 10626.99174975639 hold_10_1
   - 5311.087921272938 hold_10_2 - 3539.034251309976 hold_10_3
   - 2652.959761566372 hold_10_4 - 2121.878109506345 hold_10_5
   - 1768.658334718934 hold_10_6 - 1516.330221384021 hold_10_7
   - 1327.171593353353 hold_10_8 - 1179.995261704564 hold_10_9
   - 1062.300633344631 hold_10_10 - 782.2129404883333 hold_10_11
   - 716.5881848256739 hold_10_12 - 661.0512696181356 hold_10_13
   - 613.4564473534082 hold_10_14 - 572.3043740562133 hold_10_15
   - 536.7201941529252 hold_10_16 - 505.3169085062248 hold_10_17
   - 477.4205560692816 hold_10_18 + flow_source_10 + flow_14_10
   + flow_12_10 + flow_2_10 + flow_0_10 + flow_11_10 + flow_6_10
   + flow_7_10 + flow_15_10 + flow_8_10 + flow_3_10 + flow_13_10
   + flow_1_10 + flow_5_10 + flow_4_10 + flow_9_10 <= 0
 node_throughput_constraint_11: - 3197.727458706798 hold_11_1
   - 1506.391731513355 hold_11_2 - 892.8016261598826 hold_11_3
   - 669.9110685954037 hold_11_4 - 497.1587154008199 hold_11_5
   - 381.7272691377779 hold_11_6 - 274.8565176483288 hold_11_7
   - 240.6052653661339 hold_11_8 - 191.0876197896771 hold_11_9
   - 151.2887918459155 hold_11_10 - 229.7552419484142 hold_11_11
   + flow_source_11 + flow_3_11 + flow_15_11 + flow_9_11 + flow_6_11
   + flow_7_11 + flow_4_11 + flow_10_11 + flow_5_11 + flow_12_11
   + flow_8_11 + flow_14_11 + flow_1_11 + flow_0_11 + flow_2_11
   + flow_13_11 <= 0
 node_throughput_constraint_12: - 2320.661144544786 hold_12_1
   - 1072.19081035998 hold_12_2 - 601.432327787442 hold_12_3
   - 451.0240438081751 hold_12_4 - 323.4054440555661 hold_12_5
   - 237.7979382152295 hold_12_6 - 283.9928300288351 hold_12_7
   + flow_source_12 + flow_10_12 + flow_2_12 + flow_4_12 + flow_9_12
   + flow_5_12 + flow_15_12 + flow_0_12 + flow_11_12 + flow_7_12
   + flow_13_12 + flow_8_12 + flow_3_12 + flow_6_12 + flow_14_12
   + flow_1_12 <= 0
 node_throughput_constraint_13: - 3197.727458706798 hold_13_1
   - 1506.391731513355 hold_13_2 - 892.8016261598826 hold_13_3
   - 669.9110685954037 hold_13_4 - 497.1587154008199 hold_13_5
   - 381.7272691377779 hold_13_6 - 274.8565176483288 hold_13_7
   - 240.6052653661339 hold_13_8 - 191.0876197896771 hold_13_9
   - 151.2887918459155 hold_13_10 - 229.7552419484142 hold_13_11
   + flow_source_13 + flow_4_13 + flow_3_13 + flow_8_13 + flow_0_13
   + flow_2_13 + flow_14_13 + flow_7_13 + flow_1_13 + flow_6_13
   + flow_12_13 + flow_15_13 + flow_9_13 + flow_5_13 + flow_10_13
   + flow_11_13 <= 0
 node_throughput_constraint_14: - 3197.727458706798 hold_14_1
   - 1506.391731513355 hold_14_2 - 892.8016261598826 hold_14_3
   - 669.9110685954037 hold_14_4 - 497.1587154008199 hold_14_5
   - 381.7272691377779 hold_14_6 - 274.8565176483288 hold_14_7
   - 240.6052653661339 hold_14_8 - 191.0876197896771 hold_14_9
   - 151.2887918459155 hold_14_10 - 229.7552419484142 hold_14_11
   + flow_source_14 + flow_10_14 + flow_8_14 + flow_1_14 + flow_7_14
   + flow_13_14 + flow_2_14 + flow_15_14 + flow_4_14 + flow_6_14
   + flow_11_14 + flow_3_14 + flow_0_14 + flow_12_14 + flow_5_14
   + flow_9_14 <= 0
 node_throughput_constraint_15: - 2320.661144544786 hold_15_1
   - 1072.19081035998 hold_15_2 - 601.432327787442 hold_15_3
   - 451.0240438081751 hold_15_4 - 323.4054440555661 hold_15_5
   - 237.7979382152295 hold_15_6 - 283.9928300288351 hold_15_7
   + flow_source_15 + flow_11_15 + flow_3_15 + flow_5_15 + flow_4_15
   + flow_14_15 + flow_12_15 + flow_2_15 + flow_6_15 + flow_10_15
   + flow_0_15 + flow_1_15 + flow_13_15 + flow_7_15 + flow_9_15 + flow_8_15
   <= 0
 edge_disable_constr_source_0: start_0 + 60 switch_source_0 <= 60
 edge_disable_constr_source_1: start_1 + 60 switch_source_1 <= 60
 edge_disable_constr_source_2: start_2 + 60 switch_source_2 <= 60
 edge_disable_constr_source_3: start_3 + 60 switch_source_3 <= 60
 edge_disable_constr_source_4: start_4 + 60 switch_source_4 <= 60
 edge_disable_constr_source_5: start_5 + 60 switch_source_5 <= 60
 edge_disable_constr_source_6: start_6 + 60 switch_source_6 <= 60
 edge_disable_constr_source_7: start_7 + 60 switch_source_7 <= 60
 edge_disable_constr_source_8: start_8 + 60 switch_source_8 <= 60
 edge_disable_constr_source_9: start_9 + 60 switch_source_9 <= 60
 edge_disable_constr_source_10: start_10 + 60 switch_source_10 <= 60
 edge_disable_constr_source_11: start_11 + 60 switch_source_11 <= 60
 edge_disable_constr_source_12: start_12 + 60 switch_source_12 <= 60
 edge_disable_constr_source_13: start_13 + 60 switch_source_13 <= 60
 edge_disable_constr_source_14: start_14 + 60 switch_source_14 <= 60
 edge_disable_constr_source_15: start_15 + 60 switch_source_15 <= 60
 edge_disable_constr_0_sink: - start_0 - hold_0_1 - 2 hold_0_2 - 3 hold_0_3
   - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7 + 60 switch_0_sink
   <= 0
 edge_disable_constr_1_sink: - start_1 - hold_1_1 - 2 hold_1_2 - 3 hold_1_3
   - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7 + 60 switch_1_sink
   <= 0
 edge_disable_constr_2_sink: - start_2 - hold_2_1 - 2 hold_2_2 - 3 hold_2_3
   - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7 - 8 hold_2_8
   - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 + 60 switch_2_sink <= 0
 edge_disable_constr_3_sink: - start_3 - hold_3_1 - 2 hold_3_2 - 3 hold_3_3
   - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7 + 60 switch_3_sink
   <= 0
 edge_disable_constr_4_sink: - start_4 - hold_4_1 - 2 hold_4_2 - 3 hold_4_3
   - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7 + 60 switch_4_sink
   <= 0
 edge_disable_constr_5_sink: - start_5 - hold_5_1 - 2 hold_5_2 - 3 hold_5_3
   - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7 - 8 hold_5_8
   - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11 - 12 hold_5_12 - 13 hold_5_13
   - 14 hold_5_14 - 15 hold_5_15 - 16 hold_5_16 - 17 hold_5_17
   - 18 hold_5_18 + 60 switch_5_sink <= 0
 edge_disable_constr_6_sink: - start_6 - hold_6_1 - 2 hold_6_2 - 3 hold_6_3
   - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7 - 8 hold_6_8
   - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11 + 60 switch_6_sink <= 0
 edge_disable_constr_7_sink: - start_7 - hold_7_1 - 2 hold_7_2 - 3 hold_7_3
   - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7 + 60 switch_7_sink
   <= 0
 edge_disable_constr_8_sink: - start_8 - hold_8_1 - 2 hold_8_2 - 3 hold_8_3
   - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7 - 8 hold_8_8
   - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11 + 60 switch_8_sink <= 0
 edge_disable_constr_9_sink: - start_9 - hold_9_1 - 2 hold_9_2 - 3 hold_9_3
   - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7 + 60 switch_9_sink
   <= 0
 edge_disable_constr_10_sink: - start_10 - hold_10_1 - 2 hold_10_2
   - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6 - 7 hold_10_7
   - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10 - 11 hold_10_11
   - 12 hold_10_12 - 13 hold_10_13 - 14 hold_10_14 - 15 hold_10_15
   - 16 hold_10_16 - 17 hold_10_17 - 18 hold_10_18 + 60 switch_10_sink
   <= 0
 edge_disable_constr_11_sink: - start_11 - hold_11_1 - 2 hold_11_2
   - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6 - 7 hold_11_7
   - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10 - 11 hold_11_11
   + 60 switch_11_sink <= 0
 edge_disable_constr_12_sink: - start_12 - hold_12_1 - 2 hold_12_2
   - 3 hold_12_3 - 4 hold_12_4 - 5 hold_12_5 - 6 hold_12_6 - 7 hold_12_7
   + 60 switch_12_sink <= 0
 edge_disable_constr_13_sink: - start_13 - hold_13_1 - 2 hold_13_2
   - 3 hold_13_3 - 4 hold_13_4 - 5 hold_13_5 - 6 hold_13_6 - 7 hold_13_7
   - 8 hold_13_8 - 9 hold_13_9 - 10 hold_13_10 - 11 hold_13_11
   + 60 switch_13_sink <= 0
 edge_disable_constr_14_sink: - start_14 - hold_14_1 - 2 hold_14_2
   - 3 hold_14_3 - 4 hold_14_4 - 5 hold_14_5 - 6 hold_14_6 - 7 hold_14_7
   - 8 hold_14_8 - 9 hold_14_9 - 10 hold_14_10 - 11 hold_14_11
   + 60 switch_14_sink <= 0
 edge_disable_constr_15_sink: - start_15 - hold_15_1 - 2 hold_15_2
   - 3 hold_15_3 - 4 hold_15_4 - 5 hold_15_5 - 6 hold_15_6 - 7 hold_15_7
   + 60 switch_15_sink <= 0
 edge_disable_constr1_10_14: start_10 - start_14 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 12 hold_10_12 + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15
   + 16 hold_10_16 + 17 hold_10_17 + 18 hold_10_18 + 60 switch_10_14 <= 60
 edge_disable_constr2_10_14: - start_10 + start_14 - hold_10_1
   - 2 hold_10_2 - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6
   - 7 hold_10_7 - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10
   - 11 hold_10_11 - 12 hold_10_12 - 13 hold_10_13 - 14 hold_10_14
   - 15 hold_10_15 - 16 hold_10_16 - 17 hold_10_17 - 18 hold_10_18
   + 60 switch_10_14 <= 60
 edge_disable_constr1_14_10: - start_10 + start_14 + hold_14_1
   + 2 hold_14_2 + 3 hold_14_3 + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6
   + 7 hold_14_7 + 8 hold_14_8 + 9 hold_14_9 + 10 hold_14_10
   + 11 hold_14_11 + 60 switch_14_10 <= 60
 edge_disable_constr2_14_10: start_10 - start_14 - hold_14_1 - 2 hold_14_2
   - 3 hold_14_3 - 4 hold_14_4 - 5 hold_14_5 - 6 hold_14_6 - 7 hold_14_7
   - 8 hold_14_8 - 9 hold_14_9 - 10 hold_14_10 - 11 hold_14_11
   + 60 switch_14_10 <= 60
 edge_disable_constr1_3_11: start_3 - start_11 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 60 switch_3_11 <= 60
 edge_disable_constr2_3_11: - start_3 + start_11 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   + 60 switch_3_11 <= 60
 edge_disable_constr1_11_3: - start_3 + start_11 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 60 switch_11_3 <= 60
 edge_disable_constr2_11_3: start_3 - start_11 - hold_11_1 - 2 hold_11_2
   - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6 - 7 hold_11_7
   - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10 - 11 hold_11_11
   + 60 switch_11_3 <= 60
 edge_disable_constr1_8_14: start_8 - start_14 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 60 switch_8_14
   <= 60
 edge_disable_constr2_8_14: - start_8 + start_14 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11 + 60 switch_8_14
   <= 60
 edge_disable_constr1_14_8: - start_8 + start_14 + hold_14_1 + 2 hold_14_2
   + 3 hold_14_3 + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6 + 7 hold_14_7
   + 8 hold_14_8 + 9 hold_14_9 + 10 hold_14_10 + 11 hold_14_11
   + 60 switch_14_8 <= 60
 edge_disable_constr2_14_8: start_8 - start_14 - hold_14_1 - 2 hold_14_2
   - 3 hold_14_3 - 4 hold_14_4 - 5 hold_14_5 - 6 hold_14_6 - 7 hold_14_7
   - 8 hold_14_8 - 9 hold_14_9 - 10 hold_14_10 - 11 hold_14_11
   + 60 switch_14_8 <= 60
 edge_disable_constr1_11_15: start_11 - start_15 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 60 switch_11_15 <= 60
 edge_disable_constr2_11_15: - start_11 + start_15 - hold_11_1
   - 2 hold_11_2 - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6
   - 7 hold_11_7 - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10
   - 11 hold_11_11 + 60 switch_11_15 <= 60
 edge_disable_constr1_15_11: - start_11 + start_15 + hold_15_1
   + 2 hold_15_2 + 3 hold_15_3 + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6
   + 7 hold_15_7 + 60 switch_15_11 <= 60
 edge_disable_constr2_15_11: start_11 - start_15 - hold_15_1 - 2 hold_15_2
   - 3 hold_15_3 - 4 hold_15_4 - 5 hold_15_5 - 6 hold_15_6 - 7 hold_15_7
   + 60 switch_15_11 <= 60
 edge_disable_constr1_3_15: start_3 - start_15 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 60 switch_3_15 <= 60
 edge_disable_constr2_3_15: - start_3 + start_15 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   + 60 switch_3_15 <= 60
 edge_disable_constr1_15_3: - start_3 + start_15 + hold_15_1 + 2 hold_15_2
   + 3 hold_15_3 + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6 + 7 hold_15_7
   + 60 switch_15_3 <= 60
 edge_disable_constr2_15_3: start_3 - start_15 - hold_15_1 - 2 hold_15_2
   - 3 hold_15_3 - 4 hold_15_4 - 5 hold_15_5 - 6 hold_15_6 - 7 hold_15_7
   + 60 switch_15_3 <= 60
 edge_disable_constr1_0_6: start_0 - start_6 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 60 switch_0_6 <= 60
 edge_disable_constr2_0_6: - start_0 + start_6 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   + 60 switch_0_6 <= 60
 edge_disable_constr1_6_0: - start_0 + start_6 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 60 switch_6_0
   <= 60
 edge_disable_constr2_6_0: start_0 - start_6 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11 + 60 switch_6_0
   <= 60
 edge_disable_constr1_2_7: start_2 - start_7 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 60 switch_2_7
   <= 60
 edge_disable_constr2_2_7: - start_2 + start_7 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 + 60 switch_2_7
   <= 60
 edge_disable_constr1_7_2: - start_2 + start_7 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 60 switch_7_2 <= 60
 edge_disable_constr2_7_2: start_2 - start_7 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   + 60 switch_7_2 <= 60
 edge_disable_constr1_5_6: start_5 - start_6 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12
   + 13 hold_5_13 + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16
   + 17 hold_5_17 + 18 hold_5_18 + 60 switch_5_6 <= 60
 edge_disable_constr2_5_6: - start_5 + start_6 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11 - 12 hold_5_12
   - 13 hold_5_13 - 14 hold_5_14 - 15 hold_5_15 - 16 hold_5_16
   - 17 hold_5_17 - 18 hold_5_18 + 60 switch_5_6 <= 60
 edge_disable_constr1_6_5: - start_5 + start_6 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 60 switch_6_5
   <= 60
 edge_disable_constr2_6_5: start_5 - start_6 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11 + 60 switch_6_5
   <= 60
 edge_disable_constr1_4_13: start_4 - start_13 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 60 switch_4_13 <= 60
 edge_disable_constr2_4_13: - start_4 + start_13 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   + 60 switch_4_13 <= 60
 edge_disable_constr1_13_4: - start_4 + start_13 + hold_13_1 + 2 hold_13_2
   + 3 hold_13_3 + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6 + 7 hold_13_7
   + 8 hold_13_8 + 9 hold_13_9 + 10 hold_13_10 + 11 hold_13_11
   + 60 switch_13_4 <= 60
 edge_disable_constr2_13_4: start_4 - start_13 - hold_13_1 - 2 hold_13_2
   - 3 hold_13_3 - 4 hold_13_4 - 5 hold_13_5 - 6 hold_13_6 - 7 hold_13_7
   - 8 hold_13_8 - 9 hold_13_9 - 10 hold_13_10 - 11 hold_13_11
   + 60 switch_13_4 <= 60
 edge_disable_constr1_3_13: start_3 - start_13 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 60 switch_3_13 <= 60
 edge_disable_constr2_3_13: - start_3 + start_13 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   + 60 switch_3_13 <= 60
 edge_disable_constr1_13_3: - start_3 + start_13 + hold_13_1 + 2 hold_13_2
   + 3 hold_13_3 + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6 + 7 hold_13_7
   + 8 hold_13_8 + 9 hold_13_9 + 10 hold_13_10 + 11 hold_13_11
   + 60 switch_13_3 <= 60
 edge_disable_constr2_13_3: start_3 - start_13 - hold_13_1 - 2 hold_13_2
   - 3 hold_13_3 - 4 hold_13_4 - 5 hold_13_5 - 6 hold_13_6 - 7 hold_13_7
   - 8 hold_13_8 - 9 hold_13_9 - 10 hold_13_10 - 11 hold_13_11
   + 60 switch_13_3 <= 60
 edge_disable_constr1_9_11: start_9 - start_11 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 60 switch_9_11 <= 60
 edge_disable_constr2_9_11: - start_9 + start_11 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   + 60 switch_9_11 <= 60
 edge_disable_constr1_11_9: - start_9 + start_11 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 60 switch_11_9 <= 60
 edge_disable_constr2_11_9: start_9 - start_11 - hold_11_1 - 2 hold_11_2
   - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6 - 7 hold_11_7
   - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10 - 11 hold_11_11
   + 60 switch_11_9 <= 60
 edge_disable_constr1_10_12: start_10 - start_12 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 12 hold_10_12 + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15
   + 16 hold_10_16 + 17 hold_10_17 + 18 hold_10_18 + 60 switch_10_12 <= 60
 edge_disable_constr2_10_12: - start_10 + start_12 - hold_10_1
   - 2 hold_10_2 - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6
   - 7 hold_10_7 - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10
   - 11 hold_10_11 - 12 hold_10_12 - 13 hold_10_13 - 14 hold_10_14
   - 15 hold_10_15 - 16 hold_10_16 - 17 hold_10_17 - 18 hold_10_18
   + 60 switch_10_12 <= 60
 edge_disable_constr1_12_10: - start_10 + start_12 + hold_12_1
   + 2 hold_12_2 + 3 hold_12_3 + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6
   + 7 hold_12_7 + 60 switch_12_10 <= 60
 edge_disable_constr2_12_10: start_10 - start_12 - hold_12_1 - 2 hold_12_2
   - 3 hold_12_3 - 4 hold_12_4 - 5 hold_12_5 - 6 hold_12_6 - 7 hold_12_7
   + 60 switch_12_10 <= 60
 edge_disable_constr1_2_12: start_2 - start_12 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 60 switch_2_12
   <= 60
 edge_disable_constr2_2_12: - start_2 + start_12 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 + 60 switch_2_12
   <= 60
 edge_disable_constr1_12_2: - start_2 + start_12 + hold_12_1 + 2 hold_12_2
   + 3 hold_12_3 + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6 + 7 hold_12_7
   + 60 switch_12_2 <= 60
 edge_disable_constr2_12_2: start_2 - start_12 - hold_12_1 - 2 hold_12_2
   - 3 hold_12_3 - 4 hold_12_4 - 5 hold_12_5 - 6 hold_12_6 - 7 hold_12_7
   + 60 switch_12_2 <= 60
 edge_disable_constr1_4_12: start_4 - start_12 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 60 switch_4_12 <= 60
 edge_disable_constr2_4_12: - start_4 + start_12 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   + 60 switch_4_12 <= 60
 edge_disable_constr1_12_4: - start_4 + start_12 + hold_12_1 + 2 hold_12_2
   + 3 hold_12_3 + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6 + 7 hold_12_7
   + 60 switch_12_4 <= 60
 edge_disable_constr2_12_4: start_4 - start_12 - hold_12_1 - 2 hold_12_2
   - 3 hold_12_3 - 4 hold_12_4 - 5 hold_12_5 - 6 hold_12_6 - 7 hold_12_7
   + 60 switch_12_4 <= 60
 edge_disable_constr1_3_7: start_3 - start_7 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 60 switch_3_7 <= 60
 edge_disable_constr2_3_7: - start_3 + start_7 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   + 60 switch_3_7 <= 60
 edge_disable_constr1_7_3: - start_3 + start_7 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 60 switch_7_3 <= 60
 edge_disable_constr2_7_3: start_3 - start_7 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   + 60 switch_7_3 <= 60
 edge_disable_constr1_5_15: start_5 - start_15 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12
   + 13 hold_5_13 + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16
   + 17 hold_5_17 + 18 hold_5_18 + 60 switch_5_15 <= 60
 edge_disable_constr2_5_15: - start_5 + start_15 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11 - 12 hold_5_12
   - 13 hold_5_13 - 14 hold_5_14 - 15 hold_5_15 - 16 hold_5_16
   - 17 hold_5_17 - 18 hold_5_18 + 60 switch_5_15 <= 60
 edge_disable_constr1_15_5: - start_5 + start_15 + hold_15_1 + 2 hold_15_2
   + 3 hold_15_3 + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6 + 7 hold_15_7
   + 60 switch_15_5 <= 60
 edge_disable_constr2_15_5: start_5 - start_15 - hold_15_1 - 2 hold_15_2
   - 3 hold_15_3 - 4 hold_15_4 - 5 hold_15_5 - 6 hold_15_6 - 7 hold_15_7
   + 60 switch_15_5 <= 60
 edge_disable_constr1_1_14: start_1 - start_14 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 60 switch_1_14 <= 60
 edge_disable_constr2_1_14: - start_1 + start_14 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   + 60 switch_1_14 <= 60
 edge_disable_constr1_14_1: - start_1 + start_14 + hold_14_1 + 2 hold_14_2
   + 3 hold_14_3 + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6 + 7 hold_14_7
   + 8 hold_14_8 + 9 hold_14_9 + 10 hold_14_10 + 11 hold_14_11
   + 60 switch_14_1 <= 60
 edge_disable_constr2_14_1: start_1 - start_14 - hold_14_1 - 2 hold_14_2
   - 3 hold_14_3 - 4 hold_14_4 - 5 hold_14_5 - 6 hold_14_6 - 7 hold_14_7
   - 8 hold_14_8 - 9 hold_14_9 - 10 hold_14_10 - 11 hold_14_11
   + 60 switch_14_1 <= 60
 edge_disable_constr1_4_15: start_4 - start_15 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 60 switch_4_15 <= 60
 edge_disable_constr2_4_15: - start_4 + start_15 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   + 60 switch_4_15 <= 60
 edge_disable_constr1_15_4: - start_4 + start_15 + hold_15_1 + 2 hold_15_2
   + 3 hold_15_3 + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6 + 7 hold_15_7
   + 60 switch_15_4 <= 60
 edge_disable_constr2_15_4: start_4 - start_15 - hold_15_1 - 2 hold_15_2
   - 3 hold_15_3 - 4 hold_15_4 - 5 hold_15_5 - 6 hold_15_6 - 7 hold_15_7
   + 60 switch_15_4 <= 60
 edge_disable_constr1_1_4: start_1 - start_4 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 60 switch_1_4 <= 60
 edge_disable_constr2_1_4: - start_1 + start_4 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   + 60 switch_1_4 <= 60
 edge_disable_constr1_4_1: - start_1 + start_4 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 60 switch_4_1 <= 60
 edge_disable_constr2_4_1: start_1 - start_4 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   + 60 switch_4_1 <= 60
 edge_disable_constr1_2_10: start_2 - start_10 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 60 switch_2_10
   <= 60
 edge_disable_constr2_2_10: - start_2 + start_10 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 + 60 switch_2_10
   <= 60
 edge_disable_constr1_10_2: - start_2 + start_10 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 12 hold_10_12 + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15
   + 16 hold_10_16 + 17 hold_10_17 + 18 hold_10_18 + 60 switch_10_2 <= 60
 edge_disable_constr2_10_2: start_2 - start_10 - hold_10_1 - 2 hold_10_2
   - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6 - 7 hold_10_7
   - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10 - 11 hold_10_11
   - 12 hold_10_12 - 13 hold_10_13 - 14 hold_10_14 - 15 hold_10_15
   - 16 hold_10_16 - 17 hold_10_17 - 18 hold_10_18 + 60 switch_10_2 <= 60
 edge_disable_constr1_9_12: start_9 - start_12 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 60 switch_9_12 <= 60
 edge_disable_constr2_9_12: - start_9 + start_12 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   + 60 switch_9_12 <= 60
 edge_disable_constr1_12_9: - start_9 + start_12 + hold_12_1 + 2 hold_12_2
   + 3 hold_12_3 + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6 + 7 hold_12_7
   + 60 switch_12_9 <= 60
 edge_disable_constr2_12_9: start_9 - start_12 - hold_12_1 - 2 hold_12_2
   - 3 hold_12_3 - 4 hold_12_4 - 5 hold_12_5 - 6 hold_12_6 - 7 hold_12_7
   + 60 switch_12_9 <= 60
 edge_disable_constr1_8_13: start_8 - start_13 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 60 switch_8_13
   <= 60
 edge_disable_constr2_8_13: - start_8 + start_13 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11 + 60 switch_8_13
   <= 60
 edge_disable_constr1_13_8: - start_8 + start_13 + hold_13_1 + 2 hold_13_2
   + 3 hold_13_3 + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6 + 7 hold_13_7
   + 8 hold_13_8 + 9 hold_13_9 + 10 hold_13_10 + 11 hold_13_11
   + 60 switch_13_8 <= 60
 edge_disable_constr2_13_8: start_8 - start_13 - hold_13_1 - 2 hold_13_2
   - 3 hold_13_3 - 4 hold_13_4 - 5 hold_13_5 - 6 hold_13_6 - 7 hold_13_7
   - 8 hold_13_8 - 9 hold_13_9 - 10 hold_13_10 - 11 hold_13_11
   + 60 switch_13_8 <= 60
 edge_disable_constr1_0_13: start_0 - start_13 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 60 switch_0_13 <= 60
 edge_disable_constr2_0_13: - start_0 + start_13 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   + 60 switch_0_13 <= 60
 edge_disable_constr1_13_0: - start_0 + start_13 + hold_13_1 + 2 hold_13_2
   + 3 hold_13_3 + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6 + 7 hold_13_7
   + 8 hold_13_8 + 9 hold_13_9 + 10 hold_13_10 + 11 hold_13_11
   + 60 switch_13_0 <= 60
 edge_disable_constr2_13_0: start_0 - start_13 - hold_13_1 - 2 hold_13_2
   - 3 hold_13_3 - 4 hold_13_4 - 5 hold_13_5 - 6 hold_13_6 - 7 hold_13_7
   - 8 hold_13_8 - 9 hold_13_9 - 10 hold_13_10 - 11 hold_13_11
   + 60 switch_13_0 <= 60
 edge_disable_constr1_6_11: start_6 - start_11 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 60 switch_6_11
   <= 60
 edge_disable_constr2_6_11: - start_6 + start_11 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11 + 60 switch_6_11
   <= 60
 edge_disable_constr1_11_6: - start_6 + start_11 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 60 switch_11_6 <= 60
 edge_disable_constr2_11_6: start_6 - start_11 - hold_11_1 - 2 hold_11_2
   - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6 - 7 hold_11_7
   - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10 - 11 hold_11_11
   + 60 switch_11_6 <= 60
 edge_disable_constr1_2_6: start_2 - start_6 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 60 switch_2_6
   <= 60
 edge_disable_constr2_2_6: - start_2 + start_6 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 + 60 switch_2_6
   <= 60
 edge_disable_constr1_6_2: - start_2 + start_6 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 60 switch_6_2
   <= 60
 edge_disable_constr2_6_2: start_2 - start_6 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11 + 60 switch_6_2
   <= 60
 edge_disable_constr1_5_9: start_5 - start_9 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12
   + 13 hold_5_13 + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16
   + 17 hold_5_17 + 18 hold_5_18 + 60 switch_5_9 <= 60
 edge_disable_constr2_5_9: - start_5 + start_9 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11 - 12 hold_5_12
   - 13 hold_5_13 - 14 hold_5_14 - 15 hold_5_15 - 16 hold_5_16
   - 17 hold_5_17 - 18 hold_5_18 + 60 switch_5_9 <= 60
 edge_disable_constr1_9_5: - start_5 + start_9 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 60 switch_9_5 <= 60
 edge_disable_constr2_9_5: start_5 - start_9 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   + 60 switch_9_5 <= 60
 edge_disable_constr1_7_14: start_7 - start_14 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 60 switch_7_14 <= 60
 edge_disable_constr2_7_14: - start_7 + start_14 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   + 60 switch_7_14 <= 60
 edge_disable_constr1_14_7: - start_7 + start_14 + hold_14_1 + 2 hold_14_2
   + 3 hold_14_3 + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6 + 7 hold_14_7
   + 8 hold_14_8 + 9 hold_14_9 + 10 hold_14_10 + 11 hold_14_11
   + 60 switch_14_7 <= 60
 edge_disable_constr2_14_7: start_7 - start_14 - hold_14_1 - 2 hold_14_2
   - 3 hold_14_3 - 4 hold_14_4 - 5 hold_14_5 - 6 hold_14_6 - 7 hold_14_7
   - 8 hold_14_8 - 9 hold_14_9 - 10 hold_14_10 - 11 hold_14_11
   + 60 switch_14_7 <= 60
 edge_disable_constr1_6_9: start_6 - start_9 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 60 switch_6_9
   <= 60
 edge_disable_constr2_6_9: - start_6 + start_9 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11 + 60 switch_6_9
   <= 60
 edge_disable_constr1_9_6: - start_6 + start_9 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 60 switch_9_6 <= 60
 edge_disable_constr2_9_6: start_6 - start_9 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   + 60 switch_9_6 <= 60
 edge_disable_constr1_1_5: start_1 - start_5 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 60 switch_1_5 <= 60
 edge_disable_constr2_1_5: - start_1 + start_5 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   + 60 switch_1_5 <= 60
 edge_disable_constr1_5_1: - start_1 + start_5 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12
   + 13 hold_5_13 + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16
   + 17 hold_5_17 + 18 hold_5_18 + 60 switch_5_1 <= 60
 edge_disable_constr2_5_1: start_1 - start_5 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11 - 12 hold_5_12
   - 13 hold_5_13 - 14 hold_5_14 - 15 hold_5_15 - 16 hold_5_16
   - 17 hold_5_17 - 18 hold_5_18 + 60 switch_5_1 <= 60
 edge_disable_constr1_2_13: start_2 - start_13 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 60 switch_2_13
   <= 60
 edge_disable_constr2_2_13: - start_2 + start_13 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 + 60 switch_2_13
   <= 60
 edge_disable_constr1_13_2: - start_2 + start_13 + hold_13_1 + 2 hold_13_2
   + 3 hold_13_3 + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6 + 7 hold_13_7
   + 8 hold_13_8 + 9 hold_13_9 + 10 hold_13_10 + 11 hold_13_11
   + 60 switch_13_2 <= 60
 edge_disable_constr2_13_2: start_2 - start_13 - hold_13_1 - 2 hold_13_2
   - 3 hold_13_3 - 4 hold_13_4 - 5 hold_13_5 - 6 hold_13_6 - 7 hold_13_7
   - 8 hold_13_8 - 9 hold_13_9 - 10 hold_13_10 - 11 hold_13_11
   + 60 switch_13_2 <= 60
 edge_disable_constr1_13_14: start_13 - start_14 + hold_13_1 + 2 hold_13_2
   + 3 hold_13_3 + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6 + 7 hold_13_7
   + 8 hold_13_8 + 9 hold_13_9 + 10 hold_13_10 + 11 hold_13_11
   + 60 switch_13_14 <= 60
 edge_disable_constr2_13_14: - start_13 + start_14 - hold_13_1
   - 2 hold_13_2 - 3 hold_13_3 - 4 hold_13_4 - 5 hold_13_5 - 6 hold_13_6
   - 7 hold_13_7 - 8 hold_13_8 - 9 hold_13_9 - 10 hold_13_10
   - 11 hold_13_11 + 60 switch_13_14 <= 60
 edge_disable_constr1_14_13: - start_13 + start_14 + hold_14_1
   + 2 hold_14_2 + 3 hold_14_3 + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6
   + 7 hold_14_7 + 8 hold_14_8 + 9 hold_14_9 + 10 hold_14_10
   + 11 hold_14_11 + 60 switch_14_13 <= 60
 edge_disable_constr2_14_13: start_13 - start_14 - hold_14_1 - 2 hold_14_2
   - 3 hold_14_3 - 4 hold_14_4 - 5 hold_14_5 - 6 hold_14_6 - 7 hold_14_7
   - 8 hold_14_8 - 9 hold_14_9 - 10 hold_14_10 - 11 hold_14_11
   + 60 switch_14_13 <= 60
 edge_disable_constr1_0_10: start_0 - start_10 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 60 switch_0_10 <= 60
 edge_disable_constr2_0_10: - start_0 + start_10 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   + 60 switch_0_10 <= 60
 edge_disable_constr1_10_0: - start_0 + start_10 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 12 hold_10_12 + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15
   + 16 hold_10_16 + 17 hold_10_17 + 18 hold_10_18 + 60 switch_10_0 <= 60
 edge_disable_constr2_10_0: start_0 - start_10 - hold_10_1 - 2 hold_10_2
   - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6 - 7 hold_10_7
   - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10 - 11 hold_10_11
   - 12 hold_10_12 - 13 hold_10_13 - 14 hold_10_14 - 15 hold_10_15
   - 16 hold_10_16 - 17 hold_10_17 - 18 hold_10_18 + 60 switch_10_0 <= 60
 edge_disable_constr1_7_11: start_7 - start_11 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 60 switch_7_11 <= 60
 edge_disable_constr2_7_11: - start_7 + start_11 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   + 60 switch_7_11 <= 60
 edge_disable_constr1_11_7: - start_7 + start_11 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 60 switch_11_7 <= 60
 edge_disable_constr2_11_7: start_7 - start_11 - hold_11_1 - 2 hold_11_2
   - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6 - 7 hold_11_7
   - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10 - 11 hold_11_11
   + 60 switch_11_7 <= 60
 edge_disable_constr1_3_4: start_3 - start_4 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 60 switch_3_4 <= 60
 edge_disable_constr2_3_4: - start_3 + start_4 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   + 60 switch_3_4 <= 60
 edge_disable_constr1_4_3: - start_3 + start_4 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 60 switch_4_3 <= 60
 edge_disable_constr2_4_3: start_3 - start_4 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   + 60 switch_4_3 <= 60
 edge_disable_constr1_4_11: start_4 - start_11 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 60 switch_4_11 <= 60
 edge_disable_constr2_4_11: - start_4 + start_11 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   + 60 switch_4_11 <= 60
 edge_disable_constr1_11_4: - start_4 + start_11 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 60 switch_11_4 <= 60
 edge_disable_constr2_11_4: start_4 - start_11 - hold_11_1 - 2 hold_11_2
   - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6 - 7 hold_11_7
   - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10 - 11 hold_11_11
   + 60 switch_11_4 <= 60
 edge_disable_constr1_5_12: start_5 - start_12 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12
   + 13 hold_5_13 + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16
   + 17 hold_5_17 + 18 hold_5_18 + 60 switch_5_12 <= 60
 edge_disable_constr2_5_12: - start_5 + start_12 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11 - 12 hold_5_12
   - 13 hold_5_13 - 14 hold_5_14 - 15 hold_5_15 - 16 hold_5_16
   - 17 hold_5_17 - 18 hold_5_18 + 60 switch_5_12 <= 60
 edge_disable_constr1_12_5: - start_5 + start_12 + hold_12_1 + 2 hold_12_2
   + 3 hold_12_3 + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6 + 7 hold_12_7
   + 60 switch_12_5 <= 60
 edge_disable_constr2_12_5: start_5 - start_12 - hold_12_1 - 2 hold_12_2
   - 3 hold_12_3 - 4 hold_12_4 - 5 hold_12_5 - 6 hold_12_6 - 7 hold_12_7
   + 60 switch_12_5 <= 60
 edge_disable_constr1_7_13: start_7 - start_13 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 60 switch_7_13 <= 60
 edge_disable_constr2_7_13: - start_7 + start_13 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   + 60 switch_7_13 <= 60
 edge_disable_constr1_13_7: - start_7 + start_13 + hold_13_1 + 2 hold_13_2
   + 3 hold_13_3 + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6 + 7 hold_13_7
   + 8 hold_13_8 + 9 hold_13_9 + 10 hold_13_10 + 11 hold_13_11
   + 60 switch_13_7 <= 60
 edge_disable_constr2_13_7: start_7 - start_13 - hold_13_1 - 2 hold_13_2
   - 3 hold_13_3 - 4 hold_13_4 - 5 hold_13_5 - 6 hold_13_6 - 7 hold_13_7
   - 8 hold_13_8 - 9 hold_13_9 - 10 hold_13_10 - 11 hold_13_11
   + 60 switch_13_7 <= 60
 edge_disable_constr1_10_11: start_10 - start_11 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 12 hold_10_12 + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15
   + 16 hold_10_16 + 17 hold_10_17 + 18 hold_10_18 + 60 switch_10_11 <= 60
 edge_disable_constr2_10_11: - start_10 + start_11 - hold_10_1
   - 2 hold_10_2 - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6
   - 7 hold_10_7 - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10
   - 11 hold_10_11 - 12 hold_10_12 - 13 hold_10_13 - 14 hold_10_14
   - 15 hold_10_15 - 16 hold_10_16 - 17 hold_10_17 - 18 hold_10_18
   + 60 switch_10_11 <= 60
 edge_disable_constr1_11_10: - start_10 + start_11 + hold_11_1
   + 2 hold_11_2 + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6
   + 7 hold_11_7 + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10
   + 11 hold_11_11 + 60 switch_11_10 <= 60
 edge_disable_constr2_11_10: start_10 - start_11 - hold_11_1 - 2 hold_11_2
   - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6 - 7 hold_11_7
   - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10 - 11 hold_11_11
   + 60 switch_11_10 <= 60
 edge_disable_constr1_4_6: start_4 - start_6 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 60 switch_4_6 <= 60
 edge_disable_constr2_4_6: - start_4 + start_6 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   + 60 switch_4_6 <= 60
 edge_disable_constr1_6_4: - start_4 + start_6 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 60 switch_6_4
   <= 60
 edge_disable_constr2_6_4: start_4 - start_6 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11 + 60 switch_6_4
   <= 60
 edge_disable_constr1_2_14: start_2 - start_14 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 60 switch_2_14
   <= 60
 edge_disable_constr2_2_14: - start_2 + start_14 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 + 60 switch_2_14
   <= 60
 edge_disable_constr1_14_2: - start_2 + start_14 + hold_14_1 + 2 hold_14_2
   + 3 hold_14_3 + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6 + 7 hold_14_7
   + 8 hold_14_8 + 9 hold_14_9 + 10 hold_14_10 + 11 hold_14_11
   + 60 switch_14_2 <= 60
 edge_disable_constr2_14_2: start_2 - start_14 - hold_14_1 - 2 hold_14_2
   - 3 hold_14_3 - 4 hold_14_4 - 5 hold_14_5 - 6 hold_14_6 - 7 hold_14_7
   - 8 hold_14_8 - 9 hold_14_9 - 10 hold_14_10 - 11 hold_14_11
   + 60 switch_14_2 <= 60
 edge_disable_constr1_6_10: start_6 - start_10 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 60 switch_6_10
   <= 60
 edge_disable_constr2_6_10: - start_6 + start_10 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11 + 60 switch_6_10
   <= 60
 edge_disable_constr1_10_6: - start_6 + start_10 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 12 hold_10_12 + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15
   + 16 hold_10_16 + 17 hold_10_17 + 18 hold_10_18 + 60 switch_10_6 <= 60
 edge_disable_constr2_10_6: start_6 - start_10 - hold_10_1 - 2 hold_10_2
   - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6 - 7 hold_10_7
   - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10 - 11 hold_10_11
   - 12 hold_10_12 - 13 hold_10_13 - 14 hold_10_14 - 15 hold_10_15
   - 16 hold_10_16 - 17 hold_10_17 - 18 hold_10_18 + 60 switch_10_6 <= 60
 edge_disable_constr1_1_13: start_1 - start_13 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 60 switch_1_13 <= 60
 edge_disable_constr2_1_13: - start_1 + start_13 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   + 60 switch_1_13 <= 60
 edge_disable_constr1_13_1: - start_1 + start_13 + hold_13_1 + 2 hold_13_2
   + 3 hold_13_3 + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6 + 7 hold_13_7
   + 8 hold_13_8 + 9 hold_13_9 + 10 hold_13_10 + 11 hold_13_11
   + 60 switch_13_1 <= 60
 edge_disable_constr2_13_1: start_1 - start_13 - hold_13_1 - 2 hold_13_2
   - 3 hold_13_3 - 4 hold_13_4 - 5 hold_13_5 - 6 hold_13_6 - 7 hold_13_7
   - 8 hold_13_8 - 9 hold_13_9 - 10 hold_13_10 - 11 hold_13_11
   + 60 switch_13_1 <= 60
 edge_disable_constr1_5_11: start_5 - start_11 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12
   + 13 hold_5_13 + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16
   + 17 hold_5_17 + 18 hold_5_18 + 60 switch_5_11 <= 60
 edge_disable_constr2_5_11: - start_5 + start_11 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11 - 12 hold_5_12
   - 13 hold_5_13 - 14 hold_5_14 - 15 hold_5_15 - 16 hold_5_16
   - 17 hold_5_17 - 18 hold_5_18 + 60 switch_5_11 <= 60
 edge_disable_constr1_11_5: - start_5 + start_11 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 60 switch_11_5 <= 60
 edge_disable_constr2_11_5: start_5 - start_11 - hold_11_1 - 2 hold_11_2
   - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6 - 7 hold_11_7
   - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10 - 11 hold_11_11
   + 60 switch_11_5 <= 60
 edge_disable_constr1_14_15: start_14 - start_15 + hold_14_1 + 2 hold_14_2
   + 3 hold_14_3 + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6 + 7 hold_14_7
   + 8 hold_14_8 + 9 hold_14_9 + 10 hold_14_10 + 11 hold_14_11
   + 60 switch_14_15 <= 60
 edge_disable_constr2_14_15: - start_14 + start_15 - hold_14_1
   - 2 hold_14_2 - 3 hold_14_3 - 4 hold_14_4 - 5 hold_14_5 - 6 hold_14_6
   - 7 hold_14_7 - 8 hold_14_8 - 9 hold_14_9 - 10 hold_14_10
   - 11 hold_14_11 + 60 switch_14_15 <= 60
 edge_disable_constr1_15_14: - start_14 + start_15 + hold_15_1
   + 2 hold_15_2 + 3 hold_15_3 + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6
   + 7 hold_15_7 + 60 switch_15_14 <= 60
 edge_disable_constr2_15_14: start_14 - start_15 - hold_15_1 - 2 hold_15_2
   - 3 hold_15_3 - 4 hold_15_4 - 5 hold_15_5 - 6 hold_15_6 - 7 hold_15_7
   + 60 switch_15_14 <= 60
 edge_disable_constr1_4_7: start_4 - start_7 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 60 switch_4_7 <= 60
 edge_disable_constr2_4_7: - start_4 + start_7 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   + 60 switch_4_7 <= 60
 edge_disable_constr1_7_4: - start_4 + start_7 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 60 switch_7_4 <= 60
 edge_disable_constr2_7_4: start_4 - start_7 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   + 60 switch_7_4 <= 60
 edge_disable_constr1_5_7: start_5 - start_7 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12
   + 13 hold_5_13 + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16
   + 17 hold_5_17 + 18 hold_5_18 + 60 switch_5_7 <= 60
 edge_disable_constr2_5_7: - start_5 + start_7 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11 - 12 hold_5_12
   - 13 hold_5_13 - 14 hold_5_14 - 15 hold_5_15 - 16 hold_5_16
   - 17 hold_5_17 - 18 hold_5_18 + 60 switch_5_7 <= 60
 edge_disable_constr1_7_5: - start_5 + start_7 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 60 switch_7_5 <= 60
 edge_disable_constr2_7_5: start_5 - start_7 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   + 60 switch_7_5 <= 60
 edge_disable_constr1_12_15: start_12 - start_15 + hold_12_1 + 2 hold_12_2
   + 3 hold_12_3 + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6 + 7 hold_12_7
   + 60 switch_12_15 <= 60
 edge_disable_constr2_12_15: - start_12 + start_15 - hold_12_1
   - 2 hold_12_2 - 3 hold_12_3 - 4 hold_12_4 - 5 hold_12_5 - 6 hold_12_6
   - 7 hold_12_7 + 60 switch_12_15 <= 60
 edge_disable_constr1_15_12: - start_12 + start_15 + hold_15_1
   + 2 hold_15_2 + 3 hold_15_3 + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6
   + 7 hold_15_7 + 60 switch_15_12 <= 60
 edge_disable_constr2_15_12: start_12 - start_15 - hold_15_1 - 2 hold_15_2
   - 3 hold_15_3 - 4 hold_15_4 - 5 hold_15_5 - 6 hold_15_6 - 7 hold_15_7
   + 60 switch_15_12 <= 60
 edge_disable_constr1_0_8: start_0 - start_8 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 60 switch_0_8 <= 60
 edge_disable_constr2_0_8: - start_0 + start_8 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   + 60 switch_0_8 <= 60
 edge_disable_constr1_8_0: - start_0 + start_8 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 60 switch_8_0
   <= 60
 edge_disable_constr2_8_0: start_0 - start_8 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11 + 60 switch_8_0
   <= 60
 edge_disable_constr1_8_9: start_8 - start_9 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 60 switch_8_9
   <= 60
 edge_disable_constr2_8_9: - start_8 + start_9 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11 + 60 switch_8_9
   <= 60
 edge_disable_constr1_9_8: - start_8 + start_9 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 60 switch_9_8 <= 60
 edge_disable_constr2_9_8: start_8 - start_9 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   + 60 switch_9_8 <= 60
 edge_disable_constr1_0_2: start_0 - start_2 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 60 switch_0_2 <= 60
 edge_disable_constr2_0_2: - start_0 + start_2 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   + 60 switch_0_2 <= 60
 edge_disable_constr1_2_0: - start_0 + start_2 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 60 switch_2_0
   <= 60
 edge_disable_constr2_2_0: start_0 - start_2 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 + 60 switch_2_0
   <= 60
 edge_disable_constr1_0_12: start_0 - start_12 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 60 switch_0_12 <= 60
 edge_disable_constr2_0_12: - start_0 + start_12 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   + 60 switch_0_12 <= 60
 edge_disable_constr1_12_0: - start_0 + start_12 + hold_12_1 + 2 hold_12_2
   + 3 hold_12_3 + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6 + 7 hold_12_7
   + 60 switch_12_0 <= 60
 edge_disable_constr2_12_0: start_0 - start_12 - hold_12_1 - 2 hold_12_2
   - 3 hold_12_3 - 4 hold_12_4 - 5 hold_12_5 - 6 hold_12_6 - 7 hold_12_7
   + 60 switch_12_0 <= 60
 edge_disable_constr1_11_12: start_11 - start_12 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 60 switch_11_12 <= 60
 edge_disable_constr2_11_12: - start_11 + start_12 - hold_11_1
   - 2 hold_11_2 - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6
   - 7 hold_11_7 - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10
   - 11 hold_11_11 + 60 switch_11_12 <= 60
 edge_disable_constr1_12_11: - start_11 + start_12 + hold_12_1
   + 2 hold_12_2 + 3 hold_12_3 + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6
   + 7 hold_12_7 + 60 switch_12_11 <= 60
 edge_disable_constr2_12_11: start_11 - start_12 - hold_12_1 - 2 hold_12_2
   - 3 hold_12_3 - 4 hold_12_4 - 5 hold_12_5 - 6 hold_12_6 - 7 hold_12_7
   + 60 switch_12_11 <= 60
 edge_disable_constr1_0_1: start_0 - start_1 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 60 switch_0_1 <= 60
 edge_disable_constr2_0_1: - start_0 + start_1 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   + 60 switch_0_1 <= 60
 edge_disable_constr1_1_0: - start_0 + start_1 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 60 switch_1_0 <= 60
 edge_disable_constr2_1_0: start_0 - start_1 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   + 60 switch_1_0 <= 60
 edge_disable_constr1_4_14: start_4 - start_14 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 60 switch_4_14 <= 60
 edge_disable_constr2_4_14: - start_4 + start_14 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   + 60 switch_4_14 <= 60
 edge_disable_constr1_14_4: - start_4 + start_14 + hold_14_1 + 2 hold_14_2
   + 3 hold_14_3 + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6 + 7 hold_14_7
   + 8 hold_14_8 + 9 hold_14_9 + 10 hold_14_10 + 11 hold_14_11
   + 60 switch_14_4 <= 60
 edge_disable_constr2_14_4: start_4 - start_14 - hold_14_1 - 2 hold_14_2
   - 3 hold_14_3 - 4 hold_14_4 - 5 hold_14_5 - 6 hold_14_6 - 7 hold_14_7
   - 8 hold_14_8 - 9 hold_14_9 - 10 hold_14_10 - 11 hold_14_11
   + 60 switch_14_4 <= 60
 edge_disable_constr1_7_10: start_7 - start_10 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 60 switch_7_10 <= 60
 edge_disable_constr2_7_10: - start_7 + start_10 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   + 60 switch_7_10 <= 60
 edge_disable_constr1_10_7: - start_7 + start_10 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 12 hold_10_12 + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15
   + 16 hold_10_16 + 17 hold_10_17 + 18 hold_10_18 + 60 switch_10_7 <= 60
 edge_disable_constr2_10_7: start_7 - start_10 - hold_10_1 - 2 hold_10_2
   - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6 - 7 hold_10_7
   - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10 - 11 hold_10_11
   - 12 hold_10_12 - 13 hold_10_13 - 14 hold_10_14 - 15 hold_10_15
   - 16 hold_10_16 - 17 hold_10_17 - 18 hold_10_18 + 60 switch_10_7 <= 60
 edge_disable_constr1_2_5: start_2 - start_5 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 60 switch_2_5
   <= 60
 edge_disable_constr2_2_5: - start_2 + start_5 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 + 60 switch_2_5
   <= 60
 edge_disable_constr1_5_2: - start_2 + start_5 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12
   + 13 hold_5_13 + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16
   + 17 hold_5_17 + 18 hold_5_18 + 60 switch_5_2 <= 60
 edge_disable_constr2_5_2: start_2 - start_5 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11 - 12 hold_5_12
   - 13 hold_5_13 - 14 hold_5_14 - 15 hold_5_15 - 16 hold_5_16
   - 17 hold_5_17 - 18 hold_5_18 + 60 switch_5_2 <= 60
 edge_disable_constr1_2_15: start_2 - start_15 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 60 switch_2_15
   <= 60
 edge_disable_constr2_2_15: - start_2 + start_15 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 + 60 switch_2_15
   <= 60
 edge_disable_constr1_15_2: - start_2 + start_15 + hold_15_1 + 2 hold_15_2
   + 3 hold_15_3 + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6 + 7 hold_15_7
   + 60 switch_15_2 <= 60
 edge_disable_constr2_15_2: start_2 - start_15 - hold_15_1 - 2 hold_15_2
   - 3 hold_15_3 - 4 hold_15_4 - 5 hold_15_5 - 6 hold_15_6 - 7 hold_15_7
   + 60 switch_15_2 <= 60
 edge_disable_constr1_6_14: start_6 - start_14 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 60 switch_6_14
   <= 60
 edge_disable_constr2_6_14: - start_6 + start_14 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11 + 60 switch_6_14
   <= 60
 edge_disable_constr1_14_6: - start_6 + start_14 + hold_14_1 + 2 hold_14_2
   + 3 hold_14_3 + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6 + 7 hold_14_7
   + 8 hold_14_8 + 9 hold_14_9 + 10 hold_14_10 + 11 hold_14_11
   + 60 switch_14_6 <= 60
 edge_disable_constr2_14_6: start_6 - start_14 - hold_14_1 - 2 hold_14_2
   - 3 hold_14_3 - 4 hold_14_4 - 5 hold_14_5 - 6 hold_14_6 - 7 hold_14_7
   - 8 hold_14_8 - 9 hold_14_9 - 10 hold_14_10 - 11 hold_14_11
   + 60 switch_14_6 <= 60
 edge_disable_constr1_6_13: start_6 - start_13 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 60 switch_6_13
   <= 60
 edge_disable_constr2_6_13: - start_6 + start_13 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11 + 60 switch_6_13
   <= 60
 edge_disable_constr1_13_6: - start_6 + start_13 + hold_13_1 + 2 hold_13_2
   + 3 hold_13_3 + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6 + 7 hold_13_7
   + 8 hold_13_8 + 9 hold_13_9 + 10 hold_13_10 + 11 hold_13_11
   + 60 switch_13_6 <= 60
 edge_disable_constr2_13_6: start_6 - start_13 - hold_13_1 - 2 hold_13_2
   - 3 hold_13_3 - 4 hold_13_4 - 5 hold_13_5 - 6 hold_13_6 - 7 hold_13_7
   - 8 hold_13_8 - 9 hold_13_9 - 10 hold_13_10 - 11 hold_13_11
   + 60 switch_13_6 <= 60
 edge_disable_constr1_0_5: start_0 - start_5 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 60 switch_0_5 <= 60
 edge_disable_constr2_0_5: - start_0 + start_5 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   + 60 switch_0_5 <= 60
 edge_disable_constr1_5_0: - start_0 + start_5 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12
   + 13 hold_5_13 + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16
   + 17 hold_5_17 + 18 hold_5_18 + 60 switch_5_0 <= 60
 edge_disable_constr2_5_0: start_0 - start_5 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11 - 12 hold_5_12
   - 13 hold_5_13 - 14 hold_5_14 - 15 hold_5_15 - 16 hold_5_16
   - 17 hold_5_17 - 18 hold_5_18 + 60 switch_5_0 <= 60
 edge_disable_constr1_6_15: start_6 - start_15 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 60 switch_6_15
   <= 60
 edge_disable_constr2_6_15: - start_6 + start_15 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11 + 60 switch_6_15
   <= 60
 edge_disable_constr1_15_6: - start_6 + start_15 + hold_15_1 + 2 hold_15_2
   + 3 hold_15_3 + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6 + 7 hold_15_7
   + 60 switch_15_6 <= 60
 edge_disable_constr2_15_6: start_6 - start_15 - hold_15_1 - 2 hold_15_2
   - 3 hold_15_3 - 4 hold_15_4 - 5 hold_15_5 - 6 hold_15_6 - 7 hold_15_7
   + 60 switch_15_6 <= 60
 edge_disable_constr1_4_9: start_4 - start_9 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 60 switch_4_9 <= 60
 edge_disable_constr2_4_9: - start_4 + start_9 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   + 60 switch_4_9 <= 60
 edge_disable_constr1_9_4: - start_4 + start_9 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 60 switch_9_4 <= 60
 edge_disable_constr2_9_4: start_4 - start_9 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   + 60 switch_9_4 <= 60
 edge_disable_constr1_10_15: start_10 - start_15 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 12 hold_10_12 + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15
   + 16 hold_10_16 + 17 hold_10_17 + 18 hold_10_18 + 60 switch_10_15 <= 60
 edge_disable_constr2_10_15: - start_10 + start_15 - hold_10_1
   - 2 hold_10_2 - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6
   - 7 hold_10_7 - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10
   - 11 hold_10_11 - 12 hold_10_12 - 13 hold_10_13 - 14 hold_10_14
   - 15 hold_10_15 - 16 hold_10_16 - 17 hold_10_17 - 18 hold_10_18
   + 60 switch_10_15 <= 60
 edge_disable_constr1_15_10: - start_10 + start_15 + hold_15_1
   + 2 hold_15_2 + 3 hold_15_3 + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6
   + 7 hold_15_7 + 60 switch_15_10 <= 60
 edge_disable_constr2_15_10: start_10 - start_15 - hold_15_1 - 2 hold_15_2
   - 3 hold_15_3 - 4 hold_15_4 - 5 hold_15_5 - 6 hold_15_6 - 7 hold_15_7
   + 60 switch_15_10 <= 60
 edge_disable_constr1_0_15: start_0 - start_15 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 60 switch_0_15 <= 60
 edge_disable_constr2_0_15: - start_0 + start_15 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   + 60 switch_0_15 <= 60
 edge_disable_constr1_15_0: - start_0 + start_15 + hold_15_1 + 2 hold_15_2
   + 3 hold_15_3 + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6 + 7 hold_15_7
   + 60 switch_15_0 <= 60
 edge_disable_constr2_15_0: start_0 - start_15 - hold_15_1 - 2 hold_15_2
   - 3 hold_15_3 - 4 hold_15_4 - 5 hold_15_5 - 6 hold_15_6 - 7 hold_15_7
   + 60 switch_15_0 <= 60
 edge_disable_constr1_1_2: start_1 - start_2 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 60 switch_1_2 <= 60
 edge_disable_constr2_1_2: - start_1 + start_2 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   + 60 switch_1_2 <= 60
 edge_disable_constr1_2_1: - start_1 + start_2 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 60 switch_2_1
   <= 60
 edge_disable_constr2_2_1: start_1 - start_2 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 + 60 switch_2_1
   <= 60
 edge_disable_constr1_8_11: start_8 - start_11 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 60 switch_8_11
   <= 60
 edge_disable_constr2_8_11: - start_8 + start_11 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11 + 60 switch_8_11
   <= 60
 edge_disable_constr1_11_8: - start_8 + start_11 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 60 switch_11_8 <= 60
 edge_disable_constr2_11_8: start_8 - start_11 - hold_11_1 - 2 hold_11_2
   - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6 - 7 hold_11_7
   - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10 - 11 hold_11_11
   + 60 switch_11_8 <= 60
 edge_disable_constr1_7_12: start_7 - start_12 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 60 switch_7_12 <= 60
 edge_disable_constr2_7_12: - start_7 + start_12 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   + 60 switch_7_12 <= 60
 edge_disable_constr1_12_7: - start_7 + start_12 + hold_12_1 + 2 hold_12_2
   + 3 hold_12_3 + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6 + 7 hold_12_7
   + 60 switch_12_7 <= 60
 edge_disable_constr2_12_7: start_7 - start_12 - hold_12_1 - 2 hold_12_2
   - 3 hold_12_3 - 4 hold_12_4 - 5 hold_12_5 - 6 hold_12_6 - 7 hold_12_7
   + 60 switch_12_7 <= 60
 edge_disable_constr1_4_5: start_4 - start_5 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 60 switch_4_5 <= 60
 edge_disable_constr2_4_5: - start_4 + start_5 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   + 60 switch_4_5 <= 60
 edge_disable_constr1_5_4: - start_4 + start_5 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12
   + 13 hold_5_13 + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16
   + 17 hold_5_17 + 18 hold_5_18 + 60 switch_5_4 <= 60
 edge_disable_constr2_5_4: start_4 - start_5 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11 - 12 hold_5_12
   - 13 hold_5_13 - 14 hold_5_14 - 15 hold_5_15 - 16 hold_5_16
   - 17 hold_5_17 - 18 hold_5_18 + 60 switch_5_4 <= 60
 edge_disable_constr1_2_8: start_2 - start_8 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 60 switch_2_8
   <= 60
 edge_disable_constr2_2_8: - start_2 + start_8 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 + 60 switch_2_8
   <= 60
 edge_disable_constr1_8_2: - start_2 + start_8 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 60 switch_8_2
   <= 60
 edge_disable_constr2_8_2: start_2 - start_8 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11 + 60 switch_8_2
   <= 60
 edge_disable_constr1_1_15: start_1 - start_15 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 60 switch_1_15 <= 60
 edge_disable_constr2_1_15: - start_1 + start_15 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   + 60 switch_1_15 <= 60
 edge_disable_constr1_15_1: - start_1 + start_15 + hold_15_1 + 2 hold_15_2
   + 3 hold_15_3 + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6 + 7 hold_15_7
   + 60 switch_15_1 <= 60
 edge_disable_constr2_15_1: start_1 - start_15 - hold_15_1 - 2 hold_15_2
   - 3 hold_15_3 - 4 hold_15_4 - 5 hold_15_5 - 6 hold_15_6 - 7 hold_15_7
   + 60 switch_15_1 <= 60
 edge_disable_constr1_12_13: start_12 - start_13 + hold_12_1 + 2 hold_12_2
   + 3 hold_12_3 + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6 + 7 hold_12_7
   + 60 switch_12_13 <= 60
 edge_disable_constr2_12_13: - start_12 + start_13 - hold_12_1
   - 2 hold_12_2 - 3 hold_12_3 - 4 hold_12_4 - 5 hold_12_5 - 6 hold_12_6
   - 7 hold_12_7 + 60 switch_12_13 <= 60
 edge_disable_constr1_13_12: - start_12 + start_13 + hold_13_1
   + 2 hold_13_2 + 3 hold_13_3 + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6
   + 7 hold_13_7 + 8 hold_13_8 + 9 hold_13_9 + 10 hold_13_10
   + 11 hold_13_11 + 60 switch_13_12 <= 60
 edge_disable_constr2_13_12: start_12 - start_13 - hold_13_1 - 2 hold_13_2
   - 3 hold_13_3 - 4 hold_13_4 - 5 hold_13_5 - 6 hold_13_6 - 7 hold_13_7
   - 8 hold_13_8 - 9 hold_13_9 - 10 hold_13_10 - 11 hold_13_11
   + 60 switch_13_12 <= 60
 edge_disable_constr1_13_15: start_13 - start_15 + hold_13_1 + 2 hold_13_2
   + 3 hold_13_3 + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6 + 7 hold_13_7
   + 8 hold_13_8 + 9 hold_13_9 + 10 hold_13_10 + 11 hold_13_11
   + 60 switch_13_15 <= 60
 edge_disable_constr2_13_15: - start_13 + start_15 - hold_13_1
   - 2 hold_13_2 - 3 hold_13_3 - 4 hold_13_4 - 5 hold_13_5 - 6 hold_13_6
   - 7 hold_13_7 - 8 hold_13_8 - 9 hold_13_9 - 10 hold_13_10
   - 11 hold_13_11 + 60 switch_13_15 <= 60
 edge_disable_constr1_15_13: - start_13 + start_15 + hold_15_1
   + 2 hold_15_2 + 3 hold_15_3 + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6
   + 7 hold_15_7 + 60 switch_15_13 <= 60
 edge_disable_constr2_15_13: start_13 - start_15 - hold_15_1 - 2 hold_15_2
   - 3 hold_15_3 - 4 hold_15_4 - 5 hold_15_5 - 6 hold_15_6 - 7 hold_15_7
   + 60 switch_15_13 <= 60
 edge_disable_constr1_1_7: start_1 - start_7 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 60 switch_1_7 <= 60
 edge_disable_constr2_1_7: - start_1 + start_7 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   + 60 switch_1_7 <= 60
 edge_disable_constr1_7_1: - start_1 + start_7 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 60 switch_7_1 <= 60
 edge_disable_constr2_7_1: start_1 - start_7 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   + 60 switch_7_1 <= 60
 edge_disable_constr1_8_12: start_8 - start_12 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 60 switch_8_12
   <= 60
 edge_disable_constr2_8_12: - start_8 + start_12 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11 + 60 switch_8_12
   <= 60
 edge_disable_constr1_12_8: - start_8 + start_12 + hold_12_1 + 2 hold_12_2
   + 3 hold_12_3 + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6 + 7 hold_12_7
   + 60 switch_12_8 <= 60
 edge_disable_constr2_12_8: start_8 - start_12 - hold_12_1 - 2 hold_12_2
   - 3 hold_12_3 - 4 hold_12_4 - 5 hold_12_5 - 6 hold_12_6 - 7 hold_12_7
   + 60 switch_12_8 <= 60
 edge_disable_constr1_9_13: start_9 - start_13 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 60 switch_9_13 <= 60
 edge_disable_constr2_9_13: - start_9 + start_13 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   + 60 switch_9_13 <= 60
 edge_disable_constr1_13_9: - start_9 + start_13 + hold_13_1 + 2 hold_13_2
   + 3 hold_13_3 + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6 + 7 hold_13_7
   + 8 hold_13_8 + 9 hold_13_9 + 10 hold_13_10 + 11 hold_13_11
   + 60 switch_13_9 <= 60
 edge_disable_constr2_13_9: start_9 - start_13 - hold_13_1 - 2 hold_13_2
   - 3 hold_13_3 - 4 hold_13_4 - 5 hold_13_5 - 6 hold_13_6 - 7 hold_13_7
   - 8 hold_13_8 - 9 hold_13_9 - 10 hold_13_10 - 11 hold_13_11
   + 60 switch_13_9 <= 60
 edge_disable_constr1_0_7: start_0 - start_7 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 60 switch_0_7 <= 60
 edge_disable_constr2_0_7: - start_0 + start_7 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   + 60 switch_0_7 <= 60
 edge_disable_constr1_7_0: - start_0 + start_7 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 60 switch_7_0 <= 60
 edge_disable_constr2_7_0: start_0 - start_7 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   + 60 switch_7_0 <= 60
 edge_disable_constr1_1_6: start_1 - start_6 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 60 switch_1_6 <= 60
 edge_disable_constr2_1_6: - start_1 + start_6 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   + 60 switch_1_6 <= 60
 edge_disable_constr1_6_1: - start_1 + start_6 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 60 switch_6_1
   <= 60
 edge_disable_constr2_6_1: start_1 - start_6 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11 + 60 switch_6_1
   <= 60
 edge_disable_constr1_2_9: start_2 - start_9 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 60 switch_2_9
   <= 60
 edge_disable_constr2_2_9: - start_2 + start_9 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 + 60 switch_2_9
   <= 60
 edge_disable_constr1_9_2: - start_2 + start_9 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 60 switch_9_2 <= 60
 edge_disable_constr2_9_2: start_2 - start_9 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   + 60 switch_9_2 <= 60
 edge_disable_constr1_7_15: start_7 - start_15 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 60 switch_7_15 <= 60
 edge_disable_constr2_7_15: - start_7 + start_15 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   + 60 switch_7_15 <= 60
 edge_disable_constr1_15_7: - start_7 + start_15 + hold_15_1 + 2 hold_15_2
   + 3 hold_15_3 + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6 + 7 hold_15_7
   + 60 switch_15_7 <= 60
 edge_disable_constr2_15_7: start_7 - start_15 - hold_15_1 - 2 hold_15_2
   - 3 hold_15_3 - 4 hold_15_4 - 5 hold_15_5 - 6 hold_15_6 - 7 hold_15_7
   + 60 switch_15_7 <= 60
 edge_disable_constr1_5_13: start_5 - start_13 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12
   + 13 hold_5_13 + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16
   + 17 hold_5_17 + 18 hold_5_18 + 60 switch_5_13 <= 60
 edge_disable_constr2_5_13: - start_5 + start_13 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11 - 12 hold_5_12
   - 13 hold_5_13 - 14 hold_5_14 - 15 hold_5_15 - 16 hold_5_16
   - 17 hold_5_17 - 18 hold_5_18 + 60 switch_5_13 <= 60
 edge_disable_constr1_13_5: - start_5 + start_13 + hold_13_1 + 2 hold_13_2
   + 3 hold_13_3 + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6 + 7 hold_13_7
   + 8 hold_13_8 + 9 hold_13_9 + 10 hold_13_10 + 11 hold_13_11
   + 60 switch_13_5 <= 60
 edge_disable_constr2_13_5: start_5 - start_13 - hold_13_1 - 2 hold_13_2
   - 3 hold_13_3 - 4 hold_13_4 - 5 hold_13_5 - 6 hold_13_6 - 7 hold_13_7
   - 8 hold_13_8 - 9 hold_13_9 - 10 hold_13_10 - 11 hold_13_11
   + 60 switch_13_5 <= 60
 edge_disable_constr1_11_14: start_11 - start_14 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 60 switch_11_14 <= 60
 edge_disable_constr2_11_14: - start_11 + start_14 - hold_11_1
   - 2 hold_11_2 - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6
   - 7 hold_11_7 - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10
   - 11 hold_11_11 + 60 switch_11_14 <= 60
 edge_disable_constr1_14_11: - start_11 + start_14 + hold_14_1
   + 2 hold_14_2 + 3 hold_14_3 + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6
   + 7 hold_14_7 + 8 hold_14_8 + 9 hold_14_9 + 10 hold_14_10
   + 11 hold_14_11 + 60 switch_14_11 <= 60
 edge_disable_constr2_14_11: start_11 - start_14 - hold_14_1 - 2 hold_14_2
   - 3 hold_14_3 - 4 hold_14_4 - 5 hold_14_5 - 6 hold_14_6 - 7 hold_14_7
   - 8 hold_14_8 - 9 hold_14_9 - 10 hold_14_10 - 11 hold_14_11
   + 60 switch_14_11 <= 60
 edge_disable_constr1_1_8: start_1 - start_8 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 60 switch_1_8 <= 60
 edge_disable_constr2_1_8: - start_1 + start_8 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   + 60 switch_1_8 <= 60
 edge_disable_constr1_8_1: - start_1 + start_8 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 60 switch_8_1
   <= 60
 edge_disable_constr2_8_1: start_1 - start_8 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11 + 60 switch_8_1
   <= 60
 edge_disable_constr1_3_14: start_3 - start_14 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 60 switch_3_14 <= 60
 edge_disable_constr2_3_14: - start_3 + start_14 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   + 60 switch_3_14 <= 60
 edge_disable_constr1_14_3: - start_3 + start_14 + hold_14_1 + 2 hold_14_2
   + 3 hold_14_3 + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6 + 7 hold_14_7
   + 8 hold_14_8 + 9 hold_14_9 + 10 hold_14_10 + 11 hold_14_11
   + 60 switch_14_3 <= 60
 edge_disable_constr2_14_3: start_3 - start_14 - hold_14_1 - 2 hold_14_2
   - 3 hold_14_3 - 4 hold_14_4 - 5 hold_14_5 - 6 hold_14_6 - 7 hold_14_7
   - 8 hold_14_8 - 9 hold_14_9 - 10 hold_14_10 - 11 hold_14_11
   + 60 switch_14_3 <= 60
 edge_disable_constr1_0_14: start_0 - start_14 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 60 switch_0_14 <= 60
 edge_disable_constr2_0_14: - start_0 + start_14 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   + 60 switch_0_14 <= 60
 edge_disable_constr1_14_0: - start_0 + start_14 + hold_14_1 + 2 hold_14_2
   + 3 hold_14_3 + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6 + 7 hold_14_7
   + 8 hold_14_8 + 9 hold_14_9 + 10 hold_14_10 + 11 hold_14_11
   + 60 switch_14_0 <= 60
 edge_disable_constr2_14_0: start_0 - start_14 - hold_14_1 - 2 hold_14_2
   - 3 hold_14_3 - 4 hold_14_4 - 5 hold_14_5 - 6 hold_14_6 - 7 hold_14_7
   - 8 hold_14_8 - 9 hold_14_9 - 10 hold_14_10 - 11 hold_14_11
   + 60 switch_14_0 <= 60
 edge_disable_constr1_8_10: start_8 - start_10 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 60 switch_8_10
   <= 60
 edge_disable_constr2_8_10: - start_8 + start_10 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11 + 60 switch_8_10
   <= 60
 edge_disable_constr1_10_8: - start_8 + start_10 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 12 hold_10_12 + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15
   + 16 hold_10_16 + 17 hold_10_17 + 18 hold_10_18 + 60 switch_10_8 <= 60
 edge_disable_constr2_10_8: start_8 - start_10 - hold_10_1 - 2 hold_10_2
   - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6 - 7 hold_10_7
   - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10 - 11 hold_10_11
   - 12 hold_10_12 - 13 hold_10_13 - 14 hold_10_14 - 15 hold_10_15
   - 16 hold_10_16 - 17 hold_10_17 - 18 hold_10_18 + 60 switch_10_8 <= 60
 edge_disable_constr1_9_15: start_9 - start_15 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 60 switch_9_15 <= 60
 edge_disable_constr2_9_15: - start_9 + start_15 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   + 60 switch_9_15 <= 60
 edge_disable_constr1_15_9: - start_9 + start_15 + hold_15_1 + 2 hold_15_2
   + 3 hold_15_3 + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6 + 7 hold_15_7
   + 60 switch_15_9 <= 60
 edge_disable_constr2_15_9: start_9 - start_15 - hold_15_1 - 2 hold_15_2
   - 3 hold_15_3 - 4 hold_15_4 - 5 hold_15_5 - 6 hold_15_6 - 7 hold_15_7
   + 60 switch_15_9 <= 60
 edge_disable_constr1_3_12: start_3 - start_12 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 60 switch_3_12 <= 60
 edge_disable_constr2_3_12: - start_3 + start_12 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   + 60 switch_3_12 <= 60
 edge_disable_constr1_12_3: - start_3 + start_12 + hold_12_1 + 2 hold_12_2
   + 3 hold_12_3 + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6 + 7 hold_12_7
   + 60 switch_12_3 <= 60
 edge_disable_constr2_12_3: start_3 - start_12 - hold_12_1 - 2 hold_12_2
   - 3 hold_12_3 - 4 hold_12_4 - 5 hold_12_5 - 6 hold_12_6 - 7 hold_12_7
   + 60 switch_12_3 <= 60
 edge_disable_constr1_3_10: start_3 - start_10 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 60 switch_3_10 <= 60
 edge_disable_constr2_3_10: - start_3 + start_10 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   + 60 switch_3_10 <= 60
 edge_disable_constr1_10_3: - start_3 + start_10 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 12 hold_10_12 + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15
   + 16 hold_10_16 + 17 hold_10_17 + 18 hold_10_18 + 60 switch_10_3 <= 60
 edge_disable_constr2_10_3: start_3 - start_10 - hold_10_1 - 2 hold_10_2
   - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6 - 7 hold_10_7
   - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10 - 11 hold_10_11
   - 12 hold_10_12 - 13 hold_10_13 - 14 hold_10_14 - 15 hold_10_15
   - 16 hold_10_16 - 17 hold_10_17 - 18 hold_10_18 + 60 switch_10_3 <= 60
 edge_disable_constr1_1_11: start_1 - start_11 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 60 switch_1_11 <= 60
 edge_disable_constr2_1_11: - start_1 + start_11 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   + 60 switch_1_11 <= 60
 edge_disable_constr1_11_1: - start_1 + start_11 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 60 switch_11_1 <= 60
 edge_disable_constr2_11_1: start_1 - start_11 - hold_11_1 - 2 hold_11_2
   - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6 - 7 hold_11_7
   - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10 - 11 hold_11_11
   + 60 switch_11_1 <= 60
 edge_disable_constr1_0_11: start_0 - start_11 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 60 switch_0_11 <= 60
 edge_disable_constr2_0_11: - start_0 + start_11 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   + 60 switch_0_11 <= 60
 edge_disable_constr1_11_0: - start_0 + start_11 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 60 switch_11_0 <= 60
 edge_disable_constr2_11_0: start_0 - start_11 - hold_11_1 - 2 hold_11_2
   - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6 - 7 hold_11_7
   - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10 - 11 hold_11_11
   + 60 switch_11_0 <= 60
 edge_disable_constr1_2_11: start_2 - start_11 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 60 switch_2_11
   <= 60
 edge_disable_constr2_2_11: - start_2 + start_11 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 + 60 switch_2_11
   <= 60
 edge_disable_constr1_11_2: - start_2 + start_11 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 60 switch_11_2 <= 60
 edge_disable_constr2_11_2: start_2 - start_11 - hold_11_1 - 2 hold_11_2
   - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6 - 7 hold_11_7
   - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10 - 11 hold_11_11
   + 60 switch_11_2 <= 60
 edge_disable_constr1_6_12: start_6 - start_12 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 60 switch_6_12
   <= 60
 edge_disable_constr2_6_12: - start_6 + start_12 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11 + 60 switch_6_12
   <= 60
 edge_disable_constr1_12_6: - start_6 + start_12 + hold_12_1 + 2 hold_12_2
   + 3 hold_12_3 + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6 + 7 hold_12_7
   + 60 switch_12_6 <= 60
 edge_disable_constr2_12_6: start_6 - start_12 - hold_12_1 - 2 hold_12_2
   - 3 hold_12_3 - 4 hold_12_4 - 5 hold_12_5 - 6 hold_12_6 - 7 hold_12_7
   + 60 switch_12_6 <= 60
 edge_disable_constr1_12_14: start_12 - start_14 + hold_12_1 + 2 hold_12_2
   + 3 hold_12_3 + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6 + 7 hold_12_7
   + 60 switch_12_14 <= 60
 edge_disable_constr2_12_14: - start_12 + start_14 - hold_12_1
   - 2 hold_12_2 - 3 hold_12_3 - 4 hold_12_4 - 5 hold_12_5 - 6 hold_12_6
   - 7 hold_12_7 + 60 switch_12_14 <= 60
 edge_disable_constr1_14_12: - start_12 + start_14 + hold_14_1
   + 2 hold_14_2 + 3 hold_14_3 + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6
   + 7 hold_14_7 + 8 hold_14_8 + 9 hold_14_9 + 10 hold_14_10
   + 11 hold_14_11 + 60 switch_14_12 <= 60
 edge_disable_constr2_14_12: start_12 - start_14 - hold_14_1 - 2 hold_14_2
   - 3 hold_14_3 - 4 hold_14_4 - 5 hold_14_5 - 6 hold_14_6 - 7 hold_14_7
   - 8 hold_14_8 - 9 hold_14_9 - 10 hold_14_10 - 11 hold_14_11
   + 60 switch_14_12 <= 60
 edge_disable_constr1_3_8: start_3 - start_8 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 60 switch_3_8 <= 60
 edge_disable_constr2_3_8: - start_3 + start_8 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   + 60 switch_3_8 <= 60
 edge_disable_constr1_8_3: - start_3 + start_8 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 60 switch_8_3
   <= 60
 edge_disable_constr2_8_3: start_3 - start_8 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11 + 60 switch_8_3
   <= 60
 edge_disable_constr1_3_6: start_3 - start_6 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 60 switch_3_6 <= 60
 edge_disable_constr2_3_6: - start_3 + start_6 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   + 60 switch_3_6 <= 60
 edge_disable_constr1_6_3: - start_3 + start_6 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 60 switch_6_3
   <= 60
 edge_disable_constr2_6_3: start_3 - start_6 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11 + 60 switch_6_3
   <= 60
 edge_disable_constr1_8_15: start_8 - start_15 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 60 switch_8_15
   <= 60
 edge_disable_constr2_8_15: - start_8 + start_15 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11 + 60 switch_8_15
   <= 60
 edge_disable_constr1_15_8: - start_8 + start_15 + hold_15_1 + 2 hold_15_2
   + 3 hold_15_3 + 4 hold_15_4 + 5 hold_15_5 + 6 hold_15_6 + 7 hold_15_7
   + 60 switch_15_8 <= 60
 edge_disable_constr2_15_8: start_8 - start_15 - hold_15_1 - 2 hold_15_2
   - 3 hold_15_3 - 4 hold_15_4 - 5 hold_15_5 - 6 hold_15_6 - 7 hold_15_7
   + 60 switch_15_8 <= 60
 edge_disable_constr1_7_8: start_7 - start_8 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 60 switch_7_8 <= 60
 edge_disable_constr2_7_8: - start_7 + start_8 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   + 60 switch_7_8 <= 60
 edge_disable_constr1_8_7: - start_7 + start_8 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 60 switch_8_7
   <= 60
 edge_disable_constr2_8_7: start_7 - start_8 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11 + 60 switch_8_7
   <= 60
 edge_disable_constr1_3_5: start_3 - start_5 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 60 switch_3_5 <= 60
 edge_disable_constr2_3_5: - start_3 + start_5 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   + 60 switch_3_5 <= 60
 edge_disable_constr1_5_3: - start_3 + start_5 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12
   + 13 hold_5_13 + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16
   + 17 hold_5_17 + 18 hold_5_18 + 60 switch_5_3 <= 60
 edge_disable_constr2_5_3: start_3 - start_5 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11 - 12 hold_5_12
   - 13 hold_5_13 - 14 hold_5_14 - 15 hold_5_15 - 16 hold_5_16
   - 17 hold_5_17 - 18 hold_5_18 + 60 switch_5_3 <= 60
 edge_disable_constr1_10_13: start_10 - start_13 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 12 hold_10_12 + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15
   + 16 hold_10_16 + 17 hold_10_17 + 18 hold_10_18 + 60 switch_10_13 <= 60
 edge_disable_constr2_10_13: - start_10 + start_13 - hold_10_1
   - 2 hold_10_2 - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6
   - 7 hold_10_7 - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10
   - 11 hold_10_11 - 12 hold_10_12 - 13 hold_10_13 - 14 hold_10_14
   - 15 hold_10_15 - 16 hold_10_16 - 17 hold_10_17 - 18 hold_10_18
   + 60 switch_10_13 <= 60
 edge_disable_constr1_13_10: - start_10 + start_13 + hold_13_1
   + 2 hold_13_2 + 3 hold_13_3 + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6
   + 7 hold_13_7 + 8 hold_13_8 + 9 hold_13_9 + 10 hold_13_10
   + 11 hold_13_11 + 60 switch_13_10 <= 60
 edge_disable_constr2_13_10: start_10 - start_13 - hold_13_1 - 2 hold_13_2
   - 3 hold_13_3 - 4 hold_13_4 - 5 hold_13_5 - 6 hold_13_6 - 7 hold_13_7
   - 8 hold_13_8 - 9 hold_13_9 - 10 hold_13_10 - 11 hold_13_11
   + 60 switch_13_10 <= 60
 edge_disable_constr1_0_9: start_0 - start_9 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 60 switch_0_9 <= 60
 edge_disable_constr2_0_9: - start_0 + start_9 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   + 60 switch_0_9 <= 60
 edge_disable_constr1_9_0: - start_0 + start_9 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 60 switch_9_0 <= 60
 edge_disable_constr2_9_0: start_0 - start_9 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   + 60 switch_9_0 <= 60
 edge_disable_constr1_6_8: start_6 - start_8 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 60 switch_6_8
   <= 60
 edge_disable_constr2_6_8: - start_6 + start_8 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11 + 60 switch_6_8
   <= 60
 edge_disable_constr1_8_6: - start_6 + start_8 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 60 switch_8_6
   <= 60
 edge_disable_constr2_8_6: start_6 - start_8 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11 + 60 switch_8_6
   <= 60
 edge_disable_constr1_0_3: start_0 - start_3 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 60 switch_0_3 <= 60
 edge_disable_constr2_0_3: - start_0 + start_3 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   + 60 switch_0_3 <= 60
 edge_disable_constr1_3_0: - start_0 + start_3 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 60 switch_3_0 <= 60
 edge_disable_constr2_3_0: start_0 - start_3 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   + 60 switch_3_0 <= 60
 edge_disable_constr1_2_3: start_2 - start_3 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 60 switch_2_3
   <= 60
 edge_disable_constr2_2_3: - start_2 + start_3 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 + 60 switch_2_3
   <= 60
 edge_disable_constr1_3_2: - start_2 + start_3 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 60 switch_3_2 <= 60
 edge_disable_constr2_3_2: start_2 - start_3 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   + 60 switch_3_2 <= 60
 edge_disable_constr1_7_9: start_7 - start_9 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 60 switch_7_9 <= 60
 edge_disable_constr2_7_9: - start_7 + start_9 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   + 60 switch_7_9 <= 60
 edge_disable_constr1_9_7: - start_7 + start_9 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 60 switch_9_7 <= 60
 edge_disable_constr2_9_7: start_7 - start_9 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   + 60 switch_9_7 <= 60
 edge_disable_constr1_1_3: start_1 - start_3 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 60 switch_1_3 <= 60
 edge_disable_constr2_1_3: - start_1 + start_3 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   + 60 switch_1_3 <= 60
 edge_disable_constr1_3_1: - start_1 + start_3 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 60 switch_3_1 <= 60
 edge_disable_constr2_3_1: start_1 - start_3 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   + 60 switch_3_1 <= 60
 edge_disable_constr1_6_7: start_6 - start_7 + hold_6_1 + 2 hold_6_2
   + 3 hold_6_3 + 4 hold_6_4 + 5 hold_6_5 + 6 hold_6_6 + 7 hold_6_7
   + 8 hold_6_8 + 9 hold_6_9 + 10 hold_6_10 + 11 hold_6_11 + 60 switch_6_7
   <= 60
 edge_disable_constr2_6_7: - start_6 + start_7 - hold_6_1 - 2 hold_6_2
   - 3 hold_6_3 - 4 hold_6_4 - 5 hold_6_5 - 6 hold_6_6 - 7 hold_6_7
   - 8 hold_6_8 - 9 hold_6_9 - 10 hold_6_10 - 11 hold_6_11 + 60 switch_6_7
   <= 60
 edge_disable_constr1_7_6: - start_6 + start_7 + hold_7_1 + 2 hold_7_2
   + 3 hold_7_3 + 4 hold_7_4 + 5 hold_7_5 + 6 hold_7_6 + 7 hold_7_7
   + 60 switch_7_6 <= 60
 edge_disable_constr2_7_6: start_6 - start_7 - hold_7_1 - 2 hold_7_2
   - 3 hold_7_3 - 4 hold_7_4 - 5 hold_7_5 - 6 hold_7_6 - 7 hold_7_7
   + 60 switch_7_6 <= 60
 edge_disable_constr1_5_8: start_5 - start_8 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12
   + 13 hold_5_13 + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16
   + 17 hold_5_17 + 18 hold_5_18 + 60 switch_5_8 <= 60
 edge_disable_constr2_5_8: - start_5 + start_8 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11 - 12 hold_5_12
   - 13 hold_5_13 - 14 hold_5_14 - 15 hold_5_15 - 16 hold_5_16
   - 17 hold_5_17 - 18 hold_5_18 + 60 switch_5_8 <= 60
 edge_disable_constr1_8_5: - start_5 + start_8 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 60 switch_8_5
   <= 60
 edge_disable_constr2_8_5: start_5 - start_8 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11 + 60 switch_8_5
   <= 60
 edge_disable_constr1_1_10: start_1 - start_10 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 60 switch_1_10 <= 60
 edge_disable_constr2_1_10: - start_1 + start_10 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   + 60 switch_1_10 <= 60
 edge_disable_constr1_10_1: - start_1 + start_10 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 12 hold_10_12 + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15
   + 16 hold_10_16 + 17 hold_10_17 + 18 hold_10_18 + 60 switch_10_1 <= 60
 edge_disable_constr2_10_1: start_1 - start_10 - hold_10_1 - 2 hold_10_2
   - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6 - 7 hold_10_7
   - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10 - 11 hold_10_11
   - 12 hold_10_12 - 13 hold_10_13 - 14 hold_10_14 - 15 hold_10_15
   - 16 hold_10_16 - 17 hold_10_17 - 18 hold_10_18 + 60 switch_10_1 <= 60
 edge_disable_constr1_5_10: start_5 - start_10 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12
   + 13 hold_5_13 + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16
   + 17 hold_5_17 + 18 hold_5_18 + 60 switch_5_10 <= 60
 edge_disable_constr2_5_10: - start_5 + start_10 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11 - 12 hold_5_12
   - 13 hold_5_13 - 14 hold_5_14 - 15 hold_5_15 - 16 hold_5_16
   - 17 hold_5_17 - 18 hold_5_18 + 60 switch_5_10 <= 60
 edge_disable_constr1_10_5: - start_5 + start_10 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 12 hold_10_12 + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15
   + 16 hold_10_16 + 17 hold_10_17 + 18 hold_10_18 + 60 switch_10_5 <= 60
 edge_disable_constr2_10_5: start_5 - start_10 - hold_10_1 - 2 hold_10_2
   - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6 - 7 hold_10_7
   - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10 - 11 hold_10_11
   - 12 hold_10_12 - 13 hold_10_13 - 14 hold_10_14 - 15 hold_10_15
   - 16 hold_10_16 - 17 hold_10_17 - 18 hold_10_18 + 60 switch_10_5 <= 60
 edge_disable_constr1_3_9: start_3 - start_9 + hold_3_1 + 2 hold_3_2
   + 3 hold_3_3 + 4 hold_3_4 + 5 hold_3_5 + 6 hold_3_6 + 7 hold_3_7
   + 60 switch_3_9 <= 60
 edge_disable_constr2_3_9: - start_3 + start_9 - hold_3_1 - 2 hold_3_2
   - 3 hold_3_3 - 4 hold_3_4 - 5 hold_3_5 - 6 hold_3_6 - 7 hold_3_7
   + 60 switch_3_9 <= 60
 edge_disable_constr1_9_3: - start_3 + start_9 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 60 switch_9_3 <= 60
 edge_disable_constr2_9_3: start_3 - start_9 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   + 60 switch_9_3 <= 60
 edge_disable_constr1_1_12: start_1 - start_12 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 60 switch_1_12 <= 60
 edge_disable_constr2_1_12: - start_1 + start_12 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   + 60 switch_1_12 <= 60
 edge_disable_constr1_12_1: - start_1 + start_12 + hold_12_1 + 2 hold_12_2
   + 3 hold_12_3 + 4 hold_12_4 + 5 hold_12_5 + 6 hold_12_6 + 7 hold_12_7
   + 60 switch_12_1 <= 60
 edge_disable_constr2_12_1: start_1 - start_12 - hold_12_1 - 2 hold_12_2
   - 3 hold_12_3 - 4 hold_12_4 - 5 hold_12_5 - 6 hold_12_6 - 7 hold_12_7
   + 60 switch_12_1 <= 60
 edge_disable_constr1_2_4: start_2 - start_4 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 10 hold_2_10 + 11 hold_2_11 + 60 switch_2_4
   <= 60
 edge_disable_constr2_2_4: - start_2 + start_4 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 - 10 hold_2_10 - 11 hold_2_11 + 60 switch_2_4
   <= 60
 edge_disable_constr1_4_2: - start_2 + start_4 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 60 switch_4_2 <= 60
 edge_disable_constr2_4_2: start_2 - start_4 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   + 60 switch_4_2 <= 60
 edge_disable_constr1_1_9: start_1 - start_9 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 60 switch_1_9 <= 60
 edge_disable_constr2_1_9: - start_1 + start_9 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   + 60 switch_1_9 <= 60
 edge_disable_constr1_9_1: - start_1 + start_9 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 60 switch_9_1 <= 60
 edge_disable_constr2_9_1: start_1 - start_9 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   + 60 switch_9_1 <= 60
 edge_disable_constr1_5_14: start_5 - start_14 + hold_5_1 + 2 hold_5_2
   + 3 hold_5_3 + 4 hold_5_4 + 5 hold_5_5 + 6 hold_5_6 + 7 hold_5_7
   + 8 hold_5_8 + 9 hold_5_9 + 10 hold_5_10 + 11 hold_5_11 + 12 hold_5_12
   + 13 hold_5_13 + 14 hold_5_14 + 15 hold_5_15 + 16 hold_5_16
   + 17 hold_5_17 + 18 hold_5_18 + 60 switch_5_14 <= 60
 edge_disable_constr2_5_14: - start_5 + start_14 - hold_5_1 - 2 hold_5_2
   - 3 hold_5_3 - 4 hold_5_4 - 5 hold_5_5 - 6 hold_5_6 - 7 hold_5_7
   - 8 hold_5_8 - 9 hold_5_9 - 10 hold_5_10 - 11 hold_5_11 - 12 hold_5_12
   - 13 hold_5_13 - 14 hold_5_14 - 15 hold_5_15 - 16 hold_5_16
   - 17 hold_5_17 - 18 hold_5_18 + 60 switch_5_14 <= 60
 edge_disable_constr1_14_5: - start_5 + start_14 + hold_14_1 + 2 hold_14_2
   + 3 hold_14_3 + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6 + 7 hold_14_7
   + 8 hold_14_8 + 9 hold_14_9 + 10 hold_14_10 + 11 hold_14_11
   + 60 switch_14_5 <= 60
 edge_disable_constr2_14_5: start_5 - start_14 - hold_14_1 - 2 hold_14_2
   - 3 hold_14_3 - 4 hold_14_4 - 5 hold_14_5 - 6 hold_14_6 - 7 hold_14_7
   - 8 hold_14_8 - 9 hold_14_9 - 10 hold_14_10 - 11 hold_14_11
   + 60 switch_14_5 <= 60
 edge_disable_constr1_4_10: start_4 - start_10 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 60 switch_4_10 <= 60
 edge_disable_constr2_4_10: - start_4 + start_10 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   + 60 switch_4_10 <= 60
 edge_disable_constr1_10_4: - start_4 + start_10 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 12 hold_10_12 + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15
   + 16 hold_10_16 + 17 hold_10_17 + 18 hold_10_18 + 60 switch_10_4 <= 60
 edge_disable_constr2_10_4: start_4 - start_10 - hold_10_1 - 2 hold_10_2
   - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6 - 7 hold_10_7
   - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10 - 11 hold_10_11
   - 12 hold_10_12 - 13 hold_10_13 - 14 hold_10_14 - 15 hold_10_15
   - 16 hold_10_16 - 17 hold_10_17 - 18 hold_10_18 + 60 switch_10_4 <= 60
 edge_disable_constr1_11_13: start_11 - start_13 + hold_11_1 + 2 hold_11_2
   + 3 hold_11_3 + 4 hold_11_4 + 5 hold_11_5 + 6 hold_11_6 + 7 hold_11_7
   + 8 hold_11_8 + 9 hold_11_9 + 10 hold_11_10 + 11 hold_11_11
   + 60 switch_11_13 <= 60
 edge_disable_constr2_11_13: - start_11 + start_13 - hold_11_1
   - 2 hold_11_2 - 3 hold_11_3 - 4 hold_11_4 - 5 hold_11_5 - 6 hold_11_6
   - 7 hold_11_7 - 8 hold_11_8 - 9 hold_11_9 - 10 hold_11_10
   - 11 hold_11_11 + 60 switch_11_13 <= 60
 edge_disable_constr1_13_11: - start_11 + start_13 + hold_13_1
   + 2 hold_13_2 + 3 hold_13_3 + 4 hold_13_4 + 5 hold_13_5 + 6 hold_13_6
   + 7 hold_13_7 + 8 hold_13_8 + 9 hold_13_9 + 10 hold_13_10
   + 11 hold_13_11 + 60 switch_13_11 <= 60
 edge_disable_constr2_13_11: start_11 - start_13 - hold_13_1 - 2 hold_13_2
   - 3 hold_13_3 - 4 hold_13_4 - 5 hold_13_5 - 6 hold_13_6 - 7 hold_13_7
   - 8 hold_13_8 - 9 hold_13_9 - 10 hold_13_10 - 11 hold_13_11
   + 60 switch_13_11 <= 60
 edge_disable_constr1_0_4: start_0 - start_4 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 60 switch_0_4 <= 60
 edge_disable_constr2_0_4: - start_0 + start_4 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   + 60 switch_0_4 <= 60
 edge_disable_constr1_4_0: - start_0 + start_4 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 60 switch_4_0 <= 60
 edge_disable_constr2_4_0: start_0 - start_4 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   + 60 switch_4_0 <= 60
 edge_disable_constr1_9_10: start_9 - start_10 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 60 switch_9_10 <= 60
 edge_disable_constr2_9_10: - start_9 + start_10 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   + 60 switch_9_10 <= 60
 edge_disable_constr1_10_9: - start_9 + start_10 + hold_10_1 + 2 hold_10_2
   + 3 hold_10_3 + 4 hold_10_4 + 5 hold_10_5 + 6 hold_10_6 + 7 hold_10_7
   + 8 hold_10_8 + 9 hold_10_9 + 10 hold_10_10 + 11 hold_10_11
   + 12 hold_10_12 + 13 hold_10_13 + 14 hold_10_14 + 15 hold_10_15
   + 16 hold_10_16 + 17 hold_10_17 + 18 hold_10_18 + 60 switch_10_9 <= 60
 edge_disable_constr2_10_9: start_9 - start_10 - hold_10_1 - 2 hold_10_2
   - 3 hold_10_3 - 4 hold_10_4 - 5 hold_10_5 - 6 hold_10_6 - 7 hold_10_7
   - 8 hold_10_8 - 9 hold_10_9 - 10 hold_10_10 - 11 hold_10_11
   - 12 hold_10_12 - 13 hold_10_13 - 14 hold_10_14 - 15 hold_10_15
   - 16 hold_10_16 - 17 hold_10_17 - 18 hold_10_18 + 60 switch_10_9 <= 60
 edge_disable_constr1_9_14: start_9 - start_14 + hold_9_1 + 2 hold_9_2
   + 3 hold_9_3 + 4 hold_9_4 + 5 hold_9_5 + 6 hold_9_6 + 7 hold_9_7
   + 60 switch_9_14 <= 60
 edge_disable_constr2_9_14: - start_9 + start_14 - hold_9_1 - 2 hold_9_2
   - 3 hold_9_3 - 4 hold_9_4 - 5 hold_9_5 - 6 hold_9_6 - 7 hold_9_7
   + 60 switch_9_14 <= 60
 edge_disable_constr1_14_9: - start_9 + start_14 + hold_14_1 + 2 hold_14_2
   + 3 hold_14_3 + 4 hold_14_4 + 5 hold_14_5 + 6 hold_14_6 + 7 hold_14_7
   + 8 hold_14_8 + 9 hold_14_9 + 10 hold_14_10 + 11 hold_14_11
   + 60 switch_14_9 <= 60
 edge_disable_constr2_14_9: start_9 - start_14 - hold_14_1 - 2 hold_14_2
   - 3 hold_14_3 - 4 hold_14_4 - 5 hold_14_5 - 6 hold_14_6 - 7 hold_14_7
   - 8 hold_14_8 - 9 hold_14_9 - 10 hold_14_10 - 11 hold_14_11
   + 60 switch_14_9 <= 60
 edge_disable_constr1_4_8: start_4 - start_8 + hold_4_1 + 2 hold_4_2
   + 3 hold_4_3 + 4 hold_4_4 + 5 hold_4_5 + 6 hold_4_6 + 7 hold_4_7
   + 60 switch_4_8 <= 60
 edge_disable_constr2_4_8: - start_4 + start_8 - hold_4_1 - 2 hold_4_2
   - 3 hold_4_3 - 4 hold_4_4 - 5 hold_4_5 - 6 hold_4_6 - 7 hold_4_7
   + 60 switch_4_8 <= 60
 edge_disable_constr1_8_4: - start_4 + start_8 + hold_8_1 + 2 hold_8_2
   + 3 hold_8_3 + 4 hold_8_4 + 5 hold_8_5 + 6 hold_8_6 + 7 hold_8_7
   + 8 hold_8_8 + 9 hold_8_9 + 10 hold_8_10 + 11 hold_8_11 + 60 switch_8_4
   <= 60
 edge_disable_constr2_8_4: start_4 - start_8 - hold_8_1 - 2 hold_8_2
   - 3 hold_8_3 - 4 hold_8_4 - 5 hold_8_5 - 6 hold_8_6 - 7 hold_8_7
   - 8 hold_8_8 - 9 hold_8_9 - 10 hold_8_10 - 11 hold_8_11 + 60 switch_8_4
   <= 60
 edge_flow_constr_source_0: flow_source_0 - 5e+08 switch_source_0 <= 0
 edge_flow_constr_source_1: flow_source_1 - 5e+08 switch_source_1 <= 0
 edge_flow_constr_source_2: flow_source_2 - 5e+08 switch_source_2 <= 0
 edge_flow_constr_source_3: flow_source_3 - 5e+08 switch_source_3 <= 0
 edge_flow_constr_source_4: flow_source_4 - 5e+08 switch_source_4 <= 0
 edge_flow_constr_source_5: flow_source_5 - 5e+08 switch_source_5 <= 0
 edge_flow_constr_source_6: flow_source_6 - 5e+08 switch_source_6 <= 0
 edge_flow_constr_source_7: flow_source_7 - 5e+08 switch_source_7 <= 0
 edge_flow_constr_source_8: flow_source_8 - 5e+08 switch_source_8 <= 0
 edge_flow_constr_source_9: flow_source_9 - 5e+08 switch_source_9 <= 0
 edge_flow_constr_source_10: flow_source_10 - 5e+08 switch_source_10 <= 0
 edge_flow_constr_source_11: flow_source_11 - 5e+08 switch_source_11 <= 0
 edge_flow_constr_source_12: flow_source_12 - 5e+08 switch_source_12 <= 0
 edge_flow_constr_source_13: flow_source_13 - 5e+08 switch_source_13 <= 0
 edge_flow_constr_source_14: flow_source_14 - 5e+08 switch_source_14 <= 0
 edge_flow_constr_source_15: flow_source_15 - 5e+08 switch_source_15 <= 0
 edge_flow_constr_0_sink: flow_0_sink - 5e+08 switch_0_sink <= 0
 edge_flow_constr_1_sink: flow_1_sink - 5e+08 switch_1_sink <= 0
 edge_flow_constr_2_sink: flow_2_sink - 5e+08 switch_2_sink <= 0
 edge_flow_constr_3_sink: flow_3_sink - 5e+08 switch_3_sink <= 0
 edge_flow_constr_4_sink: flow_4_sink - 5e+08 switch_4_sink <= 0
 edge_flow_constr_5_sink: flow_5_sink - 5e+08 switch_5_sink <= 0
 edge_flow_constr_6_sink: flow_6_sink - 5e+08 switch_6_sink <= 0
 edge_flow_constr_7_sink: flow_7_sink - 5e+08 switch_7_sink <= 0
 edge_flow_constr_8_sink: flow_8_sink - 5e+08 switch_8_sink <= 0
 edge_flow_constr_9_sink: flow_9_sink - 5e+08 switch_9_sink <= 0
 edge_flow_constr_10_sink: flow_10_sink - 5e+08 switch_10_sink <= 0
 edge_flow_constr_11_sink: flow_11_sink - 5e+08 switch_11_sink <= 0
 edge_flow_constr_12_sink: flow_12_sink - 5e+08 switch_12_sink <= 0
 edge_flow_constr_13_sink: flow_13_sink - 5e+08 switch_13_sink <= 0
 edge_flow_constr_14_sink: flow_14_sink - 5e+08 switch_14_sink <= 0
 edge_flow_constr_15_sink: flow_15_sink - 5e+08 switch_15_sink <= 0
 edge_flow_constr_10_14: flow_10_14 - 75120.19230769231 switch_10_14 <= 0
 edge_flow_constr_14_10: flow_14_10 - 75120.19230769231 switch_14_10 <= 0
 edge_flow_constr_3_11: flow_3_11 - 75120.19230769231 switch_3_11 <= 0
 edge_flow_constr_11_3: flow_11_3 - 75120.19230769231 switch_11_3 <= 0
 edge_flow_constr_8_14: flow_8_14 - 75120.19230769231 switch_8_14 <= 0
 edge_flow_constr_14_8: flow_14_8 - 75120.19230769231 switch_14_8 <= 0
 edge_flow_constr_11_15: flow_11_15 - 75120.19230769231 switch_11_15 <= 0
 edge_flow_constr_15_11: flow_15_11 - 75120.19230769231 switch_15_11 <= 0
 edge_flow_constr_3_15: flow_3_15 - 75120.19230769231 switch_3_15 <= 0
 edge_flow_constr_15_3: flow_15_3 - 75120.19230769231 switch_15_3 <= 0
 edge_flow_constr_0_6: flow_0_6 - 75120.19230769231 switch_0_6 <= 0
 edge_flow_constr_6_0: flow_6_0 - 75120.19230769231 switch_6_0 <= 0
 edge_flow_constr_2_7: flow_2_7 - 75120.19230769231 switch_2_7 <= 0
 edge_flow_constr_7_2: flow_7_2 - 75120.19230769231 switch_7_2 <= 0
 edge_flow_constr_5_6: flow_5_6 - 75120.19230769231 switch_5_6 <= 0
 edge_flow_constr_6_5: flow_6_5 - 75120.19230769231 switch_6_5 <= 0
 edge_flow_constr_4_13: flow_4_13 - 75120.19230769231 switch_4_13 <= 0
 edge_flow_constr_13_4: flow_13_4 - 75120.19230769231 switch_13_4 <= 0
 edge_flow_constr_3_13: flow_3_13 - 75120.19230769231 switch_3_13 <= 0
 edge_flow_constr_13_3: flow_13_3 - 75120.19230769231 switch_13_3 <= 0
 edge_flow_constr_9_11: flow_9_11 - 75120.19230769231 switch_9_11 <= 0
 edge_flow_constr_11_9: flow_11_9 - 75120.19230769231 switch_11_9 <= 0
 edge_flow_constr_10_12: flow_10_12 - 75120.19230769231 switch_10_12 <= 0
 edge_flow_constr_12_10: flow_12_10 - 75120.19230769231 switch_12_10 <= 0
 edge_flow_constr_2_12: flow_2_12 - 75120.19230769231 switch_2_12 <= 0
 edge_flow_constr_12_2: flow_12_2 - 75120.19230769231 switch_12_2 <= 0
 edge_flow_constr_4_12: flow_4_12 - 75120.19230769231 switch_4_12 <= 0
 edge_flow_constr_12_4: flow_12_4 - 75120.19230769231 switch_12_4 <= 0
 edge_flow_constr_3_7: flow_3_7 - 75120.19230769231 switch_3_7 <= 0
 edge_flow_constr_7_3: flow_7_3 - 75120.19230769231 switch_7_3 <= 0
 edge_flow_constr_5_15: flow_5_15 - 75120.19230769231 switch_5_15 <= 0
 edge_flow_constr_15_5: flow_15_5 - 75120.19230769231 switch_15_5 <= 0
 edge_flow_constr_1_14: flow_1_14 - 75120.19230769231 switch_1_14 <= 0
 edge_flow_constr_14_1: flow_14_1 - 75120.19230769231 switch_14_1 <= 0
 edge_flow_constr_4_15: flow_4_15 - 75120.19230769231 switch_4_15 <= 0
 edge_flow_constr_15_4: flow_15_4 - 75120.19230769231 switch_15_4 <= 0
 edge_flow_constr_1_4: flow_1_4 - 75120.19230769231 switch_1_4 <= 0
 edge_flow_constr_4_1: flow_4_1 - 75120.19230769231 switch_4_1 <= 0
 edge_flow_constr_2_10: flow_2_10 - 75120.19230769231 switch_2_10 <= 0
 edge_flow_constr_10_2: flow_10_2 - 75120.19230769231 switch_10_2 <= 0
 edge_flow_constr_9_12: flow_9_12 - 75120.19230769231 switch_9_12 <= 0
 edge_flow_constr_12_9: flow_12_9 - 75120.19230769231 switch_12_9 <= 0
 edge_flow_constr_8_13: flow_8_13 - 75120.19230769231 switch_8_13 <= 0
 edge_flow_constr_13_8: flow_13_8 - 75120.19230769231 switch_13_8 <= 0
 edge_flow_constr_0_13: flow_0_13 - 75120.19230769231 switch_0_13 <= 0
 edge_flow_constr_13_0: flow_13_0 - 75120.19230769231 switch_13_0 <= 0
 edge_flow_constr_6_11: flow_6_11 - 75120.19230769231 switch_6_11 <= 0
 edge_flow_constr_11_6: flow_11_6 - 75120.19230769231 switch_11_6 <= 0
 edge_flow_constr_2_6: flow_2_6 - 75120.19230769231 switch_2_6 <= 0
 edge_flow_constr_6_2: flow_6_2 - 75120.19230769231 switch_6_2 <= 0
 edge_flow_constr_5_9: flow_5_9 - 75120.19230769231 switch_5_9 <= 0
 edge_flow_constr_9_5: flow_9_5 - 75120.19230769231 switch_9_5 <= 0
 edge_flow_constr_7_14: flow_7_14 - 75120.19230769231 switch_7_14 <= 0
 edge_flow_constr_14_7: flow_14_7 - 75120.19230769231 switch_14_7 <= 0
 edge_flow_constr_6_9: flow_6_9 - 75120.19230769231 switch_6_9 <= 0
 edge_flow_constr_9_6: flow_9_6 - 75120.19230769231 switch_9_6 <= 0
 edge_flow_constr_1_5: flow_1_5 - 75120.19230769231 switch_1_5 <= 0
 edge_flow_constr_5_1: flow_5_1 - 75120.19230769231 switch_5_1 <= 0
 edge_flow_constr_2_13: flow_2_13 - 75120.19230769231 switch_2_13 <= 0
 edge_flow_constr_13_2: flow_13_2 - 75120.19230769231 switch_13_2 <= 0
 edge_flow_constr_13_14: flow_13_14 - 75120.19230769231 switch_13_14 <= 0
 edge_flow_constr_14_13: flow_14_13 - 75120.19230769231 switch_14_13 <= 0
 edge_flow_constr_0_10: flow_0_10 - 75120.19230769231 switch_0_10 <= 0
 edge_flow_constr_10_0: flow_10_0 - 75120.19230769231 switch_10_0 <= 0
 edge_flow_constr_7_11: flow_7_11 - 75120.19230769231 switch_7_11 <= 0
 edge_flow_constr_11_7: flow_11_7 - 75120.19230769231 switch_11_7 <= 0
 edge_flow_constr_3_4: flow_3_4 - 75120.19230769231 switch_3_4 <= 0
 edge_flow_constr_4_3: flow_4_3 - 75120.19230769231 switch_4_3 <= 0
 edge_flow_constr_4_11: flow_4_11 - 75120.19230769231 switch_4_11 <= 0
 edge_flow_constr_11_4: flow_11_4 - 75120.19230769231 switch_11_4 <= 0
 edge_flow_constr_5_12: flow_5_12 - 75120.19230769231 switch_5_12 <= 0
 edge_flow_constr_12_5: flow_12_5 - 75120.19230769231 switch_12_5 <= 0
 edge_flow_constr_7_13: flow_7_13 - 75120.19230769231 switch_7_13 <= 0
 edge_flow_constr_13_7: flow_13_7 - 75120.19230769231 switch_13_7 <= 0
 edge_flow_constr_10_11: flow_10_11 - 75120.19230769231 switch_10_11 <= 0
 edge_flow_constr_11_10: flow_11_10 - 75120.19230769231 switch_11_10 <= 0
 edge_flow_constr_4_6: flow_4_6 - 75120.19230769231 switch_4_6 <= 0
 edge_flow_constr_6_4: flow_6_4 - 75120.19230769231 switch_6_4 <= 0
 edge_flow_constr_2_14: flow_2_14 - 75120.19230769231 switch_2_14 <= 0
 edge_flow_constr_14_2: flow_14_2 - 75120.19230769231 switch_14_2 <= 0
 edge_flow_constr_6_10: flow_6_10 - 75120.19230769231 switch_6_10 <= 0
 edge_flow_constr_10_6: flow_10_6 - 75120.19230769231 switch_10_6 <= 0
 edge_flow_constr_1_13: flow_1_13 - 75120.19230769231 switch_1_13 <= 0
 edge_flow_constr_13_1: flow_13_1 - 75120.19230769231 switch_13_1 <= 0
 edge_flow_constr_5_11: flow_5_11 - 75120.19230769231 switch_5_11 <= 0
 edge_flow_constr_11_5: flow_11_5 - 75120.19230769231 switch_11_5 <= 0
 edge_flow_constr_14_15: flow_14_15 - 75120.19230769231 switch_14_15 <= 0
 edge_flow_constr_15_14: flow_15_14 - 75120.19230769231 switch_15_14 <= 0
 edge_flow_constr_4_7: flow_4_7 - 75120.19230769231 switch_4_7 <= 0
 edge_flow_constr_7_4: flow_7_4 - 75120.19230769231 switch_7_4 <= 0
 edge_flow_constr_5_7: flow_5_7 - 75120.19230769231 switch_5_7 <= 0
 edge_flow_constr_7_5: flow_7_5 - 75120.19230769231 switch_7_5 <= 0
 edge_flow_constr_12_15: flow_12_15 - 75120.19230769231 switch_12_15 <= 0
 edge_flow_constr_15_12: flow_15_12 - 75120.19230769231 switch_15_12 <= 0
 edge_flow_constr_0_8: flow_0_8 - 75120.19230769231 switch_0_8 <= 0
 edge_flow_constr_8_0: flow_8_0 - 75120.19230769231 switch_8_0 <= 0
 edge_flow_constr_8_9: flow_8_9 - 75120.19230769231 switch_8_9 <= 0
 edge_flow_constr_9_8: flow_9_8 - 75120.19230769231 switch_9_8 <= 0
 edge_flow_constr_0_2: flow_0_2 - 75120.19230769231 switch_0_2 <= 0
 edge_flow_constr_2_0: flow_2_0 - 75120.19230769231 switch_2_0 <= 0
 edge_flow_constr_0_12: flow_0_12 - 75120.19230769231 switch_0_12 <= 0
 edge_flow_constr_12_0: flow_12_0 - 75120.19230769231 switch_12_0 <= 0
 edge_flow_constr_11_12: flow_11_12 - 75120.19230769231 switch_11_12 <= 0
 edge_flow_constr_12_11: flow_12_11 - 75120.19230769231 switch_12_11 <= 0
 edge_flow_constr_0_1: flow_0_1 - 75120.19230769231 switch_0_1 <= 0
 edge_flow_constr_1_0: flow_1_0 - 75120.19230769231 switch_1_0 <= 0
 edge_flow_constr_4_14: flow_4_14 - 75120.19230769231 switch_4_14 <= 0
 edge_flow_constr_14_4: flow_14_4 - 75120.19230769231 switch_14_4 <= 0
 edge_flow_constr_7_10: flow_7_10 - 75120.19230769231 switch_7_10 <= 0
 edge_flow_constr_10_7: flow_10_7 - 75120.19230769231 switch_10_7 <= 0
 edge_flow_constr_2_5: flow_2_5 - 75120.19230769231 switch_2_5 <= 0
 edge_flow_constr_5_2: flow_5_2 - 75120.19230769231 switch_5_2 <= 0
 edge_flow_constr_2_15: flow_2_15 - 75120.19230769231 switch_2_15 <= 0
 edge_flow_constr_15_2: flow_15_2 - 75120.19230769231 switch_15_2 <= 0
 edge_flow_constr_6_14: flow_6_14 - 75120.19230769231 switch_6_14 <= 0
 edge_flow_constr_14_6: flow_14_6 - 75120.19230769231 switch_14_6 <= 0
 edge_flow_constr_6_13: flow_6_13 - 75120.19230769231 switch_6_13 <= 0
 edge_flow_constr_13_6: flow_13_6 - 75120.19230769231 switch_13_6 <= 0
 edge_flow_constr_0_5: flow_0_5 - 75120.19230769231 switch_0_5 <= 0
 edge_flow_constr_5_0: flow_5_0 - 75120.19230769231 switch_5_0 <= 0
 edge_flow_constr_6_15: flow_6_15 - 75120.19230769231 switch_6_15 <= 0
 edge_flow_constr_15_6: flow_15_6 - 75120.19230769231 switch_15_6 <= 0
 edge_flow_constr_4_9: flow_4_9 - 75120.19230769231 switch_4_9 <= 0
 edge_flow_constr_9_4: flow_9_4 - 75120.19230769231 switch_9_4 <= 0
 edge_flow_constr_10_15: flow_10_15 - 75120.19230769231 switch_10_15 <= 0
 edge_flow_constr_15_10: flow_15_10 - 75120.19230769231 switch_15_10 <= 0
 edge_flow_constr_0_15: flow_0_15 - 75120.19230769231 switch_0_15 <= 0
 edge_flow_constr_15_0: flow_15_0 - 75120.19230769231 switch_15_0 <= 0
 edge_flow_constr_1_2: flow_1_2 - 75120.19230769231 switch_1_2 <= 0
 edge_flow_constr_2_1: flow_2_1 - 75120.19230769231 switch_2_1 <= 0
 edge_flow_constr_8_11: flow_8_11 - 75120.19230769231 switch_8_11 <= 0
 edge_flow_constr_11_8: flow_11_8 - 75120.19230769231 switch_11_8 <= 0
 edge_flow_constr_7_12: flow_7_12 - 75120.19230769231 switch_7_12 <= 0
 edge_flow_constr_12_7: flow_12_7 - 75120.19230769231 switch_12_7 <= 0
 edge_flow_constr_4_5: flow_4_5 - 75120.19230769231 switch_4_5 <= 0
 edge_flow_constr_5_4: flow_5_4 - 75120.19230769231 switch_5_4 <= 0
 edge_flow_constr_2_8: flow_2_8 - 75120.19230769231 switch_2_8 <= 0
 edge_flow_constr_8_2: flow_8_2 - 75120.19230769231 switch_8_2 <= 0
 edge_flow_constr_1_15: flow_1_15 - 75120.19230769231 switch_1_15 <= 0
 edge_flow_constr_15_1: flow_15_1 - 75120.19230769231 switch_15_1 <= 0
 edge_flow_constr_12_13: flow_12_13 - 75120.19230769231 switch_12_13 <= 0
 edge_flow_constr_13_12: flow_13_12 - 75120.19230769231 switch_13_12 <= 0
 edge_flow_constr_13_15: flow_13_15 - 75120.19230769231 switch_13_15 <= 0
 edge_flow_constr_15_13: flow_15_13 - 75120.19230769231 switch_15_13 <= 0
 edge_flow_constr_1_7: flow_1_7 - 75120.19230769231 switch_1_7 <= 0
 edge_flow_constr_7_1: flow_7_1 - 75120.19230769231 switch_7_1 <= 0
 edge_flow_constr_8_12: flow_8_12 - 75120.19230769231 switch_8_12 <= 0
 edge_flow_constr_12_8: flow_12_8 - 75120.19230769231 switch_12_8 <= 0
 edge_flow_constr_9_13: flow_9_13 - 75120.19230769231 switch_9_13 <= 0
 edge_flow_constr_13_9: flow_13_9 - 75120.19230769231 switch_13_9 <= 0
 edge_flow_constr_0_7: flow_0_7 - 75120.19230769231 switch_0_7 <= 0
 edge_flow_constr_7_0: flow_7_0 - 75120.19230769231 switch_7_0 <= 0
 edge_flow_constr_1_6: flow_1_6 - 75120.19230769231 switch_1_6 <= 0
 edge_flow_constr_6_1: flow_6_1 - 75120.19230769231 switch_6_1 <= 0
 edge_flow_constr_2_9: flow_2_9 - 75120.19230769231 switch_2_9 <= 0
 edge_flow_constr_9_2: flow_9_2 - 75120.19230769231 switch_9_2 <= 0
 edge_flow_constr_7_15: flow_7_15 - 75120.19230769231 switch_7_15 <= 0
 edge_flow_constr_15_7: flow_15_7 - 75120.19230769231 switch_15_7 <= 0
 edge_flow_constr_5_13: flow_5_13 - 75120.19230769231 switch_5_13 <= 0
 edge_flow_constr_13_5: flow_13_5 - 75120.19230769231 switch_13_5 <= 0
 edge_flow_constr_11_14: flow_11_14 - 75120.19230769231 switch_11_14 <= 0
 edge_flow_constr_14_11: flow_14_11 - 75120.19230769231 switch_14_11 <= 0
 edge_flow_constr_1_8: flow_1_8 - 75120.19230769231 switch_1_8 <= 0
 edge_flow_constr_8_1: flow_8_1 - 75120.19230769231 switch_8_1 <= 0
 edge_flow_constr_3_14: flow_3_14 - 75120.19230769231 switch_3_14 <= 0
 edge_flow_constr_14_3: flow_14_3 - 75120.19230769231 switch_14_3 <= 0
 edge_flow_constr_0_14: flow_0_14 - 75120.19230769231 switch_0_14 <= 0
 edge_flow_constr_14_0: flow_14_0 - 75120.19230769231 switch_14_0 <= 0
 edge_flow_constr_8_10: flow_8_10 - 75120.19230769231 switch_8_10 <= 0
 edge_flow_constr_10_8: flow_10_8 - 75120.19230769231 switch_10_8 <= 0
 edge_flow_constr_9_15: flow_9_15 - 75120.19230769231 switch_9_15 <= 0
 edge_flow_constr_15_9: flow_15_9 - 75120.19230769231 switch_15_9 <= 0
 edge_flow_constr_3_12: flow_3_12 - 75120.19230769231 switch_3_12 <= 0
 edge_flow_constr_12_3: flow_12_3 - 75120.19230769231 switch_12_3 <= 0
 edge_flow_constr_3_10: flow_3_10 - 75120.19230769231 switch_3_10 <= 0
 edge_flow_constr_10_3: flow_10_3 - 75120.19230769231 switch_10_3 <= 0
 edge_flow_constr_1_11: flow_1_11 - 75120.19230769231 switch_1_11 <= 0
 edge_flow_constr_11_1: flow_11_1 - 75120.19230769231 switch_11_1 <= 0
 edge_flow_constr_0_11: flow_0_11 - 75120.19230769231 switch_0_11 <= 0
 edge_flow_constr_11_0: flow_11_0 - 75120.19230769231 switch_11_0 <= 0
 edge_flow_constr_2_11: flow_2_11 - 75120.19230769231 switch_2_11 <= 0
 edge_flow_constr_11_2: flow_11_2 - 75120.19230769231 switch_11_2 <= 0
 edge_flow_constr_6_12: flow_6_12 - 75120.19230769231 switch_6_12 <= 0
 edge_flow_constr_12_6: flow_12_6 - 75120.19230769231 switch_12_6 <= 0
 edge_flow_constr_12_14: flow_12_14 - 75120.19230769231 switch_12_14 <= 0
 edge_flow_constr_14_12: flow_14_12 - 75120.19230769231 switch_14_12 <= 0
 edge_flow_constr_3_8: flow_3_8 - 75120.19230769231 switch_3_8 <= 0
 edge_flow_constr_8_3: flow_8_3 - 75120.19230769231 switch_8_3 <= 0
 edge_flow_constr_3_6: flow_3_6 - 75120.19230769231 switch_3_6 <= 0
 edge_flow_constr_6_3: flow_6_3 - 75120.19230769231 switch_6_3 <= 0
 edge_flow_constr_8_15: flow_8_15 - 75120.19230769231 switch_8_15 <= 0
 edge_flow_constr_15_8: flow_15_8 - 75120.19230769231 switch_15_8 <= 0
 edge_flow_constr_7_8: flow_7_8 - 75120.19230769231 switch_7_8 <= 0
 edge_flow_constr_8_7: flow_8_7 - 75120.19230769231 switch_8_7 <= 0
 edge_flow_constr_3_5: flow_3_5 - 75120.19230769231 switch_3_5 <= 0
 edge_flow_constr_5_3: flow_5_3 - 75120.19230769231 switch_5_3 <= 0
 edge_flow_constr_10_13: flow_10_13 - 75120.19230769231 switch_10_13 <= 0
 edge_flow_constr_13_10: flow_13_10 - 75120.19230769231 switch_13_10 <= 0
 edge_flow_constr_0_9: flow_0_9 - 75120.19230769231 switch_0_9 <= 0
 edge_flow_constr_9_0: flow_9_0 - 75120.19230769231 switch_9_0 <= 0
 edge_flow_constr_6_8: flow_6_8 - 75120.19230769231 switch_6_8 <= 0
 edge_flow_constr_8_6: flow_8_6 - 75120.19230769231 switch_8_6 <= 0
 edge_flow_constr_0_3: flow_0_3 - 75120.19230769231 switch_0_3 <= 0
 edge_flow_constr_3_0: flow_3_0 - 75120.19230769231 switch_3_0 <= 0
 edge_flow_constr_2_3: flow_2_3 - 75120.19230769231 switch_2_3 <= 0
 edge_flow_constr_3_2: flow_3_2 - 75120.19230769231 switch_3_2 <= 0
 edge_flow_constr_7_9: flow_7_9 - 75120.19230769231 switch_7_9 <= 0
 edge_flow_constr_9_7: flow_9_7 - 75120.19230769231 switch_9_7 <= 0
 edge_flow_constr_1_3: flow_1_3 - 75120.19230769231 switch_1_3 <= 0
 edge_flow_constr_3_1: flow_3_1 - 75120.19230769231 switch_3_1 <= 0
 edge_flow_constr_6_7: flow_6_7 - 75120.19230769231 switch_6_7 <= 0
 edge_flow_constr_7_6: flow_7_6 - 75120.19230769231 switch_7_6 <= 0
 edge_flow_constr_5_8: flow_5_8 - 75120.19230769231 switch_5_8 <= 0
 edge_flow_constr_8_5: flow_8_5 - 75120.19230769231 switch_8_5 <= 0
 edge_flow_constr_1_10: flow_1_10 - 75120.19230769231 switch_1_10 <= 0
 edge_flow_constr_10_1: flow_10_1 - 75120.19230769231 switch_10_1 <= 0
 edge_flow_constr_5_10: flow_5_10 - 75120.19230769231 switch_5_10 <= 0
 edge_flow_constr_10_5: flow_10_5 - 75120.19230769231 switch_10_5 <= 0
 edge_flow_constr_3_9: flow_3_9 - 75120.19230769231 switch_3_9 <= 0
 edge_flow_constr_9_3: flow_9_3 - 75120.19230769231 switch_9_3 <= 0
 edge_flow_constr_1_12: flow_1_12 - 75120.19230769231 switch_1_12 <= 0
 edge_flow_constr_12_1: flow_12_1 - 75120.19230769231 switch_12_1 <= 0
 edge_flow_constr_2_4: flow_2_4 - 75120.19230769231 switch_2_4 <= 0
 edge_flow_constr_4_2: flow_4_2 - 75120.19230769231 switch_4_2 <= 0
 edge_flow_constr_1_9: flow_1_9 - 75120.19230769231 switch_1_9 <= 0
 edge_flow_constr_9_1: flow_9_1 - 75120.19230769231 switch_9_1 <= 0
 edge_flow_constr_5_14: flow_5_14 - 75120.19230769231 switch_5_14 <= 0
 edge_flow_constr_14_5: flow_14_5 - 75120.19230769231 switch_14_5 <= 0
 edge_flow_constr_4_10: flow_4_10 - 75120.19230769231 switch_4_10 <= 0
 edge_flow_constr_10_4: flow_10_4 - 75120.19230769231 switch_10_4 <= 0
 edge_flow_constr_11_13: flow_11_13 - 75120.19230769231 switch_11_13 <= 0
 edge_flow_constr_13_11: flow_13_11 - 75120.19230769231 switch_13_11 <= 0
 edge_flow_constr_0_4: flow_0_4 - 75120.19230769231 switch_0_4 <= 0
 edge_flow_constr_4_0: flow_4_0 - 75120.19230769231 switch_4_0 <= 0
 edge_flow_constr_9_10: flow_9_10 - 75120.19230769231 switch_9_10 <= 0
 edge_flow_constr_10_9: flow_10_9 - 75120.19230769231 switch_10_9 <= 0
 edge_flow_constr_9_14: flow_9_14 - 75120.19230769231 switch_9_14 <= 0
 edge_flow_constr_14_9: flow_14_9 - 75120.19230769231 switch_14_9 <= 0
 edge_flow_constr_4_8: flow_4_8 - 75120.19230769231 switch_4_8 <= 0
 edge_flow_constr_8_4: flow_8_4 - 75120.19230769231 switch_8_4 <= 0
Bounds
Binaries
 hold_0_1 hold_0_2 hold_0_3 hold_0_4 hold_0_5 hold_0_6 hold_0_7 hold_1_1
 hold_1_2 hold_1_3 hold_1_4 hold_1_5 hold_1_6 hold_1_7 hold_2_1 hold_2_2
 hold_2_3 hold_2_4 hold_2_5 hold_2_6 hold_2_7 hold_2_8 hold_2_9 hold_2_10
 hold_2_11 hold_3_1 hold_3_2 hold_3_3 hold_3_4 hold_3_5 hold_3_6 hold_3_7
 hold_4_1 hold_4_2 hold_4_3 hold_4_4 hold_4_5 hold_4_6 hold_4_7 hold_5_1
 hold_5_2 hold_5_3 hold_5_4 hold_5_5 hold_5_6 hold_5_7 hold_5_8 hold_5_9
 hold_5_10 hold_5_11 hold_5_12 hold_5_13 hold_5_14 hold_5_15 hold_5_16
 hold_5_17 hold_5_18 hold_6_1 hold_6_2 hold_6_3 hold_6_4 hold_6_5 hold_6_6
 hold_6_7 hold_6_8 hold_6_9 hold_6_10 hold_6_11 hold_7_1 hold_7_2 hold_7_3
 hold_7_4 hold_7_5 hold_7_6 hold_7_7 hold_8_1 hold_8_2 hold_8_3 hold_8_4
 hold_8_5 hold_8_6 hold_8_7 hold_8_8 hold_8_9 hold_8_10 hold_8_11 hold_9_1
 hold_9_2 hold_9_3 hold_9_4 hold_9_5 hold_9_6 hold_9_7 hold_10_1 hold_10_2
 hold_10_3 hold_10_4 hold_10_5 hold_10_6 hold_10_7 hold_10_8 hold_10_9
 hold_10_10 hold_10_11 hold_10_12 hold_10_13 hold_10_14 hold_10_15
 hold_10_16 hold_10_17 hold_10_18 hold_11_1 hold_11_2 hold_11_3 hold_11_4
 hold_11_5 hold_11_6 hold_11_7 hold_11_8 hold_11_9 hold_11_10 hold_11_11
 hold_12_1 hold_12_2 hold_12_3 hold_12_4 hold_12_5 hold_12_6 hold_12_7
 hold_13_1 hold_13_2 hold_13_3 hold_13_4 hold_13_5 hold_13_6 hold_13_7
 hold_13_8 hold_13_9 hold_13_10 hold_13_11 hold_14_1 hold_14_2 hold_14_3
 hold_14_4 hold_14_5 hold_14_6 hold_14_7 hold_14_8 hold_14_9 hold_14_10
 hold_14_11 hold_15_1 hold_15_2 hold_15_3 hold_15_4 hold_15_5 hold_15_6
 hold_15_7 switch_source_0 switch_source_1 switch_source_2 switch_source_3
 switch_source_4 switch_source_5 switch_source_6 switch_source_7
 switch_source_8 switch_source_9 switch_source_10 switch_source_11
 switch_source_12 switch_source_13 switch_source_14 switch_source_15
 switch_0_sink switch_1_sink switch_2_sink switch_3_sink switch_4_sink
 switch_5_sink switch_6_sink switch_7_sink switch_8_sink switch_9_sink
 switch_10_sink switch_11_sink switch_12_sink switch_13_sink switch_14_sink
 switch_15_sink switch_10_14 switch_14_10 switch_3_11 switch_11_3
 switch_8_14 switch_14_8 switch_11_15 switch_15_11 switch_3_15 switch_15_3
 switch_0_6 switch_6_0 switch_2_7 switch_7_2 switch_5_6 switch_6_5
 switch_4_13 switch_13_4 switch_3_13 switch_13_3 switch_9_11 switch_11_9
 switch_10_12 switch_12_10 switch_2_12 switch_12_2 switch_4_12 switch_12_4
 switch_3_7 switch_7_3 switch_5_15 switch_15_5 switch_1_14 switch_14_1
 switch_4_15 switch_15_4 switch_1_4 switch_4_1 switch_2_10 switch_10_2
 switch_9_12 switch_12_9 switch_8_13 switch_13_8 switch_0_13 switch_13_0
 switch_6_11 switch_11_6 switch_2_6 switch_6_2 switch_5_9 switch_9_5
 switch_7_14 switch_14_7 switch_6_9 switch_9_6 switch_1_5 switch_5_1
 switch_2_13 switch_13_2 switch_13_14 switch_14_13 switch_0_10 switch_10_0
 switch_7_11 switch_11_7 switch_3_4 switch_4_3 switch_4_11 switch_11_4
 switch_5_12 switch_12_5 switch_7_13 switch_13_7 switch_10_11 switch_11_10
 switch_4_6 switch_6_4 switch_2_14 switch_14_2 switch_6_10 switch_10_6
 switch_1_13 switch_13_1 switch_5_11 switch_11_5 switch_14_15 switch_15_14
 switch_4_7 switch_7_4 switch_5_7 switch_7_5 switch_12_15 switch_15_12
 switch_0_8 switch_8_0 switch_8_9 switch_9_8 switch_0_2 switch_2_0
 switch_0_12 switch_12_0 switch_11_12 switch_12_11 switch_0_1 switch_1_0
 switch_4_14 switch_14_4 switch_7_10 switch_10_7 switch_2_5 switch_5_2
 switch_2_15 switch_15_2 switch_6_14 switch_14_6 switch_6_13 switch_13_6
 switch_0_5 switch_5_0 switch_6_15 switch_15_6 switch_4_9 switch_9_4
 switch_10_15 switch_15_10 switch_0_15 switch_15_0 switch_1_2 switch_2_1
 switch_8_11 switch_11_8 switch_7_12 switch_12_7 switch_4_5 switch_5_4
 switch_2_8 switch_8_2 switch_1_15 switch_15_1 switch_12_13 switch_13_12
 switch_13_15 switch_15_13 switch_1_7 switch_7_1 switch_8_12 switch_12_8
 switch_9_13 switch_13_9 switch_0_7 switch_7_0 switch_1_6 switch_6_1
 switch_2_9 switch_9_2 switch_7_15 switch_15_7 switch_5_13 switch_13_5
 switch_11_14 switch_14_11 switch_1_8 switch_8_1 switch_3_14 switch_14_3
 switch_0_14 switch_14_0 switch_8_10 switch_10_8 switch_9_15 switch_15_9
 switch_3_12 switch_12_3 switch_3_10 switch_10_3 switch_1_11 switch_11_1
 switch_0_11 switch_11_0 switch_2_11 switch_11_2 switch_6_12 switch_12_6
 switch_12_14 switch_14_12 switch_3_8 switch_8_3 switch_3_6 switch_6_3
 switch_8_15 switch_15_8 switch_7_8 switch_8_7 switch_3_5 switch_5_3
 switch_10_13 switch_13_10 switch_0_9 switch_9_0 switch_6_8 switch_8_6
 switch_0_3 switch_3_0 switch_2_3 switch_3_2 switch_7_9 switch_9_7
 switch_1_3 switch_3_1 switch_6_7 switch_7_6 switch_5_8 switch_8_5
 switch_1_10 switch_10_1 switch_5_10 switch_10_5 switch_3_9 switch_9_3
 switch_1_12 switch_12_1 switch_2_4 switch_4_2 switch_1_9 switch_9_1
 switch_5_14 switch_14_5 switch_4_10 switch_10_4 switch_11_13 switch_13_11
 switch_0_4 switch_4_0 switch_9_10 switch_10_9 switch_9_14 switch_14_9
 switch_4_8 switch_8_4
Generals
 start_0 start_1 start_2 start_3 start_4 start_5 start_6 start_7 start_8
 start_9 start_10 start_11 start_12 start_13 start_14 start_15
End
