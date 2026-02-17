**Trasversali**: ALPHA (0.9, 0.5, 0.1), Model (LLaMa30B, LLaMa70B)

**Parameter space**: 

- tipi GPU - T4, L4, A30, A4000, L40S, A100-40, A100-80, H100 (e gruppi relativi)
- regions - eu-west, eu-east, us-east, us-west, asia-east, asia-south



### E1 - Dimensione fissa, GPU Eterogenee

16 GPUs, single-region, batch_size 8.

Variano le GPU disponibili:

- Solo un GPU type (es. L4) `E1-homo-GPU-ID`
  - solo singole
  - singole + gruppi (da 2 o 4 gpu)
- Due tipi di GPU (es. A30, A30, 4xA30, L40S, L40S, 2xL40S, 2xL40S, 4xL40S) `E1-heter2-GPU-GPU-ID`
- Tre tipi di GPU (es. L4, L4, A4000, A100-40, ....) `E1-heter3-GPU-GPU-GPU-ID`
- Cloud mix (es. 2×L40S + 2×L40S + 1×L4 + 2×L4 + 4×L4 + 1×A100-80 + 2×A4000, 1xA100-40, 1xA30) `E1-mix`
- Cloud mix budget (solo GPU economiche A4000, A30, L4) `E1-budget-ID`

Per ogni categoria, faccio 3 combinazioni 
Tot. 6x3 x 3alpha x 2modelli = 108



### E2 - Multi-Region

Pool GPU fisso (16 GPU di tipo Cloud mix).

- `single-region` (all eu-west) - 1
- `eu-split`(8 eu-west, 8 eu-east) - 2
- `eu-unbalanced`(10 eu-west, 6 eu-east) - 2
- `eu-us`(8 eu-west, 8 us-east) - 2
- `us-split`(8 east, 8 west) - 2
- `eu-us-asia`(6 eu, 5, us, 5 asia) - 2

Tot. 11 x 3alpha x 2modelli = 66



### E3 - Cluster Scale

Dimensione del cluster scalata.

- `scale-30`30 GPU - partenza fino a 4 GPU
- 4 GPU in meno ogni simulazione (tolgo quelle più economiche)

Tot. 7 x 3alpha x 2modelli = 42



TOTALE = 108+66+42 = 216 simulazioni.



### E4 - Simulazioni arbitrarie

Arbitrarie. Prendo configurazioni a mano da sito rental, dove cambio GPU costosa con economica ecc...





### Nome

"test\_{esperimento}\_{config}\_{modello}\_{alpha}/"

Esempio: "test_E1_homo-L4-1_LLama30_0.5/"



### Output

- Grafico topologia risultante `solution_graph.png`

- Config file input `{name}.ini`

- ILP solution `ilp_solution.sol`

- ILP config (parameters) `{timestamp}.ini`

- File metriche `metrics.json`

  - Feasibility
  - Throughput

  - Rental Cost

  - Power Cost

  - \# GPU attive

  - Solver time