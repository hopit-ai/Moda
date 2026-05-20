# fashion200k error analysis — Marqo-FashionSigLIP

Teacher run: `Marqo-FashionSigLIP_subsample10000_seed42`
Queries analysed: 2000


### Overall (2000 queries)

- hit@1   = 0.4160
- hit@10  = 0.7940
- hit@100 = 0.7940
- MRR     = 0.5390
- median rank of first positive (excluding misses) = 1
- queries that miss top-100 entirely = 412 (20.6%)

### By query length (#words)

| #words | n queries | hit@1 | hit@10 | hit@100 |
| --- | ---: | ---: | ---: | ---: |
| 0-5 | 9 | 0.2222 | 0.5556 | 0.5556 |
| 5-10 | 22 | 0.3182 | 0.7727 | 0.7727 |
| 10-20 | 67 | 0.5075 | 0.8507 | 0.8507 |
| 20-30 | 351 | 0.4188 | 0.7920 | 0.7920 |
| 30-50 | 1149 | 0.4047 | 0.7929 | 0.7929 |
| 50-100 | 398 | 0.4422 | 0.7990 | 0.7990 |
| 100-∞ | 4 | 0.2500 | 0.5000 | 0.5000 |

### By attribute richness

| has_color | has_material | has_pattern | n queries | hit@10 |
| --- | --- | --- | ---: | ---: |
| 1 | 0 | 0 | 681 | 0.7386 |
| 1 | 0 | 1 | 489 | 0.8466 |
| 1 | 1 | 0 | 430 | 0.7698 |
| 1 | 1 | 1 | 383 | 0.8512 |

### By #colors mentioned

| #colors | n queries | hit@10 |
| --- | ---: | ---: |
| 0 | 17 | 0.8235 |
| 1 | 852 | 0.7664 |
| 2 | 812 | 0.7993 |
| 3 | 240 | 0.8458 |
| 4 | 64 | 0.8438 |

### By #garment-type words mentioned

| #garments | n queries | hit@10 |
| --- | ---: | ---: |
| 0 | 90 | 0.7111 |
| 1 | 1297 | 0.8150 |
| 2 | 471 | 0.7813 |
| 3 | 123 | 0.6992 |
| 4 | 18 | 0.6667 |

### Words most associated with FAILURE (sorted by failure rate, min 20 occurrences)

| Word | Failure rate | n queries containing it |
| --- | ---: | ---: |
| modern | 0.452 | 31 |
| hands | 0.429 | 21 |
| not | 0.423 | 26 |
| pencil | 0.400 | 25 |
| elegant | 0.381 | 21 |
| pocket | 0.372 | 121 |
| straight | 0.368 | 57 |
| chest | 0.367 | 30 |
| hair | 0.364 | 33 |
| solid | 0.364 | 22 |
| edge | 0.364 | 22 |
| they | 0.363 | 124 |
| her | 0.361 | 36 |
| pants | 0.352 | 179 |
| classic | 0.350 | 40 |
| ankle | 0.346 | 26 |
| stylish | 0.345 | 29 |
| intricate | 0.333 | 27 |
| which | 0.333 | 27 |
| leggings | 0.325 | 83 |
| pull | 0.324 | 37 |
| simple | 0.324 | 34 |
| have | 0.324 | 306 |
| cuffed | 0.321 | 28 |
| pair | 0.321 | 302 |
| leg | 0.319 | 72 |
| wide | 0.312 | 125 |
| paired | 0.308 | 26 |
| cut | 0.308 | 65 |
| left | 0.306 | 49 |

### Words most associated with SUCCESS (lowest failure rate)

| Word | Failure rate | n queries containing it |
| --- | ---: | ---: |
| mermaid | 0.000 | 22 |
| thin | 0.000 | 26 |
| fringe | 0.000 | 23 |
| purple | 0.031 | 64 |
| ribbed | 0.032 | 31 |
| stripes | 0.034 | 59 |
| scoop | 0.034 | 29 |
| sequins | 0.036 | 28 |
| leaves | 0.036 | 28 |
| quilted | 0.038 | 26 |
| yellow | 0.039 | 76 |
| summer | 0.040 | 25 |
| tulle | 0.043 | 23 |
| including | 0.050 | 20 |
| striped | 0.051 | 99 |
| mesh | 0.054 | 37 |
| gold | 0.060 | 50 |
| bow | 0.062 | 32 |
| mini | 0.062 | 32 |
| like | 0.067 | 30 |
| colors | 0.067 | 90 |
| orange | 0.067 | 30 |
| green | 0.067 | 150 |
| cuff | 0.070 | 43 |
| colored | 0.077 | 39 |
| chiffon | 0.077 | 39 |
| bohemian | 0.079 | 38 |
| brown | 0.079 | 126 |
| metallic | 0.083 | 36 |
| sleeved | 0.083 | 24 |

### Google-SigLIP2-B16-384_subsample10000 vs Marqo-FashionSigLIP (n=2000 common queries)

| | compare hit@10 = 0 | compare hit@10 = 1 |
| --- | ---: | ---: |
| **teacher hit@10 = 0** | both miss: 302 (15.1%) | only compare wins: **110** (5.5%) |
| **teacher hit@10 = 1** | only teacher wins: **149** (7.5%) | both hit: 1439 (72.0%) |

- teacher hit@10 alone: 0.7940
- compare hit@10 alone: 0.7745
- **ENSEMBLE upper bound (any model hits)**: 0.8490
- ⇒ ensembling could lift hit@10 by **+0.0550** absolute over teacher alone (if we had a perfect router that picked the right model per query).