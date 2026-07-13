# Encoder Alignment Empirical Status

## Measurement evidence

The surrogate worktree has completed read-only representation measurements for
four lightweight proxies and three disjoint held-out targets. Image-hash audit
found that Caltech has 50/50 unique source images, LLaVA has 15/60 unique source
images, and receipt OCR has 19/40 unique source images. LLaVA and receipt were
therefore rerun with explicit SHA-256 deduplication. This branch consumes those
CSVs by explicit path; it does not duplicate the embedding runs.

On Caltech, centered CKA is high (`0.928--0.969`) while NO@1 is only
`0.30--0.48`. Across the 12 proxy-target pairs, descriptive rank correlations
between datasets are:

| Metric | Caltech--LLaVA | Caltech--receipt | LLaVA--receipt |
| --- | ---: | ---: | ---: |
| centered CKA | 0.301 | 0.308 | 0.175 |
| uncentered alignment | 0.944 | 0.406 | 0.434 |
| NO@1 | -0.056 | 0.322 | 0.168 |
| NO@5 | 0.608 | 0.196 | 0.196 |

These values have only 12 pairs and no confidence interval. They are evidence
that alignment and neighborhood rankings can be data-distribution dependent,
not a transfer-ASR test. Before deduplication, receipt NO@1 was spuriously about
`0.925--0.967`; after deduplication its pairwise range is `0.053--0.368`.
Duplicate-image and neighbor-tie auditing is therefore mandatory for NO.

## Completed Stage 1 attack join

All 12 single-proxy trials are complete. The inferential join contains 720
item-target rows: four proxies, three seeds, 20 items and three held-out targets.

Global alignment does not predict transfer in this matrix:

| Predictor | Spearman with held-out ASR | 95% bootstrap interval |
| --- | ---: | ---: |
| centered linear CKA | -0.025 | [-0.739, 0.594] |
| uncentered alignment | -0.473 | [-0.742, 0.036] |
| NO@1 | 0.685 | [0.132, 0.899] |
| NO@5 | -0.025 | [-0.691, 0.578] |
| NO@10 | 0.014 | [-0.662, 0.628] |

The distance chain is partially supported. Intervals below use an item-clustered
bootstrap, retaining all seed, proxy, and target observations for each of the 20
unique images. `delta D_proxy` and `delta D_target` have Spearman 0.309, CI
[0.168, 0.451]. More negative `delta D_target` predicts target success with
Spearman -0.391, CI [-0.527, -0.226]. Successful rows reduce target distance by
0.0502 on average versus 0.0008 for failures. The direct association between
`delta D_proxy` and target success is weaker (-0.184, CI [-0.365, 0.039]).

The claimed CKA-to-neighborhood relationship depends on neighborhood scale.
Centered CKA has little association with NO@1 (-0.116, interval crosses zero),
but correlates with NO@5 and NO@10 at 0.834 and 0.805. Conversely, only NO@1
predicts transfer here. Global CKA may preserve coarse neighborhoods while
missing the local ordering relevant to transfer.

Sample sensitivity reinforces this issue. Centered CKA rankings are reasonably
stable between 50 and 200 images (0.839), whereas NO@1 rankings are unstable
(-0.227). The empirical non-degenerate-margin assumption therefore cannot be
treated as mild without reporting neighbor-margin distributions.

Evidence classification:

- `D_proxy -> D_target`: supported in direction, moderate effect;
- `D_target -> target success`: supported;
- global centered CKA predicts transfer: unsupported;
- uncentered alignment predicts transfer: unsupported;
- NO@1 predicts transfer: supported in this 12-pair matrix, requiring replication;
- CKA implies local NO@1: unsupported;
- CKA tracks broader NO@5/10: supported descriptively.

No API evidence or proprietary architecture assumption is used in these
classifications.
