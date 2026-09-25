# V5 measured results

V5 keeps the V4 candidate set and adds pair features for name frequency and
competition among candidates for the same Source 1 entity. The LightGBM model
uses 63 leaves, minimum leaf size 100, learning rate 0.04, and early stopping
on validation loss. Buckets 2-41 fit the model and buckets 42-49 select its
0.625 threshold. Prediction uses early stopping with frequency 10 and margin
10, as in validation.

| Untouched audit3, buckets 70-79 | V4 | V5 |
| --- | ---: | ---: |
| Source 1 entities | 21,878 | 21,878 |
| True links | 76,167 | 76,167 |
| Candidate pairs | 2,828,283 | 2,828,283 |
| Candidate recall | 94.48% | 94.48% |
| Macro F0.5 | 0.92402 | 0.93675 |
| India macro F0.5 | 0.90095 | 0.91454 |
| US macro F0.5 | 0.93933 | 0.95148 |

These figures are from provided training labels. The public V4 submission
scored 0.896. V5's public score is unknown until a portal evaluation. France
has no labeled training data, and the local audit does not estimate its score.

Full test output passed the complete structural validator. Validation summary:

```json
{
  "ok": true,
  "source1_rows": 1732544,
  "expected_source1_rows": 1732544,
  "candidate_pairs": 201717297,
  "matched_pairs": 5886292,
  "empty_predictions": 104061,
  "errors": []
}
```
