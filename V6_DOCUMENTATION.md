# ML Challenge 2026: Astralis V6 entity resolution

**Team:** Astralis
**Date:** 26 September 2026

| Player | Organisation or college |
| --- | --- |
| Shaurya Singh | Noida Institute of Engineering And Technology (NIET), Greater Noida |
| Dakshanya Maddala | Texas A&M University |
| Jethin kosaraju | Noida Institute of Engineering And Technology (NIET), Greater Noida |

## Methodology

For every Source 1 business, we retrieve Source 2 and Source 3 candidates,
score them with LightGBM, and rerank the uncertain retained US and India pairs
with a multilingual transformer. We return zero or more target IDs. Every
Source 1 record remains in the tab-separated output, including singletons and
France records.

## Blocking and candidate set

Blocking uses only the organizer's records. Every route requires the same
normalized country string; no country is excluded. Name, address, city,
number, rare-word, and transliteration rules form a deduplicated union.
`candidate_pairs.tsv` is the full 201,717,297-pair set fed to the V5 matching
stage. Every final match belongs to this set. No external business lookup or
geocoding service is used.

## Matching models

V5 LightGBM uses normalized and transliterated name and address similarities,
token overlap, address number agreement, source identity, name frequency, and
candidate competition. It was fit on Source 1 hash buckets 2-41, with bucket
42-49 validation selecting a 0.625 threshold.

The second stage starts from the Apache 2.0 licensed
`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` checkpoint, with about 100 million
parameters. It sees each pair's business names and addresses, without entity
IDs. We fine-tuned it for one epoch on 53,150 V5 validation pairs, including
25,150 matches and difficult nonmatches. Training took 6,644 steps and about
25 minutes on an NVIDIA RTX 2050 with 4 GB VRAM.

For saved V5 US and India pairs at or below 0.98 probability, we compute:

```text
new_probability = sigmoid(logit(v5_probability) + 0.6 * reranker_logit)
```

Pairs above 0.98 retain their V5 score. We retain saved US and India pairs
with a new score at least 0.60. France keeps the V5 score and the
portal-evaluated 0.80 threshold. The new candidate classifier cannot add
below-threshold V5 pairs in this fast submission variant.

## Evaluation

The blend weight and threshold were selected on audit2. Audit3, Source 1 hash
buckets 70-79, was not used to fit either model or select these numerical
settings, but it was used in earlier exploratory diagnostics. On its 21,878
Source 1 entities, V5 scored 0.93675 macro F0.5 and
the deployable V6 variant scored 0.94728. India's score rose from 0.91454 to
0.93049; the US score rose from 0.95148 to 0.95842.

An experimental rerank of additional lower-scored V5 candidates reached
0.95956 on audit3, but it requires an additional full test inference pass and
is not the output described here. The provided training labels contain no
France examples. The portal-evaluated public score for this V6 output is
**0.927**, compared with **0.913** for the France-threshold V5 variant.
Neither the local nor public score estimates the final private leaderboard.

## Reproduction and fair play

`code/business_entity_resolution/README.md` gives the commands from supplied
TSVs through training, blocking, inference, and validation. The package
includes both trained models, pinned dependencies, source, and this document.
The final output uses no external business data. The transformer checkpoint
is Apache 2.0 licensed and below the challenge's 8-billion-parameter limit.
