# ML Challenge 2026: Astralis V8 entity resolution

**Team:** Astralis
**Date:** 27 September 2026

| Player | Organisation or college |
| --- | --- |
| Shaurya Singh | Noida Institute of Engineering And Technology (NIET), Greater Noida |
| Dakshanya Maddala | Texas A&M University |
| Jethin kosaraju | Noida Institute of Engineering And Technology (NIET), Greater Noida |

## Methodology

V8 retrieves same country Source 2 and Source 3 candidates for every Source 1
business. A LightGBM model scores normalized name, address, transliteration,
number, source, frequency, and competition features. A multilingual MiniLM
cross encoder reranks retained US and India candidates. The decoder then gives
each target to its strongest Source 1 owner. A France recovery route retrieves
legal suffix compatible records and retains only high address similarity and
high reranker score pairs.

## Models and licensing

The first stage is LightGBM. The second stage starts from the Apache 2.0
licensed `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` checkpoint, about 100
million parameters, and was fine tuned only on the supplied training records.
Both trained artifacts are included. No external business identity data,
geocoder, or lookup service is used.

## Candidate generation

Candidate routes include normalized names and addresses, city and postal
tokens, address numbers, rare tokens, transliteration, and France legal suffix
agreement. Routes are unioned and deduplicated. Every final prediction appears
in `candidate_pairs.tsv`. Every Source 1 test record appears exactly once in
both output files, including empty predictions.

## Evaluation

The final deployable route scored 0.95958 macro F0.5 on the audit3 holdout,
up from 0.94728 for V6. A separate retrieval experiment reached 0.96561 but
was not materialized across the full test set and is not included in this
package. The last observed public leaderboard score is 0.927 for V6. V8 has
not yet received a public score. France has no supplied labels; suffix
ablations and full structural validation were used instead.

## Reproduction

`code/business_entity_resolution/README.md` contains the end to end commands.
The package includes its exact final outputs, source, pinned dependencies,
LightGBM model, transformer model, and this methodology document.
