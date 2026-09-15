# FairEval data workspace

FairEval uses six third-party recommendation datasets. The repository stores **code, schemas, provenance manifests, hashes, and small derived metadata**, but it does not redistribute raw datasets whose licenses/terms may restrict redistribution.

## Canonical local layout

Run:

```bash
python scripts/bootstrap_data_layout.py --data-root data
```

This creates:

```text
data/
  raw/
    personality2018/
    music_master_bfi2/
    reasoner/
    movielens_1m/
    lastfm_1k/
    mind/
  processed/
    personality2018/
    music_master_bfi2/
    reasoner/
    movielens_1m/
    lastfm_1k/
    mind/
```

Both `data/raw/` and `data/processed/` are gitignored by design.

## Source-of-truth manifests

- `configs/datasets.yaml` — semantic role, observed fields, exclusions, source URL.
- `configs/dataset_releases.yaml` — exact upstream release/license/checksum lock.
- `scripts/hash_raw_release.py` — deterministic raw-directory file manifest + release SHA-256.
- `faireval prepare ...` / freeze utilities — deterministic benchmark instances after release lock.

## Promotion rule

A dataset is allowed into the paid pilot only after all of the following are true:

1. the exact upstream release is present locally;
2. its license/terms are reviewed and recorded;
3. the raw release manifest/hash is generated;
4. `configs/dataset_releases.yaml` is updated with the exact release identity and digest;
5. deterministic FairEval instances are frozen and verified;
6. `python scripts/check_pilot_readiness.py --strict` passes.

Do **not** replace a missing release with a similar dataset (for example MovieLens latest instead of MovieLens-1M, HETRec Last.fm instead of the frozen Last.fm-1K source, or a different MIND split) after the experiment is frozen.

## Why raw data are not committed

Keeping the repository reproducible does not require copying third-party datasets into GitHub. In several cases the upstream terms require attribution, non-commercial use, click-through acceptance, or other restrictions. FairEval therefore commits exact acquisition instructions and cryptographic identities while leaving controlled/raw files local.
