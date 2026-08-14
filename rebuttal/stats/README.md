# Rebuttal statistics module

Read-only inferential analyses answering the reviewer's statistics comments. Nothing
here re-runs a VLM; everything is built on the existing `data/`, `external_embeds/`
and judge-parquet artifacts. Outputs land in `rebuttal/outputs/`.

## Requirements
`numpy`, `pandas`, `scipy`, `statsmodels` (installed), `scikit-learn`. Pure Python.

## Run
```bash
cd rebuttal/stats
python factorial_ratings.py 10000      # #1a #1b #13  (perm count arg)
python block2_distributions.py         # #3  #12
python permanova_embed.py mpnet cosine 5000   # #1c  (embedding, metric, perms)
python rsa_noise_ceiling.py mpnet 5000        # #4   (embedding, boots)
python judge_reanalysis.py 10000       # #8 #10 #11 #14
python power_analysis.py 5000          # #2
python length_style.py                 # #6
```
Robustness re-runs (done): `python permanova_embed.py qwen cosine 5000` and
`python rsa_noise_ceiling.py qwen 3000` -- both replicate the mpnet pattern.

## Files → reviewer comments

| Script | Comments | Core output |
|---|---|---|
| `data_io.py` | shared | loaders + Lima/NYC map + factor labels |
| `factorial_ratings.py` | #1a #1b #13 | permutation F for geo / system / interaction, effect sizes + CIs, per-subject-z robustness, mixed + ordinal back-ups |
| `block2_distributions.py` | #3 #12 | KS p (BH), Wasserstein bootstrap CI + permutation p; naive-vs-matched Human-VLM gap |
| `permanova_embed.py` | #1c | factorial PERMANOVA on embedding centroids; pseudo-F, R², perm p |
| `rsa_noise_ceiling.py` | #4 | human noise ceiling + bootstrap CIs on RSA |
| `judge_reanalysis.py` | #8 #10 #11 #14 | count reconciliation, family ablation, budget split, rubric rescale, geo perm |
| `power_analysis.py` | #2 | video-level MDE at 80% power |
| `length_style.py` | #6 | word-count distributions per group×block + length-confound on the judge |

## Key design choices
- **Geography is a video property** → permutation nulls reshuffle the 10/10 Lima/NYC
  video labels; power is computed at the **video level** (n=10/group), not per answer,
  to avoid pseudo-replication (a naive answer-level F gives ~25% type-I error).
- **Effect sizes lead, p-values follow.** With 20 reps a KS test is significant for
  tiny distributional gaps, so Wasserstein magnitude + CI is the primary read-out.
- **`system_type` main effect vs `geo×system_type` interaction** are reported
  separately: the interaction is the actual cross-cultural-generalization question;
  the geo main effect on *factual* answers is partly a trivial content difference
  between the two clip sets.
- Primary embedding = `allmpnet_batch1` (reproducible, batch=1); `qwen` as robustness.

All scripts seed `numpy.random.default_rng(20260810)`.
