# rl_acquisition — Shapley-primed RL modality acquisition

Fitted-Q acquisition policy that sits on a **frozen** sequential backbone and decides, per
sample, which modality to acquire next and when to stop. Ported here from the standalone
`RL_AAAI_branch` working tree.

## Layout

    *.py            flat module — these import each other by bare name (`import rl_core`,
                    `import environment`, `import shapley`) and resolve their artifact
                    directories relative to their own file. Keep them at this level.
    scripts/        shell runners (sweeps, ablations, cache workers)
    backups/        pre-edit copies of modules changed in place
    EXPERIMENT_PROTOCOL.md   the measurement protocol the sweeps follow
    README_BRANCH_PORT.md    notes from the original port

Untracked at runtime (see `.gitignore`), created as siblings of the modules:

    checkpoints[_<runset>]/  trained Q-networks
    cache[/<subdir>]/        prefix-tree caches (softmax + GFLOPs per ordered subset)
    results[_<runset>]/      `<design>_<ds>_{sweep,point}.csv`
    figures[_<runset>]/      generated plots
    logs/, rl_routing/       run logs and routing dumps

## Entry points

| file | role |
|---|---|
| `run_design.py` | main sweep driver — `--design dqn_qprior --dataset <ds> --ablation <a>` |
| `dqn_policy_qprior.py` | the Q-prior, residual network and FQI training loop |
| `dqn_policy_qprior_noexit.py` | `nostop` rollout (conf>=tau exit, no STOP action) |
| `environment.py` | backbone loading, prefix-tree cache build, per-modality GFLOPs |
| `rebuild_caches.py` | builds the depth-M caches: `rebuild_caches.py <ds> --splits val,test` |
| `baselines.py` | B1 Shapley-restricted / B2 all-available / B3 random-order / B_full |
| `shapley.py` | conditional and marginal Shapley advice |

## Environment variables

| var | meaning |
|---|---|
| `RLB_RUNSET` | selects `checkpoints_<runset>/`, `results_<runset>/`, `figures_<runset>/` |
| `RLB_CACHE_SUBDIR` | selects `cache/<subdir>/` (e.g. `botneck_16` for the K=16 set) |
| `RLB_CODE_REPO` | tree holding the backbone code (`script/training`, `multimodal_model`) |
| `RLB_REPO` | tree holding `runs/` with the backbone checkpoints |
| `RLB_ABLATION` | `full` \| `zeroq` \| `randorder` \| `uniform` \| `nostop` |
| `RLB_ADVICE_MODE` | `conditional` (default) or `marginal` (full Shapley) |

## Runsets

`k16` is the K=16 bottleneck set used for the reported ablations; `k16p` adds the
force-first / exact-DP / stop-BCE variants; `k16_dropaug`, `mmfi_rebal` and
`utd_mhad_dropaug` are per-dataset retrains. Each has its own checkpoints, results and
figures directory, so runsets can never overwrite one another.

## Note on paths

`environment.py` previously hard-coded `/home/group/maestro_visual/...` for both the code
tree and the runs tree, and the shell runners `cd`-ed there. Those paths no longer exist;
both now default to the current locations and are overridable via `RLB_CODE_REPO` /
`RLB_REPO`. If a script cannot find a backbone checkpoint, check those two first.
