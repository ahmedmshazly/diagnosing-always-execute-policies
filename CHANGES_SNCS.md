# Scientific revision log

The starting manuscript was `TMLR/submission/main.tex`. The earlier folders and parent research implementation were preserved. Changes below are confined to this new journal package.

| Earlier claim or issue | Revision and reason |
|---|---|
| A new RL method or broadly applicable taxonomy was implied | Framed as a reproducible empirical diagnosis of one simulator |
| Five environments | Four dynamics settings, five environment/reward conditions |
| The learner survived an 81-cell coefficient sweep | The sweep belongs to the non-learning Utility-Based agent; it is reported separately |
| The one-step probe established local optimality | Reports average sampled deviations only; five source episodes, 75 states and 16 shared future seeds are explicit |
| Probe states treated as independent | Revised descriptive intervals resample source-episode clusters; fixed-future uncertainty remains a limitation |
| Original failure penalty was effectively zero | Exact undiscounted identity retained; discounted counter contribution derived and acknowledged |
| Corrected reward meant the learner optimized exactly the evaluation metric | Equality restricted to undiscounted sums; discount and horizon differences stated |
| PPO isolated an optimizer effect | Comparison attributed to whole training procedures; unequal budgets, critic, GAE, update reuse, horizons and validation schedules documented |
| Only one tight-capacity REINFORCE run used a curriculum | All three complete tight runs used the 400-update, three-stage curriculum |
| All high-value runs treated as complete | Three REINFORCE records lack final curves/manifests and stop at validation update 125 in available logs; exploratory status disclosed |
| Reward correction treated as a training-only intervention | Checkpoint selection also uses the run's reward; both parts change |
| Holm correction dropped identical comparisons and counted a duplicate hand reference | Full 34-hypothesis family per pool: 30 learned policies plus four distinct references; identical differences retain p=1 |
| Default scaling reference significant after correction on 250 workloads | Corrected adjusted p is 0.068; positive estimate remains, family-wise significance does not |
| 0.456 described as a probability of strict improvement | Labeled paired win score with half credit for ties; strict win fraction is 0.312 |
| Reflex described as stochastically dominant | Empirical curves cross; claim removed and figure regenerated |
| A nonsignificant result was read as equivalence | Difference tests and exploratory equivalence margins discussed separately |
| Observed-effect power used as proof of adequate power | Described as paired-t sensitivity at the observed effect size, not prospective or Wilcoxon power |
| Evaluation pool called preregistered | Described as recorded and disjoint; no external preregistration claimed |
| Failure probability exogenous implied policy-invariant failure totals | Clarified that policies change running-task exposure |
| High-value mixture called a heavy-tailed distribution | Correctly described as a bounded mixture; historical filename retained for file traceability |
| Execute could be read as selecting any fitting task | Explicitly states top-ranked candidate only, with no fallback to a lower-ranked task |
| REINFORCE termination described as exactly memoryless | Exponential draw is integer-converted and clipped; full rule documented |
| Free provisioning was not prominent | Cost on resources in use and zero action tariff are stated as material scope limits |
| Constrained RL dismissed without testing it | No claim against constrained methods; potential use for hard budgets remains open |

No experiments or training histories were fabricated, and no new policy training was performed. The revision re-evaluates saved checkpoints and recomputes summaries from existing outcomes. Citations were narrowed to directly relevant work and checked against primary sources. The manuscript includes the author's confirmed Kigali affiliation and declarations.
