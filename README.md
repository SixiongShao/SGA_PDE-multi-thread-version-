# SGA\_PDE-multi-thread-version-

Based on the original SGA–PDE (Y. Chen, 2022), this is a multi-thread version designed to increase population diversity and avoid a single-term PDE sweeping the entire population.

## Conceptual outline of SGA–PDE

The Symbolic Genetic Algorithm for PDE discovery (SGA–PDE) targets equations of the generic form

$$
\partial_t u(x,t)=\sum_{k=1}^{K} \theta_k\phi_k\big(u,x,t\big),
$$

where the $\phi_k$ are *symbolically generated* candidate terms drawn from a primitive library $\mathcal{L}$ (e.g., $+, -, \times, /, \partial_x, \partial_x^2, (\cdot)^2, \sin$, etc.). Given sampled data $u(x,t)$ on a space–time grid:

1. **Term generation.** SGA builds trees from $\mathcal{L}$ to form a set $\{\phi_k\}_{k=1}^K$. Each individual (“model”) is a sum of such trees.
2. **Feature evaluation.** Each $\phi_k$ is evaluated pointwise on the grid to form a design matrix $X \in \mathbb{R}^{N\times K}$ (rows: space–time samples; columns: terms). The target vector is $y=\partial_t u$ computed by finite differences.
3. **Sparse coefficient fit.** Coefficients $\theta$ are obtained by sparse or shrinkage regression (e.g., STRidge), which promotes parsimony and numerical stability.
4. **Model scoring.** A score balances data misfit (MSE, or a task-appropriate metric) and complexity (e.g., AIC using the number of active coefficients).
5. **Evolution.** Selection (keep low-AIC models), variation (crossover at term level; subtree mutations; occasional term replacement), and repetition across generations refine both structure and coefficients.
6. **Stopping and simplification.** The best-scoring model is simplified by pruning negligible coefficients and algebraic cleanup to yield a concise PDE expression.

### Refinement: multi–pool evolution to avoid single–term collapse

The original SGA–PDE uses a *single* population. In noisy or misspecified settings this creates a common failure mode: a spurious *single–term* PDE that happens to score better than other candidates early on can sweep the population through replacement. Once this happens, the GA rarely reaches the correct equation if it requires $\ge 2$ terms, because adding a new term initially worsens the (AIC-penalized) score and is selected against (premature convergence).

To prevent this collapse, we introduce **three concurrent pools (subpopulations)** that evolve in parallel under the same scoring (STRidge + AIC) but maintain their own lists of candidate PDEs and scores, and apply selection, variation, and replacement *locally*, with operators tuned per pool.

Selection, variation, and replacement are executed *within* each pool, so a lucky single–term model cannot dominate the entire search space. The pools are weakly coupled by controlled migration and assembly, which preserves diversity while enabling multi–term discovery.

#### Pools

* **$\mathcal{P}_1$ (exploitation & recombination).**
  Individuals are sorted by score each generation; a top fraction is chosen as *parents* for crossover. Crossover swaps one randomly chosen term (tree) between two parents to form offspring; only novel offspring (checked against a global `pde_lib`) are admitted and re-scored. Elitism keeps the best one unchanged; all others undergo *mutation* at node level with probability `p_mute`, and *term replacement* with probability `p_rep`. After adding new variants, the pool is re-sorted and truncated back to capacity. This yields a balanced *exploit-then-explore* loop with strong recombination pressure inside $\mathcal{P}_1$.

* **$\mathcal{P}_2$ (diversification without crossover).**
  This pool maximizes structural diversity by applying *full* mutation and *full* replacement to each kept individual every generation (no crossover). This design acts as a high-entropy generator of two-term (and small multi-term) building blocks that are less likely to be pruned prematurely by recombination dynamics.

* **$\mathcal{P}_{\ge 3}$ (aggressive exploration).**
  As in $\mathcal{P}_2$, every individual is fully mutated and fully replaced each generation (no crossover). Empty or degenerate structures are reset to fresh random PDEs. This creates a steady stream of multi-term candidates probing wider structural neighborhoods than $\mathcal{P}_1$ can reach by incremental edits alone.

### Migration & cross–pool injection

After each generation’s within-pool updates, the current best from $\mathcal{P}_3$ and from $\mathcal{P}_2$ are *injected* into $\mathcal{P}_1$ (added and re-ranked). When $\mathcal{P}_3$ produces the overall best model, its champion is also injected into $\mathcal{P}_2$. This migration keeps $\mathcal{P}_1$ supplied with promising multi-term structures while preventing any single-term incumbent in $\mathcal{P}_1$ from monopolizing the search.
