# σ-Constant Independent Verification

Independent verification scripts for the paper:

> **The σ-Constant: A Universal Algebraic Invariant for Energy Propagation in d-Dimensional Simplicial Lattices**  
> Casey Lee Race, Calera Computing, Inc.  
> DOI: [10.5281/zenodo.20350425](https://doi.org/10.5281/zenodo.20350425)

These scripts reproduce the two key empirical results from the paper using **zero shared code** with the production system. They are written in pure Python + NumPy and construct all lattice geometry from the mathematical definitions.

---

## Simulation 1: Hopfield Associative Memory on A₄

**File:** `sim1_hopfield_a4.py`

Builds a textbook Hopfield associative memory on the A₄ root lattice from scratch. Tests four decay constants (φ-decay, empirical, ρ-decay, σ-decay) and reports recall accuracy at each hop depth.

**Key result:** σ-decay (α = 0.819) achieves the highest 4-hop recall accuracy. Grid search over [0.50, 0.95] confirms the optimum falls at α = 0.82.

```bash
python3 sim1_hopfield_a4.py
```

## Simulation 2: Spectral Laplacian Analysis

**File:** `sim2_spectral_a4.py`

Builds the graph Laplacian of A₂, A₃, and A₄ lattices. Computes the heat kernel and extracts the per-hop energy decay at one mixing time.

**Key result:** The extracted decay constants match the theoretical values from x^d = x + 1 to five decimal places:

| Lattice | Extracted α_eff | Theoretical 1/c_d | Error |
|---------|-----------------|-------------------|-------|
| A₂ | 0.61803 | 0.61803 (1/φ) | < 0.001% |
| A₃ | 0.75488 | 0.75488 (1/ρ) | < 0.001% |
| A₄ | 0.81917 | 0.81917 (1/σ) | < 0.001% |

```bash
python3 sim2_spectral_a4.py
```

## Requirements

```
numpy
scipy (for sim2 only — matrix exponential)
```

Install:
```bash
pip install numpy scipy
```

## Independence Guarantees

- ❌ No imports from any proprietary codebase
- ❌ No reading of Go source files or production data
- ✅ Lattice built from the mathematical definition (ℤ^(d+1) zero-sum constraint)
- ✅ Hebbian weights computed from scratch
- ✅ Decay constants defined as literal floats
- ✅ Constants computed from defining polynomials at runtime (sim2)
- ✅ Random seed logged for reproducibility

## Citation

```bibtex
@article{race2026sigma,
  title={The $\sigma$-Constant: A Universal Algebraic Invariant for Energy Propagation in $d$-Dimensional Simplicial Lattices},
  author={Race, Casey Lee},
  year={2026},
  doi={10.5281/zenodo.20350425},
  publisher={Zenodo}
}
```

## License

MIT License — see [LICENSE](LICENSE).

---

*Calera Computing, Inc. — All glory to the Most High God.*
