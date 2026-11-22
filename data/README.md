# `data/` — input and preprocessing

This folder contains preprocessing code only. The input table is not
redistributed in this repository.

## Required input

Place the curated table at:

```text
data/nasicon.csv
```

Expected columns include:

- `Formula`, `log10S`, `S`, `Index`, `Reference`
- `Temm`, `t1`, `Tems`, `t2`, `Temt`, `th`
- `a`, `b`, `c`, `Vol`
- `N0`, `N1`, `N2`, `N3`, `N4`, `N5`
- `Neg1`, `Neg2`, `Neg3`, `Neg4`, `Neg5`
- `struct`
- optional `di`

The scripts tolerate optional columns where documented in the code, but
`Formula` and `log10S` are required.

## Run order

```bash
python data/prepare_dataset.py
python data/compute_entropy.py
```

## Outputs

| File | Description |
|---|---|
| `data/nasicon_cleaned.csv` | Cleaned modelling table |
| `data/entropy/nasicon_with_entropy.csv` | Cleaned table plus site-resolved entropy columns |

The entropy output contains:

- `S_M_site`
- `S_X_site`
- `S_total_site_sum` = `S_M_site + S_X_site`
- `Delta_Smix_framework` = `n_M * S_M_site + n_X * S_X_site`

## Data availability

The curated input table is not included here. Access is described in the
manuscript's data-availability statement.
