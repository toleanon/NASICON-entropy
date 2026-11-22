"""Relax the selected candidates with MACE-MP-0 and ASE."""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import numpy as np
import pandas as pd
import torch
import warnings
from pymatgen.core import Species, Composition
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from ase.io import read, write
from ase.optimize import BFGS
from ase.filters import ExpCellFilter
from mace.calculators import mace_mp

from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "paths.py").exists())))
from paths import MLIP_RESULTS, MLIP_SEEDS, SCREENING_RESULTS

warnings.filterwarnings('ignore')

                                                  
CANDIDATES_CSV = SCREENING_RESULTS / "MLIP_Candidates.csv"
MOTHER_CIF = MLIP_SEEDS / "Na3Zr2Si2PO12_R3c.cif"
OUTPUT_DIR = MLIP_RESULTS / "relaxation"

N_FU = 6                                
FMAX = 0.02                            
MAX_STEPS = 500                    
MAX_RESTARTS = 1                    
MACE_MODEL = "medium"                                  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FLOAT64 = True                       
RANDOM_STATE = 42                

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUTPUT_DIR / "relaxation.log"
                                                     


def log(msg, logfile=LOG_PATH):
    """Write a message to stdout and the log file."""
    print(msg)
    with open(logfile, 'a', encoding='utf-8') as f:
        f.write(msg + "\n")


def load_mother_structure(cif_path):
    """Load the parent structure and check the site counts for N_FU=6."""
    if not cif_path.exists():
        raise FileNotFoundError(f"Parent CIF not found: {cif_path}")

    atoms = read(cif_path)
    struct = AseAtomsAdaptor.get_structure(atoms)

    zr_n = sum(1 for s in struct if s.specie.symbol == 'Zr')
    tetra_n = sum(1 for s in struct if s.specie.symbol in ('Si', 'P'))
    na_n = sum(1 for s in struct if s.specie.symbol == 'Na')
    o_n = sum(1 for s in struct if s.specie.symbol == 'O')

    log(f"Parent structure: {struct.formula}, atoms: {len(struct)}")
    log(f"  Site counts: Zr={zr_n}, Tetra={tetra_n}, Na={na_n}, O={o_n}")
    try:
        log(f"  Space group: {SpacegroupAnalyzer(struct).get_space_group_symbol()}")
    except Exception:
        log("  Space-group detection failed; structure loaded")

    expected = {'Zr': 2 * N_FU, 'Tetra': 3 * N_FU, 'Na': 4 * N_FU, 'O': 12 * N_FU}
    actual = {'Zr': zr_n, 'Tetra': tetra_n, 'Na': na_n, 'O': o_n}
    if actual != expected:
        raise ValueError(
            f"Parent site counts do not match N_FU={N_FU}.\n"
            f"  actual: {actual}\n"
            f"  expected: {expected}\n"
            f"Check the parent CIF or N_FU."
        )
    log("  Site-count check passed.")
    return struct


def parse_formula(formula):
    """Parse a formula into an element-count dictionary."""
    comp = Composition(formula)
    return {el.symbol: comp[el] for el in comp.elements}


def build_formula(comp_dict):
    """Build a standardized formula from an element-count dictionary."""
    comp_dict = {k: v for k, v in comp_dict.items() if abs(v) > 1e-6}
    comp_dict['O'] = 12
    metals = [el for el in comp_dict if el not in ['Na', 'O', 'Si', 'P']]
    tetra = ['Si', 'P']
    tetra_present = [el for el in tetra if el in comp_dict]
    order = ['Na'] + sorted(metals) + tetra_present + ['O']
    parts = []
    for el in order:
        amt = comp_dict.get(el, 0)
        if abs(amt) < 1e-6:
            continue
        if abs(amt - round(amt)) < 1e-4:
            parts.append(f"{el}{int(round(amt))}")
        else:
            s = f"{amt:.2f}".rstrip('0').rstrip('.')
            parts.append(f"{el}{s}")
    return "".join(parts)


def _largest_remainder(scaled_counts, n_sites):
    """Convert scaled target counts to integer site counts that sum to n_sites."""
    base = {k: int(np.floor(v)) for k, v in scaled_counts.items()}
    rest = int(n_sites - sum(base.values()))
    if rest < 0:
        raise ValueError(f"Target counts exceed site capacity: {scaled_counts} > {n_sites}")
    order = sorted(scaled_counts, key=lambda k: scaled_counts[k] - base[k], reverse=True)
    for k in order[:rest]:
        base[k] += 1
    if sum(base.values()) != n_sites:
        raise ValueError(f"Incomplete site assignment: sum({base}) != {n_sites}")
    return base


def build_substituted_structure(parent, target_formula, rng):
    """Randomly occupy the parent cell according to the target formula."""
    struct = parent.copy()
    target = parse_formula(target_formula)

    zr_sites = [i for i, s in enumerate(struct) if s.species.elements[0].symbol == 'Zr']
    tetra_sites = [i for i, s in enumerate(struct)
                   if s.species.elements[0].symbol in ('Si', 'P')]
    na_sites = [i for i, s in enumerate(struct) if s.species.elements[0].symbol == 'Na']

    if (len(zr_sites) != 2 * N_FU or len(tetra_sites) != 3 * N_FU
            or len(na_sites) != 4 * N_FU):
        raise ValueError(
            f"Parent site counts do not match N_FU={N_FU}: Zr {len(zr_sites)}, "
            f"tetrahedral {len(tetra_sites)}, Na {len(na_sites)}")

                              
    dopants = {el: amt for el, amt in target.items()
               if el not in ('Na', 'Zr', 'Si', 'P', 'O') and amt > 1e-6}
    zr_counts = _largest_remainder(
        {**{el: amt * N_FU for el, amt in dopants.items()},
         'Zr': target.get('Zr', 0.0) * N_FU}, len(zr_sites))
    order = [int(i) for i in rng.permutation(zr_sites)]
    pos = 0
    for el, n in zr_counts.items():
        for idx in order[pos:pos + n]:
            struct[idx].species = {Species(el): 1.0}
        pos += n

                           
    si_p_counts = _largest_remainder(
        {'Si': target.get('Si', 0.0) * N_FU, 'P': target.get('P', 0.0) * N_FU},
        len(tetra_sites))
    torder = [int(i) for i in rng.permutation(tetra_sites)]
    for idx in torder[:si_p_counts['Si']]:
        struct[idx].species = {Species('Si'): 1.0}
    for idx in torder[si_p_counts['Si']:]:
        struct[idx].species = {Species('P'): 1.0}

                                                                
                           
                                        
    valences = {'Na': 1, 'Zr': 4, 'Si': 4, 'P': 5, 'O': -2,
                'Yb': 3, 'Fe': 3, 'Cr': 3, 'Mg': 2, 'V': 3, 
                'Al': 3, 'Sn': 4, 'Ge': 4, 'In': 3}
    
                       
    current_charge = 0
    for site in struct:
        el = site.specie.symbol
        if el != 'Na':          
            current_charge += valences.get(el, 0)
            
                        
    n_na_theoretical = -current_charge
    n_na = int(round(n_na_theoretical))
    
                           
    orig_na = target.get('Na', 0.0) * N_FU
    if abs(n_na - orig_na) > 0.01:
        log(f"  [charge balance] Na demand {orig_na:.1f} -> adjusted to {n_na} sites (neutral)")
                                                                

    if not 0 < n_na <= len(na_sites):
        raise ValueError(
            f"Na target {target.get('Na', 0.0):.3f}/f.u. -> {n_na} sites; capacity {len(na_sites)}")
    
    keep_na = {int(i) for i in rng.choice(na_sites, size=n_na, replace=False)}
    struct.remove_sites(sorted([i for i in na_sites if i not in keep_na], reverse=True))

                    
    realized = struct.composition.as_dict()
    for el in set(list(realized) + list(target)):
        actual = realized.get(el, 0.0) / N_FU
        expect = target.get(el, 0.0)
        
                                   
        if el == 'Na':
            if abs(actual - expect) > 0.5:
                raise ValueError(f"Na deviation too large: actual {actual:.3f}/f.u. vs target {expect:.3f}/f.u.")
        else:
            if abs(actual - expect) > 0.1:
                raise ValueError(f"Composition deviation too large: {el} actual {actual:.3f}/f.u. vs target {expect:.3f}/f.u.")

    return struct


def relax_structure(atoms, calc, fmax=FMAX, max_steps=MAX_STEPS):
    """Relax an ASE Atoms object, restarting up to MAX_RESTARTS times."""
    atoms.calc = calc

                            
    ecf = ExpCellFilter(atoms)
    opt = BFGS(ecf, trajectory=None, logfile=None)
    converged = False

    try:
        for _ in range(MAX_RESTARTS + 1):
            opt.run(fmax=fmax, steps=max_steps)
            converged = bool(opt.converged())
            if converged:
                break
        final_fmax = float(np.abs(atoms.get_forces()).max())
        return atoms, converged, final_fmax
    except Exception as e:
        log(f"  Relaxation error: {e}")
        return atoms, False, np.inf


def main():
             
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    log("=" * 60)
    log("NASICON MLIP structure relaxation")
    log("=" * 60)

             
    log(f"PyTorch version: {torch.__version__}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"Device: {DEVICE}")
    log(f"Precision: {'float64' if USE_FLOAT64 else 'float32'}")
    log(f"Relaxation: FMAX={FMAX}, MAX_STEPS={MAX_STEPS}, MAX_RESTARTS={MAX_RESTARTS}")
    log(f"MACE model: {MACE_MODEL}")

                        
    log("\n[1] Loading parent structure...")
    parent = load_mother_structure(MOTHER_CIF)

               
    log("\n[2] Loading candidate list...")
    if not CANDIDATES_CSV.exists():
        raise FileNotFoundError(f"Candidate file not found: {CANDIDATES_CSV}")
    # Load candidate list and mother structure.
    df = pd.read_csv(CANDIDATES_CSV)
    mlip_ids = (df['MLIP_ID'].tolist() if 'MLIP_ID' in df.columns
                else [f"M{i + 1}" for i in range(len(df))])
    formulas = df['Formula'].tolist()
    preds = df['pred_log10S'].tolist()
    roles = df['role'].tolist() if 'role' in df.columns else ['unknown'] * len(df)
    s_totals = ((df['S_M'] + df['S_X']).tolist()
                if {'S_M', 'S_X'}.issubset(df.columns) else [np.nan] * len(df))
    log(f"{len(formulas)} candidates to relax")
    for i, fid in enumerate(mlip_ids):
        log(f"  {fid}: {formulas[i]} ({roles[i]})")

                     
    log("\n[3] Initializing MACE-MP calculator...")
    # Relax each candidate and write CIFs plus a summary table.
    calc = mace_mp(
        model=MACE_MODEL,
        device=DEVICE,
        default_dtype="float64" if USE_FLOAT64 else "float32"
    )
    log(f"  MACE model: {MACE_MODEL}")
    if hasattr(calc, 'model_path'):
        log(f"  model path: {calc.model_path}")
    else:
        log(f"  model: {MACE_MODEL} (loaded)")

             
    log("\n[4] Starting structure relaxation...")
    results = []

    for i, (mlip_id, formula, pred_log, role, s_tot) in enumerate(
            zip(mlip_ids, formulas, preds, roles, s_totals)):
        log(f"\n--- [{i + 1}/{len(formulas)}] {mlip_id}: {formula} ({role}) ---")
        log(f"  predicted log10(sigma) = {pred_log:.4f}")
        seed = RANDOM_STATE + i
        real_formula = None
        t_start = time.time()

        try:
            rng = np.random.default_rng(seed)
            substruct = build_substituted_structure(parent, formula, rng)
            real_formula = Composition(substruct.composition).reduced_formula
            atoms = AseAtomsAdaptor.get_atoms(substruct)
                                                    
                                                         
            atoms.info.pop('occupancy', None)
            atoms.arrays.pop('spacegroup_kinds', None)
            log(f"  realized composition: {real_formula}, atoms: {len(atoms)}")

                    
            init_file = OUTPUT_DIR / f"init_{mlip_id}.cif"
            write(init_file, atoms)
            log(f"  initial structure: {init_file}")

                
            log(f"  starting relaxation (fmax={FMAX}, {MAX_STEPS} steps/run, up to {MAX_RESTARTS + 1} runs)...")
            relaxed_atoms, converged, final_fmax = relax_structure(atoms, calc)

                     
            final_energy = relaxed_atoms.get_potential_energy()

                     
            relax_file = OUTPUT_DIR / f"relaxed_{mlip_id}.cif"
            write(relax_file, relaxed_atoms)
            log(f"  relaxed structure: {relax_file}")

                                          
            check = read(str(relax_file))
            if sorted(check.get_chemical_symbols()) != sorted(relaxed_atoms.get_chemical_symbols()):
                log("  [warning] Element counts changed after the CIF round-trip; check the output.")

            runtime = time.time() - t_start

            results.append({
                'index': i + 1,
                'MLIP_ID': mlip_id,
                'Formula': formula,
                'Formula_realized': real_formula,
                'role': role,
                'S_total': s_tot,
                'pred_log10S': pred_log,
                'n_atoms': len(atoms),
                'seed': seed,
                'converged': converged,
                'final_energy': final_energy,
                'final_fmax': final_fmax,
                'runtime_sec': round(runtime, 1),
                'relaxed_file': str(relax_file)
            })

            log(f"  converged: {converged}, final energy: {final_energy:.4f} eV, "
                f"max force: {final_fmax:.4f} eV/A, time: {runtime:.1f} s")

        except Exception as e:
            runtime = time.time() - t_start
            log(f"  [error] {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'index': i + 1,
                'MLIP_ID': mlip_id,
                'Formula': formula,
                'Formula_realized': real_formula,
                'role': role,
                'S_total': s_tot,
                'pred_log10S': pred_log,
                'n_atoms': np.nan,
                'seed': seed,
                'converged': False,
                'final_energy': np.nan,
                'final_fmax': np.nan,
                'runtime_sec': round(runtime, 1),
                'relaxed_file': None,
                'error': str(e)
            })

             
    log("\n[5] Writing summary...")
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_DIR / 'relaxation_summary.csv', index=False)

    log("\n" + "=" * 60)
    log("Relaxation summary")
    log("=" * 60)
    n_success = int(df_results['converged'].sum())
    log(f"Converged: {n_success}/{len(df_results)}")

    if 'final_energy' in df_results.columns:
        valid_energies = df_results[df_results['final_energy'].notna()]
        if len(valid_energies) > 0:
            log(f"Energy range: {valid_energies['final_energy'].min():.2f} to "
                f"{valid_energies['final_energy'].max():.2f} eV")

    if 'runtime_sec' in df_results.columns:
        total_runtime = df_results['runtime_sec'].sum()
        log(f"Total time: {total_runtime:.1f} s ({total_runtime / 60:.1f} min)")
        log(f"Mean per candidate: {total_runtime / len(df_results):.1f} s")

    log(f"\nResults written to: {OUTPUT_DIR}")
    log("Generated files:")
    log("  - init_*.cif               : initial structures")
    log("  - relaxed_*.cif            : relaxed structures")
    log("  - relaxation_summary.csv   : summary table")
    log("  - relaxation.log           : run log")

    return df_results


if __name__ == "__main__":
    main()