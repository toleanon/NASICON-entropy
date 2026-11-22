"""Run MACE-MP-0 molecular dynamics and compute ionic conductivity."""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import numpy as np
import pandas as pd
import torch
import warnings
from ase.io import read
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io.trajectory import Trajectory
from mace.calculators import mace_mp
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.diffusion.analyzer import DiffusionAnalyzer

import sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "paths.py").exists())))
from paths import MLIP_RESULTS, MLIP_TRAJECTORIES, SCREENING_RESULTS

warnings.filterwarnings('ignore')

                                                  
RELAX_DIR = MLIP_RESULTS / "relaxation"
CANDIDATE_CSV = SCREENING_RESULTS / "MLIP_Candidates.csv"
OUTPUT_DIR = MLIP_RESULTS / "md"
TRAJ_DIR = MLIP_TRAJECTORIES

TEMPERATURES = [800, 900, 1000, 1100, 1200]
MD_STEPS = 100000                                       
TIMESTEP = 1.0                            
EQUILIBRATION_STEPS = 20000                     
SAMPLE_INTERVAL = 100                                             
N_REPEATS = 1
SUPERCELL_DIMS = (2, 2, 2)                                           
ONLY_MLIP_IDS = []                                                   

MACE_MODEL = "medium"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FLOAT64 = False

RANDOM_STATE = 42
T_ROOM = 300.0
KB_EV = 8.617333e-5
                                                     

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRAJ_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUTPUT_DIR / "md.log"


def log(msg, logfile=LOG_PATH):
    """Write a message to stdout and the log file."""
    print(msg)
    with open(logfile, 'a', encoding='utf-8') as f:
        f.write(msg + "\n")


def load_candidates(relax_dir, candidate_csv):
    """Merge relaxation results with candidates, sort by S_total, and drop unconverged entries."""
    relax_file = relax_dir / "relaxation_summary.csv"
    if not relax_file.exists():
        raise FileNotFoundError(f"Relaxation summary not found: {relax_file}")
    if not candidate_csv.exists():
        raise FileNotFoundError(f"Candidate table not found: {candidate_csv}")

    relax = pd.read_csv(relax_file).drop(columns=['S_total'], errors='ignore')
    cand = pd.read_csv(candidate_csv)
    cand['S_total'] = cand['S_M'] + cand['S_X']

                                    
    keep_cols = ['MLIP_ID', 'S_total', 'S_M', 'S_X']
    df = relax.merge(cand[keep_cols], on='MLIP_ID', how='inner')
    if len(df) == 0:
        raise ValueError("Relaxation results and candidates do not match; check MLIP_ID.")

    n_bad = int((~df['converged'].astype(bool)).sum())
    if n_bad:
        log(f"[warning] {n_bad} unconverged candidates skipped")
    df = df[df['converged'].astype(bool)].sort_values('S_total').reset_index(drop=True)

    if ONLY_MLIP_IDS:
        missing = [i for i in ONLY_MLIP_IDS if i not in set(df['MLIP_ID'])]
        if missing:
            log(f"[warning] whitelist entries not available: {missing}")
        df = df[df['MLIP_ID'].isin(ONLY_MLIP_IDS)].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"ONLY_MLIP_IDS={ONLY_MLIP_IDS} matched no candidates")
    return df


def build_supercell_from_relaxed(cif_path, supercell_dims=(2, 2, 2)):
    atoms = read(str(cif_path))
    struct = AseAtomsAdaptor.get_structure(atoms)
    super_struct = struct * supercell_dims
    return AseAtomsAdaptor.get_atoms(super_struct)


def run_nvt_md(atoms, temperature, calc, output_dir, label, rng):
    """Run production MD after equilibration; write one frame every SAMPLE_INTERVAL steps."""
    atoms.calc = calc
    traj_file = output_dir / f"{label}.traj"
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, rng=rng)
    dyn = Langevin(atoms, TIMESTEP * units.fs, temperature * units.kB,
                   friction=0.002, logfile=None, rng=rng)

    log(f"    equilibration: {EQUILIBRATION_STEPS} steps "
        f"({EQUILIBRATION_STEPS * TIMESTEP / 1000:.0f} ps)")
    for step in range(EQUILIBRATION_STEPS // SAMPLE_INTERVAL):
        dyn.run(SAMPLE_INTERVAL)
        if step % 50 == 0:
            log(f"      equilibration progress: {step * SAMPLE_INTERVAL}/{EQUILIBRATION_STEPS} steps")

    log(f"    production: {MD_STEPS} steps ({MD_STEPS * TIMESTEP / 1000:.0f} ps)")
    traj = Trajectory(str(traj_file), 'w', atoms)
    for step in range(MD_STEPS // SAMPLE_INTERVAL):
        dyn.run(SAMPLE_INTERVAL)
        traj.write(atoms)
        if step % 200 == 0:
            log(f"      production progress: {step * SAMPLE_INTERVAL}/{MD_STEPS} steps")
    traj.close()
    log(f"    trajectory written: {traj_file}")
    return traj_file


def compute_diffusion_conductivity(traj_file, temperature, species='Na'):
    """Compute D and sigma with DiffusionAnalyzer from the production frames."""
    try:
        traj = read(str(traj_file), index=':')
        n_frames = len(traj)
        if n_frames < 20:
            log(f"    too few frames: {n_frames} (< 20)")
            return None, None, None
        structures = [AseAtomsAdaptor.get_structure(a) for a in traj]
        analyzer = DiffusionAnalyzer.from_structures(
            structures, specie=species, temperature=temperature,
            time_step=SAMPLE_INTERVAL * TIMESTEP, step_skip=1,
            avg_nsteps=50,                                                        
            min_obs=20,                                                     
        )
        d = analyzer.get_summary_dict()
        D_avg = d['D']
        S_S_cm = d['S'] / 1000.0
        return D_avg, S_S_cm, analyzer.msd
    except Exception as e:
        log(f"    trajectory analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def fit_arrhenius(temperatures, diffusivities):
    T_arr = np.array(temperatures)
    D_arr = np.array(diffusivities)
    valid = (D_arr > 0) & np.isfinite(D_arr)
    if np.sum(valid) < 2:
        return None, None, 0.0
    T_v = T_arr[valid]; D_v = D_arr[valid]
    x = 1.0 / (T_v * KB_EV); y = np.log(D_v)
    coeffs = np.polyfit(x, y, 1)
    Ea = -coeffs[0]
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return Ea, np.exp(coeffs[1]), r2


def extrapolate_to_room_T(temps, D_values, Ea, T_room=300.0):
    if Ea is None:
        return np.nan
    log_ratio = -Ea / KB_EV * (1.0 / T_room - 1.0 / np.array(temps))
    return (np.array(D_values) * np.exp(log_ratio)).mean()


def sigma_from_diffusivity(D_cm2_s, struct, temperature):
    """Nernst-Einstein: sigma = n q^2 D / (kB T), using the actual Na count."""
    n_na = sum(1 for s in struct if s.specie.symbol == 'Na')
    n_density = n_na / (struct.volume * 1e-24)
    q = 1.602176634e-19
    kB_J = 1.380649e-23
    return n_density * q ** 2 * (D_cm2_s * 1e-4) / (kB_J * temperature)


def aggregate_runs(runs, df):
    """Aggregate repeats by candidate and temperature, then fit Arrhenius behavior."""
    ok = runs[runs['D'].notna() & (runs['D'] > 0)]
    if len(ok) == 0:
        log("[warning] no valid D values; skipping aggregation.")
        return pd.DataFrame(), pd.DataFrame()
    means = (ok.groupby(['MLIP_ID', 'T'])
             .agg(D_mean=('D', 'mean'), D_std=('D', 'std'),
                  sigma_mean=('sigma', 'mean'), n_runs=('D', 'size'))
             .reset_index())
    means['D_std'] = means['D_std'].fillna(0.0)

    rows = []
    for _, cand in df.iterrows():
        mlip_id = cand['MLIP_ID']
        sub = means[means['MLIP_ID'] == mlip_id].sort_values('T')
        Ea, R2, D_rt, sigma_rt, log10_rt = np.nan, np.nan, np.nan, np.nan, np.nan
        if len(sub) >= 2:
            Ea, D0, R2 = fit_arrhenius(sub['T'].to_numpy(), sub['D_mean'].to_numpy())
            if Ea is not None:
                                                    
                temps_arr = sub['T'].to_numpy()
                sig_arr = sub['sigma_mean'].to_numpy()
                log_ratio = -Ea / KB_EV * (1.0 / T_ROOM - 1.0 / temps_arr)
                sigma_rt = float((sig_arr * np.exp(log_ratio)).mean())
                log10_rt = float(np.log10(sigma_rt)) if sigma_rt > 0 else np.nan
        rows.append({
            'MLIP_ID': mlip_id,
            'index': cand.get('index', np.nan),
            'Formula': cand['Formula'],
            'S_total': cand['S_total'],
            'pred_log10S': cand['pred_log10S'],
            'Ea_eV': Ea,
            'R2': R2,
            'D_RT_cm2_s': D_rt,
            'sigma_RT_S_cm': sigma_rt,
            'MD_log10S_RT': log10_rt,
        })
    return means, pd.DataFrame(rows).sort_values('S_total').reset_index(drop=True)


def generate_report(summary, means, output_dir):
    report_path = output_dir / "MD_Simulation_Report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 90 + "\n")
        f.write("NASICON MLIP-MD simulation report (100 ps)\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"Temperatures: {TEMPERATURES} K\n")
        f.write(f"Production: {MD_STEPS * TIMESTEP / 1e3:.1f} ps; "
                f"equilibration: {EQUILIBRATION_STEPS * TIMESTEP / 1e3:.1f} ps\n")
        f.write(f"Repeats: {N_REPEATS}; supercell: {SUPERCELL_DIMS}; MACE model: {MACE_MODEL}\n\n")

        f.write("Diffusion coefficient D by temperature (mean +/- std, cm^2/s):\n" + "-" * 90 + "\n")
        if len(means) > 0:
            for _, r in means.iterrows():
                f.write(f"  {r['MLIP_ID']:<4} {r['T']:>6.0f} K  "
                        f"D = {r['D_mean']:.3e} ± {r['D_std']:.3e}  (n={int(r['n_runs'])})\n")

        f.write("\nArrhenius fits and room-temperature extrapolation:\n" + "-" * 90 + "\n")
        if len(summary) > 0:
            for _, r in summary.iterrows():
                ea = f"{r['Ea_eV']:.3f}" if pd.notna(r['Ea_eV']) else "N/A"
                r2 = f"{r['R2']:.4f}" if pd.notna(r['R2']) else "N/A"
                md = f"{r['MD_log10S_RT']:.3f}" if pd.notna(r['MD_log10S_RT']) else "N/A"
                f.write(f"  {r['MLIP_ID']:<4} {r['Formula']:<38} S_total={r['S_total']:5.2f}  "
                        f"pred={r['pred_log10S']:6.3f}  MD={md}  Ea={ea} eV  R2={r2}\n")

        f.write("\nReport generated: " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
    log(f"Report written: {report_path}")


def main():
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    # Run the temperature series for each relaxed candidate.
    log("=" * 80)
    log("NASICON MLIP-MD simulation (100 ps)")
    # Run the temperature series for each relaxed candidate.
    log("=" * 80)

    log(f"PyTorch version: {torch.__version__}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"Device: {DEVICE}")
    log(f"Precision: {'float64' if USE_FLOAT64 else 'float32'}")
    log(f"Temperatures: {TEMPERATURES} K")
    log(f"Production: {MD_STEPS} steps = {MD_STEPS * TIMESTEP / 1e3:.0f} ps")
    log(f"Equilibration: {EQUILIBRATION_STEPS} steps = {EQUILIBRATION_STEPS * TIMESTEP / 1e3:.0f} ps")
    log(f"Sampling interval: {SAMPLE_INTERVAL} steps = {SAMPLE_INTERVAL * TIMESTEP / 1e3:.1f} ps")
    log(f"Repeats: {N_REPEATS}; supercell: {SUPERCELL_DIMS}")
    log(f"Candidate whitelist: {ONLY_MLIP_IDS if ONLY_MLIP_IDS else 'all'}")

    df = load_candidates(RELAX_DIR, CANDIDATE_CSV)
    log(f"\nIncluded {len(df)} candidates (sorted by S_total):")
    for _, row in df.iterrows():
        log(f"  {row['MLIP_ID']:<4} {row['Formula']:<38} "
            f"S_total={row['S_total']:5.2f}  pred={row['pred_log10S']:6.3f}")

    log("\nInitializing MACE-MP...")
    calc = mace_mp(model=MACE_MODEL, device=DEVICE,
                   default_dtype="float64" if USE_FLOAT64 else "float32")
    log(f"  {MACE_MODEL} on {DEVICE}")

    total_runs = len(df) * len(TEMPERATURES) * N_REPEATS
    log(f"\nPlanned MD runs: {len(df)} candidates x {len(TEMPERATURES)} temperatures "
        f"x {N_REPEATS} repeats = {total_runs}")

    run_rows = []
    t_global = time.time()

    for _, cand in df.iterrows():
        mlip_id = cand['MLIP_ID']
        idx = int(cand.get('index', 0))
        relax_file = RELAX_DIR / f"relaxed_{mlip_id}.cif"
        if not relax_file.exists():
            log(f"\n--- {mlip_id}: missing {relax_file}; skipped ---")
            continue

        log(f"\n--- {mlip_id}: {cand['Formula']} "
            f"(S_total={cand['S_total']:.2f}, pred={cand['pred_log10S']:.3f}) ---")
        for temperature in TEMPERATURES:
            for repeat in range(N_REPEATS):
                label = f"{mlip_id}_T{temperature}K_r{repeat}"
                seed = RANDOM_STATE + 1000 * idx + 10 * temperature + repeat
                rng = np.random.default_rng(seed)
                log(f"  {temperature} K repeat {repeat + 1}/{N_REPEATS} (seed={seed})")
                t0 = time.time()
                try:
                    atoms = build_supercell_from_relaxed(relax_file, SUPERCELL_DIMS)
                    if repeat == 0:
                        log(f"    supercell atoms: {len(atoms)}")
                    traj_file = run_nvt_md(atoms, temperature, calc, TRAJ_DIR, label, rng)
                    D, sigma, msd = compute_diffusion_conductivity(traj_file, temperature)
                    runtime = time.time() - t0
                    if D is not None and D > 0:
                        log(f"    OK: D={D:.3e} cm2/s, sigma={sigma:.3e} S/cm, "
                            f"time {runtime / 60:.1f} min")
                    else:
                        log(f"    trajectory analysis failed; time {runtime / 60:.1f} min")
                    run_rows.append({
                        'MLIP_ID': mlip_id, 'index': idx, 'Formula': cand['Formula'],
                        'S_total': cand['S_total'], 'T': temperature, 'repeat': repeat,
                        'seed': seed, 'n_atoms': len(atoms),
                        'D': D, 'sigma': sigma, 'traj_file': str(traj_file),
                        'runtime_sec': round(runtime, 1),
                    })
                except Exception as e:
                    runtime = time.time() - t0
                    log(f"    {temperature} K failed: {e}; time {runtime / 60:.1f} min")
                    run_rows.append({
                        'MLIP_ID': mlip_id, 'index': idx, 'Formula': cand['Formula'],
                        'S_total': cand['S_total'], 'T': temperature, 'repeat': repeat,
                        'seed': seed, 'n_atoms': np.nan,
                        'D': np.nan, 'sigma': np.nan, 'traj_file': None,
                        'runtime_sec': round(runtime, 1),
                        'error': str(e),
                    })

    runs = pd.DataFrame(run_rows)
    runs.to_csv(OUTPUT_DIR / 'md_runs.csv', index=False)
    log(f"\nRun table written: {OUTPUT_DIR / 'md_runs.csv'}")
    log(f"Total time: {(time.time() - t_global) / 3600:.2f} h")

    means, summary = aggregate_runs(runs, df)
    if len(means) > 0:
        means.to_csv(OUTPUT_DIR / 'md_means.csv', index=False)
    if len(summary) > 0:
        summary.to_csv(OUTPUT_DIR / 'md_summary.csv', index=False)
    generate_report(summary, means, OUTPUT_DIR)

    log("\n" + "=" * 80)
    log("Done. Output directory: " + str(OUTPUT_DIR))
    # Run the temperature series for each relaxed candidate.
    log("=" * 80)
    return summary


if __name__ == "__main__":
    main()
