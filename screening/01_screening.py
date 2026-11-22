"""High-throughput screening: generation, scoring, and stratified selection."""

import random
import re
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from pymatgen.io.cif import CifParser
from glob import glob
from scipy.spatial.distance import cdist
import joblib
import warnings

import sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "paths.py").exists())))
from paths import DATA_DIR, MODEL_DIR, SCREENING_DIR, SCREENING_RESULTS

warnings.filterwarnings('ignore')

                                                             
DATA_PATH = DATA_DIR / "nasicon_cleaned.csv"
SEED_CSV_PATH = SCREENING_DIR / "seeds" / "seed_candidates.csv"
SEED_CIF_DIR = SCREENING_DIR / "seeds" / "cif_candidates"
MODEL_PATH = MODEL_DIR / "figures" / "best_model_7d.pkl"
SCALER_PATH = MODEL_DIR / "processed" / "scaler_7d.pkl"
FEATURE_COLS = ['Na_ratio', 'Δr_MX', 'Δχ_XM', 'Sinter_intensity', 'struct', 'S_M', 'S_X']

OUTPUT_DIR = SCREENING_RESULTS
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

                                                               
NUM_CANDIDATES = 5000
MUTATION_PROB = 0.4
FIXED_SINTER_TEMP = 1050
FIXED_SINTER_TIME = 12

WEIGHT_PERF = 0.30
WEIGHT_SYNTH = 0.30
WEIGHT_ENTROPY_COV = 0.25
WEIGHT_NOVEL = 0.15
WEIGHT_RADIUS = 0.4
WEIGHT_VALENCE = 0.3
WEIGHT_ENTROPY_STAB = 0.3

ENTROPY_BINS = 5
MAX_PER_DOPANT_PER_BIN = 1
MAX_PER_DOPANT_TOTAL = 2
STRATIFIED_PER_BIN = 4
STRATIFIED_PER_DOPANT_BIN = 2
STRATIFIED_PER_DOPANT_TOTAL = 3
PERF_THRESHOLD = -3.5
NOVELTY_THRESHOLD_FACTOR = 1.0
MIN_ZR_FRACTION = 0.25

MAX_M_DOPANT = 2.0
MIN_NA = 0.5                                      
MAX_NA = 4.5
CHARGE_BALANCE_TOL = 0.4                      

SIP_STEP = 0.1
SIP_MIN = 0.5
SIP_MAX = 2.5
NA_STEP = 0.05
SEED_SAMPLING_RATIO = 0.3

                                                         
NA_DEC = 4                                
MIN_NA_CORRECTED = 1.0                 
MAX_NA_CORRECTED = 4.0                                 
TARGET_TOP = 30

                                                            
RADIUS = {
    'Zr':72, 'Ti':60.5, 'Hf':71, 'Sn':69, 'Ge':53, 'V':64, 'Fe':64.5,
    'Al':53.5, 'Sc':74.5, 'La':103.2, 'Y':90, 'Yb':86.8, 'Mg':72,
    'Ca':100, 'Sr':118, 'Ba':135, 'Cu':73, 'Nb':68, 'Mo':65, 'In':80,
    'Ga':62, 'Cr':61.5, 'Mn':83, 'Ni':60, 'Si':40, 'P':38,
    'Na':102, 'O':140, 'Co':74, 'W':74,
    'Lu':86.1, 'Pr':99.0, 'Eu':94.7, 'Ce':101.0, 'K':138.0, 'Zn':74.0,
}
NEG = {
    'Zr':1.33, 'Ti':1.54, 'Hf':1.32, 'Sn':1.96, 'Ge':2.01, 'V':1.63,
    'Fe':1.83, 'Al':1.61, 'Sc':1.36, 'La':1.11, 'Y':1.22, 'Yb':1.26,
    'Mg':1.31, 'Ca':1.0, 'Sr':0.95, 'Ba':0.89, 'Cu':1.9, 'Nb':1.59,
    'Mo':2.16, 'In':1.78, 'Ga':1.81, 'Cr':1.66, 'Mn':1.55, 'Ni':1.92,
    'Si':1.98, 'P':2.19, 'Na':0.93, 'O':3.44, 'Co':1.88, 'W':2.36,
    'Lu':1.27, 'Pr':1.13, 'Eu':1.20, 'Ce':1.12, 'K':0.82, 'Zn':1.65,
}
M_REPLACEMENTS = {
    'Zr': ['Ti', 'Hf', 'Sn', 'Ge'], 'Ti': ['Zr', 'Hf', 'Sn'],
    'Hf': ['Zr', 'Ti', 'Sn'], 'Sn': ['Zr', 'Ge'], 'Ge': ['Sn', 'Zr'],
    'Al': ['Ga', 'Sc', 'In'], 'Ga': ['Al', 'Sc', 'In'],
    'Sc': ['Al', 'Ga', 'Y', 'Yb'], 'Y': ['Sc', 'Yb', 'La'],
    'Yb': ['Y', 'Sc', 'La'], 'Mg': ['Ca', 'Sr', 'Ba', 'Zn'],
    'Ca': ['Mg', 'Sr', 'Ba'], 'Sr': ['Ca', 'Ba', 'Mg'],
    'Ba': ['Sr', 'Ca', 'Mg'], 'V': ['Nb', 'Mo', 'W'],
    'Nb': ['V', 'Mo', 'W'], 'Mo': ['Nb', 'V', 'W'],
    'W': ['Mo', 'V', 'Nb'], 'Fe': ['Mn', 'Cr', 'Ni', 'Co'],
    'Mn': ['Fe', 'Cr', 'Ni', 'Co'], 'Cr': ['Fe', 'Mn', 'Ni', 'Co'],
    'Ni': ['Fe', 'Mn', 'Cr', 'Co'], 'Co': ['Fe', 'Mn', 'Ni', 'Cr'],
    'Cu': ['Zn', 'Mg'], 'Zn': ['Cu', 'Mg'],
    'In': ['Ga', 'Al', 'Sc'], 'La': ['Y', 'Yb', 'Sc'],
}

VALENCE_MAP = {
    'Na':1, 'Zr':4, 'Ti':4, 'Hf':4, 'Sn':4, 'Ge':4, 'V':3, 'Fe':3,
    'Al':3, 'Sc':3, 'La':3, 'Y':3, 'Yb':3, 'Mg':2, 'Ca':2, 'Sr':2,
    'Ba':2, 'Cu':2, 'Nb':5, 'Mo':6, 'In':3, 'Ga':3, 'Cr':3, 'Mn':2,
    'Ni':2, 'Si':4, 'P':5, 'O':-2, 'Co':3, 'W':6,
    'Lu':3, 'Pr':3, 'Eu':3, 'Ce':3, 'K':1, 'Zn':2,
}

                                                  
def load_seeds_from_csv(csv_path):
    if not csv_path.exists(): return set()
    df = pd.read_csv(csv_path)
    if 'Formula' not in df.columns: return set()
    return set(df['Formula'].dropna().unique())

def load_seeds_from_cif_dir(cif_dir):
    if not cif_dir.exists(): return set()
    formulas = set()
    for cif_file in glob(str(cif_dir / "*.cif")):
        try:
            structures = CifParser(cif_file).get_structures()
            if structures:
                formulas.add(structures[0].composition.formula)
        except Exception as e:
            print(f"Failed to read {cif_file}: {e}")
    return formulas

                                                           
def build_formula(comp_dict):
    """Generate neutral formulas with two-decimal Na."""
    comp_dict['O'] = 12
    metals = [el for el in comp_dict if el not in ['Na','O','Si','P']]
    tetra_present = [el for el in ['Si','P'] if el in comp_dict]
    order = ['Na'] + sorted(metals) + tetra_present + ['O']
    parts = []
    for el in order:
        amt = comp_dict.get(el, 0)
        if abs(amt) < 1e-6: continue
        if abs(amt - round(amt)) < 1e-4:
            parts.append(f"{el}{int(round(amt))}")
        else:
            s = f"{amt:.2f}".rstrip('0').rstrip('.')
            parts.append(f"{el}{s}")
    return "".join(parts)

def build_formula_neutral(comp_dict):
    """Corrected formulas keep NA_DEC decimals for Na."""
    comp = dict(comp_dict); comp['O'] = 12
    metals = [el for el in comp if el not in ['Na','O','Si','P']]
    tetra_present = [el for el in ['Si','P'] if el in comp]
    order = ['Na'] + sorted(metals) + tetra_present + ['O']
    parts = []
    for el in order:
        amt = comp.get(el, 0)
        if abs(amt) < 1e-6: continue
        if el == 'Na':
            s = f"{amt:.{NA_DEC}f}".rstrip('0').rstrip('.')
            parts.append(f"Na{s}")
        else:
            if abs(amt - round(amt)) < 1e-4:
                parts.append(f"{el}{int(round(amt))}")
            else:
                s = f"{amt:.2f}".rstrip('0').rstrip('.')
                parts.append(f"{el}{s}")
    return "".join(parts)

def standardize_formula(formula):
    comp = Composition(formula)
    comp_dict = {el.symbol: comp[el] for el in comp.elements}
    return build_formula(comp_dict)

                                                    
def parse_formula(formula):
    comp = Composition(formula)
    return {el.symbol: comp[el] for el in comp.elements}

def charge_balance(comp_dict, tolerance=CHARGE_BALANCE_TOL):
    total = sum(VALENCE_MAP.get(el, 0) * a for el, a in comp_dict.items())
    return abs(total) < tolerance

def check_chemical_plausibility(comp_dict):
    m_dopants = [el for el in comp_dict if el not in ['Na','Zr','Si','P','O']]
    total_dopant = sum(comp_dict[el] for el in m_dopants)
    if total_dopant > MAX_M_DOPANT: return False
    na = comp_dict.get('Na', 0)
    if na < MIN_NA or na > MAX_NA: return False
    n_Zr = comp_dict.get('Zr', 0)
    m_total = n_Zr + total_dopant
    if m_total > 0 and n_Zr / m_total < MIN_ZR_FRACTION: return False
    return True

def get_skeleton_strong(formula):
    return re.sub(r'Na[\d.]+', 'NaX', formula)

                                                           
def mutate_single_element(formula):
    comp = parse_formula(formula)
    replaceable = [el for el in comp if el not in ['Na','O','Si','P']]
    if not replaceable: return None
    target = random.choice(replaceable)
    if target not in M_REPLACEMENTS: return None
    new_el = random.choice(M_REPLACEMENTS[target])
    amount = comp.pop(target)
    comp[new_el] = comp.get(new_el, 0) + amount
    if not charge_balance(comp) or not check_chemical_plausibility(comp):
        return None
    return build_formula(comp)

def mutate_sip_ratio(formula, si_range=(SIP_MIN, SIP_MAX), step=SIP_STEP):
    comp = parse_formula(formula)
    si = comp.get('Si', 0); p = comp.get('P', 0)
    total = si + p
    if total == 0: return None
    si_candidates = np.arange(si_range[0], si_range[1] + 1e-6, step)
    new_si = round(float(random.choice(si_candidates)), 2)
    new_p = round(total - new_si, 2)
    if new_p < 0.1 or new_si < 0.1: return None
    comp['Si'] = new_si; comp['P'] = new_p
    if not charge_balance(comp) or not check_chemical_plausibility(comp):
        return None
    return build_formula(comp)

def mutate_na_content(formula):
    comp = parse_formula(formula)
    na = comp.get('Na', 0)
    delta = random.choice([-0.20, -0.15, -0.10, -0.05, 0.05, 0.10, 0.15, 0.20])
    new_na = round((na + delta) / NA_STEP) * NA_STEP
    if new_na < MIN_NA or new_na > MAX_NA: return None
    comp['Na'] = new_na
    if not charge_balance(comp) or not check_chemical_plausibility(comp):
        return None
    return build_formula(comp)

                                                           
def compute_my_features(comp_dict):
    n0 = comp_dict.get('Na', 0)
    m_elements = [el for el in comp_dict if el not in ['Na','Zr','Si','P','O']]
    m_els = ['Zr'] + m_elements
    m_amts = [comp_dict.get('Zr', 0)] + [comp_dict[el] for el in m_elements]
    n_M = sum(m_amts)
    n4 = comp_dict.get('Si', 0); n5 = comp_dict.get('P', 0)
    n_X = n4 + n5
    if n_M == 0 or n_X == 0: return None

    def avg(amounts, elements, prop_dict):
        total = sum(amounts)
        if total == 0: return 0
        missing = [el for el in elements if el not in prop_dict]
        if missing: raise KeyError(f"[V4.7] Missing properties for: {missing}")
        return sum(a * prop_dict[el] for a, el in zip(amounts, elements)) / total

    r_M = avg(m_amts, m_els, RADIUS)
    neg_M = avg(m_amts, m_els, NEG)
    r_X = avg([n4, n5], ['Si', 'P'], RADIUS)
    neg_X = avg([n4, n5], ['Si', 'P'], NEG)

    na_ratio = n0 / n_M
    dr = r_M - r_X
    dchi = neg_X - neg_M
    sintering_intensity = FIXED_SINTER_TEMP * np.log10(FIXED_SINTER_TIME + 1)
    struct = 1

    R_gas = 8.314
    def shannon_entropy(amounts):
        total = sum(amounts)
        if total == 0: return 0
        return -R_gas * sum((a/total)*np.log(a/total) for a in amounts if a > 0)

    s_M = shannon_entropy(m_amts)
    s_X = shannon_entropy([n4, n5])

    return {'Na_ratio': na_ratio, 'Δr_MX': dr, 'Δχ_XM': dchi,
            'Sinter_intensity': sintering_intensity, 'struct': struct,
            'S_M': s_M, 'S_X': s_X}

                                                           
def generate_candidates(seeds, target_num):
    candidates = set()
    candidates_list = []
    skeleton_set = set()

    for s in seeds:
        comp_s = parse_formula(s)
        if check_chemical_plausibility(comp_s) and charge_balance(comp_s):
            candidates.add(s)
            candidates_list.append(s)
            skeleton_set.add(get_skeleton_strong(s))
    print(f"Seeds passing Zr and charge-balance checks: {len(candidates)}/{len(seeds)}")

    max_attempts = target_num * 30
    attempt = 0
    print(f"Generating candidates; target {target_num} (exploratory tolerance {CHARGE_BALANCE_TOL})...")
    while len(candidates) < target_num and attempt < max_attempts:
        attempt += 1
        if attempt % 5000 == 0:
            print(f"Attempts: {attempt}; candidates: {len(candidates)}; skeletons: {len(skeleton_set)}")

        if random.random() < SEED_SAMPLING_RATIO and candidates_list:
            parent_pool = [
                s for s in seeds
                if check_chemical_plausibility(parse_formula(s)) and charge_balance(parse_formula(s))
            ]
            parent = random.choice(parent_pool) if parent_pool else random.choice(candidates_list)
        else:
            parent = random.choice(candidates_list)

        r = random.random()
        if r < MUTATION_PROB:
            new = mutate_single_element(parent)
        elif r < MUTATION_PROB + 0.3:
            new = mutate_sip_ratio(parent)
        else:
            new = mutate_na_content(parent)

        if new and new not in candidates:
            sk = get_skeleton_strong(new)
            if sk not in skeleton_set:
                candidates.add(new)
                candidates_list.append(new)
                skeleton_set.add(sk)

    if len(candidates) < target_num:
        print(f"[warning] Only {len(candidates)} candidates generated; target not reached.")
    print(f"Generation complete: {len(candidates)} candidates")
    return list(candidates)

                                                            
def project_to_neutral(formula):
    """Project a formula to exact formal charge neutrality."""
    comp = parse_formula(formula)
    q_orig = sum(VALENCE_MAP.get(el, 0) * a for el, a in comp.items())
    na_orig = comp.get('Na', 0)
    na_ideal = -(q_orig - na_orig)

    if na_ideal < MIN_NA_CORRECTED or na_ideal > MAX_NA_CORRECTED:
        return None

    na_new = round(na_ideal, NA_DEC)
    if na_new > MAX_NA_CORRECTED or na_new < MIN_NA_CORRECTED:
        return None

    comp['Na'] = na_new
    f_new = build_formula_neutral(comp)
    comp_check = parse_formula(f_new)
    q_check = sum(VALENCE_MAP.get(el, 0) * a for el, a in comp_check.items())
    if abs(q_check) > 1e-3:
        return None
    return f_new, round(na_new - na_orig, 4), round(q_orig, 4)

def na_correct_and_dedup(candidates):
    """Correct, deduplicate, and report discarded candidates."""
    records = []
    dropped = {'na_too_high': 0, 'na_too_low': 0, 'q_residual': 0, 'parse_error': 0}

    for f_orig in candidates:
        try:
            comp = parse_formula(f_orig)
        except Exception:
            dropped['parse_error'] += 1
            continue

        q_orig = sum(VALENCE_MAP.get(el, 0) * a for el, a in comp.items())
        na_orig = comp.get('Na', 0)
        na_ideal = -(q_orig - na_orig)

        if na_ideal > MAX_NA_CORRECTED:
            dropped['na_too_high'] += 1
            continue
        if na_ideal < MIN_NA_CORRECTED:
            dropped['na_too_low'] += 1
            continue

        res = project_to_neutral(f_orig)
        if res is None:
            dropped['q_residual'] += 1
            continue

        f_new, dNa, dq_orig = res
        records.append({'Formula': f_new, 'Formula_orig': f_orig,
                        'dNa': dNa, 'abs_dNa': abs(dNa), 'dq_orig': dq_orig})

    if not records:
        raise ValueError("No valid candidates after correction")

    df = pd.DataFrame(records)
    counts = df.groupby('Formula').size().reset_index(name='merged_from_count')
    df = df.sort_values('abs_dNa').drop_duplicates(subset='Formula', keep='first')
    df = df.merge(counts, on='Formula', how='left').reset_index(drop=True)
    return df['Formula'].tolist(), df, dropped

def print_correction_stats(df_info, dropped):
    n = len(df_info)
    total_in = int(df_info['merged_from_count'].sum()) + sum(dropped.values())
    print(f"  Input candidates: {total_in}")
    print(f"  Dropped for Na above upper bound (>{MAX_NA_CORRECTED}): {dropped['na_too_high']}")
    print(f"  Dropped for Na below lower bound (<{MIN_NA_CORRECTED}): {dropped['na_too_low']}")
    print(f"  Dropped for residual charge: {dropped['q_residual']}")
    print(f"  Dropped for parse errors: {dropped['parse_error']}")
    print(f"  Unique neutral candidates after correction: {n}")
    print(f"  Merged Na variants: {int(df_info['merged_from_count'].sum()) - n}")
    print(f"  Original |dq| distribution:")
    for thr in [1e-3, 0.05, 0.10, 0.20, 0.40]:
        c = (df_info['dq_orig'].abs() < thr).sum()
        print(f"    < {thr:.3f}: {c} ({c/n*100:.1f}%)")
    print(f"  |dNa| distribution:")
    for thr in [0.05, 0.10, 0.20, 0.30]:
        c = (df_info['abs_dNa'] < thr).sum()
        print(f"    < {thr:.2f}: {c} ({c/n*100:.1f}%)")

                                                               
R_GAS = 8.314

def compute_radius_deviation(comp):
    n_Zr = comp.get('Zr', 0)
    m_elements = [el for el in comp if el not in ['Na','Zr','Si','P','O']]
    if not m_elements: return 0.0
    total_m = n_Zr + sum(comp.get(el, 0) for el in m_elements)
    if total_m == 0: return 0.0
    r_Zr = RADIUS['Zr']
    dev = 0.0
    for el in m_elements:
        if el not in RADIUS: raise KeyError(f"[V4.7] Missing radius for {el}")
        dev += abs(RADIUS[el] - r_Zr) * comp.get(el, 0)
    return dev / total_m

def compute_valence_std(comp):
    m_elements = ['Zr'] + [el for el in comp if el not in ['Na','Zr','Si','P','O']]
    vals = []
    for el in m_elements:
        amt = comp.get(el, 0)
        if amt <= 0: continue
        if el not in VALENCE_MAP: raise KeyError(f"[V4.7] Missing valence for {el}")
        v = VALENCE_MAP[el]
        n_repeat = max(1, int(round(amt * 10)))
        vals.extend([v] * n_repeat)
    if len(vals) < 2: return 0.0
    return float(np.std(vals))

                                                 
def main():
    random.seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("NASICON high-throughput screening: generation, neutral correction, and Top30")
    print("=" * 60)

                       
    df_raw = pd.read_csv(DATA_PATH)
    main_seeds = set(df_raw['Formula'].dropna().unique())
    print(f"Main-table seeds: {len(main_seeds)}")

    extra_seeds = set()
    if SEED_CSV_PATH:
        csv_seeds = load_seeds_from_csv(SEED_CSV_PATH)
        extra_seeds.update(csv_seeds)
        print(f"CSV seeds: {len(csv_seeds)}")
    if SEED_CIF_DIR:
        cif_seeds = load_seeds_from_cif_dir(SEED_CIF_DIR)
        extra_seeds.update(cif_seeds)
        print(f"CIF seeds: {len(cif_seeds)}")

    all_seeds = set()
    for f in main_seeds: all_seeds.add(standardize_formula(f))
    for f in extra_seeds: all_seeds.add(standardize_formula(f))
    seeds = list(all_seeds)
    print(f"Unique seed formulas: {len(seeds)}")

                       
    print("\nLoading model...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_cols = FEATURE_COLS

                                  
    print("\n--- Generating candidate pool (exploratory tolerance 0.4) ---")
    # Generate and neutralize candidates.
    candidates_raw = generate_candidates(seeds, NUM_CANDIDATES)
    print(f"Raw candidates: {len(candidates_raw)}")

                            
    print("\n--- Neutral Na correction and deduplication ---")
    candidates, df_correction_info, dropped = na_correct_and_dedup(candidates_raw)
    print_correction_stats(df_correction_info, dropped)
    print(f"Candidates after correction: {len(candidates)}")
    if len(candidates) < TARGET_TOP:
        print(f"[warning] Corrected candidates {len(candidates)} < {TARGET_TOP}; Top{TARGET_TOP} may be incomplete")

                               
    print("\n--- Feature calculation (corrected neutral compositions) ---")
    features_list = []
    valid_formulas = []
    for formula in candidates:
        comp = parse_formula(formula)
        feats = compute_my_features(comp)
        if feats is None: continue
        if any(np.isnan(v) or np.isinf(v) for v in feats.values()): continue
        features_list.append([feats[f] for f in feature_cols])
        valid_formulas.append(formula)
    print(f"Valid candidates after correction: {len(valid_formulas)}")
    if len(valid_formulas) == 0:
        raise ValueError("No valid candidates; check seeds and generation rules.")

                     
    X = np.array(features_list)
    X_scaled = scaler.transform(X)
    pred_log10S = model.predict(X_scaled)

    df_results = pd.DataFrame({
        'Formula': valid_formulas,
        'pred_log10S': pred_log10S,
        'pred_S': 10 ** pred_log10S,
    })
    for i, col in enumerate(feature_cols):
        df_results[col] = [f[i] for f in features_list]

    df_results = df_results.merge(
        df_correction_info[['Formula', 'Formula_orig', 'dNa', 'abs_dNa',
                             'dq_orig', 'merged_from_count']],
        on='Formula', how='left'
    )

                        
    df_results['Skeleton'] = df_results['Formula'].apply(get_skeleton_strong)
    df_unique = df_results.loc[df_results.groupby('Skeleton')['pred_S'].idxmax()].copy()
    df_unique = df_unique.sort_values('pred_S', ascending=False).reset_index(drop=True)
    df_results = df_unique
    print(f"Candidates after skeleton deduplication: {len(df_results)}")

                        
    train_feat_path = MODEL_DIR / "processed" / "X_train_7d.csv"
    if train_feat_path.exists():
        X_train_full = pd.read_csv(train_feat_path)
        X_candidates_scaled = scaler.transform(df_results[feature_cols].values)
        X_train_scaled = X_train_full[feature_cols].values
        distances = cdist(X_candidates_scaled, X_train_scaled, metric='euclidean')
        df_results['Min_Distance_to_Train'] = distances.min(axis=1)
        train_dist_matrix = cdist(X_train_scaled, X_train_scaled, metric='euclidean')
        np.fill_diagonal(train_dist_matrix, np.inf)
        train_internal_dist = np.mean(train_dist_matrix.min(axis=1))
        novelty_threshold = NOVELTY_THRESHOLD_FACTOR * train_internal_dist
        print(f"Mean nearest-neighbor distance in training: {train_internal_dist:.3f}; novelty threshold: {novelty_threshold:.3f}")
    else:
        print("[warning] Training features not found; skipping novelty")
        df_results['Min_Distance_to_Train'] = 0.0
        novelty_threshold = 0.0

                        
    print("\n--- Multi-objective scoring ---")

    def get_dopant(formula):
        comp = parse_formula(formula)
        dopants = [el for el in comp if el not in ['Na','Zr','Si','P','O']]
        return dopants[0] if dopants else 'None'
    df_results['dopant'] = df_results['Formula'].apply(get_dopant)

    # Score candidates for the multi-objective screen.
    df_results['rank_perf'] = df_results['pred_S'].rank(pct=True)
    df_results['radius_dev'] = df_results['Formula'].apply(
        lambda f: compute_radius_deviation(parse_formula(f)))
    df_results['score_radius'] = np.exp(-(df_results['radius_dev'] / 20.0) ** 2)
    df_results['valence_std'] = df_results['Formula'].apply(
        lambda f: compute_valence_std(parse_formula(f)))
    df_results['score_valence'] = np.exp(-(df_results['valence_std'] ** 2) / 0.5)

    total_entropy = df_results['S_M'] + df_results['S_X']
    df_results['score_entropy_stab'] = np.minimum(1.0, total_entropy / (2 * R_GAS))
    df_results['score_synthesizability'] = (
        WEIGHT_RADIUS * df_results['score_radius'] +
        WEIGHT_VALENCE * df_results['score_valence'] +
        WEIGHT_ENTROPY_STAB * df_results['score_entropy_stab']
    )

    try:
        entropy_bins = pd.qcut(total_entropy, q=ENTROPY_BINS, labels=False, duplicates='drop')
        bin_counts = pd.Series(entropy_bins).value_counts()
        max_bin_count = bin_counts.max()
        df_results['score_entropy_coverage'] = pd.Series(entropy_bins).apply(
            lambda x: 1 - (bin_counts[x] / max_bin_count) if max_bin_count > 0 else 0.5
        ).values
    except Exception as e:
        print(f"Entropy binning failed: {e}; using uniform bins")
        df_results['score_entropy_coverage'] = 0.5

    dist_95 = df_results['Min_Distance_to_Train'].quantile(0.95)
    df_results['distance_capped'] = np.minimum(df_results['Min_Distance_to_Train'], dist_95)
    df_results['rank_novel'] = df_results['distance_capped'].rank(pct=True)

    df_results['composite'] = (
        WEIGHT_PERF * df_results['rank_perf'] +
        WEIGHT_SYNTH * df_results['score_synthesizability'] +
        WEIGHT_ENTROPY_COV * df_results['score_entropy_coverage'] +
        WEIGHT_NOVEL * df_results['rank_novel']
    )

    df_results = df_results.sort_values('composite', ascending=False).reset_index(drop=True)
    df_all_candidates = df_results.copy()
    df_all_candidates.to_csv(OUTPUT_DIR / 'full_candidates.csv', index=False)
    print(f"Full candidate table written: {OUTPUT_DIR / 'full_candidates.csv'}")

                         
    if novelty_threshold > 0:
        before = len(df_results)
        df_results = df_results[df_results['Min_Distance_to_Train'] >= novelty_threshold]
        print(f"Novelty filter: removed {before - len(df_results)}; remaining {len(df_results)}")
    else:
        print("Novelty threshold not set; skipping filter.")
    if len(df_results) == 0:
        print("[warning] No candidates after filtering; novelty filter disabled.")
        df_results = df_all_candidates.copy()

                                
    print(f"\n--- Stratified Top{TARGET_TOP} selection ---")
    df_results['entropy_bin'] = pd.qcut(
        df_results['S_M'] + df_results['S_X'],
        q=ENTROPY_BINS, labels=False, duplicates='drop'
    )

    target_per_bin = TARGET_TOP // ENTROPY_BINS
    final_list = []
    seen_dopants_total = {}

         
    for bin_id in sorted(df_results['entropy_bin'].unique()):
        bin_df = df_results[df_results['entropy_bin'] == bin_id].sort_values(
            'composite', ascending=False)
        bin_df = bin_df[bin_df['pred_log10S'] >= PERF_THRESHOLD]
        if len(bin_df) == 0:
            print(f"[warning] Entropy bin {bin_id} has no eligible candidates; skipped")
            continue
        seen_dopants_bin = {}
        selected_in_bin = 0
        for _, row in bin_df.iterrows():
            d = row['dopant']
            if (seen_dopants_bin.get(d, 0) < MAX_PER_DOPANT_PER_BIN
                    and seen_dopants_total.get(d, 0) < MAX_PER_DOPANT_TOTAL):
                final_list.append(row)
                seen_dopants_bin[d] = seen_dopants_bin.get(d, 0) + 1
                seen_dopants_total[d] = seen_dopants_total.get(d, 0) + 1
                selected_in_bin += 1
            if selected_in_bin >= target_per_bin:
                break

           
    df_final = pd.DataFrame(final_list).drop_duplicates(subset=['Formula']) if final_list else pd.DataFrame()
    if len(df_final) < TARGET_TOP:
        remaining = df_results[~df_results['Formula'].isin(df_final['Formula'] if len(df_final) > 0 else [])]
        remaining = remaining[remaining['pred_log10S'] >= PERF_THRESHOLD]
        new_elements = remaining[~remaining['dopant'].isin(seen_dopants_total.keys())]
        new_elements = new_elements.sort_values('composite', ascending=False)
        for _, row in new_elements.iterrows():
            if len(df_final) >= TARGET_TOP: break
            df_final = pd.concat([df_final, row.to_frame().T]) if len(df_final) > 0 else row.to_frame().T
            seen_dopants_total[row['dopant']] = 1

        if len(df_final) < TARGET_TOP:
            remaining2 = remaining[~remaining['Formula'].isin(df_final['Formula'])]
            remaining2 = remaining2.sort_values('composite', ascending=False)
            for _, row in remaining2.iterrows():
                d = row['dopant']
                if seen_dopants_total.get(d, 0) < MAX_PER_DOPANT_TOTAL:
                    df_final = pd.concat([df_final, row.to_frame().T])
                    seen_dopants_total[d] = seen_dopants_total.get(d, 0) + 1
                if len(df_final) >= TARGET_TOP: break

    df_final = df_final.sort_values('composite', ascending=False).head(TARGET_TOP).reset_index(drop=True)
    top30 = df_final
    top20 = top30.head(20).reset_index(drop=True)
    top8 = top30.head(8).reset_index(drop=True)

    top30.to_csv(OUTPUT_DIR / 'Top30_Candidates.csv', index=False)
    top20.to_csv(OUTPUT_DIR / 'Top20_Candidates.csv', index=False)
    top8.to_csv(OUTPUT_DIR / 'Top8_Candidates.csv', index=False)

    display_cols = ['Formula', 'Formula_orig', 'dNa', 'pred_log10S', 'pred_S',
                    'S_M', 'S_X', 'score_synthesizability', 'entropy_bin', 'composite']
    print(f"\n========== Top {TARGET_TOP} candidates ==========")
    print(top30[display_cols].to_string())
    print(f"\nTop{TARGET_TOP} count: {len(top30)}")
    print(f"Top{TARGET_TOP} pred_log10S range: [{top30['pred_log10S'].min():.3f}, {top30['pred_log10S'].max():.3f}]")
    print(f"Top{TARGET_TOP} |dNa| classes:")
    print(f"  A_direct (|dNa|<0.05): {(top30['abs_dNa']<0.05).sum()}")
    print(f"  B_sensitivity (0.05-0.15): {((top30['abs_dNa']>=0.05)&(top30['abs_dNa']<0.15)).sum()}")
    print(f"  C_approximate (0.15-0.30): {((top30['abs_dNa']>=0.15)&(top30['abs_dNa']<0.30)).sum()}")
    print(f"  D_trend only (>=0.30): {(top30['abs_dNa']>=0.30).sum()}")


                                 
    novelty_threshold = float(top30['Min_Distance_to_Train'].min())
    eligible = df_all_candidates[df_all_candidates['pred_log10S'] >= PERF_THRESHOLD].copy()
    eligible = eligible[eligible['Min_Distance_to_Train'] >= novelty_threshold].copy()
    eligible['entropy_bin'] = pd.qcut(
        eligible['S_M'] + eligible['S_X'], q=ENTROPY_BINS, labels=False, duplicates='drop')
    eligible = eligible.sort_values('composite', ascending=False).reset_index(drop=True)
    selected_rows = []
    selected_formulas = set()
    total_count = {}
    bin_count = {int(b): {} for b in sorted(eligible['entropy_bin'].dropna().unique())}
    for bin_id in sorted(bin_count):
        count = 0
        for _, row in eligible[eligible['entropy_bin'] == bin_id].iterrows():
            if count >= STRATIFIED_PER_BIN:
                break
            dopant = row['dopant'] if pd.notna(row['dopant']) else 'None'
            if bin_count[bin_id].get(dopant, 0) < STRATIFIED_PER_DOPANT_BIN and total_count.get(dopant, 0) < STRATIFIED_PER_DOPANT_TOTAL:
                selected_rows.append(row)
                selected_formulas.add(row['Formula'])
                bin_count[bin_id][dopant] = bin_count[bin_id].get(dopant, 0) + 1
                total_count[dopant] = total_count.get(dopant, 0) + 1
                count += 1
    for _, row in eligible[~eligible['Formula'].isin(selected_formulas)].iterrows():
        if len(selected_rows) >= TARGET_TOP:
            break
        dopant = row['dopant'] if pd.notna(row['dopant']) else 'None'
        bin_id = int(row['entropy_bin'])
        if bin_count[bin_id].get(dopant, 0) < STRATIFIED_PER_DOPANT_BIN and total_count.get(dopant, 0) < STRATIFIED_PER_DOPANT_TOTAL:
            selected_rows.append(row)
            selected_formulas.add(row['Formula'])
            bin_count[bin_id][dopant] = bin_count[bin_id].get(dopant, 0) + 1
            total_count[dopant] = total_count.get(dopant, 0) + 1
    # Save the stratified Top30/20/8 lists.
    stratified = pd.DataFrame(selected_rows).sort_values('composite', ascending=False).head(TARGET_TOP).reset_index(drop=True)
    stratified.to_csv(OUTPUT_DIR / 'Top30_Stratified.csv', index=False)
    stratified.head(20).reset_index(drop=True).to_csv(OUTPUT_DIR / 'Top20_Stratified.csv', index=False)
    stratified.head(8).reset_index(drop=True).to_csv(OUTPUT_DIR / 'Top8_Stratified.csv', index=False)
    print(f'Stratified candidates: {len(stratified)}')

                      
    # Write the screening report.
    with open(OUTPUT_DIR / 'Screening_Report.txt', 'w', encoding='utf-8') as f:
        f.write("NASICON high-throughput screening report\n")
        f.write("=" * 80 + "\n")
        f.write(f"Seeds: {len(seeds)}\n")
        f.write(f"Raw candidates (exploratory tolerance {CHARGE_BALANCE_TOL}): {len(candidates_raw)}\n")
        f.write(f"Unique neutral candidates after correction: {len(candidates)}\n")
        f.write(f"After skeleton deduplication: {len(df_results)}\n")
        f.write(f"Highest predicted log10S: {df_all_candidates['pred_log10S'].max():.3f}\n")
        f.write(f"Corresponding conductivity: {10**df_all_candidates['pred_log10S'].max():.3e} S/cm\n")

        f.write("\n--- Changes in this version ---\n")
        f.write("0. Generation stage (build_formula / mutate_* / generate_candidates) unchanged\n")
        f.write(f"1. Post-generation Na correction: Na_ideal = -q_non_na, {NA_DEC} decimals\n")
        f.write(f"2. Na correction range: [{MIN_NA_CORRECTED}, {MAX_NA_CORRECTED}]; out-of-range dropped\n")
        f.write("3. Deduplicate neutral compositions; keep the smallest |dNa| per formula\n")
        f.write("4. Correction uses a separate build_formula_neutral and does not alter generation\n")
        f.write("5. Features, predictions, and scores are recomputed on corrected neutral compositions\n")
        f.write(f"6. Writes the Top8, Top20, and Top{TARGET_TOP} lists\n")

        f.write("\n--- Correction statistics ---\n")
        f.write(f"Input candidates: {int(df_correction_info['merged_from_count'].sum()) + sum(dropped.values())}\n")
        f.write(f"  Dropped for Na above upper bound (>{MAX_NA_CORRECTED}): {dropped['na_too_high']}\n")
        f.write(f"  Dropped for Na below lower bound (<{MIN_NA_CORRECTED}): {dropped['na_too_low']}\n")
        f.write(f"  Dropped for residual charge: {dropped['q_residual']}\n")
        f.write(f"  Dropped for parse errors: {dropped['parse_error']}\n")
        f.write(f"Unique neutral candidates after correction: {len(df_correction_info)}\n")
        f.write(f"Merged Na variants: {int(df_correction_info['merged_from_count'].sum()) - len(df_correction_info)}\n")
        f.write("Original |dq| distribution:\n")
        for thr in [1e-3, 0.05, 0.10, 0.20, 0.40]:
            c = (df_correction_info['dq_orig'].abs() < thr).sum()
            f.write(f"  < {thr:.3f}: {c} ({c/len(df_correction_info)*100:.1f}%)\n")
        f.write("|dNa| distribution:\n")
        for thr in [0.05, 0.10, 0.20, 0.30]:
            c = (df_correction_info['abs_dNa'] < thr).sum()
            f.write(f"  < {thr:.2f}: {c} ({c/len(df_correction_info)*100:.1f}%)\n")

        f.write("\n--- Retained improvements ---\n")
        f.write("1. Added Lu, Pr, Eu, Ce, K, and Zn to RADIUS, NEG, valence, and VALENCE_MAP\n")
        f.write("2. Strict avg and compute_radius_deviation\n")
        f.write("3. Seed charge-balance check\n")
        f.write("4. compute_valence_std includes Zr\n")
        f.write("5. compute_my_features scans all M elements\n")
        f.write("6. Random seed 42\n")

        f.write("\n--- Multi-objective weights ---\n")
        f.write(f"Performance: {WEIGHT_PERF}; synthesizability: {WEIGHT_SYNTH}; "
                f"entropy coverage: {WEIGHT_ENTROPY_COV}; novelty: {WEIGHT_NOVEL}\n")

        f.write(f"\n--- Stratified selection ---\n")
        f.write(f"Entropy bins: {ENTROPY_BINS}; target per bin: {target_per_bin}\n")
        f.write(f"Per-element cap within a bin: {MAX_PER_DOPANT_PER_BIN}\n")
        f.write(f"Global per-element cap: {MAX_PER_DOPANT_TOTAL}\n")
        f.write(f"Performance threshold (pred_log10S): {PERF_THRESHOLD}\n")

        f.write(f"\nTop{TARGET_TOP} candidates (actual {len(top30)}, sorted by composite):\n")
        full_cols = ['Formula', 'Formula_orig', 'dNa', 'pred_log10S', 'pred_S',
                     'S_M', 'S_X', 'score_radius', 'score_valence',
                     'score_entropy_stab', 'score_synthesizability',
                     'entropy_bin', 'composite']
        avail = [c for c in full_cols if c in top30.columns]
        f.write(top30[avail].to_string())

        f.write(f"\n\n--- Top{TARGET_TOP} |dNa| classes ---\n")
        f.write(f"  A_direct (|dNa|<0.05): {(top30['abs_dNa']<0.05).sum()}\n")
        f.write(f"  B_sensitivity (0.05-0.15): {((top30['abs_dNa']>=0.05)&(top30['abs_dNa']<0.15)).sum()}\n")
        f.write(f"  C_approximate (0.15-0.30): {((top30['abs_dNa']>=0.15)&(top30['abs_dNa']<0.30)).sum()}\n")
        f.write(f"  D_trend only (>=0.30): {(top30['abs_dNa']>=0.30).sum()}\n")

        f.write(f"\n--- Dopant distribution (Top{TARGET_TOP}) ---\n")
        for dop, cnt in top30['dopant'].value_counts().items():
            f.write(f"  {dop}: {cnt}\n")

        f.write(f"\n--- Performance distribution ---\n")
        f.write(f"Top{TARGET_TOP} pred_log10S range: [{top30['pred_log10S'].min():.3f}, {top30['pred_log10S'].max():.3f}]\n")
        f.write(f"Top{TARGET_TOP} pred_S range: [{top30['pred_S'].min():.3e}, {top30['pred_S'].max():.3e}]\n")
        f.write(f"Top{TARGET_TOP} Na range: [{top30['Formula'].apply(lambda x: parse_formula(x).get('Na',0)).min():.3f}, "
                f"{top30['Formula'].apply(lambda x: parse_formula(x).get('Na',0)).max():.3f}]\n")

    print(f"\n[ok] Results written to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()