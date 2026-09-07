"""Update 'AAAI Results - Ablation.csv' RL section from ablation_lever_summary.csv.

Reads the re-rolled results at (lam=0.5, delta=0.1) and replaces every RL variant
cell in the ablation CSV.  Non-RL rows (sequential backbone ablations) are untouched.

Usage:
  python update_ablation_csv.py
"""
import os, csv
import pandas as pd

CSVS = '/home/group/maestro_visual/Efficient-Multimodal-Bottleneck/runs/csvs'
ABLATION_CSV = os.path.join(CSVS, 'AAAI Results - Ablation.csv')
SUMMARY_CSV  = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'results_k16', 'ablation_lever_summary.csv')

# Mapping from dataset name in summary -> name used as header in the ablation CSV
DS_LABEL = {
    'iemocap':  'IEMOCAP',
    'mmfi':     'MM-Fi',
    'cmi':      'CMI',
    'czu_mhad': 'CZU-MHAD',
    'dsads':    'DSADS',
    'eav':      'EAV',
    'pamap2':   'PAMAP2',
    'utd_mhad': 'UTD-MHAD',
}

VARIANT_LABEL = {
    'full':              'full (Ours) — RL policy',
    'zeroq':             'zero-initialized Q-network',
    'randorder':         'random-ordering initialized',
    'nostop':            'no stop action',
    'marginal':          'marginal-Shapley advice (fullshap)',
    'marginal_randorder':'marginal-Shapley + random order',
}

DS_ORDER = ['iemocap', 'mmfi', 'cmi', 'czu_mhad', 'dsads', 'eav', 'pamap2', 'utd_mhad']
VARIANT_ORDER = ['full', 'zeroq', 'randorder', 'nostop', 'marginal', 'marginal_randorder']


def fmt_f1(mean, std):
    return f'{mean:.3f} | {std:.3f}'

def fmt_gf(v):
    return f'{v:.3f}'

def fmt_mu(v):
    return f'{v:.2f}'


def load_summary(path):
    df = pd.read_csv(path)
    result = {}
    for _, row in df.iterrows():
        ds = row['dataset']
        vk = row['variant']
        cond = row['condition']
        result[(ds, vk, cond)] = (row['mean_f1'], row['std_f1'],
                                   row['mean_gflops'], row['mean_mu'])
    return result


def cells(summary, ds, vkey):
    """Return (F1_A0, GF_A0, MU_A0, F1_A40, GF_A40, MU_A40) formatted strings."""
    def get(cond):
        k = (ds, vkey, cond)
        if k not in summary:
            return ('', '', '')
        mf1, sf1, mgf, mmu = summary[k]
        return (fmt_f1(mf1, sf1), fmt_gf(mgf), fmt_mu(mmu))
    a0  = get('A0')
    a40 = get('A40')
    return a0 + a40   # 6-tuple: F1_A0, GF_A0, MU_A0, F1_A40, GF_A40, MU_A40


def main():
    summary = load_summary(SUMMARY_CSV)
    missing = [(ds, vk) for ds in DS_ORDER for vk in VARIANT_ORDER
               if (ds, vk, 'A0') not in summary]
    if missing:
        print(f'WARNING: missing {len(missing)} (ds,variant) pairs from summary:')
        for m in missing:
            print(f'  {m}')

    # Read original CSV
    with open(ABLATION_CSV, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))

    # Identify dataset header rows and RL variant rows
    # Structure: each dataset block starts with a row like ['IEMOCAP','','0%',...]
    # followed by ['','F1','GFLOP','MU','F1','GFLOP','MU'] header,
    # then RL/Ours data rows.

    # Map: (dataset, variant_idx) -> row_index
    # variant_idx 0..5 corresponds to VARIANT_ORDER
    # For IEMOCAP: RL rows start at row 8 (0-indexed) = row 9 (1-indexed in CSV)
    # For other datasets: first row of each block is dataset header, then F1 header, then variants

    current_ds = None
    variant_counter = 0   # counts RL variant rows seen in current dataset block
    in_rl_section = False

    new_rows = []
    i = 0
    while i < len(rows):
        row = rows[i]

        # Check if this row is a dataset header (col 0 has a known label)
        if row and row[0] in DS_LABEL.values():
            current_ds = next(k for k,v in DS_LABEL.items() if v == row[0])
            variant_counter = 0
            in_rl_section = False
            new_rows.append(row)
            i += 1
            continue

        # The IEMOCAP block starts at row 1 (1-indexed); 'IEMOCAP' IS the header
        # Detect: header row 'RL' col or row starting variant sequence
        if row and row[0] == 'RL':
            in_rl_section = True
            variant_counter = 0
            # This row is the first RL variant ('full')
            vkey = VARIANT_ORDER[variant_counter]
            c = cells(summary, current_ds or 'iemocap', vkey)
            new_row = [row[0], row[1]] + list(c)
            new_rows.append(new_row)
            variant_counter += 1
            i += 1
            continue

        # Blank rows or non-RL header rows
        if not any(row):
            in_rl_section = False
            new_rows.append(row)
            i += 1
            continue

        # Detect the column header row: ['', 'F1', 'GFLOP', 'MU', 'F1', 'GFLOP', 'MU']
        if row[1:3] == ['F1', 'GFLOP'] or (len(row) > 2 and 'F1' in row[1]):
            in_rl_section = False
            variant_counter = 0
            new_rows.append(row)
            i += 1
            continue

        # Within an RL section: replace values
        if in_rl_section and current_ds and variant_counter < len(VARIANT_ORDER):
            vkey = VARIANT_ORDER[variant_counter]
            c = cells(summary, current_ds, vkey)
            new_row = [row[0], row[1]] + list(c)
            new_rows.append(new_row)
            variant_counter += 1
            i += 1
            continue

        # Detect start of RL variant rows in per-dataset blocks (col 1 matches a variant label)
        if current_ds and not in_rl_section and len(row) > 1:
            label = row[1].strip()
            if label in VARIANT_LABEL.values():
                in_rl_section = True
                variant_counter = 0
                vkey = VARIANT_ORDER[variant_counter]
                c = cells(summary, current_ds, vkey)
                new_row = [row[0], row[1]] + list(c)
                new_rows.append(new_row)
                variant_counter += 1
                i += 1
                continue

        new_rows.append(row)
        i += 1

    # Write updated CSV
    with open(ABLATION_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(new_rows)

    print(f'Updated: {ABLATION_CSV}')
    print(f'Total rows: {len(new_rows)}')

    # Quick sanity check: print updated RL values for IEMOCAP
    print('\nIEMOCAP RL rows (new values):')
    for vk in VARIANT_ORDER:
        c = cells(summary, 'iemocap', vk)
        if any(c):
            print(f'  {VARIANT_LABEL[vk][:35]:35s}  A0: {c[0]} {c[1]} {c[2]}  '
                  f'A40: {c[3]} {c[4]} {c[5]}')


if __name__ == '__main__':
    main()
