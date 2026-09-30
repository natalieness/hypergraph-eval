#!/usr/bin/env python3
"""
Co-targeting motif analysis with predicted block labels.

This script reproduces the analyses from the ChatGPT session:
  1. Compression of sparse exact postsynaptic co-target pairs into postsynaptic block-pair motifs.
  2. Abundance-based null model: do block pairs occur more/less often than expected from each presynaptic neuron's target-block abundance?
  3. Celltype-corrected residuals: do block-pair residuals remain after accounting for block/celltype abundance and empirical celltype-pair structure?
  4. Residual-pair driver summaries: which named neurons/celltypes/source neurons drive the strongest residual block pairs?
  5. Empirical permutation/null comparison, if randomized connector tables are provided.

Inputs expected by default:
  - connector table with columns: connector_id, presynaptic_to, postsynaptic_to
  - block table with columns: skeleton_id, predicted_block, optionally name/celltype/predicted_neurotransmitter
  - optional neuron details table with columns: skeleton_id, name, celltype, predicted_neurotransmitter
  - optional permutation connector tables, usually same schema as connector table

Counting convention:
  - Per connector, postsynaptic_to is converted to UNIQUE postsynaptic skeleton IDs.
  - Co-target pairs are unique unordered pairs of postsynaptic skeleton IDs on that connector.
  - Block-pair analyses only use co-target events where BOTH postsynaptic skeleton IDs have predicted_block labels.

Example:
  python cotarget_block_analysis.py \
    --connectors connector_details2025.csv \
    --blocks 20260601_details_predicted_block.csv \
    --details 'neuron_details_with_nt(1).csv' \
    --permutations 'perm_model_short_*.csv' \
    --outdir cotarget_analysis_outputs \
    --zip
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

Pair = Tuple[int, int]
CellPair = Tuple[str, str]
BlockCell = Tuple[int, str]

_NUM_RE = re.compile(r"-?\d+")


def parse_int_list(value) -> List[int]:
    """Parse strings like '[1, 2, 3]' into ints. Robust to None strings and extra text."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if not isinstance(value, str):
        try:
            return [int(value)]
        except Exception:
            return []
    return [int(m.group(0)) for m in _NUM_RE.finditer(value)]


def pair_key(a: int, b: int) -> Pair:
    return (a, b) if a <= b else (b, a)


def str_pair_key(a: int, b: int) -> str:
    a, b = pair_key(int(a), int(b))
    return f"{a}+{b}"


def cell_pair_key(a: str, b: str) -> CellPair:
    a = str(a) if a is not None and str(a) != "" else "unknown"
    b = str(b) if b is not None and str(b) != "" else "unknown"
    return (a, b) if a <= b else (b, a)


def entropy_bits(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for count in counter.values():
        p = count / total
        if p > 0:
            h -= p * math.log2(p)
    return h


def safe_div(a: float, b: float) -> float:
    return a / b if b else np.nan


def read_metadata(
    blocks_path: str,
    details_path: Optional[str],
    skeleton_col: str,
    block_col: str,
    name_col: str,
    celltype_col: str,
    nt_col: str,
) -> pd.DataFrame:
    blocks = pd.read_csv(blocks_path)
    blocks = blocks.drop(columns=[c for c in blocks.columns if str(c).startswith("Unnamed")], errors="ignore")
    required = {skeleton_col, block_col}
    missing = required - set(blocks.columns)
    if missing:
        raise ValueError(f"Block table is missing required columns: {sorted(missing)}")
    blocks = blocks.dropna(subset=[skeleton_col, block_col]).copy()
    blocks[skeleton_col] = blocks[skeleton_col].astype(int)
    blocks[block_col] = blocks[block_col].astype(int)

    # Ensure optional metadata columns exist.
    for col in [name_col, celltype_col, nt_col]:
        if col not in blocks.columns:
            blocks[col] = np.nan

    meta = blocks[[skeleton_col, block_col, name_col, celltype_col, nt_col]].copy()

    if details_path:
        details = pd.read_csv(details_path)
        details = details.drop(columns=[c for c in details.columns if str(c).startswith("Unnamed")], errors="ignore")
        if skeleton_col not in details.columns:
            raise ValueError(f"Details table is missing skeleton column {skeleton_col!r}")
        details[skeleton_col] = details[skeleton_col].astype(int)
        for col in [name_col, celltype_col, nt_col]:
            if col not in details.columns:
                details[col] = np.nan
        details = details[[skeleton_col, name_col, celltype_col, nt_col]].copy()
        meta = meta.merge(details, on=skeleton_col, how="left", suffixes=("", "_details"))
        for col in [name_col, celltype_col, nt_col]:
            detail_col = f"{col}_details"
            if detail_col in meta.columns:
                meta[col] = meta[col].where(meta[col].notna() & (meta[col].astype(str) != ""), meta[detail_col])
                meta = meta.drop(columns=[detail_col])

    meta[name_col] = meta[name_col].fillna(meta[skeleton_col].astype(str)).astype(str)
    meta[celltype_col] = meta[celltype_col].fillna("unknown").replace("", "unknown").astype(str)
    meta[nt_col] = meta[nt_col].fillna("unknown").replace("", "unknown").astype(str)
    meta = meta.rename(
        columns={
            skeleton_col: "skeleton_id",
            block_col: "predicted_block",
            name_col: "name",
            celltype_col: "celltype",
            nt_col: "predicted_neurotransmitter",
        }
    )
    return meta


class ObservedCounts:
    def __init__(self):
        self.n_connectors = 0
        self.n_presynaptic = 0
        self.all_pair_events = 0
        self.block_labeled_pair_events = 0
        self.connectors_with_pairs = 0
        self.connectors_with_block_pairs = 0
        self.post_contact_total = 0
        self.post_contact_block_labeled = 0
        self.exact_pair_counts: Counter[Pair] = Counter()
        self.block_pair_counts: Counter[Pair] = Counter()
        self.celltype_pair_counts: Counter[CellPair] = Counter()
        self.observed_blockpair_cellpair: Counter[Tuple[Pair, CellPair]] = Counter()
        self.source_block_observed_blockpair: Dict[int, Counter[Pair]] = defaultdict(Counter)
        self.source_slots: Counter[int] = Counter()
        self.sourceblock_slots: Counter[int] = Counter()
        self.source_block_incidence: Dict[int, Counter[int]] = defaultdict(Counter)
        self.source_blockcell_incidence: Dict[int, Counter[BlockCell]] = defaultdict(Counter)
        self.sameblock_pair_events = 0


def accumulate_observed_counts(
    connectors_path: str,
    meta: pd.DataFrame,
    pre_col: str,
    post_col: str,
    connector_id_col: str,
    chunksize: Optional[int] = None,
) -> ObservedCounts:
    block_map = dict(zip(meta["skeleton_id"].astype(int), meta["predicted_block"].astype(int)))
    celltype_map = dict(zip(meta["skeleton_id"].astype(int), meta["celltype"].astype(str)))
    pre_set = set(block_map)
    out = ObservedCounts()

    usecols = [pre_col, post_col]
    if connector_id_col and connector_id_col in pd.read_csv(connectors_path, nrows=0).columns:
        usecols = [connector_id_col, pre_col, post_col]

    reader = pd.read_csv(connectors_path, usecols=usecols, chunksize=chunksize) if chunksize else [pd.read_csv(connectors_path, usecols=usecols)]
    presyn_seen = set()
    for chunk in reader:
        chunk = chunk[chunk[pre_col].isin(pre_set)]
        out.n_connectors += len(chunk)
        for pre, posts_s in zip(chunk[pre_col].to_numpy(), chunk[post_col].to_numpy()):
            pre = int(pre)
            presyn_seen.add(pre)
            pre_b = block_map[pre]
            posts = sorted(set(parse_int_list(posts_s)))
            if not posts:
                continue
            m = len(posts)
            out.post_contact_total += m
            if m >= 2:
                out.all_pair_events += m * (m - 1) // 2
                out.connectors_with_pairs += 1

            labeled = [p for p in posts if p in block_map]
            out.post_contact_block_labeled += len(labeled)
            mB = len(labeled)
            if mB >= 2:
                slots = mB * (mB - 1) // 2
                out.source_slots[pre] += slots
                out.sourceblock_slots[pre_b] += slots
                out.connectors_with_block_pairs += 1

            for p in labeled:
                b = block_map[p]
                ct = celltype_map.get(p, "unknown") or "unknown"
                out.source_block_incidence[pre][b] += 1
                out.source_blockcell_incidence[pre][(b, ct)] += 1

            if mB >= 2:
                for a, b in combinations(labeled, 2):
                    pp = pair_key(a, b)
                    ba = block_map[a]
                    bb = block_map[b]
                    bp = pair_key(ba, bb)
                    ctp = cell_pair_key(celltype_map.get(a, "unknown"), celltype_map.get(b, "unknown"))
                    out.exact_pair_counts[pp] += 1
                    out.block_pair_counts[bp] += 1
                    out.celltype_pair_counts[ctp] += 1
                    out.observed_blockpair_cellpair[(bp, ctp)] += 1
                    out.source_block_observed_blockpair[pre_b][bp] += 1
                    out.block_labeled_pair_events += 1
                    if ba == bb:
                        out.sameblock_pair_events += 1
    out.n_presynaptic = len(presyn_seen)
    return out


def write_compression_tables(counts: ObservedCounts, outdir: Path) -> None:
    obs_total = counts.block_labeled_pair_events
    H_exact = entropy_bits(counts.exact_pair_counts)
    H_block = entropy_bits(counts.block_pair_counts)
    H_cell = entropy_bits(counts.celltype_pair_counts)
    rows = []

    def add(metric, value):
        rows.append({"metric": metric, "value": value})

    add("connectors_from_block_labeled_presynaptic_neurons", counts.n_connectors)
    add("unique_presynaptic_neurons", counts.n_presynaptic)
    add("all_unique_postsynaptic_pair_events", counts.all_pair_events)
    add("both_posts_have_predicted_block_pair_events", counts.block_labeled_pair_events)
    add("fraction_pair_events_with_both_post_blocks", safe_div(counts.block_labeled_pair_events, counts.all_pair_events))
    add("post_contact_total_unique_per_connector", counts.post_contact_total)
    add("post_contact_block_labeled_unique_per_connector", counts.post_contact_block_labeled)
    add("fraction_post_contacts_with_predicted_block", safe_div(counts.post_contact_block_labeled, counts.post_contact_total))
    add("unique_exact_postsynaptic_pairs_both_block_labeled", len(counts.exact_pair_counts))
    add("unique_postsynaptic_block_pairs_observed", len(counts.block_pair_counts))
    add("unique_postsynaptic_celltype_pairs_observed", len(counts.celltype_pair_counts))
    add("unique_pair_compression_ratio_exact_pairs_per_block_pair", safe_div(len(counts.exact_pair_counts), len(counts.block_pair_counts)))
    add("entropy_bits_exact_target_pair_distribution", H_exact)
    add("entropy_bits_block_pair_distribution", H_block)
    add("entropy_bits_celltype_pair_distribution", H_cell)
    add("effective_number_exact_target_pairs", 2**H_exact)
    add("effective_number_block_pairs", 2**H_block)
    add("effective_number_celltype_pairs", 2**H_cell)
    add("entropy_fraction_retained_by_block_pairs_Hblock_over_Hexact", safe_div(H_block, H_exact))
    add("effective_compression_ratio_exact_effective_pairs_per_effective_block_pair", safe_div(2**H_exact, 2**H_block))
    add("same_postsynaptic_block_pair_fraction_observed", safe_div(counts.sameblock_pair_events, obs_total))
    for k in [5, 10, 20, 50]:
        top = sum(c for _, c in counts.block_pair_counts.most_common(k))
        add(f"top_{k}_block_pairs_fraction_of_block_labeled_pair_events", safe_div(top, obs_total))
    for thresh in [2, 5, 10, 25, 50]:
        ev = sum(c for c in counts.exact_pair_counts.values() if c >= thresh)
        nrep = sum(1 for c in counts.exact_pair_counts.values() if c >= thresh)
        add(f"exact_pairs_count_repeated_at_least_{thresh}", nrep)
        add(f"exact_pairs_events_fraction_repeated_at_least_{thresh}", safe_div(ev, obs_total))
    pd.DataFrame(rows).to_csv(outdir / "compression_summary.csv", index=False)

    bp_rows = []
    for (a, b), c in counts.block_pair_counts.most_common():
        bp_rows.append(
            {
                "post_block_pair": f"{a}+{b}",
                "block_a": a,
                "block_b": b,
                "observed_pairs": c,
                "fraction": safe_div(c, obs_total),
                "same_block": a == b,
            }
        )
    pd.DataFrame(bp_rows).to_csv(outdir / "block_pair_observed_counts.csv", index=False)

    cp_rows = []
    for (a, b), c in counts.celltype_pair_counts.most_common():
        cp_rows.append({"post_celltype_pair": f"{a} + {b}", "celltype_a": a, "celltype_b": b, "observed_pairs": c, "fraction": safe_div(c, obs_total)})
    pd.DataFrame(cp_rows).to_csv(outdir / "celltype_pair_observed_counts.csv", index=False)


def compute_abundance_null(counts: ObservedCounts, pre_block_map: Dict[int, int], block_values: Sequence[int]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Counter, Dict[int, Counter]]:
    expected_blockpair: Counter[Pair] = Counter()
    expected_sourceblock_blockpair: Dict[int, Counter[Pair]] = defaultdict(Counter)

    for pre, bcounts in counts.source_block_incidence.items():
        slots = counts.source_slots[pre]
        total = sum(bcounts.values())
        if slots <= 0 or total <= 0:
            continue
        pre_b = pre_block_map[pre]
        items = sorted(bcounts.items())
        for i, (a, ca) in enumerate(items):
            pa = ca / total
            e = slots * pa * pa
            expected_blockpair[(a, a)] += e
            expected_sourceblock_blockpair[pre_b][(a, a)] += e
            for b, cb in items[i + 1 :]:
                e = slots * 2 * pa * (cb / total)
                expected_blockpair[(a, b)] += e
                expected_sourceblock_blockpair[pre_b][(a, b)] += e

    obs_total = sum(counts.block_pair_counts.values())
    exp_total = sum(expected_blockpair.values())
    keys = set(counts.block_pair_counts) | set(expected_blockpair)
    rows = []
    tv = 0.0
    kl = 0.0
    chi = 0.0
    for bp in sorted(keys):
        obs = counts.block_pair_counts.get(bp, 0)
        exp = expected_blockpair.get(bp, 0.0)
        o = safe_div(obs, obs_total) if obs_total else 0
        e = safe_div(exp, exp_total) if exp_total else 0
        tv += abs(o - e)
        if o > 0 and e > 0:
            kl += o * math.log2(o / e)
        exp_scaled = exp * obs_total / exp_total if exp_total else 0
        if exp_scaled > 0:
            chi += (obs - exp_scaled) ** 2 / exp_scaled
        rows.append(
            {
                "post_block_pair": f"{bp[0]}+{bp[1]}",
                "block_a": bp[0],
                "block_b": bp[1],
                "observed": obs,
                "expected_per_source_block_abundance": exp,
                "obs_exp_ratio": (obs + 0.5) / (exp + 0.5),
                "log2_obs_exp": math.log2((obs + 0.5) / (exp + 0.5)),
                "std_resid": (obs - exp) / math.sqrt(exp + 1),
                "same_block": bp[0] == bp[1],
            }
        )
    enrich_df = pd.DataFrame(rows).sort_values("log2_obs_exp", ascending=False)

    same_obs = sum(c for bp, c in counts.block_pair_counts.items() if bp[0] == bp[1])
    same_exp = sum(c for bp, c in expected_blockpair.items() if bp[0] == bp[1])
    summary = pd.DataFrame(
        [
            {"metric": "observed_block_labeled_pair_events", "value": obs_total},
            {"metric": "expected_pair_events_under_null", "value": exp_total},
            {"metric": "total_variation_distance_observed_vs_null", "value": tv / 2},
            {"metric": "KL_bits_observed_vs_null", "value": kl},
            {"metric": "chi_square_statistic_descriptive_large_N", "value": chi},
            {"metric": "same_block_fraction_observed", "value": safe_div(same_obs, obs_total)},
            {"metric": "same_block_fraction_expected_per_source_abundance", "value": safe_div(same_exp, exp_total)},
            {"metric": "same_block_obs_exp_ratio", "value": safe_div(safe_div(same_obs, obs_total), safe_div(same_exp, exp_total))},
        ]
    )

    sb_rows = []
    for pre_b in sorted(block_values):
        obs_counter = counts.source_block_observed_blockpair.get(pre_b, Counter())
        exp_counter = expected_sourceblock_blockpair.get(pre_b, Counter())
        ot = sum(obs_counter.values())
        et = sum(exp_counter.values())
        if ot == 0 or et == 0:
            continue
        same_o = sum(v for bp, v in obs_counter.items() if bp[0] == bp[1])
        same_e = sum(v for bp, v in exp_counter.items() if bp[0] == bp[1])
        keys2 = set(obs_counter) | set(exp_counter)
        tv2 = 0.0
        kl2 = 0.0
        tmp = []
        for bp in keys2:
            o = obs_counter.get(bp, 0) / ot
            e = exp_counter.get(bp, 0) / et
            tv2 += abs(o - e)
            if o > 0 and e > 0:
                kl2 += o * math.log2(o / e)
            obs = obs_counter.get(bp, 0)
            exp = exp_counter.get(bp, 0.0)
            if obs + exp >= 10:
                tmp.append((math.log2((obs + 0.5) / (exp + 0.5)), bp, obs, exp))
        top_en = max(tmp, default=(np.nan, None, np.nan, np.nan), key=lambda x: x[0])
        top_de = min(tmp, default=(np.nan, None, np.nan, np.nan), key=lambda x: x[0])
        sb_rows.append(
            {
                "presynaptic_block": pre_b,
                "observed_pairs": ot,
                "expected_pairs": et,
                "TV_distance_vs_per_source_abundance_null": tv2 / 2,
                "KL_bits_vs_per_source_abundance_null": kl2,
                "same_post_block_fraction_observed": same_o / ot,
                "same_post_block_fraction_expected": same_e / et,
                "same_post_block_obs_exp_ratio": safe_div(same_o / ot, same_e / et),
                "top_enriched_post_block_pair": f"{top_en[1][0]}+{top_en[1][1]}" if top_en[1] else "",
                "top_enriched_log2OE": top_en[0],
                "top_enriched_obs": top_en[2],
                "top_enriched_exp": top_en[3],
                "top_depleted_post_block_pair": f"{top_de[1][0]}+{top_de[1][1]}" if top_de[1] else "",
                "top_depleted_log2OE": top_de[0],
                "top_depleted_obs": top_de[2],
                "top_depleted_exp": top_de[3],
            }
        )
    sb_df = pd.DataFrame(sb_rows).sort_values("presynaptic_block")
    return enrich_df, summary, sb_df, expected_blockpair, expected_sourceblock_blockpair


def compute_celltype_corrected_residuals(counts: ObservedCounts) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Expected counts for (block, celltype) pairs from per-source abundance.
    expected_blockpair_cellpair: Counter[Tuple[Pair, CellPair]] = Counter()
    expected_cellpair: Counter[CellPair] = Counter()

    for pre, cat_counts in counts.source_blockcell_incidence.items():
        slots = counts.source_slots[pre]
        total = sum(cat_counts.values())
        if slots <= 0 or total <= 0:
            continue
        items = sorted(cat_counts.items())
        for i, ((ba, cta), ca) in enumerate(items):
            pa = ca / total
            bp = (ba, ba)
            cp = cell_pair_key(cta, cta)
            e = slots * pa * pa
            expected_blockpair_cellpair[(bp, cp)] += e
            expected_cellpair[cp] += e
            for (bb, ctb), cb in items[i + 1 :]:
                bp = pair_key(ba, bb)
                cp = cell_pair_key(cta, ctb)
                e = slots * 2 * pa * (cb / total)
                expected_blockpair_cellpair[(bp, cp)] += e
                expected_cellpair[cp] += e

    # Empirical multiplier for each celltype pair.
    alpha = 0.5
    multipliers: Dict[CellPair, float] = {}
    all_cellpairs = set(counts.celltype_pair_counts) | set(expected_cellpair)
    for cp in all_cellpairs:
        multipliers[cp] = (counts.celltype_pair_counts.get(cp, 0) + alpha) / (expected_cellpair.get(cp, 0.0) + alpha)

    corrected_expected_blockpair: Counter[Pair] = Counter()
    for (bp, cp), exp in expected_blockpair_cellpair.items():
        corrected_expected_blockpair[bp] += exp * multipliers.get(cp, 1.0)

    # Rescale to observed total to keep fractions/counts comparable.
    obs_total = sum(counts.block_pair_counts.values())
    corr_total = sum(corrected_expected_blockpair.values())
    scale = obs_total / corr_total if corr_total else 1.0
    for bp in list(corrected_expected_blockpair.keys()):
        corrected_expected_blockpair[bp] *= scale

    keys = set(counts.block_pair_counts) | set(corrected_expected_blockpair)
    rows = []
    tv = 0.0
    kl = 0.0
    for bp in sorted(keys):
        obs = counts.block_pair_counts.get(bp, 0)
        exp = corrected_expected_blockpair.get(bp, 0.0)
        o = obs / obs_total if obs_total else 0
        e = exp / obs_total if obs_total else 0
        tv += abs(o - e)
        if o > 0 and e > 0:
            kl += o * math.log2(o / e)
        rows.append(
            {
                "post_block_pair": f"{bp[0]}+{bp[1]}",
                "block_a": bp[0],
                "block_b": bp[1],
                "observed": obs,
                "expected_after_source_abundance_and_celltype_pair": exp,
                "obs_exp_ratio": (obs + 0.5) / (exp + 0.5),
                "log2_obs_exp": math.log2((obs + 0.5) / (exp + 0.5)),
                "std_resid": (obs - exp) / math.sqrt(exp + 1),
                "same_block": bp[0] == bp[1],
            }
        )
    residual_df = pd.DataFrame(rows).sort_values("log2_obs_exp", ascending=False)

    same_obs = sum(c for bp, c in counts.block_pair_counts.items() if bp[0] == bp[1])
    same_exp = sum(c for bp, c in corrected_expected_blockpair.items() if bp[0] == bp[1])
    summary = pd.DataFrame(
        [
            {"metric": "observed_block_labeled_pair_events", "value": obs_total},
            {"metric": "expected_pair_events_after_celltype_correction", "value": sum(corrected_expected_blockpair.values())},
            {"metric": "total_variation_distance_observed_vs_celltype_corrected", "value": tv / 2},
            {"metric": "KL_bits_observed_vs_celltype_corrected", "value": kl},
            {"metric": "same_block_fraction_observed", "value": safe_div(same_obs, obs_total)},
            {"metric": "same_block_fraction_expected_after_celltype_correction", "value": safe_div(same_exp, obs_total)},
            {"metric": "same_block_obs_exp_ratio_after_celltype_correction", "value": safe_div(safe_div(same_obs, obs_total), safe_div(same_exp, obs_total))},
        ]
    )
    return residual_df, summary


def write_block_glossary(meta: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows = []
    for b, g in meta.groupby("predicted_block"):
        cts = g["celltype"].fillna("unknown").astype(str).value_counts()
        nts = g["predicted_neurotransmitter"].fillna("unknown").astype(str).value_counts()
        examples = []
        # Sort example neurons by common celltypes then name for stable output.
        g2 = g.copy()
        g2["_ct_count"] = g2["celltype"].map(cts).fillna(0)
        g2 = g2.sort_values(["_ct_count", "celltype", "name"], ascending=[False, True, True])
        for _, r in g2.head(10).iterrows():
            examples.append(f"{r['name']} ({r['celltype']}, {r['predicted_neurotransmitter']})")
        rows.append(
            {
                "block": int(b),
                "n_neurons": len(g),
                "top_celltypes": "; ".join(f"{k}:{v}" for k, v in cts.head(10).items()),
                "top_nt": "; ".join(f"{k}:{v}" for k, v in nts.head(6).items()),
                "example_neurons": "; ".join(examples),
            }
        )
    glossary = pd.DataFrame(rows).sort_values("block")
    glossary.to_csv(outdir / "block_celltype_glossary.csv", index=False)
    return glossary


def select_residual_pairs(residual_df: pd.DataFrame, top_n: int, min_observed: int = 10) -> Tuple[List[str], List[str]]:
    filtered = residual_df[residual_df["observed"] >= min_observed].copy()
    pos = filtered.sort_values("log2_obs_exp", ascending=False).head(top_n)["post_block_pair"].tolist()
    neg = filtered.sort_values("log2_obs_exp", ascending=True).head(top_n)["post_block_pair"].tolist()
    return pos, neg


def short_label(sid: int, name_map: Dict[int, str], celltype_map: Dict[int, str], block_map: Dict[int, int]) -> str:
    sid = int(sid)
    nm = name_map.get(sid, str(sid))
    if not nm or str(nm).lower() == "nan":
        nm = str(sid)
    return f"{nm} ({celltype_map.get(sid, 'unknown')}, b{block_map.get(sid, 'NA')})"


def write_residual_pair_drivers(
    connectors_path: str,
    meta: pd.DataFrame,
    residual_df: pd.DataFrame,
    selected_pairs: Sequence[str],
    outdir: Path,
    pre_col: str,
    post_col: str,
    top_n_pairs_label: int = 10,
    chunksize: Optional[int] = None,
) -> pd.DataFrame:
    block_map = dict(zip(meta["skeleton_id"].astype(int), meta["predicted_block"].astype(int)))
    celltype_map = dict(zip(meta["skeleton_id"].astype(int), meta["celltype"].astype(str)))
    name_map = dict(zip(meta["skeleton_id"].astype(int), meta["name"].astype(str)))
    target_pairs = set(selected_pairs)
    selected_pos = set(selected_pairs[: top_n_pairs_label])
    residual_lookup = residual_df.set_index("post_block_pair").to_dict("index")

    target_part: Dict[str, Counter[int]] = defaultdict(Counter)
    exact_pair_counts: Dict[str, Counter[Pair]] = defaultdict(Counter)
    cell_pair_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    source_block_counts: Dict[str, Counter[int]] = defaultdict(Counter)
    source_celltype_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    source_counts: Dict[str, Counter[int]] = defaultdict(Counter)
    connector_counts: Dict[str, int] = defaultdict(int)

    usecols = [pre_col, post_col]
    reader = pd.read_csv(connectors_path, usecols=usecols, chunksize=chunksize) if chunksize else [pd.read_csv(connectors_path, usecols=usecols)]
    pre_set = set(block_map)
    for chunk in reader:
        chunk = chunk[chunk[pre_col].isin(pre_set)]
        for pre, posts_s in zip(chunk[pre_col].to_numpy(), chunk[post_col].to_numpy()):
            pre = int(pre)
            pre_b = block_map[pre]
            posts = []
            seen = set()
            for p in parse_int_list(posts_s):
                if p in seen:
                    continue
                seen.add(p)
                b = block_map.get(p)
                if b is not None:
                    posts.append((p, b))
            if len(posts) < 2:
                continue
            keys_this = set()
            for i in range(len(posts) - 1):
                p1, b1 = posts[i]
                for j in range(i + 1, len(posts)):
                    p2, b2 = posts[j]
                    key = str_pair_key(b1, b2)
                    if key not in target_pairs:
                        continue
                    ep = pair_key(p1, p2)
                    exact_pair_counts[key][ep] += 1
                    target_part[key][p1] += 1
                    target_part[key][p2] += 1
                    cp = cell_pair_key(celltype_map.get(p1, "unknown"), celltype_map.get(p2, "unknown"))
                    cell_pair_counts[key][f"{cp[0]} + {cp[1]}"] += 1
                    source_block_counts[key][pre_b] += 1
                    source_celltype_counts[key][celltype_map.get(pre, "unknown")] += 1
                    source_counts[key][pre] += 1
                    keys_this.add(key)
            for key in keys_this:
                connector_counts[key] += 1

    rows = []
    pos_set = set(selected_pairs[: max(1, len(selected_pairs)//2)])
    for key in selected_pairs:
        a, b = [int(x) for x in key.split("+")]
        res = residual_lookup.get(key, {})
        parts = target_part[key]
        tops_by_block = {}
        if a == b:
            cand = sorted([(sid, cnt) for sid, cnt in parts.items() if block_map.get(sid) == a], key=lambda x: x[1], reverse=True)[:8]
            tops_by_block[a] = "; ".join(f"{short_label(sid, name_map, celltype_map, block_map)}:{cnt}" for sid, cnt in cand)
        else:
            for tb in [a, b]:
                cand = sorted([(sid, cnt) for sid, cnt in parts.items() if block_map.get(sid) == tb], key=lambda x: x[1], reverse=True)[:6]
                tops_by_block[tb] = "; ".join(f"{short_label(sid, name_map, celltype_map, block_map)}:{cnt}" for sid, cnt in cand)
        exact_top = []
        for (sid1, sid2), cnt in exact_pair_counts[key].most_common(8):
            exact_top.append(f"{short_label(sid1, name_map, celltype_map, block_map)} + {short_label(sid2, name_map, celltype_map, block_map)}:{cnt}")
        rows.append(
            {
                "set": "selected_positive_or_top" if key in pos_set else "selected_negative_or_bottom",
                "post_block_pair": key,
                "observed": res.get("observed"),
                "expected_after_abundance_celltype": res.get("expected_after_source_abundance_and_celltype_pair"),
                "obs_exp_ratio": res.get("obs_exp_ratio"),
                "log2_obs_exp": res.get("log2_obs_exp"),
                "n_connectors_with_pair": connector_counts[key],
                "top_target_celltype_pairs": "; ".join(f"{cp}:{cnt}" for cp, cnt in cell_pair_counts[key].most_common(8)),
                "top_targets_block_a": tops_by_block.get(a, ""),
                "top_targets_block_b": tops_by_block.get(b, ""),
                "top_exact_target_pairs": "; ".join(exact_top),
                "source_presynaptic_blocks": "; ".join(f"{k}:{v}" for k, v in source_block_counts[key].most_common(8)),
                "source_presynaptic_celltypes": "; ".join(f"{k}:{v}" for k, v in source_celltype_counts[key].most_common(8)),
                "top_source_neurons": "; ".join(f"{short_label(sid, name_map, celltype_map, block_map)}:{cnt}" for sid, cnt in source_counts[key].most_common(8)),
            }
        )
    drivers = pd.DataFrame(rows)
    drivers.to_csv(outdir / "residual_pair_drivers_summary.csv", index=False)
    return drivers


def count_block_pairs_for_connector_table(
    path: str,
    meta: pd.DataFrame,
    pre_col: str,
    post_col: str,
    chunksize: Optional[int] = None,
) -> Tuple[Counter[Pair], Dict[int, Counter[Pair]], int, int, int, int]:
    """Count block-pair co-target motifs in observed or permuted connector table."""
    block_map = dict(zip(meta["skeleton_id"].astype(int), meta["predicted_block"].astype(int)))
    pre_set = set(block_map)
    pair_counts: Counter[Pair] = Counter()
    sourceblock_counts: Dict[int, Counter[Pair]] = defaultdict(Counter)
    total_pair_events = 0
    block_labeled_pair_events = 0
    n_connectors = 0
    n_presyn = 0
    presyn_seen = set()
    usecols = [pre_col, post_col]
    reader = pd.read_csv(path, usecols=usecols, chunksize=chunksize) if chunksize else [pd.read_csv(path, usecols=usecols)]
    for chunk in reader:
        chunk = chunk[chunk[pre_col].isin(pre_set)]
        n_connectors += len(chunk)
        for pre, posts_s in zip(chunk[pre_col].to_numpy(), chunk[post_col].to_numpy()):
            pre = int(pre)
            presyn_seen.add(pre)
            pre_b = block_map[pre]
            posts = sorted(set(parse_int_list(posts_s)))
            m = len(posts)
            if m >= 2:
                total_pair_events += m * (m - 1) // 2
            labeled = [p for p in posts if p in block_map]
            if len(labeled) < 2:
                continue
            for a, b in combinations(labeled, 2):
                bp = pair_key(block_map[a], block_map[b])
                pair_counts[bp] += 1
                sourceblock_counts[pre_b][bp] += 1
                block_labeled_pair_events += 1
    n_presyn = len(presyn_seen)
    return pair_counts, sourceblock_counts, total_pair_events, block_labeled_pair_events, n_connectors, n_presyn


def write_empirical_null_tables(
    perm_globs: Sequence[str],
    observed_counts: ObservedCounts,
    meta: pd.DataFrame,
    outdir: Path,
    pre_col: str,
    post_col: str,
    residual_pairs: Sequence[str],
    chunksize: Optional[int] = None,
) -> None:
    perm_files: List[str] = []
    for pat in perm_globs:
        perm_files.extend(glob.glob(pat))
    perm_files = sorted(set(perm_files))
    if not perm_files:
        print("No permutation files matched; skipping empirical null analysis.", file=sys.stderr)
        return

    obs_counter = observed_counts.block_pair_counts
    obs_total = sum(obs_counter.values())
    obs_sourceblock = observed_counts.source_block_observed_blockpair
    block_values = sorted(meta["predicted_block"].astype(int).unique())

    null_counters = []
    null_sourceblock = []
    file_rows = []
    for i, f in enumerate(perm_files):
        print(f"Reading permutation {i+1}/{len(perm_files)}: {f}", flush=True)
        pc, psrc, all_pairs, block_pairs, n_conn, n_pre = count_block_pairs_for_connector_table(f, meta, pre_col, post_col, chunksize=chunksize)
        null_counters.append(pc)
        null_sourceblock.append(psrc)
        same = sum(c for bp, c in pc.items() if bp[0] == bp[1])
        file_rows.append(
            {
                "file": os.path.basename(f),
                "connectors_from_block_labeled_presynaptic_neurons": n_conn,
                "unique_presynaptic_neurons": n_pre,
                "all_unique_postsynaptic_pair_events": all_pairs,
                "both_posts_have_predicted_block_pair_events": block_pairs,
                "same_block_pair_events": same,
                "same_block_fraction": safe_div(same, block_pairs),
                "unique_block_pairs": len(pc),
            }
        )
    pd.DataFrame(file_rows).to_csv(outdir / "permutation_file_summaries.csv", index=False)

    all_bps = sorted(set(obs_counter) | set().union(*[set(c) for c in null_counters]))
    rows = []
    n = len(null_counters)
    for bp in all_bps:
        obs_count = obs_counter.get(bp, 0)
        obs_frac = safe_div(obs_count, obs_total)
        null_counts = np.array([c.get(bp, 0) for c in null_counters], dtype=float)
        null_totals = np.array([sum(c.values()) for c in null_counters], dtype=float)
        null_fracs = np.divide(null_counts, null_totals, out=np.zeros_like(null_counts), where=null_totals > 0)
        mean_count = float(null_counts.mean()) if n else np.nan
        sd_count = float(null_counts.std(ddof=1)) if n > 1 else np.nan
        mean_frac = float(null_fracs.mean()) if n else np.nan
        sd_frac = float(null_fracs.std(ddof=1)) if n > 1 else np.nan
        z = (obs_frac - mean_frac) / sd_frac if sd_frac and sd_frac > 0 else np.nan
        high_p = (float(np.sum(null_fracs >= obs_frac)) + 1) / (n + 1)
        low_p = (float(np.sum(null_fracs <= obs_frac)) + 1) / (n + 1)
        rows.append(
            {
                "post_block_pair": f"{bp[0]}+{bp[1]}",
                "block_a": bp[0],
                "block_b": bp[1],
                "observed_count": obs_count,
                "observed_fraction": obs_frac,
                "null_mean_count": mean_count,
                "null_sd_count": sd_count,
                "null_mean_fraction": mean_frac,
                "null_sd_fraction": sd_frac,
                "obs_over_null_fraction": safe_div(obs_frac, mean_frac),
                "z_vs_perm_fraction": z,
                "empirical_high_tail_p_min_1_of_n_plus_1": high_p,
                "empirical_low_tail_p_min_1_of_n_plus_1": low_p,
                "n_nulls_present": n,
                "log2_obs_over_perm": math.log2((obs_frac + 1e-12) / (mean_frac + 1e-12)) if mean_frac == mean_frac else np.nan,
                "same_block": bp[0] == bp[1],
            }
        )
    emp_df = pd.DataFrame(rows)
    emp_df = emp_df.sort_values("log2_obs_over_perm", ascending=False)
    emp_df.to_csv(outdir / "all_block_pairs_vs_empirical_null.csv", index=False)
    emp_df[emp_df["observed_count"] >= 10].sort_values("log2_obs_over_perm", ascending=False).head(30).to_csv(outdir / "top_empirical_enriched_block_pairs.csv", index=False)
    emp_df[emp_df["observed_count"] >= 10].sort_values("log2_obs_over_perm", ascending=True).head(30).to_csv(outdir / "top_empirical_depleted_block_pairs.csv", index=False)

    same_obs = sum(c for bp, c in obs_counter.items() if bp[0] == bp[1])
    same_obs_frac = safe_div(same_obs, obs_total)
    same_null_fracs = []
    for c in null_counters:
        total = sum(c.values())
        same = sum(v for bp, v in c.items() if bp[0] == bp[1])
        same_null_fracs.append(safe_div(same, total))
    same_null_fracs = np.array(same_null_fracs, dtype=float)
    summary = pd.DataFrame(
        [
            {"metric": "n_permutation_files", "value": n},
            {"metric": "observed_block_labeled_pair_events", "value": obs_total},
            {"metric": "observed_same_block_fraction", "value": same_obs_frac},
            {"metric": "permutation_mean_same_block_fraction", "value": float(np.nanmean(same_null_fracs))},
            {"metric": "permutation_sd_same_block_fraction", "value": float(np.nanstd(same_null_fracs, ddof=1)) if n > 1 else np.nan},
            {"metric": "same_block_obs_over_perm", "value": safe_div(same_obs_frac, float(np.nanmean(same_null_fracs)))},
            {"metric": "same_block_z_vs_perm", "value": safe_div(same_obs_frac - float(np.nanmean(same_null_fracs)), float(np.nanstd(same_null_fracs, ddof=1))) if n > 1 else np.nan},
        ]
    )
    summary.to_csv(outdir / "empirical_null_global_summary.csv", index=False)

    # Compare selected residual pairs to empirical null.
    if residual_pairs:
        emp_subset = emp_df[emp_df["post_block_pair"].isin(residual_pairs)].copy()
        emp_subset.to_csv(outdir / "selected_residual_pairs_vs_empirical_null.csv", index=False)

    # Per presynaptic block same-block enrichment vs empirical null.
    sb_rows = []
    for pre_b in block_values:
        obs_c = obs_sourceblock.get(pre_b, Counter())
        obs_t = sum(obs_c.values())
        if obs_t <= 0:
            continue
        obs_same = sum(v for bp, v in obs_c.items() if bp[0] == bp[1])
        obs_frac = obs_same / obs_t
        null_fracs = []
        null_counts = []
        for src in null_sourceblock:
            c = src.get(pre_b, Counter())
            t = sum(c.values())
            if t <= 0:
                continue
            same = sum(v for bp, v in c.items() if bp[0] == bp[1])
            null_fracs.append(same / t)
            null_counts.append(t)
        if not null_fracs:
            continue
        nf = np.array(null_fracs, dtype=float)
        mean = float(nf.mean())
        sd = float(nf.std(ddof=1)) if len(nf) > 1 else np.nan
        sb_rows.append(
            {
                "presynaptic_block": pre_b,
                "observed_pairs": obs_t,
                "observed_same_block_fraction": obs_frac,
                "null_mean_same_block_fraction": mean,
                "null_sd_same_block_fraction": sd,
                "obs_over_null_same_block_fraction": safe_div(obs_frac, mean),
                "z_vs_perm": safe_div(obs_frac - mean, sd) if sd and sd > 0 else np.nan,
                "n_nulls_present": len(nf),
            }
        )
    pd.DataFrame(sb_rows).sort_values("presynaptic_block").to_csv(outdir / "per_presynaptic_block_same_block_vs_empirical_null.csv", index=False)


def zip_outputs(outdir: Path, zip_path: Optional[Path] = None) -> Path:
    if zip_path is None:
        zip_path = outdir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(outdir.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(outdir)))
    return zip_path


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyse co-targeting motifs with predicted block labels.")
    p.add_argument("--connectors", required=True, help="Observed connector CSV.")
    p.add_argument("--blocks", required=True, help="CSV with skeleton_id and predicted_block.")
    p.add_argument("--details", default=None, help="Optional neuron details CSV with names/celltypes/NT.")
    p.add_argument("--permutations", nargs="*", default=[], help="Optional permutation CSV globs, e.g. 'perm_model_short_*.csv'.")
    p.add_argument("--outdir", default="cotarget_block_outputs", help="Output directory.")
    p.add_argument("--zip", action="store_true", help="Zip the output directory at the end.")
    p.add_argument("--top-n", type=int, default=10, help="Number of strongest positive and negative residual pairs to summarize.")
    p.add_argument("--chunksize", type=int, default=0, help="Optional pandas read_csv chunk size; 0 reads whole file.")

    # Column names.
    p.add_argument("--connector-id-col", default="connector_id")
    p.add_argument("--pre-col", default="presynaptic_to")
    p.add_argument("--post-col", default="postsynaptic_to")
    p.add_argument("--skeleton-col", default="skeleton_id")
    p.add_argument("--block-col", default="predicted_block")
    p.add_argument("--name-col", default="name")
    p.add_argument("--celltype-col", default="celltype")
    p.add_argument("--nt-col", default="predicted_neurotransmitter")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    chunksize = args.chunksize if args.chunksize and args.chunksize > 0 else None

    print("Reading metadata...", flush=True)
    meta = read_metadata(args.blocks, args.details, args.skeleton_col, args.block_col, args.name_col, args.celltype_col, args.nt_col)
    meta.to_csv(outdir / "merged_neuron_metadata_used.csv", index=False)
    block_values = sorted(meta["predicted_block"].astype(int).unique())
    pre_block_map = dict(zip(meta["skeleton_id"].astype(int), meta["predicted_block"].astype(int)))

    print("Writing block glossary...", flush=True)
    write_block_glossary(meta, outdir)

    print("Accumulating observed co-target counts...", flush=True)
    counts = accumulate_observed_counts(args.connectors, meta, args.pre_col, args.post_col, args.connector_id_col, chunksize=chunksize)

    print("Writing compression/observed tables...", flush=True)
    write_compression_tables(counts, outdir)

    print("Computing per-source abundance null...", flush=True)
    enrich_df, null_summary, sb_df, _, _ = compute_abundance_null(counts, pre_block_map, block_values)
    enrich_df.to_csv(outdir / "block_pair_enrichment_vs_per_source_abundance_null.csv", index=False)
    null_summary.to_csv(outdir / "abundance_null_summary.csv", index=False)
    sb_df.to_csv(outdir / "per_presynaptic_block_abundance_null_summary.csv", index=False)

    print("Computing celltype-corrected residuals...", flush=True)
    residual_df, residual_summary = compute_celltype_corrected_residuals(counts)
    residual_df.to_csv(outdir / "residual_block_pair_enrichment_after_celltype_abundance.csv", index=False)
    residual_summary.to_csv(outdir / "residual_after_celltype_summary.csv", index=False)

    print("Summarizing strongest residual pair drivers...", flush=True)
    pos_pairs, neg_pairs = select_residual_pairs(residual_df, top_n=args.top_n, min_observed=10)
    selected_pairs = pos_pairs + [p for p in neg_pairs if p not in pos_pairs]
    pd.DataFrame(
        [{"set": "positive", "post_block_pair": p} for p in pos_pairs]
        + [{"set": "negative", "post_block_pair": p} for p in neg_pairs]
    ).to_csv(outdir / "selected_residual_pairs_for_driver_summary.csv", index=False)
    write_residual_pair_drivers(args.connectors, meta, residual_df, selected_pairs, outdir, args.pre_col, args.post_col, top_n_pairs_label=args.top_n, chunksize=chunksize)

    if args.permutations:
        print("Computing empirical permutation/null comparison...", flush=True)
        write_empirical_null_tables(args.permutations, counts, meta, outdir, args.pre_col, args.post_col, selected_pairs, chunksize=chunksize)

    if args.zip:
        zip_path = zip_outputs(outdir)
        print(f"Wrote zip: {zip_path}", flush=True)
    print(f"Done. Outputs in: {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
