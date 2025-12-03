"""
W&B plotting utilities with flexible config filtering, caching, and concise labels.
"""
from __future__ import annotations

import os
import itertools
import json
import hashlib
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import wandb


# -------- Filter expansion --------

def expand_config_filter(config_filter: Dict[str, Any]):
    """Yield all combinations for possibly list-valued config entries."""
    keys = list(config_filter.keys())
    values_lists = []
    for k in keys:
        v = config_filter[k]
        if isinstance(v, (list, tuple, set)):
            values_lists.append(list(v))
        else:
            values_lists.append([v])
    for combo in itertools.product(*values_lists):
        yield {k: c for k, c in zip(keys, combo)}


def filter_label(d: Dict[str, Any], sep: str = ", ", kv: str = "=") -> str:
    parts = []
    for k, v in d.items():
        parts.append(f"{k}{kv}{v}")
    return sep.join(parts)


# -------- W&B fetching --------

def fetch_runs(api: wandb.Api, entity: str, project: str, cf: Dict[str, Any]):
    filters = {f"config.{k}": v for k, v in cf.items()}
    return api.runs(f"{entity}/{project}", filters=filters)


def collect_metric_arrays(runs, metric: str, x_axis: str = "epoch"):
    """Collect metric arrays from runs.
    
    Args:
        runs: W&B runs to collect from
        metric: Metric key to collect
        x_axis: X-axis variable name ("epoch" or "step")
    
    Returns:
        Tuple of (array, x_values) or None if no data
    """
    series = []
    lengths = []
    for run in runs:
        try:
            hist = run.history(keys=[metric, x_axis])  # dataframe
            hist = hist.sort_values(x_axis)
            vals = hist[metric].to_numpy()
            series.append(vals)
            lengths.append(len(vals))
        except Exception as e:
            print(f"Warning: failed to read history for run {getattr(run, 'name', 'unknown')}: {e}")
    if not series:
        return None
    max_len = max(lengths)
    arr = np.array([np.pad(v, (0, max_len - len(v)), constant_values=np.nan) for v in series])
    x_values = np.arange(1, max_len + 1)
    return arr, x_values


# -------- Caching helpers --------

def _stable_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def make_cache_key(entity: str, project: str, cf: Dict[str, Any], metric: str) -> str:
    payload = {"entity": entity, "project": project, "metric": metric, "config": cf}
    s = _stable_dumps(payload)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def get_cache_file(cache_dir: str, key: str) -> str:
    return os.path.join(cache_dir, f"{key}.npz")


def load_cached_arrays(path: str):
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path, allow_pickle=True)
        return data["arr"], data["epochs"]
    except Exception as e:
        print(f"Warning: failed to load cache {path}: {e}")
        return None


def save_cached_arrays(path: str, arr: np.ndarray, epochs: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        np.savez_compressed(path, arr=arr, epochs=epochs)
    except Exception as e:
        print(f"Warning: failed to save cache {path}: {e}")


# -------- Project-wide config options caching --------

def get_config_options_cache_path(cache_dir: str, entity: str, project: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    name = hashlib.sha1(f"{entity}/{project}".encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"config_options_{name}.json")


def fetch_all_config_options(
    api: wandb.Api,
    entity: str,
    project: str,
    config_keys: Optional[List[str]] = None,
    max_runs: Optional[int] = None,
) -> Dict[str, List[Any]]:
    """Scan runs in a project and collect unique values per config key.

    Args:
        api: W&B API instance
        entity, project: W&B identifiers
        config_keys: if provided, only collect these keys; otherwise collect all seen keys
        max_runs: optional limit of runs to scan for speed
    Returns:
        Dict mapping config key -> sorted list of unique values (JSON-serializable)
    """
    runs_iter = api.runs(f"{entity}/{project}")

    uniques: Dict[str, set] = {}
    count = 0
    for run in runs_iter:
        cfg = getattr(run, "config", {}) or {}
        keys = config_keys if config_keys is not None else list(cfg.keys())
        for k in keys:
            if k not in cfg:
                continue
            v = cfg[k]
            # Ensure hashable/JSON-safe representation for set membership
            try:
                json.dumps(v)
                key_val = ("json", json.dumps(v, sort_keys=True))
            except TypeError:
                # Fallback to string repr if not JSON-serializable
                key_val = ("repr", repr(v))
            uniques.setdefault(k, set()).add(key_val)
        count += 1
        if max_runs is not None and count >= max_runs:
            break

    # Convert back to user-friendly values (prefer JSON decoded when available)
    result: Dict[str, List[Any]] = {}
    for k, s in uniques.items():
        values: List[Any] = []
        for tag, stored in s:
            if tag == "json":
                try:
                    values.append(json.loads(stored))
                except Exception:
                    values.append(stored)
            else:
                values.append(stored)
        # Sort values for stable presentation (stringify for mixed types)
        result[k] = sorted(values, key=lambda x: json.dumps(x, sort_keys=True, default=str))
    return result


def load_config_options_cache(path: str) -> Optional[Dict[str, List[Any]]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: failed to load config options cache {path}: {e}")
        return None


def save_config_options_cache(path: str, options: Dict[str, List[Any]]):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(options, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: failed to save config options cache {path}: {e}")


def update_and_get_config_options(
    api: wandb.Api,
    entity: str,
    project: str,
    cache_dir: str,
    force_refresh: bool = False,
    config_keys: Optional[List[str]] = None,
    max_runs: Optional[int] = None,
) -> Dict[str, List[Any]]:
    """Load cached config options or fetch from W&B and update cache."""
    path = get_config_options_cache_path(cache_dir, entity, project)
    if not force_refresh:
        cached = load_config_options_cache(path)
        if cached is not None:
            return cached
    options = fetch_all_config_options(api, entity, project, config_keys, max_runs)
    save_config_options_cache(path, options)
    return options


def print_config_options(
    options: Dict[str, List[Any]],
    only_keys: Optional[List[str]] = None,
    min_values: int = 1,
):
    """Print config options, optionally filtering to keys with at least min_values options.

    Args:
        options: mapping from config key to list of possible values
        only_keys: if provided, restrict output to this subset of keys
        min_values: only print keys that have at least this many distinct values
    """
    keys = sorted(options.keys())
    if only_keys is not None:
        keys = [k for k in keys if k in only_keys]
    for k in keys:
        vals = options.get(k, [])
        if len(vals) < min_values:
            continue
        print(f"{k}:")
        for v in vals:
            print(f"  - {v}")


# -------- Label helpers for concise legends --------

def determine_varying_keys(combos: List[Dict[str, Any]]) -> List[str]:
    """Return the list of keys whose values differ across combos (preserve key order from first combo)."""
    if not combos:
        return []
    keys = list(combos[0].keys())
    varying = []
    for k in keys:
        values = {c.get(k) for c in combos}
        if len(values) > 1:
            varying.append(k)
    return varying


def short_label(cf: Dict[str, Any], varying_keys: List[str]) -> str:
    """Build a short label showing only differing options.
    - If one key varies -> show only its value.
    - If multiple keys vary -> show key=value pairs for those keys.
    - If none vary -> return empty string (caller may fallback).
    """
    if len(varying_keys) == 0:
        return ""
    if len(varying_keys) == 1:
        k = varying_keys[0]
        return f"{cf[k]}"
    # multiple varying keys
    parts = [f"{k}={cf[k]}" for k in varying_keys]
    return ", ".join(parts)


# -------- Statistics and plotting --------

def compute_envelope(arr: np.ndarray, range_type: str):
    mean = np.nanmean(arr, axis=0)
    if range_type == "minmax":
        lower = np.nanmin(arr, axis=0)
        upper = np.nanmax(arr, axis=0)
    elif range_type == "std":
        std = np.nanstd(arr, axis=0)
        lower = mean - std
        upper = mean + std
    elif range_type == "stderr":
        stderr = np.nanstd(arr, axis=0) / np.sqrt(arr.shape[0])
        lower = mean - stderr
        upper = mean + stderr
    else:  # "none"
        lower = upper = mean
    return mean, lower, upper


def _to_safe_str(x: Any) -> str:
    try:
        s = json.dumps(x, sort_keys=True)
    except Exception:
        s = str(x)
    # make filesystem-safe: keep alnum and a small set of symbols
    safe = []
    for ch in s:
        if ch.isalnum() or ch in ['-', '_', '.', ',', '=', '[', ']', '+']:
            safe.append(ch)
        elif ch in [' ', '/']:
            safe.append('-')
        else:
            # drop other characters
            continue
    return ''.join(safe)


def build_plot_filename(
    base_filter: Dict[str, Any],
    metric: str,
    range_type: str,
    color_by_key: Optional[str] = None,
    prefix: Optional[str] = None,
    max_len: int = 180,
) -> str:
    """Build a deterministic, config-specific filename for a plot.

    The name encodes metric, range, optional color key, and the CONFIG_FILTER
    entries (list values included). It is sanitized for filesystem safety and
    truncated when too long.
    """
    parts: List[str] = []
    if prefix:
        parts.append(_to_safe_str(prefix))
    parts.append(f"metric={_to_safe_str(metric)}")
    parts.append(f"range={_to_safe_str(range_type)}")
    if color_by_key:
        parts.append(f"colorby={_to_safe_str(color_by_key)}")

    # include filter entries in sorted key order for stability
    for k in sorted(base_filter.keys()):
        v = base_filter[k]
        if isinstance(v, (list, tuple, set)):
            vals = sorted(list(v), key=lambda x: _to_safe_str(x))
            parts.append(f"{_to_safe_str(k)}=[{','.join(_to_safe_str(x) for x in vals)}]")
        else:
            parts.append(f"{_to_safe_str(k)}={_to_safe_str(v)}")

    name = "__".join(parts)
    if len(name) > max_len:
        # keep a hash suffix to ensure uniqueness
        h = hashlib.sha1(name.encode('utf-8')).hexdigest()[:8]
        name = name[: max_len - 9] + '_' + h
    return name


def compute_global_metric_range(
    api: wandb.Api,
    entity: str,
    project: str,
    config_list: List[Dict[str, Any]],
    metric: str = "val_loss",
    range_type: str = "std",
    use_cache: bool = True,
    refresh_cache: bool = False,
    cache_dir: str = "cache",
    margin: float = 0.05,
    x_axis: str = "epoch",
) -> Tuple[float, float]:
    """Compute global min/max across multiple configurations for consistent y-axis.
    
    Args:
        api: W&B API instance
        entity, project: W&B identifiers
        config_list: List of config dictionaries to scan
        metric: Metric to analyze
        range_type: How to compute envelope ("std", "stderr", "minmax", "none")
        use_cache, refresh_cache, cache_dir: Caching controls
        margin: Extra margin to add as fraction of range (default 5%)
        x_axis: X-axis variable name ("epoch" or "step")
    
    Returns:
        (global_min, global_max) tuple for y-axis limits
    """
    global_min = float('inf')
    global_max = float('-inf')
    
    for cf in config_list:
        # Try to load cached data
        arr_epochs = None
        if use_cache and not refresh_cache:
            key = make_cache_key(entity, project, cf, metric)
            cpath = get_cache_file(cache_dir, key)
            arr_epochs = load_cached_arrays(cpath)
            if arr_epochs is not None:
                arr, epochs = arr_epochs
        
        # Fetch if not cached
        if arr_epochs is None:
            runs = fetch_runs(api, entity, project, cf)
            if len(runs) == 0:
                continue
            out = collect_metric_arrays(runs, metric, x_axis)
            if out is None:
                continue
            arr, epochs = out
            if use_cache:
                key = make_cache_key(entity, project, cf, metric)
                cpath = get_cache_file(cache_dir, key)
                save_cached_arrays(cpath, arr, epochs)
        
        # Compute envelope and update global range
        mean, lower, upper = compute_envelope(arr, range_type)
        global_min = min(global_min, np.nanmin(lower))
        global_max = max(global_max, np.nanmax(upper))
    
    if global_min == float('inf') or global_max == float('-inf'):
        return None, None
    
    # Add margin
    range_span = global_max - global_min
    global_min -= margin * range_span
    global_max += margin * range_span
    
    return global_min, global_max


def plot_metric_over_epochs(
    api: wandb.Api,
    entity: str,
    project: str,
    base_filter: Dict[str, Any],
    metric: str = "val_loss",
    range_type: str = "std",
    include_fill: bool = True,
    alpha: float = 0.25,
    linewidth: float = 2.0,
    figsize: Tuple[float, float] = (8, 5),
    title: str | None = None,
    grid: bool = True,
    show_legend: bool = True,
    # color mapping
    color_by_key: Optional[str] = None,
    color_palette: Optional[List[str]] = None,
    # label mapping
    label_map: Optional[Dict[str, str]] = None,
    # font sizes for publication
    fontsize_title: int = 14,
    fontsize_labels: int = 12,
    fontsize_ticks: int = 10,
    fontsize_legend: int = 10,
    # axis controls
    ylim: Optional[Tuple[float, float]] = None,
    x_axis: str = "epoch",
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    # caching
    use_cache: bool = True,
    refresh_cache: bool = False,
    cache_dir: str = "cache",
):
    combos = list(expand_config_filter(base_filter))
    if len(combos) == 0:
        raise ValueError("CONFIG_FILTER produced no combinations.")

    varying_keys = determine_varying_keys(combos)

    # Build a consistent color map if requested
    value_to_color: Dict[Any, str] = {}
    if color_by_key is not None:
        # Collect sorted unique values for deterministic color assignment
        values = sorted({cf.get(color_by_key) for cf in combos if color_by_key in cf})
        if color_palette is None:
            default_cycle = plt.rcParams.get('axes.prop_cycle')
            if default_cycle is not None:
                palette = default_cycle.by_key().get('color', [])
            else:
                palette = []
            if not palette:
                # fallback palette
                palette = [
                    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
                ]
        else:
            palette = list(color_palette)
        # Assign colors in order
        for i, v in enumerate(values):
            value_to_color[v] = palette[i % len(palette)]

    plt.figure(figsize=figsize)
    plotted_any = False

    for i, cf in enumerate(combos):
        arr_epochs = None
        if use_cache and not refresh_cache:
            key = make_cache_key(entity, project, cf, metric)
            cpath = get_cache_file(cache_dir, key)
            if not os.path.exists(cpath):
                print(f"[cache] Missing: {cpath}. Fetching from server and caching for future runs.")
            arr_epochs = load_cached_arrays(cpath)
            if arr_epochs is not None:
                arr, epochs = arr_epochs
            elif os.path.exists(cpath):
                print(f"[cache] Failed to load existing cache at {cpath}. Refetching from server.")
        if arr_epochs is None:
            runs = fetch_runs(api, entity, project, cf)
            if len(runs) == 0:
                print(f"No runs for {cf}. Skipping.")
                continue
            out = collect_metric_arrays(runs, metric, x_axis)
            if out is None:
                print(f"No metric data for {cf}. Skipping.")
                continue
            arr, epochs = out
            if use_cache:
                key = make_cache_key(entity, project, cf, metric)
                cpath = get_cache_file(cache_dir, key)
                save_cached_arrays(cpath, arr, epochs)
                print(f"[cache] Saved: {cpath}")

        mean, lower, upper = compute_envelope(arr, range_type)

        label = short_label(cf, varying_keys)
        if not label.strip():
            # Fallback to full config label when there is only one combo
            label = filter_label(cf)

        # Apply label mapping if provided
        if label_map is not None:
            # First, try to match the entire label (for complex multi-key labels)
            if label in label_map:
                label = label_map[label]
            # If color_by_key is specified, try to replace individual values
            elif color_by_key is not None and color_by_key in cf:
                original_value = cf[color_by_key]
                if original_value in label_map:
                    # If the label is just the value (single varying key), replace entirely
                    if label == str(original_value):
                        label = label_map[original_value]
                    else:
                        # If it's a complex label with multiple keys, replace the value portion
                        label = label.replace(str(original_value), label_map[original_value])

        # Determine color
        if color_by_key is not None and color_by_key in cf and cf[color_by_key] in value_to_color:
            color = value_to_color[cf[color_by_key]]
        else:
            color = None  # let matplotlib decide

        if color is not None:
            line, = plt.plot(epochs, mean, label=label, linewidth=linewidth, color=color)
        else:
            line, = plt.plot(epochs, mean, label=label, linewidth=linewidth)
            color = line.get_color()
        if include_fill and range_type != "none":
            plt.fill_between(epochs, lower, upper, color=color, alpha=alpha)
        plotted_any = True

    if not plotted_any:
        raise ValueError("No data plotted. Check filters and metric.")

    # Set x-axis label (default based on x_axis parameter or use custom)
    if x_label is None:
        x_label = x_axis.capitalize()
    plt.xlabel(x_label, fontsize=fontsize_labels)
    
    # Set y-axis label (use custom or format metric name)
    if y_label is None:
        y_label = metric.replace("_", " ").title()
    plt.ylabel(y_label, fontsize=fontsize_labels)
    
    if title:
        plt.title(title, fontsize=fontsize_title)
    if grid:
        plt.grid(True, alpha=0.3)
    if show_legend:
        plt.legend(fontsize=fontsize_legend)
    
    # Set y-axis limits if provided
    if ylim is not None:
        plt.ylim(ylim)
    
    # Set tick label sizes
    plt.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
    
    plt.tight_layout()


# -------- Saving --------

def save_current_figure(
    save_dir: str,
    save_name: str,
    save_png: bool = False,
    save_pdf: bool = True,
    timestamp: bool = False,
):
    os.makedirs(save_dir, exist_ok=True)
    base = save_name
    if timestamp:
        base = f"{base}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    path_base = os.path.join(save_dir, base)
    if save_png:
        plt.savefig(f"{path_base}.png", dpi=200, bbox_inches="tight")
    if save_pdf:
        plt.savefig(f"{path_base}.pdf", bbox_inches="tight")


__all__ = [
    "plot_metric_over_epochs",
    "save_current_figure",
    "build_plot_filename",
    "compute_global_metric_range",
    # Expose internals if needed by power users:
    "expand_config_filter",
    "filter_label",
    "fetch_runs",
    "collect_metric_arrays",
    "make_cache_key",
    "get_cache_file",
    "load_cached_arrays",
    "save_cached_arrays",
    "determine_varying_keys",
    "short_label",
    "compute_envelope",
    # Config options utilities
    "get_config_options_cache_path",
    "fetch_all_config_options",
    "load_config_options_cache",
    "save_config_options_cache",
    "update_and_get_config_options",
    "print_config_options",
]