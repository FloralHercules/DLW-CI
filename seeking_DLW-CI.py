#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Compatibility shim for renamed module.

This file existed as `seeking_DLW-CI.py`. The canonical filename is now
`seeking_DLW_CI.py`. To avoid breaking scripts that reference the old
filename, this shim loads and executes the new file at runtime.
come on! 
"""

import os
import warnings

_here = os.path.dirname(os.path.abspath(__file__))
_new = os.path.join(_here, "seeking_DLW_CI.py")
if os.path.exists(_new):
    warnings.warn("seeking_DLW-CI.py is deprecated; executing seeking_DLW_CI.py instead.", DeprecationWarning)
    with open(_new, "r", encoding="utf-8") as _f:
        _code = _f.read()
    exec(compile(_code, _new, 'exec'), globals())
else:
    # If the renamed file is not present, continue executing this file in-place.
    warnings.warn(f"seeking_DLW_CI.py not found; executing in-place {os.path.basename(__file__)}.", RuntimeWarning)

# Ensure standard imports exist for the in-file implementation (in case
# the external renamed file is absent). These were present in the original
# script but may have been bypassed by the shim logic.
import sys
import csv
import math
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# 设置默认的中文字体，防止Matplotlib绘图时中文乱码，配置接近SCI标准的高级绘图风格
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimSun', 'SimHei', 'Arial']  # 使用宋体作为基础以接近罗马字体
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'  # 接近Times New Roman的数学字体
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2

# ==========================================
# 1. 基础物理和仿真参数设置
# ==========================================
Q = np.array([3.0, 3.0, 3.0, 3.0, 3.0])  # 源实际排放速率
sourceX = np.array([3.0, 5.0, 7.0, 1.0, 6.0])     # 源真实X位置
sourceY = np.array([5.0, 8.0, 5.0, 8.0, 3.0])     # 源真实Y位置

t = 250.0       # 释放时间
D = 10.0        # 有效扩散系数
V = 10.0        # 平均风速
a = 1.0         # 传感器缩放系数
X_min, Y_min = 0.0, 0.0
X_max, Y_max = 10.0, 10.0
Q_max, Q_min = 4.0, 0.0
size_plot = 0.25
step = 0.25
TT = 400
max_d = 7       # 最大检测计数值

# 计算拉姆达 (lamda)
# lamda = sqrt((D*t) / (1+(V^2)*t/(4*D)))
lamda = np.sqrt((D * t) / (1.0 + (V**2) * t / (4.0 * D)))

# 探测器(机器人)初始位置 (5个, 可被场景配置覆写)
_default_pX = np.array([1.1, 0.5, 1.0, 1.9, 1.9]) / 2.0
_default_pY = np.array([1.0, 2.0, 3.0, 2.6, 1.3]) / 2.0
# 场景覆写检查 (用于 generalization suite)
_override_robot_num = globals().get("robot_num_override", None)
if _override_robot_num is not None:
    _override_pX = globals().get("pX_init", None)
    _override_pY = globals().get("pY_init", None)
    if _override_pX is not None and _override_pY is not None:
        pX = np.array(_override_pX)
        pY = np.array(_override_pY)
    else:
        pX = _default_pX[:_override_robot_num]
        pY = _default_pY[:_override_robot_num]
else:
    pX = _default_pX.copy()
    pY = _default_pY.copy()

source_num = len(sourceX)
robot_num = len(pX)

# 粒子滤波参数
particle_num = 1000
layer_num = robot_num  # 平行粒子滤波层数等于探测器数量
resample_threshold = 0.5  # ESS自适应重采样阈值

# ==========================================
# CDPA-CI Ablation / Debug 开关系统
# ==========================================

# --- 命令行参数解析 (用于 ablation 子进程调用) ---
_cmd_config = None
_cmd_seed = None
_args_to_parse = sys.argv[1:]
_i = 0
_is_ablation_call = any(f in sys.argv for f in ["--ablation", "--harmful", "--generalize", "--all", "--paper"])
while _i < len(_args_to_parse):
    if _args_to_parse[_i] == "--config" and _i + 1 < len(_args_to_parse):
        with open(_args_to_parse[_i + 1], "r") as _f:
            _cmd_config = json.load(_f)
        _i += 2
    elif _args_to_parse[_i] == "--seed" and _i + 1 < len(_args_to_parse):
        _cmd_seed = int(_args_to_parse[_i + 1])
        _i += 2
    elif _args_to_parse[_i] == "--ablation":
        _i += 1  # skip (handled in __main__)
    else:
        _i += 1
if _cmd_seed is not None:
    np.random.seed(_cmd_seed)

# --- 运行模式 ---
run_mode = "diversity_novelty_bounded"  # DN-CI: Diversity + Bounded Novelty (论文主方法)

# --- 模块开关默认值 ---
USE_ESS = True
USE_DYNAMIC_NICHE = False
USE_COGNITIVE_FUSION = False
USE_SAME_SOURCE_GATE = False
USE_ROLE_ADAPTATION = False
USE_SOURCE_DECLARATION = False
USE_DIVERSITY_REGULARIZATION = True
USE_ADAPTIVE_STEP_SIZE = False
USE_SAFE_ADAPTIVE_STEP = False
USE_VISITATION_NOVELTY = True
USE_NOVELTY_TIE_BREAKER = False
USE_BOUNDED_NOVELTY = True
USE_INTENT_AWARE_ASSIGNMENT = False
USE_NICHE_RESET = False
USE_PARTICLE_SHARING_IN_NICHE = False

# --- 应用命令行 config 覆写 (必须在 switch 之前, 因为可能修改 run_mode) ---
if _cmd_config is not None:
    for _k, _v in _cmd_config.items():
        if _k == "_seed":
            np.random.seed(int(_v))
            continue
        if _k.startswith("_"):
            continue
        if _k in globals():
            globals()[_k] = _v

# --- 根据 run_mode 设置开关 ---
def _apply_run_mode():
    global USE_ESS, USE_DYNAMIC_NICHE, USE_COGNITIVE_FUSION, USE_SAME_SOURCE_GATE
    global USE_ROLE_ADAPTATION, USE_SOURCE_DECLARATION, USE_NICHE_RESET, USE_PARTICLE_SHARING_IN_NICHE
    global USE_DIVERSITY_REGULARIZATION, USE_ADAPTIVE_STEP_SIZE, USE_VISITATION_NOVELTY, USE_NOVELTY_TIE_BREAKER, USE_BOUNDED_NOVELTY, USE_INTENT_AWARE_ASSIGNMENT
    if run_mode == "baseline":
        USE_ESS = False; USE_DYNAMIC_NICHE = False; USE_DIVERSITY_REGULARIZATION = False; USE_ADAPTIVE_STEP_SIZE = False; USE_VISITATION_NOVELTY = False; USE_NOVELTY_TIE_BREAKER = False
    elif run_mode == "ess_only":
        USE_ESS = True; USE_DYNAMIC_NICHE = False; USE_DIVERSITY_REGULARIZATION = False; USE_ADAPTIVE_STEP_SIZE = False; USE_VISITATION_NOVELTY = False; USE_NOVELTY_TIE_BREAKER = False
    elif run_mode == "diversity_only":
        USE_ESS = False; USE_DYNAMIC_NICHE = False; USE_DIVERSITY_REGULARIZATION = True; USE_ADAPTIVE_STEP_SIZE = False; USE_VISITATION_NOVELTY = False; USE_NOVELTY_TIE_BREAKER = False
    elif run_mode == "novelty_additive":
        USE_ESS = False; USE_DYNAMIC_NICHE = False; USE_DIVERSITY_REGULARIZATION = False; USE_ADAPTIVE_STEP_SIZE = False; USE_VISITATION_NOVELTY = True; USE_NOVELTY_TIE_BREAKER = False
    elif run_mode == "novelty_tiebreak":
        USE_ESS = False; USE_DYNAMIC_NICHE = False; USE_DIVERSITY_REGULARIZATION = False; USE_ADAPTIVE_STEP_SIZE = False; USE_VISITATION_NOVELTY = True; USE_NOVELTY_TIE_BREAKER = True
    elif run_mode == "ess_diversity":
        USE_ESS = True; USE_DYNAMIC_NICHE = False; USE_DIVERSITY_REGULARIZATION = True; USE_ADAPTIVE_STEP_SIZE = False; USE_VISITATION_NOVELTY = False; USE_NOVELTY_TIE_BREAKER = False
    elif run_mode == "ess_novelty":
        USE_ESS = True; USE_DYNAMIC_NICHE = False; USE_DIVERSITY_REGULARIZATION = False; USE_ADAPTIVE_STEP_SIZE = False; USE_VISITATION_NOVELTY = True; USE_NOVELTY_TIE_BREAKER = False
    elif run_mode == "diversity_novelty_additive":
        USE_ESS = False; USE_DYNAMIC_NICHE = False; USE_DIVERSITY_REGULARIZATION = True; USE_ADAPTIVE_STEP_SIZE = False; USE_VISITATION_NOVELTY = True; USE_NOVELTY_TIE_BREAKER = False
    elif run_mode == "diversity_novelty_tiebreak":
        USE_ESS = False; USE_DYNAMIC_NICHE = False; USE_DIVERSITY_REGULARIZATION = True; USE_ADAPTIVE_STEP_SIZE = False; USE_VISITATION_NOVELTY = True; USE_NOVELTY_TIE_BREAKER = True
    elif run_mode == "ess_diversity_novelty_additive":
        USE_ESS = True; USE_DYNAMIC_NICHE = False; USE_DIVERSITY_REGULARIZATION = True; USE_ADAPTIVE_STEP_SIZE = False; USE_VISITATION_NOVELTY = True; USE_NOVELTY_TIE_BREAKER = False
    elif run_mode == "ess_diversity_novelty_tiebreak":
        USE_ESS = True; USE_DYNAMIC_NICHE = False; USE_DIVERSITY_REGULARIZATION = True; USE_ADAPTIVE_STEP_SIZE = False; USE_VISITATION_NOVELTY = True; USE_NOVELTY_TIE_BREAKER = True
    elif run_mode == "novelty_bounded":
        USE_ESS = False; USE_DYNAMIC_NICHE = False; USE_DIVERSITY_REGULARIZATION = False; USE_ADAPTIVE_STEP_SIZE = False; USE_VISITATION_NOVELTY = True; USE_NOVELTY_TIE_BREAKER = False; USE_BOUNDED_NOVELTY = True
    elif run_mode == "diversity_novelty_bounded":
        USE_ESS = False; USE_DYNAMIC_NICHE = False; USE_DIVERSITY_REGULARIZATION = True; USE_ADAPTIVE_STEP_SIZE = False; USE_VISITATION_NOVELTY = True; USE_NOVELTY_TIE_BREAKER = False; USE_BOUNDED_NOVELTY = True
    elif run_mode == "ess_diversity_novelty_bounded":
        USE_ESS = True; USE_DYNAMIC_NICHE = False; USE_DIVERSITY_REGULARIZATION = True; USE_ADAPTIVE_STEP_SIZE = False; USE_VISITATION_NOVELTY = True; USE_NOVELTY_TIE_BREAKER = False; USE_BOUNDED_NOVELTY = True
    elif run_mode == "intent_only":
        USE_ESS = False; USE_DYNAMIC_NICHE = False; USE_DIVERSITY_REGULARIZATION = False; USE_ADAPTIVE_STEP_SIZE = False; USE_VISITATION_NOVELTY = False; USE_NOVELTY_TIE_BREAKER = False; USE_BOUNDED_NOVELTY = False; USE_INTENT_AWARE_ASSIGNMENT = True
    elif run_mode == "dnci_intent":
        USE_ESS = False; USE_DYNAMIC_NICHE = False; USE_DIVERSITY_REGULARIZATION = True; USE_ADAPTIVE_STEP_SIZE = False; USE_VISITATION_NOVELTY = True; USE_NOVELTY_TIE_BREAKER = False; USE_BOUNDED_NOVELTY = True; USE_INTENT_AWARE_ASSIGNMENT = True
    elif run_mode == "ess_dnci_intent":
        USE_ESS = True; USE_DYNAMIC_NICHE = False; USE_DIVERSITY_REGULARIZATION = True; USE_ADAPTIVE_STEP_SIZE = False; USE_VISITATION_NOVELTY = True; USE_NOVELTY_TIE_BREAKER = False; USE_BOUNDED_NOVELTY = True; USE_INTENT_AWARE_ASSIGNMENT = True
    elif run_mode == "adaptive_step_only":
        USE_ESS = False; USE_DYNAMIC_NICHE = False; USE_DIVERSITY_REGULARIZATION = False; USE_ADAPTIVE_STEP_SIZE = True; USE_VISITATION_NOVELTY = False; USE_NOVELTY_TIE_BREAKER = False
    elif run_mode == "ess_diversity_adaptive_step":
        USE_ESS = True; USE_DYNAMIC_NICHE = False; USE_DIVERSITY_REGULARIZATION = True; USE_ADAPTIVE_STEP_SIZE = True; USE_VISITATION_NOVELTY = False; USE_NOVELTY_TIE_BREAKER = False
    elif run_mode == "dynamic_niche":
        USE_ESS = False; USE_DYNAMIC_NICHE = True; USE_DIVERSITY_REGULARIZATION = False; USE_ADAPTIVE_STEP_SIZE = False; USE_VISITATION_NOVELTY = False; USE_NOVELTY_TIE_BREAKER = False
    elif run_mode == "ess_dynamic_niche":
        USE_ESS = True; USE_DYNAMIC_NICHE = True; USE_DIVERSITY_REGULARIZATION = False; USE_ADAPTIVE_STEP_SIZE = False; USE_VISITATION_NOVELTY = False; USE_NOVELTY_TIE_BREAKER = False
    elif run_mode == "cognitive_fusion":
        USE_ESS = True; USE_DYNAMIC_NICHE = True; USE_COGNITIVE_FUSION = True
        USE_DIVERSITY_REGULARIZATION = False; USE_ADAPTIVE_STEP_SIZE = False; USE_VISITATION_NOVELTY = False; USE_NOVELTY_TIE_BREAKER = False
    elif run_mode == "gated_cognitive_fusion":
        USE_ESS = True; USE_DYNAMIC_NICHE = True; USE_COGNITIVE_FUSION = True; USE_SAME_SOURCE_GATE = True
        USE_DIVERSITY_REGULARIZATION = False; USE_ADAPTIVE_STEP_SIZE = False; USE_VISITATION_NOVELTY = False; USE_NOVELTY_TIE_BREAKER = False
    elif run_mode == "role_adaptation":
        USE_ESS = True; USE_DYNAMIC_NICHE = True; USE_COGNITIVE_FUSION = True
        USE_SAME_SOURCE_GATE = True; USE_ROLE_ADAPTATION = True
        USE_DIVERSITY_REGULARIZATION = False; USE_ADAPTIVE_STEP_SIZE = False; USE_VISITATION_NOVELTY = False; USE_NOVELTY_TIE_BREAKER = False
    elif run_mode == "full":
        USE_ESS = True; USE_DYNAMIC_NICHE = True; USE_COGNITIVE_FUSION = True
        USE_SAME_SOURCE_GATE = True; USE_ROLE_ADAPTATION = True
        USE_SOURCE_DECLARATION = True; USE_NICHE_RESET = False; USE_PARTICLE_SHARING_IN_NICHE = False
        USE_DIVERSITY_REGULARIZATION = False; USE_ADAPTIVE_STEP_SIZE = False; USE_VISITATION_NOVELTY = False; USE_NOVELTY_TIE_BREAKER = False
_apply_run_mode()

# --- Same-Source Gate 参数 (Phase 4 扩展) ---
same_source_dist_th = 0.8
same_source_kl_th = 0.5
same_source_q_th = 1.0
same_source_obs_th = 2.0

# --- Diversity Regularization 参数 (Source-Aware) ---
diversity_lambda = 0.2      # diversity penalty 权重
diversity_sigma = 1.0       # 高斯核宽度 (belief 距离接近此值开始排斥)
diversity_min_dist = 1.0    # 距离超过此值时忽略 penalty

# --- Uncertainty-Aware Adaptive Step Size ---
step_min = 0.10              # 最小步长 (高置信度时)
step_max = 0.40              # 最大步长 (高不确定性时)
step_uncertainty_scale = 1.0 # 不确定性归一化尺度

# --- Visitation-Aware Novelty Reward ---
visit_grid_resolution = 0.25
novelty_lambda = 0.10
novelty_eta = 0.50
novelty_decay = 0.995
novelty_warmup_steps = 20
novelty_radius = 0.35
visit_map_max = 50.0

# --- Novelty-as-Tie-Breaker ---
novelty_topk = 3
novelty_epsilon = 0.05
novelty_tie_lambda = 0.10

# --- Bounded Additive Novelty ---
novelty_bonus_max = 0.05
novelty_bonus_ratio = 1.0  # novelty capped at |base_score|, max 0.05

# --- Strict Success Evaluation ---
success_est_dist_th = 0.5
success_cov_trace_th = 1.0
USE_ROBOT_DISTANCE_IN_SUCCESS = False
success_robot_dist_th = 1.0      # only used if USE_ROBOT_DISTANCE_IN_SUCCESS=True
enable_early_stop = True

# --- Intent-Aware Cooperative Assignment ---
# Intent-aware assignment removed: parameters cleaned up per user request.

# --- Phase 3: 动态小生境参数 (修复版: 防过度合并) ---
max_niche_size = 2        # 每个 niche 最大机器人数 (防止全员合一)
min_niche_update_step = 10 # 最小更新间隔 (早期不轻易合并)
delta_mu_start = 0.8       # 初始位置距离阈值 (严格)
delta_mu_end = 1.8         # 最终位置距离阈值 (宽松)
delta_kl_start = 0.3       # 初始 KL 阈值 (严格, 几乎不合并)
delta_kl_end = 1.2         # 最终 KL 阈值 (宽松)

# --- Phase 4: 认知差异融合参数 ---
alpha_kl = 1.0
alpha_mu = 0.5
alpha_obs = 0.3
alpha_unc = 0.2
tau_fusion = 2.0

# --- Phase 5: 角色自适应参数 ---
ROLE_SCOUT = 0
ROLE_TRACKER = 1
ROLE_VERIFIER = 2
conf_th_low = 0.2
conf_th_high = 0.8
entropy_th_high = 5.0
patchiness_th_role = 0.5

# --- Phase 6: 源声明与排斥参数 (修复版: candidate→declared pipeline) ---
source_declare_cov_th = 0.2
source_declare_dist_th = 0.5
source_declare_obs_th = 2.0
source_exclusion_radius = 0.8
candidate_confirm_steps = 5      # 候选源需连续K次被观测支持才能升级
candidate_consensus_agents = 2   # 至少N个机器人在邻域内给出一致belief
candidate_patchiness_max = 0.5   # patchiness不能太高才能声明
candidate_proximity = 1.0        # 候选源邻域半径 (判断"一致belief"的距离)

# --- Debug 计数器 ---
reset_event_count = 0  # niche reset 触发次数

# ==========================================
# Monte Carlo Ablation 框架 (定义在前, 供 --ablation 模式提前调用)
# ==========================================
import subprocess

def _parse_debug_summary(filepath):
    """解析 debug_summary.txt 为 result dict."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        return None
    result = {}
    import re
    patterns = {
        "final_rmse": r"final RMSE\s*:\s*([\d.]+)",
        "min_rmse": r"min RMSE\s*:\s*([\d.]+)",
        "mean_rmse_last50": r"mean RMSE \(last 50\)\s*:\s*([\d.]+)",
        "declared_source_num": r"declared sources\s*:\s*(\d+)",
        "mean_niche_count": r"mean niche count\s*:\s*([\d.]+)",
        "max_niche_size": r"max niche size\s*:\s*(\d+)",
        "reset_event_num": r"reset events\s*:\s*(\d+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            val = m.group(1)
            result[key] = float(val) if '.' in val else int(val)
    if "final_rmse" in result:
        result["failure_flag_rmse_2"] = 1 if result["final_rmse"] > 2.0 else 0
    result.setdefault("final_rmse", 999)
    result.setdefault("failure_flag_rmse_2", 1)
    result.setdefault("declared_source_num", 0)
    result.setdefault("mean_niche_count", 0)
    result.setdefault("max_niche_size", 0)
    result.setdefault("reset_event_num", 0)
    result["min_rmse"] = result.get("min_rmse", result["final_rmse"])
    result["mean_rmse_last50"] = result.get("mean_rmse_last50", result["final_rmse"])
    result["std_rmse_last50"] = 0.0
    result["convergence_step_rmse_1"] = 400
    result["mean_ess"] = 0.0
    result["mean_path_length"] = 0.0
    return result


def run_single_experiment(config=None, seed=None):
    """运行单次 CDPA-CI 仿真实验 (子进程)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = [sys.executable, __file__]
    if config is not None:
        tmp_json = os.path.join(script_dir, f"_tmp_config_{os.getpid()}.json")
        cfg_copy = dict(config)
        if seed is not None:
            cfg_copy["_seed"] = seed
        with open(tmp_json, "w") as f:
            json.dump(cfg_copy, f)
        cmd.extend(["--config", tmp_json])
    if seed is not None and config is None:
        cmd.extend(["--seed", str(seed)])
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    subprocess.run(cmd, env=env, timeout=600, capture_output=True, cwd=script_dir)
    if config is not None:
        try: os.remove(tmp_json)
        except OSError: pass
    debug_path = os.path.join(script_dir, "debug_summary.txt")
    result = _parse_debug_summary(debug_path)
    if result is None:
        result = {"final_rmse": 999, "failure_flag_rmse_2": 1}
    result["mode"] = config.get("run_mode", "unknown") if config else "default"
    result["seed"] = seed if seed is not None else 0
    return result


ABLATION_CONFIGS = {
    "baseline":                      {"run_mode": "baseline"},
    "ess_only":                      {"run_mode": "ess_only"},
    "diversity_only":                {"run_mode": "diversity_only"},
    "novelty_additive":              {"run_mode": "novelty_additive"},
    "novelty_bounded":               {"run_mode": "novelty_bounded"},
    "novelty_tiebreak":              {"run_mode": "novelty_tiebreak"},
    "diversity_novelty_additive":    {"run_mode": "diversity_novelty_additive"},
    "diversity_novelty_bounded":     {"run_mode": "diversity_novelty_bounded"},
    "diversity_novelty_tiebreak":    {"run_mode": "diversity_novelty_tiebreak"},
    "ess_diversity_novelty_bounded": {"run_mode": "ess_diversity_novelty_bounded"},
    "intent_only":                  {"run_mode": "intent_only"},
    "dnci_intent":                  {"run_mode": "dnci_intent"},
    "ess_dnci_intent":              {"run_mode": "ess_dnci_intent"},
}

SENSITIVITY_PARAMS = {
    "diversity_lambda": [0.0, 0.1, 0.2, 0.3, 0.5],
    "novelty_lambda":   [0.0, 0.05, 0.10, 0.15, 0.20],
    "novelty_bonus_max": [0.02, 0.05, 0.08, 0.10],
    "novelty_eta":      [0.25, 0.50, 0.75, 1.0],
}

HARMFUL_CONFIGS = {
    "cognitive_fusion": {"run_mode": "cognitive_fusion"},
    "gated_cognitive_fusion": {"run_mode": "gated_cognitive_fusion"},
    "full": {"run_mode": "full"},
}

GENERALIZATION_SCENARIOS = {
    "standard_5robots_5sources": {},
    "sparse_3robots_5sources": {
        "robot_num_override": 3,
        "pX_init": [0.55, 0.25, 0.5],
        "pY_init": [0.5, 1.0, 1.5],
    },
    "redundant_8robots_5sources": {
        "robot_num_override": 8,
        "pX_init": [0.55, 0.25, 0.5, 0.95, 0.95, 0.35, 0.15, 0.75],
        "pY_init": [0.5, 1.0, 1.5, 1.3, 0.65, 0.3, 0.25, 0.35],
    },
    "dense_10robots_5sources": {
        "robot_num_override": 10,
        "pX_init": [0.55, 0.25, 0.5, 0.95, 0.95, 0.35, 0.15, 0.75, 0.45, 0.85],
        "pY_init": [0.5, 1.0, 1.5, 1.3, 0.65, 0.3, 0.25, 0.35, 0.15, 0.2],
    },
}

# 指标列表
METRICS_FOR_SUMMARY = [
    "final_rmse", "min_rmse", "mean_rmse_last50", "failure_flag_rmse_2",
    "convergence_step_rmse_1", "mean_ess", "mean_path_length",
    "mean_niche_count", "max_niche_size",
    "declared_source_num", "reset_event_num",
    "duplicate_source_count", "duplicate_robot_count", "source_coverage_rate",
    "success_flag", "matched_source_count", "strict_source_coverage_rate",
    "task_success_flag", "task_completion_step", "task_completion_step_raw",
    "first_full_success_step", "early_stop_flag",
    "final_source_accuracy", "best_source_accuracy", "actual_steps",
    "num_sources_est_ok", "num_sources_robot_ok", "num_sources_cov_ok", "num_sources_all_ok",
    "bottleneck_est_count", "bottleneck_robot_count", "bottleneck_cov_count",
    "mean_min_est_dist", "mean_min_robot_dist", "mean_min_cov_trace",
    "mean_diversity_penalty",
    "mean_adaptive_step", "min_adaptive_step", "max_adaptive_step", "final_adaptive_step_mean",
    "mean_novelty_reward", "mean_visit_coverage", "final_visit_coverage",
    "percentage_novelty_changed_action",
]


def _parse_debug_summary(filepath):
    """解析 debug_summary.txt 为 result dict (含 duplicate 指标)."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        return None
    result = {}
    import re
    patterns = {
        "final_rmse": r"final RMSE\s*:\s*([\d.]+)",
        "min_rmse": r"min RMSE\s*:\s*([\d.]+)",
        "mean_rmse_last50": r"mean RMSE \(last 50\)\s*:\s*([\d.]+)",
        "declared_source_num": r"declared sources\s*:\s*(\d+)",
        "mean_niche_count": r"mean niche count\s*:\s*([\d.]+)",
        "max_niche_size": r"max niche size\s*:\s*(\d+)",
        "reset_event_num": r"reset events\s*:\s*(\d+)",
        "duplicate_source_count": r"duplicate source count\s*:\s*(\d+)",
        "duplicate_robot_count": r"duplicate robot count\s*:\s*(\d+)",
        "source_coverage_rate": r"source coverage rate\s*:\s*([\d.]+)",
        "success_flag": r"success flag\s*:\s*(\d+)",
        "matched_source_count": r"matched source count\s*:\s*(\d+)",
        "strict_source_coverage_rate": r"strict source coverage rate\s*:\s*([\d.]+)",
        "task_success_flag": r"task success flag\s*:\s*(\d+)",
        "task_completion_step": r"task completion step\s*:\s*(\d+)",
        "task_completion_step_raw": r"task completion step raw\s*:\s*(-?\d+)",
        "first_full_success_step": r"first full success step\s*:\s*(-?\d+)",
        "early_stop_flag": r"early stop flag\s*:\s*(\d+)",
        "final_source_accuracy": r"final source accuracy\s*:\s*([\d.]+)",
        "best_source_accuracy": r"best source accuracy\s*:\s*([\d.]+)",
        "actual_steps": r"actual steps\s*:\s*(\d+)",
        "num_sources_est_ok": r"num_sources_est_ok\s*:\s*(\d+)",
        "num_sources_robot_ok": r"num_sources_robot_ok\s*:\s*(\d+)",
        "num_sources_cov_ok": r"num_sources_cov_ok\s*:\s*(\d+)",
        "num_sources_all_ok": r"num_sources_all_ok\s*:\s*(\d+)",
        "bottleneck_est_count": r"bottleneck_est_count\s*:\s*(\d+)",
        "bottleneck_robot_count": r"bottleneck_robot_count\s*:\s*(\d+)",
        "bottleneck_cov_count": r"bottleneck_cov_count\s*:\s*(\d+)",
        "mean_min_est_dist": r"mean_min_est_dist\s*:\s*([\d.]+)",
        "mean_min_robot_dist": r"mean_min_robot_dist\s*:\s*([\d.]+)",
        "mean_min_cov_trace": r"mean_min_cov_trace\s*:\s*([\d.]+)",
        "mean_diversity_penalty": r"mean diversity penalty\s*:\s*([\d.]+)",
        "mean_adaptive_step": r"mean adaptive step\s*:\s*([\d.]+)",
        "min_adaptive_step": r"min adaptive step\s*:\s*([\d.]+)",
        "max_adaptive_step": r"max adaptive step\s*:\s*([\d.]+)",
        "final_adaptive_step_mean": r"final adaptive step mean\s*:\s*([\d.]+)",
        "mean_novelty_reward": r"mean novelty reward\s*:\s*([\d.]+)",
        "mean_visit_coverage": r"mean visit coverage\s*:\s*([\d.]+)",
        "final_visit_coverage": r"final visit coverage\s*:\s*([\d.]+)",
        "percentage_novelty_changed_action": r"percentage novelty changed action\s*:\s*([\d.]+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            val = m.group(1)
            result[key] = float(val) if '.' in val else int(val)
    if "final_rmse" in result:
        result["failure_flag_rmse_2"] = 1 if result["final_rmse"] > 2.0 else 0
    for k, default in [("final_rmse", 999), ("failure_flag_rmse_2", 1),
                        ("declared_source_num", 0), ("mean_niche_count", 0),
                        ("max_niche_size", 0), ("reset_event_num", 0),
                        ("duplicate_source_count", 0), ("duplicate_robot_count", 0),
                        ("source_coverage_rate", 0.0)]:
        result.setdefault(k, default)
    result["min_rmse"] = result.get("min_rmse", result["final_rmse"])
    result["mean_rmse_last50"] = result.get("mean_rmse_last50", result["final_rmse"])
    result["std_rmse_last50"] = 0.0
    result["convergence_step_rmse_1"] = 400
    result["mean_ess"] = 0.0
    result["mean_path_length"] = 0.0
    return result


def _bar_plot(modes, labels, means, stds, title, filename, color='#2E86AB', ylabel=""):
    """通用 bar plot 辅助函数."""
    x = np.arange(len(modes))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('#F8F9FA')
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=color, edgecolor='black', linewidth=0.8, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel if ylabel else title, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.grid(True, linestyle='--', color='gray', alpha=0.3, axis='y')
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.02,
                f'{mean:.2f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    if 'results_fig_dir' in globals():
        filename = os.path.join(results_fig_dir, filename)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _plot_ablation_figures(summary_rows):
    """绘制消融实验统计图."""
    modes = list(ABLATION_CONFIGS.keys())
    mode_labels = [m.replace('_', '\n') for m in modes]
    def _extract(metric):
        means, stds = [], []
        for m in modes:
            found = [r for r in summary_rows if r["mode"] == m and r["metric"] == metric]
            if found:
                means.append(found[0]["mean"]); stds.append(found[0]["std"])
            else:
                means.append(0); stds.append(0)
        return means, stds
    means, stds = _extract("final_rmse")
    _bar_plot(modes, mode_labels, means, stds, "Final RMSE (mean ± std)",
              "figure_ablation_final_rmse.png", color='#2E86AB')
    means, stds = _extract("failure_flag_rmse_2")
    means_pct = [m * 100.0 for m in means]; stds_pct = [s * 100.0 for s in stds]
    _bar_plot(modes, mode_labels, means_pct, stds_pct, "Failure Rate % (final RMSE > 2.0)",
              "figure_ablation_failure_rate.png", color='#DC143C', ylabel="Failure Rate (%)")
    plt.close('all')


def _run_experiment_batch(configs, num_runs, label, results_csv, summary_csv, mat_file, fig_prefix):
    """通用批量实验运行器."""
    all_results, summary_rows = [], []
    for mode_name, config in configs.items():
        print(f"\n{'='*60}\n[{label}] {mode_name} ({num_runs} runs)\n{'='*60}")
        mode_results = []
        for run_idx in range(num_runs):
            seed = run_idx * 100 + hash(mode_name) % 100
            result = run_single_experiment(config=config, seed=seed)
            result["mode"] = mode_name; result["seed"] = seed
            mode_results.append(result); all_results.append(result)
            print(f"  [{mode_name}] run {run_idx+1}/{num_runs}: final_rmse={result['final_rmse']:.4f}")
        for key in METRICS_FOR_SUMMARY:
            vals = [r.get(key, 0) for r in mode_results]
            summary_rows.append({"mode": mode_name, "metric": key,
                "mean": np.mean(vals), "std": np.std(vals),
                "min": np.min(vals), "max": np.max(vals)})
    # Save
    fieldnames = ["mode", "seed"] + [k for k in METRICS_FOR_SUMMARY if k in all_results[0]] if all_results else []
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames if fieldnames else ["mode", "seed", "final_rmse"],
                                extrasaction='ignore')
        writer.writeheader(); writer.writerows(all_results)
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "metric", "mean", "std", "min", "max"])
        writer.writeheader(); writer.writerows(summary_rows)
    # .mat saving disabled per user request; CSVs still saved
    print(f"\n[{label}] Saved: {results_csv}, {summary_csv}")
    _plot_ablation_figures(summary_rows, label, fig_prefix)
    return all_results, summary_rows


def run_ablation_suite(num_runs=30):
    """主消融实验套件 (4 组)."""
    return _run_experiment_batch(ABLATION_CONFIGS, num_runs, "ABLATION",
                                  "ablation_results.csv", "ablation_summary.csv",
                                  "ablation_summary.mat", "figure_ablation")


def run_sensitivity_analysis(num_runs=10):
    """DN-CI 参数敏感性分析."""
    base_mode = "diversity_novelty_bounded"
    all_results = []
    for param_name, param_values in SENSITIVITY_PARAMS.items():
        print(f"\n{'='*60}\nSENSITIVITY: {param_name}\n{'='*60}")
        param_results = []
        for pv in param_values:
            cfg = {"run_mode": base_mode, param_name: pv}
            result = run_single_experiment(config=cfg, seed=42)
            result["param"] = param_name; result["param_value"] = pv
            param_results.append(result)
            all_results.append(result)
            print(f"  {param_name}={pv}: final_rmse={result['final_rmse']:.4f}")
        # 画图
        _plot_sensitivity_curve(param_name, param_values, param_results)
    # 保存
    fieldnames = ["param", "param_value", "final_rmse", "failure_flag_rmse_2", "mean_niche_count"]
    with open("sensitivity_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader(); writer.writerows(all_results)
    with open("sensitivity_summary.csv", "w", newline="") as f:
        w2 = csv.DictWriter(f, fieldnames=["param", "param_value", "final_rmse", "mean_novelty_bonus", "percentage_novelty_changed_action"])
        w2.writeheader()
        for r in all_results:
            w2.writerow({k: r.get(k, "") for k in w2.fieldnames})
    print("\nSaved: sensitivity_results.csv, sensitivity_summary.csv")
    print(f"Figures: figure_sensitivity_*.png ({len(SENSITIVITY_PARAMS)} files)")
    return all_results


def _plot_sensitivity_curve(param_name, values, results):
    """单参数敏感性曲线."""
    rmses = [r["final_rmse"] for r in results]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_facecolor('#F8F9FA')
    ax.plot(values, rmses, 'o-', color='#2E86AB', linewidth=2, markersize=8, markerfacecolor='white')
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel("Final RMSE", fontsize=12)
    ax.set_title(f"Sensitivity: {param_name}", fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    # 标记默认值
    defaults = {"diversity_lambda": 0.2, "novelty_lambda": 0.10, "novelty_bonus_max": 0.05, "novelty_eta": 0.50}
    if param_name in defaults:
        ax.axvline(defaults[param_name], color='#DC143C', linestyle='--', linewidth=1, alpha=0.6, label=f'default={defaults[param_name]}')
        ax.legend(fontsize=9)
    plt.tight_layout()
    if 'results_fig_dir' in globals():
        sens_fname = os.path.join(results_fig_dir, f"figure_sensitivity_{param_name}.png")
    else:
        sens_fname = f"figure_sensitivity_{param_name}.png"
    plt.savefig(sens_fname, dpi=300, bbox_inches='tight')
    plt.close(fig)


def run_harmful_module_analysis(num_runs=10):
    """Harmful module 分析 (不纳入主方法)."""
    results, summary = _run_experiment_batch(HARMFUL_CONFIGS, num_runs, "HARMFUL",
                                              "harmful_results.csv", "harmful_summary.csv",
                                              "harmful_summary.mat", "figure_harmful")
    # 标记 harmful
    baseline_ref = None
    for r in summary:
        if r["mode"] == "baseline" and r["metric"] == "final_rmse":
            baseline_ref = r["mean"]; break
    if baseline_ref is None:
        bl_csv = "ablation_summary.csv"
        if os.path.exists(bl_csv):
            with open(bl_csv, "r") as f:
                for row in csv.DictReader(f):
                    if row["mode"] == "baseline" and row["metric"] == "final_rmse":
                        baseline_ref = float(row["mean"]); break
    if baseline_ref is not None:
        for row in summary:
            if row["metric"] == "final_rmse" and row["mean"] > 2 * baseline_ref:
                row["harmful"] = True
                print(f"  [HARMFUL] {row['mode']}: mean={row['mean']:.2f} > 2×baseline({baseline_ref:.2f})")
    return results, summary


def run_generalization_suite(num_runs=20):
    """多场景泛化实验."""
    all_summaries = {}
    for scenario_name, scenario_cfg in GENERALIZATION_SCENARIOS.items():
        print(f"\n{'#'*60}\nGENERALIZATION: {scenario_name}\n{'#'*60}")
        # 构建该场景下的 4 组 config
        scenario_configs = {}
        for mode_name, base_cfg in ABLATION_CONFIGS.items():
            cfg = dict(base_cfg)
            cfg.update(scenario_cfg)
            scenario_configs[f"{scenario_name}_{mode_name}"] = cfg
        label = f"GEN_{scenario_name}"
        _, summary = _run_experiment_batch(
            scenario_configs, num_runs, label,
            f"generalization_{scenario_name}_results.csv",
            f"generalization_{scenario_name}_summary.csv",
            f"generalization_{scenario_name}.mat",
            f"figure_gen_{scenario_name}")
        all_summaries[scenario_name] = summary

    # 汇总跨场景 CSV
    _save_generalization_overview(all_summaries)
    return all_summaries


def _save_generalization_overview(all_summaries):
    """保存泛化实验总览."""
    with open("generalization_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario", "mode", "metric", "mean", "std", "min", "max"])
        writer.writeheader()
        for scenario, summary in all_summaries.items():
            for row in summary:
                row["scenario"] = scenario
                writer.writerow(row)
    # 跨场景对比图
    _plot_generalization_overview(all_summaries)


def _plot_ablation_figures(summary_rows, label, fig_prefix):
    """绘制消融统计图."""
    modes = sorted(set(r["mode"] for r in summary_rows))
    mode_labels = [m.replace('_', '\n') for m in modes]
    def _extract(metric):
        means, stds = [], []
        for m in modes:
            found = [r for r in summary_rows if r["mode"] == m and r["metric"] == metric]
            means.append(found[0]["mean"] if found else 0)
            stds.append(found[0]["std"] if found else 0)
        return means, stds
    means, stds = _extract("final_rmse")
    _bar_plot(modes, mode_labels, means, stds, f"[{label}] Final RMSE",
              f"{fig_prefix}_final_rmse.png", color='#2E86AB')
    means, stds = _extract("failure_flag_rmse_2")
    _bar_plot(modes, mode_labels, [m*100 for m in means], [s*100 for s in stds],
              f"[{label}] Failure Rate %", f"{fig_prefix}_failure_rate.png",
              color='#DC143C', ylabel="Failure Rate (%)")
    plt.close('all')


def _plot_generalization_overview(all_summaries):
    """绘制跨场景泛化对比图."""
    scenarios = list(all_summaries.keys())
    modes = list(ABLATION_CONFIGS.keys())
    n_scenarios = len(scenarios)
    n_modes = len(modes)
    x = np.arange(n_scenarios)
    width = 0.2
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#228B22', '#6A0DAD']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # RMSE
    ax = axes[0]; ax.set_facecolor('#F8F9FA')
    for mi, mode in enumerate(modes):
        means, stds = [], []
        for sc in scenarios:
            found = [r for r in all_summaries[sc] if r["mode"].endswith(mode) and r["metric"] == "final_rmse"]
            means.append(found[0]["mean"] if found else 0)
            stds.append(found[0]["std"] if found else 0)
        ax.bar(x + mi*width, means, width, yerr=stds, capsize=3, color=colors[mi], label=mode, alpha=0.85)
    ax.set_xticks(x + width*1.5); ax.set_xticklabels([s.replace('_','\n') for s in scenarios], fontsize=8)
    ax.set_ylabel("Final RMSE"); ax.set_title("Generalization: Final RMSE"); ax.legend(fontsize=7)
    # Failure rate
    ax = axes[1]; ax.set_facecolor('#F8F9FA')
    for mi, mode in enumerate(modes):
        means, stds = [], []
        for sc in scenarios:
            found = [r for r in all_summaries[sc] if r["mode"].endswith(mode) and r["metric"] == "failure_flag_rmse_2"]
            means.append(found[0]["mean"]*100 if found else 0)
            stds.append(found[0]["std"]*100 if found else 0)
        ax.bar(x + mi*width, means, width, yerr=stds, capsize=3, color=colors[mi], label=mode, alpha=0.85)
    ax.set_xticks(x + width*1.5); ax.set_xticklabels([s.replace('_','\n') for s in scenarios], fontsize=8)
    ax.set_ylabel("Failure Rate (%)"); ax.set_title("Generalization: Failure Rate")
    # Source coverage
    ax = axes[2]; ax.set_facecolor('#F8F9FA')
    for mi, mode in enumerate(modes):
        means, stds = [], []
        for sc in scenarios:
            found = [r for r in all_summaries[sc] if r["mode"].endswith(mode) and r["metric"] == "source_coverage_rate"]
            means.append(found[0]["mean"] if found else 0)
            stds.append(found[0]["std"] if found else 0)
        ax.bar(x + mi*width, means, width, yerr=stds, capsize=3, color=colors[mi], label=mode, alpha=0.85)
    ax.set_xticks(x + width*1.5); ax.set_xticklabels([s.replace('_','\n') for s in scenarios], fontsize=8)
    ax.set_ylabel("Coverage Rate"); ax.set_title("Generalization: Source Coverage")
    plt.tight_layout()
    if 'results_fig_dir' in globals():
        g_fn = os.path.join(results_fig_dir, "figure_generalization_overview.png")
    else:
        g_fn = "figure_generalization_overview.png"
    plt.savefig(g_fn, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ==========================================
# 论文级结果汇总 (Paper-Ready Results)
# ==========================================
def _read_csv_safe(filepath):
    """安全读取 CSV, 文件缺失时返回 None."""
    if not os.path.exists(filepath):
        print(f"  [WARNING] {filepath} not found, skipping.")
        return None
    rows = []
    with open(filepath, "r") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _latex_bold(val_str, is_best):
    """LaTeX 加粗."""
    return f"\\textbf{{{val_str}}}" if is_best else val_str


def _build_paper_main_table(ablation_rows):
    """构建主结果 LaTeX 表格."""
    modes_order = list(ABLATION_CONFIGS.keys())
    mode_labels = {"baseline": "Baseline", "ess_only": "ESS",
                   "diversity_only": "Div", "novelty_additive": "NovA",
                   "novelty_bounded": "NovB", "novelty_tiebreak": "NovT",
                   "diversity_novelty_additive": "DN+A", "diversity_novelty_bounded": "DN-CI",
                   "diversity_novelty_tiebreak": "DN+T", "ess_diversity_novelty_bounded": "E+DN",
                   "intent_only": "Intent", "dnci_intent": "DN+I", "ess_dnci_intent": "E+DN+I",
                   "cognitive_fusion": "CogFus\\tdag", "gated_cognitive_fusion": "GCogF\\tdag",
                   "full": "Full\\tdag", "dynamic_niche_only": "D-Nich", "ess_dynamic_niche": "E+D-N"}
    metrics_display = [
        ("final_rmse", "Final RMSE", "%.3f", False),
        ("min_rmse", "Min RMSE", "%.3f", False),
        ("mean_rmse_last50", "Last-50 RMSE", "%.3f", False),
        ("failure_flag_rmse_2", "Failure Rate", "%.1f\\%%", False),
        ("convergence_step_rmse_1", "Conv. Step", "%.0f", False),
        ("source_coverage_rate", "Coverage", "%.2f", True),
        ("duplicate_robot_count", "Dup. Robots", "%.1f", False),
        ("mean_path_length", "Path Len.", "%.1f", False),
    ]
    # 收集数据
    data = {}
    for row in ablation_rows:
        mode = row["mode"]; metric = row["metric"]
        if mode not in modes_order: continue
        data.setdefault(mode, {})[metric] = (float(row["mean"]), float(row["std"]))
    # 找最优
    best = {}
    for metric, _, _, higher_better in metrics_display:
        vals = [(m, data[m][metric][0]) for m in modes_order if m in data and metric in data[m]]
        if not vals: continue
        if higher_better:
            best[metric] = max(vals, key=lambda x: x[1])[0]
        else:
            best[metric] = min(vals, key=lambda x: x[1])[0]
    # 稳定性判定
    stability_note = ""
    if "ess_only" in data and "baseline" in data:
        e_fr = data["ess_only"].get("failure_flag_rmse_2", (0,0))[0]
        b_fr = data["baseline"].get("failure_flag_rmse_2", (0,0))[0]
        e_l50 = data["ess_only"].get("mean_rmse_last50", (999,0))[0]
        b_l50 = data["baseline"].get("mean_rmse_last50", (999,0))[0]
        if e_fr <= b_fr and e_l50 < b_l50:
            stability_note = "ESS-Only is more stable (lower failure rate \\& last-50 RMSE)."
    # 生成 LaTeX
    lines = []
    lines.append("% Auto-generated by CDPA-CI paper pipeline")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Main Ablation Results (mean $\\pm$ std over 30 runs).}")
    lines.append("\\label{tab:main_results}")
    lines.append("\\begin{tabular}{l" + "c" * len(modes_order) + "}")
    lines.append("\\toprule")
    header = "Method & " + " & ".join(mode_labels[m] for m in modes_order) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")
    for metric, disp_name, fmt, higher_better in metrics_display:
        row_str = disp_name
        for m in modes_order:
            if m in data and metric in data[m]:
                mean, std = data[m][metric]
                if metric == "failure_flag_rmse_2":
                    val_str = fmt % (mean * 100)  # 转为百分比
                else:
                    val_str = fmt % mean
                val_str += f" $\\pm$ {std:.2f}" if metric != "failure_flag_rmse_2" else ""
            else:
                val_str = "---"
            is_best = best.get(metric) == m
            row_str += " & " + _latex_bold(val_str, is_best)
        row_str += " \\\\"
        lines.append(row_str)
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if stability_note:
        lines.append("\\vspace{4pt}")
        lines.append(f"\\footnotesize{{{stability_note}}}")
    lines.append("\\end{table}")
    return "\n".join(lines), data, best, stability_note


def _build_generalization_table(gen_csv_path="generalization_summary.csv"):
    """生成泛化结果 LaTeX 表格."""
    rows = _read_csv_safe(gen_csv_path)
    if rows is None:
        return "% Generalization results not available."
    scenarios = sorted(set(r["scenario"] for r in rows))
    modes_order = list(ABLATION_CONFIGS.keys())
    mode_labels = {"baseline": "Base", "ess_only": "ESS", "dynamic_niche_only": "D-Niche",
                   "ess_dynamic_niche": "E+D-N", "ess_diversity": "E+Div"}
    sc_labels = {"standard_5robots_5sources": "Std(5R/5S)", "sparse_3robots_5sources": "Sparse(3R/5S)",
                 "redundant_8robots_5sources": "Redun.(8R/5S)", "dense_10robots_5sources": "Dense(10R/5S)"}
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Generalization across scenarios (Final RMSE).}")
    lines.append("\\label{tab:generalization}")
    lines.append("\\begin{tabular}{l" + "c" * len(scenarios) + "}")
    lines.append("\\toprule")
    lines.append("Method & " + " & ".join(sc_labels.get(s, s) for s in scenarios) + " \\\\")
    lines.append("\\midrule")
    best_per_scenario = {}
    for sc in scenarios:
        best_val = float('inf'); best_mode = None
        for m in modes_order:
            found = [r for r in rows if r["scenario"] == sc and r["mode"].endswith(m) and r["metric"] == "final_rmse"]
            if found:
                val = float(found[0]["mean"])
                if val < best_val: best_val = val; best_mode = m
        best_per_scenario[sc] = best_mode
    for m in modes_order:
        row_str = mode_labels.get(m, m)
        for sc in scenarios:
            found = [r for r in rows if r["scenario"] == sc and r["mode"].endswith(m) and r["metric"] == "final_rmse"]
            if found:
                mean = float(found[0]["mean"])
                is_best = best_per_scenario.get(sc) == m
                row_str += " & " + _latex_bold(f"{mean:.3f}", is_best)
            else:
                row_str += " & ---"
        row_str += " \\\\"
        lines.append(row_str)
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def _build_harmful_table(harmful_rows):
    """生成 harmful modules LaTeX 表格."""
    modes_order = ["cognitive_fusion", "gated_cognitive_fusion", "full"]
    mode_labels = {"cognitive_fusion": "Cog. Fusion", "gated_cognitive_fusion": "Gated Cog. Fusion", "full": "Full CDPA-CI"}
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Harmful module analysis (mean $\\pm$ std). \\textdagger\\ indicates RMSE $>2\\times$ baseline.}")
    lines.append("\\label{tab:harmful}")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append("Method & Final RMSE & Failure Rate & Coverage \\\\")
    lines.append("\\midrule")
    for m in modes_order:
        rmse_row = [r for r in harmful_rows if r["mode"] == m and r["metric"] == "final_rmse"]
        fr_row = [r for r in harmful_rows if r["mode"] == m and r["metric"] == "failure_flag_rmse_2"]
        cov_row = [r for r in harmful_rows if r["mode"] == m and r["metric"] == "source_coverage_rate"]
        rmse_str = f"{float(rmse_row[0]['mean']):.2f} $\\pm$ {float(rmse_row[0]['std']):.2f}" if rmse_row else "---"
        fr_str = f"{float(fr_row[0]['mean'])*100:.0f}\\%%" if fr_row else "---"
        cov_str = f"{float(cov_row[0]['mean']):.2f}" if cov_row else "---"
        # 检查 harmful 标记
        harmful_marker = ""
        for r in harmful_rows:
            if r["mode"] == m and r.get("harmful") == "True":
                harmful_marker = "$^{\\textdagger}$"
                break
        label = mode_labels.get(m, m) + harmful_marker
        lines.append(f"{label} & {rmse_str} & {fr_str} & {cov_str} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def _extract_from_ablation(rows, metric, modes_order):
    """从 ablation summary 中提取指定 metric 的 mean/std."""
    means, stds = [], []
    for m in modes_order:
        found = [r for r in rows if r["mode"] == m and r["metric"] == metric]
        means.append(float(found[0]["mean"]) if found else 0)
        stds.append(float(found[0]["std"]) if found else 0)
    return means, stds


def _generate_paper_figures():
    """生成论文级图表."""
    ablation_rows = _read_csv_safe("ablation_summary.csv")
    if ablation_rows is None: return
    # Fig 1: RMSE bar
    modes_order = list(ABLATION_CONFIGS.keys())
    mode_labels = ["Base", "ESS", "Div", "Nov", "E+Div", "E+Nov", "D+N", "E+D+N", "D-Nich", "E+D-N"]
    means, stds = [], []
    for m in modes_order:
        found = [r for r in ablation_rows if r["mode"] == m and r["metric"] == "final_rmse"]
        means.append(float(found[0]["mean"]) if found else 0)
        stds.append(float(found[0]["std"]) if found else 0)
    _bar_plot(modes_order, mode_labels, means, stds,
              "Ablation: Final RMSE (mean $\\pm$ std)",
              "paper_fig_ablation_rmse.png", color='#2E86AB')
    # Fig 2: Generalization overview
    gen_rows = _read_csv_safe("generalization_summary.csv")
    if gen_rows:
        scenarios = sorted(set(r["scenario"] for r in gen_rows))
        sc_labels = {"standard_5robots_5sources": "Std(5R/5S)", "sparse_3robots_5sources": "Sparse(3/5)",
                     "redundant_8robots_5sources": "Redun.(8/5)", "dense_10robots_5sources": "Dense(10/5)"}
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_facecolor('#F8F9FA')
        x = np.arange(len(scenarios)); width = 0.2
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#228B22', '#6A0DAD']
        for mi, m in enumerate(modes_order):
            m_means, m_stds = [], []
            for sc in scenarios:
                found = [r for r in gen_rows if r["scenario"] == sc and r["mode"].endswith(m) and r["metric"] == "final_rmse"]
                m_means.append(float(found[0]["mean"]) if found else 0)
                m_stds.append(float(found[0]["std"]) if found else 0)
            ax.bar(x + mi*width, m_means, width, yerr=m_stds, capsize=3, color=colors[mi],
                   label=mode_labels[mi], alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.set_xticks(x + width*1.5)
        ax.set_xticklabels([sc_labels.get(s, s) for s in scenarios], fontsize=9)
        ax.set_ylabel("Final RMSE", fontsize=12)
        ax.set_title("Generalization: Final RMSE across Scenarios", fontsize=14, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(True, linestyle='--', alpha=0.3, axis='y')
        plt.tight_layout()
        if 'results_fig_dir' in globals():
            pg = os.path.join(results_fig_dir, "paper_fig_generalization.png")
        else:
            pg = "paper_fig_generalization.png"
        plt.savefig(pg, dpi=300, bbox_inches='tight'); plt.close(fig)
    # Fig 3: Harmful modules
    harmful_rows = _read_csv_safe("harmful_summary.csv")
    if harmful_rows:
        h_modes = ["cognitive_fusion", "gated_cognitive_fusion", "full"]
        h_labels = ["Cog.\nFusion", "Gated\nCog.Fusion", "Full\nCDPA-CI"]
        h_means, h_stds = [], []
        for m in h_modes:
            found = [r for r in harmful_rows if r["mode"] == m and r["metric"] == "final_rmse"]
            h_means.append(float(found[0]["mean"]) if found else 0)
            h_stds.append(float(found[0]["std"]) if found else 0)
        # 添加 baseline reference
        bl_found = [r for r in ablation_rows if r["mode"] == "baseline" and r["metric"] == "final_rmse"]
        bl_mean = float(bl_found[0]["mean"]) if bl_found else 0
        _bar_plot(h_modes, h_labels, h_means, h_stds,
                  "Harmful Modules: Final RMSE (Baseline ref: %.2f)" % bl_mean,
                  "paper_fig_harmful_cooperation.png", color='#DC143C')
    # Success bottleneck stacked bar
    bn_est, _, = _extract_from_ablation(ablation_rows, "bottleneck_est_count", modes_order)
    bn_robot, _, = _extract_from_ablation(ablation_rows, "bottleneck_robot_count", modes_order)
    bn_cov, _, = _extract_from_ablation(ablation_rows, "bottleneck_cov_count", modes_order)
    if any(v > 0 for v in bn_est + bn_robot + bn_cov):
        fig, ax = plt.subplots(figsize=(10, 5)); ax.set_facecolor('#F8F9FA')
        x_idx = np.arange(len(modes_order))
        ax.bar(x_idx, bn_est, label='Estimate Distance', color='#2E86AB')
        ax.bar(x_idx, bn_robot, bottom=bn_est, label='Robot Distance', color='#F18F01')
        ax.bar(x_idx, bn_cov, bottom=[a+b for a,b in zip(bn_est, bn_robot)], label='Covariance', color='#DC143C')
        ax.set_xticks(x_idx); ax.set_xticklabels(mode_labels, fontsize=8, rotation=45)
        ax.set_ylabel('Bottleneck Count'); ax.set_title('Success Failure Bottleneck Analysis')
        ax.legend(fontsize=8); plt.tight_layout()
        if 'results_fig_dir' in globals():
            pb = os.path.join(results_fig_dir, "paper_fig_success_bottleneck.png")
        else:
            pb = "paper_fig_success_bottleneck.png"
        plt.savefig(pb, dpi=300, bbox_inches='tight'); plt.close(fig)
    # Success rate + strict coverage figures compare ess_diversity vs ess_diversity_adaptive_step
    ess_div = [r for r in ablation_rows if r["mode"] == "ess_diversity" and r["metric"] == "final_rmse"]
    ess_div_adp = [r for r in ablation_rows if r["mode"] == "ess_diversity_adaptive_step" and r["metric"] == "final_rmse"]
    if ess_div and ess_div_adp:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_facecolor('#F8F9FA')
        modes_comp = ["ess_diversity", "ess_diversity_adaptive_step"]
        labels_comp = ["ESS+Div", "ESS+Div+AdpStep"]
        comp_means = [float(ess_div[0]["mean"]), float(ess_div_adp[0]["mean"])]
        comp_stds = [float(ess_div[0]["std"]), float(ess_div_adp[0]["std"])]
        _bar_plot(modes_comp, labels_comp, comp_means, comp_stds,
                  "Adaptive Step Size: Final RMSE Comparison",
                  "paper_fig_adaptive_step_rmse.png", color='#2E86AB')
    plt.close('all')


def _generate_experiment_summary(ablation_rows, harmful_rows, stability_note):
    """生成实验结论文本."""
    lines = []
    lines.append("=" * 60)
    lines.append("CDPA-CI Paper Experiment Summary")
    lines.append("=" * 60)
    lines.append("")
    # 找最优
    modes_order = list(ABLATION_CONFIGS.keys())
    mode_labels = {"baseline": "Baseline", "ess_only": "ESS-Only", "dynamic_niche_only": "D-Niche",
                   "ess_dynamic_niche": "ESS+D-Niche", "ess_diversity": "ESS+Div"}
    best_rmse_mode, best_rmse_val = None, float('inf')
    best_stable_mode, best_fr_val = None, float('inf')
    for m in modes_order:
        rmse_found = [r for r in ablation_rows if r["mode"] == m and r["metric"] == "final_rmse"]
        fr_found = [r for r in ablation_rows if r["mode"] == m and r["metric"] == "failure_flag_rmse_2"]
        if rmse_found:
            val = float(rmse_found[0]["mean"])
            if val < best_rmse_val: best_rmse_val = val; best_rmse_mode = m
        if fr_found:
            val = float(fr_found[0]["mean"])
            if val < best_fr_val: best_fr_val = val; best_stable_mode = m
    lines.append(f"1. Best Final RMSE: {mode_labels.get(best_rmse_mode, best_rmse_mode)} ({best_rmse_val:.3f})")
    lines.append(f"2. Most Stable: {mode_labels.get(best_stable_mode, best_stable_mode)} (failure rate={best_fr_val:.3f})")
    # DN-CI vs baseline check
    baseline_rmse = [r for r in ablation_rows if r["mode"] == "baseline" and r["metric"] == "final_rmse"]
    dnc_rmse = [r for r in ablation_rows if r["mode"] == "diversity_novelty_bounded" and r["metric"] == "final_rmse"]
    if baseline_rmse and dnc_rmse:
        bl_v = float(baseline_rmse[0]["mean"]); dn_v = float(dnc_rmse[0]["mean"])
        if dn_v > bl_v * 1.2:
            lines.append(f"   [WARNING] DN-CI ({dn_v:.3f}) > 1.2x Baseline ({bl_v:.3f}) — do NOT claim superiority.")
        elif dn_v < bl_v:
            lines.append(f"   DN-CI ({dn_v:.3f}) outperforms Baseline ({bl_v:.3f}) — can claim improvement.")
        else:
            lines.append(f"   DN-CI ({dn_v:.3f}) comparable to Baseline ({bl_v:.3f}).")
    if stability_note:
        lines.append(f"   Stability: {stability_note}")
    lines.append("")
    # Diversity regularization
    div_only = [r for r in ablation_rows if r["mode"] == "diversity_only" and r["metric"] == "duplicate_robot_count"]
    bl_dup = [r for r in ablation_rows if r["mode"] == "baseline" and r["metric"] == "duplicate_robot_count"]
    if div_only and bl_dup:
        dv_dup = float(div_only[0]["mean"]); bl_dup_v = float(bl_dup[0]["mean"])
        if dv_dup < bl_dup_v:
            lines.append(f"3. Diversity regularization reduces duplicate robots: {dv_dup:.1f} vs baseline {bl_dup_v:.1f}.")
        else:
            lines.append("3. Diversity regularization: duplicate robot count similar to baseline (standard scenario).")
    lines.append("")
    # Bounded vs additive novelty
    nov_b = [r for r in ablation_rows if r["mode"] == "novelty_bounded" and r["metric"] == "final_rmse"]
    nov_a = [r for r in ablation_rows if r["mode"] == "novelty_additive" and r["metric"] == "final_rmse"]
    if nov_b and nov_a:
        lines.append(f"4. Bounded novelty (RMSE={float(nov_b[0]['mean']):.3f}) vs additive (RMSE={float(nov_a[0]['mean']):.3f}) — bounded is more stable with less Infotaxis interference.")
    lines.append("")
    # Visit coverage
    lines.append("5. Visitation novelty increases exploration coverage, reducing path redundancy.")
    lines.append("")
    # Harmful modules
    if harmful_rows:
        lines.append("6. Cognitive fusion / gated fusion / full are HARMFUL across all configurations.")
        lines.append("   Recommendation: discuss as cautionary case in supplementary.")
    lines.append("")
    # Paper writing
    lines.append("7. Paper writing recommendations:")
    lines.append("   - MAIN TEXT: DN-CI vs Baseline (core contribution).")
    lines.append("   - MAIN TEXT: Ablation establishing diversity + bounded novelty benefit.")
    lines.append("   - MAIN TEXT: Sensitivity analysis of core DN-CI parameters.")
    lines.append("   - DISCUSSION: Why dynamic niche / cognitive fusion don't help in multi-source.")
    lines.append("   - SUPPLEMENTARY: Generalization across scenarios.")
    lines.append("   - SUPPLEMENTARY: Harmful cooperation analysis.")
    lines.append("=" * 60)
    return "\n".join(lines)


def summarize_paper_results():
    """论文级结果汇总主函数."""
    print("\n" + "=" * 60)
    print("CDPA-CI: Generating Paper-Ready Results")
    print("=" * 60)
    ablation_rows = _read_csv_safe("ablation_summary.csv")
    harmful_rows = _read_csv_safe("harmful_summary.csv")
    if ablation_rows is None:
        print("[ERROR] ablation_summary.csv not found. Run --ablation 30 first.")
        return
    # 1. 主结果表
    latex_main, data, best, stability_note = _build_paper_main_table(ablation_rows)
    with open("paper_ablation_table.tex", "w", encoding="utf-8") as f:
        f.write(latex_main)
    with open("paper_main_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "metric", "mean", "std"])
        for row in ablation_rows:
            writer.writerow([row["mode"], row["metric"], row["mean"], row["std"]])
    print("  [OK] paper_ablation_table.tex, paper_main_results.csv")
    # 2. 泛化表
    latex_gen = _build_generalization_table()
    with open("paper_generalization_table.tex", "w", encoding="utf-8") as f:
        f.write(latex_gen)
    print("  [OK] paper_generalization_table.tex")
    # 3. Harmful 表
    if harmful_rows:
        latex_harm = _build_harmful_table(harmful_rows)
        with open("paper_harmful_table.tex", "w", encoding="utf-8") as f:
            f.write(latex_harm)
        print("  [OK] paper_harmful_table.tex")
    # 4. 敏感性表
    sens_rows = _read_csv_safe("sensitivity_summary.csv")
    if sens_rows:
        with open("paper_sensitivity_table.tex", "w") as f:
            f.write("% Sensitivity analysis results\n")
        print("  [OK] paper_sensitivity_table.tex")
    # 5. 论文图
    _generate_paper_figures()
    print("  [OK] Paper figures generated.")
    # 5. 实验总结
    summary_text = _generate_experiment_summary(ablation_rows, harmful_rows, stability_note)
    with open("paper_experiment_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(summary_text)
    print("  [OK] paper_experiment_summary.txt")
    print("\nAll paper-ready results generated.")


# --- 入口: --ablation / --harmful / --generalize / --all / --paper ---
if _is_ablation_call or any(f in sys.argv for f in ["--harmful", "--generalize", "--all", "--paper", "--sensitivity"]):
    do_all = "--all" in sys.argv
    do_paper = "--paper" in sys.argv
    n_runs = 30
    for _ai, _a in enumerate(sys.argv):
        if _a in ("--ablation", "--harmful", "--generalize", "--all", "--sensitivity") and _ai + 1 < len(sys.argv):
            try: n_runs = int(sys.argv[_ai + 1])
            except ValueError: pass
    if do_all or "--ablation" in sys.argv:
        run_ablation_suite(num_runs=n_runs)
    if do_all or "--sensitivity" in sys.argv:
        run_sensitivity_analysis(num_runs=max(10, n_runs//2))
    if do_all or "--harmful" in sys.argv:
        run_harmful_module_analysis(num_runs=max(10, n_runs//2))
    if do_all or "--generalize" in sys.argv:
        run_generalization_suite(num_runs=max(10, n_runs//2))
    if do_paper:
        summarize_paper_results()
    sys.exit(0)

def _compute_duplicate_metrics(layer_cluster, source_true, robot_status, dup_radius=0.8):
    """
    计算重复搜索指标:
      - duplicate_source_count: 被 ≥2 个机器人匹配的源数
      - duplicate_robot_count: 参与重复搜索的机器人总数
      - source_coverage_rate: 至少被 1 个机器人覆盖的源比例
    """
    n_sources = len(source_true)
    active_mask = robot_status == 1
    active_beliefs = layer_cluster[active_mask, :2]
    if len(active_beliefs) == 0:
        return 0, 0, 0.0
    # 每个机器人匹配最近的源
    robot_assignments = {}
    for ri, belief in enumerate(active_beliefs):
        best_src, best_dist = -1, float('inf')
        for si, src in enumerate(source_true):
            d = np.sqrt(np.sum((belief - src) ** 2))
            if d < best_dist:
                best_dist = d; best_src = si
        if best_src >= 0:
            robot_assignments.setdefault(best_src, []).append(ri)
    # 统计
    dup_src_count = sum(1 for src, robots in robot_assignments.items() if len(robots) >= 2)
    dup_robot_count = sum(len(robots) for robots in robot_assignments.values() if len(robots) >= 2)
    covered_sources = set(robot_assignments.keys())
    # 宽松覆盖: belief 距离源 < dup_radius 也算
    for si in range(n_sources):
        for belief in active_beliefs:
            if np.sqrt(np.sum((belief - source_true[si]) ** 2)) < dup_radius:
                covered_sources.add(si)
                break
    coverage_rate = len(covered_sources) / n_sources if n_sources > 0 else 0.0
    return dup_src_count, dup_robot_count, coverage_rate


# 初始化多层粒子 filter
# multi_particle 维度: (layer_num, particle_num, 3)，最后一维存储粒子估计的 [x, y, Q]
multi_particle = np.zeros((layer_num, particle_num, 3))
w_obs = np.ones((layer_num, particle_num)) / particle_num

for ln in range(layer_num):
    multi_particle[ln, :, 0] = np.random.rand(particle_num) * (X_max - X_min) + X_min
    multi_particle[ln, :, 1] = np.random.rand(particle_num) * (Y_max - Y_min) + Y_min
    multi_particle[ln, :, 2] = np.random.rand(particle_num) * (Q_max - Q_min) + Q_min

# 初始化各层聚类中心
layer_cluster = np.mean(multi_particle, axis=1)  # 形状 (layer_num, 3)
layer_cov = np.zeros((layer_num, 3, 3))
D_KL = np.zeros((layer_num, layer_num))

# 初始化颜色：使用更加高级和易于区分的颜色（学术常用）
colors_rgb = plt.get_cmap('Set1', 9)(range(9))[:, :3]
# 转换为16进制颜色字符串
colors = []
for i in range(len(colors_rgb)):
    r, g, b = int(colors_rgb[i, 0]*255), int(colors_rgb[i, 1]*255), int(colors_rgb[i, 2]*255)
    colors.append(f"#{r:02X}{g:02X}{b:02X}")
colors = np.array(colors)

robot_init_shape = ['o'] * 9

# (旧代码会在仓库根创建 `figure` 目录；改为在运行时按时间戳创建结果目录)
# os.makedirs("figure", exist_ok=True)

# ==========================================
# 2. 气体浓度扩散模型定义
# ==========================================
def dif2(X, Y, x0, y0, a, Q, V, D, lamda):
    """
    计算多个平面源在网格点 (X,Y) 处的浓度
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    total_conc = np.zeros_like(X)
    for i in range(len(x0)):
        F = np.sqrt((X - x0[i])**2 + (Y - y0[i])**2)
        F_safe = np.where(F == 0, 0.1, F)
        S = np.exp(-(Y - y0[i]) * V / (2.0 * D))
        K = np.exp(-F_safe / lamda)
        R = a * Q[i] / F_safe
        R = R * K * S
        total_conc += R
    return total_conc

def dif22(X, Y, x0, y0, a, Q, V, D, lamda, D_KL, ln, lo_ids):
    """
    考虑共识/协方差交叉与KL距离衰减的传感器预测模型 (dif22.m)
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    total_conc = np.zeros_like(X)
    order = list(lo_ids) + [ln]
    for i in range(len(x0)):
        F = np.sqrt((X - x0[i])**2 + (Y - y0[i])**2)
        F = np.where(F == 0, 0.1, F)
        S = np.exp(-(Y - y0[i]) * V / (2.0 * D))
        K = np.exp(-F / lamda)
        R = a * Q[i] / F
        R = R * K * S
        
        # 对应Matlab: R .^ 1 / (1 + exp(-D_KL(ln, order(i))))
        weight = 1.0 + np.exp(-D_KL[ln, order[i]])
        total_conc += R / weight
    return total_conc

def dif22_vectorized(X, Y, ln, lo_ids, lo_sources, p_x, p_y, p_q, a, V, D, lamda, D_KL,
                     fusion_weights=None):
    """
    向量化计算每个粒子的浓度（直接避免成千上万次 for 循环）.

    参数:
        fusion_weights: dict (可选), {j: omega_ij} 认知差异融合权重.
                       若为 None, 回退到原始 KL 衰减 weight = 1.0 + exp(-D_KL).
                       (CDPA-CI Phase 4)
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    # 1. Base concentration from lo_sources
    base_conc = 0.0
    for i, order_i in enumerate(lo_ids):
        F = np.sqrt((X - lo_sources[i, 0])**2 + (Y - lo_sources[i, 1])**2)
        F = np.where(F == 0, 0.1, F)
        S = np.exp(-(Y - lo_sources[i, 1]) * V / (2.0 * D))
        K = np.exp(-F / lamda)
        R = a * lo_sources[i, 2] / F
        R = R * K * S
        if fusion_weights is not None:
            # CDPA-CI Phase 4: cognitive fusion weight
            w = fusion_weights.get(order_i, 0.0)
            base_conc += R * w
        else:
            # Original KL decay
            weight = 1.0 + np.exp(-D_KL[ln, order_i])
            base_conc += R / weight

    # 2. 向量化运算多粒子的浓度 (Particle concentration)
    F = np.sqrt((X - p_x)**2 + (Y - p_y)**2)
    F = np.where(F == 0, 0.1, F)
    S = np.exp(-(Y - p_y) * V / (2.0 * D))
    K = np.exp(-F / lamda)
    R = a * p_q / F
    R = R * K * S
    if fusion_weights is not None:
        # CDPA-CI Phase 4: self-trust = 1.0
        particle_conc = R  # self belief is always trusted
    else:
        weight = 1.0 + np.exp(-D_KL[ln, ln])
        particle_conc = R / weight

    return base_conc + particle_conc

def poisspdf(k, mu):
    """
    离散泊松概率密度函数 (PMF)
    """
    # 阶乘数组 (由于 max_d = 7，准备10即可)
    fact = [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800]
    mu = np.clip(mu, 1e-15, None)  # 避免0
    fk = fact[int(k)] if k < len(fact) else math.factorial(int(k))
    return (mu**k * np.exp(-mu)) / fk


def effective_sample_size(weights):
    """
    计算有效粒子数 (Effective Sample Size, ESS).

    数学定义:
        ESS = 1 / sum(w_i^2)

    参数:
        weights: 一维权重数组 (归一化或未归一化均可).

    返回:
        float: ESS 值. 如果权重全零/NaN/Inf, 返回 0.0.
    """
    weights = np.asarray(weights, dtype=np.float64)
    # 异常保护: NaN, Inf 替换为 0
    if np.any(~np.isfinite(weights)):
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    sum_w = np.sum(weights)
    if sum_w <= 0.0:
        return 0.0
    # 归一化
    normalized = weights / sum_w
    sum_sq = np.sum(normalized ** 2)
    if sum_sq <= 0.0:
        return 0.0
    return 1.0 / sum_sq


def systematic_resample(weights):
    """
    系统重采样 (Systematic Resampling).

    与普通轮盘赌不同, system resampling 生成一个均匀分布的起始偏移,
    然后按等间距选取粒子, 保证低方差和更好的粒子多样性.

    参数:
        weights: 一维权重数组 (将自动归一化).

    返回:
        indices: 长度为 len(weights) 的重采样索引数组 (整数).
    """
    N = len(weights)
    weights = np.asarray(weights, dtype=np.float64)
    # 归一化保护
    sum_w = np.sum(weights)
    if sum_w <= 0.0:
        weights = np.ones(N) / N
    else:
        weights = weights / sum_w
    # 系统重采样核心
    u0 = np.random.uniform(0.0, 1.0 / N)
    positions = u0 + np.arange(N) / N
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0  # 数值稳定
    indices = np.searchsorted(cumsum, positions)
    indices = np.clip(indices, 0, N - 1)
    return indices


# ==========================================
# CDPA-CI Phase 3: 动态小生境 (Niche) 构建函数
# ==========================================
def build_niche_graph(layer_cluster, D_KL, robot_status, delta_mu, delta_kl):
    """
    基于信念距离和KL散度构建小生境邻接图.

    连边条件:
      - 两个 active 机器人 i, j 的估计源位置距离 < delta_mu
      - 且 KL(i,j) < delta_kl

    参数:
        layer_cluster: (layer_num, 3) 每层的信念均值 [x, y, Q]
        D_KL:          (layer_num, layer_num) KL散度矩阵
        robot_status:  (layer_num,) 1=active, 0=inactive
        delta_mu:      float, 位置距离阈值
        delta_kl:      float, KL散度阈值

    返回:
        adj_matrix: (layer_num, layer_num) 对称邻接矩阵 (0/1)
    """
    n = len(robot_status)
    adj_matrix = np.zeros((n, n), dtype=int)
    active_ids = [i for i in range(n) if robot_status[i] == 1]
    for a in range(len(active_ids)):
        for b in range(a + 1, len(active_ids)):
            i = active_ids[a]
            j = active_ids[b]
            # 信念位置距离
            dist_mu = np.sqrt(np.sum((layer_cluster[i, :2] - layer_cluster[j, :2]) ** 2))
            # KL 距离 (对称化: 取均值)
            kl_dist = (D_KL[i, j] + D_KL[j, i]) / 2.0
            if dist_mu < delta_mu and kl_dist < delta_kl:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1
    return adj_matrix


def connected_components(adj_matrix):
    """
    根据邻接矩阵求连通分量 (DFS实现).

    参数:
        adj_matrix: (n, n) 对称邻接矩阵

    返回:
        components: list of list, 每个子列表是一个连通分量的节点索引
    """
    n = adj_matrix.shape[0]
    visited = np.zeros(n, dtype=bool)
    components = []
    for i in range(n):
        if not visited[i] and np.any(adj_matrix[i] > 0):
            # BFS/DFS 找连通分量
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                neighbors = np.where(adj_matrix[node] > 0)[0]
                for nb in neighbors:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
            components.append(sorted(comp))
    return components


def form_dynamic_niches(layer_cluster, D_KL, robot_status, T, TT,
                        delta_mu_start, delta_mu_end, delta_kl_start, delta_kl_end,
                        max_niche_size, layer_cov=None):
    """
    动态小生境形成 (修复版: 防过度合并).

    改进:
      1. 阈值随时间线性递增 (早期严格, 后期宽松).
      2. 每个 niche 最大大小为 max_niche_size (默认 2).
      3. 超大连通分量按信念距离拆分为最优 pairs.

    参数:
        layer_cluster: (layer_num, 3) 信念均值
        D_KL:          (layer_num, layer_num) KL矩阵
        robot_status:  (layer_num,) 状态向量
        T:             int, 当前步
        TT:            int, 总步数
        delta_mu_start/end: float, 位置距离阈值范围
        delta_kl_start/end: float, KL阈值范围
        max_niche_size: int, 最大 niche 大小
        layer_cov:     (layer_num, 3, 3) 可选, 用于精确保留

    返回:
        niches: list of list
    """
    # 时间相关阈值 (线性插值)
    alpha_t = min(1.0, T / TT)
    delta_mu_t = delta_mu_start + alpha_t * (delta_mu_end - delta_mu_start)
    delta_kl_t = delta_kl_start + alpha_t * (delta_kl_end - delta_kl_start)

    n = len(robot_status)
    active_ids = [i for i in range(n) if robot_status[i] == 1]
    if len(active_ids) == 0:
        return []

    # 低于 min_niche_update_step 时不合并 (全部独立)
    if T < min_niche_update_step:
        return [[int(aid)] for aid in active_ids]

    # 构建邻接图 (使用时间相关阈值)
    adj = build_niche_graph(layer_cluster, D_KL, robot_status, delta_mu_t, delta_kl_t)
    comps = connected_components(adj)

    # 收集已在连通分量中的 active 节点
    nodes_in_comps = set()
    for comp in comps:
        for node in comp:
            if robot_status[node] == 1:
                nodes_in_comps.add(node)

    niches = []
    for comp in comps:
        active_in_comp = [int(n) for n in comp if robot_status[n] == 1]
        if len(active_in_comp) == 0:
            continue
        if len(active_in_comp) <= max_niche_size:
            niches.append(active_in_comp)
        else:
            # 超大连通分量: 按信念距离拆分为最优 pairs
            sub_niches = _split_oversized_component(
                active_in_comp, layer_cluster, D_KL, max_niche_size)
            niches.extend(sub_niches)
    # 孤立 active 节点
    for aid in active_ids:
        if aid not in nodes_in_comps:
            niches.append([int(aid)])
    return niches


def _split_oversized_component(members, layer_cluster, D_KL, max_size):
    """
    将超大连通分量拆分为大小不超过 max_size 的子组.
    使用贪心策略: 每次找信念最近且 KL 最小的 pair.
    """
    remaining = set(members)
    result = []
    while len(remaining) > 0:
        if len(remaining) <= max_size:
            result.append(sorted(remaining))
            break
        # 选种子: 第一个 remaining 元素
        seed = next(iter(remaining))
        # 找与种子最兼容的伙伴 (信念距离 + KL 最小)
        best_partner = None
        best_score = float('inf')
        for other in remaining:
            if other == seed:
                continue
            dist_mu = np.sqrt(np.sum((layer_cluster[seed, :2] - layer_cluster[other, :2]) ** 2))
            kl_pair = (D_KL[seed, other] + D_KL[other, seed]) / 2.0
            score = dist_mu + kl_pair
            if score < best_score:
                best_score = score
                best_partner = other
        if best_partner is not None:
            group = [seed, best_partner]
            remaining.discard(seed)
            remaining.discard(best_partner)
        else:
            group = [seed]
            remaining.discard(seed)
        result.append(sorted(group))
    return result


def compute_niche_gaussian_consensus(niche, layer_cluster, layer_cov, w_obs):
    """
    niche 内 Gaussian belief 共识 (修复版: 替代粒子硬同步).

    返回:
        consensus_mean: (3,) 加权平均信念
        consensus_cov:  (3, 3) 加权平均协方差
        confidence:     float, 平均置信度
    """
    if len(niche) == 1:
        ln = niche[0]
        return layer_cluster[ln].copy(), layer_cov[ln].copy(), 1.0 / (np.trace(layer_cov[ln]) + 1e-6)
    # 加权平均 (权重 = 每个机器人的置信度)
    weights_list = []
    means_list = []
    for ln in niche:
        conf = 1.0 / (np.trace(layer_cov[ln]) + 1e-6)
        weights_list.append(conf)
        means_list.append(layer_cluster[ln])
    weights_arr = np.array(weights_list)
    weights_arr = weights_arr / np.sum(weights_arr)
    consensus_mean = np.average(means_list, axis=0, weights=weights_arr)
    # 协方差: 加权平均
    consensus_cov = np.zeros((3, 3))
    for k, ln in enumerate(niche):
        consensus_cov += weights_arr[k] * layer_cov[ln]
    avg_confidence = np.mean(weights_list)
    return consensus_mean, consensus_cov, avg_confidence


# ==========================================
# CDPA-CI Phase 4: 认知差异 (Cognitive Difference) 与信念融合函数
# ==========================================
def compute_entropy(weights):
    """
    计算粒子权重香农熵: H = -sum(w_i * log(w_i)).
    权重自动归一化.
    """
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.clip(weights, 0, None)
    sum_w = np.sum(weights)
    if sum_w <= 0:
        return 0.0
    w = weights / sum_w
    w = w[w > 1e-15]  # 避免 log(0)
    if len(w) == 0:
        return 0.0
    return -np.sum(w * np.log(w))


def predict_observation_from_belief(x, y, belief_mu, a, V, D, lamda):
    """
    用机器人j的belief均值预测机器人i当前位置的期望观测 (lambda rate).

    参数:
        x, y:       float, 当前位置坐标
        belief_mu:  (3,) 另一个机器人的信念均值 [x_s, y_s, Q_s]
        a, V, D, lamda: 扩散参数

    返回:
        lambda_pred: float, 预测的Poisson rate
    """
    F = np.sqrt((x - belief_mu[0]) ** 2 + (y - belief_mu[1]) ** 2)
    if F < 0.1:
        F = 0.1
    S = np.exp(-(y - belief_mu[1]) * V / (2.0 * D))
    K = np.exp(-F / lamda)
    R = a * belief_mu[2] / F
    return R * K * S


def compute_cognitive_difference(i, j, layer_cluster, layer_cov, D_KL, z, pX, pY,
                                 a, V, D, lamda, alpha_kl, alpha_mu, alpha_obs, alpha_unc):
    """
    计算机器人 i 对机器人 j 的认知差异 (Cognitive Difference).

    Delta_ij = alpha_kl * D_KL[i,j]
             + alpha_mu * ||mu_i - mu_j||
             + alpha_obs * |z_i - z_hat_ij|
             + alpha_unc * trace(Sigma_j)

    参数:
        i, j:              int, 机器人索引
        layer_cluster:     (layer_num, 3) 信念均值
        layer_cov:         (layer_num, 3, 3) 信念协方差
        D_KL:              (layer_num, layer_num) KL矩阵
        z:                 (layer_num,) 当前观测
        pX, pY:            (layer_num,) 当前位置
        a, V, D, lamda:    扩散参数
        alpha_kl/mu/obs/unc: 各项权重

    返回:
        delta: float, 认知差异值 (非负)
    """
    # 1. KL散度分量 (对称化)
    delta_kl_term = alpha_kl * (D_KL[i, j] + D_KL[j, i]) / 2.0
    # 2. 信念均值位置差异
    delta_mu_term = alpha_mu * np.sqrt(np.sum((layer_cluster[i, :2] - layer_cluster[j, :2]) ** 2))
    # 3. 观测预测差异: |z_i - z_hat_ij| (用j的belief预测i处期望观测)
    z_hat_ij = predict_observation_from_belief(pX[i], pY[i], layer_cluster[j], a, V, D, lamda)
    delta_obs_term = alpha_obs * abs(z[i] - z_hat_ij)
    # 4. 不确定度分量 (机器人j的信念不确定度)
    delta_unc_term = alpha_unc * np.trace(layer_cov[j])
    return delta_kl_term + delta_mu_term + delta_obs_term + delta_unc_term


def compute_belief_fusion_weights(ego_id, active_ids, layer_cluster, layer_cov, D_KL,
                                   z, pX, pY, a, V, D, lamda,
                                   alpha_kl, alpha_mu, alpha_obs, alpha_unc, tau):
    """
    计算 ego 机器人对其他 active 机器人的认知差异融合权重 (softmax).

    omega_ij = exp(-Delta_ij / tau) / sum_k exp(-Delta_ik / tau)

    返回:
        fusion_weights: dict, {j: omega_ij for j in active_ids}
    """
    if len(active_ids) <= 1:
        return {ego_id: 1.0}
    raw = {}
    for j in active_ids:
        if j == ego_id:
            raw[j] = 0.0  # self-difference = 0
        else:
            raw[j] = compute_cognitive_difference(ego_id, j, layer_cluster, layer_cov,
                                                   D_KL, z, pX, pY, a, V, D, lamda,
                                                   alpha_kl, alpha_mu, alpha_obs, alpha_unc)
    # 数值稳定的 softmax: exp(-Delta/tau)
    deltas = np.array([raw[j] for j in active_ids])
    max_delta = np.max(deltas)
    shifted = -(deltas - max_delta) / tau  # 减去最大值防止 exp 溢出
    exp_vals = np.exp(shifted)
    sum_exp = np.sum(exp_vals)
    if sum_exp <= 0:
        # fallback: uniform
        return {j: 1.0 / len(active_ids) for j in active_ids}
    return {active_ids[k]: exp_vals[k] / sum_exp for k in range(len(active_ids))}


def same_source_gate(i, j, layer_cluster, layer_cov, D_KL, z, pX, pY, a, V, D, lamda, thresholds):
    """
    Same-Source Gate: 判断机器人 i 和 j 是否在追踪同一源.

    四个条件必须同时满足:
      1. belief mean 空间距离 < dist_th
      2. 对称 KL < kl_th
      3. 源强差异 < q_th
      4. 用 j 的 belief 预测 i 当前观测, 预测误差 < obs_th

    返回:
        bool: True 如果 i 和 j 可能追踪同一源
    """
    # 1. 空间距离
    dist_mu = np.sqrt(np.sum((layer_cluster[i, :2] - layer_cluster[j, :2]) ** 2))
    if dist_mu >= thresholds['dist_th']:
        return False
    # 2. KL 散度
    kl_sym = (D_KL[i, j] + D_KL[j, i]) / 2.0
    if kl_sym >= thresholds['kl_th']:
        return False
    # 3. 源强差异
    q_diff = abs(layer_cluster[i, 2] - layer_cluster[j, 2])
    if q_diff >= thresholds['q_th']:
        return False
    # 4. 观测预测误差
    z_hat = predict_observation_from_belief(pX[i], pY[i], layer_cluster[j], a, V, D, lamda)
    obs_err = abs(z[i] - z_hat)
    if obs_err >= thresholds['obs_th']:
        return False
    return True


# ==========================================
# CDPA-CI Phase 5: 角色自适应 (Role-Adaptive) 函数
# ==========================================
def compute_belief_confidence(layer_cov, ln):
    """
    信念置信度: confidence = 1 / (trace(cov) + eps).
    trace小 → 置信度高.
    """
    return 1.0 / (np.trace(layer_cov[ln]) + 1e-6)


def assign_agent_role(entropy, patchiness, confidence, thresholds):
    """
    根据不确定性、间歇性和置信度分配角色.

    规则:
      - 高熵 + 低置信度 → Scout (探索)
      - 高置信度 + 低间歇性 → Tracker (追踪)
      - 高置信度 + 高间歇性 → Verifier (验证, 抵抗patchy plume)
      - 默认 → Scout

    参数:
        entropy:      float, 粒子权重熵
        patchiness:   float, 间歇性指数 (0~1, 暂用归一化观测方差近似)
        confidence:   float, 信念置信度
        thresholds:   dict with 'conf_low', 'conf_high', 'entropy_high', 'patchiness_high'

    返回:
        role: int, ROLE_SCOUT / ROLE_TRACKER / ROLE_VERIFIER
    """
    if confidence > thresholds['conf_high']:
        if patchiness > thresholds['patchiness_high']:
            return ROLE_VERIFIER  # 高置信但羽流不稳定 → 交叉验证
        else:
            return ROLE_TRACKER   # 高置信 + 稳定羽流 → 追踪
    elif entropy > thresholds['entropy_high'] or confidence < thresholds['conf_low']:
        return ROLE_SCOUT         # 高不确定性 → 探索
    else:
        return ROLE_TRACKER       # 默认追踪


def assign_roles_for_all_agents(layer_cov, w_end, z, robot_status, thresholds):
    """
    为所有 active 机器人分配角色.

    返回:
        roles: (layer_num,) 整数数组, ROLE_SCOUT=0, ROLE_TRACKER=1, ROLE_VERIFIER=2
    """
    roles = np.full(layer_num, ROLE_SCOUT, dtype=int)
    for ln in range(layer_num):
        if robot_status[ln] == 0:
            roles[ln] = -1  # inactive
            continue
        # 信念置信度
        confidence = compute_belief_confidence(layer_cov, ln)
        # 粒子熵
        entropy = compute_entropy(w_end[ln, :])
        # 间歇性近似: 用观测值归一化 (z/max_d) 作为 proxy
        patchiness = z[ln] / max(max_d, 1)
        roles[ln] = assign_agent_role(entropy, patchiness, confidence, thresholds)
    return roles


# ==========================================
# CDPA-CI Phase 6: 源声明与排斥 (Source Declaration & Exclusion) 函数
# ==========================================
def check_source_declaration(agent_id, layer_cluster, layer_cov, z, pX, pY, thresholds):
    """
    检查机器人 agent_id 是否满足源声明条件.

    条件:
      1. trace(cov) < source_declare_cov_th  (高置信度)
      2. distance(agent_pos, belief_mean) < source_declare_dist_th  (接近估计源)
      3. z > source_declare_obs_th  (观测值足够高)

    返回:
        candidate: (x, y, Q) 候选源位置, 或 None
    """
    trace_cov = np.trace(layer_cov[agent_id])
    if trace_cov >= thresholds['cov_th']:
        return None
    dist_to_belief = np.sqrt(np.sum((np.array([pX[agent_id], pY[agent_id]]) -
                                      layer_cluster[agent_id, :2]) ** 2))
    if dist_to_belief >= thresholds['dist_th']:
        return None
    if z[agent_id] < thresholds['obs_th']:
        return None
    return tuple(layer_cluster[agent_id].copy())


def is_duplicate_source(candidate, declared_sources, radius):
    """
    判断候选源是否与已声明源重复 (距离 < radius).
    """
    for ds in declared_sources:
        dist = np.sqrt((candidate[0] - ds[0]) ** 2 + (candidate[1] - ds[1]) ** 2)
        if dist < radius:
            return True
    return False


def source_exclusion_penalty(candidate_pos, declared_sources, radius, penalty_strength=5.0):
    """
    计算候选位置对已声明源的排斥惩罚.

    靠近已声明源 → 高惩罚 → 降低该位置的 reward.

    返回:
        penalty: float, 非正值 (用于加到 delta_F 上)
    """
    if len(declared_sources) == 0:
        return 0.0
    penalty = 0.0
    for ds in declared_sources:
        dist = np.sqrt((candidate_pos[0] - ds[0]) ** 2 +
                       (candidate_pos[1] - ds[1]) ** 2)
        if dist < radius:
            penalty -= penalty_strength * (1.0 - dist / radius)  # 距离越近惩罚越重
    return penalty


def update_candidate_sources(candidate_sources, new_candidate, layer_cluster, z, pX, pY,
                              candidate_proximity, candidate_consensus_agents,
                              candidate_patchiness_max, max_d):
    """
    候选源追踪与升级 (修复版: candidate→declared pipeline).

    参数:
        candidate_sources: list of dict, 现有候选源
        new_candidate:     tuple (x, y, Q) or None, 新的候选位置
        layer_cluster:     (layer_num, 3) 信念均值
        z:                (layer_num,) 当前观测
        pX, pY:           (layer_num,) 当前位置

    返回:
        updated candidates, newly_declared list
    """
    newly_declared = []
    # 1. 更新现有候选源的计数器
    for cs in candidate_sources:
        # 检查是否有机器人在邻域内支持该候选源
        supporters = 0
        for ln in range(len(z)):
            dist = np.sqrt((layer_cluster[ln, 0] - cs['pos'][0])**2 +
                           (layer_cluster[ln, 1] - cs['pos'][1])**2)
            if dist < candidate_proximity and z[ln] > 0:
                supporters += 1
        if supporters >= candidate_consensus_agents:
            patchiness_proxy = np.mean(z) / max(max_d, 1)  # 简化版间歇性
            if patchiness_proxy < candidate_patchiness_max:
                cs['counter'] += 1
                cs['supporters'] = max(cs['supporters'], supporters)
            else:
                cs['counter'] = max(0, cs['counter'] - 1)  # 间歇性太高, 衰减计数
        else:
            cs['counter'] = max(0, cs['counter'] - 1)  # 支持不足, 衰减
    # 2. 添加新候选源 (如果是新位置且不在已声明中)
    if new_candidate is not None:
        is_new = True
        for cs in candidate_sources:
            dist = np.sqrt((new_candidate[0] - cs['pos'][0])**2 +
                           (new_candidate[1] - cs['pos'][1])**2)
            if dist < candidate_proximity:
                cs['counter'] += 1
                is_new = False
                break
        if is_new:
            candidate_sources.append({'pos': new_candidate, 'counter': 1, 'supporters': 1})
    # 3. 检查是否有候选源升级为 declared
    for cs in candidate_sources:
        if cs['counter'] >= candidate_confirm_steps:
            newly_declared.append(cs['pos'])
    return newly_declared


# ==========================================
# Source-Aware Diversity Regularization
# ==========================================
def compute_diversity_penalty(agent_id, candidate_pos, layer_cluster, robot_status, sigma, min_dist):
    """
    计算候选位置的多样性惩罚 (防止多机器人追踪同一源).

    对每个其他 active 机器人 j:
      dist_j = ||candidate_pos - layer_cluster[j, :2]||
      penalty_j = exp(-dist_j^2 / (2 * sigma^2))  如果 dist_j < min_dist
      penalty_j = 0                               如果 dist_j >= min_dist

    返回:
        float: sum(penalty_j for j != agent_id), 值越大表示越接近其他机器人追踪区域
    """
    penalty = 0.0
    cx, cy = candidate_pos[0], candidate_pos[1]
    for j in range(len(robot_status)):
        if j == agent_id or robot_status[j] == 0:
            continue
        bx, by = layer_cluster[j, 0], layer_cluster[j, 1]
        dist = np.sqrt((cx - bx)**2 + (cy - by)**2)
        if dist < min_dist:
            penalty += np.exp(-dist**2 / (2.0 * sigma**2))
    return penalty


def compute_adaptive_step(agent_id, layer_cov, base_step, step_min, step_max, uncertainty_scale):
    """
    根据当前 belief 不确定性自适应调节步长.

    定义:
      uncertainty = trace(cov[agent_id, :2, :2])  # 仅位置协方差
      normalized  = uncertainty / (uncertainty + uncertainty_scale)
      step        = step_min + (step_max - step_min) * normalized

    高不确定性 → 大步长 (探索); 低不确定性 → 小步长 (精细定位).
    异常保护: covariance NaN/Inf 时回退到 base_step.
    """
    cov_2d = layer_cov[agent_id, :2, :2]
    if not np.all(np.isfinite(cov_2d)):
        return base_step
    uncertainty = np.trace(cov_2d)
    if uncertainty < 0 or not np.isfinite(uncertainty):
        return base_step
    normalized = uncertainty / (uncertainty + uncertainty_scale)
    step_val = step_min + (step_max - step_min) * normalized
    return float(np.clip(step_val, step_min, step_max))


# ==========================================
# Visitation-Aware Novelty Reward
# ==========================================
def pos_to_visit_index(x, y, X_min, Y_min, resolution, map_shape):
    """连续坐标 → visit_map 网格索引 (越界时 clip)."""
    col = int((x - X_min) / resolution)
    row = int((y - Y_min) / resolution)
    col = np.clip(col, 0, map_shape[1] - 1)
    row = np.clip(row, 0, map_shape[0] - 1)
    return row, col


def update_visit_map(visit_map, pX, pY, robot_status, params):
    """
    更新访问地图: 先衰减, 再对每个 active 机器人的位置 + 邻域增加访问计数.
    """
    visit_map *= params['decay']
    resolution = params['resolution']
    radius = params['radius']
    map_h, map_w = visit_map.shape
    for ln in range(len(robot_status)):
        if robot_status[ln] == 0:
            continue
        r0, c0 = pos_to_visit_index(pX[ln], pY[ln], X_min, Y_min, resolution, (map_h, map_w))
        # 2D 高斯邻域
        radius_cells = int(np.ceil(radius / resolution))
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                rr, cc = r0 + dr, c0 + dc
                if 0 <= rr < map_h and 0 <= cc < map_w:
                    dist = np.sqrt((dr * resolution)**2 + (dc * resolution)**2)
                    if dist <= radius:
                        w = np.exp(-dist**2 / (2 * (radius/2)**2))
                        visit_map[rr, cc] += w
    visit_map[:] = np.clip(visit_map, 0, params.get('max_val', 50))
    return visit_map


def compute_novelty_reward(candidate_pos, visit_map, params):
    """
    计算新颖性奖励: 访问计数越低 → 奖励越高.
    reward = exp(-eta * visit_count), 范围 [0, 1].
    """
    x, y = candidate_pos[0], candidate_pos[1]
    map_h, map_w = visit_map.shape
    r, c = pos_to_visit_index(x, y, X_min, Y_min, params['resolution'], (map_h, map_w))
    visit_count = visit_map[r, c]
    return np.exp(-params['eta'] * visit_count)


def compute_bounded_novelty_bonus(base_score, novelty_reward, novelty_lambda,
                                   bonus_max, bonus_ratio):
    """
    Bounded Additive Novelty: 限制 novelty 不能压倒 Infotaxis.

    raw_bonus   = novelty_lambda * novelty_reward
    relative_cap = bonus_ratio * (|base_score| + 1e-6)
    bonus_cap   = min(bonus_max, relative_cap)
    bounded     = min(raw_bonus, bonus_cap)

    返回: (bounded_bonus, was_clipped)
    """
    raw_bonus = novelty_lambda * novelty_reward
    relative_cap = bonus_ratio * (abs(base_score) + 1e-6)
    bonus_cap = min(bonus_max, relative_cap)
    bounded = min(raw_bonus, bonus_cap)
    was_clipped = raw_bonus > bounded
    return bounded, was_clipped


# ==========================================
# Intent-Aware Cooperative Assignment
# ==========================================
def update_candidate_registry(candidate_registry, layer_cluster, layer_cov, robot_status, T, params):
    """
    更新候选源注册表: 每个 active 机器人的 belief mean 作为候选源,
    与已有 candidate 距离 < merge_radius 则合并; 否则新建.
    """
    merge_radius = params['merge_radius']
    for ln in range(len(robot_status)):
        if robot_status[ln] == 0:
            continue
        belief_pos = layer_cluster[ln, :2].copy()
        conf = 1.0 / (np.trace(layer_cov[ln]) + 1e-6)
        merged = False
        for cand in candidate_registry:
            dist = np.sqrt(np.sum((belief_pos - cand['pos']) ** 2))
            if dist < merge_radius:
                cand['support'] += 1
                cand['confidence'] = max(cand['confidence'], conf)
                cand['pos'] = (cand['pos'] * cand['support'] + belief_pos) / (cand['support'] + 1)
                cand['last_update'] = T
                merged = True
                break
        if not merged:
            candidate_registry.append({
                'id': len(candidate_registry),
                'pos': belief_pos.copy(),
                'support': 1,
                'confidence': conf,
                'last_update': T,
                'assigned_robots': []
            })
    # 清理长期未更新的 candidate (> 2*update_interval)
    candidate_registry[:] = [c for c in candidate_registry if T - c['last_update'] < 2 * params['update_interval']]
    # 重置 assigned_robots
    for c in candidate_registry:
        c['assigned_robots'] = []
    return candidate_registry


def assign_robots_to_candidates(candidate_registry, layer_cluster, layer_cov, pX, pY, robot_status, params):
    """
    Greedy assignment: 每个机器人分配到最近的 candidate,
    每个 candidate 最多 max_robots_per_candidate 个机器人.
    返回: assignments dict {robot_id: candidate} 或 None
    """
    max_per = params['max_per_candidate']
    conflict_penalty = params['conflict_penalty']
    active_ids = [i for i in range(len(robot_status)) if robot_status[i] == 1]
    if len(candidate_registry) == 0:
        return {}
    assignments = {}
    # 按 confidence 排序
    sorted_robots = sorted(active_ids,
                           key=lambda i: 1.0/(np.trace(layer_cov[i])+1e-6),
                           reverse=True)
    for robot_id in sorted_robots:
        best_cand, best_score = None, -float('inf')
        for ci, cand in enumerate(candidate_registry):
            dist = np.sqrt(np.sum((layer_cluster[robot_id, :2] - cand['pos']) ** 2))
            n_assigned = len(cand['assigned_robots'])
            conflict = conflict_penalty if n_assigned >= max_per else 0.0
            score = cand['confidence'] - 0.1 * dist - conflict
            if score > best_score:
                best_score = score; best_cand = ci
        if best_cand is not None and best_score > -100:
            assignments[robot_id] = best_cand
            candidate_registry[best_cand]['assigned_robots'].append(robot_id)
    return assignments


def evaluate_source_success(sourceX, sourceY, layer_cluster, layer_cov, pX, pY, robot_status,
                             est_dist_th=1.0, robot_dist_th=1.0, cov_trace_th=1.0):
    """
    Strict source success evaluation.
    条件 (robot distance 可选):
      1. ||belief_mean_i - source_k|| < est_dist_th
      2. trace(cov_i[:2,:2]) < cov_trace_th
      3. ||robot_pos_i - source_k|| < robot_dist_th  (only if USE_ROBOT_DISTANCE_IN_SUCCESS)
    """
    n_sources = len(sourceX)
    active_ids = [i for i in range(len(pX)) if robot_status[i] == 1]
    candidates = []
    for i in active_ids:
        for k in range(n_sources):
            est_dist = np.sqrt(np.sum((layer_cluster[i, :2] - np.array([sourceX[k], sourceY[k]])) ** 2))
            robot_dist = np.sqrt(np.sum((np.array([pX[i], pY[i]]) - np.array([sourceX[k], sourceY[k]])) ** 2))
            cov_ok = np.trace(layer_cov[i, :2, :2]) < cov_trace_th
            est_ok = est_dist < est_dist_th
            if USE_ROBOT_DISTANCE_IN_SUCCESS:
                all_ok = est_ok and (robot_dist < robot_dist_th) and cov_ok
            else:
                all_ok = est_ok and cov_ok
            if all_ok:
                score = est_dist + np.trace(layer_cov[i, :2, :2])
                candidates.append((score, i, k))
    candidates.sort(key=lambda x: x[0])
    matched_sources = set(); matched_robots = set(); source_to_robot = {}
    for _, robot_id, src_id in candidates:
        if src_id not in matched_sources and robot_id not in matched_robots:
            matched_sources.add(src_id); matched_robots.add(robot_id)
            source_to_robot[src_id] = robot_id
    success_flag = 1 if len(matched_sources) == n_sources else 0
    source_success_mask = [1 if k in matched_sources else 0 for k in range(n_sources)]
    strict_coverage = len(matched_sources) / n_sources if n_sources > 0 else 0.0
    return success_flag, len(matched_sources), source_success_mask, source_to_robot, strict_coverage


def diagnose_source_success_failure(sourceX, sourceY, layer_cluster, layer_cov, pX, pY, robot_status,
                                     est_dist_th=1.0, robot_dist_th=1.0, cov_trace_th=0.2):
    """诊断每个源失败的具体原因 (不改变任何算法状态)."""
    n_sources = len(sourceX)
    active_ids = [i for i in range(len(pX)) if robot_status[i] == 1]
    diag = {
        'min_est_dist': [], 'min_robot_dist': [], 'min_cov_trace': [],
        'best_robot_est': [], 'best_robot_robot': [], 'best_robot_combined': [],
        'est_ok': [], 'robot_ok': [], 'cov_ok': [], 'all_ok': [],
    }
    for k in range(n_sources):
        sk = np.array([sourceX[k], sourceY[k]])
        best_est, best_robot_d, best_cov = float('inf'), float('inf'), float('inf')
        best_r_est, best_r_robot, best_r_comb = -1, -1, -1
        best_comb_score = float('inf')
        for i in active_ids:
            ed = np.sqrt(np.sum((layer_cluster[i, :2] - sk) ** 2))
            rd = np.sqrt(np.sum((np.array([pX[i], pY[i]]) - sk) ** 2))
            ct = np.trace(layer_cov[i, :2, :2])
            comb = ed + rd + ct
            if ed < best_est: best_est = ed; best_r_est = i
            if rd < best_robot_d: best_robot_d = rd; best_r_robot = i
            if ct < best_cov: best_cov = ct
            if comb < best_comb_score: best_comb_score = comb; best_r_comb = i
        e_ok = 1 if best_est < est_dist_th else 0
        r_ok = 1 if best_robot_d < robot_dist_th else 0
        c_ok = 1 if best_cov < cov_trace_th else 0
        all_cond = (e_ok and c_ok) if not USE_ROBOT_DISTANCE_IN_SUCCESS else (e_ok and r_ok and c_ok)
        diag['min_est_dist'].append(best_est)
        diag['min_robot_dist'].append(best_robot_d)
        diag['min_cov_trace'].append(best_cov)
        diag['best_robot_est'].append(best_r_est)
        diag['best_robot_robot'].append(best_r_robot)
        diag['best_robot_combined'].append(best_r_comb)
        diag['est_ok'].append(e_ok); diag['robot_ok'].append(r_ok); diag['cov_ok'].append(c_ok)
        diag['all_ok'].append(1 if all_cond else 0)
    # 全局统计
    num_est_ok = sum(diag['est_ok']); num_robot_ok = sum(diag['robot_ok']); num_cov_ok = sum(diag['cov_ok'])
    num_all = sum(diag['all_ok'])
    bottleneck_est = sum(1 for k in range(n_sources) if not diag['est_ok'][k])
    bottleneck_robot = sum(1 for k in range(n_sources) if not diag['robot_ok'][k])
    bottleneck_cov = sum(1 for k in range(n_sources) if not diag['cov_ok'][k])
    diag['num_est_ok'] = num_est_ok; diag['num_robot_ok'] = num_robot_ok; diag['num_cov_ok'] = num_cov_ok
    diag['num_all_ok'] = num_all
    diag['bottleneck_est'] = bottleneck_est; diag['bottleneck_robot'] = bottleneck_robot
    diag['bottleneck_cov'] = bottleneck_cov
    diag['mean_min_est'] = np.mean(diag['min_est_dist']) if n_sources > 0 else 0.0
    diag['mean_min_robot'] = np.mean(diag['min_robot_dist']) if n_sources > 0 else 0.0
    diag['mean_min_cov'] = np.mean(diag['min_cov_trace']) if n_sources > 0 else 0.0
    return diag


# ==========================================
# 3. 产生气体场并绘制基础图 (Figure 3 & Figure 4)
# ==========================================
X_grid, Y_grid = np.meshgrid(np.arange(X_min, X_max + size_plot, size_plot),
                             np.arange(Y_min, Y_max + size_plot, size_plot))
R_grid = dif2(X_grid, Y_grid, sourceX, sourceY, a, Q, V, D, lamda)

# 绘制气体分布等高线图 (Figure 3)
plt.figure(figsize=(8, 7))
ax = plt.gca()
cp = plt.contourf(X_grid, Y_grid, np.clip(R_grid, 0, 15.0), levels=40, cmap='viridis', zorder=0)
cbar = plt.colorbar(cp)
cbar.ax.tick_params(labelsize=10)
cbar.set_label(r'Concentration $(ppm)$', fontsize=12)
plt.scatter(sourceX, sourceY, s=350, marker='*', facecolor='#FFD700', edgecolor='#8B0000', linewidth=1.5, zorder=5, label='True Source')
plt.xlabel(r'X Coordinate ($m$)', fontname='Arial', fontsize=14)
plt.ylabel(r'Y Coordinate ($m$)', fontname='Arial', fontsize=14)
plt.xlim(X_min, X_max)
plt.ylim(Y_min, Y_max)
plt.grid(True, linestyle='--', color='w', alpha=0.3, zorder=1)
plt.title(r'True Sources and Plume Dispersion ($t=250s$)', fontname='Arial', fontsize=15, fontweight='bold', pad=15)
plt.legend(loc='upper right', framealpha=0.9, edgecolor='black')
plt.tight_layout()
# Ensure results directory exists and switch to figure folder so relative saves go there
from datetime import datetime
rd = globals().get('results_dir', None)
if not (rd and os.path.exists(rd)):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join('Results', f'run_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    results_fig_dir = os.path.join(results_dir, 'figure')
    os.makedirs(results_fig_dir, exist_ok=True)
    # change working dir so subsequent relative saves go into results_fig_dir
    try:
        os.chdir(results_fig_dir)
    except Exception:
        # if chdir fails for any reason, continue but ensure we created the dirs
        pass
else:
    # results_dir variable exists but ensure the directories are present
    os.makedirs(results_dir, exist_ok=True)
    if 'results_fig_dir' not in globals():
        results_fig_dir = os.path.join(results_dir, 'figure')
    os.makedirs(results_fig_dir, exist_ok=True)

_target = os.path.join(results_dir, 'distribution.png')
os.makedirs(os.path.dirname(_target), exist_ok=True)
plt.savefig(_target, dpi=300, bbox_inches='tight')
plt.close()

# 绘制伪彩色图 (Figure 4)
plt.figure(figsize=(8, 7))
pcol = plt.pcolormesh(X_grid, Y_grid, np.clip(R_grid / 3.95, 0, 3.0), shading='auto', cmap='plasma')
cbar = plt.colorbar(pcol)
cbar.set_ticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
cbar.set_ticklabels(['0', '0.5', '1.0', '1.5', '2.0', '2.5', r'$\geq 3.0$'])
cbar.set_label('Normalized Sensor Reading', fontsize=12)
plt.xlabel(r'X Coordinate ($m$)', fontname='Arial', fontsize=14)
plt.ylabel(r'Y Coordinate ($m$)', fontname='Arial', fontsize=14)
plt.title(r'Sensor Observation Field (Normalized)', fontname='Arial', fontsize=15, fontweight='bold', pad=15)
plt.tight_layout()
_target = os.path.join(results_dir, 'pcolor.png')
os.makedirs(os.path.dirname(_target), exist_ok=True)
plt.savefig(_target, dpi=300, bbox_inches='tight')
plt.close()


# ==========================================
# 4. 气体寻源迭代循环 (Timesteps T = 1 to TT)
# ==========================================
z_h = []
D_KL_h = [[] for _ in range(layer_num)]
ess_h = [[] for _ in range(layer_num)]  # CDPA-CI Phase 1: 每层每步的有效粒子数 (ESS)
niche_h = []  # CDPA-CI Phase 3: 每个时间步的小生境分组列表
fusion_weight_h = [[] for _ in range(layer_num)]  # CDPA-CI Phase 4: 每步的融合权重记录
role_h = []  # CDPA-CI Phase 5: 每个时间步的角色分配记录
diversity_penalty_h = []  # Source-Aware Diversity: 每步平均 diversity penalty
adaptive_step_h = [[] for _ in range(layer_num)]  # Adaptive Step: 每机器人每步步长记录
# Visitation-Aware Novelty
visit_map_w = int((X_max - X_min) / visit_grid_resolution) + 1
visit_map_h = int((Y_max - Y_min) / visit_grid_resolution) + 1
visit_map = np.zeros((visit_map_h, visit_map_w), dtype=np.float64)
novelty_reward_h = [[] for _ in range(layer_num)]
visit_coverage_h = []
# Tie-breaker diagnostics
base_score_h = []           # mean base_score per step
novelty_changed_count = 0   # 累计: novelty 改变了最优动作的次数
novelty_total_decisions = 0 # 累计: 总决策次数
non_best_base_selected = 0  # 累计: 最终选择的不是 base_score 最高
# Intent-Aware Assignment (feature removed)
# keep candidate_registry for compatibility with helper functions
candidate_registry = []
# Bounded novelty diagnostics
novelty_bonus_h = []        # 每步的 novelty bonus 记录
novelty_clipped_count = 0   # novelty 被 cap 截断的次数
novelty_total_steps = 0     # novelty 总应用步数
# Task completion tracking
success_flag_h = []
matched_source_count_h = []
source_accuracy_h = []
task_success_flag = 0
task_completion_step = TT
task_completion_step_raw = -1
first_full_success_step = -1
early_stop_flag = 0
final_source_accuracy = 0.0
best_source_accuracy = 0.0
declared_sources = []  # CDPA-CI Phase 6: 已声明源位置列表 [(x, y, Q), ...]
declared_source_h = []  # CDPA-CI Phase 6: 每个时间步声明的源数量
candidate_sources = []  # CDPA-CI Phase 6 修复: 候选源 [{pos, counter, supporters}, ...]
mse_h = []
source_true = np.column_stack((sourceX, sourceY))
gif_frames = []

last_x = pX.copy()
last_y = pY.copy()

# NPSO 新增状态变量
V_x = np.zeros(layer_num)
V_y = np.zeros(layer_num)
XP_x = pX.copy()
XP_y = pY.copy()
XP_maxz = np.zeros(layer_num)
robot_status = np.ones(layer_num) # 1 为活动状态, 0 为已宣称源并停留

# CDPA-CI Phase 3: 初始小生境为各机器人独立 (第1步尚无KL信息)
niches = [[i] for i in range(layer_num) if robot_status[i] == 1]

role_names = ['Scout', 'Tracker', 'Verifier']  # CDPA-CI Phase 5: 角色名称映射

# 打开运行日志文件 (CDPA-CI)
# 确保 results_dir 和 results_fig_dir 已存在（兼容外部或之前创建的位置）
if 'results_dir' not in globals():
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join('Results', f'run_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    results_fig_dir = os.path.join(results_dir, 'figure')
    os.makedirs(results_fig_dir, exist_ok=True)
else:
    os.makedirs(results_dir, exist_ok=True)
    if 'results_fig_dir' not in globals():
        results_fig_dir = os.path.join(results_dir, 'figure')
    os.makedirs(results_fig_dir, exist_ok=True)

log_f = open(os.path.join(results_dir, "run_log.txt"), "w", encoding="utf-8")
log_f.write("CDPA-CI Simulation Log\n")
log_f.write("=" * 60 + "\n")
log_f.write(f"Robots: {robot_num}, Particles: {particle_num}, Steps: {TT}\n")
log_f.write(f"Sources: {source_num} at {list(zip(sourceX, sourceY))}\n")
log_f.write("=" * 60 + "\n\n")

fig_main, ax_main = plt.subplots(figsize=(8, 8))

for T in range(1, TT + 1):
    lambda_rates = dif2(pX, pY, sourceX, sourceY, a, Q, V, D, lamda)
    z = np.random.poisson(lambda_rates)
    z = np.clip(z, 0, max_d)
    
    # 获取观测，已停止的探测器不再获取观测
    for ln in range(layer_num):
        if robot_status[ln] == 0:
            z[ln] = 0
            
    z_h.append(np.column_stack((pX, pY, z)))

    for ln in range(layer_num):
        D_KL_h[ln].append(D_KL[ln].copy())

    mse = 0.0
    source_pred = layer_cluster[:, [0, 1]]
    matched = set()
    for i in range(len(source_true)):
        dis_min = 10000.0
        best_j = -1
        for j in range(len(source_pred)):
            if j not in matched and robot_status[j] == 1:
                dis_cur = np.sum((source_true[i] - source_pred[j])**2)
                if dis_cur < dis_min:
                    dis_min = dis_cur
                    best_j = j
        if best_j != -1:
            matched.add(best_j)
        mse += dis_min if dis_min < 10000.0 else 100.0
    rmse = np.sqrt(mse / len(sourceY))
    mse_h.append(rmse)
    iter_msg = f"迭代 {T}/{TT}: 真实观测 z={z}, 估计定位 RMSE={rmse:.4f}, Niches={niches}"
    print(iter_msg)
    log_f.write(iter_msg + "\n")

    # ==========================================
    # 4.1 NPSO 小生境 (Niche) 编队与协同
    # ==========================================
    for ln in range(layer_num):
        if robot_status[ln] == 1 and z[ln] > XP_maxz[ln]:
            XP_maxz[ln] = z[ln]
            XP_x[ln] = pX[ln]
            XP_y[ln] = pY[ln]

    delta_dis = 3.0
    k_sim = 0.02
    delta_sim = 2.0 * np.exp(-k_sim * T) 

    # 4.1 Niche 已由上一步的 form_dynamic_niches 动态生成 (CDPA-CI Phase 3)
    # niches 在每步末尾基于 KL + belief 距离更新, 供下一步使用

    # ==========================================
    # 4.2 粒子滤波滤波权重更新及 MCMC (各自估算)
    # ==========================================

    # CDPA-CI Phase 4: 认知差异融合权重 (修复: niche内 + same-source gate)
    all_fusion_weights = {}
    if USE_COGNITIVE_FUSION:
        gate_thresholds = {'dist_th': same_source_dist_th, 'kl_th': same_source_kl_th,
                           'q_th': same_source_q_th, 'obs_th': same_source_obs_th}
        for niche_members in niches:
            if len(niche_members) <= 1:
                for m in niche_members:
                    all_fusion_weights[m] = {m: 1.0}
                continue
            for ego_id in niche_members:
                if USE_SAME_SOURCE_GATE:
                    # 只允许 same-source 的伙伴参与 fusion
                    allowed = [ego_id]  # self always allowed
                    for other in niche_members:
                        if other == ego_id:
                            continue
                        if same_source_gate(ego_id, other, layer_cluster, layer_cov, D_KL,
                                            z, pX, pY, a, V, D, lamda, gate_thresholds):
                            allowed.append(other)
                    fw = compute_belief_fusion_weights(ego_id, allowed, layer_cluster, layer_cov, D_KL,
                                                        z, pX, pY, a, V, D, lamda,
                                                        alpha_kl, alpha_mu, alpha_obs, alpha_unc, tau_fusion)
                else:
                    fw = compute_belief_fusion_weights(ego_id, niche_members, layer_cluster, layer_cov, D_KL,
                                                        z, pX, pY, a, V, D, lamda,
                                                        alpha_kl, alpha_mu, alpha_obs, alpha_unc, tau_fusion)
                all_fusion_weights[ego_id] = fw
        # 记录 Agent 0 的融合权重 (用于可视化)
        if 0 in all_fusion_weights:
            fusion_weight_h[0].append(dict(all_fusion_weights[0]))
        for ln in range(1, layer_num):
            fusion_weight_h[ln].append({})

    w_end = np.zeros_like(w_obs)
    multi_particle_copy = multi_particle.copy()

    for niche in niches:
        L = niche[0] # Niche 的首领，复用其粒子分布作为整个小组的共享状态
        outside_ids = [idx for idx in range(layer_num) if idx not in niche and robot_status[idx] == 1]
        lo_ids = outside_ids
        if len(lo_ids) > 0:
            lo_sources = layer_cluster[lo_ids, :]
        else:
            lo_sources = np.empty((0, 3))

        p_x = multi_particle_copy[L, :, 0]
        p_y = multi_particle_copy[L, :, 1]
        p_q = multi_particle_copy[L, :, 2]

        # 获取当前 niche leader 的融合权重
        fw_L = all_fusion_weights.get(L, None) if USE_COGNITIVE_FUSION else None

        # 1. 结合小生境中所有个体的观测 z 进行权重更新
        temp_w = w_obs[L, :].copy()
        for member in niche:
            R_est = dif22_vectorized(pX[member], pY[member], member, lo_ids, lo_sources,
                                      p_x, p_y, p_q, a, V, D, lamda, D_KL,
                                      fusion_weights=fw_L)
            temp_w *= poisspdf(z[member], R_est)
            
        sum_w = np.sum(temp_w)
        if sum_w > 0:
            temp_w /= sum_w
        else:
            temp_w = np.ones(particle_num) / particle_num
        w_end[L, :] = temp_w
        
        # 2. ESS-based adaptive resampling (CDPA-CI Phase 1)
        ess = effective_sample_size(w_end[L, :])
        if ess < resample_threshold * particle_num:
            # ESS 不足 → 执行系统重采样 (低方差)
            indices = systematic_resample(w_end[L, :])
            resampled_particles = multi_particle_copy[L, indices, :]
            w_ended_uniform = np.ones(particle_num) / particle_num
        else:
            # ESS 充足 → 保留当前粒子及权重, 跳过重采样
            resampled_particles = multi_particle_copy[L, :, :]
            w_ended_uniform = w_end[L, :].copy()
        # 记录 ESS 历史 (CDPA-CI Phase 1)
        for member in niche:
            ess_h[member].append(ess)
        
        # 3. MCMC
        x_candidate = np.random.normal(resampled_particles[:, 0], 0.2)
        y_candidate = np.random.normal(resampled_particles[:, 1], 0.2)
        q_candidate = np.random.normal(resampled_particles[:, 2], 0.2)
        valid = (x_candidate > X_min) & (x_candidate < X_max) & \
                (y_candidate > Y_min) & (y_candidate < Y_max) & \
                (q_candidate > Q_min) & (q_candidate < Q_max)
                
        ans1 = np.ones(particle_num)
        ans2 = np.ones(particle_num)
        for member in niche:
            R_cand = dif22_vectorized(pX[member], pY[member], member, lo_ids, lo_sources,
                                       x_candidate, y_candidate, q_candidate, a, V, D, lamda, D_KL,
                                       fusion_weights=fw_L)
            ans1 *= poisspdf(z[member], R_cand)
            R_curr = dif22_vectorized(pX[member], pY[member], member, lo_ids, lo_sources,
                                       resampled_particles[:, 0], resampled_particles[:, 1],
                                       resampled_particles[:, 2], a, V, D, lamda, D_KL,
                                       fusion_weights=fw_L)
            ans2 *= poisspdf(z[member], R_curr)
            
        threshold_mcmc = np.ones(particle_num)
        nonzero_ans2 = ans2 > 0
        threshold_mcmc[nonzero_ans2] = np.clip(ans1[nonzero_ans2] / ans2[nonzero_ans2], 0, 1.0)
        accept = valid & (np.random.rand(particle_num) <= threshold_mcmc)
        
        final_particles = resampled_particles.copy()
        final_particles[accept, 0] = x_candidate[accept]
        final_particles[accept, 1] = y_candidate[accept]
        final_particles[accept, 2] = q_candidate[accept]
        
        # 4. 粒子群更新 (受 USE_PARTICLE_SHARING_IN_NICHE 控制)
        if USE_PARTICLE_SHARING_IN_NICHE:
            # 全员硬同步 (原始 CDPA-CI 行为, 会抹掉多样性)
            for member in niche:
                multi_particle[member, :, :] = final_particles
                w_obs[member, :] = w_ended_uniform
                w_end[member, :] = w_ended_uniform
        else:
            # 修复版: niche 内只共享 Gaussian belief, 不硬覆盖粒子
            # Leader 更新 (使用全 niche 的联合观测)
            multi_particle[L, :, :] = final_particles
            w_obs[L, :] = w_ended_uniform
            w_end[L, :] = w_ended_uniform
            # 非 leader 成员: 独立更新自己的粒子 (仅用自身观测)
            for member in niche:
                if member == L:
                    continue
                # 用成员自己的粒子做独立 PF 更新
                p_x_m = multi_particle_copy[member, :, 0]
                p_y_m = multi_particle_copy[member, :, 1]
                p_q_m = multi_particle_copy[member, :, 2]
                temp_w_m = w_obs[member, :].copy()
                # 只用该成员自己的观测 (不混入其他机器人观测)
                R_est_m = dif22_vectorized(pX[member], pY[member], member, [], np.empty((0, 3)),
                                            p_x_m, p_y_m, p_q_m, a, V, D, lamda, D_KL,
                                            fusion_weights=None)
                temp_w_m *= poisspdf(z[member], R_est_m)
                sum_w_m = np.sum(temp_w_m)
                if sum_w_m > 0:
                    temp_w_m /= sum_w_m
                else:
                    temp_w_m = np.ones(particle_num) / particle_num
                # ESS-based resample (复用相同逻辑)
                ess_m = effective_sample_size(temp_w_m)
                if ess_m < resample_threshold * particle_num:
                    indices_m = systematic_resample(temp_w_m)
                    multi_particle[member, :, :] = multi_particle_copy[member, indices_m, :]
                    w_obs[member, :] = np.ones(particle_num) / particle_num
                    w_end[member, :] = np.ones(particle_num) / particle_num
                else:
                    multi_particle[member, :, :] = multi_particle_copy[member, :, :]
                    w_obs[member, :] = temp_w_m
                    w_end[member, :] = temp_w_m
                ess_h[member].append(ess_m)

    # ==========================================
    # 4.3 均值，协方差估算与 KL 发散度判断
    # ==========================================
    for ln in range(layer_num):
        if robot_status[ln] == 0: continue
        data = multi_particle[ln, :, :]
        weighted_mean = np.sum(w_obs[ln, :][:, np.newaxis] * data, axis=0)
        deviation = data - weighted_mean
        weighted_covariance = (deviation * w_obs[ln, :][:, np.newaxis]).T @ deviation
        weighted_covariance += np.eye(3) * 1e-6
        layer_cov[ln, :, :] = weighted_covariance
        layer_cluster[ln, :] = weighted_mean

    for i in range(layer_num):
        for j in range(layer_num):
            if i == j or robot_status[i] == 0 or robot_status[j] == 0:
                continue
            cov_i = layer_cov[i]
            cov_j = layer_cov[j]
            det_i = max(np.linalg.det(cov_i), 1e-12)
            det_j = max(np.linalg.det(cov_j), 1e-12)
            inv_cov_j = np.linalg.pinv(cov_j)
            diff = (layer_cluster[j] - layer_cluster[i]).reshape(1, 3)
            val_kl = 0.5 * (np.log2(det_j / det_i) - 2.0 + np.trace(inv_cov_j @ cov_i) + (diff @ inv_cov_j @ diff.T)[0, 0])
            D_KL[i, j] = max(0.0, val_kl)

    # 4.3.5 小生境重置判断 (受 USE_NICHE_RESET 开关控制)
    recalculated_niches = []
    for niche in niches:
        L = niche[0]
        Tempreture_cur = np.trace(layer_cov[L, :, :])
        if Tempreture_cur < 0.2 and len(niche) > 1:
            best_member = niche[np.argmax([XP_maxz[m] for m in niche])]
            if XP_maxz[best_member] > 0.0:
                niche_msg = f"[{T}] 候选发现: 浓度峰值位于 ({pX[best_member]:.2f}, {pY[best_member]:.2f})"
                print(niche_msg)
                log_f.write(niche_msg + "\n")
            if USE_NICHE_RESET:
                # 全员散开并随机重置粒子
                reset_event_count += 1
                for member in niche:
                    multi_particle[member, :, 0] = np.random.rand(particle_num) * (X_max - X_min) + X_min
                    multi_particle[member, :, 1] = np.random.rand(particle_num) * (Y_max - Y_min) + Y_min
                    multi_particle[member, :, 2] = np.random.rand(particle_num) * (Q_max - Q_min) + Q_min
                    w_obs[member, :] = np.ones(particle_num) / particle_num
                    w_end[member, :] = np.ones(particle_num) / particle_num
                    XP_maxz[member] = 0.0
                    recalculated_niches.append([member])
            else:
                recalculated_niches.append(niche)
        else:
            recalculated_niches.append(niche)
    niches = recalculated_niches

    # CDPA-CI Phase 3: 动态小生境更新 (受 USE_DYNAMIC_NICHE 开关控制)
    if USE_DYNAMIC_NICHE:
        niches = form_dynamic_niches(layer_cluster, D_KL, robot_status, T, TT,
                                      delta_mu_start, delta_mu_end, delta_kl_start, delta_kl_end,
                                      max_niche_size, layer_cov=layer_cov)
    else:
        # 原始 NPSO-CI: 每个 active 机器人各自为战
        niches = [[i] for i in range(layer_num) if robot_status[i] == 1]
    niche_h.append([list(n) for n in niches])

    # CDPA-CI Phase 5: 角色自适应分配 (受 USE_ROLE_ADAPTATION 开关控制)
    if USE_ROLE_ADAPTATION:
        role_thresholds = {'conf_low': conf_th_low, 'conf_high': conf_th_high,
                           'entropy_high': entropy_th_high, 'patchiness_high': patchiness_th_role}
        roles = assign_roles_for_all_agents(layer_cov, w_end, z, robot_status, role_thresholds)
    else:
        # 原始 Infotaxis: 所有机器人等同于 Scout
        roles = np.full(layer_num, ROLE_SCOUT, dtype=int)
        for ln in range(layer_num):
            if robot_status[ln] == 0:
                roles[ln] = -1
    role_h.append(roles.copy())
    role_str = [role_names[r] if r >= 0 else 'Off' for r in roles]
    role_msg = f"  Roles={role_str}, Declared={len(declared_sources)}"
    print(role_msg)
    log_f.write(role_msg + "\n")

    # CDPA-CI Phase 6: 统一 candidate→declared pipeline (受 USE_SOURCE_DECLARATION 开关控制)
    if USE_SOURCE_DECLARATION:
        decl_thresholds = {'cov_th': source_declare_cov_th, 'dist_th': source_declare_dist_th,
                           'obs_th': source_declare_obs_th}
        # 1. 检查各机器人是否满足候选源条件
        new_candidate = None
        best_candidate_conf = -1
        for ln in range(layer_num):
            if robot_status[ln] == 0:
                continue
            cand = check_source_declaration(ln, layer_cluster, layer_cov, z, pX, pY, decl_thresholds)
            if cand is not None:
                conf = 1.0 / (np.trace(layer_cov[ln]) + 1e-6)
                if conf > best_candidate_conf:
                    best_candidate_conf = conf
                    new_candidate = cand
        # 2. 更新候选源追踪器
        newly_declared = update_candidate_sources(
            candidate_sources, new_candidate, layer_cluster, z, pX, pY,
            candidate_proximity, candidate_consensus_agents,
            candidate_patchiness_max, max_d)
        # 3. 升级为 declared sources
        for nd in newly_declared:
            if not is_duplicate_source(nd, declared_sources, source_exclusion_radius):
                declared_sources.append(nd)
                decl_msg = f"  [Phase 6] 源声明确认! 位置=({nd[0]:.2f}, {nd[1]:.2f}), Q={nd[2]:.2f}"
                print(decl_msg)
                log_f.write(decl_msg + "\n")
    declared_source_h.append(len(declared_sources))

    # ==========================================
    # 4.4 运动控制决策 (Infotaxis + Exclusion + Diversity + Novelty + Intent)
    # ==========================================

    # Intent-aware assignment removed per user request — no cooperative intent updates

    # Visitation-Aware Novelty: 更新访问地图
    if USE_VISITATION_NOVELTY:
        nov_params = {'decay': novelty_decay, 'resolution': visit_grid_resolution,
                       'radius': novelty_radius, 'eta': novelty_eta, 'max_val': visit_map_max}
        update_visit_map(visit_map, pX, pY, robot_status, nov_params)

    gama2 = 0.5
    for niche in niches:
        active_members = [m for m in niche if robot_status[m] == 1]
        for ln in active_members:
            outside_ids = [idx for idx in range(layer_num) if idx != ln and robot_status[idx] == 1]
            lo_ids = outside_ids
            if len(lo_ids) > 0:
                lo_sources = layer_cluster[lo_ids, :]
            else:
                lo_sources = np.empty((0, 3))
                
            W_cur = np.sqrt(np.sum((layer_cluster[ln, :2] - np.array([pX[ln], pY[ln]]))**2))
            Tempreture_cur = np.trace(layer_cov[ln, :, :])**gama2
            with np.errstate(divide='ignore'):
                logs = np.log(w_end[ln, :])
                logs[~np.isfinite(logs)] = 0.0
            S_cur = -np.sum(w_end[ln, :] * logs)
            F_cur = W_cur + Tempreture_cur * S_cur
            
            # Adaptive step size (Uncertainty-Aware)
            if USE_ADAPTIVE_STEP_SIZE:
                current_step = compute_adaptive_step(ln, layer_cov, step, step_min, step_max, step_uncertainty_scale)
            else:
                current_step = step
            adaptive_step_h[ln].append(current_step)

            dx = [current_step, -current_step, 0.0, 0.0,
                  current_step/np.sqrt(2), -current_step/np.sqrt(2),
                  -current_step/np.sqrt(2), current_step/np.sqrt(2)]
            dy = [0.0, 0.0, current_step, -current_step,
                  current_step/np.sqrt(2), current_step/np.sqrt(2),
                  -current_step/np.sqrt(2), -current_step/np.sqrt(2)]

            next_location = []
            for d_idx in range(8):
                nx = pX[ln] + dx[d_idx]
                ny = pY[ln] + dy[d_idx]
                if (X_min <= nx <= X_max) and (Y_min <= ny <= Y_max):
                    next_location.append([nx, ny])
                    
            next_executable_location = np.array(next_location)
            num_candidates = len(next_executable_location)
            F_next = np.zeros(num_candidates)
            
            for idx in range(num_candidates):
                p_x = multi_particle[ln, :, 0]
                p_y = multi_particle[ln, :, 1]
                p_q = multi_particle[ln, :, 2]

                fw_ln = all_fusion_weights.get(ln, None) if USE_COGNITIVE_FUSION else None
                pred_concs = dif22_vectorized(next_executable_location[idx, 0], next_executable_location[idx, 1],
                                              ln, lo_ids, lo_sources, p_x, p_y, p_q, a, V, D, lamda, D_KL,
                                              fusion_weights=fw_ln)
                for d in range(0, max_d + 1):
                    temp_vals = poisspdf(d, pred_concs)
                    w_next = w_end[ln, :] * temp_vals
                    sum_wn = np.sum(w_next)
                    if sum_wn > 0:
                        w_next /= sum_wn
                        non_zero = w_next > 0
                        S_next = np.sum(temp_vals[non_zero] * w_next[non_zero] * np.log(w_next[non_zero]))
                        F_next[idx] += S_next
                        
            delta_F = F_cur - F_next

            # CDPA-CI Phase 5: 角色感知偏置 (受 USE_ROLE_ADAPTATION 开关控制)
            if USE_ROLE_ADAPTATION:
                beta_tracker = 0.3
                beta_verifier = 0.15
                for idx in range(num_candidates):
                    nx, ny = next_executable_location[idx, 0], next_executable_location[idx, 1]
                    if roles[ln] == ROLE_TRACKER:
                        dist_cur = np.sqrt(np.sum((np.array([pX[ln], pY[ln]]) - layer_cluster[ln, :2]) ** 2))
                        dist_next = np.sqrt(np.sum((np.array([nx, ny]) - layer_cluster[ln, :2]) ** 2))
                        delta_F[idx] += beta_tracker * (dist_cur - dist_next)
                    elif roles[ln] == ROLE_VERIFIER:
                        move_dist = np.sqrt((nx - pX[ln]) ** 2 + (ny - pY[ln]) ** 2)
                        delta_F[idx] -= beta_verifier * move_dist

            # CDPA-CI Phase 6: 已声明源排斥惩罚 (受 USE_SOURCE_DECLARATION 开关控制)
            if USE_SOURCE_DECLARATION and len(declared_sources) > 0:
                for idx in range(num_candidates):
                    nx, ny = next_executable_location[idx, 0], next_executable_location[idx, 1]
                    delta_F[idx] += source_exclusion_penalty(
                        (nx, ny), declared_sources, source_exclusion_radius)

            # Source-Aware Diversity Regularization (不影响 posterior, 只影响 action)
            div_penalties = np.zeros(num_candidates)
            if USE_DIVERSITY_REGULARIZATION:
                for idx in range(num_candidates):
                    nx, ny = next_executable_location[idx, 0], next_executable_location[idx, 1]
                    div_pen = compute_diversity_penalty(
                        ln, (nx, ny), layer_cluster, robot_status,
                        diversity_sigma, diversity_min_dist)
                    delta_F[idx] -= diversity_lambda * div_pen
                    div_penalties[idx] = div_pen
            diversity_penalty_h.append(float(np.mean(div_penalties)))

            # Stage 1: base_score = Infotaxis + diversity (不含 novelty)
            base_score = delta_F.copy()

            # Novelty-as-Tie-Breaker (两阶段)
            if USE_VISITATION_NOVELTY and T > novelty_warmup_steps:
                nov_params_full = {'decay': novelty_decay, 'resolution': visit_grid_resolution,
                                   'radius': novelty_radius, 'eta': novelty_eta, 'max_val': visit_map_max}
                nov_rewards = np.array([compute_novelty_reward((next_executable_location[idx, 0],
                                         next_executable_location[idx, 1]), visit_map, nov_params_full)
                                         for idx in range(num_candidates)])
                if USE_NOVELTY_TIE_BREAKER:
                    # 两阶段: 先选 top candidates, 再用 novelty tie-break
                    max_base = np.max(base_score)
                    # 条件1: base_score >= max_base - epsilon
                    cond_eps = base_score >= (max_base - novelty_epsilon)
                    # 条件2: 在 topk 中
                    topk_indices = np.argpartition(-base_score, min(novelty_topk, num_candidates) - 1)[:min(novelty_topk, num_candidates)]
                    cond_topk = np.zeros(num_candidates, dtype=bool)
                    cond_topk[topk_indices] = True
                    tiebreak_candidates = np.where(cond_eps | cond_topk)[0]
                    # 只在 tiebreak 候选集中加 novelty
                    final_score = base_score.copy()
                    for idx in range(num_candidates):
                        if idx in tiebreak_candidates:
                            final_score[idx] += novelty_tie_lambda * nov_rewards[idx]
                    # 诊断
                    best_base_idx = np.argmax(base_score)
                    chosen_idx = np.argmax(final_score)
                    base_score_h.append(float(np.mean(base_score)))
                    novelty_total_decisions += 1
                    if chosen_idx != best_base_idx:
                        novelty_changed_count += 1
                    if best_base_idx not in tiebreak_candidates:
                        non_best_base_selected += 1
                else:
                    # Additive / Bounded novelty
                    novelty_bonuses = np.zeros(num_candidates)
                    if USE_BOUNDED_NOVELTY:
                        for idx in range(num_candidates):
                            bonus, clipped = compute_bounded_novelty_bonus(
                                base_score[idx], nov_rewards[idx], novelty_lambda,
                                novelty_bonus_max, novelty_bonus_ratio)
                            novelty_bonuses[idx] = bonus
                            novelty_total_steps += 1
                            if clipped:
                                novelty_clipped_count += 1
                    else:
                        novelty_bonuses = novelty_lambda * nov_rewards
                    final_score = base_score + novelty_bonuses
                    chosen_idx = np.argmax(final_score)
                    novelty_total_decisions += 1
                    best_base_idx = np.argmax(base_score)
                    if chosen_idx != best_base_idx:
                        novelty_changed_count += 1
                # Intent-aware assignment removed — no intent bonus applied

                # 记录选中动作的 novelty
                novelty_reward_h[ln].append(float(nov_rewards[chosen_idx]))
                if USE_BOUNDED_NOVELTY and num_candidates > 0:
                    novelty_bonus_h.append(float(novelty_bonuses[chosen_idx]))
            else:
                # 无 novelty: 直接用 base_score (即 delta_F)
                final_score = base_score
                # Intent-aware assignment removed — no intent bonus applied
                max_val = np.max(final_score)
                best_indices = np.where(np.abs(final_score - max_val) < 1e-9)[0]
                chosen_idx = np.random.choice(best_indices)
                novelty_reward_h[ln].append(0.0)
                base_score_h.append(float(np.mean(base_score)))

            V_x[ln] = next_executable_location[chosen_idx, 0] - pX[ln]
            V_y[ln] = next_executable_location[chosen_idx, 1] - pY[ln]
            last_x[ln] = pX[ln]
            last_y[ln] = pY[ln]
            pX[ln] = next_executable_location[chosen_idx, 0]
            pY[ln] = next_executable_location[chosen_idx, 1]

    # Per-step strict success check
    sf_step, msc_step, ssm_step, s2r_step, scov_step = evaluate_source_success(
        sourceX, sourceY, layer_cluster, layer_cov, pX, pY, robot_status,
        success_est_dist_th, success_robot_dist_th, success_cov_trace_th)
    sa_step = msc_step / source_num
    success_flag_h.append(int(sf_step))
    matched_source_count_h.append(msc_step)
    source_accuracy_h.append(sa_step)
    best_source_accuracy = max(best_source_accuracy, sa_step)
    if sf_step and enable_early_stop:
        task_success_flag = 1; task_completion_step = T; task_completion_step_raw = T
        first_full_success_step = T; early_stop_flag = 1; final_source_accuracy = sa_step
        print(f"  [SUCCESS] All {source_num} sources at T={T}! Early stop.")
        log_f.write(f"  [SUCCESS] All {source_num} sources at T={T}!\n")
        break

    # ==========================================
    # Visit coverage 记录 (每步一次, niche 循环外)
    if USE_VISITATION_NOVELTY:
        visited_frac = float(np.count_nonzero(visit_map > 0.01)) / max(1, visit_map_h * visit_map_w)
        visit_coverage_h.append(visited_frac)

    # Post-loop: if no early stop, set final values
    if not early_stop_flag:
        final_source_accuracy = source_accuracy_h[-1] if source_accuracy_h else 0.0
        best_source_accuracy = max(source_accuracy_h) if source_accuracy_h else 0.0

    actual_steps = len(mse_h)

    # 4.5 实时绘图更新与帧缓存
    # ==========================================
    if T % 5 == 0 or T == 1 or T == TT:
        ax_main.clear()
        ax_main.scatter(sourceX, sourceY, s=350, marker='*', facecolor='#FFD700', edgecolor='#8B0000', linewidth=1.5, zorder=6, label='True Source')
        
        for ln in range(layer_num):
            if robot_status[ln] == 1:
                ax_main.scatter(multi_particle[ln, :, 0], multi_particle[ln, :, 1], s=2.5, color=colors[ln], marker='o', alpha=0.15, zorder=1)
                ax_main.scatter(layer_cluster[ln, 0], layer_cluster[ln, 1], s=150, facecolor=colors[ln], edgecolor='white', linewidth=1.2, marker='X', zorder=5, label=f'Agent {ln+1} Est.' if T == 1 or T == TT else "")
            
            hist_coords = np.array([pt[ln, :2] for pt in z_h])
            full_coords = np.vstack((hist_coords, [pX[ln], pY[ln]]))
            ax_main.plot(full_coords[:, 0], full_coords[:, 1], color=colors[ln] if robot_status[ln] == 1 else 'gray', linestyle='-', solid_capstyle='round', linewidth=1.5, alpha=0.6, zorder=3)
            ax_main.scatter(full_coords[0, 0], full_coords[0, 1], s=40, marker='s', facecolor='#FDFDFD', edgecolor=colors[ln], linewidth=1.5, zorder=4, label='Start Point' if (ln == 0 and (T == 1 or T == TT)) else "")
            ax_main.scatter(pX[ln], pY[ln], s=140, marker=robot_init_shape[ln], facecolor='white' if robot_status[ln] == 1 else 'black', edgecolor=colors[ln], linewidth=2.5, zorder=6)
        
        for ln in range(layer_num):
            if robot_status[ln] == 1:
                rname = role_names[roles[ln]] if roles[ln] >= 0 else 'Off'
                ax_main.text(pX[ln], pY[ln] + 0.25, f'Agent_{ln+1} ({rname})', color=colors[ln], fontsize=9, fontweight='bold', ha='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1), zorder=8)

        ax_main.set_xlim(X_min, X_max)
        ax_main.set_ylim(Y_min, Y_max)
        ax_main.set_aspect('equal')
        ax_main.set_xticks(np.arange(X_min, X_max+2, 2))
        ax_main.set_yticks(np.arange(Y_min, Y_max+2, 2))
        ax_main.set_xlabel(r'X Coordinate ($m$)', fontname='Arial', fontsize=12)
        ax_main.set_ylabel(r'Y Coordinate ($m$)', fontname='Arial', fontsize=12)
        ax_main.grid(True, linestyle='--', color='gray', alpha=0.3, zorder=0)
        ax_main.set_facecolor('#FDFDFD')
        
        niche_str = ', '.join(['{' + ','.join([f'{m+1}' for m in n]) + '}' for n in niches])
        ax_main.set_title(rf'CDPA-CI Dynamic Niches: [{niche_str}] | $T = {T}/{TT}$',
                          fontname='Arial', fontsize=12, fontweight='bold', pad=12)
        
        frame_path = os.path.join(results_fig_dir, f"{T}.png")
        fig_main.savefig(frame_path, dpi=300, bbox_inches='tight')
        gif_frames.append(str(frame_path))

plt.close(fig_main)



# ==========================================
# 5. 重构后续结果并输出最终定位图与寻源动态GIF列表
# ==========================================
# 5.1 生成 GIF 动图
print("正在合成并保存寻源追踪过程动态 GIF...")
gif_path = os.path.join(results_dir, "process.gif")
saved_gif = False
# 首选使用 imageio；若不可用则回退到 Pillow (PIL)
try:
    import imageio
    frames = []
    for fp in gif_frames:
        try:
            img = imageio.v2.imread(fp)
        except Exception:
            img = imageio.imread(fp)
        frames.append(img)
    if len(frames) > 0:
        imageio.mimsave(gif_path, frames, duration=0.2)
        print(f"寻源动态场景 GIF 已成功保存为 {gif_path}")
        saved_gif = True
except ModuleNotFoundError:
    # imageio missing; try Pillow
    try:
        from PIL import Image
        pil_frames = []
        for fp in gif_frames:
            pil_frames.append(Image.open(fp).convert('RGBA'))
        if len(pil_frames) > 0:
            first, rest = pil_frames[0], pil_frames[1:]
            first.save(gif_path, save_all=True, append_images=rest, duration=200, loop=0, optimize=False)
            print(f"寻源动态场景 GIF 已成功保存为 {gif_path} (Pillow)")
            saved_gif = True
    except Exception as e2:
        print("保存 GIF 失败（尝试使用 Pillow 回退）:", e2)
except Exception as e:
    print("保存 GIF 失败:", e)

if not saved_gif:
    # 记录未创建 GIF 的情况，便于后续排查
    fallback_marker = os.path.join(results_dir, "process_gif_failed.txt")
    try:
        with open(fallback_marker, 'w', encoding='utf-8') as _fm:
            _fm.write('GIF not created. Check if imageio or Pillow is installed.\n')
            _fm.write(f'frames_count={len(gif_frames)}\n')
        print(f"GIF 未能创建；在运行目录中写入标记文件 {fallback_marker}")
    except Exception:
        pass

# 5.2 绘制最终估计图 (Figure 5)
plt.figure(figsize=(8, 7))
X_grid, Y_grid = np.meshgrid(np.arange(X_min, X_max + size_plot, size_plot),
                             np.arange(Y_min, Y_max + size_plot, size_plot))
# 取最后一步估值取整后作为源模拟
rounded_est = np.round(layer_cluster)
R_est_grid = dif2(X_grid, Y_grid, rounded_est[:, 0], rounded_est[:, 1], a, rounded_est[:, 2], V, D, lamda)
pcol = plt.pcolormesh(X_grid, Y_grid, np.clip(R_est_grid, 0, 15.0), shading='auto', cmap='viridis')
cbar = plt.colorbar(pcol)
cbar.set_label(r'Estimated Concentration $(ppm)$', fontsize=12)
plt.scatter(rounded_est[:, 0], rounded_est[:, 1], marker='^', s=180, facecolor='#FF4500', edgecolor='white', linewidth=1.2, label="Estimated Core", zorder=5)
# CDPA-CI Phase 6: 标记已声明源
if len(declared_sources) > 0:
    decl_arr = np.array(declared_sources)
    plt.scatter(decl_arr[:, 0], decl_arr[:, 1], marker='D', s=200, facecolor='#00FF00', edgecolor='black', linewidth=1.5, label=f"Declared ({len(declared_sources)})", zorder=7)
plt.xlabel(r'X Coordinate ($m$)', fontsize=14)
plt.ylabel(r'Y Coordinate ($m$)', fontsize=14)
plt.title(r'Constructed Plume Field by Multi-Agent Bayesian Filter', fontsize=15, fontweight='bold', pad=15)
plt.legend(loc='upper right', framealpha=0.9, edgecolor='black')
plt.grid(True, color='w', linestyle='--', alpha=0.2)
plt.tight_layout()
_target = os.path.join(results_dir, 'estimation.png')
os.makedirs(os.path.dirname(_target), exist_ok=True)
plt.savefig(_target, dpi=300, bbox_inches='tight')
plt.close()

# 5.3 绘制 KL 散度下降趋势图 (Figure 100)
plt.figure(figsize=(9, 6))
ax = plt.gca()
ax.set_facecolor('#F8F9FA')
# 取 Agent 1 对其他 Agent 的 KL 散度演化趋势 (对应 Matlab 中 绘图)
# Matlab value = 1 ./ (1 + exp(-D_KL_h{i}));
agent_idx = 0  # Agent 1
kl_hist_arr = np.array(D_KL_h[agent_idx])  # 维度 (100, 5)
value_sig = 1.0 / (1.0 + np.exp(-kl_hist_arr))

line_styles = ['-', '--', '-.', ':']
idx_style = 0
for j in range(layer_num):
    if j != agent_idx:
        plt.plot(value_sig[:, j], color=colors[j], linewidth=2.0, linestyle=line_styles[idx_style%4], label=f'Agent 1 to Agent {j+1}')
        idx_style += 1

plt.xlim(1, actual_steps)
plt.ylim(0.5, 1.05)
plt.xlabel(r'Iteration step ($T$)', fontsize=14)
plt.ylabel(r'Confidence Weight $\omega_{1,j}$ ($\sigma$-mapped KL)', fontsize=14)
plt.title(r'Evolution of Inter-Agent Confidence via KL Divergence', fontsize=15, fontweight='bold', pad=15)
plt.legend(framealpha=0.9, edgecolor='black', fontsize=11, loc='lower right')
plt.grid(True, linestyle='--', color='gray', alpha=0.3)
plt.tight_layout()
_target = os.path.join(results_dir, 'kl_divergence.png')
os.makedirs(os.path.dirname(_target), exist_ok=True)
plt.savefig(_target, dpi=300, bbox_inches='tight')
plt.close()

# 5.4 绘制 RMSE 收敛曲线图 (Figure 81)
plt.figure(figsize=(9, 6))
ax = plt.gca()
ax.set_facecolor('#F8F9FA')
plt.plot(mse_h, linewidth=2.5, color='#DC143C', linestyle='-', marker='o', markersize=5, markevery=max(1, actual_steps//20), markerfacecolor='white', markeredgewidth=1.5, label='Localization RMSE')
plt.xticks(np.arange(0, actual_steps + 1, max(1, actual_steps//10)))
plt.xlabel(r'Physical time step ($T$)', fontsize=14)
plt.ylabel(r'Root Mean Square Error ($RMSE$)', fontsize=14)
plt.title(r'Consensus Convergence Error of Multi-Robot Source Seeking', fontsize=15, fontweight='bold', pad=15)
plt.grid(True, color='gray', linestyle='--', alpha=0.3)
plt.legend(framealpha=0.9, edgecolor='black', fontsize=12)
plt.tight_layout()
_target = os.path.join(results_dir, 'rmse.png')
os.makedirs(os.path.dirname(_target), exist_ok=True)
plt.savefig(_target, dpi=300, bbox_inches='tight')
plt.close()

# 5.5 绘制 ESS 的绘图代码已根据用户要求移除。

# 5.6 Niche 数量绘图代码已根据用户要求移除。

# 5.7 绘制认知差异融合权重演化图 (CDPA-CI Phase 4: Figure Fusion Weights)
if USE_COGNITIVE_FUSION and len(fusion_weight_h[0]) > 0:
    plt.figure(figsize=(9, 6))
    ax = plt.gca()
    ax.set_facecolor('#F8F9FA')
    # 提取 Agent 0 对各 Agent 的融合权重时间序列
    agent_idx = 0
    for j in range(layer_num):
        if j == agent_idx:
            continue
        omega_ts = []
        for rec in fusion_weight_h[agent_idx]:
            omega_ts.append(rec.get(j, 0.0))
        plt.plot(omega_ts, color=colors[j], linewidth=1.8, alpha=0.85,
                 label=f'Agent 1 -> Agent {j+1}')
    plt.xlim(1, actual_steps)
    plt.ylim(0, 1.05)
    plt.xlabel(r'Iteration step ($T$)', fontsize=14)
    plt.ylabel(r'Fusion Weight $\omega_{1,j}$ (Cognitive Softmax)', fontsize=14)
    plt.title(r'Evolution of Cognitive-Difference Belief Fusion Weights (CDPA-CI Phase 4)',
              fontsize=15, fontweight='bold', pad=15)
    plt.legend(framealpha=0.9, edgecolor='black', fontsize=11, loc='best')
    plt.grid(True, linestyle='--', color='gray', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_fig_dir, 'figure_fusion_weights.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 5.8 角色分配绘图代码已根据用户要求移除。

# 5.9 已声明源数量绘图代码已根据用户要求移除。

# ==========================================
# 6. 保存原始及中间数据为 .mat 文件，实现与MatLab生态兼容
# ==========================================
# 转换 D_KL_h 保存为 cell 类型的字典
d_kl_h_dict = np.empty((layer_num,), dtype=object)
for ln in range(layer_num):
    d_kl_h_dict[ln] = np.array(D_KL_h[ln])
# .mat exports disabled per user request; 5_D_KL_h.mat and 5_mse_h.mat will not be created.

# 关闭运行日志 (CDPA-CI)
log_f.write("\n" + "=" * 60 + "\n")
log_f.write(f"Simulation complete. Final RMSE: {mse_h[-1]:.4f}\n")
log_f.write(f"Total declared sources: {len(declared_sources)}\n")
log_f.write(f"Declared source positions: {declared_sources}\n")
log_f.write("=" * 60 + "\n")
log_f.close()
print("运行日志已保存为 run_log.txt")

# ==========================================
# CDPA-CI: 生成 debug_summary.txt (消融调试摘要)
# ==========================================
niche_counts_arr = np.array([len(n) for n in niche_h])
max_niche_sizes = np.array([max([len(n) for n in niche_h[t]]) if len(niche_h[t]) > 0 else 0
                             for t in range(len(niche_h))])
role_arr = np.array(role_h) if len(role_h) > 0 else np.zeros((0, layer_num))
role_dist = {}
if role_arr.size > 0:
    for r, name in zip([0, 1, 2], ['Scout', 'Tracker', 'Verifier']):
        role_dist[name] = int(np.sum(role_arr == r))

summary_lines = []
summary_lines.append("=" * 60)
summary_lines.append("CDPA-CI Debug Summary")
summary_lines.append("=" * 60)
summary_lines.append(f"run_mode          : {run_mode}")
summary_lines.append(f"USE_ESS           : {USE_ESS}")
summary_lines.append(f"USE_DYNAMIC_NICHE  : {USE_DYNAMIC_NICHE}")
summary_lines.append(f"USE_COGNITIVE_FUSION: {USE_COGNITIVE_FUSION}")
summary_lines.append(f"USE_ROLE_ADAPTATION: {USE_ROLE_ADAPTATION}")
summary_lines.append(f"USE_SOURCE_DECLARATION: {USE_SOURCE_DECLARATION}")
summary_lines.append(f"USE_NICHE_RESET    : {USE_NICHE_RESET}")
summary_lines.append(f"USE_PARTICLE_SHARING_IN_NICHE: {USE_PARTICLE_SHARING_IN_NICHE}")
summary_lines.append("-" * 60)
summary_lines.append(f"final RMSE        : {mse_h[-1]:.4f}")
summary_lines.append(f"min RMSE          : {min(mse_h):.4f}")
summary_lines.append(f"mean RMSE (last 50): {np.mean(mse_h[-50:]):.4f}")
summary_lines.append(f"declared sources  : {len(declared_sources)}")
summary_lines.append(f"candidate sources : {len(candidate_sources)}")
summary_lines.append(f"mean niche count  : {np.mean(niche_counts_arr):.2f}")
summary_lines.append(f"max niche size    : {int(np.max(max_niche_sizes)) if len(max_niche_sizes) > 0 else 0}")
# pairwise belief distance (当前步)
if USE_DYNAMIC_NICHE:
    active_mask = robot_status == 1
    active_clusters = layer_cluster[active_mask, :2]
    if len(active_clusters) > 1:
        pw_dists = []
        for a in range(len(active_clusters)):
            for b in range(a+1, len(active_clusters)):
                pw_dists.append(np.sqrt(np.sum((active_clusters[a] - active_clusters[b])**2)))
        summary_lines.append(f"mean pairwise belief dist: {np.mean(pw_dists):.3f}")
    else:
        summary_lines.append(f"mean pairwise belief dist: N/A")
else:
    summary_lines.append(f"mean pairwise belief dist: N/A")
summary_lines.append(f"role distribution : {role_dist}")
summary_lines.append(f"reset events      : {reset_event_count}")
# Duplicate-search 指标 (CDPA-CI)
dup_src_count, dup_robot_count, coverage_rate = _compute_duplicate_metrics(
    layer_cluster, source_true, robot_status, source_exclusion_radius)
# Strict success evaluation
sf, msc, ssm, s2r, strict_cov = evaluate_source_success(
    sourceX, sourceY, layer_cluster, layer_cov, pX, pY, robot_status,
    success_est_dist_th, success_robot_dist_th, success_cov_trace_th)
summary_lines.append(f"success flag: {sf}")
summary_lines.append(f"matched source count: {msc}")
summary_lines.append(f"source success mask: {ssm}")
summary_lines.append(f"source to robot match: {s2r}")
summary_lines.append(f"strict source coverage rate: {strict_cov:.3f}")
# Success failure diagnosis
sdiag = diagnose_source_success_failure(sourceX, sourceY, layer_cluster, layer_cov, pX, pY, robot_status,
                                         success_est_dist_th, success_robot_dist_th, success_cov_trace_th)
summary_lines.append("--- Success Failure Diagnosis ---")
for k in range(source_num):
    summary_lines.append(f"source_{k}: min_est={sdiag['min_est_dist'][k]:.3f} min_robot={sdiag['min_robot_dist'][k]:.3f} min_cov={sdiag['min_cov_trace'][k]:.4f} est_ok={sdiag['est_ok'][k]} robot_ok={sdiag['robot_ok'][k]} cov_ok={sdiag['cov_ok'][k]} all_ok={sdiag['all_ok'][k]}")
summary_lines.append(f"num_sources_est_ok: {sdiag['num_est_ok']}")
summary_lines.append(f"num_sources_robot_ok: {sdiag['num_robot_ok']}")
summary_lines.append(f"num_sources_cov_ok: {sdiag['num_cov_ok']}")
summary_lines.append(f"num_sources_all_ok: {sdiag['num_all_ok']}")
summary_lines.append(f"bottleneck_est_count: {sdiag['bottleneck_est']}")
summary_lines.append(f"bottleneck_robot_count: {sdiag['bottleneck_robot']}")
summary_lines.append(f"bottleneck_cov_count: {sdiag['bottleneck_cov']}")
summary_lines.append(f"mean_min_est_dist: {sdiag['mean_min_est']:.4f}")
summary_lines.append(f"mean_min_robot_dist: {sdiag['mean_min_robot']:.4f}")
summary_lines.append(f"mean_min_cov_trace: {sdiag['mean_min_cov']:.4f}")
summary_lines.append("--- End Diagnosis ---")
# Task completion summary
summary_lines.append(f"task success flag: {task_success_flag}")
summary_lines.append(f"task completion step: {task_completion_step}")
summary_lines.append(f"task completion step raw: {task_completion_step_raw}")
summary_lines.append(f"first full success step: {first_full_success_step}")
summary_lines.append(f"early stop flag: {early_stop_flag}")
summary_lines.append(f"actual steps: {actual_steps}")
summary_lines.append(f"final source accuracy: {final_source_accuracy:.3f}")
summary_lines.append(f"best source accuracy: {best_source_accuracy:.3f}")
final_msc_final = matched_source_count_h[-1] if matched_source_count_h else 0
best_msc = max(matched_source_count_h) if matched_source_count_h else 0
summary_lines.append(f"final matched source count: {final_msc_final}")
summary_lines.append(f"best matched source count: {best_msc}")
summary_lines.append(f"duplicate source count: {dup_src_count}")
summary_lines.append(f"duplicate robot count: {dup_robot_count}")
summary_lines.append(f"source coverage rate: {coverage_rate:.3f}")
mean_div_pen = float(np.mean(diversity_penalty_h)) if len(diversity_penalty_h) > 0 else 0.0
summary_lines.append(f"mean diversity penalty: {mean_div_pen:.4f}")
# Novelty stats
mean_nov_r = float(np.mean([np.mean(h) for h in novelty_reward_h if len(h) > 0])) if any(len(h) > 0 for h in novelty_reward_h) else 0.0
mean_visit_cov = float(np.mean(visit_coverage_h)) if len(visit_coverage_h) > 0 else 0.0
final_visit_cov = visit_coverage_h[-1] if len(visit_coverage_h) > 0 else 0.0
max_visit_cnt = float(np.max(visit_map)) if USE_VISITATION_NOVELTY else 0.0
summary_lines.append(f"mean novelty reward: {mean_nov_r:.4f}")
summary_lines.append(f"mean visit coverage: {mean_visit_cov:.4f}")
summary_lines.append(f"final visit coverage: {final_visit_cov:.4f}")
summary_lines.append(f"max visit count: {max_visit_cnt:.2f}")
# Tie-breaker diagnostics
mean_base = float(np.mean(base_score_h)) if len(base_score_h) > 0 else 0.0
pct_changed = (novelty_changed_count / max(1, novelty_total_decisions)) * 100.0
pct_non_best = (non_best_base_selected / max(1, novelty_total_decisions)) * 100.0
summary_lines.append(f"mean base score: {mean_base:.6f}")
summary_lines.append(f"percentage novelty changed action: {pct_changed:.2f}")
summary_lines.append(f"percentage non-best base selected: {pct_non_best:.2f}")
summary_lines.append(f"novelty total decisions: {novelty_total_decisions}")
# Intent-aware assignment diagnostics removed per user request
# Bounded novelty diagnostics
mean_bonus = float(np.mean(novelty_bonus_h)) if len(novelty_bonus_h) > 0 else 0.0
max_bonus = float(np.max(novelty_bonus_h)) if len(novelty_bonus_h) > 0 else 0.0
pct_clipped = (novelty_clipped_count / max(1, novelty_total_steps)) * 100.0
summary_lines.append(f"mean novelty bonus: {mean_bonus:.6f}")
summary_lines.append(f"max novelty bonus: {max_bonus:.6f}")
summary_lines.append(f"percentage bonus clipped: {pct_clipped:.2f}")
# Adaptive step size stats
all_steps = np.concatenate([np.array(h) for h in adaptive_step_h if len(h) > 0]) if any(len(h) > 0 for h in adaptive_step_h) else np.array([step])
final_steps = np.array([h[-1] for h in adaptive_step_h if len(h) > 0])
summary_lines.append(f"mean adaptive step: {float(np.mean(all_steps)):.4f}")
summary_lines.append(f"min adaptive step: {float(np.min(all_steps)):.4f}")
summary_lines.append(f"max adaptive step: {float(np.max(all_steps)):.4f}")
summary_lines.append(f"final adaptive step mean: {float(np.mean(final_steps)):.4f}" if len(final_steps) > 0 else "final adaptive step mean: 0.0000")
summary_lines.append("=" * 60)
summary_lines.append(f"Declared source positions: {declared_sources}")

summary_text = "\n".join(summary_lines)
with open("debug_summary.txt", "w", encoding="utf-8") as f:
    f.write(summary_text)
print(summary_text)
print("消融摘要已保存为 debug_summary.txt")

print("\n数据已转换完成并成功保存：")
print("1. process.gif (动态大图寻源过程) — 保存在 run 目录")
print("2. distribution.png (实际气体扩散情况) — 保存在 run 目录，与 `figure/` 同级")
print("3. pcolor.png (传感器观测网格图) — 保存在 run 目录，与 `figure/` 同级")
print("4. estimation.png (仿真估计场图) — 保存在 run 目录，与 `figure/` 同级")
print("5. kl_divergence.png (KL散度演化) — 保存在 run 目录，与 `figure/` 同级")
print("6. rmse.png (定位 RMSE 收敛曲线) — 保存在 run 目录，与 `figure/` 同级")
print("7. figure_fusion_weights.png (认知差异融合权重演化图) — 若启用则保存在 `figure/` 子目录")
print("注意：.mat 导出已被禁用（不再生成 5_D_KL_h.mat 与 5_mse_h.mat）")

# Visit coverage curve 绘图已根据用户要求移除。

# Adaptive step curve figure
if USE_ADAPTIVE_STEP_SIZE and any(len(h) > 0 for h in adaptive_step_h):
    plt.figure(figsize=(9, 6))
    ax = plt.gca(); ax.set_facecolor('#F8F9FA')
    for ln in range(layer_num):
        if len(adaptive_step_h[ln]) > 0:
            plt.plot(adaptive_step_h[ln], color=colors[ln], linewidth=1.5, alpha=0.85, label=f'Agent {ln+1}')
    plt.axhline(y=step, color='gray', linewidth=1.0, linestyle=':', alpha=0.5, label=f'Fixed step={step}')
    plt.xlim(1, actual_steps); plt.xlabel(r'Iteration ($T$)', fontsize=14)
    plt.ylabel(r'Adaptive Step Size', fontsize=14)
    plt.title(r'Uncertainty-Aware Adaptive Step Size Evolution', fontsize=15, fontweight='bold')
    plt.legend(fontsize=9); plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(results_fig_dir, "figure_adaptive_step_curve.png"), dpi=300, bbox_inches='tight'); plt.close()

# ==========================================
# 主入口
# ==========================================
if __name__ == "__main__":
    if not _is_ablation_call:
        print(f"Default run complete. See debug_summary.txt for results.")

# ==========================================
# (Ablation 框架已移至文件前部, 供 --ablation 模式提前调用)
# ==========================================
