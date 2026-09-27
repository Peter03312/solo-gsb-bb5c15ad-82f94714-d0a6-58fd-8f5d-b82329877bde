"""等间隔重复读数的 A 类标准不确定度（Bartlett 窗自协方差估计）。

对 n 个等间隔重复读数 x_1..x_n（同一单位），先去均值 d_i = x_i − x̄，
按有偏估计（除以 n）计算滞后 k 的自协方差

    γ_k = (1/n) Σ_{i=1}^{n-k} d_i d_{i+k},   k = 0, …, L

再用 Bartlett（三角）权重 w_k = 1 − k/(L+1) 估计均值的方差

    u_A² = [γ_0 + 2 Σ_{k=1}^{L} w_k γ_k] / n

Bartlett 窗对应 Fejér 核，配合有偏自协方差估计保证括号内谱估计非负。
有效样本量

    n_eff = n γ_0 / [γ_0 + 2 Σ_{k=1}^{L} w_k γ_k]

分母为零（全部读数相同，γ_0 = 0，此时 u_A = 0）时 n_eff 无定义，返回 null。

计算直接在声明单位下进行（去均值后的统计量与仿射单位换算对易），结果
单位保持输入单位，u_A 数值可直接填入 /api/v1/propagate 的
standard_uncertainty（标量形式按输入量自身单位解释）。
"""
from __future__ import annotations

import math

from .errors import UnprocessableError
from .units import parse_unit, unit_str


def estimate_repeatability(req) -> dict:
    """估计等间隔重复读数均值的 A 类标准不确定度，返回可序列化字典。"""
    unit = parse_unit(req.unit, ["unit"])
    readings = [float(v) for v in req.readings]
    n = len(readings)
    max_lag = int(req.max_lag)
    if max_lag >= n:
        raise UnprocessableError(
            f"最大滞后 L={max_lag} 须满足 0 ≤ L < n（当前读数数量 n={n}）",
            ["max_lag"],
            code="invalid_max_lag",
        )

    mean = math.fsum(readings) / n
    deviations = [x - mean for x in readings]

    # 有偏自协方差 γ_k（除以 n），k = 0..L。
    gammas = [
        math.fsum(deviations[i] * deviations[i + k] for i in range(n - k)) / n
        for k in range(max_lag + 1)
    ]

    # Bartlett 加权和 S = γ_0 + 2 Σ_{k=1}^L (1 − k/(L+1)) γ_k。
    # 数学上 S ≥ 0（Fejér 核非负）；浮点舍入可能给出微小负值，钳制到 0。
    weighted = gammas[0] + 2.0 * math.fsum(
        (1.0 - k / (max_lag + 1)) * gammas[k] for k in range(1, max_lag + 1)
    )
    weighted = max(0.0, weighted)

    u_a = math.sqrt(weighted / n)
    # 分母 S 为零（读数全相同）时有效样本量无定义。
    n_eff = None if weighted == 0.0 else n * gammas[0] / weighted

    display = unit_str(unit)
    gamma_unit = unit_str(unit ** 2)
    return {
        "result": {
            "value": mean,
            "unit": display,
            "standard_uncertainty": u_a,
        },
        "number_of_readings": n,
        "max_lag": max_lag,
        "autocovariances": [
            {"lag": k, "value": gammas[k], "unit": gamma_unit}
            for k in range(max_lag + 1)
        ],
        "effective_sample_size": n_eff,
    }
