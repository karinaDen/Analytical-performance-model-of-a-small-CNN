"""Калибровка theta для latency и energy. Только numpy.

Калибруем ТОЛЬКО на базовой сетке (is_validation = 0), качество считаем
отдельно на обучающих и на валидационных точках (случайные S или B,
которые модель не видела). Память не калибруется: она считается точно.

"""
import csv
import json
from pathlib import Path

import numpy as np

import equations as eq

RES = Path(__file__).resolve().parent / "results"


def load(path=RES / "measurements.csv"):
    rows = list(csv.DictReader(open(path)))
    out = {}
    for k in rows[0]:
        col = []
        for r in rows:
            v = r[k]
            try:
                col.append(float(v))
            except ValueError:
                col.append(np.nan if v == "" else v)
        is_text = any(isinstance(v, str) for v in col)
        out[k] = np.array(col, dtype=object if is_text else float)
    return out


# ------------------------------------------------------------------ latency
# Паспортные пределы GPU (FP32 FLOP/s, байт/с DRAM). Нужны как верхние границы для P и BW.
GPU_SPECS = {"T4": (8.14e12, 320e9), "P100": (9.3e12, 732e9), "V100": (15.7e12, 900e9),
             "A100": (19.5e12, 1555e9), "L4": (30.3e12, 300e9), "A10": (31.2e12, 600e9)}


def gpu_bounds(name):
    for k, v in GPU_SPECS.items():
        if k in name:
            return v
    return (np.inf, np.inf)


def fit_latency(S, B, y, p_max=np.inf, bw_max=np.inf):
    """Минимизируем сумму квадратов log(pred) - log(meas), то есть относительную ошибку,
    чтобы маленькие (0,1 мс) и большие (300 мс) точки весили одинаково.
    Параметры ищем в логарифмах: сначала сетка по (P, BW) с оптимальным t0, потом Нелдер-Мид.

    P и BW ограничены сверху паспортом GPU. Без этого задача вырождена: все слои
    растут как B * S^2, поэтому по (S, B) нельзя отличить «медленные вычисления» от
    «медленной памяти». На T4 без границ находится второй минимум с той же ошибкой,
    но с P = 13 TFLOP/s, что больше пика T4 (8,1 TFLOP/s), то есть физически невозможно."""
    ly = np.log(y)
    rows = eq.layer_table(S, B)
    F = np.stack([np.broadcast_to(r["flops"], y.shape) for r in rows])
    M = np.stack([np.broadcast_to(r["bytes"], y.shape) for r in rows])

    def clip(p):
        t0, P, BW = np.exp(p)
        return t0, min(P, p_max), min(BW, bw_max)

    def loss(p):
        t0, P, BW = clip(p)
        pred = t0 + np.maximum(F / P, M / BW).sum(0)
        return float(np.mean((np.log(pred) - ly) ** 2))

    best = None
    for lP in np.linspace(np.log(1e11), np.log(min(5e13, p_max)), 40):
        for lBW in np.linspace(np.log(1e10), np.log(min(2e12, bw_max)), 40):
            for lt0 in np.linspace(np.log(1e-6), np.log(1e-1), 30):
                v = loss((lt0, lP, lBW))
                if best is None or v < best[0]:
                    best = (v, np.array([lt0, lP, lBW]))
    p = nelder_mead(loss, best[1])
    t0, P, BW = clip(p)
    return {"t0": float(t0), "peak_flops": float(P), "bandwidth": float(BW)}


def nelder_mead(f, x0, step=0.3, iters=2000, tol=1e-12):
    n = len(x0)
    pts = [np.array(x0, float)] + [np.array(x0, float) + step * np.eye(n)[i] for i in range(n)]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = np.argsort(vals)
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        c = np.mean(pts[:-1], axis=0)
        xr = c + (c - pts[-1]); fr = f(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1]); fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c); fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0]] + [pts[0] + 0.5 * (p - pts[0]) for p in pts[1:]]
                vals = [vals[0]] + [f(p) for p in pts[1:]]
    return pts[int(np.argmin(vals))]


# ------------------------------------------------------------------- energy
def fit_energy(S, B, y, theta):
    """E = p_static * Latency + e_flop * FLOPs + e_byte * Bytes, все коэффициенты >= 0.
    Линейная задача: взвешенные МНК по относительной ошибке, неотрицательность —
    перебором подмножеств (параметров всего три)."""
    X = np.stack([eq.latency(S, B, theta), eq.flops(S, B), eq.bytes_moved(S, B)], axis=1)
    Xw, yw = X / y[:, None], np.ones_like(y)
    scale = np.abs(Xw).max(0)
    best = None
    for mask in range(1, 8):
        idx = [j for j in range(3) if mask >> j & 1]
        c, *_ = np.linalg.lstsq(Xw[:, idx] / scale[idx], yw, rcond=None)
        c = c / scale[idx]
        if np.any(c < 0):
            continue
        coef = np.zeros(3)
        coef[idx] = c
        err = float(np.mean((Xw @ coef - 1) ** 2))
        if best is None or err < best[0]:
            best = (err, coef)
    p, ef, eb = best[1]
    return {"p_static": float(p), "e_flop": float(ef), "e_byte": float(eb)}


def memory_breakdown(S, B, meas, pred, runtime_bytes):
    """Где формула памяти точна и откуда остаток.
    Точная точка: |замер - формула| <= 1,1 МиБ (округление allocator-а и 1 МиБ рабочей области conv1).
    В остальных пик переезжает с maxpool на conv2 (5x5): живы a2 + a3 и рабочая область cuDNN.
    Из замера получаем её размер в байтах на B * S^2."""
    diff = meas - pred
    exact = np.abs(diff) <= 1.1 * 2**20
    static = eq.WEIGHT_BYTES + runtime_bytes + eq.alloc(eq.FP32 * B * 3 * S ** 2)
    a = eq.act_elems(S)
    conv2_live = eq.alloc(eq.FP32 * B * a["a2"]) + eq.alloc(eq.FP32 * B * a["a3"])
    ws = (meas - static - conv2_live)[~exact] / (B[~exact] * S[~exact] ** 2)
    return {"n_exact": int(exact.sum()), "n_total": int(len(meas)),
            "n_with_cudnn_workspace": int((~exact).sum()),
            "conv2_workspace_bytes_per_BS2_min": float(ws.min()) if len(ws) else None,
            "conv2_workspace_bytes_per_BS2_median": float(np.median(ws)) if len(ws) else None,
            "conv2_workspace_bytes_per_BS2_max": float(ws.max()) if len(ws) else None}


def metrics(pred, meas):
    rel = pred / meas - 1
    return {"mape_pct": float(100 * np.mean(np.abs(rel))), "max_abs_err_pct": float(100 * np.max(np.abs(rel))),
            "median_abs_err_pct": float(100 * np.median(np.abs(rel))), "n": int(len(rel))}


def main():
    d = load()
    ok = d["status"] == "ok"
    tr = ok & (d["is_validation"] == 0)
    va = ok & (d["is_validation"] == 1)
    S, B = d["S"], d["B"]
    runtime = json.load(open(RES / "runtime.json"))

    p_max, bw_max = gpu_bounds(runtime.get("gpu", ""))
    theta = fit_latency(S[tr], B[tr], d["latency_s"][tr], p_max, bw_max)
    has_e = np.isfinite(d["energy_j"][ok]).all()
    theta_e = fit_energy(S[tr], B[tr], d["energy_j"][tr], theta) if has_e else None

    pred_lat = eq.latency(S, B, theta)
    pred_mem = eq.memory(S, B, runtime_bytes=runtime["runtime_bytes"])
    pred_e = eq.energy(S, B, theta_e, theta) if has_e else np.full(len(S), np.nan)

    out = {
        "latency_theta": theta,
        "energy_theta": theta_e,
        "memory": {"runtime_bytes_measured": runtime["runtime_bytes"],
                   "note": "память не калибруется; runtime_bytes измерен напрямую (рабочая область cuBLAS)"},
        "split": {"train": "базовая сетка 7 x 9", "validation": "конфигурации со случайными S или B",
                  "n_train": int(tr.sum()), "n_validation": int(va.sum()), "n_oom": int((~ok).sum())},
        "latency_train": metrics(pred_lat[tr], d["latency_s"][tr]),
        "latency_validation": metrics(pred_lat[va], d["latency_s"][va]),
        "memory_all": metrics(pred_mem[ok], d["memory_bytes"][ok]),
        "memory_max_abs_err_mib": float(np.max(np.abs(pred_mem[ok] - d["memory_bytes"][ok])) / 2**20),
        "memory_exact": memory_breakdown(S[ok], B[ok], d["memory_bytes"][ok], pred_mem[ok], runtime["runtime_bytes"]),
        "latency_bounds": {"gpu": runtime.get("gpu", ""), "p_max_flops": p_max, "bw_max_bytes_per_s": bw_max},
    }
    if has_e:
        out["energy_train"] = metrics(pred_e[tr], d["energy_j"][tr])
        out["energy_validation"] = metrics(pred_e[va], d["energy_j"][va])
    json.dump(out, open(RES / "theta.json", "w"), indent=1, ensure_ascii=False)

    with open(RES / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["S", "B", "is_validation", "status", "latency_s", "pred_latency_s", "memory_bytes",
                    "pred_memory_bytes", "energy_j", "pred_energy_j", "flops", "bytes_moved", "regime"])
        reg = eq.regime(S, B, theta)
        for i in range(len(S)):
            w.writerow([int(S[i]), int(B[i]), int(d["is_validation"][i]), d["status"][i], d["latency_s"][i],
                        pred_lat[i], d["memory_bytes"][i], pred_mem[i], d["energy_j"][i], pred_e[i],
                        eq.flops(S[i], B[i]), eq.bytes_moved(S[i], B[i]), ["launch", "memory", "compute"][int(reg[i])]])
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
