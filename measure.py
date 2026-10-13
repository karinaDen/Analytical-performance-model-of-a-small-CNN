"""Замеры на GPU: latency, пиковая память, энергия по сетке (S, B).
"""
import csv
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

import equations as eq
from models import build_model

RES = Path(__file__).resolve().parent / "results"
SEED = 2026

BASE_S = [32, 64, 128, 224, 256, 384, 512]
BASE_B = [1, 2, 4, 8, 16, 32, 64, 128, 256]

LAT_MIN_REPS, LAT_MIN_SEC, LAT_MAX_REPS = 10, 0.3, 200
ENERGY_MIN_SEC, ENERGY_MIN_REPS = 2.0, 3


def make_grid(seed=SEED):
    """База + случайные точки. Случайные S: 4 кратных 16 из [32, 512] вне базы;
    случайные B: 3 целых из [1, 256], не степени двойки. Конфигурация попадает
    в валидацию, если её S или B случайные: калибруем только на базовой сетке
    7 x 9 = 63 точки, проверяем на остальных 69 (их модель не видела)."""
    rng = np.random.default_rng(seed)
    s_pool = [s for s in range(32, 513, 16) if s not in BASE_S]
    b_pool = [b for b in range(1, 257) if b & (b - 1) != 0]
    rand_s = sorted(int(s) for s in rng.choice(s_pool, 4, replace=False))
    rand_b = sorted(int(b) for b in rng.choice(b_pool, 3, replace=False))
    grid = []
    for S in sorted(BASE_S + rand_s):
        for B in sorted(BASE_B + rand_b):
            grid.append((S, B, (S in rand_s) or (B in rand_b)))
    return grid, rand_s, rand_b


class Energy:
    """Счётчик энергии всей GPU через NVML (миллиджоули с момента загрузки драйвера)."""

    def __init__(self):
        self.h = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv = pynvml
            self.h = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h)
        except Exception as e:   
            print("NVML energy counter недоступен:", e)
            self.h = None

    def joules(self):
        return self.nv.nvmlDeviceGetTotalEnergyConsumption(self.h) / 1000.0

    def info(self):
        if self.h is None:
            return {}
        try:
            return {"temp_c": self.nv.nvmlDeviceGetTemperature(self.h, 0),
                    "sm_clock_mhz": self.nv.nvmlDeviceGetClockInfo(self.h, 1),
                    "power_w_now": self.nv.nvmlDeviceGetPowerUsage(self.h) / 1000.0}
        except Exception:
            return {}


def forward(model, x):
    with torch.inference_mode():
        return model(x)


def measure_runtime_bytes(model, dev):
    """Сколько байт allocator держит вне формулы: веса проверяем точно,
    а рабочую область cuBLAS (и всё, что библиотеки выделяют один раз и
    навсегда) измеряем как разницу до и после первого прохода."""
    torch.cuda.synchronize()
    w_alloc = torch.cuda.memory_allocated()
    x = torch.randn(1, 3, 32, 32, device=dev)
    out = forward(model, x)
    torch.cuda.synchronize()
    del out, x
    torch.cuda.synchronize()
    runtime = torch.cuda.memory_allocated() - w_alloc
    return {"weights_allocated": w_alloc, "weights_predicted": eq.WEIGHT_BYTES,
            "runtime_bytes": runtime, "runtime_bytes_default_formula": eq.RUNTIME_BYTES_DEFAULT}


def flops_check(model, dev):
    """Сверка flops() с независимым счётчиком PyTorch (FlopCounterMode, тоже 2 FLOPs на MAC)."""
    from torch.utils.flop_counter import FlopCounterMode
    rows = []
    for S, B in [(32, 1), (64, 3), (224, 8), (512, 2)]:
        x = torch.randn(B, 3, S, S, device=dev)
        fc = FlopCounterMode(display=False)
        with fc:
            forward(model, x)
        rows.append({"S": S, "B": B, "torch_flops": fc.get_total_flops(),
                     "formula_flops": float(eq.flops(S, B))})
        print("FLOPs check", rows[-1])
    return rows


def memory_audit(model, dev, S=224, B=32):
    """Проход слой за слоем: сколько памяти живёт во время каждого слоя.
    Повторяет цикл nn.Sequential, поэтому времена жизни те же, что в forward."""
    torch.cuda.empty_cache()
    x = torch.randn(B, 3, S, S, device=dev)
    torch.cuda.synchronize()
    static = torch.cuda.memory_allocated()          # веса + вход + рабочие области
    pred = eq.activation_timeline(S, B)
    names = {0: "conv1", 2: "maxpool", 3: "conv2", 5: "conv3", 7: "conv4", 9: "conv5",
             11: "conv6", 13: "gap", 15: "fc1", 17: "fc2"}
    rows = []
    with torch.inference_mode():
        h = x
        for i, m in enumerate(model):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            h = m(h)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() - static
            if i in names:
                rows.append({"S": S, "B": B, "layer": names[i], "measured_bytes": peak,
                             "predicted_bytes": float(pred[names[i]])})
                print("audit", rows[-1])
    del h, x
    return rows


def measure_one(model, dev, S, B, en):
    x = torch.randn(B, 3, S, S, device=dev)
    for _ in range(2):                              # прогрев: выбор алгоритмов cuDNN, кэш
        forward(model, x)
    torch.cuda.synchronize()

    # --- память: чистый allocator, пик за один проход
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    out = forward(model, x)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    req_peak = torch.cuda.memory_stats().get("requested_bytes.all.peak", float("nan"))
    del out

    # --- latency: CUDA events вокруг каждого прохода, медиана
    times = []
    t_start = time.perf_counter()
    while len(times) < LAT_MAX_REPS and (len(times) < LAT_MIN_REPS or time.perf_counter() - t_start < LAT_MIN_SEC):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        forward(model, x)
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b) / 1000.0)
    lat = float(np.median(times))

    # --- энергия: много проходов подряд, разность счётчика NVML / число проходов
    e_j = p_w = float("nan")
    n_e = 0
    if en.h is not None:
        n_e = max(ENERGY_MIN_REPS, math.ceil(ENERGY_MIN_SEC / max(lat, 1e-6)))
        torch.cuda.synchronize()
        e0, t0 = en.joules(), time.perf_counter()
        for _ in range(n_e):
            forward(model, x)
        torch.cuda.synchronize()
        e1, t1 = en.joules(), time.perf_counter()
        e_j = (e1 - e0) / n_e
        p_w = (e1 - e0) / (t1 - t0)
    del x
    return {"status": "ok", "latency_s": lat,
            "latency_p10_s": float(np.percentile(times, 10)), "latency_p90_s": float(np.percentile(times, 90)),
            "n_latency": len(times), "memory_bytes": peak, "memory_before_bytes": before,
            "memory_requested_peak_bytes": req_peak, "energy_j": e_j, "avg_power_w": p_w, "n_energy": n_e}


def main():
    assert torch.cuda.is_available(), "Нужна GPU (Colab: Runtime -> Change runtime type -> T4 GPU)"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(SEED)
    RES.mkdir(exist_ok=True)
    dev = torch.device("cuda")

    model = build_model().to(dev).eval()
    env = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
           "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
           "python": platform.python_version(),
           "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory}
    rt = measure_runtime_bytes(model, dev)
    env.update(rt)
    print(env)

    try:
        json.dump(flops_check(model, dev), open(RES / "flops_check.json", "w"), indent=1)
    except Exception as e:      # проверка необязательная, замеры от неё не зависят
        print("FlopCounterMode не сработал:", e)
    audit = memory_audit(model, dev)
    with open(RES / "memory_audit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(audit[0].keys()))
        w.writeheader()
        w.writerows(audit)

    grid, rand_s, rand_b = make_grid()
    env.update({"random_S": rand_s, "random_B": rand_b, "seed": SEED})
    json.dump(env, open(RES / "runtime.json", "w"), indent=1)
    print("случайные S:", rand_s, " случайные B:", rand_b, " конфигураций:", len(grid))

    en = Energy()
    order = np.random.default_rng(SEED).permutation(len(grid))   # перемешиваем, чтобы нагрев GPU не коррелировал с S и B
    rows = []
    fields = ["S", "B", "is_validation", "status", "latency_s", "latency_p10_s", "latency_p90_s", "n_latency",
              "memory_bytes", "memory_before_bytes", "memory_requested_peak_bytes", "energy_j", "avg_power_w",
              "n_energy", "temp_c", "sm_clock_mhz", "power_w_now"]
    t_all = time.perf_counter()
    for k, i in enumerate(order):
        S, B, val = grid[i]
        row = {"S": S, "B": B, "is_validation": int(val)}
        try:
            row.update(measure_one(model, dev, S, B, en))
        except torch.cuda.OutOfMemoryError:
            row.update({"status": "OOM"})
        torch.cuda.empty_cache()
        row.update(en.info())
        rows.append(row)
        print(f"[{k + 1:3d}/{len(grid)}] S={S:3d} B={B:3d} {row['status']}"
              + (f"  lat={row['latency_s'] * 1e3:8.3f} ms  mem={row['memory_bytes'] / 2**20:8.1f} MiB"
                 f"  E={row['energy_j']:.4f} J" if row["status"] == "ok" else ""), flush=True)
        with open(RES / "measurements.csv", "w", newline="") as f:      # сохраняем после каждой точки
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in sorted(rows, key=lambda r: (r["S"], r["B"])):
                w.writerow(r)
    print(f"готово за {time.perf_counter() - t_all:.0f} с")


if __name__ == "__main__":
    main()
