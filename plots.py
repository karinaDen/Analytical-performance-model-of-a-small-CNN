"""Графики: измерения (точки) и предсказания формул (линии/поверхности) вместе.

"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, ListedColormap

import equations as eq
from calibrate import load

RES = Path(__file__).resolve().parent / "results"
FIG = RES / "figures"


C_TRAIN, C_VAL, C_THIRD = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SEQ = LinearSegmentedColormap.from_list("seq", ["#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#eb6834"])
MS = 7

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 2,
})


def s_color(S, S_all):
    lo, hi = np.log(min(S_all)), np.log(max(S_all))
    return SEQ((np.log(S) - lo) / (hi - lo + 1e-12))


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name)
    plt.close(fig)
    print("saved", FIG / name)


def main():
    d = load()
    th = json.load(open(RES / "theta.json"))
    rt = json.load(open(RES / "runtime.json"))
    theta, theta_e = th["latency_theta"], th["energy_theta"]
    runtime = rt["runtime_bytes"]
    gpu = rt.get("gpu", "GPU")
    cap = rt.get("gpu_total_memory_bytes", np.nan)

    S, B = d["S"], d["B"]
    ok = d["status"] == "ok"
    oom = ~ok
    val = d["is_validation"] == 1
    tr = ok & ~val
    va = ok & val
    S_all = sorted(set(S.tolist()))
    base_S = [s for s in S_all if s not in rt.get("random_S", [])]
    Bc = np.unique(np.concatenate([np.geomspace(1, 256, 200), np.arange(1, 257)]))
    has_e = theta_e is not None and np.isfinite(d["energy_j"][ok]).all()

    p_lat = eq.latency(S, B, theta)
    p_mem = eq.memory(S, B, runtime_bytes=runtime)
    p_e = eq.energy(S, B, theta_e, theta) if has_e else np.full(len(S), np.nan)

    # ------------------------------------------------ 1. предсказание против измерения
    panels = [("Latency", d["latency_s"] * 1e3, p_lat * 1e3, "мс", "latency_train", "latency_validation"),
              ("Пиковая память", d["memory_bytes"] / 2**20, p_mem / 2**20, "МиБ", "memory_all", None)]
    if has_e:
        panels.append(("Энергия", d["energy_j"], p_e, "Дж", "energy_train", "energy_validation"))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.8))
    for ax, (title, meas, pred, unit, k_tr, k_va) in zip(np.atleast_1d(axes), panels):
        lo = np.nanmin(np.concatenate([meas[ok], pred[ok]])) * 0.7
        hi = np.nanmax(np.concatenate([meas[ok], pred[ok]])) * 1.4
        g = np.geomspace(lo, hi, 50)
        ax.fill_between(g, g * 0.9, g * 1.1, color=GRID, alpha=0.8, lw=0, label="±10%")
        ax.plot(g, g, color=MUTED, lw=1, label="идеал: предсказание = измерение")
        if k_va:
            ax.scatter(meas[tr], pred[tr], s=28, color=C_TRAIN, edgecolor="white", lw=0.6,
                       label=f"калибровка (базовая сетка), MAPE {th[k_tr]['mape_pct']:.1f}%")
            ax.scatter(meas[va], pred[va], s=28, color=C_VAL, edgecolor="white", lw=0.6, marker="D",
                       label=f"валидация (новые S или B), MAPE {th[k_va]['mape_pct']:.1f}%")
        else:
            ax.scatter(meas[ok], pred[ok], s=28, color=C_TRAIN, edgecolor="white", lw=0.6,
                       label=f"все точки, без калибровки, MAPE {th[k_tr]['mape_pct']:.2f}%")
        ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi), title=title,
               xlabel=f"измерено, {unit}", ylabel=f"предсказано, {unit}")
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(f"Предсказание формул против измерений на {gpu}", fontsize=12)
    save(fig, "01_pred_vs_measured.png")

    # ------------------------------------------------ 2. latency и throughput от B
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    for s in S_all:
        c = s_color(s, S_all)
        m = ok & (S == s)
        dash = "-" if s in base_S else "--"
        a1.plot(Bc, eq.latency(s, Bc, theta) * 1e3, dash, color=c, lw=1.6)
        a2.plot(Bc, Bc / eq.latency(s, Bc, theta), dash, color=c, lw=1.6)
        fc = c if s in base_S else "white"
        a1.scatter(B[m], d["latency_s"][m] * 1e3, s=22, facecolor=fc, edgecolor=c, zorder=3)
        a2.scatter(B[m], B[m] / d["latency_s"][m], s=22, facecolor=fc, edgecolor=c, zorder=3)
        if m.any():
            bl = B[m].max()
            a1.annotate(f"S={int(s)}", (bl, d["latency_s"][m & (B == bl)][0] * 1e3), xytext=(4, 0),
                        textcoords="offset points", fontsize=7, color=INK2, va="center")
    a1.axhline(theta["t0"] * 1e3, color=MUTED, lw=1, ls=":")
    a1.text(1.1, theta["t0"] * 1e3 * 1.15, f"t0 = {theta['t0'] * 1e3:.3f} мс (launch-bound пол)", fontsize=8, color=INK2)
    a1.set(xscale="log", yscale="log", xlabel="batch B", ylabel="latency, мс",
           title="Latency: точки — замеры, линии — формула")
    a2.set(xscale="log", yscale="log", xlabel="batch B", ylabel="картинок в секунду",
           title="Throughput: рост, пока GPU недогружена, потом плато")
    for a in (a1, a2):
        a.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256], ["1", "2", "4", "8", "16", "32", "64", "128", "256"])
    a1.plot([], [], "-", color=SEQ(0.6), label="S базовой сетки (калибровка)")
    a1.plot([], [], "--", color=SEQ(0.6), label="случайные S (валидация)")
    a1.legend(loc="upper left", fontsize=8)
    save(fig, "02_latency_throughput_vs_batch.png")

    # ------------------------------------------------ 3. достигнутая производительность
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    F = eq.flops(S, B)
    for s in S_all:
        c = s_color(s, S_all)
        m = ok & (S == s)
        ax.scatter(F[m], F[m] / d["latency_s"][m] / 1e12, s=24, color=c, edgecolor="white", lw=0.5, zorder=3)
        ax.plot(eq.flops(s, Bc), eq.flops(s, Bc) / eq.latency(s, Bc, theta) / 1e12, color=c, lw=1.2, alpha=0.8)
    ax.axhline(theta["peak_flops"] / 1e12, color=C_VAL, lw=1.2, ls="--")
    ax.text(F[ok].min(), theta["peak_flops"] / 1e12 * 1.05, f"P = {theta['peak_flops'] / 1e12:.2f} TFLOP/s (подобрано)", color=INK2, fontsize=8)
    sm = plt.cm.ScalarMappable(cmap=SEQ, norm=matplotlib.colors.LogNorm(min(S_all), max(S_all)))
    fig.colorbar(sm, ax=ax, label="размер картинки S, пикс.")
    ax.set(xscale="log", yscale="log", xlabel="FLOPs одного прохода", ylabel="достигнуто, TFLOP/s",
           title="Мало работы — GPU простаивает (launch-bound), много — упираемся в P и BW")
    save(fig, "03_achieved_flops.png")

    # ------------------------------------------------ 4. из чего складывается время
    cfgs = [(32, 1), (64, 8), (128, 32), (224, 64), (384, 128), (512, 128)]
    cfgs = [c for c in cfgs if (ok & (S == c[0]) & (B == c[1])).any()]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, (s, b) in enumerate(cfgs):
        parts = {"launch": theta["t0"], "memory": 0.0, "compute": 0.0}
        for r in eq.layer_table(s, b):
            tc, tm = float(r["flops"]) / theta["peak_flops"], float(r["bytes"]) / theta["bandwidth"]
            parts["compute" if tc >= tm else "memory"] += max(tc, tm)
        tot = sum(parts.values())
        bottom = 0
        for k, col, lab in [("launch", MUTED, "постоянная задержка t0 (launch)"),
                            ("memory", C_VAL, "ядра, ограниченные памятью (ReLU, pool, 1x1)"),
                            ("compute", C_TRAIN, "ядра, ограниченные вычислениями")]:
            ax.bar(i, 100 * parts[k] / tot, bottom=bottom, color=col, width=0.6, edgecolor="white", lw=2,
                   label=lab if i == 0 else None)
            bottom += 100 * parts[k] / tot
        meas = d["latency_s"][ok & (S == s) & (B == b)][0]
        ax.text(i, 102, f"{tot * 1e3:.2f} мс\n(замер {meas * 1e3:.2f})", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(range(len(cfgs)), [f"S={s}\nB={b}" for s, b in cfgs])
    ax.set(ylabel="% предсказанного времени", ylim=(0, 118),
           title="Смена режимов: launch-bound → memory-bound → compute-bound")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8)
    ax.grid(axis="x", visible=False)
    save(fig, "04_time_breakdown.png")

    # ------------------------------------------------ 5. карта режимов
    Sg = np.linspace(32, 512, 120)[:, None]
    Bg = np.geomspace(1, 256, 120)[None, :]
    reg = eq.regime(Sg, Bg, theta)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    cmap = ListedColormap(["#d6d5cf", "#f6c3ae", "#b7d3f6"])
    ax.pcolormesh(Bg.ravel(), Sg.ravel(), reg, cmap=cmap, vmin=-0.5, vmax=2.5, shading="auto")
    cs = ax.contour(Bg.ravel(), Sg.ravel(), eq.latency(Sg, Bg, theta) * 1e3, levels=[0.5, 1, 3, 10, 30, 100, 300],
                    colors=INK2, linewidths=0.7)
    ax.clabel(cs, fmt="%g мс", fontsize=7)
    ax.scatter(B[tr], S[tr], s=14, color=C_TRAIN, label="калибровка")
    ax.scatter(B[va], S[va], s=16, color=C_VAL, marker="D", label="валидация")
    if oom.any():
        ax.scatter(B[oom], S[oom], s=40, color="#d03b3b", marker="x", label="OOM")
    for y, lab, col in [(0, "launch-bound", "#d6d5cf"), (1, "memory-bound", "#f6c3ae"), (2, "compute-bound", "#b7d3f6")]:
        ax.scatter([], [], s=80, marker="s", color=col, label=lab)
    ax.set(xscale="log", xlabel="batch B", ylabel="размер картинки S, пикс.",
           title="Какая часть времени преобладает (по формуле), изолинии latency")
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256], ["1", "2", "4", "8", "16", "32", "64", "128", "256"])
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    save(fig, "05_regime_map.png")

    # ------------------------------------------------ 6. память
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    for s in S_all:
        c = s_color(s, S_all)
        m = ok & (S == s)
        a1.plot(Bc, eq.memory(s, Bc, runtime) / 2**20, color=c, lw=1.4)
        a1.scatter(B[m], d["memory_bytes"][m] / 2**20, s=22, color=c, edgecolor="white", lw=0.5, zorder=3)
    if np.isfinite(cap):
        a1.axhline(cap / 2**20, color="#d03b3b", lw=1.2, ls="--")
        a1.text(1.1, cap / 2**20 * 1.1, f"вся память GPU: {cap / 2**30:.1f} ГиБ", fontsize=8, color=INK2)
    if oom.any():
        a1.scatter(B[oom], p_mem[oom] / 2**20, marker="x", s=50, color="#d03b3b", zorder=4, label="OOM (точка — предсказание)")
        a1.legend(loc="lower right", fontsize=8)
    a1.set(xscale="log", yscale="log", xlabel="batch B", ylabel="пик памяти, МиБ",
           title="Пик памяти: точки — замер, линии — формула")
    resid = (d["memory_bytes"] - p_mem) / 2**20
    exact = ok & (np.abs(resid) <= 1.1)
    ws = ok & ~exact
    a2.axhline(0, color=MUTED, lw=1)
    a2.scatter(p_mem[exact] / 2**20, resid[exact], s=24, color=C_TRAIN, edgecolor="white", lw=0.5,
               label=f"формула точна (±1 МиБ): {exact.sum()} из {ok.sum()}")
    a2.scatter(p_mem[ws] / 2**20, resid[ws], s=24, color=C_VAL, edgecolor="white", lw=0.5, marker="D",
               label=f"cuDNN выбрал алгоритм conv2 с рабочей областью: {ws.sum()}")
    a2.legend(loc="upper left", fontsize=8)
    a2.set(xscale="log", xlabel="предсказанный пик, МиБ", ylabel="замер − формула, МиБ",
           title="Остаток = рабочая область cuDNN у conv2")
    save(fig, "06_memory.png")

    # ------------------------------------------------ 7. память по слоям
    try:
        import csv
        au = list(csv.DictReader(open(RES / "memory_audit.csv")))
        x = np.arange(len(au))
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.bar(x - 0.2, [float(r["predicted_bytes"]) / 2**20 for r in au], 0.38, color=C_TRAIN, label="формула")
        ax.bar(x + 0.2, [float(r["measured_bytes"]) / 2**20 for r in au], 0.38, color=C_VAL, label="замер")
        ax.set_xticks(x, [r["layer"] for r in au])
        ax.set(ylabel="живые активации во время слоя, МиБ",
               title=f"Память по слоям (S={au[0]['S']}, B={au[0]['B']}): тензоры совпадают, у conv2 сверху рабочая область cuDNN")
        ax.legend()
        ax.grid(axis="x", visible=False)
        save(fig, "07_memory_by_layer.png")
    except FileNotFoundError:
        pass

    # ------------------------------------------------ 8. энергия
    if has_e:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
        for s in S_all:
            c = s_color(s, S_all)
            m = ok & (S == s)
            fc = c if s in base_S else "white"
            a1.plot(Bc, eq.energy(s, Bc, theta_e, theta) / Bc * 1e3, color=c, lw=1.4)
            a1.scatter(B[m], d["energy_j"][m] / B[m] * 1e3, s=22, facecolor=fc, edgecolor=c, zorder=3)
            a2.scatter(B[m], d["avg_power_w"][m], s=22, facecolor=fc, edgecolor=c, zorder=3)
        a1.set(xscale="log", yscale="log", xlabel="batch B", ylabel="энергия на картинку, мДж",
               title="Энергия на картинку: точки — замер, линии — формула")
        a2.axhline(theta_e["p_static"], color=MUTED, ls=":", lw=1)
        a2.text(1.1, theta_e["p_static"] * 1.02, f"p_static = {theta_e['p_static']:.1f} Вт", fontsize=8, color=INK2)
        a2.set(xscale="log", xlabel="batch B", ylabel="средняя мощность GPU, Вт",
               title="Средняя мощность во время замера энергии")
        for a in (a1, a2):
            a.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256], ["1", "2", "4", "8", "16", "32", "64", "128", "256"])
        save(fig, "08_energy.png")

    # ------------------------------------------------ 9. ошибки по сетке
    Bs = sorted(set(B.tolist()))
    maps = [("latency", d["latency_s"], p_lat)] + ([("энергия", d["energy_j"], p_e)] if has_e else [])
    fig, axes = plt.subplots(1, len(maps), figsize=(7.2 * len(maps), 5.2))
    for ax, (name, meas, pred) in zip(np.atleast_1d(axes), maps):
        Z = np.full((len(S_all), len(Bs)), np.nan)
        for i in np.nonzero(ok)[0]:
            Z[S_all.index(S[i]), Bs.index(B[i])] = 100 * (pred[i] / meas[i] - 1)
        lim = max(10.0, np.nanpercentile(np.abs(Z), 95))
        im = ax.imshow(Z, cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto", origin="lower")
        for i in range(len(S_all)):
            for j in range(len(Bs)):
                if np.isfinite(Z[i, j]):
                    ax.text(j, i, f"{Z[i, j]:+.0f}", ha="center", va="center", fontsize=6.5, color=INK)
                elif any(oom & (S == S_all[i]) & (B == Bs[j])):
                    ax.text(j, i, "OOM", ha="center", va="center", fontsize=6, color="#d03b3b")
        rs, rb = set(rt.get("random_S", [])), set(rt.get("random_B", []))
        ax.set_xticks(range(len(Bs)), [f"{int(b)}*" if b in rb else str(int(b)) for b in Bs], fontsize=8)
        ax.set_yticks(range(len(S_all)), [f"{int(s)}*" if s in rs else str(int(s)) for s in S_all], fontsize=8)
        ax.set(xlabel="batch B (* — случайные, валидация)", ylabel="S (* — случайные, валидация)",
               title=f"Ошибка формулы {name}, % (предсказание / замер − 1)")
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="%")
    save(fig, "09_error_heatmaps.png")

    # ------------------------------------------------ 10. частота и мощность: троттлинг
    if "sm_clock_mhz" in d and np.isfinite(d["sm_clock_mhz"][ok]).any():
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
        Fo = eq.flops(S, B)
        cols = [s_color(s, S_all) for s in S[ok]]
        a1.scatter(Fo[ok], d["sm_clock_mhz"][ok], s=22, c=cols, edgecolor="white", lw=0.5)
        a1.set(xscale="log", xlabel="FLOPs одного прохода", ylabel="частота SM, МГц",
               title="Под нагрузкой T4 сбрасывает частоту (лимит мощности 70 Вт)")
        a2.scatter(Fo[ok], d["avg_power_w"][ok], s=22, c=cols, edgecolor="white", lw=0.5)
        a2.axhline(70, color="#d03b3b", lw=1, ls="--")
        a2.text(Fo[ok].min(), 70.8, "TDP 70 Вт", fontsize=8, color=INK2)
        a2.set(xscale="log", xlabel="FLOPs одного прохода", ylabel="средняя мощность, Вт",
               title="Мощность упирается в TDP")
        save(fig, "10_clock_power.png")


if __name__ == "__main__":
    main()
