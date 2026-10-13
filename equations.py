"""Аналитическая модель стоимости сети из models.py.

Все функции принимают числа или numpy-массивы S (image_size) и B (batch)
и работают через broadcasting, поэтому поверхность над плоскостью (S, B)
строится одним вызовом.

Обозначения: S кратно 16, все тензоры FP32 (4 байта).
Размеры карт признаков: после conv1 S/2, после maxpool S/4, дальше S/4, S/8, S/8, S/16, S/16.
"""
import numpy as np

FP32 = 4
INT64 = 8
NUM_CLASSES = 100

# --------------------------------------------------------------- параметры
# (имя, число параметров)
PARAMS = [
    ("conv1.w", 32 * 3 * 7 * 7),        # 4 704
    ("conv2.w", 64 * 32 * 5 * 5),       # 51 200
    ("conv3.w", 128 * 64 * 3 * 3),      # 73 728
    ("conv4.w", 256 * 128 * 1 * 1),     # 32 768
    ("conv5.w", 256 * 256 * 3 * 3),     # 589 824
    ("conv6.w", 512 * 256 * 1 * 1),     # 131 072
    ("fc1.w", 256 * 512), ("fc1.b", 256),
    ("fc2.w", NUM_CLASSES * 256), ("fc2.b", NUM_CLASSES),
]
N_PARAMS = sum(n for _, n in PARAMS)    # 1 040 324


def alloc(nbytes):
    """Размер блока, который выдаёт caching allocator PyTorch:
    запрос округляется вверх до кратного 512 байт (пустой тензор не занимает памяти)."""
    nbytes = np.asarray(nbytes, dtype=np.float64)
    return np.where(nbytes > 0, 512.0 * np.ceil(nbytes / 512.0), 0.0)


WEIGHT_BYTES = float(sum(alloc(n * FP32) for _, n in PARAMS))

# Постоянная добавка библиотек: рабочая область cuBLAS, которую PyTorch
# выделяет через тот же allocator при первом матричном умножении и больше
# не освобождает. Не зависит от S и B. Значение по умолчанию — дефолт PyTorch
# для GPU до Hopper (CUBLAS_WORKSPACE_CONFIG=:4096:2:16:8):
# 2 * 4096 KiB + 8 * 16 KiB. measure.py измеряет её напрямую (не подбирает
# по целевой метрике) и сохраняет в results/theta.json.
RUNTIME_BYTES_DEFAULT = 2 * 4096 * 1024 + 8 * 16 * 1024


# ------------------------------------------------- размеры активаций (на 1 картинку)
def act_elems(S):
    """Число элементов выходного тензора каждого слоя для одной картинки."""
    S = np.asarray(S, dtype=np.float64)
    return {
        "x":   3 * S ** 2,                 # вход
        "a1": 32 * (S / 2) ** 2,           # conv1        = 8 S^2
        "a2": 32 * (S / 4) ** 2,           # maxpool      = 2 S^2
        "idx": 32 * (S / 4) ** 2,          # индексы maxpool (int64) = 2 S^2
        "a3": 64 * (S / 4) ** 2,           # conv2        = 4 S^2
        "a4": 128 * (S / 8) ** 2,          # conv3        = 2 S^2
        "a5": 256 * (S / 8) ** 2,          # conv4        = 4 S^2
        "a6": 256 * (S / 16) ** 2,         # conv5        = 1 S^2
        "a7": 512 * (S / 16) ** 2,         # conv6        = 2 S^2
        "g": 512 + 0 * S,                  # global avg pool
        "f1": 256 + 0 * S,                 # fc1
        "f2": NUM_CLASSES + 0 * S,         # fc2
    }


# --------------------------------------------------------------------- FLOPs
def layer_table(S, B):
    """FLOPs и минимальный трафик DRAM (байты) каждого ядра (kernel) прямого прохода.

    FLOPs: 1 MAC = 2 FLOPs. Для свёртки 2 * B * Cout * Hout * Wout * Cin * k^2.
    Трафик: каждое ядро один раз читает свои входы (включая веса) и один раз
    пишет выход. ReLU inplace — отдельное ядро: читает и пишет тензор.
    Поэлементные операции (ReLU, сравнения в maxpool, сложения в GAP, bias)
    дают ~0,2% FLOPs; они учтены отдельно в поле aux_flops.
    """
    S = np.asarray(S, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    e = act_elems(S)
    rows = []

    def add(name, flops, read_elems, write_bytes, weight_params=0, aux=0.0):
        rows.append({
            "name": name,
            "flops": flops,
            "bytes": FP32 * B * read_elems + write_bytes + FP32 * weight_params,
            "aux_flops": aux,
        })

    def conv(name, cin_key, out_key, cin, cout, k, so):
        mac = B * cout * so ** 2 * cin * k * k
        add(name, 2 * mac, e[cin_key], FP32 * B * e[out_key], cout * cin * k * k)
        add(name + ".relu", 0 * S, e[out_key], FP32 * B * e[out_key], aux=B * e[out_key])

    conv("conv1", "x", "a1", 3, 32, 7, S / 2)
    add("maxpool", 0 * S, e["a1"], FP32 * B * e["a2"] + INT64 * B * e["idx"], aux=8 * B * e["a2"])
    conv("conv2", "a2", "a3", 32, 64, 5, S / 4)
    conv("conv3", "a3", "a4", 64, 128, 3, S / 8)
    conv("conv4", "a4", "a5", 128, 256, 1, S / 8)
    conv("conv5", "a5", "a6", 256, 256, 3, S / 16)
    conv("conv6", "a6", "a7", 256, 512, 1, S / 16)
    add("gap", 0 * S, e["a7"], FP32 * B * e["g"], aux=B * e["a7"])
    add("fc1", 2 * B * 512 * 256 + 0 * S, e["g"], FP32 * B * e["f1"], 512 * 256 + 256, aux=B * 256)
    add("fc1.relu", 0 * S, e["f1"], FP32 * B * e["f1"], aux=B * 256)
    add("fc2", 2 * B * 256 * NUM_CLASSES + 0 * S, e["f1"], FP32 * B * e["f2"], 256 * NUM_CLASSES + NUM_CLASSES, aux=B * NUM_CLASSES)
    return rows


N_KERNELS = 17   # 6 conv + 6 relu + maxpool + gap + fc1 + relu + fc2


def flops(image_size, batch):
    """FLOPs одного прямого прохода (свёртки и линейные слои, 1 MAC = 2 FLOPs).

    Закрытая форма: FLOPs = 17 712 * B * S^2 + 313 344 * B.
    """
    return sum(r["flops"] for r in layer_table(image_size, batch))


def aux_flops(image_size, batch):
    """Поэлементные операции (ReLU, maxpool, GAP, bias): ~39 * B * S^2, около 0,2% от flops()."""
    return sum(r["aux_flops"] for r in layer_table(image_size, batch))


def bytes_moved(image_size, batch):
    """Минимальный трафик DRAM одного прямого прохода, байты.

    Закрытая форма: 380 * B * S^2 + 4 161 296 (веса) + 8 592 * B (голова).
    """
    return sum(r["bytes"] for r in layer_table(image_size, batch))


# -------------------------------------------------------------------- память
def activation_timeline(image_size, batch):
    """Сколько байт активаций живёт в момент выполнения каждого слоя.

    nn.Sequential: во время слоя живы его вход и его выход; предыдущий
    тензор освобождается, когда цикл перезаписывает переменную.
    Вход x держит вызывающий код, поэтому он учтён отдельно в memory().
    ReLU inplace ничего не выделяет. MaxPool на CUDA вызывает
    max_pool2d_with_indices и временно держит индексы int64.
    """
    S = np.asarray(image_size, dtype=np.float64)
    B = np.asarray(batch, dtype=np.float64)
    e = act_elems(S)
    a = {k: alloc(FP32 * B * v) for k, v in e.items()}
    a["idx"] = alloc(INT64 * B * e["idx"])
    return {
        "conv1":   a["a1"],
        "maxpool": a["a1"] + a["a2"] + a["idx"],
        "conv2":   a["a2"] + a["a3"],
        "conv3":   a["a3"] + a["a4"],
        "conv4":   a["a4"] + a["a5"],
        "conv5":   a["a5"] + a["a6"],
        "conv6":   a["a6"] + a["a7"],
        "gap":     a["a7"] + a["g"],
        "fc1":     a["g"] + a["f1"],
        "fc2":     a["f1"] + a["f2"],
    }


def memory(image_size, batch, runtime_bytes=RUNTIME_BYTES_DEFAULT):
    """Пик torch.cuda.max_memory_allocated() за один прямой проход, байты.

    = веса + вход + рабочая область cuBLAS + максимум по слоям живых активаций.
    Максимум всегда на maxpool: a1 + a2 + индексы = (32 + 8 + 16) * B * S^2 байт.
    Закрытая форма (без округления до 512 Б):
        Memory = 4 161 536 + runtime_bytes + 12 * B * S^2 + 56 * B * S^2
               = 4 161 536 + runtime_bytes + 68 * B * S^2.
    Здесь ничего не подбирается: формула считает память точно, с точностью
    до округления allocator-а (512 Б). Не учтена только рабочая область cuDNN
    внутри свёрток: её размер выбирает эвристика cuDNN, measure.py показывает,
    есть ли она.
    """
    S = np.asarray(image_size, dtype=np.float64)
    B = np.asarray(batch, dtype=np.float64)
    x = alloc(FP32 * B * 3 * S ** 2)
    peak_act = np.max(np.stack(np.broadcast_arrays(*activation_timeline(S, B).values())), axis=0)
    return WEIGHT_BYTES + x + runtime_bytes + peak_act


# ------------------------------------------------------------------- latency
def layer_times(image_size, batch, theta):
    """Время каждого ядра по roofline: max(FLOPs / P, bytes / BW)."""
    P, BW = theta["peak_flops"], theta["bandwidth"]
    return [np.maximum(r["flops"] / P, r["bytes"] / BW) for r in layer_table(image_size, batch)]


def latency(image_size, batch, theta):
    """Время одного прямого прохода, секунды.

    Latency = t0 + sum_k max(FLOPs_k / P, Bytes_k / BW)
    theta = {t0: постоянная задержка (запуски 17 ядер, Python, синхронизация), с;
             peak_flops: достижимая производительность свёрток, FLOP/s;
             bandwidth: достижимая пропускная способность памяти, байт/с}
    Режимы: t0 доминирует — launch-bound; доминируют ядра, где bytes/BW > FLOPs/P —
    memory-bound; доминируют свёртки с FLOPs/P > bytes/BW — compute-bound.
    """
    return theta["t0"] + sum(layer_times(image_size, batch, theta))


def regime(image_size, batch, theta):
    """0 = launch-bound, 1 = memory-bound, 2 = compute-bound: какая часть времени больше."""
    t_mem = 0.0
    t_cmp = 0.0
    for r in layer_table(image_size, batch):
        tc, tm = r["flops"] / theta["peak_flops"], r["bytes"] / theta["bandwidth"]
        t_cmp = t_cmp + np.where(tc >= tm, tc, 0.0)
        t_mem = t_mem + np.where(tc < tm, tm, 0.0)
    t0 = theta["t0"] + 0 * t_mem
    return np.argmax(np.stack(np.broadcast_arrays(t0, t_mem, t_cmp)), axis=0)


# -------------------------------------------------------------------- energy
def energy(image_size, batch, theta_energy, theta=None):
    """Энергия одного прямого прохода всей GPU, джоули.

    Energy = P_static * Latency(S, B, theta) + e_flop * FLOPs + e_byte * Bytes
    theta_energy = {p_static: мощность, которую GPU тратит, пока занята (Вт);
                    e_flop: Дж на FLOP; e_byte: Дж на байт DRAM-трафика}
    Если theta (параметры latency) не передан, он берётся из theta_energy["latency_theta"].
    """
    th = theta if theta is not None else theta_energy["latency_theta"]
    return (theta_energy["p_static"] * latency(image_size, batch, th)
            + theta_energy["e_flop"] * flops(image_size, batch)
            + theta_energy["e_byte"] * bytes_moved(image_size, batch))


if __name__ == "__main__":
    print("параметров:", N_PARAMS, " веса, байт:", WEIGHT_BYTES)
    for S, B in [(32, 1), (224, 64), (512, 256)]:
        print(f"S={S:3d} B={B:3d}  FLOPs={flops(S, B):.4g}  bytes={bytes_moved(S, B):.4g}  "
              f"memory={memory(S, B) / 2**20:.1f} MiB")
