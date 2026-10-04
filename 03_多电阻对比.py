# ==========================================================
#  第三个程序：把多条曲线画在同一张图里对比
#  好处：不用一张张关窗口，对比一目了然
# ==========================================================

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 可改的参数 ----------
U = 5.0                              # 电源电压 5V
C = 0.0001                           # 电容 100 µF
R_list = [2000, 10000, 30000]        # 三个不同的电阻（欧姆），会自动画出三条曲线
T_MAX = 10.0                         # 横轴画 0~10 秒
# --------------------------------

t = np.linspace(0, T_MAX, 500)       # 时间点：0 到 10 秒，取 500 个点

# ---------- 关键：一张画布上切出两块，上下排列 ----------
# subplot(行数, 列数, 第几块)：2 行 1 列，第 1 块 = 上面
# figsize 是整张画布的大小
fig = plt.figure(figsize=(10, 9))

# ===== 上面那块图：画充电 =====
plt.subplot(2, 1, 1)                 # 选中第 1 块（上）
for R in R_list:
    tau = R * C                                    # 每个电阻对应一个时间常数
    u = U * (1 - np.exp(-t / tau))                 # 充电公式
    plt.plot(t, u, linewidth=2, label=f"R={R/1000:.0f} kΩ（τ={tau:.2f} s）")
plt.title("不同电阻下的充电过程（R 越小，充得越快）", fontsize=14)
plt.ylabel("电容电压 u（V）", fontsize=12)
plt.grid(True, linestyle=":", alpha=0.7)
plt.legend(fontsize=11)

# ===== 下面那块图：画放电 =====
plt.subplot(2, 1, 2)                 # 选中第 2 块（下）
for R in R_list:
    tau = R * C
    u = U * np.exp(-t / tau)                       # 放电公式
    plt.plot(t, u, linewidth=2, label=f"R={R/1000:.0f} kΩ（τ={tau:.2f} s）")
plt.title("不同电阻下的放电过程", fontsize=14)
plt.xlabel("时间 t（秒）", fontsize=12)
plt.ylabel("电容电压 u（V）", fontsize=12)
plt.grid(True, linestyle=":", alpha=0.7)
plt.legend(fontsize=11)

# 让两块图不要挤在一起
plt.tight_layout()

# 保存成一整张图片（上下两张都在里面）
plt.savefig("多电阻对比.png", dpi=150, bbox_inches="tight")
print("图片已保存：多电阻对比.png（上下两张图在同一张里）")

# 只弹一次窗口，而且只显示一张图 —— 不用再一个个关了
plt.show()
