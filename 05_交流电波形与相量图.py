# ==========================================================
#  第五个程序：正弦交流电波形 + 相量图
#  对应课程：电路分析里的正弦稳态、相位差、功率因数
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 参数（可以改）----------
f = 50.0          # 频率：我国电网是 50 Hz
U_rms = 220.0     # 电压有效值 220 V（就是家里插座的电压）
phi = 36.87       # 相位差：电流滞后电压的角度（度）
                  # cos(36.87°) = 0.8，也就是功率因数 0.8
# -----------------------------------

# 角频率 ω = 2πf，单位是 弧度/秒
w = 2 * np.pi * f

# 最大值 = 有效值 × √2 。np.sqrt() 是开平方
U_m = U_rms * np.sqrt(2)
print(f"周期 T = 1/f = {1/f:.4f} 秒")
print(f"电压有效值 {U_rms} V → 最大值 Um = {U_m:.1f} V")
print(f"功率因数 cosφ = cos({phi}°) = {np.cos(np.deg2rad(phi)):.3f}\n")

# ---------- 准备时间轴：画 2 个周期（0 ~ 0.04 秒）----------
T = 1 / f
t = np.linspace(0, 2 * T, 1000)

# 正弦交流电：u(t) = Um × sin(ωt)
u = U_m * np.sin(w * t)

# 电流也按正弦变化，但相位滞后 φ 度
# np.deg2rad 把"度"转成"弧度"（Python 的三角函数只认弧度）
I_m = 10.0        # 假设电流最大值 10 A
i = I_m * np.sin(w * t - np.deg2rad(phi))

# ---------- 开始画图：上下两块 ----------
plt.figure(figsize=(11, 10))

# ========== 上面：波形图 ==========
plt.subplot(2, 1, 1)
# 电流太小不好跟电压比，这里放大 20 倍画出来，方便看相位差
plt.plot(t * 1000, u, "b-", linewidth=2, label="电压 u（V）")
plt.plot(t * 1000, i * 20, "r-", linewidth=2, label="电流 i（放大 20 倍）")

# 画一条 0 线，方便看什么时候过零
plt.axhline(0, color="black", linewidth=0.8)

plt.title(f"正弦交流电波形（f = {f:.0f} Hz，电流滞后电压 {phi}°）", fontsize=14)
plt.xlabel("时间 t（毫秒）", fontsize=12)
plt.ylabel("瞬时值", fontsize=12)
plt.grid(True, linestyle=":", alpha=0.7)
plt.legend(fontsize=11)

# 标出周期 T 的位置
plt.axvline(x=T * 1000, color="green", linestyle="--", linewidth=1.5)
plt.text(T * 1000 + 0.5, U_m * 0.6, f"一个周期 T = {T*1000:.1f} ms",
         color="green", fontsize=11)

# ========== 下面：相量图 ==========
# 相量：用一个"带方向的箭头"表示正弦量，箭头长度 = 大小，角度 = 相位
plt.subplot(2, 1, 2)

# 画个虚线圆当参考
theta = np.linspace(0, 2 * np.pi, 200)
plt.plot(np.cos(theta) * U_rms, np.sin(theta) * U_rms, "gray",
         linestyle=":", linewidth=1)

# 画电压相量：角度 0°，长度 = 220
plt.arrow(0, 0, U_rms, 0, head_width=12, head_length=14,
          fc="blue", ec="blue", linewidth=2.5)
plt.text(U_rms + 8, -12, f"U = {U_rms} V∠0°", color="blue", fontsize=12)

# 画电流相量：角度 -φ（滞后 = 顺时针），长度放大便于观看
I_len = 150       # 相量图中人为放长一点，好看
I_x = I_len * np.cos(np.deg2rad(-phi))
I_y = I_len * np.sin(np.deg2rad(-phi))
plt.arrow(0, 0, I_x, I_y, head_width=12, head_length=14,
          fc="red", ec="red", linewidth=2.5)
plt.text(I_x + 8, I_y - 15, f"I 滞后 {phi}°", color="red", fontsize=12)

# 画两条虚线连到坐标轴，能看出角度差
plt.plot([0, I_x], [0, I_y], "r:")
plt.plot([0, U_rms], [0, 0], "b:")

plt.title("相量图（箭头长度=大小，角度=相位）", fontsize=14)
plt.xlabel("实轴", fontsize=12)
plt.ylabel("虚轴", fontsize=12)
plt.axis("equal")                       # 让横纵比例一致，角度才不会变形
plt.grid(True, linestyle=":", alpha=0.7)
plt.axhline(0, color="black", linewidth=0.8)
plt.axvline(0, color="black", linewidth=0.8)

plt.tight_layout()
plt.savefig("交流电波形与相量图.png", dpi=150, bbox_inches="tight")
print("图片已保存：交流电波形与相量图.png")
plt.show()
