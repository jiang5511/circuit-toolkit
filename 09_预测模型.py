# ==========================================================
#  第九个程序：预测模型（数学建模必备）
#    模型一：灰色预测 GM(1,1) —— 数据少也能预测（8~15 个数据点就够）
#    模型二：ARIMA —— 时间序列经典方法（数据多一些更准）
#  场景：某地区年用电量预测（很适合电气专业）
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 历史数据：某地区 2014~2025 年用电量（亿千瓦时）----------
years = np.arange(2014, 2026)
x0 = np.array([118, 126, 135, 143, 152, 163, 174, 186, 198, 212, 227, 243],
              dtype=float)

print("=" * 58)
print("  用电量预测：GM(1,1) 灰色预测  vs  ARIMA")
print("=" * 58)
print(f"\n历史数据（{len(x0)} 年）：")
for y, v in zip(years, x0):
    print(f"  {y} 年：{v:.0f} 亿 kWh")


# =========================================================
#  模型一：GM(1,1) 灰色预测
# =========================================================
# 思路：原始数据看着乱，先"累加"让曲线变光滑，
#      再用一个微分方程去拟合这条光滑曲线，最后"累减"还原回去
# 优点：数据量少（七八个点）也能建模型，这是它最大的价值

def gm11(x0, 预测年数=3):
    n = len(x0)

    # 第一步：一次累加（AGO），把 [118,126,135...] 变成累加序列
    x1 = np.cumsum(x0)

    # 第二步：构造"背景值" z1，就是相邻两个累加值的平均数
    z1 = np.array([0.5 * (x1[k] + x1[k-1]) for k in range(1, n)])

    # 第三步：最小二乘法求参数 a（发展系数）和 b（灰作用量）
    B = np.column_stack([-z1, np.ones(n - 1)])      # 左边矩阵
    Y = x0[1:]                                       # 右边向量
    # np.linalg.lstsq 就是最小二乘求解，返回的第一项就是参数
    a, b = np.linalg.lstsq(B, Y, rcond=None)[0]

    # 第四步：用求出的公式预测累加序列
    def x1_hat(k):
        return (x0[0] - b / a) * np.exp(-a * k) + b / a

    # 对历史年份做拟合（用于看模型准不准）
    fit = np.zeros(n)
    fit[0] = x0[0]
    for k in range(1, n):
        fit[k] = x1_hat(k) - x1_hat(k - 1)           # 累减还原

    # 对未来的预测
    future = np.zeros(预测年数)
    for k in range(预测年数):
        future[k] = x1_hat(n + k) - x1_hat(n + k - 1)

    return a, b, fit, future


a, b, gm_fit, gm_future = gm11(x0, 预测年数=3)

# ---------- 精度检验（后验差检验，这是评委必看的）----------
e = x0 - gm_fit                        # 残差 = 真实值 - 预测值
S1 = np.std(x0)                        # 原始数据的标准差
S2 = np.std(e)                         # 残差的标准差
C = S2 / S1                            # 后验差比值：越小越好
p = np.mean(np.abs(e - e.mean()) < 0.6745 * S1)   # 小误差概率：越大越好

print("\n" + "-" * 58)
print("【模型一：GM(1,1) 灰色预测】")
print(f"  发展系数 a = {a:.5f}，灰作用量 b = {b:.5f}")
print("\n  年份    实际值     拟合值     相对误差")
for y, real, fit_v in zip(years, x0, gm_fit):
    err = abs(real - fit_v) / real * 100
    print(f"  {y}   {real:>7.1f}  {fit_v:>8.2f}   {err:>7.2f}%")

print(f"\n  平均相对误差：{np.mean(np.abs(e / x0)) * 100:.2f}%")
print(f"  后验差比值 C = {C:.4f}（< 0.35 为'好'）")
print(f"  小误差概率 p = {p:.4f}（> 0.95 为'好'）")
if C < 0.35 and p > 0.95:
    print("  → 精度等级：好（可以放心用）")
elif C < 0.5 and p > 0.8:
    print("  → 精度等级：合格")
else:
    print("  → 精度等级：勉强，建议换模型")

print("\n  未来 3 年预测：")
for i, v in enumerate(gm_future):
    print(f"  {years[-1] + 1 + i} 年：{v:.1f} 亿 kWh")


# =========================================================
#  模型二：ARIMA 时间序列
# =========================================================
# order=(p, d, q) 三个参数：
#   p = 自回归阶数（用过去几期）
#   d = 差分次数（把趋势消掉，让数据变平稳）
#   q = 移动平均阶数
# 这里取 (1, 1, 1)，是最常用的组合

print("\n" + "-" * 58)
print("【模型二：ARIMA(1,1,1)】")

model = ARIMA(x0, order=(1, 1, 1))
res = model.fit()

arima_fit = res.fittedvalues                 # 对历史的拟合
arima_future = res.forecast(steps=3)         # 预测未来 3 年

print("\n  年份    实际值     拟合值")
for y, real, fit_v in zip(years, x0, arima_fit):
    print(f"  {y}   {real:>7.1f}  {fit_v:>8.2f}")

err_arima = (x0[1:] - arima_fit[1:]) / x0[1:] * 100
print(f"\n  平均相对误差：{np.mean(np.abs(err_arima)):.2f}%（从第 2 年起算）")

print("\n  未来 3 年预测：")
for i, v in enumerate(arima_future):
    print(f"  {years[-1] + 1 + i} 年：{v:.1f} 亿 kWh")

# ---------- 两个模型对比一下 ----------
print("\n" + "-" * 58)
print("【两个模型对比】")
gm_err = np.mean(np.abs(e / x0)) * 100
ar_err = np.mean(np.abs(err_arima))
print(f"  GM(1,1) 平均误差：{gm_err:.2f}%")
print(f"  ARIMA   平均误差：{ar_err:.2f}%")
better = "GM(1,1)" if gm_err < ar_err else "ARIMA"
print(f"  → 这份数据上，{better} 更准")
print("\n经验：数据少于 10 个用 GM(1,1)；数据多且有明显周期性，用 ARIMA")


# =========================================================
#  画图：历史数据 + 两条拟合线 + 未来预测
# =========================================================
future_years = np.arange(years[-1] + 1, years[-1] + 4)

plt.figure(figsize=(11, 6))
plt.plot(years, x0, "ko-", linewidth=2, markersize=7, label="历史实际值")
plt.plot(years, gm_fit, "b--", linewidth=2, label="GM(1,1) 拟合")
plt.plot(years[1:], arima_fit[1:], "g--", linewidth=2, label="ARIMA 拟合")

# 画未来的预测（虚线往右延伸）
plt.plot(future_years, gm_future, "b-s", markersize=8, label="GM(1,1) 预测")
plt.plot(future_years, arima_future, "g-^", markersize=8, label="ARIMA 预测")

# 画一条竖线分隔历史和未来
plt.axvline(x=years[-1] + 0.5, color="red", linestyle=":", linewidth=2)
plt.text(years[-1] + 0.6, min(x0) * 1.05, "← 历史 | 预测 →", color="red")

plt.title("年用电量预测：GM(1,1) 与 ARIMA 对比", fontsize=15)
plt.xlabel("年份", fontsize=12)
plt.ylabel("用电量（亿 kWh）", fontsize=12)
plt.grid(True, linestyle=":", alpha=0.7)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("用电量预测.png", dpi=150, bbox_inches="tight")
print("\n图片已保存：用电量预测.png")
plt.show()
