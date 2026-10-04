# ==========================================================
#  真题解答：宿舍楼用电预测 + 节能方案优选
#  三问：① 预测  ② 排名  ③ 预算分配
# ==========================================================

import numpy as np
from scipy.optimize import linprog

def title(s):
    print("\n" + "=" * 56)
    print(f"  {s}")
    print("=" * 56)


# =========================================================
#  第一问：GM(1,1) 预测未来 3 年用电量
# =========================================================
title("第一问：预测未来 3 年用电量（GM(1,1)）")

x0 = np.array([118, 126, 135, 143, 152, 163, 174, 186, 198, 212, 227, 243],
              dtype=float)
years = np.arange(2014, 2026)

def gm11(x0, steps=3):
    n = len(x0)
    x1 = np.cumsum(x0)                                   # 一次累加
    z1 = np.array([0.5 * (x1[k] + x1[k-1]) for k in range(1, n)])
    B = np.column_stack([-z1, np.ones(n - 1)])
    a, b = np.linalg.lstsq(B, x0[1:], rcond=None)[0]     # 最小二乘求参数

    def x1_hat(k):
        return (x0[0] - b / a) * np.exp(-a * k) + b / a

    fit = np.zeros(n)
    fit[0] = x0[0]
    for k in range(1, n):
        fit[k] = x1_hat(k) - x1_hat(k - 1)
    future = np.array([x1_hat(n + k) - x1_hat(n + k - 1) for k in range(steps)])
    return a, b, fit, future

a, b, fit, future = gm11(x0, steps=3)

print(f"\n发展系数 a = {a:.5f}，灰作用量 b = {b:.5f}")
print("\n  年份    实际    拟合    相对误差")
for y, real, f in zip(years, x0, fit):
    print(f"  {y}   {real:>6.0f}  {f:>7.2f}   {abs(real-f)/real*100:>6.2f}%")

e = x0 - fit
C = np.std(e) / np.std(x0)
p = np.mean(np.abs(e - e.mean()) < 0.6745 * np.std(x0))
print(f"\n平均相对误差：{np.mean(np.abs(e/x0))*100:.2f}%")
print(f"后验差比值 C = {C:.4f}（<0.35 为好）")
print(f"小误差概率 p = {p:.4f}（>0.95 为好）")
print("→ 精度等级：好" if (C < 0.35 and p > 0.95) else "→ 精度等级：需改进")

print("\n未来 3 年预测用电量：")
for i, v in enumerate(future):
    print(f"  {years[-1]+1+i} 年：{v:.1f} 亿 kWh")


# =========================================================
#  第二问：熵权 + TOPSIS 给方案排名
# =========================================================
title("第二问：四个方案谁更好（TOPSIS）")

# 每行一个方案，每列一个指标：初投资、年节电量、年维护费、施工难度
data = np.array([
    [20,  30, 2, 3],      # A 更换LED照明
    [45,  65, 5, 6],      # B 空调智能控制
    [80,  55, 3, 8],      # C 外墙保温
    [120, 90, 6, 7],      # D 屋顶光伏
], dtype=float)

names = ["A LED照明", "B 空调智能控制", "C 外墙保温", "D 屋顶光伏"]
indicators = ["初投资", "年节电量", "年维护费", "施工难度"]
# 方向：True = 越大越好，False = 越小越好
# 初投资越小越好；年节电量越大越好；维护费越小越好；难度越小越好
directions = [False, True, False, False]

def topsis(data_matrix, directions):
    # 注意：变量名用小写。全大写（比如 X）在 Python 约定里表示"常量"，
    # 常量不允许被重新赋值，否则代码检查工具（basedpyright）会报错
    x = np.array(data_matrix, dtype=float)
    m, n = x.shape

    # 1) 规范化到 0~1
    Z = np.zeros_like(x)
    for j in range(n):
        col = x[:, j]
        Z[:, j] = col / col.max() if directions[j] else col.min() / col

    # 2) 熵权法算权重
    P = Z / Z.sum(axis=0)
    k = 1 / np.log(m)
    E = np.zeros(n)
    for j in range(n):
        pp = np.where(P[:, j] == 0, 1e-12, P[:, j])
        E[j] = -k * np.sum(pp * np.log(pp))
    W = (1 - E) / (1 - E).sum()

    # 3) TOPSIS：离"最好的"最近、离"最差的"最远
    V = Z * W
    best, worst = V.max(axis=0), V.min(axis=0)
    D_best = np.array([np.linalg.norm(r - best) for r in V])
    D_worst = np.array([np.linalg.norm(r - worst) for r in V])
    return W, D_worst / (D_best + D_worst)

W, score = topsis(data, directions)

print("\n各指标权重（熵权法自动算出）：")
for nm, w in zip(indicators, W):
    print(f"  {nm}：{w*100:>6.2f}%")

print("\n方案排名：")
for rank, idx in enumerate(np.argsort(-score), start=1):
    print(f"  第{rank}名：{names[idx]:<16} 得分 {score[idx]:.4f}")


# =========================================================
#  第三问：80 万预算怎么分配（线性规划）
# =========================================================
title("第三问：80 万预算的最优分配（线性规划）")

cost = np.array([20, 45, 80, 120], dtype=float)      # 各方案全额投资
save = np.array([30, 65, 55, 90], dtype=float)       # 各方案全额时的年节电量

# linprog 只求最小值，所以把目标取负号
result = linprog(-save, A_ub=[cost], b_ub=[80], bounds=[(0, 1)] * 4)

print("\n先看性价比（每万元投资能省多少电）：")
for nm, s, c in zip(names, save, cost):
    print(f"  {nm:<16} {s/c:.3f} 万kWh/万元")
print("  → 性价比排序：A > B > D > C，所以优先投 A、B")

if result.success:
    x = result.x
    print("\n最优投资方案：")
    total_save = 0
    total_cost = 0
    for nm, xi, c, s in zip(names, x, cost, save):
        print(f"  {nm:<16} 投入 {xi*100:>5.1f}%（{c*xi:>5.1f} 万元）"
              f" → 年节电 {s*xi:>6.2f} 万 kWh")
        total_save += s * xi
        total_cost += c * xi
    print(f"\n  总投入：{total_cost:.1f} 万元（预算 80 万）")
    print(f"  年节电总量：{total_save:.2f} 万 kWh")
    print("\n  思路解释：按'性价比'从高到低买，A 全买、B 全买，")
    print("  剩下 15 万投给性价比第三的 D，投 12.5%")
else:
    print("没找到解：", result.message)
