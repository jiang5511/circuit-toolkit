# ==========================================================
#  第八个程序：数学建模工具箱
#  里面装了三个比赛最常用的模型：
#    工具一：AHP 层次分析法（用来定权重）
#    工具二：熵权 + TOPSIS（用来给方案排名）
#    工具三：线性规划（用来求最优方案）
# ==========================================================

import numpy as np
from scipy.optimize import linprog


def print_title(标题):
    """打印一条分隔线和标题（def 就是"定义一个函数"，可以反复调用）"""
    print("\n" + "=" * 55)
    print(f"  {标题}")
    print("=" * 55)


# =========================================================
#  工具一：AHP 层次分析法 —— 解决"各因素该占多少权重"
# =========================================================
# 用法：专家（就是你）两两比较重要性，填成一张判断矩阵
# 数字含义：1=同样重要，3=稍微重要，5=明显重要，7=强烈重要，9=极端重要
#           2/4/6/8 是中间值；倒数表示"反过来"（比如 1/3）

def ahp(A):
    A = np.array(A, dtype=float)
    n = A.shape[0]

    # 方根法求权重：每一行相乘 → 开 n 次方 → 归一化
    # np.prod(..., axis=1) 表示"按行连乘"
    geomean = np.prod(A, axis=1) ** (1 / n)
    w = geomean / geomean.sum()

    # 一致性检验：判断你填的矩阵自不自相矛盾
    # 比如你说 A 比 B 重要、B 比 C 重要，却又说 C 比 A 重要，那就矛盾了
    lambda_max = np.mean((A @ w) / w)      # @ 是矩阵乘法
    CI = (lambda_max - n) / (n - 1)

    # RI 是随机一致性指标，查表得到的固定值
    RI_table = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12,
                6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    RI = RI_table.get(n, 1.49)
    CR = CI / RI if RI != 0 else 0.0

    return w, lambda_max, CI, CR


print_title("工具一：AHP 层次分析法")

# 例子：选一台仪器，考虑三个因素：价格、精度、售后
# 判断矩阵表示：价格(行1) 精度(行2) 售后(行3) 两两比较
# 这里认为"精度"最重要
A = [
    [1,     1/3,   1/5],     # 价格 相对 价格/精度/售后
    [3,     1,     1/3],     # 精度
    [5,     3,     1],       # 售后
]

w, lambda_max, CI, CR = ahp(A)
names = ["价格", "精度", "售后"]

print("\n权重结果：")
for name, wi in zip(names, w):
    print(f"  {name}：{wi*100:>6.2f}%")

print(f"\n最大特征值 λmax = {lambda_max:.4f}")
print(f"一致性指标 CI = {CI:.4f}")
print(f"一致性比例 CR = {CR:.4f}")
if CR < 0.1:
    print("  → CR < 0.1，判断矩阵通过一致性检验（自圆其说，可用）")
else:
    print("  → CR >= 0.1，判断矩阵自相矛盾，需要重新打分！")


# =========================================================
#  工具二：entropy_weight + TOPSIS —— 解决"哪个方案最好"
# =========================================================
# TOPSIS 的思想很朴素：
#   最好的方案 = 离"最理想的方案"最近，且离"最差的方案"最远

def entropy_weight(X, 正向指标列表):
    """根据数据的离散程度自动算权重：某指标差异越大，权重越高"""
    X = np.array(X, dtype=float)
    m, n = X.shape

    # 第一步：规范化，把所有指标压缩到 0~1
    Z = np.zeros_like(X)
    for j in range(n):
        col = X[:, j]
        if 正向指标列表[j]:        # 越大越好：除以最大值
            Z[:, j] = col / col.max()
        else:                      # 越小越好：最小值除以它
            Z[:, j] = col.min() / col

    # 第二步：算每个指标的"熵"
    # 熵越小 = 数据越分散 = 这个指标区分度越高 = 该给更大权重
    P = Z / Z.sum(axis=0)          # 每列归一化成"占比"
    k = 1 / np.log(m)
    E = np.zeros(n)
    for j in range(n):
        p = P[:, j]
        p = np.where(p == 0, 1e-12, p)     # 避免 log(0) 报错
        E[j] = -k * np.sum(p * np.log(p))
    D = 1 - E                      # 差异系数
    W = D / D.sum()                # 熵权
    return Z, W


def topsis(X, 正向指标列表):
    X = np.array(X, dtype=float)
    Z, W = entropy_weight(X, 正向指标列表)

    V = Z * W                                  # 加权后的规范化矩阵

    # 正理想解（每列最大值）和负理想解（每列最小值）
    best = V.max(axis=0)
    worst = V.min(axis=0)

    # 到两者的欧氏距离（np.linalg.norm 就是求向量长度）
    D_best = np.array([np.linalg.norm(row - best) for row in V])
    D_worst = np.array([np.linalg.norm(row - worst) for row in V])

    # 贴近度：越大越好（离最差的距离占比越大）
    score = D_worst / (D_best + D_worst)
    return W, score


print_title("工具二：熵权 + TOPSIS 方案排序")

# 例子：4 个候选电源方案，3 个指标：成本(越低越好)、效率(越高越好)、寿命(越高越好)
data = [
    [120, 0.85, 5000],
    [150, 0.92, 8000],
    [100, 0.78, 4000],
    [180, 0.95, 10000],
]
# True 表示"这个指标越大越好"，False 表示"越小越好"
directions = [False, True, True]      # 成本越小越好；效率、寿命越大越好

W, score = topsis(data, directions)

print("\n熵权法算出的指标权重：")
for name, wi in zip(["成本", "效率", "寿命"], W):
    print(f"  {name}：{wi*100:>6.2f}%")

print("\nTOPSIS 得分与排名：")
ranking = np.argsort(-score)          # 从大到小排序，返回下标
for rank, idx in enumerate(ranking, start=1):
    print(f"  第{rank}名：方案{idx+1}，得分 {score[idx]:.4f}")

print("\n结论：", end="")
print(f"推荐 方案{ranking[0]+1}")


# =========================================================
#  工具三：线性规划 —— 解决"怎么安排最划算"
# =========================================================

print_title("工具三：线性规划求最优")

# 例子：工厂生产 A、B 两种产品
#   利润：A 每件 3 元，B 每件 5 元
#   约束：
#     1) A 最多生产 4 件
#     2) B 最多生产 6 件
#     3) 总的工时限制：3×A + 2×B <= 18
#   问：各生产多少件，总利润最大？
#
# 注意：scipy 的 linprog 只会求"最小值"，
#       所以要最大化利润，就把目标函数取负号 → 变成最小化

c = [-3, -5]                          # 目标函数系数（取负，因为要求最大）
A_ub = [[1, 0],                       # 1×A + 0×B <= 4
        [0, 1],                       # 0×A + 1×B <= 6
        [3, 2]]                       # 3×A + 2×B <= 18
b_ub = [4, 6, 18]

result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None), (0, None)])

if result.success:
    xA, xB = result.x
    print(f"\n最优生产方案：")
    print(f"  A 产品：{xA:.2f} 件")
    print(f"  B 产品：{xB:.2f} 件")
    print(f"  最大利润：{-result.fun:.2f} 元")
    print("\n（手算验证：A=2, B=6 时利润 = 3×2 + 5×6 = 36，"
          "且 3×2+2×6=18 刚好用完工时）")
else:
    print("没找到最优解：", result.message)


# =========================================================
#  怎么换成你自己的题？
#    AHP：改判断矩阵 A 和名字 names
#    TOPSIS：改 data（数据）和 directions（方向）
#    线性规划：改 c（目标）、A_ub 和 b_ub（约束）
# =========================================================
