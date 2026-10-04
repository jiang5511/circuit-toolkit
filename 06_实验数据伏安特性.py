# ==========================================================
#  第六个程序：处理实验数据 —— 画伏安特性曲线并求电阻
#  场景：测了一堆电压电流数据，想验证欧姆定律、求出电阻值
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt
import csv            # csv 是 Python 自带的，用来读表格文件
import os             # os 用来判断文件是否存在

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# =========================================================
#  第一部分：拿到数据（两种方式，二选一）
# =========================================================

# 如果你把实验数据存成了 data.csv（两列：电压,电流），程序会自动读它
# 没有这个文件的话，就用下面这组模拟数据（假装是一次实验的测量结果）
csv_file = "data.csv"

if os.path.exists(csv_file):
    # ---- 方式一：从 CSV 文件读 ----
    U_list, I_list = [], []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)                # 跳过第一行表头
        for row in reader:
            U_list.append(float(row[0]))
            I_list.append(float(row[1]))
    print(f"已从 {csv_file} 读取 {len(U_list)} 组数据")
else:
    # ---- 方式二：直接用这里写好的数据 ----
    # 这是一组"电压 - 电流"测量值，带一点点测量误差
    U_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    I_list = [0.051, 0.098, 0.152, 0.203, 0.249, 0.305, 0.349, 0.402, 0.448, 0.503]
    print("未找到 data.csv，使用内置的示例数据（10 组）")

# 把普通列表转成 numpy 数组，才能做数学运算
U = np.array(U_list)
I = np.array(I_list)

# =========================================================
#  第二部分：线性拟合（也就是"画一条最贴合这些点的直线"）
# =========================================================

# np.polyfit(x, y, 1) 表示用 1 次函数（就是直线 y = kx + b）去拟合
# 它会返回两个值：k（斜率）和 b（截距）
k, b = np.polyfit(U, I, 1)

# 欧姆定律：I = U / R，所以 I-U 图线的斜率 k = 1 / R
# 于是电阻 R = 1 / k
R = 1 / k

print("\n========== 拟合结果 ==========")
print(f"拟合直线：I = {k:.5f} × U + {b:.5f}")
print(f"斜率 k = {k:.5f} A/V")
print(f"所以电阻 R = 1/k = {R:.2f} Ω")
print(f"截距 b = {b:.5f} A（理想情况下应该接近 0）")

# 计算 R²（读作 R 平方）：越接近 1，说明点越贴合直线，测量越靠谱
# np.corrcoef 是相关系数，平方后就是 R²
r2 = np.corrcoef(U, I)[0, 1] ** 2
print(f"拟合优度 R^2 = {r2:.6f}（越接近 1 越线性，说明越符合欧姆定律）")

# =========================================================
#  第三部分：画图
# =========================================================

plt.figure(figsize=(9, 6))

# 画实测数据点：'o' 表示画圆点，不连线
plt.plot(U, I, "o", color="blue", markersize=8, label="实测数据点")

# 画拟合直线：用拟合出的 k 和 b 算出一串 y 值
U_line = np.linspace(0, max(U) * 1.1, 100)
I_line = k * U_line + b
plt.plot(U_line, I_line, "r-", linewidth=2, label=f"拟合直线（R = {R:.1f} Ω）")

plt.title("伏安特性曲线", fontsize=15)
plt.xlabel("电压 U（V）", fontsize=12)
plt.ylabel("电流 I（A）", fontsize=12)
plt.grid(True, linestyle=":", alpha=0.7)
plt.legend(fontsize=11)

# 在图上标注结果，交作业时很省事
plt.text(0.5, max(I) * 0.85,
         f"R = {R:.2f} Ω\nR^2 = {r2:.5f}",
         fontsize=12, bbox=dict(boxstyle="round", facecolor="lightyellow"))

plt.tight_layout()
plt.savefig("伏安特性曲线.png", dpi=150, bbox_inches="tight")
print("\n图片已保存：伏安特性曲线.png")
plt.show()

# =========================================================
#  小提示：怎么换成你自己的实验数据？
#  在本文件夹里新建一个 data.csv，第一行写：电压,电流
#  下面每行一组数据，例如：
#  0.5,0.051
#  1.0,0.098
#  保存后再运行本程序，就会自动读你的数据了
# =========================================================
