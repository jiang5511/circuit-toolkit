# ==========================================================
#  第二个程序：画出 RC 电路的充放电曲线
#  你会第一次看到：代码能画出图！
# ==========================================================

# 第一步：把"画图工具箱"拿过来
# matplotlib 就是 Python 里专门画图的工具，pyplot 是它最常用的一个小工具
import matplotlib.pyplot as plt

# 让图能显示中文（不加这两行，中文会变成方框 □□□）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]  # 用黑体显示中文
plt.rcParams["axes.unicode_minus"] = False                       # 正常显示负号

# numpy 是"数学工具箱"，里面有很多好用的数学函数和数组
import numpy as np

# ---------- 这里是可以自己改的参数（改完再运行一次看看变化）----------
U = 5.0        # 电源电压，单位：伏特 V
R = 10000.0    # 电阻，单位：欧姆 Ω（10000 欧 = 10 kΩ）
C = 0.00005     # 电容，单位：法拉 F（0.00005 F = 50 µF）
T_MAX = 10.0   # 横轴固定画 0~10 秒（固定住它，改 R/C 才看得出快慢）
# -------------------------------------------------------------------

# 第二步：算时间常数 τ（希腊字母 tau，读作"涛"）
# τ = R × C，它决定充放电的快慢
# 物理意义：经过 1 个 τ 的时间，电容电压变化了 63%
#           经过 5 个 τ，就认为充电基本完成（达到 99%）
tau = R * C
print(f"时间常数 τ = R × C = {tau:.3f} 秒")
print(f"经过 5τ = {5*tau:.3f} 秒后，认为充放电基本完成\n")

# 第三步：准备一堆时间点
# np.linspace(起点, 终点, 个数) = 在起点到终点之间平均取 N 个数
# 这里：从 0 秒到 T_MAX 秒，取 500 个点（点越多曲线越平滑）
# 注意：横轴要"固定长度"，如果写成 5*tau，横轴会跟着 τ 一起伸缩，
#       那样无论怎么改 R 和 C，曲线形状都一模一样，看不出差别
t = np.linspace(0, T_MAX, 500)

# 第四步：算每个时刻的电容电压
# 充电公式：u(t) = U × (1 - e^(-t/τ))   电压从 0 慢慢升到 U
# np.exp(x) 就是 e 的 x 次方
u_charge = U * (1 - np.exp(-t / tau))

# 放电公式：u(t) = U × e^(-t/τ)         电压从 U 慢慢降到 0
u_discharge = U * np.exp(-t / tau)

# 第五步：开始画图
# plt.figure 就是"铺开一张画布"，figsize 是画布尺寸（宽, 高）
plt.figure(figsize=(10, 6))

# plt.plot(x轴数据, y轴数据, 线条样式, 标签)
# "b-" 表示蓝色实线（b=blue），"r--" 表示红色虚线（r=red）
plt.plot(t, u_charge, "b-", linewidth=2, label="充电曲线")
plt.plot(t, u_discharge, "r--", linewidth=2, label="放电曲线")

# 第六步：给图加上说明文字（没有这些的图是不合格的）
plt.title("RC 电路充放电曲线", fontsize=16)      # 标题
plt.xlabel("时间 t（秒）", fontsize=12)          # 横轴说明
plt.ylabel("电容电压 u（伏特）", fontsize=12)     # 纵轴说明
plt.grid(True, linestyle=":", alpha=0.7)         # 显示网格线，方便读数
plt.legend(fontsize=12)                          # 显示右上角那个小图例

# 第七步：在图里标出 τ 这个关键点
# plt.axvline 画一条竖线，plt.text 在指定位置写字
plt.axvline(x=tau, color="green", linestyle=":", linewidth=2)
plt.axhline(y=0.632 * U, color="green", linestyle=":", linewidth=1.5)
plt.text(tau * 2.1, U * 0.55, f"1τ = {tau:.2f} s 时\n电压达到 63.2%（{0.632*U:.2f} V）",
         color="green", fontsize=11)

# 第八步：把图保存成图片文件（会保存在这个 py 文件旁边）
plt.savefig("RC充放电曲线.png", dpi=150, bbox_inches="tight")
print("图片已保存为：RC充放电曲线.png（就在本文件旁边，双击就能看）")

# 第九步：把图显示在屏幕上
plt.show()
