import tables 
import matplotlib.pyplot as plt
import numpy as np

# 加载数据
file = tables.open_file('ts.mat')
X = file.root.X[:]
Y = file.root.Y[:]
dh = file.root.dh[:]
ds = file.root.ds[:]

print("X min/max:", np.nanmin(X), np.nanmax(X))
print("Y min/max:", np.nanmin(Y), np.nanmax(Y))
print("dh min/max:", np.nanmin(dh), np.nanmax(dh))
print("ds min/max:", np.nanmin(ds), np.nanmax(ds))

s_line = file.root.s_line[:]
h_line = file.root.h_line[:]
s_line2 = file.root.s_line2[:]
h_line2 = file.root.h_line2[:]

file.close()

# 修复维度
s_line = s_line.transpose().copy()  # (1401, 5)
h_line = h_line.flatten().copy()     # (1401,)
s_line2 = s_line2.flatten().copy()   # (226,)
h_line2 = h_line2.transpose().copy() # (226, 4)

# 绘制蓝色曲线
for ii in range(s_line.shape[1]):
    y_val = s_line[:, ii]
    valid = ~np.isnan(y_val)
    plt.plot(h_line[valid], y_val[valid], 'b-', linewidth=2.5)

# 绘制红色曲线
for ii in range(h_line2.shape[1]):
    x_val = h_line2[:, ii]
    valid = ~np.isnan(x_val)
    plt.plot(x_val[valid], s_line2[valid], 'r-', linewidth=2.5)


plt.title('Sir2 Double expression/Hap overexpression')
# plt.title('Sir2 Double expression')
# plt.title('WT')
# plt.title('No Mutual Inhibition')
# plt.title('hap deletion')
# plt.title('sir2 deletion')
plt.xlabel('HAP')
plt.ylabel('Sir2')
plt.ylim([-20, 460])

# 矢量场
# 调整 quiver 参数
# 矢量场绘制（关键参数调整）
Q = plt.quiver(X, Y, 0.2*dh, ds, units='width', color='gray', alpha = 1)  

Q.set_zorder(0)

plt.tight_layout()
plt.savefig('ts.eps')
plt.show()