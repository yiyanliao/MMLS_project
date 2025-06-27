import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.signal import find_peaks, correlate
import pandas as pd

# 修改后的振荡器模型（包含参数传递）
def oscil_trial(t, z, params):
    s, h, ms, mh, p, mp = z
    as0 = 0.1
    ah0 = 0.1
    ap0 = 0.1
    a_s = 30.5
    ah = 183
    kh = 326
    ks = 185
    dm = 0.3
    dh = 3.8
    ds = 1
    n1 = 3
    n2 = 4.8
    n3 = 4.8
    n4 = 4.8
    b = 3.7
    
    # 解包待探索参数
    dp, ap_val, kph_val, kp_val = params
    
    dmsdt = as0 + a_s * ((h**n1)/(kh**n1 + h**n1) + kph_val**n4/(kp_val**n4 + p**n4))/2 - dm * ms
    dmhdt = ah0 + ah * (ks**n2)/(ks**n2 + s**n2) - dm * mh
    dsdt = b * ms - ds * s
    dhdt = b * mh - dh * h
    dpdt = b * mp - dp * p
    dmpdt = ap0 + ap_val * (kp_val**n1)/(kp_val**n3 + h**n3) - dm * mp
    
    return [dsdt, dhdt, dmsdt, dmhdt, dpdt, dmpdt]

# 检测振荡的函数
def detect_oscillation(signal, min_peaks=10, autocorr_threshold=0.5):
    """检测信号是否振荡"""
    # 方法1: 峰值计数
    peaks, _ = find_peaks(signal, height=np.max(signal)*0.2, prominence=1,distance=10)
    
    # 方法2: 自相关分析
    autocorr = correlate(signal, signal, mode='same')
    autocorr = autocorr / np.max(autocorr)  # 归一化
    center = len(autocorr) // 2
    side_peaks = autocorr[center+10:]  # 避开中心点
    
    # 判断标准
    has_peaks = len(peaks) >= min_peaks
    has_autocorr_osc = np.max(side_peaks) > autocorr_threshold
    has_large_amplitude = np.max(signal) - np.min(signal) > 1
    has_reaonable_values = (np.min(signal) < 10000)
    
    return has_peaks and has_autocorr_osc and has_large_amplitude and has_reaonable_values

# 参数扫描函数
def parameter_sweep(param_ranges, n_trials=100, sim_time=300):
    """
    随机扫描参数空间并检测振荡
    
    参数:
    param_ranges : 每个参数的取值范围 (dp, ap, kph, kp)
    n_trials     : 随机采样数量
    sim_time     : 模拟时长
    
    返回:
    results : 包含参数组合和振荡结果的DataFrame
    """
    results = []
    
    for i in range(n_trials):
        # 随机采样参数
        params = [
            np.exp(np.random.uniform(*param_ranges['dp'])),
            np.exp(np.random.uniform(*param_ranges['ap'])),
            np.exp(np.random.uniform(*param_ranges['kph'])),
            np.exp(np.random.uniform(*param_ranges['kp']))
        ]
        
        # 运行模拟
        z0 = [0, 0, 0, 0, 0, 0]
        t_eval = np.linspace(0, sim_time, 1000)
        sol = integrate.solve_ivp(
            lambda t, z: oscil_trial(t, z, params),
            [0, sim_time],
            z0,
            t_eval=t_eval,
            method='RK45'
        )
        
        # 提取H蛋白浓度 (忽略前1/3瞬态)
        h_signal = sol.y[0]
        steady_start = len(h_signal) // 3
        h_steady = h_signal[steady_start:]
        
        # 检测振荡
        oscillates = detect_oscillation(h_steady)
        
        # 存储结果
        results.append({
            'dp': params[0],
            'ap': params[1],
            'kph': params[2],
            'kp': params[3],
            'oscillates': oscillates,
            'max_h': np.max(h_steady),
            'min_h': np.min(h_steady),
            'amp_h': np.max(h_steady) - np.min(h_steady)
        })
        
        print(f"Trial {i+1}/{n_trials}: params={params} -> {'OSCILLATES' if oscillates else 'no oscillation'}")
    
    return pd.DataFrame(results)

# 定义参数扫描范围 (根据文献/经验猜测)
param_ranges = {
    'dp': (np.log(0.01), np.log(10)),    # 蛋白质降解率
    'ap': (0, np.log(1000)),     # P蛋白最大生成率
    'kph': (0, np.log(1000)),     # P抑制H的阈值
    'kp': (0, np.log(1000))      # P蛋白的Michaelis常数
}

# 运行参数扫描
results_df = parameter_sweep(param_ranges, n_trials=1000, sim_time=300)

# 保存结果
results_df.to_csv('oscillation_results.csv', index=False)

# 可视化结果
plt.figure(figsize=(12, 10))

# 1. 振荡/非振荡点在参数空间的分布
plt.subplot(221, projection='3d')
osc_df = results_df[results_df['oscillates']]
non_osc_df = results_df[~results_df['oscillates']]
plt.scatter(osc_df['kph'], osc_df['kp'], osc_df['ap'], c='green', label='Oscillates')
plt.scatter(non_osc_df['kph'], non_osc_df['kp'], non_osc_df['ap'], c='red', alpha=0.3, label='No Osc')
plt.xlabel('kph')
plt.ylabel('kp')
#plt.zlabel('ap')
plt.title('Parameter Space (ap-kph-kp)')
plt.legend()

# 2. 振荡参数组合的统计
plt.subplot(222)
osc_ratio = results_df['oscillates'].mean()
plt.bar(['Oscillate', 'No Osc'], [osc_ratio, 1-osc_ratio], color=['green', 'red'])
plt.ylabel('Fraction')
plt.title(f'Oscillation Ratio: {osc_ratio:.2f}')

# 3. 振幅分布
plt.subplot(223)
plt.hist(results_df[results_df['oscillates']]['amp_h'], bins=20, color='green')
plt.xlabel('Oscillation Amplitude (H)')
plt.ylabel('Count')
plt.title('Amplitude Distribution (Osc Cases)')

# 4. 示例振荡曲线
plt.subplot(224)
if not osc_df.empty:
    example_params = osc_df.iloc[0][['dp', 'ap', 'kph', 'kp']].values
    z0 = [0, 0, 0, 0, 0, 0]
    t_eval = np.linspace(0, 100, 1000)
    sol = integrate.solve_ivp(
        lambda t, z: oscil_trial(t, z, example_params),
        [0, 100],
        z0,
        t_eval=t_eval
    )
    plt.plot(t_eval, sol.y[0], 'b-')  # H蛋白
    plt.xlabel('Time')
    plt.ylabel('H Concentration')
    plt.title(f'Example Osc: kph={example_params[2]:.1f}, kp={example_params[3]:.1f}')
else:
    plt.text(0.5, 0.5, 'No oscillations found', ha='center')

plt.tight_layout()
plt.savefig('oscillation_analysis.png', dpi=150)
plt.show()

# 输出关键参数范围
if not osc_df.empty:
    print("\n振荡参数范围总结:")
    print(f"kph: [{osc_df['kph'].min():.2f}, {osc_df['kph'].max():.2f}]")
    print(f"kp:  [{osc_df['kp'].min():.2f}, {osc_df['kp'].max():.2f}]")
    print(f"ap:  [{osc_df['ap'].min():.2f}, {osc_df['ap'].max():.2f}]")
    print(f"dp:  [{osc_df['dp'].min():.2f}, {osc_df['dp'].max():.2f}]")