# === 必要なライブラリのインポート ===
import numpy as np
import matplotlib.pyplot as plt
from qlbm.components.lqlga import LQLGA, LQGLAInitialConditions
from qlbm.lattice import LQLGALattice
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import os

# === シミュレーションパラメータの設定 ===
nx = 9                          # 格子数
velocity_model = "D1Q3"         # モデル
shots = 1024                    # ショット回数
output_dir = "C:\\Users\\..."   # 画像保存先ディレクトリ

# === 格子と境界条件の定義 ===
def definition_condition():
    config = {
            "lattice": {"dim": {"x": nx}, "velocities": velocity_model},
            "geometry": [],
            }
    lattice = LQLGALattice(config)
    sim = LQLGA(lattice=lattice)
    print("✅ 設計図の作成に成功しました。\n")

    return lattice, sim

# === 初期状態の準備 ===
def initial_state_original(lattice):
    # ((x,), (v=0, v=+1, v=-1)): それぞれの格子点 x に速度 v の粒子が存在する場合にTrueを指定
    grid_data = [
        ((0,), (True, False, False)),
        ((1,), (True, False, False)),
        ((2,), (True, False, False)),
        ((3,), (False, True, False)),
        ((4,), (True, False, False)),
        ((5,), (False, False, True)),
        ((6,), (True, False, False)),
        ((7,), (True, False, False)),
        ((8,), (True, False, False))
    ]

    # クラスをインスタンス化して、初期状態の回路を生成
    initial_conditions = LQGLAInitialConditions(lattice, grid_data)
    initial_state_circuit = initial_conditions.circuit
    print(f"✅ 初期状態の量子回路を作成しました。")
    return initial_state_circuit

# === 回路の構築 ===
def make_circuit(num_steps, sim, initial_state_circuit):
    full_circuit = initial_state_circuit.copy()
    for _ in range(num_steps):
        full_circuit = full_circuit.compose(sim.circuit)
    full_circuit.measure_all()

    return full_circuit

# === シミュレーターで実行 ===
def run_simulation(circuit, shots):
    simulator = AerSimulator()
    transpiled_circuit = transpile(circuit, simulator)
    result = simulator.run(transpiled_circuit, shots=shots).result()
    counts = result.get_counts()
    print(f"✅ t={num_steps} のシミュレーションが完了しました。")

    return counts

# === 結果の解析 ===
def analyze_results(counts, lattice, shots):
    """
    測定結果から物理量（密度、速度）を正しく計算する関数。
    """
    total_particles_per_site = np.zeros(nx)
    total_momentum_per_site = np.zeros(nx)
    d1q3_velocities = np.array([0, 1, -1]) # v0, v+1, v-1

    # Qiskitのビット文字列はリトルエンディアン (右端がqubit 0) なので注意
    for bitstring, count in counts.items():
        # サイトごとに粒子数と運動量を計算
        for i in range(nx):
            particles_at_site = 0
            momentum_at_site = 0
            # 速度チャネルごとに粒子が存在するかチェック
            for j in range(3): # j=0 (v0), j=1 (v+1), j=2 (v-1)
                qubit_index = lattice.velocity_index_tuple((i,), j)
                
                # ビット文字列の該当箇所が '1' かどうかチェック
                if bitstring[-(qubit_index + 1)] == '1':
                    particles_at_site += 1
                    momentum_at_site += d1q3_velocities[j]
            
            # サイトごとの合計に、この測定結果の分を加算
            total_particles_per_site[i] += particles_at_site * count
            total_momentum_per_site[i] += momentum_at_site * count

    # ショット数で割って期待値を計算
    densities = total_particles_per_site / shots
    avg_momentum = total_momentum_per_site / shots
    
    # 速度は (運動量の期待値) / (密度の期待値)
    # ゼロ割を避けるための処理
    velocities = np.zeros(nx)
    np.divide(avg_momentum, densities, out=velocities, where=densities!=0)

    return densities, velocities

# === グラフの作成と保存 (密度と速度を両方プロット) ===
def visualize(rho, ux, num_steps):
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # 密度の棒グラフ (左側のy軸)
    ax1.bar(range(nx), rho, color='skyblue', alpha=0.8, label='Density (ρ)')
    ax1.set_xlabel('Grid Point (x)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(0.8, 2.6 if np.max(rho) > 0 else 1)

    # 速度の折れ線グラフ (右側のy軸)
    ax2 = ax1.twinx()
    ax2.plot(range(nx), ux, color='mediumpurple', marker='o', linestyle='--', label='Average Velocity (u)')
    ax2.set_ylabel('Average Velocity', fontsize=12, color='mediumpurple')
    ax2.tick_params(axis='y', labelcolor='mediumpurple')
    ax2.set_ylim(-1.1, 1.1)
    ax2.axhline(0, color='black', linewidth=0.8, linestyle=':')

    # グラフのタイトルと凡例
    plt.title(f'1D LQLGA Density and Velocity (t={num_steps})', fontsize=14)
    fig.tight_layout()

    # 凡例をまとめる
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.xticks(range(nx))
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 👇 4. パスとファイル名を結合して、保存先のフルパスを作成
    base_filename = f'lqlga_1d_velocity_t{num_steps}.png'
    output_filepath = os.path.join(output_dir, base_filename)

    # 👇 5. フルパスを指定して保存
    plt.savefig(output_filepath, dpi=300)
    plt.close()
    print(f"✅ 速度分布グラフを '{output_filepath}' として保存しました。\n")

lattice, sim = definition_condition()           # 格子とシミュレーションの定義
initial_state_circuit = initial_state_original(lattice)  # 初期状態の回路を取得

for num_steps in [0,1,2,3,4,5]:  # シミュレーションステップ数のループ
    print(f"--- t={num_steps} のシミュレーションを開始 ---")
    
    full_circuit = make_circuit(num_steps, sim, initial_state_circuit)  # 回路の構築
    counts = run_simulation(full_circuit, shots)    # シミュレーターで実行
    densities, velocities = analyze_results(counts, lattice, shots) # 結果の解析
    visualize(densities, velocities, num_steps)     # 可視化

    # 量子回路の素描
    if num_steps == 3:
        full_circuit.draw(output="mpl", filename="lqlga_circuit.png")

    # 検証: 計算された全密度の和（全粒子数）を表示
    total_density = np.sum(densities)
