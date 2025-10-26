# === 1. 必要なライブラリのインポート ===
import numpy as np
import matplotlib.pyplot as plt
from qlbm.components.lqlga import LQLGA, LQGLAInitialConditions
from qlbm.lattice import LQLGALattice
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import os

# === 2. シミュレーションパラメータの設定 ===
nx = 9
velocity_model = "D1Q3"
shots = 1024
output_dir = "C:\\Users\\..."  # 画像保存先ディレクトリ

# === 3. 格子と境界条件の定義 ===
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

# === 初期状態の準備 (カスタム版) ===
def initial_state_modified(lattice):
    """
    指定された確率分布に従って初期状態を準備する。
    - x=3, 5以外: ρ=1, u=0 (決定論的)
    - x=3: ρ=1, <u>=0.01 (確率的)
    - x=5: ρ=1, <u>=-0.01 (確率的)
    """
    # シミュレーションに必要な全量子ビット数を持つ新しい量子回路を作成
    initial_circ = QuantumCircuit(lattice.circuit.num_qubits, name="Initial State")
    print("✅ 確率的な初期状態を含む量子回路を作成します...")

    # --- 1. 決定論的な初期状態 (x=3, 5 以外) ---
    v0_index = 0
    for i in range(nx):
        if i not in [3, 5]:
            qubit_index = lattice.velocity_index_tuple((i,), v0_index)
            initial_circ.x(qubit_index)
    print(f"   - x=[0,1,2,4,6,7,8] に決定論的な粒子 (ρ=1, u=0) を配置しました。")

    # --- 2. 確率的な初期状態 (x=3 と x=5) ---
    
    # 平均速度 <u> = p * (+1) + (1-p) * 0  より、<u>=0.01を実現する確率p
    prob = 0.01

    # --- サイト 3 (x=3): <u> = +0.01 の状態 ---
    # サイト3の各速度に対応する量子ビットインデックスを取得
    q_v0_at_3 = lattice.velocity_index_tuple((3,), 0)  # v=0 の粒子
    q_v1_at_3 = lattice.velocity_index_tuple((3,), 1)  # v=+1 の粒子
    q_vm1_at_3 = lattice.velocity_index_tuple((3,), 2) # v=-1 の粒子

    # Qiskitの慣例(リトルエンディアン)に従い、3つの量子ビットの状態は |q_vm1 q_v1 q_v0> と表現される。
    # 物理状態と基底状態の対応：
    # |001> : v=0に粒子 (状態A)
    # |010> : v=1に粒子 (状態B)
    # |100> : v=-1に粒子 (状態C)
    
    # 目的の量子状態: |ψ> = sqrt(1-p) |001> + sqrt(p) |010>
    # この状態は、測定すると確率(1-p)で|001>に、確率pで|010>になる。
    # 密度は常に1に保たれる。
    
    # 上記の量子状態を表現する8次元の状態ベクトルを作成
    state_vec_plus = np.zeros(8, dtype=complex)
    state_vec_plus[1] = np.sqrt(1 - prob)  # |001> の振幅
    state_vec_plus[2] = np.sqrt(prob)      # |010> の振幅
    
    # サイト3の量子ビット群 [q_v0, q_v1, q_vm1] を上記の状態に初期化
    initial_circ.initialize(state_vec_plus, [q_v0_at_3, q_v1_at_3, q_vm1_at_3])
    print(f"   - x=3 に確率的な状態 (ρ=1, <u>={prob}) を設定しました。")

    # --- サイト 5 (x=5): <u> = -0.01 の状態 ---
    q_v0_at_5 = lattice.velocity_index_tuple((5,), 0)
    q_v1_at_5 = lattice.velocity_index_tuple((5,), 1)
    q_vm1_at_5 = lattice.velocity_index_tuple((5,), 2)

    # 目的の量子状態: |ψ> = sqrt(1-p) |001> + sqrt(p) |100>
    state_vec_minus = np.zeros(8, dtype=complex)
    state_vec_minus[1] = np.sqrt(1 - prob)  # |001> (v=0) の振幅
    state_vec_minus[4] = np.sqrt(prob)      # |100> (v=-1) の振幅

    initial_circ.initialize(state_vec_minus, [q_v0_at_5, q_v1_at_5, q_vm1_at_5])
    print(f"   - x=5 に確率的な状態 (ρ=1, <u>={-prob}) を設定しました。")

    return initial_circ

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