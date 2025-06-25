from controller import Supervisor
from banco_dados_robo import check_parameters, save_test_result
import math
import time
import random
import pandas as pd

def analisar_movimento(position_log, limite_angulo=0.45, limite_percentual=1):
    df = pd.DataFrame(position_log, columns=["Type", "Data"])
    df_joints = df[df['Type'].str.contains('pitch_joint|yaw_joint')]
    df_joints['Angle'] = df_joints['Data'].astype(float)
    df_joints['angle_diff'] = df_joints['Angle'].diff().abs()
    movimentos_criticos = df_joints[df_joints['angle_diff'] > limite_angulo]
    percentual_nao_critico = 1 - (len(movimentos_criticos) / len(df_joints))
    return percentual_nao_critico >= limite_percentual

def calculate_pitch_angle(i):
    return 2 * alpha_v * math.sin((2 * math.pi * k) / M) * \
           math.sin(phi + (4 * math.pi * k / M) * (i - 1 + d_0 / d))

def calculate_yaw_angle(i):
    return 2 * alpha_h * math.sin((2 * math.pi * k) / M) * \
           math.sin(phi + (4 * math.pi * k / M) * (i - 0.5 + d_0 / d) + delta_phi_vh)

def reset_robot():
    global pitch_joints, yaw_joints, pitch_sensors, yaw_sensors
    pitch_joints = [robot.getDevice(f"pitch_joint_{i+1}") for i in range(6)]
    yaw_joints = [robot.getDevice(f"yaw_joint_{i+1}") for i in range(5)]
    for joint in pitch_joints + yaw_joints:
        joint.setPosition(float('inf'))

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

pitch_joints = []
yaw_joints = []
pitch_sensors = []
yaw_sensors = []

gps = robot.getDevice("gps")
gps.enable(timestep)

numero_simulacoes = 1
duracao_simulacao = 10
M = 12
phi = 0.0
d = 0.063
d_0 = 0.063 

for i in range(6):
    pitch_joint = robot.getDevice(f"pitch_joint_{i+1}")
    pitch_joint.setPosition(float('inf'))
    pitch_joints.append(pitch_joint)
    pitch_sensor = pitch_joint.getPositionSensor()
    pitch_sensor.enable(timestep)
    pitch_sensors.append(pitch_sensor)
for i in range(5):
    yaw_joint = robot.getDevice(f"yaw_joint_{i+1}")
    yaw_joint.setPosition(float('inf'))
    yaw_joints.append(yaw_joint)
    yaw_sensor = yaw_joint.getPositionSensor()
    yaw_sensor.enable(timestep)
    yaw_sensors.append(yaw_sensor)

for sim in range(numero_simulacoes):
    print(f"Iniciando simulação {sim + 1}/{numero_simulacoes}...")
    while True:
        alpha_v = random.uniform(0, 2.0944)
        alpha_h = random.uniform(0, 2.0944)
        k = random.choice([1, 2, 3])
        delta_phi_vh = random.uniform(0, 1.5708)
        parametros_str = f"alpha_v: {alpha_v}, alpha_h: {alpha_h}, k: {k}, delta_phi_vh: {delta_phi_vh}"
        if not check_parameters(parametros_str):
            break
        else:
            print("Parâmetros já testados.")

    pos_inicial = gps.getValues()
    simulation_time = 0
    position_log = []
    start_time = robot.getTime()

    while robot.step(timestep) != -1:
        phi += 0.1
        for i, (joint, sensor) in enumerate(zip(pitch_joints, pitch_sensors), start=1):
            pitch_angle = calculate_pitch_angle(i)
            joint.setPosition(pitch_angle)
            position_log.append((f"pitch_joint_{i}", sensor.getValue()))
        for i, (joint, sensor) in enumerate(zip(yaw_joints, yaw_sensors), start=1):
            yaw_angle = calculate_yaw_angle(i)
            joint.setPosition(yaw_angle)
            position_log.append((f"yaw_joint_{i}", sensor.getValue()))

        gps_position = gps.getValues()
        position_log.append(("GPS", gps_position))
        simulation_time += timestep / 1000.0
        if (robot.getTime() - start_time) >= duracao_simulacao:
            break

    pos_final = gps.getValues()
    vetor_deslocamento = [final - inicial for final, inicial in zip(pos_final, pos_inicial)]
    movimento_aceitavel = analisar_movimento(position_log)
    resultado = "aceitável" if movimento_aceitavel else "não aceitável"
    print(f"Posição Inicial: {tuple(pos_inicial)}")
    print(f"Posição Final: {tuple(pos_final)}")
    print(f"Vetor de Deslocamento: {tuple(vetor_deslocamento)}")
    print(f"Resultado: {resultado}")
    save_test_result(parametros_str, resultado, str(tuple(pos_inicial)), str(tuple(pos_final)), str(tuple(vetor_deslocamento)))

    robot.simulationReset()
    reset_robot()
    time.sleep(0.5)
print("Simulações concluídas!")