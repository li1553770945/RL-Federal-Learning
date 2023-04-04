def get_comp_energy(action: int, performance: int, co_cpu: int, co_memory: int) -> int:
    ENERGY_BASE = 10
    ENERGY_TIMES = [10, 9, 7, 4, 1]  # 对于每个action，能量消耗乘以的倍率
    PERFORMANCE_ENERGY_TIMES = [0.5, 1, 2]  # 性能从好到差的CPU，设备能量消耗倍率
    return ENERGY_BASE * ENERGY_TIMES[action] * PERFORMANCE_ENERGY_TIMES[performance]


def get_comm_energy(network_bandwidth: float) -> float:
    if network_bandwidth < 10:
        network_bandwidth = 10
    if network_bandwidth > 70:
        network_bandwidth = 70

    return (network_bandwidth - 10) * 10 / 6
