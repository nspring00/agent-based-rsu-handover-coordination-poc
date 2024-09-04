from poc import units as units, VanetTraceLoader as vanetLoader
from poc.base import RsuConfig

RSU_RANGE = 70
RSU_CAPACITY_T4 = 65 * units.TERA
RSU_CAPACITY_T4_HALF = 32.5 * units.TERA
RSU_CAPACITY_T4_QUARTER = 16.25 * units.TERA

CRETEIL_4_RSU_FULL_CAPA_CONFIG = [
    RsuConfig((35, 120), RSU_RANGE, RSU_CAPACITY_T4),  # blue
    RsuConfig((116, 150), RSU_RANGE, RSU_CAPACITY_T4),  # yellow
    RsuConfig((165, 50), RSU_RANGE, RSU_CAPACITY_T4),  # green
    RsuConfig((72, 50), RSU_RANGE, RSU_CAPACITY_T4),  # red
]

CRETEIL_4_RSU_HALF_CAPA_CONFIG = [
    RsuConfig((35, 120), RSU_RANGE, RSU_CAPACITY_T4_HALF),  # blue
    RsuConfig((116, 150), RSU_RANGE, RSU_CAPACITY_T4_HALF),  # yellow
    RsuConfig((165, 50), RSU_RANGE, RSU_CAPACITY_T4_HALF),  # green
    RsuConfig((72, 50), RSU_RANGE, RSU_CAPACITY_T4_HALF),  # red
]

CRETEIL_9_RSU_POSITIONS = [
    (35, 120),  # blue
    (116, 150),  # yellow
    (165, 50),  # green
    (72, 50),  # red
    (97, 98),  # olive
    (120, 30),  # yellow
    (30, 70),  # purple
    (70, 155),  # brown
    (160, 130),  # cyan
]

CRETEIL_9_RSU_FULL_CAPA_CONFIG = [RsuConfig(pos, RSU_RANGE, RSU_CAPACITY_T4) for pos in CRETEIL_9_RSU_POSITIONS]
CRETEIL_9_RSU_HALF_CAPA_CONFIG = [RsuConfig(pos, RSU_RANGE, RSU_CAPACITY_T4_HALF) for pos in CRETEIL_9_RSU_POSITIONS]
CRETEIL_9_RSU_QUARTER_CAPA_CONFIG = [RsuConfig(pos, RSU_RANGE, RSU_CAPACITY_T4_QUARTER) for pos in
                                     CRETEIL_9_RSU_POSITIONS]

CRETEIL_3_FAIL_RSU_FULL_CAPA_CONFIG = [
    RsuConfig((35, 120), RSU_RANGE, RSU_CAPACITY_T4),  # blue
    RsuConfig((116, 150), RSU_RANGE, RSU_CAPACITY_T4),  # yellow
    RsuConfig((165, 50), RSU_RANGE, RSU_CAPACITY_T4),  # green
]

CRETEIL_3_FAIL_RSU_HALF_CAPA_CONFIG = [
    RsuConfig((35, 120), RSU_RANGE, RSU_CAPACITY_T4_HALF),  # blue
    RsuConfig((116, 150), RSU_RANGE, RSU_CAPACITY_T4_HALF),  # yellow
    RsuConfig((165, 50), RSU_RANGE, RSU_CAPACITY_T4_HALF),  # green
]

SIMULATION_CONFIGS = {
    "creteil-morning": {
        "traces": lambda: vanetLoader.get_traces(morning=True, eval=True),
        "4-full": CRETEIL_4_RSU_FULL_CAPA_CONFIG,
        "4-half": CRETEIL_4_RSU_HALF_CAPA_CONFIG,
        "9-full": CRETEIL_9_RSU_FULL_CAPA_CONFIG,
        "9-half": CRETEIL_9_RSU_HALF_CAPA_CONFIG,
        "9-quarter": CRETEIL_9_RSU_QUARTER_CAPA_CONFIG,
        "3-fail-full": CRETEIL_3_FAIL_RSU_FULL_CAPA_CONFIG,
        "3-fail-half": CRETEIL_3_FAIL_RSU_HALF_CAPA_CONFIG,
    },
    "creteil-evening": {
        "traces": lambda: vanetLoader.get_traces(morning=False, eval=True),
        "4-full": CRETEIL_4_RSU_FULL_CAPA_CONFIG,
        "4-half": CRETEIL_4_RSU_HALF_CAPA_CONFIG,
        "9-full": CRETEIL_9_RSU_FULL_CAPA_CONFIG,
        "9-half": CRETEIL_9_RSU_HALF_CAPA_CONFIG,
        "9-quarter": CRETEIL_9_RSU_QUARTER_CAPA_CONFIG,
        "3-fail-full": CRETEIL_3_FAIL_RSU_FULL_CAPA_CONFIG,
        "3-fail-half": CRETEIL_3_FAIL_RSU_HALF_CAPA_CONFIG,
    }
}
