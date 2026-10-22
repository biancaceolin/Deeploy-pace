# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import struct
from typing import List, Sequence, Tuple

from mako.template import Template

from Deeploy.DeeployTypes import GlobalDefinition, NetworkContext, NodeTemplate, OperatorRepresentation

# Memory-mapped PACE parameter SRAM, shared by all cluster cores
PACE_PARAM_BASE = 0x10201000


def arrange_params(breakpoints: Sequence[float], coeffs: Sequence[Sequence[float]], degree: int) -> List[int]:
    """Pack breakpoints and per-segment polynomial coefficients into PACE SRAM word order.

    Parameters
    ----------
    breakpoints : Sequence[float]
        pace_parts + 1 segment boundaries
    coeffs : Sequence[Sequence[float]]
        pace_parts rows of degree + 1 coefficients, highest power first (numpy.polyfit order)
    degree : int
        Polynomial degree

    Returns
    -------
    List[int]
        Packed 32-bit words, in the order they are written to PACE_PARAM_BASE

    """

    def float_to_u32(f):
        return struct.unpack('<I', struct.pack('<f', float(f)))[0]

    words = []
    pace_parts = len(coeffs)
    assert pace_parts & (pace_parts - 1) == 0, "PACE breakpoint search tree needs a power-of-two number of parts"
    assert len(breakpoints) == pace_parts + 1

    # Coefficients: column-major, highest power first (numpy.polyfit order)
    for d in range(degree + 1):
        for seg in range(pace_parts):
            words.append(float_to_u32(coeffs[seg][d]))

    # Inner breakpoints in level order of the binary search tree; the outer bounds are not stored
    level = pace_parts // 2
    while level >= 1:
        for idx in range(level, pace_parts, 2 * level):
            words.append(float_to_u32(breakpoints[idx]))
        level //= 2

    # Fixed trailing words, independent of the approximated function
    words.append(float_to_u32(5e-7))
    words.append(0x7FC00000)  # quiet NaN

    return words


class PACEInitCode(GlobalDefinition):
    """Code emitted into InitNetwork(); the empty `definition` keeps it out of the global scope."""

    def __init__(self, name: str, initCode: str):
        super().__init__(name, "")
        self.initCode = initCode


_initTemplate = Template("""
// PACE initialization (Name: ${nodeName})
{
#define PACE_PARAM_BASE ${hex(paramBase)}
#define ${nodeName}_PARAMS_LEN ${len(params)}
    static const uint32_t ${nodeName}_pace_params[${nodeName}_PARAMS_LEN] = {
        ${', '.join(f'0x{word:08x}' for word in params)}
    };
#ifndef GVSOC_SIMULATION
    // Shared by all cores, so written once here; not modelled by GVSoC
    volatile uint32_t *${nodeName}_pace_sram = (volatile uint32_t *) PACE_PARAM_BASE;
    for (uint32_t ${nodeName}_i = 0; ${nodeName}_i < ${nodeName}_PARAMS_LEN; ${nodeName}_i++) {
        ${nodeName}_pace_sram[${nodeName}_i] = ${nodeName}_pace_params[${nodeName}_i];
    }
#else
    (void) ${nodeName}_pace_params;
#endif
#undef ${nodeName}_PARAMS_LEN
#undef PACE_PARAM_BASE
}
""")


class PACENodeTemplate(NodeTemplate):
    """NodeTemplate that also emits the PACE parameter SRAM initialization into InitNetwork()."""

    def alignToContext(
            self, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, OperatorRepresentation, List[str]]:

        if operatorRepresentation.get('pace_breakpoints_list'):
            nodeName = operatorRepresentation['nodeName']
            initName = f"{nodeName}_pace_init"
            # Hoisting may run more than once per node
            if not ctxt.is_global(initName):
                degree = operatorRepresentation['pace_degree']
                params = arrange_params(operatorRepresentation['pace_breakpoints_list'],
                                        operatorRepresentation['pace_coeffs_list'], degree)
                initCode = _initTemplate.render(nodeName = nodeName, params = params, paramBase = PACE_PARAM_BASE)
                ctxt.add(PACEInitCode(initName, initCode), 'global')

        return ctxt, operatorRepresentation, []


referenceTemplate = PACENodeTemplate("""
// PACE (Name: ${nodeName}, Op: ${nodeOp})
static const float ${nodeName}_bp[] = {${breakpoints}};
static const float ${nodeName}_c[][${pace_degree + 1}] = {${coeffs}};

#ifdef GVSOC_SIMULATION
// GVSoC cannot execute pace.s: evaluate the same piecewise polynomial in C
for (int ${nodeName}_i = 0; ${nodeName}_i < ${size}; ${nodeName}_i++) {
    float ${nodeName}_x = ${data_in}[${nodeName}_i];
    int ${nodeName}_seg = 0;
    while (${nodeName}_seg < ${pace_parts} - 1 &&
           ${nodeName}_x >= ${nodeName}_bp[${nodeName}_seg + 1]) {
        ${nodeName}_seg++;
    }
    const float *${nodeName}_k = ${nodeName}_c[${nodeName}_seg];
    // Horner, highest power first
    float ${nodeName}_acc = ${nodeName}_k[0];
    for (int ${nodeName}_d = 1; ${nodeName}_d <= ${pace_degree}; ${nodeName}_d++) {
        ${nodeName}_acc = ${nodeName}_acc * ${nodeName}_x + ${nodeName}_k[${nodeName}_d];
    }
    ${data_out}[${nodeName}_i] = ${nodeName}_acc;
}
#else
// Per-core PACE CSR: extend=0, enable=1, degree
uint32_t ${nodeName}_pace_cfg = (0 << 4) | (1 << 3) | ${pace_degree};
asm volatile("csrw 0xba0, %0" ::"r"(${nodeName}_pace_cfg));
for (int ${nodeName}_i = 0; ${nodeName}_i < ${size}; ${nodeName}_i++) {
    float ${nodeName}_x = ${data_in}[${nodeName}_i];
    float ${nodeName}_result;
    asm volatile("fmv.s ft1, %1\\n\\t"
                 "fmv.w.x ft0, zero\\n\\t"
                 "pace.s ft1, ft1, ft0\\n\\t"
                 "fmv.s %0, ft1"
                 : "=f"(${nodeName}_result)
                 : "f"(${nodeName}_x)
                 : "ft0", "ft1");
    ${data_out}[${nodeName}_i] = ${nodeName}_result;
}
#endif
""")
