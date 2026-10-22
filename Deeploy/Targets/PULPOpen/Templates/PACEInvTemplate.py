# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.Targets.PULPOpen.Templates.PACETemplate import PACENodeTemplate

referenceTemplate = PACENodeTemplate("""
// PACE Reciprocal (Name: ${nodeName}, Op: ${nodeOp})
// The polynomial approximates 1/x on the mantissa in [1, 2); the exponent is handled by frexpf / ldexpf
static const float ${nodeName}_bp[] = {${breakpoints}};
static const float ${nodeName}_c[][${pace_degree + 1}] = {${coeffs}};

extern float frexpf(float, int *);
extern float ldexpf(float, int);

#ifdef GVSOC_SIMULATION
// GVSoC cannot execute the PACE instruction: evaluate the same piecewise polynomial in C
for (int ${nodeName}_i = 0; ${nodeName}_i < ${size}; ${nodeName}_i++) {
    float ${nodeName}_x = ${data_in}[${nodeName}_i];

    // x = mant * 2^exp with mant in [1, 2)
    int ${nodeName}_exp;
    float ${nodeName}_mant = 2.0f * frexpf(${nodeName}_x, &${nodeName}_exp);
    ${nodeName}_exp -= 1;

    int ${nodeName}_seg = 0;
    while (${nodeName}_seg < ${pace_parts} - 1 &&
           ${nodeName}_mant >= ${nodeName}_bp[${nodeName}_seg + 1]) {
        ${nodeName}_seg++;
    }
    const float *${nodeName}_k = ${nodeName}_c[${nodeName}_seg];
    // Horner, highest power first
    float ${nodeName}_poly = ${nodeName}_k[0];
    for (int ${nodeName}_d = 1; ${nodeName}_d <= ${pace_degree}; ${nodeName}_d++) {
        ${nodeName}_poly = ${nodeName}_poly * ${nodeName}_mant + ${nodeName}_k[${nodeName}_d];
    }

    // 1/x = (1/mant) * 2^-exp
    ${data_out}[${nodeName}_i] = ldexpf(${nodeName}_poly, -${nodeName}_exp);
}
#else
// Per-core PACE CSR: extend=0, enable=1, degree
uint32_t ${nodeName}_pace_cfg = (0 << 4) | (1 << 3) | ${pace_degree};
asm volatile("csrw 0xba0, %0" ::"r"(${nodeName}_pace_cfg));
for (int ${nodeName}_i = 0; ${nodeName}_i < ${size}; ${nodeName}_i++) {
    float ${nodeName}_x = ${data_in}[${nodeName}_i];

    // x = mant * 2^exp with mant in [1, 2)
    int ${nodeName}_exp;
    float ${nodeName}_mant = 2.0f * frexpf(${nodeName}_x, &${nodeName}_exp);
    ${nodeName}_exp -= 1;

    float ${nodeName}_poly;
    asm volatile (
        "fmv.s ft1, %1\\n\\t"
        "fmv.w.x ft0, zero\\n\\t"
        ".word 0x601000d3\\n\\t"
        "fmv.s %0, ft1"
        : "=f"(${nodeName}_poly)
        : "f"(${nodeName}_mant)
        : "ft0", "ft1"
    );

    // 1/x = (1/mant) * 2^-exp
    ${data_out}[${nodeName}_i] = ldexpf(${nodeName}_poly, -${nodeName}_exp);
}
#endif
""")
