import numpy as np
import onnx
from onnx import helper, TensorProto

PACE_BREAKPOINTS = [-5.0, -4.375, -3.75, -3.125, -2.5, -1.875, -1.25, -0.625, 0.0, 0.625, 1.25, 1.875, 2.5, 3.125, 3.75, 4.375, 5.0]
PACE_COEFFS = [[-3.571150697789783e-05, -0.0003485977651045944, -0.0008510155066555983], 
                [-0.0006343019585271725, -0.005456013605735592, -0.011747929228868841], 
                [-0.005412657553297883, -0.04052980473391607, -0.0761459295733912], 
                [-0.02352149308463695, -0.1520374647131766, -0.24799331931098276], 
                [-0.05032479430699543, -0.2869112019252661, -0.4179561747227519], 
                [-0.022160431118592165, -0.19154432695687149, -0.3377334889585528], 
                [0.1453206976404744, 0.21345508193758367, -0.09111214330882786], 
                [0.35676959285071175, 0.4861795996290082, -0.0007841679949993788], 
                [0.3567695928507104, 0.5138204003709917, -0.0007841679949994569], 
                [0.14532069764047437, 0.7865449180624164, -0.09111214330882869], 
                [-0.022160431118592516, 1.191544326956873, -0.33773348895855415], 
                [-0.05032479430699713, 1.2869112019252735, -0.4179561747227602], 
                [-0.023521493084628232, 1.1520374647131308, -0.24799331931092414], 
                [-0.005412657553292026, 1.0405298047338778, -0.07614592957332614], 
                [-0.0006343019585314355, 1.0054560136057717, -0.011747929228942576], 
                [-3.5711506973567494e-05, 1.0003485977650648, -0.0008510155065609171]]

# Input and output
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 16])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 16])

# Single GeLU node with PACE attributes
gelu_node = helper.make_node(
    'Gelu',
    inputs=['X'],
    outputs=['Y'],
    name='pace_gelu_0',
    domain='com.microsoft',
    pace_degree=2,
    pace_parts=16,
    pace_breakpoints=PACE_BREAKPOINTS,
    pace_coeffs=[c for seg in PACE_COEFFS for c in seg]
)

graph = helper.make_graph([gelu_node], 'gelu_graph', [X], [Y])
model = helper.make_model(graph, opset_imports=[
    helper.make_opsetid('', 17),
    helper.make_opsetid('com.microsoft', 1)
])

onnx.save(model, 'gelu_single.onnx')
print("Saved gelu_single.onnx")

# Generate test inputs and expected outputs
inputs = np.random.randn(1, 16).astype(np.float32)
np.savez('inputs.npz', X=inputs)
print("inputs:", inputs)