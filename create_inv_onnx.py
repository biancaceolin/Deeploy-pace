import numpy as np
import onnx
from onnx import helper, TensorProto

PACE_BREAKPOINTS = [1.0, 1.0625, 1.125, 1.1875, 1.25, 1.3125, 1.375, 1.4375, 1.5, 1.5625, 1.625, 1.6875, 1.75, 1.8125, 1.875, 1.9375, 2.0]
PACE_COEFFS = [[0.9125504140439082, -2.822976269906448, 2.910415197638362],
                [0.7648139865659549, -2.5093668565056615, 2.743967052251652],
                [0.6473246761466982, -2.2452641309808956, 2.5955340125979416],
                [0.5527207698315184, -2.020769202424671, 2.462340595233022],
                [0.47569026714307133, -1.8283394103476245, 2.3421537639400642],
                [0.41233475200331965, -1.6621459083184698, 2.233156547986682],
                [0.3597492617541384, -1.517626703356684, 2.13385542941502],
                [0.31573901994535597, -1.3911700209555904, 2.043011419020431],
                [0.27862484637649776, -1.2798863043801687, 1.959588039697423],
                [0.24710719528678138, -1.1814415837743193, 1.8827115676080823],
                [0.22016967405272905, -1.0939340361678338, 1.8116402879710898],
                [0.19700959872838372, -1.0158013990659862, 1.745740467361906],
                [0.17698735253543355, -0.945750728162117, 1.6844673905334364],
                [0.1595890080530456, -0.8827045419901712, 1.627350258463109],
                [0.14439842930288396, -0.8257591254285811, 1.5739800603612606],
                [0.1310762322972307, -0.7741519528777362, 1.5239997580046045]]

# Input and output
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 16])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 16])

# Single Reciprocal node with PACE attributes
recip_node = helper.make_node(
    'Reciprocal',
    inputs=['X'],
    outputs=['Y'],
    name='pace_reciprocal_0',
    pace_degree=2,
    pace_parts=16,
    pace_breakpoints=PACE_BREAKPOINTS,
    pace_coeffs=[c for seg in PACE_COEFFS for c in seg]
)

graph = helper.make_graph([recip_node], 'reciprocal_graph', [X], [Y])
model = helper.make_model(graph, opset_imports=[
    helper.make_opsetid('', 13)
])

onnx.save(model, 'reciprocal_single.onnx')
print("Saved reciprocal_single.onnx")

# Generate test inputs (mantissa domain [1, 2)) and expected outputs (true 1/x)
inputs = np.random.uniform(1.0, 2.0, size=(1, 16)).astype(np.float32)
outputs = (1.0 / inputs).astype(np.float32)
np.savez('inputs.npz', X=inputs)
np.savez('outputs.npz', Y=outputs)
print("inputs:", inputs)
print("outputs:", outputs)
