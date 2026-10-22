import numpy as np
import onnx
from onnx import helper, TensorProto

PACE_BREAKPOINTS = [-11.0, -10.3125, -9.625, -8.9375, -8.25, -7.5625, -6.875, -6.1875, -5.5, -4.8125, -4.125, -3.4375, -2.75, -2.0625, -1.375, -0.6875, 0.0]
PACE_COEFFS = [[1.1878298601383288e-05, 0.00027699453542966813, 0.0016264268788982027],
                [2.3622817503480114e-05, 0.000518388037411699, 0.002866978827546879],
                [4.697958230624982e-05, 0.000966340788112275, 0.00501510373381246],
                [9.3430055637776e-05, 0.0017933318072028373, 0.0086966487920889],
                [0.00018580785243194523, 0.0033109803632830896, 0.01493122907011236],
                [0.0003695230382741377, 0.006076576531887075, 0.025341990979424057],
                [0.0007348843120895287, 0.011074249506623146, 0.04243767281565048],
                [0.001461491967260755, 0.02001422348634242, 0.06994682809085104],
                [0.0029065238367846653, 0.03580656589630037, 0.11311507719175902],
                [0.005780312860447866, 0.06326192907192532, 0.1787315151485525],
                [0.01149552477148016, 0.11000502218260226, 0.27438817679031624],
                [0.02286158084555576, 0.18733643579414688, 0.4060865797132909],
                [0.04546568244144234, 0.31004767592483595, 0.5729521808055487],
                [0.09041930625142254, 0.4922768843729071, 0.758273862020093],
                [0.17982026231584297, 0.7317566246773527, 0.9199317912783332],
                [0.35761529345763104, 0.9635507896066691, 0.9980325513722559]]

# Input and output
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 16])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 16])

# Single Exp node with PACE attributes
exp_node = helper.make_node(
    'Exp',
    inputs=['X'],
    outputs=['Y'],
    name='pace_exp_0',
    pace_degree=2,
    pace_parts=16,
    pace_breakpoints=PACE_BREAKPOINTS,
    pace_coeffs=[c for seg in PACE_COEFFS for c in seg]
)

graph = helper.make_graph([exp_node], 'exp_graph', [X], [Y])
model = helper.make_model(graph, opset_imports=[
    helper.make_opsetid('', 13)
])

onnx.save(model, 'exp_single.onnx')
print("Saved exp_single.onnx")

# Generate test inputs (domain [-11, 0]) and expected outputs (true exp)
inputs = np.random.uniform(-11.0, 0.0, size=(1, 16)).astype(np.float32)
outputs = np.exp(inputs).astype(np.float32)
np.savez('inputs.npz', X=inputs)
np.savez('outputs.npz', Y=outputs)
print("inputs:", inputs)
print("outputs:", outputs)
