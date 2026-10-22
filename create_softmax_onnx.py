import numpy as np
import onnx
from onnx import helper, TensorProto

# --- PACE coefficients for Exp over [-11, 0] (verbatim from create_exp_onnx.py) ---
EXP_BREAKPOINTS = [-11.0, -10.3125, -9.625, -8.9375, -8.25, -7.5625, -6.875, -6.1875, -5.5, -4.8125, -4.125, -3.4375, -2.75, -2.0625, -1.375, -0.6875, 0.0]
EXP_COEFFS = [[1.1878298601383288e-05, 0.00027699453542966813, 0.0016264268788982027],
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

# --- PACE coefficients for Reciprocal over [1, 2) (verbatim from create_inv_onnx.py) ---
INV_BREAKPOINTS = [1.0, 1.0625, 1.125, 1.1875, 1.25, 1.3125, 1.375, 1.4375, 1.5, 1.5625, 1.625, 1.6875, 1.75, 1.8125, 1.875, 1.9375, 2.0]
INV_COEFFS = [[0.9125504140439082, -2.822976269906448, 2.910415197638362],
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

Q, K = 1, 16

X       = helper.make_tensor_value_info('X',       TensorProto.FLOAT, [Q, K])
exp_out = helper.make_tensor_value_info('exp_out', TensorProto.FLOAT, [Q, K])
sum_out = helper.make_tensor_value_info('sum_out', TensorProto.FLOAT, [Q, 1])
inv_out = helper.make_tensor_value_info('inv_out', TensorProto.FLOAT, [Q, 1])
Y       = helper.make_tensor_value_info('Y',       TensorProto.FLOAT, [Q, K])

# Node 1 - Exp (PACE)
exp_node = helper.make_node(
    'Exp',
    inputs=['X'],
    outputs=['exp_out'],
    name='pace_exp_0',
    pace_degree=2,
    pace_parts=16,
    pace_breakpoints=EXP_BREAKPOINTS,
    pace_coeffs=[c for seg in EXP_COEFFS for c in seg]
)

# Node 2 - ReduceSum (axis 1, keepdims). opset-11 style: axes as attribute, not input.
reduce_node = helper.make_node(
    'ReduceSum',
    inputs=['exp_out'],
    outputs=['sum_out'],
    name='reduce_sum_0',
    axes=[1],
    keepdims=1
)

# Node 3 - Reciprocal (PACE)
inv_node = helper.make_node(
    'Reciprocal',
    inputs=['sum_out'],
    outputs=['inv_out'],
    name='pace_inv_0',
    pace_degree=2,
    pace_parts=16,
    pace_breakpoints=INV_BREAKPOINTS,
    pace_coeffs=[c for seg in INV_COEFFS for c in seg]
)

# Node 4 - Mul (exp_out scaled by the scalar inv_out)
mul_node = helper.make_node(
    'Mul',
    inputs=['exp_out', 'inv_out'],
    outputs=['Y'],
    name='mul_0'
)

graph = helper.make_graph(
    [exp_node, reduce_node, inv_node, mul_node],
    'softmax_pace_graph',
    [X],
    [Y],
    value_info=[exp_out, sum_out, inv_out]
)
model = helper.make_model(graph, opset_imports=[
    helper.make_opsetid('', 13)
])

onnx.save(model, 'softmax_single.onnx')
print("Saved softmax_single.onnx")

# --- test data ---
inputs = np.random.uniform(-5.0, 0.0, size=(Q, K)).astype(np.float32)
e = np.exp(inputs)
outputs = (e / np.sum(e, axis=1, keepdims=True)).astype(np.float32)
np.savez('inputs.npz', X=inputs)
np.savez('outputs.npz', Y=outputs)
print("inputs:", inputs)
print("outputs:", outputs)
