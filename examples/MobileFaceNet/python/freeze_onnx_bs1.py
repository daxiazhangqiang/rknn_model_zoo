import onnx

SRC = '../model/w600k_mbf.onnx'
DST = '../model/w600k_mbf_bs1.onnx'

model = onnx.load(SRC)
graph = model.graph

# 把输入里所有动态维度(None/dim_param 或 -1)固定为 1
for inp in graph.input:
    for d in inp.type.tensor_type.shape.dim:
        if d.dim_value <= 0:          # 动态 batch
            d.ClearField('dim_param')  # 清除 'None' 参数名
            d.dim_value = 1            # 固定为 1

onnx.checker.check_model(model)
onnx.save(model, DST)

# 验证输出
m = onnx.load(DST)
for i in m.graph.input:
    dims = [d.dim_value for d in i.type.tensor_type.shape.dim]
    print(i.name, dims)
print('saved:', DST)
