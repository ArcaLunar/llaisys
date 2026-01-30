import llaisys.models
from llaisys import LIB_LLAISYS

model = llaisys.models.Qwen2(model_path="./data")
# LIB_LLAISYS.llaisysQwen2ModelWeights(model._backend)


output = model.generate([151646, 151644, 15191, 525, 498, 30, 151645, 151648, 198], max_new_tokens=1)
print(f"{output=}")

"""
Answer:
[151646, 151644, 15191, 525, 498, 30, 151645, 151648, 198, 91786]
"""
