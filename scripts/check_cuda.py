import importlib.util

if not importlib.util.find_spec("torch"):
    print("torch missing")
else:
    import torch
    print({"torch": torch.__version__, "cuda": torch.cuda.is_available()})
