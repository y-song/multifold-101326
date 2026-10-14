TF_CUDA_LIB_DIRS=$(python - <<'PY'
from pathlib import Path
from importlib.metadata import distribution

root = Path(distribution("tensorflow").locate_file("nvidia"))
print(":".join(str(p) for p in sorted(root.glob("*/lib"))))
PY
)

export LD_LIBRARY_PATH="$TF_CUDA_LIB_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
