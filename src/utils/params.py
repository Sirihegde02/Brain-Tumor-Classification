"""
Parameter counting and model analysis utilities (Keras 2/3 safe)
"""
import os, tempfile
from typing import Dict, Any, List, Tuple
import tensorflow as tf

# ---------- shape helpers ----------
def _safe_shape(obj, candidates):
    for attr in candidates:
        if hasattr(obj, attr):
            try:
                val = getattr(obj, attr)
                if val is not None:
                    return val
            except Exception:
                pass
    return None

def _fmt_shape(s):
    if s is None:
        return "—"
    try:
        from tensorflow.python.framework.tensor_shape import TensorShape
        if isinstance(s, TensorShape):
            s = tuple(s.as_list() if s.rank is not None else [])
    except Exception:
        pass
    try:
        return str(tuple(s))
    except Exception:
        return str(s)

# ---------- parameter counting ----------
def count_parameters(model: tf.keras.Model) -> Dict[str, int]:
    total_params = model.count_params()
    trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable_params = total_params - trainable_params
    return {"total": total_params, "trainable": trainable_params, "non_trainable": non_trainable_params}

# ---------- robust layer info (no .input_shape access) ----------
def get_layer_info_safe(model: tf.keras.Model) -> List[Dict[str, Any]]:
    info = []
    for i, layer in enumerate(model.layers):
        out_shape = _safe_shape(layer, ["output_shape"])
        try:
            params = int(layer.count_params())
        except Exception:
            params = 0
        info.append({
            "index": i,
            "name": layer.name,
            "type": layer.__class__.__name__,
            "output_shape": out_shape,
            "params": params,
        })
    return info

# ---------- rough FLOPs (best-effort) ----------
def estimate_flops(model: tf.keras.Model, input_shape: Tuple[int, ...] = None) -> int:
    try:
        if input_shape is None:
            m_in = _safe_shape(model, ["input_shape", "batch_input_shape"])
            if isinstance(m_in, (list, tuple)) and len(m_in) >= 2:
                input_shape = tuple(m_in[1:])
            else:
                input_shape = (224, 224, 3)
        flops = 0
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                in_shape = _safe_shape(layer, ["input_shape", "batch_input_shape"])
                out_shape = _safe_shape(layer, ["output_shape"])
                try:
                    kh, kw = layer.kernel_size
                    filters = int(layer.filters)
                    ih, iw, ic = tuple(in_shape[-3:]) if in_shape is not None else (None, None, None)
                    oh, ow = tuple(out_shape[1:3]) if out_shape is not None else (None, None)
                    if None not in (kh, kw, filters, ih, iw, ic, oh, ow):
                        flops += kh * kw * ic * filters * oh * ow
                except Exception:
                    pass
            elif isinstance(layer, tf.keras.layers.Dense):
                in_shape = _safe_shape(layer, ["input_shape", "batch_input_shape"])
                try:
                    in_units = int(in_shape[-1]) if in_shape is not None else None
                    out_units = int(layer.units)
                    if in_units is not None:
                        flops += in_units * out_units
                except Exception:
                    pass
        return int(flops)
    except Exception:
        return 0

# ---------- size ----------
def get_model_size_mb(model: tf.keras.Model) -> float:
    try:
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            path = tmp.name
        model.save(path)
        size_bytes = os.path.getsize(path)
        os.unlink(path)
        return size_bytes / (1024 * 1024)
    except Exception:
        return 0.0

# ---------- analysis ----------
def analyze_model_complexity(model: tf.keras.Model) -> Dict[str, Any]:
    params = count_parameters(model)
    layer_info = get_layer_info_safe(model)
    layer_types: Dict[str, int] = {}
    for layer in layer_info:
        t = layer["type"]
        layer_types[t] = layer_types.get(t, 0) + 1
    model_size_mb = get_model_size_mb(model)
    flops = estimate_flops(model)
    return {
        "parameters": params,
        "model_size_mb": model_size_mb,
        "estimated_flops": flops,
        "num_layers": len(layer_info),
        "layer_types": layer_types,
        "layer_details": layer_info,
    }

# ---------- comparison ----------
def compare_models(models: Dict[str, tf.keras.Model]) -> Dict[str, Dict[str, Any]]:
    return {name: analyze_model_complexity(m) for name, m in models.items()}

# ---------- pretty print ----------
def print_model_summary(model: tf.keras.Model, show_layers: bool = True, show_parameters: bool = True) -> None:
    print("=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)
    print(f"Model: {model.name}")
    m_in  = _safe_shape(model, ["input_shape", "inputs", "batch_input_shape"])
    m_out = _safe_shape(model, ["output_shape", "outputs"])
    print(f"Input shape: {_fmt_shape(m_in)}")
    print(f"Output shape: {_fmt_shape(m_out)}\n")

    if show_parameters:
        params = count_parameters(model)
        print("PARAMETER COUNTS:")
        print(f"  Total parameters: {params['total']:,}")
        print(f"  Trainable parameters: {params['trainable']:,}")
        print(f"  Non-trainable parameters: {params['non_trainable']:,}\n")

    if show_layers:
        layer_info = get_layer_info_safe(model)
        print("LAYER DETAILS:")
        print("-" * 80)
        print(f"{'Index':<6} {'Name':<20} {'Type':<15} {'Output Shape':<20} {'Params':<10}")
        print("-" * 80)
        for layer in layer_info:
            out_str = _fmt_shape(layer["output_shape"])
            out_str = (out_str[:18] + "...") if len(out_str) > 20 else out_str
            print(f"{layer['index']:<6} {layer['name']:<20} {layer['type']:<15} {out_str:<20} {layer['params']:<10,}")
        print("-" * 80 + "\n")

    try:
        comp = analyze_model_complexity(model)
        print("COMPLEXITY ANALYSIS:")
        print(f"  Model size: {comp['model_size_mb']:.2f} MB")
        print(f"  Estimated FLOPs: {comp['estimated_flops']:,}")
        print(f"  Number of layers: {comp['num_layers']}\n")
        print("LAYER TYPE DISTRIBUTION:")
        for lt, c in comp["layer_types"].items():
            print(f"  {lt}: {c}")
    except Exception as e:
        print("Complexity analysis skipped (non-fatal):", str(e))
    print("=" * 80)
