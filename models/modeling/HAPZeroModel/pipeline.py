from .HAPZeroNet import build_HAPZeroNet
_GZSL_META_ARCHITECTURES = {
    "Model": build_HAPZeroNet,
}
def build_gzsl_pipeline(cfg):
    meta_arch = _GZSL_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
