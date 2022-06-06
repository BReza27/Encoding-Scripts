from EoEfunc import denoise
import lvsfunc as lvf
from lvsfunc.kernels import Spline64
from numpy.matrixlib.defmatrix import matrix
from placebo import deband
import vapoursynth as vs
import EoEfunc as eoe
import vardefunc as vdf
import kagefunc as kgf
from vsutil import get_depth, get_y


# region setup

DEFAULT_PATH = r"Episode 01.mkv"
DEFAULT_CACHE = 80
DEFAULT_THREADS = 32

env = eoe.setup_env(globals())
core: vs._CoreProxy = env["core"]
src_path: str = env["src_path"]
GPU: int = env["GPU"]
debug: bool = env["debug"]

# endregion setup



src = core.lsmas.LWLibavSource(src_path)
src = eoe.fmt.set(src, "16")
rescale = eoe.rs.rescale(src,  mask_detail=True, width=1432, height=806, kernel="bicubic", b=0, c=0.5, rescale_threshold=0.02)

#denoise
sloc = [0, 0, 0.35, 2, 0.4, 8, 1, 12]
prefilter = core.neo_dfttest.DFTTest(rescale, sigma=8, ssx=sloc, ssy=sloc)
bm3d = eoe.dn.BM3D(prefilter, [0.5, 0], 1)
knlm = core.knlm.KNLMeansCL(bm3d, 2, 3, 2, .3, channels="uv")

#deband
placebo = vdf.placebo.deband(knlm, 16, 2, 1, 8, chroma=False,)
deband_mask = lvf.mask.detail_mask(knlm, 0, 2)
deband = core.std.MaskedMerge(placebo, knlm, deband_mask)

#AA
sraa = lvf.aa.upscaled_sraa(knlm, rfactor=1.8, alpha=0.4,)
clamp = lvf.aa.clamp_aa(knlm, lvf.aa.eedi3(knlm,), sraa, strength=2)
aa_mask = core.tcanny.TCannyCL(deband, 0, mode=1, op=2)
aa = core.std.MaskedMerge(deband, clamp, aa_mask)
aa = core.cas.CAS(aa, 0.2, )

#grain
grain = kgf.adaptive_grain(aa, .175, luma_scaling=15)


final = grain
final = eoe.fmt.set(final, "10")
final.set_output()

if debug:
    debug_output = eoe.debug_output
    debug_output(src, "src",)
    debug_output(rescale, "rs")
    debug_output(bm3d, "bm3d")
