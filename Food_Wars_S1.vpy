from havsfunc import FineDehalo
import lvsfunc as lvf
from lvsfunc.aa import clamp_aa
from lvsfunc.mask import adaptive_mask
from muvsfunc import TCannyHelper
import vapoursynth as vs
import EoEfunc as eoe
import havsfunc as haf
from vardefunc.deband import dumb3kdb
import vsutil
import vardefunc as vdf
import kagefunc as kgf

# region setup

DEFAULT_PATH = r"Test 03.mkv"
DEFAULT_CACHE = 80
DEFAULT_THREADS = 32

env = eoe.setup_env(globals())
core: vs.Core = env["core"]
src_path: str = env["src_path"]
GPU: int = env["GPU"]
debug: bool = env["debug"]



# endregion setup


src = core.lsmas.LWLibavSource(src_path)
src = eoe.fmt.set(src, "16")


sloc = [0, 0, 0.35, 1, 0.4, 2, 1, 32]
prefilter = core.neo_dfttest.DFTTest(src, sigma=4, ssx=sloc, ssy=sloc, planes=[0])

cmd = eoe.dn.CMDegrain(prefilter, 2, 110, refine=3, contrasharp=True)
chroma = core.knlm.KNLMeansCL(cmd, d=4, a=4, s=3, h=.75, device_type="gpu", channels="uv")
sobel_mask = core.std.Sobel(prefilter)
sobel_mask = core.std.Maximum(sobel_mask)
denoised = core.std.MaskedMerge(cmd, chroma, sobel_mask)

deband = core.average.Mean([
    vdf.deband.dumb3kdb(denoised, 12, 24, 0, 4, use_neo=True),
    vdf.deband.dumb3kdb(denoised, 20, 50, 0, 4, use_neo=True),
    vdf.placebo.deband(denoised, 18, 2, 2, 4), 
    vdf.placebo.deband(denoised, 12, 1, 1, 4,),
])

deband_mask = core.tcanny.TCanny(denoised, 3, 3, mode=0, op=3)
deband_mask = lvf.mask.detail_mask(deband_mask, 1, 2)
debanded = core.std.MaskedMerge(deband, denoised, deband_mask)


eedi = lvf.aa.nneedi3_clamp(debanded, .85)
aa = core.std.MaskedMerge(debanded, eedi, sobel_mask)

grain = kgf.adaptive_grain(aa, .2, 5)

final = grain
final = eoe.fmt.set(final, "10")
final.set_output()


if debug:
    debug_output = eoe.debug_output
    debug_output(src, "source", )
    debug_output(cmd, "cmd")
    debug_output(chroma, "KNLM")
    debug_output(denoised, "masked")
    debug_output(denoise_mask, "tcanny")
    debug_output(deband, "debanded average")
    debug_output(sraa, "sraa")
    debug_output(clamp, "clamped")
    debug_output(aad, "aad")
    debug_output(grain, "grain")

