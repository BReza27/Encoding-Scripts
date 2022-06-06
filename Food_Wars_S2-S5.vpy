from EoEfunc import rescaling
from EoEfunc.denoise import BM3D
import vapoursynth as vs
import EoEfunc as eoe
import lvsfunc as lvf
import vardefunc as vdf
import kagefunc as kgf
import havsfunc as haf
import vsutil

# region setup

DEFAULT_PATH = r"Episode 01.mkv"
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
prefilter = core.neo_dfttest.DFTTest(src, sigma=2, ssx=sloc, ssy=sloc, planes=[0])

cmd = eoe.dn.CMDegrain(src, 2, 100, refine=3, contrasharp=True)
bm3d = eoe.dn.BM3D(cmd, [.65, .75, .75], 1, profile="fast")
denoised = bm3d

deband = core.average.Mean([
    vdf.deband.dumb3kdb(denoised, 20, 64, 0, 4, use_neo=True),
    vdf.placebo.deband(denoised, 18, 2, 2, 4), 
    vdf.placebo.deband(denoised, 12, 1, 1, 4,),
])
deband_mask = core.tcanny.TCanny(denoised, 3, 3, mode=0, op=3)
debanded = core.std.MaskedMerge(deband, denoised, deband_mask)


aa = lvf.aa.nneedi3_clamp(debanded, .9)


grain = kgf.adaptive_grain(aa, .3, 6)


final = grain
final = eoe.fmt.set(final, "10")
final.set_output()

