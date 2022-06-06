import EoEfunc as eoe
import lvsfunc as lvf
import rezafunc as rzf
import vapoursynth as vs
import vardefunc as vdf
from vsutil import plane, split, get_y, depth
from adptvgrnMod import adptvgrnMod

# region setup

DEFAULT_PATH = r"Episode 01.m2ts"
DEFAULT_CACHE = 80
DEFAULT_THREADS = 32

env = eoe.setup_env(globals())
core: vs.Core = env["core"]
src_path: str = env["src_path"]
GPU: int = env["GPU"]
debug: bool = env["debug"]



# endregion setup


src = core.lsmas.LWLibavSource(src_path)
src = depth(src, 16)

cmd = eoe.dn.CMDegrain(src, 2, 150, 130, 3, contrasharp=False)
bm3d = eoe.dn.BM3D(cmd, [.25,], 1, "fast", CPU=True)

denoise_mask = rzf.retinex_edgemask(src, 1.5, 3)
denoised = core.std.MaskedMerge(bm3d, src, denoise_mask)


deband = core.average.Mean([
    vdf.placebo.deband(denoised, 20, 2, 2, 8),
    vdf.placebo.deband(denoised, 14, 2, 2, 8),
    vdf.deband.dumb3kdb(denoised, 16, 24, 0),
    denoised
])
deband_mask = rzf.retinex_edgemask(denoised, 2, 3)
debanded = core.std.MaskedMerge(deband, denoised, deband_mask)

grain = adptvgrnMod(debanded, .25, size=1.25, sharp=65, grain_chroma=False, luma_scaling=9, static=True)


final = grain
final = depth(final, 10).std.Limiter()
final.set_output()

