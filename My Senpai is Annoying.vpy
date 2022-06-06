import vapoursynth as vs
import EoEfunc as eoe
import vardefunc as vdf
import lvsfunc as lvf
import adptvgrnMod 
from vsutil import depth, plane
import vsutil
import havsfunc as haf
import rezafunc as rzf


# region setup

DEFAULT_PATH = r"Episode 03.mkv"
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



sloc = [0, 0, 0.35, .5, 0.4, .75, .75, 1, 1, 4]
prefilter = core.dfttest.DFTTest(src, sigma=2, ssx=sloc, ssy=sloc, planes=[0], )
cmd = eoe.dn.CMDegrain(prefilter, 2, 150, 130, 3, contrasharp=False)
bm3d = eoe.dn.BM3D(prefilter, [.3, .25], 1, "fast", CPU=True, ref=cmd)

denoise_mask = rzf.retinex_edgemask(src, 2, 3)
denoised = core.std.MaskedMerge(bm3d, src, denoise_mask)


rgb = core.resize.Bicubic(denoised, format=vs.RGBS, matrix_in=1, transfer_in=1, primaries_in=1, )
w2x = core.w2xnvk.Waifu2x(rgb, 3, 1, 2, ).resize.Bicubic(format=vs.YUV420P16, matrix=1, transfer=1, primaries=1, range=0)z
w2x = lvf.rfs(denoised, w2x, (4628, 4747))

dehalo = haf.FineDehalo(w2x, 1.8, 1.8, brightstr=.7, darkstr=0)
edgeclean = haf.EdgeCleaner(dehalo, 9,)


deband = core.average.Mean([
    vdf.deband.dumb3kdb(dehalo, 24, 30, [4,2]),
    vdf.deband.dumb3kdb(dehalo, 30, 20, [4,2]),
    vdf.placebo.deband(dehalo, 18, 2, 2, 4)
    ])

deband_mask = rzf.retinex_edgemask(dehalo, .5, 2)
debanded = core.std.MaskedMerge(deband, dehalo, deband_mask)


grain = adptvgrnMod.adptvgrnMod(debanded, .2, size=1.15, sharp=55, luma_scaling=9, static=True, grain_chroma=False)


op = core.lsmas.LWLibavSource(r'G:\Senpai\Vol 1 enc\filteredop.mkv')
op = depth(op, 16)
op = op[1:]
final = grain[:2471] + op + grain[4627:]

final = final
final = depth(final, 10).std.Limiter()
final.set_output()


if debug:
    debug_output = eoe.debug_output
    debug_output(src, "src")
    debug_output(final, "spliced")
    debug_output(w2x, "w2x")
    debug_output(denoised, "bm3d")
    debug_output(denigger2x, "w2x")
