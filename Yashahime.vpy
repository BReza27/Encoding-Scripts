from lvsfunc.mask import detail_mask
import vapoursynth as vs
import EoEfunc as eoe
import vardefunc as vdf
import lvsfunc as lvf
import adptvgrnMod 
import havsfunc as haf
# region setup

DEFAULT_PATH = r"00001.m2ts"
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

rescale = eoe.rs.rescale(src, 1440, 810, "lanczos", mask_detail=True, taps=4, rescale_threshold=.25)


sloc = [0, 0, 0.35, 2, 0.4, 4, 1, 8]
prefilter = core.dfttest.DFTTest(rescale, sigma=1, ssx=sloc, ssy=sloc, planes=[0], )
cmd = eoe.denoise.CMDegrain(rescale, 2, 110, 100, 3, )
bm3d = eoe.denoise.BM3D(rescale, [.3, .25], 1, "fast", ref=cmd)


f3kdb = vdf.deband.dumb3kdb(bm3d, 16, 30, 0, 4, True)
mask = lvf.mask.detail_mask(bm3d, 0, 1, )
debanded = core.std.MaskedMerge(f3kdb, bm3d, mask)

aa = lvf.aa.nneedi3_clamp(debanded, .75)

grain = adptvgrnMod.adptvgrnMod(aa, .2, size=1.2, sharp=65, static=True, luma_scaling=7, grain_chroma=False)

final = grain
final = depth(final, 10).std.Limiter()
final.set_output()

