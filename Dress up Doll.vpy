import adptvgrnMod
import ccd
import EoEfunc as eoe
import havsfunc as haf
import lvsfunc as lvf
import rezafunc as rzf
import vapoursynth as vs
from debandshit import dumb3kdb, placebo, placebo_deband
from vsutil import depth, get_y, join, plane, split

# region setup

DEFAULT_PATH = r"[SubsPlease] Sono Bisque Doll wa Koi wo Suru - 10 (1080p) [B05ED11C] (1).mkv"
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


sloc = [0, 0, 0.35, 1, 0.4, 1.5, .8, 2, 1, 8]
prefilter = core.dfttest.DFTTest(src, sigma=1, ssx=sloc, ssy=sloc, planes=[0], )
degrain = eoe.dn.CMDegrain(src, 2, 170, refine=3, contrasharp=False, thSADC=150)
bm3d = core.bm3dcpu.BM3D(get_y(depth(src, 32),), get_y(depth(degrain, 32)), [.75], radius=3)
bm3d = core.bm3d.VAggregate(bm3d, 3)
ccd = ccd.ccd(src, 4 ,)

denoised = core.std.ShufflePlanes([bm3d, ccd], [0, 1, 2], vs.YUV)
denoised = core.resize.Bicubic(denoised, matrix=2, transfer=2, primaries=2, range=0)

f3k = dumb3kdb(denoised, 24, [30, 30, 30], 0)

luma_deband = core.average.Mean([
    f3k,
    placebo_deband(denoised, 10, [1.5, 0, 0], 2, 4)
])

chroma_deband = core.average.Mean([
    f3k,
    placebo_deband(denoised, 14, [0, 1.5, 1.5], 2, 4)
])

deband = join([
    plane(luma_deband, 0),
    plane(chroma_deband, 1),
    plane(chroma_deband, 2)
])

retinex = rzf.retinex_edgemask(denoised, 2, 3)
debanded = core.std.MaskedMerge(deband, denoised, retinex)


dehalo = haf.FineDehalo(deband, 2.2, 2.2, brightstr=1, darkstr=0)
dehalo_mask = haf.FineDehalo(deband, 2.2, 2.2, brightstr=1, darkstr=0, showmask=1)
edgeclean = haf.EdgeCleaner(dehalo, 7, False)
edgeclean = core.std.MaskedMerge(dehalo, edgeclean, dehalo_mask)
contra = haf.ContraSharpening(edgeclean, dehalo)

grain = adptvgrnMod.adptvgrnMod(contra, 0.15, size=1, sharp=50, static=True, luma_scaling=10, grain_chroma=False)

final = grain
final = depth(final, 10)
final.set_output()

