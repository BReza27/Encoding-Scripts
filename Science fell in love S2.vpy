import adptvgrnMod
import EoEfunc as eoe
import havsfunc as haf
import lvsfunc as lvf
import rezafunc as rzf
import vapoursynth as vs
from adptvgrnMod import adptvgrnMod
from debandshit import dumb3kdb, placebo_deband
from finedehalo import fine_dehalo
from jvsfunc import ccd
from vsutil import depth, get_y, join, plane, split

# region setup

DEFAULT_PATH = r"done/[SubsPlease] Rikei ga Koi ni Ochita no de Shoumei shitemita S2 - 01 (1080p) [6DB3465C].mkv"
DEFAULT_CACHE = 80
DEFAULT_THREADS = 32

env = eoe.setup_env(globals())
core: vs._CoreProxy = env["core"]
src_path: str = env["src_path"]
GPU: int = env["GPU"]
debug: bool = env["debug"]

# endregion setup

src = core.lsmas.LWLibavSource(src_path)
src = depth(src, 16)

#plugin = core.std.LoadPlugin(r'C:\Users\Brandon\AppData\Roaming\VapourSynth\plugins64\addnoise.dll')

sloc = [0, 0, 0.35, 1, 0.4, 1.5, .8, 2, 1, 8]
prefilter = core.dfttest.DFTTest(src, sigma=1, ssx=sloc, ssy=sloc, planes=[0], )
degrain = eoe.dn.CMDegrain(prefilter, 2, 170, refine=3, contrasharp=False, thSADC=150)
bm3d = core.bm3dcpu.BM3D(depth(prefilter, 32,), depth(degrain, 32), [.45], radius=3)
bm3d = core.bm3d.VAggregate(bm3d, 3)

rgb1 = core.resize.Bicubic(bm3d, format=vs.RGBS, matrix_in=1, transfer_in=1, primaries_in=1, range=0).std.Limiter()
ccd = ccd(rgb1, 4).resize.Bicubic(format=vs.YUV420P16, matrix=1, transfer=1, primaries=1, range=0)


denoised = join([
    plane(bm3d, 0),
    plane(ccd, 1),
    plane(ccd, 2)
])


f3k = placebo_deband(denoised, 18, 2, 3, 0)

luma_deband = core.average.Mean([
    f3k,
    placebo_deband(denoised, 10, [2.5, 0, 0], 2, 4)
])

chroma_deband = core.average.Mean([
    f3k,
    placebo_deband(denoised, 14, [0, 2, 2], 2, 4)
])

deband = core.std.ShufflePlanes([luma_deband, chroma_deband], [0, 1, 2], vs.YUV)


retinex = rzf.retinex_edgemask(denoised, 2, 2)
debanded = core.std.MaskedMerge(deband, denoised, retinex)


aa = lvf.aa.taa(debanded, lvf.aa.nnedi3(True))


dering = haf.HQDeringmod(aa, darkthr=0, sharp=0, nrmode=1)
dehalo = fine_dehalo(aa, dering, 1.7, 1.7, brightstr=.8, darkstr=0, useMtEdge=True, )

grain = adptvgrnMod(dehalo, luma_scaling=9.5, grainer=lambda x: core.noise.Add(x, type=2,  var=1.75, uvar=1.25, constant=True))



final = grain
final = depth(final, 10).std.Limiter()
final.set_output()
