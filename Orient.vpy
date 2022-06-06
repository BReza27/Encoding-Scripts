import vapoursynth as vs
import EoEfunc as eoe
import lvsfunc as lvf
import adptvgrnMod 
from vsutil import depth, plane, join
import rezafunc as rzf
import havsfunc as haf
from debandshit import placebo_deband, dumb3kdb

# region setup

DEFAULT_PATH = r"[SubsPlease] Orient - 08 (1080p) [790426ED].mkv"
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

sloc = [0, 0, 0.35, 2, 0.4, 2, .8, 1, 1, 50]

prefilter = core.dfttest.DFTTest(src, sigma=1, ssx=sloc, ssy=sloc, planes=[0], )
pfgrey = core.resize.Point(prefilter, format=vs.GRAYS)
degrain = eoe.dn.CMDegrain(prefilter, 2, 170, refine=3, contrasharp=False, thSADC=15)
degraingrey = core.resize.Point(degrain, format=vs.GRAYS)
bm3d = core.bm3dcpu.BM3D(pfgrey, degraingrey, 0.35, radius=4).bm3d.VAggregate(4).resize.Bicubic(format=vs.YUV420P16)
knlm = core.knlm.KNLMeansCL(degrain, 4, 4, 5, .7, "UV", device_type="GPU")


denoised = vsutil.join([
    plane(bm3d, 0),
    plane(knlm, 1),
    plane(knlm, 2)
])

rescale = eoe.rs.rescale(denoised, 1600, 900, "bicubic", b=0, c=0.5, mask_detail=True, rescale_threshold=0.0225)

placebo = placebo_deband(rescale, 8, 3, 4, 0)
retinex = rzf.retinex_edgemask(prefilter, 1, 3)
debanded = core.std.MaskedMerge(placebo, rescale, retinex)

deband = core.average.Mean([
    dumb3kdb(debanded, 14, 45, 0),
    placebo_deband(debanded, 18, 2, 2, 0)
])
debanded = core.std.MaskedMerge(deband, rescale, retinex)

edgeclean = haf.EdgeCleaner(debanded, 11, False)
kirsch = rzf.kirsch(debanded)


aafun = lvf.aa.nnedi3(True)
aa = lvf.aa.taa(edgeclean, aafun)
aa = core.std.MaskedMerge(edgeclean, aa, kirsch)

grain = adptvgrnMod.adptvgrnMod(aa, .25, size=1.2, sharp=65, temporal_average=75, luma_scaling=8, grain_chroma=False)

final = grain
final = depth(final, 10).std.Limiter()
final.set_output()
