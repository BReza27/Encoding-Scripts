import adptvgrnMod
import EoEfunc as eoe
import havsfunc as haf
import lvsfunc as lvf
import rezafunc as rzf
import vapoursynth as vs
from ccd import ccd
from debandshit import dumb3kdb, placebo_deband
from vsutil import depth, join, plane, split, get_y
from finedehalo import fine_dehalo
from adptvgrnMod import adptvgrnMod
import kagefunc as kgf
import jvsfunc as jvs


# region setup

DEFAULT_PATH = r"[CR] The Rising of the Shield Hero Season 2 - 1 [1080p].mkv"
DEFAULT_CACHE = 80
DEFAULT_THREADS = 32

env = eoe.setup_env(globals())
core: vs._CoreProxy = env["core"]
src_path: str = env["src_path"]
debug: bool = env["debug"]

# endregion setup

src = core.lsmas.LWLibavSource(src_path)
src = depth(src, 16)

sloc = [0, 0, 0.35, 1, 0.4, 1.5, .8, 2, 1, 16]
dft = core.dfttest.DFTTest(src, sigma=1, ssx=sloc, ssy=sloc, planes=[0], )



degrain = eoe.dn.CMDegrain(dft, 2, 120, refine=3, contrasharp=False, thSADC=80,)

bm3d = core.bm3dcpu.BM3D(get_y(depth(dft, 32,)), get_y(depth(degrain, 32)), [.45], radius=3)
bm3d = core.bm3d.VAggregate(bm3d, 3)


rgb = core.resize.Bicubic(src, format=vs.RGBS, matrix_in=1, transfer_in=1, range=0, primaries_in=1).std.Limiter()

ccd2 = jvs.ccd(rgb, 4).resize.Bicubic(format=vs.YUV420P16, matrix=1, transfer=1, primaries=1, range=0)

ccd2 = join([
    plane(bm3d, 0),
    plane(ccd2, 1),
    plane(ccd2, 2)
])


denoised = ccd2

f3k = placebo_deband(denoised, 14, 2, 3, 0)

luma_deband = core.average.Mean([
    f3k,
    placebo_deband(denoised, 10, [2.5, 0, 0], 2, 4)
])

chroma_deband = core.average.Mean([
    f3k,
    placebo_deband(denoised, 16, [0, 2, 2], 2, 4)
])

deband = join([
    plane(luma_deband, 0),
    plane(chroma_deband, 1),
    plane(chroma_deband, 2)
])

retinex = rzf.retinex_edgemask(denoised, 2, 3)
debanded = core.std.MaskedMerge(deband, denoised, retinex)



sraa = lvf.aa.upscaled_sraa(debanded, 1.4,  )
denig1 = haf.HQDeringmod(sraa,  nrmode=1, sharp=0, darkthr=0)
denig2 = haf.DeHalo_alpha(sraa, darkstr=0, ss=1.25)
denig3 = core.std.Expr([sraa, denig1, denig2], "x y - abs x z - abs < y z ?")
denig4 = fine_dehalo(sraa, ref=denig3, rx=1.7, brightstr=.5, darkstr=0)
aa_mask = core.tcanny.TCanny(get_y(debanded), 1.5, 1.5, op=2, mode=1, scale=1.25).std.Binarize(5000).std.Maximum()
aad = core.std.MaskedMerge(debanded, denig4, aa_mask)

grain = adptvgrnMod(aad, [.2, 0.05],  size=1.1, sharp=55, static=True, luma_scaling=10)


final = grain
final = depth(final, 10).std.Limiter(s)
final.set_output()

if debug:
    debug_output = eoe.debug_output
    debug_output(src,  )
    debug_output(dft, "dft")
    debug_output(test, "fft3d")
    #debug_output(ccd2, "ccd2")
