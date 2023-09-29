import vapoursynth as vs
import vodesfunc as vof
import vsaa
import vsmlrt
from vstools import depth, plane, get_y
from vsdenoise import mvtools, SADMode, SearchMode
from adptvgrnMod import adptvgrnMod
from vodesfunc import out

core = vs.core

source = core.lsmas.LWLibavSource(r"op1.mpg")

rgb = core.resize.Bicubic(source, format=vs.RGBS, matrix_in=1, primaries_in=1, transfer_in=1).std.Limiter()
dpir = vsmlrt.DPIR(rgb, 25, 2, model=3, backend=vsmlrt.BackendV2().TRT(force_fp16=True, workspace=100000, use_cublas=True, use_cuda_graph=True, use_cudnn=True)).resize.Bicubic(format=vs.YUV420P16, matrix=1, primaries=1, transfer=1)

dft = core.dfttest.DFTTest(dpir, tbsize=1, sigma=1,)
mv = mvtools.MVTools(dpir, tr=3, refine=2, sad_mode=SADMode.DCT.ADAPTIVE_SATD_DCT, block_size=32, overlap=16, )
ref = mv.degrain(thSAD=40, thSCD=60,)
bm3d = core.bm3dcuda_rtc.BM3Dv2(plane(depth(ref, 32), 0), ref=plane(depth(dft, 32), 0),  radius=3, sigma=.45)

klmn = core.knlm.KNLMeansCL(dpir, 1, 3, h=.3, wmode=3, rclip=ref)
merge = core.std.ShufflePlanes([depth(bm3d, 16), klmn], [0, 1, 2], vs.YUV)

f3k = core.neo_f3kdb.Deband(merge, 12, 64, 70, 70, 0, 0, 5).neo_f3kdb.Deband(25, 45, 60, 60, 0, 0)
dbmask = core.tcanny.TCanny(plane(merge, 0), 1.25, 1, op=3, mode=1, scale=.9).std.Binarize(5000)
debanded = core.std.MaskedMerge(merge, f3k, dbmask)

aamask = vsaa.pre_aa(debanded, )
aamask = core.tcanny.TCanny(get_y(aamask), 0, 1, mode=1, op=2, scale=1).std.Inflate().std.Minimum().std.Binarize(5000)
baa = vsaa.based_aa(depth(source, 16), rfactor=1.65)
aad = core.std.MaskedMerge(debanded, baa, aamask)

grain = adptvgrnMod(aad, luma_scaling=9, protect_neutral=False, temporal_average=30, temporal_radius=7, fade_edges=True, grainer=lambda x: core.noise.Add(x,  type=3,  var=1.85, uvar=.3, constant=False, xsize=2.25, ysize=2.25))

out = depth(grain, 10)

settings = vof.settings_builder_x265(preset='slower', crf=7, psy_rd=2.25, psy_rdoq=2.75, qcomp=0.72, rskip=0, aq_strength=0.75, chroma_qpoffsets=-4, tu_intra_depth=2, tu_inter_depth=2, ref=6, rd=4, rect=True, amp=True, print_settings=False, append="--limit-modes --ctu 32 --max-tu-size 16")
vof.Setup().encode_video(out, settings, codec='x265')
