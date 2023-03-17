import EoEfunc as eoe
import havsfunc as haf
import vapoursynth as vs
import vodesfunc as vof
import vskernels
import vsaa
from rezafunc import retinex_edgemask as mask
from vsutil import depth, get_y, join, plane
from debandshit import f3kpf
from finedehalo import fine_dehalo
from adptvgrnMod import adptvgrnMod
import subprocess
from typing import BinaryIO, cast


# region setup

DEFAULT_PATH = r"00001.m2ts"
DEFAULT_CACHE = 80
DEFAULT_THREADS = 32

env = eoe.setup_env(globals())
src_path: str = env["src_path"]
GPU: int = env["GPU"]
debug: bool = env["debug"]

# endregion setup

core = vs.core


def progress_update(value: int, endvalue: int) -> None:
    """
    Callback function used in clip.output
    :param value:       Current value
    :param endvalue:    End value
    """
    return print(
        "\rVapourSynth: %i/%i ~ %.2f%% " % (
            value, endvalue, 100 * value / endvalue
        ),
        end=""
    )

def enc(clip, file):
    y4m = True
    prefetch = 10
    threads = 32

    print("encoding", file)
    with subprocess.Popen(["x265", '-', '--y4m', '--ctu', '32', '--max-tu-size', '16', '--rd', '4',  '--aq-bias-strength', "0.85", '--preset', 'slower', '--bframes', '16', '--input-depth', '10', '--output-depth', '10', '--rect', '--no-cutree', '--no-sao', '--amp', '--no-strong-intra-smoothing', '--no-early-skip', '--rskip', '0', '--no-fast-intra', '--no-tskip', '--psy-rd', '2', '--psy-rdoq', '2.25', '--bframes', '16', '--fades', '--crf', '12.5', '--aq-mode', '5', '--aq-strength', '.8', '--cbqpoffs', '-3', '--crqpoffs', '-3', '--qcomp', '0.65', '--deblock', '-2:-2', '--me', '3', '--b-intra', '--tu-intra-depth', '3', '--tu-inter-depth', '3', '--colorprim', '1', '--transfer', '1', '--colormatrix', '1', '--range', 'limited', '--chromaloc', '0',   '-o',
             f'{file}.265'], stdin=subprocess.PIPE) as process:
        clip.output(cast(BinaryIO, process.stdin), y4m, progress_update, prefetch)


def filtering(clip ):

    src = clip
    src = depth(src, 16)

    rescale = vof.vodes_rescale(src, 846, 1504, descale_kernel=vskernels.Bicubic(), mask_threshold=0.03)[0]
    test = core.bilateral.Bilateral(rescale)
    hqdering = haf.HQDeringmod(rescale, incedge=True, sharp=0, thr=6, darkthr=0, smoothed=test)

    degrain = eoe.dn.CMDegrain(hqdering, 2, 80, refine=3, contrasharp=False, thSADC=60,)
    bm3d = core.bm3dcpu.BM3D(depth(hqdering, 32,), depth(degrain, 32), [.25], radius=3)
    bm3d = core.bm3d.VAggregate(bm3d, 3).resize.Bicubic(format=vs.YUV420P16)
    denoise = bm3d

    f3k = f3kpf(denoise, 18, [32, 16, 16], 0, )
    retinex = mask(denoise, 1, 3, 1).std.Binarize(7500)
    debanded = core.std.MaskedMerge(f3k, denoise, retinex)

    sraa = vsaa.upscaled_sraa(debanded, 1.4,  )
    aa1 = haf.HQDeringmod(sraa,  nrmode=1, sharp=0, darkthr=0)
    aa2 = haf.DeHalo_alpha(sraa, darkstr=0, ss=1.25)
    aa3 = core.std.Expr([sraa, aa1, aa2], "x y - abs x z - abs < y z ?")
    aa4 = fine_dehalo(sraa, ref=aa3, rx=1.7, brightstr=.5, darkstr=0)
    aa_mask = core.tcanny.TCanny(get_y(debanded), 1.5, 1.5, op=2, mode=1, scale=1.25).std.Binarize(5000).std.Maximum()
    aad = core.std.MaskedMerge(debanded, aa4, aa_mask)

    grain = adptvgrnMod(aad, luma_scaling=10, protect_neutral=True,  grainer=lambda x: core.noise.Add(x, type=3,  var=2, uvar=.25, constant=True, xsize=3, ysize=3, seed=1337))
    return grain


Episode01 = core.lsmas.LWLibavSource(r"N:\BDMVs\Aharen-san wa Hakarenai JP BDMV\Vol.1\AHAREN_1\BDMV\STREAM\00000.m2ts")
Episode02 = core.lsmas.LWLibavSource(r"N:\BDMVs\Aharen-san wa Hakarenai JP BDMV\Vol.1\AHAREN_1\BDMV\STREAM\00001.m2ts")
Episode03= core.lsmas.LWLibavSource(r"N:\BDMVs\Aharen-san wa Hakarenai JP BDMV\Vol.1\AHAREN_1\BDMV\STREAM\00002.m2ts")
Episode04 = core.lsmas.LWLibavSource(r"N:\BDMVs\Aharen-san wa Hakarenai JP BDMV\Vol.1\AHAREN_1\BDMV\STREAM\00003.m2ts")
Episode05 = core.lsmas.LWLibavSource(r"N:\BDMVs\Aharen-san wa Hakarenai JP BDMV\Vol.2\AHAREN_2\BDMV\STREAM\00000.m2ts")
Episode06 = core.lsmas.LWLibavSource(r"N:\BDMVs\Aharen-san wa Hakarenai JP BDMV\Vol.2\AHAREN_2\BDMV\STREAM\00001.m2ts")
Episode07 = core.lsmas.LWLibavSource(r"N:\BDMVs\Aharen-san wa Hakarenai JP BDMV\Vol.2\AHAREN_2\BDMV\STREAM\00002.m2ts")
Episode08 = core.lsmas.LWLibavSource(r"N:\BDMVs\Aharen-san wa Hakarenai JP BDMV\Vol.2\AHAREN_2\BDMV\STREAM\00003.m2ts")
Episode09 = core.lsmas.LWLibavSource(r"N:\BDMVs\Aharen-san wa Hakarenai JP BDMV\Vol.3\AHAREN_3\BDMV\STREAM\00000.m2ts")
Episode10 = core.lsmas.LWLibavSource(r"N:\BDMVs\Aharen-san wa Hakarenai JP BDMV\Vol.3\AHAREN_3\BDMV\STREAM\00001.m2ts")
Episode11 = core.lsmas.LWLibavSource(r"N:\BDMVs\Aharen-san wa Hakarenai JP BDMV\Vol.3\AHAREN_3\BDMV\STREAM\00002.m2ts")
Episode12 = core.lsmas.LWLibavSource(r"N:\BDMVs\Aharen-san wa Hakarenai JP BDMV\Vol.3\AHAREN_3\BDMV\STREAM\00003.m2ts")
ncop = core.lsmas.LWLibavSource(r"N:\BDMVs\Aharen-san wa Hakarenai JP BDMV\Vol.2\AHAREN_2\BDMV\STREAM\00005.m2ts")
nced = core.lsmas.LWLibavSource(r"N:\BDMVs\Aharen-san wa Hakarenai JP BDMV\Vol.3\AHAREN_3\BDMV\STREAM\00005.m2ts")
ep3_nced = core.lsmas.LWLibavSource(r"N:\BDMVs\Aharen-san wa Hakarenai JP BDMV\Vol.1\AHAREN_1\BDMV\STREAM\00005.m2ts")

enc(filtering(ncop), "ncop")
enc(filtering(nced), "nced")
enc(filtering(ep3_nced), "ep3 nced")
enc(filtering(Episode01), "Episode 01")
enc(filtering(Episode02), "Episode 02")
enc(filtering(Episode03), "Episode 03")
enc(filtering(Episode04), "Episode 04")
enc(filtering(Episode05), "Episode 05")
enc(filtering(Episode06), "Episode 06")
enc(filtering(Episode07), "Episode 07")
enc(filtering(Episode08), "Episode 08")
enc(filtering(Episode09), "Episode 09")
enc(filtering(Episode10), "Episode 10")
enc(filtering(Episode11), "Episode 11")
enc(filtering(Episode12), "Episode 12")