import vapoursynth as vs
import vsdeinterlace 
from vodesfunc import out
import vsdenoise
import vsmasktools
import vsmlrt
import havsfunc
from vstools import depth, plane
import adptvgrnMod
import subprocess
from typing import BinaryIO, cast
import vstools


core = vs.core

ep1 = core.core.d2v.Source(input=r'C:\Users\Brandon\Desktop\DVD files\episode01.d2v', rff=False)
ep2 = core.core.d2v.Source(input=r'C:\Users\Brandon\Desktop\DVD files\episode02.d2v', rff=False)
ep3 = core.core.d2v.Source(input=r'C:\Users\Brandon\Desktop\DVD files\episode03.d2v', rff=False)
ep4 = core.core.d2v.Source(input=r'C:\Users\Brandon\Desktop\New folder (3)\episode04.d2v', rff=False)
ep5= core.core.d2v.Source(input=r'C:\Users\Brandon\Desktop\New folder (3)\episode05.d2v', rff=False)
ep6 = core.core.d2v.Source(input=r'C:\Users\Brandon\Desktop\New folder (3)\episode06V2.d2v', rff=False)

ncop = core.core.d2v.Source(input=r'C:\Users\Brandon\Desktop\DVD files\NCOP.d2v', rff=False)
nced = core.core.d2v.Source(input=r'C:\Users\Brandon\Desktop\New folder (3)\nced.d2v', rff=False)





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
    with subprocess.Popen(["x265", '-', '--y4m', '--sar', '640:531', '--ctu', '32', '--max-tu-size', '16', '--rd', '6', '--limit-modes',  '--aq-bias-strength', "0.8", '--preset', 'slower', '--bframes', '16', '--input-depth', '10', '--output-depth', '10', '--rect', '--no-cutree', '--no-sao', '--amp', '--no-strong-intra-smoothing', '--no-early-skip', '--rskip', '0', '--no-fast-intra', '--no-tskip', '--psy-rd', '2', '--psy-rdoq', '2.25', '--bframes', '16', '--fades', '--crf', '8', '--aq-mode', '5', '--aq-strength', '.75', '--cbqpoffs', '-3', '--crqpoffs', '-3', '--qcomp', '0.75', '--deblock', '-2:-2', '--me', '3', '--b-intra', '--tu-intra-depth', '4', '--tu-inter-depth', '4', '--colorprim', 'smpte170m', '--transfer', 'smpte170m', '--colormatrix', 'smpte170m', '--range', 'limited',  '--chromaloc', '0',   '-o',
             f'{file}.265'], stdin=subprocess.PIPE) as process:
        clip.output(cast(BinaryIO, process.stdin), y4m, progress_update, prefetch)


def filtering(clip):
    src = clip.resize.Bicubic(matrix=6, transfer=6, primaries=6, matrix_in=6, transfer_in=6, primaries_in=6)
    fix = havsfunc.QTGMC(src, FPSDivisor=2, Preset="placebo")
    fix = vstools.rfs(src, fix, [(39135, 41416)])
    vinverse = havsfunc.Vinverse(fix)
    rgb = core.resize.Bicubic(vinverse, format=vs.RGBS).std.Limiter()
    dpir = vsmlrt.DPIR(rgb, 20, backend=vsmlrt.Backend().TRT(fp16=True, workspace=10000, use_cuda_graph=True, use_cublas=True, num_streams=2), model=3, ).resize.Bicubic(format=vs.YUV420P16, matrix=6, transfer=6, primaries=6)
    deband1 = core.neo_f3kdb.Deband(dpir, 12, 45, 60, 60, grainc=0, grainy=0)
    dbmask = vsmasktools.retinex(dpir)
    deband1 = core.std.MaskedMerge(deband1, dpir, dbmask)
    ccd = vsdenoise.ccd(deband1, 5, )
    wnnm = core.wnnm.WNNM(depth(ccd, 32), [1.5, 2.5, 2.5], radius=3)
    deband2 = core.neo_f3kdb.Deband(depth(wnnm, 16), 12, 30, 45, 45, grainc=0, grainy=0)
    deband1 = core.std.MaskedMerge(deband1, depth(wnnm, 16) , dbmask)
    grain = adptvgrnMod.adptvgrnMod(deband2, temporal_average=20, luma_scaling=8, grainer=lambda x: core.noise.Add(x, type=3,  var=2.5, uvar=.4, constant=False, xsize=3, ysize=3))
    grain = depth(grain, 10)
    return grain

enc(filtering(ep2), "Episode 02")
enc(filtering(ep3), "Episode 03")
enc(filtering(ep4), "Episode 04")
enc(filtering(ep5), "Episode 05")
enc(filtering(ep6), "Episode 06")
enc(filtering(ncop), "NCOP")
enc(filtering(nced), "NCED")

