import functools
import subprocess
from pydoc import importfile
from re import M
from tkinter import E
from typing import BinaryIO, cast

import adptvgrnMod
import dfmderainbow
import EoEfunc as eoe
import havsfunc as haf
import kagefunc as kgf
import lvsfunc as lvf
import muvsfunc as muf
import rezafunc as rzf
import vapoursynth as vs
import vardefunc as vdf
import vsdpir
import vsutil
from ccd import ccd
from debandshit import dumb3kdb, placebo_deband
from vardefunc.noise import AddGrain, Graigasm
from vsdpir import DPIR
from vsutil import depth, get_y, join, plane

# region setup

DEFAULT_PATH = r"test.d2v"
DEFAULT_CACHE = 400000
DEFAULT_THREADS = 32

env = eoe.setup_env(globals())
core: vs.Core = env["core"]
src_path: str = env["src_path"]
GPU: int = env["GPU"]
debug: bool = env["debug"]



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
    prefetch = 20
    threads = 32

    print("encoding", file)
    with subprocess.Popen(["x265.exe", '-', '--y4m',  '--aq-bias-strength', "0.85", '--preset', 'slow', '--bframes', '16', '--input-depth', '10', '--output-depth', '10', '--rect', '--no-cutree', '--no-sao', '--amp', '--no-strong-intra-smoothing', '--no-early-skip', '--rskip', '0', '--no-fast-intra', '--no-tskip', '--psy-rd', '2.25', '--psy-rdoq', '2.75', '--bframes', '16', '--fades', '--crf', '11', '--aq-mode', '5', '--aq-strength', '.85', '--cbqpoffs', '-3', '--crqpoffs', '-3', '--qcomp', '0.7', '--deblock', '-2:-2', '--me', '3', '--b-intra', '--colorprim', 'smpte170m', '--transfer', 'smpte170m', '--colormatrix', 'smpte170m', '--range', 'limited', '--chromaloc', '0',   '-o',
             f'{file}.265'], stdin=subprocess.PIPE) as process:
        clip.output(cast(BinaryIO, process.stdin), y4m, progress_update, prefetch)




def postprocess(n, f, clip, deinterlaced):
   if f.props['_Combed'] > 0:
      return deinterlaced
   else:
      return clip


def filtering(clip):
    matched_clip = core.vivtc.VFM(clip, 1,  clip2=depth(clip, 16))
    deinterlaced_clip = haf.QTGMC(matched_clip, TFF=True, FPSDivisor=2)
    postprocessed_clip = vs.core.std.FrameEval(matched_clip, functools.partial(postprocess, clip=matched_clip, deinterlaced=deinterlaced_clip), prop_src=matched_clip)
    decimated_clip = vs.core.vivtc.VDecimate(postprocessed_clip)

    
    crop = core.std.Crop(decimated_clip, 10, 6)
    resize = core.resize.Spline16(crop, 704, 528)
    vinverse = haf.Vinverse2(resize, amnt=3, scl=1.75)
    sloc = [0, 0, 0.35, 2, 0.4, 4, 1, 8]
    prefilter = core.dfttest.DFTTest(vinverse, sigma=1, ssx=sloc, ssy=sloc, planes=[0], )

    derainbow = dfmderainbow.DFMDerainbowMC(resize,11,  radius=2 )

    cmd = eoe.dn.CMDegrain(derainbow, 3, 90, 70, 4)

    bm3d = core.bm3dcpu.BM3D(depth(derainbow, 32), depth(cmd, 32), .35, radius=0, )
    bm3d2 = core.resize.Bicubic(bm3d, format=vs.YUV420P16, matrix=6, transfer=6, primaries=6, range=0, matrix_in=6, primaries_in=6, transfer_in=6)

    rgb = core.resize.Bicubic(bm3d, format=vs.RGBS, matrix_in=6, transfer_in=6, primaries_in=6, range=0).std.Limiter()
    dpir = DPIR(rgb, strength=25, task='deblock', provider=3, )
    #dpir = core.w2xnvk.Waifu2x(rgb, 3, 1, 2)

    dpir = ccd(dpir, )
    dpir = core.resize.Bicubic(dpir, format=vs.YUV420P16, matrix=6, transfer=6, primaries=6, range=0)

    dnmask = bm3d2.std.PlaneStats().adg.Mask(luma_scaling=50)
    masked = core.std.MaskedMerge(dpir, bm3d2, dnmask)


    denoised = join([
        plane(masked, 0),
        plane(dpir, 1),
        plane(dpir, 2)
    ])

    deband = dumb3kdb(denoised, 18, [40, 20], 0)
    deband = placebo_deband(deband, 8, 2, 2, 4)
    mask = rzf.retinex_edgemask(denoised, 1.5, 2)
    debanded = core.std.MaskedMerge(deband, denoised, mask)
    

    graigasm_args = dict(
        thrs=[x << 8 for x in (32, 80, 128, 176)],
         strengths=[(0.7, 0.4), (0.6, 0.3), (0.3, 0.25), (0.1, 0.0)],
        sizes=(1.15, 1.05, 1.1, 1),
        sharps=(65, 55, 45, 45),
        grainers=[
            AddGrain(seed=1488, constant=False),
            AddGrain(seed=1337, constant=False),
            AddGrain(seed=69, constant=True),
            AddGrain(seed=420, constant=True)
          ]
     )
    grain = Graigasm(**graigasm_args).graining(deband) 
    grain = adptvgrnMod.adptvgrnMod(grain, .2, size=1.2, sharp=65, static=True, luma_scaling=7, grain_chroma=False, temporal_average=50, seed=69420)
    grain = depth(grain, 10)
    return grain



episode01 = core.d2v.Source('episode01.d2v')
episode02 = core.d2v.Source('episode02.d2v')
episode03 = core.d2v.Source('episode03.d2v')
episode04 = core.d2v.Source('episode04.d2v')
episode05 = core.d2v.Source('episode05.d2v')
episode06 = core.d2v.Source('episode06.d2v')
episode07 = core.d2v.Source('episode07.d2v')
episode08 = core.d2v.Source('episode08.d2v')
episode09 = core.d2v.Source('episode09.d2v')
episode10 = core.d2v.Source('episode10.d2v')
episode11 = core.d2v.Source('episode11.d2v')
episode12 = core.d2v.Source('episode12.d2v')
episode13 = core.d2v.Source('episode13.d2v')
episode14 = core.d2v.Source('episode14.d2v')
episode15 = core.d2v.Source('episode15.d2v')
episode16 = core.d2v.Source('episode16.d2v')
episode17 = core.d2v.Source('episode17.d2v')
episode18 = core.d2v.Source('episode18.d2v')
episode19 = core.d2v.Source('episode19.d2v')
episode20 = core.d2v.Source('episode20.d2v')



enc(filtering(episode01), "Episode 01")
enc(filtering(episode02), "Episode 02")
enc(filtering(episode03), "Episode 03")
enc(filtering(episode04), "Episode 04")
enc(filtering(episode05), "Episode 05")
enc(filtering(episode06), "Episode 06")
enc(filtering(episode07), "Episode 07")
enc(filtering(episode08), "Episode 08")
enc(filtering(episode09), "Episode 09")
enc(filtering(episode10), "Episode 10")
enc(filtering(episode11), "Episode 11")
enc(filtering(episode12), "Episode 12")
enc(filtering(episode13), "Episode 13")
enc(filtering(episode14), "Episode 14")
enc(filtering(episode15), "Episode 15")
enc(filtering(episode16), "Episode 16")
enc(filtering(episode17), "Episode 17")
enc(filtering(episode18), "Episode 18")
enc(filtering(episode19), "Episode 19")
enc(filtering(episode20), "Episode 20")
