import descale
import EoEfunc as eoe
import rezafunc as rzf
import vapoursynth as vs
from adptvgrnMod import adptvgrnMod
from jvsfunc import ccdmod
import vardefunc
from vsutil import depth, get_y, join, plane, split
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

src = depth(src, 16)



def enc(clip, file):
    y4m = True
    prefetch = 10
    threads = 32

    print("encoding", file)
    with subprocess.Popen(["x265", '-', '--no-progress', '--y4m', '--limit-modes', '--ctu', '32', '--max-tu-size', '16', '--rd', '4',  '--aq-bias-strength', "0.95", '--preset', 'slower', '--bframes', '16', '--input-depth', '10', '--output-depth', '10', '--rect', '--no-cutree', '--no-sao', '--amp', '--no-strong-intra-smoothing', '--no-early-skip', '--rskip', '0', '--no-fast-intra', '--no-tskip', '--psy-rd', '2.25', '--psy-rdoq', '2.5', '--bframes', '16', '--fades', '--crf', '12', '--aq-mode', '5', '--aq-strength', '.9', '--cbqpoffs', '-3', '--crqpoffs', '-3', '--qcomp', '0.72', '--deblock', '-2:-2', '--me', '3', '--b-intra', '--tu-intra-depth', '3', '--tu-inter-depth', '3', '--colorprim', '1', '--transfer', '1', '--colormatrix', '1', '--range', 'limited', '--chromaloc', '0',   '-o',
             f'{file}.265'], stdin=subprocess.PIPE) as process:
        clip.output(cast(BinaryIO, process.stdin), y4m, progress_update, prefetch)


ep12 = core.lsmas.LWLibavSource(r"00002.m2ts")
ep34 = core.lsmas.LWLibavSource(r"E:\haganai\[BDMV][boku wa tomodachi ga sukunai][僕は友達が少ない][Vol.1-Vol.6 Fin+OVA]\[BDMV][120222][Boku wa Tomodachi ga Sukunai][Vol.02]\BDROM\BDMV\STREAM\00002.m2ts")
ep56 = core.lsmas.LWLibavSource(r"E:\haganai\[BDMV][boku wa tomodachi ga sukunai][僕は友達が少ない][Vol.1-Vol.6 Fin+OVA]\[BDMV][120321][Boku wa Tomodachi ga Sukunai][Vol.03]\BDROM\BDMV\STREAM\00002.m2ts")
ep78 = core.lsmas.LWLibavSource(r"E:\haganai\[BDMV][boku wa tomodachi ga sukunai][僕は友達が少ない][Vol.1-Vol.6 Fin+OVA]\[BDMV][120425][Boku wa Tomodachi ga Sukunai][Vol.04]\BDROM\BDMV\STREAM\00002.m2ts")
ep910 = core.lsmas.LWLibavSource(r"E:\haganai\[BDMV][boku wa tomodachi ga sukunai][僕は友達が少ない][Vol.1-Vol.6 Fin+OVA]\[BDMV][120523][Boku wa Tomodachi ga Sukunai][Vol.05]\BDROM\BDMV\STREAM\00002.m2ts")
ep1112 = core.lsmas.LWLibavSource(r"E:\haganai\[BDMV][boku wa tomodachi ga sukunai][僕は友達が少ない][Vol.1-Vol.6 Fin+OVA]\[BDMV][120829][Boku wa Tomodachi ga Sukunai][Vol.06]\BDROM\BDMV\STREAM\00002.m2ts")
ova = core.lsmas.LWLibavSource(r"E:\haganai\[BDMV][boku wa tomodachi ga sukunai][僕は友達が少ない][Vol.1-Vol.6 Fin+OVA]\[BDMV][120926][Boku wa Tomodachi ga Sukunai][OVA]\BDROM\BDMV\STREAM\00002.m2ts")


episode01 = core.std.Trim(ep12, 0, 35294)
episode02 = core.std.Trim(ep12, 35294, )
episode03 = core.std.Trim(ep34, 0, 35294)
episode04 = core.std.Trim(ep34, 35294, )
episode05 = core.std.Trim(ep56, 0, 35292)
episode06 = core.std.Trim(ep56, 35292, )
episode07 = core.std.Trim(ep78, 0, 35295)
episode08 = core.std.Trim(ep78, 35295, )
episode09 = core.std.Trim(ep910, 0, 35294)
episode10 = core.std.Trim(ep910, 35294, )
episode11 = core.std.Trim(ep1112, 0, 35294)
episode12 = core.std.Trim(ep1112, 35294, )
ova = ova

ed = core.lsmas.LWLibavSource(r"ED.mkv")[:2158]
op = core.lsmas.LWLibavSource(r"op (2).mkv")
op = core.std.Trim(op, 1, 2152)

#audio = core.std.

def filtering(clip, ncstart, edstart):
    luma = plane(clip, 0)
    descaled = descale.Debicubic(luma, 1280, 720, 0, 0.5, gray=True)
    upscale = vardefunc.nnedi3_upscale(descaled, use_znedi=True)
    resize = core.resize.Bicubic(upscale, 1920, 1080)
    rescaled = core.std.ShufflePlanes([resize, clip], [0, 1, 2], vs.YUV)

    luma2 = plane(rescaled, 0)

    ref = eoe.denoise.CMDegrain(luma2, 2, 80, 90, 3, freq_merge=0, )
    ref = depth(ref, 32)

    den = depth(luma2, 32).bm3dcpu.BM3D(ref=ref, sigma=0.6, radius=3).bm3d.VAggregate(radius=3, sample=1)
    den = core.std.ShufflePlanes([den, depth(rescaled, 32)], [0, 1, 2], vs.YUV)
    denoised = depth(den, 16)
    denoised = ccdmod(denoised)

    deband = core.neo_f3kdb.Deband(denoised, 14, 32, 32, 32, 0, 0, 4, )
    dbmask = rzf.retinex_edgemask(denoised, 2, 3, .9).std.Binarize(7500)
    debanded = core.std.MaskedMerge(deband, denoised, dbmask)

    grain = adptvgrnMod(debanded, [.2, .05], 1.1, 60, True, 10)

   
    opdone = grain[:ncstart] + op
    opdone = opdone + grain[len(opdone):]
    eddone = opdone[:edstart] + ed 
    eddone = eddone + grain[len(eddone):]
    final = depth(eddone, 10)
    return final



#test = filtering(episode01, 2206, 32152)


enc(filtering(episode01, 2206, 32152), "Episode 01")
enc(filtering(episode02, 1201, 32129), "Episode 02")
enc(filtering(episode03, 960, 32152), "Episode 03")
enc(filtering(episode04, 1080, 32128), "Episode 04")
enc(filtering(episode05, 2206, 32150), "Episode 05")
enc(filtering(episode06, 1774, 32129), "Episode 06")
enc(filtering(episode07, 1080, 32153), "Episode 07")
enc(filtering(episode08, 1176, 32129), "Episode 08")
enc(filtering(episode09, 1134, 32152), "Episode 09")
enc(filtering(episode10, 1654, 32128), "Episode 10")
enc(filtering(episode11, 1750, 32152), "Episode 11")
enc(filtering(episode12, 1, 32129), "Episode 12")
enc(filtering(ova), "OVA")
