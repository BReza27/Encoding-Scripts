
import vapoursynth as vs
import EoEfunc as eoe
from vsutil import depth, plane, join, get_y
from rezafunc import retinex_edgemask as mask
from adptvgrnMod import adptvgrnMod
import vodesfunc as vof
import vskernels
from debandshit import placebo_deband, dumb3kdb
from vsdpir import DPIR
import havsfunc as haf
import subprocess
from typing import BinaryIO, cast

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
    with subprocess.Popen(["x265", '-', '--y4m', '--ctu', '32', '--max-tu-size', '16', '--rd', '4', '--limit-modes',  '--aq-bias-strength', "0.95", '--preset', 'slower', '--bframes', '16', '--input-depth', '10', '--output-depth', '10', '--rect', '--no-cutree', '--no-sao', '--amp', '--no-strong-intra-smoothing', '--no-early-skip', '--rskip', '0', '--no-fast-intra', '--no-tskip', '--psy-rd', '2.25', '--psy-rdoq', '2.75', '--bframes', '16', '--fades', '--crf', '12', '--aq-mode', '5', '--aq-strength', '.85', '--cbqpoffs', '-3', '--crqpoffs', '-3', '--qcomp', '0.7', '--deblock', '-2:-2', '--me', '3', '--b-intra', '--tu-intra-depth', '3', '--tu-inter-depth', '3', '--colorprim', '1', '--transfer', '1', '--colormatrix', '1', '--range', 'limited', '--chromaloc', '0',   '-o',
             f'{file}.265'], stdin=subprocess.PIPE) as process:
        clip.output(cast(BinaryIO, process.stdin), y4m, progress_update, prefetch)



ep12 = core.lsmas.LWLibavSource(r"N:\BDMVs\[BDMV][boku wa tomodachi ga sukunai][僕は友達が少ない][Vol.1-Vol.6 Fin+OVA]\[BDMV][120113][Boku wa Tomodachi ga Sukunai][Vol.01]\BDROM\BDMV\STREAM\00002.m2ts")
ep34 = core.lsmas.LWLibavSource(r"N:\BDMVs\[BDMV][boku wa tomodachi ga sukunai][僕は友達が少ない][Vol.1-Vol.6 Fin+OVA]\[BDMV][120222][Boku wa Tomodachi ga Sukunai][Vol.02]\BDROM\BDMV\STREAM\00002.m2ts")
ep56 = core.lsmas.LWLibavSource(r"N:\BDMVs\[BDMV][boku wa tomodachi ga sukunai][僕は友達が少ない][Vol.1-Vol.6 Fin+OVA]\[BDMV][120321][Boku wa Tomodachi ga Sukunai][Vol.03]\BDROM\BDMV\STREAM\00002.m2ts")
ep78 = core.lsmas.LWLibavSource(r"N:\BDMVs\[BDMV][boku wa tomodachi ga sukunai][僕は友達が少ない][Vol.1-Vol.6 Fin+OVA]\[BDMV][120425][Boku wa Tomodachi ga Sukunai][Vol.04]\BDROM\BDMV\STREAM\00002.m2ts")
ep910 = core.lsmas.LWLibavSource(r"N:\BDMVs\[BDMV][boku wa tomodachi ga sukunai][僕は友達が少ない][Vol.1-Vol.6 Fin+OVA]\[BDMV][120523][Boku wa Tomodachi ga Sukunai][Vol.05]\BDROM\BDMV\STREAM\00002.m2ts")
ep1112 = core.lsmas.LWLibavSource(r"N:\BDMVs\[BDMV][boku wa tomodachi ga sukunai][僕は友達が少ない][Vol.1-Vol.6 Fin+OVA]\[BDMV][120829][Boku wa Tomodachi ga Sukunai][Vol.06]\BDROM\BDMV\STREAM\00002.m2ts")
ova = core.lsmas.LWLibavSource(r"N:\BDMVs\[BDMV][boku wa tomodachi ga sukunai][僕は友達が少ない][Vol.1-Vol.6 Fin+OVA]\[BDMV][120926][Boku wa Tomodachi ga Sukunai][OVA]\BDROM\BDMV\Stream\00002.m2ts")


episode01 = core.std.Trim(ep12, 0, 35293)
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

src = ep12
def filtering(clip):
    src = clip.resize.Bicubic(format=vs.YUV420P16, matrix=1, transfer=1, primaries=1, range=0, matrix_in=1, transfer_in=1, primaries_in=1, )
    rescale = vof.vodes_rescale(src, 720, 1280, 0, vskernels.Bicubic(0, 0.5), opencl=True)[0]
    ref = eoe.denoise.CMDegrain(rescale, 2, 80, 90, 3, )
    bm3d = core.bm3dcuda_rtc.BM3Dv2(depth(rescale, 32), None, [.4], radius=2)
    rgb = core.resize.Bicubic(bm3d, format=vs.RGBS).std.Limiter()
    dpir = DPIR(rgb, 20, 'deblock', provider=2, trt_fp16=True, dual=True, trt_engine_cache_path=r"C:\Users\Brandon\Desktop\cache", log_level=4).resize.Bicubic(format=vs.YUV420P16, matrix=1, transfer=1, primaries=1, range=0)

    denoised = core.std.ShufflePlanes([bm3d.resize.Bicubic(format=vs.YUV420P16, matrix=1, transfer=1, primaries=1, range=0), dpir], [0,1,2], vs.YUV)
    deband = placebo_deband(denoised, 14, [2.5, 2, 2], 2, 2)
    deband = core.neo_f3kdb.Deband(denoised, 18, 32, 48, 48, 0, 0, )
    dbmask = mask(denoised, 1.5, 2, .9).std.Binarize(7000)
    debanded = core.std.MaskedMerge(deband, denoised, dbmask)
    grain = adptvgrnMod(debanded, luma_scaling=10, protect_neutral=True, grainer=lambda x: core.noise.Add(x, type=3,  var=2.5, uvar=.35, constant=True, xsize=3, ysize=3, seed=1337))
    return grain

enc(filtering(episode01), "Episode 01")
enc(filtering(episode02), "Episode 02")
enc(filtering(episode03,), "Episode 03")
enc(filtering(episode04,), "Episode 04")
enc(filtering(episode05,), "Episode 05")
enc(filtering(episode06,), "Episode 06")
enc(filtering(episode07,), "Episode 07")
enc(filtering(episode08,), "Episode 08")
enc(filtering(episode09,), "Episode 09")
enc(filtering(episode10,), "Episode 10")
enc(filtering(episode11,), "Episode 11")
enc(filtering(episode12), "Episode 12")
enc(filtering(ova),  "OVA")
