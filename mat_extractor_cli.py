#!/usr/bin/env python3
"""
MAT Extractor CLI – Audio + LI channel

Non-GUI version with the SAME DEFAULTS as the GUI.

Defaults:
- audio index = 0
- LI index = 4
- LI channel = 3 (1-based)
- Excel = ON
- Plaintext = ON (2 rows, space-separated)
- WAV = ON
- Resample LI = OFF
- Combined WAV = OFF
- WAV format = float32
"""

import argparse
import os
import numpy as np
from scipy.io import loadmat
from scipy.io.wavfile import write as wavwrite
from scipy.signal import resample_poly


# ---------------- Utilities ----------------

def is_mat_struct(x):
    return hasattr(x, "_fieldnames")


def find_stream_array(mat, preferred):
    if preferred:
        v = mat[preferred]
        return preferred, v
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.dtype == object and is_mat_struct(v.flat[0]):
            return k, v
    raise RuntimeError("Could not auto-detect struct array")


def to_1d(x):
    a = np.asarray(x).squeeze()
    return a.reshape(-1).astype(np.float64)


def to_2d(x):
    a = np.asarray(x).squeeze()
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return a.astype(np.float64)


def normalize_int16(x):
    peak = np.max(np.abs(x)) if x.size else 0.0
    if peak == 0:
        return np.zeros_like(x, dtype=np.int16)
    scale = 32767 if peak <= 1.5 else 32767 / peak
    return np.clip(np.round(x * scale), -32768, 32767).astype(np.int16)


def upsample_poly_reflect(x, fs_in, fs_out, pad_sec=0.25):
    x = np.asarray(x, dtype=np.float64)
    pad = int(round(pad_sec * fs_in))
    pad = min(pad, max(0, len(x) - 1))
    if pad > 0:
        x = np.pad(x, (pad, pad), mode="reflect")
    g = np.gcd(fs_in, fs_out)
    y = resample_poly(x, fs_out // g, fs_in // g)
    trim = int(round(pad * fs_out / fs_in))
    if trim > 0:
        y = y[trim:-trim]
    return y


# ---------------- Writers ----------------

def write_excel(path, audio, li, sr_a, sr_l):
    import pandas as pd
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        pd.DataFrame({"audio": audio}).to_excel(w, "audio", index=False)
        pd.DataFrame({"LI_ch": li}).to_excel(w, "LI_ch", index=False)
        pd.DataFrame([
            {"stream": "audio", "srate": sr_a, "n": len(audio)},
            {"stream": "LI_ch", "srate": sr_l, "n": len(li)},
        ]).to_excel(w, "meta", index=False)


def write_text(path, audio, li):
    fmt = "%.10g"
    with open(path, "w") as f:
        f.write(" ".join(fmt % v for v in audio) + "\n")
        f.write(" ".join(fmt % v for v in li) + "\n")


def write_wav(base, audio, sr_a, li, sr_l, resample, combined, fmt, align):
    def write_one(p, sr, d):
        if fmt == "int16":
            d = normalize_int16(d)
        else:
            d = d.astype(np.float32)
        wavwrite(p, sr, d)

    if not resample:
        write_one(base + "_AUDIO.wav", sr_a, audio)
        write_one(base + "_LI_ch.wav", sr_l, li)
        return

    li_rs = upsample_poly_reflect(li, sr_l, sr_a)

    if len(audio) != len(li_rs):
        if align == "trim":
            n = min(len(audio), len(li_rs))
            audio, li_rs = audio[:n], li_rs[:n]
        else:
            n = max(len(audio), len(li_rs))
            audio = np.pad(audio, (0, n - len(audio)))
            li_rs = np.pad(li_rs, (0, n - len(li_rs)))

    if combined:
        out = np.column_stack([audio, li_rs])
        write_one(base + "_combined.wav", sr_a, out)
    else:
        write_one(base + "_AUDIO.wav", sr_a, audio)
        write_one(base + "_LI_ch_rs.wav", sr_a, li_rs)


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help=".mat file or folder")
    ap.add_argument("--var", default=None)

    ap.add_argument("--audio-index", type=int, default=0)
    ap.add_argument("--li-index", type=int, default=4)
    ap.add_argument("--li-channel", type=int, default=3)

    ap.add_argument("--no-excel", action="store_true")
    ap.add_argument("--no-text", action="store_true")
    ap.add_argument("--no-wav", action="store_true")

    ap.add_argument("--resample-li", action="store_true")
    ap.add_argument("--combined-wav", action="store_true")
    ap.add_argument("--wav-format", choices=["float32", "int16"], default="float32")
    ap.add_argument("--align", choices=["trim", "pad"], default="trim")

    args = ap.parse_args()

    if os.path.isdir(args.input):
        files = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith(".mat")]
    else:
        files = [args.input]

    for mat_path in files:
        mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        var, streams = find_stream_array(mat, args.var)

        audio_s = streams.flat[args.audio_index]
        li_s = streams.flat[args.li_index]

        audio = to_1d(audio_s.SIGNAL)
        sr_a = int(audio_s.SRATE)

        li_all = to_2d(li_s.SIGNAL)
        li = li_all[:, args.li_channel - 1]
        sr_l = int(li_s.SRATE)

        base = os.path.splitext(mat_path)[0]

        if not args.no_excel:
            write_excel(base + ".xlsx", audio, li, sr_a, sr_l)

        if not args.no_text:
            write_text(base + ".txt", audio, li)

        if not args.no_wav:
            write_wav(
                base,
                audio, sr_a,
                li, sr_l,
                args.resample_li,
                args.combined_wav,
                args.wav_format,
                args.align
            )

        print(f"Processed {os.path.basename(mat_path)}")


if __name__ == "__main__":
    main()


