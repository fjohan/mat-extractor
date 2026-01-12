#!/usr/bin/env python3
"""
MAT Extractor GUI (Tkinter)

Select a single .mat file or a folder of .mat files.
Extracts:
  - struct[audio_index]  -> audio: NAME, SRATE, SIGNAL (1D)
  - struct[li_index]     -> LI:   NAME (expected 'LI'), SRATE, SIGNAL (NxC), take li_channel (1-based)

Outputs (optional via checkboxes):
  - Excel (.xlsx): audio + LI_ch + meta sheets
  - Plaintext (.txt): two lines (rows), space-separated numbers
      line 1: audio samples
      line 2: LI channel samples (original rate; NOT resampled)
  - WAV (.wav):
      - combined (2ch) or separate files
      - optional resampling of LI to match audio SRATE using poly_reflect padding
      - alignment: trim or pad if lengths differ

Dependencies:
  pip install numpy scipy
  pip install pandas openpyxl   (only if Excel output enabled)
"""

import os
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from scipy.io import loadmat
from scipy.io.wavfile import write as wavwrite
from scipy.signal import resample_poly


# =========================
# Core extraction utilities
# =========================

def is_mat_struct(x) -> bool:
    return hasattr(x, "_fieldnames")


def as_str(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    if isinstance(x, np.ndarray) and x.size == 1:
        try:
            return str(x.item())
        except Exception:
            return str(x)
    return str(x)


def find_stream_array(mat: dict, preferred_var: str | None):
    """Return (varname, struct_object_array) where elements have NAME/SRATE/SIGNAL."""
    if preferred_var:
        if preferred_var not in mat:
            raise KeyError(f"Variable '{preferred_var}' not found.")
        v = mat[preferred_var]
        if not (isinstance(v, np.ndarray) and v.dtype == object and v.size > 0 and is_mat_struct(v.flat[0])):
            raise TypeError(f"Variable '{preferred_var}' is not a struct array.")
        return preferred_var, v

    for k, v in mat.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.dtype == object and v.size > 0 and is_mat_struct(v.flat[0]):
            s0 = v.flat[0]
            fields = set(getattr(s0, "_fieldnames", []) or [])
            if {"NAME", "SRATE", "SIGNAL"}.issubset(fields):
                return k, v

    raise RuntimeError("Could not auto-detect struct array with fields NAME, SRATE, SIGNAL.")


def get_field(s, field: str):
    return getattr(s, field)


def to_1d_float64(x) -> np.ndarray:
    arr = np.asarray(x)
    arr = np.squeeze(arr)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr.astype(np.float64, copy=False)


def to_2d_float64(x) -> np.ndarray:
    arr = np.asarray(x)
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim != 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr.astype(np.float64, copy=False)


def normalize_to_int16(x: np.ndarray) -> np.ndarray:
    """Scale float array to int16 safely."""
    x = np.asarray(x, dtype=np.float64)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak == 0.0:
        return np.zeros_like(x, dtype=np.int16)
    scale = 32767.0 if peak <= 1.5 else (32767.0 / peak)
    return np.clip(np.round(x * scale), -32768, 32767).astype(np.int16)


def upsample_poly_reflect(x_in: np.ndarray, fs_in: int, fs_out: int, pad_sec: float = 0.25) -> np.ndarray:
    """
    Resample using resample_poly with reflect padding to reduce edge artifacts.
    """
    x = np.asarray(x_in, dtype=np.float64).reshape(-1)
    if len(x) < 2:
        factor = int(round(fs_out / fs_in))
        return np.repeat(x, factor).astype(np.float64, copy=False)

    pad = int(round(pad_sec * fs_in))
    pad = min(pad, max(0, len(x) - 1))  # reflect requires at least 2 samples

    xp = np.pad(x, (pad, pad), mode="reflect") if pad > 0 else x

    g = int(np.gcd(fs_out, fs_in))
    up = fs_out // g
    down = fs_in // g

    y = resample_poly(xp, up, down)

    # Trim padding by time
    output_pad = int(round(pad * fs_out / fs_in))
    if output_pad > 0 and y.size > 2 * output_pad:
        y = y[output_pad:-output_pad]
    return y.astype(np.float64, copy=False)


def extract_audio_and_li_ch(mat_path: str,
                            varname: str | None,
                            audio_index: int,
                            li_index: int,
                            li_channel_1based: int):
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    chosen_var, streams = find_stream_array(mat, varname)

    if streams.size <= max(audio_index, li_index):
        raise IndexError(f"Struct array '{chosen_var}' has {streams.size} elements; need {audio_index} and {li_index}.")

    audio_struct = streams.flat[audio_index]
    li_struct = streams.flat[li_index]

    for label, s in [("audio", audio_struct), ("LI", li_struct)]:
        if not is_mat_struct(s):
            raise TypeError(f"{label} element is not a MATLAB struct.")
        fields = set(getattr(s, "_fieldnames", []) or [])
        for need in ("NAME", "SRATE", "SIGNAL"):
            if need not in fields:
                raise KeyError(f"{label} struct missing field '{need}' (fields={sorted(fields)})")

    audio_name = as_str(get_field(audio_struct, "NAME"))
    audio_srate = int(np.asarray(get_field(audio_struct, "SRATE")).item())
    audio_sig = to_1d_float64(get_field(audio_struct, "SIGNAL"))

    li_name = as_str(get_field(li_struct, "NAME"))
    li_srate = int(np.asarray(get_field(li_struct, "SRATE")).item())
    li_sig2d = to_2d_float64(get_field(li_struct, "SIGNAL"))

    ch = li_channel_1based
    if ch < 1 or ch > li_sig2d.shape[1]:
        raise ValueError(f"LI has {li_sig2d.shape[1]} channels; requested {ch}")

    li_ch = li_sig2d[:, ch - 1].astype(np.float64, copy=False)

    return {
        "var": chosen_var,
        "audio": {"name": audio_name, "srate": audio_srate, "signal": audio_sig},
        "li": {"name": li_name, "srate": li_srate, "signal": li_ch, "channel": ch},
    }


# =========================
# Writers
# =========================

def write_excel(path: str, audio_sig: np.ndarray, li_sig: np.ndarray, audio_srate: int, li_srate: int):
    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError("Excel output requires pandas+openpyxl. Run: pip install pandas openpyxl") from e

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"audio": audio_sig}).to_excel(writer, sheet_name="audio", index=False)
        pd.DataFrame({"LI_ch": li_sig}).to_excel(writer, sheet_name="LI_ch", index=False)
        pd.DataFrame(
            [
                {"stream": "audio", "srate_hz": audio_srate, "n_samples": int(audio_sig.size)},
                {"stream": "LI_ch", "srate_hz": li_srate, "n_samples": int(li_sig.size)},
            ]
        ).to_excel(writer, sheet_name="meta", index=False)


def write_plaintext_two_rows(path: str, audio_sig: np.ndarray, li_sig: np.ndarray):
    """
    Writes two lines:
      line1: audio samples (space-separated)
      line2: LI channel samples (space-separated)
    Intended for copy/paste into other programs.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    # Use a concise, copy/paste-friendly numeric format
    # %.10g keeps good precision without being huge; change if needed.
    fmt = "%.10g"

    with open(path, "w", encoding="utf-8") as f:
        f.write(" ".join(fmt % v for v in np.asarray(audio_sig).reshape(-1)))
        f.write("\n")
        f.write(" ".join(fmt % v for v in np.asarray(li_sig).reshape(-1)))
        f.write("\n")


def write_wavs(out_base: str,
              audio_sig: np.ndarray, audio_srate: int,
              li_sig: np.ndarray, li_srate: int,
              combined: bool,
              wav_format: str,
              align: str,
              resample_li: bool):
    """
    WAV output:
      - resample_li=True: LI -> audio SRATE using poly_reflect
      - resample_li=False: keep LI at its own SRATE
        - combined=True: NOT allowed (different sample rates). We fall back to separate.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_base)) or ".", exist_ok=True)

    def write_one(path, srate, data_1d_or_2d):
        if wav_format == "int16":
            if data_1d_or_2d.ndim == 1:
                out = normalize_to_int16(data_1d_or_2d)
            else:
                out = np.column_stack([normalize_to_int16(data_1d_or_2d[:, i]) for i in range(data_1d_or_2d.shape[1])])
            wavwrite(path, srate, out)
        else:
            wavwrite(path, srate, data_1d_or_2d.astype(np.float32, copy=False))

    if not resample_li:
        # Different sample rates => cannot meaningfully combine into a single WAV.
        if combined:
            combined = False  # force separate
        write_one(out_base + "_AUDIO.wav", audio_srate, audio_sig.reshape(-1))
        write_one(out_base + "_LI_ch.wav", li_srate, li_sig.reshape(-1))
        return

    # Resample LI to audio rate
    li_rs = upsample_poly_reflect(li_sig, fs_in=li_srate, fs_out=audio_srate, pad_sec=0.25)

    # Align lengths (only needed when combining or when you want same-length mono files)
    n_a, n_l = audio_sig.size, li_rs.size
    if n_a != n_l:
        if align == "trim":
            n = min(n_a, n_l)
            audio_al = audio_sig[:n]
            li_al = li_rs[:n]
        else:
            n = max(n_a, n_l)
            audio_al = np.pad(audio_sig, (0, n - n_a))
            li_al = np.pad(li_rs, (0, n - n_l))
    else:
        audio_al, li_al = audio_sig, li_rs

    if combined:
        stereo = np.column_stack([audio_al, li_al])
        write_one(out_base + "_combined.wav", audio_srate, stereo)
    else:
        write_one(out_base + "_AUDIO.wav", audio_srate, audio_al.reshape(-1))
        write_one(out_base + "_LI_ch_rs.wav", audio_srate, li_al.reshape(-1))


# =========================
# Tkinter GUI
# =========================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MAT Extractor: Audio + LI channel")
        self.geometry("860x560")

        self.mode = tk.StringVar(value="file")  # file | folder
        self.input_path = tk.StringVar(value="")
        self.output_dir = tk.StringVar(value="")

        self.varname = tk.StringVar(value="")  # optional
        self.audio_index = tk.IntVar(value=0)
        self.li_index = tk.IntVar(value=4)
        self.li_channel = tk.IntVar(value=3)

        self.do_excel = tk.BooleanVar(value=True)
        self.do_text = tk.BooleanVar(value=True)
        self.do_wav = tk.BooleanVar(value=True)

        self.wav_combined = tk.BooleanVar(value=False)
        self.wav_resample_li = tk.BooleanVar(value=False)
        self.wav_format = tk.StringVar(value="float32")  # int16 | float32
        self.align_mode = tk.StringVar(value="trim")   # trim | pad

        self._build()

    def _build(self):
        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)

        # Input
        input_box = ttk.LabelFrame(frm, text="Input", padding=10)
        input_box.pack(fill="x")

        ttk.Radiobutton(input_box, text="Single .mat file", variable=self.mode, value="file").grid(row=0, column=0, sticky="w", padx=(0, 12))
        ttk.Radiobutton(input_box, text="Folder of .mat files", variable=self.mode, value="folder").grid(row=0, column=1, sticky="w")

        ttk.Entry(input_box, textvariable=self.input_path).grid(row=1, column=0, columnspan=2, sticky="ew", pady=6)
        ttk.Button(input_box, text="Browse…", command=self.browse_input).grid(row=1, column=2, padx=8)

        ttk.Label(input_box, text="Output folder (optional):").grid(row=2, column=0, sticky="w")
        ttk.Entry(input_box, textvariable=self.output_dir).grid(row=3, column=0, columnspan=2, sticky="ew", pady=6)
        ttk.Button(input_box, text="Browse…", command=self.browse_output).grid(row=3, column=2, padx=8)

        input_box.columnconfigure(0, weight=1)
        input_box.columnconfigure(1, weight=1)

        # Extraction settings
        ext_box = ttk.LabelFrame(frm, text="Extraction settings", padding=10)
        ext_box.pack(fill="x", pady=(10, 0))

        ttk.Label(ext_box, text="Variable name (optional):").grid(row=0, column=0, sticky="w")
        ttk.Entry(ext_box, textvariable=self.varname, width=18).grid(row=0, column=1, sticky="w", padx=8)

        ttk.Label(ext_box, text="Audio struct index:").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(ext_box, from_=0, to=999, textvariable=self.audio_index, width=6).grid(row=0, column=3, sticky="w", padx=8)

        ttk.Label(ext_box, text="LI struct index:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(ext_box, from_=0, to=999, textvariable=self.li_index, width=6).grid(row=1, column=1, sticky="w", padx=8, pady=(6, 0))

        ttk.Label(ext_box, text="LI channel (1-based):").grid(row=1, column=2, sticky="w", pady=(6, 0))
        ttk.Spinbox(ext_box, from_=1, to=999, textvariable=self.li_channel, width=6).grid(row=1, column=3, sticky="w", padx=8, pady=(6, 0))

        # Outputs
        out_box = ttk.LabelFrame(frm, text="Outputs", padding=10)
        out_box.pack(fill="x", pady=(10, 0))

        ttk.Checkbutton(out_box, text="Excel (.xlsx): audio + LI_ch + meta", variable=self.do_excel).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(out_box, text="Plaintext (.txt): 2 rows, space-separated (audio row, LI row)", variable=self.do_text).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(out_box, text="WAV (.wav)", variable=self.do_wav).grid(row=2, column=0, sticky="w")

        wav_opts = ttk.Frame(out_box)
        wav_opts.grid(row=3, column=0, sticky="w", pady=(6, 0))

        ttk.Checkbutton(wav_opts, text="Resample LI to audio rate (poly_reflect)", variable=self.wav_resample_li).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(wav_opts, text="Write one combined WAV (2 channels) [requires resampling]", variable=self.wav_combined).grid(row=1, column=0, sticky="w", pady=(2, 0))

        row2 = ttk.Frame(out_box)
        row2.grid(row=4, column=0, sticky="w", pady=(6, 0))

        ttk.Label(row2, text="WAV format:").grid(row=0, column=0, padx=(0, 4))
        ttk.Combobox(row2, textvariable=self.wav_format, values=["int16", "float32"], width=8, state="readonly").grid(row=0, column=1)

        ttk.Label(row2, text="Align (when resampling):").grid(row=0, column=2, padx=(16, 4))
        ttk.Combobox(row2, textvariable=self.align_mode, values=["trim", "pad"], width=8, state="readonly").grid(row=0, column=3)

        ttk.Label(out_box, text="Note: If you disable resampling, WAV output is written as separate files (different sample rates).").grid(
            row=5, column=0, sticky="w", pady=(6, 0)
        )

        # Run / progress / log
        run_box = ttk.Frame(frm)
        run_box.pack(fill="x", pady=(12, 0))

        self.btn_run = ttk.Button(run_box, text="Run", command=self.run)
        self.btn_run.pack(side="left")

        self.prog = ttk.Progressbar(run_box, mode="determinate", length=260)
        self.prog.pack(side="left", padx=12)

        self.lbl_status = ttk.Label(run_box, text="")
        self.lbl_status.pack(side="left")

        log_box = ttk.LabelFrame(frm, text="Log", padding=8)
        log_box.pack(fill="both", expand=True, pady=(12, 0))

        self.txt_log = tk.Text(log_box, height=10, wrap="word")
        self.txt_log.pack(fill="both", expand=True)

    def log(self, msg: str):
        self.txt_log.insert("end", msg + "\n")
        self.txt_log.see("end")
        self.update_idletasks()

    def browse_input(self):
        if self.mode.get() == "file":
            p = filedialog.askopenfilename(
                title="Select .mat file",
                filetypes=[("MAT files", "*.mat"), ("All files", "*.*")]
            )
        else:
            p = filedialog.askdirectory(title="Select folder containing .mat files")
        if p:
            self.input_path.set(p)
            if not self.output_dir.get():
                self.output_dir.set(os.path.dirname(os.path.abspath(p)) if self.mode.get() == "file" else os.path.abspath(p))

    def browse_output(self):
        p = filedialog.askdirectory(title="Select output folder")
        if p:
            self.output_dir.set(p)

    def get_input_files(self):
        p = self.input_path.get().strip()
        if not p:
            return []
        if self.mode.get() == "file":
            return [p] if os.path.isfile(p) and p.lower().endswith(".mat") else []
        else:
            if not os.path.isdir(p):
                return []
            return [os.path.join(p, n) for n in sorted(os.listdir(p)) if n.lower().endswith(".mat")]

    def run(self):
        files = self.get_input_files()
        if not files:
            messagebox.showerror("MAT Extractor", "No valid input .mat file(s) selected.")
            return

        if not (self.do_excel.get() or self.do_text.get() or self.do_wav.get()):
            messagebox.showerror("MAT Extractor", "Select at least one output: Excel, Plaintext, or WAV.")
            return

        outdir = self.output_dir.get().strip()
        if not outdir:
            outdir = os.path.dirname(os.path.abspath(files[0]))
            self.output_dir.set(outdir)
        os.makedirs(outdir, exist_ok=True)

        self.btn_run.config(state="disabled")
        self.prog["value"] = 0
        self.prog["maximum"] = len(files)
        self.lbl_status.config(text="Running…")
        self.txt_log.delete("1.0", "end")

        var = self.varname.get().strip() or None

        try:
            for i, mat_path in enumerate(files, start=1):
                stem = os.path.splitext(os.path.basename(mat_path))[0]
                self.log(f"=== {stem} ===")
                self.log(f"Input: {mat_path}")

                try:
                    res = extract_audio_and_li_ch(
                        mat_path=mat_path,
                        varname=var,
                        audio_index=int(self.audio_index.get()),
                        li_index=int(self.li_index.get()),
                        li_channel_1based=int(self.li_channel.get()),
                    )

                    audio = res["audio"]
                    li = res["li"]

                    self.log(f"Var: {res['var']}")
                    self.log(f"Audio: NAME={audio['name']} SRATE={audio['srate']} n={audio['signal'].size}")
                    self.log(f"LI:    NAME={li['name']} SRATE={li['srate']} n={li['signal'].size} ch={li['channel']}")

                    base_out = os.path.join(outdir, stem)

                    # Excel (original signals, no resample)
                    if self.do_excel.get():
                        xlsx_path = base_out + ".xlsx"
                        write_excel(xlsx_path, audio["signal"], li["signal"], audio["srate"], li["srate"])
                        self.log(f"Wrote Excel: {xlsx_path}")

                    # Plaintext: two rows, original signals (no resample)
                    if self.do_text.get():
                        txt_path = base_out + ".txt"
                        write_plaintext_two_rows(txt_path, audio["signal"], li["signal"])
                        self.log(f"Wrote Text:  {txt_path}")

                    # WAV
                    if self.do_wav.get():
                        resample_li = bool(self.wav_resample_li.get())
                        combined = bool(self.wav_combined.get())
                        if combined and not resample_li:
                            self.log("Note: combined WAV requires resampling; writing separate WAVs instead.")
                        write_wavs(
                            out_base=base_out,
                            audio_sig=audio["signal"],
                            audio_srate=audio["srate"],
                            li_sig=li["signal"],
                            li_srate=li["srate"],
                            combined=combined,
                            wav_format=self.wav_format.get(),
                            align=self.align_mode.get(),
                            resample_li=resample_li,
                        )
                        if resample_li and combined:
                            self.log(f"Wrote WAV:   {base_out}_combined.wav")
                        else:
                            # either separate due to choice or forced by sample-rate mismatch
                            self.log(f"Wrote WAV:   {base_out}_AUDIO.wav")
                            if resample_li:
                                self.log(f"Wrote WAV:   {base_out}_LI_ch_rs.wav")
                            else:
                                self.log(f"Wrote WAV:   {base_out}_LI_ch.wav")

                except Exception as e:
                    self.log(f"ERROR: {type(e).__name__}: {e}")
                    self.log(traceback.format_exc())

                self.prog["value"] = i
                self.update_idletasks()

            self.lbl_status.config(text="Done.")
            messagebox.showinfo("MAT Extractor", f"Done.\nProcessed {len(files)} file(s).\nOutput: {outdir}")

        finally:
            self.btn_run.config(state="normal")


if __name__ == "__main__":
    App().mainloop()


