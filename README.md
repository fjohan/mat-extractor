# MAT Extractor: Audio + LI Channel

A desktop GUI tool for extracting audio and articulatory (LI) data from MATLAB v5 `.mat` files, with optional export to **Excel**, **plain text**, and **WAV**.

This tool is designed for speech and articulography workflows where:
- One struct contains **audio** (e.g. 16 kHz)
- Another struct contains **articulatory sensor data** (e.g. 200 Hz)
- You want easy export and optional resampling for alignment

---

## Features

- ✅ Tkinter-based GUI (no command line required)
- ✅ Process a **single `.mat` file** or a **folder of `.mat` files**
- ✅ Extract:
  - Audio signal from one struct (default index `0`)
  - LI sensor data from another struct (default index `4`)
  - Select **specific LI channel** (1-based)
- ✅ Optional outputs:
  - **Excel (.xlsx)** – audio, LI channel, and metadata
  - **Plaintext (.txt)** – two rows of numbers for copy/paste
  - **WAV (.wav)** – audio and LI (resampled or not)
- ✅ High-quality resampling using **polyphase filtering with reflection padding**
- ✅ Cross-platform (Linux, Windows, macOS)

---

## Expected `.mat` File Structure

The tool expects a MATLAB v5 `.mat` file containing a **struct array** like:

```
T057(0): AUDIO
  - NAME
  - SRATE
  - SIGNAL (1D)

T057(4): LI
  - NAME
  - SRATE
  - SIGNAL (N x channels)
```

The struct array must contain the fields:
- `NAME`
- `SRATE`
- `SIGNAL`

The default configuration assumes:
- Audio at index `0`
- LI at index `4`
- LI channel `3`

All of these can be changed in the GUI.

---

## GUI Overview

### Input
- **Single .mat file** – select one file
- **Folder of .mat files** – batch process all `.mat` files in a folder
- **Output folder (optional)** – defaults to the input file/folder location

### Extraction Settings
- **Variable name (optional)**  
  Leave empty to auto-detect the struct array.
- **Audio struct index**  
  Index of the audio struct (default `0`)
- **LI struct index**  
  Index of the LI struct (default `4`)
- **LI channel (1-based)**  
  Channel number inside the LI signal matrix

### Outputs

#### Excel (.xlsx)
- Sheet `audio`: audio samples
- Sheet `LI_ch`: selected LI channel
- Sheet `meta`: sample rates and lengths

#### Plaintext (.txt)
- **Line 1**: audio samples (space-separated)
- **Line 2**: LI channel samples (space-separated)
- Designed for easy copy/paste into other tools

#### WAV (.wav)
Options:
- **Resample LI to audio rate** (recommended for alignment)
- **Write one combined WAV** (2 channels: audio + LI)
- Or write **separate WAV files** if resampling is disabled

WAV formats:
- `float32` (default, safest for scientific data)
- `int16` (PCM audio)

Alignment options (when resampling):
- `trim` – cut to the shortest signal
- `pad` – zero-pad to the longest signal

---

## Output Files

For an input file `T001.mat`, outputs may include:

```
T001.xlsx
T001.txt
T001_combined.wav
T001_AUDIO.wav
T001_LI_ch.wav
T001_LI_ch_rs.wav
```

Exact files depend on selected options.

---

## Recommended Defaults

For most articulography use cases:
- ✅ Excel: ON
- ✅ Plaintext: ON
- ✅ WAV: ON
- ❌ Combined WAV: OFF
- ❌ Resampling: OFF (enable only when alignment is needed)
- WAV format: **float32**

---

## Installation (Source Version)

If running from source:

```bash
pip install numpy scipy
pip install pandas openpyxl   # only if Excel output is needed
python mat_extractor_gui.py
```

---

## Standalone Builds

Pre-built standalone executables can be created using **Nuitka** and **GitHub Actions**:
- No Python installation required on client machines
- Available for Windows, macOS, and Linux

(See build workflow documentation if you are packaging the app.)

---

## Notes & Limitations

- MATLAB **v5** `.mat` files are supported
- MATLAB **v7.3 (HDF5)** files are not currently supported
- Combined WAV output requires resampling (same sample rate)
- Large files may produce large Excel or text outputs

---

## License / Usage

Internal or research use.  
Modify and redistribute as needed for your project.

---

If you need:
- Different sensors
- Multiple LI channels
- Automatic detection by `NAME` instead of index
- CSV/JSON output

…the tool can be easily extended.
