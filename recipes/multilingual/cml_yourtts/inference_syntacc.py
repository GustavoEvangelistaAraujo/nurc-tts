import os
import shutil
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
from TTS.utils.synthesizer import Synthesizer

OUTPUT_DIR = "/workspace/data/data/inference_results"
DATA_ROOT = "/workspace/data/data/inference_data"
SAMPLE_RATE = 24000

OVERWRITE = os.environ.get("OVERWRITE", "0").lower() in {"1", "true", "yes", "y"}
MAX_ROWS = os.environ.get("MAX_ROWS")
MAX_ROWS = int(MAX_ROWS) if MAX_ROWS else None

USE_CUDA = torch.cuda.is_available()

CSV_PATH = os.environ.get(
    "CSV_PATH",
    f"{DATA_ROOT}/_nurc_tts_SELECIONADOS_normalizado.csv"
)

# =============================================================================
# Model config
# You can override these from Docker with:
# -e MODEL_CONFIG_PATH=...
# -e MODEL_CHECKPOINT_PATH=...
# -e MODEL_INFO=...

MODEL_CONFIGS = [
    {
        "config_path": os.environ.get(
            "MODEL_CONFIG_PATH",
            "/workspace/data/checkpoints/YourTTS-Syntacc-PT_NURC-September-02-2026_02+35AM-bf39f425/config.json",
        ),
        "model_path": os.environ.get(
            "MODEL_CHECKPOINT_PATH",
            "/workspace/data/checkpoints/YourTTS-Syntacc-PT_NURC-September-02-2026_02+35AM-bf39f425/checkpoint_750000.pth",
        ),
        "model_info": os.environ.get(
            "MODEL_INFO",
            "syntacc-tornado-750Ksteps.wav",
        ),
    }
]


# =============================================================================
# Helpers
# =============================================================================

def get_city_from_inquiry(inquiry):
    inquiry = str(inquiry).strip().upper()

    if inquiry.startswith("SP_") or inquiry.startswith("NURC_SP_"):
        return "sao_paulo"

    if inquiry.startswith("RE_") or inquiry.startswith("NURC_RE_"):
        return "recife"

    raise ValueError(f"Could not infer city/accent from inquiry: {inquiry}")


def get_audio_path(inquiry, segment):
    """
    Reference WAV lookup.

    This version checks OUTPUT_DIR first because your older inference setup likely
    staged reference WAVs there.
    """
    city = get_city_from_inquiry(inquiry)
    segment = str(segment).strip()

    candidates = [
        os.path.join(DATA_ROOT, segment),
        os.path.join(OUTPUT_DIR, segment),
        os.path.join(OUTPUT_DIR, city, segment),
        os.path.join(OUTPUT_DIR, city, "test", segment),
        os.path.join(DATA_ROOT, city, "test", segment),
        os.path.join(DATA_ROOT, city, "wavs", segment),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path, city

    raise FileNotFoundError(
        "Could not find reference audio for "
        f"inquiry={inquiry}, segment={segment}. Tried:\n" +
        "\n".join(candidates)
    )


def safe_model_info(model_info):
    model_info = str(model_info).strip()
    if not model_info.endswith(".wav"):
        model_info += ".wav"
    return model_info


def save_wav(synthesizer, wav, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        synthesizer.save_wav(wav, out_path)
    except Exception:
        sf.write(out_path, wav, SAMPLE_RATE)


def load_synthesizer(model_config):
    config_path = model_config["config_path"]
    model_path = model_config["model_path"]

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Missing config file: {config_path}")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Missing checkpoint file: {model_path}")

    model_dir = os.path.dirname(model_path)

    speakers_file = os.path.join(model_dir, "speakers.pth")
    language_ids_file = os.path.join(model_dir, "language_ids.json")

    kwargs = {
        "tts_checkpoint": model_path,
        "tts_config_path": config_path,
        "use_cuda": USE_CUDA,
    }

    if os.path.isfile(speakers_file):
        kwargs["tts_speakers_file"] = speakers_file

    if os.path.isfile(language_ids_file):
        kwargs["tts_languages_file"] = language_ids_file

    print("=" * 80)
    print("Loading model")
    print(f"config_path: {config_path}")
    print(f"model_path: {model_path}")
    print(f"model_info: {model_config['model_info']}")
    print(f"use_cuda: {USE_CUDA}")

    return Synthesizer(**kwargs)


def synthesize_with_fallback(synthesizer, text, speaker_wav, language_name):
    """
    Tries YourTTS-style inference first.
    Falls back if the installed Synthesizer version has slightly different args.
    """
    try:
        return synthesizer.tts(
            text=text,
            speaker_wav=speaker_wav,
            language_name=language_name,
            split_sentences=False,
        )
    except TypeError:
        try:
            return synthesizer.tts(
                text=text,
                speaker_wav=speaker_wav,
                language_name=language_name,
            )
        except TypeError:
            return synthesizer.tts(
                text=text,
                speaker_wav=speaker_wav,
            )


# =============================================================================
# Main
# =============================================================================

def synthesize_test():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print("INFERENCE SETTINGS")
    print(f"CSV_PATH: {CSV_PATH}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"SAMPLE_RATE: {SAMPLE_RATE}")
    print(f"OVERWRITE: {OVERWRITE}")
    print(f"MAX_ROWS: {MAX_ROWS}")
    print(f"USE_CUDA: {USE_CUDA}")

    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(f"Missing CSV_PATH: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    required_columns = {"inquiry", "segment", "speaker", "text"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    if MAX_ROWS is not None:
        df = df.head(MAX_ROWS)

    manifest_rows = []

    for model_config in MODEL_CONFIGS:
        model_info = safe_model_info(model_config["model_info"])
        synthesizer = load_synthesizer(model_config)

        for idx, row in df.iterrows():
            inquiry = row["inquiry"]
            segment = row["segment"]
            text = str(row["text"]).strip()

            if not text:
                print(f"Skipping row {idx}: empty text")
                continue

            try:
                audio_path, city = get_audio_path(inquiry, segment)
            except Exception as exc:
                print(f"Skipping row {idx}: {exc}")
                continue

            language_name = city

            reference_stem = Path(audio_path).stem

            synth_filename = f"{reference_stem}-{model_info}"
            gt_filename = f"{reference_stem}-gt.wav"
            txt_filename = f"{reference_stem}-gt.txt"

            synth_path = os.path.join(OUTPUT_DIR, synth_filename)
            gt_path = os.path.join(OUTPUT_DIR, gt_filename)
            txt_path = os.path.join(OUTPUT_DIR, txt_filename)

            print("=" * 80)
            print(f"row: {idx}")
            print(f"inquiry: {inquiry}")
            print(f"segment: {segment}")
            print(f"city/language: {city}")
            print(f"reference: {audio_path}")
            print(f"text: {text}")
            print(f"synth_path: {synth_path}")

            if os.path.exists(synth_path) and not OVERWRITE:
                print(f"Exists, skipping synth: {synth_path}")
            else:
                wav = synthesize_with_fallback(
                    synthesizer=synthesizer,
                    text=text,
                    speaker_wav=audio_path,
                    language_name=language_name,
                )
                save_wav(synthesizer, wav, synth_path)

            if not os.path.exists(gt_path) or OVERWRITE:
                shutil.copy2(audio_path, gt_path)

            if not os.path.exists(txt_path) or OVERWRITE:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text + "\n")

            manifest_rows.append(
                {
                    "row_index": idx,
                    "inquiry": inquiry,
                    "segment": segment,
                    "city": city,
                    "language_name": language_name,
                    "text": text,
                    "reference_audio": audio_path,
                    "synth_audio": synth_path,
                    "ground_truth_copy": gt_path,
                    "transcription": txt_path,
                    "model_config": model_config["config_path"],
                    "model_checkpoint": model_config["model_path"],
                    "model_info": model_info,
                }
            )

    manifest_path = os.path.join(OUTPUT_DIR, "inference_manifest.csv")

    if manifest_rows:
        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    print("=" * 80)
    print("Done.")
    print(f"Rows processed: {len(manifest_rows)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    synthesize_test()