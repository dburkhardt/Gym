# MolPuzzle Stage2 H-NMR image-transport repair variant

Source https://github.com/KehanGuo2/MolPuzzle at4b62e754ddb5c9b3a5c870a6221b55d71c70ebd2. Stage2 has2808 H-NMR questions;2592 have matching published image files after trimming molecule-ID whitespace. The216 missing-image records are excluded before seeded sampling and are not scored as failures. Stage1/3 and other spectra are not included.

Creator `analyze_image_with_prompt_gpt` embeds base64 inside ordinary text. This explicitly repaired variant supplies the same source question, H-NMR instruction and source image as a proper input_image part. It is not claimed to reproduce the creator's original image transport. Correct structure SMILES and label stay out of model input. Creator binary scorer checks yes substring first, then no, otherwise -1. Retained behavior may parse unrelated prose incorrectly; inspect validity and do not interpret parser correctness as scientific correctness.

Stage images from repository Data/Spectrum_images (or extract its tracked zip). Prepare with deterministic module; standard CAMPAIGN_UPSTREAM_REPO/CAMPAIGN_SAMPLE_JSONL staging then config.yaml. Samples stratify Yes/No labels, so this balanced pilot is not a population prevalence estimate. Creator accuracy is reproduced; macroF1 needs aggregation across the batch, not per-row averaging. No model run is claimed by controls.
