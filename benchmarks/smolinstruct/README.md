# SMolInstruct held-out formula conversion

Creator source OSU-NLP-Group/LlaSMol43ab5fccd14514ddf756534d18a6917e7e11d0ae, dataset osunlp/SMolInstructb3e028007c6abc043f62ed0aef4824dc9e86bacc. Uses full official instruction_tuning/test sample manifests for name_conversion-i2f and name_conversion-s2f,2993 each. Raw test records, exact selected template IDs and core input/output tags follow the custom dataset loader. No training records or test_subset shortcut.

Scoring calls creator tag extraction and calculate_formula_metrics(element_match), which compares element counts including charge while ignoring atom ordering. Gold formula is verifier-only. Other12 tasks and model-specific extraction heuristics are not claimed. Twenty tagged gold and empty controls pass.

Set SMOLINSTRUCT_ZIP to the692MB pinned archive, run deterministic preparation, then standard source/sample native staging. Scorer needs numpy but not the full tokenizer or text-generation dependency stack because exact relevant source functions are isolated via AST. Archive SHA256 is8598664507b0b7768ed34f70db194625e60fc67a14e6fa4fa415952d52c45150. Remote scoring needs pinned source plus prepared JSONL, not the full archive.
