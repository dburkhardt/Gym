# MicroVQA++ released demo only

Official source https://github.com/ieellee/MicroVQA-PlusPlus pinned57a2272e9116631db3ca136518b171cc38747f6b releases15 training/demo records. Full dataset and evaluation implementation remain unchecked TODOs. This adapter is a transport/strict-choice diagnostic, not the published6000-item benchmark or reported metric reproduction. No20-example run is possible without repetition.

Only image bytes, original question and choices enter solver prompts. Released correct_answer, correct_index, cot and image_caption remain verifier-side; caption/CoT can reveal the answer. A minimal explicit number-only response instruction and strict number parser are a documented variant. One-based gold labels are checked against the matching complete answer text for every released row. Invalid/empty output scores0; failed execution must retain separate status. Seed20261020 fixes5/15 subsets, no resampling by answer.
