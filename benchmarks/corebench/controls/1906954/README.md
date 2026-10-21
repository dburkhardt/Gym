This Oracle integration control runs the original two R Markdown renders through
`bash run.sh`. Copy this script, the maintained `rendered_subgroups.py`, and a
`source.sha256` generated from the frozen 21-file manifest into the task's solution
directory. Keep `/workspace` and `/results` in task artifacts so a failed render
retains partial output. No reference answers belong in the solution directory.

The historical workflow used 2 CPUs and 8 GiB; prepare execution with that memory
allocation while preserving the image qualification's separate 4 GiB receipt.
Keep the original 3600-second agent and 120-second verifier limits. The pinned
Oracle phase helper also accounts for 600-second environment start and 360-second
setup, totaling 4680 seconds before the explicit operational allowance. This is a
scientific reproducibility and native scorer control, not model performance.
