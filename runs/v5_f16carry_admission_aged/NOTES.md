# v5_f16carry_admission_aged

We were checking the aged, fixed-weight production stream after two related
memory changes: retained belief snapshots now use float16, and the scheduler's
job-size estimate now matches the mixed-width CUDA arena instead of the old
all-float32 layout. This used the same post-Greedy checkpoint, seed, two RTX
3090s, 36 builders, 128 actors per builder, 32 in-flight solves per builder,
and five lanes per card as `v5_aged_fast16_head_l5`. No optimizer ran.

The stream completed 247,808 solves in 180.15 seconds before stopping, or
1,375.6 solves/s. That is only 0.8% above the 1,364.6/s control over the full
three minutes, but the aged tail improved more: from 120 to 180 seconds it ran
at 1,243/s versus 1,109/s in the control. Isolated large-search routes fell
from 329 to 289, and post-stop drain fell from 22.2 to 9.7 seconds. There were
857 completed games, 1,576 node caps, no exact fallback, and no dropped work.

Average GPU use was 69.5% and 69.4%, up from 65.6% and 64.4% in the control.
Peak memory was 22,293 and 21,895 MiB. The initially written 9,813/9,095 MiB
figures came from the same string-comparison bug in the CSV summary and were
corrected from the saved samples. Float16 carry did not lower allocator high
water enough to make five lanes comfortably safe by itself.
The live games are not an exact A/B because float16 beliefs can change sampled
actions; the identical frozen tape was 1.6% faster for float16 carry and 0.9%
faster for corrected admission. At this point both changes pass the fast and
precise GPU correctness suites and are retained. They improve the expensive
tail and memory safety, but generation still misses the 1,400/s target and the
last real training gate was only 838 balanced solves/s.
