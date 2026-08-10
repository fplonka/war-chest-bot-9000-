# v5_async_guard_90s

We were testing an asynchronous version of the two-lane memory guard. An
oversized search was held on its selected lane while that lane and one helper
drained, and ordinary submissions continued on the other three lanes. Held
searches recorded their weight version at submission time so later training
updates could not silently change which network they used.

This run exposed a scheduler deadlock and is not a performance result. It
reached the 90-second stop at 91,136 solves, or 1,011.9 solves/s, with five
oversized searches completed. During final draining both GPUs became idle and
the process made no progress, so it was terminated after diagnosis. A second
held search could synchronously wait for the route worker to acknowledge it;
the first search's release command could then be queued behind that wait, while
the route worker was still waiting for the first search to finish.

There were no CUDA allocation errors before the deadlock. Sampled peak memory
was 22,137 MiB on GPU 0 and 19,915 MiB on GPU 1. The immediate correction is to
buffer each held-search acknowledgement, allowing the lane to keep consuming
commands until the earlier release arrives. This trace must not be compared as
a completed benchmark; it is retained so the failed concurrency design and its
symptom are not rediscovered.
