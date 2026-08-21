# Reassess later

**Growth stops at the round boundary.** A solve does not expand nodes past the
draw that starts the next round. The reason is cost, not soundness: the draw
re-broadens the config support, so every node beyond the boundary carries
several times the configs of one before it, and the value network is defined at
exactly that boundary state anyway. SoG itself has no depth limit; this is
DeepStack's street-boundary device. Revisit once a solve is instrumented for
frontier depth and frontier support — if past-boundary nodes turn out cheap, or
if the network prices post-draw states badly, take the stop out.
