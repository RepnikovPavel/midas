"""pandaset_pipe: streaming download + unpack + npz conversion pipeline for PandaSet.

Layout of produced data (per disk):
    <root>/pandaset/<SEQ>/...                 raw extracted files (original layout minus top dir)
    <root>/pandaset_npz/sweep_<SEQ>/          converted per-frame npz (fp16 point clouds, ego frame)

Ego frame convention everywhere in the converted data: X forward, Y left, Z up.
"""

__version__ = "1.0.0"
