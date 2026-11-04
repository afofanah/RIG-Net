"""panels.py - put (a), (b), ... beneath each sub-panel rather than in its title."""
import re
import numpy as np


def label_axes(fig, axes, labels=None, fontsize=9.2, strip_titles=True, relayout=True,
               inline=False, align="center"):
    """Move the panel letter out of the title and under the axes.

    A blank line is appended to the axis label so the layout reserves room, and the
    letter is drawn into that gap. The axis label keeps its own font size.
    """
    axes = [a for a in np.ravel(np.asarray(axes, dtype=object)).tolist() if a is not None]
    if labels is None:
        labels = [f"({chr(97 + i)})" for i in range(len(axes))]
    for ax, lab in zip(axes, labels):
        if strip_titles and ax.get_title():
            ax.set_title(re.sub(r"^\(\s*[a-z0-9]{1,2}\s*\)\s*", "", ax.get_title()))
        xl = ax.get_xlabel()
        if inline:                      # letter inside the axis label, above anything below it
            ax.set_xlabel(f"{xl}\n{lab}" if xl else lab)
        else:
            ax.set_xlabel(f"{xl}\n " if xl else " ")
    if relayout:
        try:
            fig.tight_layout()
        except Exception:
            pass
    if inline:
        return
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    for ax, lab in zip(axes, labels):
        bb = ax.get_tightbbox(rend).transformed(fig.transFigure.inverted())
        box = ax.get_position()           # the plot area, excluding tick labels
        if align == "left":
            x, ha = box.x0, "left"
        else:
            x, ha = (box.x0 + box.x1) / 2, "center"
        fig.text(x, bb.y0, lab, ha=ha, va="bottom", fontsize=fontsize, fontweight="bold")
