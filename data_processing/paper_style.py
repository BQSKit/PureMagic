"""Shared matplotlib style for the journal paper's figures.

Every figure in the paper is set at exactly one column width and included
with `width=\\columnwidth`, so the PDF scales it 1:1 and the font sizes below
are the sizes that reach the page. That is the whole point of fixing the
width here: a figure drawn at 16 inches and squeezed into a 3.2-inch column
has its 10 pt labels rendered at 2 pt.

Geometry comes from the document itself, not from the journal. Quantum states
plainly that "there are no typesetting, format or length constraints" and
that `quantumarticle` is optional, so there is no external specification to
meet. Measured from the class with `\\the\\columnwidth` under
`[twocolumn,a4paper]`:

    columnwidth  231.84843 pt   = 3.208 in      <- every figure is this wide
    textwidth    483.69687 pt   = 6.693 in
    columnsep     20 pt
    body font     10 pt
    captions      \\small + \\sffamily = 9 pt sans

Figure text is therefore sans-serif, to match the captions, and set at 8 pt
and below so it never competes with them.

Usage:

    from paper_style import use_paper_style, figsize, PALETTE
    use_paper_style()
    fig, ax = plt.subplots(figsize=figsize())
    ...
    fig.savefig(path)   # no bbox_inches="tight" -- see save() below
"""

import matplotlib

# One column, in inches: 231.84843 pt / 72.27 pt-per-inch.
COLUMN_IN = 3.208
# Both columns, for the rare figure that genuinely cannot fit in one. The
# paper does not currently use it; every figure is single-column by choice.
TEXT_IN = 6.693

# Default height as a fraction of the width. Shallow enough that a figure
# plus its caption does not dominate a column.
DEFAULT_RATIO = 0.78

# Categorical sequence, assigned in this fixed order and never cycled.
# Validated for protan/deutan/tritan separation against a light surface:
# worst adjacent pair dE 8.2 (deutan), normal-vision floor 18.6, all six
# inside the lightness band and above the chroma floor. #999933 sits a shade
# under 3:1 contrast, which is why every series also carries a legend entry
# or a direct label rather than relying on colour alone.
PALETTE = [
    "#2171b5",  # blue
    "#d95f02",  # orange
    "#7570b3",  # purple
    "#117733",  # green
    "#999933",  # olive
    "#cc6677",  # rose
]

# For a reference curve measured against models, rather than a category.
REFERENCE = "#1a1a1a"


def figsize(ratio=DEFAULT_RATIO, width=COLUMN_IN):
    """(width, height) in inches, one column wide by default."""
    return (width, width * ratio)


def figsize_panel(n, ratio=0.95, wide=True):
    """(width, height) for one panel of an `n`-panel row.

    A panel drawn at full column width and then included in a
    `0.49\\columnwidth` subfigure is rescaled to 49%, which halves its font
    sizes -- the reason several multi-panel figures came out unreadable. Draw
    each panel at the width it is actually placed at instead, so the scale
    stays 1:1 and the point sizes hold.

    `wide=True` spreads the row across both columns, which is what makes a
    three-panel row viable at all: across one column each panel would be
    1.0 inch.
    """
    # The fractions match the subfigure widths declared in the .tex exactly,
    # so the panel is drawn at the width it is placed at and nothing rescales.
    # The shortfall from 1/n is the inter-panel gap \hfill introduces.
    frac = {2: 0.49, 3: 0.32, 4: 0.24}[n]
    w = (TEXT_IN if wide else COLUMN_IN) * frac
    return (w, w * ratio)


def figsize_wide(ratio=0.38):
    """(width, height) for a figure spanning both columns.

    For a two-panel figure set side by side: squeezing two panels into one
    3.2-inch column leaves 1.6 inches each, which is too narrow for readable
    axes, so those go in a `figure*` at `width=\\textwidth` instead. The
    shallower default ratio keeps a full-width figure from eating half a
    page. Single-panel figures stay one column wide.
    """
    return (TEXT_IN, TEXT_IN * ratio)


def use_paper_style():
    """Apply the paper's rcParams. Call once, before creating figures."""
    matplotlib.rcParams.update({
        # Sans-serif, matching the \small\sffamily captions.
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.titlesize": 8,

        # Thin marks; recessive axes and grid.
        "lines.linewidth": 1.2,
        "lines.markersize": 3.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.25,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE),

        # Legends sit inside the axes without a heavy box.
        "legend.frameon": False,
        "legend.handlelength": 2.2,
        "legend.borderaxespad": 0.4,
        "legend.labelspacing": 0.3,

        "figure.figsize": figsize(),
        "figure.dpi": 400,
        "savefig.dpi": 400,
        # Deliberately NOT savefig.bbox="tight": that trims to the drawn
        # content and changes the output width, so the figure would no longer
        # be exactly one column and \columnwidth would rescale it -- which is
        # the problem this module exists to prevent. Use constrained layout
        # instead, which fits the content to a fixed canvas.
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
        "figure.constrained_layout.hspace": 0.03,
        "figure.constrained_layout.wspace": 0.03,
    })


def save(fig, path):
    """Write `fig` at exactly its declared size."""
    fig.savefig(path)
    return path
