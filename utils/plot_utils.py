import os
import matplotlib as mpl
import matplotlib.pyplot as plt



# 1) Set up your LaTeX/PGF rcParams once, but do NOT call mpl.use("pgf") here.
#    That way your default backend (usually Agg) stays in place for PNG/PDF exports.
mpl.rcParams.update({
    "pgf.texsystem":        "pdflatex",
    "text.usetex": False,
    "pgf.rcfonts": False,
    "font.size": 10,
    "font.family":          "sans-serif",
    "font.sans-serif":      ["DejaVu Sans"],
    "mathtext.fontset":     "dejavusans",
    "mathtext.rm":          "sans",       # use sans for “normal” (roman) math
    "mathtext.it":          "sans:italic",
    "mathtext.bf":          "sans:bold",
})


class Plotting:
    def __init__(self, export_folder: str):
        self.export_folder = export_folder

        self.export_folder = export_folder
        os.makedirs(export_folder, exist_ok=True)

        # Also create a subfolder for PGF if you like:
        pgf_folder = os.path.join(export_folder, "pgf")
        os.makedirs(pgf_folder, exist_ok=True)
        self._pgf_folder = pgf_folder

    def savefig(self, file_name: str, **save_kwargs):
        """
        Export as PNG/PDF via Agg, WITHOUT shelling out to LaTeX.
        Example:
            P.savefig("plot.png")
            P.savefig("plot.pdf")
        """
        # 1) Re‐attach an Agg canvas so PGF isn’t forced:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig = plt.gcf()
        fig.canvas = FigureCanvasAgg(fig)

        # 2) Temporarily turn off text.usetex, so it doesn’t run pdflatex
        with mpl.rc_context({"text.usetex": False}):
            out_path = os.path.join(self.export_folder, file_name)
            fig.savefig(
                out_path,
                dpi=300,
                bbox_inches="tight",
                pad_inches=0,
                **save_kwargs
            )

    def savepgf(self, file_name: str, **save_kwargs):
        fig = plt.gcf()
        # --- override to TeX for PGF export ---

        with mpl.rc_context({
            "text.usetex":         True,
            "pgf.rcfonts":         False,
            "text.latex.preamble": "\n".join([
                r"\usepackage[sfdefault]{dejavusans}",
                r"\usepackage{sansmath}",
                r"\sansmath",
            ]),
        }):
            from matplotlib.backends.backend_pgf import FigureCanvasPgf
            fig.canvas = FigureCanvasPgf(fig)
            fig.canvas.draw()
            out = os.path.join(self._pgf_folder, f"{file_name}.pgf")
            fig.savefig(out, bbox_inches="tight", pad_inches=0, **save_kwargs)


if __name__ == "__main__":
    P = Plotting('output')
    print(P.export_folder)
    print(P._pgf_folder)

    fig, ax = plt.subplots(figsize=(5, 3))
    plt.plot([0, 1, 2], [0, 1, 4], label=r"$y=x^2$")
    plt.legend()

    # LaTeX PGF export:
    P.savepgf("pkg_export")

    # raster/vector export:
    P.savefig("png_export_example.png")

    plt.show()
    