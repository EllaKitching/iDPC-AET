"""
PTM notebook tools for manually editing coordinates to remove errors and plotting views for data presentation
"""

from __future__ import annotations

import pathlib

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.path import Path as MplPath
from matplotlib.widgets import Button, CheckButtons, LassoSelector
from mpl_toolkits.mplot3d import proj3d
from scipy.spatial import cKDTree

# settings for consistency
CLASS_NAMES = ("FCC", "HCP", "DH", "IH", "Chaos (FCC-like)", "Chaos (non-FCC-like)", "Non-FCC assigned")
# change class names if diff templates used for PTM
POINT_SIZE = 18

COLOURS = {
    "FCC": "#FFD600",
    "HCP": "#4CAF50",
    "DH": "#FF69B4",
    "IH": "#26A69A",
    "Chaos (FCC-like)": "#1565C0",
    "Chaos (non-FCC-like)": "#9C27B0",
    "Non-FCC assigned": "#808080",
    # selected is not a class its only for the coordinate editor to ensure selected atoms are clear!
    "Selected": "#D62728",
    # chaos here is for loading in for plot editing, before assignment of FCC like vs non
    "Chaos": "#1565C0",
}

GRID_C = "#ddd"
TICK_C = "#333"
SPINE_C = "#aaa"

MIN_SAVE_DPI = 500
DEFAULT_SAVE_DPI = 1000

AXIS_LABEL_FONTSIZE = 15
TICK_LABEL_FONTSIZE = 15
PANEL_TITLE_FONTSIZE = 15
SUPTITLE_FONTSIZE = 15
LEGEND_FONTSIZE = 15


def _save_figure(fig, save_path, dpi):
    """
    Save a matplotlib figure with high quality DPI and legend friendly rendering.

    Parameters:
        fig : matplotlib.figure.Figure
            Figure to save
        save_path : str or Path
            Destination image path
        dpi : int or float
            Dots per inch of save image, minimum set to at least 500
    """
    save_dpi = max(MIN_SAVE_DPI, int(dpi))
    fig.canvas.draw()
    fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {save_path} (dpi={save_dpi})")


def load_ptm_mat(mat_file):
    """
    Load PTM coordinates, assigned class masks, and score arrays from a MAT file.

    Parameters:
        mat_file : str or Path
            Path to a PTMResult .mat file

    Returns:
        xyz : np.ndarray
            Atom coordinates with shape (N, 3)
        masks : dict
            Boolean masks keyed by class name
        scores : dict
            Score arrays keyed by class name
        threshold : float
            Classification score threshold if present, else 1.0
    """
    # load mat file and pull out the ptm result struct
    data = sio.loadmat(mat_file)
    r = data["PTMResult"][0, 0]
    fields = r.dtype.names
    n_atoms = len(r["xyz"])

    def get_bool(name):
        if name in fields:
            return r[name].flatten().astype(bool)
        return np.zeros(n_atoms, dtype=bool)

    def get_float(name):
        if name in fields:
            return r[name].flatten()
        return np.zeros(n_atoms)

    # build aligned coordinate masks and score arrays
    xyz = np.asarray(r["xyz"], dtype=float)
    masks = {
        "FCC": get_bool("ind_fcc"),
        "HCP": get_bool("ind_hcp"),
        "DH": get_bool("ind_dh"),
        "IH": get_bool("ind_ih"),
        "Chaos": get_bool("ind_chaos"),
    }
    scores = {
        "FCC": get_float("FCCscore"),
        "HCP": get_float("HCPscore"),
        "DH": get_float("DHscore"),
        "IH": get_float("IHscore"),
        "Chaos": get_float("FCCscore"),
    }
    threshold = float(r["scoreThreshold"][0, 0]) if "scoreThreshold" in fields else 1.0
    return xyz, masks, scores, threshold


def _resolve_launch_input(ptm_input):
    """
    Determine whether input is from a filepath or preloaded coordinate/mask data.

    Parameters:
        ptm_input : str, Path, or tuple
            Either a .mat filepath, or a tuple of (xyz, masks)
    """
    if isinstance(ptm_input, (str, pathlib.Path)):
        source_name = str(ptm_input)
        xyz, masks, scores, threshold = load_ptm_mat(source_name)
        return source_name, xyz, masks, scores, threshold

    if isinstance(ptm_input, tuple) and len(ptm_input) == 2:
        # when tuple data is passed in, validate shape and fill missing masks
        xyz, masks_in = ptm_input
        xyz = np.asarray(xyz)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz must have shape (N, 3)")
        n_atoms = len(xyz)

        if not hasattr(masks_in, "keys"):
            raise TypeError("masks must be a dictionary mapping from class name to boolean mask")

        masks = {}
        for name in CLASS_NAMES:
            if name in masks_in:
                m = np.asarray(masks_in[name], dtype=bool).reshape(-1)
                if len(m) != n_atoms:
                    raise ValueError(
                        f"Mask length mismatch for {name}: expected {n_atoms}, got {len(m)}"
                    )
                masks[name] = m
            else:
                masks[name] = np.zeros(n_atoms, dtype=bool)
                
        # keep plain chaos as its own class in launch if it is provided
        if "Chaos" in masks_in:
            m = np.asarray(masks_in["Chaos"], dtype=bool).reshape(-1)
            if len(m) != n_atoms:
                raise ValueError(
                    f"Mask length mismatch for Chaos: expected {n_atoms}, got {len(m)}"
                )
            masks["Chaos"] = m

        scores = {name: np.zeros(n_atoms, dtype=float) for name in CLASS_NAMES}
        threshold = 1.0
        source_name = "PTM data"
        return source_name, xyz, masks, scores, threshold

    raise TypeError("ptm_input must be a filepath or a tuple: (xyz, masks)")


def launch(ptm_input, point_size=POINT_SIZE, elev=24, azim=42, return_fig=False):
    """
    Launch an interactive coordinate editor with PTM fits shown in 3D, notebook interactive friendly.

    Parameters:
        ptm_input : str, Path, or tuple
            Input source as a .mat path or preloaded (xyz, masks)
        point_size : int, optional
            Scatter marker size for visible atoms (default: 18)
        elev : float, optional
            Initial elevation angle for 3D view (default: 24)
        azim : float, optional
            Initial azimuth angle for 3D view (default: 42)
        return_fig : bool, optional
            If True, returns the matplotlib figure object (default: False)
    """
    # resolve input once and keep everything aligned
    source_name, xyz, orig_masks, _scores, _threshold = _resolve_launch_input(ptm_input)
    n_atoms = len(xyz)
    active_types = [k for k, v in orig_masks.items() if v.sum() > 0]

    # track state for removed and selected atoms and undo history
    removed = np.zeros(n_atoms, dtype=bool)
    selected = np.zeros(n_atoms, dtype=bool)
    undo_stack = []
    show = {k: True for k in active_types}
    lasso_mode = {"enabled": False}
    lasso_ref = {"widget": None}

    # helper returns atoms that are still kept
    def alive():
        return ~removed

    # helper returns visible atoms for one class
    def visible_mask(name):
        return orig_masks[name] & alive()

    # create main editor figure
    fig = plt.figure(figsize=(14, 8), facecolor="white")
    try:
        title_name = pathlib.Path(source_name).name if source_name != "PTM data" else source_name
        fig.canvas.manager.set_window_title(f"PTM Editor: {title_name}")
    except Exception:
        pass

    # set up layout and axes
    main_ax_pos = [0.22, 0.14, 0.75, 0.80]
    ax3 = fig.add_axes(main_ax_pos, projection="3d")
    overlay_ax = fig.add_axes(main_ax_pos, facecolor="none")
    overlay_ax.set_xlim(0.0, 1.0)
    overlay_ax.set_ylim(0.0, 1.0)
    overlay_ax.set_axis_off()
    overlay_ax.set_navigate(False)
    overlay_ax.set_visible(False)
    overlay_ax.set_zorder(4)
    ax3.set_zorder(2)

    chk_h = max(0.06, min(0.28, 0.055 * max(1, len(active_types))))
    check_ax = fig.add_axes([0.01, 0.88 - chk_h, 0.115, chk_h], facecolor="0.95")
    btn_las_ax = fig.add_axes([0.01, 0.44, 0.115, 0.06])
    btn_rem_ax = fig.add_axes([0.01, 0.36, 0.115, 0.06])
    btn_undo_ax = fig.add_axes([0.01, 0.28, 0.115, 0.06])
    btn_clr_ax = fig.add_axes([0.01, 0.20, 0.115, 0.06])
    btn_res_ax = fig.add_axes([0.01, 0.12, 0.115, 0.06])
    btn_sav_ax = fig.add_axes([0.01, 0.04, 0.115, 0.06])

    check = CheckButtons(check_ax, active_types, actives=[True] * len(active_types))
    label_to_idx = {name: i for i, name in enumerate(active_types)}

    for i, name in enumerate(active_types):
        colour = COLOURS.get(name, "black")
        check.labels[i].set_color(colour)
        try:
            check.rectangles[i].set_edgecolor(colour)
            check.rectangles[i].set_facecolor("white")
        except Exception:
            pass
        try:
            for line in check.lines[i]:
                line.set_color(colour)
                line.set_linewidth(1.8)
        except Exception:
            pass

    # set up button labels
    btn_style = dict(color="0.9", hovercolor="0.8")
    btn_las = Button(btn_las_ax, "Lasso: OFF\n[L]", **btn_style)
    Button(btn_rem_ax, "Remove\n[Del]", **btn_style)
    Button(btn_undo_ax, "Undo\n[Ctrl+Z]", **btn_style)
    Button(btn_clr_ax, "Clear Sel\n[Esc]", **btn_style)
    Button(btn_res_ax, "Reset\n[R]", **btn_style)
    Button(btn_sav_ax, "Save\n[Ctrl+S]", **btn_style)

    # status text shown at top of window
    status = fig.text(
        0.22,
        0.965,
        "",
        color="black",
        fontsize=10,
        va="top",
        bbox=dict(facecolor="white", edgecolor="0.8", pad=4),
    )

    def project_alive_to_overlay_axes():
        # project live 3d points into 2d overlay axes for picking and lasso
        idx = np.where(alive())[0]
        if not len(idx):
            return idx, np.empty((0, 2), dtype=float)
        x, y, z = xyz[idx, 0], xyz[idx, 1], xyz[idx, 2]
        x2, y2, _ = proj3d.proj_transform(x, y, z, ax3.get_proj())
        disp = ax3.transData.transform(np.column_stack([x2, y2]))
        axes_xy = overlay_ax.transAxes.inverted().transform(disp)
        return idx, axes_xy

    def set_lasso(enabled):
        # toggle lasso and recreate selector so callbacks stay clean
        lasso_mode["enabled"] = enabled
        overlay_ax.set_visible(enabled)
        if lasso_ref["widget"] is not None:
            try:
                lasso_ref["widget"].disconnect_events()
            except Exception:
                pass
            lasso_ref["widget"] = None
        if enabled:
            lasso_ref["widget"] = LassoSelector(
                overlay_ax,
                on_lasso,
                props=dict(color="gold", linewidth=1.5),
                useblit=False,
            )
            lasso_ref["widget"].set_active(True)
        btn_las.label.set_text(f"Lasso: {'ON' if enabled else 'OFF'}\\n[L]")
        fig.canvas.draw_idle()

    # redraw so visibility and selection stay in sync
    def redraw(_=None):
        cur_elev = getattr(ax3, "elev", elev)
        cur_azim = getattr(ax3, "azim", azim)
        ax3.cla()
        ax3.set_facecolor("white")
        ax3.set_xlabel("X (A)")
        ax3.set_ylabel("Y (A)")
        ax3.set_zlabel("Z (A)")
        ax3.view_init(elev=cur_elev, azim=cur_azim)

        shown_count = 0
        for name in reversed(active_types):
            if not show.get(name, True):
                continue
            mask = visible_mask(name)
            if not mask.any():
                continue
            shown_count += mask.sum()
            norm_m = mask & ~selected
            sel_m = mask & selected
            if norm_m.any():
                ax3.scatter(
                    xyz[norm_m, 0],
                    xyz[norm_m, 1],
                    xyz[norm_m, 2],
                    s=max(8, point_size - 6),
                    c=COLOURS.get(name, "black"),
                    alpha=0.75,
                    linewidths=0,
                    label=f"{name} ({mask.sum()})",
                )
            if sel_m.any():
                ax3.scatter(
                    xyz[sel_m, 0],
                    xyz[sel_m, 1],
                    xyz[sel_m, 2],
                    s=max(20, point_size * 1.6),
                    c=COLOURS["Selected"],
                    alpha=1.0,
                    edgecolors="white",
                    linewidths=0.4,
                    label=f"{name} sel ({sel_m.sum()})",
                )

        ax3.set_title(
            f"3D Editor | {shown_count} shown   {alive().sum()} alive   {removed.sum()} removed",
            color="black",
            fontsize=10,
        )
        ax3.legend(loc="upper right", facecolor="white", edgecolor="0.8", fontsize=9)
        status.set_text("If lasso is OFF: left click moves and right click zooms.")
        fig.canvas.draw_idle()

    def on_lasso(verts):
        # toggle selection for points hit by the lasso path
        if not verts or not lasso_mode["enabled"]:
            return
        alive_idx, pts_axes = project_alive_to_overlay_axes()
        if not len(alive_idx):
            return
        hit = alive_idx[MplPath(verts).contains_points(pts_axes)]
        if not len(hit):
            return
        selected[hit] = False if selected[hit].all() else True
        redraw()

    # simple wrapper so buttons and keys use the same toggle path
    def do_toggle_lasso(_=None):
        set_lasso(not lasso_mode["enabled"])

    def do_remove(_=None):
        # remove selected atoms and keep state for undo
        if not selected.any():
            status.set_text("Nothing selected.")
            fig.canvas.draw_idle()
            return
        undo_stack.append(removed.copy())
        removed[selected] = True
        selected[:] = False
        redraw()

    def do_undo(_=None):
        # restore previous removed state from undo stack
        if not undo_stack:
            status.set_text("Nothing to undo.")
            fig.canvas.draw_idle()
            return
        removed[:] = undo_stack.pop()
        selected[:] = False
        redraw()

    # reset removed and selected state
    def do_reset(_=None):
        undo_stack.append(removed.copy())
        removed[:] = False
        selected[:] = False
        redraw()

    # clear only current selection and keep removed atoms as they are
    def do_clear(_=None):
        selected[:] = False
        redraw()

    # save cleaned coordinates and class masks in one npz file
    def do_save(_=None):
        al = alive()
        stem = pathlib.Path(source_name).stem if source_name != "PTM data" else "ptm_data"
        default_name = f"{stem}_edited.npz"
        out = default_name

        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.update()
            chosen = filedialog.asksaveasfilename(
                title="Save cleaned PTM data",
                defaultextension=".npz",
                filetypes=[("NumPy archive", "*.npz")],
                initialfile=default_name,
            )
            root.destroy()
            if not chosen:
                status.set_text("Save cancelled.")
                fig.canvas.draw_idle()
                return
            out = chosen
        except Exception:
            out = default_name

        kept_count = int(al.sum())
        removed_count = int(removed.sum())
        total_count = int(n_atoms)
        save_dict = {
            "xyz": xyz[al],
            "removed_indices": np.where(removed)[0],
            "orig_file": source_name,
            "total_atoms_original": total_count,
            "total_atoms_kept": kept_count,
            "total_atoms_removed": removed_count,
        }
        for k, v in orig_masks.items():
            save_dict[f"ind_{k.lower()}"] = v[al]
        np.savez(out, **save_dict)
        msg = (
            f"Saved -> {out}  "
            f"({kept_count} kept, {removed_count} removed, {total_count} total)"
        )
        status.set_text(msg)
        fig.canvas.draw_idle()
        print(msg)

    # handle all button clicks in one place and check which button was pressed
    def on_click(event):
        if event.button != 1 or event.x is None or event.y is None:
            return

        if event.inaxes is btn_rem_ax:
            do_remove()
            return
        if event.inaxes is btn_undo_ax:
            do_undo()
            return
        if event.inaxes is btn_clr_ax:
            do_clear()
            return
        if event.inaxes is btn_res_ax:
            do_reset()
            return
        if event.inaxes is btn_sav_ax:
            do_save()
            return
        if event.inaxes is btn_las_ax:
            do_toggle_lasso()
            return

        if event.inaxes not in (ax3, overlay_ax):
            return

        alive_idx, pts_axes = project_alive_to_overlay_axes()
        if not len(alive_idx):
            return
        click_axes = overlay_ax.transAxes.inverted().transform((event.x, event.y))
        d2 = np.sum((pts_axes - click_axes) ** 2, axis=1)
        if d2.min() < 0.0005:
            idx = alive_idx[d2.argmin()]
            selected[idx] = not selected[idx]
            redraw()

    # connect mouse click handler
    fig.canvas.mpl_connect("button_press_event", on_click)

    def on_check(_label):
        # sync class visibility from checkbox state
        states = check.get_status()
        for name, idx in label_to_idx.items():
            show[name] = bool(states[idx])
        redraw()

    # connect checkbox toggles
    check.on_clicked(on_check)

    key_state = {"l_down": False}

    # keyboard shortcuts for common actions
    def on_key(event):
        if event.key in ("delete", "backspace"):
            do_remove()
        elif event.key == "ctrl+z":
            do_undo()
        elif event.key and event.key.lower() == "r":
            do_reset()
        elif event.key == "ctrl+s":
            do_save()
        elif event.key == "escape":
            do_clear()
        elif event.key and event.key.lower() == "l":
            if key_state["l_down"]:
                return
            key_state["l_down"] = True
            try:
                do_toggle_lasso()
            except Exception as exc:
                status.set_text(f"Lasso toggle error: {exc}")
                fig.canvas.draw_idle()

    # stop repeated l key toggles while key is held
    def on_key_release(event):
        if event.key and event.key.lower() == "l":
            key_state["l_down"] = False

    # connect key handlers
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("key_release_event", on_key_release)

    redraw()
    plt.show()

    if return_fig:
        return fig
    return None


# plot helpers
# load edited npz outputs for quick plotting
def _load_ptm_npz(filepath):
    """
    Load PTM coordinates and classification masks from an edited NPZ file.

    Parameters:
        filepath : str or Path
            Path to an edited .npz file produced by the editor
    """
    # read edited npz and rebuild class masks with safe fallbacks
    fp = pathlib.Path(filepath)
    d = np.load(fp, allow_pickle=True)
    xyz = d["xyz"]
    n_atoms = len(xyz)
    masks = {}
    
    for name in CLASS_NAMES:
        key = f"ind_{name.lower()}"
        if key in d:
            masks[name] = d[key].astype(bool)
        else:
            masks[name] = np.zeros(n_atoms, dtype=bool)
    
    active = {k: v for k, v in masks.items() if v.sum() > 0}
    return xyz, active


def _resolve_ptm_input(ptm_input):
    """
    Determine type of plotting input whether from filepath or preloaded coordinate/mask data passed as variables.

    Parameters:
        ptm_input : str, Path, or tuple
            Either a .mat/.npz filepath, or a tuple of (xyz, masks)
    """
    # use one plotting input path for mat npz or tuple data
    if isinstance(ptm_input, (str, pathlib.Path)):
        fp = pathlib.Path(ptm_input)
        if fp.suffix.lower() == ".mat":
            xyz, masks, _scores, _threshold = load_ptm_mat(fp)
            active = {k: v for k, v in masks.items() if v.sum() > 0}
        elif fp.suffix.lower() == ".npz":
            xyz, active = _load_ptm_npz(fp)
        else:
            raise ValueError(f"Unsupported file extension: {fp.suffix}")
        return xyz, active, fp.stem

    if isinstance(ptm_input, tuple) and len(ptm_input) == 2:
        xyz, masks = ptm_input
        xyz = np.asarray(xyz)
        active = {}
        for name, mask in masks.items():
            mask_bool = np.asarray(mask, dtype=bool)
            if mask_bool.sum() > 0:
                active[str(name)] = mask_bool
        return xyz, active, "PTM data"

    raise TypeError("ptm_input must be a filepath (.mat/.npz) or a tuple: (xyz, masks)")


def _apply_crop(xyz, crop=None, auto_crop=False, margin=2.0):
    """
    Build a mask of coords to keep using explicit axis crop ranges and/or automatic edge trimming.

    Parameters:
        xyz : np.ndarray
            Atom coordinates with shape (N, 3)
        crop : dict or None, optional
            Per-axis limits like {"x": (lo, hi)} to keep (default: None)
        auto_crop : bool, optional
            If True, trims each axis by margin from min/max bounds (default: False)
        margin : float, optional
            Margin removed at low/high ends during auto-crop (default: 2.0)
    """
    # start by keeping everything then tighten with crop rules
    keep = np.ones(len(xyz), dtype=bool)
    if auto_crop:
        for i in range(3):
            keep &= (xyz[:, i] >= xyz[:, i].min() + margin) & (xyz[:, i] <= xyz[:, i].max() - margin)
    if crop:
        axis_idx = {"x": 0, "y": 1, "z": 2}
        for ax_name, (lo, hi) in crop.items():
            i = axis_idx[ax_name.lower()]
            keep &= (xyz[:, i] >= lo) & (xyz[:, i] <= hi)
    return keep


def _legend_handles(active_masks):
    """
    Create legend patch handles for each active structure mask.

    Parameters:
        active_masks : dict
            Mapping of structure names to boolean masks after filtering
    """
    return [mpatches.Patch(color=COLOURS[k], label=f"{k}  ({v.sum()})") for k, v in active_masks.items()]


def _style_2d(ax, xlabel, ylabel):
    """
    Apply consistent styling for 2D projection subplots.

    Parameters:
        ax : matplotlib.axes.Axes
            Target axes to style
        xlabel : str
            Label text for the x-axis
        ylabel : str
            Label text for the y-axis
    """
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE_C)
    ax.tick_params(colors=TICK_C, labelsize=TICK_LABEL_FONTSIZE)
    ax.set_xlabel(xlabel, color=TICK_C, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, color=TICK_C, fontsize=AXIS_LABEL_FONTSIZE)
    ax.grid(True, color=GRID_C, linewidth=0.5, linestyle="--", alpha=0.6)


def _style_3d(ax):
    """
    Apply consistent styling for a 3D subplot.

    Parameters:
        ax : mpl_toolkits.mplot3d.axes3d.Axes3D
            Target 3D axes to style
    """
    ax.set_facecolor("white")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = True
        pane.set_facecolor("#f5f5f5")
        pane.set_edgecolor(SPINE_C)
    ax.tick_params(colors=TICK_C, labelsize=TICK_LABEL_FONTSIZE)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color(TICK_C)
        axis.label.set_size(AXIS_LABEL_FONTSIZE)


def _scatter_2d(ax, xyz_c, masks_c, hi, vi, point_size, alpha):
    """
    Scatter visible atoms for each classification onto a 2D projection.

    Parameters:
        ax : matplotlib.axes.Axes
            Target axes for plotting
        xyz_c : np.ndarray
            Cropped coordinates with shape (M, 3)
        masks_c : dict
            Class masks aligned to xyz_c
        hi : int
            Horizontal coordinate index in xyz_c
        vi : int
            Vertical coordinate index in xyz_c
        point_size : int or float
            Marker size for scatter points
        alpha : float
            Marker opacity
    """
    for name in reversed(list(masks_c)):
        mask = masks_c[name]
        if not mask.any():
            continue
        ax.scatter(
            xyz_c[mask, hi],
            xyz_c[mask, vi],
            s=point_size,
            c=COLOURS[name],
            alpha=alpha,
            linewidths=0,
            rasterized=True,
        )


def _scatter_3d(ax, xyz_c, masks_c, point_size, alpha):
    """
    Scatter visible atoms for each classification in 3D.

    Parameters:
        ax : mpl_toolkits.mplot3d.axes3d.Axes3D
            Target 3D axes for plotting
        xyz_c : np.ndarray
            Cropped coordinates with shape (M, 3)
        masks_c : dict
            Class masks aligned to xyz_c
        point_size : int or float
            Marker size for scatter points
        alpha : float
            Marker opacity
    """
    for name in reversed(list(masks_c)):
        mask = masks_c[name]
        if not mask.any():
            continue
        ax.scatter(
            xyz_c[mask, 0],
            xyz_c[mask, 1],
            xyz_c[mask, 2],
            s=point_size,
            c=COLOURS[name],
            alpha=alpha,
            linewidths=0,
            rasterized=True,
        )

def _shared_rounded_limits(xyz_c, step=10.0):
    """
    Return shared min and max bounds rounded to a fixed step for consistent scaling.

    Parameters:
        xyz_c : np.ndarray
            Coordinates used to compute shared bounds
        step : float, optional
            Rounding step in Angstrom for lower/upper limits (default: 10.0)
    """
    vmin = float(np.nanmin(xyz_c))
    vmax = float(np.nanmax(xyz_c))
    lo = np.floor(vmin / step) * step
    hi = np.ceil(vmax / step) * step
    if hi <= lo:
        hi = lo + step
    return float(lo), float(hi)


def plot_all(
    ptm_input,
    crop=None,
    auto_crop=False,
    margin=2.0,
    point_size=8,
    alpha=0.65,
    elev=25,
    azim=45,
    dpi=DEFAULT_SAVE_DPI,
    save_path=None,
    include_non_fcc_assigned=False,
    labelplot=False,
):
    """
    Plot XY, XZ, YZ projections and a 3D panel in one figure with a shared legend.

    Parameters:
        ptm_input : str, Path, or tuple
            Input source as a .mat/.npz path, or preloaded (xyz, masks)
        crop : dict or None, optional
            Axis coordinate limits like {"x": (lo, hi)} (default: None)
        auto_crop : bool, optional
            If True, trims edges on each axis by margin (default: False)
        margin : float, optional
            Margin used for auto cropping (default: 2.0)
        point_size : int, optional
            Marker size for 2D points (default: 8)
        alpha : float, optional
            Marker opacity in 2D plots (default: 0.65)
        elev : float, optional
            Elevation angle for the 3D camera (default: 25)
        azim : float, optional
            Azimuth angle for the 3D camera (default: 45)
        dpi : int, optional
            DPI used when saving the figure; minimum is 500 (default: 1000)
        save_path : str or Path or None, optional
            Output image path; if None, does not save (default: None)
        include_non_fcc_assigned : bool, optional
            If True, show only FCC, both Chaos categories, and Non-FCC assigned. If False, show all except Non-FCC assigned (default: False)
        labelplot : bool, optional
            If True, add bold a)/b)/c) labels at top-left of XY/XZ/YZ panels (default: False)
    """
    # resolve input and apply crop before plotting
    xyz, active, stem = _resolve_ptm_input(ptm_input)
    keep = _apply_crop(xyz, crop=crop, auto_crop=auto_crop, margin=margin)
    # warning removing all atoms may cause crash
    xyz_c = xyz[keep]
    masks_c = {k: v[keep] for k, v in active.items()}
    
    # pick colour scheme based on non-fcc assigned mode
    if include_non_fcc_assigned:
        keep_categories = {"FCC", "Chaos (FCC-like)", "Chaos (non-FCC-like)", "Non-FCC assigned"}
        masks_c = {k: v for k, v in masks_c.items() if k in keep_categories}
    else:
        masks_c = {k: v for k, v in masks_c.items() if k != "Non-FCC assigned"}

    panels_2d = [(0, 1, "X (A)", "Y (A)", "XY"), (2, 0, "Z (A)", "X (A)", "XZ"), (1, 2, "Y (A)", "Z (A)", "YZ")]

    fig = plt.figure(figsize=(18, 5.5), facecolor="white")
    gs = fig.add_gridspec(2, 4, height_ratios=[0.08, 1], left=0.04, right=0.97, bottom=0.08, top=0.95, wspace=0.32, hspace=0.15)

    # put legend in the top row across all columns
    leg_ax = fig.add_subplot(gs[0, :])
    leg_ax.set_axis_off()

    panel_labels = ("a)", "b)", "c)")
    lim_lo, lim_hi = _shared_rounded_limits(xyz_c)

    # draw each 2d panel with shared limits so scales match
    for col, (hi, vi, xl, yl, title) in enumerate(panels_2d):
        ax = fig.add_subplot(gs[1, col])
        _style_2d(ax, xl, yl)
        _scatter_2d(ax, xyz_c, masks_c, hi, vi, point_size, alpha)
        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("")
        if labelplot:
            ax.text(
                0.02,
                0.98,
                panel_labels[col],
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=PANEL_TITLE_FONTSIZE,
                fontweight="bold",
                color=TICK_C,
            )

    ax3 = fig.add_subplot(gs[1, 3], projection="3d")
    _style_3d(ax3)
    ax3.set_xlabel("X (A)")
    ax3.set_ylabel("Y (A)")
    ax3.set_zlabel("Z (A)")
    ax3.view_init(elev=elev, azim=azim)
    ax3.set_xlim(lim_lo, lim_hi)
    ax3.set_ylim(lim_lo, lim_hi)
    ax3.set_zlim(lim_lo, lim_hi)
    _scatter_3d(ax3, xyz_c, masks_c, max(3, point_size - 3), alpha * 0.85)
    ax3.set_title("3D", color="black", fontsize=PANEL_TITLE_FONTSIZE, pad=4)

    # build and draw one shared legend
    handles = _legend_handles(masks_c)
    leg_ax.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.5, 0.08),
        ncol=len(handles),
        facecolor="white",
        edgecolor="#ccc",
        labelcolor="black",
        fontsize=LEGEND_FONTSIZE - 1,
        markerscale=1.2,
        framealpha=0.9,
        handlelength=1.0,
        borderpad=0.4,
    )

    if save_path:
        _save_figure(fig, save_path, dpi)
    return fig


def plot_projections(
    ptm_input,
    crop=None,
    auto_crop=False,
    margin=2.0,
    point_size=8,
    alpha=0.65,
    dpi=DEFAULT_SAVE_DPI,
    save_path=None,
    include_non_fcc_assigned=False,
    labelplot=False,
):
    """
    Plot XY, XZ, and YZ projection panels with a legend column.

    Parameters:
        ptm_input : str, Path, or tuple
            Input source as a .mat/.npz path, or preloaded (xyz, masks)
        crop : dict or None, optional
            Axis coordinate limits like {"x": (lo, hi)} (default: None)
        auto_crop : bool, optional
            If True, trims edges on each axis by margin (default: False)
        margin : float, optional
            Margin used for auto cropping (default: 2.0)
        point_size : int, optional
            Marker size for 2D points (default: 8)
        alpha : float, optional
            Marker opacity in 2D plots (default: 0.65)
        dpi : int, optional
            DPI used when saving the figure; minimum is 500 (default: 1000)
        save_path : str or Path or None, optional
            Output image path; if None, does not save (default: None)
        include_non_fcc_assigned : bool, optional
            If True, show only FCC, both Chaos categories, and Non-FCC assigned. If False, show all except Non-FCC assigned (default: False)
        labelplot : bool, optional
            If True, add bold a)/b)/c) labels at top-left of XY/XZ/YZ panels (default: False)
    """
    # make clean xy xz yz panels for reporting
    xyz, active, stem = _resolve_ptm_input(ptm_input)
    keep = _apply_crop(xyz, crop=crop, auto_crop=auto_crop, margin=margin)
    # warning removing all atoms may cause crash
    xyz_c = xyz[keep]
    masks_c = {k: v[keep] for k, v in active.items()}
    
    # pick colour scheme based on non-fcc assigned mode
    if include_non_fcc_assigned:
        keep_categories = {"FCC", "Chaos (FCC-like)", "Chaos (non-FCC-like)", "Non-FCC assigned"}
        masks_c = {k: v for k, v in masks_c.items() if k in keep_categories}
    else:
        masks_c = {k: v for k, v in masks_c.items() if k != "Non-FCC assigned"}

    # panel defs are coordinate index pairs plus axis labels
    panels = [(0, 1, "X (A)", "Y (A)", "XY"), (2, 0, "Z (A)", "X (A)", "XZ"), (1, 2, "Y (A)", "Z (A)", "YZ")]

    fig = plt.figure(figsize=(15, 5.5), facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[0.08, 1], left=0.05, right=0.95, bottom=0.08, top=0.95, wspace=0.28, hspace=0.15)

    # put legend in the top row across all columns
    leg_ax = fig.add_subplot(gs[0, :])
    leg_ax.set_axis_off()

    panel_labels = ("a)", "b)", "c)")
    lim_lo, lim_hi = _shared_rounded_limits(xyz_c)

    # draw each projection using same limits and optional panel labels
    for col, (hi, vi, xl, yl, title) in enumerate(panels):
        ax = fig.add_subplot(gs[1, col])
        _style_2d(ax, xl, yl)
        _scatter_2d(ax, xyz_c, masks_c, hi, vi, point_size, alpha)
        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("")
        if labelplot:
            ax.text(
                0.02,
                0.98,
                panel_labels[col],
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=PANEL_TITLE_FONTSIZE,
                fontweight="bold",
                color=TICK_C,
            )

    handles = _legend_handles(masks_c)
    leg_ax.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.5, 0.08),
        ncol=len(handles),
        facecolor="white",
        edgecolor="#ccc",
        labelcolor="black",
        fontsize=LEGEND_FONTSIZE - 1,
        markerscale=1.2,
        framealpha=0.9,
        handlelength=1.0,
        borderpad=0.4,
    )

    if save_path:
        _save_figure(fig, save_path, dpi)
    return fig


def plot_3d(
    ptm_input,
    crop=None,
    auto_crop=False,
    margin=2.0,
    point_size=9,
    alpha=0.55,
    elev=25,
    azim=45,
    dpi=DEFAULT_SAVE_DPI,
    save_path=None,
    include_non_fcc_assigned=False,
):
    """
    Plot a standalone 3D atom scatter with an external legend.

    Parameters:
        ptm_input : str, Path, or tuple
            Input source as a .mat/.npz path, or preloaded (xyz, masks)
        crop : dict or None, optional
            Axis coordinate limits like {"x": (lo, hi)} (default: None)
        auto_crop : bool, optional
            If True, trims edges on each axis by margin (default: False)
        margin : float, optional
            Margin used for auto cropping (default: 2.0)
        point_size : int, optional
            Marker size for 3D points (default: 9)
        alpha : float, optional
            Marker opacity in 3D plot (default: 0.55)
        elev : float, optional
            Elevation angle for the 3D camera (default: 25)
        azim : float, optional
            Azimuth angle for the 3D camera (default: 45)
        dpi : int, optional
            DPI used when saving the figure; minimum is 500 (default: 1000)
        save_path : str or Path or None, optional
            Output image path; if None, does not save (default: None)
        include_non_fcc_assigned : bool, optional
            If True, show only FCC, both Chaos categories, and Non-FCC assigned. If False, show all except Non-FCC assigned (default: False)
    """
    # keep a standalone 3d view for quick structure checks
    xyz, active, stem = _resolve_ptm_input(ptm_input)
    keep = _apply_crop(xyz, crop=crop, auto_crop=auto_crop, margin=margin)
    # warning removing all atoms may cause crash
    xyz_c = xyz[keep]
    masks_c = {k: v[keep] for k, v in active.items()}
    
    # pick colour scheme based on non-fcc assigned mode
    if include_non_fcc_assigned:
        keep_categories = {"FCC", "Chaos (FCC-like)", "Chaos (non-FCC-like)", "Non-FCC assigned"}
        masks_c = {k: v for k, v in masks_c.items() if k in keep_categories}
    else:
        masks_c = {k: v for k, v in masks_c.items() if k != "Non-FCC assigned"}

    # draw standalone 3d view with shared rounded limits
    fig = plt.figure(figsize=(11, 8), facecolor="white")
    ax = fig.add_axes([0.05, 0.10, 0.80, 0.85], projection="3d")
    _style_3d(ax)
    ax.set_xlabel("X (A)")
    ax.set_ylabel("Y (A)")
    ax.set_zlabel("Z (A)")
    ax.view_init(elev=elev, azim=azim)
    lim_lo, lim_hi = _shared_rounded_limits(xyz_c)
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_zlim(lim_lo, lim_hi)
    _scatter_3d(ax, xyz_c, masks_c, point_size, alpha)

    # keep legend on the right and inside the figure bounds
    leg_ax = fig.add_axes([0.82, 0.20, 0.16, 0.60])
    leg_ax.set_axis_off()
    handles = _legend_handles(masks_c)
    leg_ax.legend(
        handles=handles,
        loc="center left",
        facecolor="white",
        edgecolor="#ccc",
        labelcolor="black",
        fontsize=LEGEND_FONTSIZE,
        markerscale=1.4,
        framealpha=0.9,
        handlelength=1.2,
        borderpad=0.8,
    )

    if save_path:
        _save_figure(fig, save_path, dpi)
    return fig


# interactive viewer helpers
# keep fcc hcp and chaos grouping for quick checks
def _groups_for_viewer(xyz, masks, scores):
    fcc_score = np.asarray(scores.get("FCC", np.zeros(len(xyz))), dtype=float)
    hcp_score = np.asarray(scores.get("HCP", np.zeros(len(xyz))), dtype=float)
    n_atoms = len(xyz)
    
    groups = [
        ("FCC", np.asarray(masks.get("FCC", np.zeros(n_atoms, dtype=bool)), dtype=bool), fcc_score),
        ("HCP", np.asarray(masks.get("HCP", np.zeros(n_atoms, dtype=bool)), dtype=bool), hcp_score),
    ]
    
    # include both chaos categories if they exist in masks
    for chaos_type in ["Chaos (FCC-like)", "Chaos (non-FCC-like)"]:
        if chaos_type in masks:
            groups.append((chaos_type, np.asarray(masks[chaos_type], dtype=bool), fcc_score))
    
    return groups


__all__ = [
    "CLASS_NAMES",
    "POINT_SIZE",
    "load_ptm_mat",
    "launch",
    "plot_all",
    "plot_projections",
    "plot_3d"
]
