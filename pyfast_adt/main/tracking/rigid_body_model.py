import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tkinter import Tk, filedialog

class MastronardeRigidBody:
    def __init__(self, folder_path, fit_range, pixelsize_nm, delimiter="\t",
                 plot_intermediate=False, theta_sim_deg=0):
        self.folder_path = folder_path
        if fit_range != list() and type(fit_range) in [int, float]:
            self.fit_range = [fit_range]
        else:
            self.fit_range = fit_range
        self.pixelsize_nm = pixelsize_nm
        self.pixelsize_um = pixelsize_nm / 1000
        self.delimiter = delimiter
        self.plot_intermediate = plot_intermediate
        self.theta_sim_deg = theta_sim_deg
        self.results = []
        self.datasets = []

    def _get_dataset(self, dataset_index=0):
        if len(self.datasets) == 0:
            raise RuntimeError("No datasets fitted yet.")

        if isinstance(dataset_index, int):
            return self.datasets[dataset_index]

        # allow passing dataset number directly
        for ds in self.datasets:
            if ds["index"] == dataset_index:
                return ds

        raise ValueError(f"Dataset {dataset_index} not found.")

    # -----------------------------
    # Rotated Mastronarde model
    # -----------------------------
    @staticmethod
    def mastronarde_rot2D(alpha_rad, y0, z0, ys, theta_deg, y_mean):
        theta = np.deg2rad(theta_deg)
        x_m = (y0 + ys) * np.cos(alpha_rad) - z0 * np.sin(alpha_rad) - y0
        x_rot = x_m * np.cos(theta) - y_mean * np.sin(theta)
        y_rot = x_m * np.sin(theta) + y_mean * np.cos(theta)
        return np.concatenate([x_rot, y_rot])

    def mastronarde_point(self, alpha_rad, y0, z0, ys, theta_deg, y_mean=0.0):
        """
        Return rotated (y, x) coordinates from Mastronarde model
        """
        theta = np.deg2rad(theta_deg)

        x_m = (y0 + ys) * np.cos(alpha_rad) - z0 * np.sin(alpha_rad) - y0

        x_rot = x_m * np.cos(theta) - y_mean * np.sin(theta)
        y_rot = x_m * np.sin(theta) + y_mean * np.cos(theta)

        return y_rot, x_rot

    def zheng_point(self, alpha_rad, y0, z0, ys, theta_deg, y_mean=0.0):

        alpha = alpha_rad
        theta = np.deg2rad(theta_deg)

        n = (y0) * np.cos(alpha) + z0 * np.sin(alpha)
        z = (-y0) * np.sin(alpha) + z0 * np.cos(alpha)
        # n = (y0+ys) * np.cos(alpha) - z0 * np.sin(alpha)
        # z = (y0+ys) * np.sin(alpha) + z0 * np.cos(alpha)

        return n, z

    # -----------------------------
    # Fit single dataset
    # -----------------------------
    def fit_dataset(self, idx):
        fname = f"fit{idx}.txt"
        path = os.path.join(self.folder_path, fname)
        if not os.path.exists(path):
            print(f"[SKIP] {fname}")
            return None

        # Load data
        data = np.loadtxt(path, delimiter=self.delimiter)
        alpha = data[:, 0]
        x_ccd = data[:, 1]
        y_ccd = data[:, 2]

        # Simulate tilt rotation
        theta_sim_rad = np.deg2rad(self.theta_sim_deg)
        x_ccd_rot = x_ccd * np.cos(theta_sim_rad) - y_ccd * np.sin(theta_sim_rad)
        y_ccd_rot = x_ccd * np.sin(theta_sim_rad) + y_ccd * np.cos(theta_sim_rad)
        x_ccd, y_ccd = x_ccd_rot* self.pixelsize_um, y_ccd_rot* self.pixelsize_um

        alpha_rad = np.deg2rad(alpha)
        y_mean = 0.0
        xy_data = np.concatenate([x_ccd, y_ccd])
        ys_init = x_ccd[np.argmin(np.abs(alpha))]
        p0 = [0.0, 0.0, ys_init, 0.0, y_mean]

        # Bounds
        lower_bounds = [-np.inf, -np.inf, -np.inf, -60, -np.inf]
        upper_bounds = [np.inf, np.inf, np.inf, 100, np.inf]

        # Fit
        try:
            popt, pcov = curve_fit(
                lambda a, y0, z0, ys, theta_deg, y_mean:
                self.mastronarde_rot2D(a, y0, z0, ys, theta_deg, y_mean),
                alpha_rad, xy_data, p0=p0, maxfev=20000, bounds=(lower_bounds, upper_bounds)
            )
        except RuntimeError:
            print(f"[FAIL] Fit failed for {fname}")
            return None

        y0, z0, ys, theta_deg, _ = popt
        perr = np.sqrt(np.diag(pcov))

        self.results.append({
            "index": idx,
            "y0": y0,
            "z0": z0,
            "ys": ys,
            "theta_deg": theta_deg,
            "z0_err": perr[1],
            "y0_err": perr[0],
            "ys_err": perr[2]
        })

        # Optional intermediate plots
        if self.plot_intermediate:
            self.plot_intermediate_dataset(alpha, x_ccd, y_ccd, alpha_rad, popt, fname)

        # Print fit results
        print(f"[Dataset {fname}] Mastronarde Fit Results:")
        print(f"  y0  = {y0:.3f} µm ± {perr[0]:.3f}")
        print(f"  z0  = {z0:.3f} µm ± {perr[1]:.3f}")
        print(f"  ys  = {ys:.3f} µm ± {perr[2]:.3f}")
        print(f"  θ   = {theta_deg:.3f}°")
        print(f"  y_mean   = {y_mean:.5f} um")
        print("---------------------------------------------------\n")
        # Store full dataset for trajectory plots
        self.datasets.append({
            "index": idx,
            "alpha_rad": alpha_rad,
            "x_ccd": x_ccd,
            "y_ccd": y_ccd,
            "popt": popt
        })

        # -----------------------------
        # Fit single dataset
        # -----------------------------

    def fit_single_dataset_from_gui(self, data_path = None, switch_axis = False):
        # it expect tracking.txt file from pyfast_adt structure
        if data_path == None:
            Tk().withdraw()
            print("Select tracking data for fit")
            file_name = filedialog.askopenfilename()
        else:
            file_name = data_path
        directory_path, fname = os.path.split(file_name)
        path = file_name
        self.pixelsize_nm = float(input("Enter the pixelsize in nm:"))
        self.pixelsize_um = self.pixelsize_nm/1000
        # Load data
        tilt_min, tilt_max, step, data = self.load_tracking_data_pyfast(path)
        alpha = np.arange(tilt_min, tilt_max+step, step)
        if switch_axis == False:
            x_ccd = data[:, 0]
            y_ccd = data[:, 1]
        elif switch_axis == True:
            print("switching axis x -> y")
            x_ccd = data[:, 1]
            y_ccd = data[:, 0]

        # Simulate tilt rotation
        theta_sim_rad = np.deg2rad(0)
        x_ccd_rot = x_ccd * np.cos(theta_sim_rad) - y_ccd * np.sin(theta_sim_rad)
        y_ccd_rot = x_ccd * np.sin(theta_sim_rad) + y_ccd * np.cos(theta_sim_rad)
        x_ccd, y_ccd = x_ccd_rot * self.pixelsize_um, y_ccd_rot * self.pixelsize_um

        alpha_rad = np.deg2rad(alpha)
        y_mean = 0.0
        xy_data = np.concatenate([x_ccd, y_ccd])
        ys_init = x_ccd[np.argmin(np.abs(alpha))]
        p0 = [0.0, 0.0, ys_init, 0.0, y_mean]

        # Bounds
        lower_bounds = [-np.inf, -np.inf, -np.inf, -60, -np.inf]
        upper_bounds = [np.inf, np.inf, np.inf, 100, np.inf]

        # Fit
        try:
            popt, pcov = curve_fit(
                lambda a, y0, z0, ys, theta_deg, y_mean:
                self.mastronarde_rot2D(a, y0, z0, ys, theta_deg, y_mean),
                alpha_rad, xy_data, p0=p0, maxfev=20000, bounds=(lower_bounds, upper_bounds)
            )
        except RuntimeError:
            print(f"[FAIL] Fit failed for {fname}")
            return None

        y0, z0, ys, theta_deg, y_mean = popt
        perr = np.sqrt(np.diag(pcov))

        self.results.append({
            "index": 0,
            "y0": y0,
            "z0": z0,
            "ys": ys,
            "theta_deg": theta_deg,
            "z0_err": perr[1],
            "y0_err": perr[0],
            "ys_err": perr[2]
        })

        # Optional intermediate plots, trajectory normal and rotated
        self.plot_intermediate_dataset(alpha, x_ccd, y_ccd, alpha_rad, popt, fname)

        # Print fit results
        print(f"[Dataset {fname}] Mastronarde Fit Results:")
        print(f"  y0  = {y0:.3f} µm ± {perr[0]:.3f}")
        print(f"  z0  = {z0:.3f} µm ± {perr[1]:.3f}")
        print(f"  ys  = {ys:.3f} µm ± {perr[2]:.3f}")
        print(f"  θ   = {theta_deg:.3f}°± {perr[3]:.3f}")
        print(f"  y_mean   = {y_mean:.5f} um ± {perr[4]:.3f}")
        print("---------------------------------------------------\n")
        # Store full dataset for trajectory plots
        self.datasets = []
        self.datasets.append({
            "index": 0,
            "alpha_rad": alpha_rad,
            "x_ccd": x_ccd,
            "y_ccd": y_ccd,
            "popt": popt
        })
        self.folder_path = directory_path
        self.plot_single_dataset_summary(0, save = True)

    # ============================================================
    # LOAD TRACKING DATA from pyfast-adt format
    # ============================================================
    def load_tracking_data_pyfast(self, path):
        metadata = {}
        coords = []

        with open(path, "r") as f:
            lines = f.readlines()

        data_section = False

        for line in lines:
            line = line.strip()

            # Stop if file end marker
            if line == "end_tracking_file":
                break

            # Parse metadata (key = value)
            if "=" in line and not data_section:
                key, value = line.split("=", 1)
                metadata[key.strip()] = value.strip()
                continue

            # Detect start of coordinate section
            if line.startswith("tracking_positions"):
                data_section = True
                continue

            # Parse coordinate lines
            if data_section and "," in line:
                parts = line.split(",")
                if len(parts) == 3:
                    angle, x, y = map(float, parts)
                    coords.append([angle, x, y])

        coords = np.array(coords)

        # Extract useful metadata
        tilt_min = float(metadata.get("start_angle (deg)", 0))
        tilt_max = float(metadata.get("target_angle (deg)", 0))
        step = float(metadata.get("tilt_step (deg/img)", 0))

        return tilt_min, tilt_max, step, coords

    # -----------------------------
    # Optional intermediate plots
    # -----------------------------
    def plot_intermediate_dataset(self, alpha, x_ccd, y_ccd, alpha_rad, popt, fname):
        alpha_fit = np.linspace(alpha_rad.min(), alpha_rad.max(), 400)
        xy_fit = self.mastronarde_rot2D(alpha_fit, *popt)
        x_fit = xy_fit[:len(alpha_fit)]
        y_fit = xy_fit[len(alpha_fit):]

        # X vs Y
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(alpha, x_ccd, "o", label="CCD X data")
        plt.plot(np.rad2deg(alpha_fit), x_fit, "-", label=f"Fit θ={popt[3]:.2f}°")
        plt.xlabel("Tilt α (deg)")
        plt.ylabel("X (CCD)")
        plt.legend()
        plt.title(fname)

        plt.subplot(1, 2, 2)
        plt.plot(alpha, y_ccd, "o", label="CCD Y data")
        plt.plot(np.rad2deg(alpha_fit), y_fit, "-", label=f"Fit θ={popt[3]:.2f}°")
        plt.xlabel("Tilt α (deg)")
        plt.ylabel("Y (CCD)")
        plt.legend()
        plt.title(fname)
        plt.tight_layout()
        plt.show()

        # XY superposition
        theta_rad = -np.deg2rad(popt[3])
        x_rot = x_ccd * np.cos(theta_rad) - y_ccd * np.sin(theta_rad)
        y_rot = x_ccd * np.sin(theta_rad) + y_ccd * np.cos(theta_rad)

        plt.figure(figsize=(5, 5))
        plt.plot(x_ccd, y_ccd, "o", label="Original XY")
        plt.plot(x_rot, y_rot, "o", label="Rotated XY")
        plt.axhline(0, color="k", ls="--", alpha=0.5)
        plt.axvline(0, color="k", ls="--", alpha=0.5)
        plt.xlabel("X (CCD)")
        plt.ylabel("Y (CCD)")
        plt.title(f"{fname} — XY rotation θ={popt[3]:.2f}°")
        plt.legend()
        plt.axis("equal")
        plt.tight_layout()
        plt.show()

    # -----------------------------
    # Run all fits
    # -----------------------------
    def run_fits(self):
        for idx in self.fit_range:
            self.fit_dataset(idx)

    # -----------------------------
    # Compute linear fits
    # -----------------------------
    def compute_linear_fits(self):
        indices = np.array([r["index"] for r in self.results])
        z0_um = np.array([r["z0"] for r in self.results])
        y0_um = np.array([r["y0"] for r in self.results])
        ys_um = np.array([r["ys"] for r in self.results])
        theta_deg = np.array([r["theta_deg"] for r in self.results])

        def linear(x, m, c): return m*x+c

        # z0
        p_z0, cov_z0 = np.polyfit(indices, z0_um, 1, cov=True)
        m_z0, c_z0 = p_z0
        z0_fit = linear(indices, m_z0, c_z0)
        eucentric_height = -c_z0/m_z0 if m_z0 !=0 else np.nan

        # y0
        p_y0, cov_y0 = np.polyfit(indices, y0_um, 1, cov=True)
        m_y0, c_y0 = p_y0
        y0_fit = linear(indices, m_y0, c_y0)

        # ys
        p_ys, cov_ys = np.polyfit(indices, ys_um, 1, cov=True)
        m_ys, c_ys = p_ys
        ys_fit = linear(indices, m_ys, c_ys)

        self.linear_fits = {
            "indices": indices,
            "z0_um": z0_um,
            "y0_um": y0_um,
            "ys_um": ys_um,
            "theta_deg": theta_deg,
            "z0_fit": z0_fit,
            "y0_fit": y0_fit,
            "ys_fit": ys_fit,
            "m_z0": m_z0, "c_z0": c_z0, "eucentric_height": eucentric_height,
            "m_y0": m_y0, "c_y0": c_y0,
            "m_ys": m_ys, "c_ys": c_ys
        }

    # -----------------------------
    # Final summary plot
    # -----------------------------
    def plot_summary(self, save=True):
        lf = self.linear_fits
        fig, (ax1, ax_text) = plt.subplots(
            1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [3,1]}
        )

        # Left plot
        ax1.errorbar(lf["indices"], lf["z0_um"], yerr=0, fmt="o", label="z0", mfc='blue', mec='blue')
        ax1.plot(lf["indices"], lf["z0_fit"], "b--", label="z0 fit")
        ax1.errorbar(lf["indices"], lf["y0_um"], yerr=0, fmt="s", label="y0", mfc='orange', mec='orange')
        ax1.plot(lf["indices"], lf["y0_fit"], "y--", label="y0 fit")
        ax1.errorbar(lf["indices"], lf["ys_um"], yerr=0, fmt="^", label="ys", mfc='green', mec='green')
        ax1.plot(lf["indices"], lf["ys_fit"], "g--", label="ys fit")
        ax1.set_xlabel("Z stage (µm)")
        ax1.set_ylabel("Fitted parameter (µm)")
        ax1.grid(alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(lf["indices"], lf["theta_deg"], "d-", color="tab:red", label="θ (deg)")
        ax2.set_ylabel("θ (deg)")

        # Right text + table
        ax_text.axis('off')
        fit_info = (
            f"Linear Fit Info:\n\n"
            f"Pixelsize = {self.pixelsize_nm:.4f} nm\n"
            f"z0 linear fit: slope={lf['m_z0']:.4f}, intercept={lf['c_z0']:.4f}\n"
            f"Eucentric height (z0=0)={lf['eucentric_height']:.4f}\n"
            f"y0 linear fit: slope={lf['m_y0']:.4f}, intercept={lf['c_y0']:.4f}\n"
            f"ys linear fit: slope={lf['m_ys']:.4f}, intercept={lf['c_ys']:.4f}\n"
        )
        if self.theta_sim_deg != 0:
            fit_info += f"raw data theta_deg rotation = {self.theta_sim_deg:.2f}"

        ax_text.text(0.5, 0.95, fit_info, ha='center', va='top', fontsize=12, wrap=True)

        # Table
        table_data = np.column_stack([lf["indices"], lf["z0_um"], lf["theta_deg"], lf["y0_um"], lf["ys_um"]])
        column_labels = ["Z stage", "z0 (µm)", "θ (deg)", "y0 (µm)", "ys (µm)"]
        table = ax_text.table(
            cellText=np.round(table_data,3),
            colLabels=column_labels,
            cellLoc='center', colLoc='center',
            bbox=[0.0, -0.1, 1, 0.5]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1,1.5)

        # Legend below
        lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2]]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines, labels, loc='lower center', ncol=len(labels), frameon=False, fontsize=10)

        plt.suptitle(f"Rigid body modeling, dataset: {self.folder_path}", fontsize=16)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        if save:
            fname = os.path.join(self.folder_path,
                                 f"rigid_body_model_mastronarde_{os.path.split(self.folder_path)[1]}.png")
            plt.savefig(fname)
        plt.show()

    # def plot_trajectories_stacked(
    #         self,
    #         dy_offset=1.0,
    #         out_path=None,
    #         filename="trajectories_all_datasets_stacked.png"
    # ):
    #     """
    #     Plot stacked rotated XY trajectories with Mastronarde model overlay
    #     """
    #
    #     plt.figure(figsize=(10, 8))
    #     colors = plt.cm.viridis(np.linspace(0, 1, len(self.datasets)))
    #
    #     x0_positions = []
    #     dataset_numbers = []
    #
    #     for i, ds in enumerate(self.datasets):
    #         alpha = ds["alpha_rad"]
    #         x = ds["x_ccd"]
    #         y = ds["y_ccd"]
    #
    #
    #
    #         y0, z0, ys, theta_deg, y_mean = ds["popt"]
    #
    #         # rotate measured data back by fitted theta, ok
    #         theta_rad = -np.deg2rad(theta_deg)
    #         y_rot = x * np.sin(theta_rad) + y * np.cos(theta_rad)
    #         x_rot = x * np.cos(theta_rad) - y * np.sin(theta_rad)
    #
    #         # model trajectory
    #         y_model_pos, x_model_pos = self.mastronarde_point(
    #             alpha, y0, z0, ys, theta_deg, y_mean
    #         )
    #         # model trajectory
    #         y_model_neg, x_model_neg = self.mastronarde_point(
    #             alpha, y0, z0, ys, theta_deg, y_mean
    #         )
    #
    #
    #         # vertical stacking
    #         y_shifted = y_rot + i * dy_offset
    #         y_model_shifted = y_model + i * dy_offset + dy_offset/6
    #
    #         plt.plot(
    #             y_shifted, x_rot, "o-",
    #             color=colors[i], alpha=0.6,
    #             label=f"Data {ds['index']}"
    #         )
    #         plt.plot(
    #             y_model_shifted, x_model, "--",
    #             color="k", linewidth=1.5
    #         )
    #
    #         # α = 0 marker
    #         mid = len(alpha) // 2
    #         plt.plot(y_shifted[mid], x_rot[mid], "ko", markersize=6)
    #
    #         x0_positions.append(y_shifted[mid])
    #         dataset_numbers.append(ds["index"])
    #
    #     plt.xlabel("Z stage (µm)")
    #     plt.ylabel("Δy (µm)")
    #     plt.title("trajectory_stacked")
    #     plt.grid(True)
    #     # plt.legend()
    #     # plt.xlim(-200, 200)
    #     plt.xticks(x0_positions, dataset_numbers)
    #     plt.tight_layout()
    #
    #     if out_path is not None:
    #         plt.savefig(os.path.join(out_path, filename), dpi=1000)
    #
    #     plt.show()
    def plot_trajectories_stacked(
            self,
            dy_offset=1.0,
            out_path=None,
            filename="trajectories_all_datasets_stacked.png"
    ):
        """
        Plot stacked rotated XY trajectories with Mastronarde model overlay.
        Theoretical model is split for positive and negative alpha values
        and plotted with a small offset to inspect symmetry.
        """

        plt.figure(figsize=(10, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.datasets)))

        x0_positions = []
        dataset_numbers = []

        sym_offset = dy_offset * 0.05  # small separation between +α and −α models

        for i, ds in enumerate(self.datasets):
            alpha = ds["alpha_rad"]  # radians
            x = ds["x_ccd"]
            y = ds["y_ccd"]

            y0, z0, ys, theta_deg, y_mean = ds["popt"]

            # rotate measured data back by fitted theta
            theta_rad = -np.deg2rad(theta_deg)
            y_rot = x * np.sin(theta_rad) + y * np.cos(theta_rad)
            x_rot = x * np.cos(theta_rad) - y * np.sin(theta_rad)

            # evaluate theoretical model ONCE
            y_model, x_model = self.mastronarde_point(
                alpha, y0, z0, ys, theta_deg, y_mean
            )

            # split by alpha sign
            pos = alpha >= 0
            neg = alpha < 0

            # vertical stacking
            y_shifted = y_rot + i * dy_offset
            y_model_shifted = y_model + i * dy_offset + dy_offset / 6

            # measured data
            plt.plot(
                y_shifted, x_rot, "o-",
                color=colors[i], alpha=0.6,
                label=f"Data {ds['index']}"
            )

            # theoretical model (+alpha)
            plt.plot(
                y_model_shifted[pos] + sym_offset,
                x_model[pos],
                "--",
                color="red",
                linewidth=1.5,
                alpha=0.8
            )

            # theoretical model (-alpha)
            plt.plot(
                y_model_shifted[neg] - sym_offset,
                x_model[neg],
                "--",
                color="blue",
                linewidth=1.5,
                alpha=0.8
            )

            # α = 0 marker (middle point)
            mid = len(alpha) // 2
            plt.plot(y_shifted[mid], x_rot[mid], "ko", markersize=6)

            x0_positions.append(y_shifted[mid])
            dataset_numbers.append(ds["index"])

        plt.xlabel("Z stage (µm)")
        plt.ylabel("Δy (µm)")
        plt.title("trajectory_stacked")
        plt.grid(True)

        plt.xticks(x0_positions, dataset_numbers)
        plt.tight_layout()

        if out_path is not None:
            plt.savefig(os.path.join(out_path, filename), dpi=1000)

        plt.show()

    def plot_z_scan(
            self,
            alpha_range_deg=np.linspace(-60, 60, 300),
            out_path=None,
            filename="plot_z_scan.png"
    ):
        """
        Plot n–z arcs using Mastronarde rigid body model
        """

        plt.figure(figsize=(6, 6))

        z0_dataset = []
        n0_dataset = []

        alpha_range_rad = np.deg2rad(alpha_range_deg)

        for ds in self.datasets:
            y0, z0, ys, theta_deg, y_mean = ds["popt"] # these are correctly in um
            idx = ds["index"]

            # # --- phase shift forcing n(alpha=0)=0 ---
            # phase_shift = np.arctan2(-y0, z0)
            #
            # if np.sign(z0) == -1:
            #     phase_shift += np.pi
            # phase_shift = - np.arctan2(z0, y0+ys)
            phase_shift = 0
            alpha_corr = alpha_range_rad + phase_shift


            # --- model arc ---
            n_corr, z_corr = self.zheng_point(
                alpha_corr, y0, z0, ys, theta_deg, y_mean
            )


            plt.plot(n_corr, z_corr, label=f"z = {idx}")
            # plt.text(
            #     n_corr[-1], z_corr[-1],
            #     f"z = {idx}", fontsize=8,
            #     verticalalignment="top", horizontalalignment="left"
            # )

            # sample reference point (e.g. α = 55°)
            alpha_ref = np.deg2rad(0) + phase_shift
            n_ref, z_ref = self.zheng_point(
                alpha_ref, y0, z0, ys, theta_deg, y_mean
            )
            plt.plot(n_ref, z_ref, "kx")
            z0_dataset.append(float(z_ref))
            n0_dataset.append(float(n_ref))

            # --- model arc --- also with the contribution of ys
            # n_corr_ys, z_corr_ys = self.zheng_point(
            #     alpha_corr, y0+ys, z0, ys, theta_deg, y_mean
            # )
            # plt.plot(y0+ys, z0, "rx")
            # plt.plot(n_corr_ys, z_corr_ys, label=f"z = {idx}_with_y0+ys")

        plt.plot(0, 0, "ro")
        plt.axis("equal")
        plt.xlabel("Δn (µm)")
        plt.ylabel("Δz (µm)")
        plt.title("plot z scan (arcs)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if out_path is not None:
            plt.savefig(os.path.join(out_path, filename), dpi=1000)

        plt.show()

        return np.array(n0_dataset), np.array(z0_dataset)

    def plot_single_dataset_summary(self, dataset_index=0, save=False):
        """
        Summary plots for a SINGLE dataset.
        """

        ds = self._get_dataset(dataset_index)

        alpha = ds["alpha_rad"]

        alpha_deg = np.rad2deg(alpha)

        x = ds["x_ccd"]
        y = ds["y_ccd"]
        y0, z0, ys, theta_deg, y_mean = ds["popt"]
        idx = ds["index"]

        # -----------------------------
        # Rotate measured data back
        # -----------------------------
        theta_rad = -np.deg2rad(theta_deg)
        x_rot = x * np.cos(theta_rad) - y * np.sin(theta_rad)
        y_rot = x * np.sin(theta_rad) + y * np.cos(theta_rad)

        # -----------------------------
        # Mastronarde model trajectory
        # -----------------------------
        y_model, x_model = self.mastronarde_point(
            alpha, y0, z0, ys, theta_deg, y_mean
        )

        # -----------------------------
        # n–z arc
        # -----------------------------
        alpha_range = np.linspace(alpha.min(), alpha.max(), 400)
        n_arc, z_arc = self.zheng_point(
            alpha_range, y0, z0, ys, theta_deg, y_mean
        )

        n0, z0_ref = self.zheng_point(
            0.0, y0, z0, ys, theta_deg, y_mean
        )

        # =============================
        # FIGURE LAYOUT
        # =============================
        fig = plt.figure(figsize=(16, 9))
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.2])

        ax_xy = fig.add_subplot(gs[:, 0])
        ax_tilt = fig.add_subplot(gs[0, 1])
        ax_arc = fig.add_subplot(gs[1, 1])
        ax_text = fig.add_subplot(gs[:, 2])

        # -----------------------------
        # XY plot
        # -----------------------------
        ax_xy.plot(x, y, "o", alpha=0.5, label="Raw XY")
        ax_xy.plot(x_rot, y_rot, "o", label="Rotated XY")
        ax_xy.plot(x_model, y_model, "--k", label="Model")
        ax_xy.axhline(0, color="k", ls="--", alpha=0.4)
        ax_xy.axvline(0, color="k", ls="--", alpha=0.4)
        ax_xy.set_aspect("equal")
        ax_xy.set_xlim([-self.pixelsize_um*512, self.pixelsize_um*512])
        ax_xy.set_ylim([-self.pixelsize_um*512, self.pixelsize_um*512])
        ax_xy.set_xlabel("X (µm)")
        ax_xy.set_ylabel("Y (µm)")
        ax_xy.set_title("XY trajectory")
        ax_xy.legend()
        ax_xy.text(
            x_rot[0], y_rot[0],
            f"α={alpha_deg[0]:.1f}°",
            fontsize=9, ha="left", va="bottom"
        )
        ax_xy.text(
            x_rot[-1], y_rot[-1],
            f"α={alpha_deg[-1]:.1f}°",
            fontsize=9, ha="right", va="top"
        )

        # -----------------------------
        # Tilt plot
        # -----------------------------
        ax_tilt.plot(np.rad2deg(alpha), x, "o", label="X")
        ax_tilt.plot(np.rad2deg(alpha), y, "o", label="Y")
        ax_tilt.plot(np.rad2deg(alpha), x_model, "--k")
        ax_tilt.plot(np.rad2deg(alpha), y_model, "--k", label = "model")
        ax_tilt.set_xlabel("Tilt α (deg)")
        ax_tilt.set_ylabel("Position (µm)")
        ax_tilt.set_title("shift as function of angle")
        ax_tilt.legend()
        ax_tilt.grid(True)

        ax_tilt.text(
            alpha_deg[0], x[0],
            f"α={alpha_deg[0]:.1f}°",
            fontsize=9, ha="left", va="bottom"
        )
        ax_tilt.text(
            alpha_deg[-1], x[-1],
            f"α={alpha_deg[-1]:.1f}°",
            fontsize=9, ha="right", va="top"
        )

        # -----------------------------
        # n–z arc plot
        # -----------------------------
        ax_arc.plot(n_arc, z_arc, label="Model")
        ax_arc.plot(n0, z0_ref, "ro", label="P(n0, z0)")
        ax_arc.axhline(0, color="k", ls="--", alpha=0.3)
        ax_arc.axvline(0, color="k", ls="--", alpha=0.3)
        ax_arc.set_xlabel("Δn (µm)")
        ax_arc.set_ylabel("Δz (µm)")
        ax_arc.set_title("n–z arc")
        ax_arc.set_aspect("equal")
        ax_arc.set_ylim([-4, 4])
        ax_arc.set_xlim([-4, 4])
        ax_arc.legend()
        ax_arc.grid(True)

        ax_arc.text(
            n_arc[0], z_arc[0],
            f"α={np.rad2deg(alpha_range[0]):.1f}°",
            fontsize=9, ha="left", va="bottom"
        )
        ax_arc.text(
            n_arc[-1], z_arc[-1],
            f"α={np.rad2deg(alpha_range[-1]):.1f}°",
            fontsize=9, ha="right", va="top"
        )
        # -----------------------------
        # Text + table
        # -----------------------------
        ax_text.axis("off")

        text = (
            f"Dataset: {idx}\n"
            f"Pixelsize = {self.pixelsize_nm} nm\n\n"
            f"y0   = {y0:.4f} µm\n"
            f"z0   = {z0:.4f} µm\n"
            f"ys   = {ys:.4f} µm\n"
            f"θ    = {theta_deg:.4f} deg\n\n"
            f"n(α=0) = {float(n0):.4f} µm\n"
            f"z(α=0) = {float(z0_ref):.4f} µm\n\n"
            f"Expected eucentric height = {idx + z0:.4f} µm\n"
        )

        ax_text.text(
            0.05, 0.95, text,
            ha="left", va="top",
            fontsize=12
        )



        table_data = [[
            idx,
            round(z0, 4),
            round(theta_deg, 4),
            round(y0, 4),
            round(ys, 4)
        ]]

        table = ax_text.table(
            cellText=table_data,
            colLabels=["Z idx", "z0 (µm)", "θ (deg)", "y0 (µm)", "ys (µm)"],
            loc="center",
            cellLoc="center"
        )

        table.scale(1, 1.5)
        table.auto_set_font_size(False)
        table.set_fontsize(10)

        # -----------------------------
        plt.suptitle("Single dataset rigid-body fit", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save:
            fname = os.path.join(
                self.folder_path,
                f"single_dataset_{idx}_summary.png"
            )
            plt.savefig(fname, dpi=300)

        plt.show()

    def analyze_xy_dynamics(
            self,
            dataset_index=0,
            n_alpha=500,
            deg_step=5,
            out_path=None, scaling = 1, original_range = False, deg_markers = False, pixels = True):
        """
        Analyze velocity and acceleration of projected CCD motion
        assuming linear relation between tilt angle and time
        (1 deg = 1 time unit).

        Markers are placed EXACTLY every `deg_step` degrees.
        """

        ds = self._get_dataset(dataset_index)

        y0, z0, ys, theta_deg, y_mean = ds["popt"]
        idx = ds["index"]

        # ---------------------------------
        # Dense alpha grid (continuous model)
        # ---------------------------------
        if original_range == True:
            alpha = np.linspace(
                ds["alpha_rad"].min(),
                ds["alpha_rad"].max(),
                n_alpha
            )
        else:
            alpha = np.linspace(np.deg2rad(-180), np.deg2rad(180), 1000)


        alpha_deg = np.rad2deg(alpha)
        t = scaling*(alpha_deg - alpha_deg.min())  # time axis

        # ---------------------------------
        # Model trajectory on CCD
        # ---------------------------------
        y_model, x_model = self.mastronarde_point(
            alpha, y0, z0, ys, theta_deg, y_mean
        )
        if pixels == True:
            y_model = y_model / self.pixelsize_um
            x_model = x_model / self.pixelsize_um
        # ---------------------------------
        # Derivatives (continuous)
        # ---------------------------------
        dx_dt = np.gradient(x_model, t)
        dy_dt = np.gradient(y_model, t)

        d2x_dt2 = np.gradient(dx_dt, t)
        d2y_dt2 = np.gradient(dy_dt, t)

        v_mag = np.sqrt(dx_dt ** 2 + dy_dt ** 2)
        a_mag = np.sqrt(d2x_dt2 ** 2 + d2y_dt2 ** 2)

        # ---------------------------------
        # EXACT 5° sampling (robust)
        # ---------------------------------
        alpha5_deg = np.arange(
            np.ceil(alpha_deg.min() / deg_step) * deg_step,
            np.floor(alpha_deg.max() / deg_step) * deg_step + 0.1,
            deg_step
        )
        alpha5_rad = np.deg2rad(alpha5_deg)
        t5 = alpha5_deg - alpha5_deg.min()

        y5, x5 = self.mastronarde_point(
            alpha5_rad, y0, z0, ys, theta_deg, y_mean
        )
        if pixels == True:
            y5 = y5 / self.pixelsize_um
            x5 = x5 / self.pixelsize_um
        dx5_dt = np.gradient(x5, t5)
        dy5_dt = np.gradient(y5, t5)

        d2x5_dt2 = np.gradient(dx5_dt, t5)
        d2y5_dt2 = np.gradient(dy5_dt, t5)

        v5_mag = np.sqrt(dx5_dt ** 2 + dy5_dt ** 2)
        a5_mag = np.sqrt(d2x5_dt2 ** 2 + d2y5_dt2 ** 2)

        # ---------------------------------
        # Statistics
        # ---------------------------------
        stats = {
            "v_min": float(v_mag.min()),
            "v_max": float(v_mag.max()),
            "v_mean": float(v_mag.mean()),
            "a_min": float(a_mag.min()),
            "a_max": float(a_mag.max()),
            "a_mean": float(a_mag.mean())
        }

        print(f"\n[XY dynamics — dataset {idx}]")
        for k, v in stats.items():
            print(f"{k}: {v:.4e}")

        # ---------------------------------
        # PLOTS
        # ---------------------------------
        fig, axs = plt.subplots(2, 2, figsize=(15, 11))

        # -------------------------
        # (0,0) Trajectory
        # -------------------------
        axs[0, 0].plot(x_model, y_model, "-k", label="Model trajectory")
        if deg_markers == True:
            axs[0, 0].plot(x5, y5, "ro", markersize=6, markerfacecolor="none", label=f"{deg_step}° markers")

            # Annotate angle on every 2nd 5° marker (i.e., every 10°)
            for i, (xi, yi, ang) in enumerate(zip(x5, y5, alpha5_deg)):
                if i % 2 == 0:  # every second point
                    if original_range == False and ang == 360:
                        continue
                    else:
                        axs[0, 0].text(
                            xi,
                            yi + 0.10,  # small offset in Y
                            f"{int(ang)}°",
                            fontsize=8,
                            color="black",
                            alpha=0.8
                        )

        axs[0, 0].set_title("Projected CCD trajectory")
        axs[0, 0].set_xlabel("X (µm)")
        axs[0, 0].set_ylabel("Y (µm)")
        axs[0, 0].set_ylim([-2,2])
        axs[0, 0].grid(True)
        axs[0, 0].legend()

        # -------------------------
        # (0,1) |v| and |a| with twin y-axis
        # -------------------------
        ax_v = axs[0, 1]
        ax_a = ax_v.twinx()

        # --- velocity ---
        ax_v.plot(alpha_deg, v_mag, "r-", label="|v|")
        if deg_markers == True:
            ax_v.plot(alpha5_deg, v5_mag, "ro", markerfacecolor="none")
        ax_v.set_ylabel("Velocity |v|", color="red")
        ax_v.tick_params(axis="y", labelcolor="red")

        # --- acceleration ---
        ax_a.plot(alpha_deg, a_mag, "b-", label="|a|")
        if deg_markers == True:
            ax_a.plot(alpha5_deg, a5_mag, "bo", markerfacecolor="none")
        ax_a.set_ylabel("Acceleration |a|", color="blue")
        ax_a.tick_params(axis="y", labelcolor="blue")

        ax_v.set_xlabel("Tilt α (deg)")
        ax_v.grid(True)
        if pixels == True:
            limit_ = 10
        else:
            limit_ = 0.1
        ax_v.set_ylim([0, limit_])
        ax_a.set_ylim([0, limit_/100*3])
        if original_range == False:
            ax_v.set_xlim([-180, 180])
            ax_a.set_xlim([-180, 180])
        # -------------------------
        # Colored title (manual)
        # -------------------------
        ax_v.set_title("")  # clear default title

        ax_v.text(
            0.4, 1.08, "Velocity",
            color="red",
            fontsize=12,
            fontweight="bold",
            ha="right",
            va="bottom",
            transform=ax_v.transAxes
        )

        ax_v.text(
            0.5, 1.08, " & ",
            color="black",
            fontsize=12,
            ha="center",
            va="bottom",
            transform=ax_v.transAxes
        )

        ax_v.text(
            0.6, 1.08, "Acceleration",
            color="blue",
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="bottom",
            transform=ax_v.transAxes
        )

        # -------------------------
        # (1,0) Velocity components
        # -------------------------
        axs[1, 0].plot(alpha_deg, dx_dt, label="vₓ")
        axs[1, 0].plot(alpha_deg, dy_dt, label="vᵧ")

        if deg_markers == True:
            axs[1, 0].plot(alpha5_deg, dx5_dt, "o", markerfacecolor="none")
            axs[1, 0].plot(alpha5_deg, dy5_dt, "o", markerfacecolor="none")

        axs[1, 0].set_title("Velocity components")
        axs[1, 0].set_xlabel("Tilt α (deg)")
        axs[1, 0].set_ylabel("Velocity")
        axs[1, 0].grid(True)
        axs[1, 0].legend()

        # -------------------------
        # (1,1) Acceleration components
        # -------------------------
        axs[1, 1].plot(alpha_deg, d2x_dt2, label="aₓ")
        axs[1, 1].plot(alpha_deg, d2y_dt2, label="aᵧ")

        if deg_markers == True:
            axs[1, 1].plot(alpha5_deg, d2x5_dt2, "o", markerfacecolor="none")
            axs[1, 1].plot(alpha5_deg, d2y5_dt2, "o", markerfacecolor="none")

        axs[1, 1].set_title("Acceleration components")
        axs[1, 1].set_xlabel("Tilt α (deg)")
        axs[1, 1].set_ylabel("Acceleration")
        axs[1, 1].grid(True)
        axs[1, 1].legend()

        plt.suptitle(
            f"Projected CCD dynamics (1° = 1 time unit) — dataset {idx}",
            fontsize=16
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if out_path is not None:
            fname = os.path.join(out_path, f"xy_dynamics_dataset_{idx}.png")
            plt.savefig(fname, dpi=300)

        plt.show()

        return stats

    def analyze_xy_dynamics_multiple(
            self,
            deg_step=5,
            n_alpha=500,
            scaling=1,
            original_range=False,
            deg_markers=False,
            out_path=None,
            pixels = True,
    ):
        """
        Analyze XY dynamics for all datasets and compute max velocity & acceleration.
        Also plots max |v| and |a| vs Z stage (dataset index).

        Includes the same style as `analyze_xy_dynamics`:
        - Optional deg markers with text
        - Velocity & acceleration twin y-axis
        - Component plots
        """
        if len(self.datasets) == 0:
            raise RuntimeError("No datasets available. Run `run_fits()` first.")

        z_stage = []
        v_max_all = []
        a_max_all = []

        for ds in self.datasets:
            y0, z0, ys, theta_deg, y_mean = ds["popt"]
            idx = ds["index"]
            # ys = 0.1 * ys
            # Dense alpha grid
            if original_range:
                alpha = np.linspace(ds["alpha_rad"].min(), ds["alpha_rad"].max(), n_alpha)
            else:
                alpha = np.linspace(np.deg2rad(-180), np.deg2rad(180), 1000)

            alpha_deg = np.rad2deg(alpha)
            t = scaling * (alpha_deg - alpha_deg.min())

            # Model trajectory
            y_model, x_model = self.mastronarde_point(alpha, y0, z0, ys, theta_deg, y_mean)

            if pixels == True:
                y_model = y_model/self.pixelsize_um
                x_model = x_model / self.pixelsize_um

            # Derivatives
            dx_dt = np.gradient(x_model, t)
            dy_dt = np.gradient(y_model, t)
            d2x_dt2 = np.gradient(dx_dt, t)
            d2y_dt2 = np.gradient(dy_dt, t)

            v_mag = np.sqrt(dx_dt ** 2 + dy_dt ** 2)
            a_mag = np.sqrt(d2x_dt2 ** 2 + d2y_dt2 ** 2)

            # Exact 5° sampling
            alpha5_deg = np.arange(
                np.ceil(alpha_deg.min() / deg_step) * deg_step,
                np.floor(alpha_deg.max() / deg_step) * deg_step + 0.1,
                deg_step
            )
            alpha5_rad = np.deg2rad(alpha5_deg)
            t5 = alpha5_deg - alpha5_deg.min()

            y5, x5 = self.mastronarde_point(alpha5_rad, y0, z0, ys, theta_deg, y_mean)
            if pixels == True:
                y5 = y5/self.pixelsize_um
                x5 = x5 / self.pixelsize_um

            dx5_dt = np.gradient(x5, t5)
            dy5_dt = np.gradient(y5, t5)
            d2x5_dt2 = np.gradient(dx5_dt, t5)
            d2y5_dt2 = np.gradient(dy5_dt, t5)
            v5_mag = np.sqrt(dx5_dt ** 2 + dy5_dt ** 2)
            a5_mag = np.sqrt(d2x5_dt2 ** 2 + d2y5_dt2 ** 2)

            # Statistics
            stats = {
                "v_min": float(v_mag.min()),
                "v_max": float(v_mag.max()),
                "v_mean": float(v_mag.mean()),
                "a_min": float(a_mag.min()),
                "a_max": float(a_mag.max()),
                "a_mean": float(a_mag.mean())
            }

            print(f"\n[XY dynamics — dataset {idx}]")
            for k, v in stats.items():
                print(f"{k}: {v:.4e}")

            z_stage.append(idx)
            v_max_all.append(stats["v_max"])
            a_max_all.append(stats["a_max"])

            # -------------------------
            # Plot single dataset
            # -------------------------
            fig, axs = plt.subplots(2, 2, figsize=(15, 11))

            # Trajectory
            axs[0, 0].plot(x_model, y_model, "-k", label="Model trajectory")
            if deg_markers:
                axs[0, 0].plot(x5, y5, "ro", markersize=6, markerfacecolor="none", label=f"{deg_step}° markers")
                for i, (xi, yi, ang) in enumerate(zip(x5, y5, alpha5_deg)):
                    if i % 2 == 0:  # annotate every 2 markers
                        if original_range is False and ang == 360:
                            continue
                        axs[0, 0].text(xi, yi + 0.10, f"{int(ang)}°", fontsize=8, color="black", alpha=0.8)

            axs[0, 0].set_title("Projected CCD trajectory")
            axs[0, 0].set_xlabel("X (µm)")
            axs[0, 0].set_ylabel("Y (µm)")
            axs[0, 0].grid(True)
            axs[0, 0].legend()

            # Velocity & acceleration magnitude
            ax_v = axs[0, 1]
            ax_a = ax_v.twinx()
            ax_v.plot(alpha_deg, v_mag, "r-", label="|v|")
            ax_a.plot(alpha_deg, a_mag, "b-", label="|a|")
            if deg_markers:
                ax_v.plot(alpha5_deg, v5_mag, "ro", markerfacecolor="none")
                ax_a.plot(alpha5_deg, a5_mag, "bo", markerfacecolor="none")
            ax_v.set_xlabel("Tilt α (deg)")
            ax_v.set_ylabel("Velocity |v|", color="red")
            ax_a.set_ylabel("Acceleration |a|", color="blue")
            ax_v.tick_params(axis="y", labelcolor="red")
            ax_a.tick_params(axis="y", labelcolor="blue")
            if pixels == True:
                limit_ = 10
            else:
                limit_ = 0.1
            ax_v.set_ylim([0, limit_])
            ax_a.set_ylim([0, limit_ / 100 * 3])
            ax_v.grid(True)

            # Colored title
            ax_v.set_title("")
            ax_v.text(0.4, 1.08, "Velocity", color="red", fontsize=12, fontweight="bold", ha="right", va="bottom",
                      transform=ax_v.transAxes)
            ax_v.text(0.5, 1.08, " & ", color="black", fontsize=12, ha="center", va="bottom", transform=ax_v.transAxes)
            ax_v.text(0.6, 1.08, "Acceleration", color="blue", fontsize=12, fontweight="bold", ha="left", va="bottom",
                      transform=ax_v.transAxes)

            # Velocity components
            axs[1, 0].plot(alpha_deg, dx_dt, label="vₓ")
            axs[1, 0].plot(alpha_deg, dy_dt, label="vᵧ")
            if deg_markers:
                axs[1, 0].plot(alpha5_deg, dx5_dt, "o", markerfacecolor="none")
                axs[1, 0].plot(alpha5_deg, dy5_dt, "o", markerfacecolor="none")
            axs[1, 0].set_title("Velocity components")
            axs[1, 0].set_xlabel("Tilt α (deg)")
            axs[1, 0].set_ylabel("Velocity")
            axs[1, 0].grid(True)
            axs[1, 0].legend()

            # Acceleration components
            axs[1, 1].plot(alpha_deg, d2x_dt2, label="aₓ")
            axs[1, 1].plot(alpha_deg, d2y_dt2, label="aᵧ")
            if deg_markers:
                axs[1, 1].plot(alpha5_deg, d2x5_dt2, "o", markerfacecolor="none")
                axs[1, 1].plot(alpha5_deg, d2y5_dt2, "o", markerfacecolor="none")
            axs[1, 1].set_title("Acceleration components")
            axs[1, 1].set_xlabel("Tilt α (deg)")
            axs[1, 1].set_ylabel("Acceleration")
            axs[1, 1].grid(True)
            axs[1, 1].legend()

            plt.suptitle(f"Projected CCD dynamics — dataset {idx}", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            if out_path is not None:
                fname = os.path.join(out_path, f"xy_dynamics_dataset_{idx}.png")
                plt.savefig(fname, dpi=300)

            plt.show()

        # -------------------------
        # Plot max |v| and |a| vs Z stage with twin y-axis
        # -------------------------
        z_stage = np.array(z_stage)
        v_max_all = np.array(v_max_all)
        a_max_all = np.array(a_max_all)

        fig, ax_v = plt.subplots(figsize=(10, 6))
        ax_a = ax_v.twinx()

        # Plot velocity
        ax_v.plot(z_stage, v_max_all, "ro-", label="max |v|")
        ax_v.set_xlabel("Dataset / Z stage")
        ax_v.set_ylabel("Max velocity |v|", color="red")
        ax_v.tick_params(axis="y", labelcolor="red")

        if pixels == True:
            limit_ = 6
        else:
            limit_ = 0.2
        ax_v.set_ylim([0, limit_])
        ax_a.set_ylim([0, limit_/10])

        ax_v.grid(True)

        # Plot acceleration
        ax_a.plot(z_stage, a_max_all, "bo-", label="max |a|")
        ax_a.set_ylabel("Max acceleration |a|", color="blue")
        ax_a.tick_params(axis="y", labelcolor="blue")

        # Colored title
        ax_v.set_title("Max velocity & acceleration vs Z stage", fontsize=14)
        ax_v.text(0.4, 1.08, "Velocity", color="red", fontsize=12, fontweight="bold",
                  ha="right", va="bottom", transform=ax_v.transAxes)
        ax_v.text(0.5, 1.08, " & ", color="black", fontsize=12, ha="center", va="bottom", transform=ax_v.transAxes)
        ax_v.text(0.6, 1.08, "Acceleration", color="blue", fontsize=12, fontweight="bold",
                  ha="left", va="bottom", transform=ax_v.transAxes)

        fig.tight_layout(rect=[0, 0, 1, 0.95])

        if out_path is not None:
            fname = os.path.join(out_path, "max_v_a_vs_z.png")
            plt.savefig(fname, dpi=300)

        plt.show()

        return {"z_stage": z_stage, "v_max": v_max_all, "a_max": a_max_all}

if __name__ == "__main__":

    model = MastronardeRigidBody("", range(1), 1, plot_intermediate=False)
    model.fit_single_dataset_from_gui(data_path = None, switch_axis = False)
