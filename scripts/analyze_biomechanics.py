
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import os
from scipy.interpolate import interp1d

# Set style for publication-quality plots
try:
    plt.style.use('seaborn-v0_8-paper')
except OSError:
    # Fallback if specific style not found
    plt.style.use('ggplot')

# Custom params for "PhD style"
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 1.5,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'savefig.dpi': 300,
    'figure.figsize': (8, 6),
    'axes.prop_cycle': plt.cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
})

class BiomechanicsAnalysisSuite:
    def __init__(self, data_path, output_dir=None):
        self.data_path = Path(data_path)
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.data_path.parent / "plots"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data = self.load_data()
        
        # Parse metadata
        self.fps = int(self.data['fps'])
        self.dt = float(self.data['dt'])
        self.dof_names = [str(n) for n in self.data['dof_names']]
        self.body_names = [str(n) for n in self.data['body_names']]
        self.time = np.arange(self.data['dof_pos'].shape[0]) * self.dt

        # Process data (squeeze env dim if 1)
        # Data shape: (Time, Envs, ...)
        self.dof_pos = self.data['dof_pos']
        self.dof_vel = self.data['dof_vel']
        self.dof_forces = self.data['dof_forces']
        self.grf = self.data['rigid_body_contact_forces']
        
        # IMU Data
        self.imu_acc = self.data.get('imu_lin_acc', None)
        self.imu_gyro = self.data.get('imu_ang_vel', None)
        
        # Check for multiple environments
        self.num_envs = self.dof_pos.shape[1]
        self.is_multi_env = self.num_envs > 1
        
        # Gait Events
        self.gait_events = {} # 'Left': {'mask': ..., 'cycles': [{'start': t, 'end': t, 'type': 'stance'}]}
        self.gait_cycles = {} # 'Left': {'stance': [[start_idx, end_idx], ...], 'stride': [[hs_idx, next_hs_idx], ...]}

    def load_data(self):
        print(f"Loading data from {self.data_path}...")
        try:
            return np.load(self.data_path, allow_pickle=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}")

    def _get_dof_indices(self, joint_substrings):
        """Find indices for joints matching substrings (e.g., 'Hip', 'Knee')."""
        indices = []
        names = []
        for i, name in enumerate(self.dof_names):
            if any(sub in name for sub in joint_substrings):
                indices.append(i)
                names.append(name)
        return indices, names

    def detect_gait_events(self, force_threshold=10.0, min_stance_duration=0.1, min_gap_duration=0.05):
        """
        Detect stance and swing phases with cleaning logic.
        
        Args:
            force_threshold: Force (N) to consider contact.
            min_stance_duration: Minimum duration (s) for a valid stance.
            min_gap_duration: Gaps shorter than this (s) within a stance are filled (ignored).
        """
        feet_keywords = ["Ankle", "Toe", "Foot"]
        min_stance_frames = int(min_stance_duration / self.dt)
        min_gap_frames = int(min_gap_duration / self.dt)
        
        # Find indices for left and right feet bodies
        left_indices = [i for i, n in enumerate(self.body_names) if "L_" in n and any(k in n for k in feet_keywords)]
        right_indices = [i for i, n in enumerate(self.body_names) if "R_" in n and any(k in n for k in feet_keywords)]
        
        if not left_indices and not right_indices:
            print("Warning: Could not identify feet bodies for gait detection.")
            return

        def process_side(indices, side_name):
            # 1. Raw Thresholding
            # Sum vertical forces
            v_force = self.grf[:, :, indices, 2].sum(axis=2) # (Time, Envs)
            # Use first env for event detection for now (simplification for visualization)
            raw_mask = v_force[:, 0] > force_threshold
            
            # 2. Gap Filling (Close small holes)
            # Find gaps (0s surrounded by 1s)
            padded = np.concatenate(([True], raw_mask, [True]))
            diff = np.diff(padded.astype(int))
            starts = np.where(diff == -1)[0] # Start of a gap (1->0)
            ends = np.where(diff == 1)[0]   # End of a gap (0->1)
            
            filled_mask = raw_mask.copy()
            for s, e in zip(starts, ends):
                if (e - s) <= min_gap_frames:
                    filled_mask[s:e] = True # Fill gap
            
            # 3. Minimum Duration Filter (Remove short spikes)
            padded = np.concatenate(([False], filled_mask, [False]))
            diff = np.diff(padded.astype(int))
            starts = np.where(diff == 1)[0] # Start of stance
            ends = np.where(diff == -1)[0]  # End of stance
            
            final_mask = np.zeros_like(filled_mask, dtype=bool)
            stance_cycles = [] # List of [start_idx, end_idx] (exclusive end)
            
            for s, e in zip(starts, ends):
                if (e - s) >= min_stance_frames:
                    final_mask[s:e] = True
                    stance_cycles.append([s, e])
            
            # 4. Identify Stride Cycles (Heel Strike to Heel Strike)
            # Stride N is from stance_cycles[N][0] to stance_cycles[N+1][0]
            stride_cycles = []
            for i in range(len(stance_cycles) - 1):
                stride_cycles.append([stance_cycles[i][0], stance_cycles[i+1][0]])
                
            self.gait_events[side_name] = final_mask[:, np.newaxis] # Reshape to match (Time, 1) expectation
            self.gait_cycles[side_name] = {
                'stance': stance_cycles,
                'stride': stride_cycles
            }
            
            print(f"  {side_name}: Found {len(stance_cycles)} stance phases and {len(stride_cycles)} stride cycles.")

        if left_indices:
            process_side(left_indices, 'Left')
        if right_indices:
            process_side(right_indices, 'Right')

    def time_normalize(self, data_segment, num_points=101):
        """Normalize data segment to fixed number of points (0-100%)."""
        # data_segment shape: (Time, Envs)
        x_old = np.linspace(0, 1, data_segment.shape[0])
        x_new = np.linspace(0, 1, num_points)
        
        # Interpolate for each env
        normalized = np.zeros((num_points, data_segment.shape[1]))
        for i in range(data_segment.shape[1]):
            f = interp1d(x_old, data_segment[:, i], kind='linear')
            normalized[:, i] = f(x_new)
            
        return normalized

    def _plot_gait_events(self, ax, side='Right', alpha=0.1, color='gray'):
        """Overlay gait events (stance phase) as shaded regions."""
        if side not in self.gait_events:
            return

        # events shape: (Time, Envs)
        # For multi-env, we can plot the MEAN stance phase or just the first trial.
        # Plotting probabilistic stance for multi-env is complex visually.
        # Simplified approach: Use trial 0 for visualization if single env, 
        # or aggregate percentage for multi-env?
        # Let's stick to Trial 0 for clarity in segmentation visualization 
        # unless specifically asked for probabilistic.
        
        stance_mask = self.gait_events[side][:, 0] # Take first env
        
        # Find continuous regions
        # Pad with False to handle edge cases
        padded = np.concatenate(([False], stance_mask, [False]))
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        for start, end in zip(starts, ends):
            start_time = self.time[start]
            end_time = self.time[end-1]
            ax.axvspan(start_time, end_time, alpha=alpha, color=color, lw=0)

    def _find_stereo_dofs(self, joint_substring):
        """
        Identify Left/Right pairs for a given joint substring (e.g. 'Hip').
        Returns: List of tuples (clean_name, l_idx, r_idx)
        """
        pairs = []
        # Find all DOFs matching substring
        relevant_indices = [i for i, n in enumerate(self.dof_names) if joint_substring in n]
        
        # separate into L and R buckets
        lefts = {}
        rights = {}
        
        for idx in relevant_indices:
            name = self.dof_names[idx]
            if 'L_' in name:
                # Strip L_ and keep the rest as key
                key = name.replace('L_', '')
                lefts[key] = idx
            elif 'R_' in name:
                key = name.replace('R_', '')
                rights[key] = idx
                
        # Match them
        all_keys = sorted(list(set(lefts.keys()) | set(rights.keys())))
        
        for key in all_keys:
            l_idx = lefts.get(key)
            r_idx = rights.get(key)
            
            # Clean name for display
            display_name = key.replace('_', ' ')
            
            pairs.append((display_name, l_idx, r_idx))
            
        return pairs

    def _create_report(self, filename, title, pairs, data, unit_scale, ylabel):
        """
        Generates a multi-page PDF for the given variable group.
        
        pages:
          1. Time Series (Overlay L/R)
          2. Stance Normalized (Overlay L/R)
          3. Stride Normalized (Overlay L/R)
        """
        filepath = self.output_dir / f"{filename}.pdf"
        
        with PdfPages(filepath) as pdf:
            # 1. Time Series
            self._add_report_page(pdf, "Time Series", pairs, data, unit_scale, ylabel, mode='time')
            
            # 2. Stance
            self._add_report_page(pdf, "Stance Phase (0-100%)", pairs, data, unit_scale, ylabel, mode='stance')
            
            # 3. Stride
            self._add_report_page(pdf, "Stride Cycle (0-100%)", pairs, data, unit_scale, ylabel, mode='stride')
            
        print(f"Saved {filepath.name}")

    def _add_report_page(self, pdf, page_title, pairs, data, unit_scale, ylabel, mode):
        # Create a figure with N rows (one per DOF pair)
        num_plots = len(pairs)
        if num_plots == 0: return

        fig, axes = plt.subplots(num_plots, 1, figsize=(8.5, 3 * num_plots), sharex=True)
        if num_plots == 1: axes = [axes]
        
        fig.suptitle(f"{page_title}", fontsize=14)
        
        for ax, (name, l_idx, r_idx) in zip(axes, pairs):
            ax.set_title(name, fontsize=10, loc='left')
            ax.set_ylabel(ylabel)
            
            self._plot_pair_on_ax(ax, l_idx, r_idx, data, unit_scale, mode)
            
        axes[-1].set_xlabel("Time (s)" if mode == 'time' else "% Cycle")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def _plot_pair_on_ax(self, ax, l_idx, r_idx, data, unit_scale, mode):
        """Helper to plot L/R pair on a single axis."""
        
        colors = {'Left': '#1f77b4', 'Right': '#d62728'}
        
        def plot_single_side(idx, side):
            if idx is None: return
            
            if mode == 'time':
                # Time Series
                series = data[:, :, idx] * unit_scale
                
                # Gait Events (only for time series)
                if side: self._plot_gait_events(ax, side=side, color=colors[side], alpha=0.05)
                
                if self.is_multi_env:
                    mean = series.mean(axis=1)
                    std = series.std(axis=1)
                    ax.plot(self.time, mean, label=f'{side} Mean', color=colors[side])
                    ax.fill_between(self.time, mean - std, mean + std, alpha=0.1, color=colors[side])
                else:
                    ax.plot(self.time, series[:, 0], label=f'{side}', color=colors[side])

            else:
                # Stance or Stride Normalization
                if side not in self.gait_cycles: return
                cycle_key = mode # 'stance' or 'stride'
                if cycle_key not in self.gait_cycles[side]: return
                
                cycles = self.gait_cycles[side][cycle_key]
                all_normalized = []
                
                for start, end in cycles:
                    if end > data.shape[0]: continue
                    segment = data[start:end, :, idx] * unit_scale
                    norm_segment = self.time_normalize(segment) # returns (101, Envs)
                    all_normalized.append(norm_segment)
                
                if all_normalized:
                    # Concatenate along Env dimension: (101, Envs * NumCycles)
                    pooled = np.concatenate(all_normalized, axis=1)
                    mean = pooled.mean(axis=1)
                    std = pooled.std(axis=1)
                    x = np.linspace(0, 100, 101)
                    
                    ax.plot(x, mean, label=f'{side}', color=colors[side])
                    ax.fill_between(x, mean - std, mean + std, alpha=0.1, color=colors[side])

        if l_idx is not None: plot_single_side(l_idx, 'Left')
        if r_idx is not None: plot_single_side(r_idx, 'Right')
        
        ax.legend(loc='upper right', fontsize=8)
        if mode != 'time':
            ax.set_xlim(0, 100)

    def plot_gait_segmentation(self):
        """Plot a Gantt-style chart of gait segmentation (Stance vs Swing)."""
        if not self.gait_events:
            print("No gait events detected.")
            return

        fig, ax = plt.subplots(figsize=(10, 3))
        
        # Define y-positions and heights
        y_pos = {'Left': 1, 'Right': 0}
        colors = {'Left': '#1f77b4', 'Right': '#d62728'} # Match standard colors
        height = 0.8
        
        for side in ['Right', 'Left']: # Order matters for y-axis
            if side not in self.gait_events:
                continue
                
            # Get stance mask for first trial
            stance_mask = self.gait_events[side][:, 0]
            
            # Find continuous regions
            padded = np.concatenate(([False], stance_mask, [False]))
            diff = np.diff(padded.astype(int))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            
            # Create list of (start, duration) tuples for broken_barh
            xranges = []
            for start, end in zip(starts, ends):
                if start >= len(self.time): continue
                start_time = self.time[start]
                if end < len(self.time):
                    end_time = self.time[end]
                else:
                    end_time = self.time[-1] + self.dt
                duration = end_time - start_time
                xranges.append((start_time, duration))
            
            if xranges:
                ax.broken_barh(xranges, (y_pos[side], height), facecolors=colors[side], alpha=0.6, label=f'{side} Stance')

        ax.set_yticks([0.4, 1.4])
        ax.set_yticklabels(['Right', 'Left'])
        ax.set_xlabel("Time (s)")
        ax.set_title("Gait Segmentation (Stance Phase)")
        ax.grid(True, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "gait_segmentation.pdf")
        print("Saved gait_segmentation.pdf")
        plt.close()

    def plot_joint_kinematics(self):
        """Plot joint angles (degrees)."""
        joint_types = ['Hip', 'Knee', 'Ankle']
        for j_type in joint_types:
            pairs = self._find_stereo_dofs(j_type)
            if not pairs: continue
            
            self._create_report(
                filename=f"kinematics_{j_type.lower()}",
                title=f"{j_type} Kinematics",
                pairs=pairs,
                data=self.dof_pos,
                unit_scale=180/np.pi,
                ylabel="Angle (deg)"
            )

    def plot_joint_kinetics(self):
        """Plot joint moments/torques (Nm)."""
        joint_types = ['Hip', 'Knee', 'Ankle']
        for j_type in joint_types:
            pairs = self._find_stereo_dofs(j_type)
            if not pairs: continue

            self._create_report(
                filename=f"kinetics_{j_type.lower()}",
                title=f"{j_type} Kinetics",
                pairs=pairs,
                data=self.dof_forces,
                unit_scale=1.0,
                ylabel="Torque (Nm)"
            )

    def plot_grf(self):
        """Plot Ground Reaction Forces."""
        feet_keywords = ["Ankle", "Toe", "Foot"]
        
        # 1. Aggregate Forces
        left_indices = [i for i, n in enumerate(self.body_names) if "L_" in n and any(k in n for k in feet_keywords)]
        right_indices = [i for i, n in enumerate(self.body_names) if "R_" in n and any(k in n for k in feet_keywords)]
        
        l_grf = self.grf[:, :, left_indices, :].sum(axis=2) if left_indices else np.zeros((self.grf.shape[0], self.grf.shape[1], 3))
        r_grf = self.grf[:, :, right_indices, :].sum(axis=2) if right_indices else np.zeros((self.grf.shape[0], self.grf.shape[1], 3))
        
        # Combined GRF Array: (Time, Envs, 6) -> 0-2 Left, 3-5 Right
        # Actually, let's keep them separate arrays and pass them cleverly?
        # My _create_report expects a single 'data' array and indices.
        # So I should concatenate them.
        
        # Shape: (Time, Envs, 6)
        # 0: L_X, 1: L_Y, 2: L_Z
        # 3: R_X, 4: R_Y, 5: R_Z
        
        combined_grf = np.concatenate([l_grf, r_grf], axis=2)
        
        # Pairs: (Name, L_idx, R_idx)
        pairs = [
            ("Anterior-Posterior (X)", 0, 3),
            ("Medial-Lateral (Y)", 1, 4),
            ("Vertical (Z)", 2, 5)
        ]
        
        self._create_report(
            filename="grf",
            title="Ground Reaction Forces",
            pairs=pairs,
            data=combined_grf,
            unit_scale=1.0,
            ylabel="Force (N)"
        )

    def plot_imu(self):
        """Plot IMU data."""
        if self.imu_acc is None or self.imu_gyro is None:
            return

        # Treat IMU as "Left" (or just Mono) for simplicity, or handle as single side
        # My _plot_pair_on_ax handles None for one side.
        
        # Acc
        pairs_acc = [
            ("Acc X", 0, None),
            ("Acc Y", 1, None),
            ("Acc Z", 2, None)
        ]
        self._create_report("imu_acc", "IMU Acceleration", pairs_acc, self.imu_acc, 1.0, "m/s^2")

        # Gyro
        pairs_gyro = [
            ("Gyro X", 0, None),
            ("Gyro Y", 1, None),
            ("Gyro Z", 2, None)
        ]
        self._create_report("imu_gyro", "IMU Gyroscope", pairs_gyro, self.imu_gyro, 1.0, "rad/s")

    def run_all(self):
        print(f"Analyzing biomechanics data ({self.fps} Hz)...")
        self.detect_gait_events()
        self.plot_gait_segmentation()
        
        # New Comprehensive Plots
        self.plot_joint_kinematics()
        self.plot_joint_kinetics()
        self.plot_grf()
        self.plot_imu()
            
        print(f"All plots saved to {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate biomechanics plots from inference data.")
    parser.add_argument("data_file", type=str, help="Path to biomechanics_data.npz")
    parser.add_argument("--output", type=str, default=None, help="Output directory for plots")
    
    args = parser.parse_args()
    
    analyzer = BiomechanicsAnalysisSuite(args.data_file, args.output)
    analyzer.run_all()

if __name__ == "__main__":
    main()
