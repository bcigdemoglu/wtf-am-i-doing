"""Main GUI application for WTF Am I Doing."""

import json
import logging
import os
import queue
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from PIL import Image, ImageTk

from . import capture, config, diary, inference

# Setup logging
config.ensure_app_dir()
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class WTFApp:
    """Main application class."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("WTF Am I Doing?")
        self.root.geometry("700x600")
        self.root.minsize(600, 500)

        # State
        self.is_running = False
        self.worker_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.command_queue: queue.Queue[dict[str, Any]] = queue.Queue()

        # Current values
        self.diary_path = tk.StringVar(value=str(config.DEFAULT_DIARY_PATH))
        self.error_path = config.DEFAULT_ERROR_PATH
        self.selected_model = tk.StringVar(value="FastVLM-0.5B")
        self.interval = tk.IntVar(value=config.DEFAULT_INTERVAL)
        self.resolution = tk.StringVar(value="High")
        self.prompt = tk.StringVar(value=config.DEFAULT_PROMPT)
        self.use_apple_silicon = tk.BooleanVar(value=False)

        # Stats
        self.entry_count = tk.IntVar(value=0)
        self.error_count = tk.IntVar(value=0)
        self.status_text = tk.StringVar(value="Idle")

        # Current/Last output
        self.current_output = tk.StringVar(value="")
        self.last_output = tk.StringVar(value="")
        self.last_timestamp = tk.StringVar(value="--")

        # Thumbnail
        self.thumbnail_image: ImageTk.PhotoImage | None = None
        self.current_screenshot_path: Path | None = None
        self.previous_screenshot_path: Path | None = None  # Keep until replaced

        # Build UI
        self._create_widgets()

        # Ensure app directory exists
        config.ensure_app_dir()

        # Clean up any leftover temp files from previous sessions
        capture.cleanup_temp_dir()

        # Load cached settings
        self._load_settings()

        # Update stats from existing diary
        self._update_stats()

        # Save settings on close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _load_settings(self) -> None:
        """Load settings from cache file."""
        if not config.SETTINGS_CACHE_PATH.exists():
            return

        try:
            with open(config.SETTINGS_CACHE_PATH, "r") as f:
                settings = json.load(f)

            if "diary_path" in settings:
                self.diary_path.set(settings["diary_path"])
            if "model" in settings:
                self.selected_model.set(settings["model"])
            if "interval" in settings:
                self.interval.set(settings["interval"])
                self._on_interval_change(str(settings["interval"]))
            if "resolution" in settings:
                self.resolution.set(settings["resolution"])
            if "prompt" in settings:
                self.prompt.set(settings["prompt"])
            if "use_apple_silicon" in settings:
                self.use_apple_silicon.set(settings["use_apple_silicon"])

            logger.info("Loaded settings from cache")
        except Exception as e:
            logger.warning(f"Failed to load settings: {e}")

    def _save_settings(self) -> None:
        """Save current settings to cache file."""
        settings = {
            "diary_path": self.diary_path.get(),
            "model": self.selected_model.get(),
            "interval": self.interval.get(),
            "resolution": self.resolution.get(),
            "prompt": self.prompt.get(),
            "use_apple_silicon": self.use_apple_silicon.get(),
        }

        try:
            config.ensure_app_dir()
            with open(config.SETTINGS_CACHE_PATH, "w") as f:
                json.dump(settings, f, indent=2)
            logger.debug("Saved settings to cache")
        except Exception as e:
            logger.warning(f"Failed to save settings: {e}")

    def _on_close(self) -> None:
        """Handle window close - save settings and cleanup."""
        self._save_settings()
        if self.is_running:
            self._stop_recording()
        # Cleanup any remaining screenshots
        for path in [self.current_screenshot_path, self.previous_screenshot_path]:
            if path and path.exists():
                try:
                    path.unlink()
                except Exception:
                    pass
        self.root.destroy()

    def _create_widgets(self) -> None:
        """Create all UI widgets."""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # === Settings Section ===
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="5")
        settings_frame.pack(fill=tk.X, pady=(0, 10))

        # Diary path row
        diary_frame = ttk.Frame(settings_frame)
        diary_frame.pack(fill=tk.X, pady=2)

        ttk.Label(diary_frame, text="Diary:").pack(side=tk.LEFT)
        ttk.Entry(diary_frame, textvariable=self.diary_path, width=40).pack(
            side=tk.LEFT, padx=5, expand=True, fill=tk.X
        )
        ttk.Button(diary_frame, text="Browse...", command=self._browse_diary).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(
            diary_frame, text="Desktop Symlink", command=self._create_symlink
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(diary_frame, text="Open", command=self._open_diary).pack(
            side=tk.LEFT, padx=2
        )

        # Model and download row
        model_frame = ttk.Frame(settings_frame)
        model_frame.pack(fill=tk.X, pady=2)

        ttk.Label(model_frame, text="Model:").pack(side=tk.LEFT)
        model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.selected_model,
            values=list(config.ALL_MODELS.keys()),
            state="readonly",
            width=15,
        )
        model_combo.pack(side=tk.LEFT, padx=5)
        ttk.Button(
            model_frame, text="Download FastVLM", command=self._download_models
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            model_frame, text="Setup Apple Silicon", command=self._setup_apple_silicon
        ).pack(side=tk.LEFT, padx=5)

        # Apple Silicon checkbox with availability indicator
        mlx_available = inference.is_mlx_available()
        mlx_indicator = "\u2713" if mlx_available else "\u2717"  # ✓ or ✗
        mlx_text = f"Use Apple Silicon [{mlx_indicator}]"
        self.apple_silicon_checkbox = ttk.Checkbutton(
            model_frame,
            text=mlx_text,
            variable=self.use_apple_silicon,
        )
        self.apple_silicon_checkbox.pack(side=tk.LEFT, padx=10)

        # Interval row
        interval_frame = ttk.Frame(settings_frame)
        interval_frame.pack(fill=tk.X, pady=2)

        ttk.Label(interval_frame, text="Interval:").pack(side=tk.LEFT)
        self.interval_scale = ttk.Scale(
            interval_frame,
            from_=config.MIN_INTERVAL,
            to=config.MAX_INTERVAL,
            variable=self.interval,
            orient=tk.HORIZONTAL,
            length=200,
            command=self._on_interval_change,
        )
        self.interval_scale.pack(side=tk.LEFT, padx=5)
        self.interval_label = ttk.Label(interval_frame, text=f"{self.interval.get()}s")
        self.interval_label.pack(side=tk.LEFT)

        # Resolution row
        res_frame = ttk.Frame(settings_frame)
        res_frame.pack(fill=tk.X, pady=2)

        ttk.Label(res_frame, text="Resolution:").pack(side=tk.LEFT)
        res_combo = ttk.Combobox(
            res_frame,
            textvariable=self.resolution,
            values=list(config.RESOLUTIONS.keys()),
            state="readonly",
            width=10,
        )
        res_combo.pack(side=tk.LEFT, padx=5)

        # Prompt row
        prompt_frame = ttk.Frame(settings_frame)
        prompt_frame.pack(fill=tk.X, pady=2)

        ttk.Label(prompt_frame, text="Prompt:").pack(side=tk.LEFT)
        ttk.Entry(prompt_frame, textvariable=self.prompt, width=60).pack(
            side=tk.LEFT, padx=5, expand=True, fill=tk.X
        )

        # === Control Section ===
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)

        self.play_button = ttk.Button(
            control_frame, text="Play", command=self._start_recording, width=15
        )
        self.play_button.pack(side=tk.LEFT, padx=5, expand=True)

        self.stop_button = ttk.Button(
            control_frame,
            text="Stop",
            command=self._stop_recording,
            width=15,
            state=tk.DISABLED,
        )
        self.stop_button.pack(side=tk.LEFT, padx=5, expand=True)

        # === Status Section ===
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="5")
        status_frame.pack(fill=tk.BOTH, expand=True)

        # Status line
        status_line = ttk.Frame(status_frame)
        status_line.pack(fill=tk.X, pady=2)
        ttk.Label(status_line, text="Status:").pack(side=tk.LEFT)
        ttk.Label(status_line, textvariable=self.status_text).pack(side=tk.LEFT, padx=5)
        ttk.Button(status_line, text="Open Log", command=self._open_log).pack(side=tk.RIGHT)
        ttk.Button(status_line, text="Open Errors", command=self._open_errors).pack(side=tk.RIGHT, padx=(0, 5))
        ttk.Button(status_line, text="Open Diary", command=self._open_diary).pack(side=tk.RIGHT, padx=(0, 5))

        # Output area with thumbnail
        output_frame = ttk.Frame(status_frame)
        output_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Thumbnail on the left (clickable to open screenshot)
        thumb_frame = ttk.LabelFrame(output_frame, text="Preview (click to open)", padding="5")
        thumb_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        self.thumbnail_label = ttk.Label(thumb_frame, text="No preview", cursor="hand2")
        self.thumbnail_label.pack()
        self.thumbnail_label.bind("<Button-1>", self._open_current_screenshot)

        # Text outputs on the right
        text_frame = ttk.Frame(output_frame)
        text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Current output
        current_frame = ttk.LabelFrame(text_frame, text="Current", padding="5")
        current_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        self.current_text = tk.Text(
            current_frame, height=4, wrap=tk.WORD, state=tk.DISABLED
        )
        self.current_text.pack(fill=tk.BOTH, expand=True)

        # Last entry
        last_frame = ttk.LabelFrame(text_frame, text="Last Entry", padding="5")
        last_frame.pack(fill=tk.BOTH, expand=True)

        last_header = ttk.Frame(last_frame)
        last_header.pack(fill=tk.X)
        ttk.Label(last_header, textvariable=self.last_timestamp).pack(side=tk.RIGHT)

        self.last_text = tk.Text(last_frame, height=4, wrap=tk.WORD, state=tk.DISABLED)
        self.last_text.pack(fill=tk.BOTH, expand=True)

        # Stats bar at bottom
        stats_frame = ttk.Frame(status_frame)
        stats_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(stats_frame, text="Entries:").pack(side=tk.LEFT)
        ttk.Label(stats_frame, textvariable=self.entry_count).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(stats_frame, text="Errors:").pack(side=tk.LEFT)
        ttk.Label(stats_frame, textvariable=self.error_count).pack(side=tk.LEFT, padx=(0, 10))

    def _on_interval_change(self, value: str) -> None:
        """Update interval label when slider changes."""
        val = int(float(value))
        if val >= 60:
            mins = val // 60
            secs = val % 60
            if secs:
                self.interval_label.config(text=f"{mins}m {secs}s")
            else:
                self.interval_label.config(text=f"{mins}m")
        else:
            self.interval_label.config(text=f"{val}s")

    def _browse_diary(self) -> None:
        """Open file dialog to select diary location."""
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            initialfile="diary.json",
            title="Select Diary Location",
        )
        if path:
            self.diary_path.set(path)
            self._update_stats()

    def _create_symlink(self) -> None:
        """Create a symlink on the desktop."""
        diary = Path(self.diary_path.get())
        symlink = config.DESKTOP_SYMLINK_PATH

        if symlink.exists() or symlink.is_symlink():
            if not messagebox.askyesno(
                "Symlink Exists",
                f"{symlink} already exists.\n\nDo you want to replace it?",
            ):
                return
            symlink.unlink()

        try:
            # Ensure diary exists
            if not diary.exists():
                diary.parent.mkdir(parents=True, exist_ok=True)
                diary.write_text('{"entries": []}')

            symlink.symlink_to(diary)
            messagebox.showinfo("Success", f"Created symlink at:\n{symlink}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create symlink:\n{e}")

    def _open_diary(self) -> None:
        """Open the diary file in the default application."""
        diary = Path(self.diary_path.get())
        if diary.exists():
            config.open_file_in_default_app(diary)
        else:
            messagebox.showwarning("Not Found", "Diary file does not exist yet.")

    def _open_log(self) -> None:
        """Open the log file in the default application."""
        if config.LOG_FILE.exists():
            config.open_file_in_default_app(config.LOG_FILE)
        else:
            messagebox.showwarning("Not Found", "Log file does not exist yet.")

    def _open_errors(self) -> None:
        """Open the error file in the default application."""
        if self.error_path.exists():
            config.open_file_in_default_app(self.error_path)
        else:
            messagebox.showwarning("Not Found", "Error file does not exist yet (no errors recorded).")

    def _download_models(self) -> None:
        """Start model download process."""
        if not messagebox.askyesno(
            "Download Models",
            "This will:\n"
            "1. Clone the FastVLM repository\n"
            "2. Install dependencies\n"
            "3. Download all model checkpoints\n\n"
            "This may take a while. Continue?",
        ):
            return

        # Run in thread to not block UI
        def download_thread():
            try:
                logger.info("Starting model download process...")

                self._update_status("Cloning FastVLM...")
                logger.info("Cloning FastVLM repository...")
                inference.clone_fastvlm(self._update_status)

                self._update_status("Installing dependencies...")
                logger.info("Installing FastVLM dependencies...")
                inference.install_fastvlm_deps(self._update_status)

                self._update_status("Downloading models...")
                logger.info("Downloading model checkpoints...")
                inference.download_models(self._update_status)

                self._update_status("Check Terminal for download progress")
                logger.info("Terminal opened for model download")
            except inference.SetupError as e:
                logger.error(f"Setup failed: {e}")
                self._update_status("Setup failed")
                self.root.after(0, lambda: messagebox.showerror("Setup Error", str(e)))
            except Exception as e:
                logger.exception(f"Unexpected error during setup: {e}")
                self._update_status("Setup failed")
                self.root.after(0, lambda: messagebox.showerror("Error", f"Unexpected error:\n{e}"))

        thread = threading.Thread(target=download_thread, daemon=True)
        thread.start()

    def _setup_apple_silicon(self) -> None:
        """Setup Apple Silicon (MLX) for faster inference."""
        if not inference.is_mlx_available():
            messagebox.showwarning(
                "Warning",
                "Apple Silicon acceleration is only available on Apple Silicon Macs (M1/M2/M3).\n\n"
                "Setup may not work correctly on this machine.",
            )

        if not messagebox.askyesno(
            "Setup Apple Silicon",
            "This will:\n"
            "1. Install mlx-vlm package\n"
            "2. Convert FastVLM models to MLX format\n\n"
            "Requires: FastVLM models already downloaded.\n\n"
            "Continue?",
        ):
            return

        # Run in thread to not block UI
        def setup_thread():
            try:
                logger.info("Starting Apple Silicon setup...")
                inference.setup_mlx(self._update_status)
                self._update_status("Check Terminal for setup progress")
                logger.info("Terminal opened for MLX setup")
            except inference.SetupError as e:
                logger.error(f"MLX setup failed: {e}")
                self._update_status("Setup failed")
                self.root.after(0, lambda: messagebox.showerror("Setup Error", str(e)))
            except Exception as e:
                logger.exception(f"Unexpected error during MLX setup: {e}")
                self._update_status("Setup failed")
                self.root.after(0, lambda: messagebox.showerror("Error", f"Unexpected error:\n{e}"))

        thread = threading.Thread(target=setup_thread, daemon=True)
        thread.start()

    def _update_status(self, text: str) -> None:
        """Update status text (thread-safe)."""
        self.root.after(0, lambda: self.status_text.set(text))

    def _update_stats(self) -> None:
        """Update entry and error counts from diary and error files."""
        path = Path(self.diary_path.get())
        self.entry_count.set(diary.get_entry_count(path))
        self.error_count.set(diary.get_error_count(self.error_path))

    def _set_current_text(self, text: str) -> None:
        """Set the current output text."""
        self.current_text.config(state=tk.NORMAL)
        self.current_text.delete("1.0", tk.END)
        self.current_text.insert("1.0", text)
        self.current_text.config(state=tk.DISABLED)

    def _set_last_text(self, text: str, timestamp: str) -> None:
        """Set the last entry text."""
        self.last_text.config(state=tk.NORMAL)
        self.last_text.delete("1.0", tk.END)
        self.last_text.insert("1.0", text)
        self.last_text.config(state=tk.DISABLED)
        self.last_timestamp.set(timestamp)

    def _update_thumbnail(self, image: Image.Image, screenshot_path: Path | None = None) -> None:
        """Update the thumbnail display."""
        # Clean up previous screenshot before replacing
        if self.previous_screenshot_path and self.previous_screenshot_path.exists():
            try:
                self.previous_screenshot_path.unlink()
                logger.debug(f"Cleaned up previous screenshot: {self.previous_screenshot_path.name}")
            except Exception as e:
                logger.warning(f"Failed to cleanup previous screenshot: {e}")

        self.thumbnail_image = ImageTk.PhotoImage(image)
        self.thumbnail_label.config(image=self.thumbnail_image, text="")
        self.previous_screenshot_path = self.current_screenshot_path
        self.current_screenshot_path = screenshot_path

    def _open_current_screenshot(self, event=None) -> None:
        """Open the current screenshot in the default image viewer."""
        if self.current_screenshot_path and self.current_screenshot_path.exists():
            config.open_file_in_default_app(self.current_screenshot_path)
        else:
            messagebox.showinfo("No Screenshot", "No screenshot available to open.")

    def _start_recording(self) -> None:
        """Start the recording loop."""
        # Check screen recording permission
        if not config.check_screen_recording_permission():
            result = messagebox.askyesno(
                "Permission Required",
                "Screen Recording permission is required.\n\n"
                "Please grant permission in:\n"
                "System Preferences > Privacy & Security > Screen Recording\n\n"
                "Open System Preferences now?",
            )
            if result:
                config.open_screen_recording_settings()
            return

        # Check if model is available
        if not inference.is_model_available(self.selected_model.get()):
            provider = config.get_model_provider(self.selected_model.get())
            if provider == config.PROVIDER_FASTVLM:
                messagebox.showerror(
                    "Model Not Found",
                    f"Model '{self.selected_model.get()}' is not downloaded.\n\n"
                    "Please click 'Download FastVLM' first.",
                )
            elif provider == config.PROVIDER_CLAUDE:
                messagebox.showerror(
                    "Claude CLI Not Found",
                    "Claude CLI is not installed.\n\n"
                    "Please install it with:\n"
                    "npm install -g @anthropic-ai/claude-code",
                )
            else:
                messagebox.showerror(
                    "Model Not Available",
                    f"Model '{self.selected_model.get()}' is not available.",
                )
            return

        self.is_running = True
        self.stop_event.clear()

        self.play_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        self.worker_thread = threading.Thread(target=self._recording_loop, daemon=True)
        self.worker_thread.start()

    def _stop_recording(self) -> None:
        """Stop the recording loop."""
        self.is_running = False
        self.stop_event.set()

        self.play_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

        self._update_status("Stopped")

    def _recording_loop(self) -> None:
        """Main recording loop (runs in worker thread)."""
        while not self.stop_event.is_set():
            cycle_start = time.time()

            # Get current settings
            model = self.selected_model.get()
            prompt = self.prompt.get()
            resolution = self.resolution.get()
            diary_path = Path(self.diary_path.get())
            use_mlx = self.use_apple_silicon.get()

            # Capture all monitors
            try:
                screenshots = capture.capture_monitors(resolution)
            except Exception as e:
                self._update_status(f"Capture error: {e}")
                self._wait_for_interval(cycle_start)
                continue

            is_single_monitor = len(screenshots) == 1

            # Process each monitor
            for idx, screenshot in enumerate(screenshots):
                if self.stop_event.is_set():
                    # Cleanup remaining screenshots
                    for s in screenshots[idx:]:
                        s.cleanup()
                    break

                monitor_label = f"monitor {screenshot.monitor.index}/{len(screenshots)}"
                self._update_status(f"Processing {monitor_label}...")

                # Update thumbnail (pass path so user can click to open)
                try:
                    thumb = capture.create_thumbnail(screenshot.image_path)
                    path = screenshot.image_path
                    self.root.after(0, lambda t=thumb, p=path: self._update_thumbnail(t, p))
                except Exception:
                    pass  # Thumbnail is optional

                # Set current text to processing
                self.root.after(0, lambda: self._set_current_text("Processing..."))

                # Run inference
                try:
                    inference_start = time.time()
                    result = inference.run_inference(
                        screenshot.image_path,
                        model,
                        prompt,
                        use_mlx=use_mlx,
                    )
                    inference_ms = int((time.time() - inference_start) * 1000)

                    # Save to diary
                    entry = diary.append_entry(
                        diary_path,
                        config.get_diary_model_name(model),
                        result,
                        screenshot.monitor,
                        is_single_monitor,
                        inference_time_ms=inference_ms,
                    )

                    # Update UI
                    timestamp = datetime.fromisoformat(
                        entry["timestamp"].replace("Z", "+00:00")
                    ).strftime("%H:%M:%S")
                    time_info = f"{timestamp} ({inference_ms}ms)"

                    self.root.after(0, lambda r=result: self._set_current_text(r))
                    self.root.after(
                        0, lambda r=result, t=time_info: self._set_last_text(r, t)
                    )

                except inference.InferenceError as e:
                    # Log error to error file
                    diary.append_error(
                        self.error_path,
                        config.get_diary_model_name(model),
                        str(e),
                        screenshot.monitor,
                        is_single_monitor,
                    )
                    self.root.after(0, lambda e=e: self._set_current_text(f"Error: {e}"))

                # Note: Don't cleanup here - keep screenshot for user to view
                # Cleanup happens in _update_thumbnail when next screenshot arrives

                # Update stats
                self.root.after(0, self._update_stats)

            # Wait for remaining interval time
            self._wait_for_interval(cycle_start)

        self._update_status("Stopped")

    def _wait_for_interval(self, cycle_start: float) -> None:
        """Wait for the remaining interval time."""
        elapsed = time.time() - cycle_start
        remaining = self.interval.get() - elapsed

        if remaining > 0 and not self.stop_event.is_set():
            self._update_status(f"Waiting {int(remaining)}s...")
            # Check stop event periodically during wait
            wait_end = time.time() + remaining
            while time.time() < wait_end and not self.stop_event.is_set():
                time.sleep(0.5)


def main() -> None:
    """Main entry point."""
    root = tk.Tk()
    app = WTFApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
