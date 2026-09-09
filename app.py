"""Графический интерфейс перцептрона прогноза вечернего настроения."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from perceptron import (
    HIDDEN_NAMES,
    TOKEN_LABELS,
    ForwardResult,
    Perceptron,
    save_snapshot_csv,
    snapshot_record,
)


BG = "#0e141b"
PANEL = "#151d27"
PANEL_2 = "#1c2733"
STROKE = "#2a3a4d"
TEXT = "#e8eef4"
MUTED = "#8b9bb0"
ACCENT = "#5eead4"
ACCENT_2 = "#38bdf8"
POSITIVE = "#34d399"
NEGATIVE = "#f87171"
WARNING = "#fbbf24"
WHITE = "#ffffff"

SNAPSHOTS_DIR = Path(__file__).resolve().parent / "snapshots"

PRESETS = {
    "Типичный день": (7.0, 7.0, 6.0, 6.0, 0.0),
    "Хороший день": (8.5, 8.0, 8.0, 8.0, 0.0),
    "Недосып": (4.0, 5.0, 5.0, 4.0, 1.0),
    "Тревожное утро": (7.0, 7.5, 6.0, 4.0, 1.0),
}


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Перцептрон: прогноз вечернего настроения")
        self.geometry("1420x860")
        self.minsize(1240, 780)
        self.configure(bg=BG)
        self.perceptron = Perceptron()
        self._updating = False
        self._fact_ready = False
        self._last_fact: float | None = None

        self.sleep_today = tk.DoubleVar(value=7.0)
        self.sleep_yesterday = tk.DoubleVar(value=7.0)
        self.mood_yesterday = tk.DoubleVar(value=6.0)
        self.mood_morning = tk.DoubleVar(value=6.0)
        self.anxiety = tk.IntVar(value=0)
        self.target_mood = tk.DoubleVar(value=6.0)
        self.lr = tk.DoubleVar(value=0.35)
        self.epochs = tk.IntVar(value=25)
        self.status = tk.StringVar(value="Передвиньте ползунки — прогноз обновится сразу.")

        self._build_style()
        self._build_layout()
        self._bind_traces()
        self.after(80, self._finish_init)

    def _build_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=PANEL)
        style.configure("Inner.TFrame", background=PANEL_2)
        style.configure("Title.TLabel", background=BG, foreground=WHITE, font=("Segoe UI Semibold", 20))
        style.configure("Subtitle.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 11))
        style.configure("H.TLabel", background=PANEL, foreground=WHITE, font=("Segoe UI Semibold", 13))
        style.configure("P.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("M.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Val.TLabel", background=PANEL, foreground=ACCENT, font=("Consolas", 11, "bold"))
        style.configure("Inner.TLabel", background=PANEL_2, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("InnerM.TLabel", background=PANEL_2, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("TCheckbutton", background=PANEL, foreground=TEXT, font=("Segoe UI", 10))
        style.map("TCheckbutton", background=[("active", PANEL)])
        style.configure(
            "Accent.Horizontal.TScale",
            background=PANEL,
            troughcolor=STROKE,
        )
        style.configure(
            "TButton",
            background=PANEL_2,
            foreground=TEXT,
            padding=(12, 7),
            font=("Segoe UI", 10),
        )
        style.map(
            "TButton",
            background=[("active", "#243140")],
            foreground=[("active", WHITE)],
        )
        style.configure(
            "Accent.TButton",
            background=ACCENT,
            foreground=BG,
            padding=(16, 8),
            font=("Segoe UI Semibold", 11),
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#99f6e4")],
            foreground=[("active", BG)],
        )

    def _build_layout(self) -> None:
        header = ttk.Frame(self, style="TFrame")
        header.pack(fill="x", padx=22, pady=(16, 8))
        actions = ttk.Frame(header, style="TFrame")
        actions.pack(side="right", padx=(12, 0))
        ttk.Button(actions, text="Обучить", style="Accent.TButton", command=self.train_once).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Сбросить веса", command=self.reset_weights).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Сохранить", command=self.save_snapshot).pack(side="left")
        ttk.Label(header, text="Перцептрон прогноза вечернего настроения", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Входы → голова внимания → скрытый слой → сигмоида → балл 1–10",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=18, pady=(4, 8))

        self.left = self._card(body, side="left", fill="y", expand=False, padx=(0, 8), width=330)
        self.right = self._card(body, side="right", fill="y", expand=False, padx=(8, 0), width=370)
        self.center = self._card(body, side="left", fill="both", expand=True, padx=8)

        self._build_inputs(self.left)
        self._build_network(self.center)
        self._build_output(self.right)

        footer = ttk.Frame(self, style="TFrame")
        footer.pack(fill="x", padx=22, pady=(0, 14))
        ttk.Label(footer, textvariable=self.status, style="Subtitle.TLabel").pack(anchor="w")

    def _card(self, parent: tk.Frame, **pack_kwargs) -> tk.Frame:
        width = pack_kwargs.pop("width", None)
        outer = tk.Frame(parent, bg=PANEL, highlightbackground=STROKE, highlightthickness=1)
        if width is not None:
            outer.configure(width=width)
            outer.pack_propagate(False)
        outer.pack(**pack_kwargs)
        inner = tk.Frame(outer, bg=PANEL)
        inner.pack(fill="both", expand=True, padx=16, pady=16)
        return inner

    def _build_inputs(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Входы", style="H.TLabel").pack(anchor="w")
        ttk.Label(parent, text="Признаки, которые видит перцептрон", style="M.TLabel").pack(anchor="w", pady=(0, 10))

        self.value_labels: dict[str, ttk.Label] = {}
        self._slider(parent, "Сон сегодня", self.sleep_today, 0, 14, 0.5, "ч", "sleep_today")
        self._slider(parent, "Сон вчера", self.sleep_yesterday, 0, 14, 0.5, "ч", "sleep_yesterday")
        self._slider(parent, "Настроение вчера вечером", self.mood_yesterday, 1, 10, 1, "балл", "mood_yesterday")
        self._slider(parent, "Настроение сегодня утром", self.mood_morning, 1, 10, 1, "балл", "mood_morning")

        anxiety_row = ttk.Frame(parent, style="Card.TFrame")
        anxiety_row.pack(fill="x", pady=(8, 4))
        ttk.Label(anxiety_row, text="Тревога с утра", style="P.TLabel").pack(side="left")
        ttk.Checkbutton(
            anxiety_row,
            text="да = 1 / нет = 0",
            variable=self.anxiety,
            command=self.refresh,
        ).pack(side="right")

        ttk.Label(parent, text="Готовые сценарии", style="H.TLabel").pack(anchor="w", pady=(18, 6))
        presets = ttk.Frame(parent, style="Card.TFrame")
        presets.pack(fill="x")
        for i, name in enumerate(PRESETS):
            ttk.Button(presets, text=name, command=lambda n=name: self.apply_preset(n)).grid(
                row=i // 2, column=i % 2, sticky="ew", padx=3, pady=3
            )
            presets.columnconfigure(i % 2, weight=1)

    def _slider(
        self,
        parent: tk.Misc,
        title: str,
        var: tk.DoubleVar,
        from_: float,
        to: float,
        resolution: float,
        unit: str,
        key: str,
    ) -> None:
        block = tk.Frame(parent, bg=PANEL)
        block.pack(fill="x", pady=(0, 10))
        head = tk.Frame(block, bg=PANEL)
        head.pack(fill="x")
        tk.Label(head, text=title, bg=PANEL, fg=TEXT, font=("Segoe UI", 10)).pack(side="left")
        val = tk.Label(head, text="", bg=PANEL, fg=ACCENT, font=("Consolas", 11, "bold"))
        val.pack(side="right")
        self.value_labels[key] = val
        scale = tk.Scale(
            block,
            from_=from_,
            to=to,
            resolution=resolution,
            variable=var,
            orient="horizontal",
            showvalue=False,
            bg=PANEL,
            fg=TEXT,
            troughcolor=STROKE,
            highlightthickness=0,
            sliderrelief="flat",
            activebackground=ACCENT,
            command=lambda _: self.refresh(),
        )
        scale.pack(fill="x", pady=(4, 0))
        tk.Label(
            block,
            text=f"{from_:g} … {to:g} {unit}, шаг {resolution:g}",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 9),
        ).pack(anchor="w")

    def _build_network(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Схема сети", style="H.TLabel").pack(anchor="w")
        ttk.Label(
            parent,
            text="Клетка теплокарты: насколько признак-строка смотрит на признак-столбец",
            style="M.TLabel",
        ).pack(anchor="w", pady=(0, 8))
        self.canvas = tk.Canvas(parent, bg=PANEL_2, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _e: self._draw_network(self._last_result))
        self._last_result: ForwardResult | None = None

        formula = ttk.Frame(parent, style="Inner.TFrame")
        formula.pack(fill="x", pady=(10, 0))
        self.formula_var = tk.StringVar()
        ttk.Label(formula, textvariable=self.formula_var, style="Inner.TLabel", justify="left").pack(
            anchor="w", padx=10, pady=8
        )

    def _build_output(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Прогноз", style="H.TLabel").pack(anchor="w")
        ttk.Label(parent, text="Настроение вечером, 1–10", style="M.TLabel").pack(anchor="w")

        self.score_canvas = tk.Canvas(parent, height=88, bg=PANEL, highlightthickness=0, bd=0)
        self.score_canvas.pack(fill="x", pady=(6, 8))

        ttk.Label(parent, text="Обучение на факте", style="H.TLabel").pack(anchor="w", pady=(4, 2))
        ttk.Label(
            parent,
            text="Укажите фактическое настроение вечером и нажмите «Обучить»",
            style="M.TLabel",
        ).pack(anchor="w")
        train_head = ttk.Frame(parent, style="Card.TFrame")
        train_head.pack(fill="x", pady=(6, 0))
        ttk.Label(train_head, text="Факт вечером", style="P.TLabel").pack(side="left")
        self.target_label = ttk.Label(train_head, text="6", style="Val.TLabel")
        self.target_label.pack(side="right")
        tk.Scale(
            parent,
            from_=1,
            to=10,
            resolution=1,
            variable=self.target_mood,
            orient="horizontal",
            showvalue=False,
            bg=PANEL,
            fg=TEXT,
            troughcolor=STROKE,
            highlightthickness=0,
            sliderrelief="flat",
            activebackground=ACCENT,
            command=self._on_fact_change,
        ).pack(fill="x", pady=(4, 8))

        tk.Button(
            parent,
            text="Обучить",
            command=self.train_once,
            bg=ACCENT,
            fg=BG,
            activebackground="#99f6e4",
            activeforeground=BG,
            font=("Segoe UI Semibold", 13),
            relief="flat",
            bd=0,
            padx=16,
            pady=10,
            cursor="hand2",
        ).pack(fill="x")
        btns = ttk.Frame(parent, style="Card.TFrame")
        btns.pack(fill="x", pady=(6, 0))
        ttk.Button(btns, text="Сбросить веса", command=self.reset_weights).pack(side="left", padx=(0, 6))
        ttk.Button(btns, text="Сохранить", command=self.save_snapshot).pack(side="left")
        ttk.Label(
            parent,
            text="Кнопка «Обучить» также в шапке окна · скорость 0.35 · эпохи 25",
            style="M.TLabel",
        ).pack(anchor="w", pady=(6, 8))

        metrics = ttk.Frame(parent, style="Inner.TFrame")
        metrics.pack(fill="x")
        self.h_var = tk.StringVar()
        self.attn_var = tk.StringVar()
        self.z_var = tk.StringVar()
        self.s_var = tk.StringVar()
        for label, var in (
            ("Голова внимания (кто на кого смотрит)", self.attn_var),
            ("Скрытый слой h1 тело · h2 эмоции · h3 тревога", self.h_var),
            ("Выход: сумма z", self.z_var),
            ("Сигмоида σ(z)", self.s_var),
        ):
            row = ttk.Frame(metrics, style="Inner.TFrame")
            row.pack(fill="x", padx=10, pady=3)
            ttk.Label(row, text=label, style="InnerM.TLabel").pack(anchor="w")
            ttk.Label(row, textvariable=var, style="Inner.TLabel").pack(anchor="w")

        ttk.Label(parent, text="Веса выходного слоя", style="H.TLabel").pack(anchor="w", pady=(12, 4))
        ttk.Label(parent, text="Как выход смешивает тело, эмоции и тревогу", style="M.TLabel").pack(anchor="w")
        self.weights_frame = ttk.Frame(parent, style="Card.TFrame")
        self.weights_frame.pack(fill="x")
        self.weight_vars = [tk.StringVar() for _ in HIDDEN_NAMES]
        self.bias_var = tk.StringVar()
        self.weight_entries: list[ttk.Entry] = []
        for i, name in enumerate(HIDDEN_NAMES):
            self._weight_row(self.weights_frame, f"w₂←{name}", self.weight_vars[i], i)
        self.bias_entry = self._weight_row(self.weights_frame, "b₂  смещение", self.bias_var, None)

    def _weight_row(self, parent: ttk.Frame, title: str, var: tk.StringVar, index: int | None) -> ttk.Entry:
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=title, style="M.TLabel").pack(side="left")
        entry = ttk.Entry(row, textvariable=var, width=8, justify="right")
        entry.pack(side="right")
        entry.bind("<Return>", lambda _e, i=index: self._commit_weight(i))
        entry.bind("<FocusOut>", lambda _e, i=index: self._commit_weight(i))
        if index is not None:
            self.weight_entries.append(entry)
        return entry

    def _bind_traces(self) -> None:
        for var in (
            self.sleep_today,
            self.sleep_yesterday,
            self.mood_yesterday,
            self.mood_morning,
            self.anxiety,
        ):
            var.trace_add("write", lambda *_: self.refresh())

    def current_inputs(self) -> tuple[float, float, float, float, float]:
        return (
            self._snap(self.sleep_today, 0.5),
            self._snap(self.sleep_yesterday, 0.5),
            self._snap(self.mood_yesterday, 1.0),
            self._snap(self.mood_morning, 1.0),
            float(self.anxiety.get()),
        )

    @staticmethod
    def _snap(var: tk.DoubleVar, step: float) -> float:
        value = float(var.get())
        snapped = round(value / step) * step
        return snapped

    def apply_preset(self, name: str) -> None:
        sleep_t, sleep_y, mood_y, mood_m, anxiety = PRESETS[name]
        self._updating = True
        self.sleep_today.set(sleep_t)
        self.sleep_yesterday.set(sleep_y)
        self.mood_yesterday.set(mood_y)
        self.mood_morning.set(mood_m)
        self.anxiety.set(int(anxiety))
        self._updating = False
        self.status.set(f"Загружен сценарий «{name}».")
        self.refresh()

    def _finish_init(self) -> None:
        self.refresh()
        self._last_fact = self._snap(self.target_mood, 1.0)
        self._fact_ready = True

    def _on_fact_change(self, _value: str | None = None) -> None:
        target = self._snap(self.target_mood, 1.0)
        self.target_label.configure(text=f"{target:.0f}")

    def _commit_weight(self, index: int | None) -> None:
        try:
            if index is None:
                self.perceptron.b2 = float(self.bias_var.get().replace(",", "."))
            else:
                self.perceptron.w2[index] = float(self.weight_vars[index].get().replace(",", "."))
            self.status.set("Веса выходного слоя обновлены вручную.")
        except ValueError:
            self.status.set("Не удалось прочитать вес: введите число.")
        self.refresh()

    def save_snapshot(self) -> None:
        result = self._last_result or self.perceptron.forward(self.current_inputs())
        record = snapshot_record(self.perceptron, result)
        SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Сохранить срез данных",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Все файлы", "*.*")],
            initialdir=str(SNAPSHOTS_DIR),
            initialfile=f"srez_{stamp}.csv",
        )
        if not path:
            return
        existed = Path(path).exists() and Path(path).stat().st_size > 0
        try:
            saved = save_snapshot_csv(path, record)
        except OSError as exc:
            messagebox.showerror("Не удалось сохранить", str(exc), parent=self)
            self.status.set("Срез не сохранён.")
            return
        if existed:
            self.status.set(f"Строка среза добавлена: {saved}")
        else:
            self.status.set(f"Срез сохранён: {saved}")

    def reset_weights(self) -> None:
        self.perceptron.reset()
        self.status.set("Веса возвращены к стартовым значениям.")
        self.refresh()

    def train_once(self) -> None:
        raw = self.current_inputs()
        target = self._snap(self.target_mood, 1.0)
        before_w = list(self.perceptron.w2)
        before_b = self.perceptron.b2
        before = self.perceptron.forward(raw).mood
        self.perceptron.train(raw, target, lr=self.lr.get(), epochs=self.epochs.get())
        after = self.perceptron.forward(raw).mood
        delta = self._weight_delta_text(before_w, before_b, self.perceptron.w2, self.perceptron.b2)
        self.status.set(
            f"Обучение внимания и слоёв: факт {target:.0f}. Прогноз {before:.2f} → {after:.2f}. {delta}"
        )
        self.refresh()

    @staticmethod
    def _weight_delta_text(
        before_w: list[float],
        before_b: float,
        after_w: list[float],
        after_b: float,
    ) -> str:
        parts: list[str] = []
        for i, (old, new) in enumerate(zip(before_w, after_w)):
            if abs(new - old) >= 0.0005:
                parts.append(f"w₂←{HIDDEN_NAMES[i]} {old:+.3f}→{new:+.3f}")
        if abs(after_b - before_b) >= 0.0005:
            parts.append(f"b₂ {before_b:+.3f}→{after_b:+.3f}")
        extra = " · скрытый слой и внимание тоже сдвинулись" if parts else ""
        return ("; ".join(parts) + extra) if parts else "веса почти не сдвинулись"

    def refresh(self) -> None:
        if self._updating:
            return
        raw = self.current_inputs()
        self.value_labels["sleep_today"].configure(text=f"{raw[0]:.1f} ч")
        self.value_labels["sleep_yesterday"].configure(text=f"{raw[1]:.1f} ч")
        self.value_labels["mood_yesterday"].configure(text=f"{raw[2]:.0f}")
        self.value_labels["mood_morning"].configure(text=f"{raw[3]:.0f}")

        result = self.perceptron.forward(raw)
        self._last_result = result
        self._draw_score(result)
        self._draw_network(result)
        self.h_var.set("  ".join(f"h{i + 1}={v:.3f}" for i, v in enumerate(result.hidden)))
        self.attn_var.set(self._attn_summary(result.attention))
        self.z_var.set(f"{result.z:+.3f}")
        self.s_var.set(f"{result.sigma:.4f}  →  1 + 9·σ(z) = {result.mood:.2f}")
        focused = self.focus_get()
        for i, w in enumerate(self.perceptron.w2):
            if self.weight_entries[i] is not focused:
                self.weight_vars[i].set(f"{w:.3f}")
        if self.bias_entry is not focused:
            self.bias_var.set(f"{self.perceptron.b2:.3f}")
        h_terms = " + ".join(f"w₂{i + 1}·h{i + 1}" for i in range(3))
        self.formula_var.set(
            f"A = softmax(QKᵀ / √d)   x' = x + Attention   "
            + "   ".join(f"h{i + 1}={result.hidden[i]:.3f}" for i in range(3))
            + f"\nz = {h_terms} + b₂ = {result.z:+.3f}"
            + f"\nнастроение вечером = 1 + 9 · σ(z) = {result.mood:.2f}"
        )

    @staticmethod
    def _attn_summary(attention: tuple[tuple[float, ...], ...]) -> str:
        diag = sum(attention[i][i] for i in range(5)) / 5.0
        best_i, best_j, best_a = 0, 1, -1.0
        for i, row in enumerate(attention):
            for j, value in enumerate(row):
                if i != j and value > best_a:
                    best_i, best_j, best_a = i, j, value
        return (
            f"диагональ {diag:.2f}  ·  сильнее всего {TOKEN_LABELS[best_i]} → {TOKEN_LABELS[best_j]} ({best_a:.2f})"
        )

    def _mood_color(self, mood: float) -> str:
        if mood < 4:
            return NEGATIVE
        if mood < 6.5:
            return WARNING
        return POSITIVE

    def _draw_score(self, result: ForwardResult) -> None:
        c = self.score_canvas
        c.delete("all")
        w = c.winfo_width() or 300
        h = c.winfo_height() or 118
        color = self._mood_color(result.mood)
        c.create_rectangle(0, 0, w, h, fill=PANEL_2, outline="")
        c.create_text(18, 22, anchor="w", fill=MUTED, font=("Segoe UI", 10), text="Прогноз на вечер")
        c.create_text(18, 56, anchor="w", fill=color, font=("Segoe UI Semibold", 32), text=f"{result.mood:.1f}")
        c.create_text(w - 18, 56, anchor="e", fill=MUTED, font=("Segoe UI", 11), text="из 10")
        bar_x, bar_y, bar_w, bar_h = 18, h - 16, max(40, w - 36), 7
        c.create_rectangle(bar_x, bar_y, bar_x + bar_w, bar_y + bar_h, fill=STROKE, outline="")
        fill = bar_w * ((result.mood - 1.0) / 9.0)
        c.create_rectangle(bar_x, bar_y, bar_x + fill, bar_y + bar_h, fill=color, outline="")

    def _draw_network(self, result: ForwardResult | None) -> None:
        c = self.canvas
        c.delete("all")
        width = c.winfo_width() or 560
        height = c.winfo_height() or 420
        if width < 40 or height < 40:
            return
        c.create_rectangle(0, 0, width, height, fill=PANEL_2, outline="")

        left_x = 58
        hid_x = width * 0.62
        out_x = width * 0.81
        mood_x = width - 56
        mid_y = height / 2 + 10
        in_gap = min(64, max(48, (height - 100) / 4.4))
        hid_gap = min(88, max(66, (height - 90) / 2.4))
        in_ys = [mid_y + (i - 2) * in_gap for i in range(5)]
        hid_ys = [mid_y + (j - 1) * hid_gap for j in range(3)]

        labels = ["сон сегодня", "сон вчера", "настр. вчера", "настр. утро", "тревога"]
        if result is None:
            x_vals = (0.0,) * 5
            attended = (0.0,) * 5
            hidden = (0.5,) * 3
            contrib = (0.0,) * 3
            attention = tuple((0.2,) * 5 for _ in range(5))
            z = 0.0
            sigma = 0.5
            mood = 5.5
        else:
            x_vals = result.x
            attended = result.attended
            hidden = result.hidden
            contrib = result.contributions
            attention = result.attention
            z = result.z
            sigma = result.sigma
            mood = result.mood

        cell = min(26, max(16, int((height - 150) / 6)))
        grid = cell * 5
        attn_x = left_x + 92
        attn_y = mid_y - grid / 2
        self._draw_heatmap(c, attn_x, attn_y, cell, attention)

        w1 = self.perceptron.w1
        w2 = self.perceptron.w2
        max_abs = max(
            0.4,
            max(abs(w) for row in w1 for w in row),
            max(abs(w) for w in w2),
        )

        for i, y_in in enumerate(in_ys):
            fill = self._mix("#1e3a4c", ACCENT_2, min(1.0, abs(x_vals[i])))
            self._neuron(c, left_x, y_in, 16, fill, f"x{i + 1}", f"{x_vals[i]:.2f}")
            c.create_text(left_x, y_in + 28, text=labels[i], fill=MUTED, font=("Segoe UI", 8))
            c.create_line(left_x + 18, y_in, attn_x, attn_y + cell * (i + 0.5), fill=STROKE, width=1)

        attn_right = attn_x + grid
        for j, y_h in enumerate(hid_ys):
            for i in range(5):
                w = w1[j][i]
                color = POSITIVE if w >= 0 else NEGATIVE
                thickness = 0.7 + 4.0 * (abs(w) / max_abs)
                c.create_line(
                    attn_right,
                    attn_y + cell * (i + 0.5),
                    hid_x - 24,
                    y_h,
                    fill=color,
                    width=thickness,
                )

        hid_colors = ("#1d4e4a", "#164e63", "#4c1d24")
        for j, y_h in enumerate(hid_ys):
            w = w2[j]
            color = POSITIVE if w >= 0 else NEGATIVE
            thickness = 1.6 + 5.0 * (abs(w) / max_abs)
            c.create_line(hid_x + 24, y_h, out_x - 28, mid_y, fill=color, width=thickness)
            fill = self._mix(hid_colors[j], ACCENT, min(1.0, hidden[j]))
            self._neuron(c, hid_x, y_h, 24, fill, f"h{j + 1}", f"{hidden[j]:.2f}")
            c.create_text(hid_x, y_h + 38, text=HIDDEN_NAMES[j], fill=MUTED, font=("Segoe UI", 9))

        c.create_line(out_x + 28, mid_y, mood_x - 28, mid_y, fill=ACCENT, width=3)
        self._neuron(c, out_x, mid_y, 28, "#164e63", "σ", f"{sigma:.3f}")
        self._neuron(c, mood_x, mid_y, 28, self._mood_color(mood), f"{mood:.1f}", "вечер")
        c.create_text(out_x, mid_y + 44, text=f"z={z:+.2f}", fill=MUTED, font=("Consolas", 8))

        c.create_text(left_x, 16, fill=MUTED, font=("Segoe UI", 9), text="входы")
        c.create_text(attn_x + grid / 2, 16, fill=MUTED, font=("Segoe UI", 9), text="голова внимания")
        c.create_text(hid_x, 16, fill=MUTED, font=("Segoe UI", 9), text="скрытый слой")
        c.create_text(out_x, 16, fill=MUTED, font=("Segoe UI", 9), text="выход")

        box_y = 34
        c.create_text(12, box_y, anchor="w", fill=MUTED, font=("Segoe UI", 8), text="x после внимания")
        for i, value in enumerate(attended):
            c.create_text(
                12,
                box_y + 14 + i * 12,
                anchor="w",
                fill=ACCENT_2,
                font=("Consolas", 8),
                text=f"{TOKEN_LABELS[i]}: {value:.2f}",
            )
        c.create_text(12, box_y + 80, anchor="w", fill=MUTED, font=("Segoe UI", 8), text="вклад w₂·h")
        for i, value in enumerate(contrib):
            color = POSITIVE if value >= 0 else NEGATIVE
            c.create_text(
                12,
                box_y + 94 + i * 13,
                anchor="w",
                fill=color,
                font=("Consolas", 8),
                text=f"{HIDDEN_NAMES[i]}: {value:+.3f}",
            )

    def _draw_heatmap(
        self,
        canvas: tk.Canvas,
        origin_x: float,
        origin_y: float,
        cell: int,
        attention: tuple[tuple[float, ...], ...],
    ) -> None:
        for i, row in enumerate(attention):
            for j, value in enumerate(row):
                color = self._mix("#10202b", ACCENT, min(1.0, value * 1.35))
                x0 = origin_x + j * cell
                y0 = origin_y + i * cell
                canvas.create_rectangle(
                    x0, y0, x0 + cell - 1, y0 + cell - 1, fill=color, outline="#0e141b"
                )
        for i, name in enumerate(TOKEN_LABELS):
            canvas.create_text(
                origin_x - 4,
                origin_y + cell * (i + 0.5),
                anchor="e",
                text=name,
                fill=MUTED,
                font=("Segoe UI", 7),
            )
            canvas.create_text(
                origin_x + cell * (i + 0.5),
                origin_y + cell * 5 + 10,
                text=name,
                fill=MUTED,
                font=("Segoe UI", 7),
            )

    def _neuron(
        self,
        canvas: tk.Canvas,
        x: float,
        y: float,
        r: float,
        fill: str,
        title: str,
        subtitle: str,
    ) -> None:
        canvas.create_oval(x - r, y - r, x + r, y + r, fill=fill, outline=WHITE, width=2)
        canvas.create_text(x, y - 4, text=title, fill=WHITE, font=("Segoe UI Semibold", 10))
        canvas.create_text(x, y + 12, text=subtitle, fill="#dbeafe", font=("Consolas", 8))

    @staticmethod
    def _mix(c1: str, c2: str, t: float) -> str:
        def hex_to_rgb(h: str) -> tuple[int, int, int]:
            h = h.lstrip("#")
            return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

        a = hex_to_rgb(c1)
        b = hex_to_rgb(c2)
        rgb = tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
