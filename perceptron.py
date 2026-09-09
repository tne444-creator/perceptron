"""Учебная сеть: внимание (1 голова) → скрытый слой → прогноз настроения."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence


INPUT_NAMES = (
    "Сон сегодня, ч",
    "Сон вчера, ч",
    "Настроение вчера вечером",
    "Настроение сегодня утром",
    "Тревога с утра",
)
HIDDEN_NAMES = ("тело", "эмоции", "тревога")
TOKEN_LABELS = ("сонС", "сонВ", "нВч", "нУтр", "трев")

N_TOKENS = 5
D_HEAD = 4
ATTN_SCALE = 2.0  # √4

SLEEP_SCALE = 10.0
MOOD_MIN = 1.0
MOOD_MAX = 10.0
MOOD_SPAN = MOOD_MAX - MOOD_MIN

# Тип токена: [значение, сон, эмоции, тревога] — как крошечный positional/type embedding.
TOKEN_TYPES = (
    (1.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)

DEFAULT_W1 = (
    (1.20, 0.60, 0.10, 0.10, -0.10),
    (0.10, 0.05, 1.00, 1.40, -0.20),
    (0.05, 0.05, 0.10, 0.20, 2.50),
)
DEFAULT_B1 = (-0.80, -0.90, -1.20)
DEFAULT_W2 = (1.40, 1.80, -1.60)
DEFAULT_B2 = -0.90

ACTIVATION_NAME = "сигмоида"
ACTIVATION_FORMULA = "1 / (1 + exp(-z))"

ATTN_CSV = tuple(f"attn_{i + 1}_{j + 1}" for i in range(N_TOKENS) for j in range(N_TOKENS))
XATTN_CSV = tuple(f"x_после_внимания_{i + 1}" for i in range(N_TOKENS))

CSV_FIELDS = (
    "время",
    "сон_сегодня_ч",
    "сон_вчера_ч",
    "настроение_вчера_вечером",
    "настроение_сегодня_утром",
    "тревога_с_утра",
    "x1_сон_сегодня",
    "x2_сон_вчера",
    "x3_настроение_вчера",
    "x4_настроение_утро",
    "x5_тревога",
    *XATTN_CSV,
    *ATTN_CSV,
    "z_h1_тело",
    "z_h2_эмоции",
    "z_h3_тревога",
    "h1_тело",
    "h2_эмоции",
    "h3_тревога",
    "w1_тело_сон_сегодня",
    "w1_тело_сон_вчера",
    "w1_тело_настр_вчера",
    "w1_тело_настр_утро",
    "w1_тело_тревога",
    "w1_эмоции_сон_сегодня",
    "w1_эмоции_сон_вчера",
    "w1_эмоции_настр_вчера",
    "w1_эмоции_настр_утро",
    "w1_эмоции_тревога",
    "w1_тревога_сон_сегодня",
    "w1_тревога_сон_вчера",
    "w1_тревога_настр_вчера",
    "w1_тревога_настр_утро",
    "w1_тревога_тревога",
    "b1_тело",
    "b1_эмоции",
    "b1_тревога",
    "w2_тело",
    "w2_эмоции",
    "w2_тревога",
    "удельный_w2_тело",
    "удельный_w2_эмоции",
    "удельный_w2_тревога",
    "вклад_w2h1",
    "вклад_w2h2",
    "вклад_w2h3",
    "смещение_b2",
    "z",
    "функция_активации",
    "формула_активации",
    "значение_активации",
    "прогноз_настроения_вечером",
)


def sigmoid(z: float) -> float:
    z = max(-60.0, min(60.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def sigmoid_derivative(sigma: float) -> float:
    return sigma * (1.0 - sigma)


def clamp_mood(value: float) -> float:
    return max(MOOD_MIN, min(MOOD_MAX, value))


def mood_to_unit(mood: float) -> float:
    return (clamp_mood(mood) - MOOD_MIN) / MOOD_SPAN


def unit_to_mood(unit: float) -> float:
    return MOOD_MIN + MOOD_SPAN * unit


def _copy_matrix(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    return [list(row) for row in matrix]


def _eye(n: int, scale: float) -> list[list[float]]:
    return [[scale if i == j else 0.0 for j in range(n)] for i in range(n)]


def _zeros(rows: int, cols: int) -> list[list[float]]:
    return [[0.0] * cols for _ in range(rows)]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _transpose(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    return [list(col) for col in zip(*matrix)]


def _matmul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> list[list[float]]:
    bt = _transpose(b)
    return [[_dot(row, col) for col in bt] for row in a]


def _softmax(row: Sequence[float]) -> list[float]:
    peak = max(row)
    exps = [math.exp(max(-60.0, min(60.0, value - peak))) for value in row]
    total = sum(exps) or 1.0
    return [value / total for value in exps]


def make_tokens(x: Sequence[float]) -> list[list[float]]:
    return [[float(x[i]), *TOKEN_TYPES[i]] for i in range(N_TOKENS)]


DEFAULT_WQ = _eye(D_HEAD, 1.15)
DEFAULT_WK = _eye(D_HEAD, 1.15)
DEFAULT_WV = _eye(D_HEAD, 0.08)


@dataclass
class ForwardResult:
    raw: tuple[float, ...]
    x: tuple[float, ...]
    attended: tuple[float, ...]
    attention: tuple[tuple[float, ...], ...]
    tokens: tuple[tuple[float, ...], ...]
    queries: tuple[tuple[float, ...], ...]
    keys: tuple[tuple[float, ...], ...]
    values: tuple[tuple[float, ...], ...]
    hidden_z: tuple[float, ...]
    hidden: tuple[float, ...]
    hidden_contrib: tuple[tuple[float, ...], ...]
    z: float
    sigma: float
    mood: float
    contributions: tuple[float, ...]


def _fmt(value: float, digits: int = 6) -> str:
    if not math.isfinite(value):
        return ""
    text = f"{value:.{digits}f}"
    if digits > 0 and "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def specific_weights(weights: Sequence[float]) -> tuple[float, ...]:
    total = sum(abs(float(w)) for w in weights)
    if total == 0:
        return tuple(0.0 for _ in weights)
    return tuple(abs(float(w)) / total for w in weights)


def snapshot_record(model: "Perceptron", result: ForwardResult) -> dict[str, str]:
    shares = specific_weights(model.w2)
    raw = result.raw
    record: dict[str, str] = {
        "время": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "сон_сегодня_ч": _fmt(raw[0], 1),
        "сон_вчера_ч": _fmt(raw[1], 1),
        "настроение_вчера_вечером": _fmt(raw[2], 0),
        "настроение_сегодня_утром": _fmt(raw[3], 0),
        "тревога_с_утра": _fmt(raw[4], 0),
        "x1_сон_сегодня": _fmt(result.x[0]),
        "x2_сон_вчера": _fmt(result.x[1]),
        "x3_настроение_вчера": _fmt(result.x[2]),
        "x4_настроение_утро": _fmt(result.x[3]),
        "x5_тревога": _fmt(result.x[4]),
        "z_h1_тело": _fmt(result.hidden_z[0]),
        "z_h2_эмоции": _fmt(result.hidden_z[1]),
        "z_h3_тревога": _fmt(result.hidden_z[2]),
        "h1_тело": _fmt(result.hidden[0]),
        "h2_эмоции": _fmt(result.hidden[1]),
        "h3_тревога": _fmt(result.hidden[2]),
        "b1_тело": _fmt(model.b1[0]),
        "b1_эмоции": _fmt(model.b1[1]),
        "b1_тревога": _fmt(model.b1[2]),
        "w2_тело": _fmt(model.w2[0]),
        "w2_эмоции": _fmt(model.w2[1]),
        "w2_тревога": _fmt(model.w2[2]),
        "удельный_w2_тело": _fmt(shares[0]),
        "удельный_w2_эмоции": _fmt(shares[1]),
        "удельный_w2_тревога": _fmt(shares[2]),
        "вклад_w2h1": _fmt(result.contributions[0]),
        "вклад_w2h2": _fmt(result.contributions[1]),
        "вклад_w2h3": _fmt(result.contributions[2]),
        "смещение_b2": _fmt(model.b2),
        "z": _fmt(result.z),
        "функция_активации": model.activation_name,
        "формула_активации": model.activation_formula,
        "значение_активации": _fmt(result.sigma),
        "прогноз_настроения_вечером": _fmt(result.mood, 4),
    }
    for i, value in enumerate(result.attended):
        record[f"x_после_внимания_{i + 1}"] = _fmt(value)
    for i, row in enumerate(result.attention):
        for j, value in enumerate(row):
            record[f"attn_{i + 1}_{j + 1}"] = _fmt(value)
    keys_w1 = [name for name in CSV_FIELDS if name.startswith("w1_")]
    flat_w1 = [w for row in model.w1 for w in row]
    for key, value in zip(keys_w1, flat_w1):
        record[key] = _fmt(value)
    return record


def save_snapshot_csv(path: str | Path, record: dict[str, str]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    new_file = not target.exists() or target.stat().st_size == 0
    with target.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(CSV_FIELDS),
            delimiter=";",
            extrasaction="ignore",
        )
        if new_file:
            writer.writeheader()
        writer.writerow(record)
    return target


@dataclass
class Perceptron:
    w1: list[list[float]] = field(default_factory=lambda: _copy_matrix(DEFAULT_W1))
    b1: list[float] = field(default_factory=lambda: list(DEFAULT_B1))
    w2: list[float] = field(default_factory=lambda: list(DEFAULT_W2))
    b2: float = DEFAULT_B2
    w_q: list[list[float]] = field(default_factory=lambda: _copy_matrix(DEFAULT_WQ))
    w_k: list[list[float]] = field(default_factory=lambda: _copy_matrix(DEFAULT_WK))
    w_v: list[list[float]] = field(default_factory=lambda: _copy_matrix(DEFAULT_WV))
    activation_name: str = ACTIVATION_NAME
    activation_formula: str = ACTIVATION_FORMULA

    def reset(self) -> None:
        self.w1 = _copy_matrix(DEFAULT_W1)
        self.b1 = list(DEFAULT_B1)
        self.w2 = list(DEFAULT_W2)
        self.b2 = DEFAULT_B2
        self.w_q = _copy_matrix(DEFAULT_WQ)
        self.w_k = _copy_matrix(DEFAULT_WK)
        self.w_v = _copy_matrix(DEFAULT_WV)

    def normalize(self, raw: Sequence[float]) -> tuple[float, ...]:
        sleep_today, sleep_yesterday, mood_evening, mood_morning, anxiety = raw
        return (
            max(0.0, float(sleep_today)) / SLEEP_SCALE,
            max(0.0, float(sleep_yesterday)) / SLEEP_SCALE,
            mood_to_unit(float(mood_evening)),
            mood_to_unit(float(mood_morning)),
            1.0 if float(anxiety) >= 0.5 else 0.0,
        )

    def _attention(self, x: Sequence[float]) -> tuple[list[float], list[list[float]], dict[str, list[list[float]]]]:
        tokens = make_tokens(x)
        queries = _matmul(tokens, self.w_q)
        keys = _matmul(tokens, self.w_k)
        values = _matmul(tokens, self.w_v)
        scores = _zeros(N_TOKENS, N_TOKENS)
        for i in range(N_TOKENS):
            for j in range(N_TOKENS):
                scores[i][j] = _dot(queries[i], keys[j]) / ATTN_SCALE
        attention = [_softmax(row) for row in scores]
        context = _matmul(attention, values)
        attended = [tokens[i][0] + context[i][0] for i in range(N_TOKENS)]
        cache = {
            "tokens": tokens,
            "queries": queries,
            "keys": keys,
            "values": values,
            "scores": scores,
            "context": context,
        }
        return attended, attention, cache

    def forward(self, raw: Sequence[float]) -> ForwardResult:
        x = self.normalize(raw)
        attended, attention, cache = self._attention(x)
        hidden_z: list[float] = []
        hidden: list[float] = []
        hidden_contrib: list[tuple[float, ...]] = []
        for weights, bias in zip(self.w1, self.b1):
            contrib = tuple(w * xi for w, xi in zip(weights, attended))
            z_h = sum(contrib) + bias
            hidden_z.append(z_h)
            hidden.append(sigmoid(z_h))
            hidden_contrib.append(contrib)
        contributions = tuple(w * h for w, h in zip(self.w2, hidden))
        z = sum(contributions) + self.b2
        sigma = sigmoid(z)
        return ForwardResult(
            raw=tuple(float(v) for v in raw),
            x=x,
            attended=tuple(attended),
            attention=tuple(tuple(row) for row in attention),
            tokens=tuple(tuple(row) for row in cache["tokens"]),
            queries=tuple(tuple(row) for row in cache["queries"]),
            keys=tuple(tuple(row) for row in cache["keys"]),
            values=tuple(tuple(row) for row in cache["values"]),
            hidden_z=tuple(hidden_z),
            hidden=tuple(hidden),
            hidden_contrib=tuple(hidden_contrib),
            z=z,
            sigma=sigma,
            mood=unit_to_mood(sigma),
            contributions=contributions,
        )

    def _update_attention(self, result: ForwardResult, d_attended: Sequence[float], lr: float) -> None:
        tokens = [list(row) for row in result.tokens]
        queries = [list(row) for row in result.queries]
        keys = [list(row) for row in result.keys]
        values = [list(row) for row in result.values]
        attention = [list(row) for row in result.attention]
        d_context = _zeros(N_TOKENS, D_HEAD)
        for i in range(N_TOKENS):
            d_context[i][0] = d_attended[i]
        d_attention = _matmul(d_context, _transpose(values))
        d_values = _matmul(_transpose(attention), d_context)
        d_scores = _zeros(N_TOKENS, N_TOKENS)
        for i in range(N_TOKENS):
            row = attention[i]
            grad = d_attention[i]
            dotted = _dot(grad, row)
            for j in range(N_TOKENS):
                d_scores[i][j] = row[j] * (grad[j] - dotted)
        d_queries = _zeros(N_TOKENS, D_HEAD)
        d_keys = _zeros(N_TOKENS, D_HEAD)
        for i in range(N_TOKENS):
            for j in range(N_TOKENS):
                scale = d_scores[i][j] / ATTN_SCALE
                for k in range(D_HEAD):
                    d_queries[i][k] += scale * keys[j][k]
                    d_keys[j][k] += scale * queries[i][k]
        tokens_t = _transpose(tokens)
        d_wq = _matmul(tokens_t, d_queries)
        d_wk = _matmul(tokens_t, d_keys)
        d_wv = _matmul(tokens_t, d_values)
        for i in range(D_HEAD):
            for j in range(D_HEAD):
                self.w_q[i][j] -= lr * d_wq[i][j]
                self.w_k[i][j] -= lr * d_wk[i][j]
                self.w_v[i][j] -= lr * d_wv[i][j]

    def train(
        self,
        raw: Sequence[float],
        target_mood: float,
        lr: float = 0.35,
        epochs: int = 25,
    ) -> float:
        target = mood_to_unit(target_mood)
        last_error = 0.0
        attn_lr = lr * 0.8
        for _ in range(max(1, epochs)):
            result = self.forward(raw)
            error = result.sigma - target
            last_error = error
            delta_out = error * sigmoid_derivative(result.sigma)
            delta_h = [
                delta_out * self.w2[j] * sigmoid_derivative(result.hidden[j])
                for j in range(3)
            ]
            for j in range(3):
                self.w2[j] -= lr * delta_out * result.hidden[j]
            self.b2 -= lr * delta_out
            d_attended = [0.0] * N_TOKENS
            for j in range(3):
                for i, xi in enumerate(result.attended):
                    d_attended[i] += delta_h[j] * self.w1[j][i]
                    self.w1[j][i] -= lr * delta_h[j] * xi
                self.b1[j] -= lr * delta_h[j]
            self._update_attention(result, d_attended, attn_lr)
        return last_error
