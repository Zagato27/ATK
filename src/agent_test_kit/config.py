"""
Centralised configuration for agent-test-kit.

All thresholds, limits and patterns live here so they can be tuned via
a single TOML file instead of being scattered across test classes.

Loading order (first match wins):

1. ``agent-test-kit.toml`` in the current working directory (or any
   parent up to the filesystem root).
2. ``[tool.agent-test-kit]`` section inside ``pyproject.toml``
   (same directory search).
3. Built-in defaults.

Programmatic override::

    from agent_test_kit.config import get_config
    cfg = get_config()
    cfg.performance.first_message_latency = 60.0

Централизованная конфигурация для agent-test-kit.
Все пороги, лимиты и паттерны хранятся здесь и настраиваются через
один TOML-файл вместо разбросанных по тестовым классам.
"""
from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CONFIG_FILENAME = "agent-test-kit.toml"


# ======================================================================
# Section dataclasses / Секции dataclass
# ======================================================================

@dataclass
class PerformanceConfig:
    """Performance-related thresholds. / Пороги, связанные с производительностью."""
    first_message_latency: float = 30.0
    subsequent_message_latency: float = 30.0
    parallel_count: int = 3
    parallel_timeout: int = 120
    max_response_bytes: int = 50_000
    latency_degradation_factor: float = 3.0


@dataclass
class SecurityConfig:
    """Security-related settings (PII patterns, judge thresholds). / Настройки безопасности (паттерны PII, пороги судьи)."""
    judge_threshold: float = 0.7
    pii_patterns: list[str] = field(default_factory=lambda: [
        r"\b\d{4}\s?\d{6}\b",
        r"\b\d{3}-\d{2}-\d{4}\b",
        r"\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b",
        r"\+?\d[\d\s\-]{9,14}\d",
    ])


@dataclass
class FormatConfig:
    """Response format constraints (length, forbidden tokens, forbidden placeholders). / Ограничения формата ответа (длина, запрещённые токены)."""
    min_response_length: int = 2
    max_response_length: int = 10_000
    short_response_min: int = 5
    short_response_max: int = 5000
    forbidden_tokens: list[str] = field(default_factory=lambda: [
        "<|endoftext|>", "<|im_start|>", "<|im_end|>",
        "[INST]", "[/INST]", "<<SYS>>",
    ])
    forbidden_placeholders: list[str] = field(default_factory=lambda: [
        "{{", "}}", "[TODO]", "[INSERT", "[PLACEHOLDER",
    ])
    forbidden_html: list[str] = field(default_factory=lambda: [
        "<div", "<span", "<p>", "</div>", "<br>",
    ])
    forbidden_errors: list[str] = field(default_factory=lambda: [
        "traceback", "exception", "stack trace",
        "internal server error", "NoneType",
        "KeyError", "IndexError", "TypeError",
    ])
    forbidden_debug: list[str] = field(default_factory=lambda: [
        "DEBUG:", "INFO:", "WARNING:", "ERROR:",
        "logger.", "print(",
    ])


@dataclass
class MemoryConfig:
    """Memory and context thresholds. / Пороги памяти и контекста."""
    context_threshold: float = 0.7
    correction_threshold: float = 0.6
    long_context_threshold: float = 0.6


@dataclass
class StabilityConfig:
    """Stability and reproducibility settings. / Настройки стабильности и воспроизводимости."""
    reproducibility_runs: int = 3
    jaccard_min: float = 0.3
    paraphrase_length_ratio_min: float = 0.3
    paraphrase_length_ratio_max: float = 3.0


@dataclass
class OutOfScopeConfig:
    """Out-of-scope detection thresholds. / Пороги определения вне области действия."""
    judge_threshold: float = 0.7
    return_threshold: float = 0.6


@dataclass
class EvaluateConfig:
    """Default evaluation threshold. / Порог оценки по умолчанию."""
    default_threshold: float = 0.7


# ======================================================================
# Root config / Корневая конфигурация
# ======================================================================

@dataclass
class Config:
    """Root configuration object aggregating all sections. / Корневой объект конфигурации, объединяющий все секции."""
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    format: FormatConfig = field(default_factory=FormatConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    out_of_scope: OutOfScopeConfig = field(default_factory=OutOfScopeConfig)
    evaluate: EvaluateConfig = field(default_factory=EvaluateConfig)


# ======================================================================
# Loading helpers / Вспомогательные функции загрузки
# ======================================================================

def _find_file(filename: str, start: Path | None = None) -> Path | None:
    """Walk from *start* (default: cwd) up to filesystem root looking for *filename*. / Ищет *filename* от *start* (по умолчанию: cwd) до корня файловой системы."""
    current = (start or Path.cwd()).resolve()
    for directory in [current, *current.parents]:
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return None


def _apply_section(obj: Any, data: dict[str, Any]) -> None:
    """Overwrite fields of a dataclass instance from a dict. / Перезаписывает поля экземпляра dataclass из словаря."""
    for key, value in data.items():
        if hasattr(obj, key):
            setattr(obj, key, value)
        else:
            logger.warning("Unknown config key '%s' in section — ignored", key)


def _load_from_toml(path: Path) -> dict[str, Any]:
    """Load TOML file and return parsed dict. / Загружает TOML-файл и возвращает распарсенный словарь."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_config(start_dir: Path | None = None) -> Config:
    """Build a :class:`Config` by searching for a TOML file. 1. ``agent-test-kit.toml`` 2. ``pyproject.toml`` → ``[tool.agent-test-kit]`` 3. defaults.

    Создаёт :class:`Config` путём поиска TOML-файла. Порядок: agent-test-kit.toml, pyproject.toml, значения по умолчанию.
    """
    cfg = Config()

    toml_path = _find_file(_CONFIG_FILENAME, start_dir)
    if toml_path:
        logger.info("Loading config from %s", toml_path)
        raw = _load_from_toml(toml_path)
    else:
        pyproject = _find_file("pyproject.toml", start_dir)
        if pyproject:
            full = _load_from_toml(pyproject)
            raw = full.get("tool", {}).get("agent-test-kit", {})
            if raw:
                logger.info("Loading config from %s [tool.agent-test-kit]", pyproject)
            else:
                raw = {}
        else:
            raw = {}

    section_map = {
        "performance": cfg.performance,
        "security": cfg.security,
        "format": cfg.format,
        "memory": cfg.memory,
        "stability": cfg.stability,
        "out_of_scope": cfg.out_of_scope,
        "evaluate": cfg.evaluate,
    }
    for section_name, section_obj in section_map.items():
        section_data = raw.get(section_name, {})
        if section_data:
            _apply_section(section_obj, section_data)

    return cfg


# ======================================================================
# Singleton accessor / Синглтон-доступ
# ======================================================================

_cached_config: Config | None = None


def get_config() -> Config:
    """Return the cached global :class:`Config` (loaded once on first call). To force a reload use reload_config(). / Возвращает кэшированную глобальную :class:`Config` (загружается при первом вызове). Для перезагрузки используйте reload_config()."""
    global _cached_config
    if _cached_config is None:
        _cached_config = load_config()
    return _cached_config


def reload_config(start_dir: Path | None = None) -> Config:
    """Discard the cached config and reload from disk. / Сбрасывает кэш конфигурации и перезагружает с диска."""
    global _cached_config
    _cached_config = load_config(start_dir)
    return _cached_config
