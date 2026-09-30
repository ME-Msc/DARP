"""RDDL loading through pyRDDLGym. / 通过 pyRDDLGym 加载 RDDL。"""

from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from math import isfinite
from pathlib import Path
from threading import Lock
from typing import Any

from darp.adapter.problem import PyRDDLGymProblem, RDDLLoadError

_PARSER_LOCK = Lock()


def load_rddl(domain: str | Path, instance: str | Path) -> PyRDDLGymProblem:
    """Load standard or DARP-extended RDDL through pyRDDLGym. / 加载标准 RDDL 或 DARP 扩展 RDDL。"""
    domain_path = Path(domain).expanduser()
    instance_path = Path(instance).expanduser()
    _ensure_matplotlib_cache_dir()
    try:
        import pyRDDLGym
        from pyRDDLGym.core.compiler.model import RDDLLiftedModel
        from pyRDDLGym.core.parser.reader import RDDLReader
    except ImportError as exc:
        raise RDDLLoadError(
            "pyRDDLGym is required to load RDDL. "
            "Install with `pip install -e .` or run `tools/install.sh`."
        ) from exc

    try:
        reader = RDDLReader(str(domain_path), str(instance_path))
        parser = _extended_rddl_parser()
        native_ast = parser.parse(reader.rddltxt)
        env = pyRDDLGym.make(RDDLLiftedModel(native_ast), None)
    except Exception as exc:
        raise RDDLLoadError(
            "pyRDDLGym failed to load "
            f"domain={domain_path} instance={instance_path}: {exc}"
        ) from exc
    model = getattr(env, "model", None)
    if model is None:
        raise RDDLLoadError(
            "pyRDDLGym environment did not expose a grounded model source."
        )
    native_ast = getattr(model, "ast", None)
    if native_ast is None:
        raise RDDLLoadError(
            "pyRDDLGym model did not expose the RDDL AST required for grounding."
        )
    try:
        return PyRDDLGymProblem(native_ast=native_ast, env=env)
    except RDDLLoadError:
        env.close()
        raise


@lru_cache(maxsize=1)
def _extended_rddl_parser() -> Any:
    """Extend pyRDDLGym's top-level grammar for duration and CC-POMDP risk. / 为时长与 CC-POMDP 风险扩展顶层语法。"""

    from ply import yacc
    from pyRDDLGym.core.debug.exception import RDDLParseError
    from pyRDDLGym.core.parser.domain import Domain
    from pyRDDLGym.core.parser.parser import RDDLParser

    class DARPRDDLParser(RDDLParser):
        """Reuse the native expression grammar without patching pyRDDLGym. / 复用原生表达式语法，不修改 pyRDDLGym 源码。"""

        def build(self, **kwargs: Any) -> None:
            kwargs.setdefault("start", "rddl")
            super().build(**kwargs)

        def parse(self, input: str) -> Any:
            # PLY parsers are mutable and retain lexer line numbers between
            # calls, so serialize use of the one cached grammar instance.
            # PLY 会保留可变状态及词法行号，因此缓存的语法实例必须串行使用。
            with _PARSER_LOCK:
                self.lexer._lexer.lineno = 1
                return super().parse(input)

        def p_error(self, p: Any) -> None:
            if p is None:
                raise RDDLParseError("Unexpected end of RDDL input.")
            super().p_error(p)

        def p_domain_block(self, p: Any) -> None:
            """domain_block : DOMAIN IDENT LCURLY req_section domain_list RCURLY"""
            sections = p[5]
            domain = Domain(p[2], p[4], sections)
            for name in ("duration", "risk"):
                if name in sections:
                    setattr(domain, name, sections[name])
            p[0] = ("domain", domain)

        def p_domain_list_darp_extension(self, p: Any) -> None:
            """domain_list : domain_list darp_extension_section"""
            name, value = p[2]
            if name not in ("duration", "risk"):
                raise ValueError(f"Unknown DARP domain section {name!r}.")
            if name in p[1]:
                raise ValueError(f"DARP domain contains duplicate {name} sections.")
            p[1][name] = value
            p[0] = p[1]

        def p_instance_list_darp_extension(self, p: Any) -> None:
            """instance_list : instance_list darp_extension_section"""
            name, expression = p[2]
            if name not in ("max-duration-shortfall-probability", "risk-budget"):
                raise ValueError(f"Unknown DARP instance section {name!r}.")
            key = name.replace("-", "_")
            if key in p[1]:
                raise ValueError(f"DARP instance contains duplicate {name} sections.")
            value = _constant_probability(expression, name)
            p[1][key] = value
            p[0] = p[1]

        def p_darp_extension_section(self, p: Any) -> None:
            """darp_extension_section : IDENT ASSIGN_EQUAL expr SEMI"""
            p[0] = (p[1], p[3])

    parser = DARPRDDLParser(lexer=None, verbose=False)
    parser.build(write_tables=False, debug=False, errorlog=yacc.NullLogger())
    return parser


def _constant_probability(expression: Any, name: str) -> float:
    """Read a literal probability from a parsed RDDL expression. / 从已解析表达式读取概率字面量。"""

    if getattr(expression, "etype", (None,))[0] != "constant":
        raise ValueError(f"{name} must be a number.")
    value = getattr(expression, "value", None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number.")
    probability = float(value)
    if not isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1].")
    return probability


def _ensure_matplotlib_cache_dir() -> None:
    """Give pyRDDLGym's matplotlib import a writable cache. / 为 pyRDDLGym 的 matplotlib 导入提供可写缓存。"""
    if "MPLCONFIGDIR" in os.environ:
        return
    cache_dir = Path(tempfile.gettempdir()) / "darp-matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
