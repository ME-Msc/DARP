"""RDDL loading through pyRDDLGym."""

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
    """Load one DARP-extended RDDL domain/instance pair through pyRDDLGym."""
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
    return PyRDDLGymProblem(native_ast=native_ast, env=env)


@lru_cache(maxsize=1)
def _extended_rddl_parser() -> Any:
    """Build pyRDDLGym's parser with DARP's two top-level extensions."""

    import ply.yacc as yacc
    from pyRDDLGym.core.debug.exception import RDDLParseError
    from pyRDDLGym.core.parser.domain import Domain
    from pyRDDLGym.core.parser.parser import RDDLParser

    class DARPRDDLParser(RDDLParser):
        """Reuse the native expression grammar without patching pyRDDLGym."""

        def build(self, **kwargs: Any) -> None:
            kwargs.setdefault("start", "rddl")
            super().build(**kwargs)

        def parse(self, input: str) -> Any:
            # PLY parsers are mutable and retain lexer line numbers between
            # calls, so serialize use of the one cached grammar instance.
            with _PARSER_LOCK:
                self.lexer._lexer.lineno = 1
                ast = super().parse(input)
            if not hasattr(ast.domain, "duration"):
                raise RDDLParseError(
                    "DARP domains require a top-level 'duration = <expr>;' section."
                )
            return ast

        def p_error(self, p: Any) -> None:
            if p is None:
                raise RDDLParseError("Unexpected end of RDDL input.")
            super().p_error(p)

        def p_domain_block(self, p: Any) -> None:
            """domain_block : DOMAIN IDENT LCURLY req_section domain_list RCURLY"""
            sections = p[5]
            domain = Domain(p[2], p[4], sections)
            if "duration" in sections:
                domain.duration = sections["duration"]
            p[0] = ("domain", domain)

        def p_domain_list_darp_extension(self, p: Any) -> None:
            """domain_list : domain_list darp_extension_section"""
            name, value = p[2]
            if name != "duration":
                raise ValueError(f"Unknown DARP domain section {name!r}.")
            if name in p[1]:
                raise ValueError("DARP domain contains duplicate duration sections.")
            p[1][name] = value
            p[0] = p[1]

        def p_instance_list_darp_extension(self, p: Any) -> None:
            """instance_list : instance_list darp_extension_section"""
            name, expression = p[2]
            if name != "max-duration-shortfall-probability":
                raise ValueError(f"Unknown DARP instance section {name!r}.")
            key = "max_duration_shortfall_probability"
            if key in p[1]:
                raise ValueError(
                    "DARP instance contains duplicate "
                    "max-duration-shortfall-probability sections."
                )
            value = _constant_probability(expression)
            p[1][key] = value
            p[0] = p[1]

        def p_darp_extension_section(self, p: Any) -> None:
            """darp_extension_section : IDENT ASSIGN_EQUAL expr SEMI"""
            p[0] = (p[1], p[3])

    parser = DARPRDDLParser(lexer=None, verbose=False)
    parser.build(write_tables=False, debug=False, errorlog=yacc.NullLogger())
    return parser


def _constant_probability(expression: Any) -> float:
    """Read a literal probability from a parsed RDDL expression."""

    if getattr(expression, "etype", (None,))[0] != "constant":
        raise ValueError("max-duration-shortfall-probability must be a number.")
    value = getattr(expression, "value", None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("max-duration-shortfall-probability must be a number.")
    probability = float(value)
    if not isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(
            "max-duration-shortfall-probability must be finite and in [0, 1]."
        )
    return probability


def _ensure_matplotlib_cache_dir() -> None:
    """Give pyRDDLGym's matplotlib import a writable cache. / 为 pyRDDLGym 的 matplotlib 导入提供可写缓存。"""
    if "MPLCONFIGDIR" in os.environ:
        return
    cache_dir = Path(tempfile.gettempdir()) / "darp-matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
