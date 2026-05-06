from darp.rddl.basic_parser import BasicRDDLParser
from darp.rddl.compiler import RDDLCompiler
from darp.rddl.loader import RDDLLoader
from darp.rddl.visualizer import _RuntimeController, _runtime_markers, _runtime_payload, render_html


def test_basic_rddl_parser_builds_ast():
    """Check that the basic parser emits an inspectable AST. / 检查基础 parser 能生成可检查的 AST。"""
    ast = BasicRDDLParser().parse_files(
        "examples/rddl/tiny_grid_domain.rddl",
        "examples/rddl/tiny_grid_instance.rddl",
    )

    assert "nodes=" in ast.summary()
    top_level = {node.kind: node for file_node in ast.children for node in file_node.children}
    assert top_level["domain"].label == "tiny_grid"
    assert top_level["non-fluents"].label == "tiny_grid_nf"
    assert top_level["instance"].label == "tiny_grid_inst"


def test_basic_rddl_visualizer_renders_html():
    """Check that the HTML AST visualizer includes controls and highlighting. / 检查 HTML AST 可视化包含控件和高亮。"""
    ast = BasicRDDLParser().parse_files(
        "examples/rddl/tiny_grid_domain.rddl",
        "examples/rddl/tiny_grid_instance.rddl",
    )
    html = render_html(ast)

    assert "<html" in html
    assert "RDDL AST" in html
    assert "tiny_grid" in html
    assert "<svg" in html
    assert "node-domain" in html
    assert "id=\"expand-all\"" in html
    assert "id=\"collapse-all\"" in html
    assert "id=\"search-input\"" in html
    assert "id=\"text-outline\"" in html
    assert "RDDL Source" in html
    assert "Source and AST controls" in html
    assert "panel-toggles" in html
    assert "id=\"toggle-source-panel\"" in html
    assert "id=\"toggle-ast-panel\"" in html
    assert "id=\"toggle-runtime-panel\"" in html
    assert "setupPanelToggles" in html
    assert "setupAstCanvasPan" in html
    assert "applyAstViewBox" in html
    assert "clampAstPan" in html
    assert "astPan" in html
    assert "onlyOnePanelVisible" in html
    assert "[hidden]" in html
    assert "nodeMetrics" in html
    assert "ast-panning" in html
    assert "node-collapsed" in html
    assert "collapsed-summary" in html
    assert "collapsedLines" in html
    assert "contains:" in html
    assert "runtime-edge-label" in html
    assert "runtime-info" in html
    assert "runtimeGraphZoom" in html
    assert "runtimeDraggedPositions" in html
    assert "runtimeDraggedEdges" in html
    assert "runtimeDraggedLabels" in html
    assert "ResizeObserver" in html
    assert "setRuntimeZoom" in html
    assert "startRuntimeNodeDrag" in html
    assert "startRuntimeLinkedElementDrag" in html
    assert "runtimeSvgPoint" in html
    assert "runtime-node-group" in html
    assert "runtime-edge-drag-area" in html
    assert "runtime-edge-label-group" in html
    assert "Zoom in state machine" in html
    assert "Auto" in html
    assert "ast-tools" in html
    assert "groupedTransitions" in html
    assert "renderStateMachineSvg" in html
    assert "source-line" in html
    assert "source-gutter" in html
    assert "source-resizer" in html
    assert "split-resizer" in html
    assert "Expand all" in html
    assert "Reveal paths" in html
    assert "Match case" in html
    assert "Match whole word" in html
    assert "Use regular expression" in html
    assert "toggle-button" in html
    assert "source-block" in html
    assert "selectNode" in html
    assert "setupSplitResizer" in html
    assert "setupSourceResizers" in html
    assert "match-bar" in html
    assert "tok-keyword" in html
    assert "tok-symbol" in html
    assert "xml:space" in html
    assert "Execution State Machine" not in html


def test_basic_rddl_visualizer_renders_internal_runtime_panel():
    """Check internal simulator HTML includes a state machine panel. / 检查内部 simulator HTML 包含状态机面板。"""
    ast = BasicRDDLParser().parse_files(
        "examples/rddl/tiny_grid_domain.rddl",
        "examples/rddl/tiny_grid_instance.rddl",
    )
    problem = RDDLCompiler().compile(
        RDDLLoader("darp").load(
            "examples/rddl/tiny_grid_domain.rddl",
            "examples/rddl/tiny_grid_instance.rddl",
        )
    )
    markers = _runtime_markers(ast)
    controller = _RuntimeController(problem, markers)
    html = render_html(
        ast,
        runtime=_runtime_payload(problem, markers, controller.snapshot(), endpoint="/api"),
    )

    assert "Execution State Machine" in html
    assert "<aside class=\"runtime-panel\"" in html
    assert "laneOffset" in html
    assert "Reset dragged node, edge, and label positions" in html
    assert "runtime-edge-label-link" in html
    assert "DARP action" in html
    assert "Belief peak" in html
    assert "runtime-action" not in html


def test_basic_rddl_visualizer_hides_state_machine_for_external_simulator():
    """Check external simulator mode does not render DARP's state machine. / 检查外部 simulator 模式不渲染 DARP 状态机。"""
    ast = BasicRDDLParser().parse_files(
        "examples/rddl/tiny_grid_domain.rddl",
        "examples/rddl/tiny_grid_instance.rddl",
    )
    html = render_html(ast, runtime={"enabled": True, "mode": "external", "simulator": "rddlgym"})

    assert "Execution State Machine" not in html
    assert "<aside class=\"runtime-panel\"" not in html
    assert "RDDL Source" in html
    assert "AST" in html


def test_runtime_controller_plans_and_steps_grid_actions():
    """Check DARP selects actions while HTML only advances the runtime. / 检查 DARP 选择动作而 HTML 只推进运行时。"""
    ast = BasicRDDLParser().parse_files(
        "examples/rddl/tiny_grid_domain.rddl",
        "examples/rddl/tiny_grid_instance.rddl",
    )
    problem = RDDLCompiler().compile(
        RDDLLoader("darp").load(
            "examples/rddl/tiny_grid_domain.rddl",
            "examples/rddl/tiny_grid_instance.rddl",
        )
    )
    controller = _RuntimeController(problem, _runtime_markers(ast))

    snapshot = controller.snapshot()
    assert snapshot["state"] == "c11"
    assert snapshot["planner"] == "finite-horizon-dp"
    assert snapshot["belief"]["c11"] == 1.0
    assert snapshot["planned_action"] == "move-east"

    snapshot = controller.step()
    assert snapshot["state"] == "c12"
    assert snapshot["last_action"] == "move-east"
    assert snapshot["belief"]["c12"] == 1.0
    assert snapshot["planned_action"] == "move-east"
