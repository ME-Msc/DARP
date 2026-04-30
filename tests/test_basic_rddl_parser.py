from darp.rddl.basic_parser import BasicRDDLParser
from darp.rddl.visualizer import render_html


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
    assert "Expand all" in html
    assert "Reveal paths" in html
    assert "Match case" in html
    assert "Match whole word" in html
    assert "Use regular expression" in html
    assert "toggle-button" in html
    assert "match-bar" in html
    assert "tok-keyword" in html
    assert "tok-symbol" in html
    assert "xml:space" in html
