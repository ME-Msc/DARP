from darp.rddl.basic_parser import BasicRDDLParser


def test_basic_rddl_parser_builds_ast_and_dot():
    ast = BasicRDDLParser().parse_files(
        "examples/rddl/tiny_grid_domain.rddl",
        "examples/rddl/tiny_grid_instance.rddl",
    )
    dot = ast.to_dot()

    assert "nodes=" in ast.summary()
    assert "domain" in dot
    assert "tiny_grid" in dot
    assert "instance" in dot
    assert "tiny_grid_inst" in dot
    assert "digraph RDDLAST" in dot
