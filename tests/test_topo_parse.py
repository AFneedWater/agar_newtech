from agar.tools.parse_topo_affinity import parse_topo_affinity


def test_parse_topo_affinity_sample():
    topo = (
        "GPU0 GPU1 GPU2 CPU Affinity NUMA Affinity GPU NUMA ID\n"
        "GPU0 X SYS SYS 0-15,128-143 N/A N/A\n"
        "GPU1 SYS X SYS 96-111,224-239 N/A N/A\n"
        "GPU2 SYS SYS X 192-207,320-335 N/A N/A\n"
    )
    out = parse_topo_affinity(topo)
    assert out["GPU0"] == "0-15,128-143"
    assert out["GPU1"] == "96-111,224-239"
    assert out["GPU2"] == "192-207,320-335"
