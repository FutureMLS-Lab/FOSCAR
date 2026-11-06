"""Head-tile occupancy must reach both split and merge planners."""
import importlib.util
from pathlib import Path
import pytest
ROOT = Path(__file__).parents[1]

def load(arch):
    spec=importlib.util.spec_from_file_location("gqa_scheduler_"+arch,ROOT/"src/foscar"/arch/"scheduler.py")
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

@pytest.mark.parametrize("tiles", [1,2,4,16])
@pytest.mark.parametrize("arch", ["_sm90","_sm100"])
def test_head_tiles_have_the_same_occupancy_as_independent_groups(arch,tiles):
    m=load(arch)
    if arch=="_sm90":
        plan=lambda heads,ht:m.plan_splits(1,heads,32768,bits=2,head_dim=128,sm_count=132,head_tiles=ht)
    else:
        plan=lambda heads,ht:m.plan_splits_sm100(1,heads,32768,bits=2,sm_count=148,head_tiles=ht)
    assert plan(2,tiles)==plan(2*tiles,1)

def test_fused_merge_counts_virtual_head_groups():
    m=load("_sm90")
    assert m.plan_fused_combine(1,2,4,head_tiles=8)
    assert not m.plan_fused_combine(1,2,4,head_tiles=16)

@pytest.mark.parametrize("tiles", [0,-1,True,1.5])
@pytest.mark.parametrize("arch", ["_sm90","_sm100"])
def test_invalid_head_tiles_are_refused(arch,tiles):
    m=load(arch)
    with pytest.raises(ValueError,match="head_tiles"):
        if arch=="_sm90":m.plan_splits(1,8,1024,bits=2,head_dim=128,sm_count=132,head_tiles=tiles)
        else:m.plan_splits_sm100(1,8,1024,bits=2,head_tiles=tiles)
