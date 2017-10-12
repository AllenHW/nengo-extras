import re
import xml.etree.ElementTree as et

import nengo
import pytest

from nengo_extras.gexf import (
    CollapsingGexfConverter, DispatchTable, GexfConverter, InspectiveLabeler)


def test_can_dispatch_table_defaults():
    class A(object):
        def __init__(self):
            self.called = False

    class Test(object):
        dispatch = DispatchTable()

        @dispatch.register(A)
        def process_a(self, obj):
            obj.called = True

    a = A()
    Test().dispatch(a)
    assert a.called


def test_dispatch_obj_inheritance():
    class A(object):
        def __init__(self):
            self.called = False

    class B(A):
        pass

    class C(A):
        pass

    class Test(object):
        dispatch = DispatchTable()

        @dispatch.register(A)
        def process_a(self, obj):
            obj.called = 'a'

        @dispatch.register(B)
        def process_b(self, obj):
            obj.called = 'b'

    a = A()
    Test().dispatch(a)
    assert a.called == 'a'

    b = B()
    Test().dispatch(b)
    assert b.called == 'b'

    c = C()
    Test().dispatch(c)
    assert c.called == 'a'


def test_dispatch_cls_inheritance():
    class A(object):
        def __init__(self):
            self.called = False

    class B(object):
        def __init__(self):
            self.called = False

    class C(object):
        def __init__(self):
            self.called = False

    class Test1(object):
        dispatch = DispatchTable()

        @dispatch.register(A)
        def process_a(self, obj):
            obj.called = 'test1.process_a'

        @dispatch.register(B)
        def process_b(self, obj):
            obj.called = 'test1.process_b'

    class Test2(Test1):
        dispatch = DispatchTable(Test1.dispatch)

        @dispatch.register(B)
        def process_b(self, obj):
            obj.called = 'test2.process_b'

        @dispatch.register(C)
        def process_c(self, obj):
            obj.called = 'test2.process_c'

    a = A()
    b = B()
    Test1().dispatch(a)
    assert a.called == 'test1.process_a'
    Test1().dispatch(b)
    assert b.called == 'test1.process_b'

    a = A()
    b = B()
    c = C()
    Test2().dispatch(a)
    assert a.called == 'test1.process_a'
    Test2().dispatch(b)
    assert b.called == 'test2.process_b'
    Test2().dispatch(c)
    assert c.called == 'test2.process_c'


def test_dispatch_instance_specific():
    class A(object):
        def __init__(self):
            self.called = False

    class B(object):
        def __init__(self):
            self.called = False

    class Test(object):
        dispatch = DispatchTable()

        @dispatch.register(A)
        def process_a(self, obj):
            obj.called = 'test.process_a'

    def proc_inst(obj):
        obj.called = 'proc_inst'

    test = Test()
    test2 = Test()
    test.dispatch.register(A, proc_inst)
    test.dispatch.register(B, proc_inst)

    a = A()
    test.dispatch(a)
    assert a.called == 'proc_inst'
    test2.dispatch(a)
    assert a.called == 'test.process_a'

    b = B()
    test.dispatch(b)
    assert b.called == 'proc_inst'


def test_dispatch_errors():
    class Test(object):
        dispatch = DispatchTable()

    with pytest.raises(NotImplementedError):
        Test().dispatch(object())


def test_inspective_labeler():
    with nengo.Network() as model:
        ens = nengo.Ensemble(10, 1)
        with nengo.Network() as subnet:
            node = nengo.Node(1.)
            subnet.attr = nengo.Ensemble(10, 1)

    labels = InspectiveLabeler().get_labels(model)
    assert labels == {
        ens: 'ens',
        subnet: 'subnet',
        node: 'subnet.node',
        subnet.attr: 'subnet.attr',
    }


def test_gexf_converter():
    with nengo.Network() as model:
        ens = nengo.Ensemble(10, 1)
        with nengo.Network() as subnet:
            node = nengo.Node(1.)
            subnet.attr = nengo.Ensemble(10, 1)
        nengo.Connection(node, ens)
        nengo.Probe(ens)

    expected = (
        '<gexf version="1.3" '
        'xmlns="http://www.gexf.net/1.3draft" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xsi:schemaLocation="http://www.gexf.net/1.3draft '
        'http://www.gexf.net/1.3draft/gexf.xsd">'
        '<meta lastmodifieddate="\\d+-\\d{2}-\\d{2}">'
        '<creator>nengo_extras.gexf.GexfConverter</creator>'
        '</meta>'
        '<graph defaultedgetype="directed">'
        '<attributes class="node">'
        '<attribute id="0" title="type" type="string" />'
        '<attribute id="1" title="net" type="long" />'
        '<attribute id="2" title="net_label" type="string" />'
        '<attribute id="3" title="size_in" type="integer" />'
        '<attribute id="4" title="size_out" type="integer" />'
        '<attribute id="5" title="radius" type="float" />'
        '<attribute id="6" title="n_neurons" type="integer">'
        '<default>0</default>'
        '</attribute>'
        '<attribute id="7" title="neuron_type" type="string" />'
        '</attributes>'
        '<attributes class="edge">'
        '<attribute id="0" title="pre_type" type="string" />'
        '<attribute id="1" title="post_type" type="string" />'
        '<attribute id="2" title="synapse" type="string" />'
        '<attribute id="3" title="tau" type="float" />'
        '<attribute id="4" title="function" type="string" />'
        '<attribute id="5" title="transform" type="string" />'
        '<attribute id="6" title="scalar_transform" type="float">'
        '<default>1.0</default>'
        '</attribute>'
        '<attribute id="7" title="learning_rule_type" type="string" />'
        '</attributes>'
        '<nodes>'
        '<node id="\\d+" label="ens">'
        '<attvalues>'
        '<attvalue for="0" value="nengo.ensemble.Ensemble" />'
        '<attvalue for="1" value="\\d+" />'
        '<attvalue for="2" value="model" />'
        '<attvalue for="3" value="1" />'
        '<attvalue for="4" value="1" />'
        '<attvalue for="5" value="1.0" />'
        '<attvalue for="6" value="10" />'
        '<attvalue for="7" value="LIF\\(\\)" />'
        '</attvalues>'
        '</node>'
        '<node id="\\d+" label="subnet.attr">'
        '<attvalues>'
        '<attvalue for="0" value="nengo.ensemble.Ensemble" />'
        '<attvalue for="1" value="\\d+" />'
        '<attvalue for="2" value="subnet" />'
        '<attvalue for="3" value="1" />'
        '<attvalue for="4" value="1" />'
        '<attvalue for="5" value="1.0" />'
        '<attvalue for="6" value="10" />'
        '<attvalue for="7" value="LIF\\(\\)" />'
        '</attvalues>'
        '</node>'
        '<node id="\\d+" label="subnet.node">'
        '<attvalues>'
        '<attvalue for="0" value="nengo.node.Node" />'
        '<attvalue for="1" value="\\d+" />'
        '<attvalue for="2" value="subnet" />'
        '<attvalue for="3" value="0" />'
        '<attvalue for="4" value="1" />'
        '</attvalues>'
        '</node>'
        '</nodes>'
        '<edges>'
        '<edge id="\\d+" source="\\d+" target="\\d+">'
        '<attvalues>'
        '<attvalue for="0" value="nengo.node.Node" />'
        '<attvalue for="1" value="nengo.ensemble.Ensemble" />'
        '<attvalue for="2" value="Lowpass\\(0.005\\)" />'
        '<attvalue for="3" value="0.005" />'
        '<attvalue for="5" value="1.0" />'
        '</attvalues>'
        '</edge>'
        '</edges>'
        '</graph>'
        '</gexf>$')

    actual = et.tostring(
        GexfConverter().convert(model).getroot(), encoding='unicode')
    assert re.match(expected, actual), actual


def test_gexf_converter_hierarchical():
    with nengo.Network() as model:
        ens = nengo.Ensemble(10, 1)
        with nengo.Network() as subnet:
            node = nengo.Node(1.)
            subnet.attr = nengo.Ensemble(10, 1)
        nengo.Connection(node, ens)
        nengo.Probe(ens)

    expected = (
        '<gexf version="1.3" '
        'xmlns="http://www.gexf.net/1.3draft" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xsi:schemaLocation="http://www.gexf.net/1.3draft '
        'http://www.gexf.net/1.3draft/gexf.xsd">'
        '<meta lastmodifieddate="\\d+-\\d{2}-\\d{2}">'
        '<creator>nengo_extras.gexf.GexfConverter</creator>'
        '</meta>'
        '<graph defaultedgetype="directed">'
        '<attributes class="node">'
        '<attribute id="0" title="type" type="string" />'
        '<attribute id="1" title="net" type="long" />'
        '<attribute id="2" title="net_label" type="string" />'
        '<attribute id="3" title="size_in" type="integer" />'
        '<attribute id="4" title="size_out" type="integer" />'
        '<attribute id="5" title="radius" type="float" />'
        '<attribute id="6" title="n_neurons" type="integer">'
        '<default>0</default>'
        '</attribute>'
        '<attribute id="7" title="neuron_type" type="string" />'
        '</attributes>'
        '<attributes class="edge">'
        '<attribute id="0" title="pre_type" type="string" />'
        '<attribute id="1" title="post_type" type="string" />'
        '<attribute id="2" title="synapse" type="string" />'
        '<attribute id="3" title="tau" type="float" />'
        '<attribute id="4" title="function" type="string" />'
        '<attribute id="5" title="transform" type="string" />'
        '<attribute id="6" title="scalar_transform" type="float">'
        '<default>1.0</default>'
        '</attribute>'
        '<attribute id="7" title="learning_rule_type" type="string" />'
        '</attributes>'
        '<nodes>'
        '<node id="\\d+" label="ens">'
        '<attvalues>'
        '<attvalue for="0" value="nengo.ensemble.Ensemble" />'
        '<attvalue for="1" value="\\d+" />'
        '<attvalue for="2" value="model" />'
        '<attvalue for="3" value="1" />'
        '<attvalue for="4" value="1" />'
        '<attvalue for="5" value="1.0" />'
        '<attvalue for="6" value="10" />'
        '<attvalue for="7" value="LIF\\(\\)" />'
        '</attvalues>'
        '</node>'
        '<node id="\\d+" label="subnet">'
        '<attvalues>'
        '<attvalue for="0" value="nengo.network.Network" />'
        '<attvalue for="1" value="\\d+" />'
        '<attvalue for="2" value="model" />'
        '<attvalue for="6" value="10" />'
        '</attvalues>'
        '<nodes>'
        '<node id="\\d+" label="subnet.attr">'
        '<attvalues>'
        '<attvalue for="0" value="nengo.ensemble.Ensemble" />'
        '<attvalue for="1" value="\\d+" />'
        '<attvalue for="2" value="subnet" />'
        '<attvalue for="3" value="1" />'
        '<attvalue for="4" value="1" />'
        '<attvalue for="5" value="1.0" />'
        '<attvalue for="6" value="10" />'
        '<attvalue for="7" value="LIF\\(\\)" />'
        '</attvalues>'
        '</node>'
        '<node id="\\d+" label="subnet.node">'
        '<attvalues>'
        '<attvalue for="0" value="nengo.node.Node" />'
        '<attvalue for="1" value="\\d+" />'
        '<attvalue for="2" value="subnet" />'
        '<attvalue for="3" value="0" />'
        '<attvalue for="4" value="1" />'
        '</attvalues>'
        '</node>'
        '</nodes>'
        '</node>'
        '</nodes>'
        '<edges>'
        '<edge id="\\d+" source="\\d+" target="\\d+">'
        '<attvalues>'
        '<attvalue for="0" value="nengo.node.Node" />'
        '<attvalue for="1" value="nengo.ensemble.Ensemble" />'
        '<attvalue for="2" value="Lowpass\\(0.005\\)" />'
        '<attvalue for="3" value="0.005" />'
        '<attvalue for="5" value="1.0" />'
        '</attvalues>'
        '</edge>'
        '</edges>'
        '</graph>'
        '</gexf>$')

    actual = et.tostring(
        GexfConverter(hierarchical=True).convert(model).getroot(),
        encoding='unicode')
    assert re.match(expected, actual), actual


def test_collapsing_gexf_converter():
    with nengo.Network() as model:
        ea = nengo.networks.EnsembleArray(10, 10)

    assert ea  # hide pylint unused warning

    expected = (
        '<gexf version="1.3" '
        'xmlns="http://www.gexf.net/1.3draft" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xsi:schemaLocation="http://www.gexf.net/1.3draft '
        'http://www.gexf.net/1.3draft/gexf.xsd">'
        '<meta lastmodifieddate="\\d+-\\d{2}-\\d{2}">'
        '<creator>nengo_extras.gexf.CollapsingGexfConverter</creator>'
        '</meta>'
        '<graph defaultedgetype="directed">'
        '<attributes class="node">'
        '<attribute id="0" title="type" type="string" />'
        '<attribute id="1" title="net" type="long" />'
        '<attribute id="2" title="net_label" type="string" />'
        '<attribute id="3" title="size_in" type="integer" />'
        '<attribute id="4" title="size_out" type="integer" />'
        '<attribute id="5" title="radius" type="float" />'
        '<attribute id="6" title="n_neurons" type="integer">'
        '<default>0</default>'
        '</attribute>'
        '<attribute id="7" title="neuron_type" type="string" />'
        '</attributes>'
        '<attributes class="edge">'
        '<attribute id="0" title="pre_type" type="string" />'
        '<attribute id="1" title="post_type" type="string" />'
        '<attribute id="2" title="synapse" type="string" />'
        '<attribute id="3" title="tau" type="float" />'
        '<attribute id="4" title="function" type="string" />'
        '<attribute id="5" title="transform" type="string" />'
        '<attribute id="6" title="scalar_transform" type="float">'
        '<default>1.0</default>'
        '</attribute>'
        '<attribute id="7" title="learning_rule_type" type="string" />'
        '</attributes>'
        '<nodes>'
        '<node id="\\d+" label="ea">'
        '<attvalues>'
        '<attvalue for="0" value="'
        'nengo.networks.ensemblearray.EnsembleArray" />'
        '<attvalue for="1" value="\\d+" />'
        '<attvalue for="2" value="model" />'
        '<attvalue for="6" value="100" />'
        '</attvalues>'
        '</node>'
        '</nodes>'
        '<edges>'
        '.*'
        '</edges>'
        '</graph>'
        '</gexf>$')

    actual = et.tostring(
        CollapsingGexfConverter().convert(model).getroot(), encoding='unicode')
    assert re.match(expected, actual), actual
