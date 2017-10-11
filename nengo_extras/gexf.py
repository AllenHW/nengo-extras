from collections import namedtuple, OrderedDict
from datetime import date
import gc
import weakref
import xml.etree.ElementTree as et

import nengo
try:
    import nengo_spa as spa
except ImportError:
    spa = None
import numpy as np


class DispatchTable(object):
    class InstDispatch(object):
        __slots__ = ('param', 'inst', 'owner')

        def __init__(self, param, inst, owner):
            self.param = param
            self.inst = inst
            self.owner = owner

        def __call__(self, obj):
            for cls in obj.__class__.__mro__:
                if cls in self.param.inst_type_table.get(self.inst, {}):
                    return self.param.inst_type_table[self.inst][cls](obj)
                elif cls in self.param.type_table:
                    return self.param.type_table[cls](self.inst, obj)
                elif self.param.parent is not None:
                    try:
                        return self.param.parent.__get__(
                            self.inst, self.owner)(obj)
                    except NotImplementedError:
                        pass
            raise NotImplementedError(
                "Nothing to dispatch to for type {}.".format(type(obj)))

        def register(self, type_, fn):
            if self.inst not in self.param.inst_type_table:
                self.param.inst_type_table[self.inst] = (
                    weakref.WeakKeyDictionary())
            table = self.param.inst_type_table[self.inst]
            table[type_] = fn
            return fn

    def __init__(self, parent=None):
        self.type_table = weakref.WeakKeyDictionary()
        self.inst_type_table = weakref.WeakKeyDictionary()
        self.parent = parent

    def register(self, type_):
        def _register(fn):
            assert type_ not in self.type_table
            self.type_table[type_] = fn
            return fn
        return _register

    def __get__(self, inst, owner):
        if inst is None:
            return self
        return self.InstDispatch(self, inst, owner)


class InspectiveLabeler(object):
    dispatch = DispatchTable()

    @classmethod
    def _find_name_in_dict(cls, d, obj):
        for k, v in d.items():
            if not isinstance(k, str):
                break
            if v is obj:
                return k
        return None

    @classmethod
    def _find_name_in_frame(cls, f, obj):
        name = None
        if f.f_back is not None:
            name = cls._find_name_in_frame(f.f_back, obj)
        if name is None:
            name = cls._find_name_in_dict(f.f_locals, obj)
        if name is None:
            name = cls._find_name_in_dict(f.f_globals, obj)
        return name

    @classmethod
    def _is_frame(cls, f):
        return (
            hasattr(f, 'f_back') and
            hasattr(f, 'f_locals') and
            hasattr(f, 'f_globals')
        )

    @dispatch.register(nengo.base.NengoObject)
    def get_obj_label(self, obj):
        name = None
        if obj.label is not None:
            name = obj.label
        else:
            for referrer in gc.get_referrers(obj):
                if hasattr(referrer, 'items'):
                    new_name = self._find_name_in_dict(referrer, obj)
                    if new_name is not None:
                        name = new_name
                elif name is None and self._is_frame(referrer):
                    name = self._find_name_in_frame(referrer, obj)
        if name is None:
            name = str(obj)
        return {obj: name}

    @dispatch.register(nengo.Network)
    def get_net_labels(self, net):
        labels = self.get_obj_label(net)
        labels.update({
            k: labels[net] + '.' + v for k, v in self.get_labels(net).items()})
        return labels

    def get_labels(self, net):
        labels = {}
        for group in net.objects.values():
            for o in group:
                labels.update(self.dispatch(o))
        return labels


Attr = namedtuple('Attr', ['id', 'type', 'default'])


class GexfConverter(object):
    dispatch = DispatchTable()

    node_attrs = OrderedDict((
        ('type', Attr(0, 'string', None)),
        ('net', Attr(1, 'long', None)),
        ('net_label', Attr(2, 'string', None)),
        ('size_in', Attr(3, 'integer', None)),
        ('size_out', Attr(4, 'integer', None)),
        ('radius', Attr(5, 'float', None)),
        ('n_neurons', Attr(6, 'integer', 0)),
        ('neuron_type', Attr(7, 'string', None)),
        ))
    edge_attrs = OrderedDict((
        ('pre_type', Attr(0, 'string', None)),
        ('post_type', Attr(1, 'string', None)),
        ('synapse', Attr(2, 'string', None)),
        ('tau', Attr(3, 'float', None)),
        ('function', Attr(4, 'string', None)),
        ('transform', Attr(5, 'string', None)),
        ('scalar_transform', Attr(6, 'float', 1.)),
        ('learning_rule_type', Attr(7, 'string', None)),
    ))

    def __init__(self, labeler=None, hierarchical=False):
        if labeler is None:
            labeler = InspectiveLabeler()
        self.labeler = labeler
        self.hierarchical = hierarchical
        self.version = (1, 3)
        self.tag = 'draft'

        self._labels = {}
        self._net = None

    def convert(self, model):
        self._labels = weakref.WeakKeyDictionary(
            self.labeler.get_labels(model))
        self._labels.update(self.labeler.get_obj_label(model))
        return self.make_document(model)

    def make_document(self, model):
        version = '.'.join(str(i) for i in self.version)
        tag_version = version + self.tag
        gexf = et.Element('gexf', {
            'xmlns': 'http://www.gexf.net/' + tag_version,
            'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
            'xsi:schemaLocation': (
                'http://www.gexf.net/' + tag_version  + ' ' +
                'http://www.gexf.net/' + tag_version + '/gexf.xsd'),
            'version': version
        })

        meta = et.SubElement(gexf, 'meta', {
            'lastmodifieddate': date.today().isoformat()})
        creator = et.SubElement(meta, 'creator')
        creator.text = self._get_typename(self)

        graph = et.SubElement(gexf, 'graph', {'defaultedgetype': 'directed'})
        graph.append(self.make_attr_defs('node', self.node_attrs))
        graph.append(self.make_attr_defs('edge', self.edge_attrs))

        graph.append(self.dispatch(model))

        edges = et.SubElement(graph, 'edges')
        for c in model.all_connections:
            elem = self.dispatch(c)
            if elem is not None:
                edges.append(elem)

        return et.ElementTree(gexf)

    def make_attr_defs(self, cls, defs):
        attributes = et.Element('attributes', {'class': cls})
        for k, d in defs.items():
            attr = et.SubElement(attributes, 'attribute', {
                'id': str(d.id),
                'title': k,
                'type': d.type,
            })
            if d.default is not None:
                default = et.SubElement(attr, 'default')
                default.text = str(d.default)
        return attributes

    def make_attrs(self, defs, attrs):
        values = et.Element('attvalues')
        assert all(k in defs for k in attrs.keys())
        for k, d in defs.items():
            if k in attrs and attrs[k] is not None:
                values.append(et.Element('attvalue', {
                    'for': str(d.id),
                    'value': str(attrs[k]),
                }))
        return values

    def make_node(self, obj, **attrs):
        tag_attrib = {'id': str(id(obj))}
        if obj in self._labels:
            tag_attrib['label'] = self._labels[obj]
        node = et.Element('node', tag_attrib)
        if len(attrs) > 0:
            node.append(self.make_attrs(self.node_attrs, attrs))
        return node

    def make_edge(self, obj, source, target, **attrs):
        tag_attrib = {
            'id': str(id(obj)),
            'source': str(id(source)),
            'target': str(id(target))
        }
        edge = et.Element('edge', tag_attrib)
        if len(attrs) > 0:
            edge.append(self.make_attrs(self.edge_attrs, attrs))
        return edge

    @dispatch.register(nengo.Ensemble)
    def convert_ensemble(self, ens):
        return self.make_node(
            ens,
            type=self._get_typename(ens),
            net=id(self._net),
            net_label=self._labels.get(self._net, None),
            size_in=ens.dimensions,
            size_out=ens.dimensions,
            radius=ens.radius,
            n_neurons=ens.n_neurons,
            neuron_type=ens.neuron_type,
        )

    @dispatch.register(nengo.Node)
    def convert_node(self, node):
        return self.make_node(
            node,
            type=self._get_typename(node),
            net=id(self._net),
            net_label=self._labels.get(self._net, None),
            size_in=node.size_in,
            size_out=node.size_out,
        )

    @dispatch.register(nengo.Probe)
    def convert_probe(self, probe):
        return None

    @dispatch.register(nengo.Network)
    def convert_network(self, net):
        parent_net = self._net
        self._net = net

        nodes = et.Element('nodes')
        leaves = net.ensembles + net.nodes + net.probes
        for leave in leaves:
            leave_elem = self.dispatch(leave)
            if leave_elem is not None:
                nodes.append(leave_elem)
        if self.hierarchical:
            for subnet in net.networks:
                subnet_node = self.make_node(
                    subnet, type=self._get_typename(subnet), net=id(self._net),
                    net_label=self._labels.get(self._net, None))
                subnet_node.append(self.dispatch(subnet))
                nodes.append(subnet_node)
        else:
            for subnet in net.networks:
                nodes.extend(self.dispatch(subnet))


        self._net = parent_net
        return nodes

    def _get_node_obj(self, obj):
        if isinstance(obj, nengo.ensemble.Neurons):
            return obj.ensemble
        elif isinstance(obj, nengo.connection.LearningRule):
            return self._get_node_obj(obj.connection.pre_obj)
        return obj

    @dispatch.register(nengo.Connection)
    def convert_connection(self, conn):
        source = self._get_node_obj(conn.pre_obj)
        target = self._get_node_obj(conn.post_obj)
        return self.make_edge(
            conn, source, target,
            pre_type=self._get_typename(conn.pre_obj),
            post_type=self._get_typename(conn.post_obj),
            synapse=conn.synapse,
            tau=conn.synapse.tau if hasattr(conn.synapse, 'tau') else None,
            function=conn.function,
            transform=conn.transform,
            scalar_transform=(
                conn.transform if np.isscalar(conn.transform) else None),
            learning_rule_type=conn.learning_rule_type
        )

    @classmethod
    def _get_typename(cls, obj):
        tp = type(obj)
        return tp.__module__ + '.' + tp.__name__


class CollapsingGexfConverter(GexfConverter):
    dispatch = DispatchTable(GexfConverter.dispatch)

    NENGO_NETS = (
        nengo.networks.CircularConvolution,
        nengo.networks.EnsembleArray,
        nengo.networks.Product)
    if spa is None:
        SPA_NETS = ()
    else:
        SPA_NETS = (
            spa.networks.CircularConvolution,
            spa.AssociativeMemory,
            spa.Bind,
            spa.Compare,
            spa.Product,
            spa.Scalar,
            spa.State,
            spa.Transcode)

    def __init__(self, to_collapse=None, labeler=None, hierarchical=False):
        super(CollapsingGexfConverter, self).__init__(
            labeler=labeler, hierarchical=hierarchical)

        if to_collapse is None:
            to_collapse = self.NENGO_NETS + self.SPA_NETS

        for cls in to_collapse:
            self.dispatch.register(cls, self.convert_collapsed)

        self.obj2collapsed = weakref.WeakKeyDictionary()

    def convert_collapsed(self, net):
        nodes = et.Element('nodes')
        nodes.append(self.make_node(
            net, type=self._get_typename(net), net=id(self._net),
            net_label=self._labels.get(self._net, None)))
        self.obj2collapsed.update({
            child: net for child in net.all_objects})
        return nodes

    def _get_node_obj(self, obj):
        obj = super(CollapsingGexfConverter, self)._get_node_obj(obj)
        return self.obj2collapsed.get(obj, obj)
