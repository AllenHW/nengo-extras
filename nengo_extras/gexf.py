import gc
import weakref

import nengo


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
                    name = self._find_name_in_dict(referrer, obj)
                elif self._is_frame(referrer):
                    name = self._find_name_in_frame(referrer, obj)
                if name is not None:
                    break
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
