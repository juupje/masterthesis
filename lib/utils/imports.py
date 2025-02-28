import importlib.util as imp_util
import importlib.machinery as imp_mach
def import_mod(name:str, path:str):
    spec = imp_util.spec_from_loader(name, imp_mach.SourceFileLoader(name, path))
    module = imp_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
