def get_model_by_name(name, model_config):
    if name == "RecIDE":
        from ..model.idedeq import RecIDE
        return RecIDE(model_config)
    elif name == "IDEDEQIDO":
        from ..model.idedeq import IDEDEQIDO
        return IDEDEQIDO(model_config)
    elif name == "TinyIDEDEQIDO":
        from ..model.idedeq import TinyIDEDEQIDO
        return TinyIDEDEQIDO(model_config)
    elif name == "NanoIDEDEQIDO":
        from ..model.idedeq import NanoIDEDEQIDO
        return NanoIDEDEQIDO(model_config)
    else:
        raise ValueError("Unknown model name: {}".format(name))
