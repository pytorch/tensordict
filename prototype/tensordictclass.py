from dataclasses import dataclass

from tensordict import TensorDict


def tensordictclass(cls):
    datacls = dataclass(cls)

    class TensorDictClass(datacls):
        def __init__(self, *args, **kwargs):

            datacls.__init__(self, *args, **kwargs)

            if "batch_size" not in self.__dict__:
                raise Exception(f"Attribute batch_size is required for TensorDictClass.")

            attributes = [key for key in self.__dict__ if key != "batch_size"]

            for attr in attributes:
                if attr in dir(TensorDict):
                    raise Exception(f"Attribute name {attr} can't be used for TensorDictClass")

            self.tensordict = TensorDict(
                {attr: getattr(self, attr, None) for attr in attributes},
                batch_size=getattr(self, "batch_size", None)
            )

        def __getattr__(self, attr):
            if attr in self.__dict__:
                return getattr(self, attr)
            else:
                res = getattr(self.tensordict, attr)
                if not callable(res):
                    return res
                else:
                    func = res

                    def wrapped_func(*args, **kwargs):
                        res = func(*args, **kwargs)
                        if isinstance(res, TensorDict):
                            new = TensorDictClass(*[res[key] for key in res.keys()], res.batch_size)
                            return new
                        else:
                            return res

                    return wrapped_func

    return TensorDictClass
