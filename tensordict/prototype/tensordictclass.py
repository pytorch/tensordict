from dataclasses import dataclass, make_dataclass

import torch

from tensordict.tensordict import TensorDict, TensorDictBase


def tensordictclass(cls):
    name = cls.__name__
    datacls = make_dataclass(
        name, bases=(dataclass(cls),), fields=[("batch_size", torch.Size)]
    )

    class TensorDictClass(datacls):
        def __init__(self, *args, **kwargs):

            datacls.__init__(self, *args, **kwargs)

            # should we remove?
            if "batch_size" not in self.__dict__:
                raise Exception("Attribute batch_size is required for TensorDictClass.")

            attributes = [key for key in self.__dict__ if key != "batch_size"]

            for attr in attributes:
                if attr in dir(TensorDict):
                    raise Exception(
                        f"Attribute name {attr} can't be used for TensorDictClass"
                    )

            self.tensordict = TensorDict(
                {attr: getattr(self, attr, None) for attr in attributes},
                batch_size=getattr(self, "batch_size", None),
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
                        if isinstance(res, TensorDictBase):
                            new = TensorDictClass(
                                **res,
                                batch_size=res.batch_size,
                            )  # device=res.device)
                            return new
                        else:
                            return res

                    return wrapped_func

        def __getitem__(self, item):
            res = self.tensordict[item]
            return TensorDictClass(
                **res,
                batch_size=res.batch_size,
            )  # device=res.device)

    return TensorDictClass
