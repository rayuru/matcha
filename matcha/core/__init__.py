from typing import Optional, Callable, Dict


class Factory:

    _registry = None

    @classmethod
    def register(cls, name: str) -> Callable:
        def inner_func(wrapped_class):
            if cls._registry is None:
                cls._registry = {}
            if name in cls._registry:
                raise KeyError(f"{name} already used by {cls._registry[name].__name__}, "
                               f"please use another name.")

            cls._registry[name] = wrapped_class
            return wrapped_class
        return inner_func

    @classmethod
    def create_type(cls, name: str) -> type:
        if cls._registry is None or name not in cls._registry:
            raise KeyError(f"{name} is not registered.")
        return cls._registry[name]

    @classmethod
    def create_object(cls, name: str, init_params: Optional[Dict] = None) -> object:
        created_type = cls.create_type(name)
        if init_params is None:
            init_params = {}
        return created_type(**init_params)
