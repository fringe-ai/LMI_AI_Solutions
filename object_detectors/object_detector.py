from od_base import ODBase
import json

class ObjectDetector(ODBase):
    _registry = {}

    @classmethod
    def register(cls,metadata):
        """
        Decorator to register a wrapper class for multiple versions, model names, and tasks.
        """
        frameworks = metadata.get('frameworks', None)
        model_names = metadata.get('model_names', None)
        tasks = metadata.get('tasks', None)
        versions = metadata.get('versions', None)
        info = metadata.get('info', None)
        
        def decorator(wrapper_cls):
            assert all([frameworks, model_names, tasks, versions]), "frameworks, model_names, tasks, and versions must be specified."
            for framework in frameworks:
                for model_name in model_names:
                    for task in tasks:
                        for version in versions:
                            key = (framework, model_name, task, version,json.dumps(info))
                            if key in cls._registry:
                                raise ValueError(f"Wrapper already registered for version={version}, model_name='{model_name}', task='{task}', framework='{framework}', info='{json.dumps(info)}'.")
                        cls._registry[key] = wrapper_cls
            return wrapper_cls
        return decorator
    
    def __new__(cls, metadata,*args, **kwargs):
        """
        Override __new__ to return the appropriate wrapper instance based on version, model_name, and task.
        """
        framework = metadata.get('framework', None)
        model_name = metadata.get('model_name', None)
        task = metadata.get('task', None)
        version = metadata.get('version', None)
        info = metadata.get('info', None)
        if not all([framework, model_name, task, version]):
            raise ValueError("Framework, model_name, task, and version must be specified.")
        key = (framework, model_name, task, version, json.dumps(info))
        if key not in cls._registry:
            raise ValueError(f"No class found for version={version}, model_name='{model_name}', task='{task}', framework='{framework}', metadata='{json.dumps(info)}'.")
        wrapper_cls = cls._registry[key]
        return wrapper_cls(*args, **kwargs)

