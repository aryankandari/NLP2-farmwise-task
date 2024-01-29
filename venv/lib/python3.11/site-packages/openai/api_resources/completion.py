from openai.api_resources.abstract.engine_api_resource import EngineAPIResource
from openai.api_resources.abstract import ListableAPIResource, DeletableAPIResource


class Completion(EngineAPIResource, ListableAPIResource, DeletableAPIResource):
    engine_required = True
    OBJECT_NAME = "completion"
