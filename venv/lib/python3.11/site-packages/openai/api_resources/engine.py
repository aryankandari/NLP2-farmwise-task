from __future__ import absolute_import, division, print_function

from openai import api_requestor, util
from openai.api_resources.abstract import (
    APIResource,
    CreateableAPIResource,
    DeletableAPIResource,
    ListableAPIResource,
    UpdateableAPIResource,
)


class Engine(ListableAPIResource):
    OBJECT_NAME = "engine"

    def generate(self, **params):
        return self.request(
            "post",
            self.instance_url() + "/generate",
            params,
            stream=params.get("stream"),
            plain_old_data=True,
        )

    def search(self, **params):
        return self.request("post", self.instance_url() + "/search", params)
