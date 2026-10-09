import json
from pathlib import Path

import triton_python_backend_utils as pb_utils

from wordpiece import WordPieceTokenizer


class TritonPythonModel:
    def initialize(self, args):
        version_dir = Path(args["model_repository"]) / args["model_version"]
        config = json.loads(args["model_config"])
        max_length = int(config.get("parameters", {}).get("max_seq_length", {}).get("string_value", "256"))
        self.tokenizer = WordPieceTokenizer(version_dir / "vocab.txt", max_length=max_length)

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                texts = pb_utils.get_input_tensor_by_name(request, "TEXT").as_numpy().reshape(-1)
                texts = [t.decode("utf-8") if isinstance(t, bytes) else str(t) for t in texts]
                if not texts:
                    raise ValueError("TEXT must contain at least one string")
                ids, mask, types = self.tokenizer.batch(texts)
                responses.append(pb_utils.InferenceResponse(output_tensors=[
                    pb_utils.Tensor("input_ids", ids),
                    pb_utils.Tensor("attention_mask", mask),
                    pb_utils.Tensor("token_type_ids", types),
                ]))
            except Exception as error:
                responses.append(pb_utils.InferenceResponse(error=pb_utils.TritonError(str(error))))
        return responses
