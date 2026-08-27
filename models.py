import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import Tensor
from tqdm import tqdm
from vllm import LLM, SamplingParams


class ModelHuggingFace:
    def __init__(self, model_name, cache_dir, hf_token, for_resid=False):
        with open("model_configs.json", "r") as f:
            config = json.load(f)
        torch.manual_seed(42)

        self.device = torch.device("cuda")
        self.model_name = model_name

        if self.model_name in config["model_names"]:
            self.model_path = config["model_names"][self.model_name]["model_path"]
            gen_config = config["model_names"][self.model_name]["gen_config"]
            self.system_prompt = config["gen_configs"][gen_config]["system_prompt"]
            self.sampling_params = config["gen_configs"][gen_config]["sampling_params"]
            self.vllm = config["gen_configs"][gen_config]["vllm"]
        else:
            raise ValueError(
                f"Unknown model name '{model_name}'. "
                f"Available models: {list(config['model_names'].keys())}"
            )

        if for_resid:
            # don't use vllm for residual collection
            self.vllm = False

        if self.vllm:
            tp = min(torch.cuda.device_count(), 4)
            self.model = LLM(
                model=self.model_path,
                dtype="bfloat16",
                trust_remote_code=True,
                download_dir=cache_dir,
                hf_token=hf_token,
                tensor_parallel_size=tp,
                max_model_len=2000,
            )
        else:
            model_kwargs = {"cache_dir": cache_dir}
            model_kwargs["dtype"] = torch.bfloat16

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                token=hf_token,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                **model_kwargs,
            ).eval()

        tok_kwargs = {"cache_dir": cache_dir}
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, token=hf_token, trust_remote_code=True, **tok_kwargs
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def format_and_tokenize_prompt_with_pos(self, prompt):
        first_user_token_position = None
        conv = [{"role": "user", "content": prompt}]
        if self.system_prompt != "":
            conv = [{"role": "system", "content": self.system_prompt}] + conv
        prompt_formatted = self.tokenizer.apply_chat_template(
            conv,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        # find start: tokenize everything except user content to get system/template length
        conv_without_user = (
            [{"role": "system", "content": self.system_prompt}]
            if self.system_prompt != ""
            else []
        )
        dummy_ids = self.tokenizer.encode("DUMMY", add_special_tokens=False)
        prefix_ids = self.tokenizer.apply_chat_template(
            conv_without_user + [{"role": "user", "content": "DUMMY"}],
            tokenize=True,
            add_generation_prompt=False,
        )
        # find where the dummy tokens actually appear in the full sequence
        full_ids = prefix_ids  # just for naming clarity
        for i in range(len(full_ids) - len(dummy_ids), -1, -1):
            if full_ids[i : i + len(dummy_ids)] == dummy_ids:
                first_user_token_position = i
                break

        # find end: last token of user content
        prompt_formatted_no_special = self.tokenizer(
            prompt, add_special_tokens=False
        ).input_ids
        last_inst_token = prompt_formatted_no_special[-1]
        final_user_token_position = -1
        try:
            positions = (prompt_formatted.input_ids[0] == last_inst_token).nonzero(
                as_tuple=True
            )[0]
            final_user_token_position = positions[-1].item()
        except IndexError:
            print("index error: using alternative logic to extract final position")
            # handle case that prompt gets tokenized differently with chat template
            user_ids = prompt_formatted_no_special[:-1]
            full_ids = prompt_formatted.input_ids[0].tolist()
            # find last occurence of user_ids in full_ids
            for i in range(len(full_ids) - len(user_ids), -1, -1):
                if full_ids[i : i + len(user_ids)] == user_ids:
                    final_user_token_position = i + len(user_ids)
                    break

        prompt_formatted = prompt_formatted.to(self.device)
        # print(
        #    f"first user token: {self.tokenizer.decode(prompt_formatted.input_ids[0, first_user_token_position])}"
        # )
        # print(
        #    f"last user token: {self.tokenizer.decode(prompt_formatted.input_ids[0, final_user_token_position])}"
        # )
        return prompt_formatted, first_user_token_position, final_user_token_position

    def format_and_tokenize_prompt(self, prompt):
        prompt_formatted, _ = self.format_prompt_as_text(prompt)
        inputs = self.tokenizer(
            prompt_formatted, return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        return inputs

    def format_prompt_as_text(self, prompt):
        conv = [{"role": "user", "content": prompt}]
        if self.system_prompt != "":
            conv = [{"role": "system", "content": self.system_prompt}] + conv
        prompt_formatted = self.tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True
        )
        return prompt_formatted, conv

    def get_response_batched(self, prompt, generation_config):
        if self.vllm:
            _, messages = self.format_prompt_as_text(prompt)
            batched_messages = [messages] * 10
            sampling_params = SamplingParams(**self.sampling_params)
            outputs = self.model.chat(
                messages=batched_messages, sampling_params=sampling_params
            )
            responses = [out.outputs[0].text for out in outputs]
        else:
            inputs = self.format_and_tokenize_prompt(prompt)
            prompt_length = inputs["input_ids"].shape[1]
            batched_inputs = {k: v.repeat(10, 1) for k, v in inputs.items()}
            with torch.inference_mode():
                outputs = self.model.generate(
                    **batched_inputs,
                    **generation_config,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            responses = self.tokenizer.batch_decode(
                outputs[:, prompt_length:], skip_special_tokens=True
            )
        return responses

    def get_response(self, prompt, generation_config):
        if self.vllm:
            _, messages = self.format_prompt_as_text(prompt)
            sampling_params = SamplingParams(**self.sampling_params)
            outputs = self.model.chat(messages, sampling_params)
            response = outputs[0].outputs[0].text

        else:
            inputs = self.format_and_tokenize_prompt(prompt)
            with torch.inference_mode():
                outputs = self.model.generate(
                    **generation_config,
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            outputs_truncated = outputs[0][len(inputs["input_ids"][0]) :]
            response = self.tokenizer.decode(
                outputs_truncated, skip_special_tokens=True
            )

        return response

    def get_resids(self, prompts, location):

        try:
            n_layers = self.model.config.num_hidden_layers
        except AttributeError:
            n_layers = self.model.config.text_config.num_hidden_layers

        cache = {layer: [] for layer in range(n_layers)}

        try:
            block_modules = self.model.model.layers
        except AttributeError:
            try:
                block_modules = self.model.language_model.layers
            except AttributeError:
                block_modules = self.model.model.language_model.layers

        if location == "final":
            position = -1

            hooks = []
            for layer_idx in range(n_layers):
                hooks.append(
                    block_modules[layer_idx].register_forward_hook(
                        make_hook(layer_idx, cache, position)
                    )
                )

            for prompt in tqdm(prompts):
                inputs = self.format_and_tokenize_prompt(prompt)
                with torch.no_grad():
                    self.model(**inputs)

            # remove hooks
            for h in hooks:
                h.remove()

        elif location == "last_prompt":
            # location = last prompt token, specific for every prompt
            for prompt in tqdm(prompts):
                (
                    inputs,
                    _,
                    final_user_token_position,
                ) = self.format_and_tokenize_prompt_with_pos(prompt)
                hooks = []
                for layer_idx in range(n_layers):
                    hooks.append(
                        block_modules[layer_idx].register_forward_hook(
                            make_hook(layer_idx, cache, final_user_token_position)
                        )
                    )
                with torch.no_grad():
                    self.model(**inputs)
                for h in hooks:
                    h.remove()

        elif location == "mean":
            for prompt in tqdm(prompts):
                (
                    inputs,
                    first_user_token_position,
                    final_user_token_position,
                ) = self.format_and_tokenize_prompt_with_pos(prompt)
                hooks = []
                for layer_idx in range(n_layers):
                    hooks.append(
                        block_modules[layer_idx].register_forward_hook(
                            make_hook_mean_pool(
                                layer_idx,
                                cache,
                                first_user_token_position,
                                final_user_token_position,
                            )
                        )
                    )

                with torch.no_grad():
                    self.model(**inputs)

                for h in hooks:
                    h.remove()

        else:
            raise ValueError(
                f"Unknown location '{location}'. "
                "Expected one of: 'final', 'last_prompt', 'mean'."
            )

        # convert lists to tensors
        layerwise = {}
        for layer_idx in range(n_layers):
            # (n_prompts, n_positions, d_model)
            layerwise[layer_idx] = torch.cat(cache[layer_idx], dim=0)

        return layerwise


def make_hook(layer_idx: int, cache: dict, position: int):
    """
    Returns a forward hook that stores the residual stream (output of the block)
    for the specified positions.
    """

    def hook_fn(module, input, output):
        hidden_states: Tensor = (
            output[0].detach() if isinstance(output, tuple) else output.detach()
        )
        cache[layer_idx].append(hidden_states[:, position, :].clone())

    return hook_fn


def make_hook_mean_pool(layer_idx: int, cache: dict, start: int, end: int):
    """
    Forward hook that captures the mean of the residual stream
    across user content token positions [start:end+1] only.
    """

    def hook_fn(module, input, output):
        hidden_states: Tensor = (
            output[0].detach() if isinstance(output, tuple) else output.detach()
        )
        # slice to user tokens only, then mean pool -> (1, d_model)
        user_hidden = hidden_states[:, start : end + 1, :]
        cache[layer_idx].append(user_hidden.mean(dim=1).clone())

    return hook_fn
