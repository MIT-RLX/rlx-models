# Fixtures

Small artifacts that let the tests check this crate against the real model
without downloading it. None of them contain weights.

| file | origin | used by |
|---|---|---|
| `real_config.json` | `config.json` from [google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it), verbatim | `tests/real_checkpoint_contract.rs` |
| `chat_template.jinja` | `chat_template.jinja` from the same repo, verbatim | fixture generation for `tests/parity_reference.rs` |
| `real_checkpoint_shapes.json` | derived: tensor name → shape for all 1047 tensors, read out of the eleven safetensors shard **headers** via HTTP Range requests | `tests/real_checkpoint_contract.rs` |

`real_checkpoint_shapes.json` is metadata only — names and dimensions, no tensor
data — which is what makes it possible to verify the loader against the real
25.8 B-parameter checkpoint from a 74 KB file.

The two verbatim files are redistributed from Google's model repository and
remain under that model's terms (Gemma Terms of Use), not this crate's licence.
They are here so the tests are self-contained and offline; drop them (and let
the tests skip) if that is not wanted for a given distribution — see `exclude`
in `Cargo.toml`.

To regenerate the shape map:

```sh
python3 scripts/diffusiongemma_fetch_subset.py --help   # same Range-request trick
```
