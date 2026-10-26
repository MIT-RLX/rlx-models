# rlx-gemini-nano

Gemini Nano, the on-device model Chrome uses for its built-in AI APIs.

Google has not published Gemini Nano weights. The only copy is the TFLite file
Chrome downloads to your machine:

| OS | Location |
|---|---|
| macOS | `~/Library/Application Support/Google/Chrome/OptGuideOnDeviceModel/<version>/weights.bin` |
| Linux | `~/.config/google-chrome/OptGuideOnDeviceModel/<version>/weights.bin` |
| Windows | `%LOCALAPPDATA%\Google\Chrome\User Data\OptGuideOnDeviceModel\<version>\weights.bin` |

To get it: in `chrome://flags` set *Enables optimization guide on device* to
*Enabled BypassPerfRequirement* and *Prompt API for Gemini Nano* to *Enabled*,
relaunch, then wait until `chrome://on-device-internals` shows the model as
ready (about 4 GB).

## Status

| Part | State |
|---|---|
| TFLite container (`TFL3`): tensors, quantization, inline + >2 GiB external buffers, op codes, metadata | done, tested against the upstream `flatbuffers` encoder |
| Architecture metadata (schema-less protobuf decode) | done |
| Weight location (`locate`) on macOS / Linux / Windows | done |
| Transformer graph, tokenizer, generation | **not started** — needs the tensor manifest of a real `weights.bin` |

## Usage

```bash
cargo run -p rlx-gemini-nano --release -- locate
cargo run -p rlx-gemini-nano --release -- inspect                 # newest local weights.bin
cargo run -p rlx-gemini-nano --release -- inspect PATH --tensors 2000
cargo run -p rlx-gemini-nano --release -- inspect PATH --json > manifest.json
```

`inspect` prints the metadata entries (decoding protobuf ones field by field),
an op histogram and every tensor's name, dtype, shape, stored size and
quantization layout. Files are memory-mapped; nothing is copied.
