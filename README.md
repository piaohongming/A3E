# A3E Usage

- This version includes an initial A3E implementation and testing pipeline only for reference. The full version is on-going.

## 1) Entry and dependencies

- Entry script: `easyeditor/editors/my_analysis_all.py`
- Config file: `hparams/A3E/llama3-8b.yaml`

```bash
pip install torch transformers omegaconf pyyaml numpy
```

## 2) Edit these fields first

Edit `hparams/A3E/llama3-8b.yaml`:

- `model_name`
- `model.name`
- `device` / `model_parallel`
- `model.target_modules`
- `model.grace_layer`

Edit `my_analysis_all.py`:

```python
write_path = "/your/output/result.json"
data_path = "/your/data/xxx.json"
memory_data_path = "/your/data/benchmark_ZsRE_ZsRE-test-all.json"
```

`data_path` keyword decides the pipeline branch:
`counterfact` / `multi (MAC)` / `port` / `dc (MQC)` / `sequential`.

## 3) Run

```bash
cd /home/hmpiao/EasyEdit_method
python -m easyeditor.editors.my_analysis_all
```

## 4) Minimal code example

```python
from easyeditor.models import A3EHyperParams
from easyeditor.editors import BaseEditor

hparams = A3EHyperParams.from_hparams("hparams/A3E/llama3-8b.yaml")
editor = BaseEditor.from_hparams(hparams)

edited_model, label_tok = editor.simple_edit(
    prompts=["The capital of France is"],
    target_new=["Berlin"],
    subject_tok=0,
    relation_tok=3,
    answer_tok=[],
    subject=["France"],
    success_answers=[],
)

editor.model = edited_model
```

## 5) Output

The script writes a JSON file to `write_path`, including pre/post outputs and
`success_prob` / `false_prob` statistics.

