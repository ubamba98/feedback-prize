# Feedback Prize - Evaluating Student Writing

This is the solution for 2nd rank in Kaggle competition: Feedback Prize - Evaluating Student Writing. The competition can be found here: https://www.kaggle.com/competitions/feedback-prize-2021/

## Datasets required

* Download competition data from https://www.kaggle.com/competitions/feedback-prize-2021/data and extract it to ```../input/feedback-prize-2021/```
* Download folds from https://www.kaggle.com/datasets/ubamba98/feedbackgroupshufflesplit1337 and extract it to ```../input/feedbackgroupshufflesplit1337/groupshufflesplit_1337.p```
* Download and convert LSG Roberta from https://github.com/ccdv-ai/convert_checkpoint_to_lsg/blob/main/convert_roberta_checkpoint.py and place it in ```../input/lsg-roberta-large```

Use this command to convert roberta-large to LSG
```bash
$ python convert_roberta_checkpoint.py \
                        --initial_model roberta-large \
                        --model_name lsg-roberta-large \
                        --max_sequence_length 1536
```
* Download fast tokenizer for Deberta V2/V3 from https://www.kaggle.com/datasets/nbroad/deberta-v2-3-fast-tokenizer and extract it to ```../input/deberta-v2-3-fast-tokenizer/```

Follow following instructions to manually add fast tokenizer to transformer library:

```python
# The following is necessary if you want to use the fast tokenizer for deberta v2 or v3
# This must be done before importing transformers
import shutil
from pathlib import Path

# Path to installed transformer library
transformers_path = Path("/opt/conda/lib/python3.7/site-packages/transformers")

input_dir = Path("../input/deberta-v2-3-fast-tokenizer")

convert_file = input_dir / "convert_slow_tokenizer.py"
conversion_path = transformers_path/convert_file.name

if conversion_path.exists():
    conversion_path.unlink()

shutil.copy(convert_file, transformers_path)
deberta_v2_path = transformers_path / "models" / "deberta_v2"

for filename in ['tokenization_deberta_v2.py', 'tokenization_deberta_v2_fast.py']:
    filepath = deberta_v2_path/filename
    if filepath.exists():
        filepath.unlink()

    shutil.copy(input_dir/filename, filepath)
```

After this ```../input``` directory should look something like this.

```
.
├── input
│   ├── feedback-prize-2021
│   │   ├── train/
│   │   ├── test/
│   │   ├── sample_submission.csv
│   │   └── train.csv
│   ├── lsg-roberta-large
│   │   ├── config.json
│   │   ├── merges.txt
│   │   ├── modeling.py
│   │   ├── pytorch_model.bin
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.json
│   ├── deberta-v2-3-fast-tokenizer
│   │   ├── convert_slow_tokenizer.py
│   │   ├── deberta__init__.py
│   │   ├── tokenization_auto.py
│   │   ├── tokenization_deberta_v2.py
│   │   ├── tokenization_deberta_v2_fast.py
│   │   └── transformers__init__.py
│   └── feedbackgroupshufflesplit1337
│       └── groupshufflesplit_1337.p
```

or you can change the ```DATA_BASE_DIR``` in ```SETTINGS.json``` to download the files in your desired location.

## Models and Training

* Deberta large, Deberta xlarge, Deberta v2 xlarge, Deberta v3 large, Funnel transformer large and BigBird are trained using __trainer.py__

Example: 
```bash
$ python trainer.py --fold 0 --pretrained_model google/bigbird-roberta-large
```
where pretrained_model can be ```microsoft/deberta-large```, ```microsoft/deberta-xlarge```, ```microsoft/deberta-v2-xlarge```, ```microsoft/deberta-v3-large```, ```funnel-transformer/large``` or ```google/bigbird-roberta-large```

* Deberta large with LSTM head and jaccard loss is trained using __debertabilstm_trainer.py__

Example: 
```bash
$ python debertabilstm_trainer.py --fold 0
```

* Longformer large with LSTM head is trained using __longformerwithbilstm_trainer.py__

Example: 
```bash
$ python longformerwithbilstm_trainer.py --fold 0
```

* LSG Roberta is trained with __lsgroberta_trainer.py__

Example: 
```bash
$ python lsgroberta_trainer.py --fold 0
```

* YOSO is trained with __yoso_trainer.py__

Example: 
```bash
$ python yoso_trainer.py --fold 0
```

## Inference

After training all the models, the outputs were pushed to Kaggle Datasets.

And the final inference kernel can be found here: https://www.kaggle.com/code/cdeotte/2nd-place-solution-cv741-public727-private740?scriptVersionId=90301836


Solution writeup: https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313389
