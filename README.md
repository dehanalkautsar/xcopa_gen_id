# XGID: xcopa_gen_id
## About
This repository contains an experimental setup using a curated modified dataset derived from XCOPA (Ponti et al., 2020), tailored for text generation tasks. The modifications enable the dataset to be utilized effectively for generating coherent and contextually relevant text. This repository is intended for the final task of IF5281 Deep Learning at Institut Teknologi Bandung for the academic year 2023/2024.

## Results
We benchmark the XCOPAGenId/XGID using several T5-based models e.g. `LazarusNLP/IndoNanoT5`, `Wikidepia/IndoT5`, and `google/mT5` on F1 score and BLEU metrics.
| Model                                      | #params | lr   | Macro F1 | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | BLEU | (F1 + BLEU)/2 |
| ------------------------------------------ | :-----: | :--: | :--------: | :----: | :----: | :----: | :----: | :--: | :-----------: |
| LazarusNLP/IndoNanoT5                      |  248M   | 1e-5 |    5.40    | 23.00  |  1.41  |  0.00  |  0.00  | 6.10 |      5.75     |
| LazarusNLP/IndoNanoT5                      |  248M   | 5e-5 |   12.98    | **31.92**  |  2.53  |  0.68  |  0.00  | **8.78** |     **10.88**     |
| LazarusNLP/IndoNanoT5                      |  248M   | 1e-4 |  **13.28** | 29.60  |  **2.70**  |  **0.87**  |  0.00  | 8.29 |     10.79     |
| Wikidepia/IndoT5 (~248M)                   |  248M   | 1e-5 |   10.18    | 20.49  |  1.34  |  0.20  |  0.00  | 5.51 |      7.84     |
| Wikidepia/IndoT5 (~248M)                   |  248M   | 5e-5 |   11.93    | 25.13  |  1.87  |  0.26  |  0.00  | 6.82 |      9.37     |
| Wikidepia/IndoT5 (~248M)                   |  248M   | 1e-4 |   11.33    | 26.20  |  1.42  |  0.31  |  0.00  | 6.98 |      9.15     |
| google/mT5-small                           |  300M   | 1e-5 |    1.18    |  4.66  |  0.13  |  0.00  |  0.00  | 1.20 |      1.19     |
| google/mT5-small                           |  300M   | 5e-5 |    0.40    | 12.30  |  0.00  |  0.00  |  0.00  | 3.07 |      1.74     |
| google/mT5-small                           |  300M   | 1e-4 |    0.40    |  2.65  |  0.24  |  0.00  |  0.00  | 0.72 |      0.56     |

For more additional results and ablation study, you can refer to [this link](https://drive.google.com/file/d/1HR6OQIX3EmlUPQRiI0TA8ZBsOkMXXwm8/view?usp=sharing) (Bahasa Indonesia).

## Setup
### Step 0: Install Requirements (if needed)
Before running the code, ensure you have all the necessary dependencies installed. You can install the required packages by running:
```console
pip install -r requirements.txt
```

## Replicate Result
### Step 1: Edit Configuration
Open the `main.py` file and modify the `Args` class according to your needs. You can adjust parameters such as learning rate (`lr`), dataset size, and other relevant configurations.
For the best result based on this repository experiments, you can set the arguments as follows:
```python
@dataclass
class Args:
    model_checkpoint: str = "LazarusNLP/IndoNanoT5-base"
    dataset_name: str = "dehanalkautsar/xcopa_gen_id"
    answer_column_name: str = "label"
    input_max_length: int = 128
    target_max_length: int = 128
    num_beams: int = 5
    output_dir: str = "outputs/base-indot5-lr5e5-xcopagenid"
    num_train_epochs: int = 10 
    early_stopping_patience: int = 3 
    early_stopping_threshold: float = 0.01
    optim: str = "adamw_torch_fused"
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
```

### Step 2: Run the Script
After configuring the necessary parameters, you can execute the script by running:
```console
python main.py
```

Voila! Your text generation experiment will begin.
