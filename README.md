# XGID: xcopa_gen_id
## About
This repository contains an experimental setup using a curated modified dataset derived from XCOPA (Ponti et al., 2020), tailored for text generation tasks. The modifications enable the dataset to be utilized effectively for generating coherent and contextually relevant text. This repository is intended for the final task of IF5281 Deep Learning at Institut Teknologi Bandung for the academic year 2023/2024.

## How to Run
### Step 0: Install Requirements (if needed)
Before running the code, ensure you have all the necessary dependencies installed. You can install the required packages by running:
```console
pip install -r requirements.txt
```

### Step 1: Edit Configuration
Open the `main.py` file and modify the `Args` class according to your needs. You can adjust parameters such as learning rate (`lr`), dataset size, and other relevant configurations.

### Step 2: Run the Script
After configuring the necessary parameters, you can execute the script by running:
```console
python main.py
```

Voila! Your text generation experiment will begin.
