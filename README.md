# DMRadiomics

## Overview  
Scripts used to predict failure of active surveillance (AS) in desmoid-type fibromatosis (DTF) using WORC (Workflow for Optimal Radiomics Classification): <PLACEHOLDER>

Before trying out the code in this repository, we advice you to get
familiar with the WORC package through the WORC tutorial:
https://github.com/MStarmans91/WORCTutorial.

## Licensing
This project is licensed under the open-source [APACHE 2.0 License](LICENSE).

If you use this code, please cite both this repository and the associated publication as follows:
<PLACEHOLDER>

We also encourage citing the underlying WORC methodology as follows:
```bibtex
@article{starmans2021reproducible,
   title          = {Reproducible radiomics through automated machine learning validated on twelve clinical applications}, 
   author         = {Martijn P. A. Starmans and Sebastian R. van der Voort and Thomas Phil and Milea J. M. Timbergen and Melissa Vos and Guillaume A. Padmos and Wouter Kessels and David    Hanff and Dirk J. Grunhagen and Cornelis Verhoef and Stefan Sleijfer and Martin J. van den Bent and Marion Smits and Roy S. Dwarkasing and Christopher J. Els and Federico Fiduzi and Geert J. L. H. van Leenders and Anela Blazevic and Johannes Hofland and Tessa Brabander and Renza A. H. van Gils and Gaston J. H. Franssen and Richard A. Feelders and Wouter W. de Herder and Florian E. Buisman and Francois E. J. A. Willemssen and Bas Groot Koerkamp and Lindsay Angus and Astrid A. M. van der Veldt and Ana Rajicic and Arlette E. Odink and Mitchell Deen and Jose M. Castillo T. and Jifke Veenland and Ivo Schoots and Michel Renckens and Michail Doukas and Rob A. de Man and Jan N. M. IJzermans and Razvan L. Miclea and Peter B. Vermeulen and Esther E. Bron and Maarten G. Thomeer and Jacob J. Visser and Wiro J. Niessen and Stefan Klein},
   year           = {2021},
   eprint         = {2108.08618},
   archivePrefix  = {arXiv},
   primaryClass   = {eess.IV}
}

@software{starmans2018worc,
  author       = {Martijn P. A. Starmans and Thomas Phil and Sebastian R. van der Voort and Stefan Klein},
  title        = {Workflow for Optimal Radiomics Classification (WORC)},
  year         = {2018},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.3840534},
  url          = {https://github.com/MStarmans91/WORC}
}
```

## Quick Start  
1. Clone repository:
   ```bash
   git clone https://github.com/Douwe-Spaanderman/DMRadiomics.git
   ```
2. Install requirements:
    - Recommended to install below in virtual environment: ```python -m venv venv```
    - Install WORC:
        ```bash
        pip install WORC
        ```
    - Install other requirements:
        ```bash
        pip install -r requirements.txt
        ```
3. Run the analysis:
    ```bash
    python run.py -d /path/to/data -n experiment_name
    ```

## Input Data Structure

Organize your data as follows:

```
data_folder/
├── Patient_001/
│   ├── image.nii.gz
│   └── mask.nii.gz
├── Patient_002/
│   ├── image.nii.gz
│   └── mask.nii.gz
└── labels.txt
```

- `image.nii.gz`: The imaging data (e.g., T1-weighted MRI).
- `mask.nii.gz`: Corresponding segmentation mask.
- `labels.txt`: A tab-separated file containing patient IDs and their associated labels.

## Key Options

| Argument | Description |
|----------|-------------|
| `-d`     | Path to the data directory **(required)** |
| `-n`     | Name for the experiment output folder **(required)** |
| `-s`     | MRI sequence to use (default: `T1`) |
| `-l`     | Prediction label (default: `PD`) |
| `-co`    | Enable ComBat harmonization |

## Output

Results will be saved in a folder named `WORC_[experiment_name]`, including:

- Extracted features  
- Model performance metrics  
- Trained machine learning models  
- WORC configuration and logs

## Full Documentation

For detailed documentation, please refer to the [official WORC documentation 📚]( https://worc.readthedocs.io/)
