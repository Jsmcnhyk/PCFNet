# PCFNet

## Updates
The detailed training logs (see [_logs](logs/LongForecasting/PCFNet)).

## Usage

1. Install the dependencies
    ```bash
    pip install -r requirements.txt
    ```

2. Obtain the dataset from [Google Drive](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view?usp=drive_link) and extract it to the root directory of the project. Make sure the extracted folder is named `dataset` and has the following structure:
    ```
    dataset
    ├── electricity
    │   └── electricity.csv
    ├── ETT-small
    │   ├── ETTh1.csv
    │   ├── ETTh2.csv
    │   ├── ETTm1.csv
    │   └── ETTm2.csv
    ├── Solar
    │   └── solar_AL.txt
    ├── traffic
    │   └── traffic.csv
    └── weather
        └── weather.csv
    ```

3. Train and evaluate the model. All the training scripts are located in the `scripts` directory. For example, to train the model on the ECL dataset, run the following command:
   
   For Linux/macOS:
    ```bash
    sh ./scripts/PCFNet.sh
    ```
    For Windows:
    ```
    ./scripts/PCFNet.bat
    ```
