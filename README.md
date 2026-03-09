## Project Setup

**Clone the project:**
```bash
git clone https://github.com/nikulkaaa/nlp2-gr5.git
cd nlp2-gr5
```

**Setup with uv:**
```bash
uv sync
```

**Activate virtual environment:**

macOS/Linux/WSL:
```bash
source .venv/bin/activate
```

Windows:
```powershell
.\.venv\Scripts\activate
```

## One Command Run
To run the pipeline of this project, run:
```powershell
python main.py
```
This will load the data and then make the splits. It preprocesses the text, then tokenize, build a vocabulary, and encode sequences (padded/truncated to 128 tokens). Then trains a CNN and an LSTM model and evaluates both on the test set using Accuracy, Macro F1, and Confusion Matrices. The program selects the best performing model and saves the first 10 misclassified predictions for error analysis into the results/ folder for later analysis. Additionally, we save .json files for each model with metrics, training histories, confusion matrix plots, learning curve plots, and a .csv with misclassified samples for both models. Finally, it conducts an ablation study by re-running the pipeline with max_length at values of 10, 20, 30, 64, 128 and storing the aforementioned metrics for all of the combinations.
