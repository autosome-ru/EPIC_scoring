## EPIC Scoring
Scoring for Eukaryotic Promoter and Transcription Initiation Prediction Challenge (EPIC).
The competition website: https://epic.autosome.org/

### EPIC scoring scripts
To compute the performance metrics, we use the Python script `scorer.py` available at GitHub: https://github.com/autosome-ru/EPIC_scoring

Note: Python 3 should be preinstalled.

Library dependencies are listed in `requirements.txt`.

We recommend setting up a separate virtual environment:
```bash
cd /path/to/epic_folder
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
To run the evaluator, first activate the virtual environment if it has not yet been activated in the current terminal session:
```bash
source .venv/bin/activate
```
To verify the data format of your predictions, please use the `validate` bash script. It takes two arguments: the template length and your submission file. Template length can be extracted from `genome_metrics.json` file, e.g., using `jq` tool:

```bash
TEMPLATE_LENGTH=$(
  cat genome_metrics.json |
    jq ".assemblies[\"${ASSEMBLY}\"].template.test.length"
)

bash ./validate $TEMPLATE_LENGTH submission.txt.gz
```
Then the scorer can be executed by running:

```bash
python3 scorer.py \
    --ground-truth "${GROUND_TRUTH_FILE}" \
    --prediction "${PREDICTIONS_FILE}"
```
The scorer reports the following performance metrics:

Of those, `prauc` and `spearman` are used for online scoring, while `rocauc` and `pearson` might be used in the post-challenge analysis, but will not affect the team rankings.

The ground truth file used for scoring contains scores in the same order as predictions but are not restricted to the `0..1` range and are integers by construction (though may be stored as floats):
```
0.0
1.0
5.0
0.0
2.0
4.0
```

**Note:** the ground truth data (the challenge answers) will be provided after the challenge, except for the _Nematostella vectensis_ toy example data, which will not be used in the actual scoring.

### Leaderboard and final test data and scoring
Each test dataset is split into two parts of almost equal size, one for the leaderboard stage and another for the final evaluation. We calculate scores for each part separately. During the leaderboard stage, the participants will see scores for the first half of the test dataset.

In the final evaluation, the scores obtained for the second half of the test dataset will be revealed and treated as the final scores.

For online evaluation, the scorer script uses a plain text mask file (1=use position, 0=ignore position) to estimate the metrics (the following options are appended to the script invocation command line):
```
--mask MASK_FILE.txt.gz --mask-mode both`
```
As a result, there will be reported scores for two halves (mask and inverted mask), and additionally for the full dataset.

We will publish the actual masks with the ground truth data upon the challenge completion.

### Offline scoring
We provide the scoring scripts and the previously published data for Starlet sea anemone (_Nematostella vectensis_) to allow for replicating the challenge scoring scheme. The complete challenge data (including the test set labels and read counts) will be open upon challenge completion.
