from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from sklearn.metrics import average_precision_score, PrecisionRecallDisplay, roc_auc_score, RocCurveDisplay
from scipy.stats import pearsonr, rankdata #, spearmanr, spearmanrho
import numpy as np
import pandas as pd
import sys
import json
import math
import struct
import gzip
import contextlib
import argparse
from functools import cached_property
from pathlib import Path
from uuid import uuid4

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

if hasattr(contextlib, 'nullcontext'):
    nullcontext = contextlib.nullcontext
else:
    class nullcontext:
        def __init__(self, enter_result=None):
            self.enter_result = enter_result
        def __enter__(self):
            return self.enter_result
        def __exit__(self, *excinfo):
            pass

def choose_open_function(filename, force_gzip=None):
    '''
    If `force_gzip` is True or False, use corresponding open function.
    If `force_gzip` is None, it's guessed based on filename extension
    '''
    if (force_gzip is True) or ((force_gzip is None) and filename.endswith('.gz')):
        return gzip.open
    elif (force_gzip is False) or ((force_gzip is None) and not filename.endswith('.gz')):
        return open
    else:
        raise ValueError("`force_gzip` should be one of True/False/None")

def open_for_read(filename, force_gzip=None, mode='rt', **kwargs):
    if filename and (filename != '-') and (filename != 'stdin'):
        open_func = choose_open_function(filename=filename, force_gzip=force_gzip)
        return open_func(filename, mode, **kwargs)
    else:
        return nullcontext(sys.stdin)


@dataclass
class PredictionsGTContainer:
    scores_noisy: np.ndarray[float]
    ground_truth_noisy: np.ndarray[float]
    noise_threshold: float = 0  # 0 means noise is not removed
                                # T means that ‘singleton’ reads (0 < count <= T) are treated as noise (neither signal, nor anti-signal)
    def __init__(self, scores: np.ndarray[float], ground_truth: np.ndarray[float], noise_threshold: float):
        self.scores_noisy = scores
        self.ground_truth_noisy = ground_truth
        self.noise_threshold = noise_threshold

    @cached_property
    def scores(self) -> np.ndarray[float]:
        if self.noise_threshold > 0:
            return self.scores_noisy[self.denoise_mask]
        else:
            return self.scores_noisy

    @cached_property
    def ground_truth(self) -> np.ndarray[float]:
        if self.noise_threshold > 0:
            return self.ground_truth_noisy[self.denoise_mask]
        else:
            return self.ground_truth_noisy

    @cached_property
    def labels(self) -> np.ndarray[bool]:
        return self.ground_truth > 0

    @cached_property
    def denoise_mask(self) -> np.ndarray[bool]:
        return (self.ground_truth_noisy == 0) | (self.ground_truth_noisy > self.noise_threshold)

    @cached_property
    def scores_masked(self):
        return self.scores[self.labels]

    @cached_property
    def ground_truth_masked(self):
        return self.ground_truth[self.labels]


@dataclass
class ScorerResult:
    value: float
    metainfo: dict | None = None # any json-compatible dict with metainfo

@dataclass
class Scorer(metaclass=ABCMeta):
    @abstractmethod
    def score(self, *args, **kwargs) -> ScorerResult:
        pass

@dataclass
class ConstantScorer(Scorer):
    const: float
    def score(self, *args, **kwargs) -> float:
        return ScorerResult(value=self.const)

class BinaryScorer(Scorer):
    @abstractmethod
    def score(self, scoring_data: PredictionsGTContainer, **kwargs) -> float:
        raise NotImplementedError

@dataclass
class RegressionScorer(Scorer):
    use_masked_profile: bool = False

    def score(self, scoring_data: PredictionsGTContainer, **kwargs) -> float:
        if self.use_masked_profile:
            val = self._calc(y_score=scoring_data.scores_masked, y_real=scoring_data.ground_truth_masked)
        else:
            val = self._calc(y_score=scoring_data.scores, y_real=scoring_data.ground_truth)
        return ScorerResult(value=val, metainfo=None)

    @abstractmethod
    def _calc(self, y_score: np.ndarray[float], y_real: np.ndarray[float]) -> float:
        raise NotImplementedError

class SklearnScorer(BinaryScorer):
    pass

class SklearnROCAUC(SklearnScorer):
    def score(self, scoring_data: PredictionsGTContainer, **kwargs) -> float:
        val = float(roc_auc_score(y_true=scoring_data.labels, y_score=scoring_data.scores))
        return ScorerResult(value=val)

class SklearnPRAUC(SklearnScorer):
    def score(self, scoring_data: PredictionsGTContainer, **kwargs) -> float:
        val = float(average_precision_score(y_true=scoring_data.labels, y_score=scoring_data.scores))
        return ScorerResult(value=val)

@dataclass
class Pearson(RegressionScorer):
    def _calc(self, y_score: np.ndarray[float], y_real: np.ndarray[float]) -> float:
        # mask = np.logical_not(np.isclose(y_real, 0))
        # y_score = y_score[mask]
        # y_real = y_real[mask]
        if len(y_score) < 2:
            return 0
        cor = pearsonr(y_score, y_real).statistic
        if pd.isnull(cor):
            if len(set(y_score)) <= 1 or len(set(y_real)) <= 1 :
                return 0
            else:
                raise Exception("Unknown bug with correlation calculation occured")
        return cor

@dataclass
class Spearman(RegressionScorer):
    def _calc(self, y_score: np.ndarray[float], y_real: np.ndarray[float]) -> float:
        # mask = np.logical_not(np.isclose(y_real, 0))
        # y_score = y_score[mask]
        # y_real = y_real[mask]
        if len(y_score) < 2:
            return 0
        cor = pearsonr(rankdata(y_score, method='dense'), rankdata(y_real, method='dense')).statistic
        # cor = spearmanrho(y_score, y_real).statistic
        if pd.isnull(cor):
            if len(set(y_score)) <= 1 or len(set(y_real)) <= 1 :
                return 0
            else:
                raise Exception("Unknown bug with correlation calculation occured")
        return cor

def read_array_from_wig(filename, dtype=float, force_gzip=None):
    wig_ext = None
    if filename.endswith('.wig.gz'):
      wig_ext = '.wig.gz'
    elif filename.endswith('.wig'):
      wig_ext = '.wig'

    filename_plus = filename.rstrip(wig_ext) + '.plus' + wig_ext
    filename_minus = filename.rstrip(wig_ext) + '.minus' + wig_ext

    profile_plus, strand_plus   = read_fixedstep_wig(filename_plus, dtype=dtype, force_gzip=force_gzip)
    profile_minus, strand_minus = read_fixedstep_wig(filename_minus, dtype=dtype, force_gzip=force_gzip)

    if (strand_plus and strand_plus != '+'):
        raise ValueError(f"Strand mismatch for `{filename_plus}`")
    if (strand_minus and strand_minus != '-'):
        raise ValueError(f"Strand mismatch for `{filename_plus}`")
    return np.concatenate([profile_plus, profile_minus])

def read_array_from_file(filename, dtype, force_gzip=None):
    if filename.endswith('.wig') or filename.endswith('.wig.gz'):
        return read_array_from_wig(filename, dtype=dtype, force_gzip=force_gzip)
    if filename.endswith('.bin') or filename.endswith('.bin.gz'):
        return read_binary_packed_file_array(filename, dtype=dtype, force_gzip=force_gzip)
    elif filename.endswith('.txt') or filename.endswith('.txt.gz') or (filename in {'-', 'stdin'}):
        with open_for_read(filename, mode='rt', force_gzip=force_gzip) as fp:
            target_dtype = np.dtype(dtype)

            # Do not materialize the whole text file as a Python list of
            # strings.  For large profiles that list is much larger than the
            # resulting NumPy array.  Also avoid astype(): it makes a copy
            # even when the source and target dtypes are identical.
            if target_dtype == np.dtype(bool):
                # bool("0") is True.  Masks generated by this project are
                # textual 0/1 values, so avoid the much more expensive float
                # parse.  The first character is sufficient for this
                # project-defined mask format and avoids strip()/slicing.
                def mask_values():
                    yield from (line[0] == '1' for line in fp)

                return np.fromiter(
                    mask_values(),
                    dtype=np.bool_,
                )
            return np.fromiter(fp, dtype=target_dtype)
    else:
        raise ValueError('Unknown filename type')

def read_binary_packed_array(fp, dtype):
    n = struct.unpack("<Q", fp.read(8))[0]
    itemsize = np.dtype(dtype).itemsize
    buf = fp.read(n * itemsize)
    result = np.frombuffer(buf, dtype=dtype)
    return result

def read_binary_packed_file_array(filename, dtype, force_gzip=None):
    with open_for_read(filename, mode='rb', force_gzip=force_gzip) as fp:
        return read_binary_packed_array(fp, dtype)

def parse_wig_track_params(line: str) -> dict[str, str]:
    track_params: dict[str, str] = {}
    fields = line.split()
    if fields[0] != "track":
        raise ValueError(f"Malformed track header `{line!r}`")
    for field in fields[1:]:
        if "=" not in field:
            raise ValueError(f"Line {line_no}: malformed header field {field!r}")
        key, value = field.split("=", 1)
        if (len(key) == 0) or (len(value) == 0):
            raise ValueError(f"Line {line_no}: malformed header field {field!r}")
        if key in track_params:
            raise ValueError(f"Line {line_no}: duplicate header field {key!r}")
        track_params[key] = value.strip('"')
    return track_params

def parse_wig_fixedStep_segment_params(line: str) -> dict[str, str]:
    params: dict[str, str] = {}
    fields = line.split()
    if fields[0] != 'fixedStep':
        raise ValueError(f"Line {line_no}: malformed fixedStep header field {field!r}")
    for field in fields[1:]:
        if "=" not in field:
            raise ValueError(f"Line {line_no}: malformed header field {field!r}")
        key, value = field.split("=", 1)
        if (len(key) == 0) or (len(value) == 0):
            raise ValueError(f"Line {line_no}: malformed header field {field!r}")
        if key in params:
            raise ValueError(f"Line {line_no}: duplicate header field {key!r}")
        params[key] = value.strip('"')

    if "chrom" not in params:
        raise ValueError(f"Line {line_no}: fixedStep header lacks chrom=<...>")
    if "start" not in params:
        raise ValueError(f"Line {line_no}: fixedStep header lacks start=<...>")
    if params.get("step") != "1":
        raise ValueError(f"Line {line_no}: only fixedStep step=1 is supported")
    if "span" in params and params["span"] != "1":
        raise ValueError(f"Line {line_no}: only span=1 or omitted span is supported")

    return params


def read_fixedstep_wig(filename: str, dtype: np.dtype | str | type = np.float64, force_gzip: bool | None = None) -> np.ndarray:
    """
    Read a WIG file in fixedStep step=1 format into a single concatenated NumPy array.
    """
    values: list[str] = []
    seen_header = False

    with open_for_read(filename, mode="rt", encoding="utf-8", force_gzip=force_gzip) as f:
        track_params = None
        strand = None
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if line.startswith('track'):
                if track_params is not None:
                    raise ValueError(f'Line {line_no}: duplicate track header')
                track_params = parse_wig_track_params(line)
                track_name = track_params['name']
                if track_name == 'track_plus':
                    strand = '+'
                elif track_name == 'track_minus':
                    strand = '-'
                else:
                    pass
                    #raise ValueError('Should specify track name either `strand_plus` or `strand_minus`')
                    continue

            if line.startswith("#") or (len(line) == 0):
                continue
            elif line.startswith("variableStep"):
                raise ValueError(f"Line {line_no}: variableStep format is not supported")
            elif line.startswith("fixedStep"):
                params = parse_wig_fixedStep_segment_params(line)
                try:
                    start = int(params["start"])
                except ValueError as exc:
                    raise ValueError(f"Line {line_no}: start must be an integer") from exc

                if start < 1:
                    raise ValueError(f"Line {line_no}: start must be a 1-based positive integer")

                seen_header = True
                continue
            else:
                if not seen_header:
                    raise ValueError(f"Line {line_no}: data encountered before fixedStep header")
                values.append(line)
    try:
        return [np.asarray(values, dtype=dtype), strand]
    except ValueError as exc:
        raise ValueError(f"Could not convert WIG values to dtype {np.dtype(dtype)}") from exc

def get_argparser():
    argparser = argparse.ArgumentParser(
        # prog = "scorer",
        description = 'Calculate regression and classification metrics for profile scores prediction.\n' +
                      'Classification should distinguish zeros from non-zeros.\n' +
                      'Regression should predict score values, positions where ground-truth profile is zero are ignored.',
        usage='python3 scorer.py --ground-truth gt_profile.bin.gz --predictions profile.bin.gz [options]',
        formatter_class=argparse.RawTextHelpFormatter
    )
    argparser.add_argument('--predictions', metavar='FILE',
                           help="File with predictions profile. Allowed data types: .bin, .txt, .wig. Optionally [.gz]-ipped.\n" +
                           "Wig headers are ignored, scores are just concatenated.\n" +
                           "Important: for binary formatted file specify corresponding dtype.")
    argparser.add_argument('--ground-truth', metavar='FILE',
                           help="File with ground-truth profile. Allowed data types: .bin, .txt, .wig. Optionally [.gz]-ipped.\n" +
                           "Wig headers are ignored, scores are just concatenated.\n" +
                           "Important: for binary formatted file specify corresponding dtype.")
    argparser.add_argument('--mode', choices=['classification', 'regression', 'all'], default='all', metavar='MODE',
                           help="subset of metrics to calculate (default: %(default)s)")
    argparser.add_argument('--noise-threshold', type=float, default=0.0, metavar='VAL',
                           help='Positions with low scores (0 < score <= threshold) of ground-truth profile are filtered.')
    argparser.add_argument('--dtype', choices=['int', 'float', 'byte'], default='float', metavar='TYPE',
                           help="Type of score values (default: %(default)s)")
    argparser.add_argument('--name', metavar='STRING',
                           help="Label output score values in the form `{**metrics, name: NAME}`")
    argparser.add_argument('--mask', metavar='FILE',
                           help="Mask file (.txt/.bin, optionally .gz). Positions with value=1 are included in scoring. " +
                           "Must have the same length as input profiles.")
    argparser.add_argument('--mask-mode', choices=['mask', 'inverse', 'both', 'all'], default='mask', metavar='MODE',
                           help="How to apply mask: 'mask' (use mask=1), 'inverse' (use mask=0), 'both' (compute both separately), all (both + without masking)")
    argparser.add_argument('--plots-dir', metavar='DIR',
                           help="Directory for metric plots")

    return argparser


def generate_metric_plots(scoring_data, metric_values, plots_dir):
    # plt.rcParams.update({"font.size": 18})

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_files = {}

    for metric_name, metric_value in metric_values.items():
        output_path = plots_dir / f'{metric_name}-{uuid4().hex}.png'
        if metric_name == 'rocauc':
            plot = RocCurveDisplay.from_predictions(scoring_data.labels, scoring_data.scores)

            # n_neg = (np.asarray(scoring_data.labels) == 0).sum()
            # fig, ax = plt.subplots(figsize=(8, 6))
            # plot = RocCurveDisplay.from_predictions(scoring_data.labels, scoring_data.scores, ax=ax)
            # ax.set_aspect("auto")
            # ax.set_xlim(0, 1)   # FPR: 0–100%
            # ax.set_ylim(0, 1)
            # # ax.set_xscale("log")
            # ax.set_xscale("symlog", linthresh=1 / n_neg,)
            # ax.grid(True, which="both")
            # plt.show()
        elif metric_name == 'prauc':
            plot = PrecisionRecallDisplay.from_predictions(scoring_data.labels, scoring_data.scores)
        elif metric_name in {'pearson', 'spearman'}:
            predictions = scoring_data.scores_masked
            ground_truth = scoring_data.ground_truth_masked

            # if len(predictions) > 1_000_000:
            #     indices = np.linspace(0, len(predictions) - 1, 1_000_000, dtype=np.intp)
            #     predictions, ground_truth = predictions[indices], ground_truth[indices]

            if metric_name == 'spearman':
                predictions = rankdata(predictions, method='dense')
                ground_truth = rankdata(ground_truth, method='dense')
                axis_suffix = ' rank'
            else:
                axis_suffix = ''

            predictions = np.log1p(predictions)
            ground_truth = np.log1p(ground_truth)
            plot = sns.jointplot(
                x=ground_truth, y=predictions, kind='hex',
                joint_kws={'gridsize': 70, 'mincnt': 1, 'bins': 'log'},
            )
            plot.set_axis_labels(f'log1p(Ground truth{axis_suffix})', f'log1p(Prediction{axis_suffix})')
            plot.figure.suptitle(f'{metric_name.removeprefix("scikit_").title()} = {metric_value:.4f}')
        figure = plot.figure_ if hasattr(plot, 'figure_') else plot.figure
        figure.savefig(output_path, dpi=160, bbox_inches='tight')
        plt.close(figure)
        plot_files[metric_name] = str(output_path)
    return plot_files


def compute_all_metrics(scoring_data, scorers, plots_dir=None):
    """Compute all metrics for given scoring data."""
    all_scores = {}
    for k, v in scorers.items():
        score = v.score(scoring_data)
        all_scores[k] = score.value
        if math.isnan(all_scores[k]):
            all_scores[k] = None
    if plots_dir is not None:
        all_scores['plots'] = generate_metric_plots(
            scoring_data=scoring_data,
            metric_values=all_scores,
            plots_dir=plots_dir,
        )
    return all_scores


def main():
    argparser = get_argparser()

    if len(sys.argv)==1:
        argparser.print_help(sys.stderr)
        sys.exit(1)
    args = argparser.parse_args()

    mode = args.mode

    if args.dtype == 'int': # signed int64
        dtype = '<Q'
    elif args.dtype == 'uint8':
        dtype = 'B'  # same as np.uint8
    elif args.dtype == 'float': # float64
        dtype = '<f8'

    predictions = read_array_from_file(args.predictions, dtype=dtype)
    ground_truth = read_array_from_file(args.ground_truth, dtype=dtype)

    if mode == 'classification':
        scorers = {
            "rocauc": SklearnROCAUC(),
            "prauc": SklearnPRAUC(),
        }
    elif mode == 'regression':
        scorers = {
            "pearson": Pearson(use_masked_profile=True),
            "spearman": Spearman(use_masked_profile=True),
        }
    elif mode == 'all':
        scorers = {
            "rocauc": SklearnROCAUC(),
            "prauc": SklearnPRAUC(),
            "pearson": Pearson(use_masked_profile=True),
            "spearman": Spearman(use_masked_profile=True),
        }
    else:
        raise ValueError(f'Unknown mode `{mode}`')

    # noise_threshold:  0 reads is ok, > noise_threshold is ok, 0 < signal <= noise_threshold is probably an artifact
    # noise_threshold = 0  # no filtering
    noise_threshold = args.noise_threshold  # remove singletons

    # Load and validate mask if provided
    mask = None
    if args.mask:
        mask = read_array_from_file(args.mask, dtype=bool)
        if len(mask) != len(predictions):
            raise ValueError(f"Mask length ({len(mask)}) doesn't match profile length ({len(predictions)})")

    # noise_threshold:  0 reads is ok, > noise_threshold is ok, 0 < signal <= noise_threshold is probably an artifact
    # noise_threshold = 0  # no filtering
    noise_threshold = args.noise_threshold  # remove singletons

    if mask is not None:
        if args.mask_mode == 'all':
            # Compute each result before constructing the next large slice.
            mask_data = PredictionsGTContainer(
                scores=predictions[mask],
                ground_truth=ground_truth[mask],
                noise_threshold=noise_threshold
            )
            mask_output = compute_all_metrics(mask_data, scorers, args.plots_dir)
            del mask_data

            inverse_data = PredictionsGTContainer(
                scores=predictions[~mask],
                ground_truth=ground_truth[~mask],
                noise_threshold=noise_threshold
            )
            inverse_output = compute_all_metrics(inverse_data, scorers, args.plots_dir)
            del inverse_data

            total_data = PredictionsGTContainer(
                scores=predictions,
                ground_truth=ground_truth,
                noise_threshold=noise_threshold
            )
            total_output = compute_all_metrics(total_data, scorers, args.plots_dir)
            del total_data, predictions, ground_truth, mask

            output = {
                "mask": mask_output,
                "inverted_mask": inverse_output,
                "total": total_output,
            }
        elif args.mask_mode == 'both':
            # Compute each result before constructing the next large slice.
            mask_data = PredictionsGTContainer(
                scores=predictions[mask],
                ground_truth=ground_truth[mask],
                noise_threshold=noise_threshold
            )
            mask_output = compute_all_metrics(mask_data, scorers, args.plots_dir)
            del mask_data

            inverse_data = PredictionsGTContainer(
                scores=predictions[~mask],
                ground_truth=ground_truth[~mask],
                noise_threshold=noise_threshold
            )
            inverse_output = compute_all_metrics(inverse_data, scorers, args.plots_dir)
            del inverse_data

            output = {"mask": mask_output, "inverted_mask": inverse_output}
        elif args.mask_mode == 'mask':
            scoring_data = PredictionsGTContainer(
                scores=predictions[mask],
                ground_truth=ground_truth[mask],
                noise_threshold=noise_threshold
            )
            # The scoring container owns the selected copies now.
            del predictions, ground_truth, mask
            output = {"mask": compute_all_metrics(scoring_data, scorers, args.plots_dir)}
        elif args.mask_mode == 'inverse':
            scoring_data = PredictionsGTContainer(
                scores=predictions[~mask],
                ground_truth=ground_truth[~mask],
                noise_threshold=noise_threshold
            )
            del predictions, ground_truth, mask
            output = {"inverted_mask": compute_all_metrics(scoring_data, scorers, args.plots_dir)}
        else:
            raise ValueError(f"Unknown mask mode: {args.mask_mode}")
    else:
        # Standard scoring without mask
        scoring_data = PredictionsGTContainer(predictions, ground_truth, noise_threshold)
        output = compute_all_metrics(scoring_data, scorers, args.plots_dir)

    if args.name:
        output['name'] = args.name
    print(json.dumps(output, ensure_ascii=False))

if __name__ == '__main__':
    main()
