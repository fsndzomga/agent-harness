"""
Official AssistantBench evaluation — vendored from ``browsergym-assistantbench``.

Source: https://github.com/ServiceNow/BrowserGym  (browsergym.assistantbench.evaluation)
Paper:  https://arxiv.org/abs/2407.15711

This module is self-contained and requires only ``numpy`` and ``scipy``.
It is vendored here to avoid pulling in the full ``browsergym`` dependency
tree (playwright, etc.) just for scoring.
"""

from __future__ import annotations

import json
import re
import string
from typing import Any, Callable, Dict, List, Set, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# evaluate_numbers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def distance_function_log(pred: float, gold: float) -> float:
    if pred == gold == 0:
        return 1
    if pred == 0:
        pred = 1e-4
    if gold == 0:
        gold = 1e-4
    if pred > gold:
        return max(0, 1 - np.log(pred / gold))
    else:
        return max(0, 1 - np.log(gold / pred))


def evaluate_numbers(pred: Union[float, str], gold: float) -> float:
    res = None
    if type(pred) not in (float, int):
        try:
            pred = float(pred)
        except ValueError:
            res = 0
    if type(gold) not in (float, int):
        try:
            gold = float(gold)
        except ValueError:
            res = 0
    if res is None:
        res = distance_function_log(pred, gold)
    return res


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# evaluate_strings  (DROP benchmark F1)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_EXCLUDE = set(string.punctuation)


def _remove_articles(text: str) -> str:
    return re.sub(_ARTICLES_RE, " ", text)


def _white_space_fix(text: str) -> str:
    return " ".join(text.split())


def _remove_punc(text: str) -> str:
    if not _is_number(text):
        return "".join(ch for ch in text if ch not in _EXCLUDE)
    return text


def _lower(text: str) -> str:
    return text.lower()


def _tokenize(text: str) -> List[str]:
    return re.split(" |-", text)


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _normalize_number(text: str) -> str:
    if _is_number(text):
        return str(float(text))
    return text


def _normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    parts = [
        _white_space_fix(
            _remove_articles(_normalize_number(_remove_punc(_lower(token))))
        )
        for token in _tokenize(text)
    ]
    parts = [part for part in parts if part.strip()]
    return " ".join(parts).strip()


def _answer_to_bags(
    answer: Union[str, List[str], Tuple[str, ...]],
) -> Tuple[List[str], List[Set[str]]]:
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    normalized_spans: List[str] = []
    token_bags: List[Set[str]] = []
    for raw_span in raw_spans:
        normalized_span = _normalize_answer(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags


def _compute_f1(predicted_bag: Set[str], gold_bag: Set[str]) -> float:
    intersection = len(gold_bag.intersection(predicted_bag))
    precision = 1.0 if not predicted_bag else intersection / float(len(predicted_bag))
    recall = 1.0 if not gold_bag else intersection / float(len(gold_bag))
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def _match_numbers_if_present(
    gold_bag: Set[str], predicted_bag: Set[str]
) -> bool:
    gold_numbers = {w for w in gold_bag if _is_number(w)}
    predicted_numbers = {w for w in predicted_bag if _is_number(w)}
    if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
        return True
    return False


def _align_bags_strings(
    predicted: List[Set[str]], gold: List[Set[str]]
) -> List[float]:
    scores = np.zeros([len(gold), len(predicted)])
    for gi, g in enumerate(gold):
        for pi, p in enumerate(predicted):
            if _match_numbers_if_present(g, p):
                scores[gi, pi] = _compute_f1(p, g)
    row_ind, col_ind = linear_sum_assignment(-scores)
    max_scores = np.zeros([max(len(gold), len(predicted))])
    for r, c in zip(row_ind, col_ind):
        max_scores[r] = max(max_scores[r], scores[r, c])
    return max_scores


def evaluate_strings(prediction: Any, gold: Any) -> float:
    if not isinstance(prediction, (list, str)):
        prediction = str(prediction)
    if not isinstance(gold, (list, str)):
        gold = str(gold)
    try:
        predicted_bags = _answer_to_bags(prediction)
        gold_bags = _answer_to_bags(gold)
        f1_per_bag = _align_bags_strings(predicted_bags[1], gold_bags[1])
        f1 = float(np.mean(f1_per_bag))
    except Exception:
        f1 = 0.0
    return f1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# utils — generic alignment helper (used by evaluate_dicts)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _align_bags_generic(
    predicted: list, gold: list, method: Callable[[Any, Any], float]
) -> List[float]:
    scores = np.zeros([len(gold), len(predicted)])
    for gi, g in enumerate(gold):
        for pi, p in enumerate(predicted):
            scores[gi, pi] = method(p, g)
    row_ind, col_ind = linear_sum_assignment(-scores)
    max_scores = np.zeros([max(len(gold), len(predicted))])
    for r, c in zip(row_ind, col_ind):
        max_scores[r] = max(max_scores[r], scores[r, c])
    return max_scores


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# evaluate_dicts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_EVALUATOR_BY_TYPE: Dict[type, Callable] = {
    str: evaluate_strings,
    int: evaluate_numbers,
    float: evaluate_numbers,
    bool: evaluate_strings,
    list: evaluate_strings,
}


def _fix_number_dict(number: Any) -> Any:
    if isinstance(number, str):
        copy_ans = number
        copy_ans = " ".join(
            " ".join(" ".join(copy_ans.split("$")).split("%")).split("sqft")
        ).strip().replace(",", ".")
        try:
            return float(copy_ans)
        except Exception:
            return number
    elif isinstance(number, int):
        return float(number)
    return number


def _calc_recall(pred: Dict, gold: Dict, use_gold_for_eval: bool) -> float:
    recall = []
    for gold_key, gold_value in gold.items():
        pred_value = pred.get(gold_key)
        gold_value = _fix_number_dict(gold_value)
        pred_value = _fix_number_dict(pred_value)
        if gold_key not in pred:
            recall.append(0)
        else:
            evaluator = (
                _EVALUATOR_BY_TYPE[type(gold_value)]
                if use_gold_for_eval
                else _EVALUATOR_BY_TYPE[type(pred_value)]
            )
            if type(pred_value) != type(gold_value):
                recall.append(0)
                continue
            recall.append(evaluator(pred_value, gold_value))
    return float(np.average(recall))


def _evaluate_pair_of_dicts(pred: Dict, gold: Dict) -> float:
    recall = _calc_recall(pred, gold, True)
    precision = _calc_recall(gold, pred, False)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def evaluate_dicts(pred: Any, gold: List[Dict]) -> float:
    if not (
        isinstance(pred, dict)
        or (isinstance(pred, list) and len(pred) == 0)
        or (isinstance(pred, list) and isinstance(pred[0], dict))
    ):
        return 0
    max_scores = _align_bags_generic(pred, gold, _evaluate_pair_of_dicts)
    return float(np.average(max_scores))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# evaluate_factory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_EVALUATOR_BY_NAME: Dict[str, Callable] = {
    "string": evaluate_strings,
    "number": evaluate_numbers,
    "json": evaluate_dicts,
    "string list": evaluate_strings,
}


def get_evaluator(evaluator: str) -> Callable:
    return _EVALUATOR_BY_NAME[evaluator]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# evaluator.py — top-level question_scorer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _find_isnan(samp: Any) -> bool:
    try:
        if np.isnan(samp):
            return True
        return False
    except Exception:
        return False


def _fix_ans(answer: str) -> str:
    try:
        answer = (
            answer.replace("{'", '{"')
            .replace("', '", '", "')
            .replace("': '", '": "')
            .replace("'}", '"}')
            .replace("': ", '": ')
        )
        return answer
    except Exception:
        return answer


def _fix_number(number: Any) -> tuple:
    if isinstance(number, str):
        copy_ans = number
        copy_ans = " ".join(
            " ".join(" ".join(copy_ans.split("$")).split("%")).split("sqft")
        ).strip().replace(",", ".").replace(" square kilometers", "")
        try:
            return float(copy_ans), True
        except Exception:
            return number, False
    elif isinstance(number, int):
        return float(number), True
    else:
        return number, True


def _parse_answer(answer: list) -> tuple:
    if len(answer) == 1:
        ans, is_num = _fix_number(answer[0])
        if is_num:
            return ans, "number"
        try:
            ans = json.loads(_fix_ans(answer[0]))
            return [ans], "json"
        except Exception:
            ans, is_num = _fix_number(answer[0])
            if is_num:
                return ans, "number"
            else:
                return answer[0], "string"
    else:
        try:
            ans = [json.loads(_fix_ans(ex)) for ex in answer]
            return ans, "json"
        except Exception:
            return answer, "string list"


def _fix_prediction(prediction: Any, gold_answer: Any, evaluator: str) -> tuple:
    if (
        isinstance(prediction, list)
        and len(prediction) == 1
        and (
            isinstance(prediction[0], int)
            or (isinstance(prediction[0], str) and prediction[0].isnumeric())
        )
    ):
        prediction = _fix_number(prediction[0])

    if not isinstance(prediction, list):
        prediction, is_num = _fix_number(prediction)
        if evaluator == "json":
            try:
                prediction = [json.loads(pred) for pred in prediction.split("\n")]
            except Exception:
                prediction = [prediction]

    if hasattr(type(prediction), "__len__") and len(prediction) == 0:
        return prediction, False

    if (
        isinstance(prediction, list) and len(prediction) > 1
    ) and isinstance(gold_answer, float):
        return prediction, False

    return prediction, True


def question_scorer(
    prediction: Any, gold_answer: str
) -> Tuple[float, float]:
    """Score a prediction against a gold answer.

    Returns ``(accuracy, has_ans)`` where *accuracy* is in [0, 1] and
    *has_ans* indicates whether the prediction was non-empty.
    """
    try:
        prediction = json.loads(prediction)
    except Exception:
        pass

    answer_list = (
        [x for x in gold_answer.split("\n") if len(x.strip()) > 0]
        if not isinstance(gold_answer, list)
        else gold_answer
    )
    gold_answer_parsed, evaluator = _parse_answer(answer_list)
    prediction, run_eval = _fix_prediction(prediction, gold_answer_parsed, evaluator)

    has_ans = 1.0
    if (
        not isinstance(prediction, float)
        and len(prediction) == 0
    ) or _find_isnan(prediction):
        has_ans = 0.0

    if isinstance(prediction, list):
        if all(
            (not isinstance(pred, (float, int)) and len(pred) == 0)
            or _find_isnan(pred)
            for pred in prediction
        ):
            has_ans = 0

    if not run_eval:
        return 0.0, has_ans

    metric_eval = get_evaluator(evaluator)
    accuracy = metric_eval(prediction, gold_answer_parsed)
    return float(accuracy), float(has_ans)
