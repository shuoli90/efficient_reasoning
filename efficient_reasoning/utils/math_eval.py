# Authors: marcusm117
# License: Apache 2.0

# This file is a collection of (adpated) evaluation helpers used in the following benchmarks:
# MATH: https://github.com/hendrycks/math/blob/main/modeling/math_equivalence.py
# OlympiadBench: https://github.com/OpenBMB/OlympiadBench/blob/main/eval/auto_scoring_judge.py


import math
import re
import sympy as sp  # type: ignore
from sympy import simplify, Eq, sympify, Pow  # type: ignore
from sympy.parsing.latex import parse_latex  # type: ignore
from typing import TypeAlias, Literal, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from concurrent.futures._base import CancelledError
from tqdm import tqdm
from multiprocessing import cpu_count

from . import code_utils

Benchmark: TypeAlias = Literal["AIME_2024", "MATH-500", "OlympiadBench-674-MATH_TO_EN", "BigCodeBench"]
# =============================================================================
# adapted `last_boxed_only_string` and `remove_boxed` functions from MATH
# =============================================================================
def last_boxed_only_string(string):
    idx = -1
    patterns = ["\\boxed", "\\fbox", "\\framebox", "\boxed"]
    for pattern in patterns:
        idx = string.rfind(pattern)
        if idx >= 0:
            break

    # if no boxed, fboxed, or frameboxed string is found, return None
    if idx == -1:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left_list = ["\\boxed{", "\\fbox{", "\\framebox{", "\boxed{"]
    try:
        assert (
            s[: len(left_list[0])] == left_list[0] or
            s[: len(left_list[1])] == left_list[1] or
            s[: len(left_list[2])] == left_list[2] or
            s[: len(left_list[3])] == left_list[3]
        )
        assert s[-1] == "}"
    except:  # noqa: E722
        return None

    if s[: len(left_list[0])] == left_list[0]:
        return s[len(left_list[0]) : -1]
    elif s[: len(left_list[1])] == left_list[1]:
        return s[len(left_list[1]) : -1]
    elif s[: len(left_list[2])] == left_list[2]:
        return s[len(left_list[2]) : -1]
    else:
        return s[len(left_list[3]) : -1]


# =============================================================================
# `is_equiv` function from MATH
# =============================================================================
def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:  # noqa: E722
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:  # noqa: E722
        return string


def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except:  # noqa: E722
        return str1 == str2


# =============================================================================
# `AutoScoringJudge` class from OlympiadBench
# =============================================================================
class AutoScoringJudge:
    def __init__(self):
        # Map of special symbols to their replacements
        self.special_signal_map = {
            "\\left": "",
            "\\right": "",
            "∶": ":",
            "，": ",",
            "$": "",
            "\\approx": "=",
            "\\simeq": "=",
            "\\sim": "=",
            "^\\prime": "'",
            "^{\\prime}": "'",
            "^\\circ": "",
            "%": "",
        }
        self.pi = parse_latex("\\pi")
        self.precision = 1e-8  # Default precision for comparison

    def split_by_comma(self, expr: str):
        # Splits expressions by commas outside of brackets
        in_bracket_num = 0
        splitted_expr = []
        start_idx = 0
        for i, char in enumerate(expr):
            if char in ["(", "["]:
                in_bracket_num += 1
            elif char in [")", "]"]:
                in_bracket_num -= 1
            elif char == "," and in_bracket_num == 0:
                splitted_expr.append(expr[start_idx:i].strip())
                start_idx = i + 1

        if start_idx < len(expr):
            splitted_expr.append(expr[start_idx:].strip())

        return splitted_expr

    def trans_plus_minus_sign(self, expr_list: list):
        # Translates plus-minus signs into separate expressions
        new_expr_list = []
        for expr in expr_list:
            if "\\pm" in expr:
                new_expr_list.append(expr.replace("\\pm", "+"))
                new_expr_list.append(expr.replace("\\pm", "-"))
            else:
                new_expr_list.append(expr)

        return new_expr_list

    def judge(self, expression1, expression2, precision=1e-8):
        # Judge if two expressions are equal (expression1 is considered as the Ground Truth)
        # Default precision is a list for supporting multiple expressions
        precision = precision if isinstance(precision, list) else [precision]

        try:
            expression1, expression2 = self.preprocess(expression1, expression2)
        except:  # noqa: E722
            return False
        if expression1 == expression2:
            # print("Exactly equal")
            return True

        # Remove Chinese characters from the string, as answers like "yes" or "no" in Chinese have been considered
        expression1 = re.sub(r"[\u4e00-\u9fff]+", "", expression1)
        expression2 = re.sub(r"[\u4e00-\u9fff]+", "", expression2)

        expression1 = self.split_by_comma(expression1)
        expression2 = self.split_by_comma(expression2)

        temp_list1 = self.trans_plus_minus_sign(expression1)
        temp_list2 = self.trans_plus_minus_sign(expression2)

        # Set up a list for allowed errors
        if len(precision) <= 1:
            precision = precision * len(temp_list1)

        if len(temp_list1) != len(temp_list2):
            return False

        # Check if elements in both lists can be paired and are equal
        idx = -1
        while len(temp_list1) != 0:
            idx = (idx + 1) % len(temp_list1)

            item1 = temp_list1[idx]
            self.precision = precision[idx]

            for item2 in temp_list2:
                if self.is_equal(item1, item2):
                    temp_list1.remove(item1)
                    temp_list2.remove(item2)
                    precision.remove(self.precision)
                    break
            else:
                # If no match was found, return False
                return False

        # If all elements are matched, return True
        return True

    def is_interval(self, expr):
        # Checks if an expression is an interval
        return expr.startswith(("(", "[")) and expr.endswith((")", "]"))

    def sympy_sub_pi(self, expression_sympy):
        # Replaces the symbol for pi in sympy expressions with its numerical value
        return expression_sympy.subs(self.pi, math.pi)

    def is_equal(self, expression1, expression2):
        # Default first expression is ground truth. Check if expressions are equal in different aspects
        if expression1 == expression2 and expression1 != "" and expression2 != "":
            # print("Equivalent natively")
            return True

        # First check if both are intervals
        if self.is_interval(expression1) and self.is_interval(expression2):
            try:
                if self.interval_equal(expression1, expression2):
                    # print("Interval equivalent")
                    return True
            except:  # noqa: E722
                return False

        # Then check for numerical equality
        try:
            if self.numerical_equal(expression1, expression2):
                # print("Numerically equivalent")
                return True
        except:  # noqa: E722
            pass

        # Then check if expressions are mathematically equal
        try:
            if self.expression_equal(expression1, expression2) and not ("=" in expression1 and "=" in expression2):
                # print("Expression equivalent")
                return True
        except:  # noqa: E722
            pass

        # Lastly, check for equation equality
        try:
            if self.equation_equal(expression1, expression2):
                # print("Equation equivalent")
                return True
        except:  # noqa: E722
            pass

        return False

    def numerical_equal(self, expression1: str, expression2: str, include_percentage: bool = True):
        # Check if two numerical values are equal within an allowed error range
        # Includes possible percentage cases
        reference = float(expression1)
        prediction = float(expression2)

        if include_percentage:
            gt_result = [reference / 100, reference, reference * 100]
        else:
            gt_result = [reference]

        for item in gt_result:
            if abs(item - prediction) <= self.precision * 1.01:
                return True
        return False

    def expression_equal(self, exp1, exp2):
        # Check if two expressions are mathematically equivalent
        # Extract expression and use sympy for equivalence checking
        def extract_expression(expression):
            if "=" in expression:
                expression = expression.split("=")[1]
            return expression.strip()

        exp1 = extract_expression(exp1)
        exp2 = extract_expression(exp2)

        expr1_sym = sympify(parse_latex(exp1))
        expr2_sym = sympify(parse_latex(exp2))

        if expr1_sym == expr2_sym:
            return True
        else:
            expr1_sym = self.sympy_sub_pi(expr1_sym)
            expr2_sym = self.sympy_sub_pi(expr2_sym)

            if (expr1_sym.has(sp.Symbol) and not expr2_sym.has(sp.Symbol)) or (not expr1_sym.has(sp.Symbol) and expr2_sym.has(sp.Symbol)):
                return False
            elif not expr1_sym.has(sp.Symbol) and not expr2_sym.has(sp.Symbol):
                try:
                    if not (self.can_compute_power(expr1_sym) and self.can_compute_power(expr2_sym)):
                        print(f'These two numbers cannot be calculated by the current computer for: "{str(expr1_sym)}" and "{str(expr2_sym)}"')
                        return False

                    if abs(expr1_sym.evalf() - expr2_sym.evalf()) <= self.precision * 1.01:
                        return True
                    else:
                        return False
                except:  # noqa: E722
                    return False
            else:
                try:
                    simplified_expr = simplify(expr1_sym - expr2_sym)

                    num_value = simplified_expr.evalf()

                    return abs(num_value) < 1e-3
                except:  # noqa: E722
                    return False

    def equation_equal(self, expression1, expression2):
        # Check if two equations are mathematically equivalent
        # Simplify equations and use sympy for equivalence checking
        def simplify_equation(latex_eq):
            lhs, rhs = latex_eq.split("=")

            lhs_expr = parse_latex(lhs)
            rhs_expr = parse_latex(rhs)

            equation = Eq(lhs_expr, rhs_expr)

            simplified_eq = simplify(equation.lhs - equation.rhs)

            return simplified_eq

        expr1_sym = simplify_equation(expression1)
        expr2_sym = simplify_equation(expression2)

        division_result_1 = simplify(expr1_sym / expr2_sym)
        division_result_2 = simplify(expr2_sym / expr1_sym)

        if (division_result_1.is_Integer and division_result_1 != 0) or (division_result_2.is_Integer and division_result_2 != 0):
            return True
        else:
            return False

    def interval_equal(self, expression1, expression2):
        # Check if two intervals are mathematically equivalent
        def compare_two_interval(inter1, inter2):
            if inter1[0] != inter2[0] or inter1[-1] != inter2[-1]:
                return False

            inter1 = inter1.strip("[]()")
            inter2 = inter2.strip("[]()")

            items_1 = inter1.split(",")
            items_2 = inter2.split(",")

            for item_1, item_2 in zip(items_1, items_2):
                if not self.expression_equal(item_1, item_2):
                    return False
            return True

        interval1 = expression1
        interval2 = expression2

        if interval1 == interval2:
            return True
        else:
            inter_list1 = interval1.split("\\cup")
            inter_list2 = interval2.split("\\cup")

            if len(inter_list1) != len(inter_list2):
                return False
            else:
                for inter1, inter2 in zip(inter_list1, inter_list2):
                    if not compare_two_interval(inter1, inter2):
                        return False
                return True

    def preprocess(self, expression1, expression2):
        # Preprocess expressions to extract and replace special symbols
        def extract_boxed_content(latex_str):
            boxed_matches = re.finditer(r"\\boxed{", latex_str)
            results = ""

            for match in boxed_matches:
                start_index = match.end()
                end_index = start_index
                stack = 1

                while stack > 0 and end_index < len(latex_str):
                    if latex_str[end_index] == "{":
                        stack += 1
                    elif latex_str[end_index] == "}":
                        stack -= 1
                    end_index += 1

                if stack == 0:
                    content = latex_str[start_index : end_index - 1]
                    results += content + ","
                else:
                    raise ValueError("Mismatched braces in LaTeX string.")

            if results == "":
                last_line_ans = latex_str.strip().split("\n")[-1]
                dollar_pattern = r"\$(.*?)\$"
                answers = re.findall(dollar_pattern, last_line_ans)

                if answers:
                    for ans in answers:
                        results += ans + ","
                else:
                    results = latex_str

            return results

        def sepcial_symbol_replace(expression):
            if "\\in " in expression:
                expression = expression.split("\\in ")[1]

            for signal in self.special_signal_map:
                expression = expression.replace(signal, self.special_signal_map[signal])

            expression = expression.strip("\n$,.:;^_=+`!@#$%^&*~，。")

            pattern = r"\\(?:mathrm|mathbf)\{~?([^}]*)\}"
            expression = re.sub(pattern, r"\1", expression)

            return expression

        exp1, exp2 = extract_boxed_content(expression1), extract_boxed_content(expression2)
        exp1, exp2 = sepcial_symbol_replace(exp1), sepcial_symbol_replace(exp2)

        return exp1, exp2

    def can_compute_power(self, expr):
        # Checks if a power expression can be computed
        if isinstance(expr, Pow):
            base, exp = expr.as_base_exp()
            if base.is_number and exp.is_number:
                MAX_EXP = 1000  # Adjust based on computing environment
                if abs(exp.evalf()) > MAX_EXP:
                    return False
                else:
                    return True
            else:
                return False
        else:
            return True  # Not a power expression, can compute
        
def extract_final_answer(response_text_list: List[str], verbose: bool = False, benchmark: str = "BigCodeBench") -> Tuple[List[str], List[str]]:
    
    if benchmark == "BigCodeBench":
        final_answer_list = []
        failed_list = []
        for response_text in response_text_list:
            # This simply checks if there is a code pattern discovered in the response text. This might vary from model to model.
            patterns = [
                    r'```python(.*?)```',
                    r'```(.*?)```',
                    r'^(.*?)$',
                ]
            for pattern in patterns:
                parsed_answer = None
                s2 = re.findall(pattern, response_text, re.DOTALL)
                if s2:
                    parsed_answer = s2[-1].strip() 
                    break
            if parsed_answer == None:
                if verbose:
                    print(f"Error: no code pattern found in the generated output: {response_text}")
                # We add the response text in case the LLM directly generated code without using a code pattern
                final_answer_list.append(response_text)
                failed_list.append(response_text)
                continue
            else:
                final_answer_list.append(parsed_answer)
    else:
        # # get the last 4 lines of the response text
        last_four_lines_list = []
        for response_text in response_text_list:
            response_text = response_text.strip()
            last_four_lines = "".join(response_text.split("\n")[-4:])
            last_four_lines_list.append(last_four_lines)

        # returning `failed_last_line_list` for debugging purposes
        final_answer_list = []
        failed_list = []

        for last_four_lines in last_four_lines_list:
            # extract final answer with latex box: \boxed{}, \fbox{}, \framebox{}, \x08oxed{}
            boxed_answer = last_boxed_only_string(last_four_lines)
            # if no boxed answer is found, use an error message as the placeholder for the final answer
            if not boxed_answer:
                if verbose:
                    print(f"Error: no boxed answer found in the last four lines: {last_four_lines}")
                final_answer_list.append("Error: no boxed answer found")
                failed_list.append(last_four_lines)
                continue
            # if the boxed answer is found, remove the latex box
            else:
                final_answer = remove_boxed(boxed_answer)
                final_answer_list.append(final_answer)

    return final_answer_list, failed_list

def check_code(index: int, solution: str, ground_truth_dict: dict[str, str]) -> dict:
    gt_time_limit = 60
    max_as_limit = 30*1024
    max_data_limit = 30*1024
    max_stack_limit = 10
    min_time_limit = 10
    stat, details = code_utils.untrusted_check(
        solution,
        ground_truth_dict["test"],
        ground_truth_dict["entry_point"],
        max_as_limit,
        max_data_limit,
        max_stack_limit,
        min_time_limit,
        gt_time_limit,
    )
    return {
        "index": index,
        "original_solution": solution,
        "ground_truth": ground_truth_dict,
        "status": stat,
        "details": details,
    }

def compute_accuracy(
    benchmark: Benchmark, ground_truth_list: List[str|dict], final_answer_list: List[str], verbose: bool = False
) -> List[bool]:
    # check if the number of final answers and ground truths are equal
    assert len(final_answer_list) == len(ground_truth_list), "The number of final answers and ground truths should be equal."
    
    if benchmark == "BigCodeBench":
        
        #The target is expected to be the Task ID, so we simply load the dataset and import the test corresponding to the appropriate task ID for evaluating the generated code.
                
        n_workers = max(1, cpu_count() // 2)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for index, solution in enumerate(final_answer_list):
                ground_truth_dict = ground_truth_list[index]
                assert isinstance(ground_truth_dict, dict), f"The ground truth should be a dictionary, but instead is {ground_truth_dict} for {index}."
                assert ("test" in list(ground_truth_dict.keys()) and "entry_point" in list(ground_truth_dict.keys())), "The ground truth dictionary should contain the key 'test'."
                args = (index, solution, ground_truth_dict)
                futures.append(executor.submit(check_code, *args))
        results = {}
        for future in tqdm(as_completed(futures), total=len(final_answer_list)):
            try:
                result = future.result()
                results[result["index"]] = result
            except CancelledError:
                print("A task was cancelled.")
            except Exception as e:
                print(f"An error occurred: {e}")        
        accuracy_list = [False]*len(final_answer_list)
        for index in results.keys():
            if results[index]["status"] == "pass":
                accuracy_list[index] = True 
        print(f"Results: {results}")
        print(f"Accuracy List: {accuracy_list}")
        return accuracy_list
    else:
        # initialize the scorer from OlympiadBench, it's almost compatible with the AIME_2024 and MATH benchmarks
        # excpet for cases like \$18.90 and 18.90, which can be handled by `is_equiv` but not `AutoScoringJudge`
        # in general, the `AutoScoringJudge` is more robust and can handle more cases, see test cases in `evaluation.py`
        scorer = AutoScoringJudge()
        accuracy_result_list = []

        # use the corresponding accuracy metric for the benchmark
        for index, line in enumerate(final_answer_list):
            ground_truth = ground_truth_list[index]
            final_answer = final_answer_list[index]

            # if failed to extract the final answer, set the accuracy to False
            if final_answer == "Error: no boxed answer found":
                accuracy_result = False
            # for AIME 2024, `AutoScoringJudge` is completely compatible
            elif benchmark == "AIME_2024":
                accuracy_result = scorer.judge(ground_truth, final_answer)
            # for MATH, use both `is_equiv` and `AutoScoringJudge` for more robust equivalence checking
            elif benchmark == "MATH-500":
                accuracy_result = is_equiv(ground_truth, final_answer) or scorer.judge(ground_truth, final_answer)
            # for OlympiadBench, use the native `AutoScoringJudge`
            elif benchmark == "OlympiadBench-674-MATH_TO_EN":
                ground_truth_answer, precision = ground_truth
                if not precision:
                    accuracy_result = scorer.judge(ground_truth_answer, final_answer)
                else:
                    accuracy_result = scorer.judge(ground_truth_answer, final_answer, precision=float(precision))
            # other benchmarks are not supported
            else:
                raise ValueError(f"Benchmark: {benchmark} is not supported.")

            # if `verbose` is set True, print the final answer and ground truth
            if verbose:
                print(f"Ground Truth: {ground_truth}, Final Answer: {final_answer}, Accuracy: {accuracy_result}")

            accuracy_result_list.append(accuracy_result)

    return accuracy_result_list


def evaluate(benchmark, responses, ground_truth_list, verbose=False):

    # construct the final answer list
    final_answer_list, failed_list = extract_final_answer(responses, verbose, benchmark)

    # compute the accuracy result list
    accuracy_result_list = compute_accuracy(benchmark, ground_truth_list, final_answer_list, verbose)
    
    return accuracy_result_list


# EXAMPLE USAGE: python evaluation.py
def main():
    scorer = AutoScoringJudge()

    test_cases = [
        {
            "response_1": "This is my final answer\n\\boxed{10^{10^{10/10}}}}",
            "response_2": "This is my final answer\n\\boxed{10^{10}}",
        },
        {
            "response_1": "This is my final answer\n\\boxed{089}",
            "response_2": "This is my final answer\n\\boxed{89}",
        },
        {
            "response_1": "This is my final answer\n\\fbox{\\frac{1}{2}}",
            "response_2": "This is my final answer\n\\fbox{0.5}",
        },
        {
            "response_1": "This is my final answer\n\\framebox{\\frac{1}{2}}",
            "response_2": "This is my final answer\n\\framebox{\\frac{2}{4}}",
        },
        {
            "response_1": "\\boxed{540}",
            "response_2": "\\boxed{540.0}",
        },
        {
            "response_1": "\\boxed{\\$18.90}",
            "response_2": "\\boxed{18.90}",
        },
        {
            "response_1": "\\boxed{.9}",
            "response_2": "\\boxed{0.9}",
        },
        {
            "response_1": "\\boxed{4,0,4}",
            "response_2": "\\boxed{0,4,4}",
        },
        {
            "response_1": "\\boxed{160,000}",
            "response_2": "\\boxed{160000}",
        },
        {
            "response_1": "aaa\boxed{\\frac{1}{2}}",
            "response_2": "aaa\boxed{0.5}", 
        },
    ]

    for test_case in test_cases:
        response_1 = test_case["response_1"]
        response_2 = test_case["response_2"]

        extracted_1 = last_boxed_only_string(response_1)
        extracted_2 = last_boxed_only_string(response_2)
        final_answer_1 = remove_boxed(extracted_1)
        final_answer_2 = remove_boxed(extracted_2)

        res_math = is_equiv(final_answer_1, final_answer_2)
        res_olympiadbench = scorer.judge(final_answer_1, final_answer_2)
        res_olympiadbench_direct_from_response = scorer.judge(response_1, response_2)

        print("=======================================================")
        print(f"Response 1: {response_1}")
        print(f"Response 2: {response_2}")
        print(f"Boxed Answer 1: {extracted_1}")
        print(f"Boxed Answer 2: {extracted_2}")
        print(f"Final Answer 1: {final_answer_1}")
        print(f"Final Answer 2: {final_answer_2}")
        print(f"Judged by the `is_equiv` function from final answer: {res_math}")
        print(f"Judged by the `AutoScoringJudge` from final answer: {res_olympiadbench}")
        print(f"Judged by the `AutoScoringJudge` directly from responses: {res_olympiadbench_direct_from_response}")
        print("=======================================================")


if __name__ == "__main__":
    main()
