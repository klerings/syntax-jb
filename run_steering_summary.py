import pandas as pd
import pickle
import os
import argparse

from helper import get_valid_models

VALID_MODELS = get_valid_models()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prompt dataset analysis.")
    parser.add_argument(
        "model_name",
        type=str,
        choices=VALID_MODELS,
        help="Name of the model whose judge results are to be processed.",
    )
    args = parser.parse_args()

    model_name = args.model_name
    if model_name == "gemma2-2b-greedy":
        model_name = "gemma2-2b"
    elif model_name == "qwen2.5_7b_greedy":
        model_name = "qwen2.5_7b"
    elif model_name == "llama3.1-8b-greedy":
        model_name = "llama3.1-8b"
    else:
        raise NotImplementedError(f"{model_name} currently not supported.")

    results_dir = f"steering/{model_name}/wildguard_eval"

    steering_files = [
        file for file in os.listdir(results_dir) if file.startswith("wildguard_all")
    ]

    for case_file in steering_files:
        # if "random" not in case_file:
        #    continue

        df = pd.read_csv(f"{results_dir}/{case_file}")

        imperative_vars = ["imperative"]
        non_imperative_vars = [
            "declarative",
            "conditional",
            "nominalized",
            "present_active",
            "past_active",
            "future_active",
            "present_passive",
        ]

        for syntax_group in [imperative_vars, non_imperative_vars]:
            print(case_file[25:])
            print(f"for: {syntax_group}")
            df_s = df[df["syntax_label"].isin(syntax_group)]

            # a)  try to jailbreak
            harmful_refused = df_s[
                (df_s["prompt_mode"] == "harmful")
                & (df_s["original_label"] == "refusal")
            ]
            TP = len(harmful_refused)

            # b) try to refuse
            harmful_complied = df_s[
                (df_s["prompt_mode"] == "harmful")
                & (df_s["original_label"] == "complied")
            ]
            FN = len(harmful_complied)

            # c) try to jailbreak
            harmless_refused = df_s[
                (df_s["prompt_mode"] == "harmless")
                & (df_s["original_label"] == "refusal")
            ]
            FP = len(harmless_refused)

            # d) try to refuse
            harmless_complied = df_s[
                (df_s["prompt_mode"] == "harmless")
                & (df_s["original_label"] == "complied")
            ]
            TN = len(harmless_complied)

            # steering success a)
            steered_harmful_refused_to_compliance = harmful_refused[
                harmful_refused["response_refusal"] == "compliance"
            ]
            steered_harmful_refused_to_compliance_strict = harmful_refused[
                (harmful_refused["response_refusal"] == "compliance")
                & (harmful_refused["response_harmfulness"] == "harmful")
            ]
            TP_steered = len(steered_harmful_refused_to_compliance)
            TP_steered_strict = len(steered_harmful_refused_to_compliance_strict)

            # steering success b)
            steered_harmful_complied_to_refuse = harmful_complied[
                harmful_complied["response_refusal"] == "refusal"
            ]
            steered_harmful_complied_to_refuse_strict = harmful_complied[
                (harmful_complied["response_refusal"] == "refusal")
                & (harmful_complied["response_harmfulness"] == "unharmful")
            ]
            FN_steered = len(steered_harmful_complied_to_refuse)
            FN_steered_strict = len(steered_harmful_complied_to_refuse_strict)

            # steering success c)
            steered_harmless_refused_to_compliance = harmless_refused[
                harmless_refused["response_refusal"] == "compliance"
            ]
            steered_harmless_refused_to_compliance_strict = harmless_refused[
                (harmless_refused["response_refusal"] == "compliance")
                & (harmless_refused["response_harmfulness"] == "harmful")
            ]
            FP_steered = len(steered_harmless_refused_to_compliance)
            FP_steered_strict = len(steered_harmless_refused_to_compliance_strict)

            # steering success d)
            steered_harmless_complied_to_refuse = harmless_complied[
                harmless_complied["response_refusal"] == "refusal"
            ]
            steered_harmless_complied_to_refuse_strict = harmless_complied[
                (harmless_complied["response_refusal"] == "refusal")
                & (harmless_complied["response_harmfulness"] == "unharmful")
            ]
            TN_steered = len(steered_harmless_complied_to_refuse)
            TN_steered_strict = len(steered_harmless_complied_to_refuse_strict)

            # print(f"TP: {TP}\nFN: {FN}\nFP: {FP}\nTN: {TN}")

            print("HARMFUL Prompts")
            if TP != 0:
                print(
                    f"- originally refused: {TP} -> {round((TP_steered/TP)*100,2)} % ({round((TP_steered_strict/TP)*100,2)}) flipped"
                )
            if FN != 0:
                print(
                    f"- originally complied: {FN} -> {round((FN_steered/FN)*100,2)} % ({round((FN_steered_strict/FN)*100,2)}) flipped"
                )

            print("HARMLESS Prompts")
            if FP != 0:
                print(
                    f"- originally refused: {FP} -> {round((FP_steered/FP)*100,2)} % ({round((FP_steered_strict/FP)*100,2)}) flipped"
                )
            if TN != 0:
                print(
                    f"- originally complied: {TN} -> {round((TN_steered/TN)*100,2)} % ({round((TN_steered_strict/TN)*100,2)}) flipped"
                )

            print("ALL PROMPTS")
            if (TP + FP) != 0:
                print(
                    f"- originally refused: {TP + FP} -> {round(((TP_steered+FP_steered)/(TP+FP))*100,2)} % ({round(((TP_steered_strict+FP_steered_strict)/(TP+FP))*100,2)}) flipped"
                )
            if (TN + FN) != 0:
                print(
                    f"- originally complied: {TN + FN} -> {round(((TN_steered+FN_steered)/(TN+FN))*100,2)} % ({round(((TN_steered_strict+FN_steered_strict)/(TN+FN))*100,2)}) flipped"
                )

            print("------------")
