import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from config import model_index_map
from data_utils import search_metadata
from graph_utils import graph_generic, graph_cot_data, generate_singular_plot, graph_stepwise_comparison, \
    modality_to_label_map, modality_to_color_map, graph_dataset_comparison


def generate():
    # MMLU_full()
    # gpt4_32_vs_8()

    # answer_first_vs_suppressed()
    # explanation_first_vs_cot()
    #
    # prompt_comparison()
    # coin_flip_supp_results()
    #
    # coin_flip_full_results()
    # multiarith_full_results()
    # gsm8k_full_results()
    # aqua_full_results()
    # mmlu_full_results()
    #
    # G4_stepwise_suppression()
    # G35_stepwise_suppression()

    # G35_two_stage()

    G4_stepwise_full()
    # G35_stepwise_full()
    # G3_stepwise_full()
    #
    # G4_all_suppression()
    # G35_all_suppression()
    #
    G4_all()
    # G35_all()
    # G3_all()


def answer_first_vs_suppressed():
    x_labels = ["Zero Shot", "Suppressed CoT", "Answer First"]
    chart_labels = ["MultiArith", "GSM8k", "Aqua-RAT", "Coin Flip", "MMLU"]

    # Graph GPT-4 Answer First results
    gpt_4_results = search_metadata(models="gpt-4",
                                    modalities=["zero_shot", "suppressed_cot", "answer_first"],
                                    datasets=["stepwise"])
    data = pd.melt(gpt_4_results, id_vars=["Model", "Modality", "Dataset Index", "Modality Index", "Dataset"],
                   value_vars=["Total Accuracy"], value_name="Accuracy")
    graph_generic(title="GPT-4 Performance, Answer First vs Suppressed CoT, All Datasets",
                  data=data, group_by="Dataset Index",
                  output_path=r"AnswerFirstAnalysis\GPT4-ans-first-results",
                  chart_labels=chart_labels, x_labels=x_labels, figsize=(15, 10),
                  plot_size=(2, 3), sort_by="Modality Index")

    # CoT Quant results
    data = pd.melt(gpt_4_results, id_vars=["Model", "Modality", "Dataset Index", "Modality Index", "Dataset"],
                   value_vars=["Total Accuracy", "Answers Containing CoT", "CoT Accuracy", "Non-CoT Accuracy"],
                   var_name="Metric",
                   value_name="Percentage").sort_values("Modality Index")

    graph_cot_data(title="GPT-4 CoT Quantification, Answer First vs Suppressed CoT, All Datasets",
                   data=data,
                   figsize=(15, 10),
                   plot_size=(2, 3),
                   hue="Metric",
                   x="Modality",
                   group_by="Dataset Index",
                   xtick_labels=x_labels,
                   output_path=r"AnswerFirstAnalysis\gpt4-ans-first-cot-quant")

    # Graph GPT-3.5 Ans First results
    gpt_35_results = search_metadata(models="gpt-3.5-turbo", modalities=["zero_shot", "suppressed_cot",
                                                                        "answer_first"])
    data = pd.melt(gpt_35_results, id_vars=["Model", "Modality", "Dataset Index", "Modality Index", "Dataset"],
                   value_vars=["Total Accuracy"], value_name="Accuracy")
    graph_generic(title="GPT-3.5 Performance, Answer First vs Suppressed CoT, All Datasets",
                  data=data, group_by="Dataset Index",
                  output_path=r"AnswerFirstAnalysis\GPT35-ans-first-results",
                  chart_labels=chart_labels, x_labels=x_labels, figsize=(15, 10),
                  plot_size=(2, 3), sort_by="Modality Index")

    # CoT Quant results
    data = pd.melt(gpt_35_results, id_vars=["Model", "Modality", "Dataset Index", "Modality Index", "Dataset"],
                   value_vars=["Total Accuracy", "Answers Containing CoT", "CoT Accuracy", "Non-CoT Accuracy"],
                   var_name="Metric",
                   value_name="Percentage").sort_values("Modality Index")

    graph_cot_data(title="GPT-3.5 CoT Quantification, Answer First vs Suppressed CoT, All Datasets",
                   data=data,
                   figsize=(15, 10),
                   plot_size=(2, 3),
                   hue="Metric",
                   x="Modality",
                   group_by="Dataset Index",
                   xtick_labels=x_labels,
                   output_path=r"AnswerFirstAnalysis\gpt35-ans-first-cot-quant")

    # Graph GPT-4 Answer First Stepwise results

    gpt_4_results = search_metadata(models="gpt-4",
                                    modalities=["zero_shot", "suppressed_cot", "answer_first"],
                                    datasets=["stepwise"])
    data = pd.melt(gpt_4_results, id_vars=["Model", "Modality", "Modality Index", "Dataset", "ci_upper", "ci_lower"],
                   value_vars=["Total Accuracy"], value_name="Accuracy").sort_values("Dataset")

    graph_generic(
        title="GPT-4 Performance, Answer First vs Suppressed CoT, Stepwise Dataset",
        data=data, x="Dataset", group_by="Modality Index",
        output_path=r"AnswerFirstAnalysis\gpt4-step-ans-first", figsize=(15, 5),
        chart_labels=x_labels, x_labels=[str(i) + " Step" for i in range(1, 10)], plot_size=(1, 3))

    # CoT Quantification
    data = pd.melt(gpt_4_results,
                   id_vars=["Step", "Modality", "Modality Index", "Model", "Model Index"],
                   value_vars=["Total Accuracy", "Answers Containing CoT", "CoT Accuracy", "Non-CoT Accuracy"],
                   var_name="Metric",
                   value_name="Percentage")

    graph_cot_data(title="GPT-4 Performance, Answer First vs Suppressed CoT, Stepwise Dataset",
                   data=data,
                   x="Step",
                   figsize=(15, 5),
                   plot_size=(1, 3),
                   xtick_labels=[str(i) + " Step" for i in range(1, 10)],
                   output_path=r"AnswerFirstAnalysis\gpt4-step-ans-first-cot-quant")

    # Graph GPT-3.5 Answer First Stepwise results

    gpt_35_results = search_metadata(models="gpt-3.5-turbo",
                                     modalities=["zero_shot", "suppressed_cot", "answer_first"],
                                     datasets=["stepwise"])
    data = pd.melt(gpt_35_results, id_vars=["Model", "Modality", "Modality Index", "Dataset", "ci_upper", "ci_lower"],
                   value_vars=["Total Accuracy"], value_name="Accuracy").sort_values("Dataset")

    graph_generic(
        title="GPT-3.5 Performance, Answer First vs Suppressed CoT, Stepwise Dataset",
        data=data, x="Dataset", group_by="Modality Index",
        output_path=r"AnswerFirstAnalysis\gpt35-step-ans-first", figsize=(15, 5),
        chart_labels=x_labels, x_labels=[str(i) + " Step" for i in range(1, 10)], plot_size=(1, 3))

    # CoT Quantification
    data = pd.melt(gpt_35_results,
                   id_vars=["Step", "Modality", "Modality Index", "Model", "Model Index", "ci_upper", "ci_lower"],
                   value_vars=["Total Accuracy", "Answers Containing CoT", "CoT Accuracy", "Non-CoT Accuracy"],
                   var_name="Metric",
                   value_name="Percentage")

    graph_cot_data(title="GPT-3.5 Performance, Answer First vs Suppressed CoT, Stepwise Dataset",
                   data=data,
                   x="Step",
                   figsize=(15, 5),
                   plot_size=(1, 3),
                   xtick_labels=[str(i) + " Step" for i in range(1, 10)],
                   output_path=r"AnswerFirstAnalysis\gpt35-step-ans-first-cot-quant")


def explanation_first_vs_cot():
    x_labels = ["Zero Shot", "Zero Shot CoT", "Explanation First"]
    chart_labels = ["MultiArith", "GSM8k", "Aqua-RAT", "Coin Flip", "MMLU"]

    # Graph GPT-4 Exp First results
    gpt_4_results = search_metadata(models="gpt-4",
                                    modalities=["zero_shot", "zero_shot_cot", "explanation_first"])
    data = pd.melt(gpt_4_results, id_vars=["Model", "Modality", "Dataset Index", "Modality Index", "Dataset"],
                   value_vars=["Total Accuracy"], value_name="Accuracy")
    graph_generic(title="GPT-4 Performance, Explanation First vs Zero Shot CoT, All Datasets",
                  data=data, group_by="Dataset Index",
                  output_path=r"ExplanationFirstAnalysis\GPT4-exp-first-results",
                  chart_labels=chart_labels, x_labels=x_labels, figsize=(15, 10),
                  plot_size=(2, 3), sort_by="Modality Index")

    # CoT Quant results
    data = pd.melt(gpt_4_results, id_vars=["Model", "Modality", "Dataset Index", "Modality Index", "Dataset"],
                   value_vars=["Total Accuracy", "Answers Containing CoT", "CoT Accuracy", "Non-CoT Accuracy"],
                   var_name="Metric",
                   value_name="Percentage").sort_values("Modality Index")

    graph_cot_data(title="GPT-4 CoT Quantification, Explanation First vs Zero Shot CoT, All Datasets",
                   data=data,
                   figsize=(15, 10),
                   plot_size=(2, 3),
                   hue="Metric",
                   x="Modality",
                   group_by="Dataset Index",
                   xtick_labels=x_labels,
                   output_path=r"ExplanationFirstAnalysis\gpt4-exp-first-cot-quant")

    # Graph GPT-3.5 Exp First results
    gpt_35_results = search_metadata(models="gpt-3.5-turbo",
                                     modalities=["zero_shot", "zero_shot_cot", "explanation_first"])
    data = pd.melt(gpt_35_results, id_vars=["Model", "Modality", "Dataset Index", "Modality Index", "Dataset"],
                   value_vars=["Total Accuracy"], value_name="Accuracy")
    graph_generic(title="GPT-3.5 Performance, Explanation First vs Suppressed CoT, All Datasets",
                  data=data, group_by="Dataset Index",
                  output_path=r"ExplanationFirstAnalysis\GPT35-exp-first-results",
                  chart_labels=chart_labels, x_labels=x_labels, figsize=(15, 10),
                  plot_size=(2, 3), sort_by="Modality Index")

    # CoT Quant results
    data = pd.melt(gpt_35_results, id_vars=["Model", "Modality", "Dataset Index", "Modality Index", "Dataset"],
                   value_vars=["Total Accuracy", "Answers Containing CoT", "CoT Accuracy", "Non-CoT Accuracy"],
                   var_name="Metric",
                   value_name="Percentage").sort_values("Modality Index")

    graph_cot_data(title="GPT-3.5 CoT Quantification, Explanation First vs Zero Shot CoT, All Datasets",
                   data=data,
                   figsize=(15, 10),
                   plot_size=(2, 3),
                   hue="Metric",
                   x="Modality",
                   group_by="Dataset Index",
                   xtick_labels=x_labels,
                   output_path=r"ExplanationFirstAnalysis\gpt35-exp-first-cot-quant")

    # Graph GPT-4 Explanation First Stepwise results
    gpt_4_results = search_metadata(models="gpt-4",
                                    modalities=["zero_shot", "zero_shot_cot", "explanation_first"],
                                    datasets=["stepwise"])
    data = pd.melt(gpt_4_results, id_vars=["Model", "Modality", "Modality Index", "Dataset", "ci_upper", "ci_lower"],
                   value_vars=["Total Accuracy"], value_name="Accuracy").sort_values("Dataset")

    graph_generic(
        title="GPT-4 Performance, Explanation First vs Suppressed CoT, Stepwise Dataset",
        data=data, x="Dataset", group_by="Modality Index",
        output_path=r"ExplanationFirstAnalysis\gpt4-step-exp-first", figsize=(15, 5),
        chart_labels=x_labels, x_labels=[str(i) + " Step" for i in range(1, 10)], plot_size=(1, 3))

    # CoT Quantification
    data = pd.melt(gpt_4_results,
                   id_vars=["Steps", "Modality", "Modality Index", "Model", "Model Index", "ci_upper", "ci_lower"],
                   value_vars=["Total Accuracy", "Answers Containing CoT", "CoT Accuracy", "Non-CoT Accuracy"],
                   var_name="Metric",
                   value_name="Percentage")

    graph_cot_data(title="GPT-4 Performance, Explanation First vs Suppressed CoT, Stepwise Dataset",
                   data=data,
                   x="Steps",
                   figsize=(15, 5),
                   plot_size=(1, 3),
                   chart_labels=x_labels,
                   xtick_labels=[str(i) + " Step" for i in range(1, 10)],
                   output_path=r"ExplanationFirstAnalysis\gpt4-step-exp-first-cot-quant")

    # Graph GPT-3.5 Explanation First Stepwise results
    gpt_35_results = search_metadata(models="gpt-3.5-turbo",
                                     modalities=["zero_shot", "zero_shot_cot", "explanation_first"],
                                     datasets=["stepwise"])
    data = pd.melt(gpt_35_results, id_vars=["Model", "Modality", "Modality Index", "Dataset", "ci_upper", "ci_lower"],
                   value_vars=["Total Accuracy"], value_name="Accuracy").sort_values("Dataset")

    graph_generic(
        title="GPT-3.5 Performance, Explanation First vs Suppressed CoT, Stepwise Dataset",
        data=data, x="Dataset", group_by="Modality Index",
        output_path=r"ExplanationFirstAnalysis\gpt35-step-exp-first", figsize=(15, 5),
        chart_labels=x_labels, x_labels=[str(i) + " Step" for i in range(1, 10)], plot_size=(1, 3))

    # CoT Quantification
    data = pd.melt(gpt_35_results,
                   id_vars=["Step", "Modality", "Modality Index", "Model", "Model Index", "ci_upper", "ci_lower"],
                   value_vars=["Total Accuracy", "Answers Containing CoT", "CoT Accuracy", "Non-CoT Accuracy"],
                   var_name="Metric",
                   value_name="Percentage")

    graph_cot_data(title="GPT-3.5 Performance, Explanation First vs Suppressed CoT, Stepwise Dataset",
                   data=data,
                   x="Step",
                   figsize=(15, 5),
                   plot_size=(1, 3),
                   xtick_labels=["Zero Shot", "Suppressed CoT", "Answer First"],
                   output_path=r"ExplanationFirstAnalysis\gpt35-step-exp-first-cot-quant")


def multiarith_full_results():
    # Graph GPT-4 Coin CoT results
    modalities = ["zero_shot", "zero_shot_cot", "suppressed_cot", "explanation_first", "answer_first"]

    gpt_4_results = search_metadata(models=["gpt-4"], modalities=modalities,
                                    datasets=["multiarith"])

    gpt_35_results = search_metadata(models=["gpt-3.5-turbo"],
                                     modalities=modalities,
                                     datasets=["multiarith"])
    gpt_3_results = search_metadata(models=["text-davinci-002"],
                                    modalities=modalities,
                                    datasets=["multiarith"])
    data = pd.concat([gpt_3_results, gpt_35_results, gpt_4_results])

    compare = pd.melt(data, id_vars=["Model", "Modality", "Model Index", "Modality Index"],
                      value_vars=["Total Accuracy"], value_name="Accuracy").sort_values("Modality Index")
    graph_generic(title="MultiArith Model Comparison",
                  data=compare, group_by="Model Index", sort_by="Modality Index",
                  output_path=r"DatasetAnalysis\multiarith-full-results",
                  chart_labels=["text-davinci-002", "GPT-3.5", "GPT-4"],
                  x_labels=[modality_to_label_map[i] for i in modalities],
                  palette=[modality_to_color_map[i] for i in modalities],
                  figsize=(15, 5), plot_size=(1, 3))

    # CoT Quant results
    data = pd.melt(data, id_vars=["Model", "Modality", "Dataset Index", "Modality Index", "Dataset"],
                   value_vars=["Total Accuracy", "Answers Containing CoT", "CoT Accuracy", "Non-CoT Accuracy"],
                   var_name="Metric",
                   value_name="Percentage")

    graph_cot_data(title="MultiArith CoT Quantification, All Models, All Modalities",
                   data=data,
                   figsize=(15, 5),
                   plot_size=(2, 3),
                   hue="Metric",
                   x="Dataset",
                   output_path="multiarith-full-cot-quant")


def gsm8k_full_results():
    # Graph GPT-4 Coin CoT results
    modalities = ["zero_shot", "zero_shot_cot", "suppressed_cot", "explanation_first", "answer_first"]
    gpt_4_results = search_metadata(models="gpt-4",
                                    modalities=modalities,
                                    datasets=["gsm8k"])
    gpt_35_results = search_metadata(models="gpt-3.5-turbo",
                                     modalities=modalities,
                                     datasets=["gsm8k"])
    gpt_3_results = search_metadata(models="text-davinci-002",
                                    modalities=modalities,
                                    datasets=["gsm8k"])

    data = pd.concat([gpt_3_results, gpt_35_results, gpt_4_results])
    compare = pd.melt(data, id_vars=["Model", "Modality", "Model Index", "Modality Index"],
                      value_vars=["Total Accuracy"], value_name="Accuracy").sort_values("Modality Index")

    graph_generic(title="GSM8k Model Comparison",
                  data=compare, group_by="Model Index", sort_by="Modality Index",
                  output_path=r"DatasetAnalysis\gsm8k-full-results",
                  chart_labels=["text-davinci-002", "GPT-3.5", "GPT-4"],
                  x_labels=[modality_to_label_map[i] for i in modalities],
                  palette=[modality_to_color_map[i] for i in modalities],
                  figsize=(15, 5), plot_size=(1, 3))

    # CoT Quant results
    data = pd.melt(data, id_vars=["Model", "Modality", "Dataset Index", "Modality Index", "Dataset"],
                   value_vars=["Total Accuracy", "Answers Containing CoT", "CoT Accuracy", "Non-CoT Accuracy"],
                   var_name="Metric",
                   value_name="Percentage")

    graph_cot_data(title="GSM8k CoT Quantification, All Models, All Modalities",
                   data=data,
                   figsize=(15, 5),
                   plot_size=(2, 3),
                   hue="Metric",
                   x="Dataset",
                   output_path="gsm8k-full-cot-quant")


def aqua_full_results():
    # Graph GPT-4 Coin CoT results
    modalities = ["zero_shot", "zero_shot_cot", "suppressed_cot", "explanation_first", "answer_first"]

    gpt_4_results = search_metadata(models="gpt-4",
                                    modalities=modalities,
                                    datasets=["aqua"])
    gpt_35_results = search_metadata(models="gpt-3.5-turbo",
                                     modalities=modalities,
                                     datasets=["aqua"])
    gpt_3_results = search_metadata(models="text-davinci-002",
                                    modalities=modalities,
                                    datasets=["aqua"])

    data = pd.concat([gpt_3_results, gpt_35_results, gpt_4_results])

    compare = pd.melt(data, id_vars=["Model", "Modality", "Model Index", "Modality Index"],
                      value_vars=["Total Accuracy"], value_name="Accuracy").sort_values("Modality Index")

    graph_generic(title="AQuA Model Comparison",
                  data=compare, group_by="Model Index", sort_by="Modality Index",
                  output_path=r"DatasetAnalysis\aqua-full-results",
                  chart_labels=["text-davinci-002", "GPT-3.5", "GPT-4"],
                  x_labels=[modality_to_label_map[i] for i in modalities],
                  palette=[modality_to_color_map[i] for i in modalities],
                  figsize=(15, 5), plot_size=(1, 3))

    # CoT Quant results
    data = pd.melt(data, id_vars=["Model", "Modality", "Dataset Index", "Modality Index", "Dataset"],
                   value_vars=["Total Accuracy", "Answers Containing CoT", "CoT Accuracy", "Non-CoT Accuracy"],
                   var_name="Metric",
                   value_name="Percentage")

    graph_cot_data(title="AQuA CoT Quantification, All Models, All Modalities",
                   data=data,
                   figsize=(15, 5),
                   plot_size=(2, 3),
                   hue="Metric",
                   x="Dataset",
                   output_path="aqua-full-cot-quant")


def mmlu_full_results():
    # Graph GPT-4 Coin CoT results
    modalities = ["zero_shot", "zero_shot_cot", "suppressed_cot", "explanation_first", "answer_first"]

    gpt_4_results = search_metadata(models="gpt-4",
                                    modalities=modalities,
                                    datasets=["mmlu"])
    gpt_35_results = search_metadata(models="gpt-3.5-turbo",
                                     modalities=modalities,
                                     datasets=["mmlu"])

    data = pd.concat([gpt_35_results, gpt_4_results])
    compare = pd.melt(data, id_vars=["Model", "Modality", "Model Index", "Modality Index"],
                      value_vars=["Total Accuracy"], value_name="Accuracy").sort_values("Modality Index")
    graph_generic(title="MMLU Model Comparison",
                  data=compare, group_by="Model Index", sort_by="Modality Index",
                  output_path=r"DatasetAnalysis\mmlu-full-results", chart_labels=["GPT-3.5", "GPT-4"],
                  x_labels=[modality_to_label_map[i] for i in modalities],
                  palette=[modality_to_color_map[i] for i in modalities],
                  figsize=(15, 5), plot_size=(1, 2))

    # CoT Quant results
    data = pd.melt(data, id_vars=["Model", "Modality", "Dataset Index", "Modality Index", "Dataset"],
                   value_vars=["Total Accuracy", "Answers Containing CoT", "CoT Accuracy", "Non-CoT Accuracy"],
                   var_name="Metric",
                   value_name="Percentage")

    graph_cot_data(title="MMLU CoT Quantification, GPT-4 and GPT-3.5, All Modalities",
                   data=data,
                   figsize=(10, 5),
                   plot_size=(2, 3),
                   hue="Metric",
                   x="Dataset",
                   output_path="mmlu-full-cot-quant")


def G4_all():
    modalities = ["zero_shot", "zero_shot_cot", "suppressed_cot", "explanation_first", "answer_first"]

    # Graph GPT-4 All dataset results
    gpt_4_results = search_metadata(models="gpt-4", modalities=modalities)

    data = pd.melt(gpt_4_results, id_vars=["Model", "Modality", "Dataset Index", "Modality Index", "Dataset"],
                   value_vars=["Total Accuracy"], value_name="Accuracy")

    graph_dataset_comparison(title="GPT-4 Performance, All Modalities, All Datasets",
                             data=data,
                             modalities=modalities,
                             figsize=(10, 3),
                             plot_size=(1, 5),
                             sort_by="Modality Index",
                             add_legend=True,
                             output_path="GPT4/GPT4-full-results",
                             )

    # CoT Quant results
    data = pd.melt(gpt_4_results, id_vars=["Model", "Modality", "Dataset Index", "Modality Index", "Dataset"],
                   value_vars=["Total Accuracy", "Answers Containing CoT", "CoT Accuracy", "Non-CoT Accuracy"],
                   var_name="Metric",
                   value_name="Percentage").sort_values("Modality Index")

    graph_cot_data(title="GPT-4 CoT Quantification, All Modalities, All Datasets",
                   data=data,
                   figsize=(10, 10),
                   plot_size=(2, 3),
                   hue="Metric",
                   x="Modality",
                   group_by="Dataset Index",
                   output_path="GPT4/gpt4-full-cot-quant")


def G4_all_suppression():
    # Graph GPT-4 all non-step dataset suppression results
    modalities = ["zero_shot", "zero_shot_cot", "suppressed_cot"]

    gpt_4_results = search_metadata(models="gpt-4",
                                    modalities=modalities)
    data = pd.melt(gpt_4_results,
                   id_vars=["Model", "Modality", "Dataset Index", "Modality Index", "Dataset"],
                   value_vars=["Total Accuracy"],
                   value_name="Accuracy").sort_values("Modality Index")
    graph_dataset_comparison(title="GPT-4 Performance, Zero Shot vs CoT Suppression, All Datasets",
                             data=data,
                             output_path="GPT4/GPT4-supp-results",
                             modalities=modalities,
                             figsize=(7, 5),
                             plot_size=(2, 3))

    # CoT Quantification
    data = pd.melt(gpt_4_results,
                   id_vars=["Dataset", "Modality", "Modality Index", "Model", "Model Index", "Dataset Index"],
                   value_vars=["Total Accuracy", "Answers Containing CoT", "CoT Accuracy", "Non-CoT Accuracy"],
                   var_name="Metric",
                   value_name="Percentage").sort_values("Modality Index")

    graph_cot_data(title="GPT-4 CoT Quantification, All Datasets",
                   data=data,
                   x="Modality",
                   group_by="Dataset Index",
                   figsize=(15, 5),
                   plot_size=(2, 3),
                   output_path="GPT4/gpt4-supp-cot-quant")


def G4_stepwise_suppression():
    # Graph GPT-4 Stepwise Suppression results
    modalities = ["zero_shot", "suppressed_cot"]

    gpt_4_results = search_metadata(models="gpt-4", modalities=modalities,
                                    datasets=["stepwise"])
    gpt_4_results = gpt_4_results.loc[gpt_4_results['Step'] <= 23]

    data = pd.melt(gpt_4_results,
                   id_vars=["Model", "Modality", "Modality Index", "Dataset", "Step", "ci_upper", "ci_lower"],
                   value_vars=["Total Accuracy"], value_name="Accuracy").sort_values(by=["Step", "Modality Index"])

    graph_stepwise_comparison(
        title="GPT-4 Chain-of-Thought Performance on Stepwise Dataset",
        data=data, output_path="GPT4/Stepwise/gpt4-step-suppression",
        modalities=modalities)

    # CoT Quantification
    data = pd.melt(gpt_4_results,
                   id_vars=["Step", "Modality", "Modality Index", "Model", "Model Index", "ci_upper", "ci_lower"],
                   value_vars=["Total Accuracy", "Answers Containing CoT", "CoT Accuracy", "Non-CoT Accuracy"],
                   var_name="Metric",
                   value_name="Percentage")

    graph_cot_data(title="GPT-4 CoT Quantification, Stepwise Dataset",
                   data=data,
                   x="Step",
                   xtick_labels=[str(i) + " Step" for i in range(1, 21)],
                   figsize=(15, 5),
                   plot_size=(1, 3),
                   output_path="GPT4/Stepwise/gpt4-step-supp-cot-quant")


def G4_stepwise_full():
    # Graph GPT-4 FULL Stepwise results
    modalities = ["zero_shot", "zero_shot_cot", "suppressed_cot", "explanation_first", "answer_first"]
    gpt_4_results = search_metadata(models="gpt-4", modalities=modalities,
                                    datasets=["stepwise"])

    gpt_4_results = gpt_4_results.loc[gpt_4_results['Step'] <= 9]

    data = pd.melt(gpt_4_results,
                   id_vars=["Model", "Modality", "Modality Index", "Dataset", "Step", "ci_upper", "ci_lower"],
                   value_vars=["Total Accuracy"], value_name="Accuracy").sort_values(by=["Step", "Dataset"])

    graph_stepwise_comparison(
        title="GPT-4 Performance on Stepwise Dataset",
        data=data, output_path="GPT4/Stepwise/gpt4-step-suppression",
        modalities=modalities)

    # CoT Quantification
    data = pd.melt(gpt_4_results,
                   id_vars=["Step", "Modality", "Modality Index", "Model", "Model Index", "ci_upper", "ci_lower"],
                   value_vars=["Total Accuracy", "Answers Containing CoT", "CoT Accuracy", "Non-CoT Accuracy"],
                   var_name="Metric",
                   value_name="Percentage")

    graph_cot_data(title="GPT-4 CoT Quantification, All Modalities, Stepwise Dataset",
                   data=data,
                   x="Step",
                   xtick_labels=[str(i) + " Step" for i in range(1, 21)],
                   figsize=(15, 10),
                   plot_size=(2, 3),
                   output_path="GPT4/Stepwise/gpt4-step-cot-quant")


def G35_all():
    modalities = ["zero_shot", "zero_shot_cot", "suppressed_cot", "explanation_first", "answer_first"]

    # Graph GPT-3.5 All dataset results
    gpt_35_results = search_metadata(models="gpt-3.5-turbo",
                                     modalities=modalities)
    data = pd.melt(gpt_35_results, id_vars=["Model", "Modality", "Dataset Index", "Modality Index", "Dataset"],
                   value_vars=["Total Accuracy"], value_name="Accuracy").sort_values("Modality Index")

    graph_dataset_comparison(title="GPT-3.5 Performance, All Modalities, All Datasets",
                             data=data, output_path="GPT35-full-results",
                             modalities=modalities,
                             figsize=(15, 10),
                             plot_size=(2, 3), sort_by="Modality Index")

    # CoT Quantification
    data = pd.melt(gpt_35_results,
                   id_vars=["Dataset", "Modality", "Modality Index", "Model", "Model Index", "Dataset Index"],
                   value_vars=["Total Accuracy", "Answers Containing CoT", "CoT Accuracy", "Non-CoT Accuracy"],
                   var_name="Metric",
                   value_name="Percentage").sort_values("Modality Index")

    graph_cot_data(title="GPT-3.5 CoT Quantification, All Modalities, All Datasets",
                   data=data,
                   x="Modality",
                   group_by="Dataset Index",
                   figsize=(15, 10),
                   plot_size=(2, 3),
                   output_path="gpt35-full-cot-quant")


def G35_all_suppression():
    modalities = ["zero_shot", "zero_shot_cot", "suppressed_cot"]

    gpt_35_results = search_metadata(models="gpt-3.5-turbo",
                                     modalities=modalities)
    data = pd.melt(gpt_35_results, id_vars=["Model", "Modality", "Dataset Index", "Modality Index", "Dataset"],
                   value_vars=["Total Accuracy"], value_name="Accuracy").sort_values("Modality Index")
    graph_dataset_comparison(title="GPT-3.5 Performance, Zero Shot vs CoT Suppression, All Datasets",
                             data=data, output_path="GPT35-supp-results",
                             modalities=modalities,
                             plot_size=(2, 3), sort_by="Modality Index")

    # CoT Quantification
    data = pd.melt(gpt_35_results,
                   id_vars=["Dataset", "Modality", "Modality Index", "Model", "Model Index", "Dataset Index"],
                   value_vars=["Total Accuracy", "Answers Containing CoT", "CoT Accuracy", "Non-CoT Accuracy"],
                   var_name="Metric",
                   value_name="Percentage").sort_values("Modality Index")

    graph_cot_data(title="GPT-3.5 CoT Quantification, All Datasets",
                   data=data,
                   x="Modality",
                   group_by="Dataset Index",
                   figsize=(15, 5),
                   plot_size=(2, 3),
                   output_path="gpt35-supp-cot-quant")


def G35_stepwise_suppression():
    modalities = ["zero_shot", "zero_shot_cot", "suppressed_cot"]
    # Graph GPT-35 Stepwise Suppression results
    gpt_35_results = search_metadata(models="gpt-3.5-turbo", modalities=modalities,
                                     datasets=["stepwise"])
    data = pd.melt(gpt_35_results, id_vars=["Model", "Modality", "Modality Index", "Dataset", "ci_upper", "ci_lower"],
                   value_vars=["Total Accuracy"], value_name="Accuracy")

    graph_stepwise_comparison(title="GPT-3.5 CoT Performance on Stepwise Dataset",
                              data=data, output_path="Stepwise/gpt35-step-suppression", modalities=modalities)

    # CoT Quantification
    data = pd.melt(gpt_35_results,
                   id_vars=["Step", "Modality", "Modality Index", "Model", "Model Index", "ci_upper", "ci_lower"],
                   value_vars=["Total Accuracy", "Answers Containing CoT", "CoT Accuracy", "Non-CoT Accuracy"],
                   var_name="Metric",
                   value_name="Percentage")

    graph_cot_data(title="GPT-3.5 CoT Quantification, Stepwise Dataset",
                   data=data,
                   x="Step",
                   figsize=(15, 5),
                   plot_size=(1, 3),
                   output_path="Stepwise/gpt35-step-supp-cot-quant")


def G35_stepwise_full():
    modalities = ["zero_shot", "zero_shot_cot", "suppressed_cot",
                  "explanation_first", "answer_first"]

    gpt_35_results = search_metadata(models="gpt-3.5-turbo",
                                     modalities=modalities,
                                     datasets=["stepwise"])
    data = pd.melt(gpt_35_results, id_vars=["Model", "Modality", "Modality Index", "Dataset", "ci_upper", "ci_lower"],
                   value_vars=["Total Accuracy"], value_name="Accuracy")
    graph_stepwise_comparison(
        title="GPT-3.5 Performance on Stepwise Dataset",
        data=data, output_path="Stepwise/gpt35-full-step", modalities=modalities)

    # Full Stepwise CoT quantification results
    data = pd.melt(gpt_35_results,
                   id_vars=["Step", "Modality", "Modality Index", "Model", "Model Index", "ci_upper", "ci_lower"],
                   value_vars=["Total Accuracy", "Answers Containing CoT", "CoT Accuracy", "Non-CoT Accuracy"],
                   var_name="Metric",
                   value_name="Percentage")

    graph_cot_data(title="GPT-3.5 CoT Quantification, All Modalities, Stepwise Dataset",
                   data=data,
                   x="Step",
                   figsize=(15, 10),
                   plot_size=(2, 3),
                   output_path="Stepwise/gpt35-step-cot-quant")


def G3_all():
    modalities = ["zero_shot", "zero_shot_cot", "suppressed_cot", "explanation_first", "answer_first"]

    # Graph GPT-3 All dataset results
    gpt_3_results = search_metadata(models="text-davinci-002",
                                    modalities=modalities)

    data = pd.melt(gpt_3_results, id_vars=["Model", "Modality", "Dataset Index", "Modality Index", "Dataset"],
                   value_vars=["Total Accuracy"], value_name="Accuracy").sort_values("Dataset Index")
    graph_dataset_comparison(title="text-davinci-002 Performance, All Modalities, All Datasets",
                             data=data, output_path="davinci-full-results",
                             modalities=modalities,
                             figsize=(15, 10),
                             plot_size=(2, 3), sort_by="Modality Index")

    # CoT Quant results
    data = pd.melt(gpt_3_results, id_vars=["Model", "Modality", "Dataset Index", "Modality Index", "Dataset"],
                   value_vars=["Total Accuracy", "Answers Containing CoT", "CoT Accuracy", "Non-CoT Accuracy"],
                   var_name="Metric",
                   value_name="Percentage").sort_values("Modality Index")

    graph_cot_data(title="text-davinci-002 CoT Quantification, All Modalities, All Datasets",
                   data=data,
                   figsize=(10, 10),
                   plot_size=(2, 3),
                   hue="Metric",
                   x="Modality",
                   group_by="Dataset Index",
                   output_path="davinci-full-cot-quant")


def G3_stepwise_full():
    # Graph GPT-3 (text-davinci-002) FULL Stepwise results
    modalities = ["zero_shot", "zero_shot_cot", "suppressed_cot", "explanation_first", "answer_first"]

    gpt_3_results = search_metadata(models="text-davinci-002",
                                    modalities=modalities,
                                    datasets=["stepwise"])
    data = pd.melt(gpt_3_results, id_vars=["Model", "Modality", "Modality Index", "Dataset"],
                   value_vars=["Total Accuracy"], value_name="Accuracy").sort_values("Dataset")

    graph_stepwise_comparison(
        title="text-davinci-002 Performance on Stepwise Dataset",
        data=data, output_path="Stepwise/davinci-full-step", modalities=modalities)

    # CoT Quantification
    data = pd.melt(gpt_3_results,
                   id_vars=["Step", "Modality", "Modality Index", "Model", "Model Index"],
                   value_vars=["Total Accuracy", "Answers Containing CoT", "CoT Accuracy", "Non-CoT Accuracy"],
                   var_name="Metric",
                   value_name="Percentage")

    graph_cot_data(title="text-davinci-002 CoT Quantification, All Modalities, Stepwise Dataset",
                   data=data,
                   x="Step",
                   figsize=(15, 10),
                   plot_size=(2, 3),
                   output_path="Stepwise/davinci-step-cot-quant")


def prompt_comparison():
    frames = list()

    frames.append(pd.DataFrame({
        "Model": "text-davinci-002",
        "Model Index": model_index_map["text-davinci-002"],
        "Extraction": "Initial",
        "Extraction Index": 1,
        "Dataset": "gsm8k",
        "Total Accuracy": [15.0]
    }))

    frames.append(pd.DataFrame({
        "Model": "text-davinci-002",
        "Model Index": model_index_map["text-davinci-002"],
        "Extraction": "Simplified",
        "Extraction Index": 2,
        "Dataset": "gsm8k",
        "Total Accuracy": [12.17]
    }))

    frames.append(pd.DataFrame({
        "Model": "text-davinci-002",
        "Model Index": model_index_map["text-davinci-002"],
        "Extraction": "Simplified In-Brackets",
        "Extraction Index": 3,
        "Dataset": "gsm8k",
        "Total Accuracy": [11.0]
    }))

    frames.append(pd.DataFrame({
        "Model": "gpt-3.5-turbo",
        "Model Index": model_index_map["gpt-3.5-turbo"],
        "Extraction": "Initial",
        "Extraction Index": 1,
        "Dataset": "gsm8k",
        "Total Accuracy": [76.3]
    }))

    frames.append(pd.DataFrame({
        "Model": "gpt-3.5-turbo",
        "Model Index": model_index_map["gpt-3.5-turbo"],
        "Extraction": "Simplified",
        "Extraction Index": 2,
        "Dataset": "gsm8k",
        "Total Accuracy": [81.16]
    }))

    frames.append(pd.DataFrame({
        "Model": "gpt-3.5-turbo",
        "Model Index": model_index_map["gpt-3.5-turbo"],
        "Extraction": "Simplified In-Brackets",
        "Extraction Index": 3,
        "Dataset": "gsm8k",
        "Total Accuracy": [78.5]
    }))

    orig = pd.concat(frames)
    data = pd.melt(orig, id_vars=["Model", "Model Index", "Extraction", "Extraction Index", "Dataset"],
                   value_vars=["Total Accuracy"], value_name="Accuracy").sort_values("Dataset")
    graph_generic(title="Comparison of Answer Extraction Techniques By Model, GSM8k",
                  data=data, group_by="Model Index", output_path="gsm8k-extraction-comparison",
                  chart_labels=["text-davinci-002", "GPT 3.5", "GPT-4"],
                  x_labels=["Initial", "Simplified", "In-Brackets"], figsize=(5, 3),
                  x="Extraction Index", plot_size=(1, 2), y="Accuracy")

    frames = list()

    frames.append(pd.DataFrame({
        "Model": "text-davinci-002",
        "Model Index": model_index_map["text-davinci-002"],
        "Modality": "zero_shot",
        "Extraction": "Initial",
        "Extraction Index": 1,
        "Dataset": "multiarith",
        "Total Accuracy": [26.0]
    }))

    frames.append(pd.DataFrame({
        "Model": "text-davinci-002",
        "Model Index": model_index_map["text-davinci-002"],
        "Modality": "zero_shot",
        "Extraction": "Simplified",
        "Extraction Index": 2,
        "Dataset": "multiarith",
        "Total Accuracy": [21.2]
    }))

    frames.append(pd.DataFrame({
        "Model": "text-davinci-002",
        "Model Index": model_index_map["text-davinci-002"],
        "Modality": "zero_shot",
        "Extraction": "Simplified In-Brackets",
        "Extraction Index": 3,
        "Dataset": "multiarith",
        "Total Accuracy": [15.5]
    }))

    frames.append(pd.DataFrame({
        "Model": "gpt-3.5-turbo",
        "Model Index": model_index_map["gpt-3.5-turbo"],
        "Modality": "zero_shot",
        "Extraction": "Initial",
        "Extraction Index": 1,
        "Dataset": "multiarith",
        "Total Accuracy": [90.16]
    }))

    frames.append(pd.DataFrame({
        "Model": "gpt-3.5-turbo",
        "Model Index": model_index_map["gpt-3.5-turbo"],
        "Modality": "zero_shot",
        "Extraction": "Simplified",
        "Extraction Index": 2,
        "Dataset": "multiarith",
        "Total Accuracy": [95.3]
    }))

    frames.append(pd.DataFrame({
        "Model": "gpt-3.5-turbo",
        "Model Index": model_index_map["gpt-3.5-turbo"],
        "Extraction Index": 3,
        "Modality": "zero_shot",
        "Extraction": "Simplified In-Brackets",
        "Dataset": "multiarith",
        "Total Accuracy": [91.5]
    }))

    orig = pd.concat(frames)
    data = pd.melt(orig, id_vars=["Model", "Model Index", "Extraction", "Extraction Index", "Dataset"],
                   value_vars=["Total Accuracy"], value_name="Accuracy").sort_values("Dataset")
    graph_generic(title="Comparison of Answer Extraction Techniques By Model, MultiArith",
                  data=data, group_by="Model Index", output_path="multiarith-extraction-comparison",
                  chart_labels=["text-davinci-002", "GPT 3.5", "GPT-4"],
                  x_labels=["Initial", "Simplified", "In-Brackets"], figsize=(5, 3),
                  x="Extraction Index", plot_size=(1, 2), y="Accuracy")


def gpt4_32_vs_8():
    frames = list()

    frames.append(pd.DataFrame({
        "Model": "gpt-4-32k",
        "Modality": "zero_shot",
        "Dataset": "gsm8k",
        "Total Accuracy": [93.53233830845771]
    }))

    frames.append(pd.DataFrame({
        "Model": "gpt-4-8k",
        "Modality": "zero_shot",
        "Dataset": "gsm8k",
        "Total Accuracy": [92.0]
    }))

    data = pd.concat(frames)
    data = pd.melt(data, id_vars=["Model", "Dataset"],
                   value_vars=["Total Accuracy"], value_name="Accuracy").sort_values("Dataset")

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), layout="constrained")
    fig.suptitle("Comparison of GPT-4 8K and 32k, GSM8k, Two-Prompt Extraction")

    sns.set_style("whitegrid")

    generate_singular_plot(ax=ax, x="Model", y="Accuracy", coordinate=None,
                           data=data, xtick_labels=["GPT-4 32k", "GPT-4 8k"], title=None)

    plt.savefig(r"GPT4\gpt4-32k-8k-comparison")
    plt.close("all")


def MMLU_full():
    modalities = ["zero_shot", "zero_shot_cot", "suppressed_cot", "explanation_first", "answer_first"]
    xtick_labels = ["College", "Combined", "High-School"]

    gpt_4_results = search_metadata(models="gpt-4",
                                    modalities=["zero_shot", "zero_shot_cot", "suppressed_cot",
                                                "explanation_first", "answer_first"],
                                    datasets=["mmlu"])
    data = pd.melt(frame=gpt_4_results,
                   id_vars=["Model", "Modality", "Modality Index", "Dataset", "Discriminator"],
                   value_vars=["Total Accuracy"],
                   value_name="Accuracy").sort_values("Discriminator")
    graph_generic(title=f"GPT-4 Performance on MMLU Dataset",
                  data=data, x="Discriminator",
                  group_by="Modality Index",
                  output_path=f"mmlu/gpt4-mmlu-full-step",
                  figsize=(15, 10),
                  chart_labels=[modality_to_label_map[i] for i in modalities],
                  palette=[modality_to_color_map[i] for i in modalities],
                  x_labels=xtick_labels,
                  plot_size=(2, 3))

    # Full Stepwise CoT quantification results
    data = pd.melt(frame=gpt_4_results,
                   id_vars=["Dataset", "Modality", "Modality Index", "Model", "Model Index", "Discriminator"],
                   value_vars=["Total Accuracy", "Answers Containing CoT", "CoT Accuracy", "Non-CoT Accuracy"],
                   var_name="Metric",
                   value_name="Percentage").sort_values("Discriminator")

    graph_cot_data(title="GPT-4 CoT Quantification, All Modalities, MMLU Dataset",
                   data=data,
                   x="Discriminator",
                   xtick_labels=[modality_to_label_map[modality] for modality in modalities],
                   figsize=(15, 10),
                   plot_size=(2, 3),
                   output_path=f"mmlu/gpt4-mmlu-cot-quant")

    gpt_35_results = search_metadata(models="gpt-3.5-turbo",
                                     modalities=["zero_shot", "zero_shot_cot", "suppressed_cot",
                                                 "explanation_first", "answer_first"],
                                     datasets=["mmlu"])
    data = pd.melt(frame=gpt_35_results,
                   id_vars=["Model", "Modality", "Modality Index", "Dataset", "Discriminator"],
                   value_vars=["Total Accuracy"],
                   value_name="Accuracy").sort_values("Discriminator")
    graph_generic(title=f"GPT-3.5 Performance on MMLU Dataset",
                  data=data, x="Discriminator",
                  group_by="Modality Index",
                  output_path=f"mmlu/gpt35-mmlu-full-step",
                  figsize=(15, 10),
                  chart_labels=[modality_to_label_map[modality] for modality in modalities],
                  x_labels=xtick_labels,
                  plot_size=(2, 3),
                  sort_by="Modality Index")

    # Full Stepwise CoT quantification results
    data = pd.melt(frame=gpt_35_results,
                   id_vars=["Dataset", "Modality", "Modality Index", "Model", "Model Index", "Discriminator"],
                   value_vars=["Total Accuracy", "Answers Containing CoT", "CoT Accuracy", "Non-CoT Accuracy"],
                   var_name="Metric",
                   value_name="Percentage").sort_values("Discriminator")

    graph_cot_data(title="GPT-3.5 CoT Quantification, All Modalities, MMLU Dataset",
                   data=data,
                   x="Discriminator",
                   xtick_labels=xtick_labels,
                   figsize=(15, 10),
                   plot_size=(2, 3),
                   output_path=f"mmlu/gpt35-mmlu-cot-quant")


def G35_two_stage():
    x_labels = ["Zero Shot", "Zero Shot CoT", "Suppressed CoT", "Explanation First", "Answer First"]
    chart_labels = ["MultiArith", "GSM8k", "Aqua-RAT", "Coin Flip", "MMLU"]

    # Graph GPT-3.5 All dataset results
    gpt_35_results = search_metadata(models="gpt-3.5-turbo",
                                     modalities=["zero_shot", "zero_shot_cot", "suppressed_cot",
                                                 "explanation_first", "answer_first"])

    data = pd.melt(frame=gpt_35_results,
                   id_vars=["Model", "Modality", "Dataset Index", "Modality Index", "Dataset"],
                   value_vars=["Total Accuracy"],
                   value_name="Accuracy").sort_values("Modality Index")

    graph_generic(title="GPT-3.5 Performance, All Datasets, Two-Stage Extraction",
                  data=data,
                  group_by="Dataset Index",
                  output_path=r"Two-Stage\GPT35-full-results-Two-Stage",
                  chart_labels=chart_labels,
                  x_labels=x_labels,
                  figsize=(15, 10),
                  plot_size=(2, 3),
                  sort_by="Modality Index")

    # CoT Quantification
    data = pd.melt(frame=gpt_35_results,
                   id_vars=["Dataset", "Modality", "Modality Index", "Model", "Model Index", "Dataset Index"],
                   value_vars=["Total Accuracy", "Answers Containing CoT", "CoT Accuracy", "Non-CoT Accuracy"],
                   var_name="Metric",
                   value_name="Percentage").sort_values("Modality Index")

    graph_cot_data(title="GPT-3.5 CoT Quantification, All Modalities, All Datasets",
                   data=data,
                   x="Modality",
                   group_by="Dataset Index",
                   figsize=(15, 10),
                   plot_size=(2, 3),
                   output_path=r"Two-Stage\gpt35-full-cot-quant-two-stage")
