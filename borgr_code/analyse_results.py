import argparse
import ast
import itertools
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters

sns.set()
import matplotlib.pyplot as plt

# read from
BLIMP = "/cs/snapless/oabend/borgr/ordert/blimp/data"
POOL_SIZE = 8

BLIMP_SUPER_CAT = {"anaphor agreement": ["anaphor_gender_agreement", "anaphor_number_agreement"],
                   "argument structure": ["animate_subject_passive", "animate_subject_trans", "causative",
                                          "drop_argument", "inchoative", "intransitive", "passive_1", "passive_2",
                                          "transitive"],
                   "binding": ["principle_A_c_command", "principle_A_case_1", "principle_A_case_2",
                               "principle_A_domain_1", "principle_A_domain_2", "principle_A_domain_3",
                               "principle_A_reconstruction"],
                   "control/raising": ["existential_there_object_raising", "existential_there_subject_raising",
                                       "expletive_it_object_raising", "tough_vs_raising_1", "tough_vs_raising_2"],
                   "determiner noun agreement": ["determiner_noun_agreement_1", "determiner_noun_agreement_2",
                                                 "determiner_noun_agreement_irregular_1",
                                                 "determiner_noun_agreement_irregular_2",
                                                 "determiner_noun_agreement_with_adj_1",
                                                 "determiner_noun_agreement_with_adj_2",
                                                 "determiner_noun_agreement_with_adj_irregular_1",
                                                 "determiner_noun_agreement_with_adj_irregular_2"],
                   "elipsis": ["ellipsis_n_bar_1", "ellipsis_n_bar_2"],
                   "filler gap": ["wh_questions_object_gap", "wh_questions_subject_gap",
                                  "wh_questions_subject_gap_long_distance", "wh_vs_that_no_gap",
                                  "wh_vs_that_no_gap_long_distance", "wh_vs_that_with_gap",
                                  "wh_vs_that_with_gap_long_distance"],
                   "irregular forms": ["irregular_past_participle_adjectives", "irregular_past_participle_verbs"],
                   "island effects": ["adjunct_island", "complex_NP_ _island",
                                      "coordinate_structure_constraint_complex_left_branch",
                                      "coordinate_structure_constraint_object_extraction",
                                      "left_branch_island_echo_question", "left_branch_island_simple_question",
                                      "sentential_subject_island", "wh_island "],
                   "npi licensing": ["matrix_question_npi_licensor_present", "npi_present_1", "npi_present_2",
                                     "only_npi_licensor_present", "only_npi_scope",
                                     "sentential_negation_npi_licensor_present", "sentential_negation_npi_scope"],
                   "quantifiers": ["existential_there_quantifiers_1", "existential_there_quantifiers_2",
                                   "superlative_quantifiers_1", "superlative_quantifiers_2"],
                   "subject verb agreement": ["distractor_agreement_relational_noun",
                                              "distractor_agreement_relative_clause",
                                              "irregular_plural_subject_verb_agreement_1",
                                              "irregular_plural_subject_verb_agreement_2",
                                              "regular_plural_subject_verb_agreement_1",
                                              "regular_plural_subject_verb_agreement_2"],
                   }


def accuracy_from_file(file):
    answers = correct_from_file(file)
    correct = sum(answers)
    wrong = len(answers) - correct
    accuracy = correct / len(answers) if answers else 0
    if wrong + correct == 0:
        print(f"corrupt file {file}")
    return accuracy


def average_correlation(orders, other_orders=None):
    corr = 0
    # ranks = orders
    ranks = []
    for order in orders:
        ranks.append([orders[0].index(item) for item in order])
    if other_orders is None:
        pairs = itertools.combinations(ranks, 2)
    else:
        pairs = itertools.product(orders, other_orders)
    for pair_num, (rank_a, rank_b) in enumerate(pairs):
        corr += spearmanr(rank_a, rank_b)[0]
    corr = corr / (pair_num + 1)
    return corr


def learnt_orders(df, steps):
    orders = []
    for model in df["model"].unique():
        complexity_order = df[(df["steps"] == steps) & (df["model"] == model)].sort_values("accuracy")[
            "challenge"].tolist()
        if complexity_order:
            if orders and len(orders[0][-1]) != len(complexity_order):
                print(f"warning wrong lengths in steps {steps} model {model} and {df['model'].unique()[-1]}, skipping ")
            else:
                orders.append((model, steps, complexity_order))
    return orders


def correlate_with_base(df_base, df):
    correlations_by_steps = []
    for steps in set(df["steps"].unique()) & set(df_base["steps"].unique()):
        orders = learnt_orders(df, steps)
        orders = [order[-1] for order in orders]
        base_orders = learnt_orders(df_base, steps)
        base_orders = [order[-1] for order in base_orders]
        cor = average_correlation(orders, base_orders)
        correlations_by_steps.append((steps, cor))
    correlations_by_steps = pd.DataFrame(correlations_by_steps, columns=["steps", "correlation"])
    sns.lineplot(x="steps", y="correlation", data=correlations_by_steps)
    # plt.legend(loc="best")
    plt.title("average spearman correlation of challenges rank as a function of steps")
    plt.savefig(os.path.join(graphs_path, f"correlation_with_base_by_steps.png"))
    plt.clf()


def correlate_models(df):
    # calculate correlation between models on how hard each phenomenon is
    correlations_by_steps = []
    for steps in df["steps"].unique():
        orders = learnt_orders(df, steps)
        orders = [order[-1] for order in orders]
        if len(orders) > 3:
            cor = average_correlation(orders)
            correlations_by_steps.append((steps, cor))
    correlations_by_steps = pd.DataFrame(correlations_by_steps, columns=["steps", "correlation"])
    sns.lineplot(x="steps", y="correlation", data=correlations_by_steps)
    # plt.legend(loc="best")
    plt.title("average spearman correlation of challenges rank as a function of steps")
    plt.savefig(os.path.join(graphs_path, f"correlation_by_steps.png"))
    plt.clf()


def plot_fields(df):
    # plot per challenge together (on line per challenge)
    group = df.groupby(["steps", "challenge"]).mean()
    group = group["accuracy"].unstack()
    for field in df["field"].unique():
        field_df = df[df["field"] == field]
        for challenge in field_df["challenge"].unique():
            sns.lineplot(data=group[challenge],
                         label=challenge)
        plt.xlabel("steps")
        plt.ylabel("accuracy")
        plt.title("averaged")
        plt.legend(loc="best").remove()
        # Shrink current axis by 20%
        ax = plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 4})
        plt.savefig(os.path.join(graphs_path, f"{field.replace(os.sep,'+')}.png"))
        plt.clf()


def plot_categories(df):
    # plot per challenge together (on line per challenge)
    group = df.groupby(["steps", "challenge"]).mean()
    group = group["accuracy"].unstack()
    for category in BLIMP_SUPER_CAT:
        for challenge in df["challenge"].unique():
            if challenge in BLIMP_SUPER_CAT[category]:
                sns.lineplot(data=group[challenge],
                             label=challenge)
        plt.xlabel("steps")
        plt.ylabel("accuracy")
        plt.title("averaged")
        plt.legend(loc="best").remove()
        # Shrink current axis by 20%
        ax = plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 4})
        plt.savefig(os.path.join(graphs_path, f"{category.replace(os.sep,'+')}.png"))
        plt.clf()


def all_challenges(df):
    # plot per challenge together (on line per challenge)
    group = df.groupby(["steps", "challenge"]).mean()
    group = group["accuracy"].unstack()
    for challenge in df["challenge"].unique():
        sns.lineplot(data=group[challenge],
                     label=challenge)
    plt.xlabel("steps")
    plt.ylabel("accuracy")
    plt.title("averaged")
    plt.legend(loc="best").remove()
    # Shrink current axis by 20%
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 4})
    plt.savefig(os.path.join(graphs_path, f"aaggregation_steps.png"))
    plt.clf()


def per_challenge(df, plot_steps=True, plot_perplexity=True, max_perp=30):
    # Plot line per model (each challenge on a separate plot)
    for challenge in df["challenge"].unique():
        if plot_steps:
            for model in df["model"].unique():
                sns.lineplot(x="steps", y="accuracy", data=df[(df["model"] == model) & (df["challenge"] == challenge)],
                             label=model)
            plt.title(challenge)
            plt.legend(loc="best")
            plt.savefig(os.path.join(graphs_path, f"{challenge}_steps.png"))
            plt.clf()
        if plot_perplexity:
            df_perp = df[df["perplexity"] < max_perp]
            plt.gca().set(xscale="log")
            for model in df_perp["model"].unique():
                sns.lineplot(x="perplexity", y="accuracy",
                             data=df_perp[(df_perp["model"] == model) & (df_perp["challenge"] == challenge)],
                             label=model)
            plt.title(challenge)
            plt.legend(loc="best")
            plt.savefig(os.path.join(graphs_path, f"{challenge}_perplexity.png"))
            plt.clf()


def correct_from_file(file):
    res = []
    with open(file) as fl:
        for i, line in enumerate(fl):
            lm_loss = float(line.strip())
            if i % 2 == 1:  # lm_loss is like perplexity (need e^ [loss * token num]), lower is better
                bad_loss = lm_loss
                if bad_loss > good_loss:
                    res.append(1)
                else:
                    res.append(0)
            else:
                good_loss = lm_loss
    assert len(res) == 1000, f"{len(res)} {file}"
    return res


def calculate_outer_agreement(models_df, base_df):
    # Plot line per model (each challenge on a separate plot)
    kappas = []
    print("Calculating outer agreement...")
    for challenge in models_df["challenge"].unique():
        for steps in models_df["steps"].unique():
            sub_df = models_df[(models_df["steps"] == steps) & (models_df["challenge"] == challenge)]
            corrects = []
            for model in models_df["model"].unique():
                correct = sub_df[sub_df["model"] == model]["correct"].tolist()
                if correct:
                    corrects.append(ast.literal_eval(correct[0]))
            if len(corrects) > 3:
                raters = aggregate_raters(np.array(corrects).T, 2)[0]
                kappas.append((challenge, steps, fleiss_kappa(raters)))
    df = pd.DataFrame(kappas, columns=["challenge", "steps", "kappa"])
    group = df.groupby(["steps", "challenge"]).mean()
    # df.groupby(["challenge"]).mean()["kappa"].to_csv("/home/leshem/PycharmProjects/ordert/ordert/transformers/output/per_challenge_kappa.csv")
    # df.groupby(["challenge", "steps"]).mean()["kappa"].to_csv("/home/leshem/PycharmProjects/ordert/ordert/transformers/output/per_challenge_steps_kappa.csv")
    # df.to_csv("/home/leshem/PycharmProjects/ordert/ordert/transformers/output/kappas.csv")


def calculate_inner_agreement(df):
    # Plot line per model (each challenge on a separate plot)
    kappas = []
    print("Calculating inner agreement...")
    for challenge in df["challenge"].unique():
        for steps in df["steps"].unique():
            sub_df = df[(df["steps"] == steps) & (df["challenge"] == challenge)]
            corrects = []
            for model in df["model"].unique():
                correct = sub_df[sub_df["model"] == model]["correct"].tolist()
                if correct:
                    corrects.append(ast.literal_eval(correct[0]))
            if len(corrects) > 3:
                raters = aggregate_raters(np.array(corrects).T, 2)[0]
                kappas.append((challenge, steps, fleiss_kappa(raters)))
    df = pd.DataFrame(kappas, columns=["challenge", "steps", "kappa"])
    group = df.groupby(["steps", "challenge"]).mean()
    # df.groupby(["challenge"]).mean()["kappa"].to_csv("/home/leshem/PycharmProjects/ordert/ordert/transformers/output/per_challenge_kappa.csv")
    # df.groupby(["challenge", "steps"]).mean()["kappa"].to_csv("/home/leshem/PycharmProjects/ordert/ordert/transformers/output/per_challenge_steps_kappa.csv")
    # df.to_csv("/home/leshem/PycharmProjects/ordert/ordert/transformers/output/kappas.csv")


def get_properties(file, path):
    with open(os.path.join(path, file + ".jsonl")) as fl:
        line = json.loads(fl.readline())
        return line["field"], line["linguistics_term"]


def acquire_statistics_from_file(root, filename):
    accuracy = accuracy_from_file(os.path.join(root, filename))
    correct = correct_from_file(os.path.join(root, filename))
    challenge = filename[:-len(".txt")]
    field, phenomenon = get_properties(challenge, BLIMP)
    return phenomenon, challenge, field, accuracy, correct


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "-f", "--force", action="store_true", help="Recompute."
    )
    parser.add_argument(
        "--heavy", action="store_true", help="Rely on all data 'heavy' run."
    )
    args = parser.parse_args()

    # Ignore if less than:
    min_steps = 10
    max_perp = 1000

    # write csv to:
    path = os.path.dirname(__file__) + r"/../output"
    out_path = path
    heavy_path = os.path.join(path, "results.csv")
    if args.heavy:
        csv_path = heavy_path
    else:
        csv_path = os.path.join(path, "results_light.csv")

    base_model = "gpt2Small"
    # skip_models = ["short"]
    # force recalculation of csv
    force = args.force

    if force or ((not os.path.isfile(csv_path)) and not os.path.isfile(heavy_path)):
        metadatas = []
        paths = []
        for root, dirnames, filenames in os.walk(path):
            results_dir = os.path.basename(root)
            model_name = os.path.basename(os.path.dirname(os.path.dirname(root)))
            if results_dir.startswith("steps"):
                steps = int(results_dir.split("_")[0][len("steps"):])
                perplexity = float(results_dir.split("perplexity")[1])
                for filename in filenames:
                    paths.append((root, filename))
                    metadatas.append([model_name, steps, perplexity])
                    # if filename == "wh_island.txt":
                    # print(res[-1])
                    # print((model_name, filename, steps, perplexity, acc))
        headers = ["model", "steps", "perplexity", "phenomenon", "challenge", "field", "accuracy", "correct"]
        res = []
        # for path, metadata in zip(paths, metadatas):
        #     vals = acquire_statistics_from_file(*path)
        #     res.append(metadata + vals)

        chunksize = 1
        if (len(metadatas) > 100):
            chunksize = int(len(metadatas) / POOL_SIZE / 10)
        pool = Pool(POOL_SIZE)
        res = pool.starmap(acquire_statistics_from_file, paths)
        res = [data + list(stats) for data, stats in zip(metadatas, res)]
        df = pd.DataFrame(res, columns=headers)
        if not args.heavy:
            df.drop(["correct"], axis=1, inplace=True)
        print(df)
        df.to_csv(csv_path, index=False)
        print(f"wrote to {csv_path}")
    elif os.path.isfile(csv_path):
        print(f"reading cached {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print(f"reading heavy {heavy_path}")
        df = pd.read_csv(heavy_path)
        df.drop(["correct"], axis=1, inplace=True)
        df.to_csv(csv_path, index=False)
        print(f"wrote light version to {csv_path}")
    graphs_path = os.path.join(out_path, "graphs")
    os.makedirs(graphs_path, exist_ok=True)
    # df = df[df["model"] != "gpt2"]
    df["base_model"] = df["model"].apply(lambda x: "gpt2Small" in x)
    df = df[df["steps"] > min_steps]
    correlate_with_base(df[df["base_model"]], df[df["model"] == "Gpt2"])
    correlate_with_base(df[df["base_model"]], df[df["model"] == "xlSmallxl"])
    correlate_models(df[df["base_model"]])

    # plot_fields(df)
    plot_categories(df[df["base_model"]])
    # all_challenges(df)
    # per_challenge(df, max_perp=max_perp)
    if args.heavy:
        calculate_inner_agreement(df[df["base_model"]])
        calculate_outer_agreement(df[~df["base_model"]], df[df["base_model"]])
