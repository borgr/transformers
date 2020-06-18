import ast
import itertools
import json
import os
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
import pandas as pd
from scipy.stats import spearmanr
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt


def accuracy_from_file(file):
    answers = correct_from_file(file)
    correct = sum(answers)
    wrong = len(answers) - correct
    accuracy = correct / len(answers) if answers else 0
    if wrong + correct == 0:
        print(f"corrupt file {file}")
    return accuracy


def average_correlation(orders):
    corr = 0
    ranks = orders
    ranks = []
    for order in orders:
        ranks.append([orders[0].index(item) for item in order])
    for rank_a, rank_b in itertools.combinations(ranks, 2):
        corr += spearmanr(rank_a, rank_b)[0]
    corr = corr / (len(ranks) * (len(ranks) - 1) / 2)
    return corr


def correlate_models(df):
    # calculate correlation between models on how hard each phenomenon is
    correlations_by_steps = []
    for steps in df["steps"].unique():
        orders = []
        for model in df["model"].unique():
            complexity_order = df[(df["steps"] == steps) & (df["model"] == model)].sort_values("accuracy")[
                "challenge"].tolist()
            # orders.append((model, steps, complexity_order))
            if complexity_order:
                if orders and len(orders[0]) != len(complexity_order):
                    print(f"warning wrong lengths in {model} and {df['model'].unique()[0]}")
                orders.append(complexity_order)
        if len(orders) > 3:
            cor = average_correlation(orders)
            correlations_by_steps.append((steps, cor))
    correlations_by_steps = pd.DataFrame(correlations_by_steps, columns=["steps", "correlation"])
    sns.lineplot(x="steps", y="correlation", data=correlations_by_steps)
    # plt.legend(loc="best")
    plt.title("average spearman correlation of challenges rank as a function of steps")
    plt.savefig(os.path.join(graphs_path, f"correlation_by_steps.png"))
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


def per_challenge(df, plot_steps=True, plot_perplexity=True):
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
            for model in df["model"].unique():
                sns.lineplot(x="perplexity", y="accuracy",
                             data=df_ok[(df_ok["model"] == model) & (df_ok["challenge"] == challenge)],
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


def calculate_inner_agreement(df):
    # Plot line per model (each challenge on a separate plot)
    kappas = []
    for challenge in df["challenge"].unique():
        print(challenge)
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
    with open(os.path.join(path, file+".jsonl")) as fl:
        line = json.loads(fl.readline())
        return line["field"], line["linguistics_term"]


if __name__ == '__main__':
    min_steps = 10
    max_perp = 15
    path = os.path.dirname(__file__) + r"/../output"
    blimp_path="/cs/snapless/oabend/borgr/ordert/blimp/data"
    out_path = path
    csv_path = os.path.join(path, "results.csv")
    force = False
    force = True

    if force or not os.path.isfile(csv_path):
        res = []
        for root, dirnames, filenames in os.walk(path):
            results_dir = os.path.basename(root)
            model_name = os.path.basename(os.path.dirname(os.path.dirname(root)))
            if results_dir.startswith("steps"):
                steps = int(results_dir.split("_")[0][len("steps"):])
                perplexity = float(results_dir.split("perplexity")[1])
                for filename in filenames:
                    acc = accuracy_from_file(os.path.join(root, filename))
                    correct = correct_from_file(os.path.join(root, filename))
                    challenge=filename[:-len(".txt")]
                    field, phenomenon = get_properties(challenge, blimp_path)
                    res.append((model_name, phenomenon, challenge, field, steps, perplexity, acc, correct))
                    # if filename == "wh_island.txt":
                        # print(res[-1])
                    # print((model_name, filename, steps, perplexity, acc))
        headers = ["model", "phenomenon", "challenge", "field", "steps", "perplexity", "accuracy", "correct"]
        df = pd.DataFrame(res, columns=headers)
        print(df)
        df.to_csv(csv_path, index=False)
        print(f"wrote to {csv_path}")
    else:
        print(f"reading cached {csv_path}")
        df = pd.read_csv(csv_path)

    graphs_path = os.path.join(out_path, "graphs")
    os.makedirs(graphs_path, exist_ok=True)
    df = df[df["steps"] > min_steps]
    df_ok = df[df["perplexity"] < max_perp]

    calculate_inner_agreement(df)
    # correlate_models(df)

    # all_challenges(df)
    # per_challenge(df)




