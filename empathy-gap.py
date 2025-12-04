import numpy as np
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from scipy import stats

class colr:
    header = "\033[95m"
    blue   = "\033[94m"
    cyan   = "\033[96m"
    green  = "\033[92m"
    yellow = "\033[93m"
    red    = "\033[91m"
    bold   = "\033[1m"
    under  = "\033[4m"
    reset  = "\033[0m"

def banner(titl: str):
    rule = "=" * (len(titl) + 4)
    print(f"\n{colr.bold}{colr.cyan}{rule}")
    print(f"| {titl} |")
    print(f"{rule}{colr.reset}")

def subbanner(titl: str):
    rule = "-" * (len(titl) + 4)
    print(f"\n{colr.bold}{colr.blue}{rule}")
    print(f"| {titl} |")
    print(f"{rule}{colr.reset}")

def info(text: str):
    print(f"{colr.green}[INFO]{colr.reset} {text}")

def warn(text: str):
    print(f"{colr.yellow}[WARN]{colr.reset} {text}")

def slgn(txt):
    return re.sub(r'[^a-z0-9]+', '-', str(txt).lower()).strip('-')

inpt = "empathygap-dataset.csv"
outs = "outputs"

os.makedirs(outs, exist_ok=True)

plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["axes.grid"] = True

banner("STEP 1: LOAD DATA")

base = pd.read_csv(inpt)
info(f"Loaded data shape : {base.shape[0]} rows × {base.shape[1]} columns")
info(f"Columns           : {list(base.columns)}")

banner("STEP 2: FEATURE ENGINEERING")

data = base.copy()

amap = {
    "None": 0,
    "Coffee (~₹300)": 300,
    "Meal (~₹800)": 800,
    "Cash (~₹2000)": 2000,
}

data["EXPEDITE_AMT"] = data["EXPEDITE OFFER (INR)"].map(amap).astype(float)
data["EXPECT_AMT"]   = data["EXPECTED FAVOR (INR)"].map(amap).astype(float)

data["HEADS"] = (data["COIN REPORT (HONESTY TASK)"] == "Heads").astype(int)

for coln in ["PEER FRAME", "TA FRAME", "HONOR CODE", "PROGRAM"]:
    data[coln] = data[coln].astype("category")

data["STRESS"]        = data["STRESS (0-100)"]
data["COPY"]          = data["COPY LIKELIHOOD (0-100)"]
data["SHARE"]         = data["SHARE LIKELIHOOD (0-100)"]
data["PEER_HELP"]     = data["PEER HELPFULNESS (1-7)"]
data["GRADER_STRICT"] = data["GRADER STRICTNESS (1-7)"]
data["DECISION_TIME"] = data["DECISION TIME (SECONDS)"]

info("Feature engineering complete (EXPEDITE_AMT, EXPECT_AMT, HEADS, aliases).")

banner("STEP 3: LANGUAGE MARKERS")

emwr = ["sorry", "thanks", "thank you", "appreciate", "understand", "we", "let's", "no worries"]
imwr = ["asap", "urgent", "right now", "immediately", "running out of time", "send your solution", "!!"]

def markcnt(text, wset):
    if pd.isna(text):
        return 0
    tlow = text.lower()
    return sum(len(re.findall(re.escape(wrdx), tlow)) for wrdx in wset)

data["EMP_MARK"] = data["DM TO NIKHIL"].apply(lambda x: markcnt(x, emwr))
data["IMP_MARK"] = data["DM TO NIKHIL"].apply(lambda x: markcnt(x, imwr))

info("Added EMP_MARK and IMP_MARK columns (language-based features).")

banner("STEP 4: DESCRIPTIVE STATISTICS")

dcol = [
    "STRESS", "COPY", "SHARE", "EXPEDITE_AMT", "EXPECT_AMT",
    "PEER_HELP", "GRADER_STRICT", "DECISION_TIME", "EMP_MARK", "IMP_MARK"
]

desc = data[dcol].describe().round(2)

subbanner("Overall descriptive stats")
print(colr.bold + colr.green + desc.to_string() + colr.reset)

desc.to_csv(os.path.join(outs, "descriptives_overall.csv"))
with open(os.path.join(outs, "descriptives_overall_latex.tex"), "w") as fhan:
    fhan.write(desc.to_latex())
info("Saved overall descriptive stats (CSV + LaTeX).")

banner("STEP 5: MEANS BY PEER FRAME × TA FRAME")

gmen = (
    data.groupby(["PEER FRAME", "TA FRAME"], observed=True)[
        ["COPY", "SHARE", "EXPEDITE_AMT", "EXPECT_AMT",
         "HEADS", "STRESS", "DECISION_TIME", "EMP_MARK", "IMP_MARK"]
    ]
    .mean()
    .round(2)
)

subbanner("Means by PEER FRAME × TA FRAME")
print(colr.bold + colr.cyan + gmen.to_string() + colr.reset)

gmen.to_csv(os.path.join(outs, "means_peer_ta.csv"))
with open(os.path.join(outs, "means_peer_ta_latex.tex"), "w") as fhan:
    fhan.write(gmen.to_latex())
info("Saved means by PEER × TA (CSV + LaTeX).")

banner("STEP 6: FREQUENCY TABLES & PIE CHARTS")

def freqpie(seri, name):
    freq = seri.value_counts().rename("count")
    fpct = (freq / freq.sum() * 100).round(1).rename("percent")
    tabl = pd.concat([freq, fpct], axis=1)

    subbanner(f"Frequency: {name}")
    print(colr.bold + tabl.to_string() + colr.reset)

    tabl.to_csv(os.path.join(outs, f"freq_{name}.csv"))

    plt.figure()
    tabl["percent"].plot(kind="pie", labels=tabl.index, autopct="%1.1f%%")
    plt.ylabel("")
    plt.title(f"{name} Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(outs, f"pie_{name}.png"))
    plt.close()
    info(f"Saved frequency table + pie chart for {name}.")

freqpie(data["PROGRAM"],               "PROGRAM")
freqpie(data["HONOR CODE"],            "HONOR_CODE")
freqpie(data["PEER FRAME"],            "PEER_FRAME")
freqpie(data["TA FRAME"],              "TA_FRAME")
freqpie(data["EXPEDITE OFFER (INR)"],  "EXPEDITE_OFFER")
freqpie(data["EXPECTED FAVOR (INR)"],  "EXPECTED_FAVOR")

banner("STEP 7: HISTOGRAMS & BOXPLOTS")

def histfig(coln, bins=15):
    plt.figure()
    data[coln].hist(bins=bins)
    plt.title(f"Histogram of {coln}")
    plt.xlabel(coln)
    plt.ylabel("Frequency")
    plt.tight_layout()
    fnam = f"hist-{slgn(coln)}.png"
    plt.savefig(os.path.join(outs, fnam))
    plt.close()
    info(f"Saved histogram: {fnam}")

for coln in ["STRESS", "COPY", "SHARE", "EXPEDITE_AMT", "EXPECT_AMT", "DECISION_TIME"]:
    histfig(coln)

def boxfig(coln, bycl):
    plt.figure()
    data.boxplot(column=coln, by=bycl)
    plt.suptitle("")
    plt.title(f"{coln} by {bycl}")
    plt.ylabel(coln)
    plt.tight_layout()
    fnam = f"box-{slgn(coln)}-by-{slgn(bycl)}.png"
    plt.savefig(os.path.join(outs, fnam))
    plt.close()
    info(f"Saved boxplot: {fnam}")

for coln in ["COPY", "SHARE", "EXPEDITE_AMT", "EXPECT_AMT", "STRESS", "DECISION_TIME"]:
    boxfig(coln, "PEER FRAME")
    boxfig(coln, "TA FRAME")

banner("STEP 8: HEATMAP GRIDS (PEER × TA)")

gcpy = data.pivot_table(
    values="COPY",
    index="PEER FRAME", columns="TA FRAME",
    aggfunc="mean",
    observed=True,
).round(1)

gexp = data.pivot_table(
    values="EXPEDITE_AMT",
    index="PEER FRAME", columns="TA FRAME",
    aggfunc="mean",
    observed=True,
).round(1)

ghed = data.pivot_table(
    values="HEADS",
    index="PEER FRAME", columns="TA FRAME",
    aggfunc="mean",
    observed=True,
).round(2)

gcpy.to_csv(os.path.join(outs, "grid-copy-by-peer-ta.csv"))
gexp.to_csv(os.path.join(outs, "grid-expedite-amt-by-peer-ta.csv"))
ghed.to_csv(os.path.join(outs, "grid-heads-by-peer-ta.csv"))

def gridanno(grid, titl, fnam):
    plt.figure()
    plt.imshow(grid.values, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(grid.columns)), grid.columns)
    plt.yticks(range(len(grid.index)), grid.index)
    for irow in range(len(grid.index)):
        for jcol in range(len(grid.columns)):
            plt.text(jcol, irow, grid.iloc[irow, jcol],
                     ha="center", va="center", color="white", fontsize=9)
    plt.title(titl)
    plt.tight_layout()
    plt.savefig(os.path.join(outs, fnam))
    plt.close()
    info(f"Saved heatmap: {fnam}")

gridanno(gcpy, "Mean COPY by PEER × TA", f"heatmap-{slgn('copy')}-by-peer-ta.png")
gridanno(gexp, "Mean EXPEDITE_AMT by PEER × TA", f"heatmap-{slgn('expedite amt')}-by-peer-ta.png")
gridanno(ghed, "Mean HEADS (cheating) by PEER × TA", f"heatmap-{slgn('heads')}-by-peer-ta.png")

banner("STEP 9: OLS & ANOVA MODELS")

def olsanova(form, data, name):
    subbanner(f"OLS + ANOVA for {name}")
    mres = smf.ols(form, data=data).fit()
    print(colr.bold + colr.green + mres.summary().as_text() + colr.reset)

    anov = anova_lm(mres, typ=2).round(4)
    print(colr.bold + colr.cyan + "\nANOVA Table:\n" + anov.to_string() + colr.reset)

    anov.to_csv(os.path.join(outs, f"anova_{name}.csv"))
    with open(os.path.join(outs, f"anova_{name}_latex.tex"), "w") as fhan:
        fhan.write(anov.to_latex())
    info(f"Saved ANOVA for {name} (CSV + LaTeX).")
    return mres

fcpy = (
    "COPY ~ C(Q('PEER FRAME'))*C(Q('TA FRAME'))"
    " + STRESS + C(Q('HONOR CODE'))"
)

fshr = (
    "SHARE ~ C(Q('PEER FRAME'))*C(Q('TA FRAME'))"
    " + STRESS + C(Q('HONOR CODE'))"
)

fexp = (
    "EXPEDITE_AMT ~ C(Q('PEER FRAME'))*C(Q('TA FRAME'))"
    " + STRESS + C(Q('HONOR CODE'))"
)

mcpy = olsanova(fcpy, data, "COPY")
mshr = olsanova(fshr, data, "SHARE")
mexp = olsanova(fexp, data, "EXPEDITE_AMT")

banner("STEP 10: CHEATING ANALYSIS (COIN FLIP)")

hrat = data["HEADS"].mean()
bino = stats.binomtest(data["HEADS"].sum(), n=len(data), p=0.5)

subbanner("Overall cheating rate")
print(f"{colr.bold}Heads rate (proportion of 'Heads') : {hrat:.3f}{colr.reset}")
print(f"{colr.bold}Binomial test p-value vs 0.5      : {bino.pvalue:.4f}{colr.reset}")

chmn = (
    data.groupby(["PEER FRAME", "TA FRAME"], observed=True)["HEADS"]
        .mean()
        .round(2)
)
subbanner("Heads rate by PEER FRAME × TA FRAME")
print(colr.bold + colr.cyan + chmn.to_string() + colr.reset)
chmn.to_csv(os.path.join(outs, "heads_rate_peer_ta.csv"))
info("Saved heads-rate table by PEER × TA.")

fcht = (
    "HEADS ~ C(Q('PEER FRAME'))*C(Q('TA FRAME'))"
    " + STRESS + C(Q('HONOR CODE'))"
)

mcht = smf.logit(fcht, data=data).fit()
subbanner("Logit model: HEADS (cheating)")
print(colr.bold + colr.green + mcht.summary().as_text() + colr.reset)

cpar = mcht.summary2().tables[1]
cpar.to_csv(os.path.join(outs, "logit_cheat_params.csv"))
with open(os.path.join(outs, "logit_cheat_params_latex.tex"), "w") as fhan:
    fhan.write(cpar.to_latex())
info("Saved logit coefficients for cheating (CSV + LaTeX).")

banner("STEP 11: LANGUAGE MECHANISM MODELS")

femp = "EMP_MARK ~ C(Q('PEER FRAME'))*C(Q('TA FRAME'))"
fimp = "IMP_MARK ~ C(Q('PEER FRAME'))*C(Q('TA FRAME'))"

memp = smf.ols(femp, data=data).fit()
mimp = smf.ols(fimp, data=data).fit()

subbanner("OLS: EMP_MARK (empathetic language)")
print(colr.bold + colr.green + memp.summary().as_text() + colr.reset)

subbanner("OLS: IMP_MARK (impatient/hostile language)")
print(colr.bold + colr.green + mimp.summary().as_text() + colr.reset)

emmn = (
    data.groupby("PEER FRAME", observed=True)[["EMP_MARK", "IMP_MARK"]]
        .mean()
        .round(2)
)
subbanner("Mean EMP_MARK & IMP_MARK by PEER FRAME")
print(colr.bold + colr.cyan + emmn.to_string() + colr.reset)
emmn.to_csv(os.path.join(outs, "language_means_peer.csv"))

plt.figure()
emmn["EMP_MARK"].plot(kind="bar")
plt.title("Mean EMP_MARK by PEER FRAME")
plt.ylabel("Mean empathy markers")
plt.tight_layout()
plt.savefig(os.path.join(outs, "bar_emp_mark_peer.png"))
plt.close()
info("Saved bar plot: EMP_MARK by PEER FRAME.")

plt.figure()
emmn["IMP_MARK"].plot(kind="bar")
plt.title("Mean IMP_MARK by PEER FRAME")
plt.ylabel("Mean impatience markers")
plt.tight_layout()
plt.savefig(os.path.join(outs, "bar_imp_mark_peer.png"))
plt.close()
info("Saved bar plot: IMP_MARK by PEER FRAME.")

banner("STEP 12: CORRELATION MATRIX & HEATMAP")

ccol = [
    "STRESS", "COPY", "SHARE", "EXPEDITE_AMT", "EXPECT_AMT",
    "PEER_HELP", "GRADER_STRICT", "DECISION_TIME",
    "EMP_MARK", "IMP_MARK", "HEADS"
]

corr = data[ccol].corr().round(2)

subbanner("Correlation matrix")
print(colr.bold + colr.cyan + corr.to_string() + colr.reset)
corr.to_csv(os.path.join(outs, "correlation_matrix.csv"))

plt.figure()
plt.imshow(corr.values, cmap="viridis")
plt.colorbar(label="Correlation")
plt.xticks(range(len(ccol)), ccol, rotation=90)
plt.yticks(range(len(ccol)), ccol)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(outs, "correlation_heatmap.png"))
plt.close()
info("Saved correlation heatmap.")

banner("ALL DONE")
print(f"{colr.bold}{colr.green}All analysis complete.{colr.reset}")
print(f"{colr.bold}Tables and figures saved in: {colr.under}{outs}/{colr.reset}\n")