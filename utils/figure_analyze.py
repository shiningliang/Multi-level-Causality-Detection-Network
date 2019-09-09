import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_context(context="talk")
plt.switch_backend('agg')

results = pd.read_csv("analysis.csv")

plt.rcParams["font.family"] = "Times New Roman"
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
sns.set(style="whitegrid", font="Times New Roman")
plt.subplots_adjust(wspace=0.25, hspace=0.25)
c0 = sns.barplot(x=results["train_freq"][:10].tolist(), y=results["train_word"][:10].tolist(), ax=axes[0, 0])
c0.set_xlabel("(a) Train")
c1 = sns.barplot(x=results["test_freq"][:10].tolist(), y=results["test_word"][:10].tolist(), ax=axes[0, 1])
c1.set_xlabel("(b) Test")
c2 = sns.barplot(x=results["fn_freq"][:10].tolist(), y=results["fn_word"][:10].tolist(), ax=axes[1, 0])
c2.set_xlabel("(c) FN")
c2.set(xlim=(0, 12))
c3 = sns.barplot(x=results["fp_freq"][:10].tolist(), y=results["fp_word"][:10].tolist(), ax=axes[1, 1])
c3.set_xlabel("(d) FP")
c3.set(xlim=(0, 12))
c0.figure.savefig("word_freq.eps", dpi=400)
