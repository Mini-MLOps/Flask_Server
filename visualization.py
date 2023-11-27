from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from Word2Vec_vectorizer import Word2Vec_vectorizer

mpl.rcParams["axes.unicode_minus"] = False

plt.rc("font", family="Malgun Gothic")


def show_tsne():
    tsne = TSNE(n_components=2)
    X = tsne.fit_transform(X_show)

    df = pd.DataFrame(X, index=vocab_show, columns=["x", "y"])
    fig = plt.figure()
    fig.set_size_inches(15, 10)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(df["x"], df["y"])

    for word, pos in df.iterrows():
        ax.annotate(word, pos, fontsize=10)

    plt.xlabel("t-SNE 특성 0")
    plt.ylabel("t-SNE 특성 1")
    plt.show()


def show_pca():
    pca = PCA(n_components=2)
    pca.fit(X_show)
    x_pca = pca.transform(X_show)

    plt.figure(figsize=(15, 10))
    plt.xlim(x_pca[:, 0].min(), x_pca[:, 0].max())
    plt.ylim(x_pca[:, 1].min(), x_pca[:, 1].max())
    for i in range(len(X_show)):
        plt.text(
            x_pca[i, 0],
            x_pca[i, 1],
            str(vocab_show[i]),
            fontdict={"weight": "bold", "size": 9},
        )
    plt.xlabel("첫 번째 주성분")
    plt.ylabel("두 번째 주성분")
    plt.show()


model_path = "data/word2vec.model"
model = Word2Vec_vectorizer(model_path).model_load()

# 3D
# df = pd.DataFrame(model.wv.vectors)
# df.to_csv("data/wv_model_tsv.tsv", sep="\t", index=False)

# word_df = pd.DataFrame(model.wv.index_to_key)
# word_df.to_csv("data/wv_word_tsv.tsv", sep="\t", index=False)

# 2D
vocab = list(model.wv.index_to_key)
X = model.wv[vocab]

sz = 800
X_show = X[:sz, :]
vocab_show = vocab[:sz]

show_tsne()
# show_pca()
