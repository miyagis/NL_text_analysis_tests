# https://www.analyticsvidhya.com/blog/2018/02/the-different-methods-deal-text-data-predictive-python/

# Plursight course https://app.pluralsight.com/library/courses/building-sentiment-analysis-systems-python/table-of-contents
# naive bayes 2 parts:
# - A priori probabilities: get % of pos and neg cases --> if 70% is pos then a new case is, regardless of content, more likely to also be pos
# - conditional probabilities: give every word a probability to be positive/negative
#   occurence per word in pos / all pos occurences
#   Eg. P(text contains "worst"/label = Positive) = 0 --> because it never appears in a pos review
#       P(text contains "worst"/label = Negative) = 2/10 --> Because it appears in 2 neg reviews
# - Final calc: (condi p * a priori p)
#   Condi P("Really bad"=pos) = condi P("Really"=pos) * condi P("bad"=pos)
#   -> Drop in case the word doesn't exist at all in pos


import pandas as pd
import nltk


stopwords = ["de", "een", "het", "en", "van", "aan", "deze", "dit", "die", "er", "als", "met", "in", "zijn", "is",
             "dat", "om", "dan", "the"]
klinkers = ['a', 'e', 'u', 'o', 'i']


def clean_df(df):
    # Drop useles column
    df = df.drop(columns=["name"])

    # Pre-procesing
    # all lowercase
    df["review"] = df["review"].apply(lambda x: " ".join(x.lower() for x in x.split()))
    # remove punctiation
    df["review"] = df["review"].str.replace('[^\w\s]', '')
    # delete stopwords
    df["review"] = df["review"].apply(lambda x: " ".join(x for x in x.split() if x not in stopwords))
    # delete most common words
    # NOT DONE
    # delete most uncommon words
    # NOT DONE
    # Tokenization
    df["tokenized_review"] = df.apply(lambda row: nltk.word_tokenize(row["review"]), axis=1)
    # Stemming: Remove common endings of words to bring them back to their root. Eg. "ing", "en"
    return df


def clean_word(w):
    if len(w) > 4:
        if w[len(w)-2:] == "en":
            print(w)
            w = w[:len(w)-2]
            print(w)
            # Verb?
            # if the 2nd to last letter is a klinker then probably double it
            print('-------')
    return w


def create_df_condi_p(df):
    # create df to store conditional probability
    df_condi_p = pd.DataFrame(columns=("word", "pos_temp", "neg_temp"))

    for index, r in df.iterrows():
        for w in r['tokenized_review']:
            w = clean_word(w)

            # check if word is in df -> If not; add it
            if w not in df_condi_p['word'].unique():
                row_df_condi = df_condi_p['word'].count() + 1
                df_condi_p.loc[row_df_condi] = [w, 0, 0]
            else:
                row_df_condi = df_condi_p.loc[df_condi_p['word'] == w].index.values[0]

            # add 1 to either pos or neg column
            if r['grade'] >= 3.5:
                old_count = df_condi_p.at[row_df_condi, 'pos_temp']
                df_condi_p.set_value(row_df_condi, "pos_temp", old_count+1)
            elif r['grade'] <= 2.5:
                old_count = df_condi_p.at[row_df_condi, 'neg_temp']
                df_condi_p.set_value(row_df_condi, "neg_temp", old_count+1)

    return df_condi_p


def calc_condi_p(df_condi_p, pos_count, neg_count):
    if df_condi_p['word'].count() > 0:
        df_condi_p['pos'] = df_condi_p['pos_temp'] / pos_count
        df_condi_p['neg'] = df_condi_p['neg_temp'] / neg_count
        # df_condi_p = df_condi_p.drop(columns=["pos_temp", "neg_temp"])
    return df_condi_p


def applying_bayes_theorem(t, df, sum_pos, sum_neg):
    # get and multiply p for every word: P(t/pos)
    df_t = pd.DataFrame(columns=["name", "review"])
    df_t.loc[0] = ["", t]
    df_t = clean_df(df_t)

    p_pos = 1
    p_neg = 1

    for index, r in df_t.iterrows():
        for w in r['tokenized_review']:
            w = clean_word(w)
            temp = df.loc[df['word'] == w].index.values

            if len(temp) > 0:
                temp = temp[0]
                pos_val = df.at[temp, 'pos']
                if pos_val != 0:
                    p_pos = p_pos * pos_val
                neg_val = df.at[temp, 'neg']
                if neg_val != 0:
                    p_neg = p_neg * neg_val
    # multiply by sum_pos: P(t/pos) * P(pos)
    p_pos_times_sum = p_pos * sum_pos
    p_neg_times_sum = p_neg * sum_neg
    # / P(t/pos) * P(pos) + P(t/neg) * P(neg)
    noemer = (p_pos * sum_pos + p_neg * sum_neg)
    p_pos_final = p_pos_times_sum / noemer
    p_neg_final = p_neg_times_sum / noemer

    return p_pos_final, p_neg_final


def main_create_csv():
    df = pd.read_csv(filepath_or_buffer='film_recencies.csv', sep=";", names=["grade", "review", "name"])
    # df = df.head()
    df = clean_df(df)
    df_condi_p = create_df_condi_p(df)

    pos_count = df_condi_p['pos_temp'].sum()
    neg_count = df_condi_p['neg_temp'].sum()
    df_condi_p = calc_condi_p(df_condi_p, pos_count, neg_count)

    df_condi_p.to_csv("conditional_p.csv")


def main_sentiment_analysis():
    t = "lopen haasten rennen kaarten huizen " \
        "Doorgeslagen feelgoodfilm met genoeg potentie en thema's saboteert zichzelf met een slechte vertelwijze."
    df = pd.read_csv(filepath_or_buffer='conditional_p.csv')
    pos_count = df['pos_temp'].sum()
    neg_count = df['neg_temp'].sum()
    test = applying_bayes_theorem(t, df, pos_count, neg_count)
    if test[0] > test[1]:
        print("pos")
    else:
        print("neg")


if __name__ == '__main__':
    # main_create_csv()
    main_sentiment_analysis()

